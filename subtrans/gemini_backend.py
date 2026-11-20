# -*- coding: utf-8 -*-
"""
subtrans/gemini_backend.py
Stage 2: Primary translation backend using Google Gemini API with structured JSON output.
Adheres strictly to ADDENDUM 6 specifications:
- Priority 1: No mistranslation, flag uncertain: true when ambiguous
- Priority 2: Natural spoken Vietnamese
- Priority 3: Character names & pronouns consistency (Romaji, deliberate nicknames)
- Priority 4: Single line per cue, soft cap 80 chars, continuation split formatting
"""

import os
import re
import json
import hashlib
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from pydantic import BaseModel, field_validator
from google import genai
from google.genai import types

from subtrans.models import Cue, Glossary
from subtrans.translator import BaseTranslator
from subtrans.validation import validate_translations
from subtrans.cache import execute_and_log_api_request
from subtrans.ja_morph_tagger import (
    extract_ja_linguistic_hints,
    strip_ja_furigana,
    has_cjk_characters,
)

logger = logging.getLogger("subtrans.gemini")

SYSTEM_INSTRUCTION = (
    "You are an expert anime subtitle translator, translating dialogue into natural, readable spoken Vietnamese subtitles.\n"
    "The subtitle timing and line structure are anchored to the English dialogue from the source video, while the "
    "Japanese dialogue provides the authentic source nuance, character pronouns, honorifics, and tone.\n"
    "Strict Priorities:\n"
    "1. No mistranslation. When unsure, choose the simpler, safer rendering and categorize \"uncertain\" ('pronoun', 'referent', 'wordplay', 'archaic', 'term', 'other') instead of inventing meaning.\n"
    "2. Natural, readable spoken Vietnamese that sounds natural when read aloud (never word-for-word calques).\n"
    "3. Consistent names and pronouns within the episode.\n"
    "4. Format: Exactly ONE text line per cue (no line breaks). Soft cap: 80 characters.\n"
    "5. Punctuation: use '...' (three ASCII dots, never unicode ellipsis); use '—' at end of interrupted speech; '?!' allowed; max two consecutive ! or ?.\n"
    "6. Sentences split across cues: translate at sentence level, then divide across the SAME cues at natural phrase boundaries. "
    "The continuation cue MUST start in lowercase; the preceding fragment MUST end with ',' or '...'. Never move content beyond that split.\n"
    "7. Names, clans, and nicknames: ALWAYS keep Romaji spelling matching the English/Cast reference track (e.g. 'Rudeus', 'Emilia', 'Subaru', 'Fran', 'Reirin'). "
    "NEVER translate proper character names into Sino-Vietnamese (Hán-Việt) or literal Vietnamese words unless explicitly specified in the series glossary. "
    "Keep places, guilds, and organization names consistent with the series glossary.\n"
    "8. Addressing by proper name (Vocative): When Japanese dialogue addresses someone by their proper name (e.g. フラン -> Fran, スバル -> Subaru, ネルさん -> chị Nell / Nell, 玲琳 -> Reirin), ALWAYS KEEP the Romaji proper name directly in the Vietnamese dialogue line (e.g. 'Fran, em có sao không?', 'Subaru, cẩn thận!', 'Cảm ơn chị Nell', 'Reirin, chuyện gì vậy?'). NEVER drop or replace the proper name with generic standalone pronouns ('anh/chị/em/cậu/cô/mày') to avoid wrong pronoun assumptions and preserve audio-subtitle sync.\n"
    "9. Register & pacing: Match each character's personality, speaking_style, and pacing from <episode_cast>. Keep stoic/terse characters concise and natural; preserve noble/archaic registers when indicated. "
    "Differentiate internal monologues ('mode': 'inner', e.g. tôi/mình/ta) from telepathic ('comm') and spoken dialogue based on `mode_hint` and the relationship table.\n"
    "10. Context-First Reasoning (Chain-of-Thought): Prior to outputting 'vi', you MUST analyze and output:\n"
    "    - 'speaker': character name speaking (from `speaker_cc`, <episode_cast>, or 'unknown').\n"
    "    - 'addressee': character recipient being spoken to (or null if talking to self / general audience).\n"
    "    - 'mode': dialogue mode ('spoken', 'inner', 'narration', 'spell', 'flashback', 'comm', 'lyrics', 'onscreen').\n"
    "    - 'p1': exact Vietnamese 1st-person pronoun used in this line (e.g. 'anh', 'em', 'ta', 'tôi', 'tớ', 'mình', 'thần thiếp', or null if omitted).\n"
    "    - 'p2': exact Vietnamese 2nd-person pronoun/title used in this line (e.g. 'em', 'anh', 'ngươi', 'cậu', 'sư phụ', 'bệ hạ', or null if omitted).\n"
    "    - 'en_conflict': true if the English reference diverges significantly from or contradicts the authentic Japanese nuance, else false.\n"
    "    - 'vi': the Vietnamese dialogue translation adhering strictly to the above rules.\n"
    "    - 'uncertain': 'none' if confident; or 'pronoun', 'referent', 'wordplay', 'archaic', 'term', 'other' if unsure.\n"
    "11. Japanese Linguistic Hints & Register (Zero-Pronouns): When cues contain `ja_hints` (e.g. `1P:ore`, `2P:kisama/temee/anta/omae`, `SFP` tone markers) or `mode_hint` (`inner`/`narration`/`flashback`), use them as decisive semantic anchors for Vietnamese pronouns and sentence register, avoiding mechanical translations of English 'you/I' into generic 'bạn/tôi'.\n"
    "Output must be a valid JSON array matching the provided schema."
)

UncertainCategory = Literal[
    "none", "pronoun", "referent", "wordplay", "archaic", "term", "other"
]


class SubtitleOutputItem(BaseModel):
    id: int
    speaker: Optional[str] = "unknown"
    addressee: Optional[str] = None
    mode: Literal[
        "spoken", "inner", "narration", "spell",
        "flashback", "comm", "lyrics", "onscreen"
    ] = "spoken"
    p1: Optional[str] = None
    p2: Optional[str] = None
    en_conflict: bool = False
    vi: str
    uncertain: UncertainCategory = "none"

    @field_validator("uncertain", mode="before")
    @classmethod
    def normalize_uncertain(cls, v: Any) -> str:
        if v is True:
            return "other"
        if v is False or v is None:
            return "none"
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("none", "pronoun", "referent", "wordplay", "archaic", "term", "other"):
                return s
        return "other"


def _apply_item_metadata_to_cue(cue: Cue, item: Dict[str, Any]) -> None:
    """Applies inferred metadata (speaker, addressee, mode, p1, p2, en_conflict, uncertain) from LLM output to Cue."""
    cue.speaker = item.get("speaker") or cue.speaker or "unknown"
    cue.addressee = item.get("addressee") or cue.addressee
    cue.mode = item.get("mode") or cue.mode or cue.mode_hint or "spoken"
    if item.get("p1"):
        cue.p1 = str(item["p1"]).strip()
    if item.get("p2"):
        cue.p2 = str(item["p2"]).strip()
    cue.en_conflict = bool(item.get("en_conflict", False))
    unc = item.get("uncertain", "none")
    if unc is True:
        cue.uncertain = True
        cue.uncertain_reason = "other"
    elif isinstance(unc, str) and unc not in ("none", ""):
        cue.uncertain = True
        cue.uncertain_reason = unc
    else:
        cue.uncertain = False
        cue.uncertain_reason = "none"


def get_anime_safety_settings() -> List[types.SafetySetting]:
    """
    Cấu hình Safety Settings chuẩn cho dịch phụ đề anime (hành động, dark-fantasy, lịch sử):
    Ngăn bộ lọc mặc định chặn nhầm lời thoại chiến đấu ('殺すぞ', '死ね', '皆殺し') gây response.text = None.
    """
    return [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]


def extract_genai_response_text(response: Any) -> str:
    """
    Trích xuất văn bản JSON từ phản hồi Google GenAI an toàn.
    Nếu response.text là None hoặc rỗng (ví dụ bị safety block / empty candidate),
    ném ValueError rõ ràng để cơ chế Bisect hoặc Fail-Open xử lý êm thay vì crash AttributeError.
    """
    raw = getattr(response, "text", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    finish_reason = "unknown"
    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "unknown"))
    except Exception:
        pass
    raise ValueError(f"Empty or blocked Gemini response (finish_reason={finish_reason})")


STATIC_ANIME_TRANSLATION_HANDBOOK = (
    "<vietnamese_anime_translation_handbook>\n"
    "## SECTION A: ZERO-PRONOUN RESOLUTION & VIETNAMESE REGISTER MATRIX\n"
    "Japanese anime dialogue omits explicit 1st-person and 2nd-person pronouns in ~65-75% of sentences (zero-pronouns). "
    "English reference subtitles flatten almost every pair into 'I' and 'you'. You MUST reconstruct the exact Vietnamese "
    "dyadic relationship pair `(p1, p2)` before writing `vi`, governed by the following hierarchy of truth:\n"
    "  Priority 1: `<episode_cast>` explicit `speaks_to` pair (`self` / `other`) and `<dynamic_cast>` verified session pairs.\n"
    "  Priority 2: Explicit `speaker_cc` (from Japanese Closed Captions / ASS Actor) + `mode_hint` (`inner`, `narration`, `flashback`).\n"
    "  Priority 3: `ja_hints` morphological markers (`1P:*`, `2P:*`, `HON:*`, `KIN:*`, `REGISTER:*`, `SFP:*`, `DIALECT:*`).\n"
    "  Priority 4: `<address_rules>` archetype table and immediate scene context (`context_before` / `context_after`).\n\n"
    "### Canonical Genre & Social Hierarchy Pronoun Matrix\n"
    "1. **Master & Disciple / Living Weapon & Wielder (Sư đồ / Chủ khí)**:\n"
    "   - Young disciple -> Master (`師匠`): `p1='em'` (or `'con'` if paternal age gap), `p2='sư phụ'`. Never `'tôi/thầy'` in fantasy combat.\n"
    "   - Master -> Young female disciple: `p1='ta'` (or `'thầy'`), `p2='em'` / `'con'` / proper Romaji name (`Fran`).\n"
    "2. **Imperial Palace / Inner Court / Noble Houses (Cung đình / Hậu cung / Quý tộc cổ trang)**:\n"
    "   - Consort / Concubine -> Emperor (`陛下`): `p1='thần thiếp'` (formal) or `'thiếp'` (intimate), `p2='bệ hạ'`.\n"
    "   - Emperor -> Consort: `p1='trẫm'` (court edict) or `'ta'` (private), `p2='nàng'` / `'ái phi'` / title.\n"
    "   - Maid / Retainer -> Noble Lady (`お嬢様` / `〜様`): `p1='nô tỳ'` / `'em'` / `'thuộc hạ'`, `p2='tiểu thư'` / `'nương nương'` / `'đại nhân'`.\n"
    "   - Noble Lady -> Rival Noble Lady (polite surface, cold subtext `わたくし` / `あなた様`): `p1='tôi'` / `'ta'`, `p2='cô'` / `'tỷ'` / `'muội'`.\n"
    "   - Crown Prince / Young Lord -> Fiancée (`そなた`): `p1='ta'`, `p2='nàng'` (affectionate/formal) or `'cô'` (distant).\n"
    "3. **Isekai / High Fantasy Guilds, Knights & Royalty (Dị giới / Hiệp sĩ / Mạo hiểm giả)**:\n"
    "   - Knight / Subordinate -> King, Demon Lord, Commander (`陛下`, `閣下`, `我が君`): `p1='thuộc hạ'` / `'thần'`, `p2='bệ hạ'` / `'ngài'` / `'chúa công'`.\n"
    "   - Arrogant King / Demon Lord / Ancient Dragon (`余`, `我`, `わらわ` -> `貴様ら`, `雑種`): `p1='ta'` / `'bổn vương'`, `p2='ngươi'` / `'lũ ngươi'` / `'lũ tạp chủng'`.\n"
    "   - Senior Adventurer -> Teenager (`お嬢ちゃん`, `坊主`): `p1='ta'` / `'chị'` / `'bác'`, `p2='nhóc'` / `'cô bé'` / `'cậu nhóc'`.\n"
    "   - Party Peers (same age band): Male-Male casual (`俺` -> `お前`) = `'tôi/ông'` or `'tớ/cậu'` (if gentle) or `'tao/mày'` (only if rough delinquents/brawling); "
    "Male-Female peers = `'tôi/cô'`, `'tớ/cậu'`, or `'anh/em'` ONLY once romantic or clear older-brother seniority is established.\n"
    "4. **Samurai / Wuxia / Historical Edo (Kiếm hiệp / Võ sĩ cổ trang)**:\n"
    "   - Wandering Samurai (`拙者`, `某` -> `〜殿`, `お主`): `p1='tại hạ'`, `p2='các hạ'` / `'cô'` / `'tiểu thư'` / `'ngươi'`.\n"
    "   - Elder Grandmaster (`わし` -> `お主`, `小僧`): `p1='lão'` / `'ta'`, `p2='ngươi'` / `'tiểu tử'`.\n"
    "5. **Modern School / Slice of Life / Detective (Học đường / Đời thường / Trinh thám)**:\n"
    "   - Classmates (`僕/私` -> `君/〜さん`): `'tớ/cậu'` or `'mình/cậu'` (friendly), `'tôi/cậu'` or `'tôi/cô'` (reserved/tsundere).\n"
    "   - Junior -> Senior (`先輩`): `p1='em'`, `p2='anh'` / `'chị'` / `'tiền bối'` (keep `'anh/chị + Romaji'` when name is called).\n"
    "   - Childhood Friends in Kansai-ben (`DIALECT:kansai_ben`: `なんでやねん`, `せやな`, `あかん`, `ほんま`): keep brisk, punchy spoken rhythm "
    "(`'Sao lại thế hả?!'`, `'Thiệt tình!'`, `'Đúng rồi đó'`, `'Không ổn đâu'`), using natural close-friend pronouns (`'tôi/bà'`, `'tớ/cậu'`, `'ông/tui'`).\n"
    "6. **Internal Monologue, Narration & Telepathy (`mode` & `mode_hint` Disambiguation)**:\n"
    "   - When `mode_hint='inner'` (triggered by `＜...＞` full-width brackets or italic thought styles) or character speaks to self (`addressee=null`):\n"
    "     * Humble/modern protagonist (`俺`, `僕`, `私` thinking to self): `p1='mình'` or `'tôi'`, `p2=null`.\n"
    "     * Noble/ancient/villain monologue (`我`, `余`, `わらわ` thinking to self): `p1='ta'`, `p2=null`.\n"
    "     * NEVER invent a `p2` ('anh/em/cậu') in pure internal monologue unless mentally addressing a specific named person.\n"
    "   - When `mode='comm'` (telepathy between bonded partners, e.g., Shishou <-> Fran): use their exact spoken dyadic pair (`ta/em`, `em/sư phụ`), NOT solitary `'mình'`!\n"
    "   - When `mode='narration'` (`mode_hint='narration'`): objective epic/literary Vietnamese prose, third-person reference.\n\n"
    "## SECTION B: ANTI-CALQUE & NATURAL SPOKEN VIETNAMESE CRAFT RULES\n"
    "1. **Eliminate Translationese & Stiff Sino-Vietnamese Calques**:\n"
    "   - NEVER write `'thế giới quan'` for `世界観` in normal speech -> write `'thế giới'` or `'quy luật của thế giới này'`.\n"
    "   - NEVER write `'sự tồn tại của anh ấy'` for `彼の存在` -> write `'con người anh ấy'` or `'sự hiện diện của hắn'`.\n"
    "   - NEVER write `'không có khả năng là...'` for `はずがない` -> write `'làm gì có chuyện...'`, `'không đời nào...'`, `'sao có thể...'`.\n"
    "   - NEVER write `'tôi nghĩ rằng là một điều tốt'` -> write `'tôi thấy vậy cũng tốt mà'`, `'như vậy càng hay'`.\n"
    "   - NEVER transliterate non-Chinese fantasy character names into Hán-Việt (`フラン` -> `'Fran'`, NEVER `'Pháp Lạp'`; `ウルシ` -> `'Urushi'`, NEVER `'Ô Lỗ Thi'`; "
    "`玲琳` -> `'Reirin'`, NEVER `'Linh Lâm'`; `慧月` -> `'Keigetsu'`, NEVER `'Huệ Nguyệt'`; `スバル` -> `'Subaru'`; `エミリア` -> `'Emilia'`).\n"
    "2. **Japanese Sentence-Final Particles (SFP) -> Authentic Vietnamese Modal Particles**:\n"
    "   - `〜よ / 〜だよ / 〜わよ` (assertion/reminder) -> `'... đấy'`, `'... đó'`, `'... mà'`, `'... cơ mà'`.\n"
    "   - `〜ね / 〜だね / 〜わね` (seeking agreement/gentle observation) -> `'... nhỉ'`, `'... ha'`, `'... đúng không'`, `'... nhé'`.\n"
    "   - `〜ぞ / 〜ぜ` (masculine resolve/warning) -> `'... đấy!'`, `'... đây!'`, `'... thôi nào!'`, `'... cho xem!'`.\n"
    "   - `〜かな / 〜かしら` (wondering aloud) -> `'Không biết... nhỉ?'`, `'Liệu có... không ta?'`, `'Chẳng rõ... nữa'`.\n"
    "   - `〜じゃないか / 〜じゃん` (rhetorical realization) -> `'Chẳng phải... sao?!'`, `'... còn gì nữa!'`.\n"
    "   - `〜てくれ / 〜てくれないか` -> `'Hãy... giúp ta'`, `'... giùm tôi nhé'`, `'Nhờ cậu...'`.\n"
    "   - `〜やがる` (hostile contempt) -> `'Dám... sao?!'`, `'Tên khốn đó lại dám...'`.\n"
    "3. **Strict Vocative Name Retention & Katakana Loanword Awareness**:\n"
    "   - Whenever the Japanese speaker calls a person's name (`フラン、危ない！`, `ネルさん、お願い`, `ジャン・ドゥービーだ`), the Vietnamese line MUST contain "
    "that exact Romaji name (`'Fran, cẩn thận!'`, `'Chị Nell, nhờ chị nhé'`, `'Ta là Jean Doubee!'`).\n"
    "   - Do NOT confuse Katakana common loanwords with short character names: `プログラム` = `'chương trình'` (NOT character `'Ram'`), "
    "`パネル` / `トンネル` = `'bảng điều khiển'` / `'đường hầm'` (NOT character `'Nell'`), `フランチャイズ` = `'nhượng quyền'` (NOT `'Fran'`).\n"
    "4. **Numbers, Units, Ranks & Temporal Precision**:\n"
    "   - Preserve EVERY number, rank, floor, percentage, currency, and time expression faithfully (`300人` -> `'300 tên'`, `Sランク` -> `'hạng S'`, "
    "`第7階層` -> `'tầng thứ 7'`, `3年前` -> `'3 năm trước'`). Never round, omit, or hallucinate numbers.\n"
    "5. **Reading Speed (CPS <= 20 chars/sec) & Split-Cue Continuity**:\n"
    "   - For brisk 1.0s-1.5s exclamations, keep Vietnamese crisp and punchy (`まさか！` -> `'Không thể nào!'`; `くそっ！` -> `'Chết tiệt!'`; `行くぞ！` -> `'Đi thôi!'`).\n"
    "   - When one sentence spans Cue N and Cue N+1: Cue N MUST end with `','` or `'...'`, and Cue N+1 MUST begin in lowercase (unless Cue N+1 begins with a "
    "capitalized Romaji proper name such as `'Subaru'` or `'Fran'`). Never duplicate words already spoken in Cue N into Cue N+1.\n\n"
    "## SECTION C: GOLDEN CHAIN-OF-THOUGHT TRANSLATION EXEMPLARS\n"
    "- Exemplar 1 (Vocative + Master/Disciple): JA=`師匠、あの魔獣の核を吸収する？` | EN=`Master, are we absorbing that magic beast's core?` "
    "-> speaker=`Fran`, addressee=`Shishou`, mode=`spoken`, p1=`em`, p2=`sư phụ`, en_conflict=`false`, vi=`Sư phụ, mình hấp thụ ma thạch của con ma thú đó nhé?`, uncertain=`none`.\n"
    "- Exemplar 2 (Inner Monologue `＜...＞` after CC Sound Tag): JA=`この薬草なら慧月様の毒を解けるかもしれない…` (`mode_hint='inner'`) | EN=`This herb might cure Lady Keigetsu's poison...` "
    "-> speaker=`Reirin`, addressee=`null`, mode=`inner`, p1=`mình`, p2=`null`, en_conflict=`false`, vi=`Với loại thảo dược này, biết đâu mình có thể giải độc cho tiểu thư Keigetsu...`, uncertain=`none`.\n"
    "- Exemplar 3 (JA-EN Nuance Conflict — EN flattens hostile `貴様` to 'you'): JA=`玲琳、貴様さえいなければ！` | EN=`Reirin, if only you weren't around!` "
    "-> speaker=`Keigetsu`, addressee=`Reirin`, mode=`spoken`, p1=`ta`, p2=`ngươi`, en_conflict=`true`, vi=`Reirin, giá mà ngươi không tồn tại trên đời này!`, uncertain=`none`.\n"
    "- Exemplar 4 (Archaic Samurai `某` + Polite Imperative): JA=`薫殿、某にお任せあれ。` | EN=`Miss Kaoru, leave this to me.` "
    "-> speaker=`Kenshin`, addressee=`Kaoru`, mode=`spoken`, p1=`tại hạ`, p2=`cô`, en_conflict=`false`, vi=`Cô Kaoru, cứ giao việc này cho tại hạ.`, uncertain=`none`.\n"
    "- Exemplar 5 (Arrogant Monarch `余` + Plural Contempt `貴様ら`): JA=`貴様ら、雑種の分際で余に逆らうか！` | EN=`You mongrels dare defy me?!` "
    "-> speaker=`Gilgamesh`, addressee=`Enemies`, mode=`spoken`, p1=`ta`, p2=`lũ ngươi`, en_conflict=`false`, vi=`Lũ tạp chủng các ngươi mà cũng dám chống lại ta sao?!`, uncertain=`none`.\n"
    "- Exemplar 6 (Split Sentence Across Two Cues): Cue 101 JA=`あの時、俺を助けてくれたのは…` -> vi=`Người đã cứu tôi vào khoảnh khắc ấy...` | "
    "Cue 102 JA=`間違いなくエミリアだったんだ。` -> vi=`chắc chắn chính là Emilia.` (lowercase continuation, proper name `Emilia` capitalized).\n"
    "- Exemplar 7 (Short Zero-Pronoun Line Anchored by Neighbor Context): Context prev=`ネルさん、受付をお願い` -> Current JA=`ありがとう` (speaker=`Fran`) "
    "-> speaker=`Fran`, addressee=`Nell`, mode=`spoken`, p1=`em`, p2=`chị`, en_conflict=`false`, vi=`Cảm ơn chị nhé.`, uncertain=`none`.\n"
    "- Exemplar 8 (Unlabeled SRT with 5.1 FC Acoustic Turn Shift `audio_turn='speaker_changed'`): Cue 204 JA=`もう遅いから休め。` (speaker=`Hodgins` -> Violet) | "
    "Cue 205 JA=`いえ、この手紙を書き終えるまでは。` (`audio_turn='speaker_changed'`, female F0 shift) "
    "-> speaker=`Violet`, addressee=`Hodgins`, mode=`spoken`, p1=`tôi`, p2=`Trung tá`, en_conflict=`false`, vi=`Không ạ, tôi phải viết xong bức thư này đã.`, uncertain=`none`.\n"
    "- Exemplar 9 (Unlabeled SRT with Continuous Voice `audio_turn='same_speaker'`): Cue 310 JA=`ゾルトラークはもう人を殺す魔法じゃない。` | "
    "Cue 311 JA=`今の時代では一般攻撃魔法と呼ばれているよ。` (`audio_turn='same_speaker'`) "
    "-> speaker=`Frieren`, addressee=`Qual`, mode=`spoken`, p1=`ta`, p2=`ngươi`, en_conflict=`false`, vi=`Ở thời đại này, người ta gọi nó là Ma pháp tấn công cơ bản đấy.`, uncertain=`none`.\n"
    "- Exemplar 10 (Name Suffix `-君` `HON:kun` vs 2nd-Person Pronoun `君` `2P:kimi`): JA=`佐藤君はまだ来ていないのか？` (`ja_hints='HON:kun'`) | EN=`Hasn't Sato-kun arrived yet?` "
    "-> 3rd-person reference to Sato, NEVER misread `-君` as 2nd-person `'cậu'` addressing the listener -> vi=`Cậu Sato vẫn chưa tới sao?`.\n"
    "- Exemplar 11 (Polite Apprentice Scolding Mentor — Frieren/Fern Archetype): JA=`フリーレン様、また変な魔導書を買ったのですか。` | EN=`Mistress Frieren, did you buy another weird grimoire?` "
    "-> speaker=`Fern`, addressee=`Frieren`, mode=`spoken`, p1=`em`, p2=`cô Frieren`, en_conflict=`false`, vi=`Cô Frieren, cô lại mua thêm sách ma đạo kỳ quặc nữa rồi phải không ạ?`, uncertain=`none`.\n"
    "- Exemplar 12 (Cursed Technique / Breathing Technique Canonical Renderings): JA=`大丈夫、僕最強だから。領域展開「無量空処」。` | EN=`Don't worry, I'm the strongest. Domain Expansion: Unlimited Void.` "
    "-> speaker=`Gojo`, addressee=`Itadori`, mode=`spoken`, p1=`thầy`, p2=`em`, en_conflict=`false`, vi=`Yên tâm đi, vì thầy là kẻ mạnh nhất mà. Lãnh địa triển khai: Vô Lượng Không Xứ.`, uncertain=`none`.\n"
    "- Exemplar 13 (Kansai-ben Lively Tsukkomi `DIALECT:kansai_ben`): JA=`なんでやねん！ほんまに無茶苦茶やで！` | EN=`What the heck?! That's seriously crazy!` "
    "-> preserve energetic colloquial cadence without awkward literalisms -> vi=`Cái quái gì vậy trời! Đúng là loạn hết cả lên mà!`, uncertain=`none`.\n\n"
    "## SECTION D: ACOUSTIC SPEAKER DIARIZATION & MULTI-MODAL DISAMBIGUATION RULES\n"
    "- When `speaker_cc` is present (from `.ass` Actor or Closed Captions `(Speaker)`), treat `speaker_cc` as authoritative ground truth.\n"
    "- When `speaker_cc` is absent on plain `.srt` files, use `audio_turn` (`speaker_changed` vs `same_speaker`) together with `ja_hints` "
    "(morphological 1P/2P pronouns, keigo verbs, gendered SFPs) and `surrounding_context` (`p1`, `p2`, `vi_ref`) to track turn-taking accurately:\n"
    "  * `audio_turn='speaker_changed'`: The center-channel voice timbre and fundamental pitch F0 changed significantly from the previous cue. "
    "Flip `speaker` and `addressee` in conversational 2-person scenes unless explicit vocatives or narration indicate a 3rd participant.\n"
    "  * `audio_turn='same_speaker'`: The exact same voice continues speaking across the cue boundary. Maintain the same `speaker`, `addressee`, `p1`, and `p2`.\n"
    "</vietnamese_anime_translation_handbook>"
)


_EPISODE_PROMPT_CACHE: Dict[Tuple[str, str, bool, bool], str] = {}
_EPISODE_PROMPT_CACHE_LOCK = threading.Lock()


def _compute_prompt_cache_hashes(glossary: Optional[Glossary]) -> Tuple[str, str]:
    """Computes (glossary_hash, cast_hash) for caching static episode system prompts."""
    if not glossary:
        return "", ""
    glossary_payload = {
        "style_rules": list(glossary.style_rules or ()),
        "address_rules": glossary.address_rules.model_dump() if glossary.address_rules else None,
        "terms": [t.model_dump() for t in (glossary.terms or ())],
    }
    cast_payload = [c.model_dump() for c in (glossary.characters or ())]
    g_hash = hashlib.sha256(
        json.dumps(glossary_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    c_hash = hashlib.sha256(
        json.dumps(cast_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return g_hash, c_hash


def build_episode_system_prompt(
    glossary: Optional[Glossary] = None,
    gloss_skill_names: bool = False,
    en_only_source: bool = False,
) -> str:
    """
    Xây dựng System Prompt tĩnh ổn định theo từng tập phim (> 4,096 tokens tĩnh) để kích hoạt
    tối đa Gemini Implicit Context Caching xuyên suốt toàn bộ các batch trong tập:
    1. SYSTEM_INSTRUCTION chuẩn gốc + STATIC_ANIME_TRANSLATION_HANDBOOK chuyên sâu.
    2. Quy tắc phong cách riêng của bộ anime (glossary.style_rules).
    3. Quy tắc xưng hô tổng quát (address_rules: principle, first_person, second_person, relations).
    4. Bảng quan hệ nhân vật cố định của tập (<episode_cast>) & thuật ngữ cố định.
    """
    g_hash, c_hash = _compute_prompt_cache_hashes(glossary)
    cache_key = (g_hash, c_hash, bool(gloss_skill_names), bool(en_only_source))
    with _EPISODE_PROMPT_CACHE_LOCK:
        cached = _EPISODE_PROMPT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    blocks: List[str] = [SYSTEM_INSTRUCTION, STATIC_ANIME_TRANSLATION_HANDBOOK]

    if gloss_skill_names:
        blocks.append(
            "SKILL NAMES RULE: When a named attack/spell occurs and the EN reference provides the name, "
            "write in Vietnamese followed by the English name in parentheses, e.g. 'Độ Không Tuyệt Đối! (Absolute Zero)'."
        )

    if en_only_source:
        blocks.append(
            "ENGLISH-ONLY SOURCE FALLBACK MODE: The input 'ja' field in this episode contains official English studio "
            "subtitles (no Japanese text track was available in the container). Infer authentic anime relationship "
            "registers, seniority, and Vietnamese pronouns (anh/em/ta/ngươi/tớ/cậu/sư phụ/tiểu thư/ngài) from "
            "the character cast table and scene context—NEVER translate English 'I/you' mechanically into flat 'tôi/bạn'."
        )

    if glossary and glossary.style_rules:
        blocks.append("Additional Series Style Rules:\n" + "\n".join(glossary.style_rules))

    if glossary and glossary.address_rules:
        ar = glossary.address_rules
        ar_lines: List[str] = []
        if ar.principle:
            ar_lines.append(f"Principle: {ar.principle}")
        if ar.first_person:
            p1_strs = [
                f"- {p.ja} -> {p.vi}"
                + (f" (alt: {', '.join(p.alt)})" if p.alt else "")
                + (f" [{p.note}]" if p.note else "")
                for p in ar.first_person
            ]
            ar_lines.append("First-Person Pronouns (1P):\n" + "\n".join(p1_strs))
        if ar.second_person:
            p2_strs = [
                f"- {p.ja} -> {p.vi}"
                + (f" (alt: {', '.join(p.alt)})" if p.alt else "")
                + (f" [{p.note}]" if p.note else "")
                for p in ar.second_person
            ]
            ar_lines.append("Second-Person Pronouns (2P):\n" + "\n".join(p2_strs))
        if ar.honorifics:
            hon_strs = [
                f"- {h.ja} -> {h.vi}"
                + (f" (alt: {', '.join(h.alt)})" if h.alt else "")
                + (f" [{h.note}]" if h.note else "")
                for h in ar.honorifics
            ]
            ar_lines.append("Honorifics:\n" + "\n".join(hon_strs))
        if ar.relations:
            rel_strs = [f"- {r.case}: self='{r.self}', other='{r.other}'" + (f" ({r.note})" if r.note else "") for r in ar.relations]
            ar_lines.append("Relationship Archetypes:\n" + "\n".join(rel_strs))
        if ar_lines:
            blocks.append("<address_rules>\n" + "\n".join(ar_lines) + "\n</address_rules>")

    if glossary and glossary.characters:
        cast_lines: List[str] = []
        for c in glossary.characters:
            meta = [str(x) for x in (c.gender, c.speaking_style, c.note) if x]
            if getattr(c, "aliases", None):
                meta.append("aliases: " + ", ".join(c.aliases))
            meta_str = f" [{'; '.join(meta)}]" if meta else ""
            speaks = (
                " | speaks_to: " + "; ".join(f"{s.self}/{s.other} -> {s.listener}" for s in c.speaks_to)
                if c.speaks_to
                else ""
            )
            cast_lines.append(f"- {c.name_ja} -> {c.name_vi}{meta_str}{speaks}")
        if cast_lines:
            blocks.append("<episode_cast>\n" + "\n".join(cast_lines) + "\n</episode_cast>")

    if glossary and glossary.terms:
        term_lines: List[str] = []
        for t in glossary.terms:
            en_part = f" ({t.en})" if t.en else ""
            alt_part = f" [alt: {', '.join(t.alt)}]" if t.alt else ""
            note_part = f" — {t.note}" if t.note else ""
            term_lines.append(f"- {t.ja}{en_part} -> {t.vi}{alt_part}{note_part}")
        if term_lines:
            blocks.append("<series_glossary_terms>\n" + "\n".join(term_lines) + "\n</series_glossary_terms>")

    blocks.append("Output must be a valid JSON array matching the provided schema.")
    prompt_out = "\n\n".join(blocks)
    with _EPISODE_PROMPT_CACHE_LOCK:
        if len(_EPISODE_PROMPT_CACHE) >= 64:
            oldest_key = next(iter(_EPISODE_PROMPT_CACHE))
            _EPISODE_PROMPT_CACHE.pop(oldest_key, None)
        _EPISODE_PROMPT_CACHE[cache_key] = prompt_out
    return prompt_out


def filter_relevant_glossary(cues: List[Cue], glossary: Optional[Glossary]) -> Dict[str, Any]:
    """Filters glossary to only include terms, characters, pronouns, and honorifics present in the current batch."""
    if not glossary:
        return {}

    raw_ja_list = [c.ja for c in cues if c.ja]
    batch_ja_text = " ".join(raw_ja_list)
    batch_ja_clean = strip_ja_furigana(batch_ja_text)

    # Khi chạy chế độ fallback EN-only (c.en trống và c.ja không có CJK), dùng cả c.ja để khớp thuật ngữ tiếng Anh!
    en_fragments: List[str] = []
    for c in cues:
        if c.en:
            en_fragments.append(c.en)
        elif c.ja and not has_cjk_characters(c.ja):
            en_fragments.append(c.ja)
    batch_en_text = " ".join(en_fragments).lower()

    def _ja_contains(needle: str) -> bool:
        if not needle:
            return False
        norm_needle = strip_ja_furigana(needle)
        return (
            needle in batch_ja_text
            or needle in batch_ja_clean
            or (norm_needle != "" and norm_needle in batch_ja_clean)
        )

    relevant_terms = []
    for term in glossary.terms:
        matches = False
        en_word_match = False
        if term.en and batch_en_text and re.search(r"\b" + re.escape(term.en) + r"\b", batch_en_text, re.IGNORECASE):
            en_word_match = True

        if len(term.ja) == 1:
            # Single kanji terms (like 気 for ki/qi) need English context or alias match
            if en_word_match:
                matches = True
            elif term.aliases and batch_en_text and any(
                re.search(r"\b" + re.escape(a) + r"\b", batch_en_text, re.IGNORECASE) for a in term.aliases
            ):
                matches = True
            elif not batch_en_text and _ja_contains(term.ja):
                matches = True
        else:
            if term.ja and _ja_contains(term.ja):
                matches = True
            elif en_word_match:
                matches = True
            elif term.aliases:
                for alias in term.aliases:
                    if _ja_contains(alias) or (
                        batch_en_text and re.search(r"\b" + re.escape(alias) + r"\b", batch_en_text, re.IGNORECASE)
                    ):
                        matches = True
                        break

        if matches:
            item: Dict[str, Any] = {"ja": term.ja, "vi": term.vi}
            if term.en:
                item["en"] = term.en
            if term.alt:
                item["alt"] = term.alt
            if term.note:
                item["note"] = term.note
            relevant_terms.append(item)

    relevant_characters = []
    # Always include all episode cast characters to preserve pronoun continuity across batches,
    # as Japanese anime dialogues frequently omit subject names (zero-pronouns).
    for char in glossary.characters:
        char_entry: Dict[str, Any] = {
            "name_ja": char.name_ja,
            "name_vi": char.name_vi,
            "speaks_to": [s.model_dump(exclude_none=True) for s in char.speaks_to],
        }
        if getattr(char, "gender", None):
            char_entry["gender"] = char.gender
        if getattr(char, "speaking_style", None):
            char_entry["register"] = char.speaking_style
        if getattr(char, "note", None):
            char_entry["note"] = char.note
        relevant_characters.append(char_entry)

    # Đặt khóa TĨNH (characters -> address_guidance) lên trước khóa ĐỘNG theo batch (terms)
    # nhằm giữ tiền tố JSON bất biến dài nhất giữa các batch liên tiếp!
    res: Dict[str, Any] = {}
    if relevant_characters:
        res["characters"] = relevant_characters

    if glossary.address_rules:
        addr: Dict[str, Any] = {}
        if glossary.address_rules.principle:
            addr["principle"] = glossary.address_rules.principle
        if glossary.address_rules.relations:
            addr["relation_archetypes"] = [
                {"case": r.case, "self": r.self, "other": r.other}
                for r in glossary.address_rules.relations
            ]

        matching_1p = []
        for p in glossary.address_rules.first_person:
            tokens = [tok.strip() for tok in p.ja.split("/") if tok.strip()]
            if any(_ja_contains(tok) for tok in tokens):
                matching_1p.append({"ja": p.ja, "vi": p.vi, "alt": p.alt, "note": p.note})
        if matching_1p:
            addr["relevant_first_person"] = matching_1p

        matching_2p = []
        for p in glossary.address_rules.second_person:
            tokens = [tok.strip() for tok in p.ja.split("/") if tok.strip()]
            if any(_ja_contains(tok) for tok in tokens):
                matching_2p.append({"ja": p.ja, "vi": p.vi, "alt": p.alt, "note": p.note})
        if matching_2p:
            addr["relevant_second_person"] = matching_2p

        matching_honorifics = []
        for h in glossary.address_rules.honorifics:
            h_keys = [k.replace("〜", "").replace("~", "").strip() for k in h.ja.split("/") if k.strip()]
            if any(
                k and (_ja_contains(k) or (batch_en_text and k.lower() in batch_en_text))
                for k in h_keys
            ):
                matching_honorifics.append({"ja": h.ja, "vi": h.vi, "alt": h.alt, "note": h.note})
        if matching_honorifics:
            addr["relevant_honorifics"] = matching_honorifics

        if addr:
            res["address_guidance"] = addr

    if relevant_terms:
        res["terms"] = relevant_terms

    return res


def get_thinking_config(
    model_name: str,
    level_override: Optional[str] = None,
) -> Optional[types.ThinkingConfig]:
    """
    Cấu hình Thinking chuẩn xác theo họ model Gemini 3.5 / 3.8 chính thức (google-genai >= 2.30):
    - flash-lite (3.5/3.1): ThinkingLevel.MINIMAL (độ trễ thấp nhất, không lẹm output token).
    - gemini-3.* (3.8/3.7/3.6/3.5-flash): ThinkingLevel.LOW mặc định (hoặc MEDIUM cho Cascade repair),
      tuyệt đối không dùng MINIMAL hay thinking_budget=0 trên 3.8-flash để tránh lỗi 400 hoặc suy luận lệch.
    - gemini-2.5*: thinking_budget=0.
    """
    m = (model_name or "").lower()
    if level_override:
        lv = level_override.strip().upper()
        if hasattr(types.ThinkingLevel, lv):
            if lv == "MINIMAL" and "flash-lite" not in m and "gemini-3." in m:
                lv = "LOW"
            return types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, lv))
    if "flash-lite" in m:
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
    if "gemini-3." in m:
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    return types.ThinkingConfig(thinking_budget=0)


class GeminiTranslator(BaseTranslator):
    """Google Gemini translation backend with structured schema output and automatic split-retry."""

    all_rotation_events: List[Dict[str, Any]] = []

    @classmethod
    def reset_session_rotation_events(cls) -> None:
        """Làm sạch nhật ký xoay khóa toàn cục khi bắt đầu tập phim mới (ngăn rò rỉ giữa các tập)."""
        cls.all_rotation_events = []

    def __init__(
        self,
        api_key: Optional[Union[str, List[str]]] = None,
        api_keys: Optional[List[str]] = None,
        model: str = "gemini-3.5-flash-lite",
        temperature: float = 0.0,
        gloss_skill_names: bool = False,
        allow_latest_alias: bool = False,
        timeout_ms: int = 45000,
        http_options: Optional[types.HttpOptions] = None,
        dry_run: bool = False,
        enable_rag: bool = True,
        round_robin: bool = False,
        cache: Optional[Any] = None,
        explicit_caching: bool = False,
        kb: Optional[Any] = None,
    ):
        import datetime
        from subtrans.config import get_gemini_api_keys
        from subtrans.cache import TranslationCache, compute_key_id
        from subtrans.throttle import RateLimiter

        # Thu thập toàn bộ danh sách API Keys
        collected_keys: List[str] = []
        if api_keys:
            collected_keys.extend(api_keys)
        if isinstance(api_key, list):
            collected_keys.extend(api_key)
        elif isinstance(api_key, str) and api_key.strip():
            for k in api_key.replace(";", ",").split(","):
                k_clean = k.strip()
                if k_clean and k_clean not in collected_keys:
                    collected_keys.append(k_clean)

        if not collected_keys:
            collected_keys = get_gemini_api_keys()

        self.api_keys: List[str] = collected_keys
        self.key_ids: List[str] = [compute_key_id(k) for k in self.api_keys]
        self.cache = cache if cache is not None else TranslationCache()
        self.round_robin = round_robin

        self.model = model
        self.temperature = temperature
        self.gloss_skill_names = gloss_skill_names
        self.allow_latest_alias = allow_latest_alias
        self.dry_run = dry_run
        self.timeout_ms = timeout_ms
        self.enable_rag = enable_rag
        self.explicit_caching = explicit_caching
        self.kb = kb
        self._explicit_cache_names: Dict[Tuple[str, str, str], str] = {}
        self._explicit_cache_disabled: set = set()
        self.stage_deadline: Optional[float] = None
        self.http_options = http_options or types.HttpOptions(timeout=int(self.timeout_ms))

        # Client-side rate limiting theo khóa (A4: Lite <= 12 RPM, Flash <= 4 RPM)
        rpm_limit = 12 if "lite" in self.model.lower() else 4
        self.rate_limiters: Dict[str, RateLimiter] = {
            kid: RateLimiter(max_rpm=rpm_limit) for kid in self.key_ids
        }
        self.rotation_events: List[Dict[str, Any]] = []

        # Chọn key đầu tiên chưa bị cạn kiệt cho model này
        self.current_key_idx: int = 0
        for idx, kid in enumerate(self.key_ids):
            if not self.cache.is_key_model_exhausted(kid, self.model):
                self.current_key_idx = idx
                break
        self._visited_rotation_indices: set = {self.current_key_idx}

        self.api_key: Optional[str] = self.api_keys[self.current_key_idx] if self.api_keys else None

        if not self.api_key and not dry_run:
            raise ValueError("GEMINI_API_KEY is required for GeminiTranslator.")

        if self.api_key and not dry_run:
            self.client = genai.Client(api_key=self.api_key, http_options=self.http_options)
        else:
            self.client = None

        # Batch & request statistics
        self.stats = {
            "total_batches": 0,
            "split_batches": 0,
            "api_requests": 0,
        }

        # Protection against mutable alias 'gemini-flash-latest' per ADDENDUM 8
        if "latest" in self.model.lower() and not self.allow_latest_alias:
            raise ValueError(
                f"Model alias '{self.model}' is prohibited by default because its underlying version can change silently. "
                f"Please specify an explicit versioned model (e.g. gemini-3.8-flash) or set allow_latest_alias=True."
            )

    def select_next_round_robin_key(self, reason: str = "") -> bool:
        """
        Luân phiên (Round-Robin) sang API Key kế tiếp trong danh sách chưa bị cạn kiệt,
        ưu tiên key đang còn nhiều slot RPM trống nhất (Least-Loaded) và giữ thứ tự vòng tròn khi ngang nhau.
        """
        if not self.api_keys or len(self.api_keys) <= 1:
            return False
        import datetime
        old_idx = self.current_key_idx
        old_kid = self.key_ids[old_idx] if self.key_ids else "none"
        n_keys = len(self.api_keys)

        best_candidate: Optional[Tuple[int, int, int, str]] = None
        for offset in range(1, n_keys + 1):
            cand_idx = (old_idx + offset) % n_keys
            cand_kid = self.key_ids[cand_idx]
            if not self.cache.is_key_model_exhausted(cand_kid, self.model):
                limiter = self.rate_limiters.get(cand_kid)
                avail = limiter.available_slots() if limiter else 1
                score_tuple = (avail, -offset, cand_idx, cand_kid)
                if best_candidate is None or (score_tuple[0], score_tuple[1]) > (best_candidate[0], best_candidate[1]):
                    best_candidate = score_tuple

        if best_candidate is not None:
            _, _, cand_idx, cand_kid = best_candidate
            self.current_key_idx = cand_idx
            self._visited_rotation_indices = {cand_idx}
            self.api_key = self.api_keys[cand_idx]
            self.client = genai.Client(api_key=self.api_key, http_options=self.http_options)
            if cand_idx != old_idx and reason:
                event = {
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "model": self.model,
                    "from_key_id": old_kid,
                    "to_key_id": cand_kid,
                    "reason": reason,
                }
                self.rotation_events.append(event)
                GeminiTranslator.all_rotation_events.append(event)
                logging.getLogger("subtrans.gemini").info(
                    f"🔄 [ROUND-ROBIN] Luân phiên khóa '{self.model}': {old_kid} ➔ {cand_kid} ({reason})"
                )
            return True
        return False

    def rotate_key(self, reason: str = "") -> bool:
        """
        Xoay vòng modulo sang API Key tiếp theo chưa thử trong chu kỳ và chưa bị cạn hạn mức cho model này
        (hỗ trợ quay vòng từ Key #3 về Key #1 khi Round-Robin đang đứng ở cuối danh sách).
        """
        if not self.api_keys or len(self.api_keys) <= 1:
            return False
        import datetime
        old_idx = self.current_key_idx
        old_kid = self.key_ids[old_idx] if self.key_ids else "none"
        n_keys = len(self.api_keys)
        visited = getattr(self, "_visited_rotation_indices", {old_idx})
        visited.add(old_idx)

        for offset in range(1, n_keys):
            next_idx = (old_idx + offset) % n_keys
            if next_idx in visited:
                continue
            next_kid = self.key_ids[next_idx]
            if not self.cache.is_key_model_exhausted(next_kid, self.model):
                self.current_key_idx = next_idx
                visited.add(next_idx)
                self._visited_rotation_indices = visited
                self.api_key = self.api_keys[self.current_key_idx]
                self.client = genai.Client(api_key=self.api_key, http_options=self.http_options)
                event = {
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "model": self.model,
                    "from_key_id": old_kid,
                    "to_key_id": next_kid,
                    "reason": reason or "429 PerDay",
                }
                self.rotation_events.append(event)
                GeminiTranslator.all_rotation_events.append(event)
                logging.getLogger("subtrans.gemini").warning(
                    f"🔄 [KEY ROTATION] Chuyển khóa cho model '{self.model}': {old_kid} ➔ {next_kid} ({reason})"
                )
                return True
        return False

    def _get_or_create_explicit_cache(self, key_id: str, system_prompt: str) -> Optional[str]:
        """
        Tạo hoặc tái sử dụng bộ đệm Explicit Context Cache (TTL 15 phút = 900s) trên máy chủ Gemini
        cho bộ ba `(key_id, model, sha256(system_prompt))` khi bật `explicit_caching=True`.
        Tự động fallback êm về `None` (dùng Implicit Caching qua `system_instruction`) nếu tier/model từ chối.
        """
        if not getattr(self, "explicit_caching", False) or self.dry_run or not self.client:
            return None
        if not system_prompt or len(system_prompt) < 12000:
            return None
        if (key_id, self.model) in self._explicit_cache_disabled:
            return None

        import hashlib
        prompt_sig = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
        cache_tuple = (key_id, self.model, prompt_sig)
        existing = self._explicit_cache_names.get(cache_tuple)
        if existing:
            return existing

        try:
            caches_api = getattr(self.client, "caches", None)
            if caches_api is None or not hasattr(caches_api, "create"):
                self._explicit_cache_disabled.add((key_id, self.model))
                return None
            created = caches_api.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    ttl="900s",
                ),
            )
            c_name = getattr(created, "name", None)
            if isinstance(c_name, str) and c_name.strip():
                self._explicit_cache_names[cache_tuple] = c_name.strip()
                return c_name.strip()
        except Exception as e:
            logger.debug(f"Explicit Context Cache unavailable on {self.model}/{key_id}, falling back to Implicit Caching: {e}")
            self._explicit_cache_disabled.add((key_id, self.model))
        return None

    def _call_api(self, prompt_text: str, system_prompt: str = SYSTEM_INSTRUCTION) -> List[Dict[str, Any]]:
        """Invokes Gemini API requesting structured JSON array with rate limiting, retries, and key rotation."""
        if self.dry_run:
            return []

        import time
        import random
        from subtrans.throttle import parse_429_details, DailyQuotaExhaustedError, QuotaExhaustedError
        from subtrans.cache import get_next_pt_reset

        config_kwargs = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": list[SubtitleOutputItem],
            "temperature": self.temperature,
            "max_output_tokens": 16384,
            "http_options": self.http_options,
            "safety_settings": get_anime_safety_settings(),
        }
        thinking_cfg = get_thinking_config(self.model)
        if thinking_cfg:
            config_kwargs["thinking_config"] = thinking_cfg

        # Pre-check: nếu key hiện tại đã bị cạn kiệt cho model này thì thử xoay trước
        curr_kid = self.key_ids[self.current_key_idx] if self.key_ids else "none"
        if self.cache.is_key_model_exhausted(curr_kid, self.model):
            if not self.rotate_key(reason="Pre-check: Khóa hiện tại đã hết hạn mức ngày"):
                next_pt, next_vn = get_next_pt_reset()
                self.cache.mark_model_exhausted(
                    model=self.model,
                    exhausted_until_pt=next_pt,
                    quota_id="AllKeysExhausted",
                    reason="All configured API keys exhausted daily quota",
                )
                raise DailyQuotaExhaustedError(
                    model=self.model,
                    quota_id="AllKeysExhausted",
                    quota_value="",
                    reset_pt_str=next_pt.strftime("%Y-%m-%d %H:%M PT"),
                    reset_vn_str=next_vn.strftime("%H:%M %d/%m VN"),
                )

        max_transient_retries = 3
        transient_attempt = 0
        max_rpm_retries = max(4, len(self.api_keys) * 2)
        rpm_attempt = 0
        stage_deadline = getattr(self, "stage_deadline", None)

        while True:
            if stage_deadline is not None and time.monotonic() >= stage_deadline:
                raise QuotaExhaustedError(f"Stage deadline reached before API call on '{self.model}'.")

            curr_kid = self.key_ids[self.current_key_idx] if self.key_ids else "none"
            if curr_kid in self.rate_limiters:
                self.rate_limiters[curr_kid].wait_if_needed()

            explicit_cache_name = self._get_or_create_explicit_cache(curr_kid, system_prompt)
            req_config_kwargs = dict(config_kwargs)
            if explicit_cache_name:
                req_config_kwargs.pop("system_instruction", None)
                req_config_kwargs["cached_content"] = explicit_cache_name

            def _send():
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(**req_config_kwargs)
                )

            try:
                response = execute_and_log_api_request(
                    _send,
                    model=self.model,
                    backend="gemini",
                    cues_count=int(getattr(self, "_current_cues_count", 0) or 0),
                    request_type="translation",
                    cache=self.cache,
                    key_id=curr_kid,
                )
                raw_text = extract_genai_response_text(response)
                data = json.loads(raw_text)
                # Sau khi batch hoàn tất thành công, xoay vòng Round-Robin sang key tiếp theo (nếu bật round_robin)
                if getattr(self, "round_robin", False) and len(self.api_keys) > 1:
                    self.select_next_round_robin_key(reason="Round-Robin sau batch thành công")
                return data
            except Exception as e:
                err_str = str(e).lower()
                if explicit_cache_name and ("cachedcontent" in err_str or "cached_content" in err_str or "not found" in err_str):
                    self._explicit_cache_disabled.add((curr_kid, self.model))
                    for k_del in [k for k, v in self._explicit_cache_names.items() if v == explicit_cache_name]:
                        self._explicit_cache_names.pop(k_del, None)
                    continue
                is_429 = "429" in err_str or "resource_exhausted" in err_str or "too many requests" in err_str

                if is_429:
                    quota_id, quota_value, retry_delay = parse_429_details(e)
                    qid_lower = (quota_id or "").lower()
                    is_per_day = any(p in qid_lower for p in ["perday", "requestsperday", "daily", "day"]) or "perday" in err_str or "requestsperday" in err_str

                    if is_per_day:
                        # (a) 429 PerDay -> đánh dấu (key_id, model) cạn đến 00:00 PT, chuyển ngay SANG KHÓA KẾ của CÙNG model
                        self.cache.mark_key_model_exhausted(
                            key_id=curr_kid,
                            model=self.model,
                            quota_id=quota_id or "PerDay",
                            quota_value=quota_value or "",
                            reason=f"429 PerDay: {quota_id}"
                        )
                        if self.rotate_key(reason=f"429 PerDay ({quota_id})"):
                            transient_attempt = 0
                            continue
                        else:
                            # Hết toàn bộ khóa của model này -> đánh dấu model_exhaustion và ném DailyQuotaExhaustedError
                            next_pt, next_vn = get_next_pt_reset()
                            self.cache.mark_model_exhausted(
                                model=self.model,
                                exhausted_until_pt=next_pt,
                                quota_id=quota_id or "PerDay",
                                quota_value=quota_value or "",
                                reason=f"All keys 429 PerDay ({quota_id})",
                            )
                            raise DailyQuotaExhaustedError(
                                model=self.model,
                                quota_id=quota_id or "PerDay",
                                quota_value=quota_value or "",
                                reset_pt_str=next_pt.strftime("%Y-%m-%d %H:%M PT"),
                                reset_vn_str=next_vn.strftime("%H:%M %d/%m VN"),
                            )
                    else:
                        # (b) 429 PerMinute -> chặn trên số lần thử lại và tôn trọng stage_deadline;
                        # ghi cooldown lên khóa hiện tại TRƯỚC KHI xoay khóa.
                        if rpm_attempt >= max_rpm_retries or (stage_deadline is not None and time.monotonic() >= stage_deadline):
                            raise QuotaExhaustedError(
                                f"429 PerMinute retry limit ({max_rpm_retries}) or stage deadline exceeded on '{self.model}': {e}"
                            ) from e
                        rpm_attempt += 1

                        wait_sec = float(retry_delay) if (retry_delay and retry_delay > 0) else 10.0
                        curr_limiter = self.rate_limiters.get(curr_kid)
                        if curr_limiter is not None:
                            curr_limiter.record_cooldown(wait_sec)

                        if getattr(self, "round_robin", False) and len(self.api_keys) > 1:
                            prev_idx = self.current_key_idx
                            if self.select_next_round_robin_key(reason=f"429 PerMinute ({quota_id})") and self.current_key_idx != prev_idx:
                                next_kid = self.key_ids[self.current_key_idx]
                                limiter = self.rate_limiters.get(next_kid)
                                if not limiter or limiter.available_slots() > 0:
                                    continue

                        if stage_deadline is not None and (time.monotonic() + wait_sec) > stage_deadline:
                            raise QuotaExhaustedError(
                                f"429 PerMinute wait ({wait_sec:.1f}s) would exceed stage deadline on '{self.model}': {e}"
                            ) from e
                        logging.getLogger("subtrans.gemini").warning(
                            f"⏳ [429 PER-MINUTE] Model '{self.model}' trên key {curr_kid} cần chờ {wait_sec:.1f}s "
                            f"(Lần {rpm_attempt}/{max_rpm_retries})..."
                        )
                        time.sleep(wait_sec)
                        active_kid = self.key_ids[self.current_key_idx] if self.key_ids else "none"
                        if active_kid in self.rate_limiters:
                            self.rate_limiters[active_kid].clear_cooldown()
                        continue

                # (c) 500/502/503/504 unavailable hoặc (d) network/timeout
                is_transient = any(
                    k in err_str
                    for k in [
                        "500", "502", "503", "504",
                        "internal", "unavailable", "high demand", "bad gateway", "gateway timeout",
                        "timeout", "timed out", "connection", "remoteendclosed", "deadline",
                    ]
                )
                if is_transient:
                    if transient_attempt < max_transient_retries:
                        transient_attempt += 1
                        backoff = (2 ** transient_attempt) + random.uniform(0.5, 1.5)
                        if stage_deadline is not None and (time.monotonic() + backoff) > stage_deadline:
                            raise
                        logging.getLogger("subtrans.gemini").warning(
                            f"⚠️ [TRANSIENT {err_str[:30]}] Thử lại ({transient_attempt}/{max_transient_retries}) sau {backoff:.1f}s trên cùng khóa {curr_kid}..."
                        )
                        time.sleep(backoff)
                        continue
                    else:
                        # Thử lại tối đa 3 lần với backoff trên cùng khóa rồi sang model kế (KHÔNG xoay khóa)
                        logging.getLogger("subtrans.gemini").error(
                            f"❌ Server/Network error liên tiếp 3 lần trên {curr_kid}. Chuyển sang model kế!"
                        )
                        raise

                # Các lỗi khác: ném ra ngoài
                raise

    def translate_batch(
        self,
        cues: List[Cue],
        context: List[Cue],
        glossary: Optional[Glossary] = None,
        rag_bundle: Optional[Tuple[List[Dict[str, str]], List[Dict[str, str]], str]] = None,
        dynamic_cast_context: Optional[str] = None,
        en_only_source: Optional[bool] = None,
    ) -> List[str]:
        """Translates batch of cues (100–150 cues) with immediate bisecting (split in half) on cue/ID mismatch."""
        if not cues:
            return []

        self.stats["total_batches"] += 1

        # Chế độ Dry-Run: Không bao giờ gọi Gemini API (bảo toàn 100% quota đêm)
        if self.dry_run:
            self.stats["api_requests"] += 1
            for c in cues:
                c.uncertain = False
            return [f"[DRY-RUN VI] {c.en or c.ja}" for c in cues]

        if en_only_source is None:
            en_only_source = all(not has_cjk_characters(c.ja or "") for c in cues if (c.ja or "").strip()) and any(
                bool(re.search(r"[A-Za-z]{2,}", c.ja or "")) for c in cues
            )

        def _format_target_cue(c: Cue) -> Dict[str, Any]:
            item: Dict[str, Any] = {"id": c.id, "ja": c.ja}
            if c.en:
                item["en"] = c.en
            if c.speaker and c.speaker.strip().lower() not in ("", "unknown"):
                item["speaker_cc"] = strip_ja_furigana(c.speaker.strip())
            elif getattr(c, "audio_turn_hint", None):
                item["audio_turn"] = c.audio_turn_hint
            if getattr(c, "mode_hint", None):
                item["mode_hint"] = c.mode_hint
            if not en_only_source:
                hints = extract_ja_linguistic_hints(c.ja)
                if hints:
                    item["ja_hints"] = hints
            return item

        expected_ids = [c.id for c in cues]

        # Sắp xếp các khóa trong payload từ TĨNH NHẤT -> ĐỘNG NHẤT để tối đa hóa Gemini Implicit Prefix Caching!
        payload: Dict[str, Any] = {}

        # 1. Glossary (characters cố định đầu tiên, tiếp đến terms/address_guidance của batch)
        rel_glossary = filter_relevant_glossary(cues, glossary)
        if rel_glossary:
            payload["glossary"] = rel_glossary

        # 2. RAG few-shot examples và error cautions từ KnowledgeBase
        self.last_rag_hash = "no_rag"
        if rag_bundle is not None:
            few_shots, cautions, self.last_rag_hash = rag_bundle
            if few_shots:
                payload["approved_reference_examples"] = few_shots
            if cautions:
                payload["error_prevention_cautions"] = cautions
        elif self.enable_rag and not en_only_source:
            try:
                import hashlib
                from subtrans.rag import KnowledgeBase
                kb = getattr(self, "kb", None)
                if kb is None:
                    kb = KnowledgeBase()
                    self.kb = kb
                series_key = getattr(self, "series", None)
                few_shots = kb.retrieve_few_shots(
                    cues,
                    series=series_key,
                    max_examples=3,
                    exclude_episode=getattr(self, "exclude_episode", None),
                    fallback_builtin=True,
                )
                if few_shots:
                    payload["approved_reference_examples"] = few_shots
                cautions = kb.retrieve_cautions(cues, series=series_key, max_cautions=2)
                if cautions:
                    payload["error_prevention_cautions"] = cautions
                if few_shots or cautions:
                    rag_data = {"few_shots": few_shots, "cautions": cautions}
                    self.last_rag_hash = hashlib.sha256(json.dumps(rag_data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
            except Exception as e:
                logger.debug(f"RAG retrieval skipped: {e}")

        # 3. Dynamic Cast Overrides (tách khỏi glossary.style_rules để không làm hỏng prefix tĩnh của system_instruction!)
        if dynamic_cast_context and dynamic_cast_context.strip():
            payload["dynamic_cast_relations"] = dynamic_cast_context.strip()

        # 4. Surrounding context (+/-10 cues per ADDENDUM 6)
        if context:
            first_id = cues[0].id
            last_id = cues[-1].id
            ctx_items = []
            for c in context:
                ctx_item: Dict[str, Any] = {"id": c.id, "ja": c.ja, "context": True}
                if c.en:
                    ctx_item["en"] = c.en
                if c.speaker and c.speaker.strip().lower() not in ("", "unknown"):
                    ctx_item["speaker_cc"] = strip_ja_furigana(c.speaker.strip())
                if getattr(c, "p1", None):
                    ctx_item["p1"] = c.p1
                if getattr(c, "p2", None):
                    ctx_item["p2"] = c.p2
                if getattr(c, "vi", None) and c.vi.strip() and not c.vi.startswith("[Chưa dịch"):
                    ctx_item["vi_ref"] = c.vi
                if c.id < first_id:
                    ctx_item["role"] = "history_readonly"
                elif c.id > last_id:
                    ctx_item["role"] = "lookahead_readonly"
                ctx_items.append(ctx_item)
            payload["surrounding_context"] = ctx_items

        # 5. Cues cần dịch (thay đổi nhiều nhất giữa các batch -> đặt ở cuối payload!)
        payload["cues_to_translate"] = [_format_target_cue(c) for c in cues]

        # Giữ phần dẫn đầu prompt_lines ngắn gọn, bất biến và KHÔNG chứa dấu '{' để tương thích parser JSON
        prompt_lines = [
            "Translate the following subtitle batch into Vietnamese.",
            "Follow character relationships, pronouns, speaker_cc labels, mode_hint, ja_hints, and glossary strictly.",
            "CONTEXT RULES: Cues in 'surrounding_context' are READ-ONLY for dialogue continuity. NEVER translate or include cues marked with context=true in your output.",
            "CRITICAL NAMING & VOCATIVE RULE: Always keep proper names in Romaji matching <episode_cast> / EN reference and preserve Romaji names directly in Vietnamese lines when addressed by name.",
            "When unsure of a rendering, categorize 'uncertain' instead of inventing meaning.",
        ]
        if self.gloss_skill_names:
            prompt_lines.append(
                "SKILL NAMES: Write named attacks/spells in Vietnamese followed by the English name in parentheses."
            )
        prompt_lines.append(f"\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}")
        prompt_str = "\n".join(prompt_lines)

        # Build episode-stable system prompt for Gemini Implicit Context Caching
        system_prompt = build_episode_system_prompt(
            glossary=glossary,
            gloss_skill_names=self.gloss_skill_names,
            en_only_source=bool(en_only_source),
        )

        # Attempt 1: Call API
        try:
            self.stats["api_requests"] += 1
            self._current_cues_count = len(cues)
            items = self._call_api(prompt_str, system_prompt=system_prompt)
            for c, it in zip(cues, items):
                if isinstance(it, dict) and not (it.get("vi") or "").strip():
                    if re.match(r"^[（\(][^）\)]+[）\)]$", (c.ja or "").strip()):
                        it["vi"] = "(âm thanh)"
            is_valid, err_msg = validate_translations(expected_ids, items)
            if is_valid:
                for c, it in zip(cues, items):
                    if isinstance(it, dict):
                        if it.get("vi"):
                            c.vi = it["vi"]
                        _apply_item_metadata_to_cue(c, it)
                return [it["vi"] for it in items]
            logger.warning(f"Batch validation failed on attempt 1 ({err_msg}).")
        except Exception as e:
            from subtrans.throttle import QuotaExhaustedError
            err_str = str(e).lower()
            if isinstance(e, QuotaExhaustedError) or any(
                k in err_str
                for k in [
                    "503", "429", "timeout", "timed out", "quota", "resource_exhausted",
                    "unavailable", "500", "502", "504", "internal", "high demand",
                    "bad gateway", "gateway timeout", "deadline",
                ]
            ):
                raise
            logger.warning(f"API call failed on attempt 1 ({e}).")

        # Tách đôi batch (Bisect): Khi số lượng cue > 1 và có lỗi validation hoặc API lỗi cục bộ
        if len(cues) > 1:
            self.stats["split_batches"] += 1
            mid = len(cues) // 2
            logger.warning(
                f"🔀 Tách đôi batch ({len(cues)} cues -> {mid} + {len(cues)-mid} cues) để đảm bảo độ tin cậy và khớp ID..."
            )
            left_cues = cues[:mid]
            right_cues = cues[mid:]

            # Nửa đầu (Left): Giữ lịch sử trước đó + bổ sung lookahead từ các câu đầu của right_cues
            left_context = [c for c in (context or []) if c.id < left_cues[0].id] + right_cues[:10]
            left_half = self.translate_batch(
                left_cues,
                left_context,
                glossary,
                rag_bundle=None,
                dynamic_cast_context=dynamic_cast_context,
                en_only_source=en_only_source,
            )

            # Nửa sau (Right): Bắc cầu 10 câu cuối của left_cues (đã có bản dịch vi và metadata) làm lịch sử liền kề
            right_context = left_cues[-10:] + [c for c in (context or []) if c.id > right_cues[-1].id]
            right_half = self.translate_batch(
                right_cues,
                right_context,
                glossary,
                rag_bundle=None,
                dynamic_cast_context=dynamic_cast_context,
                en_only_source=en_only_source,
            )
            return left_half + right_half

        # Nếu chỉ có 1 cue duy nhất mà thất bại, thử lại lần cuối với prompt nghiêm ngặt
        strict_prompt = (
            "CRITICAL: Output exactly one JSON item for the given cue ID. "
            "No empty strings, no leftover Japanese characters.\n"
            + prompt_str
        )
        self.stats["api_requests"] += 1
        self._current_cues_count = len(cues)
        items = self._call_api(strict_prompt, system_prompt=system_prompt)
        for c, it in zip(cues, items):
            if isinstance(it, dict) and not (it.get("vi") or "").strip():
                if re.match(r"^[（\(][^）\)]+[）\)]$", (c.ja or "").strip()):
                    it["vi"] = "(âm thanh)"
        is_valid, err_msg = validate_translations(expected_ids, items)
        if is_valid:
            for c, it in zip(cues, items):
                if isinstance(it, dict):
                    if it.get("vi"):
                        c.vi = it["vi"]
                    _apply_item_metadata_to_cue(c, it)
            return [it["vi"] for it in items]

        raise RuntimeError(f"Failed to translate cue ID {cues[0].id} with Gemini backend after retries and split.")
