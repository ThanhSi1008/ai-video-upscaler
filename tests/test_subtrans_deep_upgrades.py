# -*- coding: utf-8 -*-
"""
tests/test_subtrans_deep_upgrades.py
Bộ kiểm thử đơn vị và tích hợp toàn diện (100% offline) cho 5 Gói Nâng Cấp Chuyên Sâu
của phân hệ dịch phụ đề AI (Subtrans Engine).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pysubs2
import pytest
from google.genai import types

from subtrans.ja_morph_tagger import (
    strip_ja_furigana,
    has_cjk_characters,
    extract_ja_linguistic_hints,
)
from subtrans.models import (
    Cue,
    Glossary,
    GlossaryTerm,
    Character,
    CharacterSpeaksTo,
    AddressRules,
    PronounRule,
    RelationRule,
)
from subtrans.gemini_backend import (
    GeminiTranslator,
    get_anime_safety_settings,
    extract_genai_response_text,
    build_episode_system_prompt,
    filter_relevant_glossary,
)
from subtrans.cast_sheet import (
    check_for_new_characters,
    generate_cast_sheet,
)
from subtrans.dynamic_cast import HysteresisCastManager
from subtrans.parser import (
    parse_and_align_subtitles,
    is_en_fallback_source,
)
from subtrans.dtw_auto_align import (
    SubCue,
    split_en_reference_for_2_to_1,
    align_ja_to_en_monotonic_dtw,
    _compute_dtw_pair_cost,
)
from subtrans.linter import (
    lint_cue,
    RULE_BANNED_TERM,
    RULE_ROMAJI_MISSING,
    RULE_GLOSSARY_TERM,
)
from subtrans.cascade import (
    score_cue_difficulty,
    find_cues_needing_repair,
    run_cascade_repair_pass,
)
from subtrans.writer import (
    clean_and_normalize_vietnamese,
    write_subtitles,
)
from subtrans.rag import KnowledgeBase
from subtrans.cache import TranslationCache
from subtrans.report import generate_qa_report
from subtrans.eval.runner import run_evaluation


# ==============================================================================
# GÓI 1: ANIME SAFETY SETTINGS, IMPLICIT CACHING & STAGE A MULTI-KEY CAST SHEET
# ==============================================================================


def test_pkg1_anime_safety_settings_and_null_response_guard():
    safety = get_anime_safety_settings()
    assert len(safety) == 4
    categories = {s.category for s in safety}
    assert types.HarmCategory.HARM_CATEGORY_HARASSMENT in categories
    assert types.HarmCategory.HARM_CATEGORY_HATE_SPEECH in categories
    assert types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT in categories
    assert types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT in categories
    assert all(s.threshold == types.HarmBlockThreshold.BLOCK_NONE for s in safety)

    # Guard against response.text == None (safety block / empty candidate)
    empty_resp = MagicMock()
    empty_resp.text = None
    empty_resp.candidates = []
    with pytest.raises(ValueError, match="Empty or blocked"):
        extract_genai_response_text(empty_resp)


def test_pkg1_implicit_caching_prompt_layout_and_full_address_rules():
    glossary = Glossary(
        style_rules=["Giữ văn phong anime tự nhiên."],
        address_rules=AddressRules(
            principle="Tiếng Nhật là chân lý.",
            first_person=[PronounRule(ja="俺", vi="ta", alt=["tôi"], note="nam mạnh mẽ")],
            second_person=[PronounRule(ja="そなた", vi="nàng", alt=["ngươi"], note="cổ trang")],
            honorifics=[PronounRule(ja="〜陛下", vi="bệ hạ")],
            relations=[RelationRule(case="chủ - tớ", self="ta", other="ngươi")],
        ),
        characters=[
            Character(
                name_ja="玲琳",
                name_vi="Reirin",
                gender="female",
                speaking_style="noble",
                speaks_to=[CharacterSpeaksTo(listener="Keigetsu", self="tôi", other="cô")],
            )
        ],
        terms=[GlossaryTerm(ja="魔剣", en="Demon Sword", vi="ma kiếm")],
    )

    # System prompt phải bất biến giữa 2 batch khác nhau để tối đa hóa Gemini Implicit Context Caching
    s1 = build_episode_system_prompt(glossary, gloss_skill_names=True, en_only_source=False)
    s2 = build_episode_system_prompt(glossary, gloss_skill_names=True, en_only_source=False)
    assert s1 == s2
    assert "<episode_cast>" in s1
    assert "Reirin" in s1
    assert "chủ - tớ" in s1

    # Kiểm tra filter_relevant_glossary nạp đủ first_person, second_person, honorifics & khớp xuyên Furigana
    cues = [
        Cue(
            id=1,
            start=0,
            end=2000,
            speaker="玲(れい)琳(りん)",
            ja="陛下、俺が魔(ま)剣(けん)をそなたに渡そう。",
            en="Your Majesty, I shall hand the Demon Sword to you.",
        )
    ]
    rel = filter_relevant_glossary(cues, glossary)
    assert any(t["vi"] == "ma kiếm" for t in rel["terms"])
    addr = rel["address_guidance"]
    assert any(p["ja"] == "俺" for p in addr["relevant_first_person"])
    assert any(p["ja"] == "そなた" for p in addr["relevant_second_person"])
    assert any(h["vi"] == "bệ hạ" for h in addr["relevant_honorifics"])

    # Kiểm tra translate_batch đặt cues_to_translate ở cuối payload và bơm speaker_cc
    translator = GeminiTranslator(api_key="dummy_key", model="gemini-3.5-flash-lite")
    captured = {}

    def fake_call_api(prompt_text, system_prompt=None):
        captured["prompt_text"] = prompt_text
        captured["system_prompt"] = system_prompt
        return [
            {
                "id": 1,
                "speaker": "Reirin",
                "addressee": "Emperor",
                "mode": "spoken",
                "en_conflict": False,
                "vi": "Bệ hạ, ta sẽ trao ma kiếm cho ngài.",
                "uncertain": "none",
            }
        ]

    translator._call_api = fake_call_api
    out = translator.translate_batch(
        cues,
        context=[],
        glossary=glossary,
        dynamic_cast_context="Dynamic Cast:\n  - Reirin -> Emperor: thần thiếp / bệ hạ",
    )
    assert out == ["Bệ hạ, ta sẽ trao ma kiếm cho ngài."]
    raw_json = captured["prompt_text"][captured["prompt_text"].find("{") :]
    payload = json.loads(raw_json)
    keys_order = list(payload.keys())
    assert keys_order[-1] == "cues_to_translate"
    assert payload["cues_to_translate"][0]["speaker_cc"] == "玲琳"
    assert "<episode_cast>" in captured["system_prompt"]


def test_pkg1_stage_a_cast_sheet_multi_key_rotation_and_new_char_signals(tmp_path):
    from subtrans.throttle import DailyQuotaExhaustedError

    cache = TranslationCache(db_path=tmp_path / "cast_cache.db")
    cues = [
        Cue(id=1, start=0, end=2000, speaker="絹(きぬ)秀(ひで)", ja="金淑妃様！", en="Pure Consort Kin!"),
        Cue(id=2, start=2000, end=4000, speaker="絹秀", ja="お待ちください", en="Please wait, Consort Kin!"),
    ]

    call_log = []

    def fake_client_factory(api_key, http_options=None):
        m_client = MagicMock()

        def gen_content(model, contents, config):
            call_log.append((api_key, contents, config))
            if api_key == "key_exhausted_1111":
                raise DailyQuotaExhaustedError(
                    model=model,
                    quota_id="PerDay",
                    quota_value="20",
                    reset_pt_str="00:00 PT",
                    reset_vn_str="15:00 VN",
                )
            resp = MagicMock()
            resp.text = json.dumps(
                {
                    "characters": [
                        {
                            "name_ja": "絹秀",
                            "name_vi": "Kinshuu",
                            "gender": "female",
                            "speaking_style": "noble",
                            "speaks_to": [{"listener": "all", "self": "nô tỳ", "other": "nương nương"}],
                        }
                    ]
                }
            )
            return resp

        m_client.models.generate_content.side_effect = gen_content
        return m_client

    out_cast = tmp_path / "series.cast.yaml"
    with patch("subtrans.cast_sheet.genai.Client", side_effect=fake_client_factory):
        cast_gloss = generate_cast_sheet(
            cues=cues,
            output_path=out_cast,
            gemini_model="gemini-3.8-flash",
            cache=cache,
            api_keys=["key_exhausted_1111", "key_healthy_2222"],
        )

    assert cast_gloss is not None
    assert len(call_log) == 2
    assert call_log[0][0] == "key_exhausted_1111"
    assert call_log[1][0] == "key_healthy_2222"
    assert "[SPEAKER_CC: 絹秀]" in call_log[1][1]
    assert out_cast.exists()

    # Kiểm tra check_for_new_characters đa tín hiệu (CC speaker, vocative >=2, mid-sentence >=3)
    existing = [Character(name_ja="ルーデウス", name_vi="Rudeus")]
    ep_cues = [
        # Single nickname occurrence -> KHÔNG báo giả
        Cue(id=1, start=0, end=1000, ja="分かった", en="Understood, Sylphie."),
        # Vocative đầu/cuối câu >= 2 lần -> Phát hiện Orsted
        Cue(id=2, start=1000, end=2000, ja="オルステッド、待て", en="Orsted, wait a second!"),
        Cue(id=3, start=2000, end=3000, ja="頼む、オルステッド", en="Please listen to me, Orsted!"),
        # Mid-sentence proper noun >= 3 lần (kể cả EN-only fallback) -> Phát hiện Nanahoshi
        Cue(id=4, start=3000, end=4000, ja="I met Nanahoshi at the guild.", en=None),
        Cue(id=5, start=4000, end=5000, ja="Did Nanahoshi say anything?", en=None),
        Cue(id=6, start=5000, end=6000, ja="Ask Nanahoshi tomorrow.", en=None),
        # Japanese CC speaker mới >= 2 lần -> Phát hiện ザノバ
        Cue(id=7, start=6000, end=7000, speaker="ザノバ", ja="師匠！", en="Master!"),
        Cue(id=8, start=7000, end=8000, speaker="ザノバ", ja="素晴らしい！", en="Wonderful!"),
    ]
    detected = check_for_new_characters(ep_cues, existing)
    assert "Orsted" in detected
    assert "Nanahoshi" in detected
    assert "ザノバ" in detected
    assert "Sylphie" not in detected


# ==============================================================================
# GÓI 2: FURIGANA TRANSPARENCY, MORPH TAGGER, CLOSED-LOOP CAST & EN-FALLBACK
# ==============================================================================


def test_pkg2_furigana_stripping_and_expanded_morph_tagger():
    assert strip_ja_furigana("金(きん)淑(しゅく)妃(ひ)様") == "金淑妃様"
    assert strip_ja_furigana("貴(き)様(さま)ら、許さん！") == "貴様ら、許さん！"
    # Không xóa nhầm chú thích âm thanh đầu câu
    assert strip_ja_furigana("（ため息）もう無理だ") == "（ため息）もう無理だ"

    hints = extract_ja_linguistic_hints("陛下、私(わたくし)にお任せください。そなたは下がれじゃ。")
    assert hints is not None
    assert "1P:watakushi" in hints
    assert "2P:sonata" in hints
    assert "HON:heika" in hints
    assert "SFP:lão_niên_cổ_trang" in hints

    # Không chạy morph tagger giả trên câu tiếng Anh khi EN-only fallback
    assert extract_ja_linguistic_hints("Your Majesty, leave this to me!") is None


def test_pkg2_closed_loop_dynamic_cast_observation():
    mgr = HysteresisCastManager(
        initial_cast_data={
            "relations": {
                "Fran -> Shishou": {
                    "1p": "em",
                    "2p": "sư phụ",
                    "reason": "baseline",
                    "confidence": 1.0,
                }
            }
        }
    )

    batch_cues = [
        # Cặp mới xuất hiện nhất quán >= 2 lần, không uncertain -> tự động ghi nhận NEW_CHARACTER
        Cue(id=1, start=0, end=1000, speaker="Amanda", addressee="Fran", mode="spoken", uncertain=False, ja="...", vi="Chị sẽ bảo vệ em, Fran!"),
        Cue(id=2, start=1000, end=2000, speaker="Amanda", addressee="Fran", mode="spoken", uncertain=False, ja="...", vi="Đi theo chị nào, em giỏi lắm."),
        # Cặp cũ bộc phát tao/mày trong lúc nóng giận -> chỉ log SITUATIONAL_SHIFT, KHÔNG ghi đè baseline
        Cue(id=3, start=2000, end=3000, speaker="Fran", addressee="Shishou", mode="spoken", uncertain=False, ja="...", vi="Sao mày dám nói thế với tao!"),
    ]
    learned = mgr.observe_batch_cues(batch_cues, min_consistent_hits=2)
    assert learned["new_characters"] == 1
    assert mgr.resolve_pronouns("Amanda", "Fran") == ("chị", "em")
    assert mgr.resolve_pronouns("Fran", "Shishou") == ("em", "sư phụ")
    assert learned["situational_shifts"] == 1
    assert len(mgr.situational_log) == 1


def test_pkg2_en_fallback_detection_and_upscale_cli_flags():
    en_cues = [
        Cue(id=1, start=0, end=1000, ja="Master, look out!", en=None),
        Cue(id=2, start=1000, end=2000, ja="Use the Demon Sword now!", en=None),
    ]
    ja_cues = [
        Cue(id=1, start=0, end=1000, ja="師匠、危ない！", en="Master, look out!"),
    ]
    assert is_en_fallback_source(en_cues) is True
    assert is_en_fallback_source(ja_cues) is False

    glossary = Glossary(terms=[GlossaryTerm(ja="魔剣", en="Demon Sword", vi="ma kiếm")])
    rel_en = filter_relevant_glossary(en_cues, glossary)
    assert any(t["vi"] == "ma kiếm" for t in rel_en["terms"])

    from upscale import parse_args
    parsed = parse_args(["--prefer-whisper", "episode01.mkv"])
    assert parsed.prefer_whisper is True


# ==============================================================================
# GÓI 3: STAGE 1 OVERLAP RATIO, MORA-AWARE DTW & 2:1 EN CLAUSE SPLITTING
# ==============================================================================


def test_pkg3_stage1_overlap_ratio_prevents_adjacent_bleed(tmp_path):
    ja_srt = tmp_path / "ep.ja.srt"
    en_srt = tmp_path / "ep.en.srt"

    # Cue JA dài 4000ms (0 -> 4000ms), cue EN kế tiếp bắt đầu tại 3650 -> 6650ms (chồng 350ms = 8.75% < 25%)
    ja_srt.write_text("1\n00:00:00,000 --> 00:00:04,000\n静かな夜だな。\n", encoding="utf-8")
    en_srt.write_text("1\n00:00:03,650 --> 00:00:06,650\nWatch out for the ambush ahead!\n", encoding="utf-8")

    cues, _, _ = parse_and_align_subtitles(str(ja_srt), str(en_srt))
    assert len(cues) == 1
    assert cues[0].en is None, "Không được gán lem câu EN kế tiếp khi tỷ lệ chồng lấn < 25%"


def test_pkg3_dtw_mora_density_and_2_to_1_en_splitting():
    p1, p2 = split_en_reference_for_2_to_1(
        "If you step forward right now, I will cut you down without mercy!",
        ratio_first=0.45,
    )
    assert p1 != p2
    assert "step forward" in p1
    assert "cut you down" in p2

    # Kiểm tra align_ja_to_en_monotonic_dtw tách mệnh đề cho cặp 2:1
    ja_cues = [
        SubCue(start_ms=1000, end_ms=2500, text="今一歩でも前に出たら、"),
        SubCue(start_ms=2550, end_ms=4200, text="容赦なく斬り捨てるぞ！"),
    ]
    en_cues = [
        SubCue(
            start_ms=1000,
            end_ms=4200,
            text="If you take one more step forward, I will cut you down without mercy!",
        )
    ]
    aligned = align_ja_to_en_monotonic_dtw(ja_cues, en_cues)
    assert len(aligned) == 2
    assert aligned[0][1] != aligned[1][1]
    assert "step forward" in aligned[0][1]
    assert "cut you down" in aligned[1][1]

    # Kiểm tra phạt lệch mật độ mora vs EN words (1 từ cảm thán ngắn vs câu EN 12 từ)
    long_en_text = "I cannot believe a secret underground labyrinth was hidden in a place like this!"
    cost_short = _compute_dtw_pair_cost(1000, 2000, 1000, 2000, "あっ", long_en_text, [], 2500)
    cost_normal = _compute_dtw_pair_cost(
        1000,
        2000,
        1000,
        2000,
        "まさかこんな場所に秘密の地下迷宮が隠されていたなんて信じられない",
        long_en_text,
        [],
        2500,
    )
    assert cost_short > cost_normal + 0.25


# ==============================================================================
# GÓI 4: LINTER GLOSSARY TERM, CALIBRATED CASCADE, SAFE WRITER, WAL RAG & BATCH
# ==============================================================================


def test_pkg4_linter_glossary_term_and_furigana_vocatives():
    glossary = Glossary(
        terms=[GlossaryTerm(ja="魔剣", en="Demon Sword", vi="ma kiếm", alt=["thanh ma kiếm"])],
        characters=[Character(name_ja="アマンダ", name_vi="Amanda")],
    )

    # 1. Calque mới A Mạn Đạt + thiếu Romaji Amanda + sai thuật ngữ 魔(ま)剣(けん)
    bad_cue = Cue(
        id=10,
        start=0,
        end=4000,
        ja="アマンダ、この魔(ま)剣(けん)を見て！",
        vi="A Mạn Đạt, nhìn thanh kiếm phép này đi!",
    )
    violations = lint_cue(bad_cue, glossary=glossary)
    rule_ids = {v.rule_id for v in violations}
    assert RULE_BANNED_TERM in rule_ids
    assert RULE_ROMAJI_MISSING in rule_ids
    assert RULE_GLOSSARY_TERM in rule_ids

    # 2. Bản dịch chuẩn -> 0 lỗi
    good_cue = Cue(
        id=11,
        start=0,
        end=4000,
        ja="玲(れい)琳(りん)、この魔(ま)剣(けん)を見て！",
        vi="Reirin, nhìn thanh ma kiếm này đi!",
    )
    assert lint_cue(good_cue, glossary=glossary) == []


def test_pkg4_cascade_en_conflict_scoring_and_null_response_fail_open():
    # Cue xung đột JA-EN (en_conflict=True) phải đạt điểm >= 4.0 để được đưa vào Flash Repair
    conflict_cue = Cue(
        id=1,
        start=0,
        end=3000,
        ja="兄上、お久しぶりです。",
        en="Hey bro, long time no see!",
        vi="Này ông anh, lâu rồi không gặp!",
        en_conflict=True,
    )
    s = score_cue_difficulty(conflict_cue)
    assert s >= 4.0
    cands = find_cues_needing_repair([conflict_cue], difficulty_threshold=4.0)
    assert len(cands) == 1

    # Khi response.text = None (bị safety block), Cascade phải Fail-Open an toàn giữ nguyên draft
    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()
    blocked_resp = MagicMock()
    blocked_resp.text = None
    blocked_resp.candidates = []
    mock_translator.client.models.generate_content.return_value = blocked_resp

    stats = run_cascade_repair_pass([conflict_cue], repair_translator=mock_translator)
    assert stats["repaired_count"] == 0
    assert stats.get("fail_open_code") == "unrepaired_bad_schema"
    assert conflict_cue.vi == "Này ông anh, lâu rồi không gặp!"


def test_pkg4_writer_preserves_parenthetical_names_and_ass_untranslated_fallback(tmp_path):
    # Tên nhân vật trong ngoặc đơn có chứa chuỗi con 'la', 'thở', 'nói' KHÔNG được bị xóa
    assert clean_and_normalize_vietnamese("(Clara) Cẩn thận đấy!") == "(Clara) Cẩn thận đấy!"
    assert clean_and_normalize_vietnamese("(Lala) Đi thôi nào!") == "(Lala) Đi thôi nào!"
    assert clean_and_normalize_vietnamese("(Black) Đứng lại!") == "(Black) Đứng lại!"
    # Chú thích âm thanh thật vẫn bị gỡ sạch
    assert clean_and_normalize_vietnamese("(tiếng cười) Đứng lại!") == "Đứng lại!"
    assert clean_and_normalize_vietnamese("(thở dài) Thật tình...") == "Thật tình..."

    # ASS writer không được xuất chuỗi '[Chưa dịch]' thô lên màn hình
    ja_ssa = pysubs2.SSAFile()
    ja_ssa.append(pysubs2.SSAEvent(start=0, end=2000, text="Fallback line"))
    cues = [Cue(id=1, start=0, end=2000, ja="テスト", en="Fallback English line", vi="[Chưa dịch]")]
    out_ass = write_subtitles(cues, ja_ssa, tmp_path / "test_out.ass", output_format="ass")
    saved_ssa = pysubs2.load(str(out_ass))
    assert len(saved_ssa) == 1
    assert "[Chưa dịch]" not in saved_ssa[0].text
    assert "Fallback English line" in saved_ssa[0].text


def test_pkg4_rag_wal_furigana_retrieval_and_report_isolated_cache(tmp_path):
    kb = KnowledgeBase(db_path=tmp_path / "knowledge.db")
    kb.add_gold_unit(
        ja_text="この魔剣は進化する。",
        vi_text="Thanh ma kiếm này có thể tiến hóa.",
        series="tenken",
        episode="ep01",
        status="gold",
    )

    # Cue mới có Furigana xen giữa chữ Hán 魔(ま)剣(けん) vẫn phải truy xuất được Gold Unit & Error Bank
    furigana_cues = [Cue(id=1, start=0, end=2000, ja="この魔(ま)剣(けん)を使え、フ(ふ)ラン！")]
    shots = kb.retrieve_few_shots(furigana_cues, series="tenken", exclude_episode="ep02")
    assert len(shots) == 1
    assert "ma kiếm" in shots[0]["vi"]

    # Học tự động từ vi phạm LINT_GLOSSARY_TERM
    glossary = Glossary(terms=[GlossaryTerm(ja="魔剣", en="Demon Sword", vi="ma kiếm")])
    bad_cue = Cue(id=1, start=0, end=2000, ja="魔(ま)剣(けん)を抜く", vi="Rút kiếm phép ra")
    violations = lint_cue(bad_cue, glossary=glossary)
    learned = kb.learn_from_linter_violations(violations, [bad_cue], series="tenken", episode="ep02")
    assert learned >= 1
    cautions = kb.retrieve_cautions([bad_cue], series="tenken")
    assert any(c["prefer"] == "ma kiếm" for c in cautions)

    # Kiểm tra generate_qa_report nhận isolated cache và kiểm định glossary có Furigana
    iso_cache = TranslationCache(db_path=tmp_path / "isolated_report_cache.db")
    out_srt = tmp_path / "ep02.vi.srt"
    out_srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nRút kiếm phép ra\n", encoding="utf-8")
    rep_path = generate_qa_report(
        cues=[bad_cue],
        ja_path=tmp_path / "ep02.ja.srt",
        en_path=None,
        output_subtitle_path=out_srt,
        report_path=tmp_path / "ep02.audit.md",
        elapsed_time_sec=1.0,
        stats={"dry_run": True},
        glossary=glossary,
        cache=iso_cache,
    )
    rep_text = rep_path.read_text(encoding="utf-8")
    assert "Potential Glossary Mismatches" in rep_text


def test_pkg4_batch_orchestrator_stops_on_daily_quota_exhaustion():
    import scripts.batch_translate_all as batch_mod

    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = "🛑 Model 'gemini-3.5-flash-lite' đã cạn kiệt hạn mức ngày (RPD)\nToàn bộ các model Gemini trong chuỗi đã hết hạn mức."
    fake_proc.stderr = ""

    with patch("scripts.batch_translate_all.subprocess.run", return_value=fake_proc) as mock_run, \
         patch("scripts.batch_translate_all.has_vietnamese_sub", return_value=False):
        res = batch_mod.run_translation_for_file("E:/dummy/ep01.mkv", max_retries=3)

    assert res["status"] == "QUOTA_EXHAUSTED"
    assert res["attempt"] == 1
    assert mock_run.call_count == 1, "Không được retry 3 lần vô ích khi đã cạn hạn mức ngày (RPD)"


# ==============================================================================
# GÓI 5: LIVE DETERMINISTIC EVALUATION HARNESS
# ==============================================================================


def test_pkg5_live_evaluation_harness():
    res = run_evaluation()
    assert res["regression_count"] == 6
    assert res["regression_passed"] == 6
    assert res["pronoun_benchmark_count"] >= 18
    assert res["status"] == "ready"
