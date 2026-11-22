"""ASS/SRT subtitle parser, CC multi-speaker splitter, and Whisper hallucination filter."""

import re
import copy
import logging
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import pysubs2
from subtrans.models import Cue
from subtrans.ja_morph_tagger import strip_ja_furigana, has_cjk_characters

logger = logging.getLogger("subtrans.parser")

DEFAULT_SKIP_STYLES = r"(?i)(^|[_\s\-])(op|ed|sign|song|karaoke|title|credit)([_\s\-]|$)"
DEFAULT_SKIP_STYLES_RE = re.compile(DEFAULT_SKIP_STYLES)
TYPESET_TAG_REGEX = re.compile(r"\\(pos|move|org|clip|iclip|p[1-9])\b", re.IGNORECASE)
ASS_TAG_REGEX = re.compile(r"\{[^\}]*\}")
_WS_REGEX = re.compile(r"\s+")
_ITEM_SORT_KEY_01 = itemgetter(0, 1)
NON_SPEAKER_ACTOR_REGEX = re.compile(
    r"^(?:sign|signs|comment|title|op|ed|ntp|sfx|text|caption|note|insert|screen|eyecatch|song|lyrics|karaoke|default|main)$",
    re.IGNORECASE,
)


def clean_subtitle_text(text: str) -> str:
    """Strips ASS tags, replaces linebreaks, collapses whitespace."""
    # Strip ASS override tags like {\an8}, {\blur1}
    text = ASS_TAG_REGEX.sub("", text)
    # Replace ASS line breaks \N, \n, \h
    text = text.replace(r"\N", " ").replace(r"\n", " ").replace(r"\h", " ")
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    # Collapse multiple whitespaces
    text = _WS_REGEX.sub(" ", text).strip()
    return text


def has_typesetting(raw_text: str) -> bool:
    """Checks if raw cue text contains position or movement tags."""
    return bool(TYPESET_TAG_REGEX.search(raw_text))


def should_skip_cue(style: str, raw_text: str, skip_styles_regex: str = DEFAULT_SKIP_STYLES) -> bool:
    """Checks if cue should be skipped from translation (non-dialogue or typesetting)."""
    if skip_styles_regex == DEFAULT_SKIP_STYLES:
        if DEFAULT_SKIP_STYLES_RE.search(style):
            return True
    elif re.search(skip_styles_regex, style):
        return True
    if has_typesetting(raw_text):
        return True
    return False


SOUND_KEYWORDS_REGEX = re.compile(
    r"(?:^息$|ため息|溜め?息|吐息|寝息|鼻息|息づかい|息切れ|荒い息|せき払い|咳払い|^せき$|^咳$|"
    r"泣き声|笑い声|歓声|罵声|怒号|悲鳴|鳴き声|話し声|歌声|ざわめき声|うめき声|喘ぎ声|"
    r"^音$|物音|足音|筆記音|効果音|雑音|騒音|爆発音|衝撃音|打撃音|金属音|水音|雨音|風音|銃声|鐘の音|鈴の音|チャイムの音|ノック音|[^\s、。！？]{1,10}(?:の音|する音|[開閉]\w*音)|"
    r"鳥の(?:さえずり|鳴き声)|風の音|雨の音|雷鳴|雷の音|ドア(?:の開閉|ノック|が[開閉]\w*)|"
    r"非常ベル|警報ベル|ベルが鳴る|^チャイム$|電話(?:の音|のベル|が鳴る)|"
    r"叫び声|^叫び$|うめき|喘ぎ|すする|啜り|ハアハア|ハァハァ|ゴホ|ケホ|ゲホ|ヒソヒソ|ざわめき|拍手|爆発|衝撃)",
    re.IGNORECASE,
)
LAUGHTER_TOKENS = (
    r"(ふは|フハ|はは|ハハ|へへ|ヘヘ|くく|クク|ひひ|ヒヒ|けけ|ケケ|わは|ワハ|あは|アハ|ほほ|ホホ|ふふ|フフ|ぐふ|グフ|うふ|ウフ|いひ|イヒ)"
    r"[はァぁハへヘくクひヒけケわワあアほホふフぐグうウいイッっー～]*"
)
PURE_INTERJECTION_REGEX = re.compile(
    rf"^((ハア|ハァ|はあ|はぁ|ふう|ふぅ|フウ|フゥ|うっ|くっ|ぐっ|あっ|えっ|ん|んっ|ンッ|ンン|フン|フッ|ふっ|へっ|へ|ほっ|ほ|ちっ|チッ|はむっ|むぐ|ゲホッ|ゴホッ|げほっ|ごほっ|うう|う～|わあ|わぁ|きゃっ|ぎゃっ|ええっ|えっ|ひっ|ひぃ|{LAUGHTER_TOKENS})[\s…\.\-—!！?？ー～]*)+$",
    re.IGNORECASE,
)
MUSIC_OR_SYMBOL_REGEX = re.compile(r"^[♪～\s\-_~♩♫♬]+$")


def is_sound_effect_label(label: str) -> bool:
    """
    Phân biệt chú thích âm thanh SDH trong ngoặc đơn (VD: '筆記音', 'ため息', '扉が開く音', '咳払い')
    với tên nhân vật thật (VD: '絹秀', '珊瑚(さんご)', 'スバル') trong phụ đề DMMTV/Netflix CC.
    """
    if not label:
        return False
    clean = strip_ja_furigana(label.strip())
    if not clean:
        return False
    if PURE_INTERJECTION_REGEX.match(clean) or MUSIC_OR_SYMBOL_REGEX.match(clean):
        return True
    if SOUND_KEYWORDS_REGEX.search(clean):
        return True
    if re.search(r"(?:SE|BGM|SFX)$", clean, re.IGNORECASE):
        return True
    return False


def detect_dialogue_mode_hint(style: str, raw_text: str, clean_ja: str) -> Tuple[Optional[str], str]:
    """
    Nhận diện gợi ý chế độ thoại (`inner`, `flashback`, `narration`) từ style ASS, thẻ `{\\i1}`
    hoặc dấu ngoặc độc thoại nội tâm chuẩn truyền hình Nhật `＜...＞` (và bóc vỏ `＜＞` sạch sẽ).
    """
    st = (style or "").strip()
    mode_hint: Optional[str] = None
    if re.search(r"(?i)(flashback|recollection|memory)", st):
        mode_hint = "flashback"
    elif re.search(r"(?i)(narration|narrator|voiceover)", st):
        mode_hint = "narration"
    elif re.search(r"(?i)(italic|inner|thought|monologue)", st) or re.search(r"\{\\i1\b", raw_text or ""):
        mode_hint = "inner"

    text_out = clean_ja
    m_fw = re.match(r"^＜([^＜＞]+)＞$", (clean_ja or "").strip())
    if m_fw:
        text_out = m_fw.group(1).strip()
        if mode_hint is None:
            mode_hint = "inner"

    return mode_hint, text_out

WHISPER_HALLUCINATION_REGEX = re.compile(
    r"(ご視聴(いただき)?ありがとう|"
    r"ご覧いただきありがとう|"
    r"チャンネル登録|"
    r"高評価|"
    r"^字幕([:：].*)?$|"
    r"字幕(制作|提供|翻訳)|"
    r"^提供([:：].*)?$|"
    r"(ご覧の)?スポンサーの提供)",
    re.IGNORECASE,
)

SCREAM_OR_REPEAT_LOOP_REGEX = re.compile(
    r"(.)\1{5,}|([^\u4e00-\u9fff\s]{1,3})\2{3,}",
    re.UNICODE,
)


def strip_speaker_prefix(text: str) -> Tuple[str, str]:
    """Tách tên người nói ở đầu câu nếu có: （珊瑚(さんご)）Lời thoại hoặc -（絹秀）Lời thoại -> ('絹秀', 'Lời thoại')."""
    t = text.strip()
    dash_match = re.match(r"^[-–—]\s*", t)
    if dash_match:
        t = t[dash_match.end():].strip()

    if not (t.startswith("（") or t.startswith("(")):
        return "", t
    depth = 0
    idx = -1
    for i, ch in enumerate(t):
        if ch in ("（", "("):
            depth += 1
        elif ch in ("）", ")"):
            depth -= 1
            if depth == 0:
                idx = i
                break
    if idx != -1:
        prefix = t[1:idx].strip()
        rest = t[idx + 1:].strip()
        return prefix, rest
    return "", t


def is_pure_sound_cue(text: str, is_whisper: bool = False) -> bool:
    """Kiểm tra cue thuần mô tả âm thanh, tiếng thở, tiếng kêu hoặc SDH không chứa lời thoại."""
    if not text:
        return True
    t = text.strip()
    if MUSIC_OR_SYMBOL_REGEX.match(t):
        return True

    if is_whisper:
        if WHISPER_HALLUCINATION_REGEX.search(t):
            return True
        if SCREAM_OR_REPEAT_LOOP_REGEX.search(t):
            return True

    prefix, rest = strip_speaker_prefix(t)
    if prefix:
        # Nếu có lời thoại thật đằng sau và không phải tiếng than thở thuần túy -> Không phải pure sound cue
        if rest and not PURE_INTERJECTION_REGEX.match(rest):
            if is_whisper and (WHISPER_HALLUCINATION_REGEX.search(rest) or SCREAM_OR_REPEAT_LOOP_REGEX.search(rest)):
                return True
            return False
        return True

    if PURE_INTERJECTION_REGEX.match(t):
        return True
    if SOUND_KEYWORDS_REGEX.search(t) and len(t) <= 25:
        bracket_match = re.match(r"^[（\(\[【].+[）\)\]】]$", t)
        if bracket_match:
            return True

    return bool(re.match(r"^[（\(][^）\)]+[）\)]$", t))


def is_multi_speaker_or_multi_event_text(raw_text: str) -> Tuple[bool, List[str]]:
    """
    Phát hiện các cue Closed Captions (CC) chứa nhiều người nói hoặc vừa có chú thích âm thanh vừa có thoại:
    Ví dụ:
        -（絹秀）金淑妃(きんしゅくひ)\\N-（清佳）あっ…
        （扉が開く音）\\N（莉莉）慧月様
    """
    clean_raw = ASS_TAG_REGEX.sub("", raw_text).strip()
    raw_lines = re.split(r"\\N|\\n|\r\n|\n", clean_raw)
    lines = [l.strip() for l in raw_lines if l.strip()]
    if len(lines) < 2:
        return False, []

    prefix_pattern = re.compile(r"^[-–—]\s*|^[（\(][^）\)]+[）\)]")
    hyphen_pattern = re.compile(r"^[-–—]\s*")

    # Nếu có từ 2 dòng trở lên bắt đầu bằng dấu gạch ngang -> chắc chắn là multi-speaker dialogue
    hyphen_count = sum(1 for l in lines if hyphen_pattern.match(l))
    if hyphen_count >= 2:
        return True, lines

    # Nếu tất cả các dòng đều bắt đầu bằng speaker prefix hoặc sound cue [（(]...[）)]
    all_prefixed = all(bool(prefix_pattern.match(l)) for l in lines)
    if all_prefixed:
        return True, lines

    # Trường hợp dòng 1 là chú thích âm thanh trọn ngoặc đơn '(筆記音)' và dòng 2 là độc thoại '＜...＞' hoặc lời thoại
    first_only_sound = bool(re.match(r"^[（\(][^）\)]+[）\)]$", lines[0])) and is_sound_effect_label(lines[0][1:-1])
    if first_only_sound:
        return True, lines

    return False, []


def split_multi_speaker_cc_events(events: List[pysubs2.SSAEvent]) -> List[pysubs2.SSAEvent]:
    """
    Tách các event Closed Captions (CC) gộp nhiều người nói thành các event độc lập cùng start/end.
    Điều này đảm bảo mỗi nhân vật đều được dịch độc lập và không bị Gemini bỏ quên câu thoại.
    """
    if not events:
        return []

    split_events: List[pysubs2.SSAEvent] = []
    for e in events:
        is_multi, lines = is_multi_speaker_or_multi_event_text(e.text)
        if is_multi:
            for line in lines:
                ev = copy.deepcopy(e)
                ev.text = line
                split_events.append(ev)
        else:
            split_events.append(copy.deepcopy(e))
    return split_events


def deduplicate_consecutive_cc_events(events: List[pysubs2.SSAEvent]) -> List[pysubs2.SSAEvent]:
    """
    Hợp nhất các event Closed Captions (CC) liên tiếp bị phân mảnh có cùng nội dung thoại.
    Các nguồn phụ đề như Netflix CC thường chia 1 câu thoại thành nhiều event nối đuôi nhau
    hoặc chèn khung roll-up lặp lại văn bản.
    """
    if not events:
        return []

    sorted_evts = sorted(events, key=lambda e: (e.start, e.end))
    deduped: List[pysubs2.SSAEvent] = []

    for e in sorted_evts:
        txt = clean_subtitle_text(e.text).strip()
        if not txt:
            continue

        if deduped:
            prev = deduped[-1]
            prev_txt = clean_subtitle_text(prev.text).strip()
            # Nếu cùng nội dung thoại và thời gian nối tiếp hoặc chồng lấn (gap <= 350ms)
            if prev_txt == txt and e.start <= prev.end + 350:
                prev.end = max(prev.end, e.end)
                continue

        deduped.append(copy.deepcopy(e))

    return deduped


def snap_whisper_cues_to_en(
    cues: List[Cue],
    en_events: List[Any],
    min_gap_ms: int = 50,
    skip_styles_regex: str = DEFAULT_SKIP_STYLES,
    cast_names_ja_en: Optional[List[Tuple[str, str]]] = None,
) -> int:
    """
    Auto-Align Subtitle Timestamps:
    Ghim mốc thời gian của Whisper (chứa jitter ±200–500ms) theo track phụ đề tiếng Anh phòng thu.

    Thuật toán 3-Pass:
      - Pass 1: Ghép cặp tối ưu (Best Overlap >= 200ms hoặc >= 20% thời lượng; Fallback khoảng cách tâm <= 500ms).
      - Pass 2: Phân tách vùng thời gian dùng chung khi 2 câu thoại Whisper liên tiếp cùng thuộc về 1 câu EN.
      - Pass 3: Đảm bảo tính đơn điệu nghiêm ngặt (Monotonicity & Gap >= 50ms, thời lượng tối thiểu 250ms).

    Returns:
      Số lượng cue thoại đã được ghim thành công theo mốc thời gian phòng thu.
    """
    if not cues or not en_events:
        return 0

    clean_en_cues: List[Dict[str, Any]] = []
    for idx, e in enumerate(en_events):
        if isinstance(e, dict):
            raw_text = e.get("text") or ""
            if should_skip_cue(e.get("style") or "Default", raw_text, skip_styles_regex):
                continue
            start = int(e.get("start", 0) or 0)
            end = int(e.get("end", 0) or 0)
            if end <= start:
                continue
            cleaned = clean_subtitle_text(raw_text)
            if not cleaned or MUSIC_OR_SYMBOL_REGEX.match(cleaned):
                continue
            raw_id = e.get("id")
            clean_en_cues.append({
                "id": idx if raw_id is None else raw_id,
                "start": start,
                "end": end,
                "duration": max(1, end - start),
                "text": cleaned,
            })
        else:
            if getattr(e, "is_comment", False):
                continue
            raw_text = getattr(e, "text", "") or ""
            style = getattr(e, "style", "Default") or "Default"
            if should_skip_cue(style, raw_text, skip_styles_regex):
                continue
            cleaned = clean_subtitle_text(raw_text)
            if not cleaned or MUSIC_OR_SYMBOL_REGEX.match(cleaned):
                continue
            start = getattr(e, "start", 0)
            end = getattr(e, "end", 0)
            if end <= start:
                continue
            clean_en_cues.append({
                "id": idx,
                "start": start,
                "end": end,
                "duration": max(1, end - start),
                "text": cleaned,
            })

    if not clean_en_cues:
        return 0
    clean_en_cues.sort(key=lambda ec: (ec["start"], ec["end"], 0 if ec.get("id") is None else ec["id"]))

    dialogue_cues = [c for c in cues if not c.is_skipped and not c.is_dropped_sound]
    if not dialogue_cues:
        return 0
    dialogue_cues.sort(key=lambda c: (c.start, c.end, c.id))

    from subtrans.dtw_auto_align import SubCue, get_dtw_alignment_plan
    ja_sub_cues = [SubCue(start_ms=c.start, end_ms=c.end, text=c.ja) for c in dialogue_cues]
    en_sub_cues = [SubCue(start_ms=ec["start"], end_ms=ec["end"], text=ec["text"]) for ec in clean_en_cues]

    cue_map, split_1_to_2_map = get_dtw_alignment_plan(
        ja_sub_cues,
        en_sub_cues,
        cast_names_ja_en=cast_names_ja_en or (),
    )

    # Cập nhật mốc thời gian và EN reference chuẩn phòng thu theo đúng chỉ mục cue gốc
    snapped_count = 0
    spawned_cues: List[Cue] = []
    for idx, c in enumerate(dialogue_cues):
        if idx in split_1_to_2_map:
            parts = split_1_to_2_map[idx]
            s1, e1, ja1, en1 = parts[0]
            s2, e2, ja2, en2 = parts[1]
            c.start = s1
            c.end = e1
            c.ja = ja1
            if en1:
                c.en = en1
            c2 = Cue(
                id=c.id,
                start=s2,
                end=e2,
                style=c.style,
                ja=ja2,
                en=en2 or None,
                speaker=c.speaker,
                mode_hint=c.mode_hint,
                raw_text=ja2,
            )
            spawned_cues.append(c2)
            snapped_count += 2
        elif idx in cue_map and cue_map[idx] is not None:
            new_start, new_end, paired_en = cue_map[idx]
            c.start = new_start
            c.end = new_end
            if paired_en:
                c.en = paired_en
            snapped_count += 1
        else:
            # Nhiệm vụ 4: Lọc bỏ Orphan Whisper Cues (không khớp với bất kỳ câu EN nào)
            min_dist = float("inf")
            for ec in clean_en_cues:
                if c.end < ec["start"]:
                    dist = ec["start"] - c.end
                elif ec["end"] < c.start:
                    dist = c.start - ec["end"]
                else:
                    dist = 0
                if dist < min_dist:
                    min_dist = dist

            if min_dist > 0 and (
                min_dist > 2500
                or (len(c.ja) <= 6 and min_dist > 600)
                or is_pure_sound_cue(c.ja, is_whisper=True)
            ):
                c.is_dropped_sound = True
                c.is_skipped = True
                c.backend_used = "dropped_sound"
                logger.debug(
                    f"Orphan Whisper cue #{c.id} ('{c.ja}') dropped: dist_to_en={min_dist}ms, len={len(c.ja)}"
                )

    if spawned_cues:
        cues.extend(spawned_cues)
        cues.sort(key=lambda c: (c.start, c.end, c.id))
        for new_id, c_item in enumerate(cues):
            c_item.id = new_id

    # Pass 3: Đảm bảo tính đơn điệu và khoảng cách tối thiểu giữa các cue liên tiếp
    # Sửa lỗi Pass 3 (Nhiệm vụ 5): Chỉ chuẩn hóa trên active_cues (không bị skip và không bị drop sound)
    active_cues = [c for c in cues if not c.is_skipped and not c.is_dropped_sound]
    active_cues.sort(key=lambda c: (c.start, c.end, c.id))

    for i in range(len(active_cues) - 1):
        c1 = active_cues[i]
        c2 = active_cues[i + 1]
        if c1.end <= c1.start:
            c1.end = c1.start + 300

        # Nếu 2 cue liên tiếp được gắn cùng mốc thời gian bắt đầu từ sự kiện tách câu nhiều người nói
        # (hoặc chênh lệch abs(c2.start - c1.start) <= 15 và cùng end), giữ nguyên cấu trúc Dual Dialogue
        is_dual = (abs(c2.start - c1.start) <= 15 and abs(c2.end - c1.end) <= 15)
        if is_dual:
            continue

        if c2.start < c1.start + min_gap_ms:
            c2.start = c1.start + min_gap_ms
        if c1.end >= c2.start:
            if c2.start - min_gap_ms >= c1.start + 250:
                c1.end = c2.start - min_gap_ms
            else:
                c1.end = max(c1.end, c1.start + 250)
                c2.start = c1.end + min_gap_ms
                if c2.end <= c2.start:
                    c2.end = c2.start + 300

    if active_cues and active_cues[-1].end <= active_cues[-1].start:
        active_cues[-1].end = active_cues[-1].start + 300

    cues.sort(key=lambda c: (c.start, c.end, c.id))

    return snapped_count


def parse_and_align_subtitles(
    ja_path: str,
    en_path: Optional[str] = None,
    en_offset_ms: int = 0,
    ja_offset_ms: int = 0,
    skip_styles_regex: str = DEFAULT_SKIP_STYLES,
    overlap_threshold: float = 0.5,
    match_warning_pct: float = 0.8,
    sound_cues: str = "drop",
    auto_align_whisper_to_en: bool = True,
    is_whisper: Optional[bool] = None,
    cast_names_ja_en: Optional[List[Tuple[str, str]]] = None,
    piecewise_ja_offsets: Optional[List[Dict[str, int]]] = None,
) -> Tuple[List[Cue], pysubs2.SSAFile, Optional[pysubs2.SSAFile]]:
    """
    Parses Japanese and optional English subtitles, cleans text, and aligns them by time overlap.
    Returns:
        (cues, original_ja_ssa, original_en_ssa)
    """
    ja_ssa = pysubs2.load(ja_path)
    # Loại bỏ các dòng Comment: trong file ASS/SSA trước khi xử lý (chú thích của fansub / karaoke template)
    ja_ssa.events = [ev for ev in ja_ssa.events if not getattr(ev, "is_comment", False)]
    if piecewise_ja_offsets and len(piecewise_ja_offsets) >= 2:
        split_ms = int(piecewise_ja_offsets[0].get("until_ms", 0))
        off_before = int(piecewise_ja_offsets[0].get("offset_ms", 0))
        off_after = int(piecewise_ja_offsets[1].get("offset_ms", 0))
        kept_events = []
        for ev in ja_ssa.events:
            shift_val = off_before if ev.start < split_ms else off_after
            new_start = int(ev.start) + shift_val
            new_end = int(ev.end) + shift_val
            if new_end <= 0:
                continue
            ev.start = max(0, new_start)
            ev.end = max(ev.start + 50, new_end)
            kept_events.append(ev)
        ja_ssa.events = kept_events
        logger.info(
            f"Đã dịch chuyển Piecewise 2-Segment phụ đề tiếng Nhật: <{split_ms}ms ({off_before}ms), >={split_ms}ms ({off_after}ms)."
        )
    elif ja_offset_ms != 0:
        ja_ssa.shift(ms=ja_offset_ms)
        kept_events = []
        for ev in ja_ssa.events:
            if int(ev.end) <= 0:
                continue
            ev.start = max(0, int(ev.start))
            ev.end = max(ev.start + 50, int(ev.end))
            kept_events.append(ev)
        ja_ssa.events = kept_events
        logger.info(f"Đã dịch chuyển mốc thời gian phụ đề tiếng Nhật (ja_offset_ms={ja_offset_ms}ms).")
    ja_ssa.events = deduplicate_consecutive_cc_events(ja_ssa.events)
    ja_ssa.events = split_multi_speaker_cc_events(ja_ssa.events)
    en_ssa = pysubs2.load(en_path) if en_path else None
    if is_whisper is None:
        is_whisper = "whisper" in Path(ja_path).name.lower()

    # Pre-process EN cues if available
    en_cues = []
    if en_ssa:
        for idx, event in enumerate(en_ssa):
            if getattr(event, "is_comment", False):
                continue
            if should_skip_cue(getattr(event, "style", "Default") or "Default", event.text, skip_styles_regex):
                continue
            clean_en = clean_subtitle_text(event.text)
            if not clean_en:
                continue
            # Apply manual offset to EN start and end times (drop shifted <= 0, clamp partial overlap)
            en_start = int(event.start) + en_offset_ms
            en_end = int(event.end) + en_offset_ms
            if en_offset_ms != 0:
                if en_end <= 0:
                    continue
                en_start = max(0, en_start)
                en_end = max(en_start + 50, en_end)
            dur = max(1, en_end - en_start)
            en_cues.append({
                "id": idx,
                "start": en_start,
                "end": en_end,
                "duration": dur,
                "text": clean_en
            })
        en_cues.sort(key=lambda ec: (ec["start"], ec["end"], ec["id"]))

    cues: List[Cue] = []
    matched_en_count = 0
    dialogue_cue_count = 0

    for idx, event in enumerate(ja_ssa):
        raw_text = event.text
        clean_ja = clean_subtitle_text(raw_text)
        is_skip = should_skip_cue(event.style, raw_text, skip_styles_regex)

        # Trích xuất cột Actor/Name của file .ass nếu chứa tên nhân vật hợp lệ
        speaker_name: Optional[str] = None
        actor_raw = (getattr(event, "name", None) or "").strip()
        if actor_raw and not NON_SPEAKER_ACTOR_REGEX.match(actor_raw) and not is_sound_effect_label(actor_raw):
            speaker_name = actor_raw

        # Tách tiền tố người nói hoặc âm thanh nếu có (ví dụ: -（絹秀）Lời thoại hoặc （筆記音）＜...＞)
        prefix, rest = strip_speaker_prefix(clean_ja)
        if prefix:
            if not is_sound_effect_label(prefix):
                speaker_name = prefix
        if rest:
            clean_ja = rest

        # Nhận diện mode_hint (inner/flashback/narration) và bóc sạch ngoặc độc thoại '＜...＞'
        mode_hint, clean_ja = detect_dialogue_mode_hint(event.style, raw_text, clean_ja)

        # Kiểm tra sound cue / SDH
        is_sound = is_pure_sound_cue(clean_ja, is_whisper=bool(is_whisper))
        is_dropped_sound = False
        if is_sound and sound_cues == "drop":
            is_dropped_sound = True
            is_skip = True

        matching_en_cues = []
        aligned_en = None
        if not is_skip and clean_ja:
            if en_cues:
                dur_ja = max(1, event.end - event.start)
                for ec in en_cues:
                    # Calculate time overlap between JA and EN cue
                    overlap_start = max(event.start, ec["start"])
                    overlap_end = min(event.end, ec["end"])
                    overlap_duration = max(0, overlap_end - overlap_start)

                    min_dur = max(1, min(dur_ja, ec["duration"]))
                    if (
                        overlap_duration >= (ec["duration"] * overlap_threshold)
                        or overlap_duration >= (dur_ja * overlap_threshold)
                        or (overlap_duration >= 300 and (overlap_duration / min_dur) >= 0.25)
                    ):
                        matching_en_cues.append(ec)

                if matching_en_cues:
                    aligned_en = " ".join(ec["text"] for ec in matching_en_cues)
                    matched_en_count += 1
                else:
                    # Fallback tìm EN cue gần nhất trong phạm vi sai số ±800ms nếu không có overlap trực tiếp
                    nearest_en = None
                    min_dist = float("inf")
                    for ec in en_cues:
                        dist = abs((event.start + event.end) / 2.0 - (ec["start"] + ec["end"]) / 2.0)
                        if dist < 800 and dist < min_dist:
                            min_dist = dist
                            nearest_en = ec
                    if nearest_en:
                        matching_en_cues.append(nearest_en)
                        aligned_en = nearest_en["text"]
                        matched_en_count += 1

            dialogue_cue_count += 1

        # Trục thời gian chuẩn xác: Luôn giữ nguyên start/end từ tiếng Nhật gốc (đã được auto-offset)
        # EN chỉ dùng làm tham chiếu dịch thuật ngữ nghĩa, không dùng để ghi đè timeline nhằm triệt tiêu lỗi đè phụ đề
        cue_start = event.start
        cue_end = event.end

        cue = Cue(
            id=idx,
            start=cue_start,
            end=cue_end,
            style=event.style,
            ja=clean_ja,
            en=aligned_en,
            vi=raw_text if (is_skip and not is_dropped_sound) else None,
            backend_used="dropped_sound" if is_dropped_sound else ("skipped" if is_skip else None),
            is_skipped=is_skip,
            is_dropped_sound=is_dropped_sound,
            speaker=speaker_name,
            mode_hint=mode_hint,
            raw_text=raw_text
        )
        cues.append(cue)

    # Sắp xếp danh sách cue tuần tự theo start time tăng dần (xử lý triệt để >=3 cue trùng start hoặc bị xen bởi skipped cue)
    cues.sort(key=lambda c: (c.start, c.end, c.id))
    last_active_start: Optional[int] = None
    for c in cues:
        if c.is_skipped:
            continue
        if last_active_start is not None and c.start <= last_active_start:
            c.start = last_active_start + 10
            c.end = max(c.end, c.start + 300)
        last_active_start = c.start
    cues.sort(key=lambda c: (c.start, c.end, c.id))

    # Tự động ghim mốc thời gian Whisper theo track phụ đề EN phòng thu (Auto-Align Subtitle Timestamps)
    if auto_align_whisper_to_en and is_whisper and en_ssa and en_cues:
        snapped_count = snap_whisper_cues_to_en(
            cues=cues,
            en_events=en_cues,
            min_gap_ms=50,
            skip_styles_regex=skip_styles_regex,
            cast_names_ja_en=cast_names_ja_en,
        )
        if snapped_count > 0:
            logger.info(
                f"Auto-Align Subtitle Timestamps: Đã ghim thành công mốc thời gian {snapped_count}/{dialogue_cue_count} "
                f"câu thoại Whisper theo track phụ đề EN phòng thu."
            )
            print(
                f"🎯 [AUTO-ALIGN] Đã tự động ghim mốc thời gian {snapped_count}/{dialogue_cue_count} "
                f"câu Whisper theo keyframe phụ đề EN phòng thu."
            )

    dropped_count = sum(1 for c in cues if c.is_dropped_sound)
    if dropped_count > 0:
        logger.info(f"Đã loại bỏ {dropped_count} cue chú thích âm thanh thuần ngoặc đơn / SDH (--sound-cues {sound_cues}).")


    # Check match warning threshold
    if en_ssa and dialogue_cue_count > 0:
        match_rate = matched_en_count / dialogue_cue_count
        if match_rate < match_warning_pct:
            print(
                f"\n⚠️  [ALIGNMENT WARNING] Only {match_rate*100:.1f}% of dialogue cues found an English match "
                f"({matched_en_count}/{dialogue_cue_count} cues). "
                f"Check if releases differ or adjust timing with --en-offset-ms.\n"
            )

    return cues, ja_ssa, en_ssa


def is_en_fallback_source(cues: List[Cue]) -> bool:
    """
    Phát hiện trường hợp luồng phụ đề truyền vào `ja_path` thực chất là phụ đề tiếng Anh phòng thu
    (ví dụ: container MKV không có softsub JA nên dùng trực tiếp `en_sub` làm gốc đồng bộ khung hình).
    Trả về True khi tập có câu thoại hoạt động nhưng 0% câu thoại chứa ký tự CJK và có chữ cái Latin.
    """
    active = [c for c in cues if not c.is_skipped and not getattr(c, "is_dropped_sound", False) and (c.ja or "").strip()]
    if not active:
        return False
    cjk_count = sum(1 for c in active if has_cjk_characters(c.ja))
    if cjk_count > 0:
        return False
    latin_count = sum(1 for c in active if re.search(r"[A-Za-z]{2,}", c.ja or ""))
    return latin_count >= max(1, int(len(active) * 0.5))


def _match_rate_sorted_precomputed(
    ja_intervals: List[Tuple[int, int]],
    en_thresholds: List[Tuple[int, int, float]],
    offset_ms: int = 0,
) -> float:
    """Compute O(N+M) dialogue match rate using precomputed (start, end, min_overlap) EN tuples."""
    if not ja_intervals or not en_thresholds:
        return 0.0

    n_en = len(en_thresholds)
    en_ptr = 0
    active: List[Tuple[int, int, float]] = []
    matched = 0

    for js, je in ja_intervals:
        j_start = js + offset_ms
        j_end = je + offset_ms
        while en_ptr < n_en and en_thresholds[en_ptr][0] < j_end:
            item = en_thresholds[en_ptr]
            if item[1] > j_start:
                active.append(item)
            en_ptr += 1
        if active:
            if len(active) == 1:
                es, ee, req = active[0]
                if ee <= j_start:
                    active.clear()
                else:
                    overlap = (j_end if j_end < ee else ee) - (j_start if j_start > es else es)
                    if overlap > 0 and overlap >= req:
                        matched += 1
            else:
                active = [item for item in active if item[1] > j_start]
                for es, ee, req in active:
                    overlap = (j_end if j_end < ee else ee) - (j_start if j_start > es else es)
                    if overlap > 0 and overlap >= req:
                        matched += 1
                        break

    return matched / len(ja_intervals)


def _match_rate_sorted(
    ja_intervals: List[Tuple[int, int]],
    en_intervals: List[Tuple[int, int]],
    offset_ms: int = 0,
    overlap_threshold: float = 0.5,
) -> float:
    """Compute O(N+M) dialogue match rate over start-sorted positive-duration intervals."""
    if not ja_intervals or not en_intervals:
        return 0.0

    en_thresholds = [(es, ee, overlap_threshold * (ee - es)) for es, ee in en_intervals]
    return _match_rate_sorted_precomputed(ja_intervals, en_thresholds, offset_ms=offset_ms)


def compute_alignment_match_rate(
    ja_events: List[Any],
    en_events: List[Any],
    offset_ms: int = 0,
    overlap_threshold: float = 0.5,
) -> float:
    """Calculates dialogue match rate (percentage of JA events having overlapping EN cue)."""
    ja_intervals = sorted((j.start, j.end) for j in ja_events if (j.end - j.start) > 0)
    if not ja_intervals:
        return 0.0
    en_intervals = sorted((e.start, e.end) for e in en_events if (e.end - e.start) > 0)
    if not en_intervals:
        return 0.0
    return _match_rate_sorted(
        ja_intervals,
        en_intervals,
        offset_ms=offset_ms,
        overlap_threshold=overlap_threshold,
    )


def detect_auto_offset(
    ja_path: str,
    en_path: Optional[str],
    skip_styles_regex: str = DEFAULT_SKIP_STYLES,
) -> Tuple[int, bool, str, dict]:
    """
    Detects timing offset between Japanese and English subtitles.

    offset is defined as median shift needed to align JA with EN (in ms).
    Criteria to apply:
      1. |offset| >= 0.2s (200 ms)
      2. >= 30 candidate pairs in main cluster
      3. quintile drift <= 0.5s (max diff among medians of 5 chronological quintiles)
      4. match rate improvement >= 15.0 percentage points

    Returns:
      (offset_ms, is_applied, reason_str, stats_dict)
    """
    if not en_path:
        return 0, False, "Không có file phụ đề tiếng Anh tham chiếu", {}

    try:
        ja_ssa = pysubs2.load(ja_path)
        en_ssa = pysubs2.load(en_path)
    except Exception as e:
        return 0, False, f"Lỗi đọc file phụ đề: {e}", {}

    skip_re = DEFAULT_SKIP_STYLES_RE if skip_styles_regex == DEFAULT_SKIP_STYLES else re.compile(skip_styles_regex)
    style_skip_cache: Dict[str, bool] = {}

    def _extract_items(ssa: pysubs2.SSAFile, check_fw_q: bool) -> List[Tuple[int, int, int, bool]]:
        items: List[Tuple[int, int, int, bool]] = []
        for e in ssa:
            if getattr(e, "is_comment", False):
                continue
            s = int(e.start)
            end = int(e.end)
            if end <= s:
                continue
            style = getattr(e, "style", "Default") or "Default"
            is_skip_style = style_skip_cache.get(style)
            if is_skip_style is None:
                is_skip_style = bool(skip_re.search(style))
                style_skip_cache[style] = is_skip_style
            if is_skip_style:
                continue
            txt = e.text or ""
            if "{" not in txt and "\\" not in txt:
                if not txt.strip():
                    continue
            else:
                if has_typesetting(txt) or not clean_subtitle_text(txt):
                    continue
            has_q = ("?" in txt or "？" in txt) if check_fw_q else ("?" in txt)
            items.append((s, end, end - s, has_q))
        items.sort(key=_ITEM_SORT_KEY_01)
        return items

    ja_items = _extract_items(ja_ssa, check_fw_q=True)
    en_items = _extract_items(en_ssa, check_fw_q=False)

    if not ja_items or not en_items:
        return 0, False, "Không tìm thấy thoại hợp lệ trong file JA hoặc EN", {}

    max_offset_ms = 600_000
    en_starts = [e[0] for e in en_items]
    n_ja = len(ja_items)
    if n_ja <= 200:
        sample_ja = ja_items
    else:
        sample_ja = [ja_items[(i * n_ja) // 200] for i in range(200)]

    # Coarse +/-600s scan using 0.2s bins and 3-bin smoothing to surface commercial-break peaks.
    coarse_bins: Counter = Counter()
    for js, _je, jd, jq in sample_ja:
        lo = bisect_left(en_starts, js - max_offset_ms)
        hi = bisect_right(en_starts, js + max_offset_ms)
        for k in range(lo, hi):
            es, _ee, ed, eq = en_items[k]
            if jq != eq:
                continue
            if (jd if jd < ed else ed) * 4 < (jd if jd > ed else ed):
                continue
            coarse_bins[round((es - js) / 200.0)] += 1

    smoothed_scores = [
        (coarse_bins.get(b - 1, 0) + cnt + coarse_bins.get(b + 1, 0), b * 0.2)
        for b, cnt in coarse_bins.items()
    ]
    smoothed_scores.sort(key=lambda x: (-x[0], abs(x[1])))

    peaks_sec: List[float] = []
    for _score, center_sec in smoothed_scores:
        # Skip peaks already well inside the unconditional +/-10s anchor band so 0s never wastes a slot.
        if abs(center_sec) <= 4.0:
            continue
        if all(abs(center_sec - p) >= 8.0 for p in peaks_sec):
            peaks_sec.append(center_sec)
            if len(peaks_sec) >= 4:
                break

    # Unconditionally include 0.0s +/-10s anchor band alongside top coarse peaks (+/-10s > 8s NMS).
    raw_bands = [(-10_000, 10_000)]
    for p in peaks_sec:
        b_lo = max(-max_offset_ms, round((p - 10.0) * 1000))
        b_hi = min(max_offset_ms, round((p + 10.0) * 1000))
        if b_lo <= b_hi:
            raw_bands.append((b_lo, b_hi))
    raw_bands.sort()

    merged_bands_ms: List[Tuple[int, int]] = []
    for b_lo, b_hi in raw_bands:
        if merged_bands_ms and b_lo <= merged_bands_ms[-1][1]:
            prev_lo, prev_hi = merged_bands_ms[-1]
            merged_bands_ms[-1] = (prev_lo, max(prev_hi, b_hi))
        else:
            merged_bands_ms.append((b_lo, b_hi))

    # Monotonic multi-band 2-pointer fine scan.
    n_en = len(en_items)
    band_ptrs = [[0, 0] for _ in merged_bands_ms]
    candidate_pairs: List[Tuple[int, float, int]] = []
    bin_counts: Counter = Counter()
    bin_pair_idxs: Dict[int, List[int]] = {}
    bin_pair_diffs: Dict[int, List[float]] = {}

    for js, _je, jd, jq in ja_items:
        for b_idx, (b_lo, b_hi) in enumerate(merged_bands_ms):
            lo_target = js + b_lo
            hi_target = js + b_hi
            ptrs = band_ptrs[b_idx]
            p_lo, p_hi = ptrs[0], ptrs[1]
            while p_lo < n_en and en_starts[p_lo] < lo_target:
                p_lo += 1
            if p_hi < p_lo:
                p_hi = p_lo
            while p_hi < n_en and en_starts[p_hi] <= hi_target:
                p_hi += 1
            ptrs[0] = p_lo
            ptrs[1] = p_hi
            for k in range(p_lo, p_hi):
                es, _ee, ed, eq = en_items[k]
                if jq != eq:
                    continue
                if (jd if jd < ed else ed) * 4 < (jd if jd > ed else ed):
                    continue
                d = (es - js) / 1000.0
                rd_int = round(d * 10)
                pair_idx = len(candidate_pairs)
                candidate_pairs.append((js, d, rd_int))
                bin_counts[rd_int] += 1
                idxs = bin_pair_idxs.get(rd_int)
                if idxs is None:
                    bin_pair_idxs[rd_int] = [pair_idx]
                    bin_pair_diffs[rd_int] = [d]
                else:
                    idxs.append(pair_idx)
                    bin_pair_diffs[rd_int].append(d)

    if not candidate_pairs:
        return 0, False, "Không tìm thấy cặp thoại tương đồng trong cửa sổ ±600s", {}

    mode_bin_int, _ = bin_counts.most_common(1)[0]
    mode_bin = mode_bin_int / 10.0
    cluster_pairs = [(js, d) for js, d, _rd in candidate_pairs if abs(d - mode_bin) <= 0.5]
    cluster_diffs = [d for _js, d in cluster_pairs]

    if not cluster_diffs:
        return 0, False, "Không xác định được cụm lệch thời gian chính", {}

    med_sec = statistics.median(cluster_diffs)
    med_ms = round(med_sec * 1000)

    # 1. Check |offset| >= 0.2s (Bù tự động từ 200ms trở lên nếu các cặp ghép đồng thuận cao)
    cond_mag = abs(med_sec) >= 0.2

    # 2. Check pairs >= 30
    cond_pairs = len(cluster_diffs) >= 30

    # 3. Check quintile drift <= 0.5s (cluster_pairs is already sorted by ja_start)
    n = len(cluster_pairs)
    q_size = n // 5
    q_medians = []
    if q_size > 0:
        for i in range(5):
            chunk = cluster_pairs[i * q_size : (i + 1) * q_size if i < 4 else n]
            if chunk:
                q_medians.append(statistics.median([p[1] for p in chunk]))
    q_drift = (max(q_medians) - min(q_medians)) if q_medians else 0.0
    cond_drift = q_drift <= 0.5

    # 4. Check match rate improvement >= 15 points using O(N+M) 2-pointer sweep.
    ja_iv = [(s, e) for s, e, _d, _q in ja_items]
    en_thresholds = [(s, e, 0.5 * (e - s)) for s, e, _d, _q in en_items]
    rate_before = _match_rate_sorted_precomputed(ja_iv, en_thresholds, offset_ms=0)
    rate_after = _match_rate_sorted_precomputed(ja_iv, en_thresholds, offset_ms=med_ms)
    rate_diff_points = (rate_after - rate_before) * 100.0
    cond_rate = rate_diff_points >= 15.0

    stats = {
        "offset_ms": med_ms,
        "offset_sec": med_sec,
        "pairs_count": len(cluster_diffs),
        "quintile_drift_sec": q_drift,
        "rate_before": rate_before,
        "rate_after": rate_after,
        "rate_diff_points": rate_diff_points,
    }

    failed_reasons = []
    if not cond_mag:
        failed_reasons.append(f"|offset| ({abs(med_sec):.3f}s) < 0.2s")
    if not cond_pairs:
        failed_reasons.append(f"số cặp ghép ({len(cluster_diffs)}) < 30")
    if not cond_drift:
        failed_reasons.append(f"độ chênh 5 phần ({q_drift:.3f}s) > 0.5s")
    if not cond_rate:
        failed_reasons.append(f"tỷ lệ khớp tăng ({rate_diff_points:.1f} điểm) < 15 điểm")

    if not failed_reasons:
        is_applied = True
        reason_str = (
            f"Thỏa mãn toàn bộ điều kiện: offset={med_ms}ms, {len(cluster_diffs)} cặp, "
            f"độ chênh 5 phần={q_drift:.3f}s, tỷ lệ khớp {rate_before*100:.1f}% -> {rate_after*100:.1f}% (+{rate_diff_points:.1f} điểm)"
        )
    else:
        is_applied = False
        reason_str = f"Không áp dụng do: {', '.join(failed_reasons)} (offset đo được: {med_ms}ms)"

    # Fallback Piecewise 2-Segment (also checks asymmetric commercial breaks where rate_before >= 0.15 and rate_after < 0.90).
    if failed_reasons or (rate_before >= 0.15 and rate_after < 0.90):
        min_pw_margin = 5.0 if failed_reasons else 10.0
        if len(candidate_pairs) >= 30 and len(ja_iv) >= 24:
            unique_starts: List[int] = []
            for s, _e in ja_iv:
                if not unique_starts or s != unique_starts[-1]:
                    unique_starts.append(s)
            step_stride = max(1, len(unique_starts) // 16)
            candidate_splits = unique_starts[step_stride:-step_stride:step_stride]
            if candidate_splits:
                pair_starts = [p[0] for p in candidate_pairs]
                ja_starts = [j[0] for j in ja_iv]
                split_pair_indices = [bisect_left(pair_starts, sp) for sp in candidate_splits]

                chunk_counters: List[Counter] = []
                prev_idx = 0
                for sp_idx in split_pair_indices:
                    chunk_counters.append(Counter(p[2] for p in candidate_pairs[prev_idx:sp_idx]))
                    prev_idx = sp_idx

                left_counter: Counter = Counter()
                right_counter: Counter = Counter(bin_counts)
                best_pw = None
                total_pairs = len(candidate_pairs)
                total_ja = len(ja_iv)

                for idx, sp_ms in enumerate(candidate_splits):
                    cc = chunk_counters[idx]
                    left_counter.update(cc)
                    right_counter.subtract(cc)
                    p_idx = split_pair_indices[idx]
                    l_cnt = p_idx
                    r_cnt = total_pairs - p_idx
                    if l_cnt < 12 or r_cnt < 12:
                        continue
                    l_mode_int, l_mode_cnt = left_counter.most_common(1)[0]
                    r_mode_int, r_mode_cnt = right_counter.most_common(1)[0]
                    if l_mode_cnt <= 0 or r_mode_cnt <= 0:
                        continue
                    if l_mode_int == r_mode_int or max(abs(l_mode_int), abs(r_mode_int)) <= 5:
                        continue

                    l_approx_cnt = sum(left_counter.get(b, 0) for b in range(l_mode_int - 6, l_mode_int + 7))
                    r_approx_cnt = sum(right_counter.get(b, 0) for b in range(r_mode_int - 6, r_mode_int + 7))
                    if l_approx_cnt < 12 or r_approx_cnt < 12 or (l_approx_cnt + r_approx_cnt) < 30:
                        continue

                    l_mode = l_mode_int / 10.0
                    r_mode = r_mode_int / 10.0

                    l_cluster: List[float] = []
                    for b in range(l_mode_int - 6, l_mode_int + 7):
                        b_idxs = bin_pair_idxs.get(b)
                        if not b_idxs:
                            continue
                        cut = bisect_left(b_idxs, p_idx)
                        if cut <= 0:
                            continue
                        b_diffs = bin_pair_diffs[b][:cut]
                        if l_mode_int - 4 <= b <= l_mode_int + 4:
                            l_cluster.extend(b_diffs)
                        else:
                            for dv in b_diffs:
                                if abs(dv - l_mode) <= 0.5:
                                    l_cluster.append(dv)

                    r_cluster: List[float] = []
                    for b in range(r_mode_int - 6, r_mode_int + 7):
                        b_idxs = bin_pair_idxs.get(b)
                        if not b_idxs:
                            continue
                        cut = bisect_left(b_idxs, p_idx)
                        if cut >= len(b_idxs):
                            continue
                        b_diffs = bin_pair_diffs[b][cut:]
                        if r_mode_int - 4 <= b <= r_mode_int + 4:
                            r_cluster.extend(b_diffs)
                        else:
                            for dv in b_diffs:
                                if abs(dv - r_mode) <= 0.5:
                                    r_cluster.append(dv)

                    if len(l_cluster) < 12 or len(r_cluster) < 12 or (len(l_cluster) + len(r_cluster)) < 30:
                        continue

                    off1_ms = round(statistics.median(l_cluster) * 1000)
                    off2_ms = round(statistics.median(r_cluster) * 1000)
                    if abs(off2_ms - off1_ms) < 1000 or max(abs(off1_ms), abs(off2_ms)) < 1000:
                        continue
                    if (max(l_cluster) - min(l_cluster)) > 1.0 or (max(r_cluster) - min(r_cluster)) > 1.0:
                        continue

                    j_idx = bisect_left(ja_starts, sp_ms)
                    if j_idx == 0 or j_idx == total_ja:
                        continue
                    r_left = _match_rate_sorted_precomputed(ja_iv[:j_idx], en_thresholds, offset_ms=off1_ms)
                    r_right = _match_rate_sorted_precomputed(ja_iv[j_idx:], en_thresholds, offset_ms=off2_ms)
                    pw_rate = (r_left * j_idx + r_right * (total_ja - j_idx)) / total_ja
                    pw_gain = (pw_rate - rate_before) * 100.0
                    if pw_gain >= 15.0 and (pw_rate - rate_after) * 100.0 >= min_pw_margin:
                        if best_pw is None or pw_rate > best_pw[0]:
                            best_pw = (
                                pw_rate,
                                pw_gain,
                                sp_ms,
                                off1_ms,
                                off2_ms,
                                len(l_cluster) + len(r_cluster),
                            )

                if best_pw is not None:
                    pw_rate, pw_gain, sp_ms, off1_ms, off2_ms, pw_pairs = best_pw
                    is_applied = True
                    med_ms = off2_ms if abs(off2_ms) >= abs(off1_ms) else off1_ms
                    stats["piecewise_segments"] = [
                        {"until_ms": sp_ms, "offset_ms": off1_ms},
                        {"from_ms": sp_ms, "offset_ms": off2_ms},
                    ]
                    stats["rate_after"] = pw_rate
                    stats["rate_diff_points"] = pw_gain
                    stats["pairs_count"] = pw_pairs
                    reason_str = (
                        f"Áp dụng Piecewise 2-Segment Auto-Offset tại {sp_ms}ms: "
                        f"đoạn 1={off1_ms}ms, đoạn 2={off2_ms}ms ({pw_pairs} cặp, "
                        f"tỷ lệ khớp {rate_before*100:.1f}% -> {pw_rate*100:.1f}%, +{pw_gain:.1f} điểm)"
                    )

    return med_ms, is_applied, reason_str, stats

