# -*- coding: utf-8 -*-
"""
subtrans/parser.py
Stage 1: Parse, clean, and align Japanese and English subtitles.
"""

import re
import copy
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import pysubs2
from subtrans.models import Cue

logger = logging.getLogger("subtrans.parser")

DEFAULT_SKIP_STYLES = r"(?i)(^|[_\s\-])(op|ed|sign|song|karaoke|title|credit)([_\s\-]|$)"
TYPESET_TAG_REGEX = re.compile(r"\\(pos|move)\b", re.IGNORECASE)
ASS_TAG_REGEX = re.compile(r"\{[^\}]*\}")


def clean_subtitle_text(text: str) -> str:
    """Strips ASS tags, replaces linebreaks, collapses whitespace."""
    # Strip ASS override tags like {\an8}, {\blur1}
    text = ASS_TAG_REGEX.sub("", text)
    # Replace ASS line breaks \N, \n, \h
    text = text.replace(r"\N", " ").replace(r"\n", " ").replace(r"\h", " ")
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_typesetting(raw_text: str) -> bool:
    """Checks if raw cue text contains position or movement tags."""
    return bool(TYPESET_TAG_REGEX.search(raw_text))


def should_skip_cue(style: str, raw_text: str, skip_styles_regex: str = DEFAULT_SKIP_STYLES) -> bool:
    """Checks if cue should be skipped from translation (non-dialogue or typesetting)."""
    if re.search(skip_styles_regex, style):
        return True
    if has_typesetting(raw_text):
        return True
    return False


SOUND_KEYWORDS_REGEX = re.compile(
    r"(息|せき|咳|ため息|泣き声|笑い声|声|音|足音|鳥|叫び|うめき|喘ぎ|すする|啜り|ハア|ハァ|ゴホ|ケホ|ヒソヒソ|ざわめき|物音|ドア|雷|雨|風|ベル|拍手|歓声|悲鳴|鳴き声)",
    re.IGNORECASE,
)
LAUGHTER_TOKENS = (
    r"(ふは|フハ|はは|ハハ|へへ|ヘヘ|くく|クク|ひひ|ヒヒ|けけ|ケケ|わは|ワハ|あは|アハ|ほほ|ホホ|ふふ|フフ|ぐふ|グフ|うふ|ウフ|いひ|イヒ)"
    r"[はァぁハへヘくクひヒけケわワあアほホふフぐグうウいイッっー～]*"
)
PURE_INTERJECTION_REGEX = re.compile(
    rf"^((ハア|ハァ|はあ|はぁ|ふう|ふぅ|フウ|フゥ|うっ|くっ|あっ|えっ|ん|んっ|フッ|ふっ|へっ|へ|ほっ|ほ|ちっ|チッ|はむっ|うう|う～|わあ|わぁ|きゃっ|ぎゃっ|ええっ|えっ|ひっ|ひぃ|{LAUGHTER_TOKENS})[\s…\.\-—!！?？ー～]*)+$",
    re.IGNORECASE,
)
MUSIC_OR_SYMBOL_REGEX = re.compile(r"^[♪～\s\-_~♩♫♬]+$")


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


def is_pure_sound_cue(text: str) -> bool:
    """Kiểm tra cue thuần mô tả âm thanh, tiếng thở, tiếng kêu hoặc SDH không chứa lời thoại."""
    if not text:
        return True
    t = text.strip()
    if MUSIC_OR_SYMBOL_REGEX.match(t):
        return True

    prefix, rest = strip_speaker_prefix(t)
    if prefix:
        # Nếu có lời thoại thật đằng sau và không phải tiếng than thở thuần túy -> Không phải pure sound cue
        if rest and not PURE_INTERJECTION_REGEX.match(rest):
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
            clean_en_cues.append(e)
        else:
            raw_text = getattr(e, "text", "")
            style = getattr(e, "style", "Default")
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

    dialogue_cues = [c for c in cues if not c.is_skipped and not c.is_dropped_sound]
    if not dialogue_cues:
        return 0

    # Thuật toán Monotonic DTW Auto-Align toàn cục:
    from subtrans.dtw_auto_align import align_ja_to_en_monotonic_dtw, SubCue
    ja_sub_cues = [SubCue(start_ms=c.start, end_ms=c.end, text=c.ja) for c in dialogue_cues]
    en_sub_cues = [SubCue(start_ms=ec["start"], end_ms=ec["end"], text=ec["text"]) for ec in clean_en_cues]

    aligned_results = align_ja_to_en_monotonic_dtw(ja_sub_cues, en_sub_cues)

    # Cập nhật mốc thời gian và EN reference chuẩn phòng thu
    snapped_count = 0
    aligned_idx = 0
    for c in dialogue_cues:
        if aligned_idx < len(aligned_results):
            sub_aligned, paired_en = aligned_results[aligned_idx]
            if paired_en:
                c.start = sub_aligned.start_ms
                c.end = sub_aligned.end_ms
                if not c.en:
                    c.en = paired_en
                snapped_count += 1
            aligned_idx += 1

    # Pass 3: Đảm bảo tính đơn điệu và khoảng cách tối thiểu giữa các cue liên tiếp
    cues.sort(key=lambda c: (c.start, c.end))
    for i in range(len(cues) - 1):
        c1 = cues[i]
        c2 = cues[i + 1]
        if c1.end <= c1.start:
            c1.end = c1.start + 300
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

    if cues and cues[-1].end <= cues[-1].start:
        cues[-1].end = cues[-1].start + 300

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
) -> Tuple[List[Cue], pysubs2.SSAFile, Optional[pysubs2.SSAFile]]:
    """
    Parses Japanese and optional English subtitles, cleans text, and aligns them by time overlap.
    Returns:
        (cues, original_ja_ssa, original_en_ssa)
    """
    ja_ssa = pysubs2.load(ja_path)
    if ja_offset_ms != 0:
        ja_ssa.shift(ms=ja_offset_ms)
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
            clean_en = clean_subtitle_text(event.text)
            if not clean_en:
                continue
            # Apply manual offset to EN start and end times
            en_start = event.start + en_offset_ms
            en_end = event.end + en_offset_ms
            dur = max(1, en_end - en_start)
            en_cues.append({
                "id": idx,
                "start": en_start,
                "end": en_end,
                "duration": dur,
                "text": clean_en
            })

    cues: List[Cue] = []
    matched_en_count = 0
    dialogue_cue_count = 0

    for idx, event in enumerate(ja_ssa):
        raw_text = event.text
        clean_ja = clean_subtitle_text(raw_text)
        is_skip = should_skip_cue(event.style, raw_text, skip_styles_regex)

        # Kiểm tra sound cue / SDH
        is_sound = is_pure_sound_cue(clean_ja)
        is_dropped_sound = False
        if is_sound and sound_cues == "drop":
            is_dropped_sound = True
            is_skip = True
        else:
            prefix, rest = strip_speaker_prefix(clean_ja)
            if rest:
                clean_ja = rest

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

                    if (
                        overlap_duration >= (ec["duration"] * overlap_threshold)
                        or overlap_duration >= (dur_ja * overlap_threshold)
                        or overlap_duration >= 300
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
            raw_text=raw_text
        )
        cues.append(cue)

    # Sắp xếp danh sách cue tuần tự theo start time tăng dần (bảo toàn trọn vẹn thời lượng âm thanh gốc)
    cues.sort(key=lambda c: (c.start, c.end))
    for i in range(len(cues) - 1):
        if cues[i].start == cues[i + 1].start:
            cues[i + 1].start += 10
            cues[i + 1].end = max(cues[i + 1].end, cues[i + 1].start + 300)

    # Tự động ghim mốc thời gian Whisper theo track phụ đề EN phòng thu (Auto-Align Subtitle Timestamps)
    if auto_align_whisper_to_en and is_whisper and en_ssa and en_cues:
        clean_en_dialogues = [
            ec for ec in en_cues
            if not should_skip_cue(en_ssa[ec["id"]].style, en_ssa[ec["id"]].text, skip_styles_regex)
        ]
        snapped_count = snap_whisper_cues_to_en(
            cues=cues,
            en_events=clean_en_dialogues if clean_en_dialogues else en_cues,
            min_gap_ms=50,
            skip_styles_regex=skip_styles_regex,
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


def compute_alignment_match_rate(
    ja_events: List[Any],
    en_events: List[Any],
    offset_ms: int = 0,
    overlap_threshold: float = 0.5,
) -> float:
    """Calculates dialogue match rate (percentage of JA events having overlapping EN cue)."""
    matched = 0
    dialogue = 0
    for j in ja_events:
        dur_j = j.end - j.start
        if dur_j <= 0:
            continue
        dialogue += 1
        j_start = j.start + offset_ms
        j_end = j.end + offset_ms
        for e in en_events:
            dur_e = max(1, e.end - e.start)
            overlap = max(0, min(j_end, e.end) - max(j_start, e.start))
            if overlap >= overlap_threshold * dur_e:
                matched += 1
                break
    return matched / dialogue if dialogue > 0 else 0.0


def detect_auto_offset(
    ja_path: str,
    en_path: Optional[str],
    skip_styles_regex: str = DEFAULT_SKIP_STYLES,
) -> Tuple[int, bool, str, dict]:
    """
    Detects timing offset between Japanese and English subtitles.

    offset is defined as median shift needed to align JA with EN (in ms).
    Criteria to apply:
      1. |offset| >= 1.0s (1000 ms)
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

    ja_events = [e for e in ja_ssa if clean_subtitle_text(e.text) and not should_skip_cue(e.style, e.text, skip_styles_regex)]
    en_events = [e for e in en_ssa if clean_subtitle_text(e.text) and not should_skip_cue(e.style, e.text, skip_styles_regex)]

    if not ja_events or not en_events:
        return 0, False, "Không tìm thấy thoại hợp lệ trong file JA hoặc EN", {}

    candidate_diffs = []
    candidate_pairs = []

    for j in ja_events:
        dur_j = j.end - j.start
        if dur_j <= 0:
            continue
        for e in en_events:
            dur_e = e.end - e.start
            if dur_e <= 0:
                continue
            # Filter pairs with low duration similarity
            if min(dur_j, dur_e) / max(dur_j, dur_e) < 0.25:
                continue
            # Filter mismatched sentence types (e.g. question vs statement)
            if ('?' in j.text or '？' in j.text) != ('?' in e.text):
                continue
            d = (e.start - j.start) / 1000.0  # seconds
            if -60.0 <= d <= 60.0:
                candidate_diffs.append(d)
                candidate_pairs.append((j.start, d))

    if not candidate_diffs:
        return 0, False, "Không tìm thấy cặp thoại tương đồng trong cửa sổ ±60s", {}

    # Cluster differences by 0.1s histogram bins
    bins = [round(d, 1) for d in candidate_diffs]
    mode_bin, _ = Counter(bins).most_common(1)[0]
    cluster_pairs = [p for p in candidate_pairs if abs(p[1] - mode_bin) <= 0.5]
    cluster_diffs = [p[1] for p in cluster_pairs]

    if not cluster_diffs:
        return 0, False, "Không xác định được cụm lệch thời gian chính", {}

    med_sec = statistics.median(cluster_diffs)
    med_ms = round(med_sec * 1000)

    # 1. Check |offset| >= 0.2s (Bù tự động từ 200ms trở lên nếu các cặp ghép đồng thuận cao)
    cond_mag = abs(med_sec) >= 0.2

    # 2. Check pairs >= 30
    cond_pairs = len(cluster_diffs) >= 30

    # 3. Check quintile drift <= 0.5s
    cluster_pairs.sort(key=lambda p: p[0])
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

    # 4. Check match rate improvement >= 15 points
    rate_before = compute_alignment_match_rate(ja_events, en_events, offset_ms=0)
    rate_after = compute_alignment_match_rate(ja_events, en_events, offset_ms=med_ms)
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

    return med_ms, is_applied, reason_str, stats

