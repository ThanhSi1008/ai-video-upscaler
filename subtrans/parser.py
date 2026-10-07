# -*- coding: utf-8 -*-
"""
subtrans/parser.py
Stage 1: Parse, clean, and align Japanese and English subtitles.
"""

import re
import logging
import statistics
from collections import Counter
from typing import List, Tuple, Optional, Any
import pysubs2
from subtrans.models import Cue

logger = logging.getLogger("subtrans.parser")

DEFAULT_SKIP_STYLES = r"(?i)(op|ed|sign|song|karaoke|title|credit)"
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
PURE_INTERJECTION_REGEX = re.compile(
    r"^((ハア|ハァ|はあ|はぁ|ふう|ふぅ|フウ|フゥ|うっ|くっ|あっ|えっ|ん|んっ|フッ|ふっ|へっ|へ|ほっ|ほ|ちっ|チッ|はむっ|うう|う～|わあ|わぁ|きゃっ|ぎゃっ|ええっ|えっ|ひっ|ひぃ)[\s…\.\-—!！?？]*)+$",
    re.IGNORECASE,
)
MUSIC_OR_SYMBOL_REGEX = re.compile(r"^[♪～\s\-_~♩♫♬]+$")


def strip_speaker_prefix(text: str) -> Tuple[str, str]:
    """Tách tên người nói ở đầu câu nếu có: （珊瑚(さんご)）Lời thoại -> ('珊瑚(さんご)', 'Lời thoại')."""
    t = text.strip()
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


def parse_and_align_subtitles(
    ja_path: str,
    en_path: Optional[str] = None,
    en_offset_ms: int = 0,
    ja_offset_ms: int = 0,
    skip_styles_regex: str = DEFAULT_SKIP_STYLES,
    overlap_threshold: float = 0.5,
    match_warning_pct: float = 0.8,
    sound_cues: str = "drop",
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
    en_ssa = pysubs2.load(en_path) if en_path else None

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
            if prefix and rest:
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

