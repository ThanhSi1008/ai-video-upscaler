"""Playback-optimized SRT/ASS subtitle writer with Dual Dialogue merging and smart-wrap."""

import re
import copy
import difflib
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple
import pysubs2
from subtrans.models import Cue

CONJUNCTIONS = {"và", "nhưng", "thì", "mà", "là", "khi", "vì", "nếu", "để", "rồi", "hay", "hoặc"}


def format_timestamp_srt(ms: int) -> str:
    """Formats milliseconds as HH:MM:SS,mmm (clamped >= 0)."""
    ms = max(0, int(round(ms)))
    total_seconds, milliseconds = divmod(ms, 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _clean_single_dialogue_line(text: str, sound_cues: str = "drop") -> str:
    """Cleans a single dialogue line without linebreaks."""
    if not text:
        return ""

    if text.strip() in ["[Chưa dịch]", "(Chưa dịch)", "[Untranslated]"]:
        return ""

    # 1. Strip actual HTML tags like <i>, </i>, <b>, <font color="...">, <ruby>, etc.
    #    Avoid wiping anime telepathy/system angle brackets such as <Rudeus!> or <Appraisal>.
    text = re.sub(
        r"</?(?:i|b|u|s|font|span|br|ruby|rt|rp|em|strong|sub|sup|p|div)\b[^>]*>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 2. Strip ASS tags like {\an8}, {\pos(1,2)}, etc.
    text = re.sub(r"\{[^\}]*\}", "", text)

    # 3. Strip speaker dashes at start of line: "- ", "-- ", "— "
    text = re.sub(r"^\s*[-–—]\s*", "", text)

    # 4. Strip music symbols
    text = re.sub(r"[♪♫♬♩#]", "", text)

    # 5. Strip bracketed sound effects/descriptions unless sound_cues == "translate".
    # Use strict word boundaries so parenthetical character names like (Clara), (Lala), (Tousetsu) are preserved.
    if sound_cues != "translate":
        text = re.sub(r"\[[^\]]*\]", "", text)
        stripped = re.sub(
            r"\([^\)]*(?<!\w)(?:tiếng\s+\w+|cười|khóc|thở\s+dài|hơi\s+thở|nhạc|âm\s+thanh|la\s+hét|gào\s+hét|laugh(?:ing|s)?|music|sigh(?:s|ing)?|gasp(?:s|ing)?|scream(?:s|ing)?)(?!\w)[^\)]*\)",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if stripped:
            text = stripped

    # 6. Punctuation rules:
    # Replace unicode ellipsis '…' with three ASCII dots '...'
    text = text.replace("…", "...")
    # Collapse 4 or more dots to 3 dots: .... -> ...
    text = re.sub(r"\.{4,}", "...", text)
    # Interrupted speech ending with dashes -> '—'
    text = re.sub(r"\s*--+\s*$", "—", text)
    text = re.sub(r"(\w)\s*-\s*$", r"\1—", text)
    # Max two consecutive ! or ?
    text = re.sub(r"!{3,}", "!!", text)
    text = re.sub(r"\?{3,}", "??", text)

    # 7. Collapse any linebreaks or whitespace into a single clean line
    text = text.replace(r"\N", " ").replace(r"\n", " ").replace(r"\h", " ")
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Strip enclosing quotation marks or fullwidth inner-monologue brackets around an ENTIRE line only
    #    (Never strip when the closing quote appears mid-line, e.g., "'Sư phụ' và 'Đệ tử'")
    for open_q, close_q in (('"', '"'), ("'", "'"), ("“", "”"), ("«", "»"), ("＜", "＞")):
        if text.startswith(open_q) and text.endswith(close_q) and len(text) >= 2:
            interior = text[1:-1]
            if close_q not in interior and open_q not in interior:
                text = interior.strip()
                break

    return unicodedata.normalize("NFC", text)


def clean_and_normalize_vietnamese(text: str, preserve_dual: bool = False, sound_cues: str = "drop") -> str:
    """
    Cleans and normalizes Vietnamese subtitle text per ADDENDUM 6:
    - If preserve_dual is True and text contains dual lines (e.g. simultaneous speakers),
      preserves both lines formatted with leading dashes: '- Line 1\n- Line 2'.
    - Otherwise collapses into a single clean line without linebreaks.
    """
    if not text:
        return ""

    if preserve_dual and ("\n" in text or r"\N" in text):
        raw_lines = [l.strip() for l in re.split(r"\\N|\r\n|\n", text) if l.strip()]
        cleaned = [_clean_single_dialogue_line(l, sound_cues=sound_cues) for l in raw_lines]
        cleaned = [c for c in cleaned if c]
        if len(cleaned) >= 2:
            return "\n".join(f"- {c.lstrip('-—– ')}" for c in cleaned[:2])
        elif len(cleaned) == 1:
            return cleaned[0]
        return ""

    return _clean_single_dialogue_line(text, sound_cues=sound_cues)


def optimize_cues_for_playback(
    cues: List[Cue],
    min_duration_ms: int = 1200,
    min_dual_duration_ms: int = 2200,
    sound_cues: str = "drop",
) -> List[Cue]:
    """
    Tối ưu hóa các khung phụ đề hiển thị cho trải nghiệm xem phim hoàn hảo (IINA/mpv/VLC/TV):
    1. Anti-Duplicate: Loại bỏ triệt để hiện tượng lặp lại cùng một câu hoặc câu con
       khi phụ đề nguồn CC bị roll-up / chẻ mảnh.
    2. Dual Dialogue: Gộp 2 người nói cùng lúc / gần như cùng lúc thành 1 cue 2 dòng
       (- Câu 1\\n- Câu 2) với thời lượng đủ đọc song song.
    3. Rapid Fragments: Hợp nhất các mảnh câu liên tiếp nói quá nhanh (< 1.2s mỗi mảnh)
       khi câu trước chưa kết thúc (kết thúc bằng ,, ..., — hoặc câu sau viết thường).
    4. Minimum Display Duration: Kéo dài thời lượng hiển thị tối thiểu cho câu ngắn.
    5. Strict Zero-Overlap & Gap Normalization: Triệt tiêu 100% hiện tượng chồng lấn thời gian
       giữa các block riêng biệt (c.end <= next_c.start - 40ms), ngăn chặn triệt để lỗi
       IINA/mpv render 2 block cùng lúc thành 3-4 dòng đè kín màn hình.
    """
    kept = [c for c in cues if not getattr(c, "is_dropped_sound", False) and not getattr(c, "is_skipped", False)]
    if not kept:
        return []

    # Sắp xếp ổn định theo start time
    sorted_cues = sorted([copy.deepcopy(c) for c in kept], key=lambda x: (x.start, x.end))

    # Giai đoạn 1: Hợp nhất (Merge Pass) cho Dual Dialogue và Rapid Fragments
    merged: List[Cue] = []
    i = 0
    n = len(sorted_cues)

    while i < n:
        curr = sorted_cues[i]
        curr_text = (curr.vi or "").strip()
        if not curr_text or curr_text in ["[Chưa dịch]", "(Chưa dịch)", "[Untranslated]"]:
            curr_text = (curr.en or curr.ja or "").strip()
            curr.used_en_fallback = True

        clean_curr = clean_and_normalize_vietnamese(curr_text, preserve_dual=False, sound_cues=sound_cues)

        if not clean_curr:
            i += 1
            continue

        if i == n - 1:
            curr.vi = clean_and_normalize_vietnamese(curr_text, preserve_dual=True, sound_cues=sound_cues)
            merged.append(curr)
            break

        next_cue = sorted_cues[i + 1]
        next_text = (next_cue.vi or "").strip()
        if not next_text or next_text in ["[Chưa dịch]", "(Chưa dịch)", "[Untranslated]"]:
            next_text = (next_cue.en or next_cue.ja or "").strip()
            next_cue.used_en_fallback = True

        clean_next = clean_and_normalize_vietnamese(next_text, preserve_dual=False, sound_cues=sound_cues)

        if not clean_next:
            curr.vi = clean_and_normalize_vietnamese(curr_text, preserve_dual=True, sound_cues=sound_cues)
            merged.append(curr)
            i += 1
            continue

        start_diff = abs(next_cue.start - curr.start)
        overlap = min(curr.end, next_cue.end) - max(curr.start, next_cue.start)
        dur1 = max(1, curr.end - curr.start)
        dur2 = max(1, next_cue.end - next_cue.start)
        gap = next_cue.start - curr.end

        # 1. Phát hiện nói đồng thời / overlap (Simultaneous speakers)
        is_simultaneous = (
            (start_diff <= 1000 and (overlap >= 350 or (overlap / min(dur1, dur2)) >= 0.35))
            or (start_diff <= 1400 and overlap >= 600 and (overlap / dur2) >= 0.50)
        )

        # 2. Chuẩn hóa văn bản so sánh trùng lặp / roll-up CC / dị bản dịch đúp:
        norm_curr = re.sub(r"[^\w\s]", "", clean_curr).strip().lower()
        norm_next = re.sub(r"[^\w\s]", "", clean_next).strip().lower()

        # Kiểm tra tiền tố chung dài nhất (Longest common prefix):
        common_len = 0
        min_len = min(len(norm_curr), len(norm_next))
        while common_len < min_len and norm_curr[common_len] == norm_next[common_len]:
            common_len += 1
        common_prefix = norm_curr[:common_len].strip()
        common_words = common_prefix.split()

        # Order-aware token similarity prevents deleting reciprocal Dual Dialogue ("Anh ... em" vs "Em ... anh")
        toks1 = norm_curr.split()
        toks2 = norm_next.split()
        has_high_word_overlap = False
        if toks1 and toks2:
            min_w = min(len(toks1), len(toks2))
            if min_w >= 4:
                seq_ratio = difflib.SequenceMatcher(None, toks1, toks2).ratio()
                if (seq_ratio >= 0.72 and toks1[0] == toks2[0]) or (min_w >= 5 and seq_ratio >= 0.82):
                    has_high_word_overlap = True

        # Preserve intentional dramatic short repetitions across sequential pauses
        # (e.g. "Subaru..." -> [150ms gap] -> "Subaru!" or "Đợi đã!" -> [200ms gap] -> "Đợi đã!")
        is_dramatic_repeat = (
            norm_curr == norm_next
            and not is_simultaneous
            and gap >= 80
            and len(clean_curr) <= 22
            and (gap >= 150 or clean_curr != clean_next)
        )

        is_same_content = (
            not is_dramatic_repeat
            and (
                (norm_curr == norm_next and (next_cue.start <= curr.end + 500 or start_diff <= 1500))
                or (
                    start_diff <= 800
                    and len(norm_curr) >= 15
                    and len(norm_next) >= 15
                    and (norm_next.startswith(norm_curr) or norm_curr.startswith(norm_next))
                )
                or (
                    (is_simultaneous or start_diff <= 1200)
                    and (
                        (len(common_prefix) >= 15 and (len(common_words) >= 3 or common_len >= min_len * 0.5))
                        or has_high_word_overlap
                    )
                )
            )
        )

        # Ca A: Trùng lặp nội dung hoặc bao hàm nhau (Duplicate / Roll-up Fragments)
        if is_same_content:
            chosen_text = clean_curr if len(clean_curr) >= len(clean_next) else clean_next
            curr.start = min(curr.start, next_cue.start)
            curr.end = max(curr.end, next_cue.end)
            curr.vi = chosen_text
            merged.append(curr)
            i += 2
            continue

        # Ca B: Hai người nói cùng lúc hoặc gần như cùng lúc (Dual Dialogue)
        if is_simultaneous and not is_same_content:
            line1 = f"- {clean_curr.lstrip('-—– ')}"
            line2 = f"- {clean_next.lstrip('-—– ')}"
            dual_str = f"{line1}\n{line2}"

            new_start = min(curr.start, next_cue.start)
            new_end = max(curr.end, next_cue.end)
            total_chars = len(clean_curr) + len(clean_next)
            needed_dual = max(min_dual_duration_ms, min(6500, int(total_chars * 42)))
            new_end = max(new_end, new_start + needed_dual)

            curr.start = new_start
            curr.end = new_end
            curr.vi = dual_str
            merged.append(curr)
            i += 2
            continue

        # Ca C: Một câu bị tách làm 2 mảnh nói quá nhanh (Chained Rapid Fragments)
        is_close_gap = (-150 <= gap <= 350)
        is_too_fast = (dur1 < 1200 or dur2 < 1200)
        total_len = len(clean_curr) + len(clean_next)
        is_compact = (total_len <= 85)

        has_continuation_start = bool(clean_next and clean_next[0].islower())
        is_continuation = (
            clean_curr.endswith((",", "—", "..."))
            and has_continuation_start
        )
        is_completed_sentence = (
            clean_curr.endswith(("!", "?"))
            or (clean_curr.endswith(".") and not clean_curr.endswith("..."))
        )

        if is_close_gap and is_too_fast and is_compact and is_continuation and not is_completed_sentence:
            if clean_curr.endswith((",", "...", "—")):
                merged_str = f"{clean_curr} {clean_next}"
            else:
                merged_str = f"{clean_curr}, {clean_next}"

            new_start = curr.start
            new_end = max(next_cue.end, curr.start + min_duration_ms)
            curr.start = new_start
            curr.end = new_end
            curr.vi = merged_str
            merged.append(curr)
            i += 2
            continue

        curr.vi = clean_and_normalize_vietnamese(curr_text, preserve_dual=True, sound_cues=sound_cues)
        merged.append(curr)
        i += 1

    # Giai đoạn 2: Post-merge Dedup Pass (Loại bỏ triệt để khối lặp còn sót)
    deduped_merged: List[Cue] = []
    for c in merged:
        if deduped_merged:
            prev = deduped_merged[-1]
            p_norm = re.sub(r"[^\w\s]", "", prev.vi or "").strip().lower()
            c_norm = re.sub(r"[^\w\s]", "", c.vi or "").strip().lower()
            start_diff = abs(c.start - prev.start)
            post_gap = c.start - prev.end
            is_dramatic_post_repeat = (
                p_norm == c_norm
                and post_gap >= 80
                and len(prev.vi or "") <= 22
                and (post_gap >= 150 or (prev.vi or "").strip() != (c.vi or "").strip())
            )
            is_post_dup = not is_dramatic_post_repeat and (
                (p_norm == c_norm and c.start <= prev.end + 500)
                or (
                    start_diff <= 800
                    and len(p_norm) >= 15
                    and len(c_norm) >= 15
                    and (c_norm.startswith(p_norm) or p_norm.startswith(c_norm))
                )
            )
            if is_post_dup:
                prev.end = max(prev.end, c.end)
                if len(c.vi or "") > len(prev.vi or ""):
                    prev.vi = c.vi
                continue
        deduped_merged.append(c)

    # Giai đoạn 3: Duration Extension (capped before non-overlapping next_cue.start - 40ms)
    for idx_m, c in enumerate(deduped_merged):
        txt = (c.vi or "").strip()
        is_dual = "\n" in txt
        base_dur = min_dual_duration_ms if is_dual else min_duration_ms
        needed = max(base_dur, min(6000, int(len(txt) * 42)))
        target_end = max(c.end, c.start + needed)
        if idx_m + 1 < len(deduped_merged):
            next_start = deduped_merged[idx_m + 1].start
            if c.end <= next_start - 40:
                target_end = min(target_end, max(c.end, next_start - 40))
        c.end = target_end

    # Giai đoạn 4: Strict Zero-Overlap Pass (Tuyệt đối không chồng lấn thời gian)
    k = 0
    while k < len(deduped_merged) - 1:
        c1 = deduped_merged[k]
        c2 = deduped_merged[k + 1]

        if c2.start < c1.start:
            c2.start = c1.start

        if c1.end > c2.start - 40:
            safe_end = c2.start - 40
            if safe_end >= c1.start + 600:
                c1.end = safe_end
                k += 1
            else:
                lines1 = [l.strip() for l in (c1.vi or "").split("\n") if l.strip()]
                lines2 = [l.strip() for l in (c2.vi or "").split("\n") if l.strip()]
                if len(lines1) == 1 and len(lines2) == 1:
                    c1.vi = f"- {lines1[0].lstrip('-—– ')}\n- {lines2[0].lstrip('-—– ')}"
                    c1.start = min(c1.start, c2.start)
                    c1.end = max(c1.end, c2.end)
                    deduped_merged.pop(k + 1)
                else:
                    # Cap forward shift on c2 so extended Dual Dialogue c1 cannot push c2 seconds out of audio sync
                    max_shift_ms = 400
                    target_c2_start = max(c1.start + 340, min(c1.end + 40, c2.start + max_shift_ms))
                    c1.end = target_c2_start - 40
                    c2.start = target_c2_start
                    c2.end = max(c2.end, c2.start + min_duration_ms)
                    k += 1
        else:
            inter_gap = c2.start - c1.end
            if 0 < inter_gap < 40:
                c1.end = max(c1.start + 400, c2.start - 40)
            k += 1

    return deduped_merged



def wrap_subtitle_line(
    text: str,
    wrap_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    delimiter: str = "\n"
) -> str:
    """
    Wraps text into at most 2 lines ONLY if wrap_chars/max_chars is provided.
    If neither is provided, returns text as-is on exactly ONE line.
    """
    text = text.strip()
    limit = wrap_chars or max_chars
    if not limit or len(text) <= limit:
        return text

    words = text.split()
    if len(words) <= 1:
        return text

    total_len = len(text)
    half_len = total_len / 2.0

    best_idx = -1
    best_penalty = float("inf")
    running_len = 0

    for i in range(len(words) - 1):
        running_len += len(words[i])
        prev_word = words[i]
        diff = abs(running_len - half_len)
        penalty = diff

        if prev_word and prev_word[-1] in {",", ".", "!", "?", "…", ";", ":", "-", "—"}:
            penalty -= 35  # strong preference for punctuation
        elif words[i + 1].lower() in CONJUNCTIONS:
            penalty -= 12  # preference for conjunction

        if penalty < best_penalty:
            best_penalty = penalty
            best_idx = i

        running_len += 1  # for space

    if best_idx >= 0:
        line1 = " ".join(words[:best_idx + 1])
        line2 = " ".join(words[best_idx + 1:])
        return f"{line1}{delimiter}{line2}"

    return text


_VI_CONTINUATION_LOWERCASE_WORDS = {
    "anh", "em", "cậu", "tớ", "tôi", "ta", "ngươi", "cô", "chú", "bác", "ông", "bà",
    "chị", "hắn", "nó", "họ", "chúng", "mình", "mày", "tao", "ngài", "người", "vị",
    "kẻ", "nhưng", "mà", "và", "vì", "nếu", "khi", "vậy", "thế", "không", "đừng",
    "hãy", "mau", "đi", "ở", "tại", "có", "là", "đây", "đó", "đã", "đang", "sẽ",
    "chỉ", "cũng", "vẫn", "thật", "sao", "ai", "cái", "con", "những", "các",
}


def _lowercase_continuation_preserving_proper_nouns(part2: str, after_comma: bool = False) -> str:
    """Lowercases continuation start unless it begins with a capitalized proper noun (e.g. Subaru, Fran)."""
    if not part2:
        return part2
    m_word = re.match(r"^([^\W\d_]+)", part2, flags=re.UNICODE)
    if not m_word:
        return part2
    first_word = m_word.group(1)
    if first_word[0].isupper():
        if first_word.lower() not in _VI_CONTINUATION_LOWERCASE_WORDS and (after_comma or len(first_word) >= 2):
            return part2
    return part2[0].lower() + part2[1:]


def split_sentence_across_cues(sentence: str, num_cues: int = 2) -> List[str]:
    """
    Splits a single sentence across multiple cues at natural phrase boundaries.
    - Preceding fragment ends with ',' or '...'.
    - Continuation cue starts in lowercase (unless starting with a proper noun).
    - Preserves full sentence content without moving content beyond split.
    """
    sentence = clean_and_normalize_vietnamese(sentence)
    if num_cues <= 1 or not sentence:
        return [sentence]

    # Look for natural split candidates (comma or ellipsis) near sentence / num_cues
    target_pos = len(sentence) / float(num_cues)

    # Find candidate punctuation marks
    candidates = []
    for m in re.finditer(r"([,\.]\.\.|\,)\s*", sentence):
        pos = m.end()
        diff = abs(pos - target_pos)
        candidates.append((diff, m.start(), m.end(), m.group(1)))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, split_start, split_end, punct = candidates[0]
        part1 = sentence[:split_start].strip() + (punct if punct.startswith(".") else ",")
        part2 = sentence[split_end:].strip()
        if part2:
            part2 = _lowercase_continuation_preserving_proper_nouns(
                part2, after_comma=not punct.startswith(".")
            )
        return [part1, part2]

    # Fallback: split at nearest space
    words = sentence.split()
    mid = len(words) // 2
    part1 = " ".join(words[:mid]) + ","
    part2 = " ".join(words[mid:])
    if part2:
        part2 = _lowercase_continuation_preserving_proper_nouns(part2, after_comma=True)
    return [part1, part2]


def write_srt_subtitles(
    cues: List[Cue],
    output_path: Path,
    wrap_chars: Optional[int] = None,
    optimize_playback: bool = False,
    smart_wrap: bool = False,
    max_line_chars: int = 72,
    sound_cues: str = "drop",
) -> Path:
    """
    Writes cues to standard SRT format matching ReZERO_S04E19_Vietsub_Chuan.srt.
    - UTF-8 without BOM, LF line endings
    - 1-based indexing
    - Exactly one blank line between cues
    - Dual dialogue lines (- Line 1\n- Line 2) preserved when speakers speak simultaneously
    - Smart Semantic Line Wrapping: automatically wraps long lines (> 72 chars) into 2 balanced lines
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    if optimize_playback:
        processed_cues = optimize_cues_for_playback(cues, sound_cues=sound_cues)
        from subtrans.linter import auto_fix_cues
        processed_cues, _ = auto_fix_cues(processed_cues)
    else:
        processed_cues = [
            c for c in cues if not getattr(c, "is_dropped_sound", False)
        ]

    for idx, cue in enumerate(processed_cues, start=1):
        t_start = format_timestamp_srt(cue.start)
        t_end = format_timestamp_srt(cue.end)

        # Chốt chặn Item 4a: KHÔNG ghi [Chưa dịch] vào SRT; cue lỗi dùng tạm text EN tham chiếu
        raw_vi = (cue.vi or "").strip()
        if not raw_vi or raw_vi in ["[Chưa dịch]", "(Chưa dịch)", "[Untranslated]"]:
            raw_vi = (cue.en or cue.ja or "").strip()
            cue.used_en_fallback = True

        clean_text = clean_and_normalize_vietnamese(raw_vi, preserve_dual=True, sound_cues=sound_cues)
        if not clean_text:
            clean_text = (cue.en or cue.ja or "").strip()

        if wrap_chars and "\n" not in clean_text:
            clean_text = wrap_subtitle_line(clean_text, wrap_chars=wrap_chars)
        elif smart_wrap and "\n" not in clean_text and len(clean_text) > max_line_chars:
            clean_text = wrap_subtitle_line(clean_text, max_chars=max_line_chars)

        block = f"{idx}\n{t_start} --> {t_end}\n{clean_text}"
        blocks.append(block)

    content = "\n\n".join(blocks) + "\n"
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    return output_path


def write_subtitles(
    cues: List[Cue],
    original_ja_ssa: pysubs2.SSAFile,
    output_path: Path,
    line_max_chars: Optional[int] = None,
    output_format: Optional[str] = None,
    wrap_chars: Optional[int] = None,
    optimize_playback: bool = True,
    smart_wrap: bool = True,
    max_line_chars: int = 72,
    sound_cues: str = "drop",
) -> Path:
    """
    Main subtitle writer entrypoint.
    Defaults to SRT unless ASS format is explicitly requested.
    Optimizes playback by default (dual dialogue lines, chained fragments merging, min duration).
    """
    out_format = output_format or output_path.suffix.lstrip(".").lower()
    if not out_format or out_format not in ("ass", "srt"):
        out_format = "srt"

    effective_wrap = wrap_chars if wrap_chars is not None else line_max_chars

    if out_format == "srt":
        if output_path.suffix.lower() != ".srt":
            output_path = output_path.with_suffix(".srt")
        return write_srt_subtitles(
            cues,
            output_path,
            wrap_chars=effective_wrap,
            optimize_playback=optimize_playback,
            smart_wrap=smart_wrap,
            max_line_chars=max_line_chars,
            sound_cues=sound_cues,
        )

    # ASS format fallback (preserves dialogue styles AND original typesetting/sign events)
    if optimize_playback:
        processed_cues = optimize_cues_for_playback(cues, sound_cues=sound_cues)
        from subtrans.linter import auto_fix_cues
        processed_cues, _ = auto_fix_cues(processed_cues)
    else:
        processed_cues = [
            c for c in cues if not getattr(c, "is_dropped_sound", False) and not getattr(c, "is_skipped", False)
        ]
    ssa = copy.deepcopy(original_ja_ssa)
    ssa.events.clear()
    for cue in processed_cues:
        raw_vi = (cue.vi or "").strip()
        if not raw_vi or raw_vi in ["[Chưa dịch]", "(Chưa dịch)", "[Untranslated]"]:
            raw_vi = cue.en or cue.ja or ""
        clean_text = clean_and_normalize_vietnamese(raw_vi, preserve_dual=True, sound_cues=sound_cues)
        if effective_wrap and "\n" not in clean_text:
            clean_text = wrap_subtitle_line(clean_text, wrap_chars=effective_wrap)
        elif smart_wrap and "\n" not in clean_text and len(clean_text) > max_line_chars:
            clean_text = wrap_subtitle_line(clean_text, max_chars=max_line_chars)
        clean_text = clean_text.replace("\n", r"\N")
        evt = pysubs2.SSAEvent(
            start=cue.start,
            end=cue.end,
            text=clean_text,
            style=cue.style or "Default",
        )
        ssa.events.append(evt)

    # Preserve skipped studio typesetting / sign events (not dropped sound cues) in ASS output
    for cue in cues:
        if getattr(cue, "is_skipped", False) and not getattr(cue, "is_dropped_sound", False):
            sign_raw = (cue.raw_text or cue.ja or "").strip()
            if sign_raw:
                ssa.events.append(
                    pysubs2.SSAEvent(
                        start=cue.start,
                        end=cue.end,
                        text=sign_raw,
                        style=cue.style or "Default",
                    )
                )
    ssa.events.sort(key=lambda e: (e.start, e.end))

    if output_path.suffix.lower() != ".ass":
        output_path = output_path.with_suffix(".ass")
    ssa.save(str(output_path), format_="ass")
    return output_path
