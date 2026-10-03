# -*- coding: utf-8 -*-
"""
subtrans/writer.py
Stage 4: Subtitle generation adhering strictly to ADDENDUM 6 specifications:
- Exact format fidelity to ReZERO_S04E19_Vietsub_Chuan.srt
- UTF-8 without BOM, LF (\n), 1-based indexing, HH:MM:SS,mmm --> HH:MM:SS,mmm
- Exactly ONE line per cue by default (no hard wrapping; up to 107 chars accepted)
- Wrap only when --wrap N is explicitly passed
- Strip HTML/ASS tags, speaker dashes, music symbols, bracketed sound cues
- Punctuation: '...' (three ASCII dots), '—' for interruptions, max two ! or ?
"""

import re
import copy
from pathlib import Path
from typing import List, Optional, Tuple
import pysubs2
from subtrans.models import Cue

CONJUNCTIONS = {"và", "nhưng", "thì", "mà", "là", "khi", "vì", "nếu", "để", "rồi", "hay", "hoặc"}


def format_timestamp_srt(ms: int) -> str:
    """Formats milliseconds as HH:MM:SS,mmm."""
    total_seconds, milliseconds = divmod(ms, 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def clean_and_normalize_vietnamese(text: str) -> str:
    """
    Cleans and normalizes Vietnamese subtitle text per ADDENDUM 6:
    - No HTML tags, no ASS tags
    - No speaker dashes at line start
    - No music symbols
    - No sound-effect or bracketed descriptions
    - Replaces unicode '…' with three ASCII dots '...'
    - Replaces interrupted speech trailing dashes with '—'
    - Never more than two consecutive '!' or '?'
    - Exactly one single line without linebreaks
    """
    if not text:
        return ""

    # 1. Strip HTML tags like <i>, </i>, <b>, <font color="...">, etc.
    text = re.sub(r"</?[a-zA-Z0-9]+[^>]*>", "", text)

    # 2. Strip ASS tags like {\an8}, {\pos(1,2)}, etc.
    text = re.sub(r"\{[^\}]*\}", "", text)

    # 3. Strip speaker dashes at start of line: "- ", "-- ", "— "
    text = re.sub(r"^\s*[-–—]\s*", "", text)

    # 4. Strip music symbols
    text = re.sub(r"[♪♫♬♩#]", "", text)

    # 5. Strip bracketed sound effects/descriptions: [cười], (tiếng nổ), [Sound], (laugh)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(
        r"\([^\)]*(?:tiếng|cười|khóc|thở|nói|nhạc|âm thanh|la|hét|laugh|music|sigh|gasp|screaming)[^\)]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

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

    return text


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
            # Lowercase the first letter of continuation cue if not capitalized name
            part2 = part2[0].lower() + part2[1:]
        return [part1, part2]

    # Fallback: split at nearest space
    words = sentence.split()
    mid = len(words) // 2
    part1 = " ".join(words[:mid]) + ","
    part2 = " ".join(words[mid:])
    if part2:
        part2 = part2[0].lower() + part2[1:]
    return [part1, part2]


def write_srt_subtitles(
    cues: List[Cue],
    output_path: Path,
    wrap_chars: Optional[int] = None,
) -> Path:
    """
    Writes cues to standard SRT format matching ReZERO_S04E19_Vietsub_Chuan.srt.
    - UTF-8 without BOM, LF line endings
    - 1-based indexing
    - Exactly one blank line between cues
    - Exactly one text line per cue by default
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []

    for idx, cue in enumerate(cues, start=1):
        t_start = format_timestamp_srt(cue.start)
        t_end = format_timestamp_srt(cue.end)
        raw_vi = cue.vi or cue.ja or ""
        clean_text = clean_and_normalize_vietnamese(raw_vi)

        if wrap_chars:
            clean_text = wrap_subtitle_line(clean_text, wrap_chars=wrap_chars)

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
) -> Path:
    """
    Main subtitle writer entrypoint.
    Defaults to SRT unless ASS format is explicitly requested.
    """
    out_format = output_format or output_path.suffix.lstrip(".").lower()
    if not out_format or out_format not in ("ass", "srt"):
        out_format = "srt"

    if out_format == "srt":
        if output_path.suffix.lower() != ".srt":
            output_path = output_path.with_suffix(".srt")
        return write_srt_subtitles(cues, output_path, wrap_chars=wrap_chars or line_max_chars if wrap_chars else None)

    # ASS format fallback (preserves styles and tags if needed)
    ssa = copy.deepcopy(original_ja_ssa)
    for cue in cues:
        if cue.id >= len(ssa):
            continue
        event = ssa[cue.id]
        if cue.is_skipped:
            event.text = cue.raw_text or cue.ja
            continue
        clean_text = clean_and_normalize_vietnamese(cue.vi or cue.ja)
        if wrap_chars:
            clean_text = wrap_subtitle_line(clean_text, wrap_chars=wrap_chars).replace("\n", r"\N")
        event.text = clean_text

    if output_path.suffix.lower() != ".ass":
        output_path = output_path.with_suffix(".ass")
    ssa.save(str(output_path), format_="ass")
    return output_path
