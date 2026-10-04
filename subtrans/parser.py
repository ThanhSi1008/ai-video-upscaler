# -*- coding: utf-8 -*-
"""
subtrans/parser.py
Stage 1: Parse, clean, and align Japanese and English subtitles.
"""

import re
import logging
from typing import List, Tuple, Optional
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


def parse_and_align_subtitles(
    ja_path: str,
    en_path: Optional[str] = None,
    en_offset_ms: int = 0,
    ja_offset_ms: int = 0,
    skip_styles_regex: str = DEFAULT_SKIP_STYLES,
    overlap_threshold: float = 0.5,
    match_warning_pct: float = 0.8
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

        aligned_en = None
        if not is_skip and clean_ja:
            dialogue_cue_count += 1
            if en_cues:
                # Find all EN cues whose overlap is at least overlap_threshold * EN duration
                matching_en_texts = []
                for ec in en_cues:
                    # Calculate time overlap between JA and EN cue
                    overlap_start = max(event.start, ec["start"])
                    overlap_end = min(event.end, ec["end"])
                    overlap_duration = max(0, overlap_end - overlap_start)

                    if overlap_duration >= (ec["duration"] * overlap_threshold):
                        matching_en_texts.append(ec["text"])

                if matching_en_texts:
                    aligned_en = " ".join(matching_en_texts)
                    matched_en_count += 1

        cue = Cue(
            id=idx,
            start=event.start,
            end=event.end,
            style=event.style,
            ja=clean_ja,
            en=aligned_en,
            vi=raw_text if is_skip else None,  # Preserved unchanged if skipped
            backend_used="skipped" if is_skip else None,
            is_skipped=is_skip,
            raw_text=raw_text
        )
        cues.append(cue)

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
