# -*- coding: utf-8 -*-
"""
subtrans/tests/test_sample_fixture.py
Tests verifying ADDENDUM 6 specifications using ReZERO_S04E19_Vietsub_Chuan.srt as reference fixture.
"""

from pathlib import Path
import pysubs2
from subtrans.models import Cue
from subtrans.writer import (
    format_timestamp_srt,
    clean_and_normalize_vietnamese,
    split_sentence_across_cues,
    write_srt_subtitles,
)

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "samples" / "ReZERO_S04E19_Vietsub_Chuan.srt"


def test_sample_exists():
    assert SAMPLE_PATH.exists(), f"Reference sample not found at {SAMPLE_PATH}"


def test_parser_writer_roundtrip(tmp_path):
    """
    Parser/writer round-trip test:
    Parse the sample and write it back; cue count, timestamps, and text must be identical.
    """
    ssa = pysubs2.load(str(SAMPLE_PATH))
    cues = [
        Cue(
            id=idx,
            start=e.start,
            end=e.end,
            style=e.style,
            ja="",
            vi=e.text,
            raw_text=e.text,
        )
        for idx, e in enumerate(ssa)
    ]

    out_file = tmp_path / "roundtrip.srt"
    write_srt_subtitles(cues, out_file)

    # Verify identical structure
    written_ssa = pysubs2.load(str(out_file))
    assert len(written_ssa) == len(ssa)
    for orig_event, written_event in zip(ssa, written_ssa):
        assert written_event.start == orig_event.start
        assert written_event.end == orig_event.end
        assert written_event.text == orig_event.text


def test_sample_stats():
    """
    Stats test:
    413 cues, exactly one text line per cue, no tags, maximum line length 107.
    """
    ssa = pysubs2.load(str(SAMPLE_PATH))
    assert len(ssa) == 413, f"Expected 413 cues, got {len(ssa)}"

    max_len = 0
    for idx, event in enumerate(ssa):
        text = event.text
        # Exactly one text line per cue (no embedded linebreaks)
        assert "\n" not in text, f"Cue {idx+1} contains newline: {repr(text)}"
        assert r"\N" not in text, f"Cue {idx+1} contains ASS newline: {repr(text)}"

        # No HTML or ASS tags
        assert "<" not in text and ">" not in text, f"Cue {idx+1} contains HTML tag: {text}"
        assert "{" not in text and "}" not in text, f"Cue {idx+1} contains ASS tag: {text}"

        line_len = len(text)
        if line_len > max_len:
            max_len = line_len

    assert max_len <= 107, f"Expected maximum line length <= 107, got {max_len}"


def test_continuation_splitting():
    """
    Continuation test:
    Cues 4-5 and 13-14 are each one sentence split across two cues.
    The splitting logic must produce this shape from a combined sentence.
    """
    ssa = pysubs2.load(str(SAMPLE_PATH))
    cue4_text = ssa[3].text   # "Hỡi kẻ đã chạm tới đỉnh tháp,"
    cue5_text = ssa[4].text   # "vị khách thỉnh cầu toàn năng đã đặt chân lên tầng một..."
    combined_4_5 = f"{cue4_text} {cue5_text}"

    split_res = split_sentence_across_cues(combined_4_5, num_cues=2)
    assert len(split_res) == 2
    # Preceding fragment ends with ',' or '...'
    assert split_res[0].endswith(",") or split_res[0].endswith("...")
    # Continuation starts with lowercase
    assert split_res[1][0].islower()
    assert split_res[0] == cue4_text
    assert split_res[1] == cue5_text

    cue13_text = ssa[12].text  # "Vị vua của 400 năm trước,"
    cue14_text = ssa[13].text  # "người được truyền tụng là đã lập minh ước cùng Thần Long Volcanica?"
    combined_13_14 = f"{cue13_text} {cue14_text}"

    split_res_13_14 = split_sentence_across_cues(combined_13_14, num_cues=2)
    assert len(split_res_13_14) == 2
    assert split_res_13_14[0].endswith(",")
    assert split_res_13_14[1][0].islower()
    assert split_res_13_14[0] == cue13_text
    assert split_res_13_14[1] == cue14_text


def test_clean_and_normalize_vietnamese():
    """Verifies that cleanup removes HTML/ASS tags, dashes, music symbols, brackets, and normalizes punctuation."""
    raw = "- <i>Hả?!</i> Thật không thể tin được... ♪ [tiếng thở dài] (cười) --"
    cleaned = clean_and_normalize_vietnamese(raw)
    assert "<i>" not in cleaned and "</i>" not in cleaned
    assert not cleaned.startswith("-")
    assert "♪" not in cleaned
    assert "[tiếng thở dài]" not in cleaned
    assert "(cười)" not in cleaned
    assert cleaned.endswith("—")  # Interrupted dash replaced with em-dash
