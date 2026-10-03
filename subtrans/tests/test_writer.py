# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
import pysubs2
from subtrans.models import Cue
from subtrans.writer import wrap_subtitle_line, write_subtitles


def test_wrap_subtitle_line_short():
    text = "Xin chào các bạn!"
    assert wrap_subtitle_line(text, max_chars=42) == text


def test_wrap_subtitle_line_punctuation_split():
    # Long text with comma near the center
    text = "Nếu như ngày hôm đó cậu không đến, tôi nhất định sẽ không thể nào vượt qua được nỗi đau này."
    wrapped = wrap_subtitle_line(text, max_chars=42, delimiter="\\N")
    assert "\\N" in wrapped
    parts = wrapped.split("\\N")
    assert len(parts) == 2
    # Should break after comma
    assert parts[0].endswith(",")


def test_write_subtitles_preserves_styles(tmp_path):
    ssa = pysubs2.SSAFile()
    event = pysubs2.SSAEvent(start=1000, end=3000, text="元の日本語", style="Main")
    ssa.events.append(event)

    cues = [
        Cue(
            id=0,
            start=1000,
            end=3000,
            style="Main",
            ja="元の日本語",
            vi="Tiếng Việt dịch chuẩn",
            raw_text="元の日本語",
        )
    ]

    out_ass = tmp_path / "test.ass"
    written_path = write_subtitles(cues, ssa, out_ass)

    assert written_path.exists()
    loaded = pysubs2.load(str(written_path))
    assert len(loaded) == 1
    assert loaded[0].text == "Tiếng Việt dịch chuẩn"
    assert loaded[0].style == "Main"
