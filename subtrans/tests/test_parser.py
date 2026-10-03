# -*- coding: utf-8 -*-
"""
subtrans/tests/test_parser.py
Unit tests for Stage 1 parser and aligner.
"""

import os
import pytest
import pysubs2
from subtrans.parser import (
    clean_subtitle_text,
    has_typesetting,
    should_skip_cue,
    parse_and_align_subtitles
)


def test_clean_subtitle_text():
    raw = r"{\an8\blur1.5}こんにちは、\N世界！{\b1}テスト\hです。"
    expected = "こんにちは、 世界！テスト です。"
    assert clean_subtitle_text(raw) == expected


def test_has_typesetting():
    assert has_typesetting(r"{\pos(100,200)}Title") is True
    assert has_typesetting(r"{\move(0,0,100,100)}Credit") is True
    assert has_typesetting(r"{\an8}Normal dialogue") is False


def test_should_skip_cue():
    assert should_skip_cue("OP", "Opening song lyrics") is True
    assert should_skip_cue("ED_Theme", "Ending lyrics") is True
    assert should_skip_cue("Sign", "Sign translation") is True
    assert should_skip_cue("Default", r"{\pos(192,108)}Typeset text") is True
    assert should_skip_cue("Default", "Regular dialogue") is False
    assert should_skip_cue("MainDialogue", "I love you.") is False


def test_alignment_1_to_1(tmp_path):
    ja_file = tmp_path / "ja.ass"
    en_file = tmp_path / "en.ass"

    ja_ssa = pysubs2.SSAFile()
    ja_ssa.events.append(pysubs2.SSAEvent(start=1000, end=3000, text="おはよう", style="Default"))
    ja_ssa.save(str(ja_file))

    en_ssa = pysubs2.SSAFile()
    en_ssa.events.append(pysubs2.SSAEvent(start=1100, end=2900, text="Good morning.", style="Default"))
    en_ssa.save(str(en_file))

    cues, _, _ = parse_and_align_subtitles(str(ja_file), str(en_file))
    assert len(cues) == 1
    assert cues[0].ja == "おはよう"
    assert cues[0].en == "Good morning."


def test_alignment_1_to_2(tmp_path):
    # One JP cue spans two short EN cues
    ja_file = tmp_path / "ja.ass"
    en_file = tmp_path / "en.ass"

    ja_ssa = pysubs2.SSAFile()
    ja_ssa.events.append(pysubs2.SSAEvent(start=1000, end=5000, text="待って！行かないで！", style="Default"))
    ja_ssa.save(str(ja_file))

    en_ssa = pysubs2.SSAFile()
    en_ssa.events.append(pysubs2.SSAEvent(start=1000, end=2500, text="Wait!", style="Default"))
    en_ssa.events.append(pysubs2.SSAEvent(start=2600, end=4800, text="Don't go!", style="Default"))
    en_ssa.save(str(en_file))

    cues, _, _ = parse_and_align_subtitles(str(ja_file), str(en_file))
    assert len(cues) == 1
    assert cues[0].en == "Wait! Don't go!"


def test_alignment_2_to_1(tmp_path):
    # Two short JP cues fall inside one long EN cue
    ja_file = tmp_path / "ja.ass"
    en_file = tmp_path / "en.ass"

    ja_ssa = pysubs2.SSAFile()
    ja_ssa.events.append(pysubs2.SSAEvent(start=1000, end=2500, text="待って！", style="Default"))
    ja_ssa.events.append(pysubs2.SSAEvent(start=2600, end=4500, text="行かないで！", style="Default"))
    ja_ssa.save(str(ja_file))

    en_ssa = pysubs2.SSAFile()
    en_ssa.events.append(pysubs2.SSAEvent(start=1000, end=4500, text="Wait, don't leave me!", style="Default"))
    en_ssa.save(str(en_file))

    cues, _, _ = parse_and_align_subtitles(str(ja_file), str(en_file))
    assert len(cues) == 2
    # Both cues get the EN sentence because both overlap by >= 50% of the EN duration
    # overlap 1: 1500ms out of 3500ms (42.8% < 50% -> not matched)
    # overlap 2: 1900ms out of 3500ms (54.2% >= 50% -> matched)
    assert cues[1].en == "Wait, don't leave me!"
