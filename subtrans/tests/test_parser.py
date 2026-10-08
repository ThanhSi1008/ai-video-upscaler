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


def test_strip_speaker_prefix():
    from subtrans.parser import strip_speaker_prefix
    # Không prefix
    p, r = strip_speaker_prefix("こんにちは")
    assert p == "" and r == "こんにちは"

    # Ngoặc đơn tiếng Nhật
    p, r = strip_speaker_prefix("（珊瑚(さんご)）お気を確かに！")
    assert p == "珊瑚(さんご)" and r == "お気を確かに！"

    # Dấu gạch ngang kèm ngoặc người nói
    p, r = strip_speaker_prefix("-（絹秀）金淑妃(きんしゅくひ)")
    assert p == "絹秀" and r == "金淑妃(きんしゅくひ)"

    # Chỉ có dấu gạch ngang
    p, r = strip_speaker_prefix("- お前は誰だ")
    assert p == "" and r == "お前は誰だ"


def test_split_multi_speaker_cc_events():
    from subtrans.parser import split_multi_speaker_cc_events
    events = [
        # Multi-speaker có dấu gạch ngang
        pysubs2.SSAEvent(start=1000, end=3000, text=r"-（絹秀）金淑妃\N-（清佳）あっ…"),
        # Cue đơn thoại dài bẻ dòng thông thường (không phải multi-speaker)
        pysubs2.SSAEvent(start=3500, end=5000, text=r"果たして 玲琳様に\Nそのような ご体力があるのか"),
        # Cue âm thanh + thoại
        pysubs2.SSAEvent(start=5500, end=7000, text=r"（扉が開く音）\N（莉莉）慧月様"),
    ]
    split_res = split_multi_speaker_cc_events(events)
    # Event 1 tách thành 2, Event 2 giữ nguyên 1, Event 3 tách thành 2 -> Tổng cộng 5
    assert len(split_res) == 5
    assert split_res[0].text == "-（絹秀）金淑妃"
    assert split_res[1].text == "-（清佳）あっ…"
    assert split_res[0].start == 1000 and split_res[1].start == 1000
    assert split_res[2].text == r"果たして 玲琳様に\Nそのような ご体力があるのか"
    assert split_res[3].text == "（扉が開く音）"
    assert split_res[4].text == "（莉莉）慧月様"


def test_multi_speaker_pipeline_alignment(tmp_path):
    ja_file = tmp_path / "ja_cc.srt"
    en_file = tmp_path / "en.srt"

    ja_ssa = pysubs2.SSAFile()
    # Event 1: Seika gọi "Aunt!"
    ja_ssa.events.append(pysubs2.SSAEvent(start=857648, end=858691, text=r"{\an8}（清佳）叔母様！"))
    # Event 2: Multi-speaker: Kenshuu quát "Kin Shukuhi!" + Seika "Ah..."
    ja_ssa.events.append(pysubs2.SSAEvent(start=858774, end=860317, text=r"-（絹秀）金淑妃(きんしゅくひ)\N-（清佳）あっ…"))
    ja_ssa.save(str(ja_file))

    en_ssa = pysubs2.SSAFile()
    en_ssa.events.append(pysubs2.SSAEvent(start=857040, end=857960, text="Aunt!"))
    en_ssa.events.append(pysubs2.SSAEvent(start=858040, end=859380, text="Pure Consort Kin."))
    en_ssa.save(str(en_file))

    cues, _, _ = parse_and_align_subtitles(str(ja_file), str(en_file), sound_cues="drop")
    # CUE 1: Seika: "叔母様！"
    # CUE 2: Kenshuu: "金淑妃(きんしゅくひ)" -> KHÔNG BỊ BỎ QUÊN!
    # CUE 3: Seika: "あっ…" (bị drop bởi sound_cues="drop")
    active_dialogue_cues = [c for c in cues if not c.is_skipped]
    assert len(active_dialogue_cues) == 2
    assert active_dialogue_cues[0].ja == "叔母様！"
    assert active_dialogue_cues[1].ja == "金淑妃(きんしゅくひ)"
    assert "Pure Consort Kin." in active_dialogue_cues[1].en


def test_laughter_sound_cue_detection():
    from subtrans.parser import is_pure_sound_cue
    assert is_pure_sound_cue("ふはははは!") is True
    assert is_pure_sound_cue("フハハハ…") is True
    assert is_pure_sound_cue("ははは！") is True
    assert is_pure_sound_cue("ククク…") is True
    assert is_pure_sound_cue("うふふ～") is True
    assert is_pure_sound_cue("まさに偉大な死霊術師 素敵ですー!") is False


