# -*- coding: utf-8 -*-
"""
subtrans/tests/test_parser.py
Unit tests for Stage 1 parser and aligner.
"""

import os
import time
import pytest
import pysubs2
from subtrans.parser import (
    clean_subtitle_text,
    compute_alignment_match_rate,
    detect_auto_offset,
    has_typesetting,
    should_skip_cue,
    parse_and_align_subtitles,
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


def test_skip_styles_boundary_regex():
    from subtrans.parser import should_skip_cue
    # Subtitle chứa 'title' nhưng không được phép bị skip
    assert should_skip_cue("Subtitle", "Regular text") is False
    assert should_skip_cue("Subtitles", "Regular text") is False
    assert should_skip_cue("MainDialogue", "Regular text") is False
    assert should_skip_cue("Default", "Regular text") is False

    # Các style cần skip
    assert should_skip_cue("OP", "Music lyrics") is True
    assert should_skip_cue("ED", "Music lyrics") is True
    assert should_skip_cue("ED_Theme", "Music lyrics") is True
    assert should_skip_cue("Sign", "On-screen text") is True
    assert should_skip_cue("Song", "Insert song") is True
    assert should_skip_cue("Karaoke", "Sing along") is True
    assert should_skip_cue("Title_Card", "Episode title") is True
    assert should_skip_cue("Credit", "Staff credit") is True


def test_snap_whisper_cues_to_en_1_to_1():
    from subtrans.models import Cue
    from subtrans.parser import snap_whisper_cues_to_en

    # Whisper cue có sai số bắt đầu trễ 200ms và kết thúc sớm 150ms
    cues = [
        Cue(id=0, start=1200, end=2850, ja="おはよう", style="Default", is_skipped=False)
    ]
    en_events = [
        {"id": 0, "start": 1000, "end": 3000, "text": "Good morning.", "duration": 2000}
    ]

    snapped = snap_whisper_cues_to_en(cues, en_events)
    assert snapped == 1
    assert cues[0].start == 1000
    assert cues[0].end == 3000
    assert cues[0].en == "Good morning."


def test_snap_whisper_cues_to_en_2_to_1_partition():
    from subtrans.models import Cue
    from subtrans.parser import snap_whisper_cues_to_en

    # Hai câu Whisper tách rời nhưng tương ứng 1 câu EN dài phòng thu
    cues = [
        Cue(id=0, start=1050, end=2400, ja="待って！", style="Default", is_skipped=False),
        Cue(id=1, start=2450, end=4800, ja="行かないで！", style="Default", is_skipped=False),
    ]
    en_events = [
        {"id": 0, "start": 1000, "end": 5000, "text": "Wait, don't leave me!", "duration": 4000}
    ]

    snapped = snap_whisper_cues_to_en(cues, en_events, min_gap_ms=50)
    assert snapped == 2
    # Cả hai cue phải được định vị trong khoảng [1000, 5000], có gap >= 50ms và không đè nhau
    assert cues[0].start == 1000
    assert cues[0].end < cues[1].start
    assert cues[1].start - cues[0].end >= 50
    assert cues[1].end == 5000


def test_snap_whisper_cues_to_en_ignores_typesetting_and_signs():
    from subtrans.models import Cue
    from subtrans.parser import snap_whisper_cues_to_en
    import pysubs2

    cues = [
        Cue(id=0, start=5050, end=6450, ja="見ろよ", style="Default", is_skipped=False)
    ]
    # EN events gồm: 1 sign typeset kéo dài 15 giây và 1 thoại thực tế
    en_events = [
        pysubs2.SSAEvent(start=5000, end=20000, text=r"{\pos(192,108)}Stats Screen", style="Sign"),
        pysubs2.SSAEvent(start=5000, end=6500, text="Look at that.", style="Default"),
    ]

    snapped = snap_whisper_cues_to_en(cues, en_events)
    assert snapped == 1
    # Phải bắt theo câu thoại Default [5000, 6500], không được bắt theo Sign [5000, 20000]
    assert cues[0].start == 5000
    assert cues[0].end == 6500
    assert cues[0].en == "Look at that."


def test_parse_and_align_whisper_auto_detection(tmp_path):
    # Test tự động kích hoạt Auto-Align khi tên file chứa 'whisper'
    whisper_file = tmp_path / "sample.whisper_ja.srt"
    en_file = tmp_path / "sample.en.ass"

    w_ssa = pysubs2.SSAFile()
    w_ssa.events.append(pysubs2.SSAEvent(start=1250, end=2750, text="こんにちは"))
    w_ssa.save(str(whisper_file))

    en_ssa = pysubs2.SSAFile()
    en_ssa.events.append(pysubs2.SSAEvent(start=1000, end=3000, text="Hello there.", style="Default"))
    en_ssa.save(str(en_file))

    # Với file whisper, timestamps tự động được snap theo EN
    cues, _, _ = parse_and_align_subtitles(str(whisper_file), str(en_file), auto_align_whisper_to_en=True)
    assert len(cues) == 1
    assert cues[0].start == 1000
    assert cues[0].end == 3000

    # Với file studio JA thông thường (không có 'whisper'), start time giữ nguyên gốc
    studio_ja_file = tmp_path / "sample.ja.srt"
    w_ssa.save(str(studio_ja_file))
    cues_studio, _, _ = parse_and_align_subtitles(str(studio_ja_file), str(en_file), auto_align_whisper_to_en=True)
    assert cues_studio[0].start == 1250  # Giữ nguyên mốc phòng thu gốc của tiếng Nhật


def test_parse_and_align_sorts_unsorted_en_ssa_before_dtw(tmp_path):
    """Verify unsorted EN ASS events are sorted before monotonic DTW whisper snap."""
    ja_p = tmp_path / "sample.whisper_ja.srt"
    en_p = tmp_path / "sample.en.ass"

    ja_ssa = pysubs2.SSAFile()
    ja_ssa.events.append(pysubs2.SSAEvent(start=1200, end=2800, text="おはよう"))
    ja_ssa.events.append(pysubs2.SSAEvent(start=5200, end=6800, text="おやすみ"))
    ja_ssa.save(str(ja_p))

    en_ssa = pysubs2.SSAFile()
    en_ssa.events.append(pysubs2.SSAEvent(start=5000, end=7000, text="Good night.", style="Default"))
    en_ssa.events.append(pysubs2.SSAEvent(start=1000, end=3000, text="Good morning.", style="Default"))
    en_ssa.save(str(en_p))

    cues, _, _ = parse_and_align_subtitles(str(ja_p), str(en_p), auto_align_whisper_to_en=True)
    assert (cues[0].start, cues[0].end) == (1000, 3000)
    assert (cues[1].start, cues[1].end) == (5000, 7000)


def test_parse_and_align_skips_en_sign_and_op_styles(tmp_path):
    """Verify Sign/OP EN ASS events are skipped in both 1:1 alignment and Whisper DTW."""
    ja_p = tmp_path / "sign_leak.ja.ass"
    en_p = tmp_path / "sign_leak.en.ass"

    ja_ssa = pysubs2.SSAFile()
    ja_ssa.events.append(pysubs2.SSAEvent(start=1200, end=2800, text="こんにちは", style="Default"))
    ja_ssa.save(str(ja_p))

    en_ssa = pysubs2.SSAFile()
    en_ssa.events.append(pysubs2.SSAEvent(start=1000, end=5000, text=r"{\pos(100,200)}CHAPTER 1", style="Sign"))
    en_ssa.events.append(pysubs2.SSAEvent(start=1000, end=4500, text="Opening Song Lyrics", style="OP"))
    en_ssa.events.append(pysubs2.SSAEvent(start=1200, end=2800, text="Hello!", style="Default"))
    en_ssa.save(str(en_p))

    cues, _, _ = parse_and_align_subtitles(str(ja_p), str(en_p))
    assert "CHAPTER 1" not in (cues[0].en or "")
    assert "Opening Song Lyrics" not in (cues[0].en or "")
    assert cues[0].en == "Hello!"


def test_snap_whisper_cues_to_en_unsorted_dicts_and_events():
    """Verify snap_whisper_cues_to_en sorts unsorted dict (without id) and SSAEvent EN inputs."""
    from subtrans.models import Cue
    from subtrans.parser import snap_whisper_cues_to_en

    cues_dict = [
        Cue(id=1, start=5200, end=6800, ja="おやすみ", style="Default", is_skipped=False),
        Cue(id=0, start=1200, end=2800, ja="おはよう", style="Default", is_skipped=False),
    ]
    en_dicts = [
        {"start": 5000, "end": 7000, "text": "Good night.", "style": "Default"},
        {"start": 1000, "end": 5000, "text": r"{\pos(10,10)}SIGN", "style": "Sign"},
        {"start": 1000, "end": 3000, "text": "Good morning.", "style": "Default"},
    ]
    assert snap_whisper_cues_to_en(cues_dict, en_dicts) == 2
    assert (cues_dict[0].start, cues_dict[0].end, cues_dict[0].en) == (1000, 3000, "Good morning.")
    assert (cues_dict[1].start, cues_dict[1].end, cues_dict[1].en) == (5000, 7000, "Good night.")

    cues_ev = [
        Cue(id=0, start=1200, end=2800, ja="おはよう", style="Default", is_skipped=False),
        Cue(id=1, start=5200, end=6800, ja="おやすみ", style="Default", is_skipped=False),
    ]
    en_events = [
        pysubs2.SSAEvent(start=5000, end=7000, text="Good night.", style="Default"),
        pysubs2.SSAEvent(start=1000, end=3000, text="Good morning.", style="Default"),
    ]
    assert snap_whisper_cues_to_en(cues_ev, en_events) == 2
    assert (cues_ev[0].start, cues_ev[0].end) == (1000, 3000)
    assert (cues_ev[1].start, cues_ev[1].end) == (5000, 7000)


def test_compute_alignment_match_rate_unsorted_and_overlapping():
    """Verify O(N+M) match rate handles unsorted events, zero-duration cues, and long overlapping EN cues."""
    ja_events = [
        pysubs2.SSAEvent(start=10000, end=12000, text="c"),
        pysubs2.SSAEvent(start=1000, end=3000, text="a"),
        pysubs2.SSAEvent(start=4000, end=4000, text="zero"),
        pysubs2.SSAEvent(start=5000, end=7000, text="b"),
    ]
    en_events = [
        pysubs2.SSAEvent(start=0, end=100000, text="long sign"),
        pysubs2.SSAEvent(start=5100, end=6900, text="b"),
        pysubs2.SSAEvent(start=1100, end=2900, text="a"),
    ]
    # 2 of 3 positive-duration JA cues match a/b; long [0, 100000] EN cue requires >= 50s overlap.
    assert compute_alignment_match_rate(ja_events, en_events, offset_ms=0) == pytest.approx(2 / 3)
    # Shifting by +4000ms moves [1000, 3000] onto [5100, 6900], leaving 1 of 3 matched.
    assert compute_alignment_match_rate(ja_events, en_events, offset_ms=4000) == pytest.approx(1 / 3)


def _write_synthetic_offset_pair(tmp_path, prefix: str, ja_specs, en_specs):
    """Write synthetic JA and EN SRT files for offset tests."""
    ja_p = tmp_path / f"{prefix}.ja.srt"
    en_p = tmp_path / f"{prefix}.en.srt"
    ja_ssa = pysubs2.SSAFile()
    en_ssa = pysubs2.SSAFile()
    for start, end, txt in ja_specs:
        ja_ssa.events.append(pysubs2.SSAEvent(start=start, end=end, text=txt))
    for start, end, txt in en_specs:
        en_ssa.events.append(pysubs2.SSAEvent(start=start, end=end, text=txt))
    ja_ssa.save(str(ja_p))
    en_ssa.save(str(en_p))
    return str(ja_p), str(en_p)


def _make_nonuniform_ja_specs(n: int, q_mod: int = 5):
    """Generate realistic non-uniformly spaced JA dialogue cues to avoid periodic aliasing."""
    specs = []
    t = 10_000
    for i in range(n):
        dur = 900 + (i % 5) * 350
        specs.append((t, t + dur, f"台詞{i}{'？' if i % q_mod == 0 else ''}"))
        t += dur + 700 + (i % 7) * 410
    return specs


def test_detect_auto_offset_commercial_break_over_60s(tmp_path):
    """Verify detect_auto_offset detects both single >60s shifts and >60s commercial breaks."""
    # Case 1: Single global +95s commercial break offset.
    ja_single = _make_nonuniform_ja_specs(80, q_mod=5)
    en_single = [(s + 95_000, e + 95_000, f"Line {i}{'?' if i % 5 == 0 else ''}") for i, (s, e, _) in enumerate(ja_single)]
    ja_s, en_s = _write_synthetic_offset_pair(tmp_path, "single_95s", ja_single, en_single)
    off_ms, applied, _reason, stats = detect_auto_offset(ja_s, en_s)
    assert applied is True
    assert off_ms == 95_000
    assert stats["rate_after"] >= 0.99
    assert stats["rate_diff_points"] >= 50.0

    # Case 2: Asymmetric commercial break (first 25% at 0ms, remaining 75% shifted +125s).
    ja_pw = _make_nonuniform_ja_specs(120, q_mod=4)
    en_pw = []
    for i, (s, e, _) in enumerate(ja_pw):
        shift = 0 if i < 30 else 125_000
        en_pw.append((s + shift, e + shift, f"Line {i}{'?' if i % 4 == 0 else ''}"))
    ja_p, en_p = _write_synthetic_offset_pair(tmp_path, "pw_125s", ja_pw, en_pw)
    off_pw, applied_pw, _reason_pw, stats_pw = detect_auto_offset(ja_p, en_p)
    assert applied_pw is True
    assert off_pw == 125_000
    pw_segs = stats_pw.get("piecewise_segments")
    assert pw_segs and len(pw_segs) == 2
    assert abs(pw_segs[0]["offset_ms"]) <= 200
    assert abs(pw_segs[1]["offset_ms"] - 125_000) <= 200


def test_detect_auto_offset_piecewise_n1200_under_300ms(tmp_path):
    """Verify N=1200 piecewise commercial-break offset detection completes in < 300 ms."""
    N = 1200
    ja_specs = _make_nonuniform_ja_specs(N, q_mod=7)
    en_specs = []
    for i, (s, e, _) in enumerate(ja_specs):
        shift = 0 if i < 600 else 110_000
        en_specs.append((s + shift, e + shift, f"Line {i}{'?' if i % 7 == 0 else ''}"))
    ja_p, en_p = _write_synthetic_offset_pair(tmp_path, "n1200_pw", ja_specs, en_specs)
    # Pre-warm newly written temp files so NTFS creation flush does not skew cold I/O.
    with open(ja_p, "rb") as f:
        f.read()
    with open(en_p, "rb") as f:
        f.read()

    # Bound up to 3 trials to reject single-shot 6-worker OS scheduling spikes while preserving strict <300ms.
    wall_trials_ms = []
    cpu_trials_ms = []
    for _ in range(3):
        c0 = time.process_time()
        t0 = time.perf_counter()
        off_ms, applied, _reason, stats = detect_auto_offset(ja_p, en_p)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        cpu_ms = (time.process_time() - c0) * 1000.0
        wall_trials_ms.append(elapsed_ms)
        cpu_trials_ms.append(cpu_ms)
        assert applied is True
        assert off_ms == 110_000
        pw_segs = stats.get("piecewise_segments")
        assert pw_segs and len(pw_segs) == 2
        assert abs(pw_segs[0]["offset_ms"]) <= 200
        assert abs(pw_segs[1]["offset_ms"] - 110_000) <= 200
        if elapsed_ms < 300.0 and cpu_ms < 300.0:
            break

    best_wall_ms = min(wall_trials_ms)
    best_cpu_ms = min(cpu_trials_ms)
    assert best_wall_ms < 300.0, (
        f"detect_auto_offset N=1200 piecewise took min {best_wall_ms:.1f}ms "
        f"(wall={[round(x, 1) for x in wall_trials_ms]}, cpu={[round(x, 1) for x in cpu_trials_ms]}) >= 300ms"
    )
    assert best_cpu_ms < 300.0, f"detect_auto_offset N=1200 CPU time {best_cpu_ms:.1f}ms >= 300ms"


def test_snap_whisper_and_match_rate_defensive_edge_cases():
    """Verify null-text/null-style/comment/invalid-duration guards and zero-overlap boundary."""
    from subtrans.models import Cue
    from subtrans.parser import snap_whisper_cues_to_en

    cues_dict = [
        Cue(id=0, start=1100, end=2900, ja="おはよう", style="Default", is_skipped=False),
        Cue(id=1, start=1100, end=2900, ja="二行目", style="Default", is_skipped=False),
    ]
    bad_and_good_dicts = [
        {"start": 1000, "end": 3000, "text": None, "style": "Default"},
        {"start": 1000, "end": 3000, "text": r"{\an8}", "style": "Default"},
        {"start": 1000, "end": 3000, "text": "♪～", "style": "Default"},
        {"start": 3000, "end": 3000, "text": "Zero duration", "style": "Default"},
        {"id": None, "start": 1000, "end": 3000, "text": "Good morning.", "style": "Default"},
        {"start": 1000, "end": 3000, "text": "Second line.", "style": "Default"},
    ]
    assert snap_whisper_cues_to_en(cues_dict, bad_and_good_dicts) == 2
    assert cues_dict[0].en == "Good morning."
    assert cues_dict[1].en == "Second line."

    cues_music = [Cue(id=0, start=1100, end=2900, ja="おはよう", style="Default", is_skipped=False)]
    assert snap_whisper_cues_to_en(cues_music, [{"start": 1000, "end": 3000, "text": "♪～", "style": "Default"}]) == 0
    assert cues_music[0].en is None

    cues_ev = [Cue(id=0, start=1100, end=2900, ja="おはよう", style="Default", is_skipped=False)]
    comment_ev = pysubs2.SSAEvent(start=1000, end=3000, text="COMMENT", style="Default", type="Comment")
    null_style_ev = pysubs2.SSAEvent(start=1000, end=3000, text="Valid line", style="Default")
    null_style_ev.style = None  # type: ignore[assignment]
    assert snap_whisper_cues_to_en(cues_ev, [comment_ev, null_style_ev]) == 1
    assert cues_ev[0].en == "Valid line"

    # Touching endpoints (overlap == 0) must not count even when overlap_threshold == 0.0.
    ja_touch = [
        pysubs2.SSAEvent(start=1000, end=5000, text="span"),
        pysubs2.SSAEvent(start=2000, end=4000, text="touch"),
    ]
    en_touch = [pysubs2.SSAEvent(start=4000, end=4500, text="next")]
    assert compute_alignment_match_rate(ja_touch, en_touch, offset_ms=0, overlap_threshold=0.0) == pytest.approx(0.5)
    assert compute_alignment_match_rate([ja_touch[1]], en_touch, offset_ms=0, overlap_threshold=0.0) == 0.0


def test_detect_auto_offset_nonzero_commercial_break_within_8s(tmp_path):
    """Verify +100s -> +108s (60/40) non-zero commercial break recovers both segments cleanly."""
    ja_specs = _make_nonuniform_ja_specs(200, q_mod=5)
    en_specs = []
    for i, (s, e, _) in enumerate(ja_specs):
        shift = 100_000 if i < 120 else 108_000
        en_specs.append((s + shift, e + shift, f"Line {i}{'?' if i % 5 == 0 else ''}"))
    ja_p, en_p = _write_synthetic_offset_pair(tmp_path, "pw_100_108", ja_specs, en_specs)
    off_ms, applied, _reason, stats = detect_auto_offset(ja_p, en_p)
    assert applied is True
    assert off_ms in (100_000, 108_000)
    assert stats["rate_after"] >= 0.99
    pw_segs = stats.get("piecewise_segments")
    assert pw_segs and len(pw_segs) == 2
    assert abs(pw_segs[0]["offset_ms"] - 100_000) <= 200
    assert abs(pw_segs[1]["offset_ms"] - 108_000) <= 200


@pytest.mark.parametrize("shift_ms", [45_000, 95_000])
def test_detect_auto_offset_accepts_quarter_to_half_duration_ratio_gt_10s(tmp_path, shift_ms):
    """Verify coarse +/-600s scan accepts 0.25 <= dur_ratio < 0.50 (0.40) for >10s shifts."""
    ja_specs = []
    en_specs = []
    t = 10_000
    for i in range(60):
        dur_en = 1000 + (i % 4) * 250
        dur_ja = int(round(dur_en * 2.5))  # min/max ratio = 0.40 (in [0.25, 0.50))
        q_mark_ja = "？" if i % 5 == 0 else ""
        q_mark_en = "?" if i % 5 == 0 else ""
        ja_specs.append((t, t + dur_ja, f"台詞{i}{q_mark_ja}"))
        en_specs.append((t + shift_ms, t + shift_ms + dur_en, f"Line {i}{q_mark_en}"))
        t += dur_ja + 700 + (i % 7) * 410

    ja_p, en_p = _write_synthetic_offset_pair(tmp_path, f"ratio40_{shift_ms}", ja_specs, en_specs)
    off_ms, applied, _reason, stats = detect_auto_offset(ja_p, en_p)
    assert applied is True
    assert off_ms == shift_ms
    assert stats["pairs_count"] >= 30


