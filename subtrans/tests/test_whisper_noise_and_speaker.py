# -*- coding: utf-8 -*-
"""
subtrans/tests/test_whisper_noise_and_speaker.py
Unit tests for subtitle extraction optimizations, Whisper anti-hallucination,
orphan cue filtering, Pass 3 monotonicity isolation, and multi-speaker preservation.
"""

import os
import pytest
import pysubs2
from unittest.mock import patch, MagicMock

from subtrans.models import Cue
from subtrans.parser import (
    is_pure_sound_cue,
    snap_whisper_cues_to_en,
    parse_and_align_subtitles,
    strip_speaker_prefix,
)
from src.subtitles import find_or_extract_subtitles, transcribe_japanese_audio


def test_whisper_hallucination_filtering():
    """Nhiệm vụ 3: Kiểm tra nhận diện và lọc đúng các câu ảo giác Whisper."""
    hallucinations = [
        "ご視聴ありがとうございました",
        "チャンネル登録よろしくお願いします",
        "高評価よろしくお願いします",
        "字幕",
        "字幕：テスト",
        "提供",
        "提供：スポンサー",
        "ご覧いただきありがとうございました",
    ]
    for h in hallucinations:
        assert is_pure_sound_cue(h, is_whisper=True) is True, f"Failed on: {h}"
        # Khi is_whisper=False, các câu này không tự động bị coi là sound cue
        assert is_pure_sound_cue(h, is_whisper=False) is False, f"Should not drop when is_whisper=False: {h}"

    # Thoại thông thường không bị nhầm là ảo giác
    normal_dialogues = [
        "こんにちは、元気ですか？",
        "待って！行かないで！",
        "おはようございます。",
    ]
    for d in normal_dialogues:
        assert is_pure_sound_cue(d, is_whisper=True) is False, f"False positive on: {d}"


def test_scream_and_repeat_loop_filtering():
    """Nhiệm vụ 3: Kiểm tra nhận diện tiếng hét lặp ký tự và vòng lặp token của Whisper."""
    screams_and_loops = [
        "ああああああああ",          # 1 ký tự lặp 8 lần
        "うわああああああ",          # 1 ký tự lặp 6 lần ở đuôi
        "わはわはわはわは",          # Cụm 2 ký tự lặp 4 lần
        "なんだなんだなんだなんだ",  # Cụm 3 ký tự lặp 4 lần
    ]
    for s in screams_and_loops:
        assert is_pure_sound_cue(s, is_whisper=True) is True, f"Failed on scream loop: {s}"

    # Thoại hợp lệ chứa từ thông thường không bị lọc
    assert is_pure_sound_cue("大丈夫だよ", is_whisper=True) is False
    assert is_pure_sound_cue("待ちなさい！", is_whisper=True) is False


def test_orphan_whisper_cues_dropped_in_music_gaps():
    """Nhiệm vụ 4: Lọc bỏ Orphan Whisper Cues (>2500ms từ câu EN gần nhất) giữa khoảng trống nhạc/BGM."""
    cues = [
        Cue(id=0, start=1000, end=3000, ja="こんにちは"),
        # Cue mồ côi nằm giữa đoạn nhạc cách câu EN gần nhất 7000ms
        Cue(id=1, start=10000, end=11500, ja="音楽の間の謎の声"),
        Cue(id=2, start=20000, end=22000, ja="さようなら"),
    ]
    en_events = [
        pysubs2.SSAEvent(start=1000, end=3000, text="Hello."),
        pysubs2.SSAEvent(start=20000, end=22000, text="Goodbye."),
    ]

    snapped = snap_whisper_cues_to_en(cues, en_events)
    assert snapped == 2
    # Cue 0 và Cue 2 được giữ lại và đồng bộ
    assert cues[0].is_dropped_sound is False
    assert cues[0].is_skipped is False
    assert cues[2].is_dropped_sound is False
    assert cues[2].is_skipped is False

    # Cue 1 (mồ côi) phải bị đánh dấu loại bỏ
    assert cues[1].is_dropped_sound is True
    assert cues[1].is_skipped is True
    assert cues[1].backend_used == "dropped_sound"


def test_orphan_short_whisper_cues_dropped():
    """Nhiệm vụ 4: Cue Whisper mồ côi ngắn (len <= 6) không match DTW cũng bị loại bỏ."""
    cues = [
        Cue(id=0, start=1000, end=3000, ja="こんにちは"),
        Cue(id=1, start=7000, end=7400, ja="短い声"),  # Unmapped bởi DTW (khoảng cách > 2500ms) và ngắn <= 6 ký tự
        Cue(id=2, start=15000, end=17000, ja="さようなら"),
    ]
    en_events = [
        pysubs2.SSAEvent(start=1000, end=3000, text="Hello."),
        pysubs2.SSAEvent(start=15000, end=17000, text="Goodbye."),
    ]

    snapped = snap_whisper_cues_to_en(cues, en_events)
    assert snapped == 2
    assert cues[1].is_dropped_sound is True
    assert cues[1].is_skipped is True
    assert cues[1].backend_used == "dropped_sound"


def test_pass3_does_not_shift_by_skipped_or_dropped_cues():
    """Nhiệm vụ 5: Pass 3 không để các cue is_skipped=True / is_dropped_sound=True làm dịch chuyển câu thoại hợp lệ."""
    cues = [
        Cue(id=0, start=1000, end=2000, ja="First line"),
        # Cue đã bị drop hoặc skip có start=1950, end=2500
        Cue(id=1, start=1950, end=2500, ja="Dropped noise", is_dropped_sound=True, is_skipped=True),
        # Cue thoại hợp lệ tiếp theo bắt đầu lúc 2050
        Cue(id=2, start=2050, end=3000, ja="Second line"),
    ]
    en_events = [
        pysubs2.SSAEvent(start=1000, end=2000, text="First line"),
        pysubs2.SSAEvent(start=2050, end=3000, text="Second line"),
    ]

    snap_whisper_cues_to_en(cues, en_events)

    # c2.start không bị đẩy lùi bởi c1 (đã bị drop)
    assert cues[0].start == 1000
    assert cues[0].end == 2000
    assert cues[2].start == 2050
    assert cues[2].end == 3000


def test_pass3_dual_dialogue_preserved():
    """Nhiệm vụ 5: Hai cue đồng thời (Dual Dialogue) không bị Pass 3 cưỡng bức đẩy cách nhau min_gap_ms."""
    cues = [
        Cue(id=0, start=1000, end=3000, ja="Speaker A line"),
        Cue(id=1, start=1010, end=3000, ja="Speaker B line"),  # abs(start diff) <= 15 và cùng end
    ]
    en_events = [
        pysubs2.SSAEvent(start=1000, end=3000, text="Line A"),
        pysubs2.SSAEvent(start=1010, end=3000, text="Line B"),
    ]

    snap_whisper_cues_to_en(cues, en_events)

    # Cả hai cue vẫn giữ cấu trúc đồng thời cho Dual Dialogue, không bị c2.start = c1.end + 50
    assert cues[0].start == 1000
    assert cues[1].start == 1010
    assert cues[1].end == 3000


def test_multi_speaker_and_speaker_prefix_preservation(tmp_path):
    """Nhiệm vụ 6: Bảo toàn thông tin speaker trên Cue và chỉ dịch chuyển +10ms cho non-skipped cues."""
    ja_file = tmp_path / "multi_speaker.ass"
    ja_ssa = pysubs2.SSAFile()
    ja_ssa.events.append(
        pysubs2.SSAEvent(
            start=2000,
            end=4000,
            text=r"-（絹秀）金淑妃\N-（清佳）行くぞ！",
            style="Default",
        )
    )
    ja_ssa.save(str(ja_file))

    cues, _, _ = parse_and_align_subtitles(str(ja_file), auto_align_whisper_to_en=False)

    # Đã được tách thành 2 cue độc lập
    assert len(cues) == 2
    assert cues[0].speaker == "絹秀"
    assert cues[1].speaker == "清佳"
    assert cues[0].ja == "金淑妃"
    assert cues[1].ja == "行くぞ！"

    # Cả 2 cue đều không bị skip -> Cue 1 được dịch chuyển +10ms so với Cue 0
    assert cues[0].start == 2000
    assert cues[1].start == 2010


def test_multi_speaker_with_interjection_preserves_speaker(tmp_path):
    """Nhiệm vụ 6: Câu thoại phụ có tiếng than thở vẫn giữ đúng speaker kể cả khi bị đánh dấu sound drop."""
    ja_file = tmp_path / "multi_speaker_sound.ass"
    ja_ssa = pysubs2.SSAFile()
    ja_ssa.events.append(
        pysubs2.SSAEvent(
            start=2000,
            end=4000,
            text=r"-（絹秀）金淑妃\N-（清佳）あっ…",
            style="Default",
        )
    )
    ja_ssa.save(str(ja_file))

    cues, _, _ = parse_and_align_subtitles(str(ja_file), auto_align_whisper_to_en=False)

    assert len(cues) == 2
    assert cues[0].speaker == "絹秀"
    assert cues[1].speaker == "清佳"
    assert cues[1].ja == "あっ…"
    assert cues[1].is_dropped_sound is True
    # Vì cue 1 bị skip/drop, không áp dụng dịch chuyển +10ms
    assert cues[1].start == 2000


def test_find_or_extract_subtitles_prefer_whisper(tmp_path, monkeypatch):
    """Nhiệm vụ 2: prefer_whisper_when_no_ja=True kích hoạt Whisper STT thay vì dùng trực tiếp EN."""
    fake_video = str(tmp_path / "ep01.mkv")
    with open(fake_video, "w", encoding="utf-8") as f:
        f.write("fake video content" * 10)

    en_file = str(tmp_path / "ep01.en.ass")
    with open(en_file, "w", encoding="utf-8") as f:
        f.write("[Script Info]\nTitle: Test\nScriptType: v4.00+\n\n[Events]\nFormat: Layer, Start, End, Style, Text\nDialogue: 0,0:00:01.00,0:00:03.00,Default,Hello World\n")

    # Giả lập transcribe_japanese_audio
    transcribe_called = []
    def mock_transcribe(video_path, output_srt, model_size="small"):
        transcribe_called.append((video_path, output_srt, model_size))
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("1\n00:00:01,000 --> 00:00:03,000\nテスト\n" + "#" * 120)
        return True

    monkeypatch.setattr("src.subtitles.transcribe_japanese_audio", mock_transcribe)

    # 1. Khi prefer_whisper_when_no_ja=False (mặc định) -> dùng trực tiếp EN
    ja_sub, en_sub = find_or_extract_subtitles(fake_video, prefer_whisper_when_no_ja=False)
    assert ja_sub == en_file
    assert len(transcribe_called) == 0

    # 2. Khi prefer_whisper_when_no_ja=True -> gọi transcribe_japanese_audio
    ja_sub2, en_sub2 = find_or_extract_subtitles(fake_video, prefer_whisper_when_no_ja=True)
    assert len(transcribe_called) == 1
    assert "whisper_ja.srt" in ja_sub2

    # 3. Qua biến môi trường SUBTRANS_WHISPER_WITH_EN=1
    transcribe_called.clear()
    if os.path.exists(ja_sub2):
        os.remove(ja_sub2)
    monkeypatch.setenv("SUBTRANS_WHISPER_WITH_EN", "1")
    ja_sub3, _ = find_or_extract_subtitles(fake_video, prefer_whisper_when_no_ja=False)
    assert len(transcribe_called) == 1
    assert "whisper_ja.srt" in ja_sub3


def test_transcribe_japanese_audio_filters_and_segment_rejection(tmp_path, monkeypatch):
    """Nhiệm vụ 1: Kiểm tra bộ lọc âm thanh FFmpeg (5.1 vs stereo) và lọc segment rác Whisper."""
    import sys
    import json
    import subprocess

    video_fake = str(tmp_path / "test.mkv")
    with open(video_fake, "w", encoding="utf-8") as f:
        f.write("fake video")

    out_srt = str(tmp_path / "test.ja.srt")

    # Mock ffprobe trả về audio 5.1 (channels=6) tiếng Nhật
    real_run = subprocess.run
    def mock_subprocess_run(cmd, *args, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        if "ffprobe" not in cmd_str and "ffmpeg" not in cmd_str:
            return real_run(cmd, *args, **kwargs)

        mock_res = MagicMock()
        mock_res.returncode = 0
        if "ffprobe" in cmd_str:
            data = {
                "streams": [
                    {"index": 1, "channels": 6, "tags": {"language": "jpn", "title": "Japanese 5.1"}}
                ]
            }
            mock_res.stdout = json.dumps(data)
            return mock_res
        elif "ffmpeg" in cmd_str:
            # Tạo audio_tmp file
            audio_tmp = cmd[-1]
            with open(audio_tmp, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 100)
            mock_res.returncode = 0
            # Lưu cmd để kiểm tra filter
            mock_subprocess_run.last_ffmpeg_cmd = cmd
            return mock_res
        return mock_res

    mock_subprocess_run.last_ffmpeg_cmd = []
    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    # Mock whisper
    mock_model = MagicMock()
    transcribe_params = {}
    def mock_transcribe(audio_path, **kwargs):
        transcribe_params.update(kwargs)
        return {
            "segments": [
                # 1. Hợp lệ
                {"start": 1.0, "end": 3.0, "text": "おはよう", "no_speech_prob": 0.05, "compression_ratio": 1.1, "avg_logprob": -0.2},
                # 2. Bị lọc do no_speech_prob > 0.65
                {"start": 3.5, "end": 5.0, "text": "雑音", "no_speech_prob": 0.70, "compression_ratio": 1.1, "avg_logprob": -0.2},
                # 3. Bị lọc do compression_ratio > 2.2
                {"start": 5.5, "end": 7.0, "text": "ループ", "no_speech_prob": 0.1, "compression_ratio": 2.5, "avg_logprob": -0.2},
                # 4. Bị lọc do avg_logprob < -1.2 và dur < 800ms
                {"start": 7.5, "end": 8.0, "text": "低い", "no_speech_prob": 0.1, "compression_ratio": 1.1, "avg_logprob": -1.5},
            ]
        }
    mock_model.transcribe = mock_transcribe

    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    monkeypatch.setitem(sys.modules, "whisper", mock_whisper)

    ok = transcribe_japanese_audio(video_fake, out_srt, model_size="small")
    assert ok is True
    assert os.path.exists(out_srt)

    # 1. Kiểm tra audio filter cho 5.1 (channels=6)
    last_cmd = mock_subprocess_run.last_ffmpeg_cmd
    assert "-af" in last_cmd
    af_idx = last_cmd.index("-af")
    assert "pan=mono|c0=0.7*FC" in last_cmd[af_idx + 1]
    assert "0:1" in last_cmd  # stream_map 0:1

    # 2. Kiểm tra tham số gọi model.transcribe
    assert transcribe_params.get("condition_on_previous_text") is False
    assert transcribe_params.get("no_speech_threshold") == 0.6
    assert transcribe_params.get("logprob_threshold") == -1.0
    assert transcribe_params.get("compression_ratio_threshold") == 2.0

    # 3. Kiểm tra chỉ segment hợp lệ được ghi ra SRT
    subs = pysubs2.load(out_srt)
    assert len(subs) == 1
    assert subs[0].text == "おはよう"

