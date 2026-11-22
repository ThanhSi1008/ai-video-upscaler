# -*- coding: utf-8 -*-
"""
tests/test_ffmpeg_ops.py
Unit tests for ffmpeg_ops optimizations:
- verify_segment_resume_frames: exact packet counting (~0.04s), invalidates stale/short segments.
- get_segments_frame_counts: Open-GOP CRA discard packet reflection and safe fallback.
- _count_leading_discard_packets: RASL leading B-frame detection on CRA boundary.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
import subprocess
from unittest import mock

import pytest

from src.ffmpeg_ops import (
    _count_leading_discard_packets,
    _count_segment_packets_ffprobe,
    _count_segment_packets_pyav,
    _probe_video_ffprobe,
    _probe_video_pyav,
    _recover_open_gop_leading_frames,
    extract_frames_raw_yuv420,
    get_segments_frame_counts,
    probe_video,
    split_video_segments,
    verify_segment_resume_frames,
)


def test_verify_segment_resume_frames_exact_match(tmp_path: Path):
    """When up_*.mp4 packet count exactly matches expected_frames, return True."""
    valid_file = tmp_path / "up_0000.mp4"
    valid_file.write_bytes(b"\x00" * 25000)

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.MagicMock(returncode=0, stdout="218\n")
        assert verify_segment_resume_frames(str(valid_file), 218) is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "-count_packets" in cmd
        assert "nb_read_packets" in cmd[cmd.index("-show_entries") + 1]


def test_verify_segment_resume_frames_mismatch_fails(tmp_path: Path):
    """When up_*.mp4 packet count does not match expected_frames, return False."""
    stale_file = tmp_path / "up_0017.mp4"
    stale_file.write_bytes(b"\x00" * 25000)

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.MagicMock(returncode=0, stdout="216\n")
        # Expected 218 frames, but file only has 216 frames
        assert verify_segment_resume_frames(str(stale_file), 218) is False


def test_verify_segment_resume_frames_small_file_fails(tmp_path: Path):
    """Files < 10,000 bytes are incomplete/stubs and must return False without calling ffprobe."""
    tiny_file = tmp_path / "up_stub.mp4"
    tiny_file.write_bytes(b"\x00" * 4096)

    with mock.patch("subprocess.run") as mock_run:
        assert verify_segment_resume_frames(str(tiny_file), 218) is False
        mock_run.assert_not_called()


def test_verify_segment_resume_frames_nonexistent_fails(tmp_path: Path):
    """Non-existent path returns False immediately."""
    assert verify_segment_resume_frames(str(tmp_path / "absent.mp4"), 218) is False


def test_verify_segment_resume_frames_corrupted_container_fails(tmp_path: Path):
    """Corrupted file causing ffprobe non-zero exit or error returns False."""
    corrupt_file = tmp_path / "up_corrupt.mp4"
    corrupt_file.write_bytes(b"\x00" * 20000)

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.MagicMock(returncode=1, stdout="")
        assert verify_segment_resume_frames(str(corrupt_file), 218) is False

    with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=10)):
        assert verify_segment_resume_frames(str(corrupt_file), 218) is False


def test_get_segments_frame_counts_open_gop_fallback(tmp_path: Path):
    """
    When count_packets succeeds, returns packet counts including discard packets.
    When count_packets fails, falls back to count_frames + _count_leading_discard_packets.
    """
    seg1 = tmp_path / "seg_0000.mp4"
    seg2 = tmp_path / "seg_0001.mp4"
    seg1.write_bytes(b"dummy1")
    seg2.write_bytes(b"dummy2")

    # Success case: returns packet counts
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.MagicMock(returncode=0, stdout="218\n")
        counts = get_segments_frame_counts([str(seg1), str(seg2)])
        assert counts == [218, 218]

    # Fallback case: count_packets returns empty or fails, count_frames + discard used
    def mock_run_side_effect(cmd, *args, **kwargs):
        if "-count_packets" in cmd:
            return mock.MagicMock(returncode=1, stdout="")
        elif "-count_frames" in cmd:
            return mock.MagicMock(returncode=0, stdout="216\n")
        return mock.MagicMock(returncode=0, stdout="")

    with mock.patch("subprocess.run", side_effect=mock_run_side_effect):
        with mock.patch("src.ffmpeg_ops._count_leading_discard_packets", return_value=2):
            counts = get_segments_frame_counts([str(seg1)])
            assert counts == [218]  # 216 frames + 2 discard


def test_count_leading_discard_packets_detection(tmp_path: Path):
    """Tests _count_leading_discard_packets correctly parses Open-GOP DISCARD flags."""
    seg = tmp_path / "seg.mp4"
    seg.write_bytes(b"dummy")

    with mock.patch("subprocess.run") as mock_run:
        # 2 discard packets with 'D' flag
        mock_run.return_value = mock.MagicMock(
            returncode=0,
            stdout="K__\n_D_\n_D_\n___\n___\n",
        )
        assert _count_leading_discard_packets(str(seg)) == 2

    with mock.patch("subprocess.run") as mock_run:
        # Closed GOP: 0 discard flags
        mock_run.return_value = mock.MagicMock(
            returncode=0,
            stdout="K__\n___\n___\n___\n",
        )
        assert _count_leading_discard_packets(str(seg)) == 0


def test_probe_video_pyav_mock_container():
    """Tests _probe_video_pyav accurately parses stream specs and callable chapters."""
    with mock.patch("av.open") as mock_open:
        mock_c = mock.MagicMock()
        mock_v = mock.MagicMock()
        mock_v.width = 1920
        mock_v.height = 1080
        mock_v.name = "hevc"
        mock_v.format.name = "yuv420p10le"
        mock_v.profile = "Main 10"
        mock_v.average_rate = mock.MagicMock(numerator=24000, denominator=1001)
        mock_v.frames = 1420
        mock_v.duration = None
        mock_v.time_base = None

        mock_c.streams.video = [mock_v]
        mock_c.streams.audio = [mock.MagicMock(), mock.MagicMock()]
        mock_c.streams.subtitles = [mock.MagicMock()]
        mock_c.streams.attachments = [mock.MagicMock(), mock.MagicMock(), mock.MagicMock()]
        mock_c.duration = 59_200_000
        mock_c.size = 12345678
        # PyAV 18.1.0 callable chapters
        mock_c.chapters = mock.MagicMock(return_value=[{"id": 0}, {"id": 1}])
        mock_open.return_value.__enter__.return_value = mock_c

        meta = _probe_video_pyav("dummy.mkv")
        assert meta["width"] == 1920
        assert meta["height"] == 1080
        assert meta["codec"] == "hevc"
        assert meta["pix_fmt"] == "yuv420p10le"
        assert meta["profile"] == "Main 10"
        assert meta["fps_num"] == 24000
        assert meta["fps_den"] == 1001
        assert meta["nb_frames"] == 1420
        assert meta["audio_count"] == 2
        assert meta["subtitle_count"] == 1
        assert meta["attachment_count"] == 3
        assert meta["chapter_count"] == 2
        assert meta["duration"] == 59.2


def test_probe_video_pyav_raw_chapters_list():
    """Tests _probe_video_pyav handles container.chapters when represented as a list."""
    with mock.patch("av.open") as mock_open:
        mock_c = mock.MagicMock()
        mock_v = mock.MagicMock()
        mock_v.width = 1280
        mock_v.height = 720
        mock_v.name = "h264"
        mock_v.format.name = "yuv420p"
        mock_v.profile = "High"
        mock_v.average_rate = None
        mock_v.base_rate = mock.MagicMock(numerator=30, denominator=1)
        mock_v.frames = 300

        mock_c.streams.video = [mock_v]
        mock_c.streams.audio = []
        mock_c.streams.subtitles = []
        mock_c.streams.attachments = []
        mock_c.duration = 10_000_000
        mock_c.size = 500000
        mock_c.chapters = [{"id": 0}]  # Non-callable list
        mock_open.return_value.__enter__.return_value = mock_c

        meta = _probe_video_pyav("dummy.mp4")
        assert meta["fps_num"] == 30
        assert meta["fps_den"] == 1
        assert meta["chapter_count"] == 1


def test_probe_video_fallback_to_ffprobe():
    """When PyAV encounters an exception, probe_video gracefully falls back to ffprobe."""
    with mock.patch("src.ffmpeg_ops._probe_video_pyav", side_effect=RuntimeError("PyAV failed")):
        with mock.patch("src.ffmpeg_ops._probe_video_ffprobe") as mock_ffprobe:
            mock_ffprobe.return_value = {"width": 1920, "height": 1080, "fps": 24.0}
            res = probe_video("test.mp4")
            assert res["width"] == 1920
            mock_ffprobe.assert_called_once_with("test.mp4")


def test_count_segment_packets_pyav_demux():
    """Tests _count_segment_packets_pyav counts packets via demux when frames is 0 or None."""
    with mock.patch("av.open") as mock_open:
        mock_c = mock.MagicMock()
        mock_v = mock.MagicMock()
        mock_v.frames = 0
        mock_c.streams.video = [mock_v]
        p1 = mock.MagicMock(size=100, dts=10)
        p2 = mock.MagicMock(size=200, dts=20)
        p3 = mock.MagicMock(size=0, dts=30)  # Empty packet ignored
        mock_c.demux.return_value = [p1, p2, p3]
        mock_open.return_value.__enter__.return_value = mock_c

        count = _count_segment_packets_pyav("seg.mp4")
        assert count == 2


def test_count_leading_discard_packets_pyav_mock(tmp_path: Path):
    """Tests _count_leading_discard_packets via in-process PyAV packet inspection."""
    seg = tmp_path / "valid_seg.mp4"
    seg.write_bytes(b"\x00" * 20000)

    with mock.patch("av.open") as mock_open:
        mock_c = mock.MagicMock()
        mock_v = mock.MagicMock()
        mock_c.streams.video = [mock_v]

        p_discard1 = mock.MagicMock(is_discard=True)
        p_discard2 = mock.MagicMock(is_discard=True)
        p_normal = mock.MagicMock(is_discard=False)
        mock_c.demux.return_value = [p_discard1, p_discard2, p_normal]
        mock_open.return_value.__enter__.return_value = mock_c

        assert _count_leading_discard_packets(str(seg)) == 2


def test_verify_segment_resume_frames_pyav(tmp_path: Path):
    """Tests verify_segment_resume_frames via PyAV in-process packet verification."""
    seg = tmp_path / "resume_seg.mp4"
    seg.write_bytes(b"\x00" * 20000)

    with mock.patch("av.open") as mock_open:
        mock_c = mock.MagicMock()
        mock_v = mock.MagicMock()
        mock_v.frames = 192
        mock_c.streams.video = [mock_v]
        mock_open.return_value.__enter__.return_value = mock_c

        assert verify_segment_resume_frames(str(seg), 192) is True
        assert verify_segment_resume_frames(str(seg), 200) is False


def test_real_open_gop_x265_cra_rasl_split_and_recovery(tmp_path: Path):
    """Verify real Open-GOP x265 CRA/RASL splitting, discard packet detection, and lossless frame recovery."""
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    src_mkv = tmp_path / "open_gop_src.mkv"
    segs_dir = tmp_path / "segs"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=2",
            "-vf", "format=yuv420p10le",
            "-frames:v", "48",
            "-c:v", "libx265", "-preset", "ultrafast",
            "-x265-params", "log-level=error:open-gop=1:keyint=24:min-keyint=24:bframes=2:b-adapt=0",
            "-pix_fmt", "yuv420p10le",
            str(src_mkv),
        ],
        check=True,
        capture_output=True,
    )

    segs = split_video_segments(str(src_mkv), str(segs_dir), segment_time=1)
    assert len(segs) == 2

    counts = get_segments_frame_counts(segs)
    assert sum(counts) == 48
    assert _count_leading_discard_packets(segs[1]) > 0

    extracted_counts = [
        len(
            extract_frames_raw_yuv420(
                seg,
                width=160,
                height=90,
                source_video_path=str(src_mkv),
                chunk_start_sec=1.0 if idx > 0 else 0.0,
                source_codec="hevc",
                source_pix_fmt="yuv420p10le",
            )
        )
        for idx, seg in enumerate(segs)
    ]
    assert sum(extracted_counts) == 48
    assert extracted_counts == counts


def test_recover_open_gop_leading_frames_black_fade_aliasing(tmp_path: Path):
    """Lock Open-GOP CRA solid-black fade anchor aliasing bug on real ffmpeg rawvideo decode."""
    import numpy as np

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    src_mkv = tmp_path / "black_fade.mkv"
    # 12 black frames, 4 red frames, 12 black frames (CRA black anchor at frame 16)
    black = np.zeros((90, 160, 3), dtype=np.uint8)
    red = np.zeros((90, 160, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    raw_bytes = b"".join([black.tobytes()] * 12 + [red.tobytes()] * 4 + [black.tobytes()] * 12)
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "160x90", "-r", "24", "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0", "-pix_fmt", "yuv420p",
            str(src_mkv),
        ],
        input=raw_bytes,
        check=True,
        capture_output=True,
    )

    recovered = _recover_open_gop_leading_frames(
        str(src_mkv),
        chunk_start_sec=16.0 / 24.0,
        cra_anchor_frame=black,
        n_discard=2,
        width=160,
        height=90,
        pix_fmt="rgb24",
    )
    assert len(recovered) == 2
    assert all(float(frame[:, :, 0].mean()) > 150.0 for frame in recovered)
