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
from pathlib import Path
import subprocess
from unittest import mock

import pytest

from src.ffmpeg_ops import (
    _count_leading_discard_packets,
    get_segments_frame_counts,
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
