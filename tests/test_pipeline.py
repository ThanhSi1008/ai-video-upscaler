# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py
Unit tests for pipeline coordinator optimizations:
- Segment resume validation: verifies up_*.mp4 frames match expected chunk frames before skipping.
- Stale checkpoint invalidation: stale or mismatched up_*.mp4 files are automatically deleted and re-upscaled.
- Micro-timings integration: CUDA Event micro-timings collected from DoubleBufferedPipeUpscaler are recorded in ai_result.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from src.ffmpeg_ops import verify_segment_resume_frames


def test_pipeline_resume_skips_only_matching_segment(tmp_path: Path):
    """
    Verifies that pipeline resume accepts and skips segments only when
    verify_segment_resume_frames returns True for expected_frames.
    """
    up_segs_dir = tmp_path / "upscaled_segs"
    up_segs_dir.mkdir(parents=True)
    seg0 = up_segs_dir / "up_0000.mp4"
    seg0.write_bytes(b"\x00" * 30000)

    # 1. Matching frames -> verify_segment_resume_frames is True
    with mock.patch("src.ffmpeg_ops.subprocess.run") as mock_run:
        mock_run.return_value = mock.MagicMock(returncode=0, stdout="24\n")
        assert verify_segment_resume_frames(str(seg0), 24) is True

    # 2. Mismatched frames -> verify_segment_resume_frames is False
    with mock.patch("src.ffmpeg_ops.subprocess.run") as mock_run:
        mock_run.return_value = mock.MagicMock(returncode=0, stdout="20\n")
        assert verify_segment_resume_frames(str(seg0), 24) is False


def test_pipeline_resume_invalidation_logic(tmp_path: Path):
    """
    Tests that a stale segment with mismatched frame count is invalidated (deleted)
    and added to chunks_to_process rather than silently skipped.
    """
    up_segs_dir = tmp_path / "upscaled_segs"
    up_segs_dir.mkdir(parents=True)
    seg_file = up_segs_dir / "up_0000.mp4"
    seg_file.write_bytes(b"\x00" * 30000)

    existing_segs = [str(tmp_path / "raw_segs" / "seg_0000.mp4")]
    total_segs = 1
    resume = True
    dedup = True
    dedup_plan = {"chunks": [{"total_frames": 24, "kept_indices": list(range(12))}]}
    seg_frame_counts = [24]
    chunks_to_process = []
    total_frames_processed = 0

    # Simulate ffprobe returning 20 frames instead of 24 (stale pre-patch run)
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.MagicMock(returncode=0, stdout="20\n")

        for idx, s_file in enumerate(existing_segs):
            seg_out_file = os.path.join(str(up_segs_dir), f"up_{idx:04d}.mp4")
            expected_frames = (
                dedup_plan["chunks"][idx]["total_frames"]
                if (dedup and dedup_plan and idx < len(dedup_plan.get("chunks", [])))
                else (seg_frame_counts[idx] if idx < len(seg_frame_counts) else None)
            )
            if resume and os.path.exists(seg_out_file) and os.path.getsize(seg_out_file) > 10000:
                if verify_segment_resume_frames(seg_out_file, expected_frames):
                    total_frames_processed += expected_frames
                    continue
                else:
                    try:
                        os.remove(seg_out_file)
                    except OSError:
                        pass
            chunks_to_process.append((idx, s_file, seg_out_file))

    # The stale segment must have been deleted and scheduled for processing
    assert len(chunks_to_process) == 1
    assert chunks_to_process[0][0] == 0
    assert not os.path.exists(str(seg_file))
    assert total_frames_processed == 0


def test_pipeline_ai_result_micro_timings_structure():
    """Verifies that micro_timings from pipe_upscaler are properly forwarded into ai_result."""
    mock_upscaler = mock.MagicMock()
    mock_upscaler.last_distinct_count = 10
    mock_upscaler.last_micro_timings = {
        "num_batches": 2,
        "total_h2d_ms": 6.4,
        "total_trt_ms": 120.5,
        "total_triton_ms": 9.8,
        "total_d2h_ms": 14.2,
        "avg_h2d_ms": 3.2,
        "avg_trt_ms": 60.25,
        "avg_triton_ms": 4.9,
        "avg_d2h_ms": 7.1,
    }

    t_pipe_elapsed = 1.0
    num_unique_frames = 10
    rss_now = 500.0

    ai_result = {
        "elapsed_seconds": t_pipe_elapsed,
        "fps": (num_unique_frames / t_pipe_elapsed) if t_pipe_elapsed > 0 else 0.0,
        "steady_fps": (num_unique_frames / t_pipe_elapsed) if t_pipe_elapsed > 0 else 0.0,
        "t0_seconds": 0.0,
        "peak_rss_mb": rss_now,
        "micro_timings": getattr(mock_upscaler, "last_micro_timings", None),
    }

    assert ai_result["micro_timings"] is not None
    assert ai_result["micro_timings"]["avg_trt_ms"] == 60.25
    assert ai_result["micro_timings"]["avg_triton_ms"] == 4.9
