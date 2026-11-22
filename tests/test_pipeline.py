# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py
Unit tests for pipeline coordinator optimizations:
- Segment resume validation: verifies up_*.mp4 frames match expected chunk frames before skipping.
- Stale checkpoint invalidation: stale or mismatched up_*.mp4 files are automatically deleted and re-upscaled.
- Micro-timings integration: CUDA Event micro-timings collected from DoubleBufferedPipeUpscaler are recorded in ai_result.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn

from src.ffmpeg_ops import get_segments_frame_counts, verify_segment_resume_frames
from src.mux_audit import verify_mux_integrity
from src.pipeline import run_upscale_pipeline


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


def _make_2s_test_mkv(path: Path) -> None:
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-vf", "format=yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "24", "-keyint_min", "24", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "96k",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


class _DummyCpuEngine:
    def __init__(self) -> None:
        self.model = nn.Upsample(scale_factor=2, mode="nearest")
        self.device = torch.device("cpu")
        self.batch_size = 2

    def restart_if_needed(self, **_kwargs) -> None:
        return None


def _run_cpu_pipeline(src_mkv: Path, out_dir: Path, **extra_kwargs) -> str:
    cpu_engine = _DummyCpuEngine()
    with (
        mock.patch.dict(os.environ, {"UPSCALER_HEVC_ENCODER": "libx265"}),
        mock.patch("src.config.HEVC_ENCODER", "libx265"),
        mock.patch("src.pipeline.HEVC_ENCODER", "libx265"),
        mock.patch("src.cuda_engine.get_cuda_inference_engine", return_value=cpu_engine),
        mock.patch("src.pipeline.acquire_gpu_lock", return_value=None),
        mock.patch("src.pipeline.release_gpu_lock", return_value=None),
        mock.patch("src.pipeline.AntiSleepContext", return_value=nullcontext()),
        mock.patch("src.pipeline.send_system_notification", return_value=None),
    ):
        return run_upscale_pipeline(
            input_path=str(src_mkv),
            output_path=str(out_dir),
            engine="cuda",
            segment_time=1,
            segment_time_explicit=True,
            zero_disk=True,
            use_tensorrt=False,
            precision="fp32",
            dedup=False,
            anti_sleep=False,
            **extra_kwargs,
        )


def test_pipeline_resume_invalidation_logic(tmp_path: Path):
    """Verify run_upscale_pipeline invalidates and re-encodes a mismatched up_0000.mp4 checkpoint."""
    src_mkv = tmp_path / "ep_stale.mkv"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _make_2s_test_mkv(src_mkv)

    up_segs_dir = out_dir / ".temp_ep_stale" / "upscaled_segs"
    up_segs_dir.mkdir(parents=True, exist_ok=True)
    stale_seg0 = up_segs_dir / "up_0000.mp4"

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-v", "error",
            "-f", "lavfi", "-i", "mandelbrot=size=320x180:rate=24",
            "-frames:v", "12",
            "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error:crf=0",
            "-pix_fmt", "yuv420p10le",
            str(stale_seg0),
        ],
        check=True,
        capture_output=True,
    )
    assert stale_seg0.stat().st_size > 10000
    assert verify_segment_resume_frames(str(stale_seg0), 24) is False

    out_4k = _run_cpu_pipeline(src_mkv, out_dir, resume=True, keep_temp=True)
    assert Path(out_4k).is_file()
    assert verify_segment_resume_frames(str(stale_seg0), 24) is True
    assert get_segments_frame_counts([out_4k]) == [48]
    assert verify_mux_integrity(str(src_mkv), out_4k)["status"] == "PASS"


def test_pipeline_ai_result_micro_timings_structure(tmp_path: Path):
    """Verify real run_upscale_pipeline emits populated micro_timings in the episode .audit.json."""
    src_mkv = tmp_path / "ep_micro.mkv"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _make_2s_test_mkv(src_mkv)

    out_4k = _run_cpu_pipeline(src_mkv, out_dir, resume=False)
    audit_path = Path(out_4k).with_suffix(".audit.json")
    assert audit_path.is_file()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    mt = audit.get("micro_timings")
    assert isinstance(mt, dict)
    assert mt["num_batches"] >= 1
    assert mt["total_in_norm_ms"] >= 0.0
    assert mt["total_trt_ms"] >= 0.0
    assert mt["total_triton_ms"] >= 0.0
