# -*- coding: utf-8 -*-
"""
tests/e2e/test_tier3_pairwise.py
Tier 3: Cross-Feature Pairwise & Integration Tests (10 Tests)
Covers pairwise interactions between major CLI features of ai-video-upscaler:
T3-01: Queue Mode + Preview Mode (--queue + --preview)
T3-02: Zero-Disk Pipeline + DIS Dedup Policy (--zero-disk + --dedup-policy dis)
T3-03: Overnight Queue + Deadline + Allow-Partial (--queue + --deadline + --allow-partial)
T3-04: Subtitle Quality Tier + Auto-Align + Sub-Offset (--subs-quality high + --auto-align-en + --sub-offset-ms)
T3-05: Dry-Run Scheduler + Dedup Policy + Keep-Temp (--dry-run + --dedup-policy + --keep-temp)
T3-06: Engine Selection + Precision Fallback (--engine cpu + --precision fp32 + --no-tensorrt)
T3-07: Model Conflict Detection & Force Override (--model + manifest.json + --force)
T3-08: Subtitle Async Background in Multi-Queue (--async-subs + --queue)
T3-09: Output Resolution & Temp Cleanup (-o [path] + --cleanup + --no-organize-audit)
T3-10: Failed-Resumable ENOSPC Simulation & Queue Retry (--queue + FailedResumableError)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import upscale
from tests.e2e.conftest import create_synthetic_video, run_upscale_cli


# =============================================================================
# T3-01: Queue Mode + Preview Mode
# =============================================================================

def test_t3_01_queue_and_preview_modes(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T3-01: Queue Mode + Preview Mode executes preview for multiple videos without collision."""
    # Use 72 frames (3.0s at 24fps) so int(duration) >= 2 for preview_seconds=2
    ep1 = create_synthetic_video(tmp_path / "ep1.mkv", num_frames=72, fps=24, audio_tracks=1)
    ep2 = create_synthetic_video(tmp_path / "ep2.mkv", num_frames=72, fps=24, audio_tracks=1)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        ["--queue", str(ep1), str(ep2), "--preview", "2", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    # Both preview outputs exist
    p1 = out_dir / "ep1_preview_2s_4K.mkv"
    p2 = out_dir / "ep2_preview_2s_4K.mkv"
    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0

    # Intermediate preview clips are cleaned up
    assert not (out_dir / ".preview_clip_ep1.mkv").exists()
    assert not (out_dir / ".preview_clip_ep2.mkv").exists()


# =============================================================================
# T3-02: Zero-Disk Pipeline + DIS Dedup Policy
# =============================================================================

def test_t3_02_zero_disk_and_dedup_policy(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T3-02: Zero-Disk Pipeline + DIS Dedup Policy (0 disk PNGs, P010LE NVENC stream)."""
    src_vid = create_synthetic_video(tmp_path / "action_scene.mkv", num_frames=8, audio_tracks=1)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--zero-disk", "--dedup-policy", "dis", "--dis-thresh", "0.30", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    out_file = out_dir / "action_scene_4K.mkv"
    assert out_file.exists()

    # Verify zero intermediate PNG frame files created on disk
    png_files = list(out_dir.rglob("*.png"))
    assert len(png_files) == 0


# =============================================================================
# T3-03: Overnight Queue + Deadline + Allow-Partial
# =============================================================================

def test_t3_03_queue_deadline_and_allow_partial(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T3-03: Overnight Queue + Deadline + Allow-Partial pauses in-flight episode at chunk boundary."""
    ep1 = create_synthetic_video(tmp_path / "ep1.mkv", num_frames=8, audio_tracks=1)
    ep2 = create_synthetic_video(tmp_path / "ep2.mkv", num_frames=8, audio_tracks=1)
    out_dir = tmp_path / "out"

    # Mock pipeline so ep1 finishes cleanly, but ep2 triggers allow-partial pause
    real_run_pipeline = upscale.run_upscale_pipeline

    def mock_pipeline_selector(input_path, *args, **kwargs):
        if "ep1" in str(input_path):
            return real_run_pipeline(input_path=input_path, *args, **kwargs)
        # For ep2, simulate allow-partial pause returning up_segs_dir
        temp_dir = out_dir / ".temp_ep2"
        up_segs_dir = temp_dir / "upscaled_segs"
        up_segs_dir.mkdir(parents=True, exist_ok=True)
        (up_segs_dir / "up_0000.mp4").write_bytes(b"VALID_CHUNK_0")
        return str(up_segs_dir)

    with patch("upscale.run_upscale_pipeline", side_effect=mock_pipeline_selector):
        exit_code, out, err = run_upscale_cli(
            ["--queue", str(ep1), str(ep2), "--deadline", "07:00", "--allow-partial", "-o", str(out_dir)],
            monkeypatch,
            capsys,
        )

    # Incomplete queue returns exit code 1
    assert exit_code == 1
    # ep1 completed and muxed
    assert (out_dir / "ep1_4K.mkv").exists()
    # ep2 was not muxed into final container
    assert not (out_dir / "ep2_4K.mkv").exists()
    # ep2 chunk segments preserved
    assert (out_dir / ".temp_ep2" / "upscaled_segs" / "up_0000.mp4").exists()


# =============================================================================
# T3-04: Subtitle Quality Tier + Auto-Align + Sub-Offset
# =============================================================================

def test_t3_04_subs_quality_auto_align_and_sub_offset(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T3-04: Subtitle Quality Tier + Auto-Align + Sub-Offset applies options without error."""
    src_vid = create_synthetic_video(
        tmp_path / "subs_test.mkv",
        num_frames=8,
        audio_tracks=1,
        subtitles=["Hello world line 1", "Second line of dialog"],
    )
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [
            "--subs-only", str(src_vid),
            "--subs-quality", "high",
            "--auto-align-en",
            "--sub-offset-ms", "-2000",
            "-o", str(out_dir),
        ],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0


# =============================================================================
# T3-05: Dry-Run Scheduler + Dedup Policy + Keep-Temp
# =============================================================================

def test_t3_05_dry_run_dedup_policy_and_keep_temp(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T3-05: Dry-Run Scheduler + Dedup Policy + Keep-Temp computes ETA without allocating GPU or files."""
    ep1 = create_synthetic_video(tmp_path / "ep1.mkv", num_frames=8)
    ep2 = create_synthetic_video(tmp_path / "ep2.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [
            "--queue", str(ep1), str(ep2),
            "--dry-run",
            "--dedup-policy", "sensitive-b",
            "--keep-temp",
            "--deadline", "08:00",
            "-o", str(out_dir),
        ],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert "DRY RUN" in out
    assert "BẢNG QUYẾT ĐỊNH" in out
    assert not Path(".gpu_active.lock").exists()
    assert not (out_dir / "ep1_4K.mkv").exists()


# =============================================================================
# T3-06: Engine Selection + Precision Fallback
# =============================================================================

def test_t3_06_engine_cpu_and_precision_fp32(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T3-06: Engine Selection + Precision Fallback (--engine cpu + --precision fp32 + --no-tensorrt)."""
    src_vid = create_synthetic_video(tmp_path / "test_cpu_fp32.mkv", num_frames=8, audio_tracks=1)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--engine", "cpu", "--precision", "fp32", "--no-tensorrt", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert (out_dir / "test_cpu_fp32_4K.mkv").exists()


# =============================================================================
# T3-07: Model Conflict Detection & Force Override
# =============================================================================

def test_t3_07_model_conflict_and_force_override(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T3-07: Model Conflict Detection rejects mixed models, while --force overrides cleanly."""
    src_vid = create_synthetic_video(tmp_path / "conflict_ep.mkv", num_frames=8, audio_tracks=1)
    out_dir = tmp_path / "out"
    temp_dir = out_dir / ".temp_conflict_ep"
    temp_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "model": "animevideov3",
        "model_id": "animevideov3",
        "scale": 2,
        "engine": "cuda",
    }
    (temp_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    # Run A: Without --force -> rejects execution
    exit_code_a, out_a, err_a = run_upscale_cli(
        [str(src_vid), "--model", "animejanai-hd-v3-compact", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code_a == 1
    assert "Xung đột mô hình" in (out_a + err_a)

    # Run B: With --force -> wipes old checkpoint and completes
    exit_code_b, out_b, err_b = run_upscale_cli(
        [str(src_vid), "--model", "animejanai-hd-v3-compact", "--force", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code_b == 0
    assert (out_dir / "conflict_ep_4K_animejanai.mkv").exists()


# =============================================================================
# T3-08: Subtitle Async Background in Multi-Queue
# =============================================================================

def test_t3_08_async_subs_in_multi_queue(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T3-08: Subtitle Async Background in Multi-Queue coordinates concurrent runs."""
    ep1 = create_synthetic_video(tmp_path / "async_ep1.mkv", num_frames=8, audio_tracks=1, subtitles=["Ep1 Sub"])
    ep2 = create_synthetic_video(tmp_path / "async_ep2.mkv", num_frames=8, audio_tracks=1, subtitles=["Ep2 Sub"])
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        ["--queue", str(ep1), str(ep2), "--async-subs", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert (out_dir / "async_ep1_4K.mkv").exists()
    assert (out_dir / "async_ep2_4K.mkv").exists()


# =============================================================================
# T3-09: Output Resolution & Temp Cleanup
# =============================================================================

def test_t3_09_output_resolution_and_temp_cleanup(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T3-09: Output Resolution & Temp Cleanup (-o [path] + --cleanup + --no-organize-audit)."""
    src_vid = create_synthetic_video(tmp_path / "clean_test.mkv", num_frames=8, audio_tracks=1)
    out_dir = tmp_path / "out"
    explicit_out = out_dir / "custom_named_4K.mkv"

    # Step 1: Run with explicit target file and no organize audit
    exit_code_1, out_1, err_1 = run_upscale_cli(
        [str(src_vid), "-o", str(explicit_out), "--no-organize-audit"],
        monkeypatch,
        capsys,
    )
    assert exit_code_1 == 0
    assert explicit_out.exists()

    # Step 2: Run cleanup on the output directory
    exit_code_2, out_2, err_2 = run_upscale_cli(
        ["--cleanup", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code_2 == 0
    assert "Đã dọn dẹp" in out_2 or "0 thư mục" in out_2 or "0.0 B" in out_2


# =============================================================================
# T3-10: Failed-Resumable ENOSPC Simulation & Queue Retry
# =============================================================================

def test_t3_10_failed_resumable_enospc_queue_retry(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T3-10: Failed-Resumable ENOSPC Simulation & Queue Retry pass at queue tail."""
    ep1 = create_synthetic_video(tmp_path / "retry_ep1.mkv", num_frames=8, audio_tracks=1)
    ep2 = create_synthetic_video(tmp_path / "retry_ep2.mkv", num_frames=8, audio_tracks=1)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.pipeline import FailedResumableError
    call_count = 0
    real_run_pipeline = upscale.run_upscale_pipeline

    def mock_pipeline_flaky(input_path, *args, **kwargs):
        nonlocal call_count
        if "retry_ep1" in str(input_path):
            call_count += 1
            if call_count == 1:
                temp_segs = str(out_dir / ".temp_retry_ep1" / "upscaled_segs")
                raise FailedResumableError("Simulated disk exhaustion", 0, temp_segs)
            # Second call (retry pass): produce file
            (out_dir / "retry_ep1_4K.mkv").write_bytes(b"RETRY_EP1_4K")
            return str(out_dir / "retry_ep1_4K.mkv")
        # ep2 runs normally
        return real_run_pipeline(input_path=input_path, *args, **kwargs)

    with patch("upscale.run_upscale_pipeline", side_effect=mock_pipeline_flaky):
        exit_code, out, err = run_upscale_cli(
            ["--queue", str(ep1), str(ep2), "-o", str(out_dir)],
            monkeypatch,
            capsys,
        )

    assert exit_code == 0
    assert "RETRY PASS" in out
    assert (out_dir / "retry_ep1_4K.mkv").exists()
    assert (out_dir / "retry_ep2_4K.mkv").exists()
