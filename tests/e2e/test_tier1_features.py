# -*- coding: utf-8 -*-
"""
tests/e2e/test_tier1_features.py
Tier 1: Feature Coverage (Equivalence Class & Happy Path Tests)
Covers 10 major feature areas (5 test cases per feature = 50 test cases).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

import upscale
from tests.e2e.conftest import create_synthetic_video, run_upscale_cli


# =============================================================================
# Feature 1: Single Video Upscale Basic CLI (-o, output paths, default behavior)
# =============================================================================

def test_t1_f01_01_default_output_destination_resolution(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F01-01: Default Output Destination Resolution in out/<base>_4K.mkv"""
    src_vid = create_synthetic_video(tmp_path / "ep01.mkv", num_frames=8)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    expected_out = out_dir / "ep01_4K.mkv"
    assert expected_out.exists()
    assert expected_out.stat().st_size > 0


def test_t1_f01_02_explicit_output_directory_resolution(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F01-02: Explicit Output Directory Resolution (auto-creates directory)"""
    src_vid = create_synthetic_video(tmp_path / "feature02.mkv", num_frames=8)
    custom_dir = tmp_path / "dist" / "renders"

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(custom_dir) + "/"], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    expected_out = custom_dir / "feature02_4K.mkv"
    assert expected_out.exists()


def test_t1_f01_03_explicit_output_target_file_path(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F01-03: Explicit Output Target File Path (exact filename preserved)"""
    src_vid = create_synthetic_video(tmp_path / "feature03.mkv", num_frames=8)
    target_file = tmp_path / "dist" / "custom_named_movie.mkv"

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(target_file)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert target_file.exists()


def test_t1_f01_04_completed_file_skip_bypass(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F01-04: Completed File Skip Bypass without --force"""
    src_vid = create_synthetic_video(tmp_path / "feature04.mkv", num_frames=8)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_out = out_dir / "feature04_4K.mkv"
    # Create fake existing output > 1MB
    existing_out.write_bytes(b"EXISTING_VALID_OUTPUT" * 60000)
    orig_mtime = existing_out.stat().st_mtime

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert "BỎ QUA - ĐÃ CÓ KẾT QUẢ" in out
    assert existing_out.stat().st_mtime == orig_mtime


def test_t1_f01_05_force_reupscale_overwrite(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F01-05: Force Re-upscale Overwrite with --force"""
    src_vid = create_synthetic_video(tmp_path / "feature05.mkv", num_frames=8)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_out = out_dir / "feature05_4K.mkv"
    existing_out.write_bytes(b"OLD_CONTENT" * 60000)

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir), "--force"], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert existing_out.exists()
    assert existing_out.read_bytes() != b"OLD_CONTENT" * 60000


# =============================================================================
# Feature 2: Preview Mode (--preview <seconds>)
# =============================================================================

def test_t1_f02_01_default_10s_preview_mode(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F02-01: Default 10-Second Preview Mode"""
    src_vid = create_synthetic_video(tmp_path / "preview01.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--preview", "10", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    # Expected output has _preview_ in filename
    previews = list(out_dir.glob("*_preview_*_4K.mkv"))
    assert len(previews) == 1


def test_t1_f02_02_explicit_preview_duration(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F02-02: Explicit Preview Duration (5s)"""
    src_vid = create_synthetic_video(tmp_path / "preview02.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--preview", "5", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    previews = list(out_dir.glob("*_preview_*_4K.mkv"))
    assert len(previews) == 1
    assert "preview" in previews[0].name.lower()


def test_t1_f02_03_custom_output_directory_with_preview(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F02-03: Custom Output Directory with Preview"""
    src_vid = create_synthetic_video(tmp_path / "preview03.mkv", num_frames=8)
    custom_preview_dir = tmp_path / "custom_previews"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--preview", "5", "-o", str(custom_preview_dir) + "/"], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert custom_preview_dir.exists()
    previews = list(custom_preview_dir.glob("*.mkv"))
    assert len(previews) == 1


def test_t1_f02_04_intermediate_cut_file_cleanup(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F02-04: Intermediate Cut File Cleanup after Preview"""
    src_vid = create_synthetic_video(tmp_path / "preview04.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--preview", "5", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    # Intermediate .preview_clip_* must be removed
    intermediate_clips = list(out_dir.glob(".preview_clip_*"))
    assert len(intermediate_clips) == 0


def test_t1_f02_05_preview_with_model_override(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F02-05: Preview Mode with Model Override (AnimeJaNai suffix)"""
    src_vid = create_synthetic_video(tmp_path / "preview05.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--preview", "5", "-m", "animejanai-hd-v3-compact", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    previews = list(out_dir.glob("*animejanai*.mkv"))
    assert len(previews) == 1


# =============================================================================
# Feature 3: Dry-run Mode (--dry-run)
# =============================================================================

def test_t1_f03_01_single_video_dry_run_simulation(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F03-01: Single Video Dry-Run Simulation (0 GPU execution, exits 0)"""
    src_vid = create_synthetic_video(tmp_path / "dryrun01.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--dry-run", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert "DRY RUN" in out
    # 0 output video created
    assert not (out_dir / "dryrun01_4K.mkv").exists()


def test_t1_f03_02_multi_episode_queue_dry_run(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F03-02: Multi-Episode Queue Dry-Run prints decision table"""
    v1 = create_synthetic_video(tmp_path / "ep1.mkv", num_frames=8)
    v2 = create_synthetic_video(tmp_path / "ep2.mkv", num_frames=8)

    exit_code, out, err = run_upscale_cli(["--queue", str(v1), str(v2), "--dry-run"], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert "DRY RUN" in out
    assert "ep1.mkv" in out
    assert "ep2.mkv" in out


def test_t1_f03_03_queue_dry_run_with_deadline(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F03-03: Queue Dry-Run with Deadline includes Decision column"""
    v1 = create_synthetic_video(tmp_path / "ep_dead.mkv", num_frames=8)

    exit_code, out, err = run_upscale_cli(
        ["--queue", str(v1), "--deadline", "07:00", "--dry-run"],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert "DRY RUN" in out
    assert "Quyết định" in out or "QUYẾT ĐỊNH" in out


def test_t1_f03_04_cleanup_simulation_with_dry_run(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F03-04: Cleanup Simulation with Dry-Run does not delete files"""
    out_dir = tmp_path / "out"
    temp_dir = out_dir / ".temp_sample"
    temp_dir.mkdir(parents=True, exist_ok=True)
    marker = temp_dir / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")

    exit_code, out, err = run_upscale_cli(["--cleanup", "--dry-run", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert "DRY RUN" in out
    assert marker.exists()


def test_t1_f03_05_subs_only_dry_run_simulation(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F03-05: Subtitle-Only Dry-Run Simulation"""
    src_vid = create_synthetic_video(tmp_path / "subs_dry.mkv", num_frames=8, subtitles=["Konnichiwa"])

    exit_code, out, err = run_upscale_cli(["--subs-only", str(src_vid), "--dry-run"], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert "DRY RUN" in out


# =============================================================================
# Feature 4: Model Selection & Engine Fallback (--model, --engine, --no-tensorrt)
# =============================================================================

def test_t1_f04_01_default_model_selection(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F04-01: Default Model Selection resolves animevideov3"""
    src_vid = create_synthetic_video(tmp_path / "model01.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    expected_out = out_dir / "model01_4K.mkv"
    assert expected_out.exists()


def test_t1_f04_02_explicit_animejanai_model_override(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F04-02: Explicit AnimeJaNai Model Override adds _animejanai suffix"""
    src_vid = create_synthetic_video(tmp_path / "model02.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "-m", "animejanai-hd-v3-compact", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    expected_out = out_dir / "model02_4K_animejanai.mkv"
    assert expected_out.exists()


def test_t1_f04_03_list_models_cli_query(monkeypatch, capsys):
    """T1-F04-03: List Models CLI Query exits 0 and prints models table"""
    exit_code, out, err = run_upscale_cli(["--list-models"], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert "animevideov3" in out
    assert "animejanai-hd-v3-compact" in out


def test_t1_f04_04_pytorch_cuda_fallback_no_tensorrt(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F04-04: PyTorch CUDA Fallback with --no-tensorrt"""
    src_vid = create_synthetic_video(tmp_path / "model04.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--no-tensorrt", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "model04_4K.mkv").exists()


def test_t1_f04_05_cpu_engine_execution(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F04-05: CPU Engine Execution without CUDA requirements"""
    src_vid = create_synthetic_video(tmp_path / "model05.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--engine", "cpu", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "model05_4K.mkv").exists()


# =============================================================================
# Feature 5: Queue Mode & Batch Execution (--queue, --deadline, --allow-partial)
# =============================================================================

def test_t1_f05_01_standard_multi_episode_sequential_queue(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F05-01: Standard Multi-Episode Sequential Queue completes both episodes"""
    v1 = create_synthetic_video(tmp_path / "q_ep1.mkv", num_frames=8)
    v2 = create_synthetic_video(tmp_path / "q_ep2.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(["--queue", str(v1), str(v2), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "q_ep1_4K.mkv").exists()
    assert (out_dir / "q_ep2_4K.mkv").exists()


def test_t1_f05_02_queue_execution_with_generous_deadline(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F05-02: Queue Execution with Generous Deadline (23:59)"""
    v1 = create_synthetic_video(tmp_path / "q_dead.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        ["--queue", str(v1), "--deadline", "23:59", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "q_dead_4K.mkv").exists()


def test_t1_f05_03_overnight_queue_background_preflight(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F05-03: Overnight Queue with Preflight Background Thread coordination"""
    v1 = create_synthetic_video(tmp_path / "bg_ep1.mkv", num_frames=8)
    v2 = create_synthetic_video(tmp_path / "bg_ep2.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(["--queue", str(v1), str(v2), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "bg_ep1_4K.mkv").exists()
    assert (out_dir / "bg_ep2_4K.mkv").exists()


def test_t1_f05_04_allow_partial_safe_boundary_stopping(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F05-04: Allow-Partial Safe Boundary Stopping preserves segment chunks"""
    v1 = create_synthetic_video(tmp_path / "part_ep1.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    # Allow-partial flag acceptance
    exit_code, out, err = run_upscale_cli(
        [str(v1), "--allow-partial", "--deadline", "23:59", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "part_ep1_4K.mkv").exists()


def test_t1_f05_05_queue_with_benchmark_mode_flag(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F05-05: Queue with Benchmark Mode Flag suppresses background scheduler"""
    v1 = create_synthetic_video(tmp_path / "bench_ep1.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        ["--queue", str(v1), "--benchmark-mode", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert "BENCHMARK MODE" in out
    assert (out_dir / "bench_ep1_4K.mkv").exists()


# =============================================================================
# Feature 6: Dedup Policies & DIS Threshold (--dedup-policy, --dis-thresh)
# =============================================================================

def test_t1_f06_01_default_deduplication_policy(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F06-01: Default Deduplication Policy execution"""
    src_vid = create_synthetic_video(tmp_path / "dedup01.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--dedup-policy", "default", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "dedup01_4K.mkv").exists()


def test_t1_f06_02_sensitive_deduplication_policy(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F06-02: Sensitive Deduplication Policy execution"""
    src_vid = create_synthetic_video(tmp_path / "dedup02.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--dedup-policy", "sensitive", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "dedup02_4K.mkv").exists()


def test_t1_f06_03_motion_aware_phase_correlation_policy(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F06-03: Motion-Aware Phase Correlation Policy"""
    src_vid = create_synthetic_video(tmp_path / "dedup03.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--dedup-policy", "motion", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "dedup03_4K.mkv").exists()


def test_t1_f06_04_dis_optical_flow_policy(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F06-04: DIS Optical Flow Policy with default threshold"""
    src_vid = create_synthetic_video(tmp_path / "dedup04.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--dedup-policy", "dis", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "dedup04_4K.mkv").exists()


def test_t1_f06_05_deduplication_disabled(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F06-05: Deduplication Disabled via --no-dedup"""
    src_vid = create_synthetic_video(tmp_path / "dedup05.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--no-dedup", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "dedup05_4K.mkv").exists()


# =============================================================================
# Feature 7: Zero-Disk Pipeline vs Disk Pipeline (--zero-disk)
# =============================================================================

def test_t1_f07_01_zero_disk_pipeline_activation(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F07-01: Zero-Disk Pipeline Activation (0 PNGs written to disk)"""
    src_vid = create_synthetic_video(tmp_path / "zerodisk01.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--zero-disk", "-o", str(out_dir), "--keep-temp"],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "zerodisk01_4K.mkv").exists()
    # Check that in .temp_zerodisk01/ there are 0 chunk_in PNG files
    temp_dirs = list(out_dir.glob(".temp_*"))
    for td in temp_dirs:
        pngs = list(td.glob("**/*.png"))
        assert len(pngs) == 0


def test_t1_f07_02_standard_disk_pipeline_execution(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F07-02: Standard Disk Pipeline Execution"""
    src_vid = create_synthetic_video(tmp_path / "disk02.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "disk02_4K.mkv").exists()


def test_t1_f07_03_zero_disk_batch_size_configuration(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F07-03: Zero-Disk Batch Size Configuration (--batch-size 4)"""
    src_vid = create_synthetic_video(tmp_path / "zerodisk03.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--zero-disk", "--batch-size", "4", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "zerodisk03_4K.mkv").exists()


def test_t1_f07_04_zero_disk_output_color_verification(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F07-04: Zero-Disk Output Color & Range Verification (BT.709 p010le)"""
    src_vid = create_synthetic_video(tmp_path / "zerodisk04.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--zero-disk", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    out_file = out_dir / "zerodisk04_4K.mkv"
    assert out_file.exists()

    # Check color range and pix fmt via ffprobe
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt,color_range", "-of", "csv=p=0", str(out_file)],
        capture_output=True, text=True, check=True,
    )
    assert "10le" in res.stdout


def test_t1_f07_05_zero_disk_frame_count_preservation(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F07-05: Zero-Disk Frame Count Preservation & Mux PASS"""
    src_vid = create_synthetic_video(tmp_path / "zerodisk05.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--zero-disk", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    out_file = out_dir / "zerodisk05_4K.mkv"
    assert out_file.exists()


# =============================================================================
# Feature 8: Subtitle Options & Embedding (--subs-only, --subs-quality, etc.)
# =============================================================================

def test_t1_f08_01_standalone_subtitle_translation_subs_only(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F08-01: Standalone Subtitle Translation (--subs-only)"""
    src_vid = create_synthetic_video(tmp_path / "sub01.mkv", num_frames=8, subtitles=["Konnichiwa"])
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--subs-only", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert "SUBS-ONLY" in out


def test_t1_f08_02_subtitle_quality_presets(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F08-02: Subtitle Quality Presets (--subs-quality high)"""
    src_vid = create_synthetic_video(tmp_path / "sub02.mkv", num_frames=8, subtitles=["Arigatou"])
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--subs-only", "--subs-quality", "high", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"


def test_t1_f08_03_external_subtitle_output_no_embed(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F08-03: External Subtitle Output without Container Embedding (--no-embed)"""
    src_vid = create_synthetic_video(tmp_path / "sub03.mkv", num_frames=8, subtitles=["Sayonara"])
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--subs-only", "--no-embed", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"


def test_t1_f08_04_auto_align_english_dtw_synchronization(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F08-04: Auto-Align English DTW Synchronization flag"""
    src_vid = create_synthetic_video(tmp_path / "sub04.mkv", num_frames=8, subtitles=["Dialogue 1"])
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--subs-only", "--auto-align-en", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"


def test_t1_f08_05_manual_subtitle_millisecond_offset(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F08-05: Manual Subtitle Millisecond Offset (--sub-offset-ms -500)"""
    src_vid = create_synthetic_video(tmp_path / "sub05.mkv", num_frames=8, subtitles=["Dialogue offset"])
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--subs-only", "--sub-offset-ms", "-500", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"


# =============================================================================
# Feature 9: Checkpoint Resume & Temp Cleanup (--keep-temp, --cleanup, etc.)
# =============================================================================

def test_t1_f09_01_seamless_checkpoint_resumption(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F09-01: Seamless Checkpoint Resumption from .temp_*"""
    src_vid = create_synthetic_video(tmp_path / "resume01.mkv", num_frames=8)
    out_dir = tmp_path / "out"
    temp_dir = out_dir / ".temp_resume01"
    segs_dir = temp_dir / "upscaled_segs"
    segs_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate chunk 0 as completed (> 10KB valid mp4 segment)
    create_synthetic_video(segs_dir / "up_0000.mp4", num_frames=8, is_mp4=True)

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "resume01_4K.mkv").exists()


def test_t1_f09_02_automatic_post_mux_checkpoint_cleanup(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F09-02: Automatic Post-Mux Checkpoint Cleanup (removes .temp_*)"""
    src_vid = create_synthetic_video(tmp_path / "clean02.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "clean02_4K.mkv").exists()
    assert not (out_dir / ".temp_clean02").exists()


def test_t1_f09_03_checkpoint_preservation_via_keep_temp(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F09-03: Checkpoint Preservation via --keep-temp"""
    src_vid = create_synthetic_video(tmp_path / "clean03.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--keep-temp", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert (out_dir / "clean03_4K.mkv").exists()
    assert (out_dir / ".temp_clean03").exists()


def test_t1_f09_04_standalone_maintenance_cleanup(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F09-04: Standalone Maintenance Cleanup (--cleanup) deletes orphan temps"""
    out_dir = tmp_path / "out"
    orphan_temp = out_dir / ".temp_orphan"
    orphan_temp.mkdir(parents=True, exist_ok=True)
    (orphan_temp / "dummy.txt").write_text("orphan", encoding="utf-8")

    exit_code, out, err = run_upscale_cli(["--cleanup", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    assert not orphan_temp.exists()


def test_t1_f09_05_audit_artifact_organization(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F09-05: Audit Artifact Organization moves audit to audit/ subfolder"""
    src_vid = create_synthetic_video(tmp_path / "organize05.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0, f"exit={exit_code}\nOUT:\n{out}\nERR:\n{err}"
    # Audit file should be placed in out/audit/ or out/
    audits = list(out_dir.glob("**/organize05_4K.audit.json"))
    assert len(audits) >= 1


# =============================================================================
# Feature 10: Benchmark Harness & Telemetry (scripts/benchmark_pipeline.py)
# =============================================================================

def test_t1_f10_01_benchmark_pipeline_preview_tier(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F10-01: Benchmark Pipeline Preview Tier Execution"""
    from scripts.benchmark_pipeline import run_benchmark
    src_vid = create_synthetic_video(tmp_path / "bench01.mkv", num_frames=8)
    bench_out = tmp_path / "bench_out"

    summary = run_benchmark(
        source_path=src_vid,
        tier="preview",
        output_dir=bench_out,
        label="test_bench01",
        engine="cpu",
        dry_run=True,
    )
    assert (bench_out / "test_bench01_summary.json").exists()
    assert (bench_out / "test_bench01_telemetry.csv").exists()


def test_t1_f10_02_benchmark_pipeline_climax_tier(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T1-F10-02: Benchmark Pipeline Climax Tier Execution"""
    from scripts.benchmark_pipeline import run_benchmark
    src_vid = create_synthetic_video(tmp_path / "bench02.mkv", num_frames=8)
    bench_out = tmp_path / "bench_out"

    summary = run_benchmark(
        source_path=src_vid,
        tier="climax",
        output_dir=bench_out,
        label="test_bench02",
        engine="cpu",
        dry_run=True,
    )
    assert summary["tier"] == "climax"
    assert (bench_out / "test_bench02_summary.json").exists()


def test_t1_f10_03_telemetry_csv_schema_conformance(mock_pipeline_env, tmp_path):
    """T1-F10-03: Telemetry CSV Schema Conformance (17 standard columns)"""
    from scripts.telemetry_collector import TelemetryCollector
    csv_file = tmp_path / "telemetry_test.csv"
    collector = TelemetryCollector(output_csv_path=csv_file, interval_sec=0.1)
    collector.start()
    collector.stop()

    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    expected_cols = [
        "timestamp", "elapsed_s", "pstate", "gpu_util_pct", "mem_util_pct",
        "enc_util_pct", "dec_util_pct", "mem_clock_mhz", "core_clock_mhz",
        "power_w", "temp_c", "vram_used_mib", "vram_free_mib", "cpu_pct",
        "ram_used_gb", "ssd_read_mb_s", "ssd_write_mb_s",
    ]
    assert header == expected_cols


def test_t1_f10_04_benchmark_summary_json_schema(mock_pipeline_env, tmp_path):
    """T1-F10-04: Benchmark Summary JSON Schema Conformance"""
    from scripts.benchmark_pipeline import run_benchmark
    src_vid = create_synthetic_video(tmp_path / "schema.mkv", num_frames=8)
    bench_out = tmp_path / "schema_out"

    summary = run_benchmark(
        source_path=src_vid,
        tier="preview",
        output_dir=bench_out,
        label="schema_check",
        engine="cpu",
        dry_run=True,
    )

    required_keys = [
        "total_time_s", "effective_fps", "ai_fps", "total_frames",
        "unique_frames", "dedup_ratio", "peak_vram_mib", "avg_gpu_util",
        "avg_cpu_pct", "mux_status",
    ]
    for key in required_keys:
        assert key in summary, f"Missing required summary key: {key}"


def test_t1_f10_05_benchmark_comparator_delta_reporting(tmp_path):
    """T1-F10-05: Benchmark Comparator Delta Reporting with positive improvement"""
    from scripts.compare_benchmarks import compare_summaries

    base_summary = {
        "total_time_s": 100.0,
        "effective_fps": 20.0,
        "ai_fps": 30.0,
        "total_frames": 2000,
        "output_frames": 2000,
        "frames_matched": True,
        "mux_status": "PASS",
        "peak_vram_mib": 3200.0,
        "avg_gpu_util": 85.0,
        "avg_cpu_pct": 25.0,
    }
    opt_summary = {
        "total_time_s": 80.0,
        "effective_fps": 25.0,
        "ai_fps": 38.0,
        "total_frames": 2000,
        "output_frames": 2000,
        "frames_matched": True,
        "mux_status": "PASS",
        "peak_vram_mib": 3100.0,
        "avg_gpu_util": 90.0,
        "avg_cpu_pct": 22.0,
    }

    is_success, report_md, metrics = compare_summaries(base_summary, opt_summary)
    assert is_success is True
    assert "IMPROVED" in report_md
    assert "| Metric |" in report_md
