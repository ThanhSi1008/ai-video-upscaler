# -*- coding: utf-8 -*-
"""
tests/e2e/test_tier2_boundaries.py
Tier 2: Boundary, Corner & Error Cases
Covers 10 major feature areas (5 test cases per feature = 50 test cases).
"""

from __future__ import annotations

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
# Feature 1: Single Video Basic CLI Boundaries & Errors
# =============================================================================

def test_t2_f01_01_zero_input_arguments_error(monkeypatch, capsys):
    """T2-F01-01: Zero Input Arguments Error (exits 1 with clear message)"""
    exit_code, out, err = run_upscale_cli([], monkeypatch, capsys)
    assert exit_code == 1
    assert "Vui lòng cung cấp ít nhất một file video đầu vào" in (out + err)


def test_t2_f01_02_nonexistent_source_video_path(monkeypatch, capsys):
    """T2-F01-02: Nonexistent Source Video Path (exits 1)"""
    exit_code, out, err = run_upscale_cli(["nonexistent_source_xyz123.mkv"], monkeypatch, capsys)
    assert exit_code == 1
    assert "Không tìm thấy file video nguồn" in (out + err)


def test_t2_f01_03_corrupt_zero_byte_source_file(tmp_path, monkeypatch, capsys):
    """T2-F01-03: Corrupt Zero-Byte Source File (probe fails cleanly, exits 1)"""
    corrupt_file = tmp_path / "empty_corrupt.mkv"
    corrupt_file.write_bytes(b"")

    exit_code, out, err = run_upscale_cli([str(corrupt_file)], monkeypatch, capsys)
    assert exit_code == 1


def test_t2_f01_04_non_video_file_input(tmp_path, monkeypatch, capsys):
    """T2-F01-04: Non-Video File Input (ASCII text, exits 1)"""
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("This is not a video file.", encoding="utf-8")

    exit_code, out, err = run_upscale_cli([str(txt_file)], monkeypatch, capsys)
    assert exit_code == 1


def test_t2_f01_05_special_characters_in_path(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F01-05: Space-Delimited & Special Character Path Handling"""
    special_name = "clip [v2] (test) #1.mkv"
    src_vid = create_synthetic_video(tmp_path / special_name, num_frames=8)
    out_dir = tmp_path / "out special [dir]"

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0
    created_files = list(out_dir.glob("*.mkv"))
    assert len(created_files) == 1


# =============================================================================
# Feature 2: Preview Mode Boundaries
# =============================================================================

def test_t2_f02_01_preview_duration_exceeding_media_length(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F02-01: Preview Duration Exceeding Media Length (clamped to duration)"""
    src_vid = create_synthetic_video(tmp_path / "short_clip.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--preview", "100", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0
    previews = list(out_dir.glob("*_preview_*_4K.mkv"))
    assert len(previews) == 1


def test_t2_f02_02_zero_duration_preview_argument(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F02-02: Zero-Duration Preview Argument (runs full video)"""
    src_vid = create_synthetic_video(tmp_path / "clip0.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--preview", "0", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0
    # Runs full upscale: clip0_4K.mkv
    assert (out_dir / "clip0_4K.mkv").exists()


def test_t2_f02_03_negative_preview_duration(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F02-03: Negative Preview Duration Argument (handled gracefully)"""
    src_vid = create_synthetic_video(tmp_path / "clip_neg.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--preview", "-5", "-o", str(out_dir)], monkeypatch, capsys)
    # Evaluates preview_seconds <= 0 -> runs full upscale cleanly
    assert exit_code == 0
    assert (out_dir / "clip_neg_4K.mkv").exists()


def test_t2_f02_04_non_integer_preview_argument(tmp_path, monkeypatch, capsys):
    """T2-F02-04: Non-Integer Preview Argument (argparse exit 2)"""
    src_vid = create_synthetic_video(tmp_path / "clip_str.mkv", num_frames=8)

    exit_code, out, err = run_upscale_cli([str(src_vid), "--preview", "ten_sec"], monkeypatch, capsys)
    assert exit_code == 2


def test_t2_f02_05_preview_mode_silent_no_subs(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F02-05: Preview Mode on Video with No Audio and No Subtitles"""
    src_vid = create_synthetic_video(tmp_path / "silent.mkv", num_frames=8, audio_tracks=0)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--preview", "5", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0
    previews = list(out_dir.glob("*_preview_*_4K.mkv"))
    assert len(previews) == 1


# =============================================================================
# Feature 3: Dry-run Mode Boundaries
# =============================================================================

def test_t2_f03_01_dry_run_nonexistent_source(monkeypatch, capsys):
    """T2-F03-01: Dry-Run with Nonexistent Source Video (estimates frames in dry-run table)"""
    exit_code, out, err = run_upscale_cli(["missing_vid_for_dryrun.mkv", "--dry-run"], monkeypatch, capsys)
    assert exit_code == 0
    assert "DRY RUN" in out or "BẢNG QUYẾT ĐỊNH" in out


def test_t2_f03_02_dry_run_invalid_deadline_format(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F03-02: Dry-Run with Invalid Deadline Format (warns and falls back)"""
    src_vid = create_synthetic_video(tmp_path / "vid_dead_bad.mkv", num_frames=8)

    exit_code, out, err = run_upscale_cli([str(src_vid), "--deadline", "25:99", "--dry-run"], monkeypatch, capsys)
    assert exit_code == 0
    assert "Không parse được deadline" in (out + err)


def test_t2_f03_03_dry_run_never_creates_gpu_lock(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F03-03: Dry-Run Never Creates GPU Lock File"""
    src_vid = create_synthetic_video(tmp_path / "vid_lock_check.mkv", num_frames=8)
    lock_file = Path(".gpu_active.lock")
    assert not lock_file.exists()

    exit_code, out, err = run_upscale_cli([str(src_vid), "--dry-run"], monkeypatch, capsys)
    assert exit_code == 0
    assert not lock_file.exists()


def test_t2_f03_04_dry_run_partial_and_deadline_combined(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F03-04: Dry-Run with Combined Partial and Deadline Flags"""
    src_vid = create_synthetic_video(tmp_path / "vid_partial.mkv", num_frames=8)

    exit_code, out, err = run_upscale_cli(
        ["--queue", str(src_vid), "--deadline", "00:01", "--allow-partial", "--dry-run"],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert "DRY RUN" in out


def test_t2_f03_05_empty_queue_dry_run(monkeypatch, capsys):
    """T2-F03-05: Empty Queue Dry-Run (argparse exit 2)"""
    exit_code, out, err = run_upscale_cli(["--queue", "--dry-run"], monkeypatch, capsys)
    assert exit_code == 2


# =============================================================================
# Feature 4: Model Selection & Fallback Boundaries
# =============================================================================

def test_t2_f04_01_unregistered_model_rejection(tmp_path, monkeypatch, capsys):
    """T2-F04-01: Unregistered Model ID Rejection (raises ValueError / exits 1)"""
    src_vid = create_synthetic_video(tmp_path / "vid_bad_model.mkv", num_frames=8)

    exit_code, out, err = run_upscale_cli([str(src_vid), "-m", "nonexistent_model_xyz"], monkeypatch, capsys)
    assert exit_code == 1
    assert "không tồn tại trong models/models.yaml" in (out + err)


def test_t2_f04_02_invalid_engine_choice_rejection(tmp_path, monkeypatch, capsys):
    """T2-F04-02: Invalid Engine Choice Rejection (argparse exit 2)"""
    src_vid = create_synthetic_video(tmp_path / "vid_bad_engine.mkv", num_frames=8)

    exit_code, out, err = run_upscale_cli([str(src_vid), "--engine", "opencl"], monkeypatch, capsys)
    assert exit_code == 2


def test_t2_f04_03_checkpoint_resume_model_conflict_trap(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F04-03: Checkpoint Resume Model Conflict Trap (prevents mixing models)"""
    src_vid = create_synthetic_video(tmp_path / "vid_conflict.mkv", num_frames=8)
    out_dir = tmp_path / "out"
    temp_dir = out_dir / ".temp_vid_conflict"
    temp_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = temp_dir / "manifest.json"
    manifest_file.write_text(json.dumps({"model_id": "animevideov3"}), encoding="utf-8")

    # Try running with animejanai without --force
    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "-m", "animejanai-hd-v3-compact", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 1
    assert "Xung đột mô hình" in (out + err)


def test_t2_f04_04_checkpoint_model_conflict_override_via_force(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F04-04: Checkpoint Model Conflict Override via --force"""
    src_vid = create_synthetic_video(tmp_path / "vid_force_override.mkv", num_frames=8)
    out_dir = tmp_path / "out"
    temp_dir = out_dir / ".temp_vid_force_override"
    temp_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = temp_dir / "manifest.json"
    manifest_file.write_text(json.dumps({"model_id": "animevideov3"}), encoding="utf-8")

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "-m", "animejanai-hd-v3-compact", "--force", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert (out_dir / "vid_force_override_4K_animejanai.mkv").exists()


def test_t2_f04_05_alias_mapping_for_legacy_model_names(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F04-05: Alias Mapping for Legacy Model Names (realesr-animevideov3 -> animevideov3)"""
    src_vid = create_synthetic_video(tmp_path / "vid_alias.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "-m", "realesr-animevideov3", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert (out_dir / "vid_alias_4K.mkv").exists()


# =============================================================================
# Feature 5: Queue Mode & Batch Execution Boundaries
# =============================================================================

def test_t2_f05_01_deadline_skip_without_allow_partial(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F05-01: Deadline Skip without Allow-Partial exits 1 with summary"""
    src_vid = create_synthetic_video(tmp_path / "skip_vid.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    # Deadline of 00:01 in the past causes tight budget skip
    with patch("upscale.check_deadline_and_eta", return_value=(False, False, "Quá hạn")):
        exit_code, out, err = run_upscale_cli(
            ["--queue", str(src_vid), "--deadline", "00:01", "-o", str(out_dir)],
            monkeypatch,
            capsys,
        )
    assert exit_code == 1
    assert "DEADLINE SKIP" in out or "CHƯA XONG" in out


def test_t2_f05_02_queue_containing_mixed_valid_and_invalid(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F05-02: Queue Containing Mixed Valid and Invalid Episodes"""
    v_valid = create_synthetic_video(tmp_path / "valid.mkv", num_frames=8)
    v_invalid = tmp_path / "missing.mkv"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    exit_code, out, err = run_upscale_cli(
        ["--queue", str(v_valid), str(v_invalid), "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 1
    # Valid completed, missing causes failure message
    assert (out_dir / "valid_4K.mkv").exists()
    assert "Không tìm thấy file video nguồn" in (out + err)


def test_t2_f05_03_empty_queue_flag(monkeypatch, capsys):
    """T2-F05-03: Empty Queue Flag (argparse exit 2)"""
    exit_code, out, err = run_upscale_cli(["--queue"], monkeypatch, capsys)
    assert exit_code == 2


def test_t2_f05_04_user_sigint_interruption_in_queue(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F05-04: User SIGINT Interruption (Ctrl+C) Handling in Queue"""
    src_vid = create_synthetic_video(tmp_path / "sigint_vid.mkv", num_frames=8)

    with patch("upscale.run_upscale_pipeline", side_effect=KeyboardInterrupt):
        exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(tmp_path)], monkeypatch, capsys)
    assert exit_code == 130


def test_t2_f05_05_failed_resumable_enospc_queue_retry(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F05-05: Failed-Resumable ENOSPC Simulation & Queue Retry Pass"""
    v1 = create_synthetic_video(tmp_path / "enospc_ep1.mkv", num_frames=8)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.pipeline import FailedResumableError
    call_count = 0

    def mock_pipeline_flaky(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            temp_segs = str(out_dir / ".temp_enospc_ep1" / "upscaled_segs")
            raise FailedResumableError("Simulated disk full (ENOSPC)", 0, temp_segs)
        return str(out_dir / "enospc_ep1_4K.mkv")

    with patch("upscale.run_upscale_pipeline", side_effect=mock_pipeline_flaky):
        # Create output file on retry
        (out_dir / "enospc_ep1_4K.mkv").write_bytes(b"RETRY_SUCCESS_4K")
        exit_code, out, err = run_upscale_cli(["--queue", str(v1), "-o", str(out_dir)], monkeypatch, capsys)

    assert "RETRY PASS" in out


# =============================================================================
# Feature 6: Dedup Policies & DIS Threshold Boundaries
# =============================================================================

def test_t2_f06_01_invalid_dedup_policy_name_rejection(tmp_path, monkeypatch, capsys):
    """T2-F06-01: Invalid Dedup Policy Name Rejection (argparse exit 2)"""
    src_vid = create_synthetic_video(tmp_path / "vid_bad_policy.mkv", num_frames=8)

    exit_code, out, err = run_upscale_cli([str(src_vid), "--dedup-policy", "ultra_aggressive"], monkeypatch, capsys)
    assert exit_code == 2


def test_t2_f06_02_custom_dis_thresh_flag_conversion(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F06-02: Custom DIS Threshold Flag Conversion (--dis-thresh 0.2 -> dis-0.2)"""
    src_vid = create_synthetic_video(tmp_path / "vid_dis_thresh.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--dis-thresh", "0.2", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0
    assert (out_dir / "vid_dis_thresh_4K.mkv").exists()


def test_t2_f06_03_custom_drift_delta_flag_conversion(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F06-03: Custom Drift Delta Flag Conversion (--drift-delta 0.50 -> drift-0.5)"""
    src_vid = create_synthetic_video(tmp_path / "vid_drift.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--drift-delta", "0.50", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0
    assert (out_dir / "vid_drift_4K.mkv").exists()


def test_t2_f06_04_synthetic_100pct_static_video_clip(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F06-04: Synthetic 100% Static Video Clip deduplication"""
    # 8 identical solid color frames
    src_vid = create_synthetic_video(
        tmp_path / "static_still.mkv",
        num_frames=8,
        video_filter="color=c=blue:size=320x180:rate=24",
    )
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0
    assert (out_dir / "static_still_4K.mkv").exists()


def test_t2_f06_05_dis_flow_extreme_zero_threshold(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F06-05: DIS Flow Extreme Zero Threshold rejection (--dis-thresh 0.0)"""
    src_vid = create_synthetic_video(tmp_path / "vid_dis_zero.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--dis-thresh", "0.0", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 1
    assert "Chính sách dedup không hợp lệ" in (out + err)


# =============================================================================
# Feature 7: Zero-Disk Pipeline Boundaries
# =============================================================================

def test_t2_f07_01_zero_disk_with_cpu_engine_incompatibility(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F07-01: Zero-Disk with CPU Engine safely falls back to standard execution"""
    src_vid = create_synthetic_video(tmp_path / "zd_cpu.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--zero-disk", "--engine", "cpu", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert (out_dir / "zd_cpu_4K.mkv").exists()


def test_t2_f07_02_zero_disk_with_pytorch_fallback_no_tensorrt(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F07-02: Zero-Disk with PyTorch Fallback (--no-tensorrt)"""
    src_vid = create_synthetic_video(tmp_path / "zd_notrt.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--zero-disk", "--no-tensorrt", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert (out_dir / "zd_notrt_4K.mkv").exists()


def test_t2_f07_03_zero_disk_output_sanity_guard_failure(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F07-03: Zero-Disk Output Sanity Guard Failure Trigger (fails with code 1)"""
    src_vid = create_synthetic_video(tmp_path / "zd_sanity_fail.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    # Mock output sanity to fail
    failed_sanity = {
        "passed": False,
        "delta_y": 12.5,
        "mean_y_src": 100.0,
        "mean_y_out": 150.0,
        "corr": 0.85,
        "max_delta_y": 6.0,
        "min_corr": 0.98,
        "elapsed_ms": 2.0,
    }
    with patch("src.output_sanity.check_frame_sanity_arrays", return_value=failed_sanity):
        exit_code, out, err = run_upscale_cli(
            [str(src_vid), "--zero-disk", "-o", str(out_dir)],
            monkeypatch,
            capsys,
        )
    assert exit_code == 1
    assert "CHỐT CHẶN ĐẦU RA THẤT BẠI" in (out + err)


def test_t2_f07_04_non_1080p_resolution_under_zero_disk(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F07-04: Non-1080p Resolution Input under Zero-Disk (160x90)"""
    src_vid = create_synthetic_video(tmp_path / "zd_small.mkv", num_frames=8, width=160, height=90)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--zero-disk", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert (out_dir / "zd_small_4K.mkv").exists()


def test_t2_f07_05_encoder_subprocess_premature_pipe_disconnect(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F07-05: Encoder Subprocess Premature Pipe Disconnect Handling"""
    src_vid = create_synthetic_video(tmp_path / "zd_pipe_break.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    import src.trt_zero_copy_upscaler as trt_mod
    import src.cuda_engine as ce

    # Clear cached upscaler on cuda_engine so new patched method is used
    engine = ce.get_cuda_inference_engine()
    if hasattr(engine, "_zero_disk_upscaler"):
        delattr(engine, "_zero_disk_upscaler")
    engine._zero_disk_key = None

    def mock_broken_pipe(self, *args, **kwargs):
        raise BrokenPipeError("Pipe disconnected by encoder")

    from tests.e2e.conftest import FastMockPipeUpscaler

    with patch.object(trt_mod.DoubleBufferedPipeUpscaler, "process_frames_to_pipe", mock_broken_pipe), \
         patch.object(FastMockPipeUpscaler, "process_frames_to_pipe", mock_broken_pipe):
        exit_code, out, err = run_upscale_cli(
            [str(src_vid), "--engine", "cuda", "--zero-disk", "-o", str(out_dir)],
            monkeypatch,
            capsys,
        )
    assert exit_code == 1
    assert "Pipe disconnected by encoder" in (out + err)


# =============================================================================
# Feature 8: Subtitle Options & Embedding Boundaries
# =============================================================================

def test_t2_f08_01_missing_japanese_subtitle_and_audio_failure(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F08-01: Missing Japanese Subtitle and Audio Failure (exits 1)"""
    # Video with 0 audio and 0 subtitles
    src_vid = create_synthetic_video(tmp_path / "no_sub_no_audio.mkv", num_frames=8, audio_tracks=0)
    out_dir = tmp_path / "out"

    with patch("src.subtitles.find_or_extract_subtitles", return_value=(None, None)):
        exit_code, out, err = run_upscale_cli(
            ["--subs-only", str(src_vid), "-o", str(out_dir)],
            monkeypatch,
            capsys,
        )
    assert exit_code == 1
    assert "Không tìm thấy hoặc không thể trích xuất phụ đề" in (out + err)


def test_t2_f08_02_untargeted_manual_subtitle_offset_in_multi_queue(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F08-02: Untargeted Manual Subtitle Offset in Multi-Video Queue logs warning"""
    v1 = create_synthetic_video(tmp_path / "q_sub1.mkv", num_frames=8, subtitles=["Sub 1"])
    v2 = create_synthetic_video(tmp_path / "q_sub2.mkv", num_frames=8, subtitles=["Sub 2"])
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        ["--queue", str(v1), str(v2), "--sub-offset-ms", "-5000", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert "QUEUE OFFSET WARNING" in out


def test_t2_f08_03_targeted_manual_subtitle_offset_in_queue(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F08-03: Targeted Manual Subtitle Offset in Queue"""
    v1 = create_synthetic_video(tmp_path / "target_ep1.mkv", num_frames=8, subtitles=["Ep1"])
    v2 = create_synthetic_video(tmp_path / "target_ep2.mkv", num_frames=8, subtitles=["Ep2"])
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        ["--queue", str(v1), str(v2), "--sub-offset-ms", "-11154", "--sub-offset-target", "target_ep2", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert (out_dir / "target_ep1_4K.mkv").exists()
    assert (out_dir / "target_ep2_4K.mkv").exists()


def test_t2_f08_04_partial_subtitle_embedding_guard(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F08-04: Partial Subtitle (.partial.srt) Embedding Guard"""
    src_vid = create_synthetic_video(tmp_path / "partial_sub_clip.mkv", num_frames=8, subtitles=["Incomplete line"])
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # If partial srt exists, it must not be embedded
    partial_srt = out_dir / "partial_sub_clip.partial.srt"
    partial_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nPartial line\n", encoding="utf-8")

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0


def test_t2_f08_05_sound_cues_drop_mode_filtering(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F08-05: Sound Cues Drop Mode Filtering (--sound-cues drop)"""
    src_vid = create_synthetic_video(tmp_path / "cues_drop.mkv", num_frames=8, subtitles=["(sigh) Hello (gasp)"])
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "--subs-only", "--sound-cues", "drop", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0


# =============================================================================
# Feature 9: Checkpoint Resume & Temp Cleanup Boundaries
# =============================================================================

def test_t2_f09_01_corrupted_truncated_checkpoint_segment_handling(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F09-01: Corrupted Truncated Checkpoint Segment (< 10000 bytes) ignored and re-upscaled"""
    src_vid = create_synthetic_video(tmp_path / "corrupt_seg.mkv", num_frames=8)
    out_dir = tmp_path / "out"
    temp_dir = out_dir / ".temp_corrupt_seg"
    segs_dir = temp_dir / "upscaled_segs"
    segs_dir.mkdir(parents=True, exist_ok=True)

    # Write corrupt segment with size 512 bytes
    corrupt_chunk = segs_dir / "up_0000.mp4"
    corrupt_chunk.write_bytes(b"CORRUPT_DATA" * 40)

    exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0
    assert (out_dir / "corrupt_seg_4K.mkv").exists()


def test_t2_f09_02_force_fresh_start_no_resume(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F09-02: Force Fresh Start Disabling Resume (--no-resume)"""
    src_vid = create_synthetic_video(tmp_path / "no_res_clip.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli([str(src_vid), "--no-resume", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0
    assert (out_dir / "no_res_clip_4K.mkv").exists()


def test_t2_f09_03_cleanup_command_on_clean_directory(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F09-03: Cleanup Command on Clean Directory reports 0 freed"""
    clean_dir = tmp_path / "empty_dir"
    clean_dir.mkdir(parents=True, exist_ok=True)

    exit_code, out, err = run_upscale_cli(["--cleanup", "-o", str(clean_dir)], monkeypatch, capsys)
    assert exit_code == 0
    assert "0 thư mục/symlink tạm" in out or "0.0 B" in out or "Đã dọn dẹp" in out


def test_t2_f09_04_broken_orphaned_symlink_cleanup(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F09-04: Broken Orphaned Symlink Cleanup unlinks safely"""
    out_dir = tmp_path / "out_symlink"
    out_dir.mkdir(parents=True, exist_ok=True)
    broken_link = out_dir / ".temp_broken_link"

    try:
        # Create a broken symlink if OS permissions allow
        os.symlink(tmp_path / "nonexistent_target", broken_link)
    except (OSError, NotImplementedError):
        # On Windows without developer mode, fallback to creating regular temp dir
        broken_link.mkdir(parents=True, exist_ok=True)

    exit_code, out, err = run_upscale_cli(["--cleanup", "-o", str(out_dir)], monkeypatch, capsys)
    assert exit_code == 0
    assert not broken_link.exists()


def test_t2_f09_05_mux_failure_preserves_checkpoint(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """T2-F09-05: Mux Failure Preserves Checkpoint for Diagnosis"""
    src_vid = create_synthetic_video(tmp_path / "mux_fail_vid.mkv", num_frames=8)
    out_dir = tmp_path / "out"

    with patch("src.mux_audit.verify_mux_integrity", side_effect=RuntimeError("Simulated Mux Failure")):
        exit_code, out, err = run_upscale_cli([str(src_vid), "-o", str(out_dir)], monkeypatch, capsys)

    assert exit_code == 1
    # Checkpoint temp folder must remain on disk for inspection
    assert (out_dir / ".temp_mux_fail_vid").exists()


# =============================================================================
# Feature 10: Benchmark Harness & Comparator Boundaries
# =============================================================================

def test_t2_f10_01_benchmark_harness_invalid_tier_selection(tmp_path):
    """T2-F10-01: Benchmark Harness Invalid Tier Selection"""
    from scripts.benchmark_pipeline import run_benchmark
    src_vid = create_synthetic_video(tmp_path / "bench_bad_tier.mkv", num_frames=8)

    with pytest.raises(ValueError, match="Invalid tier"):
        run_benchmark(
            source_path=src_vid,
            tier="hyper_fast_nonexistent",
            output_dir=tmp_path / "out",
        )


def test_t2_f10_02_benchmark_harness_nonexistent_source_media(tmp_path):
    """T2-F10-02: Benchmark Harness Nonexistent Source Media"""
    from scripts.benchmark_pipeline import main
    # Call main with missing file
    exit_code = main(["nonexistent_source_for_bench.mkv", "--tier", "preview", "--output-dir", str(tmp_path), "--label", "test_bench"])
    assert exit_code == 1


def test_t2_f10_03_comparator_performance_regression_detection():
    """T2-F10-03: Comparator Performance Regression Detection (returns False)"""
    from scripts.compare_benchmarks import compare_summaries

    base_summary = {
        "total_time_s": 100.0,
        "effective_fps": 30.0,
        "ai_fps": 40.0,
        "total_frames": 1000,
        "output_frames": 1000,
        "frames_matched": True,
        "mux_status": "PASS",
    }
    # Optimized has lower FPS by > 5% (severe regression)
    regressed_summary = {
        "total_time_s": 150.0,
        "effective_fps": 20.0,
        "ai_fps": 25.0,
        "total_frames": 1000,
        "output_frames": 1000,
        "frames_matched": True,
        "mux_status": "PASS",
    }

    is_success, report_md, metrics = compare_summaries(base_summary, regressed_summary, tolerance_pct=5.0)
    assert is_success is False
    assert "REGRESSED" in report_md


def test_t2_f10_04_comparator_frame_count_mismatch_detection():
    """T2-F10-04: Comparator Frame Count Mismatch Detection (asserts failure)"""
    from scripts.compare_benchmarks import compare_summaries

    base_summary = {
        "total_time_s": 100.0,
        "effective_fps": 20.0,
        "total_frames": 4374,
        "output_frames": 4374,
        "frames_matched": True,
        "mux_status": "PASS",
    }
    mismatch_summary = {
        "total_time_s": 90.0,
        "effective_fps": 22.0,
        "total_frames": 4372,  # 2 frames missing!
        "output_frames": 4372,
        "frames_matched": True,
        "mux_status": "PASS",
    }

    is_success, report_md, metrics = compare_summaries(base_summary, mismatch_summary)
    assert is_success is False
    assert "Frame count mismatch" in report_md


def test_t2_f10_05_comparator_corrupted_missing_summary_json(tmp_path):
    """T2-F10-05: Comparator Corrupted / Missing Summary JSON CLI call exits 1"""
    from scripts.compare_benchmarks import main

    corrupt_json = tmp_path / "corrupt.json"
    corrupt_json.write_text("{ this is not valid json }", encoding="utf-8")
    valid_json = tmp_path / "valid.json"
    valid_json.write_text("{}", encoding="utf-8")

    exit_code = main([str(valid_json), str(corrupt_json)])
    assert exit_code == 1
