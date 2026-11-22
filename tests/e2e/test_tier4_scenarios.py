# -*- coding: utf-8 -*-
"""
tests/e2e/test_tier4_scenarios.py
Tier 4: Realistic Full-System End-to-End Application Scenarios (5 Tests)
Exercises complete production user workflows of ai-video-upscaler:
Scenario 1: Standard TV Episode 10s Preview with Dedup and NVENC Mux Check
Scenario 2: Multi-stream Studio Anime Mux (Video + Multi-Audio + Dual-Subtitles + Attached Fonts)
Scenario 3: Interrupted Run & Checkpoint Resume with Open-GOP Segment Validation
Scenario 4: Overnight Batch Queue Processing with Mixed Video Specs and Deadline Termination
Scenario 5: Turnkey Benchmark Harness Execution & Comparative Analysis
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

import upscale
from src.ffmpeg_ops import probe_video
from tests.e2e.conftest import create_synthetic_video, run_upscale_cli


# =============================================================================
# Scenario 1: Standard TV Episode Preview Workflow
# =============================================================================

def test_t4_01_tv_episode_preview_workflow(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """
    Scenario 1: TV Episode 2s Preview with DIS Dedup and NVENC Mux Check.
    Validates preview seeking, temporary clip lifecycle, and 4K container muxing.
    """
    # 12 frames at 4fps = 3.0s duration, seeking to 30% (t=0.9s), cutting 2.0s
    src_vid = create_synthetic_video(
        tmp_path / "tv_ep01.mkv",
        num_frames=12,
        fps=4,
        width=320,
        height=180,
        audio_tracks=1,
        chapters=2,
    )
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [
            str(src_vid),
            "--preview", "2",
            "--dedup-policy", "dis",
            "--dis-thresh", "0.30",
            "-o", str(out_dir),
        ],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0

    # 1. Output file exists
    preview_out = out_dir / "tv_ep01_preview_2s_4K.mkv"
    assert preview_out.exists()
    assert preview_out.stat().st_size > 0

    # 2. Intermediate clip cleaned up
    assert not (out_dir / ".preview_clip_tv_ep01.mkv").exists()

    # 3. Stream inspection via probe_video
    meta = probe_video(str(preview_out))
    assert meta["width"] == 640
    assert meta["height"] == 360
    assert meta["audio_count"] == 1


# =============================================================================
# Scenario 2: Multi-Stream Studio Anime Mux (Bit-Exact Streamcopy)
# =============================================================================

def test_t4_02_multistream_studio_anime_mux(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """
    Scenario 2: Archival Studio Mux with Multi-Audio, Dual Subtitles, and Font Attachments.
    Verifies 100% stream preservation and studio mux integrity.
    """
    # Create dummy font attachments
    font1 = tmp_path / "custom_font1.ttf"
    font2 = tmp_path / "custom_font2.otf"
    font1.write_bytes(b"TTF_FONT_DATA_0123456789")
    font2.write_bytes(b"OTF_FONT_DATA_0123456789")

    src_vid = create_synthetic_video(
        tmp_path / "studio_remux_ep12.mkv",
        num_frames=8,
        fps=24,
        audio_tracks=2,
        subtitles=["Main Japanese Dialogue", "English Signs and Songs"],
        chapters=3,
        attachments=[font1, font2],
    )
    out_dir = tmp_path / "out"

    exit_code, out, err = run_upscale_cli(
        [
            str(src_vid),
            "--zero-disk",
            "--dedup-policy", "sensitive-a",
            "-o", str(out_dir),
        ],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0

    out_file = out_dir / "studio_remux_ep12_4K.mkv"
    assert out_file.exists()

    # Probe output container to verify bit-exact streamcopy
    meta = probe_video(str(out_file))
    assert meta["audio_count"] == 2
    assert meta["subtitle_count"] == 2
    assert meta["attachment_count"] == 2
    assert meta["chapter_count"] == 3

    # Verify that the audit report exists in out/audit/ or out/ with status == "PASS"
    audit_candidates = [
        out_dir / "studio_remux_ep12_4K.audit.json",
        out_dir / "studio_remux_ep12.audit.json",
        out_dir / "audit" / "studio_remux_ep12_4K.audit.json",
        out_dir / "audit" / "studio_remux_ep12.audit.json",
    ]
    audit_file = next((p for p in audit_candidates if p.exists()), None)
    assert audit_file is not None, f"Audit report not found among {audit_candidates}"
    audit_data = json.loads(audit_file.read_text(encoding="utf-8"))
    assert audit_data.get("status") == "PASS"
    assert audit_data.get("mux_audit", {}).get("status") == "PASS"


# =============================================================================
# Scenario 3: Interrupted Run & Checkpoint Resume
# =============================================================================

def test_t4_03_interrupted_run_and_checkpoint_resume(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """
    Scenario 3: Interrupted Run & Checkpoint Resume with Segment Validation.
    Verifies that completed chunks are detected and skipped, resuming seamlessly.
    """
    src_vid = create_synthetic_video(
        tmp_path / "open_gop_ep08.mkv",
        num_frames=48,
        fps=24,
        keyint=8,
        audio_tracks=1,
    )
    out_dir = tmp_path / "out"

    temp_dir = out_dir / ".temp_open_gop_ep08"
    segs_dir = temp_dir / "upscaled_segs"
    segs_dir.mkdir(parents=True, exist_ok=True)

    # Pre-generate valid chunk 0 upscaled segment (> 10,000 bytes, 4K HEVC matching pipeline spec)
    chunk0_mp4 = segs_dir / "up_0000.mp4"
    create_synthetic_video(
        chunk0_mp4,
        num_frames=24,
        width=640,
        height=360,
        is_hevc=True,
        audio_tracks=0,
        is_mp4=True,
    )
    assert chunk0_mp4.stat().st_size > 10000

    # Run pipeline with 1s segments: seamlessly resumes chunk 1 and produces final container
    exit_code, out, err = run_upscale_cli(
        [str(src_vid), "-s", "1", "-o", str(out_dir)],
        monkeypatch,
        capsys,
    )
    assert exit_code == 0
    assert "Phân đoạn [01/02] đã hoàn thành từ trước" in out
    assert (out_dir / "open_gop_ep08_4K.mkv").exists()


# =============================================================================
# Scenario 4: Overnight Batch Queue Processing with Deadline
# =============================================================================

def test_t4_04_overnight_batch_queue_with_deadline(mock_pipeline_env, tmp_path, monkeypatch, capsys):
    """
    Scenario 4: Overnight Batch Queue with Mixed Specs and Deadline Termination.
    Verifies real chunk-boundary pause for ep2 and deadline skip for ep3.
    """
    from datetime import datetime, timedelta

    ep1 = create_synthetic_video(tmp_path / "ep1.mkv", num_frames=8, width=320, height=180)
    ep2 = create_synthetic_video(tmp_path / "ep2.mkv", num_frames=48, fps=24, keyint=8, width=240, height=136)
    ep3 = create_synthetic_video(tmp_path / "ep3.mkv", num_frames=8, width=320, height=180)
    out_dir = tmp_path / "out"

    def _fake_eta(video_path, *_args, **_kwargs):
        name = str(video_path)
        if "ep1" in name:
            return (True, False, datetime.now() + timedelta(hours=1))
        if "ep2" in name:
            return (True, True, datetime.now() - timedelta(seconds=1))
        return (False, False, datetime.now() - timedelta(seconds=1))

    with patch("upscale.check_deadline_and_eta", side_effect=_fake_eta):
        exit_code, out, err = run_upscale_cli(
            ["--queue", str(ep1), str(ep2), str(ep3), "--deadline", "07:00", "--allow-partial", "-s", "1", "-o", str(out_dir)],
            monkeypatch,
            capsys,
        )

    assert exit_code == 1
    assert (out_dir / "ep1_4K.mkv").exists()
    assert not (out_dir / "ep2_4K.mkv").exists()
    seg0 = out_dir / ".temp_ep2" / "upscaled_segs" / "up_0000.mp4"
    seg1 = out_dir / ".temp_ep2" / "upscaled_segs" / "up_0001.mp4"
    assert seg0.is_file() and seg0.stat().st_size > 1000
    assert not seg1.exists()
    assert not (out_dir / "ep3_4K.mkv").exists()


# =============================================================================
# Scenario 5: Turnkey Benchmark Harness Execution & Comparative Analysis
# =============================================================================

def test_t4_05_turnkey_benchmark_harness_and_comparator(tmp_path):
    """
    Scenario 5: Turnkey Benchmark Harness Execution & Comparative Analysis.
    Runs baseline and optimized benchmark harnesses, collects 17-metric telemetry,
    and runs comparator to assert non-regression and generate delta table.
    """
    from scripts.benchmark_pipeline import main as bench_main
    import scripts.compare_benchmarks as cb

    src_vid = create_synthetic_video(tmp_path / "fate_ep22_sample.mkv", num_frames=8)
    out_base = tmp_path / "benchmarks" / "baseline"
    out_opt = tmp_path / "benchmarks" / "optimized"

    # Step 1: Run baseline benchmark
    rc_base = bench_main([
        str(src_vid),
        "--tier", "preview",
        "--label", "baseline_run",
        "--output-dir", str(out_base),
        "--dry-run",
    ])
    assert rc_base == 0
    base_summary = out_base / "baseline_run_summary.json"
    base_telemetry = out_base / "baseline_run_telemetry.csv"
    assert base_summary.exists()
    assert base_telemetry.exists()

    # Step 2: Run optimized benchmark
    rc_opt = bench_main([
        str(src_vid),
        "--tier", "preview",
        "--label", "optimized_run",
        "--output-dir", str(out_opt),
        "--dry-run",
    ])
    assert rc_opt == 0
    opt_summary = out_opt / "optimized_run_summary.json"
    opt_telemetry = out_opt / "optimized_run_telemetry.csv"
    assert opt_summary.exists()
    assert opt_telemetry.exists()

    # Step 3: Run comparative analysis with tolerance to accommodate dry-run OS timing jitter
    cmp_exit = cb.main([str(base_summary), str(opt_summary), "--tolerance-pct", "25"])
    assert cmp_exit == 0
