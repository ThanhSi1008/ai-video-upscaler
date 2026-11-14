#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_benchmark_infra.py
Unit tests for Milestone 1 Benchmark Infrastructure:
- TelemetryCollector (hardware sampler, threading, CSV format, exception safety)
- Benchmark Harness (tier durations, CLI parsing, mock execution, summary generation)
- Benchmark Comparator (delta math, safety invariants, markdown reporting, exit codes)

100% offline, zero network or live GPU requirements.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from scripts.benchmark_pipeline import (
    TIER_DURATIONS,
    main as bench_main,
    run_benchmark,
)
from scripts.compare_benchmarks import (
    calculate_delta,
    compare_summaries,
    main as compare_main,
)
from scripts.telemetry_collector import (
    TELEMETRY_HEADER,
    TelemetryCollector,
    _safe_float,
    default_gpu_query,
)


# -----------------------------------------------------------------------------
# 1. TelemetryCollector Tests
# -----------------------------------------------------------------------------

def test_safe_float_conversions():
    assert _safe_float("123.45") == 123.45
    assert _safe_float(42) == 42.0
    assert _safe_float("N/A") == 0.0
    assert _safe_float("[Not Supported]") == 0.0
    assert _safe_float(None) == 0.0
    assert _safe_float("invalid", default=-1.0) == -1.0


def test_default_gpu_query_fallback():
    """When nvidia-smi fails or is absent, default_gpu_query returns safe dict."""
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        data = default_gpu_query()
        assert data["pstate"] == "N/A"
        assert data["gpu_util_pct"] == 0.0
        assert data["vram_used_mib"] == 0.0
        assert data["core_clock_mhz"] == 0.0


def test_default_gpu_query_success():
    """Mock successful nvidia-smi execution and verify parsing."""
    mock_res = MagicMock()
    mock_res.stdout = "P0, 95.0, 45.0, 100.0, 0.0, 7501.0, 1935.0, 165.5, 68.0, 4200.0, 7900.0\n"
    with patch("subprocess.run", return_value=mock_res):
        data = default_gpu_query()
        assert data["pstate"] == "P0"
        assert data["gpu_util_pct"] == 95.0
        assert data["mem_util_pct"] == 45.0
        assert data["enc_util_pct"] == 100.0
        assert data["dec_util_pct"] == 0.0
        assert data["mem_clock_mhz"] == 7501.0
        assert data["core_clock_mhz"] == 1935.0
        assert data["power_w"] == 165.5
        assert data["temp_c"] == 68.0
        assert data["vram_used_mib"] == 4200.0
        assert data["vram_free_mib"] == 7900.0


def test_telemetry_collector_lifecycle_and_csv(tmp_path: Path):
    """Test start, sampling, CSV streaming, and summary generation."""
    csv_file = tmp_path / "telemetry.csv"

    mock_sample = {
        "pstate": "P0",
        "gpu_util_pct": 80.0,
        "mem_util_pct": 40.0,
        "enc_util_pct": 50.0,
        "dec_util_pct": 0.0,
        "mem_clock_mhz": 7501.0,
        "core_clock_mhz": 1800.0,
        "power_w": 120.0,
        "temp_c": 65.0,
        "vram_used_mib": 3500.0,
        "vram_free_mib": 8600.0,
    }

    collector = TelemetryCollector(
        output_csv_path=csv_file,
        interval_sec=0.1,
        gpu_query_fn=lambda: mock_sample,
    )

    collector.start()
    assert collector.is_running
    time.sleep(0.35)
    summary = collector.stop()
    assert not collector.is_running

    # Verify CSV file
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == TELEMETRY_HEADER
        rows = list(reader)
        assert len(rows) >= 2
        first_row = rows[0]
        assert first_row["pstate"] == "P0"
        assert float(first_row["gpu_util_pct"]) == 80.0
        assert float(first_row["vram_used_mib"]) == 3500.0

    # Verify summary dict
    assert summary["samples_count"] >= 2
    assert summary["peak_vram_mib"] == 3500.0
    assert summary["avg_gpu_util"] == 80.0
    assert summary["avg_enc_util"] == 50.0
    assert summary["peak_temp_c"] == 65.0
    assert summary["csv_path"] == str(csv_file)


def test_telemetry_collector_context_manager(tmp_path: Path):
    """Verify TelemetryCollector works as a context manager."""
    csv_file = tmp_path / "context_telemetry.csv"
    with TelemetryCollector(csv_file, interval_sec=0.1) as col:
        assert col.is_running
        time.sleep(0.15)
    assert not col.is_running
    assert csv_file.exists()
    assert col.summary["samples_count"] >= 1


def test_telemetry_collector_immediate_stop(tmp_path: Path):
    """Calling stop() immediately should not raise an error."""
    csv_file = tmp_path / "fast_stop.csv"
    col = TelemetryCollector(csv_file, interval_sec=1.0)
    col.start()
    summary = col.stop()
    assert not col.is_running
    assert summary["samples_count"] >= 1


def test_telemetry_collector_empty_summary():
    """Summary of collector with no samples should have safe defaults."""
    col = TelemetryCollector("dummy.csv")
    summary = col._compute_summary()
    assert summary["samples_count"] == 0
    assert summary["peak_vram_mib"] == 0.0
    assert summary["avg_gpu_util"] == 0.0


# -----------------------------------------------------------------------------
# 2. Benchmark Pipeline Harness Tests
# -----------------------------------------------------------------------------

def test_tier_durations():
    assert TIER_DURATIONS["preview"] == 10
    assert TIER_DURATIONS["climax"] == 60
    assert TIER_DURATIONS["full"] is None


def test_run_benchmark_invalid_tier(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid tier"):
        run_benchmark(
            source_path="fake.mkv",
            tier="nonexistent_tier",
            output_dir=tmp_path,
            label="test",
        )


def test_run_benchmark_dry_run_artifacts(tmp_path: Path):
    """Dry-run benchmark produces summary.json, telemetry.csv, and audit.json."""
    source_file = tmp_path / "sample_video.mkv"
    source_file.write_text("fake video data", encoding="utf-8")

    out_dir = tmp_path / "bench_out"

    summary = run_benchmark(
        source_path=source_file,
        tier="preview",
        output_dir=out_dir,
        label="test_dry_run",
        dry_run=True,
    )

    summary_file = out_dir / "test_dry_run_summary.json"
    telemetry_file = out_dir / "test_dry_run_telemetry.csv"
    audit_file = out_dir / "test_dry_run.audit.json"

    assert summary_file.exists()
    assert telemetry_file.exists()
    assert audit_file.exists()

    assert summary["label"] == "test_dry_run"
    assert summary["tier"] == "preview"
    assert summary["mux_status"] == "PASS"
    assert summary["frames_matched"] is True
    assert summary["total_frames"] == 240  # 10s * 24 fps
    assert summary["effective_fps"] > 0.0

    with open(summary_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["label"] == "test_dry_run"
        assert loaded["total_frames"] == 240


def test_run_benchmark_with_pipeline_mock(tmp_path: Path):
    """Test custom pipeline callback injection and telemetry capture."""
    source_file = tmp_path / "sample_mock.mkv"
    source_file.write_text("sample content", encoding="utf-8")
    out_dir = tmp_path / "mock_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    def mock_pipeline_fn(**kwargs):
        # Create mock output MKV and mock audit JSON
        out_mkv = kwargs["output_path"]
        Path(out_mkv).write_text("mock 4K MKV", encoding="utf-8")
        audit_path = Path(out_mkv).parent / f"{Path(out_mkv).stem}.audit.json"
        audit_content = {
            "status": "PASS",
            "source_frames": 100,
            "output_frames": 100,
            "frames_matched": True,
            "ai_steady_fps": 35.5,
            "average_pipeline_fps": 25.0,
        }
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_content, f)
        return out_mkv

    summary = run_benchmark(
        source_path=source_file,
        tier="preview",
        output_dir=out_dir,
        label="mock_run",
        pipeline_fn=mock_pipeline_fn,
        dry_run=False,
    )

    assert summary["total_frames"] == 100
    assert summary["output_frames"] == 100
    assert summary["mux_status"] == "PASS"
    assert summary["ai_fps"] == 35.5


def test_benchmark_cli_dry_run(tmp_path: Path):
    """Test benchmark_pipeline.py main() via CLI args in dry-run mode."""
    source_file = tmp_path / "cli_sample.mkv"
    source_file.write_text("sample", encoding="utf-8")
    out_dir = tmp_path / "cli_out"

    ret = bench_main([
        str(source_file),
        "--tier", "preview",
        "--output-dir", str(out_dir),
        "--label", "cli_test",
        "--dry-run",
    ])
    assert ret == 0
    assert (out_dir / "cli_test_summary.json").exists()


# -----------------------------------------------------------------------------
# 3. Compare Benchmarks Tests
# -----------------------------------------------------------------------------

def test_calculate_delta_math():
    delta_str, pct, status = calculate_delta(10.0, 15.0, higher_is_better=True)
    assert pct == 50.0
    assert status == "IMPROVED"
    assert "+5.00" in delta_str

    # Lower is better (time/vram)
    delta_str, pct, status = calculate_delta(100.0, 80.0, higher_is_better=False, unit="s")
    assert pct == -20.0
    assert status == "IMPROVED"
    assert "-20.00s" in delta_str

    # Regressed
    _, pct, status = calculate_delta(20.0, 15.0, higher_is_better=True)
    assert pct == -25.0
    assert status == "REGRESSED"


def test_compare_summaries_success():
    baseline = {
        "label": "baseline_run",
        "total_frames": 1000,
        "output_frames": 1000,
        "frames_matched": True,
        "effective_fps": 20.0,
        "ai_fps": 30.0,
        "total_time_s": 50.0,
        "peak_vram_mib": 6000.0,
        "avg_gpu_util": 90.0,
        "avg_cpu_pct": 25.0,
        "dedup_ratio": 30.0,
        "mux_status": "PASS",
    }
    optimized = {
        "label": "optimized_run",
        "total_frames": 1000,
        "output_frames": 1000,
        "frames_matched": True,
        "effective_fps": 25.0,  # +25%
        "ai_fps": 36.0,         # +20%
        "total_time_s": 40.0,   # -20%
        "peak_vram_mib": 5500.0, # -8.33%
        "avg_gpu_util": 95.0,
        "avg_cpu_pct": 20.0,
        "dedup_ratio": 32.0,
        "mux_status": "PASS",
    }

    is_success, report, deltas = compare_summaries(baseline, optimized)
    assert is_success is True
    assert "Overall Verdict: **PASS**" in report
    assert deltas["effective_fps_delta_pct"] == 25.0
    assert deltas["frames_matched"] is True
    assert deltas["mux_audit_pass"] is True


def test_compare_summaries_frame_count_mismatch():
    baseline = {"total_frames": 1000, "output_frames": 1000, "mux_status": "PASS"}
    optimized = {"total_frames": 998, "output_frames": 998, "mux_status": "PASS"}

    is_success, report, deltas = compare_summaries(baseline, optimized)
    assert is_success is False
    assert "Frame count mismatch" in report
    assert any("Frame count mismatch" in f for f in deltas["failures"])


def test_compare_summaries_output_frame_drop():
    baseline = {"total_frames": 1000, "output_frames": 1000, "mux_status": "PASS"}
    optimized = {"total_frames": 1000, "output_frames": 990, "frames_matched": False, "mux_status": "PASS"}

    is_success, report, deltas = compare_summaries(baseline, optimized)
    assert is_success is False
    assert "Output frame drop detected" in report


def test_compare_summaries_mux_audit_fail():
    baseline = {"total_frames": 1000, "output_frames": 1000, "mux_status": "PASS"}
    optimized = {"total_frames": 1000, "output_frames": 1000, "mux_status": "FAIL"}

    is_success, report, deltas = compare_summaries(baseline, optimized)
    assert is_success is False
    assert "Mux Audit status is 'FAIL'" in report


def test_compare_summaries_regression_tolerance():
    baseline = {"total_frames": 1000, "output_frames": 1000, "effective_fps": 30.0, "mux_status": "PASS"}
    # FPS drops by 20% (from 30 to 24), exceeding default 5% tolerance
    optimized = {"total_frames": 1000, "output_frames": 1000, "effective_fps": 24.0, "mux_status": "PASS"}

    is_success, report, deltas = compare_summaries(baseline, optimized, tolerance_pct=5.0)
    assert is_success is False
    assert any("regressed" in w for w in deltas["warnings"])


def test_compare_cli_entrypoint(tmp_path: Path):
    base_json = tmp_path / "base_summary.json"
    opt_json = tmp_path / "opt_summary.json"
    out_md = tmp_path / "comparison.md"

    data_base = {
        "label": "base",
        "total_frames": 500,
        "output_frames": 500,
        "effective_fps": 20.0,
        "mux_status": "PASS",
    }
    data_opt = {
        "label": "opt",
        "total_frames": 500,
        "output_frames": 500,
        "effective_fps": 25.0,
        "mux_status": "PASS",
    }

    base_json.write_text(json.dumps(data_base), encoding="utf-8")
    opt_json.write_text(json.dumps(data_opt), encoding="utf-8")

    ret = compare_main([
        str(base_json),
        str(opt_json),
        "--output-md", str(out_md),
    ])
    assert ret == 0
    assert out_md.exists()
    assert "Benchmark Comparison" in out_md.read_text(encoding="utf-8")


def test_safe_int_conversions():
    """Verify _to_int safely handles integers, numeric floats, strings, None, and corrupt data."""
    from scripts.compare_benchmarks import _to_int

    assert _to_int(42) == 42
    assert _to_int("123") == 123
    assert _to_int("4374.0") == 4374
    assert _to_int(4374.9) == 4374
    assert _to_int(-10) == -10
    assert _to_int(None) == 0
    assert _to_int(None, default=99) == 99
    assert _to_int("INVALID") == 0
    assert _to_int("null") == 0
    assert _to_int(float("nan")) == 0
    assert _to_int(float("inf")) == 0
    assert _to_int({"bad": "dict"}) == 0
    assert _to_int([1, 2, 3]) == 0


def test_compare_summaries_zero_or_negative_frames():
    """Verify that zero or negative frame counts strictly trigger safety violation failures."""
    # Case A: Baseline positive (1000 frames) vs Optimized zero frames
    base = {"total_frames": 1000, "output_frames": 1000, "mux_status": "PASS"}
    opt_zero = {"total_frames": 0, "output_frames": 0, "mux_status": "PASS"}
    is_success, report, deltas = compare_summaries(base, opt_zero)
    assert is_success is False
    assert any("Frame count mismatch" in f for f in deltas["failures"])
    assert deltas["frames_matched"] is False
    assert "Overall Verdict: **FAIL**" in report

    # Case B: Baseline zero vs Optimized positive (1000 frames)
    is_success, report, deltas = compare_summaries(opt_zero, base)
    assert is_success is False
    assert any("Frame count mismatch" in f for f in deltas["failures"])

    # Case C: Both zero frames
    is_success, report, deltas = compare_summaries(opt_zero, opt_zero)
    assert is_success is False
    assert any("Frame count mismatch" in f for f in deltas["failures"])

    # Case D: Negative frame count
    opt_neg = {"total_frames": -50, "output_frames": -50, "mux_status": "PASS"}
    is_success, report, deltas = compare_summaries(base, opt_neg)
    assert is_success is False
    assert any("Frame count mismatch" in f for f in deltas["failures"])


def test_compare_summaries_corrupted_null_fields():
    """Verify compare_summaries cleanly handles null/None and non-numeric string values."""
    baseline = {
        "total_frames": None,
        "output_frames": None,
        "effective_fps": None,
        "total_time_s": None,
        "peak_vram_mib": None,
        "mux_status": "PASS",
    }
    optimized = {
        "total_frames": "4374.0",
        "output_frames": "4374",
        "effective_fps": "not_a_float",
        "total_time_s": "invalid",
        "peak_vram_mib": "NaN",
        "mux_status": "PASS",
    }

    # Must NOT raise TypeError or ValueError
    is_success, report, deltas = compare_summaries(baseline, optimized)
    assert is_success is False
    assert any("Frame count mismatch" in f for f in deltas["failures"])
    assert "Overall Verdict: **FAIL**" in report


def test_compare_cli_zero_frames_exit_code_1(tmp_path: Path):
    """Adversarial check: baseline 4374 frames vs optimized 0 frames must exit with code 1."""
    base_json = tmp_path / "base_pos.json"
    opt_json = tmp_path / "opt_zero.json"

    base_json.write_text(
        json.dumps({
            "label": "base",
            "total_frames": 4374,
            "output_frames": 4374,
            "effective_fps": 30.0,
            "mux_status": "PASS",
        }),
        encoding="utf-8",
    )
    opt_json.write_text(
        json.dumps({
            "label": "opt",
            "total_frames": 0,
            "output_frames": 0,
            "effective_fps": 30.0,
            "mux_status": "PASS",
        }),
        encoding="utf-8",
    )

    ret = compare_main([str(base_json), str(opt_json)])
    assert ret == 1


def test_compare_cli_corrupted_null_fields_exit_code_1(tmp_path: Path):
    """CLI execution with null or invalid string literals in integer/float fields exits code 1 without traceback."""
    base_json = tmp_path / "base_null.json"
    opt_json = tmp_path / "opt_corrupt.json"

    base_json.write_text(
        json.dumps({
            "label": "base",
            "total_frames": None,
            "effective_fps": "INVALID_FLOAT",
            "mux_status": "PASS",
        }),
        encoding="utf-8",
    )
    opt_json.write_text(
        json.dumps({
            "label": "opt",
            "total_frames": "INVALID_INT",
            "output_frames": None,
            "effective_fps": 30.0,
            "mux_status": "PASS",
        }),
        encoding="utf-8",
    )

    ret = compare_main([str(base_json), str(opt_json)])
    assert ret == 1


def test_compare_cli_malformed_structure_exit_code_1(tmp_path: Path):
    """CLI execution with non-dict JSON structure exits code 1 gracefully via exception guard."""
    base_json = tmp_path / "base_list.json"
    opt_json = tmp_path / "opt_valid.json"

    base_json.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    opt_json.write_text(
        json.dumps({"total_frames": 100, "output_frames": 100, "mux_status": "PASS"}),
        encoding="utf-8",
    )

    ret = compare_main([str(base_json), str(opt_json)])
    assert ret == 1


def test_run_benchmark_dedup_plan_preservation(tmp_path: Path):
    """Verify that plan.json is preserved and dedup metrics are correctly extracted when keep_temp=False."""
    source_file = tmp_path / "sample_dedup.mkv"
    source_file.write_text("video content", encoding="utf-8")
    out_dir = tmp_path / "dedup_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    def mock_pipeline_fn(**kwargs):
        out_mkv = kwargs["output_path"]
        Path(out_mkv).write_text("4K MKV", encoding="utf-8")
        # Write audit.json
        audit_path = Path(out_mkv).parent / f"{Path(out_mkv).stem}.audit.json"
        audit_data = {
            "status": "PASS",
            "source_frames": 240,
            "output_frames": 240,
            "frames_matched": True,
            "ai_steady_fps": 30.0,
        }
        audit_path.write_text(json.dumps(audit_data), encoding="utf-8")

        # Simulate pipeline creating .temp_<stem>_preview_10s/plan.json
        temp_dir = Path(out_mkv).parent / f".temp_{source_file.stem}_preview_10s"
        temp_dir.mkdir(parents=True, exist_ok=True)
        plan_file = temp_dir / "plan.json"
        plan_data = {
            "total_frames": 240,
            "total_kept_frames": 150,
            "dedup_ratio_percent": 37.5,
        }
        plan_file.write_text(json.dumps(plan_data), encoding="utf-8")
        return out_mkv

    # Run with default keep_temp=False
    summary = run_benchmark(
        source_path=source_file,
        tier="preview",
        output_dir=out_dir,
        label="dedup_test",
        pipeline_fn=mock_pipeline_fn,
        keep_temp=False,
        dry_run=False,
    )

    # Invariants
    assert summary["total_frames"] == 240
    assert summary["unique_frames"] == 150
    assert summary["dedup_ratio"] == 37.5
    # Verify <label>.plan.json was preserved in out_dir
    assert (out_dir / "dedup_test.plan.json").exists()
    # Verify temp dir was cleaned up
    assert not (out_dir / f".temp_{source_file.stem}_preview_10s").exists()


# -----------------------------------------------------------------------------
# 4. Subprocess CLI Integration Tests (Prevent sys.path Regression)
# -----------------------------------------------------------------------------

def test_benchmark_cli_subprocess_dry_run(tmp_path: Path):
    """
    Subprocess CLI test verifying standalone direct execution:
    python scripts/benchmark_pipeline.py <source> --tier preview ... --dry-run
    Ensures sys.path bootstrapping and CLI entry point succeed without ModuleNotFoundError.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "benchmark_pipeline.py"
    source_file = tmp_path / "cli_sub_sample.mkv"
    source_file.write_text("mock video content", encoding="utf-8")
    out_dir = tmp_path / "cli_sub_out"

    cmd = [
        sys.executable,
        str(script_path),
        str(source_file),
        "--tier", "preview",
        "--output-dir", str(out_dir),
        "--label", "cli_sub_test",
        "--dry-run",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, f"Process failed with exit {proc.returncode}.\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"

    summary_path = out_dir / "cli_sub_test_summary.json"
    telemetry_path = out_dir / "cli_sub_test_telemetry.csv"
    audit_path = out_dir / "cli_sub_test.audit.json"

    assert summary_path.exists(), "summary.json was not created"
    assert telemetry_path.exists(), "telemetry.csv was not created"
    assert audit_path.exists(), "audit.json was not created"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert summary_data["label"] == "cli_sub_test"
    assert summary_data["mux_status"] == "PASS"
    assert summary_data["total_frames"] > 0


def test_benchmark_cli_subprocess_external_working_dir(tmp_path: Path):
    """
    Subprocess CLI test verifying execution from an external working directory (cwd != repo_root).
    Guarantees repo root resolution via Path(__file__).resolve().parents[1] works from any cwd.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "benchmark_pipeline.py"
    source_file = tmp_path / "external_sample.mkv"
    source_file.write_text("mock video content", encoding="utf-8")
    out_dir = tmp_path / "external_out"

    cmd = [
        sys.executable,
        str(script_path),
        str(source_file),
        "--tier", "preview",
        "--output-dir", str(out_dir),
        "--label", "ext_cwd_test",
        "--dry-run",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, f"Process failed with exit {proc.returncode}.\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    assert (out_dir / "ext_cwd_test_summary.json").exists()


def test_compare_cli_subprocess_success(tmp_path: Path):
    """
    Subprocess CLI test verifying standalone direct execution of comparator:
    python scripts/compare_benchmarks.py base.json opt.json --output-md comparison.md
    """
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "compare_benchmarks.py"

    base_json = tmp_path / "base_summary.json"
    opt_json = tmp_path / "opt_summary.json"
    out_md = tmp_path / "sub_comparison.md"

    data_base = {
        "label": "base",
        "total_frames": 500,
        "output_frames": 500,
        "effective_fps": 20.0,
        "mux_status": "PASS",
    }
    data_opt = {
        "label": "opt",
        "total_frames": 500,
        "output_frames": 500,
        "effective_fps": 25.0,
        "mux_status": "PASS",
    }

    base_json.write_text(json.dumps(data_base), encoding="utf-8")
    opt_json.write_text(json.dumps(data_opt), encoding="utf-8")

    cmd = [
        sys.executable,
        str(script_path),
        str(base_json),
        str(opt_json),
        "--output-md", str(out_md),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, f"Process failed with exit {proc.returncode}.\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    assert out_md.exists()
    assert "Overall Verdict: **PASS**" in proc.stdout
    assert "Benchmark Comparison: `base` vs `opt`" in out_md.read_text(encoding="utf-8")


def test_compare_cli_subprocess_safety_failure(tmp_path: Path):
    """
    Subprocess CLI test verifying that safety invariant failures return exit code 1:
    Mux status FAIL must produce exit code 1 and SAFETY VIOLATION message.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "compare_benchmarks.py"

    base_json = tmp_path / "base_fail.json"
    opt_json = tmp_path / "opt_fail.json"

    data_base = {
        "label": "base",
        "total_frames": 500,
        "output_frames": 500,
        "effective_fps": 20.0,
        "mux_status": "PASS",
    }
    data_opt = {
        "label": "opt",
        "total_frames": 500,
        "output_frames": 500,
        "effective_fps": 25.0,
        "mux_status": "FAIL",
    }

    base_json.write_text(json.dumps(data_base), encoding="utf-8")
    opt_json.write_text(json.dumps(data_opt), encoding="utf-8")

    cmd = [
        sys.executable,
        str(script_path),
        str(base_json),
        str(opt_json),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 1, f"Expected exit code 1 on safety violation, got {proc.returncode}"
    assert "SAFETY VIOLATION" in proc.stdout
    assert "Optimized Mux Audit status is 'FAIL'" in proc.stdout
