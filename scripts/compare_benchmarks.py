#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/compare_benchmarks.py
Automated comparative delta analyzer for ai-video-upscaler benchmark runs.

Compares baseline_summary.json vs optimized_summary.json:
- Computes FPS, VRAM, and Latency deltas.
- Validates safety invariants (frame count preservation, mux audit pass, zero frame drops).
- Formats a detailed Markdown comparison table.
- Returns exit code 0 if safety invariants hold and metrics satisfy regression tolerance,
  or non-zero (1) if safety invariants fail or severe regressions occur.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Repository Root Bootstrap
# Ensure repository root is on sys.path for direct CLI invocation from any cwd
# ----------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _to_int(val: Any, default: int = 0) -> int:
    """
    Safely convert val to int, handling int, float, string numeric, null/None,
    and non-numeric strings gracefully without raising exceptions.
    """
    try:
        if val is None:
            return default
        return int(float(val))
    except (ValueError, TypeError, OverflowError):
        return default


def calculate_delta(
    base: float,
    opt: float,
    higher_is_better: bool = True,
    unit: str = "",
    digits: int = 2,
) -> Tuple[str, float, str]:
    """
    Compute delta string, delta percentage, and status tag (PASS / IMPROVED / REGRESSED / INFO).
    """
    diff = opt - base
    if base != 0.0:
        pct = (diff / abs(base)) * 100.0
    else:
        pct = 0.0 if opt == 0.0 else 100.0

    sign = "+" if diff > 0 else ""
    delta_str = f"{sign}{diff:.{digits}f}{unit} ({sign}{pct:.2f}%)"

    if abs(pct) < 0.1:
        status = "NEUTRAL"
    elif higher_is_better:
        status = "IMPROVED" if pct > 0 else "REGRESSED"
    else:
        status = "IMPROVED" if pct < 0 else "REGRESSED"

    return delta_str, pct, status


def compare_summaries(
    baseline: Dict[str, Any],
    optimized: Dict[str, Any],
    tolerance_pct: float = 5.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Compare baseline and optimized benchmark summary dicts.
    Returns: (is_success, markdown_report, delta_metrics)
    """
    if not isinstance(baseline, dict) or not isinstance(optimized, dict):
        raise ValueError("Baseline and Optimized summaries must be dictionary objects.")

    failures: List[str] = []
    warnings: List[str] = []

    # 1. Total Frames Preservation Invariant
    base_src_frames = _to_int(baseline.get("total_frames", 0))
    opt_src_frames = _to_int(optimized.get("total_frames", 0))
    if base_src_frames <= 0 or opt_src_frames <= 0 or base_src_frames != opt_src_frames:
        failures.append(
            f"Frame count mismatch: Baseline has {base_src_frames} frames, "
            f"Optimized has {opt_src_frames} frames"
        )

    # 2. Output Frame Count Match Invariant
    opt_out_frames = _to_int(optimized.get("output_frames", opt_src_frames))
    opt_matched_raw = optimized.get("frames_matched", True)
    if isinstance(opt_matched_raw, str):
        opt_matched = opt_matched_raw.strip().lower() in ("true", "1", "yes")
    elif opt_matched_raw is None:
        opt_matched = False
    else:
        opt_matched = bool(opt_matched_raw)

    if not opt_matched or opt_src_frames <= 0 or opt_out_frames <= 0 or opt_out_frames != opt_src_frames:
        failures.append(
            f"Output frame drop detected: Source={opt_src_frames}, Output={opt_out_frames}"
        )

    # 3. Mux Audit Status Invariant
    opt_mux_status = str(optimized.get("mux_status", "UNKNOWN")).upper()
    if opt_mux_status != "PASS":
        failures.append(
            f"Optimized Mux Audit status is '{opt_mux_status}' (expected PASS)"
        )

    # 4. Performance & Resource Deltas
    base_eff_fps = _to_float(baseline.get("effective_fps", 0.0))
    opt_eff_fps = _to_float(optimized.get("effective_fps", 0.0))
    fps_delta_str, fps_pct, fps_status = calculate_delta(base_eff_fps, opt_eff_fps, higher_is_better=True)

    base_ai_fps = _to_float(baseline.get("ai_fps", 0.0))
    opt_ai_fps = _to_float(optimized.get("ai_fps", 0.0))
    ai_fps_delta_str, ai_fps_pct, ai_fps_status = calculate_delta(base_ai_fps, opt_ai_fps, higher_is_better=True)

    base_time = _to_float(baseline.get("total_time_s", 0.0))
    opt_time = _to_float(optimized.get("total_time_s", 0.0))
    time_delta_str, time_pct, time_status = calculate_delta(base_time, opt_time, higher_is_better=False, unit="s")

    base_vram = _to_float(baseline.get("peak_vram_mib", 0.0))
    opt_vram = _to_float(optimized.get("peak_vram_mib", 0.0))
    vram_delta_str, vram_pct, vram_status = calculate_delta(base_vram, opt_vram, higher_is_better=False, unit=" MiB")

    base_gpu = _to_float(baseline.get("avg_gpu_util", 0.0))
    opt_gpu = _to_float(optimized.get("avg_gpu_util", 0.0))
    gpu_diff = opt_gpu - base_gpu
    gpu_delta_str = f"{'+' if gpu_diff > 0 else ''}{gpu_diff:.2f}%"

    base_cpu = _to_float(baseline.get("avg_cpu_pct", 0.0))
    opt_cpu = _to_float(optimized.get("avg_cpu_pct", 0.0))
    cpu_diff = opt_cpu - base_cpu
    cpu_delta_str = f"{'+' if cpu_diff > 0 else ''}{cpu_diff:.2f}%"

    base_dedup = _to_float(baseline.get("dedup_ratio", 0.0))
    opt_dedup = _to_float(optimized.get("dedup_ratio", 0.0))
    dedup_diff = opt_dedup - base_dedup
    dedup_delta_str = f"{'+' if dedup_diff > 0 else ''}{dedup_diff:.2f}%"

    # Regression check on Effective FPS
    if base_eff_fps > 0.0 and fps_pct < -abs(tolerance_pct):
        warnings.append(
            f"Effective FPS regressed by {abs(fps_pct):.2f}% (exceeds tolerance of {tolerance_pct}%)"
        )

    # Format Markdown Report
    base_label = baseline.get("label", "baseline")
    opt_label = optimized.get("label", "optimized")

    frames_match_ok = (base_src_frames > 0 and opt_src_frames > 0 and base_src_frames == opt_src_frames)
    output_match_ok = (opt_matched and opt_src_frames > 0 and opt_out_frames > 0 and opt_out_frames == opt_src_frames)

    rows = [
        ("Effective Pipeline FPS", f"{base_eff_fps:.2f} fps", f"{opt_eff_fps:.2f} fps", fps_delta_str, fps_status),
        ("AI Inference Steady FPS", f"{base_ai_fps:.2f} fps", f"{opt_ai_fps:.2f} fps", ai_fps_delta_str, ai_fps_status),
        ("Total Processing Time", f"{base_time:.2f} s", f"{opt_time:.2f} s", time_delta_str, time_status),
        ("Peak VRAM Allocation", f"{base_vram:.1f} MiB", f"{opt_vram:.1f} MiB", vram_delta_str, vram_status),
        ("Average GPU Compute Util", f"{base_gpu:.1f}%", f"{opt_gpu:.1f}%", gpu_delta_str, "INFO"),
        ("Average Host CPU Util", f"{base_cpu:.1f}%", f"{opt_cpu:.1f}%", cpu_delta_str, "INFO"),
        ("Dedup Reduction Ratio", f"{base_dedup:.1f}%", f"{opt_dedup:.1f}%", dedup_delta_str, "INFO"),
        ("Total Source Frames", str(base_src_frames), str(opt_src_frames), "Matched" if frames_match_ok else "MISMATCH", "PASS" if frames_match_ok else "FAIL"),
        ("Output Frames Matched", str(baseline.get("frames_matched", True)), str(opt_matched), "Preserved" if output_match_ok else "DROPPED", "PASS" if output_match_ok else "FAIL"),
        ("Container Mux Audit", str(baseline.get("mux_status", "PASS")), str(opt_mux_status), "Verified", "PASS" if opt_mux_status == "PASS" else "FAIL"),
    ]

    md_lines = [
        f"# Benchmark Comparison: `{base_label}` vs `{opt_label}`",
        "",
        "| Metric | Baseline | Optimized | Delta | Verdict |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for metric, b_val, o_val, d_val, status in rows:
        md_lines.append(f"| **{metric}** | {b_val} | {o_val} | {d_val} | `{status}` |")

    is_success = (len(failures) == 0 and len(warnings) == 0)

    md_lines.append("")
    if is_success:
        md_lines.append("### Overall Verdict: **PASS** (All safety invariants hold and metrics verified)")
    else:
        md_lines.append("### Overall Verdict: **FAIL**")
        for f in failures:
            md_lines.append(f"- ❌ **SAFETY VIOLATION**: {f}")
        for w in warnings:
            md_lines.append(f"- ⚠️ **REGRESSION WARNING**: {w}")

    delta_metrics = {
        "effective_fps_delta_pct": fps_pct,
        "ai_fps_delta_pct": ai_fps_pct,
        "total_time_delta_pct": time_pct,
        "peak_vram_delta_mib": opt_vram - base_vram,
        "peak_vram_delta_pct": vram_pct,
        "frames_matched": frames_match_ok and output_match_ok,
        "mux_audit_pass": (opt_mux_status == "PASS"),
        "failures": failures,
        "warnings": warnings,
        "is_success": is_success,
    }

    report_content = "\n".join(md_lines) + "\n"
    return is_success, report_content, delta_metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark summary JSON files and validate invariants."
    )
    parser.add_argument("baseline", help="Path to baseline summary JSON")
    parser.add_argument("optimized", help="Path to optimized summary JSON")
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=5.0,
        help="Max allowable regression %% on effective FPS before failing (default: 5.0%%)",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional path to write the formatted Markdown table",
    )

    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline).resolve()
    optimized_path = Path(args.optimized).resolve()

    if not baseline_path.exists():
        print(f"❌ Error: Baseline file not found: {baseline_path}", file=sys.stderr)
        return 1
    if not optimized_path.exists():
        print(f"❌ Error: Optimized file not found: {optimized_path}", file=sys.stderr)
        return 1

    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        with open(optimized_path, "r", encoding="utf-8") as f:
            optimized_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON files: {e}", file=sys.stderr)
        return 1

    try:
        is_success, report_md, _ = compare_summaries(
            baseline=baseline_data,
            optimized=optimized_data,
            tolerance_pct=args.tolerance_pct,
        )
    except Exception as e:
        print(f"❌ Error during benchmark comparison: {e}", file=sys.stderr)
        return 1

    print(report_md)

    if args.output_md:
        try:
            out_p = Path(args.output_md).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(report_md)
            print(f"Báo cáo Markdown đã được lưu tại: {out_p}")
        except Exception as e:
            print(f"❌ Error writing output markdown file: {e}", file=sys.stderr)
            return 1

    return 0 if is_success else 1


if __name__ == "__main__":
    sys.exit(main())
