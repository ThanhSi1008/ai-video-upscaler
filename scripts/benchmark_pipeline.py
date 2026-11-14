#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/benchmark_pipeline.py
Turnkey benchmark harness CLI for ai-video-upscaler.

CLI arguments:
    python scripts/benchmark_pipeline.py <source_path> \\
        --tier [preview|climax|full] \\
        --output-dir <dir> \\
        --label <label> \\
        [--engine cuda|cpu] \\
        [--zero-disk] \\
        [--no-tensorrt] \\
        [--model <model>] \\
        [--batch-size <n>] \\
        [--dry-run]

Manages .gpu_active.lock properly.
Coordinates continuous 17-metric hardware telemetry via TelemetryCollector.
Wraps upscale execution, aggregates worker statistics, and runs mux auditing.
Generates:
    - <output_dir>/<label>_summary.json
    - <output_dir>/<label>_telemetry.csv
    - <output_dir>/<label>.audit.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional

# ----------------------------------------------------------------------
# Repository Root Bootstrap
# Ensure repository root is on sys.path for direct CLI invocation from any cwd
# ----------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.telemetry_collector import TelemetryCollector

# Tier default durations in seconds
TIER_DURATIONS: Dict[str, Optional[int]] = {
    "preview": 10,
    "climax": 60,
    "full": None,
}

# Tier default seek start times in seconds (climax seeks to 13:30 / 810.0s for anime sakuga battle)
TIER_START_TIMES: Dict[str, Optional[float]] = {
    "preview": None,
    "climax": 810.0,
    "full": None,
}


def run_benchmark(
    source_path: str | Path,
    tier: str = "preview",
    output_dir: str | Path = "out/benchmarks",
    label: str = "benchmark",
    engine: str = "cuda",
    zero_disk: bool = False,
    no_tensorrt: bool = False,
    model: str = "animevideov3",
    batch_size: int = 8,
    dedup_policy: str = "default",
    keep_temp: bool = False,
    dry_run: bool = False,
    duration: Optional[float] = None,
    start_time: Optional[float] = None,
    pipeline_fn: Optional[Callable[..., Any]] = None,
    telemetry_cls: Optional[type] = None,
) -> Dict[str, Any]:
    """
    Execute standardized benchmark run with telemetry collection, auditing, and summary generation.
    """
    tier = tier.lower().strip()
    if tier not in TIER_DURATIONS:
        raise ValueError(f"Invalid tier '{tier}'. Must be one of: {list(TIER_DURATIONS.keys())}")

    source = Path(source_path).resolve()
    if not source.exists() and not dry_run:
        raise FileNotFoundError(f"Source video file not found: {source}")
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine duration
    target_duration: Optional[int] = None
    if duration is not None and duration > 0:
        target_duration = int(duration)
    else:
        target_duration = TIER_DURATIONS.get(tier)

    telemetry_csv = out_dir / f"{label}_telemetry.csv"
    summary_json_path = out_dir / f"{label}_summary.json"
    audit_json_path = out_dir / f"{label}.audit.json"
    target_output_mkv = out_dir / f"{label}_4K.mkv"
    plan_json_path = out_dir / f"{label}.plan.json"

    collector_class = telemetry_cls or TelemetryCollector
    collector = collector_class(output_csv_path=telemetry_csv, interval_sec=1.0)

    # Manage GPU lock if running live on hardware
    lock_acquired = False
    if not dry_run and engine.lower() != "cpu":
        try:
            from src.config import acquire_gpu_lock
            acquire_gpu_lock()
            lock_acquired = True
        except Exception as e:
            # If already locked or error, report and re-raise
            print(f"❌ Failed to acquire GPU lock: {e}", file=sys.stderr)
            raise

    start_wall = time.perf_counter()
    collector.start()

    actual_output_mkv: Optional[str] = None
    audit_data: Dict[str, Any] = {}
    plan_data: Dict[str, Any] = {}

    try:
        if dry_run:
            # Mock / dry-run simulation mode
            time.sleep(0.5)
            sim_dur = target_duration if target_duration else 10
            sim_frames = int(sim_dur * 24)
            sim_unique = int(sim_frames * 0.65)
            audit_data = {
                "status": "PASS",
                "source_file": str(source),
                "output_file": str(target_output_mkv),
                "source_frames": sim_frames,
                "output_frames": sim_frames,
                "frames_matched": True,
                "duration_seconds": float(sim_dur),
                "total_processing_seconds": 1.0,
                "average_pipeline_fps": 30.0,
                "ai_steady_fps": 40.0,
                "mux_audit": {
                    "status": "PASS",
                    "failure_reasons": [],
                },
            }
            plan_data = {
                "total_frames": sim_frames,
                "total_kept_frames": sim_unique,
                "dedup_ratio_percent": round(100.0 * (1.0 - sim_unique / sim_frames), 2),
            }
            with open(audit_json_path, "w", encoding="utf-8") as f:
                json.dump(audit_data, f, indent=2)
            with open(plan_json_path, "w", encoding="utf-8") as f:
                json.dump(plan_data, f, indent=2)
        else:
            # Real pipeline execution
            custom_clip: Optional[Path] = None
            effective_input = str(source)
            preview_arg = target_duration

            if pipeline_fn is not None:
                actual_output_mkv = pipeline_fn(
                    input_path=str(source),
                    output_path=str(target_output_mkv),
                    model=model,
                    engine=engine,
                    zero_disk=zero_disk,
                    use_tensorrt=not no_tensorrt,
                    preview_seconds=target_duration,
                    batch_size=batch_size,
                    dedup_policy=dedup_policy,
                    keep_temp=True,  # ALWAYS keep temp in pipeline; harness handles cleanup
                )
            else:
                from src.pipeline import run_upscale_pipeline
                # Custom clip extraction if start_time is explicitly specified or defined for tier
                effective_start = start_time if start_time is not None else TIER_START_TIMES.get(tier)
                if effective_start is not None and target_duration is not None:
                    custom_clip = out_dir / f".bench_clip_{label}.mkv"
                    cut_cmd = [
                        "ffmpeg", "-y", "-ss", str(effective_start), "-i", str(source),
                        "-t", str(target_duration), "-c", "copy", "-map", "0",
                        str(custom_clip)
                    ]
                    subprocess.run(cut_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    effective_input = str(custom_clip)
                    preview_arg = None
                else:
                    preview_arg = target_duration

                actual_output_mkv = run_upscale_pipeline(
                    input_path=effective_input,
                    output_path=str(target_output_mkv),
                    model=model,
                    engine=engine,
                    zero_disk=zero_disk,
                    use_tensorrt=not no_tensorrt,
                    preview_seconds=preview_arg,
                    batch_size=batch_size,
                    dedup_policy=dedup_policy,
                    keep_temp=True,  # ALWAYS keep temp in pipeline; harness handles cleanup
                )

            # Discover and copy the pipeline audit artifact
            candidate_audit = Path(str(actual_output_mkv).rsplit(".", 1)[0] + ".audit.json")
            if candidate_audit.exists():
                try:
                    with open(candidate_audit, "r", encoding="utf-8") as f:
                        audit_data = json.load(f)
                    shutil.copy2(candidate_audit, audit_json_path)
                except Exception:
                    pass

            # Dynamic plan discovery to find work_dir / plan.json
            detected_temp_dirs: List[Path] = []
            candidate_dirs = [
                out_dir / f".temp_{Path(effective_input).stem}_preview_{preview_arg}s" if preview_arg else out_dir / f".temp_{Path(effective_input).stem}",
                out_dir / f".temp_{source.stem}_preview_{target_duration}s",
                out_dir / f".temp_{source.stem}",
                out_dir / f".temp_{Path(actual_output_mkv or target_output_mkv).stem}",
            ]
            for td in sorted([d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith(".temp_")], key=lambda d: d.stat().st_mtime, reverse=True):
                if td not in candidate_dirs:
                    candidate_dirs.append(td)

            found_plan_path: Optional[Path] = None
            for c_dir in candidate_dirs:
                p_file = c_dir / "plan.json"
                if p_file.exists():
                    found_plan_path = p_file
                    if c_dir not in detected_temp_dirs:
                        detected_temp_dirs.append(c_dir)
                    break

            # Fallback to standalone plan files (mock or pre-generated)
            if not found_plan_path:
                base_stem = Path(actual_output_mkv or target_output_mkv).stem
                for alt_plan in [out_dir / f"{base_stem}.plan.json", plan_json_path]:
                    if alt_plan.exists():
                        found_plan_path = alt_plan
                        break

            if found_plan_path and found_plan_path.exists():
                try:
                    with open(found_plan_path, "r", encoding="utf-8") as f:
                        plan_data = json.load(f)
                    if found_plan_path.resolve() != plan_json_path.resolve():
                        shutil.copy2(found_plan_path, plan_json_path)
                except Exception:
                    pass

            # Deferred cleanup: If caller did not request --keep-temp, clean up now that plan.json is safely preserved
            if not keep_temp:
                try:
                    from src.cleanup import cleanup_checkpoint_dir
                    for t_dir in detected_temp_dirs:
                        if t_dir.exists():
                            cleanup_checkpoint_dir(str(t_dir), output_dir=str(out_dir))
                except Exception:
                    pass

                if custom_clip is not None and Path(custom_clip).exists():
                    try:
                        Path(custom_clip).unlink()
                    except OSError:
                        pass

    finally:
        telemetry_summary = collector.stop()
        total_time_s = max(0.001, time.perf_counter() - start_wall)

        if lock_acquired:
            try:
                from src.config import release_gpu_lock
                release_gpu_lock()
            except Exception:
                pass

    # Extract metrics and build summary
    src_frames = (
        audit_data.get("source_frames")
        or audit_data.get("mux_audit", {}).get("video_frame_count", {}).get("source_frames")
        or plan_data.get("total_frames")
        or 0
    )
    src_frames = int(src_frames)
    out_frames = (
        audit_data.get("output_frames")
        or audit_data.get("mux_audit", {}).get("video_frame_count", {}).get("output_frames")
        or src_frames
    )
    out_frames = int(out_frames)
    unique_frames = int(plan_data.get("total_kept_frames", src_frames))

    if src_frames > 0:
        dedup_ratio = round(100.0 * (1.0 - (unique_frames / src_frames)), 2)
    else:
        dedup_ratio = float(plan_data.get("dedup_ratio_percent", 0.0))

    effective_fps = round(src_frames / total_time_s, 2) if total_time_s > 0 and src_frames > 0 else 0.0
    ai_fps = float(audit_data.get("ai_steady_fps") or audit_data.get("average_pipeline_fps") or effective_fps)

    frames_matched = audit_data.get("frames_matched", (src_frames == out_frames if src_frames > 0 else True))
    mux_status = str(audit_data.get("status", "PASS" if dry_run else "UNKNOWN")).upper()

    summary: Dict[str, Any] = {
        "label": label,
        "tier": tier,
        "source_path": str(source),
        "target_output": str(target_output_mkv),
        "engine": engine,
        "zero_disk": zero_disk,
        "use_tensorrt": not no_tensorrt,
        "model": model,
        "total_time_s": round(total_time_s, 2),
        "effective_fps": effective_fps,
        "ai_fps": ai_fps,
        "total_frames": src_frames,
        "output_frames": out_frames,
        "unique_frames": unique_frames,
        "dedup_ratio": dedup_ratio,
        "peak_vram_mib": telemetry_summary.get("peak_vram_mib", 0.0),
        "avg_vram_mib": telemetry_summary.get("avg_vram_mib", 0.0),
        "avg_gpu_util": telemetry_summary.get("avg_gpu_util", 0.0),
        "avg_cpu_pct": telemetry_summary.get("avg_cpu_pct", 0.0),
        "peak_cpu_pct": telemetry_summary.get("peak_cpu_pct", 0.0),
        "avg_ssd_read_mb_s": telemetry_summary.get("avg_ssd_read_mb_s", 0.0),
        "avg_ssd_write_mb_s": telemetry_summary.get("avg_ssd_write_mb_s", 0.0),
        "mux_status": mux_status,
        "frames_matched": bool(frames_matched),
        "telemetry_csv": str(telemetry_csv),
        "audit_json": str(audit_json_path),
        "plan_json": str(plan_json_path) if plan_json_path.exists() else None,
        "micro_timings": telemetry_summary.get("micro_timings") or audit_data.get("micro_timings"),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"📊 BENCHMARK SUMMARY: [{label}] (Tier: {tier})")
    print("=" * 60)
    print(f"• Total Time:      {summary['total_time_s']} s")
    print(f"• Effective FPS:   {summary['effective_fps']} fps")
    print(f"• AI Steady FPS:   {summary['ai_fps']} fps")
    print(f"• Total Frames:    {summary['total_frames']} (Kept: {summary['unique_frames']}, Dedup: {summary['dedup_ratio']}%)")
    print(f"• Peak VRAM:       {summary['peak_vram_mib']} MiB (Avg GPU Util: {summary['avg_gpu_util']}%)")
    print(f"• Host CPU / RAM:  Avg CPU {summary['avg_cpu_pct']}%, Peak CPU {summary['peak_cpu_pct']}%")
    if summary.get("micro_timings"):
        mt = summary["micro_timings"]
        print(f"• Micro-Timings:   TRT: {mt.get('avg_trt_ms', 0)}ms | Triton: {mt.get('avg_triton_ms', 0)}ms | D2H: {mt.get('avg_d2h_ms', 0)}ms | H2D: {mt.get('avg_h2d_ms', 0)}ms")
    print(f"• Container Mux:   {summary['mux_status']} (Frames Matched: {summary['frames_matched']})")
    print(f"• Output Files:    {summary_json_path}")
    print(f"                   {telemetry_csv}")
    print(f"                   {audit_json_path}")
    if summary.get("plan_json"):
        print(f"                   {summary['plan_json']}")
    print("=" * 60 + "\n")

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run turnkey video upscale benchmark with continuous 17-metric telemetry."
    )
    parser.add_argument("source_path", help="Path to raw source video file")
    parser.add_argument(
        "--tier",
        choices=["preview", "climax", "full"],
        default="preview",
        help="Benchmark tier: preview (10s), climax (60s), or full (default: preview)",
    )
    parser.add_argument(
        "--output-dir",
        default="out/benchmarks",
        help="Destination directory for benchmark outputs (default: out/benchmarks)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Identifying label for this run (e.g. baseline_preview_10s)",
    )
    parser.add_argument(
        "--engine",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Inference engine (default: cuda)",
    )
    parser.add_argument(
        "--zero-disk",
        action="store_true",
        default=False,
        help="Enable 3-stage overlapped Zero-Disk pipe",
    )
    parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        default=False,
        help="Disable TensorRT 10 and fallback to PyTorch CUDA FP16",
    )
    parser.add_argument(
        "--model",
        default="animevideov3",
        help="Super-Resolution model identifier (default: animevideov3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for AI inference (default: 8)",
    )
    parser.add_argument(
        "--dedup-policy",
        default="default",
        help="Deduplication policy: default, sensitive-a, motion, drift, dis, off",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help="Retain temporary segment files after muxing",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override duration in seconds (overrides tier default)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Custom seek start time in seconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate execution without acquiring GPU or loading models",
    )

    args = parser.parse_args(argv)

    try:
        label = args.label or f"bench_{Path(args.source_path).stem}"
        summary = run_benchmark(
            source_path=args.source_path,
            tier=args.tier,
            output_dir=args.output_dir,
            label=label,
            engine=args.engine,
            zero_disk=args.zero_disk,
            no_tensorrt=args.no_tensorrt,
            model=args.model,
            batch_size=args.batch_size,
            dedup_policy=args.dedup_policy,
            keep_temp=args.keep_temp,
            dry_run=args.dry_run,
            duration=args.duration,
            start_time=args.start_time,
        )
        if summary.get("mux_status") != "PASS":
            return 1
        return 0
    except Exception as e:
        print(f"❌ Benchmark failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
