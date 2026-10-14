#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/benchmark_ncnn_repeats.py
Đo lường chính xác hiệu năng ncnn qua 3 lần lặp (3 trials) cho các cấu hình:
1. -t 540 -j 1:2:2
2. -t 540 -j 2:2:2
3. -t 1080 -j 1:2:2
4. -t 1080 -j 2:2:2

Báo cáo:
- FPS trung bình ± độ lệch chuẩn (mean ± std)
- Wall-clock time trung bình ± std
- Đỉnh bộ nhớ RSS (Peak RSS MB)
"""

import os
import sys
import time
import json
import psutil
import shutil
import subprocess
import numpy as np

from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import REALESRGAN_BIN, MODELS_DIR

TEST_DIR = "out/benchmark_tiles_tmp/test_100"
RUN_OUT = "out/benchmark_tiles_tmp/out_run"

CONFIGS = [
    {"t": 540, "j": "1:2:2", "name": "t540_j122"},
    {"t": 540, "j": "2:2:2", "name": "t540_j222"},
    {"t": 1080, "j": "1:2:2", "name": "t1080_j122"},
    {"t": 1080, "j": "2:2:2", "name": "t1080_j222"},
]

def run_single_benchmark(tile_size: int, thread_cfg: str):
    shutil.rmtree(RUN_OUT, ignore_errors=True)
    os.makedirs(RUN_OUT, exist_ok=True)

    cmd = [
        REALESRGAN_BIN,
        "-i", TEST_DIR,
        "-o", RUN_OUT,
        "-n", "realesr-animevideov3",
        "-s", "2",
        "-f", "png",
        "-m", MODELS_DIR,
        "-t", str(tile_size),
        "-j", thread_cfg
    ]

    t0_start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t0_first_frame = None
    peak_rss = 0
    try:
        ps = psutil.Process(proc.pid)
    except Exception:
        ps = None

    while proc.poll() is None:
        if t0_first_frame is None and os.path.exists(RUN_OUT):
            try:
                if len(os.listdir(RUN_OUT)) > 0:
                    t0_first_frame = time.time() - t0_start
            except Exception:
                pass

        if ps:
            try:
                rss = ps.memory_info().rss / (1024 * 1024)
                if rss > peak_rss:
                    peak_rss = rss
            except Exception:
                pass
        time.sleep(0.05)

    proc.wait()
    total_time = time.time() - t0_start
    t0_first_frame = t0_first_frame or 1.5
    steady_time = max(0.1, total_time - t0_first_frame)
    num_frames = len(os.listdir(RUN_OUT)) if os.path.exists(RUN_OUT) else 100
    overall_fps = num_frames / total_time
    steady_fps = num_frames / steady_time

    shutil.rmtree(RUN_OUT, ignore_errors=True)

    return {
        "num_frames": num_frames,
        "total_time": total_time,
        "t0_first_frame": t0_first_frame,
        "overall_fps": overall_fps,
        "steady_fps": steady_fps,
        "peak_rss_mb": peak_rss
    }

def main():
    print("=" * 80)
    print("🔬 ĐO ĐẠC ĐỐI CHIẾU NCNN TILE & THREADS (3 TRIALS / CẤU HÌNH)")
    print(f"Cỡ mẫu: 100 frame 1080p | Thư mục: {TEST_DIR}")
    print("=" * 80)

    # Warm-up GPU 1 lần trước khi đo
    print("• Warm-up GPU...")
    _ = run_single_benchmark(540, "1:2:2")

    all_summary = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        t_val = cfg["t"]
        j_val = cfg["j"]
        print(f"\n▶ Đang đo cấu hình: -t {t_val} -j {j_val} ({name}) qua 3 lần lặp...")

        trials = []
        for trial_idx in range(1, 4):
            res = run_single_benchmark(t_val, j_val)
            trials.append(res)
            print(f"   Trial {trial_idx}: {res['steady_fps']:.3f} fps (steady) | {res['overall_fps']:.3f} fps (overall) | Wall: {res['total_time']:.2f}s | Peak RSS: {res['peak_rss_mb']:.1f} MB")
            time.sleep(1.0) # Hạ nhiệt nhẹ

        fps_list = [t["steady_fps"] for t in trials]
        overall_fps_list = [t["overall_fps"] for t in trials]
        time_list = [t["total_time"] for t in trials]
        rss_list = [t["peak_rss_mb"] for t in trials]

        mean_steady_fps = float(np.mean(fps_list))
        std_steady_fps = float(np.std(fps_list))
        mean_overall_fps = float(np.mean(overall_fps_list))
        std_overall_fps = float(np.std(overall_fps_list))
        mean_time = float(np.mean(time_list))
        std_time = float(np.std(time_list))
        mean_rss = float(np.mean(rss_list))
        max_rss = float(np.max(rss_list))

        summary_entry = {
            "tile": t_val,
            "threads": j_val,
            "mean_steady_fps": round(mean_steady_fps, 3),
            "std_steady_fps": round(std_steady_fps, 3),
            "mean_overall_fps": round(mean_overall_fps, 3),
            "std_overall_fps": round(std_overall_fps, 3),
            "mean_time_s": round(mean_time, 2),
            "std_time_s": round(std_time, 2),
            "mean_peak_rss_mb": round(mean_rss, 1),
            "max_peak_rss_mb": round(max_rss, 1),
            "trials": trials
        }
        all_summary[name] = summary_entry

    print("\n" + "=" * 80)
    print("📊 BẢNG TỔNG KẾT NCNN TILE & THREAD BENCHMARK (MEAN ± STD)")
    print("=" * 80)
    print(f"{'Cấu hình':<18} | {'Steady FPS (mean ± std)':<25} | {'Wall Time (s)':<18} | {'Peak RSS (MB)'}")
    print("-" * 80)
    for name, s in all_summary.items():
        fps_str = f"{s['mean_steady_fps']:.3f} ± {s['std_steady_fps']:.3f} fps"
        time_str = f"{s['mean_time_s']:.2f} ± {s['std_time_s']:.2f}s"
        rss_str = f"{s['max_peak_rss_mb']:.1f} MB"
        print(f"{name:<18} | {fps_str:<25} | {time_str:<18} | {rss_str}")
    print("=" * 80)

    out_file = "out/ncnn_tile_repeats_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"✅ Đã lưu dữ liệu đo đạc vào: {out_file}")

if __name__ == "__main__":
    main()
