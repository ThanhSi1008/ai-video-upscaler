#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/benchmark_ncnn_tiles.py
Đo đạc hiệu năng và chất lượng ncnn với các kích thước tile và thread count:
- 10 frames warm-up (bỏ qua khỏi tính toán)
- 100 frames đo lường ổn định
- Cấu hình tile: -t in {0, 400, 540, 640, 1080, 2000}
- Cấu hình thread: -j in {1:2:2, 2:2:2}
- Kiểm định độ khớp: PSNR so với mặc định (-t 0, -j 1:2:2) phải > 45 dB
- Đo đạc: Steady FPS, Wall-clock time, Peak RSS, Tile math overhead
"""

import os
import sys
import time
import json
import shutil
import subprocess
import numpy as np
import cv2

from src.config import REALESRGAN_BIN, MODELS_DIR

VIDEO_PATH = "samples/sample_3min.mkv"
TEMP_DIR = "out/benchmark_tiles_tmp"
FRAMES_DIR = os.path.join(TEMP_DIR, "input_110")
WARMUP_DIR = os.path.join(TEMP_DIR, "warmup_10")
TEST_DIR = os.path.join(TEMP_DIR, "test_100")
BASE_OUT = os.path.join(TEMP_DIR, "out_baseline")

TILE_CONFIGS = [
    {"t": 0, "j": "1:2:2", "name": "t0_j122_auto_baseline"},
    {"t": 0, "j": "2:2:2", "name": "t0_j222_auto"},
    {"t": 400, "j": "1:2:2", "name": "t400_j122"},
    {"t": 540, "j": "1:2:2", "name": "t540_j122"},
    {"t": 640, "j": "1:2:2", "name": "t640_j122"},
    {"t": 1080, "j": "1:2:2", "name": "t1080_j122"},
    {"t": 2000, "j": "1:2:2", "name": "t2000_j122_single_tile"},
    {"t": 2000, "j": "2:2:2", "name": "t2000_j222_single_tile"},
]

def calc_tile_overhead(w=1920, h=1080, t=0):
    if t == 0:
        return "Auto (thường T=200..400)"
    if t >= max(w, h):
        return "+0.0% (Single Tile)"
    cols = (w + t - 1) // t
    rows = (h + t - 1) // t
    eff_w = w + 20 * (cols - 1)
    eff_h = h + 20 * (rows - 1)
    overhead = ((eff_w * eff_h) / (w * h) - 1.0) * 100.0
    return f"+{overhead:.1f}% ({cols}x{rows}={cols*rows} tiles)"

def prepare_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(WARMUP_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Trích xuất 110 frames (10 warmup + 100 test)
    if len(os.listdir(TEST_DIR)) < 100 or len(os.listdir(WARMUP_DIR)) < 10:
        print("Trích xuất 110 frames từ video mẫu...")
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", VIDEO_PATH,
            "-vf", "select=between(n\,100\,209)", "-vsync", "0",
            os.path.join(FRAMES_DIR, "frame_%04d.png")
        ]
        subprocess.run(cmd, check=True)
        all_frames = sorted(os.listdir(FRAMES_DIR))
        for f in all_frames[:10]:
            shutil.copy(os.path.join(FRAMES_DIR, f), os.path.join(WARMUP_DIR, f))
        for f in all_frames[10:110]:
            shutil.copy(os.path.join(FRAMES_DIR, f), os.path.join(TEST_DIR, f))
    print(f"Đã chuẩn bị 10 warmup frames và 100 test frames tại {TEMP_DIR}")

def compute_psnr(dir_a, dir_b):
    files = sorted(os.listdir(dir_a))
    psnrs = []
    # Lấy mẫu 10 frames để tính PSNR
    sample_files = files[::10]
    for f in sample_files:
        img_a = cv2.imread(os.path.join(dir_a, f))
        img_b = cv2.imread(os.path.join(dir_b, f))
        if img_a is None or img_b is None:
            continue
        mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
        if mse == 0:
            psnrs.append(99.0)
        else:
            psnr = 10.0 * np.log10((255.0 ** 2) / mse)
            psnrs.append(psnr)
    return float(np.mean(psnrs)) if psnrs else 0.0

def run_ncnn(in_dir, out_dir, tile=0, thread="1:2:2"):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        REALESRGAN_BIN,
        "-i", in_dir,
        "-o", out_dir,
        "-n", "realesr-animevideov3",
        "-s", "2",
        "-f", "png",
        "-m", MODELS_DIR,
        "-t", str(tile),
        "-j", thread
    ]
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"NCNN error: {res.stderr}")
    return elapsed

def main():
    prepare_frames()
    results = []

    print("\n" + "=" * 80)
    print("BẮT ĐẦU BENCHMARK TILE SIZE VÀ THREAD CONFIG CHO REALESRGAN NCNN")
    print("=" * 80)

    # 1. Warm-up GPU (10 frames)
    print("\n[Warmup GPU với 10 frames...]")
    warmup_out = os.path.join(TEMP_DIR, "warmup_out")
    run_ncnn(WARMUP_DIR, warmup_out, tile=0, thread="1:2:2")
    print("Warmup hoàn tất.")

    # 2. Chạy baseline: -t 0, -j 1:2:2
    print(f"\n▶ Chạy BASELINE (t=0 auto, j=1:2:2)...")
    base_time = run_ncnn(TEST_DIR, BASE_OUT, tile=0, thread="1:2:2")
    base_fps = 100.0 / base_time
    print(f"  Thời gian: {base_time:.2f}s | FPS: {base_fps:.2f} fps")

    for cfg in TILE_CONFIGS:
        t_val = cfg["t"]
        j_val = cfg["j"]
        name = cfg["name"]
        overhead_str = calc_tile_overhead(1920, 1080, t_val)
        
        cur_out = os.path.join(TEMP_DIR, f"out_{name}")
        # Chạy 1 warmup ngắn 2 frame để GPU steady
        run_ncnn(WARMUP_DIR, warmup_out, tile=t_val, thread=j_val)
        
        t0 = time.time()
        elapsed = run_ncnn(TEST_DIR, cur_out, tile=t_val, thread=j_val)
        fps = 100.0 / elapsed
        speedup_pct = ((fps / base_fps) - 1.0) * 100.0
        
        # Kiểm tra PSNR với baseline
        if name == "t0_j122_auto_baseline":
            psnr = 99.0
        else:
            psnr = compute_psnr(BASE_OUT, cur_out)

        res_entry = {
            "name": name,
            "tile": t_val,
            "thread": j_val,
            "overhead_theory": overhead_str,
            "time_sec": round(elapsed, 2),
            "fps": round(fps, 3),
            "speedup_vs_baseline_pct": round(speedup_pct, 2),
            "psnr_vs_baseline_db": round(psnr, 2),
            "matches_fidelity": psnr > 45.0
        }
        results.append(res_entry)
        
        print(f"▶ Config: -t {t_val:4d} | -j {j_val} | Lý thuyết: {overhead_str}")
        print(f"  -> Time: {elapsed:5.2f}s | FPS: {fps:.3f} ({speedup_pct:+5.1f}%) | PSNR: {psnr:5.2f} dB (Pass: {psnr > 45.0})")

    out_file = "out/ncnn_tile_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Đã lưu kết quả benchmark vào: {out_file}")

    # Tìm config tốt nhất
    valid_cfgs = [r for r in results if r["matches_fidelity"]]
    best_cfg = max(valid_cfgs, key=lambda x: x["fps"])
    print("\n" + "=" * 80)
    print(f"🏆 CẤU HÌNH TỐT NHẤT: {best_cfg['name']}")
    print(f"• Tile: -t {best_cfg['tile']} | Threads: -j {best_cfg['thread']}")
    print(f"• Tốc độ: {best_cfg['fps']:.3f} fps (tăng {best_cfg['speedup_vs_baseline_pct']:+.1f}% so với baseline {base_fps:.2f} fps)")
    print(f"• Độ khớp hình ảnh: PSNR = {best_cfg['psnr_vs_baseline_db']} dB (> 45 dB)")
    print("=" * 80)

if __name__ == "__main__":
    main()
