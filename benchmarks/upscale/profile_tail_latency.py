#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/profile_tail_latency.py
Profile chi tiết chi phí thời gian của phần đuôi hạ mẫu Bicubic 0.5x (8K -> 4K)
so với phần thân suy luận SRVGGNetCompact (1080p -> 8K) trên cùng một frame.
Mục tiêu Task 6: Hoãn E1 trừ khi có phép đo cùng điều kiện cho thấy đuôi 8K tốn >= 5% thời gian.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cuda_engine import RealESRAnimeVideoV3
from src.config import DEVICE_NAME


def run_tail_profiling(precision: str = "fp16", num_iters: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"🔬 BẮT ĐẦU PROFILE ĐUÔI 8K TRÊN {DEVICE_NAME.upper()} ({precision.upper()})")
    print(f"   • Thiết bị: {device} ({DEVICE_NAME})")
    print(f"   • Số lần lặp đo đạc: {num_iters} iterations (sau 5 iterations warm-up)")
    print(f"{'='*70}")

    weights_path = "models_bench/realesr-animevideov3.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_path}")

    # Khởi tạo mô hình
    model = RealESRAnimeVideoV3().to(device)
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = sd.get("params_ema", sd.get("params", sd))
    clean_weights = {k: v for k, v in weights.items() if k.startswith("body.")}

    if precision == "fp16":
        model = model.half()
        clean_weights = {k: v.half() for k, v in clean_weights.items()}
    else:
        model = model.float()
        clean_weights = {k: v.float() for k, v in clean_weights.items()}

    model.load_state_dict(clean_weights, strict=True)
    model.eval()

    # Tạo frame 1080p chuẩn (1 x 3 x 1080 x 1920)
    input_tensor = torch.rand(1, 3, 1080, 1920, device=device)
    if precision == "fp16":
        input_tensor = input_tensor.half()

    # Tách các thành phần của model:
    # Part 1: Body + Upsampler + Nearest 4x base -> out_4x (8K)
    # Part 2: Bicubic 0.5x tail -> out_2x (4K)
    body = model.body
    upsampler = model.upsampler

    # Warm-up 5 iterations
    print(f"🔥 Đang warm-up GPU {device} (5 frames)...")
    for _ in range(5):
        with torch.inference_mode():
            feat = input_tensor
            for layer in body:
                feat = layer(feat)
            out = upsampler(feat)
            base = F.interpolate(input_tensor, scale_factor=4.0, mode='nearest')
            out_4x = out + base
            out_2x = F.interpolate(out_4x, scale_factor=0.5, mode='bicubic', align_corners=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

    t_body_times = []
    t_tail_times = []
    t_total_times = []

    print(f"⏱️ Đang đo đạc {num_iters} lần lặp...")
    with torch.inference_mode():
        for i in range(num_iters):
            # Đo Part 1: Thân mô hình 1080p -> 8K
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            feat = input_tensor
            for layer in body:
                feat = layer(feat)
            out = upsampler(feat)
            base = F.interpolate(input_tensor, scale_factor=4.0, mode='nearest')
            out_4x = out + base

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Đo Part 2: Đuôi hạ mẫu 8K -> 4K Bicubic 0.5x
            out_2x = F.interpolate(out_4x, scale_factor=0.5, mode='bicubic', align_corners=False)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            dt_body = (t1 - t0) * 1000.0  # ms
            dt_tail = (t2 - t1) * 1000.0  # ms
            dt_total = dt_body + dt_tail

            t_body_times.append(dt_body)
            t_tail_times.append(dt_tail)
            t_total_times.append(dt_total)

    # Thống kê
    mean_body = np.mean(t_body_times)
    median_body = np.median(t_body_times)
    mean_tail = np.mean(t_tail_times)
    median_tail = np.median(t_tail_times)
    mean_total = np.mean(t_total_times)
    median_total = np.median(t_total_times)

    tail_pct_mean = (mean_tail / mean_total) * 100.0
    tail_pct_median = (median_tail / median_total) * 100.0

    print(f"\n📊 KẾT QUẢ ĐO ĐẠC PROFILE ({precision.upper()}):")
    print(f"   • Thân SRVGGNet (1080p -> 8K): Mean = {mean_body:.2f} ms | Median = {median_body:.2f} ms")
    print(f"   • Đuôi Bicubic (8K -> 4K):     Mean = {mean_tail:.2f} ms | Median = {median_tail:.2f} ms")
    print(f"   • Tổng thời gian forward:     Mean = {mean_total:.2f} ms | Median = {median_total:.2f} ms")
    print(f"   ---------------------------------------------------------------------")
    print(f"   👉 Tỷ lệ thời gian đuôi 8K chiếm dụng: {tail_pct_mean:.2f}% (Median: {tail_pct_median:.2f}%)")

    verdict = "HOÃN E1 (Tail < 5%)" if tail_pct_mean < 5.0 else "TIẾN HÀNH E1 (Tail >= 5%)"
    print(f"   👉 Quyết định theo tiêu chuẩn Task 6: [ {verdict} ]\n")

    return {
        "precision": precision,
        "num_iters": num_iters,
        "mean_body_ms": round(float(mean_body), 2),
        "median_body_ms": round(float(median_body), 2),
        "mean_tail_ms": round(float(mean_tail), 2),
        "median_tail_ms": round(float(median_tail), 2),
        "mean_total_ms": round(float(mean_total), 2),
        "median_total_ms": round(float(median_total), 2),
        "tail_pct_mean": round(float(tail_pct_mean), 2),
        "tail_pct_median": round(float(tail_pct_median), 2),
        "verdict": verdict,
    }


if __name__ == "__main__":
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    results_fp16 = run_tail_profiling("fp16", num_iters=50)
    results_fp32 = run_tail_profiling("fp32", num_iters=30)

    combined = {
        "device": DEVICE_NAME,
        "model": "realesr-animevideov3 (SRVGGNetCompact 64f16c x4)",
        "input_resolution": "1920x1080",
        "intermediate_resolution": "7680x4320 (8K)",
        "output_resolution": "3840x2160 (4K)",
        "fp16": results_fp16,
        "fp32": results_fp32,
    }

    report_path = out_dir / "profile_tail_8k_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"💾 Báo cáo số đo profile đã lưu tại: {report_path}")
