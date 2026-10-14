#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/run_mps_soak_600.py
Chạy soak test 600 frame trên PyTorch MPS FP16 (realesr-animevideov3) trong môi trường biệt lập hoàn toàn
để kiểm chứng thực nghiệm: hiện tượng sụt tốc ở frames 101–400 có tái hiện khi máy rảnh hoàn toàn không.
"""

import os
import sys
import time
import json
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

PTH_PATH = "models_bench/realesr-animevideov3.pth"


class RealESRAnimeVideoV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(3, 64, 3, 1, 1))
        self.body.append(nn.PReLU(64))
        for _ in range(16):
            self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
            self.body.append(nn.PReLU(64))
        self.body.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(4)

    def forward(self, x):
        feat = x
        for layer in self.body:
            feat = layer(feat)
        out = self.upsampler(feat)
        base = F.interpolate(x, scale_factor=4.0, mode='nearest')
        out_4x = out + base
        out_2x = F.interpolate(out_4x, scale_factor=0.5, mode='bicubic', align_corners=False)
        return out_2x


def main():
    os.environ["BENCHMARK_MODE"] = "1"
    print("=" * 80)
    print("🔬 SOAK TEST 600 FRAMES PYTORCH MPS FP16 (KIỂM CHỨNG HIỆN TƯỢNG SỤT TỐC A4)")
    print("=" * 80)

    from src.config import DEVICE_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type not in ["cuda", "mps"]:
        print("❌ Lỗi: GPU (CUDA / MPS) không khả dụng!")
        sys.exit(1)

    print(f"• Thiết bị: {DEVICE_NAME} ({device.type.upper()})")
    print(f"• Weights: {PTH_PATH}")

    sd = torch.load(PTH_PATH, map_location="cpu", weights_only=False)
    weights = sd.get("params_ema", sd.get("params", sd))
    model_weights = {k: v.half() for k, v in weights.items() if k.startswith("body.")}

    model = RealESRAnimeVideoV3().to(device).half()
    model.load_state_dict(model_weights, strict=True)
    model.eval()

    dummy_input = torch.rand(1, 3, 1080, 1920, dtype=torch.float16, device=device)

    print("• Warm-up 10 frames...")
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)
            torch.mps.synchronize()

    total_frames = 600
    check_interval = 100
    telemetry = []

    proc = psutil.Process()
    print(f"• Bắt đầu chạy đo lường {total_frames} frames (ghi nhận mỗi {check_interval} frames)...")

    t0_start = time.time()
    last_t = t0_start

    with torch.inference_mode():
        for i in range(1, total_frames + 1):
            _ = model(dummy_input)
            torch.mps.synchronize()

            if i % check_interval == 0:
                now = time.time()
                batch_time = now - last_t
                batch_fps = check_interval / batch_time
                last_t = now

                mps_mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                rss_mb = proc.memory_info().rss / (1024 * 1024)

                entry = {
                    "frame": i,
                    "batch_label": f"frames {i-check_interval+1:03d}–{i:03d}",
                    "batch_time_seconds": round(batch_time, 2),
                    "batch_fps": round(batch_fps, 3),
                    "mps_allocated_mb": round(mps_mem_mb, 2),
                    "rss_mb": round(rss_mb, 2)
                }
                telemetry.append(entry)
                print(f"  [{entry['batch_label']}] Time: {batch_time:.2f}s | FPS: {batch_fps:.3f} | MPS Alloc: {mps_mem_mb:.2f}MB | RSS: {rss_mb:.2f}MB")

    total_duration = time.time() - t0_start
    overall_fps = total_frames / total_duration

    fps_list = [e["batch_fps"] for e in telemetry]
    import numpy as np
    median_fps = float(np.median(fps_list))
    p5_fps = float(np.percentile(fps_list, 5))
    min_fps = float(np.min(fps_list))
    max_fps = float(np.max(fps_list))

    print("-" * 80)
    print(f"🎯 KẾT QUẢ SOAK 600 FRAMES:")
    print(f"• Tổng thời gian : {total_duration:.2f}s")
    print(f"• Overall FPS    : {overall_fps:.3f} fps")
    print(f"• Median FPS     : {median_fps:.3f} fps")
    print(f"• P5 FPS         : {p5_fps:.3f} fps")
    print(f"• Min FPS / Max  : {min_fps:.3f} / {max_fps:.3f} fps")
    print("=" * 80)

    report = {
        "test": "gpu_soak_600_frames",
        "device": DEVICE_NAME,
        "precision": "fp16",
        "total_frames": total_frames,
        "total_duration_seconds": round(total_duration, 2),
        "overall_fps": round(overall_fps, 3),
        "median_fps": round(median_fps, 3),
        "p5_fps": round(p5_fps, 3),
        "min_fps": round(min_fps, 3),
        "max_fps": round(max_fps, 3),
        "telemetry": telemetry
    }

    os.makedirs("out", exist_ok=True)
    out_path = "out/mps_soak_600_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Đã lưu báo cáo tại: {out_path}")


if __name__ == "__main__":
    main()
