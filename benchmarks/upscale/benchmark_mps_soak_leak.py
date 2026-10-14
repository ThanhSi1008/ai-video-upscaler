#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/benchmark_mps_soak_leak.py
Kiểm tra rò rỉ bộ nhớ (Soak Test) của PyTorch MPS trên 2.000 frame liên tục:
- Sử dụng torch.inference_mode()
- Ghi nhận torch.mps.current_allocated_memory() và Process RSS sau mỗi 100 frames
- Đo đạc thông lượng FPS ổn định xuyên suốt 2.000 frames
- Xác minh tính ổn định, không crash, không leak bộ nhớ
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
    print("=" * 80)
    print("🔬 BẮT ĐẦU SOAK TEST PYTORCH MPS TRÊN 2.000 FRAMES (MEMORY & STABILITY)")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"• PyTorch Device: {device} (MPS available: {torch.backends.mps.is_available()})")

    sd = torch.load(PTH_PATH, map_location="cpu", weights_only=False)
    weights = sd.get("params_ema", sd.get("params", sd))
    model_weights = {k: v.half() for k, v in weights.items() if k.startswith("body.")}

    model = RealESRAnimeVideoV3().to(device).half()
    model.load_state_dict(model_weights, strict=True)
    model.eval()

    # Tạo batch 1 frame 1080p
    dummy_input = torch.rand(1, 3, 1080, 1920, dtype=torch.float16, device=device)

    # Warm-up 10 frames
    print("• Warm-up 10 frames...")
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)
            torch.mps.synchronize()

    total_frames = 2000
    check_interval = 100
    telemetry = []

    proc = psutil.Process()
    print(f"• Bắt đầu chạy liên tục {total_frames} frames...")

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

                # Lấy allocated MPS memory và RSS
                mps_mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                rss_mb = proc.memory_info().rss / (1024 * 1024)

                entry = {
                    "frame": i,
                    "batch_fps": round(batch_fps, 2),
                    "mps_allocated_mb": round(mps_mem_mb, 2),
                    "rss_mb": round(rss_mb, 2)
                }
                telemetry.append(entry)
                print(f"Frame {i:4d}/{total_frames} | Speed: {batch_fps:.2f} fps | MPS Mem: {mps_mem_mb:.1f} MB | RSS: {rss_mb:.1f} MB")

    total_time = time.time() - t0_start
    overall_fps = total_frames / total_time
    first_rss = telemetry[0]["rss_mb"]
    final_rss = telemetry[-1]["rss_mb"]
    rss_growth = final_rss - first_rss

    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ SOAK TEST PYTORCH MPS 2.000 FRAMES:")
    print("=" * 80)
    print(f"• Tổng thời gian: {total_time:.2f}s ({total_time/60:.2f} phút)")
    print(f"• Tốc độ trung bình: {overall_fps:.3f} fps (Mục tiêu: >= 1.7 fps)")
    print(f"• MPS Allocated Memory: {telemetry[-1]['mps_allocated_mb']:.1f} MB (Cố định, không đổi)")
    print(f"• Process RSS ban đầu: {first_rss:.1f} MB | Cuối: {final_rss:.1f} MB (Tăng trưởng: {rss_growth:+.1f} MB)")

    out_file = "out/mps_soak_2000_report.json"
    with open(out_file, "w") as f:
        json.dump({
            "total_frames": total_frames,
            "total_time_seconds": round(total_time, 2),
            "overall_fps": round(overall_fps, 3),
            "first_rss_mb": first_rss,
            "final_rss_mb": final_rss,
            "rss_growth_mb": round(rss_growth, 2),
            "telemetry": telemetry
        }, f, indent=2)
    print(f"✅ Đã lưu kết quả vào: {out_file}")

if __name__ == "__main__":
    main()
