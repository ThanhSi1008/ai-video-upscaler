#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/benchmark_animejanai_fp16_vs_fp32.py
Đo đạc độ chính xác FP16 vs FP32 của AnimeJaNai HD V3 Compact trên >= 25 frames đa dạng (Mục D4):
- Gradient tối, vùng phẳng, cảnh hành động nhanh.
- Đo PSNR từng frame và toàn tập.
- Kiểm tra tính hợp lệ: NaN / Inf check.
- Tiêu chí: PSNR trung bình >= 45 dB và không có NaN/Inf thì mới kích hoạt FP16.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

VIDEO_SAMPLE = "samples/sample_3min.mkv"
TEMP_FRAMES_DIR = "scratch/animejanai_25_frames"
MODEL_PATH = "models_bench/2x_AnimeJaNai_HD_V3_ModelsOnly/2x_AnimeJaNai_HD_V3_Compact.pth"

class SRVGGNetCompact2x(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2):
        super().__init__()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        feat = x
        for layer in self.body:
            feat = layer(feat)
        out = self.upsampler(feat)
        base = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return out + base

def extract_25_diverse_frames():
    os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)
    # Trích xuất 25 frames tại các mốc giây phân bố đều qua 180s:
    # giây: 5, 12, 18, 25, 32, 40, 48, 55, 62, 70, 78, 85, 92, 100, 108, 115, 122, 130, 138, 145, 152, 160, 168, 172, 178
    timestamps = [5, 12, 18, 25, 32, 40, 48, 55, 62, 70, 78, 85, 92, 100, 108, 115, 122, 130, 138, 145, 152, 160, 168, 172, 178]
    frame_paths = []
    for idx, ts in enumerate(timestamps):
        out_png = os.path.join(TEMP_FRAMES_DIR, f"frame_{idx+1:02d}_t{ts:03d}s.png")
        if not os.path.exists(out_png):
            cmd = [
                "ffmpeg", "-y", "-ss", str(ts), "-i", VIDEO_SAMPLE,
                "-vframes", "1",
                "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc",
                out_png
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frame_paths.append(out_png)
    return frame_paths

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0:
        return 999.0
    return float(10 * np.log10((255.0 ** 2) / mse))

def main():
    print("=" * 80)
    print("🔬 KIỂM TRA ĐỘ CHÍNH XÁC ANIMEJANAI HD V3 COMPACT: FP16 VS FP32 TRÊN 25 FRAMES (D4)")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"• PyTorch Device: {device}")

    # 1. Trích xuất 25 frames
    print("• Đang trích xuất 25 frames đa dạng từ sample_3min.mkv...")
    frame_paths = extract_25_diverse_frames()
    print(f"  -> Đã chuẩn bị {len(frame_paths)} frames kiểm thử.")

    # 2. Nạp model FP32 và FP16
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    weights = sd.get("params_ema", sd)
    weights = {k: v for k, v in weights.items() if k.startswith("body.")}

    model_fp32 = SRVGGNetCompact2x().to(device).float()
    model_fp32.load_state_dict({k: v.float() for k, v in weights.items()}, strict=True)
    model_fp32.eval()

    model_fp16 = SRVGGNetCompact2x().to(device).half()
    model_fp16.load_state_dict({k: v.half() for k, v in weights.items()}, strict=True)
    model_fp16.eval()

    # Warm-up
    print("• Warm-up GPU...")
    dummy = torch.rand(1, 3, 1080, 1920, device=device)
    with torch.inference_mode():
        _ = model_fp32(dummy.float())
        _ = model_fp16(dummy.half())
        torch.mps.synchronize()

    # 3. Chạy từng frame
    results = []
    has_nan_or_inf = False
    psnr_scores = []

    print("\n• Bắt đầu suy luận đối chiếu FP16 vs FP32 trên từng frame...")
    for idx, fpath in enumerate(frame_paths):
        fname = os.path.basename(fpath)
        img = Image.open(fpath).convert("RGB")
        arr = (np.array(img).transpose((2, 0, 1)) / 255.0).astype(np.float32) # (3, H, W) in [0, 1] float32
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device) # (1, 3, H, W)

        with torch.inference_mode():
            # FP32
            out_fp32 = model_fp32(tensor.float())
            torch.mps.synchronize()

            # FP16
            out_fp16 = model_fp16(tensor.half())
            torch.mps.synchronize()

        # Kiểm tra NaN/Inf trên tensor FP16
        nan_count = int(torch.isnan(out_fp16).sum().item())
        inf_count = int(torch.isinf(out_fp16).sum().item())
        if nan_count > 0 or inf_count > 0:
            has_nan_or_inf = True

        # Đổi về uint8 numpy để đo PSNR
        out_fp32_np = (out_fp32.squeeze(0).clamp(0, 1).cpu().numpy().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)
        out_fp16_np = (out_fp16.squeeze(0).clamp(0, 1).float().cpu().numpy().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)

        psnr_val = calculate_psnr(out_fp32_np, out_fp16_np)
        psnr_scores.append(psnr_val)

        mad = float(np.mean(np.abs(out_fp32_np.astype(float) - out_fp16_np.astype(float))))
        max_diff = int(np.max(np.abs(out_fp32_np.astype(int) - out_fp16_np.astype(int))))

        res_entry = {
            "index": idx + 1,
            "filename": fname,
            "psnr_db": round(psnr_val, 2),
            "mad_255": round(mad, 3),
            "max_diff_255": max_diff,
            "nan_count": nan_count,
            "inf_count": inf_count
        }
        results.append(res_entry)
        print(f"  [{idx+1:02d}/25] {fname:<25} | PSNR: {psnr_val:.2f} dB | MAD: {mad:.3f}/255 | Max diff: {max_diff}/255 | NaN/Inf: {nan_count}/{inf_count}")

    mean_psnr = float(np.mean(psnr_scores))
    min_psnr = float(np.min(psnr_scores))
    max_psnr = float(np.max(psnr_scores))

    print("\n" + "=" * 80)
    print("📊 TỔNG KẾT KẾT QUẢ ĐỘ CHÍNH XÁC ANIMEJANAI FP16 VS FP32 (D4):")
    print("=" * 80)
    print(f"• Số frame kiểm nghiệm: {len(results)}")
    print(f"• Mean PSNR: {mean_psnr:.2f} dB (Ngưỡng yêu cầu: >= 45.0 dB)")
    print(f"• Min PSNR: {min_psnr:.2f} dB | Max PSNR: {max_psnr:.2f} dB")
    print(f"• Phát hiện NaN/Inf: {'CÓ (LỖI)' if has_nan_or_inf else 'HOÀN TOÀN KHÔNG CÓ (0 NaN, 0 Inf)'}")

    passed = (mean_psnr >= 45.0) and (not has_nan_or_inf)
    print(f"• Quyết định kỹ thuật: {'ĐẠT CHUẨN -> BẬT FP16 CHO ANIMEJANAI' if passed else 'KHÔNG ĐẠT -> PHẢI DÙNG FP32'}")
    print("=" * 80)

    out_file = "out/animejanai_fp16_vs_fp32_report.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": "AnimeJaNai HD V3 Compact",
            "weights_path": MODEL_PATH,
            "num_frames": len(results),
            "mean_psnr_db": round(mean_psnr, 2),
            "min_psnr_db": round(min_psnr, 2),
            "max_psnr_db": round(max_psnr, 2),
            "has_nan_or_inf": has_nan_or_inf,
            "meets_criteria": passed,
            "recommended_precision": "fp16" if passed else "fp32",
            "frames": results
        }, f, indent=2)
    print(f"✅ Đã lưu báo cáo chi tiết vào: {out_file}")

    shutil.rmtree(TEMP_FRAMES_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
