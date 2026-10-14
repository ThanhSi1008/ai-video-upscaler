#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/generate_mps_blind_ab.py
Tạo bộ ảnh MÙ (True Blind A/B) giữa NCNN Vulkan và PyTorch MPS:
- 5 frames tiêu biểu:
  1. Close-up (Cận cảnh mặt nhân vật, viền mắt, đồng tử)
  2. Dark scene (Cảnh tối, gradient đổ bóng)
  3. Scenery (Phong cảnh, ngói nhà, cây cỏ)
  4. Action (Hành động nhanh, viền chuyển động)
  5. Line-art / Text (Nét vẽ thanh mảnh, chi tiết viền)
- Xáo trộn ngẫu nhiên thành Candidate A và Candidate B
- Ghi khóa giải mã vào secret_key.json (Agent không được mở)
- Xuất ảnh ra: out/mps_blind_ab/
"""

import os
import sys
import time
import json
import random
import shutil
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from src.config import REALESRGAN_BIN, MODELS_DIR

VIDEO_PATH = "samples/sample_3min.mkv"
OUT_DIR = "out/mps_blind_ab"
os.makedirs(OUT_DIR, exist_ok=True)
PTH_PATH = "models_bench/realesr-animevideov3.pth"

# 5 Frame indices tiêu biểu trong sample_3min.mkv (4.360 frames)
FRAME_MAP = {
    "scene1_closeup": 150,     # Cận cảnh mặt Eris và Rudeus
    "scene2_dark_shadow": 750, # Cảnh bóng tối trong nhà, gradient
    "scene3_scenery": 2000,    # Phong cảnh thị trấn, mái ngói
    "scene4_action": 3500,     # Hành động kiếm thuật, chuyển động nhanh
    "scene5_lineart": 4200     # Chi tiết nét tóc, trang phục tinh xảo
}

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

def extract_frame(frame_idx, out_path):
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", VIDEO_PATH,
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-vsync", "0",
        "-frames:v", "1",
        out_path
    ]
    subprocess.run(cmd, check=True)

def upscale_ncnn(in_png, out_png):
    tmp_in = os.path.join(OUT_DIR, "tmp_ncnn_in")
    tmp_out = os.path.join(OUT_DIR, "tmp_ncnn_out")
    shutil.rmtree(tmp_in, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)
    os.makedirs(tmp_in, exist_ok=True)
    os.makedirs(tmp_out, exist_ok=True)
    
    shutil.copy(in_png, os.path.join(tmp_in, "input.png"))
    cmd = [
        REALESRGAN_BIN,
        "-i", tmp_in,
        "-o", tmp_out,
        "-n", "realesr-animevideov3",
        "-s", "2",
        "-f", "png",
        "-m", MODELS_DIR,
        "-t", "0"
    ]
    subprocess.run(cmd, check=True)
    shutil.copy(os.path.join(tmp_out, "input.png"), out_png)
    shutil.rmtree(tmp_in, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)

def upscale_mps(in_png, out_png, model, device):
    img_bgr = cv2.imread(in_png)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t_in = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).half().to(device) / 255.0
    with torch.inference_mode():
        out = model(t_in)
        torch.mps.synchronize()
    np_out = (torch.clamp(out, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    cv2.imwrite(out_png, cv2.cvtColor(np_out, cv2.COLOR_RGB2BGR))

def main():
    print("=" * 80)
    print("🎯 TẠO BỘ ẢNH TRUE BLIND A/B (5 FRAMES): NCNN VULKAN vs PYTORCH MPS")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sd = torch.load(PTH_PATH, map_location="cpu", weights_only=False)
    weights = sd.get("params_ema", sd.get("params", sd))
    model_weights = {k: v.half() for k, v in weights.items() if k.startswith("body.")}
    model = RealESRAnimeVideoV3().to(device).half()
    model.load_state_dict(model_weights, strict=True)
    model.eval()

    secret_key = {}

    for scene_name, f_idx in FRAME_MAP.items():
        print(f"\n▶ Đang xử lý: {scene_name} (Frame #{f_idx})...")
        orig_png = os.path.join(OUT_DIR, f"{scene_name}_orig_1080p.png")
        extract_frame(f_idx, orig_png)

        ncnn_png = os.path.join(OUT_DIR, f"{scene_name}_raw_ncnn.png")
        mps_png = os.path.join(OUT_DIR, f"{scene_name}_raw_mps.png")

        upscale_ncnn(orig_png, ncnn_png)
        upscale_mps(orig_png, mps_png, model, device)

        # Xáo trộn A/B
        is_a_ncnn = random.choice([True, False])
        cand_a_png = os.path.join(OUT_DIR, f"{scene_name}_candidate_A.png")
        cand_b_png = os.path.join(OUT_DIR, f"{scene_name}_candidate_B.png")

        if is_a_ncnn:
            shutil.copy(ncnn_png, cand_a_png)
            shutil.copy(mps_png, cand_b_png)
            secret_key[scene_name] = {"A": "NCNN Vulkan", "B": "PyTorch MPS FP16", "frame_idx": f_idx}
        else:
            shutil.copy(mps_png, cand_a_png)
            shutil.copy(ncnn_png, cand_b_png)
            secret_key[scene_name] = {"A": "PyTorch MPS FP16", "B": "NCNN Vulkan", "frame_idx": f_idx}

        # Dọn file trung gian có tên model để đảm bảo tính mù 100%
        os.remove(ncnn_png)
        os.remove(mps_png)

        print(f"  -> Đã tạo {scene_name}_candidate_A.png và candidate_B.png")

    key_file = os.path.join(OUT_DIR, "secret_key.json")
    with open(key_file, "w", encoding="utf-8") as f:
        json.dump(secret_key, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✅ ĐÃ HOÀN TẤT BỘ TEST MÙ A/B TẠI: out/mps_blind_ab/")
    print(f"🔒 Khóa bí mật đã lưu an toàn tại: {key_file} (Chờ người dùng chấm mù)")
    print("=" * 80)

if __name__ == "__main__":
    main()
