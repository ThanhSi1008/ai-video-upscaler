# -*- coding: utf-8 -*-
"""
tests/true_blind_test.py
Thực hiện bài kiểm tra mù thực sự (True Blind Test) cho 3 nhánh Super-Resolution:
- Branch: realesr-animevideov3 (NCNN Vulkan Metal API)
- Branch: AnimeJaNai HD V3 Compact (PyTorch MPS)
- Branch: ArtCNN C4F32_DS (PyTorch MPS với ma trận chuẩn ITU-R BT.709 Luma)
- Baseline: Bicubic 2x

Đặc điểm bắt buộc:
1. Không có bất kỳ nhãn model nào trên ảnh.
2. Tên file hoàn toàn ẩn danh (candidate_A, candidate_B, candidate_C).
3. Khóa ánh xạ lưu riêng trong SECRET_KEY.json.
4. Đo đạc tốc độ chuẩn xác trên GPU Metal (NCNN / PyTorch MPS).
"""

import os
import sys
import time
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import REALESRGAN_BIN, MODELS_DIR

OUT_DIR = Path("out/true_blind")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = Path("test_frames")

# =============================================================================
# 1. ĐỊNH NGHĨA MODEL PYTORCH MPS
# =============================================================================

class SRVGGNetCompact(nn.Module):
    """Kiến trúc SRVGGNetCompact dùng chung cho RealESRGAN và AnimeJaNai"""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu'):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat) if act_type == 'prelu' else nn.ReLU(inplace=True))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat) if act_type == 'prelu' else nn.ReLU(inplace=True))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        base = torch.repeat_interleave(x, self.upscale, dim=2)
        base = torch.repeat_interleave(base, self.upscale, dim=3)
        return out + base


class ArtCNN_C4F32(nn.Module):
    """Kiến trúc 4-layer CNN chuyên dụng cho LUMA kênh đơn của ArtCNN"""
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 4, 3, padding=1)
        self.d2s = nn.PixelShuffle(2)

    def forward(self, x):
        c0 = self.conv0(x)
        c1 = self.relu1(self.conv1(c0))
        c2 = self.relu2(self.conv2(c1))
        c3 = self.relu3(self.conv3(c2))
        c4 = self.relu4(self.conv4(c3))
        c5 = self.conv5(c4)
        c6 = self.conv6(c5 + c0)
        return torch.clamp(self.d2s(c6), 0.0, 1.0)


# =============================================================================
# 2. CHUYỂN ĐỔI KHÔNG GIAN MÀU BT.709 CHUẨN
# =============================================================================

def rgb_to_ycbcr_bt709(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chuyển RGB float32 [0, 1] sang BT.709 Full-range Y, Cb, Cr"""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = -0.114572 * r - 0.385428 * g + 0.5 * b
    cr = 0.5 * r - 0.454153 * g - 0.045847 * b
    return y, cb, cr

def ycbcr_to_rgb_bt709(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Chuyển BT.709 Y, Cb, Cr về RGB float32 [0, 1]"""
    r = y + 1.5748 * cr
    g = y - 0.187324 * cb - 0.468124 * cr
    b = y + 1.8556 * cb
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


# =============================================================================
# 3. RUNNERS CHO TỪNG NHÁNH
# =============================================================================

def run_realesr_animevideov3(input_path: Path, output_path: Path) -> float:
    cmd = [
        REALESRGAN_BIN,
        "-i", str(input_path),
        "-o", str(output_path),
        "-n", "realesr-animevideov3",
        "-s", "2",
        "-f", "png",
        "-m", MODELS_DIR
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    return t1 - t0

def run_animejanai_mps(model: nn.Module, device: torch.device, input_path: Path, output_path: Path) -> float:
    img = Image.open(input_path).convert("RGB")
    arr = (np.array(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    inp = torch.from_numpy(arr).unsqueeze(0).to(device)

    torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(inp)
    torch.mps.synchronize()
    t1 = time.perf_counter()

    out_arr = (torch.clamp(out[0], 0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(out_arr).save(output_path)
    return t1 - t0

def run_artcnn_mps(model: nn.Module, device: torch.device, input_path: Path, output_path: Path) -> Tuple[float, float]:
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    rgb = np.array(img).astype(np.float32) / 255.0

    # Chuyển sang chuẩn ITU-R BT.709 Luma
    y, cb, cr = rgb_to_ycbcr_bt709(rgb)
    inp_y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)

    torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_y = model(inp_y)
    torch.mps.synchronize()
    t1 = time.perf_counter()

    out_y_np = out_y[0, 0].cpu().numpy()

    # Chroma nội suy Bicubic lên 4K
    cb_4k = cv2.resize(cb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    cr_4k = cv2.resize(cr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Chuyển ngược lại RGB BT.709
    rgb_4k = ycbcr_to_rgb_bt709(out_y_np, cb_4k, cr_4k)
    out_img = (rgb_4k * 255.0).astype(np.uint8)
    Image.fromarray(out_img).save(output_path)

    # Đo độ lệch Mean Absolute Diff so với Bicubic Luma
    bicubic_y = cv2.resize(y, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    mad_vs_bicubic = float(np.mean(np.abs(out_y_np - bicubic_y)) * 255.0)

    return (t1 - t0), mad_vs_bicubic


# =============================================================================
# 4. THỰC THI BENCHMARK & TẠO ẢNH MÙ
# =============================================================================

def main():
    print("=" * 72)
    print("🔬 KHỞI TẠO BÀI TEST MÙ THỰC SỰ (TRUE BLIND TEST) — 3 NHÁNH AI UPSCALE")
    print("=" * 72)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"⚡ Thiết bị suy luận PyTorch: {device}")

    # 1. Load AnimeJaNai V3 Compact
    aj_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
    sd_aj = torch.load('models_bench/2x_AnimeJaNai_HD_V3_ModelsOnly/2x_AnimeJaNai_HD_V3_Compact.pth', map_location='cpu', weights_only=False)
    aj_model.load_state_dict(sd_aj.get('params_ema', sd_aj.get('params', sd_aj)), strict=True)
    aj_model = aj_model.to(device).eval()

    # 2. Load ArtCNN C4F32_DS
    art_model = ArtCNN_C4F32()
    onnx_m = onnx.load('models_bench/ArtCNN_C4F32_DS.onnx')
    weights_art = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in onnx_m.graph.initializer}
    art_model.conv0.weight.data = weights_art['kernel_oih_0']
    art_model.conv0.bias.data = weights_art['const_1']
    art_model.conv1.weight.data = weights_art['kernel_oih_1']
    art_model.conv1.bias.data = weights_art['const_3']
    art_model.conv2.weight.data = weights_art['kernel_oih_2']
    art_model.conv2.bias.data = weights_art['const_5']
    art_model.conv3.weight.data = weights_art['kernel_oih_3']
    art_model.conv3.bias.data = weights_art['const_7']
    art_model.conv4.weight.data = weights_art['kernel_oih_4']
    art_model.conv4.bias.data = weights_art['const_9']
    art_model.conv5.weight.data = weights_art['kernel_oih_5']
    art_model.conv5.bias.data = weights_art['const_11']
    art_model.conv6.weight.data = weights_art['kernel_oih_6']
    art_model.conv6.bias.data = weights_art['const_13']
    art_model = art_model.to(device).eval()

    # Danh sách cảnh test
    scenes = [
        {"name": "scene1_closeup", "title": "Cận cảnh nhân vật (Mắt/Tóc/Viền)", "file": SAMPLES_DIR / "frame_close_up.png", "crop": (800, 300, 1300, 800)},
        {"name": "scene2_scenery", "title": "Hậu cảnh kiến trúc & nếp áo", "file": SAMPLES_DIR / "frame_scenery.png", "crop": (600, 400, 1100, 900)},
        {"name": "scene3_action",  "title": "Chuyển động nhanh & ruy băng bay", "file": SAMPLES_DIR / "frame_action.png", "crop": (700, 350, 1200, 850)}
    ]

    all_keys = {}
    speed_stats = {"realesr-animevideov3": [], "AnimeJaNai_HD_V3_Compact": [], "ArtCNN_C4F32_DS": []}
    art_mads = []

    for scene in scenes:
        s_id = scene["name"]
        inp_f = scene["file"]
        print(f"\n🎬 Xử lý: {scene['title']} ({inp_f.name})...")

        orig = Image.open(inp_f).convert("RGB")
        w, h = orig.size

        # A. Baseline Bicubic 2x
        bicubic_4k = orig.resize((w * 2, h * 2), Image.BICUBIC)

        # B. Chạy 3 mô hình ra 3 file tạm
        tmp_dir = OUT_DIR / "temp"
        tmp_dir.mkdir(exist_ok=True)

        f_realesr = tmp_dir / f"{s_id}_realesr.png"
        t_realesr = run_realesr_animevideov3(inp_f, f_realesr)
        speed_stats["realesr-animevideov3"].append(t_realesr)

        f_aj = tmp_dir / f"{s_id}_animejanai.png"
        t_aj = run_animejanai_mps(aj_model, device, inp_f, f_aj)
        speed_stats["AnimeJaNai_HD_V3_Compact"].append(t_aj)

        f_art = tmp_dir / f"{s_id}_artcnn.png"
        t_art, mad_art = run_artcnn_mps(art_model, device, inp_f, f_art)
        speed_stats["ArtCNN_C4F32_DS"].append(t_art)
        art_mads.append(mad_art)

        print(f"  ├─ realesr-animevideov3 (NCNN Metal): {t_realesr:.3f}s ({1.0/t_realesr:.2f} fps)")
        print(f"  ├─ AnimeJaNai V3 Compact (PyTorch MPS): {t_aj:.3f}s ({1.0/t_aj:.2f} fps)")
        print(f"  └─ ArtCNN C4F32_DS (PyTorch MPS BT.709): {t_art:.3f}s ({1.0/t_art:.2f} fps) | Luma MAD vs Bicubic: {mad_art:.2f}/255")

        # C. Trích xuất Crop 100% kích thước thật (không nén, không nhãn)
        x1, y1, x2, y2 = [c * 2 for c in scene["crop"]]
        crop_bicubic = bicubic_4k.crop((x1, y1, x2, y2))
        crop_realesr = Image.open(f_realesr).convert("RGB").crop((x1, y1, x2, y2))
        crop_aj = Image.open(f_aj).convert("RGB").crop((x1, y1, x2, y2))
        crop_art = Image.open(f_art).convert("RGB").crop((x1, y1, x2, y2))

        # Lưu ảnh gốc Bicubic baseline
        crop_bicubic.save(OUT_DIR / f"{s_id}_baseline_bicubic.png")

        # Xáo trộn ngẫu nhiên thành Candidate A, B, C (Blind Mapping độc lập từng cảnh)
        candidates = [
            ("realesr-animevideov3", crop_realesr),
            ("AnimeJaNai_HD_V3_Compact", crop_aj),
            ("ArtCNN_C4F32_DS", crop_art)
        ]
        # Xáo trộn
        random.shuffle(candidates)
        letters = ["A", "B", "C"]
        mapping = {}

        for letter, (m_name, c_img) in zip(letters, candidates):
            out_name = f"{s_id}_candidate_{letter}.png"
            c_img.save(OUT_DIR / out_name)
            mapping[letter] = m_name

        all_keys[s_id] = mapping

    # Dọn dẹp thư mục tạm
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Lưu khóa bí mật riêng biệt
    with open(OUT_DIR / "SECRET_KEY.json", "w", encoding="utf-8") as f:
        json.dump(all_keys, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("🎉 ĐÃ XUẤT TOÀN BỘ ẢNH MÙ VÀO THƯ MỤC: out/true_blind/")
    print("=" * 72)
    print("Tốc độ trung bình thực đo trên GPU:")
    for m, times in speed_stats.items():
        avg_t = sum(times) / len(times)
        print(f"  • {m:26s}: {avg_t:.3f} s/frame ({1.0/avg_t:.2f} fps)")
    print(f"  • Độ lệch Luma ArtCNN vs Bicubic: {sum(art_mads)/len(art_mads):.2f} / 255")
    print("\n⚠️  KHÓA ÁNH XẠ ĐÃ ĐƯỢC CẤT KÍN TẠI: out/true_blind/SECRET_KEY.json")

if __name__ == "__main__":
    main()
