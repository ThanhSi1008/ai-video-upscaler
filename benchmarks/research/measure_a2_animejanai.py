# -*- coding: utf-8 -*-
"""
scripts/measure_a2_animejanai.py
Đo đạc số liệu thô A2 cho AnimeJaNai HD V3 Compact:
So sánh độc lập giữa:
1. Tham chiếu độc lập: Spandrel CPU (FP32)
2. Pipeline nội bộ: AnimeJaNai CPU (FP32)
3. Pipeline nội bộ: AnimeJaNai GPU / CUDA (FP16)
trên 5 frame ảnh thực tế (crop 960x540).
"""

import os
import sys
from pathlib import Path
import hashlib
import numpy as np
import torch
import spandrel
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models_resolver import load_models_registry
from src.cuda_engine import AnimeJaNaiCompact2x


def calc_metrics(arr_ref: np.ndarray, arr_test: np.ndarray, border_px: int = 10):
    """
    arr_ref, arr_test: float32 trong [0.0, 1.0], shape (H, W, 3).
    """
    diff_float = np.abs(arr_ref - arr_test)
    max_diff_float = float(np.max(diff_float))

    ref_u8 = np.clip(np.round(arr_ref * 255.0), 0, 255).astype(np.uint8)
    test_u8 = np.clip(np.round(arr_test * 255.0), 0, 255).astype(np.uint8)
    diff_u8 = np.abs(ref_u8.astype(np.int16) - test_u8.astype(np.int16))
    max_diff_u8 = int(np.max(diff_u8))

    # PSNR toàn khung
    mse_all = np.mean((arr_ref - arr_test) ** 2)
    psnr_all = 999.0 if mse_all == 0 else float(10.0 * np.log10(1.0 / mse_all))

    # Tách vùng giữa và viền
    H, W, _ = arr_ref.shape
    b = border_px
    center_ref = arr_ref[b:H-b, b:W-b, :]
    center_test = arr_test[b:H-b, b:W-b, :]
    mse_center = np.mean((center_ref - center_test) ** 2)
    psnr_center = 999.0 if mse_center == 0 else float(10.0 * np.log10(1.0 / mse_center))

    # Dải viền
    border_mask = np.ones((H, W), dtype=bool)
    border_mask[b:H-b, b:W-b] = False
    border_ref = arr_ref[border_mask]
    border_test = arr_test[border_mask]
    mse_border = np.mean((border_ref - border_test) ** 2)
    psnr_border = 999.0 if mse_border == 0 else float(10.0 * np.log10(1.0 / mse_border))

    # Chênh độ sáng |ΔY| theo ITU-R BT.601/709 luma
    y_ref = 0.299 * arr_ref[:, :, 0] + 0.587 * arr_ref[:, :, 1] + 0.114 * arr_ref[:, :, 2]
    y_test = 0.299 * arr_test[:, :, 0] + 0.587 * arr_test[:, :, 1] + 0.114 * arr_test[:, :, 2]
    delta_y = float(np.abs(np.mean(y_ref) - np.mean(y_test)) * 255.0)

    return {
        "max_diff_float": max_diff_float,
        "max_diff_u8": max_diff_u8,
        "psnr_all": round(psnr_all, 2),
        "psnr_center": round(psnr_center, 2),
        "psnr_border": round(psnr_border, 2),
        "delta_y": round(delta_y, 4),
    }


def main():
    registry = load_models_registry()
    m_info = registry["animejanai-hd-v3-compact"]
    weights_path = m_info["weights_path"]

    # 1. Nạp Spandrel CPU FP32 (tham chiếu độc lập)
    loader = spandrel.ModelLoader()
    ref_desc = loader.load_from_file(weights_path)
    ref_model = ref_desc.model.eval().to("cpu")

    # 2. Nạp Our AnimeJaNai CPU FP32
    our_cpu = AnimeJaNaiCompact2x().eval().to("cpu")
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = sd.get("params_ema", sd.get("params", sd))
    clean_weights = {k: v for k, v in weights.items() if k.startswith("body.")}
    our_cpu.load_state_dict(clean_weights, strict=True)

    # 3. Nạp Our AnimeJaNai GPU FP16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    our_gpu = AnimeJaNaiCompact2x().eval().to(device).half()
    our_gpu.load_state_dict(clean_weights, strict=True)

    # Thu thập 5 frame (crop 960x540)
    img_paths = [
        "test_frames/frame_action.png",
        "test_frames/frame_close_up.png",
        "test_frames/frame_scenery.png",
    ]
    # Trích thêm 2 frame nếu cần
    import subprocess
    os.makedirs("test_frames/extra", exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-ss", "00:00:05", "-i", "samples/sample_30s.mkv",
        "-vframes", "1", "test_frames/extra/frame_04.png"
    ], capture_output=True)
    subprocess.run([
        "ffmpeg", "-y", "-ss", "00:00:15", "-i", "samples/sample_30s.mkv",
        "-vframes", "1", "test_frames/extra/frame_05.png"
    ], capture_output=True)
    img_paths.extend(["test_frames/extra/frame_04.png", "test_frames/extra/frame_05.png"])

    print("=== KẾT QUẢ ĐO ĐẠC ĐỘC LẬP A2: ANIMEJANAI ===")
    print(f"Tổng số frame: {len(img_paths)} (Crop 960x540)")

    for idx, p in enumerate(img_paths, 1):
        img = Image.open(p).convert("RGB")
        # Crop 960x540 từ giữa
        w, h = img.size
        left = (w - 960) // 2
        top = (h - 540) // 2
        crop_img = img.crop((left, top, left + 960, top + 540))

        arr_in = np.array(crop_img).astype(np.float32) / 255.0
        t_in_cpu = torch.from_numpy(arr_in).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 540, 960)

        with torch.no_grad():
            # Ref Spandrel CPU FP32
            out_ref = torch.clamp(ref_model(t_in_cpu), 0.0, 1.0)
            arr_ref = out_ref.squeeze(0).permute(1, 2, 0).numpy()

            # Our CPU FP32
            out_cpu = torch.clamp(our_cpu(t_in_cpu), 0.0, 1.0)
            arr_cpu = out_cpu.squeeze(0).permute(1, 2, 0).numpy()

            # Our CUDA FP16
            t_in_gpu = t_in_cpu.to(device).half()
            out_gpu = torch.clamp(our_gpu(t_in_gpu), 0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            arr_gpu = out_gpu.to("cpu", dtype=torch.float32).squeeze(0).permute(1, 2, 0).numpy()

        # So sánh 1: Our CPU FP32 vs Spandrel CPU FP32
        m_cpu = calc_metrics(arr_ref, arr_cpu)
        # So sánh 2: Our CUDA FP16 vs Spandrel CPU FP32
        m_gpu = calc_metrics(arr_ref, arr_gpu)

        print(f"\n--- Frame {idx}: {os.path.basename(p)} ---")
        print(f"  [Our CPU FP32 vs Spandrel Ref FP32]:")
        print(f"    max |diff| float: {m_cpu['max_diff_float']:.8e}, uint8: {m_cpu['max_diff_u8']}")
        print(f"    PSNR all: {m_cpu['psnr_all']} dB, center: {m_cpu['psnr_center']} dB, border: {m_cpu['psnr_border']} dB, |ΔY|: {m_cpu['delta_y']:.4f}/255")
        print(f"  [Our CUDA FP16 vs Spandrel Ref FP32]:")
        print(f"    max |diff| float: {m_gpu['max_diff_float']:.6f}, uint8: {m_gpu['max_diff_u8']}")
        print(f"    PSNR all: {m_gpu['psnr_all']} dB, center: {m_gpu['psnr_center']} dB, border: {m_gpu['psnr_border']} dB, |ΔY|: {m_gpu['delta_y']:.4f}/255")


if __name__ == "__main__":
    main()
