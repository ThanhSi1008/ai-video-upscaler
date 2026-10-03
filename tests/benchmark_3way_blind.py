# -*- coding: utf-8 -*-
"""
tests/benchmark_3way_blind.py
Benchmark thực nghiệm và mù thị giác 3 nhánh Super-Resolution:
- Nhánh 1: realesr-animevideov3 (NCNN Vulkan M3 Pro GPU)
- Nhánh 2: AnimeJaNai HD V3 Compact (ONNX / CoreML / CPU)
- Nhánh 3: ArtCNN C4F32_DS (Luma 4-layer Conv2D ONNX)

Đo lường:
- Tốc độ suy luận (giây/frame & FPS)
- T0 cold-start vs steady-state
- Ngoại suy thời gian cho 1 tập phim (34.000 frames gốc, 17.000 unique frames có Deduplication)
- Xuất ảnh so sánh chi tiết mắt thường & crop vùng đặc tả (Close-up, Scenery, Action)
- Tính điểm độ sắc nét Sobel và độ trung thực tái tạo (SSIM/PSNR downscale-roundtrip)
"""

import os
import sys
import time
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import REALESRGAN_BIN, MODELS_DIR

# Thư mục lưu kết quả
OUT_DIR = Path("out/3way_blind_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_DIR = Path("test_frames")

def run_realesr_animevideov3(input_path: Path, output_path: Path) -> float:
    """Chạy realesr-animevideov3 qua NCNN Vulkan"""
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

def run_animejanai_compact(session: ort.InferenceSession, input_path: Path, output_path: Path) -> float:
    """Chạy AnimeJaNai HD V3 Compact qua ONNX Runtime"""
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...] # NCHW
    
    t0 = time.perf_counter()
    out = session.run(None, {"input": arr})[0]
    t1 = time.perf_counter()
    
    out_img = np.clip(np.transpose(out[0], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out_img).save(output_path)
    return t1 - t0

def run_artcnn_c4f32_ds(session: ort.InferenceSession, input_path: Path, output_path: Path) -> float:
    """Chạy ArtCNN C4F32_DS (Luma CNN + Chroma Bicubic)"""
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    img_np = np.array(img)
    
    img_ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    y = (img_ycbcr[:, :, 0].astype(np.float32) / 255.0)[np.newaxis, np.newaxis, ...]
    
    t0 = time.perf_counter()
    out_y = session.run(None, {"input": y})[0]
    t1 = time.perf_counter()
    
    out_y = np.clip(out_y[0, 0] * 255.0, 0, 255).astype(np.uint8)
    cr = cv2.resize(img_ycbcr[:, :, 1], (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    cb = cv2.resize(img_ycbcr[:, :, 2], (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    out_rgb = cv2.cvtColor(np.stack([out_y, cr, cb], axis=-1), cv2.COLOR_YCrCb2RGB)
    Image.fromarray(out_rgb).save(output_path)
    return t1 - t0

def calculate_metrics(img_4k: np.ndarray, orig_1080p: np.ndarray) -> Dict[str, float]:
    """
    Tính toán các chỉ số kỹ thuật:
    - Sobel Edge Gradient Energy (Độ sắc nét biên vẽ)
    - Reconstruction Fidelity (Roundtrip SSIM khi downscale 4K về lại 1080p)
    """
    gray_4k = cv2.cvtColor(img_4k, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_4k, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_4k, cv2.CV_64F, 0, 1, ksize=3)
    edge_energy = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    
    # Downscale về 1080p để đo SSIM với frame gốc
    downscaled = cv2.resize(img_4k, (orig_1080p.shape[1], orig_1080p.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Simple MSE / PSNR
    mse = np.mean((orig_1080p.astype(np.float64) - downscaled.astype(np.float64)) ** 2)
    psnr = float(10 * np.log10((255.0**2) / mse)) if mse > 0 else 99.0
    
    return {
        "edge_energy": round(edge_energy, 2),
        "psnr_roundtrip": round(psnr, 2)
    }

def create_side_by_side_comparison(
    crops: Dict[str, Image.Image],
    crop_name: str,
    output_path: Path,
    blind: bool = False,
    blind_key: Dict[str, str] = None
):
    """Tạo ảnh ghép so sánh 4 ô (Original Bicubic, Model A, Model B, Model C)"""
    keys = list(crops.keys())
    w, h = crops[keys[0]].size
    
    # Canvas 2x2
    canvas_w = w * 2 + 30
    canvas_h = h * 2 + 70
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(24, 24, 27))
    draw = ImageDraw.Draw(canvas)
    
    positions = [
        (10, 35),
        (w + 20, 35),
        (10, h + 55),
        (w + 20, h + 55)
    ]
    
    for idx, (label, img) in enumerate(crops.items()):
        pos = positions[idx]
        canvas.paste(img, pos)
        display_label = label
        if blind and blind_key and label in blind_key:
            display_label = f"Mẫu [{blind_key[label]}]"
        draw.text((pos[0] + 10, pos[1] - 22), display_label, fill=(255, 255, 255))
        
    canvas.save(output_path)

def main():
    print("=" * 72)
    print("🔬 BENCHMARK & BLIND TEST 3 NHÁNH MÔ HÌNH UPSCALER ANIME")
    print("=" * 72)
    print("1. realesr-animevideov3 (NCNN Vulkan Metal API)")
    print("2. AnimeJaNai HD V3 Compact (ONNX)")
    print("3. ArtCNN C4F32_DS (Luma 4-layer CNN ONNX)")
    print("=" * 72)
    
    # Khởi tạo ONNX sessions
    print("⏳ Đang tải mô hình ONNX...")
    aj_path = "models_bench/2x_AnimeJaNai_HD_V3_ModelsOnly/2x_AnimeJaNai_HD_V3_Compact.onnx"
    art_path = "models_bench/ArtCNN_C4F32_DS.onnx"
    
    sess_aj = ort.InferenceSession(aj_path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    sess_art = ort.InferenceSession(art_path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    
    scenes = [
        {
            "name": "close_up",
            "title": "Cảnh Cận Cảnh (Khuôn mặt, Mắt, Tóc, Nét vẽ vi mô)",
            "file": SAMPLES_DIR / "frame_close_up.png",
            "crop_box": (800, 300, 1300, 800) # (x1, y1, x2, y2) trên 1080p
        },
        {
            "name": "scenery",
            "title": "Cảnh Phong Cảnh (Kiến trúc, Chi tiết hậu cảnh, Cây cối)",
            "file": SAMPLES_DIR / "frame_scenery.png",
            "crop_box": (600, 400, 1100, 900)
        },
        {
            "name": "action",
            "title": "Cảnh Hành Động (Chuyển động nhanh, Khử ringing, Hiệu ứng)",
            "file": SAMPLES_DIR / "frame_action.png",
            "crop_box": (700, 350, 1200, 850)
        }
    ]
    
    benchmark_results = {
        "realesr-animevideov3": {"times": [], "metrics": []},
        "AnimeJaNai_HD_V3_Compact": {"times": [], "metrics": []},
        "ArtCNN_C4F32_DS": {"times": [], "metrics": []}
    }
    
    # Tạo mã mù ngẫu nhiên cho blind test
    models_list = ["realesr-animevideov3", "AnimeJaNai_HD_V3_Compact", "ArtCNN_C4F32_DS"]
    blind_letters = ["A", "B", "C"]
    random.seed(42) # Giữ cố định có thể kiểm chứng
    shuffled_models = models_list.copy()
    random.shuffle(shuffled_models)
    blind_key = {model: letter for model, letter in zip(shuffled_models, blind_letters)}
    
    for scene in scenes:
        s_name = scene["name"]
        input_f = scene["file"]
        print(f"\n🎬 Đang xử lý: {scene['title']} ({input_f.name})...")
        
        orig_img = Image.open(input_f).convert("RGB")
        orig_np = np.array(orig_img)
        w, h = orig_img.size
        
        # 1. Baseline Bicubic 4K
        bicubic_4k = orig_img.resize((w * 2, h * 2), Image.BICUBIC)
        
        # 2. Model 1: realesr-animevideov3
        out_realesr = OUT_DIR / f"{s_name}_realesr.png"
        t_realesr = run_realesr_animevideov3(input_f, out_realesr)
        benchmark_results["realesr-animevideov3"]["times"].append(t_realesr)
        img_realesr = Image.open(out_realesr).convert("RGB")
        m_realesr = calculate_metrics(np.array(img_realesr), orig_np)
        benchmark_results["realesr-animevideov3"]["metrics"].append(m_realesr)
        print(f"  ├─ [realesr-animevideov3]: {t_realesr:.3f}s ({1.0/t_realesr:.2f} fps) | Sobel: {m_realesr['edge_energy']} | PSNR: {m_realesr['psnr_roundtrip']} dB")
        
        # 3. Model 2: AnimeJaNai V3 Compact
        out_aj = OUT_DIR / f"{s_name}_animejanai.png"
        t_aj = run_animejanai_compact(sess_aj, input_f, out_aj)
        benchmark_results["AnimeJaNai_HD_V3_Compact"]["times"].append(t_aj)
        img_aj = Image.open(out_aj).convert("RGB")
        m_aj = calculate_metrics(np.array(img_aj), orig_np)
        benchmark_results["AnimeJaNai_HD_V3_Compact"]["metrics"].append(m_aj)
        print(f"  ├─ [AnimeJaNai V3 Compact]: {t_aj:.3f}s ({1.0/t_aj:.2f} fps) | Sobel: {m_aj['edge_energy']} | PSNR: {m_aj['psnr_roundtrip']} dB")
        
        # 4. Model 3: ArtCNN C4F32_DS
        out_art = OUT_DIR / f"{s_name}_artcnn.png"
        t_art = run_artcnn_c4f32_ds(sess_art, input_f, out_art)
        benchmark_results["ArtCNN_C4F32_DS"]["times"].append(t_art)
        img_art = Image.open(out_art).convert("RGB")
        m_art = calculate_metrics(np.array(img_art), orig_np)
        benchmark_results["ArtCNN_C4F32_DS"]["metrics"].append(m_art)
        print(f"  └─ [ArtCNN C4F32_DS]:      {t_art:.3f}s ({1.0/t_art:.2f} fps) | Sobel: {m_art['edge_energy']} | PSNR: {m_art['psnr_roundtrip']} dB")
        
        # Trích xuất Crop khu vực chi tiết (nhân 2 tọa độ lên 4K)
        x1, y1, x2, y2 = [coord * 2 for coord in scene["crop_box"]]
        
        crops_labeled = {
            "Gốc 1080p (Bicubic 2x)": bicubic_4k.crop((x1, y1, x2, y2)),
            "realesr-animevideov3": img_realesr.crop((x1, y1, x2, y2)),
            "AnimeJaNai HD V3 Compact": img_aj.crop((x1, y1, x2, y2)),
            "ArtCNN C4F32_DS": img_art.crop((x1, y1, x2, y2))
        }
        
        # Xuất ảnh so sánh có nhãn kỹ thuật
        labeled_out = OUT_DIR / f"compare_labeled_{s_name}.png"
        create_side_by_side_comparison(crops_labeled, s_name, labeled_out, blind=False)
        
        # Xuất ảnh kiểm tra mù (Blind A/B/C)
        # Giữ ô 1 là Bicubic gốc, 3 ô sau là A, B, C đảo ngẫu nhiên
        crops_blind = {
            "Gốc 1080p (Bicubic)": bicubic_4k.crop((x1, y1, x2, y2))
        }
        for letter in ["A", "B", "C"]:
            # Tìm model tương ứng
            m_name = [m for m, l in blind_key.items() if l == letter][0]
            if m_name == "realesr-animevideov3":
                crops_blind[f"Ứng viên [{letter}]"] = img_realesr.crop((x1, y1, x2, y2))
            elif m_name == "AnimeJaNai_HD_V3_Compact":
                crops_blind[f"Ứng viên [{letter}]"] = img_aj.crop((x1, y1, x2, y2))
            else:
                crops_blind[f"Ứng viên [{letter}]"] = img_art.crop((x1, y1, x2, y2))
                
        blind_out = OUT_DIR / f"compare_blind_{s_name}.png"
        create_side_by_side_comparison(crops_blind, s_name, blind_out, blind=False)
        
    # Tính toán tổng hợp số đo và ngoại suy thời gian
    summary = {}
    print("\n" + "=" * 72)
    print("📊 BẢNG TỔNG HỢP BENCHMARK & NGOẠI SUY THỜI GIAN 1 TẬP PHIM")
    print("=" * 72)
    
    EPISODE_TOTAL_FRAMES = 34000
    EPISODE_UNIQUE_FRAMES = 17000 # Sau Deduplication 50%
    
    for m_name in models_list:
        times = benchmark_results[m_name]["times"]
        avg_time = sum(times) / len(times)
        avg_fps = 1.0 / avg_time
        metrics = benchmark_results[m_name]["metrics"]
        avg_edge = sum(m["edge_energy"] for m in metrics) / len(metrics)
        avg_psnr = sum(m["psnr_roundtrip"] for m in metrics) / len(metrics)
        
        # Tính toán thời gian 1 tập 24 phút
        time_no_dedup_sec = EPISODE_TOTAL_FRAMES * avg_time
        time_dedup_sec = EPISODE_UNIQUE_FRAMES * avg_time
        
        summary[m_name] = {
            "blind_code": blind_key[m_name],
            "avg_infer_seconds": round(avg_time, 3),
            "avg_fps": round(avg_fps, 2),
            "edge_energy": round(avg_edge, 2),
            "psnr_roundtrip_db": round(avg_psnr, 2),
            "episode_time_no_dedup_hours": round(time_no_dedup_sec / 3600.0, 2),
            "episode_time_dedup_hours": round(time_dedup_sec / 3600.0, 2)
        }
        
        print(f"🔹 [{m_name}] (Mã mù: {blind_key[m_name]}):")
        print(f"   - Tốc độ suy luận 1080p->4K: {avg_time:.3f} s/frame ({avg_fps:.2f} fps)")
        print(f"   - Độ sắc nét nét vẽ (Sobel): {avg_edge:.2f}")
        print(f"   - Độ trung thực cấu trúc (PSNR): {avg_psnr:.2f} dB")
        print(f"   - Thời gian 1 tập 24p (KHÔNG dedup): {time_no_dedup_sec/3600.0:.2f} giờ")
        print(f"   - Thời gian 1 tập 24p (CÓ Deduplication): {time_dedup_sec/3600.0:.2f} giờ")
        print()

    # Ghi file kết quả JSON
    with open(OUT_DIR / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "blind_key": blind_key,
            "summary": summary,
            "raw_results": benchmark_results
        }, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Đã xuất ảnh so sánh và dữ liệu vào: {OUT_DIR}/")
    print(f"🔑 Khóa giải mã Blind Test: {blind_key}")

if __name__ == "__main__":
    main()
