#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/make_blind_ab.py
Công cụ tạo bài kiểm thử mù A/B Super-Resolution theo từng bộ anime (D9):
Lấy 5 frame đa dạng từ video nguồn, suy luận qua các mô hình yêu cầu (ví dụ: animevideov3 và animejanai-hd-v3-compact),
xáo trộn thứ tự ứng viên ngẫu nhiên (Candidate A / Candidate B),
và lưu khóa giải mã riêng trong secret_key.json.
"""

import os
import sys
import time
import json
import random
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
import numpy as np
import torch

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models_resolver import resolve_model_for_video, load_models_registry
from src.mps_engine import get_mps_worker
from src.ffmpeg_ops import probe_video


def extract_diverse_frames(video_path: str, num_frames: int = 5, tmp_dir: str = "out/blind_tmp") -> List[str]:
    """Trích xuất N khung hình phân bố đều trên chiều dài video."""
    os.makedirs(tmp_dir, exist_ok=True)
    meta = probe_video(video_path)
    duration = meta.get("duration", 180.0)

    # Trích xuất từ 15% đến 85% thời lượng phim
    start_time = duration * 0.15
    end_time = duration * 0.85
    step = (end_time - start_time) / max(1, num_frames - 1)

    frame_paths = []
    for i in range(num_frames):
        ts = start_time + i * step
        out_f = os.path.join(tmp_dir, f"source_frame_{i+1:02d}.png")
        cmd = [
            "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path,
            "-frames:v", "1", "-pix_fmt", "rgb24", out_f
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(out_f):
            frame_paths.append(out_f)

    return frame_paths


def upscale_single_frame_mps(worker, img_path: str) -> np.ndarray:
    """Suy luận 1 khung hình qua MPSPersistentWorker và trả về RGB uint8 array."""
    img = Image.open(img_path).convert("RGB")
    arr = (np.array(img).transpose((2, 0, 1)) / 255.0).astype(np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).to(worker.device)
    if worker.precision == "fp16":
        tensor = tensor.half()

    with torch.inference_mode():
        out_tensor = worker.model(tensor)
        if worker.device.type == "mps":
            torch.mps.synchronize()

    out_np = (out_tensor.squeeze(0).clamp(0, 1).float().cpu().numpy().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)
    return out_np


def make_blind_ab(
    video_path: str,
    models: List[str],
    out_dir: str = "out/blind_ab",
    num_frames: int = 5
):
    """Tạo bộ ảnh test mù A/B và file secret_key.json."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path / "_tmp_extract"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    registry = load_models_registry()
    print(f"🎬 Đang chuẩn bị bài kiểm tra mù A/B từ video: {os.path.basename(video_path)}")
    print(f"   • Các mô hình đối đầu: {models}")
    print(f"   • Số khung hình: {num_frames}")

    # 1. Trích xuất frames
    src_frames = extract_diverse_frames(video_path, num_frames=num_frames, tmp_dir=str(tmp_dir))
    if not src_frames:
        raise RuntimeError("Không thể trích xuất khung hình từ video!")

    # 2. Khởi tạo workers cho các models
    workers = {}
    for m_id in models:
        m_info = resolve_model_for_video(video_path, m_id)
        workers[m_id] = get_mps_worker(
            model_id=m_info["id"],
            weights_path=m_info["weights_path"],
            precision="fp16"
        )

    letters = ["A", "B", "C", "D"]
    secret_key = {
        "video": os.path.basename(video_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated": models,
        "frames": {}
    }

    # 3. Suy luận và xuất ảnh mù
    for idx, f_path in enumerate(src_frames):
        frame_num = idx + 1
        print(f"\n🖼️ Đang xử lý khung hình {frame_num}/{len(src_frames)}...")

        # Bicubic baseline 2x
        src_img = Image.open(f_path).convert("RGB")
        w, h = src_img.size
        bicubic_img = src_img.resize((w * 2, h * 2), resample=Image.Resampling.BICUBIC)
        bicubic_path = out_path / f"frame_{frame_num:02d}_baseline_bicubic.png"
        bicubic_img.save(bicubic_path, compress_level=1)

        # Suy luận các candidates
        candidates = []
        for m_id in models:
            worker = workers[m_id]
            out_arr = upscale_single_frame_mps(worker, f_path)
            candidates.append((m_id, out_arr))

        # Xáo trộn ngẫu nhiên thứ tự các candidate
        random.shuffle(candidates)

        frame_mapping = {}
        for c_idx, (m_id, out_arr) in enumerate(candidates):
            letter = letters[c_idx]
            cand_filename = f"frame_{frame_num:02d}_candidate_{letter}.png"
            cand_path = out_path / cand_filename
            Image.fromarray(out_arr).save(cand_path, compress_level=1)
            frame_mapping[f"Candidate_{letter}"] = {
                "model_id": m_id,
                "file": cand_filename
            }

        secret_key["frames"][f"frame_{frame_num:02d}"] = {
            "source_extract": os.path.basename(f_path),
            "baseline": f"frame_{frame_num:02d}_baseline_bicubic.png",
            "mapping": frame_mapping
        }

    # 4. Ghi secret_key.json
    key_path = out_path / "secret_key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(secret_key, f, indent=2, ensure_ascii=False)

    # Dọn dẹp thư mục tạm
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n✅ Đã tạo thành công bộ ảnh kiểm tra mù A/B tại: {out_dir}")
    print(f"🔒 Khóa giải mã ẩn danh được lưu tại: {key_path}")
    print(f"💡 Hướng dẫn: Mở các file frame_XX_candidate_A.png và frame_XX_candidate_B.png để so sánh mù ở 100% tỷ lệ phóng đại.")


def main():
    parser = argparse.ArgumentParser(description="Tạo bộ kiểm tra mù A/B Super-Resolution (D9)")
    parser.add_argument("video", help="Đường dẫn đến file video nguồn")
    parser.add_argument(
        "--models",
        default="animevideov3,animejanai-hd-v3-compact",
        help="Danh sách các model id cần đối đầu (phân tách bởi dấu phẩy)"
    )
    parser.add_argument(
        "-o", "--out-dir",
        default="out/blind_ab",
        help="Thư mục xuất ảnh kiểm tra mù (Mặc định: out/blind_ab)"
    )
    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=5,
        help="Số lượng khung hình đa dạng cần trích xuất (Mặc định: 5)"
    )

    args = parser.parse_args()
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    make_blind_ab(
        video_path=args.video,
        models=model_list,
        out_dir=args.out_dir,
        num_frames=args.num_frames
    )


if __name__ == "__main__":
    main()
