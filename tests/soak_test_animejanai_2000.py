#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/soak_test_animejanai_2000.py
Thực hiện bài kiểm thử ngâm (Soak Test) D8 làm lại cho mô hình AnimeJaNai HD V3 Compact:
- Quy mô: 2.000 frame end-to-end trên Apple Silicon MPS FP16.
- Phân đoạn: 20 batch, mỗi batch 100 frames để phân vị P5 có ý nghĩa thống kê.
- Báo cáo fps theo từng chunk (chunk 1 đến chunk 20).
- Ghi nhận: Mean, Median, P5, P95, Min, Max fps, VRAM allocated, Process RSS.
- Cập nhật dự toán ETA toàn tập (Mushoku Tensei 14: 13.324 unique frames) theo số end-to-end thật.
"""

import os
import sys
import time
import json
import shutil
import subprocess
import numpy as np
import psutil
import torch
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mps_engine import MPSPersistentWorker, AnimeJaNaiCompact2x


def prepare_frame_pool(pool_dir: str, num_frames: int = 100):
    """Trích xuất 100 frames từ sample_3min.mkv làm pool tái sử dụng cho 20 batches."""
    os.makedirs(pool_dir, exist_ok=True)
    existing = [f for f in os.listdir(pool_dir) if f.endswith(".png")]
    if len(existing) >= num_frames:
        return sorted([os.path.join(pool_dir, f) for f in existing])[:num_frames]

    print(f"🎬 Đang trích xuất {num_frames} frames từ samples/sample_3min.mkv vào pool...")
    cmd = [
        "ffmpeg", "-y", "-i", "samples/sample_3min.mkv",
        "-vframes", str(num_frames),
        "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc",
        os.path.join(pool_dir, "frame_%04d.png")
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted([os.path.join(pool_dir, f) for f in os.listdir(pool_dir) if f.endswith(".png")])[:num_frames]


def run_soak_test_2000():
    total_frames = 2000
    batch_size = 100
    num_batches = total_frames // batch_size
    weights_path = "models_bench/2x_AnimeJaNai_HD_V3_ModelsOnly/2x_AnimeJaNai_HD_V3_Compact.pth"

    print(f"\n{'='*75}")
    print(f"🚀 BẮT ĐẦU SOAK TEST 2.000 FRAMES ANIMEJANAI HD V3 COMPACT (D8 LÀM LẠI)")
    print(f"   • Tổng số frames: {total_frames} frames end-to-end")
    print(f"   • Số lượng batch: {num_batches} batches ({batch_size} frames/batch)")
    print(f"   • Động cơ: PyTorch Metal Performance Shaders (MPS FP16)")
    print(f"   • File trọng số: {weights_path}")
    print(f"{'='*75}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise RuntimeError("Yêu cầu Apple Silicon MPS để chạy soak test này!")

    # Chuẩn bị pool 100 frames
    scratch_dir = Path("scratch/soak_animejanai_2000")
    pool_dir = scratch_dir / "pool_100"
    out_chunk_dir = scratch_dir / "out_chunk_4k"
    os.makedirs(out_chunk_dir, exist_ok=True)

    frame_files = prepare_frame_pool(str(pool_dir), num_frames=batch_size)
    print(f"✅ Pool 100 frames chuẩn bị sẵn sàng tại: {pool_dir}")

    # Khởi tạo mô hình
    print("🧠 Đang nạp mô hình AnimeJaNai HD V3 Compact vào Unified Memory...")
    worker = MPSPersistentWorker(
        model_id="animejanai-hd-v3-compact",
        weights_path=weights_path,
        precision="fp16",
        max_chunks_before_restart=100
    )

    proc = psutil.Process()
    batch_results = []
    t_test_start = time.time()

    print(f"\n{'Chunk':<8} | {'Frames':<8} | {'Duration (s)':<14} | {'Chunk FPS':<12} | {'Allocated MB':<14} | {'Driver MB':<12} | {'Process RSS'}")
    print("-" * 85)

    writer_executor = ThreadPoolExecutor(max_workers=4)

    try:
        for b_idx in range(num_batches):
            t_chunk_start = time.time()
            write_futures = []

            for idx, fpath in enumerate(frame_files):
                img = Image.open(fpath).convert("RGB")
                arr = (np.array(img).transpose((2, 0, 1)) / 255.0).astype(np.float32)
                tensor = torch.from_numpy(arr).unsqueeze(0).to(device).half()

                with torch.inference_mode():
                    out_tensor = worker.model(tensor)
                    if device.type == "mps":
                        torch.mps.synchronize()

                if torch.isnan(out_tensor).any() or torch.isinf(out_tensor).any():
                    raise RuntimeError(f"NaN/Inf detected in batch {b_idx+1}, frame {idx+1}!")

                # Xuất ảnh 4K sang thư mục chunk
                out_path = out_chunk_dir / f"chunk_{b_idx+1:02d}_{idx:04d}.png"
                out_np = (out_tensor.squeeze(0).clamp(0, 1).float().cpu().numpy().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)

                def save_task(path, array):
                    out_img = Image.fromarray(array)
                    out_img.save(path, format="PNG", compress_level=1)

                write_futures.append(writer_executor.submit(save_task, out_path, out_np))

            # Chờ ghi xong toàn bộ batch
            for f in write_futures:
                f.result()

            chunk_duration = time.time() - t_chunk_start
            chunk_fps = batch_size / chunk_duration

            c_alloc = torch.mps.current_allocated_memory() / (1024 * 1024)
            d_alloc = torch.mps.driver_allocated_memory() / (1024 * 1024) if hasattr(torch.mps, "driver_allocated_memory") else 0.0
            rss_mb = proc.memory_info().rss / (1024 * 1024)

            batch_results.append({
                "chunk_index": b_idx + 1,
                "frames": batch_size,
                "duration_seconds": round(chunk_duration, 2),
                "fps": round(chunk_fps, 3),
                "allocated_mb": round(c_alloc, 2),
                "driver_allocated_mb": round(d_alloc, 2),
                "process_rss_mb": round(rss_mb, 2)
            })

            print(f"[{b_idx+1:02d}/{num_batches:02d}]   | {batch_size:<8} | {chunk_duration:<14.2f} | {chunk_fps:<12.3f} | {c_alloc:<14.2f} | {d_alloc:<12.2f} | {rss_mb:.1f} MB")

            # Xoá các file 4K trung gian của chunk để giải phóng đĩa ngay lập tức
            for f in os.listdir(out_chunk_dir):
                try:
                    os.remove(out_chunk_dir / f)
                except Exception:
                    pass

    finally:
        writer_executor.shutdown(wait=False)

    total_test_duration = time.time() - t_test_start
    all_fps = [r["fps"] for r in batch_results]

    mean_fps = float(np.mean(all_fps))
    median_fps = float(np.median(all_fps))
    p5_fps = float(np.percentile(all_fps, 5))
    p95_fps = float(np.percentile(all_fps, 95))
    min_fps = float(np.min(all_fps))
    max_fps = float(np.max(all_fps))
    std_fps = float(np.std(all_fps))

    # Tính toán ETA toàn tập (Mushoku Tensei 14: 13.324 unique frames)
    n_unique_ep14 = 13324
    overhead_seconds = 360.0  # 6 phút overhead decode/encode/mux/copy

    eta_mean_sec = (n_unique_ep14 / mean_fps) + overhead_seconds
    eta_p5_sec = (n_unique_ep14 / p5_fps) + overhead_seconds

    def fmt_time(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h}h {m:02d}m {s:02d}s ({sec:.1f}s)"

    print(f"\n{'='*75}")
    print(f"📊 KẾT QUẢ SOAK TEST 2.000 FRAMES (20 BATCHES) - ANIMEJANAI HD V3 COMPACT")
    print(f"{'='*75}")
    print(f"   • Tổng thời gian:     {fmt_time(total_test_duration)}")
    print(f"   • Mean FPS:           {mean_fps:.3f} fps")
    print(f"   • Median FPS:         {median_fps:.3f} fps")
    print(f"   • Phân vị P5 (Worst): {p5_fps:.3f} fps (độ đo chuẩn cho ca qua đêm)")
    print(f"   • Phân vị P95 (Peak): {p95_fps:.3f} fps")
    print(f"   • Min / Max FPS:      {min_fps:.3f} / {max_fps:.3f} fps (Độ lệch chuẩn: ±{std_fps:.3f})")
    print(f"   • Đỉnh VRAM Driver:   {max(r['driver_allocated_mb'] for r in batch_results):.2f} MB")
    print(f"   • Đỉnh Process RSS:   {max(r['process_rss_mb'] for r in batch_results):.2f} MB")
    print(f"   ---------------------------------------------------------------------")
    print(f"   ⏰ CẬP NHẬT DỰ TOÁN ETA TOÀN TẬP (Mushoku Tensei 14 - 13.324 unique frames):")
    print(f"      - Theo Mean FPS ({mean_fps:.2f} fps): {fmt_time(eta_mean_sec)}")
    print(f"      - Theo P5 FPS   ({p5_fps:.2f} fps):   {fmt_time(eta_p5_sec)}")
    print(f"{'='*75}\n")

    report_data = {
        "model_id": "animejanai-hd-v3-compact",
        "precision": "fp16",
        "device": "Apple Silicon M3 Pro (MPS Metal)",
        "total_frames": total_frames,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "total_duration_seconds": round(total_test_duration, 2),
        "mean_fps": round(mean_fps, 3),
        "median_fps": round(median_fps, 3),
        "p5_fps": round(p5_fps, 3),
        "p95_fps": round(p95_fps, 3),
        "min_fps": round(min_fps, 3),
        "max_fps": round(max_fps, 3),
        "std_fps": round(std_fps, 3),
        "peak_driver_allocated_mb": round(max(r['driver_allocated_mb'] for r in batch_results), 2),
        "peak_process_rss_mb": round(max(r['process_rss_mb'] for r in batch_results), 2),
        "ep14_unique_frames": n_unique_ep14,
        "ep14_eta_mean": fmt_time(eta_mean_sec),
        "ep14_eta_p5": fmt_time(eta_p5_sec),
        "chunk_breakdown": batch_results
    }

    out_json = Path("out/animejanai_soak_2000_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Báo cáo số liệu chi tiết 20 chunks đã lưu tại: {out_json}")


if __name__ == "__main__":
    run_soak_test_2000()
