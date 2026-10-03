#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/soak_end_to_end_mps.py
Chương trình kiểm thử độ bền (Soak Test) End-to-End cho PyTorch MPS FP16:
- Đọc PNG 1080p từ RAM Disk -> Suy luận Metal MPS FP16 -> Ghi PNG 4K (mức nén 1, luồng riêng) vào RAM Disk.
- Ghi nhận telemetry mỗi 100 frames:
  + batch fps, steady fps
  + current_allocated_memory (MB)
  + driver_allocated_memory (MB)
  + footprint <pid> (MB)
  + pageouts (vm_stat)
  + thermal state
- Báo cáo thống kê: Median fps, P5 fps, Mean fps, Peak driver memory, Peak footprint.
"""

import os
import sys
import time
import json
import re
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import psutil
from PIL import Image
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_RAMDISK_SIZE_MB
from src.ramdisk import RAMDisk
from src.mps_engine import MPSPersistentWorker
from src.logger import get_thermal_state
from src.models_resolver import resolve_model_for_video


def get_footprint_mb(pid: int) -> float:
    """Đo bộ nhớ thực tế (footprint) của tiến trình bằng lệnh footprint macOS."""
    try:
        r = subprocess.run(["footprint", str(pid)], capture_output=True, text=True, timeout=3)
        m = re.search(r"Footprint:\s+(\d+)\s+KB", r.stdout)
        if m:
            return round(float(m.group(1)) / 1024.0, 2)
    except Exception:
        pass
    # Fallback sang psutil rss nếu footprint không khả dụng
    return round(psutil.Process(pid).memory_info().rss / (1024.0 * 1024.0), 2)


def get_system_pageouts() -> int:
    """Lấy số lượng pageouts từ vm_stat."""
    try:
        r = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3)
        m = re.search(r"Pageouts:\s+(\d+)", r.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 0


def run_soak_test(
    model_id: str = "animevideov3",
    total_frames: int = 6000,
    batch_size: int = 100,
    report_path: str = "out/mps_soak_6000_report.json",
    video_source: str = "samples/sample_3min.mkv"
):
    print(f"\n{'='*80}")
    print(f"🚀 BẮT ĐẦU SOAK TEST END-TO-END MPS FP16: {model_id.upper()} ({total_frames} FRAMES)")
    print(f"{'='*80}")
    print(f"• Thiết bị: Apple Silicon GPU Metal ({torch.backends.mps.is_available()})")
    print(f"• Batch size ghi nhận viễn trắc: {batch_size} frames")
    print(f"• Báo cáo xuất ra: {report_path}")

    # Lấy thông tin model
    m_info = resolve_model_for_video("sample.mkv", model_id)
    weights_path = m_info["weights_path"]

    # Khởi tạo RAM Disk 4GB
    with RAMDisk(size_mb=DEFAULT_RAMDISK_SIZE_MB, enabled=True) as ramdisk_path:
        print(f"🛡️  RAM Disk đã mount tại: {ramdisk_path}")
        in_dir = os.path.join(ramdisk_path, "soak_in")
        out_dir = os.path.join(ramdisk_path, "soak_out")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # 1. Trích xuất 192 frames 1080p mẫu vào RAM Disk
        print("📥 Đang trích xuất 192 frames 1080p mẫu vào RAM Disk...")
        ext_cmd = [
            "ffmpeg", "-y", "-ss", "30", "-i", video_source,
            "-frames:v", "192", "-pix_fmt", "rgb24",
            os.path.join(in_dir, "frame_%04d.png")
        ]
        subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        sample_files = sorted([os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith(".png")])
        num_samples = len(sample_files)
        print(f"✅ Đã trích xuất {num_samples} frames 1080p vào RAM Disk.")

        # 2. Khởi tạo Persistent Worker
        t0_worker = time.time()
        worker = MPSPersistentWorker(
            model_id=model_id,
            weights_path=weights_path,
            precision="fp16",
            max_chunks_before_restart=10
        )
        print(f"⚡ Worker MPS Persistent đã sẵn sàng ({time.time() - t0_worker:.3f}s).")

        # 3. Chạy vòng lặp Soak End-to-End
        telemetry = []
        writer_pool = ThreadPoolExecutor(max_workers=4)
        write_futures = []

        pid = os.getpid()
        init_pageouts = get_system_pageouts()
        start_time = time.time()
        batch_start_time = time.time()
        driver_alloc_peak = 0.0
        curr_alloc_peak = 0.0

        for frame_idx in range(1, total_frames + 1):
            # Chọn frame input tuần hoàn từ RAM Disk
            src_path = sample_files[(frame_idx - 1) % num_samples]
            dest_path = os.path.join(out_dir, f"out_{frame_idx:06d}.png")

            # A. Đọc PNG từ RAM Disk
            img = Image.open(src_path).convert("RGB")
            arr = (np.array(img).transpose((2, 0, 1)) / 255.0).astype(np.float32)
            tensor = torch.from_numpy(arr).unsqueeze(0).to(worker.device).half()

            # B. Suy luận MPS Metal
            with torch.inference_mode():
                out_tensor = worker.model(tensor)
                torch.mps.synchronize()

            # Ghi nhận VRAM peak trong infer
            c_alloc = torch.mps.current_allocated_memory() / (1024 * 1024)
            if c_alloc > curr_alloc_peak:
                curr_alloc_peak = c_alloc
            if hasattr(torch.mps, "driver_allocated_memory"):
                d_alloc = torch.mps.driver_allocated_memory() / (1024 * 1024)
                if d_alloc > driver_alloc_peak:
                    driver_alloc_peak = d_alloc

            # C. Ghi PNG mức nén 1 qua thread pool riêng vào RAM Disk
            out_np = (out_tensor.squeeze(0).clamp(0, 1).float().cpu().numpy().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)
            def write_png_task(path, array):
                out_img = Image.fromarray(array)
                out_img.save(path, format="PNG", compress_level=1)

            write_futures.append(writer_pool.submit(write_png_task, dest_path, out_np))

            # D. Ghi nhận viễn trắc mỗi batch_size frames
            if frame_idx % batch_size == 0:
                # Chờ các file PNG trong batch ghi xong
                for f in write_futures:
                    f.result()
                write_futures.clear()

                batch_elapsed = time.time() - batch_start_time
                batch_fps = batch_size / batch_elapsed if batch_elapsed > 0 else 0.0

                current_pageouts = get_system_pageouts()
                delta_pageouts = current_pageouts - init_pageouts
                proc_footprint_mb = get_footprint_mb(pid)
                thermal_info = get_thermal_state()

                entry = {
                    "frame": frame_idx,
                    "batch_fps": round(batch_fps, 3),
                    "current_allocated_mb": round(c_alloc, 2),
                    "driver_allocated_peak_mb": round(driver_alloc_peak, 2),
                    "footprint_mb": proc_footprint_mb,
                    "pageouts_delta": delta_pageouts,
                    "thermal_state": thermal_info
                }
                telemetry.append(entry)

                print(
                    f"[{frame_idx:04d}/{total_frames:04d}] "
                    f"FPS: {batch_fps:.2f} | "
                    f"Driver VRAM: {driver_alloc_peak:.1f} MB | "
                    f"Footprint: {proc_footprint_mb:.1f} MB | "
                    f"Pageouts: +{delta_pageouts} | "
                    f"Nhiệt: {thermal_info}"
                )

                # Dọn sạch output files trên RAM Disk để giữ dung lượng an toàn (< 4GB)
                for f in os.listdir(out_dir):
                    if f.endswith(".png"):
                        try:
                            os.remove(os.path.join(out_dir, f))
                        except OSError:
                            pass

                # Reset timer batch
                batch_start_time = time.time()

        # Chờ toàn bộ writer còn lại
        for f in write_futures:
            f.result()
        writer_pool.shutdown(wait=True)

        total_elapsed = time.time() - start_time
        overall_fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0

        # Tính toán phân vị: Median và P5 (phân vị thứ 5)
        batch_fps_list = [e["batch_fps"] for e in telemetry]
        median_fps = float(np.median(batch_fps_list)) if batch_fps_list else 0.0
        p5_fps = float(np.percentile(batch_fps_list, 5)) if batch_fps_list else 0.0
        p95_fps = float(np.percentile(batch_fps_list, 95)) if batch_fps_list else 0.0
        min_fps = float(np.min(batch_fps_list)) if batch_fps_list else 0.0
        max_fps = float(np.max(batch_fps_list)) if batch_fps_list else 0.0

        report = {
            "model_id": model_id,
            "weights_path": weights_path,
            "precision": "fp16",
            "total_frames": total_frames,
            "total_time_seconds": round(total_elapsed, 2),
            "overall_fps": round(overall_fps, 3),
            "median_fps": round(median_fps, 3),
            "p5_fps": round(p5_fps, 3),
            "p95_fps": round(p95_fps, 3),
            "min_fps": round(min_fps, 3),
            "max_fps": round(max_fps, 3),
            "peak_driver_allocated_mb": round(driver_alloc_peak, 2),
            "peak_current_allocated_mb": round(curr_alloc_peak, 2),
            "initial_footprint_mb": telemetry[0]["footprint_mb"] if telemetry else 0.0,
            "final_footprint_mb": telemetry[-1]["footprint_mb"] if telemetry else 0.0,
            "total_pageouts_delta": telemetry[-1]["pageouts_delta"] if telemetry else 0,
            "telemetry": telemetry
        }

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"📊 BÁO CÁO KẾT QUẢ SOAK TEST {model_id.upper()} ({total_frames} FRAMES)")
        print(f"{'='*80}")
        print(f"• Tổng thời gian:           {total_elapsed:.2f}s ({total_elapsed/60:.2f} phút)")
        print(f"• Tốc độ trung bình (Mean): {overall_fps:.3f} fps")
        print(f"• Tốc độ Trung vị (Median): {median_fps:.3f} fps")
        print(f"• Phân vị P5 (Worst 5%):    {p5_fps:.3f} fps")
        print(f"• Phân vị P95 (Best 5%):    {p95_fps:.3f} fps")
        print(f"• Đỉnh Driver Allocated:    {driver_alloc_peak:.2f} MB")
        print(f"• Đỉnh Footprint RSS:       {telemetry[-1]['footprint_mb']} MB")
        print(f"• Tăng trưởng Pageouts:     +{report['total_pageouts_delta']}")
        print(f"• Báo cáo chi tiết:         {report_path}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Chạy kiểm thử Soak Test End-to-End cho PyTorch MPS")
    parser.add_argument("--model", default="animevideov3", choices=["animevideov3", "animejanai-hd-v3-compact"])
    parser.add_argument("--frames", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("-o", "--output", default="out/mps_soak_6000_report.json")
    parser.add_argument("--video", default="samples/sample_3min.mkv")

    args = parser.parse_args()
    run_soak_test(
        model_id=args.model,
        total_frames=args.frames,
        batch_size=args.batch,
        report_path=args.output,
        video_source=args.video
    )


if __name__ == "__main__":
    main()
