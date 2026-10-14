# -*- coding: utf-8 -*-
"""
src/ai_engine.py
Điều phối việc suy luận AI Super-Resolution bằng realesrgan-ncnn-vulkan trên GPU (Vulkan).
Đo lường chính xác thời gian suy luận và tốc độ FPS thực tế cho từng phân đoạn.
"""

import os
import time
import subprocess
import psutil
import shutil
from typing import Dict, Any

from src.config import REALESRGAN_BIN, MODELS_DIR


def upscale_frames(
    input_dir: str,
    output_dir: str,
    model: str = "realesr-animevideov3",
    scale: int = 2,
    fmt: str = "jpg",
    num_frames: int = 0,
    tile_size: int = 540,
    thread_cfg: str = "1:2:2"
) -> Dict[str, Any]:
    """
    Chạy suy luận NCNN Vulkan trên GPU cho một thư mục chứa khung hình.
    Mặc định tối ưu: -t 540 -j 1:2:2 triệt tiêu chi phí prepadding auto-tile.
    Trả về thời gian suy luận và tốc độ FPS thực tế.
    """
    start_time = time.time()

    ai_cmd = [
        REALESRGAN_BIN,
        "-i", input_dir,
        "-o", output_dir,
        "-n", model,
        "-s", str(scale),
        "-f", fmt,
        "-m", MODELS_DIR,
        "-t", str(tile_size),
        "-j", thread_cfg
    ]

    proc = subprocess.Popen(
        ai_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    t0_seconds = None
    peak_rss = 0
    init_count = len(os.listdir(output_dir)) if os.path.exists(output_dir) else 0

    try:
        ps = psutil.Process(proc.pid)
    except Exception:
        ps = None

    while proc.poll() is None:
        if t0_seconds is None and os.path.exists(output_dir):
            try:
                if len(os.listdir(output_dir)) > init_count:
                    t0_seconds = time.time() - start_time
            except Exception:
                pass
        if ps:
            try:
                rss = ps.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
            except Exception:
                pass

        # Giám sát dung lượng ổ đĩa (RAM Disk) trong lúc AI đang xuất frames:
        # Nếu đĩa còn dưới 50 MB, lập tức dừng tiến trình NCNN để tránh crash/hỏng đĩa
        if os.path.exists(output_dir):
            try:
                free_bytes = shutil.disk_usage(output_dir).free
                if free_bytes < 50 * 1024 * 1024:  # < 50 MB
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        proc.kill()
                    raise OSError(28, f"RAM Disk cạn kiệt dung lượng khi xuất frame 4K (còn {free_bytes / (1024*1024):.1f} MB < 50 MB). Dừng khẩn cấp ENOSPC.")
            except OSError:
                raise
            except Exception:
                pass

        time.sleep(0.02)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"NCNN Vulkan lỗi (exit code: {proc.returncode})")

    elapsed_time = time.time() - start_time
    t0_seconds = t0_seconds or elapsed_time
    fps = (num_frames / elapsed_time) if elapsed_time > 0 and num_frames > 0 else 0.0
    steady_time = max(0.001, elapsed_time - t0_seconds)
    steady_frames = max(1, num_frames - 1) if num_frames > 1 else num_frames
    steady_fps = (steady_frames / steady_time) if steady_time > 0 else fps

    return {
        "elapsed_seconds": elapsed_time,
        "t0_seconds": t0_seconds,
        "steady_fps": steady_fps,
        "fps": fps,
        "frames": num_frames,
        "peak_rss_mb": peak_rss / (1024 * 1024)
    }
