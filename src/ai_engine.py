# -*- coding: utf-8 -*-
"""
src/ai_engine.py
Điều phối việc suy luận AI Super-Resolution bằng realesrgan-ncnn-vulkan trên Apple GPU Metal / Vulkan.
Đo lường chính xác thời gian suy luận và tốc độ FPS thực tế cho từng phân đoạn.
"""

import time
import subprocess
from typing import Dict, Any

from src.config import REALESRGAN_BIN, MODELS_DIR


def upscale_frames(
    input_dir: str,
    output_dir: str,
    model: str = "realesr-animevideov3",
    scale: int = 2,
    fmt: str = "jpg",
    num_frames: int = 0
) -> Dict[str, Any]:
    """
    Chạy suy luận NCNN Vulkan trên GPU M3 Pro cho một thư mục chứa khung hình.
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
        "-m", MODELS_DIR
    ]

    subprocess.run(
        ai_cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    elapsed_time = time.time() - start_time
    fps = (num_frames / elapsed_time) if elapsed_time > 0 and num_frames > 0 else 0.0

    return {
        "elapsed_seconds": elapsed_time,
        "fps": fps,
        "frames": num_frames
    }
