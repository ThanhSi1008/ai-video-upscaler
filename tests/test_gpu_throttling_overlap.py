#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_gpu_throttling_overlap.py
Kiểm tra xem GPU Metal AI FPS có bị sụt giảm khi chạy chồng gối đầu (Pipelined Double-Buffering)
với CPU Decode (FFmpeg) và Media Engine Encode (VideoToolbox 10-bit) hay không.
Chạy trên 3 phân đoạn đầu tiên của sample_3min.mkv.
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_upscale_pipeline
from src.logger import get_swap_usage, get_thermal_state

def test_overlap_fps():
    print("=" * 75)
    print("⚡ KIỂM TRA ĐỘ ỔN ĐỊNH FPS GPU KHI CHẠY CHỒNG DECODE & ENCODE")
    print("=" * 75)
    
    # Dọn dẹp thư mục tạm
    temp_dir = "out/.temp_sample_3min"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    print(f"• Thermal State ban đầu: {get_thermal_state()}")
    print(f"• Swap ban đầu: {get_swap_usage()}")
    
    # Chạy preview 24 giây (= đúng 3 chunks 8s) để kiểm tra triple overlap ở Chunk 1
    t0 = time.time()
    out_file = run_upscale_pipeline(
        input_path="samples/sample_3min.mkv",
        output_path="out/test_overlap_3chunks.mkv",
        model="realesr-animevideov3",
        segment_time=8,
        use_ramdisk=True,
        resume=False,
        preview_seconds=24, # 3 chunks x 8s
        dedup=True
    )
    dur = time.time() - t0
    
    print("\n" + "=" * 75)
    print(f"✅ Hoàn tất 3 phân đoạn trong {dur:.2f}s ({dur/60:.2f} phút)")
    print(f"• File xuất: {out_file}")
    print(f"• Thermal State sau khi chạy: {get_thermal_state()}")
    print("=" * 75)

if __name__ == "__main__":
    test_overlap_fps()
