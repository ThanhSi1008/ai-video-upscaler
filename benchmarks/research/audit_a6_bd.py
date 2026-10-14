# -*- coding: utf-8 -*-
"""
scripts/audit_a6_bd.py
Thực hiện Mục A6:
1. Tách stream copy trên file BD thật (Fate Stay Night S01E02 Fateful Night.mkv).
2. Thống kê: số segment, độ dài min/median/max, tổng frame so với nguồn, kiểu frame đầu từng segment.
3. Tạo clip mới sample_1min_bd_420.mkv (00:04:00 - 00:05:00) với -pix_fmt yuv420p -g 24 -crf 0.
"""

import os
import sys
import shutil
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ffmpeg_ops import split_video_segments, get_segments_frame_counts


def main():
    bd_file = "/Users/xis108/Documents/RawSources/Fate Apocrypha/[AC] Fate Apocrypha - 02 [BD 1080p Hi10 FLAC][DualAudio][27EA0270].mkv"
    assert os.path.exists(bd_file), f"Không tìm thấy file BD: {bd_file}"

    print(f"=== A6. KIỂM TRA PHÂN ĐOẠN STREAM COPY TRÊN BD THẬT ===")
    print(f"File nguồn: {bd_file}")

    # 1. Đếm tổng frame nguồn
    print("1. Đang đếm tổng frame nguồn bằng ffprobe...")
    r_src = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        bd_file
    ], capture_output=True, text=True, check=True)
    src_total_frames = int(r_src.stdout.strip())
    print(f"   Tổng frame nguồn: {src_total_frames}")

    # 2. Tách stream copy vào scratch/segs_bd_audit
    segs_dir = "scratch/segs_bd_audit"
    shutil.rmtree(segs_dir, ignore_errors=True)
    print("2. Đang thực hiện split_video_segments (stream copy, segment_time=8s)...")
    seg_paths = split_video_segments(bd_file, segs_dir, segment_time=8)
    num_segs = len(seg_paths)
    print(f"   Số phân đoạn tạo ra: {num_segs} segments")

    # 3. Đo frame count từng segment
    print("3. Đang đếm frame từng segment...")
    seg_counts = get_segments_frame_counts(seg_paths)
    total_segs_frames = sum(seg_counts)
    min_len = int(np.min(seg_counts))
    median_len = float(np.median(seg_counts))
    max_len = int(np.max(seg_counts))

    print(f"   Độ dài min: {min_len} frames")
    print(f"   Độ dài median: {median_len} frames")
    print(f"   Độ dài max: {max_len} frames")
    print(f"   Tổng frame các segments: {total_segs_frames}")
    diff = total_segs_frames - src_total_frames
    print(f"   Chênh lệch so với nguồn: {diff:+d} frames ({'KHỚP 100%' if diff == 0 else 'LỆCH'})")

    # 4. Kiểm tra kiểu frame đầu từng segment (5 segment đầu và 5 segment ngẫu nhiên)
    print("4. Đang kiểm tra kiểu frame đầu (pict_type) từng phân đoạn...")
    first_frame_types = []
    for p in seg_paths[:10]:
        r_type = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-read_intervals", "%+1",
            "-show_entries", "frame=pict_type,key_frame",
            "-of", "csv=p=0",
            p
        ], capture_output=True, text=True)
        out_line = r_type.stdout.strip().split("\n")[0] if r_type.stdout.strip() else "Unknown"
        first_frame_types.append(out_line)
    print(f"   10 phân đoạn đầu (key_frame, pict_type): {first_frame_types[:5]}")

    # 5. Dọn dẹp thư mục tạm
    shutil.rmtree(segs_dir, ignore_errors=True)

    # 6. Tạo clip mới sample_1min_bd_420.mkv (00:04:00 - 00:05:00) đúng 4:2:0 8-bit, GOP 24
    out_clip = "samples/sample_1min_bd_420.mkv"
    os.makedirs("samples", exist_ok=True)
    print(f"\n5. Đang tạo clip chuẩn: {out_clip}...")
    cmd_clip = [
        "ffmpeg", "-y",
        "-ss", "00:04:00", "-t", "60",
        "-i", bd_file,
        "-map", "0:v:0", "-map", "0:a:1",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-g", "24",
        "-crf", "0",
        "-c:a", "copy",
        "-sn", "-map_chapters", "-1",
        out_clip
    ]
    subprocess.run(cmd_clip, check=True)

    r_clip = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets,r_frame_rate,pix_fmt",
        "-of", "json",
        out_clip
    ], capture_output=True, text=True, check=True)
    print(f"   Clip info: {r_clip.stdout.strip()}")


if __name__ == "__main__":
    main()
