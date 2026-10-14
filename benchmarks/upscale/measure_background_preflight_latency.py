#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/measure_background_preflight_latency.py
Đo đạc thực tế độ trễ / sụt giảm tốc độ do tiến trình preflight nền (nice -n 10):
- Case 1: Chạy 3 chunk liên tiếp (24s) KHÔNG có preflight nền
- Case 2: Chạy 3 chunk liên tiếp (24s) CÓ preflight nền (mpdecimate quét tập 14)
- Báo cáo chi tiết AI FPS và Total Chunk FPS từng chunk để so sánh đối đầu có số liệu đo đạc chứng minh cụ thể.
"""

import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path

VIDEO_SOURCE = "samples/sample_3min.mkv"
SCAN_SOURCE = "/Users/xis108/Documents/mushoku-tensei/[SubsPlease] Mushoku Tensei S3 - 14 (1080p) [73CF848B].mkv"
TEMP_DIR = "out/test_preflight_latency"


def run_test_case(case_name: str, has_bg_preflight: bool) -> dict:
    print("\n" + "=" * 80)
    print(f"🔬 BẮT ĐẦU ĐO ĐẠC: {case_name}")
    print(f"• Preflight nền: {'CÓ (nice -n 10)' if has_bg_preflight else 'KHÔNG'}")
    print("=" * 80)

    bg_proc = None
    if has_bg_preflight:
        bg_cmd = [
            "nice", "-n", "10",
            "ffmpeg", "-v", "error",
            "-i", SCAN_SOURCE,
            "-vf", "mpdecimate=hi=768:lo=320:frac=0.33,showinfo",
            "-f", "null", "-"
        ]
        print(f"⚡ Đang khởi động tiến trình preflight nền (nice -n 10) quét tập 14...")
        bg_proc = subprocess.Popen(bg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)  # Đợi tiến trình nền bắt đầu tiêu thụ CPU

    out_subdir = os.path.join(TEMP_DIR, "with_bg" if has_bg_preflight else "no_bg")
    shutil.rmtree(out_subdir, ignore_errors=True)
    os.makedirs(out_subdir, exist_ok=True)

    cmd = [
        sys.executable, "upscale.py",
        VIDEO_SOURCE,
        "--preview", "24",
        "-o", out_subdir,
        "--engine", "mps",
        "--precision", "fp16",
        "--no-subs",
        "--force"
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        wall_time = time.time() - t0
        output_text = proc.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi thực thi: {e.stderr}")
        if bg_proc:
            bg_proc.terminate()
        sys.exit(1)
    finally:
        if bg_proc:
            bg_proc.terminate()
            try:
                bg_proc.wait(timeout=2.0)
            except Exception:
                bg_proc.kill()
            print("⏹️ Đã dừng tiến trình preflight nền an toàn.")

    # Tìm file log mới nhất được sinh ra
    import glob
    log_files = glob.glob(f"logs/*sample_3min_preview_24s*.log")
    log_files.sort(key=os.path.getmtime, reverse=True)
    latest_log = log_files[0] if log_files else None

    chunks_data = []
    if latest_log and os.path.exists(latest_log):
        with open(latest_log, "r", encoding="utf-8") as f:
            for line in f:
                if "[CHUNK " in line and "Hoàn tất" in line:
                    # Example: [CHUNK 01/03] Hoàn tất 192 frames (AI Metal: 38.20s @ 1.65 fps | Steady: 1.65 fps, Peak RSS: ... | Encode: 11.2s | Tổng: 51.4s | File: up_0000.mp4)
                    import re
                    m_chunk = re.search(r'\[CHUNK (\d+)/(\d+)\]', line)
                    m_frames = re.search(r'Hoàn tất (\d+) frames', line)
                    m_ai_fps = re.search(r'@ ([\d\.]+) fps', line)
                    m_steady = re.search(r'Steady: ([\d\.]+) fps', line)
                    m_ai_time = re.search(r'AI Metal: ([\d\.]+)s', line)
                    m_total_time = re.search(r'Tổng: ([\d\.]+)s', line)

                    c_idx = int(m_chunk.group(1)) if m_chunk else len(chunks_data) + 1
                    frames = int(m_frames.group(1)) if m_frames else 192
                    ai_fps = float(m_ai_fps.group(1)) if m_ai_fps else 0.0
                    steady_fps = float(m_steady.group(1)) if m_steady else ai_fps
                    ai_time = float(m_ai_time.group(1)) if m_ai_time else 0.0
                    total_time = float(m_total_time.group(1)) if m_total_time else 0.0

                    chunks_data.append({
                        "chunk": c_idx,
                        "frames": frames,
                        "ai_time_seconds": ai_time,
                        "ai_fps": ai_fps,
                        "steady_fps": steady_fps,
                        "total_chunk_time_seconds": total_time
                    })

    return {
        "case_name": case_name,
        "has_bg_preflight": has_bg_preflight,
        "total_wall_time_seconds": round(wall_time, 2),
        "chunks": chunks_data
    }


def main():
    print("=" * 80)
    print("🔬 BÀI ĐO ĐỐI ĐẦU: ĐỘ TRỄ DO PREFLIGHT NỀN (NICE -N 10) TRÊN 3 CHUNKS (24S)")
    print("=" * 80)

    # 1. Đo không có preflight nền
    res_no_bg = run_test_case("Case A: KHÔNG có Preflight nền", has_bg_preflight=False)

    # Đợi 3 giây để hệ thống ổn định
    time.sleep(3.0)

    # 2. Đo có preflight nền
    res_with_bg = run_test_case("Case B: CÓ Preflight nền (mpdecimate nice -n 10)", has_bg_preflight=True)

    print("\n" + "=" * 90)
    print("📊 BẢNG TỔNG HỢP SO SÁNH ĐỐI ĐẦU 3 CHUNK (CÓ VS KHÔNG CÓ PREFLIGHT NỀN)")
    print("=" * 90)
    print(f"{'Chunk':<8} | {'KHÔNG nền (AI FPS)':<22} | {'CÓ nền (AI FPS)':<22} | {'Chênh lệch FPS':<18} | {'% Ảnh hưởng'}")
    print("-" * 90)

    comparison_chunks = []
    c_count = min(len(res_no_bg["chunks"]), len(res_with_bg["chunks"]))
    for i in range(c_count):
        c_no = res_no_bg["chunks"][i]
        c_bg = res_with_bg["chunks"][i]
        fps_no = c_no["steady_fps"]
        fps_bg = c_bg["steady_fps"]
        diff = fps_bg - fps_no
        pct = (diff / fps_no) * 100.0 if fps_no > 0 else 0.0

        comparison_chunks.append({
            "chunk": i + 1,
            "no_bg_steady_fps": fps_no,
            "with_bg_steady_fps": fps_bg,
            "diff_fps": round(diff, 3),
            "diff_percent": round(pct, 2)
        })

        sign = "+" if diff >= 0 else ""
        print(f"Chunk {i+1:02d} | {fps_no:<22.3f} | {fps_bg:<22.3f} | {sign}{diff:<17.3f} | {sign}{pct:.2f}%")

    print("-" * 90)
    print(f"Tổng wall-time : {res_no_bg['total_wall_time_seconds']:.2f}s (không nền) vs {res_with_bg['total_wall_time_seconds']:.2f}s (có nền)")
    time_diff = res_with_bg["total_wall_time_seconds"] - res_no_bg["total_wall_time_seconds"]
    time_pct = (time_diff / res_no_bg["total_wall_time_seconds"]) * 100.0
    sign_t = "+" if time_diff >= 0 else ""
    print(f"Độ trễ tổng thể: {sign_t}{time_diff:.2f}s ({sign_t}{time_pct:.2f}%)")
    print("=" * 90)

    report = {
        "benchmark": "background_preflight_latency",
        "video_source": VIDEO_SOURCE,
        "scan_source": SCAN_SOURCE,
        "no_bg": res_no_bg,
        "with_bg": res_with_bg,
        "comparison": comparison_chunks,
        "wall_time_diff_seconds": round(time_diff, 2),
        "wall_time_diff_percent": round(time_pct, 2)
    }

    out_file = "out/preflight_latency_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Đã lưu kết quả tại: {out_file}\n")


if __name__ == "__main__":
    main()
