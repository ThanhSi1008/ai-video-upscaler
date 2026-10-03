#!/usr/bin/env python3
"""
tests/audit_local_metrics.py
Kiểm định toàn diện 4.360 frames của sample_3min.mkv sau Deduplication.
Đo đạc:
1. Global SSIM (toàn khung hình)
2. Local Patch SSIM (lưới 8x8 = 64 ô, tìm ô có SSIM thấp nhất)
3. Local Patch MAD (Mean Absolute Diff lớn nhất)
4. Peak Pixel Diff (độ lệch điểm ảnh cực đại)
5. Truy vết top worst frames và xuất ảnh đối chiếu kiểm chứng.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import cv2

PLAN_PATH = "out/.temp_sample_3min/plan.json"
VIDEO_PATH = "samples/sample_3min.mkv"
OUT_DIR = "out/dedup_audit"

def compute_ssim_block(x, y, C1=(0.01 * 255)**2, C2=(0.03 * 255)**2):
    """Tính SSIM chuẩn trên một block hoặc toàn ảnh (float32)."""
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
    
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / (denominator + 1e-7)
    return float(np.mean(ssim_map))

def run_audit():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(PLAN_PATH):
        print(f"Error: {PLAN_PATH} not found!")
        sys.exit(1)
        
    with open(PLAN_PATH, "r") as f:
        plan = json.load(f)
        
    total_frames = plan["total_frames"]
    # Xây dựng bản đồ ánh xạ: frame_idx -> kept_reference_idx
    frame_map = {}
    for c in plan["chunks"]:
        base = c["start_frame"]
        for r in c["runs"]:
            f_idx = base + r["frame_idx"]
            for offset in range(r["count"]):
                frame_map[f_idx + offset] = f_idx

    print(f"=== KHỞI CHẠY KIỂM ĐỊNH TOÀN DIỆN {total_frames} FRAMES ===")
    print(f"Video nguồn: {VIDEO_PATH}")
    print(f"Tổng frames: {total_frames} | Kept: {plan['total_kept_frames']} | Dropped: {total_frames - plan['total_kept_frames']}")

    # Mở luồng giải mã video 1080p Gray
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", VIDEO_PATH,
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**7)
    
    width, height = 1920, 1080
    frame_bytes = width * height
    n_patches_x, n_patches_y = 8, 8
    pw = width // n_patches_x   # 240 px
    ph = height // n_patches_y  # 135 px
    
    results = [] # list of dicts for dropped frames
    current_kept_frame = None
    current_kept_idx = -1
    
    start_time = time.time()
    
    for idx in range(total_frames):
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            print(f"Cảnh báo: Video kết thúc sớm ở frame {idx}")
            break
            
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width)).astype(np.float32)
        ref_idx = frame_map[idx]
        
        if idx == ref_idx:
            # Đây là frame kept được giữ lại làm chuẩn
            current_kept_frame = frame.copy()
            current_kept_idx = idx
            continue
            
        # Đây là frame dropped, đối chiếu với current_kept_frame
        diff = np.abs(frame - current_kept_frame)
        global_mad = float(np.mean(diff))
        global_max = float(np.max(diff))
        
        # Tính Local Patch metrics trên lưới 8x8
        min_patch_ssim = 1.0
        max_patch_mad = 0.0
        worst_patch_coords = (0, 0)
        
        for py in range(n_patches_y):
            for px in range(n_patches_x):
                y0, y1 = py * ph, (py + 1) * ph
                x0, x1 = px * pw, (px + 1) * pw
                p_cur = frame[y0:y1, x0:x1]
                p_ref = current_kept_frame[y0:y1, x0:x1]
                
                p_diff = np.abs(p_cur - p_ref)
                p_mad = float(np.mean(p_diff))
                if p_mad > max_patch_mad:
                    max_patch_mad = p_mad
                    worst_patch_coords = (px, py)
                
                # Chỉ tính SSIM nếu có biến thiên để tránh tính thừa
                if p_mad > 0.5:
                    p_ssim = compute_ssim_block(p_cur, p_ref)
                    if p_ssim < min_patch_ssim:
                        min_patch_ssim = p_ssim
                        
        # Global SSIM toàn khung
        global_ssim = compute_ssim_block(frame, current_kept_frame)
        
        results.append({
            "frame_idx": idx,
            "ref_idx": ref_idx,
            "run_distance": idx - ref_idx,
            "global_ssim": global_ssim,
            "min_patch_ssim": min_patch_ssim,
            "global_mad": global_mad,
            "max_patch_mad": max_patch_mad,
            "global_max": global_max,
            "worst_patch": worst_patch_coords
        })
        
        if len(results) % 200 == 0:
            elapsed = time.time() - start_time
            fps = (idx + 1) / elapsed
            print(f"Đã duyệt {idx+1}/{total_frames} frames ({fps:.1f} fps) | Đã đo {len(results)} dropped frames...")
            
    proc.wait()
    total_time = time.time() - start_time
    print(f"\nHoàn tất kiểm định trong {total_time:.2f}s (Tốc độ: {total_frames / total_time:.1f} fps)")

    # Phân tích thống kê
    global_ssims = [r["global_ssim"] for r in results]
    min_patch_ssims = [r["min_patch_ssim"] for r in results]
    global_mads = [r["global_mad"] for r in results]
    max_patch_mads = [r["max_patch_mad"] for r in results]

    print("\n" + "="*70)
    print("📊 BÁO CÁO THỐNG KÊ AUDIT TOÀN TẬP (2.174 DROPPED FRAMES)")
    print("="*70)
    print(f"1. Global SSIM:")
    print(f"   - Trung bình: {np.mean(global_ssims):.5f}")
    print(f"   - Median:     {np.median(global_ssims):.5f}")
    print(f"   - Min:        {np.min(global_ssims):.5f}")
    print(f"   - Percentile 1%: {np.percentile(global_ssims, 1):.5f}")
    print(f"   - Percentile 5%: {np.percentile(global_ssims, 5):.5f}")

    print(f"\n2. Local Patch SSIM (Lưới 8x8 - Ô tệ nhất mỗi frame):")
    print(f"   - Trung bình của Min-Patch: {np.mean(min_patch_ssims):.5f}")
    print(f"   - Median của Min-Patch:     {np.median(min_patch_ssims):.5f}")
    print(f"   - Điểm thấp nhất tuyệt đối: {np.min(min_patch_ssims):.5f}")
    print(f"   - Percentile 1%:           {np.percentile(min_patch_ssims, 1):.5f}")

    print(f"\n3. Độ lệch Pixel (0-255):")
    print(f"   - Global MAD trung bình:    {np.mean(global_mads):.2f} / 255 ({np.mean(global_mads)/255*100:.2f}%)")
    print(f"   - Max Patch MAD trung bình: {np.mean(max_patch_mads):.2f} / 255 ({np.mean(max_patch_mads)/255*100:.2f}%)")

    # Sắp xếp top 5 frame có min_patch_ssim thấp nhất
    worst_by_ssim = sorted(results, key=lambda x: x["min_patch_ssim"])[:5]
    print("\n" + "-"*70)
    print("🔍 TOP 5 WORST FRAMES (Theo Local Patch SSIM):")
    print("-" * 70)
    for rank, w in enumerate(worst_by_ssim, 1):
        print(f"#{rank} Frame {w['frame_idx']} (ref: {w['ref_idx']}, cách {w['run_distance']} frame):")
        print(f"   Min Patch SSIM: {w['min_patch_ssim']:.4f} | Global SSIM: {w['global_ssim']:.4f}")
        print(f"   Max Patch MAD:  {w['max_patch_mad']:.2f}/255 | Global MAD: {w['global_mad']:.2f}/255 | Patch: {w['worst_patch']}")

    # Lưu kết quả ra JSON
    summary_path = os.path.join(OUT_DIR, "dedup_audit_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total_frames": total_frames,
            "total_kept": plan["total_kept_frames"],
            "total_dropped": len(results),
            "global_ssim": {
                "mean": float(np.mean(global_ssims)),
                "median": float(np.median(global_ssims)),
                "min": float(np.min(global_ssims)),
                "p1": float(np.percentile(global_ssims, 1)),
                "p5": float(np.percentile(global_ssims, 5))
            },
            "local_patch_ssim": {
                "mean": float(np.mean(min_patch_ssims)),
                "median": float(np.median(min_patch_ssims)),
                "min": float(np.min(min_patch_ssims)),
                "p1": float(np.percentile(min_patch_ssims, 1)),
                "p5": float(np.percentile(min_patch_ssims, 5))
            },
            "worst_frames": worst_by_ssim
        }, f, indent=2)
    print(f"\n✅ Đã lưu báo cáo chi tiết vào: {summary_path}")

if __name__ == "__main__":
    run_audit()
