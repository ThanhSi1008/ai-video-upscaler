#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/audit_small_block_dedup.py
Kiểm định Deduplication với kích thước khối chuẩn 16x16 và 8x8 pixel:
- Đo đạc trên TOÀN BỘ 2.174 dropped frames của sample_3min.mkv so với reference kept frame tương ứng.
- Tính max-block MAD trên lưới 16x16 (8.160 khối/frame) và 8x8 (32.400 khối/frame).
- Vẽ/thống kê phân bố sai lệch (histogram & percentiles: P50, P90, P95, P99, Max).
- Xuất chi tiết top 30 frames có max-block MAD cao nhất và tạo ảnh đối chiếu trực quan (heatmaps & zoom-in).
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import cv2
from pathlib import Path

PLAN_PATH = "out/audit_3min_plan.json"
VIDEO_PATH = "samples/sample_3min.mkv"
OUT_DIR = "out/dedup_audit_16x16"
TOP_FRAMES_DIR = os.path.join(OUT_DIR, "top_30_worst")

def get_block_mad_8x8(diff):
    """Tính MAD cho toàn bộ lưới 8x8 pixel (32.400 blocks). 1080=135*8, 1920=240*8."""
    # diff: (1080, 1920) float32
    h, w = diff.shape
    blocks = diff.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3) # (135, 240, 8, 8)
    return blocks.mean(axis=(2, 3)) # (135, 240)

def get_block_mad_16x16(diff):
    """Tính MAD cho lưới 16x16 pixel (8.160 blocks)."""
    h, w = diff.shape
    main_h = (h // 16) * 16 # 1072
    blocks_main = diff[:main_h, :].reshape(main_h // 16, 16, w // 16, 16).transpose(0, 2, 1, 3).mean(axis=(2, 3))
    # Phần 8 hàng cuối: (8, 1920) -> chia thành 120 blocks 8x16
    rem_blocks = diff[main_h:, :].reshape(8, w // 16, 16).transpose(1, 0, 2).mean(axis=(1, 2))[np.newaxis, :]
    return np.vstack([blocks_main, rem_blocks]) # (68, 120)

def run_small_block_audit():
    os.makedirs(TOP_FRAMES_DIR, exist_ok=True)
    if not os.path.exists(PLAN_PATH):
        print(f"Error: {PLAN_PATH} not found!")
        sys.exit(1)
        
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)
        
    total_frames = plan["total_frames"]
    kept_count = plan["total_kept_frames"]
    dropped_count = total_frames - kept_count
    
    # Bản đồ ánh xạ: frame_idx -> reference kept frame_idx
    frame_map = {}
    for c in plan["chunks"]:
        base = c["start_frame"]
        for r in c["runs"]:
            f_idx = base + r["frame_idx"]
            for offset in range(r["count"]):
                frame_map[f_idx + offset] = f_idx

    print("=" * 70)
    print("🔬 BẮT ĐẦU AUDIT DEDUP VỚI CỠ KHỐI 16x16 & 8x8 PIXEL")
    print(f"• Video: {VIDEO_PATH}")
    print(f"• Tổng số frames: {total_frames} | Kept: {kept_count} | Dropped: {dropped_count}")
    print("=" * 70)

    # Đọc video rawvideo BGR24 để vừa tính luma/diff vừa có thể trích frame màu cho top 30
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", VIDEO_PATH,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=2 * 10**7)
    
    width, height = 1920, 1080
    frame_bytes = width * height * 3
    
    results = []
    current_kept_bgr = None
    current_kept_gray = None
    current_kept_idx = -1
    
    t0 = time.time()
    
    # Để lưu ảnh raw của top 30, trước hết duyệt qua và ghi nhận chỉ số metric
    # Lưu frame kept vào bộ đệm cache cục bộ khi cần
    kept_frames_cache = {}
    
    for idx in range(total_frames):
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            print(f"Cảnh báo: Video kết thúc sớm ở frame {idx}")
            break
            
        bgr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        # Chuẩn luma BT.709 cho anime HD: Y = 0.2126*R + 0.7152*G + 0.0722*B
        # OpenCV BGR: B=0, G=1, R=2
        gray = (0.2126 * bgr[:, :, 2] + 0.7152 * bgr[:, :, 1] + 0.0722 * bgr[:, :, 0]).astype(np.float32)
        
        ref_idx = frame_map[idx]
        if idx == ref_idx:
            current_kept_bgr = bgr.copy()
            current_kept_gray = gray.copy()
            current_kept_idx = idx
            kept_frames_cache[idx] = bgr.copy()
            continue
            
        # Đây là frame dropped -> so sánh với reference kept
        diff = np.abs(gray - current_kept_gray)
        global_mad = float(np.mean(diff))
        peak_diff = float(np.max(diff))
        
        # 1. Khối 8x8 (chuẩn mpdecimate)
        mad_8x8 = get_block_mad_8x8(diff)
        max_mad_8x8 = float(np.max(mad_8x8))
        pos_8x8 = np.unravel_index(np.argmax(mad_8x8), mad_8x8.shape)
        
        # 2. Khối 16x16
        mad_16x16 = get_block_mad_16x16(diff)
        max_mad_16x16 = float(np.max(mad_16x16))
        pos_16x16 = np.unravel_index(np.argmax(mad_16x16), mad_16x16.shape)
        
        results.append({
            "frame_idx": idx,
            "ref_idx": ref_idx,
            "gap": idx - ref_idx,
            "global_mad": global_mad,
            "peak_diff": peak_diff,
            "max_mad_16x16": max_mad_16x16,
            "block_16x16_pos": [int(pos_16x16[0]), int(pos_16x16[1])],
            "max_mad_8x8": max_mad_8x8,
            "block_8x8_pos": [int(pos_8x8[0]), int(pos_8x8[1])],
            # Lưu tạm bgr để trích xuất sau nếu lọt top
        })
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total_frames} frames ({(time.time()-t0):.1f}s)...")
            
    proc.stdout.close()
    proc.wait()
    
    elapsed = time.time() - t0
    print(f"\n✅ Đã xử lý xong {len(results)} dropped frames trong {elapsed:.2f}s ({len(results)/elapsed:.1f} fps)")
    
    # Sắp xếp để tìm top worst frames theo max_mad_16x16
    sorted_by_16x16 = sorted(results, key=lambda x: x["max_mad_16x16"], reverse=True)
    top_30 = sorted_by_16x16[:30]
    
    # Thống kê phân bố (distribution)
    mads_16 = np.array([r["max_mad_16x16"] for r in results])
    mads_8 = np.array([r["max_mad_8x8"] for r in results])
    global_mads = np.array([r["global_mad"] for r in results])
    
    stats = {
        "total_dropped_evaluated": len(results),
        "evaluation_time_seconds": round(elapsed, 2),
        "global_mad": {
            "mean": float(np.mean(global_mads)),
            "p50": float(np.percentile(global_mads, 50)),
            "p90": float(np.percentile(global_mads, 90)),
            "p95": float(np.percentile(global_mads, 95)),
            "p99": float(np.percentile(global_mads, 99)),
            "max": float(np.max(global_mads))
        },
        "max_block_mad_16x16": {
            "mean": float(np.mean(mads_16)),
            "p50": float(np.percentile(mads_16, 50)),
            "p90": float(np.percentile(mads_16, 90)),
            "p95": float(np.percentile(mads_16, 95)),
            "p99": float(np.percentile(mads_16, 99)),
            "max": float(np.max(mads_16))
        },
        "max_block_mad_8x8": {
            "mean": float(np.mean(mads_8)),
            "p50": float(np.percentile(mads_8, 50)),
            "p90": float(np.percentile(mads_8, 90)),
            "p95": float(np.percentile(mads_8, 95)),
            "p99": float(np.percentile(mads_8, 99)),
            "max": float(np.max(mads_8))
        },
        "histogram_16x16": {
            "< 2.0 (Grain/Noise)": int(np.sum(mads_16 < 2.0)),
            "2.0 - 4.0 (Subtle)": int(np.sum((mads_16 >= 2.0) & (mads_16 < 4.0))),
            "4.0 - 6.0 (Minor)": int(np.sum((mads_16 >= 4.0) & (mads_16 < 6.0))),
            "6.0 - 8.0 (Low motion)": int(np.sum((mads_16 >= 6.0) & (mads_16 < 8.0))),
            "8.0 - 12.0 (Threshold boundary)": int(np.sum((mads_16 >= 8.0) & (mads_16 < 12.0))),
            ">= 12.0 (Above mpdecimate hi=768/64)": int(np.sum(mads_16 >= 12.0))
        },
        "top_30_worst": top_30
    }
    
    summary_path = os.path.join(OUT_DIR, "small_block_audit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 70)
    print("📊 PHÂN BỐ VÀ PHẦN TRĂM PHÂN VỊ MAX-BLOCK MAD (16x16 PIXEL):")
    print("=" * 70)
    print(f"• Trung bình: {stats['max_block_mad_16x16']['mean']:.2f} / 255")
    print(f"• P50 (Trung vị): {stats['max_block_mad_16x16']['p50']:.2f} / 255")
    print(f"• P90: {stats['max_block_mad_16x16']['p90']:.2f} / 255")
    print(f"• P95: {stats['max_block_mad_16x16']['p95']:.2f} / 255")
    print(f"• P99: {stats['max_block_mad_16x16']['p99']:.2f} / 255")
    print(f"• Max (Worst-case toàn tập 3 phút): {stats['max_block_mad_16x16']['max']:.2f} / 255")
    print("\n[BẢNG PHÂN BỐ TẦN SUẤT 16x16]")
    for k, v in stats["histogram_16x16"].items():
        pct = (v / len(results)) * 100.0
        bar = "█" * int(pct // 2)
        print(f"  {k:38s}: {v:5d} ({pct:5.2f}%) {bar}")
        
    print(f"\n• Đối với khối 8x8 (chuẩn mpdecimate hi=768 => 12.0/255):")
    print(f"  P50: {stats['max_block_mad_8x8']['p50']:.2f} | P99: {stats['max_block_mad_8x8']['p99']:.2f} | Max: {stats['max_block_mad_8x8']['max']:.2f}")
    print(f"  Số frame có 8x8 MAD >= 12.0: {int(np.sum(mads_8 >= 12.0))} / {len(results)}")
    
    # Bây giờ trích xuất ảnh chi tiết cho TOP 30 worst frames
    print("\n" + "=" * 70)
    print("🖼️  TRÍCH XUẤT ẢNH ĐỐI CHIẾU CHO TOP 30 WORST FRAMES...")
    print("=" * 70)
    
    top_indices = set(r["frame_idx"] for r in top_30) | set(r["ref_idx"] for r in top_30)
    # Lần 2: giải mã chọn lọc các frame top để tạo ảnh composite
    extracted_frames = {}
    proc2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=2 * 10**7)
    for idx in range(total_frames):
        raw = proc2.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        if idx in top_indices:
            bgr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            extracted_frames[idx] = bgr.copy()
    proc2.stdout.close()
    proc2.wait()
    
    for rank, r in enumerate(top_30, 1):
        f_idx = r["frame_idx"]
        ref_idx = r["ref_idx"]
        mad16 = r["max_mad_16x16"]
        mad8 = r["max_mad_8x8"]
        
        f_bgr = extracted_frames.get(f_idx)
        ref_bgr = extracted_frames.get(ref_idx)
        if f_bgr is None or ref_bgr is None:
            continue
            
        # Tọa độ khối 16x16 tệ nhất
        by, bx = r["block_16x16_pos"]
        y1, y2 = by * 16, min((by + 1) * 16, height)
        x1, x2 = bx * 16, min((bx + 1) * 16, width)
        
        # Tạo diff heatmap
        diff_gray = np.abs(
            cv2.cvtColor(f_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) - 
            cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        )
        # Tăng cường độ tương phản nhiệt để dễ nhìn bằng mắt (x 5)
        heatmap_norm = np.clip(diff_gray * 5.0, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Vẽ bounding box màu đỏ quanh ô 16x16 tệ nhất trên cả 3 ảnh
        box_color = (0, 0, 255)
        # Mở rộng một chút cho dễ thấy: 32x32
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        crop_y1, crop_y2 = max(0, cy - 64), min(height, cy + 64)
        crop_x1, crop_x2 = max(0, cx - 64), min(width, cx + 64)
        
        cv2.rectangle(heatmap_color, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Tạo crop zoom-in (128x128 phóng to lên 256x256)
        zoom_orig = cv2.resize(f_bgr[crop_y1:crop_y2, crop_x1:crop_x2], (256, 256), interpolation=cv2.INTER_NEAREST)
        zoom_ref = cv2.resize(ref_bgr[crop_y1:crop_y2, crop_x1:crop_x2], (256, 256), interpolation=cv2.INTER_NEAREST)
        zoom_heat = cv2.resize(heatmap_color[crop_y1:crop_y2, crop_x1:crop_x2], (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Ghép dải zoom
        zoom_strip = np.hstack([zoom_orig, zoom_ref, zoom_heat])
        cv2.putText(zoom_strip, f"Dropped Frame #{f_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(zoom_strip, f"Ref Frame #{ref_idx}", (266, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(zoom_strip, f"Zoom 16x16 MAD: {mad16:.2f}", (522, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Ghép ảnh tổng quát thu nhỏ
        preview_h = 360
        preview_w = 640
        f_small = cv2.resize(f_bgr, (preview_w, preview_h))
        heat_small = cv2.resize(heatmap_color, (preview_w, preview_h))
        top_strip = np.hstack([f_small, heat_small])
        
        # Ghép ảnh hoàn chỉnh
        # Header text
        header = np.zeros((50, top_strip.shape[1], 3), dtype=np.uint8)
        title = f"Rank {rank:02d} | Dropped #{f_idx} vs Ref #{ref_idx} (Gap: {f_idx-ref_idx}) | Max 16x16 MAD: {mad16:.2f} | 8x8 MAD: {mad8:.2f}"
        cv2.putText(header, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Căn chỉnh zoom_strip giữa
        zoom_padded = np.zeros((zoom_strip.shape[0], top_strip.shape[1], 3), dtype=np.uint8)
        start_x = (top_strip.shape[1] - zoom_strip.shape[1]) // 2
        zoom_padded[:, start_x:start_x + zoom_strip.shape[1]] = zoom_strip
        
        composite = np.vstack([header, top_strip, zoom_padded])
        
        out_img_name = f"rank_{rank:02d}_frame_{f_idx}_vs_{ref_idx}_mad16_{mad16:.2f}.png"
        cv2.imwrite(os.path.join(TOP_FRAMES_DIR, out_img_name), composite)
        print(f"  [Rank {rank:02d}] Frame #{f_idx} (Ref #{ref_idx}): 16x16 MAD = {mad16:.2f}, 8x8 MAD = {mad8:.2f} -> {out_img_name}")

    print(f"\n✅ Đã trích xuất và vẽ thành công 30 ảnh chi tiết vào: {TOP_FRAMES_DIR}")
    return stats

if __name__ == "__main__":
    run_small_block_audit()
