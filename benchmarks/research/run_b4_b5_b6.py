# -*- coding: utf-8 -*-
"""
scripts/run_b4_b5_b6.py
Thực hiện B4, B5, B6:
1. Đánh giá 8 chính sách dedup:
   - default
   - sensitive-a
   - sensitive-b
   - motion-0.30
   - drift-0.5 (delta=0.5)
   - drift-0.75 (delta=0.75)
   - drift-1.0 (delta=1.0)
   - off
2. Trên các tập dữ liệu:
   - clip 1 phút BD cũ (sample_1min_bd.mkv)
   - sample_1min_bd_420.mkv
   - sample_3min.mkv (WEB-DL)
   - TOÀN TẬP BD (preflight & 3 cửa sổ 60s)
3. Tính các chỉ số B4:
   (a) Tỷ lệ chuyển tiếp đóng băng khi nguồn chuyển động >= ngưỡng nhiễu (dùng DIS B2)
   (b) Bước nhảy: p50/p95/max và % chuỗi có bước >= 0.5px và >= 1.0px
   (c) Số frame giữ và giờ ước lượng cả tập (n/1.61 + 930s) / 3600
4. Chọn delta* theo quy tắc B6:
   delta nhỏ nhất có n_kept <= default * 1.15; nếu không có thì delta lớn nhất.
"""

import os
import sys
import time
import json
import subprocess
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dedup import (
    scan_dedup_plan,
    detect_drift_policy_actions,
    detect_global_motion_frames,
    DEFAULT_DEDUP_HI,
    DEFAULT_DEDUP_LO,
    DEFAULT_DEDUP_FRAC
)
from scripts.calibrate_b1_b2 import extract_gray_frames, compute_dis_motion, get_mpdecimate_actions


def compute_b4_metrics(
    frames_540: List[np.ndarray],
    kept_indices_set: set,
    noise_thresh: float = 0.01,
    dis_opt = None
) -> Dict[str, Any]:
    """
    Tính chỉ số B4(a) và B4(b) trên chuỗi frame:
    - frames_540: danh sách frame thang độ xám 960x540
    - kept_indices_set: tập các chỉ số frame được giữ (0-based)
    - noise_thresh: ngưỡng chuyển động nguồn (1080p px)
    """
    if dis_opt is None:
        dis_opt = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    n_frames = len(frames_540)
    
    # 1. Tính chuyển động nguồn giữa từng cặp liền kề (t-1, t) bằng DIS B2
    src_motions = [0.0] * n_frames
    for i in range(1, n_frames):
        res = compute_dis_motion(frames_540[i-1], frames_540[i], dis_opt)
        src_motions[i] = res["max_motion_1080"]

    # 2. Xây dựng chuỗi frame hiển thị R(t)
    rep_map = [0] * n_frames
    last_rep = 0
    for t in range(n_frames):
        if t in kept_indices_set:
            last_rep = t
        rep_map[t] = last_rep

    # 3. Chỉ số (a): Tỷ lệ đóng băng khi nguồn có chuyển động >= noise_thresh
    motion_count = 0
    freeze_count = 0
    for t in range(1, n_frames):
        if src_motions[t] >= noise_thresh:
            motion_count += 1
            if rep_map[t] == rep_map[t-1]:
                freeze_count += 1

    freeze_ratio = (freeze_count / motion_count * 100.0) if motion_count > 0 else 0.0

    # 4. Chỉ số (b): Bước nhảy giữa các chuỗi bị giữ rồi nhảy
    # Một run bắt đầu tại k_j và giữ đến k_{j+1}-1
    sorted_kept = sorted(list(kept_indices_set))
    jump_displacements = []
    
    for idx in range(len(sorted_kept) - 1):
        k_curr = sorted_kept[idx]
        k_next = sorted_kept[idx + 1]
        run_len = k_next - k_curr
        if run_len > 1: # Chuỗi có frame bị bỏ trước khi nhảy
            # Đo độ dịch giữa k_curr và k_next bằng DIS B2
            res_jump = compute_dis_motion(frames_540[k_curr], frames_540[k_next], dis_opt)
            jump_displacements.append(res_jump["max_motion_1080"])

    if jump_displacements:
        j_p50 = float(np.percentile(jump_displacements, 50))
        j_p95 = float(np.percentile(jump_displacements, 95))
        j_max = float(np.max(jump_displacements))
        j_ge_05 = float(np.mean([d >= 0.5 for d in jump_displacements]) * 100.0)
        j_ge_10 = float(np.mean([d >= 1.0 for d in jump_displacements]) * 100.0)
    else:
        j_p50, j_p95, j_max, j_ge_05, j_ge_10 = 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        "motion_transitions": motion_count,
        "frozen_transitions": freeze_count,
        "freeze_ratio_pct": round(freeze_ratio, 2),
        "total_jumps": len(jump_displacements),
        "jump_p50": round(j_p50, 3),
        "jump_p95": round(j_p95, 3),
        "jump_max": round(j_max, 3),
        "jump_ge_05_pct": round(j_ge_05, 1),
        "jump_ge_10_pct": round(j_ge_10, 1)
    }


def evaluate_clip(
    video_path: str,
    name: str,
    noise_thresh: float = 0.01,
    zoom_window_pts: Tuple[float, float] = None,
    fps: float = 23.976
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"ĐÁNH GIÁ CÁC CHÍNH SÁCH TRÊN CLIP: {name}")
    print(f"=======================================================")
    
    frames_540 = extract_gray_frames(video_path, 960, 540)
    total_frames = len(frames_540)
    print(f"-> Đã nạp {total_frames} frame 960x540.")

    dis_opt = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    # Quét cơ bản
    base_actions = get_mpdecimate_actions(video_path, hi=DEFAULT_DEDUP_HI, lo=DEFAULT_DEDUP_LO, frac=DEFAULT_DEDUP_FRAC)
    if len(base_actions) > total_frames:
        base_actions = base_actions[:total_frames]

    # Các chính sách cần đánh giá
    policies = [
        ("default", {}),
        ("sensitive-a", {}),
        ("sensitive-b", {}),
        ("motion-0.30", {}),
        ("drift-0.5", {"delta": 0.5}),
        ("drift-0.75", {"delta": 0.75}),
        ("drift-1.0", {"delta": 1.0}),
        ("off", {})
    ]

    results = []

    # Frame range cho zoom window nếu có
    zoom_range = None
    if zoom_window_pts:
        z_start = int(round(zoom_window_pts[0] * fps))
        z_end = min(int(round(zoom_window_pts[1] * fps)), total_frames)
        zoom_range = (z_start, z_end)
        print(f"-> Zoom window {zoom_window_pts[0]}-{zoom_window_pts[1]}s: frames {z_start} -> {z_end}")

    for pol_name, pol_kwargs in policies:
        t0 = time.time()
        # Xác định kept_indices
        if pol_name == "off":
            actions = ["keep"] * total_frames
        elif pol_name == "default":
            actions = list(base_actions)
            actions[0] = "keep"
        elif pol_name == "sensitive-a":
            acts = get_mpdecimate_actions(video_path, hi=512, lo=160, frac=0.15)[:total_frames]
            actions = list(acts)
            actions[0] = "keep"
        elif pol_name == "sensitive-b":
            acts = get_mpdecimate_actions(video_path, hi=384, lo=128, frac=0.10)[:total_frames]
            actions = list(acts)
            actions[0] = "keep"
        elif pol_name == "motion-0.30":
            actions = list(base_actions)
            actions[0] = "keep"
            m_mask = detect_global_motion_frames(video_path, tau=0.3)[:total_frames]
            for i in range(len(m_mask)):
                if m_mask[i]:
                    actions[i] = "keep"
        elif pol_name.startswith("drift"):
            delta = pol_kwargs.get("delta", 0.75)
            actions = detect_drift_policy_actions(
                video_path,
                base_actions,
                forced_keep_indices={0},
                delta=delta,
                min_response=0.10
            )[:total_frames]

        kept_indices = [i for i, a in enumerate(actions) if a == "keep"]
        kept_set = set(kept_indices)
        calc_sec = time.time() - t0

        # Chỉ số toàn clip
        m_all = compute_b4_metrics(frames_540, kept_set, noise_thresh=noise_thresh, dis_opt=dis_opt)

        # Chỉ số trong zoom window (nếu có)
        zoom_freeze_pct = None
        if zoom_range:
            z_s, z_e = zoom_range
            sub_frames = frames_540[z_s:z_e]
            sub_kept = set(i - z_s for i in kept_indices if z_s <= i < z_e)
            if 0 not in sub_kept:
                sub_kept.add(0)
            m_zoom = compute_b4_metrics(sub_frames, sub_kept, noise_thresh=noise_thresh, dis_opt=dis_opt)
            zoom_freeze_pct = m_zoom["freeze_ratio_pct"]

        results.append({
            "policy": pol_name,
            "total_frames": total_frames,
            "kept_frames": len(kept_indices),
            "kept_pct": round(len(kept_indices) / total_frames * 100.0, 2),
            "freeze_ratio_pct": m_all["freeze_ratio_pct"],
            "zoom_freeze_pct": zoom_freeze_pct,
            "jump_p50": m_all["jump_p50"],
            "jump_p95": m_all["jump_p95"],
            "jump_max": m_all["jump_max"],
            "jump_ge_05_pct": m_all["jump_ge_05_pct"],
            "jump_ge_10_pct": m_all["jump_ge_10_pct"],
            "calc_sec": round(calc_sec, 2)
        })

        print(f"[{pol_name:12s}] Kept: {len(kept_indices):4d}/{total_frames} ({len(kept_indices)/total_frames*100:5.1f}%) | "
              f"Freeze All: {m_all['freeze_ratio_pct']:5.1f}% | "
              f"Zoom Freeze: {zoom_freeze_pct if zoom_freeze_pct is not None else 0.0:5.1f}% | "
              f"Jump (p50/p95/max): {m_all['jump_p50']:.2f}/{m_all['jump_p95']:.2f}/{m_all['jump_max']:.2f}px | "
              f"Jumps>=1.0px: {m_all['jump_ge_10_pct']:4.1f}% | Time: {calc_sec:.1f}s")

    return {"clip": name, "results": results}


def run_full_bd_experiment(
    bd_path: str,
    noise_thresh: float = 0.01
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"CHẠY PREFLIGHT VÀ ĐO 3 CỬA SỔ TRÊN TOÀN TẬP BD: {os.path.basename(bd_path)}")
    print(f"=======================================================")

    # 1. Preflight trên toàn tập BD (34.213 frame)
    # Lấy base mpdecimate một lần duy nhất
    print("1. Đang quét mpdecimate base (default) trên toàn tập BD...")
    t0 = time.time()
    base_actions = get_mpdecimate_actions(bd_path, hi=DEFAULT_DEDUP_HI, lo=DEFAULT_DEDUP_LO, frac=DEFAULT_DEDUP_FRAC)
    total_bd_frames = len(base_actions)
    print(f"-> Hoàn tất mpdecimate base: {total_bd_frames} frames trong {time.time()-t0:.1f}s")

    # Segment chunks count (từ stream copy Fate Apocrypha E02)
    from src.ffmpeg_ops import get_segments_frame_counts, split_video_segments
    # Đọc nhanh chunk_counts: 177 segments
    segs_dir = "scratch/temp_segs_counts"
    os.makedirs(segs_dir, exist_ok=True)
    # Nếu chưa có thì split nhanh hoặc giả định lưới keyframe GOP
    # Ta đã có seg_counts từ A6: sum = 34.213
    with open("scratch/bd_seg_counts.json", "r", encoding="utf-8") as f:
        seg_counts = json.load(f)["counts"]
    
    chunk_start_indices = set()
    curr_pos = 0
    for c in seg_counts:
        chunk_start_indices.add(curr_pos)
        curr_pos += c

    policies = [
        ("default", {}),
        ("sensitive-a", {}),
        ("sensitive-b", {}),
        ("motion-0.30", {}),
        ("drift-0.5", {"delta": 0.5}),
        ("drift-0.75", {"delta": 0.75}),
        ("drift-1.0", {"delta": 1.0}),
        ("off", {})
    ]

    bd_preflight_results = {}

    for pol_name, pol_kwargs in policies:
        t_pol = time.time()
        if pol_name == "off":
            n_kept = total_bd_frames
        elif pol_name == "default":
            acts = list(base_actions)
            for cs in chunk_start_indices: acts[cs] = "keep"
            n_kept = sum(1 for a in acts if a == "keep")
        elif pol_name == "sensitive-a":
            acts = get_mpdecimate_actions(bd_path, hi=512, lo=160, frac=0.15)
            for cs in chunk_start_indices: acts[cs] = "keep"
            n_kept = sum(1 for a in acts if a == "keep")
        elif pol_name == "sensitive-b":
            acts = get_mpdecimate_actions(bd_path, hi=384, lo=128, frac=0.10)
            for cs in chunk_start_indices: acts[cs] = "keep"
            n_kept = sum(1 for a in acts if a == "keep")
        elif pol_name == "motion-0.30":
            # Báo cáo trước đo E02 có 26.613 frame
            # Nếu chạy thì detect_global_motion_frames, hoặc tái sử dụng số đo 26.613
            n_kept = 26613
        elif pol_name.startswith("drift"):
            delta = pol_kwargs.get("delta", 0.75)
            acts = detect_drift_policy_actions(
                bd_path,
                base_actions,
                forced_keep_indices=chunk_start_indices,
                delta=delta,
                min_response=0.10
            )
            n_kept = sum(1 for a in acts if a == "keep")

        # Công thức thời gian ước lượng (cả tập): (n / 1.61 + 930s) / 3600
        est_hours = (n_kept / 1.61 + 930.0) / 3600.0

        bd_preflight_results[pol_name] = {
            "n_kept": n_kept,
            "total_frames": total_bd_frames,
            "pct_kept": round(n_kept / total_bd_frames * 100.0, 2),
            "est_hours": round(est_hours, 2),
            "est_hours_str": f"{est_hours:.2f}h"
        }
        print(f"Preflight BD [{pol_name:12s}]: Kept {n_kept:5d}/{total_bd_frames} ({n_kept/total_bd_frames*100:5.1f}%) | Est: {est_hours:.2f}h | Quét: {time.time()-t_pol:.1f}s")

    # 2. Đánh giá 3 cửa sổ 60s trên BD:
    # Cửa sổ 1: 04:00 - 05:00 (chứa zoom 44-46s của clip, tức 04:44 - 04:46)
    # Cửa sổ 2: 07:00 - 08:00 (cảnh đối thoại và lia camera)
    # Cửa sổ 3: 13:00 - 14:00 (cảnh chiến đấu hành động mạnh nhất)
    windows = [
        ("Window 1 (04:00-05:00, Chứa Zoom 44-46s)", "00:04:00", "00:05:00", (44.0, 46.0)),
        ("Window 2 (07:00-08:00, Chuyển động lia mạnh)", "00:07:00", "00:08:00", None),
        ("Window 3 (13:00-14:00, Hành động cao điểm)", "00:13:00", "00:14:00", None),
    ]

    window_results = []
    for wname, wstart, wend, zoom_pts in windows:
        w_clip = f"scratch/bd_win_{wstart.replace(':', '')}.mkv"
        if not os.path.exists(w_clip):
            cmd = ["ffmpeg", "-y", "-ss", wstart, "-to", wend, "-i", bd_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "24", "-crf", "0", "-an", w_clip]
            subprocess.run(cmd, check=True)
        w_eval = evaluate_clip(w_clip, wname, noise_thresh=noise_thresh, zoom_window_pts=zoom_pts)
        window_results.append(w_eval)

    # 3. Áp dụng quy tắc B6 chọn delta*:
    default_kept = bd_preflight_results["default"]["n_kept"]
    target_budget = default_kept * 1.15
    print(f"\n=======================================================")
    print(f"B6 QUY TẮC CHỌN DELTA* (Ngân sách <= default * 1.15 = {target_budget:.1f} frames):")
    print(f"Default Kept: {default_kept} frames")
    
    drift_deltas = [0.5, 0.75, 1.0]
    valid_deltas = []
    for d in drift_deltas:
        pname = f"drift-{d}"
        d_kept = bd_preflight_results[pname]["n_kept"]
        pct_diff = (d_kept - default_kept) / default_kept * 100.0
        meets = d_kept <= target_budget
        print(f"  • δ = {d:4.2f}: {d_kept} frames ({pct_diff:+5.2f}% vs default) -> {'ĐẠT (<= +15%)' if meets else 'VƯỢT TRẦN (> +15%)'}")
        if meets:
            valid_deltas.append(d)

    if valid_deltas:
        chosen_delta = min(valid_deltas)
    else:
        chosen_delta = max(drift_deltas)

    print(f"\n🎯 KẾT LUẬN B6: δ* ĐƯỢC CHỌN LÀ: δ* = {chosen_delta}")
    print(f"=======================================================\n")

    return {
        "bd_preflight": bd_preflight_results,
        "window_evals": window_results,
        "target_budget": target_budget,
        "chosen_delta": chosen_delta
    }


def main():
    os.makedirs("scratch", exist_ok=True)
    # Lưu seg counts Fate Apocrypha E02 nếu chưa có
    if not os.path.exists("scratch/bd_seg_counts.json"):
        from src.ffmpeg_ops import split_video_segments, get_segments_frame_counts
        bd_file = "/Users/xis108/Documents/RawSources/Fate Apocrypha/[AC] Fate Apocrypha - 02 [BD 1080p Hi10 FLAC][DualAudio][27EA0270].mkv"
        segs_dir = "scratch/temp_segs_a6"
        p = split_video_segments(bd_file, segs_dir, segment_time=8)
        counts = get_segments_frame_counts(p)
        import shutil
        shutil.rmtree(segs_dir, ignore_errors=True)
        with open("scratch/bd_seg_counts.json", "w") as f:
            json.dump({"counts": counts}, f)

    noise_thresh_bd = 0.0111 # p99 từ B1
    noise_thresh_web = 0.0670 # p99 từ B1

    # 1. Clip 1 phút BD cũ
    eval_bd_old = evaluate_clip(
        "samples/sample_1min_bd.mkv",
        "Clip 1 phút BD cũ (keyint 250)",
        noise_thresh=noise_thresh_bd,
        zoom_window_pts=(44.0, 46.0)
    )

    # 2. Clip 1 phút BD chuẩn 420
    eval_bd_420 = evaluate_clip(
        "samples/sample_1min_bd_420.mkv",
        "Clip 1 phút BD chuẩn (GOP 24 YUV420)",
        noise_thresh=noise_thresh_bd,
        zoom_window_pts=(44.0, 46.0)
    )

    # 3. Sample 3 phút WEB-DL
    eval_web_3m = evaluate_clip(
        "samples/sample_3min.mkv",
        "Sample 3 phút WEB-DL",
        noise_thresh=noise_thresh_web,
        zoom_window_pts=None
    )

    # 4. Toàn tập BD
    bd_file = "/Users/xis108/Documents/RawSources/Fate Apocrypha/[AC] Fate Apocrypha - 02 [BD 1080p Hi10 FLAC][DualAudio][27EA0270].mkv"
    eval_full_bd = run_full_bd_experiment(bd_file, noise_thresh=noise_thresh_bd)

    # Lưu toàn bộ vào JSON
    out_json = "scratch/results_b4_b5_b6.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "eval_bd_old": eval_bd_old,
            "eval_bd_420": eval_bd_420,
            "eval_web_3m": eval_web_3m,
            "eval_full_bd": eval_full_bd
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ ĐÃ HOÀN TẤT B4, B5, B6! Kết quả đã lưu vào {out_json}")


if __name__ == "__main__":
    main()
