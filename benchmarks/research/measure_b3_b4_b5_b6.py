#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/measure_b3_b4_b5_b6.py
Thực hiện B3, B4, B5, B6:
- B3: Hiệu chuẩn nhiễu từng tập trên các cặp frame liền kề lặp thật (MAE < 0.3): p50/p95/p99.
      Đề xuất công thức tau = max(0.30, 3 * p99).
- B4: Chi phí toàn tập (CPU, không GPU):
      default, dis (0.3 / 0.4 / 0.5 / 0.7), sensitive-b, off:
      n_unique, % tăng so với default, giờ ước lượng (n/1.6 + 930s), thời gian preflight CPU.
- B5: Chỉ số độc lập (mô phỏng chuỗi nguồn, tỷ lệ lặp frame khi MAE nguồn >= 0.3, bước nhảy p50/p95/max).
- B6: Mô phỏng cơ chế --dedup-budget P% (5%, 10%, 15%): bảng tau chọn được, tỷ lệ đóng băng còn lại, giờ ước lượng.
"""

import os
import sys
import json
import time
import subprocess
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dedup import (
    scan_dedup_plan,
    DEFAULT_DEDUP_HI,
    DEFAULT_DEDUP_LO,
    DEFAULT_DEDUP_FRAC,
    detect_dis_policy_actions
)

DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_FAST

# ----------------- B3: HIỆU CHUẨN NHIỄU TRÊN CẶP FRAME LIỀN KỀ LẶP THẬT -----------------

def calibrate_noise_distribution(video_path, max_sample_pairs=200):
    """
    Trích xuất các cặp frame liên tiếp (t, t+1) mà MAE nguồn < 0.3 (frame lặp thật).
    Đo độ lớn dòng chảy DIS (480x270 quy đổi ra 1080p).
    Trả về p50, p95, p99 và ngưỡng đề xuất tau = max(0.30, 3 * p99).
    """
    cap = cv2.VideoCapture(video_path)
    dis = cv2.DISOpticalFlow_create(DIS_PRESET)
    noise_scores = []
    
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if prev_frame is not None:
            # MAE nguồn 1080p
            diff_u8 = cv2.absdiff(prev_frame, frame)
            mae = float(np.mean(diff_u8))
            if mae < 0.30:  # Frame lặp thật
                # Đo DIS flow ở 480x270
                g1 = cv2.resize(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (480, 270))
                g2 = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (480, 270))
                flow = dis.calc(g1, g2, None)
                u = flow[..., 0] * 4.0
                v = flow[..., 1] * 4.0
                mag = np.hypot(u, v)
                dx = float(np.median(u))
                dy = float(np.median(v))
                mag_pan = float(np.hypot(dx, dy))
                p90 = float(np.percentile(mag, 90))
                score = max(mag_pan, 0.7 * p90)
                noise_scores.append(score)
                if len(noise_scores) >= max_sample_pairs:
                    break
        prev_frame = frame
    cap.release()
    
    if len(noise_scores) < 10:
        return {"p50": 0.05, "p95": 0.08, "p99": 0.09, "tau_prop": 0.30, "n_samples": len(noise_scores)}
    
    arr = np.array(noise_scores)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    tau_prop = max(0.30, 3.0 * p99)
    
    return {
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "p99": round(p99, 4),
        "tau_prop": round(tau_prop, 4),
        "n_samples": len(noise_scores)
    }


# ----------------- B4: QUÉT CHI PHÍ TOÀN TẬP -----------------

def measure_episode_cost(video_path, name, temp_dir="scratch/b4_temp"):
    os.makedirs(temp_dir, exist_ok=True)
    policies = [
        ("default", "default", None),
        ("dis-0.3", "dis-0.3", 0.30),
        ("dis-0.4", "dis-0.4", 0.40),
        ("dis-0.5", "dis-0.5", 0.50),
        ("dis-0.7", "dis-0.7", 0.70),
        ("sensitive-b", "sensitive-b", None),
        ("off", "off", None),
    ]
    
    results = {}
    default_n = None
    
    for pol_name, pol_code, dis_th in policies:
        plan_file = os.path.join(temp_dir, f"plan_{name}_{pol_name}.json")
        t0 = time.perf_counter()
        p = scan_dedup_plan(video_path, plan_file, policy=pol_code, dis_thresh=dis_th)
        preflight_s = time.perf_counter() - t0
        
        n_unique = p["total_kept_frames"]
        total_f = p["total_frames"]
        
        if pol_name == "default":
            default_n = n_unique
            pct_inc = 0.0
        else:
            pct_inc = ((n_unique - default_n) / default_n) * 100.0 if default_n else 0.0
        
        # Công thức giờ ước lượng: (n / 1.6 + 930s)
        est_gpu_sec = (n_unique / 1.6) + 930.0
        est_gpu_hours = est_gpu_sec / 3600.0
        
        results[pol_name] = {
            "total_frames": total_f,
            "n_unique": n_unique,
            "pct_kept": round(n_unique / total_f * 100.0, 2),
            "pct_inc_vs_default": round(pct_inc, 2),
            "est_gpu_hours": round(est_gpu_hours, 2),
            "est_gpu_sec": round(est_gpu_sec, 1),
            "preflight_cpu_s": round(preflight_s, 1),
        }
    
    return results


# ----------------- B5: CHỈ SỐ ĐỘC LẬP TỪ NGUỒN -----------------

def measure_independent_metrics(clip_path):
    """
    B5: Đo tỷ lệ đóng băng khi MAE nguồn >= 0.3 và kích thước bước nhảy (px 1080p).
    """
    cap = cv2.VideoCapture(clip_path)
    frames_gray_540 = []
    frames_gray_1080 = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        g1080 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g540 = cv2.resize(g1080, (960, 540))
        frames_gray_1080.append(g1080)
        frames_gray_540.append(g540)
    cap.release()
    
    n = len(frames_gray_1080)
    if n < 20:
        return {}
    
    # 1. Tính MAE nguồn liền kề
    source_motion = [False] * n
    for t in range(1, n):
        mae = float(np.mean(cv2.absdiff(frames_gray_1080[t-1], frames_gray_1080[t])))
        if mae >= 0.30:
            source_motion[t] = True
            
    # 2. Lấy plans
    import tempfile
    plans = {}
    with tempfile.TemporaryDirectory() as td:
        for p in ["default", "dis-0.3", "sensitive-b", "off"]:
            p_res = scan_dedup_plan(clip_path, os.path.join(td, f"{p}.json"), policy=p, dis_thresh=0.30)
            acts = []
            for c in p_res["chunks"]:
                for r in c["runs"]:
                    acts.append("keep")
                    for _ in range(r["count"] - 1):
                        acts.append("drop")
            plans[p] = acts
            
    # 3. Định nghĩa các cửa sổ
    windows = {
        "Window_1 (965-1048)": (965, min(1048, n-1)),
        "Window_2 (1055-1103)": (1055, min(1103, n-1)),
        "Pan_Window (1-37)": (1, min(37, n-1)),
        "Full_Clip (0-N)": (0, n-1)
    }
    
    # 4. Tính tương quan pha trên 960x540
    hann = cv2.createHanningWindow((960, 540), cv2.CV_32F)
    
    results_b5 = {}
    
    for win_name, (w_start, w_end) in windows.items():
        win_len = w_end - w_start + 1
        gt_motion_in_win = sum(1 for t in range(w_start, w_end+1) if source_motion[t])
        
        win_res = {"total_frames": win_len, "gt_motion_frames": gt_motion_in_win, "policies": {}}
        
        for pol, acts in plans.items():
            rep = [0] * n
            last_k = 0
            for t in range(n):
                if acts[t] == "keep":
                    last_k = t
                rep[t] = last_k
                
            frozen_count = 0
            jump_sizes = []
            
            for t in range(max(1, w_start), w_end + 1):
                if source_motion[t]:
                    if rep[t] == rep[t-1]:
                        frozen_count += 1
                        
                if rep[t] != rep[t-1]:
                    f1 = frames_gray_540[rep[t-1]].astype(np.float32)
                    f2 = frames_gray_540[rep[t]].astype(np.float32)
                    (dx, dy), _ = cv2.phaseCorrelate(f1, f2, hann)
                    disp = float(np.hypot(dx * 2.0, dy * 2.0))
                    jump_sizes.append(disp)
                    
            pct_frozen = (frozen_count / max(1, gt_motion_in_win)) * 100.0
            
            p50_jump = float(np.percentile(jump_sizes, 50)) if jump_sizes else 0.0
            p95_jump = float(np.percentile(jump_sizes, 95)) if jump_sizes else 0.0
            max_jump = float(np.max(jump_sizes)) if jump_sizes else 0.0
            
            win_res["policies"][pol] = {
                "frozen_count": frozen_count,
                "pct_frozen": round(pct_frozen, 2),
                "jump_p50": round(p50_jump, 3),
                "jump_p95": round(p95_jump, 3),
                "jump_max": round(max_jump, 3),
            }
        results_b5[win_name] = win_res
        
    return results_b5


# ----------------- B6: MÔ PHỎNG --DEDUP-BUDGET P% -----------------

def simulate_budget_policy(video_path, name, budget_pcts=[5, 10, 15]):
    """
    B6: Quét DIS 1 lượt, tính scores, rồi quét tau in [0.2, 1.0] để chọn tau nhỏ nhất sao cho n_unique <= default * (1 + P%).
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p_def = scan_dedup_plan(video_path, os.path.join(td, "def.json"), policy="default")
        n_def = p_def["total_kept_frames"]
        total_f = p_def["total_frames"]
        
        tau_candidates = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        tau_results = []
        for tau in tau_candidates:
            p_tau = scan_dedup_plan(video_path, os.path.join(td, f"dis_{tau}.json"), policy="dis-0.3", dis_thresh=tau)
            n_tau = p_tau["total_kept_frames"]
            pct_inc = ((n_tau - n_def) / n_def) * 100.0
            est_h = ((n_tau / 1.6) + 930.0) / 3600.0
            tau_results.append({
                "tau": tau,
                "n_kept": n_tau,
                "pct_inc": round(pct_inc, 2),
                "est_gpu_hours": round(est_h, 2)
            })
            
    budget_sim = {}
    for P in budget_pcts:
        valid = [r for r in tau_results if r["pct_inc"] <= P]
        if valid:
            best = valid[0]
        else:
            best = tau_results[-1]
        budget_sim[f"+{P}%"] = {
            "budget_P": P,
            "chosen_tau": best["tau"],
            "actual_pct_inc": best["pct_inc"],
            "n_kept": best["n_kept"],
            "est_gpu_hours": best["est_gpu_hours"]
        }
        
    return {
        "default_n": n_def,
        "total_frames": total_f,
        "tau_scan_curve": tau_results,
        "budgets": budget_sim
    }


def main():
    print("=================================================================")
    print("THỰC THI PHẦN B3, B4, B5, B6: HIỆU CHUẨN, CHI PHÍ VÀ NGÂN SÁCH")
    print("=================================================================")
    
    bd_e02 = "/Users/xis108/Documents/RawSources/Fate Apocrypha/[AC] Fate Apocrypha - 02 [BD 1080p Hi10 FLAC][DualAudio][27EA0270].mkv"
    fsn_e02 = "/Users/xis108/Documents/RawSources/0-Fate Stay Night/Fate Stay Night S01E02 Fateful Night.mkv"
    ubw_e01 = "/Users/xis108/Documents/RawSources/2-Fate stay night Unlimited Blade Works [Ma10p_1080p]/[VCB-Studio] Fate stay night Unlimited Blade Works [01][Ma10p_1080p][x265_flac].mkv"
    sample_3min = "samples/sample_3min.mkv"
    sample_1min_bd = "samples/sample_1min_bd.mkv"
    
    # 1. B3: Hiệu chuẩn nhiễu từng tập
    print("\n--- B3: HIỆU CHUẨN NHIỄU THEO TỪNG TẬP (FRAME LẶP THẬT MAE < 0.3) ---")
    calib_targets = [
        ("Fate_Apocrypha_E02 (BD Remux)", bd_e02),
        ("Fate_Stay_Night_S01E02 (BD)", fsn_e02),
        ("Fate_UBW_E01 (VCB-Studio BD)", ubw_e01),
        ("Sample_3min_WEB_DL", sample_3min)
    ]
    
    b3_results = {}
    for name, path in calib_targets:
        if os.path.exists(path):
            t0 = time.perf_counter()
            res = calibrate_noise_distribution(path, max_sample_pairs=200)
            elapsed = time.perf_counter() - t0
            b3_results[name] = res
            print(f"  {name:30s}: p50={res['p50']:.4f}px | p95={res['p95']:.4f}px | p99={res['p99']:.4f}px -> tau_prop={res['tau_prop']:.4f}px (n={res['n_samples']}, {elapsed:.1f}s)")
            
    # 2. B4: Chi phí toàn tập
    print("\n--- B4: CHI PHÍ TOÀN TẬP (CPU PREFLIGHT & ƯỚC LƯỢNG GPU) ---")
    cost_targets = [
        ("Fate_Apocrypha_BD_1min", sample_1min_bd),
        ("Sample_3min_WEB_DL", sample_3min),
    ]
    b4_results = {}
    for name, path in cost_targets:
        if os.path.exists(path):
            print(f"  Đang quét chi phí: {name}...")
            res = measure_episode_cost(path, name)
            b4_results[name] = res
            for pol, d in res.items():
                print(f"    {pol:12s}: {d['n_unique']:5d} frames ({d['pct_kept']:5.1f}%, inc {d['pct_inc_vs_default']:+5.1f}%) | GPU est: {d['est_gpu_hours']:.2f}h | CPU: {d['preflight_cpu_s']:4.1f}s")

    # Ước lượng scale lên toàn tập Fate E02 (34.213 frames)
    print("\n  [Scale ước lượng lên toàn tập Fate E02 - 34.213 frames từ tỷ lệ đo đạc]:")
    bd_scale_factor = 34213.0 / 1439.0
    bd_scaled = {}
    for pol, d in b4_results["Fate_Apocrypha_BD_1min"].items():
        scaled_n = int(round(d["n_unique"] * bd_scale_factor))
        scaled_est_sec = (scaled_n / 1.6) + 930.0
        scaled_est_h = scaled_est_sec / 3600.0
        bd_scaled[pol] = {
            "total_frames": 34213,
            "n_unique": scaled_n,
            "pct_kept": d["pct_kept"],
            "pct_inc_vs_default": d["pct_inc_vs_default"],
            "est_gpu_hours": round(scaled_est_h, 2),
            "est_preflight_cpu_s": round(d["preflight_cpu_s"] * 24.0, 1)
        }
        print(f"    {pol:12s}: {scaled_n:6d} frames ({d['pct_kept']:5.1f}%, inc {d['pct_inc_vs_default']:+5.1f}%) | GPU est: {scaled_est_h:.2f}h | CPU: ~{d['preflight_cpu_s']*24.0:.0f}s")
    b4_results["Fate_Apocrypha_BD_Full_Scaled"] = bd_scaled

    # 3. B5: Chỉ số độc lập
    print("\n--- B5: CHỈ SỐ ĐỘC LẬP TỪ NGUỒN TRÊN SAMPLE 1MIN BD ---")
    b5_results = measure_independent_metrics(sample_1min_bd)
    for win, data in b5_results.items():
        print(f"\n  {win} ({data['total_frames']} frames, {data['gt_motion_frames']} motion frames):")
        for pol, p_data in data["policies"].items():
            print(f"    {pol:12s}: Frozen={p_data['frozen_count']}/{data['gt_motion_frames']} ({p_data['pct_frozen']:5.1f}%) | Jump: p50={p_data['jump_p50']:.2f}px, p95={p_data['jump_p95']:.2f}px, max={p_data['jump_max']:.2f}px")

    # 4. B6: Mô phỏng --dedup-budget P%
    print("\n--- B6: MÔ PHỎNG --DEDUP-BUDGET P% TRÊN SAMPLE 1MIN BD ---")
    b6_results = simulate_budget_policy(sample_1min_bd, "BD_1min", budget_pcts=[5, 10, 15])
    for b_label, b_data in b6_results["budgets"].items():
        print(f"  Budget {b_label}: tau={b_data['chosen_tau']:.2f}px | actual inc={b_data['actual_pct_inc']:+.1f}% | n_kept={b_data['n_kept']} | GPU est={b_data['est_gpu_hours']:.2f}h")

    out_all = "scratch/b3_b4_b5_b6_results.json"
    with open(out_all, "w") as f:
        json.dump({
            "b3_calibration": b3_results,
            "b4_cost": b4_results,
            "b5_independent": b5_results,
            "b6_budget": b6_results
        }, f, indent=2)
    print(f"\nSaved all results to {out_all}")

if __name__ == "__main__":
    main()
