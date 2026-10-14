#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/mine_stutter_c2_c3.py
Thực hiện Phần C2 & C3:
1. Quét các cửa sổ 10s trên các tập video:
   - Fate Apocrypha E02 (tập BD đầy đủ, 34.213 frames)
   - sample_1min_bd.mkv (clip BD cũ, chứa đoạn khựng 44-46s)
   - sample_3min.mkv (WEB-DL 3 phút)
2. Với từng cửa sổ 10s (khoảng 240 frames):
   - Đánh giá hành vi của default, sensitive-b, drift-0.5, off.
   - Dùng DIS Optical Flow (960x540, luma) tính dòng chảy.
   - Định nghĩa sự kiện giật (stutter event):
     Tại các bước hiển thị R(t), độ lệch DIS tích lũy từ frame được giữ gần nhất R(t)
     đến frame nguồn thực tế t có bước nhảy >= 0.7 px (1080p) VÀ diện tích >= 5% khung hình.
   - Gợi ý chuyển động:
     Toàn cục nếu |dòng chảy trung vị| >= 0.2 px, ngược lại cục bộ.
3. Xếp hạng 10 cửa sổ tệ nhất mỗi tập (cách nhau >= 30s).
4. Lưu bảng tổng hợp ~/bd_model_test/stutter_candidates.csv và xuất ảnh thumbnail / contact sheet.
"""

import os
import sys
import csv
import json
import time
import subprocess
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dedup import (
    DEFAULT_DEDUP_HI,
    DEFAULT_DEDUP_LO,
    DEFAULT_DEDUP_FRAC,
    detect_drift_policy_actions
)
from scripts.calibrate_b1_b2 import get_mpdecimate_actions

DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_FAST


def analyze_window(
    video_path: str,
    start_sec: float,
    duration_sec: float = 10.0,
    fps: float = 23.976
):
    """
    Trích xuất cửa sổ 10s và tính toán số sự kiện giật dưới 4 chính sách.
    """
    cap = cv2.VideoCapture(video_path)
    start_frame = int(round(start_sec * fps))
    num_frames = int(round(duration_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_gray_540 = []
    frames_bgr_orig = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_540 = cv2.resize(gray, (960, 540))
        frames_gray_540.append(gray_540)
        frames_bgr_orig.append(frame)
    cap.release()

    n = len(frames_gray_540)
    if n < 20:
        return None

    # Tính DIS optical flow frame-by-frame
    dis_opt = cv2.DISOpticalFlow_create(DIS_PRESET)
    flows_consecutive = []
    for i in range(n - 1):
        f = dis_opt.calc(frames_gray_540[i], frames_gray_540[i+1], None)
        flows_consecutive.append(f)

    # Đo dòng chảy trung vị toàn cửa sổ (để phân loại toàn cục vs cục bộ)
    all_med_u = [np.median(f[..., 0]) * 2.0 for f in flows_consecutive]
    all_med_v = [np.median(f[..., 1]) * 2.0 for f in flows_consecutive]
    win_med_u = float(np.median(all_med_u))
    win_med_v = float(np.median(all_med_v))
    win_med_mag = float(np.hypot(win_med_u, win_med_v))

    motion_type = "Toàn cục (pan/zoom)" if win_med_mag >= 0.2 else "Cục bộ (mây/khói/sub-pixel)"

    # Lấy actions cho các chính sách trong cửa sổ này
    # Cắt một đoạn ngắn tạm để chạy mpdecimate nhanh
    temp_clip = f"/tmp/win_{start_sec:.1f}.mp4"
    cmd_cut = [
        "ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", video_path,
        "-t", f"{duration_sec:.3f}", "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
        "-an", temp_clip
    ]
    subprocess.run(cmd_cut, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    acts_default = get_mpdecimate_actions(temp_clip, hi=DEFAULT_DEDUP_HI, lo=DEFAULT_DEDUP_LO, frac=DEFAULT_DEDUP_FRAC)[:n]
    acts_sens_b = get_mpdecimate_actions(temp_clip, hi=384, lo=128, frac=0.10)[:n]
    acts_drift = detect_drift_policy_actions(temp_clip, acts_default, forced_keep_indices={0}, delta=0.5, min_response=0.10)[:n]
    acts_off = ["keep"] * n

    if os.path.exists(temp_clip):
        os.remove(temp_clip)

    def count_stutter_events(actions):
        # rep[t] là frame được giữ gần nhất
        rep = [0] * n
        last_k = 0
        for t in range(n):
            if actions[t] == "keep":
                last_k = t
            rep[t] = last_k

        stutters = 0
        max_flow_obs = 0.0
        for t in range(1, n):
            held_ref = rep[t]
            if held_ref != t:  # Frame hiện tại bị drop, đang hiển thị frame held_ref
                # Tính độ lệch giữa held_ref và frame t
                f_acc = dis_opt.calc(frames_gray_540[held_ref], frames_gray_540[t], None)
                mag = np.hypot(f_acc[..., 0], f_acc[..., 1]) * 2.0
                area_ge_07 = float(np.mean(mag >= 0.7) * 100.0)
                max_f = float(np.max(mag))
                if max_f > max_flow_obs:
                    max_flow_obs = max_f
                if max_f >= 0.7 and area_ge_07 >= 5.0:
                    stutters += 1
        return stutters, max_flow_obs

    stutter_def, max_f = count_stutter_events(acts_default)
    stutter_sb, _ = count_stutter_events(acts_sens_b)
    stutter_drift, _ = count_stutter_events(acts_drift)
    stutter_off, _ = count_stutter_events(acts_off)

    mid_frame_bgr = frames_bgr_orig[n // 2]

    return {
        "start_sec": start_sec,
        "n_frames": n,
        "motion_type": motion_type,
        "med_mag": win_med_mag,
        "max_flow": max_f,
        "stutter_default": stutter_def,
        "stutter_sensitive_b": stutter_sb,
        "stutter_drift_05": stutter_drift,
        "stutter_off": stutter_off,
        "mid_frame": mid_frame_bgr,
    }


def main():
    out_dir = os.path.expanduser("~/bd_model_test")
    cs_dir = os.path.join(out_dir, "contact_sheets")
    os.makedirs(cs_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "stutter_candidates.csv")

    targets = [
        {
            "name": "Fate_Apocrypha_E02_BD",
            "path": "/Users/xis108/Documents/RawSources/Fate Apocrypha/[AC] Fate Apocrypha - 02 [BD 1080p Hi10 FLAC][DualAudio][27EA0270].mkv",
            "step_sec": 30.0,  # Quét cách nhau 30s để bao phủ toàn bộ 24 phút
            "max_sec": 1400.0
        },
        {
            "name": "Reincarnated_Sword_E02_BD_1min",
            "path": "samples/sample_1min_bd.mkv",
            "step_sec": 5.0,
            "max_sec": 50.0
        },
        {
            "name": "Sample_3min_WEB_DL",
            "path": "samples/sample_3min.mkv",
            "step_sec": 15.0,
            "max_sec": 165.0
        }
    ]

    all_ranked_rows = []

    for tgt in targets:
        video_name = tgt["name"]
        video_path = tgt["path"]
        if not os.path.exists(video_path):
            print(f"Bỏ qua {video_name} (không tìm thấy file: {video_path})")
            continue

        print(f"\n=======================================================")
        print(f"ĐANG KHAI THÁC CỬA SỔ GIẬT: {video_name}")
        print(f"=======================================================")

        windows = []
        t = 0.0
        while t <= tgt["max_sec"]:
            try:
                res = analyze_window(video_path, start_sec=t, duration_sec=10.0)
                if res and res["stutter_default"] > 0:
                    windows.append(res)
                    print(f"[{video_name}] Cửa sổ {t:.0f}s - {t+10:.0f}s: Stutters={res['stutter_default']}, Type={res['motion_type']}, med={res['med_mag']:.2f}px")
            except Exception as e:
                print(f"Lỗi tại {t}s: {e}")
            t += tgt["step_sec"]

        # Sắp xếp theo số sự kiện giật giảm dần
        windows.sort(key=lambda x: x["stutter_default"], reverse=True)

        # Chọn top 10 cửa sổ cách nhau >= 30s
        selected = []
        for w in windows:
            # Kiểm tra khoảng cách >= 30s
            if all(abs(w["start_sec"] - s["start_sec"]) >= 30.0 for s in selected):
                selected.append(w)
            if len(selected) >= 10:
                break

        print(f"\n=> Đã chọn {len(selected)} cửa sổ giật tiêu biểu cho {video_name}.")

        # Lưu ảnh thumbnail và contact sheet
        thumb_paths = []
        for rank, s in enumerate(selected, 1):
            ts_str = f"{int(s['start_sec'])//60:02d}m{int(s['start_sec'])%60:02d}s"
            thumb_name = f"{video_name}_rank{rank:02d}_{ts_str}.jpg"
            thumb_file = os.path.join(cs_dir, thumb_name)
            
            # Ghi chú lên ảnh
            img = s["mid_frame"].copy()
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (960, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            text1 = f"#{rank} {video_name} | {ts_str} | {s['motion_type']}"
            text2 = f"Stutters: default={s['stutter_default']} | sens-b={s['stutter_sensitive_b']} | drift0.5={s['stutter_drift_05']} | off={s['stutter_off']}"
            cv2.putText(img, text1, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, text2, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Resize thumbnail 480x270
            thumb_small = cv2.resize(img, (480, 270))
            cv2.imwrite(thumb_file, thumb_small)
            thumb_paths.append(thumb_file)

            all_ranked_rows.append({
                "episode": video_name,
                "rank": rank,
                "start_sec": s["start_sec"],
                "timestamp": f"{int(s['start_sec'])//60:02d}:{int(s['start_sec'])%60:02d} - {int(s['start_sec']+10)//60:02d}:{int(s['start_sec']+10)%60:02d}",
                "motion_type": s["motion_type"],
                "med_mag_px": round(s["med_mag"], 3),
                "max_flow_px": round(s["max_flow"], 2),
                "stutter_default": s["stutter_default"],
                "stutter_sensitive_b": s["stutter_sensitive_b"],
                "stutter_drift_05": s["stutter_drift_05"],
                "stutter_off": s["stutter_off"],
                "thumbnail": thumb_file
            })

    # Ghi file CSV tổng hợp
    fieldnames = [
        "episode", "rank", "start_sec", "timestamp", "motion_type",
        "med_mag_px", "max_flow_px",
        "stutter_default", "stutter_sensitive_b", "stutter_drift_05", "stutter_off",
        "thumbnail"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_ranked_rows:
            writer.writerow(r)

    print(f"\n✅ ĐÃ HOÀN TẤT C2 & C3! Đã ghi {len(all_ranked_rows)} dòng vào {csv_path}")


if __name__ == "__main__":
    main()
