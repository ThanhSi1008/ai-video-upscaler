#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/benchmark_dis_synthetic_b1_b2.py
Thực hiện B1 & B2:
1. Sinh chuỗi 60 frames tổng hợp từ 3 ảnh nguồn 1080p:
   - frame_action.png (BD)
   - frame_close_up.png (WEB-DL)
   - frame_flat_anime.png (Flat anime)
2. Các mẫu chuyển động:
   (a) Lia thuần: 0.05, 0.1, 0.2, 0.3, 0.5, 1.0 px/frame (1080p)
   (b) Zoom tâm giữa và lệch tâm (1/3 khung): 0.02%, 0.05%, 0.1%/frame
   (c) Chuyển động cục bộ: 2%, 5%, 20% khung hình
   (d) Tĩnh + nhiễu Gaussian (0.5, 1.0, 2.0) và x264 CRF (20, 26, 32)
3. So sánh 3 phương pháp DIS:
   (i) Trung vị toàn khung (hiện tại)
   (ii) Trung vị trên gradient mask Sobel >= 20%
   (iii) Khớp mô hình affine (Huber / RANSAC) trên gradient mask Sobel
4. Đo đạc các chính sách: default, dis-0.2, dis-0.3, dis-0.4, dis-0.5
   - Tỷ lệ frames giữ
   - Tỷ lệ chuyển tiếp đóng băng (frozen transitions: ground truth >= 0.1 px nhưng mô phỏng lặp frame)
   - Tỷ lệ kích hoạt nhầm (false alarms trên chuỗi tĩnh)
   - So sánh độ phân giải 480x270, 640x360, 960x540.
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dedup import scan_dedup_plan, DEFAULT_DEDUP_HI, DEFAULT_DEDUP_LO, DEFAULT_DEDUP_FRAC

DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_FAST

# ----------------- HÀM PHÙ TRỢ HỒI QUY HUBER & AFFINE -----------------

def huber_irls_affine(X, y, delta=1.0, max_iter=5):
    """Hồi quy Huber bền vững qua Iteratively Reweighted Least Squares (IRLS)."""
    n = len(y)
    w = np.ones(n, dtype=np.float32)
    beta = np.zeros(X.shape[1], dtype=np.float32)
    for _ in range(max_iter):
        XtW = X.T * w
        try:
            beta = np.linalg.solve(XtW @ X, XtW @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0]
        res = np.abs(X @ beta - y)
        w = np.where(res <= delta, 1.0, delta / np.maximum(res, 1e-6))
    return beta

def calc_dis_scores(ref_gray, curr_gray, scale_factor=4.0):
    """
    Tính các chỉ số chuyển động theo 3 phương pháp từ cặp ảnh grayscale:
    scale_factor = 1080 / H (vd 1080/270 = 4.0, 1080/360 = 3.0, 1080/540 = 2.0).
    Trả về:
    - full_median_score: Phương pháp (i) hiện tại
    - grad_median_score: Phương pháp (ii) Sobel gradient mask >= 20%
    - affine_score: Phương pháp (iii) Max corner displacement của mô hình affine Huber
    - affine_params: (a, b, c, d, e, f)
    """
    h, w = ref_gray.shape
    dis = cv2.DISOpticalFlow_create(DIS_PRESET)
    flow = dis.calc(ref_gray, curr_gray, None)
    u = flow[..., 0] * scale_factor
    v = flow[..., 1] * scale_factor
    mag = np.hypot(u, v)

    # 1. Phương pháp (i): Full-frame median
    dx_full = float(np.median(u))
    dy_full = float(np.median(v))
    mag_pan_full = float(np.hypot(dx_full, dy_full))

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    rx = x_coords - w / 2.0
    ry = y_coords - h / 2.0
    r = np.sqrt(rx**2 + ry**2)
    r_mask = r > (20.0 if h <= 270 else 40.0)
    r_max = np.sqrt((w / 2.0)**2 + (h / 2.0)**2)

    u_m = u[r_mask]
    v_m = v[r_mask]
    r_m = r[r_mask]
    v_rad = (rx[r_mask] * u_m + ry[r_mask] * v_m) / r_m
    k_radial_full = float(np.median(v_rad / r_m))
    zoom_edge_full = abs(k_radial_full * r_max)

    p90_full = float(np.percentile(mag, 90))
    area_moving_full = float(np.mean(mag >= 0.3) * 100.0)
    score_full = max(mag_pan_full, zoom_edge_full, 0.7 * p90_full)

    # 2. Phương pháp (ii): Sobel gradient mask (top 20% gradient)
    gx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    thresh_grad = np.percentile(grad_mag, 80)
    mask_grad = grad_mag >= thresh_grad

    if np.sum(mask_grad) > 50:
        u_grad = u[mask_grad]
        v_grad = v[mask_grad]
        dx_grad = float(np.median(u_grad))
        dy_grad = float(np.median(v_grad))
        mag_pan_grad = float(np.hypot(dx_grad, dy_grad))

        mask_grad_r = mask_grad & r_mask
        if np.sum(mask_grad_r) > 20:
            u_gr = u[mask_grad_r]
            v_gr = v[mask_grad_r]
            rx_gr = rx[mask_grad_r]
            ry_gr = ry[mask_grad_r]
            r_gr = r[mask_grad_r]
            v_rad_gr = (rx_gr * u_gr + ry_gr * v_gr) / r_gr
            k_radial_grad = float(np.median(v_rad_gr / r_gr))
            zoom_edge_grad = abs(k_radial_grad * r_max)
        else:
            zoom_edge_grad = zoom_edge_full
        score_grad = max(mag_pan_grad, zoom_edge_grad, 0.7 * float(np.percentile(mag[mask_grad], 90)))
    else:
        score_grad = score_full

    # 3. Phương pháp (iii): Khớp mô hình Affine bằng Huber regression trên mask_grad
    # u(x, y) = a + b * x + c * y
    # v(x, y) = d + e * x + f * y
    # Lấy mẫu tối đa 1000 điểm trên mask_grad để tối ưu tốc độ CPU
    grad_indices = np.argwhere(mask_grad)
    if len(grad_indices) > 1000:
        step = len(grad_indices) // 1000
        grad_indices = grad_indices[::step][:1000]

    if len(grad_indices) >= 20:
        pts_y = grad_indices[:, 0]
        pts_x = grad_indices[:, 1]
        X = np.column_stack([np.ones(len(pts_x), dtype=np.float32), pts_x.astype(np.float32), pts_y.astype(np.float32)])
        u_pts = u[pts_y, pts_x]
        v_pts = v[pts_y, pts_x]

        beta_u = huber_irls_affine(X, u_pts, delta=1.0) # a, b, c
        beta_v = huber_irls_affine(X, v_pts, delta=1.0) # d, e, f

        # Tính độ dịch tại 4 góc khung hình (x, y) in {(0,0), (w,0), (0,h), (w,h)}
        corners = np.array([
            [1.0, 0.0, 0.0],
            [1.0, float(w), 0.0],
            [1.0, 0.0, float(h)],
            [1.0, float(w), float(h)]
        ], dtype=np.float32)

        u_corners = corners @ beta_u
        v_corners = corners @ beta_v
        displacements_corners = np.hypot(u_corners, v_corners)
        affine_score = float(np.max(displacements_corners))
        affine_params = (float(beta_u[0]), float(beta_u[1]), float(beta_u[2]),
                         float(beta_v[0]), float(beta_v[1]), float(beta_v[2]))
    else:
        affine_score = score_full
        affine_params = (dx_full, 0.0, 0.0, dy_full, 0.0, 0.0)

    return {
        "full_median": score_full,
        "grad_median": score_grad,
        "affine_score": affine_score,
        "affine_params": affine_params,
        "area_moving": area_moving_full,
        "mag_pan": mag_pan_full,
        "zoom_edge": zoom_edge_full,
    }


# ----------------- SINH DỮ LIỆU TỔNG HỢP VỚI GROUND TRUTH -----------------

def create_encoded_clip(frames_bgr, out_mp4_path, crf=22):
    """Ghi video và nén bằng x264 CRF để có nhiễu nén thực tế."""
    h, w = frames_bgr[0].shape[:2]
    # Pipe vào ffmpeg x264
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", "24000/1001",
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", "veryfast",
        "-g", "24",
        out_mp4_path
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames_bgr:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()

def generate_pan_frames(base_img, dx_per_frame, n=60):
    """(a) Lia thuần dx_per_frame px/frame."""
    frames = []
    h, w = base_img.shape[:2]
    for i in range(n):
        shift = i * dx_per_frame
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        warped = cv2.warpAffine(base_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        frames.append(warped)
    return frames

def generate_zoom_frames(base_img, rate_per_frame, center=None, n=60):
    """(b) Zoom tâm giữa hoặc lệch tâm rate_per_frame mỗi frame."""
    frames = []
    h, w = base_img.shape[:2]
    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = center
    for i in range(n):
        scale = (1.0 + rate_per_frame) ** i
        # M = [S 0 cx*(1-S); 0 S cy*(1-S)]
        M = np.float32([
            [scale, 0, cx * (1.0 - scale)],
            [0, scale, cy * (1.0 - scale)]
        ])
        warped = cv2.warpAffine(base_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        frames.append(warped)
    return frames

def generate_local_motion_frames(base_img, patch_pct, dx_per_frame=0.5, n=60):
    """(c) Chuyển động cục bộ một vùng chiếm patch_pct (2%, 5%, 20%) khung hình."""
    frames = []
    h, w = base_img.shape[:2]
    side = int(np.sqrt(h * w * (patch_pct / 100.0)))
    y0, x0 = h // 4, w // 4
    patch_orig = base_img[y0:y0+side, x0:x0+side].copy()
    for i in range(n):
        f = base_img.copy()
        shift = int(round(i * dx_per_frame))
        x_curr = x0 + shift
        if x_curr + side < w:
            f[y0:y0+side, x_curr:x_curr+side] = patch_orig
        frames.append(f)
    return frames

def generate_noise_frames(base_img, gaussian_sigma=0.0, n=60):
    """(d) Tĩnh + Gaussian noise sigma mức xám."""
    frames = []
    for _ in range(n):
        if gaussian_sigma > 0:
            noise = np.random.normal(0, gaussian_sigma, base_img.shape).astype(np.float32)
            f_noisy = np.clip(base_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            frames.append(f_noisy)
        else:
            frames.append(base_img.copy())
    return frames


# ----------------- HÀM ĐÁNH GIÁ VÀ ĐO ĐẠC -----------------

def evaluate_clip_policies(clip_path, ground_truth_motion_per_frame, thresh_list=[0.2, 0.3, 0.4, 0.5]):
    """
    Quét dedup plan trên clip và so sánh với ground truth.
    ground_truth_motion_per_frame: danh sách bool [có chuyển động >= 0.1 px so với frame liền trước]
    """
    results = {}
    with tempfile.TemporaryDirectory() as td:
        # 1. Default
        p_def = scan_dedup_plan(clip_path, os.path.join(td, "def.json"), policy="default")
        acts_def = []
        for c in p_def["chunks"]:
            for r in c["runs"]:
                acts_def.append("keep")
                for _ in range(r["count"] - 1):
                    acts_def.append("drop")

        n_frames = len(acts_def)
        kept_def = sum(1 for a in acts_def if a == "keep")
        
        # Frozen transitions: ground truth is True (chuyển động >= 0.1 px) nhưng actions là 'drop'
        frozen_def = sum(1 for i in range(1, n_frames) if ground_truth_motion_per_frame[i] and acts_def[i] == "drop")
        gt_motion_count = sum(1 for m in ground_truth_motion_per_frame[1:] if m)

        results["default"] = {
            "kept_frames": kept_def,
            "pct_kept": round(kept_def / n_frames * 100.0, 2),
            "frozen_transitions": frozen_def,
            "pct_frozen": round(frozen_def / max(1, gt_motion_count) * 100.0, 2),
        }

        # 2. DIS policies
        for th in thresh_list:
            policy_name = f"dis-{th:.1f}"
            p_dis = scan_dedup_plan(clip_path, os.path.join(td, f"dis_{th}.json"), policy=policy_name, dis_thresh=th)
            acts_dis = []
            for c in p_dis["chunks"]:
                for r in c["runs"]:
                    acts_dis.append("keep")
                    for _ in range(r["count"] - 1):
                        acts_dis.append("drop")
            kept_dis = sum(1 for a in acts_dis if a == "keep")
            frozen_dis = sum(1 for i in range(1, n_frames) if ground_truth_motion_per_frame[i] and acts_dis[i] == "drop")
            results[policy_name] = {
                "kept_frames": kept_dis,
                "pct_kept": round(kept_dis / n_frames * 100.0, 2),
                "frozen_transitions": frozen_dis,
                "pct_frozen": round(frozen_dis / max(1, gt_motion_count) * 100.0, 2),
            }

    return results


def run_benchmark():
    print("=================================================================")
    print("B1 & B2: XÁC THỰC CHÍNH XÁC DIS TRÊN DỮ LIỆU TỔNG HỢP VÀ THỰC TẾ")
    print("=================================================================")

    # 1. Nạp 3 frame nguồn
    src_paths = {
        "BD_Action": "test_frames/frame_action.png",
        "WEB_DL_Closeup": "test_frames/frame_close_up.png",
        "Flat_Anime": "test_frames/frame_flat_anime.png"
    }

    base_images = {}
    for name, path in src_paths.items():
        assert os.path.exists(path), f"Không tìm thấy: {path}"
        img = cv2.imread(path)
        base_images[name] = img
        print(f"Loaded {name}: {img.shape}")

    work_dir = "scratch/dis_benchmark_b1"
    os.makedirs(work_dir, exist_ok=True)

    benchmark_data = {
        "pan_experiments": [],
        "zoom_experiments": [],
        "local_motion_experiments": [],
        "noise_experiments": [],
        "b2_flat_comparison": []
    }

    # ==================== (A) LIA THUẦN (PAN) ====================
    print("\n--- B1(a): Thí nghiệm lia thuần (Pan) ---")
    pan_speeds = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    for speed in pan_speeds:
        for name, img in base_images.items():
            clip_path = os.path.join(work_dir, f"pan_{name}_{speed}px.mp4")
            frames = generate_pan_frames(img, speed, n=60)
            create_encoded_clip(frames, clip_path, crf=22)
            gt_motion = [False] + [speed >= 0.1] * 59
            res = evaluate_clip_policies(clip_path, gt_motion)
            res_entry = {
                "type": "pan",
                "source": name,
                "speed_px": speed,
                "results": res
            }
            benchmark_data["pan_experiments"].append(res_entry)
            print(f"  Pan {name:15s} @ {speed:4.2f} px/f: default={res['default']['pct_kept']:5.1f}% | dis-0.3={res['dis-0.3']['pct_kept']:5.1f}% (Frozen: def={res['default']['pct_frozen']}% vs dis-0.3={res['dis-0.3']['pct_frozen']}%)")

    # ==================== (B) ZOOM TÂM GIỮA VÀ LỆCH TÂM ====================
    print("\n--- B1(b): Thí nghiệm zoom tâm giữa và lệch tâm ---")
    zoom_rates = [0.0002, 0.0005, 0.0010] # 0.02%, 0.05%, 0.1%
    centers = [
        ("center", None),
        ("offcenter_1_3", (1920 / 3.0, 1080 / 3.0))
    ]
    for c_name, c_pos in centers:
        for rate in zoom_rates:
            rate_pct = rate * 100.0
            for name, img in base_images.items():
                clip_path = os.path.join(work_dir, f"zoom_{c_name}_{name}_{rate_pct:.2f}pct.mp4")
                frames = generate_zoom_frames(img, rate, center=c_pos, n=60)
                create_encoded_clip(frames, clip_path, crf=22)
                # Max corner displacement: delta_x = rate * max_dist >= 0.1 px
                max_dist = 1100.0
                edge_disp = rate * max_dist
                gt_motion = [False] + [edge_disp >= 0.1] * 59
                res = evaluate_clip_policies(clip_path, gt_motion)
                res_entry = {
                    "type": "zoom",
                    "center": c_name,
                    "source": name,
                    "rate_pct": rate_pct,
                    "edge_disp_px": round(edge_disp, 3),
                    "results": res
                }
                benchmark_data["zoom_experiments"].append(res_entry)
                print(f"  Zoom ({c_name:13s}) {name:15s} @ {rate_pct:4.2f}%/f (edge {edge_disp:4.2f}px): def={res['default']['pct_kept']:5.1f}% | dis-0.3={res['dis-0.3']['pct_kept']:5.1f}%")

    # ==================== (C) CHUYỂN ĐỘNG CỤC BỘ ====================
    print("\n--- B1(c): Thí nghiệm chuyển động cục bộ (2%, 5%, 20%) ---")
    patch_pcts = [2.0, 5.0, 20.0]
    for patch_pct in patch_pcts:
        for name, img in base_images.items():
            clip_path = os.path.join(work_dir, f"local_{name}_{patch_pct}pct.mp4")
            frames = generate_local_motion_frames(img, patch_pct, dx_per_frame=0.5, n=60)
            create_encoded_clip(frames, clip_path, crf=22)
            gt_motion = [False] + [True] * 59
            res = evaluate_clip_policies(clip_path, gt_motion)
            res_entry = {
                "type": "local_motion",
                "source": name,
                "patch_pct": patch_pct,
                "results": res
            }
            benchmark_data["local_motion_experiments"].append(res_entry)
            print(f"  Local motion {name:15s} area {patch_pct:4.1f}%: def={res['default']['pct_kept']:5.1f}% | dis-0.3={res['dis-0.3']['pct_kept']:5.1f}%")

    # ==================== (D) TĨNH + NHIỄU (GAUSSIAN & CRF) ====================
    print("\n--- B1(d): Thí nghiệm tĩnh + nhiễu (Đo kích hoạt nhầm) ---")
    sigmas = [0.5, 1.0, 2.0]
    for sig in sigmas:
        for name, img in base_images.items():
            clip_path = os.path.join(work_dir, f"noise_gauss_{name}_{sig}.mp4")
            frames = generate_noise_frames(img, gaussian_sigma=sig, n=60)
            create_encoded_clip(frames, clip_path, crf=22)
            gt_motion = [False] * 60 # Static scene -> no real motion
            res = evaluate_clip_policies(clip_path, gt_motion)
            res_entry = {
                "type": "gaussian_noise",
                "source": name,
                "sigma": sig,
                "results": res
            }
            benchmark_data["noise_experiments"].append(res_entry)
            false_alarm_def = max(0, res['default']['kept_frames'] - 1)
            false_alarm_dis = max(0, res['dis-0.3']['kept_frames'] - 1)
            print(f"  Noise Gauss sigma={sig:3.1f} {name:15s}: extra kept frames def={false_alarm_def} | dis-0.3={false_alarm_dis}")

    crf_list = [20, 26, 32]
    for crf_val in crf_list:
        for name, img in base_images.items():
            clip_path = os.path.join(work_dir, f"static_crf_{name}_{crf_val}.mp4")
            frames = generate_noise_frames(img, gaussian_sigma=0.0, n=60)
            create_encoded_clip(frames, clip_path, crf=crf_val)
            gt_motion = [False] * 60
            res = evaluate_clip_policies(clip_path, gt_motion)
            res_entry = {
                "type": "crf_compression",
                "source": name,
                "crf": crf_val,
                "results": res
            }
            benchmark_data["noise_experiments"].append(res_entry)
            false_alarm_def = max(0, res['default']['kept_frames'] - 1)
            false_alarm_dis = max(0, res['dis-0.3']['kept_frames'] - 1)
            print(f"  Static CRF={crf_val:2d} {name:15s}: extra kept frames def={false_alarm_def} | dis-0.3={false_alarm_dis}")

    # ==================== B2: SO SÁNH 3 PHƯƠNG PHÁP TRÊN NỀN PHẲNG ====================
    print("\n--- B2: Đo sai số 3 phương pháp trên nền phẳng vs đáp án thật ---")
    flat_img = base_images["Flat_Anime"]
    resolutions = [
        ("480x270", 480, 270, 4.0),
        ("640x360", 640, 360, 3.0),
        ("960x540", 960, 540, 2.0)
    ]

    test_cases_b2 = [
        ("Pan 0.3 px", generate_pan_frames(flat_img, 0.3, n=2), 0.30),
        ("Pan 0.5 px", generate_pan_frames(flat_img, 0.5, n=2), 0.50),
        ("Zoom center 0.05%", generate_zoom_frames(flat_img, 0.0005, center=None, n=2), 0.55),
        ("Zoom offcenter (1/3) 0.05%", generate_zoom_frames(flat_img, 0.0005, center=(1920/3.0, 1080/3.0), n=2), 0.65),
        ("Static flat frame", generate_noise_frames(flat_img, 0.0, n=2), 0.00),
    ]

    for label, pair_frames, truth_score in test_cases_b2:
        print(f"\n  Case: {label} (Ground Truth displacement: {truth_score:.3f} px)")
        for res_name, sw, sh, sfac in resolutions:
            g1 = cv2.resize(cv2.cvtColor(pair_frames[0], cv2.COLOR_BGR2GRAY), (sw, sh))
            g2 = cv2.resize(cv2.cvtColor(pair_frames[1], cv2.COLOR_BGR2GRAY), (sw, sh))
            t0 = time.perf_counter()
            scores = calc_dis_scores(g1, g2, scale_factor=sfac)
            calc_time_ms = (time.perf_counter() - t0) * 1000.0

            s_full = scores["full_median"]
            s_grad = scores["grad_median"]
            s_affine = scores["affine_score"]

            err_full = abs(s_full - truth_score)
            err_grad = abs(s_grad - truth_score)
            err_affine = abs(s_affine - truth_score)

            entry_b2 = {
                "case": label,
                "truth_score": truth_score,
                "res": res_name,
                "calc_time_ms": round(calc_time_ms, 2),
                "full_median": round(s_full, 3),
                "grad_median": round(s_grad, 3),
                "affine_score": round(s_affine, 3),
                "err_full": round(err_full, 3),
                "err_grad": round(err_grad, 3),
                "err_affine": round(err_affine, 3),
            }
            benchmark_data["b2_flat_comparison"].append(entry_b2)
            print(f"    [{res_name:7s} - {calc_time_ms:5.1f}ms]: (i)FullMed={s_full:5.3f} (err {err_full:5.3f}) | (ii)GradMed={s_grad:5.3f} (err {err_grad:5.3f}) | (iii)Affine={s_affine:5.3f} (err {err_affine:5.3f})")

    # Lưu kết quả ra JSON
    out_json = "scratch/b1_b2_results.json"
    with open(out_json, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\nSaved benchmark results to {out_json}")


if __name__ == "__main__":
    run_benchmark()
