# -*- coding: utf-8 -*-
"""
scripts/calibrate_b1_b2.py
Thực hiện Mục B1 & B2:
B1. Hiệu chuẩn bộ phát hiện chuyển động:
- Lấy các cặp frame liền kề mà mpdecimate coi là lặp thật (MAE < 0.3).
- Tính độ dịch của bộ phát hiện hiện tại (480x270 phase correlation) và bộ mới B2 (960x540 DIS/Farneback + phase correlation).
- Báo phân phối |dịch| (p50/p95/p99) = ngưỡng nhiễu.
- Báo kích thước ảnh dùng và quy đổi sang px 1080p.
- Kiểm tra lại tỷ lệ "có chuyển động" khi dùng ngưỡng cao hơn nhiễu.

B2. Bộ phát hiện độc lập (DIS Optical Flow 960x540):
- Trung vị trường dòng toàn cục (pan/tilt) và phân kỳ xuyên tâm (zoom), quy đổi sang px 1080p.
- So với tương quan pha trên cửa sổ 44-46s và 3 cửa sổ ngẫu nhiên (seed cố định).
"""

import os
import sys
import json
import subprocess
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_gray_frames(video_path: str, width: int, height: int, max_frames: int = None):
    """Trích xuất chuỗi frame thang độ xám (grayscale) từ video."""
    frame_bytes = width * height
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-fps_mode", "passthrough",
        "-vf", f"scale={width}:{height},format=gray",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 10)
    frames = []
    while True:
        if max_frames and len(frames) >= max_frames:
            break
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
        frames.append(frame)
    proc.stdout.close()
    proc.wait()
    return frames


def get_mpdecimate_actions(video_path: str, hi=768, lo=320, frac=0.33):
    """Lấy danh sách hành động mpdecimate (keep/drop) cho từng frame."""
    import re
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"mpdecimate=hi={hi}:lo={lo}:frac={frac}",
        "-v", "debug",
        "-f", "null", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    pattern = re.compile(r"\[Parsed_mpdecimate_0.*?\]\s+.*?\b(keep|drop)\s+pts:")
    actions = []
    for line in proc.stderr:
        m = pattern.search(line)
        if m:
            actions.append(m.group(1))
    proc.wait()
    return actions


def compute_dis_motion(prev_u8, curr_u8, dis_opt):
    """
    Tính optical flow bằng DIS ở 960x540.
    Trả về:
    - global_dx_1080, global_dy_1080 (px 1080p)
    - global_mag_1080 (px 1080p)
    - radial_div_1080 (độ dịch xuyên tâm tối đa ở viền do zoom, px 1080p)
    """
    flow = dis_opt.calc(prev_u8, curr_u8, None) # shape (H, W, 2)
    u = flow[..., 0] # ngang (540p px)
    v = flow[..., 1] # dọc (540p px)
    
    # 1. Chuyển động tịnh tiến toàn cục = median của trường dòng
    u_med = float(np.median(u))
    v_med = float(np.median(v))
    dx_1080 = u_med * 2.0
    dy_1080 = v_med * 2.0
    global_mag_1080 = float(np.sqrt(dx_1080**2 + dy_1080**2))
    
    # 2. Phân kỳ xuyên tâm (Radial divergence cho Zoom)
    h, w = prev_u8.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    rx = x_coords - w / 2.0
    ry = y_coords - h / 2.0
    r = np.sqrt(rx**2 + ry**2)
    mask = r > 40.0 # Bỏ vùng sát tâm
    
    # Thành phần xuyên tâm của flow dọc theo vector r
    # v_rad = (rx*u + ry*v) / r
    v_rad = (rx[mask] * u[mask] + ry[mask] * v[mask]) / r[mask]
    # Tỷ lệ dãn nở k = v_rad / r
    k_radial = v_rad / r[mask]
    k_median = float(np.median(k_radial))
    
    # Độ dịch chuyển ở biên r_max (~550px) quy đổi ra 1080p
    r_max = np.sqrt((w/2)**2 + (h/2)**2)
    radial_zoom_1080 = abs(k_median * r_max * 2.0)
    
    return {
        "dx_1080": dx_1080,
        "dy_1080": dy_1080,
        "mag_1080": global_mag_1080,
        "zoom_1080": radial_zoom_1080,
        "max_motion_1080": max(global_mag_1080, radial_zoom_1080)
    }


def compute_phase_correlation(prev_f32, curr_f32, hann_win, scale_factor=2.0):
    """Tính tương quan pha và response."""
    (dx, dy), resp = cv2.phaseCorrelate(prev_f32, curr_f32, hann_win)
    dx_1080 = abs(dx * scale_factor)
    dy_1080 = abs(dy * scale_factor)
    mag_1080 = np.sqrt(dx_1080**2 + dy_1080**2)
    max_disp_1080 = max(dx_1080, dy_1080)
    return {
        "dx_1080": dx_1080,
        "dy_1080": dy_1080,
        "max_disp_1080": max_disp_1080,
        "mag_1080": mag_1080,
        "resp": resp
    }


def run_calibration(video_path: str, name: str):
    print(f"\n=======================================================")
    print(f"B1 & B2 HIỆU CHUẨN TRÊN VIDEO: {name} ({os.path.basename(video_path)})")
    print(f"=======================================================")
    
    # Trích xuất frames ở 960x540
    print("1. Đang trích xuất frame 960x540...")
    frames_540 = extract_gray_frames(video_path, 960, 540)
    n_frames = len(frames_540)
    print(f"   Đã trích xuất {n_frames} frames.")
    
    # Tạo frames thu nhỏ 480x270 cho detector cũ
    frames_270 = [cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA) for f in frames_540]
    
    # Lấy mpdecimate actions
    print("2. Đang quét mpdecimate default...")
    mp_actions = get_mpdecimate_actions(video_path)
    if len(mp_actions) != n_frames:
        print(f"   Cảnh báo: len(mp_actions)={len(mp_actions)} != n_frames={n_frames}, căn chỉnh min...")
        n_frames = min(len(mp_actions), n_frames)
        frames_540 = frames_540[:n_frames]
        frames_270 = frames_270[:n_frames]
        mp_actions = mp_actions[:n_frames]
    
    # Khởi tạo các bộ phát hiện
    dis_opt = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    hann_540 = cv2.createHanningWindow((960, 540), cv2.CV_32F)
    hann_270 = cv2.createHanningWindow((480, 270), cv2.CV_32F)
    
    # 3. Tính MAE cho tất cả các cặp liền kề (t-1, t)
    print("3. Đang tính MAE và phân loại duplicate thật (mpdecimate drop + MAE < 0.3)...")
    mae_list = []
    dup_indices = []
    
    for i in range(1, n_frames):
        mae = float(np.mean(np.abs(frames_540[i].astype(np.float32) - frames_540[i-1].astype(np.float32))))
        mae_list.append(mae)
        if mp_actions[i] == "drop" and mae < 0.3:
            dup_indices.append(i)
            
    print(f"   Tổng số cặp chuyển tiếp: {n_frames - 1}")
    print(f"   Số cặp mpdecimate drop: {sum(1 for a in mp_actions[1:] if a == 'drop')}")
    print(f"   Số cặp lặp thật (drop & MAE < 0.3): {len(dup_indices)} ({len(dup_indices)/(n_frames-1)*100:.1f}%)")
    
    # 4. Đo độ dịch trên CÁC CẶP LẶP THẬT để xác định NGƯỠNG NHIỄU (Noise Floor)
    old_pc_noise = []
    old_pc_resp = []
    new_pc540_noise = []
    new_pc540_resp = []
    new_dis_noise = []
    new_dis_zoom_noise = []
    
    for idx in dup_indices:
        # Cũ: 480x270 phase correlation (scale factor = 4.0)
        p_prev_270 = frames_270[idx-1].astype(np.float32)
        p_curr_270 = frames_270[idx].astype(np.float32)
        res_old = compute_phase_correlation(p_prev_270, p_curr_270, hann_270, scale_factor=4.0)
        old_pc_noise.append(res_old["max_disp_1080"])
        old_pc_resp.append(res_old["resp"])
        
        # Mới: 960x540 phase correlation (scale factor = 2.0)
        p_prev_540 = frames_540[idx-1].astype(np.float32)
        p_curr_540 = frames_540[idx].astype(np.float32)
        res_pc540 = compute_phase_correlation(p_prev_540, p_curr_540, hann_540, scale_factor=2.0)
        new_pc540_noise.append(res_pc540["max_disp_1080"])
        new_pc540_resp.append(res_pc540["resp"])
        
        # Mới: 960x540 DIS optical flow (scale factor = 2.0)
        res_dis = compute_dis_motion(frames_540[idx-1], frames_540[idx], dis_opt)
        new_dis_noise.append(res_dis["mag_1080"])
        new_dis_zoom_noise.append(res_dis["zoom_1080"])
        
    def stats(arr):
        if not arr:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr))
        }
        
    s_old = stats(old_pc_noise)
    s_pc540 = stats(new_pc540_noise)
    s_dis = stats(new_dis_noise)
    s_dis_zoom = stats(new_dis_zoom_noise)
    
    print("\n--- PHÂN PHỐI NHIỄU TRÊN CẶP LẶP THẬT (px 1080p) ---")
    print(f"1. Bộ cũ (Phase Correlate 480x270, scale=4.0x):")
    print(f"   p50={s_old['p50']:.4f} px | p95={s_old['p95']:.4f} px | p99={s_old['p99']:.4f} px | max={s_old['max']:.4f} px")
    print(f"   (Lưu ý: 0.15 px ở 1080p chỉ là {0.15/4.0:.4f} px ở 480x270 -> NẰM HOÀN TOÀN TRONG NHIỄU p50/p95!)")
    
    print(f"2. Bộ tương quan pha 960x540 (scale=2.0x):")
    print(f"   p50={s_pc540['p50']:.4f} px | p95={s_pc540['p95']:.4f} px | p99={s_pc540['p99']:.4f} px | max={s_pc540['max']:.4f} px")
    
    print(f"3. Bộ độc lập B2 (DIS Optical Flow 960x540, scale=2.0x):")
    print(f"   Tịnh tiến toàn cục: p50={s_dis['p50']:.4f} px | p95={s_dis['p95']:.4f} px | p99={s_dis['p99']:.4f} px | max={s_dis['max']:.4f} px")
    print(f"   Phân kỳ zoom viền:  p50={s_dis_zoom['p50']:.4f} px | p95={s_dis_zoom['p95']:.4f} px | p99={s_dis_zoom['p99']:.4f} px | max={s_dis_zoom['max']:.4f} px")
    
    # 5. Đo chuyển động trên TOÀN BỘ CÁC CẶP CHUYỂN TIẾP
    print("\n--- KIỂM TRA LẠI TỶ LỆ 'CÓ CHUYỂN ĐỘNG' TRÊN TOÀN CLIP ---")
    all_old_disp = []
    all_old_motion_015 = []
    all_dis_disp = []
    all_dis_motion_noise_p95 = []
    all_dis_motion_030 = []
    all_dis_motion_050 = []
    
    noise_thresh_dis = s_dis['p99'] # Ngưỡng p99 nhiễu DIS
    
    for i in range(1, n_frames):
        # Bộ cũ
        p_prev_270 = frames_270[i-1].astype(np.float32)
        p_curr_270 = frames_270[i].astype(np.float32)
        r_old = compute_phase_correlation(p_prev_270, p_curr_270, hann_270, scale_factor=4.0)
        has_mot_old_015 = (r_old["max_disp_1080"] >= 0.15 and r_old["resp"] >= 0.10)
        all_old_disp.append(r_old["max_disp_1080"])
        all_old_motion_015.append(has_mot_old_015)
        
        # Bộ DIS B2
        r_dis = compute_dis_motion(frames_540[i-1], frames_540[i], dis_opt)
        all_dis_disp.append(r_dis["max_motion_1080"])
        all_dis_motion_noise_p95.append(r_dis["max_motion_1080"] >= noise_thresh_dis)
        all_dis_motion_030.append(r_dis["max_motion_1080"] >= 0.30)
        all_dis_motion_050.append(r_dis["max_motion_1080"] >= 0.50)
        
    pct_old_015 = np.mean(all_old_motion_015) * 100.0
    pct_dis_noise = np.mean(all_dis_motion_noise_p95) * 100.0
    pct_dis_030 = np.mean(all_dis_motion_030) * 100.0
    pct_dis_050 = np.mean(all_dis_motion_050) * 100.0
    
    print(f"   Bộ cũ với ngưỡng 0.15 px (báo cáo cũ): {pct_old_015:.1f}% được coi là có chuyển động.")
    print(f"   Bộ DIS B2 với ngưỡng nhiễu p99 ({noise_thresh_dis:.4f} px): {pct_dis_noise:.1f}% có chuyển động.")
    print(f"   Bộ DIS B2 với ngưỡng 0.30 px 1080p: {pct_dis_030:.1f}% có chuyển động.")
    print(f"   Bộ DIS B2 với ngưỡng 0.50 px 1080p: {pct_dis_050:.1f}% có chuyển động.")
    
    # 6. B2: So khớp giữa DIS và Phase Correlate trên các cửa sổ
    # Cửa sổ 44-46s (frame idx ~ 1055 - 1103 @ 23.976 fps)
    fps = 23.976
    w44_start = int(round(44.0 * fps))
    w44_end = min(int(round(46.0 * fps)), n_frames - 1)
    
    # 3 cửa sổ ngẫu nhiên 2s (seed 42)
    np.random.seed(42)
    max_win_start = max(1, n_frames - int(2.0 * fps) - 2)
    rand_starts = np.random.choice(range(1, max_win_start), size=min(3, max_win_start), replace=False)
    rand_windows = [(s, s + int(2.0 * fps)) for s in rand_starts]
    
    all_test_windows = [("Cửa sổ 44-46s (Zoom)", w44_start, w44_end)] + [
        (f"Cửa sổ ngẫu nhiên #{idx+1} ({s/fps:.1f}s - {e/fps:.1f}s)", s, e)
        for idx, (s, e) in enumerate(rand_windows)
    ]
    
    print("\n--- B2: SO KHỚP GIỮA DIS B2 VÀ TƯƠNG QUAN PHA (PHASE CORRELATE) 960x540 ---")
    agreement_results = []
    for wname, ws, we in all_test_windows:
        dis_motions = []
        pc_motions = []
        for i in range(ws, we):
            if i >= n_frames:
                break
            r_dis = compute_dis_motion(frames_540[i-1], frames_540[i], dis_opt)
            r_pc = compute_phase_correlation(frames_540[i-1].astype(np.float32), frames_540[i].astype(np.float32), hann_540, scale_factor=2.0)
            
            # Tiêu chí: chuyển động >= 0.30 px 1080p
            dis_motions.append(r_dis["max_motion_1080"] >= 0.30)
            pc_motions.append(r_pc["max_disp_1080"] >= 0.30 and r_pc["resp"] >= 0.10)
            
        agree = sum(1 for d, p in zip(dis_motions, pc_motions) if d == p)
        total_w = len(dis_motions)
        pct_agree = (agree / total_w * 100.0) if total_w > 0 else 0.0
        print(f"   {wname}: Độ khớp nhị phân (motion >= 0.30px): {agree}/{total_w} ({pct_agree:.1f}%)")
        agreement_results.append({
            "window": wname,
            "agree_pct": pct_agree,
            "total_frames": total_w
        })
        
    return {
        "video": name,
        "n_frames": n_frames,
        "n_dup_true": len(dup_indices),
        "noise_old_pc270": s_old,
        "noise_pc540": s_pc540,
        "noise_dis540": s_dis,
        "noise_dis_zoom540": s_dis_zoom,
        "pct_old_015": pct_old_015,
        "pct_dis_noise": pct_dis_noise,
        "pct_dis_030": pct_dis_030,
        "pct_dis_050": pct_dis_050,
        "agreement_windows": agreement_results
    }


def main():
    res_bd_old = run_calibration("samples/sample_1min_bd.mkv", "BD 1min (Old keyint 250)")
    res_bd_420 = run_calibration("samples/sample_1min_bd_420.mkv", "BD 1min (GOP 24 YUV420)")
    res_web_3m = run_calibration("samples/sample_3min.mkv", "WEB-DL 3min")
    
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/calibration_results_b1_b2.json", "w", encoding="utf-8") as f:
        json.dump({
            "bd_old": res_bd_old,
            "bd_420": res_bd_420,
            "web_3m": res_web_3m
        }, f, indent=2)
    print("\n✅ Đã lưu kết quả hiệu chuẩn vào scratch/calibration_results_b1_b2.json")


if __name__ == "__main__":
    main()
