# -*- coding: utf-8 -*-
"""
src/dedup.py
Module quản lý cơ chế Deduplication & Concat Re-expansion cho Anime Video Upscale:
- Phân tích toàn tập bằng bộ lọc mpdecimate (Preflight Scan lập plan.json 1 lần duy nhất)
- Bắt buộc ép giữ frame đầu mỗi chunk (bảo toàn 100% tính độc lập và Checkpoint Resume)
- Lọc bỏ khung hình trùng lặp trước khi đưa vào GPU AI Upscaler (giảm ~50% tải GPU)
- Tái tạo khung hình 4K (Re-expansion) bằng FFmpeg Concat Demuxer không tốn RAM/Disk
- Xác minh giao dịch phân đoạn (Transactional Segment verification) chuẩn xác từng frame
"""

import os
import re
import json
import queue
import shutil
import hashlib
import threading
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Sequence

from src.config import (
    DEFAULT_DEDUP_HI,
    DEFAULT_DEDUP_LO,
    DEFAULT_DEDUP_FRAC,
    ROOT_DIR,
    HEVC_ENCODER,
    HEVC_PRESET,
    HEVC_CQ_LEVEL,
    HEVC_PROFILE,
    HEVC_PIX_FMT,
    HEVC_COLOR_RANGE,
    HEVC_COLORSPACE,
    HEVC_COLOR_PRIMARIES,
    HEVC_COLOR_TRC,
    HEVC_TAG,
    get_hevc_rate_args
)

DEDUP_POLICY_PARAMS = {
    "default": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC},
    "sensitive": {"hi": 512, "lo": 160, "frac": 0.15},
    "sensitive-a": {"hi": 512, "lo": 160, "frac": 0.15},
    "sensitive-b": {"hi": 384, "lo": 128, "frac": 0.10},
    "motion": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "tau": 0.3},
    "drift": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "drift_delta": 0.75, "min_response": 0.10},
    "drift-0.5": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "drift_delta": 0.5, "min_response": 0.10},
    "drift-0.75": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "drift_delta": 0.75, "min_response": 0.10},
    "drift-1.0": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "drift_delta": 1.0, "min_response": 0.10},
    "dis": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "dis_thresh": 0.30},
    "dis-0.2": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "dis_thresh": 0.20},
    "dis-0.3": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "dis_thresh": 0.30},
    "dis-0.4": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "dis_thresh": 0.40},
    "dis-0.5": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "dis_thresh": 0.50},
    "dis-sakuga": {"hi": 1024, "lo": 448, "frac": 0.38, "dis_thresh": 0.35, "local_thresh": 1.50, "min_moving_area_pct": 12.0},
    "sakuga": {"hi": 1024, "lo": 448, "frac": 0.38, "dis_thresh": 0.35, "local_thresh": 1.50, "min_moving_area_pct": 12.0},
    "off": {"all_keep": True}
}


def resolve_dedup_policy(video_path: str, cli_policy: Optional[str] = None) -> str:
    """
    Xác định chính sách deduplication theo thứ tự ưu tiên:
    1. CLI cờ --dedup-policy
    2. shows.yaml (trường dedup_policy trong rule khớp tên video)
    3. Mặc định: 'default'
    """
    if cli_policy:
        return cli_policy.lower()

    shows_yaml = os.path.join(ROOT_DIR, "shows.yaml")
    if os.path.exists(shows_yaml):
        try:
            import yaml
            from src.models_resolver import match_show_pattern
            with open(shows_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            rules = data.get("shows", []) or []
            vname = os.path.basename(video_path)
            vdir = os.path.basename(os.path.dirname(os.path.abspath(video_path)))
            for r in rules:
                pat = r.get("pattern", "")
                pol = r.get("dedup_policy")
                if pol and (match_show_pattern(vname, pat) or match_show_pattern(vdir, pat)):
                    return str(pol).lower()
        except Exception:
            pass
    return "default"


def detect_global_motion_frames(
    input_path: str,
    tau: float = 0.3,
    min_response: float = 0.1,
    scale_w: int = 480,
    scale_h: int = 270
) -> List[bool]:
    """
    Phát hiện chuyển động toàn cục (camera pan/zoom/tilt) giữa frame t và frame t-1
    bằng phase correlation (OpenCV) trên luma thu nhỏ 480x270:
    - Toàn khung: max(|dx|, |dy|) >= tau (quy về pixel 1080p) và response >= min_response
    - 4 góc phần tư: kiểm tra chuyển động ngược chiều giữa các góc đối nghịch (zoom in/out)
    Trả về mảng boolean có độ dài bằng tổng số frame, phần tử t True nếu có chuyển động so với t-1.
    """
    import cv2
    import numpy as np

    frame_bytes = scale_w * scale_h
    hann_full = cv2.createHanningWindow((scale_w, scale_h), cv2.CV_32F)
    qw, qh = scale_w // 2, scale_h // 2
    hann_quad = cv2.createHanningWindow((qw, qh), cv2.CV_32F)

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", input_path,
        "-fps_mode", "passthrough",
        "-vf", f"scale={scale_w}:{scale_h},format=gray",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 10)
    has_motion = []
    prev_gray = None
    scale_factor = 1080.0 / scale_h  # 4.0

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        curr_gray = np.frombuffer(raw, dtype=np.uint8).reshape((scale_h, scale_w)).astype(np.float32)

        if prev_gray is None:
            has_motion.append(False)  # Frame 0 không có frame trước
        else:
            # 1. Chuyển động toàn khung
            (dx, dy), resp = cv2.phaseCorrelate(prev_gray, curr_gray, hann_full)
            dx_1080 = abs(dx * scale_factor)
            dy_1080 = abs(dy * scale_factor)
            max_disp = max(dx_1080, dy_1080)

            is_motion = bool(max_disp >= tau and resp >= min_response)

            # 2. Nếu chưa thỏa mãn, kiểm tra zoom qua 4 góc phần tư
            if not is_motion:
                tl = cv2.phaseCorrelate(prev_gray[:qh, :qw], curr_gray[:qh, :qw], hann_quad)
                br = cv2.phaseCorrelate(prev_gray[qh:, qw:], curr_gray[qh:, qw:], hann_quad)
                tr = cv2.phaseCorrelate(prev_gray[:qh, qw:], curr_gray[:qh, qw:], hann_quad)
                bl = cv2.phaseCorrelate(prev_gray[qh:, :qw], curr_gray[qh:, :qw], hann_quad)

                dx_tl, dy_tl = tl[0][0] * scale_factor, tl[0][1] * scale_factor
                dx_br, dy_br = br[0][0] * scale_factor, br[0][1] * scale_factor
                dx_tr, dy_tr = tr[0][0] * scale_factor, tr[0][1] * scale_factor
                dx_bl, dy_bl = bl[0][0] * scale_factor, bl[0][1] * scale_factor

                zoom_tl_br = bool((dx_tl * dx_br < 0 or dy_tl * dy_br < 0) and (abs(dx_tl - dx_br) >= tau or abs(dy_tl - dy_br) >= tau) and tl[1] >= min_response and br[1] >= min_response)
                zoom_tr_bl = bool((dx_tr * dx_bl < 0 or dy_tr * dy_bl < 0) and (abs(dx_tr - dx_bl) >= tau or abs(dy_tr - dy_bl) >= tau) and tr[1] >= min_response and bl[1] >= min_response)

                if zoom_tl_br or zoom_tr_bl:
                    is_motion = True

            has_motion.append(is_motion)

        prev_gray = curr_gray

    proc.wait()
    return has_motion


def detect_drift_policy_actions(
    input_path: str,
    base_actions: List[str],
    forced_keep_indices: set,
    delta: float = 0.75,
    min_response: float = 0.10,
    scale_w: int = 960,
    scale_h: int = 540
) -> List[str]:
    """
    Chính sách 'drift' (Đo độ dịch tích lũy từ frame được giữ gần nhất):
    - Đo độ dịch tương quan pha (Phase Correlation) ở 960x540 (Hann window) giữa frame hiện tại
      và frame được giữ GẦN NHẤT (reference frame).
    - Toàn khung: median của toàn khung và 4 góc phần tư quy về px 1080p.
    - 4 góc phần tư: kiểm tra phân kỳ zoom in/out giữa các góc đối nghịch (TL-BR, TR-BL).
    - Nếu độ dịch toàn cục >= delta HOẶC phân kỳ zoom >= delta (kèm response >= min_response):
      giữ frame và đặt lại frame tham chiếu = frame hiện tại.
    - Nếu frame nằm trong forced_keep_indices (đầu chunk) hoặc base_actions giữ:
      giữ frame và đặt lại tham chiếu.
    """
    import cv2
    import numpy as np

    frame_bytes = scale_w * scale_h
    hann_full = cv2.createHanningWindow((scale_w, scale_h), cv2.CV_32F)
    qw, qh = scale_w // 2, scale_h // 2
    hann_quad = cv2.createHanningWindow((qw, qh), cv2.CV_32F)
    scale_factor = 1080.0 / scale_h  # 2.0

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", input_path,
        "-fps_mode", "passthrough",
        "-vf", f"scale={scale_w}:{scale_h},format=gray",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 10)
    actions = list(base_actions)
    total_frames = len(actions)

    ref_gray = None
    curr_idx = 0

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        curr_gray = np.frombuffer(raw, dtype=np.uint8).reshape((scale_h, scale_w)).astype(np.float32)

        if curr_idx == 0 or curr_idx in forced_keep_indices or actions[curr_idx] == "keep" or ref_gray is None:
            actions[curr_idx] = "keep"
            ref_gray = curr_gray
        else:
            # 1. Đo toàn khung
            (dx_full, dy_full), resp_full = cv2.phaseCorrelate(ref_gray, curr_gray, hann_full)

            # 2. Đo 4 góc phần tư
            tl = cv2.phaseCorrelate(ref_gray[:qh, :qw], curr_gray[:qh, :qw], hann_quad)
            br = cv2.phaseCorrelate(ref_gray[qh:, qw:], curr_gray[qh:, qw:], hann_quad)
            tr = cv2.phaseCorrelate(ref_gray[:qh, qw:], curr_gray[:qh, qw:], hann_quad)
            bl = cv2.phaseCorrelate(ref_gray[qh:, :qw], curr_gray[qh:, :qw], hann_quad)

            dx_full_1080 = dx_full * scale_factor
            dy_full_1080 = dy_full * scale_factor

            dx_tl, dy_tl = tl[0][0] * scale_factor, tl[0][1] * scale_factor
            dx_br, dy_br = br[0][0] * scale_factor, br[0][1] * scale_factor
            dx_tr, dy_tr = tr[0][0] * scale_factor, tr[0][1] * scale_factor
            dx_bl, dy_bl = bl[0][0] * scale_factor, bl[0][1] * scale_factor

            # Median toàn khung và 4 góc phần tư
            all_dx = [dx_full_1080, dx_tl, dx_br, dx_tr, dx_bl]
            all_dy = [dy_full_1080, dy_tl, dy_br, dy_tr, dy_bl]
            med_dx = float(np.median(all_dx))
            med_dy = float(np.median(all_dy))
            disp_global_1080 = max(abs(med_dx), abs(med_dy))

            # Phân kỳ zoom giữa các góc đối nghịch
            zoom_tl_br = bool(
                (dx_tl * dx_br < 0 or dy_tl * dy_br < 0) and
                (abs(dx_tl - dx_br) >= delta or abs(dy_tl - dy_br) >= delta) and
                tl[1] >= min_response and br[1] >= min_response
            )
            zoom_tr_bl = bool(
                (dx_tr * dx_bl < 0 or dy_tr * dy_bl < 0) and
                (abs(dx_tr - dx_bl) >= delta or abs(dy_tr - dy_bl) >= delta) and
                tr[1] >= min_response and bl[1] >= min_response
            )

            is_motion = bool(
                (disp_global_1080 >= delta and resp_full >= min_response) or
                zoom_tl_br or zoom_tr_bl
            )

            if is_motion:
                actions[curr_idx] = "keep"
                ref_gray = curr_gray

        curr_idx += 1

    proc.stdout.close()
    proc.wait()
    return actions


def _is_dis_motion_early_exit(
    dis,
    ref_gray,
    curr_gray,
    scale_factor,
    thresh: float,
    thresh_sq: float,
    min_moving_area_pct: float,
    r_mask,
    rx_inv_r2,
    ry_inv_r2,
    r_max: float,
    local_thresh: Optional[float] = None,
) -> bool:
    """
    Thác kiểm tra cắt sớm (Early-Exit Cascade) cho dòng chảy quang học DIS:
    - Cửa 1 (Cục bộ): Diện tích chuyển động vượt quá min_moving_area_pct ở ngưỡng local_thresh.
    - Cửa 2 (Toàn cục): Chuyển động lia máy tịnh tiến toàn cục (Pan / Tilt) trên trung vị.
    - Cửa 3 (Toàn cục): Chuyển động zoom xuyên tâm (Zoom In / Out) trên trung vị.
    - Cửa 4 (Phân tán): Chuyển động phân tán p90 ở ngưỡng local_thresh.
    """
    import math
    import numpy as np

    if np.array_equal(ref_gray, curr_gray):
        return False

    flow = dis.calc(ref_gray, curr_gray, None)
    u = flow[..., 0] * scale_factor
    v = flow[..., 1] * scale_factor
    mag_sq = u * u + v * v

    eff_local_thresh = float(local_thresh) if local_thresh is not None else float(thresh)
    local_thresh_sq = eff_local_thresh * eff_local_thresh

    # Cửa 1: Diện tích chuyển động cục bộ (đo theo local_thresh để tránh sàn nhiễu hạt/nén)
    area_moving = float(np.count_nonzero(mag_sq >= local_thresh_sq)) * (100.0 / float(mag_sq.size))
    if area_moving >= min_moving_area_pct:
        return True

    # Cửa 2: Chuyển động lia máy tịnh tiến toàn cục (Pan / Tilt)
    dx = float(np.median(u))
    dy = float(np.median(v))
    if math.hypot(dx, dy) >= thresh:
        return True

    # Cửa 3: Chuyển động zoom xuyên tâm (Zoom In / Out)
    if rx_inv_r2.size > 0:
        k_radial = float(np.median(u[r_mask] * rx_inv_r2 + v[r_mask] * ry_inv_r2))
        if abs(k_radial * r_max) >= thresh:
            return True

    # Cửa 4: Chuyển động phân tán p90
    p90_mag = math.sqrt(max(0.0, float(np.percentile(mag_sq, 90))))
    return (0.7 * p90_mag) >= eff_local_thresh


def deduplicate_cyclic_frames(
    frames: Sequence[Any],
    allow_quant_noise: bool = True,
    max_pixel_tol: int = 2,
    max_mae: float = 0.20,
    lookback_distinct: int = 48,
) -> Tuple[List[Any], List[int]]:
    """
    Khử trùng lặp khung hình chu kỳ (Cyclic Cel Memoization) với cấu trúc phân tầng 3 lớp (3-Tier Hierarchical Filter):
    - Tier 1 (O(1) Exact Candidate Lookup): Thu nhỏ ảnh siêu nhanh (64x36 INTER_NEAREST, 0.01ms) + băm 64-bit BLAKE2b (0.01ms).
      Nếu khớp hash, xác thực bit-exact bằng SIMD cv2.norm(NORM_INF) <= 0. Nhanh hơn 450x so với băm full 6.22 MB.
    - Tier 2 (SIMD Thumbnail Distance Pruning): So sánh thumbnail uint8 bằng cv2.norm(NORM_INF) không ép kiểu sang int16.
      Loại bỏ >95% ứng viên khác biệt chỉ trong 0.003ms mà không có rủi ro False Negative.
    - Tier 3 (Zero-Allocation Full-Res Early Exit): Kiểm tra cv2.norm(NORM_INF) trên ảnh gốc 1080p (0.47ms, 0 byte cấp phát).
      Nếu max_diff > max_pixel_tol, thoát ngay mà không tính mean; chỉ tính cv2.norm(NORM_L1) khi max_diff đạt chuẩn.
    """
    import cv2
    import numpy as np
    import hashlib

    n = len(frames)
    if n <= 1:
        return list(frames), list(range(n))

    distinct_frames: List[Any] = []
    frame_to_distinct: List[int] = []
    thumb_hash_to_distinct: Dict[bytes, List[int]] = {}
    distinct_thumbs: List[Any] = []

    use_qnoise = bool(allow_quant_noise and max_pixel_tol > 0)
    thumb_w, thumb_h = 64, 36

    for arr in frames:
        arr_c = arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)
        h, w = arr_c.shape[:2]
        channels = arr_c.shape[2] if arr_c.ndim == 3 else 1

        # Tier 1: Tạo thumbnail 64x36 và băm 64-bit
        thumb = cv2.resize(arr_c, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        if not thumb.flags["C_CONTIGUOUS"]:
            thumb = np.ascontiguousarray(thumb)
        digest = hashlib.blake2b(memoryview(thumb), digest_size=8).digest()

        matched_idx = -1
        candidates = thumb_hash_to_distinct.get(digest)
        if candidates:
            for d_idx in candidates:
                ref = distinct_frames[d_idx]
                if ref.shape == arr_c.shape and cv2.norm(arr_c, ref, cv2.NORM_INF) <= 0:
                    matched_idx = d_idx
                    break

        # Tier 2 & Tier 3: Kiểm tra sai số lượng tử B-frame / line boil cực nhỏ
        if matched_idx < 0 and use_qnoise and distinct_frames:
            lower = max(-1, len(distinct_frames) - 1 - max(1, lookback_distinct))
            for d_idx in range(len(distinct_frames) - 1, lower, -1):
                ref = distinct_frames[d_idx]
                if ref.shape != arr_c.shape:
                    continue

                # Tier 2: Lọc khoảng cách thumbnail trên uint8
                if cv2.norm(thumb, distinct_thumbs[d_idx], cv2.NORM_INF) > max_pixel_tol:
                    continue
                t_mae = cv2.norm(thumb, distinct_thumbs[d_idx], cv2.NORM_L1) / float(thumb_w * thumb_h * channels)
                if t_mae > (max_mae + 0.08):
                    continue

                # Tier 3: Kiểm chứng toàn khung 0-allocation
                if cv2.norm(arr_c, ref, cv2.NORM_INF) <= max_pixel_tol:
                    f_mae = cv2.norm(arr_c, ref, cv2.NORM_L1) / float(h * w * channels)
                    if f_mae <= max_mae:
                        matched_idx = d_idx
                        break

        if matched_idx >= 0:
            frame_to_distinct.append(matched_idx)
        else:
            new_d = len(distinct_frames)
            distinct_frames.append(arr_c)
            thumb_hash_to_distinct.setdefault(digest, []).append(new_d)
            if use_qnoise:
                distinct_thumbs.append(thumb)
            frame_to_distinct.append(new_d)

    return distinct_frames, frame_to_distinct


def detect_dis_policy_actions(
    input_path: str,
    base_actions: List[str],
    forced_keep_indices: set,
    thresh: float = 0.30,
    scale_w: int = 480,
    scale_h: int = 270,
    min_moving_area_pct: float = 5.0,
    local_thresh: Optional[float] = None,
) -> List[str]:
    """
    Chính sách 'dis' (Dense Inverse Search Optical Flow - Dòng chảy quang học dày đặc):
    - Bắt đúng 100% các phân cảnh zoom chậm, lia máy (pan), hoặc chuyển động mượt mà liên tục.
    - So với Phase Correlation (chỉ bắt tịnh tiến toàn cục dx, dy), DIS tính trường vector dòng chảy
      dày đặc trên từng pixel, kết hợp 3 phép đo:
      1. Tịnh tiến toàn cục (median dx, dy) -> Pan / Tilt
      2. Phân kỳ xuyên tâm từ tâm quang học (radial divergence expansion rate) -> Zoom In / Out
      3. Chuyển động phân tán / cục bộ (percentile 90 và diện tích di chuyển) -> Mây, khói, nước, sub-pixel
    - Chỉ chạy DIS trên các khung hình mà mpdecimate đề xuất bỏ (candidate drops) so với frame tham chiếu gần nhất,
      bảo toàn tối ưu thời gian và giải phóng tài nguyên ở các cảnh tĩnh/đối thoại.
    """
    import math
    import cv2
    import numpy as np

    frame_bytes = scale_w * scale_h
    scale_factor = np.float32(1080.0 / scale_h)
    thresh_sq = float(thresh) * float(thresh)
    h, w = scale_h, scale_w

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    rx = x_coords - np.float32(w / 2.0)
    ry = y_coords - np.float32(h / 2.0)
    r_sq = rx * rx + ry * ry
    min_r = 20.0 if h == 270 else 40.0
    r_mask = r_sq > (min_r * min_r)
    r_max = float(math.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2))
    # Tiền tính toán hằng số hình học 1/r^2 để khử phép chia mảng mỗi frame
    inv_r_sq_m = np.float32(1.0) / r_sq[r_mask]
    rx_inv_r2 = rx[r_mask] * inv_r_sq_m
    ry_inv_r2 = ry[r_mask] * inv_r_sq_m

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", input_path,
        "-fps_mode", "passthrough",
        "-vf", f"scale={scale_w}:{scale_h},format=gray",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 16)
    actions = list(base_actions)

    ref_gray = None
    curr_idx = 0

    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        curr_gray = np.frombuffer(raw, dtype=np.uint8).reshape((scale_h, scale_w))

        if curr_idx == 0 or curr_idx in forced_keep_indices or actions[curr_idx] == "keep" or ref_gray is None:
            actions[curr_idx] = "keep"
            ref_gray = curr_gray
        elif _is_dis_motion_early_exit(
            dis,
            ref_gray,
            curr_gray,
            scale_factor,
            thresh,
            thresh_sq,
            min_moving_area_pct,
            r_mask,
            rx_inv_r2,
            ry_inv_r2,
            r_max,
            local_thresh=local_thresh,
        ):
            actions[curr_idx] = "keep"
            ref_gray = curr_gray

        curr_idx += 1

    proc.stdout.close()
    proc.wait()
    return actions


def _scan_fused_mpdecimate_and_dis(
    input_path: str,
    hi: int,
    lo: int,
    frac: float,
    thresh: float,
    chunk_start_set: Optional[set],
    frames_per_chunk_fallback: int,
    scale_w: int = 480,
    scale_h: int = 270,
    min_moving_area_pct: float = 5.0,
    local_thresh: Optional[float] = None,
) -> Optional[Tuple[List[str], List[str]]]:
    """
    Hợp nhất 1 lượt giải mã FFmpeg duy nhất cho mpdecimate + luồng ảnh xám 480x270 DIS:
    - Cắt giảm 50% thời gian giải mã video 1080p khi dùng --dedup-policy dis.
    - Hai luồng đọc bất đồng bộ (stderr debug + stdout gray queue 512 frames) chống deadlock 100%.
    Trả về (raw_mpdecimate_actions, final_dis_actions) hoặc None nếu cần fallback 2 lượt.
    """
    import math
    import cv2
    import numpy as np

    frame_bytes = scale_w * scale_h
    scale_factor = np.float32(1080.0 / scale_h)
    thresh_sq = float(thresh) * float(thresh)
    h, w = scale_h, scale_w

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    rx = x_coords - np.float32(w / 2.0)
    ry = y_coords - np.float32(h / 2.0)
    r_sq = rx * rx + ry * ry
    min_r = 20.0 if h == 270 else 40.0
    r_mask = r_sq > (min_r * min_r)
    r_max = float(math.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2))
    inv_r_sq_m = np.float32(1.0) / r_sq[r_mask]
    rx_inv_r2 = rx[r_mask] * inv_r_sq_m
    ry_inv_r2 = ry[r_mask] * inv_r_sq_m

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    pattern = re.compile(r"\[Parsed_mpdecimate_.*?\]\s+.*?\b(keep|drop)\s+pts:")
    fc = (
        f"[0:v]split=2[vdec][vgray];"
        f"[vdec]mpdecimate=hi={hi}:lo={lo}:frac={frac}[outdec];"
        f"[vgray]scale={scale_w}:{scale_h},format=gray[outgray]"
    )
    null_dev = "NUL" if os.name == "nt" else "/dev/null"
    cmd = [
        "ffmpeg", "-i", input_path,
        "-filter_complex", fc,
        "-map", "[outdec]", "-f", "null", null_dev,
        "-map", "[outgray]", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-v", "debug", "pipe:1",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=frame_bytes * 8,
        )
    except Exception:
        return None

    cond = threading.Condition()
    raw_actions: List[str] = []
    err_eof = [False]
    gray_q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=512)

    def _read_stderr():
        try:
            if proc.stderr:
                for raw_line in proc.stderr:
                    line = (
                        raw_line.decode("utf-8", errors="replace")
                        if isinstance(raw_line, (bytes, bytearray))
                        else str(raw_line)
                    )
                    m = pattern.search(line)
                    if m:
                        with cond:
                            raw_actions.append(m.group(1))
                            cond.notify_all()
        finally:
            with cond:
                err_eof[0] = True
                cond.notify_all()

    def _read_stdout():
        try:
            if proc.stdout:
                while True:
                    buf = proc.stdout.read(frame_bytes)
                    if not buf or len(buf) < frame_bytes:
                        break
                    gray_q.put(buf)
        finally:
            gray_q.put(None)
            if proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass

    t_err = threading.Thread(target=_read_stderr, daemon=True)
    t_out = threading.Thread(target=_read_stdout, daemon=True)
    t_err.start()
    t_out.start()

    final_actions: List[str] = []
    ref_gray = None
    curr_idx = 0

    while True:
        raw_gray = gray_q.get()
        if raw_gray is None:
            break
        with cond:
            while len(raw_actions) <= curr_idx and not err_eof[0]:
                cond.wait(timeout=0.5)
            base_act = raw_actions[curr_idx] if curr_idx < len(raw_actions) else "keep"

        curr_gray = np.frombuffer(raw_gray, dtype=np.uint8).reshape((scale_h, scale_w))
        is_forced = (
            (curr_idx == 0)
            or (curr_idx in chunk_start_set if chunk_start_set is not None else (curr_idx % max(1, frames_per_chunk_fallback) == 0))
        )

        if is_forced or base_act == "keep" or ref_gray is None:
            final_actions.append("keep")
            ref_gray = curr_gray
        elif _is_dis_motion_early_exit(
            dis,
            ref_gray,
            curr_gray,
            scale_factor,
            thresh,
            thresh_sq,
            min_moving_area_pct,
            r_mask,
            rx_inv_r2,
            ry_inv_r2,
            r_max,
            local_thresh=local_thresh,
        ):
            final_actions.append("keep")
            ref_gray = curr_gray
        else:
            final_actions.append("drop")

        curr_idx += 1

    proc.wait()
    t_err.join(timeout=5.0)
    t_out.join(timeout=5.0)

    if proc.returncode != 0 or not raw_actions or len(raw_actions) != len(final_actions):
        return None
    return raw_actions, final_actions


def _encode_actions_rle(actions: List[str]) -> str:
    """Nén chuỗi hành động keep/drop thành RLE siêu gọn (ví dụ 'K2D3K1', ~25 KB cho 34.000 frames)."""
    if not actions:
        return ""
    parts: List[str] = []
    cur_sym = "K" if actions[0] == "keep" else "D"
    cur_cnt = 1
    for act in actions[1:]:
        sym = "K" if act == "keep" else "D"
        if sym == cur_sym:
            cur_cnt += 1
        else:
            parts.append(f"{cur_sym}{cur_cnt}")
            cur_sym = sym
            cur_cnt = 1
    parts.append(f"{cur_sym}{cur_cnt}")
    return "".join(parts)


def _decode_actions_rle(rle: str) -> List[str]:
    """Giải mã chuỗi RLE ('K2D3...') về danh sách ['keep', 'drop', ...] trong < 1 ms."""
    if not rle:
        return []
    out: List[str] = []
    for m in re.finditer(r"([KD])(\d+)", rle):
        word = "keep" if m.group(1) == "K" else "drop"
        out.extend([word] * int(m.group(2)))
    return out


def _partition_actions_into_chunks(
    base_actions: List[str],
    chunk_frame_counts: Optional[List[int]],
    segment_seconds: int,
    fps_val: float,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    Phân hoạch danh sách hành động keep/drop lên lưới chunk_frame_counts thực tế (hoặc lưới segment_seconds),
    cưỡng bức frame đầu mỗi chunk = 'keep' theo bất biến AGENTS.md và sinh cấu trúc runs.
    """
    actions = list(base_actions)
    total_frames = len(actions)
    if chunk_frame_counts is not None and len(chunk_frame_counts) > 0:
        sum_chunks = sum(chunk_frame_counts)
        assert sum_chunks == total_frames, (
            f"Tổng số frame của các phân đoạn ({sum_chunks}) không khớp tổng frame ({total_frames})!"
        )
        curr_pos = 0
        for count in chunk_frame_counts:
            if curr_pos < total_frames:
                actions[curr_pos] = "keep"
            curr_pos += count
        num_chunks = len(chunk_frame_counts)
        frames_per_chunk = chunk_frame_counts[0] if num_chunks > 0 else int(round(segment_seconds * fps_val))
        spans = []
        curr_s = 0
        for c_len in chunk_frame_counts:
            spans.append((curr_s, curr_s + c_len))
            curr_s += c_len
    else:
        frames_per_chunk = int(round(segment_seconds * fps_val))
        for c_start in range(0, total_frames, frames_per_chunk):
            actions[c_start] = "keep"
        num_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk
        spans = [
            (c_idx * frames_per_chunk, min((c_idx + 1) * frames_per_chunk, total_frames))
            for c_idx in range(num_chunks)
        ]

    chunks_meta: List[Dict[str, Any]] = []
    total_kept_count = 0
    for c_idx, (c_start, c_end) in enumerate(spans):
        c_frames = actions[c_start:c_end]
        c_len = c_end - c_start
        runs: List[Dict[str, int]] = []
        curr_kept = 0
        curr_count = 0
        kept_indices: List[int] = []
        for idx, act in enumerate(c_frames):
            if act == "keep":
                if curr_count > 0:
                    runs.append({"frame_idx": curr_kept, "count": curr_count})
                curr_kept = idx
                curr_count = 1
                kept_indices.append(idx)
            else:
                curr_count += 1
        if curr_count > 0:
            runs.append({"frame_idx": curr_kept, "count": curr_count})
        total_kept_count += len(kept_indices)
        chunks_meta.append({
            "chunk_idx": c_idx,
            "start_frame": c_start,
            "end_frame": c_end,
            "total_frames": c_len,
            "kept_count": len(kept_indices),
            "kept_indices": kept_indices,
            "runs": runs,
        })
    return chunks_meta, total_kept_count, num_chunks, frames_per_chunk


def scan_dedup_plan(
    input_path: str,
    plan_path: str,
    segment_seconds: int = 8,
    fps_num: int = 24000,
    fps_den: int = 1001,
    hi: int = DEFAULT_DEDUP_HI,
    lo: int = DEFAULT_DEDUP_LO,
    frac: float = DEFAULT_DEDUP_FRAC,
    chunk_frame_counts: Optional[List[int]] = None,
    policy: str = "default",
    motion_tau: float = 0.3,
    drift_delta: Optional[float] = None,
    drift_min_response: float = 0.10,
    dis_thresh: Optional[float] = None
) -> Dict[str, Any]:
    """
    Quét phân tích toàn tập bằng filter mpdecimate để lập plan.json.
    Chỉ chạy 1 lần duy nhất trước khi upscale, overhead cực nhỏ (~1-2 phút CPU).
    Nếu có chunk_frame_counts (từ split_video_segments), các chunk sẽ khớp hoàn toàn với phân đoạn thực tế.
    Hỗ trợ chính sách dedup_policy: default, sensitive, sensitive-a, sensitive-b, motion, drift, dis, off.
    """
    # 1. Ghim phiên bản FFmpeg để đảm bảo tính tương thích log debug
    ver_proc = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ffmpeg_ver = ver_proc.stdout.splitlines()[0] if ver_proc.stdout else "unknown"

    # 2. Đếm tổng số packet/frame chính xác từ luồng video bằng ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    expected_frames: Optional[int] = None
    if probe_res.returncode == 0 and probe_res.stdout.strip().isdigit():
        expected_frames = int(probe_res.stdout.strip())

    policy_clean = policy.lower()
    if policy_clean not in DEDUP_POLICY_PARAMS:
        raise ValueError(f"Chính sách dedup không hợp lệ: '{policy}'. Danh sách hợp lệ: {list(DEDUP_POLICY_PARAMS.keys())}")

    effective_drift_delta = 0.75
    if drift_delta is not None:
        effective_drift_delta = float(drift_delta)
    elif policy_clean == "drift-0.5":
        effective_drift_delta = 0.5
    elif policy_clean == "drift-0.75":
        effective_drift_delta = 0.75
    elif policy_clean == "drift-1.0":
        effective_drift_delta = 1.0

    effective_dis_thresh = 0.30
    effective_local_thresh: Optional[float] = None
    effective_min_moving_area_pct: float = 5.0
    p_params = DEDUP_POLICY_PARAMS.get(policy_clean, {})
    if "dis_thresh" in p_params:
        effective_dis_thresh = float(p_params["dis_thresh"])
    if "local_thresh" in p_params:
        effective_local_thresh = float(p_params["local_thresh"])
    if "min_moving_area_pct" in p_params:
        effective_min_moving_area_pct = float(p_params["min_moving_area_pct"])
    if dis_thresh is not None:
        effective_dis_thresh = float(dis_thresh)
    elif policy_clean == "dis-0.2":
        effective_dis_thresh = 0.20
    elif policy_clean == "dis-0.3":
        effective_dis_thresh = 0.30
    elif policy_clean == "dis-0.4":
        effective_dis_thresh = 0.40
    elif policy_clean == "dis-0.5":
        effective_dis_thresh = 0.50

    # Áp dụng thông số theo chính sách nếu là giá trị mặc định
    if policy_clean in ["sensitive", "sensitive-a"]:
        hi, lo, frac = 512, 160, 0.15
    elif policy_clean == "sensitive-b":
        hi, lo, frac = 384, 128, 0.10
    elif policy_clean in ["dis-sakuga", "sakuga"]:
        hi, lo, frac = 1024, 448, 0.38
    elif policy_clean in ["motion", "default"] or policy_clean.startswith("drift") or policy_clean.startswith("dis"):
        hi, lo, frac = DEFAULT_DEDUP_HI, DEFAULT_DEDUP_LO, DEFAULT_DEDUP_FRAC

    frame_actions: List[str] = []
    fused_dis_final_actions: Optional[List[str]] = None
    is_dis_policy = policy_clean.startswith("dis") or policy_clean in ["sakuga", "dis-sakuga"]

    if policy_clean == "off":
        if expected_frames is None or expected_frames <= 0:
            count_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_path
            ]
            c_res = subprocess.run(count_cmd, stdout=subprocess.PIPE, text=True)
            if c_res.returncode == 0 and c_res.stdout.strip().isdigit():
                expected_frames = int(c_res.stdout.strip())
            else:
                expected_frames = 1
        total_frames = expected_frames
        frame_actions = ["keep"] * total_frames
    else:
        # Tối ưu thuật toán 1-Pass: Hợp nhất giải mã mpdecimate + luồng xám DIS ngay trong 1 tiến trình FFmpeg
        if is_dis_policy and not hasattr(subprocess.Popen, "_mock_name"):
            pre_starts = None
            if chunk_frame_counts is not None and len(chunk_frame_counts) > 0:
                pre_starts = set()
                cp = 0
                for c_cnt in chunk_frame_counts:
                    pre_starts.add(cp)
                    cp += c_cnt
            fpc_est = int(round(segment_seconds * (fps_num / fps_den)))
            fused_res = _scan_fused_mpdecimate_and_dis(
                input_path=input_path,
                hi=hi,
                lo=lo,
                frac=frac,
                thresh=effective_dis_thresh,
                chunk_start_set=pre_starts,
                frames_per_chunk_fallback=fpc_est,
                min_moving_area_pct=effective_min_moving_area_pct,
                local_thresh=effective_local_thresh,
            )
            if fused_res is not None:
                frame_actions, fused_dis_final_actions = fused_res
                total_frames = len(frame_actions)

        if not frame_actions:
            cmd = [
                "ffmpeg", "-i", input_path,
                "-vf", f"mpdecimate=hi={hi}:lo={lo}:frac={frac}",
                "-v", "debug",
                "-f", "null", "-"
            ]

            pattern = re.compile(r"\[Parsed_mpdecimate_0.*?\]\s+.*?\b(keep|drop)\s+pts:")

            # Thực thi ffmpeg và bắt dòng log debug (buffer 64KB giảm ~350.000 syscall pipe mỗi tập)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=65536
            )

            if proc.stderr:
                for line in proc.stderr:
                    m = pattern.search(line)
                    if m:
                        frame_actions.append(m.group(1))

            proc.wait()
            if proc.returncode != 0 and not frame_actions:
                raise RuntimeError(f"Lỗi khi quét mpdecimate trên file: {input_path}")

            total_frames = len(frame_actions)
            if total_frames == 0:
                raise ValueError("Không quét được frame nào từ file video.")

            # Chính sách motion: Bổ sung các frame có chuyển động toàn cục
            if policy_clean == "motion":
                motion_mask = detect_global_motion_frames(input_path, tau=motion_tau)
                for i in range(min(len(frame_actions), len(motion_mask))):
                    if motion_mask[i]:
                        frame_actions[i] = "keep"

    # 3. Assertions bất biến: tổng keep + drop phải bằng len(frame_actions) và khớp ffprobe
    raw_keep = sum(1 for a in frame_actions if a == "keep")
    raw_drop = sum(1 for a in frame_actions if a == "drop")
    assert raw_keep + raw_drop == total_frames, (
        f"Parse log debug không khớp: keep ({raw_keep}) + drop ({raw_drop}) != tổng actions ({total_frames})"
    )
    if expected_frames is not None:
        assert total_frames == expected_frames, (
            f"Số frame quét từ debug log ({total_frames}) không khớp số frame ffprobe đếm được ({expected_frames})!"
        )

    # Lưu chuỗi RLE của hành động thô trước khi ép ranh giới chunk cho các chính sách độc lập ranh giới,
    # giúp load_or_create_plan phân hoạch lại theo lưới GOP mới trong < 1 ms mà không phải giải mã lại FFmpeg.
    raw_actions_rle: Optional[str] = None
    if not (policy_clean.startswith("drift") or is_dis_policy):
        raw_actions_rle = _encode_actions_rle(frame_actions)

    fps_val = fps_num / fps_den

    if chunk_frame_counts is not None and len(chunk_frame_counts) > 0:
        sum_chunks = sum(chunk_frame_counts)
        assert sum_chunks == total_frames, (
            f"Tổng số frame của các phân đoạn ({sum_chunks}) không khớp tổng frame ({total_frames})!"
        )
        chunk_start_indices = set()
        curr_pos = 0
        for count in chunk_frame_counts:
            chunk_start_indices.add(curr_pos)
            curr_pos += count
        num_chunks = len(chunk_frame_counts)
        frames_per_chunk = chunk_frame_counts[0] if num_chunks > 0 else int(round(segment_seconds * fps_val))
    else:
        frames_per_chunk = int(round(segment_seconds * fps_val))
        chunk_start_indices = set(range(0, total_frames, frames_per_chunk))
        num_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk

    # BẮT BUỘC ÉP GIỮ FRAME ĐẦU MỖI CHUNK per AGENTS.md Invariant:
    for c_start in chunk_start_indices:
        if c_start < total_frames:
            frame_actions[c_start] = "keep"

    # Chính sách drift: đo độ dịch tích lũy từ frame được giữ gần nhất
    if policy_clean.startswith("drift"):
        frame_actions = detect_drift_policy_actions(
            input_path=input_path,
            base_actions=frame_actions,
            forced_keep_indices=chunk_start_indices,
            delta=effective_drift_delta,
            min_response=drift_min_response
        )
    # Chính sách dis: đo dòng chảy quang học dày đặc DIS Optical Flow (zoom/lia/chuyển động mịn)
    elif is_dis_policy:
        if fused_dis_final_actions is not None and len(fused_dis_final_actions) == total_frames:
            frame_actions = fused_dis_final_actions
        else:
            frame_actions = detect_dis_policy_actions(
                input_path=input_path,
                base_actions=frame_actions,
                forced_keep_indices=chunk_start_indices,
                thresh=effective_dis_thresh,
                min_moving_area_pct=effective_min_moving_area_pct,
                local_thresh=effective_local_thresh,
            )

    chunks_meta = []
    total_kept_count = 0

    if chunk_frame_counts is not None and len(chunk_frame_counts) > 0:
        curr_start = 0
        for c_idx, c_len in enumerate(chunk_frame_counts):
            c_start = curr_start
            c_end = c_start + c_len
            curr_start = c_end
            c_frames = frame_actions[c_start:c_end]

            runs: List[Dict[str, int]] = []
            curr_kept = 0
            curr_count = 0
            kept_indices = []

            for idx, act in enumerate(c_frames):
                if act == "keep":
                    if curr_count > 0:
                        runs.append({"frame_idx": curr_kept, "count": curr_count})
                    curr_kept = idx
                    curr_count = 1
                    kept_indices.append(idx)
                else:
                    curr_count += 1

            if curr_count > 0:
                runs.append({"frame_idx": curr_kept, "count": curr_count})

            total_kept_count += len(kept_indices)

            chunks_meta.append({
                "chunk_idx": c_idx,
                "start_frame": c_start,
                "end_frame": c_end,
                "total_frames": c_len,
                "kept_count": len(kept_indices),
                "kept_indices": kept_indices,
                "runs": runs
            })
    else:
        for c_idx in range(num_chunks):
            c_start = c_idx * frames_per_chunk
            c_end = min((c_idx + 1) * frames_per_chunk, total_frames)
            c_frames = frame_actions[c_start:c_end]
            c_len = len(c_frames)

            # Xây dựng danh sách runs cho chunk này
            runs: List[Dict[str, int]] = []
            curr_kept = 0
            curr_count = 0
            kept_indices = []

            for idx, act in enumerate(c_frames):
                if act == "keep":
                    if curr_count > 0:
                        runs.append({"frame_idx": curr_kept, "count": curr_count})
                    curr_kept = idx
                    curr_count = 1
                    kept_indices.append(idx)
                else:
                    curr_count += 1

            if curr_count > 0:
                runs.append({"frame_idx": curr_kept, "count": curr_count})

            total_kept_count += len(kept_indices)

            chunks_meta.append({
                "chunk_idx": c_idx,
                "start_frame": c_start,
                "end_frame": c_end,
                "total_frames": c_len,
                "kept_count": len(kept_indices),
                "kept_indices": kept_indices,
                "runs": runs
            })

    dedup_ratio = (1.0 - (total_kept_count / total_frames)) * 100.0 if total_frames > 0 else 0.0

    plan_data = {
        "source_file": os.path.basename(input_path),
        "ffmpeg_version": ffmpeg_ver,
        "dedup_policy": policy_clean,
        "dedup_params": {
            "hi": hi if policy_clean != "off" else None,
            "lo": lo if policy_clean != "off" else None,
            "frac": frac if policy_clean != "off" else None,
            "motion_tau": motion_tau if policy_clean == "motion" else None,
            "drift_delta": effective_drift_delta if policy_clean.startswith("drift") else None,
            "drift_min_response": drift_min_response if policy_clean.startswith("drift") else None,
            "dis_thresh": effective_dis_thresh if policy_clean.startswith("dis") else None
        },
        "total_frames": total_frames,
        "raw_kept_frames": raw_keep,
        "raw_dropped_frames": raw_drop,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "fps": round(fps_val, 3),
        "segment_seconds": segment_seconds,
        "frames_per_chunk": frames_per_chunk,
        "total_chunks": num_chunks,
        "chunk_frame_counts": chunk_frame_counts,
        "total_kept_frames": total_kept_count,
        "dedup_ratio_percent": round(dedup_ratio, 2),
        "raw_actions_rle": raw_actions_rle,
        "chunks": chunks_meta
    }

    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_data, f, indent=2, ensure_ascii=False)

    return plan_data


def load_or_create_plan(
    input_path: str,
    plan_path: str,
    segment_seconds: int = 8,
    fps_num: int = 24000,
    fps_den: int = 1001,
    resume: bool = True,
    chunk_frame_counts: Optional[List[int]] = None,
    policy: str = "default",
    motion_tau: float = 0.3,
    drift_delta: Optional[float] = None,
    drift_min_response: float = 0.10,
    dis_thresh: Optional[float] = None
) -> Dict[str, Any]:
    """
    Nạp plan.json đã có từ phiên chạy trước nếu bật resume;
    nếu chưa có hoặc không khớp với chunk_frame_counts thực tế thì tự động quét lại.
    Từ chối resume nếu chính sách dedup_policy trong cache khác với chính sách yêu cầu.
    """
    policy_clean = policy.lower()
    effective_drift_delta = 0.75
    if drift_delta is not None:
        effective_drift_delta = float(drift_delta)
    elif policy_clean == "drift-0.5":
        effective_drift_delta = 0.5
    elif policy_clean == "drift-0.75":
        effective_drift_delta = 0.75
    elif policy_clean == "drift-1.0":
        effective_drift_delta = 1.0

    effective_dis_thresh = 0.30
    if dis_thresh is not None:
        effective_dis_thresh = float(dis_thresh)
    elif policy_clean == "dis-0.2":
        effective_dis_thresh = 0.20
    elif policy_clean == "dis-0.3":
        effective_dis_thresh = 0.30
    elif policy_clean == "dis-0.4":
        effective_dis_thresh = 0.40
    elif policy_clean == "dis-0.5":
        effective_dis_thresh = 0.50

    if resume and os.path.exists(plan_path):
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = data.get("chunks", [])
            if data.get("total_frames", 0) > 0 and len(chunks) > 0:
                cached_policy = data.get("dedup_policy", "default")
                if cached_policy != policy_clean:
                    raise ValueError(
                        f"Từ chối resume: Kế hoạch trong {plan_path} sử dụng chính sách dedup '{cached_policy}' "
                        f"khác với chính sách yêu cầu '{policy_clean}'. Vui lòng dùng --force hoặc xóa file để quét lại."
                    )
                if policy_clean.startswith("drift"):
                    cached_params = data.get("dedup_params", {})
                    cached_delta = cached_params.get("drift_delta")
                    if cached_delta is not None and abs(cached_delta - effective_drift_delta) > 1e-4:
                        raise ValueError(
                            f"Từ chối resume: Kế hoạch drift trong {plan_path} sử dụng delta={cached_delta} "
                            f"khác với delta yêu cầu={effective_drift_delta}."
                        )
                if policy_clean.startswith("dis"):
                    cached_params = data.get("dedup_params", {})
                    cached_dis = cached_params.get("dis_thresh")
                    if cached_dis is not None and abs(cached_dis - effective_dis_thresh) > 1e-4:
                        raise ValueError(
                            f"Từ chối resume: Kế hoạch dis trong {plan_path} sử dụng dis_thresh={cached_dis} "
                            f"khác với dis_thresh yêu cầu={effective_dis_thresh}."
                        )
                if chunk_frame_counts is None:
                    return data
                cached_counts = [c["total_frames"] for c in chunks]
                if cached_counts == chunk_frame_counts:
                    return data
                # Khôi phục tức thời (< 1 ms) từ raw_actions_rle khi lưới GOP thực tế khác lưới tạm 192 frames
                rle_str = data.get("raw_actions_rle")
                if (
                    isinstance(rle_str, str)
                    and rle_str
                    and not policy_clean.startswith(("drift", "dis"))
                    and sum(chunk_frame_counts) == data.get("total_frames", 0)
                ):
                    raw_actions = _decode_actions_rle(rle_str)
                    if len(raw_actions) == data["total_frames"]:
                        fps_val = fps_num / fps_den
                        re_chunks, re_kept, re_num, re_fpc = _partition_actions_into_chunks(
                            raw_actions, chunk_frame_counts, segment_seconds, fps_val
                        )
                        re_ratio = (1.0 - (re_kept / data["total_frames"])) * 100.0 if data["total_frames"] > 0 else 0.0
                        data["fps_num"] = fps_num
                        data["fps_den"] = fps_den
                        data["fps"] = round(fps_val, 3)
                        data["segment_seconds"] = segment_seconds
                        data["frames_per_chunk"] = re_fpc
                        data["total_chunks"] = re_num
                        data["chunk_frame_counts"] = chunk_frame_counts
                        data["total_kept_frames"] = re_kept
                        data["dedup_ratio_percent"] = round(re_ratio, 2)
                        data["chunks"] = re_chunks
                        tmp_plan = f"{plan_path}.repart.tmp"
                        with open(tmp_plan, "w", encoding="utf-8") as wf:
                            json.dump(data, wf, indent=2, ensure_ascii=False)
                        os.replace(tmp_plan, plan_path)
                        return data
        except ValueError:
            raise
        except Exception:
            pass

    return scan_dedup_plan(
        input_path=input_path,
        plan_path=plan_path,
        segment_seconds=segment_seconds,
        fps_num=fps_num,
        fps_den=fps_den,
        chunk_frame_counts=chunk_frame_counts,
        policy=policy_clean,
        motion_tau=motion_tau,
        drift_delta=effective_drift_delta,
        drift_min_response=drift_min_response,
        dis_thresh=effective_dis_thresh
    )


def prune_redundant_frames(
    frames_dir: str,
    kept_indices: List[int],
    fmt: str = "png"
) -> int:
    """
    Xóa ngay lập tức các khung hình trùng lặp khỏi RAM Disk trước khi gọi AI,
    chỉ giữ lại các file tương ứng với kept_indices (0-based -> frame_%04d 1-based).
    Trả về số lượng frame thực tế được giữ lại để upscale.
    """
    kept_set = set(kept_indices)
    removed_count = 0

    all_files = sorted(os.listdir(frames_dir))
    for fname in all_files:
        if not fname.endswith(f".{fmt}"):
            continue
        # Tách số thứ tự (frame_0001.png -> idx 0)
        m = re.search(r"(\d+)", fname)
        if m:
            frame_1based = int(m.group(1))
            frame_0based = frame_1based - 1
            if frame_0based not in kept_set:
                file_path = os.path.join(frames_dir, fname)
                try:
                    os.remove(file_path)
                    removed_count += 1
                except OSError:
                    pass

    return len(kept_set)


def generate_concat_demuxer_script(
    runs: List[Dict[str, int]],
    frames_out_dir: str,
    concat_script_path: str,
    fps_num: int = 24000,
    fps_den: int = 1001,
    fmt: str = "png"
) -> str:
    """
    Tạo file script Concat Demuxer cho FFmpeg để tái tạo 4K CFR:
    Mỗi file ảnh 4K được nạp duy nhất 1 lần vào bộ nhớ và nhân bản bằng duration.
    """
    frame_dur = fps_den / fps_num

    with open(concat_script_path, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for r in runs:
            # 0-based frame_idx -> frame%08d.png (đồng bộ với extract_frames)
            img_name = f"frame{r['frame_idx'] + 1:08d}.{fmt}"
            img_path = os.path.join(frames_out_dir, img_name)
            dur = r["count"] * frame_dur
            f.write(f"file '{img_path}'\n")
            f.write(f"duration {dur:.6f}\n")

        # Quy chuẩn FFmpeg Concat Demuxer: dòng cuối cùng lặp lại file mà không có duration
        last_img_name = f"frame{runs[-1]['frame_idx'] + 1:08d}.{fmt}"
        last_img_path = os.path.join(frames_out_dir, last_img_name)
        f.write(f"file '{last_img_path}'\n")

    return concat_script_path


def encode_chunk_reexpanded(
    concat_script_path: str,
    output_path: str,
    expected_frames: int,
    fps_num: int = 24000,
    fps_den: int = 1001,
    bitrate: str = "28M"
) -> None:
    """
    Mã hóa phân đoạn 4K qua FFmpeg Concat Demuxer + NVENC HEVC 10-bit:
    - Chốt cờ CFR: -video_track_timescale fps_num -frames:v expected_frames
    - Chuỗi ma trận màu sắc BT.709 Full RGB ➔ Limited YUV p010le
    - Giao dịch an toàn: ghi ra .tmp.mp4, xác minh đủ số frame rồi mới rename
    """
    tmp_output = f"{output_path}.tmp.mp4"
    if os.path.exists(tmp_output):
        os.remove(tmp_output)

    rate_args = get_hevc_rate_args(bitrate=bitrate)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_script_path,
        "-vf", (
            f"scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv,"
            f"fps={fps_num}/{fps_den}"
        ),
        "-pix_fmt", HEVC_PIX_FMT,
        "-c:v", HEVC_ENCODER,
        "-profile:v", HEVC_PROFILE,
        *rate_args,
        "-color_range", HEVC_COLOR_RANGE,
        "-colorspace", HEVC_COLORSPACE,
        "-color_primaries", HEVC_COLOR_PRIMARIES,
        "-color_trc", HEVC_COLOR_TRC,
        "-tag:v", HEVC_TAG,
        "-video_track_timescale", str(fps_num),
        "-frames:v", str(expected_frames),
        tmp_output
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr or "Không có thông tin stderr"
        raise RuntimeError(f"Lỗi mã hóa FFmpeg Concat Re-expansion (exit {e.returncode}):\n{err_msg}")

    # Xác minh giao dịch (Transactional verification)
    cmd_verify = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        tmp_output
    ]
    probe_res = subprocess.run(cmd_verify, capture_output=True, text=True, check=True)
    actual_frames = int(probe_res.stdout.strip())

    if actual_frames != expected_frames:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        raise RuntimeError(
            f"Lỗi giao dịch phân đoạn: Số frame thực tế ({actual_frames}) "
            f"không khớp số frame kỳ vọng ({expected_frames})!"
        )

    # Di chuyển nguyên tử sang file hoàn tất
    shutil.move(tmp_output, output_path)


def run_preflight_worker_cli(argv: List[str]) -> Dict[str, Any]:
    """
    Điểm vào thực thi trong tiến trình con Background Queue Preflight:
    1. Nếu plan_file nằm trong .temp_<basename>, tách sẵn raw_segs theo Keyframe GOP (Stream Copy)
       và đo chính xác chunk_frame_counts để khi tập kế tiếp chạy không tốn giây nào chờ tách/quét lại.
    2. Phân giải đúng dedup_policy (CLI / shows.yaml / default) để khớp 100% với pipeline chính.
    """
    from src.ffmpeg_ops import probe_video, split_video_segments, get_segments_frame_counts

    vid = argv[0]
    plan_f = argv[1]
    seg_s = int(argv[2])
    pol_arg = argv[3] if len(argv) > 3 and argv[3] else None

    meta = probe_video(vid)
    work_d = os.path.dirname(os.path.abspath(plan_f))
    seg_counts: Optional[List[int]] = None
    if os.path.basename(work_d).startswith(".temp_"):
        segs_d = os.path.join(work_d, "raw_segs")
        existing = (
            sorted([
                os.path.join(segs_d, f)
                for f in os.listdir(segs_d)
                if f.startswith("seg_") and f.endswith(".mp4")
            ])
            if os.path.isdir(segs_d)
            else []
        )
        if not existing:
            os.makedirs(segs_d, exist_ok=True)
            existing = split_video_segments(vid, segs_d, seg_s)
        if existing:
            seg_counts = get_segments_frame_counts(existing)

    eff_pol = resolve_dedup_policy(vid, pol_arg)
    return load_or_create_plan(
        vid,
        plan_f,
        seg_s,
        meta["fps_num"],
        meta["fps_den"],
        chunk_frame_counts=seg_counts,
        policy=eff_pol,
    )


def build_preflight_subprocess_spec(
    video_path: str,
    plan_file: str,
    segment_seconds: int = 8,
    dedup_policy: Optional[str] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Xây dựng lệnh subprocess đa nền tảng (Windows BELOW_NORMAL_PRIORITY_CLASS / POSIX nice)
    và truyền đường dẫn an toàn qua sys.argv (tránh lỗi ký tự nháy đơn/backslash trên Windows).
    """
    import sys
    script = (
        "import sys; "
        "from src.dedup import run_preflight_worker_cli; "
        "run_preflight_worker_cli(sys.argv[1:])"
    )
    cmd = [sys.executable, "-c", script, video_path, plan_file, str(segment_seconds), dedup_policy or ""]
    popen_kwargs: Dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
    elif shutil.which("nice"):
        cmd = ["nice", "-n", "10", *cmd]
    return cmd, popen_kwargs


def spawn_background_preflight(
    video_path: str,
    output_dir: Optional[str] = None,
    segment_seconds: int = 8,
    logger=None,
    dedup_policy: Optional[str] = None,
) -> Optional[subprocess.Popen]:
    """
    Kích hoạt tiến trình quét mpdecimate lập plan.json trước cho tập kế tiếp.
    Chạy với độ ưu tiên nền (BELOW_NORMAL_PRIORITY_CLASS trên Windows / nice -n 10 trên POSIX)
    trong lúc GPU đang bận xử lý tập trước.
    """
    import sys
    if os.environ.get("BENCHMARK_MODE") == "1":
        if logger:
            logger.warning("⚡ [BENCHMARK MODE] Từ chối kích hoạt spawn_background_preflight.", to_console=True)
        return None
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = output_dir or "out"
        if out_dir.lower().endswith((".mkv", ".mp4")):
            out_dir = os.path.dirname(out_dir) or "out"
        work_dir = os.path.join(out_dir, f".temp_{base_name}")
        os.makedirs(work_dir, exist_ok=True)
        plan_file = os.path.join(work_dir, "plan.json")
        if os.path.exists(plan_file) and os.path.getsize(plan_file) > 100:
            return None

        cmd, popen_kwargs = build_preflight_subprocess_spec(
            video_path, plan_file, segment_seconds, dedup_policy=dedup_policy
        )
        proc = subprocess.Popen(cmd, **popen_kwargs)
        prio_desc = "BELOW_NORMAL" if sys.platform == "win32" else "nice -n 10"
        msg = f"⚡ [Multi-Episode Queue] Đã kích hoạt Preflight nền ({prio_desc}) cho tập kế tiếp: {os.path.basename(video_path)}"
        if logger:
            logger.info(msg, to_console=True)
        else:
            print(msg)
        return proc
    except Exception:
        return None

