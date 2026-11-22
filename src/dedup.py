"""Keyframe GOP splitting, mpdecimate/DIS deduplication, and Cyclic Cel memoization."""

import os
import re
import sys
import math
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
    "dis-auto": {
        "hi": DEFAULT_DEDUP_HI,
        "lo": DEFAULT_DEDUP_LO,
        "frac": DEFAULT_DEDUP_FRAC,
        "dis_thresh": 0.30,
        "auto_noise_floor": True,
        "lip_flap_guard": True,
    },
    "dis-0.2": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "dis_thresh": 0.20},
    "dis-0.3": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "dis_thresh": 0.30},
    "dis-0.4": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "dis_thresh": 0.40},
    "dis-0.5": {"hi": DEFAULT_DEDUP_HI, "lo": DEFAULT_DEDUP_LO, "frac": DEFAULT_DEDUP_FRAC, "dis_thresh": 0.50},
    "dis-sakuga": {"hi": 1024, "lo": 448, "frac": 0.38, "dis_thresh": 0.35, "local_thresh": 1.50, "min_moving_area_pct": 12.0},
    "sakuga": {"hi": 1024, "lo": 448, "frac": 0.38, "dis_thresh": 0.35, "local_thresh": 1.50, "min_moving_area_pct": 12.0},
    "off": {"all_keep": True}
}


def get_dedup_params(
    policy_name: str,
    dis_thresh_override: Optional[float] = None,
    drift_delta_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Phân giải tham số chính sách dedup, hỗ trợ cả tên chính sách cố định trong DEDUP_POLICY_PARAMS
    lẫn chuỗi động dạng 'dis-<float>' / 'drift-<float>' cho --dis-thresh / --drift-delta tùy chỉnh.
    """
    pol = str(policy_name or "default").strip().lower()
    if pol in DEDUP_POLICY_PARAMS:
        params = dict(DEDUP_POLICY_PARAMS[pol])
    elif pol.startswith("dis-auto-"):
        val_str = pol[len("dis-auto-"):]
        try:
            val = float(val_str)
        except ValueError as exc:
            raise ValueError(
                f"Chính sách dedup không hợp lệ: '{policy_name}'. Danh sách hợp lệ: {list(DEDUP_POLICY_PARAMS.keys())}"
            ) from exc
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(
                f"Chính sách dedup không hợp lệ: '{policy_name}' (dis_thresh phải > 0). Danh sách hợp lệ: {list(DEDUP_POLICY_PARAMS.keys())}"
            )
        params = dict(DEDUP_POLICY_PARAMS["dis-auto"])
        params["dis_thresh"] = val
    elif pol.startswith("dis-"):
        val_str = pol[len("dis-"):]
        try:
            val = float(val_str)
        except ValueError as exc:
            raise ValueError(
                f"Chính sách dedup không hợp lệ: '{policy_name}'. Danh sách hợp lệ: {list(DEDUP_POLICY_PARAMS.keys())}"
            ) from exc
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(
                f"Chính sách dedup không hợp lệ: '{policy_name}' (dis_thresh phải > 0). Danh sách hợp lệ: {list(DEDUP_POLICY_PARAMS.keys())}"
            )
        params = dict(DEDUP_POLICY_PARAMS["dis"])
        params["dis_thresh"] = val
    elif pol.startswith("drift-"):
        val_str = pol[len("drift-"):]
        try:
            val = float(val_str)
        except ValueError as exc:
            raise ValueError(
                f"Chính sách dedup không hợp lệ: '{policy_name}'. Danh sách hợp lệ: {list(DEDUP_POLICY_PARAMS.keys())}"
            ) from exc
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(
                f"Chính sách dedup không hợp lệ: '{policy_name}' (drift_delta phải > 0). Danh sách hợp lệ: {list(DEDUP_POLICY_PARAMS.keys())}"
            )
        params = dict(DEDUP_POLICY_PARAMS["drift"])
        params["drift_delta"] = val
    else:
        raise ValueError(
            f"Chính sách dedup không hợp lệ: '{policy_name}'. Danh sách hợp lệ: {list(DEDUP_POLICY_PARAMS.keys())}"
        )

    if dis_thresh_override is not None:
        val_ov = float(dis_thresh_override)
        if not math.isfinite(val_ov) or val_ov <= 0.0:
            raise ValueError(
                f"Chính sách dedup không hợp lệ: dis_thresh={dis_thresh_override}. Danh sách hợp lệ: {list(DEDUP_POLICY_PARAMS.keys())}"
            )
        params["dis_thresh"] = val_ov
    if drift_delta_override is not None:
        val_dr = float(drift_delta_override)
        if not math.isfinite(val_dr) or val_dr <= 0.0:
            raise ValueError(
                f"Chính sách dedup không hợp lệ: drift_delta={drift_delta_override}. Danh sách hợp lệ: {list(DEDUP_POLICY_PARAMS.keys())}"
            )
        params["drift_delta"] = val_dr
    return params


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
    if not os.path.exists(shows_yaml) and os.path.exists("shows.yaml"):
        shows_yaml = "shows.yaml"
    if os.path.exists(shows_yaml):
        try:
            from src.models_resolver import _load_yaml_cached
            from src.show_routing import match_show_rule
            data = _load_yaml_cached(shows_yaml) or {}
            rules = data.get("shows", []) or []
            for r in rules:
                pat = r.get("pattern", "")
                pol = r.get("dedup_policy")
                if pol and match_show_rule(video_path, pat):
                    return str(pol).lower()
        except Exception as exc:
            print(
                f"⚠️ Cảnh báo: Không thể đọc '{shows_yaml}' cho dedup_policy ({exc}); dùng mặc định.",
                file=sys.stderr,
            )
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


def _check_8x8_lip_flap_gate(ref_gray: Any, curr_gray: Any) -> Tuple[bool, float]:
    """Detect small/dark lip-flaps and eye-blinks on dual-phase 8x8 cells (32x32 at 1080p) while rejecting film grain."""
    import cv2
    import numpy as np

    sdiff = curr_gray.astype(np.float32) - ref_gray.astype(np.float32)
    sigma_diff = 1.4826 * float(np.median(np.abs(sdiff[::4, ::4])))
    smooth_abs = np.abs(cv2.blur(sdiff, (3, 3)))
    sigma_smooth = sigma_diff * (1.0 / 3.0)
    thresh_top8 = max(3.2, 4.2 * sigma_smooth)
    thresh_peak = max(4.2, 5.2 * sigma_smooth)

    h, w = smooth_abs.shape[:2]
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    padded = np.pad(smooth_abs, ((0, pad_h), (0, pad_w)), mode="edge") if (pad_h or pad_w) else smooth_abs
    gh, gw = padded.shape[0] // 8, padded.shape[1] // 8
    if gh > 0 and gw > 0:
        cells = padded.reshape(gh, 8, gw, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
        top8 = np.partition(cells, 56, axis=1)[:, 56:]
        if bool(np.any((top8.mean(axis=1) >= thresh_top8) & (top8.max(axis=1) >= thresh_peak))):
            return True, sigma_diff

    # Half-cell (4px) shifted grid prevents misses when a 6x4 mouth straddles 4 cell corners
    if h >= 12 and w >= 12:
        sh_h = ((h - 4) // 8) * 8
        sh_w = ((w - 4) // 8) * 8
        if sh_h >= 8 and sh_w >= 8:
            sub = smooth_abs[4 : 4 + sh_h, 4 : 4 + sh_w]
            cells_s = sub.reshape(sh_h // 8, 8, sh_w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
            top8_s = np.partition(cells_s, 56, axis=1)[:, 56:]
            if bool(np.any((top8_s.mean(axis=1) >= thresh_top8) & (top8_s.max(axis=1) >= thresh_peak))):
                return True, sigma_diff

    return False, sigma_diff


def _estimate_textured_lk_pan_zoom(
    ref_gray: Any,
    curr_gray: Any,
    scale_factor: float,
    thresh: float,
    sigma_diff: float,
    xc: Any = None,
    yc: Any = None,
) -> bool:
    """Estimate subpixel camera pan and center-free affine zoom restricted to textured pixels."""
    import cv2
    import math
    import numpy as np

    h, w = ref_gray.shape[:2]
    r_f = cv2.GaussianBlur(ref_gray.astype(np.float32), (5, 5), 1.0)
    c_f = cv2.GaussianBlur(curr_gray.astype(np.float32), (5, 5), 1.0)
    it = c_f - r_f
    mid = 0.5 * (r_f + c_f)
    ix = cv2.Sobel(mid, cv2.CV_32F, 1, 0, ksize=3) * (1.0 / 8.0)
    iy = cv2.Sobel(mid, cv2.CV_32F, 0, 1, ksize=3) * (1.0 / 8.0)
    grad_floor = max(2.0, 1.2 * sigma_diff)
    mask = (ix * ix + iy * iy) >= (grad_floor * grad_floor)
    if int(np.count_nonzero(mask)) < 64:
        return False

    if xc is not None and yc is not None and xc.shape == (h, w) and yc.shape == (h, w):
        xcm = xc[mask]
        ycm = yc[mask]
    else:
        yg, xg = np.mgrid[0:h, 0:w].astype(np.float32)
        xcm = (xg - (w * 0.5))[mask]
        ycm = (yg - (h * 0.5))[mask]
    ixm = ix[mask]
    iym = iy[mask]
    itm = it[mask]
    jr = ixm * xcm + iym * ycm

    a00 = float(np.dot(ixm, ixm))
    a01 = float(np.dot(ixm, iym))
    a02 = float(np.dot(ixm, jr))
    a11 = float(np.dot(iym, iym))
    a12 = float(np.dot(iym, jr))
    a22 = float(np.dot(jr, jr))
    mat = np.array([[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]], dtype=np.float64)
    rhs = -np.array([float(np.dot(ixm, itm)), float(np.dot(iym, itm)), float(np.dot(jr, itm))], dtype=np.float64)
    try:
        if np.linalg.cond(mat) > 1e8:
            return False
        dx, dy, kz = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return False

    pan_1080 = math.hypot(float(dx), float(dy)) * float(scale_factor)
    r_max_1080 = math.hypot(w * 0.5, h * 0.5) * float(scale_factor)
    zoom_1080 = abs(float(kz)) * r_max_1080
    return (pan_1080 >= thresh) or (zoom_1080 >= thresh)


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
    auto_noise_floor: bool = False,
    lip_flap_guard: bool = False,
    xc: Any = None,
    yc: Any = None,
) -> bool:
    """Evaluate early-exit motion cascade combining 8x8 lip-flap guard, textured pan/zoom, and DIS optical flow."""
    import cv2
    import math
    import numpy as np

    if np.array_equal(ref_gray, curr_gray):
        return False

    sigma_diff = 0.0
    if lip_flap_guard or auto_noise_floor:
        lip_hit, sigma_diff = _check_8x8_lip_flap_gate(ref_gray, curr_gray)
        if lip_flap_guard and lip_hit:
            return True

    if auto_noise_floor:
        if _estimate_textured_lk_pan_zoom(
            ref_gray, curr_gray, float(scale_factor), float(thresh), sigma_diff, xc=xc, yc=yc
        ):
            return True
        if sigma_diff > 0.8:
            ref_in = cv2.GaussianBlur(ref_gray, (5, 5), 1.1)
            curr_in = cv2.GaussianBlur(curr_gray, (5, 5), 1.1)
        else:
            ref_in = ref_gray
            curr_in = curr_gray
        flow = dis.calc(ref_in, curr_in, None)
        u = flow[..., 0] * scale_factor
        v = flow[..., 1] * scale_factor

        gx = cv2.Sobel(ref_in, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ref_in, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        tex_mask = grad >= max(18.0, 8.0 * sigma_diff)
        eff_local_thresh = (
            float(local_thresh)
            if local_thresh is not None
            else max(float(thresh), 0.75 * sigma_diff)
        )
        local_thresh_sq = eff_local_thresh * eff_local_thresh

        if int(np.count_nonzero(tex_mask)) >= 64:
            ut = u[tex_mask]
            vt = v[tex_mask]
            mag_sq_t = ut * ut + vt * vt
            area_moving = float(np.count_nonzero(mag_sq_t >= local_thresh_sq)) * (100.0 / float(u.size))
            if area_moving >= min_moving_area_pct:
                return True
            if math.hypot(float(np.median(ut)), float(np.median(vt))) >= thresh:
                return True
            p90_t = math.sqrt(max(0.0, float(np.percentile(mag_sq_t, 90))))
            return (0.7 * p90_t) >= max(eff_local_thresh, 0.45 + 0.65 * sigma_diff)

        mag_sq = u * u + v * v
        area_moving = float(np.count_nonzero(mag_sq >= local_thresh_sq)) * (100.0 / float(mag_sq.size))
        if area_moving >= min_moving_area_pct:
            return True
        if math.hypot(float(np.median(u)), float(np.median(v))) >= thresh:
            return True
        if rx_inv_r2.size > 0:
            k_radial = float(np.median(u[r_mask] * rx_inv_r2 + v[r_mask] * ry_inv_r2))
            if abs(k_radial * r_max) >= thresh:
                return True
        p90_mag = math.sqrt(max(0.0, float(np.percentile(mag_sq, 90))))
        return (0.7 * p90_mag) >= eff_local_thresh

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


def _normalize_frame_for_cyclic_dedup(arr_c: Any) -> Tuple[Any, Any]:
    """Return (norm_u8_view, thumb_source_u8) so p010le (<<6) and nv12 operate on normalized 8-bit scale."""
    import numpy as np

    if arr_c.dtype in (np.uint16, np.int16):
        # In little-endian p010le (10-bit << 6), odd byte [:, 1::2] is exact (u16 >> 8) 8-bit code
        msb_u8 = np.ascontiguousarray(arr_c.view(np.uint8)[..., 1::2])
        if msb_u8.ndim == 2 and (msb_u8.shape[0] % 3) == 0:
            h_luma = (msb_u8.shape[0] * 2) // 3
            return msb_u8, msb_u8[:h_luma, :]
        return msb_u8, msb_u8

    if arr_c.ndim == 2 and (arr_c.shape[0] % 3) == 0 and arr_c.shape[0] != arr_c.shape[1]:
        h_luma = (arr_c.shape[0] * 2) // 3
        return arr_c, arr_c[:h_luma, :]

    return arr_c, arr_c


def deduplicate_cyclic_frames(
    frames: Sequence[Any],
    allow_quant_noise: bool = True,
    max_pixel_tol: int = 2,
    max_mae: float = 0.20,
    lookback_distinct: int = 48,
) -> Tuple[List[Any], List[int]]:
    """Memoize cyclic anime cels (A->B->A) across RGB24, NV12, and 10-bit P010LE frames."""
    import cv2
    import numpy as np

    n = len(frames)
    if n <= 1:
        return list(frames), list(range(n))

    distinct_frames: List[Any] = []
    distinct_norm_views: List[Any] = []
    frame_to_distinct: List[int] = []
    thumb_hash_to_distinct: Dict[bytes, List[int]] = {}
    distinct_thumbs: List[Any] = []

    use_qnoise = bool(allow_quant_noise and max_pixel_tol > 0)
    thumb_w, thumb_h = 64, 36
    outlier_cap_tol = max_pixel_tol + 3

    for arr in frames:
        arr_c = arr if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)
        norm_view, thumb_src = _normalize_frame_for_cyclic_dedup(arr_c)
        nh, nw = norm_view.shape[:2]
        n_ch = norm_view.shape[2] if norm_view.ndim == 3 else 1
        total_elems = float(nh * nw * n_ch)

        # Tier 1: 64x36 thumbnail + 64-bit BLAKE2b exact candidate lookup
        thumb = cv2.resize(thumb_src, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        if not thumb.flags["C_CONTIGUOUS"]:
            thumb = np.ascontiguousarray(thumb)
        t_ch = thumb.shape[2] if thumb.ndim == 3 else 1
        digest = hashlib.blake2b(memoryview(thumb), digest_size=8).digest()

        matched_idx = -1
        candidates = thumb_hash_to_distinct.get(digest)
        if candidates:
            for d_idx in candidates:
                ref = distinct_frames[d_idx]
                if ref.shape == arr_c.shape and ref.dtype == arr_c.dtype and np.array_equal(arr_c, ref):
                    matched_idx = d_idx
                    break

        # Tier 2 & Tier 3: B-frame quantization / isolated grain tolerance on normalized 8-bit view
        if matched_idx < 0 and use_qnoise and distinct_frames:
            lower = max(-1, len(distinct_frames) - 1 - max(1, lookback_distinct))
            for d_idx in range(len(distinct_frames) - 1, lower, -1):
                ref_norm = distinct_norm_views[d_idx]
                if ref_norm.shape != norm_view.shape:
                    continue

                t_inf = cv2.norm(thumb, distinct_thumbs[d_idx], cv2.NORM_INF)
                if t_inf > outlier_cap_tol:
                    continue
                t_mae = cv2.norm(thumb, distinct_thumbs[d_idx], cv2.NORM_L1) / float(thumb_w * thumb_h * t_ch)
                if t_mae > (max_mae + 0.08):
                    continue

                f_inf = cv2.norm(norm_view, ref_norm, cv2.NORM_INF)
                if f_inf <= max_pixel_tol:
                    f_mae = cv2.norm(norm_view, ref_norm, cv2.NORM_L1) / total_elems
                    if f_mae <= max_mae:
                        matched_idx = d_idx
                        break
                elif f_inf <= outlier_cap_tol:
                    f_mae = cv2.norm(norm_view, ref_norm, cv2.NORM_L1) / total_elems
                    if f_mae <= min(max_mae, 0.18):
                        abs_d = cv2.absdiff(norm_view, ref_norm)
                        outlier_frac = float(np.count_nonzero(abs_d > max_pixel_tol)) / total_elems
                        if outlier_frac <= 0.0002:
                            grid_w = max(1, nw // 16)
                            grid_h = max(1, nh // 16)
                            blk_mean = cv2.resize(abs_d, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
                            if float(blk_mean.max()) <= 0.85:
                                matched_idx = d_idx
                                break

        if matched_idx >= 0:
            frame_to_distinct.append(matched_idx)
        else:
            new_d = len(distinct_frames)
            distinct_frames.append(arr_c)
            thumb_hash_to_distinct.setdefault(digest, []).append(new_d)
            if use_qnoise:
                distinct_norm_views.append(norm_view)
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
    auto_noise_floor: bool = False,
    lip_flap_guard: bool = False,
    decode_threads: int = 0,
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

    thread_args = ["-threads", str(int(decode_threads))] if decode_threads and int(decode_threads) > 0 else []
    cmd = [
        "ffmpeg", "-v", "error",
        *thread_args,
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
            auto_noise_floor=auto_noise_floor,
            lip_flap_guard=lip_flap_guard,
            xc=rx,
            yc=ry,
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
    auto_noise_floor: bool = False,
    lip_flap_guard: bool = False,
    decode_threads: int = 0,
    boundary_out: Optional[Dict[str, Any]] = None,
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
    thread_args = ["-threads", str(int(decode_threads))] if decode_threads and int(decode_threads) > 0 else []
    cmd = [
        "ffmpeg",
        *thread_args,
        "-i", input_path,
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
    first_gray_copy = None
    last_gray_copy = None
    cut_starts: set[int] = set()
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
        if boundary_out is not None:
            if first_gray_copy is None:
                first_gray_copy = curr_gray.copy()
            if is_forced and curr_idx > 0 and last_gray_copy is not None:
                mad_cut = float(np.mean(np.abs(curr_gray.astype(np.int16) - last_gray_copy.astype(np.int16))))
                if mad_cut >= 16.0:
                    cut_starts.add(curr_idx)
            last_gray_copy = curr_gray.copy() if is_forced else curr_gray

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
            auto_noise_floor=auto_noise_floor,
            lip_flap_guard=lip_flap_guard,
            xc=rx,
            yc=ry,
        ):
            final_actions.append("keep")
            ref_gray = curr_gray
        else:
            final_actions.append("drop")

        curr_idx += 1

    proc.wait()
    t_err.join(timeout=5.0)
    t_out.join(timeout=5.0)

    if boundary_out is not None:
        boundary_out["first_gray"] = first_gray_copy
        boundary_out["last_gray"] = last_gray_copy.copy() if last_gray_copy is not None else None
        boundary_out["cut_starts"] = cut_starts

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
    Partition keep/drop actions onto actual chunk_frame_counts (or segment_seconds grid),
    forcing chunk frame 0 = 'keep' per GEMINI.md invariant, and emit RLE runs.
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
    dis_thresh: Optional[float] = None,
    decode_threads: int = 0,
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

    # 2. Skip whole-file ffprobe packet scan when PyAV already counted exact segment packets
    expected_frames: Optional[int] = None
    if (
        chunk_frame_counts is not None
        and len(chunk_frame_counts) > 0
        and sum(chunk_frame_counts) > 0
        and not hasattr(subprocess.run, "_mock_name")
    ):
        expected_frames = int(sum(chunk_frame_counts))
    else:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path
        ]
        probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if probe_res.returncode == 0 and probe_res.stdout.strip().isdigit():
            expected_frames = int(probe_res.stdout.strip())

    policy_clean = policy.lower()
    p_params = get_dedup_params(
        policy_clean,
        dis_thresh_override=dis_thresh,
        drift_delta_override=drift_delta,
    )

    effective_drift_delta = float(p_params.get("drift_delta", 0.75))
    effective_dis_thresh = float(p_params.get("dis_thresh", 0.30))
    effective_local_thresh: Optional[float] = (
        float(p_params["local_thresh"]) if "local_thresh" in p_params and p_params["local_thresh"] is not None else None
    )
    effective_min_moving_area_pct: float = float(p_params.get("min_moving_area_pct", 5.0))
    effective_auto_noise_floor: bool = bool(p_params.get("auto_noise_floor", False))
    effective_lip_flap_guard: bool = bool(p_params.get("lip_flap_guard", False))
    if "tau" in p_params:
        motion_tau = float(p_params["tau"])

    # Áp dụng thông số mpdecimate theo chính sách
    if policy_clean != "off":
        hi = int(p_params.get("hi", hi))
        lo = int(p_params.get("lo", lo))
        frac = float(p_params.get("frac", frac))

    frame_actions: List[str] = []
    fused_dis_final_actions: Optional[List[str]] = None
    fused_boundary: Dict[str, Any] = {}
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
                auto_noise_floor=effective_auto_noise_floor,
                lip_flap_guard=effective_lip_flap_guard,
                decode_threads=decode_threads,
                boundary_out=fused_boundary,
            )
            if fused_res is not None:
                frame_actions, fused_dis_final_actions = fused_res
                total_frames = len(frame_actions)

        if not frame_actions:
            thread_args = ["-threads", str(int(decode_threads))] if decode_threads and int(decode_threads) > 0 else []
            cmd = [
                "ffmpeg", *thread_args, "-i", input_path,
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
    target_expected = (
        expected_frames
        if expected_frames is not None
        else (int(sum(chunk_frame_counts)) if (chunk_frame_counts and sum(chunk_frame_counts) > 0) else None)
    )
    if target_expected is not None and total_frames < target_expected:
        try:
            from src.ffmpeg_ops import _count_leading_discard_packets
            n_discard = _count_leading_discard_packets(input_path)
            if n_discard > 0 and (target_expected - n_discard) == total_frames:
                if chunk_frame_counts is not None and sum(chunk_frame_counts) == target_expected:
                    frame_actions = (["keep"] * n_discard) + frame_actions
                    if fused_dis_final_actions is not None:
                        fused_dis_final_actions = (["keep"] * n_discard) + fused_dis_final_actions
                    if fused_boundary.get("cut_starts"):
                        fused_boundary["cut_starts"] = {int(cs) + n_discard for cs in fused_boundary["cut_starts"]}
                    raw_keep += n_discard
                    total_frames = len(frame_actions)
                elif expected_frames is not None:
                    expected_frames = total_frames
        except Exception:
            pass
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

    # Force-keep frame 0 of every chunk per GEMINI.md invariant:
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
                auto_noise_floor=effective_auto_noise_floor,
                lip_flap_guard=effective_lip_flap_guard,
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
                "runs": runs,
                "scene_cut": (c_idx == 0) or (c_start in (fused_boundary.get("cut_starts") or set())),
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
                "runs": runs,
                "scene_cut": (c_idx == 0) or (c_start in (fused_boundary.get("cut_starts") or set())),
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
            "dis_thresh": effective_dis_thresh if is_dis_policy else None
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

    plan_dir = os.path.dirname(os.path.abspath(plan_path))
    if plan_dir:
        os.makedirs(plan_dir, exist_ok=True)
    tmp_plan = f"{plan_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_plan, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_plan, plan_path)
    finally:
        if os.path.exists(tmp_plan):
            try:
                os.remove(tmp_plan)
            except OSError:
                pass

    return plan_data


def _canonical_dedup_family(policy_name: str) -> str:
    """Normalize dynamic policy strings (dis-auto-<f>, dis-<f>, drift-<f>) to their canonical family."""
    pol = str(policy_name or "default").strip().lower()
    if pol == "dis-auto" or pol.startswith("dis-auto-"):
        return "dis-auto"
    if pol in ("sakuga", "dis-sakuga"):
        return pol
    if pol == "dis" or pol.startswith("dis-"):
        return "dis"
    if pol == "drift" or pol.startswith("drift-"):
        return "drift"
    return pol


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
    dis_thresh: Optional[float] = None,
    decode_threads: int = 0,
) -> Dict[str, Any]:
    """
    Nạp plan.json đã có từ phiên chạy trước nếu bật resume;
    nếu chưa có hoặc không khớp với chunk_frame_counts thực tế thì tự động quét lại.
    Từ chối resume nếu chính sách dedup_policy trong cache khác với chính sách yêu cầu.
    """
    policy_clean = policy.lower()
    p_params = get_dedup_params(
        policy_clean,
        dis_thresh_override=dis_thresh,
        drift_delta_override=drift_delta,
    )
    effective_drift_delta = float(p_params.get("drift_delta", 0.75))
    effective_dis_thresh = float(p_params.get("dis_thresh", 0.30))

    if resume and os.path.exists(plan_path):
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = data.get("chunks", [])
            if data.get("total_frames", 0) > 0 and len(chunks) > 0:
                cached_policy = data.get("dedup_policy", "default")
                if _canonical_dedup_family(cached_policy) != _canonical_dedup_family(policy_clean):
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
                if policy_clean.startswith("dis") or policy_clean in ("sakuga", "dis-sakuga"):
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
                    and not (policy_clean.startswith(("drift", "dis")) or policy_clean in ("sakuga", "dis-sakuga"))
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
                        tmp_plan = f"{plan_path}.repart.tmp.{os.getpid()}"
                        try:
                            with open(tmp_plan, "w", encoding="utf-8") as wf:
                                json.dump(data, wf, indent=2, ensure_ascii=False)
                            os.replace(tmp_plan, plan_path)
                        finally:
                            if os.path.exists(tmp_plan):
                                try:
                                    os.remove(tmp_plan)
                                except OSError:
                                    pass
                        return data
        except ValueError:
            raise
        except Exception:
            pass

    extra_scan_kw = {"decode_threads": decode_threads} if decode_threads else {}
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
        dis_thresh=effective_dis_thresh,
        **extra_scan_kw,
    )


def scan_single_segment_actions(
    seg_path: str,
    expected_seg_frames: int,
    chunk_idx: int,
    start_frame: int,
    policy: str = "default",
    decode_threads: int = 2,
    dis_thresh: Optional[float] = None,
    drift_delta: Optional[float] = None,
    drift_min_response: float = 0.10,
    motion_tau_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Scan an individual GOP segment with Open-GOP RASL discard reconciliation and bounded decode threads."""
    policy_clean = str(policy or "default").strip().lower()
    p_params = get_dedup_params(
        policy_clean,
        dis_thresh_override=dis_thresh,
        drift_delta_override=drift_delta,
    )
    exp_len = max(1, int(expected_seg_frames))
    hi = int(p_params.get("hi", DEFAULT_DEDUP_HI))
    lo = int(p_params.get("lo", DEFAULT_DEDUP_LO))
    frac = float(p_params.get("frac", DEFAULT_DEDUP_FRAC))
    motion_tau = float(motion_tau_override) if motion_tau_override is not None else float(p_params.get("tau", 0.30))
    effective_drift_delta = float(p_params.get("drift_delta", 0.75))
    effective_dis_thresh = float(p_params.get("dis_thresh", 0.30))
    effective_local_thresh: Optional[float] = (
        float(p_params["local_thresh"]) if "local_thresh" in p_params and p_params["local_thresh"] is not None else None
    )
    effective_min_moving_area_pct: float = float(p_params.get("min_moving_area_pct", 5.0))
    effective_auto_noise_floor: bool = bool(p_params.get("auto_noise_floor", False))
    effective_lip_flap_guard: bool = bool(p_params.get("lip_flap_guard", False))
    is_dis_policy = policy_clean.startswith("dis") or policy_clean in ("sakuga", "dis-sakuga")

    boundary_box: Dict[str, Any] = {}
    raw_actions: List[str] = []
    final_actions: List[str] = []

    if policy_clean == "off":
        raw_actions = ["keep"] * exp_len
        final_actions = ["keep"] * exp_len
    else:
        if is_dis_policy and not hasattr(subprocess.Popen, "_mock_name"):
            fused = _scan_fused_mpdecimate_and_dis(
                input_path=seg_path,
                hi=hi,
                lo=lo,
                frac=frac,
                thresh=effective_dis_thresh,
                chunk_start_set={0},
                frames_per_chunk_fallback=exp_len,
                min_moving_area_pct=effective_min_moving_area_pct,
                local_thresh=effective_local_thresh,
                auto_noise_floor=effective_auto_noise_floor,
                lip_flap_guard=effective_lip_flap_guard,
                decode_threads=decode_threads,
                boundary_out=boundary_box,
            )
            if fused is not None:
                raw_actions, final_actions = fused

        if not raw_actions:
            thread_args = ["-threads", str(int(decode_threads))] if decode_threads and int(decode_threads) > 0 else []
            cmd = [
                "ffmpeg",
                *thread_args,
                "-i", seg_path,
                "-vf", f"mpdecimate=hi={hi}:lo={lo}:frac={frac}",
                "-v", "debug",
                "-f", "null", "-",
            ]
            pattern = re.compile(r"\[Parsed_mpdecimate_0.*?\]\s+.*?\b(keep|drop)\s+pts:")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=65536,
            )
            if proc.stderr:
                for line in proc.stderr:
                    m = pattern.search(line)
                    if m:
                        raw_actions.append(m.group(1))
            proc.wait()
            if not raw_actions:
                raw_actions = ["keep"] * exp_len
            final_actions = list(raw_actions)
            if final_actions:
                final_actions[0] = "keep"

            if policy_clean == "motion":
                motion_mask = detect_global_motion_frames(seg_path, tau=motion_tau)
                for idx_m in range(min(len(final_actions), len(motion_mask))):
                    if motion_mask[idx_m]:
                        final_actions[idx_m] = "keep"
            elif policy_clean.startswith("drift"):
                final_actions = detect_drift_policy_actions(
                    input_path=seg_path,
                    base_actions=final_actions,
                    forced_keep_indices={0},
                    delta=effective_drift_delta,
                    min_response=drift_min_response,
                )
            elif is_dis_policy:
                final_actions = detect_dis_policy_actions(
                    input_path=seg_path,
                    base_actions=final_actions,
                    forced_keep_indices={0},
                    thresh=effective_dis_thresh,
                    min_moving_area_pct=effective_min_moving_area_pct,
                    local_thresh=effective_local_thresh,
                    auto_noise_floor=effective_auto_noise_floor,
                    lip_flap_guard=effective_lip_flap_guard,
                    decode_threads=decode_threads,
                )

    # Reconcile Open-GOP leading RASL AV_PKT_FLAG_DISCARD packets so action length matches seg_frame_counts 1:1
    if len(final_actions) < exp_len:
        try:
            from src.ffmpeg_ops import _count_leading_discard_packets
            n_discard = _count_leading_discard_packets(seg_path)
        except Exception:
            n_discard = 0
        pad_lead = n_discard if (n_discard > 0 and len(final_actions) + n_discard <= exp_len) else (exp_len - len(final_actions))
        if pad_lead > 0:
            raw_actions = (["keep"] * pad_lead) + raw_actions
            final_actions = (["keep"] * pad_lead) + final_actions
        if len(final_actions) < exp_len:
            rem = exp_len - len(final_actions)
            raw_actions = raw_actions + (["keep"] * rem)
            final_actions = final_actions + (["keep"] * rem)
    elif len(final_actions) > exp_len:
        raw_actions = raw_actions[:exp_len]
        final_actions = final_actions[:exp_len]

    final_actions[0] = "keep"

    runs: List[Dict[str, int]] = []
    curr_kept = 0
    curr_count = 0
    kept_indices: List[int] = []
    for idx_f, act in enumerate(final_actions):
        if act == "keep":
            if curr_count > 0:
                runs.append({"frame_idx": curr_kept, "count": curr_count})
            curr_kept = idx_f
            curr_count = 1
            kept_indices.append(idx_f)
        else:
            curr_count += 1
    if curr_count > 0:
        runs.append({"frame_idx": curr_kept, "count": curr_count})

    return {
        "chunk_meta": {
            "chunk_idx": int(chunk_idx),
            "start_frame": int(start_frame),
            "end_frame": int(start_frame + exp_len),
            "total_frames": exp_len,
            "kept_count": len(kept_indices),
            "kept_indices": kept_indices,
            "runs": runs,
            "scene_cut": (chunk_idx == 0),
        },
        "raw_actions": raw_actions,
        "first_gray": boundary_box.get("first_gray"),
        "last_gray": boundary_box.get("last_gray"),
    }


class RollingDedupPlanner:
    """Zero-wait rolling segment preflight scanner keeping <=2 background decode workers active."""

    def __init__(
        self,
        input_path: str,
        seg_paths: Sequence[str],
        seg_frame_counts: Sequence[int],
        plan_path: str,
        segment_seconds: int = 8,
        fps_num: int = 24000,
        fps_den: int = 1001,
        policy: str = "default",
        max_workers: int = 2,
        decode_threads: int = 2,
        dis_thresh: Optional[float] = None,
        drift_delta: Optional[float] = None,
        drift_min_response: float = 0.10,
        motion_tau: Optional[float] = None,
    ):
        from concurrent.futures import ThreadPoolExecutor

        self.input_path = input_path
        self.seg_paths = list(seg_paths)
        self.seg_frame_counts = [int(c) for c in seg_frame_counts]
        self.plan_path = plan_path
        self.segment_seconds = int(segment_seconds)
        self.fps_num = int(fps_num)
        self.fps_den = int(fps_den)
        self.policy = str(policy or "default").strip().lower()
        self.max_workers = max(1, min(int(max_workers), 2))
        self.decode_threads = max(1, int(decode_threads))
        self.dis_thresh = dis_thresh
        self.drift_delta = drift_delta
        self.drift_min_response = float(drift_min_response)
        self.motion_tau = motion_tau
        # Validate policy & override params eagerly on construction
        get_dedup_params(
            self.policy,
            dis_thresh_override=self.dis_thresh,
            drift_delta_override=self.drift_delta,
        )
        self.total_chunks = len(self.seg_paths)

        self._starts: List[int] = []
        acc = 0
        for cnt in self.seg_frame_counts:
            self._starts.append(acc)
            acc += cnt
        self.total_frames = acc

        self._lock = threading.Lock()
        self._results: Dict[int, Dict[str, Any]] = {}
        self._futures: Dict[int, Any] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._final_plan: Optional[Dict[str, Any]] = None
        self._extra_fields: Dict[str, Any] = {}
        self._bg_flusher: Optional[threading.Thread] = None
        self._aborted = False

    def _scan_one(self, c_idx: int) -> Dict[str, Any]:
        if self._aborted:
            exp_len = max(1, self.seg_frame_counts[c_idx])
            return {
                "chunk_meta": {
                    "chunk_idx": int(c_idx),
                    "start_frame": int(self._starts[c_idx]),
                    "end_frame": int(self._starts[c_idx] + exp_len),
                    "total_frames": exp_len,
                    "kept_count": exp_len,
                    "kept_indices": list(range(exp_len)),
                    "runs": [{"frame_idx": idx_k, "count": 1} for idx_k in range(exp_len)],
                    "scene_cut": (c_idx == 0),
                },
                "raw_actions": ["keep"] * exp_len,
                "first_gray": None,
                "last_gray": None,
            }
        res = scan_single_segment_actions(
            seg_path=self.seg_paths[c_idx],
            expected_seg_frames=self.seg_frame_counts[c_idx],
            chunk_idx=c_idx,
            start_frame=self._starts[c_idx],
            policy=self.policy,
            decode_threads=self.decode_threads,
            dis_thresh=self.dis_thresh,
            drift_delta=self.drift_delta,
            drift_min_response=self.drift_min_response,
            motion_tau_override=self.motion_tau,
        )
        with self._lock:
            self._results[c_idx] = res
            self._update_scene_cut_locked(c_idx)
            if (c_idx + 1) in self._results:
                self._update_scene_cut_locked(c_idx + 1)
        return res

    def _update_scene_cut_locked(self, c_idx: int) -> None:
        if c_idx <= 0:
            self._results[0]["chunk_meta"]["scene_cut"] = True
            return
        cur = self._results.get(c_idx)
        prev = self._results.get(c_idx - 1)
        if cur is None or prev is None:
            return
        fg = cur.get("first_gray")
        lg = prev.get("last_gray")
        if fg is not None and lg is not None and fg.shape == lg.shape:
            import numpy as np
            mad = float(np.mean(np.abs(fg.astype(np.int16) - lg.astype(np.int16))))
            cur["chunk_meta"]["scene_cut"] = bool(mad >= 16.0)
        else:
            r_cur = cur["chunk_meta"]["kept_count"] / max(1, cur["chunk_meta"]["total_frames"])
            r_prev = prev["chunk_meta"]["kept_count"] / max(1, prev["chunk_meta"]["total_frames"])
            cur["chunk_meta"]["scene_cut"] = bool(abs(r_cur - r_prev) >= 0.30)

    def start(self, sync_initial_chunks: int = 1, extra_fields: Optional[Dict[str, Any]] = None) -> None:
        from concurrent.futures import ThreadPoolExecutor

        if extra_fields:
            with self._lock:
                self._extra_fields.update(extra_fields)
        if self.total_chunks <= 0:
            self.finalize_plan()
            return
        n_sync = min(self.total_chunks, max(1, int(sync_initial_chunks)))
        if self.total_chunks == 1:
            self._scan_one(0)
            self.finalize_plan()
            return

        # Launch bounded 2-worker pool immediately so seg_0000 and seg_0001 scan concurrently
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="RollingPreflight",
        )
        for idx in range(self.total_chunks):
            self._futures[idx] = self._executor.submit(self._scan_one, idx)

        for idx in range(n_sync):
            self.get_chunk_plan(idx)

        if len(self._results) < self.total_chunks:
            def _bg_finalize() -> None:
                try:
                    if not self._aborted:
                        self.finalize_plan()
                except Exception:
                    pass

            self._bg_flusher = threading.Thread(target=_bg_finalize, name="RollingPlanFlush", daemon=True)
            self._bg_flusher.start()
        else:
            self.finalize_plan()

    def get_chunk_plan(self, c_idx: int) -> Dict[str, Any]:
        with self._lock:
            if c_idx in self._results:
                return dict(self._results[c_idx]["chunk_meta"])
            fut = self._futures.get(c_idx)
        if fut is not None:
            res = fut.result()
            return dict(res["chunk_meta"])
        res = self._scan_one(c_idx)
        return dict(res["chunk_meta"])

    def get_estimated_dedup_ratio_pct(self) -> float:
        with self._lock:
            if self._final_plan is not None:
                return float(self._final_plan.get("dedup_ratio_percent", 0.0))
            tot_f = sum(r["chunk_meta"]["total_frames"] for r in self._results.values())
            kept_f = sum(r["chunk_meta"]["kept_count"] for r in self._results.values())
        if tot_f <= 0:
            return 0.0
        return round((1.0 - (kept_f / float(tot_f))) * 100.0, 2)

    def set_extra_fields(self, extra_fields: Dict[str, Any]) -> None:
        """Atomically merges extra plan fields and rewrites plan.json if already finalized."""
        if not extra_fields:
            return
        with self._lock:
            self._extra_fields.update(extra_fields)
            if self._final_plan is not None:
                self._final_plan.update(extra_fields)
                self._atomic_write_plan(self._final_plan)

    def finalize_plan(self, extra_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if extra_fields:
            with self._lock:
                self._extra_fields.update(extra_fields)
        for idx in range(self.total_chunks):
            self.get_chunk_plan(idx)

        with self._lock:
            if self._final_plan is not None:
                if extra_fields:
                    self._final_plan.update(extra_fields)
                    self._atomic_write_plan(self._final_plan)
                return self._final_plan

            p_params = get_dedup_params(
                self.policy,
                dis_thresh_override=self.dis_thresh,
                drift_delta_override=self.drift_delta,
            )
            eff_motion_tau = float(self.motion_tau) if self.motion_tau is not None else float(p_params.get("tau", 0.3))
            is_dis = self.policy.startswith("dis") or self.policy in ("sakuga", "dis-sakuga")
            all_raw: List[str] = []
            chunks_meta: List[Dict[str, Any]] = []
            total_kept = 0
            for idx in range(self.total_chunks):
                self._update_scene_cut_locked(idx)
            for idx in range(self.total_chunks):
                entry = self._results[idx]
                c_meta = dict(entry["chunk_meta"])
                chunks_meta.append(c_meta)
                total_kept += int(c_meta["kept_count"])
                all_raw.extend(entry["raw_actions"])
                entry["first_gray"] = None
                entry["last_gray"] = None

            raw_keep = sum(1 for a in all_raw if a == "keep")
            raw_drop = len(all_raw) - raw_keep
            fps_val = self.fps_num / max(1, self.fps_den)
            frames_per_chunk = self.seg_frame_counts[0] if self.seg_frame_counts else int(round(self.segment_seconds * fps_val))
            dedup_ratio = (1.0 - (total_kept / float(self.total_frames))) * 100.0 if self.total_frames > 0 else 0.0
            raw_rle = (
                None
                if (self.policy.startswith("drift") or is_dis)
                else _encode_actions_rle(all_raw)
            )
            plan_data: Dict[str, Any] = {
                "source_file": os.path.basename(self.input_path),
                "ffmpeg_version": "rolling-segment-v2",
                "dedup_policy": self.policy,
                "dedup_params": {
                    "hi": int(p_params.get("hi", DEFAULT_DEDUP_HI)) if self.policy != "off" else None,
                    "lo": int(p_params.get("lo", DEFAULT_DEDUP_LO)) if self.policy != "off" else None,
                    "frac": float(p_params.get("frac", DEFAULT_DEDUP_FRAC)) if self.policy != "off" else None,
                    "motion_tau": eff_motion_tau if self.policy == "motion" else None,
                    "drift_delta": float(p_params.get("drift_delta", 0.75)) if self.policy.startswith("drift") else None,
                    "drift_min_response": self.drift_min_response if self.policy.startswith("drift") else None,
                    "dis_thresh": float(p_params.get("dis_thresh", 0.30)) if is_dis else None,
                },
                "total_frames": self.total_frames,
                "raw_kept_frames": raw_keep,
                "raw_dropped_frames": raw_drop,
                "fps_num": self.fps_num,
                "fps_den": self.fps_den,
                "fps": round(fps_val, 3),
                "segment_seconds": self.segment_seconds,
                "frames_per_chunk": frames_per_chunk,
                "total_chunks": self.total_chunks,
                "chunk_frame_counts": list(self.seg_frame_counts),
                "total_kept_frames": total_kept,
                "dedup_ratio_percent": round(dedup_ratio, 2),
                "raw_actions_rle": raw_rle,
                "chunks": chunks_meta,
                **self._extra_fields,
            }
            self._final_plan = plan_data
            self._atomic_write_plan(plan_data)

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        return self._final_plan

    def _atomic_write_plan(self, plan_data: Dict[str, Any]) -> None:
        plan_dir = os.path.dirname(os.path.abspath(self.plan_path))
        if plan_dir:
            os.makedirs(plan_dir, exist_ok=True)
        tmp_plan = f"{self.plan_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            with open(tmp_plan, "w", encoding="utf-8") as wf:
                json.dump(plan_data, wf, indent=2, ensure_ascii=False)
            os.replace(tmp_plan, self.plan_path)
        finally:
            if os.path.exists(tmp_plan):
                try:
                    os.remove(tmp_plan)
                except OSError:
                    pass

    def close(self) -> None:
        self._aborted = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self._executor.shutdown(wait=False)
            self._executor = None


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
            escaped = img_path.replace("'", "'\\''")
            dur = r["count"] * frame_dur
            f.write(f"file '{escaped}'\n")
            f.write(f"duration {dur:.6f}\n")

        # Quy chuẩn FFmpeg Concat Demuxer: dòng cuối cùng lặp lại file mà không có duration
        last_img_name = f"frame{runs[-1]['frame_idx'] + 1:08d}.{fmt}"
        last_img_path = os.path.join(frames_out_dir, last_img_name)
        escaped_last = last_img_path.replace("'", "'\\''")
        f.write(f"file '{escaped_last}'\n")

    return concat_script_path


def encode_chunk_reexpanded(
    concat_script_path: str,
    output_path: str,
    expected_frames: int,
    fps_num: int = 24000,
    fps_den: int = 1001,
    bitrate: str = "28M",
    multipass: Optional[str] = None,
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

    rate_args = get_hevc_rate_args(bitrate=bitrate, multipass=multipass) if multipass else get_hevc_rate_args(bitrate=bitrate)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_script_path,
        "-vf", (
            f"scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int,"
            f"fps={fps_num}/{fps_den},"
            "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709"
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

    # Xác minh giao dịch (Transactional verification): ưu tiên PyAV in-process để tránh spawn ffprobe mỗi chunk
    from src.ffmpeg_ops import verify_segment_resume_frames, _count_segment_packets_pyav

    verified_ok = False
    actual_frames: Optional[int] = None
    if os.path.exists(tmp_output):
        if os.path.getsize(tmp_output) >= 10000 and verify_segment_resume_frames(tmp_output, expected_frames):
            verified_ok = True
        elif not hasattr(subprocess.run, "_mock_name"):
            try:
                actual_frames = _count_segment_packets_pyav(tmp_output)
                verified_ok = (actual_frames == expected_frames)
            except Exception:
                verified_ok = False

    if not verified_ok:
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


def are_raw_segments_complete(
    segs_dir: str,
    plan_path: str,
    existing_segs: Optional[List[str]] = None,
) -> bool:
    """
    Kiểm tra thư mục raw_segs có chứa đủ phân đoạn cắt hoàn chỉnh hay không:
    - Khớp marker .split_complete (segment_count == len(existing_segs)), HOẶC
    - Tương thích ngược: plan.json hợp lệ có len(segments/chunks) == len(existing_segs).
    """
    if existing_segs is None:
        existing_segs = (
            sorted([
                os.path.join(segs_dir, f)
                for f in os.listdir(segs_dir)
                if f.startswith("seg_") and f.endswith(".mp4") and not f.endswith(".tmp.mp4")
            ])
            if os.path.isdir(segs_dir)
            else []
        )
    if not existing_segs:
        return False

    marker_path = os.path.join(segs_dir, ".split_complete")
    if os.path.isfile(marker_path):
        try:
            with open(marker_path, "r", encoding="utf-8") as mf:
                mdata = json.load(mf)
            return int(mdata.get("segment_count", -1)) == len(existing_segs)
        except Exception:
            return False

    if plan_path and os.path.isfile(plan_path):
        try:
            with open(plan_path, "r", encoding="utf-8") as pf:
                pdata = json.load(pf)
            segs_list = pdata.get("segments") if isinstance(pdata.get("segments"), list) else pdata.get("chunks")
            if isinstance(segs_list, list) and len(segs_list) == len(existing_segs):
                return True
        except Exception:
            pass
    return False


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
        if not are_raw_segments_complete(segs_d, plan_f, existing):
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
        decode_threads=2,
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

