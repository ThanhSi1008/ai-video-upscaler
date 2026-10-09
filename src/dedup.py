# -*- coding: utf-8 -*-
"""
src/dedup.py
Module quản lý cơ chế Deduplication & Concat Re-expansion cho Anime Video Upscale:
- Phân tích toàn tập bằng bộ lọc mpdecimate (Preflight Scan lập plan.json 1 lần duy nhất)
- Bắt buộc ép giữ frame đầu mỗi chunk (bảo toàn 100% tính độc lập và Checkpoint Resume)
- Lọc bỏ khung hình trùng lặp trước khi đưa vào NCNN Vulkan Metal (giảm ~50% tải GPU)
- Tái tạo khung hình 4K (Re-expansion) bằng FFmpeg Concat Demuxer không tốn RAM/Disk
- Xác minh giao dịch phân đoạn (Transactional Segment verification) chuẩn xác từng frame
"""

import os
import re
import json
import shutil
import subprocess
from typing import List, Dict, Any, Optional

from src.config import (
    DEFAULT_DEDUP_HI,
    DEFAULT_DEDUP_LO,
    DEFAULT_DEDUP_FRAC,
    ROOT_DIR,
    HEVC_ENCODER,
    HEVC_PIX_FMT,
    HEVC_COLOR_RANGE,
    HEVC_COLORSPACE,
    HEVC_COLOR_PRIMARIES,
    HEVC_COLOR_TRC,
    HEVC_TAG
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
            import fnmatch
            with open(shows_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            rules = data.get("shows", []) or []
            vname = os.path.basename(video_path)
            vdir = os.path.basename(os.path.dirname(os.path.abspath(video_path)))
            for r in rules:
                pat = r.get("pattern", "")
                pol = r.get("dedup_policy")
                if pol and (fnmatch.fnmatch(vname, pat) or fnmatch.fnmatch(vdir, pat) or (pat and pat.lower() in vname.lower())):
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
    drift_min_response: float = 0.10
) -> Dict[str, Any]:
    """
    Quét phân tích toàn tập bằng filter mpdecimate để lập plan.json.
    Chỉ chạy 1 lần duy nhất trước khi upscale, overhead cực nhỏ (~1-2 phút CPU).
    Nếu có chunk_frame_counts (từ split_video_segments), các chunk sẽ khớp hoàn toàn với phân đoạn thực tế.
    Hỗ trợ chính sách dedup_policy: default, sensitive, sensitive-a, sensitive-b, motion, drift, off.
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

    # Áp dụng thông số theo chính sách nếu là giá trị mặc định
    if policy_clean in ["sensitive", "sensitive-a"]:
        hi, lo, frac = 512, 160, 0.15
    elif policy_clean == "sensitive-b":
        hi, lo, frac = 384, 128, 0.10
    elif policy_clean in ["motion", "default"] or policy_clean.startswith("drift"):
        hi, lo, frac = DEFAULT_DEDUP_HI, DEFAULT_DEDUP_LO, DEFAULT_DEDUP_FRAC

    frame_actions: List[str] = []

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
        cmd = [
            "ffmpeg", "-i", input_path,
            "-vf", f"mpdecimate=hi={hi}:lo={lo}:frac={frac}",
            "-v", "debug",
            "-f", "null", "-"
        ]

        pattern = re.compile(r"\[Parsed_mpdecimate_0.*?\]\s+.*?\b(keep|drop)\s+pts:")

        # Thực thi ffmpeg và bắt dòng log debug
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
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
            "drift_min_response": drift_min_response if policy_clean.startswith("drift") else None
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
    drift_min_response: float = 0.10
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
                if chunk_frame_counts is None:
                    return data
                cached_counts = [c["total_frames"] for c in chunks]
                if cached_counts == chunk_frame_counts:
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
        drift_min_response=drift_min_response
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
    Mã hóa phân đoạn 4K qua FFmpeg Concat Demuxer + VideoToolbox HEVC 10-bit:
    - Chốt cờ CFR: -video_track_timescale fps_num -frames:v expected_frames
    - Chuỗi ma trận màu sắc BT.709 Full RGB ➔ Limited YUV p010le
    - Giao dịch an toàn: ghi ra .tmp.mp4, xác minh đủ số frame rồi mới rename
    """
    tmp_output = f"{output_path}.tmp.mp4"
    if os.path.exists(tmp_output):
        os.remove(tmp_output)

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
        "-b:v", bitrate,
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


def spawn_background_preflight(
    video_path: str,
    output_dir: Optional[str] = None,
    segment_seconds: int = 8,
    logger=None
) -> Optional[subprocess.Popen]:
    """
    Kích hoạt tiến trình quét mpdecimate lập plan.json trước cho tập kế tiếp.
    Chạy với độ ưu tiên thấp (nice -n 10) trong nền lúc GPU đang bận xử lý tập trước.
    """
    import sys
    if os.environ.get("BENCHMARK_MODE") == "1":
        if logger:
            logger.warning("⚡ [BENCHMARK MODE] Từ chối kích hoạt spawn_background_preflight.", to_console=True)
        return None
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = output_dir or "out"
        work_dir = os.path.join(out_dir, f".temp_{base_name}")
        os.makedirs(work_dir, exist_ok=True)
        plan_file = os.path.join(work_dir, "plan.json")
        if os.path.exists(plan_file) and os.path.getsize(plan_file) > 100:
            return None

        # Gọi python chạy preflight với nice -n 10
        cmd = [
            "nice", "-n", "10",
            sys.executable, "-c",
            f"import sys; from src.dedup import load_or_create_plan; from src.ffmpeg_ops import probe_video; "
            f"meta = probe_video('{video_path}'); "
            f"load_or_create_plan('{video_path}', '{plan_file}', {segment_seconds}, meta['fps_num'], meta['fps_den'])"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        msg = f"⚡ [Multi-Episode Queue] Đã kích hoạt Preflight nền (nice -n 10) cho tập kế tiếp: {os.path.basename(video_path)}"
        if logger:
            logger.info(msg, to_console=True)
        else:
            print(msg)
        return proc
    except Exception:
        return None

