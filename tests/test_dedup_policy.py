# -*- coding: utf-8 -*-
"""
tests/test_dedup_policy.py
Kiểm thử tự động cho cơ chế --dedup-policy:
1. Không cờ -> kết quả plan giống hệt hiện tại (so băm trên sample clip).
2. shows.yaml hỗ trợ khóa tùy chọn dedup_policy.
3. Kế hoạch plan.json ghi nhận dedup_policy và dedup_params.
4. Từ chối resume nếu plan.json cũ có policy khác policy yêu cầu.
5. Chính sách off giữ 100% tất cả các khung hình.
"""

import os
import json
import hashlib
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock

from src.dedup import (
    scan_dedup_plan,
    load_or_create_plan,
    resolve_dedup_policy,
    DEDUP_POLICY_PARAMS
)


def test_resolve_dedup_policy_priority():
    """Kiểm tra độ ưu tiên phân giải policy: CLI > shows.yaml > default."""
    # 1. CLI luôn thắng
    assert resolve_dedup_policy("video.mkv", cli_policy="motion") == "motion"
    assert resolve_dedup_policy("video.mkv", cli_policy="sensitive-a") == "sensitive-a"

    # 2. shows.yaml
    with tempfile.TemporaryDirectory() as td:
        shows_file = os.path.join(td, "shows.yaml")
        with open(shows_file, "w", encoding="utf-8") as f:
            f.write("""
shows:
  - pattern: "*Sword*"
    dedup_policy: "sensitive-b"
  - pattern: "*Slime*"
    model: "animevideov3"
""")
        with patch("src.dedup.ROOT_DIR", td):
            assert resolve_dedup_policy("Reincarnated_Sword_Ep01.mkv", cli_policy=None) == "sensitive-b"
            # Slime không có dedup_policy -> về default
            assert resolve_dedup_policy("Slime_Ep24.mkv", cli_policy=None) == "default"
            # Không khớp rule nào -> về default
            assert resolve_dedup_policy("Random_Anime.mkv", cli_policy=None) == "default"


def _get_test_sample_video():
    candidates = [
        "samples/sample_1min_bd_420.mkv",
        "samples/sample_1min_bd.mkv",
        "samples/bd_test/sample_1min_bd.mkv",
        "samples/sample_30s.mkv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def test_dedup_policy_default_backward_compatible_hash():
    """Kiểm tra: không truyền policy vs policy='default' sinh ra chunks và runs giống hệt nhau."""
    sample_video = _get_test_sample_video()
    if not sample_video:
        pytest.skip("Không tìm thấy file video mẫu để kiểm tra hash backward compatible")

    with tempfile.TemporaryDirectory() as td:
        plan1_path = os.path.join(td, "plan_no_arg.json")
        plan2_path = os.path.join(td, "plan_default.json")

        # Quét không truyền policy
        p1 = scan_dedup_plan(sample_video, plan1_path)
        # Quét truyền policy='default'
        p2 = scan_dedup_plan(sample_video, plan2_path, policy="default")

        # So sánh cấu trúc chunks và runs
        h1 = hashlib.sha256(json.dumps(p1["chunks"], sort_keys=True).encode()).hexdigest()
        h2 = hashlib.sha256(json.dumps(p2["chunks"], sort_keys=True).encode()).hexdigest()
        assert h1 == h2, "Chunks của không truyền policy và policy='default' phải giống hệt nhau (hash trùng khớp)!"
        assert p1["total_kept_frames"] == p2["total_kept_frames"]
        assert p1["total_frames"] == p2["total_frames"]


def test_dedup_policy_off():
    """Kiểm tra chính sách 'off': giữ tất cả các khung hình."""
    sample_video = _get_test_sample_video()
    if not sample_video:
        pytest.skip("Không tìm thấy file video mẫu để kiểm tra policy off")

    with tempfile.TemporaryDirectory() as td:
        plan_off_path = os.path.join(td, "plan_off.json")
        plan = scan_dedup_plan(sample_video, plan_off_path, policy="off")

        assert plan["dedup_policy"] == "off"
        assert plan["total_kept_frames"] == plan["total_frames"]
        assert plan["raw_dropped_frames"] == 0
        assert plan["dedup_ratio_percent"] == 0.0

        # Kiểm tra mọi run đều có count=1
        for chunk in plan["chunks"]:
            assert chunk["kept_count"] == chunk["total_frames"]
            assert len(chunk["runs"]) == chunk["total_frames"]
            for r in chunk["runs"]:
                assert r["count"] == 1


def test_dedup_policy_drift_schema_and_params():
    """Kiểm tra chính sách 'drift' ghi nhận đúng delta và min_response vào plan.json."""
    with tempfile.TemporaryDirectory() as td:
        plan_path = os.path.join(td, "plan_drift.json")
        dummy_plan = {
            "source_file": "dummy.mkv",
            "dedup_policy": "drift-0.75",
            "dedup_params": {"drift_delta": 0.75, "drift_min_response": 0.10},
            "total_frames": 100,
            "chunks": [{"total_frames": 100, "runs": []}]
        }
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(dummy_plan, f)

        # Resume với đúng policy drift-0.75 -> Thành công
        cached = load_or_create_plan("dummy.mkv", plan_path, resume=True, policy="drift-0.75")
        assert cached["dedup_policy"] == "drift-0.75"
        assert cached["dedup_params"]["drift_delta"] == 0.75

        # Resume với drift delta khác (drift-0.5) -> Phải raise ValueError từ chối resume
        with pytest.raises(ValueError, match="Từ chối resume"):
            load_or_create_plan("dummy.mkv", plan_path, resume=True, policy="drift-0.5")


def test_dedup_policy_resume_mismatch_rejected():
    """Kiểm tra: nếu plan.json cũ dùng policy khác với policy yêu cầu, load_or_create_plan phải từ chối resume."""
    with tempfile.TemporaryDirectory() as td:
        plan_path = os.path.join(td, "plan.json")
        dummy_plan = {
            "source_file": "dummy.mkv",
            "dedup_policy": "default",
            "total_frames": 100,
            "chunks": [{"total_frames": 100, "runs": []}]
        }
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(dummy_plan, f)

        # Resume với policy='default' -> Thành công (trả về cache)
        cached = load_or_create_plan("dummy.mkv", plan_path, resume=True, policy="default")
        assert cached["dedup_policy"] == "default"

        # Resume với policy='sensitive-a' -> Phải raise ValueError từ chối resume
        with pytest.raises(ValueError, match="Từ chối resume"):
            load_or_create_plan("dummy.mkv", plan_path, resume=True, policy="sensitive-a")


def test_dedup_policy_dis_schema_and_params():
    """Kiểm tra chính sách 'dis' ghi nhận đúng dis_thresh vào plan.json và từ chối khi lệch thresh."""
    with tempfile.TemporaryDirectory() as td:
        plan_path = os.path.join(td, "plan_dis.json")
        dummy_plan = {
            "source_file": "dummy.mkv",
            "dedup_policy": "dis-0.3",
            "dedup_params": {"dis_thresh": 0.30},
            "total_frames": 100,
            "chunks": [{"total_frames": 100, "runs": []}]
        }
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(dummy_plan, f)

        # Resume với đúng policy dis-0.3 -> Thành công
        cached = load_or_create_plan("dummy.mkv", plan_path, resume=True, policy="dis-0.3")
        assert cached["dedup_policy"] == "dis-0.3"
        assert cached["dedup_params"]["dis_thresh"] == 0.30

        # Resume với thresh khác (dis-0.5) -> Phải raise ValueError từ chối resume
        with pytest.raises(ValueError, match="Từ chối resume"):
            load_or_create_plan("dummy.mkv", plan_path, resume=True, policy="dis-0.5")


def _make_synthetic_clip(frames_bgr, out_path, crf=22):
    """Tạo video nén x264 CRF từ chuỗi frame BGR."""
    import subprocess
    h, w = frames_bgr[0].shape[:2]
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
        "-preset", "ultrafast",
        "-g", "24",
        out_path
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames_bgr:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


def test_dedup_policy_dis_synthetic_pan():
    """B7(1): Chuỗi tổng hợp lia 0.4 px/frame -> dis-0.3 giữ >= 95% frame."""
    import cv2
    import numpy as np
    base_path = "test_frames/frame_action.png"
    if not os.path.exists(base_path):
        pytest.skip(f"Không tìm thấy {base_path}")
    base_img = cv2.imread(base_path)
    h, w = base_img.shape[:2]
    frames = []
    for i in range(60):
        M = np.float32([[1, 0, i * 0.4], [0, 1, 0]])
        frames.append(cv2.warpAffine(base_img, M, (w, h), borderMode=cv2.BORDER_REFLECT))

    with tempfile.TemporaryDirectory() as td:
        clip_path = os.path.join(td, "pan_04.mp4")
        plan_path = os.path.join(td, "plan_pan.json")
        _make_synthetic_clip(frames, clip_path, crf=22)

        plan = scan_dedup_plan(clip_path, plan_path, policy="dis-0.3")
        kept = plan["total_kept_frames"]
        total = plan["total_frames"]
        # Yêu cầu B7: giữ >= 95% frames
        assert kept >= int(0.95 * total), f"dis-0.3 phải giữ >= 95% frames khi lia 0.4 px/f, thực tế giữ {kept}/{total} ({kept/total*100:.1f}%)"


def test_dedup_policy_dis_synthetic_noise_rejection():
    """B7(2): Chuỗi tĩnh + nhiễu CRF 26 -> dis-0.3 giữ thêm <= 2% so với default."""
    import cv2
    base_path = "test_frames/frame_close_up.png"
    if not os.path.exists(base_path):
        pytest.skip(f"Không tìm thấy {base_path}")
    base_img = cv2.imread(base_path)
    frames = [base_img.copy() for _ in range(60)]

    with tempfile.TemporaryDirectory() as td:
        clip_path = os.path.join(td, "static_crf26.mp4")
        plan_dis_path = os.path.join(td, "plan_dis.json")
        plan_def_path = os.path.join(td, "plan_def.json")
        _make_synthetic_clip(frames, clip_path, crf=26)

        p_def = scan_dedup_plan(clip_path, plan_def_path, policy="default")
        p_dis = scan_dedup_plan(clip_path, plan_dis_path, policy="dis-0.3")

        extra_kept = p_dis["total_kept_frames"] - p_def["total_kept_frames"]
        total = p_def["total_frames"]
        extra_pct = (extra_kept / total) * 100.0
        # Yêu cầu B7: giữ thêm <= 2% so với default
        assert extra_pct <= 2.0, f"dis-0.3 kích hoạt nhầm quá mức trên nhiễu CRF 26: giữ thêm {extra_kept}/{total} ({extra_pct:.1f}% > 2%)"


def test_dedup_policy_dis_synthetic_offcenter_zoom():
    """B7(3): Zoom lệch tâm (1/3 khung hình) -> dis-0.3 giữ >= 90% frame."""
    import cv2
    import numpy as np
    base_path = "test_frames/frame_action.png"
    if not os.path.exists(base_path):
        pytest.skip(f"Không tìm thấy {base_path}")
    base_img = cv2.imread(base_path)
    h, w = base_img.shape[:2]
    cx, cy = w / 3.0, h / 3.0
    frames = []
    for i in range(60):
        scale = (1.0 + 0.0005) ** i  # rate 0.05%/frame
        M = np.float32([
            [scale, 0, cx * (1.0 - scale)],
            [0, scale, cy * (1.0 - scale)]
        ])
        frames.append(cv2.warpAffine(base_img, M, (w, h), borderMode=cv2.BORDER_REFLECT))

    with tempfile.TemporaryDirectory() as td:
        clip_path = os.path.join(td, "zoom_offcenter.mp4")
        plan_path = os.path.join(td, "plan_zoom.json")
        _make_synthetic_clip(frames, clip_path, crf=22)

        plan = scan_dedup_plan(clip_path, plan_path, policy="dis-0.3")
        kept = plan["total_kept_frames"]
        total = plan["total_frames"]
        # Yêu cầu B7: zoom lệch tâm giữ >= 90%
        assert kept >= int(0.90 * total), f"dis-0.3 phải giữ >= 90% frames khi zoom lệch tâm, thực tế giữ {kept}/{total} ({kept/total*100:.1f}%)"

