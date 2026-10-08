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


def test_dedup_policy_default_backward_compatible_hash():
    """Kiểm tra: không truyền policy vs policy='default' sinh ra chunks và runs giống hệt nhau."""
    sample_video = "samples/bd_test/sample_1min_bd.mkv"
    if not os.path.exists(sample_video):
        pytest.skip(f"Không tìm thấy file mẫu {sample_video}")

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
    sample_video = "samples/bd_test/sample_1min_bd.mkv"
    if not os.path.exists(sample_video):
        pytest.skip(f"Không tìm thấy file mẫu {sample_video}")

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
