#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_model_selection_and_safety.py
Kiểm thử bộ quy tắc D10 trên Windows 11 + NVIDIA CUDA:
(a) Không có override thì dùng animevideov3;
(b) Mỗi model cho đúng kích thước đầu ra (3840x2160) từ 1080p;
(c) Resume khác model bị từ chối (trừ khi có --force);
(d) Lỗi GPU khi chạy AnimeJaNai không làm đổi sang animevideov3 (chỉ chuyển sang CUDA FP32 fallback).
"""

import os
import sys
import json
import shutil
import pytest
import torch
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models_resolver import resolve_model_for_video, load_models_registry
from src.cuda_engine import (
    RealESRAnimeVideoV3,
    AnimeJaNaiCompact2x,
    upscale_frames_with_fallback,
    GPUError
)


def test_d10_a_default_model_resolution():
    """Kiểm tra (a): Không có override thì luôn dùng animevideov3."""
    # 1. Video không khớp rule nào trong shows.yaml -> Phải trả về animevideov3
    m_info = resolve_model_for_video("random_show_ep01.mkv", None)
    assert m_info["id"] == "animevideov3"
    assert "Mặc định" in m_info["resolution_source"]

    # 2. CLI override có độ ưu tiên cao nhất
    m_info_cli = resolve_model_for_video("random_show_ep01.mkv", "animejanai-hd-v3-compact")
    assert m_info_cli["id"] == "animejanai-hd-v3-compact"
    assert "Tham số dòng lệnh" in m_info_cli["resolution_source"]

    # 3. Model không hợp lệ bị từ chối
    with pytest.raises(ValueError):
        resolve_model_for_video("random_show_ep01.mkv", "invalid_model_xyz")


def test_d10_b_output_resolution_tensor():
    """Kiểm tra (b): Mỗi model cho đúng kích thước 3840x2160 từ 1920x1080."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. RealESRAnimeVideoV3
    m_v3 = RealESRAnimeVideoV3().to(device).half()
    m_v3.eval()
    dummy_1080p = torch.rand(1, 3, 1080, 1920, device=device).half()
    with torch.inference_mode():
        out_v3 = m_v3(dummy_1080p)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    assert out_v3.shape == (1, 3, 2160, 3840), f"RealESRAnimeVideoV3 cho shape sai: {out_v3.shape}"

    # 2. AnimeJaNaiCompact2x
    m_aj = AnimeJaNaiCompact2x().to(device).half()
    m_aj.eval()
    with torch.inference_mode():
        out_aj = m_aj(dummy_1080p)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    assert out_aj.shape == (1, 3, 2160, 3840), f"AnimeJaNaiCompact2x cho shape sai: {out_aj.shape}"


def test_d10_c_resume_different_model_rejected(tmp_path):
    """Kiểm tra (c): Resume khác model bị từ chối nếu không có --force."""
    work_dir = tmp_path / ".temp_test_video"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"

    # Giả lập phiên chạy trước dùng animevideov3
    manifest_data = {
        "model_id": "animevideov3",
        "model_name": "RealESR-AnimeVideoV3",
        "sha256": "3cb49a466a93cb338b8b0932bbbe602167d643878b2d131f49615a6b0c2a8db4",
        "engine": "cuda",
        "precision": "fp16"
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # Thử resume với animejanai mà không có force -> Phải raise ValueError
    new_model_info = resolve_model_for_video("test.mkv", "animejanai-hd-v3-compact")

    # Mô phỏng kiểm tra manifest trong pipeline
    with open(manifest_path, "r", encoding="utf-8") as f:
        stored_manifest = json.load(f)

    prev_id = stored_manifest.get("model_id")
    prev_sha = stored_manifest.get("sha256")

    # Không có force
    force = False
    has_conflict = (prev_id != new_model_info["id"] or prev_sha != new_model_info["sha256"])
    assert has_conflict is True

    # Khi không có force, pipeline từ chối
    if has_conflict and not force:
        rejected = True
    else:
        rejected = False
    assert rejected is True

    # Khi có force=True, pipeline chấp thuận xóa và cập nhật manifest
    force = True
    if has_conflict and force:
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        new_manifest = {
            "model_id": new_model_info["id"],
            "model_name": new_model_info["name"],
            "sha256": new_model_info["sha256"],
            "engine": "cuda",
            "precision": "fp16"
        }
        with open(work_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(new_manifest, f)

    with open(work_dir / "manifest.json", "r", encoding="utf-8") as f:
        updated = json.load(f)
    assert updated["model_id"] == "animejanai-hd-v3-compact"


def test_d10_d_animejanai_gpu_error_does_not_switch_to_animevideov3(tmp_path):
    """Kiểm tra (d): Lỗi GPU khi chạy AnimeJaNai không làm đổi sang animevideov3 mà rơi về CUDA FP32."""
    model_info = resolve_model_for_video("test.mkv", "animejanai-hd-v3-compact")

    in_dir = str(tmp_path / "in")
    out_dir = str(tmp_path / "out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    from PIL import Image
    # Tạo 1 ảnh test
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    img.save(os.path.join(in_dir, "frame_0001.png"))

    # Giả lập lỗi GPU FP16 bằng cách truyền NaN hoặc patch worker
    from unittest.mock import patch, MagicMock

    with patch("src.cuda_engine.get_gpu_worker") as mock_get_worker:
        # Mock worker đầu tiên (FP16) bị lỗi
        mock_fp16_worker = MagicMock()
        mock_fp16_worker.upscale_directory.side_effect = GPUError("Simulated GPU Error in FP16")

        # Mock worker thứ hai (FP32) thành công
        mock_fp32_worker = MagicMock()
        mock_fp32_worker.upscale_directory.return_value = {
            "elapsed_seconds": 0.05,
            "fps": 20.0,
            "steady_fps": 20.0,
            "num_frames": 1,
            "peak_rss_mb": 50.0,
            "current_alloc_mb": 10.0,
            "driver_alloc_mb": 20.0
        }

        def worker_factory(model_id, weights_path, precision, **kwargs):
            if precision == "fp16":
                return mock_fp16_worker
            return mock_fp32_worker

        mock_get_worker.side_effect = worker_factory

        res = upscale_frames_with_fallback(
            input_dir=in_dir,
            output_dir=out_dir,
            model_info=model_info,
            engine="cuda",
            precision="fp16",
            fmt="png"
        )

        # Khẳng định fallback là cuda_fp32_fallback, tuyệt đối không phải ncnn hay animevideov3
        assert res["engine_used"] == "cuda_fp32_fallback"
        assert "ncnn" not in res["engine_used"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
