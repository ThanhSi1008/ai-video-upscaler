"""
tests/test_output_sanity.py - Bộ kiểm thử chốt chặn chất lượng hình ảnh đầu ra (C3)
Kiểm tra chốt chặn phát hiện chính xác ảnh đen / sai lệch cấu trúc và chi phí < 1s.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest
import torch

from src.output_sanity import check_frame_sanity, OutputSanityError, write_failed_sanity_audit
from src.mps_engine import AnimeJaNaiCompact2x


@pytest.fixture
def sample_frames(tmp_path):
    """Tạo 1 frame nguồn 1080p và frame đầu ra 4K mẫu."""
    # Tạo ảnh mẫu 1080p với gradient và chi tiết
    src_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for y in range(1080):
        src_img[y, :, 0] = int((y / 1080) * 200) # B
        src_img[y, :, 1] = int(((1080 - y) / 1080) * 180) # G
    src_img[:, :, 2] = 120 # R
    
    src_path = str(tmp_path / "src_1080p.png")
    cv2.imwrite(src_path, src_img)

    # Frame 4K upscale bình thường (2x bicubic tương đương SR)
    out_img = cv2.resize(src_img, (3840, 2160), interpolation=cv2.INTER_CUBIC)
    out_path = str(tmp_path / "out_4k_normal.png")
    cv2.imwrite(out_path, out_img)

    # Frame 4K bị đen (mô phỏng lỗi thiếu phép cộng residual)
    black_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    black_path = str(tmp_path / "out_4k_black.png")
    cv2.imwrite(black_path, black_img)

    return {
        "src_path": src_path,
        "out_normal": out_path,
        "out_black": black_path,
        "src_img": src_img
    }


def test_output_sanity_normal(sample_frames):
    """(b) Đầu ra bình thường: Chốt chặn phải PASS với |ΔY| <= 6 và Corr >= 0.98."""
    res = check_frame_sanity(sample_frames["src_path"], sample_frames["out_normal"])
    assert res["passed"] is True
    assert res["delta_y"] <= 6.0
    assert res["corr"] >= 0.98
    assert res["elapsed_ms"] < 1000.0  # Chi phí < 1s


def test_output_sanity_injected_black_screen(sample_frames):
    """(a) Cố tình tiêm lỗi màn hình đen (bỏ phép cộng residual): Chốt chặn phải kích hoạt FAIL."""
    res = check_frame_sanity(sample_frames["src_path"], sample_frames["out_black"])
    assert res["passed"] is False
    assert res["delta_y"] > 6.0  # Lệch độ sáng rất lớn do đen hoàn toàn
    assert res["corr"] < 0.98


def test_output_sanity_monkeypatch_bug_injection(tmp_path):
    """Tiêm lỗi trực tiếp vào AnimeJaNaiCompact2x (bỏ phép cộng nearest) và kiểm tra bắt lỗi."""
    # Khởi tạo mô hình
    model = AnimeJaNaiCompact2x().eval()
    for p in model.body.parameters():
        p.data.zero_()
    
    # Tạo tensor ảnh 1080p
    img = cv2.imread("test_frames/frame_close_up.png")
    if img is None:
        img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    
    src_file = str(tmp_path / "src.png")
    cv2.imwrite(src_file, img)

    # 1. Chạy với mô hình chuẩn (có residual)
    x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        out_good = (torch.clamp(model(x), 0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    good_file = str(tmp_path / "out_good.png")
    cv2.imwrite(good_file, cv2.cvtColor(out_good, cv2.COLOR_RGB2BGR))
    
    res_good = check_frame_sanity(src_file, good_file)
    assert res_good["passed"] is True

    # 2. Monkeypatch tiêm lại lỗi cũ (bỏ phép cộng residual)
    def flawed_forward(self, inp):
        feat = inp
        for l in self.body:
            feat = l(feat)
        return self.upsampler(feat)
    
    model.forward = flawed_forward.__get__(model, AnimeJaNaiCompact2x)
    with torch.no_grad():
        out_bad = (torch.clamp(model(x), 0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bad_file = str(tmp_path / "out_bad.png")
    cv2.imwrite(bad_file, cv2.cvtColor(out_bad, cv2.COLOR_RGB2BGR))

    res_bad = check_frame_sanity(src_file, bad_file)
    assert res_bad["passed"] is False
    assert res_bad["delta_y"] > 50.0  # Lệch độ sáng cực kỳ nghiêm trọng


def test_output_sanity_audit_generation(tmp_path, sample_frames):
    """Kiểm tra ghi file audit.json khi chốt chặn thất bại."""
    audit_file = str(tmp_path / "test_video.audit.json")
    res = check_frame_sanity(sample_frames["src_path"], sample_frames["out_black"])
    
    model_info = {
        "id": "animejanai-hd-v3-compact",
        "name": "AnimeJaNai HD V3 Compact",
        "sha256": "af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f"
    }

    write_failed_sanity_audit(
        audit_file_path=audit_file,
        source_file=sample_frames["src_path"],
        output_file=str(tmp_path / "test_video_4K.mkv"),
        model_info=model_info,
        engine="cuda",
        precision="fp16",
        chunk_idx=0,
        metrics=res,
        all_checks=[res]
    )

    assert os.path.exists(audit_file)
    import json
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "FAIL (OUTPUT_SANITY)"
    assert data["failed_chunk"] == 1
    assert "Chốt chặn đầu ra thất bại" in data["failure_reason"]


def test_output_sanity_latency_under_1s(sample_frames):
    """Đo chi phí thực thi: phải < 1s (yêu cầu C3)."""
    t0 = cv2.getTickCount()
    res = check_frame_sanity(sample_frames["src_path"], sample_frames["out_normal"])
    t1 = cv2.getTickCount()
    elapsed_sec = (t1 - t0) / cv2.getTickFrequency()
    assert elapsed_sec < 1.0
    assert res["elapsed_ms"] < 1000.0
