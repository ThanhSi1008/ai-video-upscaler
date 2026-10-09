# -*- coding: utf-8 -*-
"""
tests/test_mux_audit_regression.py
Kiểm thử hồi quy Mux Audit (Chuẩn phòng thu):
1. Tái hiện lỗi audio 1 track bị mất/dồn khối -> verify_mux_integrity trả về FAIL.
2. Tái hiện lỗi audio track 2 bị dồn khối (BD Remux 2 audio tracks FLAC) -> verify_mux_integrity bắt được 100% và trả về FAIL.
3. Xác nhận file 2 audio tracks lành lặn đầy đủ -> verify_mux_integrity trả về PASS.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mux_audit import verify_mux_integrity, check_window_interleaving


def test_mux_audit_catches_bugged_audio():
    """Kiểm tra phát hiện lỗi khi audio đơn track bị cắt ngắn dồn khối."""
    temp_dir = tempfile.mkdtemp(prefix="test_mux_audit_")
    try:
        src_mkv = os.path.join(temp_dir, "test_source.mkv")
        bugged_mkv = os.path.join(temp_dir, "test_bugged.mkv")
        healthy_mkv = os.path.join(temp_dir, "test_healthy.mkv")

        # 1. Tạo file nguồn 60 giây có video testsrc và audio sine wave 440Hz
        cmd_src = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=60:size=640x360:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            src_mkv
        ]
        subprocess.run(cmd_src, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Tạo file LỖI (Bugged): Video 60 giây nhưng Audio chỉ có 5 giây đầu
        cmd_bug = [
            "ffmpeg", "-y",
            "-i", src_mkv,
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            bugged_mkv
        ]
        subprocess.run(cmd_bug, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Tạo file CHUẨN (Healthy): Video 60s và Audio 60s đầy đủ
        cmd_healthy = [
            "ffmpeg", "-y",
            "-i", src_mkv,
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-c", "copy",
            "-max_interleave_delta", "0",
            healthy_mkv
        ]
        subprocess.run(cmd_healthy, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4. Kiểm tra file lỗi: verify_mux_integrity BẮT BUỘC phải phát hiện và trả về FAIL
        res_bug = verify_mux_integrity(src_mkv, bugged_mkv)
        assert res_bug["status"] == "FAIL", "LỖI: Mux audit không bắt được file bị mất audio!"
        assert len(res_bug["failure_reasons"]) > 0

        # 5. Kiểm tra file chuẩn: verify_mux_integrity BẮT BUỘC phải PASS
        res_healthy = verify_mux_integrity(src_mkv, healthy_mkv)
        assert res_healthy["status"] == "PASS", f"LỖI: File chuẩn bị đánh rớt: {res_healthy['failure_reasons']}"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_mux_audit_catches_bugged_audio_track_2():
    """
    Kiểm tra phát hiện lỗi trên container có 2 track audio:
    Track 1 bình thường, nhưng Track 2 bị dồn khối (chỉ có 5s audio ở đầu, sau đó rỗng gói tin).
    verify_mux_integrity BẮT BUỘC phải bắt trúng và trả về FAIL.
    """
    temp_dir = tempfile.mkdtemp(prefix="test_mux_audit_t2_")
    try:
        src_mkv = os.path.join(temp_dir, "test_source_2audio.mkv")
        bugged_mkv = os.path.join(temp_dir, "test_bugged_track2.mkv")
        healthy_mkv = os.path.join(temp_dir, "test_healthy_2audio.mkv")

        # 1. Tạo file nguồn 60 giây có video và 2 track audio (440Hz và 880Hz)
        cmd_src = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=60:size=640x360:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=60",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-map", "2:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            src_mkv
        ]
        subprocess.run(cmd_src, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Tạo file LỖI: Track 1 bình thường (60s), nhưng Track 2 chỉ có 5s (dồn khối đầu file)
        cmd_bug = [
            "ffmpeg", "-y",
            "-i", src_mkv,
            "-f", "lavfi", "-i", "sine=frequency=880:duration=5",
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "copy",
            bugged_mkv
        ]
        subprocess.run(cmd_bug, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Tạo file CHUẨN: Cả 2 track audio 60s đầy đủ
        cmd_healthy = [
            "ffmpeg", "-y",
            "-i", src_mkv,
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-map", "0:a:1",
            "-c", "copy",
            "-max_interleave_delta", "0",
            healthy_mkv
        ]
        subprocess.run(cmd_healthy, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4. Kiểm tra file lỗi: verify_mux_integrity BẮT BUỘC phải FAIL do track 2
        res_bug = verify_mux_integrity(src_mkv, bugged_mkv)
        print("\n--- TEST BUGGED TRACK 2 AUDIT ---")
        print(f"Status: {res_bug['status']}")
        print(f"Failure reasons: {res_bug['failure_reasons']}")

        assert res_bug["status"] == "FAIL", "LỖI: Mux audit không bắt được lỗi audio track 2 bị dồn khối!"
        # Phải có lỗi chỉ rõ track 1 (index 1 / 0:a:1) hoặc interleaving
        has_t2_err = any("track 1" in r.lower() or "0:a:1" in r or "xen kẽ" in r.lower() for r in res_bug["failure_reasons"])
        assert has_t2_err, f"LỖI: Không chỉ ra được track 2 bị lỗi trong failure reasons: {res_bug['failure_reasons']}"

        # 5. Kiểm tra file chuẩn: BẮT BUỘC phải PASS
        res_healthy = verify_mux_integrity(src_mkv, healthy_mkv)
        assert res_healthy["status"] == "PASS", f"LỖI: File 2 audio chuẩn bị đánh rớt: {res_healthy['failure_reasons']}"
        assert res_healthy["all_audio_tracks_md5"]["passed"] is True
        assert len(res_healthy["all_audio_tracks_md5"]["tracks"]) == 2

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_mux_audit_catches_bugged_audio()
    test_mux_audit_catches_bugged_audio_track_2()
    print("\n🎉 Toàn bộ bài test hồi quy Mux Audit (Multi-track) thành công 100%!")
