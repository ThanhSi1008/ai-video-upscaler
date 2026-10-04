# -*- coding: utf-8 -*-
"""
tests/test_mux_audit_regression.py
Kiểm thử hồi quy: Tái hiện lỗi cũ (audio dồn khối / thiếu gói tin ở các cửa sổ sau)
và chứng minh rằng verify_mux_integrity trong src/mux_audit.py bắt được lỗi 100% (trả về status='FAIL').
"""

import os
import sys
import shutil
import tempfile
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mux_audit import verify_mux_integrity, check_window_interleaving


def test_mux_audit_catches_bugged_audio():
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

        # 2. Tạo file LỖI (Bugged): Video 60 giây nhưng Audio chỉ có 5 giây đầu (sau 5s bị câm hoàn toàn)
        # Tái hiện chính xác tình trạng audio dồn khối đầu file và câm hoàn toàn ở các cửa sổ sau
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

        # 3. Tạo file CHUẨN (Healthy): Video 60s và Audio 60s đầy đủ với xen kẽ chặt chẽ
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
        print("\n--- KẾT QUẢ KIỂM THỬ TRÊN FILE LỖI (BUGGED) ---")
        print(f"Status: {res_bug['status']}")
        print(f"Failure reasons: {res_bug['failure_reasons']}")

        assert res_bug["status"] == "FAIL", "LỖI: Mux audit không bắt được file bị mất audio!"
        assert len(res_bug["failure_reasons"]) > 0, "LỖI: Phải có ít nhất 1 lý do thất bại được liệt kê!"
        assert not res_bug["window_interleaving_10s"]["passed"] or not res_bug["volume_seek_test"]["passed"], \
            "LỖI: Window interleaving hoặc volume seek test phải đánh dấu thất bại!"
        print("✅ PASS: verify_mux_integrity đã phát hiện thành công lỗi mất audio/dồn khối!")

        # 5. Kiểm tra file chuẩn: verify_mux_integrity BẮT BUỘC phải PASS
        res_healthy = verify_mux_integrity(src_mkv, healthy_mkv)
        print("\n--- KẾT QUẢ KIỂM THỬ TRÊN FILE CHUẨN (HEALTHY) ---")
        print(f"Status: {res_healthy['status']}")
        assert res_healthy["status"] == "PASS", f"LỖI: File chuẩn bị đánh rớt: {res_healthy['failure_reasons']}"
        print("✅ PASS: verify_mux_integrity đã xác nhận thành công file chuẩn!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_mux_audit_catches_bugged_audio()
    print("\n🎉 Toàn bộ bài test hồi quy Mux Audit Regression đã thành công 100%!")
