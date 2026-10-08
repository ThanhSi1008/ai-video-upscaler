# -*- coding: utf-8 -*-
"""
tests/test_bd_gop_desync.py
Kiểm thử tự động khắc phục lỗi lệch khung hình (frame count desync)
giữa bộ cắt split_video_segments và kế hoạch dedup.py trên nguồn BD Remux có GOP dài.
"""

import os
import json
import tempfile
import shutil
import pytest
from src.dedup import scan_dedup_plan, load_or_create_plan
from src.ffmpeg_ops import get_segments_frame_counts


def test_bd_gop_variable_chunk_plan():
    """
    Kiểm tra kế hoạch dedup với GOP dài 250 frames (250, 250, 219 = 719 frames):
    - Đảm bảo tạo ra chính xác 3 chunks với độ dài [250, 250, 219].
    - Đảm bảo frame đầu mỗi chunk được ép 'keep'.
    - Đảm bảo tổng số frame trong runs của từng chunk khớp chính xác với segment tương ứng.
    """
    temp_dir = tempfile.mkdtemp(prefix="test_bd_gop_")
    try:
        plan_file = os.path.join(temp_dir, "plan.json")
        total_frames = 719
        chunk_counts = [250, 250, 219]

        # Giả lập frame_actions (xen kẽ keep/drop)
        # Giả lập scan_dedup_plan với frame_actions
        # Test scan_dedup_plan logic với chunk_frame_counts
        # Tạo file video mẫu giả lập nhỏ nếu cần hoặc gọi trực tiếp logic
        actions = ["keep" if i % 2 == 0 else "drop" for i in range(total_frames)]
        # Ép frame 0 của các chunk
        actions[0] = "drop" # sẽ bị ép thành keep
        actions[250] = "drop" # sẽ bị ép thành keep
        actions[500] = "drop" # sẽ bị ép thành keep

        from src.dedup import scan_dedup_plan
        # Mock hoặc patch để kiểm tra hàm
        import unittest.mock as mock

        with mock.patch("subprocess.run") as mock_run, \
             mock.patch("subprocess.Popen") as mock_popen:
            # Mock ffmpeg -version
            mock_run.return_value = mock.MagicMock(stdout="ffmpeg version 8.1.2\n", returncode=0)

            # Mock ffprobe count_packets
            probe_mock = mock.MagicMock(returncode=0, stdout=str(total_frames))

            def run_side_effect(cmd, *args, **kwargs):
                if "ffprobe" in cmd[0]:
                    return probe_mock
                return mock.MagicMock(stdout="ffmpeg version 8.1.2\n", returncode=0)

            mock_run.side_effect = run_side_effect

            # Mock ffmpeg mpdecimate stderr
            stderr_lines = [
                f"[Parsed_mpdecimate_0 @ 0x123] keep pts:{i*1000}\n" if a == "keep"
                else f"[Parsed_mpdecimate_0 @ 0x123] drop pts:{i*1000}\n"
                for i, a in enumerate(actions)
            ]
            mock_proc = mock.MagicMock()
            mock_proc.stderr = stderr_lines
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            plan = scan_dedup_plan(
                input_path="dummy_bd.mkv",
                plan_path=plan_file,
                segment_seconds=8,
                chunk_frame_counts=chunk_counts
            )

            assert plan["total_frames"] == 719
            assert plan["total_chunks"] == 3
            assert len(plan["chunks"]) == 3

            # Chunk 0
            c0 = plan["chunks"][0]
            assert c0["start_frame"] == 0
            assert c0["end_frame"] == 250
            assert c0["total_frames"] == 250
            assert sum(r["count"] for r in c0["runs"]) == 250
            assert 0 in c0["kept_indices"], "Frame 0 của chunk 0 phải được giữ"

            # Chunk 1
            c1 = plan["chunks"][1]
            assert c1["start_frame"] == 250
            assert c1["end_frame"] == 500
            assert c1["total_frames"] == 250
            assert sum(r["count"] for r in c1["runs"]) == 250
            assert 0 in c1["kept_indices"], "Frame 0 (tương ứng frame 250 của video) phải được giữ"

            # Chunk 2
            c2 = plan["chunks"][2]
            assert c2["start_frame"] == 500
            assert c2["end_frame"] == 719
            assert c2["total_frames"] == 219
            assert sum(r["count"] for r in c2["runs"]) == 219
            assert 0 in c2["kept_indices"], "Frame 0 (tương ứng frame 500 của video) phải được giữ"

            # Tổng số frame của tất cả runs
            total_reconstructed = sum(sum(r["count"] for r in c["runs"]) for c in plan["chunks"])
            assert total_reconstructed == 719, f"Tổng frame tái tạo ({total_reconstructed}) phải đúng 719"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_stale_plan_invalidation_on_gop_mismatch():
    """
    Kiểm tra load_or_create_plan tự động hủy bỏ plan cũ (lưới 192 cố định)
    khi phát hiện không khớp với chunk_frame_counts thực tế [250, 250, 219].
    """
    temp_dir = tempfile.mkdtemp(prefix="test_stale_plan_")
    try:
        plan_file = os.path.join(temp_dir, "plan.json")
        # Ghi một plan.json cũ theo lưới 192 frames (4 chunks)
        stale_plan = {
            "total_frames": 719,
            "chunks": [
                {"chunk_idx": 0, "total_frames": 192},
                {"chunk_idx": 1, "total_frames": 192},
                {"chunk_idx": 2, "total_frames": 192},
                {"chunk_idx": 3, "total_frames": 143}
            ]
        }
        with open(plan_file, "w") as f:
            json.dump(stale_plan, f)

        # Khi gọi load_or_create_plan với chunk_frame_counts=[250, 250, 219]
        # và resume=True, nó phải phát hiện không khớp và gọi scan_dedup_plan để tạo lại
        import unittest.mock as mock
        with mock.patch("src.dedup.scan_dedup_plan") as mock_scan:
            mock_scan.return_value = {"total_frames": 719, "chunks": [{"chunk_idx": 0, "total_frames": 250}]}
            res = load_or_create_plan(
                input_path="dummy.mkv",
                plan_path=plan_file,
                resume=True,
                chunk_frame_counts=[250, 250, 219]
            )
            assert mock_scan.called, "Phải quét lại khi plan.json cũ không khớp với phân đoạn thực tế!"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_mux_audit_catches_frame_count_mismatch():
    """
    Kiểm tra verify_mux_integrity trong src/mux_audit.py bắt được lỗi lệch frame:
    Nếu file output có 576 frames trong khi nguồn có 719 frames -> BẮT BUỘC status='FAIL'.
    """
    import unittest.mock as mock
    from src.mux_audit import verify_mux_integrity

    with mock.patch("src.mux_audit.probe_video") as mock_probe, \
         mock.patch("src.mux_audit.get_audio_stream_md5") as mock_md5, \
         mock.patch("src.mux_audit.check_window_interleaving") as mock_interleave, \
         mock.patch("subprocess.run") as mock_run:

        mock_probe.return_value = {
            "duration": 30.0,
            "audio_count": 1,
            "subtitle_count": 0,
            "attachment_count": 0,
            "chapter_count": 0
        }
        mock_md5.return_value = "fake_md5"
        mock_interleave.return_value = {"passed": True, "windows_tested": 3, "failed_windows": []}

        def fake_run(cmd, *args, **kwargs):
            m = mock.MagicMock(returncode=0)
            if "src.mkv" in cmd:
                m.stdout = "719\n"
            elif "out.mkv" in cmd:
                m.stdout = "576\n"
            return m

        mock_run.side_effect = fake_run

        res = verify_mux_integrity("src.mkv", "out.mkv")
        assert res["status"] == "FAIL", "verify_mux_integrity phải trả về FAIL khi lệch frame!"
        assert not res["video_frame_count"]["passed"]
        assert res["video_frame_count"]["source_frames"] == 719
        assert res["video_frame_count"]["output_frames"] == 576
        assert any("Lệch số frame video" in r for r in res["failure_reasons"])

