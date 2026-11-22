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
from unittest import mock
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mux_audit import (
    verify_mux_integrity,
    check_window_interleaving,
    get_all_audio_streams_md5,
    get_audio_stream_md5,
    get_detailed_stream_info,
    count_subtitle_packets,
    _count_video_packets,
    _get_detailed_stream_info_pyav,
    _check_window_interleaving_pyav,
)


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
            "-f", "lavfi", "-i", "testsrc=duration=60:size=160x90:rate=8",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
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
            "-f", "lavfi", "-i", "testsrc=duration=60:size=160x90:rate=8",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=60",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-map", "2:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
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


def test_get_detailed_stream_info_pyav():
    """Tests _get_detailed_stream_info_pyav parses streams, languages, and dispositions."""
    with mock.patch("av.open") as mock_open:
        mock_c = mock.MagicMock()

        s_audio = mock.MagicMock(type="audio", index=1)
        s_audio.name = "flac"
        s_audio.metadata = {"language": "jpn"}
        s_audio.disposition = mock.MagicMock(default=1, forced=0)

        s_sub = mock.MagicMock(type="subtitle", index=2)
        s_sub.name = "ass"
        s_sub.metadata = {"language": "vie"}
        s_sub.disposition = None

        mock_c.streams = [s_audio, s_sub]
        mock_open.return_value.__enter__.return_value = mock_c

        info = _get_detailed_stream_info_pyav("test.mkv")
        assert len(info["audio_streams"]) == 1
        assert info["audio_streams"][0]["language"] == "jpn"
        assert info["audio_streams"][0]["disposition"]["default"] == 1
        assert len(info["subtitle_streams"]) == 1
        assert info["subtitle_streams"][0]["language"] == "vie"
        assert info["subtitle_streams"][0]["disposition"]["default"] == 0


def test_count_subtitle_packets_pyav():
    """Tests count_subtitle_packets via PyAV demux."""
    with mock.patch("av.open") as mock_open:
        mock_c = mock.MagicMock()
        mock_sub = mock.MagicMock(frames=0)
        mock_c.streams.subtitles = [mock_sub]

        p1 = mock.MagicMock(size=50)
        p2 = mock.MagicMock(size=60)
        mock_c.demux.return_value = [p1, p2]
        mock_open.return_value.__enter__.return_value = mock_c

        cnt = count_subtitle_packets("test.mkv", sub_stream_idx=0)
        assert cnt == 2

        # Stream index out of range returns None
        cnt_out = count_subtitle_packets("test.mkv", sub_stream_idx=5)
        assert cnt_out is None


def test_count_video_packets_pyav():
    """Tests _count_video_packets via PyAV frames or demux."""
    with mock.patch("av.open") as mock_open:
        mock_c = mock.MagicMock()
        mock_v = mock.MagicMock(frames=500)
        mock_c.streams.video = [mock_v]
        mock_open.return_value.__enter__.return_value = mock_c

        assert _count_video_packets("test.mkv") == 500


def test_check_window_interleaving_pyav():
    """Tests _check_window_interleaving_pyav detects audio and video presence across windows."""
    with mock.patch("av.open") as mock_open:
        mock_c = mock.MagicMock()
        v_stream = mock.MagicMock()
        a_stream = mock.MagicMock()
        mock_c.streams.video = [v_stream]
        mock_c.streams.audio = [a_stream]

        def mock_demux():
            seek_val = mock_c.seek.call_args[0][0] / 1_000_000.0 if mock_c.seek.call_args else 0.0
            return [
                mock.MagicMock(stream=v_stream, pts=int((seek_val + 1) * 1000), dts=int((seek_val + 1) * 1000), time_base=0.001),
                mock.MagicMock(stream=a_stream, pts=int((seek_val + 2) * 1000), dts=int((seek_val + 2) * 1000), time_base=0.001),
            ]
        mock_c.demux.side_effect = mock_demux
        mock_open.return_value.__enter__.return_value = mock_c

        # 1. Healthy interleaving: both video and audio present in all windows
        res_pass = _check_window_interleaving_pyav(
            "test.mkv",
            duration_sec=30.0,
            window_size_sec=10.0,
            num_samples=2,
            audio_count=1,
        )
        assert res_pass["passed"] is True
        assert res_pass["windows_tested"] >= 1

        # 2. Bugged interleaving: audio missing
        def mock_demux_no_audio():
            seek_val = mock_c.seek.call_args[0][0] / 1_000_000.0 if mock_c.seek.call_args else 0.0
            return [
                mock.MagicMock(stream=v_stream, pts=int((seek_val + 1) * 1000), dts=int((seek_val + 1) * 1000), time_base=0.001),
            ]
        mock_c.demux.side_effect = mock_demux_no_audio

        res_fail = _check_window_interleaving_pyav(
            "test.mkv",
            duration_sec=30.0,
            window_size_sec=10.0,
            num_samples=2,
            audio_count=1,
        )
        assert res_fail["passed"] is False
        assert len(res_fail["failed_windows"]) > 0


def test_pyav_vs_ffmpeg_audio_md5_parity_across_containers(tmp_path):
    """Verify PyAV packet MD5 matches ffmpeg -c copy -f md5 across MP4 (AAC) and MKV (FLAC+AC3)."""
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    mp4_path = tmp_path / "audio_aac.mp4"
    mkv_path = tmp_path / "audio_multi.mkv"

    subprocess.run(
        [
            ffmpeg_bin, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k",
            "-shortest",
            str(mp4_path),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            ffmpeg_bin, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=1",
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a:0", "flac", "-c:a:1", "ac3",
            "-shortest",
            str(mkv_path),
        ],
        check=True,
        capture_output=True,
    )

    mp4_tracks = get_all_audio_streams_md5(str(mp4_path), 1)
    assert len(mp4_tracks) == 1
    assert mp4_tracks[0]["md5"] == get_audio_stream_md5(str(mp4_path), 0)

    mkv_tracks = get_all_audio_streams_md5(str(mkv_path), 2)
    assert len(mkv_tracks) == 2
    assert mkv_tracks[0]["md5"] == get_audio_stream_md5(str(mkv_path), 0)
    assert mkv_tracks[1]["md5"] == get_audio_stream_md5(str(mkv_path), 1)


if __name__ == "__main__":
    test_mux_audit_catches_bugged_audio()
    test_mux_audit_catches_bugged_audio_track_2()
    print("\n🎉 Toàn bộ bài test hồi quy Mux Audit (Multi-track) thành công 100%!")
