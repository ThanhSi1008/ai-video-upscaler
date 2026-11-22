#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_allow_partial.py
Kiểm thử tự động cho cơ chế --allow-partial:
1. Xác minh mặc định: --allow-partial KHÔNG bật mặc định (mặc định False).
2. Kiểm tra bộ lập lịch (Scheduler):
   - Khi không bật --allow-partial: Tập quá hạn chót bị từ chối (can_run=False, BỎ QUA).
   - Khi bật --allow-partial: Tập quá hạn chót được duyệt chạy 1 phần (can_run=True, is_partial=True).
3. Kiểm tra ranh giới dừng trong pipeline:
   - Mô phỏng deadline đến giữa tập: dừng sạch tại ranh giới chunk hiện tại.
   - Các segment up_*.mp4 đã encode được bảo toàn nguyên vẹn trên SSD.
   - Không thực hiện ghép nối/mux video dở dang.
   - Lần chạy tiếp theo (--resume): tiếp tục kế thừa segment đã hoàn tất.
"""

import os
import sys
import time
import shutil
import subprocess
import tempfile
import unittest
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mux_audit import verify_mux_integrity
from src.pipeline import run_upscale_pipeline
from src.scheduler import compute_overnight_schedule
import upscale


class TestAllowPartial(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_allow_partial_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cli_default_is_false(self):
        """1. Xác minh --allow-partial mặc định là False trong CLI parser."""
        parser = upscale.build_arg_parser()
        # Parse với tham số tối thiểu
        args = parser.parse_args(["samples/sample_30s.mkv"])
        self.assertFalse(args.allow_partial, "--allow-partial phải có giá trị mặc định là False (chưa bật mặc định)")

    def test_scheduler_decision_with_and_without_allow_partial(self):
        """2. Xác minh quyết định lập lịch khi quá deadline: BỎ QUA vs CHẠY 1 PHẦN."""
        queue = ["samples/sample_3min.mkv"]
        now = datetime(2026, 10, 4, 22, 0, 0)
        # Đặt deadline sau 10 phút (22:10) -> sample_3min cần ~30-35 phút nên chắc chắn không kịp
        deadline_str = "22:10"

        # Kịch bản A: KHÔNG bật allow_partial
        sched_no_partial = compute_overnight_schedule(
            queue=queue,
            deadline_str=deadline_str,
            fps_median=1.66,
            fps_p5=1.62,
            allow_partial=False,
            start_time=now
        )
        ep_no = sched_no_partial["episodes"][0]
        self.assertFalse(ep_no["can_run"], "Khi không có allow_partial, tập quá hạn phải can_run=False")
        self.assertFalse(ep_no["is_partial"])
        self.assertIn("BỎ QUA", ep_no["decision"])

        # Kịch bản B: CÓ bật allow_partial
        sched_with_partial = compute_overnight_schedule(
            queue=queue,
            deadline_str=deadline_str,
            fps_median=1.66,
            fps_p5=1.62,
            allow_partial=True,
            start_time=now
        )
        ep_partial = sched_with_partial["episodes"][0]
        self.assertTrue(ep_partial["can_run"], "Khi bật allow_partial, tập quá hạn phải được can_run=True")
        self.assertTrue(ep_partial["is_partial"], "Tập phải được đánh dấu is_partial=True")
        self.assertIn("CHẠY 1 PHẦN", ep_partial["decision"])

    def test_pipeline_stops_at_chunk_boundary_and_preserves_segments(self):
        """Exercise real run_upscale_pipeline partial pause at chunk 0 and clean resume to 4K MKV."""
        root = Path(self.test_dir)
        src_mkv = root / "ep_partial.mkv"
        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

        subprocess.run(
            [
                ffmpeg_bin, "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                "-vf", "format=yuv420p",
                "-c:v", "libx264", "-preset", "ultrafast", "-g", "24", "-keyint_min", "24", "-sc_threshold", "0",
                "-c:a", "aac", "-b:a", "96k",
                "-shortest",
                str(src_mkv),
            ],
            check=True,
            capture_output=True,
        )

        class _DummyCpuEngine:
            def __init__(self):
                self.model = nn.Upsample(scale_factor=2, mode="nearest")
                self.device = torch.device("cpu")
                self.batch_size = 2

            def restart_if_needed(self, **_kwargs):
                return None

        cpu_engine = _DummyCpuEngine()

        with (
            patch.dict(os.environ, {"UPSCALER_HEVC_ENCODER": "libx265"}),
            patch("src.config.HEVC_ENCODER", "libx265"),
            patch("src.trt_zero_copy_upscaler.HEVC_ENCODER", "libx265", create=True),
            patch("src.pipeline.HEVC_ENCODER", "libx265"),
            patch("src.cuda_engine.get_cuda_inference_engine", return_value=cpu_engine),
            patch("src.pipeline.acquire_gpu_lock", return_value=None),
            patch("src.pipeline.release_gpu_lock", return_value=None),
            patch("src.pipeline.AntiSleepContext", return_value=nullcontext()),
            patch("src.pipeline.send_system_notification", return_value=None),
        ):
            res_partial = run_upscale_pipeline(
                input_path=str(src_mkv),
                output_path=str(out_dir),
                engine="cuda",
                segment_time=1,
                segment_time_explicit=True,
                allow_partial=True,
                deadline_dt=datetime.now() - timedelta(seconds=1),
                zero_disk=True,
                use_tensorrt=False,
                precision="fp32",
                dedup=False,
                anti_sleep=False,
            )

            up_segs_dir = out_dir / ".temp_ep_partial" / "upscaled_segs"
            seg0 = up_segs_dir / "up_0000.mp4"
            seg1 = up_segs_dir / "up_0001.mp4"
            expected_4k = out_dir / "ep_partial_4K.mkv"

            self.assertEqual(os.path.abspath(res_partial), os.path.abspath(str(up_segs_dir)))
            self.assertTrue(seg0.is_file() and seg0.stat().st_size > 10000)
            self.assertFalse(seg1.exists())
            self.assertFalse(expected_4k.exists())
            self.assertFalse((out_dir / "ep_partial_4K.mkv.mux_tmp.mkv").exists())

            seg0_mtime_ns = seg0.stat().st_mtime_ns
            time.sleep(0.02)

            res_final = run_upscale_pipeline(
                input_path=str(src_mkv),
                output_path=str(out_dir),
                engine="cuda",
                segment_time=1,
                segment_time_explicit=True,
                resume=True,
                keep_temp=True,
                allow_partial=False,
                deadline_dt=None,
                zero_disk=True,
                use_tensorrt=False,
                precision="fp32",
                dedup=False,
                anti_sleep=False,
            )

        self.assertEqual(os.path.abspath(res_final), os.path.abspath(str(expected_4k)))
        self.assertTrue(expected_4k.is_file())
        self.assertEqual(seg0.stat().st_mtime_ns, seg0_mtime_ns)
        self.assertTrue(seg1.is_file() and seg1.stat().st_size > 10000)
        audit = verify_mux_integrity(str(src_mkv), str(expected_4k))
        self.assertEqual(audit["status"], "PASS", f"Mux audit failed: {audit}")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "upscale.py:1242 omits break on allow-partial pause and upscale.py:152-153 "
        "wraps past HH:MM deadline +1 day for subsequent queue episodes"
    ),
)
def test_queue_stops_after_allow_partial_pause_without_deadline_day_wrap(monkeypatch, tmp_path):
    """Lock queue continuation + HH:MM +1 day wrap bug after allow-partial pauses ep2."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    ep1 = tmp_path / "ep1.mkv"
    ep2 = tmp_path / "ep2.mkv"
    ep3 = tmp_path / "ep3.mkv"
    for p in (ep1, ep2, ep3):
        p.touch()

    clock = {"now": datetime(2026, 11, 22, 6, 59, 0)}

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"]

    called_inputs = []

    def _fake_run_upscale(input_path, **kwargs):
        called_inputs.append(Path(input_path).name)
        if Path(input_path).name == "ep1.mkv":
            out1 = out_dir / "ep1_4K.mkv"
            out1.touch()
            return str(out1)
        if Path(input_path).name == "ep2.mkv":
            clock["now"] = datetime(2026, 11, 22, 7, 1, 0)
            up_dir = out_dir / ".temp_ep2" / "upscaled_segs"
            up_dir.mkdir(parents=True, exist_ok=True)
            return str(up_dir)
        out3 = out_dir / "ep3_4K.mkv"
        out3.touch()
        return str(out3)

    monkeypatch.setattr(upscale, "datetime", _FakeDatetime)
    monkeypatch.setattr(upscale, "probe_video", lambda _p: {"nb_frames": 28800, "duration": 1200.0, "fps": 24.0})
    monkeypatch.setattr("src.scheduler.probe_video", lambda _p: {"nb_frames": 28800, "duration": 1200.0, "fps": 24.0})
    monkeypatch.setattr(upscale, "run_upscale_pipeline", _fake_run_upscale)
    monkeypatch.setattr(
        "sys.argv",
        [
            "upscale.py",
            "--queue", str(ep1), str(ep2), str(ep3),
            "--deadline", "07:00",
            "--allow-partial",
            "--no-subs",
            "-o", str(out_dir),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        upscale.main()

    assert exc_info.value.code == 1
    assert "ep3.mkv" not in called_inputs


if __name__ == "__main__":
    unittest.main()
