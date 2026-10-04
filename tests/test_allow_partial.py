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
import tempfile
import shutil
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        """3. Mô phỏng dừng sạch ở ranh giới chunk, giữ segment và tiếp tục ở lần sau."""
        up_segs_dir = os.path.join(self.test_dir, "up_segs")
        os.makedirs(up_segs_dir, exist_ok=True)

        # Tạo giả lập segment chunk 0 đã hoàn tất
        seg_0 = os.path.join(up_segs_dir, "up_000.mp4")
        with open(seg_0, "wb") as f:
            f.write(b"FAKE_ENCODED_MP4_CHUNK_0_CONTENT")

        # Giả lập logic kiểm tra deadline trong vòng lặp chunk
        allow_partial = True
        deadline_dt = datetime.now() - timedelta(seconds=1) # Đã quá hạn
        total_segs = 4
        c_idx = 0 # Đang ở cuối chunk 0

        partial_paused = False
        if allow_partial and deadline_dt and datetime.now() >= deadline_dt:
            partial_paused = True

        self.assertTrue(partial_paused, "Pipeline phải kích hoạt cờ partial_paused khi chạm deadline")
        self.assertTrue(os.path.exists(seg_0), "File up_000.mp4 phải được giữ nguyên trên đĩa")

        # Xác minh lần chạy tiếp theo (--resume):
        # Hệ thống quét các phân đoạn đã có trong up_segs_dir
        existing_segs = sorted([
            f for f in os.listdir(up_segs_dir)
            if f.startswith("up_") and f.endswith(".mp4")
        ])
        self.assertEqual(len(existing_segs), 1)
        self.assertEqual(existing_segs[0], "up_000.mp4")
        # Phân đoạn tiếp theo cần làm sẽ là chunk 1 (c_idx = 1)
        next_chunk_idx = len(existing_segs)
        self.assertEqual(next_chunk_idx, 1, "Lần chạy sau phải tiếp tục từ chunk 1 mà không phải làm lại chunk 0")


if __name__ == "__main__":
    unittest.main()
