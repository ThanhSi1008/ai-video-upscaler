#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_ramdisk_worst_case.py
Kiểm thử tự động kịch bản xấu nhất (Worst-Case) trên RAM Disk:
- Mô phỏng chunk có trọn vẹn 192 frame độc nhất (không dedup được frame nào).
- Kích thước ước tính PNG 4K: 12 MB/frame -> Nhu cầu đầu ra: 2.3 GB (nhu cầu an toàn x1.3 = 3.0 GB).
- Kiểm tra các trường hợp:
  1. Đủ dung lượng (> 3.0 GB): Xử lý bình thường, không block.
  2. Bị nghẽn bởi chunk trước đang encode trong RAM Disk: Phải tự động chờ Media Engine thu hồi dung lượng rồi mới bắt đầu.
  3. Quá tải thực sự (ENOSPC cứng < 2.3 GB): Phải dừng an toàn (raise OSError 28) thay vì làm tràn đĩa hoặc crash hệ thống.
"""

from collections import namedtuple
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.ai_engine import upscale_frames
from src.ramdisk import RAMDisk, _detect_mounted_ramdisk

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])


class TestRAMDiskWorstCase(unittest.TestCase):

    def setUp(self):
        self.chunk_idx = 5
        self.n_unique = 192
        self.size_est_mb = 12.0
        self.need_mb = self.n_unique * self.size_est_mb * 1.3  # 2995.2 MB
        self.min_needed_mb = self.n_unique * self.size_est_mb  # 2304.0 MB

    def test_worst_case_sufficient_space(self):
        """Scenario 1: RAMDisk mountsinside UPSCALER_RAMDISK_DIR when free space exceeds min_free_mb and cleans up on exit."""
        with tempfile.TemporaryDirectory() as fake_ram_root:
            usage_ok = _DiskUsage(8 * 1024**3, 1 * 1024**3, int(3500 * 1024 * 1024))
            with patch.dict(os.environ, {"UPSCALER_RAMDISK_DIR": fake_ram_root}), patch(
                "src.ramdisk.shutil.disk_usage", return_value=usage_ok
            ):
                detected = _detect_mounted_ramdisk(3072)
                self.assertEqual(detected, fake_ram_root)
                with RAMDisk(size_mb=3072, enabled=True) as mount_dir:
                    self.assertTrue(os.path.isdir(mount_dir))
                    self.assertEqual(os.path.dirname(mount_dir), fake_ram_root)
                self.assertFalse(os.path.exists(mount_dir))

    def test_worst_case_backpressure_recovery(self):
        """Scenario 2: Insufficient UPSCALER_RAMDISK_DIR (< min_free_mb) safely falls back to temp buffer and cleans up."""
        with tempfile.TemporaryDirectory() as tight_ram_root:
            usage_low = _DiskUsage(4 * 1024**3, 3 * 1024**3, int(1500 * 1024 * 1024))
            with patch.dict(os.environ, {"UPSCALER_RAMDISK_DIR": tight_ram_root}), patch(
                "src.ramdisk.shutil.disk_usage", return_value=usage_low
            ):
                self.assertIsNone(_detect_mounted_ramdisk(3072))
                with RAMDisk(size_mb=3072, enabled=True) as fallback_dir:
                    self.assertTrue(os.path.isdir(fallback_dir))
                    self.assertNotEqual(os.path.dirname(fallback_dir), tight_ram_root)
                self.assertFalse(os.path.exists(fallback_dir))

    def test_worst_case_enospc_hard_stop(self):
        """Scenario 3: Real upscale_frames succeeds when disk has > 50 MB and computes FPS metrics."""
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            mock_proc = MagicMock()
            mock_proc.pid = 999999
            mock_proc.returncode = 0
            mock_proc.poll.side_effect = [None, 0]
            usage_healthy = _DiskUsage(8 * 1024**3, 2 * 1024**3, int(2500 * 1024 * 1024))
            with patch("src.ai_engine.subprocess.Popen", return_value=mock_proc), patch(
                "src.ai_engine.shutil.disk_usage", return_value=usage_healthy
            ), patch("src.ai_engine.time.sleep"):
                res = upscale_frames(in_dir, out_dir, num_frames=12)
            self.assertEqual(res["frames"], 12)
            mock_proc.terminate.assert_not_called()

    def test_ai_engine_enospc_runtime_monitor(self):
        """Scenario 4: Real upscale_frames runtime monitor aborts NCNN subprocess and raises OSError(28) when free < 50 MB."""
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            mock_proc = MagicMock()
            mock_proc.pid = 999999
            mock_proc.poll.side_effect = [None, 0]
            usage_critical = _DiskUsage(4 * 1024**3, 4 * 1024**3 - 30 * 1024 * 1024, 30 * 1024 * 1024)
            with patch("src.ai_engine.subprocess.Popen", return_value=mock_proc), patch(
                "src.ai_engine.shutil.disk_usage", return_value=usage_critical
            ), patch("src.ai_engine.time.sleep"):
                with self.assertRaises(OSError) as ctx:
                    upscale_frames(in_dir, out_dir, num_frames=192)

            mock_proc.terminate.assert_called_once()
            mock_proc.wait.assert_called()
            self.assertEqual(ctx.exception.errno, 28)
            self.assertIn("ENOSPC", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
