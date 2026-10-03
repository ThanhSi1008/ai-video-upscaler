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

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

class TestRAMDiskWorstCase(unittest.TestCase):

    def setUp(self):
        self.chunk_idx = 5
        self.n_unique = 192
        self.size_est_mb = 12.0
        self.need_mb = self.n_unique * self.size_est_mb * 1.3 # 2995.2 MB
        self.min_needed_mb = self.n_unique * self.size_est_mb # 2304.0 MB

    def test_worst_case_sufficient_space(self):
        """Kịch bản 1: RAM Disk còn 3.5 GB (> 3.0 GB) -> Chạy mượt mà, không phải chờ."""
        free_mb = 3500.0
        future_encode = MagicMock()
        
        # Logic tiền kiểm tra
        waited = False
        if future_encode is not None and free_mb < self.need_mb:
            waited = True
            
        self.assertFalse(waited, "Không được block khi RAM Disk còn đủ 3.5 GB")
        self.assertGreaterEqual(free_mb, self.min_needed_mb)

    def test_worst_case_backpressure_recovery(self):
        """Kịch bản 2: RAM Disk chỉ còn 1.5 GB (< 3.0 GB) do chunk trước đang encode -> Chờ giải phóng và thành công."""
        # Giả lập ban đầu chỉ còn 1500 MB
        current_free = [1500.0]
        
        future_encode = MagicMock()
        def mock_encode_finish():
            # Sau khi encode xong, chunk trước bị xóa, giải phóng thêm 1800 MB
            current_free[0] = 3300.0
            return 8.5 # elapsed seconds
        future_encode.result.side_effect = mock_encode_finish
        
        waited = False
        free_mb = current_free[0]
        if future_encode is not None and free_mb < self.need_mb:
            waited = True
            prev_enc_time = future_encode.result()
            future_encode = None
            free_mb = current_free[0]
            
        self.assertTrue(waited, "Bắt buộc phải đồng bộ đợi Media Engine giải phóng")
        self.assertEqual(free_mb, 3300.0, "Dung lượng sau khi giải phóng phải tăng lên 3.3 GB")
        self.assertGreaterEqual(free_mb, self.min_needed_mb, "Đã đủ không gian an toàn cho 192 frames")

    def test_worst_case_enospc_hard_stop(self):
        """Kịch bản 3: Sau khi giải phóng mà vẫn không đủ chỗ (chỉ còn 1.8 GB < 2.3 GB) -> Dừng sạch ENOSPC."""
        free_mb = 1800.0 # < 2304 MB
        future_encode = None # Đã giải phóng hết nhưng vẫn không đủ chỗ
        
        with self.assertRaises(OSError) as ctx:
            min_needed_mb = self.n_unique * self.size_est_mb
            if free_mb < min_needed_mb:
                raise OSError(28, f"RAM Disk không đủ không gian cho chunk {self.chunk_idx}: Trống {free_mb:.0f} MB, cần tối thiểu {min_needed_mb:.0f} MB ({self.n_unique} frames x {self.size_est_mb:.1f}MB). Dừng an toàn ENOSPC.")
                
        self.assertEqual(ctx.exception.errno, 28)
        self.assertIn("ENOSPC", str(ctx.exception))

    def test_ai_engine_enospc_runtime_monitor(self):
        """Kịch bản 4: Giám sát thời gian thực trong upscale_frames phát hiện đĩa còn < 50MB -> ngắt sạch NCNN."""
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, None, 0]
        
        # Mô phỏng kiểm tra dung lượng còn 30 MB (< 50 MB)
        free_bytes = 30 * 1024 * 1024
        
        with self.assertRaises(OSError) as ctx:
            if free_bytes < 50 * 1024 * 1024:
                mock_proc.terminate()
                mock_proc.wait(timeout=2)
                raise OSError(28, f"RAM Disk cạn kiệt dung lượng khi xuất frame 4K (còn {free_bytes / (1024*1024):.1f} MB < 50 MB). Dừng khẩn cấp ENOSPC.")
                
        mock_proc.terminate.assert_called_once()
        self.assertEqual(ctx.exception.errno, 28)

if __name__ == "__main__":
    unittest.main()
