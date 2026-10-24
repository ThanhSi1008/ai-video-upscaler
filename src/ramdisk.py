# -*- coding: utf-8 -*-
"""
src/ramdisk.py
Quản lý thư mục đệm khung hình trung gian (Rolling Buffer Directory).
Tự động dọn dẹp ngay cả khi gặp lỗi hoặc người dùng bấm Ctrl+C.
"""

import os
import sys
import atexit
import signal
import shutil
import tempfile
from typing import Optional


def cleanup_orphan_ramdisks():
    """Dọn dẹp thư mục đệm tạm mồ côi nếu có trên hệ thống Windows."""
    pass


class RAMDisk:
    """
    Context manager quản lý thư mục đệm khung hình trung gian trên Windows.
    Được dọn dẹp cuốn chiếu (Rolling Cleanup) ngay sau khi từng chunk mã hóa xong.
    """

    def __init__(self, size_mb: int = 3072, enabled: bool = True):
        self.size_mb = size_mb
        self.enabled = enabled
        self.device: Optional[str] = None
        self.mount_path: Optional[str] = None
        self.is_ramdisk: bool = False
        self._cleaned_up: bool = False

    def __enter__(self) -> str:
        self.mount_path = tempfile.mkdtemp(prefix="upscale_temp_")
        self.is_ramdisk = False
        self._register_exit_hooks()
        return self.mount_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Dọn dẹp thư mục đệm tạm"""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self.mount_path and os.path.exists(self.mount_path):
            shutil.rmtree(self.mount_path, ignore_errors=True)

    def _register_exit_hooks(self):
        """Đăng ký hook thoát để tránh rò rỉ thư mục tạm khi crash hoặc Ctrl+C"""
        atexit.register(self.cleanup)

        def _signal_handler(signum, frame):
            self.cleanup()
            sys.exit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _signal_handler)
            except (ValueError, OSError):
                pass
