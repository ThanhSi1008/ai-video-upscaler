# -*- coding: utf-8 -*-
"""
src/ramdisk.py
Quản lý RAM Disk ảo trên macOS để đệm khung hình trung gian (Zero NVMe SSD Wear).
Tự động dọn dẹp và unmount ngay cả khi gặp lỗi hoặc người dùng bấm Ctrl+C.
"""

import os
import sys
import atexit
import signal
import shutil
import tempfile
import subprocess
from typing import Optional


def cleanup_orphan_ramdisks():
    """Tự động dọn dẹp các RAM disk UpscaleRAM_<pid> bị mồ côi từ tiến trình cũ đã thoát."""
    if sys.platform != "darwin":
        return
    vols_dir = "/Volumes"
    if not os.path.exists(vols_dir):
        return
    for item in os.listdir(vols_dir):
        if item.startswith("UpscaleRAM_"):
            parts = item.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                pid = int(parts[1])
                is_running = False
                try:
                    os.kill(pid, 0)
                    is_running = True
                except (OSError, ProcessLookupError):
                    is_running = False

                if not is_running:
                    mount_path = os.path.join(vols_dir, item)
                    try:
                        subprocess.run(
                            ["diskutil", "eject", mount_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=5
                        )
                    except Exception:
                        try:
                            subprocess.run(
                                ["hdiutil", "detach", mount_path, "-force"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=5
                            )
                        except Exception:
                            pass


class RAMDisk:
    """
    Context manager quản lý RAM Disk HFS+ trên macOS.
    Nếu hệ thống không cho phép hoặc người dùng tắt cờ, tự động fallback về thư mục tạm trên SSD.
    """

    def __init__(self, size_mb: int = 3072, enabled: bool = True):
        self.size_mb = size_mb
        self.enabled = enabled
        self.device: Optional[str] = None
        self.mount_path: Optional[str] = None
        self.is_ramdisk: bool = False
        self._cleaned_up: bool = False

    def __enter__(self) -> str:
        # Tự động dọn dẹp các RAM disk mồ côi từ phiên trước (nếu có)
        cleanup_orphan_ramdisks()

        if self.enabled and sys.platform == "darwin":
            try:
                # 1. Tạo RAM block device (512 bytes per sector)
                sectors = (self.size_mb * 1024 * 1024) // 512
                attach_cmd = ["hdiutil", "attach", "-nomount", f"ram://{sectors}"]
                res = subprocess.run(
                    attach_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.device = res.stdout.strip().split()[0]

                # 2. Định dạng HFS+ volume
                vol_name = f"UpscaleRAM_{os.getpid()}"
                erase_cmd = ["diskutil", "erasevolume", "HFS+", vol_name, self.device]
                subprocess.run(
                    erase_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )

                self.mount_path = f"/Volumes/{vol_name}"
                if os.path.exists(self.mount_path):
                    self.is_ramdisk = True
                    self._register_exit_hooks()
                    return self.mount_path
            except Exception as e:
                # Fallback nếu RAM disk không thể tạo được
                self._safe_eject()
                self.is_ramdisk = False

        # Fallback về thư mục đệm thông thường
        self.mount_path = tempfile.mkdtemp(prefix="upscale_temp_")
        self.is_ramdisk = False
        self._register_exit_hooks()
        return self.mount_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Dọn dẹp và ngắt kết nối RAM disk hoặc thư mục tạm"""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if self.is_ramdisk and self.device:
            self._safe_eject()
        elif self.mount_path and os.path.exists(self.mount_path):
            shutil.rmtree(self.mount_path, ignore_errors=True)

    def _safe_eject(self):
        """Tháo kết nối an toàn thiết bị RAM disk"""
        if not self.device:
            return
        try:
            subprocess.run(
                ["diskutil", "eject", self.device],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
        except Exception:
            try:
                subprocess.run(
                    ["hdiutil", "detach", self.device, "-force"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5
                )
            except Exception:
                pass
        self.device = None

    def _register_exit_hooks(self):
        """Đăng ký hook thoát để tránh rò rỉ RAM disk khi crash hoặc Ctrl+C"""
        atexit.register(self.cleanup)

        def _signal_handler(signum, frame):
            self.cleanup()
            sys.exit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _signal_handler)
            except (ValueError, OSError):
                pass
