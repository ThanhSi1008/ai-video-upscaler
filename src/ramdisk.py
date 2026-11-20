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
from typing import Any, Optional


def _detect_mounted_ramdisk(min_free_mb: int) -> Optional[str]:
    """
    Phát hiện ổ đĩa RAM Disk thực tế trên Windows (qua biến môi trường UPSCALER_RAMDISK_DIR
    hoặc ổ R:/Z: có nhãn chứa 'RAM' như ImDisk/OSFMount) nếu đủ dung lượng trống.
    """
    env_dir = os.environ.get("UPSCALER_RAMDISK_DIR", "").strip()
    if env_dir:
        try:
            os.makedirs(env_dir, exist_ok=True)
            if (shutil.disk_usage(env_dir).free / (1024 * 1024)) >= min_free_mb:
                return env_dir
        except Exception:
            pass

    if sys.platform == "win32" and not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            for drv in ("R:\\", "Z:\\"):
                if not os.path.exists(drv):
                    continue
                vol_buf = ctypes.create_unicode_buffer(261)
                ok = kernel32.GetVolumeInformationW(
                    ctypes.c_wchar_p(drv),
                    vol_buf,
                    ctypes.sizeof(vol_buf),
                    None,
                    None,
                    None,
                    None,
                    0,
                )
                if ok and "RAM" in (vol_buf.value or "").upper():
                    if (shutil.disk_usage(drv).free / (1024 * 1024)) >= min_free_mb:
                        return drv
        except Exception:
            pass
    return None


def _resolve_fast_nvme_buffer_parent(min_free_mb: int) -> Optional[str]:
    """
    Định tuyến bộ đệm tạm sang ổ NVMe Gen4 tốc độ cao chứa workspace (ví dụ E: WD_BLACK SN7100 ~7.25 GB/s)
    thay vì đổ mặc định vào %TEMP% trên ổ C: tốc độ thấp hơn.
    Trong môi trường kiểm thử cô lập (PYTEST_CURRENT_TEST), giữ nguyên tempfile mặc định.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        from src.config import ROOT_DIR
        fast_parent = os.path.join(ROOT_DIR, "scratch", "rolling_buffer")
        os.makedirs(fast_parent, exist_ok=True)
        free_mb = shutil.disk_usage(fast_parent).free / (1024 * 1024)
        if free_mb >= max(min_free_mb * 2, 10240):
            return fast_parent
    except Exception:
        pass
    return None


def cleanup_orphan_ramdisks():
    """Dọn dẹp thư mục đệm tạm mồ côi trong scratch/rolling_buffer khi không có tiến trình GPU hoạt động."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        from src.config import ROOT_DIR, is_gpu_lock_held_by_other
        if is_gpu_lock_held_by_other() is not None:
            return
        fast_parent = os.path.join(ROOT_DIR, "scratch", "rolling_buffer")
        if os.path.isdir(fast_parent):
            for name in os.listdir(fast_parent):
                if name.startswith("upscale_temp_"):
                    shutil.rmtree(os.path.join(fast_parent, name), ignore_errors=True)
    except Exception:
        pass


class RAMDisk:
    """
    Context manager quản lý thư mục đệm khung hình trung gian trên Windows:
    1. Ưu tiên RAM Disk thật (ImDisk / UPSCALER_RAMDISK_DIR) nếu được mount.
    2. Ưu tiên ổ NVMe PCIe Gen4 tốc độ cao của workspace (scratch/rolling_buffer).
    3. Fallback an toàn về tempfile hệ thống, tự động dọn dẹp cuốn chiếu (Rolling Cleanup).
    """

    def __init__(self, size_mb: int = 3072, enabled: bool = True):
        self.size_mb = size_mb
        self.enabled = enabled
        self.device: Optional[str] = None
        self.mount_path: Optional[str] = None
        self.is_ramdisk: bool = False
        self._cleaned_up: bool = False
        self._prev_signal_handlers: dict[int, Any] = {}

    def __enter__(self) -> str:
        min_free = max(int(self.size_mb), 2048)
        ram_root = _detect_mounted_ramdisk(min_free) if self.enabled else None
        if ram_root:
            self.mount_path = tempfile.mkdtemp(prefix="upscale_temp_", dir=ram_root)
            self.is_ramdisk = True
        else:
            nvme_parent = _resolve_fast_nvme_buffer_parent(min_free) if self.enabled else None
            if nvme_parent:
                self.mount_path = tempfile.mkdtemp(prefix="upscale_temp_", dir=nvme_parent)
            else:
                self.mount_path = tempfile.mkdtemp(prefix="upscale_temp_")
            self.is_ramdisk = False
        self._register_exit_hooks()
        return self.mount_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Dọn dẹp thư mục đệm tạm và khôi phục signal handler trước đó."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        for sig, prev_handler in list(self._prev_signal_handlers.items()):
            try:
                if prev_handler is not None:
                    signal.signal(sig, prev_handler)
            except (ValueError, OSError):
                pass
        self._prev_signal_handlers.clear()

        if self.mount_path and os.path.exists(self.mount_path):
            shutil.rmtree(self.mount_path, ignore_errors=True)

    def _register_exit_hooks(self):
        """Đăng ký hook thoát để tránh rò rỉ thư mục tạm khi crash hoặc Ctrl+C"""
        atexit.register(self.cleanup)

        def _signal_handler(signum, frame):
            self.cleanup()
            if signum == signal.SIGINT:
                raise KeyboardInterrupt()
            sys.exit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.getsignal(sig)
                self._prev_signal_handlers[sig] = prev
                signal.signal(sig, _signal_handler)
            except (ValueError, OSError):
                pass
