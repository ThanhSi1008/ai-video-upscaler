# -*- coding: utf-8 -*-
"""
src/preflight.py
Hệ thống kiểm tra an toàn trước khi chạy (Pre-flight Health Checks) và quản lý năng lượng:
1. Cơ chế chống ngủ máy tự động (Windows SetThreadExecutionState) & khóa P-Core HighQoS cho Intel Gen 12+
2. Kiểm tra nguồn điện sạc (AC Power vs Battery)
3. Kiểm tra dung lượng ổ đĩa trống an toàn (Free Disk Space)
4. Gửi thông báo hệ thống và âm báo khi hoàn thành
"""

import os
import sys
import atexit
import shutil
import subprocess
from typing import Dict, Any, Optional

from src.logger import format_bytes


def configure_alder_lake_pcore_qos(enable_boost: bool = True) -> Optional[int]:
    """
    Tắt chế độ EcoQoS / Execution Speed Throttling của Windows 11 Thread Director
    trên vi xử lý lai Intel Gen 12+ (Core i5-12600K 6P+4E), ngăn hệ điều hành
    đẩy luồng điều phối CUDA & FFmpeg xuống nhân tiết kiệm điện E-Core khi thu nhỏ cửa sổ.
    Trả về PriorityClass trước đó để khôi phục khi kết thúc.
    """
    if sys.platform != "win32" or os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    prev_prio = None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        h_proc = kernel32.GetCurrentProcess()

        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            ]

        state = PROCESS_POWER_THROTTLING_STATE(
            Version=1,
            ControlMask=0x1,  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            StateMask=0 if enable_boost else 0x1,
        )
        kernel32.SetProcessInformation(
            h_proc,
            4,  # ProcessPowerThrottling
            ctypes.byref(state),
            ctypes.sizeof(state),
        )

        if enable_boost:
            ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
            NORMAL_PRIORITY_CLASS = 0x00000020
            cur = int(kernel32.GetPriorityClass(h_proc))
            if cur in (0, NORMAL_PRIORITY_CLASS):
                prev_prio = cur or NORMAL_PRIORITY_CLASS
                kernel32.SetPriorityClass(h_proc, ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    return prev_prio


class AntiSleepContext:
    """
    Context manager tự động ngăn Windows sleep khi đang chạy tác vụ nặng:
    Sử dụng Windows API native ctypes.windll.kernel32.SetThreadExecutionState + HighQoS P-Core.
    Tự động thoát và phục hồi trạng thái khi tiến trình hoàn tất hoặc bị ngắt (Ctrl+C).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._win_prev = None
        self._prev_prio = None

    def __enter__(self):
        if not self.enabled:
            return self

        if sys.platform == "win32":
            try:
                import ctypes
                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                ES_DISPLAY_REQUIRED = 0x00000002
                self._win_prev = ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                )
                self._prev_prio = configure_alder_lake_pcore_qos(enable_boost=True)
                atexit.register(self.cleanup)
            except Exception:
                pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Khôi phục trạng thái power trên Windows an toàn"""
        if sys.platform == "win32":
            try:
                import ctypes
                ES_CONTINUOUS = 0x80000000
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                if self._prev_prio is not None:
                    ctypes.windll.kernel32.SetPriorityClass(
                        ctypes.windll.kernel32.GetCurrentProcess(),
                        int(self._prev_prio),
                    )
                    self._prev_prio = None
            except Exception:
                pass


def check_power_source() -> Dict[str, Any]:
    """Kiểm tra nguồn điện máy tính qua psutil.sensors_battery"""
    try:
        import psutil
        batt = psutil.sensors_battery()
        if batt is None:
            # Máy tính để bàn Workstation (Intel Core i5-12600K) -> Luôn là AC Power
            return {"is_ac": True, "percent": None, "raw": "AC Desktop Power"}
        return {
            "is_ac": bool(batt.power_plugged),
            "percent": str(int(batt.percent)) if batt.percent is not None else None,
            "raw": f"{'AC Power' if batt.power_plugged else 'Battery'} ({batt.percent}%)"
        }
    except Exception:
        return {"is_ac": True, "details": "Desktop/Unknown"}


def check_disk_space(output_dir: str, duration_seconds: float, bitrate_str: str = "28M") -> Dict[str, Any]:
    """Kiểm tra dung lượng đĩa trống an toàn trước khi bắt đầu phiên chạy"""
    os.makedirs(output_dir, exist_ok=True)
    usage = shutil.disk_usage(output_dir)

    # Ước lượng dung lượng file xuất 4K (28Mbps ~ 3.5 MB/s)
    try:
        bitrate_mbps = float(bitrate_str.upper().replace("M", ""))
    except Exception:
        bitrate_mbps = 28.0

    est_file_size = int(duration_seconds * (bitrate_mbps * 1024 * 1024 / 8))
    # Yêu cầu an toàn: Cần ít nhất (dung lượng xuất * 2 + 5GB đệm)
    required_space = max(5 * 1024**3, est_file_size * 2 + 3 * 1024**3)

    is_safe = usage.free >= required_space

    return {
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "est_output_bytes": est_file_size,
        "required_bytes": required_space,
        "is_safe": is_safe
    }


def send_system_notification(title: str, message: str, sound: str = "Glass"):
    """Gửi âm báo hệ thống Windows khi hoàn thành"""
    if sys.platform == "win32":
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass


def run_preflight_checks(output_dir: str, duration_seconds: float, bitrate: str, logger=None) -> bool:
    """
    Thực thi toàn bộ các bước kiểm tra an toàn trước khi chạy.
    In thông báo rõ ràng ra Terminal và file log.
    """
    print("\n🔍 ĐANG KIỂM TRA ĐIỂM KIỂM SOÁT HỆ THỐNG TRƯỚC KHI CHẠY (PRE-FLIGHT CHECKS)...")

    # 1. Kiểm tra nguồn điện
    power = check_power_source()
    if power["is_ac"]:
        msg_power = "✅ Nguồn điện: Đang cắm sạc AC Power (Đủ điều kiện tải nặng an toàn)."
        print(msg_power)
        if logger:
            logger.info(msg_power)
    else:
        pct_info = f" ({power['percent']}%)" if power.get("percent") else ""
        msg_power = (
            f"⚠️  CẢNH BÁO NGUỒN ĐIỆN: Máy đang chạy bằng PIN{pct_info}!\n"
            f"   Phiên upscale AI tiêu tốn nhiều GPU và có thể làm cạn pin sập nguồn.\n"
            f"   💡 Khuyến nghị: Hãy cắm sạc nguồn AC trước khi chạy dài."
        )
        print(msg_power)
        if logger:
            logger.warn(msg_power)

    # 2. Kiểm tra dung lượng ổ đĩa
    disk = check_disk_space(output_dir, duration_seconds, bitrate)
    if disk["is_safe"]:
        msg_disk = f"✅ Dung lượng ổ đĩa trống: {format_bytes(disk['free_bytes'])} (Đủ điều kiện an toàn, ước tính cần ~{format_bytes(disk['required_bytes'])})."
        print(msg_disk)
        if logger:
            logger.info(msg_disk)
    else:
        msg_disk = (
            f"❌ CẢNH BÁO DUNG LƯỢNG ĐĨA: Ổ đĩa chỉ còn {format_bytes(disk['free_bytes'])} trống!\n"
            f"   Yêu cầu tối thiểu an toàn là {format_bytes(disk['required_bytes'])} để lưu file 4K và đệm phân đoạn.\n"
            f"   💡 Vui lòng dọn dẹp thêm ổ cứng trước khi bắt đầu."
        )
        print(msg_disk)
        if logger:
            logger.error(msg_disk)
        raise RuntimeError("Ổ đĩa không đủ dung lượng trống an toàn để bắt đầu!")

    print("🛡️  Toàn bộ kiểm tra an toàn đã hoàn tất thành công.\n")
    return True
