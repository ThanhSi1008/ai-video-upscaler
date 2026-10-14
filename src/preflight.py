# -*- coding: utf-8 -*-
"""
src/preflight.py
Hệ thống kiểm tra an toàn trước khi chạy (Pre-flight Health Checks) và quản lý năng lượng:
1. Cơ chế chống ngủ máy tự động (caffeinate -dims)
2. Kiểm tra nguồn điện sạc (AC Power vs Battery)
3. Kiểm tra dung lượng ổ đĩa trống an toàn (Free Disk Space)
4. Gửi thông báo hệ thống macOS qua Notification Center và chuông báo khi hoàn thành
"""

import os
import sys
import atexit
import shutil
import subprocess
from typing import Dict, Any, Optional

from src.logger import format_bytes


class CaffeinateContext:
    """
    Context manager tự động ngăn máy tính sleep khi đang chạy tác vụ nặng:
    - macOS: caffeinate -dims -w <pid>
    - Windows: ctypes.windll.kernel32.SetThreadExecutionState
    Tự động thoát và phục hồi trạng thái khi tiến trình hoàn tất hoặc bị ngắt (Ctrl+C).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.process: Optional[subprocess.Popen] = None
        self._win_prev = None

    def __enter__(self):
        if not self.enabled:
            return self

        if sys.platform == "darwin":
            try:
                # -d: chống display sleep, -i: chống idle sleep, -m: chống disk sleep, -s: chống system sleep
                # -w <pid>: tự động hủy khi PID của Python kết thúc
                self.process = subprocess.Popen(
                    ["caffeinate", "-dims", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                atexit.register(self.cleanup)
            except Exception:
                self.process = None
        elif sys.platform == "win32":
            try:
                import ctypes
                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                ES_DISPLAY_REQUIRED = 0x00000002
                self._win_prev = ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                )
                atexit.register(self.cleanup)
            except Exception:
                pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Dừng tiến trình caffeinate hoặc khôi phục trạng thái power trên Windows an toàn"""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        if sys.platform == "win32":
            try:
                import ctypes
                ES_CONTINUOUS = 0x80000000
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            except Exception:
                pass


def check_power_source() -> Dict[str, Any]:
    """Kiểm tra nguồn điện máy tính (đang cắm sạc AC hay dùng Pin)"""
    if sys.platform == "darwin":
        try:
            res = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True)
            output = res.stdout
            is_ac = "AC Power" in output
            percent = None
            for token in output.split():
                if "%" in token:
                    percent = token.replace(";", "").replace("%", "")
                    break
            return {
                "is_ac": is_ac,
                "percent": percent,
                "raw": output.strip().splitlines()[0] if output else ""
            }
        except Exception:
            return {"is_ac": True, "details": "Unknown"}

    # Trên Windows / Linux: kiểm tra qua psutil.sensors_battery
    try:
        import psutil
        batt = psutil.sensors_battery()
        if batt is None:
            # Máy tính để bàn (như PC Intel i5-12600K) không có pin -> Luôn là AC Power
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


def send_macos_notification(title: str, message: str, sound: str = "Glass"):
    """Gửi thông báo hệ thống khi hoàn thành (macOS / Windows)"""
    if sys.platform == "darwin":
        script = f'display notification "{message}" with title "{title}" sound name "{sound}"'
        try:
            subprocess.run(
                ["osascript", "-e", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
        except Exception:
            pass
    elif sys.platform == "win32":
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
    print("\n🔍 ĐANG KIỂM TRA ĐIỀU KIỆN HỆ THỐNG TRƯỚC KHI CHẠY (PRE-FLIGHT CHECKS)...")

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
            f"   💡 Khuyến nghị: Hãy cắm sạc MagSafe / USB-C trước khi chạy dài."
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
