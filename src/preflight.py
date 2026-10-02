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
    Context manager tự động ngăn máy tính sleep khi đang chạy tác vụ nặng trên macOS.
    Tự động thoát khi tiến trình hoàn tất hoặc bị ngắt (Ctrl+C).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and (sys.platform == "darwin")
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self):
        if not self.enabled:
            return self

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

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Dừng tiến trình caffeinate an toàn"""
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


def check_power_source() -> Dict[str, Any]:
    """Kiểm tra nguồn điện máy tính (đang cắm sạc AC hay dùng Pin)"""
    if sys.platform != "darwin":
        return {"is_ac": True, "details": "Non-macOS"}

    try:
        res = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True)
        output = res.stdout
        is_ac = "AC Power" in output

        # Tìm tỷ lệ % pin nếu có
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
    """Gửi thông báo native lên macOS Notification Center kèm âm thanh"""
    if sys.platform != "darwin":
        return

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
