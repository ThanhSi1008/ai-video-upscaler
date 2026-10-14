# -*- coding: utf-8 -*-
"""
src/logger.py
Quản lý hiển thị terminal và ghi file log có cấu trúc:
- Tự động tạo file log chuyên dụng theo tên video và timestamp: logs/<name>_<timestamp>.log
- Tự động duy trì symlink logs/latest.log trỏ tới lần chạy mới nhất
- Lưu chi tiết viễn trắc từng chunk (thời gian trích xuất, FPS AI, VideoToolbox encode)
- Thu thập và lưu bảng tổng kết Benchmark thực tế
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

from src.config import LOGS_DIR, HEVC_ENCODER, DEVICE_NAME


def format_time(seconds: float) -> str:
    """Định dạng thời gian sang chuỗi trực quan: mm:ss hoặc hh:mm:ss"""
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}h {m:02d}m {s:02d}s"
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}m {s:02d}s"


def format_bytes(bytes_count: int) -> str:
    """Định dạng dung lượng bytes sang MB hoặc GB"""
    mb = bytes_count / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def render_progress(current: int, total: int, prefix: str = "", suffix: str = "", length: int = 25):
    """Vẽ thanh tiến trình terminal mượt mà"""
    percent = (current / total) * 100 if total > 0 else 0
    filled = int(length * current // total) if total > 0 else 0
    bar = "█" * filled + "░" * (length - filled)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent:5.1f}% {suffix}")
    sys.stdout.flush()


import subprocess
import shutil

def get_swap_usage() -> str:
    """Lấy thông số bộ nhớ RAM và swap/pagefile hiện tại"""
    try:
        import psutil
        v = psutil.virtual_memory()
        s = psutil.swap_memory()
        return f"RAM: {v.percent}% used ({v.used // (1024*1024)}MB/{v.total // (1024*1024)}MB) | Pagefile: {s.percent}% used ({s.used // (1024*1024)}MB/{s.total // (1024*1024)}MB)"
    except Exception:
        pass
    try:
        res = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "Memory/Swap: N/A"

def get_thermal_state() -> str:
    """Lấy trạng thái nhiệt và throttle từ phần cứng"""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        temp = res.stdout.strip()
        if temp:
            return f"GPU {temp}°C (Nominal)"
    except Exception:
        pass
    try:
        res = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, check=True)
        lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip() and not line.startswith("Note:")]
        return " | ".join(lines) if lines else "Nominal"
    except Exception:
        return "Nominal"


class RunLogger:
    """
    Quản lý luồng log song song ra Console và File log có cấu trúc.
    Tự động cập nhật file logs/latest.log để dễ dàng kiểm tra.
    """

    def __init__(self, video_name: str):
        os.makedirs(LOGS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = os.path.splitext(os.path.basename(video_name))[0]
        self.log_filename = f"{safe_name}_{timestamp}.log"
        self.log_filepath = os.path.join(LOGS_DIR, self.log_filename)
        self.latest_symlink = os.path.join(LOGS_DIR, "latest.log")

        # Khởi tạo file log
        with open(self.log_filepath, "w", encoding="utf-8") as f:
            f.write(f"# AI Video Upscaler 4K - Execution Log\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Video: {video_name}\n\n")

        # Tạo/cập nhật symlink hoặc file copy latest.log
        try:
            if os.path.lexists(self.latest_symlink):
                os.remove(self.latest_symlink)
            os.symlink(self.log_filename, self.latest_symlink)
        except Exception:
            try:
                shutil.copyfile(self.log_filepath, self.latest_symlink)
            except Exception:
                pass

    def _write_file(self, message: str):
        """Ghi dòng log kèm timestamp vào file"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(f"[{now}] {message}\n")
            # Đồng bộ sang latest.log nếu latest.log là regular file (Windows không có symlink)
            if os.path.exists(self.latest_symlink) and not os.path.islink(self.latest_symlink):
                try:
                    with open(self.latest_symlink, "a", encoding="utf-8") as lf:
                        lf.write(f"[{now}] {message}\n")
                except Exception:
                    pass
        except Exception:
            pass

    def info(self, message: str, to_console: bool = False):
        """Ghi log mức INFO"""
        self._write_file(f"[INFO] {message}")
        if to_console:
            print(f"ℹ️  {message}")

    def warn(self, message: str, to_console: bool = True):
        """Ghi log mức WARNING"""
        self._write_file(f"[WARN] {message}")
        if to_console:
            print(f"⚠️  {message}")

    def error(self, message: str, to_console: bool = True):
        """Ghi log mức ERROR"""
        self._write_file(f"[ERROR] {message}")
        if to_console:
            print(f"❌ {message}")

    def print_banner(self, info: Dict[str, Any]):
        """In banner ra Terminal và ghi đầy đủ vào file log"""
        banner_lines = [
            "=" * 72,
            f"🎬  AI VIDEO UPSCALER 4K — {DEVICE_NAME.upper()} & {HEVC_ENCODER.upper()}",
            "=" * 72,
            f"📁 Video nguồn:    {info['filename']} ({format_bytes(info['size_bytes'])})",
            f"📐 Độ phân giải:   {info['width']}x{info['height']} ➔ {info['target_w']}x{info['target_h']} (Chuẩn 4K UHD)",
            f"⏱️  Thời lượng:     {format_time(info['duration'])} ({info['duration']:.2f}s) @ {info['fps']:.3f} fps",
            f"🧠 AI Engine:      {info['model']} ({DEVICE_NAME} FP16)",
            f"🖼️  Định dạng đệm:  {info['frame_format'].upper()} (Lossless RGB24)",
            f"💾 Bộ nhớ đệm:     {info['storage_info']}",
            f"🎞️  Mã hóa video:   {HEVC_ENCODER} (10-bit Main 10 @ {info['bitrate']}, BT.709 Limited)",
            f"🔊 Streams gốc:    {info['audio_count']} Audio, {info['sub_count']} Subtitles, {info['attach_count']} Fonts, {info['chap_count']} Chapters (100% Stream Copy)",
            f"🎯 File xuất:      {info['output_path']}",
            f"📝 File log:       logs/{self.log_filename} (Symlink: logs/latest.log)",
            "=" * 72
        ]

        # In ra console
        print("\n" + "\n".join(banner_lines) + "\n")

        # Ghi vào file log
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write("\n".join(banner_lines) + "\n\n")

    def log_chunk_done(
        self,
        chunk_idx: int,
        total_chunks: int,
        frames: int,
        ai_time: float,
        ai_fps: float,
        enc_time: float,
        total_chunk_time: float,
        output_seg: str,
        t0_seconds: Optional[float] = None,
        steady_fps: Optional[float] = None,
        rss_mb: Optional[float] = None
    ):
        """Ghi lại chi tiết viễn trắc của từng phân đoạn hoàn thành"""
        seg_size = os.path.getsize(output_seg) if os.path.exists(output_seg) else 0
        extra_info = []
        if t0_seconds is not None:
            extra_info.append(f"T0: {t0_seconds:.2f}s")
        if steady_fps is not None and steady_fps > 0:
            extra_info.append(f"Steady: {steady_fps:.2f} fps")
        if rss_mb is not None and rss_mb > 0:
            extra_info.append(f"Peak RSS: {rss_mb:.1f} MB")
        
        extra_str = f" | {', '.join(extra_info)}" if extra_info else ""
        log_entry = (
            f"[CHUNK {chunk_idx + 1:02d}/{total_chunks:02d}] Hoàn tất {frames} frames "
            f"(AI GPU: {ai_time:.2f}s @ {ai_fps:.2f} fps{extra_str} | Encode: {enc_time:.2f}s | "
            f"Tổng: {total_chunk_time:.2f}s | File: {os.path.basename(output_seg)} {format_bytes(seg_size)})"
        )
        self._write_file(log_entry)

    def print_benchmark_summary(self, stats: Dict[str, Any]):
        """In bảng tổng kết Benchmark ra console và lưu vĩnh viễn vào file log"""
        dev_name = stats.get('device_name') or DEVICE_NAME
        summary_lines = [
            "=" * 72,
            "🎉  UPSCALE 4K HOÀN TẤT THÀNH CÔNG!",
            "=" * 72,
            f"📦 File xuất thành phẩm: {stats['output_path']}",
            f"💾 Dung lượng video:     {format_bytes(stats['final_size_bytes'])}",
            f"⏱️  Tổng thời gian:       {format_time(stats['total_seconds'])} ({stats['total_seconds']:.1f}s)",
            f"🎞️  Tổng số khung hình:   {stats['total_frames']} frames",
            f"🚀 Tốc độ trung bình:    {stats['avg_fps']:.2f} fps",
            f"⚡ Tốc độ suy luận AI:   {stats['ai_fps']:.2f} fps ({dev_name})",
            f"🎯 Tối ưu Deduplication: {stats.get('dedup_info', 'Không kích hoạt')}",
            f"🛡️  Độ bền ổ cứng SSD:   {stats['ssd_protection']}",
        ]
        if "swap_info" in stats:
            summary_lines.append(f"📊 System Swap:          {stats['swap_info']}")
        if "thermal_info" in stats:
            summary_lines.append(f"🌡️  Nhiệt độ / Throttle:  {stats['thermal_info']}")
        if "stage_timings" in stats:
            summary_lines.append("⏱️  Thời gian theo Stage:")
            for s_name, s_sec in stats["stage_timings"].items():
                summary_lines.append(f"   • {s_name}: {s_sec:.2f}s")
                
        summary_lines.extend([
            f"📝 Nhật ký chi tiết:     logs/{self.log_filename}",
            "=" * 72
        ])

        print("\n" + "\n".join(summary_lines))
        if sys.platform == "win32":
            print(f"\n💡 Thưởng thức ngay video 4K chất lượng cao:")
            print(f"   start \"\" \"{stats['output_path']}\"\n")
        else:
            print(f"\n💡 Thưởng thức ngay với màn hình Liquid Retina XDR / Mini-LED:")
            print(f"   open -a IINA \"{stats['output_path']}\"\n")

        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(summary_lines) + "\n")
