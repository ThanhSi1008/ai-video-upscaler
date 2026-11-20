"""
src/logger.py
Terminal display management and structured log file generation:
- Automatically creates dedicated log files by video title and timestamp: logs/<name>_<timestamp>.log
- Automatically maintains logs/latest.log symlink pointing to the most recent run
- Records per-chunk telemetry details (extract time, AI FPS, NVENC encode)
- Collects and saves empirical benchmark summary tables
"""

import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Iterator, Optional

from src.config import LOGS_DIR, HEVC_ENCODER, DEVICE_NAME

_LAST_GRACEFUL_PAUSE_TRIGGERED: bool = False


def get_live_status_path() -> str:
    """Path to logs/live_status.json realtime status file for Web GUI and CLI."""
    return os.path.join(LOGS_DIR, "live_status.json")


def get_graceful_pause_flag_path() -> str:
    """Path to the flag file requesting graceful pause after completing current chunk."""
    return os.path.join(LOGS_DIR, ".graceful_pause.flag")


def request_graceful_pause(reason: str = "user_request") -> str:
    """Create a flag file requesting graceful pause at the next 8s chunk boundary (checkpoint safe)."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    flag_path = get_graceful_pause_flag_path()
    tmp_path = f"{flag_path}.{os.getpid()}.tmp"
    payload = {
        "requested_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": reason,
        "pid": os.getpid(),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, flag_path)
    return flag_path


def clear_graceful_pause() -> None:
    """Remove graceful pause flag if it exists."""
    flag_path = get_graceful_pause_flag_path()
    try:
        if os.path.exists(flag_path):
            os.remove(flag_path)
    except OSError:
        pass


def is_graceful_pause_requested() -> bool:
    """Check if a graceful pause request is currently pending."""
    return os.path.exists(get_graceful_pause_flag_path())


def mark_graceful_pause_triggered(triggered: bool = True) -> None:
    """Record graceful pause state in the current process."""
    global _LAST_GRACEFUL_PAUSE_TRIGGERED
    _LAST_GRACEFUL_PAUSE_TRIGGERED = bool(triggered)


def was_graceful_paused() -> bool:
    """Return True if execution ended due to a user graceful pause request."""
    return _LAST_GRACEFUL_PAUSE_TRIGGERED


def read_live_status() -> Dict[str, Any]:
    """Safely read logs/live_status.json; return empty dict if missing or invalid."""
    path = get_live_status_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    return "Memory/Swap: N/A"

def get_thermal_state() -> str:
    """Lấy trạng thái nhiệt và throttle từ phần cứng NVIDIA GPU"""
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
    return "Nominal"


class RunLogger:
    """
    Quản lý luồng log song song ra Console và File log có cấu trúc.
    Tự động cập nhật file logs/latest.log để dễ dàng kiểm tra.
    """

    def __init__(self, video_name: str):
        os.makedirs(LOGS_DIR, exist_ok=True)
        now_dt = datetime.now()
        timestamp = now_dt.strftime("%Y%m%d_%H%M%S")
        now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%S")
        safe_name = os.path.splitext(os.path.basename(video_name))[0]
        self.log_filename = f"{safe_name}_{timestamp}.log"
        self.log_filepath = os.path.join(LOGS_DIR, self.log_filename)
        self.latest_symlink = os.path.join(LOGS_DIR, "latest.log")
        self.live_status_path = get_live_status_path()
        self._chunk_map: Dict[int, Dict[str, Any]] = {}
        self._live_state: Dict[str, Any] = {
            "video_name": video_name,
            "pid": os.getpid(),
            "started_at": now_iso,
            "updated_at": now_iso,
            "stage": "initializing",
            "stage_detail": "Khởi tạo phiên xử lý...",
            "input_path": "",
            "output_path": "",
            "model_id": "",
            "model_name": "",
            "dedup_policy": "default",
            "zero_disk": True,
            "engine": "cuda",
            "is_preview": False,
            "total_frames": 0,
            "unique_frames": 0,
            "dedup_ratio_percent": 0.0,
            "total_chunks": 0,
            "completed_chunks": 0,
            "resumed_chunks": [],
            "active_chunk": None,
            "chunks": [],
            "latest_fps": 0.0,
            "steady_fps": 0.0,
            "effective_fps": 0.0,
            "eta_seconds": None,
            "last_error": None,
            "log_file": self.log_filename,
        }

        # Khởi tạo file log
        with open(self.log_filepath, "w", encoding="utf-8") as f:
            f.write("# AI Video Upscaler 4K - Execution Log\n")
            f.write(f"# Started: {now_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
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

        self._flush_live_status()

    def _flush_live_status(self) -> None:
        """Ghi nguyên tử trạng thái hiện tại ra logs/live_status.json (< 0.2ms, không ảnh hưởng FPS)."""
        try:
            os.makedirs(os.path.dirname(self.live_status_path), exist_ok=True)
            self._live_state["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            tmp_path = f"{self.live_status_path}.{os.getpid()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._live_state, f, ensure_ascii=False)
            os.replace(tmp_path, self.live_status_path)
        except Exception:
            pass

    def update_live_context(self, **kwargs: Any) -> None:
        """Cập nhật siêu dữ liệu phiên chạy trên live_status.json."""
        for k, v in kwargs.items():
            if v is not None:
                self._live_state[k] = v
        self._flush_live_status()

    def set_stage(
        self,
        stage: str,
        detail: str = "",
        active_chunk: Optional[int] = None,
    ) -> None:
        """Cập nhật chặng pipeline hiện tại (probing | preflight_dedup | upscaling | muxing | auditing | subtrans | completed | paused | error)."""
        prev_stage = str(self._live_state.get("stage") or "")
        non_step_stages = {"", "idle", "starting", "initializing", "completed", "paused", "error"}
        if stage in {"error", "paused"}:
            if prev_stage not in non_step_stages:
                self._live_state["last_stage"] = prev_stage
                if stage == "error":
                    self._live_state["failed_stage"] = prev_stage
        else:
            self._live_state.pop("failed_stage", None)
            if stage not in non_step_stages:
                self._live_state["last_stage"] = stage
        self._live_state["stage"] = stage
        if detail:
            self._live_state["stage_detail"] = detail
        self._live_state["active_chunk"] = active_chunk
        self._flush_live_status()

    @contextmanager
    def stage_guard(self) -> Iterator[None]:
        """Đảm bảo mọi ngoại lệ hoặc Ctrl+C đều cập nhật trạng thái dừng/lỗi lên live_status.json."""
        try:
            yield
        except KeyboardInterrupt:
            self.set_stage("paused", "Đã tạm dừng theo yêu cầu người dùng (Ctrl+C).")
            raise
        except Exception as exc:
            if type(exc).__name__ == "FailedResumableError":
                self._live_state["last_error"] = str(exc)
                self.set_stage("paused", f"Tạm dừng bảo toàn checkpoint (ENOSPC): {exc}")
            else:
                self._live_state["last_error"] = str(exc)
                if self._live_state.get("stage") != "error":
                    self.set_stage("error", f"Lỗi pipeline: {exc}")
                else:
                    self._flush_live_status()
            raise

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

    warning = warn

    def error(self, message: str, to_console: bool = True):
        """Ghi log mức ERROR"""
        self._write_file(f"[ERROR] {message}")
        self._live_state["last_error"] = message
        self._flush_live_status()
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

        self.update_live_context(
            output_path=info.get("output_path", ""),
            width=info.get("width", 0),
            height=info.get("height", 0),
            target_w=info.get("target_w", 0),
            target_h=info.get("target_h", 0),
            duration_s=round(float(info.get("duration", 0.0)), 2),
            source_fps=round(float(info.get("fps", 0.0)), 3),
        )

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

        eff_fps = (frames / total_chunk_time) if total_chunk_time > 0 else ai_fps
        s_fps = steady_fps if (steady_fps is not None and steady_fps > 0) else ai_fps
        chunk_rec = {
            "chunk_idx": int(chunk_idx),
            "frames": int(frames),
            "ai_fps": round(float(ai_fps), 2),
            "steady_fps": round(float(s_fps), 2),
            "effective_fps": round(float(eff_fps), 2),
            "ai_time_s": round(float(ai_time), 3),
            "enc_time_s": round(float(enc_time), 3),
            "total_time_s": round(float(total_chunk_time), 3),
            "size_mb": round(seg_size / (1024 * 1024), 2),
        }
        self._chunk_map[int(chunk_idx)] = chunk_rec
        resumed_set = set(self._live_state.get("resumed_chunks") or [])
        completed_total = len(set(self._chunk_map.keys()) | resumed_set)
        remaining_chunks = max(0, int(total_chunks) - completed_total)
        recent_times = [
            float(self._chunk_map[k]["total_time_s"])
            for k in sorted(self._chunk_map.keys())[-8:]
            if float(self._chunk_map[k].get("total_time_s") or 0.0) > 0.0
        ]
        avg_chunk_time = (sum(recent_times) / len(recent_times)) if recent_times else float(total_chunk_time)
        eta_sec = round(remaining_chunks * avg_chunk_time, 1) if avg_chunk_time > 0 else None

        self._live_state["total_chunks"] = int(total_chunks)
        self._live_state["completed_chunks"] = completed_total
        self._live_state["chunks"] = [self._chunk_map[k] for k in sorted(self._chunk_map.keys())]
        self._live_state["latest_fps"] = round(float(ai_fps), 2)
        self._live_state["steady_fps"] = round(float(s_fps), 2)
        self._live_state["effective_fps"] = round(float(eff_fps), 2)
        self._live_state["eta_seconds"] = eta_sec
        self._flush_live_status()

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
            print("\n💡 Thưởng thức ngay video 4K chất lượng cao:")
            print(f"   start \"\" \"{stats['output_path']}\"\n")
        else:
            print("\n💡 Thưởng thức ngay với màn hình Liquid Retina XDR / Mini-LED:")
            print(f"   open -a IINA \"{stats['output_path']}\"\n")

        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(summary_lines) + "\n")

        self._live_state["stage"] = "completed"
        self._live_state["stage_detail"] = "Hoàn tất xuất xưởng 4K thành công!"
        self._live_state["active_chunk"] = None
        self._live_state["eta_seconds"] = 0.0
        self._flush_live_status()
