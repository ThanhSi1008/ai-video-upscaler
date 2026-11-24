"""Terminal telemetry, structured run logs, and atomic logs/live_status.json updates."""

import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Iterator, Optional

import src.config as _cfg
from src.config import LOGS_DIR as _CONFIG_LOGS_DIR

LOGS_DIR = _CONFIG_LOGS_DIR
_LAST_GRACEFUL_PAUSE_TRIGGERED: bool = False


def __getattr__(name: str) -> Any:
    if name in ("HEVC_ENCODER", "DEVICE_NAME"):
        return getattr(_cfg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _prune_old_run_logs(logs_dir: str, keep: int = 40, protect_filename: Optional[str] = None) -> None:
    """Best-effort retention prune keeping newest run logs while protecting active/recent files."""
    try:
        now_ts = time.time()
        candidates = []
        for entry in os.scandir(logs_dir):
            if not entry.is_file(follow_symlinks=False):
                continue
            fname = entry.name
            if not fname.endswith(".log") or fname == "latest.log" or fname == protect_filename:
                continue
            st = entry.stat(follow_symlinks=False)
            candidates.append((st.st_mtime, entry.path))
        if len(candidates) <= keep:
            return
        candidates.sort(key=lambda item: item[0], reverse=True)
        for mtime, fpath in candidates[keep:]:
            if (now_ts - mtime) >= 300.0:
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    except Exception:
        pass


def get_logs_dir() -> str:
    """Resolve active logs directory, honoring LOGS_DIR monkeypatches and UPSCALER_LOGS_DIR."""
    if LOGS_DIR != _CONFIG_LOGS_DIR:
        return LOGS_DIR
    env_dir = os.environ.get("UPSCALER_LOGS_DIR", "").strip()
    if env_dir:
        return os.path.abspath(os.path.expanduser(env_dir))
    return LOGS_DIR


def get_live_status_path() -> str:
    """Path to logs/live_status.json realtime status file for Web GUI and CLI."""
    return os.path.join(get_logs_dir(), "live_status.json")


def get_graceful_pause_flag_path() -> str:
    """Path to the flag file requesting graceful pause after completing current chunk."""
    return os.path.join(get_logs_dir(), ".graceful_pause.flag")


def request_graceful_pause(reason: str = "user_request") -> str:
    """Create a flag file requesting graceful pause at the next 8s chunk boundary (checkpoint safe)."""
    logs_dir = get_logs_dir()
    os.makedirs(logs_dir, exist_ok=True)
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
    """Format duration as mm:ss or hh:mm:ss."""
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
    """Format byte size into MB or GB."""
    mb = bytes_count / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def render_progress(current: int, total: int, prefix: str = "", suffix: str = "", length: int = 25):
    """Render an inline terminal progress bar."""
    percent = (current / total) * 100 if total > 0 else 0
    filled = int(length * current // total) if total > 0 else 0
    bar = "█" * filled + "░" * (length - filled)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent:5.1f}% {suffix}")
    sys.stdout.flush()


import subprocess
import shutil

def get_swap_usage() -> str:
    """Return formatted host RAM and swap/pagefile utilization."""
    try:
        import psutil
        v = psutil.virtual_memory()
        s = psutil.swap_memory()
        return f"RAM: {v.percent}% used ({v.used // (1024*1024)}MB/{v.total // (1024*1024)}MB) | Pagefile: {s.percent}% used ({s.used // (1024*1024)}MB/{s.total // (1024*1024)}MB)"
    except Exception:
        pass
    return "Memory/Swap: N/A"


def get_thermal_state() -> str:
    """Query GPU temperature from nvidia-smi with bounded timeout and hidden console window."""
    try:
        run_kwargs: Dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": 2.0,
            "check": True,
        }
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            run_kwargs["startupinfo"] = startupinfo
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            **run_kwargs,
        )
        temp = res.stdout.strip()
        if temp:
            return f"GPU {temp}°C (Nominal)"
    except Exception:
        pass
    return "Nominal"


class RunLogger:
    """Manage structured console/log-file output and atomic Web Studio telemetry JSON."""

    def __init__(self, video_name: str):
        logs_dir = get_logs_dir()
        os.makedirs(logs_dir, exist_ok=True)
        now_dt = datetime.now()
        timestamp = now_dt.strftime("%Y%m%d_%H%M%S")
        now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%S")
        safe_name = os.path.splitext(os.path.basename(video_name))[0]
        self.log_filename = f"{safe_name}_{timestamp}.log"
        self.log_filepath = os.path.join(logs_dir, self.log_filename)
        self.latest_symlink = os.path.join(logs_dir, "latest.log")
        self.live_status_path = get_live_status_path()
        self._chunk_map: Dict[int, Dict[str, Any]] = {}
        self._live_state: Dict[str, Any] = {
            "video_name": video_name,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "job_id": os.environ.get("UPSCALER_WEB_JOB_ID", "").strip() or None,
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
            "runtime_backend": "",
            "is_preview": False,
            "total_frames": 0,
            "unique_frames": 0,
            "dedup_ratio_percent": 0.0,
            "total_chunks": 0,
            "completed_chunks": 0,
            "resumed_chunks": [],
            "active_chunk": None,
            "active_chunk_fraction": 0.0,
            "chunks": [],
            "latest_fps": 0.0,
            "steady_fps": 0.0,
            "effective_fps": 0.0,
            "eta_seconds": None,
            "last_error": None,
            "log_file": self.log_filename,
        }

        with open(self.log_filepath, "w", encoding="utf-8") as f:
            f.write("# AI Video Upscaler 4K - Execution Log\n")
            f.write(f"# Started: {now_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Video: {video_name}\n\n")

        # Update latest.log via symlink when permitted, falling back to regular copy on Windows.
        try:
            if os.path.lexists(self.latest_symlink):
                os.remove(self.latest_symlink)
            os.symlink(self.log_filename, self.latest_symlink)
        except Exception:
            try:
                shutil.copyfile(self.log_filepath, self.latest_symlink)
            except Exception:
                pass

        self._last_intra_flush_ts: float = 0.0
        _prune_old_run_logs(logs_dir, keep=40, protect_filename=self.log_filename)
        self._flush_live_status()

    def _flush_live_status(self) -> None:
        """Atomically write logs/live_status.json (< 0.2ms)."""
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
        """Update run-level metadata fields in live_status.json."""
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
        """Update active pipeline stage and preserve last active stage on pause/error."""
        prev_stage = str(self._live_state.get("stage") or "")
        prev_active = self._live_state.get("active_chunk")
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
        if active_chunk != prev_active or active_chunk is None:
            self._live_state["active_chunk_fraction"] = 0.0
            self._last_intra_flush_ts = 0.0
        self._flush_live_status()

    def update_active_chunk_progress(
        self,
        chunk_idx: int,
        done_distinct: int,
        total_distinct: int,
        chunk_elapsed_sec: float,
        frames_written: int = 0,
        chunk_total_frames: int = 0,
        force: bool = False,
    ) -> None:
        """Flush throttled intra-chunk progress, live FPS, and ETA without altering completed_chunks."""
        now_m = time.monotonic()
        if (not force) and done_distinct < total_distinct and (now_m - self._last_intra_flush_ts) < 0.90:
            return
        self._last_intra_flush_ts = now_m

        tot_d = max(1, int(total_distinct))
        done_d = min(tot_d, max(0, int(done_distinct)))
        frac = min(0.99, max(0.0, float(done_d) / float(tot_d)))
        elap = max(0.001, float(chunk_elapsed_sec))

        self._live_state["active_chunk"] = int(chunk_idx)
        self._live_state["active_chunk_fraction"] = round(frac, 4)

        total_c = int(self._live_state.get("total_chunks") or 0)
        if total_c > 0:
            self._live_state["stage_detail"] = (
                f"Đang suy luận 4K phân đoạn {int(chunk_idx) + 1}/{total_c} "
                f"({done_d}/{tot_d} unique • {frac * 100.0:.0f}%)"
            )

        if done_d >= 2 and elap >= 0.20:
            live_ai_fps = float(done_d) / elap
            est_out_frames = (
                float(frames_written)
                if frames_written > 0
                else (frac * float(chunk_total_frames) if chunk_total_frames > 0 else float(done_d))
            )
            live_eff_fps = max(live_ai_fps, est_out_frames / elap)
            self._live_state["latest_fps"] = round(live_ai_fps, 2)
            if not self._chunk_map or float(self._live_state.get("steady_fps") or 0.0) <= 0.0:
                self._live_state["steady_fps"] = round(live_ai_fps, 2)
            self._live_state["effective_fps"] = round(live_eff_fps, 2)

            if total_c > 0:
                resumed_set = set(self._live_state.get("resumed_chunks") or [])
                done_c = len(set(self._chunk_map.keys()) | resumed_set)
                rem_chunks_f = max(0.0, float(total_c - done_c) - frac)
                recent_times = [
                    float(self._chunk_map[k]["total_time_s"])
                    for k in sorted(self._chunk_map.keys())[-8:]
                    if float(self._chunk_map[k].get("total_time_s") or 0.0) > 0.0
                ]
                cur_est_chunk_s = (elap / max(frac, 0.02)) + 1.0
                if recent_times:
                    avg_prior = sum(recent_times) / len(recent_times)
                    blend_w = min(0.5, frac * 0.6)
                    est_chunk_s = (1.0 - blend_w) * avg_prior + blend_w * cur_est_chunk_s
                else:
                    est_chunk_s = cur_est_chunk_s
                self._live_state["eta_seconds"] = round(rem_chunks_f * est_chunk_s, 1)

        self._flush_live_status()

    @contextmanager
    def stage_guard(self) -> Iterator[None]:
        """Ensure KeyboardInterrupt, ENOSPC pauses, and errors update live_status.json."""
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

    def _append_log_and_mirror(self, raw_text: str) -> None:
        """Append raw_text to log_filepath and mirror to latest.log when it is a regular file."""
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(raw_text)
            if os.path.exists(self.latest_symlink) and not os.path.islink(self.latest_symlink):
                try:
                    with open(self.latest_symlink, "a", encoding="utf-8") as lf:
                        lf.write(raw_text)
                except Exception:
                    pass
        except Exception:
            pass

    def _write_file(self, message: str):
        """Append a timestamped log line to the log file."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_log_and_mirror(f"[{now}] {message}\n")

    def info(self, message: str, to_console: bool = False):
        """Log an INFO message."""
        self._write_file(f"[INFO] {message}")
        if to_console:
            print(f"ℹ️  {message}")
        elif self._live_state.get("job_id") and (
            "[ZERO-DISK SANITY" in message
            or "[OUTPUT SANITY" in message
            or "[CUDA MICRO-TIMINGS]" in message
        ):
            print(f"ℹ️  {message}", flush=True)

    def warn(self, message: str, to_console: bool = True):
        """Log a WARNING message."""
        self._write_file(f"[WARN] {message}")
        if to_console:
            print(f"⚠️  {message}")

    warning = warn

    def error(self, message: str, to_console: bool = True):
        """Log an ERROR message and update last_error in live_status.json."""
        self._write_file(f"[ERROR] {message}")
        self._live_state["last_error"] = message
        self._flush_live_status()
        if to_console:
            print(f"❌ {message}")

    def print_banner(self, info: Dict[str, Any]):
        """Print run startup banner to console and append to the log file."""
        dev_name = str(globals().get("DEVICE_NAME") or getattr(_cfg, "DEVICE_NAME", "CUDA"))
        hevc_enc = str(globals().get("HEVC_ENCODER") or getattr(_cfg, "HEVC_ENCODER", "hevc_nvenc"))
        frame_desc = str(
            info.get("frame_format_desc")
            or f"{str(info.get('frame_format', 'png')).upper()} (Lossless RGB24)"
        )
        encode_desc = str(info.get("encode_desc") or info.get("bitrate") or "")
        banner_lines = [
            "=" * 72,
            f"🎬  AI VIDEO UPSCALER 4K — {dev_name.upper()} & {hevc_enc.upper()}",
            "=" * 72,
            f"📁 Video nguồn:    {info['filename']} ({format_bytes(info['size_bytes'])})",
            f"📐 Độ phân giải:   {info['width']}x{info['height']} ➔ {info['target_w']}x{info['target_h']} (Chuẩn 4K UHD)",
            f"⏱️  Thời lượng:     {format_time(info['duration'])} ({info['duration']:.2f}s) @ {info['fps']:.3f} fps",
            f"🧠 AI Engine:      {info['model']} ({dev_name} FP16)",
            f"🖼️  Định dạng đệm:  {frame_desc}",
            f"💾 Bộ nhớ đệm:     {info['storage_info']}",
            f"🎞️  Mã hóa video:   {hevc_enc} (10-bit Main 10 @ {encode_desc}, BT.709 Limited)",
            f"🔊 Streams gốc:    {info['audio_count']} Audio, {info['sub_count']} Subtitles, {info['attach_count']} Fonts, {info['chap_count']} Chapters (100% Stream Copy)",
            f"🎯 File xuất:      {info['output_path']}",
            f"📝 File log:       logs/{self.log_filename} (Symlink: logs/latest.log)",
            "=" * 72
        ]

        print("\n" + "\n".join(banner_lines) + "\n")
        self._append_log_and_mirror("\n".join(banner_lines) + "\n\n")

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
        """Record per-chunk telemetry to log and live_status.json."""
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
        if self._live_state.get("job_id"):
            print(log_entry, flush=True)

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
        if self._live_state.get("active_chunk") == int(chunk_idx) or self._live_state.get("active_chunk") is None:
            self._live_state["active_chunk_fraction"] = 0.0
        active_frac = float(self._live_state.get("active_chunk_fraction") or 0.0)
        resumed_set = set(self._live_state.get("resumed_chunks") or [])
        completed_total = len(set(self._chunk_map.keys()) | resumed_set)
        remaining_chunks_f = max(0.0, float(int(total_chunks) - completed_total) - active_frac)
        recent_times = [
            float(self._chunk_map[k]["total_time_s"])
            for k in sorted(self._chunk_map.keys())[-8:]
            if float(self._chunk_map[k].get("total_time_s") or 0.0) > 0.0
        ]
        avg_chunk_time = (sum(recent_times) / len(recent_times)) if recent_times else float(total_chunk_time)
        eta_sec = round(remaining_chunks_f * avg_chunk_time, 1) if avg_chunk_time > 0 else None

        self._live_state["total_chunks"] = int(total_chunks)
        self._live_state["completed_chunks"] = completed_total
        self._live_state["chunks"] = [self._chunk_map[k] for k in sorted(self._chunk_map.keys())]
        self._live_state["latest_fps"] = round(float(ai_fps), 2)
        self._live_state["steady_fps"] = round(float(s_fps), 2)
        self._live_state["effective_fps"] = round(float(eff_fps), 2)
        self._live_state["eta_seconds"] = eta_sec
        self._flush_live_status()

    def print_benchmark_summary(self, stats: Dict[str, Any]):
        """Print final benchmark summary table to console and append to the log file."""
        dev_name = stats.get("device_name") or globals().get("DEVICE_NAME") or getattr(_cfg, "DEVICE_NAME", "CUDA")
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
        print("\n💡 Thưởng thức ngay video 4K chất lượng cao:")
        print(f"   start \"\" \"{stats['output_path']}\"\n")

        self._append_log_and_mirror("\n" + "\n".join(summary_lines) + "\n")

        self._live_state["stage"] = "completed"
        self._live_state["stage_detail"] = "Hoàn tất xuất xưởng 4K thành công!"
        self._live_state["active_chunk"] = None
        self._live_state["active_chunk_fraction"] = 0.0
        self._live_state["eta_seconds"] = 0.0
        self._flush_live_status()
