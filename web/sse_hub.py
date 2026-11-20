# -*- coding: utf-8 -*-
"""
web/sse_hub.py
Trung tâm thu thập viễn trắc thời gian thực 1.0 Hz & phát luồng Server-Sent Events (SSE):
- Tái sử dụng default_gpu_query() (SW_HIDE) từ scripts/telemetry_collector.py.
- Kết hợp psutil đo CPU %, System RAM GB, tốc độ đọc/ghi NVMe MB/s và dung lượng trống ổ đĩa.
- Đồng bộ hai chiều CLI <-> Web GUI qua logs/live_status.json, .gpu_active.lock, .temp_* và logs/latest.log.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, AsyncIterator, Callable, Deque, Dict, List, Optional, Sequence

from scripts.telemetry_collector import default_gpu_query


def _query_gpu_static_info() -> Dict[str, Any]:
    """Đọc 1 lần thông tin tĩnh của GPU NVIDIA (tên card, driver, tổng VRAM MiB)."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
            startupinfo=startupinfo,
        )
        lines = res.stdout.strip().splitlines()
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 3:
                return {
                    "gpu_name": parts[0],
                    "driver_version": parts[1],
                    "vram_total_mib": float(parts[2] or 12288.0),
                }
    except Exception:
        pass
    return {
        "gpu_name": "NVIDIA RTX GPU",
        "driver_version": "N/A",
        "vram_total_mib": 12288.0,
    }


def check_active_gpu_lock(workspace_root: Path) -> Dict[str, Any]:
    """Kiểm tra file .gpu_active.lock và xác thực PID còn sống."""
    lock_file = workspace_root / ".gpu_active.lock"
    if not lock_file.is_file():
        return {"locked": False, "pid": None}
    try:
        raw = lock_file.read_text(encoding="utf-8").strip()
        if not raw:
            return {"locked": False, "pid": None}
        pid = int(raw.splitlines()[0].strip())
        import psutil
        if psutil.pid_exists(pid):
            return {"locked": True, "pid": pid}
    except Exception:
        pass
    return {"locked": False, "pid": None, "stale_lock": True}


_CHECKPOINT_VIDEO_SUFFIXES = {".mkv", ".mp4", ".webm", ".mov", ".avi", ".ts", ".m2ts"}


def _resolve_checkpoint_search_dir(workspace_root: Path, raw_dir_or_file: Optional[str]) -> Optional[Path]:
    if not raw_dir_or_file or not str(raw_dir_or_file).strip():
        return None
    raw = str(raw_dir_or_file).strip()
    if "\x00" in raw:
        return None
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = workspace_root / p
    try:
        resolved = p.resolve()
    except OSError:
        return None
    if resolved.suffix.lower() in _CHECKPOINT_VIDEO_SUFFIXES:
        return resolved.parent
    return resolved


def scan_temp_checkpoints(
    workspace_root: Path,
    extra_dirs: Optional[Sequence[Path]] = None,
) -> List[Dict[str, Any]]:
    """Quét danh sách các thư mục checkpoint .temp_* hiện có (không xóa hay chỉnh sửa)."""
    results: List[Dict[str, Any]] = []
    search_dirs: List[Path] = [workspace_root, workspace_root / "out"]

    # Bổ sung thư mục đích tùy chỉnh từ logs/live_status.json nếu người dùng dùng -o ngoài thư mục mặc định
    live_file = workspace_root / "logs" / "live_status.json"
    if live_file.is_file():
        try:
            ldata = json.loads(live_file.read_text(encoding="utf-8"))
            if isinstance(ldata, dict):
                out_dir = _resolve_checkpoint_search_dir(workspace_root, ldata.get("output_path"))
                if out_dir is not None:
                    search_dirs.append(out_dir)
        except Exception:
            pass

    if extra_dirs:
        for d in extra_dirs:
            if isinstance(d, Path):
                search_dirs.append(d)

    unique_dirs: List[Path] = []
    seen_dirs: set[str] = set()
    for cand_dir in search_dirs:
        try:
            norm_d = os.path.normcase(os.path.abspath(str(cand_dir)))
        except OSError:
            continue
        if norm_d not in seen_dirs:
            seen_dirs.add(norm_d)
            unique_dirs.append(cand_dir)

    seen: set[str] = set()

    for base in unique_dirs:
        if not base.is_dir():
            continue
        try:
            entries = list(os.scandir(base))
        except OSError:
            continue
        for entry in entries:
            if not entry.name.startswith(".temp_"):
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            norm = os.path.normcase(os.path.abspath(entry.path))
            if norm in seen:
                continue
            seen.add(norm)

            t_path = Path(entry.path)
            raw_segs_dir = t_path / "raw_segs"
            up_segs_dir = t_path / "upscaled_segs"
            manifest_path = t_path / "manifest.json"
            plan_path = t_path / "plan.json"

            raw_count = 0
            up_count = 0
            size_bytes = 0
            if raw_segs_dir.is_dir():
                for f in os.scandir(raw_segs_dir):
                    if f.name.startswith("seg_") and f.name.endswith(".mp4"):
                        raw_count += 1
                        try:
                            size_bytes += f.stat().st_size
                        except OSError:
                            pass
            if up_segs_dir.is_dir():
                for f in os.scandir(up_segs_dir):
                    if f.name.startswith("up_") and f.name.endswith(".mp4"):
                        up_count += 1
                        try:
                            size_bytes += f.stat().st_size
                        except OSError:
                            pass

            model_id = ""
            if manifest_path.is_file():
                try:
                    mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
                    model_id = str(mdata.get("model_id") or "")
                except Exception:
                    pass

            total_chunks = raw_count
            if plan_path.is_file() and total_chunks == 0:
                try:
                    pdata = json.loads(plan_path.read_text(encoding="utf-8"))
                    total_chunks = len(pdata.get("chunks") or [])
                except Exception:
                    pass

            results.append({
                "name": entry.name,
                "episode_name": entry.name[len(".temp_"):],
                "path": str(t_path),
                "completed_chunks": up_count,
                "total_chunks": total_chunks,
                "model_id": model_id,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
            })
    return results


class TelemetryHub:
    """Bộ thu thập viễn trắc 1.0 Hz & phân phối SSE tới mọi trình duyệt đang mở."""

    def __init__(
        self,
        workspace_root: Path,
        gpu_query_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        job_state_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        history_size: int = 120,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.gpu_query_fn = gpu_query_fn or default_gpu_query
        self.job_state_provider = job_state_provider
        self.history: Deque[Dict[str, Any]] = deque(maxlen=max(10, history_size))
        self.gpu_static = (
            _query_gpu_static_info()
            if gpu_query_fn is None
            else {"gpu_name": "Simulated RTX 3060 12GB", "driver_version": "Test", "vram_total_mib": 12288.0}
        )

        self._last_disk_time: float = 0.0
        self._last_read_bytes: int = 0
        self._last_write_bytes: int = 0
        self._counter_lock = threading.Lock()
        self._prime_counters()

        self._subscribers: set[asyncio.Queue[Optional[Dict[str, Any]]]] = set()
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._shutting_down: bool = False
        self._history_lock = threading.Lock()
        self._last_log_cursor: Dict[str, Any] = {"path": "", "prefix_len": 0, "sig": "", "offset": 0}
        self._prime_log_cursor()

    def _prime_counters(self) -> None:
        try:
            import psutil
            psutil.cpu_percent(interval=None)
            io = psutil.disk_io_counters()
            if io is not None:
                with self._counter_lock:
                    self._last_read_bytes = io.read_bytes
                    self._last_write_bytes = io.write_bytes
                    self._last_disk_time = time.perf_counter()
        except Exception:
            pass

    def sample_telemetry(self, record_history: bool = True) -> Dict[str, Any]:
        """Thu thập 1 mẫu 17+ chỉ số phần cứng và lưu vào vòng đệm 120 giây khi record_history=True."""
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        try:
            gpu = self.gpu_query_fn()
        except Exception:
            gpu = default_gpu_query()

        cpu_pct = 0.0
        ram_used_gb = 0.0
        ram_total_gb = 32.0
        ssd_read_mb_s = 0.0
        ssd_write_mb_s = 0.0

        try:
            import psutil
            cpu_pct = round(float(psutil.cpu_percent(interval=None)), 1)
            mem = psutil.virtual_memory()
            ram_used_gb = round(float(mem.used) / (1024 ** 3), 2)
            ram_total_gb = round(float(mem.total) / (1024 ** 3), 1)

            now_perf = time.perf_counter()
            io = psutil.disk_io_counters()
            if io is not None:
                with self._counter_lock:
                    if self._last_disk_time > 0:
                        dt = now_perf - self._last_disk_time
                        if dt > 0.05:
                            d_r = max(0, io.read_bytes - self._last_read_bytes)
                            d_w = max(0, io.write_bytes - self._last_write_bytes)
                            ssd_read_mb_s = round(d_r / (dt * 1024 * 1024), 2)
                            ssd_write_mb_s = round(d_w / (dt * 1024 * 1024), 2)
                            self._last_read_bytes = io.read_bytes
                            self._last_write_bytes = io.write_bytes
                            self._last_disk_time = now_perf
                    else:
                        self._last_read_bytes = io.read_bytes
                        self._last_write_bytes = io.write_bytes
                        self._last_disk_time = now_perf
        except Exception:
            pass

        disk_free_gb = 0.0
        disk_total_gb = 0.0
        try:
            usage = shutil.disk_usage(self.workspace_root)
            disk_free_gb = round(usage.free / (1024 ** 3), 2)
            disk_total_gb = round(usage.total / (1024 ** 3), 2)
        except OSError:
            pass

        vram_used = float(gpu.get("vram_used_mib", 0.0))
        vram_free = float(gpu.get("vram_free_mib", 0.0))
        vram_total = (
            round(vram_used + vram_free, 0)
            if (vram_used + vram_free) > 0
            else float(self.gpu_static.get("vram_total_mib", 12288.0))
        )

        sample: Dict[str, Any] = {
            "timestamp": now_iso,
            "pstate": str(gpu.get("pstate", "N/A")),
            "gpu_util_pct": float(gpu.get("gpu_util_pct", 0.0)),
            "mem_util_pct": float(gpu.get("mem_util_pct", 0.0)),
            "enc_util_pct": float(gpu.get("enc_util_pct", 0.0)),
            "dec_util_pct": float(gpu.get("dec_util_pct", 0.0)),
            "mem_clock_mhz": float(gpu.get("mem_clock_mhz", 0.0)),
            "core_clock_mhz": float(gpu.get("core_clock_mhz", 0.0)),
            "power_w": float(gpu.get("power_w", 0.0)),
            "temp_c": float(gpu.get("temp_c", 0.0)),
            "vram_used_mib": vram_used,
            "vram_free_mib": vram_free,
            "vram_total_mib": vram_total,
            "cpu_pct": cpu_pct,
            "ram_used_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
            "ssd_read_mb_s": ssd_read_mb_s,
            "ssd_write_mb_s": ssd_write_mb_s,
            "disk_free_gb": disk_free_gb,
            "disk_total_gb": disk_total_gb,
        }
        if record_history:
            with self._history_lock:
                self.history.append(sample)
        return sample

    def read_pipeline_state(self) -> Dict[str, Any]:
        """Đọc trạng thái pipeline thời gian thực (tương thích cả CLI ngoài lẫn Web Job)."""
        logs_dir = self.workspace_root / "logs"
        live_file = logs_dir / "live_status.json"
        pause_flag = logs_dir / ".graceful_pause.flag"
        gpu_lock = check_active_gpu_lock(self.workspace_root)
        live_file_mtime = 0.0
        live_data: Dict[str, Any] = {}
        if live_file.is_file():
            try:
                live_file_mtime = live_file.stat().st_mtime
                parsed = json.loads(live_file.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    live_data = parsed
            except Exception:
                pass

        # Kiểm tra PID của live_status còn đang chạy hay đã kết thúc
        live_pid = live_data.get("pid")
        pid_alive = False
        if isinstance(live_pid, int) and live_pid > 0:
            try:
                import psutil
                pid_alive = psutil.pid_exists(live_pid)
            except Exception:
                pid_alive = False

        job_state = self.job_state_provider() if self.job_state_provider else {}
        active_job = (
            job_state.get("active_job")
            if isinstance(job_state.get("active_job"), dict)
            else None
        )
        web_job_running = job_state.get("status") in {"running", "pausing"} and active_job is not None

        extra_cp_dirs: List[Path] = []
        if live_data.get("output_path"):
            d_live = _resolve_checkpoint_search_dir(self.workspace_root, str(live_data.get("output_path")))
            if d_live is not None:
                extra_cp_dirs.append(d_live)
        cand_jobs: List[Dict[str, Any]] = []
        if active_job is not None:
            cand_jobs.append(active_job)
        for hj in (job_state.get("history") or [])[:3]:
            if isinstance(hj, dict):
                cand_jobs.append(hj)
        for cj in cand_jobs:
            opts = cj.get("options") if isinstance(cj.get("options"), dict) else {}
            if opts.get("output_dir"):
                d_job = _resolve_checkpoint_search_dir(self.workspace_root, str(opts.get("output_dir")))
                if d_job is not None:
                    extra_cp_dirs.append(d_job)
        checkpoints = scan_temp_checkpoints(self.workspace_root, extra_dirs=extra_cp_dirs)

        # Xác định nguồn khởi chạy (Web GUI vs Terminal CLI ngoài)
        active_source = "idle"
        if web_job_running:
            active_source = "web"
        elif gpu_lock.get("locked") or (
            pid_alive and live_data.get("stage") not in {"completed", "paused", "error", None}
        ):
            active_source = "cli"

        effective_stage = str(live_data.get("stage") or "idle")

        if web_job_running and active_job is not None:
            active_type = str(active_job.get("job_type") or "upscale")
            active_pid = active_job.get("pid")
            q_items = active_job.get("queue_items") or []
            target_label = os.path.basename(str(q_items[0])) if q_items else "pending_subs.json"

            if active_type in {"subtrans", "subs_pending"}:
                # Tiến trình dịch phụ đề độc lập không ghi live_status.json: không để trạng thái tập cũ tràn sang
                effective_stage = "subtrans"
                sub_qual = str((active_job.get("options") or {}).get("subs_quality") or "normal")
                live_data = {
                    "video_name": target_label,
                    "pid": active_pid,
                    "started_at": active_job.get("started_at"),
                    "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "stage": "subtrans",
                    "stage_detail": (
                        f"Đang dịch hàng đợi phụ đề tồn đọng ({target_label})..."
                        if active_type == "subs_pending"
                        else f"Đang trích xuất & dịch phụ đề Anime ({target_label})..."
                    ),
                    "model_id": f"subtrans ({sub_qual})",
                    "zero_disk": False,
                    "total_frames": 0,
                    "unique_frames": 0,
                    "total_chunks": 0,
                    "completed_chunks": 0,
                    "chunks": [],
                    "resumed_chunks": [],
                    "active_chunk": None,
                    "steady_fps": 0.0,
                    "latest_fps": 0.0,
                    "effective_fps": 0.0,
                    "eta_seconds": None,
                }
            elif isinstance(active_pid, int) and live_pid != active_pid:
                # Tập mới vừa được Web GUI khởi chạy nhưng chưa kịp tạo RunLogger mới -> tránh báo nhầm 'completed' của tập cũ
                effective_stage = "initializing"
                live_data = {
                    "video_name": target_label,
                    "pid": active_pid,
                    "started_at": active_job.get("started_at"),
                    "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "stage": "initializing",
                    "stage_detail": "Đang khởi tạo tiến trình Upscale & kiểm tra môi trường...",
                    "model_id": str((active_job.get("options") or {}).get("model") or ""),
                    "zero_disk": bool((active_job.get("options") or {}).get("zero_disk", True)),
                    "total_frames": 0,
                    "unique_frames": 0,
                    "total_chunks": 0,
                    "completed_chunks": 0,
                    "chunks": [],
                    "resumed_chunks": [],
                    "active_chunk": None,
                    "steady_fps": 0.0,
                    "latest_fps": 0.0,
                    "effective_fps": 0.0,
                    "eta_seconds": None,
                }
        else:
            history_list = job_state.get("history") or []
            last_job = (
                history_list[0]
                if (history_list and isinstance(history_list[0], dict))
                else (
                    job_state.get("last_finished_job")
                    if isinstance(job_state.get("last_finished_job"), dict)
                    else None
                )
            )
            if (
                last_job is not None
                and not pid_alive
                and not gpu_lock.get("locked")
            ):
                last_type = str(last_job.get("job_type") or "")
                last_status = str(last_job.get("status") or "")
                last_finished_ts = 0.0
                fin_str = str(last_job.get("finished_at") or "")
                if fin_str:
                    try:
                        last_finished_ts = datetime.strptime(fin_str, "%Y-%m-%dT%H:%M:%S").timestamp()
                    except Exception:
                        last_finished_ts = 0.0
                live_from_newer_upscale = bool(
                    live_file.is_file()
                    and last_finished_ts > 0.0
                    and live_file_mtime > last_finished_ts + 0.5
                    and live_data.get("pid") != last_job.get("pid")
                    and str(live_data.get("stage") or "idle") != "idle"
                )
                q_items_last = last_job.get("queue_items") or []
                target_label_last = (
                    os.path.basename(str(q_items_last[0]))
                    if q_items_last
                    else ("pending_subs.json" if last_type == "subs_pending" else "")
                )

                if last_type in {"subtrans", "subs_pending"} and not live_from_newer_upscale:
                    if last_status == "completed":
                        effective_stage = "completed"
                        live_data = {
                            "video_name": target_label_last,
                            "pid": last_job.get("pid"),
                            "job_id": last_job.get("job_id"),
                            "job_type": last_type,
                            "started_at": last_job.get("started_at"),
                            "updated_at": fin_str or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            "stage": "completed",
                            "stage_detail": (
                                f"Hoàn tất dịch hàng đợi phụ đề ({target_label_last})."
                                if last_type == "subs_pending"
                                else f"Hoàn tất dịch & nhúng phụ đề ({target_label_last})."
                            ),
                            "model_id": f"subtrans ({(last_job.get('options') or {}).get('subs_quality', 'normal')})",
                            "zero_disk": False,
                            "total_frames": 0,
                            "unique_frames": 0,
                            "total_chunks": 0,
                            "completed_chunks": 0,
                            "chunks": [],
                            "resumed_chunks": [],
                            "active_chunk": None,
                            "steady_fps": 0.0,
                            "latest_fps": 0.0,
                            "effective_fps": 0.0,
                            "eta_seconds": None,
                        }
                    elif last_status in {"failed", "error"}:
                        effective_stage = "error"
                        live_data = {
                            "video_name": target_label_last,
                            "pid": last_job.get("pid"),
                            "job_id": last_job.get("job_id"),
                            "job_type": last_type,
                            "started_at": last_job.get("started_at"),
                            "updated_at": fin_str or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            "stage": "error",
                            "failed_stage": "subtrans",
                            "last_stage": "subtrans",
                            "stage_detail": (
                                f"Lỗi dịch phụ đề ({target_label_last}, mã thoát {last_job.get('return_code')})."
                            ),
                            "chunks": [],
                            "resumed_chunks": [],
                            "active_chunk": None,
                        }
                    elif last_status == "aborted":
                        effective_stage = "paused" if checkpoints else "idle"
                elif (
                    last_type in {"upscale", "preview"}
                    and last_status in {"failed", "error"}
                    and effective_stage != "error"
                    and (not live_file.is_file() or last_finished_ts >= live_file_mtime - 1.0)
                ):
                    failed_step = (
                        effective_stage
                        if effective_stage not in {"idle", "starting", "initializing", "completed"}
                        else "probing"
                    )
                    live_data = dict(live_data)
                    live_data["failed_stage"] = failed_step
                    live_data["last_stage"] = failed_step
                    if effective_stage in {"idle", "starting", "initializing", "completed"} or not live_data.get("stage_detail"):
                        live_data["stage_detail"] = (
                            f"Tác vụ {last_type} thất bại ({target_label_last}, mã thoát {last_job.get('return_code')})."
                        )
                    live_data["stage"] = "error"
                    effective_stage = "error"
                elif (
                    last_type in {"upscale", "preview"}
                    and last_status == "aborted"
                    and effective_stage not in {"completed", "paused", "error", "idle"}
                    and (not live_file.is_file() or last_finished_ts >= live_file_mtime - 1.0)
                ):
                    live_data = dict(live_data)
                    if effective_stage not in {"starting", "initializing"}:
                        live_data["last_stage"] = effective_stage
                    live_data["stage"] = "paused"
                    effective_stage = "paused"

            if (
                effective_stage in {"initializing", "probing", "preflight_dedup", "upscaling", "muxing", "auditing", "subtrans"}
                and not pid_alive
                and not gpu_lock.get("locked")
            ):
                effective_stage = "paused" if checkpoints else "idle"

        return {
            "active_source": active_source,
            "gpu_lock": gpu_lock,
            "graceful_pause_pending": pause_flag.is_file(),
            "effective_stage": effective_stage,
            "pid_alive": pid_alive,
            "live": live_data,
            "checkpoints": checkpoints,
            "job": job_state,
        }

    @staticmethod
    def _safe_resolve_str(path: Path) -> str:
        try:
            return str(path.resolve())
        except OSError:
            return str(path)

    @staticmethod
    def _file_prefix_sig(path: Path, max_bytes: int = 96) -> tuple[int, str]:
        try:
            n = max(0, min(96, int(max_bytes)))
            if n <= 0:
                return 0, ""
            with path.open("rb") as f:
                head = f.read(n)
            return len(head), (hashlib.blake2s(head, digest_size=4).hexdigest() if head else "")
        except OSError:
            return 0, ""

    @staticmethod
    def _file_header_sig(path: Path) -> str:
        _, sig = TelemetryHub._file_prefix_sig(path, 96)
        return sig

    @staticmethod
    def _aligned_eof_offset(path: Path) -> int:
        try:
            size = path.stat().st_size
            if size <= 0:
                return 0
            tail_len = min(size, 16384)
            with path.open("rb") as f:
                f.seek(size - tail_len)
                tail = f.read(tail_len)
            nl = tail.rfind(b"\n")
            if nl >= 0:
                return (size - tail_len) + nl + 1
            return size
        except OSError:
            return 0

    def _select_active_log_file(self) -> Optional[Path]:
        """Chọn file nhật ký đang hoạt động chính xác cho cả Web Job lẫn CLI ngoài trên Windows."""
        logs_dir = self.workspace_root / "logs"
        if not logs_dir.is_dir():
            return None

        web_job_log = logs_dir / "web_job.log"
        job_state = self.job_state_provider() if self.job_state_provider else {}
        active_job = (
            job_state.get("active_job")
            if isinstance(job_state.get("active_job"), dict)
            else None
        )
        if (
            job_state.get("status") in {"running", "pausing"}
            and active_job is not None
            and web_job_log.is_file()
        ):
            return web_job_log

        raw_candidates: List[Path] = []
        if web_job_log.is_file():
            raw_candidates.append(web_job_log)

        live_file = logs_dir / "live_status.json"
        if live_file.is_file():
            try:
                ldata = json.loads(live_file.read_text(encoding="utf-8"))
                log_name = str(ldata.get("log_file") or "").strip() if isinstance(ldata, dict) else ""
                if log_name and os.path.basename(log_name) == log_name:
                    cand_log = logs_dir / log_name
                    if cand_log.is_file():
                        raw_candidates.append(cand_log)
            except Exception:
                pass

        latest_link = logs_dir / "latest.log"
        if latest_link.is_file():
            raw_candidates.append(latest_link)

        try:
            recent_logs = sorted(
                (p for p in logs_dir.glob("*.log") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            raw_candidates.extend(recent_logs[:2])
        except OSError:
            pass

        seen: set[str] = set()
        unique: List[Path] = []
        for cand in raw_candidates:
            key = self._safe_resolve_str(cand)
            if key not in seen:
                seen.add(key)
                unique.append(cand)

        if not unique:
            return None
        return max(unique, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)

    def _prime_log_cursor(self) -> None:
        best = self._select_active_log_file()
        if best is None:
            return
        resolved_str = self._safe_resolve_str(best)
        offset = self._aligned_eof_offset(best)
        prefix_len, sig = self._file_prefix_sig(best, offset)
        self._last_log_cursor = {
            "path": resolved_str,
            "prefix_len": prefix_len,
            "sig": sig,
            "offset": offset,
        }

    def read_recent_logs_with_cursor(self, max_lines: int = 120) -> tuple[List[str], Dict[str, Any]]:
        """Đọc các dòng hoàn chỉnh mới nhất cùng con trỏ byte để đồng bộ không trùng lặp với SSE tick."""
        best = self._select_active_log_file()
        if best is None:
            return [], {"path": "", "prefix_len": 0, "sig": "", "offset": 0}

        resolved_str = self._safe_resolve_str(best)
        try:
            aligned_end = self._aligned_eof_offset(best)
            prefix_len, sig = self._file_prefix_sig(best, aligned_end)
            window = min(aligned_end, 131072)
            start_pos = aligned_end - window
            with best.open("rb") as f:
                f.seek(start_pos)
                raw = f.read(window)
            if start_pos > 0:
                first_nl = raw.find(b"\n")
                if first_nl >= 0:
                    raw = raw[first_nl + 1 :]
            text = raw.decode("utf-8", errors="replace")
            lines = [ln.rstrip("\r") for ln in text.splitlines() if ln.strip()]
            cursor = {"path": resolved_str, "prefix_len": prefix_len, "sig": sig, "offset": aligned_end}
            if not self._last_log_cursor.get("path"):
                self._last_log_cursor = dict(cursor)
            return lines[-max_lines:], cursor
        except Exception:
            return [], {"path": resolved_str, "prefix_len": 0, "sig": "", "offset": 0}

    def read_recent_logs(self, max_lines: int = 120) -> List[str]:
        """Đọc các dòng mới nhất từ logs/latest.log, live log_file hoặc log Web Job."""
        lines, _ = self.read_recent_logs_with_cursor(max_lines=max_lines)
        return lines

    def _read_incremental_logs_from_cursor(self, cursor: Optional[Dict[str, Any]]) -> tuple[List[str], Dict[str, Any]]:
        """
        Đọc phần dòng hoàn chỉnh mới phát sinh cho một con trỏ độc lập (per-subscriber).
        So sánh đúng độ dài tiền tố cũ (prefix_len) để tránh phát lại dòng khi log nhỏ (<96B) tăng kích thước.
        """
        cur = dict(cursor or {"path": "", "prefix_len": 0, "sig": "", "offset": 0})
        prev_path = str(cur.get("path") or "")
        prev_prefix_len = int(cur.get("prefix_len") or 0)
        prev_sig = str(cur.get("sig") or "")
        prev_offset = int(cur.get("offset") or 0)

        best = self._select_active_log_file()
        if best is None:
            return [], cur

        resolved_str = self._safe_resolve_str(best)
        try:
            size = best.stat().st_size
            file_changed = resolved_str != prev_path or size < prev_offset
            if not file_changed and prev_prefix_len > 0:
                cur_prefix_len, cur_prefix_sig = self._file_prefix_sig(best, prev_prefix_len)
                if cur_prefix_len != prev_prefix_len or cur_prefix_sig != prev_sig:
                    file_changed = True

            if file_changed:
                if size <= 32768:
                    prev_offset = 0
                else:
                    seek_base = size - 16384
                    with best.open("rb") as f:
                        f.seek(seek_base)
                        probe = f.read(min(4096, size - seek_base))
                    first_nl = probe.find(b"\n")
                    prev_offset = (seek_base + first_nl + 1) if first_nl >= 0 else seek_base
                prev_prefix_len = 0
                prev_sig = ""

            if size <= prev_offset:
                new_len, new_sig = self._file_prefix_sig(best, min(96, size))
                next_cur = {"path": resolved_str, "prefix_len": new_len, "sig": new_sig, "offset": size, "reset": bool(file_changed)}
                return [], next_cur

            with best.open("rb") as f:
                f.seek(prev_offset)
                chunk = f.read(min(65536, size - prev_offset))

            last_nl = chunk.rfind(b"\n")
            if last_nl >= 0:
                complete = chunk[: last_nl + 1]
                new_offset = prev_offset + last_nl + 1
            elif len(chunk) >= 16384:
                complete = chunk
                new_offset = prev_offset + len(chunk)
            else:
                # Dòng cuối chưa kết thúc bằng '\n': giữ nguyên offset để không xé đôi dòng
                return [], {
                    "path": resolved_str,
                    "prefix_len": prev_prefix_len,
                    "sig": prev_sig,
                    "offset": prev_offset,
                    "reset": bool(file_changed),
                }

            new_len, new_sig = self._file_prefix_sig(best, min(96, new_offset))
            next_cur = {"path": resolved_str, "prefix_len": new_len, "sig": new_sig, "offset": new_offset, "reset": bool(file_changed)}
            decoded = complete.decode("utf-8", errors="replace")
            out_lines = [ln.rstrip("\r") for ln in decoded.splitlines() if ln.strip()][-40:]
            return out_lines, next_cur
        except Exception:
            return [], cur

    def _read_incremental_logs_with_cursor(self) -> tuple[List[str], Dict[str, Any]]:
        """Đọc phần dòng hoàn chỉnh mới phát sinh trên con trỏ dùng chung nội bộ."""
        lines, next_cur = self._read_incremental_logs_from_cursor(self._last_log_cursor)
        self._last_log_cursor = next_cur
        return lines, next_cur

    def _read_incremental_logs(self) -> List[str]:
        lines, _ = self._read_incremental_logs_with_cursor()
        return lines

    def build_snapshot(self) -> Dict[str, Any]:
        """Tạo gói dữ liệu tổng hợp cho REST /api/status và sự kiện đầu tiên của SSE."""
        loop_running = self._poll_task is not None and not self._poll_task.done()
        if loop_running:
            with self._history_lock:
                latest = dict(self.history[-1]) if self.history else None
                hist_copy = list(self.history)
            if latest is None:
                latest = self.sample_telemetry(record_history=False)
        else:
            latest = self.sample_telemetry(record_history=True)
            with self._history_lock:
                hist_copy = list(self.history)

        pipeline = self.read_pipeline_state()
        logs, log_cursor = self.read_recent_logs_with_cursor(100)
        return {
            "gpu_info": self.gpu_static,
            "telemetry": latest,
            "history": hist_copy,
            "pipeline": pipeline,
            "logs": logs,
            "log_cursor": log_cursor,
        }

    def _poll_tick_sync(self) -> Dict[str, Any]:
        """Thu thập 1 nhịp telemetry + trạng thái pipeline (chạy trong threadpool để không block asyncio loop)."""
        telem = self.sample_telemetry(record_history=True)
        pipe = self.read_pipeline_state()
        return {
            "telemetry": telem,
            "pipeline": pipe,
        }

    async def shutdown(self) -> None:
        """Ngắt sạch các luồng SSE đang mở và hủy vòng lặp thu thập telemetry khi đóng máy chủ."""
        self._shutting_down = True
        for q in list(self._subscribers):
            try:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                q.put_nowait(None)
            except Exception:
                pass
        task = self._poll_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._poll_task = None

    async def _broadcast_loop(self) -> None:
        try:
            while self._subscribers and not self._shutting_down:
                await asyncio.sleep(1.0)
                if not self._subscribers or self._shutting_down:
                    break
                packet = await asyncio.to_thread(self._poll_tick_sync)
                if self._shutting_down:
                    break
                dead: List[asyncio.Queue[Optional[Dict[str, Any]]]] = []
                for q in list(self._subscribers):
                    try:
                        if q.full():
                            _ = q.get_nowait()
                        q.put_nowait(packet)
                    except Exception:
                        dead.append(q)
                for q in dead:
                    self._subscribers.discard(q)
        except asyncio.CancelledError:
            pass
        finally:
            self._poll_task = None

    async def stream_events(self, once: bool = False) -> AsyncIterator[str]:
        """Trình sinh Server-Sent Events chuẩn text/event-stream với con trỏ log riêng cho từng tab."""
        if self._shutting_down:
            return
        snap = await asyncio.to_thread(self.build_snapshot)
        sub_cursor: Dict[str, Any] = dict(
            snap.get("log_cursor") or {"path": "", "prefix_len": 0, "sig": "", "offset": 0}
        )
        yield f"event: snapshot\ndata: {json.dumps(snap, ensure_ascii=False)}\n\n"
        if once or self._shutting_down:
            return

        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(maxsize=8)
        self._subscribers.add(queue)
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._broadcast_loop())

        try:
            while not self._shutting_down:
                base_packet = await queue.get()
                if base_packet is None or self._shutting_down:
                    break
                new_logs, sub_cursor = await asyncio.to_thread(
                    self._read_incremental_logs_from_cursor, sub_cursor
                )
                packet = {
                    "telemetry": base_packet["telemetry"],
                    "pipeline": base_packet["pipeline"],
                    "logs": new_logs,
                    "log_cursor": sub_cursor,
                }
                yield f"event: tick\ndata: {json.dumps(packet, ensure_ascii=False)}\n\n"
        finally:
            self._subscribers.discard(queue)
