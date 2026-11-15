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
import time
from typing import Any, AsyncIterator, Callable, Deque, Dict, List, Optional

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


def scan_temp_checkpoints(workspace_root: Path) -> List[Dict[str, Any]]:
    """Quét danh sách các thư mục checkpoint .temp_* hiện có (không xóa hay chỉnh sửa)."""
    results: List[Dict[str, Any]] = []
    search_dirs = [workspace_root, workspace_root / "out"]
    seen: set[str] = set()

    for base in search_dirs:
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
        self._prime_counters()

        self._subscribers: set[asyncio.Queue[Dict[str, Any]]] = set()
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._last_log_key: tuple[str, str, int] = ("", "", 0)
        self._prime_log_cursor()

    def _prime_counters(self) -> None:
        try:
            import psutil
            psutil.cpu_percent(interval=None)
            io = psutil.disk_io_counters()
            if io is not None:
                self._last_read_bytes = io.read_bytes
                self._last_write_bytes = io.write_bytes
                self._last_disk_time = time.perf_counter()
        except Exception:
            pass

    def sample_telemetry(self) -> Dict[str, Any]:
        """Thu thập 1 mẫu 17+ chỉ số phần cứng và lưu vào vòng đệm 120 giây."""
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
            if io is not None and self._last_disk_time > 0:
                dt = now_perf - self._last_disk_time
                if dt > 0.05:
                    d_r = max(0, io.read_bytes - self._last_read_bytes)
                    d_w = max(0, io.write_bytes - self._last_write_bytes)
                    ssd_read_mb_s = round(d_r / (dt * 1024 * 1024), 2)
                    ssd_write_mb_s = round(d_w / (dt * 1024 * 1024), 2)
                    self._last_read_bytes = io.read_bytes
                    self._last_write_bytes = io.write_bytes
                    self._last_disk_time = now_perf
            elif io is not None:
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
        self.history.append(sample)
        return sample

    def read_pipeline_state(self) -> Dict[str, Any]:
        """Đọc trạng thái pipeline thời gian thực (tương thích cả CLI ngoài lẫn Web Job)."""
        logs_dir = self.workspace_root / "logs"
        live_file = logs_dir / "live_status.json"
        pause_flag = logs_dir / ".graceful_pause.flag"
        gpu_lock = check_active_gpu_lock(self.workspace_root)
        checkpoints = scan_temp_checkpoints(self.workspace_root)

        live_data: Dict[str, Any] = {}
        if live_file.is_file():
            try:
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
            elif effective_stage == "completed":
                # Stage A video đã xong nhưng tiến trình upscale.py vẫn đang chạy Stage B dịch phụ đề cuối hàng đợi
                effective_stage = "subtrans"
                live_data = dict(live_data)
                live_data["stage_detail"] = "Đang hoàn tất dịch & nhúng phụ đề (Stage B)..."
        elif (
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
    def _file_header_sig(path: Path) -> str:
        try:
            with path.open("rb") as f:
                head = f.read(96)
            return hashlib.blake2s(head, digest_size=4).hexdigest() if head else ""
        except OSError:
            return ""

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
            and active_job.get("job_type") in {"subtrans", "subs_pending"}
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
        sig = self._file_header_sig(best)
        offset = self._aligned_eof_offset(best)
        self._last_log_key = (resolved_str, sig, offset)

    def read_recent_logs_with_cursor(self, max_lines: int = 120) -> tuple[List[str], Dict[str, Any]]:
        """Đọc các dòng hoàn chỉnh mới nhất cùng con trỏ byte để đồng bộ không trùng lặp với SSE tick."""
        best = self._select_active_log_file()
        if best is None:
            return [], {"path": "", "sig": "", "offset": 0}

        resolved_str = self._safe_resolve_str(best)
        sig = self._file_header_sig(best)
        try:
            aligned_end = self._aligned_eof_offset(best)
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
            if self._last_log_key == ("", "", 0):
                self._last_log_key = (resolved_str, sig, aligned_end)
            return lines[-max_lines:], {"path": resolved_str, "sig": sig, "offset": aligned_end}
        except Exception:
            return [], {"path": resolved_str, "sig": sig, "offset": 0}

    def read_recent_logs(self, max_lines: int = 120) -> List[str]:
        """Đọc các dòng mới nhất từ logs/latest.log, live log_file hoặc log Web Job."""
        lines, _ = self.read_recent_logs_with_cursor(max_lines=max_lines)
        return lines

    def _read_incremental_logs_with_cursor(self) -> tuple[List[str], Dict[str, Any]]:
        """Đọc phần dòng hoàn chỉnh mới phát sinh kể từ chu kỳ 1Hz trước (căn biên newline, chống xé dòng)."""
        best = self._select_active_log_file()
        if best is None:
            prev_p, prev_s, prev_o = self._last_log_key
            return [], {"path": prev_p, "sig": prev_s, "offset": prev_o}

        resolved_str = self._safe_resolve_str(best)
        sig = self._file_header_sig(best)
        try:
            size = best.stat().st_size
            prev_path, prev_sig, prev_offset = self._last_log_key

            if resolved_str != prev_path or sig != prev_sig or size < prev_offset:
                # File log mới xuất hiện hoặc web_job.log vừa mở ghi mới
                if size <= 32768:
                    prev_offset = 0
                else:
                    seek_base = size - 16384
                    with best.open("rb") as f:
                        f.seek(seek_base)
                        probe = f.read(min(4096, size - seek_base))
                    first_nl = probe.find(b"\n")
                    prev_offset = (seek_base + first_nl + 1) if first_nl >= 0 else seek_base

            if size <= prev_offset:
                self._last_log_key = (resolved_str, sig, size)
                return [], {"path": resolved_str, "sig": sig, "offset": size}

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
                # Dòng cuối chưa kết thúc bằng '\n': giữ lại con trỏ chờ chu kỳ sau để không xé đôi dòng
                return [], {"path": resolved_str, "sig": sig, "offset": prev_offset}

            self._last_log_key = (resolved_str, sig, new_offset)
            decoded = complete.decode("utf-8", errors="replace")
            out_lines = [ln.rstrip("\r") for ln in decoded.splitlines() if ln.strip()][-40:]
            return out_lines, {"path": resolved_str, "sig": sig, "offset": new_offset}
        except Exception:
            return [], {"path": resolved_str, "sig": sig, "offset": self._last_log_key[2]}

    def _read_incremental_logs(self) -> List[str]:
        lines, _ = self._read_incremental_logs_with_cursor()
        return lines

    def build_snapshot(self) -> Dict[str, Any]:
        """Tạo gói dữ liệu tổng hợp cho REST /api/status và sự kiện đầu tiên của SSE."""
        latest = self.sample_telemetry()
        pipeline = self.read_pipeline_state()
        logs, log_cursor = self.read_recent_logs_with_cursor(100)
        return {
            "gpu_info": self.gpu_static,
            "telemetry": latest,
            "history": list(self.history),
            "pipeline": pipeline,
            "logs": logs,
            "log_cursor": log_cursor,
        }

    async def _broadcast_loop(self) -> None:
        try:
            while self._subscribers:
                await asyncio.sleep(1.0)
                if not self._subscribers:
                    break
                telem = self.sample_telemetry()
                pipe = self.read_pipeline_state()
                new_logs, log_cursor = self._read_incremental_logs_with_cursor()
                packet = {
                    "telemetry": telem,
                    "pipeline": pipe,
                    "logs": new_logs,
                    "log_cursor": log_cursor,
                }
                dead: List[asyncio.Queue[Dict[str, Any]]] = []
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
        """Trình sinh Server-Sent Events chuẩn text/event-stream."""
        snap = self.build_snapshot()
        yield f"event: snapshot\ndata: {json.dumps(snap, ensure_ascii=False)}\n\n"
        if once:
            return

        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=8)
        self._subscribers.add(queue)
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._broadcast_loop())

        try:
            while True:
                packet = await queue.get()
                yield f"event: tick\ndata: {json.dumps(packet, ensure_ascii=False)}\n\n"
        finally:
            self._subscribers.discard(queue)
