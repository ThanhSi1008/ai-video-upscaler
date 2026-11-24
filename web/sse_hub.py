"""1 Hz telemetry collector and Server-Sent Events hub synchronizing CLI and Web UI."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, AsyncIterator, Callable, Deque, Dict, List, Optional, Sequence

from scripts.telemetry_collector import default_gpu_query
from web.fs_browser import is_unc_path, strip_wrapping_quotes


def _sanitize_json_payload(val: Any) -> Any:
    """Recursively convert non-finite floats (NaN/Inf/-Inf) to None for strict RFC 8259 JSON."""
    if isinstance(val, float):
        return val if math.isfinite(val) else None
    if isinstance(val, dict):
        return {k: _sanitize_json_payload(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_json_payload(item) for item in val]
    return val


def _coerce_pid(raw_pid: Any) -> Optional[int]:
    """Return positive integer PID, rejecting booleans and non-positive values."""
    if isinstance(raw_pid, bool) or not isinstance(raw_pid, int):
        return None
    return raw_pid if raw_pid > 0 else None


def _parse_iso_timestamp(raw: Any) -> float:
    """Parse ISO-8601 timestamp string (with or without microseconds) into epoch seconds."""
    text = str(raw or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        try:
            return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").timestamp()
        except Exception:
            return 0.0


def _resolve_last_job(
    job_state: Dict[str, Any], active_job: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Return latest finished job, falling back to an unfinalized aborted active_job."""
    history_list = job_state.get("history") or []
    last_hist = (
        history_list[0]
        if (history_list and isinstance(history_list[0], dict))
        else (
            job_state.get("last_finished_job")
            if isinstance(job_state.get("last_finished_job"), dict)
            else None
        )
    )
    if active_job is not None and str(active_job.get("status") or "") == "aborted":
        active_jid = str(active_job.get("job_id") or "").strip()
        in_history = bool(
            active_jid
            and any(
                isinstance(h, dict) and str(h.get("job_id") or "").strip() == active_jid
                for h in history_list
            )
        )
        # Check job_id / finished_at so an older history[0] job never masks a newly aborted job.
        if not in_history and (
            last_hist is None
            or active_jid
            or active_job.get("finished_at") is None
        ):
            return active_job
    return last_hist


def live_matches_web_job(
    live_data: Dict[str, Any],
    job_dict: Dict[str, Any],
    live_file_mtime: float = 0.0,
) -> bool:
    """Return True when live_status.json belongs to job_dict via job_id, PID, or parent-child lineage."""
    if not isinstance(live_data, dict) or not isinstance(job_dict, dict):
        return False
    live_jid = str(live_data.get("job_id") or "").strip()
    job_jid = str(job_dict.get("job_id") or "").strip()
    if live_jid and job_jid:
        return live_jid == job_jid
    started_ts = _parse_iso_timestamp(job_dict.get("started_at"))
    if started_ts > 0.0 and live_file_mtime > 0.0 and live_file_mtime < started_ts - 1.5:
        return False
    job_pid = _coerce_pid(job_dict.get("pid"))
    live_pid = _coerce_pid(live_data.get("pid"))
    if job_pid is None or live_pid is None:
        return False
    if live_pid == job_pid:
        return True
    if _coerce_pid(live_data.get("ppid")) == job_pid:
        return True
    try:
        import psutil
        proc_ppid = psutil.Process(live_pid).ppid()
        return _coerce_pid(proc_ppid) == job_pid
    except Exception:
        return False


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
        if pid > 0 and psutil.pid_exists(pid):
            return {"locked": True, "pid": pid}
    except Exception:
        pass
    return {"locked": False, "pid": None, "stale_lock": True}


_CHECKPOINT_VIDEO_SUFFIXES = {".mkv", ".mp4", ".webm", ".mov", ".avi", ".ts", ".m2ts"}


def _resolve_checkpoint_search_dir(workspace_root: Path, raw_dir_or_file: Optional[str]) -> Optional[Path]:
    if not raw_dir_or_file or not str(raw_dir_or_file).strip():
        return None
    raw_str = str(raw_dir_or_file)
    if "\x00" in raw_str or is_unc_path(raw_str):
        return None
    raw = strip_wrapping_quotes(raw_str)
    if not raw or "\x00" in raw or is_unc_path(raw):
        return None
    p = Path(os.path.expanduser(raw))
    if p.drive and not p.is_absolute():
        return None
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
    live_data_disk: Dict[str, Any] = {}
    live_file = workspace_root / "logs" / "live_status.json"
    if live_file.is_file():
        try:
            ldata = json.loads(live_file.read_text(encoding="utf-8"))
            if isinstance(ldata, dict):
                live_data_disk = ldata
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
            resolved_d = cand_dir.resolve()
            norm_d = os.path.normcase(str(resolved_d))
        except OSError:
            try:
                resolved_d = cand_dir
                norm_d = os.path.normcase(os.path.abspath(str(cand_dir)))
            except OSError:
                continue
        if norm_d not in seen_dirs:
            seen_dirs.add(norm_d)
            unique_dirs.append(resolved_d)

    seen: set[str] = set()

    for base in unique_dirs:
        if not base.is_dir():
            continue
        try:
            entries = list(os.scandir(base))
        except OSError:
            continue
        for entry in entries:
            if not entry.name.startswith(".temp_") or len(entry.name) <= len(".temp_"):
                continue
            try:
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    continue
                st_no_follow = entry.stat(follow_symlinks=False)
                if getattr(st_no_follow, "st_file_attributes", 0) & 0x0400:
                    continue
            except OSError:
                continue
            try:
                t_path = Path(entry.path).resolve()
                norm = os.path.normcase(str(t_path))
            except OSError:
                t_path = Path(entry.path)
                norm = os.path.normcase(os.path.abspath(entry.path))
            if norm in seen:
                continue
            seen.add(norm)
            raw_segs_dir = t_path / "raw_segs"
            up_segs_dir = t_path / "upscaled_segs"
            manifest_path = t_path / "manifest.json"
            plan_path = t_path / "plan.json"

            raw_count = 0
            up_count = 0
            size_bytes = 0
            if raw_segs_dir.is_dir():
                for f in os.scandir(raw_segs_dir):
                    if f.name.startswith("seg_") and f.name.endswith(".mp4") and not f.name.endswith(".tmp.mp4"):
                        raw_count += 1
                        try:
                            size_bytes += f.stat().st_size
                        except OSError:
                            pass
            if up_segs_dir.is_dir():
                for f in os.scandir(up_segs_dir):
                    if f.name.startswith("up_") and f.name.endswith(".mp4") and not f.name.endswith(".tmp.mp4"):
                        up_count += 1
                        try:
                            size_bytes += f.stat().st_size
                        except OSError:
                            pass

            episode_name = entry.name[len(".temp_"):]
            model_id = ""
            model_requested = ""
            dedup_policy = ""
            dedup_policy_requested = ""
            grain_restore = "auto"
            nvenc_multipass_policy = "auto"
            zero_disk = True
            use_tensorrt = True
            batch_size_val: Optional[int] = None
            subs_after_upscale: Optional[bool] = None
            subs_quality = ""
            prefer_whisper: Optional[bool] = None
            async_subs: Optional[bool] = None
            glossary_str = ""
            subs_upgrade_val: Optional[bool] = None
            no_embed_val: Optional[bool] = None
            sound_cues_str = ""
            source_file = ""
            output_dir_str = ""
            if manifest_path.is_file():
                try:
                    mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(mdata, dict):
                        model_id = str(mdata.get("model_id") or "")
                        if "model_requested" in mdata:
                            model_requested = str(mdata.get("model_requested") or "").strip()
                        else:
                            res_src = str(mdata.get("resolution_source") or "").strip().lower()
                            if res_src.startswith(("shows.yaml", "models.yaml", "default")) and model_id != "auto-pareto":
                                model_requested = ""
                            else:
                                model_requested = model_id
                        dedup_policy = str(mdata.get("dedup_policy") or "")
                        dedup_policy_requested = str(mdata.get("dedup_policy_requested") or "")
                        if mdata.get("grain_restore"):
                            grain_restore = str(mdata.get("grain_restore") or "auto")
                        if mdata.get("nvenc_multipass_policy"):
                            nvenc_multipass_policy = str(mdata.get("nvenc_multipass_policy") or "auto")
                        if "zero_disk" in mdata:
                            zero_disk = bool(mdata.get("zero_disk"))
                        if "use_tensorrt" in mdata:
                            use_tensorrt = bool(mdata.get("use_tensorrt"))
                        if "subs_after_upscale" in mdata and isinstance(mdata.get("subs_after_upscale"), bool):
                            subs_after_upscale = bool(mdata.get("subs_after_upscale"))
                        if mdata.get("subs_quality"):
                            subs_quality = str(mdata.get("subs_quality") or "")
                        if "prefer_whisper" in mdata and isinstance(mdata.get("prefer_whisper"), bool):
                            prefer_whisper = bool(mdata.get("prefer_whisper"))
                        if "async_subs" in mdata and isinstance(mdata.get("async_subs"), bool):
                            async_subs = bool(mdata.get("async_subs"))
                        if mdata.get("glossary"):
                            glossary_str = str(mdata.get("glossary") or "").strip()
                        if "subs_upgrade" in mdata and isinstance(mdata.get("subs_upgrade"), bool):
                            subs_upgrade_val = bool(mdata.get("subs_upgrade"))
                        if "no_embed" in mdata and isinstance(mdata.get("no_embed"), bool):
                            no_embed_val = bool(mdata.get("no_embed"))
                        if mdata.get("sound_cues"):
                            sound_cues_str = str(mdata.get("sound_cues") or "").strip()
                        raw_bs = mdata.get("batch_size")
                        if isinstance(raw_bs, int) and not isinstance(raw_bs, bool) and 1 <= raw_bs <= 32:
                            batch_size_val = raw_bs
                        output_dir_str = str(mdata.get("output_dir") or "").strip()
                        for key in ("original_source_file", "source_file", "input_path"):
                            cand = str(mdata.get(key) or "").strip()
                            if cand and not os.path.basename(cand).startswith(".preview_clip_"):
                                source_file = cand
                                break
                except Exception:
                    pass

            total_chunks = raw_count
            manifest_total_chunks = 0
            if manifest_path.is_file():
                try:
                    mdata_tc = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(mdata_tc, dict):
                        manifest_total_chunks = int(mdata_tc.get("total_chunks") or 0)
                except Exception:
                    manifest_total_chunks = 0
            if plan_path.is_file():
                try:
                    pdata = json.loads(plan_path.read_text(encoding="utf-8"))
                    if isinstance(pdata, dict):
                        if total_chunks == 0:
                            total_chunks = len(pdata.get("chunks") or []) or int(pdata.get("n_segments") or 0)
                        if not dedup_policy and pdata.get("dedup_policy"):
                            dedup_policy = str(pdata.get("dedup_policy") or "")
                        p_params = pdata.get("dedup_params") or {}
                        if grain_restore == "auto" and isinstance(p_params, dict) and p_params.get("grain_restore"):
                            grain_restore = str(p_params.get("grain_restore") or "auto")
                        if not source_file:
                            cand_plan_path = str(pdata.get("source_path") or "").strip()
                            if cand_plan_path and not os.path.basename(cand_plan_path).startswith(".preview_clip_"):
                                source_file = cand_plan_path
                            else:
                                cand_plan_file = str(pdata.get("source_file") or "").strip()
                                if (
                                    cand_plan_file
                                    and os.path.isabs(cand_plan_file)
                                    and not os.path.basename(cand_plan_file).startswith(".preview_clip_")
                                ):
                                    source_file = cand_plan_file
                except Exception:
                    pass
            if total_chunks == 0 and manifest_total_chunks > 0:
                total_chunks = manifest_total_chunks

            if not source_file and live_data_disk:
                cand_live_in = str(live_data_disk.get("input_path") or "").strip()
                live_vid = str(live_data_disk.get("video_name") or "").strip()
                if (
                    cand_live_in
                    and not os.path.basename(cand_live_in).startswith(".preview_clip_")
                    and (
                        Path(cand_live_in).stem == episode_name
                        or live_vid == episode_name
                        or (live_vid and Path(live_vid).stem == episode_name)
                    )
                ):
                    source_file = cand_live_in

            if not output_dir_str:
                try:
                    output_dir_str = str(t_path.parent.resolve())
                except OSError:
                    output_dir_str = str(t_path.parent)
            default_out_norm = os.path.normcase(os.path.abspath(str(workspace_root / "out")))
            try:
                out_dir_norm = os.path.normcase(os.path.abspath(output_dir_str))
                is_custom_output_dir = out_dir_norm != default_out_norm
            except OSError:
                is_custom_output_dir = False

            results.append({
                "name": entry.name,
                "episode_name": episode_name,
                "source_file": source_file,
                "output_dir": output_dir_str,
                "is_custom_output_dir": is_custom_output_dir,
                "path": str(t_path),
                "completed_chunks": up_count,
                "total_chunks": total_chunks,
                "model_id": model_id,
                "model_requested": model_requested,
                "dedup_policy": dedup_policy,
                "dedup_policy_requested": dedup_policy_requested,
                "grain_restore": grain_restore,
                "nvenc_multipass_policy": nvenc_multipass_policy,
                "zero_disk": zero_disk,
                "use_tensorrt": use_tensorrt,
                "batch_size": batch_size_val,
                "subs_after_upscale": subs_after_upscale,
                "subs_quality": subs_quality,
                "prefer_whisper": prefer_whisper,
                "async_subs": async_subs,
                "glossary": glossary_str,
                "subs_upgrade": subs_upgrade_val,
                "no_embed": no_embed_val,
                "sound_cues": sound_cues_str,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
            })
    return results


class TelemetryHub:
    """Bộ thu thập viễn trắc 1.0 Hz & phân phối SSE tới mọi trình duyệt đang mở."""

    @staticmethod
    def _live_matches_web_job(
        live_data: Dict[str, Any],
        job_dict: Dict[str, Any],
        live_file_mtime: float = 0.0,
    ) -> bool:
        return live_matches_web_job(live_data, job_dict, live_file_mtime)

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
        self._log_cursor_lock = threading.Lock()
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
        static_vram_total = float(self.gpu_static.get("vram_total_mib", 12288.0) or 12288.0)
        vram_total = (
            max(round(vram_used + vram_free, 0), static_vram_total)
            if (vram_used + vram_free) > 0
            else static_vram_total
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
        live_pid = _coerce_pid(live_data.get("pid"))
        pid_alive = False
        if live_pid is not None:
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
                sub_opts = active_job.get("options") or {}
                sub_qual = str(sub_opts.get("subs_quality") or "normal")
                is_sub_dry = bool(sub_opts.get("dry_run"))
                dry_prefix = "[Dry-Run] " if is_sub_dry else ""
                live_data = {
                    "video_name": target_label,
                    "pid": active_pid,
                    "job_id": active_job.get("job_id"),
                    "job_type": active_type,
                    "dry_run": is_sub_dry,
                    "started_at": active_job.get("started_at"),
                    "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "stage": "subtrans",
                    "stage_detail": (
                        f"{dry_prefix}Đang dịch hàng đợi phụ đề tồn đọng ({target_label})..."
                        if active_type == "subs_pending"
                        else f"{dry_prefix}Đang trích xuất & dịch phụ đề Anime ({target_label})..."
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
            elif (
                ((live_data.get("job_id") and active_job.get("job_id")) or _coerce_pid(active_pid) is not None)
                and not self._live_matches_web_job(live_data, active_job, live_file_mtime)
            ):
                # Tập mới vừa được Web GUI khởi chạy nhưng chưa kịp tạo RunLogger mới -> tránh báo nhầm 'completed' của tập cũ
                effective_stage = "initializing"
                up_opts = active_job.get("options") or {}
                live_data = {
                    "video_name": target_label,
                    "pid": active_pid,
                    "job_id": active_job.get("job_id"),
                    "job_type": active_type,
                    "dry_run": bool(up_opts.get("dry_run")),
                    "started_at": active_job.get("started_at"),
                    "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "stage": "initializing",
                    "stage_detail": "Đang khởi tạo tiến trình Upscale & kiểm tra môi trường...",
                    "model_id": str(up_opts.get("model") or ""),
                    "zero_disk": bool(up_opts.get("zero_disk", True)),
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
            last_job = _resolve_last_job(job_state, active_job)
            if (
                last_job is not None
                and not pid_alive
                and not gpu_lock.get("locked")
            ):
                last_type = str(last_job.get("job_type") or "")
                last_status = str(last_job.get("status") or "")
                fin_str = str(last_job.get("finished_at") or "")
                raw_fin_epoch = last_job.get("finished_epoch")
                has_exact_fin_epoch = (
                    isinstance(raw_fin_epoch, (int, float))
                    and not isinstance(raw_fin_epoch, bool)
                    and float(raw_fin_epoch) > 0.0
                )
                last_finished_ts = float(raw_fin_epoch) if has_exact_fin_epoch else _parse_iso_timestamp(fin_str)
                newer_slack = 0.002 if has_exact_fin_epoch else 1.05
                live_from_newer_upscale = bool(
                    live_file.is_file()
                    and last_finished_ts > 0.0
                    and live_file_mtime > last_finished_ts + newer_slack
                    and not self._live_matches_web_job(live_data, last_job, live_file_mtime)
                    and str(live_data.get("stage") or "idle") != "idle"
                )
                live_matches_last_job = bool(
                    live_file.is_file()
                    and self._live_matches_web_job(live_data, last_job, live_file_mtime)
                )
                q_items_last = last_job.get("queue_items") or []
                target_label_last = (
                    os.path.basename(str(q_items_last[0]))
                    if q_items_last
                    else ("pending_subs.json" if last_type == "subs_pending" else "")
                )

                if last_type in {"subtrans", "subs_pending"} and not live_from_newer_upscale:
                    if last_status == "completed":
                        is_sub_dry = bool((last_job.get("options") or {}).get("dry_run"))
                        effective_stage = "completed"
                        live_data = {
                            "video_name": target_label_last,
                            "pid": last_job.get("pid"),
                            "job_id": last_job.get("job_id"),
                            "job_type": last_type,
                            "dry_run": is_sub_dry,
                            "started_at": last_job.get("started_at"),
                            "updated_at": fin_str or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            "stage": "completed",
                            "stage_detail": (
                                f"Hoàn tất mô phỏng dịch phụ đề (--dry-run) ({target_label_last})."
                                if is_sub_dry
                                else (
                                    f"Hoàn tất dịch hàng đợi phụ đề ({target_label_last})."
                                    if last_type == "subs_pending"
                                    else f"Hoàn tất dịch & nhúng phụ đề ({target_label_last})."
                                )
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
                    elif last_status == "aborted":
                        aborted_stage = "paused" if checkpoints else "idle"
                        effective_stage = aborted_stage
                        live_data = {
                            "video_name": target_label_last,
                            "pid": last_job.get("pid"),
                            "job_id": last_job.get("job_id"),
                            "job_type": last_type,
                            "started_at": last_job.get("started_at"),
                            "updated_at": fin_str or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            "stage": aborted_stage,
                            "last_stage": "subtrans",
                            "stage_detail": (
                                f"Đã dừng hàng đợi phụ đề ({target_label_last})."
                                if last_type == "subs_pending"
                                else f"Đã dừng dịch phụ đề ({target_label_last})."
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
                elif (
                    last_type == "upscale"
                    and bool((last_job.get("options") or {}).get("dry_run"))
                    and last_status == "completed"
                    and not live_from_newer_upscale
                ):
                    effective_stage = "completed"
                    live_data = {
                        "video_name": target_label_last,
                        "pid": last_job.get("pid"),
                        "job_id": last_job.get("job_id"),
                        "job_type": last_type,
                        "dry_run": True,
                        "started_at": last_job.get("started_at"),
                        "updated_at": fin_str or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        "stage": "completed",
                        "stage_detail": "Hoàn tất mô phỏng lập lịch (--dry-run) — xem bảng quyết định tại Nhật ký.",
                        "model_id": str((last_job.get("options") or {}).get("model") or ""),
                        "zero_disk": bool((last_job.get("options") or {}).get("zero_disk", True)),
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
                elif (
                    last_type in {"upscale", "preview"}
                    and last_status in {"failed", "error"}
                    and not live_from_newer_upscale
                    and (not live_file.is_file() or last_finished_ts == 0.0 or last_finished_ts >= live_file_mtime - 1.0)
                ):
                    if live_matches_last_job:
                        if effective_stage != "error":
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
                    else:
                        # Synthesize clean failure state when job crashed before RunLogger wrote live_status.json
                        effective_stage = "error"
                        live_data = {
                            "video_name": target_label_last,
                            "pid": last_job.get("pid"),
                            "job_id": last_job.get("job_id"),
                            "job_type": last_type,
                            "started_at": last_job.get("started_at"),
                            "updated_at": fin_str or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            "stage": "error",
                            "failed_stage": "probing",
                            "last_stage": "probing",
                            "stage_detail": (
                                f"Tác vụ {last_type} thất bại ({target_label_last}, mã thoát {last_job.get('return_code')})."
                            ),
                            "model_id": str((last_job.get("options") or {}).get("model") or ""),
                            "zero_disk": bool((last_job.get("options") or {}).get("zero_disk", True)),
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
                elif (
                    last_type in {"upscale", "preview"}
                    and last_status == "aborted"
                    and not live_from_newer_upscale
                    and (not live_file.is_file() or last_finished_ts == 0.0 or last_finished_ts >= live_file_mtime - 1.0)
                ):
                    if live_matches_last_job:
                        if effective_stage not in {"completed", "paused", "error", "idle"}:
                            live_data = dict(live_data)
                            if effective_stage not in {"starting", "initializing"}:
                                live_data["last_stage"] = effective_stage
                            live_data["stage"] = "paused"
                            live_data["active_chunk_fraction"] = 0.0
                            live_data["eta_seconds"] = None
                            live_data["stage_detail"] = f"Tác vụ {last_type} đã bị dừng ({target_label_last})."
                            effective_stage = "paused"
                    else:
                        aborted_stage = "paused" if checkpoints else "idle"
                        effective_stage = aborted_stage
                        live_data = {
                            "video_name": target_label_last,
                            "pid": last_job.get("pid"),
                            "job_id": last_job.get("job_id"),
                            "job_type": last_type,
                            "started_at": last_job.get("started_at"),
                            "updated_at": fin_str or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            "stage": aborted_stage,
                            "last_stage": "probing",
                            "stage_detail": f"Tác vụ {last_type} đã bị dừng ({target_label_last}).",
                            "model_id": str((last_job.get("options") or {}).get("model") or ""),
                            "zero_disk": bool((last_job.get("options") or {}).get("zero_disk", True)),
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

            if (
                effective_stage in {"initializing", "probing", "preflight_dedup", "upscaling", "muxing", "auditing", "subtrans"}
                and not pid_alive
                and not gpu_lock.get("locked")
            ):
                stale_prev_stage = effective_stage
                effective_stage = "paused" if checkpoints else "idle"
                if live_data:
                    live_data = dict(live_data)
                    if stale_prev_stage not in {"starting", "initializing"}:
                        live_data["last_stage"] = stale_prev_stage
                    live_data["stage"] = effective_stage
                    live_data["active_chunk_fraction"] = 0.0
                    live_data["eta_seconds"] = None
                    if effective_stage == "paused":
                        v_label = str(live_data.get("video_name") or "").strip()
                        live_data["stage_detail"] = (
                            f"Phiên trước đã dừng ({v_label}) — có checkpoint sẵn sàng tiếp tục."
                            if v_label
                            else "Phiên trước đã dừng — có checkpoint sẵn sàng tiếp tục."
                        )
                    else:
                        live_data["active_chunk"] = None
                        live_data["stage_detail"] = "Hệ thống sẵn sàng chờ lệnh."

            if (
                effective_stage == "paused"
                and not pid_alive
                and not gpu_lock.get("locked")
                and live_data
                and checkpoints
            ):
                live_in_raw = str(live_data.get("input_path") or "").strip()
                live_vname = str(live_data.get("video_name") or "").strip()
                live_vstem = Path(live_vname).stem.lower() if live_vname else ""
                live_in_norm = os.path.normcase(os.path.abspath(live_in_raw)) if live_in_raw else ""
                matched_paused_cp = next(
                    (
                        cp
                        for cp in checkpoints
                        if (
                            live_in_norm
                            and cp.get("source_file")
                            and os.path.normcase(os.path.abspath(str(cp["source_file"]))) == live_in_norm
                        )
                        or (live_vstem and str(cp.get("episode_name") or "").lower() in {live_vstem, live_vname.lower()})
                    ),
                    None,
                )
                if matched_paused_cp is not None:
                    cp_done = int(matched_paused_cp.get("completed_chunks") or 0)
                    cp_tot = int(matched_paused_cp.get("total_chunks") or 0)
                    if cp_done > int(live_data.get("completed_chunks") or 0):
                        live_data = dict(live_data)
                        live_data["completed_chunks"] = cp_done
                        if cp_tot > 0 and int(live_data.get("total_chunks") or 0) <= 0:
                            live_data["total_chunks"] = cp_tot

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
        if job_state.get("status") in {"running", "pausing"} and active_job is not None:
            per_job_candidates: List[Path] = []
            raw_lp = str(active_job.get("log_path") or "").strip()
            if raw_lp:
                per_job_candidates.append(Path(raw_lp))
            jid = str(active_job.get("job_id") or "").strip()
            if jid and os.path.basename(jid) == jid:
                per_job_candidates.append(logs_dir / f"web_job_{jid}.log")
            per_job_candidates.append(web_job_log)
            for cand_active in per_job_candidates:
                if cand_active.is_file():
                    return cand_active

        live_file = logs_dir / "live_status.json"
        live_mtime = 0.0
        if live_file.is_file():
            try:
                live_mtime = live_file.stat().st_mtime
            except OSError:
                live_mtime = 0.0

        recent_logs: List[Path] = []
        try:
            recent_logs = sorted(
                (p for p in logs_dir.glob("*.log") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            recent_logs = []

        last_hist = _resolve_last_job(job_state, active_job)
        if isinstance(last_hist, dict):
            hist_lp = str(last_hist.get("log_path") or "").strip()
            if hist_lp:
                hist_path = Path(hist_lp)
                if hist_path.is_file():
                    try:
                        hist_lp_mtime = hist_path.stat().st_mtime
                    except OSError:
                        hist_lp_mtime = 0.0
                    hist_fin_ts = _parse_iso_timestamp(last_hist.get("finished_at"))
                    hist_ref_ts = max(hist_lp_mtime, hist_fin_ts)
                    hist_key = self._safe_resolve_str(hist_path)
                    newest_other_mtime = 0.0
                    for cand_recent in recent_logs:
                        if self._safe_resolve_str(cand_recent) != hist_key:
                            try:
                                newest_other_mtime = cand_recent.stat().st_mtime
                            except OSError:
                                newest_other_mtime = 0.0
                            break
                    if (
                        (live_mtime <= 0.0 or hist_ref_ts >= live_mtime - 1.0)
                        and newest_other_mtime <= hist_ref_ts + 1.0
                    ):
                        return hist_path

        raw_candidates: List[Path] = []
        if isinstance(last_hist, dict):
            hist_lp = str(last_hist.get("log_path") or "").strip()
            if hist_lp and Path(hist_lp).is_file():
                raw_candidates.append(Path(hist_lp))
        if web_job_log.is_file():
            raw_candidates.append(web_job_log)

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

        raw_candidates.extend(recent_logs[:2])

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
            with self._log_cursor_lock:
                if (
                    not self._last_log_cursor.get("path")
                    or self._poll_task is None
                    or self._poll_task.done()
                ):
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

    @staticmethod
    def _log_cursors_match(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        return (
            str(a.get("path") or "") == str(b.get("path") or "")
            and int(a.get("offset") or 0) == int(b.get("offset") or 0)
            and int(a.get("prefix_len") or 0) == int(b.get("prefix_len") or 0)
            and str(a.get("sig") or "") == str(b.get("sig") or "")
        )

    def _read_incremental_logs_with_cursor(self) -> tuple[List[str], Dict[str, Any]]:
        """Đọc phần dòng hoàn chỉnh mới phát sinh trên con trỏ dùng chung nội bộ."""
        with self._log_cursor_lock:
            prev_cursor = dict(self._last_log_cursor)
        lines, next_cur = self._read_incremental_logs_from_cursor(prev_cursor)
        with self._log_cursor_lock:
            self._last_log_cursor = dict(next_cur)
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
        return _sanitize_json_payload({
            "gpu_info": self.gpu_static,
            "telemetry": latest,
            "history": hist_copy,
            "pipeline": pipeline,
            "logs": logs,
            "log_cursor": log_cursor,
        })

    def _poll_tick_sync(self) -> Dict[str, Any]:
        """Thu thập 1 nhịp telemetry + trạng thái pipeline + log tăng dần O(1) (chạy trong threadpool)."""
        telem = self.sample_telemetry(record_history=True)
        pipe = self.read_pipeline_state()
        with self._log_cursor_lock:
            prev_cursor = dict(self._last_log_cursor)
        new_logs, next_cursor = self._read_incremental_logs_from_cursor(prev_cursor)
        with self._log_cursor_lock:
            self._last_log_cursor = dict(next_cursor)
        return _sanitize_json_payload({
            "telemetry": telem,
            "pipeline": pipe,
            "logs": new_logs,
            "log_cursor": next_cursor,
            "log_prev_cursor": prev_cursor,
        })

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
        snap_payload = _sanitize_json_payload(snap)
        yield f"event: snapshot\ndata: {json.dumps(snap_payload, ensure_ascii=False, allow_nan=False)}\n\n"
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
                base_prev = base_packet.get("log_prev_cursor")
                base_next = base_packet.get("log_cursor")
                if self._log_cursors_match(sub_cursor, base_prev):
                    new_logs = list(base_packet.get("logs") or [])
                    sub_cursor = dict(base_next) if isinstance(base_next, dict) else sub_cursor
                elif self._log_cursors_match(sub_cursor, base_next):
                    new_logs = []
                    sub_cursor = dict(base_next) if isinstance(base_next, dict) else sub_cursor
                    sub_cursor["reset"] = False
                else:
                    new_logs, sub_cursor = await asyncio.to_thread(
                        self._read_incremental_logs_from_cursor, sub_cursor
                    )
                packet = _sanitize_json_payload({
                    "telemetry": base_packet["telemetry"],
                    "pipeline": base_packet["pipeline"],
                    "logs": new_logs,
                    "log_cursor": sub_cursor,
                })
                yield f"event: tick\ndata: {json.dumps(packet, ensure_ascii=False, allow_nan=False)}\n\n"
        finally:
            self._subscribers.discard(queue)
