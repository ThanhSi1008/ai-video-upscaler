"""Subprocess job queue manager for 24/7 upscale, preview, and subtrans studio tasks."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

from web.fs_browser import VALID_NVENC_MULTIPASS_POLICIES, SafeLocalFSBrowser, is_unc_path
from web.sse_hub import check_active_gpu_lock

# Tốc độ suy luận khung hình độc nhất tham chiếu trên RTX 3060 TensorRT 10 FP16 Zero-Disk (từ STATE.md R3)
MODEL_REFERENCE_FPS: Dict[str, tuple[float, float]] = {
    "animevideov3": (10.5, 8.5),
    "animejanai-hd-v3-compact": (10.8, 8.8),
    "animejanai-hd-v3-ultracompact": (18.5, 15.5),
    "animejanai-hd-v3-sharp1-ultracompact": (18.5, 15.5),
    "animejanai-hd-v3-superultracompact": (20.8, 17.5),
    "animejanai-hd-v3-sharp1-superultracompact": (20.8, 17.5),
    "auto-pareto": (16.5, 13.5),
    "span-2x": (14.5, 12.0),
    "2x_AnimeJaNai_V3": (14.5, 12.0),
}


_FRAME_PROBE_CACHE: Dict[tuple[str, int, int], int] = {}
_FRAME_PROBE_CACHE_LOCK = threading.Lock()
_FRAME_PROBE_CACHE_MAX = 2048


def _probe_frames_lightweight(video_path: str) -> int:
    """Đo nhanh tổng số frame video bằng PyAV in-process (cache theo abs_path, mtime_ns, size)."""
    cache_key: Optional[tuple[str, int, int]] = None
    try:
        abs_p = os.path.abspath(video_path)
        st = os.stat(abs_p)
        cache_key = (os.path.normcase(abs_p), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        cache_key = None

    if cache_key is not None:
        with _FRAME_PROBE_CACHE_LOCK:
            cached = _FRAME_PROBE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    probed = 34000
    try:
        import av
        with av.open(video_path) as container:
            if container.streams.video:
                vs = container.streams.video[0]
                if vs.frames and vs.frames > 0:
                    probed = int(vs.frames)
                else:
                    dur = float(container.duration / av.time_base) if container.duration else 0.0
                    fps = float(vs.average_rate) if vs.average_rate else 23.976
                    if dur > 0:
                        probed = max(1, int(round(dur * fps)))
    except Exception:
        pass

    if cache_key is not None:
        with _FRAME_PROBE_CACHE_LOCK:
            if len(_FRAME_PROBE_CACHE) >= _FRAME_PROBE_CACHE_MAX:
                _FRAME_PROBE_CACHE.clear()
            _FRAME_PROBE_CACHE[cache_key] = probed
    return probed


def _has_valid_upscaled_output(out_4k: Path) -> bool:
    """Kiểm tra bản thành phẩm 4K tồn tại (>1 MiB) và không có biên bản kiểm định FAIL liền kề."""
    try:
        if not (out_4k.is_file() and out_4k.stat().st_size > 1024 * 1024):
            return False
    except OSError:
        return False

    audit_cands = [
        out_4k.with_suffix(".audit.json"),
        out_4k.parent / "audit" / f"{out_4k.stem}.audit.json",
    ]
    for ac in audit_cands:
        if ac.is_file():
            try:
                adata = json.loads(ac.read_text(encoding="utf-8"))
                if isinstance(adata, dict):
                    status = str(adata.get("status") or "").strip().upper()
                    if status.startswith("FAIL"):
                        return False
            except Exception:
                pass
    return True


_VALID_SUBS_QUALITIES = {"normal", "fast", "high"}
_DEADLINE_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_VIDEO_FILE_SUFFIXES = {".mkv", ".mp4", ".webm", ".mov", ".avi", ".ts", ".m2ts"}


def _validate_no_nul(val: Optional[str], field: str = "path") -> Optional[str]:
    if val is not None:
        if "\x00" in val:
            raise ValueError(f"Trường '{field}' chứa ký tự NUL (\\x00) không hợp lệ.")
        if is_unc_path(val):
            raise ValueError(f"Từ chối đường dẫn mạng UNC tại trường '{field}' để bảo mật máy chủ cục bộ.")
    return val


def _validate_subs_quality(val: Optional[str]) -> str:
    norm = (val if val is not None else "normal").strip().lower()
    if norm not in _VALID_SUBS_QUALITIES:
        raise ValueError(
            f"Giá trị subs_quality '{val}' không hợp lệ (chỉ chấp nhận normal, fast hoặc high)."
        )
    return norm


def _validate_nvenc_multipass(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    norm = str(val).strip().lower()
    if not norm:
        return None
    if norm not in VALID_NVENC_MULTIPASS_POLICIES:
        raise ValueError(
            f"Giá trị nvenc_multipass '{val}' không hợp lệ (chỉ chấp nhận auto, fullres hoặc qres)."
        )
    return norm


def _validate_deadline(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    m = _DEADLINE_RE.match(s)
    if not m:
        raise ValueError(f"Định dạng deadline '{val}' không hợp lệ (cần HH:MM, ví dụ 07:00).")
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Giờ deadline '{val}' phải nằm trong 00:00..23:59.")
    return f"{hh:02d}:{mm:02d}"


def _resolve_output_target(workspace_root: Path, output_dir: Optional[str]) -> tuple[Path, Optional[Path]]:
    """
    Chuẩn hóa đường dẫn output (-o):
    - Đường dẫn tương đối được phân giải theo workspace_root (không dùng os.getcwd()).
    - Nếu trỏ thẳng tới tệp video (.mkv, .mp4, ...), tách thành (thư mục cha, tệp đích).
    """
    if not output_dir or not str(output_dir).strip():
        return (workspace_root / "out").resolve(), None
    raw = str(output_dir).strip()
    _validate_no_nul(raw, "output_dir")
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = workspace_root / p
    resolved = p.resolve()
    if resolved.suffix.lower() in _VIDEO_FILE_SUFFIXES:
        return resolved.parent, resolved
    return resolved, None


_MAX_RETAINED_WEB_JOB_LOGS = 20


def _prune_old_web_job_logs(
    logs_dir: Path,
    active_names: set[str],
    max_keep: int = _MAX_RETAINED_WEB_JOB_LOGS,
) -> None:
    """Keep at most max_keep newest web_job_*.log files while preserving active job logs."""
    candidates: List[tuple[tuple[int, str], Path]] = []
    try:
        for p in logs_dir.glob("web_job_*.log"):
            if not p.is_file() or p.name in active_names:
                continue
            try:
                mtime_ns = int(p.stat().st_mtime_ns)
            except OSError:
                mtime_ns = 0
            candidates.append(((mtime_ns, p.name), p))
    except OSError:
        return
    inactive_budget = max(0, max_keep - len(active_names))
    if len(candidates) <= inactive_budget:
        return
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, old_path in candidates[inactive_budget:]:
        try:
            old_path.unlink()
        except OSError:
            pass


def _find_cached_plan(
    workspace_root: Path,
    video_path: str,
    output_dir: Optional[str] = None,
    expected_policy: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_base, _ = _resolve_output_target(workspace_root, output_dir)
    vid_dir = Path(video_path).resolve().parent
    candidates = [
        out_base / f".temp_{base_name}" / "plan.json",
        workspace_root / "out" / f".temp_{base_name}" / "plan.json",
        vid_dir / f".temp_{base_name}" / "plan.json",
        out_base / f"{base_name}.plan.json",
        Path(os.path.splitext(video_path)[0] + ".plan.json"),
    ]
    for c in candidates:
        if c.is_file():
            try:
                data = json.loads(c.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("total_kept_frames"):
                    plan_pol = data.get("dedup_policy")
                    if expected_policy and plan_pol is not None:
                        if str(plan_pol).strip().lower() != str(expected_policy).strip().lower():
                            continue
                    return data
            except Exception:
                pass
    return None


class JobManager:
    """Quản lý thực thi tác vụ nền và lập lịch hàng đợi cho Web GUI."""

    def __init__(
        self,
        workspace_root: Path,
        python_executable: Optional[str] = None,
        popen_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.python_executable = python_executable or sys.executable
        self.popen_factory = popen_factory or subprocess.Popen
        self.fs_browser = SafeLocalFSBrowser(self.workspace_root)

        self._lock = threading.Lock()
        self._proc: Optional[Any] = None
        self._active_job: Optional[Dict[str, Any]] = None
        self._history: Deque[Dict[str, Any]] = deque(maxlen=25)

    def get_state(self) -> Dict[str, Any]:
        """Trả về bản sao an toàn trạng thái tác vụ hiện tại & lịch sử gần đây."""
        with self._lock:
            if self._active_job and self._proc is not None:
                rc = self._proc.poll()
                if rc is not None and self._active_job["status"] in {"running", "pausing"}:
                    self._finalize_locked(rc)
            active = dict(self._active_job) if self._active_job else None
            hist = [dict(x) for x in self._history]
        return {
            "status": active["status"] if active else "idle",
            "active_job": active,
            "history": hist,
        }

    def _finalize_locked(self, return_code: int) -> None:
        if not self._active_job or self._active_job.get("finished_at") is not None:
            return
        prev_status = self._active_job.get("status")
        job_pid = self._active_job.get("pid")

        live_stage = ""
        live_file = self.workspace_root / "logs" / "live_status.json"
        if live_file.is_file():
            try:
                ldata = json.loads(live_file.read_text(encoding="utf-8"))
                if isinstance(ldata, dict) and job_pid and ldata.get("pid") == job_pid:
                    live_stage = str(ldata.get("stage") or "")
            except Exception:
                live_stage = ""

        # Dọn cờ .graceful_pause.flag còn sót (ví dụ người dùng bấm dừng êm lúc đang Mux cuối tập)
        pause_flag = self.workspace_root / "logs" / ".graceful_pause.flag"
        if pause_flag.exists():
            try:
                pause_flag.unlink()
            except OSError:
                pass

        if prev_status == "aborted":
            final_status = "aborted"
        elif live_stage == "error":
            final_status = "failed"
        elif live_stage == "paused":
            final_status = "paused"
        elif prev_status == "pausing" and return_code in (0, 1) and live_stage != "completed":
            final_status = "paused"
        elif return_code == 0:
            final_status = "completed"
        else:
            final_status = "failed"

        self._active_job["status"] = final_status
        self._active_job["return_code"] = int(return_code)
        self._active_job["finished_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._history.appendleft(dict(self._active_job))
        self._proc = None

    def compute_schedule_preview(
        self,
        inputs: Sequence[str],
        deadline_str: Optional[str] = None,
        allow_partial: bool = False,
        model_override: Optional[str] = None,
        dedup_override: Optional[str] = None,
        zero_disk: bool = True,
        use_tensorrt: bool = True,
        subs_after_upscale: bool = True,
        output_dir: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Dự báo chi tiết danh sách tập, model phân giải từ shows.yaml và thời gian hoàn thành theo --deadline."""
        videos = self.fs_browser.expand_inputs(inputs)
        deadline_str = _validate_deadline(deadline_str)
        now = datetime.now()
        deadline_dt: Optional[datetime] = None
        if deadline_str:
            parts = deadline_str.split(":")
            dh, dm = int(parts[0]), int(parts[1])
            deadline_dt = now.replace(hour=dh, minute=dm, second=0, microsecond=0)
            if deadline_dt <= now:
                deadline_dt += timedelta(days=1)

        routing_tables = self.fs_browser._load_routing_tables()
        out_base, explicit_out_file = _resolve_output_target(self.workspace_root, output_dir)
        sub_overhead = 300.0 if subs_after_upscale else 0.0
        if zero_disk and use_tensorrt:
            vid_overhead = 420.0
        elif not zero_disk and use_tensorrt:
            vid_overhead = 660.0
        elif zero_disk and not use_tensorrt:
            vid_overhead = 540.0
        else:
            vid_overhead = 780.0
        total_overhead = vid_overhead + sub_overhead

        sim_med = now
        sim_p5 = now
        episodes: List[Dict[str, Any]] = []
        total_sched_s = 0.0

        for idx, vid in enumerate(videos):
            base_name = os.path.splitext(os.path.basename(vid))[0]
            routing = self.fs_browser.resolve_routing_for_path(
                vid,
                override_model=model_override,
                override_dedup=dedup_override,
                routing_tables=routing_tables,
            )
            m_id = routing["model_id"]
            base_med_fps, base_p5_fps = MODEL_REFERENCE_FPS.get(m_id, (10.5, 8.5))
            if not use_tensorrt:
                base_med_fps *= 0.52
                base_p5_fps *= 0.52
            if not zero_disk:
                base_med_fps *= 0.82
                base_p5_fps *= 0.82

            out_suffix = "_4K_animejanai.mkv" if m_id.startswith("animejanai-") else "_4K.mkv"
            if explicit_out_file is not None and len(videos) == 1:
                out_4k = explicit_out_file
            else:
                out_4k = out_base / f"{base_name}{out_suffix}"

            already_done = (not force) and _has_valid_upscaled_output(out_4k)

            plan = _find_cached_plan(
                self.workspace_root,
                vid,
                str(out_base),
                expected_policy=routing["dedup_policy"],
            )
            if already_done:
                if plan:
                    total_f = int(plan.get("total_frames") or 0)
                    n_unique = int(plan.get("total_kept_frames") or 0)
                    dedup_pct = float(plan.get("dedup_ratio_percent") or 0.0)
                    source_note = "đã có 4K (plan.json)"
                else:
                    total_f = 0
                    n_unique = 0
                    dedup_pct = 0.0
                    source_note = "đã có 4K (bỏ qua probe)"
            elif plan:
                total_f = int(plan.get("total_frames") or 34000)
                n_unique = int(plan.get("total_kept_frames") or int(total_f * 0.6))
                dedup_pct = float(plan.get("dedup_ratio_percent") or 0.0)
                source_note = "plan.json (đã quét)"
            else:
                total_f = _probe_frames_lightweight(vid)
                pol = routing["dedup_policy"]
                keep_factor = 1.0 if pol == "off" else 0.60
                n_unique = max(1, int(round(total_f * keep_factor)))
                dedup_pct = round((1.0 - keep_factor) * 100.0, 1)
                source_note = "ước tính thận trọng"

            if already_done:
                eta_med_s = 0.0
                dec_eta_s = 0.0
                can_run = False
                is_partial = False
                decision = "ĐÃ CÓ 4K (BỎ QUA)"
                ep_start = sim_med
                ep_finish = sim_med
            else:
                eta_med_s = (n_unique / max(0.5, base_med_fps)) + total_overhead
                eta_p5_s = (n_unique / max(0.5, base_p5_fps)) + total_overhead
                dec_eta_s = eta_p5_s * 1.07
                ep_start = sim_med
                ep_finish = sim_med + timedelta(seconds=eta_med_s)
                ep_dec_finish = sim_p5 + timedelta(seconds=dec_eta_s)

                can_run = True
                is_partial = False
                decision = "KỊP HẠN CHÓT" if deadline_dt else "SẴN SÀNG CHẠY"

                if deadline_dt:
                    eff_deadline = deadline_dt
                    if ep_dec_finish <= eff_deadline:
                        decision = "KỊP HẠN CHÓT"
                    elif allow_partial and eff_deadline > sim_p5:
                        decision = "CHẠY 1 PHẦN (GIỮ CHECKPOINT)"
                        is_partial = True
                    else:
                        decision = "QUÁ DEADLINE (TỰ BỎ QUA)"
                        can_run = False

                if can_run:
                    if is_partial and deadline_dt:
                        part_s = max(0.0, (eff_deadline - sim_p5).total_seconds())
                        sim_p5 = eff_deadline
                        sim_med += timedelta(seconds=part_s)
                        total_sched_s += part_s
                    else:
                        sim_p5 = ep_dec_finish
                        sim_med = ep_finish
                        total_sched_s += eta_med_s

            episodes.append({
                "index": idx + 1,
                "file_path": vid,
                "file_name": os.path.basename(vid),
                "model_id": m_id,
                "model_name": routing["model_name"],
                "dedup_policy": routing["dedup_policy"],
                "routing_source": routing["resolution_source"],
                "total_frames": total_f,
                "n_unique": n_unique,
                "dedup_ratio_percent": round(dedup_pct, 1),
                "n_unique_source": source_note,
                "ref_fps": round(base_med_fps, 1),
                "eta_minutes": round(eta_med_s / 60.0, 1),
                "start_time": ep_start.strftime("%H:%M"),
                "finish_time": ep_finish.strftime("%H:%M"),
                "can_run": can_run,
                "is_partial": is_partial,
                "already_done": already_done,
                "decision": decision,
            })

        return {
            "total_episodes": len(episodes),
            "accepted_count": sum(1 for e in episodes if e["can_run"]),
            "already_done_count": sum(1 for e in episodes if e["already_done"]),
            "skipped_count": sum(1 for e in episodes if not e["can_run"] and not e["already_done"]),
            "total_eta_minutes": round(total_sched_s / 60.0, 1),
            "deadline_str": deadline_str or "",
            "episodes": episodes,
        }

    def _ensure_ready_to_launch_locked(self, requires_gpu: bool = True) -> None:
        if self._active_job and self._proc is not None and self._proc.poll() is None:
            raise RuntimeError(
                f"Đang có một tiến trình Web Job hoạt động (PID={self._proc.pid}, loại={self._active_job['job_type']}). "
                f"Vui lòng chờ xong hoặc dừng trước khi chạy tác vụ mới."
            )
        if requires_gpu:
            lock_info = check_active_gpu_lock(self.workspace_root)
            if lock_info.get("locked"):
                raise RuntimeError(
                    f"GPU đang bận xử lý tiến trình khác (PID={lock_info['pid']}). "
                    f"Chỉ cho phép 1 tiến trình suy luận GPU hoạt động tại một thời điểm."
                )
            live_file = self.workspace_root / "logs" / "live_status.json"
            if live_file.is_file():
                try:
                    ldata = json.loads(live_file.read_text(encoding="utf-8"))
                    if isinstance(ldata, dict):
                        live_pid = ldata.get("pid")
                        live_stage = str(ldata.get("stage") or "").strip().lower()
                        if (
                            not isinstance(live_pid, bool)
                            and isinstance(live_pid, int)
                            and live_pid > 0
                            and live_stage in {
                                "initializing",
                                "probing",
                                "preflight_dedup",
                                "upscaling",
                                "muxing",
                                "auditing",
                            }
                        ):
                            import psutil
                            if psutil.pid_exists(live_pid):
                                raise RuntimeError(
                                    f"Đang có tiến trình Upscale CLI khác hoạt động ở giai đoạn '{live_stage}' (PID={live_pid}). "
                                    f"Vui lòng chờ tiến trình đó kết thúc hoặc tạm dừng trước khi khởi chạy tác vụ GPU mới."
                                )
                except RuntimeError:
                    raise
                except Exception:
                    pass

    def _ensure_ready_to_launch(self, requires_gpu: bool = True) -> None:
        with self._lock:
            self._ensure_ready_to_launch_locked(requires_gpu=requires_gpu)

    def _spawn_command(
        self,
        job_type: str,
        cmd: List[str],
        queue_items: List[str],
        options: Dict[str, Any],
        extra_env: Optional[Dict[str, str]] = None,
        requires_gpu: bool = True,
    ) -> Dict[str, Any]:
        logs_dir = self.workspace_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        pause_flag = logs_dir / ".graceful_pause.flag"

        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if extra_env:
            env.update(extra_env)

        popen_kwargs: Dict[str, Any] = {
            "cwd": str(self.workspace_root),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            popen_kwargs["startupinfo"] = startupinfo

        # Giữ self._lock xuyên suốt kiểm tra sẵn sàng + khởi tạo tiến trình con + đăng ký state để triệt tiêu TOCTOU race
        with self._lock:
            self._ensure_ready_to_launch_locked(requires_gpu=requires_gpu)
            if pause_flag.exists():
                try:
                    pause_flag.unlink()
                except OSError:
                    pass
            proc = self.popen_factory(cmd, **popen_kwargs)
            if proc is None:
                raise RuntimeError("Không thể khởi tạo tiến trình con (popen_factory trả về None).")
            job_id = uuid.uuid4().hex[:10]
            web_log_path = logs_dir / f"web_job_{job_id}.log"
            now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            web_log_path.write_text(
                f"# Web Studio Job {job_id} ({job_type}) started at {now_iso}\n",
                encoding="utf-8",
            )
            active_log_names: set[str] = {web_log_path.name}
            if (
                self._active_job
                and self._active_job.get("status") in {"running", "pausing"}
                and self._active_job.get("log_path")
            ):
                active_log_names.add(Path(str(self._active_job["log_path"])).name)
            _prune_old_web_job_logs(logs_dir, active_log_names)
            job_record: Dict[str, Any] = {
                "job_id": job_id,
                "job_type": job_type,
                "status": "running",
                "pid": getattr(proc, "pid", None),
                "started_at": now_iso,
                "finished_at": None,
                "return_code": None,
                "log_path": str(web_log_path),
                "command": cmd,
                "queue_items": queue_items,
                "options": options,
            }
            self._proc = proc
            self._active_job = job_record
            initial_snapshot = dict(job_record)

        def _drain_output() -> None:
            rc = -1
            try:
                with web_log_path.open("a", encoding="utf-8") as wf:
                    if proc is not None:
                        stdout_pipe = getattr(proc, "stdout", None)
                        if stdout_pipe is not None:
                            for raw_line in stdout_pipe:
                                wf.write(raw_line)
                                wf.flush()
                if proc is not None:
                    rc = int(proc.wait())
            except Exception:
                if proc is not None:
                    try:
                        polled = proc.poll()
                        rc = int(polled) if polled is not None else -1
                    except Exception:
                        rc = -1
            finally:
                with self._lock:
                    if self._active_job and self._active_job.get("job_id") == job_id:
                        self._finalize_locked(int(rc))

        t = threading.Thread(target=_drain_output, name=f"WebJobDrain-{job_id}", daemon=True)
        t.start()
        return initial_snapshot

    def start_upscale_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Khởi chạy hàng đợi hoặc tập đơn lẻ qua upscale.py."""
        raw_inputs = payload.get("inputs") or []
        if not raw_inputs:
            raise ValueError("Vui lòng chọn ít nhất một file video hoặc thư mục nguồn.")
        for item in raw_inputs:
            _validate_no_nul(str(item), "inputs")
        if payload.get("output_dir"):
            _validate_no_nul(str(payload.get("output_dir")), "output_dir")

        subs_quality = _validate_subs_quality(payload.get("subs_quality"))
        deadline_norm = _validate_deadline(payload.get("deadline"))
        nvenc_mp_norm = _validate_nvenc_multipass(payload.get("nvenc_multipass"))

        videos = self.fs_browser.expand_inputs(raw_inputs)
        dry_run = bool(payload.get("dry_run", False))
        self._ensure_ready_to_launch(requires_gpu=not dry_run)

        upscale_script = str(self.workspace_root / "upscale.py")
        cmd = [self.python_executable, upscale_script]
        if len(videos) > 1:
            cmd.extend(["--queue", *videos])
        else:
            cmd.append(videos[0])

        if payload.get("output_dir"):
            cmd.extend(["-o", str(payload["output_dir"]).strip()])
        if payload.get("model"):
            cmd.extend(["--model", str(payload["model"]).strip()])
        if payload.get("dedup_policy"):
            cmd.extend(["--dedup-policy", str(payload["dedup_policy"]).strip()])
        if nvenc_mp_norm:
            cmd.extend(["--nvenc-multipass", nvenc_mp_norm])
        if payload.get("batch_size") is not None:
            bs = int(payload["batch_size"])
            if bs < 1:
                raise ValueError("batch_size phải >= 1.")
            cmd.extend(["--batch-size", str(bs)])
        if deadline_norm:
            cmd.extend(["--deadline", deadline_norm])
        if payload.get("allow_partial"):
            cmd.append("--allow-partial")
        if payload.get("zero_disk") is False:
            cmd.append("--no-zero-disk")
        if payload.get("use_tensorrt") is False:
            cmd.append("--no-tensorrt")
        if payload.get("subs_after_upscale") is False:
            cmd.append("--no-subs")
        cmd.extend(["--subs-quality", subs_quality])
        if payload.get("async_subs"):
            cmd.append("--async-subs")
        if payload.get("prefer_whisper"):
            cmd.append("--prefer-whisper")
        if payload.get("auto_align_en") is False:
            cmd.append("--no-auto-align-en")
        if payload.get("sub_offset_ms"):
            cmd.extend(["--sub-offset-ms", str(int(payload["sub_offset_ms"]))])
        if payload.get("keep_temp"):
            cmd.append("--keep-temp")
        if payload.get("force"):
            cmd.append("--force")
        if dry_run:
            cmd.append("--dry-run")

        return self._spawn_command("upscale", cmd, videos, payload, requires_gpu=not dry_run)

    def start_preview_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Khởi chạy xem trước nhanh (--preview N giây ở 30% tập phim)."""
        video_raw = str(payload.get("video_path") or "").strip()
        _validate_no_nul(video_raw, "video_path")
        if not video_raw:
            raise ValueError("Vui lòng chọn một file video để trích xuất Preview.")
        videos = self.fs_browser.expand_inputs([video_raw])
        target_video = videos[0]
        raw_sec = payload.get("seconds", 10)
        sec_f = float(10 if raw_sec is None else raw_sec)
        if not math.isfinite(sec_f):
            raise ValueError("seconds phải là số hữu hạn.")
        seconds = max(3, min(60, int(sec_f)))
        nvenc_mp_norm = _validate_nvenc_multipass(payload.get("nvenc_multipass"))
        self._ensure_ready_to_launch(requires_gpu=True)

        cmd = [
            self.python_executable,
            str(self.workspace_root / "upscale.py"),
            target_video,
            "--preview",
            str(seconds),
            "--no-subs",
        ]
        if payload.get("model"):
            cmd.extend(["--model", str(payload["model"]).strip()])
        if payload.get("dedup_policy"):
            cmd.extend(["--dedup-policy", str(payload["dedup_policy"]).strip()])
        if nvenc_mp_norm:
            cmd.extend(["--nvenc-multipass", nvenc_mp_norm])
        if payload.get("zero_disk") is False:
            cmd.append("--no-zero-disk")
        if payload.get("use_tensorrt") is False:
            cmd.append("--no-tensorrt")
        if payload.get("force", True):
            cmd.append("--force")

        extra_env = {}
        if not payload.get("auto_open_player", False):
            extra_env["UPSCALER_NO_AUTO_OPEN"] = "1"

        return self._spawn_command("preview", cmd, [target_video], payload, extra_env=extra_env, requires_gpu=True)

    def start_subtrans_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Khởi chạy dịch phụ đề trực tiếp (--subs-only hoặc subtrans.py) bình thường ở mọi khung giờ."""
        target_raw = str(payload.get("input_path") or "").strip()
        _validate_no_nul(target_raw, "input_path")
        if payload.get("output_dir"):
            _validate_no_nul(str(payload.get("output_dir")), "output_dir")
        if payload.get("glossary"):
            _validate_no_nul(str(payload.get("glossary")), "glossary")

        if not target_raw:
            raise ValueError("Vui lòng chọn file video (.mkv/.mp4) hoặc phụ đề (.srt/.ass) để dịch.")

        subs_quality = _validate_subs_quality(payload.get("subs_quality"))
        items = self.fs_browser.expand_inputs([target_raw])
        prefer_whisper = bool(payload.get("prefer_whisper", False))
        dry_run = bool(payload.get("dry_run", False))
        needs_gpu = bool(prefer_whisper and not dry_run)
        self._ensure_ready_to_launch(requires_gpu=needs_gpu)

        glossary_val = str(payload.get("glossary") or "").strip()
        sub_exts = {".srt", ".ass", ".ssa", ".vtt"}
        is_subtitle_only = len(items) == 1 and os.path.splitext(items[0])[1].lower() in sub_exts

        if is_subtitle_only:
            target_file = items[0]
            out_dir = str(payload.get("output_dir") or (self.workspace_root / "out"))
            cmd = [
                self.python_executable,
                str(self.workspace_root / "subtrans.py"),
                "run",
                target_file,
                "-o",
                out_dir,
                "--subs-quality",
                subs_quality,
            ]
            if glossary_val:
                cmd.extend(["--glossary", glossary_val])
            if payload.get("subs_upgrade"):
                cmd.append("--subs-upgrade")
            if payload.get("sub_offset_ms"):
                cmd.extend(["--sub-offset-ms", str(int(payload["sub_offset_ms"]))])
            if payload.get("auto_align_en") is False:
                cmd.append("--no-auto-align-en")
            if dry_run:
                cmd.append("--dry-run")
        else:
            cmd = [
                self.python_executable,
                str(self.workspace_root / "upscale.py"),
                "--subs-only",
            ]
            if len(items) == 1:
                cmd.append(items[0])
            else:
                cmd.extend(["--queue", *items])
            cmd.extend(["--subs-quality", subs_quality])
            if payload.get("output_dir"):
                cmd.extend(["-o", str(payload["output_dir"]).strip()])
            if glossary_val:
                cmd.extend(["--glossary", glossary_val])
            if payload.get("subs_upgrade"):
                cmd.append("--subs-upgrade")
            if prefer_whisper:
                cmd.append("--prefer-whisper")
            if payload.get("auto_align_en") is False:
                cmd.append("--no-auto-align-en")
            if payload.get("no_embed"):
                cmd.append("--no-embed")
            if payload.get("sub_offset_ms"):
                cmd.extend(["--sub-offset-ms", str(int(payload["sub_offset_ms"]))])
            if dry_run:
                cmd.append("--dry-run")

        return self._spawn_command("subtrans", cmd, items, payload, requires_gpu=needs_gpu)

    def start_subs_pending_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Chạy lại danh sách phụ đề tồn đọng trong pending_subs.json."""
        subs_quality = _validate_subs_quality(payload.get("subs_quality"))
        dry_run = bool(payload.get("dry_run", False))
        self._ensure_ready_to_launch(requires_gpu=False)
        cmd = [
            self.python_executable,
            str(self.workspace_root / "upscale.py"),
            "--subs-pending",
            "--subs-quality",
            subs_quality,
        ]
        if dry_run:
            cmd.append("--dry-run")
        return self._spawn_command("subs_pending", cmd, ["pending_subs.json"], payload, requires_gpu=False)

    def request_graceful_pause(self) -> Dict[str, Any]:
        """Ghi cờ dừng êm logs/.graceful_pause.flag cho cả tiến trình Web lẫn CLI ngoài."""
        logs_dir = self.workspace_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        flag_path = logs_dir / ".graceful_pause.flag"
        tmp_path = logs_dir / f".graceful_pause.flag.{os.getpid()}.tmp"
        data = {
            "requested_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "reason": "web_gui_graceful_pause",
        }
        tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, flag_path)
        with self._lock:
            if (
                self._active_job
                and self._active_job.get("status") == "running"
                and self._active_job.get("job_type") in {"upscale", "preview"}
            ):
                self._active_job["status"] = "pausing"
        return {"ok": True, "graceful_pause_pending": True, "flag_path": str(flag_path)}

    def clear_graceful_pause(self) -> Dict[str, Any]:
        """Hủy cờ dừng êm nếu chunk hiện tại chưa kết thúc."""
        flag_path = self.workspace_root / "logs" / ".graceful_pause.flag"
        try:
            if flag_path.exists():
                flag_path.unlink()
        except OSError:
            pass
        with self._lock:
            if self._active_job and self._active_job.get("status") == "pausing":
                self._active_job["status"] = "running"
        return {"ok": True, "graceful_pause_pending": False}

    def abort_job(self) -> Dict[str, Any]:
        """Buộc dừng ngay tiến trình con do Web GUI khởi chạy và dọn khóa GPU mồ côi."""
        killed_pid: Optional[int] = None
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                killed_pid = getattr(proc, "pid", None)
                if self._active_job:
                    self._active_job["status"] = "aborted"
                try:
                    if (
                        sys.platform == "win32"
                        and killed_pid
                        and self.popen_factory is subprocess.Popen
                    ):
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        startupinfo.wShowWindow = subprocess.SW_HIDE
                        subprocess.run(
                            ["taskkill", "/PID", str(killed_pid), "/T", "/F"],
                            capture_output=True,
                            timeout=5.0,
                            check=False,
                            startupinfo=startupinfo,
                        )
                    else:
                        proc.terminate()
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        # Thu hồi khóa .gpu_active.lock nếu thuộc PID vừa kết thúc hoặc phát hiện tiến trình CLI ngoài
        lock_file = self.workspace_root / ".gpu_active.lock"
        external_lock_pid: Optional[int] = None
        if lock_file.is_file():
            lock_state = check_active_gpu_lock(self.workspace_root)
            if lock_state.get("locked"):
                external_lock_pid = lock_state.get("pid")
            else:
                try:
                    lock_file.unlink()
                except OSError:
                    pass

        if killed_pid is not None:
            return {
                "ok": True,
                "aborted_pid": killed_pid,
                "status": "aborted",
                "message": f"Đã dừng tiến trình Web Job (PID={killed_pid}).",
            }
        if external_lock_pid is not None:
            return {
                "ok": False,
                "aborted_pid": None,
                "status": "external_lock",
                "external_lock_pid": external_lock_pid,
                "message": (
                    f"Không có Web Job đang chạy; tiến trình GPU bên ngoài (PID={external_lock_pid}) "
                    f"vẫn đang hoạt động và không thuộc quyền hủy từ Web Studio."
                ),
            }
        return {
            "ok": False,
            "aborted_pid": None,
            "status": "idle",
            "message": "Không có tác vụ Web Job nào đang chạy để dừng.",
        }
