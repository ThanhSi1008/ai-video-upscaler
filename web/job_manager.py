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
import time
import uuid
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

from web.fs_browser import (
    AUTO_PARETO_MODEL_ID,
    VALID_GRAIN_RESTORE_POLICIES,
    VALID_NVENC_MULTIPASS_POLICIES,
    SafeLocalFSBrowser,
    inspect_model_trt_cache,
    is_secret_filename,
    is_unc_path,
    strip_wrapping_quotes,
    validate_win32_path_hygiene,
)
from web.sse_hub import _coerce_pid, check_active_gpu_lock, live_matches_web_job

# Tốc độ suy luận khung hình độc nhất tham chiếu trên RTX 3060 TensorRT 10 FP16 Zero-Disk (từ STATE.md R3)
MODEL_REFERENCE_FPS: Dict[str, tuple[float, float]] = {
    "animevideov3": (10.5, 8.5),
    "animejanai-hd-v3-compact": (12.8, 10.2),
    "animejanai-hd-v3-ultracompact": (18.5, 15.5),
    "animejanai-hd-v3-sharp1-ultracompact": (18.5, 15.5),
    "animejanai-hd-v3-superultracompact": (20.8, 17.5),
    "animejanai-hd-v3-sharp1-superultracompact": (20.8, 17.5),
    "auto-pareto": (16.5, 13.5),
    "span-2x": (10.5, 8.5),
    "2x_AnimeJaNai_V3": (10.5, 8.5),
    "modernspanimation-v2-2x": (10.5, 8.5),
    "realcugan-pro-conservative-2x": (4.5, 3.6),
    "realcugan-pro-denoise3x-2x": (4.5, 3.6),
}


_FRAME_PROBE_CACHE: Dict[tuple[str, int, int], int] = {}
_DIMS_PROBE_CACHE: Dict[tuple[str, int, int], tuple[int, int]] = {}
_FRAME_PROBE_CACHE_LOCK = threading.Lock()
_FRAME_PROBE_CACHE_MAX = 2048


def _probe_video_meta_lightweight(video_path: str) -> tuple[int, int, int]:
    """Probe (frames, height, width) in a single PyAV header open cached by (abs_path, mtime_ns, size)."""
    cache_key: Optional[tuple[str, int, int]] = None
    try:
        abs_p = os.path.abspath(video_path)
        st = os.stat(abs_p)
        cache_key = (os.path.normcase(abs_p), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        cache_key = None

    if cache_key is not None:
        with _FRAME_PROBE_CACHE_LOCK:
            cached_f = _FRAME_PROBE_CACHE.get(cache_key)
            cached_hw = _DIMS_PROBE_CACHE.get(cache_key)
        if cached_f is not None and cached_hw is not None:
            return cached_f, cached_hw[0], cached_hw[1]

    probed = 34000
    v_h, v_w = 0, 0
    try:
        import av
        with av.open(video_path) as container:
            if container.streams.video:
                vs = container.streams.video[0]
                v_h = int(vs.height or 0)
                v_w = int(vs.width or 0)
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
                _DIMS_PROBE_CACHE.clear()
            _FRAME_PROBE_CACHE[cache_key] = probed
            _DIMS_PROBE_CACHE[cache_key] = (v_h, v_w)
    return probed, v_h, v_w


def _probe_frames_lightweight(video_path: str) -> int:
    """Đo nhanh tổng số frame video bằng PyAV in-process (cache theo abs_path, mtime_ns, size)."""
    frames, _, _ = _probe_video_meta_lightweight(video_path)
    return frames


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
_ASCII_INT_RE = re.compile(r"^[0-9]+$")
_ASCII_NONNEG_FLOAT_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
_VIDEO_FILE_SUFFIXES = {".mkv", ".mp4", ".webm", ".mov", ".avi", ".ts", ".m2ts"}


def parse_preview_start_value(raw: Any) -> Optional[float]:
    """Parse and validate preview_start_sec in seconds or strict MM:SS / HH:MM:SS format."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, bool):
        raise ValueError("preview_start_sec không được là kiểu boolean.")
    if isinstance(raw, (int, float)):
        sec = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if ":" in text:
            parts = text.split(":")
            if len(parts) not in (2, 3):
                raise ValueError("preview_start_sec phải là số giây hoặc MM:SS / HH:MM:SS.")
            if not all(_ASCII_INT_RE.fullmatch(p) for p in parts[:-1]) or not _ASCII_NONNEG_FLOAT_RE.fullmatch(parts[-1]):
                raise ValueError("Định dạng thời gian MM:SS / HH:MM:SS không hợp lệ (HH/MM phải là số nguyên >= 0).")
            ss = float(parts[-1])
            if ss >= 60.0:
                raise ValueError("Số giây trong MM:SS / HH:MM:SS phải nhỏ hơn 60.")
            if len(parts) == 2:
                sec = int(parts[0]) * 60.0 + ss
            else:
                mm = int(parts[1])
                if mm >= 60:
                    raise ValueError("Số phút trong HH:MM:SS phải nhỏ hơn 60.")
                sec = int(parts[0]) * 3600.0 + mm * 60.0 + ss
        else:
            if not _ASCII_NONNEG_FLOAT_RE.fullmatch(text):
                raise ValueError("preview_start_sec phải là số giây >= 0 hoặc MM:SS / HH:MM:SS.")
            sec = float(text)
    else:
        raise ValueError("preview_start_sec phải là số giây hoặc chuỗi MM:SS / HH:MM:SS.")
    if not math.isfinite(sec) or sec < 0.0 or sec > 86400.0:
        raise ValueError("preview_start_sec phải nằm trong khoảng 0..86400 giây.")
    return round(sec, 3)


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


def _validate_sound_cues(val: Optional[str]) -> str:
    norm = (val if val is not None else "drop").strip().lower() or "drop"
    if norm not in {"drop", "translate"}:
        raise ValueError("sound_cues chỉ chấp nhận 'drop' hoặc 'translate'.")
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


def _validate_grain_restore(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    norm = str(val).strip().lower()
    if not norm:
        return None
    if norm not in VALID_GRAIN_RESTORE_POLICIES:
        raise ValueError(
            f"Giá trị grain_restore '{val}' không hợp lệ (chỉ chấp nhận auto, off, light hoặc heavy)."
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


def _validate_batch_size(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, bool):
        raise ValueError("batch_size phải là số nguyên từ 1 đến 32 (không nhận boolean).")
    if isinstance(val, int):
        bs = val
    elif isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        if not re.fullmatch(r"[+-]?\d+", s):
            raise ValueError("batch_size phải là số nguyên từ 1 đến 32.")
        bs = int(s)
    else:
        raise ValueError("batch_size phải là số nguyên từ 1 đến 32.")
    if bs < 1 or bs > 32:
        raise ValueError("batch_size phải nằm trong khoảng 1..32.")
    return bs


def _validate_output_dir(workspace_root: Path, val: Optional[str], field: str = "output_dir") -> Optional[str]:
    if val is None:
        return None
    _validate_no_nul(str(val), field)
    raw = strip_wrapping_quotes(str(val))
    if not raw:
        return None
    _validate_no_nul(raw, field)
    p = Path(os.path.expanduser(raw))
    if p.drive and not p.is_absolute():
        raise ValueError(f"Đường dẫn tương đối theo ổ đĩa tại '{field}' không hợp lệ.")
    try:
        target = (p if p.is_absolute() else workspace_root / p).resolve()
    except OSError:
        target = p if p.is_absolute() else (workspace_root / p)
    if (
        is_secret_filename(raw)
        or is_secret_filename(p.name)
        or is_secret_filename(target.name)
        or any(is_secret_filename(part) for part in p.parts)
        or any(is_secret_filename(part) for part in target.parts)
    ):
        raise PermissionError("Từ chối đường dẫn đích trỏ tới tệp cấu hình nhạy cảm.")
    validate_win32_path_hygiene(p, field=field)
    if field == "output_dir":
        if target.exists() and target.is_file():
            raise ValueError(
                f"Đường dẫn '{field}' phải là thư mục xuất, không được trỏ vào tệp tin có sẵn: '{target.name}'."
            )
        cur_parent = target.parent
        while cur_parent and cur_parent != cur_parent.parent:
            if cur_parent.exists() and cur_parent.is_file():
                raise ValueError(
                    f"Thành phần cha '{cur_parent.name}' là tệp tin, không thể tạo thư mục đích tại '{field}'."
                )
            cur_parent = cur_parent.parent
    return raw


def _resolve_output_target(workspace_root: Path, output_dir: Optional[str]) -> tuple[Path, Optional[Path]]:
    """
    Chuẩn hóa đường dẫn output (-o):
    - Đường dẫn tương đối được phân giải theo workspace_root (không dùng os.getcwd()).
    - Nếu trỏ thẳng tới tệp video (.mkv, .mp4, ...), tách thành (thư mục cha, tệp đích).
    """
    if not output_dir:
        return (workspace_root / "out").resolve(), None
    _validate_no_nul(str(output_dir), "output_dir")
    raw = strip_wrapping_quotes(str(output_dir))
    if not raw:
        return (workspace_root / "out").resolve(), None
    _validate_no_nul(raw, "output_dir")
    p = Path(os.path.expanduser(raw))
    if p.drive and not p.is_absolute():
        raise ValueError("Đường dẫn tương đối theo ổ đĩa tại 'output_dir' không hợp lệ.")
    validate_win32_path_hygiene(p, field="output_dir")
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
    expected_grain_restore: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_base, _ = _resolve_output_target(workspace_root, output_dir)
    vid_dir = Path(video_path).resolve().parent
    norm_grain = _validate_grain_restore(expected_grain_restore)
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
                    if norm_grain is not None:
                        c_params = data.get("dedup_params") or {}
                        c_grain = c_params.get("grain_restore")
                        if c_grain is not None and str(c_grain).strip().lower() != norm_grain:
                            continue
                        if c_grain is None and norm_grain != "auto":
                            continue
                    return data
            except Exception:
                pass
    return None


def _inspect_web_checkpoint_progress(
    workspace_root: Path,
    video_path: str,
    out_base: Path,
    plan: Optional[Dict[str, Any]],
    total_frames: int,
    n_unique: int,
) -> tuple[int, int, float, int]:
    """Return (done_chunks, total_chunks, remaining_fraction, remaining_n_unique) for an active .temp_* checkpoint."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    vid_dir = Path(video_path).resolve().parent
    cand_dirs = [
        out_base / f".temp_{base_name}",
        workspace_root / "out" / f".temp_{base_name}",
        vid_dir / f".temp_{base_name}",
    ]
    for w_dir in cand_dirs:
        up_dir = w_dir / "upscaled_segs"
        if not up_dir.is_dir():
            continue
        done_idxs: set[int] = set()
        try:
            for entry in os.scandir(up_dir):
                fn = entry.name
                if fn.startswith("up_") and fn.endswith(".mp4") and not fn.endswith(".tmp.mp4"):
                    if entry.stat().st_size > 0:
                        stem_num = fn[3:-4]
                        if stem_num.isdigit():
                            done_idxs.add(int(stem_num))
        except OSError:
            continue
        if not done_idxs:
            continue
        chunks_list = plan.get("chunks") if isinstance(plan, dict) else None
        if isinstance(chunks_list, list) and len(chunks_list) > 0:
            tot_c = len(chunks_list)
            rem_u = 0
            valid_done = 0
            for idx_c, ch in enumerate(chunks_list):
                c_id = int(ch.get("chunk_idx", idx_c)) if isinstance(ch, dict) else idx_c
                if c_id in done_idxs:
                    valid_done += 1
                elif isinstance(ch, dict):
                    rem_u += int(ch.get("kept_count", ch.get("total_frames", 0)))
            if valid_done > 0:
                rem_frac = max(0.0, float(tot_c - valid_done) / float(max(1, tot_c)))
                return valid_done, tot_c, rem_frac, max(0, rem_u)
        raw_dir = w_dir / "raw_segs"
        tot_c_est = 0
        if raw_dir.is_dir():
            try:
                tot_c_est = sum(
                    1
                    for entry in os.scandir(raw_dir)
                    if entry.name.startswith("seg_")
                    and entry.name.endswith(".mp4")
                    and not entry.name.endswith(".tmp.mp4")
                )
            except OSError:
                tot_c_est = 0
        seg_s_est = 24.0
        if tot_c_est <= 0:
            mf_path = w_dir / "manifest.json"
            if mf_path.is_file():
                try:
                    mdata = json.loads(mf_path.read_text(encoding="utf-8"))
                    if isinstance(mdata, dict):
                        tot_c_est = int(mdata.get("total_chunks") or 0)
                        if mdata.get("segment_time"):
                            seg_s_est = max(1.0, float(mdata["segment_time"]))
                except Exception:
                    tot_c_est = 0
        if tot_c_est <= 0:
            tot_c_est = max(len(done_idxs) + 1, int(round(max(1, total_frames) / (seg_s_est * 23.976))))
        valid_done = min(len(done_idxs), tot_c_est)
        rem_frac = max(0.0, float(tot_c_est - valid_done) / float(max(1, tot_c_est)))
        rem_u = max(0, int(round(float(n_unique) * rem_frac)))
        return valid_done, tot_c_est, rem_frac, rem_u
    return 0, 0, 1.0, n_unique


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
        self._aborting_tokens: set[str] = set()
        self._aborted_owned_pids_by_job: Dict[str, set[int]] = {}

    def get_state(self) -> Dict[str, Any]:
        """Trả về bản sao an toàn trạng thái tác vụ hiện tại & lịch sử gần đây."""
        with self._lock:
            if self._active_job and self._proc is not None:
                rc = self._proc.poll()
                if rc is not None and self._active_job["status"] in {"running", "pausing", "aborted"}:
                    self._finalize_locked(rc)
            active = dict(self._active_job) if self._active_job else None
            hist = [dict(x) for x in self._history]
            is_aborting = bool(self._aborting_tokens)
            aborted_pids = sorted({int(p) for pid_set in self._aborted_owned_pids_by_job.values() for p in pid_set})
        return {
            "status": active["status"] if active else "idle",
            "active_job": active,
            "history": hist,
            "aborting": is_aborting,
            "aborted_pids": aborted_pids,
        }

    def _finalize_locked(self, return_code: int) -> None:
        if not self._active_job or self._active_job.get("finished_at") is not None:
            return
        prev_status = self._active_job.get("status")
        job_type = str(self._active_job.get("job_type") or "")

        live_stage = ""
        live_file = self.workspace_root / "logs" / "live_status.json"
        if job_type in {"upscale", "preview"} and live_file.is_file():
            try:
                live_mtime = live_file.stat().st_mtime
                ldata = json.loads(live_file.read_text(encoding="utf-8"))
                if isinstance(ldata, dict) and live_matches_web_job(ldata, self._active_job, live_mtime):
                    live_stage = str(ldata.get("stage") or "")
            except Exception:
                live_stage = ""

        # Dọn cờ .graceful_pause.flag còn sót (ví dụ người dùng bấm dừng êm lúc đang Mux cuối tập)
        pause_flag = self.workspace_root / "logs" / ".graceful_pause.flag"
        if pause_flag.exists() and not self._has_live_external_cli_locked(self._active_job):
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
        self._active_job["finished_epoch"] = time.time()
        self._history.appendleft(dict(self._active_job))
        self._proc = None

    def compute_schedule_preview(
        self,
        inputs: Sequence[str],
        deadline_str: Optional[str] = None,
        allow_partial: bool = False,
        model_override: Optional[str] = None,
        dedup_override: Optional[str] = None,
        grain_restore: Optional[str] = None,
        zero_disk: bool = True,
        use_tensorrt: bool = True,
        subs_after_upscale: bool = True,
        output_dir: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Dự báo chi tiết danh sách tập, model phân giải từ shows.yaml và thời gian hoàn thành theo --deadline."""
        videos = self.fs_browser.expand_inputs(
            inputs,
            allow_subtitles=False,
            allow_upscaled_4k=False,
        )
        deadline_str = _validate_deadline(deadline_str)
        grain_norm = _validate_grain_restore(grain_restore)
        now = datetime.now()
        deadline_dt: Optional[datetime] = None
        if deadline_str:
            parts = deadline_str.split(":")
            dh, dm = int(parts[0]), int(parts[1])
            deadline_dt = now.replace(hour=dh, minute=dm, second=0, microsecond=0)
            if deadline_dt <= now:
                deadline_dt += timedelta(days=1)

        routing_tables = self.fs_browser._load_routing_tables()
        models_reg = routing_tables[0] if isinstance(routing_tables, tuple) and routing_tables else {}
        trt_cache_dir = self.workspace_root / ".cache" / "trt_engines"
        trt_valid_memo: Dict[str, List[str]] = {}

        def _get_valid_trt_engines(mid_check: str) -> List[str]:
            if mid_check in trt_valid_memo:
                return trt_valid_memo[mid_check]
            m_info = models_reg.get(mid_check) if isinstance(models_reg, dict) else None
            w_sha = str(m_info.get("sha256") or "") if isinstance(m_info, dict) else None
            arch_type = str(((m_info or {}).get("arch") or {}).get("type") or "") if isinstance(m_info, dict) else ""
            valid_engs, _ = inspect_model_trt_cache(trt_cache_dir, mid_check, expected_weights_sha256=w_sha)
            if arch_type == "UpCunet2x":
                valid_engs = [n for n in valid_engs if "_ps2x_" in n.lower()]
                if (grain_norm or "auto") == "off" and w_sha:
                    base_tag = f"_ps2x_{w_sha.strip().lower()[:8]}_"
                    valid_engs = [n for n in valid_engs if base_tag in n.lower()]
            trt_valid_memo[mid_check] = valid_engs
            return valid_engs

        def _is_model_trt_ready(mid_check: str, in_h: int = 0, in_w: int = 0) -> bool:
            if not use_tensorrt:
                return True
            if mid_check == AUTO_PARETO_MODEL_ID:
                sub_ids = [
                    k for k in ("animejanai-hd-v3-compact", "animejanai-hd-v3-ultracompact") if k in models_reg
                ]
                return bool(sub_ids) and all(_is_model_trt_ready(sid, in_h, in_w) for sid in sub_ids)
            valid_engs = _get_valid_trt_engines(mid_check)
            if not valid_engs:
                return False
            if in_h > 0 and in_w > 0:
                hw_tokens = (f"_{in_h}x{in_w}_", f"_{in_w}x{in_h}_")
                return any(any(tok in n.lower() for tok in hw_tokens) for n in valid_engs)
            return True

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
                expected_grain_restore=grain_norm or "auto",
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

            resumed_done_c = 0
            resumed_tot_c = 0
            ep_total_overhead = total_overhead
            if not already_done and not force:
                resumed_done_c, resumed_tot_c, rem_frac, rem_u = _inspect_web_checkpoint_progress(
                    self.workspace_root, vid, out_base, plan, total_f, n_unique
                )
                if resumed_done_c > 0:
                    n_unique = rem_u
                    ep_total_overhead = (vid_overhead * max(0.15, rem_frac)) + sub_overhead
                    source_note = f"{source_note} (còn {resumed_tot_c - resumed_done_c}/{resumed_tot_c} chunk)"

            if already_done:
                eta_med_s = 0.0
                dec_eta_s = 0.0
                can_run = False
                is_partial = False
                decision = "ĐÃ CÓ 4K (BỎ QUA)"
                ep_start = sim_med
                ep_finish = sim_med
            else:
                eta_med_s = (n_unique / max(0.5, base_med_fps)) + ep_total_overhead
                eta_p5_s = (n_unique / max(0.5, base_p5_fps)) + ep_total_overhead
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

            _, vid_h, vid_w = _probe_video_meta_lightweight(vid)
            episodes.append({
                "index": idx + 1,
                "file_path": vid,
                "file_name": os.path.basename(vid),
                "model_id": m_id,
                "model_name": routing["model_name"],
                "dedup_policy": routing["dedup_policy"],
                "routing_source": (
                    f"{routing['resolution_source']} [Tiếp tục checkpoint {resumed_done_c}/{resumed_tot_c} chunk]"
                    if resumed_done_c > 0
                    else routing["resolution_source"]
                ),
                "total_frames": total_f,
                "n_unique": n_unique,
                "dedup_ratio_percent": round(dedup_pct, 1),
                "n_unique_source": source_note,
                "ref_fps": round(base_med_fps, 1),
                "eta_seconds": round(eta_med_s, 1),
                "eta_minutes": round(eta_med_s / 60.0, 1),
                "start_time": ep_start.strftime("%H:%M"),
                "finish_time": ep_finish.strftime("%H:%M"),
                "can_run": can_run,
                "is_partial": is_partial,
                "already_done": already_done,
                "trt_ready": _is_model_trt_ready(m_id, vid_h, vid_w),
                "resumed_done_chunks": resumed_done_c,
                "resumed_total_chunks": resumed_tot_c,
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

        job_id = uuid.uuid4().hex[:10]
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if extra_env:
            env.update(extra_env)
        env["UPSCALER_WEB_JOB_ID"] = job_id

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
            # Finalize an exited prior job if the new launch races ahead of _drain_output().
            if self._active_job and self._proc is not None:
                prev_rc = self._proc.poll()
                if prev_rc is not None:
                    self._finalize_locked(int(prev_rc))
            if pause_flag.exists() and (
                requires_gpu or not self._has_live_external_cli_locked(self._active_job or {})
            ):
                try:
                    pause_flag.unlink()
                except OSError:
                    pass
            proc = self.popen_factory(cmd, **popen_kwargs)
            if proc is None:
                raise RuntimeError("Không thể khởi tạo tiến trình con (popen_factory trả về None).")
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
        out_dir_val = _validate_output_dir(self.workspace_root, payload.get("output_dir"))
        glossary_val = _validate_output_dir(self.workspace_root, payload.get("glossary"), field="glossary") or ""
        batch_size = _validate_batch_size(payload.get("batch_size"))

        subs_quality = _validate_subs_quality(payload.get("subs_quality"))
        sound_cues = _validate_sound_cues(payload.get("sound_cues"))
        deadline_norm = _validate_deadline(payload.get("deadline"))
        nvenc_mp_norm = _validate_nvenc_multipass(payload.get("nvenc_multipass"))
        grain_norm = _validate_grain_restore(payload.get("grain_restore"))

        videos = self.fs_browser.expand_inputs(
            raw_inputs,
            allow_subtitles=False,
            allow_upscaled_4k=False,
        )
        dry_run = bool(payload.get("dry_run", False))
        self._ensure_ready_to_launch(requires_gpu=not dry_run)

        upscale_script = str(self.workspace_root / "upscale.py")
        cmd = [self.python_executable, upscale_script]
        if len(videos) > 1:
            cmd.extend(["--queue", *videos])
        else:
            cmd.append(videos[0])

        if out_dir_val:
            cmd.extend(["-o", out_dir_val])
        if payload.get("model"):
            cmd.extend(["--model", str(payload["model"]).strip()])
        if payload.get("dedup_policy"):
            cmd.extend(["--dedup-policy", str(payload["dedup_policy"]).strip()])
        if nvenc_mp_norm:
            cmd.extend(["--nvenc-multipass", nvenc_mp_norm])
        if grain_norm:
            cmd.extend(["--grain-restore", grain_norm])
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])
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
        else:
            cmd.extend(["--subs-quality", subs_quality])
            if glossary_val:
                cmd.extend(["--glossary", glossary_val])
            if payload.get("subs_upgrade"):
                cmd.append("--subs-upgrade")
            if payload.get("no_embed"):
                cmd.append("--no-embed")
            if sound_cues == "translate":
                cmd.extend(["--sound-cues", "translate"])
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
        out_dir_val = _validate_output_dir(self.workspace_root, payload.get("output_dir"))
        batch_size = _validate_batch_size(payload.get("batch_size"))
        videos = self.fs_browser.expand_inputs(
            [video_raw],
            allow_subtitles=False,
            allow_upscaled_4k=False,
        )
        target_video = videos[0]
        raw_sec = payload.get("seconds", 10)
        sec_f = float(10 if raw_sec is None else raw_sec)
        if not math.isfinite(sec_f):
            raise ValueError("seconds phải là số hữu hạn.")
        seconds = max(3, min(60, int(sec_f)))
        nvenc_mp_norm = _validate_nvenc_multipass(payload.get("nvenc_multipass"))
        grain_norm = _validate_grain_restore(payload.get("grain_restore"))
        self._ensure_ready_to_launch(requires_gpu=True)

        preview_start_f = parse_preview_start_value(payload.get("preview_start_sec"))

        cmd = [
            self.python_executable,
            str(self.workspace_root / "upscale.py"),
            target_video,
            "--preview",
            str(seconds),
            "--no-subs",
        ]
        if preview_start_f is not None:
            cmd.extend(["--preview-start", str(preview_start_f)])
        if out_dir_val:
            cmd.extend(["-o", out_dir_val])
        if payload.get("model"):
            cmd.extend(["--model", str(payload["model"]).strip()])
        if payload.get("dedup_policy"):
            cmd.extend(["--dedup-policy", str(payload["dedup_policy"]).strip()])
        if nvenc_mp_norm:
            cmd.extend(["--nvenc-multipass", nvenc_mp_norm])
        if grain_norm:
            cmd.extend(["--grain-restore", grain_norm])
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])
        if payload.get("zero_disk") is False:
            cmd.append("--no-zero-disk")
        if payload.get("use_tensorrt") is False:
            cmd.append("--no-tensorrt")
        if payload.get("force", False):
            cmd.append("--force")

        extra_env = {}
        if not payload.get("auto_open_player", False):
            extra_env["UPSCALER_NO_AUTO_OPEN"] = "1"

        return self._spawn_command("preview", cmd, [target_video], payload, extra_env=extra_env, requires_gpu=True)

    def start_subtrans_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Khởi chạy dịch phụ đề trực tiếp (--subs-only hoặc subtrans.py) bình thường ở mọi khung giờ."""
        raw_input_val = str(payload.get("input_path") or "")
        _validate_no_nul(raw_input_val, "input_path")
        target_raw = strip_wrapping_quotes(raw_input_val)
        out_dir_val = _validate_output_dir(self.workspace_root, payload.get("output_dir"), field="output_dir")
        glossary_val = _validate_output_dir(self.workspace_root, payload.get("glossary"), field="glossary") or ""

        raw_inputs_list = payload.get("inputs")
        if isinstance(raw_inputs_list, list) and raw_inputs_list:
            for raw_el in raw_inputs_list:
                _validate_no_nul(str(raw_el or ""), "inputs")
            sub_raw_targets = [strip_wrapping_quotes(str(x)) for x in raw_inputs_list if str(x or "").strip()]
        else:
            sub_raw_targets = [target_raw] if target_raw else []
        if not sub_raw_targets:
            raise ValueError("Vui lòng chọn file video (.mkv/.mp4) hoặc phụ đề (.srt/.ass) để dịch.")
        for s_item in sub_raw_targets:
            _validate_no_nul(s_item, "input_path")

        subs_quality = _validate_subs_quality(payload.get("subs_quality"))
        items = self.fs_browser.expand_inputs(sub_raw_targets, allow_subtitles=True, allow_upscaled_4k=True)
        sub_exts = {".srt", ".ass", ".ssa", ".vtt"}
        has_sub_items = any(os.path.splitext(x)[1].lower() in sub_exts for x in items)
        all_subtitles = bool(items) and all(os.path.splitext(x)[1].lower() in sub_exts for x in items)
        if has_sub_items and not all_subtitles:
            raise ValueError(
                "Không thể trộn lẫn tệp video (.mkv/.mp4) và tệp phụ đề rời (.srt/.ass) trong cùng một tác vụ Subtrans."
            )

        resolved_first_in = Path(os.path.expanduser(sub_raw_targets[0]))
        if not resolved_first_in.is_absolute():
            resolved_first_in = (self.workspace_root / resolved_first_in).resolve()
        else:
            resolved_first_in = resolved_first_in.resolve()
        is_single_dir_input = len(sub_raw_targets) == 1 and resolved_first_in.is_dir()

        prefer_whisper = bool(payload.get("prefer_whisper", False))
        dry_run = bool(payload.get("dry_run", False))
        needs_gpu = bool(prefer_whisper and not dry_run and not all_subtitles)
        self._ensure_ready_to_launch(requires_gpu=needs_gpu)

        if all_subtitles:
            out_dir = out_dir_val or str(self.workspace_root / "out")
            if is_single_dir_input or len(items) > 1:
                batch_dir = str(resolved_first_in) if is_single_dir_input else os.path.dirname(os.path.abspath(items[0]))
                cmd = [
                    self.python_executable,
                    str(self.workspace_root / "subtrans.py"),
                    "batch",
                    batch_dir,
                    "-o",
                    out_dir,
                    "--subs-quality",
                    subs_quality,
                ]
                if not is_single_dir_input:
                    for sel_sub in items:
                        cmd.extend(["--file", sel_sub])
            else:
                target_file = items[0]
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
            out_fmt = str(payload.get("output_format") or "srt").strip().lower()
            if out_fmt == "ass":
                cmd.extend(["--format", "ass"])
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
            if out_dir_val:
                cmd.extend(["-o", out_dir_val])
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

    def start_build_trt_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compile and SHA-256-verify TensorRT 10 FP16 engine via src/build_trt_engine.py subprocess."""
        raw_model = str(payload.get("model_id") or "").strip()
        _validate_no_nul(raw_model, "model_id")
        if not raw_model:
            raise ValueError("Vui lòng chọn ID mô hình cần biên dịch TensorRT.")
        models_reg, _, _ = self.fs_browser._load_routing_tables()
        if raw_model.lower() != "all" and raw_model not in models_reg:
            raise ValueError(f"Mô hình '{raw_model}' không có trong danh mục models/models.yaml.")
        force = bool(payload.get("force", False))
        dry_run = bool(payload.get("dry_run", False))
        self._ensure_ready_to_launch(requires_gpu=not dry_run)
        cmd = [
            self.python_executable,
            str(self.workspace_root / "src" / "build_trt_engine.py"),
            "--model",
            raw_model,
        ]
        if force:
            cmd.append("--force")
        if dry_run:
            cmd.append("--dry-run")
        return self._spawn_command("build_trt", cmd, [raw_model], payload, requires_gpu=not dry_run)

    def start_clean_temp_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run global temporary artifact cleanup and audit folder organization via upscale.py --clean-temp."""
        out_dir_val = _validate_output_dir(self.workspace_root, payload.get("output_dir"), field="output_dir")
        dry_run = bool(payload.get("dry_run", False))
        self._ensure_ready_to_launch(requires_gpu=False)
        lock_state = check_active_gpu_lock(self.workspace_root)
        if lock_state.get("locked") and not dry_run:
            raise RuntimeError(
                f"GPU đang bận xử lý (PID={lock_state.get('pid')}); từ chối dọn tệp tạm để bảo toàn phiên đang chạy."
            )
        cmd = [
            self.python_executable,
            str(self.workspace_root / "upscale.py"),
            "--clean-temp",
        ]
        if out_dir_val:
            cmd.extend(["-o", out_dir_val])
        if dry_run:
            cmd.append("--dry-run")
        target_label = out_dir_val or "out/"
        return self._spawn_command("clean_temp", cmd, [target_label], payload, requires_gpu=False)

    @staticmethod
    def _job_abort_token(job: Optional[Dict[str, Any]], proc: Optional[Any] = None) -> Optional[str]:
        if job:
            jid = str(job.get("job_id") or "").strip()
            if jid:
                return jid
            pid = _coerce_pid(job.get("pid"))
            if pid is not None:
                return f"pid:{pid}"
        if proc is not None:
            return f"proc:{id(proc)}"
        return None

    def _record_aborted_owned_pids_locked(self, token: str, pids: set[int]) -> None:
        if not token or not pids:
            return
        self._aborted_owned_pids_by_job[token] = set(pids)
        while len(self._aborted_owned_pids_by_job) > 25:
            oldest_key = next(iter(self._aborted_owned_pids_by_job))
            self._aborted_owned_pids_by_job.pop(oldest_key, None)

    def _is_aborted_job_lineage_locked(
        self,
        candidate_pid: Optional[int],
        aborted_job: Dict[str, Any],
        live_data: Optional[Dict[str, Any]] = None,
        live_mtime: float = 0.0,
    ) -> bool:
        if candidate_pid is None:
            return False
        aborted_pid = _coerce_pid(aborted_job.get("pid"))
        if aborted_pid is not None and candidate_pid == aborted_pid:
            return True
        token = self._job_abort_token(aborted_job)
        owned_pids = self._aborted_owned_pids_by_job.get(token, set()) if token else set()
        if candidate_pid in owned_pids:
            return True
        if (
            isinstance(live_data, dict)
            and live_matches_web_job(live_data, aborted_job, live_mtime)
            and _coerce_pid(live_data.get("pid")) == candidate_pid
        ):
            return True
        try:
            import psutil
            proc_ppid = _coerce_pid(psutil.Process(candidate_pid).ppid())
            if proc_ppid is not None and (proc_ppid == aborted_pid or proc_ppid in owned_pids):
                return True
        except Exception:
            pass
        return False

    def _has_live_external_cli_locked(self, aborted_job: Dict[str, Any]) -> bool:
        """Return True when a live external CLI process is active after an aborted Web Job."""
        live_file = self.workspace_root / "logs" / "live_status.json"
        ldata: Optional[Dict[str, Any]] = None
        live_mtime = 0.0
        if live_file.is_file():
            try:
                live_mtime = live_file.stat().st_mtime
                parsed = json.loads(live_file.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    ldata = parsed
            except Exception:
                ldata = None

        lock_state = check_active_gpu_lock(self.workspace_root)
        lock_pid = _coerce_pid(lock_state.get("pid"))
        if (
            lock_state.get("locked")
            and lock_pid is not None
            and not self._is_aborted_job_lineage_locked(lock_pid, aborted_job, ldata, live_mtime)
        ):
            return True

        if isinstance(ldata, dict) and not live_matches_web_job(ldata, aborted_job, live_mtime):
            live_job_id = str(ldata.get("job_id") or "").strip()
            live_pid = _coerce_pid(ldata.get("pid"))
            live_stage = str(ldata.get("stage") or "").strip().lower()
            if (
                not live_job_id
                and live_pid is not None
                and not self._is_aborted_job_lineage_locked(live_pid, aborted_job, ldata, live_mtime)
                and live_stage not in {"", "idle", "completed", "paused", "error", "aborted"}
            ):
                import psutil
                if psutil.pid_exists(live_pid):
                    return True
        return False

    def request_graceful_pause(self) -> Dict[str, Any]:
        """Ghi cờ dừng êm logs/.graceful_pause.flag cho cả tiến trình Web lẫn CLI ngoài."""
        logs_dir = self.workspace_root / "logs"
        flag_path = logs_dir / ".graceful_pause.flag"
        with self._lock:
            active_token = self._job_abort_token(self._active_job, self._proc)
            if (
                self._active_job
                and self._active_job.get("status") == "aborted"
                and active_token is not None
                and active_token in self._aborting_tokens
            ):
                return {
                    "ok": False,
                    "graceful_pause_pending": flag_path.is_file(),
                    "status": "aborted",
                    "flag_path": str(flag_path),
                    "message": "Tác vụ đã bị buộc dừng; không thể đặt cờ dừng êm.",
                }
            if self._active_job and self._proc is not None:
                rc = self._proc.poll()
                if rc is not None and self._active_job.get("status") in {"running", "pausing", "aborted"}:
                    self._finalize_locked(int(rc))
            if self._active_job and self._active_job.get("status") == "aborted":
                is_finalized = (
                    self._proc is None and self._active_job.get("finished_at") is not None
                )
                if not is_finalized or not self._has_live_external_cli_locked(self._active_job):
                    return {
                        "ok": False,
                        "graceful_pause_pending": flag_path.is_file(),
                        "status": "aborted",
                        "flag_path": str(flag_path),
                        "message": "Tác vụ đã bị buộc dừng; không thể đặt cờ dừng êm.",
                    }
            logs_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = logs_dir / (
                f".graceful_pause.flag.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp"
            )
            data = {
                "requested_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "reason": "web_gui_graceful_pause",
            }
            try:
                tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp_path, flag_path)
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
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
        with self._lock:
            if self._active_job and self._proc is not None:
                rc = self._proc.poll()
                if rc is not None and self._active_job.get("status") in {"running", "pausing", "aborted"}:
                    self._finalize_locked(int(rc))
            try:
                flag_path.unlink(missing_ok=True)
            except OSError:
                pass
            if self._active_job and self._active_job.get("status") == "pausing":
                self._active_job["status"] = "running"
        return {"ok": True, "graceful_pause_pending": False}

    def abort_job(self) -> Dict[str, Any]:
        """Buộc dừng ngay tiến trình con do Web GUI khởi chạy và dọn khóa GPU mồ côi."""
        killed_pid: Optional[int] = None
        killed_proc: Optional[Any] = None
        killed_job_id: Optional[str] = None
        abort_token: Optional[str] = None
        owned_tree_pids: set[int] = set()
        aborted_queue_items: List[str] = []
        aborted_output_dir: Optional[str] = None
        external_lock_pid: Optional[int] = None
        concurrent_abort_pid: Optional[int] = None
        reclaimed_lineage_pid: Optional[int] = None
        try:
            with self._lock:
                proc = self._proc
                if proc is not None and proc.poll() is None:
                    killed_pid = getattr(proc, "pid", None)
                    killed_proc = proc
                    killed_job_id = str(self._active_job.get("job_id") or "") if self._active_job else None
                    if self._active_job and str(self._active_job.get("job_type") or "") == "upscale":
                        aborted_queue_items = [
                            str(x) for x in (self._active_job.get("queue_items") or []) if str(x or "").strip()
                        ]
                        raw_opts = self._active_job.get("options")
                        if isinstance(raw_opts, dict) and raw_opts.get("output_dir"):
                            aborted_output_dir = str(raw_opts.get("output_dir"))
                    abort_token = self._job_abort_token(self._active_job, killed_proc)
                    if abort_token is not None:
                        self._aborting_tokens.add(abort_token)
                    killed_pid_int = _coerce_pid(killed_pid)
                    if killed_pid_int is not None:
                        owned_tree_pids.add(killed_pid_int)
                        try:
                            import psutil
                            parent_proc = psutil.Process(killed_pid_int)
                            for child in parent_proc.children(recursive=True):
                                c_pid = _coerce_pid(getattr(child, "pid", None))
                                if c_pid is not None:
                                    owned_tree_pids.add(c_pid)
                        except Exception:
                            pass
                    live_file = self.workspace_root / "logs" / "live_status.json"
                    if live_file.is_file() and self._active_job is not None:
                        try:
                            live_mtime = live_file.stat().st_mtime
                            ldata = json.loads(live_file.read_text(encoding="utf-8"))
                            if isinstance(ldata, dict) and live_matches_web_job(ldata, self._active_job, live_mtime):
                                l_pid = _coerce_pid(ldata.get("pid"))
                                if l_pid is not None:
                                    owned_tree_pids.add(l_pid)
                        except Exception:
                            pass
                    if abort_token is not None and owned_tree_pids:
                        self._record_aborted_owned_pids_locked(abort_token, owned_tree_pids)
                    if self._active_job:
                        self._active_job["status"] = "aborted"
                    pause_flag = self.workspace_root / "logs" / ".graceful_pause.flag"
                    if pause_flag.exists() and (
                        not self._active_job or not self._has_live_external_cli_locked(self._active_job)
                    ):
                        try:
                            pause_flag.unlink(missing_ok=True)
                        except OSError:
                            pass
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
                elif self._aborting_tokens:
                    concurrent_abort_pid = _coerce_pid(
                        getattr(proc, "pid", None)
                        or (self._active_job or {}).get("pid")
                        or (self._history[0].get("pid") if self._history else None)
                    )

            if concurrent_abort_pid is not None and killed_proc is None:
                return {
                    "ok": True,
                    "aborted_pid": concurrent_abort_pid,
                    "status": "aborted",
                    "message": f"Đang hoàn tất dừng tiến trình Web Job (PID={concurrent_abort_pid}).",
                }

            if killed_proc is not None:
                try:
                    killed_proc.wait(timeout=2.0)
                except TypeError:
                    if killed_proc.poll() is not None:
                        try:
                            killed_proc.wait()
                        except Exception:
                            pass
                except Exception:
                    pass

            if owned_tree_pids:
                try:
                    import psutil
                    deadline = time.monotonic() + 0.4
                    while time.monotonic() < deadline:
                        if not any(psutil.pid_exists(p) for p in owned_tree_pids):
                            break
                        time.sleep(0.05)
                except Exception:
                    pass

            # Reclaim .gpu_active.lock only if owned by the killed tree or a dead PID; preserve external CLI owners.
            lock_file = self.workspace_root / ".gpu_active.lock"
            if lock_file.is_file():
                lock_state = check_active_gpu_lock(self.workspace_root)
                lock_pid = _coerce_pid(lock_state.get("pid"))
                is_owned_lock = False
                if lock_pid is not None:
                    if killed_pid is not None:
                        if lock_pid in owned_tree_pids:
                            is_owned_lock = True
                        else:
                            try:
                                import psutil
                                proc_ppid = _coerce_pid(psutil.Process(lock_pid).ppid())
                                if proc_ppid is not None and proc_ppid in owned_tree_pids:
                                    owned_tree_pids.add(lock_pid)
                                    is_owned_lock = True
                            except Exception:
                                pass
                    if not is_owned_lock:
                        with self._lock:
                            cand_aborted_jobs = [
                                j for j in ([self._active_job] if self._active_job else []) + list(self._history)
                                if isinstance(j, dict) and j.get("status") == "aborted"
                            ]
                            for ab_job in cand_aborted_jobs:
                                if self._is_aborted_job_lineage_locked(lock_pid, ab_job):
                                    is_owned_lock = True
                                    reclaimed_lineage_pid = lock_pid
                                    break
                if not lock_state.get("locked") or is_owned_lock:
                    try:
                        lock_file.unlink(missing_ok=True)
                    except OSError:
                        pass
                else:
                    external_lock_pid = lock_state.get("pid")

            # Clean unclaimed 0-chunk preflight .temp_* directories left behind after aborting a multi-item queue.
            if killed_proc is not None and len(aborted_queue_items) > 1 and external_lock_pid is None:
                import shutil
                try:
                    out_base_clean, _ = _resolve_output_target(self.workspace_root, aborted_output_dir)
                except Exception:
                    out_base_clean = (self.workspace_root / "out").resolve()
                for q_vid in aborted_queue_items:
                    q_stem = os.path.splitext(os.path.basename(q_vid))[0]
                    if not q_stem:
                        continue
                    temp_cand = out_base_clean / f".temp_{q_stem}"
                    if not temp_cand.exists() or temp_cand.is_symlink() or os.path.islink(str(temp_cand)):
                        continue
                    try:
                        st_cand = temp_cand.lstat()
                        if bool(getattr(st_cand, "st_file_attributes", 0) & 0x0400) or not temp_cand.is_dir():
                            continue
                        up_dir_cand = temp_cand / "upscaled_segs"
                        has_done_up = False
                        if up_dir_cand.is_dir():
                            for up_entry in os.scandir(up_dir_cand):
                                fn = up_entry.name
                                if fn.startswith("up_") and fn.endswith(".mp4") and not fn.endswith(".tmp.mp4"):
                                    try:
                                        if up_entry.is_file() and up_entry.stat().st_size > 0:
                                            has_done_up = True
                                            break
                                    except OSError:
                                        pass
                        if not has_done_up:
                            shutil.rmtree(temp_cand, ignore_errors=True)
                    except OSError:
                        pass
        finally:
            if killed_proc is not None or abort_token is not None:
                with self._lock:
                    try:
                        if abort_token is not None and owned_tree_pids:
                            self._record_aborted_owned_pids_locked(abort_token, owned_tree_pids)
                    finally:
                        if abort_token is not None:
                            self._aborting_tokens.discard(abort_token)
                    # Finalize deterministically once killed_proc has exited instead of waiting for _drain_output().
                    if (
                        killed_proc is not None
                        and self._active_job
                        and self._proc is killed_proc
                        and (not killed_job_id or self._active_job.get("job_id") == killed_job_id)
                    ):
                        rc = killed_proc.poll()
                        if rc is not None:
                            self._finalize_locked(int(rc))
                    if killed_proc is not None:
                        same_aborted_job = self._active_job is None or (
                            (
                                (bool(killed_job_id) and self._active_job.get("job_id") == killed_job_id)
                                or (
                                    not killed_job_id
                                    and not self._active_job.get("job_id")
                                    and (self._proc is None or self._proc is killed_proc)
                                )
                            )
                            and self._active_job.get("status") == "aborted"
                        )
                        if same_aborted_job:
                            pause_flag = self.workspace_root / "logs" / ".graceful_pause.flag"
                            if pause_flag.exists() and (
                                not self._active_job or not self._has_live_external_cli_locked(self._active_job)
                            ):
                                try:
                                    pause_flag.unlink(missing_ok=True)
                                except OSError:
                                    pass

        if killed_pid is not None:
            return {
                "ok": True,
                "aborted_pid": killed_pid,
                "status": "aborted",
                "message": f"Đã dừng tiến trình Web Job (PID={killed_pid}).",
            }
        if reclaimed_lineage_pid is not None:
            return {
                "ok": True,
                "aborted_pid": reclaimed_lineage_pid,
                "status": "aborted",
                "message": f"Đã thu hồi khóa GPU của tiến trình Web Job vừa dừng (PID={reclaimed_lineage_pid}).",
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
