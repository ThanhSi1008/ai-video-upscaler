# -*- coding: utf-8 -*-
"""
web/job_manager.py
Bộ điều phối tiến trình Upscale, Preview, Hàng đợi ca đêm và Dịch phụ đề Subtrans:
- Cho phép vận hành bình thường mọi tác vụ (Upscale, Preview, Subtrans, Subs-Pending) vào bất kỳ thời điểm nào.
- Chống xung đột 2 tiến trình GPU qua khóa .gpu_active.lock.
- Hỗ trợ Dừng êm sau Chunk hiện tại (Graceful Pause) và Hủy khẩn cấp (Force Stop).
- Tính toán trước bảng ETA ca đêm theo từng model SR từ shows.yaml & plan.json.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

from web.fs_browser import SafeLocalFSBrowser
from web.sse_hub import check_active_gpu_lock

# Tốc độ suy luận khung hình độc nhất tham chiếu trên RTX 3060 TensorRT 10 FP16 Zero-Disk (từ STATE.md R3)
MODEL_REFERENCE_FPS: Dict[str, tuple[float, float]] = {
    "animevideov3": (10.5, 8.5),
    "animejanai-hd-v3-compact": (10.8, 8.8),
    "animejanai-hd-v3-ultracompact": (18.5, 15.5),
    "animejanai-hd-v3-sharp1-ultracompact": (18.5, 15.5),
    "animejanai-hd-v3-superultracompact": (20.8, 17.5),
    "animejanai-hd-v3-sharp1-superultracompact": (20.8, 17.5),
    "span-2x": (14.5, 12.0),
    "2x_AnimeJaNai_V3": (14.5, 12.0),
}


def _probe_frames_lightweight(video_path: str) -> int:
    """Đo nhanh tổng số frame video bằng PyAV in-process (fallback ước lượng theo dung lượng/mặc định)."""
    try:
        import av
        with av.open(video_path) as container:
            if container.streams.video:
                vs = container.streams.video[0]
                if vs.frames and vs.frames > 0:
                    return int(vs.frames)
                dur = float(container.duration / av.time_base) if container.duration else 0.0
                fps = float(vs.average_rate) if vs.average_rate else 23.976
                if dur > 0:
                    return max(1, int(round(dur * fps)))
    except Exception:
        pass
    return 34000


def _find_cached_plan(workspace_root: Path, video_path: str, output_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_base = Path(output_dir).resolve() if output_dir else (workspace_root / "out")
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
        if not self._active_job:
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
    ) -> Dict[str, Any]:
        """Dự báo chi tiết danh sách tập, model phân giải từ shows.yaml và thời gian hoàn thành theo --deadline."""
        videos = self.fs_browser.expand_inputs(inputs)
        now = datetime.now()
        deadline_dt: Optional[datetime] = None
        if deadline_str and deadline_str.strip():
            parts = deadline_str.strip().split(":")
            if len(parts) != 2:
                raise ValueError(f"Định dạng deadline '{deadline_str}' không hợp lệ (cần HH:MM).")
            dh, dm = int(parts[0]), int(parts[1])
            if not (0 <= dh <= 23 and 0 <= dm <= 59):
                raise ValueError(f"Giờ hạn chót '{deadline_str}' ngoài phạm vi 00:00..23:59.")
            deadline_dt = now.replace(hour=dh, minute=dm, second=0, microsecond=0)
            if deadline_dt <= now:
                deadline_dt += timedelta(days=1)

        out_base = Path(output_dir).resolve() if output_dir else (self.workspace_root / "out")
        sub_overhead = 300.0 if subs_after_upscale else 0.0
        vid_overhead = 420.0 if (zero_disk and use_tensorrt) else 600.0
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
            )
            m_id = routing["model_id"]
            base_med_fps, base_p5_fps = MODEL_REFERENCE_FPS.get(m_id, (10.5, 8.5))
            if not use_tensorrt:
                base_med_fps *= 0.52
                base_p5_fps *= 0.52
            elif not zero_disk:
                base_med_fps *= 0.82
                base_p5_fps *= 0.82

            plan = _find_cached_plan(self.workspace_root, vid, str(out_base))
            if plan:
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

            out_suffix = "_4K_animejanai.mkv" if m_id.startswith("animejanai-") else "_4K.mkv"
            out_4k = out_base / f"{base_name}{out_suffix}"
            already_done = out_4k.is_file() and out_4k.stat().st_size > 1024 * 1024

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
                    prior_done = sum(
                        1 for e in episodes if e["can_run"] and not e["is_partial"] and not e["already_done"]
                    )
                    eff_deadline = deadline_dt - timedelta(seconds=prior_done * sub_overhead)
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

    def _ensure_ready_to_launch(self, requires_gpu: bool = True) -> None:
        with self._lock:
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

    def _spawn_command(
        self,
        job_type: str,
        cmd: List[str],
        queue_items: List[str],
        options: Dict[str, Any],
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        logs_dir = self.workspace_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        pause_flag = logs_dir / ".graceful_pause.flag"
        if pause_flag.exists():
            try:
                pause_flag.unlink()
            except OSError:
                pass

        web_log_path = logs_dir / "web_job.log"
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

        proc = self.popen_factory(cmd, **popen_kwargs)
        job_id = uuid.uuid4().hex[:10]
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        job_record: Dict[str, Any] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "running",
            "pid": getattr(proc, "pid", None),
            "started_at": now_iso,
            "finished_at": None,
            "return_code": None,
            "command": cmd,
            "queue_items": queue_items,
            "options": options,
        }

        with self._lock:
            self._proc = proc
            self._active_job = job_record

        def _drain_output() -> None:
            try:
                with web_log_path.open("w", encoding="utf-8") as wf:
                    wf.write(f"# Web Studio Job {job_id} ({job_type}) started at {now_iso}\n")
                    wf.flush()
                    stdout_pipe = getattr(proc, "stdout", None)
                    if stdout_pipe is not None:
                        for raw_line in stdout_pipe:
                            wf.write(raw_line)
                            wf.flush()
                rc = proc.wait()
            except Exception:
                rc = proc.poll() if proc.poll() is not None else -1
            with self._lock:
                if self._active_job and self._active_job.get("job_id") == job_id:
                    self._finalize_locked(int(rc))

        t = threading.Thread(target=_drain_output, name=f"WebJobDrain-{job_id}", daemon=True)
        t.start()
        return dict(job_record)

    def start_upscale_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Khởi chạy hàng đợi hoặc tập đơn lẻ qua upscale.py."""
        raw_inputs = payload.get("inputs") or []
        if not raw_inputs:
            raise ValueError("Vui lòng chọn ít nhất một file video hoặc thư mục nguồn.")
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
        if payload.get("batch_size"):
            cmd.extend(["--batch-size", str(int(payload["batch_size"]))])
        if payload.get("deadline"):
            cmd.extend(["--deadline", str(payload["deadline"]).strip()])
        if payload.get("allow_partial"):
            cmd.append("--allow-partial")
        if payload.get("zero_disk") is False:
            cmd.append("--no-zero-disk")
        if payload.get("use_tensorrt") is False:
            cmd.append("--no-tensorrt")
        if payload.get("subs_after_upscale") is False:
            cmd.append("--no-subs")
        if payload.get("subs_quality") in {"normal", "fast", "high"}:
            cmd.extend(["--subs-quality", str(payload["subs_quality"])])
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

        return self._spawn_command("upscale", cmd, videos, payload)

    def start_preview_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Khởi chạy xem trước nhanh (--preview N giây ở 30% tập phim)."""
        video_raw = str(payload.get("video_path") or "").strip()
        if not video_raw:
            raise ValueError("Vui lòng chọn một file video để trích xuất Preview.")
        videos = self.fs_browser.expand_inputs([video_raw])
        target_video = videos[0]
        seconds = max(3, min(60, int(payload.get("seconds") or 10)))
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
        if payload.get("zero_disk") is False:
            cmd.append("--no-zero-disk")
        if payload.get("use_tensorrt") is False:
            cmd.append("--no-tensorrt")
        if payload.get("force", True):
            cmd.append("--force")

        extra_env = {}
        if not payload.get("auto_open_player", False):
            extra_env["UPSCALER_NO_AUTO_OPEN"] = "1"

        return self._spawn_command("preview", cmd, [target_video], payload, extra_env=extra_env)

    def start_subtrans_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Khởi chạy dịch phụ đề trực tiếp (--subs-only hoặc subtrans.py) bình thường ở mọi khung giờ."""
        target_raw = str(payload.get("input_path") or "").strip()
        if not target_raw:
            raise ValueError("Vui lòng chọn file video (.mkv/.mp4) hoặc phụ đề (.srt/.ass) để dịch.")
        items = self.fs_browser.expand_inputs([target_raw])
        target_file = items[0]
        prefer_whisper = bool(payload.get("prefer_whisper", False))
        dry_run = bool(payload.get("dry_run", False))
        self._ensure_ready_to_launch(requires_gpu=(prefer_whisper and not dry_run))

        ext = os.path.splitext(target_file)[1].lower()
        if ext in {".srt", ".ass", ".ssa", ".vtt"}:
            out_dir = str(payload.get("output_dir") or (self.workspace_root / "out"))
            cmd = [
                self.python_executable,
                str(self.workspace_root / "subtrans.py"),
                "run",
                target_file,
                "-o",
                out_dir,
                "--subs-quality",
                str(payload.get("subs_quality") or "normal"),
                "--yes",
            ]
            if payload.get("glossary"):
                cmd.extend(["--glossary", str(payload["glossary"]).strip()])
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
                target_file,
                "--subs-quality",
                str(payload.get("subs_quality") or "normal"),
            ]
            if payload.get("output_dir"):
                cmd.extend(["-o", str(payload["output_dir"]).strip()])
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

        return self._spawn_command("subtrans", cmd, [target_file], payload)

    def start_subs_pending_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Chạy lại danh sách phụ đề tồn đọng trong pending_subs.json."""
        dry_run = bool(payload.get("dry_run", False))
        self._ensure_ready_to_launch(requires_gpu=False)
        cmd = [
            self.python_executable,
            str(self.workspace_root / "upscale.py"),
            "--subs-pending",
            "--subs-quality",
            str(payload.get("subs_quality") or "normal"),
        ]
        if dry_run:
            cmd.append("--dry-run")
        return self._spawn_command("subs_pending", cmd, ["pending_subs.json"], payload)

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

        # Thu hồi khóa .gpu_active.lock nếu thuộc PID vừa kết thúc
        lock_file = self.workspace_root / ".gpu_active.lock"
        if lock_file.is_file():
            lock_state = check_active_gpu_lock(self.workspace_root)
            if not lock_state.get("locked"):
                try:
                    lock_file.unlink()
                except OSError:
                    pass

        return {"ok": True, "aborted_pid": killed_pid}
