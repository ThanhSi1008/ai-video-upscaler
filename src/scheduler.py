"""Overnight batch queue scheduler with background preflight and unique-FPS ETA modeling."""

import os
import sys
import json
import time
import shutil
import threading
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from src.ffmpeg_ops import probe_video
from src.dedup import scan_dedup_plan, load_or_create_plan, resolve_dedup_policy
from src.config import DEVICE_NAME
from src.models_resolver import find_existing_4k_output_path


def find_cached_plan(
    video_path: str,
    output_dir: Optional[str] = None,
    expected_policy: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Tìm file plan.json đã quét trước đó cho video trong các đường dẫn tiêu chuẩn.
    Trả về nội dung dictionary nếu hợp lệ, None nếu chưa có.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = output_dir or "out"
    if str(out_dir).lower().endswith((".mkv", ".mp4")):
        out_dir = os.path.dirname(out_dir) or "out"
    vid_dir = os.path.dirname(os.path.abspath(video_path))

    candidate_paths = [
        os.path.join(out_dir, f".temp_{base_name}", "plan.json"),
        os.path.join("out", f".temp_{base_name}", "plan.json"),
        os.path.join(vid_dir, f".temp_{base_name}", "plan.json"),
        os.path.join(out_dir, f"{base_name}.plan.json"),
        os.path.join("out", f"{base_name}.plan.json"),
        os.path.splitext(video_path)[0] + ".plan.json",
        os.path.join(out_dir, "full_ep_plan.json"),
        os.path.join("out", "full_ep_plan.json"),
    ]

    for p in candidate_paths:
        if os.path.exists(p) and os.path.getsize(p) > 100:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("total_kept_frames") and len(data.get("chunks", [])) > 0:
                    plan_pol = data.get("dedup_policy")
                    if expected_policy and plan_pol is not None:
                        if str(plan_pol).strip().lower() != str(expected_policy).strip().lower():
                            continue
                    # Nếu là file generic full_ep_plan.json thì phải kiểm tra đúng source_file
                    if "full_ep_plan.json" in p:
                        if data.get("source_file") == os.path.basename(video_path):
                            return data
                    else:
                        return data
            except Exception:
                pass

    return None


def preflight_video_plan(
    video_path: str,
    output_dir: Optional[str] = None,
    segment_seconds: int = 8,
    progress_callback=None,
    dedup_policy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quét hoặc nạp plan.json cho một video cụ thể.
    """
    eff_policy = resolve_dedup_policy(video_path, dedup_policy)
    cached = find_cached_plan(video_path, output_dir, expected_policy=eff_policy)
    if cached:
        return cached

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = output_dir or "out"
    if str(out_dir).lower().endswith((".mkv", ".mp4")):
        out_dir = os.path.dirname(out_dir) or "out"
    work_dir = os.path.join(out_dir, f".temp_{base_name}")
    os.makedirs(work_dir, exist_ok=True)
    plan_path = os.path.join(work_dir, "plan.json")

    meta = probe_video(video_path)
    fps_num = meta.get("fps_num", 24000)
    fps_den = meta.get("fps_den", 1001)

    if progress_callback:
        progress_callback(f"Đang quét mpdecimate preflight cho: {os.path.basename(video_path)}...")

    plan_data = scan_dedup_plan(
        input_path=video_path,
        plan_path=plan_path,
        segment_seconds=segment_seconds,
        fps_num=fps_num,
        fps_den=fps_den,
        policy=eff_policy,
    )
    return plan_data


class BackgroundQueuePreflightManager:
    """
    Quản lý luồng quét preflight nền cho toàn bộ các tập tiếp theo trong hàng đợi.
    Chạy với nice -n 10 (độ ưu tiên CPU thấp) song song trong lúc GPU đang xử lý tập đầu.
    """

    def __init__(
        self,
        queue: List[str],
        output_dir: Optional[str] = None,
        segment_seconds: int = 8,
        skip_first: bool = True,
        logger=None,
        benchmark_mode: bool = False,
        on_plan_ready=None,
        dedup_policy: Optional[str] = None,
    ):
        self.queue = queue
        out_clean = output_dir or "out"
        if out_clean.lower().endswith((".mkv", ".mp4")):
            out_clean = os.path.dirname(out_clean) or "out"
        self.output_dir = out_clean
        self.segment_seconds = segment_seconds
        self.skip_first = skip_first
        self.logger = logger
        self.benchmark_mode = benchmark_mode or (os.environ.get("BENCHMARK_MODE") == "1")
        self.on_plan_ready = on_plan_ready
        self.dedup_policy = dedup_policy
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._video_events: Dict[str, threading.Event] = {}
        self._claimed_by_main: set[str] = set()
        self._done_videos: set[str] = set()
        self._current_video: Optional[str] = None
        self._current_proc: Optional[subprocess.Popen] = None
        self.results: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _norm_key(video_path: str) -> str:
        return os.path.normcase(os.path.abspath(os.path.expanduser(str(video_path))))

    def mark_main_started(self, video_path: str) -> None:
        """Đánh dấu luồng chính đã tiếp nhận tập video để luồng nền bỏ qua nếu chưa bắt đầu."""
        vkey = self._norm_key(video_path)
        with self._lock:
            self._claimed_by_main.add(vkey)

    def wait_for_video(self, video_path: str, timeout: float = 600.0) -> bool:
        """
        Đợi luồng preflight nền hoàn tất nếu nó đang xử lý đúng tập `video_path`;
        ngược lại đánh dấu claimed và trả về ngay để luồng chính không tranh chấp `.temp_*`.
        """
        vkey = self._norm_key(video_path)
        with self._lock:
            self._claimed_by_main.add(vkey)
            if (
                vkey in self._done_videos
                or self.thread is None
                or not self.thread.is_alive()
                or self._current_video != vkey
            ):
                return True
            evt = self._video_events.setdefault(vkey, threading.Event())

        finished = evt.wait(timeout=timeout)
        if not finished:
            with self._lock:
                proc = self._current_proc if self._current_video == vkey else None
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
        return finished

    def start(self):
        """Khởi động luồng preflight nền (từ chối chạy nếu đang ở benchmark mode)"""
        if self.benchmark_mode or os.environ.get("BENCHMARK_MODE") == "1":
            msg = "⚡ [BENCHMARK MODE] Từ chối khởi động BackgroundQueuePreflightManager để giữ môi trường đo lường thuần túy."
            if self.logger:
                self.logger.warning(msg, to_console=True)
            else:
                print(msg)
            return

        items_to_scan = self.queue[1:] if (self.skip_first and len(self.queue) > 1) else self.queue
        if not items_to_scan:
            return

        with self._lock:
            for item_vid in items_to_scan:
                vkey = self._norm_key(item_vid)
                if vkey not in self._video_events:
                    self._video_events[vkey] = threading.Event()

        self.thread = threading.Thread(
            target=self._worker,
            args=(items_to_scan,),
            daemon=True,
            name="QueuePreflightWorker"
        )
        self.thread.start()

    def _worker(self, items: List[str]):
        try:
            for vid in items:
                if self._stop_event.is_set():
                    break

                vkey = self._norm_key(vid)
                with self._lock:
                    if vkey in self._claimed_by_main:
                        self._done_videos.add(vkey)
                        evt_skip = self._video_events.get(vkey)
                        if evt_skip is not None:
                            evt_skip.set()
                        continue
                    self._current_video = vkey
                    evt = self._video_events.setdefault(vkey, threading.Event())

                try:
                    if not os.path.isfile(vid):
                        continue
                    eff_pol = resolve_dedup_policy(vid, self.dedup_policy)
                    cached = find_cached_plan(vid, self.output_dir, expected_policy=eff_pol)
                    if cached:
                        self.results[vid] = cached
                        continue

                    prio_desc = "BELOW_NORMAL" if sys.platform == "win32" else "nice -n 10"
                    msg = f"⚡ [Background Preflight] Đang quét nền ({prio_desc}) cho: {os.path.basename(vid)}"
                    if self.logger:
                        self.logger.info(msg, to_console=True)
                    else:
                        print(msg)

                    base_name = os.path.splitext(os.path.basename(vid))[0]
                    work_dir = os.path.join(self.output_dir, f".temp_{base_name}")
                    os.makedirs(work_dir, exist_ok=True)
                    plan_file = os.path.join(work_dir, "plan.json")

                    from src.dedup import build_preflight_subprocess_spec
                    cmd, popen_kwargs = build_preflight_subprocess_spec(
                        vid, plan_file, self.segment_seconds, dedup_policy=self.dedup_policy
                    )
                    proc = subprocess.Popen(cmd, **popen_kwargs)
                    with self._lock:
                        self._current_proc = proc

                    while proc.poll() is None:
                        if self._stop_event.is_set():
                            try:
                                proc.terminate()
                                proc.wait(timeout=2.0)
                            except Exception:
                                try:
                                    proc.kill()
                                    proc.wait(timeout=1.0)
                                except Exception:
                                    pass
                            break
                        time.sleep(0.2)

                    if (not self._stop_event.is_set()) and proc.returncode == 0 and os.path.exists(plan_file):
                        with open(plan_file, "r", encoding="utf-8") as f:
                            plan_data = json.load(f)
                        self.results[vid] = plan_data
                        n_k = plan_data.get("total_kept_frames", 0)
                        dedup_p = plan_data.get("dedup_ratio_percent", 0.0)
                        msg_done = (
                            f"📊 [Background Preflight] Đã quét xong '{os.path.basename(vid)}': "
                            f"n_unique đo được = {n_k} frames (Dedup: {dedup_p:.1f}%)."
                        )
                        if self.logger:
                            self.logger.info(msg_done, to_console=True)
                        else:
                            print(f"\n{msg_done}")

                        if self.on_plan_ready:
                            try:
                                self.on_plan_ready(vid, plan_data)
                            except Exception:
                                pass
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Lỗi khi quét preflight nền cho {os.path.basename(vid)}: {e}")
                finally:
                    with self._lock:
                        self._current_proc = None
                        self._current_video = None
                        self._done_videos.add(vkey)
                        evt.set()
        finally:
            with self._lock:
                for evt_rem in self._video_events.values():
                    evt_rem.set()

    def stop(self):
        self._stop_event.set()
        with self._lock:
            proc = self._current_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)


def compute_overnight_schedule(
    queue: List[str],
    deadline_str: Optional[str] = None,
    fps_median: float = 10.5,
    fps_p5: float = 8.5,
    video_overhead_seconds: float = 420.0,
    subtitles_overhead_seconds: float = 300.0,
    eta_margin_pct: float = 0.07,
    allow_partial: bool = False,
    output_dir: Optional[str] = None,
    segment_seconds: int = 8,
    preflight_now: bool = False,
    start_time: Optional[datetime] = None,
    subs_quality: str = "normal",
    dedup_policy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tính toán lịch trình chi tiết và đưa ra quyết định cho từng tập trong hàng đợi.

    Quy chuẩn:
      - Tốc độ AI: Tốc độ AI suy luận khung hình độc lập (Unique Inference FPS) trung vị 10.5 fps, P5 8.5 fps.
      - Overhead: video ~420s (TensorRT FP16 Zero-Disk) + subtitles ~300s (Stage A + B2 100-cues + audit) = 720s (~12.0m).
      - Quyết định deadline: dựa trên ETA P5 + biên an toàn 7% ETA (decision_eta = eta_p5 * 1.07).
      - Nếu không kịp và có --allow-partial: cho phép chạy đến hạn chót, dừng sạch ở ranh giới chunk.
    """
    now = start_time or datetime.now()
    deadline_dt: Optional[datetime] = None

    if deadline_str:
        try:
            dh, dm = map(int, deadline_str.split(":"))
            deadline_dt = now.replace(hour=dh, minute=dm, second=0, microsecond=0)
            if deadline_dt <= now:
                deadline_dt += timedelta(days=1)
        except Exception as e:
            print(f"⚠️ Không parse được deadline '{deadline_str}' (định dạng HH:MM): {e}")
            deadline_dt = None

    total_overhead = float(video_overhead_seconds) + float(subtitles_overhead_seconds)
    fps_med = max(0.5, float(fps_median))
    fps_conservative = max(0.5, float(fps_p5))

    episodes_meta = []
    sim_time_median = now
    sim_time_conservative = now
    total_scheduled_seconds = 0.0

    for idx, vid in enumerate(queue):
        base_name = os.path.splitext(os.path.basename(vid))[0]

        # 1. Tìm hoặc quét plan.json
        plan = None
        if preflight_now:
            plan = preflight_video_plan(vid, output_dir, segment_seconds, dedup_policy=dedup_policy)
        else:
            eff_pol = resolve_dedup_policy(vid, dedup_policy) if dedup_policy else None
            plan = find_cached_plan(vid, output_dir, expected_policy=eff_pol)

        is_exact = plan is not None
        if plan:
            total_frames = plan.get("total_frames", 0)
            n_unique = plan.get("total_kept_frames", 0)
            dedup_ratio = plan.get("dedup_ratio_percent", 0.0)
            n_unique_source = "đo (plan.json)"
        else:
            # Dự toán nếu chưa quét: dùng 60.0% kept frames làm giá trị thận trọng (thay cho 39.1% cũ, đo n=4: 39.1%, 46.0%, 54.4%, 61.3%)
            try:
                meta = probe_video(vid)
                total_frames = meta.get("nb_frames") or int(meta.get("duration", 1440) * 23.976)
                n_unique = int(total_frames * 0.60)
                dedup_ratio = (1.0 - (n_unique / total_frames)) * 100.0 if total_frames > 0 else 40.0
            except Exception:
                total_frames = 34000
                n_unique = int(34000 * 0.60)
                dedup_ratio = 40.0
            n_unique_source = "ước lượng thận trọng (n=4)"

        # Kiểm tra nếu file 4K thành phẩm đã tồn tại (hỗ trợ cả _4K_animejanai.mkv, _4K.mkv và file -o tường minh)
        existing_4k = find_existing_4k_output_path(vid, output_arg=output_dir)
        already_completed = existing_4k is not None

        # 2. Tính ETA trung vị (thời gian kỳ vọng) và ETA P5 (+7% biên an toàn cho quyết định)
        if already_completed:
            ai_seconds_median = 0.0
            eta_median_seconds = 0.0
            ai_seconds_p5 = 0.0
            eta_p5_seconds = 0.0
            decision_eta_seconds = 0.0
            decision = "⏩ ĐÃ XONG (BỎ QUA)"
            can_run = False
            is_partial = False
            is_already_completed = True
            ep_start_time = sim_time_median
            ep_finish_time = sim_time_median
            ep_decision_finish_time = sim_time_conservative
        else:
            is_already_completed = False
            ai_seconds_median = n_unique / fps_med
            eta_median_seconds = ai_seconds_median + total_overhead

            ai_seconds_p5 = n_unique / fps_conservative
            eta_p5_seconds = ai_seconds_p5 + total_overhead
            decision_eta_seconds = eta_p5_seconds * (1.0 + eta_margin_pct)

            ep_start_time = sim_time_median
            ep_finish_time = sim_time_median + timedelta(seconds=eta_median_seconds)
            ep_decision_finish_time = sim_time_conservative + timedelta(seconds=decision_eta_seconds)

            # 3. Đối chiếu deadline theo P5 + 7% margin cộng dồn
            decision = "KỊP"
            can_run = True
            is_partial = False

            if deadline_dt:
                # subtitles_overhead_seconds is already accumulated inside sim_time_conservative per episode
                effective_deadline_dt = deadline_dt

                if ep_decision_finish_time <= effective_deadline_dt:
                    decision = "KỊP"
                    can_run = True
                else:
                    if allow_partial and effective_deadline_dt > sim_time_conservative:
                        decision = "CHẠY 1 PHẦN (ĐẾN HẠN CHÓT)"
                        can_run = True
                        is_partial = True
                    else:
                        decision = "BỎ QUA (QUÁ DEADLINE)"
                        can_run = False

        if can_run:
            if is_partial and deadline_dt:
                # Chạy dở dang đến hạn chót (đã trừ thời gian phụ đề dành riêng)
                partial_run_sec = max(0.0, (effective_deadline_dt - sim_time_conservative).total_seconds())
                sim_time_conservative = effective_deadline_dt
                sim_time_median += timedelta(seconds=partial_run_sec)
                total_scheduled_seconds += partial_run_sec
            else:
                sim_time_conservative = ep_decision_finish_time
                sim_time_median = ep_finish_time
                total_scheduled_seconds += eta_median_seconds

        episodes_meta.append({
            "index": idx + 1,
            "file_path": vid,
            "file_name": os.path.basename(vid),
            "is_exact_plan": is_exact,
            "n_unique_source": n_unique_source,
            "total_frames": total_frames,
            "n_unique": n_unique,
            "dedup_ratio_percent": round(dedup_ratio, 2),
            "ai_seconds_median": round(ai_seconds_median, 1),
            "total_overhead_seconds": round(total_overhead, 1),
            "eta_median_seconds": round(eta_median_seconds, 1),
            "eta_p5_seconds": round(eta_p5_seconds, 1),
            "decision_eta_seconds": round(decision_eta_seconds, 1),
            "eta_formatted": str(timedelta(seconds=int(round(eta_median_seconds)))),
            "decision_eta_formatted": str(timedelta(seconds=int(round(decision_eta_seconds)))),
            "start_time": ep_start_time.strftime("%H:%M:%S"),
            "finish_time": ep_finish_time.strftime("%H:%M:%S"),
            "decision_finish_time": ep_decision_finish_time.strftime("%H:%M:%S"),
            "can_run": can_run,
            "is_partial": is_partial,
            "is_already_completed": is_already_completed,
            "decision": decision
        })

    available_seconds = (deadline_dt - now).total_seconds() if deadline_dt else None

    return {
        "start_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "deadline_str": deadline_str,
        "deadline_time": deadline_dt.strftime("%Y-%m-%d %H:%M:%S") if deadline_dt else None,
        "available_time_formatted": str(timedelta(seconds=int(available_seconds))) if available_seconds else "Không giới hạn",
        "fps_median": fps_med,
        "fps_p5": fps_conservative,
        "video_overhead_seconds": video_overhead_seconds,
        "subtitles_overhead_seconds": subtitles_overhead_seconds,
        "total_overhead_seconds": total_overhead,
        "eta_margin_pct": eta_margin_pct,
        "allow_partial": allow_partial,
        "total_scheduled_seconds": total_scheduled_seconds,
        "total_scheduled_formatted": str(timedelta(seconds=int(round(total_scheduled_seconds)))),
        "accepted_count": sum(1 for ep in episodes_meta if ep["can_run"]),
        "already_completed_count": sum(1 for ep in episodes_meta if ep.get("is_already_completed")),
        "skipped_count": sum(1 for ep in episodes_meta if (not ep["can_run"] and not ep.get("is_already_completed"))),
        "partial_count": sum(1 for ep in episodes_meta if ep.get("is_partial")),
        "total_episodes": len(episodes_meta),
        "subs_quality": subs_quality,
        "episodes": episodes_meta
    }


def check_gemini_quota_for_queue(
    queue_len: int,
    subs_quality: str = "normal",
    cache: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Kiểm tra hạn mức Gemini theo giờ Thái Bình Dương (PT) dựa trên chế độ chất lượng thực tế
    (`normal`/`fast` -> `gemini-3.5-flash-lite`, `high` -> `gemini-3.8-flash`) và nhân đúng số lượng API keys.
    """
    try:
        from subtrans.cache import TranslationCache, get_next_pt_reset, compute_key_id
        from subtrans.config import load_models_config, get_gemini_api_keys
        cache_inst = cache if cache is not None else TranslationCache()
        cues_today = cache_inst.get_cues_today_pt()
        reqs_today = cache_inst.get_requests_today_pt()
        models_cfg = load_models_config()
        configured_models = models_cfg.get("models", {})
        api_keys = get_gemini_api_keys()
        next_pt, next_vn = get_next_pt_reset()
    except Exception:
        cache_inst = None
        cues_today = {}
        reqs_today = {}
        configured_models = {}
        api_keys = []
        next_pt, next_vn = None, None
        compute_key_id = lambda k: ""

    num_keys = max(1, len(api_keys))
    # Với batch_size = 100, tập ~300 cues cần ~3-4 batch requests (+ 0-1 request Cast/Cascade)
    est_batches_needed = queue_len * 4
    primary_model = "gemini-3.8-flash" if subs_quality == "high" else "gemini-3.5-flash-lite"
    default_per_key_rpd = 20 if primary_model == "gemini-3.8-flash" else 500
    primary_info = configured_models.get(primary_model, {})
    per_key_rpd = primary_info.get("rpd", default_per_key_rpd)
    if not isinstance(per_key_rpd, int):
        per_key_rpd = default_per_key_rpd
    primary_limit_rpd = per_key_rpd * num_keys

    primary_reqs_used = reqs_today.get(primary_model, 0)
    primary_cues_used = cues_today.get(primary_model, 0)
    if cache_inst is not None:
        if api_keys:
            key_hashes = [compute_key_id(k) for k in api_keys]
            is_exhausted = cache_inst.are_all_keys_exhausted(key_hashes, primary_model)
        else:
            is_exhausted = cache_inst.is_model_exhausted(primary_model)
    else:
        is_exhausted = False
    primary_exhausted = is_exhausted or (primary_reqs_used >= primary_limit_rpd)
    primary_remaining = max(0, primary_limit_rpd - primary_reqs_used) if not primary_exhausted else 0

    reset_str = f"00:00 PT ({next_vn.strftime('%H:%M ngày %d/%m')} giờ VN)" if next_vn else "14:00 giờ VN"

    warnings = []
    if primary_exhausted:
        warnings.append(
            f"⚠️ Model chính '{primary_model}' đã hết hạn mức ngày ({primary_limit_rpd} RPD / {num_keys} key; khóa đến {reset_str}). "
            f"Hàng đợi phụ đề sẽ tự động chuyển sang chuỗi fallback kế tiếp trong models.yaml!"
        )
    elif primary_remaining < est_batches_needed:
        warnings.append(
            f"⚠️ Hạn mức còn lại của model chính '{primary_model}' ({primary_remaining}/{primary_limit_rpd} reqs trên {num_keys} key) nhỏ hơn "
            f"nhu cầu dự kiến (~{est_batches_needed} reqs cho {queue_len} tập @ 100 cues/batch). "
            f"Các tập sau sẽ tự động kế thừa chuỗi fallback!"
        )

    return {
        "primary_model": primary_model,
        "num_keys": num_keys,
        "per_key_limit_rpd": per_key_rpd,
        "primary_limit_rpd": primary_limit_rpd,
        "primary_reqs_used": primary_reqs_used,
        "primary_cues_used": primary_cues_used,
        "primary_remaining_requests": primary_remaining,
        "estimated_requests_needed": est_batches_needed,
        "cues_today_pt": cues_today,
        "requests_today_pt": reqs_today,
        "next_reset_vn": next_vn.strftime('%H:%M %d/%m') if next_vn else "14:00 VN",
        "warnings": warnings
    }


def format_schedule_decision_table(sched: Dict[str, Any]) -> str:
    """
    Định dạng bảng quyết định lịch trình qua đêm rõ ràng, chuyên nghiệp.
    In rõ cột n_unique là "đo (plan.json)" hay "ước lượng" cho từng tập.
    """
    lines = []
    lines.append("\n" + "=" * 126)
    lines.append("📊 BẢNG QUYẾT ĐỊNH LẬP LỊCH HÀNG ĐỢI QUA ĐÊM (OVERNIGHT BATCH SCHEDULER)")
    lines.append("=" * 126)
    lines.append(f"• Thời điểm bắt đầu    : {sched['start_time']}")
    lines.append(f"• Mốc hạn chót          : {sched['deadline_str'] or 'Không đặt'} ({sched['deadline_time'] or 'N/A'})")
    lines.append(f"• Quỹ thời gian có sẵn : {sched['available_time_formatted']}")
    lines.append(f"• Tốc độ AI suy luận    : Trung vị {sched['fps_median']:.2f} fps | P5 {sched['fps_p5']:.2f} fps (Unique Inference FPS trên {DEVICE_NAME})")
    lines.append(f"• Overhead mỗi tập      : {sched['total_overhead_seconds']:.0f}s (~{sched['total_overhead_seconds']/60:.1f} phút: {sched['video_overhead_seconds']:.0f}s video + {sched['subtitles_overhead_seconds']:.0f}s phụ đề)")
    lines.append(f"• Biên an toàn ETA      : +{sched['eta_margin_pct']*100:.1f}% trên ETA P5 để ra quyết định hạn chót")
    lines.append(f"• Cho phép dở dang      : --allow-partial = {'BẬT' if sched.get('allow_partial') else 'TẮT'}")
    lines.append("-" * 126)

    header = (
        f"{'STT':<4} | {'Tập Video':<34} | {'Tổng Frames':<11} | "
        f"{'n_unique':<9} | {'Nguồn n_unique':<25} | {'Dedup %':<7} | {'ETA (Med)':<9} | "
        f"{'ETA P5 (+7%)':<12} | {'Kết Thúc':<8} | {'Quyết Định'}"
    )
    lines.append(header)
    lines.append("-" * 126)

    for ep in sched["episodes"]:
        name_display = ep["file_name"]
        if len(name_display) > 32:
            name_display = name_display[:29] + "..."

        if ep.get("is_already_completed"):
            dec_symbol = "⏩ ĐÃ CÓ (BỎ QUA)"
        elif ep.get("is_partial"):
            dec_symbol = "⏸️ CHẠY 1 PHẦN (HẠN CHÓT)"
        elif ep["can_run"]:
            dec_symbol = "✅ KỊP"
        else:
            dec_symbol = "❌ BỎ QUA (QUÁ HẠN)"

        row = (
            f"[{ep['index']:02d}] | {name_display:<34} | {ep['total_frames']:<11} | "
            f"{ep['n_unique']:<9} | {ep['n_unique_source']:<25} | {ep['dedup_ratio_percent']:<6.1f}% | "
            f"{ep['eta_formatted']:<9} | {ep['decision_eta_formatted']:<12} | {ep['finish_time']:<8} | {dec_symbol}"
        )
        lines.append(row)

    lines.append("-" * 126)
    lines.append("Ghi chú: Cột 'Nguồn n_unique' phân định rõ giữa số đo thực tế từ plan.json và số ước lượng thận trọng (~60.0% kept, n=4).")
    lines.append(
        f"         Quyết định hạn chót dựa trên P5 ({sched['fps_p5']:.2f} fps) + overhead "
        f"(video ~{sched['video_overhead_seconds']:.0f}s + {sched['subtitles_overhead_seconds']:.0f}s phụ đề) "
        f"+ biên an toàn {sched['eta_margin_pct']*100:.1f}% ETA."
    )
    partial_note = f" (gồm {sched['partial_count']} tập chạy 1 phần)" if sched.get("partial_count") else ""
    already_note = f", {sched['already_completed_count']} tập đã có kết quả bỏ qua" if sched.get("already_completed_count") else ""
    lines.append(f"🎯 TỔNG KẾT: Duyệt thực thi {sched['accepted_count']}/{sched['total_episodes']} tập{partial_note}{already_note} "
                 f"| Tổng thời lượng chạy dự kiến: {sched['total_scheduled_formatted']}.")

    # Hiển thị cảnh báo hạn mức API Gemini nếu có
    quota_info = sched.get("quota_info") or check_gemini_quota_for_queue(
        len(sched.get("episodes", [])),
        subs_quality=sched.get("subs_quality", "normal"),
    )
    if quota_info.get("warnings"):
        lines.append("-" * 126)
        lines.append("🔔 GIÁM SÁT HẠN MỨC SUBTITLES (GEMINI API - PACIFIC TIME):")
        for w in quota_info["warnings"]:
            lines.append(f"  {w}")
        lines.append(f"  • Thống kê hôm nay (PT): {quota_info['cues_today_pt']}")

    lines.append("=" * 126 + "\n")

    return "\n".join(lines)
