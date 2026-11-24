#!/usr/bin/env python3
"""CLI entrypoint for 4K anime super-resolution, deduplication, and optional subtitle translation."""

import os
import sys

# Đảm bảo console Windows xuất UTF-8 an toàn với emoji và tiếng Việt có dấu
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import inspect
import json
import re
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Any, Optional, List, Dict, Tuple, Sequence
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env_isolation import configure_e_drive_isolation
configure_e_drive_isolation(project_root=ROOT_DIR)

from src.config import (
    DEFAULT_MODEL,
    DEFAULT_ENGINE,
    DEFAULT_PRECISION,
    DEFAULT_BITRATE,
    DEFAULT_SEGMENT_TIME,
    DEFAULT_ZERO_DISK_SEGMENT_TIME,
    DEFAULT_FRAME_FORMAT,
    DEFAULT_BATCH_SIZE,
    DEVICE_NAME,
    HEVC_ENCODER,
    HEVC_MULTIPASS_POLICY,
    VALID_NVENC_MULTIPASS_POLICIES,
    DEFAULT_GRAIN_RESTORE,
    VALID_GRAIN_RESTORE_POLICIES,
)
from src.pending_queue import load_pending_subs, remove_pending_sub
from src.pipeline import run_upscale_pipeline, FailedResumableError
from src.config import is_gpu_lock_held_by_other
from src.logger import (
    get_live_status_path,
    is_graceful_pause_requested,
    clear_graceful_pause,
    was_graceful_paused,
)
from src.ffmpeg_ops import probe_video
from src.dedup import DEDUP_POLICY_PARAMS, get_dedup_params, resolve_dedup_policy, spawn_background_preflight
from src.models_resolver import (
    list_models_formatted,
    resolve_expected_4k_output_path,
    find_existing_4k_output_path,
    resolve_model_for_video,
)
from src.cleanup import (
    cleanup_checkpoint_dir,
    cleanup_subtitle_extracts,
    organize_output_artifacts,
    clean_all_temporary_data,
)
from src.subtitles import find_or_extract_subtitles, transcribe_japanese_audio


_MODEL_TRT_ZERO_DISK_FPS: Dict[str, Tuple[float, float]] = {
    "animevideov3": (10.5, 8.5),
    "animejanai-hd-v3-compact": (12.8, 10.2),
    "animejanai-hd-v3-ultracompact": (18.5, 15.5),
    "animejanai-hd-v3-sharp1-ultracompact": (18.5, 15.5),
    "animejanai-hd-v3-superultracompact": (20.8, 17.5),
    "animejanai-hd-v3-sharp1-superultracompact": (20.8, 17.5),
    "span-2x": (10.5, 8.5),
    "2x_animejanai_v3": (10.5, 8.5),
    "modernspanimation-v2-2x": (10.5, 8.5),
    "realcugan-pro-conservative-2x": (4.5, 3.6),
    "realcugan-pro-denoise3x-2x": (4.5, 3.6),
    "auto-pareto": (16.5, 13.5),
}


def _resolve_model_id_safe(video_path: Optional[str], cli_model: Optional[str]) -> str:
    try:
        info = resolve_model_for_video(video_path or "", cli_model_arg=cli_model)
        return str(info.get("id") or DEFAULT_MODEL).strip().lower()
    except Exception:
        return str(cli_model or DEFAULT_MODEL).strip().lower()


def _resolve_model_fps_pair(
    model_id: str,
    engine: str,
    has_trt: bool,
    zero_disk: bool,
) -> Tuple[float, float]:
    base_med, base_p5 = _MODEL_TRT_ZERO_DISK_FPS.get(str(model_id or "").lower(), (10.5, 8.5))
    if engine != "cuda":
        return (round(1.35 * (base_med / 10.5), 2), round(1.25 * (base_p5 / 8.5), 2))
    if has_trt and zero_disk:
        return (base_med, base_p5)
    if has_trt:
        return (round(base_med * (8.5 / 10.5), 2), round(base_p5 * (7.0 / 8.5), 2))
    return (round(base_med * (5.2 / 10.5), 2), round(base_p5 * (4.5 / 8.5), 2))


def _validate_dedup_policy_arg(raw_value: str) -> str:
    norm = str(raw_value or "").strip().lower()
    if not norm:
        raise argparse.ArgumentTypeError("Empty --dedup-policy value.")
    try:
        get_dedup_params(norm)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return norm


class _DedupPolicyChoices(tuple):
    """Choices tuple preserving canonical keys while accepting dynamic dis-<float>/drift-<float>."""

    def __contains__(self, item: object) -> bool:
        if super().__contains__(item):
            return True
        if not isinstance(item, str):
            return False
        try:
            get_dedup_params(item)
            return True
        except ValueError:
            return False


def check_deadline_and_eta(
    video_path: str,
    deadline_str: str,
    fps_median: float = 10.5,
    fps_p5: float = 8.5,
    video_overhead: float = 420.0,
    subtitles_overhead: float = 300.0,
    eta_margin_pct: float = 0.07,
    allow_partial: bool = False,
    output_dir: Optional[str] = None,
    reserved_seconds: float = 0.0,
    dedup_policy: Optional[str] = None,
    base_deadline_dt: Optional[datetime] = None,
    grain_restore: str = "auto",
    force: bool = False,
) -> Tuple[bool, bool, Optional[datetime]]:
    """
    Kiểm tra thời gian hoàn thành dự toán của tập video so với mốc deadline.
    Quyết định dựa trên P5 Unique Inference FPS (8.5 fps) + overhead (video ~420s + 300s subs) + 7% ETA margin.
    Trung vị Unique Inference FPS: 10.5 fps (RTX 3060 TensorRT FP16 Zero-Disk).
    Returns: (can_run, is_partial, deadline_dt)
    """
    now = datetime.now()
    if base_deadline_dt is not None:
        deadline_dt = base_deadline_dt
    else:
        try:
            dh, dm = map(int, deadline_str.split(":"))
            deadline_dt = now.replace(hour=dh, minute=dm, second=0, microsecond=0)
            if deadline_dt <= now:
                deadline_dt += timedelta(days=1)
        except Exception as e:
            print(f"⚠️ Định dạng deadline không hợp lệ '{deadline_str}' (cần HH:MM): {e}")
            return True, False, None

    if reserved_seconds and reserved_seconds > 0:
        deadline_dt = deadline_dt - timedelta(seconds=float(reserved_seconds))

    seconds_left = (deadline_dt - now).total_seconds()
    eff_policy = resolve_dedup_policy(video_path, dedup_policy)

    # Đọc số frame độc nhất từ plan.json nếu có (ưu tiên thư mục output -o nếu được truyền)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_root = output_dir or "out"
    if out_root.lower().endswith((".mkv", ".mp4")) or os.path.isfile(out_root):
        out_root = os.path.dirname(os.path.abspath(out_root)) or "out"
    candidate_plans = []
    for p in (
        os.path.splitext(video_path)[0] + ".plan.json",
        os.path.join(out_root, f"{base_name}.plan.json"),
        os.path.join(out_root, f".temp_{base_name}", "plan.json"),
        f"out/{base_name}.plan.json",
        f"out/.temp_{base_name}/plan.json",
    ):
        if p not in candidate_plans:
            candidate_plans.append(p)
    n_unique = None
    is_measured = False
    norm_grain = str(grain_restore or "auto").strip().lower()
    matched_plan: Optional[Dict[str, Any]] = None
    total_f_est = 34000
    for p in candidate_plans:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                    plan_pol = pdata.get("dedup_policy")
                    if eff_policy and plan_pol is not None:
                        if str(plan_pol).strip().lower() != str(eff_policy).strip().lower():
                            continue
                    c_params = pdata.get("dedup_params") or {}
                    c_grain = c_params.get("grain_restore")
                    if c_grain is not None and str(c_grain).strip().lower() != norm_grain:
                        continue
                    if c_grain is None and norm_grain != "auto":
                        continue
                    cand_kept = pdata.get("total_kept_frames")
                    if cand_kept:
                        n_unique = cand_kept
                        total_f_est = int(pdata.get("total_frames") or 34000)
                        matched_plan = pdata
                        is_measured = True
                        break
            except Exception:
                pass

    if n_unique is None:
        # Ước lượng thận trọng 60.0% frame giữ nếu chưa quét (thay thế 39.1% cũ, đo n=4: 39.1%, 46.0%, 54.4%, 61.3%)
        keep_factor = 1.0 if eff_policy == "off" else 0.60
        try:
            meta = probe_video(video_path)
            total_f = meta.get("nb_frames") or int(meta.get("duration", 1440) * 23.976)
            total_f_est = int(total_f)
            n_unique = int(total_f * keep_factor)
        except Exception:
            n_unique = int(34000 * keep_factor)

    source_tag = "đo (plan.json)" if is_measured else "ước lượng thận trọng (n=4)"
    eff_vid_overhead = float(video_overhead)
    if not force:
        try:
            from src.scheduler import _inspect_resumable_checkpoint
            done_c, tot_c, rem_frac, rem_u = _inspect_resumable_checkpoint(
                video_path, output_dir, matched_plan, total_f_est, int(n_unique), DEFAULT_ZERO_DISK_SEGMENT_TIME
            )
            if done_c > 0:
                n_unique = rem_u
                eff_vid_overhead = float(video_overhead) * max(0.15, rem_frac)
                source_tag = f"{source_tag} (còn {tot_c - done_c}/{tot_c} chunk)"
        except Exception:
            pass

    total_overhead = eff_vid_overhead + subtitles_overhead
    eta_p5 = (n_unique / max(0.5, fps_p5)) + total_overhead
    decision_eta = eta_p5 * (1.0 + eta_margin_pct)
    eta_med = (n_unique / max(0.5, fps_median)) + total_overhead

    if decision_eta > seconds_left:
        if allow_partial and seconds_left > 0:
            print(f"\n⏳ [ALLOW PARTIAL] Tập '{os.path.basename(video_path)}':")
            print(f"   • Cần: ~{eta_med/60:.1f}m (kỳ vọng trung vị) | Quyết định hạn chót: {decision_eta/60:.1f}m (P5 {fps_p5:.2f} fps + 5m subs + 7% margin) cho {n_unique} unique frames [{source_tag}]")
            print(f"   • Thời gian còn lại trước {deadline_str}: {seconds_left/60:.1f} phút ({seconds_left/3600:.2f}h)")
            print(f"   -> Chế độ --allow-partial đang BẬT: Sẽ chạy đến hạn chót {deadline_str}, dừng sạch ở ranh giới chunk và giữ segment!\n")
            return True, True, deadline_dt
        else:
            print(f"\n⏰ [DEADLINE SKIP] Tập '{os.path.basename(video_path)}':")
            print(f"   • Dự toán cần: {decision_eta/60:.1f} phút (P5 {fps_p5:.2f} fps + 5m subs + 7% margin) cho ~{n_unique} unique frames [{source_tag}]")
            print(f"   • Thời gian còn lại trước {deadline_str}: {seconds_left/60:.1f} phút ({seconds_left/3600:.2f}h)")
            print(f"   -> Bỏ qua tập này để không bị dở dang ca qua đêm! (Dùng --allow-partial nếu muốn chạy một phần)\n")
            return False, False, deadline_dt

    print(f"⏳ [DEADLINE OK] Tập '{os.path.basename(video_path)}' ({n_unique} unique frames [{source_tag}]): Cần ~{eta_med/60:.1f}m (Quyết định P5+7%: {decision_eta/60:.1f}m), còn {seconds_left/60:.1f}m trước hạn chót {deadline_str}.")
    return True, False, deadline_dt


def compute_pending_subs_reserved_seconds(
    completed_videos: Sequence[Tuple[str, str]],
    async_sub_procs: Sequence[Tuple[str, str, Any, Any]],
    subtitles_overhead: float,
    subs_after_upscale: bool,
) -> float:
    """Return remaining sequential Stage B subtitle seconds not yet launched in async_sub_procs."""
    if not subs_after_upscale or subtitles_overhead <= 0:
        return 0.0
    async_by_src = {src: proc for src, _, proc, _ in async_sub_procs}
    pending_count = sum(1 for src, _ in completed_videos if src not in async_by_src)
    return float(pending_count) * float(subtitles_overhead)


_ASCII_INT_ARG_RE = re.compile(r"^[0-9]+$")
_ASCII_FLOAT_ARG_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")


def _parse_preview_start_arg(raw_val: str) -> float:
    """Parse preview start timestamp in seconds ('845.5') or strict 'MM:SS' / 'HH:MM:SS' format."""
    cleaned = str(raw_val or "").strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("--preview-start không được để trống.")
    if ":" in cleaned:
        parts = cleaned.split(":")
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(f"Định dạng thời gian không hợp lệ: {raw_val!r} (dùng giây, MM:SS hoặc HH:MM:SS).")
        if not all(_ASCII_INT_ARG_RE.fullmatch(p) for p in parts[:-1]) or not _ASCII_FLOAT_ARG_RE.fullmatch(parts[-1]):
            raise argparse.ArgumentTypeError(f"Định dạng thời gian không hợp lệ: {raw_val!r} (HH/MM phải là số nguyên >= 0).")
        ss = float(parts[-1])
        if ss >= 60.0:
            raise argparse.ArgumentTypeError(f"Số giây trong {raw_val!r} phải nhỏ hơn 60.")
        if len(parts) == 2:
            sec = int(parts[0]) * 60.0 + ss
        else:
            mm = int(parts[1])
            if mm >= 60:
                raise argparse.ArgumentTypeError(f"Số phút trong {raw_val!r} phải nhỏ hơn 60.")
            sec = int(parts[0]) * 3600.0 + mm * 60.0 + ss
    else:
        if not _ASCII_FLOAT_ARG_RE.fullmatch(cleaned):
            raise argparse.ArgumentTypeError(f"Số giây --preview-start không hợp lệ: {raw_val!r}.")
        sec = float(cleaned)
    if not (0.0 <= sec <= 86400.0 and sec == sec and sec != float("inf")):
        raise argparse.ArgumentTypeError("--preview-start phải nằm trong khoảng 0..86400 giây.")
    return round(sec, 3)


def _filter_pipeline_kwargs(fn, extra_kw: Dict[str, Any]) -> Dict[str, Any]:
    """Pass optional pipeline kwargs only when accepted by fn (preserving strict test fakes)."""
    if hasattr(fn, "_mock_name"):
        return dict(extra_kw)
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(extra_kw)
    return {k: v for k, v in extra_kw.items() if k in params}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=f"🎬 AI Video Upscaler 4K - Tối ưu hóa chuyên sâu cho {DEVICE_NAME} ({DEFAULT_ENGINE.upper()} + {HEVC_ENCODER} 10-bit)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input",
        nargs="*",
        default=[],
        help="Đường dẫn đến một/nhiều file video nguồn (.mkv, .mp4, .webm...) hoặc thư mục chứa các tập video"
    )
    parser.add_argument(
        "--queue",
        nargs="+",
        default=None,
        help="Danh sách nhiều tập video cần chạy tuần tự trong ca qua đêm"
    )
    parser.add_argument(
        "--deadline",
        default=None,
        metavar="HH:MM",
        help="Mốc hạn chót dừng hàng đợi (ví dụ: 07:00). Nếu tập tiếp theo dự kiến không kịp thì không bắt đầu."
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Nếu tập tiếp theo không kịp hạn chót --deadline, vẫn cho phép chạy đến hạn chót, dừng sạch ở ranh giới chunk, giữ các phân đoạn up_*.mp4 để tiếp tục lần sau."
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Chế độ benchmark thuần túy: Bộ lập lịch và preflight nền từ chối chạy để tránh gây nhiễu tài nguyên hệ thống."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Đường dẫn file hoặc thư mục xuất 4K (Mặc định: out/<tên>_4K.mkv)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Kích thước lô xử lý khung hình (Mặc định: {DEFAULT_BATCH_SIZE} cho {DEVICE_NAME})"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Mô hình AI upscale (Mặc định: tự động chọn qua shows.yaml, mặc định animevideov3, hoặc auto-pareto)"
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        choices=["cuda", "cpu"],
        help=f"Động cơ suy luận AI (Mặc định: {DEFAULT_ENGINE} - PyTorch {DEFAULT_ENGINE.upper()} FP16)"
    )
    parser.add_argument(
        "--precision",
        default=DEFAULT_PRECISION,
        choices=["fp16", "fp32"],
        help=f"Độ chính xác suy luận (Mặc định: {DEFAULT_PRECISION})"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Liệt kê danh sách các mô hình Super-Resolution đã đăng ký trong models/models.yaml (D1, D2)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bắt buộc chạy và xóa phân đoạn cũ nếu phát hiện xung đột mô hình (D7)"
    )
    parser.add_argument(
        "-b", "--bitrate",
        default=DEFAULT_BITRATE,
        help=f"Bitrate video 4K HEVC 10-bit (Mặc định: {DEFAULT_BITRATE})"
    )
    parser.add_argument(
        "-s", "--segment",
        type=int,
        default=DEFAULT_SEGMENT_TIME,
        help=f"Độ dài phân đoạn cuốn chiếu tính bằng giây (Mặc định: {DEFAULT_SEGMENT_TIME}s - tối ưu cho RAM Disk)"
    )
    parser.add_argument(
        "-f", "--format",
        default=DEFAULT_FRAME_FORMAT,
        choices=["jpg", "png"],
        help=f"Định dạng khung hình trung gian (Mặc định: {DEFAULT_FRAME_FORMAT})"
    )
    parser.add_argument(
        "--no-ramdisk",
        action="store_true",
        help="Không sử dụng RAM Disk ảo mà dùng thư mục tạm trên ổ cứng SSD"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Không tiếp tục từ phân đoạn đã làm dở mà chạy lại từ đầu"
    )
    parser.add_argument(
        "--no-anti-sleep",

        action="store_true",
        help="Không tự động kích hoạt cơ chế chống sleep máy (SetThreadExecutionState trên Windows)"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Không sử dụng bộ lọc frame trùng (Deduplication) mà tính toán 100%% tất cả các khung hình"
    )
    parser.add_argument(
        "--dedup-policy",
        default=None,
        type=_validate_dedup_policy_arg,
        choices=_DedupPolicyChoices(DEDUP_POLICY_PARAMS.keys()),
        help="Chính sách Deduplication: default (hi=768:lo=320:frac=0.33), dis-auto (tự hiệu chuẩn sàn nhiễu hạt + lưới khẩu hình 8x8 + lia máy bầu trời phẳng), dis/dis-<float>, sakuga/dis-sakuga, sensitive/sensitive-a/sensitive-b, motion, drift/drift-<float>, off"
    )
    parser.add_argument(
        "--nvenc-multipass",
        default=HEVC_MULTIPASS_POLICY,
        choices=list(VALID_NVENC_MULTIPASS_POLICIES),
        help="Chính sách NVENC 2-pass toàn tập (không trộn SPS/PPS giữa các chunk): auto (qres khi AI > 21.5 fps, fullres khi AI-bound), fullres, qres"
    )
    parser.add_argument(
        "--grain-restore",
        default=DEFAULT_GRAIN_RESTORE,
        choices=list(VALID_GRAIN_RESTORE_POLICIES),
        help="Chính sách khử hạt & pha trọng số cấp tập (auto: tự đo hạt phẳng-tĩnh <0.35s, off: tắt, light: nhẹ alpha=0.45, heavy: nặng alpha=1.0)"
    )
    parser.add_argument(
        "--drift-delta",
        type=float,
        default=None,
        help="Ngưỡng dịch tích lũy δ (px 1080p) cho chính sách 'drift' (mặc định: 0.75)"
    )
    parser.add_argument(
        "--dis-thresh",
        type=float,
        default=None,
        help="Ngưỡng dòng chảy quang học τ (px 1080p) cho chính sách 'dis' (mặc định: 0.30)"
    )
    parser.add_argument(
        "--preview",
        type=int,
        nargs="?",
        const=10,
        default=None,
        metavar="SECONDS",
        help="Chế độ xem trước: trích xuất nhanh N giây (mặc định: 10s) ở giữa phim, upscale và tự mở bằng trình phát video"
    )
    parser.add_argument(
        "--preview-start",
        type=_parse_preview_start_arg,
        default=None,
        metavar="SEC_OR_MM:SS",
        help="Mốc thời gian bắt đầu trích xuất Preview (giây hoặc MM:SS / HH:MM:SS; mặc định: 30%% thời lượng tập)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chế độ chạy thử lập lịch (Dry Run): Quét preflight hàng đợi, tính toán ETA từng tập và xuất bảng quyết định theo --deadline mà không chạy GPU."
    )
    parser.add_argument(
        "--subs-after-upscale",
        action="store_true",
        default=True,
        help="Tự động chạy dịch phụ đề (subtrans) sau khi toàn bộ hàng đợi upscale hoàn tất (Mặc định: Bật)"
    )
    parser.add_argument(
        "--no-subs", "--no-subs-after-upscale",
        dest="subs_after_upscale",
        action="store_false",
        help="Tắt bước dịch phụ đề sau upscale"
    )
    parser.add_argument(
        "--subs-only",
        nargs="?",
        const="__INPUT__",
        default=None,
        metavar="VIDEO",
        help="Chỉ chạy riêng bước dịch phụ đề (subtrans) cho tập video chỉ định trong tiến trình riêng (không chạy AI upscale)"
    )
    parser.add_argument(
        "--subs-upgrade",
        action="store_true",
        help="Dịch nâng cấp lại các câu thoại trước đó được dịch bằng Lite/Local bằng model chất lượng cao hơn khi còn hạn mức"
    )
    parser.add_argument(
        "--sub-offset-ms",
        type=int,
        default=0,
        help="Độ lệch thời gian dịch chuyển phụ đề tiếng Nhật / tiếng Việt (mili-giây, ví dụ: -11154 cho bản Crunchyroll lệch intro)"
    )
    parser.add_argument(
        "--sub-offset-target",
        type=str,
        default="",
        help="Tên file hoặc chuỗi nhận diện tập cụ thể áp dụng cờ --sub-offset-ms trong hàng đợi"
    )
    parser.add_argument(
        "--auto-offset",
        action="store_true",
        default=None,
        help="Tự động phát hiện và áp dụng bù lệch thời gian JA-EN cho phụ đề (Mặc định: Bật trong hàng đợi, Tắt khi chạy 1 tập đơn lẻ)"
    )
    parser.add_argument(
        "--no-auto-offset",
        action="store_true",
        default=False,
        help="Tắt cơ chế tự động bù lệch thời gian phụ đề trong hàng đợi"
    )
    parser.add_argument(
        "--auto-align-en",
        action="store_true",
        default=True,
        help="Tự động ghim mốc thời gian Whisper theo track tiếng Anh phòng thu (Mặc định: Bật)"
    )
    parser.add_argument(
        "--no-auto-align-en",
        dest="auto_align_en",
        action="store_false",
        help="Tắt tính năng tự động ghim mốc thời gian Whisper theo track tiếng Anh phòng thu"
    )
    parser.add_argument(
        "--prefer-whisper",
        action="store_true",
        default=False,
        help="Khi MKV thiếu phụ đề JA (chỉ có audio JA + sub EN), chạy Whisper STT + DTW align EN để giữ đầy đủ đại từ/kính ngữ tiếng Nhật"
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        default=False,
        help="Không nhúng track phụ đề vào container MKV 4K, chỉ xuất và giữ file .vi.srt ngoài"
    )
    parser.add_argument(
        "--night-report",
        nargs="?",
        const="night1",
        default=None,
        metavar="SESSION",
        help="Sinh báo cáo tổng kết phiên chạy đêm từ log, telemetry CSV, audit.json và api_requests (Chỉ đọc, không gọi GPU/API)"
    )
    parser.add_argument(
        "--sound-cues",
        choices=["drop", "translate"],
        default="drop",
        help="Xử lý chú thích âm thanh thuần ngoặc đơn: 'drop' (bỏ qua, mặc định) hoặc 'translate' (dịch)"
    )
    parser.add_argument(
        "--subs-quality",
        choices=["normal", "fast", "high"],
        default="normal",
        help="Chất lượng dịch phụ đề: 'normal'/'fast' (Lite trước, tối ưu quota) hoặc 'high' (Flash trước)"
    )
    parser.add_argument(
        "--glossary", "-g",
        type=str,
        default=None,
        help="Đường dẫn file Glossary YAML tùy chọn dành cho bước dịch phụ đề (Subtrans)"
    )
    parser.add_argument(
        "--subs-pending",
        action="store_true",
        help="Chạy lại quy trình dịch phụ đề cho các tập ghi nhận trong pending_subs.json"
    )
    parser.add_argument(
        "--target-4k",
        default=None,
        help="Đường dẫn container MKV 4K đích cụ thể để nhúng phụ đề trong chế độ --subs-only / Stage B"
    )
    parser.add_argument(
        "--async-subs",
        action="store_true",
        default=False,
        help="Chạy bất đồng bộ tiến trình dịch phụ đề Cloud Gemini của tập N song song khi GPU đang upscale tập N+1 (tự tắt nếu bật --prefer-whisper)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help="Giữ lại thư mục tạm và checkpoint (.temp_<basename>) sau khi upscale thành công (Mặc định: Tự động dọn dẹp sạch sẽ để tiết kiệm dung lượng SSD)"
    )
    parser.add_argument(
        "--clean-temp", "--cleanup",
        action="store_true",
        default=False,
        help="Quét và dọn dẹp sạch sẽ toàn bộ thư mục tạm (.temp_*), symlinks mồ côi và gom báo cáo audit trong thư mục xuất xưởng"
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        default=False,
        help="Khi đi kèm --clean-temp: giữ nguyên các thư mục checkpoint (.temp_*) có thể tiếp tục, chỉ dọn nháp phụ và gom audit/"
    )
    parser.add_argument(
        "--sub-format",
        choices=["srt", "ass"],
        default="srt",
        help="Định dạng phụ đề xuất trong chế độ --subs-only kèm --no-embed ('srt' mặc định hoặc 'ass')"
    )
    parser.add_argument(
        "--no-organize-audit",
        action="store_true",
        default=False,
        help="Không tự động gom các file báo cáo kiểm định (.audit.json, .audit.md, .cast.yaml) vào thư mục con audit/"
    )
    parser.add_argument(
        "--zero-disk",
        action="store_true",
        default=True,
        help="Kích hoạt pipeline Zero-Disk (Mặc định: Bật): Giải mã NVDEC raw RGB24 pipe -> TensorRT/CUDA FP16 -> p010le pipe -> NVENC, triệt tiêu 100%% file PNG trung gian trên SSD"
    )
    parser.add_argument(
        "--no-zero-disk", "--disk",
        dest="zero_disk",
        action="store_false",
        help="Tắt chế độ Zero-Disk và dùng pipeline truyền thống lưu file PNG trung gian trên SSD/RAM-disk (chế độ dự phòng đối soát/debug)"
    )
    parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        default=False,
        help="Tắt tự động biên dịch/suy luận bằng NVIDIA TensorRT FP16 và dùng thuần PyTorch CUDA FP16"
    )

    return parser
 
 
SUPPORTED_VIDEO_EXTS = {".mkv", ".mp4", ".webm", ".avi", ".mov", ".ts", ".m2ts"}


def _natural_sort_key(filename: str) -> List[object]:
    """Khóa sắp xếp tự nhiên theo số tập (ví dụ: Tập 2 đứng trước Tập 10)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", filename)]


def expand_video_inputs(raw_inputs: Sequence[str]) -> List[str]:
    """
    Mở rộng danh sách đầu vào: nếu phần tử là thư mục hiện hữu, tự động quét các file video gốc
    bên trong theo thứ tự tập tự nhiên (bỏ qua file ẩn/tạm và các bản thành phẩm *_4K / *_4K_animejanai),
    giúp người dùng truyền cả thư mục chứa tên file có ngoặc vuông '[SubsPlease]...' mà không vướng lỗi wildcard của PowerShell.
    """
    expanded: List[str] = []
    seen = set()

    def _append_unique(p_str: str) -> None:
        norm_key = os.path.normcase(os.path.abspath(os.path.expanduser(p_str)))
        if norm_key not in seen:
            seen.add(norm_key)
            expanded.append(p_str)

    for item in raw_inputs:
        if not item:
            continue
        cand_dir = os.path.expanduser(item)
        if os.path.isdir(cand_dir):
            found_in_dir: List[str] = []
            for entry in sorted(os.listdir(cand_dir), key=_natural_sort_key):
                if entry.startswith("."):
                    continue
                full_entry = os.path.join(cand_dir, entry)
                if not os.path.isfile(full_entry):
                    continue
                stem, ext = os.path.splitext(entry)
                if ext.lower() not in SUPPORTED_VIDEO_EXTS:
                    continue
                stem_low = stem.lower()
                if stem_low.endswith("_4k") or stem_low.endswith("_4k_animejanai"):
                    continue
                found_in_dir.append(full_entry)
            if not found_in_dir:
                raise ValueError(
                    f"Thư mục '{item}' không chứa file video gốc hợp lệ nào (hoặc tất cả đã là bản _4K)."
                )
            print(f"📂 [QUÉT THƯ MỤC] Đã nạp {len(found_in_dir)} tập video gốc từ '{item}'.")
            for f_path in found_in_dir:
                _append_unique(f_path)
        else:
            _append_unique(item)
    return expanded


def _resolve_sub_output_dir_and_target_4k(
    output_arg: Optional[str],
    default_dir: str,
    base_vid_name: str,
    explicit_target_4k: Optional[str] = None,
    input_path: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Tuple[Path, str]:
    """Chuẩn hóa thư mục xuất phụ đề và đường dẫn container 4K đích kể cả khi -o là file .mkv cụ thể hoặc _4K_animejanai.mkv."""
    resolved_dir = Path(default_dir)
    candidate_4k = None

    if explicit_target_4k:
        candidate_4k = os.path.abspath(explicit_target_4k)
        parent = os.path.dirname(candidate_4k)
        if parent:
            resolved_dir = Path(parent)

    if output_arg:
        abs_out = os.path.abspath(output_arg)
        if output_arg.lower().endswith((".mkv", ".mp4")) or os.path.isfile(abs_out):
            resolved_dir = Path(os.path.dirname(abs_out) or default_dir)
            if not candidate_4k:
                candidate_4k = abs_out
        else:
            resolved_dir = Path(abs_out)

    if not candidate_4k:
        ref_input = input_path or os.path.join(str(resolved_dir), f"{base_vid_name}.mkv")
        existing = find_existing_4k_output_path(
            ref_input, output_arg=str(resolved_dir), model_id=model_id
        )
        if existing:
            candidate_4k = existing
        else:
            expected = resolve_expected_4k_output_path(
                ref_input, output_arg=str(resolved_dir), model_id=model_id
            )
            stem_dir = os.path.dirname(expected)
            alt_names = [f"{base_vid_name}_4K_animejanai.mkv", f"{base_vid_name}_4K.mkv"]
            found_small = next(
                (os.path.join(stem_dir, n) for n in alt_names if os.path.isfile(os.path.join(stem_dir, n))),
                None,
            )
            candidate_4k = found_small or expected

    return resolved_dir, candidate_4k


def _build_stage_b_sub_cmd(
    args,
    videos: List[str],
    src_vid: str,
    out_vid: str,
    ep_sub_meta: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Build Stage B --subs-only subprocess command honoring explicit CLI flags and checkpoint metadata."""
    meta = ep_sub_meta if isinstance(ep_sub_meta, dict) else {}
    abs_out_vid = os.path.abspath(out_vid)
    sub_out_dir = os.path.dirname(abs_out_vid)
    eff_sound_cues = str(meta.get("sound_cues") or args.sound_cues or "drop")
    eff_subs_quality = str(meta.get("subs_quality") or args.subs_quality or "normal")
    eff_subs_upgrade = bool(meta["subs_upgrade"]) if "subs_upgrade" in meta else bool(args.subs_upgrade)
    eff_no_embed = bool(meta["no_embed"]) if "no_embed" in meta else bool(getattr(args, "no_embed", False))
    eff_prefer_whisper = (
        bool(meta["prefer_whisper"]) if "prefer_whisper" in meta else bool(getattr(args, "prefer_whisper", False))
    )
    eff_glossary = meta["glossary"] if ("glossary" in meta and meta["glossary"]) else getattr(args, "glossary", None)
    sub_cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--subs-only",
        src_vid,
        "-o", sub_out_dir,
        "--target-4k", abs_out_vid,
        "--sound-cues", eff_sound_cues,
        "--subs-quality", eff_subs_quality,
    ]
    if eff_subs_upgrade:
        sub_cmd.append("--subs-upgrade")
    if eff_no_embed:
        sub_cmd.append("--no-embed")
    if not getattr(args, "auto_align_en", True):
        sub_cmd.append("--no-auto-align-en")
    if eff_prefer_whisper:
        sub_cmd.append("--prefer-whisper")
    if eff_glossary:
        sub_cmd.extend(["--glossary", str(eff_glossary)])

    queue_auto_offset = not getattr(args, "no_auto_offset", False)
    manual_offset_applied = False
    if args.sub_offset_ms:
        if len(videos) == 1 or (
            args.sub_offset_target and args.sub_offset_target.lower() in os.path.basename(src_vid).lower()
        ):
            manual_offset_applied = True
            print(f"  • [OFFSET] Áp dụng cờ thủ công --sub-offset-ms={args.sub_offset_ms}ms cho: '{os.path.basename(src_vid)}'")
            sub_cmd.extend(["--sub-offset-ms", str(args.sub_offset_ms)])
        elif not args.sub_offset_target:
            print(
                f"  ⚠️ [QUEUE OFFSET WARNING] Cờ thủ công --sub-offset-ms={args.sub_offset_ms}ms được truyền trong hàng đợi "
                f"nhiều tập nhưng KHÔNG có --sub-offset-target. Để bảo toàn đồng bộ của các tập khác, offset thủ công "
                f"KHÔNG áp dụng toàn cục cho '{os.path.basename(src_vid)}' (dùng --auto-offset tính riêng)."
            )

    if not manual_offset_applied and queue_auto_offset:
        sub_cmd.append("--auto-offset")

    return sub_cmd


_RESERVED_WIN32_CLI_STEMS = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)
_CLI_QUOTE_PAIRS = {'"': '"', "'": "'", "\u201c": "\u201d", "\u201d": "\u201c", "\u2018": "\u2019", "\u2019": "\u2018"}


def _unwrap_cli_path_preserving_trailing_space(raw_val: str) -> str:
    trimmed = raw_val.strip()
    if not trimmed:
        return ""
    if len(trimmed) >= 2 and (
        (trimmed[0] in ('"', "'") and trimmed[-1] == trimmed[0])
        or _CLI_QUOTE_PAIRS.get(trimmed[0]) == trimmed[-1]
    ):
        cur = trimmed
        while len(cur) >= 2 and (
            (cur[0] in ('"', "'") and cur[-1] == cur[0])
            or _CLI_QUOTE_PAIRS.get(cur[0]) == cur[-1]
        ):
            cur = cur[1:-1]
        return cur
    return raw_val.lstrip()


def _validate_cli_output_path(raw_output: Optional[str], multi_queue: bool = False) -> None:
    """Validate -o target drive, Win32 path hygiene, DOS stems, trailing dots/spaces, and queue conflicts."""
    if raw_output is None:
        return
    raw_orig = str(raw_output)
    if "\x00" in raw_orig:
        print("❌ Lỗi: Đường dẫn đích -o chứa ký tự NUL không hợp lệ.")
        sys.exit(1)
    raw_str = _unwrap_cli_path_preserving_trailing_space(raw_orig)
    if not raw_str or not raw_str.strip():
        return
    expanded = os.path.expanduser(raw_str)
    p = Path(expanded)
    if sys.platform == "win32":
        norm_bs = expanded.replace("/", "\\")
        if norm_bs.startswith("\\\\") and not re.match(r"^\\\\\?\\[A-Za-z]:(?:\\|$)", norm_bs):
            print(f"❌ Lỗi: Đường dẫn mạng UNC hoặc thiết bị thô không hợp lệ tại -o: '{raw_str}'")
            sys.exit(1)
        if (p.drive and not p.is_absolute()) or (not p.drive and p.root):
            print(f"❌ Lỗi: Đường dẫn tương đối theo ổ đĩa hoặc thiếu ký tự ổ đĩa tại -o không hợp lệ: '{raw_str}'")
            sys.exit(1)
        drive_norm = p.drive[4:] if p.drive.startswith("\\\\?\\") else p.drive
        if drive_norm and len(drive_norm) == 2 and drive_norm[0].isalpha() and drive_norm[1] == ":" and not os.path.exists(f"{drive_norm}\\"):
            print(f"❌ Lỗi: Ổ đĩa đích '{drive_norm}\\' không tồn tại trên hệ thống.")
            sys.exit(1)
        raw_tail = norm_bs
        if re.match(r"^\\\\\?\\[A-Za-z]:", raw_tail):
            raw_tail = raw_tail[6:]
        elif re.match(r"^[A-Za-z]:", raw_tail):
            raw_tail = raw_tail[2:]
        raw_segments = [seg for seg in raw_tail.split("\\") if seg]
        non_drive_parts = list(p.parts[1:] if (p.drive or p.root) else p.parts)
        forbidden_win = set('<>:"|?*')
        for part in [*raw_segments, *non_drive_parts]:
            if part in (".", ".."):
                continue
            if any(ch in forbidden_win or ord(ch) < 32 for ch in part):
                print(f"❌ Lỗi: Đường dẫn đích -o chứa ký tự cấm của Windows (<>:\"|?*): '{raw_str}'")
                sys.exit(1)
            if part.endswith(".") or part.endswith(" "):
                print(f"❌ Lỗi: Thành phần đường dẫn đích -o không được kết thúc bằng dấu chấm '.' hoặc khoảng trắng: '{raw_str}'")
                sys.exit(1)
            stem0 = part.split(".", 1)[0].strip().lower()
            if stem0 in _RESERVED_WIN32_CLI_STEMS:
                print(f"❌ Lỗi: Đường dẫn đích -o chứa tên thiết bị dành riêng của Windows ({stem0.upper()}): '{raw_str}'")
                sys.exit(1)

    abs_out = os.path.abspath(expanded.strip())
    vid_exts = {".mkv", ".mp4", ".webm", ".avi", ".mov", ".ts", ".m2ts"}
    out_ext = os.path.splitext(abs_out)[1].lower()
    if multi_queue and (os.path.isfile(abs_out) or out_ext in vid_exts):
        print(
            f"❌ Lỗi: Khi chạy hàng đợi nhiều tập (--queue), đường dẫn đích -o phải là thư mục, "
            f"không được trỏ vào tệp video đơn lẻ: '{raw_str}'"
        )
        sys.exit(1)
    if os.path.isfile(abs_out) and out_ext not in vid_exts:
        print(f"❌ Lỗi: Đường dẫn đích -o đang trỏ tới một tệp tin có sẵn không hợp lệ: '{raw_str}'")
        sys.exit(1)
    cur_parent = os.path.dirname(abs_out)
    while cur_parent and cur_parent != os.path.dirname(cur_parent):
        if os.path.isfile(cur_parent):
            print(f"❌ Lỗi: Thành phần đường dẫn cha '{cur_parent}' là tệp tin, không thể tạo thư mục đích.")
            sys.exit(1)
        cur_parent = os.path.dirname(cur_parent)


def parse_args(args=None):
    parsed = build_arg_parser().parse_args(args)
    if getattr(parsed, "subs_quality", None) == "fast":
        parsed.subs_quality = "normal"
    return parsed


def main():
    args = parse_args()

    if getattr(args, "drift_delta", None) is not None:
        if args.dedup_policy is None or args.dedup_policy == "drift" or str(args.dedup_policy).startswith("drift-"):
            args.dedup_policy = f"drift-{args.drift_delta}"

    if getattr(args, "dis_thresh", None) is not None:
        pol_cur = str(args.dedup_policy or "").strip().lower()
        if pol_cur == "dis-auto" or pol_cur.startswith("dis-auto-"):
            args.dedup_policy = f"dis-auto-{args.dis_thresh}"
        elif args.dedup_policy is None or pol_cur == "dis" or pol_cur.startswith("dis-"):
            args.dedup_policy = f"dis-{args.dis_thresh}"

    # Xử lý cờ --clean-temp / --cleanup (Dọn dẹp nhanh thư mục tạm và gom gọn audit)
    if getattr(args, "clean_temp", False):
        _validate_cli_output_path(args.output, multi_queue=True)
        from src.cleanup import _is_symlink_or_reparse
        from src.config import DEFAULT_OUTPUT_DIR
        from src.logger import format_bytes
        if args.output and str(args.output).strip():
            target_clean_dir = os.path.expanduser(str(args.output).strip())
            if _is_symlink_or_reparse(target_clean_dir):
                print(f"❌ Lỗi: Từ chối dọn dẹp qua symlink/reparse point tại -o: '{target_clean_dir}'")
                sys.exit(1)
            if not os.path.isdir(target_clean_dir):
                print(
                    f"ℹ️ Thư mục đích '{target_clean_dir}' chưa tồn tại — bỏ qua dọn dẹp an toàn "
                    f"(không tự động chuyển sang '{DEFAULT_OUTPUT_DIR}')."
                )
                sys.exit(0)
        else:
            target_clean_dir = DEFAULT_OUTPUT_DIR
            if _is_symlink_or_reparse(target_clean_dir):
                print(f"❌ Lỗi: Thư mục xuất mặc định '{target_clean_dir}' là symlink/reparse point.")
                sys.exit(1)
            if not os.path.isdir(target_clean_dir):
                print(f"ℹ️ Thư mục '{target_clean_dir}' chưa tồn tại — không có tệp tạm cần dọn.")
                sys.exit(0)
        include_cps = not bool(getattr(args, "keep_checkpoints", False))
        if getattr(args, "dry_run", False):
            mode_note = "gồm cả checkpoint .temp_*" if include_cps else "bảo toàn checkpoint .temp_*"
            print(f"\n🔍 [DRY RUN] Mô phỏng quét dọn thư mục tạm tại '{target_clean_dir}' ({mode_note}, không xóa file thật).")
            sys.exit(0)
        print(f"\n🧹 Bắt đầu quét và dọn dẹp thư mục: {target_clean_dir}...")
        report = clean_all_temporary_data(target_clean_dir, include_checkpoints=include_cps)
        print(f"✅ Đã dọn dẹp {len(report['cleaned_dirs'])} thư mục/symlink tạm:")
        for cd in report['cleaned_dirs']:
            print(f"   • {cd}")
        if report.get("preserved_checkpoints"):
            print(f"🛡️  Đã bảo toàn {len(report['preserved_checkpoints'])} thư mục checkpoint (.temp_*) có thể tiếp tục:")
            for p_cp in report["preserved_checkpoints"]:
                print(f"   • {p_cp}")
        print(f"📁 Đã gom {len(report['organized_files'])} file báo cáo kỹ thuật vào 'audit/':")
        for of in report['organized_files']:
            print(f"   • {of}")
        print(f"🛡️  Tổng dung lượng SSD đã giải phóng: {format_bytes(report['bytes_freed'])}\n")
        sys.exit(0)

    # Xử lý cờ --night-report (Sinh báo cáo tổng kết phiên đêm, chỉ đọc)
    if args.night_report:
        from src.night_report import generate_night_report_text
        report = generate_night_report_text(args.night_report)
        print(report)
        sys.exit(0)

    # Xử lý cờ --subs-pending (Chạy lại phụ đề cho các tập dở dang trong pending_subs.json)
    if args.subs_pending:
        _validate_cli_output_path(args.output, multi_queue=False)
        pending_items = load_pending_subs()
        if len(pending_items) > 1:
            _validate_cli_output_path(args.output, multi_queue=True)
        if not pending_items:
            print("ℹ️ Không có tập nào trong danh sách phụ đề dở dang (pending_subs.json).")
            sys.exit(0)
        print(f"\n📋 [SUBS PENDING] Tìm thấy {len(pending_items)} tập cần xử lý lại phụ đề:")
        for it in pending_items:
            label = os.path.basename(it.get("video_path") or it.get("base_name") or it.get("ja_path") or "unknown")
            print(f"  • {label} (Tỷ lệ thật: {it.get('real_rate', 0)*100:.1f}%, uncertain: {it.get('uncertain_rate', 0)*100:.1f}%)")

        glossary_arg = getattr(args, "glossary", None)
        glossary_path_obj = Path(os.path.expanduser(str(glossary_arg).strip())) if glossary_arg else None
        is_dry_run = bool(getattr(args, "dry_run", False))
        cli_subs_quality_explicit = any(
            tok == "--subs-quality" or tok.startswith("--subs-quality=")
            for tok in sys.argv[1:]
        )
        cli_sub_offset_explicit = any(
            tok == "--sub-offset-ms" or tok.startswith("--sub-offset-ms=")
            for tok in sys.argv[1:]
        )
        cli_sound_cues_explicit = any(
            tok == "--sound-cues" or tok.startswith("--sound-cues=")
            for tok in sys.argv[1:]
        )
        from subtrans.pipeline import TranslationPipeline
        for it in pending_items:
            vid_p = it.get("video_path")
            ja_s = it.get("ja_path")
            en_s = it.get("en_path")
            base_name_rec = it.get("base_name")
            saved_out_dir = it.get("output_dir")
            eff_prefer_whisper = bool(getattr(args, "prefer_whisper", False) or it.get("prefer_whisper", False))
            eff_subs_upgrade = bool(getattr(args, "subs_upgrade", False) or it.get("subs_upgrade", False))
            eff_no_embed = bool(getattr(args, "no_embed", False) or it.get("no_embed", False))
            eff_sound_cues = (
                args.sound_cues
                if cli_sound_cues_explicit
                else str(it.get("sound_cues") or args.sound_cues or "drop")
            )
            eff_glossary_obj = glossary_path_obj
            if eff_glossary_obj is None and it.get("glossary"):
                cand_g = Path(os.path.expanduser(str(it["glossary"]).strip()))
                if cand_g.exists():
                    eff_glossary_obj = cand_g
            eff_subs_quality = (
                getattr(args, "subs_quality", "normal")
                if cli_subs_quality_explicit
                else str(it.get("subs_quality") or getattr(args, "subs_quality", "normal"))
            )
            eff_sub_offset_ms = (
                args.sub_offset_ms
                if cli_sub_offset_explicit
                else int(it.get("sub_offset_ms") if it.get("sub_offset_ms") is not None else args.sub_offset_ms)
            )
            raw_pw_offsets = None if cli_sub_offset_explicit else it.get("piecewise_ja_offsets")
            eff_piecewise_ja_offsets = (
                raw_pw_offsets
                if (isinstance(raw_pw_offsets, list) and len(raw_pw_offsets) >= 2)
                else None
            )

            if vid_p and os.path.exists(vid_p):
                if not (ja_s and os.path.exists(ja_s)):
                    ja_s, en_extract = find_or_extract_subtitles(
                        vid_p,
                        prefer_whisper_when_no_ja=eff_prefer_whisper,
                        dry_run=is_dry_run,
                    )
                    if not (en_s and os.path.exists(en_s)):
                        en_s = en_extract
                elif not (en_s and os.path.exists(en_s)):
                    _, en_s = find_or_extract_subtitles(
                        vid_p,
                        prefer_whisper_when_no_ja=False,
                        dry_run=is_dry_run,
                    )
                if ja_s == en_s:
                    en_s = None
                base_vid = os.path.splitext(os.path.basename(vid_p))[0]
                default_out_d = saved_out_dir if (saved_out_dir and os.path.isdir(saved_out_dir)) else os.path.dirname(os.path.abspath(vid_p))
            elif ja_s and os.path.exists(ja_s):
                vid_p = None
                if ja_s == en_s:
                    en_s = None
                stem = base_name_rec or os.path.splitext(os.path.basename(ja_s))[0]
                base_vid = stem[:-3] if stem.endswith("_4K") else stem
                default_out_d = saved_out_dir if (saved_out_dir and os.path.isdir(saved_out_dir)) else os.path.dirname(os.path.abspath(ja_s))
            else:
                continue

            if not ja_s or not os.path.exists(ja_s):
                if is_dry_run:
                    print(f"🔍 [DRY RUN] Bỏ qua trích xuất ffmpeg/Whisper cho '{os.path.basename(vid_p or base_vid)}'.")
                continue

            out_d, target_4k = _resolve_sub_output_dir_and_target_4k(
                args.output,
                str(default_out_d),
                base_vid,
                getattr(args, "target_4k", None),
                input_path=vid_p,
                model_id=getattr(args, "model", None),
            )
            out_basename = base_name_rec if base_name_rec else (
                os.path.splitext(os.path.basename(target_4k))[0] if os.path.exists(target_4k) else f"{base_vid}_4K"
            )
            sub_pipe = TranslationPipeline(
                local_only=False,
                output_dir=out_d,
                output_format="srt",
                dry_run=is_dry_run,
                require_approved_cast=False,
                allow_latest_alias=False,
                subs_quality=eff_subs_quality,
                auto_align_whisper_to_en=getattr(args, "auto_align_en", True),
            )
            pending_auto_offset = (
                False
                if eff_piecewise_ja_offsets is not None
                else (not getattr(args, "no_auto_offset", False))
            )
            run_kwargs: Dict[str, Any] = dict(
                ja_path=Path(ja_s),
                en_path=Path(en_s) if (en_s and os.path.exists(en_s)) else None,
                output_basename=out_basename,
                glossary_path=eff_glossary_obj,
                subs_upgrade=eff_subs_upgrade,
                sub_offset_ms=eff_sub_offset_ms,
                auto_offset=pending_auto_offset,
                sound_cues=eff_sound_cues,
                video_path=Path(vid_p) if vid_p else None,
                prefer_whisper=eff_prefer_whisper,
            )
            sig_params = inspect.signature(sub_pipe.run).parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
            if eff_piecewise_ja_offsets is not None and ("piecewise_ja_offsets" in sig_params or has_var_kw):
                run_kwargs["piecewise_ja_offsets"] = eff_piecewise_ja_offsets
            if "no_embed" in sig_params or has_var_kw:
                run_kwargs["no_embed"] = eff_no_embed
            res = sub_pipe.run(**run_kwargs)
            srt_p = res.get('subtitle_path') or res.get('output_path')
            is_partial_pending = bool(not srt_p or str(srt_p).endswith((".partial.srt", ".partial.ass")))
            if vid_p and not is_dry_run:
                cleanup_subtitle_extracts(
                    os.path.dirname(os.path.abspath(vid_p)),
                    base_vid,
                    keep_whisper=is_partial_pending,
                )
            if not is_dry_run and srt_p and os.path.exists(srt_p) and not str(srt_p).endswith((".partial.srt", ".partial.ass")):
                remove_pending_sub(base_name_rec or vid_p or out_basename)
                if os.path.exists(target_4k) and not eff_no_embed:
                    from src.ffmpeg_ops import embed_subtitle_track
                    embed_subtitle_track(target_4k, str(srt_p), original_source_path=vid_p)
        sys.exit(0)

    # Xử lý cờ --subs-only (Chạy riêng phụ đề trong tiến trình con độc lập, hỗ trợ 1 hoặc nhiều tập / thư mục)
    if args.subs_only:
        subs_raw_targets: List[str] = []
        if args.subs_only != "__INPUT__":
            subs_raw_targets.append(args.subs_only)
        if args.queue:
            subs_raw_targets.extend(args.queue)
        if isinstance(args.input, list):
            subs_raw_targets.extend(args.input)
        elif args.input:
            subs_raw_targets.append(args.input)

        try:
            subs_targets = expand_video_inputs(subs_raw_targets) if subs_raw_targets else []
        except ValueError as exp_err:
            print(f"❌ Lỗi đầu vào --subs-only: {exp_err}")
            sys.exit(1)

        if not subs_targets:
            print("❌ Lỗi: Cần cung cấp đường dẫn video khi dùng cờ --subs-only (ví dụ: ./upscale.py --subs-only <tập.mkv>)")
            sys.exit(1)

        _validate_cli_output_path(args.output, multi_queue=len(subs_targets) > 1)

        glossary_arg = getattr(args, "glossary", None)
        glossary_path_obj = Path(os.path.expanduser(str(glossary_arg).strip())) if glossary_arg else None
        from subtrans.pipeline import TranslationPipeline

        use_auto_offset = (
            False
            if getattr(args, "no_auto_offset", False)
            else (bool(args.auto_offset) if getattr(args, "auto_offset", None) is not None else len(subs_targets) > 1)
        )
        multi_mode = len(subs_targets) > 1
        sub_failures = 0

        sub_out_format = str(getattr(args, "sub_format", "srt") or "srt").strip().lower()
        if sub_out_format == "ass" and not getattr(args, "no_embed", False):
            print("❌ Lỗi: Định dạng xuất ASS chỉ hỗ trợ khi bật --no-embed; nhúng trực tiếp vào MKV yêu cầu chuẩn SRT.")
            sys.exit(1)

        is_sub_dry_run = bool(getattr(args, "dry_run", False))
        for idx_sub, target_subs_vid in enumerate(subs_targets, start=1):
            print(f"\n{'='*75}")
            prefix_label = f"[{idx_sub}/{len(subs_targets)}] " if multi_mode else ""
            print(f"📝 [SUBS-ONLY] {prefix_label}BẮT ĐẦU DỊCH PHỤ ĐỀ ĐỘC LẬP CHO: {os.path.basename(target_subs_vid)}")
            print(f"{'='*75}")
            ja_sub, en_sub = find_or_extract_subtitles(
                target_subs_vid,
                prefer_whisper_when_no_ja=getattr(args, "prefer_whisper", False),
                dry_run=is_sub_dry_run,
            )
            if not ja_sub:
                if is_sub_dry_run:
                    print(f"🔍 [DRY RUN] Mô phỏng dịch phụ đề cho '{os.path.basename(target_subs_vid)}' (bỏ qua trích xuất ffmpeg/Whisper & ghi file).")
                    continue
                print(f"❌ Không tìm thấy hoặc không thể trích xuất phụ đề tiếng Nhật cho '{os.path.basename(target_subs_vid)}'.")
                sub_failures += 1
                if not multi_mode:
                    sys.exit(1)
                continue

            if ja_sub == en_sub:
                print(f"   • Nguồn phụ đề: Dùng track tiếng Anh phòng thu làm gốc ({os.path.basename(ja_sub)}) để đảm bảo mốc thời gian frame-perfect 0ms sai lệch.")
                en_sub = None
            else:
                print(f"   • File gốc JA: {ja_sub}")
                if en_sub:
                    print(f"   • File tham chiếu EN: {en_sub}")

            vid_dir = os.path.dirname(os.path.abspath(target_subs_vid))
            base_vid_name = os.path.splitext(os.path.basename(target_subs_vid))[0]
            explicit_t4k = getattr(args, "target_4k", None) if not multi_mode else None
            sub_out_dir, target_4k = _resolve_sub_output_dir_and_target_4k(
                args.output,
                vid_dir,
                base_vid_name,
                explicit_t4k,
                input_path=target_subs_vid,
                model_id=getattr(args, "model", None),
            )
            if not is_sub_dry_run:
                os.makedirs(sub_out_dir, exist_ok=True)
            sub_pipeline = TranslationPipeline(
                local_only=False,
                output_dir=sub_out_dir,
                output_format=sub_out_format,
                dry_run=is_sub_dry_run,
                require_approved_cast=False,
                allow_latest_alias=False,
                subs_quality=getattr(args, "subs_quality", "normal"),
                auto_align_whisper_to_en=getattr(args, "auto_align_en", True),
            )
            out_base = os.path.splitext(os.path.basename(target_4k))[0] if os.path.exists(target_4k) else base_vid_name
            sub_run_kw: Dict[str, Any] = dict(
                ja_path=Path(ja_sub),
                en_path=Path(en_sub) if en_sub else None,
                output_basename=out_base,
                glossary_path=glossary_path_obj,
                subs_upgrade=args.subs_upgrade,
                sub_offset_ms=args.sub_offset_ms,
                auto_offset=use_auto_offset,
                sound_cues=args.sound_cues,
                video_path=Path(target_subs_vid),
                prefer_whisper=bool(getattr(args, "prefer_whisper", False)),
            )
            sub_sig = inspect.signature(sub_pipeline.run).parameters
            if "no_embed" in sub_sig or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sub_sig.values()):
                sub_run_kw["no_embed"] = bool(getattr(args, "no_embed", False))
            res = sub_pipeline.run(**sub_run_kw)
            srt_path = res.get('subtitle_path') or res.get('output_path')
            print(f"\n✅ Hoàn tất dịch phụ đề độc lập cho '{os.path.basename(target_subs_vid)}'. File: {srt_path}")

            if is_sub_dry_run:
                print("🔍 [DRY RUN] Bỏ qua bước nhúng MKV, dọn file và ghi đè phụ đề cạnh video gốc.")
                continue

            # Đồng bộ file SRT và nhúng trực tiếp vào container MKV 4K nếu đã có file 4K, hoặc container MKV đích
            is_partial_sub = True
            if srt_path and os.path.exists(srt_path):
                is_partial_sub = str(srt_path).endswith((".partial.srt", ".partial.ass"))
                if is_partial_sub:
                    print(f"⚠️ [CHƯA ĐỦ ĐIỀU KIỆN NHÚNG] File phụ đề là bản một phần ({os.path.basename(str(srt_path))}). Không nhúng vào MKV.")
                else:
                    from src.ffmpeg_ops import embed_subtitle_track
                    target_mkv = target_4k if os.path.exists(target_4k) else (target_subs_vid if target_subs_vid.lower().endswith(".mkv") else None)
                    if target_mkv and os.path.exists(target_mkv):
                        if not getattr(args, "no_embed", False):
                            print(f"🎬 Đang nhúng track phụ đề tiếng Việt vào container MKV: {os.path.basename(target_mkv)}...")
                            ok = embed_subtitle_track(target_mkv, str(srt_path), original_source_path=target_subs_vid)
                            if ok:
                                print("✅ Đã nhúng phụ đề tiếng Việt (mặc định) trực tiếp vào container MKV!")
                            else:
                                print("⚠️ Không thể nhúng phụ đề vào container (audit không đạt hoặc lỗi), file phụ đề ngoài vẫn sẵn sàng.")
                        else:
                            print(f"ℹ️ Cờ --no-embed được bật: Giữ nguyên container MKV, chỉ dùng file phụ đề ngoài '{os.path.basename(str(srt_path))}'.")

                    # Chỉ đồng bộ file .vi.srt cạnh video gốc khi bản dịch đạt chuẩn (KHÔNG copy đè .partial.srt)
                    raw_side_srt = os.path.join(vid_dir, f"{base_vid_name}.vi.srt")
                    if srt_path and os.path.exists(srt_path) and os.path.abspath(srt_path) != os.path.abspath(raw_side_srt):
                        import shutil
                        shutil.copy2(str(srt_path), raw_side_srt)
                        print(f"📄 Đã đồng bộ file phụ đề cạnh video gốc cho trình phát: {os.path.basename(raw_side_srt)}")
            cleanup_subtitle_extracts(vid_dir, base_vid_name, keep_whisper=is_partial_sub)
            if not getattr(args, "no_organize_audit", False):
                organize_output_artifacts(str(sub_out_dir), base_name=base_vid_name)

        sys.exit(1 if sub_failures else 0)

    # Kiểm tra cờ liệt kê mô hình (--list-models)
    if args.list_models:
        print(list_models_formatted())
        sys.exit(0)

    # Xác định danh sách video cần xử lý (hỗ trợ cả file lẻ lẫn thư mục chứa nhiều tập)
    raw_videos = []
    if args.queue:
        raw_videos.extend(args.queue)
    if isinstance(args.input, list):
        raw_videos.extend(args.input)
    elif args.input:
        raw_videos.append(args.input)

    if not raw_videos:
        print("❌ Lỗi: Vui lòng cung cấp ít nhất một file video đầu vào (hoặc thư mục / dùng cờ --queue).")
        sys.exit(1)

    try:
        videos = expand_video_inputs(raw_videos)
    except ValueError as e_dir:
        print(f"❌ Lỗi: {e_dir}")
        sys.exit(1)

    _validate_cli_output_path(args.output, multi_queue=len(videos) > 1)

    video_overhead = 420.0  # TensorRT FP16 Zero-Disk mặc định
    has_trt = False
    use_zero_disk = bool(getattr(args, "zero_disk", True))
    raw_cli_argv = sys.argv[1:]
    explicit_segment_arg = any(
        tok in ("-s", "--segment") or tok.startswith("--segment=") or (tok.startswith("-s") and tok[2:].isdigit())
        for tok in raw_cli_argv
    )
    eff_segment_time = (
        DEFAULT_ZERO_DISK_SEGMENT_TIME
        if (not explicit_segment_arg and use_zero_disk and args.engine == "cuda" and int(args.segment) == DEFAULT_SEGMENT_TIME)
        else int(args.segment)
    )
    eff_cli_dedup_policy = "off" if args.no_dedup else args.dedup_policy
    if args.engine == "cuda":
        if not getattr(args, "no_tensorrt", False) and args.precision == "fp16":
            import importlib.util
            has_trt = importlib.util.find_spec("tensorrt") is not None
        if has_trt and use_zero_disk:
            video_overhead = 420.0
        elif has_trt:
            video_overhead = 660.0
        else:
            video_overhead = 540.0 if use_zero_disk else 780.0
    first_model_id = _resolve_model_id_safe(videos[0] if videos else None, args.model)
    fps_median, fps_p5 = _resolve_model_fps_pair(
        first_model_id, args.engine, has_trt, use_zero_disk
    )
    measured_model_fps: Dict[str, Tuple[float, float]] = {}
    subtitles_overhead = 300.0 if args.subs_after_upscale else 0.0

    if args.benchmark_mode:
        os.environ["BENCHMARK_MODE"] = "1"
        print("\n⚡ [BENCHMARK MODE] Đã kích hoạt chế độ benchmark thuần túy: Bộ lập lịch và preflight nền từ chối chạy để bảo toàn môi trường đo đạc!")

    eff_sched_grain = str(getattr(args, "grain_restore", DEFAULT_GRAIN_RESTORE) or "auto").strip().lower()
    sched_grain_kw: Dict[str, Any] = {}
    if eff_sched_grain != "auto":
        sched_grain_kw["grain_restore"] = eff_sched_grain
    if getattr(args, "force", False):
        sched_grain_kw["force"] = True

    # 1. Chế độ Dry Run (--dry-run): Preflight toàn bộ hàng đợi và in bảng quyết định, không chạy GPU
    if args.dry_run:
        from src.scheduler import compute_overnight_schedule, format_schedule_decision_table
        print(f"\n🔍 [DRY RUN] BẮT ĐẦU KIỂM TRA LẬP LỊCH CHO {len(videos)} TẬP TRONG HÀNG ĐỢI...")
        sched = compute_overnight_schedule(
            queue=videos,
            deadline_str=args.deadline,
            fps_median=fps_median,
            fps_p5=fps_p5,
            video_overhead_seconds=video_overhead,
            subtitles_overhead_seconds=subtitles_overhead,
            allow_partial=args.allow_partial,
            output_dir=args.output,
            segment_seconds=eff_segment_time,
            preflight_now=False,
            subs_quality=args.subs_quality,
            dedup_policy=eff_cli_dedup_policy,
            **sched_grain_kw,
        )
        print(format_schedule_decision_table(sched))
        sys.exit(0)

    # Dọn cờ dừng êm mồ côi từ phiên bị tắt đột ngột trước đó nếu không có GPU worker nào đang giữ khóa
    if is_graceful_pause_requested() and is_gpu_lock_held_by_other() is None:
        clear_graceful_pause()
        print("🧹 [CLEANUP] Đã dọn cờ dừng êm mồ côi (.graceful_pause.flag) trước phiên chạy mới.")

    # 2. Phiên chạy thật: Lập lịch và in bảng quyết định ngay đầu phiên (nếu không ở benchmark mode)
    bg_preflight = None
    if not args.benchmark_mode and not (args.preview and len(videos) == 1 and not args.deadline):
        from src.scheduler import compute_overnight_schedule, format_schedule_decision_table, BackgroundQueuePreflightManager
        sched = compute_overnight_schedule(
            queue=videos,
            deadline_str=args.deadline,
            fps_median=fps_median,
            fps_p5=fps_p5,
            video_overhead_seconds=video_overhead,
            subtitles_overhead_seconds=subtitles_overhead,
            allow_partial=args.allow_partial,
            output_dir=args.output,
            segment_seconds=eff_segment_time,
            preflight_now=False,
            subs_quality=args.subs_quality,
            dedup_policy=eff_cli_dedup_policy,
            **sched_grain_kw,
        )
        print(format_schedule_decision_table(sched))

        # Kích hoạt preflight nền cho toàn bộ các tập tiếp theo trong hàng đợi (nice -n 10)
        if len(videos) > 1 and not args.no_dedup:
            def _on_bg_plan_ready(scanned_vid, pdata):
                try:
                    updated_sched = compute_overnight_schedule(
                        queue=videos,
                        deadline_str=args.deadline,
                        fps_median=fps_median,
                        fps_p5=fps_p5,
                        video_overhead_seconds=video_overhead,
                        subtitles_overhead_seconds=subtitles_overhead,
                        allow_partial=args.allow_partial,
                        output_dir=args.output,
                        segment_seconds=eff_segment_time,
                        preflight_now=False,
                        subs_quality=args.subs_quality,
                        dedup_policy=eff_cli_dedup_policy,
                        **sched_grain_kw,
                    )
                    print(f"\n🔄 [BẢNG LẬP LỊCH CẬP NHẬT] Đã nạp số đo n_unique thật cho '{os.path.basename(scanned_vid)}':")
                    print(format_schedule_decision_table(updated_sched))
                except Exception:
                    pass

            bg_preflight = BackgroundQueuePreflightManager(
                queue=videos,
                output_dir=args.output,
                segment_seconds=eff_segment_time,
                skip_first=True,
                benchmark_mode=args.benchmark_mode,
                on_plan_ready=_on_bg_plan_ready,
                dedup_policy=eff_cli_dedup_policy,
                **({"grain_restore": eff_sched_grain} if eff_sched_grain != "auto" else {}),
            )
            bg_preflight.start()

    completed_videos = []
    skipped_existing_videos: set[str] = set()
    ep_stage_b_meta: Dict[str, Dict[str, Any]] = {}
    async_sub_procs = []          # [(src_vid, out_vid, Popen, launch_ts)]
    failed_resumable_videos = {}  # vid -> (up_segs_dir, chunk_idx)
    unfinished_videos = {}        # vid -> reason
    queue_halted = False

    raw_cli_tokens = list(sys.argv[1:])

    def _has_cli_opt(*names: str) -> bool:
        return any(
            tok in names or any(tok.startswith(f"{nm}=") for nm in names)
            for tok in raw_cli_tokens
        )

    cli_explicit_map = {
        "model_requested": _has_cli_opt("-m", "--model"),
        "subs_after_upscale": _has_cli_opt("--no-subs", "--no-subs-after-upscale", "--subs-after-upscale"),
        "subs_quality": _has_cli_opt("--subs-quality"),
        "prefer_whisper": _has_cli_opt("--prefer-whisper"),
        "async_subs": _has_cli_opt("--async-subs"),
        "glossary": _has_cli_opt("-g", "--glossary"),
        "subs_upgrade": _has_cli_opt("--subs-upgrade"),
        "no_embed": _has_cli_opt("--no-embed"),
        "sound_cues": _has_cli_opt("--sound-cues"),
    }

    def _read_existing_checkpoint_manifest(src_v: str) -> Dict[str, Any]:
        if args.force or args.no_resume:
            return {}
        from src.config import DEFAULT_OUTPUT_DIR
        v_stem = os.path.splitext(os.path.basename(src_v))[0].rstrip(". ")
        out_root = DEFAULT_OUTPUT_DIR
        if args.output:
            abs_o = os.path.abspath(os.path.expanduser(str(args.output).strip()))
            out_root = os.path.dirname(abs_o) if abs_o.lower().endswith(tuple(SUPPORTED_VIDEO_EXTS)) else abs_o
        for cand_m in (
            os.path.join(out_root, f".temp_{v_stem}", "manifest.json"),
            os.path.join(DEFAULT_OUTPUT_DIR, f".temp_{v_stem}", "manifest.json"),
            os.path.join(os.path.dirname(os.path.abspath(src_v)), f".temp_{v_stem}", "manifest.json"),
        ):
            if os.path.isfile(cand_m):
                try:
                    with open(cand_m, "r", encoding="utf-8") as mf:
                        data_m = json.load(mf)
                    if isinstance(data_m, dict):
                        return data_m
                except Exception:
                    pass
        return {}

    def _current_queue_stats(cur_idx_1based: int, in_stage_b: bool = False) -> Dict[str, Any]:
        newly_done = sum(1 for cv, _ in completed_videos if cv not in skipped_existing_videos)
        skipped_cnt = sum(1 for cv, _ in completed_videos if cv in skipped_existing_videos)
        failed_cnt = len(unfinished_videos)
        return {
            "total_items": len(videos),
            "current_index": max(1, min(len(videos), int(cur_idx_1based))),
            "completed_items": newly_done + skipped_cnt,
            "rendered_items": newly_done,
            "skipped_items": skipped_cnt,
            "failed_items": failed_cnt,
            "stage_b": bool(in_stage_b),
        }

    def _write_live_status_update(
        stage_val: str,
        detail_val: str,
        ref_vid: Optional[str] = None,
        cur_idx_1based: int = 1,
        in_stage_b: bool = False,
        create_if_missing: bool = False,
    ) -> None:
        live_p = get_live_status_path()
        cur_job_env = os.environ.get("UPSCALER_WEB_JOB_ID", "")
        cur_pid = os.getpid()
        cur_dict: Dict[str, Any] = {}
        if os.path.isfile(live_p):
            try:
                with open(live_p, "r", encoding="utf-8") as rf:
                    loaded = json.load(rf)
                if isinstance(loaded, dict):
                    same_lineage = (
                        (cur_job_env and str(loaded.get("job_id") or "") == cur_job_env)
                        or (not cur_job_env and loaded.get("pid") == cur_pid)
                    )
                    if same_lineage or not create_if_missing:
                        cur_dict = loaded
            except Exception:
                cur_dict = {}
        elif not create_if_missing:
            return
        if not cur_dict and not create_if_missing:
            return
        os.makedirs(os.path.dirname(live_p), exist_ok=True)
        v_ref = ref_vid or (videos[0] if videos else "")
        cur_dict.setdefault("video_name", os.path.basename(v_ref) if v_ref else "")
        cur_dict.setdefault("input_path", str(v_ref) if v_ref else "")
        cur_dict.setdefault("model_id", str(args.model or ""))
        cur_dict.setdefault("zero_disk", bool(getattr(args, "zero_disk", True)))
        cur_dict.setdefault("total_frames", 0)
        cur_dict.setdefault("unique_frames", 0)
        cur_dict.setdefault("total_chunks", 0)
        cur_dict.setdefault("completed_chunks", 0)
        cur_dict.setdefault("chunks", [])
        cur_dict.setdefault("resumed_chunks", [])
        cur_dict.setdefault("active_chunk", None)
        cur_dict.setdefault("steady_fps", 0.0)
        cur_dict.setdefault("latest_fps", 0.0)
        cur_dict.setdefault("effective_fps", 0.0)
        cur_dict.setdefault("eta_seconds", None)
        if ref_vid:
            cur_dict["video_name"] = os.path.basename(ref_vid)
            cur_dict["input_path"] = str(ref_vid)
        cur_dict["pid"] = cur_pid
        cur_dict["ppid"] = os.getppid()
        if cur_job_env:
            cur_dict["job_id"] = cur_job_env
        cur_dict["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        cur_dict["stage"] = stage_val
        cur_dict["stage_detail"] = detail_val
        if len(videos) > 1:
            cur_dict["queue_stats"] = _current_queue_stats(cur_idx_1based, in_stage_b=in_stage_b)
        tmp_lp = f"{live_p}.{cur_pid}.tmp"
        try:
            with open(tmp_lp, "w", encoding="utf-8") as wf:
                json.dump(cur_dict, wf, ensure_ascii=False, indent=2)
            os.replace(tmp_lp, live_p)
        except Exception:
            if os.path.exists(tmp_lp):
                try:
                    os.remove(tmp_lp)
                except OSError:
                    pass

    can_overlap_async_subs = bool(
        getattr(args, "async_subs", False)
        and args.subs_after_upscale
        and not getattr(args, "prefer_whisper", False)
        and len(videos) > 1
    )
    session_deadline_dt: Optional[datetime] = None
    if args.deadline:
        try:
            _dh, _dm = map(int, args.deadline.split(":"))
            _now = datetime.now()
            _dt = _now.replace(hour=_dh, minute=_dm, second=0, microsecond=0)
            if _dt <= _now:
                _dt += timedelta(days=1)
            session_deadline_dt = _dt
        except Exception:
            session_deadline_dt = None

    try:
        for idx, vid in enumerate(videos):
            print(f"\n{'='*75}")
            print(f"▶ [{idx+1}/{len(videos)}] Bắt đầu xử lý: {os.path.basename(vid)}")
            print(f"{'='*75}")

            if len(videos) > 1 and idx > 0:
                _write_live_status_update(
                    "initializing",
                    f"Đang chuẩn bị tập {idx+1}/{len(videos)}: {os.path.basename(vid)}...",
                    ref_vid=vid,
                    cur_idx_1based=idx + 1,
                )

            if bg_preflight is not None:
                bg_preflight.mark_main_started(vid)
                bg_preflight.wait_for_video(vid, timeout=600.0)

            already_done_path = (
                None
                if (args.force or args.preview)
                else find_existing_4k_output_path(vid, output_arg=args.output, model_id=args.model)
            )

            ep_model_id = _resolve_model_id_safe(vid, args.model)
            ep_base_med, ep_base_p5 = _resolve_model_fps_pair(
                ep_model_id, args.engine, has_trt, use_zero_disk
            )
            fps_median, fps_p5 = measured_model_fps.get(ep_model_id, (ep_base_med, ep_base_p5))

            # Kiểm tra deadline nếu có (A6), bỏ qua nếu tập đã hoàn tất từ trước
            deadline_dt = None
            if args.deadline and not already_done_path:
                subs_reserved_sec = compute_pending_subs_reserved_seconds(
                    completed_videos,
                    async_sub_procs,
                    subtitles_overhead,
                    args.subs_after_upscale,
                )
                can_finish, is_partial, deadline_dt = check_deadline_and_eta(
                    vid,
                    args.deadline,
                    fps_median=fps_median,
                    fps_p5=fps_p5,
                    video_overhead=video_overhead,
                    subtitles_overhead=subtitles_overhead,
                    eta_margin_pct=0.07,
                    allow_partial=args.allow_partial,
                    output_dir=args.output,
                    reserved_seconds=subs_reserved_sec,
                    dedup_policy=eff_cli_dedup_policy,
                    base_deadline_dt=session_deadline_dt,
                    **sched_grain_kw,
                )
                if not can_finish:
                    print(f"⏹️ Dừng hàng đợi tại tập {idx+1}/{len(videos)} do chạm ngưỡng hạn chót {args.deadline}.")
                    unfinished_videos[vid] = f"Bỏ qua do chạm hạn chót {args.deadline}"
                    queue_halted = True
                    break

            try:
                prior_cp_manifest = _read_existing_checkpoint_manifest(vid)
                full_cli_meta = {
                    "model_requested": str(args.model or "").strip(),
                    "subs_after_upscale": bool(getattr(args, "subs_after_upscale", True)),
                    "subs_quality": str(getattr(args, "subs_quality", "normal") or "normal"),
                    "prefer_whisper": bool(getattr(args, "prefer_whisper", False)),
                    "async_subs": bool(getattr(args, "async_subs", False)),
                    "glossary": str(getattr(args, "glossary", "") or "").strip(),
                    "subs_upgrade": bool(getattr(args, "subs_upgrade", False)),
                    "no_embed": bool(getattr(args, "no_embed", False)),
                    "sound_cues": str(getattr(args, "sound_cues", "drop") or "drop"),
                }
                if prior_cp_manifest:
                    eff_manifest_extra = {
                        mk: mv for mk, mv in full_cli_meta.items() if cli_explicit_map.get(mk, False)
                    }
                    merged_ep_sub = {}
                    for mk, mv in full_cli_meta.items():
                        if cli_explicit_map.get(mk, False) or mk not in prior_cp_manifest:
                            merged_ep_sub[mk] = mv
                        else:
                            merged_ep_sub[mk] = prior_cp_manifest[mk]
                    ep_stage_b_meta[vid] = merged_ep_sub
                else:
                    eff_manifest_extra = full_cli_meta
                    ep_stage_b_meta[vid] = dict(full_cli_meta)

                eff_cli_grain = str(getattr(args, "grain_restore", DEFAULT_GRAIN_RESTORE) or "auto").strip().lower()
                grain_pipe_kw = {"grain_restore": eff_cli_grain} if eff_cli_grain != "auto" else {}
                extra_pipe_kw = _filter_pipeline_kwargs(
                    run_upscale_pipeline,
                    {
                        **(
                            {"preview_start_sec_override": getattr(args, "preview_start", None)}
                            if getattr(args, "preview_start", None) is not None
                            else {}
                        ),
                        "manifest_extra_meta": eff_manifest_extra,
                    },
                )
                out_file = run_upscale_pipeline(
                    input_path=vid,
                    output_path=args.output,
                    model=args.model,
                    engine=args.engine,
                    precision=args.precision,
                    force=args.force,
                    bitrate=args.bitrate,
                    segment_time=eff_segment_time,
                    fmt=args.format,
                    use_ramdisk=not args.no_ramdisk,
                    resume=not args.no_resume,
                    anti_sleep=not args.no_anti_sleep,
                    preview_seconds=args.preview,
                    dedup=not args.no_dedup,
                    allow_partial=args.allow_partial,
                    deadline_dt=deadline_dt,
                    keep_temp=args.keep_temp,
                    dedup_policy=args.dedup_policy,
                    batch_size=args.batch_size,
                    zero_disk=getattr(args, "zero_disk", True),
                    use_tensorrt=not getattr(args, "no_tensorrt", False),
                    nvenc_multipass=getattr(args, "nvenc_multipass", HEVC_MULTIPASS_POLICY),
                    segment_time_explicit=explicit_segment_arg,
                    **grain_pipe_kw,
                    **extra_pipe_kw,
                )
                # (b) Tập dang dở (do allow-partial) hoặc không hợp lệ bị loại khỏi mux/phụ đề:
                if out_file and os.path.isfile(out_file) and out_file.endswith((".mkv", ".mp4")):
                    completed_videos.append((vid, out_file))
                    if already_done_path:
                        skipped_existing_videos.add(vid)
                    if vid in unfinished_videos:
                        del unfinished_videos[vid]
                    if vid in failed_resumable_videos:
                        del failed_resumable_videos[vid]
                    if len(videos) > 1 and idx < len(videos) - 1:
                        _write_live_status_update(
                            "initializing",
                            f"Đã xong {idx+1}/{len(videos)} tập — đang chuyển sang tập kế tiếp...",
                            ref_vid=videos[idx + 1],
                            cur_idx_1based=idx + 2,
                        )
                    ep_meta_cur = ep_stage_b_meta.get(vid, {})
                    ep_wants_subs = bool(ep_meta_cur.get("subs_after_upscale", args.subs_after_upscale))
                    ep_async_ok = bool(
                        (ep_meta_cur.get("async_subs", getattr(args, "async_subs", False)))
                        and ep_wants_subs
                        and not ep_meta_cur.get("prefer_whisper", getattr(args, "prefer_whisper", False))
                        and len(videos) > 1
                    )
                    if (can_overlap_async_subs or ep_async_ok) and ep_wants_subs and idx < len(videos) - 1:
                        _sb_params = inspect.signature(_build_stage_b_sub_cmd).parameters
                        bg_sub_cmd = (
                            _build_stage_b_sub_cmd(args, videos, vid, out_file, ep_sub_meta=ep_meta_cur)
                            if "ep_sub_meta" in _sb_params
                            else _build_stage_b_sub_cmd(args, videos, vid, out_file)
                        )
                        print(f"\n⚡ [ASYNC-SUBS] Khởi chạy nền dịch phụ đề Cloud Gemini cho '{os.path.basename(vid)}' song song tập kế tiếp...")
                        try:
                            bg_sub_env = {**os.environ, "UPSCALER_FORCE_WHISPER_CPU": "1"}
                            bg_proc = subprocess.Popen(bg_sub_cmd, env=bg_sub_env)
                            async_sub_procs.append((vid, out_file, bg_proc, datetime.now()))
                        except Exception as bg_err:
                            print(f"⚠️ [ASYNC-SUBS] Không thể chạy nền cho '{os.path.basename(vid)}' ({bg_err}), sẽ chạy ở Stage B cuối hàng đợi.")
                else:
                    if was_graceful_paused():
                        unfinished_videos[vid] = "Dừng êm theo yêu cầu người dùng (bảo toàn checkpoint)"
                        print(f"\n⏸️ [GRACEFUL PAUSE] Tập '{os.path.basename(vid)}' đã dừng êm và giữ nguyên checkpoint. Tạm ngưng hàng đợi.")
                        queue_halted = True
                        break
                    unfinished_videos[vid] = "Dừng sạch ranh giới phân đoạn theo --allow-partial"
                    print(f"\n⏸️ [PARTIAL] Tập '{os.path.basename(vid)}' chưa hoàn tất 4K, tự động loại khỏi danh sách dịch phụ đề (Stage B) và không mux thành phẩm.")
                    queue_halted = True
                    break

                # Cập nhật fps từ audit.json nếu có
                audit_path = os.path.splitext(out_file)[0] + ".audit.json" if out_file else None
                if audit_path and not os.path.exists(audit_path):
                    alt_audit = os.path.join(os.path.dirname(out_file), "audit", os.path.basename(audit_path))
                    if os.path.exists(alt_audit):
                        audit_path = alt_audit
                if audit_path and os.path.exists(audit_path):
                    try:
                        with open(audit_path, "r", encoding="utf-8") as f:
                            adata = json.load(f)
                        measured_fps = adata.get("ai_steady_fps")
                        if measured_fps and measured_fps > 0:
                            p5_ratio = (ep_base_p5 / ep_base_med) if ep_base_med > 0 else (8.5 / 10.5)
                            fps_median = float(measured_fps)
                            fps_p5 = round(fps_median * p5_ratio, 2)
                            measured_model_fps[ep_model_id] = (fps_median, fps_p5)
                    except Exception:
                        pass

            except FailedResumableError as fre:
                print(f"\n⚠️ [FAILED-RESUMABLE] Tập '{os.path.basename(vid)}' dừng do ENOSPC tại chunk {fre.chunk_idx + 1}. Đánh dấu để thử lại cuối hàng đợi.")
                failed_resumable_videos[vid] = (fre.up_segs_dir, fre.chunk_idx)
                unfinished_videos[vid] = f"Dừng ở chunk {fre.chunk_idx + 1} do ENOSPC (failed-resumable)"
                if len(videos) > 1:
                    print("Chuyển sang tập tiếp theo trong hàng đợi...\n")
                    continue
                else:
                    pass
            except KeyboardInterrupt:
                print("\n\n⚠️ Đã tạm dừng quá trình theo yêu cầu (Ctrl+C).")
                print("🛡️  RAM Disk và các file đệm đã được tự động giải phóng an toàn.")
                print("💡 Lần sau khi chạy lại cùng video này, hệ thống sẽ tự động tiếp tục từ phân đoạn đang dở!\n")
                sys.exit(130)
            except Exception as e:
                print(f"\n❌ Đã xảy ra lỗi khi xử lý tập '{vid}': {e}\n")
                unfinished_videos[vid] = f"Lỗi: {e}"
                if len(videos) > 1:
                    print("Chuyển sang tập tiếp theo trong hàng đợi...\n")
                    continue
                else:
                    sys.exit(1)

        # TỰ ĐỘNG THỬ LẠI CÁC TẬP FAILED-RESUMABLE MỘT LẦN Ở CUỐI HÀNG ĐỢI
        if failed_resumable_videos and not queue_halted:
            print(f"\n{'='*75}")
            print(f"🔄 [RETRY PASS] TỰ ĐỘNG THỬ LẠI {len(failed_resumable_videos)} TẬP FAILED-RESUMABLE MỘT LẦN")
            print(f"{'='*75}")
            for vid, (up_dir, last_c_idx) in list(failed_resumable_videos.items()):
                print(f"▶ Thử lại: {os.path.basename(vid)} (tiếp tục từ phân đoạn {last_c_idx + 1})...")
                try:
                    out_file = run_upscale_pipeline(
                        input_path=vid,
                        output_path=args.output,
                        model=args.model,
                        engine=args.engine,
                        precision=args.precision,
                        batch_size=args.batch_size,
                        force=args.force,
                        bitrate=args.bitrate,
                        segment_time=eff_segment_time,
                        fmt=args.format,
                        use_ramdisk=not args.no_ramdisk,
                        resume=not args.no_resume,
                        anti_sleep=not args.no_anti_sleep,
                        preview_seconds=args.preview,
                        dedup=not args.no_dedup,
                        allow_partial=args.allow_partial,
                        deadline_dt=None,
                        keep_temp=args.keep_temp,
                        dedup_policy=args.dedup_policy,
                        zero_disk=getattr(args, "zero_disk", True),
                        use_tensorrt=not getattr(args, "no_tensorrt", False),
                        nvenc_multipass=getattr(args, "nvenc_multipass", HEVC_MULTIPASS_POLICY),
                        segment_time_explicit=explicit_segment_arg,
                        **(
                            {"grain_restore": str(getattr(args, "grain_restore", "auto")).strip().lower()}
                            if str(getattr(args, "grain_restore", "auto") or "auto").strip().lower() != "auto"
                            else {}
                        ),
                    )
                    if out_file and os.path.isfile(out_file) and out_file.endswith((".mkv", ".mp4")):
                        completed_videos.append((vid, out_file))
                        del failed_resumable_videos[vid]
                        if vid in unfinished_videos:
                            del unfinished_videos[vid]
                        print(f"✅ Thử lại thành công: '{os.path.basename(vid)}' đã hoàn tất 4K!")
                except Exception as retry_err:
                    print(f"❌ Thử lại vẫn thất bại cho '{os.path.basename(vid)}': {retry_err}")
                    unfinished_videos[vid] = f"Thử lại thất bại (chunk {last_c_idx + 1}): {retry_err}"
    finally:
        if bg_preflight is not None:
            should_clean_bg = bool(
                not getattr(args, "keep_temp", False)
                and (queue_halted or was_graceful_paused() or len(completed_videos) < len(videos))
            )
            try:
                bg_preflight.stop(cleanup_unclaimed=should_clean_bg)
            except TypeError:
                bg_preflight.stop()

    # STAGE B: DỊCH PHỤ ĐỀ (CHỈ CHẠY CHO CÁC TẬP ĐÃ HOÀN TẤT)
    # CHẠY TRONG TIẾN TRÌNH CON ĐỘC LẬP (SUBPROCESS), KHÔNG CHẶN HAY HỎNG VIDEO 4K
    stage_b_candidates = [
        (src_v, out_v)
        for src_v, out_v in completed_videos
        if bool((ep_stage_b_meta.get(src_v) or {}).get("subs_after_upscale", args.subs_after_upscale))
    ]
    if (stage_b_candidates or async_sub_procs) and completed_videos:
        print(f"\n{'='*75}")
        print("📝 [STAGE B] HOÀN TẤT & ĐỒNG BỘ TIẾN TRÌNH DỊCH PHỤ ĐỀ ĐỘC LẬP")
        print(f"{'='*75}")
        _write_live_status_update(
            "subtrans",
            "Đang hoàn tất dịch & nhúng phụ đề (Stage B)...",
            cur_idx_1based=len(completed_videos),
            in_stage_b=True,
        )

        sub_timeout = 1560  # Watchdog 25 phút + 60s buffer
        launched_async_srcs = set()

        for src_vid, out_vid, bg_proc, _launch_ts in async_sub_procs:
            launched_async_srcs.add(src_vid)
            print(f"\n⏳ [ASYNC-SUBS] Đang chờ tiến trình nền phụ đề của '{os.path.basename(src_vid)}' hoàn tất...")
            try:
                ret = bg_proc.wait(timeout=sub_timeout)
                if ret == 0:
                    print(f"✅ [ASYNC-SUBS] Phụ đề hoàn tất thành công cho '{os.path.basename(src_vid)}'.")
                else:
                    print(f"⚠️ [ASYNC-SUBS] Tiến trình phụ đề thoát với mã lỗi {ret}. Video 4K vẫn được bảo toàn nguyên vẹn!")
            except subprocess.TimeoutExpired:
                bg_proc.kill()
                print(f"⏱️ [ASYNC-SUBS] Quá thời gian ({sub_timeout}s), đã ngắt an toàn tiến trình nền của '{os.path.basename(src_vid)}'!")
            except Exception as bg_wait_err:
                print(f"⚠️ [ASYNC-SUBS] Lỗi khi đồng bộ tiến trình phụ đề nền: {bg_wait_err}")

        for src_vid, out_vid in stage_b_candidates:
            if src_vid in launched_async_srcs:
                continue
            _sb_params = inspect.signature(_build_stage_b_sub_cmd).parameters
            sub_cmd = (
                _build_stage_b_sub_cmd(args, videos, src_vid, out_vid, ep_sub_meta=ep_stage_b_meta.get(src_vid))
                if "ep_sub_meta" in _sb_params
                else _build_stage_b_sub_cmd(args, videos, src_vid, out_vid)
            )
            print(f"\n🚀 Khởi chạy tiến trình con dịch phụ đề cho tập '{os.path.basename(src_vid)}'...")
            try:
                proc = subprocess.run(sub_cmd, timeout=sub_timeout, check=False)
                if proc.returncode == 0:
                    print(f"✅ Tiến trình con phụ đề hoàn tất thành công cho '{os.path.basename(src_vid)}'.")
                else:
                    print(f"⚠️ Tiến trình con phụ đề thoát với mã lỗi {proc.returncode}. Video 4K và audit.json vẫn được bảo toàn nguyên vẹn!")
            except subprocess.TimeoutExpired:
                print(f"⏱️ Tiến trình con phụ đề vượt quá thời gian tối đa ({sub_timeout}s) và đã bị ngắt an toàn. Video 4K không bị ảnh hưởng!")
            except Exception as se:
                print(f"⚠️ Lỗi khởi chạy tiến trình con phụ đề: {se}")

        if not unfinished_videos and len(completed_videos) == len(videos):
            _write_live_status_update(
                "completed",
                "Hoàn tất dịch & nhúng phụ đề Stage B",
                cur_idx_1based=len(videos),
                in_stage_b=False,
            )

    # DỌN DẸP PHỤ ĐỀ TẠM & GOM GỌN CÁC FILE KỸ THUẬT (AUDIT/CAST) SAU KHI HOÀN TẤT
    if not getattr(args, "no_organize_audit", False) and completed_videos:
        pending_bases = {str(it.get("base_name") or "") for it in load_pending_subs() if isinstance(it, dict)}
        for src_vid, out_vid in completed_videos:
            sub_out_dir = os.path.dirname(os.path.abspath(out_vid))
            base_vid_name = os.path.splitext(os.path.basename(src_vid))[0]
            out_stem = os.path.splitext(os.path.basename(out_vid))[0]
            has_partial = (
                os.path.exists(os.path.join(sub_out_dir, f"{out_stem}.vi.partial.srt"))
                or out_stem in pending_bases
                or base_vid_name in pending_bases
            )
            cleanup_subtitle_extracts(
                os.path.dirname(os.path.abspath(src_vid)),
                base_vid_name,
                keep_whisper=has_partial,
            )
            organize_output_artifacts(sub_out_dir, base_name=base_vid_name, out_stem=out_stem)

    # KIỂM ĐỊNH TẤT CẢ CÁC TẬP: NẾU CÒN TẬP CHƯA XONG THÌ THOÁT MÃ ≠ 0 VÀ IN BẢNG
    unfinished_list = [v for v in videos if v not in [cv[0] for cv in completed_videos]]
    if unfinished_list:
        print(f"\n{'='*75}")
        print("🚨 BẢNG TỔNG KẾT HÀNG ĐỢI: CÁC TẬP CHƯA XONG")
        print(f"{'='*75}")
        print(f"{'STT':<4} | {'Tập Video':<45} | {'Trạng Thái / Ghi Chú'}")
        print("-" * 75)
        for u_idx, u_vid in enumerate(unfinished_list):
            reason = unfinished_videos.get(u_vid, "Chưa hoàn tất")
            print(f"[{u_idx+1:02d}] | {os.path.basename(u_vid)[:45]:<45} | {reason}")
        print(f"{'='*75}")
        print(f"⚠️ Có {len(unfinished_list)}/{len(videos)} tập chưa hoàn tất. Hàng đợi kết thúc với trạng thái CHƯA XONG.\n")
        is_paused_exit = was_graceful_paused() or any(
            "allow-partial" in unfinished_videos.get(u, "").lower()
            or "dừng êm" in unfinished_videos.get(u, "").lower()
            or "hạn chót" in unfinished_videos.get(u, "").lower()
            for u in unfinished_list
        )
        target_stage = "paused" if is_paused_exit else "error"
        first_unfin = unfinished_list[0]
        unfin_idx_1b = (videos.index(first_unfin) + 1) if first_unfin in videos else 1
        _write_live_status_update(
            target_stage,
            (
                f"Hàng đợi kết thúc với {len(unfinished_list)}/{len(videos)} tập chưa hoàn tất "
                f"({unfinished_videos.get(first_unfin, 'Chưa hoàn tất')})."
            ),
            ref_vid=first_unfin,
            cur_idx_1based=unfin_idx_1b,
            in_stage_b=False,
            create_if_missing=True,
        )
        sys.exit(1)

    print("\n🎉 Toàn bộ công việc trong hàng đợi đã hoàn tất thành công!")
    sys.exit(0)


if __name__ == "__main__":
    main()

