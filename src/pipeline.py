# -*- coding: utf-8 -*-
"""
src/pipeline.py
Bộ điều phối chính (Orchestrator) cho toàn bộ quy trình Upscale 4K:
- Kiến trúc phân đoạn cuốn chiếu (Rolling Segments)
- Bộ nhớ đệm RAM Disk ảo (Zero NVMe SSD wear)
- Cơ chế chống ngủ máy tự động (Windows SetThreadExecutionState)
- Kiểm tra an toàn trước khi chạy (Pre-flight Health Checks: Pin, Ổ đĩa)
- Tự động tiếp tục từ phân đoạn dở (Checkpoint Resume)
- Thông báo hệ thống khi hoàn tất
- Thu thập viễn trắc và lưu log có cấu trúc tại logs/
"""

import os
import sys
import time
import json
import shutil
import subprocess
from datetime import datetime
from glob import glob
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.config import (
    validate_environment,
    DEFAULT_MODEL,
    DEFAULT_ENGINE,
    DEFAULT_PRECISION,
    DEFAULT_SCALE,
    DEFAULT_BITRATE,
    DEFAULT_SEGMENT_TIME,
    DEFAULT_FRAME_FORMAT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAMDISK_SIZE_MB,
    DEFAULT_PNG_UNIT_SIZE_MB,
    DEVICE_NAME,
    HEVC_ENCODER,
    acquire_gpu_lock,
    release_gpu_lock,
)
from src.ramdisk import RAMDisk


class FailedResumableError(RuntimeError):
    """Lỗi dừng chunk do ENOSPC sau số lần thử lại; tập có thể resume an toàn từ checkpoint."""
    def __init__(self, message: str, chunk_idx: int, up_segs_dir: str):
        super().__init__(message)
        self.chunk_idx = chunk_idx
        self.up_segs_dir = up_segs_dir

from src.ffmpeg_ops import (
    probe_video,
    is_nvdec_supported,
    split_video_segments,
    get_segments_frame_counts,
    verify_segment_resume_frames,
    extract_frames,
    extract_frames_raw_rgb24,
    encode_chunk_hevc,
    concat_segments,
    mux_final_container,
    mux_segments_directly,
)
from src.dedup import (
    load_or_create_plan,
    prune_redundant_frames,
    generate_concat_demuxer_script,
    encode_chunk_reexpanded,
    resolve_dedup_policy
)
from src.ai_engine import upscale_frames
from src.models_resolver import resolve_model_for_video
from src.cuda_engine import upscale_frames_with_fallback
from src.logger import (
    format_time,
    format_bytes,
    render_progress,
    RunLogger,
    get_swap_usage,
    get_thermal_state
)
from src.preflight import (
    AntiSleepContext,
    run_preflight_checks,
    send_system_notification
)
from src.cleanup import cleanup_checkpoint_dir, organize_output_artifacts


def run_upscale_pipeline(
    input_path: str,
    output_path: Optional[str] = None,
    model: Optional[str] = None,
    engine: str = DEFAULT_ENGINE,
    precision: str = DEFAULT_PRECISION,
    force: bool = False,
    bitrate: str = DEFAULT_BITRATE,
    segment_time: int = DEFAULT_SEGMENT_TIME,
    fmt: str = DEFAULT_FRAME_FORMAT,
    use_ramdisk: bool = True,
    resume: bool = True,
    anti_sleep: bool = True,

    preview_seconds: Optional[int] = None,
    dedup: bool = True,
    allow_partial: bool = False,
    deadline_dt: Optional[datetime] = None,
    keep_temp: bool = False,
    dedup_policy: Optional[str] = None,
    batch_size: Optional[int] = None,
    zero_disk: bool = False,
    use_tensorrt: bool = True,
) -> str:
    """Điều phối toàn bộ quy trình Upscale 4K trên NVIDIA CUDA"""
    validate_environment(engine=engine)

    # Chuẩn hóa đường dẫn đầu vào
    input_path = os.path.abspath(os.path.expanduser(input_path))
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file video nguồn: '{input_path}'")
        sys.exit(1)

    # Phân giải mô hình SR theo thứ tự ưu tiên: CLI > shows.yaml > default (D2)
    model_info = resolve_model_for_video(input_path, model)

    # 1. Thu thập thông số kỹ thuật chi tiết
    meta = probe_video(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Xử lý chế độ Preview nhanh nếu người dùng yêu cầu
    is_preview = bool(preview_seconds and preview_seconds > 0)
    preview_temp_input = None

    if is_preview:
        max_dur = max(1, int(round(meta["duration"]))) if meta.get("duration") else int(preview_seconds)
        preview_dur = min(int(preview_seconds), max_dur)
        if preview_dur >= max_dur:
            start_pt = 0
        else:
            start_pt = max(0, min(int(meta["duration"] * 0.3), max_dur - preview_dur))
        preview_temp_input = os.path.join(DEFAULT_OUTPUT_DIR, f".preview_clip_{base_name}.mkv")
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        print(f"\n🔍 [PREVIEW] Đang trích xuất đoạn xem trước {preview_dur}s (từ {format_time(start_pt)})...")
        cut_cmd = [
            "ffmpeg", "-y", "-ss", str(start_pt), "-i", input_path,
            "-t", str(preview_dur), "-c", "copy", "-map", "0",
            preview_temp_input
        ]
        subprocess.run(cut_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        input_path = preview_temp_input
        meta = probe_video(input_path)
        base_name = f"{base_name}_preview_{preview_dur}s"

    # Khởi tạo logger cho phiên chạy
    logger = RunLogger(base_name)

    # Xác định hậu tố tên file đầu ra theo model (D7)
    out_suffix = "_4K_animejanai.mkv" if str(model_info.get("id", "")).startswith("animejanai-") else "_4K.mkv"

    # Xác định đường dẫn file thành phẩm
    if not output_path:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_name}{out_suffix}")
    else:
        output_path = os.path.abspath(os.path.expanduser(output_path))
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, f"{base_name}{out_suffix}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Kiểm tra nếu file 4K thành phẩm đã tồn tại (không ghi đè trừ khi có cờ --force)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024 * 1024 and not force:
        logger.info(
            f"⏩ [BỎ QUA - ĐÃ CÓ KẾT QUẢ] File 4K đã hoàn tất trước đó: '{os.path.basename(output_path)}' "
            f"({format_bytes(os.path.getsize(output_path))}). Bỏ qua không xử lý lại (dùng --force để ghi đè).",
            to_console=True
        )
        return output_path

    # 2. Thực thi kiểm tra an toàn trước khi chạy (Pre-flight checks)
    run_preflight_checks(
        output_dir=os.path.dirname(output_path),
        duration_seconds=meta["duration"],
        bitrate=bitrate,
        logger=logger
    )

    # Kích hoạt AntiSleep để chống Sleep máy trong suốt phiên chạy
    with AntiSleepContext(enabled=anti_sleep):
        if anti_sleep:
            logger.info("Cơ chế chống ngủ máy (Windows SetThreadExecutionState) đã được kích hoạt thành công.")

        target_w = meta["width"] * DEFAULT_SCALE
        target_h = meta["height"] * DEFAULT_SCALE

        # Thư mục lưu phân đoạn mã hóa trên SSD (phục vụ Checkpoint Resume)
        work_dir = os.path.join(os.path.dirname(output_path), f".temp_{base_name}")
        segs_dir = os.path.join(work_dir, "raw_segs")
        up_segs_dir = os.path.join(work_dir, "upscaled_segs")
        os.makedirs(segs_dir, exist_ok=True)
        os.makedirs(up_segs_dir, exist_ok=True)

        # Kiểm tra tính đồng nhất của model qua manifest.json (D7)
        model_sha256 = str(model_info.get("sha256") or "")
        manifest_path = os.path.join(work_dir, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                prev_model_id = manifest_data.get("model_id")
                prev_sha256 = manifest_data.get("sha256") or ""
                if prev_model_id and (prev_model_id != model_info["id"] or prev_sha256 != model_sha256):
                    if not force:
                        raise ValueError(
                            f"❌ Xung đột mô hình: Thư mục tạm '{work_dir}' chứa phân đoạn của mô hình '{prev_model_id}' "
                            f"(sha256: {prev_sha256[:8] if prev_sha256 else 'N/A'}), nhưng phiên hiện tại yêu cầu '{model_info['id']}' "
                            f"(sha256: {model_sha256[:8] if model_sha256 else 'N/A'}).\n"
                            f"Từ chối trộn lẫn mô hình trong cùng một tập (D7)! "
                            f"Sử dụng cờ --force để xóa sạch phân đoạn cũ và chạy lại từ đầu."
                        )
                    else:
                        logger.warn("⚠️ Cờ --force được kích hoạt: Xóa phân đoạn cũ khác mô hình và chạy lại...", to_console=True)
                        shutil.rmtree(segs_dir, ignore_errors=True)
                        shutil.rmtree(up_segs_dir, ignore_errors=True)
                        plan_file_del = os.path.join(work_dir, "plan.json")
                        if os.path.exists(plan_file_del):
                            os.remove(plan_file_del)
                        os.makedirs(segs_dir, exist_ok=True)
                        os.makedirs(up_segs_dir, exist_ok=True)
            except ValueError:
                raise
            except Exception as e:
                logger.warn(f"Không thể đọc manifest.json cũ: {e}")

        # Ghi nhận manifest mới
        manifest_data = {
            "model_id": model_info["id"],
            "model_name": model_info["name"],
            "sha256": model_sha256,
            "engine": engine,
            "precision": precision,
            "resolution_source": model_info.get("resolution_source", "")
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        # Kích thước RAM Disk (ước lượng dựa theo độ dài phân đoạn và định dạng)
        multiplier = 18.0 if fmt == "png" else 4.5
        ramdisk_size = max(DEFAULT_RAMDISK_SIZE_MB, int(segment_time * 24 * multiplier / 1024) + 1024)

        # Khởi tạo RAM Disk Context
        with RAMDisk(size_mb=ramdisk_size, enabled=use_ramdisk) as cache_root:
            frames_in_dir = os.path.join(cache_root, "chunk_in")
            frames_out_dir = os.path.join(cache_root, "chunk_out")

            storage_desc = (
                f"RAM Disk {ramdisk_size}MB ảo (Zero SSD Wear 🛡️)"
                if getattr(cache_root, "is_ramdisk", False) or "RAM" in cache_root
                else "Thư mục tạm NVMe SSD"
            )

            banner_info = {
                "filename": os.path.basename(input_path),
                "size_bytes": meta["size_bytes"],
                "width": meta["width"],
                "height": meta["height"],
                "target_w": target_w,
                "target_h": target_h,
                "duration": meta["duration"],
                "fps": meta["fps"],
                "model": f"{model_info['name']} ({model_info['id']}) [{engine.upper()} {precision.upper()}] - {model_info.get('resolution_source', '')}",
                "frame_format": fmt,
                "storage_info": storage_desc,
                "bitrate": bitrate,
                "audio_count": meta["audio_count"],
                "sub_count": meta["subtitle_count"],
                "attach_count": meta["attachment_count"],
                "chap_count": meta["chapter_count"],
                "output_path": output_path
            }
            logger.print_banner(banner_info)

            start_total_time = time.time()
            stage_timings = {}
            swap_before = get_swap_usage()
            logger.info(f"📊 Trạng thái Bộ nhớ/Swap ban đầu: {swap_before}")

            # -----------------------------------------------------------------
            # BƯỚC 1: PHÂN ĐOẠN VIDEO THÔ & KẾ HOẠCH DEDUPLICATION
            # -----------------------------------------------------------------
            existing_segs = sorted([
                os.path.join(segs_dir, f)
                for f in os.listdir(segs_dir)
                if f.startswith("seg_") and f.endswith(".mp4")
            ]) if os.path.exists(segs_dir) else []
            if not existing_segs or not resume:
                logger.info(f"Đang phân đoạn video thành các block {segment_time}s (Stream Copy)...", to_console=True)
                t_split_start = time.time()
                existing_segs = split_video_segments(input_path, segs_dir, segment_time)
                stage_timings["Segment Split (Stream Copy)"] = time.time() - t_split_start

            total_segs = len(existing_segs)
            # Đo chính xác số frame thực tế của từng phân đoạn (khớp bit-exact theo Keyframe GOP của nguồn)
            seg_frame_counts = get_segments_frame_counts(existing_segs)

            dedup_plan = None
            if dedup:
                effective_dedup_policy = resolve_dedup_policy(input_path, dedup_policy)
                plan_file = os.path.join(work_dir, "plan.json")
                logger.info(f"Đang kiểm tra / phân tích kế hoạch Deduplication toàn tập ({effective_dedup_policy})...", to_console=True)
                t_pref_start = time.time()
                dedup_plan = load_or_create_plan(
                    input_path=input_path,
                    plan_path=plan_file,
                    segment_seconds=segment_time,
                    fps_num=meta["fps_num"],
                    fps_den=meta["fps_den"],
                    resume=resume,
                    chunk_frame_counts=seg_frame_counts,
                    policy=effective_dedup_policy
                )
                if dedup_plan:
                    dedup_plan["model_id"] = model_info["id"]
                    dedup_plan["model_sha256"] = model_sha256
                    dedup_plan["engine"] = engine
                    dedup_plan["precision"] = precision
                    try:
                        with open(plan_file, "w", encoding="utf-8") as f:
                            json.dump(dedup_plan, f, indent=2)
                    except Exception:
                        pass
                stage_timings["Preflight Plan (mpdecimate)"] = time.time() - t_pref_start
                logger.info(
                    f"⚡ Kế hoạch Deduplication: Giữ {dedup_plan['total_kept_frames']}/{dedup_plan['total_frames']} unique frames "
                    f"({dedup_plan['dedup_ratio_percent']}% trùng lặp được loại bỏ - Tối ưu ~50% thời gian GPU)",
                    to_console=True
                )

            logger.info(f"Đã chuẩn bị {total_segs} phân đoạn. Bắt đầu quá trình Upscale GPU ({DEVICE_NAME})...", to_console=True)
            print()

            # -----------------------------------------------------------------
            # BƯỚC 2: XỬ LÝ UPSCALE & MÃ HÓA PHÂN ĐOẠN CUỐN CHIẾU
            # -----------------------------------------------------------------
            t_chunks_start = time.time()
            total_frames_processed = 0
            total_ai_frames = 0
            total_ai_time = 0.0
            total_enc_time = 0.0
            total_ext_time = 0.0

            # Lọc các phân đoạn cần xử lý (bỏ qua những đoạn đã xong nếu bật resume và kiểm tra frame khớp)
            chunks_to_process = []
            for idx, seg_file in enumerate(existing_segs):
                seg_out_file = os.path.join(up_segs_dir, f"up_{idx:04d}.mp4")
                expected_frames = (
                    dedup_plan["chunks"][idx]["total_frames"]
                    if (dedup and dedup_plan and idx < len(dedup_plan.get("chunks", [])))
                    else (seg_frame_counts[idx] if idx < len(seg_frame_counts) else None)
                )
                if resume and os.path.exists(seg_out_file) and os.path.getsize(seg_out_file) > 10000:
                    if verify_segment_resume_frames(seg_out_file, expected_frames):
                        logger.info(f"Phân đoạn [{idx+1:02d}/{total_segs:02d}] đã hoàn thành từ trước ({expected_frames} frames), tiếp tục...")
                        print(f"⏭️  Phân đoạn [{idx+1:02d}/{total_segs:02d}] đã hoàn thành từ trước ({expected_frames} frames), tiếp tục...")
                        total_frames_processed += expected_frames if expected_frames is not None else 192
                        continue
                    else:
                        logger.warning(
                            f"⚠️ Phân đoạn [{idx+1:02d}/{total_segs:02d}] '{seg_out_file}' không khớp số frame kỳ vọng "
                            f"({expected_frames} frames) hoặc container bị lỗi. Tự động loại bỏ checkpoint hỏng và xử lý lại...",
                            to_console=True
                        )
                        try:
                            os.remove(seg_out_file)
                        except OSError:
                            pass
                chunks_to_process.append((idx, seg_file, seg_out_file))

            if not chunks_to_process:
                logger.info("Tất cả phân đoạn đã hoàn thành từ trước!", to_console=True)
            else:
                # Định nghĩa các worker function cho từng stage của kiến trúc Overlapped Double-Buffering
                def worker_extract(item):
                    """Decode Worker: Ở chế độ --zero-disk giải mã thẳng RGB24 vào RAM qua NVDEC pipe (0 byte PNG trên SSD)."""
                    c_idx, c_seg_file, _ = item
                    c_in_dir = os.path.join(cache_root, f"chunk_{c_idx:04d}_in")
                    c_plan = (
                        dedup_plan["chunks"][c_idx]
                        if (dedup and dedup_plan and c_idx < len(dedup_plan.get("chunks", [])))
                        else None
                    )
                    kept_indices = c_plan["kept_indices"] if c_plan else None
                    chunk_total_frames = (
                        c_plan["total_frames"]
                        if c_plan
                        else (seg_frame_counts[c_idx] if c_idx < len(seg_frame_counts) else None)
                    )
                    fps_val = float(meta.get("fps_num", 24000)) / max(1.0, float(meta.get("fps_den", 1001)))
                    chunk_start_sec = (
                        sum(seg_frame_counts[:c_idx]) / fps_val
                        if (fps_val > 0 and c_idx <= len(seg_frame_counts))
                        else float(c_idx * segment_time)
                    )

                    if zero_disk and engine == "cuda":
                        t0 = time.time()
                        safe_hwaccel = is_nvdec_supported(
                            codec=meta.get("codec", ""),
                            pix_fmt=meta.get("pix_fmt", ""),
                            profile=meta.get("profile", ""),
                        )
                        raw_rgb = extract_frames_raw_rgb24(
                            c_seg_file,
                            width=int(meta.get("width", 1920)),
                            height=int(meta.get("height", 1080)),
                            kept_indices=kept_indices,
                            hwaccel=safe_hwaccel,
                            total_frames=chunk_total_frames,
                            source_video_path=input_path,
                            chunk_start_sec=chunk_start_sec,
                            source_codec=meta.get("codec"),
                        )
                        t_ext = time.time() - t0
                        f_count = len(raw_rgb)
                        distinct_frames = raw_rgb
                        frame_to_distinct = list(range(f_count))
                        if dedup and f_count > 1:
                            from src.dedup import deduplicate_cyclic_frames
                            distinct_frames, frame_to_distinct = deduplicate_cyclic_frames(raw_rgb)
                        exp_frames = chunk_total_frames if chunk_total_frames is not None else f_count
                        return {
                            "idx": c_idx,
                            "seg_file": c_seg_file,
                            "in_dir": c_in_dir,
                            "raw_rgb_frames": raw_rgb,
                            "distinct_frames": distinct_frames,
                            "frame_to_distinct": frame_to_distinct,
                            "frame_count": f_count,
                            "ai_frames": len(distinct_frames),
                            "expected_frames": exp_frames,
                            "ext_time": t_ext,
                            "chunk_plan": c_plan,
                        }

                    os.makedirs(c_in_dir, exist_ok=True)
                    # Đảm bảo RAM Disk có tối thiểu không gian an toàn cho 1080p frames (~350MB) trước khi extract
                    t_wait_ext = time.time()
                    while (shutil.disk_usage(cache_root).free / (1024 * 1024)) < 450.0:
                        if time.time() - t_wait_ext > 120.0:
                            break
                        time.sleep(0.2)

                    t0 = time.time()
                    f_count = extract_frames(
                        c_seg_file,
                        c_in_dir,
                        fmt=fmt,
                        kept_indices=kept_indices,
                        source_video_path=input_path,
                        chunk_start_sec=chunk_start_sec,
                        total_frames=chunk_total_frames,
                        width=int(meta.get("width", 1920)),
                        height=int(meta.get("height", 1080)),
                    )
                    t_ext = time.time() - t0

                    ai_count = f_count
                    exp_frames = chunk_total_frames if chunk_total_frames is not None else f_count

                    return {
                        "idx": c_idx,
                        "seg_file": c_seg_file,
                        "in_dir": c_in_dir,
                        "raw_rgb_frames": None,
                        "frame_count": f_count,
                        "ai_frames": ai_count,
                        "expected_frames": exp_frames,
                        "ext_time": t_ext,
                        "chunk_plan": c_plan
                    }

                def worker_encode(item, ai_res):
                    """Hardware Encode Worker: Nén HEVC 10-bit và dọn sạch thư mục tạm ngay lập tức."""
                    c_idx, _, c_seg_out = item["target"]
                    c_in_dir = item["extracted"]["in_dir"]
                    c_out_dir = item["out_dir"]
                    c_plan = item["extracted"]["chunk_plan"]
                    exp_frames = item["extracted"]["expected_frames"]

                    concat_script = os.path.join(cache_root, f"concat_{c_idx:04d}.txt") if c_plan else None
                    t0 = time.time()
                    try:
                        if c_plan:
                            generate_concat_demuxer_script(
                                runs=c_plan["runs"],
                                frames_out_dir=c_out_dir,
                                concat_script_path=concat_script,
                                fps_num=meta["fps_num"],
                                fps_den=meta["fps_den"],
                                fmt=fmt
                            )
                            encode_chunk_reexpanded(
                                concat_script_path=concat_script,
                                output_path=c_seg_out,
                                expected_frames=exp_frames,
                                fps_num=meta["fps_num"],
                                fps_den=meta["fps_den"],
                                bitrate=bitrate
                            )
                        else:
                            encode_chunk_hevc(
                                frames_dir=c_out_dir,
                                fps=meta["fps"],
                                output_path=c_seg_out,
                                bitrate=bitrate,
                                fmt=fmt
                            )
                        return time.time() - t0
                    finally:
                        if concat_script and os.path.exists(concat_script):
                            try:
                                os.remove(concat_script)
                            except OSError:
                                pass
                        shutil.rmtree(c_in_dir, ignore_errors=True)
                        shutil.rmtree(c_out_dir, ignore_errors=True)

                def worker_finalize_zero_disk(proc_obj, tmp_p: str, final_p: str, seg_no: int) -> float:
                    """Hoàn tất đóng gói MP4 từ NVENC pipe (flush 32-frame lookahead + atomic rename) ở luồng nền."""
                    t0_w = time.time()
                    err_txt = ""
                    try:
                        if getattr(proc_obj, "stderr", None) is not None:
                            raw_err = proc_obj.stderr.read()
                            if isinstance(raw_err, bytes):
                                err_txt = raw_err.decode("utf-8", errors="replace").strip()
                            elif isinstance(raw_err, str):
                                err_txt = raw_err.strip()
                            proc_obj.stderr.close()
                    except Exception:
                        pass
                    rc = proc_obj.wait()
                    if rc != 0 or not os.path.exists(tmp_p) or os.path.getsize(tmp_p) == 0:
                        if os.path.exists(tmp_p):
                            try:
                                os.remove(tmp_p)
                            except OSError:
                                pass
                        err_suffix = f" | stderr: {err_txt}" if err_txt else ""
                        raise RuntimeError(
                            f"FFmpeg NVENC Zero-Disk pipe thất bại tại phân đoạn {seg_no+1:02d} (exit code {rc}){err_suffix}."
                        )
                    os.replace(tmp_p, final_p)
                    return max(0.001, time.time() - t0_w)

                acquire_gpu_lock()
                future_extract = None
                prev_c_idx = 0
                try:
                    # Thực thi gối đầu song song (Double-Buffering Bounded Queue <= 2 chunks trên RAM Disk)
                    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="DecodeWorker") as dec_exec, \
                         ThreadPoolExecutor(max_workers=1, thread_name_prefix="EncodeWorker") as enc_exec:

                        # Bắt đầu trích xuất chunk đầu tiên trên luồng DecodeWorker
                        future_extract = dec_exec.submit(worker_extract, chunks_to_process[0])

                        # Khởi tạo song song GPU Engine + TensorRT + Pinned Ring + Triton Warmup ngay trong lúc chờ giải mã Chunk 0
                        if zero_disk and engine == "cuda" and not os.environ.get("PYTEST_CURRENT_TEST"):
                            try:
                                import torch
                                from src.cuda_engine import get_cuda_inference_engine
                                from src.build_trt_engine import get_hybrid_engine_or_wrapper
                                from src.trt_zero_copy_upscaler import DoubleBufferedPipeUpscaler

                                pre_h = int(meta.get("height", 1080))
                                pre_w = int(meta.get("width", 1920))
                                pre_cuda = get_cuda_inference_engine(
                                    model_info,
                                    precision=precision,
                                    batch_size=batch_size,
                                    use_tensorrt=use_tensorrt,
                                )
                                pre_bs = batch_size or pre_cuda.batch_size or 8
                                pre_key = (model_info["id"], pre_h, pre_w, int(pre_bs), bool(use_tensorrt), precision)
                                if (
                                    getattr(pre_cuda, "_zero_disk_upscaler", None) is None
                                    or getattr(pre_cuda, "_zero_disk_key", None) != pre_key
                                ):
                                    pre_wrapped = get_hybrid_engine_or_wrapper(
                                        pre_cuda.model,
                                        max_batch=pre_bs,
                                        force_pytorch=(not use_tensorrt) or (precision != "fp16"),
                                        model_id=model_info["id"],
                                        input_hw=(pre_h, pre_w),
                                        scale=2.0,
                                    )
                                    pre_cuda._zero_disk_wrapper = pre_wrapped
                                    pre_cuda._zero_disk_key = pre_key
                                    pre_cuda._zero_disk_upscaler = DoubleBufferedPipeUpscaler(
                                        pre_wrapped,
                                        batch_size=pre_bs,
                                        device=pre_cuda.device,
                                        input_hw=(pre_h, pre_w),
                                        scale=2,
                                    )
                                    if pre_cuda.device.type == "cuda" and torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                            except (FileNotFoundError, ValueError):
                                raise
                            except RuntimeError as pre_rt_err:
                                if "SHA-256" in str(pre_rt_err):
                                    raise
                            except Exception:
                                pass
                        future_encode = None
                        last_encode_item = None
                        last_ai_result = None
                        last_seg_start = 0.0
                        ai_fps_history = []
                        initial_baseline_fps = None
                        throttling_warned = False
                        partial_paused = False
                        chunk_png_unit_sizes = []
                        sanity_checks_history = []
                        last_pipeline_micro_timings = None
                        accumulated_micro_timings = {
                            "num_batches": 0,
                            "total_h2d_ms": 0.0,
                            "total_trt_ms": 0.0,
                            "total_triton_ms": 0.0,
                            "total_d2h_ms": 0.0,
                        }

                        for i, target_chunk in enumerate(chunks_to_process):
                            c_idx, c_seg_file, c_seg_out = target_chunk
                            prev_c_idx = c_idx

                            # Chờ giải mã CPU chunk hiện tại sẵn sàng (chỉ lấy 1 lần cho chunk i)
                            if future_extract is not None:
                                extracted = future_extract.result()
                                future_extract = None
                            else:
                                extracted = worker_extract(target_chunk)

                            # Gối đầu thực sự (True 3-Stage Overlap):
                            # Khởi chạy giải mã CPU/NVDEC cho chunk i+1 NGAY TRƯỚC KHI GPU suy luận chunk i!
                            if (
                                i + 1 < len(chunks_to_process)
                                and not (allow_partial and deadline_dt and datetime.now() >= deadline_dt)
                            ):
                                can_prefetch_early = (zero_disk and engine == "cuda") or (
                                    shutil.disk_usage(cache_root).free / (1024 * 1024) >= 1600.0
                                )
                                if can_prefetch_early:
                                    future_extract = dec_exec.submit(worker_extract, chunks_to_process[i + 1])

                            c_in_dir = extracted["in_dir"]
                            c_out_dir = os.path.join(cache_root, f"chunk_{c_idx:04d}_out")
                            if not (zero_disk and engine == "cuda"):
                                os.makedirs(c_out_dir, exist_ok=True)

                            chunk_attempt = 0
                            max_chunk_retries = 2
                            zero_disk_pending_finalize = None

                            while True:
                                seg_start_time = time.time()
                                n_unique = extracted["ai_frames"]

                                # 2.2 Cổng Backpressure: need = số frame × max(kích thước PNG chunk gần nhất, p95 tích lũy, sàn DEFAULT_PNG_UNIT_SIZE_MB) × 1.2
                                floor_png_mb = DEFAULT_PNG_UNIT_SIZE_MB
                                recent_png_mb = chunk_png_unit_sizes[-1] if chunk_png_unit_sizes else floor_png_mb
                                p95_png_mb = float(np.percentile(chunk_png_unit_sizes, 95)) if chunk_png_unit_sizes else floor_png_mb
                                unit_size_mb = max(recent_png_mb, p95_png_mb, floor_png_mb) if fmt.lower() == "png" else 1.2
                                need_mb = n_unique * unit_size_mb * 1.2
                                free_mb = shutil.disk_usage(cache_root).free / (1024 * 1024)

                                # Nếu RAM Disk không đủ chỗ an toàn và đang có chunk trước encode ở luồng nền:
                                # Đồng bộ đợi Media Engine encode xong và dọn sạch frame chunk trước
                                if future_encode is not None and free_mb < need_mb:
                                    logger.info(
                                        f"⚠️ [BACKPRESSURE] RAM Disk trống {free_mb:.0f} MB < nhu cầu an toàn {need_mb:.0f} MB "
                                        f"({n_unique} frames x {unit_size_mb:.1f}MB x 1.2). Chờ Media Engine giải phóng chunk trước..."
                                    )
                                    prev_enc_elapsed = future_encode.result()
                                    total_enc_time += prev_enc_elapsed
                                    prev_c_idx = last_encode_item["target"][0]
                                    prev_exp_frames = last_encode_item["extracted"]["expected_frames"]
                                    prev_seg_elapsed = time.time() - last_seg_start
                                    prev_seg_fps = (prev_exp_frames / prev_seg_elapsed) if prev_seg_elapsed > 0 else 0.0
                                    render_progress(
                                        prev_c_idx + 1, total_segs,
                                        prefix=f"✨ Đoạn {prev_c_idx+1:02d}/{total_segs:02d}",
                                        suffix=f"[{prev_seg_fps:.2f} fps | Đã giải phóng RAM Disk]"
                                    )
                                    logger.log_chunk_done(
                                        chunk_idx=prev_c_idx,
                                        total_chunks=total_segs,
                                        frames=prev_exp_frames,
                                        ai_time=last_ai_result["elapsed_seconds"],
                                        ai_fps=last_ai_result["fps"],
                                        enc_time=prev_enc_elapsed,
                                        total_chunk_time=prev_seg_elapsed,
                                        output_seg=last_encode_item["target"][2],
                                        t0_seconds=last_ai_result.get("t0_seconds"),
                                        steady_fps=last_ai_result.get("steady_fps"),
                                        rss_mb=last_ai_result.get("peak_rss_mb")
                                    )
                                    future_encode = None
                                    free_mb = shutil.disk_usage(cache_root).free / (1024 * 1024)

                                # Nếu RAM Disk còn dưới 800 MB và chunk trước đang encode ở luồng nền:
                                # Chờ Media Engine giải phóng trước khi bắt đầu infer
                                if future_encode is not None and not future_encode.done():
                                    cur_free = shutil.disk_usage(cache_root).free / (1024 * 1024)
                                    if cur_free < 800.0:
                                        prev_enc_elapsed = future_encode.result()
                                        total_enc_time += prev_enc_elapsed
                                        future_encode = None
                                        free_mb = shutil.disk_usage(cache_root).free / (1024 * 1024)

                                # Kiểm tra giới hạn cứng ENOSPC trước khi infer
                                # Ở chế độ Zero-Disk, không xuất PNG 4K trung gian nên chỉ cần biên an toàn nhỏ cho MP4 nén
                                min_needed_mb = max(256.0, n_unique * 0.5) if (zero_disk and engine == "cuda") else (n_unique * unit_size_mb)
                                if free_mb < min_needed_mb:
                                    raise OSError(
                                        28,
                                        f"RAM Disk không đủ không gian cho chunk {c_idx}: Trống {free_mb:.0f} MB, "
                                        f"cần tối thiểu {min_needed_mb:.0f} MB ({n_unique} frames x {unit_size_mb:.1f}MB). Dừng an toàn ENOSPC."
                                    )

                                # Hàm callback kiểm tra lại giữa chunk khi chunk trước chưa encode xong
                                def _mid_chunk_bp_check(curr_idx: int, tot_frames: int):
                                    nonlocal future_encode, total_enc_time, free_mb, last_encode_item, last_ai_result, last_seg_start
                                    if future_encode is not None and not future_encode.done():
                                        cur_free_mb = shutil.disk_usage(cache_root).free / (1024 * 1024)
                                        rem_frames = max(1, tot_frames - curr_idx)
                                        rem_need_mb = rem_frames * unit_size_mb * 1.2
                                        if cur_free_mb < rem_need_mb:
                                            logger.info(
                                                f"⚠️ [MID-CHUNK BACKPRESSURE] Giữa chunk {c_idx+1}: RAM Disk trống {cur_free_mb:.0f} MB < "
                                                f"nhu cầu còn lại {rem_need_mb:.0f} MB. Chờ Media Engine hoàn tất chunk trước..."
                                            )
                                            prev_enc = future_encode.result()
                                            total_enc_time += prev_enc
                                            prev_c = last_encode_item["target"][0]
                                            prev_exp = last_encode_item["extracted"]["expected_frames"]
                                            prev_elap = time.time() - last_seg_start
                                            prev_fps = (prev_exp / prev_elap) if prev_elap > 0 else 0.0
                                            render_progress(
                                                prev_c + 1, total_segs,
                                                prefix=f"✨ Đoạn {prev_c+1:02d}/{total_segs:02d}",
                                                suffix=f"[{prev_fps:.2f} fps | Đã giải phóng RAM Disk]"
                                            )
                                            logger.log_chunk_done(
                                                chunk_idx=prev_c,
                                                total_chunks=total_segs,
                                                frames=prev_exp,
                                                ai_time=last_ai_result["elapsed_seconds"],
                                                ai_fps=last_ai_result["fps"],
                                                enc_time=prev_enc,
                                                total_chunk_time=prev_elap,
                                                output_seg=last_encode_item["target"][2],
                                                t0_seconds=last_ai_result.get("t0_seconds"),
                                                steady_fps=last_ai_result.get("steady_fps"),
                                                rss_mb=last_ai_result.get("peak_rss_mb")
                                            )
                                            future_encode = None
                                            free_mb = shutil.disk_usage(cache_root).free / (1024 * 1024)

                                try:
                                    # 2.2 Độc quyền suy luận AI (GPU Main thread)
                                    dedup_info = f" (Deduplication: {extracted['ai_frames']}/{extracted['frame_count']} frames)" if extracted["chunk_plan"] else ""
                                    render_progress(
                                        c_idx, total_segs,
                                        prefix=f"🚀 Đoạn {c_idx+1:02d}/{total_segs:02d}",
                                        suffix=f"AI Inferencing ({engine.upper()}){dedup_info}..."
                                    )
                                    if zero_disk and engine == "cuda":
                                        import cv2
                                        import psutil
                                        import torch
                                        from src.trt_zero_copy_upscaler import DoubleBufferedPipeUpscaler
                                        from src.build_trt_engine import get_hybrid_engine_or_wrapper
                                        from src.cuda_engine import get_cuda_inference_engine
                                        from src.output_sanity import check_frame_sanity_arrays, OutputSanityError, write_failed_sanity_audit

                                        in_files = []
                                        if extracted.get("raw_rgb_frames") is not None:
                                            unique_frames_rgb = extracted["raw_rgb_frames"]
                                        else:
                                            in_files = sorted([os.path.join(c_in_dir, f) for f in os.listdir(c_in_dir) if f.lower().endswith(f".{fmt}")]) if os.path.exists(c_in_dir) else []
                                            unique_frames_rgb = []
                                            for f in in_files:
                                                bgr = cv2.imread(f, cv2.IMREAD_COLOR)
                                                if bgr is None:
                                                    raise RuntimeError(f"Không thể đọc khung hình trích xuất: {f}")
                                                unique_frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                                        runs = extracted["chunk_plan"].get("runs", []) if extracted.get("chunk_plan") else []
                                        repeat_counts = [r["count"] for r in runs] if runs else [1] * len(unique_frames_rgb)

                                        in_h, in_w = (
                                            unique_frames_rgb[0].shape[:2]
                                            if unique_frames_rgb
                                            else (int(meta.get("height", 1080)), int(meta.get("width", 1920)))
                                        )
                                        cuda_engine = get_cuda_inference_engine(
                                            model_info,
                                            precision=precision,
                                            batch_size=batch_size,
                                            use_tensorrt=use_tensorrt,
                                        )
                                        eff_bs = batch_size or cuda_engine.batch_size or 8
                                        runner_key = (model_info["id"], int(in_h), int(in_w), int(eff_bs), bool(use_tensorrt), precision)

                                        # Khởi tạo & tái sử dụng Engine (TensorRT 10 FP16 / Hybrid Wrapper) + Pinned Buffers xuyên suốt tập
                                        if (
                                            getattr(cuda_engine, "_zero_disk_upscaler", None) is None
                                            or getattr(cuda_engine, "_zero_disk_key", None) != runner_key
                                        ):
                                            wrapped_model = get_hybrid_engine_or_wrapper(
                                                cuda_engine.model,
                                                max_batch=eff_bs,
                                                force_pytorch=(not use_tensorrt) or (precision != "fp16"),
                                                model_id=model_info["id"],
                                                input_hw=(in_h, in_w),
                                                scale=2.0,
                                            )
                                            cuda_engine._zero_disk_wrapper = wrapped_model
                                            cuda_engine._zero_disk_key = runner_key
                                            cuda_engine._zero_disk_upscaler = DoubleBufferedPipeUpscaler(
                                                wrapped_model,
                                                batch_size=eff_bs,
                                                device=cuda_engine.device,
                                                input_hw=(in_h, in_w),
                                                scale=2,
                                            )
                                            if cuda_engine.device.type == "cuda" and torch.cuda.is_available():
                                                torch.cuda.empty_cache()
                                        else:
                                            wrapped_model = cuda_engine._zero_disk_wrapper
                                            cuda_engine._zero_disk_upscaler.ensure_resolution(in_h, in_w)

                                        # Chốt an toàn VRAM cho NVENC 4K 10-bit (lookahead 32 + fullres cần ~1.1 GB VRAM):
                                        # Nếu chunk trước vẫn đang flush NVENC nền và VRAM trống < 2.2 GB, đợi flush (~1.2s) xong trước khi mở phiên NVENC mới.
                                        if (
                                            future_encode is not None
                                            and not future_encode.done()
                                            and cuda_engine.device.type == "cuda"
                                            and torch.cuda.is_available()
                                        ):
                                            try:
                                                free_vram, _ = torch.cuda.mem_get_info(cuda_engine.device)
                                                if free_vram < int(2.4 * (1024 ** 3)):
                                                    torch.cuda.empty_cache()
                                                    free_vram, _ = torch.cuda.mem_get_info(cuda_engine.device)
                                                if free_vram < int(2.2 * (1024 ** 3)):
                                                    prev_enc_elapsed = future_encode.result()
                                                    total_enc_time += prev_enc_elapsed
                                                    prev_c_idx = last_encode_item["target"][0]
                                                    prev_exp_frames = last_encode_item["extracted"]["expected_frames"]
                                                    prev_seg_elapsed = time.time() - last_seg_start
                                                    logger.log_chunk_done(
                                                        chunk_idx=prev_c_idx,
                                                        total_chunks=total_segs,
                                                        frames=prev_exp_frames,
                                                        ai_time=last_ai_result["elapsed_seconds"],
                                                        ai_fps=last_ai_result["fps"],
                                                        enc_time=prev_enc_elapsed,
                                                        total_chunk_time=prev_seg_elapsed,
                                                        output_seg=last_encode_item["target"][2],
                                                        t0_seconds=last_ai_result.get("t0_seconds"),
                                                        steady_fps=last_ai_result.get("steady_fps"),
                                                        rss_mb=last_ai_result.get("peak_rss_mb"),
                                                    )
                                                    future_encode = None
                                            except Exception:
                                                pass

                                        pipe_upscaler = cuda_engine._zero_disk_upscaler
                                        out_h, out_w = pipe_upscaler.output_hw
                                        tmp_seg_out = f"{c_seg_out}.tmp.mp4"

                                        should_check_sanity = (c_idx == 0) or ((c_idx + 1) % 10 == 0)
                                        fps_str = f"{meta['fps_num']}/{meta['fps_den']}"
                                        t0_pipe = time.time()
                                        enc_proc = pipe_upscaler.spawn_nvenc_pipe_writer(
                                            tmp_seg_out, fps_str=fps_str, width=out_w, height=out_h
                                        )
                                        try:
                                            pre_dedup = (
                                                extracted.get("distinct_frames"),
                                                extracted.get("frame_to_distinct"),
                                            ) if (extracted.get("distinct_frames") is not None and extracted.get("frame_to_distinct") is not None) else None
                                            pipe_upscaler.process_frames_to_pipe(
                                                unique_frames_rgb,
                                                repeat_counts,
                                                enc_proc,
                                                cyclic_dedup=bool(dedup),
                                                capture_first_rgb_small=should_check_sanity,
                                                precomputed_dedup=pre_dedup,
                                            )
                                        except Exception:
                                            if enc_proc.stdin:
                                                try:
                                                    enc_proc.stdin.close()
                                                except Exception:
                                                    pass
                                            try:
                                                enc_proc.kill()
                                                enc_proc.wait(timeout=3)
                                            except Exception:
                                                pass
                                            if os.path.exists(tmp_seg_out):
                                                try:
                                                    os.remove(tmp_seg_out)
                                                except OSError:
                                                    pass
                                            raise
                                        finally:
                                            if enc_proc.stdin:
                                                try:
                                                    enc_proc.stdin.close()
                                                except Exception:
                                                    pass
                                        t_pipe_elapsed = time.time() - t0_pipe

                                        # In-memory sanity check ngay sau khi truyền xong frame (không cần chờ NVENC flush xong)
                                        if should_check_sanity and unique_frames_rgb:
                                            sample_in = unique_frames_rgb[0]
                                            sample_out = pipe_upscaler.last_first_rgb_small
                                            if sample_out is None:
                                                with torch.inference_mode():
                                                    s_in_tensor = torch.from_numpy(sample_in).permute(2, 0, 1).unsqueeze(0).to(cuda_engine.device)
                                                    s_dtype = torch.float16 if precision == "fp16" else torch.float32
                                                    s_in_tensor = s_in_tensor.to(memory_format=torch.channels_last, dtype=s_dtype).mul_(1.0 / 255.0)
                                                    s_out_tensor = wrapped_model(s_in_tensor)
                                                    sample_out = (s_out_tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy())
                                            s_res = check_frame_sanity_arrays(sample_in, sample_out, is_rgb=True)
                                            s_res["chunk"] = c_idx + 1
                                            s_res["frame"] = os.path.basename(in_files[0]) if in_files else "frame_0000.png"
                                            sanity_checks_history.append(s_res)
                                            if not s_res["passed"]:
                                                try:
                                                    enc_proc.kill()
                                                    enc_proc.wait(timeout=3)
                                                except Exception:
                                                    pass
                                                if os.path.exists(tmp_seg_out):
                                                    try:
                                                        os.remove(tmp_seg_out)
                                                    except OSError:
                                                        pass
                                                err_msg = (
                                                    f"❌ [CHỐT CHẶN ĐẦU RA THẤT BẠI] Phân đoạn {c_idx+1:02d}: "
                                                    f"|ΔY| = {s_res['delta_y']} (trần {s_res['max_delta_y']}), "
                                                    f"Corr = {s_res['corr']} (sàn {s_res['min_corr']}). "
                                                )
                                                logger.error(err_msg, to_console=True)
                                                audit_file = os.path.splitext(output_path)[0] + ".audit.json"
                                                write_failed_sanity_audit(
                                                    audit_file_path=audit_file,
                                                    source_file=input_path,
                                                    output_file=output_path,
                                                    model_info=model_info,
                                                    engine=engine,
                                                    precision=precision,
                                                    chunk_idx=c_idx,
                                                    metrics=s_res,
                                                    all_checks=sanity_checks_history,
                                                )
                                                raise OutputSanityError(err_msg, metrics=s_res)
                                            else:
                                                logger.info(
                                                    f"🛡️ [ZERO-DISK SANITY PASS] Đoạn {c_idx+1:02d}: |ΔY| = {s_res['delta_y']:.2f}, "
                                                    f"Corr = {s_res['corr']:.4f} ({s_res['elapsed_ms']:.1f}ms)",
                                                    to_console=False
                                                )

                                        can_async_finalize = (
                                            (i + 1 < len(chunks_to_process))
                                            and (free_mb >= 1024.0)
                                            and not (allow_partial and deadline_dt and datetime.now() >= deadline_dt)
                                        )
                                        if can_async_finalize:
                                            zero_disk_pending_finalize = (enc_proc, tmp_seg_out, c_seg_out, c_idx)
                                        else:
                                            worker_finalize_zero_disk(enc_proc, tmp_seg_out, c_seg_out, c_idx)
                                            zero_disk_pending_finalize = None

                                        num_unique_frames = pipe_upscaler.last_distinct_count or len(unique_frames_rgb)
                                        extracted["ai_frames"] = num_unique_frames
                                        extracted["raw_rgb_frames"] = None
                                        del unique_frames_rgb
                                        cuda_engine.restart_if_needed(logger=logger)
                                        rss_now = psutil.Process().memory_info().rss / (1024 * 1024)

                                        ai_result = {
                                            "elapsed_seconds": t_pipe_elapsed,
                                            "fps": (num_unique_frames / t_pipe_elapsed) if t_pipe_elapsed > 0 else 0.0,
                                            "steady_fps": (num_unique_frames / t_pipe_elapsed) if t_pipe_elapsed > 0 else 0.0,
                                            "t0_seconds": 0.0,
                                            "peak_rss_mb": rss_now,
                                            "micro_timings": getattr(pipe_upscaler, "last_micro_timings", None),
                                        }
                                        if getattr(pipe_upscaler, "last_micro_timings", None):
                                            mt = pipe_upscaler.last_micro_timings
                                            last_pipeline_micro_timings = mt
                                            nb = mt.get("num_batches", 0)
                                            if nb > 0:
                                                accumulated_micro_timings["num_batches"] += nb
                                                accumulated_micro_timings["total_h2d_ms"] += mt.get("total_h2d_ms", 0.0)
                                                accumulated_micro_timings["total_trt_ms"] += mt.get("total_trt_ms", 0.0)
                                                accumulated_micro_timings["total_triton_ms"] += mt.get("total_triton_ms", 0.0)
                                                accumulated_micro_timings["total_d2h_ms"] += mt.get("total_d2h_ms", 0.0)
                                            logger.info(
                                                f"⏱️ [CUDA MICRO-TIMINGS] Phân đoạn {c_idx+1:02d}: "
                                                f"TRT={mt.get('avg_trt_ms', 0):.1f}ms, Triton={mt.get('avg_triton_ms', 0):.1f}ms, "
                                                f"D2H={mt.get('avg_d2h_ms', 0):.1f}ms, H2D={mt.get('avg_h2d_ms', 0):.1f}ms (avg/batch)",
                                                to_console=False
                                            )
                                    else:
                                        ai_result = upscale_frames_with_fallback(
                                            input_dir=c_in_dir,
                                            output_dir=c_out_dir,
                                            model_info=model_info,
                                            engine=engine,
                                            precision=precision,
                                            batch_size=batch_size,
                                            fmt=fmt,
                                            logger=logger,
                                            mid_chunk_callback=_mid_chunk_bp_check,
                                            use_tensorrt=use_tensorrt,
                                        )

                                        # Tích lũy kích thước frame PNG thực tế
                                        try:
                                            out_files = [os.path.join(c_out_dir, f) for f in os.listdir(c_out_dir) if f.lower().endswith(f".{fmt}")]
                                            if out_files:
                                                actual_avg_mb = (sum(os.path.getsize(f) for f in out_files) / len(out_files)) / (1024 * 1024)
                                                chunk_png_unit_sizes.append(actual_avg_mb)
                                        except Exception:
                                            pass

                                        # Chốt chặn kiểm định chất lượng hình ảnh đầu ra C3:
                                        # Thực hiện sau chunk đầu tiên (c_idx == 0) và mỗi 10 chunk sau
                                        should_check_sanity = (c_idx == 0) or ((c_idx + 1) % 10 == 0)
                                        if should_check_sanity:
                                            out_frame_list = sorted([f for f in os.listdir(c_out_dir) if f.lower().endswith(f".{fmt}")])
                                            if out_frame_list:
                                                sample_f = out_frame_list[0]
                                                src_f = os.path.join(c_in_dir, sample_f)
                                                out_f = os.path.join(c_out_dir, sample_f)
                                                if os.path.exists(src_f) and os.path.exists(out_f):
                                                    from src.output_sanity import check_frame_sanity, OutputSanityError, write_failed_sanity_audit
                                                    s_res = check_frame_sanity(src_f, out_f)
                                                    s_res["chunk"] = c_idx + 1
                                                    s_res["frame"] = sample_f
                                                    sanity_checks_history.append(s_res)
                                                    if not s_res["passed"]:
                                                        err_msg = (
                                                            f"❌ [CHỐT CHẶN ĐẦU RA THẤT BẠI] Phân đoạn {c_idx+1:02d}: "
                                                            f"|ΔY| = {s_res['delta_y']} (trần {s_res['max_delta_y']}), "
                                                            f"Corr = {s_res['corr']} (sàn {s_res['min_corr']}). "
                                                            f"Mô hình có dấu hiệu xuất video đen hoặc sai lệch cấu trúc!"
                                                        )
                                                        logger.error(err_msg, to_console=True)
                                                        audit_file = os.path.splitext(output_path)[0] + ".audit.json"
                                                        write_failed_sanity_audit(
                                                            audit_file_path=audit_file,
                                                            source_file=input_path,
                                                            output_file=output_path,
                                                            model_info=model_info,
                                                            engine=engine,
                                                            precision=precision,
                                                            chunk_idx=c_idx,
                                                            metrics=s_res,
                                                            all_checks=sanity_checks_history
                                                        )
                                                        raise OutputSanityError(err_msg, metrics=s_res)
                                                    else:
                                                        logger.info(
                                                            f"🛡️ [OUTPUT SANITY PASS] Đoạn {c_idx+1:02d}: |ΔY| = {s_res['delta_y']:.2f}, "
                                                            f"Corr = {s_res['corr']:.4f} ({s_res['elapsed_ms']:.1f}ms)",
                                                            to_console=False
                                                        )

                                    # Chunk hoàn thành thành công, thoát vòng lặp retry
                                    break

                                except Exception as chunk_err:
                                    err_msg = str(chunk_err)
                                    is_enospc = (
                                        (isinstance(chunk_err, OSError) and getattr(chunk_err, "errno", None) == 28)
                                        or "No space left on device" in err_msg
                                        or "ENOSPC" in err_msg
                                        or "exit code 228" in err_msg
                                    )
                                    if is_enospc and chunk_attempt < max_chunk_retries:
                                        chunk_attempt += 1
                                        logger.warn(
                                            f"⚠️ [ENOSPC RETRY] Gặp lỗi thiếu đĩa tại phân đoạn [{c_idx+1}/{total_segs}] "
                                            f"(lần thử {chunk_attempt}/{max_chunk_retries}): {chunk_err}. "
                                            f"Đang dọn sạch file dở, chờ Media Engine và thử lại...",
                                            to_console=True
                                        )
                                        shutil.rmtree(c_out_dir, ignore_errors=True)
                                        if os.path.exists(c_seg_out):
                                            try:
                                                os.remove(c_seg_out)
                                            except OSError:
                                                pass
                                        if future_encode is not None:
                                            try:
                                                prev_enc = future_encode.result()
                                                total_enc_time += prev_enc
                                            except Exception:
                                                pass
                                            future_encode = None
                                        if future_extract is not None and not (zero_disk and engine == "cuda"):
                                            try:
                                                res_pref = future_extract.result()
                                                shutil.rmtree(res_pref.get("in_dir", ""), ignore_errors=True)
                                            except Exception:
                                                pass
                                            future_extract = None
                                        # Tái tạo lại frame trích xuất nếu cần
                                        kept_idx_retry = (
                                            extracted.get("chunk_plan", {}).get("kept_indices")
                                            if extracted.get("chunk_plan")
                                            else None
                                        )
                                        retry_total_frames = (
                                            extracted["chunk_plan"]["total_frames"]
                                            if extracted.get("chunk_plan")
                                            else (seg_frame_counts[c_idx] if c_idx < len(seg_frame_counts) else None)
                                        )
                                        fps_retry = float(meta.get("fps_num", 24000)) / max(1.0, float(meta.get("fps_den", 1001)))
                                        retry_start_sec = (
                                            sum(seg_frame_counts[:c_idx]) / fps_retry
                                            if (fps_retry > 0 and c_idx <= len(seg_frame_counts))
                                            else float(c_idx * segment_time)
                                        )
                                        if zero_disk and engine == "cuda":
                                            if extracted.get("raw_rgb_frames") is None:
                                                safe_hwaccel_retry = is_nvdec_supported(
                                                    codec=meta.get("codec", ""),
                                                    pix_fmt=meta.get("pix_fmt", ""),
                                                    profile=meta.get("profile", ""),
                                                )
                                                extracted["raw_rgb_frames"] = extract_frames_raw_rgb24(
                                                    c_seg_file,
                                                    width=int(meta["width"]),
                                                    height=int(meta["height"]),
                                                    kept_indices=kept_idx_retry,
                                                    hwaccel=safe_hwaccel_retry,
                                                    total_frames=retry_total_frames,
                                                    source_video_path=input_path,
                                                    chunk_start_sec=retry_start_sec,
                                                    source_codec=meta.get("codec"),
                                                )
                                        elif not os.path.exists(c_in_dir) or len(os.listdir(c_in_dir)) == 0:
                                            os.makedirs(c_in_dir, exist_ok=True)
                                            extract_frames(
                                                c_seg_file,
                                                c_in_dir,
                                                fmt=fmt,
                                                kept_indices=kept_idx_retry,
                                                source_video_path=input_path,
                                                chunk_start_sec=retry_start_sec,
                                                total_frames=retry_total_frames,
                                                width=int(meta["width"]),
                                                height=int(meta["height"]),
                                            )
                                        time.sleep(1.0)
                                        continue
                                    elif is_enospc:
                                        shutil.rmtree(c_out_dir, ignore_errors=True)
                                        if os.path.exists(c_seg_out):
                                            try:
                                                os.remove(c_seg_out)
                                            except OSError:
                                                pass
                                        raise FailedResumableError(
                                            f"Tập dừng tại phân đoạn {c_idx+1}/{total_segs} do ENOSPC sau {max_chunk_retries} lần thử lại.",
                                            chunk_idx=c_idx,
                                            up_segs_dir=up_segs_dir
                                        ) from chunk_err
                                    else:
                                        raise

                            total_frames_processed += extracted["expected_frames"]
                            total_ai_frames += extracted["ai_frames"]
                            total_ext_time += extracted["ext_time"]
                            total_ai_time += ai_result["elapsed_seconds"]
                            curr_fps = ai_result.get("fps", 0.0)
                            if curr_fps > 0:
                                ai_fps_history.append(curr_fps)
                                if initial_baseline_fps is None and len(ai_fps_history) >= 2:
                                    initial_baseline_fps = float(np.mean(ai_fps_history[:2]))
                                elif initial_baseline_fps is not None and len(ai_fps_history) >= 10:
                                    recent_10 = ai_fps_history[-10:]
                                    drop_pct = ((initial_baseline_fps - float(np.mean(recent_10))) / initial_baseline_fps) * 100.0
                                    if drop_pct > 8.0 and all(f < initial_baseline_fps * 0.92 for f in recent_10) and not throttling_warned:
                                        logger.warn(
                                            f"⚠️ CẢNH BÁO NHIỆT/THROTTLE: Tốc độ GPU ({float(np.mean(recent_10)):.2f} fps) "
                                            f"đã giảm {drop_pct:.1f}% (> 8%) so với đầu phiên ({initial_baseline_fps:.2f} fps) "
                                            f"trong 10 chunk liên tiếp.",
                                            to_console=True
                                        )
                                        throttling_warned = True

                            # Dọn sạch input frames của chunk hiện tại ngay lập tức sau khi AI infer xong
                            shutil.rmtree(c_in_dir, ignore_errors=True)

                            # Nếu phân đoạn trước đang mã hóa ở luồng nền, chờ hoàn tất và ghi log
                            if future_encode is not None:
                                prev_enc_elapsed = future_encode.result()
                                total_enc_time += prev_enc_elapsed
                                prev_c_idx = last_encode_item["target"][0]
                                prev_exp_frames = last_encode_item["extracted"]["expected_frames"]
                                prev_seg_elapsed = time.time() - last_seg_start
                                prev_seg_fps = (prev_exp_frames / prev_seg_elapsed) if prev_seg_elapsed > 0 else 0.0
                                remaining_segs = total_segs - (prev_c_idx + 1)
                                eta_seconds = remaining_segs * prev_seg_elapsed

                                render_progress(
                                    prev_c_idx + 1, total_segs,
                                    prefix=f"✨ Đoạn {prev_c_idx+1:02d}/{total_segs:02d}",
                                    suffix=f"[{prev_seg_fps:.2f} fps (AI: {last_ai_result['fps']:.2f}) | ETA: {format_time(eta_seconds)}]"
                                )
                                print()
                                logger.log_chunk_done(
                                    chunk_idx=prev_c_idx,
                                    total_chunks=total_segs,
                                    frames=prev_exp_frames,
                                    ai_time=last_ai_result["elapsed_seconds"],
                                    ai_fps=last_ai_result["fps"],
                                    enc_time=prev_enc_elapsed,
                                    total_chunk_time=prev_seg_elapsed,
                                    output_seg=last_encode_item["target"][2],
                                    t0_seconds=last_ai_result.get("t0_seconds"),
                                    steady_fps=last_ai_result.get("steady_fps"),
                                    rss_mb=last_ai_result.get("peak_rss_mb")
                                )
                                future_encode = None

                            # Khởi động mã hóa phần cứng NVENC cho chunk hiện tại ở luồng nền (hoặc đẩy flush zero-disk sang EncodeWorker)
                            if zero_disk and engine == "cuda":
                                shutil.rmtree(c_out_dir, ignore_errors=True)
                                if zero_disk_pending_finalize is not None:
                                    encode_item = {
                                        "target": target_chunk,
                                        "extracted": extracted,
                                        "out_dir": c_out_dir,
                                    }
                                    last_encode_item = encode_item
                                    last_ai_result = ai_result
                                    last_seg_start = seg_start_time
                                    future_encode = enc_exec.submit(
                                        worker_finalize_zero_disk, *zero_disk_pending_finalize
                                    )
                                    zero_disk_pending_finalize = None
                                else:
                                    chunk_elapsed = time.time() - seg_start_time
                                    total_enc_time += 0.001
                                    render_progress(
                                        c_idx + 1, total_segs,
                                        prefix=f"✨ Đoạn {c_idx+1:02d}/{total_segs:02d}",
                                        suffix=f"[{extracted['expected_frames']/max(0.01, chunk_elapsed):.2f} fps (Zero-Disk Pipe)]"
                                    )
                                    print()
                                    logger.log_chunk_done(
                                        chunk_idx=c_idx,
                                        total_chunks=total_segs,
                                        frames=extracted["expected_frames"],
                                        ai_time=ai_result["elapsed_seconds"],
                                        ai_fps=ai_result["fps"],
                                        enc_time=0.001,
                                        total_chunk_time=chunk_elapsed,
                                        output_seg=c_seg_out,
                                        t0_seconds=0.0,
                                        steady_fps=ai_result["fps"],
                                        rss_mb=0.0
                                    )
                                    future_encode = None
                            else:
                                encode_item = {
                                    "target": target_chunk,
                                    "extracted": extracted,
                                    "out_dir": c_out_dir
                                }
                                last_encode_item = encode_item
                                last_ai_result = ai_result
                                last_seg_start = seg_start_time
                                render_progress(
                                    c_idx, total_segs,
                                    prefix=f"🚀 Đoạn {c_idx+1:02d}/{total_segs:02d}",
                                    suffix="Hardware Encoding 10-bit (Pipelined)..."
                                )
                                future_encode = enc_exec.submit(worker_encode, encode_item, ai_result)

                            # Khởi động giải mã CPU cho chunk tiếp theo ở luồng nền (nếu chưa prefetch sớm ở đầu chunk)
                            if future_extract is None and i + 1 < len(chunks_to_process):
                                future_extract = dec_exec.submit(worker_extract, chunks_to_process[i + 1])

                            # Kiểm tra mốc hạn chót nếu bật --allow-partial
                            if allow_partial and deadline_dt and datetime.now() >= deadline_dt:
                                logger.info(
                                    f"⏰ [ALLOW-PARTIAL DEADLINE] Đã chạm mốc hạn chót ({deadline_dt.strftime('%H:%M:%S')}). "
                                    f"Dừng sạch tại ranh giới phân đoạn {c_idx+1}/{total_segs}.",
                                    to_console=True
                                )
                                partial_paused = True
                                if future_extract is not None:
                                    try:
                                        res_ext = future_extract.result()
                                        shutil.rmtree(res_ext.get("in_dir", ""), ignore_errors=True)
                                    except Exception:
                                        pass
                                    future_extract = None
                                break

                    # Sau khi hoàn tất toàn bộ loop: Chờ chunk cuối cùng mã hóa xong
                    if future_encode is not None:
                        prev_enc_elapsed = future_encode.result()
                        total_enc_time += prev_enc_elapsed
                        prev_c_idx = last_encode_item["target"][0]
                        prev_exp_frames = last_encode_item["extracted"]["expected_frames"]
                        prev_seg_elapsed = time.time() - last_seg_start
                        prev_seg_fps = (prev_exp_frames / prev_seg_elapsed) if prev_seg_elapsed > 0 else 0.0

                        render_progress(
                            prev_c_idx + 1, total_segs,
                            prefix=f"✨ Đoạn {prev_c_idx+1:02d}/{total_segs:02d}",
                            suffix=f"[{prev_seg_fps:.2f} fps (AI: {last_ai_result['fps']:.2f}) | Hoàn tất]"
                        )
                        print()
                        logger.log_chunk_done(
                            chunk_idx=prev_c_idx,
                            total_chunks=total_segs,
                            frames=prev_exp_frames,
                            ai_time=last_ai_result["elapsed_seconds"],
                            ai_fps=last_ai_result["fps"],
                            enc_time=prev_enc_elapsed,
                            total_chunk_time=prev_seg_elapsed,
                            output_seg=last_encode_item["target"][2],
                            t0_seconds=last_ai_result.get("t0_seconds"),
                            steady_fps=last_ai_result.get("steady_fps"),
                            rss_mb=last_ai_result.get("peak_rss_mb")
                        )

                    if partial_paused:
                        resume_out_dir = os.path.dirname(os.path.abspath(output_path))
                        logger.info(
                            f"\n{'='*75}\n"
                            f"⏸️ [PARTIAL RUN PAUSED] ĐÃ DỪNG SẠCH TẠI RANH GIỚI PHÂN ĐOẠN {prev_c_idx+1}/{total_segs} THEO --allow-partial.\n"
                            f"• File nguồn: {input_path}\n"
                            f"• Thư mục lưu phân đoạn: '{up_segs_dir}' (đã giữ an toàn {prev_c_idx+1} file up_*.mp4)\n"
                            f"• Phân đoạn tiếp theo: {prev_c_idx+2}/{total_segs}\n"
                            f"💡 CÁCH TIẾP TỤC: Chạy lại lệnh sau (hệ thống sẽ tự động resume từ đoạn {prev_c_idx+2}):\n"
                            f"   python upscale.py \"{input_path}\" -o \"{resume_out_dir}\"\n"
                            f"{'='*75}\n",
                            to_console=True
                        )
                        return up_segs_dir
                finally:
                    if future_extract is not None:
                        try:
                            res_ext = future_extract.result()
                            shutil.rmtree(res_ext.get("in_dir", ""), ignore_errors=True)
                        except Exception:
                            pass
                        future_extract = None
                    release_gpu_lock()

            stage_timings["AI Inference & Hardware Encoding (Pipelined)"] = time.time() - t_chunks_start
            logger.info("Hoàn tất quá trình suy luận 4K cho toàn bộ phân đoạn!", to_console=True)

            # -----------------------------------------------------------------
            # BƯỚC 3 & 4 (HỢP NHẤT 1 LƯỢT): CONCAT + MUX CONTAINER PHÒNG THU
            # -----------------------------------------------------------------
            logger.info("Đang ghép nối phân đoạn 4K và đóng gói Audio, Subtitles, Fonts, Chapters gốc (1-Pass Concat-Mux)...", to_console=True)
            all_upscaled_segs = sorted([
                os.path.join(up_segs_dir, f)
                for f in os.listdir(up_segs_dir)
                if f.startswith("up_") and f.endswith(".mp4")
            ]) if os.path.exists(up_segs_dir) else []
            t_mux_start = time.time()
            mux_segments_directly(all_upscaled_segs, work_dir, input_path, output_path)
            stage_timings["Ghép nối 4K & Đóng gói Container gốc (1-Pass Mux)"] = time.time() - t_mux_start

            total_elapsed = time.time() - start_total_time
            final_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            avg_fps = (total_frames_processed / total_elapsed) if total_elapsed > 0 and total_frames_processed > 0 else 0.0
            computed_frames = total_ai_frames if (dedup and total_ai_frames > 0) else total_frames_processed
            ai_fps = (computed_frames / total_ai_time) if total_ai_time > 0 and computed_frames > 0 else 0.0

            if dedup and dedup_plan and total_frames_processed > 0:
                saved_pct = (1.0 - (computed_frames / total_frames_processed)) * 100.0
                dedup_desc = f"Bật (Tiết kiệm {saved_pct:.1f}% tải GPU | {computed_frames}/{total_frames_processed} frames unique)"
            else:
                dedup_desc = "Tắt (Tính toán 100% khung hình)"

            stage_timings["Trích xuất ảnh đệm (Extract)"] = total_ext_time
            stage_timings[f"Suy luận AI ({engine.upper()})"] = total_ai_time
            stage_timings[f"Nén HEVC 10-bit ({HEVC_ENCODER})"] = total_enc_time

            swap_after = get_swap_usage()
            thermal_info = get_thermal_state()

            summary_stats = {
                "output_path": output_path,
                "final_size_bytes": final_size,
                "total_seconds": total_elapsed,
                "total_frames": total_frames_processed,
                "avg_fps": avg_fps,
                "ai_fps": ai_fps,
                "dedup_info": dedup_desc,
                "device_name": DEVICE_NAME,
                "ssd_protection": "100% An toàn (Frame trung gian chỉ chạy trên RAM Disk)"
                if use_ramdisk else "Rolling Segments Cleanup (Xóa cuốn chiếu sau từng chunk)",
                "swap_info": f"{swap_before} ➔ {swap_after}",
                "thermal_info": thermal_info,
                "stage_timings": stage_timings
            }
            # Ghi báo cáo kiểm định tập (audit.json)
            audit_file = os.path.splitext(output_path)[0] + ".audit.json"
            try:
                from src.mux_audit import verify_mux_integrity
                mux_res = verify_mux_integrity(input_path, output_path, src_meta=meta)

                probe_final = mux_res.get("output_meta") or probe_video(output_path)
                probe_src = mux_res.get("source_meta") or meta
                final_frames = probe_final.get("nb_frames") or mux_res.get("video_frame_count", {}).get("output_frames", 0)
                src_frames = probe_src.get("nb_frames") or mux_res.get("video_frame_count", {}).get("source_frames", 0)
                final_dur = probe_final.get("duration", 0.0)
                src_dur = probe_src.get("duration", 0.0)
                drift_s = abs(final_dur - src_dur)

                final_micro_timings = None
                if accumulated_micro_timings.get("num_batches", 0) > 0:
                    nb = accumulated_micro_timings["num_batches"]
                    final_micro_timings = {
                        "num_batches": nb,
                        "total_h2d_ms": round(accumulated_micro_timings["total_h2d_ms"], 2),
                        "total_trt_ms": round(accumulated_micro_timings["total_trt_ms"], 2),
                        "total_triton_ms": round(accumulated_micro_timings["total_triton_ms"], 2),
                        "total_d2h_ms": round(accumulated_micro_timings["total_d2h_ms"], 2),
                        "avg_h2d_ms": round(accumulated_micro_timings["total_h2d_ms"] / nb, 2),
                        "avg_trt_ms": round(accumulated_micro_timings["total_trt_ms"] / nb, 2),
                        "avg_triton_ms": round(accumulated_micro_timings["total_triton_ms"] / nb, 2),
                        "avg_d2h_ms": round(accumulated_micro_timings["total_d2h_ms"] / nb, 2),
                    }
                elif last_pipeline_micro_timings is not None:
                    final_micro_timings = last_pipeline_micro_timings

                audit_data = {
                    "status": mux_res.get("status", "PASS"),
                    "source_file": os.path.abspath(input_path),
                    "output_file": os.path.abspath(output_path),
                    "model_id": model_info["id"],
                    "model_name": model_info["name"],
                    "model_sha256": model_sha256,
                    "engine": engine,
                    "precision": precision,
                    "resolution_source": model_info.get("resolution_source", ""),
                    "source_frames": src_frames,
                    "output_frames": final_frames,
                    "frames_matched": (src_frames == final_frames) if (src_frames and final_frames) else True,
                    "duration_seconds": round(final_dur, 3),
                    "duration_drift_seconds": round(drift_s, 4),
                    "av_drift_frames": round(drift_s * meta.get("fps", 23.976), 3),
                    "audio_tracks_count": probe_final.get("audio_count", 0),
                    "subtitle_tracks_count": probe_final.get("subtitle_count", 0),
                    "fonts_count": probe_final.get("attachment_count", 0),
                    "chapters_count": probe_final.get("chapter_count", 0),
                    "output_file_size_mb": round(final_size / (1024 * 1024), 2),
                    "total_processing_seconds": round(total_elapsed, 2),
                    "average_pipeline_fps": round(avg_fps, 2),
                    "ai_steady_fps": round(ai_fps, 2),
                    "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "output_sanity": {
                        "status": "PASS" if (sanity_checks_history and all(c.get("passed", False) for c in sanity_checks_history)) else ("SKIPPED" if not sanity_checks_history else "FAIL"),
                        "total_checks": len(sanity_checks_history),
                        "checks": sanity_checks_history
                    },
                    "micro_timings": final_micro_timings,
                    "mux_audit": mux_res
                }
                with open(audit_file, "w", encoding="utf-8") as f:
                    json.dump(audit_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Đã xuất báo cáo kiểm định tập: {audit_file} (Mux Audit: {mux_res.get('status')})", to_console=True)

                if mux_res.get("status") == "FAIL":
                    logger.error(f"❌ MUX AUDIT THẤT BẠI: {mux_res.get('failure_reasons')}", to_console=True)
                    raise RuntimeError(f"Mux audit container thất bại: {mux_res.get('failure_reasons')}")

                # -------------------------------------------------------------
                # TỰ ĐỘNG DỌN DẸP THƯ MỤC CHECKPOINT TRÊN SSD KHI XUẤT XƯỞNG PASS
                # -------------------------------------------------------------
                if not keep_temp:
                    cleanup_checkpoint_dir(work_dir, output_dir=os.path.dirname(output_path), logger=logger)
                else:
                    logger.info(f"ℹ️ Cờ --keep-temp được bật: Giữ lại thư mục tạm '{work_dir}'.", to_console=True)
            finally:
                if preview_temp_input and os.path.exists(preview_temp_input):
                    try:
                        os.remove(preview_temp_input)
                    except OSError:
                        pass

            # Gửi thông báo hệ thống khi hoàn thành
            send_system_notification(
                title="🎬 AI Video Upscaler 4K",
                message=f"Hoàn tất thành công {meta['width']}x{meta['height']} ➔ 4K ({format_time(total_elapsed)})",
                sound="Glass"
            )

            # Tự động mở trình phát nếu đang chạy chế độ preview (trừ khi chạy benchmark tự động)
            if is_preview and not os.environ.get("UPSCALER_NO_AUTO_OPEN"):
                try:
                    os.startfile(output_path)
                except Exception:
                    pass

    return output_path
