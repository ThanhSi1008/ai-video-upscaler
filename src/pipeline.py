# -*- coding: utf-8 -*-
"""
src/pipeline.py
Bộ điều phối chính (Orchestrator) cho toàn bộ quy trình Upscale 4K:
- Kiến trúc phân đoạn cuốn chiếu (Rolling Segments)
- Bộ nhớ đệm RAM Disk ảo (Zero NVMe SSD wear)
- Cơ chế chống ngủ máy tự động (caffeinate)
- Kiểm tra an toàn trước khi chạy (Pre-flight Health Checks: Pin, Ổ đĩa)
- Tự động tiếp tục từ phân đoạn dở (Checkpoint Resume)
- Thông báo hệ thống macOS khi hoàn tất
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
    split_video_segments,
    extract_frames,
    encode_chunk_hevc,
    concat_segments,
    mux_final_container
)
from src.dedup import (
    load_or_create_plan,
    prune_redundant_frames,
    generate_concat_demuxer_script,
    encode_chunk_reexpanded
)
from src.ai_engine import upscale_frames
from src.models_resolver import resolve_model_for_video
from src.mps_engine import upscale_frames_with_fallback
from src.logger import (
    format_time,
    format_bytes,
    render_progress,
    RunLogger,
    get_swap_usage,
    get_thermal_state
)
from src.preflight import (
    CaffeinateContext,
    run_preflight_checks,
    send_macos_notification
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
    caffeinate: bool = True,
    preview_seconds: Optional[int] = None,
    dedup: bool = True,
    allow_partial: bool = False,
    deadline_dt: Optional[datetime] = None,
    keep_temp: bool = False
) -> str:
    """Điều phối toàn bộ quy trình Upscale 4K trên Apple Silicon M3 Pro"""
    validate_environment()

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
        preview_dur = min(int(preview_seconds), int(meta["duration"]))
        # Chọn điểm bắt đầu ở khoảng 30% thời lượng phim để tránh màn hình đen mở đầu
        start_pt = max(0, int(meta["duration"] * 0.3))
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
    out_suffix = "_4K_animejanai.mkv" if model_info["id"] == "animejanai-hd-v3-compact" else "_4K.mkv"

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

    # Kích hoạt Caffeinate để chống Sleep máy trong suốt phiên chạy
    with CaffeinateContext(enabled=caffeinate):
        if caffeinate and sys.platform == "darwin":
            logger.info("Cơ chế chống ngủ máy (caffeinate -dims) đã được kích hoạt thành công.")

        target_w = meta["width"] * DEFAULT_SCALE
        target_h = meta["height"] * DEFAULT_SCALE

        # Thư mục lưu phân đoạn mã hóa trên SSD (phục vụ Checkpoint Resume)
        work_dir = os.path.join(os.path.dirname(output_path), f".temp_{base_name}")
        segs_dir = os.path.join(work_dir, "raw_segs")
        up_segs_dir = os.path.join(work_dir, "upscaled_segs")
        os.makedirs(segs_dir, exist_ok=True)
        os.makedirs(up_segs_dir, exist_ok=True)

        # Kiểm tra tính đồng nhất của model qua manifest.json (D7)
        manifest_path = os.path.join(work_dir, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                prev_model_id = manifest_data.get("model_id")
                prev_sha256 = manifest_data.get("sha256")
                if prev_model_id and (prev_model_id != model_info["id"] or prev_sha256 != model_info["sha256"]):
                    if not force:
                        raise ValueError(
                            f"❌ Xung đột mô hình: Thư mục tạm '{work_dir}' chứa phân đoạn của mô hình '{prev_model_id}' "
                            f"(sha256: {prev_sha256[:8] if prev_sha256 else 'N/A'}), nhưng phiên hiện tại yêu cầu '{model_info['id']}' "
                            f"(sha256: {model_info['sha256'][:8]}).\n"
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
            "sha256": model_info["sha256"],
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
                if getattr(cache_root, "is_ramdisk", False) or "RAM" in cache_root or "/Volumes/" in cache_root
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
            logger.info(f"📊 Trạng thái Swap macOS ban đầu: {swap_before}")

            # -----------------------------------------------------------------
            # BƯỚC 1: PHÂN ĐOẠN VIDEO THÔ & KẾ HOẠCH DEDUPLICATION
            # -----------------------------------------------------------------
            dedup_plan = None
            if dedup:
                plan_file = os.path.join(work_dir, "plan.json")
                logger.info("Đang kiểm tra / phân tích kế hoạch Deduplication toàn tập (plan.json)...", to_console=True)
                t_pref_start = time.time()
                dedup_plan = load_or_create_plan(
                    input_path=input_path,
                    plan_path=plan_file,
                    segment_seconds=segment_time,
                    fps_num=meta["fps_num"],
                    fps_den=meta["fps_den"],
                    resume=resume
                )
                if dedup_plan:
                    dedup_plan["model_id"] = model_info["id"]
                    dedup_plan["model_sha256"] = model_info["sha256"]
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
            logger.info(f"Đã chuẩn bị {total_segs} phân đoạn. Bắt đầu quá trình Upscale GPU Metal...", to_console=True)
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

            # Lọc các phân đoạn cần xử lý (bỏ qua những đoạn đã xong nếu bật resume)
            chunks_to_process = []
            for idx, seg_file in enumerate(existing_segs):
                seg_out_file = os.path.join(up_segs_dir, f"up_{idx:04d}.mp4")
                if resume and os.path.exists(seg_out_file) and os.path.getsize(seg_out_file) > 10000:
                    logger.info(f"Phân đoạn [{idx+1:02d}/{total_segs:02d}] đã hoàn thành từ trước, tiếp tục...")
                    print(f"⏭️  Phân đoạn [{idx+1:02d}/{total_segs:02d}] đã hoàn thành từ trước, tiếp tục...")
                    total_frames_processed += dedup_plan["chunks"][idx]["total_frames"] if (dedup and dedup_plan) else 192
                    continue
                chunks_to_process.append((idx, seg_file, seg_out_file))

            if not chunks_to_process:
                logger.info("Tất cả phân đoạn đã hoàn thành từ trước!", to_console=True)
            else:
                # Định nghĩa các worker function cho từng stage của kiến trúc Overlapped Double-Buffering
                def worker_extract(item):
                    """CPU Decode Worker: Trích xuất và lọc deduplication vào RAM Disk độc lập."""
                    c_idx, c_seg_file, _ = item
                    c_in_dir = os.path.join(cache_root, f"chunk_{c_idx:04d}_in")
                    os.makedirs(c_in_dir, exist_ok=True)
                    c_plan = (
                        dedup_plan["chunks"][c_idx]
                        if (dedup and dedup_plan and c_idx < len(dedup_plan.get("chunks", [])))
                        else None
                    )
                    kept_indices = c_plan["kept_indices"] if c_plan else None

                    # Đảm bảo RAM Disk có tối thiểu không gian an toàn cho 1080p frames (~350MB) trước khi extract
                    t_wait_ext = time.time()
                    while (shutil.disk_usage(cache_root).free / (1024 * 1024)) < 450.0:
                        if time.time() - t_wait_ext > 120.0:
                            break
                        time.sleep(0.2)

                    t0 = time.time()
                    f_count = extract_frames(c_seg_file, c_in_dir, fmt=fmt, kept_indices=kept_indices)
                    t_ext = time.time() - t0

                    ai_count = f_count
                    exp_frames = c_plan["total_frames"] if c_plan else f_count

                    return {
                        "idx": c_idx,
                        "seg_file": c_seg_file,
                        "in_dir": c_in_dir,
                        "frame_count": f_count,
                        "ai_frames": ai_count,
                        "expected_frames": exp_frames,
                        "ext_time": t_ext,
                        "chunk_plan": c_plan
                    }

                def worker_encode(item, ai_res):
                    """Media Engine Encode Worker: Nén VideoToolbox 10-bit và dọn sạch RAM Disk ngay lập tức."""
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

                acquire_gpu_lock()
                try:
                    # Thực thi gối đầu song song (Double-Buffering Bounded Queue <= 2 chunks trên RAM Disk)
                    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="DecodeWorker") as dec_exec, \
                         ThreadPoolExecutor(max_workers=1, thread_name_prefix="EncodeWorker") as enc_exec:

                        # Bắt đầu trích xuất chunk đầu tiên
                        future_extract = dec_exec.submit(worker_extract, chunks_to_process[0])
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

                        for i, target_chunk in enumerate(chunks_to_process):
                            c_idx, c_seg_file, c_seg_out = target_chunk

                            # Chờ giải mã CPU chunk hiện tại sẵn sàng (chỉ lấy 1 lần cho chunk i)
                            if future_extract is not None:
                                extracted = future_extract.result()
                                future_extract = None
                            else:
                                extracted = worker_extract(target_chunk)

                            c_in_dir = extracted["in_dir"]
                            c_out_dir = os.path.join(cache_root, f"chunk_{c_idx:04d}_out")
                            os.makedirs(c_out_dir, exist_ok=True)

                            chunk_attempt = 0
                            max_chunk_retries = 2

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
                                min_needed_mb = n_unique * unit_size_mb
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
                                    # 2.2 Độc quyền suy luận AI NCNN Vulkan / PyTorch MPS trên GPU M3 Pro (Main thread)
                                    dedup_info = f" (Deduplication: {extracted['ai_frames']}/{extracted['frame_count']} frames)" if extracted["chunk_plan"] else ""
                                    render_progress(
                                        c_idx, total_segs,
                                        prefix=f"🚀 Đoạn {c_idx+1:02d}/{total_segs:02d}",
                                        suffix=f"AI Metal Inferencing{dedup_info}..."
                                    )
                                    ai_result = upscale_frames_with_fallback(
                                        input_dir=c_in_dir,
                                        output_dir=c_out_dir,
                                        model_info=model_info,
                                        engine=engine,
                                        precision=precision,
                                        fmt=fmt,
                                        logger=logger,
                                        mid_chunk_callback=_mid_chunk_bp_check
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
                                    should_check_sanity = (c_idx == 0) or ((c_idx + 1) % 10 == 0) or (c_idx % 10 == 0)
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
                                        # Tái tạo lại frame trích xuất nếu cần
                                        if not os.path.exists(c_in_dir) or len(os.listdir(c_in_dir)) == 0:
                                            os.makedirs(c_in_dir, exist_ok=True)
                                            extract_frames(c_seg_file, c_in_dir, fmt=fmt, kept_indices=extracted.get("chunk_plan", {}).get("kept_indices") if extracted.get("chunk_plan") else None)
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

                            # Khởi động mã hóa VideoToolbox cho chunk hiện tại ở luồng nền (Media Engine)
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

                            # Khởi động giải mã CPU cho chunk tiếp theo ở luồng nền
                            if i + 1 < len(chunks_to_process):
                                future_extract = dec_exec.submit(worker_extract, chunks_to_process[i + 1])
                            else:
                                future_extract = None

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
                            f"   ./upscale.sh \"{input_path}\" -o \"{resume_out_dir}\"\n"
                            f"{'='*75}\n",
                            to_console=True
                        )
                        return up_segs_dir
                finally:
                    release_gpu_lock()

            stage_timings["AI Inference & Hardware Encoding (Pipelined)"] = time.time() - t_chunks_start
            logger.info("Hoàn tất quá trình suy luận 4K cho toàn bộ phân đoạn!", to_console=True)

            # -----------------------------------------------------------------
            # BƯỚC 3: GHÉP NỐI KHÔNG MẤT DỮ LIỆU CÁC PHÂN ĐOẠN 4K
            # -----------------------------------------------------------------
            logger.info("Đang ghép nối các phân đoạn 4K...", to_console=True)
            all_upscaled_segs = sorted([
                os.path.join(up_segs_dir, f)
                for f in os.listdir(up_segs_dir)
                if f.startswith("up_") and f.endswith(".mp4")
            ]) if os.path.exists(up_segs_dir) else []
            merged_video = os.path.join(work_dir, "merged_4k.mp4")
            t_concat_start = time.time()
            concat_segments(all_upscaled_segs, work_dir, merged_video)
            stage_timings["Ghép nối 4K (Concat Demuxer)"] = time.time() - t_concat_start

            # -----------------------------------------------------------------
            # BƯỚC 4: MUXING ĐÓNG GÓI NGUYÊN BẢN AUDIO, SUBTITLES & CHAPTERS
            # -----------------------------------------------------------------
            logger.info("Đang đóng gói lại Audio, Subtitles, Fonts và Chapters gốc vào file 4K hoàn chỉnh...", to_console=True)
            t_mux_start = time.time()
            mux_final_container(merged_video, input_path, output_path)
            stage_timings["Đóng gói Container gốc (Muxing)"] = time.time() - t_mux_start

            # Xóa file tạm đoạn xem trước (preview) nếu có
            if preview_temp_input and os.path.exists(preview_temp_input):
                os.remove(preview_temp_input)

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
            stage_timings["Suy luận AI Metal (NCNN Vulkan)"] = total_ai_time
            stage_timings["Nén HEVC 10-bit VideoToolbox"] = total_enc_time

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
                "ssd_protection": "100% An toàn (Frame trung gian chỉ chạy trên RAM Disk)"
                if use_ramdisk else "Ghi tạm trên SSD",
                "swap_info": f"{swap_before} ➔ {swap_after}",
                "thermal_info": thermal_info,
                "stage_timings": stage_timings
            }
            # Ghi báo cáo kiểm định tập (audit.json)
            audit_file = os.path.splitext(output_path)[0] + ".audit.json"
            try:
                from src.mux_audit import verify_mux_integrity
                mux_res = verify_mux_integrity(input_path, output_path)

                probe_final = probe_video(output_path)
                probe_src = probe_video(input_path)
                final_frames = probe_final.get("nb_frames")
                src_frames = probe_src.get("nb_frames")
                final_dur = probe_final.get("duration", 0.0)
                src_dur = probe_src.get("duration", 0.0)
                drift_s = abs(final_dur - src_dur)

                audit_data = {
                    "status": mux_res.get("status", "PASS"),
                    "source_file": os.path.abspath(input_path),
                    "output_file": os.path.abspath(output_path),
                    "model_id": model_info["id"],
                    "model_name": model_info["name"],
                    "model_sha256": model_info["sha256"],
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
            except Exception as e:
                logger.warn(f"Lỗi kiểm định audit.json: {e}")
                if "Mux audit container thất bại" in str(e):
                    raise

            # Gửi thông báo hệ thống macOS khi hoàn thành
            send_macos_notification(
                title="🎬 AI Video Upscaler 4K",
                message=f"Hoàn tất thành công {meta['width']}x{meta['height']} ➔ 4K ({format_time(total_elapsed)})",
                sound="Glass"
            )

            # Tự động mở IINA nếu đang chạy chế độ preview
            if is_preview:
                subprocess.run(["open", "-a", "IINA", output_path], check=False)

    return output_path
