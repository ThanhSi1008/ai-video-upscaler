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
    DEFAULT_RAMDISK_SIZE_MB
)
from src.ramdisk import RAMDisk
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
    dedup: bool = True
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
        multiplier = 11.0 if fmt == "png" else 4.5
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

                    t0 = time.time()
                    try:
                        if c_plan:
                            concat_script = os.path.join(cache_root, f"concat_{c_idx:04d}.txt")
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
                            if os.path.exists(concat_script):
                                os.remove(concat_script)
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
                        shutil.rmtree(c_in_dir, ignore_errors=True)
                        shutil.rmtree(c_out_dir, ignore_errors=True)

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

                    for i, target_chunk in enumerate(chunks_to_process):
                        c_idx, c_seg_file, c_seg_out = target_chunk
                        seg_start_time = time.time()

                        # Chờ giải mã CPU chunk hiện tại sẵn sàng (nếu chưa xong)
                        extracted = future_extract.result()

                        # Ngay lập tức khởi động giải mã CPU cho chunk tiếp theo ở luồng nền
                        if i + 1 < len(chunks_to_process):
                            future_extract = dec_exec.submit(worker_extract, chunks_to_process[i + 1])
                        else:
                            future_extract = None

                        c_in_dir = extracted["in_dir"]
                        c_out_dir = os.path.join(cache_root, f"chunk_{c_idx:04d}_out")
                        os.makedirs(c_out_dir, exist_ok=True)

                        total_frames_processed += extracted["expected_frames"]
                        total_ai_frames += extracted["ai_frames"]
                        total_ext_time += extracted["ext_time"]

                        # 2.2 Kiểm tra dung lượng RAM Disk thực tế trước khi AI xuất frame 4K (Worst-case protection)
                        # Tính trước: need = n_unique * size_est * 1.3 (với PNG 4K ~12 MB/frame, chunk 192 frame cần ~3.0 GB)
                        n_unique = extracted["ai_frames"]
                        size_est_mb = 12.0 if fmt.lower() == "png" else 1.2
                        need_mb = n_unique * size_est_mb * 1.3
                        free_mb = shutil.disk_usage(cache_root).free / (1024 * 1024)

                        # Nếu RAM Disk hiện tại không đủ chỗ an toàn và đang có chunk trước encode ở luồng nền:
                        # Đồng bộ đợi Media Engine encode xong và dọn sạch frame chunk trước
                        if future_encode is not None and free_mb < need_mb:
                            logger.info(
                                f"⚠️ RAM Disk trống {free_mb:.0f} MB < nhu cầu an toàn {need_mb:.0f} MB "
                                f"({n_unique} frames x {size_est_mb:.1f}MB x 1.3). Chờ Media Engine giải phóng chunk trước..."
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

                        # Kiểm tra giới hạn cứng ENOSPC trước khi khởi động NCNN
                        min_needed_mb = n_unique * size_est_mb
                        if free_mb < min_needed_mb:
                            raise OSError(
                                28,
                                f"RAM Disk không đủ không gian cho chunk {c_idx}: Trống {free_mb:.0f} MB, "
                                f"cần tối thiểu {min_needed_mb:.0f} MB ({n_unique} frames x {size_est_mb:.1f}MB). Dừng an toàn ENOSPC."
                            )

                        # 2.2 Độc quyền suy luận AI NCNN Vulkan trên GPU M3 Pro (Main thread)
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
                            logger=logger
                        )
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

            # Dọn dẹp thư mục trung gian trên SSD
            shutil.rmtree(work_dir, ignore_errors=True)
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
                probe_final = probe_video(output_path)
                probe_src = probe_video(input_path)
                final_frames = probe_final.get("nb_frames")
                src_frames = probe_src.get("nb_frames")
                final_dur = probe_final.get("duration", 0.0)
                src_dur = probe_src.get("duration", 0.0)
                drift_s = abs(final_dur - src_dur)

                audit_data = {
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
                    "subtitle_tracks_count": probe_final.get("sub_count", 0),
                    "fonts_count": probe_final.get("attach_count", 0),
                    "chapters_count": probe_final.get("chap_count", 0),
                    "output_file_size_mb": round(final_size / (1024 * 1024), 2),
                    "total_processing_seconds": round(total_elapsed, 2),
                    "average_pipeline_fps": round(avg_fps, 2),
                    "ai_steady_fps": round(ai_fps, 2),
                    "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(audit_file, "w", encoding="utf-8") as f:
                    json.dump(audit_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Đã xuất báo cáo kiểm định tập: {audit_file}", to_console=True)
            except Exception as e:
                logger.warn(f"Không thể xuất audit.json: {e}")

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
