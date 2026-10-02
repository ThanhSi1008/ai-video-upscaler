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
import shutil
import subprocess
from glob import glob
from typing import Optional

from src.config import (
    validate_environment,
    DEFAULT_MODEL,
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
from src.ai_engine import upscale_frames
from src.logger import (
    format_time,
    render_progress,
    RunLogger
)
from src.preflight import (
    CaffeinateContext,
    run_preflight_checks,
    send_macos_notification
)


def run_upscale_pipeline(
    input_path: str,
    output_path: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    bitrate: str = DEFAULT_BITRATE,
    segment_time: int = DEFAULT_SEGMENT_TIME,
    fmt: str = DEFAULT_FRAME_FORMAT,
    use_ramdisk: bool = True,
    resume: bool = True,
    caffeinate: bool = True,
    preview_seconds: Optional[int] = None
) -> str:
    """Điều phối toàn bộ quy trình Upscale 4K trên Apple Silicon M3 Pro"""
    validate_environment()

    # Chuẩn hóa đường dẫn đầu vào
    input_path = os.path.abspath(os.path.expanduser(input_path))
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file video nguồn: '{input_path}'")
        sys.exit(1)

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

    # Xác định đường dẫn file thành phẩm
    if not output_path:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base_name}_4K.mkv")
    else:
        output_path = os.path.abspath(os.path.expanduser(output_path))
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, f"{base_name}_4K.mkv")
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
                "model": model,
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

            # -----------------------------------------------------------------
            # BƯỚC 1: PHÂN ĐOẠN VIDEO THÔ (LOSSLESS STREAM COPY)
            # -----------------------------------------------------------------
            existing_segs = sorted(glob(os.path.join(segs_dir, "seg_*.mp4")))
            if not existing_segs or not resume:
                logger.info(f"Đang phân đoạn video thành các block {segment_time}s (Stream Copy)...", to_console=True)
                existing_segs = split_video_segments(input_path, segs_dir, segment_time)

            total_segs = len(existing_segs)
            logger.info(f"Đã chuẩn bị {total_segs} phân đoạn. Bắt đầu quá trình Upscale GPU Metal...", to_console=True)
            print()

            # -----------------------------------------------------------------
            # BƯỚC 2: XỬ LÝ UPSCALE & MÃ HÓA PHÂN ĐOẠN CUỐN CHIẾU
            # -----------------------------------------------------------------
            start_total_time = time.time()
            total_frames_processed = 0
            total_ai_time = 0.0

            for idx, seg_file in enumerate(existing_segs):
                seg_out_file = os.path.join(up_segs_dir, f"up_{idx:04d}.mp4")

                # Checkpoint Resume: Bỏ qua phân đoạn đã hoàn thành trước đó
                if resume and os.path.exists(seg_out_file) and os.path.getsize(seg_out_file) > 10000:
                    logger.info(f"Phân đoạn [{idx+1:02d}/{total_segs:02d}] đã hoàn thành từ trước, tiếp tục...")
                    print(f"⏭️  Phân đoạn [{idx+1:02d}/{total_segs:02d}] đã hoàn thành từ trước, tiếp tục...")
                    continue

                seg_start_time = time.time()
                os.makedirs(frames_in_dir, exist_ok=True)
                os.makedirs(frames_out_dir, exist_ok=True)

                try:
                    # 2.1 Trích xuất khung hình vào RAM Disk
                    t_ext_start = time.time()
                    frame_count = extract_frames(seg_file, frames_in_dir, fmt=fmt)
                    t_ext_elapsed = time.time() - t_ext_start
                    if frame_count == 0:
                        continue

                    total_frames_processed += frame_count

                    # 2.2 Suy luận AI bằng NCNN Vulkan trên GPU M3 Pro
                    render_progress(
                        idx, total_segs,
                        prefix=f"🚀 Đoạn {idx+1:02d}/{total_segs:02d}",
                        suffix="AI Metal Inferencing..."
                    )
                    ai_result = upscale_frames(
                        input_dir=frames_in_dir,
                        output_dir=frames_out_dir,
                        model=model,
                        scale=DEFAULT_SCALE,
                        fmt=fmt,
                        num_frames=frame_count
                    )
                    total_ai_time += ai_result["elapsed_seconds"]

                    # 2.3 Nén phần cứng 10-bit qua Apple VideoToolbox
                    render_progress(
                        idx, total_segs,
                        prefix=f"🚀 Đoạn {idx+1:02d}/{total_segs:02d}",
                        suffix="Hardware Encoding 10-bit..."
                    )
                    t_enc_start = time.time()
                    encode_chunk_hevc(
                        frames_dir=frames_out_dir,
                        fps=meta["fps"],
                        output_path=seg_out_file,
                        bitrate=bitrate,
                        fmt=fmt
                    )
                    t_enc_elapsed = time.time() - t_enc_start

                    # 2.4 Cập nhật thống kê và ETA thực tế
                    seg_elapsed = time.time() - seg_start_time
                    seg_fps = (frame_count / seg_elapsed) if seg_elapsed > 0 else 0.0
                    remaining_segs = total_segs - (idx + 1)
                    eta_seconds = remaining_segs * seg_elapsed

                    render_progress(
                        idx + 1, total_segs,
                        prefix=f"✨ Đoạn {idx+1:02d}/{total_segs:02d}",
                        suffix=f"[{seg_fps:.2f} fps (AI: {ai_result['fps']:.2f}) | ETA: {format_time(eta_seconds)}]"
                    )
                    print()

                    # Ghi log phân đoạn
                    logger.log_chunk_done(
                        chunk_idx=idx,
                        total_chunks=total_segs,
                        frames=frame_count,
                        ai_time=ai_result["elapsed_seconds"],
                        ai_fps=ai_result["fps"],
                        enc_time=t_enc_elapsed,
                        total_chunk_time=seg_elapsed,
                        output_seg=seg_out_file
                    )

                finally:
                    # Xóa toàn bộ ảnh đệm trong RAM Disk ngay lập tức
                    shutil.rmtree(frames_in_dir, ignore_errors=True)
                    shutil.rmtree(frames_out_dir, ignore_errors=True)

            logger.info("Hoàn tất quá trình suy luận 4K cho toàn bộ phân đoạn!", to_console=True)

            # -----------------------------------------------------------------
            # BƯỚC 3: GHÉP NỐI KHÔNG MẤT DỮ LIỆU CÁC PHÂN ĐOẠN 4K
            # -----------------------------------------------------------------
            logger.info("Đang ghép nối các phân đoạn 4K...", to_console=True)
            all_upscaled_segs = sorted(glob(os.path.join(up_segs_dir, "up_*.mp4")))
            merged_video = os.path.join(work_dir, "merged_4k.mp4")
            concat_segments(all_upscaled_segs, work_dir, merged_video)

            # -----------------------------------------------------------------
            # BƯỚC 4: MUXING ĐÓNG GÓI NGUYÊN BẢN AUDIO, SUBTITLES & CHAPTERS
            # -----------------------------------------------------------------
            logger.info("Đang đóng gói lại Audio, Subtitles, Fonts và Chapters gốc vào file 4K hoàn chỉnh...", to_console=True)
            mux_final_container(merged_video, input_path, output_path)

            # Dọn dẹp thư mục trung gian trên SSD
            shutil.rmtree(work_dir, ignore_errors=True)
            if preview_temp_input and os.path.exists(preview_temp_input):
                os.remove(preview_temp_input)

            total_elapsed = time.time() - start_total_time
            final_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            avg_fps = (total_frames_processed / total_elapsed) if total_elapsed > 0 and total_frames_processed > 0 else 0.0
            ai_fps = (total_frames_processed / total_ai_time) if total_ai_time > 0 and total_frames_processed > 0 else 0.0

            summary_stats = {
                "output_path": output_path,
                "final_size_bytes": final_size,
                "total_seconds": total_elapsed,
                "total_frames": total_frames_processed,
                "avg_fps": avg_fps,
                "ai_fps": ai_fps,
                "ssd_protection": "100% An toàn (Frame trung gian chỉ chạy trên RAM Disk)"
                if use_ramdisk else "Ghi tạm trên SSD"
            }
            logger.print_benchmark_summary(summary_stats)

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
