#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 AI Video Upscaler 4K - CLI Entry Point
Công cụ Upscale Video Anime lên 4K chuyên dụng trên Terminal cho macOS (Apple Silicon M3 Pro).
- Tăng độ phân giải 2x bằng NCNN Vulkan Metal API (realesr-animevideov3).
- Mã hóa phần cứng 10-bit HEVC qua Apple Media Engine (VideoToolbox).
- Bảo vệ tối đa ổ cứng NVMe SSD với bộ nhớ đệm RAM Disk ảo.
- Bảo toàn 100% âm thanh đa kênh, phụ đề đa ngôn ngữ, font chữ và chapters gốc.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

from src.config import (
    DEFAULT_MODEL,
    DEFAULT_ENGINE,
    DEFAULT_PRECISION,
    DEFAULT_BITRATE,
    DEFAULT_SEGMENT_TIME,
    DEFAULT_FRAME_FORMAT
)
from src.pipeline import run_upscale_pipeline
from src.ffmpeg_ops import probe_video
from src.dedup import spawn_background_preflight
from src.models_resolver import list_models_formatted
from pathlib import Path


def check_deadline_and_eta(video_path: str, deadline_str: str, last_measured_fps: float = 1.75) -> bool:
    """Kiểm tra thời gian hoàn thành dự toán của tập video so với mốc deadline (A6)."""
    now = datetime.now()
    try:
        dh, dm = map(int, deadline_str.split(":"))
        deadline_dt = now.replace(hour=dh, minute=dm, second=0, microsecond=0)
        if deadline_dt <= now:
            deadline_dt += timedelta(days=1)
    except Exception as e:
        print(f"⚠️ Định dạng deadline không hợp lệ '{deadline_str}' (cần HH:MM): {e}")
        return True

    seconds_left = (deadline_dt - now).total_seconds()

    # Đọc số frame độc nhất từ plan.json nếu có
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    candidate_plans = [
        os.path.splitext(video_path)[0] + ".plan.json",
        f"out/{base_name}.plan.json",
        f"out/.temp_{base_name}/plan.json"
    ]
    n_unique = None
    for p in candidate_plans:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                    n_unique = pdata.get("total_kept_frames")
                    if n_unique:
                        break
            except Exception:
                pass

    if n_unique is None:
        # Ước lượng 40% của tổng số frames nếu chưa có plan.json
        try:
            meta = probe_video(video_path)
            total_f = meta.get("nb_frames") or int(meta.get("duration", 1440) * 23.976)
            n_unique = int(total_f * 0.40)
        except Exception:
            n_unique = 13500

    # Tính toán ETA: n_unique / fps đo gần nhất + 300s overhead (A6)
    fps_to_use = max(0.5, last_measured_fps)
    eta_seconds = (n_unique / fps_to_use) + 300.0

    if eta_seconds > seconds_left:
        print(f"\n⏰ [DEADLINE SKIP] Tập '{os.path.basename(video_path)}':")
        print(f"   • Dự toán cần: {eta_seconds/60:.1f} phút ({eta_seconds/3600:.2f}h cho ~{n_unique} unique frames @ {fps_to_use:.2f} fps)")
        print(f"   • Thời gian còn lại trước {deadline_str}: {seconds_left/60:.1f} phút ({seconds_left/3600:.2f}h)")
        print(f"   -> Bỏ qua tập này để không bị dở dang ca qua đêm!\n")
        return False

    print(f"⏳ [DEADLINE OK] Tập '{os.path.basename(video_path)}' cần ~{eta_seconds/60:.1f}m (@ {fps_to_use:.2f} fps), còn {seconds_left/60:.1f}m trước hạn chót {deadline_str}.")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="🎬 AI Video Upscaler 4K - Tối ưu hóa cho Apple Silicon M3 Pro (PyTorch MPS / NCNN Metal + VideoToolbox 10-bit)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Đường dẫn đến file video nguồn (.mkv, .mp4, .webm)"
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
        "-o", "--output",
        default=None,
        help="Đường dẫn file hoặc thư mục xuất 4K (Mặc định: ~/Documents/Upscaled/<tên>_4K.mkv)"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"Mô hình AI upscale (Mặc định: tự động chọn qua shows.yaml hoặc mặc định animevideov3)"
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        choices=["mps", "ncnn"],
        help=f"Động cơ suy luận AI (Mặc định: {DEFAULT_ENGINE} - PyTorch MPS FP16)"
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
        "--no-caffeinate",
        action="store_true",
        help="Không tự động kích hoạt caffeinate để chống sleep máy trong khi chạy"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Không sử dụng bộ lọc frame trùng (Deduplication) mà tính toán 100%% tất cả các khung hình"
    )
    parser.add_argument(
        "--preview",
        type=int,
        nargs="?",
        const=10,
        default=None,
        metavar="SECONDS",
        help="Chế độ xem trước: trích xuất nhanh N giây (mặc định: 10s) ở giữa phim, upscale và tự mở bằng IINA"
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

    return parser.parse_args()


def find_or_extract_subtitles(video_path: str) -> tuple[Optional[str], Optional[str]]:
    """Tìm hoặc trích xuất phụ đề tiếng Nhật (ja) và tiếng Anh (en) cho video."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    vid_dir = os.path.dirname(os.path.abspath(video_path))

    ja_sub = None
    en_sub = None

    # 1. Tìm các file phụ đề sẵn có cùng thư mục
    candidates_ja = [
        os.path.join(vid_dir, f"{base_name}.ja.ass"),
        os.path.join(vid_dir, f"{base_name}_ja.ass"),
        os.path.join(vid_dir, f"{base_name}_jpn.ass"),
        os.path.join(vid_dir, f"{base_name}.ass"),
        f"subtrans/samples/{base_name}_jpn.ass",
        f"subtrans/samples/{base_name}.ass",
    ]
    for c in candidates_ja:
        if os.path.exists(c):
            ja_sub = c
            break

    candidates_en = [
        os.path.join(vid_dir, f"{base_name}.en.ass"),
        os.path.join(vid_dir, f"{base_name}_en.ass"),
        os.path.join(vid_dir, f"{base_name}_eng.ass"),
        f"subtrans/samples/{base_name}_eng.ass",
    ]
    for c in candidates_en:
        if os.path.exists(c):
            en_sub = c
            break

    # 2. Nếu chưa có ja_sub, thử trích xuất từ container video qua ffmpeg
    if not ja_sub:
        try:
            meta = probe_video(video_path)
            if meta.get("subtitle_count", 0) > 0:
                tmp_dir = os.path.join(vid_dir, f".temp_{base_name}")
                os.makedirs(tmp_dir, exist_ok=True)
                extracted_ja = os.path.join(tmp_dir, "extracted_ja.ass")
                # Trích xuất stream subtitle đầu tiên
                cmd = ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", "-c:s", "copy", extracted_ja]
                r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if r.returncode == 0 and os.path.exists(extracted_ja) and os.path.getsize(extracted_ja) > 100:
                    ja_sub = extracted_ja
        except Exception:
            pass

    return ja_sub, en_sub


def main():
    args = parse_args()

    # Kiểm tra cờ liệt kê mô hình (--list-models)
    if args.list_models:
        print(list_models_formatted())
        sys.exit(0)

    # Xác định danh sách video cần xử lý
    videos = []
    if args.queue:
        videos.extend(args.queue)
    if args.input:
        videos.append(args.input)

    if not videos:
        print("❌ Lỗi: Vui lòng cung cấp ít nhất một file video đầu vào hoặc dùng cờ --queue.")
        sys.exit(1)

    print(f"🎬 Hàng đợi xử lý gồm {len(videos)} tập video.")
    last_measured_fps = 1.75 if args.engine == "mps" else 1.35
    completed_videos = []

    for idx, vid in enumerate(videos):
        print(f"\n{'='*75}")
        print(f"▶ [{idx+1}/{len(videos)}] Bắt đầu xử lý: {os.path.basename(vid)}")
        print(f"{'='*75}")

        # Kiểm tra deadline nếu có (A6)
        if args.deadline:
            can_finish = check_deadline_and_eta(vid, args.deadline, last_measured_fps)
            if not can_finish:
                print(f"⏹️ Dừng hàng đợi tại tập {idx+1}/{len(videos)} do chạm ngưỡng hạn chót {args.deadline}.")
                break

        # Kích hoạt preflight nền cho tập tiếp theo nếu có (nice -n 10) (A6)
        if idx + 1 < len(videos) and not args.no_dedup:
            spawn_background_preflight(
                video_path=videos[idx + 1],
                output_dir=args.output,
                segment_seconds=args.segment
            )

        try:
            out_file = run_upscale_pipeline(
                input_path=vid,
                output_path=args.output,
                model=args.model,
                engine=args.engine,
                precision=args.precision,
                force=args.force,
                bitrate=args.bitrate,
                segment_time=args.segment,
                fmt=args.format,
                use_ramdisk=not args.no_ramdisk,
                resume=not args.no_resume,
                caffeinate=not args.no_caffeinate,
                preview_seconds=args.preview,
                dedup=not args.no_dedup
            )
            completed_videos.append((vid, out_file))

            # Cập nhật last_measured_fps từ audit.json nếu có
            audit_path = os.path.splitext(out_file)[0] + ".audit.json"
            if os.path.exists(audit_path):
                try:
                    with open(audit_path, "r", encoding="utf-8") as f:
                        adata = json.load(f)
                    measured_fps = adata.get("ai_steady_fps")
                    if measured_fps and measured_fps > 0:
                        last_measured_fps = float(measured_fps)
                except Exception:
                    pass

        except KeyboardInterrupt:
            print("\n\n⚠️ Đã tạm dừng quá trình theo yêu cầu (Ctrl+C).")
            print("🛡️  RAM Disk và các file đệm đã được tự động giải phóng an toàn.")
            print("💡 Lần sau khi chạy lại cùng video này, hệ thống sẽ tự động tiếp tục từ phân đoạn đang dở!\n")
            sys.exit(130)
        except Exception as e:
            print(f"\n❌ Đã xảy ra lỗi khi xử lý tập '{vid}': {e}\n")
            if len(videos) > 1:
                print("Chuyển sang tập tiếp theo trong hàng đợi...\n")
                continue
            else:
                sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE B: DỊCH PHỤ ĐỀ (CHỈ CHẠY SAU KHI CẢ HÀNG ĐỢI UPSCALE XONG - B1, B3)
    # -------------------------------------------------------------------------
    if args.subs_after_upscale and completed_videos:
        print(f"\n{'='*75}")
        print("📝 [STAGE B] BẮT ĐẦU DỊCH PHỤ ĐỀ CHO CÁC TẬP ĐÃ HOÀN TẤT UPSCALE")
        print(f"{'='*75}")
        from subtrans.pipeline import TranslationPipeline

        for src_vid, out_vid in completed_videos:
            ja_sub, en_sub = find_or_extract_subtitles(src_vid)
            if not ja_sub:
                print(f"ℹ️ Không tìm thấy phụ đề tiếng Nhật cho '{os.path.basename(src_vid)}', bỏ qua bước dịch phụ đề.")
                continue

            print(f"\n🌐 Bắt đầu dịch phụ đề cho tập: {os.path.basename(src_vid)}")
            print(f"   • File gốc JA: {ja_sub}")
            if en_sub:
                print(f"   • File tham chiếu EN: {en_sub}")

            try:
                sub_out_dir = Path(os.path.dirname(out_vid))
                sub_pipeline = TranslationPipeline(
                    local_only=False,
                    output_dir=sub_out_dir,
                    output_format="srt",
                    require_approved_cast=False,
                    allow_latest_alias=False
                )
                sub_pipeline.run(
                    ja_path=Path(ja_sub),
                    en_path=Path(en_sub) if en_sub else None
                )
                print(f"✅ Hoàn tất dịch phụ đề cho tập '{os.path.basename(src_vid)}'.")
            except Exception as se:
                print(f"⚠️ Lỗi trong quá trình dịch phụ đề tập '{os.path.basename(src_vid)}': {se}")

    print("\n🎉 Toàn bộ công việc trong hàng đợi đã hoàn tất thành công!")


if __name__ == "__main__":
    main()

