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
from datetime import datetime, timedelta

from src.config import (
    DEFAULT_MODEL,
    DEFAULT_BITRATE,
    DEFAULT_SEGMENT_TIME,
    DEFAULT_FRAME_FORMAT
)
from src.pipeline import run_upscale_pipeline
from src.ffmpeg_ops import probe_video


def check_deadline_and_eta(video_path: str, deadline_str: str) -> bool:
    """Kiểm tra thời gian hoàn thành dự toán của tập video so với mốc deadline."""
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

    # Tính toán ETA: @ 1.25 fps + 300s overhead (preflight, decode, encode, mux)
    eta_seconds = (n_unique / 1.25) + 300.0

    if eta_seconds > seconds_left:
        print(f"\n⏰ [DEADLINE SKIP] Tập '{os.path.basename(video_path)}':")
        print(f"   • Dự toán cần: {eta_seconds/60:.1f} phút ({eta_seconds/3600:.2f}h cho ~{n_unique} unique frames)")
        print(f"   • Thời gian còn lại trước {deadline_str}: {seconds_left/60:.1f} phút ({seconds_left/3600:.2f}h)")
        print(f"   -> Bỏ qua tập này để không bị dở dang ca qua đêm!\n")
        return False

    print(f"⏳ [DEADLINE OK] Tập '{os.path.basename(video_path)}' cần ~{eta_seconds/60:.1f}m, còn {seconds_left/60:.1f}m trước hạn chót {deadline_str}.")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="🎬 AI Video Upscaler 4K - Tối ưu hóa cho Apple Silicon M3 Pro (Vulkan / Metal + VideoToolbox 10-bit)",
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
        default=DEFAULT_MODEL,
        choices=["realesr-animevideov3", "realesrgan-x4plus-anime", "realesrgan-x4plus"],
        help=f"Mô hình AI upscale (Mặc định: {DEFAULT_MODEL})"
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

    return parser.parse_args()


def main():
    args = parse_args()

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

    for idx, vid in enumerate(videos):
        print(f"\n{'='*75}")
        print(f"▶ [{idx+1}/{len(videos)}] Bắt đầu xử lý: {os.path.basename(vid)}")
        print(f"{'='*75}")

        # Kiểm tra deadline nếu có
        if args.deadline:
            can_finish = check_deadline_and_eta(vid, args.deadline)
            if not can_finish:
                print(f"⏹️ Dừng hàng đợi tại tập {idx+1}/{len(videos)} do chạm ngưỡng hạn chót {args.deadline}.")
                break

        try:
            run_upscale_pipeline(
                input_path=vid,
                output_path=args.output,
                model=args.model,
                bitrate=args.bitrate,
                segment_time=args.segment,
                fmt=args.format,
                use_ramdisk=not args.no_ramdisk,
                resume=not args.no_resume,
                caffeinate=not args.no_caffeinate,
                preview_seconds=args.preview,
                dedup=not args.no_dedup
            )
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


if __name__ == "__main__":
    main()

