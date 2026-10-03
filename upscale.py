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

import sys
import argparse

from src.config import (
    DEFAULT_MODEL,
    DEFAULT_BITRATE,
    DEFAULT_SEGMENT_TIME,
    DEFAULT_FRAME_FORMAT
)
from src.pipeline import run_upscale_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="🎬 AI Video Upscaler 4K - Tối ưu hóa cho Apple Silicon M3 Pro (Vulkan / Metal + VideoToolbox 10-bit)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input",
        help="Đường dẫn đến file video nguồn (.mkv, .mp4, .webm)"
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

    try:
        run_upscale_pipeline(
            input_path=args.input,
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
        print(f"\n❌ Đã xảy ra lỗi: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
