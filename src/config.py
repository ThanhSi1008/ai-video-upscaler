# -*- coding: utf-8 -*-
"""
src/config.py
Quản lý cấu hình tập trung, đường dẫn binary, thông số mã hóa phần cứng và RAM Disk.
"""

import os
import shutil
import sys
import json

# -----------------------------------------------------------------------------
# ĐƯỜNG DẪN DỰ ÁN & BINARY
# -----------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

BIN_DIR = os.path.join(ROOT_DIR, "bin", "realesrgan")
REALESRGAN_BIN = os.path.join(BIN_DIR, "realesrgan-ncnn-vulkan")
MODELS_DIR = os.path.join(BIN_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
GPU_LOCK_FILE = os.path.join(ROOT_DIR, ".gpu_active.lock")
PENDING_SUBS_FILE = os.path.join(ROOT_DIR, "pending_subs.json")


def load_pending_subs() -> list:
    """Đọc danh sách các tập phụ đề chưa đạt chuẩn (Item 4b)."""
    if not os.path.exists(PENDING_SUBS_FILE):
        return []
    try:
        with open(PENDING_SUBS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def add_pending_sub(record: dict) -> None:
    """Thêm hoặc cập nhật một tập vào pending_subs.json."""
    if os.environ.get("PYTEST_CURRENT_TEST") and "dummy" in record.get("base_name", ""):
        return
    items = load_pending_subs()
    # Loại bỏ bản ghi cũ cùng base_name nếu có
    items = [it for it in items if it.get("base_name") != record.get("base_name")]
    items.append(record)
    try:
        with open(PENDING_SUBS_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def remove_pending_sub(base_name: str) -> None:
    """Xóa một tập khỏi pending_subs.json khi đã đạt chuẩn."""
    items = load_pending_subs()
    new_items = [it for it in items if it.get("base_name") != base_name]
    if len(new_items) != len(items):
        try:
            with open(PENDING_SUBS_FILE, "w", encoding="utf-8") as f:
                json.dump(new_items, f, ensure_ascii=False, indent=2)
        except Exception:
            pass



def acquire_gpu_lock() -> None:
    """Tạo file khóa đánh dấu giai đoạn upscale GPU đang hoạt động."""
    try:
        with open(GPU_LOCK_FILE, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n")
    except Exception:
        pass


def release_gpu_lock() -> None:
    """Giải phóng file khóa giai đoạn upscale GPU."""
    try:
        if os.path.exists(GPU_LOCK_FILE):
            os.remove(GPU_LOCK_FILE)
    except Exception:
        pass


import atexit
atexit.register(release_gpu_lock)



# -----------------------------------------------------------------------------
# THAM SỐ UPSCALE MẶC ĐỊNH CHO ANIME & APPLE SILICON (M3 PRO)
# -----------------------------------------------------------------------------
DEFAULT_MODEL = "realesr-animevideov3"
DEFAULT_ENGINE = "mps"  # Mặc định: PyTorch MPS FP16 (A1)
DEFAULT_PRECISION = "fp16"
DEFAULT_MPS_RESTART_CHUNKS = 10  # Khởi động lại worker mỗi N chunks để giải phóng driver pool (A2)
DEFAULT_SCALE = 2
DEFAULT_BITRATE = "28M"
# Phân đoạn 8s (~192 frames) là kích thước lý tưởng cho PNG Lossless:
# 192 frames 1080p (~480MB) + 192 frames 4K (~1.6GB) = ~2.1GB, vừa vặn trong 4GB RAM Disk
DEFAULT_SEGMENT_TIME = 8  # giây
DEFAULT_FRAME_FORMAT = "png"  # Mặc định PNG Lossless (RGB24, không nén suy hao)
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/Documents/Upscaled")

# -----------------------------------------------------------------------------
# CẤU HÌNH DEDUPLICATION (PREFLIGHT SCAN & CONCAT RE-EXPANSION - MỤC TIÊU 4 GIỜ)
# -----------------------------------------------------------------------------
DEFAULT_DEDUP_HI = 768
DEFAULT_DEDUP_LO = 320
DEFAULT_DEDUP_FRAC = 0.33
DEFAULT_DEDUP_ENABLED = True

# -----------------------------------------------------------------------------
# CẤU HÌNH RAM DISK (ZERO NVME SSD WEAR)
# -----------------------------------------------------------------------------
DEFAULT_RAMDISK_SIZE_MB = 5120  # 5 GB RAM Disk ảo trong Unified Memory (đủ đệm an toàn chunk 192 frames 4K ~3.1GB + input)
DEFAULT_PNG_UNIT_SIZE_MB = 16.0  # Sàn ước tính an toàn dung lượng frame 4K PNG (RGB24 3840x2160 cho Anime ~14-16MB)

# -----------------------------------------------------------------------------
# CẤU HÌNH APPLE MEDIA ENGINE (VIDEOTOOLBOX HARDWARE ENCODER 10-BIT)
# -----------------------------------------------------------------------------
HEVC_ENCODER = "hevc_videotoolbox"
HEVC_PROFILE = "main10"
HEVC_PIX_FMT = "p010le"
HEVC_COLOR_RANGE = "tv"
HEVC_COLORSPACE = "bt709"
HEVC_COLOR_PRIMARIES = "bt709"
HEVC_COLOR_TRC = "bt709"
HEVC_TAG = "hvc1"


def validate_environment():
    """Kiểm tra sự hiện diện của FFmpeg, FFprobe và binary realesrgan-ncnn-vulkan"""
    if shutil.which("ffmpeg") is None:
        print("❌ Lỗi: Không tìm thấy 'ffmpeg' trên hệ thống!")
        print("💡 Cài đặt qua Homebrew: brew install ffmpeg")
        sys.exit(1)

    if shutil.which("ffprobe") is None:
        print("❌ Lỗi: Không tìm thấy 'ffprobe' trên hệ thống!")
        print("💡 Cài đặt qua Homebrew: brew install ffmpeg")
        sys.exit(1)

    if not os.path.exists(REALESRGAN_BIN):
        print(f"❌ Lỗi: Không tìm thấy binary tại: {REALESRGAN_BIN}")
        sys.exit(1)

    if not os.access(REALESRGAN_BIN, os.X_OK):
        try:
            os.chmod(REALESRGAN_BIN, 0o755)
        except Exception as e:
            print(f"❌ Không thể cấp quyền thực thi cho binary: {e}")
            sys.exit(1)
