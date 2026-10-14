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
# THAM SỐ UPSCALE MẶC ĐỊNH (TỰ ĐỘNG NHẬN DIỆN NVIDIA CUDA / APPLE MPS)
# -----------------------------------------------------------------------------
def _detect_default_engine() -> str:
    """Tự động chọn engine suy luận tốt nhất theo phần cứng hiện có."""
    try:
        import torch
        if torch.cuda.is_available():
            # Kích hoạt tối ưu hóa NVIDIA Tensor Cores & cuDNN benchmark
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_device_name() -> str:
    """Lấy tên card đồ họa hoặc thiết bị đang được sử dụng."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon GPU (Metal)"
    except Exception:
        pass
    return "CPU"


DEFAULT_MODEL = "realesr-animevideov3"
DEFAULT_ENGINE = _detect_default_engine()  # Mặc định: 'cuda' trên NVIDIA, 'mps' trên Apple Silicon
DEVICE_NAME = get_device_name()
DEFAULT_PRECISION = "fp16"
DEFAULT_BATCH_SIZE = 6 if DEFAULT_ENGINE == "cuda" else 1  # Lô 6 frames tối ưu cho 12GB VRAM RTX 3060 (đo thực nghiệm 1.37 fps @ 6.20 GB VRAM)
DEFAULT_MPS_RESTART_CHUNKS = 10  # Khởi động lại worker mỗi N chunks để giải phóng driver pool (A2)
DEFAULT_SCALE = 2
DEFAULT_BITRATE = "28M"
# Phân đoạn 8s (~192 frames) là kích thước lý tưởng cho PNG Lossless:
DEFAULT_SEGMENT_TIME = 8  # giây
DEFAULT_FRAME_FORMAT = "png"  # Mặc định PNG Lossless (RGB24, không nén suy hao)
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "out") if sys.platform == "win32" else os.path.expanduser("~/Documents/Upscaled")

# -----------------------------------------------------------------------------
# CẤU HÌNH DEDUPLICATION (PREFLIGHT SCAN & CONCAT RE-EXPANSION - MỤC TIÊU 4 GIỜ)
# -----------------------------------------------------------------------------
DEFAULT_DEDUP_HI = 768
DEFAULT_DEDUP_LO = 320
DEFAULT_DEDUP_FRAC = 0.33
DEFAULT_DEDUP_ENABLED = True

# -----------------------------------------------------------------------------
# CẤU HÌNH BỘ NHỚ ĐỆM RAM (ZERO NVME SSD WEAR)
# -----------------------------------------------------------------------------
DEFAULT_RAMDISK_SIZE_MB = 5120  # 5 GB RAM Disk ảo trong Unified Memory (đủ đệm an toàn chunk 192 frames 4K ~3.1GB + input)
DEFAULT_PNG_UNIT_SIZE_MB = 16.0  # Sàn ước tính an toàn dung lượng frame 4K PNG (RGB24 3840x2160 cho Anime ~14-16MB)

# -----------------------------------------------------------------------------
# CẤU HÌNH BỘ MÃ HÓA PHẦN CỨNG HEVC 10-BIT (NVENC / VIDEOTOOLBOX / QSV / X265)
# -----------------------------------------------------------------------------
def _detect_hevc_encoder() -> str:
    """Tự động nhận diện bộ mã hóa phần cứng HEVC 10-bit tốt nhất."""
    if sys.platform == "darwin":
        return "hevc_videotoolbox"
    # Trên Windows / Linux: ưu tiên NVIDIA NVENC, sau đó Intel QSV, cuối cùng libx265
    try:
        import subprocess
        res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
        out = res.stdout or ""
        if "hevc_nvenc" in out:
            return "hevc_nvenc"
        if "hevc_qsv" in out:
            return "hevc_qsv"
        if "libx265" in out:
            return "libx265"
    except Exception:
        pass
    return "hevc_nvenc" if sys.platform == "win32" else "hevc_videotoolbox"


HEVC_ENCODER = _detect_hevc_encoder()
HEVC_PRESET = "p6" if HEVC_ENCODER == "hevc_nvenc" else "p5"  # Preset p6 (Slower/Better Quality) cho RTX NVENC
HEVC_CQ_LEVEL = "19"  # Constant Quality cho NVENC (visually lossless, thích ứng theo cảnh phim)
HEVC_PROFILE = "main10"
HEVC_PIX_FMT = "p010le"
HEVC_COLOR_RANGE = "tv"
HEVC_COLORSPACE = "bt709"
HEVC_COLOR_PRIMARIES = "bt709"
HEVC_COLOR_TRC = "bt709"
HEVC_TAG = "hvc1"


def validate_environment(engine: str = DEFAULT_ENGINE):
    """Kiểm tra sự hiện diện của FFmpeg, FFprobe và binary/engine tương ứng."""
    if shutil.which("ffmpeg") is None:
        print("❌ Lỗi: Không tìm thấy 'ffmpeg' trên hệ thống!")
        print("💡 Cài đặt qua Homebrew: brew install ffmpeg (macOS) hoặc tải từ gyan.dev (Windows)")
        sys.exit(1)

    if shutil.which("ffprobe") is None:
        print("❌ Lỗi: Không tìm thấy 'ffprobe' trên hệ thống!")
        print("💡 Cài đặt qua Homebrew: brew install ffmpeg (macOS) hoặc tải từ gyan.dev (Windows)")
        sys.exit(1)

    # Chỉ kiểm tra binary NCNN nếu người dùng chủ động chọn engine ncnn
    if engine == "ncnn":
        if not os.path.exists(REALESRGAN_BIN):
            print(f"❌ Lỗi: Không tìm thấy binary NCNN tại: {REALESRGAN_BIN}")
            sys.exit(1)

        if sys.platform != "win32" and not os.access(REALESRGAN_BIN, os.X_OK):
            try:
                os.chmod(REALESRGAN_BIN, 0o755)
            except Exception as e:
                print(f"❌ Không thể cấp quyền thực thi cho binary: {e}")
                sys.exit(1)

