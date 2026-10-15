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
# THAM SỐ UPSCALE MẶC ĐỊNH (TỐI ƯU HÓA NVIDIA CUDA FP16 & TENSOR CORES)
# -----------------------------------------------------------------------------
def _detect_default_engine() -> str:
    """Tự động chọn engine suy luận tốt nhất theo phần cứng hiện có (Ưu tiên CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            # Kích hoạt tối ưu hóa NVIDIA Tensor Cores & cuDNN benchmark
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_device_name() -> str:
    """Lấy tên card đồ họa NVIDIA hoặc CPU đang được sử dụng."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


DEFAULT_MODEL = "realesr-animevideov3"
DEFAULT_ENGINE = _detect_default_engine()  # Mặc định: 'cuda' trên NVIDIA RTX
DEVICE_NAME = get_device_name()
DEFAULT_PRECISION = "fp16"
DEFAULT_BATCH_SIZE = 8 if DEFAULT_ENGINE == "cuda" else 1  # Lô 8 frames (Điểm ngọt tối ưu cho GA106 30 SMs trên RTX 3060 12GB)
DEFAULT_VRAM_CLEANUP_THRESHOLD_GB = 9.5  # Ngưỡng VRAM (GB) để dọn dẹp CUDA memory cache thay vì restart tiến trình
DEFAULT_GPU_RESTART_CHUNKS = 10  # Ngưỡng dự phòng cho worker
DEFAULT_MPS_RESTART_CHUNKS = DEFAULT_GPU_RESTART_CHUNKS  # Alias tương thích ngược
DEFAULT_SCALE = 2
DEFAULT_BITRATE = "28M"
# Phân đoạn 8s (~192 frames) là kích thước lý tưởng cho PNG Lossless:
DEFAULT_SEGMENT_TIME = 8  # giây
DEFAULT_FRAME_FORMAT = "png"  # Mặc định PNG Lossless (RGB24, không nén suy hao)
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "out")

# -----------------------------------------------------------------------------
# CẤU HÌNH DEDUPLICATION (PREFLIGHT SCAN & CONCAT RE-EXPANSION)
# -----------------------------------------------------------------------------
DEFAULT_DEDUP_HI = 768
DEFAULT_DEDUP_LO = 320
DEFAULT_DEDUP_FRAC = 0.33
DEFAULT_DEDUP_ENABLED = True

# -----------------------------------------------------------------------------
# CẤU HÌNH BỘ NHỚ ĐỆM RAM & PHÂN ĐOẠN CUỐN CHIẾU
# -----------------------------------------------------------------------------
DEFAULT_RAMDISK_SIZE_MB = 5120
DEFAULT_PNG_UNIT_SIZE_MB = 16.0  # Sàn ước tính an toàn dung lượng frame 4K PNG (RGB24 3840x2160 cho Anime ~14-16MB)

# -----------------------------------------------------------------------------
# CẤU HÌNH BỘ MÃ HÓA PHẦN CỨNG HEVC 10-BIT (NVIDIA NVENC)
# -----------------------------------------------------------------------------
def _detect_hevc_encoder() -> str:
    """Tự động nhận diện bộ mã hóa phần cứng HEVC 10-bit cho Windows (NVIDIA NVENC)."""
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
    return "hevc_nvenc"


HEVC_ENCODER = _detect_hevc_encoder()
HEVC_PRESET = "p6"  # Preset p6 (Slower/Better Quality) cho RTX NVENC
HEVC_CQ_LEVEL = "18"  # Constant Quality CQ 18 cho NVENC (visually lossless chuẩn Blu-ray, giảm 35-45% file size cảnh tĩnh)
HEVC_MAXRATE = "45M"  # Trần bitrate cho các cảnh hành động/chiến đấu nhiều hạt
HEVC_BUFSIZE = "90M"  # Bộ đệm rate-control
HEVC_MULTIPASS = "fullres"  # Khai thác 100% ASIC NVENC chuyên dụng (không tốn CUDA cores)
HEVC_RC_LOOKAHEAD = 32  # Lookahead tối đa 32 frames để NVENC tối ưu hóa Coding Units (CU) 8x8
HEVC_SPATIAL_AQ = 1  # Adaptive Quantization không gian
HEVC_AQ_STRENGTH = 6  # Hạ từ 8 -> 6 để bảo vệ độ sắc bén của nét vẽ anime từ realesr-animevideov3
HEVC_TEMPORAL_AQ = 1  # Adaptive Quantization thời gian (khóa chất lượng phông nền tĩnh anime)
HEVC_BF = 4  # Số khung hình B tối đa
HEVC_B_REF_MODE = "middle"  # B-frame reference mode middle giúp tăng hiệu quả nén cảnh chuyển động nhẹ
HEVC_PROFILE = "main10"
HEVC_PIX_FMT = "p010le"
HEVC_COLOR_RANGE = "tv"
HEVC_COLORSPACE = "bt709"
HEVC_COLOR_PRIMARIES = "bt709"
HEVC_COLOR_TRC = "bt709"
HEVC_TAG = "hvc1"


def get_hevc_rate_args(bitrate: str = "28M") -> list:
    """
    Trả về bộ tham số rate-control tối ưu cho HEVC Main 10 theo phần cứng hiện hành.
    Trên NVIDIA NVENC: áp dụng cấu hình CQ 18 + fullres + lookahead 32 + AQ kép chuyên biệt cho Anime 4K.
    """
    if HEVC_ENCODER == "hevc_nvenc":
        return [
            "-preset", HEVC_PRESET,
            "-tune", "hq",
            "-multipass", HEVC_MULTIPASS,
            "-rc", "vbr",
            "-cq", str(HEVC_CQ_LEVEL),
            "-b:v", "0",
            "-maxrate", HEVC_MAXRATE,
            "-bufsize", HEVC_BUFSIZE,
            "-rc-lookahead", str(HEVC_RC_LOOKAHEAD),
            "-spatial-aq", str(HEVC_SPATIAL_AQ),
            "-aq-strength", str(HEVC_AQ_STRENGTH),
            "-temporal-aq", str(HEVC_TEMPORAL_AQ),
            "-bf", str(HEVC_BF),
            "-b_ref_mode", HEVC_B_REF_MODE,
        ]
    return [
        "-b:v", bitrate,
        "-spatial_aq", "1"
    ]


def validate_environment(engine: str = DEFAULT_ENGINE):
    """Kiểm tra sự hiện diện của FFmpeg, FFprobe trên Windows."""
    if shutil.which("ffmpeg") is None:
        print("❌ Lỗi: Không tìm thấy 'ffmpeg' trên hệ thống!")
        print("💡 Vui lòng tải FFmpeg (Gyan.dev Full Build) và thêm vào biến môi trường PATH.")
        sys.exit(1)

    if shutil.which("ffprobe") is None:
        print("❌ Lỗi: Không tìm thấy 'ffprobe' trên hệ thống!")
        print("💡 Vui lòng tải FFmpeg (Gyan.dev Full Build) và thêm vào biến môi trường PATH.")
        sys.exit(1)

