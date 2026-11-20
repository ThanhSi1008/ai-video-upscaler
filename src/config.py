# -*- coding: utf-8 -*-
"""
src/config.py
Quản lý cấu hình tập trung, đường dẫn binary, thông số mã hóa phần cứng và RAM Disk.
"""

import contextlib
import os
from pathlib import Path
import shutil
import sys
import json

# -----------------------------------------------------------------------------
# ĐƯỜNG DẪN DỰ ÁN & BINARY
# -----------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
PROJECT_ROOT = Path(ROOT_DIR)

try:
    from src.env_isolation import configure_e_drive_isolation
except ImportError:
    from env_isolation import configure_e_drive_isolation

configure_e_drive_isolation(project_root=PROJECT_ROOT)

CACHE_ROOT_DIR = PROJECT_ROOT / ".cache"
CUDA_COMPUTE_CACHE_DIR = CACHE_ROOT_DIR / "cuda_compute_cache"
TRITON_CACHE_DIR_DEFAULT = CACHE_ROOT_DIR / "triton_cache"
INDUCTOR_CACHE_DIR_DEFAULT = CACHE_ROOT_DIR / "inductor_cache"
NVME_JIT_CACHE_DEFAULTS: dict[str, Path] = {
    "CUDA_CACHE_PATH": CUDA_COMPUTE_CACHE_DIR,
    "TRITON_CACHE_DIR": TRITON_CACHE_DIR_DEFAULT,
    "TORCHINDUCTOR_CACHE_DIR": INDUCTOR_CACHE_DIR_DEFAULT,
}


def configure_nvme_jit_caches(
    project_root: str | os.PathLike[str] | None = None,
    environ: dict[str, str] | os._Environ[str] | None = None,
    create_dirs: bool = True,
) -> dict[str, str]:
    """
    Định tuyến mặc định các thư mục cache biên dịch GPU/JIT/Triton sang ổ NVMe Gen4
    (PROJECT_ROOT / ".cache" / {cuda_compute_cache, triton_cache, inductor_cache})
    thông qua setdefault; tạo thư mục bên trong try/except OSError để không bao giờ
    gây lỗi trên môi trường sandbox hoặc chỉ đọc.
    """
    base_root = Path(project_root) if project_root is not None else PROJECT_ROOT
    cache_base = base_root / ".cache"
    target_env = os.environ if environ is None else environ
    defaults = {
        "CUDA_CACHE_PATH": str(cache_base / "cuda_compute_cache"),
        "TRITON_CACHE_DIR": str(cache_base / "triton_cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_base / "inductor_cache"),
    }
    resolved: dict[str, str] = {}
    for env_key, default_dir in defaults.items():
        chosen = target_env.setdefault(env_key, default_dir)
        resolved[env_key] = chosen
        if create_dirs and chosen:
            try:
                Path(chosen).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
    return resolved


# Tối ưu hóa cấp Driver CUDA trước khi nạp PyTorch/TensorRT:
# - CUDA_MODULE_LOADING=LAZY: giảm ~300-500 MB VRAM chiếm dụng ban đầu và rút ngắn thời gian khởi động.
# - CUDA_CACHE_MAXSIZE=2GB & định tuyến cache biên dịch JIT/Triton/Inductor sang ổ NVMe Gen4 (PROJECT_ROOT/.cache).
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUDA_CACHE_MAXSIZE", "2147483648")
configure_nvme_jit_caches()

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
    """Xóa một tập khỏi pending_subs.json khi đã đạt chuẩn (khớp base_name hoặc video_path)."""
    if not base_name:
        return
    items = load_pending_subs()
    stem = os.path.splitext(os.path.basename(str(base_name)))[0]
    candidate_names = {str(base_name), stem, f"{stem}_4K"}
    if stem.endswith("_4K"):
        candidate_names.add(stem[:-3])
    abs_target = os.path.abspath(str(base_name)) if os.path.sep in str(base_name) or "/" in str(base_name) else None

    def _is_match(it: dict) -> bool:
        if it.get("base_name") in candidate_names:
            return True
        vp = it.get("video_path")
        if vp:
            if vp == base_name or (abs_target and os.path.abspath(vp) == abs_target):
                return True
            if os.path.splitext(os.path.basename(vp))[0] in candidate_names:
                return True
        return False

    new_items = [it for it in items if not _is_match(it)]
    if len(new_items) != len(items):
        try:
            with open(PENDING_SUBS_FILE, "w", encoding="utf-8") as f:
                json.dump(new_items, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


_GPU_LOCK_OWNED: bool = False


def is_gpu_lock_held_by_other() -> int | None:
    """
    Kiểm tra xem .gpu_active.lock có đang thuộc về một tiến trình khác còn sống hay không.
    Trả về PID đang giữ khóa nếu tiến trình đó còn tồn tại, ngược lại trả về None (stale lock hoặc không có khóa).
    """
    if not os.path.exists(GPU_LOCK_FILE):
        return None
    try:
        with open(GPU_LOCK_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return None
        pid = int(raw.splitlines()[0].strip())
        if pid == os.getpid():
            return None
        import psutil
        if psutil.pid_exists(pid):
            return pid
    except Exception:
        return None
    return None


def acquire_gpu_lock() -> None:
    """Tạo file khóa nguyên tử đánh dấu giai đoạn upscale GPU đang hoạt động (tự động thu hồi stale lock nếu PID cũ đã chết)."""
    global _GPU_LOCK_OWNED
    other_pid = is_gpu_lock_held_by_other()
    if other_pid is not None:
        raise RuntimeError(
            f"GPU đang bận bởi tiến trình khác đang hoạt động (PID={other_pid}, lock: {GPU_LOCK_FILE}). "
            f"Chỉ cho phép duy nhất 1 tiến trình suy luận GPU tại một thời điểm."
        )
    lock_dir = os.path.dirname(os.path.abspath(GPU_LOCK_FILE))
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    tmp_lock = f"{GPU_LOCK_FILE}.{os.getpid()}.tmp"
    my_pid = os.getpid()
    try:
        with open(tmp_lock, "w", encoding="utf-8") as f:
            f.write(f"{my_pid}\n")
        os.replace(tmp_lock, GPU_LOCK_FILE)
        # Xác minh lại chủ sở hữu khóa để chặn cửa sổ đua TOCTOU khi 2 tiến trình cùng khởi tạo
        with open(GPU_LOCK_FILE, "r", encoding="utf-8") as vf:
            holder_raw = vf.read().strip()
        holder_pid = int(holder_raw.splitlines()[0].strip()) if holder_raw else None
        if holder_pid != my_pid:
            raise RuntimeError(
                f"GPU đang bận bởi tiến trình khác đang hoạt động (PID={holder_pid}, lock: {GPU_LOCK_FILE}). "
                f"Chỉ cho phép duy nhất 1 tiến trình suy luận GPU tại một thời điểm."
            )
        _GPU_LOCK_OWNED = True
    except Exception:
        if os.path.exists(tmp_lock):
            try:
                os.remove(tmp_lock)
            except OSError:
                pass
        raise


def release_gpu_lock() -> None:
    """Giải phóng file khóa giai đoạn upscale GPU (không xóa nhầm khóa của PID khác đang chạy)."""
    global _GPU_LOCK_OWNED
    _GPU_LOCK_OWNED = False
    if is_gpu_lock_held_by_other() is not None:
        return
    try:
        if os.path.exists(GPU_LOCK_FILE):
            os.remove(GPU_LOCK_FILE)
    except Exception:
        pass


@contextlib.contextmanager
def gpu_exclusivity_lock():
    """Context manager đảm bảo độc quyền GPU qua .gpu_active.lock và luôn thu hồi sạch sẽ."""
    acquire_gpu_lock()
    try:
        yield GPU_LOCK_FILE
    finally:
        release_gpu_lock()


def _atexit_release_gpu_lock() -> None:
    """Chỉ tự động xóa khóa khi chính tiến trình này đã acquire_gpu_lock()."""
    if _GPU_LOCK_OWNED:
        release_gpu_lock()


import atexit
atexit.register(_atexit_release_gpu_lock)



# -----------------------------------------------------------------------------
# THAM SỐ UPSCALE MẶC ĐỊNH (TỐI ƯU HÓA NVIDIA CUDA FP16 & TENSOR CORES)
# -----------------------------------------------------------------------------
def configure_cuda_fast_math() -> dict[str, bool]:
    """
    Mở khóa tối đa thông lượng Tensor Cores kiến trúc Ampere (GA106 - RTX 3060):
    - Bật cuDNN autotuner (benchmark=True) chọn kernel tích chập nhanh nhất.
    - Bật TF32 cho matmul/cuDNN FP32 fallback.
    - Bật allow_fp16_reduced_precision_reduction & allow_bf16_reduced_precision_reduction
      để cho phép lệnh Tensor Core HMMA.16816.F16 chạy full-rate (51 TFLOPS dense)
      thay vì bị giới hạn half-rate bởi thanh ghi FP32-accumulate trên GeForce.
    """
    applied = {
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "fp16_fast_accum": False,
    }
    try:
        import torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        applied["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        applied["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            applied["fp16_fast_accum"] = bool(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            )
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return applied


def _detect_default_engine() -> str:
    """Tự động chọn engine suy luận tốt nhất theo phần cứng hiện có (Ưu tiên CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            configure_cuda_fast_math()
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
DEFAULT_BATCH_SIZE = 8 if DEFAULT_ENGINE == "cuda" else 1  # Lô 8 frames (Điểm ngọt cho GA106 28 SMs / 112 Tensor Cores trên RTX 3060 12GB)
DEFAULT_VRAM_CLEANUP_THRESHOLD_GB = 9.5  # Ngưỡng VRAM (GB) để dọn dẹp CUDA memory cache thay vì restart tiến trình
DEFAULT_GPU_RESTART_CHUNKS = 10  # Ngưỡng dự phòng cho worker

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
    env_enc = os.environ.get("UPSCALER_HEVC_ENCODER", "").strip()
    if env_enc in ("hevc_nvenc", "hevc_qsv", "libx265"):
        return env_enc
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
if not os.environ.get("PYTEST_CURRENT_TEST"):
    os.environ.setdefault("UPSCALER_HEVC_ENCODER", HEVC_ENCODER)
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
        print("💡 Vui lòng đặt ffmpeg.exe vào 'bin/ffmpeg/' hoặc thêm FFmpeg (Gyan.dev Full Build) vào biến môi trường PATH.")
        sys.exit(1)

    if shutil.which("ffprobe") is None:
        print("❌ Lỗi: Không tìm thấy 'ffprobe' trên hệ thống!")
        print("💡 Vui lòng đặt ffprobe.exe vào 'bin/ffmpeg/' hoặc thêm FFmpeg (Gyan.dev Full Build) vào biến môi trường PATH.")
        sys.exit(1)

