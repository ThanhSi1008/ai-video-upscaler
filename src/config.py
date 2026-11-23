"""Central runtime paths, E-drive cache routing, GPU exclusivity lock, and NVENC presets."""

import atexit
import contextlib
from functools import lru_cache
import os
from pathlib import Path
import shutil
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
PROJECT_ROOT = Path(ROOT_DIR)

try:
    from src.env_isolation import configure_e_drive_isolation
    from src.pending_queue import (
        LEGACY_PENDING_SUBS_FILE,
        PENDING_SUBS_FILE,
        add_pending_sub,
        load_pending_subs,
        remove_pending_sub,
    )
except ImportError:
    from env_isolation import configure_e_drive_isolation
    from pending_queue import (
        LEGACY_PENDING_SUBS_FILE,
        PENDING_SUBS_FILE,
        add_pending_sub,
        load_pending_subs,
        remove_pending_sub,
    )

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
    """Route GPU/JIT/Triton compilation caches into NVMe-backed .cache subdirectories."""
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


# Configure lazy CUDA module loading and cache paths before importing torch/TRT
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUDA_CACHE_MAXSIZE", "2147483648")
configure_nvme_jit_caches()

BIN_DIR = os.path.join(ROOT_DIR, "bin", "realesrgan")
REALESRGAN_BIN = os.path.join(BIN_DIR, "realesrgan-ncnn-vulkan")
MODELS_DIR = os.path.join(BIN_DIR, "models")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
_env_logs_dir = os.environ.get("UPSCALER_LOGS_DIR", "").strip()
LOGS_DIR = os.path.abspath(os.path.expanduser(_env_logs_dir)) if _env_logs_dir else DEFAULT_LOGS_DIR
GPU_LOCK_FILE = os.path.join(ROOT_DIR, ".gpu_active.lock")

_GPU_LOCK_OWNED: bool = False
_GPU_LOCK_DEPTH: int = 0


def is_gpu_lock_held_by_other() -> int | None:
    """Return holding PID if another live process owns .gpu_active.lock, else None."""
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
    """Atomically claim .gpu_active.lock for this PID with re-entrant reference counting."""
    global _GPU_LOCK_OWNED, _GPU_LOCK_DEPTH
    if not _GPU_LOCK_OWNED and _GPU_LOCK_DEPTH > 0:
        _GPU_LOCK_DEPTH = 0
    other_pid = is_gpu_lock_held_by_other()
    if other_pid is not None:
        raise RuntimeError(
            f"GPU đang bận bởi tiến trình khác đang hoạt động (PID={other_pid}, lock: {GPU_LOCK_FILE}). "
            f"Chỉ cho phép duy nhất 1 tiến trình suy luận GPU tại một thời điểm."
        )
    if _GPU_LOCK_DEPTH > 0 and os.path.exists(GPU_LOCK_FILE):
        _GPU_LOCK_DEPTH += 1
        _GPU_LOCK_OWNED = True
        return
    lock_dir = os.path.dirname(os.path.abspath(GPU_LOCK_FILE))
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    tmp_lock = f"{GPU_LOCK_FILE}.{os.getpid()}.tmp"
    my_pid = os.getpid()
    try:
        with open(tmp_lock, "w", encoding="utf-8") as f:
            f.write(f"{my_pid}\n")
        os.replace(tmp_lock, GPU_LOCK_FILE)
        # Re-read lock file to guard against concurrent TOCTOU lock creation.
        with open(GPU_LOCK_FILE, "r", encoding="utf-8") as vf:
            holder_raw = vf.read().strip()
        holder_pid = int(holder_raw.splitlines()[0].strip()) if holder_raw else None
        if holder_pid != my_pid:
            raise RuntimeError(
                f"GPU đang bận bởi tiến trình khác đang hoạt động (PID={holder_pid}, lock: {GPU_LOCK_FILE}). "
                f"Chỉ cho phép duy nhất 1 tiến trình suy luận GPU tại một thời điểm."
            )
        _GPU_LOCK_DEPTH = 1
        _GPU_LOCK_OWNED = True
    except Exception:
        if os.path.exists(tmp_lock):
            try:
                os.remove(tmp_lock)
            except OSError:
                pass
        raise


def release_gpu_lock() -> None:
    """Decrement re-entrant GPU lock depth and remove .gpu_active.lock when outer holder exits."""
    global _GPU_LOCK_OWNED, _GPU_LOCK_DEPTH
    if not _GPU_LOCK_OWNED and _GPU_LOCK_DEPTH > 0:
        _GPU_LOCK_DEPTH = 0
    if _GPU_LOCK_DEPTH > 1:
        _GPU_LOCK_DEPTH -= 1
        _GPU_LOCK_OWNED = True
        return
    _GPU_LOCK_DEPTH = 0
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
    """Re-entrant context manager enforcing single-process GPU ownership via .gpu_active.lock."""
    acquire_gpu_lock()
    try:
        yield GPU_LOCK_FILE
    finally:
        release_gpu_lock()


def _atexit_release_gpu_lock() -> None:
    """Release GPU lock on process exit only when owned by this process."""
    global _GPU_LOCK_DEPTH
    if _GPU_LOCK_OWNED or _GPU_LOCK_DEPTH > 0:
        _GPU_LOCK_DEPTH = 1
        release_gpu_lock()


atexit.register(_atexit_release_gpu_lock)



def configure_cuda_fast_math() -> dict[str, bool]:
    """Enable Ampere FP16 reduced-precision Tensor Core accumulation, TF32, and cuDNN autotuning."""
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
    """Prefer CUDA engine when torch reports an available GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            configure_cuda_fast_math()
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_device_name() -> str:
    """Return active NVIDIA GPU device name or 'CPU'."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


@lru_cache(maxsize=1)
def _get_default_engine() -> str:
    return _detect_default_engine()


@lru_cache(maxsize=1)
def _get_device_name() -> str:
    return get_device_name()


@lru_cache(maxsize=1)
def _get_default_batch_size() -> int:
    return 8 if _get_default_engine() == "cuda" else 1


DEFAULT_MODEL = "realesr-animevideov3"
DEFAULT_PRECISION = "fp16"
DEFAULT_VRAM_CLEANUP_THRESHOLD_GB = 9.5
DEFAULT_GPU_RESTART_CHUNKS = 10

DEFAULT_SCALE = 2
DEFAULT_BITRATE = "28M"
DEFAULT_SEGMENT_TIME = 8
DEFAULT_ZERO_DISK_SEGMENT_TIME = 24
DEFAULT_FRAME_FORMAT = "png"
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "out")

DEFAULT_DEDUP_HI = 768
DEFAULT_DEDUP_LO = 320
DEFAULT_DEDUP_FRAC = 0.33
DEFAULT_DEDUP_ENABLED = True

DEFAULT_RAMDISK_SIZE_MB = 5120
DEFAULT_PNG_UNIT_SIZE_MB = 16.0


@lru_cache(maxsize=1)
def _probe_ffmpeg_hevc_encoder() -> str:
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


def _detect_hevc_encoder() -> str:
    """Detect available 10-bit HEVC hardware/software encoder."""
    env_enc = os.environ.get("UPSCALER_HEVC_ENCODER", "").strip()
    if env_enc in ("hevc_nvenc", "hevc_qsv", "libx265"):
        return env_enc
    chosen = _probe_ffmpeg_hevc_encoder()
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        os.environ.setdefault("UPSCALER_HEVC_ENCODER", chosen)
    return chosen


_LAZY_ATTRS = frozenset({
    "DEFAULT_ENGINE",
    "DEVICE_NAME",
    "DEFAULT_BATCH_SIZE",
    "HEVC_ENCODER",
})


def __getattr__(name: str):
    if name == "DEFAULT_ENGINE":
        return _get_default_engine()
    if name == "DEVICE_NAME":
        return _get_device_name()
    if name == "DEFAULT_BATCH_SIZE":
        return _get_default_batch_size()
    if name == "HEVC_ENCODER":
        return _detect_hevc_encoder()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _LAZY_ATTRS)
HEVC_PRESET = "p6"
HEVC_CQ_LEVEL = "18"
HEVC_MAXRATE = "45M"
HEVC_BUFSIZE = "90M"
HEVC_MULTIPASS = "fullres"
HEVC_MULTIPASS_POLICY = "auto"
VALID_NVENC_MULTIPASS_POLICIES = ("auto", "fullres", "qres")
VALID_GRAIN_RESTORE_POLICIES = ("auto", "off", "light", "heavy")
DEFAULT_GRAIN_RESTORE = "auto"
HEVC_RC_LOOKAHEAD = 32
HEVC_SPATIAL_AQ = 1
HEVC_AQ_STRENGTH = 6
HEVC_TEMPORAL_AQ = 1
HEVC_BF = 4
HEVC_B_REF_MODE = "middle"
HEVC_PROFILE = "main10"
HEVC_PIX_FMT = "p010le"
HEVC_COLOR_RANGE = "tv"
HEVC_COLORSPACE = "bt709"
HEVC_COLOR_PRIMARIES = "bt709"
HEVC_COLOR_TRC = "bt709"
HEVC_TAG = "hvc1"

_MODEL_TRT_INFER_FPS_ESTIMATE: dict[str, float] = {
    "animevideov3": 10.5,
    "realesr-animevideov3": 10.5,
    "animejanai-hd-v3-compact": 12.8,
    "animejanai-hd-v3-ultracompact": 22.8,
    "animejanai-hd-v3-sharp1-ultracompact": 22.8,
    "animejanai-hd-v3-superultracompact": 44.9,
    "animejanai-hd-v3-sharp1-superultracompact": 44.9,
    "span-2x": 10.5,
    "2x_animejanai_v3": 10.5,
    "modernspanimation-v2-2x": 10.5,
    "realcugan-pro-conservative-2x": 4.9,
    "realcugan-pro-denoise3x-2x": 4.9,
    "auto-pareto": 18.0,
}


def resolve_episode_nvenc_multipass(
    policy: str = "auto",
    model_id: str = "animevideov3",
    dedup_ratio_pct: float = 0.0,
    engine: str = "cuda",
    use_tensorrt: bool = True,
    zero_disk: bool = True,
) -> str:
    """Resolve uniform per-episode NVENC multipass ('fullres' or 'qres') to preserve identical HEVC SPS/PPS."""
    norm = str(policy or "auto").strip().lower()
    if norm == "fullres":
        return "fullres"
    if norm == "qres":
        return "qres"
    if norm != "auto":
        raise ValueError(
            f"Invalid --nvenc-multipass policy '{policy}'. Expected one of {VALID_NVENC_MULTIPASS_POLICIES}."
        )
    if engine != "cuda" or not zero_disk:
        return "fullres"
    base_trt_fps = _MODEL_TRT_INFER_FPS_ESTIMATE.get(str(model_id or "").strip().lower(), 10.5)
    if not use_tensorrt:
        base_trt_fps *= 0.52
    keep_frac = max(0.05, min(1.0, 1.0 - (float(dedup_ratio_pct or 0.0) / 100.0)))
    eff_ai_fps = base_trt_fps / keep_frac
    return "qres" if eff_ai_fps > 21.5 else "fullres"


def get_hevc_rate_args(bitrate: str = "28M", multipass: str | None = None) -> list:
    """Return 10-bit HEVC Main 10 rate-control flags tuned for NVENC or fallback encoders."""
    hevc_encoder = getattr(sys.modules[__name__], "HEVC_ENCODER")
    if hevc_encoder == "hevc_nvenc":
        eff_mp = str(multipass).strip().lower() if multipass else HEVC_MULTIPASS
        if eff_mp not in ("fullres", "qres"):
            eff_mp = HEVC_MULTIPASS
        return [
            "-preset", HEVC_PRESET,
            "-tune", "hq",
            "-multipass", eff_mp,
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
    if hevc_encoder == "libx265":
        return [
            "-preset", "ultrafast",
            "-crf", str(HEVC_CQ_LEVEL),
            "-x265-params", "log-level=error",
        ]
    return [
        "-b:v", bitrate,
        "-spatial_aq", "1"
    ]


def validate_environment(engine: str | None = None):
    """Verify that ffmpeg and ffprobe are available on PATH."""
    if shutil.which("ffmpeg") is None:
        print("❌ Lỗi: Không tìm thấy 'ffmpeg' trên hệ thống!")
        print("💡 Vui lòng đặt ffmpeg.exe vào 'bin/ffmpeg/' hoặc thêm FFmpeg (Gyan.dev Full Build) vào biến môi trường PATH.")
        sys.exit(1)

    if shutil.which("ffprobe") is None:
        print("❌ Lỗi: Không tìm thấy 'ffprobe' trên hệ thống!")
        print("💡 Vui lòng đặt ffprobe.exe vào 'bin/ffmpeg/' hoặc thêm FFmpeg (Gyan.dev Full Build) vào biến môi trường PATH.")
        sys.exit(1)

