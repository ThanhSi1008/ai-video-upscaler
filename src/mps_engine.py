# -*- coding: utf-8 -*-
"""
src/mps_engine.py
Deprecated legacy shim. Toàn bộ mã nguồn đã chuyển sang src/cuda_engine.py
được tối ưu hóa cho Windows 11 + NVIDIA CUDA (RTX 3060 12GB).
Giữ lại file này để đảm bảo tính tương thích ngược tuyệt đối cho các script/test cũ.
"""

from src.cuda_engine import (
    RealESRAnimeVideoV3,
    AnimeJaNaiCompact2x,
    GPUError,
    GPUError as MPSError,
    GPUError as GPUEngineError,
    GPUPersistentWorker,
    GPUPersistentWorker as MPSPersistentWorker,
    get_gpu_worker,
    get_gpu_worker as get_mps_worker,
    get_cuda_inference_engine,
    upscale_frames_with_fallback,
)

__all__ = [
    "RealESRAnimeVideoV3",
    "AnimeJaNaiCompact2x",
    "GPUError",
    "MPSError",
    "GPUEngineError",
    "GPUPersistentWorker",
    "MPSPersistentWorker",
    "get_gpu_worker",
    "get_mps_worker",
    "get_cuda_inference_engine",
    "upscale_frames_with_fallback",
]
