# -*- coding: utf-8 -*-
"""
src/mps_engine.py
Động cơ suy luận PyTorch MPS FP16 trên Apple Silicon GPU Metal:
- Worker bền (persistent): Nạp model một lần, tái sử dụng qua các chunk, tự động khởi động lại mỗi N chunk (A2).
- Hỗ trợ đa mô hình (D3): animevideov3 (x4 + bicubic 0.5) và animejanai-hd-v3-compact (x2 native).
- Giám sát độ an toàn: Phát hiện NaN/Inf, lỗi Metal/OOM để tự động kích hoạt fallback cùng model (A1, D6).
- Ghi nhận viễn trắc chi tiết: current_allocated, driver_allocated_memory và process RSS.
"""

import os
import sys
import time
import glob
import psutil
import shutil
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import REALESRGAN_BIN, MODELS_DIR

# -----------------------------------------------------------------------------
# 1. Định nghĩa Kiến trúc Mạng Nơ-ron (SRVGGNetCompact)
# -----------------------------------------------------------------------------

class RealESRAnimeVideoV3(nn.Module):
    """Mô hình RealESR-AnimeVideoV3: 64f16c x4 + Nearest 4x + Bicubic 0.5x -> 2x 4K"""
    def __init__(self):
        super().__init__()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(3, 64, 3, 1, 1))
        self.body.append(nn.PReLU(64))
        for _ in range(16):
            self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
            self.body.append(nn.PReLU(64))
        self.body.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(4)

    def forward(self, x):
        feat = x
        for layer in self.body:
            feat = layer(feat)
        out = self.upsampler(feat)
        base = F.interpolate(x, scale_factor=4.0, mode='nearest')
        out_4x = out + base
        out_2x = F.interpolate(out_4x, scale_factor=0.5, mode='bicubic', align_corners=False)
        return out_2x


class AnimeJaNaiCompact2x(nn.Module):
    """Mô hình AnimeJaNai HD V3 Compact: 64f16c x2 native (PixelShuffle 2) -> 2x 4K"""
    def __init__(self):
        super().__init__()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(3, 64, 3, 1, 1))
        self.body.append(nn.PReLU(64))
        for _ in range(16):
            self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
            self.body.append(nn.PReLU(64))
        self.body.append(nn.Conv2d(64, 12, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(2)

    def forward(self, x):
        feat = x
        for layer in self.body:
            feat = layer(feat)
        out = self.upsampler(feat)
        base = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return out + base


class MPSError(Exception):
    """Lỗi suy luận trên động cơ PyTorch MPS (NaN, Inf, OOM hoặc Metal error)."""
    pass


# -----------------------------------------------------------------------------
# 2. Worker MPS Bền (Persistent Worker & Auto-Restart)
# -----------------------------------------------------------------------------

class MPSPersistentWorker:
    """
    Worker PyTorch MPS bền: nạp mô hình vào Unified Memory một lần duy nhất,
    tự động khởi động lại sau mỗi max_chunks để giải phóng driver pool fragments.
    """
    def __init__(
        self,
        model_id: str = "animevideov3",
        weights_path: str = "models_bench/realesr-animevideov3.pth",
        precision: str = "fp16",
        max_chunks_before_restart: int = 10
    ):
        self.model_id = model_id
        self.weights_path = weights_path
        self.precision = precision.lower()
        self.max_chunks_before_restart = max_chunks_before_restart
        self.chunks_processed = 0
        self.total_restarts = 0
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Nạp trọng số và khởi tạo mô hình trên device MPS."""
        t0 = time.time()
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số mô hình: {self.weights_path}")

        sd = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        weights = sd.get("params_ema", sd.get("params", sd))
        clean_weights = {k: v for k, v in weights.items() if k.startswith("body.")}

        if self.model_id == "animejanai-hd-v3-compact":
            model = AnimeJaNaiCompact2x().to(self.device)
        else:
            model = RealESRAnimeVideoV3().to(self.device)

        if self.precision == "fp16":
            model = model.half()
            clean_weights = {k: v.half() for k, v in clean_weights.items()}
        else:
            model = model.float()
            clean_weights = {k: v.float() for k, v in clean_weights.items()}

        model.load_state_dict(clean_weights, strict=True)
        model.eval()

        # Warm-up 1 frame nhỏ
        dummy = torch.rand(1, 3, 540, 960, device=self.device)
        if self.precision == "fp16":
            dummy = dummy.half()
        with torch.inference_mode():
            _ = model(dummy)
            if self.device.type == "mps":
                torch.mps.synchronize()

        self.model = model
        self.last_load_time = time.time() - t0

    def restart_if_needed(self, logger=None) -> bool:
        """Kiểm tra và tự động khởi động lại worker nếu đã vượt ngưỡng N chunks."""
        self.chunks_processed += 1
        if self.chunks_processed >= self.max_chunks_before_restart:
            t0 = time.time()
            self.model = None
            if self.device.type == "mps":
                torch.mps.empty_cache()
            self._load_model()
            restart_elapsed = time.time() - t0
            self.chunks_processed = 0
            self.total_restarts += 1
            msg = (
                f"♻️ [MPS Worker Restart] Đã làm mới worker sau {self.max_chunks_before_restart} chunks "
                f"(Chi phí nạp lại: {restart_elapsed*1000:.1f} ms | Tổng lần restart: {self.total_restarts})"
            )
            if logger:
                logger.info(msg)
            return True
        return False

    def upscale_directory(
        self,
        input_dir: str,
        output_dir: str,
        fmt: str = "png",
        logger=None,
        mid_chunk_callback=None
    ) -> Dict[str, Any]:
        """
        Suy luận toàn bộ ảnh trong input_dir và lưu 4K vào output_dir trên RAM Disk.
        Đo đạc tốc độ, đỉnh bộ nhớ và phát hiện NaN/Inf.
        """
        os.makedirs(output_dir, exist_ok=True)
        img_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(f".{fmt}")
        ]) if os.path.exists(input_dir) else []
        if not img_files:
            return {"elapsed_seconds": 0.0, "fps": 0.0, "steady_fps": 0.0, "num_frames": 0}

        t_start = time.time()
        steady_start = None
        driver_alloc_peak = 0.0
        curr_alloc_peak = 0.0
        proc = psutil.Process()

        # Thread pool để ghi PNG song song (mức nén 1 tối ưu tốc độ)
        writer_executor = ThreadPoolExecutor(max_workers=4)
        write_futures = []

        try:
            for idx, fpath in enumerate(img_files):
                fname = os.path.basename(fpath)
                out_path = os.path.join(output_dir, fname)

                # 1. Đọc ảnh RGB24
                img = Image.open(fpath).convert("RGB")
                arr = (np.array(img).transpose((2, 0, 1)) / 255.0).astype(np.float32)
                tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
                if self.precision == "fp16":
                    tensor = tensor.half()

                # 2. Suy luận GPU Metal
                with torch.inference_mode():
                    out_tensor = self.model(tensor)
                    if self.device.type == "mps":
                        torch.mps.synchronize()

                if idx == 0:
                    steady_start = time.time()

                # 3. Kiểm tra NaN / Inf (Safety Gate A1)
                if torch.isnan(out_tensor).any() or torch.isinf(out_tensor).any():
                    raise MPSError(f"Phát hiện giá trị bất thường (NaN/Inf) tại khung hình: '{fname}'")

                # 4. Ghi nhận đỉnh VRAM
                if self.device.type == "mps":
                    c_alloc = torch.mps.current_allocated_memory() / (1024 * 1024)
                    if c_alloc > curr_alloc_peak:
                        curr_alloc_peak = c_alloc
                    if hasattr(torch.mps, "driver_allocated_memory"):
                        d_alloc = torch.mps.driver_allocated_memory() / (1024 * 1024)
                        if d_alloc > driver_alloc_peak:
                            driver_alloc_peak = d_alloc

                # 5. Xuất ảnh 4K (async writer)
                out_np = (out_tensor.squeeze(0).clamp(0, 1).float().cpu().numpy().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)
                def save_task(path, array):
                    out_img = Image.fromarray(array)
                    out_img.save(path, format="PNG", compress_level=1)

                write_futures.append(writer_executor.submit(save_task, out_path, out_np))

                if mid_chunk_callback and idx == (len(img_files) // 2):
                    try:
                        mid_chunk_callback(idx, len(img_files))
                    except Exception as e_cb:
                        if "No space left on device" in str(e_cb) or (isinstance(e_cb, OSError) and getattr(e_cb, "errno", None) == 28):
                            raise
                        if logger:
                            logger.warn(f"Lỗi mid_chunk_callback: {e_cb}")

            # Chờ toàn bộ writer threads hoàn tất
            for f in write_futures:
                f.result()

        except Exception as e:
            writer_executor.shutdown(wait=False)
            raise MPSError(f"Lỗi suy luận MPS tại frame: {e}") from e
        finally:
            writer_executor.shutdown(wait=True)

        elapsed = time.time() - t_start
        steady_elapsed = (time.time() - steady_start) if (steady_start and len(img_files) > 1) else elapsed
        steady_count = (len(img_files) - 1) if len(img_files) > 1 else len(img_files)

        overall_fps = len(img_files) / elapsed if elapsed > 0 else 0.0
        steady_fps = steady_count / steady_elapsed if steady_elapsed > 0 else overall_fps

        rss_mb = proc.memory_info().rss / (1024 * 1024)

        return {
            "elapsed_seconds": elapsed,
            "fps": overall_fps,
            "steady_fps": steady_fps,
            "num_frames": len(img_files),
            "peak_rss_mb": rss_mb,
            "current_alloc_mb": curr_alloc_peak,
            "driver_alloc_mb": driver_alloc_peak
        }


# -----------------------------------------------------------------------------
# 3. Singleton Worker & Điều Phối Fallback (A1, D6)
# -----------------------------------------------------------------------------

_GLOBAL_MPS_WORKER: Optional[MPSPersistentWorker] = None

def get_mps_worker(
    model_id: str,
    weights_path: str,
    precision: str = "fp16",
    max_chunks_before_restart: int = 10
) -> MPSPersistentWorker:
    """Lấy hoặc khởi tạo instance Singleton cho MPSPersistentWorker."""
    global _GLOBAL_MPS_WORKER
    if (_GLOBAL_MPS_WORKER is None
        or _GLOBAL_MPS_WORKER.model_id != model_id
        or _GLOBAL_MPS_WORKER.precision != precision.lower()):
        _GLOBAL_MPS_WORKER = MPSPersistentWorker(
            model_id=model_id,
            weights_path=weights_path,
            precision=precision,
            max_chunks_before_restart=max_chunks_before_restart
        )
    return _GLOBAL_MPS_WORKER


def upscale_frames_with_fallback(
    input_dir: str,
    output_dir: str,
    model_info: Dict[str, Any],
    engine: str = "mps",
    precision: str = "fp16",
    fmt: str = "png",
    logger=None,
    mid_chunk_callback=None
) -> Dict[str, Any]:
    """
    Hàm giao tiếp chính thực hiện suy luận khung hình cho 1 chunk:
    - Mặc định: PyTorch MPS FP16
    - Tự động fallback khi gặp lỗi (A1, D6):
      + animevideov3: fallback sang NCNN Vulkan (-t 540 -j 1:2:2)
      + animejanai: fallback sang MPS FP32
    - Bảo đảm KHÔNG tự ý đổi sang model khác!
    """
    model_id = model_info["id"]
    weights_path = model_info["weights_path"]

    # 1. Nếu người dùng chỉ định engine NCNN hoặc model được yêu cầu NCNN
    if engine == "ncnn":
        from src.ai_engine import upscale_frames as ncnn_upscale
        ncnn_model_name = "realesr-animevideov3" if model_id == "animevideov3" else model_id
        return ncnn_upscale(
            input_dir=input_dir,
            output_dir=output_dir,
            model=ncnn_model_name,
            scale=2,
            fmt=fmt,
            tile_size=540,
            thread_cfg="1:2:2"
        )

    # 2. Thử nghiệm trên PyTorch MPS FP16
    try:
        worker = get_mps_worker(
            model_id=model_id,
            weights_path=weights_path,
            precision=precision
        )
        res = worker.upscale_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            fmt=fmt,
            logger=logger,
            mid_chunk_callback=mid_chunk_callback
        )
        worker.restart_if_needed(logger=logger)
        res["engine_used"] = f"mps_{precision}"
        return res

    except Exception as e:
        warn_msg = f"⚠️ [MPS Fallback Activated] Gặp lỗi khi suy luận MPS {precision.upper()}: {e}."
        if logger:
            logger.warn(warn_msg, to_console=True)
        else:
            print(warn_msg)

        # Fallback theo quy chuẩn D6: chỉ đổi engine / độ chính xác của CÙNG model
        if model_id == "animevideov3":
            fallback_msg = "🔄 Chuyển sang NCNN Vulkan (-t 540 -j 1:2:2) cho RealESR-AnimeVideoV3..."
            if logger:
                logger.info(fallback_msg, to_console=True)
            else:
                print(fallback_msg)
            from src.ai_engine import upscale_frames as ncnn_upscale
            res = ncnn_upscale(
                input_dir=input_dir,
                output_dir=output_dir,
                model="realesr-animevideov3",
                scale=2,
                fmt=fmt,
                tile_size=540,
                thread_cfg="1:2:2"
            )
            res["engine_used"] = "ncnn_fallback"
            return res

        elif model_id == "animejanai-hd-v3-compact":
            fallback_msg = "🔄 Chuyển sang MPS FP32 cho AnimeJaNai HD V3 Compact..."
            if logger:
                logger.info(fallback_msg, to_console=True)
            else:
                print(fallback_msg)
            worker_fp32 = get_mps_worker(
                model_id=model_id,
                weights_path=weights_path,
                precision="fp32"
            )
            res = worker_fp32.upscale_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                fmt=fmt,
                logger=logger
            )
            worker_fp32.restart_if_needed(logger=logger)
            res["engine_used"] = "mps_fp32_fallback"
            return res

        else:
            raise RuntimeError(f"Lỗi suy luận và không có đường fallback khả dụng cho model '{model_id}': {e}")
