# -*- coding: utf-8 -*-
"""
src/cuda_engine.py
Động cơ suy luận PyTorch FP16 tối ưu hóa cho NVIDIA CUDA & Tensor Cores trên Windows 11:
- GPUPersistentWorker: Nạp model vào VRAM một lần, tái sử dụng qua các phân đoạn chunk.
- VRAM Garbage Collection thông minh: Dọn dẹp CUDA memory cache khi reserved >= 9.5 GB.
- Hỗ trợ đa mô hình: RealESR-AnimeVideoV3 (x4 + bicubic 0.5 -> 2x) và AnimeJaNai HD V3 Compact (x2 native).
- Giám sát an toàn chất lượng: Phát hiện NaN/Inf, lỗi GPU/OOM để tự động kích hoạt fallback FP32.
- Ghi nhận viễn trắc chi tiết: VRAM allocated, peak VRAM, và host RAM RSS.
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
import cv2
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

    def _tail_4x_to_2x(self, feat_slice: torch.Tensor, x_slice: torch.Tensor) -> torch.Tensor:
        out = self.upsampler(feat_slice)
        base = F.interpolate(x_slice, scale_factor=4.0, mode='nearest')
        out_4x = out + base
        return F.interpolate(out_4x, scale_factor=0.5, mode='bicubic', align_corners=False)

    def forward(self, x):
        feat = x
        for layer in self.body:
            feat = layer(feat)
        # Cắt lát nhánh phóng đại 8K trung gian mỗi 2 frames khi batch lớn trên Full HD
        # để tiết kiệm ~4.5 GB VRAM mà vẫn khớp bit-exact 100%
        if (not torch.onnx.is_in_onnx_export()) and x.shape[0] > 2 and (x.shape[2] * x.shape[3]) >= (540 * 960):
            B, _, H, W = x.shape
            out_buf = torch.empty((B, 3, H * 2, W * 2), dtype=x.dtype, device=x.device)
            for s in range(0, B, 2):
                e = min(s + 2, B)
                out_buf[s:e].copy_(self._tail_4x_to_2x(feat[s:e], x[s:e]))
            return out_buf
        return self._tail_4x_to_2x(feat, x)


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


class SpandrelModelWrapper(nn.Module):
    """
    Wrapper mô hình tải qua Spandrel 0.4.2 cho động cơ suy luận PyTorch/CUDA:
    - Nếu descriptor.scale == 2 (Native 2x, ví dụ: SPAN-2x, 2x_AnimeJaNai_V3):
      Suy luận trực tiếp ra 2x (4K), hoàn toàn không downsample với bicubic 0.5x tail.
    - Nếu descriptor.scale == 4 và target_scale == 2 (ví dụ: RealESR-AnimeVideoV3):
      Áp dụng đuôi downsampling bicubic 0.5x từ 4x -> 2x.
      Khi batch lớn (>2) trên ảnh Full HD (>=540p), cắt lát vi mô 8K (micro-slicing)
      mỗi 2 frames để giảm ~75% đỉnh VRAM đệm trung gian.
    - Phơi bày đầy đủ thuộc tính body, upsampler, upscale, scale (nếu có) để tương thích 100%
      với TensorRT builder và inference runner.
    """
    def __init__(self, inner_model: nn.Module, scale: int = 2, target_scale: float = 2.0):
        super().__init__()
        self.inner_model = inner_model
        self.scale = int(scale)
        self.target_scale = float(target_scale)
        if hasattr(inner_model, "body"):
            self.body = inner_model.body
        if hasattr(inner_model, "upsampler"):
            self.upsampler = inner_model.upsampler
        if hasattr(inner_model, "upscale"):
            self.upscale = inner_model.upscale
        else:
            self.upscale = self.scale

    def _tail_4x_to_2x(self, out_4x: torch.Tensor) -> torch.Tensor:
        interp_in = out_4x.float() if (out_4x.device.type == "cpu" and out_4x.dtype == torch.float16) else out_4x
        out_2x = F.interpolate(interp_in, scale_factor=0.5, mode="bicubic", align_corners=False)
        return out_2x.to(out_4x.dtype) if out_2x.dtype != out_4x.dtype else out_2x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            return self.inner_model(x)

        if abs(self.scale - self.target_scale) > 1e-4:
            # Tối ưu micro-slicing 8K cho mẻ lớn trên Full HD
            if (not torch.onnx.is_in_onnx_export()) and x.shape[0] > 2 and (x.shape[2] * x.shape[3]) >= (540 * 960):
                B, _, H, W = x.shape
                out_h = int(round(H * self.target_scale))
                out_w = int(round(W * self.target_scale))
                out_buf = torch.empty((B, 3, out_h, out_w), dtype=x.dtype, device=x.device)
                for s in range(0, B, 2):
                    e = min(s + 2, B)
                    out_4x = self.inner_model(x[s:e])
                    out_buf[s:e].copy_(self._tail_4x_to_2x(out_4x))
                return out_buf

            out_4x = self.inner_model(x)
            return self._tail_4x_to_2x(out_4x)

        return self.inner_model(x)


class GPUError(Exception):
    """Lỗi suy luận trên động cơ PyTorch GPU (NaN, Inf, OOM hoặc lỗi CUDA runtime)."""
    pass


# -----------------------------------------------------------------------------
# 2. Worker PyTorch CUDA Bền (Persistent Worker & Auto-Restart / Smart Cleanup)
# -----------------------------------------------------------------------------

class GPUPersistentWorker:
    """
    Worker PyTorch GPU bền: nạp mô hình vào VRAM một lần duy nhất,
    hỗ trợ NVIDIA CUDA (Tensor Cores Ampere/Turing/Ada) và CPU fallback;
    áp dụng Channels-Last memory format (NHWC) cho thông lượng tối đa.
    """
    def __init__(
        self,
        model_id: str = "animevideov3",
        weights_path: str = "models_bench/realesr-animevideov3.pth",
        precision: str = "fp16",
        max_chunks_before_restart: int = 10,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        use_tensorrt: bool = True,
    ):
        self.model_id = model_id
        self.weights_path = weights_path
        self.precision = precision.lower()
        self.max_chunks_before_restart = max_chunks_before_restart
        self.chunks_processed = 0
        self.total_restarts = 0
        self.use_tensorrt = bool(use_tensorrt)
        self._trt_runners: Dict[Any, Any] = {}
        
        # Nhận diện device: Ưu tiên NVIDIA CUDA, fallback CPU
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.device.type == "cuda":
            from src.config import configure_cuda_fast_math
            configure_cuda_fast_math()

        if batch_size is not None:
            self.batch_size = max(1, int(batch_size))
        else:
            from src.config import DEFAULT_BATCH_SIZE
            self.batch_size = DEFAULT_BATCH_SIZE if self.device.type == "cuda" else 1

        self.model = None
        self._load_model()

    def _load_model(self):
        """Nạp trọng số và khởi tạo mô hình trên device (CUDA/CPU)."""
        t0 = time.time()
        if not os.path.exists(self.weights_path):
            # Tự động tải trọng số chính thức nếu chưa có file cục bộ
            if self.model_id == "animevideov3":
                os.makedirs(os.path.dirname(os.path.abspath(self.weights_path)), exist_ok=True)
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
                print(f"📥 Đang tự động tải trọng số RealESR-AnimeVideoV3 từ {url}...")
                import urllib.request
                from src.models_resolver import load_models_registry, verify_weights_sha256
                tmp_dl = self.weights_path + ".part"
                urllib.request.urlretrieve(url, tmp_dl)
                expected_sha = None
                try:
                    expected_sha = load_models_registry().get("animevideov3", {}).get("sha256")
                except Exception:
                    pass
                if not verify_weights_sha256(tmp_dl, expected_sha):
                    try:
                        os.remove(tmp_dl)
                    except OSError:
                        pass
                    raise RuntimeError(f"Kiểm tra toàn vẹn SHA-256 thất bại khi tải trọng số {self.weights_path}")
                os.replace(tmp_dl, self.weights_path)
            else:
                raise FileNotFoundError(f"Không tìm thấy file trọng số mô hình: {self.weights_path}")

        model = None
        use_spandrel = True
        try:
            import spandrel
            loader = spandrel.ModelLoader()
            desc = loader.load_from_file(self.weights_path)
            model_core = desc.model.to(self.device)
            scale = int(getattr(desc, "scale", 2))
            model = SpandrelModelWrapper(model_core, scale=scale, target_scale=2.0)
        except Exception:
            use_spandrel = False

        if not use_spandrel or model is None:
            sd = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            weights = sd.get("params_ema", sd.get("params", sd))
            clean_weights = {k: v for k, v in weights.items() if k.startswith("body.")}

            if self.model_id == "animejanai-hd-v3-compact":
                model = AnimeJaNaiCompact2x().to(self.device)
            else:
                model = RealESRAnimeVideoV3().to(self.device)

            if self.precision == "fp16":
                clean_weights = {k: v.half() for k, v in clean_weights.items()}
            else:
                clean_weights = {k: v.float() for k, v in clean_weights.items()}

            model.load_state_dict(clean_weights, strict=True)

        if self.precision == "fp16":
            model = model.half()
        else:
            model = model.float()

        if self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        model.eval()

        # Warm-up lô frame
        warm_b = min(self.batch_size, 2)
        dummy = torch.rand(warm_b, 3, 540, 960, device=self.device)
        if self.precision == "fp16":
            dummy = dummy.half()
        if self.device.type == "cuda":
            dummy = dummy.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            _ = model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        self.model = model
        self.last_load_time = time.time() - t0

    def restart_if_needed(self, logger=None) -> bool:
        """
        Quản lý bộ nhớ VRAM thông minh cho GPU NVIDIA (Quick Win):
        Ưu tiên dọn dẹp CUDA memory cache khi vượt ngưỡng an toàn (9.5 GB) để loại bỏ
        hoàn toàn việc dừng pipeline 3-4s nạp lại mô hình từ đĩa.
        """
        self.chunks_processed += 1
        if self.device.type == "cuda":
            reserved_gb = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            from src.config import DEFAULT_VRAM_CLEANUP_THRESHOLD_GB
            if reserved_gb >= DEFAULT_VRAM_CLEANUP_THRESHOLD_GB:
                torch.cuda.empty_cache()
                msg = (
                    f"🧹 [VRAM Garbage Collection] Đã dọn dẹp CUDA memory cache "
                    f"(Reserved: {reserved_gb:.2f} GB >= {DEFAULT_VRAM_CLEANUP_THRESHOLD_GB} GB)"
                )
                if logger:
                    logger.info(msg)
                return True
            return False

        if self.max_chunks_before_restart > 0 and self.chunks_processed >= self.max_chunks_before_restart:
            t0 = time.time()
            self.model = None
            self._load_model()
            restart_elapsed = time.time() - t0
            self.chunks_processed = 0
            self.total_restarts += 1
            msg = (
                f"♻️ [Worker Restart] Đã làm mới worker sau {self.max_chunks_before_restart} chunks "
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
        batch_size: Optional[int] = None,
        logger=None,
        mid_chunk_callback=None
    ) -> Dict[str, Any]:
        """
        Suy luận toàn bộ ảnh trong input_dir và lưu 4K vào output_dir trên RAM Disk.
        Hỗ trợ Batch Inference đa khung hình tối ưu cho NVIDIA CUDA Tensor Cores.
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

        b_size = batch_size if batch_size is not None else self.batch_size

        # Thread pool để ghi PNG song song (12 workers trên 16-thread CPU i5-12600K)
        writer_workers = min(12, max(4, os.cpu_count() or 4))
        writer_executor = ThreadPoolExecutor(max_workers=writer_workers)
        write_futures = []

        loader_workers = min(4, b_size)
        loader_executor = ThreadPoolExecutor(max_workers=loader_workers) if loader_workers > 1 else None

        def _read_img_bgr_u8(p: str) -> np.ndarray:
            im_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if im_bgr is not None:
                return im_bgr
            im = Image.open(p).convert("RGB")
            return np.ascontiguousarray(np.array(im, dtype=np.uint8)[:, :, ::-1])

        def _save_group_task(out_paths: List[str], bgr_array: np.ndarray):
            primary_path = out_paths[0]
            cv2.imwrite(primary_path, bgr_array, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            for dup_path in out_paths[1:]:
                try:
                    if os.path.exists(dup_path):
                        os.remove(dup_path)
                    os.link(primary_path, dup_path)
                except OSError:
                    shutil.copyfile(primary_path, dup_path)

        total_frames = len(img_files)
        target_dtype = torch.float16 if self.precision == "fp16" else torch.float32
        pinned_in: Optional[torch.Tensor] = None
        pinned_out_slots: List[Optional[torch.Tensor]] = [None, None]
        slot_futures: List[list] = [[], []]
        num_distinct = total_frames

        def _resolve_runner(in_h: int, in_w: int):
            if (
                self.device.type == "cuda"
                and self.precision == "fp16"
                and getattr(self, "use_tensorrt", False)
                and isinstance(self.model, nn.Module)
                and not os.environ.get("PYTEST_CURRENT_TEST")
            ):
                runners = getattr(self, "_trt_runners", None)
                if runners is None:
                    runners = {}
                    self._trt_runners = runners
                key = (int(in_h), int(in_w), int(b_size))
                if key not in runners:
                    from src.build_trt_engine import get_hybrid_engine_or_wrapper
                    runners[key] = get_hybrid_engine_or_wrapper(
                        self.model,
                        max_batch=b_size,
                        force_pytorch=False,
                        model_id=getattr(self, "model_id", "realesr-animevideov3"),
                        input_hw=(in_h, in_w),
                        scale=2.0,
                    )
                return runners[key]
            return self.model

        try:
            # Đọc song song ảnh đầu vào của phân đoạn và khử trùng lặp chu kỳ (A->B->A lip-flap / blink)
            if loader_executor and total_frames > 1:
                all_in_arrs = list(loader_executor.map(_read_img_bgr_u8, img_files))
            else:
                all_in_arrs = [_read_img_bgr_u8(p) for p in img_files]

            if total_frames > 1:
                from src.dedup import deduplicate_cyclic_frames
                distinct_arrs, frame_to_distinct = deduplicate_cyclic_frames(all_in_arrs)
            else:
                distinct_arrs = list(all_in_arrs)
                frame_to_distinct = [0]
            del all_in_arrs

            num_distinct = len(distinct_arrs)
            distinct_out_groups: List[List[str]] = [[] for _ in range(num_distinct)]
            distinct_first_names: List[str] = [""] * num_distinct
            for orig_idx, fpath in enumerate(img_files):
                d_idx = frame_to_distinct[orig_idx]
                fname = os.path.basename(fpath)
                if not distinct_first_names[d_idx]:
                    distinct_first_names[d_idx] = fname
                distinct_out_groups[d_idx].append(os.path.join(output_dir, fname))

            for b_start in range(0, num_distinct, b_size):
                b_end = min(b_start + b_size, num_distinct)
                batch_arrs = distinct_arrs[b_start:b_end]
                cur_b = len(batch_arrs)

                h0, w0 = batch_arrs[0].shape[:2]
                same_shape = all(a.shape[:2] == (h0, w0) for a in batch_arrs)

                if self.device.type == "cuda" and same_shape:
                    if pinned_in is None or pinned_in.shape[0] < cur_b or pinned_in.shape[1:3] != (h0, w0):
                        pinned_in = torch.empty((max(b_size, cur_b), h0, w0, 3), dtype=torch.uint8, pin_memory=True)
                    for i, arr in enumerate(batch_arrs):
                        pinned_in[i].copy_(torch.from_numpy(arr))
                    gpu_bgr = pinned_in[:cur_b].to(self.device, non_blocking=True)
                    # Lật kênh BGR -> RGB, chuyển NHWC -> NCHW channels_last và chuẩn hóa [0,1] ngay trên VRAM
                    tensor = (
                        gpu_bgr[..., [2, 1, 0]]
                        .permute(0, 3, 1, 2)
                        .to(memory_format=torch.channels_last, dtype=target_dtype)
                        .mul_(1.0 / 255.0)
                    )
                else:
                    batch_np = np.stack(batch_arrs, axis=0)
                    raw_t = torch.from_numpy(batch_np).to(self.device)
                    tensor = (
                        raw_t[..., [2, 1, 0]]
                        .permute(0, 3, 1, 2)
                        .contiguous()
                        .to(dtype=target_dtype)
                        .mul_(1.0 / 255.0)
                    )
                    if self.device.type == "cuda":
                        tensor = tensor.to(memory_format=torch.channels_last)

                # Giải phóng sớm tham chiếu mảng đầu vào của batch đã nạp lên GPU
                for clear_i in range(b_start, b_end):
                    distinct_arrs[clear_i] = None

                active_runner = _resolve_runner(h0, w0) if same_shape else self.model

                # 2. Suy luận GPU + Kiểm tra hữu hạn bất đồng bộ + Lượng tử hóa uint8 BGR trên VRAM (chỉ 1 lần sync D2H)
                with torch.inference_mode():
                    out_tensor = active_runner(tensor)
                    finite_flag = torch.isfinite(out_tensor).all()
                    gpu_bgr_out = (
                        out_tensor.clamp(0.0, 1.0)
                        .mul(255.0)
                        .round()
                        .to(torch.uint8)[:, [2, 1, 0], :, :]
                        .permute(0, 2, 3, 1)
                        .contiguous()
                    )
                    slot = (b_start // b_size) % 2
                    if self.device.type == "cuda":
                        # Đảm bảo các luồng ghi PNG của batch trước dùng cùng slot đã hoàn tất trước khi ghi đè (0 copy RAM)
                        if slot_futures[slot]:
                            for prev_fut in slot_futures[slot]:
                                prev_fut.result()
                            slot_futures[slot].clear()
                        out_h, out_w = gpu_bgr_out.shape[1:3]
                        cur_pin_out = pinned_out_slots[slot]
                        if (
                            cur_pin_out is None
                            or cur_pin_out.shape[0] < cur_b
                            or cur_pin_out.shape[1:3] != (out_h, out_w)
                        ):
                            cur_pin_out = torch.empty(
                                (max(b_size, cur_b), out_h, out_w, 3),
                                dtype=torch.uint8,
                                pin_memory=True,
                            )
                            pinned_out_slots[slot] = cur_pin_out
                        cur_pin_out[:cur_b].copy_(gpu_bgr_out, non_blocking=True)
                        torch.cuda.current_stream(self.device).synchronize()
                        is_finite = bool(finite_flag.item())
                        out_bgr_u8 = cur_pin_out[:cur_b].numpy()
                    else:
                        is_finite = bool(finite_flag.item())
                        out_bgr_u8 = gpu_bgr_out.cpu().numpy()

                if b_start == 0:
                    steady_start = time.time()

                # 3. Kiểm tra NaN / Inf (Safety Gate)
                if not is_finite:
                    sample_name = distinct_first_names[b_start]
                    raise GPUError(f"Phát hiện giá trị bất thường (NaN/Inf) tại khung hình lô: '{sample_name}'")

                # 4. Ghi nhận đỉnh VRAM
                if self.device.type == "cuda":
                    c_alloc = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                    if c_alloc > curr_alloc_peak:
                        curr_alloc_peak = c_alloc
                    d_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                    if d_alloc > driver_alloc_peak:
                        driver_alloc_peak = d_alloc
                for b_idx in range(cur_b):
                    out_group = distinct_out_groups[b_start + b_idx]
                    fut = writer_executor.submit(_save_group_task, out_group, out_bgr_u8[b_idx])
                    write_futures.append(fut)
                    if self.device.type == "cuda":
                        slot_futures[slot].append(fut)

                if mid_chunk_callback and (b_start <= (num_distinct // 2) < b_end):
                    try:
                        mid_chunk_callback(b_start, num_distinct)
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
            if loader_executor:
                loader_executor.shutdown(wait=False)
            raise GPUError(f"Lỗi suy luận GPU tại frame: {e}") from e
        finally:
            writer_executor.shutdown(wait=True)
            if loader_executor:
                loader_executor.shutdown(wait=True)

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
            "distinct_frames": num_distinct,
            "peak_rss_mb": rss_mb,
            "current_alloc_mb": curr_alloc_peak,
            "driver_alloc_mb": driver_alloc_peak
        }


# -----------------------------------------------------------------------------
# 3. Singleton Worker & Điều Phối Fallback
# -----------------------------------------------------------------------------

_GLOBAL_GPU_WORKER: Optional[GPUPersistentWorker] = None

def get_gpu_worker(
    model_id: str,
    weights_path: str,
    precision: str = "fp16",
    max_chunks_before_restart: int = 10,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    use_tensorrt: bool = True,
) -> GPUPersistentWorker:
    """Lấy hoặc khởi tạo instance Singleton cho GPUPersistentWorker."""
    global _GLOBAL_GPU_WORKER
    if (_GLOBAL_GPU_WORKER is None
        or _GLOBAL_GPU_WORKER.model_id != model_id
        or _GLOBAL_GPU_WORKER.precision != precision.lower()
        or (batch_size is not None and _GLOBAL_GPU_WORKER.batch_size != batch_size)):
        _GLOBAL_GPU_WORKER = GPUPersistentWorker(
            model_id=model_id,
            weights_path=weights_path,
            precision=precision,
            max_chunks_before_restart=max_chunks_before_restart,
            device=device,
            batch_size=batch_size,
            use_tensorrt=use_tensorrt,
        )
    else:
        _GLOBAL_GPU_WORKER.use_tensorrt = bool(use_tensorrt)
    return _GLOBAL_GPU_WORKER


def get_cuda_inference_engine(
    model_info: Dict[str, Any],
    precision: str = "fp16",
    batch_size: Optional[int] = None,
    use_tensorrt: bool = True,
) -> GPUPersistentWorker:
    """Khởi tạo hoặc lấy worker CUDA cho mô hình đã chỉ định."""
    return get_gpu_worker(
        model_id=model_info["id"],
        weights_path=model_info["weights_path"],
        precision=precision,
        device="cuda",
        batch_size=batch_size,
        use_tensorrt=use_tensorrt,
    )


def upscale_frames_with_fallback(
    input_dir: str,
    output_dir: str,
    model_info: Dict[str, Any],
    engine: Optional[str] = None,
    precision: str = "fp16",
    batch_size: Optional[int] = None,
    fmt: str = "png",
    logger=None,
    mid_chunk_callback=None,
    use_tensorrt: bool = True,
) -> Dict[str, Any]:
    """
    Hàm giao tiếp chính thực hiện suy luận khung hình cho 1 chunk:
    - Mặc định: PyTorch CUDA FP16 với Batch Inference
    - Tự động fallback khi gặp lỗi:
      + CUDA FP16 -> CUDA FP32
      + animevideov3: fallback sang NCNN Vulkan (-t 540 -j 1:2:2) nếu có NCNN binary
      + animejanai: fallback sang FP32
    - Bảo đảm KHÔNG tự ý đổi sang model khác!
    """
    if engine is None:
        from src.config import DEFAULT_ENGINE
        engine = DEFAULT_ENGINE

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

    # 2. Thử nghiệm trên PyTorch / TensorRT (NVIDIA CUDA / CPU)
    try:
        worker = get_gpu_worker(
            model_id=model_id,
            weights_path=weights_path,
            precision=precision,
            device=engine if engine in ["cuda", "cpu"] else None,
            batch_size=batch_size,
            use_tensorrt=use_tensorrt,
        )
        res = worker.upscale_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            fmt=fmt,
            batch_size=batch_size,
            logger=logger,
            mid_chunk_callback=mid_chunk_callback
        )
        worker.restart_if_needed(logger=logger)
        res["engine_used"] = f"{engine}_{precision}"
        return res

    except Exception as e:
        warn_msg = f"⚠️ [GPU Fallback Activated] Gặp lỗi khi suy luận {engine} {precision.upper()}: {e}."
        if logger:
            logger.warn(warn_msg, to_console=True)
        else:
            print(warn_msg)

        # Fallback 1: Nếu lỗi FP16 trên GPU (CUDA), thử fallback sang FP32 của chính device đó
        if precision == "fp16":
            fallback_msg = f"🔄 Chuyển sang {engine.upper()} FP32 cho model {model_id}..."
            if logger:
                logger.info(fallback_msg, to_console=True)
            else:
                print(fallback_msg)
            worker_fp32 = get_gpu_worker(
                model_id=model_id,
                weights_path=weights_path,
                precision="fp32",
                device=engine if engine in ["cuda", "cpu"] else None,
                batch_size=batch_size
            )
            res = worker_fp32.upscale_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                fmt=fmt,
                batch_size=batch_size,
                logger=logger
            )
            worker_fp32.restart_if_needed(logger=logger)
            res["engine_used"] = f"{engine}_fp32_fallback"
            return res

        # Fallback 2: Nếu có NCNN binary trên hệ thống thì fallback sang NCNN
        from src.config import REALESRGAN_BIN
        if os.path.exists(REALESRGAN_BIN):
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

        raise RuntimeError(f"Lỗi suy luận và không có đường fallback khả dụng cho model '{model_id}': {e}")


# Alias tiện ích
GPUEngineError = GPUError
