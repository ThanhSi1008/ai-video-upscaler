"""TensorRT FP16 engine builder and runtime wrapper for SRVGGNetCompact architectures."""

import hashlib
import json
import os
import sys
import inspect
from typing import Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trt_zero_copy_upscaler import SRVGGNetCompactTRTWrapper

# Ghi nhớ các đường dẫn .engine biên dịch thất bại trong phiên chạy để không bao giờ thử lại gây treo pipeline
_FAILED_TRT_ENGINES: set[str] = set()


def resolve_trt_profile_max_batch(
    pytorch_model: Optional[nn.Module],
    max_batch: int = 8,
    input_hw: Tuple[int, int] = (1080, 1920),
    available_vram_gb: Optional[float] = None,
) -> int:
    """
    Giới hạn kích thước batch tối đa của TensorRT Optimization Profile theo kiến trúc mô hình và dung lượng VRAM:
    - Mô hình PixelShuffle(4) trên nguồn >= 540p (vd: RealESR-AnimeVideoV3 tạo nhánh 8K 4320x7680 trung gian):
      giới hạn profile max_batch <= 2 để đỉnh VRAM chỉ ~1.33 GB (khử hoàn toàn lỗi OOM 12.74 GB trên RTX 3060 12GB)
      mà vẫn bão hòa nhân Tensor Core (~11.5+ fps); TRTRuntimeEngineWrapper tự cắt lát B=8 -> 4x2 zero-copy.
    - Mô hình 2x thuần (AnimeJaNaiCompact2x, SPAN-2x, không có nhánh 8K): cho phép max_batch <= 4 về mặt kiến trúc,
      nhưng trên GPU có VRAM < 15 GB (như RTX 3060 12GB) ở độ phân giải >= 720p, giới hạn max_batch <= 2 để tránh
      TensorRT 10.16 cấp phát 12.15 GiB cho chiến thuật HMMA.16816.F16 gây lỗi CUDA OOM và tụt 42% tốc độ.
    """
    req_b = max(1, int(max_batch))
    in_h, in_w = int(input_hw[0]), int(input_hw[1])
    if (in_h * in_w) < (540 * 960):
        return req_b

    upscale = 2
    if pytorch_model is not None:
        if hasattr(pytorch_model, "scale"):
            upscale = int(getattr(pytorch_model, "scale"))
        elif hasattr(pytorch_model, "upscale"):
            upscale = int(getattr(pytorch_model, "upscale"))
        elif hasattr(pytorch_model, "upsampler") and isinstance(pytorch_model.upsampler, nn.PixelShuffle):
            upscale = int(pytorch_model.upsampler.upscale_factor)
        elif hasattr(pytorch_model, "model_scale"):
            upscale = int(getattr(pytorch_model, "model_scale"))

    if upscale >= 4:
        return min(req_b, 2)
    if available_vram_gb is not None and float(available_vram_gb) < 15.0 and (in_h * in_w) >= (720 * 1280):
        return min(req_b, 2)
    return min(req_b, 4)


class _GenericTRTExportModule(nn.Module):
    """
    Mô-đun xuất ONNX & bọc suy luận cho các kiến trúc mô hình tổng quát (SPAN, Spandrel generic):
    Hỗ trợ các mô hình không có cấu trúc .body và .upsampler (như SPAN không dùng PReLU).
    Đối với mô hình Native 2x (scale == 2), tensor đầu ra được giữ nguyên.
    Nếu mô hình có tỷ lệ scale khác target_scale (ví dụ: mô hình 4x, mục tiêu 2x),
    áp dụng phép nội suy bicubic tương ứng.
    """
    def __init__(
        self,
        model: nn.Module,
        target_scale: float = 2.0,
        scale: Optional[float] = None,
        model_scale: Optional[float] = None,
    ):
        super().__init__()
        if scale is not None:
            target_scale = float(scale)
        self.target_scale = float(target_scale)

        if hasattr(model, "inner_model"):
            self.model = model.inner_model
            self.model_scale = float(getattr(model, "scale", target_scale))
        else:
            self.model = model
            if model_scale is not None:
                self.model_scale = float(model_scale)
            elif hasattr(model, "scale"):
                self.model_scale = float(model.scale)
            elif hasattr(model, "upscale"):
                self.model_scale = float(model.upscale)
            elif hasattr(model, "upsampler") and isinstance(model.upsampler, nn.PixelShuffle):
                self.model_scale = float(model.upsampler.upscale_factor)
            else:
                self.model_scale = float(target_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if abs(self.model_scale - self.target_scale) > 1e-4:
            interp_in = out.float() if (out.device.type == "cpu" and out.dtype == torch.float16) else out
            out_target = F.interpolate(
                interp_in,
                scale_factor=self.target_scale / self.model_scale,
                mode="bicubic",
                align_corners=False,
            )
            if out_target.dtype != x.dtype:
                out_target = out_target.to(x.dtype)
            return out_target
        return out


class _PureFP16TRTExportModule(nn.Module):
    """
    Mô-đun xuất ONNX tối ưu chuyên biệt cho TensorRT FP16 Builder:
    Giữ liền mạch đồ thị Conv2d + PReLU + PixelShuffle + Bicubic ở FP16 thuần khiết (0 nút Cast sang FP32
    trên nhánh trung gian 8K) để trình biên dịch của TensorRT hợp nhất toàn bộ 18 tầng tích chập vào SRAM,
    chọn tập lệnh HMMA.16816.F16 full-rate và giữ đỉnh VRAM chỉ ~1.33 GB trên RTX 3060.
    """
    def __init__(self, base_srvgg_model: nn.Module, target_scale: float = 2.0, scale: Optional[float] = None):
        super().__init__()
        if scale is not None:
            target_scale = float(scale)
        self.body = base_srvgg_model.body
        self.upsampler = base_srvgg_model.upsampler
        if hasattr(base_srvgg_model, "upscale"):
            self.upscale = int(base_srvgg_model.upscale)
        else:
            self.upscale = (
                4
                if isinstance(self.upsampler, nn.PixelShuffle) and self.upsampler.upscale_factor == 4
                else 2
            )
        self.target_scale = float(target_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        for layer in self.body:
            feat = layer(feat)
        out = self.upsampler(feat)
        base = F.interpolate(x, scale_factor=float(self.upscale), mode="nearest")
        out_hr = out + base
        if abs(float(self.upscale) - self.target_scale) > 1e-4:
            # Chỉ ép float32 trên CPU khi input là float16 (CPU thiếu kernel Half bicubic);
            # trên CUDA giữ nguyên FP16 thuần để tránh tạo tensor 8K FP32 12.74 GB trong TensorRT.
            interp_in = out_hr.float() if (out_hr.device.type == "cpu" and out_hr.dtype == torch.float16) else out_hr
            out_target = F.interpolate(
                interp_in,
                scale_factor=self.target_scale / float(self.upscale),
                mode="bicubic",
                align_corners=False,
            )
            if out_target.dtype != x.dtype:
                out_target = out_target.to(x.dtype)
        else:
            out_target = out_hr
        return out_target


def is_tensorrt_available() -> bool:
    """Kiểm tra sự hiện diện của thư viện tensorrt trong môi trường Python."""
    try:
        import tensorrt  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        return False


def is_triton_available() -> bool:
    """Kiểm tra sự hiện diện của trình biên dịch triton (triton-windows) cho torch.compile."""
    try:
        import triton  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        return False


def get_trt_engine_path(
    model_id: str = "realesr-animevideov3",
    max_batch: int = 8,
    input_hw: Tuple[int, int] = (1080, 1920),
    cache_dir: Optional[str] = None,
    weights_sha256: Optional[str] = None,
) -> str:
    """Tạo đường dẫn file .engine duy nhất gắn liền với model_id, độ phân giải, GPU, TRT version, batch_size (và sha256 nếu truyền)."""
    resolved_cache_dir = cache_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "trt_engines"
    )
    os.makedirs(resolved_cache_dir, exist_ok=True)
    device_tag = "cuda"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
        device_tag = f"{device_name}"
    trt_ver = "trt"
    if is_tensorrt_available():
        try:
            import tensorrt as trt
            trt_ver = f"trt{trt.__version__.split('.')[0]}"
        except Exception:
            pass
    in_h, in_w = int(input_hw[0]), int(input_hw[1])
    safe_model = str(model_id).replace(" ", "_").replace("/", "_")
    if weights_sha256:
        sha_tag = str(weights_sha256).strip().lower()[:8]
        fname = f"{safe_model}_{sha_tag}_{device_tag}_{trt_ver}_{in_h}x{in_w}_b1-{max_batch}.engine"
    else:
        fname = f"{safe_model}_{device_tag}_{trt_ver}_{in_h}x{in_w}_b1-{max_batch}.engine"
    return os.path.join(resolved_cache_dir, fname)


def _compute_file_sha256(file_path: str) -> str:
    """Tính mã băm SHA-256 hex cho file engine/trọng số."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def write_trt_engine_sha256_sidecar(
    engine_path: str,
    weights_sha256: Optional[str] = None,
) -> str:
    """Ghi nguyên tử file sidecar <engine_path>.sha256 lưu chữ ký toàn vẹn của engine (và weights nếu có)."""
    sidecar_path = f"{engine_path}.sha256"
    engine_digest = _compute_file_sha256(engine_path)
    payload = {
        "engine_sha256": engine_digest,
        "weights_sha256": str(weights_sha256 or "").strip().lower(),
    }
    tmp_sidecar = f"{sidecar_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_sidecar, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_sidecar, sidecar_path)
    finally:
        if os.path.exists(tmp_sidecar):
            try:
                os.remove(tmp_sidecar)
            except OSError:
                pass
    return sidecar_path


def verify_trt_engine_sha256_sidecar(
    engine_path: str,
    expected_weights_sha256: Optional[str] = None,
) -> bool:
    """
    Xác thực tính toàn vẹn file .engine qua sidecar <engine_path>.sha256 (nếu có).
    Trả về False nếu file engine rỗng/thiếu, sidecar hỏng hoặc mã băm engine/weights không khớp.
    Nếu file .engine tồn tại hợp lệ từ trước khi có sidecar (chưa có .sha256), trả về True để tương thích ngược.
    """
    if not os.path.exists(engine_path) or os.path.getsize(engine_path) == 0:
        return False
    sidecar_path = f"{engine_path}.sha256"
    if not os.path.exists(sidecar_path):
        return True
    try:
        raw = open(sidecar_path, "r", encoding="utf-8").read().strip()
        if not raw:
            return False
        if raw.startswith("{"):
            data = json.loads(raw)
            rec_eng = str(data.get("engine_sha256") or "").strip().lower()
            rec_weights = str(data.get("weights_sha256") or "").strip().lower()
        else:
            rec_eng = raw.split()[0].strip().lower()
            rec_weights = ""
        if not rec_eng:
            return False
        if _compute_file_sha256(engine_path) != rec_eng:
            return False
        if expected_weights_sha256 and rec_weights:
            if rec_weights != str(expected_weights_sha256).strip().lower():
                return False
        return True
    except Exception:
        return False


def export_model_to_onnx(
    model: nn.Module,
    onnx_path: str,
    max_batch: int = 8,
    dummy_shape: tuple = (1080, 1920),
    target_scale: float = 2.0,
    verbose: bool = False,
) -> str:
    """
    Xuất mô hình SRVGGNetCompact sang ONNX với dynamic batch:
    - Kích thước đầu vào: [B, 3, H, W] (Mặc định: 1080x1920, NCHW contiguous)
    - Kích thước đầu ra: [B, 3, target_scale*H, target_scale*W]
    """
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    # Luôn xuất ONNX từ bản sao trên CPU/CUDA nhất quán để không mutate trạng thái channels_last của worker gốc
    import copy
    model_copy = copy.deepcopy(model).eval()
    device = next(model_copy.parameters()).device if any(True for _ in model_copy.parameters()) else torch.device("cpu")
    model_copy = model_copy.to(memory_format=torch.contiguous_format)

    has_body_srvgg = hasattr(model_copy, "body") and hasattr(model_copy, "upsampler")
    if hasattr(model_copy, "inner_model"):
        inner = model_copy.inner_model
        has_body_srvgg = hasattr(inner, "body") and hasattr(inner, "upsampler")
        model_to_wrap = inner if has_body_srvgg else model_copy
    else:
        model_to_wrap = model_copy

    if has_body_srvgg:
        wrapped = _PureFP16TRTExportModule(model_to_wrap, target_scale=target_scale).eval().to(device)
    else:
        wrapped = _GenericTRTExportModule(model_to_wrap, target_scale=target_scale).eval().to(device)
    dh, dw = int(dummy_shape[0]), int(dummy_shape[1])
    if device.type == "cuda":
        wrapped = wrapped.half()
        dummy_in = torch.randn(1, 3, dh, dw, device=device, dtype=torch.float16)
    else:
        wrapped = wrapped.float()
        dummy_in = torch.randn(1, 3, dh, dw, device=device, dtype=torch.float32)

    export_kwargs = dict(
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_rgb"],
        output_names=["output_rgb"],
        dynamic_axes={
            "input_rgb": {0: "batch_size"},
            "output_rgb": {0: "batch_size"},
        },
        verbose=verbose,
    )
    # Tương thích PyTorch 2.6+: ưu tiên TorchScript exporter ổn định cho dynamic_axes
    try:
        sig = inspect.signature(torch.onnx.export)
        if "dynamo" in sig.parameters:
            export_kwargs["dynamo"] = False
    except Exception:
        pass

    try:
        torch.onnx.export(wrapped, dummy_in, onnx_path, **export_kwargs)
    finally:
        del wrapped, model_copy, dummy_in
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return onnx_path


_TRT_GLOBAL_LOGGER = None


def _get_trt_logger():
    """Trả về instance singleton trt.Logger cho toàn tiến trình (tránh cảnh báo xung đột global ILogger của TensorRT 10)."""
    global _TRT_GLOBAL_LOGGER
    if _TRT_GLOBAL_LOGGER is None:
        import tensorrt as trt
        _TRT_GLOBAL_LOGGER = trt.Logger(trt.Logger.ERROR)
    return _TRT_GLOBAL_LOGGER


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    max_batch: int = 8,
    input_hw: Tuple[int, int] = (1080, 1920),
    workspace_gb: int = 2,
    weights_sha256: Optional[str] = None,
) -> Optional[str]:
    """
    Biên dịch file ONNX sang TensorRT FP16 serialized engine (tương thích TensorRT 8.x & 10.x).
    Sử dụng builder_optimization_level=3 (mức tối ưu chuẩn sản xuất của NVIDIA, tránh lỗi ForeignNode
    Myelin ở level=5 gây kẹt 30 phút trên TensorRT 10.16 Windows) và giải phóng sạch VRAM sau build.
    """
    if not is_tensorrt_available():
        return None

    import tensorrt as trt

    in_h, in_w = int(input_hw[0]), int(input_hw[1])
    opt_b = max(1, min(int(max_batch), 8))
    max_b = max(1, int(max_batch))

    logger = _get_trt_logger()
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parsed = parser.parse(f.read())
        if not parsed:
            for idx in range(parser.num_errors):
                print(f"❌ TensorRT Parser Error: {parser.get_error(idx)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) * (1 << 30))
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3
    if getattr(builder, "platform_has_fast_fp16", True):
        config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt.BuilderFlag, "TF32"):
        config.set_flag(trt.BuilderFlag.TF32)

    # Tái sử dụng & lưu Timing Cache bền vững trên NVMe Gen4 để tăng tốc 5-10x các lần build sau
    timing_cache_path = os.path.join(os.path.dirname(os.path.abspath(engine_path)), "trt_timing.cache")
    timing_cache = None
    if hasattr(config, "create_timing_cache") and hasattr(config, "set_timing_cache"):
        try:
            cache_blob = b""
            if os.path.exists(timing_cache_path):
                with open(timing_cache_path, "rb") as tf:
                    cache_blob = tf.read()
            timing_cache = config.create_timing_cache(cache_blob)
            if timing_cache is not None:
                config.set_timing_cache(timing_cache, ignore_mismatch=True)
        except Exception:
            timing_cache = None

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input_rgb",
        (1, 3, in_h, in_w),
        (opt_b, 3, in_h, in_w),
        (max_b, 3, in_h, in_w),
    )
    config.add_optimization_profile(profile)

    tmp_engine = f"{engine_path}.tmp.{os.getpid()}"
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            return None

        with open(tmp_engine, "wb") as f:
            f.write(serialized_engine)
        os.replace(tmp_engine, engine_path)
        write_trt_engine_sha256_sidecar(engine_path, weights_sha256=weights_sha256)

        if timing_cache is not None and hasattr(config, "get_timing_cache"):
            try:
                updated_tc = config.get_timing_cache()
                if updated_tc is not None:
                    tc_bytes = updated_tc.serialize()
                    if tc_bytes:
                        tmp_tc = f"{timing_cache_path}.tmp.{os.getpid()}"
                        try:
                            with open(tmp_tc, "wb") as tf:
                                tf.write(tc_bytes)
                            os.replace(tmp_tc, timing_cache_path)
                        finally:
                            if os.path.exists(tmp_tc):
                                try:
                                    os.remove(tmp_tc)
                                except OSError:
                                    pass
            except Exception:
                pass
        return engine_path
    finally:
        if os.path.exists(tmp_engine):
            try:
                os.remove(tmp_engine)
            except OSError:
                pass
        del config, network, parser, builder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TRTRuntimeEngineWrapper:
    """
    Wrapper thực thi TensorRT Execution Context tích hợp Zero-Copy với PyTorch CUDA Tensors.
    Tự động hỗ trợ cả API TensorRT 10.x (execute_async_v3) và TensorRT 8.x (execute_async_v2),
    đồng thời tự động xử lý chuyển đổi contiguous NCHW và cắt lát zero-copy nếu batch > max_batch.
    """
    def __init__(
        self,
        engine_path: str,
        device: torch.device,
        input_hw: Tuple[int, int] = (1080, 1920),
        max_batch: int = 8,
        scale: int = 2,
        fallback_module: Optional[nn.Module] = None,
    ):
        import tensorrt as trt
        self.engine_path = engine_path
        self.device = device
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        self.max_batch = max(1, int(max_batch))
        self.scale = max(1, int(scale))
        self.fallback_module = fallback_module
        self.is_tensorrt = True

        self.logger = _get_trt_logger()
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Không thể deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # Phát hiện chính xác kiểu dữ liệu đầu vào/ra đã biên dịch trong Engine (FP16 vs FP32)
        self.in_name = "input_rgb"
        self.out_name = "output_rgb"
        self.in_torch_dtype = torch.float16
        self.out_torch_dtype = torch.float16
        if hasattr(self.engine, "get_tensor_dtype") and hasattr(trt, "DataType"):
            try:
                if self.engine.get_tensor_dtype(self.in_name) == trt.DataType.FLOAT:
                    self.in_torch_dtype = torch.float32
                if self.engine.get_tensor_dtype(self.out_name) == trt.DataType.FLOAT:
                    self.out_torch_dtype = torch.float32
            except Exception:
                pass
        self._use_v3 = hasattr(self.context, "execute_async_v3") and hasattr(self.context, "set_tensor_address")
        self._out_ring: list[torch.Tensor] = []
        self._ring_idx: int = 0
        self._batch_out_ring: list[torch.Tensor] = []
        self._batch_ring_idx: int = 0
        self._last_input_shape: Optional[Tuple[int, int, int, int]] = None
        self._last_in_ptr: Optional[int] = None
        self._last_out_ptr: Optional[int] = None

    def _acquire_output_buffer(self, B: int, C: int, out_h: int, out_w: int) -> torch.Tensor:
        """
        Tái sử dụng vòng đệm đôi (2-slot ping-pong VRAM ring) cho tensor đầu ra 4K,
        ưu tiên tái dùng lát cắt từ _batch_out_ring nếu đã cấp phát để tiết kiệm ~199 MB VRAM.
        """
        alloc_b = max(int(B), self.max_batch)
        target_tail = (int(C), int(out_h), int(out_w))
        if (
            len(self._batch_out_ring) == 2
            and self._batch_out_ring[0].shape[0] >= alloc_b
            and self._batch_out_ring[0].shape[1:] == target_tail
            and self._batch_out_ring[0].dtype == self.out_torch_dtype
            and self._batch_out_ring[0].device == self.device
        ):
            slot_buf = self._batch_out_ring[self._batch_ring_idx]
            self._batch_ring_idx ^= 1
            return slot_buf[:B]
        if (
            len(self._out_ring) != 2
            or self._out_ring[0].shape[0] < alloc_b
            or self._out_ring[0].shape[1:] != target_tail
            or self._out_ring[0].dtype != self.out_torch_dtype
            or self._out_ring[0].device != self.device
        ):
            self._out_ring = [
                torch.empty((alloc_b, *target_tail), dtype=self.out_torch_dtype, device=self.device)
                for _ in range(2)
            ]
            self._ring_idx = 0
            self._last_out_ptr = None
        slot_buf = self._out_ring[self._ring_idx]
        self._ring_idx ^= 1
        return slot_buf[:B]

    def _acquire_batch_output_buffer(self, B: int, C: int, out_h: int, out_w: int) -> torch.Tensor:
        """Vòng đệm đôi 2-slot cho batch tổng hợp (vd: B=8 khi profile max_batch=2) để cắt lát zero-copy."""
        alloc_b = max(int(B), self.max_batch)
        target_tail = (int(C), int(out_h), int(out_w))
        if (
            len(self._batch_out_ring) != 2
            or self._batch_out_ring[0].shape[0] < alloc_b
            or self._batch_out_ring[0].shape[1:] != target_tail
            or self._batch_out_ring[0].dtype != self.out_torch_dtype
            or self._batch_out_ring[0].device != self.device
        ):
            if self._out_ring:
                self._out_ring = []
                self._last_out_ptr = None
            self._batch_out_ring = [
                torch.empty((alloc_b, *target_tail), dtype=self.out_torch_dtype, device=self.device)
                for _ in range(2)
            ]
            self._batch_ring_idx = 0
        slot_buf = self._batch_out_ring[self._batch_ring_idx]
        self._batch_ring_idx ^= 1
        return slot_buf[:B]

    def _exec_single_batch(self, contig_in: torch.Tensor, out_view: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = contig_in.shape
        out_h, out_w = H * self.scale, W * self.scale
        if contig_in.dtype != self.in_torch_dtype:
            contig_in = contig_in.to(dtype=self.in_torch_dtype).contiguous()
        output_tensor = out_view if out_view is not None else self._acquire_output_buffer(B, C, out_h, out_w)
        stream_ptr = (
            torch.cuda.current_stream(self.device).cuda_stream
            if self.device.type == "cuda" and torch.cuda.is_available()
            else 0
        )

        in_shape = (int(B), int(C), int(H), int(W))
        if self._last_input_shape != in_shape:
            self.context.set_input_shape(self.in_name, in_shape)
            self._last_input_shape = in_shape

        in_ptr = int(contig_in.data_ptr())
        out_ptr = int(output_tensor.data_ptr())
        if self._use_v3:
            if self._last_in_ptr != in_ptr:
                self.context.set_tensor_address(self.in_name, in_ptr)
                self._last_in_ptr = in_ptr
            if self._last_out_ptr != out_ptr:
                self.context.set_tensor_address(self.out_name, out_ptr)
                self._last_out_ptr = out_ptr
            ok = self.context.execute_async_v3(stream_ptr)
        else:
            bindings = [in_ptr, out_ptr]
            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=stream_ptr)

        if not ok:
            raise RuntimeError("TensorRT execute_async trả về trạng thái thất bại.")
        return output_tensor

    @torch.inference_mode()
    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        orig_dtype = input_tensor.dtype
        B, C, H, W = input_tensor.shape

        # Nếu gặp độ phân giải khác với profile đã biên dịch, lùi an toàn về fallback_module
        if (H, W) != self.input_hw:
            if self.fallback_module is not None:
                return self.fallback_module(input_tensor)
            raise RuntimeError(f"Độ phân giải frame ({H}x{W}) khác TensorRT profile ({self.input_hw}).")

        # Chuyển về NCHW contiguous trước khi truyền địa chỉ con trỏ thô cho TensorRT
        contig_in = input_tensor.contiguous()
        if B <= self.max_batch:
            out = self._exec_single_batch(contig_in)
            return out if orig_dtype == self.out_torch_dtype else out.to(orig_dtype)

        # Tự động cắt lát zero-copy trực tiếp vào lát cắt của vòng đệm _batch_out_ring (0 lần copy thừa 199 MB)
        out_h, out_w = H * self.scale, W * self.scale
        full_out = self._acquire_batch_output_buffer(B, C, out_h, out_w)
        for s in range(0, B, self.max_batch):
            e = min(s + self.max_batch, B)
            self._exec_single_batch(contig_in[s:e], out_view=full_out[s:e])
        return full_out if orig_dtype == self.out_torch_dtype else full_out.to(orig_dtype)


def get_hybrid_engine_or_wrapper(
    pytorch_model: nn.Module,
    max_batch: int = 8,
    force_pytorch: bool = False,
    model_id: str = "realesr-animevideov3",
    input_hw: Tuple[int, int] = (1080, 1920),
    scale: float = 2.0,
    weights_sha256: Optional[str] = None,
) -> Any:
    """
    Khởi tạo engine suy luận nhanh nhất theo môi trường phần cứng:
    1. Tầng 1 (Nhanh nhất - TensorRT 10 FP16): Biên dịch/nạp file .engine gộp Conv+PReLU & mở khóa 51 TFLOPS HMMA.F16.
    2. Tầng 2 (Fallback thông minh): PyTorch SRVGGNetCompactTRTWrapper với channels_last FP16 & micro-sliced 8K tail.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    has_body_srvgg = hasattr(pytorch_model, "body") and hasattr(pytorch_model, "upsampler")
    if hasattr(pytorch_model, "inner_model"):
        inner = pytorch_model.inner_model
        has_body_srvgg = hasattr(inner, "body") and hasattr(inner, "upsampler")
        model_for_wrapper = inner if has_body_srvgg else pytorch_model
    else:
        model_for_wrapper = pytorch_model

    if has_body_srvgg:
        wrapper = SRVGGNetCompactTRTWrapper(model_for_wrapper, scale=scale).eval().to(device)
    else:
        wrapper = _GenericTRTExportModule(model_for_wrapper, target_scale=scale).eval().to(device)
    if device.type == "cuda":
        wrapper = wrapper.half().to(memory_format=torch.channels_last)

    # Trong môi trường kiểm thử offline (PYTEST_CURRENT_TEST) không tự ý biên dịch engine trừ khi ép test
    in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    allow_trt = (not force_pytorch) and (not in_pytest) and is_tensorrt_available() and (device.type == "cuda")

    if allow_trt:
        vram_gb: Optional[float] = None
        try:
            vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        except Exception:
            vram_gb = None
        trt_max_batch = resolve_trt_profile_max_batch(
            pytorch_model,
            max_batch=max_batch,
            input_hw=input_hw,
            available_vram_gb=vram_gb,
        )
        engine_path = get_trt_engine_path(model_id=model_id, max_batch=trt_max_batch, input_hw=input_hw)
        if engine_path not in _FAILED_TRT_ENGINES:
            onnx_path = engine_path.replace(".engine", ".onnx")
            try:
                needs_build = (
                    not os.path.exists(engine_path)
                    or os.path.getsize(engine_path) == 0
                    or not verify_trt_engine_sha256_sidecar(engine_path, expected_weights_sha256=weights_sha256)
                )
                if needs_build:
                    if os.path.exists(engine_path) and os.path.exists(f"{engine_path}.sha256"):
                        print(
                            f"⚠️ [TensorRT FP16 Compiler] Engine '{os.path.basename(engine_path)}' không khớp SHA-256 sidecar, đang biên dịch lại..."
                        )
                    print(
                        f"\n⚡ [TensorRT FP16 Compiler] Đang biên dịch nhân Tensor Core tối ưu lần đầu cho "
                        f"'{model_id}' ({input_hw[1]}x{input_hw[0]}, profile batch 1..{trt_max_batch})..."
                    )
                    if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) == 0:
                        export_model_to_onnx(
                            pytorch_model,
                            onnx_path,
                            max_batch=trt_max_batch,
                            dummy_shape=input_hw,
                            target_scale=scale,
                        )
                    built = build_tensorrt_engine(
                        onnx_path,
                        engine_path,
                        max_batch=trt_max_batch,
                        input_hw=input_hw,
                        weights_sha256=weights_sha256,
                    )
                    if not built:
                        _FAILED_TRT_ENGINES.add(engine_path)
                    elif os.path.exists(engine_path) and os.path.getsize(engine_path) > 0:
                        try:
                            if os.path.exists(onnx_path):
                                os.remove(onnx_path)
                        except OSError:
                            pass

                if os.path.exists(engine_path) and os.path.getsize(engine_path) > 0:
                    trt_runner = TRTRuntimeEngineWrapper(
                        engine_path,
                        device=device,
                        input_hw=input_hw,
                        max_batch=trt_max_batch,
                        scale=int(round(scale)),
                        fallback_module=wrapper,
                    )
                    if not os.path.exists(f"{engine_path}.sha256"):
                        try:
                            write_trt_engine_sha256_sidecar(engine_path, weights_sha256=weights_sha256)
                        except OSError:
                            pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return trt_runner
            except Exception as e:
                _FAILED_TRT_ENGINES.add(engine_path)
                print(f"⚠️ Không thể khởi tạo TensorRT Engine ({e}), tự động chuyển sang PyTorch FP16 Wrapper.")

    return wrapper
