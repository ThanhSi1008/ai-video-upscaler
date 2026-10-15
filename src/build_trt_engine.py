# -*- coding: utf-8 -*-
"""
src/build_trt_engine.py
Biên dịch và quản lý TensorRT FP16 Engine cho RealESR-AnimeVideoV3:
- Tự động xuất kiến trúc SRVGGNetCompactTRTWrapper sang ONNX (opset 18, dynamic batch)
- Biên dịch ONNX sang TensorRT serialized .engine với FP16 Builder Flag và Optimization Profile (1..8..8)
- Cơ chế nạp engine nhanh (< 0.8s) kèm fallback sang PyTorch channels_last FP16 nếu môi trường chưa cài TensorRT.
"""

import os
import sys
import hashlib
from typing import Optional, Any
import torch
import torch.nn as nn

from src.trt_zero_copy_upscaler import SRVGGNetCompactTRTWrapper


def is_tensorrt_available() -> bool:
    """Kiểm tra sự hiện diện của thư viện tensorrt trong môi trường Python."""
    try:
        import tensorrt
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        return False


def get_trt_engine_path(model_id: str = "realesr-animevideov3", max_batch: int = 8) -> str:
    """Tạo đường dẫn file .engine duy nhất gắn liền với GPU và batch_size."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    device_tag = "cuda"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
        device_tag = f"{device_name}"
    return os.path.join(cache_dir, f"{model_id}_{device_tag}_b1-{max_batch}-{max_batch}.engine")


def export_model_to_onnx(
    model: nn.Module,
    onnx_path: str,
    max_batch: int = 8,
    dummy_shape: tuple = (1080, 1920),
    verbose: bool = False
) -> str:
    """
    Xuất mô hình RealESR sang ONNX với dynamic batch:
    - Kích thước đầu vào: [B, 3, H, W] (Mặc định: 1080x1920)
    - Kích thước đầu ra: [B, 3, 2*H, 2*W] (Mặc định: 2160x3840)
    """
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    wrapped = SRVGGNetCompactTRTWrapper(model).eval().to(device)
    dh, dw = dummy_shape
    if device.type == "cuda":
        wrapped = wrapped.half()
        dummy_in = torch.randn(1, 3, dh, dw, device=device, dtype=torch.float16)
    else:
        wrapped = wrapped.float()
        dummy_in = torch.randn(1, 3, dh, dw, device=device, dtype=torch.float32)

    torch.onnx.export(
        wrapped,
        dummy_in,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_rgb"],
        output_names=["output_rgb"],
        dynamic_axes={
            "input_rgb": {0: "batch_size"},
            "output_rgb": {0: "batch_size"},
        },
        verbose=verbose
    )
    return onnx_path


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    max_batch: int = 8,
    workspace_gb: int = 2
) -> Optional[str]:
    """
    Biên dịch file ONNX sang TensorRT FP16 serialized engine.
    Yêu cầu gói tensorrt được cài đặt trong môi trường.
    """
    if not is_tensorrt_available():
        return None

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
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
    # Giới hạn bộ nhớ workspace 2GB theo khuyến nghị kiến trúc
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Cấu hình dynamic batch profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input_rgb",
        (1, 3, 1080, 1920),
        (max_batch, 3, 1080, 1920),
        (max_batch, 3, 1080, 1920),
    )
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        return None

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    return engine_path


class TRTRuntimeEngineWrapper:
    """
    Wrapper thực thi TensorRT Execution Context tích hợp trực tiếp với PyTorch CUDA Tensors.
    """
    def __init__(self, engine_path: str, device: torch.device):
        import tensorrt as trt
        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input: input_tensor [B, 3, 1080, 1920] FP16 trên GPU
        Output: output_tensor [B, 3, 2160, 3840] FP16 trên GPU
        """
        B = input_tensor.shape[0]
        self.context.set_input_shape("input_rgb", (B, 3, 1080, 1920))
        output_tensor = torch.empty((B, 3, 2160, 3840), dtype=torch.float16, device=self.device)

        # Gán địa chỉ con trỏ VRAM trực tiếp (Zero-Copy)
        bindings = [
            int(input_tensor.data_ptr()),
            int(output_tensor.data_ptr())
        ]
        stream_ptr = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream_ptr)
        return output_tensor


def get_hybrid_engine_or_wrapper(
    pytorch_model: nn.Module,
    max_batch: int = 8,
    force_pytorch: bool = False
) -> Any:
    """
    Khởi tạo engine tốt nhất theo môi trường:
    1. Nếu có TensorRT và file .engine đã tồn tại -> Nạp TRTRuntimeEngineWrapper (< 0.8s).
    2. Nếu có TensorRT nhưng chưa có .engine -> Tự động build sang .engine rồi nạp.
    3. Nếu không có TensorRT hoặc force_pytorch=True -> Dùng SRVGGNetCompactTRTWrapper (PyTorch FP16 Channels-Last).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not force_pytorch and is_tensorrt_available() and device.type == "cuda":
        engine_path = get_trt_engine_path(max_batch=max_batch)
        onnx_path = engine_path.replace(".engine", ".onnx")
        if not os.path.exists(engine_path):
            if not os.path.exists(onnx_path):
                export_model_to_onnx(pytorch_model, onnx_path, max_batch=max_batch)
            build_tensorrt_engine(onnx_path, engine_path, max_batch=max_batch)

        if os.path.exists(engine_path):
            try:
                return TRTRuntimeEngineWrapper(engine_path, device=device)
            except Exception as e:
                print(f"⚠️ Không thể nạp TensorRT Engine ({e}), fallback sang PyTorch Wrapper")

    # Fallback tiêu chuẩn: PyTorch SRVGGNetCompactTRTWrapper với channels_last FP16
    wrapper = SRVGGNetCompactTRTWrapper(pytorch_model).eval().to(device)
    if device.type == "cuda":
        wrapper = wrapper.half().to(memory_format=torch.channels_last)
    return wrapper
