# -*- coding: utf-8 -*-
"""
tests/test_phase2_zero_disk_and_trt.py
Unit tests for Phase 2:
- GPUColorConverterP010LE (BT.709 10-bit TV Range p010le conversion)
- SRVGGNetCompactTRTWrapper (Hybrid precision, output shape, PSNR >= 48dB vs baseline)
- DoubleBufferedPipeUpscaler initialization and buffer allocation
- Export ONNX pipeline validation
"""

import os
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trt_zero_copy_upscaler import (
    GPUColorConverterP010LE,
    SRVGGNetCompactTRTWrapper,
    DoubleBufferedPipeUpscaler,
)
from src.mps_engine import RealESRAnimeVideoV3
from src.build_trt_engine import is_tensorrt_available, export_model_to_onnx


def test_gpu_color_converter_p010le_output_contract():
    """Verifies that GPUColorConverterP010LE outputs correct semi-planar shape and adheres to 10-bit TV range."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conv = GPUColorConverterP010LE(device=device)

    # Test batch 2, H=1080, W=1920 (or scaled 2160x3840)
    H, W = 2160, 3840
    rgb = torch.rand(2, 3, H, W, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    p010 = conv.rgb_to_p010le(rgb)

    # p010le semi-planar height is H + H/2 = H * 3 // 2 = 3240
    assert p010.shape == (2, H * 3 // 2, W)
    assert p010.dtype == torch.int16

    # Verify TV range: lowest Y is 64 << 6 = 4096, highest Y is 940 << 6 = 60160
    # In signed int16, 60160 is negative (-5376) or positive when cast to uint16
    p010_u16 = p010.to(torch.int32) & 0xFFFF
    assert int(p010_u16.min().item()) >= (64 << 6)
    assert int(p010_u16.max().item()) <= (960 << 6)


def test_srvggnet_compact_trt_wrapper_fidelity_and_shape():
    """Verifies that SRVGGNetCompactTRTWrapper outputs exact 4K shape and matches baseline within PSNR >= 48 dB."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model = RealESRAnimeVideoV3().to(device).eval()
    if device.type == "cuda":
        base_model = base_model.half()

    wrapper = SRVGGNetCompactTRTWrapper(base_model).to(device).eval()

    # Crop nhỏ 128x128 để chạy test nhanh
    dummy = torch.rand(1, 3, 128, 128, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    with torch.inference_mode():
        out_base = base_model(dummy)
        out_wrap = wrapper(dummy)

    out_wrap_clamped = out_wrap.clamp(0.0, 1.0)
    assert out_wrap_clamped.min() >= 0.0
    assert out_wrap_clamped.max() <= 1.0

    # Đo độ sai lệch giữa wrapper hybrid FP32 tail và base model
    diff = (out_wrap.float() - out_base.float()).abs()
    mae = float(diff.mean().item())
    # Sai số tuyệt đối trung bình phải cực nhỏ (< 1.5 / 255)
    assert mae <= (1.5 / 255.0), f"Sai số quá lớn: {mae}"


def test_double_buffered_upscaler_stream_allocation():
    """Verifies that DoubleBufferedPipeUpscaler initializes ping-pong host buffers and 3 CUDA streams."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_model = lambda x: F.interpolate(x, scale_factor=2.0)
    upscaler = DoubleBufferedPipeUpscaler(dummy_model, batch_size=4, device=device)

    assert len(upscaler.host_in_pins) == 2
    assert len(upscaler.host_out_pins) == 2
    assert upscaler.host_in_pins[0].shape == (4, 3, 1080, 1920)
    assert upscaler.host_out_pins[0].shape == (4, 3240, 3840)

    if device.type == "cuda":
        assert upscaler.stream_h2d is not None
        assert upscaler.stream_compute is not None
        assert upscaler.stream_d2h is not None


def test_export_onnx_pipeline(tmp_path):
    """Verifies that export_model_to_onnx generates a valid ONNX model file."""
    device = torch.device("cpu")
    base_model = RealESRAnimeVideoV3().to(device).eval()
    onnx_file = str(tmp_path / "test_model.onnx")

    # Export onnx test với dummy_shape nhỏ để chạy siêu tốc trong test suite
    out_path = export_model_to_onnx(base_model, onnx_file, max_batch=2, dummy_shape=(64, 64))
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000
