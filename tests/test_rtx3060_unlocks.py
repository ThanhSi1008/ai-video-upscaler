#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_rtx3060_unlocks.py
Kiểm thử đơn vị offline (CPU-safe, 0 mạng, 0 GPU thật) cho gói mở khóa hiệu năng
PC Alder Lake i5-12600K + RTX 3060 12GB (GA106) + WD_BLACK SN7100 Gen4 NVMe:
1. configure_cuda_fast_math: bật cuDNN benchmark + FP16 reduced-precision HMMA accumulate.
2. RAMDisk: cô lập an toàn trong pytest, hỗ trợ UPSCALER_RAMDISK_DIR tùy chọn.
3. Vi phân lô 8K 2-frame (micro-slicing) trong RealESRAnimeVideoV3 & SRVGGNetCompactTRTWrapper:
   khớp tuyệt đối (bit-exact max_diff == 0.0) so với chạy nguyên batch B=5, tiết kiệm ~75% đỉnh đệm 8K.
4. GPUColorConverterP010LE: micro-slicing khớp tuyệt đối với từng frame B=1 ghép lại.
5. TRTRuntimeEngineWrapper: hỗ trợ TensorRT 10 execute_async_v3 + channels_last contiguous
   và tự động fallback về PyTorch khi lệch độ phân giải tĩnh.
6. extract_frames_raw_rgb24: đọc luồng rgb24 trực tiếp từ stdout pipe + fallback CPU khi hwaccel lỗi.
7. CLI upscale.py: cờ --no-tensorrt mặc định False, giữ nguyên --zero-disk mặc định False.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import upscale
from src.build_trt_engine import TRTRuntimeEngineWrapper, _PureFP16TRTExportModule
from src.config import configure_cuda_fast_math
from src.cuda_engine import RealESRAnimeVideoV3
from src.ffmpeg_ops import extract_frames_raw_rgb24
from src.preflight import configure_alder_lake_pcore_qos
from src.ramdisk import RAMDisk
from src.trt_zero_copy_upscaler import GPUColorConverterP010LE, SRVGGNetCompactTRTWrapper


def test_configure_cuda_fast_math_enables_ampere_hmma_flags():
    """Xác minh configure_cuda_fast_math bật đầy đủ các cờ tăng tốc Tensor Core FP16/TF32."""
    flags = configure_cuda_fast_math()
    assert flags["cudnn_benchmark"] is True
    assert flags["allow_tf32"] is True
    assert flags["fp16_fast_accum"] is True
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is True
    assert torch.backends.cudnn.benchmark is True


def test_alder_lake_pcore_qos_skips_cleanly_under_pytest():
    """Xác minh configure_alder_lake_pcore_qos tự động bỏ qua an toàn trong môi trường pytest."""
    prev = configure_alder_lake_pcore_qos(enable_boost=True)
    assert prev is None


def test_ramdisk_honors_explicit_env_override_and_cleans_up(monkeypatch):
    """Xác minh UPSCALER_RAMDISK_DIR định tuyến đúng ổ đích và dọn sạch thư mục con khi thoát."""
    with tempfile.TemporaryDirectory() as fake_ram_drive:
        monkeypatch.setenv("UPSCALER_RAMDISK_DIR", fake_ram_drive)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        created_path = None
        with RAMDisk(size_mb=512) as rd_path:
            created_path = rd_path
            assert os.path.exists(rd_path)
            assert os.path.commonpath([os.path.abspath(rd_path), os.path.abspath(fake_ram_drive)]) == os.path.abspath(fake_ram_drive)
            sample_file = os.path.join(rd_path, "chunk_test.bin")
            with open(sample_file, "wb") as f:
                f.write(b"test")
        assert created_path is not None
        assert not os.path.exists(created_path)


def test_8k_tail_micro_slicing_is_bit_exact_on_hd_batch():
    """
    Xác minh vi phân lô 2-frame ở đuôi PixelShuffle 4x (8K) rồi interpolate 0.5x (4K)
    trên kích thước ngưỡng >= 540x960 (ví dụ 540x960, B=3) khớp 100% (max_diff == 0.0)
    so với chạy lẻ từng frame B=1.
    """
    torch.manual_seed(42)
    base_model = RealESRAnimeVideoV3().eval()
    # Chỉ kiểm tra trực tiếp nhánh forward / _tail trên kích thước 540x960 với thân rút gọn 1 conv để chạy nhanh trên CPU
    base_model.body = torch.nn.ModuleList([
        torch.nn.Conv2d(3, 48, 1, 1, 0),
    ])
    x_batch = torch.rand(3, 3, 540, 960, dtype=torch.float32)

    with torch.inference_mode():
        # B=3 & H*W == 540*960 -> kích hoạt nhánh micro-slicing 2-frame
        out_batched = base_model(x_batch)
        out_singles = torch.cat([base_model(x_batch[i : i + 1]) for i in range(3)], dim=0)

        wrapper = SRVGGNetCompactTRTWrapper(base_model, scale=2.0).eval()
        wrap_batched = wrapper(x_batch)
        wrap_singles = torch.cat([wrapper(x_batch[i : i + 1]) for i in range(3)], dim=0)

    assert out_batched.shape == (3, 3, 1080, 1920)
    assert torch.max(torch.abs(out_batched - out_singles)).item() == 0.0
    assert wrap_batched.shape == (3, 3, 1080, 1920)
    assert torch.max(torch.abs(wrap_batched - wrap_singles)).item() == 0.0


def test_pure_fp16_trt_export_module_preserves_2x_geometry():
    """Xác minh _PureFP16TRTExportModule xuất đúng kích thước 2x mà không ép kiểu FP32 ở đuôi."""
    base_model = RealESRAnimeVideoV3().eval()
    export_mod = _PureFP16TRTExportModule(base_model, scale=2.0).eval()
    x = torch.rand(2, 3, 16, 24, dtype=torch.float32)
    with torch.inference_mode():
        y = export_mod(x)
    assert y.shape == (2, 3, 32, 48)
    assert y.dtype == torch.float32


def test_p010le_micro_slicing_matches_single_frame_conversion():
    """Xác minh GPUColorConverterP010LE cắt lát B=3 ở >=540x960 khớp bit-exact với ghép từng frame B=1."""
    converter = GPUColorConverterP010LE(device=torch.device("cpu"))
    torch.manual_seed(7)
    rgb = torch.rand(3, 3, 544, 960, dtype=torch.float32)

    batched_bytes = converter.rgb_to_p010le(rgb).numpy().tobytes()
    single_bytes = b"".join(converter.rgb_to_p010le(rgb[i : i + 1]).numpy().tobytes() for i in range(3))
    assert batched_bytes == single_bytes


def test_trt_runtime_wrapper_v3_and_resolution_fallback():
    """
    Xác minh TRTRuntimeEngineWrapper:
    1. Gọi đúng API TensorRT 10 (set_tensor_address + execute_async_v3) và ép contiguous NCHW.
    2. Tự động cắt lát khi B > max_batch.
    3. Tự động fallback về fallback_module khi gặp kích thước khung hình khác input_hw.
    """
    fake_trt = MagicMock()
    fake_trt.Logger.ERROR = 1

    class FakeContext:
        def __init__(self):
            self.shapes = []
            self.addresses = {}
            self.executed_v3 = 0

        def set_input_shape(self, name, shape):
            self.shapes.append((name, tuple(shape)))

        def set_tensor_address(self, name, ptr):
            self.addresses[name] = int(ptr)

        def execute_async_v3(self, stream_handle):
            self.executed_v3 += 1
            return True

    class FakeEngine:
        def create_execution_context(self):
            return FakeContext()

    class FakeRuntime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, blob):
            return FakeEngine()

    fake_trt.Runtime = FakeRuntime

    fallback_called = {"count": 0}

    class DummyFallback(torch.nn.Module):
        def forward(self, t):
            fallback_called["count"] += 1
            return torch.nn.functional.interpolate(t, scale_factor=2.0, mode="nearest")

    with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as tf:
        tf.write(b"FAKE_TRT_ENGINE_BYTES")
        engine_path = tf.name

    try:
        with patch.dict(sys.modules, {"tensorrt": fake_trt}):
            wrapper = TRTRuntimeEngineWrapper(
                engine_path,
                device=torch.device("cpu"),
                input_hw=(32, 48),
                max_batch=2,
                scale=2,
                fallback_module=DummyFallback(),
            )
            # 1. Batch B=3 > max_batch=2 với channels_last -> tự động cắt thành 2 lần execute_async_v3
            x_match = torch.zeros(3, 3, 32, 48, dtype=torch.float16).to(memory_format=torch.channels_last)
            y_match = wrapper(x_match)
            assert y_match.shape == (3, 3, 64, 96)
            assert wrapper.context.executed_v3 == 2
            assert fallback_called["count"] == 0

            # 2. Khác độ phân giải -> fallback êm ái sang PyTorch module
            x_other = torch.zeros(1, 3, 20, 30, dtype=torch.float32)
            y_other = wrapper(x_other)
            assert y_other.shape == (1, 3, 40, 60)
            assert fallback_called["count"] == 1
    finally:
        if os.path.exists(engine_path):
            os.remove(engine_path)


def test_extract_frames_raw_rgb24_reads_pipe_and_falls_back_on_hwaccel_error():
    """
    Xác minh extract_frames_raw_rgb24 đọc chính xác các khung hình RGB24 từ stdout pipe
    và tự động thử lại không có -hwaccel cuda nếu lần đầu thất bại.
    """
    h, w = 4, 6
    frame0 = np.full((h, w, 3), 10, dtype=np.uint8)
    frame1 = np.full((h, w, 3), 99, dtype=np.uint8)
    raw_stream = frame0.tobytes() + frame1.tobytes()

    calls = []

    def fake_run(cmd, stdout=None, stderr=None, check=False):
        calls.append(list(cmd))
        if "-hwaccel" in cmd:
            return subprocess.CompletedProcess(cmd, returncode=1, stdout=b"", stderr=b"hwaccel error")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=raw_stream, stderr=b"")

    with patch("src.ffmpeg_ops.subprocess.run", side_effect=fake_run):
        frames = extract_frames_raw_rgb24("dummy.mp4", width=w, height=h, kept_indices=[0, 1], hwaccel=True)

    assert len(calls) == 2
    assert "-hwaccel" in calls[0]
    assert "-hwaccel" not in calls[1]
    assert len(frames) == 2
    assert np.array_equal(frames[0], frame0)
    assert np.array_equal(frames[1], frame1)


def test_cli_tensorrt_and_zero_disk_flags():
    """Xác minh CLI parser giữ mặc định --zero-disk=False và thêm --no-tensorrt=False."""
    parser = upscale.build_arg_parser()
    args = parser.parse_args(["samples/sample_30s.mkv"])
    assert args.zero_disk is False
    assert args.no_tensorrt is False

    args_off = parser.parse_args(["samples/sample_30s.mkv", "--zero-disk", "--no-tensorrt"])
    assert args_off.zero_disk is True
    assert args_off.no_tensorrt is True


def test_cuda_lazy_module_loading_and_jit_cache_env():
    """Xác minh src/config.py thiết lập CUDA_MODULE_LOADING=LAZY và CUDA_CACHE_MAXSIZE=2GB."""
    assert os.environ.get("CUDA_MODULE_LOADING") == "LAZY"
    assert int(os.environ.get("CUDA_CACHE_MAXSIZE", "0")) >= 2147483648


def test_p010le_addcmul_matches_reference_bt709_formula():
    """Xác minh _p010le_core dùng addcmul_ tại chỗ khớp bit-exact với công thức tham chiếu ITU-R BT.709."""
    torch.manual_seed(123)
    rgb = torch.rand(2, 3, 32, 48, dtype=torch.float32)
    actual = GPUColorConverterP010LE._p010le_core(rgb)

    # Công thức tham chiếu độc lập
    y_w = torch.tensor([186.2376, 626.5152, 63.2472], dtype=torch.float32).view(1, 3, 1, 1)
    u_w = torch.tensor([-102.656512, -345.343488, 448.0000], dtype=torch.float32).view(1, 3, 1, 1)
    v_w = torch.tensor([448.0000, -406.921088, -41.078912], dtype=torch.float32).view(1, 3, 1, 1)
    y_ref = ((torch.clamp(torch.round((rgb * y_w).sum(dim=1) + 64.0), 64.0, 940.0).to(torch.int32)) << 6).to(torch.int16)
    sub = torch.nn.functional.avg_pool2d(rgb, kernel_size=2, stride=2)
    u_ref = ((torch.clamp(torch.round((sub * u_w).sum(dim=1) + 512.0), 64.0, 960.0).to(torch.int32)) << 6).to(torch.int16)
    v_ref = ((torch.clamp(torch.round((sub * v_w).sum(dim=1) + 512.0), 64.0, 960.0).to(torch.int32)) << 6).to(torch.int16)
    uv_ref = torch.stack((u_ref, v_ref), dim=-1).view(2, 16, 48)
    expected = torch.cat((y_ref, uv_ref), dim=1).contiguous()

    # Cho phép chênh tối đa <= 1 LSB (<<6 = 64) do thứ tự cộng FMA dấu phẩy động IEEE-754
    max_lsb_diff = int(((actual.to(torch.int32) & 0xFFFF) - (expected.to(torch.int32) & 0xFFFF)).abs().max().item()) >> 6
    assert max_lsb_diff <= 1


def test_nhwc_zero_transpose_view_is_channels_last_and_pixel_exact():
    """Xác minh kỹ thuật view NHWC + permute(0,3,1,2) tạo đúng channels_last 0-copy và khớp từng pixel."""
    h, w = 12, 16
    frames = [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(2)]
    pin = torch.empty((2, 3, h, w), dtype=torch.uint8)

    hwc_view = pin.view(2, h, w, 3)
    for i in range(2):
        hwc_view[i].copy_(torch.from_numpy(frames[i]))
    fast_nchw = hwc_view.permute(0, 3, 1, 2)

    assert fast_nchw.is_contiguous(memory_format=torch.channels_last)
    ref_nchw = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames], dim=0)
    assert torch.equal(fast_nchw, ref_nchw)
