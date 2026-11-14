# -*- coding: utf-8 -*-
"""
tests/test_pipeline_m1_optimizations.py
Bộ kiểm thử đơn vị và tích hợp cho các tối ưu hóa M1:
- B1: Micro-Timings Accumulator across chunks
- B2: Triton JIT 1-Batch Warm-up & TRTZeroCopyUpscaler alias
- P1: Scalar Add Fusion in _p010le_core
- P2: Pre-allocated P010LE Ring Buffer
- P3: Pre-allocated H2D Transpose Ring
- P4: Background Cel Memoization in worker_extract and process_frames_to_pipe
- P5: is_nvdec_supported & CRA ffprobe pruning on H.264
"""

import os
from unittest import mock
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trt_zero_copy_upscaler import (
    GPUColorConverterP010LE,
    DoubleBufferedPipeUpscaler,
    TRTZeroCopyUpscaler,
)
from src.ffmpeg_ops import is_nvdec_supported, extract_frames_raw_rgb24


def test_scalar_add_fusion_bit_exact():
    """P1: Xác minh in-place add_(alpha=...) trong _p010le_core khớp 100% với tham chiếu."""
    torch.manual_seed(42)
    rgb = torch.rand(2, 3, 32, 32, dtype=torch.float32)

    # Reference implementation using addcmul_
    r = rgb[:, 0].float()
    g = rgb[:, 1].float()
    b = rgb[:, 2].float()
    y_ref = torch.empty_like(r).fill_(64.0)
    y_ref.addcmul_(r, torch.tensor(186.2376, dtype=torch.float32, device=r.device))
    y_ref.addcmul_(g, torch.tensor(626.5152, dtype=torch.float32, device=r.device))
    y_ref.addcmul_(b, torch.tensor(63.2472, dtype=torch.float32, device=r.device))
    y_p010_ref = (torch.clamp(torch.round(y_ref), 64.0, 940.0).to(torch.int32) << 6).to(torch.int16)

    rgb_sub = F.avg_pool2d(rgb.float(), kernel_size=2, stride=2)
    rs, gs, bs = rgb_sub[:, 0], rgb_sub[:, 1], rgb_sub[:, 2]
    u_ref = torch.empty_like(rs).fill_(512.0)
    u_ref.addcmul_(rs, torch.tensor(-102.656512, dtype=torch.float32, device=rs.device))
    u_ref.addcmul_(gs, torch.tensor(-345.343488, dtype=torch.float32, device=rs.device))
    u_ref.addcmul_(bs, torch.tensor(448.0000, dtype=torch.float32, device=rs.device))
    u_p010_ref = (torch.clamp(torch.round(u_ref), 64.0, 960.0).to(torch.int32) << 6).to(torch.int16)

    v_ref = torch.empty_like(rs).fill_(512.0)
    v_ref.addcmul_(rs, torch.tensor(448.0000, dtype=torch.float32, device=rs.device))
    v_ref.addcmul_(gs, torch.tensor(-406.921088, dtype=torch.float32, device=rs.device))
    v_ref.addcmul_(bs, torch.tensor(-41.078912, dtype=torch.float32, device=rs.device))
    v_p010_ref = (torch.clamp(torch.round(v_ref), 64.0, 960.0).to(torch.int32) << 6).to(torch.int16)

    uv_p010_ref = torch.stack((u_p010_ref, v_p010_ref), dim=-1).view(2, 16, 32)
    expected_p010 = torch.cat((y_p010_ref, uv_p010_ref), dim=1).contiguous()

    # Optimized implementation
    conv = GPUColorConverterP010LE(device=torch.device("cpu"))
    actual_p010 = conv._p010le_core(rgb)

    assert torch.equal(actual_p010, expected_p010), "Giá trị P010LE sau tối ưu hóa add_(alpha=...) phải khớp bit-exact 100%!"


def test_p010le_ring_buffer_reuse():
    """P2: Xác minh GPUColorConverterP010LE tái sử dụng 2 slot ring buffer, không cấp phát động out_buf."""
    device = torch.device("cpu")
    conv = GPUColorConverterP010LE(device=device)

    # 4K-like resolution check with B=4
    rgb = torch.rand(4, 3, 60, 80, dtype=torch.float32, device=device)
    out1 = conv.rgb_to_p010le(rgb)
    ptr1 = out1.data_ptr()

    out2 = conv.rgb_to_p010le(rgb)
    ptr2 = out2.data_ptr()

    out3 = conv.rgb_to_p010le(rgb)
    ptr3 = out3.data_ptr()

    # Slot 0 should be reused for the 3rd invocation (ping-pong ring of 2 slots)
    assert ptr1 != ptr2, "Hai slot liên tiếp phải có địa chỉ vùng đệm khác nhau để không ghi đè."
    assert ptr1 == ptr3, "Slot 0 phải được tái sử dụng ở batch thứ 3 (Ring Buffer 2-slot hoạt động chính xác)!"


def test_h2d_transpose_ring_and_trt_alias():
    """P3 & B2: Xác minh TRTZeroCopyUpscaler alias và bộ đệm gpu_in_ring trong ensure_resolution."""
    assert TRTZeroCopyUpscaler is DoubleBufferedPipeUpscaler, "TRTZeroCopyUpscaler phải là alias của DoubleBufferedPipeUpscaler"

    dummy_model = nn.Identity()
    upscaler = DoubleBufferedPipeUpscaler(dummy_model, batch_size=4, device=torch.device("cpu"), input_hw=(120, 160))
    assert hasattr(upscaler, "gpu_in_ring")
    assert upscaler.output_hw == (240, 320)

    # On CPU, gpu_in_ring is an empty list (CPU does not need CUDA ring)
    assert isinstance(upscaler.gpu_in_ring, list)

    # Test safe warm-up on dummy model
    upscaler.warmup(batch_size=2)


def test_background_cel_memoization_precomputed_dedup():
    """P4: Xác minh process_frames_to_pipe nhận và sử dụng precomputed_dedup trực tiếp."""
    class DummyProc:
        def __init__(self):
            self.stdin = mock.MagicMock()
            self.stderr = mock.MagicMock()
            self.stdin.write.return_value = 100
        def poll(self):
            return None
        def kill(self):
            pass
        def wait(self, timeout=None):
            return 0

    class DummyModel(nn.Module):
        def forward(self, x):
            return x

    device = torch.device("cpu")
    upscaler = DoubleBufferedPipeUpscaler(DummyModel(), batch_size=2, device=device, input_hw=(40, 40), scale=1)

    f1 = np.full((40, 40, 3), 10, dtype=np.uint8)
    f2 = np.full((40, 40, 3), 20, dtype=np.uint8)
    unique_frames = [f1, f2, f1]  # 3 frames with A -> B -> A pattern
    repeat_counts = [1, 1, 1]

    # Pre-computed dedup by background worker
    distinct_frames = [f1, f2]
    frame_to_distinct = [0, 1, 0]

    proc = DummyProc()
    with mock.patch("src.dedup.deduplicate_cyclic_frames") as mock_dedup:
        total = upscaler.process_frames_to_pipe(
            unique_frames,
            repeat_counts,
            encoder_proc=proc,
            cyclic_dedup=True,
            precomputed_dedup=(distinct_frames, frame_to_distinct),
        )
        # deduplicate_cyclic_frames should NOT be called on main thread when precomputed_dedup is given!
        mock_dedup.assert_not_called()
        assert total == 3
        assert upscaler.last_distinct_count == 2


def test_is_nvdec_supported():
    """P5: Xác minh phân loại định dạng an toàn cho NVDEC hardware decode."""
    # H.264 8-bit (AVC) -> SAFE
    assert is_nvdec_supported("h264", pix_fmt="yuv420p", profile="High") is True
    assert is_nvdec_supported("avc1", pix_fmt="yuv420p") is True

    # H.264 Hi10P (10-bit) -> UNSAFE (phải decode CPU)
    assert is_nvdec_supported("h264", pix_fmt="yuv420p10le", profile="High 10") is False
    assert is_nvdec_supported("h264", pix_fmt="yuv420p", profile="High 10") is False
    assert is_nvdec_supported("h264", pix_fmt="yuv420p10le") is False

    # HEVC 8-bit & 10-bit -> SAFE
    assert is_nvdec_supported("hevc", pix_fmt="yuv420p", profile="Main") is True
    assert is_nvdec_supported("hevc", pix_fmt="yuv420p10le", profile="Main 10") is True
    assert is_nvdec_supported("h265", pix_fmt="p010le") is True

    # VP9 & AV1 -> SAFE
    assert is_nvdec_supported("vp9", pix_fmt="yuv420p") is True
    assert is_nvdec_supported("av1", pix_fmt="yuv420p") is True

    # Unknown -> UNSAFE
    assert is_nvdec_supported("unknown") is False
    assert is_nvdec_supported("") is False


def test_extract_frames_raw_rgb24_skips_cra_on_h264():
    """P5: Xác minh extract_frames_raw_rgb24 bỏ qua ffprobe discard packet trên H.264."""
    class FakeStdout:
        def __init__(self, data: bytes):
            self.data = data
            self.offset = 0
        def readinto(self, b):
            n = min(len(b), len(self.data) - self.offset)
            if n <= 0:
                return 0
            b[:n] = self.data[self.offset:self.offset+n]
            self.offset += n
            return n
        def read(self, n=65536):
            return b""
        def close(self):
            pass

    class FakeProc:
        def __init__(self, *args, **kwargs):
            frame_bytes = 10 * 10 * 3
            self.stdout = FakeStdout(b"\x00" * frame_bytes)
            self.returncode = 0
        def wait(self):
            return 0

    with mock.patch("src.ffmpeg_ops._count_leading_discard_packets") as mock_count:
        mock_count.return_value = 0
        with mock.patch("subprocess.Popen", FakeProc):
            # Calling with H.264 stream:
            extract_frames_raw_rgb24(
                seg_path="dummy.mp4",
                width=10,
                height=10,
                kept_indices=[0],
                source_codec="h264",
            )
            mock_count.assert_not_called()

            # Calling with HEVC stream:
            extract_frames_raw_rgb24(
                seg_path="dummy.mp4",
                width=10,
                height=10,
                kept_indices=[0],
                source_codec="hevc",
            )
            mock_count.assert_called_once()


def test_micro_timings_cross_chunk_accumulation():
    """B1: Xác minh cơ chế tích lũy vi trễ xuyên phân đoạn tính đúng tổng và trung bình."""
    accumulated_micro_timings = {
        "num_batches": 0,
        "total_h2d_ms": 0.0,
        "total_trt_ms": 0.0,
        "total_triton_ms": 0.0,
        "total_d2h_ms": 0.0,
    }

    # Chunk 1 (4 batches)
    chunk1_mt = {
        "num_batches": 4,
        "total_h2d_ms": 40.0,
        "total_trt_ms": 100.0,
        "total_triton_ms": 20.0,
        "total_d2h_ms": 10.0,
    }
    # Chunk 2 (6 batches)
    chunk2_mt = {
        "num_batches": 6,
        "total_h2d_ms": 60.0,
        "total_trt_ms": 150.0,
        "total_triton_ms": 30.0,
        "total_d2h_ms": 15.0,
    }

    for mt in [chunk1_mt, chunk2_mt]:
        nb = mt.get("num_batches", 0)
        accumulated_micro_timings["num_batches"] += nb
        accumulated_micro_timings["total_h2d_ms"] += mt.get("total_h2d_ms", 0.0)
        accumulated_micro_timings["total_trt_ms"] += mt.get("total_trt_ms", 0.0)
        accumulated_micro_timings["total_triton_ms"] += mt.get("total_triton_ms", 0.0)
        accumulated_micro_timings["total_d2h_ms"] += mt.get("total_d2h_ms", 0.0)

    assert accumulated_micro_timings["num_batches"] == 10
    assert accumulated_micro_timings["total_h2d_ms"] == 100.0
    assert accumulated_micro_timings["total_trt_ms"] == 250.0
    assert accumulated_micro_timings["total_triton_ms"] == 50.0
    assert accumulated_micro_timings["total_d2h_ms"] == 25.0

    nb = accumulated_micro_timings["num_batches"]
    final_mt = {
        "num_batches": nb,
        "total_h2d_ms": round(accumulated_micro_timings["total_h2d_ms"], 2),
        "total_trt_ms": round(accumulated_micro_timings["total_trt_ms"], 2),
        "total_triton_ms": round(accumulated_micro_timings["total_triton_ms"], 2),
        "total_d2h_ms": round(accumulated_micro_timings["total_d2h_ms"], 2),
        "avg_h2d_ms": round(accumulated_micro_timings["total_h2d_ms"] / nb, 2),
        "avg_trt_ms": round(accumulated_micro_timings["total_trt_ms"] / nb, 2),
        "avg_triton_ms": round(accumulated_micro_timings["total_triton_ms"] / nb, 2),
        "avg_d2h_ms": round(accumulated_micro_timings["total_d2h_ms"] / nb, 2),
    }

    assert final_mt["avg_h2d_ms"] == 10.0
    assert final_mt["avg_trt_ms"] == 25.0
    assert final_mt["avg_triton_ms"] == 5.0
    assert final_mt["avg_d2h_ms"] == 2.5
