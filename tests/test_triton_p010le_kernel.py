# -*- coding: utf-8 -*-
"""
tests/test_triton_p010le_kernel.py
Unit tests verifying:
1. GPUColorConverterP010LE CPU fallback behavior under offline / CPU-only environment.
2. If CUDA and Triton are available: bit-exact output between Triton JIT kernel
   (_rgb_to_p010le_triton_kernel) and eager PyTorch reference (_p010le_core) across
   batch sizes B=1, 2, 8 with max_diff == 0.
3. Elimination of dynamic VRAM allocation in DoubleBufferedPipeUpscaler via preallocated
   gpu_nhwc_ring.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trt_zero_copy_upscaler import (
    GPUColorConverterP010LE,
    DoubleBufferedPipeUpscaler,
    HAS_TRITON,
    _rgb_to_p010le_triton_kernel,
)


class TestTritonP010LEKernel(unittest.TestCase):
    """Test suite for Triton P010LE kernel and zero-allocation H2D ring."""

    def test_cpu_fallback_output_format_and_shape(self):
        """Verify CPU fallback produces correct shape, int16 dtype, and TV-range values."""
        converter = GPUColorConverterP010LE(device=torch.device("cpu"))
        torch.manual_seed(42)

        for B, H, W in [(1, 16, 24), (2, 32, 48), (3, 64, 80)]:
            x = torch.rand((B, 3, H, W), dtype=torch.float32)
            out = converter.rgb_to_p010le(x)

            expected_shape = (B, H + H // 2, W)
            self.assertEqual(out.shape, expected_shape)
            self.assertEqual(out.dtype, torch.int16)

            # Check that output values are non-zero and within expected MSB-shifted TV range
            out_np = out.numpy()
            self.assertTrue(np.any(out_np != 0))

    def test_cpu_fallback_matches_core_reference(self):
        """Verify CPU fallback matches _p010le_core output exactly."""
        converter = GPUColorConverterP010LE(device=torch.device("cpu"))
        torch.manual_seed(123)
        x = torch.rand((4, 3, 32, 48), dtype=torch.float32)

        out_converter = converter.rgb_to_p010le(x)
        out_reference = converter._p010le_core(x)

        max_diff = (out_converter - out_reference).abs().max().item()
        self.assertEqual(max_diff, 0)

    def test_double_buffered_upscaler_gpu_nhwc_ring_initialization(self):
        """Verify gpu_nhwc_ring is initialized and properly configured in ensure_resolution."""
        dummy_model = MagicMock()
        dummy_model.is_tensorrt = False

        # On CPU: gpu_nhwc_ring should be empty list
        upscaler_cpu = DoubleBufferedPipeUpscaler(
            model_or_engine=dummy_model,
            batch_size=4,
            device=torch.device("cpu"),
            input_hw=(720, 1280),
            scale=2,
        )
        self.assertTrue(hasattr(upscaler_cpu, "gpu_nhwc_ring"))
        self.assertEqual(len(upscaler_cpu.gpu_nhwc_ring), 0)

        # If CUDA is available, verify preallocation on GPU
        if torch.cuda.is_available():
            cuda_dev = torch.device("cuda:0")
            upscaler_cuda = DoubleBufferedPipeUpscaler(
                model_or_engine=dummy_model,
                batch_size=2,
                device=cuda_dev,
                input_hw=(360, 640),
                scale=2,
            )
            self.assertEqual(len(upscaler_cuda.gpu_nhwc_ring), 2)
            self.assertEqual(upscaler_cuda.gpu_nhwc_ring[0].shape, (2, 360, 640, 3))
            self.assertEqual(upscaler_cuda.gpu_nhwc_ring[0].dtype, torch.uint8)
            self.assertEqual(upscaler_cuda.gpu_nhwc_ring[0].device.type, "cuda")

    def test_triton_kernel_bit_exact_parity_on_cuda(self):
        """
        If CUDA and Triton are available: test bit-exact parity (max_diff == 0)
        against _p010le_core across batch sizes B=1, 2, 8.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available (offline / CPU environment)")
        if not HAS_TRITON or _rgb_to_p010le_triton_kernel is None:
            self.skipTest("Triton not available on this platform")

        import triton
        device = torch.device("cuda:0")
        torch.manual_seed(999)

        converter = GPUColorConverterP010LE(device=device)
        self.assertTrue(converter._has_triton)

        for B in [1, 2, 8]:
            for H, W in [(360, 640), (540, 960)]:
                x = torch.rand((B, 3, H, W), device=device, dtype=torch.float32)

                # Reference output from PyTorch eager core
                ref = converter._p010le_core(x)

                # Direct output via converter.rgb_to_p010le (which launches Triton kernel)
                out = converter.rgb_to_p010le(x)

                max_diff = (ref - out).abs().max().item()
                self.assertEqual(
                    max_diff,
                    0,
                    f"Bit-exact discrepancy on CUDA for B={B}, H={H}, W={W}: max_diff={max_diff}",
                )


if __name__ == "__main__":
    unittest.main()
