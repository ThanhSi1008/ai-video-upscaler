# -*- coding: utf-8 -*-
"""
tests/test_nvme_cache_and_input_triton.py
Offline CPU-only unit & integration tests (CUDA_VISIBLE_DEVICES=-1, 0 GPU, 0 network) for:
1. NVMe Gen4 (.cache) routing in src/config.py & TEMP/TMP/TMPDIR isolation in scripts/verify_offline.py.
2. Fused 1-pass @triton.jit _nhwc_u8_to_nchw_fp16_norm_kernel + PyTorch fallback bit-exactness & pointer preservation.
3. Pre-allocated CUDA Event pool reuse & last_micro_timings backward-compatible + true PCIe H2D DMA keys.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import scripts.verify_offline as verify_offline
import src.config as app_config
import src.trt_zero_copy_upscaler as zcu
from src.trt_zero_copy_upscaler import (
    DoubleBufferedPipeUpscaler,
    _HAS_TRITON,
    _nhwc_u8_to_nchw_fp16_norm,
)


# =============================================================================
# (a) NVMe .cache ENV DEFAULTS & verify_offline.py TEMP/TMP ISOLATION
# =============================================================================


def test_nvme_jit_cache_constants_point_under_project_root():
    """Verify NVMe JIT cache constants resolve under PROJECT_ROOT / '.cache'."""
    expected_cache_root = app_config.PROJECT_ROOT / ".cache"
    assert app_config.CACHE_ROOT_DIR == expected_cache_root
    assert app_config.CUDA_COMPUTE_CACHE_DIR == expected_cache_root / "cuda_compute_cache"
    assert app_config.TRITON_CACHE_DIR_DEFAULT == expected_cache_root / "triton_cache"
    assert app_config.INDUCTOR_CACHE_DIR_DEFAULT == expected_cache_root / "inductor_cache"
    assert set(app_config.NVME_JIT_CACHE_DEFAULTS.keys()) == {
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
    }


def test_configure_nvme_jit_caches_sets_defaults_and_creates_dirs(tmp_path: Path):
    """Verify configure_nvme_jit_caches populates missing env vars and creates .cache subdirs."""
    fake_env: dict[str, str] = {}
    resolved = app_config.configure_nvme_jit_caches(
        project_root=tmp_path,
        environ=fake_env,
        create_dirs=True,
    )

    expected = {
        "CUDA_CACHE_PATH": str(tmp_path / ".cache" / "cuda_compute_cache"),
        "TRITON_CACHE_DIR": str(tmp_path / ".cache" / "triton_cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / ".cache" / "inductor_cache"),
    }
    assert resolved == expected
    assert fake_env == expected
    for dir_str in expected.values():
        assert Path(dir_str).is_dir()


def test_configure_nvme_jit_caches_preserves_existing_overrides(tmp_path: Path):
    """Verify os.environ.setdefault semantics preserve caller-specified cache directories."""
    custom_triton = tmp_path / "custom_triton"
    fake_env: dict[str, str] = {"TRITON_CACHE_DIR": str(custom_triton)}
    resolved = app_config.configure_nvme_jit_caches(
        project_root=tmp_path,
        environ=fake_env,
        create_dirs=True,
    )

    assert resolved["TRITON_CACHE_DIR"] == str(custom_triton)
    assert custom_triton.is_dir()
    assert resolved["CUDA_CACHE_PATH"] == str(tmp_path / ".cache" / "cuda_compute_cache")
    assert resolved["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / ".cache" / "inductor_cache")


def test_configure_nvme_jit_caches_swallows_oserror_on_readonly_fs(tmp_path: Path, monkeypatch):
    """Verify OSError during directory creation never crashes sandboxed/read-only environments."""
    def deny_mkdir(self, *args, **kwargs):
        raise PermissionError("Read-only filesystem")

    monkeypatch.setattr(Path, "mkdir", deny_mkdir)
    fake_env: dict[str, str] = {}
    resolved = app_config.configure_nvme_jit_caches(
        project_root=tmp_path,
        environ=fake_env,
        create_dirs=True,
    )
    assert resolved["CUDA_CACHE_PATH"].endswith("cuda_compute_cache")
    assert resolved["TRITON_CACHE_DIR"].endswith("triton_cache")
    assert resolved["TORCHINDUCTOR_CACHE_DIR"].endswith("inductor_cache")


def test_verify_offline_isolated_environment_routes_temp_to_scratch(tmp_path: Path):
    """Verify isolated_environment strips system Temp (case-insensitively) and sets TEMP/TMP/TMPDIR."""
    parent = {
        "Temp": r"C:\Users\Admin\AppData\Local\Temp",
        "tmp": r"C:\Users\Admin\AppData\Local\Temp",
        "TMPDIR": "/tmp",
        "GEMINI_API_KEY": "secret_key",
        "PATH": r"C:\Windows\System32",
    }
    nvme_tmp = tmp_path / "agent-validation-123" / "tmp"
    child = verify_offline.isolated_environment(parent, tmp_dir=nvme_tmp)

    assert "GEMINI_API_KEY" not in child
    assert "Temp" not in child and "tmp" not in child
    assert child["TEMP"] == str(nvme_tmp)
    assert child["TMP"] == str(nvme_tmp)
    assert child["TMPDIR"] == str(nvme_tmp)
    assert child["CUDA_VISIBLE_DEVICES"] == "-1"
    assert child["RUN_GPU_TESTS"] == "0"

    # Default (no explicit tmp_dir) still routes under ROOT / "scratch" / "tmp"
    default_child = verify_offline.isolated_environment(parent)
    expected_default_tmp = str(verify_offline.ROOT / "scratch" / "tmp")
    assert default_child["TEMP"] == expected_default_tmp
    assert default_child["TMP"] == expected_default_tmp
    assert default_child["TMPDIR"] == expected_default_tmp


def test_verify_offline_main_creates_tmp_dir_and_spawns_isolated_pytest(tmp_path: Path, monkeypatch):
    """Verify verify_offline.main() creates run_dir/tmp and passes TEMP/TMP/TMPDIR to pytest."""
    captured: dict[str, object] = {}

    class FakeStdout:
        def __iter__(self):
            return iter(["1 passed in 0.01s\n"])

    class FakePopen:
        def __init__(self, command, cwd=None, env=None, **kwargs):
            captured["command"] = list(command)
            captured["cwd"] = Path(cwd)
            captured["env"] = dict(env)
            self.stdout = FakeStdout()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def wait(self):
            return 0

    monkeypatch.setattr(verify_offline, "ROOT", tmp_path)
    monkeypatch.setattr(verify_offline, "snapshot_sources", lambda root, dest: 42)
    monkeypatch.setattr(verify_offline.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(sys, "argv", ["scripts/verify_offline.py"])

    rc = verify_offline.main()
    assert rc == 0

    env = captured["env"]
    assert isinstance(env, dict)
    tmp_dir = Path(env["TEMP"])
    assert tmp_dir.is_dir()
    assert tmp_dir.parent.parent == tmp_path / "scratch"
    assert env["TEMP"] == env["TMP"] == env["TMPDIR"] == str(tmp_dir)
    assert env["CUDA_VISIBLE_DEVICES"] == "-1"

    cmd = captured["command"]
    assert isinstance(cmd, list)
    basetemp_idx = cmd.index("--basetemp")
    basetemp_path = Path(cmd[basetemp_idx + 1])
    assert basetemp_path == tmp_dir / "pytest"


# =============================================================================
# (b) MATHEMATICAL EQUIVALENCE OF _nhwc_u8_to_nchw_fp16_norm & TRITON WRAPPER
# =============================================================================


def test_nhwc_u8_to_nchw_fp16_norm_all_256_values_bit_exact():
    """Verify CPU/PyTorch fallback matches FP32->FP16 reference bit-for-bit across all 0..255 uint8 values."""
    vals = torch.arange(256, dtype=torch.uint8)
    # Shape (3, 17, 23, 3) = 3519 elements > 256 -> covers every uint8 byte value + odd spatial dims
    nhwc = vals.repeat((3 * 17 * 23 * 3 // 256) + 1)[: 3 * 17 * 23 * 3].view(3, 17, 23, 3).contiguous()

    actual = _nhwc_u8_to_nchw_fp16_norm(nhwc, use_triton=False)
    expected = (nhwc.permute(0, 3, 1, 2).contiguous().to(torch.float32) * (1.0 / 255.0)).to(torch.float16)

    assert actual.shape == (3, 3, 17, 23)
    assert actual.dtype == torch.float16
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)

    # Exact endpoints: 0 -> 0.0, 255 -> 1.0
    endpoints = torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8)
    ep_out = _nhwc_u8_to_nchw_fp16_norm(endpoints, use_triton=False)
    assert float(ep_out[0, 0, 0, 0]) == 0.0
    assert float(ep_out[0, 2, 0, 0]) == 1.0


def test_nhwc_u8_to_nchw_fp16_norm_preserves_fixed_ring_pointer():
    """Verify writing into pre-allocated gpu_in_ring[slot][:cur_B] preserves exact base data_ptr."""
    ring_buf = torch.empty((8, 3, 36, 64), dtype=torch.float16)
    base_ptr = ring_buf.data_ptr()

    for cur_b in (1, 3, 8):
        nhwc = torch.randint(0, 256, (cur_b, 36, 64, 3), dtype=torch.uint8)
        in_slice = ring_buf[:cur_b]
        out = _nhwc_u8_to_nchw_fp16_norm(nhwc, out_nchw_fp16=in_slice, use_triton=False)
        assert out.data_ptr() == base_ptr
        assert out.shape == (cur_b, 3, 36, 64)
        ref = (nhwc.permute(0, 3, 1, 2).float() * (1.0 / 255.0)).to(torch.float16)
        assert torch.equal(out, ref)


def test_nhwc_u8_to_nchw_fp16_norm_triton_launch_grid_and_strides(monkeypatch):
    """Verify Triton launch wrapper computes exact grid, strides, BLOCK_P, and preserves out_buf pointer."""
    launch_calls: list[dict[str, object]] = []

    class FakeTritonKernel:
        def __getitem__(self, grid):
            def _launcher(in_t, out_t, hw, s_in_b, s_out_b, s_out_c, BLOCK_P, num_warps=4):
                launch_calls.append(
                    {
                        "grid": grid,
                        "hw": hw,
                        "s_in_b": s_in_b,
                        "s_out_b": s_out_b,
                        "s_out_c": s_out_c,
                        "BLOCK_P": BLOCK_P,
                        "num_warps": num_warps,
                        "out_ptr": out_t.data_ptr(),
                    }
                )
                # Populate output with exact reference math to verify end-to-end return value
                out_t.copy_(in_t.permute(0, 3, 1, 2))
                out_t.mul_(1.0 / 255.0)

            return _launcher

    class FakeTritonMod:
        @staticmethod
        def cdiv(x: int, y: int) -> int:
            return (x + y - 1) // y

    monkeypatch.setattr(zcu, "_HAS_TRITON", True)
    monkeypatch.setattr(zcu, "_nhwc_u8_to_nchw_fp16_norm_kernel", FakeTritonKernel())
    monkeypatch.setattr(zcu, "triton", FakeTritonMod)

    nhwc = torch.randint(0, 256, (5, 36, 64, 3), dtype=torch.uint8)
    ring_slot = torch.empty((8, 3, 36, 64), dtype=torch.float16)
    expected_ptr = ring_slot.data_ptr()

    out = _nhwc_u8_to_nchw_fp16_norm(nhwc, out_nchw_fp16=ring_slot[:5], use_triton=True)
    assert out.data_ptr() == expected_ptr
    assert len(launch_calls) == 1
    call = launch_calls[0]
    assert call["hw"] == 36 * 64  # 2304 pixels
    assert call["BLOCK_P"] == 1024
    assert call["grid"] == (3, 5)  # cdiv(2304, 1024) == 3, B == 5
    assert call["s_in_b"] == 36 * 64 * 3
    assert call["s_out_b"] == 3 * 36 * 64
    assert call["s_out_c"] == 36 * 64

    ref = (nhwc.permute(0, 3, 1, 2).float() * (1.0 / 255.0)).to(torch.float16)
    assert torch.equal(out, ref)

    # Non-planar output buffer (e.g. channels_last) cleanly falls back to PyTorch without calling Triton kernel
    cl_out = torch.empty((5, 3, 36, 64), dtype=torch.float16).to(memory_format=torch.channels_last)
    out_cl = _nhwc_u8_to_nchw_fp16_norm(nhwc, out_nchw_fp16=cl_out, use_triton=True)
    assert len(launch_calls) == 1  # Not incremented
    assert torch.equal(out_cl, ref)


def test_nhwc_u8_to_nchw_fp16_norm_triton_cpu_interpreter_bit_exact():
    """Execute the real @triton.jit kernel on CPU via TRITON_INTERPRET=1 (CUDA_VISIBLE_DEVICES=-1) if Triton is installed."""
    if not _HAS_TRITON:
        pytest.skip("triton package not installed in this environment")

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["TRITON_INTERPRET"] = "1"
    env["PYTHONUTF8"] = "1"

    code = (
        "import torch\n"
        "from src.trt_zero_copy_upscaler import _nhwc_u8_to_nchw_fp16_norm\n"
        "vals = torch.arange(256, dtype=torch.uint8)\n"
        "x = vals.repeat(14)[: 2 * 19 * 29 * 3].view(2, 19, 29, 3).contiguous()\n"
        "ring = torch.empty((4, 3, 19, 29), dtype=torch.float16)\n"
        "ptr0 = ring.data_ptr()\n"
        "y_tri = _nhwc_u8_to_nchw_fp16_norm(x, out_nchw_fp16=ring[:2], use_triton=True)\n"
        "y_ref = _nhwc_u8_to_nchw_fp16_norm(x, use_triton=False)\n"
        "assert y_tri.data_ptr() == ptr0\n"
        "assert torch.equal(y_tri, y_ref), f'Max diff: {(y_tri.float() - y_ref.float()).abs().max().item()}'\n"
        "print('TRITON_INTERPRET_OK')\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert res.returncode == 0, f"Subprocess failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "TRITON_INTERPRET_OK" in res.stdout


def test_nhwc_u8_to_nchw_fp16_norm_validation_and_zero_batch():
    """Verify shape/dtype validation errors and safe empty-batch handling."""
    with pytest.raises(ValueError, match="NHWC"):
        _nhwc_u8_to_nchw_fp16_norm(torch.zeros((2, 3, 16, 16), dtype=torch.uint8))

    valid_in = torch.zeros((2, 16, 16, 3), dtype=torch.uint8)
    with pytest.raises(ValueError, match="out_nchw_fp16"):
        _nhwc_u8_to_nchw_fp16_norm(valid_in, out_nchw_fp16=torch.zeros((2, 3, 16, 16), dtype=torch.float32))

    empty_in = torch.zeros((0, 16, 16, 3), dtype=torch.uint8)
    empty_out = _nhwc_u8_to_nchw_fp16_norm(empty_in, use_triton=False)
    assert empty_out.shape == (0, 3, 16, 16)
    assert empty_out.dtype == torch.float16


# =============================================================================
# (c) PRE-ALLOCATED CUDA EVENT POOL & last_micro_timings DMA + BACKWARD COMPAT
# =============================================================================


REQUIRED_LEGACY_TIMING_KEYS = {
    "num_batches",
    "total_h2d_ms",
    "total_trt_ms",
    "total_triton_ms",
    "total_d2h_ms",
    "avg_h2d_ms",
    "avg_trt_ms",
    "avg_triton_ms",
    "avg_d2h_ms",
}

REQUIRED_NEW_DMA_KEYS = {
    "total_h2d_dma_ms",
    "avg_h2d_dma_ms",
    "total_in_norm_ms",
    "avg_in_norm_ms",
}


def test_last_micro_timings_cpu_fallback_preserves_legacy_and_adds_dma_keys():
    """Verify CPU path emits 100% backward-compatible keys plus new DMA & input norm keys."""
    upscaler = DoubleBufferedPipeUpscaler(
        lambda x: F.interpolate(x, scale_factor=2.0, mode="nearest"),
        batch_size=2,
        device=torch.device("cpu"),
        input_hw=(32, 48),
        scale=2,
        enable_micro_timings=True,
    )
    fake_proc = mock.MagicMock()
    fake_proc.stdin.write.side_effect = lambda chunk: len(chunk)

    frames = [np.full((32, 48, 3), i * 15, dtype=np.uint8) for i in range(5)]
    repeats = [1, 2, 1, 1, 1]
    written = upscaler.process_frames_to_pipe(frames, repeats, fake_proc, cyclic_dedup=False)

    assert written == sum(repeats)
    mt = upscaler.last_micro_timings
    assert mt is not None
    assert REQUIRED_LEGACY_TIMING_KEYS.issubset(mt.keys())
    assert REQUIRED_NEW_DMA_KEYS.issubset(mt.keys())
    assert mt["num_batches"] == 3  # ceil(5 / 2)
    assert mt["total_h2d_dma_ms"] == 0.0
    assert mt["avg_h2d_dma_ms"] == 0.0
    assert mt["total_in_norm_ms"] >= 0.0
    assert mt["avg_in_norm_ms"] >= 0.0


def test_cuda_event_pool_reuse_and_true_pcie_dma_vs_sm_wait_separation(monkeypatch):
    """
    Simulate the 3-stream CUDA dispatch loop completely offline on CPU (0 GPU used) to verify:
    1. _ensure_cuda_event_pools pre-allocates reusable Event pools and makes ZERO new Event() calls on Chunk 2.
    2. evt_h2d_dma_e records pure PCIe H2D DMA (~1.6ms) separately from SM queue wait + input norm (~48.4ms),
       while preserving avg_h2d_ms (~50.0ms) and all legacy keys 100% intact.
    """
    created_events: list[FakeCudaEvent] = []
    clock = {"h2d": 0.0, "compute": 0.0, "d2h": 0.0}

    class FakeStream:
        def __init__(self, name: str):
            self.name = name

        def wait_stream(self, other: FakeStream) -> None:
            clock[self.name] = max(clock[self.name], clock[other.name])

    class FakeCudaEvent:
        def __init__(self, enable_timing: bool = False):
            self.enable_timing = enable_timing
            self.ts: float = 0.0
            created_events.append(self)

        def record(self, stream: FakeStream) -> None:
            self.ts = clock[stream.name]

        def elapsed_time(self, end_event: FakeCudaEvent) -> float:
            return float(end_event.ts - self.ts)

        def synchronize(self) -> None:
            pass

    @contextlib.contextmanager
    def fake_cuda_stream_ctx(stream: FakeStream):
        yield stream

    class FakeDevice:
        type = "cuda"

        def __eq__(self, other):
            return getattr(other, "type", None) == "cuda"

    upscaler = DoubleBufferedPipeUpscaler(
        nn.Identity(),
        batch_size=2,
        device=torch.device("cpu"),
        input_hw=(16, 24),
        scale=1,
        enable_micro_timings=True,
    )

    # Swap into simulated CUDA mode backed by CPU ring tensors (100% offline, 0 CUDA calls)
    upscaler.device = FakeDevice()  # type: ignore[assignment]
    upscaler.stream_h2d = FakeStream("h2d")  # type: ignore[assignment]
    upscaler.stream_compute = FakeStream("compute")  # type: ignore[assignment]
    upscaler.stream_d2h = FakeStream("d2h")  # type: ignore[assignment]
    upscaler.gpu_in_ring = [torch.empty((2, 3, 16, 24), dtype=torch.float16) for _ in range(2)]
    upscaler.gpu_nhwc_ring = [torch.empty((2, 16, 24, 3), dtype=torch.uint8) for _ in range(2)]

    # Advance simulated stream clocks on DMA copy, input norm, TRT model, P010LE Triton, and D2H copy
    real_norm = zcu._nhwc_u8_to_nchw_fp16_norm

    def timed_norm(nhwc_u8, out_nchw_fp16=None, use_triton=None):
        # Advance h2d clock by 1.6ms for preceding PCIe DMA before norm's end event?
        # Wait: evt_h2d_dma_e was ALREADY recorded right before timed_norm is called!
        # Let's check: we advance +1.6ms during gpu_nhwc.copy_ and +48.4ms during timed_norm!
        clock["h2d"] += 48.4
        return real_norm(nhwc_u8, out_nchw_fp16=out_nchw_fp16, use_triton=False)

    class SpyNHWCSlice:
        """Wraps CPU uint8 tensor to advance simulated PCIe DMA clock by +1.60ms on copy_."""
        def __init__(self, backing: torch.Tensor):
            self._backing = backing

        def __getitem__(self, item):
            return SpyNHWCSlice(self._backing[item])

        def copy_(self, src: torch.Tensor, non_blocking: bool = False):
            self._backing.copy_(src)
            clock["h2d"] += 1.6
            return self._backing

        @property
        def ndim(self):
            return self._backing.ndim

        @property
        def shape(self):
            return self._backing.shape

        @property
        def dtype(self):
            return self._backing.dtype

        @property
        def is_cuda(self):
            return False

        @property
        def device(self):
            return self._backing.device

        def stride(self, dim: int):
            return self._backing.stride(dim)

        def permute(self, *dims):
            return self._backing.permute(*dims)

    upscaler.gpu_nhwc_ring = [SpyNHWCSlice(t) for t in upscaler.gpu_nhwc_ring]  # type: ignore[misc]

    def timed_model(x: torch.Tensor) -> torch.Tensor:
        clock["compute"] += 70.0
        return x.float()

    upscaler.model = timed_model
    real_p010 = upscaler.color_conv.rgb_to_p010le

    def timed_p010(rgb_t: torch.Tensor, slot=None) -> torch.Tensor:
        out = real_p010(rgb_t, slot=slot)
        clock["compute"] += 1.8
        return out

    upscaler.color_conv.rgb_to_p010le = timed_p010  # type: ignore[method-assign]

    class SpyHostOutPin:
        def __init__(self, backing: torch.Tensor):
            self._backing = backing

        def __getitem__(self, item):
            sub = self._backing[item]

            class _Slice:
                def copy_(self_s, src: torch.Tensor, non_blocking: bool = False):
                    sub.copy_(src)
                    clock["d2h"] += 3.2
                    return sub

                def numpy(self_s):
                    return sub.numpy()

            return _Slice()

    upscaler.host_out_pins = [SpyHostOutPin(t) for t in upscaler.host_out_pins]  # type: ignore[misc]

    monkeypatch.setattr(torch.cuda, "Event", FakeCudaEvent)
    monkeypatch.setattr(torch.cuda, "stream", fake_cuda_stream_ctx)
    monkeypatch.setattr(zcu, "_nhwc_u8_to_nchw_fp16_norm", timed_norm)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, s: None)

    fake_proc = mock.MagicMock()
    fake_proc.stdin.write.side_effect = lambda chunk: len(chunk)
    frames = [np.full((16, 24, 3), k * 20, dtype=np.uint8) for k in range(4)]
    repeats = [1, 1, 1, 1]

    # Chunk 1: 4 frames / batch_size 2 = 2 batches -> allocates pool once (2 sync + 2*9 timing = 20 events)
    w1 = upscaler.process_frames_to_pipe(frames, repeats, fake_proc, cyclic_dedup=False)
    assert w1 == 4
    events_after_chunk1 = len(created_events)
    assert events_after_chunk1 == 2 * (1 + 9)

    mt1 = upscaler.last_micro_timings
    assert mt1 is not None
    assert REQUIRED_LEGACY_TIMING_KEYS.issubset(mt1.keys())
    assert REQUIRED_NEW_DMA_KEYS.issubset(mt1.keys())
    assert mt1["num_batches"] == 2
    assert mt1["avg_h2d_dma_ms"] == pytest.approx(1.6, abs=0.01)
    assert mt1["total_h2d_dma_ms"] == pytest.approx(3.2, abs=0.01)
    assert mt1["avg_in_norm_ms"] == pytest.approx(48.4, abs=0.01)
    assert mt1["total_in_norm_ms"] == pytest.approx(96.8, abs=0.01)
    assert mt1["avg_h2d_ms"] == pytest.approx(50.0, abs=0.01)
    assert mt1["total_h2d_ms"] == pytest.approx(100.0, abs=0.01)
    assert mt1["avg_trt_ms"] == pytest.approx(70.0, abs=0.01)
    assert mt1["avg_triton_ms"] == pytest.approx(1.8, abs=0.01)
    assert mt1["avg_d2h_ms"] == pytest.approx(3.2, abs=0.01)

    # Chunk 2: same 2 batches -> MUST reuse pre-allocated event pool (0 new cudaEventCreate calls!)
    w2 = upscaler.process_frames_to_pipe(frames, repeats, fake_proc, cyclic_dedup=False)
    assert w2 == 4
    assert len(created_events) == events_after_chunk1, "CUDA Event pool must be reused with 0 new Event() allocations!"
