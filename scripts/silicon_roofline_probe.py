#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/silicon_roofline_probe.py
Standalone empirical silicon ceiling micro-benchmark for NVIDIA Ampere GA106 (RTX 3060 12GB).

Measures physical hardware ceilings directly on local silicon (~30-35s total):
1. Sustained IEEE FP32 CUDA Core GEMM TFLOPS (4096^3, allow_tf32=False) + live clock/power.
2. Sustained Dense 3rd-Gen Tensor Core FP16 GEMM TFLOPS (NVIDIA GA10x Whitepaper verified):
   - FP32 accumulate (HMMA.16816.F32, GeForce half-rate 256 FMAs/SM/clk = 25.48 TFLOPS @ 1777 MHz):
     measured via torch.matmul(allow_fp16_reduced_precision_reduction=False/True) & cublasGemmEx(32F).
   - FP16 accumulate (HMMA.16816.F16, GeForce full-rate 512 FMAs/SM/clk = 50.95 TFLOPS @ 1777 MHz;
     101.90 TFLOPS requires 2:4 structured sparsity): measured via cublasGemmEx(CUBLAS_COMPUTE_16F)
     with numerical cosine-similarity verification, plus isolated 8x1080p Conv2d(64->64, 3x3) FP16
     throughput showing activation-DRAM + N=64 tile efficiency (~28 TFLOPS realized in SR convs).
3. Sustained GDDR6 VRAM Bandwidth (GB/s) via 256 MiB clone/copy & STREAM triad (a = b + s * c)
   vs theoretical 360.0 GB/s @ 7,501 MHz (192-bit bus).
4. Sustained PCIe 4.0 x16 Pinned DMA Host->Device (8x 1080p RGB24 = 49.77 MB) and
   Device->Host (8x 4K P010LE = 199.07 MB) latency (ms) and bandwidth (GB/s).
5. Isolated @triton.jit Input Norm (8x 1080p uint8 NHWC -> float16 NCHW) and Output BT.709
   P010LE (8x 4K float16 NCHW -> int16 P010LE) kernel latencies (ms/batch) and effective GB/s.
6. Isolated 7th-Gen NVENC 4K HEVC Main10 (p010le, preset p6, CQ 18, lookahead 32, spatial/temporal AQ)
   raw stdin pipe ceiling FPS (production -multipass fullres 2-pass vs single-pass -multipass disabled).

Safety:
- Never initializes CUDA or runs GPU workloads on import (safe for AST/pytest collection).
- Acquires exclusive .gpu_active.lock via src.config.gpu_exclusivity_lock.
- Never calls any external/cloud API.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Consumer Ampere GA106-302-A1 (RTX 3060 12GB) architectural constants
# Reference: NVIDIA Ampere GA102 GPU Architecture Whitepaper V2.1, Table 1 & Table 3:
# - Shader FP32: 128 FMAs/SM/clk (256 FLOPs/SM/clk) -> 12.74 TFLOPS @ 1,777 MHz
# - Dense Tensor FP16 w/ FP32 Accum (HMMA.16816.F32, GeForce half-rate): 256 FMAs/SM/clk (512 FLOPs/SM/clk)
#   -> 25.48 TFLOPS dense @ 1,777 MHz (50.95 TFLOPS w/ 2:4 sparsity)
# - Dense Tensor FP16 w/ FP16 Accum (HMMA.16816.F16, GeForce full-rate): 512 FMAs/SM/clk (1,024 FLOPs/SM/clk)
#   -> 50.95 TFLOPS dense @ 1,777 MHz (101.90 TFLOPS ONLY w/ 2:4 sparsity)
GA106_SMS = 28
GA106_FP32_FMAS_PER_SM_CLK = 128
GA106_TC_FP32_ACCUM_DENSE_FMAS_PER_SM_CLK = 256
GA106_TC_FP16_ACCUM_DENSE_FMAS_PER_SM_CLK = 512
GA106_RATED_BOOST_MHZ = 1777.0
GA106_THEORETICAL_VRAM_GB_S = 360.0  # 192-bit / 8 * 15.0 Gbps (7501 MHz GDDR6 P0)
PCIE_GEN4_X16_UNIDIR_GB_S = 31.508   # 16 lanes * 16 GT/s * (128/130) / 8


def _theoretical_ga106_tflops(clock_mhz: float) -> Dict[str, float]:
    """Compute official NVIDIA Ampere GA10x Whitepaper dense & 2:4 sparse TFLOPS at clock_mhz."""
    clk_ghz = float(clock_mhz) / 1000.0
    fp32_dense = GA106_SMS * GA106_FP32_FMAS_PER_SM_CLK * 2 * clk_ghz / 1000.0
    tc_fp16_fp32_accum_dense = (
        GA106_SMS * GA106_TC_FP32_ACCUM_DENSE_FMAS_PER_SM_CLK * 2 * clk_ghz / 1000.0
    )
    tc_fp16_fp16_accum_dense = (
        GA106_SMS * GA106_TC_FP16_ACCUM_DENSE_FMAS_PER_SM_CLK * 2 * clk_ghz / 1000.0
    )
    return {
        "clock_mhz": round(float(clock_mhz), 2),
        "fp32_cuda_dense_tflops": round(fp32_dense, 4),
        "tc_fp16_fp32_accum_dense_tflops": round(tc_fp16_fp32_accum_dense, 4),
        "tc_fp16_fp16_accum_dense_tflops": round(tc_fp16_fp16_accum_dense, 4),
        "tc_fp16_fp32_accum_sparse24_tflops": round(tc_fp16_fp32_accum_dense * 2.0, 4),
        "tc_fp16_fp16_accum_sparse24_tflops": round(tc_fp16_fp16_accum_dense * 2.0, 4),
    }


class _LiveGpuBurstSampler:
    """Lightweight high-frequency nvidia-smi sampler during short 1-2s micro-benchmark bursts."""

    def __init__(self, interval_sec: float = 0.20) -> None:
        from scripts.telemetry_collector import default_gpu_query
        self._query_fn: Callable[[], Dict[str, Any]] = default_gpu_query
        self._interval = max(0.10, float(interval_sec))
        self._stop = threading.Event()
        self._samples: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_LiveGpuBurstSampler":
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="SiliconBurstSampler", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            snap = self._query_fn()
            if snap.get("core_clock_mhz", 0.0) > 0:
                self._samples.append(snap)
            if self._stop.wait(timeout=self._interval):
                break

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def summarize(self) -> Dict[str, Any]:
        if not self._samples:
            snap = self._query_fn()
            self._samples = [snap]
        active = self._samples[1:] if len(self._samples) > 2 else self._samples

        def _avg(k: str) -> float:
            vals = [float(s.get(k, 0.0)) for s in active]
            return round(sum(vals) / max(1, len(vals)), 2)

        def _peak(k: str) -> float:
            vals = [float(s.get(k, 0.0)) for s in active]
            return round(max(vals) if vals else 0.0, 2)

        pstates = [str(s.get("pstate", "N/A")) for s in active if s.get("pstate")]
        return {
            "samples": len(active),
            "pstate": pstates[-1] if pstates else "N/A",
            "avg_core_clock_mhz": _avg("core_clock_mhz"),
            "peak_core_clock_mhz": _peak("core_clock_mhz"),
            "avg_mem_clock_mhz": _avg("mem_clock_mhz"),
            "avg_power_w": _avg("power_w"),
            "peak_power_w": _peak("power_w"),
            "avg_gpu_util_pct": _avg("gpu_util_pct"),
            "avg_mem_util_pct": _avg("mem_util_pct"),
            "avg_enc_util_pct": _avg("enc_util_pct"),
            "peak_enc_util_pct": _peak("enc_util_pct"),
            "peak_temp_c": _peak("temp_c"),
        }


def _load_cublas_dll() -> Optional[ctypes.CDLL]:
    """Locate and load cuBLAS shared library bundled with PyTorch CUDA on Windows/Linux."""
    import torch
    candidates: List[str] = []
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if sys.platform == "win32":
        for pattern in ("cublas64_*.dll", "cublas64_12.dll", "cublas64_11.dll"):
            for p in sorted(torch_lib.glob(pattern), reverse=True):
                candidates.append(str(p))
        candidates.extend(["cublas64_12.dll", "cublas64_11.dll"])
    else:
        candidates.extend(["libcublas.so.12", "libcublas.so.11", "libcublas.so"])

    for cand in candidates:
        try:
            dll = ctypes.CDLL(cand)
            if hasattr(dll, "cublasGemmEx"):
                dll.cublasGemmEx.argtypes = [
                    ctypes.c_void_p,  # handle
                    ctypes.c_int,     # transa
                    ctypes.c_int,     # transb
                    ctypes.c_int,     # m
                    ctypes.c_int,     # n
                    ctypes.c_int,     # k
                    ctypes.c_void_p,  # alpha
                    ctypes.c_void_p,  # A
                    ctypes.c_int,     # Atype
                    ctypes.c_int,     # lda
                    ctypes.c_void_p,  # B
                    ctypes.c_int,     # Btype
                    ctypes.c_int,     # ldb
                    ctypes.c_void_p,  # beta
                    ctypes.c_void_p,  # C
                    ctypes.c_int,     # Ctype
                    ctypes.c_int,     # ldc
                    ctypes.c_int,     # computeType
                    ctypes.c_int,     # algo
                ]
                dll.cublasGemmEx.restype = ctypes.c_int
                return dll
        except Exception:
            continue
    return None


def _bench_cublas_gemm_ex_fp16(
    cublas_dll: ctypes.CDLL,
    a_fp16: Any,
    b_fp16: Any,
    c_fp16: Any,
    fp16_accum: bool,
    warmup_iters: int = 15,
    bench_iters: int = 240,
) -> Dict[str, Any]:
    """
    Directly invoke cuBLAS cublasGemmEx on dense (N, N) float16 matrices with explicit computeType:
    - fp16_accum=False -> CUBLAS_COMPUTE_32F (68) -> HMMA.16816.F32 (FP32 accumulate, 256 FMAs/SM/clk)
    - fp16_accum=True  -> CUBLAS_COMPUTE_16F (64) -> HMMA.16816.F16 (FP16 accumulate, 512 FMAs/SM/clk)
    """
    import torch
    import torch.nn.functional as F

    n = int(a_fp16.shape[0])
    handle = ctypes.c_void_p(torch.cuda.current_blas_handle())
    CUBLAS_OP_N = 0
    CUDA_R_16F = 2
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99

    if fp16_accum:
        compute_type = 64  # CUBLAS_COMPUTE_16F
        alpha_buf = ctypes.c_uint16(0x3C00)  # IEEE-754 float16 1.0
        beta_buf = ctypes.c_uint16(0x0000)   # IEEE-754 float16 0.0
    else:
        compute_type = 68  # CUBLAS_COMPUTE_32F
        alpha_buf = ctypes.c_float(1.0)
        beta_buf = ctypes.c_float(0.0)

    a_ptr = ctypes.c_void_p(a_fp16.data_ptr())
    b_ptr = ctypes.c_void_p(b_fp16.data_ptr())
    c_ptr = ctypes.c_void_p(c_fp16.data_ptr())
    alpha_ptr = ctypes.byref(alpha_buf)
    beta_ptr = ctypes.byref(beta_buf)

    def _step() -> None:
        status = cublas_dll.cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            n,
            n,
            alpha_ptr,
            b_ptr,
            CUDA_R_16F,
            n,
            a_ptr,
            CUDA_R_16F,
            n,
            beta_ptr,
            c_ptr,
            CUDA_R_16F,
            n,
            compute_type,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        if status != 0:
            raise RuntimeError(f"cublasGemmEx failed with status={status} (computeType={compute_type})")

    c_fp16.zero_()
    for _ in range(warmup_iters):
        _step()
    torch.cuda.synchronize()

    # Numerical sanity verification on top-left 128x128 tile against IEEE FP32 matmul
    ref_tile = a_fp16[:128, :].float() @ b_fp16[:, :128].float()
    got_tile = c_fp16[:128, :128].float()
    cos_sim = float(
        F.cosine_similarity(ref_tile.reshape(1, -1), got_tile.reshape(1, -1), dim=1).item()
    )

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    with _LiveGpuBurstSampler(interval_sec=0.15) as sampler:
        start_evt.record()
        for _ in range(bench_iters):
            _step()
        end_evt.record()
        torch.cuda.synchronize()
    live = sampler.summarize()

    total_ms = float(start_evt.elapsed_time(end_evt))
    avg_ms = total_ms / bench_iters
    flops_per_gemm = 2.0 * (n ** 3)
    sustained_tflops = (flops_per_gemm * bench_iters) / (total_ms * 1e-3) / 1e12
    live_clk = live["avg_core_clock_mhz"] or GA106_RATED_BOOST_MHZ
    theo_live = _theoretical_ga106_tflops(live_clk)
    theo_key = "tc_fp16_fp16_accum_dense_tflops" if fp16_accum else "tc_fp16_fp32_accum_dense_tflops"
    theo_ceil = theo_live[theo_key]

    return {
        "backend": f"cublasGemmEx(computeType={'CUBLAS_COMPUTE_16F' if fp16_accum else 'CUBLAS_COMPUTE_32F'})",
        "instruction_class": "HMMA.16816.F16" if fp16_accum else "HMMA.16816.F32",
        "matrix_shape": [n, n, n],
        "iters": bench_iters,
        "avg_ms": round(avg_ms, 4),
        "sustained_tflops": round(sustained_tflops, 3),
        "cosine_similarity_vs_fp32": round(cos_sim, 6),
        "numerical_check_pass": bool(cos_sim > 0.9999),
        "theoretical_dense_tflops_at_1777mhz": _theoretical_ga106_tflops(GA106_RATED_BOOST_MHZ)[theo_key],
        "theoretical_dense_tflops_at_live_clock": round(theo_ceil, 3),
        "silicon_efficiency_pct_at_live_clock": round(100.0 * sustained_tflops / max(0.001, theo_ceil), 2),
        "live_telemetry": live,
    }


DEFAULT_SILICON_PROBE_OUTPUT = "out/benchmark_r3_final/silicon_probe.json"


def run_silicon_probe(
    output_path: Optional[str | Path] = None,
    nvenc_frames: int = 120,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute the complete empirical silicon ceiling micro-probe on RTX 3060 12GB."""
    canonical_default = (_REPO_ROOT / DEFAULT_SILICON_PROBE_OUTPUT).resolve()
    explicit_output = output_path is not None
    out_file = Path(output_path).resolve() if explicit_output else canonical_default

    if dry_run:
        # Protect the canonical empirical silicon_probe.json from being overwritten by --dry-run.
        write_dry_run_file = explicit_output and (out_file != canonical_default)
        mock_payload: Dict[str, Any] = {
            "status": "DRY_RUN",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "theoretical_ga106_1777mhz": _theoretical_ga106_tflops(GA106_RATED_BOOST_MHZ),
            "output_path": str(out_file) if write_dry_run_file else None,
            "target_output_path": str(out_file),
            "wrote_file": write_dry_run_file,
        }
        if write_dry_run_file:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(mock_payload, indent=2), encoding="utf-8")
        return mock_payload

    out_file.parent.mkdir(parents=True, exist_ok=True)

    from src.config import (
        HEVC_COLOR_PRIMARIES,
        HEVC_COLOR_RANGE,
        HEVC_COLOR_TRC,
        HEVC_COLORSPACE,
        HEVC_ENCODER,
        HEVC_PIX_FMT,
        HEVC_PRESET,
        HEVC_PROFILE,
        HEVC_TAG,
        gpu_exclusivity_lock,
        is_gpu_lock_held_by_other,
    )
    held_pid = is_gpu_lock_held_by_other()
    if held_pid is not None:
        raise RuntimeError(f"Refusing to start silicon probe: .gpu_active.lock held by PID={held_pid}")

    probe_wall_start = time.perf_counter()

    with gpu_exclusivity_lock():
        import torch
        import torch.nn as nn
        from scripts.telemetry_collector import default_gpu_query
        from src.ffmpeg_ops import probe_video
        from src.trt_zero_copy_upscaler import (
            DoubleBufferedPipeUpscaler,
            GPUColorConverterP010LE,
            _nhwc_u8_to_nchw_fp16_norm,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; cannot run live silicon probe.")

        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        idle_before = default_gpu_query()

        orig_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        orig_tf32_cudnn = torch.backends.cudnn.allow_tf32
        orig_cudnn_bench = torch.backends.cudnn.benchmark
        orig_fp16_red = getattr(
            torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", True
        )

        results: Dict[str, Any] = {
            "status": "PASS",
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "device": {
                "name": torch.cuda.get_device_name(device),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": int(props.multi_processor_count),
                "total_vram_mib": round(props.total_memory / (1024 * 1024), 1),
                "l2_cache_bytes": int(getattr(props, "L2_cache_size", 2359296)),
                "l2_cache_mib": round(int(getattr(props, "L2_cache_size", 2359296)) / (1024 * 1024), 2),
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "idle_telemetry_before": idle_before,
                "architecture_whitepaper_note": (
                    "NVIDIA Ampere GA10x Whitepaper (Table 1 & Table 3): Consumer GeForce GA106 (28 SMs) "
                    "executes 128 FP32 FMAs/SM/clk (12.74 TFLOPS @ 1777 MHz), 256 dense FP16 FMAs/SM/clk "
                    "with FP32 accumulate (25.48 TFLOPS @ 1777 MHz; 50.95 TFLOPS w/ 2:4 sparsity), and "
                    "512 dense FP16 FMAs/SM/clk with FP16 accumulate (50.95 TFLOPS @ 1777 MHz; 101.90 TFLOPS "
                    "ONLY w/ 2:4 structured sparsity). Note also GA106-302-A1 reports 2.25 MiB (2,359,296 B) L2 cache."
                ),
                "theoretical_ga106_at_1777mhz": _theoretical_ga106_tflops(GA106_RATED_BOOST_MHZ),
                "theoretical_ga106_at_1950mhz": _theoretical_ga106_tflops(1950.0),
                "theoretical_gddr6_p0_gb_s": GA106_THEORETICAL_VRAM_GB_S,
                "theoretical_pcie_gen4_x16_gb_s": PCIE_GEN4_X16_UNIDIR_GB_S,
            },
        }

        try:
            # =================================================================
            # 1. SUSTAINED IEEE FP32 CUDA CORE GEMM (4096^3, allow_tf32=False)
            # =================================================================
            n_gemm = 4096
            flops_4096 = 2.0 * (n_gemm ** 3)  # 137,438,953,472 FLOPs
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")

            a_f32 = torch.randn((n_gemm, n_gemm), dtype=torch.float32, device=device)
            b_f32 = torch.randn((n_gemm, n_gemm), dtype=torch.float32, device=device)
            c_f32 = torch.empty((n_gemm, n_gemm), dtype=torch.float32, device=device)

            for _ in range(10):
                torch.matmul(a_f32, b_f32, out=c_f32)
            torch.cuda.synchronize(device)

            fp32_iters = 90
            evt_s = torch.cuda.Event(enable_timing=True)
            evt_e = torch.cuda.Event(enable_timing=True)
            with _LiveGpuBurstSampler(interval_sec=0.15) as sampler:
                evt_s.record()
                for _ in range(fp32_iters):
                    torch.matmul(a_f32, b_f32, out=c_f32)
                evt_e.record()
                torch.cuda.synchronize(device)
            fp32_live = sampler.summarize()
            fp32_total_ms = float(evt_s.elapsed_time(evt_e))
            fp32_avg_ms = fp32_total_ms / fp32_iters
            fp32_tflops = (flops_4096 * fp32_iters) / (fp32_total_ms * 1e-3) / 1e12
            fp32_clk = fp32_live["avg_core_clock_mhz"] or GA106_RATED_BOOST_MHZ
            fp32_theo_live = _theoretical_ga106_tflops(fp32_clk)["fp32_cuda_dense_tflops"]

            results["fp32_cuda_gemm"] = {
                "matrix_shape": [n_gemm, n_gemm, n_gemm],
                "allow_tf32": False,
                "iters": fp32_iters,
                "avg_ms": round(fp32_avg_ms, 4),
                "sustained_tflops": round(fp32_tflops, 3),
                "theoretical_dense_tflops_at_1777mhz": _theoretical_ga106_tflops(GA106_RATED_BOOST_MHZ)[
                    "fp32_cuda_dense_tflops"
                ],
                "theoretical_dense_tflops_at_live_clock": round(fp32_theo_live, 3),
                "silicon_efficiency_pct_at_live_clock": round(
                    100.0 * fp32_tflops / max(0.001, fp32_theo_live), 2
                ),
                "live_telemetry": fp32_live,
            }
            del a_f32, b_f32, c_f32
            torch.cuda.empty_cache()

            # =================================================================
            # 2. DENSE TENSOR CORE FP16 GEMM (FP32-ACCUM VS FP16-ACCUM) + SR CONV
            # =================================================================
            # Scale by 1/sqrt(4096)=1/64 so dot products stay well inside FP16 range without Inf
            a_f16 = torch.randn((n_gemm, n_gemm), dtype=torch.float16, device=device) * 0.125
            b_f16 = torch.randn((n_gemm, n_gemm), dtype=torch.float16, device=device) * 0.125
            c_f16 = torch.empty((n_gemm, n_gemm), dtype=torch.float16, device=device)

            def _bench_torch_fp16_matmul(
                a_mat: Any,
                b_mat: Any,
                c_mat: Any,
                reduced_accum: bool,
                iters: int,
            ) -> Dict[str, Any]:
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced_accum
                if hasattr(torch.backends.cuda.matmul, "allow_fp16_accum"):
                    torch.backends.cuda.matmul.allow_fp16_accum = reduced_accum
                for _ in range(15):
                    torch.matmul(a_mat, b_mat, out=c_mat)
                torch.cuda.synchronize(device)

                s_e = torch.cuda.Event(enable_timing=True)
                e_e = torch.cuda.Event(enable_timing=True)
                with _LiveGpuBurstSampler(interval_sec=0.15) as smp:
                    s_e.record()
                    for _ in range(iters):
                        torch.matmul(a_mat, b_mat, out=c_mat)
                    e_e.record()
                    torch.cuda.synchronize(device)
                lv = smp.summarize()
                tot_ms = float(s_e.elapsed_time(e_e))
                avg_m = tot_ms / iters
                tfl = (flops_4096 * iters) / (tot_ms * 1e-3) / 1e12
                clk = lv["avg_core_clock_mhz"] or GA106_RATED_BOOST_MHZ
                theo_lv = _theoretical_ga106_tflops(clk)
                # Note: PyTorch 2.6.0+cu124 CUDABlas.cpp dispatches CUBLAS_COMPUTE_32F for Half matmul
                # regardless of allow_fp16_reduced_precision_reduction (allow_fp16_accum arrived in 2.7).
                key = "tc_fp16_fp32_accum_dense_tflops"
                return {
                    "backend": "torch.matmul",
                    "allow_fp16_reduced_precision_reduction": reduced_accum,
                    "effective_cublas_compute_type": "CUBLAS_COMPUTE_32F (PyTorch 2.6 Half matmul uses FP32 accum)",
                    "matrix_shape": [n_gemm, n_gemm, n_gemm],
                    "iters": iters,
                    "avg_ms": round(avg_m, 4),
                    "sustained_tflops": round(tfl, 3),
                    "theoretical_fp32_accum_dense_tflops_at_1777mhz": _theoretical_ga106_tflops(
                        GA106_RATED_BOOST_MHZ
                    )[key],
                    "theoretical_fp32_accum_dense_tflops_at_live_clock": round(theo_lv[key], 3),
                    "silicon_efficiency_pct_vs_fp32_accum": round(
                        100.0 * tfl / max(0.001, theo_lv[key]), 2
                    ),
                    "live_telemetry": lv,
                }

            torch_fp32_acc = _bench_torch_fp16_matmul(
                a_f16, b_f16, c_f16, reduced_accum=False, iters=180
            )
            torch_fp16_acc = _bench_torch_fp16_matmul(
                a_f16, b_f16, c_f16, reduced_accum=True, iters=180
            )

            cublas_dll = _load_cublas_dll()
            cublas_fp32_acc = None
            cublas_fp16_acc = None
            if cublas_dll is not None:
                cublas_fp32_acc = _bench_cublas_gemm_ex_fp16(
                    cublas_dll, a_f16, b_f16, c_f16, fp16_accum=False, warmup_iters=15, bench_iters=180
                )
                cublas_fp16_acc = _bench_cublas_gemm_ex_fp16(
                    cublas_dll, a_f16, b_f16, c_f16, fp16_accum=True, warmup_iters=20, bench_iters=300
                )

            del a_f16, b_f16, c_f16
            torch.cuda.empty_cache()

            # 2b. Isolated SR Backbone Conv2d(64->64, 3x3, pad=1) FP16 on 8x1080p (exact animevideov3 hidden layer!)
            # Explains why square 4096^3 GEMM (AI = 1,365 FLOPs/B, N=4096) achieves ~42.6 TFLOPS while
            # 3x3 Conv2d(64->64) (N=64 small channel tile + 4.25 GB/batch off-chip GDDR6 activation R/W) achieves ~28 TFLOPS!
            torch.backends.cudnn.benchmark = True
            conv_layer = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1).to(
                device=device, dtype=torch.float16
            ).eval()
            conv_in = torch.randn((8, 64, 1080, 1920), dtype=torch.float16, device=device) * 0.1
            conv_flops_per_batch = 8.0 * 2.0 * 9.0 * 64.0 * 64.0 * 1080.0 * 1920.0  # 1,223,059,046,400 FLOPs
            conv_dram_bytes_per_batch = 2 * (8 * 64 * 1080 * 1920 * 2)  # 4,246,732,800 B (read + write)
            with torch.inference_mode():
                for _ in range(6):
                    _ = conv_layer(conv_in)
                torch.cuda.synchronize(device)
                conv_iters = 24
                s_c = torch.cuda.Event(enable_timing=True)
                e_c = torch.cuda.Event(enable_timing=True)
                with _LiveGpuBurstSampler(interval_sec=0.15) as c_smp:
                    s_c.record()
                    for _ in range(conv_iters):
                        _ = conv_layer(conv_in)
                    e_c.record()
                    torch.cuda.synchronize(device)
                conv_live = c_smp.summarize()
            conv_ms = float(s_c.elapsed_time(e_c)) / conv_iters
            conv_tflops = (conv_flops_per_batch / (conv_ms * 1e-3)) / 1e12
            conv_dram_gb_s = (conv_dram_bytes_per_batch / (conv_ms * 1e-3)) / 1e9
            del conv_layer, conv_in
            torch.cuda.empty_cache()

            authoritative_fp32_accum = cublas_fp32_acc or torch_fp32_acc
            authoritative_fp16_accum = cublas_fp16_acc or torch_fp16_acc

            results["tensor_core_fp16_gemm"] = {
                "fp32_accumulate_dense": authoritative_fp32_accum,
                "fp16_accumulate_dense": authoritative_fp16_accum,
                "fp16_vs_fp32_accum_speedup_ratio": round(
                    authoritative_fp16_accum["sustained_tflops"]
                    / max(0.001, authoritative_fp32_accum["sustained_tflops"]),
                    4,
                ),
                "torch_matmul_allow_fp16_reduced_false": torch_fp32_acc,
                "torch_matmul_allow_fp16_reduced_true": torch_fp16_acc,
                "cublas_gemm_ex_compute_32f": cublas_fp32_acc,
                "cublas_gemm_ex_compute_16f": cublas_fp16_acc,
                "sr_conv2d_64ch_3x3_8x1080p_fp16": {
                    "shape": [8, 64, 1080, 1920],
                    "flops_per_batch": int(conv_flops_per_batch),
                    "compulsory_dram_bytes_per_batch": conv_dram_bytes_per_batch,
                    "arithmetic_intensity_flops_per_byte": round(
                        conv_flops_per_batch / conv_dram_bytes_per_batch, 2
                    ),
                    "iters": conv_iters,
                    "avg_ms_per_layer_batch8": round(conv_ms, 4),
                    "sustained_tflops": round(conv_tflops, 3),
                    "compulsory_dram_bandwidth_gb_s": round(conv_dram_gb_s, 2),
                    "live_telemetry": conv_live,
                },
            }

            # =================================================================
            # 3. SUSTAINED GDDR6 VRAM BANDWIDTH (256 MiB COPY & STREAM TRIAD)
            # =================================================================
            num_elems = 67_108_864
            tensor_bytes = num_elems * 4  # 268,435,456 B (256 MiB)
            t_a = torch.ones(num_elems, dtype=torch.float32, device=device)
            t_b = torch.full((num_elems,), 2.0, dtype=torch.float32, device=device)
            t_c = torch.full((num_elems,), 0.5, dtype=torch.float32, device=device)
            scalar_s = 1.75

            copy_bytes = 2 * tensor_bytes
            for _ in range(15):
                t_a.copy_(t_b)
            torch.cuda.synchronize(device)

            bw_iters = 200
            evt_s = torch.cuda.Event(enable_timing=True)
            evt_e = torch.cuda.Event(enable_timing=True)
            with _LiveGpuBurstSampler(interval_sec=0.10) as sampler:
                evt_s.record()
                for _ in range(bw_iters):
                    t_a.copy_(t_b)
                evt_e.record()
                torch.cuda.synchronize(device)
            copy_live = sampler.summarize()
            copy_ms = float(evt_s.elapsed_time(evt_e)) / bw_iters
            copy_gb_s = (copy_bytes / (copy_ms * 1e-3)) / 1e9

            triad_bytes = 3 * tensor_bytes
            for _ in range(15):
                torch.add(t_b, t_c, alpha=scalar_s, out=t_a)
            torch.cuda.synchronize(device)

            evt_s = torch.cuda.Event(enable_timing=True)
            evt_e = torch.cuda.Event(enable_timing=True)
            with _LiveGpuBurstSampler(interval_sec=0.10) as sampler:
                evt_s.record()
                for _ in range(bw_iters):
                    torch.add(t_b, t_c, alpha=scalar_s, out=t_a)
                evt_e.record()
                torch.cuda.synchronize(device)
            triad_live = sampler.summarize()
            triad_ms = float(evt_s.elapsed_time(evt_e)) / bw_iters
            triad_gb_s = (triad_bytes / (triad_ms * 1e-3)) / 1e9

            sustained_dram_gb_s = max(copy_gb_s, triad_gb_s)
            theo_1777 = _theoretical_ga106_tflops(GA106_RATED_BOOST_MHZ)
            results["gddr6_vram_bandwidth"] = {
                "tensor_size_mib": 256.0,
                "tensor_bytes": tensor_bytes,
                "theoretical_peak_gb_s": GA106_THEORETICAL_VRAM_GB_S,
                "d2d_copy": {
                    "bytes_per_iter": copy_bytes,
                    "iters": bw_iters,
                    "avg_ms": round(copy_ms, 4),
                    "sustained_gb_s": round(copy_gb_s, 2),
                    "pct_of_theoretical_360gb_s": round(100.0 * copy_gb_s / GA106_THEORETICAL_VRAM_GB_S, 2),
                    "live_telemetry": copy_live,
                },
                "stream_triad_fused": {
                    "formula": "torch.add(b, c, alpha=s, out=a)",
                    "bytes_per_iter": triad_bytes,
                    "iters": bw_iters,
                    "avg_ms": round(triad_ms, 4),
                    "sustained_gb_s": round(triad_gb_s, 2),
                    "pct_of_theoretical_360gb_s": round(100.0 * triad_gb_s / GA106_THEORETICAL_VRAM_GB_S, 2),
                    "live_telemetry": triad_live,
                },
                "best_sustained_gb_s": round(sustained_dram_gb_s, 2),
                "theoretical_roofline_knees_at_1777mhz_vs_360gb_s": {
                    "fp32_cuda": round(theo_1777["fp32_cuda_dense_tflops"] * 1000.0 / 360.0, 2),
                    "tc_fp16_fp32_accum_dense": round(
                        theo_1777["tc_fp16_fp32_accum_dense_tflops"] * 1000.0 / 360.0, 2
                    ),
                    "tc_fp16_fp16_accum_dense": round(
                        theo_1777["tc_fp16_fp16_accum_dense_tflops"] * 1000.0 / 360.0, 2
                    ),
                },
                "empirical_roofline_knees_flops_per_byte": {
                    "vs_theoretical_360gb_s": {
                        "fp32_cuda": round(results["fp32_cuda_gemm"]["sustained_tflops"] * 1000.0 / 360.0, 2),
                        "tc_fp16_fp32_accum": round(
                            authoritative_fp32_accum["sustained_tflops"] * 1000.0 / 360.0, 2
                        ),
                        "tc_fp16_fp16_accum": round(
                            authoritative_fp16_accum["sustained_tflops"] * 1000.0 / 360.0, 2
                        ),
                    },
                    "vs_empirical_sustained_dram_gb_s": {
                        "fp32_cuda": round(
                            results["fp32_cuda_gemm"]["sustained_tflops"] * 1000.0 / sustained_dram_gb_s, 2
                        ),
                        "tc_fp16_fp32_accum": round(
                            authoritative_fp32_accum["sustained_tflops"] * 1000.0 / sustained_dram_gb_s, 2
                        ),
                        "tc_fp16_fp16_accum": round(
                            authoritative_fp16_accum["sustained_tflops"] * 1000.0 / sustained_dram_gb_s, 2
                        ),
                    },
                },
            }
            del t_a, t_b, t_c
            torch.cuda.empty_cache()

            # =================================================================
            # 4. SUSTAINED PCIe 4.0 x16 PINNED H2D (49.77 MB) & D2H (199.07 MB)
            # =================================================================
            batch_sz = 8
            in_h, in_w = 1080, 1920
            out_h, out_w = 2160, 3840

            h2d_bytes = batch_sz * in_h * in_w * 3  # 49,766,400 B (49.77 MB)
            d2h_bytes = batch_sz * (out_h * 3 // 2) * out_w * 2  # 199,065,600 B (199.07 MB)

            host_h2d_pin = torch.randint(
                0, 256, (batch_sz, in_h, in_w, 3), dtype=torch.uint8, pin_memory=True
            )
            gpu_h2d_dst = torch.empty((batch_sz, in_h, in_w, 3), dtype=torch.uint8, device=device)

            dma_stream = torch.cuda.Stream(device=device)
            pcie_iters = 120
            with torch.cuda.stream(dma_stream):
                for _ in range(15):
                    gpu_h2d_dst.copy_(host_h2d_pin, non_blocking=True)
            dma_stream.synchronize()

            evt_s = torch.cuda.Event(enable_timing=True)
            evt_e = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(dma_stream):
                evt_s.record(dma_stream)
                for _ in range(pcie_iters):
                    gpu_h2d_dst.copy_(host_h2d_pin, non_blocking=True)
                evt_e.record(dma_stream)
            dma_stream.synchronize()
            h2d_avg_ms = float(evt_s.elapsed_time(evt_e)) / pcie_iters
            h2d_gb_s = (h2d_bytes / (h2d_avg_ms * 1e-3)) / 1e9

            gpu_d2h_src = torch.randint(
                4096, 30000, (batch_sz, out_h * 3 // 2, out_w), dtype=torch.int16, device=device
            )
            host_d2h_pin = torch.empty(
                (batch_sz, out_h * 3 // 2, out_w), dtype=torch.int16, pin_memory=True
            )
            with torch.cuda.stream(dma_stream):
                for _ in range(15):
                    host_d2h_pin.copy_(gpu_d2h_src, non_blocking=True)
            dma_stream.synchronize()

            evt_s = torch.cuda.Event(enable_timing=True)
            evt_e = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(dma_stream):
                evt_s.record(dma_stream)
                for _ in range(pcie_iters):
                    host_d2h_pin.copy_(gpu_d2h_src, non_blocking=True)
                evt_e.record(dma_stream)
            dma_stream.synchronize()
            d2h_avg_ms = float(evt_s.elapsed_time(evt_e)) / pcie_iters
            d2h_gb_s = (d2h_bytes / (d2h_avg_ms * 1e-3)) / 1e9

            results["pcie_pinned_dma"] = {
                "theoretical_pcie_gen4_x16_gb_s": PCIE_GEN4_X16_UNIDIR_GB_S,
                "theoretical_ddr4_2133_dual_channel_gb_s": 34.13,
                "h2d_8x_1080p_rgb24": {
                    "shape": [batch_sz, in_h, in_w, 3],
                    "dtype": "uint8",
                    "bytes_per_batch": h2d_bytes,
                    "mb_decimal": round(h2d_bytes / 1e6, 2),
                    "iters": pcie_iters,
                    "avg_ms_per_batch": round(h2d_avg_ms, 4),
                    "avg_ms_per_frame": round(h2d_avg_ms / batch_sz, 4),
                    "sustained_gb_s": round(h2d_gb_s, 2),
                    "pct_of_pcie_gen4_x16": round(100.0 * h2d_gb_s / PCIE_GEN4_X16_UNIDIR_GB_S, 2),
                },
                "d2h_8x_4k_p010le": {
                    "shape": [batch_sz, out_h * 3 // 2, out_w],
                    "dtype": "int16",
                    "bytes_per_batch": d2h_bytes,
                    "mb_decimal": round(d2h_bytes / 1e6, 2),
                    "iters": pcie_iters,
                    "avg_ms_per_batch": round(d2h_avg_ms, 4),
                    "avg_ms_per_frame": round(d2h_avg_ms / batch_sz, 4),
                    "sustained_gb_s": round(d2h_gb_s, 2),
                    "pct_of_pcie_gen4_x16": round(100.0 * d2h_gb_s / PCIE_GEN4_X16_UNIDIR_GB_S, 2),
                },
            }

            # =================================================================
            # 5. ISOLATED @triton.jit INPUT NORM & OUTPUT BT.709 P010LE KERNELS
            # =================================================================
            gpu_nchw_fp16 = torch.empty((batch_sz, 3, in_h, in_w), dtype=torch.float16, device=device)
            in_norm_bytes = h2d_bytes + (batch_sz * 3 * in_h * in_w * 2)  # 149,299,200 B (149.30 MB)

            for _ in range(25):
                _nhwc_u8_to_nchw_fp16_norm(gpu_h2d_dst, out_nchw_fp16=gpu_nchw_fp16, use_triton=True)
            torch.cuda.synchronize(device)

            kern_iters = 300
            evt_s = torch.cuda.Event(enable_timing=True)
            evt_e = torch.cuda.Event(enable_timing=True)
            evt_s.record()
            for _ in range(kern_iters):
                _nhwc_u8_to_nchw_fp16_norm(gpu_h2d_dst, out_nchw_fp16=gpu_nchw_fp16, use_triton=True)
            evt_e.record()
            torch.cuda.synchronize(device)
            in_triton_ms = float(evt_s.elapsed_time(evt_e)) / kern_iters
            in_triton_gb_s = (in_norm_bytes / (in_triton_ms * 1e-3)) / 1e9

            legacy_traffic_bytes = h2d_bytes + 3 * (batch_sz * 3 * in_h * in_w * 2)  # 348.36 MB
            for _ in range(25):
                gpu_nchw_fp16.copy_(gpu_h2d_dst.permute(0, 3, 1, 2))
                gpu_nchw_fp16.mul_(1.0 / 255.0)
            torch.cuda.synchronize(device)

            evt_s = torch.cuda.Event(enable_timing=True)
            evt_e = torch.cuda.Event(enable_timing=True)
            evt_s.record()
            for _ in range(kern_iters):
                gpu_nchw_fp16.copy_(gpu_h2d_dst.permute(0, 3, 1, 2))
                gpu_nchw_fp16.mul_(1.0 / 255.0)
            evt_e.record()
            torch.cuda.synchronize(device)
            in_legacy_ms = float(evt_s.elapsed_time(evt_e)) / kern_iters

            del gpu_nchw_fp16, gpu_h2d_dst, host_h2d_pin, gpu_d2h_src
            torch.cuda.empty_cache()

            gpu_4k_rgb = torch.rand((batch_sz, 3, out_h, out_w), dtype=torch.float16, device=device)
            color_conv = GPUColorConverterP010LE(device=device)
            out_p010_bytes = (batch_sz * 3 * out_h * out_w * 2) + d2h_bytes  # 597,196,800 B (597.20 MB)

            for i in range(25):
                color_conv.rgb_to_p010le(gpu_4k_rgb, slot=i % 2)
            torch.cuda.synchronize(device)

            p010_iters = 200
            evt_s = torch.cuda.Event(enable_timing=True)
            evt_e = torch.cuda.Event(enable_timing=True)
            evt_s.record()
            for i in range(p010_iters):
                color_conv.rgb_to_p010le(gpu_4k_rgb, slot=i % 2)
            evt_e.record()
            torch.cuda.synchronize(device)
            out_triton_ms = float(evt_s.elapsed_time(evt_e)) / p010_iters
            out_triton_gb_s = (out_p010_bytes / (out_triton_ms * 1e-3)) / 1e9

            yy = torch.linspace(0.05, 0.95, out_h, device=device, dtype=torch.float16).view(1, 1, out_h, 1)
            xx = torch.linspace(0.05, 0.95, out_w, device=device, dtype=torch.float16).view(1, 1, 1, out_w)
            synth_4k = torch.empty((2, 3, out_h, out_w), dtype=torch.float16, device=device)
            synth_4k[0, 0] = yy * 0.8 + 0.1 * ((xx * 32.0).floor() % 2)
            synth_4k[0, 1] = xx * 0.7 + 0.15 * ((yy * 24.0).floor() % 2)
            synth_4k[0, 2] = (1.0 - yy * 0.5) * 0.85
            synth_4k[1] = (synth_4k[0] * 0.96 + 0.03).clamp(0.0, 1.0)
            synth_p010_gpu = color_conv.rgb_to_p010le(synth_4k, slot=0)
            host_d2h_pin[:2].copy_(synth_p010_gpu, non_blocking=False)
            torch.cuda.synchronize(device)

            del gpu_4k_rgb, synth_4k, synth_p010_gpu, color_conv
            torch.cuda.empty_cache()

            results["triton_kernels"] = {
                "input_norm_nhwc_u8_to_nchw_fp16": {
                    "batch_shape_in": [batch_sz, in_h, in_w, 3],
                    "batch_shape_out": [batch_sz, 3, in_h, in_w],
                    "vram_traffic_bytes": in_norm_bytes,
                    "vram_traffic_mb_decimal": round(in_norm_bytes / 1e6, 2),
                    "iters": kern_iters,
                    "avg_ms_per_batch": round(in_triton_ms, 4),
                    "avg_ms_per_frame": round(in_triton_ms / batch_sz, 4),
                    "effective_gb_s": round(in_triton_gb_s, 2),
                    "pct_of_theoretical_360gb_s": round(
                        100.0 * in_triton_gb_s / GA106_THEORETICAL_VRAM_GB_S, 2
                    ),
                    "legacy_pytorch_2kernel_avg_ms_per_batch": round(in_legacy_ms, 4),
                    "legacy_pytorch_traffic_mb_decimal": round(legacy_traffic_bytes / 1e6, 2),
                    "speedup_vs_legacy_2kernel": round(in_legacy_ms / max(1e-6, in_triton_ms), 2),
                },
                "output_bt709_rgb_fp16_to_p010le": {
                    "batch_shape_in": [batch_sz, 3, out_h, out_w],
                    "batch_shape_out": [batch_sz, out_h * 3 // 2, out_w],
                    "vram_traffic_bytes": out_p010_bytes,
                    "vram_traffic_mb_decimal": round(out_p010_bytes / 1e6, 2),
                    "iters": p010_iters,
                    "avg_ms_per_batch": round(out_triton_ms, 4),
                    "avg_ms_per_frame": round(out_triton_ms / batch_sz, 4),
                    "effective_gb_s": round(out_triton_gb_s, 2),
                    "pct_of_theoretical_360gb_s": round(
                        100.0 * out_triton_gb_s / GA106_THEORETICAL_VRAM_GB_S, 2
                    ),
                },
            }

            # =================================================================
            # 6. ISOLATED 7th-GEN NVENC 4K HEVC MAIN10 STDIN PIPE CEILING FPS
            # =================================================================
            frame_np_0 = host_d2h_pin[0].numpy()
            frame_np_1 = host_d2h_pin[1].numpy()
            frame_bytes = int(frame_np_0.nbytes)

            def _run_nvenc_burst(num_f: int, tmp_tag: str, multipass: str = "fullres") -> Dict[str, Any]:
                tmp_mp4 = out_file.parent / f".nvenc_probe_{tmp_tag}.mp4"
                if tmp_mp4.exists():
                    tmp_mp4.unlink()
                t_spawn_s = time.perf_counter()
                if multipass == "fullres":
                    proc = DoubleBufferedPipeUpscaler.spawn_nvenc_pipe_writer(
                        output_mp4_path=str(tmp_mp4),
                        fps_str="24000/1001",
                        width=out_w,
                        height=out_h,
                    )
                else:
                    cmd = [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "rawvideo", "-pix_fmt", "p010le", "-s", f"{out_w}x{out_h}",
                        "-color_range", HEVC_COLOR_RANGE, "-colorspace", HEVC_COLORSPACE,
                        "-color_primaries", HEVC_COLOR_PRIMARIES, "-color_trc", HEVC_COLOR_TRC,
                        "-chroma_sample_location", "center", "-r", "24000/1001", "-i", "pipe:0",
                        "-c:v", HEVC_ENCODER, "-profile:v", HEVC_PROFILE, "-pix_fmt", HEVC_PIX_FMT,
                        "-preset", HEVC_PRESET, "-tune", "hq", "-multipass", multipass,
                        "-rc", "vbr", "-cq", "18", "-b:v", "0", "-maxrate", "45M", "-bufsize", "90M",
                        "-rc-lookahead", "32", "-spatial-aq", "1", "-aq-strength", "6",
                        "-temporal-aq", "1", "-bf", "4", "-b_ref_mode", "middle",
                        "-tag:v", HEVC_TAG, str(tmp_mp4),
                    ]
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                try:
                    t_feed_s = time.perf_counter()
                    with _LiveGpuBurstSampler(interval_sec=0.15) as nv_smp:
                        for idx in range(num_f):
                            DoubleBufferedPipeUpscaler._write_frame_to_pipe(
                                proc, frame_np_0 if (idx % 2 == 0) else frame_np_1, repeats=1
                            )
                        t_feed_done = time.perf_counter()
                        if proc.stdin:
                            proc.stdin.close()
                        rc = proc.wait(timeout=30.0)
                    nv_live = nv_smp.summarize()
                    t_done = time.perf_counter()
                    if rc != 0:
                        stderr_txt = ""
                        if proc.stderr:
                            stderr_txt = proc.stderr.read().decode("utf-8", errors="replace")
                        raise RuntimeError(f"NVENC probe failed rc={rc}: {stderr_txt}")
                    meta = probe_video(str(tmp_mp4))
                    encoded_frames = int(meta.get("nb_frames") or num_f)
                finally:
                    if proc.stderr:
                        try:
                            proc.stderr.close()
                        except Exception:
                            pass
                    if tmp_mp4.exists():
                        try:
                            tmp_mp4.unlink()
                        except OSError:
                            pass

                total_wall_s = t_done - t_spawn_s
                feed_and_flush_s = t_done - t_feed_s
                pure_feed_s = t_feed_done - t_feed_s
                return {
                    "multipass": multipass,
                    "frames_requested": num_f,
                    "frames_verified": encoded_frames,
                    "total_wall_s": round(total_wall_s, 4),
                    "feed_and_flush_s": round(feed_and_flush_s, 4),
                    "pure_stdin_feed_s": round(pure_feed_s, 4),
                    "e2e_including_spawn_fps": round(encoded_frames / max(1e-6, total_wall_s), 2),
                    "asic_feed_to_flush_fps": round(encoded_frames / max(1e-6, feed_and_flush_s), 2),
                    "ms_per_frame_feed_to_flush": round((feed_and_flush_s * 1000.0) / max(1, encoded_frames), 3),
                    "pipe_throughput_gb_s": round(
                        (encoded_frames * frame_bytes) / max(1e-6, feed_and_flush_s) / 1e9, 2
                    ),
                    "live_telemetry": nv_live,
                }

            _run_nvenc_burst(16, "warmup", multipass="fullres")
            nvenc_120_fullres = _run_nvenc_burst(int(nvenc_frames), "120f_fullres", multipass="fullres")
            nvenc_240_fullres = _run_nvenc_burst(240, "240f_fullres", multipass="fullres")
            nvenc_120_singlepass = _run_nvenc_burst(int(nvenc_frames), "120f_disabled", multipass="disabled")

            delta_f = nvenc_240_fullres["frames_verified"] - nvenc_120_fullres["frames_verified"]
            delta_s = max(1e-4, nvenc_240_fullres["feed_and_flush_s"] - nvenc_120_fullres["feed_and_flush_s"])
            marginal_steady_asic_fps = round(delta_f / delta_s, 2)

            results["nvenc_4k_hevc_main10_ceiling"] = {
                "encoder": "hevc_nvenc",
                "profile": "main10",
                "pix_fmt": "p010le",
                "resolution": "3840x2160",
                "preset": "p6",
                "cq": 18,
                "production_multipass": "fullres",
                "rc_lookahead": 32,
                "spatial_aq": 1,
                "aq_strength": 6,
                "temporal_aq": 1,
                "frame_bytes": frame_bytes,
                "probe_120_frames": nvenc_120_fullres,
                "probe_240_frames": nvenc_240_fullres,
                "marginal_steady_state_asic_fps": marginal_steady_asic_fps,
                "marginal_steady_state_ms_per_frame": round(1000.0 / max(1e-6, marginal_steady_asic_fps), 3),
                "single_pass_reference_120_frames_multipass_disabled": nvenc_120_singlepass,
            }

            del host_d2h_pin
            torch.cuda.empty_cache()

        finally:
            torch.backends.cuda.matmul.allow_tf32 = orig_tf32_matmul
            torch.backends.cudnn.allow_tf32 = orig_tf32_cudnn
            torch.backends.cudnn.benchmark = orig_cudnn_bench
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig_fp16_red

    results["total_probe_wall_seconds"] = round(time.perf_counter() - probe_wall_start, 2)
    results["output_path"] = str(out_file)

    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Empirical silicon ceiling micro-probe for RTX 3060 12GB (GA106)."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            f"Output JSON path (default: {DEFAULT_SILICON_PROBE_OUTPUT}; "
            "omitted --output in --dry-run prints JSON to stdout without writing disk)"
        ),
    )
    parser.add_argument(
        "--nvenc-frames",
        type=int,
        default=120,
        help="Frame count for canonical NVENC 4K Main10 stdin pipe probe (default: 120)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate CLI and print dry-run JSON without touching GPU or overwriting canonical output",
    )
    args = parser.parse_args(argv)

    res = run_silicon_probe(
        output_path=args.output,
        nvenc_frames=args.nvenc_frames,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    fp32 = res["fp32_cuda_gemm"]
    tc = res["tensor_core_fp16_gemm"]
    sr_c = tc["sr_conv2d_64ch_3x3_8x1080p_fp16"]
    bw = res["gddr6_vram_bandwidth"]
    pcie = res["pcie_pinned_dma"]
    tri = res["triton_kernels"]
    nv = res["nvenc_4k_hevc_main10_ceiling"]

    print("\n" + "=" * 76)
    print("🔬 EMPIRICAL SILICON CEILING PROBE — RTX 3060 12GB (GA106-302-A1)")
    print("=" * 76)
    print(
        f"• FP32 CUDA GEMM (4096^3):      {fp32['sustained_tflops']} TFLOPS "
        f"@ {fp32['live_telemetry']['avg_core_clock_mhz']} MHz / {fp32['live_telemetry']['avg_power_w']} W "
        f"({fp32['silicon_efficiency_pct_at_live_clock']}% of {fp32['theoretical_dense_tflops_at_live_clock']} TFLOPS)"
    )
    print(
        f"• TC FP16 GEMM (FP32-accum):    {tc['fp32_accumulate_dense']['sustained_tflops']} TFLOPS "
        f"@ {tc['fp32_accumulate_dense']['live_telemetry']['avg_core_clock_mhz']} MHz "
        f"({tc['fp32_accumulate_dense']['silicon_efficiency_pct_at_live_clock']}% of "
        f"{tc['fp32_accumulate_dense']['theoretical_dense_tflops_at_live_clock']} TFLOPS)"
    )
    print(
        f"• TC FP16 GEMM (FP16-accum):    {tc['fp16_accumulate_dense']['sustained_tflops']} TFLOPS "
        f"@ {tc['fp16_accumulate_dense']['live_telemetry']['avg_core_clock_mhz']} MHz / "
        f"{tc['fp16_accumulate_dense']['live_telemetry']['avg_power_w']} W "
        f"({tc['fp16_accumulate_dense']['silicon_efficiency_pct_at_live_clock']}% of "
        f"{tc['fp16_accumulate_dense']['theoretical_dense_tflops_at_live_clock']} TFLOPS, cos={tc['fp16_accumulate_dense']['cosine_similarity_vs_fp32']})"
    )
    print(
        f"• SR Conv2d(64->64,3x3) 8x1080p:{sr_c['sustained_tflops']} TFLOPS "
        f"({sr_c['avg_ms_per_layer_batch8']} ms/layer, DRAM {sr_c['compulsory_dram_bandwidth_gb_s']} GB/s)"
    )
    print(
        f"• GDDR6 VRAM Bandwidth:         Copy {bw['d2d_copy']['sustained_gb_s']} GB/s | "
        f"Triad {bw['stream_triad_fused']['sustained_gb_s']} GB/s (vs 360.0 GB/s)"
    )
    print(
        f"• PCIe 4.0 x16 Pinned DMA:      H2D (49.77MB): {pcie['h2d_8x_1080p_rgb24']['avg_ms_per_batch']} ms "
        f"({pcie['h2d_8x_1080p_rgb24']['sustained_gb_s']} GB/s) | "
        f"D2H (199.07MB): {pcie['d2h_8x_4k_p010le']['avg_ms_per_batch']} ms "
        f"({pcie['d2h_8x_4k_p010le']['sustained_gb_s']} GB/s)"
    )
    in_k = tri["input_norm_nhwc_u8_to_nchw_fp16"]
    out_k = tri["output_bt709_rgb_fp16_to_p010le"]
    print(
        f"• Triton Input Norm (8x1080p):  {in_k['avg_ms_per_batch']} ms/batch "
        f"({in_k['effective_gb_s']} GB/s, {in_k['speedup_vs_legacy_2kernel']}x vs legacy {in_k['legacy_pytorch_2kernel_avg_ms_per_batch']} ms)"
    )
    print(
        f"• Triton Output P010LE (8x4K):  {out_k['avg_ms_per_batch']} ms/batch "
        f"({out_k['effective_gb_s']} GB/s)"
    )
    print(
        f"• 7th-Gen NVENC 4K Main10 p6:   fullres 120f={nv['probe_120_frames']['asic_feed_to_flush_fps']} fps, "
        f"240f={nv['probe_240_frames']['asic_feed_to_flush_fps']} fps, "
        f"steady={nv['marginal_steady_state_asic_fps']} fps | "
        f"single-pass 120f={nv['single_pass_reference_120_frames_multipass_disabled']['asic_feed_to_flush_fps']} fps"
    )
    print(f"• Saved Probe JSON:             {res['output_path']} ({res['total_probe_wall_seconds']}s)")
    print("=" * 76 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
