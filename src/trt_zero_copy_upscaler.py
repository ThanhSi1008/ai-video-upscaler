"""Zero-disk, zero-copy GPU inference pipeline using Triton P010LE conversion and 3-stream CUDA overlap."""

import collections
import os
import queue
import subprocess
import threading
import time
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False

_HAS_TRITON = HAS_TRITON

from src.config import (
    HEVC_ENCODER,
    HEVC_PROFILE,
    HEVC_PIX_FMT,
    HEVC_COLOR_RANGE,
    HEVC_COLORSPACE,
    HEVC_COLOR_PRIMARIES,
    HEVC_COLOR_TRC,
    HEVC_TAG,
    get_hevc_rate_args,
)


if _HAS_TRITON:
    @triton.jit
    def _nhwc_u8_to_nchw_fp16_norm_kernel(
        in_ptr,
        out_ptr,
        HW,
        stride_in_b,
        stride_out_b,
        stride_out_c,
        BLOCK_P: tl.constexpr,
    ):
        pid_p = tl.program_id(axis=0)
        pid_b = tl.program_id(axis=1).to(tl.int64)

        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        mask = offs_p < HW

        in_pix = in_ptr + pid_b * stride_in_b + offs_p * 3
        r_u8 = tl.load(in_pix + 0, mask=mask, other=0).to(tl.float32)
        g_u8 = tl.load(in_pix + 1, mask=mask, other=0).to(tl.float32)
        b_u8 = tl.load(in_pix + 2, mask=mask, other=0).to(tl.float32)

        inv_255 = 1.0 / 255.0
        r_f16 = (r_u8 * inv_255).to(tl.float16)
        g_f16 = (g_u8 * inv_255).to(tl.float16)
        b_f16 = (b_u8 * inv_255).to(tl.float16)

        out_base = out_ptr + pid_b * stride_out_b
        tl.store(out_base + 0 * stride_out_c + offs_p, r_f16, mask=mask)
        tl.store(out_base + 1 * stride_out_c + offs_p, g_f16, mask=mask)
        tl.store(out_base + 2 * stride_out_c + offs_p, b_f16, mask=mask)

    @triton.jit
    def _rgb_to_p010le_triton_kernel(
        rgb_ptr,
        out_ptr,
        stride_rgb_b, stride_rgb_c, stride_rgb_h, stride_rgb_w,
        stride_out_b, stride_out_h, stride_out_w,
        H, W,
        BLOCK_X: tl.constexpr,
        BLOCK_Y: tl.constexpr,
    ):
        pid_x = tl.program_id(axis=0)
        pid_y = tl.program_id(axis=1)
        pid_b = tl.program_id(axis=2)

        H_uv = H // 2
        W_uv = W // 2

        offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
        offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)

        mask = (offs_x[None, :] < W_uv) & (offs_y[:, None] < H_uv)

        rgb_b = rgb_ptr + pid_b * stride_rgb_b
        out_b = out_ptr + pid_b * stride_out_b

        y0 = offs_y[:, None] * 2
        y1 = y0 + 1
        x0 = offs_x[None, :] * 2
        x1 = x0 + 1

        # Load 4 pixels of R, G, B
        r00 = tl.load(rgb_b + 0 * stride_rgb_c + y0 * stride_rgb_h + x0 * stride_rgb_w, mask=mask).to(tl.float32)
        r01 = tl.load(rgb_b + 0 * stride_rgb_c + y0 * stride_rgb_h + x1 * stride_rgb_w, mask=mask).to(tl.float32)
        r10 = tl.load(rgb_b + 0 * stride_rgb_c + y1 * stride_rgb_h + x0 * stride_rgb_w, mask=mask).to(tl.float32)
        r11 = tl.load(rgb_b + 0 * stride_rgb_c + y1 * stride_rgb_h + x1 * stride_rgb_w, mask=mask).to(tl.float32)

        g00 = tl.load(rgb_b + 1 * stride_rgb_c + y0 * stride_rgb_h + x0 * stride_rgb_w, mask=mask).to(tl.float32)
        g01 = tl.load(rgb_b + 1 * stride_rgb_c + y0 * stride_rgb_h + x1 * stride_rgb_w, mask=mask).to(tl.float32)
        g10 = tl.load(rgb_b + 1 * stride_rgb_c + y1 * stride_rgb_h + x0 * stride_rgb_w, mask=mask).to(tl.float32)
        g11 = tl.load(rgb_b + 1 * stride_rgb_c + y1 * stride_rgb_h + x1 * stride_rgb_w, mask=mask).to(tl.float32)

        b00 = tl.load(rgb_b + 2 * stride_rgb_c + y0 * stride_rgb_h + x0 * stride_rgb_w, mask=mask).to(tl.float32)
        b01 = tl.load(rgb_b + 2 * stride_rgb_c + y0 * stride_rgb_h + x1 * stride_rgb_w, mask=mask).to(tl.float32)
        b10 = tl.load(rgb_b + 2 * stride_rgb_c + y1 * stride_rgb_h + x0 * stride_rgb_w, mask=mask).to(tl.float32)
        b11 = tl.load(rgb_b + 2 * stride_rgb_c + y1 * stride_rgb_h + x1 * stride_rgb_w, mask=mask).to(tl.float32)

        # Khử NaN (NaN == NaN là False theo IEEE-754) về 0.0 trước khi clamp [0.0, 1.0]
        r00 = tl.where(r00 == r00, r00, 0.0)
        r01 = tl.where(r01 == r01, r01, 0.0)
        r10 = tl.where(r10 == r10, r10, 0.0)
        r11 = tl.where(r11 == r11, r11, 0.0)
        g00 = tl.where(g00 == g00, g00, 0.0)
        g01 = tl.where(g01 == g01, g01, 0.0)
        g10 = tl.where(g10 == g10, g10, 0.0)
        g11 = tl.where(g11 == g11, g11, 0.0)
        b00 = tl.where(b00 == b00, b00, 0.0)
        b01 = tl.where(b01 == b01, b01, 0.0)
        b10 = tl.where(b10 == b10, b10, 0.0)
        b11 = tl.where(b11 == b11, b11, 0.0)

        # Clamp out-of-gamut SR model RGB overshoots (< 0.0 or > 1.0) in registers before signed BT.709 matrix
        # so Cb/Cr channels never cross-bleed on bright Sakuga highlights (matches PNG mode clamp(0.0, 1.0))
        r00 = tl.minimum(tl.maximum(r00, 0.0), 1.0)
        r01 = tl.minimum(tl.maximum(r01, 0.0), 1.0)
        r10 = tl.minimum(tl.maximum(r10, 0.0), 1.0)
        r11 = tl.minimum(tl.maximum(r11, 0.0), 1.0)
        g00 = tl.minimum(tl.maximum(g00, 0.0), 1.0)
        g01 = tl.minimum(tl.maximum(g01, 0.0), 1.0)
        g10 = tl.minimum(tl.maximum(g10, 0.0), 1.0)
        g11 = tl.minimum(tl.maximum(g11, 0.0), 1.0)
        b00 = tl.minimum(tl.maximum(b00, 0.0), 1.0)
        b01 = tl.minimum(tl.maximum(b01, 0.0), 1.0)
        b10 = tl.minimum(tl.maximum(b10, 0.0), 1.0)
        b11 = tl.minimum(tl.maximum(b11, 0.0), 1.0)

        # 1. Compute Luma Y for 4 pixels (ITU-R BT.709 Limited Range [64..940])
        y_val00 = 64.0 + 186.2376 * r00 + 626.5152 * g00 + 63.2472 * b00
        y_val01 = 64.0 + 186.2376 * r01 + 626.5152 * g01 + 63.2472 * b01
        y_val10 = 64.0 + 186.2376 * r10 + 626.5152 * g10 + 63.2472 * b10
        y_val11 = 64.0 + 186.2376 * r11 + 626.5152 * g11 + 63.2472 * b11

        y_r00 = tl.extra.cuda.libdevice.rint(y_val00)
        y_r01 = tl.extra.cuda.libdevice.rint(y_val01)
        y_r10 = tl.extra.cuda.libdevice.rint(y_val10)
        y_r11 = tl.extra.cuda.libdevice.rint(y_val11)

        y_p010_00 = (tl.minimum(tl.maximum(y_r00, 64.0), 940.0).to(tl.int32) << 6).to(tl.int16)
        y_p010_01 = (tl.minimum(tl.maximum(y_r01, 64.0), 940.0).to(tl.int32) << 6).to(tl.int16)
        y_p010_10 = (tl.minimum(tl.maximum(y_r10, 64.0), 940.0).to(tl.int32) << 6).to(tl.int16)
        y_p010_11 = (tl.minimum(tl.maximum(y_r11, 64.0), 940.0).to(tl.int32) << 6).to(tl.int16)

        tl.store(out_b + y0 * stride_out_h + x0 * stride_out_w, y_p010_00, mask=mask)
        tl.store(out_b + y0 * stride_out_h + x1 * stride_out_w, y_p010_01, mask=mask)
        tl.store(out_b + y1 * stride_out_h + x0 * stride_out_w, y_p010_10, mask=mask)
        tl.store(out_b + y1 * stride_out_h + x1 * stride_out_w, y_p010_11, mask=mask)

        # 2. 2x2 Chroma Subsampling & BT.709 Chroma calculation [64..960]
        rs = 0.25 * (r00 + r01 + r10 + r11)
        gs = 0.25 * (g00 + g01 + g10 + g11)
        bs = 0.25 * (b00 + b01 + b10 + b11)

        u_val = 512.0 - 102.656512 * rs - 345.343488 * gs + 448.0 * bs
        v_val = 512.0 + 448.0 * rs - 406.921088 * gs - 41.078912 * bs

        u_r = tl.extra.cuda.libdevice.rint(u_val)
        v_r = tl.extra.cuda.libdevice.rint(v_val)

        u_p010 = (tl.minimum(tl.maximum(u_r, 64.0), 960.0).to(tl.int32) << 6).to(tl.int16)
        v_p010 = (tl.minimum(tl.maximum(v_r, 64.0), 960.0).to(tl.int32) << 6).to(tl.int16)

        # Interleaved UV plane at row H + offs_y
        uv_row = H + offs_y[:, None]
        tl.store(out_b + uv_row * stride_out_h + x0 * stride_out_w, u_p010, mask=mask)
        tl.store(out_b + uv_row * stride_out_h + x1 * stride_out_w, v_p010, mask=mask)
else:
    _nhwc_u8_to_nchw_fp16_norm_kernel = None
    _rgb_to_p010le_triton_kernel = None


def _nhwc_u8_to_nchw_fp16_norm(
    nhwc_u8: torch.Tensor,
    out_nchw_fp16: Optional[torch.Tensor] = None,
    use_triton: Optional[bool] = None,
) -> torch.Tensor:
    """Convert (B, H, W, 3) uint8 to (B, 3, H, W) fp16 [0, 1] directly into out_nchw_fp16."""
    if nhwc_u8.ndim != 4 or nhwc_u8.shape[-1] != 3:
        raise ValueError(f"Kỳ vọng tensor NHWC (B, H, W, 3), nhận được shape={tuple(nhwc_u8.shape)}")

    B, H, W, _ = nhwc_u8.shape
    if out_nchw_fp16 is None:
        out_buf = torch.empty((B, 3, H, W), dtype=torch.float16, device=nhwc_u8.device)
    else:
        if out_nchw_fp16.shape != (B, 3, H, W) or out_nchw_fp16.dtype != torch.float16:
            raise ValueError(
                f"out_nchw_fp16 phải có shape={(B, 3, H, W)} và dtype=float16, "
                f"nhận được shape={tuple(out_nchw_fp16.shape)}, dtype={out_nchw_fp16.dtype}"
            )
        out_buf = out_nchw_fp16

    if B == 0 or H == 0 or W == 0:
        return out_buf

    should_use_triton = (
        bool(use_triton)
        if use_triton is not None
        else (_HAS_TRITON and nhwc_u8.is_cuda and out_buf.is_cuda)
    )
    if (
        should_use_triton
        and _HAS_TRITON
        and _nhwc_u8_to_nchw_fp16_norm_kernel is not None
        and nhwc_u8.dtype == torch.uint8
        and nhwc_u8.stride(1) == W * 3
        and nhwc_u8.stride(2) == 3
        and nhwc_u8.stride(3) == 1
        and out_buf.stride(2) == W
        and out_buf.stride(3) == 1
    ):
        HW = H * W
        BLOCK_P = 1024
        grid = (triton.cdiv(HW, BLOCK_P), B)
        _nhwc_u8_to_nchw_fp16_norm_kernel[grid](
            nhwc_u8,
            out_buf,
            HW,
            nhwc_u8.stride(0),
            out_buf.stride(0),
            out_buf.stride(1),
            BLOCK_P=BLOCK_P,
            num_warps=4,
        )
        return out_buf

    out_buf.copy_(nhwc_u8.permute(0, 3, 1, 2))
    out_buf.mul_(1.0 / 255.0)
    return out_buf


# 1. GPU COLOR CONVERTER P010LE (BT.709 10-BIT LIMITED RANGE)

class GPUColorConverterP010LE:
    """Convert RGB [B, 3, H, W] tensors in [0.0, 1.0] to ITU-R BT.709 limited-range P010LE."""

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._p010_ring: List[torch.Tensor] = []
        self._p010_ring_slot: int = 0
        self._has_triton = False

        if device.type == "cuda" and HAS_TRITON and _rgb_to_p010le_triton_kernel is not None:
            self._has_triton = True

    def _ensure_ring(self, B: int, H: int, W: int, device: torch.device) -> None:
        alloc_b = max(int(B), 8)
        out_shape = (alloc_b, int(H * 3 // 2), int(W))
        if (
            len(self._p010_ring) != 2
            or self._p010_ring[0].shape[0] < alloc_b
            or self._p010_ring[0].shape[1:] != out_shape[1:]
            or self._p010_ring[0].device != device
        ):
            self._p010_ring = [
                torch.empty(out_shape, dtype=torch.int16, device=device)
                for _ in range(2)
            ]
            self._p010_ring_slot = 0

    @staticmethod
    def _p010le_core(rgb_slice: torch.Tensor) -> torch.Tensor:
        B, _, H, W = rgb_slice.shape
        # Clamp RGB overshoots and NaNs to [0, 1] before BT.709 matrix transform to avoid chroma leak
        out_fp16 = torch.nan_to_num(rgb_slice.float(), nan=0.0, posinf=1.0, neginf=0.0)
        rgb_clamped = out_fp16.clamp_(0.0, 1.0)
        r = rgb_clamped[:, 0]
        g = rgb_clamped[:, 1]
        b = rgb_clamped[:, 2]

        # 1. BT.709 limited-range luma Y [64..940]
        y = torch.empty_like(r).fill_(64.0)
        y.add_(r, alpha=186.2376)
        y.add_(g, alpha=626.5152)
        y.add_(b, alpha=63.2472)
        y_p010 = (torch.clamp(torch.round(y), 64.0, 940.0).to(torch.int32) << 6).to(torch.int16)

        # 2. 4:2:0 chroma subsampling before BT.709 UV transform [64..960]
        rgb_sub = F.avg_pool2d(rgb_clamped, kernel_size=2, stride=2)
        rs = rgb_sub[:, 0]
        gs = rgb_sub[:, 1]
        bs = rgb_sub[:, 2]

        u = torch.empty_like(rs).fill_(512.0)
        u.add_(rs, alpha=-102.656512)
        u.add_(gs, alpha=-345.343488)
        u.add_(bs, alpha=448.0000)

        v = torch.empty_like(rs).fill_(512.0)
        v.add_(rs, alpha=448.0000)
        v.add_(gs, alpha=-406.921088)
        v.add_(bs, alpha=-41.078912)

        u_p010 = (torch.clamp(torch.round(u), 64.0, 960.0).to(torch.int32) << 6).to(torch.int16)
        v_p010 = (torch.clamp(torch.round(v), 64.0, 960.0).to(torch.int32) << 6).to(torch.int16)

        # 3. Interleave semi-planar UV [B, H/2, W] after Y [B, H, W]
        uv_p010 = torch.stack((u_p010, v_p010), dim=-1).view(B, H // 2, W)
        return torch.cat((y_p010, uv_p010), dim=1).contiguous()

    @torch.inference_mode()
    def _convert_slice(self, rgb_slice: torch.Tensor) -> torch.Tensor:
        return self._p010le_core(rgb_slice)

    @torch.inference_mode()
    def rgb_to_p010le(self, rgb_tensor: torch.Tensor, slot: Optional[int] = None) -> torch.Tensor:
        """Convert RGB [B, 3, H, W] into 2-slot ring-buffered Semi-Planar P010LE [B, 3*H//2, W]."""
        B, _, H, W = rgb_tensor.shape
        self._ensure_ring(B, H, W, rgb_tensor.device)
        ring_slot = self._p010_ring_slot if slot is None else (slot % len(self._p010_ring))
        if slot is None:
            self._p010_ring_slot = (self._p010_ring_slot + 1) % len(self._p010_ring)

        out_buf = self._p010_ring[ring_slot][:B]

        # Prefer Triton kernel on CUDA for single-pass conversion
        if self._has_triton and rgb_tensor.is_cuda and _rgb_to_p010le_triton_kernel is not None:
            BLOCK_X = 16
            BLOCK_Y = 16
            grid = (triton.cdiv(W // 2, BLOCK_X), triton.cdiv(H // 2, BLOCK_Y), B)
            _rgb_to_p010le_triton_kernel[grid](
                rgb_tensor, out_buf,
                rgb_tensor.stride(0), rgb_tensor.stride(1), rgb_tensor.stride(2), rgb_tensor.stride(3),
                out_buf.stride(0), out_buf.stride(1), out_buf.stride(2),
                H, W,
                BLOCK_X=BLOCK_X, BLOCK_Y=BLOCK_Y
            )
            return out_buf

        # CPU / non-Triton fallback
        if B <= 2:
            out_buf.copy_(self._convert_slice(rgb_tensor))
            return out_buf

        for s in range(0, B, 2):
            e = min(s + 2, B)
            out_buf[s:e].copy_(self._convert_slice(rgb_tensor[s:e]))
        return out_buf


class SRVGGNetCompactTRTWrapper(nn.Module):
    """Hybrid-precision wrapper running conv backbone in FP16 and residual/bicubic tail in FP32."""
    def __init__(self, base_srvgg_model: nn.Module, scale: Optional[float] = None):
        super().__init__()
        self.body = base_srvgg_model.body
        self.upsampler = base_srvgg_model.upsampler
        if hasattr(base_srvgg_model, "upscale"):
            self.upscale = base_srvgg_model.upscale
        else:
            self.upscale = 4 if isinstance(self.upsampler, nn.PixelShuffle) and self.upsampler.upscale_factor == 4 else 2
        self.target_scale = float(scale) if scale is not None else 2.0

    def _tail_upsample(self, feat_slice: torch.Tensor, x_slice: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        out = self.upsampler(feat_slice)
        base = F.interpolate(x_slice.float(), scale_factor=float(self.upscale), mode="nearest")
        out_f32 = out.float() + base

        if abs(float(self.upscale) - self.target_scale) > 1e-4:
            out_target = F.interpolate(
                out_f32,
                scale_factor=self.target_scale / float(self.upscale),
                mode="bicubic",
                align_corners=False,
            )
        else:
            out_target = out_f32
        return out_target.to(out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = torch.float16 if x.dtype == torch.float16 else torch.float32
        feat = x
        for layer in self.body:
            feat = layer(feat)

        B, _, H, W = x.shape
        # Micro-slice 8K tail across 2 frames on Full HD+ batches outside ONNX export
        if (not torch.onnx.is_in_onnx_export()) and B > 2 and self.upscale >= 4 and (H * W) >= (540 * 960):
            out_h = int(round(H * self.target_scale))
            out_w = int(round(W * self.target_scale))
            out_buf = torch.empty((B, 3, out_h, out_w), dtype=out_dtype, device=x.device)
            for s in range(0, B, 2):
                e = min(s + 2, B)
                out_buf[s:e].copy_(self._tail_upsample(feat[s:e], x[s:e], out_dtype))
            return out_buf

        return self._tail_upsample(feat, x, out_dtype)


def _extract_proc_stderr(encoder_proc: Any) -> str:
    """Read buffered stderr tail lines or pipe stderr from an FFmpeg subprocess."""
    if encoder_proc is None:
        return ""
    try:
        t_err = getattr(encoder_proc, "_stderr_thread", None)
        if t_err is not None and t_err.is_alive():
            t_err.join(timeout=0.3)
        tail = getattr(encoder_proc, "_stderr_tail", None)
        if tail is not None:
            return "\n".join(list(tail)).strip()
        stderr_obj = getattr(encoder_proc, "stderr", None)
        if stderr_obj is not None:
            raw_err = stderr_obj.read()
            if isinstance(raw_err, bytes):
                return raw_err.decode("utf-8", errors="replace").strip()
            if isinstance(raw_err, str):
                return raw_err.strip()
    except Exception:
        pass
    return ""


class _PipeDrainWorker:
    """
    Worker thread chuyên trách xả dữ liệu khung hình P010LE vào FFmpeg NVENC pipe nền.
    Giải phóng luồng CPU chính để phát lệnh liên tục cho GPU, triệt tiêu GPU bubble.
    """
    def __init__(self, encoder_proc: Any, write_fn: Any):
        self.encoder_proc = encoder_proc
        self._pipe = getattr(encoder_proc, "stdin", None)
        self.write_fn = write_fn
        self.queue: queue.Queue = queue.Queue(maxsize=16)
        self.error: Optional[Exception] = None
        self._aborted = False
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="PipeDrainWorker", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._aborted:
                try:
                    item = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    self.queue.task_done()
                    break
                buf_arr, repeats, done_evt = item
                try:
                    if not self._aborted and buf_arr is not None and repeats > 0:
                        self.write_fn(self.encoder_proc, buf_arr, repeats)
                finally:
                    if done_evt is not None:
                        done_evt.set()
                    self.queue.task_done()
        except Exception as exc:
            self.error = exc

    def check_error(self) -> None:
        """Kiểm tra lỗi hoặc trạng thái dừng bất thường của luồng xả pipe."""
        if self.error is not None:
            raise self.error
        if self._aborted:
            raise RuntimeError("PipeDrainWorker đã bị hủy (aborted).")
        if not self._closed and not self._thread.is_alive():
            raise RuntimeError("PipeDrainWorker thread died unexpectedly")

    def submit(self, buf_arr: Optional[np.ndarray], repeats: int, done_evt: Optional[threading.Event] = None) -> None:
        while True:
            if self._aborted:
                raise RuntimeError("PipeDrainWorker đã bị hủy.")
            if self.error is not None:
                raise self.error
            if not self._thread.is_alive():
                raise RuntimeError("PipeDrainWorker thread died unexpectedly") from self.error
            try:
                self.queue.put((buf_arr, repeats, done_evt), timeout=0.1)
                break
            except queue.Full:
                continue

    def close(self) -> None:
        self._closed = True
        if self.error is not None:
            if self._thread.is_alive():
                self._thread.join(timeout=3.0)
            raise self.error
        if self._thread.is_alive():
            while self._thread.is_alive():
                if self.error is not None:
                    if self._thread.is_alive():
                        self._thread.join(timeout=3.0)
                    raise self.error
                try:
                    self.queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self._thread.join()
        if self.error is not None:
            raise self.error

    def abort(self, encoder_proc: Any = None) -> None:
        """Hủy khẩn cấp worker: đóng pipe, ngắt encoder nếu còn sống, giải phóng mọi event đang chờ."""
        self._aborted = True
        proc = encoder_proc if encoder_proc is not None else self.encoder_proc
        pipe = self._pipe or getattr(proc, "stdin", None)
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass
        if proc is not None:
            poll_fn = getattr(proc, "poll", None)
            rc = poll_fn() if callable(poll_fn) else None
            if rc is None:
                if hasattr(proc, "terminate"):
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                elif hasattr(proc, "kill"):
                    try:
                        proc.kill()
                    except Exception:
                        pass
                wait_fn = getattr(proc, "wait", None)
                if callable(wait_fn):
                    try:
                        wait_fn(timeout=1.0)
                    except Exception:
                        if hasattr(proc, "kill"):
                            try:
                                proc.kill()
                            except Exception:
                                pass
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                _, _, done_evt = item
                if done_evt is not None:
                    done_evt.set()
            try:
                self.queue.task_done()
            except ValueError:
                pass
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


class DoubleBufferedPipeUpscaler:
    """Three-stream overlapped H2D/Compute/D2H upscaler writing directly to an FFmpeg NVENC pipe."""
    def __init__(
        self,
        model_or_engine: Any,
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        input_hw: Tuple[int, int] = (1080, 1920),
        scale: int = 2,
        enable_micro_timings: bool = True,
    ):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.batch_size = max(1, batch_size)
        self.model = model_or_engine
        self.scale = max(1, int(scale))
        self.color_conv = GPUColorConverterP010LE(self.device)
        self.input_hw = (0, 0)
        self.output_hw = (0, 0)
        self.enable_micro_timings = enable_micro_timings
        self.last_micro_timings: Optional[Dict[str, Any]] = None

        if self.device.type == "cuda":
            self.stream_h2d = torch.cuda.Stream(device=self.device)
            self.stream_compute = torch.cuda.Stream(device=self.device)
            self.stream_d2h = torch.cuda.Stream(device=self.device)
        else:
            self.stream_h2d = None
            self.stream_compute = None
            self.stream_d2h = None

        self.host_in_pins: List[torch.Tensor] = []
        self.host_out_pins: List[torch.Tensor] = []
        self.gpu_in_ring: List[torch.Tensor] = []
        self.gpu_nhwc_ring: List[torch.Tensor] = []
        self._micro_event_pool: List[Tuple[Any, ...]] = []
        self._sync_event_pool: List[Any] = []
        self.last_first_rgb_small: Optional[np.ndarray] = None
        self.last_distinct_count: int = 0
        self.ensure_resolution(input_hw[0], input_hw[1])
        if self.device.type == "cuda" and not os.environ.get("PYTEST_CURRENT_TEST"):
            self.warmup(batch_size=min(2, self.batch_size))

    def _ensure_cuda_event_pools(self, num_batches: int, record_micro: bool) -> None:
        """Pre-allocate reusable CUDA events outside the hot loop."""
        if self.device.type != "cuda":
            return
        while len(self._sync_event_pool) < num_batches:
            self._sync_event_pool.append(torch.cuda.Event(enable_timing=False))
        if record_micro:
            while len(self._micro_event_pool) < num_batches:
                self._micro_event_pool.append(
                    tuple(torch.cuda.Event(enable_timing=True) for _ in range(9))
                )

    def ensure_resolution(self, in_h: int, in_w: int) -> Tuple[int, int]:
        """Resize pinned host and VRAM ring buffers when chunk frame dimensions change."""
        out_h = in_h * self.scale
        out_w = in_w * self.scale
        if (out_h % 2 != 0) or (out_w % 2 != 0):
            raise ValueError(f"Độ phân giải đầu ra p010le phải chẵn, nhận được: {out_w}x{out_h}")
        if self.input_hw == (in_h, in_w) and len(self.host_in_pins) == 2:
            return self.output_hw

        self.input_hw = (in_h, in_w)
        self.output_hw = (out_h, out_w)
        pin = self.device.type == "cuda"
        self.host_in_pins = [
            torch.empty((self.batch_size, 3, in_h, in_w), dtype=torch.uint8, pin_memory=pin)
            for _ in range(2)
        ]
        self.host_out_pins = [
            torch.empty((self.batch_size, out_h * 3 // 2, out_w), dtype=torch.int16, pin_memory=pin)
            for _ in range(2)
        ]
        if self.device.type == "cuda":
            self.gpu_in_ring = [
                torch.empty((self.batch_size, 3, in_h, in_w), dtype=torch.float16, device=self.device)
                for _ in range(2)
            ]
            self.gpu_nhwc_ring = [
                torch.empty((self.batch_size, in_h, in_w, 3), dtype=torch.uint8, device=self.device)
                for _ in range(2)
            ]
        else:
            self.gpu_in_ring = []
            self.gpu_nhwc_ring = []
        return self.output_hw

    def warmup(self, batch_size: int = 2) -> None:
        """Warm up Triton input norm, color converter, and TRT execution context before timing."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return
        try:
            cur_bs = min(max(1, batch_size), self.batch_size)
            in_h, in_w = self.input_hw if self.input_hw[0] > 0 else (1080, 1920)
            out_h, out_w = self.output_hw if self.output_hw[0] > 0 else (in_h * self.scale, in_w * self.scale)
            if self.gpu_nhwc_ring and self.gpu_nhwc_ring[0].shape[0] >= cur_bs and self.gpu_nhwc_ring[0].shape[1:3] == (in_h, in_w):
                dummy_nhwc = self.gpu_nhwc_ring[0][:cur_bs]
                dummy_nhwc.zero_()
            else:
                dummy_nhwc = torch.zeros((cur_bs, in_h, in_w, 3), dtype=torch.uint8, device=self.device)
            dummy_in_buf = (
                self.gpu_in_ring[0][:cur_bs]
                if self.gpu_in_ring and self.gpu_in_ring[0].shape[0] >= cur_bs and self.gpu_in_ring[0].shape[2:4] == (in_h, in_w)
                else torch.empty((cur_bs, 3, in_h, in_w), dtype=torch.float16, device=self.device)
            )
            with torch.inference_mode():
                dummy_in = _nhwc_u8_to_nchw_fp16_norm(dummy_nhwc, out_nchw_fp16=dummy_in_buf)
                if callable(self.model):
                    dummy_out = self.model(dummy_in)
                else:
                    dummy_out = torch.zeros((cur_bs, 3, out_h, out_w), dtype=torch.float16, device=self.device)
                self.color_conv.rgb_to_p010le(dummy_out, slot=0)
                torch.cuda.synchronize(self.device)
        except Exception:
            try:
                cur_bs = min(max(1, batch_size), self.batch_size)
                in_h, in_w = self.input_hw if self.input_hw[0] > 0 else (1080, 1920)
                out_h = self.output_hw[0] if self.output_hw[0] > 0 else 2160
                out_w = self.output_hw[1] if self.output_hw[1] > 0 else 3840
                dummy_nhwc = torch.zeros((cur_bs, in_h, in_w, 3), dtype=torch.uint8, device=self.device)
                dummy_out = torch.zeros((cur_bs, 3, out_h, out_w), dtype=torch.float16, device=self.device)
                with torch.inference_mode():
                    _nhwc_u8_to_nchw_fp16_norm(dummy_nhwc)
                    self.color_conv.rgb_to_p010le(dummy_out, slot=0)
                    torch.cuda.synchronize(self.device)
            except Exception:
                pass

    @staticmethod
    def spawn_nvenc_pipe_writer(
        output_mp4_path: str,
        fps_str: str = "24000/1001",
        width: int = 3840,
        height: int = 2160,
    ) -> subprocess.Popen:
        """Spawn FFmpeg NVENC receiving rawvideo P010LE over stdin pipe."""
        rate_args = get_hevc_rate_args(bitrate="28M")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "p010le",
            "-s", f"{width}x{height}",
            "-color_range", HEVC_COLOR_RANGE,
            "-colorspace", HEVC_COLORSPACE,
            "-color_primaries", HEVC_COLOR_PRIMARIES,
            "-color_trc", HEVC_COLOR_TRC,
            "-chroma_sample_location", "center",
            "-r", fps_str,
            "-i", "pipe:0",
            "-c:v", HEVC_ENCODER,
            "-profile:v", HEVC_PROFILE,
            "-pix_fmt", HEVC_PIX_FMT,
            *rate_args,
            "-tag:v", HEVC_TAG,
            output_mp4_path
        ]
        # Unbuffered stdin + daemon stderr drain avoid 64 MiB MDL lock / 4 KiB stderr deadlock
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        if getattr(proc, "stderr", None) is not None:
            stderr_tail: collections.deque[str] = collections.deque(maxlen=64)
            proc._stderr_tail = stderr_tail

            def _drain_stderr(stream=proc.stderr, tail=stderr_tail) -> None:
                try:
                    for raw_line in iter(stream.readline, b""):
                        if isinstance(raw_line, (bytes, bytearray)):
                            line = raw_line.decode("utf-8", errors="replace").rstrip()
                        else:
                            line = str(raw_line).rstrip()
                        if line:
                            tail.append(line)
                except Exception:
                    pass
                finally:
                    try:
                        stream.close()
                    except Exception:
                        pass

            t_err = threading.Thread(target=_drain_stderr, name="NVENCStderrDrain", daemon=True)
            proc._stderr_thread = t_err
            t_err.start()
        return proc

    @staticmethod
    def _write_frame_to_pipe(encoder_proc: Any, buf_arr: np.ndarray, repeats: int = 1) -> None:
        """Write repeated P010LE frame buffers to stdin in 8 MiB chunks to avoid Windows MDL exhaustion."""
        if repeats <= 0:
            return
        mv = memoryview(buf_arr).cast("B")
        total_len = len(mv)
        # 8 MiB slices stay safely below the Windows 64 MiB pipe MDL page-lock limit
        write_step = 8 * 1024 * 1024
        try:
            for _ in range(repeats):
                offset = 0
                while offset < total_len:
                    chunk = mv[offset : min(offset + write_step, total_len)]
                    written = encoder_proc.stdin.write(chunk)
                    if written is None or written <= 0:
                        raise BrokenPipeError(
                            f"FFmpeg NVENC pipe stdin.write() trả về {written!r} tại offset {offset}/{total_len}"
                        )
                    offset += int(written)
        except (BrokenPipeError, OSError) as pipe_err:
            rc = None
            try:
                if hasattr(encoder_proc, "poll"):
                    rc = encoder_proc.poll()
            except Exception:
                pass
            stderr_detail = _extract_proc_stderr(encoder_proc)
            detail_suffix = f" | FFmpeg stderr: {stderr_detail}" if stderr_detail else ""
            raise RuntimeError(
                f"Không thể ghi khung hình vào FFmpeg NVENC pipe (rc={rc}, err={pipe_err}){detail_suffix}"
            ) from pipe_err

    @torch.inference_mode()
    def process_frames_to_pipe(
        self,
        unique_frames_rgb: List[np.ndarray],
        repeat_counts: List[int],
        encoder_proc: subprocess.Popen,
        cyclic_dedup: bool = True,
        capture_first_rgb_small: bool = False,
        enable_micro_timings: Optional[bool] = None,
        precomputed_dedup: Optional[Tuple[List[np.ndarray], List[int]]] = None,
    ) -> int:
        """Upscale deduplicated segment frames with cyclic memoization and stream them as P010LE into encoder_proc."""
        self.last_first_rgb_small = None
        num_unique = len(unique_frames_rgb)
        if num_unique == 0:
            self.last_distinct_count = 0
            return 0

        if repeat_counts and num_unique < len(repeat_counts):
            missing = len(repeat_counts) - num_unique
            unique_frames_rgb = list(unique_frames_rgb) + [unique_frames_rgb[-1]] * missing
            num_unique = len(unique_frames_rgb)

        f_h, f_w = unique_frames_rgb[0].shape[:2]
        self.ensure_resolution(f_h, f_w)

        if precomputed_dedup is not None and precomputed_dedup[0] is not None and precomputed_dedup[1] is not None:
            distinct_frames, frame_to_distinct = precomputed_dedup
        elif cyclic_dedup and num_unique > 1:
            from src.dedup import deduplicate_cyclic_frames
            distinct_frames, frame_to_distinct = deduplicate_cyclic_frames(unique_frames_rgb)
        else:
            distinct_frames = unique_frames_rgb
            frame_to_distinct = list(range(num_unique))

        num_distinct = len(distinct_frames)
        self.last_distinct_count = num_distinct

        remaining_refs = [0] * num_distinct
        for d_idx in frame_to_distinct:
            remaining_refs[d_idx] += 1

        memo_p010: dict[int, np.ndarray] = {}
        next_u_idx = 0
        total_frames_written = 0

        record_micro = self.enable_micro_timings if enable_micro_timings is None else bool(enable_micro_timings)
        self.last_micro_timings = None
        drain_worker = _PipeDrainWorker(encoder_proc, self._write_frame_to_pipe)
        slot_drain_events = [threading.Event(), threading.Event()]
        slot_drain_events[0].set()
        slot_drain_events[1].set()

        def _drain_ready_frames(d_limit: int, slot_views: dict[int, np.ndarray], slot_idx: Optional[int] = None) -> None:
            nonlocal next_u_idx, total_frames_written
            last_submitted_slot = False
            while next_u_idx < num_unique:
                req_d = frame_to_distinct[next_u_idx]
                if req_d >= d_limit:
                    break
                buf_arr = slot_views.get(req_d)
                if buf_arr is not None:
                    last_submitted_slot = True
                    if remaining_refs[req_d] > 1 and req_d not in memo_p010:
                        memo_p010[req_d] = buf_arr.copy()
                else:
                    buf_arr = memo_p010[req_d]
                repeats = repeat_counts[next_u_idx] if repeat_counts else 1
                drain_worker.submit(buf_arr, repeats)
                total_frames_written += repeats
                remaining_refs[req_d] -= 1
                if remaining_refs[req_d] <= 0 and req_d in memo_p010:
                    del memo_p010[req_d]
                next_u_idx += 1

            # Only retain frames still referenced by future batches (Belady OPT eviction)
            for d_idx, view_arr in slot_views.items():
                if remaining_refs[d_idx] > 0 and d_idx not in memo_p010:
                    memo_p010[d_idx] = view_arr.copy()

            if slot_idx is not None and last_submitted_slot:
                slot_drain_events[slot_idx].clear()
                drain_worker.submit(None, 0, done_evt=slot_drain_events[slot_idx])

        ok = False
        try:
            if self.device.type == "cuda":
                batches = []
                for start_idx in range(0, num_distinct, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_distinct)
                    batches.append((start_idx, end_idx))
                num_batches = len(batches)
                sanity_gpu_box: List[Optional[torch.Tensor]] = [None]
                self._ensure_cuda_event_pools(num_batches, record_micro)
                batch_timings = []

                def _dispatch_batch_cuda(batch_k: int):
                    slot = batch_k % 2
                    s_idx, e_idx = batches[batch_k]
                    cur_B = e_idx - s_idx

                    if record_micro:
                        (
                            evt_h2d_s,
                            evt_h2d_dma_e,
                            evt_h2d_e,
                            evt_trt_s,
                            evt_trt_e,
                            evt_tri_s,
                            evt_tri_e,
                            evt_d2h_s,
                            evt_d2h_e,
                        ) = self._micro_event_pool[batch_k]
                    else:
                        evt_h2d_s = evt_h2d_dma_e = evt_h2d_e = None
                        evt_trt_s = evt_trt_e = None
                        evt_tri_s = evt_tri_e = None
                        evt_d2h_s = evt_d2h_e = None

                    # 1. Stage NHWC input to pinned memory and enqueue async H2D copy
                    with torch.cuda.stream(self.stream_h2d):
                        if record_micro and evt_h2d_s is not None:
                            evt_h2d_s.record(self.stream_h2d)
                        first_arr = distinct_frames[s_idx]
                        is_hwc = first_arr.ndim == 3 and first_arr.shape[-1] == 3
                        is_trt = bool(getattr(self.model, "is_tensorrt", False))
                        if is_hwc:
                            hwc_pin = self.host_in_pins[slot].view(self.batch_size, f_h, f_w, 3)
                            for i in range(cur_B):
                                hwc_pin[i].copy_(torch.from_numpy(distinct_frames[s_idx + i]))
                            if self.gpu_nhwc_ring:
                                gpu_nhwc = self.gpu_nhwc_ring[slot][:cur_B]
                                gpu_nhwc.copy_(hwc_pin[:cur_B], non_blocking=True)
                            else:
                                gpu_nhwc = hwc_pin[:cur_B].to(self.device, non_blocking=True)
                            if record_micro and evt_h2d_dma_e is not None:
                                evt_h2d_dma_e.record(self.stream_h2d)
                            if self.gpu_in_ring:
                                # Fused Triton NHWC u8->NCHW fp16 directly into fixed gpu_in_ring slot for TRT execute_async_v3
                                in_buf = self.gpu_in_ring[slot][:cur_B]
                                gpu_tensor = _nhwc_u8_to_nchw_fp16_norm(gpu_nhwc, out_nchw_fp16=in_buf)
                            else:
                                gpu_tensor = _nhwc_u8_to_nchw_fp16_norm(gpu_nhwc)
                        else:
                            for i in range(cur_B):
                                self.host_in_pins[slot][i].copy_(torch.from_numpy(distinct_frames[s_idx + i]))
                            gpu_in = self.host_in_pins[slot][:cur_B].to(self.device, non_blocking=True)
                            if record_micro and evt_h2d_dma_e is not None:
                                evt_h2d_dma_e.record(self.stream_h2d)
                            if is_trt and self.gpu_in_ring:
                                in_buf = self.gpu_in_ring[slot][:cur_B]
                                in_buf.copy_(gpu_in)
                                gpu_tensor = in_buf.mul_(1.0 / 255.0)
                            elif is_trt:
                                gpu_tensor = gpu_in.contiguous().to(torch.float16).mul_(1.0 / 255.0)
                            else:
                                gpu_tensor = (
                                    gpu_in.to(memory_format=torch.channels_last)
                                    .to(torch.float16)
                                    .mul_(1.0 / 255.0)
                                )
                        if record_micro and evt_h2d_e is not None:
                            evt_h2d_e.record(self.stream_h2d)

                    # 2. Wait for H2D transfer, run SR inference, and convert to P010LE in VRAM
                    self.stream_compute.wait_stream(self.stream_h2d)
                    with torch.cuda.stream(self.stream_compute):
                        gpu_tensor.record_stream(self.stream_compute)
                        if record_micro and evt_trt_s is not None:
                            evt_trt_s.record(self.stream_compute)
                        upscaled_rgb = self.model(gpu_tensor)
                        if record_micro and evt_trt_e is not None:
                            evt_trt_e.record(self.stream_compute)
                        if capture_first_rgb_small and batch_k == 0:
                            u0 = torch.nan_to_num(
                                upscaled_rgb[0:1].float(), nan=0.0, posinf=1.0, neginf=0.0
                            ).clamp(0.0, 1.0)
                            oh, ow = int(u0.shape[2]), int(u0.shape[3])
                            if oh >= 270 and ow >= 480 and (oh % 270 == 0) and (ow % 480 == 0):
                                u0 = F.avg_pool2d(u0, kernel_size=(oh // 270, ow // 480))
                            sanity_gpu_box[0] = (
                                u0.mul(255.0)
                                .round()
                                .to(torch.uint8)
                                .squeeze(0)
                                .permute(1, 2, 0)
                                .contiguous()
                            )
                        if record_micro and evt_tri_s is not None:
                            evt_tri_s.record(self.stream_compute)
                        gpu_p010 = self.color_conv.rgb_to_p010le(upscaled_rgb, slot=slot)
                        if record_micro and evt_tri_e is not None:
                            evt_tri_e.record(self.stream_compute)

                    # 3. Wait for _PipeDrainWorker to finish draining prior slot before D2H overwrite
                    drain_worker.check_error()
                    wait_start = time.monotonic()
                    while not slot_drain_events[slot].wait(timeout=0.25):
                        drain_worker.check_error()
                        poll_fn = getattr(encoder_proc, "poll", None)
                        rc = poll_fn() if callable(poll_fn) else None
                        if isinstance(rc, int):
                            stderr_detail = _extract_proc_stderr(encoder_proc)
                            suffix = f" | FFmpeg stderr: {stderr_detail}" if stderr_detail else ""
                            raise RuntimeError(
                                f"FFmpeg NVENC encoder thoát sớm khi chờ pipe drain (rc={rc}){suffix}"
                            )
                        if (time.monotonic() - wait_start) >= 120.0:
                            raise RuntimeError(
                                f"Quá thời gian chờ xả khung hình P010LE vào NVENC pipe (>120s) tại slot {slot}."
                            )
                    self.stream_d2h.wait_stream(self.stream_compute)
                    with torch.cuda.stream(self.stream_d2h):
                        if record_micro and evt_d2h_s is not None:
                            evt_d2h_s.record(self.stream_d2h)
                        gpu_p010.record_stream(self.stream_d2h)
                        self.host_out_pins[slot][:cur_B].copy_(gpu_p010, non_blocking=True)
                        if batch_k == 0 and sanity_gpu_box[0] is not None:
                            sanity_gpu_box[0].record_stream(self.stream_d2h)
                            sanity_cpu = sanity_gpu_box[0].to("cpu", non_blocking=True)
                        else:
                            sanity_cpu = None
                        if record_micro and evt_d2h_e is not None:
                            evt_d2h_e.record(self.stream_d2h)

                    evt = self._sync_event_pool[batch_k]
                    evt.record(self.stream_d2h)
                    if record_micro:
                        batch_timings.append(
                            (
                                evt_h2d_s,
                                evt_h2d_dma_e,
                                evt_h2d_e,
                                evt_trt_s,
                                evt_trt_e,
                                evt_tri_s,
                                evt_tri_e,
                                evt_d2h_s,
                                evt_d2h_e,
                            )
                        )
                    return evt, sanity_cpu

                # Prologue: dispatch batch 0
                curr_event, curr_sanity_cpu = _dispatch_batch_cuda(0)

                # Steady-state: overlap H2D/Compute(b+1) with D2H/pipe write(b)
                for b in range(num_batches):
                    next_event, next_sanity_cpu = (None, None)
                    if b + 1 < num_batches:
                        next_event, next_sanity_cpu = _dispatch_batch_cuda(b + 1)

                    curr_event.synchronize()
                    if b == 0 and curr_sanity_cpu is not None:
                        self.last_first_rgb_small = curr_sanity_cpu.numpy()

                    slot = b % 2
                    s_idx, e_idx = batches[b]
                    cur_B = e_idx - s_idx
                    out_np = self.host_out_pins[slot][:cur_B].numpy()

                    slot_views = {s_idx + i: out_np[i] for i in range(cur_B)}
                    _drain_ready_frames(e_idx, slot_views, slot_idx=slot)

                    curr_event, curr_sanity_cpu = next_event, next_sanity_cpu

                if record_micro and batch_timings:
                    total_h2d_dma = 0.0
                    total_in_norm = 0.0
                    total_h2d = 0.0
                    total_trt = 0.0
                    total_tri = 0.0
                    total_d2h = 0.0
                    for h_s, h_dma_e, h_e, t_s, t_e, r_s, r_e, d_s, d_e in batch_timings:
                        try:
                            if h_s is not None and h_dma_e is not None:
                                total_h2d_dma += h_s.elapsed_time(h_dma_e)
                            if h_dma_e is not None and h_e is not None:
                                total_in_norm += h_dma_e.elapsed_time(h_e)
                            if h_s is not None and h_e is not None:
                                total_h2d += h_s.elapsed_time(h_e)
                            if t_s is not None and t_e is not None:
                                total_trt += t_s.elapsed_time(t_e)
                            if r_s is not None and r_e is not None:
                                total_tri += r_s.elapsed_time(r_e)
                            if d_s is not None and d_e is not None:
                                total_d2h += d_s.elapsed_time(d_e)
                        except Exception:
                            pass
                    n_b = max(1, len(batch_timings))
                    self.last_micro_timings = {
                        "num_batches": len(batch_timings),
                        "total_h2d_dma_ms": round(total_h2d_dma, 2),
                        "total_in_norm_ms": round(total_in_norm, 2),
                        "total_h2d_ms": round(total_h2d, 2),
                        "total_trt_ms": round(total_trt, 2),
                        "total_triton_ms": round(total_tri, 2),
                        "total_d2h_ms": round(total_d2h, 2),
                        "avg_h2d_dma_ms": round(total_h2d_dma / n_b, 2),
                        "avg_in_norm_ms": round(total_in_norm / n_b, 2),
                        "avg_h2d_ms": round(total_h2d / n_b, 2),
                        "avg_trt_ms": round(total_trt / n_b, 2),
                        "avg_triton_ms": round(total_tri / n_b, 2),
                        "avg_d2h_ms": round(total_d2h / n_b, 2),
                    }

            else:
                # CPU Fallback
                cpu_total_norm_ms = 0.0
                cpu_total_infer_ms = 0.0
                cpu_total_conv_ms = 0.0
                for start_idx in range(0, num_distinct, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_distinct)
                    cur_B = end_idx - start_idx
                    t_nrm_s = time.perf_counter()
                    batch_tensors = []
                    for i in range(cur_B):
                        t = torch.from_numpy(distinct_frames[start_idx + i])
                        if t.ndim == 3 and t.shape[-1] == 3:
                            t = t.permute(2, 0, 1)
                        batch_tensors.append(t)
                    tensor = torch.stack(batch_tensors, dim=0).float().div_(255.0)
                    cpu_total_norm_ms += (time.perf_counter() - t_nrm_s) * 1000.0
                    t_inf_s = time.perf_counter()
                    out_rgb = self.model(tensor)
                    cpu_total_infer_ms += (time.perf_counter() - t_inf_s) * 1000.0

                    if capture_first_rgb_small and start_idx == 0:
                        u0 = torch.nan_to_num(
                            out_rgb[0:1].float(), nan=0.0, posinf=1.0, neginf=0.0
                        ).clamp(0.0, 1.0)
                        oh, ow = int(u0.shape[2]), int(u0.shape[3])
                        if oh >= 270 and ow >= 480 and (oh % 270 == 0) and (ow % 480 == 0):
                            u0 = F.avg_pool2d(u0, kernel_size=(oh // 270, ow // 480))
                        self.last_first_rgb_small = (
                            u0.mul(255.0)
                            .round()
                            .to(torch.uint8)
                            .squeeze(0)
                            .permute(1, 2, 0)
                            .contiguous()
                            .numpy()
                        )
                    t_cnv_s = time.perf_counter()
                    out_p010 = self.color_conv.rgb_to_p010le(out_rgb)
                    cpu_total_conv_ms += (time.perf_counter() - t_cnv_s) * 1000.0
                    out_np = out_p010.numpy()

                    slot_views = {start_idx + i: out_np[i] for i in range(cur_B)}
                    _drain_ready_frames(end_idx, slot_views)

                if record_micro:
                    n_b = max(1, (num_distinct + self.batch_size - 1) // self.batch_size)
                    self.last_micro_timings = {
                        "num_batches": n_b,
                        "total_h2d_dma_ms": 0.0,
                        "total_in_norm_ms": round(cpu_total_norm_ms, 2),
                        "total_h2d_ms": 0.0,
                        "total_trt_ms": round(cpu_total_infer_ms, 2),
                        "total_triton_ms": round(cpu_total_conv_ms, 2),
                        "total_d2h_ms": 0.0,
                        "avg_h2d_dma_ms": 0.0,
                        "avg_in_norm_ms": round(cpu_total_norm_ms / n_b, 2),
                        "avg_h2d_ms": 0.0,
                        "avg_trt_ms": round(cpu_total_infer_ms / n_b, 2),
                        "avg_triton_ms": round(cpu_total_conv_ms / n_b, 2),
                        "avg_d2h_ms": 0.0,
                    }
            ok = True
        finally:
            if ok:
                drain_worker.close()
            else:
                for evt in slot_drain_events:
                    evt.set()
                drain_worker.abort(encoder_proc)

        return total_frames_written

    def release(self) -> None:
        """Release pinned host buffers, VRAM ring buffers, and pooled CUDA events."""
        self.host_in_pins.clear()
        self.host_out_pins.clear()
        self.gpu_in_ring.clear()
        self.gpu_nhwc_ring.clear()
        if getattr(self, "color_conv", None) is not None:
            self.color_conv._p010_ring.clear()
        self._micro_event_pool.clear()
        self._sync_event_pool.clear()
        self.last_first_rgb_small = None
        self.input_hw = (0, 0)
        self.output_hw = (0, 0)


TRTZeroCopyUpscaler = DoubleBufferedPipeUpscaler
