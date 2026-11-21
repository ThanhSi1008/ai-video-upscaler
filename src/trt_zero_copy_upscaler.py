"""Zero-disk, zero-copy GPU inference pipeline using Triton P010LE conversion and 3-stream CUDA overlap."""

import collections
import os
import queue
import subprocess
import threading
import time
from typing import List, Tuple, Optional, Any, Dict, Sequence
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
    def _yuv420_to_nchw_fp16_bt709_kernel(
        yuv_ptr,
        out_ptr,
        stride_yuv_b, stride_yuv_h, stride_yuv_w,
        stride_out_b, stride_out_c, stride_out_h, stride_out_w,
        H, W,
        IS_P010: tl.constexpr,
        CHROMA_LEFT: tl.constexpr,
        BLOCK_X: tl.constexpr,
        BLOCK_Y: tl.constexpr,
    ):
        pid_x = tl.program_id(axis=0)
        pid_y = tl.program_id(axis=1)
        pid_b = tl.program_id(axis=2).to(tl.int64)

        H_uv = H // 2
        W_uv = W // 2

        offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
        offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
        mask = (offs_x[None, :] < W_uv) & (offs_y[:, None] < H_uv)

        yuv_b = yuv_ptr + pid_b * stride_yuv_b
        out_b = out_ptr + pid_b * stride_out_b

        y0 = offs_y[:, None] * 2
        y1 = y0 + 1
        x0 = offs_x[None, :] * 2
        x1 = x0 + 1

        raw_y00 = tl.load(yuv_b + y0 * stride_yuv_h + x0 * stride_yuv_w, mask=mask, other=0)
        raw_y01 = tl.load(yuv_b + y0 * stride_yuv_h + x1 * stride_yuv_w, mask=mask, other=0)
        raw_y10 = tl.load(yuv_b + y1 * stride_yuv_h + x0 * stride_yuv_w, mask=mask, other=0)
        raw_y11 = tl.load(yuv_b + y1 * stride_yuv_h + x1 * stride_yuv_w, mask=mask, other=0)

        cy = offs_y[:, None]
        cy_prev = tl.maximum(cy - 1, 0)
        cy_next = tl.minimum(cy + 1, H_uv - 1)
        cx_next = tl.minimum(offs_x[None, :] + 1, W_uv - 1)
        uv_r_prev = H + cy_prev
        uv_r_curr = H + cy
        uv_r_next = H + cy_next
        ux0 = x0
        vx0 = x1
        ux1 = cx_next * 2
        vx1 = ux1 + 1

        raw_uc0 = tl.load(yuv_b + uv_r_curr * stride_yuv_h + ux0 * stride_yuv_w, mask=mask, other=0)
        raw_vc0 = tl.load(yuv_b + uv_r_curr * stride_yuv_h + vx0 * stride_yuv_w, mask=mask, other=0)
        raw_up0 = tl.load(yuv_b + uv_r_prev * stride_yuv_h + ux0 * stride_yuv_w, mask=mask, other=0)
        raw_vp0 = tl.load(yuv_b + uv_r_prev * stride_yuv_h + vx0 * stride_yuv_w, mask=mask, other=0)
        raw_un0 = tl.load(yuv_b + uv_r_next * stride_yuv_h + ux0 * stride_yuv_w, mask=mask, other=0)
        raw_vn0 = tl.load(yuv_b + uv_r_next * stride_yuv_h + vx0 * stride_yuv_w, mask=mask, other=0)
        raw_uc1 = tl.load(yuv_b + uv_r_curr * stride_yuv_h + ux1 * stride_yuv_w, mask=mask, other=0)
        raw_vc1 = tl.load(yuv_b + uv_r_curr * stride_yuv_h + vx1 * stride_yuv_w, mask=mask, other=0)
        raw_up1 = tl.load(yuv_b + uv_r_prev * stride_yuv_h + ux1 * stride_yuv_w, mask=mask, other=0)
        raw_vp1 = tl.load(yuv_b + uv_r_prev * stride_yuv_h + vx1 * stride_yuv_w, mask=mask, other=0)
        raw_un1 = tl.load(yuv_b + uv_r_next * stride_yuv_h + ux1 * stride_yuv_w, mask=mask, other=0)
        raw_vn1 = tl.load(yuv_b + uv_r_next * stride_yuv_h + vx1 * stride_yuv_w, mask=mask, other=0)

        if IS_P010:
            inv_64 = 1.0 / 64.0
            y00_c = (raw_y00.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            y01_c = (raw_y01.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            y10_c = (raw_y10.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            y11_c = (raw_y11.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            uc0 = (raw_uc0.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            vc0 = (raw_vc0.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            up0 = (raw_up0.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            vp0 = (raw_vp0.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            un0 = (raw_un0.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            vn0 = (raw_vn0.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            uc1 = (raw_uc1.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            vc1 = (raw_vc1.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            up1 = (raw_up1.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            vp1 = (raw_vp1.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            un1 = (raw_un1.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            vn1 = (raw_vn1.to(tl.int32) & 0xFFFF).to(tl.float32) * inv_64
            y_off = 64.0
            y_inv = 1.0 / 876.0
            c_off = 512.0
            c_inv = 1.0 / 896.0
        else:
            y00_c = raw_y00.to(tl.float32)
            y01_c = raw_y01.to(tl.float32)
            y10_c = raw_y10.to(tl.float32)
            y11_c = raw_y11.to(tl.float32)
            uc0 = raw_uc0.to(tl.float32)
            vc0 = raw_vc0.to(tl.float32)
            up0 = raw_up0.to(tl.float32)
            vp0 = raw_vp0.to(tl.float32)
            un0 = raw_un0.to(tl.float32)
            vn0 = raw_vn0.to(tl.float32)
            uc1 = raw_uc1.to(tl.float32)
            vc1 = raw_vc1.to(tl.float32)
            up1 = raw_up1.to(tl.float32)
            vp1 = raw_vp1.to(tl.float32)
            un1 = raw_un1.to(tl.float32)
            vn1 = raw_vn1.to(tl.float32)
            y_off = 16.0
            y_inv = 1.0 / 219.0
            c_off = 128.0
            c_inv = 1.0 / 224.0

        # Bilinear MPEG-2 'left' (0.0, 0.5) chroma upsampling (or nearest box when CHROMA_LEFT=False)
        if CHROMA_LEFT:
            u_top0 = 0.75 * uc0 + 0.25 * up0
            v_top0 = 0.75 * vc0 + 0.25 * vp0
            u_bot0 = 0.75 * uc0 + 0.25 * un0
            v_bot0 = 0.75 * vc0 + 0.25 * vn0
            u_top1 = 0.75 * uc1 + 0.25 * up1
            v_top1 = 0.75 * vc1 + 0.25 * vp1
            u_bot1 = 0.75 * uc1 + 0.25 * un1
            v_bot1 = 0.75 * vc1 + 0.25 * vn1
            u00_c = u_top0
            v00_c = v_top0
            u01_c = 0.5 * (u_top0 + u_top1)
            v01_c = 0.5 * (v_top0 + v_top1)
            u10_c = u_bot0
            v10_c = v_bot0
            u11_c = 0.5 * (u_bot0 + u_bot1)
            v11_c = 0.5 * (v_bot0 + v_bot1)
        else:
            u00_c = uc0
            v00_c = vc0
            u01_c = uc0
            v01_c = vc0
            u10_c = uc0
            v10_c = vc0
            u11_c = uc0
            v11_c = vc0

        yn00 = (y00_c - y_off) * y_inv
        yn01 = (y01_c - y_off) * y_inv
        yn10 = (y10_c - y_off) * y_inv
        yn11 = (y11_c - y_off) * y_inv

        cb00 = (u00_c - c_off) * c_inv
        cr00 = (v00_c - c_off) * c_inv
        cb01 = (u01_c - c_off) * c_inv
        cr01 = (v01_c - c_off) * c_inv
        cb10 = (u10_c - c_off) * c_inv
        cr10 = (v10_c - c_off) * c_inv
        cb11 = (u11_c - c_off) * c_inv
        cr11 = (v11_c - c_off) * c_inv

        k_rv = 1.5748
        k_gu = -0.1873242729306487
        k_gv = -0.4681242729306487
        k_bu = 1.8556

        r00 = tl.minimum(tl.maximum(yn00 + k_rv * cr00, 0.0), 1.0).to(tl.float16)
        g00 = tl.minimum(tl.maximum(yn00 + k_gu * cb00 + k_gv * cr00, 0.0), 1.0).to(tl.float16)
        b00 = tl.minimum(tl.maximum(yn00 + k_bu * cb00, 0.0), 1.0).to(tl.float16)

        r01 = tl.minimum(tl.maximum(yn01 + k_rv * cr01, 0.0), 1.0).to(tl.float16)
        g01 = tl.minimum(tl.maximum(yn01 + k_gu * cb01 + k_gv * cr01, 0.0), 1.0).to(tl.float16)
        b01 = tl.minimum(tl.maximum(yn01 + k_bu * cb01, 0.0), 1.0).to(tl.float16)

        r10 = tl.minimum(tl.maximum(yn10 + k_rv * cr10, 0.0), 1.0).to(tl.float16)
        g10 = tl.minimum(tl.maximum(yn10 + k_gu * cb10 + k_gv * cr10, 0.0), 1.0).to(tl.float16)
        b10 = tl.minimum(tl.maximum(yn10 + k_bu * cb10, 0.0), 1.0).to(tl.float16)

        r11 = tl.minimum(tl.maximum(yn11 + k_rv * cr11, 0.0), 1.0).to(tl.float16)
        g11 = tl.minimum(tl.maximum(yn11 + k_gu * cb11 + k_gv * cr11, 0.0), 1.0).to(tl.float16)
        b11 = tl.minimum(tl.maximum(yn11 + k_bu * cb11, 0.0), 1.0).to(tl.float16)

        tl.store(out_b + 0 * stride_out_c + y0 * stride_out_h + x0 * stride_out_w, r00, mask=mask)
        tl.store(out_b + 0 * stride_out_c + y0 * stride_out_h + x1 * stride_out_w, r01, mask=mask)
        tl.store(out_b + 0 * stride_out_c + y1 * stride_out_h + x0 * stride_out_w, r10, mask=mask)
        tl.store(out_b + 0 * stride_out_c + y1 * stride_out_h + x1 * stride_out_w, r11, mask=mask)

        tl.store(out_b + 1 * stride_out_c + y0 * stride_out_h + x0 * stride_out_w, g00, mask=mask)
        tl.store(out_b + 1 * stride_out_c + y0 * stride_out_h + x1 * stride_out_w, g01, mask=mask)
        tl.store(out_b + 1 * stride_out_c + y1 * stride_out_h + x0 * stride_out_w, g10, mask=mask)
        tl.store(out_b + 1 * stride_out_c + y1 * stride_out_h + x1 * stride_out_w, g11, mask=mask)

        tl.store(out_b + 2 * stride_out_c + y0 * stride_out_h + x0 * stride_out_w, b00, mask=mask)
        tl.store(out_b + 2 * stride_out_c + y0 * stride_out_h + x1 * stride_out_w, b01, mask=mask)
        tl.store(out_b + 2 * stride_out_c + y1 * stride_out_h + x0 * stride_out_w, b10, mask=mask)
        tl.store(out_b + 2 * stride_out_c + y1 * stride_out_h + x1 * stride_out_w, b11, mask=mask)

    @triton.jit
    def _rgb_to_p010le_triton_kernel(
        rgb_ptr,
        out_ptr,
        stride_rgb_b, stride_rgb_c, stride_rgb_h, stride_rgb_w,
        stride_out_b, stride_out_h, stride_out_w,
        H, W,
        BLOCK_X: tl.constexpr,
        BLOCK_Y: tl.constexpr,
        CHROMA_LEFT: tl.constexpr = False,
        USE_DITHER: tl.constexpr = False,
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

        # Sanitize IEEE-754 NaNs before [0, 1] gamut clamping
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

        if USE_DITHER:
            xf0 = x0.to(tl.float32)
            xf1 = x1.to(tl.float32)
            yf0 = y0.to(tl.float32)
            yf1 = y1.to(tl.float32)
            p00 = 0.06711056 * xf0 + 0.00583715 * yf0
            p01 = 0.06711056 * xf1 + 0.00583715 * yf0
            p10 = 0.06711056 * xf0 + 0.00583715 * yf1
            p11 = 0.06711056 * xf1 + 0.00583715 * yf1
            q00 = 52.9829189 * (p00 - tl.extra.cuda.libdevice.floor(p00))
            q01 = 52.9829189 * (p01 - tl.extra.cuda.libdevice.floor(p01))
            q10 = 52.9829189 * (p10 - tl.extra.cuda.libdevice.floor(p10))
            q11 = 52.9829189 * (p11 - tl.extra.cuda.libdevice.floor(p11))
            d00 = ((q00 - tl.extra.cuda.libdevice.floor(q00)) - 0.5) * 0.998
            d01 = ((q01 - tl.extra.cuda.libdevice.floor(q01)) - 0.5) * 0.998
            d10 = ((q10 - tl.extra.cuda.libdevice.floor(q10)) - 0.5) * 0.998
            d11 = ((q11 - tl.extra.cuda.libdevice.floor(q11)) - 0.5) * 0.998
            y_val00 = y_val00 + d00
            y_val01 = y_val01 + d01
            y_val10 = y_val10 + d10
            y_val11 = y_val11 + d11

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

        # 2. Chroma Subsampling ('left' (0.0, 0.5) [1,2,1]/4 x [1,1]/2 vs 'center' 2x2 box)
        if CHROMA_LEFT:
            xm1 = tl.maximum(x0 - 1, 0)
            r0m = tl.load(rgb_b + 0 * stride_rgb_c + y0 * stride_rgb_h + xm1 * stride_rgb_w, mask=mask).to(tl.float32)
            r1m = tl.load(rgb_b + 0 * stride_rgb_c + y1 * stride_rgb_h + xm1 * stride_rgb_w, mask=mask).to(tl.float32)
            g0m = tl.load(rgb_b + 1 * stride_rgb_c + y0 * stride_rgb_h + xm1 * stride_rgb_w, mask=mask).to(tl.float32)
            g1m = tl.load(rgb_b + 1 * stride_rgb_c + y1 * stride_rgb_h + xm1 * stride_rgb_w, mask=mask).to(tl.float32)
            b0m = tl.load(rgb_b + 2 * stride_rgb_c + y0 * stride_rgb_h + xm1 * stride_rgb_w, mask=mask).to(tl.float32)
            b1m = tl.load(rgb_b + 2 * stride_rgb_c + y1 * stride_rgb_h + xm1 * stride_rgb_w, mask=mask).to(tl.float32)
            r0m = tl.minimum(tl.maximum(tl.where(r0m == r0m, r0m, 0.0), 0.0), 1.0)
            r1m = tl.minimum(tl.maximum(tl.where(r1m == r1m, r1m, 0.0), 0.0), 1.0)
            g0m = tl.minimum(tl.maximum(tl.where(g0m == g0m, g0m, 0.0), 0.0), 1.0)
            g1m = tl.minimum(tl.maximum(tl.where(g1m == g1m, g1m, 0.0), 0.0), 1.0)
            b0m = tl.minimum(tl.maximum(tl.where(b0m == b0m, b0m, 0.0), 0.0), 1.0)
            b1m = tl.minimum(tl.maximum(tl.where(b1m == b1m, b1m, 0.0), 0.0), 1.0)
            rs = 0.125 * (r0m + r1m) + 0.25 * (r00 + r10) + 0.125 * (r01 + r11)
            gs = 0.125 * (g0m + g1m) + 0.25 * (g00 + g10) + 0.125 * (g01 + g11)
            bs = 0.125 * (b0m + b1m) + 0.25 * (b00 + b10) + 0.125 * (b01 + b11)
        else:
            rs = 0.25 * (r00 + r01 + r10 + r11)
            gs = 0.25 * (g00 + g01 + g10 + g11)
            bs = 0.25 * (b00 + b01 + b10 + b11)

        u_val = 512.0 - 102.656512 * rs - 345.343488 * gs + 448.0 * bs
        v_val = 512.0 + 448.0 * rs - 406.921088 * gs - 41.078912 * bs

        if USE_DITHER:
            pc = 0.06711056 * offs_x[None, :].to(tl.float32) + 0.00583715 * offs_y[:, None].to(tl.float32)
            qc = 52.9829189 * (pc - tl.extra.cuda.libdevice.floor(pc))
            dc = ((qc - tl.extra.cuda.libdevice.floor(qc)) - 0.5) * 0.998
            u_val = u_val + dc
            v_val = v_val - dc

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
    _yuv420_to_nchw_fp16_bt709_kernel = None
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


def _yuv420_to_nchw_fp16_bt709(
    yuv_semi: torch.Tensor,
    is_p010: bool,
    out_nchw_fp16: Optional[torch.Tensor] = None,
    chroma_siting: str = "left",
    out_dtype: torch.dtype = torch.float16,
    use_triton: Optional[bool] = None,
) -> torch.Tensor:
    """Convert semi-planar P010LE (int16) or NV12 (uint8) [B, 3*H//2, W] into BT.709 RGB [B, 3, H, W]."""
    if yuv_semi.ndim != 3 or (yuv_semi.shape[1] % 3) != 0:
        raise ValueError(f"Expected semi-planar YUV420 (B, 3*H//2, W), got shape={tuple(yuv_semi.shape)}")

    B, semi_h, W = yuv_semi.shape
    H = (semi_h * 2) // 3
    if out_nchw_fp16 is None:
        out_buf = torch.empty((B, 3, H, W), dtype=out_dtype, device=yuv_semi.device)
    else:
        out_buf = out_nchw_fp16

    if B == 0 or H == 0 or W == 0:
        return out_buf

    is_left = str(chroma_siting or "left").strip().lower() == "left"
    should_use_triton = (
        bool(use_triton)
        if use_triton is not None
        else (_HAS_TRITON and yuv_semi.is_cuda and out_buf.is_cuda and out_buf.dtype == torch.float16)
    )
    if (
        should_use_triton
        and _HAS_TRITON
        and _yuv420_to_nchw_fp16_bt709_kernel is not None
        and out_buf.dtype == torch.float16
    ):
        BLOCK_X = 16
        BLOCK_Y = 16
        grid = (triton.cdiv(W // 2, BLOCK_X), triton.cdiv(H // 2, BLOCK_Y), B)
        _yuv420_to_nchw_fp16_bt709_kernel[grid](
            yuv_semi,
            out_buf,
            yuv_semi.stride(0), yuv_semi.stride(1), yuv_semi.stride(2),
            out_buf.stride(0), out_buf.stride(1), out_buf.stride(2), out_buf.stride(3),
            H, W,
            IS_P010=bool(is_p010),
            CHROMA_LEFT=is_left,
            BLOCK_X=BLOCK_X,
            BLOCK_Y=BLOCK_Y,
        )
        return out_buf

    if is_p010:
        f_all = (yuv_semi.to(torch.int32) & 0xFFFF).to(torch.float32) * (1.0 / 64.0)
        y_off, y_inv = 64.0, 1.0 / 876.0
        c_off, c_inv = 512.0, 1.0 / 896.0
    else:
        f_all = yuv_semi.to(torch.float32)
        y_off, y_inv = 16.0, 1.0 / 219.0
        c_off, c_inv = 128.0, 1.0 / 224.0

    y_plane = (f_all[:, :H, :] - y_off) * y_inv
    u_sub = f_all[:, H:, 0::2]
    v_sub = f_all[:, H:, 1::2]

    if is_left:
        u_prev = torch.cat((u_sub[:, :1, :], u_sub[:, :-1, :]), dim=1)
        u_next_y = torch.cat((u_sub[:, 1:, :], u_sub[:, -1:, :]), dim=1)
        v_prev = torch.cat((v_sub[:, :1, :], v_sub[:, :-1, :]), dim=1)
        v_next_y = torch.cat((v_sub[:, 1:, :], v_sub[:, -1:, :]), dim=1)

        u_top = 0.75 * u_sub + 0.25 * u_prev
        u_bot = 0.75 * u_sub + 0.25 * u_next_y
        v_top = 0.75 * v_sub + 0.25 * v_prev
        v_bot = 0.75 * v_sub + 0.25 * v_next_y

        u_top_r = torch.cat((u_top[:, :, 1:], u_top[:, :, -1:]), dim=2)
        u_bot_r = torch.cat((u_bot[:, :, 1:], u_bot[:, :, -1:]), dim=2)
        v_top_r = torch.cat((v_top[:, :, 1:], v_top[:, :, -1:]), dim=2)
        v_bot_r = torch.cat((v_bot[:, :, 1:], v_bot[:, :, -1:]), dim=2)

        u_full = torch.empty((B, H, W), dtype=torch.float32, device=yuv_semi.device)
        v_full = torch.empty((B, H, W), dtype=torch.float32, device=yuv_semi.device)
        u_full[:, 0::2, 0::2] = u_top
        u_full[:, 0::2, 1::2] = 0.5 * (u_top + u_top_r)
        u_full[:, 1::2, 0::2] = u_bot
        u_full[:, 1::2, 1::2] = 0.5 * (u_bot + u_bot_r)
        v_full[:, 0::2, 0::2] = v_top
        v_full[:, 0::2, 1::2] = 0.5 * (v_top + v_top_r)
        v_full[:, 1::2, 0::2] = v_bot
        v_full[:, 1::2, 1::2] = 0.5 * (v_bot + v_bot_r)
    else:
        u_full = u_sub.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        v_full = v_sub.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    cb = (u_full - c_off) * c_inv
    cr = (v_full - c_off) * c_inv
    r = torch.clamp(y_plane + 1.5748 * cr, 0.0, 1.0)
    g = torch.clamp(y_plane - 0.1873242729306487 * cb - 0.4681242729306487 * cr, 0.0, 1.0)
    b = torch.clamp(y_plane + 1.8556 * cb, 0.0, 1.0)

    out_buf[:, 0].copy_(r.to(out_buf.dtype))
    out_buf[:, 1].copy_(g.to(out_buf.dtype))
    out_buf[:, 2].copy_(b.to(out_buf.dtype))
    return out_buf


# 1. GPU COLOR CONVERTER P010LE (BT.709 10-BIT LIMITED RANGE)

class GPUColorConverterP010LE:
    """Convert RGB [B, 3, H, W] tensors in [0.0, 1.0] to ITU-R BT.709 limited-range P010LE."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        chroma_siting: str = "center",
        dither: bool = False,
    ):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.chroma_siting = str(chroma_siting or "center").strip().lower()
        self.dither = bool(dither)
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
    def _ign_dither_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
        yy = torch.arange(H, dtype=torch.float32, device=device)[:, None]
        xx = torch.arange(W, dtype=torch.float32, device=device)[None, :]
        p = 0.06711056 * xx + 0.00583715 * yy
        q = 52.9829189 * (p - torch.floor(p))
        return ((q - torch.floor(q)) - 0.5) * 0.998

    @staticmethod
    def _p010le_core(
        rgb_slice: torch.Tensor,
        chroma_siting: str = "center",
        dither: bool = False,
    ) -> torch.Tensor:
        B, _, H, W = rgb_slice.shape
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
        if dither:
            y = y + GPUColorConverterP010LE._ign_dither_grid(H, W, rgb_slice.device).unsqueeze(0)
        y_p010 = (torch.clamp(torch.round(y), 64.0, 940.0).to(torch.int32) << 6).to(torch.int16)

        # 2. 4:2:0 chroma subsampling ('left' (0.0, 0.5) [1,2,1]/4 x [1,1]/2 vs 'center' 2x2 box)
        if str(chroma_siting or "center").strip().lower() == "left":
            vert_avg = 0.5 * (rgb_clamped[:, :, 0::2, :] + rgb_clamped[:, :, 1::2, :])
            mid = vert_avg[:, :, :, 0::2]
            right = vert_avg[:, :, :, 1::2]
            left_pad = torch.cat((vert_avg[:, :, :, :1], vert_avg[:, :, :, 1:-1:2]), dim=3)
            rgb_sub = 0.25 * left_pad + 0.50 * mid + 0.25 * right
        else:
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

        if dither:
            dc = GPUColorConverterP010LE._ign_dither_grid(H // 2, W // 2, rgb_slice.device).unsqueeze(0)
            u = u + dc
            v = v - dc

        u_p010 = (torch.clamp(torch.round(u), 64.0, 960.0).to(torch.int32) << 6).to(torch.int16)
        v_p010 = (torch.clamp(torch.round(v), 64.0, 960.0).to(torch.int32) << 6).to(torch.int16)

        # 3. Interleave semi-planar UV [B, H/2, W] after Y [B, H, W]
        uv_p010 = torch.stack((u_p010, v_p010), dim=-1).view(B, H // 2, W)
        return torch.cat((y_p010, uv_p010), dim=1).contiguous()

    @torch.inference_mode()
    def _convert_slice(self, rgb_slice: torch.Tensor) -> torch.Tensor:
        return self._p010le_core(rgb_slice, chroma_siting=self.chroma_siting, dither=self.dither)

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
                BLOCK_X=BLOCK_X, BLOCK_Y=BLOCK_Y,
                CHROMA_LEFT=(self.chroma_siting == "left"),
                USE_DITHER=self.dither,
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
    """Hybrid-precision wrapper collapsing SRVGG 4x->2x via Catmull-Rom AA (zero 8K intermediate tensors)."""
    def __init__(self, base_srvgg_model: nn.Module, scale: Optional[float] = None):
        super().__init__()
        self.body = base_srvgg_model.body
        self.upsampler = base_srvgg_model.upsampler
        if hasattr(base_srvgg_model, "upscale"):
            self.upscale = int(base_srvgg_model.upscale)
        else:
            self.upscale = 4 if isinstance(self.upsampler, nn.PixelShuffle) and self.upsampler.upscale_factor == 4 else 2
        self.target_scale = float(scale) if scale is not None else 2.0
        self._use_collapsed_aa2x = (
            self.upscale == 4
            and abs(self.target_scale - 2.0) <= 1e-4
            and isinstance(self.upsampler, nn.PixelShuffle)
            and int(self.upsampler.upscale_factor) == 4
        )
        if self._use_collapsed_aa2x:
            from src.cuda_engine import build_collapsed_4x_to_2x_buffers
            fw, bw = build_collapsed_4x_to_2x_buffers(b_param=0.0, c_param=0.5)
            ref_p = next(base_srvgg_model.parameters(), None)
            if ref_p is not None:
                fw = fw.to(device=ref_p.device, dtype=ref_p.dtype)
                bw = bw.to(device=ref_p.device, dtype=ref_p.dtype)
            self.register_buffer("aa_feat_weight", fw, persistent=False)
            self.register_buffer("aa_base_weight", bw, persistent=False)

    def _tail_upsample(self, feat_slice: torch.Tensor, x_slice: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        if self._use_collapsed_aa2x:
            from src.cuda_engine import collapsed_4x_to_2x_forward
            return collapsed_4x_to_2x_forward(
                feat_slice.float(),
                x_slice.float(),
                self.aa_feat_weight.float(),
                self.aa_base_weight.float(),
            ).to(out_dtype)

        out = self.upsampler(feat_slice)
        base = F.interpolate(x_slice.float(), scale_factor=float(self.upscale), mode="nearest")
        out_f32 = out.float() + base

        if abs(float(self.upscale) - self.target_scale) > 1e-4:
            scale_ratio = self.target_scale / float(self.upscale)
            out_target = F.interpolate(
                out_f32,
                scale_factor=scale_ratio,
                mode="bicubic",
                align_corners=False,
                antialias=(scale_ratio < 1.0),
            )
        else:
            out_target = out_f32
        return out_target.to(out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = torch.float16 if x.dtype == torch.float16 else torch.float32
        feat = x
        for layer in self.body:
            feat = layer(feat)

        if self._use_collapsed_aa2x:
            from src.cuda_engine import collapsed_4x_to_2x_forward
            return collapsed_4x_to_2x_forward(feat, x, self.aa_feat_weight, self.aa_base_weight).to(out_dtype)

        B, _, H, W = x.shape
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


def _infer_frame_luma_hw(arr: np.ndarray) -> Tuple[int, int]:
    """Infer (H, W) luma dimensions from 2D semi-planar YUV420, NHWC, or NCHW frame arrays."""
    if arr.ndim == 2:
        return (int(arr.shape[0]) * 2) // 3, int(arr.shape[1])
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        return int(arr.shape[1]), int(arr.shape[2])
    return int(arr.shape[0]), int(arr.shape[1])


def _bound_live_cyclic_memo(
    distinct_frames: Sequence[np.ndarray],
    frame_to_distinct: Sequence[int],
    max_memo_frames: int = 64,
) -> Tuple[List[np.ndarray], List[int]]:
    """Bound concurrent live cyclic memo entries (<= max_memo_frames) via Belady's OPT with monotonic IDs."""
    cap = max(1, int(max_memo_frames))
    num_u = len(frame_to_distinct)
    if num_u <= 1 or len(distinct_frames) <= cap:
        return list(distinct_frames), list(frame_to_distinct)

    occ_deques: dict[int, collections.deque[int]] = {}
    for pos, d_idx in enumerate(frame_to_distinct):
        occ_deques.setdefault(int(d_idx), collections.deque()).append(pos)

    if sum(1 for dq in occ_deques.values() if len(dq) > 1) <= cap:
        return list(distinct_frames), list(frame_to_distinct)

    live_tokens: dict[int, int] = {}
    active_token_for_orig: dict[int, int] = {}
    token_source_frame: dict[int, np.ndarray] = {}
    raw_tokens: List[int] = []
    next_tok = 0

    for u_idx, raw_d in enumerate(frame_to_distinct):
        orig_d = int(raw_d)
        dq = occ_deques[orig_d]
        dq.popleft()

        if orig_d in active_token_for_orig:
            tok = active_token_for_orig[orig_d]
        else:
            tok = next_tok
            next_tok += 1
            active_token_for_orig[orig_d] = tok
            token_source_frame[tok] = distinct_frames[orig_d]

        raw_tokens.append(tok)

        if not dq:
            live_tokens.pop(orig_d, None)
            active_token_for_orig.pop(orig_d, None)
        else:
            next_use = dq[0]
            if orig_d in live_tokens:
                live_tokens[orig_d] = next_use
            elif len(live_tokens) < cap:
                live_tokens[orig_d] = next_use
            else:
                farthest_orig, farthest_pos = max(live_tokens.items(), key=lambda kv: kv[1])
                if next_use >= farthest_pos:
                    # Current frame's next use is farthest: split its future occurrences instead of caching now
                    active_token_for_orig.pop(orig_d, None)
                else:
                    # Evict farthest live entry and split its remaining future occurrences
                    live_tokens.pop(farthest_orig, None)
                    active_token_for_orig.pop(farthest_orig, None)
                    live_tokens[orig_d] = next_use

    # Re-index tokens in strict first-appearance order so batching & _drain_ready_frames stay monotonic
    tok_to_new_d: dict[int, int] = {}
    new_distinct: List[np.ndarray] = []
    new_mapping: List[int] = []
    for tok in raw_tokens:
        nd = tok_to_new_d.get(tok)
        if nd is None:
            nd = len(new_distinct)
            tok_to_new_d[tok] = nd
            new_distinct.append(token_source_frame[tok])
        new_mapping.append(nd)
    return new_distinct, new_mapping


class _NoOpStdin:
    def close(self) -> None:
        pass


class _InProcessPyAVNVENCEncoder:
    """In-process PyAV hevc_nvenc p010le MP4 writer eliminating OS pipe writes and ffmpeg.exe spawns."""

    def __init__(
        self,
        output_mp4_path: str,
        fps_str: str = "24000/1001",
        width: int = 3840,
        height: int = 2160,
        chroma_location: str = "left",
        multipass: Optional[str] = None,
    ):
        import fractions
        import av

        if HEVC_ENCODER != "hevc_nvenc":
            raise ValueError(f"In-process PyAV encoder requires hevc_nvenc, got {HEVC_ENCODER}")
        if "hevc_nvenc" not in getattr(av, "codecs_available", set()):
            raise RuntimeError("hevc_nvenc unavailable in PyAV build")

        if "/" in str(fps_str):
            num_s, den_s = str(fps_str).split("/", 1)
            fps_num = max(1, int(round(float(num_s))))
            fps_den = max(1, int(round(float(den_s))))
        else:
            fps_f = float(fps_str)
            if abs(fps_f - 23.976) < 0.02:
                fps_num, fps_den = 24000, 1001
            else:
                fps_num, fps_den = max(1, int(round(fps_f * 1000))), 1000

        self.output_mp4_path = output_mp4_path
        self.width = int(width)
        self.height = int(height)
        self._fps_frac = fractions.Fraction(fps_num, fps_den)
        self._time_base = fractions.Fraction(fps_den, fps_num)
        self._pts_counter = 0
        self._closed = False
        self._aborted = False
        self._rc: Optional[int] = None
        self.stdin = _NoOpStdin()
        self.stderr = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=16)

        chroma_loc_norm = "left" if str(chroma_location or "left").strip().lower() == "left" else "center"
        rate_args = get_hevc_rate_args(bitrate="28M", multipass=multipass)
        parsed_opts: dict[str, str] = {
            "profile": str(HEVC_PROFILE),
            "chroma_sample_location": chroma_loc_norm,
        }
        idx = 0
        while idx < len(rate_args):
            tok = str(rate_args[idx])
            if tok.startswith("-") and (idx + 1) < len(rate_args):
                opt_key = tok.lstrip("-")
                opt_val = str(rate_args[idx + 1])
                if opt_key not in ("b:v",):
                    parsed_opts[opt_key] = opt_val
                idx += 2
            else:
                idx += 1

        self._av = av
        self._container = av.open(
            output_mp4_path,
            mode="w",
            options={"video_track_timescale": str(fps_num)},
        )
        try:
            stream = self._container.add_stream("hevc_nvenc", rate=self._fps_frac)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = HEVC_PIX_FMT
            stream.time_base = self._time_base
            # Explicitly zero bit_rate on both stream and codec_context so libavcodec never defaults to 2 Mbps
            stream.bit_rate = 0
            stream.codec_context.bit_rate = 0
            stream.codec_context.color_range = 1
            stream.codec_context.colorspace = 1
            stream.codec_context.color_primaries = 1
            stream.codec_context.color_trc = 1
            stream.codec_context.codec_tag = HEVC_TAG
            stream.options = dict(parsed_opts)
            stream.codec_context.open()
            if stream.options:
                raise ValueError(f"Unconsumed PyAV hevc_nvenc options: {stream.options}")
            self._stream = stream
        except Exception:
            try:
                self._container.close()
            except Exception:
                pass
            if os.path.exists(output_mp4_path):
                try:
                    os.remove(output_mp4_path)
                except OSError:
                    pass
            raise

    def encode_p010le_frame(self, buf_arr: np.ndarray, repeats: int = 1) -> None:
        """Encode semi-planar P010LE frame with distinct AVFrame instances per repeat (safe for rc-lookahead=32)."""
        if self._closed or self._aborted or repeats <= 0:
            return
        H, W = self.height, self.width
        row_bytes = W * 2
        y_mv = memoryview(buf_arr[:H, :]).cast("B")
        uv_mv = memoryview(buf_arr[H:, :]).cast("B")
        av_mod = self._av
        for _ in range(int(repeats)):
            fr = av_mod.VideoFrame(W, H, "p010le")
            p0 = fr.planes[0]
            p1 = fr.planes[1]
            if p0.line_size == row_bytes:
                p0.update(y_mv)
            else:
                dst0 = np.frombuffer(p0, dtype=np.uint8).reshape(H, p0.line_size)
                dst0[:, :row_bytes] = np.frombuffer(y_mv, dtype=np.uint8).reshape(H, row_bytes)
            if p1.line_size == row_bytes:
                p1.update(uv_mv)
            else:
                dst1 = np.frombuffer(p1, dtype=np.uint8).reshape(H // 2, p1.line_size)
                dst1[:, :row_bytes] = np.frombuffer(uv_mv, dtype=np.uint8).reshape(H // 2, row_bytes)
            # Assign strictly monotonic PTS on a fresh AVFrame (never mutate an in-flight lookahead AVFrame)
            fr.pts = self._pts_counter
            self._pts_counter += 1
            fr.time_base = self._time_base
            fr.color_range = 1
            fr.colorspace = 1
            fr.color_primaries = 1
            fr.color_trc = 1
            for pkt in self._stream.encode(fr):
                self._container.mux(pkt)

    def poll(self) -> Optional[int]:
        return self._rc

    def wait(self, timeout: Optional[float] = None) -> int:
        if self._closed:
            return 0 if self._rc is None else self._rc
        self._closed = True
        try:
            if not self._aborted:
                for pkt in self._stream.encode(None):
                    self._container.mux(pkt)
            self._container.close()
            self._rc = 1 if self._aborted else 0
        except Exception as exc:
            self._stderr_tail.append(str(exc))
            try:
                self._container.close()
            except Exception:
                pass
            self._rc = 1
        return self._rc

    def terminate(self) -> None:
        self._aborted = True
        if not self._closed:
            self._closed = True
            self._rc = 1
            try:
                self._container.close()
            except Exception:
                pass

    def kill(self) -> None:
        self.terminate()


class DoubleBufferedPipeUpscaler:
    """Three-stream overlapped H2D/Compute/D2H upscaler writing to in-process PyAV or FFmpeg NVENC."""
    def __init__(
        self,
        model_or_engine: Any,
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        input_hw: Tuple[int, int] = (1080, 1920),
        scale: int = 2,
        enable_micro_timings: bool = True,
        chroma_siting: str = "left",
        dither: bool = True,
        max_memo_frames: int = 64,
    ):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.batch_size = max(1, batch_size)
        self.model = model_or_engine
        self.scale = max(1, int(scale))
        self.chroma_siting = str(chroma_siting or "left").strip().lower()
        self.dither = bool(dither)
        self.max_memo_frames = max(1, int(max_memo_frames))
        self.color_conv = GPUColorConverterP010LE(
            self.device,
            chroma_siting=self.chroma_siting,
            dither=self.dither,
        )
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
        self.host_yuv_pins: List[torch.Tensor] = []
        self.gpu_in_ring: List[torch.Tensor] = []
        self.gpu_nhwc_ring: List[torch.Tensor] = []
        self.gpu_yuv_ring: List[torch.Tensor] = []
        self._yuv_ring_dtype: Optional[torch.dtype] = None
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
        self.host_yuv_pins = []
        self.gpu_yuv_ring = []
        self._yuv_ring_dtype = None
        return self.output_hw

    def _ensure_yuv_rings(self, in_h: int, in_w: int, target_dtype: torch.dtype) -> None:
        semi_h = (int(in_h) * 3) // 2
        if (
            len(self.host_yuv_pins) == 2
            and self._yuv_ring_dtype == target_dtype
            and self.host_yuv_pins[0].shape == (self.batch_size, semi_h, int(in_w))
        ):
            return
        pin = self.device.type == "cuda"
        self.host_yuv_pins = [
            torch.empty((self.batch_size, semi_h, int(in_w)), dtype=target_dtype, pin_memory=pin)
            for _ in range(2)
        ]
        if self.device.type == "cuda":
            self.gpu_yuv_ring = [
                torch.empty((self.batch_size, semi_h, int(in_w)), dtype=target_dtype, device=self.device)
                for _ in range(2)
            ]
        else:
            self.gpu_yuv_ring = []
        self._yuv_ring_dtype = target_dtype

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
                semi_h = (in_h * 3) // 2
                dummy_p010 = torch.zeros((cur_bs, semi_h, in_w), dtype=torch.int16, device=self.device)
                dummy_nv12 = torch.zeros((cur_bs, semi_h, in_w), dtype=torch.uint8, device=self.device)
                _yuv420_to_nchw_fp16_bt709(
                    dummy_p010, is_p010=True, out_nchw_fp16=dummy_in_buf, chroma_siting=self.chroma_siting
                )
                _yuv420_to_nchw_fp16_bt709(
                    dummy_nv12, is_p010=False, out_nchw_fp16=dummy_in_buf, chroma_siting=self.chroma_siting
                )
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
        chroma_location: str = "center",
        multipass: Optional[str] = None,
    ) -> subprocess.Popen:
        """Spawn FFmpeg NVENC receiving rawvideo P010LE over stdin pipe."""
        rate_args = get_hevc_rate_args(bitrate="28M", multipass=multipass) if multipass else get_hevc_rate_args(bitrate="28M")
        ts_args: List[str] = []
        if "/" in str(fps_str):
            num_part = str(fps_str).split("/", 1)[0].strip()
            if num_part.isdigit() and int(num_part) > 0:
                ts_args = ["-video_track_timescale", num_part]
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "p010le",
            "-s", f"{width}x{height}",
            "-color_range", HEVC_COLOR_RANGE,
            "-colorspace", HEVC_COLORSPACE,
            "-color_primaries", HEVC_COLOR_PRIMARIES,
            "-color_trc", HEVC_COLOR_TRC,
            "-chroma_sample_location", str(chroma_location or "center"),
            "-r", fps_str,
            "-i", "pipe:0",
            "-c:v", HEVC_ENCODER,
            "-profile:v", HEVC_PROFILE,
            "-pix_fmt", HEVC_PIX_FMT,
            *rate_args,
            *ts_args,
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

    def open_chunk_encoder(
        self,
        output_mp4_path: str,
        fps_str: str = "24000/1001",
        width: int = 3840,
        height: int = 2160,
        multipass: Optional[str] = None,
        prefer_pyav: Optional[bool] = None,
    ) -> Any:
        """Open _InProcessPyAVNVENCEncoder when safe, falling back cleanly to spawn_nvenc_pipe_writer."""
        is_popen_mocked = hasattr(subprocess.Popen, "_mock_name")
        in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        allow_pyav = (
            bool(prefer_pyav)
            if prefer_pyav is not None
            else (HEVC_ENCODER == "hevc_nvenc" and not is_popen_mocked and not in_pytest)
        )
        if allow_pyav and HEVC_ENCODER == "hevc_nvenc" and not is_popen_mocked:
            try:
                return _InProcessPyAVNVENCEncoder(
                    output_mp4_path=output_mp4_path,
                    fps_str=fps_str,
                    width=width,
                    height=height,
                    chroma_location=self.chroma_siting,
                    multipass=multipass,
                )
            except Exception:
                pass
        return self.spawn_nvenc_pipe_writer(
            output_mp4_path,
            fps_str=fps_str,
            width=width,
            height=height,
            chroma_location=self.chroma_siting,
            multipass=multipass,
        )

    @staticmethod
    def _write_frame_to_pipe(encoder_proc: Any, buf_arr: np.ndarray, repeats: int = 1) -> None:
        """Dispatch repeated P010LE frames to _InProcessPyAVNVENCEncoder or 8 MiB stdin pipe slices."""
        if repeats <= 0:
            return
        if isinstance(encoder_proc, _InProcessPyAVNVENCEncoder):
            encoder_proc.encode_p010le_frame(buf_arr, repeats)
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

        f_h, f_w = _infer_frame_luma_hw(unique_frames_rgb[0])
        self.ensure_resolution(f_h, f_w)

        if precomputed_dedup is not None and precomputed_dedup[0] is not None and precomputed_dedup[1] is not None:
            distinct_frames, frame_to_distinct = precomputed_dedup
        elif cyclic_dedup and num_unique > 1:
            from src.dedup import deduplicate_cyclic_frames
            distinct_frames, frame_to_distinct = deduplicate_cyclic_frames(unique_frames_rgb)
        else:
            distinct_frames = unique_frames_rgb
            frame_to_distinct = list(range(num_unique))

        distinct_frames, frame_to_distinct = _bound_live_cyclic_memo(
            distinct_frames,
            frame_to_distinct,
            max_memo_frames=self.max_memo_frames,
        )

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
                        is_yuv = first_arr.ndim == 2
                        is_hwc = first_arr.ndim == 3 and first_arr.shape[-1] == 3
                        is_trt = bool(getattr(self.model, "is_tensorrt", False))
                        if is_yuv:
                            is_p010 = first_arr.dtype in (np.uint16, np.int16)
                            t_dtype = torch.int16 if is_p010 else torch.uint8
                            self._ensure_yuv_rings(f_h, f_w, t_dtype)
                            yuv_pin = self.host_yuv_pins[slot]
                            for i in range(cur_B):
                                arr_i = distinct_frames[s_idx + i]
                                src_view = arr_i.view(np.int16) if is_p010 else arr_i
                                yuv_pin[i].copy_(torch.from_numpy(src_view))
                            if self.gpu_yuv_ring:
                                gpu_yuv = self.gpu_yuv_ring[slot][:cur_B]
                                gpu_yuv.copy_(yuv_pin[:cur_B], non_blocking=True)
                            else:
                                gpu_yuv = yuv_pin[:cur_B].to(self.device, non_blocking=True)
                            if record_micro and evt_h2d_dma_e is not None:
                                evt_h2d_dma_e.record(self.stream_h2d)
                            if self.gpu_in_ring:
                                in_buf = self.gpu_in_ring[slot][:cur_B]
                                gpu_tensor = _yuv420_to_nchw_fp16_bt709(
                                    gpu_yuv,
                                    is_p010=is_p010,
                                    out_nchw_fp16=in_buf,
                                    chroma_siting=self.chroma_siting,
                                )
                            else:
                                gpu_tensor = _yuv420_to_nchw_fp16_bt709(
                                    gpu_yuv,
                                    is_p010=is_p010,
                                    chroma_siting=self.chroma_siting,
                                )
                            if not is_trt:
                                gpu_tensor = gpu_tensor.to(memory_format=torch.channels_last)
                        elif is_hwc:
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
                for batch_idx, start_idx in enumerate(range(0, num_distinct, self.batch_size)):
                    slot = batch_idx % 2
                    end_idx = min(start_idx + self.batch_size, num_distinct)
                    cur_B = end_idx - start_idx
                    t_nrm_s = time.perf_counter()
                    first_cpu_arr = distinct_frames[start_idx]
                    if first_cpu_arr.ndim == 2:
                        is_p010_cpu = first_cpu_arr.dtype in (np.uint16, np.int16)
                        yuv_list = [
                            torch.from_numpy(
                                distinct_frames[start_idx + i].view(np.int16)
                                if is_p010_cpu
                                else distinct_frames[start_idx + i]
                            )
                            for i in range(cur_B)
                        ]
                        yuv_batch = torch.stack(yuv_list, dim=0)
                        tensor = _yuv420_to_nchw_fp16_bt709(
                            yuv_batch,
                            is_p010=is_p010_cpu,
                            chroma_siting=self.chroma_siting,
                            out_dtype=torch.float32,
                            use_triton=False,
                        )
                    else:
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
                    while not slot_drain_events[slot].wait(timeout=0.1):
                        drain_worker.check_error()
                    t_cnv_s = time.perf_counter()
                    out_p010 = self.color_conv.rgb_to_p010le(out_rgb, slot=slot)
                    cpu_total_conv_ms += (time.perf_counter() - t_cnv_s) * 1000.0
                    out_np = out_p010.numpy()

                    slot_views = {start_idx + i: out_np[i] for i in range(cur_B)}
                    _drain_ready_frames(end_idx, slot_views, slot_idx=slot)

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
        self.host_yuv_pins.clear()
        self.gpu_in_ring.clear()
        self.gpu_nhwc_ring.clear()
        self.gpu_yuv_ring.clear()
        self._yuv_ring_dtype = None
        if getattr(self, "color_conv", None) is not None:
            self.color_conv._p010_ring.clear()
        self._micro_event_pool.clear()
        self._sync_event_pool.clear()
        self.last_first_rgb_small = None
        self.input_hw = (0, 0)
        self.output_hw = (0, 0)


TRTZeroCopyUpscaler = DoubleBufferedPipeUpscaler
