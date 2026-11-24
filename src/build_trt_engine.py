"""TensorRT FP16 engine builder and runtime wrapper for SRVGGNetCompact architectures."""

import argparse
import hashlib
import json
import os
import sys
import inspect
from typing import Optional, Any, Tuple

if __name__ == "__main__":
    _repo_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root_dir not in sys.path:
        sys.path.insert(0, _repo_root_dir)
    from src.env_isolation import configure_e_drive_isolation
    configure_e_drive_isolation(_repo_root_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trt_zero_copy_upscaler import SRVGGNetCompactTRTWrapper

# Ghi nhớ các đường dẫn .engine biên dịch thất bại trong phiên chạy để không bao giờ thử lại gây treo pipeline
_FAILED_TRT_ENGINES: set[str] = set()


def is_upcunet2x_model(model: Optional[nn.Module]) -> bool:
    """Return True when model (or its inner_model) is a Real-CUGAN UpCunet2x architecture."""
    if model is None or not isinstance(model, nn.Module):
        return False
    core = getattr(model, "inner_model", model)
    if type(core).__name__ == "UpCunet2x":
        return True
    if hasattr(core, "unet1") and hasattr(core, "unet2") and not hasattr(core, "conv_final"):
        cb = getattr(core.unet1, "conv_bottom", None)
        ks = tuple(getattr(cb, "kernel_size", ())) if cb is not None else ()
        return ks == (4, 4)
    return False


def _fold_span_conv3xc_fp32(module: nn.Module) -> None:
    """Fold SPAN Conv3XC branches into eval_conv once in FP32 and freeze per-forward re-folding."""
    for sub in module.modules():
        upd = getattr(sub, "update_params", None)
        eval_conv = getattr(sub, "eval_conv", None)
        if callable(upd) and isinstance(eval_conv, nn.Conv2d) and hasattr(sub, "sk"):
            if getattr(sub, "_span_params_folded", False):
                continue
            orig_dtype = eval_conv.weight.dtype
            if orig_dtype != torch.float32:
                sub.float()
            sub.eval()
            upd()
            if orig_dtype != torch.float32:
                sub.to(dtype=orig_dtype)
            sub._span_params_folded = True
            sub.update_params = lambda: None


def resolve_trt_profile_max_batch(
    pytorch_model: Optional[nn.Module],
    max_batch: int = 8,
    input_hw: Tuple[int, int] = (1080, 1920),
    available_vram_gb: Optional[float] = None,
) -> int:
    """Resolve safe TensorRT Optimization Profile max_batch by architecture and VRAM bounds."""
    req_b = max(1, int(max_batch))
    in_h, in_w = int(input_hw[0]), int(input_hw[1])
    if (in_h * in_w) < (540 * 960):
        return req_b

    if is_upcunet2x_model(pytorch_model) and (in_h * in_w) >= (720 * 1280):
        if (in_h * in_w) > (800 * 1920):
            return 1
        return min(req_b, 2)

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


class _UpCunet2xTRTExportModule(nn.Module):
    """ONNX/TensorRT & FP16-safe wrapper for Real-CUGAN UpCunet2x with PixelShuffle duality, dynamic reflect pad, and FP32 SEBlock."""

    @staticmethod
    def _deconv2x_to_subpixel_conv(deconv: nn.ConvTranspose2d) -> Tuple[nn.Conv2d, int]:
        c_in, c_out = int(deconv.in_channels), int(deconv.out_channels)
        kh, kw = int(deconv.kernel_size[0]), int(deconv.kernel_size[1])
        ph, pw = int(deconv.padding[0]), int(deconv.padding[1])
        dt, dev = deconv.weight.dtype, deconv.weight.device
        has_bias = deconv.bias is not None
        if (kh, kw) == (2, 2) and (ph, pw) == (0, 0):
            conv = nn.Conv2d(c_in, c_out * 4, kernel_size=1, stride=1, padding=0, bias=has_bias)
            w = deconv.weight.detach().float().permute(1, 2, 3, 0).reshape(c_out * 4, c_in, 1, 1).contiguous()
            crop = 0
        elif (kh, kw) == (4, 4) and (ph, pw) == (3, 3):
            conv = nn.Conv2d(c_in, c_out * 4, kernel_size=2, stride=1, padding=1, bias=has_bias)
            w = (
                deconv.weight.detach()
                .float()
                .view(c_in, c_out, 2, 2, 2, 2)
                .flip(2, 4)
                .permute(1, 3, 5, 0, 2, 4)
                .reshape(c_out * 4, c_in, 2, 2)
                .contiguous()
            )
            crop = 3
        else:
            raise ValueError(f"Unsupported UpCunet2x ConvTranspose2d spec: k={(kh, kw)}, pad={(ph, pw)}")
        conv.weight.data.copy_(w.to(dtype=dt, device=dev))
        if has_bias and conv.bias is not None and deconv.bias is not None:
            conv.bias.data.copy_(deconv.bias.detach().float().repeat_interleave(4).to(dtype=dt, device=dev))
        return conv.to(device=dev, dtype=dt), crop

    def __init__(self, model: nn.Module, target_scale: float = 2.0, scale: Optional[float] = None):
        super().__init__()
        if scale is not None:
            target_scale = float(scale)
        core = getattr(model, "inner_model", model)
        self.unet1 = core.unet1
        self.unet2 = core.unet2
        self.u1_up2_conv, _ = self._deconv2x_to_subpixel_conv(self.unet1.conv2_up)
        self.u1_bot_conv, self._u1_bot_crop = self._deconv2x_to_subpixel_conv(self.unet1.conv_bottom)
        self.u2_up3_conv, _ = self._deconv2x_to_subpixel_conv(self.unet2.conv3_up)
        self.u2_up4_conv, _ = self._deconv2x_to_subpixel_conv(self.unet2.conv4_up)
        self.is_pro = bool(getattr(core, "is_pro", False) or getattr(core, "pro", None) is not None)
        self.model_scale = 2.0
        self.target_scale = float(target_scale)
        self._export_hw: Optional[Tuple[int, int]] = None

    @staticmethod
    def _reflect_pad_2d(x: torch.Tensor, pad_l: int, pad_r: int, pad_t: int, pad_b: int) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            h, w = int(x.shape[2]), int(x.shape[3])
            if h <= max(pad_t, pad_b) or w <= max(pad_l, pad_r):
                return F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="reflect")
        if pad_l > 0 or pad_r > 0:
            parts_w = []
            if pad_l > 0:
                parts_w.append(torch.flip(x[:, :, :, 1 : pad_l + 1], dims=[3]))
            parts_w.append(x)
            if pad_r > 0:
                parts_w.append(torch.flip(x[:, :, :, -pad_r - 1 : -1], dims=[3]))
            x = torch.cat(parts_w, dim=3)
        if pad_t > 0 or pad_b > 0:
            parts_h = []
            if pad_t > 0:
                parts_h.append(torch.flip(x[:, :, 1 : pad_t + 1, :], dims=[2]))
            parts_h.append(x)
            if pad_b > 0:
                parts_h.append(torch.flip(x[:, :, -pad_b - 1 : -1, :], dims=[2]))
            x = torch.cat(parts_h, dim=2)
        return x

    @staticmethod
    def _se_forward(seblock: nn.Module, z: torch.Tensor) -> torch.Tensor:
        s = z.float().mean(dim=(2, 3), keepdim=True)
        if z.dtype == torch.float16 and not torch.onnx.is_in_onnx_export():
            w1 = seblock.conv1.weight.float()
            b1 = seblock.conv1.bias.float() if seblock.conv1.bias is not None else None
            w2 = seblock.conv2.weight.float()
            b2 = seblock.conv2.bias.float() if seblock.conv2.bias is not None else None
            s = F.conv2d(s, w1, b1)
            s = F.relu(s)
            s = F.conv2d(s, w2, b2)
            s = torch.sigmoid(s).to(dtype=z.dtype)
            return z * s
        s = s.to(dtype=seblock.conv1.weight.dtype)
        s = seblock.conv1(s)
        s = F.relu(s)
        s = seblock.conv2(s)
        s = torch.sigmoid(s).to(dtype=z.dtype)
        return z * s

    @classmethod
    def _unet_conv_forward(cls, uconv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        z = uconv.conv(x)
        if getattr(uconv, "seblock", None) is not None:
            z = cls._se_forward(uconv.seblock, z)
        return z

    def _unet1_forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._unet_conv_forward(self.unet1.conv1, x)
        x2 = F.leaky_relu(self.unet1.conv1_down(x1), 0.1)
        x1 = x1[:, :, 4:-4, 4:-4]
        x2 = self._unet_conv_forward(self.unet1.conv2, x2)
        x2 = F.leaky_relu(F.pixel_shuffle(self.u1_up2_conv(x2), 2), 0.1)
        x3 = F.leaky_relu(self.unet1.conv3(x1 + x2), 0.1)
        bot = F.pixel_shuffle(self.u1_bot_conv(x3), 2)
        c = self._u1_bot_crop
        return bot[:, :, c:-c, c:-c] if c > 0 else bot

    def _unet2_forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._unet_conv_forward(self.unet2.conv1, x)
        x2 = F.leaky_relu(self.unet2.conv1_down(x1), 0.1)
        x1 = x1[:, :, 16:-16, 16:-16]
        x2 = self._unet_conv_forward(self.unet2.conv2, x2)
        x3 = F.leaky_relu(self.unet2.conv2_down(x2), 0.1)
        x2 = x2[:, :, 4:-4, 4:-4]
        x3 = self._unet_conv_forward(self.unet2.conv3, x3)
        x3 = F.leaky_relu(F.pixel_shuffle(self.u2_up3_conv(x3), 2), 0.1)
        x4 = self._unet_conv_forward(self.unet2.conv4, x2 + x3)
        x4 = F.leaky_relu(F.pixel_shuffle(self.u2_up4_conv(x4), 2), 0.1)
        x5 = F.leaky_relu(self.unet2.conv5(x1 + x4), 0.1)
        return self.unet2.conv_bottom(x5)

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        if self._export_hw is not None:
            h0, w0 = self._export_hw
        else:
            h0, w0 = int(x.shape[2]), int(x.shape[3])
        if self.is_pro:
            x = x * 0.7 + 0.15
        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x_pad = self._reflect_pad_2d(x, 18, 18 + pw - w0, 18, 18 + ph - h0)
        u1 = self._unet1_forward(x_pad)
        u2 = self._unet2_forward(u1)
        out = u2 + u1[:, :, 20:-20, 20:-20]
        out = out[:, :, : h0 * 2, : w0 * 2]
        if self.is_pro:
            out = (out - 0.15) / 0.7
        if abs(self.model_scale - self.target_scale) > 1e-4:
            interp_in = out.float() if (out.device.type == "cpu" and out.dtype == torch.float16) else out
            scale_ratio = self.target_scale / self.model_scale
            out_target = F.interpolate(
                interp_in,
                scale_factor=scale_ratio,
                mode="bicubic",
                align_corners=False,
                antialias=(scale_ratio < 1.0),
            )
            return out_target.to(x.dtype) if out_target.dtype != x.dtype else out_target
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not torch.onnx.is_in_onnx_export()
            and int(x.shape[0]) > 1
            and (int(x.shape[2]) * int(x.shape[3])) >= (720 * 1280)
        ):
            return torch.cat([self._forward_core(x[i : i + 1]) for i in range(int(x.shape[0]))], dim=0)
        return self._forward_core(x)


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
            scale_ratio = self.target_scale / self.model_scale
            out_target = F.interpolate(
                interp_in,
                scale_factor=scale_ratio,
                mode="bicubic",
                align_corners=False,
                antialias=(scale_ratio < 1.0),
            )
            if out_target.dtype != x.dtype:
                out_target = out_target.to(x.dtype)
            return out_target
        return out


class _PureFP16TRTExportModule(nn.Module):
    """Pure-FP16 ONNX export wrapper collapsing SRVGG 4x->2x analytically via Catmull-Rom AA."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        for layer in self.body:
            feat = layer(feat)
        if self._use_collapsed_aa2x:
            from src.cuda_engine import collapsed_4x_to_2x_forward
            return collapsed_4x_to_2x_forward(feat, x, self.aa_feat_weight, self.aa_base_weight)

        out = self.upsampler(feat)
        base = F.interpolate(x, scale_factor=float(self.upscale), mode="nearest")
        out_hr = out + base
        if abs(float(self.upscale) - self.target_scale) > 1e-4:
            interp_in = out_hr.float() if (out_hr.device.type == "cpu" and out_hr.dtype == torch.float16) else out_hr
            scale_ratio = self.target_scale / float(self.upscale)
            out_target = F.interpolate(
                interp_in,
                scale_factor=scale_ratio,
                mode="bicubic",
                align_corners=False,
                antialias=(scale_ratio < 1.0),
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
    arch_tag: Optional[str] = None,
) -> str:
    """Build deterministic .engine cache path keyed by model_id, arch_tag, GPU, TRT major, resolution, and batch."""
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
    if arch_tag:
        safe_model = f"{safe_model}_{str(arch_tag).strip()}"
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
    """Verify .engine integrity against its required .sha256 sidecar and optional weights SHA-256."""
    if not os.path.exists(engine_path) or os.path.getsize(engine_path) == 0:
        return False
    sidecar_path = f"{engine_path}.sha256"
    if not os.path.exists(sidecar_path):
        return False
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
        if expected_weights_sha256:
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

    from src.cuda_engine import is_srvgg_compact_model
    has_body_srvgg = is_srvgg_compact_model(model_copy)
    if hasattr(model_copy, "inner_model"):
        inner = model_copy.inner_model
        has_body_srvgg = is_srvgg_compact_model(inner)
        model_to_wrap = inner if has_body_srvgg else model_copy
    else:
        model_to_wrap = model_copy

    is_cugan2x = is_upcunet2x_model(model_copy)
    export_dev = device if (device.type == "cuda" and has_body_srvgg) else torch.device("cpu")
    if export_dev.type == "cpu" and device.type != "cpu":
        model_copy = model_copy.cpu()
        model_to_wrap = model_copy.inner_model if (hasattr(model_copy, "inner_model") and has_body_srvgg) else model_copy
    if not has_body_srvgg:
        _fold_span_conv3xc_fp32(model_copy)

    if has_body_srvgg:
        wrapped = _PureFP16TRTExportModule(model_to_wrap, target_scale=target_scale).eval().to(export_dev)
    elif is_cugan2x:
        wrapped = _UpCunet2xTRTExportModule(model_to_wrap, target_scale=target_scale).eval().to(export_dev)
    else:
        wrapped = _GenericTRTExportModule(model_to_wrap, target_scale=target_scale).eval().to(export_dev)
    dh, dw = int(dummy_shape[0]), int(dummy_shape[1])
    if isinstance(wrapped, _UpCunet2xTRTExportModule):
        wrapped._export_hw = (dh, dw)
    if export_dev.type == "cuda" and has_body_srvgg:
        wrapped = wrapped.half()
        dummy_in = torch.randn(1, 3, dh, dw, device=export_dev, dtype=torch.float16)
    else:
        wrapped = wrapped.float()
        dummy_in = torch.randn(1, 3, dh, dw, device=export_dev, dtype=torch.float32)

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
        with torch.no_grad():
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
    if hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        try:
            for l_idx in range(network.num_layers):
                layer = network.get_layer(l_idx)
                if layer is not None and layer.type == trt.LayerType.REDUCE:
                    layer.precision = trt.DataType.FLOAT
        except Exception:
            pass

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
        shared_batch_out_ring: Optional[list[torch.Tensor]] = None,
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
        self._in_cast_buf: Optional[torch.Tensor] = None
        self._out_cast_buf: Optional[torch.Tensor] = None
        self._shared_batch_out_ring = shared_batch_out_ring
        self._batch_out_ring: list[torch.Tensor] = shared_batch_out_ring if shared_batch_out_ring is not None else []
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

    def _acquire_batch_output_buffer(
        self,
        B: int,
        C: int,
        out_h: int,
        out_w: int,
        target_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Vòng đệm đôi 2-slot cho batch tổng hợp (vd: B=8 khi profile max_batch=2) để cắt lát zero-copy."""
        alloc_b = max(int(B), self.max_batch)
        target_tail = (int(C), int(out_h), int(out_w))
        eff_dtype = target_dtype if target_dtype is not None else self.out_torch_dtype
        if (
            len(self._batch_out_ring) != 2
            or self._batch_out_ring[0].shape[0] < alloc_b
            or self._batch_out_ring[0].shape[1:] != target_tail
            or self._batch_out_ring[0].dtype != eff_dtype
            or self._batch_out_ring[0].device != self.device
        ):
            if self._out_ring:
                self._out_ring = []
                self._last_out_ptr = None
            new_ring = [
                torch.empty((alloc_b, *target_tail), dtype=eff_dtype, device=self.device)
                for _ in range(2)
            ]
            if self._batch_out_ring is not None:
                self._batch_out_ring[:] = new_ring
            else:
                self._batch_out_ring = new_ring
            self._batch_ring_idx = 0
            if eff_dtype == self.out_torch_dtype:
                self._last_out_ptr = None
        slot_buf = self._batch_out_ring[self._batch_ring_idx]
        self._batch_ring_idx ^= 1
        return slot_buf[:B]

    def _acquire_out_cast_buffer(self, B: int, C: int, out_h: int, out_w: int) -> torch.Tensor:
        alloc_b = max(int(B), self.max_batch)
        target_tail = (int(C), int(out_h), int(out_w))
        buf = getattr(self, "_out_cast_buf", None)
        if (
            buf is None
            or buf.shape[0] < alloc_b
            or buf.shape[1:] != target_tail
            or buf.dtype != self.out_torch_dtype
            or buf.device != self.device
        ):
            buf = torch.empty((alloc_b, *target_tail), dtype=self.out_torch_dtype, device=self.device)
            self._out_cast_buf = buf
            self._last_out_ptr = None
        return buf[:B]

    def _exec_single_batch(self, contig_in: torch.Tensor, out_view: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = contig_in.shape
        out_h, out_w = H * self.scale, W * self.scale
        if contig_in.dtype != self.in_torch_dtype:
            alloc_in_b = max(int(B), self.max_batch)
            in_cast = getattr(self, "_in_cast_buf", None)
            if (
                in_cast is None
                or in_cast.shape[0] < alloc_in_b
                or in_cast.shape[1:] != (int(C), int(H), int(W))
                or in_cast.dtype != self.in_torch_dtype
                or in_cast.device != self.device
            ):
                in_cast = torch.empty(
                    (alloc_in_b, int(C), int(H), int(W)),
                    dtype=self.in_torch_dtype,
                    device=self.device,
                )
                self._in_cast_buf = in_cast
                self._last_in_ptr = None
            in_slice = in_cast[:B]
            in_slice.copy_(contig_in)
            contig_in = in_slice
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
        out_h, out_w = H * self.scale, W * self.scale
        if orig_dtype == self.out_torch_dtype:
            if B <= self.max_batch:
                return self._exec_single_batch(contig_in)
            full_out = self._acquire_batch_output_buffer(B, C, out_h, out_w, target_dtype=orig_dtype)
            for s in range(0, B, self.max_batch):
                e = min(s + self.max_batch, B)
                self._exec_single_batch(contig_in[s:e], out_view=full_out[s:e])
            return full_out

        # Engine I/O dtype differs from caller dtype (e.g. FP32 ONNX I/O vs FP16 caller):
        # keep the 2-slot B-frame output ring in orig_dtype (saving ~1.0-1.4 GiB VRAM) and cast per max_batch slice.
        full_out = self._acquire_batch_output_buffer(B, C, out_h, out_w, target_dtype=orig_dtype)
        for s in range(0, B, self.max_batch):
            e = min(s + self.max_batch, B)
            cast_view = self._acquire_out_cast_buffer(e - s, C, out_h, out_w)
            slice_out = self._exec_single_batch(contig_in[s:e], out_view=cast_view)
            full_out[s:e].copy_(slice_out)
        return full_out

    def release(self) -> None:
        """Deterministically free TensorRT IExecutionContext/ICudaEngine device memory and output rings."""
        self._out_ring = []
        self._in_cast_buf = None
        self._out_cast_buf = None
        if self._shared_batch_out_ring is None:
            self._batch_out_ring = []
        self._last_input_shape = None
        self._last_in_ptr = None
        self._last_out_ptr = None
        for attr in ("context", "engine", "runtime", "fallback_module"):
            if hasattr(self, attr):
                setattr(self, attr, None)


def get_hybrid_engine_or_wrapper(
    pytorch_model: nn.Module,
    max_batch: int = 8,
    force_pytorch: bool = False,
    model_id: str = "realesr-animevideov3",
    input_hw: Tuple[int, int] = (1080, 1920),
    scale: float = 2.0,
    weights_sha256: Optional[str] = None,
    shared_batch_out_ring: Optional[list[torch.Tensor]] = None,
) -> Any:
    """Build or load verified TensorRT FP16 engine, falling back safely to PyTorch FP16 wrapper."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from src.cuda_engine import is_srvgg_compact_model
    has_body_srvgg = is_srvgg_compact_model(pytorch_model)
    if hasattr(pytorch_model, "inner_model"):
        inner = pytorch_model.inner_model
        has_body_srvgg = is_srvgg_compact_model(inner)
        model_for_wrapper = inner if has_body_srvgg else pytorch_model
    else:
        model_for_wrapper = pytorch_model

    is_cugan2x = is_upcunet2x_model(pytorch_model)
    if has_body_srvgg:
        wrapper = SRVGGNetCompactTRTWrapper(model_for_wrapper, scale=scale).eval().to(device)
    elif is_cugan2x:
        wrapper = _UpCunet2xTRTExportModule(model_for_wrapper, target_scale=scale).eval().to(device)
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
        is_collapsed_4x = bool(
            getattr(wrapper, "_use_collapsed_aa2x", False)
            or (getattr(wrapper, "model_scale", 2.0) > float(scale) + 1e-4)
        )
        arch_tag_val = "aa2x" if is_collapsed_4x else ("ps2x" if is_cugan2x else None)
        engine_path = get_trt_engine_path(
            model_id=model_id,
            max_batch=trt_max_batch,
            input_hw=input_hw,
            weights_sha256=weights_sha256 if is_cugan2x else None,
            arch_tag=arch_tag_val,
        )
        if engine_path not in _FAILED_TRT_ENGINES:
            onnx_path = engine_path.replace(".engine", ".onnx")
            try:
                needs_build = (
                    not os.path.exists(engine_path)
                    or os.path.getsize(engine_path) == 0
                    or not verify_trt_engine_sha256_sidecar(engine_path, expected_weights_sha256=weights_sha256)
                )
                built = None
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

                if (not needs_build or built) and os.path.exists(engine_path) and os.path.getsize(engine_path) > 0:
                    trt_runner = TRTRuntimeEngineWrapper(
                        engine_path,
                        device=device,
                        input_hw=input_hw,
                        max_batch=trt_max_batch,
                        scale=int(round(scale)),
                        fallback_module=wrapper,
                        shared_batch_out_ring=shared_batch_out_ring,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return trt_runner
            except Exception as e:
                _FAILED_TRT_ENGINES.add(engine_path)
                print(f"⚠️ Không thể khởi tạo TensorRT Engine ({e}), tự động chuyển sang PyTorch FP16 Wrapper.")

    return wrapper


def build_cli_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for standalone TensorRT FP16 engine compilation and verification."""
    parser = argparse.ArgumentParser(
        prog="src/build_trt_engine.py",
        description="Compile and SHA-256-verify TensorRT 10 FP16 .engine artifacts for registered SR models.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model ID from models/models.yaml, or 'all' (default: registry default model).",
    )
    parser.add_argument("--width", type=int, default=1920, help="Input frame width in pixels (default: 1920).")
    parser.add_argument("--height", type=int, default=1080, help="Input frame height in pixels (default: 1080).")
    parser.add_argument("--max-batch", type=int, default=8, help="Requested max profile batch size (default: 8).")
    parser.add_argument("--workspace-gb", type=int, default=2, help="TensorRT workspace cap in GiB (default: 2).")
    parser.add_argument("--cache-dir", default=None, help="Optional directory for .engine and .sha256 outputs.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even when verified .engine exists.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve target .engine paths and sidecar verification state without GPU lock or compilation.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint for standalone TensorRT FP16 engine builds under .gpu_active.lock."""
    args = build_cli_parser().parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.max_batch < 1 or args.workspace_gb < 1:
        print("[TRT-ERROR] width, height, max-batch, and workspace-gb must be positive integers.", file=sys.stderr)
        return 2

    from src.models_resolver import get_default_model_id, load_models_registry

    registry = load_models_registry()
    raw_model = str(args.model).strip() if args.model else ""
    if raw_model.lower() == "all":
        model_ids = list(registry.keys())
    elif raw_model.lower() == "auto-pareto":
        model_ids = [
            k for k in ("animejanai-hd-v3-compact", "animejanai-hd-v3-ultracompact") if k in registry
        ]
        if not model_ids:
            model_ids = [get_default_model_id(registry)]
    elif raw_model:
        model_ids = [raw_model]
    else:
        model_ids = [get_default_model_id(registry)]

    unknown = [m for m in model_ids if m not in registry]
    if unknown:
        print(f"[TRT-ERROR] Unknown model ID(s): {', '.join(unknown)}. Valid: {', '.join(registry)}", file=sys.stderr)
        return 2

    input_hw = (int(args.height), int(args.width))
    if args.dry_run:
        vram_gb = 12.0
        if torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                vram_gb = 12.0
        for mid in model_ids:
            info = registry[mid]
            arch_cfg = info.get("arch") or {}
            arch_type = str(arch_cfg.get("type") or "")
            up_scale = int(arch_cfg.get("upscale") or 2)
            is_cugan = arch_type == "UpCunet2x"
            arch_tag = "aa2x" if (arch_type == "SRVGGNetCompact" and up_scale > 2) else ("ps2x" if is_cugan else None)
            req_b = max(1, int(args.max_batch))
            if (input_hw[0] * input_hw[1]) < (540 * 960):
                eff_b = req_b
            elif is_cugan and (input_hw[0] * input_hw[1]) > (800 * 1920):
                eff_b = 1
            elif (is_cugan and (input_hw[0] * input_hw[1]) >= (720 * 1280)) or up_scale >= 4:
                eff_b = min(req_b, 2)
            elif vram_gb < 15.0 and (input_hw[0] * input_hw[1]) >= (720 * 1280):
                eff_b = min(req_b, 2)
            else:
                eff_b = min(req_b, 4)
            w_sha = str(info.get("sha256") or "").strip().lower() or None
            eng_path = get_trt_engine_path(
                model_id=mid,
                max_batch=eff_b,
                input_hw=input_hw,
                cache_dir=args.cache_dir,
                weights_sha256=w_sha if is_cugan else None,
                arch_tag=arch_tag,
            )
            verified = bool(
                os.path.exists(eng_path)
                and os.path.getsize(eng_path) > 0
                and verify_trt_engine_sha256_sidecar(eng_path, expected_weights_sha256=w_sha)
            )
            print(f"[DRY-RUN] model={mid} batch=1..{eff_b} verified={verified} engine={eng_path}")
        return 0

    if not torch.cuda.is_available() or not is_tensorrt_available():
        print("[TRT-ERROR] CUDA GPU and TensorRT runtime are required for live engine compilation.", file=sys.stderr)
        return 1

    from src.config import configure_cuda_fast_math, configure_nvme_jit_caches, gpu_exclusivity_lock
    from src.cuda_engine import GPUPersistentWorker, _cleanup_gpu_worker, is_srvgg_compact_model

    configure_nvme_jit_caches()
    configure_cuda_fast_math()
    device = torch.device("cuda:0")
    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

    with gpu_exclusivity_lock():
        for mid in model_ids:
            info = registry[mid]
            w_sha = str(info.get("sha256") or "").strip().lower() or None
            worker = GPUPersistentWorker(
                model_id=mid,
                weights_path=info["weights_path"],
                precision="fp16",
                device="cuda",
                batch_size=int(args.max_batch),
                use_tensorrt=False,
                weights_sha256=w_sha,
            )
            try:
                worker._load_model()
                py_model = worker.model
                if py_model is None:
                    print(f"[TRT-ERROR] Failed to load PyTorch weights for '{mid}'.", file=sys.stderr)
                    return 1
                trt_max_batch = resolve_trt_profile_max_batch(
                    py_model,
                    max_batch=int(args.max_batch),
                    input_hw=input_hw,
                    available_vram_gb=vram_gb,
                )
                is_cugan2x = is_upcunet2x_model(py_model)
                inner_cand = py_model.inner_model if hasattr(py_model, "inner_model") else py_model
                has_srvgg = is_srvgg_compact_model(inner_cand)
                if has_srvgg:
                    tmp_wrap: nn.Module = SRVGGNetCompactTRTWrapper(inner_cand, scale=2.0)
                elif is_cugan2x:
                    tmp_wrap = _UpCunet2xTRTExportModule(py_model, target_scale=2.0)
                else:
                    tmp_wrap = _GenericTRTExportModule(py_model, target_scale=2.0)
                is_collapsed_4x = bool(
                    getattr(tmp_wrap, "_use_collapsed_aa2x", False)
                    or (getattr(tmp_wrap, "model_scale", 2.0) > 2.0 + 1e-4)
                )
                arch_tag_val = "aa2x" if is_collapsed_4x else ("ps2x" if is_cugan2x else None)
                del tmp_wrap

                engine_path = get_trt_engine_path(
                    model_id=mid,
                    max_batch=trt_max_batch,
                    input_hw=input_hw,
                    cache_dir=args.cache_dir,
                    weights_sha256=w_sha if is_cugan2x else None,
                    arch_tag=arch_tag_val,
                )
                needs_build = (
                    bool(args.force)
                    or not os.path.exists(engine_path)
                    or os.path.getsize(engine_path) == 0
                    or not verify_trt_engine_sha256_sidecar(engine_path, expected_weights_sha256=w_sha)
                )
                if needs_build:
                    onnx_path = engine_path.replace(".engine", ".onnx")
                    try:
                        export_model_to_onnx(
                            py_model,
                            onnx_path,
                            max_batch=trt_max_batch,
                            dummy_shape=input_hw,
                            target_scale=2.0,
                        )
                        built = build_tensorrt_engine(
                            onnx_path,
                            engine_path,
                            max_batch=trt_max_batch,
                            input_hw=input_hw,
                            workspace_gb=int(args.workspace_gb),
                            weights_sha256=w_sha,
                        )
                    finally:
                        if os.path.exists(onnx_path):
                            try:
                                os.remove(onnx_path)
                            except OSError:
                                pass
                    if not built or not verify_trt_engine_sha256_sidecar(engine_path, expected_weights_sha256=w_sha):
                        print(f"[TRT-ERROR] Engine build or SHA-256 verification failed for '{mid}'.", file=sys.stderr)
                        return 1

                runner = TRTRuntimeEngineWrapper(
                    engine_path,
                    device=device,
                    input_hw=input_hw,
                    max_batch=trt_max_batch,
                    scale=2,
                )
                probe_in = torch.zeros((1, 3, input_hw[0], input_hw[1]), dtype=torch.float16, device=device)
                probe_out = runner(probe_in)
                expected_shape = (1, 3, input_hw[0] * 2, input_hw[1] * 2)
                if tuple(probe_out.shape) != expected_shape or not bool(torch.isfinite(probe_out).all()):
                    print(
                        f"[TRT-ERROR] Runtime verification failed for '{mid}': got shape {tuple(probe_out.shape)}.",
                        file=sys.stderr,
                    )
                    return 1
                del runner, probe_in, probe_out
                torch.cuda.empty_cache()
                status_tag = "BUILT" if needs_build else "CACHED"
                print(f"[TRT-{status_tag}] model={mid} batch=1..{trt_max_batch} engine={engine_path}")
            finally:
                _cleanup_gpu_worker(worker)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
