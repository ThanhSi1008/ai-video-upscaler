"""PyTorch CUDA FP16 and TensorRT persistent inference worker for 4K anime upscaling."""

import hashlib
import inspect
import os
import sys
import time
import glob
import psutil
import shutil
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DEFAULT_BATCH_SIZE, REALESRGAN_BIN

def _pad_replicate_1px(t: torch.Tensor) -> torch.Tensor:
    """Replicate-pad spatial H, W borders by 1 pixel via Slice+Concat (zero ONNX/TRT ForeignNode)."""
    t_h = torch.cat((t[:, :, :1, :], t, t[:, :, -1:, :]), dim=2)
    return torch.cat((t_h[:, :, :, :1], t_h, t_h[:, :, :, -1:]), dim=3)


def build_collapsed_4x_to_2x_buffers(
    b_param: float = 0.0,
    c_param: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact Catmull-Rom (B=0, C=0.5) anti-aliased 4x->2x collapsed 3x3 conv weights (g=3)."""
    def _cubic_keys(d: torch.Tensor) -> torch.Tensor:
        xa = d.abs()
        y1 = (
            (12.0 - 9.0 * b_param - 6.0 * c_param) * xa ** 3
            + (-18.0 + 12.0 * b_param + 6.0 * c_param) * xa ** 2
            + (6.0 - 2.0 * b_param)
        ) / 6.0
        y2 = (
            (-b_param - 6.0 * c_param) * xa ** 3
            + (6.0 * b_param + 30.0 * c_param) * xa ** 2
            + (-12.0 * b_param - 48.0 * c_param) * xa
            + (8.0 * b_param + 24.0 * c_param)
        ) / 6.0
        return torch.where(xa < 1.0, y1, torch.where(xa < 2.0, y2, torch.zeros_like(xa)))

    k1d = torch.zeros((2, 3, 4), dtype=torch.float64)
    for q, center_4x in enumerate((0.5, 2.5)):
        for dn in (-1, 0, 1):
            for r in range(4):
                pos_4x = torch.tensor(float(dn * 4 + r), dtype=torch.float64)
                k1d[q, dn + 1, r] = _cubic_keys((pos_4x - center_4x) * 0.5)
        k1d[q] /= k1d[q].sum()

    feat_weight = torch.zeros((12, 16, 3, 3), dtype=torch.float64)
    base_weight = torch.zeros((12, 1, 3, 3), dtype=torch.float64)
    for ch in range(3):
        for qy in range(2):
            for qx in range(2):
                out_ch = ch * 4 + qy * 2 + qx
                for dy_idx in range(3):
                    for dx_idx in range(3):
                        outer = k1d[qy, dy_idx][:, None] * k1d[qx, dx_idx][None, :]
                        feat_weight[out_ch, :, dy_idx, dx_idx] = outer.reshape(16)
                        base_weight[out_ch, 0, dy_idx, dx_idx] = outer.sum()
    return feat_weight.float(), base_weight.float()


def collapsed_4x_to_2x_forward(
    feat_48: torch.Tensor,
    x_rgb: torch.Tensor,
    aa_feat_weight: torch.Tensor,
    aa_base_weight: torch.Tensor,
) -> torch.Tensor:
    """Collapse PixelShuffle(4) + Nearest(4x) + AA Bicubic(0.5x) into 1080p grouped convs + PixelShuffle(2)."""
    fw = (
        aa_feat_weight
        if (aa_feat_weight.device == feat_48.device and aa_feat_weight.dtype == feat_48.dtype)
        else aa_feat_weight.to(device=feat_48.device, dtype=feat_48.dtype)
    )
    bw = (
        aa_base_weight
        if (aa_base_weight.device == x_rgb.device and aa_base_weight.dtype == x_rgb.dtype)
        else aa_base_weight.to(device=x_rgb.device, dtype=x_rgb.dtype)
    )
    if (not torch.onnx.is_in_onnx_export()) and feat_48.device.type == "cpu" and feat_48.shape[0] > 1:
        B, _, H, W = feat_48.shape
        out_buf = torch.empty((B, 3, H * 2, W * 2), dtype=feat_48.dtype, device=feat_48.device)
        for i in range(B):
            sf = F.conv2d(_pad_replicate_1px(feat_48[i : i + 1]), fw, groups=3)
            sb = F.conv2d(_pad_replicate_1px(x_rgb[i : i + 1]), bw, groups=3)
            if sb.dtype != sf.dtype:
                sb = sb.to(dtype=sf.dtype)
            out_buf[i : i + 1].copy_(F.pixel_shuffle(sf + sb, 2))
        return out_buf
    sub_feat = F.conv2d(_pad_replicate_1px(feat_48), fw, groups=3)
    sub_base = F.conv2d(_pad_replicate_1px(x_rgb), bw, groups=3)
    if sub_base.dtype != sub_feat.dtype:
        sub_base = sub_base.to(dtype=sub_feat.dtype)
    return F.pixel_shuffle(sub_feat + sub_base, 2)


class RealESRAnimeVideoV3(nn.Module):
    """RealESR-AnimeVideoV3 (64f16c x4 collapsed analytically with Catmull-Rom AA -> 2x 4K)."""
    def __init__(self):
        super().__init__()
        self.upscale = 4
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(3, 64, 3, 1, 1))
        self.body.append(nn.PReLU(64))
        for _ in range(16):
            self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
            self.body.append(nn.PReLU(64))
        self.body.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(4)
        fw, bw = build_collapsed_4x_to_2x_buffers(b_param=0.0, c_param=0.5)
        self.register_buffer("aa_feat_weight", fw, persistent=False)
        self.register_buffer("aa_base_weight", bw, persistent=False)

    def _tail_4x_to_2x(self, feat_slice: torch.Tensor, x_slice: torch.Tensor) -> torch.Tensor:
        return collapsed_4x_to_2x_forward(feat_slice, x_slice, self.aa_feat_weight, self.aa_base_weight)

    def forward(self, x):
        feat = x
        for layer in self.body:
            feat = layer(feat)
        return self._tail_4x_to_2x(feat, x)


class AnimeJaNaiCompact2x(nn.Module):
    """AnimeJaNai HD V3 Compact / UltraCompact / SuperUltraCompact SRVGGNetCompact x2."""
    def __init__(self, num_feat: int = 64, num_conv: int = 16):
        super().__init__()
        self.num_feat = int(num_feat)
        self.num_conv = int(num_conv)
        self.upscale = 2
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(3, self.num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(self.num_feat))
        for _ in range(self.num_conv):
            self.body.append(nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(self.num_feat))
        self.body.append(nn.Conv2d(self.num_feat, 12, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(2)

    def forward(self, x):
        feat = x
        for layer in self.body:
            feat = layer(feat)
        out = self.upsampler(feat)
        base = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return out + base


def is_srvgg_compact_model(model: Optional[nn.Module]) -> bool:
    """Return True strictly for SRVGGNetCompact modules (pure body ModuleList + PixelShuffle)."""
    if model is None or not isinstance(model, nn.Module):
        return False
    body = getattr(model, "body", None)
    upsampler = getattr(model, "upsampler", None)
    if not isinstance(body, (nn.ModuleList, nn.Sequential)) or len(body) < 1 or not isinstance(upsampler, nn.PixelShuffle):
        return False
    child_names = {name for name, _ in model.named_children()}
    if not child_names.issubset({"body", "upsampler"}):
        return False
    first_layer, last_layer = body[0], body[-1]
    if not isinstance(first_layer, nn.Conv2d) or not isinstance(last_layer, nn.Conv2d):
        return False
    up_r = int(upsampler.upscale_factor)
    return int(first_layer.in_channels) == 3 and int(last_layer.out_channels) == 3 * (up_r * up_r)


class SpandrelModelWrapper(nn.Module):
    """Wrap Spandrel-loaded SR models with collapsed SRVGG 4x->2x AA or generic 2-frame-sliced AA bicubic."""
    def __init__(self, inner_model: nn.Module, scale: int = 2, target_scale: float = 2.0):
        super().__init__()
        self.inner_model = inner_model
        self.scale = int(scale)
        self.target_scale = float(target_scale)
        is_pure_srvgg = is_srvgg_compact_model(inner_model)
        if is_pure_srvgg:
            self.body = inner_model.body
            self.upsampler = inner_model.upsampler
        if hasattr(inner_model, "upscale"):
            self.upscale = inner_model.upscale
        else:
            self.upscale = self.scale

        self._can_collapse_srvgg_4x = (
            is_pure_srvgg
            and self.scale == 4
            and abs(self.target_scale - 2.0) <= 1e-4
            and int(inner_model.upsampler.upscale_factor) == 4
        )
        if self._can_collapse_srvgg_4x:
            fw, bw = build_collapsed_4x_to_2x_buffers(b_param=0.0, c_param=0.5)
            ref_p = next(inner_model.parameters(), None)
            if ref_p is not None:
                fw = fw.to(device=ref_p.device, dtype=ref_p.dtype)
                bw = bw.to(device=ref_p.device, dtype=ref_p.dtype)
            self.register_buffer("aa_feat_weight", fw, persistent=False)
            self.register_buffer("aa_base_weight", bw, persistent=False)

    def _tail_4x_to_2x(self, out_4x: torch.Tensor) -> torch.Tensor:
        interp_in = out_4x.float() if (out_4x.device.type == "cpu" and out_4x.dtype == torch.float16) else out_4x
        scale_ratio = (self.target_scale / float(self.scale)) if self.scale > self.target_scale else 0.5
        use_aa = scale_ratio < 1.0
        out_2x = F.interpolate(
            interp_in,
            scale_factor=scale_ratio,
            mode="bicubic",
            align_corners=False,
            antialias=use_aa,
        )
        return out_2x.to(out_4x.dtype) if out_2x.dtype != out_4x.dtype else out_2x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            if (
                not torch.onnx.is_in_onnx_export()
                and int(x.shape[0]) > 1
                and (int(x.shape[2]) * int(x.shape[3])) >= (720 * 1280)
                and hasattr(self.inner_model, "unet1")
                and hasattr(self.inner_model, "unet2")
            ):
                return torch.cat([self.inner_model(x[i : i + 1]) for i in range(int(x.shape[0]))], dim=0)
            return self.inner_model(x)

        if self._can_collapse_srvgg_4x:
            feat = x
            for layer in self.body:
                feat = layer(feat)
            return collapsed_4x_to_2x_forward(feat, x, self.aa_feat_weight, self.aa_base_weight)

        if abs(self.scale - self.target_scale) > 1e-4:
            # Micro-slice 8K intermediate tensors across 2 frames on Full HD+ batches for generic 4x models
            if (not torch.onnx.is_in_onnx_export()) and x.shape[0] > 2 and (x.shape[2] * x.shape[3]) >= (540 * 960):
                B, _, H, W = x.shape
                eff_mult = self.target_scale if self.scale > self.target_scale else (self.scale * 0.5)
                out_h = int(round(H * eff_mult))
                out_w = int(round(W * eff_mult))
                out_buf = torch.empty((B, 3, out_h, out_w), dtype=x.dtype, device=x.device)
                for s in range(0, B, 2):
                    e = min(s + 2, B)
                    out_4x = self.inner_model(x[s:e])
                    out_buf[s:e].copy_(self._tail_4x_to_2x(out_4x))
                return out_buf

            out_4x = self.inner_model(x)
            return self._tail_4x_to_2x(out_4x)

        return self.inner_model(x)


class GPUError(Exception):
    """Raised when PyTorch/CUDA inference produces NaN, Inf, OOM, or runtime errors."""
    pass


def _filter_supported_kwargs(fn: Any, extra_kw: Dict[str, Any]) -> Dict[str, Any]:
    """Filter optional kwargs so strict test stubs without new parameters never raise TypeError."""
    if hasattr(fn, "_mock_name"):
        return dict(extra_kw)
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(extra_kw)
    return {k: v for k, v in extra_kw.items() if k in params}


def _sha256_of_file(path_str: str) -> str:
    h = hashlib.sha256()
    with open(path_str, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def compute_blended_weights_sha256(cons_sha: str, dn3x_sha: str, alpha: float) -> str:
    """Compute deterministic 64-hex SHA-256 digest for episode-level Real-CUGAN grain adaptation."""
    a = round(max(0.0, min(1.0, float(alpha))), 4)
    c_clean = str(cons_sha or "").strip().lower()
    d_clean = str(dn3x_sha or "").strip().lower()
    if a <= 0.0:
        return c_clean
    if a >= 1.0:
        return d_clean
    token = f"upcunet2x_blend_v2:{c_clean}:{d_clean}:{a:.4f}".encode("utf-8")
    return hashlib.sha256(token).hexdigest().lower()


def blend_upcunet_state_dicts(
    cons_path: str,
    dn3x_path: str,
    alpha: float,
    cons_sha: Optional[str] = None,
    dn3x_sha: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], str]:
    """Adapt Real-CUGAN Pro weights across grain tiers without scrambling disjoint hidden channel basins."""
    a = round(max(0.0, min(1.0, float(alpha))), 4)
    c_sha = str(cons_sha or "").strip().lower() or _sha256_of_file(cons_path)
    d_sha = str(dn3x_sha or "").strip().lower() or _sha256_of_file(dn3x_path)
    blended_sha = compute_blended_weights_sha256(c_sha, d_sha, a)

    cons_raw = torch.load(cons_path, map_location="cpu", weights_only=True)
    dn3x_raw = torch.load(dn3x_path, map_location="cpu", weights_only=True)
    cons_sd = cons_raw.get("params_ema", cons_raw.get("params", cons_raw)) if isinstance(cons_raw, dict) else cons_raw
    dn3x_sd = dn3x_raw.get("params_ema", dn3x_raw.get("params", dn3x_raw)) if isinstance(dn3x_raw, dict) else dn3x_raw
    if set(cons_sd.keys()) != set(dn3x_sd.keys()):
        raise ValueError("Incompatible state_dict keys between pro-conservative and pro-denoise3x weights.")

    # Upstream pro-conservative and pro-denoise3x were trained from independent seeds (cos ~ 0.04-0.18 on
    # unet1.conv3 / unet2.conv5), so naive weight-space lerp shifts DC luma by >15 LSB; fold native UNet2
    # conv4_up detail attenuation (u2_alpha = 1 - a/3 -> 0.85 at a=0.45) unless checkpoints share a basin.
    basin_aligned = True
    if 0.0 < a < 1.0:
        for probe_k in ("unet1.conv3.weight", "unet2.conv5.weight"):
            if probe_k in cons_sd and probe_k in dn3x_sd:
                vc_f = cons_sd[probe_k].detach().float().flatten()
                vd_f = dn3x_sd[probe_k].detach().float().flatten()
                vc_c = vc_f - vc_f.mean()
                vd_c = vd_f - vd_f.mean()
                if vc_c.numel() > 1 and float(vc_c.norm()) > 1e-8 and float(vd_c.norm()) > 1e-8:
                    cos_v = float(F.cosine_similarity(vc_c.unsqueeze(0), vd_c.unsqueeze(0)).item())
                    if cos_v < 0.85:
                        basin_aligned = False
                        break

    u2_alpha = 1.0 - (a / 3.0)
    blended: Dict[str, torch.Tensor] = {}
    has_pro = ("pro" in cons_sd) or ("pro" in dn3x_sd)
    for k, v_c in cons_sd.items():
        if k == "pro":
            continue
        v_d = dn3x_sd[k]
        if a <= 0.0:
            blended[k] = v_c.detach().clone()
        elif a >= 1.0:
            blended[k] = v_d.detach().clone()
        elif basin_aligned and torch.is_floating_point(v_c):
            blended[k] = torch.lerp(v_c.detach().float(), v_d.detach().float(), a)
        elif (not basin_aligned) and k == "unet2.conv4_up.weight" and torch.is_floating_point(v_c):
            blended[k] = (v_c.detach().float() * u2_alpha).to(dtype=v_c.dtype)
        else:
            blended[k] = v_c.detach().clone()
    if has_pro:
        blended["pro"] = torch.zeros(1, dtype=torch.float32)
    return blended, blended_sha


def resolve_episode_model_blend(
    model_info: Dict[str, Any],
    grain_tier: str = "clean",
) -> Dict[str, Any]:
    """Attach episode-level Real-CUGAN Pro alpha-blend metadata and blended_weights_sha256 for grained sources."""
    resolved = dict(model_info)
    base_sha = str(resolved.get("base_sha256") or resolved.get("sha256") or "").strip().lower()
    resolved["base_sha256"] = base_sha
    m_id = str(resolved.get("id") or "").strip().lower()
    tier = str(grain_tier or "clean").strip().lower()
    if m_id != "realcugan-pro-conservative-2x" or tier not in ("light", "heavy"):
        resolved["blend_alpha"] = 0.0
        resolved["blend_secondary_weights_path"] = None
        resolved["blended_weights_sha256"] = base_sha
        resolved["sha256"] = base_sha
        return resolved

    alpha = 0.45 if tier == "light" else 1.0
    try:
        from src.models_resolver import _resolve_repo_path, load_models_registry
        reg = load_models_registry()
        dn3x_entry = reg.get("realcugan-pro-denoise3x-2x")
    except Exception:
        dn3x_entry = None

    if not isinstance(dn3x_entry, dict) or not dn3x_entry.get("weights_path"):
        resolved["blend_alpha"] = 0.0
        resolved["blend_secondary_weights_path"] = None
        resolved["blended_weights_sha256"] = base_sha
        resolved["sha256"] = base_sha
        return resolved

    sec_path = _resolve_repo_path(str(dn3x_entry["weights_path"]))
    sec_sha = str(dn3x_entry.get("sha256") or "").strip().lower()
    if (not base_sha and os.path.isfile(str(resolved.get("weights_path") or ""))):
        base_sha = _sha256_of_file(str(resolved["weights_path"]))
    if (not sec_sha and os.path.isfile(sec_path)):
        sec_sha = _sha256_of_file(sec_path)
    blended_sha = compute_blended_weights_sha256(base_sha, sec_sha, alpha)
    resolved["blend_alpha"] = alpha
    resolved["blend_secondary_model_id"] = "realcugan-pro-denoise3x-2x"
    resolved["blend_secondary_weights_path"] = sec_path
    resolved["blend_secondary_sha256"] = sec_sha
    resolved["blended_weights_sha256"] = blended_sha
    resolved["sha256"] = blended_sha
    return resolved


class GPUPersistentWorker:
    """Persistent PyTorch CUDA/CPU worker with NHWC channels-last execution and TRT caching."""
    def __init__(
        self,
        model_id: str = "animevideov3",
        weights_path: str = "models_bench/realesr-animevideov3.pth",
        precision: str = "fp16",
        max_chunks_before_restart: int = 10,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        use_tensorrt: bool = True,
        blend_alpha: float = 0.0,
        secondary_weights_path: Optional[str] = None,
        weights_sha256: Optional[str] = None,
    ):
        self.model_id = model_id
        self.weights_path = weights_path
        self.precision = precision.lower()
        self.max_chunks_before_restart = max_chunks_before_restart
        self.chunks_processed = 0
        self.total_restarts = 0
        self.use_tensorrt = bool(use_tensorrt)
        self.blend_alpha = round(max(0.0, min(1.0, float(blend_alpha or 0.0))), 4)
        self.secondary_weights_path = secondary_weights_path
        self.weights_sha256 = str(weights_sha256 or "").strip().lower()
        self._trt_runners: Dict[Any, Any] = {}
        self._zero_disk_upscaler: Any = None
        self._zero_disk_wrapper: Any = None
        self._zero_disk_key: Any = None
        
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.device.type == "cuda":
            from src.config import configure_cuda_fast_math
            configure_cuda_fast_math()

        if batch_size is not None:
            self.batch_size = max(1, int(batch_size))
        else:
            from src.config import DEFAULT_BATCH_SIZE
            self.batch_size = DEFAULT_BATCH_SIZE if self.device.type == "cuda" else 1

        self.model = None
        self._load_model()

    def _resolve_registry_entry(self) -> Optional[Dict[str, Any]]:
        """Look up model metadata in models/models.yaml (case-insensitive)."""
        try:
            from src.models_resolver import load_models_registry
            registry = load_models_registry()
            if self.model_id in registry:
                return registry[self.model_id]
            target_lower = str(self.model_id or "").strip().lower()
            for k, v in registry.items():
                if str(k).lower() == target_lower:
                    return v
        except Exception:
            pass
        return None

    def _is_canonical_registered_path(self, reg_info: Optional[Dict[str, Any]]) -> bool:
        """Return whether self.weights_path matches the canonical path in models/models.yaml."""
        if not reg_info or not reg_info.get("weights_path"):
            return False
        from src.models_resolver import _resolve_repo_path
        reg_path = _resolve_repo_path(str(reg_info["weights_path"]))
        norm_self = os.path.normcase(os.path.abspath(self.weights_path))
        norm_reg = os.path.normcase(os.path.abspath(reg_path))
        norm_raw = os.path.normcase(os.path.abspath(str(reg_info["weights_path"])))
        return norm_self in (norm_reg, norm_raw)

    def _auto_download_registered_weights(self, reg_info: Dict[str, Any]) -> None:
        """Download and SHA-256 verify registered model weights from HTTPS source_url."""
        import urllib.parse
        import urllib.request
        import zipfile
        from src.models_resolver import verify_weights_sha256

        url = str(reg_info.get("source_url") or "").strip()
        expected_sha = str(reg_info.get("sha256") or "").strip().lower()
        if not url.startswith("https://"):
            raise RuntimeError(f"Từ chối tự động tải trọng số từ URL không phải HTTPS: {url!r}")
        if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
            raise RuntimeError(
                f"Từ chối tự động tải trọng số '{self.weights_path}' vì thiếu mã băm SHA-256 64-hex hợp lệ để chống nạp pickle không xác thực."
            )

        os.makedirs(os.path.dirname(os.path.abspath(self.weights_path)), exist_ok=True)
        tmp_dl = self.weights_path + f".part.{os.getpid()}"
        tmp_zip = self.weights_path + f".zip.part.{os.getpid()}"
        url_path = urllib.parse.urlsplit(url).path.lower()

        print(f"📥 Đang tự động tải trọng số '{reg_info.get('name', self.model_id)}' từ {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            if url_path.endswith(".zip"):
                with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_zip, "wb") as out_f:
                    shutil.copyfileobj(resp, out_f)
                target_base = os.path.basename(self.weights_path)
                matched_member = None
                with zipfile.ZipFile(tmp_zip, "r") as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        norm_member = info.filename.replace("\\", "/")
                        if norm_member.startswith("__MACOSX/") or "/__MACOSX/" in norm_member:
                            continue
                        member_base = os.path.basename(norm_member)
                        if member_base.startswith("._"):
                            continue
                        if member_base == target_base:
                            matched_member = info.filename
                            break
                    if matched_member is None:
                        raise FileNotFoundError(
                            f"Không tìm thấy '{target_base}' bên trong gói nén {url}"
                        )
                    with zf.open(matched_member) as zsrc, open(tmp_dl, "wb") as zdst:
                        shutil.copyfileobj(zsrc, zdst)
            else:
                with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_dl, "wb") as out_f:
                    shutil.copyfileobj(resp, out_f)

            if not verify_weights_sha256(tmp_dl, expected_sha):
                raise RuntimeError(
                    f"Kiểm tra toàn vẹn SHA-256 thất bại khi tải trọng số {self.weights_path}"
                )
            os.replace(tmp_dl, self.weights_path)
        finally:
            for p in (tmp_dl, tmp_zip):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    def _load_model(self):
        """Load model weights and initialize inference on CUDA or CPU."""
        t0 = time.time()
        in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        reg_info = self._resolve_registry_entry()

        if not os.path.exists(self.weights_path):
            can_auto_dl = (
                not in_pytest
                and reg_info is not None
                and str(reg_info.get("source_url") or "").startswith("https://")
                and self._is_canonical_registered_path(reg_info)
            )
            if can_auto_dl:
                self._auto_download_registered_weights(reg_info)
            else:
                raise FileNotFoundError(f"Không tìm thấy file trọng số mô hình: {self.weights_path}")
        elif not in_pytest and reg_info is not None and reg_info.get("sha256") and self._is_canonical_registered_path(reg_info):
            from src.models_resolver import verify_weights_sha256
            st = os.stat(self.weights_path)
            cache_sig = (os.path.normcase(os.path.abspath(self.weights_path)), st.st_size, st.st_mtime_ns, str(reg_info["sha256"]).lower())
            verified_cache = getattr(GPUPersistentWorker, "_verified_weights_cache", None)
            if verified_cache is None:
                verified_cache = set()
                GPUPersistentWorker._verified_weights_cache = verified_cache
            if cache_sig not in verified_cache:
                if not verify_weights_sha256(self.weights_path, reg_info.get("sha256")):
                    raise RuntimeError(
                        f"Trọng số mô hình '{self.weights_path}' không khớp mã băm SHA-256 đã đăng ký!"
                    )
                verified_cache.add(cache_sig)

        eff_alpha = round(max(0.0, min(1.0, float(getattr(self, "blend_alpha", 0.0) or 0.0))), 4)
        sec_path = getattr(self, "secondary_weights_path", None)
        req_sha = str(getattr(self, "weights_sha256", "") or "").strip().lower()
        if not req_sha and reg_info is not None and reg_info.get("sha256") and eff_alpha <= 0.0:
            req_sha = str(reg_info["sha256"]).strip().lower()
        self.weights_sha256 = req_sha

        model = None
        use_spandrel = True
        if eff_alpha > 0.0 and sec_path:
            if not os.path.exists(sec_path):
                sec_reg = None
                try:
                    from src.models_resolver import load_models_registry
                    sec_reg = load_models_registry().get("realcugan-pro-denoise3x-2x")
                except Exception:
                    sec_reg = None
                if not in_pytest and isinstance(sec_reg, dict) and str(sec_reg.get("source_url") or "").startswith("https://"):
                    orig_wp = self.weights_path
                    try:
                        self.weights_path = sec_path
                        if self._is_canonical_registered_path(sec_reg):
                            self._auto_download_registered_weights(sec_reg)
                    finally:
                        self.weights_path = orig_wp
                if not os.path.exists(sec_path):
                    raise FileNotFoundError(f"Không tìm thấy file trọng số pha thứ cấp: {sec_path}")
            elif not in_pytest:
                try:
                    from src.models_resolver import load_models_registry, verify_weights_sha256
                    sec_reg = load_models_registry().get("realcugan-pro-denoise3x-2x")
                    if isinstance(sec_reg, dict) and sec_reg.get("sha256"):
                        if not verify_weights_sha256(sec_path, sec_reg.get("sha256")):
                            raise RuntimeError(
                                f"Trọng số mô hình '{sec_path}' không khớp mã băm SHA-256 đã đăng ký!"
                            )
                except RuntimeError:
                    raise
                except Exception:
                    pass
            blended_sd, blended_sha = blend_upcunet_state_dicts(
                self.weights_path,
                sec_path,
                eff_alpha,
                cons_sha=str((reg_info or {}).get("sha256") or "") or None,
            )
            self.weights_sha256 = req_sha or blended_sha
            import spandrel
            loader = spandrel.ModelLoader()
            desc = loader.load_from_state_dict(blended_sd)
            model_core = desc.model.to(self.device)
            scale = int(getattr(desc, "scale", 2))
            model = SpandrelModelWrapper(model_core, scale=scale, target_scale=2.0)
        else:
            try:
                import spandrel
                from src.build_trt_engine import _fold_span_conv3xc_fp32
                loader = spandrel.ModelLoader()
                desc = loader.load_from_file(self.weights_path)
                model_core = desc.model.eval()
                _fold_span_conv3xc_fp32(model_core)
                model_core = model_core.to(self.device)
                scale = int(getattr(desc, "scale", 2))
                model = SpandrelModelWrapper(model_core, scale=scale, target_scale=2.0)
            except Exception:
                use_spandrel = False

        if not use_spandrel or model is None:
            sd = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            weights = sd.get("params_ema", sd.get("params", sd))
            clean_weights = {k: v for k, v in weights.items() if k.startswith("body.")}

            arch_cfg = (reg_info or {}).get("arch", {})
            if self.model_id == "animejanai-hd-v3-compact" or (
                arch_cfg.get("type") == "SRVGGNetCompact" and int(arch_cfg.get("upscale", 4)) == 2
            ):
                nf = int(arch_cfg.get("num_feat", 64))
                nc = int(arch_cfg.get("num_conv", 16))
                model = AnimeJaNaiCompact2x(num_feat=nf, num_conv=nc).to(self.device)
            else:
                model = RealESRAnimeVideoV3().to(self.device)

            if self.precision == "fp16":
                clean_weights = {k: v.half() for k, v in clean_weights.items()}
            else:
                clean_weights = {k: v.float() for k, v in clean_weights.items()}

            model.load_state_dict(clean_weights, strict=True)

        if self.precision == "fp16":
            model = model.half()
        else:
            model = model.float()

        if self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        model.eval()

        # Warm up 1 batch to initialize kernel caches
        warm_b = min(self.batch_size, 2)
        dummy = torch.rand(warm_b, 3, 540, 960, device=self.device)
        if self.precision == "fp16":
            dummy = dummy.half()
        if self.device.type == "cuda":
            dummy = dummy.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            _ = model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        self.model = model
        self.last_load_time = time.time() - t0

    def restart_if_needed(self, logger=None) -> bool:
        """Trim CUDA reserved memory cache when exceeding DEFAULT_VRAM_CLEANUP_THRESHOLD_GB."""
        self.chunks_processed += 1
        if self.device.type == "cuda":
            reserved_gb = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            from src.config import DEFAULT_VRAM_CLEANUP_THRESHOLD_GB
            if reserved_gb >= DEFAULT_VRAM_CLEANUP_THRESHOLD_GB:
                torch.cuda.empty_cache()
                msg = (
                    f"🧹 [VRAM Garbage Collection] Đã dọn dẹp CUDA memory cache "
                    f"(Reserved: {reserved_gb:.2f} GB >= {DEFAULT_VRAM_CLEANUP_THRESHOLD_GB} GB)"
                )
                if logger:
                    logger.info(msg)
                return True
            return False

        if self.max_chunks_before_restart > 0 and self.chunks_processed >= self.max_chunks_before_restart:
            t0 = time.time()
            self.model = None
            self._load_model()
            restart_elapsed = time.time() - t0
            self.chunks_processed = 0
            self.total_restarts += 1
            msg = (
                f"♻️ [Worker Restart] Đã làm mới worker sau {self.max_chunks_before_restart} chunks "
                f"(Chi phí nạp lại: {restart_elapsed*1000:.1f} ms | Tổng lần restart: {self.total_restarts})"
            )
            if logger:
                logger.info(msg)
            return True
        return False

    def upscale_directory(
        self,
        input_dir: str,
        output_dir: str,
        fmt: str = "png",
        batch_size: Optional[int] = None,
        logger=None,
        mid_chunk_callback=None,
        dedup: bool = True,
    ) -> Dict[str, Any]:
        """Run batched SR inference on frames in input_dir and write outputs to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        img_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(f".{fmt}")
        ]) if os.path.exists(input_dir) else []
        if not img_files:
            return {"elapsed_seconds": 0.0, "fps": 0.0, "steady_fps": 0.0, "num_frames": 0}

        t_start = time.time()
        steady_start = None
        driver_alloc_peak = 0.0
        curr_alloc_peak = 0.0
        proc = psutil.Process()

        b_size = batch_size if batch_size is not None else self.batch_size

        # Parallel PNG writer pool
        writer_workers = min(12, max(4, os.cpu_count() or 4))
        writer_executor = ThreadPoolExecutor(max_workers=writer_workers)
        write_futures = []

        loader_workers = min(4, b_size)
        loader_executor = ThreadPoolExecutor(max_workers=loader_workers) if loader_workers > 1 else None

        def _read_img_bgr_u8(p: str) -> np.ndarray:
            im_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if im_bgr is not None:
                return im_bgr
            im = Image.open(p).convert("RGB")
            return np.ascontiguousarray(np.array(im, dtype=np.uint8)[:, :, ::-1])

        def _save_group_task(out_paths: List[str], bgr_array: np.ndarray):
            primary_path = out_paths[0]
            cv2.imwrite(primary_path, bgr_array, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            for dup_path in out_paths[1:]:
                try:
                    if os.path.exists(dup_path):
                        os.remove(dup_path)
                    os.link(primary_path, dup_path)
                except OSError:
                    shutil.copyfile(primary_path, dup_path)

        total_frames = len(img_files)
        target_dtype = torch.float16 if self.precision == "fp16" else torch.float32
        pinned_in: Optional[torch.Tensor] = None
        pinned_out_slots: List[Optional[torch.Tensor]] = [None, None]
        slot_futures: List[list] = [[], []]
        num_distinct = total_frames

        def _resolve_runner(in_h: int, in_w: int):
            if (
                self.device.type == "cuda"
                and self.precision == "fp16"
                and getattr(self, "use_tensorrt", False)
                and isinstance(self.model, nn.Module)
                and not os.environ.get("PYTEST_CURRENT_TEST")
            ):
                runners = getattr(self, "_trt_runners", None)
                if runners is None:
                    runners = {}
                    self._trt_runners = runners
                eff_sha = str(getattr(self, "weights_sha256", "") or "").strip().lower()
                eff_alpha = round(float(getattr(self, "blend_alpha", 0.0) or 0.0), 4)
                key = (int(in_h), int(in_w), int(b_size), eff_alpha, eff_sha)
                if key not in runners:
                    from src.build_trt_engine import get_hybrid_engine_or_wrapper
                    runners[key] = get_hybrid_engine_or_wrapper(
                        self.model,
                        max_batch=b_size,
                        force_pytorch=False,
                        model_id=getattr(self, "model_id", "realesr-animevideov3"),
                        input_hw=(in_h, in_w),
                        scale=2.0,
                        **_filter_supported_kwargs(
                            get_hybrid_engine_or_wrapper,
                            {"weights_sha256": eff_sha or None},
                        ),
                    )
                return runners[key]
            return self.model

        try:
            # Load chunk frames in parallel before cyclic deduplication
            if loader_executor and total_frames > 1:
                all_in_arrs = list(loader_executor.map(_read_img_bgr_u8, img_files))
            else:
                all_in_arrs = [_read_img_bgr_u8(p) for p in img_files]

            if dedup and total_frames > 1:
                from src.dedup import deduplicate_cyclic_frames
                distinct_arrs, frame_to_distinct = deduplicate_cyclic_frames(all_in_arrs)
            else:
                distinct_arrs = list(all_in_arrs)
                frame_to_distinct = list(range(total_frames))
            del all_in_arrs

            num_distinct = len(distinct_arrs)
            distinct_out_groups: List[List[str]] = [[] for _ in range(num_distinct)]
            distinct_first_names: List[str] = [""] * num_distinct
            for orig_idx, fpath in enumerate(img_files):
                d_idx = frame_to_distinct[orig_idx]
                fname = os.path.basename(fpath)
                if not distinct_first_names[d_idx]:
                    distinct_first_names[d_idx] = fname
                distinct_out_groups[d_idx].append(os.path.join(output_dir, fname))

            for b_start in range(0, num_distinct, b_size):
                b_end = min(b_start + b_size, num_distinct)
                batch_arrs = distinct_arrs[b_start:b_end]
                cur_b = len(batch_arrs)

                h0, w0 = batch_arrs[0].shape[:2]
                same_shape = all(a.shape[:2] == (h0, w0) for a in batch_arrs)

                if self.device.type == "cuda" and same_shape:
                    if pinned_in is None or pinned_in.shape[0] < cur_b or pinned_in.shape[1:3] != (h0, w0):
                        pinned_in = torch.empty((max(b_size, cur_b), h0, w0, 3), dtype=torch.uint8, pin_memory=True)
                    for i, arr in enumerate(batch_arrs):
                        pinned_in[i].copy_(torch.from_numpy(arr))
                    gpu_bgr = pinned_in[:cur_b].to(self.device, non_blocking=True)
                    # Convert BGR->RGB, permute NHWC->NCHW channels-last, and normalize on VRAM
                    tensor = (
                        gpu_bgr[..., [2, 1, 0]]
                        .permute(0, 3, 1, 2)
                        .to(memory_format=torch.channels_last, dtype=target_dtype)
                        .mul_(1.0 / 255.0)
                    )
                else:
                    batch_np = np.stack(batch_arrs, axis=0)
                    raw_t = torch.from_numpy(batch_np).to(self.device)
                    tensor = (
                        raw_t[..., [2, 1, 0]]
                        .permute(0, 3, 1, 2)
                        .contiguous()
                        .to(dtype=target_dtype)
                        .mul_(1.0 / 255.0)
                    )
                    if self.device.type == "cuda":
                        tensor = tensor.to(memory_format=torch.channels_last)

                # Drop host array references once batch tensors are staged on device
                for clear_i in range(b_start, b_end):
                    distinct_arrs[clear_i] = None

                active_runner = _resolve_runner(h0, w0) if same_shape else self.model

                # Run GPU inference and uint8 BGR quantization in VRAM with one D2H sync
                with torch.inference_mode():
                    out_tensor = active_runner(tensor)
                    finite_flag = torch.isfinite(out_tensor).all()
                    gpu_bgr_out = (
                        out_tensor.clamp(0.0, 1.0)
                        .mul(255.0)
                        .round()
                        .to(torch.uint8)[:, [2, 1, 0], :, :]
                        .permute(0, 2, 3, 1)
                        .contiguous()
                    )
                    slot = (b_start // b_size) % 2
                    if self.device.type == "cuda":
                        # Wait until previous PNG writers on this ring slot finish before overwriting
                        if slot_futures[slot]:
                            for prev_fut in slot_futures[slot]:
                                prev_fut.result()
                            slot_futures[slot].clear()
                        out_h, out_w = gpu_bgr_out.shape[1:3]
                        cur_pin_out = pinned_out_slots[slot]
                        if (
                            cur_pin_out is None
                            or cur_pin_out.shape[0] < cur_b
                            or cur_pin_out.shape[1:3] != (out_h, out_w)
                        ):
                            cur_pin_out = torch.empty(
                                (max(b_size, cur_b), out_h, out_w, 3),
                                dtype=torch.uint8,
                                pin_memory=True,
                            )
                            pinned_out_slots[slot] = cur_pin_out
                        cur_pin_out[:cur_b].copy_(gpu_bgr_out, non_blocking=True)
                        torch.cuda.current_stream(self.device).synchronize()
                        is_finite = bool(finite_flag.item())
                        out_bgr_u8 = cur_pin_out[:cur_b].numpy()
                    else:
                        is_finite = bool(finite_flag.item())
                        out_bgr_u8 = gpu_bgr_out.cpu().numpy()

                if b_start == 0:
                    steady_start = time.time()

                # NaN/Inf safety gate
                if not is_finite:
                    sample_name = distinct_first_names[b_start]
                    raise GPUError(f"Phát hiện giá trị bất thường (NaN/Inf) tại khung hình lô: '{sample_name}'")

                # Record peak VRAM allocation
                if self.device.type == "cuda":
                    c_alloc = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                    if c_alloc > curr_alloc_peak:
                        curr_alloc_peak = c_alloc
                    d_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                    if d_alloc > driver_alloc_peak:
                        driver_alloc_peak = d_alloc
                for b_idx in range(cur_b):
                    out_group = distinct_out_groups[b_start + b_idx]
                    fut = writer_executor.submit(_save_group_task, out_group, out_bgr_u8[b_idx])
                    write_futures.append(fut)
                    if self.device.type == "cuda":
                        slot_futures[slot].append(fut)

                if mid_chunk_callback and (b_start <= (num_distinct // 2) < b_end):
                    try:
                        mid_chunk_callback(b_start, num_distinct)
                    except Exception as e_cb:
                        if "No space left on device" in str(e_cb) or (isinstance(e_cb, OSError) and getattr(e_cb, "errno", None) == 28):
                            raise
                        if logger:
                            logger.warn(f"Lỗi mid_chunk_callback: {e_cb}")

            for f in write_futures:
                f.result()

        except Exception as e:
            writer_executor.shutdown(wait=False)
            if loader_executor:
                loader_executor.shutdown(wait=False)
            raise GPUError(f"Lỗi suy luận GPU tại frame: {e}") from e
        finally:
            writer_executor.shutdown(wait=True)
            if loader_executor:
                loader_executor.shutdown(wait=True)

        elapsed = time.time() - t_start
        steady_elapsed = (time.time() - steady_start) if (steady_start and len(img_files) > 1) else elapsed
        steady_count = (len(img_files) - 1) if len(img_files) > 1 else len(img_files)

        overall_fps = len(img_files) / elapsed if elapsed > 0 else 0.0
        steady_fps = steady_count / steady_elapsed if steady_elapsed > 0 else overall_fps

        rss_mb = proc.memory_info().rss / (1024 * 1024)

        return {
            "elapsed_seconds": elapsed,
            "fps": overall_fps,
            "steady_fps": steady_fps,
            "num_frames": len(img_files),
            "distinct_frames": num_distinct,
            "peak_rss_mb": rss_mb,
            "current_alloc_mb": curr_alloc_peak,
            "driver_alloc_mb": driver_alloc_peak
        }


_GLOBAL_GPU_WORKER: Optional[GPUPersistentWorker] = None

def _cleanup_gpu_worker(worker: Optional[GPUPersistentWorker]) -> None:
    """Release cached TRT runners, zero-disk ring buffers, and VRAM allocations."""
    if worker is None:
        return
    runners = getattr(worker, "_trt_runners", None)
    if isinstance(runners, dict):
        for runner in runners.values():
            for attr in ("_out_ring", "_batch_out_ring", "context", "engine", "runtime", "fallback_module"):
                if hasattr(runner, attr):
                    try:
                        setattr(runner, attr, None)
                    except Exception:
                        pass
        runners.clear()
    pareto_bundle = getattr(worker, "_auto_pareto_bundle", None)
    if isinstance(pareto_bundle, dict):
        fast_w = pareto_bundle.get("fast_worker")
        if fast_w is not None and fast_w is not worker:
            _cleanup_gpu_worker(fast_w)
        up_sh = pareto_bundle.get("shared_upscaler")
        if up_sh is not None and hasattr(up_sh, "release"):
            try:
                up_sh.release()
            except Exception:
                pass
        pareto_bundle.clear()
        worker._auto_pareto_bundle = None
    for attr in ("_zero_disk_upscaler", "_zero_disk_wrapper"):
        obj = getattr(worker, attr, None)
        if obj is not None:
            if hasattr(obj, "release"):
                try:
                    obj.release()
                except Exception:
                    pass
            for ring_attr in ("host_in_pins", "host_out_pins", "gpu_in_ring", "gpu_nhwc_ring", "color_conv", "model"):
                if hasattr(obj, ring_attr):
                    try:
                        setattr(obj, ring_attr, None)
                    except Exception:
                        pass
            try:
                setattr(worker, attr, None)
            except Exception:
                pass
    worker._zero_disk_key = None
    worker.model = None
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def get_gpu_worker(
    model_id: str,
    weights_path: str,
    precision: str = "fp16",
    max_chunks_before_restart: int = 10,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    use_tensorrt: bool = True,
    blend_alpha: float = 0.0,
    secondary_weights_path: Optional[str] = None,
    weights_sha256: Optional[str] = None,
) -> GPUPersistentWorker:
    """Return or initialize the singleton GPUPersistentWorker instance (keyed by model, weights, alpha, and SHA-256)."""
    global _GLOBAL_GPU_WORKER
    req_dev_type: Optional[str] = None
    if device:
        cand_type = torch.device(device).type
        if cand_type != "cuda" or torch.cuda.is_available():
            req_dev_type = cand_type
    else:
        req_dev_type = "cuda" if torch.cuda.is_available() else "cpu"

    eff_batch = int(batch_size) if (batch_size is not None and int(batch_size) > 0) else (
        1 if req_dev_type == "cpu" else DEFAULT_BATCH_SIZE
    )
    norm_alpha = round(max(0.0, min(1.0, float(blend_alpha or 0.0))), 4)
    req_sha = str(weights_sha256 or "").strip().lower()

    existing_weights = getattr(_GLOBAL_GPU_WORKER, "weights_path", None) if _GLOBAL_GPU_WORKER is not None else None
    existing_dev = getattr(_GLOBAL_GPU_WORKER, "device", None) if _GLOBAL_GPU_WORKER is not None else None
    existing_alpha = (
        round(float(getattr(_GLOBAL_GPU_WORKER, "blend_alpha", 0.0) or 0.0), 4)
        if _GLOBAL_GPU_WORKER is not None
        else 0.0
    )
    existing_sec = (
        getattr(_GLOBAL_GPU_WORKER, "secondary_weights_path", None)
        if _GLOBAL_GPU_WORKER is not None
        else None
    )
    existing_sha = (
        str(getattr(_GLOBAL_GPU_WORKER, "weights_sha256", "") or "").strip().lower()
        if _GLOBAL_GPU_WORKER is not None
        else ""
    )
    weights_mismatch = (
        existing_weights is not None
        and os.path.normcase(os.path.abspath(str(existing_weights)))
        != os.path.normcase(os.path.abspath(str(weights_path)))
    )
    sec_mismatch = (
        norm_alpha > 0.0
        and bool(secondary_weights_path)
        and os.path.normcase(os.path.abspath(str(existing_sec or "")))
        != os.path.normcase(os.path.abspath(str(secondary_weights_path)))
    )
    blend_or_sha_mismatch = (
        abs(existing_alpha - norm_alpha) > 1e-4
        or sec_mismatch
        or (bool(req_sha) and existing_sha != req_sha)
    )
    device_mismatch = (
        req_dev_type is not None
        and existing_dev is not None
        and getattr(existing_dev, "type", str(existing_dev)) != req_dev_type
    )
    if (
        _GLOBAL_GPU_WORKER is None
        or _GLOBAL_GPU_WORKER.model_id != model_id
        or weights_mismatch
        or blend_or_sha_mismatch
        or device_mismatch
        or _GLOBAL_GPU_WORKER.precision != precision.lower()
        or _GLOBAL_GPU_WORKER.batch_size != eff_batch
    ):
        old_worker = _GLOBAL_GPU_WORKER
        _GLOBAL_GPU_WORKER = None
        _cleanup_gpu_worker(old_worker)
        del old_worker
        _GLOBAL_GPU_WORKER = GPUPersistentWorker(
            model_id=model_id,
            weights_path=weights_path,
            precision=precision,
            max_chunks_before_restart=max_chunks_before_restart,
            device=device,
            batch_size=eff_batch,
            use_tensorrt=use_tensorrt,
            **_filter_supported_kwargs(
                GPUPersistentWorker,
                {
                    "blend_alpha": norm_alpha,
                    "secondary_weights_path": secondary_weights_path,
                    "weights_sha256": req_sha or None,
                },
            ),
        )
        if not hasattr(_GLOBAL_GPU_WORKER, "blend_alpha"):
            _GLOBAL_GPU_WORKER.blend_alpha = norm_alpha
        if not hasattr(_GLOBAL_GPU_WORKER, "secondary_weights_path"):
            _GLOBAL_GPU_WORKER.secondary_weights_path = secondary_weights_path
        if not hasattr(_GLOBAL_GPU_WORKER, "weights_sha256"):
            _GLOBAL_GPU_WORKER.weights_sha256 = req_sha or None
    else:
        _GLOBAL_GPU_WORKER.use_tensorrt = bool(use_tensorrt)
    return _GLOBAL_GPU_WORKER


def get_cuda_inference_engine(
    model_info: Dict[str, Any],
    precision: str = "fp16",
    batch_size: Optional[int] = None,
    use_tensorrt: bool = True,
) -> GPUPersistentWorker:
    """Return a CUDA persistent worker configured for model_info (including episode alpha-blend)."""
    eff_sha = str(model_info.get("blended_weights_sha256") or model_info.get("sha256") or "").strip().lower()
    return get_gpu_worker(
        model_id=model_info["id"],
        weights_path=model_info["weights_path"],
        precision=precision,
        device="cuda",
        batch_size=batch_size,
        use_tensorrt=use_tensorrt,
        **_filter_supported_kwargs(
            get_gpu_worker,
            {
                "blend_alpha": float(model_info.get("blend_alpha", 0.0) or 0.0),
                "secondary_weights_path": model_info.get("blend_secondary_weights_path"),
                "weights_sha256": eff_sha or None,
            },
        ),
    )


def upscale_frames_with_fallback(
    input_dir: str,
    output_dir: str,
    model_info: Dict[str, Any],
    engine: Optional[str] = None,
    precision: str = "fp16",
    batch_size: Optional[int] = None,
    fmt: str = "png",
    logger=None,
    mid_chunk_callback=None,
    use_tensorrt: bool = True,
    dedup: bool = True,
) -> Dict[str, Any]:
    """Run chunk SR inference with FP16 -> FP32 -> NCNN fallback without changing model identity."""
    if engine is None:
        from src.config import DEFAULT_ENGINE
        engine = DEFAULT_ENGINE

    model_id = model_info["id"]
    weights_path = model_info["weights_path"]
    eff_alpha = float(model_info.get("blend_alpha", 0.0) or 0.0)
    sec_weights = model_info.get("blend_secondary_weights_path")
    eff_sha = str(model_info.get("blended_weights_sha256") or model_info.get("sha256") or "").strip().lower() or None

    if engine == "ncnn":
        from src.ai_engine import upscale_frames as ncnn_upscale
        ncnn_model_name = "realesr-animevideov3" if model_id == "animevideov3" else model_id
        return ncnn_upscale(
            input_dir=input_dir,
            output_dir=output_dir,
            model=ncnn_model_name,
            scale=2,
            fmt=fmt,
            tile_size=540,
            thread_cfg="1:2:2"
        )

    try:
        worker = get_gpu_worker(
            model_id=model_id,
            weights_path=weights_path,
            precision=precision,
            device=engine if engine in ["cuda", "cpu"] else None,
            batch_size=batch_size,
            use_tensorrt=use_tensorrt,
            **_filter_supported_kwargs(
                get_gpu_worker,
                {
                    "blend_alpha": eff_alpha,
                    "secondary_weights_path": sec_weights,
                    "weights_sha256": eff_sha,
                },
            ),
        )
        res = worker.upscale_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            fmt=fmt,
            batch_size=batch_size,
            logger=logger,
            mid_chunk_callback=mid_chunk_callback,
            dedup=dedup,
        )
        worker.restart_if_needed(logger=logger)
        res["engine_used"] = f"{engine}_{precision}"
        return res

    except Exception as e:
        warn_msg = f"⚠️ [GPU Fallback Activated] Gặp lỗi khi suy luận {engine} {precision.upper()}: {e}."
        if logger:
            logger.warn(warn_msg, to_console=True)
        else:
            print(warn_msg)

        # Fall back from FP16 to FP32 on the same device
        if precision == "fp16":
            fallback_msg = f"🔄 Chuyển sang {engine.upper()} FP32 cho model {model_id}..."
            if logger:
                logger.info(fallback_msg, to_console=True)
            else:
                print(fallback_msg)
            worker_fp32 = get_gpu_worker(
                model_id=model_id,
                weights_path=weights_path,
                precision="fp32",
                device=engine if engine in ["cuda", "cpu"] else None,
                batch_size=batch_size,
                use_tensorrt=False,
                **_filter_supported_kwargs(
                    get_gpu_worker,
                    {
                        "blend_alpha": eff_alpha,
                        "secondary_weights_path": sec_weights,
                        "weights_sha256": eff_sha,
                    },
                ),
            )
            res = worker_fp32.upscale_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                fmt=fmt,
                batch_size=batch_size,
                logger=logger,
                mid_chunk_callback=mid_chunk_callback,
                dedup=dedup,
            )
            worker_fp32.restart_if_needed(logger=logger)
            res["engine_used"] = f"{engine}_fp32_fallback"
            return res

        if os.path.exists(REALESRGAN_BIN) or os.path.exists(REALESRGAN_BIN + ".exe"):
            if model_id == "animevideov3":
                fallback_msg = "🔄 Chuyển sang NCNN Vulkan (-t 540 -j 1:2:2) cho RealESR-AnimeVideoV3..."
                if logger:
                    logger.info(fallback_msg, to_console=True)
                else:
                    print(fallback_msg)
                from src.ai_engine import upscale_frames as ncnn_upscale
                res = ncnn_upscale(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    model="realesr-animevideov3",
                    scale=2,
                    fmt=fmt,
                    tile_size=540,
                    thread_cfg="1:2:2"
                )
                res["engine_used"] = "ncnn_fallback"
                return res

        raise RuntimeError(
            f"Lỗi suy luận PyTorch {engine.upper()} cho model '{model_id}' "
            f"(đã thử fallback FP32 nhưng không khả dụng): {e}"
        )


GPUEngineError = GPUError
