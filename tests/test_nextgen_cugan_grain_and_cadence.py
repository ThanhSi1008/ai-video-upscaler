"""Offline unit and integration tests for Real-CUGAN Pro TRT export, grain restoration, Triton deband/anti-ring, and 60Hz cadence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import onnx
import pytest
import spandrel
import torch
import torch.nn as nn

import src.build_trt_engine as trt_mod
from src.build_trt_engine import (
    TRTRuntimeEngineWrapper,
    _UpCunet2xTRTExportModule,
    export_model_to_onnx,
    is_upcunet2x_model,
    resolve_trt_profile_max_batch,
)
import src.cuda_engine as cuda_mod
from src.cuda_engine import (
    SpandrelModelWrapper,
    blend_upcunet_state_dicts,
    compute_blended_weights_sha256,
    get_gpu_worker,
    resolve_episode_model_blend,
)
from src.dedup import (
    PanZoomCadenceTracker,
    _build_fused_mpdecimate_dis_filtergraph,
    _check_8x8_lip_flap_gate,
    deduplicate_cyclic_frames,
    load_or_create_plan,
)
from src.ffmpeg_ops import (
    _apply_symmetric_s5_p010_filter,
    is_8bit_pix_fmt,
    resolve_effective_grain_profile,
)
from src.models_resolver import list_models_formatted, load_models_registry, resolve_model_for_video
from src.trt_zero_copy_upscaler import (
    DoubleBufferedPipeUpscaler,
    GPUColorConverterP010LE,
    _yuv420_to_nchw_fp16_bt709,
)
import upscale
import web.app as web_app
from web.job_manager import JobManager, _find_cached_plan


pytestmark = pytest.mark.unit


def _build_upcunet2x(pro: bool = True) -> nn.Module:
    from spandrel.architectures.RealCUGAN.__arch.upcunet_v3 import UpCunet2x

    torch.manual_seed(7)
    model = UpCunet2x(in_channels=3, out_channels=3, pro=pro).eval()
    return model


@pytest.mark.parametrize("pro_flag", [True, False])
@pytest.mark.parametrize("hw", [(80, 120), (82, 122), (81, 121)])
def test_upcunet2x_trt_export_module_exact_parity_across_even_mod4_and_odd_crops(pro_flag, hw):
    """Verify _UpCunet2xTRTExportModule matches spandrel UpCunet2x across %4==0, %4==2 (802p), and odd dims."""
    ref_model = _build_upcunet2x(pro=pro_flag)
    wrapped = _UpCunet2xTRTExportModule(SpandrelModelWrapper(ref_model, scale=2, target_scale=2.0)).eval()
    assert is_upcunet2x_model(ref_model) is True
    assert wrapped.is_pro is pro_flag

    torch.manual_seed(19)
    x = torch.rand(1, 3, hw[0], hw[1], dtype=torch.float32)
    with torch.no_grad():
        out_ref = ref_model(x)
        out_wrap = wrapped(x)

    assert out_wrap.shape == (1, 3, hw[0] * 2, hw[1] * 2)
    max_diff = float((out_ref - out_wrap).abs().max().item())
    assert max_diff <= 1e-5, f"Mismatch max_diff={max_diff} for pro={pro_flag}, hw={hw}"


def test_upcunet2x_onnx_export_has_zero_pad_nodes_and_batch_profile_cap(tmp_path: Path, capsys):
    """Verify UpCunet2x ONNX graph has 0 Pad/ConvTranspose ops, keeps b1-2 at <=800x1920, and caps >800x1920 to b1-1."""

    ref_model = _build_upcunet2x(pro=True)
    assert resolve_trt_profile_max_batch(ref_model, max_batch=8, input_hw=(720, 1280), available_vram_gb=24.0) == 2
    assert resolve_trt_profile_max_batch(ref_model, max_batch=8, input_hw=(800, 1920), available_vram_gb=24.0) == 2
    assert resolve_trt_profile_max_batch(ref_model, max_batch=1, input_hw=(800, 1920), available_vram_gb=24.0) == 1
    assert resolve_trt_profile_max_batch(ref_model, max_batch=8, input_hw=(801, 1920), available_vram_gb=24.0) == 1
    assert resolve_trt_profile_max_batch(ref_model, max_batch=8, input_hw=(1080, 1920), available_vram_gb=24.0) == 1

    assert trt_mod.main(["--dry-run", "--model", "realcugan-pro-conservative-2x", "--width", "1920", "--height", "800"]) == 0
    out_800 = capsys.readouterr().out
    assert "batch=1..2" in out_800 and "_800x1920_b1-2.engine" in out_800

    assert trt_mod.main(["--dry-run", "--model", "realcugan-pro-conservative-2x", "--width", "1920", "--height", "1080"]) == 0
    out_1080 = capsys.readouterr().out
    assert "batch=1..1" in out_1080 and "_1080x1920_b1-1.engine" in out_1080

    onnx_path = tmp_path / "upcunet2x_800p.onnx"
    export_model_to_onnx(ref_model, str(onnx_path), max_batch=2, dummy_shape=(80, 120))
    graph = onnx.load(str(onnx_path)).graph
    op_types = [node.op_type for node in graph.node]
    assert "Pad" not in op_types, f"Unexpected Pad op in ONNX graph: {op_types}"
    assert "ConvTranspose" not in op_types, f"Unexpected ConvTranspose op in ONNX graph: {op_types}"


def test_upcunet2x_seblock_fp32_reduction_prevents_fp16_overflow():
    """Verify _UpCunet2xTRTExportModule._se_forward stays finite in FP16 at high spatial activation = 50.0."""
    from spandrel.architectures.RealCUGAN.__arch.upcunet_v3 import SEBlock

    torch.manual_seed(11)
    se = SEBlock(64, reduction=8, bias=True).half().eval()
    # 50.0 * 540 * 960 = 25.92M >> 65504 FP16 max; naive FP16 sum overflows to Inf.
    z_high = torch.full((1, 64, 540, 960), 50.0, dtype=torch.float16)
    with torch.no_grad():
        out = _UpCunet2xTRTExportModule._se_forward(se, z_high)
    assert out.dtype == torch.float16
    assert bool(torch.isfinite(out).all().item()) is True


def test_blend_upcunet_state_dicts_and_singleton_worker_switching(tmp_path: Path, monkeypatch):
    """Verify episode alpha-blending preserves 'pro' buffer and switches get_gpu_worker singleton per alpha."""
    m_cons = _build_upcunet2x(pro=True)
    m_dn3x = _build_upcunet2x(pro=True)
    with torch.no_grad():
        for p in m_dn3x.parameters():
            p.add_(0.2)

    cons_p = tmp_path / "cons.pth"
    dn3x_p = tmp_path / "dn3x.pth"
    torch.save(m_cons.state_dict(), cons_p)
    torch.save(m_dn3x.state_dict(), dn3x_p)

    sd_mid, sha_mid = blend_upcunet_state_dicts(str(cons_p), str(dn3x_p), alpha=0.45)
    sd_clean, sha_clean = blend_upcunet_state_dicts(str(cons_p), str(dn3x_p), alpha=0.0)
    sd_heavy, sha_heavy = blend_upcunet_state_dicts(str(cons_p), str(dn3x_p), alpha=1.0)

    assert "pro" in sd_mid
    assert len(sha_mid) == 64 and sha_mid != sha_clean and sha_mid != sha_heavy
    desc_mid = spandrel.ModelLoader(device="cpu").load_from_state_dict(sd_mid)
    assert getattr(desc_mid.model, "is_pro", False) is True

    expected_w = torch.lerp(
        m_cons.state_dict()["unet1.conv3.weight"].float(),
        m_dn3x.state_dict()["unet1.conv3.weight"].float(),
        0.45,
    )
    assert torch.allclose(sd_mid["unet1.conv3.weight"], expected_w, atol=1e-6)

    # Disjoint-basin checkpoints (independent seeds, cos ~ 0) must fold UNet2 conv4_up u2_alpha=0.85
    # rather than scrambling unaligned hidden channels with naive weight-space lerp.
    from spandrel.architectures.RealCUGAN.__arch.upcunet_v3 import UpCunet2x

    torch.manual_seed(99)
    m_disjoint = UpCunet2x(in_channels=3, out_channels=3, pro=True).eval()
    disjoint_p = tmp_path / "dn3x_disjoint.pth"
    torch.save(m_disjoint.state_dict(), disjoint_p)
    sd_fold, _ = blend_upcunet_state_dicts(str(cons_p), str(disjoint_p), alpha=0.45)
    assert torch.equal(sd_fold["unet1.conv3.weight"], m_cons.state_dict()["unet1.conv3.weight"])
    assert torch.allclose(
        sd_fold["unet2.conv4_up.weight"].float(),
        m_cons.state_dict()["unet2.conv4_up.weight"].float() * 0.85,
        atol=1e-6,
    )
    m_fold = spandrel.ModelLoader(device="cpu").load_from_state_dict(sd_fold).model.eval().float()
    x_probe = torch.rand(1, 3, 48, 48)
    with torch.no_grad():
        assert float((m_cons(x_probe, alpha=0.85) - m_fold(x_probe)).abs().max().item()) <= 1e-4

    # Verify get_gpu_worker replaces singleton cleanly when queue switches clean -> light -> heavy -> clean.
    monkeypatch.setattr(cuda_mod, "_GLOBAL_GPU_WORKER", None)
    w0 = get_gpu_worker(
        "realcugan-pro-conservative-2x",
        str(cons_p),
        precision="fp32",
        device="cpu",
        use_tensorrt=False,
        blend_alpha=0.0,
        weights_sha256=sha_clean,
    )
    w0_same = get_gpu_worker(
        "realcugan-pro-conservative-2x",
        str(cons_p),
        precision="fp32",
        device="cpu",
        use_tensorrt=False,
        blend_alpha=0.0,
        weights_sha256=sha_clean,
    )
    assert w0_same is w0

    w_light = get_gpu_worker(
        "realcugan-pro-conservative-2x",
        str(cons_p),
        precision="fp32",
        device="cpu",
        use_tensorrt=False,
        blend_alpha=0.45,
        secondary_weights_path=str(dn3x_p),
        weights_sha256=sha_mid,
    )
    assert w_light is not w0
    assert w_light.blend_alpha == pytest.approx(0.45)
    assert w_light.weights_sha256 == sha_mid

    w_heavy = get_gpu_worker(
        "realcugan-pro-conservative-2x",
        str(cons_p),
        precision="fp32",
        device="cpu",
        use_tensorrt=False,
        blend_alpha=1.0,
        secondary_weights_path=str(dn3x_p),
        weights_sha256=sha_heavy,
    )
    assert w_heavy is not w_light
    assert w_heavy.blend_alpha == pytest.approx(1.0)
    assert w_heavy.weights_sha256 == sha_heavy
    cuda_mod._cleanup_gpu_worker(w_heavy)
    monkeypatch.setattr(cuda_mod, "_GLOBAL_GPU_WORKER", None)


def test_registry_and_episode_blend_routing_for_new_models():
    """Verify models.yaml registers Real-CUGAN Pro 2x & ModernSpanimationV2 with valid SHA-256 and episode blend."""
    reg = load_models_registry()
    for mid in ("realcugan-pro-conservative-2x", "realcugan-pro-denoise3x-2x", "modernspanimation-v2-2x"):
        assert mid in reg
        assert len(str(reg[mid]["sha256"])) == 64
        assert str(reg[mid]["source_url"]).startswith("https://")

    formatted = list_models_formatted()
    assert "realcugan-pro-conservative-2x" in formatted
    assert "modernspanimation-v2-2x" in formatted

    base_info = resolve_model_for_video("sample.mkv", "realcugan-pro-conservative-2x")
    clean_info = resolve_episode_model_blend(base_info, grain_tier="clean")
    light_info = resolve_episode_model_blend(base_info, grain_tier="light")
    heavy_info = resolve_episode_model_blend(base_info, grain_tier="heavy")

    assert clean_info["blend_alpha"] == 0.0
    assert light_info["blend_alpha"] == pytest.approx(0.45)
    assert heavy_info["blend_alpha"] == pytest.approx(1.0)
    assert len({clean_info["sha256"], light_info["sha256"], heavy_info["sha256"]}) == 3
    assert light_info["sha256"] == compute_blended_weights_sha256(
        reg["realcugan-pro-conservative-2x"]["sha256"],
        reg["realcugan-pro-denoise3x-2x"]["sha256"],
        0.45,
    )


def test_split_vdec_atadenoise_and_weber_dark_blink_preservation():
    """Verify [vdec] and tier-aware [vgray] filtergraphs preserve dark blinks while suppressing heavy grain."""
    fc_clean = _build_fused_mpdecimate_dis_filtergraph(480, 270, 768, 320, 0.33, grain_tier="clean")
    fc_light = _build_fused_mpdecimate_dis_filtergraph(480, 270, 768, 320, 0.33, grain_tier="light")
    fc_heavy = _build_fused_mpdecimate_dis_filtergraph(480, 270, 768, 320, 0.33, grain_tier="heavy")

    assert "scale=480:270:flags=area,format=gray,split=2[vdec][vgray]" in fc_light
    assert "[vdec]atadenoise=" in fc_light
    assert "[vgray]null[outgray]" in fc_light
    assert "[vgray]null[outgray]" in fc_clean
    assert "atadenoise" not in fc_clean
    assert "[vdec]atadenoise=" in fc_heavy
    assert "[vgray]removegrain=m0=3,atadenoise=0a=0.02:0b=0.04:s=5[outgray]" in fc_heavy

    # Dark night scene (Y=22) with subtle +5 LSB 4x4 eye blink must trigger Weber-scaled gate on clean/light.
    dark_a = np.full((270, 480), 22, dtype=np.uint8)
    dark_b = dark_a.copy()
    dark_b[100:104, 200:204] = 27
    hit, _ = _check_8x8_lip_flap_gate(dark_a, dark_b)
    assert hit is True

    # Under grain_tier='heavy', dY=+8 LSB dark 4x4 blinks (both aligned & corner-straddling) pass 100%,
    # while residual +/-3 LSB heavy-grain clumps after [vgray] filtering are rejected.
    dark_b8_aligned = dark_a.copy()
    dark_b8_aligned[100:104, 200:204] = 30
    dark_b8_straddle = dark_a.copy()
    dark_b8_straddle[102:106, 202:206] = 30
    assert _check_8x8_lip_flap_gate(dark_a, dark_b8_aligned, grain_tier="heavy")[0] is True
    assert _check_8x8_lip_flap_gate(dark_a, dark_b8_straddle, grain_tier="heavy")[0] is True

    dark_grain_clump = dark_a.copy()
    dark_grain_clump[80:83, 160:163] = 25
    assert _check_8x8_lip_flap_gate(dark_a, dark_grain_clump, grain_tier="heavy")[0] is False


def test_open_gop_symmetric_s5_filter_smooths_grain_and_preserves_dark_blink():
    """Verify _apply_symmetric_s5_p010_filter smooths grain on R_u=1 frames while preserving +8 LSB dark blinks."""
    rng = np.random.default_rng(2026)
    h, w = 96, 128
    semi_h = (h * 3) // 2
    frames_u8 = []
    for idx in range(7):
        y = np.clip(32.0 + rng.normal(0.0, 2.5, size=(h, w)), 16, 235).astype(np.uint8)
        if idx == 3:
            # Dark eye blink (+8 LSB in 8-bit == +32 in 10-bit) on center frame only
            y[40:52, 50:70] = 44
        uv = np.full((h // 2, w), 128, dtype=np.uint8)
        frames_u8.append(np.vstack([y, uv]))

    # Even when kept_indices=[3] (R_u=1), full_frames window [1..5] smooths background grain into p010le!
    filtered = _apply_symmetric_s5_p010_filter(
        full_frames=frames_u8,
        target_indices=[3],
        is_10bit_src=False,
        grain_tier="heavy",
        luma_h=h,
    )
    assert len(filtered) == 1
    out_p010 = filtered[0]
    assert out_p010.shape == (semi_h, w)
    assert out_p010.dtype == np.uint16

    raw_y_bg = frames_u8[3][:30, :].astype(np.float32)
    flt_y_bg = (out_p010[:30, :] >> 6).astype(np.float32) * 0.25
    assert float(np.std(flt_y_bg)) < float(np.std(raw_y_bg)) * 0.65

    # Blink patch [40:52, 50:70] (+12 LSB above Y=32 -> +48 in 10-bit > thra=25) must retain full amplitude!
    flt_blink_mean = float(np.mean((out_p010[42:50, 52:68] >> 6).astype(np.float32) * 0.25))
    assert flt_blink_mean >= 43.0


def test_uv_pair_safe_pre_deband_and_promoted_p010le_8bit_flag():
    """Verify PRE_DEBAND never swaps U<->V, smooths 1-LSB steps on promoted p010le, and protects watercolor texture."""
    assert is_8bit_pix_fmt("yuv420p") is True
    assert is_8bit_pix_fmt("yuv420p10le") is False

    B, H, W = 1, 64, 64
    semi_h = (H * 3) // 2
    # Construct promoted p010le from 8-bit staircase (left half Y=40, right half Y=41 -> 160<<6 vs 164<<6)
    # with strong pure-blue chroma (U=210 -> 840<<6, V=60 -> 240<<6) to detect any Cb<->Cr swap!
    yuv_p010 = np.zeros((B, semi_h, W), dtype=np.uint16)
    y_u8 = np.full((H, W), 40, dtype=np.uint16)
    y_u8[:, W // 2 :] = 41
    yuv_p010[0, :H, :] = (y_u8 << 2) << 6
    yuv_p010[0, H:, 0::2] = (210 << 2) << 6
    yuv_p010[0, H:, 1::2] = (60 << 2) << 6

    t_yuv = torch.from_numpy(yuv_p010.view(np.int16))
    rgb_raw = _yuv420_to_nchw_fp16_bt709(t_yuv, is_p010=True, out_dtype=torch.float32, use_triton=False, pre_deband=False)
    rgb_deb = _yuv420_to_nchw_fp16_bt709(t_yuv, is_p010=True, out_dtype=torch.float32, use_triton=False, pre_deband=True)

    # Blue channel must stay dominant (zero U<->V swap into Red!)
    assert float(rgb_deb[0, 2].mean()) > float(rgb_deb[0, 0].mean()) + 0.35
    # Across the staircase boundary (x=31 vs x=32), pre_deband smooths the jump!
    step_raw = float((rgb_raw[0, 1, :, 32] - rgb_raw[0, 1, :, 31]).abs().mean())
    step_deb = float((rgb_deb[0, 1, :, 32] - rgb_deb[0, 1, :, 31]).abs().mean())
    assert step_deb < step_raw

    # Watercolor brush texture (+/- 2.0 LSB checkerboard > 1.25 LSB gate, both harmonic & non-harmonic) must stay 100% unblurred!
    for brush_sz in (2, 3):
        wc_nv12 = np.zeros((1, semi_h, W), dtype=np.uint8)
        yy, xx = np.mgrid[0:H, 0:W]
        wc_nv12[0, :H, :] = np.where(((yy // brush_sz + xx // brush_sz) % 2) == 0, 122, 118).astype(np.uint8)
        wc_nv12[0, H:, :] = 128
        t_wc = torch.from_numpy(wc_nv12)
        wc_off = _yuv420_to_nchw_fp16_bt709(t_wc, is_p010=False, out_dtype=torch.float32, use_triton=False, pre_deband=False)
        wc_on = _yuv420_to_nchw_fp16_bt709(t_wc, is_p010=False, out_dtype=torch.float32, use_triton=False, pre_deband=True)
        assert torch.allclose(wc_off, wc_on, atol=1e-6)


def test_1080p_envelope_soft_antiring_nan_safety_and_1px_hair_protection():
    """Verify SOFT_ANTIRING suppresses artificial SR halos, sanitizes NaNs, and preserves 1px silver hair."""
    conv_ar = GPUColorConverterP010LE(device=torch.device("cpu"), chroma_siting="left", dither=False, soft_antiring=True)
    conv_no = GPUColorConverterP010LE(device=torch.device("cpu"), chroma_siting="left", dither=False, soft_antiring=False)

    H_lr, W_lr = 16, 16
    H_hr, W_hr = 32, 32
    lr = torch.full((1, 3, H_lr, W_lr), 0.50, dtype=torch.float32)
    # Dark ink line at x=4 (0.08) next to 0.50 background -> artificial SR white halo 0.92 at 4K x=10
    lr[:, :, :, 4] = 0.08
    # Real 1px bright silver hair at 1080p x=12 (0.92) -> must NOT be darkened at 4K x=24!
    lr[:, :, :, 12] = 0.92

    hr = lr.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    hr[:, :, :, 10] = 0.92  # Artificial ringing halo outside 1080p 3x3 envelope [0.08..0.50]
    hr[0, 0, 0, 0] = float("nan")
    lr_nan = lr.clone()
    lr_nan[0, 1, 2, 2] = float("nan")

    p010_no = (conv_no.rgb_to_p010le(hr, ref_lr_rgb=lr_nan).to(torch.int32) & 0xFFFF) >> 6
    p010_ar = (conv_ar.rgb_to_p010le(hr, ref_lr_rgb=lr_nan).to(torch.int32) & 0xFFFF) >> 6

    assert int(p010_ar.min()) >= 64 and int(p010_ar.max()) <= 960
    # Halo at 4K col 10 is pulled down significantly
    assert float(p010_ar[0, 8, 10]) < float(p010_no[0, 8, 10]) - 120.0
    # Genuine 1px silver hair at 4K col 24 stays 100% intact!
    assert int(p010_ar[0, 8, 24]) == int(p010_no[0, 8, 24])


def test_cv16uc1_dual_phase_cyclic_cel_absorbs_edge_ringing_but_keeps_8x6_mouth():
    """Verify 1080p p010le cyclic cel collapses L_inf=24 oscillatory ringing while keeping dark 8x6 mouth flaps."""
    h, w = 1080, 1920
    semi_h = (h * 3) // 2
    cel0 = np.empty((semi_h, w), dtype=np.uint16)
    cel0[:h, :] = (28 << 2) << 6  # Dark scene Y=28
    cel0[h:, :] = 512 << 6

    # Cel repeat with +/-24 LSB oscillatory B-frame edge ringing along a thin contour (zero-mean in 8x8)
    cel_ring = cel0.copy()
    row_osc = np.where((np.arange(w) % 2) == 0, 24, -20).astype(np.int32)
    cel_ring[300, :] = (((28 << 2) + row_osc) << 6).astype(np.uint16)
    d_ring, map_ring = deduplicate_cyclic_frames([cel0, cel_ring], allow_quant_noise=True)
    assert len(d_ring) == 1
    assert map_ring == [0, 0]

    # Dark 8x6 mouth flap (dY=+22 in 8-bit) straddling 4 corners of phase-(0,0) 8x8 cells at (540, 960)
    cel_mouth = cel0.copy()
    cel_mouth[537:543, 956:964] = ((50 << 2) << 6)
    d_mouth, map_mouth = deduplicate_cyclic_frames([cel0, cel_mouth], allow_quant_noise=True)
    assert len(d_mouth) == 2
    assert map_mouth == [0, 1]


def test_phase_locked_1s_pan_zoom_hysteresis_eliminates_slow_pan_judder():
    """Verify PanZoomCadenceTracker locks 1-1-1-1 cadence on 0.20 px/f slow pans and stays off on static grain."""
    import cv2

    h, w = 270, 480
    scale_factor = 1080.0 / h  # 4.0x -> 0.05 px/f at 270p == 0.20 px/f at 1080p (< 0.30 single-step thresh!)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base_tex = np.clip(120.0 + 45.0 * np.sin(xx * 0.11) * np.cos(yy * 0.09), 20, 230).astype(np.float32)

    tracker = PanZoomCadenceTracker(hold_frames=24)
    locks = []
    for t in range(12):
        shift_x = t * 0.055  # ~0.22 px/frame at 1080p
        m = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, 0.0]], dtype=np.float32)
        fr = cv2.warpAffine(base_tex, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.uint8)
        locks.append(tracker.update(fr, scale_factor=scale_factor))

    assert locks[3] is True
    assert all(locks[3:12]), f"Expected continuous 1-1-1-1 lock during slow pan, got {locks}"

    # Camera stops completely -> lock releases within 2 stationary frames
    stopped_frame = fr.copy()
    tracker.update(stopped_frame, scale_factor=scale_factor)
    assert tracker.update(stopped_frame, scale_factor=scale_factor) is False

    # Static textured scene with grain must not false-lock
    rng = np.random.default_rng(77)
    tracker.reset()
    grain_locks = [
        tracker.update(
            np.clip(base_tex + rng.normal(0.0, 1.5, size=(h, w)), 16, 235).astype(np.uint8),
            scale_factor=scale_factor,
        )
        for _ in range(16)
    ]
    assert not any(grain_locks)


def test_shared_batch_out_ring_and_web_studio_grain_restore_sync(tmp_path: Path):
    """Verify TRTRuntimeEngineWrapper shared ring reuse, plan.json grain_restore guard, and Web Studio CLI sync."""
    shared_ring: list[torch.Tensor] = []
    w_a = TRTRuntimeEngineWrapper.__new__(TRTRuntimeEngineWrapper)
    w_b = TRTRuntimeEngineWrapper.__new__(TRTRuntimeEngineWrapper)
    for w_inst in (w_a, w_b):
        w_inst.device = torch.device("cpu")
        w_inst.max_batch = 2
        w_inst.out_torch_dtype = torch.float16
        w_inst._out_ring = []
        w_inst._ring_idx = 0
        w_inst._shared_batch_out_ring = shared_ring
        w_inst._batch_out_ring = shared_ring
        w_inst._batch_ring_idx = 0
        w_inst._last_out_ptr = None

    buf_a = w_a._acquire_batch_output_buffer(8, 3, 64, 64)
    buf_b = w_b._acquire_batch_output_buffer(8, 3, 64, 64)
    assert len(shared_ring) == 2
    assert buf_a.data_ptr() == shared_ring[0].data_ptr()
    assert buf_b.data_ptr() == shared_ring[0].data_ptr()

    # CLI parser & plan.json grain_restore compatibility guard
    parsed = upscale.build_arg_parser().parse_args(["ep.mkv", "--grain-restore", "heavy"])
    assert parsed.grain_restore == "heavy"
    assert resolve_effective_grain_profile("nonexistent.mkv", grain_restore="off")["grain_tier"] == "clean"

    ep_file = tmp_path / "[SubsPlease] Sample - 01 (1080p).mkv"
    ep_file.write_bytes(b"\x00" * 512)
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / "models.yaml").write_text(
        Path("models/models.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "shows.yaml").write_text(Path("shows.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    temp_plan_dir = tmp_path / "out" / f".temp_{ep_file.stem}"
    temp_plan_dir.mkdir(parents=True, exist_ok=True)
    plan_file = temp_plan_dir / "plan.json"
    plan_file.write_text(
        '{"total_frames": 24, "total_kept_frames": 12, "dedup_policy": "dis-auto", '
        '"dedup_params": {"dis_thresh": 0.3, "grain_restore": "off", "grain_tier": "clean", "sigma_y": 0.0}, '
        '"chunks": [{"total_frames": 24}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="grain_restore"):
        load_or_create_plan(str(ep_file), str(plan_file), policy="dis-auto", grain_restore="heavy")
    assert _find_cached_plan(tmp_path, str(ep_file), expected_policy="dis-auto", expected_grain_restore="heavy") is None
    assert _find_cached_plan(tmp_path, str(ep_file), expected_policy="dis-auto", expected_grain_restore="off") is not None

    captured_cmds: list[list[str]] = []

    def _fake_popen(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        proc = MagicMock(pid=51515, stdout=iter([]))
        proc.wait.return_value = 0
        proc.poll.return_value = 0
        return proc

    jm = JobManager(tmp_path, python_executable="python", popen_factory=_fake_popen)
    jm.start_upscale_job({
        "inputs": [str(ep_file)],
        "grain_restore": "light",
        "dry_run": True,
    })
    assert "--grain-restore" in captured_cmds[-1]
    assert "light" in captured_cmds[-1]

    app = web_app.create_app(workspace_root=tmp_path, mock_mode=True)
    with TestClient(app) as client:
        cfg = client.get("/api/config/models-and-shows").json()
        assert cfg["grain_restore_policies"] == ["auto", "off", "light", "heavy"]
        model_ids = {m["id"] for m in cfg["models"]}
        assert "realcugan-pro-conservative-2x" in model_ids
        assert "modernspanimation-v2-2x" in model_ids


def test_pipe_upscaler_promoted_p010_8bit_flag_and_manifest_blend_conflict(tmp_path: Path):
    """Verify DoubleBufferedPipeUpscaler honors explicit source_is_8bit on p010le and pipeline guards manifest blend_alpha."""
    import io
    import json
    import src.pipeline as pipeline_mod

    seen_pre_deband: list[bool] = []
    orig_fn = pipeline_mod.run_upscale_pipeline  # noqa: F841
    import src.trt_zero_copy_upscaler as zcu

    real_yuv_fn = zcu._yuv420_to_nchw_fp16_bt709

    def _spy_yuv(*args, **kwargs):
        seen_pre_deband.append(bool(kwargs.get("pre_deband", False)))
        return real_yuv_fn(*args, **kwargs)

    upscaler = DoubleBufferedPipeUpscaler(
        lambda x: torch.nn.functional.interpolate(x, scale_factor=2.0),
        batch_size=2,
        device=torch.device("cpu"),
        input_hw=(32, 32),
        scale=2,
        source_is_8bit=True,
    )
    f_p010 = np.full((48, 32), 256 << 6, dtype=np.uint16)

    class _MockEnc:
        def __init__(self):
            self.stdin = io.BytesIO()

        def write(self, b):
            self.stdin.write(b)

    with patch.object(zcu, "_yuv420_to_nchw_fp16_bt709", side_effect=_spy_yuv):
        written = upscaler.process_frames_to_pipe([f_p010], [1], _MockEnc(), cyclic_dedup=False)
    assert written == 1
    assert seen_pre_deband == [True]

    # Verify manifest.json rejects mismatched blend_alpha / blended_weights_sha256 without --force
    src_mkv = tmp_path / "ep_grain.mkv"
    src_mkv.write_bytes(b"\x00" * 1024)
    out_dir = tmp_path / "out"
    temp_dir = out_dir / ".temp_ep_grain"
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "manifest.json").write_text(
        json.dumps({
            "model_id": "realcugan-pro-conservative-2x",
            "sha256": "a" * 64,
            "blended_weights_sha256": "a" * 64,
            "blend_alpha": 0.0,
        }),
        encoding="utf-8",
    )
    fake_meta = {
        "width": 1920,
        "height": 1080,
        "fps": 24.0,
        "fps_num": 24,
        "fps_den": 1,
        "duration": 4.0,
        "size_bytes": 1024,
        "codec": "hevc",
        "pix_fmt": "yuv420p10le",
        "profile": "Main 10",
        "nb_frames": 96,
        "source_is_8bit": False,
        "audio_count": 0,
        "subtitle_count": 0,
        "attachment_count": 0,
        "chapter_count": 0,
    }
    with (
        patch.object(pipeline_mod, "validate_environment"),
        patch.object(pipeline_mod, "probe_video", return_value=fake_meta),
        patch.object(pipeline_mod, "acquire_gpu_lock"),
        patch.object(pipeline_mod, "release_gpu_lock"),
        patch.object(pipeline_mod, "run_preflight_checks", return_value=True),
    ):
        with pytest.raises(ValueError, match="Xung đột mô hình"):
            pipeline_mod.run_upscale_pipeline(
                input_path=str(src_mkv),
                output_path=str(out_dir),
                model="realcugan-pro-conservative-2x",
                engine="cpu",
                grain_restore="light",
                force=False,
            )


def test_h264_open_gop_leading_b_frame_pts_detection(tmp_path: Path, monkeypatch):
    """Verify _count_leading_discard_packets detects H.264 Open-GOP leading B-frames (pts < kf_pts) without DISCARD flag."""
    import src.ffmpeg_ops as ff_mod

    seg = tmp_path / "h264_opengop.mkv"
    seg.write_bytes(b"fake")

    pkts = [
        SimpleNamespace(size=4096, pts=243, is_keyframe=True, is_discard=False),
        SimpleNamespace(size=512, pts=160, is_keyframe=False, is_discard=False),
        SimpleNamespace(size=512, pts=202, is_keyframe=False, is_discard=False),
        SimpleNamespace(size=1024, pts=327, is_keyframe=False, is_discard=False),
        SimpleNamespace(size=512, pts=285, is_keyframe=False, is_discard=False),
    ]

    class _FakeContainer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @property
        def streams(self):
            return SimpleNamespace(video=["v0"])

        def demux(self, _v):
            return iter(pkts)

    monkeypatch.setattr(ff_mod.av, "open", lambda _p: _FakeContainer())
    assert ff_mod._count_leading_discard_packets(str(seg)) == 2


def test_background_queue_preflight_forwards_grain_restore(tmp_path: Path, monkeypatch):
    """Verify BackgroundQueuePreflightManager forwards explicit --grain-restore to subprocess preflight and plan lookup."""
    import json
    import src.dedup as dedup_mod
    import src.scheduler as sched_mod

    ep1 = tmp_path / "ep01.mkv"
    ep2 = tmp_path / "ep02.mkv"
    ep1.write_bytes(b"ep1")
    ep2.write_bytes(b"ep2")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    captured_specs: list[tuple[list[str], str]] = []
    orig_build = dedup_mod.build_preflight_subprocess_spec

    def _spy_build(video_path, plan_file, segment_seconds=8, dedup_policy=None, grain_restore="auto"):
        cmd, kw = orig_build(video_path, plan_file, segment_seconds, dedup_policy=dedup_policy, grain_restore=grain_restore)
        captured_specs.append((cmd, grain_restore))
        Path(plan_file).parent.mkdir(parents=True, exist_ok=True)
        Path(plan_file).write_text(
            json.dumps(
                {
                    "source_path": str(video_path),
                    "dedup_policy": dedup_policy or "dis-auto",
                    "dedup_params": {"grain_restore": grain_restore, "grain_tier": "clean", "sigma_y": 0.0},
                    "total_frames": 24,
                    "total_kept_frames": 12,
                    "dedup_ratio_percent": 50.0,
                    "chunks": [{"chunk_index": 0, "kept_indices": [0], "total_frames": 24}],
                }
            ),
            encoding="utf-8",
        )
        return cmd, kw

    class _FakeProc:
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(dedup_mod, "build_preflight_subprocess_spec", _spy_build)
    monkeypatch.setattr(sched_mod.subprocess, "Popen", lambda *a, **kw: _FakeProc())

    mgr = sched_mod.BackgroundQueuePreflightManager(
        queue=[str(ep1), str(ep2)],
        output_dir=str(out_dir),
        segment_seconds=24,
        skip_first=True,
        dedup_policy="dis-auto",
        grain_restore="off",
    )
    mgr._worker([str(ep2)])
    assert len(captured_specs) == 1
    assert captured_specs[0][1] == "off"
    assert any("grain_restore='off'" in part or '"off"' in part or "off" in part for part in captured_specs[0][0])
    assert sched_mod.find_cached_plan(str(ep2), str(out_dir), expected_policy="dis-auto", expected_grain_restore="off") is not None
    assert sched_mod.find_cached_plan(str(ep2), str(out_dir), expected_policy="dis-auto", expected_grain_restore="heavy") is None
    assert sched_mod.find_cached_plan(str(ep2), str(out_dir), expected_policy="dis-auto", expected_grain_restore="auto") is None

    # Default --grain-restore auto must also reject stale grain_restore="off" plan.json across scheduler, web, and CLI ETA:
    monkeypatch.setattr(sched_mod, "probe_video", lambda _p: {"nb_frames": 480, "duration": 20.0, "fps": 24.0})
    sched_auto = sched_mod.compute_overnight_schedule(
        [str(ep2)],
        output_dir=str(out_dir),
        preflight_now=False,
        dedup_policy="dis-auto",
    )
    assert sched_auto["episodes"][0]["is_exact_plan"] is False

    import upscale as up_mod

    monkeypatch.setattr(up_mod, "probe_video", lambda _p: {"nb_frames": 480, "duration": 20.0, "fps": 24.0})
    jm = JobManager(tmp_path)
    monkeypatch.setattr("web.job_manager._probe_frames_lightweight", lambda _p: 480)
    web_sched = jm.compute_schedule_preview([str(ep2)], dedup_override="dis-auto", grain_restore=None, output_dir="out")
    assert web_sched["episodes"][0]["n_unique_source"] != "plan.json (đã quét)"

    # Re-running background preflight under default grain_restore="auto" must rescan ep2 instead of reusing "off" plan:
    mgr_auto = sched_mod.BackgroundQueuePreflightManager(
        queue=[str(ep1), str(ep2)],
        output_dir=str(out_dir),
        segment_seconds=24,
        skip_first=True,
        dedup_policy="dis-auto",
        grain_restore="auto",
    )
    mgr_auto._worker([str(ep2)])
    assert len(captured_specs) == 2
    assert captured_specs[1][1] == "auto"

    # check_deadline_and_eta must reject mismatched grain_restore in plan.json while accepting matching grain_restore:
    plan_ep2 = out_dir / ".temp_ep02" / "plan.json"
    plan_ep2.write_text(
        json.dumps(
            {
                "source_path": str(ep2),
                "dedup_policy": "dis-auto",
                "dedup_params": {"grain_restore": "off", "grain_tier": "clean", "sigma_y": 0.0},
                "total_frames": 30000,
                "total_kept_frames": 30000,
                "chunks": [{"chunk_index": 0, "kept_indices": [0], "total_frames": 30000}],
            }
        ),
        encoding="utf-8",
    )
    from datetime import datetime as _dt_cls

    class _FrozenDt(_dt_cls):
        @classmethod
        def now(cls, tz=None):
            return _dt_cls(2026, 11, 23, 23, 39, 0)

    monkeypatch.setattr(up_mod, "datetime", _FrozenDt)
    can_auto, _, _ = up_mod.check_deadline_and_eta(
        str(ep2), "23:59", output_dir=str(out_dir), dedup_policy="dis-auto", grain_restore="auto"
    )
    can_off, _, _ = up_mod.check_deadline_and_eta(
        str(ep2), "23:59", output_dir=str(out_dir), dedup_policy="dis-auto", grain_restore="off"
    )
    assert can_auto is True
    assert can_off is False


def test_trt_wrapper_fp32_io_uses_fp16_batch_ring_and_fixed_slice_cast_buffers(tmp_path: Path, monkeypatch):
    """Verify FP32-I/O TRT engines store B-frame output rings in FP16 and reuse constant slice cast pointers."""
    import ctypes
    import sys
    import types

    fake_trt = types.ModuleType("tensorrt")

    class _FakeDataType:
        FLOAT = "FLOAT"
        HALF = "HALF"

    class _FakeLogger:
        ERROR = 1

        def __init__(self, severity):
            self.severity = severity

    class _FakeContext:
        def __init__(self):
            self.shape_calls = 0
            self.addr_calls = {"input_rgb": 0, "output_rgb": 0}
            self.ptrs = {}
            self.last_shape = None
            self.executed_v3 = 0

        def set_input_shape(self, name, shape):
            self.shape_calls += 1
            self.last_shape = tuple(shape)

        def set_tensor_address(self, name, ptr):
            self.addr_calls[name] = self.addr_calls.get(name, 0) + 1
            self.ptrs[name] = int(ptr)

        def execute_async_v3(self, _stream_handle):
            self.executed_v3 += 1
            b, c, h, w = self.last_shape
            in_n = b * c * h * w
            out_n = b * c * (h * 2) * (w * 2)
            in_arr = np.ctypeslib.as_array((ctypes.c_float * in_n).from_address(self.ptrs["input_rgb"])).reshape(b, c, h, w)
            out_arr = np.ctypeslib.as_array((ctypes.c_float * out_n).from_address(self.ptrs["output_rgb"])).reshape(
                b, c, h * 2, w * 2
            )
            out_arr[...] = np.repeat(np.repeat(in_arr * 0.5 + 0.125, 2, axis=2), 2, axis=3)
            return True

    class _FakeEngine:
        def get_tensor_dtype(self, _name):
            return _FakeDataType.FLOAT

        def create_execution_context(self):
            return _FakeContext()

    class _FakeRuntime:
        def __init__(self, _logger):
            pass

        def deserialize_cuda_engine(self, _blob):
            return _FakeEngine()

    fake_trt.DataType = _FakeDataType
    fake_trt.Logger = _FakeLogger
    fake_trt.Runtime = _FakeRuntime

    eng_file = tmp_path / "fp32_io.engine"
    eng_file.write_bytes(b"FAKE_FP32_ENGINE")
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    wrapper = TRTRuntimeEngineWrapper(
        str(eng_file),
        device=torch.device("cpu"),
        input_hw=(8, 12),
        max_batch=1,
        scale=2,
    )
    assert wrapper.in_torch_dtype == torch.float32
    assert wrapper.out_torch_dtype == torch.float32

    x = torch.linspace(0.0, 1.0, 8 * 3 * 8 * 12, dtype=torch.float16).reshape(8, 3, 8, 12)
    y1 = wrapper(x)
    y2 = wrapper(x)
    y3 = wrapper(x)

    expected = torch.nn.functional.interpolate(x.float() * 0.5 + 0.125, scale_factor=2.0, mode="nearest").to(torch.float16)
    assert y1.dtype == torch.float16
    assert torch.equal(y1, expected)
    assert len(wrapper._batch_out_ring) == 2
    assert wrapper._batch_out_ring[0].dtype == torch.float16
    assert wrapper._out_cast_buf is not None and wrapper._out_cast_buf.shape == (1, 3, 16, 24)
    assert wrapper._in_cast_buf is not None and wrapper._in_cast_buf.shape == (1, 3, 8, 12)
    assert wrapper.context.executed_v3 == 24
    assert wrapper.context.shape_calls == 1
    assert wrapper.context.addr_calls["input_rgb"] == 1
    assert wrapper.context.addr_calls["output_rgb"] == 1
    assert y1.data_ptr() != y2.data_ptr()
    assert y3.data_ptr() == y1.data_ptr()

    wrapper.release()
    assert wrapper._in_cast_buf is None
    assert wrapper._out_cast_buf is None
    assert wrapper._batch_out_ring == []


def test_no_zero_disk_resolve_runner_evicts_prior_resolution_trt_context(tmp_path: Path, monkeypatch):
    """Verify GPUPersistentWorker.upscale_directory releases prior resolution TRT runner before building a new one."""
    import cv2
    import src.build_trt_engine as trt_b
    import src.cuda_engine as ce_mod

    released_keys: list[tuple[int, int]] = []

    class _SpyTRTRunner:
        def __init__(self, hw: tuple[int, int]):
            self.hw = hw

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

        def release(self) -> None:
            released_keys.append(self.hw)

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(
        trt_b,
        "get_hybrid_engine_or_wrapper",
        lambda _m, input_hw=(80, 120), **_kw: _SpyTRTRunner((int(input_hw[0]), int(input_hw[1]))),
    )

    worker = ce_mod.GPUPersistentWorker.__new__(ce_mod.GPUPersistentWorker)
    worker.model_id = "realcugan-pro-conservative-2x"
    worker.precision = "fp16"
    worker.batch_size = 2
    worker.use_tensorrt = True
    worker.blend_alpha = 0.45
    worker.weights_sha256 = "abcd1234"
    worker.device = type("_FakeCudaDev", (), {"type": "cuda"})()
    worker._trt_runners = {}
    worker.model = torch.nn.Identity()

    for idx, (h, w) in enumerate(((16, 24), (20, 28), (22, 30))):
        in_d = tmp_path / f"in_{idx}"
        out_d = tmp_path / f"out_{idx}"
        in_d.mkdir()
        cv2.imwrite(str(in_d / "0001.png"), np.full((h, w, 3), 64, dtype=np.uint8))
        res = worker.upscale_directory(str(in_d), str(out_d), fmt="png", batch_size=2, dedup=False)
        assert res["num_frames"] == 1
        assert len(worker._trt_runners) == 1

    assert released_keys == [(16, 24), (20, 28)]
    ce_mod._cleanup_gpu_worker(worker)
    assert released_keys == [(16, 24), (20, 28), (22, 30)]
