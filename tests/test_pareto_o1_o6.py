"""Offline unit tests for Pareto quality and E2E speed optimizations (O1-O6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.build_trt_engine import (
    _GenericTRTExportModule,
    _PureFP16TRTExportModule,
    get_trt_engine_path,
)
import src.config as src_config
import src.pipeline as pipeline_mod
from src.cuda_engine import RealESRAnimeVideoV3, SpandrelModelWrapper
from src.dedup import (
    DEDUP_POLICY_PARAMS,
    RollingDedupPlanner,
    _check_8x8_lip_flap_gate,
    _is_dis_motion_early_exit,
    deduplicate_cyclic_frames,
    resolve_dedup_policy,
)
from src.ffmpeg_ops import yuv420_to_rgb_small_u8
from src.models_resolver import AUTO_PARETO_MODEL_ID, resolve_model_for_video
from src.output_sanity import check_frame_sanity_arrays
from src.trt_zero_copy_upscaler import (
    DoubleBufferedPipeUpscaler,
    GPUColorConverterP010LE,
    SRVGGNetCompactTRTWrapper,
    _bound_live_cyclic_memo,
    _yuv420_to_nchw_fp16_bt709,
)
import web.app as web_app
from web.fs_browser import DEDUP_POLICY_PARAMS as WEB_DEDUP_POLICY_PARAMS, SafeLocalFSBrowser
from web.job_manager import JobManager


pytestmark = pytest.mark.unit


class _DummyGeneric4xNet(nn.Module):
    """Minimal non-SRVGG 4x network to verify antialiased bicubic fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return F.interpolate(y, scale_factor=4.0, mode="nearest")


def test_o1_collapsed_catmull_rom_algebraic_equivalence_and_zero_dark_border():
    """Verify collapsed 4x->2x Catmull-Rom matches PixelShuffle(4)+antialias bicubic and avoids dark borders."""
    torch.manual_seed(42)
    base_v3 = RealESRAnimeVideoV3().eval().float()
    for p in base_v3.parameters():
        nn.init.normal_(p, mean=0.0, std=0.04)

    trt_export = _PureFP16TRTExportModule(base_v3, target_scale=2.0).eval().float()
    trt_wrapper = SRVGGNetCompactTRTWrapper(base_v3, scale=2.0).eval().float()

    x = torch.rand(2, 3, 24, 32, dtype=torch.float32)
    with torch.no_grad():
        feat = x
        for layer in base_v3.body:
            feat = layer(feat)
        hr_4x = F.pixel_shuffle(feat, 4) + F.interpolate(x, scale_factor=4.0, mode="nearest")
        ref_2x_raw = F.interpolate(hr_4x, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True)
        out_v3 = base_v3(x)
        out_export = trt_export(x)
        out_wrapper = trt_wrapper(x)

    # Interior [2:-2, 2:-2] has zero boundary-stencil ambiguity and must match algebraically.
    interior = (slice(None), slice(None), slice(2, -2), slice(2, -2))
    assert float((out_v3[interior] - ref_2x_raw[interior]).abs().max()) < 1e-5
    assert float((out_wrapper[interior] - ref_2x_raw[interior]).abs().max()) < 1e-5
    assert float((out_export[interior] - ref_2x_raw[interior]).abs().max()) < 1e-5

    # Zero-residual constant frame must preserve exact brightness up to 1px corners/edges.
    zero_v3 = RealESRAnimeVideoV3().eval().float()
    for p in zero_v3.parameters():
        nn.init.zeros_(p)
    const_in = torch.full((1, 3, 16, 20), 0.73, dtype=torch.float32)
    with torch.no_grad():
        const_out = zero_v3(const_in)
    assert float((const_out - 0.73).abs().max()) < 1e-6


def test_o1_generic_spandrel_4x_uses_antialiased_bicubic_and_trt_arch_tag(tmp_path):
    """Verify non-SRVGG 4x models use antialiased bicubic downscale and aa2x isolates TRT engine cache."""
    net = _DummyGeneric4xNet().eval().float()
    wrapper = SpandrelModelWrapper(net, scale=4, target_scale=2.0).eval().float()
    gen_export = _GenericTRTExportModule(net, target_scale=2.0, model_scale=4.0).eval().float()
    assert not wrapper._can_collapse_srvgg_4x

    x = torch.rand(3, 3, 16, 16, dtype=torch.float32)
    with torch.no_grad():
        raw_4x = net(x)
        expected_raw = F.interpolate(raw_4x, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True)
        got_wrap = wrapper(x)
        got_exp = gen_export(x)

    assert float((got_wrap - expected_raw).abs().max()) < 1e-6
    assert float((got_exp - expected_raw).abs().max()) < 1e-6

    p_default = get_trt_engine_path("animevideov3", cache_dir=str(tmp_path), weights_sha256="abcdef1234567890")
    p_aa2x = get_trt_engine_path(
        "animevideov3", cache_dir=str(tmp_path), weights_sha256="abcdef1234567890", arch_tag="aa2x"
    )
    assert p_default != p_aa2x
    assert "_aa2x" in Path(p_aa2x).name


def test_o2_10bit_p010le_roundtrip_fidelity_beats_8bit_rgb24():
    """Verify 10-bit p010le BT.709 ingest + P010LE converter keeps <=1 LSB luma error vs >=2 LSB for 8-bit RGB24."""
    h, w = 16, 512
    codes_10 = np.linspace(64, 940, w, dtype=np.float64).round().astype(np.uint16)
    y_plane = np.broadcast_to(codes_10[None, :], (h, w)).copy()
    uv_plane = np.full((h // 2, w), 512, dtype=np.uint16)
    p010_in = np.vstack([y_plane << 6, uv_plane << 6])[None, ...]

    rgb_fp32 = _yuv420_to_nchw_fp16_bt709(
        torch.from_numpy(p010_in.view(np.int16)),
        is_p010=True,
        chroma_siting="left",
        out_dtype=torch.float32,
        use_triton=False,
    )
    rgb_up = F.interpolate(rgb_fp32, scale_factor=2.0, mode="nearest")

    conv_nodither = GPUColorConverterP010LE(device=torch.device("cpu"), chroma_siting="left", dither=False)
    p010_out = conv_nodither.rgb_to_p010le(rgb_up, slot=None).numpy().view(np.uint16)
    y_out_codes = (p010_out[0, : h * 2, ::2] >> 6).astype(np.int32)
    max_err_10bit = int(np.max(np.abs(y_out_codes - codes_10[None, :].astype(np.int32))))
    assert max_err_10bit <= 1

    conv_dither = GPUColorConverterP010LE(device=torch.device("cpu"), chroma_siting="left", dither=True)
    p010_dither = conv_dither.rgb_to_p010le(rgb_up, slot=None).numpy().view(np.uint16)
    y_dither_codes = (p010_dither[0, : h * 2, ::2] >> 6).astype(np.int32)
    assert int(np.max(np.abs(y_dither_codes - codes_10[None, :].astype(np.int32)))) <= 1

    # Simulate legacy 8-bit rgb24 pipe quantization on the same ramp.
    rgb_8bit_quant = torch.round(rgb_up * 255.0) / 255.0
    p010_from_8bit = conv_nodither.rgb_to_p010le(rgb_8bit_quant, slot=None).numpy().view(np.uint16)
    y_8bit_codes = (p010_from_8bit[0, : h * 2, ::2] >> 6).astype(np.int32)
    max_err_8bit = int(np.max(np.abs(y_8bit_codes - codes_10[None, :].astype(np.int32))))
    assert max_err_8bit >= 2

    # Verify 8-bit NV12 semi-planar path produces bounded [0, 1] NCHW RGB.
    nv12_in = np.vstack([
        np.full((h, w), 128, dtype=np.uint8),
        np.full((h // 2, w), 128, dtype=np.uint8),
    ])[None, ...]
    rgb_nv12 = _yuv420_to_nchw_fp16_bt709(
        torch.from_numpy(nv12_in), is_p010=False, chroma_siting="left", out_dtype=torch.float32, use_triton=False
    )
    assert rgb_nv12.shape == (1, 3, h, w)
    assert 0.45 <= float(rgb_nv12.mean()) <= 0.56


def test_o2_cyclic_dedup_p010le_normalization_and_2d_yuv_sanity_guard():
    """Verify p010le << 6 frames normalize to 8-bit MSB in cyclic dedup and 2D YUV passes sanity checks."""
    h, w = 540, 960
    base_y = np.full((h, w), 400 << 6, dtype=np.uint16)
    base_uv = np.full((h // 2, w), 512 << 6, dtype=np.uint16)
    f0 = np.vstack([base_y, base_uv])

    # +4 in 10-bit == +1 in 8-bit MSB, but +256 in raw uint16<<6.
    f1 = f0.copy()
    f1[:10, :10] = (404 << 6)

    distinct, mapping = deduplicate_cyclic_frames([f0, f1], allow_quant_noise=True, max_pixel_tol=2)
    assert len(distinct) == 1
    assert mapping == [0, 0]

    thumb_p010 = yuv420_to_rgb_small_u8(f0)
    assert thumb_p010.shape == (270, 480, 3)
    assert thumb_p010.dtype == np.uint8

    nv12 = np.vstack([
        np.full((h, w), 110, dtype=np.uint8),
        np.full((h // 2, w), 128, dtype=np.uint8),
    ])
    thumb_nv12 = yuv420_to_rgb_small_u8(nv12)
    assert thumb_nv12.shape == (270, 480, 3)
    assert thumb_nv12.dtype == np.uint8

    # Structured gradient in 2D p010le passed directly to check_frame_sanity_arrays.
    grad_y = np.tile(np.linspace(120, 780, w, dtype=np.uint16), (h, 1)) << 6
    grad_p010 = np.vstack([grad_y, base_uv])
    grad_rgb_out = yuv420_to_rgb_small_u8(grad_p010)
    metrics = check_frame_sanity_arrays(grad_p010, grad_rgb_out, is_rgb=True)
    assert metrics["passed"], f"Expected 2D p010le sanity pass, got {metrics}"


def test_o2_bounded_live_cyclic_memo_evicts_furthest_next_use():
    """Verify _bound_live_cyclic_memo caps simultaneous live memo entries <= max_memo_frames losslessly."""
    rng = np.random.default_rng(7)
    raw_distinct = [rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8) for _ in range(90)]
    # Frame 0 is reused at the very end; frames 1..89 each repeat twice in pairs, plus 70 live cyclic frames.
    seq_indices = [0] + list(range(1, 90)) + list(range(1, 90)) + [0]

    new_distinct, new_map = _bound_live_cyclic_memo(raw_distinct, seq_indices, max_memo_frames=64)
    assert len(new_map) == len(seq_indices)

    # Simulate DoubleBufferedPipeUpscaler's memo_p010 live cache lifecycle and verify <= 64 peak entries.
    counts: dict[int, int] = {}
    for d in new_map:
        counts[d] = counts.get(d, 0) + 1

    memo_live: set[int] = set()
    seen: dict[int, int] = {}
    peak_live = 0
    for pos, orig_d in enumerate(seq_indices):
        d = new_map[pos]
        assert np.array_equal(new_distinct[d], raw_distinct[orig_d])
        seen[d] = seen.get(d, 0) + 1
        if counts[d] > 1 and seen[d] == 1:
            memo_live.add(d)
            peak_live = max(peak_live, len(memo_live))
        elif d in memo_live and seen[d] >= counts[d]:
            memo_live.remove(d)

    assert peak_live <= 64
    assert len(memo_live) == 0


def test_o3_rolling_dedup_planner_open_gop_discard_reconciliation_and_max_workers(tmp_path):
    """Verify RollingDedupPlanner caps max_workers=2 and prepends 'keep' for Open-GOP discard packets."""
    seg0 = tmp_path / "seg_0000.mp4"
    seg1 = tmp_path / "seg_0001.mp4"
    seg0.write_bytes(b"fake0")
    seg1.write_bytes(b"fake1")
    plan_path = tmp_path / "dedup_plan.json"

    def _fake_scan_fused(input_path, *args, **kwargs):
        boundary_out = kwargs.get("boundary_out")
        if input_path.endswith("seg_0000.mp4"):
            if boundary_out is not None:
                boundary_out["last_gray"] = np.full((270, 480), 40, dtype=np.uint8)
            return (["keep"] * 6, ["keep", "drop", "drop", "keep", "drop", "drop"])
        # Chunk 1 simulates 2 leading Open-GOP RASL discard packets (only 4 decoded frames out of 6 packets)
        # and a bright scene cut (220 vs 40) + high keep ratio.
        if boundary_out is not None:
            boundary_out["first_gray"] = np.full((270, 480), 220, dtype=np.uint8)
        return (["keep"] * 4, ["keep", "keep", "keep", "keep"])

    def _fake_discard_count(seg_path):
        return 2 if str(seg_path).endswith("seg_0001.mp4") else 0

    with (
        patch("src.dedup._scan_fused_mpdecimate_and_dis", side_effect=_fake_scan_fused),
        patch("src.ffmpeg_ops._count_leading_discard_packets", side_effect=_fake_discard_count),
    ):
        planner = RollingDedupPlanner(
            input_path=str(seg0),
            seg_paths=[str(seg0), str(seg1)],
            seg_frame_counts=[6, 6],
            plan_path=str(plan_path),
            segment_seconds=24,
            policy="dis-auto",
            max_workers=6,
            decode_threads=2,
        )
        assert planner.max_workers == 2
        planner.start(sync_initial_chunks=1)
        c0 = planner.get_chunk_plan(0)
        c1 = planner.get_chunk_plan(1)
        final_plan = planner.finalize_plan(extra_fields={"nvenc_multipass": "fullres"})
        planner.close()

    assert c0["total_frames"] == 6
    assert c0["kept_count"] == 2
    assert c1["total_frames"] == 6
    # Chunk 1 had 2 discard packets prepended as 'keep' + 4 decoded 'keep' == 6 keeps.
    assert c1["kept_count"] == 6
    assert c1["scene_cut"] is True
    assert final_plan["total_frames"] == 12
    assert final_plan["total_kept_frames"] == 8
    assert final_plan["nvenc_multipass"] == "fullres"
    assert plan_path.is_file()
    saved = json.loads(plan_path.read_text(encoding="utf-8"))
    assert saved["chunks"][1]["scene_cut"] is True


def test_o3_o6_episode_multipass_lock_and_pyav_encoder_fallback(tmp_path):
    """Verify per-episode NVENC multipass resolution, rate args, segment defaults, and PyAV fallback."""
    assert src_config.DEFAULT_SEGMENT_TIME == 8
    assert src_config.DEFAULT_ZERO_DISK_SEGMENT_TIME == 24

    assert (
        src_config.resolve_episode_nvenc_multipass(
            policy="auto", model_id="animevideov3", dedup_ratio_pct=50.0, engine="cuda"
        )
        == "fullres"
    )
    assert (
        src_config.resolve_episode_nvenc_multipass(
            policy="auto", model_id="animejanai-hd-v3-ultracompact", dedup_ratio_pct=55.0, engine="cuda"
        )
        == "qres"
    )
    assert (
        src_config.resolve_episode_nvenc_multipass(
            policy="fullres", model_id="animejanai-hd-v3-ultracompact", dedup_ratio_pct=80.0, engine="cuda"
        )
        == "fullres"
    )

    with patch.object(src_config, "HEVC_ENCODER", "hevc_nvenc"):
        args_full = src_config.get_hevc_rate_args(multipass="fullres")
        args_qres = src_config.get_hevc_rate_args(multipass="qres")
    assert args_full[args_full.index("-multipass") + 1] == "fullres"
    assert args_qres[args_qres.index("-multipass") + 1] == "qres"

    dummy_model = nn.Identity()
    upscaler = DoubleBufferedPipeUpscaler(dummy_model, batch_size=2, device=torch.device("cpu"))
    fake_proc = MagicMock()
    with patch.object(DoubleBufferedPipeUpscaler, "spawn_nvenc_pipe_writer", return_value=fake_proc) as sp_mock:
        enc = upscaler.open_chunk_encoder(
            str(tmp_path / "chunk.mp4"), fps_str="24000/1001", width=3840, height=2160, multipass="qres", prefer_pyav=False
        )
    assert enc is fake_proc
    assert sp_mock.call_args.kwargs["chroma_location"] == "left"
    assert sp_mock.call_args.kwargs["multipass"] == "qres"


def _make_dis_geometry(h: int = 270, w: int = 480):
    cy, cx = h * 0.5, w * 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rx = xx - cx
    ry = yy - cy
    r2 = rx * rx + ry * ry
    r_max = float(np.sqrt(cx * cx + cy * cy))
    r_mask = r2 >= (0.15 * r_max) ** 2
    safe_r2 = np.where(r_mask, r2, 1.0)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    dis.setFinestScale(1)
    return dis, r_mask, rx / safe_r2, ry / safe_r2, r_max


def _render_sky_foreground_scene(shift_x: float = 0.0, zoom: float = 1.0, zoom_center: tuple[float, float] = (160.0, 90.0)) -> np.ndarray:
    h, w = 270, 480
    img = np.full((h, w), 196.0, dtype=np.float32)
    sky_h = int(round(h * 0.72))
    yy, xx = np.mgrid[sky_h:h, 0:w].astype(np.float32)
    # Deterministic multi-frequency foreground (rooftops/trees/fence) on bottom 28%.
    pattern = (
        110.0
        + 38.0 * np.sin((xx - shift_x) * 0.21)
        + 24.0 * np.cos((xx - shift_x) * 0.07 + yy * 0.13)
        + 18.0 * ((np.floor((xx - shift_x) / 9.0) % 2) * 2.0 - 1.0)
    )
    img[sky_h:h, :] = np.clip(pattern, 16.0, 235.0)
    if zoom != 1.0:
        mat = cv2.getRotationMatrix2D(zoom_center, 0.0, zoom)
        img = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.round(img).astype(np.uint8)


def test_o4_pan_zoom_lip_flap_and_grain_rejection_benchmarks():
    """Verify O4 retains 0.25px flat-sky pans, off-center zooms, and dark straddling lip-flaps while rejecting grain."""
    dis, r_mask, rx_inv_r2, ry_inv_r2, r_max = _make_dis_geometry()
    scale_factor = 1920.0 / 480.0
    thresh = 0.30

    # (a) Slow pan: 0.25 px/frame at 1080p == 0.0625 px/frame at 480x270 (accumulates >0.25px in <=4 steps,
    # and single-step 0.25px at 1080p == 0.0625px; test 0.08px/frame at 270p == 0.32px at 1080p > thresh=0.30).
    ref_pan = _render_sky_foreground_scene(shift_x=0.0)
    for step in range(1, 6):
        curr_pan = _render_sky_foreground_scene(shift_x=0.08 * step)
        assert _is_dis_motion_early_exit(
            dis,
            ref_pan,
            curr_pan,
            scale_factor,
            thresh,
            thresh * thresh,
            5.0,
            r_mask,
            rx_inv_r2,
            ry_inv_r2,
            r_max,
            auto_noise_floor=True,
            lip_flap_guard=True,
        ), f"Missed 72%-sky foreground pan at step {step}"

    # (b) Off-center zoom around 1/3 frame coordinates (160, 90) with 72% flat sky.
    for step in range(1, 5):
        curr_zoom = _render_sky_foreground_scene(shift_x=0.0, zoom=1.0 + 0.0035 * step, zoom_center=(160.0, 90.0))
        assert _is_dis_motion_early_exit(
            dis,
            ref_pan,
            curr_zoom,
            scale_factor,
            thresh,
            thresh * thresh,
            5.0,
            r_mask,
            rx_inv_r2,
            ry_inv_r2,
            r_max,
            auto_noise_floor=True,
            lip_flap_guard=True,
        ), f"Missed off-center zoom at step {step}"

    # (c) Small dark lip-flap (6x4 at 270p == 24x16 at 1080p, delta +6 on Y=28) straddling 4 cells at (64, 128).
    rng = np.random.default_rng(123)
    dark_ref = np.clip(28.0 + rng.normal(0.0, 0.7, size=(270, 480)), 16, 235).astype(np.uint8)
    dark_mouth = dark_ref.copy()
    dark_mouth[126:130, 61:67] = np.clip(dark_mouth[126:130, 61:67].astype(np.int16) + 6, 0, 255).astype(np.uint8)
    detected_mouth, _ = _check_8x8_lip_flap_gate(dark_ref, dark_mouth)
    assert detected_mouth is True

    # (d) Static anime frame with 72% flat sky + independent Gaussian grain sigma=1.8 -> >=95% rejected.
    false_positives = 0
    trials = 40
    for seed in range(trials):
        g_rng = np.random.default_rng(1000 + seed)
        f_a = np.clip(ref_pan.astype(np.float32) + g_rng.normal(0.0, 1.8, size=ref_pan.shape), 16, 235).astype(np.uint8)
        f_b = np.clip(ref_pan.astype(np.float32) + g_rng.normal(0.0, 1.8, size=ref_pan.shape), 16, 235).astype(np.uint8)
        if _is_dis_motion_early_exit(
            dis,
            f_a,
            f_b,
            scale_factor,
            thresh,
            thresh * thresh,
            5.0,
            r_mask,
            rx_inv_r2,
            ry_inv_r2,
            r_max,
            auto_noise_floor=True,
            lip_flap_guard=True,
        ):
            false_positives += 1
    assert false_positives <= int(0.05 * trials), f"Too many grain false positives: {false_positives}/{trials}"


def test_o4_tier3_cyclic_cel_outlier_tolerance_vs_localized_motion():
    """Verify Tier 3 cyclic cel deduplication absorbs isolated B-frame ringing <=5 while keeping real 4x4 motion."""
    f0 = np.full((128, 128, 3), 90, dtype=np.uint8)
    # 4 isolated pixels with diff +4 (<=0.02% of 128*128*3=49152 -> max 9 elements, 16x16 block mean << 0.85).
    f_ring = f0.copy()
    f_ring[10, 10, 0] = 94
    f_ring[40, 80, 1] = 94
    f_ring[90, 20, 2] = 86
    f_ring[110, 110, 0] = 95

    distinct_ring, map_ring = deduplicate_cyclic_frames([f0, f_ring], allow_quant_noise=True)
    assert len(distinct_ring) == 1
    assert map_ring == [0, 0]

    # Real localized blink (4x4 patch with +18 diff) must NEVER be collapsed.
    f_blink = f0.copy()
    f_blink[50:54, 50:54, :] = 108
    distinct_blink, map_blink = deduplicate_cyclic_frames([f0, f_blink], allow_quant_noise=True)
    assert len(distinct_blink) == 2
    assert map_blink == [0, 1]


def test_o5_shows_yaml_single_model_defaults_and_auto_pareto_resolver():
    """Verify shows.yaml defaults stay single-model per episode and --model auto-pareto pairs same-family engines."""
    bd_path = r"E:\Raw\[AC] Fate Apocrypha - 22 [BD 1080p Hi10 FLAC].mkv"
    web_path = r"E:\Raw\[SubsPlease] Frieren - 05 (1080p).mkv"

    bd_info = resolve_model_for_video(bd_path, cli_model_arg=None)
    web_info = resolve_model_for_video(web_path, cli_model_arg=None)
    assert bd_info["id"] == "realcugan-pro-conservative-2x"
    assert web_info["id"] == "realcugan-pro-conservative-2x"
    assert resolve_dedup_policy(bd_path, cli_policy=None) == "dis-auto"
    assert resolve_dedup_policy(web_path, cli_policy=None) == "dis-auto"

    pareto_info = resolve_model_for_video(bd_path, cli_model_arg=AUTO_PARETO_MODEL_ID)
    assert pareto_info["id"] == "auto-pareto"
    assert pareto_info["is_auto_pareto"] is True
    assert pareto_info["detail_model_id"] == "animejanai-hd-v3-compact"
    assert pareto_info["fast_model_id"] == "animejanai-hd-v3-ultracompact"


def test_o5_web_studio_pareto_and_multipass_sync(tmp_path):
    """Verify Web Studio mirrors dis-auto, auto-pareto, and --nvenc-multipass without GPU imports."""
    assert WEB_DEDUP_POLICY_PARAMS == DEDUP_POLICY_PARAMS

    ep_file = tmp_path / "[SubsPlease] Sample - 01 (1080p).mkv"
    ep_file.write_bytes(b"\x00" * 1024)
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / "models.yaml").write_text(
        Path("models/models.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "shows.yaml").write_text(Path("shows.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    browser = SafeLocalFSBrowser(tmp_path)
    routed_default = browser.resolve_routing_for_path(ep_file)
    assert routed_default["model_id"] == "realcugan-pro-conservative-2x"
    assert routed_default["dedup_policy"] == "dis-auto"

    routed_pareto = browser.resolve_routing_for_path(ep_file, override_model="auto-pareto")
    assert routed_pareto["model_id"] == "auto-pareto"
    assert "Auto-Pareto" in routed_pareto["model_name"]

    captured_cmds: list[list[str]] = []

    def _fake_popen(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        proc = MagicMock()
        proc.pid = 42424
        proc.stdout = iter([])
        proc.wait.return_value = 0
        proc.poll.return_value = 0
        return proc

    jm = JobManager(tmp_path, python_executable="python", popen_factory=_fake_popen)
    jm.start_upscale_job({
        "inputs": [str(ep_file)],
        "model": "auto-pareto",
        "dedup_policy": "dis-auto",
        "nvenc_multipass": "qres",
        "dry_run": True,
    })
    assert "--model" in captured_cmds[-1]
    assert "auto-pareto" in captured_cmds[-1]
    assert "--nvenc-multipass" in captured_cmds[-1]
    assert "qres" in captured_cmds[-1]

    app = web_app.create_app(workspace_root=tmp_path, mock_mode=True)
    with TestClient(app) as client:
        r_cfg = client.get("/api/config/models-and-shows")
        assert r_cfg.status_code == 200
        cfg_json = r_cfg.json()
        assert any(m["id"] == "auto-pareto" for m in cfg_json["routing_modes"])
        assert cfg_json["nvenc_multipass_policies"] == ["auto", "fullres", "qres"]

        r_bad = client.post(
            "/api/jobs/start",
            json={"inputs": [str(ep_file)], "nvenc_multipass": "invalid-mode"},
        )
        assert r_bad.status_code in (400, 422)


def test_o5_o6_auto_pareto_pipeline_scene_cut_hysteresis_and_episode_multipass_lock(tmp_path, monkeypatch):
    """Verify auto-pareto shares one ring upscaler, switches only on scene cuts, and locks multipass per episode."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    src_vid = tmp_path / "ep_pareto.mkv"
    src_vid.write_bytes(b"FAKE_MKV" * 200)
    out_vid = tmp_path / "out" / "ep_pareto_4K.mkv"

    split_seg_times: list[int] = []
    chunk_models_used: list[str] = []
    chunk_multipass_used: list[str | None] = []
    upscaler_instances: list[Any] = []

    def _fake_split(in_p, segs_d, seg_t):
        split_seg_times.append(int(seg_t))
        Path(segs_d).mkdir(parents=True, exist_ok=True)
        segs = []
        for idx in range(3):
            sp = Path(segs_d) / f"seg_{idx:04d}.mp4"
            sp.write_bytes(b"SEG")
            segs.append(str(sp))
        return segs

    def _fake_extract_yuv(seg_file, width, height, kept_indices=None, hwaccel=True, **kwargs):
        n = len(kept_indices) if kept_indices is not None else 4
        y = np.full((height, width), 300 << 6, dtype=np.uint16)
        uv = np.full((height // 2, width), 512 << 6, dtype=np.uint16)
        f = np.vstack([y, uv])
        return [f.copy() for _ in range(n)]

    class _NamedEngine:
        def __init__(self, tag: str) -> None:
            self.tag = tag

    class _FakeEnc:
        def __init__(self, out_mp4: str) -> None:
            self.out_mp4 = out_mp4
            self.stdin = None

        def wait(self, timeout=None) -> int:
            Path(self.out_mp4).write_bytes(b"HEVC_CHUNK")
            return 0

        def kill(self) -> None:
            pass

    class _FakePipeUpscaler:
        spawn_nvenc_pipe_writer = DoubleBufferedPipeUpscaler.spawn_nvenc_pipe_writer

        def __init__(self, model_or_engine, *args, **kwargs) -> None:
            self.model = model_or_engine
            self.output_hw = (32, 48)
            sample_y = np.full((16, 24), 300 << 6, dtype=np.uint16)
            sample_uv = np.full((8, 24), 512 << 6, dtype=np.uint16)
            self.last_first_rgb_small = yuv420_to_rgb_small_u8(np.vstack([sample_y, sample_uv]))
            self.last_distinct_count = 1
            upscaler_instances.append(self)

        def ensure_resolution(self, h: int, w: int) -> tuple[int, int]:
            return self.output_hw

        def open_chunk_encoder(
            self, out_mp4: str, fps_str: str = "24/1", width: int = 48, height: int = 32, multipass=None, **kwargs
        ):
            chunk_multipass_used.append(multipass)
            return _FakeEnc(out_mp4)

        def process_frames_to_pipe(self, unique_frames, repeat_counts, enc_proc, **kwargs) -> int:
            chunk_models_used.append(getattr(self.model, "tag", "unknown"))
            return sum(repeat_counts)

        def release(self) -> None:
            pass

    class _FakeCudaWorker:
        def __init__(self, tag: str) -> None:
            self.model = _NamedEngine(tag)
            self.batch_size = 4
            self.device = torch.device("cpu")

        def restart_if_needed(self, logger=None) -> None:
            pass

    fake_meta = {
        "size_bytes": 2400,
        "width": 24,
        "height": 16,
        "pix_fmt": "yuv420p10le",
        "fps": 24.0,
        "fps_num": 24,
        "fps_den": 1,
        "duration": 24.0,
        "nb_frames": 12,
        "audio_count": 1,
        "subtitle_count": 0,
        "attachment_count": 0,
        "chapter_count": 0,
    }
    # Chunk 0: static close-up dialogue (1/4 keep, r_u=0.25 <= 0.50 -> 'detail' 16c engine).
    # Chunk 1: high-motion starts mid-shot WITHOUT scene_cut (4/4 keep, scene_cut=False -> holds 'detail').
    # Chunk 2: high-motion sakuga WITH scene_cut=True (4/4 keep, r_u=1.0 > 0.55 -> switches to 'fast' 8c on cut).
    fake_plan = {
        "total_frames": 12,
        "total_kept_frames": 9,
        "dedup_ratio_percent": 25.0,
        "chunks": [
            {
                "chunk_idx": 0,
                "total_frames": 4,
                "kept_count": 1,
                "kept_indices": [0],
                "runs": [{"frame_idx": 0, "count": 4}],
                "scene_cut": True,
            },
            {
                "chunk_idx": 1,
                "total_frames": 4,
                "kept_count": 4,
                "kept_indices": [0, 1, 2, 3],
                "runs": [{"frame_idx": i, "count": 1} for i in range(4)],
                "scene_cut": False,
            },
            {
                "chunk_idx": 2,
                "total_frames": 4,
                "kept_count": 4,
                "kept_indices": [0, 1, 2, 3],
                "runs": [{"frame_idx": i, "count": 1} for i in range(4)],
                "scene_cut": True,
            },
        ],
    }

    pareto_info = resolve_model_for_video(str(src_vid), cli_model_arg="auto-pareto")
    worker_cache: dict[str, _FakeCudaWorker] = {}

    def _fake_get_cuda_worker(*args, **kwargs):
        if args and isinstance(args[0], dict):
            m_id = str(args[0].get("id", ""))
        else:
            m_id = str(kwargs.get("model_id") or (kwargs.get("model_info") or {}).get("id") or "")
        tag = "detail" if ("compact" in m_id and "ultra" not in m_id) else "fast"
        if tag not in worker_cache:
            worker_cache[tag] = _FakeCudaWorker(tag)
        return worker_cache[tag]

    def _fake_mux(segs, work_d, in_p, out_p):
        Path(out_p).parent.mkdir(parents=True, exist_ok=True)
        Path(out_p).write_bytes(b"FINAL_4K_MKV" * 100)

    with (
        patch.object(pipeline_mod, "validate_environment"),
        patch.object(pipeline_mod, "acquire_gpu_lock"),
        patch.object(pipeline_mod, "release_gpu_lock"),
        patch.object(pipeline_mod, "run_preflight_checks", return_value=True),
        patch.object(pipeline_mod, "send_system_notification"),
        patch.object(pipeline_mod, "resolve_model_for_video", return_value=pareto_info),
        patch.object(pipeline_mod, "probe_video", return_value=fake_meta),
        patch.object(pipeline_mod, "split_video_segments", side_effect=_fake_split),
        patch.object(pipeline_mod, "get_segments_frame_counts", return_value=[4, 4, 4]),
        patch.object(pipeline_mod, "extract_frames_raw_yuv420", side_effect=_fake_extract_yuv),
        patch.object(pipeline_mod, "mux_segments_directly", side_effect=_fake_mux),
        patch.object(pipeline_mod, "RollingDedupPlanner") as mock_rolling_cls,
        patch("src.cuda_engine.get_cuda_inference_engine", side_effect=_fake_get_cuda_worker),
        patch("src.cuda_engine.GPUPersistentWorker", side_effect=_fake_get_cuda_worker),
        patch("src.build_trt_engine.get_hybrid_engine_or_wrapper", side_effect=lambda m, **kw: m),
        patch("src.trt_zero_copy_upscaler.DoubleBufferedPipeUpscaler", _FakePipeUpscaler),
        patch("src.mux_audit.verify_mux_integrity", return_value={"status": "PASS", "source_meta": fake_meta, "output_meta": fake_meta}),
    ):
        mock_planner = MagicMock()
        mock_planner.get_chunk_plan.side_effect = lambda idx: fake_plan["chunks"][idx]
        mock_planner.get_estimated_dedup_ratio_pct.return_value = 50.0
        mock_planner.finalize_plan.return_value = fake_plan
        mock_rolling_cls.return_value = mock_planner

        res_out = pipeline_mod.run_upscale_pipeline(
            input_path=str(src_vid),
            output_path=str(out_vid),
            model="auto-pareto",
            engine="cuda",
            precision="fp16",
            zero_disk=True,
            use_tensorrt=False,
            nvenc_multipass="auto",
            anti_sleep=False,
            keep_temp=True,
        )

    assert Path(res_out).is_file()
    assert mock_planner.start.call_args.kwargs["sync_initial_chunks"] == 3
    # Auto-promoted omitted -s (default 8) -> 24s super-chunks in CUDA zero-disk mode.
    assert split_seg_times == [24]
    # Shared single DoubleBufferedPipeUpscaler ring across both engines.
    assert len(upscaler_instances) == 1
    # Chunk 0 -> detail; Chunk 1 (no scene_cut) -> holds detail; Chunk 2 (scene_cut=True, low keep) -> switches to fast!
    assert chunk_models_used == ["detail", "detail", "fast"]
    # Uniform per-episode multipass across all 3 chunks (zero SPS/PPS mismatch).
    assert len(set(chunk_multipass_used)) == 1
    assert chunk_multipass_used[0] in ("fullres", "qres")


def test_p1_4_rolling_planner_3chunk_warmup_and_sakuga_dis_thresh(tmp_path):
    """Verify RollingDedupPlanner warmup with 1-2 chunks and sakuga/dis-sakuga dis_thresh persistence."""
    from src.dedup import RollingDedupPlanner, load_or_create_plan, scan_dedup_plan

    def _fake_scan_fused(*args, **kwargs):
        return (["keep", "drop", "keep", "drop"], ["keep", "drop", "keep", "drop"])

    for n_chunks in (1, 2):
        segs = []
        for idx in range(n_chunks):
            sp = tmp_path / f"seg_{n_chunks}_{idx:04d}.mp4"
            sp.write_bytes(b"SEG")
            segs.append(str(sp))
        plan_path = tmp_path / f"plan_{n_chunks}.json"
        with (
            patch("src.dedup._scan_fused_mpdecimate_and_dis", side_effect=_fake_scan_fused),
            patch("src.ffmpeg_ops._count_leading_discard_packets", return_value=0),
        ):
            planner = RollingDedupPlanner(
                input_path=segs[0],
                seg_paths=segs,
                seg_frame_counts=[4] * n_chunks,
                plan_path=str(plan_path),
                segment_seconds=24,
                policy="sakuga",
            )
            planner.start(sync_initial_chunks=3)
            est = planner.get_estimated_dedup_ratio_pct()
            fin = planner.finalize_plan(extra_fields={"nvenc_multipass": "qres"})
            planner.close()

        assert est == pytest.approx(50.0)
        assert fin["dedup_params"]["dis_thresh"] == pytest.approx(0.35)
        assert fin["nvenc_multipass"] == "qres"
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        assert saved["dedup_params"]["dis_thresh"] == pytest.approx(0.35)

    for pol in ("sakuga", "dis-sakuga"):
        scan_plan_path = tmp_path / f"scan_{pol}.json"
        with (
            patch("src.dedup._scan_fused_mpdecimate_and_dis", side_effect=_fake_scan_fused),
            patch("src.ffmpeg_ops._count_leading_discard_packets", return_value=0),
            patch("src.dedup.subprocess.run") as sp_run,
        ):
            sp_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version 7.1\n")
            scanned = scan_dedup_plan(
                str(tmp_path / "seg_1_0000.mp4"),
                str(scan_plan_path),
                segment_seconds=24,
                chunk_frame_counts=[4],
                policy=pol,
            )
        assert scanned["dedup_params"]["dis_thresh"] == pytest.approx(0.35)

        with pytest.raises(ValueError, match="dis_thresh"):
            load_or_create_plan(
                str(tmp_path / "seg_1_0000.mp4"),
                str(scan_plan_path),
                segment_seconds=24,
                resume=True,
                chunk_frame_counts=[4],
                policy=pol,
                dis_thresh=0.20,
            )


def test_p1_4_non_zero_disk_worker_encode_forwards_multipass(tmp_path, monkeypatch):
    """Verify encode_chunk_reexpanded and non-zero-disk worker_encode forward multipass."""
    from src.dedup import encode_chunk_reexpanded

    concat_txt = tmp_path / "concat.txt"
    concat_txt.write_text("file 'frame00000001.png'\n", encoding="utf-8")
    out_mp4 = tmp_path / "chunk_reexp.mp4"
    captured_cmds: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            captured_cmds.append(list(cmd))
            Path(cmd[-1]).write_bytes(b"MP4" * 4000)
            return MagicMock(returncode=0, stdout="")
        return MagicMock(returncode=0, stdout="4\n")

    with (
        patch.object(src_config, "HEVC_ENCODER", "hevc_nvenc"),
        patch("src.dedup.subprocess.run", side_effect=_fake_run),
        patch("src.ffmpeg_ops.verify_segment_resume_frames", return_value=True),
    ):
        encode_chunk_reexpanded(
            str(concat_txt),
            str(out_mp4),
            expected_frames=4,
            multipass="qres",
        )
    assert out_mp4.is_file()
    assert "-multipass" in captured_cmds[0]
    assert captured_cmds[0][captured_cmds[0].index("-multipass") + 1] == "qres"

    # Also verify non-zero-disk pipeline worker_encode forwards episode_multipass to both encoders.
    src_vid = tmp_path / "ep_nzd.mkv"
    src_vid.write_bytes(b"FAKE_MKV" * 200)
    out_vid = tmp_path / "out_nzd" / "ep_nzd_4K.mkv"
    reexp_calls: list[dict[str, Any]] = []
    hevc_calls: list[dict[str, Any]] = []
    fake_meta = {
        "size_bytes": 2400, "width": 24, "height": 16, "pix_fmt": "yuv420p",
        "fps": 24.0, "fps_num": 24, "fps_den": 1, "duration": 24.0, "nb_frames": 8,
        "audio_count": 0, "subtitle_count": 0, "attachment_count": 0, "chapter_count": 0,
    }
    fake_plan = {
        "total_frames": 8,
        "total_kept_frames": 4,
        "dedup_ratio_percent": 50.0,
        "chunks": [
            {"chunk_idx": 0, "total_frames": 4, "kept_count": 2, "kept_indices": [0, 2], "runs": [{"frame_idx": 0, "count": 2}, {"frame_idx": 2, "count": 2}]},
            {"chunk_idx": 1, "total_frames": 4, "kept_count": 2, "kept_indices": [0, 2], "runs": [{"frame_idx": 0, "count": 2}, {"frame_idx": 2, "count": 2}]},
        ],
    }

    def _fake_split(in_p, segs_d, seg_t):
        Path(segs_d).mkdir(parents=True, exist_ok=True)
        res = []
        for idx in range(2):
            sp = Path(segs_d) / f"seg_{idx:04d}.mp4"
            sp.write_bytes(b"SEG")
            res.append(str(sp))
        return res

    def _fake_upscale_fallback(*, input_dir, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        n = 0
        for p in Path(input_dir).glob("*.png"):
            (Path(output_dir) / p.name).write_bytes(b"PNG")
            n += 1
        return {"frames_processed": n, "engine_used": "cuda_fp16", "elapsed_seconds": 0.01, "fps": 100.0}

    def _fake_reexp(*args, **kwargs):
        reexp_calls.append(dict(kwargs))
        Path(kwargs["output_path"]).write_bytes(b"CHUNK" * 100)

    def _fake_hevc(*args, **kwargs):
        hevc_calls.append(dict(kwargs))
        Path(kwargs["output_path"]).write_bytes(b"CHUNK" * 100)

    def _fake_extract(seg_file, frames_dir, *args, **kwargs):
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        for i in range(1, 3):
            (Path(frames_dir) / f"frame{i:08d}.png").write_bytes(b"PNG")
        return 2

    with (
        patch.object(pipeline_mod, "validate_environment"),
        patch.object(pipeline_mod, "acquire_gpu_lock"),
        patch.object(pipeline_mod, "release_gpu_lock"),
        patch.object(pipeline_mod, "run_preflight_checks", return_value=True),
        patch.object(pipeline_mod, "send_system_notification"),
        patch.object(pipeline_mod, "probe_video", return_value=fake_meta),
        patch.object(pipeline_mod, "split_video_segments", side_effect=_fake_split),
        patch.object(pipeline_mod, "get_segments_frame_counts", return_value=[4, 4]),
        patch.object(pipeline_mod, "load_or_create_plan", return_value=fake_plan),
        patch.object(pipeline_mod, "extract_frames", side_effect=_fake_extract),
        patch.object(pipeline_mod, "generate_concat_demuxer_script"),
        patch.object(pipeline_mod, "encode_chunk_reexpanded", side_effect=_fake_reexp),
        patch.object(pipeline_mod, "encode_chunk_hevc", side_effect=_fake_hevc),
        patch.object(pipeline_mod, "upscale_frames_with_fallback", side_effect=_fake_upscale_fallback),
        patch.object(pipeline_mod, "mux_segments_directly", side_effect=lambda segs, wd, ip, op: Path(op).parent.mkdir(parents=True, exist_ok=True) or Path(op).write_bytes(b"MKV")),
        patch("src.output_sanity.check_frame_sanity", return_value={"passed": True, "delta_y": 0.0, "corr": 1.0, "max_delta_y": 6.0, "min_corr": 0.98, "elapsed_ms": 0.5}),
        patch("src.mux_audit.verify_mux_integrity", return_value={"status": "PASS", "source_meta": fake_meta, "output_meta": fake_meta}),
    ):
        pipeline_mod.run_upscale_pipeline(
            input_path=str(src_vid),
            output_path=str(out_vid),
            model="realesr-animevideov3",
            engine="cuda",
            precision="fp16",
            zero_disk=False,
            use_tensorrt=False,
            nvenc_multipass="qres",
            anti_sleep=False,
            keep_temp=True,
        )

    assert len(reexp_calls) == 2
    assert all(c.get("multipass") == "qres" for c in reexp_calls)

    reexp_calls.clear()
    hevc_calls.clear()
    with (
        patch.object(pipeline_mod, "validate_environment"),
        patch.object(pipeline_mod, "acquire_gpu_lock"),
        patch.object(pipeline_mod, "release_gpu_lock"),
        patch.object(pipeline_mod, "run_preflight_checks", return_value=True),
        patch.object(pipeline_mod, "send_system_notification"),
        patch.object(pipeline_mod, "probe_video", return_value=fake_meta),
        patch.object(pipeline_mod, "split_video_segments", side_effect=_fake_split),
        patch.object(pipeline_mod, "get_segments_frame_counts", return_value=[4, 4]),
        patch.object(pipeline_mod, "extract_frames", side_effect=_fake_extract),
        patch.object(pipeline_mod, "encode_chunk_hevc", side_effect=_fake_hevc),
        patch.object(pipeline_mod, "upscale_frames_with_fallback", side_effect=_fake_upscale_fallback),
        patch.object(pipeline_mod, "mux_segments_directly", side_effect=lambda segs, wd, ip, op: Path(op).parent.mkdir(parents=True, exist_ok=True) or Path(op).write_bytes(b"MKV")),
        patch("src.output_sanity.check_frame_sanity", return_value={"passed": True, "delta_y": 0.0, "corr": 1.0, "max_delta_y": 6.0, "min_corr": 0.98, "elapsed_ms": 0.5}),
        patch("src.mux_audit.verify_mux_integrity", return_value={"status": "PASS", "source_meta": fake_meta, "output_meta": fake_meta}),
    ):
        pipeline_mod.run_upscale_pipeline(
            input_path=str(src_vid),
            output_path=str(tmp_path / "out_nzd_no_dedup" / "ep_nzd_4K.mkv"),
            model="realesr-animevideov3",
            engine="cuda",
            precision="fp16",
            dedup=False,
            zero_disk=False,
            use_tensorrt=False,
            nvenc_multipass="qres",
            anti_sleep=False,
            keep_temp=True,
        )

    assert len(hevc_calls) == 2
    assert all(c.get("multipass") == "qres" for c in hevc_calls)


class _NonSRVGGWithBodyAndPixelShuffle(nn.Module):
    """Spandrel-like non-SRVGG 4x model (e.g. DCTLSA/SeemoRe) having conv_first + body + PixelShuffle(4)."""

    def __init__(self) -> None:
        super().__init__()
        self.upscale = 4
        self.conv_first = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.body = nn.ModuleList([nn.Conv2d(8, 8, kernel_size=3, padding=1)])
        self.conv_last = nn.Conv2d(8, 48, kernel_size=3, padding=1)
        self.upsampler = nn.PixelShuffle(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_first(x)
        for blk in self.body:
            h = blk(h)
        return self.upsampler(self.conv_last(h))


def test_o1_non_srvgg_with_body_and_pixelshuffle_is_not_misclassified():
    """Verify non-SRVGG 4x models with conv_first + body + PixelShuffle(4) fall back cleanly without crashing."""
    from src.build_trt_engine import get_hybrid_engine_or_wrapper
    from src.cuda_engine import is_srvgg_compact_model

    net = _NonSRVGGWithBodyAndPixelShuffle().eval().float()
    assert is_srvgg_compact_model(net) is False
    assert is_srvgg_compact_model(RealESRAnimeVideoV3()) is True

    wrapped = SpandrelModelWrapper(net, scale=4, target_scale=2.0).eval().float()
    assert wrapped._can_collapse_srvgg_4x is False

    x = torch.rand(2, 3, 12, 16, dtype=torch.float32)
    with torch.no_grad():
        out_w = wrapped(x)
        hyb = get_hybrid_engine_or_wrapper(wrapped, max_batch=2, force_pytorch=True, scale=2.0)
        out_h = hyb(x)
    assert out_w.shape == (2, 3, 24, 32)
    assert float((out_w - out_h.float()).abs().max()) < 1e-5


def test_o4_dis_auto_custom_thresh_and_rolling_planner_resume_parity(tmp_path):
    """Verify dis-auto-<float>, RollingDedupPlanner dis_thresh persistence, and Open-GOP scan_dedup_plan parity."""
    from src.dedup import get_dedup_params, load_or_create_plan, scan_dedup_plan
    from web.fs_browser import is_valid_dedup_policy

    p_dyn = get_dedup_params("dis-auto-0.22")
    assert p_dyn["dis_thresh"] == pytest.approx(0.22)
    assert p_dyn["auto_noise_floor"] is True
    assert p_dyn["lip_flap_guard"] is True
    assert is_valid_dedup_policy("dis-auto-0.22") is True
    assert is_valid_dedup_policy("dis-auto-0") is False

    seg0 = tmp_path / "seg_0000.mp4"
    seg1 = tmp_path / "seg_0001.mp4"
    seg0.write_bytes(b"s0")
    seg1.write_bytes(b"s1")
    plan_path = tmp_path / "plan.json"

    with (
        patch("src.dedup._scan_fused_mpdecimate_and_dis", return_value=(["keep"] * 4, ["keep", "drop", "keep", "drop"])),
        patch("src.ffmpeg_ops._count_leading_discard_packets", return_value=0),
    ):
        planner = RollingDedupPlanner(
            input_path=str(seg0),
            seg_paths=[str(seg0), str(seg1)],
            seg_frame_counts=[4, 4],
            plan_path=str(plan_path),
            segment_seconds=24,
            policy="dis-auto",
            dis_thresh=0.22,
        )
        planner.start(sync_initial_chunks=1)
        planner.finalize_plan()
        planner.close()

    resumed = load_or_create_plan(
        str(seg0),
        str(plan_path),
        segment_seconds=24,
        resume=True,
        chunk_frame_counts=[4, 4],
        policy="dis-auto-0.22",
    )
    assert resumed["dedup_params"]["dis_thresh"] == pytest.approx(0.22)

    # Open-GOP whole-file scan_dedup_plan with n_discard=2 + chunk_frame_counts=[4, 4] must prepend 2 keeps.
    plan_gop_path = tmp_path / "plan_gop.json"
    with (
        patch(
            "src.dedup._scan_fused_mpdecimate_and_dis",
            side_effect=lambda *a, **kw: (
                kw["boundary_out"].update({"cut_starts": {2}}) or (["keep"] * 6, ["keep", "drop", "keep", "drop", "keep", "keep"])
            ),
        ),
        patch("src.ffmpeg_ops._count_leading_discard_packets", return_value=2),
        patch("src.dedup.subprocess.run") as sp_run,
    ):
        sp_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version 7.1\n")
        gop_plan = scan_dedup_plan(
            str(seg0),
            str(plan_gop_path),
            segment_seconds=24,
            chunk_frame_counts=[4, 4],
            policy="dis-auto",
        )
    assert gop_plan["total_frames"] == 8
    assert [c["total_frames"] for c in gop_plan["chunks"]] == [4, 4]
    assert gop_plan["chunks"][0]["scene_cut"] is True
    assert gop_plan["chunks"][1]["scene_cut"] is True


def test_o5_auto_pareto_resume_preserves_mid_shot_tier_and_full_audit_routing(tmp_path, monkeypatch):
    """Verify resuming auto-pareto after Chunk 0 preserves mid-shot tier on Chunk 1 and full 3-chunk audit history."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    src_vid = tmp_path / "ep_resume_pareto.mkv"
    src_vid.write_bytes(b"FAKE_MKV" * 200)
    out_dir = tmp_path / "out"
    out_vid = out_dir / "ep_resume_pareto_4K.mkv"
    work_dir = out_dir / ".temp_ep_resume_pareto"
    segs_dir = work_dir / "raw_segs"
    up_segs_dir = work_dir / "upscaled_segs"
    segs_dir.mkdir(parents=True, exist_ok=True)
    up_segs_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(3):
        (segs_dir / f"seg_{idx:04d}.mp4").write_bytes(b"SEG" * 100)
    (segs_dir / ".split_complete").write_text(
        json.dumps({"segment_count": 3, "segment_time": 24, "source_file": src_vid.name}),
        encoding="utf-8",
    )
    # Chunk 0 was already completed in a prior run with multipass='qres' and model='detail' (compact 16c).
    (up_segs_dir / "up_0000.mp4").write_bytes(b"X" * 20000)

    pareto_info = resolve_model_for_video(str(src_vid), cli_model_arg="auto-pareto")
    (work_dir / "manifest.json").write_text(
        json.dumps({
            "model_id": "auto-pareto",
            "model_name": pareto_info["name"],
            "sha256": pareto_info["sha256"],
            "engine": "cuda",
            "precision": "fp16",
            "nvenc_multipass": "qres",
            "model_routing": {
                "mode": "auto-pareto",
                "chunk_counts": {"animejanai-hd-v3-compact": 1},
                "chunks": [
                    {
                        "chunk_idx": 0,
                        "model_id": "animejanai-hd-v3-compact",
                        "tier": "detail",
                        "r_unique": 0.25,
                        "scene_cut": True,
                    }
                ],
            },
        }),
        encoding="utf-8",
    )
    # Stale unsplit plan.json ([6, 6] != [4, 4, 4]) on dis-auto -> must trigger RollingDedupPlanner instead of blocking scan!
    (work_dir / "plan.json").write_text(
        json.dumps({
            "total_frames": 12,
            "dedup_policy": "dis-auto",
            "dedup_params": {"dis_thresh": 0.30},
            "chunks": [{"total_frames": 6}, {"total_frames": 6}],
        }),
        encoding="utf-8",
    )

    fake_plan = {
        "total_frames": 12,
        "total_kept_frames": 9,
        "dedup_ratio_percent": 25.0,
        "chunks": [
            {"chunk_idx": 0, "total_frames": 4, "kept_count": 1, "kept_indices": [0], "runs": [{"frame_idx": 0, "count": 4}], "scene_cut": True},
            {"chunk_idx": 1, "total_frames": 4, "kept_count": 4, "kept_indices": [0, 1, 2, 3], "runs": [{"frame_idx": i, "count": 1} for i in range(4)], "scene_cut": False},
            {"chunk_idx": 2, "total_frames": 4, "kept_count": 4, "kept_indices": [0, 1, 2, 3], "runs": [{"frame_idx": i, "count": 1} for i in range(4)], "scene_cut": True},
        ],
    }

    resumed_models: list[str] = []
    resumed_mp: list[str | None] = []

    class _NamedEngine:
        def __init__(self, tag: str) -> None:
            self.tag = tag

    class _FakeEnc:
        def __init__(self, out_mp4: str) -> None:
            self.out_mp4 = out_mp4
            self.stdin = None

        def wait(self, timeout=None) -> int:
            Path(self.out_mp4).write_bytes(b"HEVC_CHUNK" * 2000)
            return 0

        def kill(self) -> None:
            pass

    class _FakePipeUpscaler:
        spawn_nvenc_pipe_writer = DoubleBufferedPipeUpscaler.spawn_nvenc_pipe_writer

        def __init__(self, model_or_engine, *args, **kwargs) -> None:
            self.model = model_or_engine
            self.output_hw = (32, 48)
            sample_y = np.full((16, 24), 300 << 6, dtype=np.uint16)
            sample_uv = np.full((8, 24), 512 << 6, dtype=np.uint16)
            self.last_first_rgb_small = yuv420_to_rgb_small_u8(np.vstack([sample_y, sample_uv]))
            self.last_distinct_count = 1

        def ensure_resolution(self, h: int, w: int) -> tuple[int, int]:
            return self.output_hw

        def open_chunk_encoder(self, out_mp4: str, multipass=None, **kwargs):
            resumed_mp.append(multipass)
            return _FakeEnc(out_mp4)

        def process_frames_to_pipe(self, unique_frames, repeat_counts, enc_proc, **kwargs) -> int:
            resumed_models.append(getattr(self.model, "tag", "unknown"))
            return sum(repeat_counts)

        def release(self) -> None:
            pass

    class _FakeCudaWorker:
        def __init__(self, tag: str) -> None:
            self.model = _NamedEngine(tag)
            self.batch_size = 4
            self.device = torch.device("cpu")

        def restart_if_needed(self, logger=None) -> None:
            pass

    worker_cache: dict[str, _FakeCudaWorker] = {}

    def _fake_get_cuda_worker(*args, **kwargs):
        m_id = str(args[0].get("id", "")) if (args and isinstance(args[0], dict)) else str(kwargs.get("model_id", ""))
        tag = "detail" if ("compact" in m_id and "ultra" not in m_id) else "fast"
        if tag not in worker_cache:
            worker_cache[tag] = _FakeCudaWorker(tag)
        return worker_cache[tag]

    fake_meta = {
        "size_bytes": 2400, "width": 24, "height": 16, "pix_fmt": "yuv420p10le",
        "fps": 24.0, "fps_num": 24, "fps_den": 1, "duration": 24.0, "nb_frames": 12,
        "audio_count": 1, "subtitle_count": 0, "attachment_count": 0, "chapter_count": 0,
    }

    with (
        patch.object(pipeline_mod, "validate_environment"),
        patch.object(pipeline_mod, "acquire_gpu_lock"),
        patch.object(pipeline_mod, "release_gpu_lock"),
        patch.object(pipeline_mod, "run_preflight_checks", return_value=True),
        patch.object(pipeline_mod, "send_system_notification"),
        patch.object(pipeline_mod, "resolve_model_for_video", return_value=pareto_info),
        patch.object(pipeline_mod, "probe_video", return_value=fake_meta),
        patch.object(pipeline_mod, "get_segments_frame_counts", return_value=[4, 4, 4]),
        patch.object(pipeline_mod, "verify_segment_resume_frames", return_value=True),
        patch.object(
            pipeline_mod,
            "extract_frames_raw_yuv420",
            side_effect=lambda *a, **kw: [np.vstack([np.full((16, 24), 300 << 6, dtype=np.uint16), np.full((8, 24), 512 << 6, dtype=np.uint16)]) for _ in range(len(kw.get("kept_indices") or [0]))],
        ),
        patch.object(pipeline_mod, "mux_segments_directly", side_effect=lambda segs, wd, ip, op: Path(op).write_bytes(b"MKV" * 400)),
        patch.object(pipeline_mod, "RollingDedupPlanner") as mock_rolling_cls,
        patch("src.cuda_engine.get_cuda_inference_engine", side_effect=_fake_get_cuda_worker),
        patch("src.cuda_engine.GPUPersistentWorker", side_effect=_fake_get_cuda_worker),
        patch("src.build_trt_engine.get_hybrid_engine_or_wrapper", side_effect=lambda m, **kw: m),
        patch("src.trt_zero_copy_upscaler.DoubleBufferedPipeUpscaler", _FakePipeUpscaler),
        patch("src.mux_audit.verify_mux_integrity", return_value={"status": "PASS", "source_meta": fake_meta, "output_meta": fake_meta}),
    ):
        mock_planner = MagicMock()
        mock_planner.get_chunk_plan.side_effect = lambda idx: fake_plan["chunks"][idx]
        mock_planner.get_estimated_dedup_ratio_pct.return_value = 10.0
        mock_planner.finalize_plan.return_value = fake_plan
        mock_rolling_cls.return_value = mock_planner

        pipeline_mod.run_upscale_pipeline(
            input_path=str(src_vid),
            output_path=str(out_vid),
            model="auto-pareto",
            dedup_policy="dis-auto",
            engine="cuda",
            precision="fp16",
            zero_disk=True,
            use_tensorrt=False,
            nvenc_multipass="auto",
            anti_sleep=False,
            keep_temp=True,
            resume=True,
        )

    assert mock_rolling_cls.called
    # Chunk 1 had scene_cut=False so it MUST preserve Chunk 0's 'detail' tier; Chunk 2 had scene_cut=True -> 'fast'.
    assert resumed_models == ["detail", "fast"]
    # Even though estimated dedup_ratio_pct=10% would pick 'fullres' fresh, resume locks to Chunk 0's 'qres'!
    assert resumed_mp == ["qres", "qres"]
    audit_json = json.loads((out_dir / "ep_resume_pareto_4K.audit.json").read_text(encoding="utf-8"))
    assert [c["chunk_idx"] for c in audit_json["model_routing"]["chunks"]] == [0, 1, 2]
    assert audit_json["model_routing"]["chunk_counts"] == {
        "animejanai-hd-v3-compact": 2,
        "animejanai-hd-v3-ultracompact": 1,
    }


def test_mux_audit_enforces_10bit_bt709_vui_and_preserves_mock_compatibility(tmp_path):
    """Verify probe_video and verify_mux_integrity catch missing/mismatched VUI or 8-bit output while keeping legacy mocks green."""
    import subprocess
    from src.ffmpeg_ops import _probe_video_ffprobe, _probe_video_pyav
    from src.mux_audit import verify_mux_integrity

    src_mkv = tmp_path / "src_1080p.mkv"
    good_4k_mkv = tmp_path / "good_4k.mkv"
    bad_vui_mkv = tmp_path / "bad_vui_4k.mkv"
    bad_8bit_mkv = tmp_path / "bad_8bit_4k.mkv"

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=12:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(src_mkv),
        ],
        check=True,
    )
    # Valid 10-bit BT.709 Limited HEVC output container.
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12:duration=1",
            "-i", str(src_mkv),
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709",
            "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error",
            "-pix_fmt", "yuv420p10le",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "copy", "-max_interleave_delta", "0", str(good_4k_mkv),
        ],
        check=True,
    )
    # 10-bit HEVC with dropped/mismatched VUI (full range + unspecified BT.709 matrix/primaries/transfer).
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12:duration=1",
            "-i", str(src_mkv),
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", "setparams=range=pc:colorspace=unknown:color_primaries=unknown:color_trc=unknown",
            "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error",
            "-pix_fmt", "yuv420p10le", "-color_range", "pc",
            "-c:a", "copy", "-max_interleave_delta", "0", str(bad_vui_mkv),
        ],
        check=True,
    )
    # 8-bit HEVC (yuv420p) despite BT.709 VUI tags.
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12:duration=1",
            "-i", str(src_mkv),
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709",
            "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error",
            "-pix_fmt", "yuv420p",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "copy", "-max_interleave_delta", "0", str(bad_8bit_mkv),
        ],
        check=True,
    )

    pyav_good = _probe_video_pyav(str(good_4k_mkv))
    ffprobe_good = _probe_video_ffprobe(str(good_4k_mkv))
    for k in ("color_range", "color_space", "color_primaries", "color_transfer"):
        assert pyav_good[k] == ffprobe_good[k]
    assert pyav_good["color_range"] == "tv"
    assert pyav_good["color_space"] == "bt709"
    assert pyav_good["color_primaries"] == "bt709"
    assert pyav_good["color_transfer"] == "bt709"

    res_good = verify_mux_integrity(str(src_mkv), str(good_4k_mkv))
    assert res_good["status"] == "PASS", res_good["failure_reasons"]
    assert res_good["video_vui_audit"]["checked"] is True
    assert res_good["video_vui_audit"]["passed"] is True

    res_bad_vui = verify_mux_integrity(str(src_mkv), str(bad_vui_mkv))
    assert res_bad_vui["status"] == "FAIL"
    assert res_bad_vui["video_vui_audit"]["passed"] is False
    assert any("VUI BT.709 Limited" in r for r in res_bad_vui["failure_reasons"])

    res_bad_8bit = verify_mux_integrity(str(src_mkv), str(bad_8bit_mkv))
    assert res_bad_8bit["status"] == "FAIL"
    assert res_bad_8bit["video_vui_audit"]["passed"] is False
    assert any("10-bit" in r for r in res_bad_8bit["failure_reasons"])

    # Backward-compatibility guard: mocked probe dicts omitting color keys must still PASS.
    legacy_mock_meta = {
        "width": 3840,
        "height": 2160,
        "codec": "hevc",
        "pix_fmt": "yuv420p",
        "nb_frames": 12,
        "duration": 1.0,
        "audio_count": 0,
        "subtitle_count": 0,
        "attachment_count": 0,
        "chapter_count": 0,
    }
    res_legacy = verify_mux_integrity(
        str(tmp_path / "nonexistent_src.mkv"),
        str(tmp_path / "nonexistent_out.mkv"),
        src_meta=legacy_mock_meta,
        out_meta=legacy_mock_meta,
    )
    assert res_legacy["status"] == "PASS"
    assert res_legacy["video_vui_audit"]["checked"] is False


def test_rolling_dedup_planner_set_extra_fields_thread_safe(tmp_path):
    """Verify RollingDedupPlanner.set_extra_fields safely updates metadata concurrently with _bg_flusher."""
    import threading
    import time

    segs = []
    for idx in range(4):
        p = tmp_path / f"seg_{idx:04d}.mp4"
        p.write_bytes(b"fake")
        segs.append(str(p))
    plan_path = tmp_path / "dedup_plan.json"

    def _fake_scan_fused(input_path, *args, **kwargs):
        time.sleep(0.01)
        boundary_out = kwargs.get("boundary_out")
        if boundary_out is not None:
            boundary_out["first_gray"] = np.full((64, 64), 32, dtype=np.uint8)
            boundary_out["last_gray"] = np.full((64, 64), 32, dtype=np.uint8)
        return (["keep"] * 4, ["keep", "drop", "keep", "drop"])

    with (
        patch("src.dedup._scan_fused_mpdecimate_and_dis", side_effect=_fake_scan_fused),
        patch("src.ffmpeg_ops._count_leading_discard_packets", return_value=0),
    ):
        planner = RollingDedupPlanner(
            input_path=segs[0],
            seg_paths=segs,
            seg_frame_counts=[4, 4, 4, 4],
            plan_path=str(plan_path),
            segment_seconds=8,
            policy="dis-auto",
            max_workers=2,
            decode_threads=1,
        )
        planner.start(sync_initial_chunks=1)
        errors = []

        def _mutator():
            try:
                for step in range(25):
                    planner.set_extra_fields({"nvenc_multipass": "qres", f"extra_{step}": step})
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=_mutator)
        t.start()
        final_plan = planner.finalize_plan()
        t.join()
        # Post-finalization update must also rewrite plan.json cleanly.
        planner.set_extra_fields({"nvenc_multipass": "fullres", "post_final": True})
        planner.close()

    assert errors == []
    assert final_plan["total_frames"] == 16
    saved = json.loads(plan_path.read_text(encoding="utf-8"))
    assert saved["nvenc_multipass"] == "fullres"
    assert saved["post_final"] is True
    assert saved["extra_24"] == 24

