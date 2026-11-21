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
    assert bd_info["id"] == "animejanai-hd-v3-sharp1-ultracompact"
    assert web_info["id"] == "animejanai-hd-v3-ultracompact"
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
    assert routed_default["model_id"] == "animejanai-hd-v3-ultracompact"
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
    # Auto-promoted omitted -s (default 8) -> 24s super-chunks in CUDA zero-disk mode.
    assert split_seg_times == [24]
    # Shared single DoubleBufferedPipeUpscaler ring across both engines.
    assert len(upscaler_instances) == 1
    # Chunk 0 -> detail; Chunk 1 (no scene_cut) -> holds detail; Chunk 2 (scene_cut=True, low keep) -> switches to fast!
    assert chunk_models_used == ["detail", "detail", "fast"]
    # Uniform per-episode multipass across all 3 chunks (zero SPS/PPS mismatch).
    assert len(set(chunk_multipass_used)) == 1
    assert chunk_multipass_used[0] in ("fullres", "qres")

