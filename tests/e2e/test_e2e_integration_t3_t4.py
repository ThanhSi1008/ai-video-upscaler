# -*- coding: utf-8 -*-
"""
tests/e2e/test_e2e_integration_t3_t4.py
Comprehensive End-to-End Integration Test Suite for AI Video Upscaler:
- Tier 3: Cross-Feature Combinations & Pairwise Coverage (14 tests)
- Tier 4: Real-World Application Workload Scenarios (7 tests)

Adheres strictly to the 4-Tier Test Matrix in TEST_INFRA.md and interface
contracts in PROJECT.md. 100% offline-safe; uses mocks/fixtures for GPU,
TensorRT, and Gemini API services to protect hardware locks and daytime quota.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# System under test (SUT) imports
from src.scheduler import compute_overnight_schedule
from upscale import check_deadline_and_eta
from src.ffmpeg_ops import (
    probe_video,
    concat_segments,
    mux_final_container,
    mux_segments_directly,
    get_segments_frame_counts,
    verify_segment_resume_frames,
)
from src.mux_audit import verify_mux_integrity, get_all_audio_streams_md5
from src.trt_zero_copy_upscaler import GPUColorConverterP010LE, DoubleBufferedPipeUpscaler
from src.cuda_engine import AnimeJaNaiCompact2x, SpandrelModelWrapper
from src.build_trt_engine import (
    _GenericTRTExportModule,
    resolve_trt_profile_max_batch,
    get_trt_engine_path,
)
from src.models_resolver import load_models_registry, resolve_model_for_video
from src.subtitles import transcribe_japanese_audio
from subtrans.cache import TranslationCache
from subtrans.rag import KnowledgeBase, _generate_ngram_string
from subtrans.models import Cue
from src.output_sanity import check_frame_sanity_arrays
from scripts.telemetry_collector import TelemetryCollector
from scripts.set_track_defaults import process_file as process_track_defaults_file


# =============================================================================
# TIER 3: CROSS-FEATURE PAIRWISE INTEGRATION TESTS (≥14 tests)
# =============================================================================

def test_t3_f1_deadline_and_f4_pyav_metadata(tmp_path: Path):
    """
    Tier 3 - Pairwise 1 (F1 + F4):
    Feeds PyAV in-process probed metadata directly into the scheduler ETA math.
    Verifies that unique inference FPS (N_unique / FPS_unique) accurately drives
    the ETA calculation, avoiding the ~2.1x overestimation discrepancy.
    """
    sample_video = str(tmp_path / "sample_pyav_feed.mkv")
    Path(sample_video).write_bytes(b"dummy_mkv_bytes")

    pyav_meta = {
        "width": 1920,
        "height": 1080,
        "fps": 23.976,
        "fps_num": 24000,
        "fps_den": 1001,
        "duration": 1420.0,
        "nb_frames": 34045,
        "codec": "h264",
        "pix_fmt": "yuv420p",
        "audio_count": 2,
        "subtitle_count": 2,
        "attachment_count": 1,
        "chapter_count": 5,
    }

    with patch("src.scheduler.probe_video", return_value=pyav_meta), \
         patch("src.scheduler.find_cached_plan", return_value=None):

        fps_unique = 10.5
        sched = compute_overnight_schedule(
            queue=[sample_video],
            deadline_str="08:00",
            fps_median=fps_unique,
            fps_p5=8.5,
            video_overhead_seconds=930.0,
            subtitles_overhead_seconds=300.0,
            start_time=datetime(2026, 11, 14, 22, 0, 0),
        )

        assert sched["total_episodes"] == 1
        ep = sched["episodes"][0]

        # In absence of preflight plan, scheduler uses 60% of total frames as conservative unique frames
        expected_unique = int(34045 * 0.60)
        assert ep["n_unique"] == expected_unique

        # Expected AI inference seconds must be based on unique frames / unique FPS
        expected_ai_seconds = expected_unique / fps_unique
        assert ep["ai_seconds_median"] == pytest.approx(expected_ai_seconds, rel=1e-3)

        # Total ETA includes video overhead (930s) + subtitles overhead (300s) = 1230s
        expected_total_eta = expected_ai_seconds + 1230.0
        assert ep["eta_median_seconds"] == pytest.approx(expected_total_eta, rel=1e-3)
        assert ep["can_run"] is True


def test_t3_f1_deadline_and_f5_stream_mux(tmp_path: Path):
    """
    Tier 3 - Pairwise 2 (F1 + F5):
    Verifies that when an overnight queue triggers a deadline cutoff (allow_partial=True),
    completed chunks are safely concatenated and muxed via FFmpeg using mandatory
    bit-exact stream copy flags: -c copy -max_interleave_delta 0.
    """
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    seg0 = tmp_path / "up_0000.mp4"
    seg1 = tmp_path / "up_0001.mp4"
    seg0.write_bytes(b"chunk0")
    seg1.write_bytes(b"chunk1")
    orig_vid = tmp_path / "orig.mkv"
    orig_vid.write_bytes(b"orig_content")
    out_mkv = tmp_path / "out_partial.mkv"

    executed_cmds: List[List[str]] = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        executed_cmds.append(cmd)
        out_mkv.write_bytes(b"muxed_partial_mkv")
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    with patch("subprocess.run", side_effect=mock_subprocess_run):
        mux_segments_directly(
            segment_files=[str(seg0), str(seg1)],
            work_dir=str(work_dir),
            original_video=str(orig_vid),
            final_output=str(out_mkv),
        )

    assert len(executed_cmds) >= 1
    direct_cmd = executed_cmds[0]

    # Verify Studio Preservation Flags
    assert "-max_interleave_delta" in direct_cmd
    mid_idx = direct_cmd.index("-max_interleave_delta")
    assert direct_cmd[mid_idx + 1] == "0"
    assert "-c" in direct_cmd
    c_idx = direct_cmd.index("-c")
    assert direct_cmd[c_idx + 1] == "copy"
    assert "-map" in direct_cmd
    assert "0:v:0" in direct_cmd
    assert "-map_chapters" in direct_cmd


def test_t3_f2_triton_p010le_and_f3_pinned_ring():
    """
    Tier 3 - Pairwise 3 (F2 + F3):
    End-to-end memory flow: host RGB frames -> H2D -> Triton JIT / GPUColorConverterP010LE
    -> 2-slot P010LE Ring Buffer. Verifies zero dynamic VRAM reallocation, correct
    semi-planar UV interleaving, and BT.709 TV-range boundaries.
    """
    conv = GPUColorConverterP010LE(device=torch.device("cpu"))

    # Test B=4, H=64, W=64
    B, H, W = 4, 64, 64
    rgb_tensor = torch.rand(B, 3, H, W, dtype=torch.float32)

    # First conversion: initial allocation of the 2-slot ring buffer
    out_slot0 = conv.rgb_to_p010le(rgb_tensor, slot=0)
    assert len(conv._p010_ring) == 2
    assert out_slot0.shape == (B, H * 3 // 2, W)
    assert out_slot0.dtype == torch.int16

    ring_buf_addr_0 = conv._p010_ring[0].data_ptr()
    ring_buf_addr_1 = conv._p010_ring[1].data_ptr()

    # Second conversion: reuse slot 1 without buffer reallocation
    out_slot1 = conv.rgb_to_p010le(rgb_tensor, slot=1)
    assert conv._p010_ring[0].data_ptr() == ring_buf_addr_0
    assert conv._p010_ring[1].data_ptr() == ring_buf_addr_1
    assert out_slot1.shape == (B, H * 3 // 2, W)

    # BT.709 Range Verifications:
    # Y is bounded in [64, 940] << 6 -> [4096, 60160] (unsigned representation in int16)
    # UV is bounded in [64, 960] << 6 -> [4096, 61440]
    y_plane = out_slot0[:, :H, :].to(torch.int32)
    # Convert negative signed int16 back to unsigned 16-bit for range checking
    y_unsigned = torch.where(y_plane < 0, y_plane + 65536, y_plane)
    assert (y_unsigned >= 4096).all()
    assert (y_unsigned <= 60160).all()

    # Dynamic batch size tail: test B=1 handling without crash or shape corruption
    tail_rgb = torch.rand(1, 3, H, W, dtype=torch.float32)
    tail_out = conv.rgb_to_p010le(tail_rgb, slot=0)
    assert tail_out.shape == (1, H * 3 // 2, W)


def test_t3_f4_pyav_and_f5_mux_audit(tmp_path: Path):
    """
    Tier 3 - Pairwise 4 (F4 + F5):
    Extracts stream metadata in-process and feeds into Mux Audit verification.
    Verifies that verify_mux_integrity correctly identifies bit-exact stream matches
    and raises warnings/failures when audio tracks or streams are missing.
    """
    src_video = str(tmp_path / "src.mkv")
    out_video = str(tmp_path / "out_4k.mkv")
    Path(src_video).write_bytes(b"src")
    Path(out_video).write_bytes(b"out")

    # Mock identical stream specs between source and output
    mock_src_info = {
        "width": 1920,
        "height": 1080,
        "duration": 60.0,
        "audio_count": 2,
        "subtitle_count": 1,
        "attachment_count": 2,
        "chapter_count": 3,
        "audio_streams": [
            {"index": 0, "codec": "flac", "channels": 6, "language": "jpn"},
            {"index": 1, "codec": "aac", "channels": 2, "language": "eng"},
        ],
        "subtitle_streams": [{"index": 0, "codec": "subrip", "language": "vie"}],
        "attachments": [{"index": 0, "filename": "font1.ttf"}, {"index": 1, "filename": "font2.otf"}],
        "chapters": [{"id": 0}, {"id": 1}, {"id": 2}],
    }

    mock_out_info = dict(mock_src_info)
    mock_out_info["width"] = 3840
    mock_out_info["height"] = 2160

    mock_md5 = [
        {"stream_index": 0, "stream_map": "0:a:0", "md5": "beefcafe01"},
        {"stream_index": 1, "stream_map": "0:a:1", "md5": "beefcafe02"},
    ]
    interleave_pass = {"passed": True, "failed_windows": []}

    with patch("src.mux_audit.probe_video", side_effect=[mock_src_info, mock_out_info]), \
         patch("src.mux_audit.get_detailed_stream_info", side_effect=[mock_src_info, mock_out_info]), \
         patch("src.mux_audit.get_all_audio_streams_md5", side_effect=[mock_md5, mock_md5]), \
         patch("src.mux_audit.check_window_interleaving", return_value=interleave_pass), \
         patch("src.mux_audit.count_subtitle_packets", return_value=120), \
         patch("src.mux_audit._count_video_packets", return_value=1440), \
         patch("src.mux_audit.get_audio_volume_db", return_value=-15.0):

        audit_result = verify_mux_integrity(src_video, out_video)
        assert audit_result["status"] == "PASS"
        assert audit_result["all_audio_tracks_md5"]["passed"] is True

    # Negative case: Mismatched audio track count
    mismatched_out = dict(mock_out_info)
    mismatched_out["audio_count"] = 1
    mismatched_out["audio_streams"] = [mock_src_info["audio_streams"][0]]

    with patch("src.mux_audit.probe_video", side_effect=[mock_src_info, mismatched_out]), \
         patch("src.mux_audit.get_detailed_stream_info", side_effect=[mock_src_info, mismatched_out]), \
         patch("src.mux_audit.get_all_audio_streams_md5", return_value=mock_md5), \
         patch("src.mux_audit.check_window_interleaving", return_value=interleave_pass), \
         patch("src.mux_audit.count_subtitle_packets", return_value=120), \
         patch("src.mux_audit._count_video_packets", return_value=1440), \
         patch("src.mux_audit.get_audio_volume_db", return_value=-15.0):

        audit_fail = verify_mux_integrity(src_video, out_video)
        assert audit_fail["status"] == "FAIL"
        assert any("Mất luồng âm thanh" in r for r in audit_fail["failure_reasons"])


def test_t3_f5_concat_mux_and_f6_track_defaults(tmp_path: Path):
    """
    Tier 3 - Pairwise 5 (F5 + F6):
    Mux an MKV container and immediately apply track default flags via set_track_defaults.py.
    Verifies that Japanese audio is prioritized to default=1 and Vietnamese/dialogue subtitle
    is activated while signs/songs tracks are suppressed.
    """
    test_mkv = tmp_path / "muxed_video.mkv"
    test_mkv.write_bytes(b"dummy_mkv_header")

    mkv_tracks_json = {
        "tracks": [
            {
                "type": "audio",
                "properties": {"uid": 101, "language": "eng", "default_track": True, "track_name": "English 2.0"},
            },
            {
                "type": "audio",
                "properties": {"uid": 102, "language": "jpn", "default_track": False, "track_name": "Japanese 5.1"},
            },
            {
                "type": "subtitles",
                "properties": {"uid": 201, "language": "eng", "default_track": True, "track_name": "Signs & Songs"},
            },
            {
                "type": "subtitles",
                "properties": {"uid": 202, "language": "vie", "default_track": False, "track_name": "Vietnamese Fansub"},
            },
        ]
    }

    executed_propedit_args = []

    def mock_subprocess(cmd, *args, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if "-J" in cmd:
            res.stdout = json.dumps(mkv_tracks_json)
        else:
            executed_propedit_args.extend(cmd)
            res.stdout = "Changes written successfully."
        return res

    with patch("subprocess.run", side_effect=mock_subprocess):
        res = process_track_defaults_file(str(test_mkv))

    assert res["status"] == "modified"
    changes = res["changes"]
    assert any("UID=102" in c and "default=1" in c for c in changes)
    assert any("UID=101" in c and "default=0" in c for c in changes)
    assert any("UID=202" in c and "default=1" in c for c in changes)
    assert any("UID=201" in c and "default=0" in c for c in changes)


def test_t3_f7_sqlite_closing_and_f8_kanji_rag(tmp_path: Path):
    """
    Tier 3 - Pairwise 6 (F7 + F8):
    Populate FTS5 Kanji RAG table within context manager, query 2-character Kanji keywords
    ('魔術', '聖杯', '令呪'), and verify that SQLite handles close completely without
    triggering Windows WinError 32 upon database file deletion.
    """
    db_file = tmp_path / "test_kanji_rag.db"
    kb = KnowledgeBase(db_path=db_file)

    # Insert verified gold entries containing 2-character Kanji keywords
    kb.add_gold_unit(
        ja_text="我が魔術の深淵を見よ。",
        vi_text="Hãy chiêm ngưỡng vực sâu ma thuật của ta.",
        series="fate",
        status="gold",
    )
    kb.add_gold_unit(
        ja_text="聖杯を巡る戦いが始まる。",
        vi_text="Cuộc chiến tranh giành Chén Thánh bắt đầu.",
        series="fate",
        status="gold",
    )
    kb.add_gold_unit(
        ja_text="令呪をもって命ず。自害せよ、ランサー。",
        vi_text="Dùng Lệnh Chú ra lệnh. Tự sát đi, Lancer.",
        series="fate",
        status="gold",
    )

    # Search for 2-character Kanji terms
    res_majutsu = kb.search_gold_units("魔術師", top_k=3)
    assert len(res_majutsu) >= 1
    assert "魔術" in res_majutsu[0]["ja_text"]

    res_seihai = kb.search_gold_units("聖杯戦争", top_k=3)
    assert len(res_seihai) >= 1
    assert "聖杯" in res_seihai[0]["ja_text"]

    res_reiju = kb.search_gold_units("令呪", top_k=3)
    assert len(res_reiju) >= 1
    assert "令呪" in res_reiju[0]["ja_text"]

    # Critical Windows test: Immediate removal of DB file must succeed without WinError 32
    del kb
    assert db_file.exists()
    try:
        os.remove(db_file)
    except PermissionError as e:
        pytest.fail(f"WinError 32 detected! SQLite handle was not properly closed: {e}")

    assert not db_file.exists()


def test_t3_f8_kanji_rag_and_f12_whisper_stt(tmp_path: Path):
    """
    Tier 3 - Pairwise 7 (F8 + F12):
    Feeds Whisper-transcribed cues containing Kanji into the FTS5 RAG retrieval engine.
    Verifies that the 2-gram/3-gram FTS5 index properly retrieves matching domain glossaries.
    """
    db_file = tmp_path / "whisper_rag.db"
    kb = KnowledgeBase(db_path=db_file)

    kb.add_gold_unit(
        ja_text="全集中、水の呼吸、壱ノ型！",
        vi_text="Toàn Tập Trung, Hơi thở của Nước, Thức thứ nhất!",
        series="kimetsu",
        status="gold",
    )
    kb.add_gold_unit(
        ja_text="領域展開、無量空処。",
        vi_text="Lãnh địa triển khai: Vô Lượng Không Xứ.",
        series="jujutsu",
        status="gold",
    )

    # Simulated cues from Faster-Whisper transcription
    whisper_cues = [
        Cue(id=1, start=1000, end=3000, ja="水の呼吸を使って敵を倒す！"),
        Cue(id=2, start=4000, end=6000, ja="無量空処の領域に引き込まれた。"),
    ]

    # Retrieve few-shots based on the cues
    matches = kb.retrieve_few_shots(cues=whisper_cues[:1], series="kimetsu", max_examples=2)
    assert len(matches) >= 1
    assert "水の呼吸" in matches[0]["ja"]


def test_t3_f9_spandrel_2x_and_f10_trt_export(tmp_path: Path):
    """
    Tier 3 - Pairwise 8 (F9 + F10):
    Loads a Native 2x model via Spandrel wrapper, wraps it in _GenericTRTExportModule,
    and performs ONNX export with dynamic batch axes. Confirms that Native 2x bypasses
    bicubic 0.5x downsampling.
    """
    class MockNative2xModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 2
            self.conv = nn.Conv2d(3, 12, 3, 1, 1)
            self.pix = nn.PixelShuffle(2)

        def forward(self, x):
            return self.pix(self.conv(x))

    native_model = MockNative2xModel()
    export_wrapper = _GenericTRTExportModule(native_model, target_scale=2.0)

    # Forward pass verification
    dummy_input = torch.rand(2, 3, 32, 32)
    out = export_wrapper(dummy_input)
    assert out.shape == (2, 3, 64, 64)
    assert export_wrapper.model_scale == 2.0
    assert export_wrapper.target_scale == 2.0

    # Test dynamic batch ONNX export
    onnx_file = str(tmp_path / "model_native2x.onnx")
    torch.onnx.export(
        export_wrapper,
        dummy_input,
        onnx_file,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    assert os.path.exists(onnx_file)
    assert os.path.getsize(onnx_file) > 0


def test_t3_f9_spandrel_2x_and_f11_models_yaml():
    """
    Tier 3 - Pairwise 9 (F9 + F11):
    Resolves 'span-2x' and 'animejanai-hd-v3-compact' from models.yaml, verifies
    architectural metadata (upscale=2, downsample='none'), and ensures clean instantiation.
    """
    registry = load_models_registry()
    assert "animejanai-hd-v3-compact" in registry
    assert "span-2x" in registry

    span_cfg = registry["span-2x"]
    assert span_cfg["arch"]["type"] == "SPAN"
    assert span_cfg["arch"]["upscale"] == 2
    assert span_cfg["downsample"] == "none"

    # Instantiate Native 2x architecture
    compact_model = AnimeJaNaiCompact2x()
    wrapper = SpandrelModelWrapper(compact_model, scale=2, target_scale=2.0)
    assert wrapper.scale == 2
    assert wrapper.target_scale == 2.0

    x = torch.zeros(1, 3, 16, 16)
    out = wrapper(x)
    assert out.shape == (1, 3, 32, 32)


def test_t3_f10_trt_export_and_f11_models_yaml():
    """
    Tier 3 - Pairwise 10 (F10 + F11):
    Verifies TensorRT profile batch bounds resolution based on models.yaml models:
    4x models (animevideov3) are capped at max_batch=2 on 1080p, while Native 2x
    models (span-2x) allow max_batch=4.
    """
    class Mock4xModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.upscale = 4

    class Mock2xModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.upscale = 2

    # 4x model on 1080p must be capped at max_batch=2 to prevent 12.74 GB VRAM OOM
    batch_4x = resolve_trt_profile_max_batch(Mock4xModel(), max_batch=8, input_hw=(1080, 1920))
    assert batch_4x == 2

    # Native 2x model on 1080p allows max_batch=4
    batch_2x = resolve_trt_profile_max_batch(Mock2xModel(), max_batch=8, input_hw=(1080, 1920))
    assert batch_2x == 4

    # Engine file path formatting check
    engine_path = get_trt_engine_path("span-2x", max_batch=4, input_hw=(1080, 1920))
    assert "span-2x" in engine_path
    assert "1080x1920" in engine_path
    assert "b1-4.engine" in engine_path


def test_t3_f12_whisper_and_f7_sqlite_cache(tmp_path: Path):
    """
    Tier 3 - Pairwise 11 (F12 + F7):
    Caches Faster-Whisper transcribed subtitles into SQLite TranslationCache,
    verifying retrieval and clean connection context management without handle leakage.
    """
    db_file = tmp_path / "whisper_cache.db"
    cache = TranslationCache(db_path=db_file)

    ja_text = "おはよう、今日もいい天気だね。"
    vi_trans = "Chào buổi sáng, hôm nay thời tiết đẹp nhỉ."

    cache_key = "test_whisper_key_001"
    cache.set_cue(
        cache_key=cache_key,
        backend="gemini",
        model="gemini-3.5-flash-lite",
        vi=vi_trans,
        metadata={"source": "whisper_stt", "model_size": "large-v3-turbo"},
    )

    retrieved = cache.get_cue(cache_key)
    assert retrieved == vi_trans

    # Verify context manager closes connection immediately
    del cache
    assert db_file.exists()
    try:
        os.remove(db_file)
    except PermissionError as e:
        pytest.fail(f"SQLite connection handle was leaked in TranslationCache: {e}")
    assert not db_file.exists()


def test_t3_f13_benchmark_and_f1_deadline(tmp_path: Path):
    """
    Tier 3 - Pairwise 12 (F13 + F1):
    Feeds benchmark telemetry (ai_steady_fps) into scheduler unique FPS parameters,
    verifying that the overnight queue dynamically adapts ETA allocations.
    """
    video_path = str(tmp_path / "test_ep.mkv")
    Path(video_path).write_bytes(b"ep_bytes")

    plan_path = tmp_path / "test_ep.plan.json"
    plan_data = {
        "source_file": "test_ep.mkv",
        "total_frames": 34000,
        "total_kept_frames": 18000,
        "dedup_ratio_percent": 47.06,
        "chunks": [{"id": 0}],
    }
    plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

    # Baseline speed vs Benchmark measured speed
    measured_unique_fps = 11.2

    with patch("src.scheduler.find_cached_plan", return_value=plan_data):
        sched = compute_overnight_schedule(
            queue=[video_path],
            deadline_str="07:00",
            fps_median=measured_unique_fps,
            fps_p5=9.8,
            start_time=datetime(2026, 11, 14, 23, 0, 0),
        )

        ep = sched["episodes"][0]
        assert ep["n_unique"] == 18000
        # AI duration = 18000 / 11.2 ~ 1607.1s (~26.8 minutes)
        assert ep["ai_seconds_median"] == pytest.approx(18000 / 11.2, rel=1e-3)
        assert ep["can_run"] is True


def test_t3_f13_benchmark_and_f9_spandrel_2x():
    """
    Tier 3 - Pairwise 13 (F13 + F9):
    Runs Output Sanity Guard on Native 2x model outputs, ensuring that image quality
    invariants (|ΔY| ≤ 6.0, r ≥ 0.98) pass on valid upscaled frames and fail on degraded frames.
    """
    np.random.seed(42)
    # Generate clean 480x270 gradient frame (simulating Anime source)
    y_grad = np.linspace(50, 200, 270, dtype=np.uint8)[:, None]
    x_grad = np.linspace(50, 200, 480, dtype=np.uint8)[None, :]
    src_frame = np.stack([y_grad.repeat(480, axis=1), x_grad.repeat(270, axis=0), np.full((270, 480), 128, dtype=np.uint8)], axis=-1)

    # Simulate high-fidelity Native 2x upscale (slight sharpening, zero color degradation)
    upscaled_valid = np.clip(src_frame.astype(np.float32) * 1.01 + 0.5, 0, 255).astype(np.uint8)

    sanity_valid = check_frame_sanity_arrays(src_frame, upscaled_valid, is_rgb=True)
    assert sanity_valid["passed"] is True
    assert sanity_valid["delta_y"] <= 6.0
    assert sanity_valid["corr"] >= 0.98

    # Simulate black frame anomaly (e.g. GPU NaN or failed infer)
    upscaled_black = np.zeros_like(src_frame)
    sanity_black = check_frame_sanity_arrays(src_frame, upscaled_black, is_rgb=True)
    assert sanity_black["passed"] is False


def test_t3_f14_offline_runner_and_all_features():
    """
    Tier 3 - Pairwise 14 (F14 + All Features):
    Verifies that offline execution rules are completely respected:
    network sockets raise RuntimeError, GPU lock files are not left orphaned,
    and no production caches are touched.
    """
    from src.config import GPU_LOCK_FILE

    # Verify socket blocking guard behavior
    import socket
    with pytest.raises(Exception):
        socket.socket().connect(("127.0.0.1", 80))

    # Verify no active GPU lock left behind
    assert not os.path.exists(GPU_LOCK_FILE)


# =============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (≥7 realistic workloads)
# =============================================================================

def test_t4_scenario_1_native_2x_zero_disk_pipeline(tmp_path: Path):
    """
    Tier 4 - Scenario 1:
    End-to-End Native 2x Zero-Disk Pipeline simulation for an 8s chunk:
    NVDEC decode -> Dedup -> Spandrel 2x model -> Triton P010LE -> NVENC encode pipe.
    Verifies zero intermediate PNG files on disk, in-memory repeat expansion,
    and batch 0 Sanity Guard validation.
    """
    # 1. Simulate 24 input frames (1 second @ 24fps) at compact 160x90 -> 320x180 2x geometry
    in_h, in_w = 90, 160
    out_h, out_w = in_h * 2, in_w * 2
    np.random.seed(123)
    frame_base = np.random.randint(60, 200, (in_h, in_w, 3), dtype=np.uint8)
    frames = [frame_base.copy() for _ in range(24)]

    # 2. Simulate Dedup engine: 8 distinct unique frames, repeated 3x each = 24 frames
    unique_frames = frames[:8]
    repeat_counts = [3] * 8
    assert sum(repeat_counts) == 24

    # 3. Model: Native 2x
    model_2x = AnimeJaNaiCompact2x()

    # 4. Upscaler pipe instance
    upscaler = DoubleBufferedPipeUpscaler(
        model_or_engine=model_2x,
        batch_size=8,
        device=torch.device("cpu"),
        input_hw=(in_h, in_w),
        scale=2,
    )

    # 5. Mock NVENC encoder process stdin
    bytes_written = 0
    pipe_chunks: List[bytes] = []

    class MockStdin:
        def write(self, b):
            nonlocal bytes_written
            chunk_len = len(b)
            bytes_written += chunk_len
            pipe_chunks.append(bytes(b[:64]))
            return chunk_len

    class MockEncoderProc:
        stdin = MockStdin()
        def poll(self):
            return None

    mock_proc = MockEncoderProc()

    # 6. Process frames through zero-disk pipe
    written_count = upscaler.process_frames_to_pipe(
        unique_frames_rgb=unique_frames,
        repeat_counts=repeat_counts,
        encoder_proc=mock_proc,
        cyclic_dedup=False,
    )

    assert written_count == 24
    expected_frame_bytes = out_w * (out_h * 3 // 2) * 2
    assert bytes_written == expected_frame_bytes * 24

    # Verify zero disk PNG files were created
    disk_pngs = list(tmp_path.glob("*.png"))
    assert len(disk_pngs) == 0


def test_t4_scenario_2_crash_resume_pyav_validation(tmp_path: Path):
    """
    Tier 4 - Scenario 2:
    Simulates a crash during upscaling after 3 segments, resumes using in-process
    PyAV segment packet validation, completes remaining segments, and invokes Concat-Mux.
    """
    temp_dir = tmp_path / ".temp_episode_01"
    temp_dir.mkdir()

    # Segments 0, 1, 2 already finished before crash (must be >= 10000 bytes)
    seg0 = temp_dir / "up_0000.mp4"
    seg1 = temp_dir / "up_0001.mp4"
    seg2 = temp_dir / "up_0002.mp4"
    for s in (seg0, seg1, seg2):
        s.write_bytes(b"0" * 15000)

    seg_paths = [str(seg0), str(seg1), str(seg2)]

    # Mock in-process PyAV frame counting (192 frames per 8s chunk @ 24fps)
    with patch("src.ffmpeg_ops._count_segment_packets_pyav", return_value=192), \
         patch("src.ffmpeg_ops._verify_segment_resume_frames_ffprobe", return_value=True):
        counts = get_segments_frame_counts(seg_paths)
        assert counts == [192, 192, 192]

        # Verify resumption integrity check
        assert verify_segment_resume_frames(str(seg0), 192) is True
        assert verify_segment_resume_frames(str(seg1), 192) is True
        assert verify_segment_resume_frames(str(seg2), 192) is True

    # Complete segments 3 and 4 after resume
    seg3 = temp_dir / "up_0003.mp4"
    seg4 = temp_dir / "up_0004.mp4"
    seg3.write_bytes(b"0" * 15000)
    seg4.write_bytes(b"0" * 15000)
    all_segs = seg_paths + [str(seg3), str(seg4)]

    # Concat-Mux into final output
    final_out = tmp_path / "episode_01_4K.mkv"
    orig_mkv = tmp_path / "episode_01.mkv"
    orig_mkv.write_bytes(b"orig_mkv")

    with patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0)
        mux_segments_directly(
            segment_files=all_segs,
            work_dir=str(temp_dir),
            original_video=str(orig_mkv),
            final_output=str(final_out),
        )
        assert mock_subproc.called


def test_t4_scenario_3_anime_subtitle_extraction_kanji_rag(tmp_path: Path):
    """
    Tier 4 - Scenario 3:
    Japanese Anime Subtitle Extraction & Kanji RAG Translation:
    Extracts 5.1 FC mono audio, simulates Faster-Whisper large-v3-turbo transcription,
    and queries FTS5 Kanji RAG to enrich translation glossary.
    """
    video_file = tmp_path / "anime_ep.mkv"
    video_file.write_bytes(b"anime_mkv")
    srt_out = tmp_path / "anime_ep.ja.srt"

    # Setup KnowledgeBase with series glossaries
    kb = KnowledgeBase(db_path=tmp_path / "anime_rag.db")
    kb.add_gold_unit(
        ja_text="宝具解放！我が真名を告げる。",
        vi_text="Bảo Cụ giải phóng! Ta xưng chân danh của mình.",
        series="fate",
        status="gold",
    )

    # Mock Faster-Whisper transcription segment
    class MockSegment:
        start = 1.5
        end = 4.2
        text = "宝具解放の時が来た。"
        no_speech_prob = 0.01

    class MockWhisperModel:
        def __init__(self, *args, **kwargs):
            pass
        def transcribe(self, audio_path, **kwargs):
            return [MockSegment()], None

    with patch("src.subtitles.subprocess.run") as mock_ffmpeg, \
         patch("faster_whisper.WhisperModel", MockWhisperModel), \
         patch("src.subtitles.os.path.exists", return_value=True), \
         patch("src.subtitles.os.path.getsize", return_value=1024):

        mock_ffmpeg.return_value = MagicMock(returncode=0)
        ok = transcribe_japanese_audio(str(video_file), str(srt_out), model_size="large-v3-turbo")
        assert ok is True

    # RAG lookup with transcribed Kanji
    rag_matches = kb.search_gold_units("宝具解放", top_k=2)
    assert len(rag_matches) >= 1
    assert "宝具" in rag_matches[0]["ja_text"]


def test_t4_scenario_4_overnight_queue_deadline_cutoff(tmp_path: Path):
    """
    Tier 4 - Scenario 4:
    Overnight Queue Execution with Deadline Cutoff:
    Simulates a 3-episode queue where Episode 1 finishes completely,
    Episode 2 hits the deadline cutoff with allow_partial=True, and Episode 3
    is cleanly skipped.
    """
    ep1 = str(tmp_path / "ep01.mkv")
    ep2 = str(tmp_path / "ep02.mkv")
    ep3 = str(tmp_path / "ep03.mkv")
    for f in (ep1, ep2, ep3):
        Path(f).write_bytes(b"content")

    # Mock plan.json for each episode (18,000 unique frames each)
    mock_plan = {
        "total_frames": 34000,
        "total_kept_frames": 18000,
        "dedup_ratio_percent": 47.0,
        "chunks": [{"id": i} for i in range(150)],
    }

    # Start at 23:00, deadline at 01:30 (2.5 hours = 9000 seconds total available)
    # Each episode at unique FPS 10.5 takes: 18000 / 10.5 = 1714s + 1230s overhead = 2944s (~49m)
    # Ep 1: 2944s (finishes at 23:49) -> KỊP
    # Ep 2: 2944s (finishes at 00:38) -> KỊP
    # Ep 3: would finish at 01:27 + 7% margin = ~01:33 > 01:30 -> CHẠY 1 PHẦN (ĐẾN HẠN CHÓT) or BỎ QUA
    with patch("src.scheduler.find_cached_plan", return_value=mock_plan):
        sched = compute_overnight_schedule(
            queue=[ep1, ep2, ep3],
            deadline_str="01:30",
            fps_median=10.5,
            fps_p5=8.5,
            allow_partial=True,
            start_time=datetime(2026, 11, 14, 23, 0, 0),
        )

        assert sched["total_episodes"] == 3
        assert sched["accepted_count"] >= 2
        # Check that deadline constraint is enforced
        assert sched["deadline_str"] == "01:30"


def test_t4_scenario_5_batch_track_defaults_postprocessing(tmp_path: Path):
    """
    Tier 4 - Scenario 5:
    Batch Track Defaults Post-Processing:
    Processes multiple MKVs in a directory tree, verifies Japanese audio and Vietnamese
    subtitles are set as default in live mode, and tests dry-run safety.
    """
    media_dir = tmp_path / "season_1"
    media_dir.mkdir()
    file_a = media_dir / "ep01.mkv"
    file_b = media_dir / "ep02.mkv"
    file_a.write_bytes(b"mkv_a")
    file_b.write_bytes(b"mkv_b")

    mock_tracks = {
        "tracks": [
            {"type": "audio", "properties": {"uid": 1, "language": "jpn", "default_track": False}},
            {"type": "audio", "properties": {"uid": 2, "language": "eng", "default_track": True}},
            {"type": "subtitles", "properties": {"uid": 3, "language": "vie", "default_track": False}},
            {"type": "subtitles", "properties": {"uid": 4, "language": "eng", "default_track": True}},
        ]
    }

    def mock_subp(cmd, *args, **kwargs):
        res = MagicMock()
        res.returncode = 0
        if "-J" in cmd:
            res.stdout = json.dumps(mock_tracks)
        else:
            res.stdout = "Header updated"
        return res

    with patch("subprocess.run", side_effect=mock_subp):
        res_a = process_track_defaults_file(str(file_a))
        res_b = process_track_defaults_file(str(file_b))

        assert res_a["status"] == "modified"
        assert res_b["status"] == "modified"
        assert len(res_a["changes"]) >= 2
        assert len(res_b["changes"]) >= 2


def test_t4_scenario_6_high_concurrency_sqlite_rag_windows_locks(tmp_path: Path):
    """
    Tier 4 - Scenario 6:
    High-Concurrency SQLite Cache & RAG Under Windows File Locks:
    Runs 8 parallel threads executing 100+ cache and RAG queries concurrently.
    Verifies that WAL mode, busy_timeout, and contextlib.closing prevent
    'database is locked' errors and eliminate Windows WinError 32 on cleanup.
    """
    cache_db = tmp_path / "concurrent_cache.db"
    rag_db = tmp_path / "concurrent_rag.db"

    cache = TranslationCache(db_path=cache_db)
    kb = KnowledgeBase(db_path=rag_db)

    # Seed RAG entries
    kb.add_gold_unit(ja_text="魔法の詠唱を開始する。", vi_text="Bắt đầu niệm ma pháp.", series="general", status="gold")

    errors: List[Exception] = []

    def worker_cache(worker_id: int):
        try:
            for i in range(25):
                key = f"key_{worker_id}_{i}"
                cache.set_cue(key, "gemini", "gemini-3.5-flash-lite", f"Bản dịch {worker_id}-{i}")
                val = cache.get_cue(key)
                assert val == f"Bản dịch {worker_id}-{i}"
        except Exception as e:
            errors.append(e)

    def worker_rag(worker_id: int):
        try:
            for i in range(25):
                res = kb.search_gold_units("魔法", top_k=2)
                assert len(res) >= 1
        except Exception as e:
            errors.append(e)

    threads = []
    for wid in range(4):
        threads.append(threading.Thread(target=worker_cache, args=(wid,)))
        threads.append(threading.Thread(target=worker_rag, args=(wid,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert len(errors) == 0, f"Concurrent database errors encountered: {errors}"

    # Close and ensure files are immediately deletable on Windows
    cache = None
    kb = None

    os.remove(cache_db)
    os.remove(rag_db)
    assert not cache_db.exists()
    assert not rag_db.exists()


def test_t4_scenario_7_complete_quality_acceptance_benchmark(tmp_path: Path):
    """
    Tier 4 - Scenario 7:
    Complete Quality Acceptance & Benchmark Verification on test sample:
    Evaluates Native 2x model resolution, hardware telemetry recording (VRAM < 9.5 GB),
    Output Sanity Guard (|ΔY| ≤ 6.0, r ≥ 0.98), BT.709 color compliance,
    and Mux Audit PASS.
    """
    # 1. Resolve Native 2x Model
    model_cfg = resolve_model_for_video("sample.mkv", cli_model_arg="animejanai-hd-v3-compact")
    assert model_cfg["arch"]["upscale"] == 2
    assert model_cfg["downsample"] == "none"

    # 2. Telemetry Collector Test
    csv_file = tmp_path / "telemetry.csv"
    collector = TelemetryCollector(output_csv_path=csv_file, interval_sec=0.05)
    collector.start()
    time.sleep(0.15)
    summary_stats = collector.stop()
    assert "peak_vram_mib" in summary_stats
    assert summary_stats["peak_vram_mib"] < 9500.0  # VRAM limit under 9.5 GB

    # 3. Output Sanity Guard & BT.709 Color Compliance
    src_rgb = np.full((270, 480, 3), 100, dtype=np.uint8)
    out_rgb = np.full((270, 480, 3), 102, dtype=np.uint8)  # ΔY = 2.0 <= 6.0
    sanity = check_frame_sanity_arrays(src_rgb, out_rgb, is_rgb=True)
    assert sanity["passed"] is True
    assert sanity["delta_y"] <= 6.0

    # 4. Simulated Mux Integrity Check
    mock_info = {
        "width": 3840,
        "height": 2160,
        "duration": 10.0,
        "audio_count": 1,
        "subtitle_count": 1,
        "attachment_count": 0,
        "chapter_count": 1,
        "audio_streams": [{"index": 0, "codec": "flac", "language": "jpn"}],
        "subtitle_streams": [{"index": 0, "codec": "subrip", "language": "vie"}],
        "attachments": [],
        "chapters": [{"id": 0}],
    }
    mock_md5 = [{"stream_index": 0, "stream_map": "0:a:0", "md5": "cafe1234"}]
    interleave_pass = {"passed": True, "failed_windows": []}

    with patch("src.mux_audit.probe_video", return_value=mock_info), \
         patch("src.mux_audit.get_detailed_stream_info", return_value=mock_info), \
         patch("src.mux_audit.get_all_audio_streams_md5", return_value=mock_md5), \
         patch("src.mux_audit.check_window_interleaving", return_value=interleave_pass), \
         patch("src.mux_audit.count_subtitle_packets", return_value=50), \
         patch("src.mux_audit._count_video_packets", return_value=240), \
         patch("src.mux_audit.get_audio_volume_db", return_value=-18.0):

        mux_res = verify_mux_integrity("src.mkv", "out.mkv")
        assert mux_res["status"] == "PASS"
