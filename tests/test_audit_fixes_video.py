# -*- coding: utf-8 -*-
"""
tests/test_audit_fixes_video.py
Kiểm thử đơn vị offline cho toàn bộ bản vá P0/P1/P2 của phân hệ Video Core, Pipeline & CLI.
Tuyệt đối không dùng GPU infer thật (>100 frames), không gọi Gemini API và không chạm dữ liệu sản xuất.
"""

import io
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import src.config as app_config
import src.dedup as dedup_mod
import src.ffmpeg_ops as ffmpeg_ops
import src.mux_audit as mux_audit
import src.night_report as night_report
import src.pipeline as pipeline_mod
import src.scheduler as scheduler_mod
import upscale as upscale_cli
from src.build_trt_engine import (
    get_trt_engine_path,
    verify_trt_engine_sha256_sidecar,
    write_trt_engine_sha256_sidecar,
)
from src.cleanup import organize_output_artifacts
from src.cuda_engine import _cleanup_gpu_worker, get_gpu_worker
from src.logger import RunLogger
from src.models_resolver import (
    find_existing_4k_output_path,
    is_valid_completed_4k_output,
    resolve_expected_4k_output_path,
)
from src.ramdisk import RAMDisk
from src.subtitles import find_or_extract_subtitles, transcribe_japanese_audio
from src.trt_zero_copy_upscaler import (
    DoubleBufferedPipeUpscaler,
    GPUColorConverterP010LE,
    _PipeDrainWorker,
)


# ---------------------------------------------------------------------------
# Fix 1: Preview 10s Source Path Preservation & Always-Delete Cleanup
# ---------------------------------------------------------------------------

def test_fix1_preview_preserves_original_source_and_cleans_clip_on_error(tmp_path, monkeypatch):
    """Đảm bảo preview_temp_input luôn bị xóa kể cả khi lỗi sớm và audit.json lưu original_input_path."""
    orig_src = tmp_path / "show_ep01.mkv"
    orig_src.write_bytes(b"SRC_MKV")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(pipeline_mod, "DEFAULT_OUTPUT_DIR", str(out_dir))

    fake_meta = {
        "width": 64,
        "height": 64,
        "duration": 60.0,
        "fps": 24.0,
        "fps_num": 24,
        "fps_den": 1,
        "nb_frames": 2,
        "size_bytes": 1000,
        "audio_count": 1,
        "subtitle_count": 0,
        "attachment_count": 0,
        "chapter_count": 0,
        "codec": "h264",
        "pix_fmt": "yuv420p",
        "profile": "high",
    }

    created_clips = []

    def fake_subproc_run(cmd, *args, **kwargs):
        out_p = Path(cmd[-1])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"PREVIEW_CLIP")
        created_clips.append(out_p)
        return SimpleNamespace(returncode=0)

    # Nhánh A: lỗi sớm ngay ở run_preflight_checks -> vẫn giải phóng lock và xóa preview clip
    release_calls = []
    with patch.object(pipeline_mod, "validate_environment"), \
         patch.object(pipeline_mod, "resolve_model_for_video", return_value={"id": "realesr-animevideov3", "name": "AnimeV3", "sha256": "abc"}), \
         patch.object(pipeline_mod, "probe_video", return_value=fake_meta), \
         patch.object(pipeline_mod.subprocess, "run", side_effect=fake_subproc_run), \
         patch.object(pipeline_mod, "acquire_gpu_lock"), \
         patch.object(pipeline_mod, "release_gpu_lock", side_effect=lambda: release_calls.append(True)), \
         patch.object(pipeline_mod, "run_preflight_checks", side_effect=RuntimeError("preflight fail")):
        with pytest.raises(RuntimeError, match="preflight fail"):
            pipeline_mod.run_upscale_pipeline(
                input_path=str(orig_src),
                output_path=str(out_dir),
                preview_seconds=10,
                anti_sleep=False,
                use_ramdisk=False,
            )

    assert len(release_calls) == 1
    assert len(created_clips) == 1
    assert not created_clips[0].exists(), "preview_temp_input phải được xóa trong finally khi gặp lỗi sớm!"

    # Nhánh B: chạy thành công hoàn tất -> audit.json ghi source_file là orig_src và vẫn xóa preview_temp_input
    created_clips.clear()

    def fake_split(in_p, segs_d, seg_t):
        os.makedirs(segs_d, exist_ok=True)
        s0 = os.path.join(segs_d, "seg_0000.mp4")
        Path(s0).write_bytes(b"SEG0")
        Path(os.path.join(segs_d, ".split_complete")).write_text(json.dumps({"segment_count": 1}), encoding="utf-8")
        return [s0]

    def fake_extract(c_seg, c_in, **kwargs):
        os.makedirs(c_in, exist_ok=True)
        Path(os.path.join(c_in, "frame00000001.png")).write_bytes(b"PNG")
        return 1

    def fake_upscale_frames(input_dir, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        Path(os.path.join(output_dir, "frame00000001.png")).write_bytes(b"PNG4K")
        return {"elapsed_seconds": 0.1, "fps": 10.0, "steady_fps": 10.0, "t0_seconds": 0.0, "peak_rss_mb": 100.0}

    def fake_encode(frames_dir, fps, output_path, **kwargs):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Path(output_path).write_bytes(b"UP0" * 5000)

    def fake_mux(segs, work_d, in_p, out_p):
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        Path(out_p).write_bytes(b"MUXED_OK" * 500)

    with patch.object(pipeline_mod, "validate_environment"), \
         patch.object(pipeline_mod, "resolve_model_for_video", return_value={"id": "realesr-animevideov3", "name": "AnimeV3", "sha256": "abc"}), \
         patch.object(pipeline_mod, "probe_video", return_value=fake_meta), \
         patch.object(pipeline_mod.subprocess, "run", side_effect=fake_subproc_run), \
         patch.object(pipeline_mod, "acquire_gpu_lock"), \
         patch.object(pipeline_mod, "release_gpu_lock"), \
         patch.object(pipeline_mod, "run_preflight_checks", return_value=True), \
         patch.object(pipeline_mod, "split_video_segments", side_effect=fake_split), \
         patch.object(pipeline_mod, "get_segments_frame_counts", return_value=[1]), \
         patch.object(pipeline_mod, "extract_frames", side_effect=fake_extract), \
         patch.object(pipeline_mod, "upscale_frames_with_fallback", side_effect=fake_upscale_frames), \
         patch.object(pipeline_mod, "encode_chunk_hevc", side_effect=fake_encode), \
         patch.object(pipeline_mod, "mux_segments_directly", side_effect=fake_mux), \
         patch.object(pipeline_mod, "cleanup_checkpoint_dir"), \
         patch.object(pipeline_mod, "send_system_notification"), \
         patch("src.mux_audit.verify_mux_integrity", return_value={"status": "PASS", "source_meta": fake_meta, "output_meta": fake_meta}), \
         patch("src.output_sanity.check_frame_sanity", return_value={"passed": True, "delta_y": 0.2, "max_delta_y": 6.0, "corr": 0.999, "min_corr": 0.98, "elapsed_ms": 1.0}):
        res = pipeline_mod.run_upscale_pipeline(
            input_path=str(orig_src),
            output_path=str(out_dir),
            preview_seconds=10,
            dedup=False,
            zero_disk=False,
            engine="cpu",
            anti_sleep=False,
            use_ramdisk=False,
        )

    assert os.path.isfile(res)
    assert not created_clips[0].exists()
    audit_path = Path(os.path.splitext(res)[0] + ".audit.json")
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_data["source_file"] == os.path.abspath(str(orig_src))
    assert ".preview_clip_" in audit_data["preview_clip_file"]


# ---------------------------------------------------------------------------
# Fix 2: Zero-Disk Pipe Safety & NaN Protection
# ---------------------------------------------------------------------------

def test_fix2_write_frame_to_pipe_zero_written_raises():
    """stdin.write() trả về 0 hoặc None phải raise RuntimeError (BrokenPipeError) ngay lập tức."""
    buf = np.zeros((4, 6, 4), dtype=np.uint8)
    for ret_val in (0, None):
        proc = SimpleNamespace(
            stdin=SimpleNamespace(write=lambda _chunk, rv=ret_val: rv),
            stderr=io.BytesIO(b"NVENC encoder error\n"),
            poll=lambda: 1,
        )
        with pytest.raises(RuntimeError, match="BrokenPipeError|NVENC pipe"):
            DoubleBufferedPipeUpscaler._write_frame_to_pipe(proc, buf, repeats=1)


def test_fix2_spawn_nvenc_pipe_writer_drains_stderr_without_deadlock(tmp_path):
    """spawn_nvenc_pipe_writer gắn daemon thread rút stderr vào deque(maxlen=64)."""
    huge_stderr = b"".join(f"ffmpeg log line {i}\n".encode("utf-8") for i in range(120))
    fake_proc = SimpleNamespace(
        stdin=io.BytesIO(),
        stderr=io.BytesIO(huge_stderr),
        poll=lambda: 0,
    )
    with patch("src.trt_zero_copy_upscaler.subprocess.Popen", return_value=fake_proc):
        proc = DoubleBufferedPipeUpscaler.spawn_nvenc_pipe_writer(
            str(tmp_path / "seg.mp4"), "24000/1001", width=32, height=32
        )
    proc._stderr_thread.join(timeout=2.0)
    assert not proc._stderr_thread.is_alive()
    assert len(proc._stderr_tail) == 64
    assert proc._stderr_tail[-1] == "ffmpeg log line 119"


def test_fix2_pipe_drain_worker_abort_and_process_pipe_calls_abort():
    """_PipeDrainWorker.abort() giải phóng event/pipe; process_frames_to_pipe gọi abort() khi có lỗi."""
    mock_pipe = MagicMock()
    mock_proc = MagicMock()
    mock_proc.stdin = mock_pipe
    mock_proc.poll.return_value = None

    block_gate = threading.Event()

    def slow_write(_proc, _arr, _rep):
        block_gate.wait(timeout=2.0)

    worker = _PipeDrainWorker(mock_proc, slow_write)
    evt1 = threading.Event()
    evt2 = threading.Event()
    worker.submit(np.zeros((2, 2), dtype=np.int16), 1, evt1)
    worker.submit(np.zeros((2, 2), dtype=np.int16), 1, evt2)
    worker.abort(encoder_proc=mock_proc)
    assert evt1.is_set() and evt2.is_set()
    mock_proc.terminate.assert_called()
    with pytest.raises(RuntimeError, match="aborted"):
        worker.check_error()

    # process_frames_to_pipe phải gọi drain_worker.abort(encoder_proc) thay vì close() khi lỗi xảy ra
    class FailingModel(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("simulated inference error")

    upscaler = DoubleBufferedPipeUpscaler(
        FailingModel(),
        batch_size=1,
        device=torch.device("cpu"),
        input_hw=(4, 4),
        scale=2,
    )
    dead_proc = MagicMock()
    dead_proc.stdin = MagicMock()
    dead_proc.poll.return_value = None
    abort_spy = []
    orig_abort = _PipeDrainWorker.abort

    def tracked_abort(self, encoder_proc=None):
        abort_spy.append(encoder_proc)
        return orig_abort(self, encoder_proc=encoder_proc)

    with patch.object(_PipeDrainWorker, "abort", tracked_abort):
        with pytest.raises(RuntimeError, match="simulated inference error"):
            upscaler.process_frames_to_pipe(
                [np.zeros((4, 4, 3), dtype=np.uint8)],
                [1],
                dead_proc,
                cyclic_dedup=False,
            )
    assert abort_spy == [dead_proc]


def test_fix2_p010le_nan_inf_protection():
    """GPUColorConverterP010LE._p010le_core phải làm sạch NaN/+Inf/-Inf trước nhân ma trận BT.709."""
    rgb = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    rgb[0, :, 0, 0] = float("nan")
    rgb[0, :, 0, 1] = float("inf")
    rgb[0, :, 1, 0] = float("-inf")
    out = GPUColorConverterP010LE._p010le_core(rgb)
    # P010LE lưu unsigned 10-bit dịch trái 6 bit trong tensor int16 -> mask 0xFFFF trước >> 6
    y_plane = ((out[0, :4].to(torch.int32) & 0xFFFF) >> 6)
    uv_plane = ((out[0, 4:].to(torch.int32) & 0xFFFF) >> 6)
    assert torch.all((y_plane >= 64) & (y_plane <= 940))
    assert torch.all((uv_plane >= 64) & (uv_plane <= 960))
    assert int(y_plane[0, 0]) == 64   # NaN -> 0.0 -> TV black Y=64
    assert int(y_plane[0, 1]) == 940  # +Inf -> 1.0 -> TV white Y=940
    assert int(y_plane[1, 0]) == 64   # -Inf -> 0.0 -> TV black Y=64


# ---------------------------------------------------------------------------
# Fix 3: BackgroundQueuePreflightManager Race, Atomic plan.json & .split_complete
# ---------------------------------------------------------------------------

def test_fix3_preflight_manager_wait_and_claim_and_split_complete(tmp_path, monkeypatch):
    """wait_for_video đợi xong tập đang quét; mark_main_started bỏ qua tập kế; .split_complete kiểm soát resume."""
    monkeypatch.delenv("BENCHMARK_MODE", raising=False)
    vid1 = tmp_path / "ep1.mkv"
    vid2 = tmp_path / "ep2.mkv"
    vid3 = tmp_path / "ep3.mkv"
    for v in (vid1, vid2, vid3):
        v.write_bytes(b"VID")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    started_v2 = threading.Event()
    release_v2 = threading.Event()
    scanned = []

    mgr = scheduler_mod.BackgroundQueuePreflightManager(
        queue=[str(vid1), str(vid2), str(vid3)],
        output_dir=str(out_dir),
        skip_first=True,
        benchmark_mode=False,
    )

    def fake_popen(cmd, **kwargs):
        target_vid = cmd[-4]
        scanned.append(os.path.basename(target_vid))
        if os.path.basename(target_vid) == "ep2.mkv":
            started_v2.set()
            release_v2.wait(timeout=3.0)
        plan_p = cmd[-2]
        os.makedirs(os.path.dirname(plan_p), exist_ok=True)
        Path(plan_p).write_text(json.dumps({"total_kept_frames": 10, "segments": []}), encoding="utf-8")
        proc = MagicMock()
        proc.wait.return_value = 0
        proc.poll.return_value = 0
        return proc

    with patch.object(scheduler_mod.subprocess, "Popen", side_effect=fake_popen):
        mgr.start()
        assert started_v2.wait(timeout=3.0)
        mgr.mark_main_started(str(vid3))  # Claim ep3 before worker reaches it

        wait_done = threading.Event()

        def waiter():
            mgr.wait_for_video(str(vid2), timeout=3.0)
            wait_done.set()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        assert not wait_done.is_set(), "wait_for_video phải đợi cho đến khi ep2 quét xong!"
        release_v2.set()
        assert wait_done.wait(timeout=3.0)
        t.join(timeout=2.0)
        mgr.stop()

    assert scanned == ["ep2.mkv"], f"ep3 đã claim không được quét nền nữa: {scanned}"

    # .split_complete validation & resume=False stale purge
    segs_dir = tmp_path / "raw_segs"
    segs_dir.mkdir()
    (segs_dir / "seg_0000.mp4").write_bytes(b"A")
    (segs_dir / "seg_0001.mp4").write_bytes(b"B")
    plan_file = tmp_path / "plan.json"
    assert not dedup_mod.are_raw_segments_complete(str(segs_dir), str(plan_file))

    (segs_dir / ".split_complete").write_text(json.dumps({"segment_count": 2}), encoding="utf-8")
    assert dedup_mod.are_raw_segments_complete(str(segs_dir), str(plan_file))

    (segs_dir / ".split_complete").write_text(json.dumps({"segment_count": 3}), encoding="utf-8")
    assert not dedup_mod.are_raw_segments_complete(str(segs_dir), str(plan_file))

    # Backward compatibility: legacy plan.json matching length when .split_complete absent
    (segs_dir / ".split_complete").unlink()
    plan_file.write_text(json.dumps({"chunks": [{"idx": 0}, {"idx": 1}]}), encoding="utf-8")
    assert dedup_mod.are_raw_segments_complete(str(segs_dir), str(plan_file))


def test_fix3_split_video_segments_writes_marker_and_purges_stale(tmp_path):
    """split_video_segments xóa seg_* cũ và ghi nguyên tử .split_complete khớp số phân đoạn."""
    segs_dir = tmp_path / "raw_segs"
    segs_dir.mkdir()
    stale = segs_dir / "seg_0099.mp4"
    stale.write_bytes(b"STALE")

    def fake_run(cmd, **kwargs):
        Path(segs_dir / "seg_0000.mp4").write_bytes(b"NEW0")
        Path(segs_dir / "seg_0001.mp4").write_bytes(b"NEW1")
        return SimpleNamespace(returncode=0)

    with patch.object(ffmpeg_ops.subprocess, "run", side_effect=fake_run):
        segs = ffmpeg_ops.split_video_segments("src.mkv", str(segs_dir), segment_time=8)

    assert not stale.exists()
    assert len(segs) == 2
    marker = json.loads((segs_dir / ".split_complete").read_text(encoding="utf-8"))
    assert marker["segment_count"] == 2


# ---------------------------------------------------------------------------
# Fix 4: Shared Output Path Resolver (_4K_animejanai.mkv & .audit.json FAIL)
# ---------------------------------------------------------------------------

def test_fix4_models_resolver_and_sub_target_4k(tmp_path):
    """Resolver hỗ trợ đúng hậu tố _4K_animejanai.mkv, kích thước > 1 MiB và từ chối audit FAIL."""
    src_vid = tmp_path / "ep02.mkv"
    src_vid.write_bytes(b"SRC")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    exp_aj = resolve_expected_4k_output_path(str(src_vid), output_arg=str(out_dir), model_id="animejanai-hd-v3-compact")
    exp_re = resolve_expected_4k_output_path(str(src_vid), output_arg=str(out_dir), model_id="realesr-animevideov3")
    exp_explicit = resolve_expected_4k_output_path(str(src_vid), output_arg=str(tmp_path / "custom.mkv"), model_id="animejanai-hd-v3-compact")
    assert exp_aj.endswith("ep02_4K_animejanai.mkv")
    assert exp_re.endswith("ep02_4K.mkv")
    assert exp_explicit == os.path.abspath(str(tmp_path / "custom.mkv"))

    # File > 1 MiB nhưng audit.json báo FAIL -> không được coi là đã hoàn tất
    aj_path = Path(exp_aj)
    aj_path.write_bytes(b"X" * (1048576 + 10))
    audit_path = Path(os.path.splitext(exp_aj)[0] + ".audit.json")
    audit_path.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    assert not is_valid_completed_4k_output(str(aj_path))
    assert find_existing_4k_output_path(str(src_vid), output_arg=str(out_dir), model_id="animejanai-hd-v3-compact") is None

    # Sửa audit thành PASS -> nhận ra _4K_animejanai.mkv
    audit_path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    found = find_existing_4k_output_path(str(src_vid), output_arg=str(out_dir), model_id="animejanai-hd-v3-compact")
    assert found == str(aj_path)

    sub_dir, t4k = upscale_cli._resolve_sub_output_dir_and_target_4k(
        str(out_dir), str(tmp_path), "ep02", input_path=str(src_vid), model_id="animejanai-hd-v3-compact"
    )
    assert sub_dir == out_dir.resolve()
    assert os.path.abspath(t4k) == os.path.abspath(str(aj_path))


# ---------------------------------------------------------------------------
# Fix 5: Staging Mux Output & mux_audit Integrity
# ---------------------------------------------------------------------------

def test_fix5_staging_mux_prevents_corrupt_output_on_audit_fail(tmp_path):
    """Khi verify_mux_integrity trả FAIL, staging .mux_tmp.mkv phải bị xóa và *_4K.mkv đích không tồn tại."""
    src_vid = tmp_path / "ep03.mkv"
    src_vid.write_bytes(b"SRC")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    expected_final = out_dir / "ep03_4K.mkv"

    fake_meta = {
        "width": 64, "height": 64, "duration": 1.0, "fps": 24.0, "fps_num": 24, "fps_den": 1,
        "nb_frames": 1, "size_bytes": 500, "audio_count": 1, "subtitle_count": 0,
        "attachment_count": 0, "chapter_count": 0, "codec": "h264", "pix_fmt": "yuv420p", "profile": "high",
    }

    def fake_split(in_p, segs_d, seg_t):
        os.makedirs(segs_d, exist_ok=True)
        s0 = os.path.join(segs_d, "seg_0000.mp4")
        Path(s0).write_bytes(b"SEG0")
        Path(os.path.join(segs_d, ".split_complete")).write_text(json.dumps({"segment_count": 1}), encoding="utf-8")
        return [s0]

    def fake_extract(c_seg, c_in, **kwargs):
        os.makedirs(c_in, exist_ok=True)
        Path(os.path.join(c_in, "frame00000001.png")).write_bytes(b"PNG")
        return 1

    def fake_upscale_frames(input_dir, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        Path(os.path.join(output_dir, "frame00000001.png")).write_bytes(b"PNG4K")
        return {"elapsed_seconds": 0.1, "fps": 10.0, "steady_fps": 10.0, "t0_seconds": 0.0, "peak_rss_mb": 100.0}

    def fake_encode(frames_dir, fps, output_path, **kwargs):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Path(output_path).write_bytes(b"UP0" * 5000)

    seen_mux_targets = []

    def fake_mux(segs, work_d, in_p, out_p):
        seen_mux_targets.append(out_p)
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        Path(out_p).write_bytes(b"CORRUPT_STAGING")

    with patch.object(pipeline_mod, "validate_environment"), \
         patch.object(pipeline_mod, "resolve_model_for_video", return_value={"id": "realesr-animevideov3", "name": "AnimeV3", "sha256": "abc"}), \
         patch.object(pipeline_mod, "probe_video", return_value=fake_meta), \
         patch.object(pipeline_mod, "acquire_gpu_lock"), \
         patch.object(pipeline_mod, "release_gpu_lock"), \
         patch.object(pipeline_mod, "run_preflight_checks", return_value=True), \
         patch.object(pipeline_mod, "split_video_segments", side_effect=fake_split), \
         patch.object(pipeline_mod, "get_segments_frame_counts", return_value=[1]), \
         patch.object(pipeline_mod, "extract_frames", side_effect=fake_extract), \
         patch.object(pipeline_mod, "upscale_frames_with_fallback", side_effect=fake_upscale_frames), \
         patch.object(pipeline_mod, "encode_chunk_hevc", side_effect=fake_encode), \
         patch.object(pipeline_mod, "mux_segments_directly", side_effect=fake_mux), \
         patch("src.mux_audit.verify_mux_integrity", return_value={"status": "FAIL", "failure_reasons": ["audio mismatch"], "source_meta": fake_meta, "output_meta": fake_meta}), \
         patch("src.output_sanity.check_frame_sanity", return_value={"passed": True, "delta_y": 0.2, "max_delta_y": 6.0, "corr": 0.999, "min_corr": 0.98, "elapsed_ms": 1.0}):
        with pytest.raises(RuntimeError, match="Mux audit container thất bại"):
            pipeline_mod.run_upscale_pipeline(
                input_path=str(src_vid),
                output_path=str(out_dir),
                dedup=False,
                zero_disk=False,
                engine="cpu",
                anti_sleep=False,
                use_ramdisk=False,
            )

    assert len(seen_mux_targets) == 1 and seen_mux_targets[0].endswith(".mux_tmp.mkv")
    assert not Path(seen_mux_targets[0]).exists(), "File staging .mux_tmp.mkv phải bị xóa khi FAIL!"
    assert not expected_final.exists(), "Không được để lại *_4K.mkv hỏng khi Mux Audit FAIL!"
    audit_json = json.loads((out_dir / "ep03_4K.audit.json").read_text(encoding="utf-8"))
    assert audit_json["status"] == "FAIL"


def test_fix5_mux_audit_checks_silent_video_subs_md5_none_and_frames(tmp_path):
    """verify_mux_integrity: bỏ interleave khi 0 audio, bắt thiếu sub / events / MD5 None / frames None."""
    src_f = tmp_path / "s.mkv"
    out_f = tmp_path / "o.mkv"
    src_f.write_bytes(b"1")
    out_f.write_bytes(b"2")

    silent_meta = {
        "duration": 10.0, "audio_count": 0, "subtitle_count": 1, "attachment_count": 0,
        "chapter_count": 0, "nb_frames": 240, "fps": 24.0, "video_count": 1,
    }

    # 1. Video không có audio + phụ đề khớp -> PASS và không test audio interleaving
    with patch.object(mux_audit, "get_detailed_stream_info", side_effect=[
        {"audio_streams": [], "subtitle_streams": [{"index": 1, "language": "eng"}], "attachment_count": 0, "chapters_count": 0, "video_count": 1},
        {"audio_streams": [], "subtitle_streams": [{"index": 1, "language": "eng"}], "attachment_count": 0, "chapters_count": 0, "video_count": 1},
    ]), \
         patch.object(mux_audit, "_count_video_packets", return_value=240), \
         patch.object(mux_audit, "count_subtitle_packets", return_value=50):
        res_silent = mux_audit.verify_mux_integrity(str(src_f), str(out_f), src_meta=silent_meta, out_meta=silent_meta)
    assert res_silent["status"] == "PASS"
    assert res_silent["window_interleaving_10s"]["windows_tested"] == 0

    # 2. Thiếu luồng sub hoặc lệch subtitle events -> FAIL
    out_meta_missing_sub = {**silent_meta, "subtitle_count": 0}
    with patch.object(mux_audit, "get_detailed_stream_info", side_effect=[
        {"audio_streams": [], "subtitle_streams": [{"index": 1, "language": "eng"}], "attachment_count": 0, "chapters_count": 0, "video_count": 1},
        {"audio_streams": [], "subtitle_streams": [], "attachment_count": 0, "chapters_count": 0, "video_count": 1},
    ]), \
         patch.object(mux_audit, "_count_video_packets", return_value=240), \
         patch.object(mux_audit, "count_subtitle_packets", return_value=50):
        res_sub_fail = mux_audit.verify_mux_integrity(str(src_f), str(out_f), src_meta=silent_meta, out_meta=out_meta_missing_sub)
    assert res_sub_fail["status"] == "FAIL"

    # Lệch subtitle event packets -> FAIL
    with patch.object(mux_audit, "get_detailed_stream_info", side_effect=[
        {"audio_streams": [], "subtitle_streams": [{"index": 1, "language": "eng"}], "attachment_count": 0, "chapters_count": 0, "video_count": 1},
        {"audio_streams": [], "subtitle_streams": [{"index": 1, "language": "eng"}], "attachment_count": 0, "chapters_count": 0, "video_count": 1},
    ]), \
         patch.object(mux_audit, "_count_video_packets", return_value=240), \
         patch.object(mux_audit, "count_subtitle_packets", side_effect=[50, 42]):
        res_ev_fail = mux_audit.verify_mux_integrity(str(src_f), str(out_f), src_meta=silent_meta, out_meta=silent_meta)
    assert res_ev_fail["status"] == "FAIL"

    # 3. Audio > 0 nhưng MD5 trả về None -> FAIL
    audio_meta = {**silent_meta, "audio_count": 1, "subtitle_count": 0}
    with patch.object(mux_audit, "get_detailed_stream_info", side_effect=[
        {"audio_streams": [{"index": 1}], "subtitle_streams": [], "attachment_count": 0, "chapters_count": 0, "video_count": 1},
        {"audio_streams": [{"index": 1}], "subtitle_streams": [], "attachment_count": 0, "chapters_count": 0, "video_count": 1},
    ]), \
         patch.object(mux_audit, "get_all_audio_streams_md5", side_effect=[
             [{"stream_index": 0, "stream_map": "0:a:0", "md5": None}],
             [{"stream_index": 0, "stream_map": "0:a:0", "md5": "abc"}],
         ]), \
         patch.object(mux_audit, "_count_video_packets", return_value=240), \
         patch.object(mux_audit, "check_window_interleaving", return_value={"passed": True, "windows_tested": 1, "failed_windows": []}):
        res_md5_fail = mux_audit.verify_mux_integrity(str(src_f), str(out_f), src_meta=audio_meta, out_meta=audio_meta)
    assert res_md5_fail["status"] == "FAIL"

    # 4. Không đếm được frame video dù có video stream -> FAIL
    no_frames_meta = {**silent_meta, "subtitle_count": 0, "nb_frames": None}
    with patch.object(mux_audit, "get_detailed_stream_info", side_effect=[
        {"audio_streams": [], "subtitle_streams": [], "attachment_count": 0, "chapters_count": 0, "video_count": 1},
        {"audio_streams": [], "subtitle_streams": [], "attachment_count": 0, "chapters_count": 0, "video_count": 1},
    ]), \
         patch.object(mux_audit, "_count_video_packets", return_value=None):
        res_frames_fail = mux_audit.verify_mux_integrity(str(src_f), str(out_f), src_meta=no_frames_meta, out_meta=no_frames_meta)
    assert res_frames_fail["status"] == "FAIL"


def test_fix5_count_video_packets_prefers_demux_over_metadata():
    """_count_video_packets ưu tiên đếm packet thực từ demux, fallback v.frames khi demux rỗng."""
    mock_v = MagicMock(frames=500)
    mock_c = MagicMock()
    mock_c.__enter__ = MagicMock(return_value=mock_c)
    mock_c.__exit__ = MagicMock(return_value=False)
    mock_c.streams.video = [mock_v]
    mock_c.demux.return_value = [SimpleNamespace(size=10), SimpleNamespace(size=0), SimpleNamespace(size=20)]

    with patch.object(mux_audit, "av", MagicMock(open=MagicMock(return_value=mock_c))):
        assert mux_audit._count_video_packets("dummy.mkv") == 2

    mock_c.demux.return_value = []
    with patch.object(mux_audit, "av", MagicMock(open=MagicMock(return_value=mock_c))):
        assert mux_audit._count_video_packets("dummy.mkv") == 500


# ---------------------------------------------------------------------------
# Fix 6: Atomic .gpu_active.lock & UPSCALER_FORCE_WHISPER_CPU
# ---------------------------------------------------------------------------

def test_fix6_atomic_gpu_lock_and_whisper_force_cpu(tmp_path, monkeypatch):
    """acquire_gpu_lock ghi nguyên tử không sót .tmp; UPSCALER_FORCE_WHISPER_CPU=1 ép CPU int8."""
    lock_p = tmp_path / ".gpu_active.lock"
    monkeypatch.setattr(app_config, "GPU_LOCK_FILE", str(lock_p))
    monkeypatch.setattr(app_config, "_GPU_LOCK_OWNED", False)

    app_config.acquire_gpu_lock()
    try:
        assert lock_p.exists()
        assert int(lock_p.read_text(encoding="utf-8").strip()) == os.getpid()
        assert not (tmp_path / f".gpu_active.lock.{os.getpid()}.tmp").exists()

        # Khi PID khác đang giữ khóa -> phải raise RuntimeError
        monkeypatch.setattr(app_config, "is_gpu_lock_held_by_other", lambda: 999999)
        with pytest.raises(RuntimeError, match="999999"):
            app_config.acquire_gpu_lock()
    finally:
        monkeypatch.setattr(app_config, "is_gpu_lock_held_by_other", lambda: None)
        app_config.release_gpu_lock()
    assert not lock_p.exists()

    # Whisper CPU forcing
    vid = tmp_path / "ep.mkv"
    vid.write_bytes(b"MKV")
    out_srt = tmp_path / "out.srt"
    captured_fw = {}

    class FakeFW:
        def __init__(self, model_size, device="cpu", compute_type="int8"):
            captured_fw["device"] = device
            captured_fw["compute_type"] = compute_type

        def transcribe(self, audio_path, **kwargs):
            return [SimpleNamespace(start=0.0, end=1.5, text="こんにちは", no_speech_prob=0.1, compression_ratio=1.1, avg_logprob=-0.2)], None

    def fake_ffmpeg_extract(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": []}))
        Path(cmd[-1]).write_bytes(b"RIFFWAVE")
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("UPSCALER_FORCE_WHISPER_CPU", "1")
    with patch("src.subtitles.subprocess.run", side_effect=fake_ffmpeg_extract), \
         patch("torch.cuda.is_available", return_value=True), \
         patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=FakeFW), "whisper": None}):
        ok = transcribe_japanese_audio(str(vid), str(out_srt))
    assert ok is True
    assert captured_fw == {"device": "cpu", "compute_type": "int8"}


# ---------------------------------------------------------------------------
# Fix 7: get_gpu_worker Singleton Batch Normalization & mid_chunk_callback
# ---------------------------------------------------------------------------

def test_fix7_get_gpu_worker_normalization_and_cleanup(tmp_path):
    """get_gpu_worker tái sử dụng singleton khi batch_size=None khớp eff_batch mặc định và dọn worker cũ khi đổi."""
    fake_weights = tmp_path / "model.pth"
    fake_weights.write_bytes(b"W")
    created_workers = []

    class FakeWorker:
        def __init__(self, model_id, weights_path, precision="fp16", max_chunks_before_restart=10, device=None, batch_size=None, use_tensorrt=True):
            self.model_id = model_id
            self.weights_path = weights_path
            self.precision = precision.lower()
            self.device = torch.device("cpu")
            self.batch_size = batch_size
            self._trt_runners = {"k": SimpleNamespace(_out_ring=[torch.zeros(1)], _batch_out_ring=[torch.zeros(1)])}
            self.model = object()
            created_workers.append(self)

    import src.cuda_engine as cuda_mod
    old_global = cuda_mod._GLOBAL_GPU_WORKER
    cuda_mod._GLOBAL_GPU_WORKER = None
    try:
        with patch.object(cuda_mod, "GPUPersistentWorker", FakeWorker), \
             patch.object(cuda_mod.torch.cuda, "is_available", return_value=False):
            w1 = get_gpu_worker("realesr-animevideov3", str(fake_weights), device="cpu", batch_size=None)
            w2 = get_gpu_worker("realesr-animevideov3", str(fake_weights), device="cpu", batch_size=1)
            assert w1 is w2, "batch_size=None và batch_size=1 (CPU) phải dùng lại đúng singleton!"
            w3 = get_gpu_worker("realesr-animevideov3", str(fake_weights), device="cpu", batch_size=2)
            assert w3 is not w1
            assert w1.model is None and len(w1._trt_runners) == 0
    finally:
        _cleanup_gpu_worker(cuda_mod._GLOBAL_GPU_WORKER)
        cuda_mod._GLOBAL_GPU_WORKER = old_global


# ---------------------------------------------------------------------------
# Fix 8: Open-GOP Sliding Ring Buffer & Anchor Seek
# ---------------------------------------------------------------------------

def test_fix8_recover_open_gop_leading_frames_sliding_ring(tmp_path):
    """_recover_open_gop_leading_frames dùng sliding ring và dừng ngay khi khớp CRA anchor."""
    src_vid = tmp_path / "src.mkv"
    src_vid.write_bytes(b"SRC")
    h, w = 4, 16
    frames = [np.full((h, w, 3), fill_value=idx * 20, dtype=np.uint8) for idx in range(5)]
    cra_anchor = frames[3].copy()
    raw_stream = b"".join(f.tobytes() for f in frames)

    mock_proc = MagicMock()
    mock_proc.stdout = io.BytesIO(raw_stream)
    mock_proc.poll.return_value = None
    mock_proc.wait.return_value = 0

    with patch.object(ffmpeg_ops.subprocess, "Popen", return_value=mock_proc):
        recovered = ffmpeg_ops._recover_open_gop_leading_frames(
            source_video_path=str(src_vid),
            cra_anchor_frame=cra_anchor,
            chunk_start_sec=8.0,
            n_discard=2,
            width=w,
            height=h,
            color_vf="scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc",
        )

    assert len(recovered) == 2
    assert np.array_equal(recovered[0], frames[1])
    assert np.array_equal(recovered[1], frames[2])
    mock_proc.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 9: RAMDisk SIGINT, Graceful Flag Clear, Dynamic Dedup, Dry-Run, Subs, TRT Sidecar
# ---------------------------------------------------------------------------

def test_fix9_ramdisk_sigint_and_dynamic_dedup_and_trt_sidecar(tmp_path, monkeypatch):
    """Kiểm tra RAMDisk SIGINT -> KeyboardInterrupt, get_dedup_params động và TRT SHA-256 sidecar."""
    rd = RAMDisk(enabled=True)
    rd.mount_path = str(tmp_path / "ram_tmp")
    os.makedirs(rd.mount_path, exist_ok=True)
    registered_handlers = {}
    with patch("src.ramdisk.signal.getsignal", return_value=signal.SIG_DFL), \
         patch("src.ramdisk.signal.signal", side_effect=lambda s, h: registered_handlers.__setitem__(s, h)), \
         patch("src.ramdisk.atexit.register"):
        rd._register_exit_hooks()
        with pytest.raises(KeyboardInterrupt):
            registered_handlers[signal.SIGINT](signal.SIGINT, None)
    assert not os.path.exists(rd.mount_path)

    # DEDUP_POLICY_PARAMS giữ nguyên keys; get_dedup_params hỗ trợ ngưỡng động
    base_keys = set(dedup_mod.DEDUP_POLICY_PARAMS.keys())
    dyn_dis = dedup_mod.get_dedup_params("dis-0.25")
    assert dyn_dis["dis_thresh"] == pytest.approx(0.25)
    dyn_drift = dedup_mod.get_dedup_params("drift-0.6")
    assert dyn_drift["drift_delta"] == pytest.approx(0.6)
    assert set(dedup_mod.DEDUP_POLICY_PARAMS.keys()) == base_keys

    # TRT engine path giữ nguyên b1-4.engine và sidecar SHA-256 kiểm tra tính toàn vẹn
    eng_p = Path(get_trt_engine_path("animejanai-hd-v3-compact", max_batch=4, input_hw=(1080, 1920), cache_dir=str(tmp_path)))
    assert eng_p.name.endswith("b1-4.engine")
    eng_p.write_bytes(b"ENGINE_V1")
    write_trt_engine_sha256_sidecar(str(eng_p), weights_sha256="deadbeef")
    assert verify_trt_engine_sha256_sidecar(str(eng_p), expected_weights_sha256="deadbeef")
    assert not verify_trt_engine_sha256_sidecar(str(eng_p), expected_weights_sha256="cafebabe")
    eng_p.write_bytes(b"TAMPERED")
    assert not verify_trt_engine_sha256_sidecar(str(eng_p), expected_weights_sha256="deadbeef")


def test_fix9_subtitles_dry_run_and_non_en_guard_and_orphan_graceful_pause(tmp_path, monkeypatch, capsys):
    """find_or_extract_subtitles(dry_run=True) không gọi ffmpeg/Whisper; non-EN không bị gán nhầm EN."""
    monkeypatch.delenv("BENCHMARK_MODE", raising=False)
    monkeypatch.delenv("SUBTRANS_WHISPER_WITH_EN", raising=False)
    monkeypatch.delenv("SUBTRANS_AUTO_WHISPER", raising=False)
    vid = tmp_path / "show.mkv"
    vid.write_bytes(b"MKV")
    probe_fre = json.dumps({"streams": [{"index": 2, "codec_name": "ass", "tags": {"language": "fre"}}]})

    with patch("src.subtitles.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=probe_fre)) as mock_run, \
         patch("src.subtitles.transcribe_japanese_audio") as mock_whisper:
        ja_s, en_s = find_or_extract_subtitles(str(vid), dry_run=True)
    assert ja_s is None and en_s is None
    mock_whisper.assert_not_called()

    # Với dry_run=False, track 'fre' không được chọn nhầm làm en_sub
    with patch("src.subtitles.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=probe_fre)) as mock_run, \
         patch("src.subtitles.transcribe_japanese_audio", return_value=False):
        ja_s2, en_s2 = find_or_extract_subtitles(str(vid), dry_run=False)
    assert en_s2 is None
    assert mock_run.call_count == 1  # Chỉ chạy ffprobe, không chạy ffmpeg extract cho track 'fre'

    # CLI --subs-only --dry-run thoát 0 mà không chạy ffmpeg/Whisper
    monkeypatch.setattr(sys, "argv", ["upscale.py", "--subs-only", str(vid), "--dry-run"])
    with patch.object(upscale_cli, "find_or_extract_subtitles", return_value=(None, None)) as mock_find:
        with pytest.raises(SystemExit) as exc:
            upscale_cli.main()
    assert exc.value.code == 0
    assert mock_find.call_args.kwargs.get("dry_run") is True
    assert "DRY RUN" in capsys.readouterr().out

    # Orphan .graceful_pause.flag tự dọn trước khi chạy thật nếu không có live GPU lock
    cleared = []
    monkeypatch.setattr(sys, "argv", ["upscale.py", str(vid), "-o", str(tmp_path / "out"), "--no-subs-after-upscale"])
    with patch.object(upscale_cli, "is_graceful_pause_requested", return_value=True), \
         patch.object(upscale_cli, "is_gpu_lock_held_by_other", return_value=None), \
         patch.object(upscale_cli, "clear_graceful_pause", side_effect=lambda: cleared.append(True)), \
         patch.object(upscale_cli, "run_upscale_pipeline", return_value=str(tmp_path / "out" / "show_4K.mkv")):
        (tmp_path / "out").mkdir(exist_ok=True)
        (tmp_path / "out" / "show_4K.mkv").write_bytes(b"OK")
        with pytest.raises(SystemExit) as exc2:
            upscale_cli.main()
    assert exc2.value.code == 0
    assert cleared == [True]


# ---------------------------------------------------------------------------
# Fix 10: Cleanup Prefix Guard, Rolling ETA, Dynamic Night Report & Scheduler
# ---------------------------------------------------------------------------

def test_fix10_cleanup_prefix_and_rolling_eta_and_night_report(tmp_path):
    """cleanup tách biệt Ep1/Ep10, RunLogger dùng rolling ETA, night_report parse được chunk hiện đại."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for name in ("Ep1.audit.json", "Ep1_4K.audit.json", "Ep10.audit.json", "Ep10_4K.audit.json"):
        (out_dir / name).write_text("{}", encoding="utf-8")
    organize_output_artifacts(str(out_dir), base_name="Ep1")
    assert (out_dir / "audit" / "Ep1.audit.json").exists()
    assert (out_dir / "audit" / "Ep1_4K.audit.json").exists()
    assert (out_dir / "Ep10.audit.json").exists()
    assert (out_dir / "Ep10_4K.audit.json").exists()

    # Rolling 8-chunk ETA trong RunLogger
    orig_logs = Path("logs")
    rl_dir = tmp_path / "logs"
    rl_dir.mkdir()
    with patch("src.logger.LOGS_DIR", str(rl_dir)):
        logger = RunLogger("test_ep")
        for idx, dur in enumerate([10.0, 20.0]):
            logger.log_chunk_done(
                chunk_idx=idx,
                total_chunks=10,
                frames=100,
                ai_time=dur * 0.8,
                ai_fps=12.0,
                enc_time=dur * 0.2,
                total_chunk_time=dur,
                output_seg="up_0000.mp4",
            )
        state = json.loads((rl_dir / "live_status.json").read_text(encoding="utf-8"))
        # Rolling mean của [10.0, 20.0] là 15.0s * 8 chunks còn lại = 120s (thay vì 20.0 * 8 = 160s)
        assert state["eta_seconds"] == pytest.approx(120.0, abs=0.5)
    assert orig_logs.name == "logs"

    # night_report parser hỗ trợ cả định dạng mới (không có T0, đơn vị GB/MB) lẫn định dạng cũ
    sample_log = (
        "2026-11-18 01:00:00 [INFO] [CHUNK 01/10] Hoàn tất 192 frames "
        "(AI CUDA: 10.00s @ 19.20 fps | Encode: 1.50s | Tổng: 12.00s | File: up_0000.mp4 120.5 MB)\n"
        "2026-11-18 01:00:12 [INFO] [CHUNK 02/10] Hoàn tất 192 frames "
        "(AI GPU: 9.50s @ 20.21 fps | T0: 0.10s, Steady: 21.00 fps, Peak RSS: 450.0 MB | "
        "Encode: 1.20s | Tổng: 11.00s | File: up_0001.mp4 1.05 GB)\n"
    )
    log_p = tmp_path / "ep.log"
    log_p.write_text(sample_log, encoding="utf-8")
    parsed = night_report.parse_episode_chunks_log(str(log_p))
    assert parsed["chunks_completed"] == 2
    assert parsed["total_chunks"] == 10
    assert parsed["total_ai_time_sec"] == pytest.approx(19.5)

    # query_subtrans_api_stats không còn hardcode '2026-10-05' và chọn cửa sổ động theo MAX(created_at)
    assert night_report.query_subtrans_api_stats(str(tmp_path / "missing.db")) == {}
    import sqlite3
    fake_db = tmp_path / "trans.db"
    con = sqlite3.connect(str(fake_db))
    con.execute("CREATE TABLE translations (model TEXT)")
    con.execute("INSERT INTO translations VALUES ('gemini-3.8-flash')")
    con.execute("CREATE TABLE api_requests (created_at TEXT, model TEXT, status TEXT, error_type TEXT)")
    con.execute("INSERT INTO api_requests VALUES ('2026-11-18 18:00:00', 'gemini-3.8-flash', 'success', NULL)")
    con.commit()
    con.close()
    stats = night_report.query_subtrans_api_stats(str(fake_db))
    assert stats["model_summary"]["gemini-3.8-flash"]["success"] == 1
    assert stats["cues_by_model"]["gemini-3.8-flash"] == 1

    # Edge cases: ffprobe 'N/A' -> 0 subtitle packets; --dis-thresh 0.0 bị từ chối
    with patch("src.mux_audit.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="N/A\n")):
        assert mux_audit._count_subtitle_packets_ffprobe("dummy.mkv", sub_stream_idx=0) == 0
    with pytest.raises(ValueError, match="Chính sách dedup không hợp lệ"):
        dedup_mod.get_dedup_params("dis-0.0")

    # check_deadline_and_eta trừ reserved_seconds và đọc plan.json trong output_dir
    ep_out = tmp_path / "custom_out"
    temp_dir = ep_out / ".temp_ep_plan"
    temp_dir.mkdir(parents=True)
    (temp_dir / "plan.json").write_text(json.dumps({"total_kept_frames": 1000}), encoding="utf-8")
    vid_plan = tmp_path / "ep_plan.mkv"
    vid_plan.write_bytes(b"V")
    target_dl = (datetime.now() + timedelta(minutes=25)).strftime("%H:%M")
    can_run, _, adj_dl = upscale_cli.check_deadline_and_eta(
        str(vid_plan),
        target_dl,
        allow_partial=False,
        output_dir=str(ep_out),
        reserved_seconds=300.0,
    )
    assert can_run is True
    assert adj_dl is not None

    # Scheduler truyền subs_quality và không dùng footer lỗi thời '1.62 fps'
    sched = scheduler_mod.compute_overnight_schedule(
        queue=[str(vid_plan)],
        deadline_str=None,
        output_dir=str(ep_out),
        preflight_now=False,
        subs_quality="high",
    )
    assert sched["subs_quality"] == "high"
    table_txt = scheduler_mod.format_schedule_decision_table(sched)
    assert "1.62 fps" not in table_txt
