# -*- coding: utf-8 -*-
"""
tests/test_final_ceiling_audit.py
Kiểm thử toàn diện cho đợt kiểm toán & tối ưu kịch trần cuối cùng:
1. Nén RLE `raw_actions_rle` trong `plan.json` & tái phân hoạch GOP < 1 ms (0 lượt gọi lại FFmpeg mpdecimate).
2. `run_preflight_worker_cli` & `build_preflight_subprocess_spec`: tách sẵn `raw_segs` theo GOP + bảo toàn `dedup_policy`.
3. `TRTRuntimeEngineWrapper`: bộ đệm vòng đôi VRAM (`_out_ring` 2-slot) khử cấp phát động + cache `set_input_shape`/`set_tensor_address`.
4. `run_upscale_pipeline(..., zero_disk=True)`: chồng lấp thực sự 3 tầng (`Decode N+1` chạy song song `GPU Infer N` và `NVENC Flush N` chạy nền trên `EncodeWorker`).
5. `_detect_hevc_encoder`: kế thừa biến môi trường `UPSCALER_HEVC_ENCODER` khử spawn `ffmpeg -encoders` ở tiến trình con.
"""

import io
import json
import os
import sys
import tempfile
import time
import types
from unittest.mock import patch

import numpy as np
import torch

from src.build_trt_engine import TRTRuntimeEngineWrapper
import src.config as app_config
from src.dedup import (
    _decode_actions_rle,
    _encode_actions_rle,
    _partition_actions_into_chunks,
    build_preflight_subprocess_spec,
    load_or_create_plan,
    run_preflight_worker_cli,
)
import src.pipeline as pipeline_mod


def test_rle_actions_repartition_on_gop_mismatch_zero_ffmpeg_decode(tmp_path):
    """
    Khi plan.json đã lưu `raw_actions_rle` (ví dụ quét sớm trên lưới cố định 192 frames),
    `load_or_create_plan` gặp `chunk_frame_counts` GOP thật `[250, 250, 219]` phải phân hoạch lại
    tức thì trong RAM (0 lần gọi `scan_dedup_plan`), ép `keep` đầu mỗi chunk và khớp 100% tổng frame.
    """
    rng = np.random.default_rng(42)
    raw_actions = ["keep" if rng.random() > 0.55 else "drop" for _ in range(719)]
    raw_actions[0] = "keep"

    rle = _encode_actions_rle(raw_actions)
    assert len(rle) < len(raw_actions) * 2
    assert _decode_actions_rle(rle) == raw_actions

    init_chunks, init_kept, init_num, init_fpc = _partition_actions_into_chunks(
        raw_actions, None, segment_seconds=8, fps_val=24000 / 1001
    )
    plan_file = str(tmp_path / "plan.json")
    initial_plan = {
        "source_file": "ep02.mkv",
        "ffmpeg_version": "mock",
        "dedup_policy": "default",
        "dedup_params": {"hi": 768, "lo": 320, "frac": 0.33},
        "total_frames": 719,
        "raw_kept_frames": sum(1 for a in raw_actions if a == "keep"),
        "raw_dropped_frames": sum(1 for a in raw_actions if a == "drop"),
        "fps_num": 24000,
        "fps_den": 1001,
        "fps": 23.976,
        "segment_seconds": 8,
        "frames_per_chunk": init_fpc,
        "total_chunks": init_num,
        "chunk_frame_counts": None,
        "total_kept_frames": init_kept,
        "dedup_ratio_percent": round((1.0 - init_kept / 719) * 100.0, 2),
        "raw_actions_rle": rle,
        "chunks": init_chunks,
    }
    with open(plan_file, "w", encoding="utf-8") as f:
        json.dump(initial_plan, f)

    gop_counts = [250, 250, 219]
    with patch("src.dedup.scan_dedup_plan") as mock_scan:
        repart = load_or_create_plan(
            input_path="ep02.mkv",
            plan_path=plan_file,
            segment_seconds=8,
            fps_num=24000,
            fps_den=1001,
            resume=True,
            chunk_frame_counts=gop_counts,
            policy="default",
        )
        assert not mock_scan.called, "Có raw_actions_rle thì TUYỆT ĐỐI không được gọi lại scan_dedup_plan!"

    assert [c["total_frames"] for c in repart["chunks"]] == gop_counts
    for c in repart["chunks"]:
        assert c["kept_indices"][0] == 0, "Frame đầu tiên của mỗi GOP chunk phải được ép giữ (keep)"
    reconstructed = sum(sum(r["count"] for r in c["runs"]) for c in repart["chunks"])
    assert reconstructed == 719


def test_preflight_worker_cli_splits_gop_and_forwards_dedup_policy(tmp_path):
    """
    Xác minh Background Queue Preflight tách sẵn `raw_segs` theo GOP và truyền đúng `dedup_policy`
    vào `load_or_create_plan`, triệt tiêu hoàn toàn độ trễ tách/quét lại giữa các tập trong `--queue`.
    """
    work_dir = tmp_path / ".temp_Anime_Ep02"
    work_dir.mkdir()
    plan_file = str(work_dir / "plan.json")
    vid_path = str(tmp_path / "Anime_Ep02.mkv")

    cmd, _ = build_preflight_subprocess_spec(vid_path, plan_file, segment_seconds=8, dedup_policy="sensitive-b")
    assert cmd[-1] == "sensitive-b"

    captured = {}

    def fake_split(v_in, segs_d, seg_s):
        s0 = os.path.join(segs_d, "seg_0000.mp4")
        s1 = os.path.join(segs_d, "seg_0001.mp4")
        with open(s0, "wb") as f0, open(s1, "wb") as f1:
            f0.write(b"00")
            f1.write(b"11")
        return [s0, s1]

    def fake_load_or_create(vid, plan_f, seg_s, fps_n, fps_d, chunk_frame_counts=None, policy="default", **kw):
        captured["chunk_frame_counts"] = chunk_frame_counts
        captured["policy"] = policy
        return {"total_frames": 420, "total_kept_frames": 200}

    with patch("src.ffmpeg_ops.probe_video", return_value={"fps_num": 24000, "fps_den": 1001}), \
         patch("src.ffmpeg_ops.split_video_segments", side_effect=fake_split) as mock_split, \
         patch("src.ffmpeg_ops.get_segments_frame_counts", return_value=[240, 180]), \
         patch("src.dedup.load_or_create_plan", side_effect=fake_load_or_create):
        res = run_preflight_worker_cli([vid_path, plan_file, "8", "sensitive-b"])

    assert mock_split.called
    assert captured["chunk_frame_counts"] == [240, 180]
    assert captured["policy"] == "sensitive-b"
    assert res["total_kept_frames"] == 200


def test_trt_wrapper_shape_address_cache_and_ping_pong_output_ring():
    """
    Xác minh TRTRuntimeEngineWrapper:
    1. Chỉ gọi `set_input_shape` và `set_tensor_address(in_name)` đúng 1 lần khi lặp lại cùng tensor/shape.
    2. Xoay vòng 2 slot `_out_ring` để không cấp phát lại `torch.empty` mỗi batch và không ghi đè kết quả liền kề.
    """
    fake_trt = types.ModuleType("tensorrt")

    class FakeLogger:
        ERROR = 1

        def __init__(self, severity):
            self.severity = severity

    class FakeContext:
        def __init__(self):
            self.shape_calls = 0
            self.addr_calls = {"input_rgb": 0, "output_rgb": 0}
            self.executed_v3 = 0

        def set_input_shape(self, name, shape):
            self.shape_calls += 1

        def set_tensor_address(self, name, ptr):
            self.addr_calls[name] = self.addr_calls.get(name, 0) + 1

        def execute_async_v3(self, stream_handle):
            self.executed_v3 += 1
            return True

    class FakeEngine:
        def create_execution_context(self):
            return FakeContext()

    class FakeRuntime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, blob):
            return FakeEngine()

    fake_trt.Logger = FakeLogger
    fake_trt.Runtime = FakeRuntime

    with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as tf:
        tf.write(b"FAKE_TRT")
        eng_path = tf.name

    try:
        with patch.dict(sys.modules, {"tensorrt": fake_trt}):
            wrapper = TRTRuntimeEngineWrapper(
                eng_path,
                device=torch.device("cpu"),
                input_hw=(16, 24),
                max_batch=4,
                scale=2,
            )
            x = torch.zeros((4, 3, 16, 24), dtype=torch.float16).contiguous()
            y1 = wrapper(x)
            y2 = wrapper(x)
            y3 = wrapper(x)

            assert wrapper.context.executed_v3 == 3
            assert wrapper.context.shape_calls == 1, "Cùng shape cố định chỉ được gọi set_input_shape 1 lần!"
            assert wrapper.context.addr_calls["input_rgb"] == 1, "Cùng địa chỉ input chỉ được bind 1 lần!"
            assert y1.data_ptr() != y2.data_ptr(), "2 lần gọi liên tiếp phải nằm ở 2 slot độc lập trong _out_ring!"
            assert y3.data_ptr() == y1.data_ptr(), "Lần gọi thứ 3 phải tái sử dụng slot 0 (0 cấp phát động mới)!"
    finally:
        if os.path.exists(eng_path):
            os.remove(eng_path)


def test_zero_disk_pipeline_overlaps_decode_and_async_nvenc_flush(tmp_path):
    """
    Xác minh pipeline `--zero-disk` chồng lấp thực sự:
    1. `extract_frames_raw_rgb24` của Chunk 1 bắt đầu NGAY TRONG LÚC GPU đang chạy `process_frames_to_pipe` của Chunk 0.
    2. `process_frames_to_pipe` của Chunk 1 bắt đầu TRƯỚC KHI `enc_proc.wait()` (flush NVENC) của Chunk 0 kết thúc.
    """
    src_vid = tmp_path / "ep_overlap.mkv"
    src_vid.write_bytes(b"FAKE_MKV" * 200)
    out_vid = tmp_path / "out" / "ep_overlap_4K.mkv"

    events = {}

    def fake_split(in_p, segs_d, seg_t):
        os.makedirs(segs_d, exist_ok=True)
        segs = []
        for idx in range(2):
            sp = os.path.join(segs_d, f"seg_{idx:04d}.mp4")
            with open(sp, "wb") as f:
                f.write(b"SEG")
            segs.append(sp)
        return segs

    def fake_extract_raw(seg_file, width, height, kept_indices=None, hwaccel=True, **kwargs):
        if "seg_0001" in seg_file:
            events["extract1_start"] = time.perf_counter()
            time.sleep(0.02)
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        return [frame, frame]

    class FakeEncProc:
        def __init__(self, out_mp4, idx):
            self.out_mp4 = out_mp4
            self.idx = idx
            self.stdin = io.BytesIO()

        def wait(self, timeout=None):
            if self.idx == 0:
                time.sleep(0.12)
                events["enc0_wait_end"] = time.perf_counter()
            with open(self.out_mp4, "wb") as wf:
                wf.write(b"ENCODED_HEVC_MP4")
            return 0

        def kill(self):
            pass

    class FakePipeUpscaler:
        def __init__(self, *args, **kwargs):
            self.output_hw = (32, 48)
            self.last_first_rgb_small = np.full((16, 24, 3), 128, dtype=np.uint8)
            self.last_distinct_count = 2
            self._spawn_idx = 0

        def ensure_resolution(self, h, w):
            return self.output_hw

        def spawn_nvenc_pipe_writer(self, out_mp4, fps_str="24000/1001", width=48, height=32):
            proc = FakeEncProc(out_mp4, self._spawn_idx)
            self._spawn_idx += 1
            return proc

        def process_frames_to_pipe(self, unique_frames, repeat_counts, enc_proc, **kwargs):
            if "gpu0_end" not in events:
                time.sleep(0.10)
                events["gpu0_end"] = time.perf_counter()
            else:
                events["gpu1_start"] = time.perf_counter()
            return sum(repeat_counts)

    class FakeCudaEngine:
        def __init__(self):
            self.model = torch.nn.Identity()
            self.batch_size = 4
            self.device = torch.device("cpu")

        def restart_if_needed(self, logger=None):
            pass

    fake_cuda = FakeCudaEngine()
    fake_meta = {
        "size_bytes": 1600,
        "width": 24,
        "height": 16,
        "fps": 24.0,
        "fps_num": 24,
        "fps_den": 1,
        "duration": 16.0,
        "nb_frames": 8,
        "audio_count": 1,
        "subtitle_count": 0,
        "attachment_count": 0,
        "chapter_count": 0,
    }
    fake_plan = {
        "total_frames": 8,
        "total_kept_frames": 4,
        "dedup_ratio_percent": 50.0,
        "chunks": [
            {
                "chunk_idx": 0,
                "total_frames": 4,
                "kept_count": 2,
                "kept_indices": [0, 2],
                "runs": [{"frame_idx": 0, "count": 2}, {"frame_idx": 2, "count": 2}],
            },
            {
                "chunk_idx": 1,
                "total_frames": 4,
                "kept_count": 2,
                "kept_indices": [0, 2],
                "runs": [{"frame_idx": 0, "count": 2}, {"frame_idx": 2, "count": 2}],
            },
        ],
    }

    def fake_mux(segs, work_d, in_p, out_p):
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        with open(out_p, "wb") as wf:
            wf.write(b"FINAL_4K_MKV" * 200)

    with patch.object(pipeline_mod, "validate_environment"), \
         patch.object(pipeline_mod, "acquire_gpu_lock"), \
         patch.object(pipeline_mod, "release_gpu_lock"), \
         patch.object(pipeline_mod, "run_preflight_checks", return_value=True), \
         patch.object(pipeline_mod, "send_system_notification"), \
         patch.object(pipeline_mod, "resolve_model_for_video", return_value={"id": "animevideov3", "name": "AnimeV3", "sha256": "abc12345", "resolution_source": "test"}), \
         patch.object(pipeline_mod, "probe_video", return_value=fake_meta), \
         patch.object(pipeline_mod, "split_video_segments", side_effect=fake_split), \
         patch.object(pipeline_mod, "get_segments_frame_counts", return_value=[4, 4]), \
         patch.object(pipeline_mod, "load_or_create_plan", return_value=fake_plan), \
         patch.object(pipeline_mod, "extract_frames_raw_rgb24", side_effect=fake_extract_raw), \
         patch.object(pipeline_mod, "mux_segments_directly", side_effect=fake_mux), \
         patch("src.cuda_engine.get_cuda_inference_engine", return_value=fake_cuda), \
         patch("src.build_trt_engine.get_hybrid_engine_or_wrapper", return_value=torch.nn.Identity()), \
         patch("src.trt_zero_copy_upscaler.DoubleBufferedPipeUpscaler", FakePipeUpscaler), \
         patch("src.mux_audit.verify_mux_integrity", return_value={"status": "PASS", "source_meta": fake_meta, "output_meta": fake_meta}):
        res_out = pipeline_mod.run_upscale_pipeline(
            input_path=str(src_vid),
            output_path=str(out_vid),
            engine="cuda",
            precision="fp16",
            zero_disk=True,
            use_tensorrt=False,
            anti_sleep=False,
            keep_temp=True,
        )

    assert os.path.exists(res_out)
    assert events["extract1_start"] < events["gpu0_end"], (
        "DecodeWorker của Chunk 1 phải chạy gối đầu song song TRONG LÚC GPU đang suy luận Chunk 0!"
    )
    assert events["gpu1_start"] < events["enc0_wait_end"], (
        "GPU Chunk 1 phải khởi chạy ngay mà không bị chặn bởi enc_proc.wait() flush của Chunk 0!"
    )


def test_config_detect_hevc_encoder_uses_inherited_env(monkeypatch):
    """Xác minh _detect_hevc_encoder dùng ngay UPSCALER_HEVC_ENCODER mà không spawn subprocess."""
    monkeypatch.setenv("UPSCALER_HEVC_ENCODER", "hevc_nvenc")
    with patch("subprocess.run", side_effect=AssertionError("Không được spawn ffmpeg khi đã có UPSCALER_HEVC_ENCODER")):
        assert app_config._detect_hevc_encoder() == "hevc_nvenc"


def test_resolve_trt_profile_max_batch_caps_4x_8k_backbone():
    """
    Xác minh resolve_trt_profile_max_batch giới hạn max_batch=2 cho mô hình PixelShuffle(4) ở Full HD
    (tránh OOM 12.74 GB trên RTX 3060 12GB) nhưng cho phép max_batch=4 cho mô hình 2x thuần.
    """
    from src.build_trt_engine import resolve_trt_profile_max_batch
    from src.cuda_engine import AnimeJaNaiCompact2x, RealESRAnimeVideoV3

    m4 = RealESRAnimeVideoV3()
    m2 = AnimeJaNaiCompact2x()
    assert resolve_trt_profile_max_batch(m4, max_batch=8, input_hw=(1080, 1920)) == 2
    assert resolve_trt_profile_max_batch(m2, max_batch=8, input_hw=(1080, 1920)) == 4
    assert resolve_trt_profile_max_batch(m4, max_batch=8, input_hw=(270, 480)) == 8


def test_pipe_upscaler_write_frame_surfaces_ffmpeg_stderr_on_errno22():
    """
    Xác minh khi FFmpeg NVENC thoát sớm khiến pipe trên Windows ném OSError(errno=22 EINVAL),
    _write_frame_to_pipe trích xuất đầy đủ thông báo lỗi từ stderr để chẩn đoán chính xác.
    """
    from src.trt_zero_copy_upscaler import DoubleBufferedPipeUpscaler

    class BrokenStdin:
        def write(self, data):
            raise OSError(22, "Invalid argument")

    class CrashedEncoderProc:
        stdin = BrokenStdin()
        stderr = io.BytesIO(b"[hevc_nvenc] OpenEncodeSessionEx failed: out of memory (10)\n")

        def poll(self):
            return 1

    arr = np.zeros((16, 16), dtype=np.uint16)
    try:
        DoubleBufferedPipeUpscaler._write_frame_to_pipe(CrashedEncoderProc(), arr)
        raise AssertionError("Phải ném RuntimeError kèm stderr của FFmpeg")
    except RuntimeError as err:
        msg = str(err)
        assert "rc=1" in msg
        assert "OpenEncodeSessionEx failed: out of memory" in msg


def test_extract_frames_raw_rgb24_skips_select_when_all_frames_kept():
    """
    Xác minh khi kept_indices giữ đủ 100% số frame của chunk (0..total_frames-1),
    extract_frames_raw_rgb24 bỏ qua biểu thức select='...' thừa thãi.
    """
    from src.ffmpeg_ops import extract_frames_raw_rgb24

    h, w = 4, 6
    raw = np.zeros((3, h, w, 3), dtype=np.uint8).tobytes()
    recorded_cmds = []

    def fake_run(cmd, stdout=None, stderr=None, check=False):
        recorded_cmds.append(list(cmd))
        return types.SimpleNamespace(returncode=0, stdout=raw, stderr=b"")

    with patch("src.ffmpeg_ops.subprocess.run", side_effect=fake_run):
        out = extract_frames_raw_rgb24(
            "seg.mp4",
            width=w,
            height=h,
            kept_indices=[0, 1, 2],
            hwaccel=False,
            total_frames=3,
        )
    assert len(out) == 3
    vf_arg = recorded_cmds[0][recorded_cmds[0].index("-vf") + 1]
    assert "select=" not in vf_arg
    assert "scale=in_color_matrix=bt709" in vf_arg


def test_recover_open_gop_leading_frames_matches_cra_anchor(tmp_path):
    """
    Xác minh _recover_open_gop_leading_frames khôi phục chính xác từng pixel của các B-frame RASL dẫn đầu
    trước điểm neo CRA keyframe trên các bản encode x265 Open-GOP, và fallback an toàn khi thiếu file nguồn.
    """
    from src.ffmpeg_ops import _recover_open_gop_leading_frames

    h, w = 8, 32
    f_rasl0 = np.full((h, w, 3), 11, dtype=np.uint8)
    f_rasl1 = np.full((h, w, 3), 22, dtype=np.uint8)
    f_cra = np.full((h, w, 3), 99, dtype=np.uint8)
    f_after = np.full((h, w, 3), 144, dtype=np.uint8)

    # Fallback khi không có file nguồn trên đĩa
    fb = _recover_open_gop_leading_frames(
        source_video_path=None,
        chunk_start_sec=80.0,
        cra_anchor_frame=f_cra,
        n_discard=2,
        width=w,
        height=h,
        color_vf="null",
    )
    assert len(fb) == 2
    assert np.array_equal(fb[0], f_cra)

    # Khôi phục từ luồng giải mã cửa sổ ngắn của file nguồn
    src_mkv = tmp_path / "source_opengop.mkv"
    src_mkv.write_bytes(b"MKV")
    window_bytes = b"".join(arr.tobytes() for arr in [f_after, f_rasl0, f_rasl1, f_cra, f_after])

    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.stdout = io.BytesIO(window_bytes)

        def wait(self):
            return 0

    with patch("src.ffmpeg_ops.subprocess.Popen", FakePopen):
        recovered = _recover_open_gop_leading_frames(
            source_video_path=str(src_mkv),
            chunk_start_sec=120.0,
            cra_anchor_frame=f_cra,
            n_discard=2,
            width=w,
            height=h,
            color_vf="null",
        )
    assert len(recovered) == 2
    assert np.array_equal(recovered[0], f_rasl0)
    assert np.array_equal(recovered[1], f_rasl1)
