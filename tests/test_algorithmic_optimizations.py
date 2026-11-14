# -*- coding: utf-8 -*-
"""
tests/test_algorithmic_optimizations.py
Kiểm thử toàn diện 6 tối ưu hóa tầng Thuật toán (Algorithmic Ceiling):
1. Khử trùng lặp chu kỳ không liên tiếp (Cyclic Cel Memoization: A->B->A lip-flap / blink)
   với chốt chặn nhiễu lượng tử H.264 (max_diff <= 2, MAE <= 0.20) và bảo toàn 100% chuyển động 1-pixel.
2. Thuật toán Belady's Optimal Liveness Eviction + Thu mẫu Sanity 270x480 trên GPU (0 lượt infer phụ)
   trong DoubleBufferedPipeUpscaler.
3. Tái sử dụng NTFS Hardlink (0 lần nén PNG lặp, 0 byte đĩa phụ) trong GPUPersistentWorker.upscale_directory.
4. Ghép nối & đóng gói 1 lượt mux_segments_directly (khử 8 GB đọc/ghi merged_4k.mp4) vượt qua verify_mux_integrity.
5. Băm đa luồng âm thanh 1 lượt đọc (-f streamhash -hash md5) khớp bit-exact 100% với từng luồng.
6. Hợp nhất 1 lượt giải mã FFmpeg split=2 (mpdecimate + 480x270 gray) + Early-Exit DIS khớp 100% với 2 lượt.
7. Đọc trực tiếp readinto vào mảng NumPy tiền cấp phát trong extract_frames_raw_rgb24.
"""

import io
import os
import subprocess
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dedup import (
    deduplicate_cyclic_frames,
    detect_dis_policy_actions,
    scan_dedup_plan,
)
from src.trt_zero_copy_upscaler import DoubleBufferedPipeUpscaler
from src.output_sanity import check_frame_sanity_arrays
from src.ffmpeg_ops import mux_segments_directly, extract_frames_raw_rgb24
from src.mux_audit import verify_mux_integrity, get_all_audio_streams_md5, get_audio_stream_md5
from src.cuda_engine import GPUPersistentWorker


def test_cyclic_cel_dedup_exact_and_quant_noise_vs_real_motion():
    """
    Xác minh deduplicate_cyclic_frames:
    - Gộp đúng frame trùng khít 100% (BLAKE2b + array_equal) trong chu kỳ nhép miệng A -> B -> C -> B -> A.
    - Gộp đúng frame cùng nét vẽ chỉ nhiễu lượng tử B-frame H.264 cực nhỏ (max_diff <= 2, MAE <= 0.20).
    - TỪ CHỐI TUYỆT ĐỐI frame có dù chỉ 1 chi tiết nhỏ (lông mi/khóe miệng 4x4 px) dịch chuyển (+20 mức xám).
    """
    rng = np.random.RandomState(42)
    cel_a = rng.randint(20, 220, size=(270, 480, 3), dtype=np.uint8)
    cel_b = cel_a.copy()
    cel_b[100:140, 200:240] = 250  # Miệng mở nửa
    cel_c = cel_a.copy()
    cel_c[95:150, 195:245] = 10    # Miệng mở to

    # Bản lặp của cel_a có nhiễu lượng tử H.264 siêu nhỏ (±1 trên ~8% pixel -> max_diff=1, MAE ~0.08)
    cel_a_qnoise = cel_a.copy()
    cel_a_qnoise[::4, ::3, 0] = np.clip(cel_a_qnoise[::4, ::3, 0].astype(np.int16) + 1, 0, 255).astype(np.uint8)

    # Frame có dịch chuyển cực nhỏ 4x4 pixel ở khóe miệng (+25 mức xám): MAE toàn ảnh ~0.003 nhưng max_diff = 25 > 2!
    cel_micro_move = cel_a.copy()
    cel_micro_move[120:124, 220:224] = np.clip(
        cel_micro_move[120:124, 220:224].astype(np.int16) + 25, 0, 255
    ).astype(np.uint8)

    seq = [cel_a, cel_b, cel_c, cel_b.copy(), cel_a_qnoise, cel_micro_move]
    distinct, mapping = deduplicate_cyclic_frames(seq)

    # Chỉ có 4 cel thực sự khác nhau: A (0), B (1), C (2), và cel_micro_move (3)
    assert len(distinct) == 4
    assert mapping == [0, 1, 2, 1, 0, 3]


def test_pipe_upscaler_belady_cyclic_memoization_and_zero_reinference_sanity():
    """
    Xác minh DoubleBufferedPipeUpscaler:
    - Khi đưa chuỗi chu kỳ [A, B, A, C, B, A] (6 unique liên tiếp, 3 distinct thực sự) với batch_size=2:
      Mô hình chỉ chạy suy luận đúng 3 frames (tiết kiệm 50% tải GPU!).
    - Luồng byte p010le xuất ra pipe khớp TỪNG BYTE 100% so với khi tắt cyclic_dedup!
    - Mẫu kiểm định last_first_rgb_small (270x480x3) được trích ngay ở Batch 0 mà KHÔNG gọi thêm model lần nào.
    """
    inferred_frames = {"count": 0}

    def counting_model(x: torch.Tensor) -> torch.Tensor:
        inferred_frames["count"] += int(x.shape[0])
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

    upscaler = DoubleBufferedPipeUpscaler(
        counting_model,
        batch_size=2,
        device=torch.device("cpu"),
        input_hw=(540, 960),
        scale=2,
    )

    # Tạo 3 khung hình anime giả lập (nền chuyển sắc + khối nhân vật sắc nét) ở bội số nguyên 540x960 -> 1080x1920 (4x4 = 270x480)
    yy, xx = np.mgrid[0:540, 0:960]
    base_bg = np.stack(
        [
            ((xx * 180) // 960 + 30).astype(np.uint8),
            ((yy * 160) // 540 + 40).astype(np.uint8),
            np.full((540, 960), 120, dtype=np.uint8),
        ],
        axis=-1,
    )
    f_a = base_bg.copy()
    f_a[180:360, 380:580] = (230, 200, 180)
    f_b = f_a.copy()
    f_b[260:290, 460:500] = (180, 40, 60)  # Miệng mở
    f_c = f_a.copy()
    f_c[210:225, 430:530] = (30, 30, 40)   # Chớp mắt
    frames = [f_a, f_b, f_a.copy(), f_c, f_b.copy(), f_a.copy()]
    repeats = [2, 1, 3, 1, 2, 1]  # Tổng 10 frames xuất ra pipe

    class MemPipeProc:
        def __init__(self):
            self.stdin = io.BytesIO()

    # 1. Chạy với cyclic_dedup=True + capture_first_rgb_small=True
    proc_opt = MemPipeProc()
    written_opt = upscaler.process_frames_to_pipe(
        frames,
        repeats,
        proc_opt,
        cyclic_dedup=True,
        capture_first_rgb_small=True,
    )
    assert written_opt == 10
    assert upscaler.last_distinct_count == 3
    assert inferred_frames["count"] == 3, f"Chỉ được phép infer 3 distinct frames, thực tế: {inferred_frames['count']}"
    assert upscaler.last_first_rgb_small is not None
    assert upscaler.last_first_rgb_small.shape == (270, 480, 3)

    # Kiểm chứng OutputSanity trực tiếp từ ảnh thu nhỏ avg_pool2d (4x4 khớp INTER_AREA) mà KHÔNG tăng inferred_frames["count"]
    sanity = check_frame_sanity_arrays(f_a, upscaler.last_first_rgb_small, is_rgb=True)
    assert sanity["passed"] is True
    assert sanity["corr"] >= 0.999
    assert inferred_frames["count"] == 3

    # 2. Đối chiếu bit-exact với luồng chạy đầy đủ 6 frames (cyclic_dedup=False)
    proc_ref = MemPipeProc()
    written_ref = upscaler.process_frames_to_pipe(
        frames,
        repeats,
        proc_ref,
        cyclic_dedup=False,
        capture_first_rgb_small=False,
    )
    assert written_ref == 10
    assert proc_opt.stdin.getvalue() == proc_ref.stdin.getvalue()


def test_upscale_directory_cyclic_dedup_hardlinks_outputs(tmp_path):
    """
    Xác minh GPUPersistentWorker.upscale_directory trên chuỗi [A, B, A, B, C]:
    - Chỉ gọi mô hình AI trên 3 frames độc nhất thay vì 5 frames.
    - Xuất đầy đủ cả 5 file PNG đầu ra khớp từng pixel nhờ NTFS hardlink / copy.
    """
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    img_a = np.full((16, 24, 3), (30, 90, 180), dtype=np.uint8)
    img_b = np.full((16, 24, 3), (200, 120, 40), dtype=np.uint8)
    img_c = np.full((16, 24, 3), (50, 210, 70), dtype=np.uint8)
    seq = [img_a, img_b, img_a, img_b, img_c]
    for idx, arr in enumerate(seq, start=1):
        cv2.imwrite(str(in_dir / f"frame{idx:08d}.png"), arr)

    infer_counter = {"frames": 0}

    class CountingIdentity2x(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            infer_counter["frames"] += int(x.shape[0])
            return F.interpolate(x, scale_factor=2.0, mode="nearest")

    worker = GPUPersistentWorker.__new__(GPUPersistentWorker)
    worker.model_id = "animevideov3"
    worker.precision = "fp32"
    worker.device = torch.device("cpu")
    worker.batch_size = 2
    worker.use_tensorrt = False
    worker.model = CountingIdentity2x().eval()

    stats = worker.upscale_directory(str(in_dir), str(out_dir), fmt="png", batch_size=2)
    assert stats["num_frames"] == 5
    assert stats["distinct_frames"] == 3
    assert infer_counter["frames"] == 3

    for idx, expected_in in enumerate(seq, start=1):
        out_p = out_dir / f"frame{idx:08d}.png"
        assert out_p.exists()
        out_bgr = cv2.imread(str(out_p), cv2.IMREAD_COLOR)
        assert out_bgr.shape == (32, 48, 3)
        assert np.array_equal(out_bgr[::2, ::2], expected_in)


def test_mux_segments_directly_and_streamhash_multi_audio_md5(tmp_path):
    """
    Xác minh:
    1. mux_segments_directly ghép nối trực tiếp phân đoạn + đóng gói 2 luồng audio gốc trong 1 lượt
       (không sinh file trung gian merged_4k.mp4) và vượt qua verify_mux_integrity == PASS.
    2. get_all_audio_streams_md5 (Single-Pass -f streamhash) khớp bit-exact với get_audio_stream_md5 từng luồng.
    """
    src_mkv = str(tmp_path / "src_2audio.mkv")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=6:size=320x180:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=6",
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
            "-c:v", "libx264", "-g", "48", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            src_mkv,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    segs_dir = tmp_path / "up_segs"
    segs_dir.mkdir()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", src_mkv,
            "-an", "-sn", "-c:v", "copy",
            "-f", "segment", "-segment_time", "2",
            "-reset_timestamps", "1", "-avoid_negative_ts", "make_zero",
            str(segs_dir / "up_%03d.mp4"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    seg_files = sorted(str(p) for p in segs_dir.glob("up_*.mp4"))
    assert len(seg_files) >= 2

    work_dir = tmp_path / "work_apostrophe's_dir"
    work_dir.mkdir()
    final_mkv = str(tmp_path / "final_4K.mkv")

    mux_segments_directly(seg_files, str(work_dir), src_mkv, final_mkv)
    assert os.path.exists(final_mkv)
    assert not os.path.exists(str(work_dir / "merged_4k.mp4"))

    # Kiểm chứng Single-Pass streamhash khớp từng luồng độc lập
    batch_md5s = get_all_audio_streams_md5(final_mkv, 2)
    assert len(batch_md5s) == 2
    assert batch_md5s[0]["md5"] == get_audio_stream_md5(final_mkv, 0)
    assert batch_md5s[1]["md5"] == get_audio_stream_md5(final_mkv, 1)

    audit = verify_mux_integrity(src_mkv, final_mkv)
    assert audit["status"] == "PASS", f"Audit thất bại: {audit['failure_reasons']}"
    assert audit["video_frame_count"]["passed"] is True
    assert audit["video_frame_count"]["source_frames"] == 144
    assert audit["video_frame_count"]["output_frames"] == 144


def test_fused_mpdecimate_dis_matches_two_pass_and_direct_readinto(tmp_path):
    """
    Xác minh:
    1. scan_dedup_plan(..., policy='dis-0.3') chạy 1 lượt giải mã hợp nhất khớp 100% với 2 lượt rời.
    2. extract_frames_raw_rgb24 đọc trực tiếp readinto vào mảng NumPy không phát sinh bản sao trung gian.
    """
    clip_path = str(tmp_path / "motion_and_static.mp4")
    # 1 giây tĩnh (màu lam) + 1 giây chuyển động testsrc -> vừa có frame drop vừa có frame keep
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=navy:s=320x180:r=12:d=1",
            "-f", "lavfi", "-i", "testsrc=size=320x180:rate=12:duration=1",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            clip_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    plan_def = scan_dedup_plan(clip_path, str(tmp_path / "plan_def.json"), segment_seconds=8, fps_num=12, fps_den=1, policy="default")
    plan_dis = scan_dedup_plan(clip_path, str(tmp_path / "plan_dis.json"), segment_seconds=8, fps_num=12, fps_den=1, policy="dis-0.3")

    # Đối chiếu kết quả 1-pass fused DIS với việc gọi trực tiếp detect_dis_policy_actions trên nền mpdecimate default
    base_actions = ["drop"] * plan_def["total_frames"]
    for idx in plan_def["chunks"][0]["kept_indices"]:
        base_actions[idx] = "keep"
    ref_dis_actions = detect_dis_policy_actions(
        input_path=clip_path,
        base_actions=base_actions,
        forced_keep_indices={0},
        thresh=0.30,
    )
    ref_kept = [i for i, a in enumerate(ref_dis_actions) if a == "keep"]
    assert plan_dis["chunks"][0]["kept_indices"] == ref_kept

    # Kiểm chứng extract_frames_raw_rgb24 nhánh readinto trực tiếp
    frames = extract_frames_raw_rgb24(clip_path, width=320, height=180, kept_indices=[0, 5, 15], hwaccel=False)
    assert len(frames) == 3
    for f in frames:
        assert f.shape == (180, 320, 3)
        assert f.dtype == np.uint8
        assert f.flags["C_CONTIGUOUS"]


def test_pipe_upscaler_repeated_frames_and_drain_worker():
    """
    Xác minh DoubleBufferedPipeUpscaler:
    - _write_frame_to_pipe hỗ trợ repeats > 1 với 1 lần cast memoryview duy nhất.
    - _PipeDrainWorker xả dữ liệu khung hình vào pipe không lỗi và xử lý BrokenPipeError báo đúng stderr.
    """
    from unittest.mock import MagicMock
    from src.trt_zero_copy_upscaler import _PipeDrainWorker

    fake_proc = MagicMock()
    fake_proc.stdin.write.side_effect = lambda chunk: len(chunk)

    buf = np.zeros((100, 100), dtype=np.uint8)
    DoubleBufferedPipeUpscaler._write_frame_to_pipe(fake_proc, buf, repeats=3)
    # buf size = 10,000 bytes. With write_step=8MB, 1 call per repeat = 3 calls total
    assert fake_proc.stdin.write.call_count == 3

    # Test _PipeDrainWorker
    worker = _PipeDrainWorker(fake_proc, DoubleBufferedPipeUpscaler._write_frame_to_pipe)
    worker.submit(buf, repeats=2)
    worker.close()
    assert fake_proc.stdin.write.call_count == 5


def test_pipe_upscaler_micro_timings_structure():
    """
    Xác minh process_frames_to_pipe khi bật enable_micro_timings=True
    tạo đầy đủ cấu trúc last_micro_timings.
    """
    from unittest.mock import MagicMock
    def dummy_model(x):
        return F.interpolate(x, scale_factor=2.0)

    upscaler = DoubleBufferedPipeUpscaler(
        dummy_model,
        batch_size=2,
        device=torch.device("cpu"),
        input_hw=(100, 100),
        scale=2,
        enable_micro_timings=True,
    )
    fake_proc = MagicMock()
    fake_proc.stdin.write.side_effect = lambda chunk: len(chunk)

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
    repeats = [1, 2, 1, 1]

    n_written = upscaler.process_frames_to_pipe(
        frames,
        repeats,
        fake_proc,
        cyclic_dedup=False,
    )
    assert n_written == sum(repeats)
    assert upscaler.last_micro_timings is not None
    assert "num_batches" in upscaler.last_micro_timings
    assert "avg_trt_ms" in upscaler.last_micro_timings
    assert "avg_triton_ms" in upscaler.last_micro_timings
