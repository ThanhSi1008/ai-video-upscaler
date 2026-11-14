# -*- coding: utf-8 -*-
"""
tests/e2e/conftest.py
Shared test fixtures, synthetic media generators, and isolated mock runners
for the opaque-box E2E test suite of ai-video-upscaler.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock, patch

# Preemptively disable dotenv loading and isolate CUDA devices before any module imports
os.environ["SUBTRANS_DISABLE_DOTENV"] = "1"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np
import pytest
import torch

import upscale


# -----------------------------------------------------------------------------
# 1. Synthetic Media Generator Helper
# -----------------------------------------------------------------------------

def create_synthetic_video(
    dest_path: Path | str,
    num_frames: int = 8,
    fps: int = 24,
    width: int = 320,
    height: int = 180,
    audio_tracks: int = 1,
    subtitles: Optional[List[str]] = None,
    open_gop: bool = False,
    is_mp4: bool = False,
    attachments: Optional[List[Path | str]] = None,
    chapters: int = 0,
    video_filter: Optional[str] = None,
    is_hevc: bool = False,
    keyint: Optional[int] = None,
) -> Path:
    """
    Tạo video mẫu tổng hợp siêu nhẹ (4-12 frames, ~6-25 KB) bằng FFmpeg lavfi.
    Thời gian thực thi: ~0.02 - 0.05s. Chạy hoàn toàn offline, 0 bytes tải mạng.
    """
    dest_path = Path(dest_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, num_frames / fps)

    v_filter = video_filter or f"testsrc=size={width}x{height}:rate={fps}"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", v_filter,
    ]

    # Thêm audio streams (sine wave 440Hz, 880Hz, ...)
    for i in range(audio_tracks):
        cmd.extend(["-f", "lavfi", "-i", f"sine=frequency={440 * (i + 1)}:duration={duration}"])

    # Thêm subtitle streams nếu có
    sub_files: List[Path] = []
    if subtitles:
        for idx, sub_text in enumerate(subtitles):
            sf = dest_path.parent / f"_tmp_sub_{idx}_{os.getpid()}.srt"
            sf.write_text(f"1\n00:00:00,000 --> 00:00:01,000\n{sub_text}\n", encoding="utf-8")
            sub_files.append(sf)
            cmd.extend(["-i", str(sf)])

    # Thêm chapters file nếu có
    chap_file: Optional[Path] = None
    if chapters > 0:
        chap_file = dest_path.parent / f"_tmp_chap_{os.getpid()}.txt"
        chap_lines = [";FFMETADATA1\n"]
        chap_dur_ms = int((duration * 1000) / chapters) if chapters else 1000
        for c in range(chapters):
            start_ms = c * chap_dur_ms
            end_ms = (c + 1) * chap_dur_ms
            chap_lines.append(
                f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_ms}\nEND={end_ms}\ntitle=Chapter {c + 1}\n\n"
            )
        chap_file.write_text("".join(chap_lines), encoding="utf-8")
        cmd.extend(["-i", str(chap_file)])

    # Map video stream
    cmd.extend(["-map", "0:v:0"])

    # Map audio streams
    for i in range(audio_tracks):
        cmd.extend(["-map", f"{i + 1}:a:0"])

    # Map subtitle streams
    if subtitles:
        base_sub_idx = 1 + audio_tracks
        for i in range(len(subtitles)):
            cmd.extend(["-map", f"{base_sub_idx + i}:s:0"])

    # Metadata map cho chapters
    if chapters > 0 and chap_file:
        chap_input_idx = 1 + audio_tracks + len(subtitles or [])
        cmd.extend(["-map_metadata", str(chap_input_idx)])

    if is_hevc:
        cmd.extend(["-vframes", str(num_frames), "-c:v", "hevc_nvenc", "-pix_fmt", "p010le", "-tag:v", "hvc1"])
    else:
        cmd.extend(["-vframes", str(num_frames), "-c:v", "libx264", "-pix_fmt", "yuv420p"])

    if keyint:
        cmd.extend(["-g", str(keyint)])

    if open_gop:
        cmd.extend(["-open_gop", "1", "-g", str(max(2, num_frames // 2)), "-bf", "2"])

    if audio_tracks > 0:
        cmd.extend(["-c:a", "aac"])

    if subtitles:
        cmd.extend(["-c:s", "srt" if not str(dest_path).endswith(".mkv") else "ass"])

    # Đính kèm font attachments (cho container MKV)
    if attachments:
        for att in attachments:
            cmd.extend([
                "-attach", str(att),
                "-metadata:s:t", "mimetype=application/x-truetype-font",
            ])

    cmd.append(str(dest_path))
    try:
        subprocess.run(cmd, check=True)
    finally:
        # Dọn dẹp file phụ trợ tạm
        for sf in sub_files:
            try:
                sf.unlink(missing_ok=True)
            except OSError:
                pass
        if chap_file:
            try:
                chap_file.unlink(missing_ok=True)
            except OSError:
                pass

    return dest_path


# -----------------------------------------------------------------------------
# 2. Dual CLI Runner Helpers
# -----------------------------------------------------------------------------

def run_upscale_cli(args: List[str], monkeypatch, capsys) -> Tuple[int, str, str]:
    """
    Thực thi trực tiếp upscale.main() trong tiến trình (in-process).
    Thời gian thực thi: ~5-15ms. Bắt chính xác exit code qua SystemExit.
    """
    # Nếu -o/--output trỏ vào thư mục (không có đuôi video), tự động tạo thư mục để isdir() nhận diện đúng
    for flag in ("-o", "--output"):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                p = Path(args[idx + 1])
                if p.suffix.lower() not in (".mkv", ".mp4"):
                    p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(sys, "argv", ["upscale.py", *args])
    exit_code = 0
    try:
        upscale.main()
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    out, err = capsys.readouterr()
    return exit_code, out, err


def run_upscale_subprocess(args: List[str], env_overrides: Optional[dict] = None) -> subprocess.CompletedProcess:
    """
    Thực thi lệnh CLI thông qua tiến trình con ngoài (opaque-box subprocess).
    Dành riêng cho các kịch bản Tier 4 cần kiểm tra vòng đời tiến trình Windows.
    """
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "-1",
        "RUN_GPU_TESTS": "0",
        "BENCHMARK_MODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "SUBTRANS_DISABLE_DOTENV": "1",
        "PYTHON_DOTENV_DISABLED": "1",
    })
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "upscale.py", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


# -----------------------------------------------------------------------------
# 3. Fast Mock Pipeline Environment Fixture & Classes
# -----------------------------------------------------------------------------

class FastMockWorker:
    def __init__(self, *args, **kwargs):
        self.model_id = kwargs.get("model_id", "animevideov3")
        self.chunks_processed = 0
        self.total_restarts = 0
        self.device = torch.device("cpu")
        self.batch_size = kwargs.get("batch_size", 8) or 8
        self.precision = kwargs.get("precision", "fp16")
        self.model = torch.nn.Identity()
        self._trt_runners = {}

    def upscale_directory(self, input_dir: str, output_dir: str, fmt: str = "png", **kwargs) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        files = sorted([f for f in os.listdir(input_dir) if f.endswith(f".{fmt}")])
        for f in files:
            im = cv2.imread(os.path.join(input_dir, f))
            if im is None:
                im = np.zeros((180, 320, 3), dtype=np.uint8)
            im_up = cv2.resize(im, (im.shape[1] * 2, im.shape[0] * 2))
            cv2.imwrite(os.path.join(output_dir, f), im_up)
        return {
            "elapsed_seconds": 0.05,
            "fps": 100.0,
            "steady_fps": 100.0,
            "num_frames": len(files),
            "distinct_frames": len(files),
            "driver_alloc_peak_gib": 0.0,
            "reserved_peak_gib": 0.0,
            "peak_rss_mb": 50.0,
        }

    def restart_if_needed(self, **kwargs):
        pass


class FastMockPipeUpscaler:
    def __init__(self, model_or_engine, batch_size=8, device=None, input_hw=(180, 320), scale=2):
        self.batch_size = batch_size
        self.model = model_or_engine
        self.scale = scale
        self.input_hw = input_hw
        self.output_hw = (input_hw[0] * scale, input_hw[1] * scale)
        self.last_first_rgb_small = None
        self.last_distinct_count = 1

    def ensure_resolution(self, in_h: int, in_w: int):
        self.input_hw = (in_h, in_w)
        self.output_hw = (in_h * self.scale, in_w * self.scale)
        return self.output_hw

    def spawn_nvenc_pipe_writer(self, output_mp4_path: str, fps_str: str = "24/1", width: int = 640, height: int = 360):
        from src.config import (
            HEVC_ENCODER, HEVC_PROFILE, HEVC_PIX_FMT, HEVC_TAG,
            HEVC_COLOR_RANGE, HEVC_COLORSPACE, HEVC_COLOR_PRIMARIES, HEVC_COLOR_TRC,
        )
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "p010le", "-s", f"{width}x{height}",
            "-color_range", HEVC_COLOR_RANGE, "-colorspace", HEVC_COLORSPACE,
            "-color_primaries", HEVC_COLOR_PRIMARIES, "-color_trc", HEVC_COLOR_TRC,
            "-r", fps_str, "-i", "pipe:0",
            "-c:v", HEVC_ENCODER, "-profile:v", HEVC_PROFILE, "-pix_fmt", HEVC_PIX_FMT,
            "-preset", "p6", "-cq", "18", "-tag:v", HEVC_TAG,
            output_mp4_path,
        ]
        sub_env = os.environ.copy()
        sub_env.pop("CUDA_VISIBLE_DEVICES", None)
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=sub_env)

    def process_frames_to_pipe(self, unique_frames_rgb, repeat_counts, enc_proc, **kwargs) -> int:
        out_h, out_w = self.output_hw
        if unique_frames_rgb:
            self.last_first_rgb_small = cv2.resize(unique_frames_rgb[0], (480, 270))
        frame_bytes = int(out_w * out_h * 3)  # p010le 4:2:0 format: 3 bytes/pixel
        dummy_p010 = np.full(frame_bytes, 128, dtype=np.uint8).tobytes()
        total = 0
        for count in repeat_counts:
            for _ in range(count):
                enc_proc.stdin.write(dummy_p010)
                total += 1
        enc_proc.stdin.close()
        enc_proc.wait()
        return total


@pytest.fixture(autouse=True)
def mock_pipeline_env(monkeypatch, tmp_path):
    """
    Fixture tự động cô lập phần cứng và mạng cho toàn bộ các bài test E2E:
    - Triệt tiêu file khóa GPU (.gpu_active.lock)
    - Thay thế AI upscaler bằng bộ phóng to 2x siêu tốc OpenCV (0.02s)
    - Tự động điều phối Zero-Disk P010LE writer qua FFmpeg mà không tốn GPU VRAM
    - Tắt AntiSleep Win32 native calls
    - Mock Gemini API và chặn gọi whisper model download ngoài luồng
    """
    orig_run = subprocess.run
    def safe_subprocess_run(cmd, *args, **kwargs):
        # Chặn Stage B spawn recursive whisper subprocesses ngoài tầm kiểm soát
        if isinstance(cmd, list):
            if any("upscale.py" in str(x) for x in cmd) and any("--subs-only" in str(x) for x in cmd):
                return subprocess.CompletedProcess(cmd, 0, stdout="Mock Stage B Subtitle Success", stderr="")
            # Cho phép FFmpeg sử dụng phần cứng NVENC mà không ảnh hưởng tới cô lập PyTorch CPU
            if len(cmd) > 0 and str(cmd[0]).lower() in ("ffmpeg", "ffprobe"):
                sub_env = kwargs.get("env")
                if sub_env is None:
                    sub_env = os.environ.copy()
                else:
                    sub_env = sub_env.copy()
                sub_env.pop("CUDA_VISIBLE_DEVICES", None)
                kwargs["env"] = sub_env

        return orig_run(cmd, *args, **kwargs)

    monkeypatch.setenv("GEMINI_API_KEY", "fake_test_key_e2e_isolated")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("SUBTRANS_DISABLE_DOTENV", "1")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    with patch("src.cuda_engine.GPUPersistentWorker", FastMockWorker), \
         patch("src.cuda_engine.get_cuda_inference_engine", return_value=FastMockWorker()), \
         patch("src.build_trt_engine.get_hybrid_engine_or_wrapper", return_value=torch.nn.Identity()), \
         patch("src.trt_zero_copy_upscaler.DoubleBufferedPipeUpscaler", FastMockPipeUpscaler), \
         patch("src.pipeline.acquire_gpu_lock"), \
         patch("src.pipeline.release_gpu_lock"), \
         patch("src.config.acquire_gpu_lock"), \
         patch("src.config.release_gpu_lock"), \
         patch("src.preflight.AntiSleepContext"), \
         patch("google.genai.Client"), \
         patch("subprocess.run", side_effect=safe_subprocess_run), \
         patch("os.startfile", create=True, return_value=None):
        yield


# -----------------------------------------------------------------------------
# 4. Standard Reusable Media Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def synthetic_10s_mkv(tmp_path: Path) -> Path:
    """Tạo video mẫu 10s (4 frames, 320x180, AAC stereo)."""
    p = tmp_path / "sample_10s.mkv"
    return create_synthetic_video(p, num_frames=4, fps=24, width=320, height=180, audio_tracks=1)


@pytest.fixture
def synthetic_30s_mkv(tmp_path: Path) -> Path:
    """Tạo video mẫu 30s (8 frames, 320x180, AAC stereo)."""
    p = tmp_path / "sample_30s.mkv"
    return create_synthetic_video(p, num_frames=8, fps=24, width=320, height=180, audio_tracks=1)
