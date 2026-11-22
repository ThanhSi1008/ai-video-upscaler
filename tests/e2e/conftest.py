# -*- coding: utf-8 -*-
"""
tests/e2e/conftest.py
Shared test fixtures, synthetic media generators, and isolated mock runners
for the opaque-box E2E test suite of ai-video-upscaler.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock, patch

# Preemptively disable dotenv loading and isolate CUDA devices before any module imports
os.environ["SUBTRANS_DISABLE_DOTENV"] = "1"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("UPSCALER_HEVC_ENCODER", "libx265")

import cv2
import numpy as np
import pytest
import torch

import upscale


# -----------------------------------------------------------------------------
# 1. Synthetic Media Generator Helper
# -----------------------------------------------------------------------------

_SYNTHETIC_CACHE_DIR = Path(tempfile.mkdtemp(prefix="e2e_synth_tpl_"))
atexit.register(lambda: shutil.rmtree(_SYNTHETIC_CACHE_DIR, ignore_errors=True))
_SYNTHETIC_CACHE_MAP: dict[tuple, Path] = {}
_DEDUP_PLAN_CACHE: dict[tuple, dict[str, Any]] = {}
_SPLIT_SEGS_CACHE: dict[tuple[str, int], Path] = {}


def create_synthetic_video(
    dest_path: Path | str,
    num_frames: int = 8,
    fps: int = 24,
    width: int = 160,
    height: int = 90,
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
    cache_key: Optional[tuple] = None
    if not attachments:
        cache_key = (
            dest_path.suffix.lower(),
            num_frames,
            fps,
            width,
            height,
            audio_tracks,
            tuple(subtitles or ()),
            open_gop,
            is_mp4,
            chapters,
            video_filter,
            is_hevc,
            keyint,
        )
        cached = _SYNTHETIC_CACHE_MAP.get(cache_key)
        if cached is not None and cached.exists():
            shutil.copy2(cached, dest_path)
            return dest_path
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
        cmd.extend([
            "-vframes", str(num_frames),
            "-vf", "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709",
            "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error",
            "-pix_fmt", "p010le",
            "-color_range", "tv", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709",
            "-tag:v", "hvc1",
        ])
    else:
        cmd.extend(["-vframes", str(num_frames), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"])

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

    if cache_key is not None and dest_path.exists():
        digest = hashlib.sha1(repr(cache_key).encode("utf-8")).hexdigest()[:12]
        tpl_file = _SYNTHETIC_CACHE_DIR / f"tpl_{digest}{dest_path.suffix.lower()}"
        shutil.copy2(dest_path, tpl_file)
        _SYNTHETIC_CACHE_MAP[cache_key] = tpl_file

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
            cv2.imwrite(os.path.join(output_dir, f), im_up, [cv2.IMWRITE_PNG_COMPRESSION, 0])
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


class _DeferredPipeHandle:
    def __init__(self, output_mp4_path: str, fps_str: str, width: int, height: int) -> None:
        self.output_mp4_path = output_mp4_path
        self.fps_str = fps_str
        self.width = int(width)
        self.height = int(height)
        self.stdin = None
        self.stderr = None
        self.returncode = 0

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        return self.returncode

    def kill(self) -> None:
        pass

    def terminate(self) -> None:
        pass


class FastMockPipeUpscaler:
    def __init__(self, model_or_engine, batch_size=8, device=None, input_hw=(90, 160), scale=2):
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
        return _DeferredPipeHandle(output_mp4_path, fps_str, width, height)

    def process_frames_to_pipe(self, unique_frames_rgb, repeat_counts, enc_proc, **kwargs) -> int:
        from src.config import (
            HEVC_PROFILE, HEVC_PIX_FMT, HEVC_TAG,
            HEVC_COLOR_RANGE, HEVC_COLORSPACE, HEVC_COLOR_PRIMARIES, HEVC_COLOR_TRC,
        )

        out_h, out_w = self.output_hw
        if unique_frames_rgb:
            self.last_first_rgb_small = cv2.resize(unique_frames_rgb[0], (480, 270))
            self.last_distinct_count = len(unique_frames_rgb)
        total = int(sum(repeat_counts))
        seg_key = (
            "p010_seg",
            int(enc_proc.width),
            int(enc_proc.height),
            str(enc_proc.fps_str),
            total,
            HEVC_PROFILE,
            HEVC_PIX_FMT,
            HEVC_TAG,
            HEVC_COLOR_RANGE,
            HEVC_COLORSPACE,
            HEVC_COLOR_PRIMARIES,
            HEVC_COLOR_TRC,
        )
        cached_seg = _SYNTHETIC_CACHE_MAP.get(seg_key)
        if cached_seg is None or not cached_seg.exists():
            digest = hashlib.sha1(repr(seg_key).encode("utf-8")).hexdigest()[:12]
            tpl_seg = _SYNTHETIC_CACHE_DIR / f"p010_{digest}.mp4"
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "p010le", "-s", f"{enc_proc.width}x{enc_proc.height}",
                "-color_range", HEVC_COLOR_RANGE, "-colorspace", HEVC_COLORSPACE,
                "-color_primaries", HEVC_COLOR_PRIMARIES, "-color_trc", HEVC_COLOR_TRC,
                "-r", enc_proc.fps_str, "-i", "pipe:0",
                "-c:v", "libx265", "-preset", "ultrafast",
                "-x265-params", "log-level=error:pools=none:frame-threads=1:bframes=0:aq-mode=0:no-sao=1",
                "-profile:v", HEVC_PROFILE, "-pix_fmt", HEVC_PIX_FMT,
                "-crf", "18", "-tag:v", HEVC_TAG,
                str(tpl_seg),
            ]
            frame_bytes = int(enc_proc.width * enc_proc.height * 3)
            dummy_p010 = np.full(frame_bytes, 128, dtype=np.uint8).tobytes()
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            assert proc.stdin is not None
            for _ in range(total):
                proc.stdin.write(dummy_p010)
            proc.stdin.close()
            enc_proc.returncode = int(proc.wait())
            if enc_proc.returncode == 0 and tpl_seg.exists():
                _SYNTHETIC_CACHE_MAP[seg_key] = tpl_seg
                cached_seg = tpl_seg
        if cached_seg is not None and cached_seg.exists():
            os.makedirs(os.path.dirname(os.path.abspath(enc_proc.output_mp4_path)), exist_ok=True)
            shutil.copyfile(cached_seg, enc_proc.output_mp4_path)
        return total


@pytest.fixture(autouse=True)
def mock_pipeline_env(request, monkeypatch, tmp_path):
    """
    Auto-isolated hardware & network mock environment for opaque-box E2E tier tests (test_tier*.py):
    - Stubs GPU exclusivity locks, AI upscalers (2x OpenCV fast mock), and AntiSleep Win32 calls.
    - Routes Zero-Disk P010LE writes through FFmpeg without GPU VRAM and mocks Gemini/Whisper calls.
    - Scoped to test_tier* unless explicitly requested as a fixture so unit-mocked E2E suites
      (test_e2e_*.py) retain their original subprocess.run environment.
    """
    test_file = Path(str(getattr(request, "path", getattr(request.node, "fspath", "")))).name
    if not test_file.startswith("test_tier"):
        yield
        return

    orig_run = subprocess.run
    orig_popen = subprocess.Popen

    def _is_stage_b_subs_cmd(cmd: Any) -> bool:
        return (
            isinstance(cmd, list)
            and any("upscale.py" in str(x) for x in cmd)
            and any("--subs-only" in str(x) for x in cmd)
        )

    class _MockStageBPopen:
        def __init__(self, args: Any) -> None:
            self.args = args
            self.pid = 99999
            self.returncode = 0
            self.stdin = None
            self.stdout = None
            self.stderr = None

        def poll(self) -> int:
            return 0

        def wait(self, timeout: Optional[float] = None) -> int:
            return 0

        def communicate(self, input: Any = None, timeout: Optional[float] = None) -> Tuple[str, str]:
            return ("Mock Stage B Subtitle Success", "")

        def kill(self) -> None:
            pass

        def terminate(self) -> None:
            pass

        def __enter__(self) -> "_MockStageBPopen":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

    def _extract_preflight_worker_args(cmd: Any) -> Optional[List[str]]:
        if not isinstance(cmd, list):
            return None
        for idx, tok in enumerate(cmd):
            if "run_preflight_worker_cli" in str(tok) and idx + 1 < len(cmd):
                return [str(x) for x in cmd[idx + 1 :]]
        return None

    class _InProcessPreflightPopen:
        def __init__(self, args: Any, cli_args: List[str]) -> None:
            import threading
            from src.dedup import run_preflight_worker_cli

            self.args = args
            self.pid = 99998
            self.returncode: Optional[int] = None
            self.stdin = None
            self.stdout = None
            self.stderr = None

            def _run() -> None:
                try:
                    rc = run_preflight_worker_cli(cli_args)
                    self.returncode = int(rc) if isinstance(rc, int) else 0
                except SystemExit as exc:
                    self.returncode = int(exc.code) if isinstance(exc.code, int) else 0
                except Exception:
                    self.returncode = 1

            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()

        def poll(self) -> Optional[int]:
            return None if self._thread.is_alive() else (0 if self.returncode is None else self.returncode)

        def wait(self, timeout: Optional[float] = None) -> int:
            self._thread.join(timeout=timeout)
            return 0 if self.returncode is None else self.returncode

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    def safe_subprocess_run(cmd, *args, **kwargs):
        if _is_stage_b_subs_cmd(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="Mock Stage B Subtitle Success", stderr="")
        if isinstance(cmd, list) and cmd and "nvidia-smi" in str(cmd[0]).lower():
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="offline cpu mode")
        return orig_run(cmd, *args, **kwargs)

    def safe_subprocess_popen(cmd, *args, **kwargs):
        if _is_stage_b_subs_cmd(cmd):
            return _MockStageBPopen(cmd)
        pf_args = _extract_preflight_worker_args(cmd)
        if pf_args is not None:
            return _InProcessPreflightPopen(cmd, pf_args)
        return orig_popen(cmd, *args, **kwargs)

    monkeypatch.setenv("GEMINI_API_KEY", "fake_test_key_e2e_isolated")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("UPSCALER_HEVC_ENCODER", "libx265")
    monkeypatch.setenv("SUBTRANS_DISABLE_DOTENV", "1")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setattr("src.config.DEFAULT_ENGINE", "cuda", raising=False)
    monkeypatch.setattr("src.pipeline.DEFAULT_ENGINE", "cuda", raising=False)
    monkeypatch.setattr("upscale.DEFAULT_ENGINE", "cuda", raising=False)
    monkeypatch.setattr("src.config.HEVC_ENCODER", "libx265", raising=False)
    monkeypatch.setattr("src.dedup.HEVC_ENCODER", "libx265", raising=False)
    monkeypatch.setattr("src.ffmpeg_ops.HEVC_ENCODER", "libx265", raising=False)
    monkeypatch.setattr("src.pipeline.HEVC_ENCODER", "libx265", raising=False)
    monkeypatch.setattr("src.trt_zero_copy_upscaler.HEVC_ENCODER", "libx265", raising=False)
    monkeypatch.setattr(subprocess, "run", safe_subprocess_run)
    monkeypatch.setattr(subprocess, "Popen", safe_subprocess_popen)
    from src import ffmpeg_ops as _fops

    real_extract_raw = _fops.extract_frames_raw_rgb24

    def _fast_pyav_extract_raw(
        seg_path: str,
        width: int,
        height: int,
        kept_indices: Optional[List[int]] = None,
        *extra_args: Any,
        **extra_kwargs: Any,
    ) -> List[Any]:
        src_codec = str(extra_kwargs.get("source_codec") or "")
        if "hevc" in src_codec.lower() or "265" in src_codec.lower():
            return real_extract_raw(seg_path, width, height, kept_indices, *extra_args, **extra_kwargs)
        if kept_indices is not None and len(kept_indices) == 0:
            return []
        try:
            import av

            with av.open(seg_path) as container:
                if not container.streams.video:
                    return real_extract_raw(seg_path, width, height, kept_indices, *extra_args, **extra_kwargs)
                decoded = [np.ascontiguousarray(frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)]
            if not decoded:
                return real_extract_raw(seg_path, width, height, kept_indices, *extra_args, **extra_kwargs)
            if kept_indices is not None:
                return [decoded[i] if i < len(decoded) else decoded[-1] for i in kept_indices]
            return decoded
        except Exception:
            return real_extract_raw(seg_path, width, height, kept_indices, *extra_args, **extra_kwargs)

    from src import dedup as _dedup_mod

    real_scan_dedup = _dedup_mod.scan_dedup_plan

    def _cached_scan_dedup_plan(
        input_path: str,
        plan_path: str,
        segment_seconds: int = 8,
        fps_num: int = 24000,
        fps_den: int = 1001,
        hi: int = _dedup_mod.DEFAULT_DEDUP_HI,
        lo: int = _dedup_mod.DEFAULT_DEDUP_LO,
        frac: float = _dedup_mod.DEFAULT_DEDUP_FRAC,
        chunk_frame_counts: Optional[List[int]] = None,
        policy: str = "default",
        motion_tau: float = 0.3,
        drift_delta: Optional[float] = None,
        drift_min_response: float = 0.10,
        dis_thresh: Optional[float] = None,
    ) -> dict[str, Any]:
        cache_key = None
        try:
            if os.path.isfile(input_path) and os.path.getsize(input_path) <= 512 * 1024:
                with open(input_path, "rb") as vf:
                    vid_sha1 = hashlib.sha1(vf.read()).hexdigest()
                cache_key = (
                    vid_sha1,
                    int(segment_seconds),
                    int(fps_num),
                    int(fps_den),
                    int(hi),
                    int(lo),
                    float(frac),
                    tuple(chunk_frame_counts) if chunk_frame_counts is not None else None,
                    str(policy),
                    float(motion_tau),
                    drift_delta,
                    float(drift_min_response),
                    dis_thresh,
                )
        except OSError:
            cache_key = None

        if cache_key is not None and cache_key in _DEDUP_PLAN_CACHE:
            plan_copy = copy.deepcopy(_DEDUP_PLAN_CACHE[cache_key])
            plan_copy["source_file"] = os.path.basename(input_path)
            os.makedirs(os.path.dirname(os.path.abspath(plan_path)), exist_ok=True)
            with open(plan_path, "w", encoding="utf-8") as pf:
                json.dump(plan_copy, pf, ensure_ascii=False)
            return plan_copy

        res_plan = real_scan_dedup(
            input_path=input_path,
            plan_path=plan_path,
            segment_seconds=segment_seconds,
            fps_num=fps_num,
            fps_den=fps_den,
            hi=hi,
            lo=lo,
            frac=frac,
            chunk_frame_counts=chunk_frame_counts,
            policy=policy,
            motion_tau=motion_tau,
            drift_delta=drift_delta,
            drift_min_response=drift_min_response,
            dis_thresh=dis_thresh,
        )
        if cache_key is not None and isinstance(res_plan, dict):
            _DEDUP_PLAN_CACHE[cache_key] = copy.deepcopy(res_plan)
        return res_plan

    real_split_segs = _fops.split_video_segments

    def _cached_split_video_segments(input_path: str, segs_dir: str, segment_time: int) -> List[str]:
        cache_key = None
        try:
            if os.path.isfile(input_path) and os.path.getsize(input_path) <= 512 * 1024:
                with open(input_path, "rb") as vf:
                    vid_sha1 = hashlib.sha1(vf.read()).hexdigest()
                cache_key = (vid_sha1, int(segment_time))
        except OSError:
            cache_key = None

        cached_dir = _SPLIT_SEGS_CACHE.get(cache_key) if cache_key is not None else None
        if cached_dir is not None and cached_dir.is_dir():
            os.makedirs(segs_dir, exist_ok=True)
            for fname in os.listdir(segs_dir):
                if (fname.startswith("seg_") and fname.endswith(".mp4")) or fname.startswith(".split_complete"):
                    try:
                        os.remove(os.path.join(segs_dir, fname))
                    except OSError:
                        pass
            out_segs: List[str] = []
            for src_seg in sorted(cached_dir.glob("seg_*.mp4")):
                dst_seg = os.path.join(segs_dir, src_seg.name)
                shutil.copyfile(src_seg, dst_seg)
                out_segs.append(dst_seg)
            if out_segs:
                with open(os.path.join(segs_dir, ".split_complete"), "w", encoding="utf-8") as mf:
                    json.dump(
                        {
                            "segment_count": len(out_segs),
                            "segment_time": int(segment_time),
                            "source_file": os.path.basename(input_path),
                        },
                        mf,
                        ensure_ascii=False,
                    )
                return out_segs

        segs = real_split_segs(input_path, segs_dir, segment_time)
        if cache_key is not None and segs:
            tpl_dir = _SYNTHETIC_CACHE_DIR / f"segs_{cache_key[0][:12]}_{cache_key[1]}"
            tpl_dir.mkdir(parents=True, exist_ok=True)
            for sp in segs:
                shutil.copyfile(sp, tpl_dir / os.path.basename(sp))
            _SPLIT_SEGS_CACHE[cache_key] = tpl_dir
        return segs

    monkeypatch.setattr("src.dedup.scan_dedup_plan", _cached_scan_dedup_plan, raising=False)
    monkeypatch.setattr("src.scheduler.scan_dedup_plan", _cached_scan_dedup_plan, raising=False)
    monkeypatch.setattr("src.ffmpeg_ops.split_video_segments", _cached_split_video_segments, raising=False)
    monkeypatch.setattr("src.pipeline.split_video_segments", _cached_split_video_segments, raising=False)
    monkeypatch.setattr("src.ffmpeg_ops.extract_frames_raw_rgb24", _fast_pyav_extract_raw, raising=False)
    monkeypatch.setattr("src.pipeline.extract_frames_raw_rgb24", _fast_pyav_extract_raw, raising=False)
    monkeypatch.setattr("gc.collect", lambda *a, **kw: 0)
    monkeypatch.setattr("src.pipeline.get_thermal_state", lambda: "Nominal", raising=False)
    monkeypatch.setattr(
        "src.pipeline.get_swap_usage",
        lambda: "RAM: 50% used (8192MB/16384MB) | Pagefile: 10% used (1024MB/10240MB)",
        raising=False,
    )

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
         patch("os.startfile", create=True, return_value=None):
        yield


@pytest.fixture(scope="session")
def _synthetic_clip_templates(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("e2e_clip_templates")
    return {
        "10s": create_synthetic_video(base / "tpl_10s.mkv", num_frames=4, fps=24, width=320, height=180, audio_tracks=1),
        "30s": create_synthetic_video(base / "tpl_30s.mkv", num_frames=8, fps=24, width=320, height=180, audio_tracks=1),
    }


@pytest.fixture
def synthetic_10s_mkv(tmp_path: Path, _synthetic_clip_templates: dict[str, Path]) -> Path:
    """Copy reusable 4-frame synthetic MKV into test-local tmp_path."""
    p = tmp_path / "sample_10s.mkv"
    shutil.copy2(_synthetic_clip_templates["10s"], p)
    return p


@pytest.fixture
def synthetic_30s_mkv(tmp_path: Path, _synthetic_clip_templates: dict[str, Path]) -> Path:
    """Copy reusable 8-frame synthetic MKV into test-local tmp_path."""
    p = tmp_path / "sample_30s.mkv"
    shutil.copy2(_synthetic_clip_templates["30s"], p)
    return p
