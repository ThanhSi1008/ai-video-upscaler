#!/usr/bin/env python3
"""Fast (< 5s) CPU-safe end-to-end smoke test for e:\\ai-video-upscaler.

Verifies without GPU/TensorRT/network:
1. Synthetic 1s (24 frames, 320x180) video + audio generation with duplicate frames.
2. Two-way BT.709 color conversion (tv -> pc RGB24 extraction & pc -> tv 10-bit encode).
3. `src.dedup.deduplicate_cyclic_frames` duplicate reduction & `src.output_sanity.check_frame_sanity_arrays`.
4. `src.ffmpeg_ops.mux_segments_directly`, `mux_final_container`, `embed_subtitle_track(..., is_default=True)`,
   and `src.mux_audit.verify_mux_integrity` (`"PASS"`).
5. `ffprobe -count_frames` confirming exact 24 frames, 10-bit pixel format, and default subtitle disposition.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dedup import deduplicate_cyclic_frames  # noqa: E402
from src.env_isolation import configure_e_drive_isolation  # noqa: E402
from src.ffmpeg_ops import (  # noqa: E402
    embed_subtitle_track,
    extract_frames_raw_yuv420,
    mux_final_container,
    mux_segments_directly,
)
from src.mux_audit import verify_mux_integrity  # noqa: E402
from src.output_sanity import check_frame_sanity_arrays  # noqa: E402
from src.trt_zero_copy_upscaler import GPUColorConverterP010LE, _yuv420_to_nchw_fp16_bt709  # noqa: E402


def _resolve_ffmpeg_tools() -> tuple[str, str]:
    configure_e_drive_isolation(project_root=ROOT_DIR, create_dirs=False)
    bin_dir = ROOT_DIR / "bin" / "ffmpeg"
    if bin_dir.is_dir():
        cur_path = os.environ.get("PATH", "")
        if str(bin_dir) not in cur_path.split(os.pathsep):
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{cur_path}"
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    if not ffmpeg_bin or not ffprobe_bin:
        raise RuntimeError("Neither bin/ffmpeg nor system PATH provides ffmpeg and ffprobe.")
    return ffmpeg_bin, ffprobe_bin


def run_smoke_test() -> dict[str, Any]:
    """Execute the CPU-safe synthetic 24-frame smoke pipeline and return structured metrics."""
    t0 = time.perf_counter()
    ffmpeg_bin, ffprobe_bin = _resolve_ffmpeg_tools()

    src_w, src_h, target_frames = 320, 180, 24
    up_w, up_h = 640, 360

    with tempfile.TemporaryDirectory(prefix="smoke_pipeline_") as tmp_dir_str:
        work = Path(tmp_dir_str)
        src_mkv = work / "smoke_src.mkv"
        up_vid = work / "smoke_up_video.mkv"
        out_mkv = work / "smoke_out_4K.mkv"
        srt_file = work / "smoke.vi.srt"

        # 1. Synthesize 1s (24 frames @ 24fps from 6fps source -> deliberate 4x holds + motion) + 1s AAC audio
        gen_cmd = [
            ffmpeg_bin, "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc2=size={src_w}x{src_h}:rate=6:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
            "-vf", "fps=24,format=yuv420p",
            "-frames:v", str(target_frames),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "aac", "-b:a", "96k",
            "-shortest",
            str(src_mkv),
        ]
        subprocess.run(gen_cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")

        # 2. Extract 24 frames via BT.709 Limited -> RGB24 Full
        ext_cmd = [
            ffmpeg_bin, "-v", "error",
            "-i", str(src_mkv),
            "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        ext_res = subprocess.run(ext_cmd, check=True, capture_output=True)
        frame_bytes = src_w * src_h * 3
        raw_buf = ext_res.stdout
        extracted_count = len(raw_buf) // frame_bytes
        if extracted_count != target_frames:
            raise AssertionError(f"Expected {target_frames} extracted frames, got {extracted_count}")

        frames_rgb = [
            np.frombuffer(raw_buf[i * frame_bytes : (i + 1) * frame_bytes], dtype=np.uint8)
            .reshape((src_h, src_w, 3))
            .copy()
            for i in range(extracted_count)
        ]

        # 3. Exercise deduplication + stubbed 2x upscale + Output Sanity check
        unique_frames, frame_map = deduplicate_cyclic_frames(frames_rgb)
        dedup_saved = len(frames_rgb) - len(unique_frames)
        if dedup_saved <= 0:
            raise AssertionError(f"Expected duplicate compression on 12fps->24fps video, got dedup_saved={dedup_saved}")

        upscaled_unique = [
            cv2.resize(f, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
            for f in unique_frames
        ]
        upscaled_all = [upscaled_unique[idx] for idx in frame_map]
        if len(upscaled_all) != target_frames:
            raise AssertionError(f"Expected {target_frames} reconstructed frames, got {len(upscaled_all)}")

        sanity = check_frame_sanity_arrays(frames_rgb[0], upscaled_all[0], is_rgb=True)
        if not sanity.get("passed"):
            raise AssertionError(f"Output sanity check failed: {sanity}")

        # 4. Encode 24 upscaled frames as two 12-frame 10-bit HEVC segments to exercise mux_segments_directly
        segs_dir = work / "up_segs"
        segs_dir.mkdir(parents=True, exist_ok=True)
        seg_files: list[str] = []
        half = target_frames // 2
        for seg_idx, seg_frames in enumerate((upscaled_all[:half], upscaled_all[half:])):
            seg_path = segs_dir / f"up_{seg_idx:04d}.mp4"
            enc_cmd = [
                ffmpeg_bin, "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{up_w}x{up_h}", "-r", "24", "-i", "pipe:0",
                "-vf", (
                    "scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv,"
                    "format=yuv420p10le,"
                    "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709"
                ),
                "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error:open-gop=0:keyint=12:min-keyint=12",
                "-pix_fmt", "yuv420p10le",
                "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-tag:v", "hvc1",
                str(seg_path),
            ]
            enc_input = b"".join(np.ascontiguousarray(f).tobytes() for f in seg_frames)
            subprocess.run(enc_cmd, input=enc_input, check=True, capture_output=True)
            seg_files.append(str(seg_path))

        # 5. Direct multi-segment concat + stream-copy mux + Vietnamese softsub embed + mux audit
        mux_segments_directly(seg_files, str(work), str(src_mkv), str(out_mkv))
        mux_final_container(seg_files[0], str(src_mkv), str(up_vid))
        srt_file.write_text(
            "1\n00:00:00,050 --> 00:00:00,900\nKiểm tra phụ đề tiếng Việt.\n",
            encoding="utf-8",
        )
        embedded_ok = embed_subtitle_track(
            str(out_mkv),
            str(srt_file),
            original_source_path=str(src_mkv),
            is_default=True,
        )
        if not embedded_ok:
            raise AssertionError("embed_subtitle_track failed during smoke pipeline run")

        audit = verify_mux_integrity(str(src_mkv), str(out_mkv), allow_added_vie_sub=True)
        if audit.get("status") != "PASS":
            raise AssertionError(f"verify_mux_integrity failed: {audit.get('failure_reasons')}")

        # 6. Probe exact frame count, 10-bit pixel format, and default subtitle disposition
        probe_cmd = [
            ffprobe_bin, "-v", "error",
            "-count_frames",
            "-show_streams",
            "-of", "json",
            str(out_mkv),
        ]
        probe_res = subprocess.run(probe_cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        probe_data = json.loads(probe_res.stdout)
        streams = probe_data.get("streams", [])
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        s_stream = next((s for s in streams if s.get("codec_type") == "subtitle"), None)
        if not v_stream:
            raise AssertionError("No video stream in final smoke MKV")

        nb_frames = int(v_stream.get("nb_read_frames") or v_stream.get("nb_frames") or 0)
        pix_fmt = str(v_stream.get("pix_fmt") or "")
        sub_default = bool(s_stream and (s_stream.get("disposition") or {}).get("default") == 1)

        if nb_frames != target_frames:
            raise AssertionError(f"Expected nb_read_frames={target_frames}, got {nb_frames}")
        if pix_fmt not in {"yuv420p10le", "p010le"}:
            raise AssertionError(f"Expected 10-bit pixel format (yuv420p10le/p010le), got '{pix_fmt}'")
        if not sub_default:
            raise AssertionError(f"Expected default subtitle stream disposition=1, got {s_stream}")

        # 7. Lightweight CPU 4-frame extract_frames_raw_yuv420 -> _yuv420_to_nchw_fp16_bt709 -> GPUColorConverterP010LE check
        import torch

        yuv_frames = extract_frames_raw_yuv420(
            seg_files[0],
            width=up_w,
            height=up_h,
            kept_indices=[0, 1, 2, 3],
            hwaccel=False,
            total_frames=half,
            source_codec="hevc",
            source_pix_fmt="yuv420p10le",
        )
        if len(yuv_frames) != 4:
            raise AssertionError(f"Expected 4 raw P010LE frames, got {len(yuv_frames)}")
        yuv_t = torch.from_numpy(np.stack([f.view(np.int16) for f in yuv_frames], axis=0))
        rgb_t = _yuv420_to_nchw_fp16_bt709(
            yuv_t,
            is_p010=True,
            chroma_siting="left",
            out_dtype=torch.float32,
            use_triton=False,
        )
        if not torch.isfinite(rgb_t).all():
            raise AssertionError("Non-finite values produced by _yuv420_to_nchw_fp16_bt709")
        p010_out = GPUColorConverterP010LE(device=torch.device("cpu"), chroma_siting="left", dither=True).rgb_to_p010le(rgb_t)
        if p010_out.shape != (4, up_h * 3 // 2, up_w) or p010_out.dtype != torch.int16:
            raise AssertionError(f"Unexpected P010LE roundtrip output: {p010_out.shape}, {p010_out.dtype}")

    elapsed_s = round(time.perf_counter() - t0, 2)
    return {
        "status": "PASS",
        "frames": nb_frames,
        "target_frames": target_frames,
        "dedup_saved": dedup_saved,
        "pix_fmt": pix_fmt,
        "sub_default": sub_default,
        "mux_audit": audit.get("status"),
        "elapsed_s": elapsed_s,
    }


def main() -> int:
    try:
        res = run_smoke_test()
    except Exception as exc:
        print(f"FAIL: smoke_test_pipeline ({exc})")
        return 1
    print(
        f"PASS: smoke_test_pipeline (frames={res['frames']}/{res['target_frames']}, "
        f"dedup_saved={res['dedup_saved']}, pix_fmt={res['pix_fmt']}, "
        f"sub_default={1 if res['sub_default'] else 0}, mux_audit={res['mux_audit']}, "
        f"elapsed={res['elapsed_s']:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
