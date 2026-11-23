---
name: video-upscale-ops
description: >-
  Operate, debug, and extend the local 4K Anime Video Upscale pipeline (upscale.py, src/).
  Activate when configuring SR models (models.yaml, shows.yaml), tuning GOP/DIS deduplication,
  inspecting .temp_* checkpoints, running previews/overnight queues, or auditing BT.709 / 10-bit NVENC / MKV mux integrity.
---

# 4K Anime Video Upscale Pipeline Runbook

## 1. Non-Negotiable Hardware & Quality Invariants

- **100% Local Execution & Autonomous Hardware Authority:** Windows 11, multi-core Intel Core i5-12600K (P-Core HighQoS, AVX2 `cv2.imwrite`), 32GB RAM / NVMe Gen4, NVIDIA RTX 3060 12GB (TensorRT 10 FP16 `src/build_trt_engine.py` / PyTorch `cuda` FP16, 10-bit `hevc_nvenc`, `batch_size=8`, `--zero-disk` default `True` with `--no-zero-disk` fallback). Full autonomous authority is granted across 100% of local hardware without asking for approval on GPU frame counts, TensorRT engine builds, full-episode upscales (pass `--no-subs` unless Cloud Gemini subtrans is approved), or hardware benchmarks (`scripts/benchmark_pipeline.py`, `scripts/silicon_roofline_probe.py`, `RUN_GPU_TESTS=1`) whenever `.gpu_active.lock` is idle. Never propose Cloud GPUs, load local LLMs concurrently, or persistently alter repo default `animevideov3` / NVENC `CQ 18` configs without user confirmation.
- **Checkpoint Safety:** Never delete or modify `.temp_<basename>/` (`manifest.json`, `plan.json`, `raw_segs/`, `upscaled_segs/`) without explicit user confirmation.
- **Two-Way BT.709 Matrix:**
  - Extract: `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc`
  - Encode: `scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv` (`p010le`, `hevc_nvenc`, `main10`, `p6`, `cq=18`, `multipass=fullres`, `rc-lookahead=32`, `spatial-aq=1`, `aq-strength=6`, `temporal-aq=1`, `-tag:v hvc1`).
- **Bit-Exact Stream Copy Mux:**
  - Always include: `-map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 -c copy -max_interleave_delta 0`

## 2. Diagnostic & Safe Inspection Commands

```powershell
# Inspect registered SR models without GPU inference:
.\.venv\Scripts\python.exe upscale.py --list-models

# Inspect AST context & bounded slices in video modules before reading large files:
.\.venv\Scripts\python.exe scripts/agent_context.py --pack src/pipeline.py:worker_extract src/dedup.py:get_dedup_params
.\.venv\Scripts\python.exe scripts/agent_context.py --symbols src/pipeline.py src/dedup.py
.\.venv\Scripts\python.exe scripts/agent_context.py --slice src/trt_zero_copy_upscaler.py DoubleBufferedPipeUpscaler.release

# Run < 5s CPU-safe 24f pipeline smoke check:
.\.venv\Scripts\python.exe scripts/smoke_test_pipeline.py

# Autonomous TensorRT FP16 build, hardware benchmarks, or full-episode local upscale:
.\.venv\Scripts\python.exe src/build_trt_engine.py
.\.venv\Scripts\python.exe scripts/silicon_roofline_probe.py
.\.venv\Scripts\python.exe upscale.py "path/to/episode.mkv" -o out/ --no-subs

# Probe source container streams & frame count (read-only):
ffprobe -v error -show_entries stream=index,codec_name,codec_type,channels,pix_fmt,r_frame_rate -of json "path/to/source.mkv"
```

## 3. Deduplication Policy Selection (`--dedup-policy`)

- `default` (`mpdecimate` `768:320:0.33`): Standard ~55–61% frame reduction on TV/WEB-DL anime.
- `dis` (`--dis-thresh 0.30`): Adds OpenCV DIS optical flow to preserve slow camera pans/zooms on BD Remuxes or cinematic episodes.
- `sensitive-b` (`384:128:0.10`): Pure FFmpeg low-overhead option for subtle motion without OpenCV flow pass.
- `off`: 100% frames inferred (zero frame drop).

## 4. Production Output Verification Checklist

Whenever real video output is produced:
1. `src/output_sanity.py` passes ($|\Delta Y| \le 6/255$, Pearson $r \ge 0.98$).
2. `src/mux_audit.py:verify_mux_integrity` returns `"PASS"` (matching audio MD5/streams, subtitle tracks, font attachments).
3. `ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames` matches source and 4K output bit-for-bit.
