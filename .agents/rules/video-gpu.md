---
trigger: glob
globs: "upscale.py, shows.yaml, src/**/*.py, scripts/*upscale*.py, scripts/smoke_test_pipeline.py, models/*.yaml"
description: "10-bit p010le end-to-end invariant, RTX 3060 12GB VRAM/RAMDisk bounds, zero-VRAM web boundary, E-drive temp isolation, smoke_test_pipeline check, and GPU execution guards."
---

# Video & GPU Pipeline Scoped Rules

- **End-to-End 10-bit `p010le` & Two-Way BT.709 Invariant:**
  - Zero-Disk (`--zero-disk`): decode semi-planar `p010le` (10-bit) / `nv12` (8-bit) -> `@triton.jit` `_yuv420_to_nchw_fp16_bt709_kernel` (`left` chroma) -> SR -> `_rgb_to_p010le_triton_kernel` (`left` `[1,2,1]/4 x [1,1]/2` chroma + spatial IGN dither). Keep `GPUColorConverterP010LE()` default `(chroma_siting="center", dither=False)` for legacy tests.
  - Disk/RGB24 fallback: `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc:flags=bicubic+accurate_rnd+full_chroma_int` and inverse `pc -> tv` (`p010le`).
  - Metadata tags: `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`.
- **10-bit NVENC Studio Preset & Bit-Exact Stream-Copy Mux:**
  - `hevc_nvenc`, `main10`, `p010le`, preset `p6`, `CQ 18` (`maxrate 45M`), `--nvenc-multipass auto|fullres|qres` (default `auto`, locked **per-episode** so `-f concat -c copy` never mixes SPS/PPS `extradata`), `rc-lookahead 32`, `spatial-aq 1`, `aq-strength 6`, `temporal-aq 1`. In-process PyAV `hevc_nvenc` MUST set `stream.bit_rate = 0` and never mutate `AVFrame.pts` in-place during lookahead.
  - Mux original audio, softsubs, fonts, and chapters with `-map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 -c copy -max_interleave_delta 0`.
- **RTX 3060 12GB VRAM, Super-Chunks, Zero-VRAM Web Boundary & GPU Lock:**
  - TensorRT 10.16 FP16 / PyTorch CUDA FP16 (`batch_size=8`, Catmull-Rom collapsed 2x `_aa2x` for SRVGG 4x->2x, `memo_p010 <= 64` frames, `24s` CUDA Zero-Disk super-chunks, `max_workers=2` Open-GOP rolling preflight). Keep working set inside 12GB VRAM and `scratch/` / NVMe RAMDisk bounds.
  - Enforce Zero-VRAM isolation: `web/` and `web_gui.py` MUST never import `torch`, `tensorrt`, or `src.config` (mirror `DEDUP_POLICY_PARAMS`, `auto-pareto`, and `nvenc_multipass` in `web/fs_browser.py`).
  - Only 1 process may hold `.gpu_active.lock` (PID-checked). Zero-Disk (`--zero-disk`) is default; `--no-zero-disk` uses rolling chunk cleanup (`up_*.mp4` in `.temp_<basename>`).
- **E-Drive Isolation (`src/env_isolation.py`):** Keep all caches, ONNX/TensorRT artifacts, and temp directories on E-drive; never write to `C:\Users\...`.
- **Fast Verification & Execution Bans:**
  - Run `.\.venv\Scripts\python.exe scripts/smoke_test_pipeline.py` (`< 5s`) after pipeline, dedup, or mux edits (or delegate to [test-verifier](../agents/test-verifier.md)).
  - Never run `src/build_trt_engine.py`, delete `.temp_*` checkpoints, or launch full-episode upscales (`> 100` frames) without explicit user approval.
