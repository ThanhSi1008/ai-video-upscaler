---
trigger: glob
globs: "upscale.py, shows.yaml, src/**/*.py, scripts/*upscale*.py, scripts/smoke_test_pipeline.py, models/*.yaml"
description: "10-bit p010le end-to-end invariant, RTX 3060 12GB VRAM/RAMDisk bounds, zero-VRAM web boundary, E-drive temp isolation, smoke_test_pipeline check, and autonomous local hardware execution."
---

# Video & GPU Pipeline Scoped Rules

- **End-to-End 10-bit `p010le` & Two-Way BT.709 Invariant:**
  - Zero-Disk (`--zero-disk`): decode semi-planar `p010le` / `nv12` (promoted to `p010le` on grained sources while keeping `source_is_8bit`) -> `@triton.jit` `_yuv420_to_nchw_fp16_bt709_kernel` (`left` chroma + UV-pair-safe `PRE_DEBAND`) -> SR -> `_rgb_to_p010le_triton_kernel` (`left` `[1,2,1]/4 x [1,1]/2` chroma + `1080p` `SOFT_ANTIRING` + IGN dither). Keep `GPUColorConverterP010LE()` default `(chroma_siting="center", dither=False, soft_antiring=False)` for legacy tests.
  - Disk/RGB24 fallback: `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc:flags=bicubic+accurate_rnd+full_chroma_int` and inverse `pc -> tv` (`p010le`).
  - Metadata tags: `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`.
- **10-bit NVENC Studio Preset & Bit-Exact Stream-Copy Mux:**
  - `hevc_nvenc`, `main10`, `p010le`, preset `p6`, `CQ 18` (`maxrate 45M`), `--nvenc-multipass auto|fullres|qres` (default `auto`, `21.5 fps` knee, locked **per-episode** so `-f concat -c copy` never mixes SPS/PPS `extradata`), `rc-lookahead 32`, `spatial-aq 1`, `aq-strength 6`, `temporal-aq 1`. In-process PyAV `hevc_nvenc` MUST set `stream.bit_rate = 0` and never mutate `AVFrame.pts` in-place during lookahead.
  - Mux original audio, softsubs, fonts, and chapters with `-map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 -c copy -max_interleave_delta 0`.
- **RTX 3060 12GB VRAM, Super-Chunks, Zero-VRAM Web Boundary & GPU Lock:**
  - TensorRT 10.16 FP16 / PyTorch CUDA FP16 (`UpCunet2x` dynamic reflect pad + FP32 `SEBlock` + `trt_max_batch <= 2` at `>= 720p`, episode `--grain-restore` blend keyed by `(blend_alpha, blended_weights_sha256)`, Catmull-Rom `_aa2x` for SRVGG 4x->2x, `memo_p010 <= 64`, `24s` Zero-Disk super-chunks, `free_vram >= 2.2 GiB` NVENC guard). Keep working set inside 12GB VRAM and `scratch/` / NVMe RAMDisk bounds.
  - Enforce Zero-VRAM isolation: `web/` and `web_gui.py` MUST never import `torch`, `tensorrt`, `src.config`, or `src.logger` (mirror `DEDUP_POLICY_PARAMS`, `auto-pareto`, `grain_restore`, and `nvenc_multipass` in `web/fs_browser.py`).
  - Only 1 process may hold `.gpu_active.lock` (PID-checked). Zero-Disk (`--zero-disk`) is default; `--no-zero-disk` uses rolling chunk cleanup (`up_*.mp4` in `.temp_<basename>`).
- **E-Drive Isolation (`src/env_isolation.py`):** Keep all caches, ONNX/TensorRT artifacts, and temp directories on E-drive; never write to `C:\Users\...`.
- **Fast Verification & Autonomous Local Hardware Authority:**
  - Run `.\.venv\Scripts\python.exe scripts/smoke_test_pipeline.py` (`< 5s`) after pipeline, dedup, or mux edits (or delegate to [test-verifier](../agents/test-verifier.md)).
  - Full autonomous authority is granted to utilize 100% of local PC hardware (Core i5-12600K P-Core/E-Core CPU, RAM/NVMe, TensorRT 10 FP16 `src/build_trt_engine.py` compilation, PyTorch CUDA FP16, and 10-bit `hevc_nvenc` on RTX 3060 12GB) without asking for approval on GPU frame counts, TensorRT engine builds, full-episode upscales (pass `--no-subs` unless Cloud Gemini subtrans is approved), or hardware benchmarks (`scripts/benchmark_pipeline.py`, `scripts/silicon_roofline_probe.py`, `benchmarks/upscale/`, `RUN_GPU_TESTS=1`) whenever `.gpu_active.lock` is idle. Never delete `.temp_*` checkpoints without user confirmation.
