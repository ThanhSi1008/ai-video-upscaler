---
trigger: glob
globs: "src/**/*.py, scripts/*upscale*.py, scripts/smoke_test_pipeline.py, models/*.yaml"
description: "10-bit p010le end-to-end invariant, RTX 3060 12GB VRAM/RAMDisk bounds, E-drive temp isolation, smoke_test_pipeline check, and GPU execution guards."
---

# Video & GPU Pipeline Scoped Rules

- **End-to-End 10-bit `p010le` & Two-Way BT.709 Invariant:**
  - Extract (Limited -> RGB24 Full): `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc`
  - Encode (RGB24 Full -> `p010le` Limited): `scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv`
  - Metadata tags: `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`
- **10-bit NVENC Studio Preset & Bit-Exact Stream-Copy Mux:**
  - `hevc_nvenc`, `main10`, `p010le`, preset `p6`, `CQ 18` (`maxrate 45M`), multipass `fullres`, `rc-lookahead 32`, `spatial-aq 1`, `aq-strength 6`, `temporal-aq 1`.
  - Mux original audio, softsubs, fonts, and chapters with `-map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 -c copy -max_interleave_delta 0`.
- **RTX 3060 12GB VRAM, RAMDisk & Single-GPU Lock:**
  - TensorRT 10.16 FP16 / PyTorch CUDA FP16 (`batch_size=8`, 3-stage decode/infer/encode overlap). Keep working set inside 12GB VRAM and `scratch/` / NVMe RAMDisk bounds.
  - Only 1 process may hold `.gpu_active.lock` (PID-checked). Zero-Disk (`--zero-disk`) is default; `--no-zero-disk` uses rolling chunk cleanup (`up_*.mp4` in `.temp_<basename>`).
- **E-Drive Isolation (`src/env_isolation.py`):** Keep all caches, ONNX/TensorRT artifacts, and temp directories on E-drive; never write to `C:\Users\...`.
- **Fast Verification & Execution Bans:**
  - Run `.\.venv\Scripts\python.exe scripts/smoke_test_pipeline.py` (`< 5s`) after pipeline, dedup, or mux edits (or delegate to [test-verifier](../agents/test-verifier.md)).
  - Never run `scripts/build_trt_engine.py`, delete `.temp_*` checkpoints, or launch full-episode upscales (`> 100` frames) without explicit user approval.
