# Video Pipeline Technical Reference

> **Super-Resolution Video Pipeline & GPU Inference Architecture**
> **Primary Contract:** [GEMINI.md](../GEMINI.md) | **Historical Measurements:** [engineering-log.md](archive/reports/engineering-log.md)

## 1. Core Modules

| Module | Responsibility |
| :--- | :--- |
| [`src/pipeline.py`](../src/pipeline.py) | Main orchestrator: true 3-stage overlap (`Decode N+1 \|\| GPU N \|\| Encode N-1`), `24s` Zero-Disk super-chunks, `RollingDedupPlanner`, `--model auto-pareto`, 1-pass Concat-Mux, checkpoint resume |
| [`src/build_trt_engine.py`](../src/build_trt_engine.py) | ONNX export & TensorRT 10.16 FP16 runtime (`_PureFP16TRTExportModule` Catmull-Rom collapsed 2x `_aa2x`, `execute_async_v3`, 2-slot `_out_ring`, shape/address cache) |
| [`src/trt_zero_copy_upscaler.py`](../src/trt_zero_copy_upscaler.py) | Zero-Disk engine: direct `p010le`/`nv12` pinned H2D, `@triton.jit` BT.709 YUV->RGB & RGB->`p010le` (`left` chroma + IGN dither), bounded Cyclic Cel (`<= 64` frames), in-process PyAV `hevc_nvenc` |
| [`src/cuda_engine.py`](../src/cuda_engine.py) | PyTorch CUDA FP16 engine: `RealESRAnimeVideoV3` (analytical Catmull-Rom collapsed 4x->2x), `AnimeJaNaiCompact2x` (`janai-fix`), Spandrel antialiased wrapper, `batch_size=8` |
| [`src/models_resolver.py`](../src/models_resolver.py) | Model registry (`models/models.yaml`), per-series mapping (`shows.yaml`), scene-adaptive `--model auto-pareto`, SHA-256 verification |
| [`src/ramdisk.py`](../src/ramdisk.py) | Frame buffer routing: physical RAM disk (`UPSCALER_RAMDISK_DIR` / `"RAM"` label) -> NVMe Gen4 `scratch/rolling_buffer` with rolling cleanup |
| [`src/ffmpeg_ops.py`](../src/ffmpeg_ops.py) | PyAV probing + FFmpeg ops: `extract_frames_raw_yuv420` (`p010le`/`nv12` `readinto`), Open-GOP recovery, two-way BT.709 (`bicubic+accurate_rnd+full_chroma_int`), `mux_segments_directly` |
| [`src/config.py`](../src/config.py) | Runtime constants, Ampere `HMMA.F16` fast math, `DEFAULT_ZERO_DISK_SEGMENT_TIME=24`, `resolve_episode_nvenc_multipass` (`auto\|fullres\|qres`), PID `.gpu_active.lock` |

## 2. True 3-Stage Overlapped Execution

### A. Default Turbo Zero-Disk Mode (`--zero-disk=True` / `upscale.bat` / `upscale_turbo.bat`)
```text
Time ──────►  (24s GOP super-chunks + Rolling Parallel Preflight max_workers=2)
Chunk N-1: [EncodeWorker: in-process PyAV hevc_nvenc flush (or FFmpeg pipe)] ──► up_{N-1}.mp4
Chunk N:   [GPU: Pinned p010le/nv12 -> Triton YUV2RGB -> TRT 10 FP16 -> Triton p010le]
Chunk N+1: [DecodeWorker: FFmpeg GOP -> readinto 10-bit p010le / 8-bit nv12] ──► Prefetched in RAM
```

### B. Fallback Rolling Disk Mode (`--no-zero-disk` / `--disk`)
```text
Time ──────►
Chunk N-1: [EncodeWorker: Concat Demuxer + NVENC HEVC 10-bit]            ──► up_{N-1}.mp4 & delete PNGs
Chunk N:   [GPU: TRT 10 / CUDA FP16 + NTFS Hardlink Cyclic Cel]          ──► 1080p PNG -> 4K PNG
Chunk N+1: [DecodeWorker: FFmpeg extract deduped 1080p PNGs]             ──► Prefetched when buffer >= 1600 MB free
```

## 3. Inference Engine: TensorRT 10.16 FP16 & PyTorch CUDA FP16

- **Target Workstation:** RTX 3060 12GB (`GA106`, VRAM P0 `7500 MHz` + `HAGS=2` via [unlock_rtx3060_admin.bat](../scripts/unlock_rtx3060_admin.bat)) + Core i5-12600K (`HighQoS` P-Core affinity).
- **Default & Per-Show Routing:** `shows.yaml` maps `[BD*]` / `BluRay` to `animejanai-hd-v3-sharp1-ultracompact` + `dis-auto` and `[SubsPlease]` / `[Erai-raws]` Web-DL to `animejanai-hd-v3-ultracompact` + `dis-auto`, with `realesr-animevideov3` (`O1` Catmull-Rom collapsed 2x) as universal fallback. Explicit `--model auto-pareto` pairs same-sharpness-family `animejanai-hd-v3-compact` (16c static/close-up) + `animejanai-hd-v3-ultracompact` (8c sakuga) on a shared `DoubleBufferedPipeUpscaler` ring, switching only on scene cuts.
- **TensorRT 10 FP16 (`HMMA.16816.F16`):** Auto-builds `.engine` (`opt=3` + `TimingCache`, `_aa2x` cache tag for collapsed 4x->2x models), fuses `Conv2d + PReLU + PixelShuffle`, and reuses a 2-slot VRAM `_out_ring` (~199 MB/slot at `batch_size=8`).
- **Analytical 4x->2x Catmull-Rom Collapse (`O1`):** For `SRVGGNetCompact` 4x models (`animevideov3`), `PixelShuffle(4) + Nearest(4x) + AA Bicubic(0.5x)` (`B=0.0, C=0.5`) is collapsed algebraically into `Conv2d(48->12, 3x3, groups=3) + Conv2d(3->12, 3x3, groups=3) + PixelShuffle(2)` at 1080p (`Slice+Concat` replicate border), eliminating 100% of 8K intermediate tensors (`~2.0 GiB` VRAM saved, `677 ms` vs `727 ms/batch`). Non-SRVGG 4x Spandrel models fall back to 2-frame-sliced `F.interpolate(..., mode="bicubic", antialias=True)`.
- **No `INT8 PTQ`:** Preserves 16-bit FP16 range across 33 `PReLU` layers to prevent 8-bit color banding on 10-bit `p010le` masters.
- **`AnimeJaNaiCompact2x` Fix (`janai-fix`):** Adds `F.interpolate(x, scale_factor=2.0, mode="nearest")` residual skip, matching Spandrel CPU bit-for-bit (PSNR 75-158 dB).

## 4. End-to-End 10-Bit BT.709 Color Pipeline & In-Process NVENC (`O2`, `O3`, `O6`)

- **1080p Extraction:**
  - Zero-Disk (`extract_frames_raw_yuv420`): Decodes native 10-bit `p010le` (or 8-bit `nv12`) semi-planar frames directly into RAM (`50%` smaller pipe vs `rgb24` for 8-bit, zero 8-bit quantization loss on Hi10P/HEVC-10 sources), converted in VRAM via `@triton.jit` `_yuv420_to_nchw_fp16_bt709_kernel` (`left` `(0.0, 0.5)` bilinear chroma upsampling).
  - Disk/Fallback (`extract_frames`): `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc:flags=bicubic+accurate_rnd+full_chroma_int`.
- **4K Encoding (RGB `[0, 1]` -> `p010le` TV Limited):**
  - Zero-Disk: `@triton.jit` `_rgb_to_p010le_triton_kernel` (`Y: 64..940`, `UV: 64..960`, MPEG-2/HEVC `left` `[1,2,1]/4 x [1,1]/2` chroma siting + spatial Interleaved Gradient Noise `±0.499 LSB` dither), fed in-process to `_InProcessPyAVNVENCEncoder` (`stream.bit_rate = 0`, zero OS pipe copies; auto-falls back to `spawn_nvenc_pipe_writer`).
  - Standard: `-vf scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int -pix_fmt p010le`.
- **Container Color Metadata:** `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`
- **NVENC Parameters (`src/config.py:get_hevc_rate_args`):** `hevc_nvenc`, `main10` (`p010le`), `-preset p6`, `-tune hq`, `-cq 18 -b:v 0 -maxrate 45M -bufsize 90M`, `--nvenc-multipass auto|fullres|qres` (locked per-episode: `qres` when effective AI throughput exceeds the `22.70 fps` `fullres` wall, `fullres` otherwise), `-rc-lookahead 32`, `-bf 4 -b_ref_mode middle`, `-spatial-aq 1 -aq-strength 6 -temporal-aq 1`.

## 5. 1-Pass Concat-Mux & Bit-Exact Stream Copy

`src/ffmpeg_ops.py:mux_segments_directly` combines `up_*.mp4` directly with source streams in one pass (eliminating ~8 GB `merged_4k.mp4` temp writes):
```bash
ffmpeg -y -f concat -safe 0 -i "concat_4k_list.txt" -i "source.mkv" \
  -map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 \
  -c copy -max_interleave_delta 0 "output_4K.mkv"
```
