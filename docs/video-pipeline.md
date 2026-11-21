# Video Pipeline Technical Reference

> **Super-Resolution Video Pipeline & GPU Inference Architecture**
> **Primary Contract:** [GEMINI.md](../GEMINI.md) | **Historical Measurements:** [engineering-log.md](archive/reports/engineering-log.md)

## 1. Core Modules

| Module | Responsibility |
| :--- | :--- |
| [`src/pipeline.py`](../src/pipeline.py) | Main orchestrator: true 3-stage overlap (`Decode N+1 \|\| GPU N \|\| Encode N-1`), 1-pass Concat-Mux, checkpoint resume |
| [`src/build_trt_engine.py`](../src/build_trt_engine.py) | ONNX export & TensorRT 10.16 FP16 runtime (`builder_optimization_level=3`, `execute_async_v3`, 2-slot `_out_ring`, shape/address cache) |
| [`src/trt_zero_copy_upscaler.py`](../src/trt_zero_copy_upscaler.py) | Zero-Disk Pipe engine: pinned `NHWC` `uint8` H2D, Triton JIT input norm & `p010le` color kernels, Cyclic Cel Belady `OPT`, Batch-0 sanity downsample |
| [`src/cuda_engine.py`](../src/cuda_engine.py) | PyTorch CUDA FP16 engine: `RealESRAnimeVideoV3` (8K 2-frame tail slicing), `AnimeJaNaiCompact2x` (`janai-fix`), Spandrel wrapper, `batch_size=8`, Cyclic Hardlink worker |
| [`src/models_resolver.py`](../src/models_resolver.py) | Model registry (`models/models.yaml`), per-series mapping (`shows.yaml`), SHA-256 verification |
| [`src/ramdisk.py`](../src/ramdisk.py) | Frame buffer routing: physical RAM disk (`UPSCALER_RAMDISK_DIR` / `"RAM"` label) -> NVMe Gen4 `scratch/rolling_buffer` with rolling cleanup |
| [`src/ffmpeg_ops.py`](../src/ffmpeg_ops.py) | PyAV probing + FFmpeg ops: `extract_frames_raw_rgb24` (`readinto` zero-copy), two-way BT.709, 10-bit NVENC, `mux_segments_directly` |
| [`src/config.py`](../src/config.py) | Runtime constants, Ampere `HMMA.F16` fast math, `CUDA_MODULE_LOADING=LAZY`, NVENC `CQ 18` `fullres`, PID `.gpu_active.lock` |

## 2. True 3-Stage Overlapped Execution

### A. Default Turbo Zero-Disk Mode (`--zero-disk=True` / `upscale.bat` / `upscale_turbo.bat`)
```text
Time ──────►
Chunk N-1: [EncodeWorker: drain 32-frame NVENC lookahead & finalize MP4] ──► up_{N-1}.mp4
Chunk N:   [GPU: Pinned NHWC -> TRT 10 FP16 -> Triton p010le]            ──► NVENC stdin pipe
Chunk N+1: [DecodeWorker: NVDEC/FFmpeg GOP -> readinto RGB24 RAM]        ──► Prefetched in 32GB RAM (0 disk PNGs)
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
- **Default Model:** `realesr-animevideov3` (`SRVGGNetCompact` 4x + Bicubic 0.5x -> 2x) plus official Native 2x models (`AnimeJaNaiCompact2x`, `SPAN`) via Spandrel `0.4.2`.
- **TensorRT 10 FP16 (`HMMA.16816.F16`):** Auto-builds `.engine` (`opt=3` + `TimingCache`), fuses `Conv2d + PReLU + PixelShuffle`, and reuses a 2-slot VRAM `_out_ring` (~199 MB/slot at `batch_size=8`) to avoid ~2,100 `torch.empty` calls/episode.
- **8K 2-Frame Micro-Slicing:** Intermediate `PixelShuffle(4)` (`7680x4320`) is sliced in 2-frame pairs before 0.5x downsampling, cutting 8K VRAM buffer spikes by 75%.
- **No `INT8 PTQ`:** Preserves 16-bit FP16 range across 33 `PReLU` layers to prevent 8-bit color banding on 10-bit `p010le` masters.
- **`AnimeJaNaiCompact2x` Fix (`janai-fix`):** Adds `F.interpolate(x, scale_factor=2.0, mode="nearest")` residual skip, matching Spandrel CPU bit-for-bit (PSNR 75-158 dB).

## 4. Two-Way BT.709 Color Matrix & 10-bit NVENC

- **1080p Extraction (YUV TV Limited -> RGB24 Full):** `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc`
- **4K Encoding (RGB24 Full -> `p010le` TV Limited):**
  - Zero-Disk: `@triton.jit` `_rgb_to_p010le_triton_kernel` (RGB `[0, 1]` clamp, BT.709 `Y: 64..940`, `UV: 64..960`, `<< 6`, center chroma).
  - Standard: `-vf scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv -pix_fmt p010le`
- **Container Color Metadata:** `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`
- **NVENC Parameters (`src/config.py:get_hevc_rate_args`):** `hevc_nvenc`, `main10` (`p010le`), `-preset p6`, `-tune hq`, `-cq 18 -b:v 0 -maxrate 45M -bufsize 90M`, `-multipass fullres`, `-rc-lookahead 32`, `-bf 4 -b_ref_mode middle`, `-spatial-aq 1 -aq-strength 6 -temporal-aq 1`.

## 5. 1-Pass Concat-Mux & Bit-Exact Stream Copy

`src/ffmpeg_ops.py:mux_segments_directly` combines `up_*.mp4` directly with source streams in one pass (eliminating ~8 GB `merged_4k.mp4` temp writes):
```bash
ffmpeg -y -f concat -safe 0 -i "concat_4k_list.txt" -i "source.mkv" \
  -map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 \
  -c copy -max_interleave_delta 0 "output_4K.mkv"
```
