# ROOFLINE THEORETICAL MODEL & HARDWARE PERFORMANCE ANALYSIS
## First-Principles Computational Complexity, Consumer Ampere GA106 Silicon Ceilings, Empirical Micro-Probing, and Overlapped Concurrency Modeling for 4K Anime Video Upscaling

> - **Document Class:** Publication-Grade Architectural & Theoretical Specification (Calibrated R3 Final)
> - **Target Workstation:** Intel Core i5-12600K | Gigabyte B760M G P WIFI DDR4 (`32 GiB` DDR4-2133 JEDEC) | NVIDIA GeForce RTX 3060 12GB (`GA106-302-A1`, P0 Locked @ `7,501 MHz`) | `WD_BLACK SN7100 500GB` NVMe Gen4
> - **Canonical Models:** `realesr-animevideov3` (`64f16c` 4x $\to$ 2x), `animejanai-hd-v3-sharp1-ultracompact` (`64f8c` Native 2x), & `animejanai-hd-v3-sharp1-superultracompact` (`24f8c` Native 2x)
> - **Associated Modules & Probes:** `src/pipeline.py`, `src/trt_zero_copy_upscaler.py`, `src/build_trt_engine.py`, `src/config.py`, `scripts/silicon_roofline_probe.py`, `out/benchmark_r3_final/silicon_probe.json`
> - **Verification Command:** `.\.venv\Scripts\python.exe scripts/verify_offline.py`

---

## Executive Summary

This specification establishes the formal first-principles computational complexity, consumer **NVIDIA Ampere GA106 (`sm_86`)** silicon boundaries, and pipelined concurrency theorems governing the local 4K anime video upscaling engine, directly cross-verified against hardware micro-benchmarks (`out/benchmark_r3_final/silicon_probe.json`) and audited workstation telemetry (`scratch/hw_and_model_exact_audit.json`):

1. **Exact Consumer Ampere GA106 Silicon Correction (Resolving the Legacy 2x–4x Sparsity/A100 Error):**
   Earlier drafts plugged datacenter A100 (`GA100`) / 2:4-sparse throughput (`512` and `1,024` dense FMAs/SM/cycle = `50.96` and `101.90 TFLOPS` @ `1,777 MHz`, knee `283.06 FLOPs/byte`) into the GA106 Roofline. Per the *NVIDIA Ampere GA102 GPU Architecture Whitepaper V2.1* (Tables 1 & 3) and live `cublasGemmEx` silicon probing on this `RTX 3060 12GB` (`28 SMs`, `4` 3rd-Gen Tensor Cores/SM, `0%` weight sparsity):
   - **IEEE FP32 Shader Cores (`128 FMAs/SM/clk`):** **`12.74 TFLOPS`** @ `1,777 MHz` (`13.98 TFLOPS` @ `1,950 MHz`), Roofline knee **`35.38 FLOPs/byte`** vs `360.0 GB/s` GDDR6 (empirically **`6.702 TFLOPS`** on `4096^3` `allow_tf32=False` GEMM due to `sm_86` shared FP32/INT32 issue).
   - **Dense Tensor Core FP16 w/ FP32 Accumulate (`HMMA.16816.F32`, GeForce half-rate `256 FMAs/SM/clk` = `512 FLOPs/SM/clk`):** **`25.48 TFLOPS`** @ `1,777 MHz` and **`27.96 TFLOPS`** @ `1,950 MHz` (`50.95 TFLOPS` applies *only* to 2:4 structured sparsity), Roofline knee **`70.76 FLOPs/byte`** (@ `1.777 GHz`) / **`77.65 FLOPs/byte`** (@ `1.95 GHz`). Empirically verified at **`23.211 TFLOPS`** (`83.03%` of live `1,950 MHz` silicon; PyTorch `2.6.0+cu124` `torch.matmul` FP16 always dispatches `CUBLAS_COMPUTE_32F` at `23.178–23.344 TFLOPS`).
   - **Dense Tensor Core FP16 w/ FP16 Accumulate (`HMMA.16816.F16`, GeForce full-rate `512 FMAs/SM/clk` = `1,024 FLOPs/SM/clk`):** **`50.95 TFLOPS`** @ `1,777 MHz` (`49.78 TFLOPS` @ `1,736.25 MHz` TDP-governed clock; `101.90 TFLOPS` applies *only* to 2:4 structured sparsity), theoretical Roofline knee **`141.53 FLOPs/byte`** vs `360.0 GB/s`, and empirical sustained knee **`128.36 FLOPs/byte`** (`38.844 TFLOPS` `cublasGemmEx(16F)` / `302.61 GB/s` STREAM Triad).
   - **Narrow-Channel `3x3 Conv2d(64->64)` Implicit-GEMM Ceiling under `170W` TGP:** Compiled TensorRT 10.16 engines (`sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32...tensor16x8x16`) achieve **`28.27–28.77 TFLOPS` sustained** (`11.00 FPS` on `animevideov3`, `22.83 FPS` on `sharp1-ultracompact` — **`2.37x` faster** than eager cuDNN NCHW's `12.140 TFLOPS`), bounded by `N = C_out = 64` CTA tile width, compulsory `4.25 GB/batch/layer` off-chip GDDR6 activation streaming (`265.42 MB/frame` = `112.5x`, and `4.25 GB/batch` = `1,800x` larger than GA106's `2.25 MiB` L2 cache), and `100.2%–100.5%` saturation of the **`170.0 W` physical board power limit** (`170.38–170.93 W` peak).

2. **First-Principles Resolution of Every Telemetry Paradox Across All 3 Canonical Models:**
   - **`animevideov3` (`64f16c`, `2.571 TFLOPs/frame`, `I_realized = 252.70 FLOPs/B`):** Because `252.70 FLOPs/B` sits **`1.79x–3.57x` to the right** of every GA106 Roofline knee (`141.53`, `128.36`, `77.65`, and `70.76 FLOPs/B`), the model is **purely Tensor-Core Compute & `170W` TGP Bound** — *not* memory-bandwidth bound. At `11.00 FPS` isolated TRT (`10.52 FPS` steady), compulsory external DRAM traffic is `106.9–111.8 GB/s` (`31.1%` of `360.0 GB/s` peak / `36.9%` of `302.61 GB/s` STREAM Triad), which plus L2 halo sector fetches explains the measured **`44.53%` GDDR6 memory controller utilization** from first principles.
   - **`sharp1-ultracompact` (`64f8c` Native 2x, `1.260 TFLOPs/frame`, `I_realized = 272.45 FLOPs/B`):** Isolated TRT reaches **`22.83 FPS` (`43.80 ms/frame`)**, matching the empirically measured **7th-Gen NVENC `-multipass fullres` steady-state ASIC ceiling (`22.70 FPS`, `44.05 ms/frame`)** to within **`0.57%`** — achieving simultaneous dual-ASIC saturation (`94.24%` GPU compute + `89.06%` NVENC).
   - **`sharp1-superultracompact` (`24f8c` Native 2x, `0.1859 TFLOPs/frame`, `I_realized = 94.32 FLOPs/B`):** Isolated TRT reaches **`44.91 FPS` (`178.15 ms/batch`)**, outrunning the `22.70 FPS` 7th-Gen NVENC `-multipass fullres` ASIC by `1.98x` and shifting the E2E pipeline **100% into the NVENC ASIC-Bound + Backpressure Regime** (`96.36%` avg / `100%` peak NVENC, `64.20%` GPU, `20.80 FPS` E2E / `24.56 FPS` steady).

3. **Overlapped Concurrency & Empirically Calibrated `-multipass fullres` Amdahl Bounds:**
   Proves 100% hiding of CPU/NVDEC Decode ($N+1$) and NVENC Encode ($N-1$) behind GPU Inference ($N$) for interior chunks whenever $T_2 > T_3$, quantifies boundary overhead (`~5.02 s`, `77.17%` single-chunk preview efficiency), and establishes exact Amdahl deduplication knees against both production 2-pass `-multipass fullres` (`22.70 FPS` steady / `44.05 ms/f`: `2.06x` ceiling at `51.56%` dedup for `v3`) and single-pass NVENC (`~45–50 FPS`).

---

## 1. Audited Host Workstation Hardware & Toolchain Matrix

All physical parameters below were queried directly from WMI/CIM (`Win32_Processor`, `Win32_PhysicalMemory`, `Get-PhysicalDisk`), `nvidia-smi -q`, `.venv` package metadata (`scratch/hw_and_model_exact_audit.json`), and live CUDA device properties (`out/benchmark_r3_final/silicon_probe.json`).

```text
+----------------------------------------------------------------------------------------------------+
|                              AUDITED HOST WORKSTATION SILICON TOPOLOGY                             |
+----------------------------------------------------------------------------------------------------+
|  INTEL CORE i5-12600K CPU on GIGABYTE B760M G P WIFI DDR4 (Windows 11 Pro 64-bit Build 26300)      |
|  [6 Golden Cove P-Cores @ 3.7-4.9 GHz (4.5 GHz All-Core)]  [4 Gracemont E-Cores @ 2.8-3.6 GHz]     |
|  L1D/I: 48K/32K (P), 32K/64K (E) | L2: 9,728 KB (9.5 MiB) | L3: 20,480 KB (20.0 MiB Smart Cache)   |
|  Peak Theoretical CPU FP32 Throughput: 1,081.6 GFLOPS (1.08 TFLOPS, AVX2 Dual FMA3)                |
+--------------------------------------------------+-------------------------------------------------+
                                                   | Host Memory Bus: Dual-Channel DDR4 (128-bit)
                                                   | Active JEDEC 2,133 MT/s @ 1.20V -> 34.13 GB/s
                                                   v
+----------------------------------------------------------------------------------------------------+
|  SYSTEM INTERCONNECT & NVMe STORAGE TOPOLOGY                                                       |
|  - GPU Link: PCIe 4.0 x16 (31.51 GB/s Unidir Theoretical | 22.81 GB/s H2D / 22.64 GB/s D2H Meas.)  |
|    BAR1 Aperture: 256 MiB (Resizable BAR: Disabled)                                                |
|  - Disk 0 (C:/D: OS): OCPC MFL-300 512GB (NVMe PCIe Gen3 x4)                                       |
|  - Disk 1 (E: Workspace/.cache/scratch): WD_BLACK SN7100 500GB (NVMe PCIe Gen4 x4, 6800/5800 MB/s) |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|  NVIDIA GEFORCE RTX 3060 12GB (GA106-302-A1 AMPERE, Compute Capability 8.6 / sm_86)                |
|  28 SMs | 3,584 FP32 CUDA Cores (128/SM) | 112 3rd-Gen Tensor Cores (4/SM) | 28 2nd-Gen RT Cores   |
|  Clocks: Base 1,320 MHz | Rated Boost 1,777 MHz | Sustained Boost 1,701-1,957 MHz | Max 2,130 MHz  |
|  Active On-Chip L2 Cache: 2,359,296 Bytes = 2,304 KiB (2.25 MiB on GA106-302-A1)                   |
|  Board Power Envelope (TGP): 170.0 W (Default = Max = 170.0 W)                                     |
|                                                                                                    |
|  VRAM Memory Subsystem (P0 Locked State @ 7,501 MHz / 15.0 Gbps, 192-bit GDDR6, 12,288 MiB):       |
|  - Theoretical Peak Bandwidth: 360.0 GB/s | Empirical Sustained STREAM Triad: 302.61 GB/s (84.1%)  |
|                                                                                                    |
|  Dedicated Video ASICs:                                                                            |
|  - 1x 5th-Gen NVDEC (8/10-bit HEVC, 8-bit H.264, AV1; 1080p H.264 Hi10P decoded on CPU P-Cores)    |
|  - 1x 7th-Gen NVENC (4K HEVC Main10 p010le p6 CQ18 lookahead 32 AQ):                               |
|      * Production `-multipass fullres` (2-pass 4K): 22.70 FPS steady (44.05 ms/f; 20.67 FPS/240f)  |
|      * Single-pass reference (`-multipass disabled`): 29.50 FPS @ 120f (~45-50 FPS steady)         |
+----------------------------------------------------------------------------------------------------+
```

### 1.1 Audited Hardware & Software Parameter Table

| Subsystem | Metric / Parameter | Audited Value / Specification | Theoretical / Empirical Bound |
| :--- | :--- | :--- | :--- |
| **OS & Board** | OS & Kernel | Windows 11 Pro 64-bit `10.0.26300` (Build `26300`, WDDM) | Gigabyte `B760M G P WIFI DDR4` (PCIe 5.0 root) |
| **CPU** | Processor Model | `12th Gen Intel(R) Core(TM) i5-12600K` | 10 Cores (6P + 4E) / 16 Threads |
| | Operating Clocks | P: `3.7` Base / `4.5` All-Core / `4.9 GHz` Max; E: `2.8–3.6 GHz` | AVX2, FMA3, BMI2, AES-NI, VNNI |
| | Total CPU FP32 Peak | $(6 \times 4.5 \times 32) + (4 \times 3.4 \times 16)$ | **`1,081.6 GFLOPS` (`1.08 TFLOPS`)** |
| | Cache Hierarchy | L2: `9,728 KB` (`9.5 MiB` = `7.5M` P + `2.0M` E); L3: `20,480 KB` (`20 MiB`) | 64-byte cache lines |
| **Host RAM** | Installed DIMMs | `32 GiB` (`16 GiB` G.Skill `F4-3200C16-16GVK` + `16 GiB` PNY `16GF2X16QFHH36`) | Dual-Channel 128-bit (`2 x 64-bit`) |
| | Active Clock & Rate | **`2,133 MT/s` @ `1.20 V` JEDEC** (`ConfiguredClockSpeed = 2133`) | **`34.13 GB/s`** (`34.128 GB/s`; XMP off due to mixed kit) |
| **Interconnect**| GPU Host Link | PCI Express 4.0 x16 (`16.0 GT/s`, `128b/130b`, `BAR1 = 256 MiB`, ReBAR Off) | **`31.51 GB/s`** Unidir (**`22.81 GB/s` H2D / `22.64 GB/s` D2H** meas.) |
| **Storage** | Disk 0 (`C:` / `D:`) | `OCPC MFL-300 512GB` (NVMe PCIe Gen3 x4, FW `APF1M3R1`) | OS partition (`C:` `378.21 GiB`); bypassed by `.cache` & `scratch/` |
| | Disk 1 (`E:`) | **`WD_BLACK SN7100 500GB`** (NVMe PCIe Gen4 x4, FW `7615M0WD`) | Rated **`6,800 MB/s` Read / `5,800 MB/s` Write** (`0` PNGs in Zero-Disk) |
| **GPU Silicon** | Architecture & Die | NVIDIA GeForce RTX 3060 (`GA106-302-A1`, `sm_86`, Driver `617.42`) | `28` SMs, `3,584` CUDA Cores, `112` 3rd-Gen Tensor Cores |
| | Clocks & Power | Base `1,320 MHz` \| Rated Boost `1,777 MHz` \| VBIOS Max `2,130 MHz` | Board TGP Limit: **`170.0 W`** (`Min 100 W`, `Max 170 W`) |
| | Shader FP32 Peak | $28 \times 128 \text{ FMAs/SM/clk} \times 2 \times f_{\text{clk}}$ | **`12.74 TFLOPS`** @ `1.777 GHz` (**`6.702 TFLOPS`** GEMM meas.) |
| | Dense TC FP32-Accum | `HMMA.16816.F32`: $28 \times 256 \text{ FMAs/SM/clk} \times 2 \times f_{\text{clk}}$ | **`25.48 TFLOPS`** @ `1.777 GHz` / **`27.96 TFLOPS`** @ `1.95 GHz` (**`23.211 T`** meas.) |
| | Dense TC FP16-Accum | `HMMA.16816.F16`: $28 \times 512 \text{ FMAs/SM/clk} \times 2 \times f_{\text{clk}}$ | **`50.95 TFLOPS`** @ `1.777 GHz` (**`38.844 T`** `4096^3` GEMM; **`28.27–28.77 T`** `N=64` SR Conv) |
| | 2:4 Sparse TC Peak | Requires 50% pruned `2:4` structured weights (unused by dense SR) | `50.95 TFLOPS` (FP32-acc) / `101.90 TFLOPS` (FP16-acc) @ `1.777 GHz` |
| **VRAM & L2** | GDDR6 Bus (P0) | `12,288 MiB` (`12,115 MiB` WDDM usable), `192-bit` @ `7,501 MHz` (`15 Gbps`) | **`360.0 GB/s`** Peak \| **`302.61 GB/s`** STREAM Triad (`295.69 GB/s` Copy) |
| | On-Chip L2 Cache | **`2,359,296 bytes` = `2,304 KiB` (`2.25 MiB`)** on `GA106-302-A1` | `torch.cuda.get_device_properties(0).L2_cache_size` |
| **Video ASICs**| 5th-Gen NVDEC | `1` hardware decoder | 8-bit H.264/HEVC ~800–1000 FPS; 10-bit H.264 Hi10P on CPU P-Cores (`~166 FPS`) |
| | 7th-Gen NVENC | `1` hardware encoder (`hevc_nvenc` Main10 `p010le` `p6` `CQ 18` `lookahead 32` AQ) | **`22.70 FPS`** steady (`44.05 ms/f`) w/ `-multipass fullres` (`29.50 FPS` 120f 1-pass) |
| **`.venv` Stack**| AI Toolchain | Python **`3.11.9`**, PyTorch **`2.6.0+cu124`** (CUDA `12.4`, cuDNN `90100`) | TensorRT **`10.16.1.11`**, Triton **`3.2.0`** (`triton-windows 3.2.0.post21`), PyAV **`18.1.0`**, FFmpeg **`8.1.2`** |

---

## 2. Analytical AI Model Architecture & Computational Complexity (All 3 Canonical Regimes)

### 2.1 Convolution MAC / FLOP Accounting Convention

For a dense 2D convolutional layer with spatial dimensions $(H, W)$, channels $(C_{\text{in}}, C_{\text{out}})$, kernel $(K_h, K_w) = (3, 3)$, stride $S=1$, and padding $P=1$ on 1080p anime frames ($H = 1080, W = 1920, N_{\text{pix}} = 2,073,600$):
$$\text{MACs} = K_h \cdot K_w \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W, \qquad \text{FLOPs} = 2 \times \text{MACs}$$

---

### 2.2 Regime 1 Model: `realesr-animevideov3` (`SRVGGNetCompact` `64f16c`, 4x + 0.5x Bicubic $\to$ 2x)

`realesr-animevideov3` executes 18 `Conv2d` layers (`num_feat=64, num_conv=16`), 17 `PReLU` layers, a 4x `PixelShuffle` + nearest-neighbor 8K ($4320 \times 7680$) residual branch, and a separable 0.5x bicubic downsampler to emit $2160 \times 3840$ 4K frames.

| Layer Identifier | Layer Type | Weight Shape | Bias | Parameters | Computational MACs | Computational FLOPs |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `body.0` | `Conv2d` | `(64, 3, 3, 3)` | `(64,)` | `1,792` | `3,583,180,800` | `7,166,361,600` |
| `body.1` | `PReLU` | `(64,)` | — | `64` | — | `132,710,400` |
| `body.2..33` | `16x (Conv2d + PReLU)` | `16 x (64, 64, 3, 3)` | `16 x (64,)` | `591,872` | `1,223,059,046,400` | `2,448,241,459,200` |
| `body.34` | `Conv2d` | `(48, 64, 3, 3)` | `(48,)` | `27,696` | `57,330,892,800` | `114,661,785,600` |
| `upsampler` | `PixelShuffle(4)` | — | — | `0` | `0` | `0` |
| `residual` | `Nearest(4x) + Add` | — | — | `0` | `0` | `99,532,800` |
| `tail` | `Bicubic(0.5x) 8K->4K`| — | — | `0` | `199,065,600` | `398,131,200` |
| **Total (`v3`)** | **18 Convs + 17 PReLU + Tail** | — | — | **`621,424`** (`1.243 MB` FP16) | **`1,283,973,120,000` Conv** | **`2,570,699,980,800` (`2.571 TFLOPs`)** |

- **Convolutional FLOPs:** `2,567,946,240,000 FLOPs` (`2.5679 TFLOPs` = `99.893%` of grand total `2.5707 TFLOPs`).
- **Off-Chip GDDR6 DRAM Traffic & Arithmetic Intensity (FP16):**
  Each 64-channel 1080p activation tensor occupies $64 \times 1080 \times 1920 \times 2 = \mathbf{265,420,800\text{ bytes}}$ (`265.42 MB`), exceeding GA106's **`2.25 MiB` (`2,359,296 B`)** L2 cache by **`112.5x`** and forcing full external GDDR6 streaming between layers:
  - **Ideal Fused Baseline (`M_DRAM,ideal`):** Input read (`12.44 MB`) + `16` hidden writes (`4,246.73 MB`) + `16` hidden reads (`4,246.73 MB`) + `Conv34` write (`199.07 MB`) + fused tail read (`211.51 MB`) + 4K write (`49.77 MB`) + weights (`1.24 MB`) = **`8,967,489,248 bytes` (`8.968 GB`)** $\implies$ **`AI_ideal = 286.36 FLOPs/byte`** (conv) / `286.67 FLOPs/byte` (total).
  - **Realized TensorRT 10 Engine (`M_DRAM,realized`):** TensorRT compiles the 8K tail into 4 discrete kernels (`DepthToSpace`, `/Resize`, `PWN(/Add)`, `/Resize_1`), cycling 6 additional passes over `[B, 3, 4320, 7680]` FP16 (`6 x 199,065,600 = 1,194,393,600 bytes` = `+1.194 GB`):
    $$M_{\text{DRAM, realized}} = 8,967,489,248 + 1,194,393,600 = \mathbf{10,161,882,848\text{ bytes}} \approx \mathbf{10.162\text{ GB/frame}}$$
    $$\text{AI}_{\text{realized, v3}} = \frac{2,567,946,240,000}{10,161,882,848} = \mathbf{252.70\text{ FLOPs/byte}}$$

---

### 2.3 Regime 2 Model: `animejanai-hd-v3-sharp1-ultracompact` (`64f8c`, Native 2x)

`sharp1-ultracompact` halves backbone depth to 8 intermediate `64 -> 64` convolutions (`10` `Conv2d` total) and synthesizes 4K directly via `Conv2d(64 -> 12)` + `PixelShuffle(2)`, eliminating the 8K bicubic tail completely.

| Layer Identifier | Layer Type | Weight Shape | Bias | Parameters | Computational MACs | Computational FLOPs |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `body.0..1` | `Conv2d + PReLU` | `(64, 3, 3, 3)` | `(64,)` | `1,856` | `3,583,180,800` | `7,299,072,000` |
| `body.2..17` | `8x (Conv2d + PReLU)` | `8 x (64, 64, 3, 3)` | `8 x (64,)` | `295,936` | `611,529,523,200` | `1,224,120,729,600` |
| `body.18` | `Conv2d` | `(12, 64, 3, 3)` | `(12,)` | `6,924` | `14,332,723,200` | `28,665,446,400` |
| `tail` | `PixelShuffle(2) + Nearest(2x) Add` | — | — | `0` | `0` | `24,883,200` |
| **Total (`sharp1_uc`)** | **10 Convs + 9 PReLU** | — | — | **`304,716`** (`0.609 MB` FP16) | **`629,445,427,200` Conv** | **`1,260,110,131,200` (`1.260 TFLOPs`)** |

- **Workload Reduction Ratio vs. `animevideov3`:** `2,570.70 / 1,260.11 =` **`2.0401x`** (`2.0398x` conv FLOPs).
- **GDDR6 DRAM Traffic & Arithmetic Intensity (FP16):**
  - **Ideal Fused (`M_DRAM,ideal`):** Input (`12.44 MB`) + `8` writes & `8` reads (`4,246.73 MB`) + `Conv18` write (`49.77 MB`) + fused 4K tail read (`62.21 MB`) + 4K write (`49.77 MB`) + weights (`0.61 MB`) = **`4,421,524,632 bytes` (`4.422 GB`)** $\implies$ **`AI_ideal = 284.72 FLOPs/byte`**.
  - **Realized TensorRT 10 Engine (`M_DRAM,realized`):** Discrete 4K `DepthToSpace` + `Nearest` + `Add` cycles `+4 x 49,766,400 = +199,065,600 bytes` (`+0.199 GB`, **6.0x smaller** tail traffic than `animevideov3`'s 8K tail!):
    $$M_{\text{DRAM, realized}} = \mathbf{4,620,590,232\text{ bytes}} \approx \mathbf{4.621\text{ GB/frame}}, \qquad \text{AI}_{\text{realized, uc}} = \mathbf{272.45\text{ FLOPs/byte}}$$

---

### 2.4 Regime 3 Model: `animejanai-hd-v3-sharp1-superultracompact` (`24f8c`, Native 2x)

Audited directly from `models_bench/2x_AnimeJaNai_HD_V3_ModelsOnly/2x_AnimeJaNai_HD_V3Sharp1_SuperUltraCompact.pth` (`SHA-256: 200b6f82...`) and its compiled `.engine` binary, `sharp1-superultracompact` narrows feature channels to **`num_feat = 24`** across `num_conv = 8` hidden layers (`10` `Conv2d` total):

| Layer Identifier | Layer Type | Weight Shape | Bias | Parameters | Computational MACs | Computational FLOPs |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `body.0` | `Conv2d` | `(24, 3, 3, 3)` | `(24,)` | `672` | `1,343,692,800` | `2,687,385,600` |
| `body.1` | `PReLU` | `(24,)` | — | `24` | — | `49,766,400` |
| `body.2..17` | `8x (Conv2d + PReLU)` | `8 x (24, 24, 3, 3)` | `8 x (24,)` | `41,856` | `85,996,339,200` | `172,390,809,600` |
| `body.18` | `Conv2d` | `(12, 24, 3, 3)` | `(12,)` | `2,604` | `5,374,771,200` | `10,749,542,400` |
| `tail` | `PixelShuffle(2) + Nearest(2x) Add` | — | — | `0` | `0` | `24,883,200` |
| **Total (`sharp1_suc`)** | **10 Convs + 9 PReLU** | — | — | **`45,156`** (`90.3 KB` FP16) | **`92,714,803,200` Conv** | **`185,902,387,200` (`0.1859 TFLOPs`)** |

- **Workload Reduction Ratio:** **`13.83x` lighter** than `animevideov3` (`2,570.70 / 185.90`) and **`6.78x` lighter** than `sharp1-ultracompact`!
- **GDDR6 DRAM Traffic, Arithmetic Intensity, & `C=24` Tensor Core Tile Padding:**
  - Single 24-channel 1080p activation = `24 x 1080 x 1920 x 2 = 99,532,800 bytes` (`99.53 MB`, `42.2x` larger than `2.25 MiB` L2).
  - **Ideal Fused DRAM:** **`1,766,797,512 bytes` (`1.767 GB/frame`)** $\implies$ **`AI_ideal = 104.95 FLOPs/byte`** (`105.22` total).
  - **Realized TensorRT 10 DRAM:** **`1,965,863,112 bytes` (`1.966 GB/frame`)** $\implies$ **`AI_realized = 94.32 FLOPs/byte`** (`78.44 FLOPs/B` including unfused `body.1` PReLU).
  - **HMMA `256x32x32` Tile Quantization:** Because `C = 24` is not a multiple of 32, TensorRT's `sm80_xmma_fprop_implicit_gemm_..._tilesize256x32x32` kernel pads `N: 24 -> 32` (`+33.3%`) and `K: 216 -> 224`, issuing `0.2567 TFLOPs/frame` of hardware HMMA operations (`1.383x` tile padding) and reaching **`44.91 FPS` (`178.148 ms/batch`)** isolated on GPU — nearly **2x faster** than the 7th-Gen NVENC `-multipass fullres` encoder (`22.70 FPS`), completely shifting the E2E bottleneck from Tensor Cores to the NVENC ASIC!

---

## 3. Calibrated Consumer Ampere GA106 Roofline Model

Under the classical Roofline Model (Williams et al., 2009), attainable compute throughput $P$ (GFLOPS) as a function of operational arithmetic intensity $I$ (FLOPs/byte) is bounded by:
$$P(I) = \min\left(P_{\text{peak}}, \; I \times B_{\text{peak}}\right), \qquad I_{\text{knee}} = \frac{P_{\text{peak}}}{B_{\text{peak}}}$$

```text
  Attainable Compute Performance on GeForce RTX 3060 12GB (GA106-302-A1, sm_86) [TFLOPS]
    ^
50.95 T + - - - - - - - - - - - - - - - - - - - - - - - +=====================================================+
        |   (101.90 T requires 2:4 sparsity - unused!) /| GA106 Dense TC FP16-Accum Peak (50.95 T @ 1777 MHz) |
38.84 T + - - - - - - - - - - - - - - - - - - - - - - *-+-----------------------------------------------------+
        |                                            /  | Empirical cublasGemmEx(16F) 4096^3 Ceiling (38.844T)|
28.77 T + - - - - - - - - - - - - - - - - - - - - - / - | - - - - - - - - - - - - - - *[OP-2: sharp1_uc 28.77T]
28.27 T + - - - - - - - - - - - - - - - - - - - - -/ - -|- - - - - - - - - *[OP-1: animevideov3 28.27T @ 170W]
25.48 T + - - - - - - - - - - - - - - +-----------+-----+-----------------------------------------------------+
        |                            /| GA106 Dense TC FP32-Accum Peak: 25.48T @ 1.777G / 27.96T @ 1.95G      |
23.21 T + - - - - - - - - - - - - - * | (Empirical cublasGemmEx(32F) / torch.matmul FP16 = 23.211 TFLOPS)     |
        |                          /  |                 |
12.74 T + - - - - - - +-----------+   |                 |
        |            /| FP32 Peak |   |                 |
 8.35 T + - - - - - / +-(6.70T m.)+ - | - *[OP-3: sharp1_suc 24f8c = 8.35T @ 94.32 FLOPs/B -> 44.91 FPS TRT,  |
        |          /  |           |   |    capped E2E at ~22.70 FPS by 7th-Gen NVENC -multipass fullres ASIC!]|
        |  GDDR6  /   |           |   |                 |
        |  Slope /    |           |   |                 |
 0.00 T +-------+-----+-----------+---+-----------------+------------------+---------------+------------------>
        0     35.38 70.76       77.65 94.32          128.36/141.53       252.70          272.45   I (FLOPs/B)
             (FP32) (TC-FP32    (TC-FP32 (OP-3        (Emp./Theo.        (OP-1 v3)      (OP-2 uc)
                     @1.78G)     @1.95G)  suc)         TC-FP16 Knee)
```

### 3.1 Exact GA106 Silicon Ceilings & Roofline Knees (Theoretical vs. `silicon_probe.json`)

Unlike datacenter `GA100` (`A100`, `1,024` dense FP16 FMAs/SM/clk), consumer GeForce Ampere `GA10x` (`sm_86`, `28 SMs`, `4` 3rd-Gen Tensor Cores/SM) executes **`128`** IEEE FP32 FMAs/SM/clk, **`256`** dense FP16 FMAs/SM/clk with FP32 accumulate (`HMMA.16816.F32`, halved on GeForce), and **`512`** dense FP16 FMAs/SM/clk with FP16 accumulate (`HMMA.16816.F16`). Against **`360.0 GB/s`** theoretical GDDR6 (`7,501 MHz` P0) and **`302.61 GB/s`** empirical sustained STREAM Triad (`silicon_probe.json`):

| Execution Regime (`GA106-302-A1`, `28 SMs`, Dense `0%` Sparsity) | Formula (FLOPs/cycle) | Theoretical Peak @ `1,777 MHz` (`1,950 MHz` Boost) | Empirical `silicon_probe.json` Sustained Measurement | Theoretical Knee vs `360.0 GB/s` | Empirical Knee vs `302.61 GB/s` Triad |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. IEEE FP32 CUDA Cores** | $28 \times 128 \times 2 = 7,168$ | **`12.74 TFLOPS`** (`13.98 TFLOPS`) | **`6.702 TFLOPS`** (`4096^3`, `1,702 MHz`, `54.93%`) | **`35.38 FLOPs/B`** (`38.83` @ `1.95G`) | **`22.15 FLOPs/B`** (`18.62` vs `360`) |
| **2. Dense TC FP16 (`FP32-Accum` `HMMA.16816.F32`)** | $28 \times 256 \times 2 = 14,336$ | **`25.48 TFLOPS`** (**`27.96 TFLOPS`**) *(2:4 sparse: `50.95 T`)* | **`23.211 TFLOPS`** (`cublasGemmEx 32F`, `83.03%` @ `1.95G`; `torch.matmul` = `23.18–23.34 T`) | **`70.76 FLOPs/B`** (**`77.65`** @ `1.95G`) | **`76.70 FLOPs/B`** (`64.47` vs `360`) |
| **3. Dense TC FP16 (`FP16-Accum` `HMMA.16816.F16`)** | $28 \times 512 \times 2 = 28,672$ | **`50.95 TFLOPS`** (`49.78 T` @ `1,736 MHz` TDP; `55.91 T` @ `1.95G`) *(2:4 sparse: `101.90 T`)* | **`38.844 TFLOPS`** (`cublasGemmEx 16F`, `78.03%` @ `1,736 MHz`, `cos=0.999995`; `42.63 T` burst) | **`141.53 FLOPs/B`** (`138.28` @ `1.74G`) | **`128.36 FLOPs/B`** (`107.90` vs `360`) |
| **4. SR `3x3 Conv2d(64->64)` Implicit GEMM (`170W` TGP)** | `tilesize256x64x32` `HMMA.16816.F16` (`N=64`) | Practical `N=64` + `170W` TGP Envelope (`~28.3–29.0 TFLOPS`) | **`28.27 TFLOPS`** (`v3` `11.00 FPS`) / **`28.77 TFLOPS`** (`uc` `22.83 FPS`) vs `12.14 T` eager cuDNN | `I_model = 252.7–272.5 >> 141.53` | **Purely Compute & `170W` TGP Bound** |

### 3.2 First-Principles Paradox Resolution Across All 3 Canonical Models

| Canonical Model | Params & FLOPs / Frame | Realized DRAM & AI ($I_{\text{realized}}$) | Position vs. GA106 Knees (`70.76 / 77.65 / 128.36 / 141.53`) | Isolated TRT Latency & FPS (`R3` Meas.) | Realized Compute & Active GDDR6 Util | Physical Governing Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`animevideov3`** (`64f16c` 4x$\to$2x) | `621,424` params **`2.571 TFLOPs`** | `10.162 GB/f` **`252.70 FLOPs/B`** (`286.36` ideal) | **`1.79x–3.57x` Right of Knee** (Deep Compute Regime) | `727.48 ms/b` (`90.93 ms/f`) **`11.00 FPS`** | **`28.27 TFLOPS`** (`170.38W` pk) `mem_util = 44.53%` (`~111.8 GB/s`) | **Tensor-Core Compute + `170W` TGP Wall** (`94.90%` GPU util) |
| **`sharp1-ultracompact`** (`64f8c` Native 2x) | `304,716` params **`1.260 TFLOPs`** | `4.621 GB/f` **`272.45 FLOPs/B`** (`284.72` ideal) | **`1.93x–3.85x` Right of Knee** (Deep Compute Regime) | `350.36 ms/b` (`43.80 ms/f`) **`22.83 FPS`** | **`28.77 TFLOPS`** (`170.93W` pk) `mem_util = 50.70%` (`~105.5 GB/s`) | **Exact Dual-Saturation Equilibrium** (`22.83` TRT $\approx$ `22.70` NVENC `fullres`!) |
| **`sharp1-superultracompact`** (`24f8c` Native 2x) | `45,156` params **`0.1859 TFLOPs`** | `1.966 GB/f` **`94.32 FLOPs/B`** (`104.95` ideal) | Transitional + `1.383x` `C=24->32` tile padding | `178.15 ms/b` (`22.27 ms/f`) **`44.91 FPS`** | `8.35 TFLOPS` (`121.53W` avg) `mem_util = 31.27%` (`enc = 96.36%`) | **100% 7th-Gen NVENC `-multipass fullres` ASIC Bound** (`22.70 FPS` steady) |

#### Why Every Legacy Paradox Disappears:
1. **Why `animevideov3` achieves `28.27 TFLOPS` (`11.00 FPS` isolated TRT) instead of "only 27% of 101.9 TFLOPS":**
   `101.90 TFLOPS` never existed for dense weights on GA106 — it required 2:4 structured sparsity (`50%` zeroed weights). Furthermore, TensorRT's `28.27–28.77 TFLOPS` exceeds 100% of GA106's `HMMA.16816.F32` dense peak (`25.48–27.96 TFLOPS`), proving `HMMA.16816.F16` full-rate Tensor Cores are active (`2.37x` over eager PyTorch cuDNN's `12.140 TFLOPS`), reaching **`~99%` of GA106's physical `N=64` implicit-GEMM silicon capacity under the `170.0 W` TGP ceiling** (`170.38–170.93 W` peak power, throttling sustained clock to `1,701–1,786 MHz`).
2. **Why GDDR6 `mem_util_pct` is `44.53%` (not `100%`) during `animevideov3` Climax:**
   Because `I_realized = 252.70 FLOPs/B >> 141.53 FLOPs/B` (and `>> 77.65 FLOPs/B`), Tensor Cores take `90.93 ms/frame` while transferring `10.162 GB/frame` across the `360.0 GB/s` bus (`302.61 GB/s` STREAM Triad) requires only `28.23–33.58 ms/frame`. Compulsory external bandwidth is $11.00 \times 10.162 = \mathbf{111.78\text{ GB/s}}$ (`31.05%` of `360 GB/s` / `36.94%` of `302.61 GB/s`), which together with `3x3` halo and L2 sector traffic accounts directly for the measured **`44.53%` memory controller utilization**.
3. **Why `sharp1-superultracompact` (`44.91 FPS` TRT) caps near `20.80–24.56 FPS` E2E/Steady:**
   `silicon_probe.json` proved that production **`hevc_nvenc -preset p6 -cq 18 -multipass fullres -rc-lookahead 32`** encodes every 4K 10-bit frame **twice at full `3840x2160` resolution** on GA106's single 7th-Gen NVENC ASIC, setting a hard physical steady-state ceiling of **`22.70 FPS` (`44.053 ms/frame`; `20.67 FPS` over a 240-frame chunk with lookahead flush)**!

---

## 4. End-to-End Pipeline Concurrency & Calibrated Amdahl Modeling

### 4.1 3-Stage Overlapped Concurrency Engine & R3 Micro-Timings

```text
Timeline (Chunk Steps) ────────────────────────────────────────────────────────────────────────►
Stage 1 (DecodeWorker):  [Decode Chunk 0]   [Decode Chunk 1]   [Decode Chunk 2]   [Decode Chunk 3]
Stage 2 (Main GPU Engine):                  [Infer Chunk 0]    [Infer Chunk 1]    [Infer Chunk 2]
Stage 3 (EncodeWorker):                                        [Encode Chunk 0]   [Encode Chunk 1]
```

Measured isolated & pipelined batch latencies (`Batch = 8` frames, `1080p -> 4K`, `silicon_probe.json` & `r3_climax_*_summary.json`):

1. **Stage 1 (`DecodeWorker` on Alder Lake P-Cores / NVDEC):**
   - 1080p H.264 Hi10P software decode on 6 P-Cores: **`~48.2 ms/batch`** (`6.02 ms/frame` = `166.1 FPS`); 8-bit NVDEC: `~8.4 ms/batch`.
2. **Stage 2 (`DoubleBufferedPipeUpscaler` on `stream_h2d`, `stream_compute`, `stream_d2h`):**
   - **Pure PCIe 4.0 x16 Pinned H2D DMA** (`49.77 MB` `uint8` RGB24): **`2.1822 ms/batch`** isolated (`0.2728 ms/frame`, **`22.81 GB/s`** = `72.38%` of PCIe 4.0 x16), overlapped on `stream_h2d` *(note: in-pipeline `avg_h2d_dma_ms` of `715.0 / 335.5 / 66.3 ms` reflects Windows WDDM `HwSchMode=2` queuing the non-blocking Copy-Engine packet behind the active `stream_compute` batch after `evt_h2d_s` stamps on idle `stream_h2d`)*.
   - **Fused R3 `@triton.jit` Input Normalization** (`_nhwc_u8_to_nchw_fp16_norm`, `149.30 MB` 1-pass VRAM): **`0.5133 ms/batch`** isolated (`0.0642 ms/frame`, **`290.88 GB/s`** = `80.80%` of `360 GB/s`, **`3.20x` faster** than legacy 2-kernel PyTorch `1.6429 ms` / `348.36 MB`), and **`0.593–0.724 ms/batch`** in live Climax runs (`avg_in_norm_ms`).
   - **TensorRT 10.16 FP16 Compute (`avg_trt_ms`):** **`727.477 ms/batch`** (`animevideov3`), **`350.358 ms/batch`** (`sharp1-uc`), **`178.148 ms/batch`** (`sharp1-suc`).
   - **Output `@triton.jit` BT.709 RGB $\to$ P010LE** (`597.20 MB` VRAM): **`5.6683 ms/batch`** isolated (`105.36 GB/s`), **`5.504–5.939 ms/batch`** in live Climax runs.
   - **PCIe 4.0 x16 Pinned D2H DMA** (`199.07 MB` 4K `p010le`): **`8.7918 ms/batch`** isolated (`1.0990 ms/frame`, **`22.64 GB/s`** = `71.86%` of PCIe 4.0 x16), **`8.836–9.535 ms/batch`** in compute-bound Climax runs (`14.542 ms` under NVENC backpressure in `sharp1-suc`).
   - **Total Unoverlapped Stage 2 Cycle ($T_2$):** `727.48 + 5.94 + 8.84 =` **`742.25 ms/batch` (`92.78 ms/f` = `10.78 FPS`)** for `v3`; **`365.62 ms/batch` (`45.70 ms/f` = `21.88 FPS`)** for `sharp1-uc`; **`198.19 ms/batch` (`24.77 ms/f` = `40.36 FPS`)** for `sharp1-suc`.
3. **Stage 3 (`EncodeWorker` — 7th-Gen NVENC `hevc_nvenc` Main10 `p6 CQ 18 lookahead 32`):**
   - **Production `-multipass fullres` (2-pass full-4K):** **`44.053 ms/frame` (`352.42 ms/batch` = `22.70 FPS`)** marginal steady-state; **`48.388 ms/frame` (`387.10 ms/batch` = `20.67 FPS`)** per 240-frame chunk including 32-frame lookahead flush (`18.96 FPS` @ 120f).
   - **Single-Pass Reference (`-multipass disabled`):** **`33.895 ms/frame` (`29.50 FPS`)** @ 120f; `~20.0 ms/frame` (`~50.0 FPS`) steady without lookahead flush.

---

### 4.2 Stage Hiding Theorem & Boundary Overhead Qualification

**Theorem 1 (Steady-State Concurrency Hiding for Interior Chunks):**
For interior chunks $k \in [1, K-2]$, pipeline period is $T_{\text{chunk}}(k) = \max(T_1, T_2, T_3)$:
- **`animevideov3`:** $T_2 (742.3\text{ ms}) > T_3 (352.4\text{ ms}) > T_1 (48.2\text{ ms}) \implies$ Decode and NVENC are **100% hidden** behind TensorRT inference (`NVENC slack = 389.8 ms/batch`).
- **`sharp1-ultracompact`:** $T_2 (365.6\text{ ms}) \approx T_3 (352.4\text{--}387.1\text{ ms}) \implies$ **Simultaneous Dual-ASIC Saturation** (`94.24%` GPU + `89.06%` NVENC).
- **`sharp1-superultracompact`:** $T_3 (352.4\text{--}387.1\text{ ms}) > T_2 (198.2\text{ ms}) \implies$ GPU finishes early and experiences **NVENC Backpressure Stall** (`96.36%` NVENC, `64.20%` GPU).

**Static Boundary Floor & Single-Chunk Preview Efficiency:**
Non-overlapped Chunk 0 ramp-up, Chunk $K-1$ lookahead drain, and 1-pass Concat-Mux + `streamhash` audit impose a static boundary floor $T_{\text{boundary}} \approx \mathbf{5.02\text{ s}}$. For a single-chunk 10s preview ($T_{\text{compute}} \approx 16.97\text{ s}$):
$$\eta_{\text{hiding, preview}} = \frac{16.97}{16.97 + 5.02} = \mathbf{77.17\%}, \qquad \eta_{\text{hiding, full\_ep}} = \frac{2,400}{2,400 + 5.02} = \mathbf{99.79\%}$$

---

### 4.3 Piecewise Bounded Amdahl Formulation (Production `fullres` vs. Single-Pass NVENC)

Every output frame (newly inferred or Cyclic Cel memoized in VRAM, $T_{\text{memo}} \approx 0.05\text{ ms}$) must traverse the 7th-Gen NVENC ASIC:
$$T_{\text{eff}}(r_{\text{unique}}) = \max\left(T_{\text{NVENC}}, \; r_{\text{unique}} T_{\text{infer}} + (1 - r_{\text{unique}}) T_{\text{memo}}\right), \qquad S_{\text{max}} = \frac{T_{\text{infer}}}{T_{\text{NVENC}}}, \qquad r_{\text{dedup, knee}} = 1 - \frac{T_{\text{NVENC}}}{T_{\text{infer}}}$$

| Model & Isolated $T_{\text{infer}}$ (R3) | Production `-multipass fullres` ($T_{\text{NVENC}} = \mathbf{44.053\text{ ms}}$, **`22.70 FPS`** steady) | Single-Pass Reference ($T_{\text{NVENC,1p}} \approx 20.0\text{ ms}$, `~50.0 FPS`) |
| :--- | :--- | :--- |
| **`animevideov3`** (`90.935 ms/f`, `11.00 FPS`) | **$S_{\text{max}} = \mathbf{2.06\times}$**, Saturation Knee = **`51.56%` dedup** | $S_{\text{max,1p}} = 4.55\times$, Knee = `78.01%` dedup |
| **`sharp1-ultracompact`** (`43.795 ms/f`, `22.83 FPS`) | **$S_{\text{max}} = \mathbf{1.00\times}$**, Knee = **`0.0%` dedup** *(already at NVENC `fullres` ceiling!)* | $S_{\text{max,1p}} = 2.19\times$, Knee = `54.33%` dedup |
| **`sharp1-superultracompact`** (`22.269 ms/f`, `44.91 FPS`) | **NVENC-Bound at `0%` dedup** ($T_{\text{infer}} < T_{\text{NVENC}}$; `49.5%` GPU idle stall even at `0%` dedup) | $S_{\text{max,1p}} = 1.11\times$, Knee = `10.19%` dedup |

> **Architectural Insight:** Because `AGENTS.md` §4 Invariant #4 mandates Studio Master `-multipass fullres` (`CQ 18`, `preset p6`, `lookahead 32`), the single 7th-Gen NVENC ASIC (`22.70 FPS` marginal steady / `~24.56 FPS` peak on lower-entropy anime cels) is the **true ultimate throughput wall** of the RTX 3060 workstation (`~33–35 min` per 24-min episode).

---

## 5. Workstation Resource Saturation & Headroom Summary (R3 Calibrated)

```text
+----------------------------------------------------------------------------------------------------+
|                      CALIBRATED WORKSTATION RESOURCE SATURATION (R3 CLIMAX)                        |
+----------------------------------------------------------------------------------------------------+
|  GA106 Tensor Cores (v3 / sharp1_uc):  [==================================================>] 98.5% |
|  GA106 Board Power (170.0 W TGP):      [===================================================] 100.5%|
|  7th-Gen NVENC `fullres` (uc / suc):   [================================================>  ] 96.4% |
|  GDDR6 Bandwidth (302.6 GB/s Triad):   [========================>                          ] 50.7% |
|  VRAM Capacity (8,156 / 12,288 MiB):   [=================================>                 ] 66.4% |
|  CPU i5-12600K (Hi10P Decode P-Cores): [===========>                                       ] 23.4% |
|  PCIe 4.0 x16 Bus (22.8 GB/s DMA):     [===>                                               ]  5.2% |
|  WD_BLACK SN7100 NVMe Gen4 (Zero-Disk):[>                                                  ]  0.1% |
+----------------------------------------------------------------------------------------------------+
```

- **VRAM Pareto Point (`batch_size = 8`):** Peak VRAM across all 3 R3 regimes is **`7,425–8,156 MiB`** (`60.4%–66.4%` of `12,288 MiB`), leaving **`4.0–4.7 GiB` free** to guarantee zero OOM across arbitrary aspect ratios while `resolve_trt_profile_max_batch` caps TensorRT tactics to `max_batch <= 2` (`_batch_out_ring` slicing `8 -> 4x2` zero-copy).
- **NVMe Gen4 Cache Routing (`E:\ai-video-upscaler\.cache`):** Routing `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and offline test `TEMP`/`TMP` to Disk 1 (`WD_BLACK SN7100 500GB` Gen4) eliminates Gen3 OS drive (`OCPC MFL-300`) contention completely.

---

## 6. Conclusions

1. **Silicon Roofline Calibrated to Exact GA106 Physics:** Consumer Ampere `GA106` delivers `12.74 TFLOPS` FP32 (`35.38 FLOPs/B` knee), `25.48 / 27.96 TFLOPS` dense `HMMA.16816.F32` (`70.76 / 77.65 FLOPs/B` knee; `23.211 TFLOPS` empirical), and `50.95 TFLOPS` dense `HMMA.16816.F16` (`141.53 FLOPs/B` theoretical / `128.36 FLOPs/B` empirical knee; `38.844 TFLOPS` `4096^3` GEMM and `28.27–28.77 TFLOPS` `N=64` 170W-TGP SR implicit GEMM).
2. **All 3 Regimes Accounted For Without Contradiction:** `animevideov3` (`10.13 FPS` E2E, `11.00 FPS` TRT) saturates Tensor Cores & `170W` TGP; `sharp1-ultracompact` (`18.52 FPS` E2E, `22.83 FPS` TRT) hits the exact dual Tensor-Core + NVENC `fullres` (`22.70 FPS`) equilibrium; `sharp1-superultracompact` (`20.80 FPS` E2E, `44.91 FPS` TRT) saturates the 7th-Gen NVENC `fullres` ASIC at `96.36%` avg / `100%` peak.

---

## 7. Verification & Reproducibility Guide

### A. Self-Contained Python Verification of All GA106 Equations, 3-Model FLOPs/AI, and Knees
```powershell
.\.venv\Scripts\python.exe -c "
pix = 1080 * 1920

# 1. All 3 Canonical Models: Params, Conv MACs/FLOPs, Total FLOPs, DRAM Traffic, & AI
p_v3 = (64*3*3*3 + 64) + 16*(64*64*3*3 + 64) + 17*64 + (48*64*3*3 + 48)
macs_v3 = (9*3*64 + 16*9*64*64 + 9*64*48) * pix
flops_v3_conv = macs_v3 * 2
flops_v3_tot = flops_v3_conv + 17*(64*pix) + (3*4320*7680) + 16*(3*2160*3840)
dram_v3_ideal = (3*pix*2) + 32*(64*pix*2) + (48*pix*2) + (48*pix*2 + 3*pix*2) + (3*2160*3840*2) + (p_v3*2)
dram_v3_real = dram_v3_ideal + 6*(3*4320*7680*2)
assert (p_v3, macs_v3, flops_v3_conv, flops_v3_tot) == (621424, 1283973120000, 2567946240000, 2570699980800)
assert (dram_v3_ideal, dram_v3_real) == (8967489248, 10161882848)
assert round(flops_v3_conv / dram_v3_ideal, 2) == 286.36
assert round(flops_v3_conv / dram_v3_real, 2) == 252.70

p_uc = (64*3*3*3 + 64) + 8*(64*64*3*3 + 64) + 9*64 + (12*64*3*3 + 12)
macs_uc = (9*3*64 + 8*9*64*64 + 9*64*12) * pix
flops_uc_conv = macs_uc * 2
flops_uc_tot = flops_uc_conv + 9*(64*pix) + (3*2160*3840)
dram_uc_ideal = (3*pix*2) + 16*(64*pix*2) + (12*pix*2) + (12*pix*2 + 3*pix*2) + (3*2160*3840*2) + (p_uc*2)
dram_uc_real = dram_uc_ideal + 4*(3*2160*3840*2)
assert (p_uc, macs_uc, flops_uc_conv, flops_uc_tot) == (304716, 629445427200, 1258890854400, 1260110131200)
assert (dram_uc_ideal, dram_uc_real) == (4421524632, 4620590232)
assert round(flops_uc_conv / dram_uc_ideal, 2) == 284.72
assert round(flops_uc_conv / dram_uc_real, 2) == 272.45

p_suc = (24*3*3*3 + 24) + 8*(24*24*3*3 + 24) + 9*24 + (12*24*3*3 + 12)
macs_suc = (9*3*24 + 8*9*24*24 + 9*24*12) * pix
flops_suc_conv = macs_suc * 2
flops_suc_tot = flops_suc_conv + 9*(24*pix) + (3*2160*3840)
dram_suc_ideal = (3*pix*2) + 16*(24*pix*2) + (12*pix*2) + (12*pix*2 + 3*pix*2) + (3*2160*3840*2) + (p_suc*2)
dram_suc_real = dram_suc_ideal + 4*(3*2160*3840*2)
assert (p_suc, macs_suc, flops_suc_conv, flops_suc_tot) == (45156, 92714803200, 185429606400, 185902387200)
assert (dram_suc_ideal, dram_suc_real) == (1766797512, 1965863112)
assert round(flops_suc_conv / dram_suc_ideal, 2) == 104.95
assert round(flops_suc_conv / dram_suc_real, 2) == 94.32

# 2. Consumer Ampere GA106 (28 SMs) Dense Compute Ceilings & Roofline Knees
bw_theo, bw_emp = 360.0, 302.61
fp32_1777 = 28 * 128 * 2 * 1.777       # 12,737.536 GFLOPS (12.74 TFLOPS)
tc_fp32acc_1777 = 28 * 256 * 2 * 1.777 # 25,475.072 GFLOPS (25.48 TFLOPS)
tc_fp32acc_1950 = 28 * 256 * 2 * 1.950 # 27,955.200 GFLOPS (27.96 TFLOPS)
tc_fp16acc_1777 = 28 * 512 * 2 * 1.777 # 50,950.144 GFLOPS (50.95 TFLOPS; 101.90T is 2:4 sparse)
assert round(fp32_1777 / 1000, 2) == 12.74 and round(fp32_1777 / bw_theo, 2) == 35.38
assert round(tc_fp32acc_1777 / 1000, 2) == 25.48 and round(tc_fp32acc_1777 / bw_theo, 2) == 70.76
assert round(tc_fp32acc_1950 / 1000, 2) == 27.96 and round(tc_fp32acc_1950 / bw_theo, 2) == 77.65
assert round(tc_fp16acc_1777 / 1000, 2) == 50.95 and round(tc_fp16acc_1777 / bw_theo, 2) == 141.53
assert round(38844.0 / bw_emp, 2) == 128.36
assert (flops_v3_conv / dram_v3_real) > (tc_fp16acc_1777 / bw_theo) > 77.65

# 3. Empirical R3 Realized TFLOPS, NVENC multipass fullres Ceiling & Amdahl Knees
t_v3_f = 727.477 / 8.0   # 90.9346 ms/f (11.00 FPS -> 28.27 TFLOPS)
t_uc_f = 350.358 / 8.0   # 43.7948 ms/f (22.83 FPS -> 28.77 TFLOPS)
t_nvenc_fullres = 44.053 # 22.70 FPS marginal steady-state
assert round((1000.0 / t_v3_f) * (flops_v3_tot / 1e12), 2) == 28.27
assert round((1000.0 / t_uc_f) * (flops_uc_tot / 1e12), 2) == 28.77
assert round(t_v3_f / t_nvenc_fullres, 2) == 2.06
assert round((1.0 - t_nvenc_fullres / t_v3_f) * 100, 2) == 51.56
assert round((16.9686 / (16.9686 + 5.02)) * 100, 2) == 77.17

print('ALL GA106 SILICON ROOFLINE, 3-MODEL COMPLEXITY & R3 AMDAHL EQUATIONS VERIFIED SUCCESSFULLY!')
"
```

### B. Live Silicon Probe, Offline Test Suite, & Agent Memory Budget Checks
```powershell
.\.venv\Scripts\python.exe scripts/silicon_roofline_probe.py --dry-run
.\.venv\Scripts\python.exe scripts/verify_offline.py
.\.venv\Scripts\python.exe scripts/agent_context.py --check
```
