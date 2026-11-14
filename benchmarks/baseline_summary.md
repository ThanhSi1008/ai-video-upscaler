# Canonical Baseline Benchmark Summary — Fate Apocrypha Ep 22

> **Document:** `benchmarks/baseline_summary.md`
> **Milestone:** Milestone 2 (Canonical Baseline Setup — R1)
> **Benchmark Date:** 2026-11-14T15:21:20Z
> **Status:** GOLDEN BASELINE ESTABLISHED (Verified via `compare_benchmarks.py` PASS on both Preview and Climax tiers)

---

## 1. System & Environment Specifications

- **Operating System:** Windows 11 Pro 64-bit (100% Local Execution)
- **Host CPU:** Intel Core i5-12600K (10 Cores: 6 P-Cores + 4 E-Cores, 16 Threads, AVX2 enabled, P-Core pinned)
- **Target GPU:** NVIDIA GeForce RTX 3060 12GB GDDR6 (GA106 Ampere, 3584 CUDA Cores, 112 Tensor Cores, P0 Lock 7500 MHz, HAGS enabled)
- **System Memory:** 32 GB DDR4
- **Storage Subsystem:** NVMe PCIe Gen4 High-Speed SSD (`WD_BLACK SN7100`)
- **Software Stack:** Python 3.11.9, TensorRT 10.16.0 FP16 (`HMMA.16816.F16`), PyTorch 2.x CUDA 12.x, FFmpeg 8.1.2-full_build (`hevc_nvenc`)
- **Pipeline Architecture:** 3-Stage Overlapped Zero-Disk Pipe (`Decode N+1` || `GPU N` || `Encode N-1`), Triton JIT `p010le` BT.709 VRAM Kernel, Cyclic Cel Memoization (Belady's OPT).

---

## 2. Canonical Benchmark Sample Profile

- **Source File:** `E:\Films\Raw_Sources\Fate Apocrypha\[AC] Fate Apocrypha - 22 [BD 1080p Hi10 FLAC][DualAudio][695AB63E].mkv`
- **Container Duration:** 1422.09s (23m 42.09s) | **File Size:** 2.12 GB (2,274,671,063 bytes)
- **Video Stream:** 1920x1080 @ 23.976 fps, H.264 High 10 (`yuv420p10le`, 9.71 Mbps video / 12.80 Mbps total container, CPU software decode)
- **Audio Streams:**
  - Track 0: `flac` 2.0 (ENG, 24-bit / 48 kHz, 1.50 Mbps)
  - Track 1: `flac` 2.0 (JPN, 24-bit / 48 kHz, 1.58 Mbps)
- **Subtitle Streams:** 2x ASS (`Signs/Songs` 128 events + `Full Subtitles [UTW/Netflix]` 353 events)
- **Font Attachments:** 5 fonts (MatissePro EB/B, SkipStd B, Profile-Medium / MediumItalic)
- **Chapters:** 9 chapters (Avant, OP, Part A, Part B, Part C, Part D, Part E, ED, Part F)
- **Complexity Rationale:** Ranked #1 in motion entropy across library; high CPU decode load; sakuga line deformation; 95.06% unique frames (only 4.94% dedup) in action climax.

---

## 3. Dual-Tier Baseline Performance Overview

| Benchmark Metric | Tier 1: Preview (Rapid Iteration) | Tier 2: Climax (Sakuga Stress) | Measurement Unit |
| :--- | :--- | :--- | :--- |
| **Run Label** | `baseline_preview` | `baseline_climax` | String |
| **Time Window** | 10.0s @ 30% (`t=424.84s - 438.75s`, 13.91s container) | 60.0s @ 13:30 (`t=808.47s - 870.12s`, 61.65s container) | Start / Duration |
| **Total Wall-Clock Time** | 74.69 | 173.24 | seconds (s) |
| **Effective Pipeline FPS** | 3.61 | 8.53 | frames/second |
| **AI Inference Steady FPS** | 4.88 | 8.90 | frames/second |
| **Total Source Frames** | 270 | 1478 | frames |
| **Output Frames Count** | 270 | 1478 | frames |
| **Unique Kept Frames (AI)** | 224 | 1405 | frames |
| **Dedup Skip Ratio** | 17.04% | 4.94% | percentage (%) |
| **Frame Preservation Status** | `PASS` (`frames_matched=True`) | `PASS` (`frames_matched=True`) | Verdict |
| **Container Mux Audit Status** | `PASS` | `PASS` | Verdict |

---

## 4. Standardized 17-Metric Hardware Telemetry Profile

Continuous hardware metrics sampled at 1.0s intervals during live baseline runs:

| # | Hardware Metric | Preview (10s / 67 samples) Avg / Peak | Climax (60s / 163 samples) Avg / Peak | Specification & State |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **GPU Performance State (`pstate`)** | P0 | P0 | Maximum Performance Mode |
| 2 | **GPU Compute Util (`gpu_util_pct`)** | 34.52% / 100.00% | 73.88% / 100.00% | Tensor Core inference load |
| 3 | **GPU Memory Controller (`mem_util_pct`)** | 27.04% / 79.00% | 39.36% / 60.00% | VRAM memory bus traffic |
| 4 | **NVENC Encoder Util (`enc_util_pct`)** | 12.85% / 100.00% | 36.23% / 100.00% | 10-bit HEVC encoder load |
| 5 | **NVDEC Decoder Util (`dec_util_pct`)** | 0.00% / 0.00% | 0.00% / 0.00% | Hi10P CPU software decode |
| 6 | **GPU Core Clock (`core_clock_mhz`)** | 989.7 MHz / 1972.0 MHz | 1654.8 MHz / 1972.0 MHz | Dynamic Boost Clock |
| 7 | **GPU Memory Clock (`mem_clock_mhz`)** | 4229.9 MHz / 7501.0 MHz | 7040.0 MHz / 7501.0 MHz | P0 Locked Memory Frequency |
| 8 | **GPU Board Power Draw (`power_w`)** | 66.98 W / 170.40 W | 133.48 W / 170.58 W | Board Power (TGP 170W) |
| 9 | **GPU Temperature (`temp_c`)** | 51.1 °C / 66.0 °C | 65.7 °C / 73.0 °C | Maximum Die Temperature |
| 10 | **Dedicated VRAM Used (`vram_used_mib`)** | 6057.9 MiB / 8023.0 MiB | 7416.6 MiB / 8119.0 MiB | Out of 12,288 MiB |
| 11 | **Dedicated VRAM Free (`vram_free_mib`)** | 4092.0 MiB (min headroom) | 3996.0 MiB (min headroom) | Safety Headroom |
| 12 | **Host CPU Util (`cpu_pct`)** | 33.19% / 60.90% | 19.85% / 94.80% | Alder Lake i5-12600K |
| 13 | **System RAM Used (`ram_used_gb`)** | 22.56 GB / 24.93 GB | 24.26 GB / 26.17 GB | Out of 32 GB RAM |
| 14 | **SSD Read Throughput (`ssd_read_mb_s`)** | 0.09 MB/s / 1.88 MB/s | 0.05 MB/s / 1.73 MB/s | Raw input read throughput |
| 15 | **SSD Write Throughput (`ssd_write_mb_s`)**| 1.80 MB/s / 13.36 MB/s | 3.26 MB/s / 20.47 MB/s | Segment MP4 / 4K MKV write |
| 16 | **Sampled Duration (`elapsed_s`)** | 74.58 s | 172.26 s | Telemetry active duration |
| 17 | **Total Samples Collected** | 67 samples | 163 samples | 1.0s interval snapshots |

---

## 5. Video Quality & NVENC Encoding Verification

- **Output Resolution:** 3840x2160 (Exact 2x Super-Resolution 4K UHD)
- **Output Container:** Matroska (`.mkv`)
- **Video Codec & Format:** `hevc_nvenc` Main 10 profile (`p010le`, 10-bit YUV 4:2:0)
- **Color Matrix & Primaries:**
  - Standard Two-Way BT.709 TV Range Matrix applied directly in VRAM via Triton JIT Kernel
  - Output Metadata Flags: `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`
- **Rate Control:** Constant Quality `CQ 18` (`-cq 18 -b:v 0 -maxrate 45M -bufsize 90M`, Preset `p6`, Tune `hq`)
- **Lookahead & AQ:** Multipass `fullres`, Lookahead `32` frames, B-frames `4`, Spatial AQ (`-spatial-aq 1`, `-aq-strength 6`), Temporal AQ (`-temporal-aq 1`)
- **Output Video Metrics:**
  - Preview (10s / 13.91s container): File Size 26.80 MB (28,105,542 bytes), Bitrate 16,165.4 kbps
  - Climax (60s / 61.64s container): File Size 249.11 MB (261,208,663 bytes), Bitrate 33,899.0 kbps
- **Output Sanity Guard:**
  - Preview: $|\Delta Y| = 1.0378 \le 6.0/255$ (`PASS`), Pearson $r = 0.999814 \ge 0.98$ (`PASS`)
  - Climax: $|\Delta Y| = 1.8706 \le 6.0/255$ (`PASS`), Pearson $r = 0.999846 \ge 0.98$ (`PASS`)

---

## 6. Studio-Grade Mux Integrity & Invariant Audit

All invariants audited via `src/mux_audit.py:verify_mux_integrity`:

| Audit Check | Preview (10s) Verdict | Climax (60s) Verdict | Verification Details |
| :--- | :--- | :--- | :--- |
| **Stream Copy Completeness** | `PASS` | `PASS` | 2 Audio, 2 Subtitles, 5 Fonts, 1 Chapter preserved |
| **Track 0 Audio MD5 (ENG)** | `PASS` (`12524002e5795a755b6a1a2ac6b3f29c`) | `PASS` (`b7840791f53a0653c08a9fdaaf9c1f67`) | Bit-exact streamhash match |
| **Track 1 Audio MD5 (JPN)** | `PASS` (`1d2d96ed8d595b5d2b9814c86fa0c13c`) | `PASS` (`12f8a4845d7f67e0e2da1a58298da5a3`) | Bit-exact streamhash match |
| **Subtitle Events Audit** | `PASS` (Full sub: 2 packets matched) | `PASS` (Full sub: 11 packets matched) | Exact packet count match on all ASS tracks |
| **10s Multi-Track Interleave**| `PASS` (2 windows checked) | `PASS` (7 windows checked) | `-max_interleave_delta 0` verified |
| **Multi-Point Seek Volume** | `PASS` ($\Delta \le 0.5\text{ dB}$) | `PASS` ($\Delta = 0.0\text{ dB}$ across 7 sample points) | Volume verified across checkpoints |
| **Frame Preservation** | `PASS` ($\pm 0$ frames) | `PASS` ($\pm 0$ frames) | 270 / 270 (Preview), 1478 / 1478 (Climax) |

---

## 7. Artifact Provenance & Traceability

Generated benchmark artifacts stored in `out/benchmark_baseline/`:

```text
out/benchmark_baseline/
├── baseline_preview_summary.json       # Preview tier aggregated performance & telemetry summary
├── baseline_preview_telemetry.csv      # Preview tier 17-metric continuous time-series CSV (67 samples)
├── baseline_preview.audit.json         # Preview tier forensic stream copy & mux audit report
├── baseline_preview.plan.json          # Preview tier deduplication chunk plan (224/270 frames kept)
├── baseline_preview_4K.mkv             # Preview tier 4K output video (26.80 MB)
├── baseline_preview_4K.audit.json      # Companion mux audit JSON for preview 4K MKV
├── baseline_climax_summary.json        # Climax tier aggregated performance & telemetry summary
├── baseline_climax_telemetry.csv       # Climax tier 17-metric continuous time-series CSV (163 samples)
├── baseline_climax.audit.json          # Climax tier forensic stream copy & mux audit report
├── baseline_climax.plan.json           # Climax tier deduplication chunk plan (1405/1478 frames kept)
├── baseline_climax_4K.mkv              # Climax tier 4K output video (249.11 MB)
└── baseline_climax_4K.audit.json       # Companion mux audit JSON for climax 4K MKV
```

---

## 8. Automated Validation Commands

To validate the canonical baseline against itself or future optimization runs:

```powershell
# 1. Validate Preview Baseline Self-Consistency (returns exit code 0):
.\.venv\Scripts\python.exe scripts/compare_benchmarks.py out/benchmark_baseline/baseline_preview_summary.json out/benchmark_baseline/baseline_preview_summary.json

# 2. Validate Climax Baseline Self-Consistency (returns exit code 0):
.\.venv\Scripts\python.exe scripts/compare_benchmarks.py out/benchmark_baseline/baseline_climax_summary.json out/benchmark_baseline/baseline_climax_summary.json

# 3. Future Optimization Loop Validation (Milestone 3 / Milestone 4):
.\.venv\Scripts\python.exe scripts/compare_benchmarks.py out/benchmark_baseline/baseline_preview_summary.json out/benchmark_opt1/opt1_preview_summary.json --tolerance-pct 5.0 --output-md out/benchmark_opt1/comparison_report.md
```
