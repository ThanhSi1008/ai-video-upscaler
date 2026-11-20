# Deduplication, Cyclic Cel Memoization & DIS Optical Flow Architecture

> **Pre-Inference Frame Deduplication, `A->B->A` Cel Reuse & 4K Reconstruction**
> **Primary Contract:** [GEMINI.md](../GEMINI.md) | **Historical Measurements:** [engineering-log.md](archive/reports/engineering-log.md)

## 1. Core Modules

| Module | Responsibility |
| :--- | :--- |
| [`src/dedup.py`](../src/dedup.py) | 1-pass fused `mpdecimate+DIS`, early-exit DIS optical flow, `deduplicate_cyclic_frames` (`A->B->A`), `raw_actions_rle` compression, `run_preflight_worker_cli` |
| [`src/scheduler.py`](../src/scheduler.py) | `BackgroundQueuePreflightManager`: pre-splits GOP `raw_segs` & scans `plan.json` for upcoming `--queue` episodes |
| [`upscale.py`](../upscale.py) | CLI flags (`--dedup-policy`, `--dis-thresh`, `--drift-delta`, `--no-dedup`) and queue orchestration |
| [`shows.yaml`](../shows.yaml) | Per-series SR model & `dedup_policy` matching rules |
| [`tests/test_dedup_policy.py`](../tests/test_dedup_policy.py) | Policy schema, resume rejection, DIS scan, and RLE re-partitioning tests |

## 2. Deduplication Policies (`--dedup-policy`)

Defined in `src/dedup.py:DEDUP_POLICY_PARAMS`:

| Policy | Key Parameters | Best Use Case |
| :--- | :--- | :--- |
| `default` | `hi=768 : lo=320 : frac=0.33` | Pareto balance: drops ~55-62% adjacent duplicates |
| `sensitive-a` | `hi=512 : lo=160 : frac=0.15` | More sensitive to small character/facial motion |
| `sensitive-b` | `hi=384 : lo=128 : frac=0.10` | Low-CPU option for gentle pans/zooms |
| `motion` | `hi=768 : lo=320 : frac=0.33`, phase corr | Adds 4-quadrant phase correlation |
| `drift` / `drift-<f>` | `drift_delta = 0.5-1.0 px` | Cumulative phase drift from last kept frame |
| `dis` / `dis-<f>` | `dis_thresh = 0.30 px` (1080p) | 1-pass DIS optical flow + early-exit for slow pans/zooms/BD masters |
| `sakuga` / `dis-sakuga` | Tighter motion/DIS thresholds | Action-heavy sakuga cuts |
| `off` | Keep 100% of frames | Disables deduplication completely |

## 3. Dynamic GOP Sync, `raw_actions_rle` & Zero-Wait Queue Preflight

1. `split_video_segments` performs stream-copy GOP splitting and reads exact segment frame counts (`get_segments_frame_counts`).
2. `scan_dedup_plan` force-keeps frame `0` of every chunk so every segment has an independent anchor frame.
3. For boundary-independent policies (`default`, `sensitive*`, `motion`, `off`), `plan.json` stores `raw_actions_rle` (~25 KB/episode). When `chunk_frame_counts` differ on resume/preflight, `load_or_create_plan` re-partitions in RAM in `< 1 ms` without re-decoding video.
4. In `--queue` mode, `BackgroundQueuePreflightManager` runs at `BELOW_NORMAL_PRIORITY_CLASS` to pre-split `raw_segs` and scan `plan.json` for upcoming episodes while the GPU processes the current one.

## 4. Cyclic Cel Memoization (`A->B->A`)

Adjacent `mpdecimate` misses alternating anime loops (`A->B->A->C->A` lip flaps, blinking, wind cycles):
1. **Two-Stage Detection (`deduplicate_cyclic_frames`):** Fast `BLAKE2b` 128-bit hash + coarse strided sampling (`24 KB`), validated with OpenCV SIMD `cv2.absdiff` (`max_diff <= 2/255` and `MAE <= 0.20/255`, `PSNR >= 57 dB`).
2. **4K Reuse Without Re-Inference:**
   - **Zero-Disk ([`src/trt_zero_copy_upscaler.py`](../src/trt_zero_copy_upscaler.py)):** Uses Belady's Optimal (`OPT`) eviction to retain only `p010le` frames needed by later batches in RAM.
   - **Standard ([`src/cuda_engine.py`](../src/cuda_engine.py)):** Infers distinct frames once and materializes repeats via 0-byte NTFS hardlinks (`os.link`).

## 5. 1-Pass DIS Optical Flow & Early-Exit Cascade

Implemented in `src/dedup.py:_scan_fused_mpdecimate_and_dis` and `detect_dis_policy_actions`:
1. **1-Pass Split (`[0:v]split=2`):** One FFmpeg decode simultaneously logs `mpdecimate` on `stderr` and streams `480x270` grayscale frames on `stdout`.
2. **Early-Exit Cascade:** `np.array_equal` fast-skips identical frames in `0 ms`, then checks sequentially:
   - Pan/tilt median vector magnitude $\ge \tau$;
   - Radial zoom edge flow $\ge \tau$;
   - Local/dispersed motion ($0.7 \times P_{90}(\|\vec{V}\|) \ge \tau$ or moving area $\ge 5.0\%$).

## 6. Integrity Guard & Frame Reconstruction

- **Zero-Disk:** Re-emits cached `p010le` `memoryview` slices to NVENC per `runs` `repeat_counts`.
- **Standard:** Expands exact frame durations via FFmpeg Concat Demuxer (`generate_concat_demuxer_script` + `encode_chunk_reexpanded`).
- **Strict Resume Check:** `load_or_create_plan` rejects (`ValueError`) mismatched `dedup_policy`, `drift_delta`, or `dis_thresh`.
