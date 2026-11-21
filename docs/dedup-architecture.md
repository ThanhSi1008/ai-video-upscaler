# Deduplication, Cyclic Cel Memoization & DIS Optical Flow Architecture

> **Pre-Inference Frame Deduplication, `A->B->A` Cel Reuse & 4K Reconstruction**
> **Primary Contract:** [GEMINI.md](../GEMINI.md) | **Historical Measurements:** [engineering-log.md](archive/reports/engineering-log.md)

## 1. Core Modules

| Module | Responsibility |
| :--- | :--- |
| [`src/dedup.py`](../src/dedup.py) | 1-pass fused `mpdecimate+DIS` (`dis-auto` V2), `RollingDedupPlanner` (`max_workers=2`), `deduplicate_cyclic_frames` (`A->B->A`), `raw_actions_rle`, `run_preflight_worker_cli` |
| [`src/scheduler.py`](../src/scheduler.py) | `BackgroundQueuePreflightManager`: pre-splits GOP `raw_segs` & scans `plan.json` for upcoming `--queue` episodes |
| [`upscale.py`](../upscale.py) | CLI flags (`--dedup-policy`, `--dis-thresh`, `--drift-delta`, `--no-dedup`) and queue orchestration |
| [`shows.yaml`](../shows.yaml) | Per-series SR model & `dedup_policy` matching rules (`dis-auto` default for BD & Web-DL) |
| [`tests/test_pareto_o1_o6.py`](../tests/test_pareto_o1_o6.py) | Synthetic pan/zoom/lip-flap/grain benchmarks, Open-GOP rolling preflight, and Tier 3 cyclic cel tests |

## 2. Deduplication Policies (`--dedup-policy`)

Defined in `src/dedup.py:DEDUP_POLICY_PARAMS` (mirrored in `web/fs_browser.py`):

| Policy | Key Parameters | Best Use Case |
| :--- | :--- | :--- |
| `default` | `hi=768 : lo=320 : frac=0.33` | Fast `mpdecimate`-only baseline: drops ~55-62% adjacent duplicates |
| `dis-auto` | `dis_thresh=0.30`, `auto_noise_floor=True`, `lip_flap_guard=True` | **Recommended V2 default (`shows.yaml`):** retains 0.25px flat-sky pans, off-center zooms & dark `8x8` lip-flaps while rejecting film grain |
| `sensitive-a` | `hi=512 : lo=160 : frac=0.15` | More sensitive to small character/facial motion |
| `sensitive-b` | `hi=384 : lo=128 : frac=0.10` | Low-CPU option for gentle pans/zooms |
| `motion` | `hi=768 : lo=320 : frac=0.33`, phase corr | Adds 4-quadrant phase correlation |
| `drift` / `drift-<f>` | `drift_delta = 0.5-1.0 px` | Cumulative phase drift from last kept frame |
| `dis` / `dis-<f>` | `dis_thresh = 0.30 px` (1080p) | 1-pass DIS optical flow + early-exit for slow pans/zooms/BD masters |
| `sakuga` / `dis-sakuga` | Tighter motion/DIS thresholds | Action-heavy sakuga cuts |
| `off` | Keep 100% of frames | Disables deduplication completely |

## 3. Dynamic GOP Sync, Rolling Parallel Preflight & `raw_actions_rle`

1. `split_video_segments` performs stream-copy GOP splitting (`24s` super-chunks in CUDA Zero-Disk mode when `-s` is omitted; `8s` default otherwise) and reads exact segment frame counts (`get_segments_frame_counts`).
2. `RollingDedupPlanner` scans `seg_0000..seg_0001` synchronously so GPU inference starts in ~2-6s, while `<= 2` bounded worker threads (`-threads 2`, protecting `DDR4-2133` bandwidth) scan remaining segments in parallel and reconcile Open-GOP RASL `_count_leading_discard_packets` (`"keep"` prepended) so action counts match `seg_frame_counts` 1:1.
3. For boundary-independent policies (`default`, `sensitive*`, `motion`, `off`), `plan.json` stores `raw_actions_rle` (~25 KB/episode). When `chunk_frame_counts` differ on resume/preflight, `load_or_create_plan` re-partitions in RAM in `< 1 ms` without re-decoding video.
4. In `--queue` mode, `BackgroundQueuePreflightManager` runs at `BELOW_NORMAL_PRIORITY_CLASS` to pre-split `raw_segs` and scan `plan.json` for upcoming episodes while the GPU processes the current one.

## 4. Cyclic Cel Memoization (`A->B->A`)

Adjacent `mpdecimate` misses alternating anime loops (`A->B->A->C->A` lip flaps, blinking, wind cycles):
1. **Normalized Multi-Tier Detection (`deduplicate_cyclic_frames`):** Normalizes `p010le` (`uint16 << 6`) and `nv12` to 8-bit luma/planar views, hashes coarse strided samples (`BLAKE2b`), and validates with OpenCV SIMD `cv2.absdiff` (`max_diff <= 2` + Tier 3 isolated B-frame ringing tolerance `<= 0.02%` pixels in `[3..5]`, `MAE <= 0.18`, `16x16` block mean `<= 0.85`).
2. **Bounded 4K Reuse Without Re-Inference:**
   - **Zero-Disk ([`src/trt_zero_copy_upscaler.py`](../src/trt_zero_copy_upscaler.py)):** `_bound_live_cyclic_memo` applies Belady's Optimal (`OPT`) eviction capped at `max_memo_frames=64` (`<= 1.54 GB` host RAM on `24s` super-chunks).
   - **Standard ([`src/cuda_engine.py`](../src/cuda_engine.py)):** Infers distinct frames once and materializes repeats via 0-byte NTFS hardlinks (`os.link`).

## 5. Perceptual Dedup V2 (`dis-auto` Early-Exit Cascade)

Implemented in `src/dedup.py:_is_dis_motion_early_exit`, `_check_8x8_lip_flap_gate`, and `_estimate_textured_lk_pan_zoom`:
1. **1-Pass Split (`[0:v]split=2`):** One FFmpeg decode simultaneously logs `mpdecimate` on `stderr` and streams `480x270` grayscale frames on `stdout`.
2. **`8x8` Dual-Phase Lip-Flap Gate (`32x32` at `1080p`):** Signed `3x3` box-blurred frame difference evaluated over aligned and `(4, 4)`-shifted `8x8` cells against a robust MAD noise floor retains small/dark mouth flaps and eye-blinks while canceling zero-mean grain.
3. **Textured 3-DOF Lucas-Kanade & Noise-Calibrated DIS:** Pre-blurs grain (`GaussianBlur((5,5), 1.1)`), fits 3-DOF affine `(tx, ty, s)` on high-gradient foreground pixels (`>= 48` textured samples) to catch `0.25 px` pans with `72%+` flat sky and off-center zooms, and evaluates DIS vectors restricted to textured masks above an adaptive grain floor.

## 6. Integrity Guard & Frame Reconstruction

- **Zero-Disk:** Re-emits cached `p010le` `memoryview` slices to NVENC per `runs` `repeat_counts`.
- **Standard:** Expands exact frame durations via FFmpeg Concat Demuxer (`generate_concat_demuxer_script` + `encode_chunk_reexpanded`).
- **Strict Resume Check:** `load_or_create_plan` rejects (`ValueError`) mismatched `dedup_policy`, `drift_delta`, or `dis_thresh`.
