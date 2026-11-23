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
1. **Zero-Copy Dual-Phase `CV_16UC1` / `CV_8UC1` Detection (`deduplicate_cyclic_frames`):** Resizes contiguous `arr_c[:h_luma, :]` (`CV_16UC1` for `p010le` preserving fractional 10-bit `/ 256.0`, or `CV_8UC1` for `nv12`) via `cv2.INTER_AREA` `(240, 135)` on aligned `(0, 0)` and shifted `(4, 4)` phases without intermediate `msb_u8` allocation, matching candidates when `MAE <= 0.30` and Weber-weighted single-cell `1x1` peak delta on both phases `<= 2.4` LSB plus 16 strided luma+chroma row checks (`0.0%` mouth false positives while capturing `L_inf = 7..32` ringed cels).
2. **Bounded 4K Reuse Without Re-Inference:**
   - **Zero-Disk ([`src/trt_zero_copy_upscaler.py`](../src/trt_zero_copy_upscaler.py)):** `_bound_live_cyclic_memo` applies Belady's Optimal (`OPT`) eviction capped at `max_memo_frames=64` (`<= 1.54 GB` host RAM on `24s` super-chunks).
   - **Standard ([`src/cuda_engine.py`](../src/cuda_engine.py)):** Infers distinct frames once and materializes repeats via 0-byte NTFS hardlinks (`os.link`).

## 5. Perceptual Dedup V2 (`dis-auto` Early-Exit Cascade & `60Hz` Cadence Lock)

Implemented in `src/dedup.py:_is_dis_motion_early_exit`, `_check_8x8_lip_flap_gate`, `PanZoomCadenceTracker`, and `_build_fused_mpdecimate_dis_filtergraph`:
1. **Isolated `[vdec]` / `[vgray]` 1-Pass Split:** `scale=480:270:flags=area,format=gray,split=2[vdec][vgray]` applies temporal `atadenoise` on `[vdec]` (for grained `mpdecimate`) while keeping `[vgray]` raw on `clean`/`light` and attaching edge-preserving `removegrain=m0=3,atadenoise=0a=0.02:0b=0.04:s=5` strictly on `heavy` (`sigma_y >= 0.75`) so `dY = 8` LSB `4x4` dark blinks stay `100%` unattenuated (`thra = 5 < 8`) while `1-3`px film-dust clumps are suppressed.
2. **Weber-Scaled `2x2 + 3x3` Lip-Flap & Micro-Blink Gate:** Evaluates dual-phase `8x8` cells (`32x32` at `1080p`) alongside `2x2` (`8x8` at `1080p`) eye-blink cells with dark-luma Weber scaling (`max(0.50, min(1.0, sqrt(cell_luma / 96.0)))`) to preserve dark `Y=22` blinks while rejecting Gaussian film grain.
3. **1-Second Phase-Locked Camera Pan/Zoom Hysteresis (`PanZoomCadenceTracker`):** Tracks directional camera motion vectors `(tx, ty, s)` across frames (`>= 3` consistent steps, cosine `>= 0.85`) to enter `pan_zoom_lock` (`0.12 px` release floor, `<= 1` missed step), locking `24p` slow pans to uniform `1-1-1-1` cadence without `3/2` judder on `60Hz` displays.

## 6. Integrity Guard & Frame Reconstruction

- **Zero-Disk:** Re-emits cached `p010le` `memoryview` slices to NVENC per `runs` `repeat_counts`.
- **Standard:** Expands exact frame durations via FFmpeg Concat Demuxer (`generate_concat_demuxer_script` + `encode_chunk_reexpanded`).
- **Strict Resume Check:** `load_or_create_plan` rejects (`ValueError`) mismatched `dedup_policy`, `drift_delta`, or `dis_thresh`.
