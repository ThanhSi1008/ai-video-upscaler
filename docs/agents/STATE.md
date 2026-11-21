# System State & Operational Baseline (STATE)

> **Updated:** 2026-11-22
>
> **Commit Baseline:** `7a7b60c` (`fix(guard,subtrans,cli): close shell-chain, subs-upgrade, telemetry, batch & async-deadline edge cases`)
>
> **Related:** [GEMINI](../../GEMINI.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md)

## 1. Current Baseline Summary

- **Offline Verification Baseline (`2026-11-22`):** **100% PASS** (`861 passed, 6 skipped, 0 failed, 0 errors` out of `867` tests via 6-worker sharded `scripts/verify_offline.py` (`E:\ai-video-upscaler\scratch\agent-validation-gmnr3pcf`) with `sitecustomize.py` subprocess socket guard & `UPSCALER_HEVC_ENCODER=libx265`; `smoke_test_pipeline.py` = `PASS` in `0.51s` incl. `mux_segments_directly`; `ruff check .` = `0` errors; `agent_context.py --check` = `PASS`).
- **Pareto Visual-Quality & E2E Speed Architecture (`O1`–`O6`, `7a7b60c`+):**
  - **`O1` Analytical Catmull-Rom 4x->2x Collapse (`_aa2x`):** Collapsed `SRVGGNetCompact` `PixelShuffle(4) + Nearest(4x) + AA Bicubic(0.5x, B=0.0, C=0.5)` into 1080p `Conv2d(48->12, groups=3) + Conv2d(3->12, groups=3) + PixelShuffle(2)` (`Slice+Concat` replicate border), eliminating 8K intermediate tensors (`animevideov3` 60s preview: **`10.91 fps` E2E** vs `10.13 fps` R3, `avg_trt_ms` `677.14 ms` vs `727.48 ms/batch`, **`4,953 MiB`** active VRAM vs `8,156 MiB` peak).
  - **`O2` 10-Bit End-to-End Semi-Planar Pipe:** Added `extract_frames_raw_yuv420` (`p010le` / `nv12`), `@triton.jit` `_yuv420_to_nchw_fp16_bt709_kernel` (`left` chroma), upgraded `_rgb_to_p010le_triton_kernel` (`left` `[1,2,1]/4 x [1,1]/2` chroma + spatial IGN `±0.499 LSB` dither -> `|ΔY| = 0.258`, Pearson `r = 0.999936`), normalized `p010le` MSBs in `deduplicate_cyclic_frames`, and capped `memo_p010 <= 64` frames (`_bound_live_cyclic_memo`).
  - **`O3` & `O6` `24s` Super-Chunks, Bounded Rolling Preflight & In-Process PyAV NVENC:** Auto-promoted omitted `-s` to `24s` in CUDA Zero-Disk mode, added `RollingDedupPlanner` (`max_workers=2`, `-threads 2`, Open-GOP RASL `_count_leading_discard_packets` reconciliation), `_InProcessPyAVNVENCEncoder` (`stream.bit_rate = 0`, monotonic `AVFrame` allocation), and per-episode `--nvenc-multipass auto|fullres|qres` (preserving byte-identical HEVC VPS/SPS/PPS `extradata` across `-f concat -c copy`).
  - **`O4`, `O5` & Web Sync:** Added `dis-auto` (`8x8` dual-phase lip-flap guard + 3-DOF textured Lucas-Kanade + noise-calibrated DIS), updated `shows.yaml` (`sharp1-ultracompact` + `dis-auto` BD -> **`18.06 fps` E2E / `24.44 fps` chunk peak**, `4,649 MiB` active VRAM; `ultracompact` + `dis-auto` Web-DL), added explicit `--model auto-pareto` (`compact` 16c + `ultracompact` 8c, **`16.33 fps` E2E**, `6,425 MiB` active VRAM), and synced pure-stdlib Web Studio mirrors (`web/fs_browser.py`, `web/app.py`, `web/job_manager.py`).
- **Edge-Case Remediation Pass (`c9aa756`+):** Per-segment shell-chain guard enforcement (`_split_command_chains` in `.agents/plugins/upscaler-guard/hooks/pre_tool_guard.py`), `TranslationCache` column model/backend hydration & Cascade `(repair)` provenance retention under `--subs-upgrade` (`subtrans/cache.py`, `subtrans/pipeline.py`), Windows 17-column `TelemetryCollector` CSV parsing/reporting (`src/night_report.py`), call-time path resolution + `--source-dir`/`-o`/`--state-file`/`--dry-run` CLI flags (`scripts/batch_translate_all.py`), and `compute_pending_subs_reserved_seconds` `--async-subs` deadline reservation fix (`upscale.py`).
- **4-Batch Audit Remediation (`f5352e3`, `52dcb5b`, `ecec85e` + Web/CLI/Ruff Batch 4):**
  - **Batch 1 — Test Isolation & Harness (`f5352e3`):** Subprocess-wide `sitecustomize.py` socket guard, CPU `libx265` E2E isolation, bounded `<=280`L `read_file`/`view_file` Context Firewall (`>350`L files), `benchmarks/subtrans/*.py` `--dry-run` + quota guard, scoped `@pytest.mark.subtrans`, and 6-worker LPT sharding.
  - **Batch 2 — Video Pipeline & Season Routing (`52dcb5b`):** Stale `*.tmp.mp4`/`.tmp_*` purge on resume across all 4 enumerators, `src/show_routing.py` safe season/volume/disc grandparent matching, Open-GOP `_count_leading_discard_packets` fix, `cudaEventSynchronize` before `_out_ring` reuse, `RAMDisk` free-space verification, and stat-cached YAML/glossary loaders.
  - **Batch 3 — Subtrans Cache, Sticky Keys & Dialogue (`ecec85e`):** Deterministic cache fingerprints (`cast_sheet_hash` + `genre`), per-episode sticky key selection (with `<=5 RPM` non-explicit-cache overflow), `include_static_cast=False` prompt deduplication, `KnowledgeBase` production auto-seeding, `Cue.speaker` preservation, 2-speaker Dual Dialogue cap, and thread-safe SQLite lock guards.
  - **Batch 4 — Web Security, Scheduler/Env Parity, Ruff & Real-SUT Tests:** `web_gui.py --mock` auto `--no-browser` & simulated job launches, `docs_url=None`, UNC path + non-loopback `output_dir`/masked-key guards, `subtrans/.env` multi-token key sync, per-job `logs/web_job_<id>.log`, unified `compute_overnight_schedule` / `compute_schedule_preview` overhead, dynamic `--dedup-policy` (`dis-<float>`/`drift-<float>`), PyAV audio MD5, `ruff.toml` `B023`/`F601`/`F811` (29 fixes), and real-SUT DIS/RAMDisk/track-default tests.
- **Pass-2 & Pass-1 Baselines (`b996a09`, `e117af6`):** Generated local untracked `requirements.lock` snapshot, pruned `GEMINI.md` (27L) and `modern-web-guidance` (209 -> 39 guides), added glob rules, `scripts/smoke_test_pipeline.py`, `web_gui.py --mock`, and archived legacy docs (`docs/archive/`).
- **Hardware Calibration & 3-Regime Silicon Baseline (`Fate/Apocrypha Ep 22`, RTX 3060 12GB `GA106`):**
  - Silicon ceilings: `cublasGemmEx(16F)` = `38.84 TFLOPS`, GDDR6 STREAM Triad = `302.61 GB/s`, Triton input norm = `0.513 ms/batch` (`290.88 GB/s`), 7th-Gen NVENC 4K Main10 `-multipass fullres` = `22.70 fps` (`44.05 ms/frame`).
  - Canonical regimes: `animevideov3` (`10.13 fps` E2E / `11.00 fps` TRT, 170W TGP bound) | `animejanai-hd-v3-sharp1-ultracompact` (`18.52 fps` E2E / `22.83 fps` TRT) | `animejanai-hd-v3-sharp1-superultracompact` (`20.80 fps` E2E / `44.91 fps` TRT, NVENC-bound).

## 2. Verified Subsystems

| Subsystem | Key Modules | Status | Summary |
| :--- | :--- | :--- | :--- |
| **E-Drive 100% Isolation** | `src/env_isolation.py`, `.venv/`, `*.bat`, `scripts/verify_offline.py` | PASS | Isolates Python 3.11.9, FFmpeg, Ruff, `TEMP`, and compiler caches on `E:`; `--target` / `--impacted` fast snapshotting |
| **Real-Time Web Studio** | `web_gui.py`, `web/`, `tests/test_web_gui.py` | PASS | FastAPI + SSE 1Hz, `--mock` 0-based chunks simulator, `--lan` RO + `--lan-control`, strict CSP `sha256`, 0 VRAM imports |
| **PyAV & 1-Pass Concat-Mux** | `src/ffmpeg_ops.py`, `src/mux_audit.py` | PASS | PyAV 18.1.0 probing, `-max_interleave_delta 0`, parallel `-f streamhash` MD5 audit |
| **Triton Kernels & Zero-Disk TRT** | `src/trt_zero_copy_upscaler.py`, `src/cuda_engine.py`, `src/pipeline.py` | PASS | `--zero-disk=True` default, pinned `NHWC`, `@triton.jit` norm & `p010le`, 2-slot VRAM rings + `.release()`, 3-stage overlap |
| **Spandrel Native 2x & Whisper** | `src/cuda_engine.py`, `src/models_resolver.py`, `src/subtitles.py` | PASS | Spandrel `0.4.2`, SHA-256 weight verification, PID-aware GPU lock + CPU int8 fallback `faster-whisper` `large-v3-turbo` |
| **Dedup, GOP-RLE & Queue** | `src/dedup.py`, `src/scheduler.py`, `src/config.py`, `src/logger.py` | PASS | 1-pass `mpdecimate+DIS`, Cyclic Cel, `<1ms` `raw_actions_rle`, re-entrant GPU lock, `SW_HIDE` thermal query |
| **Subtrans V2.5 Engine** | `subtrans/*.py`, `subtrans/prompts.py`, `glossaries/*.yaml` | PASS | Verbatim prompt module, 3-key rotation, >4K context cache, MeCab `fugashi`, Linter V2, Flash Cascade, RAG, shared DTW DP |
| **Agent Tooling & Guards** | `scripts/{agent_context,export_review_bundle}.py`, `.agents/`, `tests/test_architecture_boundaries.py` | PASS | Compact `GEMINI.md` (27L), glob rules, 3 subagents (`inheritMcp: false`, `enableWriteTools: true` solely for `run_command`), Context/Rewrite Firewall hooks, 6 atomic review bundles (incl. unnumbered `focus_full.txt`) |

## 3. Operational Caveats
- **Daytime Gemini Quota Guard:** Never invoke live Gemini API calls (including `subtrans.py check`/`models`) from `14:00` until overnight runs; use `--dry-run` or stubs.
- **Single GPU Owner:** Only one GPU process may hold `.gpu_active.lock` at a time; never load local LLMs concurrently.
- **Rejected Architecture (ADR-010):** Cross-episode OP/ED video caching remains permanently rejected.
