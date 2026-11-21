# System State & Operational Baseline (STATE)

> **Updated:** 2026-11-21
>
> **Commit Baseline:** `52dcb5b` (`fix(video): harden resume isolation, season show routing, open-gop & ring sync`)
>
> **Related:** [GEMINI](../../GEMINI.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md)

## 1. Current Baseline Summary

- **Offline Verification Baseline (`2026-11-21`):** **100% PASS** (`835 passed, 11 skipped, 0 failed/errors` out of `846` tests, `43.39s` via 6-worker sharded `scripts/verify_offline.py` (`E:\ai-video-upscaler\scratch\agent-validation-ata3fuzg`) with `sitecustomize.py` subprocess socket guard & `UPSCALER_HEVC_ENCODER=libx265`; `ruff check .` = `0` errors; `agent_context.py --check` = `PASS`; `git diff --check` = `0` errors).
- **Unified Stabilization, Context Tooling & Harness Upgrade (`2026-11-21`):**
  - **Tiered AST Context & Bundling (`scripts/`):** Upgraded `agent_context.py` with `_first_meaningful_doc_line`, `BaseModel`/`@dataclass` fields, depth-2 nested AST & JS/CSS discovery, batched `--symbols`/`--pack`, bounded `--slice`, 2-hop + 8 `CROSS_BOUNDARY_CONTRACTS` `--impact`, same-commit `STATE.md` freshness verification, and independent `pre-commit` + `post-commit` hooks; added `scripts/export_review_bundle.py --sync-latest` writing 6 bundles (`latest_diff.txt`, 4 prefix-cache-ordered domain `focus_*.txt`, and deduplicated unnumbered `focus_full.txt` via `--focus full`) atomically in `scratch/review_bundles/`; added `--target`, `--impacted`, and sharded `--workers` to `scripts/verify_offline.py`.
  - **Guardrails & MCP (`upscaler-guard`, `upscaler-mcp`, `repo-scout`):** Allowed local `.env` reads (`write/edit` = `force_ask`, `git add/commit` = `deny`), enforced Context/Rewrite Firewall (`scratch/review_bundles/` read `deny`, `>250`L overwrite `deny`, bounded `<=280`L `read_file`/`view_file` & capped shell reads for `>350`L files), extended quota guard to `benchmarks/subtrans/*.py` (`--dry-run` supported), `.venv`-first Windows `cmd /c` hook & MCP dispatch, `.json`/`.yaml` syntax audit in `stop_audit.py`, `out/.temp_*` + `STATE` freshness checks, workspace-relative `upscaler_mcp.py`, and renamed local `.agents/skills/modern-web-guidance` frontmatter to `local-web-guidance`.
  - **Code Quality & Cleanup:** Unified `is_valid_dedup_policy` in `web/fs_browser.py` and fixed 0-based mock `chunks` in `web/app.py`; added `DoubleBufferedPipeUpscaler.release()` and removed dead `_compiled_fn`; consolidated `src/config.py` queue re-exports + re-entrant GPU lock; hardened `src/logger.py` thermal `SW_HIDE`/2s timeout + `latest.log` mirroring; added PID-aware GPU lock fallback in `src/subtitles.py`; extracted `subtrans/prompts.py` verbatim; factored DP forward pass in `subtrans/dtw_auto_align.py`; purged legacy `# -*- coding` / banners across 64 runtime `.py` + 6 `.js` files and cleaned docstrings/comments to English.
  - **Boundary & Harness Tests:** Added `tests/test_architecture_boundaries.py` & `tests/test_export_review_bundle.py`; expanded `tests/test_{agent_context,agent_setup,verify_offline,web_gui,zero_disk_ring_buffers_and_cel}.py`.
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
