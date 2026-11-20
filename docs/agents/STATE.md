# System State & Operational Baseline (STATE)

> **Updated:** 2026-11-21
>
> **Commit Baseline:** `b996a09` (`feat(agent): add Antigravity scoped rules, isolated subagents, lifecycle hooks, smoke pipeline, and web mock mode`)
>
> **Related:** [GEMINI](../../GEMINI.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md)

## 1. Current Baseline Summary

- **Offline Verification Baseline (`2026-11-21`):** **100% PASS** (`810 passed, 14 skipped, 0 failed/errors` out of `824` executed tests, `264.94s`, `Source files: 284`; `ruff check .` = `0` errors; `smoke_test_pipeline.py` = `PASS` in `0.40s`; `agent_context.py --check` = `PASS`; `git diff --check` = `0` errors) via `scripts/verify_offline.py` (`E:\ai-video-upscaler\scratch\agent-validation-_9tdf66l`).
- **Pass-2 Antigravity-Native Architecture & Fast Verification (2026-11-21):**
  - Exported local `requirements.lock` (`pip freeze`, ignored in Git/Antigravity) and tagged `pass1-clean-95621e7`; trimmed root `GEMINI.md` to 27 lines / 3,451 bytes (`<= 45 lines / 4,500 bytes`).
  - Added flat glob-scoped rules in `.agents/rules/` (`video-gpu.md`, `subtrans.md`, `web-ui.md`) and context-isolated read-only custom subagents in `.agents/agents/` (`test-verifier.md`, `subtrans-evaluator.md` with `inheritMcp: false`, `enableWriteTools: false`); enforced rule/subagent validation in `scripts/agent_context.py`.
  - Upgraded `upscaler-guard`: hardened `pre_tool_guard.py` (`pip`/`uv`/`conda`, `build_trt_engine.py`, unflagged `upscale.py`, live Gemini APIs), added `post_tool_lint.py` (`PostToolUse` <1.5s Ruff fix), and added `stop_audit.py` (`Stop` <2s git/Ruff/memory check).
  - Added `scripts/smoke_test_pipeline.py` (<5s CPU-safe 24f 10-bit dedup/mux/audit pipeline) + `@pytest.mark.smoke`, `web_gui.py --mock` (`create_app(..., mock_mode=True)` + SSE/job mocks), and `pytest.ini` strict markers (`web`, `smoke`, `addopts`).
- **Pass-1 Cleanup Baseline (`e117af6`):** Consolidated root `GEMINI.md`, removed `CLAUDE.md`/`AGENTS.md`/`subtrans/AGENTS.md`/`START_HERE.md`, migrated Subtrans rules to `subtrans-anime-ops`, compacted `upscaler-mcp`, pruned `modern-web-guidance` (209 -> 39 guides), archived legacy tasks/reports (`docs/archive/`), converted glossary symlinks to text pointers, and organized tests (`tests/`, `tests/e2e/`, `subtrans/tests/`).
- **Hardware Calibration & 3-Regime Silicon Baseline (`Fate/Apocrypha Ep 22`, RTX 3060 12GB `GA106`):**
  - Silicon ceilings: `cublasGemmEx(16F)` = `38.84 TFLOPS`, GDDR6 STREAM Triad = `302.61 GB/s`, Triton input norm = `0.513 ms/batch` (`290.88 GB/s`), 7th-Gen NVENC 4K Main10 `-multipass fullres` = `22.70 fps` (`44.05 ms/frame`).
  - Canonical regimes: `animevideov3` (`10.13 fps` E2E / `11.00 fps` TRT, 170W TGP bound) | `animejanai-hd-v3-sharp1-ultracompact` (`18.52 fps` E2E / `22.83 fps` TRT) | `animejanai-hd-v3-sharp1-superultracompact` (`20.80 fps` E2E / `44.91 fps` TRT, NVENC-bound).

## 2. Verified Subsystems

| Subsystem | Key Modules | Status | Summary |
| :--- | :--- | :--- | :--- |
| **E-Drive 100% Isolation** | `src/env_isolation.py`, `.venv/`, `*.bat`, `scripts/verify_offline.py` | PASS | Isolates Python 3.11.9, FFmpeg, Ruff, `TEMP`, and all AI/compiler caches onto `E:` |
| **Real-Time Web Studio** | `web_gui.py`, `web/`, `tests/test_web_gui.py` | PASS | FastAPI + SSE 1Hz, `--mock` deterministic simulator, `--lan` read-only default + `--lan-control`, strict CSP `sha256`, 0 `innerHTML`, 1080p vs 4K compare |
| **PyAV & 1-Pass Concat-Mux** | `src/ffmpeg_ops.py`, `src/mux_audit.py` | PASS | PyAV 18.1.0 probing, `-max_interleave_delta 0`, parallel `-f streamhash` MD5 audit |
| **Triton Kernels & Zero-Disk TRT** | `src/trt_zero_copy_upscaler.py`, `src/build_trt_engine.py`, `src/pipeline.py` | PASS | `--zero-disk=True` default, pinned `NHWC`, `@triton.jit` norm & `p010le`, 2-slot VRAM rings, 3-stage overlap |
| **Spandrel Native 2x & Whisper** | `src/cuda_engine.py`, `src/models_resolver.py`, `src/subtitles.py` | PASS | Spandrel `0.4.2`, SHA-256 weight verification, `faster-whisper` `large-v3-turbo` 5.1 FC + Silero VAD |
| **Dedup, GOP-RLE & Queue** | `src/dedup.py`, `src/scheduler.py`, `shows.yaml` | PASS | 1-pass `mpdecimate+DIS`, Cyclic Cel (`A->B->A`), `<1ms` `raw_actions_rle`, Unique-FPS ETA |
| **Subtrans V2.5 Engine** | `subtrans/*.py`, `glossaries/*.yaml` | PASS | Gemini 3-key rotation, >4K context cache, MeCab `fugashi`, Linter V2, Flash Cascade, FTS5 2/3-gram RAG, DTW align |
| **Agent Tooling & Guards** | `scripts/agent_context.py`, `scripts/smoke_test_pipeline.py`, `.agents/` | PASS | Compact `GEMINI.md` (27L), glob rules, `test-verifier`/`subtrans-evaluator` (`inheritMcp: false`), `PreToolUse`/`PostToolUse`/`Stop` hooks, <1s smoke pipeline |

## 3. Operational Caveats
- **Daytime Gemini Quota Guard:** Never invoke live Gemini API calls (including `subtrans.py check`/`models`) from `14:00` until overnight runs; use `--dry-run` or stubs.
- **Single GPU Owner:** Only one GPU process may hold `.gpu_active.lock` at a time; never load local LLMs concurrently.
- **Rejected Architecture (ADR-010):** Cross-episode OP/ED video caching remains permanently rejected.
