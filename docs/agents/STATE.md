# System State & Operational Baseline (STATE)

> **Updated:** 2026-11-21
>
> **Commit Baseline:** `d702fac` (`fix(all): resolve verified P0/P1/P2 audit findings across video, subtrans, scripts, and web studio`)
>
> **Related:** [GEMINI](../../GEMINI.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md)

## 1. Current Baseline Summary

- **Offline Verification Baseline (`2026-11-21`):** **100% PASS** (`805 passed, 14 skipped, 1 deselected`, `0` failed in `279.95s`, `Source files: 275` vs `445` pre-cleanup; `ruff check .` = `0` errors; `agent_context.py --check` = `PASS`; `git diff --check` = `0` errors) via `scripts/verify_offline.py` (`E:\ai-video-upscaler\scratch\agent-validation-sqn0fqv3`).
- **Antigravity-Exclusive Cleanup & Context Optimization (2026-11-21):**
  - Consolidated root instructions into single compact `GEMINI.md` (<120 lines), removed `CLAUDE.md`, `AGENTS.md`, `subtrans/AGENTS.md`, and `docs/agents/START_HERE.md`, and migrated Subtrans rules into lazy skill `.agents/skills/subtrans-anime-ops/SKILL.md`.
  - Tightened `upscaler-mcp` tool schemas and added 4000-char / 80-line payload truncation; compacted `upscaler-guard` hooks; pruned `modern-web-guidance` from 209 to 39 local dashboard guides.
  - Archived 20 dated tasks (`docs/archive/tasks/`) and 7 historical reports (`docs/archive/reports/`), added `.antigravityignore`, relocated `unlock_rtx3060_admin.bat` -> `scripts/`, `embed_fate_subtitles.py` & `organize_vietnamese_subs.py` -> `scripts/oneoff/`, and `pending_subs.json` -> `state/pending_subs.json` (with fallback), converted 3 Git `120000` glossary symlinks to `100644` text pointers, and organized tests into `tests/` (unit), `tests/e2e/`, and `subtrans/tests/` with `pytest.ini` markers.
- **Hardware Calibration & 3-Regime Silicon Baseline (`Fate/Apocrypha Ep 22`, RTX 3060 12GB `GA106`):**
  - Silicon ceilings: `cublasGemmEx(16F)` = `38.84 TFLOPS`, GDDR6 STREAM Triad = `302.61 GB/s`, Triton input norm = `0.513 ms/batch` (`290.88 GB/s`), 7th-Gen NVENC 4K Main10 `-multipass fullres` = `22.70 fps` (`44.05 ms/frame`).
  - Canonical regimes: `animevideov3` (`10.13 fps` E2E / `11.00 fps` TRT, 170W TGP bound) | `animejanai-hd-v3-sharp1-ultracompact` (`18.52 fps` E2E / `22.83 fps` TRT) | `animejanai-hd-v3-sharp1-superultracompact` (`20.80 fps` E2E / `44.91 fps` TRT, NVENC-bound).

## 2. Verified Subsystems

| Subsystem | Key Modules | Status | Summary |
| :--- | :--- | :--- | :--- |
| **E-Drive 100% Isolation** | `src/env_isolation.py`, `.venv/`, `*.bat`, `scripts/verify_offline.py` | PASS | Isolates Python 3.11.9, FFmpeg, Ruff, `TEMP`, and all AI/compiler caches onto `E:` |
| **Real-Time Web Studio** | `web_gui.py`, `web/`, `tests/test_web_gui.py` | PASS | FastAPI + SSE 1Hz, `--lan` read-only default + `--lan-control`, strict CSP `sha256`, 0 `innerHTML`, 1080p vs 4K compare |
| **PyAV & 1-Pass Concat-Mux** | `src/ffmpeg_ops.py`, `src/mux_audit.py` | PASS | PyAV 18.1.0 probing, `-max_interleave_delta 0`, parallel `-f streamhash` MD5 audit |
| **Triton Kernels & Zero-Disk TRT** | `src/trt_zero_copy_upscaler.py`, `src/build_trt_engine.py`, `src/pipeline.py` | PASS | `--zero-disk=True` default, pinned `NHWC`, `@triton.jit` norm & `p010le`, 2-slot VRAM rings, 3-stage overlap |
| **Spandrel Native 2x & Whisper** | `src/cuda_engine.py`, `src/models_resolver.py`, `src/subtitles.py` | PASS | Spandrel `0.4.2`, SHA-256 weight verification, `faster-whisper` `large-v3-turbo` 5.1 FC + Silero VAD |
| **Dedup, GOP-RLE & Queue** | `src/dedup.py`, `src/scheduler.py`, `shows.yaml` | PASS | 1-pass `mpdecimate+DIS`, Cyclic Cel (`A->B->A`), `<1ms` `raw_actions_rle`, Unique-FPS ETA |
| **Subtrans V2.5 Engine** | `subtrans/*.py`, `glossaries/*.yaml` | PASS | Gemini 3-key rotation, >4K context cache, MeCab `fugashi`, Linter V2, Flash Cascade, FTS5 2/3-gram RAG, DTW align |
| **Agent Tooling & Guards** | `scripts/agent_context.py`, `.agents/` | PASS | Compact `GEMINI.md` check, AST lookup, `upscaler-guard` hooks, bounded `upscaler-mcp` |

## 3. Operational Caveats
- **Daytime Gemini Quota Guard:** Never invoke live Gemini API calls (including `subtrans.py check`/`models`) from `14:00` until overnight runs; use `--dry-run` or stubs.
- **Single GPU Owner:** Only one GPU process may hold `.gpu_active.lock` at a time; never load local LLMs concurrently.
- **Rejected Architecture (ADR-010):** Cross-episode OP/ED video caching remains permanently rejected.
