# Validation & Testing Guide

> **Updated:** 2026-11-24 | **Related:** [GEMINI](../../GEMINI.md) | [STATE](STATE.md) | [PROJECT_MAP](PROJECT_MAP.md)

## Default Offline Verification Sequence (PowerShell)

Always invoke the repository `.venv` interpreter from the repo root (system `PATH` Python may lack dependencies):

```powershell
.\.venv\Scripts\python.exe scripts/agent_context.py --check
.\.venv\Scripts\python.exe scripts/verify_offline.py
.\.venv\Scripts\ruff.exe check .
git diff --check
```

- `verify_offline.py` copies tracked/untracked source/config/fixtures into a disposable snapshot (`scratch/agent-validation-*/repo`), hides CUDA (`CUDA_VISIBLE_DEVICES=-1`), caps BLAS/OMP threads (`OMP/MKL/OPENBLAS/NUMEXPR=1`) with Win11 P-core QoS (`_configure_win32_pcore_qos`), forces CPU software HEVC encoding (`UPSCALER_HEVC_ENCODER=libx265`), strips API keys/tokens, disables `.env`, blocks external Python sockets across the entire subprocess tree via `sitecustomize.py` + `agent_offline_guard` (preserving loopback/socketpair for local ASGI/EventLoop tests), and executes calibrated 6-worker LPT `pytest -m "not gpu"` (`--show-capture=no`, `~55–70s` wall time, `1051 passed, 6 skipped, 0 xfailed, 0 xpassed`).
- Fast iteration modes (`< 5s`, always clean up `repo/` snapshot even on failure): `scripts/verify_offline.py --target <PATH_OR_NODEID>` and `scripts/verify_offline.py --impacted` (auto-resolving dirty paths via `agent_context.py` 2-hop + cross-boundary contracts, `--maxfail=3` default). Use `agent_context.py --pack`, `--symbols`, `--slice`, and `--impact` to inspect code cleanly before testing.
- Logs (`pytest.log`, `junit.xml`) remain under `scratch/agent-validation-*/` (up to 2 recent passing runs retained); snapshot `repo/` and `tmp/pytest/` are deleted automatically on pass.

## Why Bare `pytest` on Live Repo Is Prohibited

1. Un-isolated tests calling `acquire_gpu_lock()` without monkeypatching `GPU_LOCK_FILE` contend on live repo `.gpu_active.lock` (`_atexit_release_gpu_lock` in `src/config.py:84-190` is ownership-gated via `_GPU_LOCK_OWNED`/`_GPU_LOCK_DEPTH` and `is_gpu_lock_held_by_other()`, so it will not delete another live PID's lock, but un-isolated tests can still race or fail on repo lock contention; `tests/conftest.py` `_isolate_logs_dir` already isolates `UPSCALER_LOGS_DIR` per test, while bare `pytest` remains banned for lock/DB/CUDA/API hazards).
2. Default `TranslationCache()` / `KnowledgeBase()` calls can write to production SQLite DBs in `subtrans/.cache/`.
3. Some tests without `@pytest.mark.gpu` auto-select CUDA if visible.
4. `subtrans.py check` / `models` hit real Gemini endpoints.

## Module -> Targeted Test Command

Run targeted tests directly ONLY for pure stdlib/web/smoke allowlisted suites, or inside a disposable snapshot via `scripts/verify_offline.py --target` / `--impacted` for runtime modules (`src/`, `subtrans/`, `tests/e2e/`):

| Subsystem / Module | Targeted Command |
| :--- | :--- |
| **Agent Context & Setup** (`scripts/agent_context.py`, `.agents/`) | `.\.venv\Scripts\python.exe -m pytest tests/test_agent_context.py tests/test_agent_setup.py tests/test_verify_offline.py tests/test_architecture_boundaries.py tests/test_export_review_bundle.py -q` |
| **Fast Video/Mux Smoke** (`< 5s` CPU-safe 24f pipeline) | `.\.venv\Scripts\python.exe scripts/smoke_test_pipeline.py` *(or `.\.venv\Scripts\python.exe -m pytest -m smoke`)* |
| **Web Studio Offline & Mock** (`web/`, `web_gui.py --mock`) | `.\.venv\Scripts\python.exe -m pytest tests/test_web_gui.py tests/test_architecture_boundaries.py -q` *(`79` tests in `test_web_gui.py` + `5` AST/zero-VRAM/Night-Report tests in `test_architecture_boundaries.py` forbidding `src.logger` in `web/`; or `.\.venv\Scripts\python.exe web_gui.py --mock` on `127.0.0.1`)* |
| **Impacted / Targeted Snapshot** (`< 5s` `src/` & `subtrans/`) | `.\.venv\Scripts\python.exe scripts/verify_offline.py --impacted` *(or `--target <PATH_OR_NODEID>`)* |
| **Full Offline Suite** (All non-GPU modules, final gate) | `.\.venv\Scripts\python.exe scripts/verify_offline.py` |
| **E2E Acceptance Suite** (`tests/test_e2e_acceptance_p0_p2.py`, `tests/e2e/`) | `pytest -m "e2e and not gpu" -q` *(or `--target tests/test_e2e_acceptance_p0_p2.py` inside `verify_offline` snapshot)* |
| **Subtrans Engine & Evals** (`subtrans/*.py`, `glossaries/`) | `pytest -m "subtrans and not gpu" -q` *(inside `verify_offline` snapshot; `.\.venv\Scripts\python.exe subtrans.py eval` & `benchmarks/subtrans/eval_golden_pronouns*.py --dry-run`; delegate to [subtrans-evaluator](../../.agents/agents/subtrans-evaluator.md))* |
| **Video Core / Dedup / TRT / Mux** (`src/`, `upscale.py`) | `pytest -m "unit and not subtrans and not gpu" -q` *(inside `verify_offline` snapshot)* |
| **Live Local GPU / TRT / Benchmarks** (`src/build_trt_engine.py`, `scripts/benchmark_pipeline.py`, `scripts/silicon_roofline_probe.py`) | Autonomous when `.gpu_active.lock` is idle: `.\.venv\Scripts\python.exe src/build_trt_engine.py [--model <id>|all] [--dry-run]`, `.\.venv\Scripts\python.exe scripts/silicon_roofline_probe.py`, or `$env:RUN_GPU_TESTS=1; .\.venv\Scripts\python.exe -m pytest -m gpu -q` |
| **Static Linter** | `.\.venv\Scripts\ruff.exe check .` |

## Subagent Delegation & Pytest Markers
- **Custom Subagents (`inheritMcp: false`, `enableWriteTools: true` strictly for read-only CLI `run_command`; file-mutation tools prohibited):** [repo-scout](../../.agents/agents/repo-scout.md) (read-only `>= 3`-module AST/pack exploration) | [test-verifier](../../.agents/agents/test-verifier.md) (pytest, lint, `smoke_test_pipeline.py`, `verify_offline.py`) | [subtrans-evaluator](../../.agents/agents/subtrans-evaluator.md) (`subtrans/eval/runner.py`, pronoun benchmarks).
- **`pytest.ini` Markers (`--strict-markers`):** `unit`, `subtrans`, `e2e`, `gpu`, `web`, `smoke`. During iteration use `verify_offline.py --target` / `--impacted` or `smoke_test_pipeline.py`; run full `verify_offline.py` **once** at the final regression gate.

## Acceptance & Reporting Rules
- Record exact commands, interpreter (`.venv`), `passed`/`failed`/`skipped`/`deselected` counts, wall-clock duration, and `scratch/agent-validation-*` evidence path.
- For tooling/docs/offline code changes without real video output, report Mux Audit / Frame Count / FPS as **`N/A — no media artifact produced`**.
- Never relax assertions to force green tests. Full autonomous authority is granted for local CPU/RAM/NVMe/TensorRT/CUDA/NVENC runs, full-episode upscales (`--no-subs`), hardware benchmarks, and `@pytest.mark.gpu` (`RUN_GPU_TESTS=1`) tests when `.gpu_active.lock` is idle; never invoke live paid Cloud Gemini API calls without explicit user approval.
- **Live Headless Chrome CDP + GPU E2E Evidence (`2026-11-24`, real `web_gui.py --no-browser` + `chrome.exe --headless=new`, isolated `scratch/web_live_qa_out`):** Verified 5 tabs + dialogs (`0` JS/console/CSP errors, `0 MiB` Web VRAM), zero horizontal overflow across `1920x1080`/`900x800`/`390x844`, `#fs-picker-dialog` path jump + filename filter + `foldersOnly` `-o` browse, `<input type="number">` `validity.badInput` `"e"` rejection, `localStorage` config persistence across reload (`6s` preview, `dis-auto`, `force=false`), live 6s 4K GPU preview (`Violet Evergarden`, `.audit.json` auto-organized into `audit/`) + Tab 3 wheel zoom/pan/quick wipes, and live `Mushoku Tensei` 48s GPU upscale (`1Hz` intra-chunk `active_chunk_fraction` DOM progress, graceful pause `1/2` chunk `270.12 MB`, checkpoint ETA deduction `872.4s -> 566.7s`, 1-touch resume, concurrent double-abort `.gpu_active.lock` reclaim, and 2-step UI checkpoint delete).
