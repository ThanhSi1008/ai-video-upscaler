# Validation & Testing Guide

> **Updated:** 2026-11-22 | **Related:** [GEMINI](../../GEMINI.md) | [STATE](STATE.md) | [PROJECT_MAP](PROJECT_MAP.md)

## Default Offline Verification Sequence (PowerShell)

Always invoke the repository `.venv` interpreter from the repo root (system `PATH` Python may lack dependencies):

```powershell
.\.venv\Scripts\python.exe scripts/agent_context.py --check
.\.venv\Scripts\python.exe scripts/verify_offline.py
.\.venv\Scripts\ruff.exe check .
git diff --check
```

- `verify_offline.py` copies tracked/untracked source/config/fixtures into a disposable snapshot (`scratch/agent-validation-*/repo`), hides CUDA (`CUDA_VISIBLE_DEVICES=-1`), caps BLAS/OMP threads (`OMP/MKL/OPENBLAS/NUMEXPR=1`) with Win11 P-core QoS (`_configure_win32_pcore_qos`), forces CPU software HEVC encoding (`UPSCALER_HEVC_ENCODER=libx265`), strips API keys/tokens, disables `.env`, blocks external Python sockets across the entire subprocess tree via `sitecustomize.py` + `agent_offline_guard` (preserving loopback/socketpair for local ASGI/EventLoop tests), and executes calibrated 6-worker LPT `pytest -m "not gpu"` (`--show-capture=no`, `~55–60s` wall time, `1011 passed, 6 skipped, 0 xfailed, 0 xpassed`).
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
| **Web Studio Offline & Mock** (`web/`, `web_gui.py --mock`) | `.\.venv\Scripts\python.exe -m pytest tests/test_web_gui.py tests/test_architecture_boundaries.py -q` *(`64` tests in `test_web_gui.py` + `5` AST/zero-VRAM/Night-Report tests in `test_architecture_boundaries.py` forbidding `src.logger` in `web/`; or `.\.venv\Scripts\python.exe web_gui.py --mock` on `127.0.0.1`)* |
| **Impacted / Targeted Snapshot** (`< 5s` `src/` & `subtrans/`) | `.\.venv\Scripts\python.exe scripts/verify_offline.py --impacted` *(or `--target <PATH_OR_NODEID>`)* |
| **Full Offline Suite** (All non-GPU modules, final gate) | `.\.venv\Scripts\python.exe scripts/verify_offline.py` |
| **E2E Acceptance Suite** (`tests/test_e2e_acceptance_p0_p2.py`, `tests/e2e/`) | `pytest -m "e2e and not gpu" -q` *(or `--target tests/test_e2e_acceptance_p0_p2.py` inside `verify_offline` snapshot)* |
| **Subtrans Engine & Evals** (`subtrans/*.py`, `glossaries/`) | `pytest -m "subtrans and not gpu" -q` *(inside `verify_offline` snapshot; `.\.venv\Scripts\python.exe subtrans.py eval` & `benchmarks/subtrans/eval_golden_pronouns*.py --dry-run`; delegate to [subtrans-evaluator](../../.agents/agents/subtrans-evaluator.md))* |
| **Video Core / Dedup / TRT / Mux** (`src/`, `upscale.py`) | `pytest -m "unit and not subtrans and not gpu" -q` *(inside `verify_offline` snapshot)* |
| **Static Linter** | `.\.venv\Scripts\ruff.exe check .` |

## Subagent Delegation & Pytest Markers
- **Custom Subagents (`inheritMcp: false`, `enableWriteTools: true` strictly for read-only CLI `run_command`; file-mutation tools prohibited):** [repo-scout](../../.agents/agents/repo-scout.md) (read-only `>= 3`-module AST/pack exploration) | [test-verifier](../../.agents/agents/test-verifier.md) (pytest, lint, `smoke_test_pipeline.py`, `verify_offline.py`) | [subtrans-evaluator](../../.agents/agents/subtrans-evaluator.md) (`subtrans/eval/runner.py`, pronoun benchmarks).
- **`pytest.ini` Markers (`--strict-markers`):** `unit`, `subtrans`, `e2e`, `gpu`, `web`, `smoke`. During iteration use `verify_offline.py --target` / `--impacted` or `smoke_test_pipeline.py`; run full `verify_offline.py` **once** at the final regression gate.

## Acceptance & Reporting Rules
- Record exact commands, interpreter (`.venv`), `passed`/`failed`/`skipped`/`deselected` counts, wall-clock duration, and `scratch/agent-validation-*` evidence path.
- For tooling/docs/offline code changes without real video output, report Mux Audit / Frame Count / FPS as **`N/A — no media artifact produced`**.
- Never relax assertions to force green tests; never run GPU or live API tests without explicit user scope approval.
- **Live `/browser` E2E Evidence (`2026-11-22`, real `web_gui.py --no-browser` without `--mock`, isolated `scratch/live_ui_test_out`):** Verified headless Chrome CDP across 5 tabs + 2 dialogs (`0` JS/console/CSP errors), `1440x900`/`970x900`/`390x844` zero overflow (`103.6px` mobile header), `#fs-picker-dialog` browse & inline error + `.env` `403`, real 10s GPU preview (`9.97 fps`, `dis-auto`, `preview_start_sec=529.077`, `0` files in `out/`), real GPU episode heatmap (`17.05 fps`) + graceful pause + Resume CTA sync + `abort_job` `.gpu_active.lock` cleanup, Tab 3 Compare (`topDiff=0`, boundary `999999`), Tab 4 `shows.yaml` LF round-trip, Tab 5 deduplicated audits (`5` `benchmark_group_b` `_4K` / `10` total) + Night Report HTTP `200`, and clean shutdown.
