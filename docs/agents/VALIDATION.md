# Validation & Testing Guide

> **Updated:** 2026-11-21 | **Related:** [GEMINI](../../GEMINI.md) | [STATE](STATE.md) | [PROJECT_MAP](PROJECT_MAP.md)

## Default Offline Verification Sequence (PowerShell)

Always invoke the repository `.venv` interpreter from the repo root (system `PATH` Python may lack dependencies):

```powershell
.\.venv\Scripts\python.exe scripts/agent_context.py --check
.\.venv\Scripts\python.exe scripts/verify_offline.py
.\.venv\Scripts\ruff.exe check .
git diff --check
```

- `verify_offline.py` copies tracked/untracked source/config/fixtures into a disposable snapshot (`scratch/agent-validation-*/repo`), hides CUDA (`CUDA_VISIBLE_DEVICES=-1`), strips API keys/tokens, disables `.env`, blocks Python sockets before collection, and executes `pytest -m "not gpu"`.
- Logs (`pytest.log`, `junit.xml`) remain under `scratch/agent-validation-*/` (up to 2 recent passing runs retained); snapshot `repo/` and `tmp/pytest/` are deleted automatically on pass.

## Why Bare `pytest` on Live Repo Is Prohibited

1. `src/config.py` registers an `atexit` handler that can delete live `.gpu_active.lock`.
2. Default `TranslationCache()` / `KnowledgeBase()` calls can write to production SQLite DBs in `subtrans/.cache/`.
3. Some tests without `@pytest.mark.gpu` auto-select CUDA if visible.
4. `subtrans.py check` / `models` hit real Gemini endpoints.

## Module -> Targeted Test Command

Run targeted tests directly ONLY for pure stdlib agent-tooling tests, or inside a disposable snapshot (with `CUDA_VISIBLE_DEVICES=-1 -p agent_offline_guard`) for runtime modules:

| Subsystem / Module | Targeted Command |
| :--- | :--- |
| **Agent Context & Setup** (`scripts/agent_context.py`, `.agents/`) | `.\.venv\Scripts\python.exe -m pytest tests/test_agent_context.py tests/test_agent_setup.py tests/test_verify_offline.py -q` |
| **Full Offline Suite** (All non-GPU modules) | `.\.venv\Scripts\python.exe scripts/verify_offline.py` |
| **E2E Acceptance Suite** (`tests/e2e/`, Group A/B, T1-T4) | `pytest -m "e2e and not gpu" -q` *(inside `verify_offline` snapshot)* |
| **Subtrans Engine** (`subtrans/*.py`, `glossaries/`) | `pytest -m "subtrans and not gpu" -q` *(inside `verify_offline` snapshot)* |
| **Video Core / Dedup / TRT / Mux** (`src/`, `upscale.py`) | `pytest -m "unit and not subtrans and not gpu" -q` *(inside `verify_offline` snapshot)* |
| **Web Studio** (`web/`, `web_gui.py`) | `pytest tests/test_web_gui.py -q` *(inside `verify_offline` snapshot)* |
| **Static Linter** | `.\.venv\Scripts\ruff.exe check .` |

## Acceptance & Reporting Rules
- Record exact commands, interpreter (`.venv`), `passed`/`failed`/`skipped`/`deselected` counts, wall-clock duration, and `scratch/agent-validation-*` evidence path.
- For tooling/docs/offline code changes without real video output, report Mux Audit / Frame Count / FPS as **`N/A — no media artifact produced`**.
- Never relax assertions to force green tests; never run GPU or live API tests without explicit user scope approval.
