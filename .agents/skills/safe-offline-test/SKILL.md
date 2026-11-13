---
name: safe-offline-test
description: >-
  Run isolated offline pytest suites and memory integrity checks in e:\ai-video-upscaler
  without touching production SQLite caches, .gpu_active.lock, models.yaml, checkpoints,
  GPU VRAM, or daytime Gemini API quota. Activate whenever running tests or validating code changes.
---

# Safe Offline Validation & Testing Runbook

Never run bare `pytest` directly on the production repository root (`e:\ai-video-upscaler`).
Importing runtime modules directly in the live repo can:
1. Trigger `src/config.py` `atexit` handlers that delete `.gpu_active.lock`.
2. Write to `subtrans/.cache/translations.db` or `subtrans/.cache/knowledge.db`.
3. Mutate `subtrans/models.yaml` during retry/quota tests.
4. Auto-select CUDA on tests lacking `@pytest.mark.gpu`.

## 1. Standard Validation Sequence (PowerShell)

Always use the `.venv` interpreter from the repository root:

```powershell
# Step 1: Verify agent memory files, links, byte budgets, and baseline freshness
.\.venv\Scripts\python.exe scripts/agent_context.py --check

# Step 2: Run full non-GPU test suite inside a disposable snapshot (sockets blocked, CUDA hidden, API keys stripped)
.\.venv\Scripts\python.exe scripts/verify_offline.py

# Step 3: Check critical Python syntax/undefined-name issues via Ruff
ruff check upscale.py subtrans.py src subtrans scripts tests

# Step 4: Verify clean whitespace / line endings
git diff --check
```

## 2. Fast Targeted Test Iteration

- **Pure stdlib agent-tooling tests** (`tests/test_agent_context.py`, `tests/test_agent_setup.py`) do not import `src/` or `subtrans/` and may be run directly:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_agent_context.py tests/test_agent_setup.py -q
  ```
- **Targeted `src/` or `subtrans/` tests**: First run `verify_offline.py` once (or inspect the latest `scratch/agent-validation-*/repo` snapshot), then invoke `pytest` inside that snapshot directory with `CUDA_VISIBLE_DEVICES=-1` and `-p agent_offline_guard` so production caches remain untouched.

## 3. Reporting Evidence

Record in the active task handoff (`docs/agents/tasks/YYYY-MM-DD-<slug>.md`):
- Exact command & interpreter (`.\.venv\Scripts\python.exe`).
- Counts: `passed`, `failed`, `skipped`, wall-clock duration, and `scratch/agent-validation-*` evidence path.
- If no video was rendered/muxed in the task, record Mux Audit / Frame Count / FPS as **`N/A — không tạo thành phẩm`** (never claim `PASS` without real media verification).
