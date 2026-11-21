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
3. Mutate runtime quota state or auto-select CUDA on tests lacking `@pytest.mark.gpu`.

## 1. Standard Validation Sequence (PowerShell)

Always use the `.venv` interpreter from the repository root:

```powershell
# Step 1: Verify agent memory files, links, byte/line budgets, and baseline freshness
.\.venv\Scripts\python.exe scripts/agent_context.py --check

# Step 2: Run full non-GPU test suite inside a disposable snapshot (sockets blocked, CUDA hidden, API keys stripped)
.\.venv\Scripts\python.exe scripts/verify_offline.py

# Step 3: Check Python correctness/linting via Ruff
.\.venv\Scripts\ruff.exe check .

# Step 4: Verify clean whitespace / line endings
git diff --check
```

## 2. Fast Targeted Test Iteration

- **Pure stdlib / web / smoke tests** (`test_agent_context.py`, `test_agent_setup.py`, `test_verify_offline.py`, `test_architecture_boundaries.py`, `test_export_review_bundle.py`, `smoke_test_pipeline.py`, `test_web_gui.py`) may be run directly:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_agent_context.py tests/test_agent_setup.py tests/test_verify_offline.py tests/test_architecture_boundaries.py tests/test_export_review_bundle.py tests/test_web_gui.py -q
  .\.venv\Scripts\python.exe scripts/smoke_test_pipeline.py
  ```
- **Targeted `src/`, `subtrans/`, or `e2e` iteration (`< 5s`)**: Run inside a disposable snapshot (auto-cleaned on exit) via:
  ```powershell
  .\.venv\Scripts\python.exe scripts/verify_offline.py --target tests/test_zero_disk_ring_buffers_and_cel.py
  .\.venv\Scripts\python.exe scripts/verify_offline.py --impacted
  ```

## 3. Reporting Evidence

Record in [STATE.md](../../../docs/agents/STATE.md) and the Git commit message (create `docs/agents/tasks/YYYY-MM-DD-<slug>.md` only when explicitly requested by the user):
- Exact command & interpreter (`.\.venv\Scripts\python.exe`).
- Counts: `passed`, `failed`, `skipped`, wall-clock duration, and `scratch/agent-validation-*` evidence path.
- If no video was rendered/muxed in the task, record Mux Audit / Frame Count / FPS as **`N/A — no media artifact produced`** (never claim `PASS` without real media verification).
