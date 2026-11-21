---
name: agent-handoff-commit
description: >-
  Manage persistent multi-agent context handoffs (in-place updates to docs/agents/STATE.md and DECISIONS.md), AST symbol lookups,
  memory byte-budget checks, and mandatory per-request Git commits in e:\ai-video-upscaler.
  Activate at the start of multi-step work, before context compaction, and before committing.
---

# Agent Context, Handoff & Mandatory Commit Workflow

## 1. Start / Resume & Daily Handoff Protocol

```powershell
# Read-only bootstrap (only needed when SessionStart does not report STATE=fresh):
.\.venv\Scripts\python.exe scripts/agent_context.py

# Tiered AST context discovery & impact lookup (saves tokens vs reading >350-line files):
.\.venv\Scripts\python.exe scripts/agent_context.py --pack src/pipeline.py:worker_extract web/app.py:api_subtrans_overview
.\.venv\Scripts\python.exe scripts/agent_context.py --symbols src/pipeline.py web/fs_browser.py
.\.venv\Scripts\python.exe scripts/agent_context.py --slice src/logger.py RunLogger.log_chunk_done
.\.venv\Scripts\python.exe scripts/agent_context.py --impact src/logger.py
```

For daily development handoffs, update [STATE.md](../../../docs/agents/STATE.md) (and [DECISIONS.md](../../../docs/agents/DECISIONS.md) when an architectural decision changes) **in place**, keeping `Commit Baseline` aligned with the latest runtime commit so `agent_context.py --check` reports 0 freshness warnings. Do **not** generate new dated task files in `docs/agents/tasks/` unless explicitly requested by the user (use [TEMPLATE.md](../../../docs/agents/tasks/TEMPLATE.md) only when requested).

## 2. Pre-Commit Verification & Doc-Sync Gate

- **Mandatory Same-Commit Doc-Sync Checklist:** Whenever module layout, symbols, CLI flags, schemas, hooks, or runtime invariants change, sync affected docs (`docs/agents/STATE.md`, `PROJECT_MAP.md`, `VALIDATION.md`, `DECISIONS.md`, `docs/*.md`, `README.md`, `.agents/`) in the same commit (leave unrelated docs untouched on internal fixes).
- **Hooks & Post-Commit Bundle Sync:** Run `--install-hooks` once to install both managed `pre-commit` (`agent_context.py --check`) and non-blocking `post-commit` (`scripts/export_review_bundle.py --sync-latest` -> `scratch/review_bundles/`). Never read `scratch/review_bundles/` into agent context.

```powershell
# 1. Instruction budgets, links, and hook setup:
.\.venv\Scripts\python.exe scripts/agent_context.py --check
.\.venv\Scripts\python.exe scripts/agent_context.py --install-hooks

# 2. Fast targeted iteration & final isolated offline test suite + linter:
.\.venv\Scripts\python.exe scripts/verify_offline.py --impacted
.\.venv\Scripts\python.exe scripts/verify_offline.py
.\.venv\Scripts\ruff.exe check .

# 3. Ensure no secrets (.env), checkpoints, caches, or scratch files are staged:
git status --short
git diff --cached --check
```

## 3. Mandatory Per-Request Commit Format (`GEMINI.md` §4)

Commit **immediately** upon completing each user or research request (never batch unrelated requests). Every commit message body MUST be in English and contain 3 explicit sections:

```text
<type>(<scope>): <concise summary>

- Request Source: <Direct user request / AI research proposal>
- Actions:
  + <Module/file changed, bug fixed, or capability added>
- Metrics & Verification:
  + <pytest counts & time, agent_context.py --check status, ruff status, mux/fps or N/A>
```
