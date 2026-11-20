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
# Read-only bootstrap (Git branch, HEAD, dirty tree, STATE, freshness check):
.\.venv\Scripts\python.exe scripts/agent_context.py

# Fast AST symbol discovery (saves context tokens vs reading entire large modules):
.\.venv\Scripts\python.exe scripts/agent_context.py --find <symbol_name>
.\.venv\Scripts\python.exe scripts/agent_context.py --symbols src/pipeline.py
```

For daily development handoffs, update [STATE.md](../../../docs/agents/STATE.md) (and [DECISIONS.md](../../../docs/agents/DECISIONS.md) when an architectural decision changes) **in place**, keeping `Commit Baseline` aligned with the latest runtime commit so `agent_context.py --check` reports 0 freshness warnings. Do **not** generate new dated task files in `docs/agents/tasks/` unless explicitly requested by the user (use [TEMPLATE.md](../../../docs/agents/tasks/TEMPLATE.md) only when requested).

## 2. Pre-Commit Verification Gate

Before staging or committing, verify all Definition of Done gates:

```powershell
# 1. Instruction budgets (GEMINI.md <= 4.5 KB & <= 45 lines, scoped .agents/rules/*.md <= 4 KB, GEMINI + subtrans skill <= 32 KiB) & relative links:
.\.venv\Scripts\python.exe scripts/agent_context.py --check

# 2. Isolated offline test suite & linter:
.\.venv\Scripts\python.exe scripts/verify_offline.py
.\.venv\Scripts\ruff.exe check .

# 3. Ensure no secrets, checkpoints, caches, or scratch files are staged:
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
