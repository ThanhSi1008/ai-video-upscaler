---
name: agent-handoff-commit
description: >-
  Manage persistent multi-agent context handoffs (docs/agents/tasks/), AST symbol lookups,
  memory byte-budget checks, and mandatory per-request Git commits in e:\ai-video-upscaler.
  Activate at the start of multi-step work, before context compaction, and before committing.
---

# Agent Context, Handoff & Mandatory Commit Workflow

## 1. Start / Resume Protocol

```powershell
# Read-only bootstrap (Git branch, HEAD, dirty tree, STATE, freshness check):
.\.venv\Scripts\python.exe scripts/agent_context.py

# Fast AST symbol discovery (saves context tokens vs reading entire large modules):
.\.venv\Scripts\python.exe scripts/agent_context.py --find <symbol_name>
.\.venv\Scripts\python.exe scripts/agent_context.py --symbols src/pipeline.py
```

For multi-step tasks, immediately create `docs/agents/tasks/YYYY-MM-DD-<slug>.md` from [TEMPLATE.md](../../../docs/agents/tasks/TEMPLATE.md) and update it as milestones complete (never wait until the end of a long session).

## 2. Pre-Commit Verification Gate

Before staging or committing, verify all Definition of Done gates:

```powershell
# 1. Instruction budgets (GEMINI.md <= 12 KiB & <= 120 lines, GEMINI + subtrans-anime-ops <= 32 KiB) & relative links:
.\.venv\Scripts\python.exe scripts/agent_context.py --check

# 2. Isolated offline test suite & linter:
.\.venv\Scripts\python.exe scripts/verify_offline.py
.\.venv\Scripts\ruff.exe check .

# 3. Ensure no secrets, checkpoints, caches, or scratch files are staged:
git status --short
git diff --cached --check
```

## 3. Mandatory Per-Request Commit Format (`GEMINI.md` §5)

Commit **immediately** upon completing each user or research request (never batch unrelated requests). Every commit message body MUST be in English and contain 3 explicit sections:

```text
<type>(<scope>): <concise summary>

- Request Source: <Direct user request / AI research proposal>
- Actions:
  + <Module/file changed, bug fixed, or capability added>
- Metrics & Verification:
  + <pytest counts & time, agent_context.py --check status, ruff status, mux/fps or N/A>
```
