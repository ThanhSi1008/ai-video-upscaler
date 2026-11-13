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
# Read-only bootstrap (Git branch, HEAD, dirty tree, START_HERE, STATE, freshness check):
.\.venv\Scripts\python.exe scripts/agent_context.py

# Fast AST symbol discovery (saves context tokens vs reading entire large modules):
.\.venv\Scripts\python.exe scripts/agent_context.py --find <symbol_name>
.\.venv\Scripts\python.exe scripts/agent_context.py --symbols src/pipeline.py
```

For multi-step tasks, immediately create `docs/agents/tasks/YYYY-MM-DD-<slug>.md` from [TEMPLATE.md](../../../docs/agents/tasks/TEMPLATE.md) and update it as milestones complete (never wait until the end of a long session).

## 2. Pre-Commit Verification Gate

Before staging or committing, verify all Definition of Done gates:

```powershell
# 1. Instruction byte budgets (AGENTS.md <= 12 KiB, AGENTS + subtrans/AGENTS <= 32 KiB) & relative links:
.\.venv\Scripts\python.exe scripts/agent_context.py --check

# 2. Isolated offline test suite (when code/tests/configs changed):
.\.venv\Scripts\python.exe scripts/verify_offline.py

# 3. Ensure no secrets, checkpoints, caches, or scratch files are staged:
git status --short
git diff --cached --check
```

## 3. Mandatory Per-Request Commit Format (`AGENTS.md` §5)

Commit **immediately** upon completing each user or research request (never batch unrelated requests). Every commit message body MUST contain 3 explicit sections:

```text
<type>(<scope>): <concise summary>

- Nguon yeu cau: <Nguoi dung yeu cau truc tiep / AI nghien cuu de xuat>
- Hanh dong cu the:
  + <Module/file changed, bug fixed, or capability added>
- So lieu & Ket qua:
  + <pytest counts & time, agent_context.py --check byte sizes, mux/fps or N/A>
```
