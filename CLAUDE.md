# CLAUDE.md — Guidelines for Anthropic Claude Code & AI Agents

> **Quick bootstrap compass for Anthropic Claude Code & CLI agents.**
> **Start here:** [docs/agents/START_HERE.md](docs/agents/START_HERE.md) | **Full Architecture Compass:** [AGENTS.md](AGENTS.md).

---

## 1. Core Invariants (Do NOT Break)

1. **100% LOCAL on Windows 11 Workstation:** (Intel Core i5-12600K + NVIDIA RTX 3060 12GB). Never propose Cloud GPUs, Google Colab, or Linux-specific workarounds.
2. **Mandatory Commit Per Request:** Commit git immediately after completing EACH request (stating source, concrete actions, and experimental metrics).
3. **Protect API Keys & Daytime Quota:** Never print, log, or commit `GEMINI_API_KEY`. Never make real API calls between 14:00 and overnight runs; tests must use stubs or `--dry-run`.
4. **Preserve Checkpoints:** Never delete or tamper with `.temp_<basename>` on SSD without user confirmation.
5. **GPU Exclusivity:** Only 1 GPU inference process at a time; never load local LLMs while PyTorch GPU upscale is active.

---

## 2. Persistent Memory & Context Hierarchy

- **Bootstrap Entry Point:** [`docs/agents/START_HERE.md`](docs/agents/START_HERE.md)
- **Codebase Index:** [`docs/agents/PROJECT_MAP.md`](docs/agents/PROJECT_MAP.md)
- **Current State & Limitations:** [`docs/agents/STATE.md`](docs/agents/STATE.md)
- **Architectural Decision Records (ADR):** [`docs/agents/DECISIONS.md`](docs/agents/DECISIONS.md)
- **Safe Validation Playbook:** [`docs/agents/VALIDATION.md`](docs/agents/VALIDATION.md)
- **Subtitle Translation Engine:** [`subtrans/AGENTS.md`](subtrans/AGENTS.md)

---

## 3. Standard Commands

```powershell
# Check agent memory integrity (bounded context window):
python scripts/agent_context.py
python scripts/agent_context.py --check

# Safe offline validation:
python scripts/verify_offline.py

# Run full test suite:
pytest
```
