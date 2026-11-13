# Claude / other coding agents — project entry point

Read [AGENTS.md](AGENTS.md), then [START_HERE](docs/agents/START_HERE.md), [STATE](docs/agents/STATE.md) and the current task handoff. These are the shared repository memory; do not maintain a second policy here.

After compaction or transfer, reconcile Git and the latest user request with the handoff before acting. Read [Subtrans instructions](subtrans/AGENTS.md) when editing that subsystem. Update the handoff during work, and assign file ownership to collaborators.

Bootstrap: `python scripts/agent_context.py` (standard library, no API calls).
Tests: follow [VALIDATION](docs/agents/VALIDATION.md), using the existing .venv interpreter and isolated runner.
If your tool does not discover instructions automatically, explicitly open the files above.

