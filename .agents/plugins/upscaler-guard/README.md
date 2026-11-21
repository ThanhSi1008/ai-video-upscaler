# AI Video Upscaler Safety Guard (`upscaler-guard`)

Workspace plugin for **Google Antigravity** providing zero-overhead lifecycle safety hooks tailored to `e:\ai-video-upscaler`:

- **`SessionStart` (`hooks/session_start.py`)**: Injects a single compact status line (`< 260` chars) with Git `branch@HEAD`, `.gpu_active.lock` state, `.temp_*` checkpoint count (`root` + `out/`), `Dirty=<N>`, `STATE=fresh|stale`, daytime Gemini quota guard (`>= 14:00`), and latest task handoff path.
- **`PreToolUse` (`hooks/pre_tool_guard.py`)**:
  - Allows local `.env` / `subtrans/.env` reads; prompts (`force_ask`) on `.env` writes/edits; **hard-denies (`deny`)** `git add`/`git commit` targeting `.env`.
  - **Context & Rewrite Firewall (`deny`)**: Blocks reading `scratch/review_bundles/`, blocks full-file `write_file`/`write_to_file` on existing `> 250`-line files, requires `<= 280`-line bounded ranges on `view_file` for `> 350`-line files, and blocks unbounded shell dumps on `> 350`-line files.
  - **Prompts (`force_ask`)** before deleting `.temp_*` checkpoints or calling live Gemini APIs; **hard-denies** unapproved `build_trt_engine.py`, environment mutations, and unflagged full-episode upscale/benchmark runs; **prompts** on unisolated `pytest` or concurrent GPU runs.
- **`PostToolUse` (`hooks/post_tool_lint.py`)**: Runs fast Ruff `--fix` on modified `.py` files and syntax checks on `.json`/`.yaml`/`.yml` files (`timeout=3s`).
- **`Stop` (`hooks/stop_audit.py`)**: Audits staged `.env` secrets, stray root temp files, modified `.py` lint, and memory budget/link integrity (`timeout=3s`).
