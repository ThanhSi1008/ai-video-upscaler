# AI Video Upscaler Safety Guard (`upscaler-guard`)

Workspace plugin for **Google Antigravity** providing zero-overhead lifecycle safety hooks tailored to `e:\ai-video-upscaler`:

- **`SessionStart` (`hooks/session_start.py`)**: Injects a single compact status line (`< 260` chars) with Git `branch@HEAD`, `.gpu_active.lock` state, `.temp_*` checkpoint count (`root` + `out/`), `Dirty=<N>`, `STATE=fresh|stale`, daytime Gemini quota guard (`>= 14:00`), and latest task handoff path.
- **`PreToolUse` (`hooks/pre_tool_guard.py`)**:
  - Allows local `.env` / `subtrans/.env` reads; prompts (`force_ask`) on `.env` writes/edits; **hard-denies (`deny`)** `git add`/`git commit` targeting `.env`.
  - **Context & Rewrite Firewall (`deny`)**: Blocks reading `scratch/review_bundles/`, blocks full-file `write_file`/`write_to_file` on existing `> 250`-line files, requires `<= 280`-line bounded ranges on `view_file` for `> 350`-line files, and blocks unbounded shell dumps on `> 350`-line files.
  - **Autonomous Local Hardware & Safety Prompts**: Allows full autonomous local CPU/RAM/NVMe/TensorRT (`src/build_trt_engine.py`)/CUDA/NVENC upscales and hardware benchmarks across any GPU frame counts when `.gpu_active.lock` is idle; **hard-denies** `.venv` environment mutations; **prompts (`force_ask`)** before deleting `.temp_*` checkpoints, calling live Cloud Gemini APIs, or colliding with an active `.gpu_active.lock`; **prompts (`ask`)** on unisolated non-GPU `pytest`.
- **`PostToolUse` (`hooks/post_tool_lint.py`)**: Runs fast Ruff `--fix` on modified `.py` files and syntax checks on `.json`/`.yaml`/`.yml` files (`timeout=3s`).
- **`Stop` (`hooks/stop_audit.py`)**: Audits staged `.env` secrets, stray root temp files, modified `.py` lint, and memory budget/link integrity (`timeout=3s`).
