# AI Video Upscaler Safety Guard (`upscaler-guard`)

Workspace plugin for **Google Antigravity** providing zero-overhead lifecycle safety hooks tailored to `e:\ai-video-upscaler`:

- **`SessionStart` (`hooks/session_start.py`)**: Injects a single compact status line with current Git `branch@HEAD`, `.gpu_active.lock` state, `.temp_*` checkpoint count, daytime Gemini quota guard state (`>= 14:00`), and the latest task handoff file.
- **`PreToolUse` (`hooks/pre_tool_guard.py`)**:
  - **Hard-blocks (`deny`)** reading/editing `.env` or printing `GEMINI_API_KEY`.
  - **Prompts (`force_ask`)** before deleting `.temp_*` resume checkpoints or running `--clean-temp` / `--force`.
  - **Prompts (`force_ask`)** before calling live Gemini APIs (`subtrans.py check/models/run` without `--dry-run`).
  - **Prompts (`ask` / `force_ask`)** before running unisolated `pytest` on the production repo or starting GPU jobs while `.gpu_active.lock` exists.
