---
trigger: glob
globs: "web/**/*, web_gui.py, tests/test_web_gui.py"
description: "Vanilla HTML/CSS/JS local dashboard conventions, FastAPI + SSE responsiveness, zero JS build toolchains, and web_gui.py --mock verification on 127.0.0.1."
---

# Web Studio UI & FastAPI Scoped Rules

- **Zero Build Toolchain & Vanilla Stack:**
  - Keep `web/static/` pure Vanilla HTML5, CSS3 (`light-dark()`, custom properties, responsive grid/flex), and ES modules (`app.js`, `ui.js`, `panels.js`, `charts.js`, `compare_stage.js`) per [modern-web-guidance](../skills/modern-web-guidance/SKILL.md).
  - Never introduce Node/npm bundlers, React/Vue/Tailwind build toolchains, or external CDN scripts (preserve strict self-hosted `_CSP_HEADER_VALUE`).
- **FastAPI & SSE Responsiveness (`web/app.py`, `web/sse_hub.py`, `web/job_manager.py`):**
  - Bind `127.0.0.1:8765` by default with loopback/Origin anti-DNS-rebinding guards.
  - Keep `/api/status` and `/api/events` (SSE) non-blocking; query `subtrans/.cache/translations.db` exclusively via read-only URI (`?mode=ro`) and mask all API key slots.
- **UI Performance & Accessibility:**
  - Batch chart/canvas updates with `requestAnimationFrame`, maintain keyboard focus and ARIA attributes (`aria-busy`, `tabindex`), and sanitize dynamic DOM insertions.
- **Fast `--mock` & Offline Verification:**
  - Launch `.\.venv\Scripts\python.exe web_gui.py --mock` on `127.0.0.1` for deterministic simulated jobs, GPU telemetry, and SSE ticks during Browser subagent verification without touching CUDA/TensorRT or external drives.
  - Run offline route/SSE assertions via `.\.venv\Scripts\python.exe -m pytest tests/test_web_gui.py -q`.
