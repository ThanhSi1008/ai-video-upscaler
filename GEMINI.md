# GEMINI.md — Antigravity Single Source of Truth

> **Platform:** 100% AI-driven inside **Google Antigravity** (Desktop App & `agy` CLI).
> **Memory & Index:** [STATE](docs/agents/STATE.md) | [PROJECT_MAP](docs/agents/PROJECT_MAP.md) | [DECISIONS](docs/agents/DECISIONS.md) | [VALIDATION](docs/agents/VALIDATION.md)

## 1. AI Vibe-Coding & Language Contract
- **100% English Repository:** Code, comments, docstrings, configs, rules, subagents, skills, logs, search queries, and commits MUST be concise technical English.
- **Domain-Data Exception:** Strictly preserve VI/JA linguistic data, pronoun rules/regexes, translation prompts, subtitle fixtures (`subtrans/`, `glossaries/`, `tests/`), MKV title `'Tiếng Việt (Vietsub)'`, and existing test-asserted strings.
- **Vietnamese Chat Only:** Communicate directly with the user in Antigravity Desktop chat in concise, natural technical Vietnamese.

## 2. Workspace Environment & Context Discipline
- **Windows Toolchain Always:** Invoke Python/Ruff via `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\ruff.exe`; resolve `bin/ffmpeg/` before system `PATH`. Never mutate `.venv` packages (`requirements.lock` locked ABI).
- **E-Drive & Single GPU Owner:** Isolate caches/temps on E-drive (`src/env_isolation.py`). At most 1 GPU process holds `.gpu_active.lock` (RTX 3060 12GB, 32GB RAM); never run Cloud/Colab proposals or concurrent local LLMs.
- **Progressive Disclosure:** Bootstrap with `.\.venv\Scripts\python.exe scripts/agent_context.py`. Use `--find <sym>` / `--symbols <file>` before reading large modules. Subsystem rules auto-load via [video-gpu](.agents/rules/video-gpu.md), [subtrans](.agents/rules/subtrans.md), and [web-ui](.agents/rules/web-ui.md); read `.agents/skills/*/SKILL.md` only on task match.
- **Config Disambiguation:** `models/models.yaml` (SR) | `subtrans/models.yaml` (Gemini quota) | `shows.yaml` (per-series SR/dedup) | `subtrans/glossary.yaml` -> `glossaries/base.yaml`.

## 3. Core Commands & Subagent Delegation
- **Entrypoints:** `upscale.py <src.mkv> -o out/` | `subtrans.py run <ja.srt> -o out/` | `web_gui.py --mock` (`127.0.0.1:8765`).
- **Context-Isolated Subagents (`inheritMcp: false`):**
  - Delegate test suites, lint sweeps, and pipeline smoke checks (`scripts/smoke_test_pipeline.py`) to [test-verifier](.agents/agents/test-verifier.md).
  - Delegate subtitle/pronoun evals (`subtrans/eval/runner.py`, `benchmarks/subtrans/eval_golden_pronouns*.py`) to [subtrans-evaluator](.agents/agents/subtrans-evaluator.md).

## 4. Safety Boundaries, Verification & Commit
- **Secrets & Quota:** Read Gemini keys from `.env` at runtime only; never read, print, or commit `.env`. No live Gemini API calls (except `--dry-run`/`--offline`/stubs) without user approval, especially 14:00 until overnight runs.
- **Approval Gates:** Ask before deleting `.temp_*` checkpoints, running `build_trt_engine.py`, upscaling `> 100` frames, or altering default `animevideov3` / NVENC CQ 18 parameters.
- **Test Isolation & Gate:** Never run bare `pytest` on live repo root. Iterate via fast targeted `pytest` / `smoke_test_pipeline.py` (`< 5s`), and run `scripts/verify_offline.py` **once** before final commit.
- **Definition of Done & Commit:** Pass `verify_offline.py`, `agent_context.py --check`, `ruff check .`, `git diff --check`; update [STATE](docs/agents/STATE.md) in place and commit immediately in English (`Request Source`, `Actions`, `Metrics & Verification`).
