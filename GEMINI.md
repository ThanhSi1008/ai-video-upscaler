# GEMINI.md — Antigravity Single Source of Truth

> **Platform:** 100% AI-driven inside **Google Antigravity** (Desktop App & `agy` CLI).
> **Memory & Index:** [STATE](docs/agents/STATE.md) | [PROJECT_MAP](docs/agents/PROJECT_MAP.md) | [DECISIONS](docs/agents/DECISIONS.md) | [VALIDATION](docs/agents/VALIDATION.md)

## 1. AI Vibe-Coding, Comment & Language Contract
- **100% English & AI-Only Comment Standard:** Code, docstrings, configs, rules, subagents, skills, search queries, and commits MUST be concise technical English. Use 1-line English contract docstrings (no filename echoes) and concise `#` comments ONLY for non-obvious `WHY`/invariants, `# noqa`, or `# type: ignore`; zero `# -*- coding: utf-8 -*-`, `# ===` banners, or task-history tags.
- **Domain-Data & Runtime String Exception:** Strictly preserve VI/JA linguistic data, pronoun rules/regexes, translation prompts, subtitle fixtures (`subtrans/`, `glossaries/`, `tests/`), MKV title `'Tiếng Việt (AI Vietsub)'`, and existing Vietnamese `print()`/`logger`/UI strings in `upscale.py`, `src/`, and `web/`.
- **Vietnamese Chat Only:** Communicate directly with the user in Antigravity Desktop chat in concise, natural technical Vietnamese.

## 2. Workspace Environment & Context Discipline
- **Windows Toolchain & Autonomous Hardware:** Invoke `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\ruff.exe`; resolve `bin/ffmpeg/` before `PATH`; never mutate `.venv`; isolate caches on E-drive (`src/env_isolation.py`). Full autonomous authority on 100% of local hardware (i5-12600K CPU, RAM/NVMe, TensorRT 10 FP16 `src/build_trt_engine.py`, PyTorch CUDA FP16, 10-bit `hevc_nvenc` on RTX 3060 12GB) for any frame counts, TRT builds, full-episode upscales (`--no-subs` unless Cloud subtrans is approved), and benchmarks when `.gpu_active.lock` is idle (max 1 GPU owner; no Cloud/Colab or local LLMs).
- **Tiered Context & Rewrite Firewall:** Skip no-arg `agent_context.py` when `SessionStart` reports `STATE=fresh`. Run batched `agent_context.py --pack <sym...>`, `--symbols <path...>`, `--slice`, and `--impact` before reading files `> 350` lines; edit existing files `> 250` lines surgically (never full-file overwrite); never read generated bundles in `scratch/review_bundles/`.
- **Scoped Rules & Configs:** Auto-loaded via [video-gpu](.agents/rules/video-gpu.md), [subtrans](.agents/rules/subtrans.md), [web-ui](.agents/rules/web-ui.md); read `.agents/skills/*/SKILL.md` on task match. Configs: `models/models.yaml` (SR) | `subtrans/models.yaml` (Gemini) | `shows.yaml` (show rules) | `subtrans/glossary.yaml` -> `glossaries/base.yaml`.

## 3. Core Commands & Subagent Delegation
- **Entrypoints:** `upscale.py <src.mkv> -o out/` | `subtrans.py run <ja.srt> -o out/` | `web_gui.py --mock` (`127.0.0.1:8765`).
- **Context-Isolated Subagents (`inheritMcp: false`):**
  - Delegate `>= 3`-module discovery to [repo-scout](.agents/agents/repo-scout.md).
  - Delegate test/lint/`verify_offline.py` and `scripts/smoke_test_pipeline.py` runs to [test-verifier](.agents/agents/test-verifier.md).
  - Delegate subtitle/pronoun evals (`subtrans/eval/runner.py`, `benchmarks/subtrans/eval_golden_pronouns*.py`) to [subtrans-evaluator](.agents/agents/subtrans-evaluator.md).

## 4. Safety, Fast Testing, Doc-Sync & Commit
- **Secrets & Quota:** Local `.env` read/write is allowed freely (free-tier keys); never commit, log, or export `.env`. No live Gemini API calls (except `--dry-run`/`--offline`/stubs) without approval, especially `>= 14:00`.
- **Approval Gates:** Zero approval needed for GPU frame counts, `build_trt_engine.py`, full-episode upscales, or hardware benchmarks. Ask before deleting `.temp_*` checkpoints or altering repo `animevideov3` / NVENC `CQ 18` defaults.
- **Fast Verification & Gate:** Never run bare `pytest` on live repo root. Iterate via `< 5s` `verify_offline.py --impacted` / `--target <test>` or `smoke_test_pipeline.py`, and run full `scripts/verify_offline.py` **once** before commit.
- **Same-Commit Doc-Sync & Commit:** Pass `verify_offline.py`, `agent_context.py --check`, `ruff check .`, `git diff --check`. Sync affected docs (`STATE.md`, `PROJECT_MAP.md`, `VALIDATION.md`, `DECISIONS.md`, `README.md`, `docs/*.md`, `.agents/`) in the same commit when layout, symbols, CLI flags, schemas, hooks, or invariants change (never touch unrelated docs on internal fixes); commit immediately in English (`Request Source`, `Actions`, `Metrics & Verification`).
