# GEMINI.md — Antigravity Single Source of Truth

> **Platform:** 100% AI-driven development inside **Google Antigravity** (Desktop App & `agy` CLI).
> **Memory & Docs:** [STATE](docs/agents/STATE.md) | [PROJECT_MAP](docs/agents/PROJECT_MAP.md) | [DECISIONS](docs/agents/DECISIONS.md) | [VALIDATION](docs/agents/VALIDATION.md)

## 1. AI Vibe-Coding & Language Contract
- **100% English Repository:** All files, code, comments, docstrings, CLI/log additions, configs, Markdown docs, skills, internal reasoning, search queries, and Git commit messages MUST be concise technical English.
- **Mandatory Domain-Data Exception:** Strictly preserve VI/JA linguistic data, pronoun rules, LLM translation prompts, subtitle fixtures (`subtrans/`, `glossaries/`, `tests/`), MKV track title `'Tiếng Việt (Vietsub)'`, and existing test-asserted runtime strings.
- **Vietnamese Chat Reports Only:** Communicate directly with the user in the Antigravity Desktop App chat in concise, natural technical Vietnamese.
- **Autonomous Verification:** Never hand off unverified edits; run `agent_context.py --check`, `verify_offline.py`, `ruff check .`, and `git diff --check` before final reporting.

## 2. Context Bootstrap & Lazy Loading
- Start/resume: run `.\.venv\Scripts\python.exe scripts/agent_context.py`; read [STATE](docs/agents/STATE.md) and any active task in `docs/agents/tasks/`.
- Use progressive disclosure: read [PROJECT_MAP](docs/agents/PROJECT_MAP.md), [DECISIONS](docs/agents/DECISIONS.md), [VALIDATION](docs/agents/VALIDATION.md), and `.agents/skills/*/SKILL.md` only when relevant (`subtrans-anime-ops`, `video-upscale-ops`, `modern-web-guidance`, `safe-offline-test`, `agent-handoff-commit`).
- Use `.\.venv\Scripts\python.exe scripts/agent_context.py --find <sym>` or `--symbols <file>` before reading large modules. Never bulk-load `docs/archive/`, `logs/`, `out/`, `models_bench/`, media, SQLite caches, or `.env`.
- **Config Disambiguation:** `models/models.yaml` = video SR models | `subtrans/models.yaml` = Gemini LLM & quota chain | `shows.yaml` = per-series SR & dedup rules | `subtrans/glossary.yaml` = base subtitle glossary (aliased by `glossaries/base.yaml`).

## 3. Hardware & Pipeline Invariants (Do NOT Break)
- **100% Local Windows 11:** Intel i5-12600K (P-Core HighQoS, AVX2), NVIDIA RTX 3060 12GB (TensorRT 10.16 FP16 / PyTorch CUDA FP16, `batch_size=8`), 32GB RAM, E-drive isolation (`src/env_isolation.py`). Never propose Cloud GPUs, Colab, or Kaggle.
- **Zero-Disk Default & Storage Discipline:** `--zero-disk` defaults `True` (`NVDEC RGB24 pipe -> TRT/CUDA FP16 -> Triton p010le -> hevc_nvenc`); fallback `--no-zero-disk` (`--disk`) uses rolling chunk cleanup. SSD checkpoints keep only compressed `up_*.mp4` under `.temp_<basename>`.
- **3-Stage Overlap & Single GPU Owner:** CPU/NVDEC Decode (`N+1`), GPU Infer (`N`), and NVENC Encode (`N-1`) run overlapped. Only 1 GPU process may hold `.gpu_active.lock` (PID-checked); never load local LLMs concurrently.
- **Two-Way BT.709 Color Matrix:**
  - Extract: `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc` (Limited -> RGB24 Full)
  - Encode: `scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv` (RGB24 Full -> `p010le` Limited)
  - Metadata: `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`
- **10-bit NVENC Studio Preset:** `hevc_nvenc`, `main10`, `p010le`, preset `p6`, `CQ 18` (`maxrate 45M`), multipass `fullres`, `rc-lookahead 32`, `spatial-aq 1`, `aq-strength 6`, `temporal-aq 1`.
- **Bit-Exact Stream Copy Mux:** Always preserve original audio, softsubs, font attachments, and chapters using `-map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 -c copy -max_interleave_delta 0` (`-max_interleave_delta 0` prevents audio buffer dropouts).
- **Safe Lifecycle:** Win32 `SetThreadExecutionState` AntiSleep + Context Manager/`atexit` cleanup for `Ctrl+C` safety.

## 4. Core CLI Commands (PowerShell)
```powershell
.\.venv\Scripts\python.exe upscale.py <source.mkv> -o out/                     # Upscale 1 episode (Zero-Disk TRT default)
.\.venv\Scripts\python.exe upscale.py <source.mkv> --preview 10                # 10s preview at 30% runtime
.\.venv\Scripts\python.exe upscale.py --queue ep1.mkv ep2.mkv -o out/ --deadline 07:00 --allow-partial
.\.venv\Scripts\python.exe subtrans.py run <ja.srt> --en <en.ass> -o out/      # Translate subtitles (Cloud Gemini)
.\.venv\Scripts\python.exe web_gui.py                                          # Launch local Web Studio (127.0.0.1:8765)
.\.venv\Scripts\python.exe scripts/agent_context.py --check                    # Check memory files, budgets & links
.\.venv\Scripts\python.exe scripts/verify_offline.py                           # Run isolated offline pytest suite
```
- Key `upscale.py` flags: `--model` (`animevideov3`), `--engine` (`cuda`|`cpu`), `--batch-size` (`8`), `--no-zero-disk`/`--disk`, `--no-tensorrt`, `--dedup-policy` (`default`|`sensitive-a/b`|`motion`|`drift`|`dis`|`sakuga`|`dis-sakuga`|`off`), `--dis-thresh` (`0.30`), `--subs-quality` (`normal`|`fast`|`high`), `--subs-only`, `--subs-upgrade`, `--auto-align-en`/`--no-auto-align-en`, `--auto-offset`/`--no-auto-offset`, `--sub-offset-ms`, `--no-embed`, `--keep-temp`, `--clean-temp`, `--dry-run`.

## 5. Agent Boundaries & Mandatory Per-Request Commit
- **Secrets & Quota:** Read Gemini keys from `.env` at runtime only; never print, log, read, or commit `.env` or keys. **No live Gemini API calls** (including `subtrans.py check`/`models`) from **14:00 until overnight runs**; use `--dry-run` or stubs.
- **Approval Gates:** Ask the user before deleting/modifying `.temp_*` checkpoints, running GPU inference `> 100` frames, or changing default SR model (`animevideov3`) / NVENC quality parameters.
- **Test Isolation:** Never run bare `pytest` on the live repo root; always use `.\.venv\Scripts\python.exe scripts/verify_offline.py` (or targeted snapshot/pure-stdlib tests per [VALIDATION](docs/agents/VALIDATION.md)).
- **Mandatory Per-Request Commit:** Commit immediately after every completed request with concise English body: `(1) Request Source`, `(2) Actions`, `(3) Metrics & Verification`.

## 6. Definition of Done & Reference Docs
- Offline `pytest` passes 100% (`-m "not gpu"` via `verify_offline.py`), `agent_context.py --check` passes with 0 errors, `ruff check .` passes with 0 errors, and `git status` is clean.
- Real media runs (when user-approved) must also pass `verify_mux_integrity` (`"PASS"`) and exact `ffprobe -count_frames` equality.
- Architecture specs: [Video Pipeline](docs/video-pipeline.md) | [Dedup](docs/dedup-architecture.md) | [Audit & Safety](docs/audit-and-safety.md) | [Subtrans Roadmap](docs/subtrans-architecture-roadmap.md).
