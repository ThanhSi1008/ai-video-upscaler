---
trigger: glob
globs: "subtrans.py, subtrans/**/*, glossaries/*.yaml, scripts/batch_translate_all.py"
description: "Anime subtitle translation rules synced with subtrans-anime-ops: VI pronoun consistency, glossary pointer precedence, FTS5/RAG cache safety, offline discipline, and subtrans-evaluator delegation."
---

# Subtrans & Glossary Scoped Rules

- **Domain Data, Comments & Pronoun Consistency:**
  - Strictly preserve VI/JA linguistic data, pronoun regexes (`subtrans/linter.py`), LLM prompts (`subtrans/prompts.py`, re-exported by `subtrans/gemini_backend.py`), eval benchmarks (`subtrans/eval/*.yaml`), and MKV title `'Tiếng Việt (AI Vietsub)'`.
  - Follow the AI-Only Comment Standard on all code edits (concise English 1-line docstrings and invariant `#` comments; zero `# -*- coding: utf-8 -*-` or decorative banners).
  - Enforce canonical pair pronouns and character vocal profiles across episode batches via [subtrans-anime-ops](../skills/subtrans-anime-ops/SKILL.md).
- **Glossary Precedence & Text Pointers:**
  - Always load `subtrans/glossary.yaml` (`DEFAULT_BASE_GLOSSARY_PATH`) as `base`, select `override` via explicit `-g` else `find_series_glossary()` (`glossaries/glossary.<series>.yaml`), and merge with `merge_glossaries(base, override)` (`override` wins on matching terms/characters while `AddressRules` deep-merges).
  - Maintain alias files in `glossaries/*.yaml` (`base.yaml`, `glossary.demonslayer.yaml`, `glossary.reincarnatedasasword.yaml`, `glossary.tenseishitarakendeshita.yaml`) as Git `100644` single-line relative text pointers resolved via `resolve_glossary_target()`, never OS symlinks.
- **SQLite & FTS5/RAG Cache Safety:**
  - Never instantiate default `TranslationCache()` or `KnowledgeBase()` against live `subtrans/.cache/` in tests; always inject `tmp_path` SQLite paths.
- **Secrets, Quota & Subagent Delegation:**
  - Local reading and editing of `.env` / `subtrans/.env` is allowed freely; never commit, export, or print keys to logs.
  - Block live Gemini API calls (`subtrans.py`, `scripts/batch_translate_all.py`, `benchmarks/subtrans/eval_golden_pronouns*.py`) unless `--dry-run`, `--help`, local subcommands (`eval`, `compare`, `rag stats`), or mock/test flags are passed.
  - Delegate subtitle evals (`subtrans/eval/runner.py`, `benchmarks/subtrans/eval_golden_pronouns*.py`) to [subtrans-evaluator](../agents/subtrans-evaluator.md) and offline pytest runs (`verify_offline.py --target` / `--impacted`) to [test-verifier](../agents/test-verifier.md).
