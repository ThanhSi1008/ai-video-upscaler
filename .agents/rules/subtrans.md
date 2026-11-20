---
trigger: glob
globs: "subtrans/**/*, glossaries/*.yaml"
description: "Anime subtitle translation rules synced with subtrans-anime-ops: VI pronoun consistency, glossary pointer precedence, FTS5/RAG cache safety, offline discipline, and subtrans-evaluator delegation."
---

# Subtrans & Glossary Scoped Rules

- **Domain Data & Pronoun Consistency:**
  - Strictly preserve VI/JA linguistic data, pronoun regexes (`subtrans/linter.py`), LLM prompts (`subtrans/translator.py`), eval benchmarks (`subtrans/eval/*.yaml`), and MKV title `'Tiếng Việt (Vietsub)'`.
  - Enforce canonical pair pronouns and character vocal profiles across episode batches via [subtrans-anime-ops](../skills/subtrans-anime-ops/SKILL.md).
- **Glossary Precedence & Text Pointers:**
  - Resolution order: explicit `-g` -> `glossaries/series_registry.yaml` auto-discovery -> sidecar `glossary.yaml` -> `subtrans/glossary.yaml`.
  - Maintain alias files in `glossaries/*.yaml` (`base.yaml`, `shin_chan.yaml`, `doraemon.yaml`) as Git `100644` single-line relative text pointers (`_resolve_yaml_pointer_path`), never OS symlinks.
- **SQLite & FTS5/RAG Cache Safety:**
  - Never instantiate default `TranslationCache()` or `KnowledgeBase()` against live `subtrans/.cache/` in tests; always inject `tmp_path` SQLite paths.
- **Quota Protection & Offline Discipline:**
  - Never read, print, or log `.env` or `GEMINI_API_KEY*`.
  - Block live Gemini API calls (`subtrans.py`, `scripts/batch_translate_all.py`) unless `--dry-run`, `--offline`, `--help`, or mock/test flags are passed.
- **Subagent Delegation:**
  - Delegate subtitle evaluations (`subtrans/eval/runner.py`, `benchmarks/subtrans/eval_golden_pronouns*.py`) to [subtrans-evaluator](../agents/subtrans-evaluator.md) and pytest runs to [test-verifier](../agents/test-verifier.md).
