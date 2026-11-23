---
name: subtrans-anime-ops
description: >-
  Develop, test, and operate the Anime Subtitle Translation Engine (subtrans/, glossaries/).
  Activate when creating per-series glossaries, tuning Gemini 3-key translation/cascade/RAG/linter,
  debugging Whisper 5.1 FC + DTW auto-align, or embedding Vietnamese softsubs into MKV containers.
---

# Subtrans Anime Translation & Glossary Runbook

> Reference baseline: `subtrans/` architecture, [Subtrans Roadmap](../../../docs/subtrans-architecture-roadmap.md), and [GEMINI.md](../../../GEMINI.md).

## 1. Strict Cloud-Only, Quota & Cache Rules
1. **100% Gemini Cloud API Only (NO Local LLMs):**
   - Primary (`--subs-quality normal`/`fast`): `gemini-3.5-flash-lite` (`thinking_level=MINIMAL`).
   - High quality (`--subs-quality high`) & Cascade repair: `gemini-3.8-flash` (`thinking_level=LOW`).
   - Fallback chain in `subtrans/models.yaml:chain` (`gemini-3.8/3.7/3.6/3.5-flash`). Never default to `gemini-2.*`, never use `gemini-flash-latest` without `--allow-latest-alias`, and never enable Ollama/TranslateGemma (`--local`/`--review` are rejected with non-zero exit).
   - All GenAI calls (`gemini_backend.py`, `cast_sheet.py`, `cascade.py`) attach `get_anime_safety_settings()` (`BLOCK_NONE`), `extract_genai_response_text()`, and `max_output_tokens=16384`.
2. **Protect `.env` & Daytime Quota:**
   - Local reading and editing of `.env` / `subtrans/.env` is allowed freely (free-tier keys); never commit, export, or print `GEMINI_API_KEY*` to logs.
   - Never call live Gemini APIs between **14:00 and overnight runs** (including `subtrans.py check` and `subtrans.py models`, which perform live requests!).
3. **Multi-Key Round-Robin & 429 Handling:**
   - Rotates across `GEMINI_API_KEYS`, `GEMINI_API_KEY`, and `GEMINI_API_KEY_1..N` in natural numeric order (modulo + least-loaded).
   - `429 PerMinute` / `503`: retries with `retryDelay`/backoff+jitter and rotates keys per `gemini_backend._call_api` & `throttle.call_with_retry`.
   - `429 PerDay`: marks `(model, key_hash)` exhausted until `00:00 PT` (`14:00 PDT` / `15:00 PST` VN via `cache.get_next_pt_reset`), stored in `subtrans/.cache/models_runtime_quota.yaml` (never dirtying `subtrans/models.yaml`). When all keys & models exhaust RPD, performs a graceful exit while preserving partial translations in SQLite.
4. **Quality Gate & Dry-Run Isolation:**
   - Quality gate (`pipeline.py`): requires `>= 95%` real translations and `< 10%` uncertain; otherwise outputs `.vi.partial.*` and queues in `state/pending_subs.json` (never written under `--dry-run`).
   - Never store `[DRY-RUN]` or `[Chưa dịch]` in SQLite cache (`subtrans/.cache/translations.db`). Both `TranslationCache` and `KnowledgeBase` use SQLite `WAL` mode. Always isolate dry-runs:
     ```powershell
     .\.venv\Scripts\python.exe subtrans.py run subtrans/samples/sample_30s_jpn.ass --en subtrans/samples/sample_30s_eng.ass -o scratch/dry-run/ --dry-run
     ```

## 2. Translation Fidelity, SRT Formatting & Vocative Rules
- **Source of Truth & Cast Sheet:** Japanese audio/text is the Source of Truth; English assists only with zero-pronoun subjects/addressees. Stage A Cast Sheet (`out/<name>.cast.yaml`) is unconditionally injected into every batch prompt.
- **Romaji Vocative Retention (CRITICAL):** When Japanese dialogue addresses someone by proper name (e.g., `フラン` -> `Fran`, `ジャン` -> `Jean`, `ネルさん` -> `chị Nell`, `玲琳` -> `Reirin`), the Vietnamese cue MUST keep the Romaji proper name (e.g., `"Fran, cẩn thận!"`, `"Cảm ơn chị Nell"`)—never replace it with bare pronouns (`cậu`, `anh`, `em`, `mày`).
- **Punctuation, Wrapping & Quotes:**
  - Use ASCII 3-dot ellipsis `...` (not `…`), em-dash `—` for interrupted speech, and at most 2 consecutive `!` or `?` (`!!`, `??`).
  - Strip outer quotes only when wrapping the full cue (`"Xin chào!"` -> `Xin chào!`); preserve inner single quotes (`'Sư phụ' của cô...`).
  - `cli.run` and `TranslationPipeline` default `smart_wrap=True` (`writer.write_subtitles` wraps single cues `> 72` chars at punctuation/conjunctions into max 2 lines; `--no-smart-wrap` keeps 1 line; `--wrap N` overrides; direct `write_srt_subtitles` defaults `smart_wrap=False`).

## 3. Authoring a Per-Series Glossary (`glossaries/glossary.<series>.yaml`)
1. Inspect [glossaries/README.md](../../../glossaries/README.md) and a reference glossary (`glossaries/glossary.fate.yaml` or `glossaries/glossary.tenken.yaml`). Note: `glossaries/base.yaml` is a cross-platform text pointer to `subtrans/glossary.yaml`.
2. Define `address_rules` (`self` / `other` per speaker -> addressee & register), `characters` (including `aliases`), and canonical `terms` (avoiding unwanted Han-Viet calques).

## 4. Whisper Fallback, CC Splitting & DTW Alignment
- **5.1 FC Extraction & STT:** Multi-channel Japanese audio extracts center dialogue via `pan=mono|c0=0.7*FC+0.15*FL+0.15*FR,highpass=f=80,lowpass=f=8000,afftdn=nf=-25` (`src/subtitles.py`) using `faster-whisper` (`large-v3-turbo`) + Silero VAD + Glossary `initial_prompt` (plus `audio_diarization.py` 8kHz CMS timbre + F0 `audio_turn` hints for plain `.srt`).
- **Noise & Hallucination Filters (`subtrans/parser.py`):** `WHISPER_HALLUCINATION_REGEX` removes OP/ED/silence hallucinations (`ご視聴ありがとうございました`, `チャンネル登録`, `提供`, etc.); `SCREAM_OR_REPEAT_LOOP_REGEX` & anime laughter interjections (`ふはは+`, `ははは+`, `フハハ+`, `ククク+`) mark non-speech cues so they are dropped under `--sound-cues drop` and never mis-aligned to EN dialogue.
- **Multi-Speaker CC & Dual Dialogue:** `split_multi_speaker_cc_events` splits `-（Speaker1）... \N -（Speaker2）...` into separate cues (`Cue.speaker`); `writer.optimize_cues_for_playback` merges simultaneous cues into 2-line Dual Dialogue (`- Line 1\n- Line 2`, deduplicating variants with `>= 15` common prefix chars or `>= 60%` token overlap) and enforces `>= 1.2s–1.4s` reading time + `40ms` anti-flicker gap.
- **Monotonic DTW Auto-Align:** `--auto-align-en` snaps Whisper cues (filenames containing `whisper` or `is_whisper=True`; studio JA keeps source timeline after offset) via `dtw_auto_align.py` (1:1, 1:2 `>= 250ms` pause split, 2:1), dropping unmapped orphan cues `> 2500ms` from nearest EN or `<= 6` chars / sound keywords. Supports Piecewise 2-Segment Auto-Offset.

## 5. Subsystem Architecture & Commands
| Module | Responsibility |
| :--- | :--- |
| `cli.py`, `pipeline.py`, `config.py` | Typer CLI (`run`, `batch`, `eval`, `rag`), 4-stage pipeline, scene-cut clamp `>= 4500ms`, `StageAPrefetchWorker` |
| `prompts.py`, `gemini_backend.py`, `translator.py`, `throttle.py`, `cache.py` | `> 4096`-token static prefix Implicit & `--explicit-cache`, CoT `p1`/`p2`, Contextual Cache `v2.2`, 3-key rotation & RPD tracking |
| `cast_sheet.py`, `dynamic_cast.py`, `ja_morph_tagger.py`, `glossary.py` | Stage A `.cast.yaml`, `HysteresisCastManager`, MeCab `fugashi`+`unidic-lite` (zero Kanji-compound false positives), glossary merger |
| `parser.py`, `dtw_auto_align.py`, `audio_diarization.py`, `writer.py` | ASS/SRT parsing, CC split, DTW alignment, Dual Dialogue, smart-wrap, zero-overlap |
| `linter.py`, `cascade.py`, `rag.py`, `report.py`, `eval/` | Linter V2 (CJK, CPS, NFC, Romaji vocative, glossary/pronoun), Flash cascade w/ digit quarantine, Hybrid BM25 + FTS5 2/3-gram RAG |

- **MKV Embedding:** `src/ffmpeg_ops.py:embed_subtitle_track` preserves all original streams/attachments/chapters with `-c copy -max_interleave_delta 0`, verified by `verify_mux_integrity`.
- **Offline Tests:** `.\.venv\Scripts\python.exe scripts/verify_offline.py` (or targeted `subtrans` suite inside disposable snapshot; pure help check: `.\.venv\Scripts\python.exe subtrans.py --help`).
