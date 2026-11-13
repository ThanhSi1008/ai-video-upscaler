---
name: subtrans-anime-ops
description: >-
  Develop, test, and operate the Anime Subtitle Translation Engine (subtrans/, glossaries/).
  Activate when creating per-series glossaries, tuning Gemini 3-key translation/cascade/RAG/linter,
  debugging Whisper 5.1 FC + DTW auto-align, or embedding Vietnamese softsubs into MKV containers.
---

# Subtrans Anime Translation & Glossary Runbook

Always read [subtrans/AGENTS.md](../../../subtrans/AGENTS.md) before modifying `subtrans/`.

## 1. Strict Safety & Quota Rules

1. **100% Gemini Cloud API Only:** `gemini-3.5-flash-lite` (default) and `gemini-3.8-flash` (high / cascade repair). Never enable Ollama or local LLMs.
2. **Protect `.env` & Daytime Quota:**
   - Never read, print, or log `.env` or `GEMINI_API_KEY*`.
   - Never call real Gemini APIs between **14:00 and overnight runs** (including `subtrans.py check` and `subtrans.py models`, which make live network requests!).
3. **Dry-Run Isolation:** `--dry-run` skips network calls but still writes cast sheets, audit reports, and telemetry. Always direct dry-run outputs to `scratch/dry-run/`:
   ```powershell
   .\.venv\Scripts\python.exe subtrans.py run subtrans/samples/sample.ja.srt -o scratch/dry-run/ --dry-run
   ```

## 2. Authoring a Per-Series Glossary (`glossaries/glossary.<series>.yaml`)

1. Inspect [glossaries/README.md](../../../glossaries/README.md) and an existing high-standard glossary such as `glossaries/glossary.fate.yaml` or `glossaries/glossary.tenken.yaml`.
2. Define:
   - **Character Pronouns & Relationships (`address_rules`)**: Exact `self_pronoun` / `target_pronoun` pairs by speaker -> addressee and social register.
   - **Romaji Vocative Retention**: Proper names (`Fran`, `Shirou`, `Sakura`, `Nell`) MUST remain in Vietnamese cues when spoken as vocatives in Japanese (e.g. `chị Nell`, `Shirou, cẩn thận!`), never flattened into generic pronouns (`cậu`, `anh`, `em`).
   - **Canonical Terms (`terms`)**: Skills, noble phantasms, titles, factions, and banned Sino-Vietnamese calques.

## 3. Whisper Fallback & Timeline Alignment

- Multi-channel 5.1 Japanese audio extracts center dialogue via `pan=mono|c0=0.7*FC+0.15*FL+0.15*FR,highpass=f=80,lowpass=f=8000,afftdn=nf=-25` (`src/subtitles.py`).
- Hallucinations (`ご視聴ありがとうございました`, token screams) are filtered by `WHISPER_HALLUCINATION_REGEX` and `SCREAM_OR_REPEAT_LOOP_REGEX` (`subtrans/parser.py`).
- `--auto-align-en` snaps Whisper cues to studio English timestamps via monotonic DTW (`subtrans/dtw_auto_align.py`) while preserving Dual Dialogue (`- Line 1\n- Line 2`).

## 4. Embedding Softsubs into MKV

When muxing translated `.vi.srt` into MKV containers (`src/ffmpeg_ops.py:embed_subtitle_track`), preserve all original video, multi-channel audio, existing subtitle tracks, attachments (fonts), and chapters using `-c copy -max_interleave_delta 0`, then verify via `verify_mux_integrity`.
