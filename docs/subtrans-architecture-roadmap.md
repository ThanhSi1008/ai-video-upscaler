# Subtrans Architecture Roadmap & Quantitative Spec (`subtrans`)

> **Engine:** Anime Subtitle Translation Engine (`subtrans/`)
> **Operating Policy:** **100% Cloud Gemini API (NO Ollama / Local LLMs)**
> **Pipeline Order:** **Eval Harness -> Schema & Context -> Linter -> Cascade Repair -> Hybrid RAG & Error Bank**

## 1. P0: Deterministic Evaluation Harness (`subtrans/eval/`)

1. **Regression Suite (`regression_suite.yaml`, 6 cases):**
   - Multi-speaker CC splitting (Kenshuu / *"Thuần phi Kin"* in *Futsutsuka* Ep 07)
   - Strict zero-overlap `40ms` gap preventing 3-4 line overflow
   - SQLite cache isolation rejecting `[DRY-RUN]` / `[Chưa dịch]` pollution
   - Romaji vocative name retention
   - Anime laughter detection (`ふはははは!`) avoiding false EN alignment
   - Dual Dialogue duplicate-variant suppression (`>= 15` common chars or `>= 60%` token overlap)
2. **Pronoun & Relationship Suite (`pronoun_suite.yaml`):** Annotated `(speaker, addressee, mode, allowed_pronouns)` benchmark cues.
3. **Deterministic Metrics (`metrics.py`):**
   - **Pronoun Accuracy (%)**, **Romaji Retention (%)**, **CJK Leakage (%)**
   - **CPS Violation (%):** Netflix target `<= 17 CPS`; Anime ceiling `<= 22-25 CPS`
   - **Zero-Overlap Compliance (%):** `c.end <= next.start - 40ms`

## 2. P1: Think-Before-Translate Schema & Linter (`subtrans/models.py`, `linter.py`)

Preserving JSON schema property order forces context reasoning before emitting `vi`:

```python
class CueOutputItem(BaseModel):
    id: int
    speaker: str
    addressee: Optional[str]
    mode: Literal[
        "spoken", "inner", "narration", "spell",
        "flashback", "comm", "lyrics", "onscreen"
    ]
    en_conflict: bool
    vi: str
    uncertain: Literal[
        "none", "pronoun", "referent", "wordplay",
        "archaic", "term", "other"
    ]
```

Post-pass checks (`linter.py`): CPS bounds, CJK/Katakana leakage, Unicode NFC normalization, Romaji vocatives, `RULE_GLOSSARY_TERM`, and `RULE_PRONOUN_CONSISTENCY`.

## 3. P2: Smart Cloud Cascade (`subtrans/cascade.py`)

- **Draft Pass:** `gemini-3.5-flash-lite` batch translation (`thinking_level=MINIMAL`).
- **Repair Pass:** Routes high-difficulty $S(\text{cue})$ or `uncertain != "none"` cues to `gemini-3.8-flash` (`thinking_level=LOW`) in an editing review pass with `digit_mismatch` quarantine.

## 4. P3: Hybrid BM25 + FTS5 RAG & Error Bank (`subtrans/rag.py`)

- Local SQLite WAL knowledge base (`subtrans/.cache/knowledge.db`) storing `gold_units` (seeded with `BUILTIN_GOLD_EXEMPLARS` via `rag seed`, plus curated `silver` units) and `error_bank` cautions.
- Uses SQLite FTS5 2-gram/3-gram character tokenization for 2-character Kanji terms (`魔術`, `聖杯`, `令呪`) + role-pair BM25 scoring.
