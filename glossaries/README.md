# Anime Subtitle Translation Glossaries

Per-series and base glossaries (Schema v2) for `subtrans`:

1. **`base.yaml`** (cross-platform text pointer to `../subtrans/glossary.yaml`):
   - Shared base glossary loaded automatically for every translation run:
     - Style rules (`style_rules`): Japanese source-of-truth, natural spoken Vietnamese, Romaji vocative retention, smart-wrap `> 72` chars.
     - Pronoun & honorific rules (`address_rules`): 1st/2nd person, honorifics, relationship templates.
     - 230+ common anime terms (titles, school, romcom, isekai, shounen, martial arts, xianxia, cuisine).

2. **Series-Specific Glossaries (`glossary.<series>.yaml`)**:
   - Entries and characters in series glossaries override base entries (`glossaries/glossary.reincarnatedasasword.yaml` and `glossary.tenseishitarakendeshita.yaml` alias `glossary.tenken.yaml`; `glossary.demonslayer.yaml` aliases `glossary.kimetsu.yaml`).
   - Usage:
     ```powershell
     .\.venv\Scripts\python.exe subtrans.py run anime_ep01.ass -g glossaries/glossary.rezero.yaml
     ```
