# -*- coding: utf-8 -*-
"""
subtrans/cast_sheet.py
Stage A: Cast Sheet generation and persistence.
Analyzes episode dialogue once per episode to standardize character names,
nicknames (e.g. Emilia-tan, Barusu), register (archaic/divine vs modern casual),
and speaker-to-listener pronoun consistency.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
from google import genai
from google.genai import types

from subtrans.models import Cue, Character, CharacterSpeaksTo, Glossary
from subtrans.config import get_gemini_api_key
from subtrans.gemini_backend import get_thinking_config

logger = logging.getLogger("subtrans.cast_sheet")

CAST_PROMPT_TEMPLATE = """
You are an expert anime subtitle director. Analyze the following anime dialogue lines and determine the cast members, their character relationships, registers, and Vietnamese pronouns.

Rules:
1. Names: Keep Romaji spelling exactly as in English subtitles (e.g. Subaru, Emilia, Ram, Rem, Shaula, Reid, Volcanica, Farsale).
2. Nicknames: Keep deliberate nicknames intact (e.g. "Emilia-tan", "Barusu").
3. Register:
   - Archaic, divine, dragons, royalty, powerful elders: ta / ngươi / ngài / bệ hạ / tiểu thư / chàng.
   - Modern casual, peers, teens: tôi / cậu / tớ / mình / em / anh / mày / tao.
4. Output MUST be valid JSON matching this schema:
{
  "characters": [
    {
      "name_ja": "Japanese name if known or empty",
      "name_vi": "Romaji name",
      "gender": "male | female | other",
      "register": "archaic | divine | noble | modern casual",
      "speaks_to": [
        {
          "listener": "Target character name or 'all'",
          "self": "tôi | ta | tớ | em ...",
          "other": "cậu | ngươi | ngài | anh ...",
          "note": "Short rationale"
        }
      ]
    }
  ]
}

Dialogue samples from the episode:
"""


def generate_cast_sheet(
    cues: List[Cue],
    output_path: Path,
    gemini_model: str = "gemini-3.8-flash",
) -> Optional[Glossary]:
    """
    Stage A: Discovers characters and pronoun tables across the episode.
    Persists cast sheet to output_path and returns a Glossary object.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        logger.info("GEMINI_API_KEY not found; skipping automatic Cast Sheet generation.")
        return None

    # Sample dialogue cues: up to 60 distinct dialogue cues across the episode
    dialogue_cues = [c for c in cues if not c.is_skipped and (c.ja or c.en)]
    step = max(1, len(dialogue_cues) // 60)
    sampled = dialogue_cues[::step][:60]

    sample_lines = []
    for c in sampled:
        en_part = f" | EN: {c.en}" if c.en else ""
        sample_lines.append(f"[{c.id}] JA: {c.ja}{en_part}")

    prompt = CAST_PROMPT_TEMPLATE + "\n".join(sample_lines)

    try:
        client = genai.Client(api_key=api_key)
        cfg_args = {
            "response_mime_type": "application/json",
            "temperature": 0.1,
        }
        thinking_cfg = get_thinking_config(gemini_model)
        if thinking_cfg:
            cfg_args["thinking_config"] = thinking_cfg

        res = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg_args),
        )

        data = json.loads(res.text.strip())
        chars_data = data.get("characters", [])

        characters = []
        for cd in chars_data:
            speaks = [
                CharacterSpeaksTo(
                    listener=s.get("listener", "all"),
                    self=s.get("self", "tôi"),
                    other=s.get("other", "cậu"),
                    note=s.get("note"),
                )
                for s in cd.get("speaks_to", [])
            ]
            characters.append(
                Character(
                    name_ja=cd.get("name_ja", ""),
                    name_vi=cd.get("name_vi", ""),
                    gender=cd.get("gender"),
                    speaks_to=speaks,
                )
            )

        cast_glossary = Glossary(characters=characters)

        # Persist to YAML
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "meta": {"schema": 2, "description": "Auto-generated Episode Cast Sheet"},
                    "characters": [c.model_dump() for c in characters],
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )

        return cast_glossary

    except Exception as e:
        logger.warning(f"Failed to generate Stage A Cast Sheet: {e}")
        return None
