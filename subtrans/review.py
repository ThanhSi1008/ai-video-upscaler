# -*- coding: utf-8 -*-
"""
subtrans/review.py
Stage 2c: Optional local review/post-editing pass using Ollama (gemma3:12b).
Refines translated Vietnamese anime subtitles for natural flow and tone.
"""

import re
import logging
import requests
from typing import List, Dict, Optional

from subtrans.models import Cue, Glossary
from subtrans.local_backend import LINE_PATTERN, OLLAMA_DEFAULT_URL
from subtrans.validation import validate_translations

logger = logging.getLogger("subtrans.review")


class LocalReviewer:
    """Post-editing reviewer utilizing Ollama (gemma3:12b)."""

    def __init__(
        self,
        endpoint: str = OLLAMA_DEFAULT_URL,
        model: str = "gemma3:12b",
        temperature: float = 0.2,
        num_ctx: int = 4096,
        timeout: int = 120,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout

    def is_available(self) -> bool:
        """Checks if Ollama server is running and the review model is available."""
        try:
            r = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                return any(self.model in m or m.startswith(self.model.split(":")[0]) for m in models)
        except Exception:
            return False
        return False

    def unload(self):
        """Frees model from unified memory."""
        try:
            requests.post(
                f"{self.endpoint}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=5,
            )
            logger.info(f"Unloaded review model {self.model} from memory.")
        except Exception as e:
            logger.debug(f"Failed to unload {self.model}: {e}")

    def review_batch(
        self,
        cues: List[Cue],
        glossary: Optional[Glossary] = None,
    ) -> List[str]:
        """
        Reviews and refines Vietnamese translations in a batch.
        Returns a list of refined Vietnamese strings matching cues order.
        If review fails or errors occur, falls back to original cue.vi.
        """
        if not cues:
            return []

        expected_ids = [c.id for c in cues]
        original_vi = [c.vi or "" for c in cues]

        prompt_lines = [
            "You are an expert anime subtitle editor for Vietnamese.",
            "Review and polish the following Vietnamese translations for natural phrasing, fluency, and proper anime tone.",
            "Rules:",
            "1. Output format MUST be strictly: [number] Polished Vietnamese line",
            "2. Do not change the meaning or add new content.",
            "3. Keep subtitle length reasonable (around 42 characters per line, max 2 lines).",
            "4. Do NOT output notes or conversational text.",
        ]

        if glossary and glossary.characters:
            prompt_lines.append("\nCharacter Reference:")
            for ch in glossary.characters:
                prompt_lines.append(f"- {ch.name_vi} ({ch.name_ja})")

        prompt_lines.append("\nLines to review:")
        for c in cues:
            en_info = f" (EN: {c.en})" if c.en else ""
            prompt_lines.append(f"[{c.id}] JA: {c.ja}{en_info} | VI: {c.vi}")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "\n".join(prompt_lines)}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        try:
            res = requests.post(f"{self.endpoint}/api/chat", json=payload, timeout=self.timeout)
            res.raise_for_status()
            content = res.json().get("message", {}).get("content", "").strip()

            parsed_map: Dict[int, str] = {}
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                m = LINE_PATTERN.match(line)
                if m:
                    cid = int(m.group(1))
                    text = m.group(2).strip()
                    parsed_map[cid] = text

            if all(cid in parsed_map for cid in expected_ids):
                items = [{"id": cid, "vi": parsed_map[cid]} for cid in expected_ids]
                is_valid, _ = validate_translations(expected_ids, items)
                if is_valid:
                    return [it["vi"] for it in items]
        except Exception as e:
            logger.warning(f"Review pass failed ({e}), keeping original translations.")

        return original_vi
