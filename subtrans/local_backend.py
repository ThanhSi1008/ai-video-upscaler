# -*- coding: utf-8 -*-
"""
subtrans/local_backend.py
Stage 2b: Local fallback translation backend using Ollama (translategemma:12b).
"""

import re
import json
import logging
import requests
from typing import List, Dict, Any, Optional

from subtrans.models import Cue, Glossary
from subtrans.translator import BaseTranslator
from subtrans.validation import validate_translations, JAPANESE_CHAR_REGEX, CHATTER_REGEX

logger = logging.getLogger("subtrans.local")

OLLAMA_DEFAULT_URL = "http://localhost:11434"
LINE_PATTERN = re.compile(r"^\[?(\d+)\]?[\.\:\s\-]+(.*)$")


class OllamaTranslator(BaseTranslator):
    """Local translation backend utilizing Ollama API."""

    def __init__(
        self,
        endpoint: str = OLLAMA_DEFAULT_URL,
        model: str = "translategemma:12b",
        temperature: float = 0.0,
        num_ctx: int = 4096,
        timeout: int = 120,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout

    def is_available(self) -> bool:
        """Checks if Ollama server is running and model is pulled."""
        try:
            r = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                # Check exact or prefix match (e.g. translategemma:12b or translategemma:latest)
                return any(self.model in m or m.startswith(self.model.split(":")[0]) for m in models)
        except Exception:
            return False
        return False

    def unload(self):
        """Unloads the model from unified memory (keep_alive: 0)."""
        try:
            requests.post(
                f"{self.endpoint}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=5,
            )
            logger.info(f"Unloaded model {self.model} from memory.")
        except Exception as e:
            logger.debug(f"Failed to unload {self.model}: {e}")

    def _call_chat(self, messages: List[Dict[str, str]]) -> str:
        """Sends chat request to Ollama /api/chat."""
        url = f"{self.endpoint}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        res = requests.post(url, json=payload, timeout=self.timeout)
        res.raise_for_status()
        data = res.json()
        return data.get("message", {}).get("content", "").strip()

    def _format_glossary(self, glossary: Optional[Glossary], cues: List[Cue]) -> str:
        if not glossary:
            return ""

        batch_ja = " ".join(c.ja for c in cues)
        batch_en = " ".join(c.en for c in cues if c.en).lower()

        lines = []
        if glossary.terms:
            matched = []
            for t in glossary.terms:
                hit = False
                if t.ja and t.ja in batch_ja:
                    hit = True
                elif t.en and t.en.lower() in batch_en:
                    hit = True
                elif t.aliases:
                    for a in t.aliases:
                        if a in batch_ja or a.lower() in batch_en:
                            hit = True
                            break
                if hit:
                    alt_str = f" (or: {', '.join(t.alt)})" if t.alt else ""
                    note_str = f" [{t.note}]" if t.note else ""
                    matched.append(f"- {t.ja} -> {t.vi}{alt_str}{note_str}")

            if matched:
                lines.append("Glossary:")
                lines.extend(matched)

        if glossary.characters:
            matched_chars = []
            for c in glossary.characters:
                if (c.name_ja and c.name_ja in batch_ja) or (c.name_vi and c.name_vi.lower() in batch_en):
                    matched_chars.append(f"- {c.name_ja} -> {c.name_vi}")
            if matched_chars:
                lines.append("Characters:")
                lines.extend(matched_chars)

        return "\n".join(lines)

    def _translate_single(self, cue: Cue, glossary: Optional[Glossary]) -> str:
        """Fallback to translating a single cue if batching fails."""
        sys_msg = (
            "You are a professional anime subtitle translator from Japanese to Vietnamese. "
            "Translate the given line faithfully and naturally into Vietnamese. "
            "Output ONLY the translated line. Do not include notes, quotes, or markdown."
        )
        glossary_txt = self._format_glossary(glossary, [cue])
        user_content = f"Japanese: {cue.ja}"
        if cue.en:
            user_content += f"\nEnglish context: {cue.en}"
        if glossary_txt:
            user_content += f"\n\n{glossary_txt}"

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content},
        ]
        raw = self._call_chat(messages)
        # Clean quotes and chatter
        clean = re.sub(r'^["\']|["\']$', "", raw.strip())
        clean = CHATTER_REGEX.sub("", clean).strip()
        return clean or cue.ja

    def translate_batch(
        self,
        cues: List[Cue],
        context: List[Cue],
        glossary: Optional[Glossary] = None,
    ) -> List[str]:
        """Translates a batch of cues with fallback to single-cue on parsing failure."""
        if not cues:
            return []

        # Build numbered prompt
        prompt_lines = [
            "Translate each of the following numbered Japanese anime subtitle lines into natural Vietnamese.",
            "Use the exact format: [number] Vietnamese translation",
            "Do not merge or omit lines. Output nothing else.",
        ]
        glossary_txt = self._format_glossary(glossary, cues)
        if glossary_txt:
            prompt_lines.append("\n" + glossary_txt)

        prompt_lines.append("\nLines to translate:")
        for cue in cues:
            en_part = f" (EN: {cue.en})" if cue.en else ""
            prompt_lines.append(f"[{cue.id}] {cue.ja}{en_part}")

        messages = [
            {"role": "user", "content": "\n".join(prompt_lines)}
        ]

        try:
            raw_response = self._call_chat(messages)
            parsed_map: Dict[int, str] = {}
            for line in raw_response.splitlines():
                line = line.strip()
                if not line:
                    continue
                m = LINE_PATTERN.match(line)
                if m:
                    cid = int(m.group(1))
                    text = m.group(2).strip()
                    parsed_map[cid] = text

            expected_ids = [c.id for c in cues]
            # Check if all expected ids were parsed
            if all(cid in parsed_map for cid in expected_ids):
                items = [{"id": cid, "vi": parsed_map[cid]} for cid in expected_ids]
                is_valid, err = validate_translations(expected_ids, items)
                if is_valid:
                    return [it["vi"] for it in items]
                logger.warning(f"Batch validation failed ({err}). Falling back to cue-by-cue...")
            else:
                logger.warning(
                    f"Count mismatch in local batch: expected {len(expected_ids)}, "
                    f"got {len(parsed_map)}. Falling back to cue-by-cue..."
                )
        except Exception as e:
            logger.warning(f"Ollama batch request failed: {e}. Falling back to cue-by-cue...")

        # Fallback: cue-by-cue
        results = []
        for cue in cues:
            try:
                res = self._translate_single(cue, glossary)
                results.append(res)
            except Exception as ex:
                logger.error(f"Single cue translation failed for cue {cue.id}: {ex}")
                results.append(cue.ja)
        return results
