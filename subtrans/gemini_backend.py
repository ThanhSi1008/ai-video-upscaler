# -*- coding: utf-8 -*-
"""
subtrans/gemini_backend.py
Stage 2: Primary translation backend using Google Gemini API with structured JSON output.
Adheres strictly to ADDENDUM 6 specifications:
- Priority 1: No mistranslation, flag uncertain: true when ambiguous
- Priority 2: Natural spoken Vietnamese
- Priority 3: Character names & pronouns consistency (Romaji, deliberate nicknames)
- Priority 4: Single line per cue, soft cap 80 chars, continuation split formatting
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from google import genai
from google.genai import types

from subtrans.models import Cue, Glossary
from subtrans.translator import BaseTranslator
from subtrans.validation import validate_translations
from subtrans.cache import execute_and_log_api_request

logger = logging.getLogger("subtrans.gemini")

SYSTEM_INSTRUCTION = (
    "You are an expert anime subtitle translator, translating Japanese (source of truth) "
    "with English reference context into natural, readable spoken Vietnamese subtitles.\n"
    "Strict Priorities:\n"
    "1. No mistranslation. When unsure, choose the simpler, safer rendering and set \"uncertain\": true instead of inventing meaning.\n"
    "2. Natural, readable spoken Vietnamese that sounds natural when read aloud (never word-for-word calques).\n"
    "3. Consistent names and pronouns within the episode.\n"
    "4. Format: Exactly ONE text line per cue (no line breaks). Soft cap: 80 characters.\n"
    "5. Punctuation: use '...' (three ASCII dots, never unicode ellipsis); use '—' at end of interrupted speech; '?!' allowed; max two consecutive ! or ?.\n"
    "6. Sentences split across cues: translate at sentence level, then divide across the SAME cues at natural phrase boundaries. "
    "The continuation cue MUST start in lowercase; the preceding fragment MUST end with ',' or '...'. Never move content beyond that split.\n"
    "7. Names and nicknames: keep Romaji spelling. Keep deliberate nicknames exactly as source uses them (e.g. 'Emilia-tan', 'Barusu', 'Ram').\n"
    "8. Register: archaic or divine (ta/ngươi/ngài, tiểu thư, chàng) vs modern casual (tôi/cậu/em/anh). Decide from character profile.\n"
    "Output must be a valid JSON array matching the provided schema."
)


class SubtitleOutputItem(BaseModel):
    id: int
    vi: str
    uncertain: bool = False


def filter_relevant_glossary(cues: List[Cue], glossary: Optional[Glossary]) -> Dict[str, Any]:
    """Filters glossary to only include terms, characters, and honorifics present in the current batch."""
    if not glossary:
        return {}

    batch_ja_text = " ".join(c.ja for c in cues)
    batch_en_text = " ".join(c.en for c in cues if c.en).lower()

    relevant_terms = []
    for term in glossary.terms:
        matches = False
        en_word_match = False
        if term.en and re.search(r"\b" + re.escape(term.en) + r"\b", batch_en_text, re.IGNORECASE):
            en_word_match = True

        if len(term.ja) == 1:
            # Single kanji terms (like 気 for ki/qi) need English context or alias match
            if en_word_match:
                matches = True
            elif term.aliases and any(re.search(r"\b" + re.escape(a) + r"\b", batch_en_text, re.IGNORECASE) for a in term.aliases):
                matches = True
            elif not batch_en_text and term.ja in batch_ja_text:
                matches = True
        else:
            if term.ja and term.ja in batch_ja_text:
                matches = True
            elif en_word_match:
                matches = True
            elif term.aliases:
                for alias in term.aliases:
                    if alias in batch_ja_text or re.search(r"\b" + re.escape(alias) + r"\b", batch_en_text, re.IGNORECASE):
                        matches = True
                        break

        if matches:
            item: Dict[str, Any] = {"ja": term.ja, "vi": term.vi}
            if term.en:
                item["en"] = term.en
            if term.alt:
                item["alt"] = term.alt
            if term.note:
                item["note"] = term.note
            relevant_terms.append(item)

    relevant_characters = []
    # Always include all episode cast characters to preserve pronoun continuity across batches,
    # as Japanese anime dialogues frequently omit subject names (zero-pronouns).
    for char in glossary.characters:
        relevant_characters.append({
            "name_ja": char.name_ja,
            "name_vi": char.name_vi,
            "speaks_to": [s.model_dump() for s in char.speaks_to]
        })

    res: Dict[str, Any] = {}
    if relevant_terms:
        res["terms"] = relevant_terms
    if relevant_characters:
        res["characters"] = relevant_characters

    if glossary.address_rules:
        addr = {}
        if glossary.address_rules.principle:
            addr["principle"] = glossary.address_rules.principle
        matching_honorifics = []
        for h in glossary.address_rules.honorifics:
            h_key = h.ja.replace("〜", "").replace("~", "").split("/")[0].strip()
            if h_key and (h_key in batch_ja_text or h_key.lower() in batch_en_text):
                matching_honorifics.append({"ja": h.ja, "vi": h.vi, "alt": h.alt, "note": h.note})
        if matching_honorifics:
            addr["relevant_honorifics"] = matching_honorifics
        if addr:
            res["address_guidance"] = addr

    return res


def get_thinking_config(model_name: str) -> Optional[types.ThinkingConfig]:
    """
    Configures lowest reasonable thinking setting for translation per ADDENDUM 8:
    - Minimizes latency and quota usage (thinking tokens count towards output token limit).
    - For flash-lite: uses ThinkingLevel.MINIMAL.
    - For standard 3.x/2.5 models: sets thinking_budget=0 to disable thinking.
    """
    m = model_name.lower()
    if "flash-lite" in m:
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
    else:
        return types.ThinkingConfig(thinking_budget=0)


class GeminiTranslator(BaseTranslator):
    """Google Gemini translation backend with structured schema output and automatic split-retry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.8-flash",
        temperature: float = 0.0,
        gloss_skill_names: bool = False,
        allow_latest_alias: bool = False,
        timeout_ms: int = 45000,
        http_options: Optional[types.HttpOptions] = None,
        dry_run: bool = False,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key and not dry_run:
            raise ValueError("GEMINI_API_KEY is required for GeminiTranslator.")
        self.timeout_ms = timeout_ms
        self.http_options = http_options or types.HttpOptions(timeout=int(self.timeout_ms))
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key, http_options=self.http_options)
        else:
            self.client = None
        self.model = model
        self.temperature = temperature
        self.gloss_skill_names = gloss_skill_names
        self.allow_latest_alias = allow_latest_alias
        self.dry_run = dry_run

        # Batch & request statistics
        self.stats = {
            "total_batches": 0,
            "split_batches": 0,
            "api_requests": 0,
        }

        # Protection against mutable alias 'gemini-flash-latest' per ADDENDUM 8
        if "latest" in self.model.lower() and not self.allow_latest_alias:
            raise ValueError(
                f"Model alias '{self.model}' is prohibited by default because its underlying version can change silently. "
                f"Please specify an explicit versioned model (e.g. gemini-3.8-flash) or set allow_latest_alias=True."
            )

    def _call_api(self, prompt_text: str, system_prompt: str = SYSTEM_INSTRUCTION) -> List[Dict[str, Any]]:
        """Invokes Gemini API requesting structured JSON array with minimal thinking overhead."""
        if self.dry_run:
            return []

        config_kwargs = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": list[SubtitleOutputItem],
            "temperature": self.temperature,
            "http_options": self.http_options,
        }
        thinking_cfg = get_thinking_config(self.model)
        if thinking_cfg:
            config_kwargs["thinking_config"] = thinking_cfg

        def _send():
            return self.client.models.generate_content(
                model=self.model,
                contents=prompt_text,
                config=types.GenerateContentConfig(**config_kwargs)
            )

        response = execute_and_log_api_request(
            _send,
            model=self.model,
            backend="gemini",
            cues_count=0,
            request_type="translation"
        )
        raw_text = response.text.strip()
        data = json.loads(raw_text)
        return data

    def translate_batch(
        self,
        cues: List[Cue],
        context: List[Cue],
        glossary: Optional[Glossary] = None
    ) -> List[str]:
        """Translates batch of cues (100–150 cues) with immediate bisecting (split in half) on cue/ID mismatch."""
        if not cues:
            return []

        self.stats["total_batches"] += 1

        # Chế độ Dry-Run: Không bao giờ gọi Gemini API (bảo toàn 100% quota đêm)
        if self.dry_run:
            self.stats["api_requests"] += 1
            for c in cues:
                c.uncertain = False
            return [f"[DRY-RUN VI] {c.en or c.ja}" for c in cues]

        expected_ids = [c.id for c in cues]
        payload = {
            "cues_to_translate": [
                {"id": c.id, "ja": c.ja, "en": c.en} if c.en else {"id": c.id, "ja": c.ja}
                for c in cues
            ]
        }

        # Add surrounding context (up to +/-10 cues per ADDENDUM 6)
        if context:
            payload["surrounding_context"] = [
                {"id": c.id, "ja": c.ja, "en": c.en, "context": True}
                for c in context
            ]

        # Add relevant glossary entries
        rel_glossary = filter_relevant_glossary(cues, glossary)
        if rel_glossary:
            payload["glossary"] = rel_glossary

        prompt_lines = [
            "Translate the following subtitle batch into Vietnamese.",
            "Follow character relationships, pronouns, and glossary strictly.",
            "When unsure of a rendering, set 'uncertain': true instead of inventing meaning.",
        ]
        if self.gloss_skill_names:
            prompt_lines.append(
                "SKILL NAMES: When a named attack/spell occurs and the EN reference provides the name, "
                "write in Vietnamese followed by the English name in parentheses, e.g. 'Độ Không Tuyệt Đối! (Absolute Zero)'."
            )
        prompt_lines.append(f"\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
        prompt_str = "\n".join(prompt_lines)

        # Build system prompt from glossary style_rules if provided
        system_prompt = SYSTEM_INSTRUCTION
        if glossary and glossary.style_rules:
            system_prompt = "\n".join(glossary.style_rules) + "\nOutput must be a valid JSON array matching the provided schema."

        # Attempt 1: Call API
        try:
            self.stats["api_requests"] += 1
            items = self._call_api(prompt_str, system_prompt=system_prompt)
            is_valid, err_msg = validate_translations(expected_ids, items)
            if is_valid:
                for c, it in zip(cues, items):
                    c.uncertain = bool(it.get("uncertain", False))
                return [it["vi"] for it in items]
            logger.warning(f"Batch validation failed on attempt 1 ({err_msg}).")
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["503", "429", "timeout", "timed out", "quota", "resource_exhausted", "unavailable", "500", "502", "504", "deadline"]):
                raise
            logger.warning(f"API call failed on attempt 1 ({e}).")

        # Tách đôi batch (Bisect): Khi số lượng cue > 1 và có lỗi validation hoặc API lỗi cục bộ
        if len(cues) > 1:
            self.stats["split_batches"] += 1
            mid = len(cues) // 2
            logger.warning(
                f"🔀 Tách đôi batch ({len(cues)} cues -> {mid} + {len(cues)-mid} cues) để đảm bảo độ tin cậy và khớp ID..."
            )
            left_half = self.translate_batch(cues[:mid], context, glossary)
            right_half = self.translate_batch(cues[mid:], context, glossary)
            return left_half + right_half

        # Nếu chỉ có 1 cue duy nhất mà thất bại, thử lại lần cuối với prompt nghiêm ngặt
        strict_prompt = (
            "CRITICAL: Output exactly one JSON item for the given cue ID. "
            "No empty strings, no leftover Japanese characters.\n"
            + prompt_str
        )
        self.stats["api_requests"] += 1
        items = self._call_api(strict_prompt, system_prompt=system_prompt)
        is_valid, err_msg = validate_translations(expected_ids, items)
        if is_valid:
            for c, it in zip(cues, items):
                c.uncertain = bool(it.get("uncertain", False))
            return [it["vi"] for it in items]

        raise RuntimeError(f"Failed to translate cue ID {cues[0].id} with Gemini backend after retries and split.")
