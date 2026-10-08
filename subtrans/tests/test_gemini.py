# -*- coding: utf-8 -*-
"""
subtrans/tests/test_gemini.py
Unit tests for Gemini translation backend with mocking and comprehensive scenario checks:
- Glossary filtering
- Translation success & retry/bisect logic
- Dry-run behavior (zero API cost)
- Schema reasoning (CoT) metadata assignment
- RAG few-shot examples & error cautions payload injection
- Empty translation fallback for anime sound cues
- Thinking configuration across model families
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from subtrans.models import Cue, Glossary, GlossaryTerm
from subtrans.gemini_backend import (
    GeminiTranslator,
    filter_relevant_glossary,
    get_thinking_config,
    SYSTEM_INSTRUCTION,
)


def test_filter_relevant_glossary():
    """Verifies filtering of glossary terms based on presence in Japanese/English cues."""
    cues = [
        Cue(id=1, start=0, end=1000, ja="魔法少女になりたい", en="I want to become a magical girl")
    ]
    glossary = Glossary(
        terms=[
            GlossaryTerm(ja="魔法少女", en="magical girl", vi="ma pháp thiếu nữ"),
            GlossaryTerm(ja="冒険者", en="adventurer", vi="mạo hiểm giả"),  # Not present
        ]
    )
    filtered = filter_relevant_glossary(cues, glossary)
    assert "terms" in filtered
    assert len(filtered["terms"]) == 1
    assert filtered["terms"][0]["ja"] == "魔法少女"


def test_gemini_translator_success():
    """Verifies normal batch translation returning structured outputs."""
    translator = GeminiTranslator(api_key="mock_key")
    mock_response = MagicMock()
    mock_response.text = '[{"id": 1, "vi": "Chào bạn"}, {"id": 2, "vi": "Hẹn gặp lại"}]'

    translator.client = MagicMock()
    translator.client.models.generate_content.return_value = mock_response

    cues = [
        Cue(id=1, start=0, end=1000, ja="こんにちは"),
        Cue(id=2, start=1000, end=2000, ja="さようなら"),
    ]
    res = translator.translate_batch(cues, [])
    assert res == ["Chào bạn", "Hẹn gặp lại"]


def test_gemini_translator_dry_run_mode():
    """Verifies that dry_run=True performs zero API calls and returns simulated strings."""
    translator = GeminiTranslator(api_key="mock_key", dry_run=True)
    translator.client = MagicMock()

    cues = [
        Cue(id=1, start=0, end=1000, ja="こんにちは", en="Hello"),
        Cue(id=2, start=1000, end=2000, ja="さようなら", en="Goodbye"),
    ]
    res = translator.translate_batch(cues, [])
    assert res == ["[DRY-RUN VI] Hello", "[DRY-RUN VI] Goodbye"]
    # Ensure client was never called
    translator.client.models.generate_content.assert_not_called()
    assert cues[0].uncertain is False
    assert cues[1].uncertain is False


def test_gemini_translator_retry_and_split():
    """Verifies that bad/mismatched translations trigger retry and bisection splitting."""
    translator = GeminiTranslator(api_key="mock_key")
    translator.client = MagicMock()

    # First call: returns invalid response (leftover Japanese)
    bad_response = MagicMock()
    bad_response.text = '[{"id": 1, "vi": "こんにちは"}, {"id": 2, "vi": "さようなら"}]'

    # Second call (retry): also returns invalid
    bad_response2 = MagicMock()
    bad_response2.text = '[{"id": 1, "vi": "invalid"}, {"id": 2, "vi": "まだ日本語"}]'

    # Halved call 1: returns valid for id 1
    good_half1 = MagicMock()
    good_half1.text = '[{"id": 1, "vi": "Chào bạn"}]'

    # Halved call 2: returns valid for id 2
    good_half2 = MagicMock()
    good_half2.text = '[{"id": 2, "vi": "Tạm biệt"}]'

    translator.client.models.generate_content.side_effect = [
        bad_response,
        bad_response2,
        good_half1,
        good_half2,
    ]

    cues = [
        Cue(id=1, start=0, end=1000, ja="こんにちは"),
        Cue(id=2, start=1000, end=2000, ja="さようなら"),
    ]
    res = translator.translate_batch(cues, [])
    assert res == ["Chào bạn", "Tạm biệt"]


def test_vocative_proper_name_prompt_instruction():
    """Verifies that system prompt contains critical vocative and proper name instructions."""
    assert "Addressing by proper name (Vocative)" in SYSTEM_INSTRUCTION
    assert "ALWAYS KEEP the Romaji proper name directly in the Vietnamese dialogue line" in SYSTEM_INSTRUCTION
    assert "Context-First Reasoning (Chain-of-Thought)" in SYSTEM_INSTRUCTION


def test_gemini_schema_reasoning_metadata():
    """Verifies extraction and assignment of CoT reasoning fields from JSON output."""
    translator = GeminiTranslator(api_key="mock_key")
    mock_response = MagicMock()
    mock_response.text = """[
        {
            "id": 1,
            "speaker": "Fran",
            "addressee": "Teacher",
            "mode": "comm",
            "en_conflict": true,
            "vi": "Sư phụ, em đói rồi.",
            "uncertain": "none"
        },
        {
            "id": 2,
            "speaker": "Teacher",
            "addressee": "Fran",
            "mode": "spoken",
            "en_conflict": false,
            "vi": "Được rồi, đợi một chút nhé.",
            "uncertain": "pronoun"
        }
    ]"""

    translator.client = MagicMock()
    translator.client.models.generate_content.return_value = mock_response

    cues = [
        Cue(id=1, start=0, end=1000, ja="師匠、お腹空いた", en="Master, hungry"),
        Cue(id=2, start=1000, end=2000, ja="よし、ちょっと待ってくれ", en="Alright, wait a bit"),
    ]
    res = translator.translate_batch(cues, [])
    assert res == ["Sư phụ, em đói rồi.", "Được rồi, đợi một chút nhé."]

    # Verify reasoning CoT fields were assigned to Cue objects
    assert cues[0].speaker == "Fran"
    assert cues[0].addressee == "Teacher"
    assert cues[0].mode == "comm"
    assert cues[0].en_conflict is True
    assert cues[0].uncertain is False
    assert cues[0].uncertain_reason == "none"

    assert cues[1].speaker == "Teacher"
    assert cues[1].addressee == "Fran"
    assert cues[1].mode == "spoken"
    assert cues[1].en_conflict is False
    assert cues[1].uncertain is True
    assert cues[1].uncertain_reason == "pronoun"


def test_gemini_translator_with_rag_payload_injection():
    """Verifies that RAG few-shot examples and error cautions are embedded into the prompt payload."""
    translator = GeminiTranslator(api_key="mock_key")
    translator.client = MagicMock()

    captured_prompt = {}

    def mock_generate_content(model, contents, config):
        captured_prompt["contents"] = contents
        res = MagicMock()
        res.text = '[{"id": 1, "vi": "Sư phụ, em muốn ăn cà ri"}]'
        return res

    translator.client.models.generate_content.side_effect = mock_generate_content

    # Mock KnowledgeBase retrieval
    with patch("subtrans.rag.KnowledgeBase") as MockKB:
        mock_kb_instance = MockKB.return_value
        mock_kb_instance.retrieve_few_shots.return_value = [
            {"ja": "師匠、カレー", "en": "Master, curry", "vi": "Sư phụ, cà ri"}
        ]
        mock_kb_instance.retrieve_cautions.return_value = [
            {"category": "vocative_name", "rule": "Keep Fran", "avoid": "em", "prefer": "Fran"}
        ]

        cues = [Cue(id=1, start=0, end=1000, ja="師匠、カレー食べたい")]
        res = translator.translate_batch(cues, [])
        assert res == ["Sư phụ, em muốn ăn cà ri"]

    assert "contents" in captured_prompt
    prompt_str = captured_prompt["contents"]
    assert "approved_reference_examples" in prompt_str
    assert "error_prevention_cautions" in prompt_str
    assert "Sư phụ, cà ri" in prompt_str
    assert "Keep Fran" in prompt_str


def test_gemini_sound_cue_fallback():
    """Verifies that empty translation for sound cues (e.g. （ため息）) falls back to (âm thanh)."""
    translator = GeminiTranslator(api_key="mock_key")
    mock_response = MagicMock()
    # LLM returns empty string for sound cue
    mock_response.text = '[{"id": 1, "vi": ""}]'

    translator.client = MagicMock()
    translator.client.models.generate_content.return_value = mock_response

    cues = [Cue(id=1, start=0, end=1000, ja="（ため息）")]
    res = translator.translate_batch(cues, [])
    assert res == ["(âm thanh)"]


def test_gemini_uncertain_categories():
    """Verifies handling of different uncertain categories in _apply_item_metadata_to_cue."""
    from subtrans.gemini_backend import _apply_item_metadata_to_cue

    categories = ["pronoun", "referent", "wordplay", "archaic", "term", "other"]
    for cat in categories:
        cue = Cue(id=1, start=0, end=1000, ja="...")
        item = {"id": 1, "vi": "Test", "uncertain": cat}
        _apply_item_metadata_to_cue(cue, item)
        assert cue.uncertain is True
        assert cue.uncertain_reason == cat

    # None or empty
    cue_none = Cue(id=2, start=0, end=1000, ja="...")
    _apply_item_metadata_to_cue(cue_none, {"id": 2, "vi": "Test", "uncertain": "none"})
    assert cue_none.uncertain is False
    assert cue_none.uncertain_reason == "none"


def test_gemini_thinking_config_mapping():
    """Verifies that thinking config is set to MINIMAL for flash-lite and disabled (budget=0) for other models."""
    cfg_lite = get_thinking_config("gemini-3.5-flash-lite")
    assert cfg_lite is not None
    assert cfg_lite.thinking_level.name == "MINIMAL"

    cfg_flash = get_thinking_config("gemini-3.8-flash")
    assert cfg_flash is not None
    assert cfg_flash.thinking_budget == 0
