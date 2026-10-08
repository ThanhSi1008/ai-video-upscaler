# -*- coding: utf-8 -*-
"""
subtrans/tests/test_gemini.py
Unit tests for Gemini translation backend with mocking.
"""

from unittest.mock import MagicMock, patch
import pytest
from subtrans.models import Cue, Glossary, GlossaryTerm
from subtrans.gemini_backend import GeminiTranslator, filter_relevant_glossary


def test_filter_relevant_glossary():
    cues = [
        Cue(id=1, start=0, end=1000, ja="魔法少女になりたい", en="I want to become a magical girl")
    ]
    glossary = Glossary(
        terms=[
            GlossaryTerm(ja="魔法少女", en="magical girl", vi="ma pháp thiếu nữ"),
            GlossaryTerm(ja="冒険者", en="adventurer", vi="mạo hiểm giả")  # Not present
        ]
    )
    filtered = filter_relevant_glossary(cues, glossary)
    assert "terms" in filtered
    assert len(filtered["terms"]) == 1
    assert filtered["terms"][0]["ja"] == "魔法少女"


def test_gemini_translator_success():
    translator = GeminiTranslator(api_key="mock_key")
    mock_response = MagicMock()
    mock_response.text = '[{"id": 1, "vi": "Chào bạn"}, {"id": 2, "vi": "Hẹn gặp lại"}]'

    translator.client = MagicMock()
    translator.client.models.generate_content.return_value = mock_response

    cues = [
        Cue(id=1, start=0, end=1000, ja="こんにちは"),
        Cue(id=2, start=1000, end=2000, ja="さようなら")
    ]
    res = translator.translate_batch(cues, [])
    assert res == ["Chào bạn", "Hẹn gặp lại"]


def test_gemini_translator_retry_and_split():
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
        good_half2
    ]

    cues = [
        Cue(id=1, start=0, end=1000, ja="こんにちは"),
        Cue(id=2, start=1000, end=2000, ja="さようなら")
    ]
    res = translator.translate_batch(cues, [])
    assert res == ["Chào bạn", "Tạm biệt"]


def test_vocative_proper_name_prompt_instruction():
    from subtrans.gemini_backend import SYSTEM_INSTRUCTION
    assert "Addressing by proper name (Vocative)" in SYSTEM_INSTRUCTION
    assert "ALWAYS KEEP the Romaji proper name directly in the Vietnamese dialogue line" in SYSTEM_INSTRUCTION
    assert "Context-First Reasoning (Chain-of-Thought)" in SYSTEM_INSTRUCTION


def test_gemini_schema_reasoning_metadata():
    translator = GeminiTranslator(api_key="mock_key")
    mock_response = MagicMock()
    mock_response.text = '''[
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
    ]'''

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

