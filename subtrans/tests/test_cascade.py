# -*- coding: utf-8 -*-
"""
subtrans/tests/test_cascade.py
Unit tests for Smart Cloud Cascade (subtrans/cascade.py).
"""

from unittest.mock import MagicMock
import pytest
from subtrans.models import Cue
from subtrans.gemini_backend import GeminiTranslator
from subtrans.cascade import (
    score_cue_difficulty,
    find_cues_needing_repair,
    run_cascade_repair_pass,
)


def test_score_cue_difficulty():
    # Simple safe cue
    cue_safe = Cue(id=1, start=0, end=2000, ja="こんにちは", vi="Xin chào")
    score_safe = score_cue_difficulty(cue_safe)
    assert score_safe < 3.0

    # Difficult cue: uncertain + archaic Japanese + en_conflict
    cue_hard = Cue(
        id=2,
        start=0,
        end=2000,
        ja="我が名はアリス、汝に命ず",
        vi="Ta là Alice, lệnh cho ngươi",
        uncertain=True,
        uncertain_reason="pronoun",
        en_conflict=True,
    )
    score_hard = score_cue_difficulty(cue_hard)
    assert score_hard >= 7.0


def test_find_cues_needing_repair():
    cues = [
        Cue(id=1, start=0, end=2000, ja="おはよう", vi="Chào buổi sáng"),
        # Cue 2 has Linter violation: missing Romaji name Fran
        Cue(id=2, start=2000, end=4000, ja="フラン、行くぞ！", vi="Em, đi thôi!"),
    ]
    candidates = find_cues_needing_repair(cues)
    assert len(candidates) == 1
    assert candidates[0][0].id == 2


def test_run_cascade_repair_pass():
    cues = [
        Cue(id=1, start=0, end=2000, ja="おはよう", vi="Chào buổi sáng"),
        Cue(id=2, start=2000, end=4000, ja="フラン、行くぞ！", vi="Em, đi thôi!", uncertain=True),
    ]

    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()

    mock_response = MagicMock()
    mock_response.text = '[{"id": 2, "vi": "Fran, đi thôi!", "fix_rationale": "Preserved Romaji name Fran"}]'
    mock_translator.client.models.generate_content.return_value = mock_response

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator)
    assert stats["repaired_count"] == 1
    assert cues[1].vi == "Fran, đi thôi!"
    assert cues[1].uncertain is False
    assert "gemini-3.8-flash" in cues[1].model_used
