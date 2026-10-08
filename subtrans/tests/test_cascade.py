# -*- coding: utf-8 -*-
"""
subtrans/tests/test_cascade.py
Comprehensive unit tests for Smart Cloud Cascade (subtrans/cascade.py).
"""

from unittest.mock import MagicMock
import pytest
from subtrans.models import Cue, Glossary
from subtrans.linter import LintViolation
from subtrans.gemini_backend import GeminiTranslator
from subtrans.cascade import (
    score_cue_difficulty,
    find_cues_needing_repair,
    run_cascade_repair_pass,
)


def test_score_cue_difficulty_baseline():
    # Simple safe cue -> low difficulty
    cue_safe = Cue(id=1, start=0, end=2000, ja="こんにちは", vi="Xin chào")
    score_safe = score_cue_difficulty(cue_safe)
    assert score_safe < 3.0


def test_score_cue_difficulty_components():
    # 1. Uncertainty flag (+4.0) and reason (+2.0)
    c1 = Cue(id=1, start=0, end=2000, ja="A", vi="A", uncertain=True, uncertain_reason="pronoun")
    assert score_cue_difficulty(c1) == 6.0

    # 2. English conflict (+3.0)
    c2 = Cue(id=2, start=0, end=2000, ja="A", vi="A", en_conflict=True)
    assert score_cue_difficulty(c2) == 3.0

    # 3. Archaic Japanese words (+2.0)
    c3 = Cue(id=3, start=0, end=2000, ja="妾の願いを聞き届けよ", vi="A")
    assert score_cue_difficulty(c3) == 2.0

    # 4. Long dialogue (+1.5, length > 35)
    c4 = Cue(id=4, start=0, end=2000, ja="これは非常に長くて複雑なアニメのセリフであり、文構造が重層的であるのだよ", vi="A")
    assert score_cue_difficulty(c4) == 1.5

    # 5. Combined factors
    violations = [
        LintViolation(rule_id="LINT_ROMAJI_MISSING", severity="ERROR", cue_id=5, message="Missing Fran"),
        LintViolation(rule_id="LINT_HIGH_CPS", severity="WARNING", cue_id=5, message="High CPS"),
    ]
    c5 = Cue(id=5, start=0, end=2000, ja="貴様、許さん！", vi="Mày, tao không tha!", uncertain=True, en_conflict=True)
    score5 = score_cue_difficulty(c5, violations=violations)
    # uncertain(4) + en_conflict(3) + archaic(2) + error(4) + warning(1.5) = 14.5
    assert score5 >= 14.0


def test_find_cues_needing_repair_filtering():
    cues = [
        # Cue 1: Clean, no issues
        Cue(id=1, start=0, end=2000, ja="おはよう", vi="Chào buổi sáng"),
        # Cue 2: Linter error (omitted vocative name)
        Cue(id=2, start=2000, end=4000, ja="フラン、行くぞ！", vi="Em, đi thôi!"),
        # Cue 3: Uncertain flag
        Cue(id=3, start=4000, end=6000, ja="何だこれ？", vi="Cái gì đây?", uncertain=True, uncertain_reason="term"),
        # Cue 4: Dropped sound cue -> should be skipped
        Cue(id=4, start=6000, end=7000, ja="（ため息）", vi="", is_dropped_sound=True),
        # Cue 5: Untranslated cue -> should be skipped (cannot repair non-existent draft)
        Cue(id=5, start=7000, end=8000, ja="...", vi="[Chưa dịch]", backend_used="untranslated"),
    ]
    candidates = find_cues_needing_repair(cues, difficulty_threshold=4.0)
    candidate_ids = [c[0].id for c in candidates]
    assert candidate_ids == [2, 3]


def test_find_cues_needing_repair_custom_threshold():
    cues = [
        # Cue with score 3.0 (en_conflict)
        Cue(id=1, start=0, end=2000, ja="Hello", vi="Xin chào", en_conflict=True),
    ]
    # Default threshold 4.0 -> not selected
    assert len(find_cues_needing_repair(cues, difficulty_threshold=4.0)) == 0
    # Lower threshold 2.5 -> selected
    assert len(find_cues_needing_repair(cues, difficulty_threshold=2.5)) == 1


def test_run_cascade_repair_pass_all_clean():
    # If all cues are clean, no API calls should be made
    cues = [
        Cue(id=1, start=0, end=2000, ja="おはよう", vi="Chào buổi sáng"),
        Cue(id=2, start=2000, end=4000, ja="こんにちは", vi="Xin chào"),
    ]
    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator)
    assert stats["repaired_count"] == 0
    assert stats["repair_ratio_pct"] == 0.0
    mock_translator.client.models.generate_content.assert_not_called()


def test_run_cascade_repair_pass_dry_run():
    cues = [
        Cue(id=1, start=0, end=2000, ja="フラン、待って", vi="Em, đợi đã", uncertain=True),
    ]
    dry_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash", dry_run=True)
    stats = run_cascade_repair_pass(cues, repair_translator=dry_translator)
    assert stats["repaired_count"] == 1
    assert "dry-run repair" in cues[0].model_used
    assert cues[0].uncertain is False


def test_run_cascade_repair_pass_api_success():
    cues = [
        Cue(id=1, start=0, end=2000, ja="おはよう", vi="Chào buổi sáng"),
        Cue(id=2, start=2000, end=4000, ja="フラン、行くぞ！", vi="Em, đi thôi!", uncertain=True),
    ]

    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()

    mock_response = MagicMock()
    mock_response.text = '[{"id": 2, "vi": "Fran, đi thôi!", "fix_rationale": "Preserved Romaji vocative name Fran"}]'
    mock_translator.client.models.generate_content.return_value = mock_response

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator)
    assert stats["repaired_count"] == 1
    assert cues[1].vi == "Fran, đi thôi!"
    assert cues[1].uncertain is False
    assert cues[1].uncertain_reason == "repaired"
    assert "gemini-3.8-flash" in cues[1].model_used


def test_run_cascade_repair_pass_api_failure_fallback():
    # If the repair call fails with an exception, keep original draft lines without crashing
    cues = [
        Cue(id=1, start=0, end=2000, ja="フラン、行くぞ！", vi="Em, đi thôi!", uncertain=True),
    ]

    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()
    mock_translator.client.models.generate_content.side_effect = RuntimeError("Network timeout")

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator)
    assert stats["repaired_count"] == 0
    # Original draft line is preserved
    assert cues[0].vi == "Em, đi thôi!"


def test_run_cascade_repair_pass_with_glossary():
    from subtrans.models import Glossary, GlossaryTerm
    cues = [
        Cue(id=1, start=0, end=2000, ja="魔剣を使う", vi="Dùng kiếm ma", uncertain=True),
    ]
    glossary = Glossary(
        terms=[GlossaryTerm(ja="魔剣", en="demon sword", vi="ma kiếm")]
    )

    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()

    captured_prompt = {}

    def mock_generate(model, contents, config):
        captured_prompt["contents"] = contents
        res = MagicMock()
        res.text = '[{"id": 1, "vi": "Dùng ma kiếm"}]'
        return res

    mock_translator.client.models.generate_content.side_effect = mock_generate

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator, glossary=glossary)
    assert stats["repaired_count"] == 1
    assert "glossary" in captured_prompt["contents"]
    assert "ma kiếm" in captured_prompt["contents"]


def test_run_cascade_repair_pass_partial_id_response():
    # If LLM only returns repair for cue 1 but omits cue 2, cue 2 preserves draft
    cues = [
        Cue(id=1, start=0, end=2000, ja="フラン", vi="Em", uncertain=True),
        Cue(id=2, start=2000, end=4000, ja="ウルシ", vi="Nó", uncertain=True),
    ]

    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()
    mock_response = MagicMock()
    # Only cue 1 returned
    mock_response.text = '[{"id": 1, "vi": "Fran"}]'
    mock_translator.client.models.generate_content.return_value = mock_response

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator)
    assert stats["repaired_count"] == 1
    assert cues[0].vi == "Fran"
    assert cues[0].uncertain is False
    assert cues[1].vi == "Nó"  # Preserved draft
    assert cues[1].uncertain is True

