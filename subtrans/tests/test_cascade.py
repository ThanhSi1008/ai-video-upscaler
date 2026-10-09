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

    # 2. Difficulty marker from series config (+2.0)
    c2 = Cue(id=2, start=0, end=2000, ja="妾の願いを聞き届けよ", vi="A")
    assert score_cue_difficulty(c2, difficulty_markers=["妾"]) == 2.0
    assert c2.difficulty_features["marker_matched"] is True

    # 3. Linter error (+4.0)
    violations = [
        LintViolation(rule_id="LINT_ROMAJI_MISSING", severity="ERROR", cue_id=3, message="Missing Fran"),
    ]
    c3 = Cue(id=3, start=0, end=2000, ja="フラン、行くぞ", vi="Em, đi thôi")
    assert score_cue_difficulty(c3, violations=violations) == 4.0

    # 4. Combined factors
    c4 = Cue(id=4, start=0, end=2000, ja="妾、許さん！", vi="Mày, tao không tha!", uncertain=True, uncertain_reason="archaic")
    score4 = score_cue_difficulty(c4, violations=violations, difficulty_markers=["妾"])
    # uncertain(4) + reason(2) + marker(2) + error(4) = 12.0
    assert score4 == 12.0


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
        # Cue with marker score 2.0
        Cue(id=1, start=0, end=2000, ja="妾はここにいる", vi="Tôi ở đây"),
    ]
    # Default threshold 4.0 -> not selected
    assert len(find_cues_needing_repair(cues, difficulty_threshold=4.0, difficulty_markers=["妾"])) == 0
    # Lower threshold 1.5 -> selected
    assert len(find_cues_needing_repair(cues, difficulty_threshold=1.5, difficulty_markers=["妾"])) == 1


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


def test_cascade_fail_open_on_429_quota():
    """Verifies that 429 Resource Exhausted fails open to Lite draft and sets unrepaired_quota_exceeded."""
    cues = [
        Cue(id=1, start=0, end=2000, ja="フラン", vi="Bản nháp Lite", uncertain=True),
    ]
    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()
    mock_translator.client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator)
    assert stats["repaired_count"] == 0
    assert stats["fail_open_code"] == "unrepaired_quota_exceeded"
    assert cues[0].vi == "Bản nháp Lite"
    assert cues[0].repair_reason_code == "unrepaired_quota_exceeded"


def test_cascade_fail_open_on_timeout():
    """Verifies that network timeout fails open to Lite draft and sets unrepaired_timeout."""
    cues = [
        Cue(id=1, start=0, end=2000, ja="フラン", vi="Bản nháp Lite", uncertain=True),
    ]
    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()
    mock_translator.client.models.generate_content.side_effect = TimeoutError("Request timed out after 60s")

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator)
    assert stats["repaired_count"] == 0
    assert stats["fail_open_code"] == "unrepaired_timeout"
    assert cues[0].vi == "Bản nháp Lite"
    assert cues[0].repair_reason_code == "unrepaired_timeout"


def test_cascade_fail_open_on_invalid_json_schema():
    """Verifies that malformed JSON response fails open to Lite draft and sets unrepaired_bad_schema."""
    cues = [
        Cue(id=1, start=0, end=2000, ja="フラン", vi="Bản nháp Lite", uncertain=True),
    ]
    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()
    mock_res = MagicMock()
    mock_res.text = "This is not valid JSON content at all"
    mock_translator.client.models.generate_content.return_value = mock_res

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator)
    assert stats["repaired_count"] == 0
    assert stats["fail_open_code"] == "unrepaired_bad_schema"
    assert cues[0].vi == "Bản nháp Lite"
    assert cues[0].repair_reason_code == "unrepaired_bad_schema"


def test_cascade_relint_gate_rejects_new_critical_violation():
    """Verifies that Re-lint gate rejects repair introducing critical violations (e.g. banned calque 'Pháp Lạp')."""
    cues = [
        Cue(id=1, start=0, end=2000, ja="フラン", vi="Cô bé", uncertain=True),
    ]
    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()
    mock_res = MagicMock()
    # Flash returns a banned calque
    mock_res.text = '[{"id": 1, "vi": "Pháp Lạp"}]'
    mock_translator.client.models.generate_content.return_value = mock_res

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator)
    assert stats["repaired_count"] == 0
    assert stats["rejected_count"] == 1
    assert cues[0].vi == "Cô bé"  # Preserved original draft
    assert cues[0].repair_reason_code == "rejected_critical_violation"


def test_relint_gate_identity_comparison():
    """Verifies that gate uses _ident and catches a newly introduced error even if error count is equal."""
    from subtrans.cascade import accept_or_reject_repaired_cue
    # Base has CJK leakage
    cue = Cue(id=1, start=0, end=2000, ja="フラン", draft_vi="Fran 日本語", vi="Fran 日本語")
    # Proposed fix removes Japanese but introduces banned Sino-Vietnamese calque "Pháp Lạp"
    # Both base and fix have exactly 1 error, but identity is different!
    is_accepted, reason_code, _ = accept_or_reject_repaired_cue(cue, fixed_vi="Pháp Lạp")
    assert not is_accepted
    assert reason_code in ("rejected_critical_violation", "rejected_new_error")


def test_relint_gate_semantic_drift():
    """Verifies that large edit distance (>0.5) and digit mismatches are flagged."""
    from subtrans.cascade import accept_or_reject_repaired_cue
    cue1 = Cue(id=1, start=0, end=3000, ja="待って", draft_vi="Đợi một chút nhé", vi="Đợi một chút nhé")
    # Substantially rewritten text with edit distance > 0.5
    is_accepted, reason_code, _ = accept_or_reject_repaired_cue(cue1, fixed_vi="Hãy dừng lại ngay lập tức tại đây hỡi người bạn")
    assert is_accepted
    assert reason_code == "accepted_large_edit"
    assert cue1.is_large_edit is True

    # Digit mismatch
    cue2 = Cue(id=2, start=0, end=3000, ja="5階へ行く", draft_vi="Lên tầng 5 thôi", vi="Lên tầng 5 thôi")
    is_accepted2, reason_code2, _ = accept_or_reject_repaired_cue(cue2, fixed_vi="Lên tầng 6 thôi")
    assert is_accepted2
    assert reason_code2 == "accepted_digit_mismatch"
    assert cue2.is_digit_mismatch is True


def test_relint_gate_cps_hard_ceiling():
    """Verifies that extreme CPS overflow (>25 CPS) is rejected via is_hard_ceiling."""
    from subtrans.cascade import accept_or_reject_repaired_cue
    # Short 1.0 second duration
    cue = Cue(id=1, start=0, end=1000, ja="短い", draft_vi="Ngắn thôi", vi="Ngắn thôi")
    # Long text (35 chars in 1s = 35 CPS > 25 ceiling)
    fixed_long = "Đây là một câu rất dài nhằm mục đích vượt trần cứng"
    is_accepted, reason_code, _ = accept_or_reject_repaired_cue(cue, fixed_vi=fixed_long)
    assert not is_accepted
    assert reason_code == "rejected_cps_overflow"


def test_cascade_metadata_disagreement_telemetry():
    """Verifies that Flash disagreeing with Lite on speaker/addressee flags metadata_disagreement."""
    cues = [
        Cue(id=1, start=0, end=2000, ja="フラン", vi="Fran", speaker="Jean", addressee="Fran", uncertain=True),
    ]
    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash")
    mock_translator.client = MagicMock()
    mock_res = MagicMock()
    # Flash outputs a different speaker ("Urushi" instead of "Jean")
    mock_res.text = '[{"id": 1, "vi": "Fran ơi", "speaker": "Urushi", "addressee": "Fran"}]'
    mock_translator.client.models.generate_content.return_value = mock_res

    stats = run_cascade_repair_pass(cues, repair_translator=mock_translator)
    assert stats["repaired_count"] == 1
    assert cues[0].metadata_disagreement is True


