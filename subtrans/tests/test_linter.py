# -*- coding: utf-8 -*-
"""
subtrans/tests/test_linter.py
Unit tests for Linter V1 (subtrans/linter.py).
Covers CJK leakage, CPS thresholds & boundaries, Unicode NFC normalization,
banned calques & transliterations, Romaji vocative retention (built-in, Glossary, CastSheet),
zero-overlap gap boundary checks, deterministic auto-fixing, and markdown reports.
"""

import unicodedata
from unittest.mock import MagicMock
import pytest

from subtrans.models import Cue, Glossary, Character
from subtrans.linter import (
    lint_cue,
    lint_subtitles,
    auto_fix_cues,
    format_lint_report,
    RULE_CJK_LEAK,
    RULE_HIGH_CPS,
    RULE_UNICODE_NOT_NFC,
    RULE_ROMAJI_MISSING,
    RULE_BANNED_TERM,
    RULE_EMPTY_TEXT,
    RULE_ZERO_OVERLAP,
    BANNED_TERMS_MAP,
)


def test_lint_empty_or_placeholder_translations():
    """Verifies that empty strings, whitespace, and placeholder strings are flagged as ERROR."""
    placeholders = ["", "   \t\n", "[Chưa dịch]", "(Chưa dịch)", "[Untranslated]", "[DRY-RUN]"]
    for ph in placeholders:
        cue = Cue(id=1, start=0, end=1000, ja="こんにちは", vi=ph)
        violations = lint_cue(cue)
        assert len(violations) == 1
        assert violations[0].rule_id == RULE_EMPTY_TEXT
        assert violations[0].severity == "ERROR"


def test_lint_cjk_leak():
    """Verifies detection of leftover Kanji, Hiragana, or Katakana in Vietnamese output."""
    cue = Cue(id=1, start=0, end=2000, ja="こんにちは", vi="Xin chào こんにちは")
    violations = lint_cue(cue)
    rule_ids = [v.rule_id for v in violations]
    assert RULE_CJK_LEAK in rule_ids
    cjk_v = next(v for v in violations if v.rule_id == RULE_CJK_LEAK)
    assert cjk_v.severity == "ERROR"
    assert "こんにちは" in cjk_v.message


def test_lint_cps_boundary_conditions():
    """Verifies exact boundaries of Characters Per Second calculation."""
    # 0 ms duration -> should not crash with ZeroDivisionError
    cue_zero = Cue(id=1, start=1000, end=1000, ja="...", vi="Một hai ba")
    violations_zero = lint_cue(cue_zero)
    # Zero duration returns CPS 0.0 in calculate_cps, thus no crash

    # Safe cap: 20 chars in 1000ms = 20.0 CPS -> PASS (no violation)
    cue_safe = Cue(id=2, start=0, end=1000, ja="...", vi="Một hai ba bốn năm s")  # 20 chars
    v_safe = lint_cue(cue_safe)
    assert not any(v.rule_id == RULE_HIGH_CPS for v in v_safe)

    # Warning: 201 chars in 10,000ms = 20.1 CPS (> 20.0, <= 25.0) -> WARNING
    cue_warn = Cue(id=3, start=0, end=10000, ja="...", vi="a" * 201)
    v_warn = lint_cue(cue_warn)
    cps_warn = [v for v in v_warn if v.rule_id == RULE_HIGH_CPS]
    assert len(cps_warn) == 1
    assert cps_warn[0].severity == "WARNING"

    # Hard cap boundary: 250 chars in 10,000ms = 25.0 CPS (<= 25.0) -> WARNING
    cue_hard_boundary = Cue(id=4, start=0, end=10000, ja="...", vi="a" * 250)
    v_hard_b = lint_cue(cue_hard_boundary)
    cps_hb = [v for v in v_hard_b if v.rule_id == RULE_HIGH_CPS]
    assert len(cps_hb) == 1
    assert cps_hb[0].severity == "WARNING"

    # Hard cap exceeded: 251 chars in 10,000ms = 25.1 CPS (> 25.0) -> ERROR
    cue_hard_exceed = Cue(id=5, start=0, end=10000, ja="...", vi="a" * 251)
    v_hard_e = lint_cue(cue_hard_exceed)
    cps_he = [v for v in v_hard_e if v.rule_id == RULE_HIGH_CPS]
    assert len(cps_he) == 1
    assert cps_he[0].severity == "ERROR"


def test_lint_unicode_nfc():
    """Verifies detection of non-NFC normalized strings (e.g., NFD decomposed accents)."""
    nfd_text = unicodedata.normalize("NFD", "Tiếng Việt chuẩn xác")
    cue = Cue(id=1, start=0, end=2000, ja="...", vi=nfd_text)
    violations = lint_cue(cue)
    assert any(v.rule_id == RULE_UNICODE_NOT_NFC for v in violations)

    # NFC should pass
    nfc_text = unicodedata.normalize("NFC", nfd_text)
    cue.vi = nfc_text
    violations_pass = lint_cue(cue)
    assert not any(v.rule_id == RULE_UNICODE_NOT_NFC for v in violations_pass)


def test_lint_banned_terms_case_insensitive_and_all_terms():
    """Verifies that all terms in BANNED_TERMS_MAP are caught case-insensitively."""
    for banned, repl in BANNED_TERMS_MAP.items():
        # Lowercase test
        cue_lower = Cue(id=1, start=0, end=2000, ja="...", vi=f"Đây là {banned.lower()} bị cấm.")
        v_lower = lint_cue(cue_lower)
        assert any(v.rule_id == RULE_BANNED_TERM for v in v_lower)

        # Uppercase test
        cue_upper = Cue(id=2, start=0, end=2000, ja="...", vi=f"ĐÂY LÀ {banned.upper()} BỊ CẤM.")
        v_upper = lint_cue(cue_upper)
        assert any(v.rule_id == RULE_BANNED_TERM for v in v_upper)


def test_lint_romaji_vocatives():
    """Verifies that proper names called in Japanese must be retained as Romaji."""
    # Built-in vocatives: フラン -> Fran
    cue_bad = Cue(id=1, start=0, end=2000, ja="フラン、行くぞ！", vi="Em, đi thôi!")
    violations_bad = lint_cue(cue_bad)
    assert any(v.rule_id == RULE_ROMAJI_MISSING for v in violations_bad)

    cue_good = Cue(id=2, start=0, end=2000, ja="フラン、行くぞ！", vi="Fran, đi thôi!")
    violations_good = lint_cue(cue_good)
    assert not any(v.rule_id == RULE_ROMAJI_MISSING for v in violations_good)


def test_lint_romaji_vocatives_with_glossary_and_cast():
    """Verifies that character names defined in Glossary or CastSheet are enforced in vocatives."""
    # 1. Custom character in Glossary
    glossary = Glossary(
        characters=[
            Character(name_ja="ステファン", name_vi="Stephane", role="Knight"),
        ]
    )

    cue_gloss_bad = Cue(id=1, start=0, end=2000, ja="ステファン、待ってくれ！", vi="Cậu ơi, đợi đã!")
    v_gbad = lint_cue(cue_gloss_bad, glossary=glossary)
    assert any(v.rule_id == RULE_ROMAJI_MISSING and "Stephane" in v.message for v in v_gbad)

    cue_gloss_good = Cue(id=2, start=0, end=2000, ja="ステファン、待ってくれ！", vi="Stephane, đợi đã!")
    v_ggood = lint_cue(cue_gloss_good, glossary=glossary)
    assert not any(v.rule_id == RULE_ROMAJI_MISSING for v in v_ggood)

    # 2. Custom character in CastSheet
    mock_cast = MagicMock()
    mock_char = MagicMock()
    mock_char.name_ja = "フェル"
    mock_char.name_vi = "Fel"
    mock_cast.characters = [mock_char]

    cue_cast_bad = Cue(id=3, start=0, end=2000, ja="フェル、ご飯だよ", vi="Này, cơm đây rồi")
    v_cbad = lint_cue(cue_cast_bad, cast_sheet=mock_cast)
    assert any(v.rule_id == RULE_ROMAJI_MISSING and "Fel" in v.message for v in v_cbad)

    cue_cast_good = Cue(id=4, start=0, end=2000, ja="フェル、ご飯だよ", vi="Fel, cơm đây rồi")
    v_cgood = lint_cue(cue_cast_good, cast_sheet=mock_cast)
    assert not any(v.rule_id == RULE_ROMAJI_MISSING for v in v_cgood)


def test_lint_zero_overlap_exact_boundaries():
    """Verifies inter-cue gap boundary checks: exact 40ms pass, 39ms fail, overlap fail."""
    # Gap = 40ms -> PASS
    cues_pass = [
        Cue(id=1, start=1000, end=1960, ja="A", vi="Câu một"),
        Cue(id=2, start=2000, end=3000, ja="B", vi="Câu hai"),
    ]
    v_pass = lint_subtitles(cues_pass, min_gap_ms=40)
    assert not any(v.rule_id == RULE_ZERO_OVERLAP for v in v_pass)

    # Gap = 39ms -> FAIL
    cues_fail_gap = [
        Cue(id=1, start=1000, end=1961, ja="A", vi="Câu một"),
        Cue(id=2, start=2000, end=3000, ja="B", vi="Câu hai"),
    ]
    v_fail_gap = lint_subtitles(cues_fail_gap, min_gap_ms=40)
    assert any(v.rule_id == RULE_ZERO_OVERLAP for v in v_fail_gap)

    # Overlap (end > start) -> FAIL
    cues_overlap = [
        Cue(id=1, start=1000, end=2050, ja="A", vi="Câu một"),
        Cue(id=2, start=2000, end=3000, ja="B", vi="Câu hai"),
    ]
    v_overlap = lint_subtitles(cues_overlap, min_gap_ms=40)
    assert any(v.rule_id == RULE_ZERO_OVERLAP for v in v_overlap)


def test_auto_fix_cues_extended_edge_cases():
    """Verifies duration stretching for high CPS, edge cases, and non-modifiable cues."""
    nfd_text = unicodedata.normalize("NFD", "Một câu phụ đề có dấu tiếng Việt…")
    cues = [
        # Cue 1: Normal with large gap ahead (starts 0, ends 500, next starts 3000)
        # 33 chars in 500ms = 66 CPS (> 20.0) -> Should be stretched
        Cue(id=1, start=0, end=500, ja="...", vi=nfd_text),
        # Cue 2: Tight gap ahead (starts 3000, ends 3960, next starts 4000)
        # Gap is only 40ms, exactly min_gap_ms -> Cannot stretch into gap
        Cue(id=2, start=3000, end=3960, ja="...", vi="Câu này cũng hơi dài đấy nhé bạn"),
        # Cue 3: Last cue in the sequence (starts 4000, ends 4500, no cue following)
        # 32 chars in 500ms = 64 CPS -> Should freely stretch to needed duration
        Cue(id=3, start=4000, end=4500, ja="...", vi="Đây là câu kết thúc của tập phim"),
    ]

    fixed, fixes_count = auto_fix_cues(cues, min_gap_ms=40, max_safe_cps=20.0)
    assert fixes_count > 0

    # Cue 1 stretched into gap
    assert fixed[0].end > 500
    assert fixed[0].end <= fixed[1].start - 40
    assert unicodedata.is_normalized("NFC", fixed[0].vi)
    assert "..." in fixed[0].vi

    # Cue 2 could not stretch because available_end == cue.end
    assert fixed[1].end == 3960

    # Cue 3 stretched freely at the end of the subtitle stream
    assert fixed[2].end > 4500
    # Clean len is 32 chars -> at 20 CPS, needs ~1600ms -> end >= 4000 + 1600 = 5600
    assert fixed[2].end >= 5600


def test_format_lint_report_empty_and_mixed():
    """Verifies formatting of markdown report under clean and violated conditions."""
    # 1. Clean report
    clean_report = format_lint_report([])
    assert "Tất cả kiểm tra chất lượng đều ĐẠT" in clean_report
    assert "PASS - 0 violations" in clean_report

    # 2. Mixed report with ERROR and WARNING
    cues = [
        Cue(id=1, start=0, end=2000, ja="こんにちは", vi="Xin chào こんにちは"),  # CJK leak (ERROR)
        Cue(id=2, start=0, end=1000, ja="...", vi="Một câu hơi nhanh này!"),  # High CPS (WARNING)
    ]
    violations = lint_subtitles(cues)
    report = format_lint_report(violations)
    assert "Báo Cáo Kiểm Định Phụ Đề (Linter V1)" in report
    assert "🔴 ERROR" in report
    assert "🟡 WARNING" in report
    assert "LINT_CJK_LEAK" in report
    assert "LINT_HIGH_CPS" in report
