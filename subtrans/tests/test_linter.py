# -*- coding: utf-8 -*-
"""
subtrans/tests/test_linter.py
Unit tests for Linter V1 (subtrans/linter.py).
"""

import unicodedata
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
)


def test_lint_cjk_leak():
    cue = Cue(id=1, start=0, end=2000, ja="こんにちは", vi="Xin chào こんにちは")
    violations = lint_cue(cue)
    rule_ids = [v.rule_id for v in violations]
    assert RULE_CJK_LEAK in rule_ids
    cjk_v = next(v for v in violations if v.rule_id == RULE_CJK_LEAK)
    assert cjk_v.severity == "ERROR"
    assert "こんにちは" in cjk_v.message


def test_lint_cps_warning_and_error():
    # Cue 1: 50 characters in 1500ms = 33.3 CPS (Error > 25)
    long_text = "Đây là một câu rất dài nhằm kiểm tra tốc độ đọc rất cao"
    cue_fast = Cue(id=1, start=0, end=1500, ja="...", vi=long_text)
    violations1 = lint_cue(cue_fast)
    assert any(v.rule_id == RULE_HIGH_CPS and v.severity == "ERROR" for v in violations1)

    # Cue 2: 22 characters in 1000ms = 22 CPS (Warning > 20, <= 25)
    cue_warn = Cue(id=2, start=0, end=1000, ja="...", vi="Một câu hơi nhanh này!")
    violations2 = lint_cue(cue_warn)
    assert any(v.rule_id == RULE_HIGH_CPS and v.severity == "WARNING" for v in violations2)

    # Cue 3: 15 characters in 1500ms = 10 CPS (Safe <= 20)
    cue_safe = Cue(id=3, start=0, end=1500, ja="...", vi="Câu này rất vừa")
    violations3 = lint_cue(cue_safe)
    assert not any(v.rule_id == RULE_HIGH_CPS for v in violations3)


def test_lint_unicode_nfc():
    # Composed vs decomposed: 'ế' as 'e' + combining acute + combining circumflex (NFD)
    nfd_text = unicodedata.normalize("NFD", "Tiếng Việt chuẩn xác")
    cue = Cue(id=1, start=0, end=2000, ja="...", vi=nfd_text)
    violations = lint_cue(cue)
    assert any(v.rule_id == RULE_UNICODE_NOT_NFC for v in violations)

    # NFC should pass
    nfc_text = unicodedata.normalize("NFC", nfd_text)
    cue.vi = nfc_text
    violations_pass = lint_cue(cue)
    assert not any(v.rule_id == RULE_UNICODE_NOT_NFC for v in violations_pass)


def test_lint_banned_terms():
    cue = Cue(id=1, start=0, end=2000, ja="フラン、大丈夫か？", vi="Pháp Lạp, em có sao không?")
    violations = lint_cue(cue)
    assert any(v.rule_id == RULE_BANNED_TERM for v in violations)
    term_v = next(v for v in violations if v.rule_id == RULE_BANNED_TERM)
    assert "Pháp Lạp" in term_v.message

    cue_calque = Cue(id=2, start=0, end=2000, ja="...", vi="Khám phá thế giới quan mới.")
    violations2 = lint_cue(cue_calque)
    assert any(v.rule_id == RULE_BANNED_TERM for v in violations2)


def test_lint_romaji_vocatives():
    # Fran vocative in Japanese: フラン -> expected Fran in VI
    # Bad VI replaces Fran with generic "Em"
    cue_bad = Cue(id=1, start=0, end=2000, ja="フラン、行くぞ！", vi="Em, đi thôi!")
    violations_bad = lint_cue(cue_bad)
    assert any(v.rule_id == RULE_ROMAJI_MISSING for v in violations_bad)

    # Good VI preserves Romaji name "Fran"
    cue_good = Cue(id=2, start=0, end=2000, ja="フラン、行くぞ！", vi="Fran, đi thôi!")
    violations_good = lint_cue(cue_good)
    assert not any(v.rule_id == RULE_ROMAJI_MISSING for v in violations_good)


def test_lint_zero_overlap():
    cues = [
        Cue(id=1, start=1000, end=2000, ja="A", vi="Câu một"),
        Cue(id=2, start=2020, end=3000, ja="B", vi="Câu hai"),  # Gap = 20ms < 40ms -> violation
    ]
    violations = lint_subtitles(cues, min_gap_ms=40)
    assert any(v.rule_id == RULE_ZERO_OVERLAP for v in violations)

    # With proper gap (e.g. 50ms)
    cues[0].end = 1950
    violations_ok = lint_subtitles(cues, min_gap_ms=40)
    assert not any(v.rule_id == RULE_ZERO_OVERLAP for v in violations_ok)


def test_auto_fix_cues():
    nfd_text = unicodedata.normalize("NFD", "Đây là một câu phụ đề tiếng Việt khá dài cần kéo giãn…")
    cues = [
        # Cue 1: NFD text with unicode ellipsis + high CPS (54 chars in 1000ms = 54 CPS)
        Cue(id=1, start=0, end=1000, ja="...", vi=nfd_text),
        # Cue 2: start at 4000ms (gap of 3000ms available)
        Cue(id=2, start=4000, end=5000, ja="...", vi="Tiếp theo"),
    ]
    fixed, count = auto_fix_cues(cues, min_gap_ms=40, max_safe_cps=20.0)
    assert count > 0
    # Text must be NFC
    assert unicodedata.is_normalized("NFC", fixed[0].vi)
    # Ellipsis must be standardized
    assert "..." in fixed[0].vi
    # Duration of cue 1 must be stretched into the gap so CPS decreases
    assert fixed[0].end > 1000
    assert fixed[0].end <= fixed[1].start - 40


def test_format_lint_report():
    cue = Cue(id=1, start=0, end=1000, ja="こんにちは", vi="Xin chào こんにちは")
    violations = lint_cue(cue)
    report = format_lint_report(violations)
    assert "Báo Cáo Kiểm Định Phụ Đề (Linter V1)" in report
    assert "LINT_CJK_LEAK" in report
    assert "ERROR" in report
