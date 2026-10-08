# -*- coding: utf-8 -*-
"""
subtrans/tests/test_eval_runner.py
Unit tests for Evaluation Harness (subtrans/eval/runner.py & subtrans/eval/metrics.py).
Ensures data integrity of benchmark suites (regression_suite.yaml, pronoun_suite.yaml)
and validates execution of the evaluation report runner.
"""

from pathlib import Path
import pytest
import yaml

from subtrans.eval.runner import load_yaml, run_evaluation
from subtrans.eval.metrics import (
    check_pronoun_match,
    compute_harness_summary,
    calculate_cps,
    detect_cjk_leakage,
    check_romaji_retention,
    check_zero_overlap_violation,
)


def test_eval_load_yaml():
    """Verifies YAML loading of evaluation suites and missing file handling."""
    reg = load_yaml("regression_suite.yaml")
    assert "cases" in reg
    assert len(reg["cases"]) == 6

    pronouns = load_yaml("pronoun_suite.yaml")
    assert "cues" in pronouns
    assert len(pronouns["cues"]) >= 13

    # Non-existent file returns empty dict
    missing = load_yaml("non_existent_suite.yaml")
    assert missing == {}


def test_regression_suite_integrity():
    """Verifies that all 6 regression cases contain required fields."""
    reg = load_yaml("regression_suite.yaml")
    cases = reg.get("cases", [])
    assert len(cases) == 6

    required_keys = {"id", "category", "name", "description"}
    for c in cases:
        for k in required_keys:
            assert k in c, f"Case {c.get('id')} missing key '{k}'"


def test_pronoun_suite_integrity():
    """Verifies that all pronoun benchmark cues contain proper annotations."""
    pronouns = load_yaml("pronoun_suite.yaml")
    cues = pronouns.get("cues", [])
    assert len(cues) >= 13

    required_keys = {"id", "series", "speaker", "ja", "allowed_pronouns"}
    for cue in cues:
        for k in required_keys:
            assert k in cue, f"Pronoun cue {cue.get('id')} missing key '{k}'"
        assert isinstance(cue["allowed_pronouns"], list)
        assert len(cue["allowed_pronouns"]) > 0


def test_eval_metrics_pronoun_and_romaji_match():
    """Verifies pronoun matching and Romaji retention logic."""
    # Pronoun match
    assert check_pronoun_match("Sư phụ, em đói rồi", ["em", "con"]) is True
    assert check_pronoun_match("Tôi không biết", ["em", "con"]) is False
    assert check_pronoun_match("", ["em"]) is False

    # Romaji retention
    assert check_romaji_retention("Chào Fran nhé", ["Fran"]) is True
    assert check_romaji_retention("Chào em nhé", ["Fran"]) is False
    assert check_romaji_retention("", ["Fran"]) is False


def test_eval_metrics_summary_empty():
    """Verifies summary calculation with empty results."""
    summary = compute_harness_summary([])
    assert summary["total"] == 0
    assert summary["pass_rate_pct"] == 100.0


def test_eval_runner_execution():
    """Verifies execution of the evaluation runner and output return structure."""
    result = run_evaluation()
    assert result["status"] == "ready"
    assert result["regression_count"] == 6
    assert result["pronoun_benchmark_count"] >= 13
