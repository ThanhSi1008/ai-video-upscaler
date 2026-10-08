# -*- coding: utf-8 -*-
"""
subtrans/tests/test_harness.py
Automated validation of the 6 canonical regression cases defined in regression_suite.yaml,
plus unit tests for the deterministic metrics.
"""

from pathlib import Path
import yaml
import pytest
import pysubs2

from subtrans.models import Cue
from subtrans.parser import split_multi_speaker_cc_events, is_pure_sound_cue
from subtrans.writer import optimize_cues_for_playback
from subtrans.eval.metrics import (
    calculate_cps,
    detect_cjk_leakage,
    check_romaji_retention,
    check_zero_overlap_violation,
    compute_harness_summary,
)

SUITE_PATH = Path(__file__).resolve().parent.parent / "eval" / "regression_suite.yaml"


@pytest.fixture
def regression_cases():
    assert SUITE_PATH.exists(), f"Suite file not found at {SUITE_PATH}"
    with open(SUITE_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {c["id"]: c for c in data.get("cases", [])}


def test_regression_case_01_multi_speaker_split(regression_cases):
    """Case 1: Tách sự kiện Closed Captions gộp 2 người nói."""
    case = regression_cases["case_01_multi_speaker_cc_split"]
    events = [pysubs2.SSAEvent(start=1000, end=3000, text=case["raw_ja"])]
    split_res = split_multi_speaker_cc_events(events)

    assert len(split_res) == case["expected_cue_count"]
    assert any(case["must_contain_phrase"] in e.text for e in split_res)


def test_regression_case_02_strict_zero_overlap(regression_cases):
    """Case 2: Triệt tiêu 100% chồng lấn thời gian (Strict 40ms Gap)."""
    case = regression_cases["case_02_strict_zero_overlap"]
    cues = [
        Cue(id=1, start=c["start"], end=c["end"], ja="JA", vi=c["vi"])
        for c in case["cues_input"]
    ]
    optimized = optimize_cues_for_playback(cues)

    assert len(optimized) == 2
    assert optimized[0].end <= case["expected_max_end_cue_1"]
    assert not check_zero_overlap_violation(optimized[0].end, optimized[1].start, min_gap_ms=40)


def test_regression_case_03_cache_pollution_guard(regression_cases, tmp_path):
    """Case 3: Chống ô nhiễm SQLite Cache do chuỗi Dry-Run / Untranslated."""
    from subtrans.cache import TranslationCache, UNTRANSLATED_STRINGS
    case = regression_cases["case_03_cache_pollution_guard"]
    test_db = tmp_path / "test_cache.db"
    cache = TranslationCache(test_db)

    for idx, forbidden in enumerate(case["forbidden_strings"]):
        cache_key = f"key_{idx}"
        cache.set_cue(cache_key, "gemini", "gemini-3.5-flash-lite", forbidden)
        # Kiểm tra không được lấy ra từ cache
        assert cache.get_cue(cache_key) is None



def test_regression_case_04_vocative_name_retention(regression_cases):
    """Case 4: Bảo tồn tên riêng Romaji khi đối thoại gọi tên."""
    case = regression_cases["case_04_vocative_name_retention"]
    name = case["target_character_romaji"]

    good_vi = f"{name}, em có sao không?"
    bad_vi = "Cậu có sao không?"

    assert check_romaji_retention(good_vi, [name]) is True
    assert check_romaji_retention(bad_vi, [name]) is False


def test_regression_case_05_anime_laughter_sound_cue(regression_cases):
    """Case 5: Phát hiện tiếng cười anime, chống gán nhầm câu thoại tiếng Anh."""
    case = regression_cases["case_05_anime_laughter_sound_cue"]
    ja_text = case["raw_ja"]

    assert is_pure_sound_cue(ja_text) is True


def test_regression_case_06_anti_duplicate_dual_dialogue(regression_cases):
    """Case 6: Triệt tiêu dị bản trùng lặp trong Dual Dialogue."""
    case = regression_cases["case_06_anti_duplicate_dual_dialogue"]
    cues = [
        Cue(id=c["id"], start=c["start"], end=c["end"], ja="JA", vi=c["vi"])
        for c in case["simultaneous_cues"]
    ]
    optimized = optimize_cues_for_playback(cues)

    assert len(optimized) == case["expected_output_cues_count"]
    assert case["forbidden_in_output"] not in optimized[0].vi


def test_eval_metrics_deterministic():
    """Kiểm tra các hàm tính toán chỉ số định lượng."""
    # CPS
    cps_normal = calculate_cps("Xin chào bạn, hôm nay thế nào?", 2000)
    assert 10.0 <= cps_normal <= 18.0

    cps_insane = calculate_cps("Một câu thoại cực kỳ dài dòng vượt quá giới hạn đọc thông thường của con người!", 1000)
    assert cps_insane > 50.0

    # CJK Leakage
    assert len(detect_cjk_leakage("Dòng này hoàn toàn là tiếng Việt")) == 0
    leaked = detect_cjk_leakage("Đây là câu có chữ 日本語 bị sót lại")
    assert len(leaked) == 3

    # Summary
    results = [
        {"passed": True, "cps": 15.0, "cjk_leakage": False, "romaji_retained": True, "expected_romaji": True},
        {"passed": True, "cps": 16.0, "cjk_leakage": False, "romaji_retained": True, "expected_romaji": True},
        {"passed": False, "cps": 25.0, "cjk_leakage": True, "romaji_retained": False, "expected_romaji": True},
    ]
    summary = compute_harness_summary(results)
    assert summary["total_cases"] == 3
    assert summary["passed_cases"] == 2
    assert summary["pass_rate_pct"] == 66.67
    assert summary["cjk_leak_count"] == 1
    assert summary["high_cps_count"] == 1
