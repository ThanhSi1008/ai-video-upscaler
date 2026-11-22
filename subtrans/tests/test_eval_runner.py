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


def test_batch3_dialogue_preservation_and_prompt_dedup():
    """Verify Batch 3 parser sound-label boundaries, HON:sama guards, Dual Dialogue, and glossary dedup."""
    from subtrans.cascade import accept_or_reject_repaired_cue
    from subtrans.cast_sheet import _merge_cast_characters
    from subtrans.gemini_backend import filter_relevant_glossary
    from subtrans.ja_morph_tagger import extract_ja_linguistic_hints
    from subtrans.models import AddressRules, Character, Cue, Glossary, GlossaryTerm
    from subtrans.parser import is_sound_effect_label
    from subtrans.writer import optimize_cues_for_playback

    for name in ("ベルナルド", "琴音", "風太郎", "時雨", "ベル"):
        assert is_sound_effect_label(name) is False
    for sfx in ("筆記音", "ため息", "扉が開く音", "足音"):
        assert is_sound_effect_label(sfx) is True

    for non_hon in ("さまざまな様子だ", "同様の様式", "お疲れさまでした"):
        assert "HON:sama" not in (extract_ja_linguistic_hints(non_hon) or "")
    assert "HON:sama" in (extract_ja_linguistic_hints("慧月様、フリーレンさま") or "")

    recips = [
        Cue(id=1, start=1000, end=3000, ja="A", vi="Anh nhớ em nhiều lắm!"),
        Cue(id=2, start=1050, end=3000, ja="B", vi="Em nhớ anh nhiều lắm!"),
    ]
    merged_dual = optimize_cues_for_playback(recips)
    assert len(merged_dual) == 1 and "\n" in (merged_dual[0].vi or "")

    seq_cues = [
        Cue(id=1, start=1000, end=1400, ja="A", vi="Câu ngắn một."),
        Cue(id=2, start=1600, end=3000, ja="B", vi="Câu kế tiếp riêng biệt."),
    ]
    opt_seq = optimize_cues_for_playback(seq_cues)
    assert len(opt_seq) == 2 and "\n" not in (opt_seq[0].vi or "")

    g = Glossary(
        characters=[Character(name_ja="フラン", name_vi="Fran")],
        terms=[GlossaryTerm(ja="黒猫族", vi="Hắc Miêu Tộc")],
        address_rules=AddressRules(principle="Keep hierarchy"),
    )
    c_list = [Cue(id=1, start=0, end=1000, ja="黒猫族の誇り")]
    assert "characters" in filter_relevant_glossary(c_list, g)
    deduped_g = filter_relevant_glossary(c_list, g, include_static_cast=False)
    assert "characters" not in deduped_g and "terms" in deduped_g

    cue_num = Cue(id=1, start=0, end=5000, ja="3日後に会おう", vi="3 ngày nữa gặp", draft_vi="3 ngày nữa gặp")
    ok, code, _ = accept_or_reject_repaired_cue(cue_num, "Hoàn toàn viết lại thành 5 ngày sau chúng ta hội ngộ nhé!")
    assert ok is True and code == "accepted_digit_mismatch" and cue_num.is_digit_mismatch is True

    merged_cast = _merge_cast_characters(
        [Character(name_ja="フラン", name_vi="Fran")],
        [Character(name_ja="ウルシ", name_vi="Urushi")],
    )
    assert [c.name_vi for c in merged_cast] == ["Fran", "Urushi"]


def test_verify_pronoun_cue_passes_and_rejects_mismatches():
    """Verify verify_pronoun_cue validates valid cues, linguistic exceptions, and rejects mismatches."""
    from subtrans.eval.runner import verify_pronoun_cue

    cues_by_id = {c["id"]: c for c in load_yaml("pronoun_suite.yaml").get("cues", [])}
    for cid in (
        "tenken_01",
        "tenken_03",
        "tenken_05",
        "tenken_07",
        "dmmtv_cc_01",
        "futsutsuka_01",
        "mushoku_01",
        "katakana_loanword_01",
    ):
        ok, msg = verify_pronoun_cue(cues_by_id[cid])
        assert ok is True, f"{cid} unexpectedly failed: {msg}"

    # Reject comm-mode pronoun mismatch against Tenken glossary
    bad_comm = {
        "id": "neg_tenken_comm_mismatch",
        "series": "tenken",
        "speaker": "Fran",
        "addressee": "Shishou",
        "mode": "comm",
        "ja": "師匠、大丈夫？",
        "allowed_pronouns": ["ông"],
        "expected_names": [],
    }
    ok_comm, _ = verify_pronoun_cue(bad_comm)
    assert ok_comm is False

    # Reject non-honorific pronoun on HON:sama cue
    bad_hon = dict(cues_by_id["futsutsuka_01"])
    bad_hon["allowed_pronouns"] = ["ông"]
    ok_hon, _ = verify_pronoun_cue(bad_hon)
    assert ok_hon is False

    # Reject missing expected hint or expected name
    bad_hint = dict(cues_by_id["kansai_01"])
    bad_hint["ja"] = "和葉、こんにちは。"
    ok_hint, _ = verify_pronoun_cue(bad_hint)
    assert ok_hint is False


def test_verify_pronoun_cue_invokes_cast_manager_and_linter(monkeypatch):
    """Verify verify_pronoun_cue genuinely exercises HysteresisCastManager and lint_cue."""
    import subtrans.eval.runner as runner_mod
    from subtrans.dynamic_cast import HysteresisCastManager

    cues_by_id = {c["id"]: c for c in load_yaml("pronoun_suite.yaml").get("cues", [])}
    calls = {"from_glossary": 0, "lint_cue": 0}
    orig_from_glossary = HysteresisCastManager.from_glossary
    orig_lint_cue = runner_mod.lint_cue

    def _spy_from_glossary(cls, glossary):
        calls["from_glossary"] += 1
        return orig_from_glossary(glossary)

    def _spy_lint_cue(*args, **kwargs):
        calls["lint_cue"] += 1
        return orig_lint_cue(*args, **kwargs)

    monkeypatch.setattr(HysteresisCastManager, "from_glossary", classmethod(_spy_from_glossary))
    monkeypatch.setattr(runner_mod, "lint_cue", _spy_lint_cue)

    ok, msg = runner_mod.verify_pronoun_cue(cues_by_id["tenken_01"])
    assert ok is True, msg
    assert calls["from_glossary"] >= 1
    assert calls["lint_cue"] >= 3
