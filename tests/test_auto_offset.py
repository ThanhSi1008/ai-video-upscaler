# -*- coding: utf-8 -*-
"""
tests/test_auto_offset.py
Unit tests for automatic subtitle offset detection and application.
"""

import os
from pathlib import Path
import pytest
from subtrans.parser import detect_auto_offset, compute_alignment_match_rate


def test_detect_auto_offset_on_real_episodes():
    base_dir = Path(os.environ.get("RAW_SOURCES_DIR", "samples/raw_sources"))
    if not base_dir.exists():
        pytest.skip("RawSources directory not available")

    # Ep 1: Kimi ga Shinu made Koi wo Shitai 13 (natural sync, should not apply)
    ja_ep1 = base_dir / "Kimi_ga_Shinu_made_Koi_wo_Shitai" / "Kimi_ga_Shinu_made_Koi_wo_Shitai_S01E13.ja.srt"
    en_ep1 = base_dir / "Kimi_ga_Shinu_made_Koi_wo_Shitai" / "Kimi_ga_Shinu_made_Koi_wo_Shitai_S01E13.en.srt"
    if ja_ep1.exists() and en_ep1.exists():
        offset_ms, is_applied, reason, stats = detect_auto_offset(str(ja_ep1), str(en_ep1))
        assert not is_applied, f"Ep 1 should not apply auto-offset: {reason}"
        assert abs(offset_ms) < 1000

    # Ep 2: Re Zero 84 (natural sync, should not apply)
    ja_ep2 = base_dir / "Re_Zero_kara_Hajimeru_Isekai_Seikatsu" / "Re_Zero_kara_Hajimeru_Isekai_Seikatsu_S04E84.ja.srt"
    en_ep2 = base_dir / "Re_Zero_kara_Hajimeru_Isekai_Seikatsu" / "Re_Zero_kara_Hajimeru_Isekai_Seikatsu_S04E84.en.srt"
    if ja_ep2.exists() and en_ep2.exists():
        offset_ms, is_applied, reason, stats = detect_auto_offset(str(ja_ep2), str(en_ep2))
        assert not is_applied, f"Ep 2 should not apply auto-offset: {reason}"
        assert abs(offset_ms) < 1000

    # Ep 3: Slime S4-24 (sponsor card gap, should detect ~ -11141 ms and apply)
    ja_ep3 = base_dir / "Tensei_Shitara_Slime_Datta_Ken" / "Tensei_Shitara_Slime_Datta_Ken_S04E24.ja.srt"
    en_ep3 = base_dir / "Tensei_Shitara_Slime_Datta_Ken" / "Tensei_Shitara_Slime_Datta_Ken_S04E24.en.srt"
    if ja_ep3.exists() and en_ep3.exists():
        offset_ms, is_applied, reason, stats = detect_auto_offset(str(ja_ep3), str(en_ep3))
        assert is_applied, f"Ep 3 should apply auto-offset: {reason}"
        assert -11300 <= offset_ms <= -11000, f"Offset {offset_ms} not within expected range"
        assert stats["pairs_count"] >= 30
        assert stats["quintile_drift_sec"] <= 0.5
        assert stats["rate_diff_points"] >= 15.0


def test_detect_auto_offset_missing_file():
    offset_ms, is_applied, reason, stats = detect_auto_offset("/nonexistent/ja.srt", None)
    assert not is_applied
    assert offset_ms == 0
    assert "Không có file phụ đề tiếng Anh" in reason


def test_queue_auto_offset_cli_flags():
    from upscale import build_arg_parser
    parser = build_arg_parser()

    # Case 1: Default queue flags
    args = parser.parse_args(["--queue", "a.mkv", "b.mkv"])
    assert args.auto_offset is None
    assert args.no_auto_offset is False
    assert args.sub_offset_target == ""

    # Case 2: Explicit --no-auto-offset
    args2 = parser.parse_args(["--queue", "a.mkv", "b.mkv", "--no-auto-offset"])
    assert args2.no_auto_offset is True

    # Case 3: Targeted sub-offset
    args3 = parser.parse_args(["--queue", "a.mkv", "b.mkv", "--sub-offset-ms", "-11154", "--sub-offset-target", "b.mkv"])
    assert args3.sub_offset_ms == -11154
    assert args3.sub_offset_target == "b.mkv"


def test_auto_align_en_cli_flags():
    from upscale import build_arg_parser
    parser = build_arg_parser()

    # Mặc định auto_align_en là True
    args_default = parser.parse_args(["video.mkv"])
    assert args_default.auto_align_en is True

    # Cờ tắt chủ động
    args_disabled = parser.parse_args(["video.mkv", "--no-auto-align-en"])
    assert args_disabled.auto_align_en is False

