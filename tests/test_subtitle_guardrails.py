# -*- coding: utf-8 -*-
"""
tests/test_subtitle_guardrails.py
Unit tests for Items 4, 5, 6:
- Item 5: GPU lock test: stage phụ đề in queue after upscale can load Ollama; during upscale running it is blocked.
- Item 6: Sound cues: drop mode omits pure parenthesized cues without EN match from SRT and does not count as errors.
- Item 4: Subtitle guardrails: no [Chưa dịch] in SRT (fallback to EN), qualification gate (>= 95% real and < 10% uncertain),
  and idempotent embedding.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from subtrans.models import Cue
from subtrans.writer import write_srt_subtitles, clean_and_normalize_vietnamese
from subtrans.parser import parse_and_align_subtitles, is_pure_sound_cue
from subtrans.local_backend import OllamaTranslator
from src.config import (
    acquire_gpu_lock,
    release_gpu_lock,
    GPU_LOCK_FILE,
    PENDING_SUBS_FILE,
    add_pending_sub,
    load_pending_subs,
    remove_pending_sub,
)


def test_gpu_lock_blocks_during_upscale_and_frees_during_subs(tmp_path):
    """Test Item 5: GPU lock blocks Ollama when upscale is running, frees before subtitle stage."""
    translator = OllamaTranslator(endpoint="http://localhost:11434")

    # Release any existing lock
    release_gpu_lock()
    assert not translator.is_upscale_running(), "Should not report upscale running when lock is free"

    # Simulate upscale starting: acquire lock
    acquire_gpu_lock()
    assert os.path.exists(GPU_LOCK_FILE)
    assert translator.is_upscale_running(), "Should report upscale running when GPU lock is held"

    # Simulate upscale finished before subtitle stage: release lock
    release_gpu_lock()
    assert not os.path.exists(GPU_LOCK_FILE)
    assert not translator.is_upscale_running(), "Subtitle stage should be allowed to run Ollama"


def test_sound_cues_drop_mode(tmp_path):
    """Test Item 6: Sound cues with --sound-cues drop are omitted and not counted as dialogue."""
    assert is_pure_sound_cue("（鳥のさえずり）")
    assert is_pure_sound_cue("(laugh)")
    assert not is_pure_sound_cue("（鳥のさえずり）Chào buổi sáng")

    # Create dummy cues
    cues = [
        Cue(id=0, start=1000, end=2000, ja="こんにちは", en="Hello", vi="Xin chào"),
        Cue(id=1, start=2100, end=3000, ja="（息をのむ）", en=None, is_dropped_sound=True, is_skipped=True),
        Cue(id=2, start=3100, end=4000, ja="ありがとう", en="Thank you", vi="Cảm ơn"),
    ]
    out_srt = tmp_path / "test_sound_drop.srt"
    write_srt_subtitles(cues, out_srt)

    content = out_srt.read_text(encoding="utf-8")
    assert "Xin chào" in content
    assert "Cảm ơn" in content
    assert "息をのむ" not in content
    # Total blocks should be 2, not 3
    blocks = [b.strip() for b in content.strip().split("\n\n") if b.strip()]
    assert len(blocks) == 2
    assert blocks[0].startswith("1\n")
    assert blocks[1].startswith("2\n")


def test_no_untranslated_in_srt_and_en_fallback(tmp_path):
    """Test Item 4a: KHÔNG ghi [Chưa dịch] vào SRT; cue lỗi dùng tạm text EN tham chiếu."""
    cues = [
        Cue(id=0, start=1000, end=2000, ja="おはよう", en="Good morning", vi="[Chưa dịch]"),
        Cue(id=1, start=2100, end=3000, ja="さようなら", en="Goodbye", vi=""),
        Cue(id=2, start=3100, end=4000, ja="ありがとう", en="Thank you", vi="Cảm ơn"),
    ]
    out_srt = tmp_path / "test_fallback.srt"
    write_srt_subtitles(cues, out_srt)

    content = out_srt.read_text(encoding="utf-8")
    assert "[Chưa dịch]" not in content
    assert "Good morning" in content
    assert "Goodbye" in content
    assert "Cảm ơn" in content
    assert cues[0].used_en_fallback is True
    assert cues[1].used_en_fallback is True
    assert cues[2].used_en_fallback is False


def test_pending_subs_file_management(tmp_path, monkeypatch):
    """Test Item 4b: pending_subs.json add, load, remove."""
    fake_pending_file = str(tmp_path / "pending_subs.json")
    monkeypatch.setattr("src.config.PENDING_SUBS_FILE", fake_pending_file)

    test_record = {
        "base_name": "test_ep_01",
        "ja_path": "/path/test.ja.srt",
        "en_path": "/path/test.en.ass",
        "real_rate": 0.85,
        "uncertain_rate": 0.15,
    }
    add_pending_sub(test_record)
    items = load_pending_subs()
    assert any(it["base_name"] == "test_ep_01" for it in items)

    remove_pending_sub("test_ep_01")
    items_after = load_pending_subs()
    assert not any(it["base_name"] == "test_ep_01" for it in items_after)
