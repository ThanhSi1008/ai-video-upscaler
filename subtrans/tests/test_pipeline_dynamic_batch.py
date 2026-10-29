# -*- coding: utf-8 -*-
"""
subtrans/tests/test_pipeline_dynamic_batch.py
Unit tests verifying Dynamic Batch Sizing in subtrans/pipeline.py:
- High dialogue density (>380 cues) automatically scales batch down to 70
- Very high dialogue density (>450 cues) automatically scales batch down to 60
- User-specified batch_size overrides dynamic sizing unconditionally
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from subtrans.pipeline import TranslationPipeline
from subtrans.models import Cue


def test_dynamic_batch_sizing_scales_for_dense_dialogue(tmp_path, monkeypatch):
    """Verifies that dense dialogue (>380 cues) automatically scales batch size to 70."""
    pipeline = TranslationPipeline(dry_run=True, output_dir=tmp_path, batch_size=None, enable_cascade=False, enable_rag=False)

    # 400 dialogue cues
    cues = [Cue(id=i, start=i * 1000, end=(i + 1) * 1000, ja=f"セリフ {i}", en=f"Line {i}") for i in range(400)]

    # Mock parse_and_align_subtitles to return our 400 cues
    monkeypatch.setattr("subtrans.pipeline.parse_and_align_subtitles", lambda **kw: (cues, MagicMock(), MagicMock()))
    monkeypatch.setattr("subtrans.pipeline.load_or_generate_series_cast_sheet", lambda **kw: None)

    batches_captured = []
    original_translate_batch = pipeline.cache

    # We mock _call_api or check dialogue_batches by running pipeline with dry_run
    dummy_ja = tmp_path / "dense.ja.srt"
    dummy_ja.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")

    stats = pipeline.run(dummy_ja)
    assert stats is not None
    # With 400 cues and optimal batch = 70: total batches = ceil(400 / 70) = 6 batches
    # If it remained 100: total batches would be 4
    total_batches = pipeline.gemini_backends["gemini-3.5-flash-lite"].stats["total_batches"]
    assert total_batches == 6  # 70 * 5 + 50 = 400 cues across 6 batches


def test_dynamic_batch_sizing_scales_for_very_dense_dialogue(tmp_path, monkeypatch):
    """Verifies that very dense dialogue (>=450 cues) automatically scales batch size to 60."""
    pipeline = TranslationPipeline(dry_run=True, output_dir=tmp_path, batch_size=None, enable_cascade=False, enable_rag=False)

    # 480 dialogue cues
    cues = [Cue(id=i, start=i * 1000, end=(i + 1) * 1000, ja=f"セリフ {i}", en=f"Line {i}") for i in range(480)]

    monkeypatch.setattr("subtrans.pipeline.parse_and_align_subtitles", lambda **kw: (cues, MagicMock(), MagicMock()))
    monkeypatch.setattr("subtrans.pipeline.load_or_generate_series_cast_sheet", lambda **kw: None)

    dummy_ja = tmp_path / "very_dense.ja.srt"
    dummy_ja.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")

    stats = pipeline.run(dummy_ja)
    assert stats is not None
    # With 480 cues and optimal batch = 60: total batches = 480 / 60 = 8 batches
    total_batches = pipeline.gemini_backends["gemini-3.5-flash-lite"].stats["total_batches"]
    assert total_batches == 8


def test_user_explicit_batch_size_preserved(tmp_path, monkeypatch):
    """Verifies that user's explicit batch_size is strictly preserved even on dense episodes."""
    pipeline = TranslationPipeline(dry_run=True, output_dir=tmp_path, batch_size=50, enable_cascade=False, enable_rag=False)

    # 400 dialogue cues
    cues = [Cue(id=i, start=i * 1000, end=(i + 1) * 1000, ja=f"セリフ {i}", en=f"Line {i}") for i in range(400)]

    monkeypatch.setattr("subtrans.pipeline.parse_and_align_subtitles", lambda **kw: (cues, MagicMock(), MagicMock()))
    monkeypatch.setattr("subtrans.pipeline.load_or_generate_series_cast_sheet", lambda **kw: None)

    dummy_ja = tmp_path / "user_explicit.ja.srt"
    dummy_ja.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")

    stats = pipeline.run(dummy_ja)
    assert stats is not None
    # 400 cues with explicit batch 50 -> 8 batches
    total_batches = pipeline.gemini_backends["gemini-3.5-flash-lite"].stats["total_batches"]
    assert total_batches == 8
