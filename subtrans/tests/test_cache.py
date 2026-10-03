# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from subtrans.cache import TranslationCache, compute_cue_cache_key


def test_translation_cache(tmp_path):
    db_file = tmp_path / "test_trans.db"
    cache = TranslationCache(db_path=db_file)

    key = compute_cue_cache_key("gemini-2.5-flash", "こんにちは", "Hello")
    assert cache.get_cue(key) is None

    cache.set_cue(key, "gemini", "gemini-2.5-flash", "Xin chào")
    assert cache.get_cue(key) == "Xin chào"
    assert cache.count() == 1
