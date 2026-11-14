# -*- coding: utf-8 -*-
"""
subtrans/tests/test_fts5_kanji.py
Verification suite for Milestone 3 (M3):
1. Clean SQLite connection handle closing preventing Windows WinError 32 on immediate cleanup.
2. SQLite FTS5 character 2-gram and 3-gram indexing for 2-character Kanji keywords (魔術, 聖杯, 令呪).
3. Search and few-shot retrieval accuracy, series isolation, and schema migration synchronization.
"""

import os
import shutil
import tempfile
import sqlite3
from pathlib import Path
import pytest

from subtrans.cache import TranslationCache, compute_cue_cache_key
from subtrans.rag import KnowledgeBase, _generate_ngram_string
from subtrans.models import Cue


def test_sqlite_handle_clean_closing_no_winerror32(tmp_path):
    """
    Verifies that TranslationCache and KnowledgeBase cleanly close their SQLite handles
    upon leaving connection context blocks, allowing immediate file deletion without WinError 32.
    """
    cache_db = tmp_path / "test_cleanup_cache.db"
    kb_db = tmp_path / "test_cleanup_kb.db"

    # 1. Exercise TranslationCache
    cache = TranslationCache(db_path=cache_db)
    key = compute_cue_cache_key("gemini-3.5-flash-lite", "これはテストです", "This is a test")
    cache.set_cue(key, "gemini", "gemini-3.5-flash-lite", "Đây là bài kiểm tra")
    cached = cache.get_cue(key)
    assert cached == "Đây là bài kiểm tra"
    assert cache.count() >= 1

    cache.log_request("gemini-3.5-flash-lite", "gemini", cues_count=1)
    cache.mark_model_exhausted("gemini-test-model", reason="test")
    assert cache.is_model_exhausted("gemini-test-model") is True
    assert cache.get_model_exhaustion_info("gemini-test-model") is not None

    # 2. Exercise KnowledgeBase
    kb = KnowledgeBase(db_path=kb_db)
    kb.add_gold_unit(
        ja_text="魔術師たちの戦い",
        vi_text="Trận chiến của các ma thuật sư",
        series="fate",
        status="gold",
    )
    cues = [Cue(id=1, start=0, end=1000, ja="魔術を使う")]
    shots = kb.retrieve_few_shots(cues, series="fate", max_examples=1)
    assert len(shots) == 1

    kb.record_correction(
        ja_text="ウルシ、待て！",
        wrong_vi="Ô Lỗ Thi, đợi đã!",
        corrected_vi="Urushi, đợi đã!",
        category="banned_transliteration",
        series="tenken",
    )
    cautions = kb.retrieve_cautions([Cue(id=2, start=0, end=1000, ja="ウルシ、待て！")], series="tenken")
    assert len(cautions) >= 1
    stats = kb.get_stats()
    assert stats["gold_units_count"] >= 1

    # 3. Immediate deletion: must NOT raise PermissionError / WinError 32
    # If connection was leaked, Windows raises PermissionError: [WinError 32]
    os.remove(cache_db)
    os.remove(kb_db)

    # Verify files are deleted
    assert not cache_db.exists()
    assert not kb_db.exists()


def test_fts5_kanji_2char_keywords_retrieval(tmp_path):
    """
    Verifies SQLite FTS5 2-character Kanji keywords retrieval (魔術, 聖杯, 令呪)
    which fail with built-in trigram tokenizers.
    """
    db_file = tmp_path / "test_kanji_rag.db"
    kb = KnowledgeBase(db_path=db_file)

    # Insert verified Anime terminology units
    kb.add_gold_unit(
        ja_text="これは聖杯戦争の魔術です",
        vi_text="Đây là ma thuật của Chén Thánh",
        en_text="This is Holy Grail War magecraft.",
        series="fate",
        status="gold",
    )
    kb.add_gold_unit(
        ja_text="令呪をもって命ずる、我がサーヴァントよ",
        vi_text="Dùng lệnh chú ra lệnh, hỡi Servant của ta",
        en_text="By the command seal I order you, my Servant.",
        series="fate",
        status="gold",
    )
    kb.add_gold_unit(
        ja_text="マスターとサーヴァントの契約",
        vi_text="Khế ước giữa Master và Servant",
        en_text="The contract between Master and Servant.",
        series="fate",
        status="gold",
    )

    # Test 1: 2-character Kanji query "魔術"
    cues_majutsu = [Cue(id=1, start=0, end=1000, ja="強力な魔術を行使する")]
    shots_majutsu = kb.retrieve_few_shots(cues_majutsu, series="fate")
    assert len(shots_majutsu) >= 1
    assert "ma thuật của Chén Thánh" in shots_majutsu[0]["vi"]

    # Test 2: 2-character Kanji query "聖杯"
    cues_seihai = [Cue(id=2, start=1000, end=2000, ja="万能の願望機たる聖杯")]
    shots_seihai = kb.retrieve_few_shots(cues_seihai, series="fate")
    assert len(shots_seihai) >= 1
    assert "Chén Thánh" in shots_seihai[0]["vi"]

    # Test 3: 2-character Kanji query "令呪"
    cues_reiju = [Cue(id=3, start=2000, end=3000, ja="令呪を消費して宝具を解放する")]
    shots_reiju = kb.retrieve_few_shots(cues_reiju, series="fate")
    assert len(shots_reiju) >= 1
    assert "lệnh chú" in shots_reiju[0]["vi"]

    # Test 4: Katakana term "マスター"
    cues_master = [Cue(id=4, start=3000, end=4000, ja="マスターの指示に従え")]
    shots_master = kb.retrieve_few_shots(cues_master, series="fate")
    assert len(shots_master) >= 1
    assert any("Master" in s["vi"] for s in shots_master)


def test_search_gold_units_inverted_index(tmp_path):
    """
    Verifies KnowledgeBase.search_gold_units directly querying gold_units_fts via FTS5 MATCH.
    """
    db_file = tmp_path / "test_search_fts5.db"
    kb = KnowledgeBase(db_path=db_file)

    kb.add_gold_unit(
        ja_text="聖杯戦争における魔術師の役割",
        vi_text="Vai trò của ma thuật sư trong Chén Thánh Chiến",
        series="fate",
        status="gold",
    )
    kb.add_gold_unit(
        ja_text="令呪は三画のみ与えられる",
        vi_text="Lệnh chú chỉ được ban cho ba nét",
        series="fate",
        status="reviewed",
    )

    # Search for 2-character Kanji "魔術"
    hits_majutsu = kb.search_gold_units("魔術", top_k=5, series="fate")
    assert len(hits_majutsu) == 1
    assert "ma thuật sư" in hits_majutsu[0]["vi"]
    assert hits_majutsu[0]["score"] > 0

    # Search for 2-character Kanji "令呪"
    hits_reiju = kb.search_gold_units("令呪", top_k=5, series="fate")
    assert len(hits_reiju) == 1
    assert "Lệnh chú" in hits_reiju[0]["vi"]

    # Search for multi-term query
    hits_multi = kb.search_gold_units("聖杯と魔術", top_k=5, series="fate")
    assert len(hits_multi) >= 1

    # Empty query should return empty list
    assert kb.search_gold_units("") == []
    assert kb.search_gold_units("   ") == []


def test_generate_ngram_string_spec():
    """
    Verifies _generate_ngram_string behavior:
    - Strips furigana and non-word characters.
    - If length == 1, includes the single character.
    - If length >= 2, includes all 2-char slices.
    - If length >= 3, includes all 3-char slices.
    """
    # Empty
    assert _generate_ngram_string("") == ""
    assert _generate_ngram_string("   ") == ""
    assert _generate_ngram_string("！？。、") == ""

    # Length == 1
    assert _generate_ngram_string("剣") == "剣"

    # Length == 2
    assert _generate_ngram_string("魔術") == "魔術"

    # Length == 3
    # 2-grams: "聖杯", "杯戦"; 3-grams: "聖杯戦"
    assert _generate_ngram_string("聖杯戦") == "聖杯 杯戦 聖杯戦"

    # Furigana transparency: strips [ふりがな]
    raw_with_furi = "魔術[まじゅつ]"
    assert _generate_ngram_string(raw_with_furi) == "魔術"

    # Complex Japanese sentence with punctuation
    text = "これは、聖杯戦争の魔術！"
    ngram_str = _generate_ngram_string(text)
    # Check that punctuation is stripped and 2-grams / 3-grams exist
    assert "、" not in ngram_str
    assert "！" not in ngram_str
    assert "魔術" in ngram_str
    assert "聖杯" in ngram_str
    assert "聖杯戦" in ngram_str


def test_fts5_series_isolation(tmp_path):
    """
    Verifies that FTS5 query results respect series filtering and do not leak across series.
    """
    db_file = tmp_path / "test_isolation.db"
    kb = KnowledgeBase(db_path=db_file)

    kb.add_gold_unit(ja_text="古代の魔術", vi_text="Ma thuật cổ đại (Fate)", series="fate", status="gold")
    kb.add_gold_unit(ja_text="黒猫の魔術", vi_text="Ma thuật của Hắc Miêu (Tenken)", series="tenken", status="gold")
    kb.add_gold_unit(ja_text="一般魔術概論", vi_text="Khái luận ma thuật chung", series=None, status="gold")

    # Query Fate
    cues = [Cue(id=1, start=0, end=1000, ja="魔術を調べる")]
    shots_fate = kb.retrieve_few_shots(cues, series="fate", max_examples=10)
    fate_vis = [s["vi"] for s in shots_fate]
    assert "Ma thuật cổ đại (Fate)" in fate_vis
    assert "Khái luận ma thuật chung" in fate_vis
    assert "Ma thuật của Hắc Miêu (Tenken)" not in fate_vis

    # Query Tenken
    shots_tenken = kb.retrieve_few_shots(cues, series="tenken", max_examples=10)
    tenken_vis = [s["vi"] for s in shots_tenken]
    assert "Ma thuật của Hắc Miêu (Tenken)" in tenken_vis
    assert "Khái luận ma thuật chung" in tenken_vis
    assert "Ma thuật cổ đại (Fate)" not in tenken_vis


def test_gold_units_fts_schema_migration_sync(tmp_path):
    """
    Verifies that an existing database without gold_units_fts or ja_ngram column
    is automatically migrated and synchronized when opened with KnowledgeBase.
    """
    db_file = tmp_path / "test_legacy_db.db"

    # Simulate legacy database created by previous version
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE gold_units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series TEXT,
            episode TEXT,
            ja_text TEXT NOT NULL,
            en_text TEXT,
            vi_text TEXT NOT NULL,
            speaker TEXT,
            addressee TEXT,
            status TEXT NOT NULL DEFAULT 'gold',
            lang_pair TEXT NOT NULL DEFAULT 'ja-vi',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        INSERT INTO gold_units (series, ja_text, vi_text, status)
        VALUES ('fate', '聖杯探索の旅', 'Hành trình tìm kiếm Chén Thánh', 'gold')
    """)
    conn.commit()
    conn.close()

    # Now open with KnowledgeBase: should migrate schema and populate FTS5
    kb = KnowledgeBase(db_path=db_file)
    hits = kb.search_gold_units("聖杯", series="fate")
    assert len(hits) == 1
    assert "Chén Thánh" in hits[0]["vi"]
