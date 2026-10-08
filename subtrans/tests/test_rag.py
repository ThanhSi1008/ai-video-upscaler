# -*- coding: utf-8 -*-
"""
subtrans/tests/test_rag.py
Unit tests for Subtitle RAG & Error Bank (subtrans/rag.py).
"""

import pytest
from subtrans.models import Cue
from subtrans.rag import KnowledgeBase


def test_knowledge_base_seeding(tmp_path):
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)
    stats = kb.get_stats()
    assert stats["error_bank_rules_count"] >= 7
    assert stats["gold_units_count"] == 0


def test_add_gold_unit_and_retrieval(tmp_path):
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    # Insert sample gold units
    ok1 = kb.add_gold_unit(
        ja_text="師匠、カレー食べたい",
        vi_text="Sư phụ, em muốn ăn cà ri.",
        en_text="Master, I want to eat curry.",
        series="tenken",
        episode="E01",
    )
    assert ok1 is True

    # Duplicate should not be re-inserted
    ok_dup = kb.add_gold_unit(
        ja_text="師匠、カレー食べたい",
        vi_text="Sư phụ, em muốn ăn cà ri.",
        series="tenken",
    )
    assert ok_dup is False

    # Test retrieval
    cues = [Cue(id=1, start=0, end=1000, ja="カレーを作ろう")]
    few_shots = kb.retrieve_few_shots(cues, series="tenken", max_examples=2)
    assert len(few_shots) == 1
    assert "cà ri" in few_shots[0]["vi"]


def test_retrieve_error_cautions(tmp_path):
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    # Cue calling Fran -> should match vocative error bank caution
    cues = [Cue(id=1, start=0, end=1000, ja="フラン、行くよ！")]
    cautions = kb.retrieve_cautions(cues, series="tenken", max_cautions=2)
    assert len(cautions) >= 1
    assert any("Fran" in c["prefer"] for c in cautions)
