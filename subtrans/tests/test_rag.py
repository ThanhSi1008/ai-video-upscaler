# -*- coding: utf-8 -*-
"""
subtrans/tests/test_rag.py
Unit tests for Subtitle RAG & Error Bank (subtrans/rag.py).
Comprehensive test suite verifying SQLite persistence, normalization, series isolation,
keyword matching, error bank cautions, and multi-file triplet import.
"""

from pathlib import Path
import unicodedata
import pytest
import pysubs2

from subtrans.models import Cue
from subtrans.rag import KnowledgeBase


def test_knowledge_base_seeding(tmp_path):
    """Verifies that KnowledgeBase initializes tables and seeds standard error bank rules."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)
    stats = kb.get_stats()
    assert stats["error_bank_rules_count"] >= 7
    assert stats["gold_units_count"] == 0
    assert stats["distinct_series_count"] == 0


def test_knowledge_base_empty_or_whitespace_rejection(tmp_path):
    """Verifies that empty or whitespace-only inputs are rejected."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    assert kb.add_gold_unit("", "Xin chào") is False
    assert kb.add_gold_unit("   ", "Xin chào") is False
    assert kb.add_gold_unit("こんにちは", "") is False
    assert kb.add_gold_unit("こんにちは", "   \t\n") is False

    stats = kb.get_stats()
    assert stats["gold_units_count"] == 0


def test_knowledge_base_unicode_nfc_auto_normalization(tmp_path):
    """Verifies that inserted strings are automatically normalized to Unicode NFC."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    nfd_vi = unicodedata.normalize("NFD", "Tiếng Việt chuẩn xác")
    assert not unicodedata.is_normalized("NFC", nfd_vi)

    ok = kb.add_gold_unit(
        ja_text="テスト",
        vi_text=nfd_vi,
        series="test_series",
    )
    assert ok is True

    cues = [Cue(id=1, start=0, end=1000, ja="テスト")]
    retrieved = kb.retrieve_few_shots(cues, series="test_series", max_examples=1)
    assert len(retrieved) == 1
    assert unicodedata.is_normalized("NFC", retrieved[0]["vi"])


def test_add_gold_unit_and_retrieval(tmp_path):
    """Verifies basic gold unit insertion, duplicate prevention, and keyword-based retrieval."""
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
    assert few_shots[0]["en"] == "Master, I want to eat curry."


def test_retrieve_few_shots_series_isolation(tmp_path):
    """Verifies that retrieval strictly isolates series and does not leak cross-series units."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    # Tenken gold unit
    kb.add_gold_unit(
        ja_text="冒険者ギルドへ行こう",
        vi_text="Đi tới Guild Mạo hiểm giả thôi.",
        series="tenken",
    )
    # Futsutsuka gold unit (uses same term 冒険者 for testing isolation)
    kb.add_gold_unit(
        ja_text="冒険者の身分を隠す",
        vi_text="Che giấu thân phận người phiêu lưu.",
        series="futsutsuka",
    )
    # Generic gold unit (series is None)
    kb.add_gold_unit(
        ja_text="冒険者とは何か",
        vi_text="Thế nào là một mạo hiểm giả?",
        series=None,
    )

    cues = [Cue(id=1, start=0, end=1000, ja="あの冒険者は誰？")]

    # Query Tenken: should get tenken + None, but NOT futsutsuka
    res_tenken = kb.retrieve_few_shots(cues, series="tenken", max_examples=5)
    tenken_vis = [r["vi"] for r in res_tenken]
    assert "Đi tới Guild Mạo hiểm giả thôi." in tenken_vis
    assert "Thế nào là một mạo hiểm giả?" in tenken_vis
    assert "Che giấu thân phận người phiêu lưu." not in tenken_vis

    # Query Futsutsuka: should get futsutsuka + None, but NOT tenken
    res_futsutsuka = kb.retrieve_few_shots(cues, series="futsutsuka", max_examples=5)
    futsutsuka_vis = [r["vi"] for r in res_futsutsuka]
    assert "Che giấu thân phận người phiêu lưu." in futsutsuka_vis
    assert "Thế nào là một mạo hiểm giả?" in futsutsuka_vis
    assert "Đi tới Guild Mạo hiểm giả thôi." not in futsutsuka_vis


def test_retrieve_few_shots_no_keywords_or_empty(tmp_path):
    """Verifies that empty cues or cues without Kanji/Katakana gracefully return empty lists."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)
    kb.add_gold_unit(ja_text="魔法の剣", vi_text="Kiếm ma thuật", series="tenken")

    # Empty list
    assert kb.retrieve_few_shots([], series="tenken") == []

    # Only hiragana / punctuation (no Kanji / Katakana >= 2 chars)
    cues_hiragana = [Cue(id=1, start=0, end=1000, ja="はい、そうです。ありがとう！")]
    assert kb.retrieve_few_shots(cues_hiragana, series="tenken") == []


def test_retrieve_few_shots_deduplication_and_limit(tmp_path):
    """Verifies that multiple cues matching the same gold unit do not duplicate results."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)
    kb.add_gold_unit(ja_text="黒猫族の誇り", vi_text="Niềm tự hào của Hắc Miêu tộc", series="tenken")
    kb.add_gold_unit(ja_text="黒猫族の少女", vi_text="Thiếu nữ người Hắc Miêu tộc", series="tenken")

    # Two cues both containing 黒猫
    cues = [
        Cue(id=1, start=0, end=1000, ja="黒猫族だ！"),
        Cue(id=2, start=1000, end=2000, ja="本当に黒猫族？"),
    ]
    res = kb.retrieve_few_shots(cues, series="tenken", max_examples=1)
    assert len(res) == 1

    res2 = kb.retrieve_few_shots(cues, series="tenken", max_examples=5)
    assert len(res2) == 2
    # Ensure items are distinct
    assert res2[0]["vi"] != res2[1]["vi"]


def test_retrieve_error_cautions(tmp_path):
    """Verifies pattern matching and caution retrieval from Error Bank."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    # Cue calling Fran -> should match vocative error bank caution
    cues = [Cue(id=1, start=0, end=1000, ja="フラン、行くよ！")]
    cautions = kb.retrieve_cautions(cues, series="tenken", max_cautions=2)
    assert len(cautions) >= 1
    assert any("Fran" in c["prefer"] for c in cautions)


def test_retrieve_error_cautions_series_isolation_and_generic(tmp_path):
    """Verifies series-specific caution rules and generic rules."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    # Reirin is futsutsuka-specific; Fran is tenken-specific
    cues_reirin = [Cue(id=1, start=0, end=1000, ja="玲琳、待ってください")]

    # Querying as Tenken should NOT trigger Reirin caution
    cautions_tenken = kb.retrieve_cautions(cues_reirin, series="tenken")
    assert not any("Reirin" in c["prefer"] for c in cautions_tenken)

    # Querying as Futsutsuka SHOULD trigger Reirin caution
    cautions_futsutsuka = kb.retrieve_cautions(cues_reirin, series="futsutsuka")
    assert any("Reirin" in c["prefer"] for c in cautions_futsutsuka)

    # Generic caution: 世界観 applies to both series
    cues_sekai = [Cue(id=2, start=0, end=1000, ja="この世界観は素晴らしい")]
    assert any("thế giới" in c["prefer"] for c in kb.retrieve_cautions(cues_sekai, series="tenken"))
    assert any("thế giới" in c["prefer"] for c in kb.retrieve_cautions(cues_sekai, series="futsutsuka"))


def test_retrieve_error_cautions_empty_or_no_match(tmp_path):
    """Verifies empty cues or non-matching cues return empty caution list."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    assert kb.retrieve_cautions([], series="tenken") == []

    cues_normal = [Cue(id=1, start=0, end=1000, ja="普通の日常会話です。")]
    assert kb.retrieve_cautions(cues_normal, series="tenken") == []


def test_import_subtitles_triplet_full_flow(tmp_path):
    """
    Verifies full import of JA, EN, VI subtitle triplet:
    - Strips ASS styling tags (e.g. {\\an8}) and newline escapes (\\N).
    - Skips comment events.
    - Requires timestamp overlap >= 300ms.
    - Matches English event if present.
    """
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    ja_file = tmp_path / "ep01.ja.ass"
    vi_file = tmp_path / "ep01.vi.ass"
    en_file = tmp_path / "ep01.en.ass"

    # Create JA ASS file
    ja_ssa = pysubs2.SSAFile()
    ja_ssa.events.append(pysubs2.SSAEvent(start=1000, end=3000, text=r"{\an8}師匠、お腹空いた\N早くして"))
    evt_comment = pysubs2.SSAEvent(start=4000, end=6000, text="これはコメント行")
    evt_comment.is_comment = True
    ja_ssa.events.append(evt_comment)
    ja_ssa.events.append(pysubs2.SSAEvent(start=7000, end=9000, text="敵が来るぞ！"))
    ja_ssa.events.append(pysubs2.SSAEvent(start=10000, end=10500, text="ほんの少しのセリフ"))  # Overlap will be < 300ms
    ja_ssa.save(str(ja_file))

    # Create VI ASS file
    vi_ssa = pysubs2.SSAFile()
    vi_ssa.events.append(pysubs2.SSAEvent(start=1050, end=3000, text=r"Sư phụ, em đói rồi!\NNhanh lên nào."))
    vi_ssa.events.append(pysubs2.SSAEvent(start=7000, end=9000, text=r"{\b1}Kẻ địch đang tới!{\b0}"))
    vi_ssa.events.append(pysubs2.SSAEvent(start=10450, end=10550, text="Ngắn quá"))  # Overlap only 50ms (< 300ms)
    vi_ssa.save(str(vi_file))

    # Create EN ASS file
    en_ssa = pysubs2.SSAFile()
    en_ssa.events.append(pysubs2.SSAEvent(start=1000, end=2900, text="Master, I am hungry. Hurry up."))
    en_ssa.events.append(pysubs2.SSAEvent(start=7050, end=8950, text="Enemy is coming!"))
    en_ssa.save(str(en_file))

    count = kb.import_subtitles_triplet(
        ja_path=ja_file,
        en_path=en_file,
        vi_path=vi_file,
        series="tenken",
        episode="E01",
    )
    # Event 1 & Event 2 should be imported (Event with comment skipped, Event with <300ms overlap skipped)
    assert count == 2

    stats = kb.get_stats()
    assert stats["gold_units_count"] == 2
    assert stats["distinct_series_count"] == 1

    # Verify content in DB
    cues_query = [Cue(id=1, start=0, end=1000, ja="お腹空いた")]
    shots = kb.retrieve_few_shots(cues_query, series="tenken", max_examples=1)
    assert len(shots) == 1
    # Check that {\an8} and \N were cleaned
    assert r"{\an8}" not in shots[0]["ja"]
    assert r"\N" not in shots[0]["ja"]
    assert "Sư phụ, em đói rồi!" in shots[0]["vi"]
    assert "Master, I am hungry" in shots[0]["en"]


def test_import_subtitles_triplet_missing_files(tmp_path):
    """Verifies that missing files are handled gracefully without exceptions."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    missing_ja = tmp_path / "non_existent.ja.ass"
    valid_vi = tmp_path / "valid.vi.ass"
    pysubs2.SSAFile().save(str(valid_vi))

    assert kb.import_subtitles_triplet(missing_ja, None, valid_vi, series="tenken") == 0


def test_import_subtitles_triplet_without_en(tmp_path):
    """Verifies triplet import when EN subtitle file is absent (JA + VI only)."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    ja_file = tmp_path / "ep02.ja.ass"
    vi_file = tmp_path / "ep02.vi.ass"

    ja_ssa = pysubs2.SSAFile()
    ja_ssa.events.append(pysubs2.SSAEvent(start=1000, end=3000, text="勝利した！"))
    ja_ssa.save(str(ja_file))

    vi_ssa = pysubs2.SSAFile()
    vi_ssa.events.append(pysubs2.SSAEvent(start=1000, end=3000, text="Chiến thắng rồi!"))
    vi_ssa.save(str(vi_file))

    count = kb.import_subtitles_triplet(ja_file, None, vi_file, series="tenken", episode="E02")
    assert count == 1

    cues = [Cue(id=1, start=0, end=1000, ja="勝利")]
    shots = kb.retrieve_few_shots(cues, series="tenken")
    assert len(shots) == 1
    assert shots[0]["vi"] == "Chiến thắng rồi!"
    assert "en" not in shots[0]


def test_rag_status_filtering_draft_vs_gold(tmp_path):
    """Verifies that 'draft' units are strictly ignored by retrieve_few_shots, while 'reviewed'/'gold' are used."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    # Add a draft unit (e.g. unverified pipeline output)
    kb.add_gold_unit(
        ja_text="強力な魔法",
        vi_text="Phép thuật mạnh mẽ (bản nháp lỗi)",
        series="tenken",
        status="draft",
    )
    # Add a gold unit
    kb.add_gold_unit(
        ja_text="古代の魔法",
        vi_text="Ma thuật cổ đại chuẩn xác",
        series="tenken",
        status="gold",
    )

    cues = [Cue(id=1, start=0, end=1000, ja="魔法を使う")]
    shots = kb.retrieve_few_shots(cues, series="tenken")
    assert len(shots) == 1
    assert shots[0]["vi"] == "Ma thuật cổ đại chuẩn xác"
    # Ensure draft text never leaked into few-shot context
    assert not any("nháp lỗi" in s["vi"] for s in shots)


def test_rag_learn_from_correction_feedback_loop(tmp_path):
    """Verifies record_correction properly updates both Error Bank and Gold Units."""
    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    res = kb.record_correction(
        ja_text="ウルシ、伏せろ！",
        wrong_vi="Ô Lỗ Thi, nằm xuống!",
        corrected_vi="Urushi, nằm xuống!",
        category="banned_transliteration",
        explanation="Sửa âm Hán-Việt Ô Lỗ Thi thành Urushi",
        series="tenken",
    )
    assert res["status"] == "success"
    assert res["gold_added"] is True

    # Check Error Bank caution retrieval
    cues = [Cue(id=1, start=0, end=1000, ja="ウルシ、伏せろ！")]
    cautions = kb.retrieve_cautions(cues, series="tenken")
    assert any("Urushi" in c["prefer"] for c in cautions)

    # Check gold few-shot retrieval
    shots = kb.retrieve_few_shots(cues, series="tenken")
    assert any("Urushi, nằm xuống!" in s["vi"] for s in shots)


def test_learn_from_linter_violations_auto_feedback(tmp_path):
    """Verifies that linter violations are automatically extracted and recorded into error_bank."""
    from subtrans.linter import LintViolation, RULE_BANNED_TERM, RULE_CJK_LEAK, RULE_ROMAJI_MISSING

    db_file = tmp_path / "test_knowledge.db"
    kb = KnowledgeBase(db_path=db_file)

    cues = [
        Cue(id=1, start=1000, end=3000, ja="世界観が壊れる", vi="thế giới quan bị phá vỡ"),
        Cue(id=2, start=4000, end=6000, ja="フラン、待って", vi="Này cô em, đợi đã"),
        Cue(id=3, start=7000, end=9000, ja="漢字の漏れ", vi="Để sót 漢字 trong câu"),
    ]

    violations = [
        LintViolation(
            rule_id=RULE_BANNED_TERM,
            severity="ERROR",
            cue_id=1,
            message="Contains banned translationese/calque 'thế giới quan'",
            suggestion="Use 'thế giới' instead",
        ),
        LintViolation(
            rule_id=RULE_ROMAJI_MISSING,
            severity="ERROR",
            cue_id=2,
            message="Cue 2 called 'Fran' in Japanese, but the Romaji name was omitted or changed.",
            suggestion="Keep Romaji proper name 'Fran'",
        ),
        LintViolation(
            rule_id=RULE_CJK_LEAK,
            severity="ERROR",
            cue_id=3,
            message="Contains CJK characters in Vietnamese translation",
        ),
    ]

    learned = kb.learn_from_linter_violations(violations, cues, series="tenken")
    assert learned == 3

    # Check caution retrieval for the banned calque
    check_cues = [Cue(id=10, start=0, end=1000, ja="世界観が壊れる")]
    cautions = kb.retrieve_cautions(check_cues, series="tenken")
    assert any("thế giới quan" in c["avoid"] for c in cautions)

    # Re-learning the same violations should be idempotent (count = 0)
    relearned = kb.learn_from_linter_violations(violations, cues, series="tenken")
    assert relearned == 0
