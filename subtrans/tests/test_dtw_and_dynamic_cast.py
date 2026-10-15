# -*- coding: utf-8 -*-
"""
subtrans/tests/test_dtw_and_dynamic_cast.py
Unit tests for Phase 3:
- Monotonic Dynamic Time Warping (DTW) subtitle alignment (1:1, 1:2, 2:1, OP/ED quarantine, Anchor bonus)
- HysteresisCastManager (Situational vs Permanent Arc Shift vs New Character)
"""

import pytest
from subtrans.dtw_auto_align import (
    SubCue,
    align_ja_to_en_monotonic_dtw,
    split_ja_text_proportionally,
    estimate_ja_mora,
    estimate_en_words,
)
from subtrans.dynamic_cast import (
    CastRelationDelta,
    HysteresisCastManager,
)


def test_dtw_alignment_1_to_1_snap():
    """Verifies that 1 JA cue snaps cleanly to 1 EN studio cue."""
    ja = [SubCue(1250, 2800, "こんにちは")]
    en = [SubCue(1000, 3000, "Hello there.")]

    aligned = align_ja_to_en_monotonic_dtw(ja, en)
    assert len(aligned) == 1
    cue, matched_en = aligned[0]
    assert cue.start_ms == 1000
    assert cue.end_ms == 3000
    assert cue.text == "こんにちは"
    assert matched_en == "Hello there."


def test_dtw_alignment_1_to_2_split():
    """Verifies that 1 long Whisper JA cue splits into 2 cues matching 2 EN cues."""
    ja = [SubCue(1000, 5000, "待ってください、行かないで！")]
    en = [
        SubCue(1000, 2500, "Wait for me,"),
        SubCue(2600, 5000, "don't go!"),
    ]

    aligned = align_ja_to_en_monotonic_dtw(ja, en)
    assert len(aligned) == 2
    c1, en1 = aligned[0]
    c2, en2 = aligned[1]
    assert c1.start_ms == 1000
    assert c1.end_ms == 2500
    assert c2.start_ms == 2600
    assert c2.end_ms == 5000
    assert "待って" in c1.text
    assert "行かないで" in c2.text


def test_dtw_alignment_2_to_1_partition():
    """Verifies that 2 short Whisper JA cues are partitioned across 1 EN cue."""
    ja = [
        SubCue(1050, 2400, "待って！"),
        SubCue(2500, 4900, "行かないで！"),
    ]
    en = [SubCue(1000, 5000, "Wait, don't leave me!")]

    aligned = align_ja_to_en_monotonic_dtw(ja, en)
    assert len(aligned) == 2
    c1, en1 = aligned[0]
    c2, en2 = aligned[1]
    assert c1.start_ms == 1000
    assert c1.end_ms < c2.start_ms
    assert c2.end_ms == 5000
    assert c1.text == "待って！"
    assert c2.text == "行かないで！"


def test_dtw_alignment_op_ed_quarantine():
    """Verifies that Whisper cues in OP/ED gaps keep their original timestamps and are not dragged."""
    ja = [
        SubCue(1000, 3000, "オープニング前のセリフ"),
        SubCue(30000, 35000, "♪ 歌の歌詞１ ♪"),
        SubCue(40000, 45000, "♪ 歌の歌詞２ ♪"),
        SubCue(120000, 123000, "本編開始のセリフ"),
    ]
    en = [
        SubCue(1000, 3000, "Pre-opening dialogue."),
        # Gap > 45s for OP theme
        SubCue(120000, 123000, "Episode starts dialogue."),
    ]

    aligned = align_ja_to_en_monotonic_dtw(ja, en)
    assert len(aligned) == 4
    # Câu đầu snap 1:1
    assert aligned[0][0].start_ms == 1000
    # Cụm bài hát OP giữ nguyên timestamp gốc Whisper
    assert aligned[1][0].start_ms == 30000
    assert aligned[2][0].start_ms == 40000
    # Câu cuối sau OP snap 1:1
    assert aligned[3][0].start_ms == 120000


def test_hysteresis_cast_manager_prevents_naive_auto_update_trap():
    """Verifies that SITUATIONAL_SHIFT does NOT overwrite baseline relations."""
    mgr = HysteresisCastManager(
        initial_cast_data={
            "relations": {
                "Stark -> Fern": {"1p": "tớ", "2p": "cậu", "reason": "bạn bè đồng hành"}
            }
        }
    )

    # 1. Tình huống nổi nóng nhất thời (SITUATIONAL_SHIFT)
    situational_delta = CastRelationDelta(
        scope="SITUATIONAL_SHIFT",
        speaker="Stark",
        listener="Fern",
        pronoun_1p="tao",
        pronoun_2p="mày",
        reason="cãi vã tức giận trong cảnh",
        confidence=0.95,
    )
    res1 = mgr.apply_deltas([situational_delta])
    assert res1["situational_shifts"] == 1
    # Quan hệ gốc vẫn được bảo toàn (tớ - cậu, không bị đổi thành tao - mày)
    pronouns = mgr.resolve_pronouns("Stark", "Fern")
    assert pronouns == ("tớ", "cậu")

    # 2. Phát hiện nhân vật mới (NEW_CHARACTER)
    new_char_delta = CastRelationDelta(
        scope="NEW_CHARACTER",
        speaker="Sein",
        listener="Frieren",
        pronoun_1p="tôi",
        pronoun_2p="tiểu thư",
        reason="mục sư mới gia nhập",
        confidence=0.85,
    )
    res2 = mgr.apply_deltas([new_char_delta])
    assert res2["new_characters"] == 1
    assert mgr.resolve_pronouns("Sein", "Frieren") == ("tôi", "tiểu thư")

    # 3. Bước ngoặt cốt truyện vĩnh viễn (PERMANENT_ARC_SHIFT)
    arc_delta = CastRelationDelta(
        scope="PERMANENT_ARC_SHIFT",
        speaker="Stark",
        listener="Fern",
        pronoun_1p="anh",
        pronoun_2p="em",
        reason="chính thức hẹn hò",
        confidence=0.92,
    )
    res3 = mgr.apply_deltas([arc_delta])
    assert res3["permanent_shifts"] == 1
    # Giờ đây xưng hô được nâng cấp thành anh - em qua arc_overrides
    assert mgr.resolve_pronouns("Stark", "Fern") == ("anh", "em")
