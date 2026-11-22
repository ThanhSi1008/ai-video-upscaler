# -*- coding: utf-8 -*-
"""
subtrans/tests/test_dtw_and_dynamic_cast.py
Unit tests for Phase 3:
- Monotonic Dynamic Time Warping (DTW) subtitle alignment (1:1, 1:2, 2:1, OP/ED quarantine, Anchor bonus)
- HysteresisCastManager (Situational vs Permanent Arc Shift vs New Character)
"""

import ctypes
import ctypes.wintypes
import gc
import sys
import time
import tracemalloc
import pytest
from subtrans.dtw_auto_align import (
    SubCue,
    _split_2_to_1_timing,
    align_ja_to_en_monotonic_dtw,
    get_dtw_alignment_plan,
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


def test_hysteresis_cast_manager_from_glossary_and_to_prompt_context():
    """Verifies that HysteresisCastManager initializes from glossary and generates structured prompt context."""
    from subtrans.models import Glossary, Character, CharacterSpeaksTo

    char = Character(
        name_ja="フェルン",
        name_vi="Fern",
        speaks_to=[
            CharacterSpeaksTo(listener="Stark", self="em", other="anh", note="tình cảm"),
        ]
    )
    glossary = Glossary(characters=[char])
    mgr = HysteresisCastManager.from_glossary(glossary)

    assert mgr.resolve_pronouns("Fern", "Stark") == ("em", "anh")

    # Kiểm tra prompt context
    prompt_ctx = mgr.to_prompt_context()
    assert "BASELINE RELATIONS" in prompt_ctx
    assert "Fern -> Stark: em/anh" in prompt_ctx

    # Bổ sung arc override và kiểm tra ưu tiên
    mgr.apply_deltas([
        CastRelationDelta(
            scope="PERMANENT_ARC_SHIFT",
            speaker="Fern",
            listener="Stark",
            pronoun_1p="vợ",
            pronoun_2p="chồng",
            reason="kết hôn",
            confidence=0.95
        )
    ])
    prompt_ctx_arc = mgr.to_prompt_context()
    assert "ARC OVERRIDES" in prompt_ctx_arc
    assert "Fern -> Stark: vợ/chồng" in prompt_ctx_arc


def test_dtw_2_to_1_short_en_cue_no_end_overflow():
    """Verify 2:1 DTW timing stays clamped inside en_cue boundaries on short (< 450ms) EN cues."""
    ja = [SubCue(1000, 1200, "あ"), SubCue(1210, 1400, "い")]
    en = [SubCue(1000, 1400, "Hey!")]
    aligned = align_ja_to_en_monotonic_dtw(ja, en)
    assert len(aligned) == 2
    assert aligned[0][0].start_ms >= 1000
    assert aligned[0][0].start_ms <= aligned[0][0].end_ms
    assert aligned[0][0].end_ms < aligned[1][0].start_ms
    assert aligned[1][0].start_ms <= aligned[1][0].end_ms <= 1400


def test_split_2_to_1_timing_clamps_extreme_durations():
    """Verify _split_2_to_1_timing never overflows or inverts across extreme durations and skewed ratios."""
    base_start = 5000
    for dur in (0, 1, 2, 3, 100, 399, 450, 599):
        en_cue = SubCue(base_start, base_start + dur, "Wait, please stop right now!")
        (s1, e1, _), (s2, e2, _) = _split_2_to_1_timing(
            SubCue(base_start, base_start + 50, "あ"),
            SubCue(base_start + 60, base_start + 120, "い"),
            en_cue,
        )
        assert base_start <= s1 <= e1 <= s2 <= e2 <= base_start + dur
        if dur >= 3:
            assert e1 < s2
            assert (e1 - s1) >= 1 and (e2 - s2) >= 1
        if dur >= 450:
            assert (e1 - s1) >= 200 and (e2 - s2) >= 200
            assert s2 - e1 == 50

    # Skewed JA durations on a 1000ms EN cue must still reserve >= 200ms for part 2.
    en_skew = SubCue(1000, 2000, "First clause, second clause.")
    (s1, e1, _), (s2, e2, _) = _split_2_to_1_timing(
        SubCue(1000, 1990, "とても長い最初の台詞"),
        SubCue(1995, 2005, "うん"),
        en_skew,
    )
    assert 1000 <= s1 <= e1 < s2 <= e2 == 2000
    assert (e1 - s1) >= 200 and (e2 - s2) >= 200


def test_dtw_windowed_edge_cases_and_cast_preindex():
    """Verify windowed DTW handles empty/skewed shapes and pre-indexed furigana cast anchors."""
    assert get_dtw_alignment_plan([], [SubCue(0, 1000, "Hi")]) == ({}, {})
    assert get_dtw_alignment_plan([SubCue(0, 1000, "やあ")], []) == ({}, {})

    # Extreme aspect ratios remain reachable within the widened Sakoe-Chiba band.
    ja_one = [SubCue(12500, 14000, "本当か？")]
    en_many = [SubCue(i * 2500, i * 2500 + 800, f"Line {i}.") for i in range(100)]
    en_many[5] = SubCue(12500, 14000, "Really?")
    map_1_n, _ = get_dtw_alignment_plan(ja_one, en_many)
    assert map_1_n[0] == (12500, 14000, "Really?")

    ja_many = [SubCue(i * 2500, i * 2500 + 800, f"台詞{i}") for i in range(100)]
    ja_many[5] = SubCue(12500, 14000, "本当か？")
    en_one = [SubCue(12500, 14000, "Really?")]
    map_n_1, _ = get_dtw_alignment_plan(ja_many, en_one)
    assert map_n_1[5] == (12500, 14000, "Really?")

    # Furigana cast matching across 1:1, 1:2, and 2:1 cues.
    cast = [("絹秀(けんしゅう)", "Kenshuu")]
    ja_cast = [
        SubCue(1000, 2000, "絹秀様！"),
        SubCue(3000, 6000, "絹秀様、どうかなさいましたか？"),
        SubCue(7000, 8200, "絹秀様"),
        SubCue(8300, 9500, "お答えを！"),
    ]
    en_cast = [
        SubCue(1000, 2000, "Kenshuu!"),
        SubCue(3000, 4400, "Kenshuu,"),
        SubCue(4600, 6000, "what happened?"),
        SubCue(7000, 9500, "Kenshuu, please answer me!"),
    ]
    cmap, s12 = get_dtw_alignment_plan(ja_cast, en_cast, cast_names_ja_en=cast)
    assert cmap[0] == (1000, 2000, "Kenshuu!")
    assert 1 in s12 or cmap[1] is not None
    assert cmap[2] is not None and cmap[3] is not None


def _get_process_memory_bytes() -> tuple[int, int]:
    """Return current OS process (working_set_bytes, peak_working_set_bytes)."""
    if sys.platform == "win32":
        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
            ctypes.wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(pmc.WorkingSetSize), int(pmc.PeakWorkingSetSize)
    import resource
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_kb * 1024, rss_kb * 1024


def test_dtw_alignment_plan_n1200_peak_ram_under_35mb():
    """Verify DTW alignment stays under 5 MB traced RAM at N=250 and < 35 MB OS RAM delta at N=1200."""
    # (a) Representative tracemalloc check at N=250 (keeps test runtime < 2s vs ~60s at N=1200).
    n_small = 250
    ja_small = [SubCue(i * 2000, i * 2000 + 1500, f"フラン、セリフ{i}？") for i in range(n_small)]
    en_small = [SubCue(i * 2000 + 50, i * 2000 + 1550, f"Fran, line {i}?") for i in range(n_small)]
    tracemalloc.start(1)
    try:
        cue_map_small, _ = get_dtw_alignment_plan(ja_small, en_small, cast_names_ja_en=[("フラン", "Fran")])
        _cur_small, peak_small = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(cue_map_small) == n_small
    assert peak_small < 5 * 1024 * 1024, f"DTW N=250 traced peak RAM too high: {peak_small / (1024 * 1024):.2f} MB"

    # (b) Uninstrumented N=1200, M=1200 and N=1200, M=1350 OS WorkingSet/PeakWorkingSet delta checks.
    N = 1200
    ja = [SubCue(i * 2000, i * 2000 + 1500, f"フラン、セリフ{i}？") for i in range(N)]
    en = [SubCue(i * 2000 + 50, i * 2000 + 1550, f"Fran, line {i}?") for i in range(N)]
    en_skew = en + [SubCue((N + k) * 2000, (N + k) * 2000 + 1200, f"Extra {k}") for k in range(150)]

    gc.collect()
    ws0, pk0 = _get_process_memory_bytes()
    t0 = time.perf_counter()
    cue_map, _ = get_dtw_alignment_plan(ja, en, cast_names_ja_en=[("フラン", "Fran")])
    elapsed_eq = time.perf_counter() - t0
    ws1, pk1 = _get_process_memory_bytes()

    assert len(cue_map) == N
    assert elapsed_eq < 5.0, f"DTW N=1200, M=1200 took {elapsed_eq:.2f}s (>= 5.0s)"
    assert (ws1 - ws0) < 35 * 1024 * 1024, f"DTW N=1200 WorkingSet delta too high: {(ws1 - ws0) / (1024 * 1024):.2f} MB"
    assert (pk1 - pk0) < 35 * 1024 * 1024, f"DTW N=1200 PeakWorkingSet delta too high: {(pk1 - pk0) / (1024 * 1024):.2f} MB"

    t1 = time.perf_counter()
    cue_map_skew, _ = get_dtw_alignment_plan(ja, en_skew, cast_names_ja_en=[("フラン", "Fran")])
    elapsed_skew = time.perf_counter() - t1
    ws2, pk2 = _get_process_memory_bytes()

    assert len(cue_map_skew) == N
    assert elapsed_skew < 5.0, f"DTW N=1200, M=1350 took {elapsed_skew:.2f}s (>= 5.0s)"
    assert (ws2 - ws1) < 35 * 1024 * 1024, f"DTW N=1200 skewed WorkingSet delta too high: {(ws2 - ws1) / (1024 * 1024):.2f} MB"
    assert (pk2 - pk1) < 35 * 1024 * 1024, f"DTW N=1200 skewed PeakWorkingSet delta too high: {(pk2 - pk1) / (1024 * 1024):.2f} MB"

