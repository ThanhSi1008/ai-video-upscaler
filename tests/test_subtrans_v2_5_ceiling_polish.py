# -*- coding: utf-8 -*-
"""
tests/test_subtrans_v2_5_ceiling_polish.py
Bộ kiểm thử đơn vị và tích hợp 100% offline kiểm chứng toàn diện Gói V2.5 (Chạm trần 100%):
1. Môi trường `.venv`: `faster-whisper`, `onnxruntime` (Silero VAD), `fugashi` + `unidic-lite`.
2. Tách từ hình thái học MeCab/UniDic (`ja_morph_tagger.py`): chống 100% báo giả từ ghép Kanji
   (`私立`, `我慢`, `下僕`, `君臨`, `余裕`, `某所`, `手前味噌`) & phân biệt đại từ `君` (`2P:kimi`) vs `-君` (`HON:kun`).
3. Vòng đời `KnowledgeBase` 1 lần/tập, thứ tự khóa tĩnh trước khóa động trong `filter_relevant_glossary`,
   `build_repair_system_prompt` (> 4.096 tokens tĩnh cho Cascade), và Explicit Context Caching (`cached_content` + fallback).
4. Bộ Glossary chuẩn cho `Frieren`, `Violet`, `Jujutsu`, `Kimetsu`/`DemonSlayer`, `ReZero`, tự nhận diện theo thư mục cha
   (`Frieren/S01E01.mkv`), và kho câu mẫu tích hợp `BUILTIN_GOLD_EXEMPLARS` xóa bỏ cold-start của Hybrid RAG.
5. Phân tích âm sắc giọng nói 5.1 FC thuần CPU (`subtrans/audio_diarization.py`): phát hiện `speaker_changed` vs `same_speaker`
   bất biến âm lượng và đính kèm `audio_turn` cho `.srt` trần khuyết tên người nói.
6. Bộ điều phối gối đầu `StageAPrefetchWorker` trong `scripts/batch_translate_all.py`.
"""

import json
import math
from pathlib import Path
import time
from unittest.mock import MagicMock

import numpy as np

from scripts.batch_translate_all import StageAPrefetchWorker
from subtrans.audio_diarization import (
    SAMPLE_RATE_HZ,
    annotate_cues_from_pcm,
    classify_turn_transition,
    compute_voice_signature,
)
from subtrans.cascade import build_repair_system_prompt
from subtrans.gemini_backend import GeminiTranslator, filter_relevant_glossary
from subtrans.glossary import (
    DEFAULT_BASE_GLOSSARY_PATH,
    GLOSSARIES_DIR,
    find_series_glossary,
    load_glossary,
    merge_glossaries,
)
from subtrans.ja_morph_tagger import (
    _COMPILED_FALLBACK_KANJI,
    extract_ja_linguistic_hints,
    is_morphological_backend_active,
    tokenize_ja_morphemes,
)
from subtrans.models import Cue
from subtrans.rag import BUILTIN_GOLD_EXEMPLARS, KnowledgeBase


def _synth_harmonic_voice(
    f0_hz: float,
    formants_hz: tuple[float, float, float],
    duration_sec: float = 0.85,
    amplitude: float = 0.35,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Tạo tín hiệu tiếng nói tổng hợp tuần hoàn có cấu trúc formant rõ rệt phục vụ test DSP tất định."""
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / float(sample_rate)
    sig = np.zeros_like(t)
    for k in range(1, int((sample_rate * 0.46) // f0_hz) + 1):
        fk = k * f0_hz
        # Phổ cộng hưởng 3 formant Gaussian
        env = (
            1.0 * math.exp(-0.5 * ((fk - formants_hz[0]) / 140.0) ** 2)
            + 0.85 * math.exp(-0.5 * ((fk - formants_hz[1]) / 220.0) ** 2)
            + 0.65 * math.exp(-0.5 * ((fk - formants_hz[2]) / 320.0) ** 2)
            + 0.12 / (1.0 + 0.15 * k)
        )
        sig += float(env) * np.sin(2.0 * math.pi * fk * t)
    peak = float(np.max(np.abs(sig)) + 1e-8)
    return (sig / peak) * float(amplitude)


# ==============================================================================
# 1 & 2. RUNTIME PACKAGES & MECAB MORPHOLOGICAL KANJI-COMPOUND IMMUNITY
# ==============================================================================


def test_v25_runtime_dependencies_faster_whisper_onnx_and_fugashi_installed():
    import faster_whisper
    import fugashi
    import onnxruntime

    assert faster_whisper.__version__
    assert onnxruntime.__version__
    assert fugashi is not None
    assert is_morphological_backend_active() is True
    toks = tokenize_ja_morphemes("私立高校に通っています")
    surfaces = [t[0] for t in toks]
    assert "私立" in surfaces
    assert "私" not in surfaces


def test_v25_morphological_tagger_eliminates_kanji_compound_false_positives():
    # Các từ ghép chứa chữ Hán trùng mặt chữ đại từ phải trả về None (không báo giả)
    compound_sentences = [
        "私立高校に通っています。",
        "もう我慢できない！",
        "下僕の分際で口を利くな。",
        "世界に君臨する王。",
        "今はまだ余裕がある。",
        "某所で落ち合おう。",
        "手前味噌ですが美味しいです。",
        "駅の少し手前で待っている。",
    ]
    for sent in compound_sentences:
        hints = extract_ja_linguistic_hints(sent)
        assert hints is None, f"Unexpected false-positive hint on '{sent}': {hints}"
        # Ngay cả nhánh fallback Regex khi không có fugashi cũng không được báo giả
        for pat, desc in _COMPILED_FALLBACK_KANJI:
            assert not pat.search(sent), f"Fallback regex '{desc}' falsely matched '{sent}'"

    # Phân biệt đại từ ngôi 2 `君` (2P:kimi) với hậu tố tên `-君` (HON:kun)
    kun_hint = extract_ja_linguistic_hints("佐藤君が来たよ。") or ""
    assert "HON:kun" in kun_hint
    assert "2P:kimi" not in kun_hint

    kimi_hint = extract_ja_linguistic_hints("君の名前は？") or ""
    assert "2P:kimi" in kimi_hint

    # Các đại từ thật vẫn bắt chính xác 100%
    assert "1P:watashi" in (extract_ja_linguistic_hints("私がやります。") or "")
    assert "1P:ware" in (extract_ja_linguistic_hints("我が名はめぐみん！") or "")
    assert "1P:boku" in (extract_ja_linguistic_hints("僕らの街を守る。") or "")
    assert "1P:yo" in (extract_ja_linguistic_hints("余は魔王である。") or "")
    assert "1P:soregashi" in (extract_ja_linguistic_hints("某は侍である。") or "")
    assert "2P:temee" in (extract_ja_linguistic_hints("手前ら、覚悟しろ！") or "")


# ==============================================================================
# 3. PREFIX CACHING, CASCADE >4096 STATIC PROMPT & EXPLICIT CACHE FALLBACK
# ==============================================================================


def test_v25_static_first_glossary_payload_and_cascade_static_prefix_cache():
    base_g = load_glossary(DEFAULT_BASE_GLOSSARY_PATH)
    frieren_g = load_glossary(GLOSSARIES_DIR / "glossary.frieren.yaml")
    merged = merge_glossaries(base_g, frieren_g)

    cues = [Cue(id=1, start=0, end=2000, ja="ゾルトラークは一般攻撃魔法だよ。", en="Zoltraak is ordinary offensive magic.")]
    rel = filter_relevant_glossary(cues, merged)
    keys = list(rel.keys())
    # Khóa tĩnh ('characters', 'address_guidance') bắt buộc đứng trước khóa động theo batch ('terms')
    assert keys.index("characters") < keys.index("terms")
    assert keys.index("address_guidance") < keys.index("terms")

    # Cascade repair system prompt phải vượt ngưỡng >16.000 ký tự (> 4.096 tokens) để hưởng Implicit Caching
    repair_sys = build_repair_system_prompt(merged)
    assert len(repair_sys) >= 16000
    assert "Zoltraak" in repair_sys
    assert "<vietnamese_anime_translation_handbook>" in repair_sys


def test_v25_explicit_context_caching_reuse_and_graceful_fallback(tmp_path):
    cache_db = tmp_path / "exp_cache.db"
    from subtrans.cache import TranslationCache

    t_cache = TranslationCache(db_path=cache_db)
    translator = GeminiTranslator(
        api_key="mock_key_exp",
        model="gemini-3.8-flash",
        explicit_caching=True,
        cache=t_cache,
    )
    mock_client = MagicMock()
    mock_cached_obj = MagicMock()
    mock_cached_obj.name = "cachedContents/anime-ep01-static-v25"
    mock_client.caches.create.return_value = mock_cached_obj

    mock_resp = MagicMock()
    mock_resp.text = json.dumps([{"id": 1, "vi": "Xin chào Frieren."}])
    mock_client.models.generate_content.return_value = mock_resp
    translator.client = mock_client

    cues_b1 = [Cue(id=1, start=0, end=1500, ja="こんにちは、フリーレン。", en="Hello, Frieren.")]
    out1 = translator.translate_batch(cues_b1, context=[])
    assert out1 == ["Xin chào Frieren."]
    assert mock_client.caches.create.call_count == 1
    gen_cfg_1 = mock_client.models.generate_content.call_args.kwargs["config"]
    assert getattr(gen_cfg_1, "cached_content", None) == "cachedContents/anime-ep01-static-v25"
    assert getattr(gen_cfg_1, "system_instruction", None) is None

    # Batch thứ 2 cùng tập tái sử dụng cùng cached_content, KHÔNG gọi lại caches.create
    cues_b2 = [Cue(id=1, start=2000, end=3500, ja="また明日ね。", en="See you tomorrow.")]
    translator.translate_batch(cues_b2, context=[])
    assert mock_client.caches.create.call_count == 1

    # Khi máy chủ từ chối caches.create (VD Free Tier), tự động rơi êm về system_instruction Implicit Caching
    translator_fb = GeminiTranslator(
        api_key="mock_key_free",
        model="gemini-3.5-flash-lite",
        explicit_caching=True,
        cache=t_cache,
    )
    mock_client_fb = MagicMock()
    mock_client_fb.caches.create.side_effect = RuntimeError("403 PermissionDenied: CachedContent not enabled on free tier")
    mock_client_fb.models.generate_content.return_value = mock_resp
    translator_fb.client = mock_client_fb

    cues_fb = [Cue(id=1, start=0, end=1500, ja="こんにちは。", en="Hello.")]
    out_fb = translator_fb.translate_batch(cues_fb, context=[])
    assert out_fb == ["Xin chào Frieren."]
    gen_cfg_fb = mock_client_fb.models.generate_content.call_args.kwargs["config"]
    assert getattr(gen_cfg_fb, "cached_content", None) is None
    assert len(getattr(gen_cfg_fb, "system_instruction", "") or "") >= 16000


# ==============================================================================
# 4. SERIES GLOSSARIES, PARENT-FOLDER RESOLUTION & RAG COLD-START EXEMPLARS
# ==============================================================================


def test_v25_priority_series_glossaries_and_parent_folder_auto_detection():
    # Kiểm tra nhận diện cả tên file trực tiếp lẫn thư mục cha (ví dụ: Raw_Sources/Frieren/S01E01.mkv)
    cases = [
        ("E:/Films/Raw_Sources/Frieren/S01E01.mkv", "frieren", "Zoltraak"),
        ("Sousou_no_Frieren_E10.mkv", "frieren", "Pháp sư cấp 1"),
        ("E:/Films/Raw_Sources/Violet Evergarden/01.mkv", "violet", "Búp bê Ký ức Tự động"),
        ("Jujutsu_Kaisen_S02E05.mkv", "jujutsu", "Lãnh địa triển khai"),
        ("Kimetsu_no_Yaiba_E19.mkv", "kimetsu", "Điệu múa Hỏa Thần"),
        ("E:/Films/Raw_Sources/Demon Slayer/S01E01.mkv", "kimetsu", "Sát Quỷ Đội"),
        ("ReZero_S02E01.mkv", "rezero", "Trở về từ Cái chết"),
    ]
    base_g = load_glossary(DEFAULT_BASE_GLOSSARY_PATH)
    for sample_path, expected_key, expected_vi_term in cases:
        found = find_series_glossary(sample_path)
        assert found is not None, f"Failed to auto-detect glossary for {sample_path}"
        loaded = load_glossary(found)
        assert loaded is not None
        assert expected_key in found.name.lower() or expected_key == "kimetsu"
        merged = merge_glossaries(base_g, loaded)
        vi_terms = {t.vi for t in merged.terms}
        assert expected_vi_term in vi_terms
        assert len(loaded.characters) >= 6


def test_v25_rag_cold_start_builtin_exemplars_and_seed_command(tmp_path):
    kb = KnowledgeBase(db_path=tmp_path / "cold_start_kb.db")
    # Mặc định DB trống có gold_units_count == 0 (tương thích ngược tuyệt đối với test_rag.py)
    assert kb.get_stats()["gold_units_count"] == 0

    # Khi pipeline bật fallback_builtin=True trên DB rỗng, vẫn truy hồi được ví dụ chuẩn của Frieren mà không ghi bẩn DB
    cues_frieren = [Cue(id=1, start=0, end=2000, ja="ゾルトラークは一般攻撃魔法だよ。")]
    shots_cold = kb.retrieve_few_shots(cues_frieren, series="frieren", fallback_builtin=True)
    assert len(shots_cold) >= 1
    assert any("Zoltraak" in s["vi"] for s in shots_cold)
    assert kb.get_stats()["gold_units_count"] == 0

    # Lệnh seed_default_gold_exemplars nạp vĩnh viễn kho câu mẫu vào DB một cách idempotent
    inserted = kb.seed_default_gold_exemplars()
    assert inserted == len(BUILTIN_GOLD_EXEMPLARS)
    assert kb.seed_default_gold_exemplars() == 0
    assert kb.get_stats()["gold_units_count"] == len(BUILTIN_GOLD_EXEMPLARS)


# ==============================================================================
# 5. 5.1 FC ACOUSTIC SPEAKER DIARIZATION (VOLUME-INVARIANT TIMBRE + F0)
# ==============================================================================


def test_v25_acoustic_speaker_turn_detection_and_prompt_injection():
    # Giọng nam trầm (F0 = 120 Hz, formant trầm) vs Giọng nữ cao (F0 = 245 Hz, formant sáng)
    male_loud = _synth_harmonic_voice(120.0, (420.0, 1150.0, 2350.0), duration_sec=0.9, amplitude=0.55)
    # Cùng người nói nam nhưng nói nhỏ hơn 4 lần -> CMS chuẩn hóa năng lượng phải nhận ra là 'same_speaker'!
    male_quiet = _synth_harmonic_voice(122.0, (425.0, 1160.0, 2360.0), duration_sec=0.9, amplitude=0.14)
    female_reply = _synth_harmonic_voice(245.0, (680.0, 1950.0, 3100.0), duration_sec=0.9, amplitude=0.40)

    sig_m1 = compute_voice_signature(male_loud)
    sig_m2 = compute_voice_signature(male_quiet)
    sig_f1 = compute_voice_signature(female_reply)

    assert sig_m1 is not None and sig_m2 is not None and sig_f1 is not None
    assert sig_m1.f0_hz is not None and 105.0 <= sig_m1.f0_hz <= 135.0
    assert sig_f1.f0_hz is not None and 220.0 <= sig_f1.f0_hz <= 270.0

    assert classify_turn_transition(sig_m1, sig_m2, gap_ms=400) == "same_speaker"
    assert classify_turn_transition(sig_m2, sig_f1, gap_ms=350) == "speaker_changed"
    # Khoảng lặng chuyển cảnh > 3000ms không đoán mò
    assert classify_turn_transition(sig_m2, sig_f1, gap_ms=4500) is None

    # Ghép thành timeline tập phim giả lập (0.0s-0.9s: Nam 1, 1.2s-2.1s: Nam 1 nói nhỏ, 2.4s-3.3s: Nữ đáp lời)
    sr = SAMPLE_RATE_HZ
    timeline = np.zeros(int(sr * 4.0), dtype=np.float32)
    timeline[0 : male_loud.size] = male_loud
    s2 = int(sr * 1.2)
    timeline[s2 : s2 + male_quiet.size] = male_quiet
    s3 = int(sr * 2.4)
    timeline[s3 : s3 + female_reply.size] = female_reply

    cues = [
        Cue(id=1, start=0, end=900, ja="行くぞ。", en="Let's go."),
        Cue(id=2, start=1200, end=2100, ja="準備はいいか？", en="Are you ready?"),
        Cue(id=3, start=2400, end=3300, ja="うん、いつでも。", en="Yes, anytime."),
    ]
    stats = annotate_cues_from_pcm(cues, timeline, sample_rate=sr)
    assert stats["analyzed"] == 3
    assert cues[1].audio_turn_hint == "same_speaker"
    assert cues[2].audio_turn_hint == "speaker_changed"

    # Kiểm chứng GeminiTranslator tự động đính kèm 'audio_turn' khi khuyết speaker_cc
    translator = GeminiTranslator(api_key="mock_k", model="gemini-3.5-flash-lite")
    translator.client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps([
        {"id": 1, "vi": "Đi thôi."},
        {"id": 2, "vi": "Em sẵn sàng chưa?"},
        {"id": 3, "vi": "Vâng, bất cứ lúc nào."},
    ])
    translator.client.models.generate_content.return_value = mock_resp
    translator.translate_batch(cues, context=[])
    sent_contents = translator.client.models.generate_content.call_args.kwargs["contents"]
    assert '"audio_turn":"same_speaker"' in sent_contents
    assert '"audio_turn":"speaker_changed"' in sent_contents


# ==============================================================================
# 6. INTER-EPISODE STAGE A PREFETCH WORKER & QUOTA CANCELLATION
# ==============================================================================


def test_v25_stage_a_prefetch_worker_overlap_and_quota_cancellation(tmp_path):
    calls = []

    def _fake_prefetch(fpath: str, output_dir: Path, subs_quality: str = "normal"):
        calls.append((fpath, subs_quality))
        return {"status": "PREFETCHED", "episode": Path(fpath).stem, "characters": 8}

    worker = StageAPrefetchWorker(prefetch_fn=_fake_prefetch)
    ep1 = str(tmp_path / "SeriesA" / "E01.mkv")
    ep2 = str(tmp_path / "SeriesB" / "E01.mkv")
    worker.submit_next(ep2, current_fpath=ep1, output_dir=tmp_path, subs_quality="high", same_series_headstart_sec=0.01)
    res = worker.wait_if_inflight(ep2, timeout=2.0)
    assert res is not None
    assert res["status"] == "PREFETCHED"
    assert res["characters"] == 8
    assert calls == [(ep2, "high")]

    # Kiểm chứng hủy sạch (cancel=True) ngay lập tức khi tập hiện tại báo QUOTA_EXHAUSTED
    ep3_same = str(tmp_path / "SeriesB" / "E02.mkv")
    worker.submit_next(ep3_same, current_fpath=ep2, output_dir=tmp_path, subs_quality="normal", same_series_headstart_sec=5.0)
    t0 = time.monotonic()
    worker.shutdown(cancel=True)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0
    # Tác vụ đang chờ headstart bị hủy ngay, không gọi _fake_prefetch cho ep3_same
    assert len(calls) == 1
