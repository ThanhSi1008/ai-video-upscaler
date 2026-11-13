# -*- coding: utf-8 -*-
"""
tests/test_subtrans_v2_max_quality.py
Bộ kiểm thử đơn vị và tích hợp 100% offline kiểm chứng toàn diện 6 gói nâng cấp
chất lượng tối đa phân hệ dịch phụ đề AI (P0-A, P0-B, P1-A, P1-B, P2, P3).
Không gọi Gemini API thật, không đọc .env, không chạm SQLite cache vận hành hay GPU.
"""

from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pysubs2

from src.cleanup import cleanup_subtitle_extracts
from src.scheduler import check_gemini_quota_for_queue
from subtrans.cache import (
    TranslationCache,
    compute_contextual_cue_sig,
    compute_cue_cache_key,
    compute_key_id,
)
from subtrans.cascade import accept_or_reject_repaired_cue
from subtrans.dtw_auto_align import SubCue, get_dtw_alignment_plan
from subtrans.dynamic_cast import HysteresisCastManager, _extract_vi_pronoun_pair
from subtrans.eval.runner import run_evaluation
from subtrans.gemini_backend import (
    GeminiTranslator,
    SubtitleOutputItem,
    _apply_item_metadata_to_cue,
    build_episode_system_prompt,
    get_thinking_config,
)
from subtrans.glossary import merge_glossaries
from subtrans.ja_morph_tagger import extract_ja_linguistic_hints
from subtrans.linter import (
    RULE_PRONOUN_CONSISTENCY,
    RULE_ROMAJI_MISSING,
    lint_cue,
)
from subtrans.models import (
    AddressRules,
    Character,
    CharacterSpeaksTo,
    Cue,
    Glossary,
    PronounRule,
    RelationRule,
)
from subtrans.parser import (
    detect_auto_offset,
    parse_and_align_subtitles,
)
from subtrans.rag import KnowledgeBase
from subtrans.report import generate_qa_report
from subtrans.throttle import DailyQuotaExhaustedError, call_with_retry
from subtrans.validation import validate_translations
from subtrans.writer import (
    clean_and_normalize_vietnamese,
    optimize_cues_for_playback,
    split_sentence_across_cues,
    write_subtitles,
)


# ==============================================================================
# GÓI P0-A: CONTEXT-AWARE SQLITE CACHE & EXPLICIT COT PRONOUN HYSTERESIS
# ==============================================================================


def test_p0a_contextual_cache_key_differentiates_short_ambiguous_dialogue(tmp_path):
    sig_fran = compute_contextual_cue_sig(
        ja="ありがとう",
        speaker="Fran",
        mode_hint="spoken",
        prev_ja="師匠、これ食べて",
        next_ja="おいしい",
    )
    sig_reirin = compute_contextual_cue_sig(
        ja="ありがとう",
        speaker="Reirin",
        mode_hint="inner",
        prev_ja="陛下のお言葉",
        next_ja="感謝いたします",
    )
    assert sig_fran != ""
    assert sig_reirin != ""
    assert sig_fran != sig_reirin

    key_1 = compute_cue_cache_key(
        model="gemini-3.5-flash-lite",
        ja="ありがとう",
        en="Thank you.",
        prompt_version="v2.2",
        context_sig=sig_fran,
    )
    key_2 = compute_cue_cache_key(
        model="gemini-3.5-flash-lite",
        ja="ありがとう",
        en="Thank you.",
        prompt_version="v2.2",
        context_sig=sig_reirin,
    )
    assert key_1 != key_2

    # Câu dài đã có chủ ngữ tường minh (> 28 ký tự và không có speaker/mode đặc biệt) dùng sig rỗng để tối đa cache hit
    long_ja = "我こそは偉大なる死霊術師ジャン・ドゥービーであり、この迷宮の支配者なのだ！"
    assert compute_contextual_cue_sig(ja=long_ja) == ""

    cache = TranslationCache(db_path=tmp_path / "p0a_cache.db")
    cache.set_cue(
        key_1,
        backend="gemini",
        model="gemini-3.5-flash-lite",
        vi="Cảm ơn sư phụ.",
        metadata={"speaker": "Fran", "addressee": "Shishou", "p1": "em", "p2": "sư phụ"},
    )
    hit = cache.get_cue_entry(key_1)
    assert hit is not None
    vi_val, meta = hit
    assert vi_val == "Cảm ơn sư phụ."
    assert meta["p1"] == "em"
    assert meta["p2"] == "sư phụ"
    assert cache.get_cue(key_2) is None


def test_p0a_explicit_cot_p1_p2_and_compound_vietnamese_immunity():
    # Kiểm tra thứ tự trường CoT: p1, p2 đứng trước vi
    field_names = list(SubtitleOutputItem.model_fields.keys())
    assert field_names.index("p1") < field_names.index("vi")
    assert field_names.index("p2") < field_names.index("vi")

    cue = Cue(id=1, start=0, end=2000, ja="行くぞ")
    _apply_item_metadata_to_cue(
        cue,
        {
            "id": 1,
            "speaker": "Shishou",
            "addressee": "Fran",
            "mode": "spoken",
            "p1": "ta",
            "p2": "con",
            "vi": "Đi thôi nào!",
        },
    )
    assert cue.p1 == "ta"
    assert cue.p2 == "con"

    # Từ ghép tiếng Việt không được sinh đại từ giả trong regex fallback
    assert _extract_vi_pronoun_pair("Con rồng khổng lồ đang bay một mình trên bầu trời.") is None
    assert _extract_vi_pronoun_pair("Cô gái ấy bước đi trên con đường vắng.") is None

    # HysteresisCastManager học trực tiếp từ cặp CoT (p1, p2) ngay cả khi câu VI lược đại từ
    mgr = HysteresisCastManager()
    batch = [
        Cue(id=1, start=0, end=1500, ja="...", vi="Đi thôi!", speaker="Nell", addressee="Fran", p1="chị", p2="em"),
        Cue(id=2, start=1600, end=3000, ja="...", vi="Cẩn thận đấy!", speaker="Nell", addressee="Fran", p1="chị", p2="em"),
    ]
    res = mgr.observe_batch_cues(batch, min_consistent_hits=2)
    assert res["new_characters"] == 1
    assert mgr.resolve_pronouns("Nell", "Fran") == ("chị", "em")


# ==============================================================================
# GÓI P0-B: IMPLICIT CACHING, THINKING CONFIG, 3-KEY MODULO & PER-KEY RPD
# ==============================================================================


def test_p0b_implicit_cache_static_prefix_and_cached_token_telemetry(tmp_path):
    glossary = Glossary(
        style_rules=["Dịch tự nhiên, chuẩn nhịp thoại anime."],
        address_rules=AddressRules(
            principle="Giữ đúng tôn ti hậu cung.",
            first_person=[PronounRule(ja="わたくし", vi="thần thiếp")],
        ),
        characters=[Character(name_ja="玲琳", name_vi="Reirin")],
    )
    sys_prompt = build_episode_system_prompt(glossary=glossary)
    # Bảo đảm vượt ngưỡng tối thiểu > 4,096 tokens (~14,000+ ký tự tĩnh) cho Gemini Implicit Context Caching
    assert len(sys_prompt) >= 12000

    cache = TranslationCache(db_path=tmp_path / "telemetry.db")
    cache.log_request(
        model="gemini-3.5-flash-lite",
        backend="gemini",
        cues_count=100,
        prompt_tokens=6000,
        candidates_tokens=1800,
        total_tokens=7900,
        thinking_tokens=100,
        cached_tokens=4500,
    )
    tel = cache.get_token_telemetry_today_pt()
    assert tel["gemini-3.5-flash-lite"]["cached_tokens"] == 4500
    assert tel["gemini-3.5-flash-lite"]["prompt_tokens"] == 6000

    srt_out = tmp_path / "ep01.vi.srt"
    srt_out.write_text("1\n00:00:00,000 --> 00:00:02,000\nXin chào.\n", encoding="utf-8")
    rep = generate_qa_report(
        cues=[Cue(id=1, start=0, end=2000, ja="こんにちは", vi="Xin chào.")],
        ja_path=tmp_path / "ep01.ja.srt",
        en_path=None,
        output_subtitle_path=srt_out,
        report_path=tmp_path / "ep01.audit.md",
        elapsed_time_sec=1.0,
        stats={"dry_run": True},
        cache=cache,
    )
    rep_md = rep.read_text(encoding="utf-8")
    assert "Implicit Context Caching" in rep_md
    assert "75.0%" in rep_md


def test_p0b_thinking_config_cjk_validation_and_3key_modulo_wrap_around(tmp_path):
    cfg_lite = get_thinking_config("gemini-3.5-flash-lite")
    cfg_flash = get_thinking_config("gemini-3.8-flash")
    assert cfg_lite is not None and "MINIMAL" in str(cfg_lite.thinking_level).upper()
    assert cfg_flash is not None and "LOW" in str(cfg_flash.thinking_level).upper()

    # CJK Extension-A (\u3400) & dấu ngoặc vuông Nhật 「」 phải bị chặn bởi validation
    ok_ext_a, err_ext_a = validate_translations([1], [{"id": 1, "vi": "Thanh kiếm 㐀 huyền thoại"}])
    ok_brackets, err_brackets = validate_translations([1], [{"id": 1, "vi": "「Xin chào」"}])
    assert not ok_ext_a and "Leftover Japanese" in err_ext_a
    assert not ok_brackets

    # Kiểm tra xoay vòng modulo từ Key #3 (index=2) vòng về Key #1 (index=0) -> Key #2 (index=1)
    cache = TranslationCache(db_path=tmp_path / "keys.db")
    GeminiTranslator.reset_session_rotation_events()
    translator = GeminiTranslator(
        api_keys=["fake-key-alpha", "fake-key-beta", "fake-key-gamma"],
        model="gemini-3.5-flash-lite",
        dry_run=True,
        round_robin=True,
        cache=cache,
    )
    # Giả lập round-robin đang đứng ở Key #3 (index=2)
    translator.current_key_idx = 2
    translator._visited_rotation_indices = {2}
    assert translator.rotate_key("429 PerDay on Key #3") is True
    assert translator.current_key_idx == 0
    assert translator.rotate_key("429 PerDay on Key #1") is True
    assert translator.current_key_idx == 1
    assert translator.rotate_key("429 PerDay on Key #2") is False

    # Kiểm tra cô lập cạn RPD theo từng key trong call_with_retry
    k1, k2 = compute_key_id("k1"), compute_key_id("k2")

    class FakePerDay429(Exception):
        pass

    def fail_per_day():
        e = FakePerDay429("429 RESOURCE_EXHAUSTED")
        e.details = [{"metadata": {"quotaId": "GenerateRequestsPerDayPerProjectPerModel", "quotaValue": "500"}}]
        raise e

    try:
        call_with_retry(fail_per_day, model_name="gemini-3.5-flash-lite", cache=cache, key_id=k1, all_key_ids=[k1, k2])
    except DailyQuotaExhaustedError:
        pass

    assert cache.is_key_model_exhausted(k1, "gemini-3.5-flash-lite") is True
    assert cache.is_key_model_exhausted(k2, "gemini-3.5-flash-lite") is False
    # Chưa được khóa toàn model khi k2 vẫn còn sống!
    assert cache.is_model_exhausted("gemini-3.5-flash-lite") is False


# ==============================================================================
# GÓI P1-A: PARSER CC SOUND PREFIX, MODE HINT, 1:2 DTW SPLIT & PIECEWISE OFFSET
# ==============================================================================


def test_p1a_parser_cc_sound_prefix_mode_hint_and_ass_actor(tmp_path):
    ass = pysubs2.SSAFile()
    # Event 0: comment -> phải bị loại bỏ hoàn toàn
    evt_comment = pysubs2.SSAEvent(start=100, end=900, text="Ghi chú biên tập")
    evt_comment.is_comment = True
    ass.append(evt_comment)
    # Event 1: Typesetting \clip -> phải được đánh dấu is_skipped=True nhưng giữ raw_text
    ass.append(pysubs2.SSAEvent(start=1000, end=2500, text=r"{\clip(0,0,100,100)}제1話 始まり"))
    # Event 2: DMMTV CC sound prefix + ngoặc độc thoại ＜...＞ + cột Actor Name
    evt_cc = pysubs2.SSAEvent(start=3000, end=5500, text="（筆記音）＜この薬草なら解毒できる…＞")
    evt_cc.name = "玲琳"
    ass.append(evt_cc)

    ja_file = tmp_path / "ep01.ja.ass"
    ass.save(str(ja_file))

    cues, _, _ = parse_and_align_subtitles(str(ja_file), sound_cues="drop")
    assert len(cues) == 2
    assert cues[0].is_skipped is True
    assert cues[1].is_skipped is False
    assert cues[1].speaker == "玲琳"  # Không bị ghi đè bởi （筆記音）!
    assert cues[1].mode_hint == "inner"
    assert "＜" not in cues[1].ja and "＞" not in cues[1].ja


def test_p1a_dtw_1_to_2_pause_split_and_piecewise_auto_offset(tmp_path):
    # 1 cue Whisper gộp nhưng bên EN có 2 câu cách nhau 400ms (>= 250ms) -> tách thành 2 sub-cues
    ja_cues = [SubCue(1000, 5200, "待ってくれ、まだ話が終わっていない！")]
    en_cues = [
        SubCue(1000, 2400, "Wait a second!"),
        SubCue(2850, 5200, "We are not done talking yet!"),
    ]
    alignment_map, split_plan = get_dtw_alignment_plan(ja_cues, en_cues, min_pause_for_1_to_2_split_ms=250)
    assert 0 in split_plan
    assert len(split_plan[0]) == 2
    assert split_plan[0][0][1] <= split_plan[0][1][0]

    # Piecewise 2-segment auto-offset: Cold Open 0ms, sau OP (sponsor card 10s) lệch +10000ms với khoảng cách không tuần hoàn
    ja_ssa = pysubs2.SSAFile()
    en_ssa = pysubs2.SSAFile()
    cur_t = 5000
    for i in range(20):
        dur = 1800 + (i * 173) % 900
        en_ssa.append(pysubs2.SSAEvent(start=cur_t, end=cur_t + dur, text=f"Cold open line {i}"))
        ja_ssa.append(pysubs2.SSAEvent(start=cur_t, end=cur_t + dur, text=f"アバンのセリフ{i}"))
        cur_t += dur + 1400 + (i * 617) % 2300
    # Sau OP + sponsor card (bước nhảy +10000ms)
    cur_t += 90000
    for i in range(25):
        dur = 1900 + (i * 211) % 850
        en_ssa.append(pysubs2.SSAEvent(start=cur_t, end=cur_t + dur, text=f"Post-OP line {i}"))
        ja_ssa.append(pysubs2.SSAEvent(start=cur_t - 10000, end=cur_t - 10000 + dur, text=f"本編のセリフ{i}"))
        cur_t += dur + 1500 + (i * 733) % 2500

    ja_p = tmp_path / "piecewise.ja.srt"
    en_p = tmp_path / "piecewise.en.srt"
    ja_ssa.save(str(ja_p))
    en_ssa.save(str(en_p))

    best_off, should_apply, _reason, diag = detect_auto_offset(str(ja_p), str(en_p))
    assert should_apply is True
    pw = diag.get("piecewise_segments")
    assert pw and len(pw) == 2
    assert abs(pw[0]["offset_ms"] - 0) <= 250
    assert abs(pw[1]["offset_ms"] - 10000) <= 250


# ==============================================================================
# GÓI P1-B: GLOSSARY DEEP-MERGE, LINTER BOUNDARY, CASCADE DIGIT & WRITER SAFETY
# ==============================================================================


def test_p1b_glossary_deep_merge_linter_katakana_boundary_and_cascade_digits():
    base_g = Glossary(
        address_rules=AddressRules(
            principle="Base rule",
            first_person=[PronounRule(ja="俺", vi="tôi"), PronounRule(ja="私", vi="tôi")],
            honorifics=[PronounRule(ja="様", vi="ngài")],
        ),
        characters=[Character(name_ja="ジャン・ドゥービー", name_vi="Jean Doubee")],
    )
    series_g = Glossary(
        address_rules=AddressRules(
            first_person=[PronounRule(ja="俺", vi="ta")],
            relations=[RelationRule(case="Sư đồ", self="em", other="sư phụ")],
        ),
        characters=[
            Character(
                name_ja="ラム",
                name_vi="Ram",
                speaks_to=[CharacterSpeaksTo(listener="Subaru", self="Ram", other="Barusu")],
            ),
            Character(name_ja="ジャン・ドゥービー", name_vi="Jean Doubee", aliases=["Jean"]),
        ],
    )
    merged = merge_glossaries(base_g, series_g)
    assert merged is not None and merged.address_rules is not None
    fp_map = {r.ja: r.vi for r in merged.address_rules.first_person}
    assert fp_map["俺"] == "ta" and fp_map["私"] == "tôi"
    assert len(merged.address_rules.honorifics) == 1
    assert len(merged.address_rules.relations) == 1

    # Katakana loanword "プログラム" không được kích hoạt giả RULE_ROMAJI_MISSING cho "ラム" (Ram)
    loanword_cue = Cue(id=1, start=0, end=2500, ja="このプログラムを起動する。", vi="Khởi động chương trình này.")
    v_loan = lint_cue(loanword_cue, glossary=merged)
    assert not any(v.rule_id == RULE_ROMAJI_MISSING for v in v_loan)

    # Tên đầy đủ "Jean Doubee" chấp nhận câu tiếng Việt gọi "Jean"
    jean_cue = Cue(id=2, start=0, end=2500, ja="ジャン・ドゥービー、待て！", vi="Jean, đợi đã!")
    v_jean = lint_cue(jean_cue, glossary=merged)
    assert not any(v.rule_id == RULE_ROMAJI_MISSING for v in v_jean)

    # RULE_PRONOUN_CONSISTENCY phát hiện lệch đại từ CoT p1/p2 so với speaks_to
    mismatch_cue = Cue(
        id=3,
        start=0,
        end=2500,
        ja="バルス、起きなさい。",
        vi="Anh dậy đi.",
        speaker="Ram",
        addressee="Subaru",
        p1="em",
        p2="anh",
        mode="spoken",
    )
    v_pron = lint_cue(mismatch_cue, glossary=merged)
    assert any(v.rule_id == RULE_PRONOUN_CONSISTENCY for v in v_pron)

    # Morph tagger bắt đa mệnh đề & Kansai-ben & 某 (soregashi)
    hints = extract_ja_linguistic_hints("某に任せろよ、なんでやねん！") or ""
    assert "1P:soregashi" in hints
    assert "DIALECT:kansai_ben" in hints

    # Cascade đánh dấu uncertain=True khi Flash làm lệch con số trong câu gốc
    num_cue = Cue(id=4, start=0, end=3000, ja="敵は300人いる！", vi="Kẻ địch có 300 tên!")
    accepted, reason, _ = accept_or_reject_repaired_cue(num_cue, "Kẻ địch có 500 tên!")
    assert accepted is True
    assert num_cue.is_digit_mismatch is True or reason != ""


def test_p1b_writer_preserves_angle_brackets_inner_quotes_dramatic_repeats_and_ass_signs(tmp_path):
    # 1. Không xóa <Rudeus!> (tên riêng trong ngoặc nhọn không phải thẻ HTML)
    assert "<Rudeus!>" in clean_and_normalize_vietnamese("<Rudeus!> Cẩn thận!")
    # 2. Giữ nguyên cặp nháy đơn giữa dòng 'Sư phụ' và 'Đệ tử'
    kept_quotes = clean_and_normalize_vietnamese("'Sư phụ' và 'Đệ tử' cùng chiến đấu.")
    assert kept_quotes.startswith("'Sư phụ'")
    # Nhưng bóc ngoặc bao ngoài toàn câu
    assert clean_and_normalize_vietnamese("＜Chuyện gì đang xảy ra vậy?＞") == "Chuyện gì đang xảy ra vậy?"

    # 3. Không nuốt lời gọi cảm xúc lặp lại cách nhau >= 80ms ("Subaru..." -> "Subaru!")
    dramatic = [
        Cue(id=1, start=1000, end=2200, ja="スバル…", vi="Subaru..."),
        Cue(id=2, start=2400, end=3800, ja="スバル！", vi="Subaru!"),
    ]
    opt = optimize_cues_for_playback(dramatic)
    assert len(opt) == 2

    # 4. Không viết thường tên riêng ở nửa sau split_sentence_across_cues
    parts = split_sentence_across_cues("Cảm ơn anh rất nhiều, Subaru đã cứu em khỏi nguy hiểm!", num_cues=2)
    assert len(parts) == 2
    assert "Subaru" in parts[1]

    # 5. Xuất .ass bảo toàn event Typesetting/Sign đã skip ở đúng vị trí thời gian
    orig_ass = pysubs2.SSAFile()
    orig_ass.append(pysubs2.SSAEvent(start=500, end=2000, style="Sign", text=r"{\pos(960,120)}王都アルカンレティア"))
    orig_ass.append(pysubs2.SSAEvent(start=2500, end=4500, style="Default", text="こんにちは！"))

    cues_for_ass = [
        Cue(id=1, start=500, end=2000, style="Sign", ja="王都アルカンレティア", is_skipped=True, raw_text=r"{\pos(960,120)}王都アルカンレティア"),
        Cue(id=2, start=2500, end=4500, style="Default", ja="こんにちは！", vi="Xin chào!"),
    ]
    out_ass_path = write_subtitles(cues_for_ass, orig_ass, tmp_path / "out.vi.ass", output_format="ass")
    saved_ass = pysubs2.load(str(out_ass_path))
    texts = [e.text for e in saved_ass]
    assert any("王都アルカンレティア" in t for t in texts)
    assert any("Xin chào!" in t for t in texts)


# ==============================================================================
# GÓI P2 & P3: HYBRID BM25 RAG, SILVER CURATION, EVAL, ISOLATION & SCHEDULER
# ==============================================================================


def test_p2_hybrid_bm25_rag_silver_promotion_and_live_eval(tmp_path):
    kb = KnowledgeBase(db_path=tmp_path / "kb.db")
    kb.add_gold_unit(
        ja_text="師匠、魔獣の核を吸収する？",
        vi_text="Sư phụ, mình hấp thụ hạch ma thú nhé?",
        series="tenken",
        episode="ep01",
        speaker="Fran",
        addressee="Shishou",
        status="gold",
    )
    kb.add_gold_unit(
        ja_text="魔獣の森は危険だ。",
        vi_text="Khu rừng ma thú rất nguy hiểm.",
        series="tenken",
        episode="ep01",
        speaker="Garrus",
        addressee="Fran",
        status="reviewed",
    )

    query_cues = [
        Cue(id=1, start=0, end=2000, ja="師匠、あの魔獣を見て！", speaker="Fran", addressee="Shishou"),
    ]
    shots = kb.retrieve_few_shots(query_cues, series="tenken", max_examples=2, exclude_episode="ep02")
    assert len(shots) >= 1
    assert shots[0]["speaker"] == "Fran"

    # Tự động thăng hạng câu sửa sạch từ Cascade lên silver
    repaired_cue = Cue(
        id=10,
        start=0,
        end=2500,
        ja="黒猫族の誇りを取り戻す！",
        vi="Ta sẽ giành lại niềm kiêu hãnh của tộc Hắc Miêu!",
        speaker="Fran",
        addressee="Shishou",
        model_used="gemini-3.8-flash (repair)",
        repair_reason_code="accepted_lint_clean",
        uncertain=False,
    )
    promoted = kb.promote_cascade_repairs([repaired_cue], series="tenken", episode="ep02")
    assert promoted == 1

    # Chống tự nhiễm độc error_bank khi câu nháp đang gắn cờ uncertain=True
    g = Glossary(terms=[])
    from subtrans.linter import LintViolation, RULE_GLOSSARY_TERM

    uncertain_cue = Cue(id=11, start=0, end=2000, ja="魔剣を抜く", vi="Rút kiếm", uncertain=True)
    fake_v = LintViolation(
        rule_id=RULE_GLOSSARY_TERM,
        severity="WARNING",
        cue_id=11,
        message="Cue 11 contains glossary term '魔剣' but expected translation 'ma kiếm' was not found.",
        suggestion="Use 'ma kiếm'.",
    )
    assert kb.learn_from_linter_violations([fake_v], [uncertain_cue], series="tenken") == 0

    eval_summary = run_evaluation()
    assert eval_summary["status"] == "ready"
    assert eval_summary["pronoun_verified"] == eval_summary["pronoun_benchmark_count"]
    assert eval_summary["pronoun_benchmark_count"] >= 22


def test_p3_pending_queue_no_torch_cleanup_whisper_dotenv_guard_and_scheduler_quota(tmp_path, monkeypatch):
    # 1. Nạp src.pending_queue trong tiến trình con sạch KHÔNG kéo theo torch
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, src.pending_queue; assert 'torch' not in sys.modules",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    # 2. cleanup_subtitle_extracts xóa sạch cả *.whisper_ja.srt và thư mục .temp rỗng
    temp_sub_dir = tmp_path / ".temp_AnimeEp01"
    temp_sub_dir.mkdir()
    whisper_srt = temp_sub_dir / "AnimeEp01.whisper_ja.srt"
    whisper_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nテスト\n", encoding="utf-8")
    freed = cleanup_subtitle_extracts(str(tmp_path), "AnimeEp01")
    assert freed > 0
    assert not whisper_srt.exists()
    assert not temp_sub_dir.exists()

    # 3. PYTHON_DOTENV_DISABLED=1 chặn tuyệt đối việc đọc .env khi import subtrans.config
    fake_work = tmp_path / "dotenv_trap"
    fake_work.mkdir()
    (fake_work / ".env").write_text("GEMINI_API_KEY=LEAKED_SECRET_VALUE\n", encoding="utf-8")
    env_clean = {
        k: v
        for k, v in dict(__import__("os").environ).items()
        if "API_KEY" not in k.upper()
    }
    env_clean["PYTHON_DOTENV_DISABLED"] = "1"
    env_clean["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc_env = subprocess.run(
        [
            sys.executable,
            "-c",
            "from subtrans.config import get_gemini_api_keys; assert get_gemini_api_keys() == []",
        ],
        cwd=str(fake_work),
        env=env_clean,
        capture_output=True,
        text=True,
    )
    assert proc_env.returncode == 0, proc_env.stderr

    # 4. check_gemini_quota_for_queue tính đúng gemini-3.5-flash-lite x 3 keys = 1500 RPD ở chế độ normal
    iso_cache = TranslationCache(db_path=tmp_path / "sched_quota.db")
    monkeypatch.setattr("subtrans.config.get_gemini_api_keys", lambda: ["k1", "k2", "k3"])
    q_normal = check_gemini_quota_for_queue(queue_len=6, subs_quality="normal", cache=iso_cache)
    assert q_normal["primary_model"] == "gemini-3.5-flash-lite"
    assert q_normal["num_keys"] == 3
    assert q_normal["primary_limit_rpd"] == 1500
    assert q_normal["warnings"] == []

    q_high = check_gemini_quota_for_queue(queue_len=20, subs_quality="high", cache=iso_cache)
    assert q_high["primary_model"] == "gemini-3.8-flash"
    assert q_high["primary_limit_rpd"] == 60
    assert len(q_high["warnings"]) == 1
