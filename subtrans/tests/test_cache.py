import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pysubs2
import pytest

from subtrans.cache import TranslationCache, compute_contextual_cue_sig, compute_cue_cache_key
from subtrans.glossary import compute_glossary_hash
from subtrans.models import Glossary
from subtrans.pipeline import TranslationPipeline


def test_translation_cache(tmp_path):
    db_file = tmp_path / "test_trans.db"
    cache = TranslationCache(db_path=db_file)

    key = compute_cue_cache_key("gemini-2.5-flash", "こんにちは", "Hello")
    assert cache.get_cue(key) is None

    cache.set_cue(key, "gemini", "gemini-2.5-flash", "Xin chào")
    assert cache.get_cue(key) == "Xin chào"
    assert cache.count() == 1


def test_cache_hydrates_model_and_backend_from_columns(tmp_path):
    db_file = tmp_path / "hydrate.db"
    cache = TranslationCache(db_path=db_file)

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO translations (cache_key, backend, model, vi, pt_date, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("k_plain", "gemini", "gemini-3.5-flash-lite", "Xin chào", "2026-11-22", None),
        )
        conn.execute(
            "INSERT INTO translations (cache_key, backend, model, vi, pt_date, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "k_rep",
                "gemini",
                "gemini-3.8-flash (repair)",
                "Chào Sư phụ!",
                "2026-11-22",
                json.dumps({"speaker": "Fran"}),
            ),
        )
        conn.commit()

    single_plain = cache.get_cue_entry("k_plain")
    assert single_plain == ("Xin chào", {"model": "gemini-3.5-flash-lite", "backend": "gemini"})

    single_rep = cache.get_cue_entry("k_rep")
    assert single_rep == (
        "Chào Sư phụ!",
        {"speaker": "Fran", "model": "gemini-3.8-flash (repair)", "backend": "gemini"},
    )

    batched = cache.get_cues_batch(["k_plain", "k_rep"])
    assert batched["k_plain"] == single_plain
    assert batched["k_rep"] == single_rep


def test_subs_upgrade_rejects_lite_local_and_preserves_repair_provenance(tmp_path, monkeypatch):
    import src.config as app_config
    import subtrans.rag as rag_mod

    monkeypatch.setattr(app_config, "PENDING_SUBS_FILE", str(tmp_path / "pending_subs.json"))
    monkeypatch.setattr(rag_mod, "KNOWLEDGE_DB_PATH", tmp_path / "knowledge.db")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    ja_srt = tmp_path / "ep_prov.ja.srt"
    ja_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n何だと！\n", encoding="utf-8")

    # 1. Cascade repair stores repaired model provenance and subsequent cache hit preserves it
    db_repair = tmp_path / "repair_prov.db"
    p1 = TranslationPipeline(
        dry_run=False,
        enable_cascade=True,
        enable_rag=False,
        subs_quality="high",
        output_dir=tmp_path / "out1",
        cache_db_path=db_repair,
    )
    flash_backend = MagicMock(model="gemini-3.8-flash", stats={})
    flash_backend.translate_batch.return_value = ["Cái gì!"]
    p1.gemini_backends = {"gemini-3.8-flash": flash_backend}
    p1.gemini_backend = flash_backend

    def _fake_cascade(cues, repair_translator, **kwargs):
        cues[0].vi = "Chuyện gì thế!"
        cues[0].model_used = "gemini-3.8-flash (repair)"
        cues[0].uncertain = False
        return {"repaired_count": 1}

    with patch("subtrans.cascade.run_cascade_repair_pass", side_effect=_fake_cascade), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        p1.run(ja_srt)

    p2 = TranslationPipeline(
        dry_run=False,
        enable_cascade=False,
        enable_rag=False,
        subs_quality="high",
        output_dir=tmp_path / "out2",
        cache_db_path=db_repair,
    )
    no_call_backend = MagicMock(model="gemini-3.8-flash", stats={})
    no_call_backend.translate_batch.side_effect = AssertionError("Should use repaired cache entry")
    p2.gemini_backends = {"gemini-3.8-flash": no_call_backend}
    p2.gemini_backend = no_call_backend

    seen_models = []
    from subtrans import report as report_mod
    orig_report = report_mod.generate_qa_report

    def _capture_report(*args, **kwargs):
        seen_models.extend(c.model_used for c in kwargs["cues"])
        return orig_report(*args, **kwargs)

    with patch("subtrans.pipeline.generate_qa_report", side_effect=_capture_report), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        res2 = p2.run(ja_srt)

    assert res2["stats"]["cache_hits"] == 1
    assert seen_models == ["gemini-3.8-flash (repair)"]

    # 2. --subs-upgrade rejects Lite-tagged SQLite cache entries and upgrades via Flash
    db_up = tmp_path / "upgrade.db"
    up_cache = TranslationCache(db_path=db_up)
    ctx_sig = compute_contextual_cue_sig(ja="何だと！")
    ck_target = compute_cue_cache_key(
        model="gemini-3.8-flash",
        ja="何だと！",
        en="",
        glossary_hash=compute_glossary_hash(Glossary()),
        cast_sheet_hash="",
        prompt_version="v2.2",
        stage="draft",
        rag_hash="no_rag",
        context_sig=ctx_sig,
    )
    up_cache.set_cue(ck_target, "gemini", "gemini-3.5-flash-lite", "Bản Lite cũ")

    p_up = TranslationPipeline(
        dry_run=False,
        enable_cascade=False,
        enable_rag=False,
        subs_upgrade=True,
        output_dir=tmp_path / "out_up",
        cache_db_path=db_up,
    )
    up_flash = MagicMock(model="gemini-3.8-flash", stats={})
    up_flash.translate_batch.return_value = ["Bản Flash mới!"]
    p_up.gemini_backends = {"gemini-3.8-flash": up_flash}
    p_up.gemini_backend = up_flash

    with patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        res_up = p_up.run(ja_srt)

    assert res_up["stats"]["cache_hits"] == 0
    assert up_flash.translate_batch.call_count == 1

    # 3. Non-TranslationCache custom cache uses get_cue_entry and rejects "local" under --subs-upgrade
    class _CustomEntryCache:
        def get_cue_entry(self, _key):
            return ("Bản local cũ", {"model": "local-gemma"})

        def get_cue(self, _key):
            return "Bản local cũ"

        def set_cue(self, *_args, **_kwargs):
            pass

        def log_request(self, *_args, **_kwargs):
            pass

        def get_cues_today_pt(self):
            return {}

        def get_requests_today_pt(self):
            return {}

        def is_model_exhausted(self, _model):
            return False

        def is_key_model_exhausted(self, _key_id, _model):
            return False

    p_custom = TranslationPipeline(
        dry_run=False,
        enable_cascade=False,
        enable_rag=False,
        subs_upgrade=True,
        output_dir=tmp_path / "out_custom",
        cache_db_path=tmp_path / "unused.db",
    )
    p_custom.cache = _CustomEntryCache()
    custom_flash = MagicMock(model="gemini-3.8-flash", stats={})
    custom_flash.translate_batch.return_value = ["Bản Flash thay local!"]
    p_custom.gemini_backends = {"gemini-3.8-flash": custom_flash}
    p_custom.gemini_backend = custom_flash

    with patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        res_custom = p_custom.run(ja_srt)

    assert res_custom["stats"]["cache_hits"] == 0
    assert custom_flash.translate_batch.call_count == 1


def _isolate_pipeline_env(tmp_path, monkeypatch):
    import src.config as app_config
    import subtrans.rag as rag_mod

    monkeypatch.setenv("GEMINI_API_KEY", "AIzaTestKey1")
    monkeypatch.setattr(app_config, "PENDING_SUBS_FILE", str(tmp_path / "pending_subs.json"))
    monkeypatch.setattr(rag_mod, "KNOWLEDGE_DB_PATH", tmp_path / "knowledge.db")


def test_cache_key_differentiates_gloss_skill_names(tmp_path, monkeypatch):
    """Verify cache key differentiates gloss_skill_names on identical JA/EN cues."""
    _isolate_pipeline_env(tmp_path, monkeypatch)
    ja_srt = tmp_path / "ep_skill.ja.srt"
    en_srt = tmp_path / "ep_skill.en.srt"
    ja_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n絶対零度！\n", encoding="utf-8")
    en_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nAbsolute Zero!\n", encoding="utf-8")

    mock_client = MagicMock()

    def _fake_generate(model, contents, config):
        resp = MagicMock()
        resp.text = json.dumps([{"id": 0, "vi": "Độ Không Tuyệt Đối!"}])
        return resp

    mock_client.models.generate_content.side_effect = _fake_generate
    db_path = tmp_path / "skill.db"

    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_client), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        TranslationPipeline(
            dry_run=False,
            enable_cascade=False,
            enable_rag=False,
            gloss_skill_names=False,
            output_dir=tmp_path / "o1",
            cache_db_path=db_path,
        ).run(ja_srt, en_srt)
        res2 = TranslationPipeline(
            dry_run=False,
            enable_cascade=False,
            enable_rag=False,
            gloss_skill_names=True,
            output_dir=tmp_path / "o2",
            cache_db_path=db_path,
        ).run(ja_srt, en_srt)

    assert res2["stats"]["cache_hits"] == 0
    assert mock_client.models.generate_content.call_count == 2


def test_cache_key_differentiates_en_only_source(tmp_path, monkeypatch):
    """Verify cache key differentiates JA-mixed and EN-only episodes sharing ASCII lines."""
    _isolate_pipeline_env(tmp_path, monkeypatch)
    mix_srt = tmp_path / "mix.ja.srt"
    enonly_srt = tmp_path / "enonly.ja.srt"
    mix_srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nはい\n\n"
        "2\n00:00:02,500 --> 00:00:03,500\nAlright.\n\n"
        "3\n00:00:04,000 --> 00:00:05,000\nOK.\n",
        encoding="utf-8",
    )
    enonly_srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHello.\n\n"
        "2\n00:00:02,500 --> 00:00:03,500\nAlright.\n\n"
        "3\n00:00:04,000 --> 00:00:05,000\nOK.\n",
        encoding="utf-8",
    )

    mock_client = MagicMock()

    def _fake_generate(model, contents, config):
        payload = json.loads(contents.splitlines()[-1])
        items = payload.get("cues_to_translate", [])
        resp = MagicMock()
        resp.text = json.dumps([{"id": item["id"], "vi": f"Dịch {item['id']}"} for item in items])
        return resp

    mock_client.models.generate_content.side_effect = _fake_generate
    db_path = tmp_path / "enonly.db"

    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_client), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        rm = TranslationPipeline(
            dry_run=False,
            enable_cascade=False,
            enable_rag=False,
            output_dir=tmp_path / "om",
            cache_db_path=db_path,
        ).run(mix_srt)
        re_only = TranslationPipeline(
            dry_run=False,
            enable_cascade=False,
            enable_rag=False,
            output_dir=tmp_path / "oe",
            cache_db_path=db_path,
        ).run(enonly_srt)

    assert rm["stats"]["en_only_source"] is False
    assert re_only["stats"]["en_only_source"] is True
    assert re_only["stats"]["cache_hits"] == 0


def test_subs_upgrade_reuses_lite_keyed_cascade_repair(tmp_path, monkeypatch):
    """Verify --subs-upgrade reuses Cascade Flash repair rows written during normal Lite pass."""
    _isolate_pipeline_env(tmp_path, monkeypatch)
    ja_srt = tmp_path / "ep_cascade_up.ja.srt"
    ja_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nフラン、行くぞ！\n", encoding="utf-8")

    mock_client = MagicMock()

    def _fake_generate(model, contents, config):
        text = str(contents)
        if "<cues>" in text and "</cues>" in text:
            raw_json = text.rsplit("<cues>", 1)[1].rsplit("</cues>", 1)[0].strip()
        else:
            raw_json = text.splitlines()[-1]
        payload = json.loads(raw_json)
        resp = MagicMock()
        if "cues_to_review" in payload:
            resp.text = json.dumps([
                {"id": 0, "vi": "Fran, đi thôi!", "changed": True, "change_reason": "fixed"}
            ])
        else:
            resp.text = json.dumps([
                {"id": 0, "vi": "Em, đi thôi!", "uncertain": "pronoun"}
            ])
        return resp

    mock_client.models.generate_content.side_effect = _fake_generate
    db_path = tmp_path / "cascade_upgrade.db"

    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_client), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        r1 = TranslationPipeline(
            dry_run=False,
            subs_quality="normal",
            enable_cascade=True,
            enable_rag=False,
            output_dir=tmp_path / "r1",
            cache_db_path=db_path,
        ).run(ja_srt)
        assert r1["stats"]["cascade_repaired"] == 1

        mock_client.models.generate_content.reset_mock()
        r2 = TranslationPipeline(
            dry_run=False,
            subs_upgrade=True,
            enable_cascade=False,
            enable_rag=False,
            output_dir=tmp_path / "r2",
            cache_db_path=db_path,
        ).run(ja_srt)

    assert r2["stats"]["cache_hits"] == 1
    assert mock_client.models.generate_content.call_count == 0


def test_pending_subs_preserves_piecewise_offsets(tmp_path, monkeypatch):
    """Verify pending_subs.json persists piecewise_ja_offsets for incomplete episodes."""
    from src.pending_queue import load_pending_subs

    _isolate_pipeline_env(tmp_path, monkeypatch)
    ja_ssa = pysubs2.SSAFile()
    en_ssa = pysubs2.SSAFile()
    cur_t = 5000
    for i in range(20):
        dur = 1800 + (i * 173) % 900
        en_ssa.append(pysubs2.SSAEvent(start=cur_t, end=cur_t + dur, text=f"Cold open line {i}"))
        ja_ssa.append(pysubs2.SSAEvent(start=cur_t, end=cur_t + dur, text=f"アバンのセリフ{i}"))
        cur_t += dur + 1400 + (i * 617) % 2300
    cur_t += 90000
    for i in range(25):
        dur = 1900 + (i * 211) % 850
        en_ssa.append(pysubs2.SSAEvent(start=cur_t, end=cur_t + dur, text=f"Post-OP line {i}"))
        ja_ssa.append(pysubs2.SSAEvent(start=cur_t - 10000, end=cur_t - 10000 + dur, text=f"本編のセリフ{i}"))
        cur_t += dur + 1500 + (i * 733) % 2500

    ja_p = tmp_path / "pw.ja.srt"
    en_p = tmp_path / "pw.en.srt"
    ja_ssa.save(str(ja_p))
    en_ssa.save(str(en_p))

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("400 Bad Request")

    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_client), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        TranslationPipeline(
            dry_run=False,
            enable_cascade=False,
            enable_rag=False,
            output_dir=tmp_path / "opw",
            cache_db_path=tmp_path / "pw.db",
        ).run(ja_p, en_p, output_basename="ep_piecewise_pending", auto_offset=True)

    recs = load_pending_subs()
    assert len(recs) == 1
    assert "piecewise_ja_offsets" in recs[0]
    assert len(recs[0]["piecewise_ja_offsets"]) == 2


def test_pipeline_run_honors_piecewise_offsets_with_nonzero_scalar_offset(tmp_path, monkeypatch):
    """Verify TranslationPipeline.run applies piecewise_ja_offsets even when sub_offset_ms != 0 and auto_offset=False."""
    _isolate_pipeline_env(tmp_path, monkeypatch)
    ja_p = tmp_path / "pw_retry.ja.srt"
    ja_p.write_text(
        "1\n00:00:05,000 --> 00:00:07,000\nアバンのセリフ\n\n"
        "2\n00:01:50,000 --> 00:01:52,000\n本編のセリフ\n",
        encoding="utf-8",
    )

    with patch("subtrans.pipeline.detect_auto_offset") as mock_det, \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        res = TranslationPipeline(
            dry_run=True,
            enable_cascade=False,
            enable_rag=False,
            output_dir=tmp_path / "retry_out",
            cache_db_path=tmp_path / "retry.db",
        ).run(
            ja_p,
            sub_offset_ms=-10000,
            auto_offset=False,
            piecewise_ja_offsets=[
                {"until_ms": 100000, "offset_ms": 0},
                {"from_ms": 100000, "offset_ms": 10000},
            ],
        )

    assert mock_det.call_count == 0
    srt_text = Path(res["output_path"]).read_text(encoding="utf-8")
    assert "00:00:05,000 --> 00:00:07,000" in srt_text
    assert "00:02:00,000 --> 00:02:02,000" in srt_text


def test_concurrent_cache_writers_and_night_report_reader_wal(tmp_path):
    """Verify concurrent WAL writes from TranslationCache and reads from night_report succeed cleanly."""
    from src.night_report import query_subtrans_api_stats

    cdb = tmp_path / "wal_stress.db"
    TranslationCache(db_path=cdb)
    errors = []

    def writer(tid: int):
        try:
            c = TranslationCache(db_path=cdb)
            for i in range(15):
                c.set_cue(f"k_{tid}_{i}", "gemini", "gemini-3.5-flash-lite", f"Câu {tid}-{i}")
                c.log_request(model="gemini-3.5-flash-lite", backend="gemini", cues_count=1, status="success")
        except Exception as exc:
            errors.append(exc)

    def reader():
        try:
            for _ in range(20):
                query_subtrans_api_stats(str(cdb))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    threads.append(threading.Thread(target=reader))
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    stats = query_subtrans_api_stats(str(cdb))
    assert stats["model_summary"]["gemini-3.5-flash-lite"]["total"] == 60


def test_pipeline_runs_optimize_once_propagates_fallback_and_drops_enable_review(tmp_path, monkeypatch):
    """Verify single optimize_cues_for_playback pass, used_en_fallback propagation, and enable_review removal."""
    _isolate_pipeline_env(tmp_path, monkeypatch)

    with pytest.raises(TypeError):
        TranslationPipeline(enable_review=True, output_dir=tmp_path, cache_db_path=tmp_path / "dead.db")

    ja_srt = tmp_path / "opt_once.ja.srt"
    en_srt = tmp_path / "opt_once.en.srt"
    ja_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nテスト\n", encoding="utf-8")
    en_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello fallback\n", encoding="utf-8")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("400 Bad Request")

    import subtrans.pipeline as pipe_mod
    opt_calls = 0
    orig_opt = pipe_mod.optimize_cues_for_playback

    def _spy_opt(cues, **kwargs):
        nonlocal opt_calls
        opt_calls += 1
        return orig_opt(cues, **kwargs)

    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_client), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()), \
         patch("subtrans.pipeline.optimize_cues_for_playback", side_effect=_spy_opt):
        res = TranslationPipeline(
            dry_run=False,
            enable_cascade=False,
            enable_rag=False,
            output_dir=tmp_path / "o_opt",
            cache_db_path=tmp_path / "opt.db",
        ).run(ja_srt, en_srt)

    assert opt_calls == 1
    assert res["is_qualified"] is False
    assert res["real_rate"] == 0.0
    assert Path(res["output_path"]).name.endswith(".vi.partial.srt")


def test_qualification_gate_rejects_empty_after_clean_sfx_cues(tmp_path, monkeypatch):
    """Verify cues normalizing to empty (e.g. '[SFX]') are excluded from real_translated_cues."""
    from src.pending_queue import load_pending_subs

    _isolate_pipeline_env(tmp_path, monkeypatch)
    ja_srt = tmp_path / "sfx_only.ja.srt"
    ja_srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nおはよう\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nこんにちは\n",
        encoding="utf-8",
    )

    mock_client = MagicMock()

    def _fake_generate(model, contents, config):
        resp = MagicMock()
        resp.text = json.dumps([
            {"id": 0, "vi": "[SFX]"},
            {"id": 1, "vi": "Xin chào [SFX]"},
        ])
        return resp

    mock_client.models.generate_content.side_effect = _fake_generate

    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_client), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        res = TranslationPipeline(
            dry_run=False,
            enable_cascade=False,
            enable_rag=False,
            output_dir=tmp_path / "o_sfx",
            cache_db_path=tmp_path / "sfx.db",
        ).run(ja_srt, output_basename="ep_sfx_gate")

    assert res["real_rate"] == pytest.approx(0.5)
    assert res["is_qualified"] is False
    assert Path(res["output_path"]).name.endswith(".vi.partial.srt")
    recs = load_pending_subs()
    assert len(recs) == 1 and recs[0]["base_name"] == "ep_sfx_gate"
