import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

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
