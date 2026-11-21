# -*- coding: utf-8 -*-
"""Offline unit tests covering Subtrans & operational script audit fixes."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pysubs2
import pytest
from typer.testing import CliRunner

import scripts.agent_context as agent_context_mod
import scripts.batch_translate_all as batch_mod
import scripts.benchmark_pipeline as bench_mod
import scripts.oneoff.embed_fate_subtitles as embed_fate_mod
import scripts.oneoff.organize_vietnamese_subs as org_subs_mod
import scripts.set_track_defaults as set_track_mod
import scripts.upscale_x4_single as upscale_single_mod
import subtrans.config as st_config
from subtrans import audio_diarization, cascade, cast_sheet, linter, parser, writer
from subtrans.cache import TranslationCache, execute_and_log_api_request
from subtrans.cli import app as subtrans_cli_app
from subtrans.gemini_backend import GeminiTranslator, build_episode_system_prompt
from subtrans.models import Cue, Glossary, GlossaryTerm
from subtrans.pipeline import TranslationPipeline
from subtrans.rag import KnowledgeBase
from subtrans.report import generate_qa_report
from subtrans.throttle import QuotaExhaustedError, call_with_retry


# ---------------------------------------------------------------------------
# P0.1: GeminiBackend bounded 429 PerMinute, cooldown-before-rotate, 500/502/504
# ---------------------------------------------------------------------------


def test_gemini_429_per_minute_bounded_raises_quota_exhausted(tmp_path):
    cache = TranslationCache(db_path=tmp_path / "cache.db")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        "429 RESOURCE_EXHAUSTED: PerMinute rate limit reached"
    )
    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_client):
        tr = GeminiTranslator(
            api_keys=["AIzaKeyOne111", "AIzaKeyTwo222"],
            model="gemini-3.5-flash-lite",
            cache=cache,
            round_robin=True,
        )
        with patch("time.sleep", return_value=None):
            with pytest.raises(QuotaExhaustedError):
                tr._call_api("hello world")

    # With 2 keys: initial attempt + max_rpm_retries (4) => 5 attempts max, no infinite loop
    assert mock_client.models.generate_content.call_count <= 5


def test_gemini_429_per_minute_respects_stage_deadline_and_cooldown_before_rotate(tmp_path):
    cache = TranslationCache(db_path=tmp_path / "cache.db")
    mock_client = MagicMock()
    ok_resp = MagicMock()
    ok_resp.text = "[]"
    ok_resp.usage_metadata = None
    mock_client.models.generate_content.side_effect = [
        Exception("429 RESOURCE_EXHAUSTED PerMinute quota"),
        ok_resp,
    ]

    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_client):
        tr = GeminiTranslator(
            api_keys=["AIzaKeyOne111", "AIzaKeyTwo222"],
            model="gemini-3.5-flash-lite",
            cache=cache,
            round_robin=True,
        )
        events = []
        orig_limiter = tr.rate_limiters[tr.key_ids[0]]
        orig_record_cooldown = orig_limiter.record_cooldown
        orig_select_rr = tr.select_next_round_robin_key

        def wrapped_cooldown(seconds):
            events.append(("cooldown", seconds))
            return orig_record_cooldown(seconds)

        def wrapped_select_rr(reason=""):
            events.append(("rotate", tr.current_key_idx))
            return orig_select_rr(reason=reason)

        tr.rate_limiters[tr.key_ids[0]].record_cooldown = wrapped_cooldown
        tr.select_next_round_robin_key = wrapped_select_rr
        tr.stage_deadline = time.monotonic() + 30.0

        with patch("time.sleep", return_value=None):
            res = tr._call_api("prompt")

        assert res == []
        assert events[0][0] == "cooldown"
        assert events[1][0] == "rotate"

        # Past deadline should abort cleanly before sleeping
        tr.stage_deadline = time.monotonic() - 1.0
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED PerMinute quota")
        with pytest.raises(QuotaExhaustedError):
            tr._call_api("prompt")


@pytest.mark.parametrize(
    "msg",
    [
        "500 INTERNAL server error",
        "502 Bad Gateway from proxy",
        "504 Gateway Timeout upstream",
        "Service experiencing high demand",
    ],
)
def test_gemini_and_throttle_transient_tokens_aligned(tmp_path, msg):
    cache = TranslationCache(db_path=tmp_path / "cache.db")
    ok_resp = MagicMock()
    ok_resp.text = '[{"id": 1, "vi": "Xin chao"}]'
    ok_resp.usage_metadata = None
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [Exception(msg), ok_resp]

    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_client):
        tr = GeminiTranslator(api_keys=["AIzaKeyOne111"], model="gemini-3.5-flash-lite", cache=cache)
        with patch("time.sleep", return_value=None):
            out = tr._call_api("prompt")
    assert out[0]["vi"] == "Xin chao"

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(msg)
        return "ok"

    with patch("subtrans.throttle.time.sleep", return_value=None):
        assert call_with_retry(flaky, max_retries=1, base_delay=0.01) == "ok"


def test_build_episode_system_prompt_cached_by_hash():
    g = Glossary(terms=[GlossaryTerm(ja="魔剣", en="Magic Sword", vi="Ma kiếm")])
    p1 = build_episode_system_prompt(g)
    p2 = build_episode_system_prompt(g)
    assert p1 == p2
    assert "Ma kiếm" in p1


# ---------------------------------------------------------------------------
# P0.2: StageAPrefetchWorker race, atomic .cast.yaml, cleanup, embed no-retry
# ---------------------------------------------------------------------------


def test_stage_a_prefetch_worker_waits_for_running_task_past_timeout(tmp_path):
    finished = {"done": False}

    def slow_prefetch(fpath: str, output_dir: Path, subs_quality: str = "normal"):
        time.sleep(0.18)
        finished["done"] = True
        return {"status": "PREFETCHED", "episode": Path(fpath).stem, "characters": 3}

    worker = batch_mod.StageAPrefetchWorker(prefetch_fn=slow_prefetch)
    ep = str(tmp_path / "SeriesX" / "E01.mkv")
    worker.submit_next(ep, output_dir=tmp_path, same_series_headstart_sec=0.0)
    time.sleep(0.03)  # ensure thread has started

    res = worker.wait_if_inflight(ep, timeout=0.02)
    worker.shutdown(cancel=False)

    assert finished["done"] is True
    assert res == {"status": "PREFETCHED", "episode": "E01", "characters": 3}


def test_stage_a_prefetch_skips_same_series_when_series_cast_missing(tmp_path):
    calls = []

    def fake_prefetch(fpath: str, output_dir: Path, subs_quality: str = "normal"):
        calls.append(fpath)
        return {"status": "PREFETCHED"}

    worker = batch_mod.StageAPrefetchWorker(prefetch_fn=fake_prefetch)
    ep1 = str(tmp_path / "Fate" / "Fate_E01.mkv")
    ep2 = str(tmp_path / "Fate" / "Fate_E02.mkv")
    worker.submit_next(ep2, current_fpath=ep1, output_dir=tmp_path, same_series_headstart_sec=0.01)
    res = worker.wait_if_inflight(ep2, timeout=2.0)
    worker.shutdown(cancel=False)

    assert res is not None and res["status"] == "SKIPPED_WAIT_SERIES_CAST"
    assert calls == []

    # Direct call also skips if skip_if_series_cast_missing=True and series cast file is absent
    res_direct = batch_mod.prefetch_episode_stage_a(
        ep2,
        output_dir=tmp_path,
        skip_if_series_cast_missing=True,
    )
    assert res_direct["status"] == "SKIPPED_WAIT_SERIES_CAST"


def test_prefetch_episode_stage_a_passes_cache_and_cleans_temp_in_finally(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_mod, "PROJECT_ROOT", Path(batch_mod.__file__).resolve().parent.parent)
    cleaned = []
    seen_cache = []
    ja_file = tmp_path / "Ep01.ja.srt"
    ja_file.write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n", encoding="utf-8")

    def fake_extract(video_path, prefer_whisper_when_no_ja=False, dry_run=False):
        return (str(ja_file), None)

    def fake_load_cast(**kwargs):
        seen_cache.append(kwargs.get("cache"))
        raise RuntimeError("boom quota")

    def fake_cleanup(vid_dir, base_name, logger=None):
        cleaned.append((vid_dir, base_name))
        return 1

    with patch("src.subtitles.find_or_extract_subtitles", side_effect=fake_extract), \
         patch("src.cleanup.cleanup_subtitle_extracts", side_effect=fake_cleanup), \
         patch("subtrans.cast_sheet.load_or_generate_series_cast_sheet", side_effect=fake_load_cast), \
         patch("subtrans.cache.TranslationCache") as mock_cache_cls:
        mock_cache_cls.return_value = "CACHE_OBJ"
        res = batch_mod.prefetch_episode_stage_a(str(tmp_path / "Ep01.mkv"), output_dir=tmp_path / "out")

    assert res["status"] == "ERROR"
    assert seen_cache == ["CACHE_OBJ"]
    assert cleaned == [(str(tmp_path.resolve()), "Ep01")]


def test_cast_sheet_atomic_write_yaml_leaves_no_tmp(tmp_path):
    target = tmp_path / "series.cast.yaml"
    cast_sheet._atomic_write_yaml(target, {"characters": [{"name_ja": "剣", "name_vi": "Ken"}]})
    assert target.exists()
    assert "Ken" in target.read_text(encoding="utf-8")
    assert [p for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_batch_translate_does_not_retry_3x_when_only_embed_fails():
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = (
        "✅ Hoàn tất dịch phụ đề độc lập cho 'Ep01.mkv'. File: E:/Films/Subtitles/Ep01.srt\n"
        "⚠️ Không thể nhúng phụ đề vào container (audit không đạt hoặc lỗi), file phụ đề ngoài vẫn sẵn sàng.\n"
    )
    fake_proc.stderr = ""

    with patch("scripts.batch_translate_all.subprocess.run", return_value=fake_proc) as mock_run, \
         patch("scripts.batch_translate_all.has_vietnamese_sub", return_value=False):
        res = batch_mod.run_translation_for_file("E:/dummy/Ep01.mkv", max_retries=3)

    assert res["status"] == "EMBED_FAILED"
    assert res["attempt"] == 1
    assert mock_run.call_count == 1


def test_batch_translate_cli_overrides_monkeypatch_and_dry_run(tmp_path, monkeypatch):
    src_dir = tmp_path / "raw_sources"
    frieren_dir = src_dir / "Frieren"
    fate_dir = src_dir / "Fate"
    frieren_dir.mkdir(parents=True)
    fate_dir.mkdir(parents=True)
    ep1 = frieren_dir / "Frieren_E01.mkv"
    ep2 = fate_dir / "Fate_E01.mkv"
    ep1.write_bytes(b"MKV1")
    ep2.write_bytes(b"MKV2")

    out_dir = tmp_path / "subs_out"
    state_file = tmp_path / "custom_state" / "batch_state.json"

    # 1. Parser accepts --source-dir, -o/--output-dir, --state-file, and --dry-run
    parsed = batch_mod.build_cli_parser().parse_args([
        "--source-dir", str(src_dir),
        "-o", str(out_dir),
        "--state-file", str(state_file),
        "--dry-run",
    ])
    assert parsed.source_dir == src_dir
    assert parsed.output_dir == out_dir
    assert parsed.state_file == state_file
    assert parsed.dry_run is True

    # 2. monkeypatch.setattr resolves module globals at call time (never writes live logs/)
    monkeypatch.setattr(batch_mod, "RAW_SOURCES_DIR", src_dir)
    monkeypatch.setattr(batch_mod, "SUB_OUT_DIR", out_dir)
    monkeypatch.setattr(batch_mod, "STATE_FILE", state_file)

    queued = batch_mod.collect_queue()
    assert len(queued) == 2
    batch_mod.save_state({"total": 2, "completed": 1, "skipped": 0, "failed": 0, "history": {}})
    assert state_file.exists()
    assert batch_mod.load_state()["completed"] == 1

    # 3. --dry-run without --state-file does NOT create or mutate default STATE_FILE on disk
    default_state_dry = tmp_path / "untouched" / "default_state.json"
    monkeypatch.setattr(batch_mod, "STATE_FILE", default_state_dry)
    sub_cmds = []
    prefetch_calls = []

    def _fake_subproc(cmd, **kwargs):
        sub_cmds.append(cmd)
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "[DRY-RUN] ok\n"
        proc.stderr = ""
        return proc

    def _fake_prefetch(fpath, output_dir=None, subs_quality="normal", dry_run=False):
        prefetch_calls.append((fpath, output_dir, subs_quality, dry_run))
        return {"status": "SKIPPED_NO_TEXT_SUB", "episode": Path(fpath).stem}

    orig_worker_cls = batch_mod.StageAPrefetchWorker
    monkeypatch.setattr(
        batch_mod,
        "StageAPrefetchWorker",
        lambda: orig_worker_cls(prefetch_fn=_fake_prefetch),
    )
    with patch("scripts.batch_translate_all.subprocess.run", side_effect=_fake_subproc), \
         patch("scripts.batch_translate_all.has_vietnamese_sub", return_value=False):
        batch_mod.main([
            "--source-dir", str(src_dir),
            "-o", str(out_dir),
            "--dry-run",
        ])

    assert not default_state_dry.exists()
    assert len(sub_cmds) == 2
    assert all("--dry-run" in cmd and str(out_dir) in cmd for cmd in sub_cmds)
    assert len(prefetch_calls) == 1
    assert prefetch_calls[0][1] == out_dir
    assert prefetch_calls[0][3] is True

    # 4. Direct prefetch_episode_stage_a(dry_run=True) avoids TranslationCache and forwards dry_run
    seen_dry = []
    with patch("src.subtitles.find_or_extract_subtitles", side_effect=lambda f, **kw: seen_dry.append(kw.get("dry_run")) or (None, None)), \
         patch("subtrans.cache.TranslationCache", side_effect=AssertionError("Should not open cache in dry_run")):
        pf = batch_mod.prefetch_episode_stage_a(str(ep1), output_dir=out_dir, dry_run=True)
    assert pf["status"] == "SKIPPED_NO_TEXT_SUB"
    assert seen_dry == [True]


# ---------------------------------------------------------------------------
# P0.3: Clamp negative subtitle timestamps (parser.py & writer.py)
# ---------------------------------------------------------------------------


def test_negative_offset_clamps_partial_and_drops_fully_negative_events(tmp_path):
    ja_srt = tmp_path / "ep.ja.srt"
    ja_srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n完全に負\n\n"
        "2\n00:00:04,000 --> 00:00:07,000\n一部が負\n\n"
        "3\n00:00:08,000 --> 00:00:08,010\n正の極小\n",
        encoding="utf-8",
    )
    cues, _, _ = parser.parse_and_align_subtitles(str(ja_srt), ja_offset_ms=-5000)
    assert len(cues) == 2
    assert cues[0].ja == "一部が負"
    assert cues[0].start == 0
    assert cues[0].end == 2000
    assert cues[1].start == 3000
    assert cues[1].end == 3050

    assert writer.format_timestamp_srt(-1234) == "00:00:00,000"


# ---------------------------------------------------------------------------
# P1.4: SQLite WAL, busy_timeout=10000, batch exhaustion, single 429 log
# ---------------------------------------------------------------------------


def test_sqlite_wal_busy_timeout_and_batched_keys(tmp_path):
    cache = TranslationCache(db_path=tmp_path / "cache.db")
    with cache._connect() as conn:
        busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert busy == 10000
    assert str(mode).lower() == "wal"

    assert cache.are_all_keys_exhausted([], "gemini-3.5-flash-lite") is False
    cache.mark_key_model_exhausted("k1", "gemini-3.5-flash-lite", reason="429 PerDay")
    assert cache.are_all_keys_exhausted(["k1", "k2"], "gemini-3.5-flash-lite") is False
    cache.mark_key_model_exhausted("k2", "gemini-3.5-flash-lite", reason="429 PerDay")
    assert cache.are_all_keys_exhausted(["k1", "k2"], "gemini-3.5-flash-lite") is True

    # Expired row cleanup works cleanly under BEGIN IMMEDIATE
    with cache._connect(immediate=True) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO model_exhaustion (model, exhausted_until_pt, reason) VALUES (?, ?, ?)",
            ("gemini-expired", "2020-01-01T00:00:00", "old"),
        )
    assert cache.is_model_exhausted("gemini-expired") is False

    kb = KnowledgeBase(db_path=tmp_path / "kb.db")
    with kb._connect() as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"


def test_dedup_429_perday_logging_via_execute_and_log(tmp_path):
    cache = TranslationCache(db_path=tmp_path / "cache.db")
    call_count = {"n": 0}

    def fail_429():
        call_count["n"] += 1
        return execute_and_log_api_request(
            lambda: (_ for _ in ()).throw(
                Exception("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier quotaValue: 20")
            ),
            model="gemini-3.8-flash",
            cache=cache,
            key_id="k_test",
        )

    with pytest.raises(Exception):
        call_with_retry(fail_429, model_name="gemini-3.8-flash", cache=cache, key_id="k_test", max_retries=0)

    assert call_count["n"] == 1
    with cache._connect() as conn:
        rows = conn.execute(
            "SELECT status FROM api_requests WHERE status = 'exhausted_429_perday'"
        ).fetchall()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# P1.5: Runtime quota YAML, subs_upgrade chain, Cascade exhausted skip, Linter/RAG, --local
# ---------------------------------------------------------------------------


def test_update_model_quota_writes_runtime_yaml_and_ignores_non_positive(tmp_path, monkeypatch):
    rt_file = tmp_path / "models_runtime_quota.yaml"
    monkeypatch.setattr(st_config, "RUNTIME_QUOTA_PATH", rt_file)
    orig_models_yaml = st_config.MODELS_CONFIG_PATH.read_text(encoding="utf-8")

    st_config.update_model_quota_in_yaml("gemini-3.5-flash-lite", "GenerateRequestsPerMinutePerProjectPerModel", 0, custom_path=None)
    assert not rt_file.exists()

    st_config.update_model_quota_in_yaml("gemini-3.5-flash-lite", "GenerateRequestsPerDayPerProjectPerModel", 500, custom_path=None)
    assert rt_file.exists()
    assert "500" in rt_file.read_text(encoding="utf-8")
    assert st_config.MODELS_CONFIG_PATH.read_text(encoding="utf-8") == orig_models_yaml

    custom = tmp_path / "custom_models.yaml"
    custom.write_text("models:\n  gemini-3.8-flash:\n    rpd: 20\n", encoding="utf-8")
    st_config.update_model_quota_in_yaml("gemini-3.8-flash", "GenerateRequestsPerDayPerProjectPerModel", 40, custom_path=custom)
    assert "rpd: 40" in custom.read_text(encoding="utf-8")


def test_subs_upgrade_orders_flash_before_lite_and_cascade_skips_exhausted(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    p_up = TranslationPipeline(subs_upgrade=True, output_dir=tmp_path, cache_db_path=tmp_path / "u1.db")
    assert "lite" not in p_up.gemini_model.lower()

    p_ovr = TranslationPipeline(
        subs_upgrade=True,
        model_override="gemini-3.5-flash-lite",
        output_dir=tmp_path,
        cache_db_path=tmp_path / "u2.db",
    )
    assert p_ovr.gemini_model == "gemini-3.5-flash-lite"

    # Cascade exhausted skip
    ja_srt = tmp_path / "ep.ja.srt"
    ja_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\n何だと！\n", encoding="utf-8")
    p_cas = TranslationPipeline(
        dry_run=True,
        enable_cascade=True,
        enable_rag=False,
        output_dir=tmp_path,
        cache_db_path=tmp_path / "c.db",
    )
    p_cas.cache.mark_model_exhausted("gemini-3.8-flash", reason="quota")

    b38 = MagicMock(model="gemini-3.8-flash", stats={})
    b37 = MagicMock(model="gemini-3.7-flash", stats={})
    b_lite = MagicMock(model="gemini-3.5-flash-lite", stats={})
    b_lite.translate_batch.return_value = ["Cái gì!"]
    p_cas.gemini_backends = {
        "gemini-3.8-flash": b38,
        "gemini-3.7-flash": b37,
        "gemini-3.5-flash-lite": b_lite,
    }
    picked = []
    with patch("subtrans.cascade.run_cascade_repair_pass", side_effect=lambda cues, repair_translator, **kw: picked.append(repair_translator.model) or {"repaired_count": 0}), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        p_cas.run(ja_srt)
    assert picked == ["gemini-3.7-flash"]

    # Unconfigured MagicMock cache should not wrongly mark Flash models as exhausted
    p_mock = TranslationPipeline(
        dry_run=True,
        enable_cascade=True,
        enable_rag=False,
        output_dir=tmp_path,
        cache_db_path=tmp_path / "c_mock.db",
    )
    p_mock.cache = MagicMock()
    p_mock.cache.get_cue.return_value = None
    p_mock.gemini_backends = {
        "gemini-3.8-flash": b38,
        "gemini-3.5-flash-lite": b_lite,
    }
    picked_mock = []
    with patch("subtrans.cascade.run_cascade_repair_pass", side_effect=lambda cues, repair_translator, **kw: picked_mock.append(repair_translator.model) or {"repaired_count": 0}), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        p_mock.run(ja_srt)
    assert picked_mock == ["gemini-3.8-flash"]


def test_local_only_raises_runtime_error_and_cli_exits_1(tmp_path):
    with pytest.raises(RuntimeError, match="Cloud Gemini"):
        TranslationPipeline(local_only=True, output_dir=tmp_path, cache_db_path=tmp_path / "loc.db")

    ja = tmp_path / "ep.ja.srt"
    ja.write_text("1\n00:00:01,000 --> 00:00:02,000\nはい\n", encoding="utf-8")
    runner = CliRunner()
    res_run = runner.invoke(subtrans_cli_app, ["run", str(ja), "--local"])
    assert res_run.exit_code == 1
    assert "Cloud Gemini" in res_run.output

    res_batch = runner.invoke(subtrans_cli_app, ["batch", str(tmp_path), "--local"])
    assert res_batch.exit_code == 1

    res_help = runner.invoke(subtrans_cli_app, ["run", "--help"], env={"COLUMNS": "240"})
    normalized_help = " ".join(res_help.output.split())
    assert "smart-wrap > 72 chars" in normalized_help
    assert "--subs-upgrade" in normalized_help


def test_linter_and_rag_hygiene(tmp_path):
    assert all(pat != r"師匠" for pat, _ in linter.VOCATIVE_MAPPINGS)

    # Short common glossary term does NOT false-positive inside compound word;
    # non-common term enforces token boundary, and Fran/Stephane/Fel remain intact.
    g = Glossary(
        terms=[
            GlossaryTerm(ja="今", vi="bây giờ", genre="common"),
            GlossaryTerm(ja="魔剣", vi="Ma kiếm", genre="battle"),
        ]
    )
    cue_ok = Cue(id=1, start=0, end=2000, ja="今日", vi="Hôm nay")
    assert linter.lint_cue(cue_ok, glossary=g) == []

    cue_bad = Cue(id=2, start=2100, end=4000, ja="それが魔剣だ！", vi="Đó là thanh kiếm!")
    viols = linter.lint_cue(cue_bad, glossary=g)
    assert any(v.rule_id == linter.RULE_GLOSSARY_TERM for v in viols)

    # RAG learn stores matched term in error_pattern + cache invalidates + fallback_builtin works for Hiragana
    kb = KnowledgeBase(db_path=tmp_path / "kb.db")
    fs = kb.retrieve_few_shots([Cue(id=1, start=0, end=1000, ja="はい、わかりました。")], fallback_builtin=True)
    assert len(fs) >= 1

    learned = kb.learn_from_linter_violations(viols, [cue_bad], series="TestSeries", episode="Ep01", glossary=g)
    assert learned == 1
    with kb._connect() as conn:
        rows = conn.execute(
            "SELECT ja_pattern, bad_pattern FROM error_bank WHERE series = 'testseries'"
        ).fetchall()
    assert rows[0][0] == "魔剣"
    assert "Ma kiếm" in rows[0][1]

    # Cache invalidates on add_error_pattern
    cautions_before = kb.retrieve_cautions([Cue(id=3, start=0, end=1000, ja="新ワード")], series="TestSeries")
    assert cautions_before == []
    kb.add_error_pattern("term", "新ワード", "wrong", "right", "use right", series="TestSeries")
    cautions_after = kb.retrieve_cautions([Cue(id=3, start=0, end=1000, ja="新ワード")], series="TestSeries")
    assert len(cautions_after) == 1


# ---------------------------------------------------------------------------
# P2: Scripts, Parser, Writer, Cascade, Audio Diarization, Report, CLI models
# ---------------------------------------------------------------------------


def test_p2_operational_scripts_safety_and_bugs(tmp_path):
    # agent_context MANAGED_HOOK keeps fallback string while preferring venv
    assert "python scripts/agent_context.py --check" in agent_context_mod.MANAGED_HOOK
    assert ".venv/Scripts/python.exe" in agent_context_mod.MANAGED_HOOK

    # set_track_defaults: parse_args([]) stays backward compatible, main([]) fails safe with code 2
    assert set_track_mod.parse_args([]).targets == [r"E:\Films\Raw_Sources"]
    assert set_track_mod.main([]) == 2
    with patch("scripts.set_track_defaults.collect_files", return_value=[]):
        assert set_track_mod.main(["--dry-run"]) == 0
        assert set_track_mod.main(["--yes-default-target"]) == 0

    # benchmark_pipeline defaults zero_disk=True and accepts --no-zero-disk
    src_v = tmp_path / "ep.mkv"
    src_v.write_bytes(b"v")
    with patch("scripts.benchmark_pipeline.run_benchmark", return_value={"mux_status": "PASS", "frames_matched": True}) as mock_rb:
        assert bench_mod.main([str(src_v), "--dry-run"]) == 0
        assert mock_rb.call_args.kwargs["zero_disk"] is True
        assert bench_mod.main([str(src_v), "--no-zero-disk", "--dry-run"]) == 0
        assert mock_rb.call_args.kwargs["zero_disk"] is False

    # upscale_x4_single requires input and supports --dry-run safely
    img = tmp_path / "frame.png"
    img.write_bytes(b"png")
    out_dry = upscale_single_mod.upscale_single_image(str(img), dry_run=True)
    assert out_dry.endswith("frame_4x.png")
    assert not Path(out_dry).exists()

    # organize_vietnamese_subs UBW S2 + HF3 ASCII dir + dry_run
    ubw_vid = tmp_path / "2-Fate stay night Unlimited Blade Works [Ma10p_1080p]"
    ubw_s2 = tmp_path / "fatestay-night-unlimited-blade-works-season-2_vietnamese_3304773-[SubSource]"
    hf3_vid = tmp_path / "3.3-Fate stay night Heaven's Feel III - spring song [Ma10p_1080p]"
    hf3_sub = tmp_path / "fatestay-night-heavens-feel-iii-spring-song-2020_vietnamese_3304670-[SubSource]"
    for d in (ubw_vid, ubw_s2, hf3_vid, hf3_sub):
        d.mkdir(parents=True)
    (ubw_vid / "Fate [13].mkv").write_bytes(b"v")
    (ubw_s2 / "UBW.S02E01.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nvi\n", encoding="utf-8")
    (hf3_vid / "Fate Heaven III.mkv").write_bytes(b"v")
    (hf3_sub / "hf3.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nvi\n", encoding="utf-8")

    dry_stats = org_subs_mod.organize(str(tmp_path), dry_run=True)
    assert dry_stats["Fate stay night UBW"] == 1
    assert dry_stats["Heaven's Feel Trilogy"] == 1
    assert not (ubw_vid / "Fate [13].vi.srt").exists()

    stats = org_subs_mod.organize(str(tmp_path), dry_run=False)
    assert stats["Fate stay night UBW"] == 1
    assert (ubw_vid / "Fate [13].vi.srt").exists()
    assert (hf3_vid / "Fate Heaven III.vi.srt").exists()

    # embed_fate_subtitles maps Fate/Zero S2 to FZ_Z2 and UBW S2 safely
    dl = tmp_path / "dl"
    raw = tmp_path / "raw"
    dl.mkdir()
    raw.mkdir()
    fz_s2_zip = dl / "fatezero-season-2-2011_vietnamese_3231427-[SubSource].zip"
    with zipfile.ZipFile(fz_s2_zip, "w") as z:
        z.writestr("Fate Zero - S02E01.vi.srt", "sub")
    fz_s2_dir = raw / "[Tenrai-Sensei] Fate Zero [BD][1080p][HEVC 10bit x265][Dual Audio]" / "Season 2"
    fz_s2_dir.mkdir(parents=True)
    (fz_s2_dir / "FateZero - S02E01.mkv").write_bytes(b"v")

    wl = embed_fate_mod.build_worklist(dl, raw)
    assert len(wl) == 1
    assert wl[0]["zip_path"] == fz_s2_zip
    assert embed_fate_mod.main(["--downloads-dir", str(dl), "--raw-sources-dir", str(raw), "--dry-run"]) == 0
    assert fz_s2_zip.exists()


def test_p2_subtrans_parser_writer_cascade_diarization_report_pipeline(tmp_path, monkeypatch):
    # Pipeline skips detect_auto_offset when auto_offset=False or sub_offset_ms != 0
    ja = tmp_path / "Ep.ja.srt"
    en = tmp_path / "Ep.en.srt"
    ja.write_text("1\n00:00:01,000 --> 00:00:03,000\nテスト\n", encoding="utf-8")
    en.write_text("1\n00:00:01,000 --> 00:00:03,000\nTest\n", encoding="utf-8")
    p = TranslationPipeline(
        dry_run=True,
        enable_cascade=False,
        enable_rag=False,
        output_dir=tmp_path,
        cache_db_path=tmp_path / "p_pipe.db",
    )
    with patch("subtrans.pipeline.detect_auto_offset") as mock_det, \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=None):
        p.run(ja_path=ja, en_path=en, auto_offset=False)
        p.run(ja_path=ja, en_path=en, auto_offset=True, sub_offset_ms=500)
    assert mock_det.call_count == 0

    # Whisper snap_whisper_cues_to_en keeps close orphan (min_dist <= 600ms) and drops distant orphan (> 600ms)
    close_cue = Cue(id=0, start=1300, end=2200, ja="近い！")
    far_cue = Cue(id=1, start=8000, end=8500, ja="遠い！")
    en_ref = [{"id": 0, "start": 1000, "end": 1100, "duration": 100, "text": "Near"}]
    with patch("subtrans.dtw_auto_align.get_dtw_alignment_plan", return_value=({}, {})):
        parser.snap_whisper_cues_to_en([close_cue, far_cue], en_ref)
    assert close_cue.is_dropped_sound is False
    assert far_cue.is_dropped_sound is True

    # Parser: 3-way tie-break strictly increasing + close JA orphan kept (min_dist <= 600)
    ja_3 = tmp_path / "t3.ja.srt"
    en_3 = tmp_path / "t3.en.srt"
    ja_3.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nA\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nB\n\n"
        "3\n00:00:01,000 --> 00:00:02,000\nC\n",
        encoding="utf-8",
    )
    en_3.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nA\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nB\n\n"
        "3\n00:00:01,000 --> 00:00:02,000\nC\n",
        encoding="utf-8",
    )
    cues_3, _, _ = parser.parse_and_align_subtitles(str(ja_3), str(en_3))
    starts = [c.start for c in cues_3 if not c.is_skipped]
    assert starts[0] < starts[1] < starts[2]

    # Writer: sound_cues="translate" preserves [thở dài], Dual Dialogue shift <= 400ms, line_max_chars applied
    c_sound = Cue(id=1, start=1000, end=2000, ja="ため息", vi="[thở dài] Được rồi")
    cleaned = writer.clean_and_normalize_vietnamese(c_sound.vi, sound_cues="translate")
    assert "[thở dài]" in cleaned

    c1 = Cue(id=1, start=1000, end=3000, ja="A", vi="Xin chào")
    c2 = Cue(id=2, start=1050, end=3100, ja="B", vi="Chào bạn")
    opt = writer.optimize_cues_for_playback([c1, c2])
    if len(opt) == 2:
        assert abs(opt[1].start - 1050) <= 400

    out_srt = tmp_path / "wrap.srt"
    c_wrap = Cue(id=1, start=1000, end=4000, ja="長い", vi="Đây là một câu thoại rất dài để kiểm tra giới hạn ngắt dòng theo tham số line_max_chars")
    writer.write_subtitles([c_wrap], pysubs2.SSAFile(), out_srt, line_max_chars=30, smart_wrap=True)
    srt_lines = out_srt.read_text(encoding="utf-8").splitlines()
    text_lines = [l for l in srt_lines if l and not l.isdigit() and "-->" not in l]
    assert len(text_lines) == 2

    # Cascade: accepted_identical keeps uncertain=True if linter ERROR remains (and via run_cascade_repair_pass)
    c_err = Cue(id=1, start=1000, end=2000, ja="テスト", vi="テスト")
    c_err.uncertain = True
    accepted, reason, _ = cascade.accept_or_reject_repaired_cue(c_err, "テスト")
    assert accepted is True
    assert reason == "accepted_identical"

    mock_repair_tr = MagicMock()
    mock_repair_tr.model = "gemini-3.8-flash"
    mock_repair_tr.api_keys = ["AIzaTestKey1"]
    mock_repair_tr.key_ids = ["kid1"]
    mock_repair_tr.current_key_idx = 0
    mock_repair_tr.http_options = None
    mock_repair_tr.cache = TranslationCache(db_path=tmp_path / "cas_id.db")
    mock_repair_tr.dry_run = False
    ok_cas_resp = MagicMock()
    ok_cas_resp.text = '[{"id": 1, "vi": "テスト", "changed": false, "change_reason": "same"}]'
    ok_cas_resp.usage_metadata = None
    mock_repair_tr.client.models.generate_content.return_value = ok_cas_resp
    cascade.run_cascade_repair_pass([c_err], repair_translator=mock_repair_tr, difficulty_threshold=0.0)
    assert c_err.uncertain is True

    # Cascade & cast_sheet 429 PerDay single log + 429 PerMinute rotation
    c_rot = Cue(id=2, start=2500, end=4000, ja="テスト２", vi="テスト２")
    c_rot.uncertain = True
    rot_cache = TranslationCache(db_path=tmp_path / "cas_rot.db")
    mock_rot_client = MagicMock()
    ok_rot_resp = MagicMock()
    ok_rot_resp.text = '[{"id": 2, "vi": "Thu nghiem 2", "changed": true, "change_reason": "fixed"}]'
    ok_rot_resp.usage_metadata = None
    mock_rot_client.models.generate_content.side_effect = [
        Exception("429 RESOURCE_EXHAUSTED: PerMinute rate limit"),
        ok_rot_resp,
    ]
    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_rot_client), \
         patch("subtrans.throttle.time.sleep", return_value=None):
        rot_tr = GeminiTranslator(
            api_keys=["AIzaCasKeyA", "AIzaCasKeyB"],
            model="gemini-3.8-flash",
            cache=rot_cache,
            round_robin=True,
        )
        res_cas = cascade.run_cascade_repair_pass([c_rot], repair_translator=rot_tr, difficulty_threshold=0.0)
    assert res_cas["repaired_count"] == 1

    # Single exhausted_429_perday log when cast_sheet encounters 429 PerDay
    cs_cache = TranslationCache(db_path=tmp_path / "cs_log.db")
    mock_cs_client = MagicMock()
    mock_cs_client.models.generate_content.side_effect = Exception(
        "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier quotaValue: 20"
    )
    with patch("subtrans.cast_sheet.genai.Client", return_value=mock_cs_client), \
         patch("subtrans.throttle.time.sleep", return_value=None):
        cast_sheet.generate_cast_sheet(
            [Cue(id=1, start=0, end=1000, ja="フラン", en="Fran")],
            output_path=tmp_path / "cs_fail.cast.yaml",
            gemini_model="gemini-3.8-flash",
            cache=cs_cache,
            api_keys=["AIzaCastOnlyKey"],
        )
    with cs_cache._connect() as conn:
        cs_rows = conn.execute(
            "SELECT status FROM api_requests WHERE status = 'exhausted_429_perday'"
        ).fetchall()
    assert len(cs_rows) == 1

    # Audio diarization cached DSP constants & sliding window framing
    win1, mask1 = audio_diarization._get_dsp_constants(8000)
    win2, mask2 = audio_diarization._get_dsp_constants(8000)
    assert win1 is win2 and len(mask1) == 12
    t_arr = np.linspace(0, 0.5, 4000, dtype=np.float32)
    pcm = (0.5 * np.sin(2 * np.pi * 180.0 * t_arr) + 0.3 * np.sin(2 * np.pi * 450.0 * t_arr)).astype(np.float32)
    sig = audio_diarization.compute_voice_signature(pcm, 8000)
    assert sig is not None and sig.timbre.shape == (12,)

    # Report: next fallback is NOT self-referential and no Ollama text
    rep_path = tmp_path / "ep.audit.md"
    generate_qa_report(
        cues=[Cue(id=1, start=0, end=1000, ja="a", vi="a", model_used="gemini-3.5-flash-lite")],
        ja_path=ja,
        en_path=None,
        output_subtitle_path=out_srt,
        report_path=rep_path,
        elapsed_time_sec=1.0,
        stats={
            "provenance": {"primary_model": "gemini-3.5-flash-lite"},
            "pacific_time_requests": {"gemini-3.5-flash-lite": 498},
        },
        cache=TranslationCache(db_path=tmp_path / "rc.db"),
    )
    rep_txt = rep_path.read_text(encoding="utf-8")
    assert "Local Ollama" not in rep_txt
    assert "chuyển sang `gemini-3.8-flash`" in rep_txt
    assert "Không giới hạn (Local)" not in rep_txt

    # cli.models reads chain from models.yaml
    import re
    runner = CliRunner()
    monkeypatch.setattr(st_config, "get_gemini_api_key", lambda: None)
    with patch("subtrans.cli.get_gemini_api_key", return_value=None):
        res_m = runner.invoke(subtrans_cli_app, ["models"], env={"COLUMNS": "240", "NO_COLOR": "1"})
    assert res_m.exit_code == 0
    clean_models_out = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", res_m.output).split())
    assert "gemini-3.5-flash-lite -> gemini-3.8-flash" in clean_models_out
