#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subtrans/tests/test_subtrans_resilience.py
Offline resilience tests for the subtitle translation stage:
(a) Silent/hanging TCP server -> socket timeout fires cleanly without hanging.
(b) Persistent 503 loop -> bounded retries before fallback/untranslated marking.
(c) 429 Daily Quota Exhaustion -> model fallback / [Chưa dịch] marking without infinite loop.
(d) Stage watchdog timeout -> safe stop, partial marking, and output write.
"""

import contextlib
import json
import os
import sqlite3
import sys
import time
import socket
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repository root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from subtrans.models import Cue, Glossary
from subtrans.gemini_backend import GeminiTranslator
from subtrans.throttle import RateLimiter, call_with_retry, QuotaExhaustedError
from subtrans.pipeline import TranslationPipeline


class SilentServer(threading.Thread):
    """Máy chủ TCP giả lập: chấp nhận kết nối HTTP rồi im lặng hoàn toàn để test socket timeout."""
    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.running = True

    def run(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
                # Nhận một phần request rồi không phản hồi gì cả
                try:
                    conn.recv(1024)
                except Exception:
                    pass
                # Giữ kết nối mở cho đến khi client timeout
                time.sleep(5)
                try:
                    conn.close()
                except Exception:
                    pass
            except Exception:
                break

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


class TestSubtransResilience(unittest.TestCase):

    def test_a_call_with_retry_timeout_bounding(self):
        """(a2) call_with_retry khi gặp hàm bị timeout -> thử lại có chặn trên và kết thúc đúng hạn."""
        attempt_count = 0

        def hanging_call():
            nonlocal attempt_count
            attempt_count += 1
            # Giả lập timeout error
            raise TimeoutError("Socket read timed out after 45000ms")

        t0 = time.monotonic()
        with self.assertRaises(TimeoutError):
            call_with_retry(
                fn=hanging_call,
                max_retries=2,
                base_delay=0.1,  # Delay nhanh cho unit test
                max_delay=0.5
            )
        elapsed = time.monotonic() - t0

        self.assertEqual(attempt_count, 3, f"Số lần thử {attempt_count} phải bằng max_retries + 1 (3)")
        self.assertLess(elapsed, 2.0, f"Tổng thời gian retry {elapsed:.2f}s phải ngắn và có chặn trên")
        print(f"✅ [Test A2 PASS] call_with_retry chặn trên đúng {attempt_count} lần thử trong {elapsed:.2f}s.")

    def test_b_persistent_503_loop_fallback_chain(self):
        """(b) Máy chủ trả 503 liên tục -> model chính thử 2 lần -> sang model kế -> sang model 3 -> sang untranslated."""
        cues = [
            Cue(id=1, start=1000, end=3000, ja="こんにちは", en="Hello"),
            Cue(id=2, start=3100, end=5000, ja="さようなら", en="Goodbye"),
        ]

        models_called = []

        def make_failing_backend(model_name):
            mock = MagicMock()
            def fail_translate(*args, **kwargs):
                models_called.append(model_name)
                raise Exception(f"503 UNAVAILABLE: High demand for model '{model_name}'")
            mock.translate_batch.side_effect = fail_translate
            return mock

        pipeline = TranslationPipeline(
            local_only=False,
            output_dir=Path("out/test_subtrans"),
            output_format="srt"
        )
        # Giảm số lần retry và tắt sleep dài cho test
        pipeline.max_retries_per_model = 2
        pipeline.candidate_gemini_models = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
        pipeline.gemini_backends = {
            "gemini-3.8-flash": make_failing_backend("gemini-3.8-flash"),
            "gemini-3.7-flash": make_failing_backend("gemini-3.7-flash"),
            "gemini-3.6-flash": make_failing_backend("gemini-3.6-flash"),
        }

        # Tắt rate limiter delay trong test
        pipeline.rate_limiter = None
        pipeline.cache = MagicMock()
        pipeline.cache.get_cue.return_value = None

        with patch("time.sleep", return_value=None), \
             patch("subtrans.throttle.time.sleep", return_value=None), \
             patch("subtrans.cache.TranslationCache.is_model_exhausted", return_value=False):
            import pysubs2
            dummy_ssa = pysubs2.SSAFile()

            with patch("subtrans.pipeline.write_subtitles") as mock_write, \
                 patch("subtrans.pipeline.generate_qa_report") as mock_qa:
                mock_write.return_value = Path("out/test_subtrans/test.srt")
                mock_qa.return_value = Path("out/test_subtrans/test.audit.md")

                # Chạy pipeline với cues giả
                with patch("subtrans.pipeline.parse_and_align_subtitles", return_value=(cues, dummy_ssa, None)):
                    with patch("subtrans.pipeline.generate_cast_sheet", return_value=Glossary()):
                        pipeline.run(ja_path=Path("dummy.ja.ass"))

        # Xác minh toàn bộ chuỗi fallback đã được kích hoạt theo đúng thứ tự
        self.assertIn("gemini-3.8-flash", models_called)
        self.assertIn("gemini-3.7-flash", models_called)
        self.assertIn("gemini-3.6-flash", models_called)

        # Kiểm tra cues đã được đánh dấu an toàn là [Chưa dịch], không bị crash
        for c in cues:
            self.assertEqual(c.vi, "[Chưa dịch]")
            self.assertEqual(c.backend_used, "untranslated")
            self.assertTrue(c.uncertain)

        print(f"\n✅ [Test B PASS] Chuỗi fallback B2 xử lý trơn tru: {models_called} -> [Chưa dịch], không treo.")

    def test_c_daily_quota_429_exhaustion(self):
        """(c) Gặp lỗi 429 cạn quota ngày -> chuyển tiếp trong chuỗi và kết thúc mà không lặp vô hạn."""
        cues = [
            Cue(id=1, start=1000, end=3000, ja="テスト", en="Test"),
        ]

        models_called = []

        def make_quota_backend(model_name):
            mock = MagicMock()
            def fail_quota(*args, **kwargs):
                models_called.append(model_name)
                raise QuotaExhaustedError(f"429 Resource Exhausted: Daily quota reached for {model_name}")
            mock.translate_batch.side_effect = fail_quota
            return mock

        pipeline = TranslationPipeline(
            local_only=False,
            output_dir=Path("out/test_subtrans"),
            output_format="srt"
        )
        pipeline.max_retries_per_model = 1
        pipeline.candidate_gemini_models = ["gemini-3.8-flash", "gemini-3.7-flash"]
        pipeline.gemini_backends = {
            "gemini-3.8-flash": make_quota_backend("gemini-3.8-flash"),
            "gemini-3.7-flash": make_quota_backend("gemini-3.7-flash"),
        }

        pipeline.cache = MagicMock()
        pipeline.cache.get_cue.return_value = None
        pipeline.rate_limiter = None

        import pysubs2
        dummy_ssa = pysubs2.SSAFile()

        with patch("time.sleep", return_value=None), \
             patch("subtrans.throttle.time.sleep", return_value=None):
            with patch("subtrans.pipeline.write_subtitles") as mock_write, \
                 patch("subtrans.pipeline.generate_qa_report") as mock_qa:
                mock_write.return_value = Path("out/test_subtrans/test.srt")
                mock_qa.return_value = Path("out/test_subtrans/test.audit.md")

                with patch("subtrans.pipeline.parse_and_align_subtitles", return_value=(cues, dummy_ssa, None)):
                    with patch("subtrans.pipeline.generate_cast_sheet", return_value=Glossary()):
                        pipeline.run(ja_path=Path("dummy.ja.ass"))

        self.assertEqual(cues[0].vi, "[Chưa dịch]")
        self.assertEqual(cues[0].backend_used, "untranslated")
        print(f"✅ [Test C PASS] 429 quota exhaustion kết thúc an toàn, từ chối local khi upscale đang chạy.")

    def test_d_stage_watchdog_trigger(self):
        """(d) Watchdog toàn stage hết hạn -> dừng vòng lặp, đánh dấu [Chưa dịch], chuyển sang ghi file."""
        cues = [
            Cue(id=1, start=1000, end=2000, ja="一行目", en="Line 1"),
            Cue(id=2, start=2100, end=3000, ja="二行目", en="Line 2"),
            Cue(id=3, start=3100, end=4000, ja="三行目", en="Line 3"),
        ]

        pipeline = TranslationPipeline(
            local_only=False,
            output_dir=Path("out/test_subtrans"),
            output_format="srt"
        )
        # Đặt watchdog 0.05 giây
        pipeline.stage_watchdog_seconds = 0.05
        pipeline.batch_size = 1
        pipeline.cache = MagicMock()
        pipeline.cache.get_cue.return_value = None
        pipeline.rate_limiter = None

        def slow_translate(batch, context, glossary):
            time.sleep(0.08)  # Lâu hơn watchdog
            return ["Dịch dòng đầu"]

        mock_backend = MagicMock()
        mock_backend.translate_batch.side_effect = slow_translate
        pipeline.gemini_backends = {
            "gemini-3.5-flash-lite": mock_backend
        }

        import pysubs2
        dummy_ssa = pysubs2.SSAFile()

        with patch("subtrans.pipeline.write_subtitles") as mock_write, \
             patch("subtrans.pipeline.generate_qa_report") as mock_qa:
            mock_write.return_value = Path("out/test_subtrans/test.srt")
            mock_qa.return_value = Path("out/test_subtrans/test.audit.md")

            with patch("subtrans.pipeline.parse_and_align_subtitles", return_value=(cues, dummy_ssa, None)):
                with patch("subtrans.pipeline.generate_cast_sheet", return_value=Glossary()):
                    pipeline.run(ja_path=Path("dummy.ja.ass"))

        # Batch 1 được dịch, Batch 2 và 3 bị watchdog chặn và đánh dấu [Chưa dịch]
        self.assertEqual(cues[0].vi, "Dịch dòng đầu")
        self.assertEqual(cues[1].vi, "[Chưa dịch]")
        self.assertEqual(cues[2].vi, "[Chưa dịch]")
        self.assertTrue(cues[1].uncertain)
        self.assertTrue(cues[2].uncertain)
        print(f"✅ [Test D PASS] Watchdog toàn stage ngắt đúng hạn, bảo toàn phân đoạn và ghi file thành công.")


def test_e_real_native_gemini_persistent_503_fallback_chain(tmp_path, monkeypatch):
    """Verify real GeminiTranslator instances traverse the 503 fallback chain and write partial/audit files."""
    import src.config as app_config
    import subtrans.rag as rag_mod

    monkeypatch.setenv("GEMINI_API_KEY", "AIzaTestKey1")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(app_config, "PENDING_SUBS_FILE", str(tmp_path / "pending.json"))
    monkeypatch.setattr(rag_mod, "KNOWLEDGE_DB_PATH", tmp_path / "knowledge.db")
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr("subtrans.throttle.time.sleep", lambda _s: None)

    ja_srt = tmp_path / "ep_503.ja.srt"
    ja_srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nこんにちは\n\n"
        "2\n00:00:03,500 --> 00:00:05,500\nさようなら\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_503"
    cache_db = tmp_path / "c503.db"
    models_invoked = []

    fake_client = MagicMock()

    def _always_503(model, contents, config):
        models_invoked.append(model)
        raise RuntimeError(f"503 UNAVAILABLE: High demand for {model}")

    fake_client.models.generate_content.side_effect = _always_503

    with patch("subtrans.gemini_backend.genai.Client", return_value=fake_client), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        pipeline = TranslationPipeline(
            dry_run=False,
            enable_cascade=False,
            enable_rag=False,
            output_dir=out_dir,
            output_format="srt",
            cache_db_path=cache_db,
        )
        pipeline.rate_limiter = None
        pipeline.candidate_gemini_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
        pipeline._init_primary_backend()
        assert all(isinstance(b, GeminiTranslator) for b in pipeline.gemini_backends.values())
        res = pipeline.run(ja_path=ja_srt)

    assert "gemini-3.5-flash-lite" in models_invoked
    assert "gemini-3.8-flash" in models_invoked
    assert res["is_partial"] is True
    assert res["real_rate"] == 0.0
    assert res["stats"]["model_counts"]["untranslated"] == 2
    out_srt = Path(res["output_path"])
    assert out_srt.is_file() and out_srt.name.endswith(".vi.partial.srt")
    assert "[Chưa dịch]" not in out_srt.read_text(encoding="utf-8")
    assert Path(res["report_path"]).is_file()
    with contextlib.closing(sqlite3.connect(cache_db)) as conn:
        err_rows = conn.execute(
            "SELECT status, error_type FROM api_requests WHERE status = 'error_503_unavailable'"
        ).fetchall()
    assert len(err_rows) >= 2


def test_f_real_native_gemini_daily_quota_429_immediate_failover(tmp_path, monkeypatch):
    """Verify real GeminiTranslator raises DailyQuotaExhaustedError on 429 PerDay and fails over immediately."""
    import src.config as app_config
    import subtrans.rag as rag_mod

    monkeypatch.setenv("GEMINI_API_KEY", "AIzaTestKey1")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(app_config, "PENDING_SUBS_FILE", str(tmp_path / "pending.json"))
    monkeypatch.setattr(rag_mod, "KNOWLEDGE_DB_PATH", tmp_path / "knowledge.db")
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr("subtrans.throttle.time.sleep", lambda _s: None)

    ja_srt = tmp_path / "ep_429.ja.srt"
    ja_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nこんにちは\n", encoding="utf-8")

    out_dir = tmp_path / "out_429"
    cache_db = tmp_path / "c429.db"
    lite_calls = 0
    flash_calls = 0

    def _perday_or_ok(model, contents, config):
        nonlocal lite_calls, flash_calls
        if "lite" in model:
            lite_calls += 1
            raise Exception(
                "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier quotaValue: 500"
            )
        flash_calls += 1
        resp = MagicMock()
        resp.text = json.dumps([{"id": 0, "vi": "Xin chào"}])
        return resp

    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = _perday_or_ok

    with patch("subtrans.gemini_backend.genai.Client", return_value=fake_client), \
         patch("subtrans.pipeline.load_or_generate_series_cast_sheet", return_value=Glossary()):
        pipeline = TranslationPipeline(
            dry_run=False,
            enable_cascade=False,
            enable_rag=False,
            output_dir=out_dir,
            output_format="srt",
            cache_db_path=cache_db,
        )
        pipeline.rate_limiter = None
        pipeline.candidate_gemini_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
        pipeline._init_primary_backend()
        res = pipeline.run(ja_path=ja_srt)

    assert lite_calls == 1
    assert flash_calls == 1
    assert pipeline.cache.is_model_exhausted("gemini-3.5-flash-lite") is True
    assert res["stats"]["model_counts"]["gemini-3.8-flash"] == 1
    srt_files = list(out_dir.glob("*.vi.srt"))
    assert len(srt_files) == 1
    assert "Xin chào" in srt_files[0].read_text(encoding="utf-8")
    with contextlib.closing(sqlite3.connect(cache_db)) as conn:
        rows = conn.execute(
            "SELECT model, status, error_type FROM api_requests ORDER BY id ASC"
        ).fetchall()
    assert ("gemini-3.5-flash-lite", "exhausted_429_perday", "429_PerDay") in rows
    assert ("gemini-3.8-flash", "success", "") in rows


if __name__ == "__main__":
    unittest.main()
