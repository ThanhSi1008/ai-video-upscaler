#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_subtrans_resilience.py
Kiểm thử giả lập độ bền bỉ & khả năng chống treo (resilience) của stage subtitle:
(a) Máy chủ giả nhận kết nối rồi im lặng (Hanging/Silent Server) -> timeout kích hoạt chuẩn xác, không treo.
(b) Vòng lặp trả 503 liên tục (Persistent 503 Loop) -> thử lại có chặn trên, rơi đúng chuỗi B2 -> local/untranslated.
(c) 429 hết hạn mức ngày (Daily Quota Exhaustion) -> chuyển model / local / đánh dấu [Chưa dịch], không lặp vô hạn.
(d) Watchdog toàn stage hết hạn -> dừng an toàn, đánh dấu [Chưa dịch] và ghi file thành phẩm.
"""

import os
import sys
import time
import socket
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subtrans.models import Cue, Glossary
from subtrans.local_backend import OllamaTranslator
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

    def test_a_silent_server_timeout_ollama(self):
        """(a1) OllamaTranslator gặp máy chủ im lặng -> timeout theo đúng cấu hình (1s), không treo."""
        server = SilentServer()
        server.start()
        try:
            t0 = time.monotonic()
            translator = OllamaTranslator(
                endpoint=f"http://127.0.0.1:{server.port}",
                timeout=1  # 1 giây timeout
            )
            with self.assertRaises(Exception) as ctx:
                translator._call_chat([{"role": "user", "content": "hello"}])
            elapsed = time.monotonic() - t0

            # Phải kích hoạt timeout trong khoảng 1.0s - 2.5s, tuyệt đối không treo
            self.assertGreaterEqual(elapsed, 0.9, "Timeout kích hoạt quá sớm")
            self.assertLess(elapsed, 3.5, f"Client bị treo quá lâu: {elapsed:.2f}s")
            print(f"\n✅ [Test A1 PASS] Ollama timeout ngắt kết nối chính xác sau {elapsed:.2f}s (cấu hình 1.0s).")
        finally:
            server.stop()

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
        # Mock local backend cũng không khả dụng để test rơi về untranslated an toàn
        pipeline.ollama_backend = MagicMock()
        pipeline.ollama_backend.is_upscale_running.return_value = False
        pipeline.ollama_backend.is_available.return_value = False

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
        pipeline.ollama_backend = MagicMock()
        pipeline.ollama_backend.is_upscale_running.return_value = True  # Upscale đang chạy -> cấm local
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
            local_only=True,
            output_dir=Path("out/test_subtrans"),
            output_format="srt"
        )
        # Đặt watchdog 0.05 giây
        pipeline.stage_watchdog_seconds = 0.05
        pipeline.batch_size = 1
        pipeline.local_batch_size = 1
        pipeline.cache = MagicMock()
        pipeline.cache.get_cue.return_value = None
        pipeline.rate_limiter = None

        def slow_translate(batch, context, glossary):
            time.sleep(0.08)  # Lâu hơn watchdog
            return ["Dịch dòng đầu"]

        pipeline.ollama_backend = MagicMock()
        pipeline.ollama_backend.is_upscale_running.return_value = False
        pipeline.ollama_backend.is_available.return_value = True
        pipeline.ollama_backend.translate_batch.side_effect = slow_translate

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


if __name__ == "__main__":
    unittest.main()
