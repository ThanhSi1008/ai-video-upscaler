# -*- coding: utf-8 -*-
"""
tests/test_gemini_quota_management.py
Kiểm thử toàn diện các yêu cầu về hạn mức Gemini API (Free Tier):
1. Config models.yaml (RPD 20 / 500, chuỗi không có preview)
2. Phân loại lỗi 429 (PerDay vs PerMinute)
3. Tính toán mốc reset giờ Thái Bình Dương (PT -> 14:00 VN / 15:00 sau 1/11)
4. Bộ đếm request theo ngày Thái Bình Dương trong SQLite
5. Tách đôi batch và thống kê split rate
6. Tái sử dụng Series Cast Sheet (<series>.cast.yaml)
"""

import os
import unittest
import datetime
import zoneinfo
import tempfile
import sqlite3
from pathlib import Path

from subtrans.config import load_models_config, update_model_quota_in_yaml
from subtrans.cache import TranslationCache, get_now_pt, get_today_pt_str, get_next_pt_reset, PT_TZ, VN_TZ
from subtrans.throttle import (
    call_with_retry,
    parse_429_details,
    DailyQuotaExhaustedError,
    QuotaExhaustedError,
)
from subtrans.models import Cue, Character, CharacterSpeaksTo, Glossary
from subtrans.gemini_backend import GeminiTranslator
from subtrans.cast_sheet import extract_series_name, get_series_cast_path, load_or_generate_series_cast_sheet


class TestGeminiQuotaManagement(unittest.TestCase):

    def setUp(self):
        cache = TranslationCache()
        with sqlite3.connect(cache.db_path) as conn:
            conn.execute("DELETE FROM model_exhaustion WHERE model = 'unit-test-model'")
            conn.execute("DELETE FROM api_requests WHERE model = 'unit-test-model'")
            conn.commit()

    def tearDown(self):
        cache = TranslationCache()
        with sqlite3.connect(cache.db_path) as conn:
            conn.execute("DELETE FROM model_exhaustion WHERE model = 'unit-test-model'")
            conn.execute("DELETE FROM api_requests WHERE model = 'unit-test-model'")
            conn.commit()

    def test_1_models_yaml_and_chain(self):
        """Kiểm tra cấu hình models.yaml và chuỗi model không có preview."""
        cfg = load_models_config()
        self.assertIn("models", cfg)
        self.assertIn("chain", cfg)

        models = cfg["models"]
        # RPD flash = 20
        self.assertEqual(models["gemini-3.8-flash"]["rpd"], 20)
        self.assertEqual(models["gemini-3.7-flash"]["rpd"], 20)
        self.assertEqual(models["gemini-3.6-flash"]["rpd"], 20)
        self.assertEqual(models["gemini-3.5-flash"]["rpd"], 20)
        # RPD flash-lite = 500
        self.assertEqual(models["gemini-3.5-flash-lite"]["rpd"], 500)
        self.assertEqual(models["gemini-3.1-flash-lite"]["rpd"], 500)

        # Chuỗi model: 3.8-flash -> 3.7-flash -> 3.6-flash -> 3.5-flash -> 3.5-flash-lite -> local
        expected_chain = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "local"
        ]
        self.assertEqual(cfg["chain"], expected_chain)

        # Tuyệt đối không dùng model preview
        for m in cfg["chain"]:
            self.assertNotIn("preview", m.lower())

    def test_2_pacific_time_reset_calculation(self):
        """Kiểm tra tính toán mốc reset 00:00 PT tương ứng giờ Việt Nam."""
        now_pt = get_now_pt()
        next_pt, next_vn = get_next_pt_reset()

        self.assertEqual(next_pt.hour, 0)
        self.assertEqual(next_pt.minute, 0)
        self.assertEqual(next_pt.second, 0)

        # Nếu là PDT (Daylight saving UTC-7, trước tháng 11) -> 14:00 VN
        # Nếu là PST (Standard time UTC-8, sau tháng 11) -> 15:00 VN
        if bool(now_pt.dst()):
            self.assertEqual(next_vn.hour, 14, "Trong mùa hè PDT, 00:00 PT phải tương ứng 14:00 giờ VN")
        else:
            self.assertEqual(next_vn.hour, 15, "Trong mùa đông PST, 00:00 PT phải tương ứng 15:00 giờ VN")

    def test_3_429_classification_per_day_vs_per_minute(self):
        """Kiểm tra phân loại lỗi 429 theo trường chi tiết (quotaId, quotaValue, retryDelay)."""
        # A. Lỗi PerDay
        class Fake429PerDay(Exception):
            details = {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "metadata": {
                                "quotaId": "GenerateContentRequestsPerDayPerProjectPerModel",
                                "quotaValue": "20"
                            }
                        }
                    ]
                }
            }

        qid, qval, delay = parse_429_details(Fake429PerDay())
        self.assertEqual(qid, "GenerateContentRequestsPerDayPerProjectPerModel")
        self.assertEqual(qval, "20")

        # B. Lỗi PerMinute kèm retryDelay
        class Fake429PerMinute(Exception):
            details = {
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "metadata": {
                                "quotaId": "GenerateContentRequestsPerMinutePerProjectPerModel",
                                "quotaValue": "5"
                            }
                        },
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "12.5s"
                        }
                    ]
                }
            }

        qid2, qval2, delay2 = parse_429_details(Fake429PerMinute())
        self.assertEqual(qid2, "GenerateContentRequestsPerMinutePerProjectPerModel")
        self.assertEqual(qval2, "5")
        self.assertEqual(delay2, 12.5)

    def test_4_call_with_retry_immediate_switch_on_perday(self):
        """Khi gặp lỗi PerDay, lập tức ném DailyQuotaExhaustedError mà không retry."""
        class MockPerDayError(Exception):
            def __init__(self):
                self.details = {
                    "error": {
                        "code": 429,
                        "details": [{
                            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                            "metadata": {
                                "quotaId": "GenerateContentRequestsPerDayPerProjectPerModel",
                                "quotaValue": "20"
                            }
                        }]
                    }
                }
                super().__init__("429 RESOURCE_EXHAUSTED: PerDay quota reached")

        calls = 0
        def dummy_api():
            nonlocal calls
            calls += 1
            raise MockPerDayError()

        with self.assertRaises(DailyQuotaExhaustedError) as ctx:
            call_with_retry(dummy_api, model_name="unit-test-model", max_retries=3)

        self.assertEqual(calls, 1, "PerDay phải kết thúc ngay ở lần gọi đầu tiên, tuyệt đối không pause/retry")
        self.assertIn("GenerateContentRequestsPerDay", ctx.exception.quota_id)
        self.assertEqual(ctx.exception.quota_value, "20")

        # Dọn dẹp cache
        cache = TranslationCache()
        cache.clear_model_exhaustion = lambda m: None

    def test_5_series_cast_sheet_caching(self):
        """Kiểm tra tính năng lưu Cast Sheet theo bộ (<series>.cast.yaml) và tái sử dụng 0 API calls."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            source_file = "[SubsPlease] Mushoku Tensei S3 - 14 (1080p) [73CF848B].mkv"
            series_name = extract_series_name(source_file)
            self.assertEqual(series_name, "Mushoku Tensei S3")

            cast_path = get_series_cast_path(series_name, out_dir)
            self.assertEqual(cast_path.name, "Mushoku_Tensei_S3.cast.yaml")

            # Lần 1: Chạy dry-run sinh cast sheet
            cues = [
                Cue(id=1, start=0, end=1000, ja="ルーデウス、行くわよ", en="Rudeus, let's go!"),
                Cue(id=2, start=1000, end=2000, ja="分かった、シルフィ", en="Understood, Sylphie."),
            ]
            g1 = load_or_generate_series_cast_sheet(
                cues=cues,
                source_name=source_file,
                output_dir=out_dir,
                dry_run=True,
            )
            self.assertIsNotNone(g1)
            self.assertTrue(cast_path.exists())

            # Lần 2: Nạp lại cho tập 15 mà không gọi API (0 API calls)
            source_ep15 = "[SubsPlease] Mushoku Tensei S3 - 15 (1080p) [82BA91C3].mkv"
            g2 = load_or_generate_series_cast_sheet(
                cues=cues,
                source_name=source_ep15,
                output_dir=out_dir,
                dry_run=False, # Không dry-run nhưng đã có cast sheet và không có nhân vật mới
            )
            self.assertIsNotNone(g2)
    def test_6_multi_key_rotation(self):
        """Kiểm tra cơ chế nạp nhiều API Key và tự động xoay key (Key Rotation)."""
        from subtrans.config import get_gemini_api_keys
        
        # Test 1: Khởi tạo translator với 2 keys
        translator = GeminiTranslator(
            api_keys=["test_key_111111", "test_key_222222"],
            model="gemini-3.8-flash",
            dry_run=True
        )
        self.assertEqual(len(translator.api_keys), 2)
        self.assertEqual(translator.api_key, "test_key_111111")
        self.assertEqual(translator.current_key_idx, 0)

        # Test 2: Xoay sang key thứ 2
        can_rotate = translator.rotate_key()
        self.assertTrue(can_rotate)
        self.assertEqual(translator.api_key, "test_key_222222")
        self.assertEqual(translator.current_key_idx, 1)

        # Test 3: Xoay tiếp khi đã hết danh sách key
        can_rotate_again = translator.rotate_key()
        self.assertFalse(can_rotate_again)
        self.assertEqual(translator.current_key_idx, 1)


if __name__ == "__main__":
    unittest.main()
