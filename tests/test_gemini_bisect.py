# -*- coding: utf-8 -*-
"""
tests/test_gemini_bisect.py
Unit test verifying batch bisecting (splitting in half) on validation/ID mismatch,
and verifying that no real API calls are made under dry_run.
"""

import unittest
from subtrans.models import Cue
from subtrans.gemini_backend import GeminiTranslator


class TestGeminiBisect(unittest.TestCase):
    def test_dry_run_no_api_calls(self):
        translator = GeminiTranslator(dry_run=True, model="gemini-3.8-flash")
        cues = [
            Cue(id=1, start=0, end=1000, ja="こんにちは", en="Hello"),
            Cue(id=2, start=1000, end=2000, ja="さようなら", en="Goodbye"),
        ]
        results = translator.translate_batch(cues, context=[])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "[DRY-RUN VI] Hello")
        self.assertEqual(results[1], "[DRY-RUN VI] Goodbye")
        self.assertEqual(translator.stats["total_batches"], 1)
        self.assertEqual(translator.stats["split_batches"], 0)

    def test_bisect_on_validation_failure(self):
        """
        Simulates a batch of 4 cues where the first call returns an invalid count (e.g. only 2 items),
        triggering bisecting into 2 cues + 2 cues.
        """
        translator = GeminiTranslator(dry_run=False, api_key="dummy_key", model="gemini-3.8-flash")
        
        call_count = 0
        def mock_call_api(prompt_text, system_prompt=None):
            nonlocal call_count
            call_count += 1
            import json
            payload = json.loads(prompt_text[prompt_text.find("{"):])
            target_ids = [c["id"] for c in payload.get("cues_to_translate", [])]

            # If batch has 4 cues (first attempt), return invalid count (only 1 item)
            if target_ids == [1, 2, 3, 4]:
                return [{"id": 1, "vi": "Dịch lỗi thiếu câu"}]
            # If left half (1, 2)
            elif target_ids == [1, 2]:
                return [
                    {"id": 1, "vi": "Xin chào"},
                    {"id": 2, "vi": "Tạm biệt"}
                ]
            # If right half (3, 4)
            elif target_ids == [3, 4]:
                return [
                    {"id": 3, "vi": "Cảm ơn"},
                    {"id": 4, "vi": "Không có chi"}
                ]
            return []

        translator._call_api = mock_call_api

        cues = [
            Cue(id=1, start=0, end=1000, ja="こんにちは", en="Hello"),
            Cue(id=2, start=1000, end=2000, ja="さようなら", en="Goodbye"),
            Cue(id=3, start=2000, end=3000, ja="ありがとう", en="Thank you"),
            Cue(id=4, start=3000, end=4000, ja="どういたしまして", en="You're welcome"),
        ]

        results = translator.translate_batch(cues, context=[])
        self.assertEqual(len(results), 4)
        self.assertEqual(results, ["Xin chào", "Tạm biệt", "Cảm ơn", "Không có chi"])
        self.assertEqual(translator.stats["split_batches"], 1)
        self.assertEqual(translator.stats["total_batches"], 3)  # 1 initial + 2 halves

    def test_bisect_preserves_context_bridge(self):
        """
        Verifies that when bisecting, the right half receives the left half's cues
        as history_readonly context with their translated vi_ref.
        """
        translator = GeminiTranslator(dry_run=False, api_key="dummy_key", model="gemini-3.8-flash")
        recorded_prompts = []

        def mock_call_api(prompt_text, system_prompt=None):
            recorded_prompts.append(prompt_text)
            import json
            payload = json.loads(prompt_text[prompt_text.find("{"):])
            target_ids = [c["id"] for c in payload.get("cues_to_translate", [])]

            if target_ids == [1, 2, 3, 4]:
                return [{"id": 1, "vi": "Bị thiếu"}]
            elif target_ids == [1, 2]:
                return [{"id": 1, "vi": "Chào bạn"}, {"id": 2, "vi": "Đi đâu đấy"}]
            elif target_ids == [3, 4]:
                return [{"id": 3, "vi": "Tôi đi chợ"}, {"id": 4, "vi": "Hẹn gặp lại"}]
            return []

        translator._call_api = mock_call_api

        cues = [
            Cue(id=1, start=0, end=1000, ja="こんにちは", en="Hello"),
            Cue(id=2, start=1000, end=2000, ja="どこへ", en="Where"),
            Cue(id=3, start=2000, end=3000, ja="市場へ", en="To market"),
            Cue(id=4, start=3000, end=4000, ja="またね", en="See ya"),
        ]

        results = translator.translate_batch(cues, context=[])
        self.assertEqual(results, ["Chào bạn", "Đi đâu đấy", "Tôi đi chợ", "Hẹn gặp lại"])
        # The prompt for the right half must contain surrounding_context bridging left half cues with vi_ref
        right_prompt = recorded_prompts[-1]
        self.assertIn("history_readonly", right_prompt)
        self.assertIn("vi_ref", right_prompt)
        self.assertIn("Đi đâu đấy", right_prompt)


if __name__ == "__main__":
    unittest.main()
