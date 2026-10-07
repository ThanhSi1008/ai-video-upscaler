# -*- coding: utf-8 -*-
"""
tests/test_key_rotation_stub.py
Unit tests using mock/stub (NO real API calls) to verify:
1. 429 PerDay rotates to next key within the same model and marks key exhausted until 00:00 PT.
2. 429 PerMinute waits retryDelay and does NOT rotate key.
3. 503 retries up to 3 times on the same key, then raises to fall back to next model (does NOT rotate key).
4. All 4 keys exhausted raises DailyQuotaExhaustedError to fall back to next model.
5. Midnight PT rollover resets key exhaustion status.
"""

from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path
import datetime
import zoneinfo

from subtrans.models import Cue
from subtrans.gemini_backend import GeminiTranslator
from subtrans.cache import TranslationCache, compute_key_id, PT_TZ
from subtrans.throttle import DailyQuotaExhaustedError


@pytest.fixture
def temp_cache(tmp_path):
    db_file = tmp_path / "test_translations.db"
    cache = TranslationCache(db_path=db_file)
    return cache


def test_429_per_day_rotates_key(temp_cache):
    """(1) 429 PerDay must mark key exhausted until 00:00 PT and rotate to next key of the SAME model."""
    keys = ["KEY_A_111111111", "KEY_B_222222222", "KEY_C_333333333", "KEY_D_444444444"]
    key_ids = [compute_key_id(k) for k in keys]

    translator = GeminiTranslator(
        api_keys=keys,
        model="gemini-3.5-flash-lite",
        dry_run=False
    )
    translator.cache = temp_cache

    # Key 0 raises 429 PerDay
    err_429_per_day = Exception(
        '429 ResourceExhausted: Quota exceeded for quota metric "RequestsPerDayPerProject" and limit "500"'
    )

    success_resp = MagicMock()
    success_resp.text = '[{"id": 1, "vi": "Xin chào"}]'

    # Mock genai.Client
    mock_cli_0 = MagicMock()
    mock_cli_0.models.generate_content.side_effect = err_429_per_day

    mock_cli_1 = MagicMock()
    mock_cli_1.models.generate_content.return_value = success_resp

    def get_client(api_key=None, **kwargs):
        if api_key == keys[0]:
            return mock_cli_0
        return mock_cli_1

    with patch("google.genai.Client", side_effect=get_client):
        translator.client = mock_cli_0

        cues = [Cue(id=1, start=0, end=1000, ja="こんにちは")]
        res = translator.translate_batch(cues, [])

        assert res == ["Xin chào"]
        # Verify key_0 is marked exhausted in cache
        assert temp_cache.is_key_model_exhausted(key_ids[0], "gemini-3.5-flash-lite") is True
        # Verify key_1 is now the current key and was NOT exhausted
        assert translator.current_key_idx == 1
        assert temp_cache.is_key_model_exhausted(key_ids[1], "gemini-3.5-flash-lite") is False
        # Verify rotation event was logged
        assert len(translator.rotation_events) >= 1
        assert translator.rotation_events[-1]["from_key_id"] == key_ids[0]
        assert translator.rotation_events[-1]["to_key_id"] == key_ids[1]


def test_429_per_minute_waits_and_does_not_rotate(temp_cache):
    """(2) 429 PerMinute must wait retryDelay and NOT rotate key."""
    keys = ["KEY_A_111111111", "KEY_B_222222222"]
    key_ids = [compute_key_id(k) for k in keys]

    translator = GeminiTranslator(
        api_keys=keys,
        model="gemini-3.5-flash-lite",
        dry_run=False
    )
    translator.cache = temp_cache

    err_429_rpm = Exception(
        '429 ResourceExhausted: Quota exceeded for quota metric "RequestsPerMinutePerProject" retryDelay: "0.01s"'
    )
    success_resp = MagicMock()
    success_resp.text = '[{"id": 1, "vi": "Đã chờ xong"}]'

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [err_429_rpm, success_resp]
    translator.client = mock_client

    with patch("time.sleep") as mock_sleep:
        cues = [Cue(id=1, start=0, end=1000, ja="こんにちは")]
        res = translator.translate_batch(cues, [])

        assert res == ["Đã chờ xong"]
        # Must NOT rotate key
        assert translator.current_key_idx == 0
        assert temp_cache.is_key_model_exhausted(key_ids[0], "gemini-3.5-flash-lite") is False
        assert len(translator.rotation_events) == 0
        mock_sleep.assert_called()


def test_503_retries_3_times_then_raises_without_rotating(temp_cache):
    """(3) 503 must retry up to 3 times on the SAME key with backoff, then raise so pipeline falls back to next model."""
    keys = ["KEY_A_111111111", "KEY_B_222222222"]
    key_ids = [compute_key_id(k) for k in keys]

    translator = GeminiTranslator(
        api_keys=keys,
        model="gemini-3.5-flash-lite",
        dry_run=False
    )
    translator.cache = temp_cache

    err_503 = Exception("503 The service is temporarily unavailable.")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [err_503, err_503, err_503, err_503]
    translator.client = mock_client

    with patch("time.sleep"):
        cues = [Cue(id=1, start=0, end=1000, ja="こんにちは")]
        # translate_batch catches and splits, on repeated 503 it falls back to EN/JA or raises
        # But directly testing _call_api verifies the exact 3 retries and error propagation
        with pytest.raises(Exception) as exc_info:
            translator._call_api("prompt")

        assert "503" in str(exc_info.value)
        # Verify key was NOT rotated
        assert translator.current_key_idx == 0
        assert temp_cache.is_key_model_exhausted(key_ids[0], "gemini-3.5-flash-lite") is False
        assert len(translator.rotation_events) == 0
        # Verify it attempted 4 times total (1 initial + 3 retries)
        assert mock_client.models.generate_content.call_count == 4


def test_all_four_keys_exhausted_raises_daily_quota(temp_cache):
    """(4) When all 4 keys exhaust daily quota, translator raises DailyQuotaExhaustedError to move to next model."""
    keys = ["KEY_1", "KEY_2", "KEY_3", "KEY_4"]
    key_ids = [compute_key_id(k) for k in keys]

    translator = GeminiTranslator(
        api_keys=keys,
        model="gemini-3.5-flash-lite",
        dry_run=False
    )
    translator.cache = temp_cache

    err_429_rpd = Exception('429 ResourceExhausted: RequestsPerDay exceeded')

    with patch("google.genai.Client") as MockClient:
        mock_cli = MagicMock()
        mock_cli.models.generate_content.side_effect = err_429_rpd
        MockClient.return_value = mock_cli
        translator.client = mock_cli

        with pytest.raises(DailyQuotaExhaustedError) as exc_info:
            translator._call_api("prompt")

        assert "gemini-3.5-flash-lite" in exc_info.value.model
        # All 4 keys should now be marked exhausted
        for kid in key_ids:
            assert temp_cache.is_key_model_exhausted(kid, "gemini-3.5-flash-lite") is True


def test_midnight_pt_rollover_resets_exhaustion(temp_cache):
    """(5) Key exhaustion must be bound to PT date; crossing midnight PT resets the exhausted state."""
    key_id = "test_key_abc"
    model = "gemini-3.5-flash-lite"

    # Mark exhausted on day 1 (PT)
    day1_str = "2026-10-06"
    day2_str = "2026-10-07"

    with patch("subtrans.cache.get_today_pt_str", return_value=day1_str):
        temp_cache.mark_key_model_exhausted(
            key_id=key_id,
            model=model,
            quota_id="PerDay",
            reason="Exhausted on day 1"
        )
        # On day 1, it is exhausted
        assert temp_cache.is_key_model_exhausted(key_id, model) is True

    # On day 2 (after midnight PT), querying with day2_str must NOT be exhausted
    with patch("subtrans.cache.get_today_pt_str", return_value=day2_str):
        assert temp_cache.is_key_model_exhausted(key_id, model) is False
