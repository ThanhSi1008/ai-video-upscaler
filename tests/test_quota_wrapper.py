# -*- coding: utf-8 -*-
"""
tests/test_quota_wrapper.py
Kiểm thử bộ đếm request theo ngày Thái Bình Dương và hàm bọc duy nhất execute_and_log_api_request.
"""

import os
import pytest
import sqlite3
import zoneinfo
from pathlib import Path
from unittest.mock import MagicMock
from subtrans.cache import (
    TranslationCache,
    execute_and_log_api_request,
    get_today_pt_str,
    get_now_pt,
    PT_TZ
)


def test_execute_and_log_api_request_success(tmp_path):
    db_file = tmp_path / "test_cache.db"
    cache = TranslationCache(db_path=db_file)

    def mock_api():
        return {"status": "ok"}

    res = execute_and_log_api_request(
        mock_api,
        model="gemini-3.8-flash",
        backend="gemini",
        cues_count=100,
        request_type="translation",
        cache=cache
    )
    assert res == {"status": "ok"}

    today_pt = get_today_pt_str()
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT model, status, pt_date FROM api_requests")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "gemini-3.8-flash"
        assert rows[0][1] == "success"
        assert rows[0][2] == today_pt


def test_execute_and_log_api_request_429_perday(tmp_path):
    db_file = tmp_path / "test_cache.db"
    cache = TranslationCache(db_path=db_file)

    class Mock429Error(Exception):
        pass

    def mock_failing_api():
        err = Mock429Error("429 RESOURCE_EXHAUSTED")
        err.details = [{
            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
            "metadata": {
                "quotaId": "GenerateContentRequestsPerDayPerProjectPerModel",
                "quotaValue": "20"
            }
        }]
        raise err

    with pytest.raises(Mock429Error):
        execute_and_log_api_request(
            mock_failing_api,
            model="gemini-3.8-flash",
            backend="gemini",
            cues_count=0,
            cache=cache
        )

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, error_type, quota_id, quota_value FROM api_requests")
        row = cursor.fetchone()
        assert row[0] == "exhausted_429_perday"
        assert row[1] == "429_PerDay"
        assert "PerDay" in row[2]
        assert row[3] == "20"


def test_pt_date_counting_without_fixed_offset(tmp_path):
    db_file = tmp_path / "test_cache.db"
    cache = TranslationCache(db_path=db_file)

    # Thêm bản ghi với ngày PT hiện tại
    cache.log_request(model="gemini-3.8-flash", backend="gemini", cues_count=100)
    cache.log_request(model="gemini-3.7-flash", backend="gemini", cues_count=100)

    reqs = cache.get_requests_today_pt()
    assert reqs.get("gemini-3.8-flash") == 1
    assert reqs.get("gemini-3.7-flash") == 1
