# -*- coding: utf-8 -*-
import pytest
import time
from unittest.mock import MagicMock
from subtrans.throttle import RateLimiter, call_with_retry, QuotaExhaustedError


def test_rate_limiter_timing():
    limiter = RateLimiter(max_rpm=60)
    t0 = time.monotonic()
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    t1 = time.monotonic()
    # Should not block heavily for 2 requests at 60 RPM
    assert (t1 - t0) < 1.0


def test_call_with_retry_quota_exhausted():
    mock_fn = MagicMock(side_effect=Exception("Resource has been exhausted (e.g. check quota)."))
    with pytest.raises(QuotaExhaustedError):
        call_with_retry(mock_fn, max_retries=1, base_delay=0.01)
