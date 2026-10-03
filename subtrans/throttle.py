# -*- coding: utf-8 -*-
"""
subtrans/throttle.py
Stage 3: Rate limiting, exponential backoff with jitter, and quota exhaustion handling.
"""

import time
import random
import logging
from collections import deque
from typing import Optional, Callable, Any

logger = logging.getLogger("subtrans.throttle")


class QuotaExhaustedError(Exception):
    """Raised when the Gemini API daily quota (RPD) or sustained rate limit is exhausted."""
    pass


class RateLimiter:
    """Sliding-window rate limiter to enforce max requests per minute (RPM)."""

    def __init__(self, max_rpm: int = 15):
        self.max_rpm = max_rpm
        self.interval = 60.0
        self.timestamps = deque()

    def wait_if_needed(self):
        """Blocks until a request slot is available in the sliding 60s window."""
        now = time.monotonic()
        # Remove timestamps older than 60s
        while self.timestamps and (now - self.timestamps[0]) > self.interval:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.max_rpm:
            # Need to wait until the oldest timestamp is out of the window
            wait_time = self.interval - (now - self.timestamps[0]) + 0.1
            if wait_time > 0:
                logger.info(f"Rate limit reached ({self.max_rpm} RPM). Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                now = time.monotonic()
                while self.timestamps and (now - self.timestamps[0]) > self.interval:
                    self.timestamps.popleft()

        self.timestamps.append(time.monotonic())


def call_with_retry(
    fn: Callable[[], Any],
    rate_limiter: Optional[RateLimiter] = None,
    max_retries: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> Any:
    """
    Executes a callable with rate limiting and exponential backoff + jitter for 429/5xx errors.
    Raises QuotaExhaustedError if quota exhaustion is confirmed or retries are exceeded.
    """
    for attempt in range(max_retries + 1):
        if rate_limiter:
            rate_limiter.wait_if_needed()

        try:
            return fn()
        except Exception as e:
            err_str = str(e).lower()
            is_quota_exhausted = (
                "quota" in err_str or
                "resource_exhausted" in err_str or
                "exceeded your current quota" in err_str or
                "check quota" in err_str
            )
            is_rate_limit = "429" in err_str or "too many requests" in err_str
            is_server_error = "500" in err_str or "503" in err_str or "internal" in err_str

            if is_quota_exhausted:
                logger.error(f"Gemini API quota exhausted: {e}")
                raise QuotaExhaustedError(str(e)) from e

            if not (is_rate_limit or is_server_error) or attempt == max_retries:
                # Unrecoverable error or max retries reached
                if is_rate_limit:
                    raise QuotaExhaustedError(f"Repeated rate limits exceeded after {max_retries} retries: {e}") from e
                raise

            delay = min(base_delay * (2 ** attempt) + random.uniform(0.5, 1.5), max_delay)
            logger.warning(
                f"Request hit error ({e}). Retrying in {delay:.1f}s (Attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)
