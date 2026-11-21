"""Gemini rate limiter, exponential backoff with jitter, and PerDay quota failover."""

import os
import time
import random
import logging
import re
import json
import requests
from collections import deque
from typing import Optional, Callable, Any, Tuple, Dict

from subtrans.config import update_model_quota_in_yaml
from subtrans.cache import TranslationCache, get_next_pt_reset

logger = logging.getLogger("subtrans.throttle")


class QuotaExhaustedError(Exception):
    """Raised when the Gemini API quota is exhausted."""
    pass


class DailyQuotaExhaustedError(QuotaExhaustedError):
    """Raised when the Gemini API daily quota (RPD) is exhausted (quotaId contains PerDay)."""
    def __init__(
        self,
        model: str,
        quota_id: str = "",
        quota_value: str = "",
        reset_pt_str: str = "",
        reset_vn_str: str = "",
        details: Any = None
    ):
        self.model = model
        self.quota_id = quota_id
        self.quota_value = quota_value
        self.reset_pt_str = reset_pt_str
        self.reset_vn_str = reset_vn_str
        self.details = details
        super().__init__(
            f"Daily quota exhausted for '{model}': quotaId='{quota_id}', quotaValue='{quota_value}'. "
            f"Model locked until 00:00 Pacific Time ({reset_vn_str} VN)."
        )


class MinuteRateLimitError(Exception):
    """Raised when per-minute rate limit (RPM) is hit."""
    def __init__(self, model: str, quota_id: str = "", quota_value: str = "", retry_delay: float = 0.0):
        self.model = model
        self.quota_id = quota_id
        self.quota_value = quota_value
        self.retry_delay = retry_delay
        super().__init__(f"Per-minute rate limit for '{model}': retryDelay={retry_delay}s, quotaId='{quota_id}'")


def parse_429_details(exc: Exception) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Trích xuất quotaId, quotaValue, và retryDelay từ cấu trúc phản hồi lỗi của Google GenAI / HTTP 429.
    Returns: (quota_id, quota_value, retry_delay_seconds)
    """
    quota_id = None
    quota_value = None
    retry_delay = None

    # 1. Trích xuất từ exc.details (chuẩn Google GenAI ClientError / APIError)
    details_obj = getattr(exc, "details", None)
    if details_obj is None and hasattr(exc, "response_json"):
        details_obj = getattr(exc, "response_json", None)

    details_list = []
    if isinstance(details_obj, dict):
        # Có thể nằm trong details_obj["error"]["details"] hoặc details_obj["details"]
        if "error" in details_obj and isinstance(details_obj["error"], dict):
            details_list = details_obj["error"].get("details", [])
        elif "details" in details_obj:
            details_list = details_obj.get("details", [])
    elif isinstance(details_obj, list):
        details_list = details_obj

    if isinstance(details_list, list):
        for item in details_list:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type", "")

            # Trích xuất ErrorInfo / metadata
            if "ErrorInfo" in item_type or "metadata" in item:
                meta = item.get("metadata", {})
                if isinstance(meta, dict):
                    if "quotaId" in meta:
                        quota_id = str(meta["quotaId"])
                    if "quotaValue" in meta:
                        quota_value = str(meta["quotaValue"])
                    elif "quotaLimit" in meta:
                        quota_value = str(meta["quotaLimit"])

            # Trích xuất QuotaFailure / violations
            if "QuotaFailure" in item_type or "violations" in item:
                violations = item.get("violations", [])
                if isinstance(violations, list):
                    for v in violations:
                        desc = v.get("description", "")
                        # e.g. "Quota exceeded for quota metric 'GenerateContentRequestsPerDayPerProjectPerModel' and limit '20'..."
                        m_metric = re.search(r"metric\s+['\"]?([A-Za-z0-9_-]+)['\"]?", desc)
                        if m_metric and not quota_id:
                            quota_id = m_metric.group(1)
                        m_val = re.search(r"limit\s+['\"]?(\d+)['\"]?", desc)
                        if m_val and not quota_value:
                            quota_value = m_val.group(1)

            # Trích xuất RetryInfo / retryDelay
            if "RetryInfo" in item_type or "retryDelay" in item:
                delay_val = item.get("retryDelay")
                if delay_val:
                    if isinstance(delay_val, (int, float)):
                        retry_delay = float(delay_val)
                    elif isinstance(delay_val, str):
                        m = re.search(r"([\d\.]+)", delay_val)
                        if m:
                            retry_delay = float(m.group(1))

    # 2. Regex fallback quét trực tiếp trên chuỗi thông báo lỗi (str(exc))
    err_str = str(exc)
    if not quota_id:
        m_qid = re.search(r"quotaId['\":\s]+([A-Za-z0-9_-]+)", err_str, re.IGNORECASE)
        if m_qid:
            quota_id = m_qid.group(1)
        else:
            m_metric = re.search(r"metric\s+['\"]?([A-Za-z0-9_-]+)['\"]?", err_str)
            if m_metric:
                quota_id = m_metric.group(1)
            else:
                m_metric2 = re.search(r"((?:GenerateContent)?RequestsPer[A-Za-z0-9]+)", err_str)
                if m_metric2:
                    quota_id = m_metric2.group(1)

    if not quota_value:
        m_qval = re.search(r"quotaValue['\":\s]+['\"]?(\d+)['\"]?", err_str, re.IGNORECASE)
        if m_qval:
            quota_value = m_qval.group(1)
        else:
            m_limit = re.search(r"limit\s+['\"]?(\d+)['\"]?", err_str, re.IGNORECASE)
            if m_limit:
                quota_value = m_limit.group(1)

    if retry_delay is None:
        m_delay = re.search(r"retryDelay['\":\s]+['\"]?([\d\.]+)s?['\"]?", err_str, re.IGNORECASE)
        if m_delay:
            retry_delay = float(m_delay.group(1))
        else:
            m_wait = re.search(r"retry in ([\d\.]+)s", err_str, re.IGNORECASE)
            if m_wait:
                retry_delay = float(m_wait.group(1))

    return quota_id, quota_value, retry_delay


class RateLimiter:
    """Sliding-window rate limiter to enforce max requests per minute (RPM)."""

    def __init__(self, max_rpm: Any = 5):
        try:
            self.max_rpm = int(max_rpm)
        except Exception:
            self.max_rpm = 5
        self.interval = 60.0
        self.timestamps = deque()
        self.cooldown_until: float = 0.0

    def record_cooldown(self, wait_seconds: float) -> None:
        """Records a per-key cooldown timestamp after hitting 429 PerMinute."""
        if wait_seconds and wait_seconds > 0:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + float(wait_seconds))

    def clear_cooldown(self) -> None:
        """Clears active cooldown once wait has been honored."""
        self.cooldown_until = 0.0

    def available_slots(self) -> int:
        """Trả về số slot RPM còn trống ngay tức thì trong cửa sổ trượt 60s."""
        now = time.monotonic()
        if now < self.cooldown_until:
            return 0
        while self.timestamps and (now - self.timestamps[0]) > self.interval:
            self.timestamps.popleft()
        return max(0, self.max_rpm - len(self.timestamps))

    def wait_if_needed(self):
        """Blocks until any active cooldown expires and a request slot is available in the 60s window."""
        now = time.monotonic()
        if now < self.cooldown_until:
            cd_wait = self.cooldown_until - now
            if cd_wait > 0:
                logger.info(f"Key cooldown active. Waiting {cd_wait:.1f}s...")
                time.sleep(cd_wait)
            self.cooldown_until = 0.0
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
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    model_name: str = "gemini",
    cache: Optional[TranslationCache] = None,
    key_id: str = "",
    all_key_ids: Optional[list] = None,
    raise_on_per_minute: bool = False,
) -> Any:
    """
    Executes a callable with rate limiting and intelligent 429 handling.
    - If 429 contains PerDay -> marks (key_id, model) exhausted until PT reset; only marks
      global model_exhaustion when all keys are exhausted (or when no key_id is provided),
      and raises DailyQuotaExhaustedError IMMEDIATELY (no retries/pause).
    - If 429 contains PerMinute -> sleeps retryDelay and retries.
    """
    cache = cache or TranslationCache()

    # Pre-check: Nếu key cụ thể hoặc toàn bộ model đã hết hạn mức ngày trong DB, chuyển ngay!
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        if key_id and cache.is_key_model_exhausted(key_id, model_name):
            next_pt, next_vn = get_next_pt_reset()
            raise DailyQuotaExhaustedError(
                model=model_name,
                quota_id="CachedKeyExhaustion",
                quota_value="",
                reset_pt_str=next_pt.strftime("%Y-%m-%d %H:%M PT"),
                reset_vn_str=next_vn.strftime("%H:%M %d/%m VN"),
            )
        if cache.is_model_exhausted(model_name):
            if not all_key_ids or cache.are_all_keys_exhausted(all_key_ids, model_name):
                ex_info = cache.get_model_exhaustion_info(model_name) or {}
                next_pt, next_vn = get_next_pt_reset()
                logger.warning(
                    f"Model '{model_name}' đã được đánh dấu hết hạn mức ngày trước đó (đến {next_vn.strftime('%H:%M %d/%m')}). "
                    f"Bỏ qua ngay lập tức để chuyển model kế tiếp!"
                )
                raise DailyQuotaExhaustedError(
                    model=model_name,
                    quota_id=ex_info.get("quota_id", "CachedExhaustion"),
                    quota_value=ex_info.get("quota_value", ""),
                    reset_pt_str=next_pt.strftime("%Y-%m-%d %H:%M PT"),
                    reset_vn_str=next_vn.strftime("%H:%M %d/%m VN"),
                )

    for attempt in range(max_retries + 1):
        if rate_limiter:
            rate_limiter.wait_if_needed()

        try:
            return fn()
        except Exception as e:
            err_str = str(e).lower()
            is_429 = (
                "429" in err_str or
                "too many requests" in err_str or
                "resource_exhausted" in err_str or
                "resource has been exhausted" in err_str
            )

            if is_429:
                quota_id, quota_value, retry_delay = parse_429_details(e)
                qid_lower = (quota_id or "").lower()

                # Phân loại 1: Lỗi PerDay (Daily Quota RPD)
                is_per_day = (
                    "perday" in qid_lower or
                    "requestsperday" in qid_lower or
                    "daily" in qid_lower or
                    "day" in qid_lower
                )

                if is_per_day:
                    next_pt, next_vn = get_next_pt_reset()
                    vn_str = next_vn.strftime("%H:%M ngày %d/%m")
                    pt_str = next_pt.strftime("%Y-%m-%d %H:%M PT")

                    logger.error(
                        f"🛑 [429 PER-DAY] Model '{model_name}' đã cạn kiệt hạn mức ngày (RPD)! "
                        f"quotaId='{quota_id}', quotaValue='{quota_value}'. "
                        f"Tự động khóa model đến 00:00 PT ({vn_str} VN) và CHUYỂN MODEL NGAY LẬP TỨC!"
                    )

                    # Ghi quotaValue vào log và tự cập nhật models.yaml
                    if quota_value:
                        update_model_quota_in_yaml(
                            model_name=model_name,
                            quota_id=quota_id or "PerDay",
                            quota_value=quota_value,
                            rpd_source=f"đọc từ lỗi 429 ({quota_id})"
                        )

                    # Đánh dấu (key_id, model) hoặc model toàn cục hết hạn mức trong SQLite cache
                    if key_id and key_id != "none":
                        cache.mark_key_model_exhausted(
                            key_id=key_id,
                            model=model_name,
                            quota_id=quota_id or "PerDay",
                            quota_value=quota_value or "",
                            reason=f"429 PerDay: {quota_id}",
                        )
                        if all_key_ids and cache.are_all_keys_exhausted(all_key_ids, model_name):
                            cache.mark_model_exhausted(
                                model=model_name,
                                exhausted_until_pt=next_pt,
                                quota_id=quota_id or "PerDay",
                                quota_value=quota_value or "",
                                reason=f"All keys 429 PerDay: {quota_id}",
                            )
                    else:
                        cache.mark_model_exhausted(
                            model=model_name,
                            exhausted_until_pt=next_pt,
                            quota_id=quota_id or "PerDay",
                            quota_value=quota_value or "",
                            reason=f"429 PerDay: {quota_id}",
                        )

                    # Log request lỗi vào DB nếu inner callable chưa log
                    if not getattr(e, "_logged_api_request", False):
                        cache.log_request(
                            model=model_name,
                            backend="gemini",
                            cues_count=0,
                            status="exhausted_429_perday",
                            error_type="429_PerDay",
                            quota_id=quota_id or "",
                            quota_value=quota_value or "",
                            key_id=key_id,
                        )

                    # Ném DailyQuotaExhaustedError ngay lập tức, không thử lại, không pause!
                    dqe = DailyQuotaExhaustedError(
                        model=model_name,
                        quota_id=quota_id or "",
                        quota_value=quota_value or "",
                        reset_pt_str=pt_str,
                        reset_vn_str=vn_str,
                        details=getattr(e, "details", None)
                    )
                    try:
                        setattr(dqe, "_logged_api_request", True)
                    except Exception:
                        pass
                    raise dqe from e

                # Phân loại 2: Lỗi PerMinute (Rate Limit RPM)
                is_per_minute = (
                    "perminute" in qid_lower or
                    "requestsperminute" in qid_lower or
                    "minute" in qid_lower or
                    (retry_delay is not None and retry_delay > 0)
                )

                if is_per_minute and raise_on_per_minute:
                    raise MinuteRateLimitError(
                        model=model_name,
                        quota_id=quota_id or "",
                        quota_value=quota_value or "",
                        retry_delay=float(retry_delay) if (retry_delay and retry_delay > 0) else 10.0,
                    ) from e

                if attempt < max_retries:
                    wait_sec = retry_delay if (retry_delay and retry_delay > 0) else min(base_delay * (2 ** attempt) + random.uniform(0.5, 1.5), max_delay)
                    logger.warning(
                        f"⏳ [429 PER-MINUTE] Model '{model_name}' chạm giới hạn tốc độ/phút (quotaId='{quota_id}', quotaValue='{quota_value}'). "
                        f"Chờ {wait_sec:.1f}s theo retryDelay rồi thử lại (Lần {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_sec)
                    continue

                # Nếu không xác định rõ và hết số lần thử -> coi như cạn kiệt
                raise QuotaExhaustedError(f"Repeated 429 rate limit exceeded on '{model_name}' after {max_retries} retries: {e}") from e

            # Các lỗi mạng hoặc server khác (500, 502, 503, 504, timeout)
            is_server_error = (
                "500" in err_str or "502" in err_str or "503" in err_str or "504" in err_str or
                "internal" in err_str or "unavailable" in err_str or "high demand" in err_str
            )
            is_timeout = (
                "timeout" in err_str or "timed out" in err_str or "deadline" in err_str or
                isinstance(e, (TimeoutError, requests.exceptions.Timeout))
            )

            if (is_server_error or is_timeout) and attempt < max_retries:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0.5, 1.5), max_delay)
                logger.warning(
                    f"⚠️ Request hit server/timeout error ({e}). Retrying in {delay:.1f}s (Attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(delay)
                continue

            # Unrecoverable error or max retries reached
            raise

