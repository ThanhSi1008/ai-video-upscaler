# -*- coding: utf-8 -*-
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from subtrans.cache import TranslationCache, compute_key_id
from subtrans.cast_sheet import generate_cast_sheet
from subtrans.gemini_backend import GeminiTranslator
from subtrans.models import Cue
from subtrans.throttle import (
    QuotaExhaustedError,
    RateLimiter,
    call_with_retry,
    clear_rate_limiter_registry,
    get_client_rpm_for_model,
    get_shared_rate_limiter,
)


def test_rate_limiter_timing():
    limiter = RateLimiter(max_rpm=60)
    t0 = time.monotonic()
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    t1 = time.monotonic()
    # Should not block heavily for 2 requests at 60 RPM
    assert (t1 - t0) < 1.0


def test_rate_limiter_invalid_rpm_falls_back_to_five():
    assert RateLimiter(max_rpm=0).max_rpm == 5
    assert RateLimiter(max_rpm=-4).max_rpm == 5
    assert RateLimiter(max_rpm="bad").max_rpm == 5


def test_call_with_retry_quota_exhausted(monkeypatch):
    monkeypatch.setattr("subtrans.throttle.time.sleep", lambda _s: None)
    mock_fn = MagicMock(side_effect=Exception("Resource has been exhausted (e.g. check quota)."))
    with pytest.raises(QuotaExhaustedError):
        call_with_retry(mock_fn, max_retries=1, base_delay=0.01)


def test_shared_rate_limiter_rpm_ceilings_and_unknown_fallback():
    assert get_client_rpm_for_model("gemini-3.8-flash") == 4
    assert get_client_rpm_for_model("gemini-3.5-flash-lite") == 12
    assert get_client_rpm_for_model("unknown-custom-flash") == 4
    assert get_client_rpm_for_model("unknown-custom-lite") == 12
    assert get_client_rpm_for_model("m1", models_cfg={"models": {"m1": {"rpm": 1}}}) == 1
    assert get_client_rpm_for_model("m10", models_cfg={"models": {"m10": {"rpm": 10}}}) == 8


def test_shared_rate_limiter_identity_and_clear():
    lim1 = get_shared_rate_limiter("kid_a", "gemini-3.8-flash")
    lim2 = get_shared_rate_limiter("kid_a", "gemini-3.8-flash")
    lim_other_key = get_shared_rate_limiter("kid_b", "gemini-3.8-flash")
    lim_other_model = get_shared_rate_limiter("kid_a", "gemini-3.5-flash-lite")

    assert lim1 is lim2
    assert lim1 is not lim_other_key
    assert lim1 is not lim_other_model
    assert lim1.max_rpm == 4
    assert lim_other_model.max_rpm == 12

    clear_rate_limiter_registry()
    lim_after_clear = get_shared_rate_limiter("kid_a", "gemini-3.8-flash")
    assert lim_after_clear is not lim1


def test_shared_rate_limiter_across_translator_and_cast_sheet(tmp_path, monkeypatch):
    monkeypatch.setattr("subtrans.throttle.time.sleep", lambda _s: None)
    cache = TranslationCache(db_path=tmp_path / "shared_lim.db")
    api_key = "AIzaSharedTestKey123"
    kid = compute_key_id(api_key)

    mock_tr_client = MagicMock()
    mock_cs_client = MagicMock()
    cs_resp = MagicMock()
    cs_resp.text = '{"characters": [], "relations": [], "notes": ""}'
    cs_resp.usage_metadata = None
    mock_cs_client.models.generate_content.return_value = cs_resp

    with patch("subtrans.gemini_backend.genai.Client", return_value=mock_tr_client), \
         patch("subtrans.cast_sheet.genai.Client", return_value=mock_cs_client):
        tr = GeminiTranslator(
            api_keys=[api_key],
            model="gemini-3.8-flash",
            cache=cache,
        )
        assert tr.rate_limiters[kid].available_slots() == 4
        generate_cast_sheet(
            [Cue(id=1, start=0, end=1000, ja="テスト", en="Test")],
            output_path=tmp_path / "ep.cast.yaml",
            gemini_model="gemini-3.8-flash",
            cache=cache,
            api_keys=[api_key],
        )
        assert tr.rate_limiters[kid] is get_shared_rate_limiter(kid, "gemini-3.8-flash")
        assert tr.rate_limiters[kid].available_slots() == 3


def test_rate_limiter_concurrent_window_enforcement():
    limiter = RateLimiter(max_rpm=3)
    limiter.interval = 0.25
    barrier = threading.Barrier(6)
    completion_times: list[float] = []
    times_lock = threading.Lock()
    t_start = time.monotonic()

    def worker():
        barrier.wait(timeout=5.0)
        limiter.wait_if_needed()
        with times_lock:
            completion_times.append(time.monotonic() - t_start)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(completion_times) == 6
    completion_times.sort()
    # First 3 slots pass immediately; next 3 must wait at least ~0.20s for window rollover
    first_wave = [dt for dt in completion_times if dt < 0.15]
    second_wave = [dt for dt in completion_times if dt >= 0.20]
    assert len(first_wave) <= 3
    assert len(second_wave) >= 3


def test_rate_limiter_wait_does_not_block_slots_or_cooldown():
    limiter = RateLimiter(max_rpm=2)
    limiter.record_cooldown(0.35)
    t_done: list[float] = []

    def waiter():
        limiter.wait_if_needed()
        t_done.append(time.monotonic())

    t0 = time.monotonic()
    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.05)

    # available_slots and record_cooldown must not block on the sleeping thread
    probe_start = time.monotonic()
    slots = limiter.available_slots()
    limiter.record_cooldown(0.45)
    probe_elapsed = time.monotonic() - probe_start
    th.join(timeout=3.0)

    assert slots == 0
    assert probe_elapsed < 0.15
    assert len(t_done) == 1
    assert (t_done[0] - t0) >= 0.40


def test_rate_limiter_concurrent_registry_lookup():
    barrier = threading.Barrier(8)
    results: list[RateLimiter] = []
    res_lock = threading.Lock()

    def lookup():
        barrier.wait(timeout=5.0)
        lim = get_shared_rate_limiter("conc_kid", "gemini-3.8-flash")
        with res_lock:
            results.append(lim)

    threads = [threading.Thread(target=lookup) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(results) == 8
    assert all(r is results[0] for r in results)


def test_rate_limiter_mocked_sleep_no_spin(monkeypatch):
    monkeypatch.setattr("subtrans.throttle.time.sleep", lambda _s: None)
    limiter = RateLimiter(max_rpm=2)
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    # Window is now full; mocked sleep must break cleanly instead of spinning forever
    limiter.wait_if_needed()
    limiter.record_cooldown(30.0)
    limiter.wait_if_needed()


def test_rate_limiter_bool_rpm_falls_back_to_five():
    assert RateLimiter(max_rpm=True).max_rpm == 5
    assert RateLimiter(max_rpm=False).max_rpm == 5


def test_rate_limiter_short_real_sleep_respects_extended_cooldown(monkeypatch):
    orig_sleep = time.sleep
    main_thread = threading.current_thread()

    for _ in range(4):
        limiter = RateLimiter(max_rpm=4)
        ready = threading.Event()
        entered_sleep = threading.Event()
        extended_done = threading.Event()
        extend_ran = [False]

        def spy_sleep(seconds, entered=entered_sleep, done=extended_done):
            if threading.current_thread() is main_thread and not done.is_set():
                entered.set()
                assert done.wait(timeout=5.0)
            orig_sleep(seconds)

        def extend(
            lim=limiter,
            rdy=ready,
            entered=entered_sleep,
            done=extended_done,
            ran=extend_ran,
        ):
            try:
                rdy.set()
                if entered.wait(timeout=5.0):
                    lim.record_cooldown(0.45)
                    ran[0] = True
            finally:
                done.set()

        monkeypatch.setattr("subtrans.throttle.time.sleep", spy_sleep)
        th = threading.Thread(target=extend)
        th.start()
        assert ready.wait(timeout=5.0)

        # Align to a monotonic tick boundary so 10ms real sleep finishes within one Windows tick
        t_tick = time.monotonic()
        while time.monotonic() == t_tick:
            pass
        limiter.record_cooldown(0.01)
        t_start = time.perf_counter()
        try:
            limiter.wait_if_needed()
            elapsed = time.perf_counter() - t_start
        finally:
            th.join(timeout=5.0)

        assert extend_ran[0]
        assert elapsed >= 0.40


def test_rate_limiter_tiny_cooldown_mocked_sleep_no_spin(monkeypatch):
    monkeypatch.setattr("subtrans.throttle.time.sleep", lambda _s: None)
    log_calls = 0

    def counting_info(*_args, **_kwargs):
        nonlocal log_calls
        log_calls += 1

    monkeypatch.setattr("subtrans.throttle.logger.info", counting_info)

    lim_real_clock = RateLimiter(max_rpm=2)
    lim_real_clock.record_cooldown(0.001)
    lim_real_clock.wait_if_needed()
    assert log_calls <= 5

    log_calls = 0
    monkeypatch.setattr("subtrans.throttle.time.monotonic", lambda: 1000.0)
    lim_frozen_clock = RateLimiter(max_rpm=2)
    lim_frozen_clock.record_cooldown(0.001)
    lim_frozen_clock.wait_if_needed()
    assert log_calls <= 5

    log_calls = 0
    lim_sub_us = RateLimiter(max_rpm=2)
    lim_sub_us.record_cooldown(1e-6)
    lim_sub_us.wait_if_needed()
    assert log_calls <= 5


def test_shared_rate_limiter_inf_rpm_falls_back_without_overflow():
    assert get_shared_rate_limiter("inf_kid", "gemini-3.8-flash", max_rpm=float("inf")).max_rpm == 4
    assert get_shared_rate_limiter("neg_inf_kid", "gemini-3.8-flash", max_rpm=float("-inf")).max_rpm == 4
    assert get_client_rpm_for_model("m", models_cfg={"models": {"m": {"rpm": float("inf")}}}) == 4
    assert get_client_rpm_for_model("m", models_cfg={"models": {"m": {"rpm": float("-inf")}}}) == 4


def test_rate_limiter_record_cooldown_ignores_non_finite_and_invalid():
    limiter = RateLimiter(max_rpm=4)
    invalid_values = [
        float("inf"),
        float("-inf"),
        float("nan"),
        True,
        False,
        "bad",
        0,
        -1.0,
    ]
    for val in invalid_values:
        limiter.record_cooldown(val)  # type: ignore[arg-type]
        assert limiter.cooldown_until == 0.0

    limiter.wait_if_needed()

    limiter.record_cooldown(0.05)
    baseline = limiter.cooldown_until
    assert baseline > 0.0
    for val in invalid_values:
        limiter.record_cooldown(val)  # type: ignore[arg-type]
        assert limiter.cooldown_until == baseline
