"""Auto-apply subtrans and unit markers to all tests under subtrans/tests/."""

from __future__ import annotations

from pathlib import Path
import ssl
import time
from unittest.mock import MagicMock
import pytest

_SUBTRANS_TESTS_DIR = Path(__file__).resolve().parent
_ORIG_CREATE_DEFAULT_CTX = ssl.create_default_context
_CACHED_DEFAULT_SSL_CTX: ssl.SSLContext | None = None


def _fast_cached_ssl_context(
    purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
    *,
    cafile: str | None = None,
    capath: str | None = None,
    cadata: str | bytes | None = None,
) -> ssl.SSLContext:
    global _CACHED_DEFAULT_SSL_CTX
    if purpose == ssl.Purpose.SERVER_AUTH and cafile is None and capath is None and cadata is None:
        if _CACHED_DEFAULT_SSL_CTX is None:
            _CACHED_DEFAULT_SSL_CTX = _ORIG_CREATE_DEFAULT_CTX(purpose)
        return _CACHED_DEFAULT_SSL_CTX
    return _ORIG_CREATE_DEFAULT_CTX(purpose, cafile=cafile, capath=capath, cadata=cadata)


ssl.create_default_context = _fast_cached_ssl_context

_KEEP_REAL_THROTTLE_SLEEP = {
    "test_throttle.py",
    "test_subtrans_resilience.py",
    "test_audit_fixes_subtrans.py",
}


_KEEP_REAL_GENAI_CLIENT = {
    "test_audit_fixes_subtrans.py",
    "test_subtrans_deep_upgrades.py",
    "test_subtrans_stub_pipeline.py",
}


@pytest.fixture(autouse=True)
def _fast_subtrans_retry_sleep(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    item_path = Path(str(getattr(request, "path", getattr(request.node, "fspath", "")))).resolve()
    if not item_path.is_relative_to(_SUBTRANS_TESTS_DIR):
        return
    if item_path.name not in _KEEP_REAL_GENAI_CLIENT:
        monkeypatch.setattr("subtrans.gemini_backend.genai.Client", lambda **_kw: MagicMock(), raising=False)
        monkeypatch.setattr("subtrans.cast_sheet.genai.Client", lambda **_kw: MagicMock(), raising=False)
    if item_path.name in _KEEP_REAL_THROTTLE_SLEEP:
        return
    monkeypatch.setattr(time, "sleep", lambda _s: None)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item_path = Path(str(getattr(item, "path", getattr(item, "fspath", "")))).resolve()
        if not item_path.is_relative_to(_SUBTRANS_TESTS_DIR):
            continue
        item.add_marker(pytest.mark.subtrans)
        if not item.get_closest_marker("gpu") and not item.get_closest_marker("e2e"):
            item.add_marker(pytest.mark.unit)
