"""Auto-apply e2e, web, smoke, and unit markers based on test file location under tests/."""

from __future__ import annotations

from pathlib import Path
import os
import sys
import pytest

os.environ.setdefault("SUBTRANS_DISABLE_DOTENV", "1")


@pytest.fixture(autouse=True)
def _isolate_gemini_env(monkeypatch: pytest.MonkeyPatch):
    """Prevent live shell or .env GEMINI_API_KEY* variables from leaking into tests."""
    monkeypatch.setenv("SUBTRANS_DISABLE_DOTENV", "1")
    for env_k in list(os.environ.keys()):
        if env_k.startswith("GEMINI_API_KEY"):
            monkeypatch.delenv(env_k, raising=False)
    yield


@pytest.fixture(autouse=True)
def _isolate_logs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate UPSCALER_LOGS_DIR/LOGS_DIR per test so synthetic runs never write to repo logs/."""
    test_logs = str(tmp_path / "_test_logs")
    monkeypatch.setenv("UPSCALER_LOGS_DIR", test_logs)
    cfg_mod = sys.modules.get("src.config")
    if cfg_mod is not None and hasattr(cfg_mod, "LOGS_DIR"):
        monkeypatch.setattr(cfg_mod, "LOGS_DIR", test_logs)
    logger_mod = sys.modules.get("src.logger")
    if logger_mod is not None:
        if hasattr(logger_mod, "_CONFIG_LOGS_DIR"):
            monkeypatch.setattr(logger_mod, "_CONFIG_LOGS_DIR", test_logs)
        if hasattr(logger_mod, "LOGS_DIR"):
            monkeypatch.setattr(logger_mod, "LOGS_DIR", test_logs)
    yield


@pytest.fixture(autouse=True)
def _reset_shared_rate_limiters():
    """Clear any already-imported subtrans RateLimiter registry without adding import overhead."""
    mod = sys.modules.get("subtrans.throttle")
    if mod is not None:
        mod.clear_rate_limiter_registry()
    yield
    mod = sys.modules.get("subtrans.throttle")
    if mod is not None:
        mod.clear_rate_limiter_registry()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item_path = Path(str(getattr(item, "path", getattr(item, "fspath", ""))))
        parts = {part.lower() for part in item_path.parts}
        name_lower = item_path.name.lower()
        if "e2e" in parts:
            item.add_marker(pytest.mark.e2e)
            continue
        if name_lower == "test_web_gui.py":
            item.add_marker(pytest.mark.web)
        if name_lower == "test_smoke_pipeline.py":
            item.add_marker(pytest.mark.smoke)
        if not item.get_closest_marker("gpu") and not item.get_closest_marker("e2e"):
            item.add_marker(pytest.mark.unit)
