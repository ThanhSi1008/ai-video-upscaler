"""Auto-apply e2e, web, smoke, and unit markers based on test file location under tests/."""

from __future__ import annotations

from pathlib import Path
import sys
import pytest


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
