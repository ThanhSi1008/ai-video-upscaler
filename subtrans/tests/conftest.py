"""Auto-apply subtrans and unit markers to all tests under subtrans/tests/."""

from __future__ import annotations

from pathlib import Path
import pytest

_SUBTRANS_TESTS_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item_path = Path(str(getattr(item, "path", getattr(item, "fspath", "")))).resolve()
        if not item_path.is_relative_to(_SUBTRANS_TESTS_DIR):
            continue
        item.add_marker(pytest.mark.subtrans)
        if not item.get_closest_marker("gpu") and not item.get_closest_marker("e2e"):
            item.add_marker(pytest.mark.unit)
