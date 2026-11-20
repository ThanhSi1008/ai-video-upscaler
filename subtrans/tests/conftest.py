"""Auto-apply subtrans and unit markers to all tests under subtrans/tests/."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.subtrans)
        if not item.get_closest_marker("gpu") and not item.get_closest_marker("e2e"):
            item.add_marker(pytest.mark.unit)
