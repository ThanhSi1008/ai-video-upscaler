"""Fast offline smoke test exercising dedup, two-way BT.709 conversion, 10-bit encode, and MKV mux audit."""

from __future__ import annotations

import pytest

from scripts.smoke_test_pipeline import run_smoke_test


@pytest.mark.smoke
def test_smoke_pipeline_end_to_end():
    res = run_smoke_test()
    assert res["status"] == "PASS"
    assert res["frames"] == 24
    assert res["dedup_saved"] > 0
    assert res["pix_fmt"] in {"yuv420p10le", "p010le"}
    assert res["sub_default"] is True
    assert res["mux_audit"] == "PASS"
    assert res["elapsed_s"] < 5.0
