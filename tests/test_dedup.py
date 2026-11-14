# -*- coding: utf-8 -*-
"""
tests/test_dedup.py
Unit tests for dedup engine optimizations:
- 3-Tier Hierarchical Cel Memoization (64x36 INTER_NEAREST + BLAKE2b 64-bit + SIMD uint8 NORM_INF).
- Decoupled DIS Optical Flow: Global motion (0.35px pan/zoom) vs Local motion (1.2-1.5px area).
- Sakuga dedup policy parameter registration and execution.
"""

from __future__ import annotations

import math
from unittest import mock

import cv2
import numpy as np
import pytest

from src.dedup import (
    DEDUP_POLICY_PARAMS,
    _is_dis_motion_early_exit,
    deduplicate_cyclic_frames,
)


# -----------------------------------------------------------------------------
# 1. 3-Tier Hierarchical Cel Memoization Tests
# -----------------------------------------------------------------------------

def test_deduplicate_cyclic_frames_3tier_exact_and_quant_noise():
    """
    Verifies 3-tier hierarchical filter:
    - Tier 1: exact matches detected via 64x36 thumbnail hash + SIMD NORM_INF == 0.
    - Tier 2 & 3: quantization noise (within max_pixel_tol and max_mae) memoized.
    - True motion (eyebrow, mouth, pan > max_pixel_tol) kept as new distinct frames.
    """
    rng = np.random.RandomState(42)
    cel_a = rng.randint(30, 220, size=(180, 320, 3), dtype=np.uint8)
    cel_b = cel_a.copy()
    cel_b[60:90, 140:180] = 250  # Mouth shape B
    cel_c = cel_a.copy()
    cel_c[55:95, 135:185] = 10   # Mouth shape C

    # cel_a with subtle compression noise (max_diff=1, mean ~0.05)
    cel_a_noise = cel_a.copy()
    cel_a_noise[::4, ::4, 0] = np.clip(cel_a_noise[::4, ::4, 0].astype(np.int16) + 1, 0, 255).astype(np.uint8)

    # cel_a with 4x4 micro motion (max_diff=25 > 2)
    cel_a_motion = cel_a.copy()
    cel_a_motion[80:84, 150:154] = np.clip(cel_a_motion[80:84, 150:154].astype(np.int16) + 25, 0, 255).astype(np.uint8)

    seq = [cel_a, cel_b, cel_c, cel_b.copy(), cel_a_noise, cel_a_motion]
    distinct, mapping = deduplicate_cyclic_frames(seq, max_pixel_tol=2, max_mae=0.20)

    assert len(distinct) == 4
    assert mapping == [0, 1, 2, 1, 0, 3]


def test_deduplicate_cyclic_frames_edge_cases():
    """Single frame or empty sequence returns empty or single item immediately."""
    assert deduplicate_cyclic_frames([]) == ([], [])
    single = [np.zeros((10, 10, 3), dtype=np.uint8)]
    d, m = deduplicate_cyclic_frames(single)
    assert len(d) == 1
    assert m == [0]


def test_deduplicate_cyclic_frames_2d_and_channels():
    """Handles 2D grayscale frames as well as 3-channel RGB without crashing."""
    gray_a = np.full((100, 100), 50, dtype=np.uint8)
    gray_b = np.full((100, 100), 50, dtype=np.uint8)
    gray_c = np.full((100, 100), 120, dtype=np.uint8)

    d, m = deduplicate_cyclic_frames([gray_a, gray_b, gray_c])
    assert len(d) == 2
    assert m == [0, 0, 1]


# -----------------------------------------------------------------------------
# 2. Decoupled DIS Optical Flow Tests
# -----------------------------------------------------------------------------

def test_dis_sakuga_policy_params_registered():
    """Verifies that 'sakuga' and 'dis-sakuga' are properly registered in DEDUP_POLICY_PARAMS."""
    assert "sakuga" in DEDUP_POLICY_PARAMS
    assert "dis-sakuga" in DEDUP_POLICY_PARAMS

    p = DEDUP_POLICY_PARAMS["sakuga"]
    assert p["dis_thresh"] == 0.35
    assert p["local_thresh"] == 1.50
    assert p["min_moving_area_pct"] == 12.0
    assert p["hi"] == 1024
    assert p["lo"] == 448
    assert p["frac"] == 0.38


def test_dis_motion_early_exit_identical_frames():
    """np.array_equal frames return False in 0ms without invoking dis.calc."""
    mock_dis = mock.MagicMock()
    frame = np.full((270, 480), 100, dtype=np.uint8)

    result = _is_dis_motion_early_exit(
        dis=mock_dis,
        ref_gray=frame,
        curr_gray=frame,
        scale_factor=np.float32(4.0),
        thresh=0.30,
        thresh_sq=0.09,
        min_moving_area_pct=5.0,
        r_mask=np.zeros((270, 480), dtype=bool),
        rx_inv_r2=np.zeros(0, dtype=np.float32),
        ry_inv_r2=np.zeros(0, dtype=np.float32),
        r_max=275.0,
    )
    assert result is False
    mock_dis.calc.assert_not_called()


def test_dis_motion_early_exit_line_boil_rejection_with_local_thresh():
    """
    Verifies that local_thresh=1.50 rejects sub-pixel line boil (e.g. 0.8px jitter),
    whereas default thresh=0.30 would erroneously trigger on noise floor.
    """
    mock_dis = mock.MagicMock()
    # Mock a flow field with 0.8px motion across ~7.7% of pixels (simulating line boil / compression noise)
    h, w = 270, 480
    flow = np.zeros((h, w, 2), dtype=np.float32)
    # Set 0.20px in 270p -> 0.80px in 1080p
    flow[80:180, 100:200, 0] = 0.20

    mock_dis.calc.return_value = flow
    ref = np.zeros((h, w), dtype=np.uint8)
    curr = ref.copy()
    curr[0, 0] = 1  # break array_equal

    r_mask = np.zeros((h, w), dtype=bool)

    # With local_thresh=1.50 (sakuga policy): 0.80px line boil is REJECTED (returns False)
    res_sakuga = _is_dis_motion_early_exit(
        dis=mock_dis,
        ref_gray=ref,
        curr_gray=curr,
        scale_factor=np.float32(4.0),
        thresh=0.35,
        thresh_sq=0.1225,
        min_moving_area_pct=5.0,
        r_mask=r_mask,
        rx_inv_r2=np.zeros(0, dtype=np.float32),
        ry_inv_r2=np.zeros(0, dtype=np.float32),
        r_max=275.0,
        local_thresh=1.50,
    )
    assert res_sakuga is False

    # With local_thresh=0.30 (old behavior): 0.80px line boil triggers Cửa 1 (returns True)
    res_old = _is_dis_motion_early_exit(
        dis=mock_dis,
        ref_gray=ref,
        curr_gray=curr,
        scale_factor=np.float32(4.0),
        thresh=0.30,
        thresh_sq=0.09,
        min_moving_area_pct=5.0,
        r_mask=r_mask,
        rx_inv_r2=np.zeros(0, dtype=np.float32),
        ry_inv_r2=np.zeros(0, dtype=np.float32),
        r_max=275.0,
        local_thresh=0.30,
    )
    assert res_old is True


def test_dis_motion_early_exit_camera_pan_preserved():
    """
    Verifies that camera pan (median motion >= thresh) is correctly caught by Cửa 2
    even when local_thresh is high.
    """
    mock_dis = mock.MagicMock()
    h, w = 270, 480
    flow = np.zeros((h, w, 2), dtype=np.float32)
    # Global pan: 0.125px in 270p -> 0.50px in 1080p (> 0.35px threshold) across whole frame
    flow[..., 0] = 0.125

    mock_dis.calc.return_value = flow
    ref = np.zeros((h, w), dtype=np.uint8)
    curr = ref.copy()
    curr[0, 0] = 1

    r_mask = np.zeros((h, w), dtype=bool)

    result = _is_dis_motion_early_exit(
        dis=mock_dis,
        ref_gray=ref,
        curr_gray=curr,
        scale_factor=np.float32(4.0),
        thresh=0.35,
        thresh_sq=0.1225,
        min_moving_area_pct=15.0,
        r_mask=r_mask,
        rx_inv_r2=np.zeros(0, dtype=np.float32),
        ry_inv_r2=np.zeros(0, dtype=np.float32),
        r_max=275.0,
        local_thresh=1.50,
    )
    # Must trigger Cửa 2 (Pan/Tilt)
    assert result is True
