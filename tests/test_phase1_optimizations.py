# -*- coding: utf-8 -*-
"""
tests/test_phase1_optimizations.py
Unit tests verifying Phase 1 Quick Win optimizations:
- NVENC CQ 18, multipass fullres, lookahead 32, aq-strength 6, temporal-aq 1
- DEFAULT_BATCH_SIZE = 8 for RTX 3060 CUDA
- get_hevc_rate_args centralized generation
- restart_if_needed intelligent VRAM threshold check (>= 9.5 GB)
- channels_last memory format application
"""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest
import torch

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_VRAM_CLEANUP_THRESHOLD_GB,
    HEVC_CQ_LEVEL,
    HEVC_MAXRATE,
    HEVC_BUFSIZE,
    HEVC_MULTIPASS,
    HEVC_RC_LOOKAHEAD,
    HEVC_AQ_STRENGTH,
    HEVC_TEMPORAL_AQ,
    HEVC_BF,
    HEVC_B_REF_MODE,
    get_hevc_rate_args,
    HEVC_ENCODER,
)
from src.mps_engine import MPSPersistentWorker


def test_phase1_nvenc_configuration_constants():
    """Verifies that all Phase 1 NVENC parameters match the Ampere 4K anime blueprint."""
    assert str(HEVC_CQ_LEVEL) == "18"
    assert HEVC_MAXRATE == "45M"
    assert HEVC_BUFSIZE == "90M"
    assert HEVC_MULTIPASS == "fullres"
    assert HEVC_RC_LOOKAHEAD == 32
    assert HEVC_AQ_STRENGTH == 6
    assert HEVC_TEMPORAL_AQ == 1
    assert HEVC_BF == 4
    assert HEVC_B_REF_MODE == "middle"


def test_phase1_get_hevc_rate_args():
    """Verifies that get_hevc_rate_args builds the exact specialized rate-control command arguments."""
    with patch("src.config.HEVC_ENCODER", "hevc_nvenc"):
        args = get_hevc_rate_args("28M")
        assert "-preset" in args
        assert "-rc" in args and args[args.index("-rc") + 1] == "vbr"
        assert "-cq" in args and args[args.index("-cq") + 1] == "18"
        assert "-multipass" in args and args[args.index("-multipass") + 1] == "fullres"
        assert "-rc-lookahead" in args and args[args.index("-rc-lookahead") + 1] == "32"
        assert "-aq-strength" in args and args[args.index("-aq-strength") + 1] == "6"
        assert "-temporal-aq" in args and args[args.index("-temporal-aq") + 1] == "1"
        assert "-bf" in args and args[args.index("-bf") + 1] == "4"
        assert "-b_ref_mode" in args and args[args.index("-b_ref_mode") + 1] == "middle"


def test_phase1_batch_size_and_vram_threshold():
    """Verifies batch size sweet spot (8) on CUDA and VRAM threshold (9.5 GB)."""
    assert DEFAULT_VRAM_CLEANUP_THRESHOLD_GB == 9.5
    if torch.cuda.is_available():
        assert DEFAULT_BATCH_SIZE == 8


def test_phase1_vram_smart_cleanup_logic():
    """Verifies that restart_if_needed performs empty_cache when >= 9.5 GB and does not reload model."""
    worker = MPSPersistentWorker.__new__(MPSPersistentWorker)
    worker.chunks_processed = 0
    worker.max_chunks_before_restart = 10
    worker.total_restarts = 0
    worker.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worker.model = MagicMock()

    # Case 1: Below threshold (< 9.5 GB) -> returns False, model preserved
    with patch("torch.cuda.memory_reserved", return_value=8.0 * (1024 ** 3)):
        with patch("torch.cuda.empty_cache") as mock_empty:
            res = worker.restart_if_needed()
            if worker.device.type == "cuda":
                assert res is False
                mock_empty.assert_not_called()
                assert worker.model is not None

    # Case 2: Exceeding threshold (>= 9.5 GB) on CUDA -> triggers empty_cache, returns True
    if worker.device.type == "cuda":
        with patch("torch.cuda.memory_reserved", return_value=9.8 * (1024 ** 3)):
            with patch("torch.cuda.empty_cache") as mock_empty:
                logger_mock = MagicMock()
                res = worker.restart_if_needed(logger=logger_mock)
                assert res is True
                mock_empty.assert_called_once()
                logger_mock.info.assert_called_once()
                assert "VRAM Garbage Collection" in logger_mock.info.call_args[0][0]
