# -*- coding: utf-8 -*-
"""
tests/test_deadline_math_realignment.py
Unit tests verifying the realignment of --deadline calculation in upscale.py and src/scheduler.py
to genuine Unique Inference FPS (frames_unique / time_ai), eliminating the ~2.1x estimation
underestimation discrepancy and post-audit step-change bug.
"""

from datetime import datetime, timedelta
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scheduler import compute_overnight_schedule, format_schedule_decision_table
import upscale


class TestDeadlineMathRealignment(unittest.TestCase):
    """Test suite for Unique Inference FPS realignment in scheduler and upscale.py."""

    def test_compute_overnight_schedule_default_arguments(self):
        """Verify compute_overnight_schedule defaults are aligned to Unique Inference FPS."""
        sig = inspect.signature(compute_overnight_schedule)
        params = sig.parameters

        self.assertEqual(params["fps_median"].default, 10.5)
        self.assertEqual(params["fps_p5"].default, 8.5)
        self.assertEqual(params["video_overhead_seconds"].default, 420.0)
        self.assertEqual(params["subtitles_overhead_seconds"].default, 300.0)

    def test_check_deadline_and_eta_default_arguments(self):
        """Verify upscale.check_deadline_and_eta defaults match scheduler Unique Inference FPS."""
        sig = inspect.signature(upscale.check_deadline_and_eta)
        params = sig.parameters

        self.assertEqual(params["fps_median"].default, 10.5)
        self.assertEqual(params["fps_p5"].default, 8.5)
        self.assertEqual(params["video_overhead"].default, 420.0)
        self.assertEqual(params["subtitles_overhead"].default, 300.0)

    def test_no_step_change_between_preset_and_post_audit_update(self):
        """
        Verify that updating fps_median from audit.json's ai_steady_fps (~10.5 fps)
        does not cause a step-change in ETA.

        In the flawed implementation, preset fps_median was 22.0 (effective FPS),
        causing Episode 1 ETA to be calculated as 16,000 / 22.0 = ~727s (~12.1m).
        When audit.json reported ai_steady_fps = 10.5 (unique FPS), fps_median was
        overwritten to 10.5, causing Episode 2 ETA to jump to 16,000 / 10.5 = ~1524s (~25.4m),
        a sudden +2.1x step change.
        """
        queue = ["samples/sample_ep1.mkv", "samples/sample_ep2.mkv"]
        now = datetime(2026, 11, 14, 22, 0, 0)
        deadline_str = "07:00"

        # Preflight schedule with realigned presets (fps_median=10.5, fps_p5=8.5)
        sched_preflight = compute_overnight_schedule(
            queue=queue,
            deadline_str=deadline_str,
            fps_median=10.5,
            fps_p5=8.5,
            video_overhead_seconds=420.0,
            subtitles_overhead_seconds=300.0,
            start_time=now,
        )

        ep1_preflight = sched_preflight["episodes"][0]
        n_unique = ep1_preflight["n_unique"]
        eta_med_preflight = ep1_preflight["eta_median_seconds"]

        # Simulate post-audit update with measured ai_steady_fps = 10.48 (actual unique inference FPS)
        measured_ai_steady_fps = 10.48
        sched_post_audit = compute_overnight_schedule(
            queue=["samples/sample_ep2.mkv"],
            deadline_str=deadline_str,
            fps_median=measured_ai_steady_fps,
            fps_p5=8.5,
            video_overhead_seconds=420.0,
            subtitles_overhead_seconds=300.0,
            start_time=now,
        )

        ep2_post_audit = sched_post_audit["episodes"][0]
        eta_med_post_audit = ep2_post_audit["eta_median_seconds"]

        # ETA difference should be less than 1% (no 2.1x step-change!)
        pct_diff = abs(eta_med_post_audit - eta_med_preflight) / eta_med_preflight
        self.assertLess(
            pct_diff,
            0.01,
            f"Step-change detected! Preflight ETA={eta_med_preflight}s, Post-audit ETA={eta_med_post_audit}s (diff: {pct_diff*100:.2f}%)"
        )

    def test_decision_table_telemetry_mentions_unique_inference_fps(self):
        """Verify format_schedule_decision_table explicitly cites Unique Inference FPS."""
        sched = compute_overnight_schedule(
            queue=["samples/sample_ep1.mkv"],
            deadline_str="06:00",
            fps_median=10.5,
            fps_p5=8.5,
        )
        table_output = format_schedule_decision_table(sched)
        self.assertIn("Unique Inference FPS", table_output)
        self.assertIn("10.50 fps", table_output)
        self.assertIn("8.50 fps", table_output)

    def test_mathematical_ai_duration_formula(self):
        """
        Verify that ai_seconds is computed as n_unique / fps_unique.
        For 16,000 unique frames and fps_median=10.5:
        ai_seconds = 16000 / 10.5 = 1523.81s (~25.40 min).
        """
        n_unique = 16000
        fps_med = 10.5
        expected_ai_sec = n_unique / fps_med

        # Overhead: 420s video + 300s subs = 720s
        total_overhead = 420.0 + 300.0
        expected_total_med = expected_ai_sec + total_overhead

        with patch("src.scheduler.probe_video", return_value={"nb_frames": 34000}):
            sched = compute_overnight_schedule(
                queue=["ep_synthetic.mkv"],
                deadline_str="08:00",
                fps_median=10.5,
                fps_p5=8.5,
                video_overhead_seconds=420.0,
                subtitles_overhead_seconds=300.0,
            )
            ep = sched["episodes"][0]
            # n_unique is estimated as 34000 * 0.60 = 20400
            self.assertEqual(ep["n_unique"], 20400)
            expected_ai = 20400 / 10.5
            self.assertAlmostEqual(ep["eta_median_seconds"], expected_ai + 720.0, places=1)


if __name__ == "__main__":
    unittest.main()
