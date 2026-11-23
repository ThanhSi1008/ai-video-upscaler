"""Offline unit tests for scheduler checkpoint ETA deduction, preflight hygiene, and preview audit organization."""

from __future__ import annotations

import json
from pathlib import Path

from src.cleanup import organize_output_artifacts
from src.scheduler import BackgroundQueuePreflightManager, _inspect_resumable_checkpoint


def test_organize_output_artifacts_moves_preview_and_custom_stem_audits(tmp_path: Path) -> None:
    """Verify preview and custom out_stem .audit.json files move into audit/ while unrelated files stay untouched."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_audit = out_dir / "Violet_Evergarden_preview_6s_4K.audit.json"
    custom_audit = out_dir / "Custom_Ep01_Master.audit.json"
    other_audit = out_dir / "Another_Episode_4K.audit.json"
    preview_audit.write_text('{"status": "PASS"}', encoding="utf-8")
    custom_audit.write_text('{"status": "PASS"}', encoding="utf-8")
    other_audit.write_text('{"status": "PASS"}', encoding="utf-8")

    moved_prev = organize_output_artifacts(str(out_dir), base_name="Violet_Evergarden")
    assert "Violet_Evergarden_preview_6s_4K.audit.json" in moved_prev
    assert (out_dir / "audit" / "Violet_Evergarden_preview_6s_4K.audit.json").is_file()
    assert other_audit.is_file()

    moved_custom = organize_output_artifacts(
        str(out_dir),
        base_name="Original_Source_Name",
        out_stem="Custom_Ep01_Master",
    )
    assert "Custom_Ep01_Master.audit.json" in moved_custom
    assert (out_dir / "audit" / "Custom_Ep01_Master.audit.json").is_file()
    assert other_audit.is_file()


def test_inspect_resumable_checkpoint_deducts_completed_chunks(tmp_path: Path) -> None:
    """Verify _inspect_resumable_checkpoint deducts completed up_*.mp4 chunks with both plan.json and manifest fallback."""
    out_dir = tmp_path / "out"
    vid = tmp_path / "Mushoku_Ep21.mkv"
    vid.write_bytes(b"\x00" * 128)
    cp_dir = out_dir / ".temp_Mushoku_Ep21"
    up_dir = cp_dir / "upscaled_segs"
    up_dir.mkdir(parents=True, exist_ok=True)
    (up_dir / "up_0000.mp4").write_bytes(b"0" * 16384)
    (up_dir / "up_0001.mp4").write_bytes(b"1" * 16384)
    (cp_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "total_chunks": 4}),
        encoding="utf-8",
    )

    plan = {
        "chunks": [
            {"chunk_idx": 0, "kept_count": 300},
            {"chunk_idx": 1, "kept_count": 300},
            {"chunk_idx": 2, "kept_count": 250},
            {"chunk_idx": 3, "kept_count": 250},
        ]
    }
    done_c, tot_c, rem_frac, rem_u = _inspect_resumable_checkpoint(
        str(vid), str(out_dir), plan, total_frames=2000, n_unique=1100, segment_seconds=24
    )
    assert done_c == 2
    assert tot_c == 4
    assert rem_frac == 0.5
    assert rem_u == 500

    done_m, tot_m, rem_frac_m, rem_u_m = _inspect_resumable_checkpoint(
        str(vid), str(out_dir), None, total_frames=2000, n_unique=1100, segment_seconds=24
    )
    assert done_m == 2
    assert tot_m == 4
    assert rem_frac_m == 0.5
    assert rem_u_m == 550


def test_background_preflight_stop_cleans_unclaimed_zero_chunk_dirs_only(tmp_path: Path) -> None:
    """Verify stop(cleanup_unclaimed=True) removes 0-chunk preflight dirs while preserving real checkpoints."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_unclaimed = str(tmp_path / "Ep02_Unclaimed.mkv")
    vid_claimed = str(tmp_path / "Ep01_Claimed.mkv")
    vid_with_chunks = str(tmp_path / "Ep03_HasChunks.mkv")

    wd_unclaimed = out_dir / ".temp_Ep02_Unclaimed"
    wd_unclaimed.mkdir(parents=True, exist_ok=True)
    (wd_unclaimed / "plan.json").write_text("{}", encoding="utf-8")

    wd_claimed = out_dir / ".temp_Ep01_Claimed"
    wd_claimed.mkdir(parents=True, exist_ok=True)
    (wd_claimed / "plan.json").write_text("{}", encoding="utf-8")

    wd_chunks = out_dir / ".temp_Ep03_HasChunks"
    (wd_chunks / "upscaled_segs").mkdir(parents=True, exist_ok=True)
    (wd_chunks / "upscaled_segs" / "up_0000.mp4").write_bytes(b"x" * 16384)

    mgr = BackgroundQueuePreflightManager([], output_dir=str(out_dir))
    k_unclaimed = mgr._norm_key(vid_unclaimed)
    k_claimed = mgr._norm_key(vid_claimed)
    k_chunks = mgr._norm_key(vid_with_chunks)
    mgr._created_work_dirs[k_unclaimed] = str(wd_unclaimed)
    mgr._created_work_dirs[k_claimed] = str(wd_claimed)
    mgr._created_work_dirs[k_chunks] = str(wd_chunks)
    mgr._claimed_by_main.add(k_claimed)

    mgr.stop(cleanup_unclaimed=True)

    assert not wd_unclaimed.exists()
    assert wd_claimed.is_dir()
    assert wd_chunks.is_dir()
