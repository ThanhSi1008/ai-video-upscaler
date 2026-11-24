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


def test_concat_lists_resolve_relative_paths_and_preview_start_parser(tmp_path: Path, monkeypatch) -> None:
    """Verify relative paths become absolute in concat scripts and --preview-start parses seconds and MM:SS."""
    import numpy as np
    from src.dedup import generate_concat_demuxer_script
    from src.ffmpeg_ops import _apply_symmetric_s5_p010_filter, _write_concat_list
    from upscale import _parse_preview_start_arg

    monkeypatch.chdir(tmp_path)
    rel_frames = Path("sub_cache") / "chunk_out"
    rel_frames.mkdir(parents=True, exist_ok=True)
    c_script = Path("sub_cache") / "concat.txt"
    generate_concat_demuxer_script(
        [{"frame_idx": 0, "count": 2}],
        str(rel_frames),
        str(c_script),
        fps_num=24000,
        fps_den=1001,
        fmt="png",
    )
    script_txt = c_script.read_text(encoding="utf-8")
    assert str((tmp_path / rel_frames / "frame00000001.png").resolve()) in script_txt

    seg_list = _write_concat_list([str(Path("sub_cache") / "up_0000.mp4")], str(Path("sub_cache") / "segs.txt"))
    seg_txt = Path(seg_list).read_text(encoding="utf-8")
    assert str((tmp_path / "sub_cache" / "up_0000.mp4").resolve()) in seg_txt

    assert _parse_preview_start_arg("860.5") == 860.5
    assert _parse_preview_start_arg("14:20") == 860.0
    assert _parse_preview_start_arg("85:30") == 5130.0
    assert _parse_preview_start_arg("01:02:03.5") == 3723.5

    import argparse
    import pytest

    for invalid_arg in (
        "",
        "14:60",
        "14:75",
        "01:60:00",
        "01:02:60",
        "1.5:20",
        "1.5:02:03",
        "01:2.5:03",
        "-1:20",
        "14: 20",
        "25:00:00",
        "1_00",
        "1e2",
        "+10",
        ".5",
        "١٢",
        "nan",
        "inf",
    ):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_preview_start_arg(invalid_arg)

    # Verify directional s=5 filter matches calibrated FFmpeg vf_atadenoise.c (a=s) boundary (25 for heavy Y)
    c_frame = np.full((6, 4), 272 << 6, dtype=np.uint16)
    l1_over = np.full((6, 4), 298 << 6, dtype=np.uint16)  # |298 - 272| = 26 > 25 (excluded)
    r1_ok = np.full((6, 4), 292 << 6, dtype=np.uint16)    # |292 - 272| = 20 <= 25 (included via a=s!)
    out = _apply_symmetric_s5_p010_filter(
        [l1_over, l1_over, c_frame, r1_ok, l1_over],
        [2],
        is_10bit_src=True,
        grain_tier="heavy",
        luma_h=4,
    )[0]
    # Serial a=s includes r1_ok (292) while rejecting l1_over (296): ((272 + 292 + 1) // 2) == 282
    assert int(out[0, 0] >> 6) == 282


def test_night_report_excludes_web_job_mirror_logs_and_lists_sessions(tmp_path: Path) -> None:
    """Verify Night Report ignores mirrored web_job_*.log files and discovers available CSV sessions."""
    from src.night_report import generate_night_report_text, list_night_report_sessions

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ep_log = logs_dir / "Frieren_Ep01_20261124_001000.log"
    web_log = logs_dir / "web_job_20261124_001000_ab12.log"
    sample_body = (
        "# AI Video Upscaler 4K - Execution Log\n"
        "# Started: 2026-11-24 00:10:00\n"
        "# Video: Frieren_Ep01.mkv\n"
        "[2026-11-24 00:10:15] [CHUNK 01/01] Hoàn tất 240 frames (AI GPU: 20.0s @ 12.00 fps | "
        "Encode: 1.0s | Tổng: 21.0s | File: up_0000.mp4 12.0 MB)\n"
        "[2026-11-24 00:10:20] [INFO] Mux Audit: PASS\n"
    )
    ep_log.write_text(sample_body, encoding="utf-8")
    web_log.write_text("# Web Studio Job\n" + sample_body, encoding="utf-8")
    (logs_dir / "soak_telemetry_friday_batch.csv").write_text("timestamp,elapsed_s\n", encoding="utf-8")

    md = generate_night_report_text(session_name="night1", workspace_root=tmp_path)
    assert md.count("### Tập ") == 1
    assert "web_job_" not in md
    sessions = list_night_report_sessions(workspace_root=tmp_path)
    assert "night1" in sessions
    assert "friday_batch" in sessions


def test_pipeline_preview_start_override_preserves_subsecond_precision(tmp_path: Path, monkeypatch) -> None:
    """Verify run_upscale_pipeline preserves sub-second preview_start_sec_override in ffmpeg -ss and live context."""
    import contextlib
    import subprocess
    import pytest
    import src.dedup as dedup_mod
    import src.logger as logger_mod
    import src.models_resolver as mr_mod
    import src.pipeline as pipeline_mod

    iso_root = tmp_path / "subsec_preview_ws"
    (iso_root / "logs").mkdir(parents=True, exist_ok=True)
    (iso_root / "models").mkdir(parents=True, exist_ok=True)
    src_vid = iso_root / "Sample_Ep01.mkv"
    src_vid.write_bytes(b"\x00" * 4096)
    (iso_root / "shows.yaml").write_text("shows: []\n", encoding="utf-8")
    repo_models = Path(__file__).resolve().parents[1] / "models" / "models.yaml"
    (iso_root / "models" / "models.yaml").write_text(repo_models.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(dedup_mod, "ROOT_DIR", str(iso_root))
    monkeypatch.setattr(mr_mod, "ROOT_DIR", str(iso_root))
    monkeypatch.setattr(logger_mod, "LOGS_DIR", str(iso_root / "logs"))
    mr_mod._YAML_STAT_CACHE.clear()

    recorded_cut_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **_kw):
        recorded_cut_cmds.append(list(cmd))
        out_clip = Path(cmd[-1])
        out_clip.parent.mkdir(parents=True, exist_ok=True)
        out_clip.write_bytes(b"\x00" * 2048)
        return subprocess.CompletedProcess(cmd, 0)

    class _StopAfterLiveContext(RuntimeError):
        pass

    @contextlib.contextmanager
    def fake_ramdisk(**_kw):
        ram_d = iso_root / "ram"
        ram_d.mkdir(parents=True, exist_ok=True)
        yield str(ram_d)

    monkeypatch.setattr(pipeline_mod, "validate_environment", lambda **_kw: None)
    monkeypatch.setattr(pipeline_mod, "acquire_gpu_lock", lambda: None)
    monkeypatch.setattr(pipeline_mod, "release_gpu_lock", lambda: None)
    monkeypatch.setattr(pipeline_mod, "run_preflight_checks", lambda **_kw: None)
    monkeypatch.setattr(pipeline_mod, "RAMDisk", fake_ramdisk)
    monkeypatch.setattr(pipeline_mod.subprocess, "run", fake_run)
    fake_run._mock_name = "fake_run"  # type: ignore[attr-defined]
    monkeypatch.setattr(
        pipeline_mod,
        "probe_video",
        lambda _p: {
            "duration": 30.75,
            "nb_frames": 738,
            "fps": 24.0,
            "fps_num": 24,
            "fps_den": 1,
            "width": 1920,
            "height": 1080,
            "size_bytes": 4096,
            "audio_count": 1,
            "subtitle_count": 0,
            "attachment_count": 0,
            "chapter_count": 0,
        },
    )
    monkeypatch.setattr(pipeline_mod, "is_valid_completed_4k_output", lambda _p: False)
    monkeypatch.setattr(
        pipeline_mod,
        "split_video_segments",
        lambda *_a, **_kw: (_ for _ in ()).throw(_StopAfterLiveContext("stop")),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "RollingDedupPlanner",
        lambda *_a, **_kw: (_ for _ in ()).throw(_StopAfterLiveContext("stop")),
    )

    out_dir = iso_root / "out"
    with pytest.raises(_StopAfterLiveContext):
        pipeline_mod.run_upscale_pipeline(
            input_path=str(src_vid),
            output_path=str(out_dir),
            preview_seconds=10,
            preview_start_sec_override=14.625,
            zero_disk=True,
        )

    assert len(recorded_cut_cmds) == 1
    cut_cmd = recorded_cut_cmds[0]
    ss_val = float(cut_cmd[cut_cmd.index("-ss") + 1])
    assert ss_val == pytest.approx(14.625)
    live_snap = logger_mod.read_live_status()
    assert live_snap["preview_start_sec"] == pytest.approx(14.625)
