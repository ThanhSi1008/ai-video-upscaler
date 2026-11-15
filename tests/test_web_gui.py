# -*- coding: utf-8 -*-
"""
tests/test_web_gui.py
Comprehensive offline unit & integration test suite for the Local Real-Time Web Studio:
- 100% isolated in pytest tmp_path (never touches production .env, SQLite caches, or GPU).
- Covers CLI parser, CRLF batch launcher, RunLogger live_status/graceful-pause bridge,
  SafeLocalFSBrowser security boundaries, TelemetryHub SSE 1Hz & dead-PID recovery,
  JobManager queue scheduling & 24/7 normal Subtrans/Upscale/Preview controls,
  Split-Compare frame extraction, validated shows.yaml editor, read-only SQLite overview,
  masked .env API key management, Audit Vault inspector, and static Web UI assets.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from typing import Any, Dict, List

from fastapi.testclient import TestClient
import pytest

import src.logger as logger_mod
from web.app import create_app
from web.fs_browser import SafeLocalFSBrowser, is_secret_filename, natural_sort_key
import web_gui

PNG_1X1_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakePopenProcess:
    """Deterministic in-memory subprocess stub for JobManager tests."""

    def __init__(
        self,
        cmd: List[str],
        pid: int = 42420,
        running: bool = False,
        return_code: int = 0,
        output_lines: List[str] | None = None,
    ) -> None:
        self.cmd = cmd
        self.pid = pid
        self._running = running
        self._return_code = return_code
        self._done = threading.Event()
        if not running:
            self._done.set()
        text = "".join(output_lines or ["[INFO] Simulated job line 1\n", "[CHUNK 01/04] 18.5 fps\n"])
        self.stdout = io.StringIO(text)

    def poll(self) -> int | None:
        return None if self._running else self._return_code

    def wait(self) -> int:
        if self._running:
            self._done.wait(timeout=5.0)
        self._running = False
        return self._return_code

    def terminate(self) -> None:
        self._running = False
        self._return_code = -15
        self._done.set()

    def kill(self) -> None:
        self._running = False
        self._return_code = -9
        self._done.set()


def _seed_workspace(root: Path) -> Path:
    """Create a realistic isolated workspace directory tree inside tmp_path."""
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "subtrans" / ".cache").mkdir(parents=True, exist_ok=True)
    (root / "glossaries").mkdir(parents=True, exist_ok=True)
    (root / "out" / "audit").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / ".cache" / "trt_engines").mkdir(parents=True, exist_ok=True)

    (root / "models" / "models.yaml").write_text(
        """models:
  animevideov3:
    name: RealESRGAN AnimeVideo-v3
    arch:
      type: SRVGGNetCompact
      num_feat: 64
      num_conv: 16
      upscale: 4
    downsample: 2x
    weights_path: models_bench/realesr-animevideov3.pth
    sha256: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    license: BSD-3-Clause
    default: true
  animejanai-hd-v3-ultracompact:
    name: AnimeJaNai V3 HD UltraCompact
    arch:
      type: SRVGGNetCompact
      num_feat: 64
      num_conv: 8
      upscale: 2
    downsample: none
    weights_path: models_bench/animejanai_ultracompact.pth
    sha256: "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff"
    license: CC-BY-NC-4.0
    default: false
""",
        encoding="utf-8",
    )

    (root / "shows.yaml").write_text(
        """shows:
  - pattern: "*[SubsPlease]*"
    model: animejanai-hd-v3-ultracompact
    dedup_policy: default
  - pattern: "*[BD*"
    model: animevideov3
    dedup_policy: dis
""",
        encoding="utf-8",
    )

    (root / "subtrans" / "models.yaml").write_text(
        """chain:
  - gemini-3.5-flash-lite
  - gemini-3.8-flash
models:
  gemini-3.5-flash-lite:
    rpm: 15
    rpd: 500
    tpm: 250000
    status: active
""",
        encoding="utf-8",
    )

    (root / "glossaries" / "frieren.yaml").write_text("terms: []\n", encoding="utf-8")
    (root / ".env").write_text(
        "# Local studio env\nCUSTOM_FLAG=keep_me\nGEMINI_API_KEY_1=AIzaSyTestSecretKeySlotOne9999\n",
        encoding="utf-8",
    )
    (root / ".env.example").write_text("GEMINI_API_KEY_1=\n", encoding="utf-8")

    # Episode files
    anime_dir = root / "anime_season"
    anime_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "[SubsPlease] Frieren - 10 (1080p).mkv",
        "[SubsPlease] Frieren - 02 (1080p).mkv",
        "[BD 1080p] Fate_Ep22.mkv",
    ):
        (anime_dir / name).write_bytes(b"\x00" * 4096)
    (anime_dir / "Frieren_02.ja.srt").write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nこんにちは\n",
        encoding="utf-8",
    )
    return root


def _fake_gpu_metrics() -> Dict[str, Any]:
    return {
        "pstate": "P0",
        "gpu_util_pct": 93.0,
        "mem_util_pct": 44.0,
        "enc_util_pct": 88.0,
        "dec_util_pct": 19.0,
        "mem_clock_mhz": 7501.0,
        "core_clock_mhz": 1882.0,
        "power_w": 156.4,
        "temp_c": 67.0,
        "vram_used_mib": 7840.0,
        "vram_free_mib": 4448.0,
    }


@pytest.fixture()
def studio_env(tmp_path: Path):
    root = _seed_workspace(tmp_path / "workspace")
    spawned_calls: List[Dict[str, Any]] = []
    opened_media: List[tuple[str, bool]] = []
    next_proc_running = {"value": False}

    def fake_popen(cmd: List[str], **kwargs: Any) -> FakePopenProcess:
        proc = FakePopenProcess(
            cmd,
            pid=51000 + len(spawned_calls),
            running=next_proc_running["value"],
        )
        spawned_calls.append({"cmd": cmd, "kwargs": kwargs, "proc": proc})
        return proc

    def fake_open_media(target: Path, prefer_mpv: bool) -> str:
        opened_media.append((str(target), prefer_mpv))
        return "mpv-stub"

    def fake_extract_frame(video_path: Path, ts: float, out_png: Path) -> None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(PNG_1X1_BYTES)

    app = create_app(
        workspace_root=root,
        gpu_query_fn=_fake_gpu_metrics,
        popen_factory=fake_popen,
        open_media_fn=fake_open_media,
        frame_extract_fn=fake_extract_frame,
    )
    with TestClient(app) as client:
        yield {
            "root": root,
            "client": client,
            "spawned": spawned_calls,
            "opened_media": opened_media,
            "next_proc_running": next_proc_running,
        }


# =====================================================================
# 1. CLI PARSER, CRLF LAUNCHER & LOGGER TELEMETRY BRIDGE
# =====================================================================

def test_web_gui_cli_parser_and_crlf_launcher():
    parser = web_gui.build_parser()
    args = parser.parse_args(["--lan", "--port", "9090", "--no-browser", "--log-level", "info"])
    assert args.lan is True
    assert args.port == 9090
    assert args.no_browser is True
    assert args.log_level == "info"

    repo_root = Path(__file__).resolve().parents[1]
    bat_file = repo_root / "web_gui.bat"
    assert bat_file.is_file()
    raw_bytes = bat_file.read_bytes()
    assert b"\r\n" in raw_bytes
    assert b"\n" not in raw_bytes.replace(b"\r\n", b""), "web_gui.bat must strictly use CRLF"

    # Ensure stdlib copies in web.fs_browser stay 100% synced with src.dedup / src.models_resolver
    from src.dedup import DEDUP_POLICY_PARAMS as src_dedup_params
    from src.models_resolver import match_show_pattern as src_match_show_pattern
    from web.fs_browser import (
        DEDUP_POLICY_PARAMS as web_dedup_params,
        match_show_pattern as web_match_show_pattern,
    )

    assert web_dedup_params == src_dedup_params
    for sample_name, pat in (
        ("[SubsPlease] Frieren - 02 (1080p).mkv", "*[SubsPlease]*"),
        ("[BD 1080p] Fate_Ep22.mkv", "*[BD*"),
        ("Solo Leveling S01E01.mkv", "*[SubsPlease]*"),
    ):
        assert web_match_show_pattern(sample_name, pat) is src_match_show_pattern(sample_name, pat)

    # Ensure importing web.app in a fresh interpreter never imports torch / initializes CUDA context
    chk = subprocess.run(
        [sys.executable, "-c", "import sys, web.app; assert 'torch' not in sys.modules"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert chk.returncode == 0, f"web.app unexpectedly imported torch: {chk.stderr}"


def test_run_logger_live_status_and_graceful_pause_bridge(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(logger_mod, "LOGS_DIR", str(tmp_path / "logs"))
    logger_mod.clear_graceful_pause()
    logger_mod.mark_graceful_pause_triggered(False)
    assert logger_mod.is_graceful_pause_requested() is False
    assert logger_mod.was_graceful_paused() is False

    flag_file = logger_mod.request_graceful_pause("unit_test")
    assert Path(flag_file).is_file()
    assert logger_mod.is_graceful_pause_requested() is True
    logger_mod.mark_graceful_pause_triggered(True)
    assert logger_mod.was_graceful_paused() is True
    logger_mod.clear_graceful_pause()
    assert logger_mod.is_graceful_pause_requested() is False

    run_log = logger_mod.RunLogger("Frieren_Ep08")
    run_log.update_live_context(
        model_id="animejanai-hd-v3-ultracompact",
        total_frames=1440,
        unique_frames=900,
        resumed_chunks=[0],
    )
    run_log.set_stage("upscaling", "Chunk 2/4", active_chunk=1)

    dummy_seg = tmp_path / "up_0001.mp4"
    dummy_seg.write_bytes(b"x" * 2048)
    run_log.log_chunk_done(
        chunk_idx=1,
        total_chunks=4,
        frames=360,
        ai_time=18.0,
        ai_fps=20.0,
        enc_time=0.5,
        total_chunk_time=18.5,
        output_seg=str(dummy_seg),
        steady_fps=20.8,
    )

    state = logger_mod.read_live_status()
    assert state["video_name"] == "Frieren_Ep08"
    assert state["stage"] == "upscaling"
    assert state["completed_chunks"] == 2  # resumed chunk 0 + chunk 1
    assert state["total_chunks"] == 4
    assert state["steady_fps"] == 20.8
    assert state["eta_seconds"] == pytest.approx(37.0, rel=1e-2)

    run_log.error("Simulated mux warning")
    assert logger_mod.read_live_status()["last_error"] == "Simulated mux warning"


# =====================================================================
# 2. SAFE FILE SYSTEM BROWSER & SECURITY BOUNDARIES
# =====================================================================

def test_fs_browser_natural_sort_routing_and_secret_protection(studio_env):
    client: TestClient = studio_env["client"]
    root: Path = studio_env["root"]

    assert is_secret_filename(".env") is True
    assert is_secret_filename(".env.production") is True
    assert is_secret_filename(".env.example") is False
    assert sorted(["Ep10.mkv", "Ep2.mkv", "Ep1.mkv"], key=natural_sort_key) == [
        "Ep1.mkv",
        "Ep2.mkv",
        "Ep10.mkv",
    ]

    # Drives listing
    r_drives = client.get("/api/fs/browse?path=drives")
    assert r_drives.status_code == 200
    assert len(r_drives.json()["entries"]) >= 1

    # Folder listing: natural sort (02 before 10) + shows.yaml model routing
    r_dir = client.get("/api/fs/browse", params={"path": "anime_season", "include_subtitles": "true"})
    assert r_dir.status_code == 200
    data = r_dir.json()
    assert data["source_video_count"] == 3
    names = [e["name"] for e in data["entries"]]
    assert names.index("[SubsPlease] Frieren - 02 (1080p).mkv") < names.index(
        "[SubsPlease] Frieren - 10 (1080p).mkv"
    )
    assert "Frieren_02.ja.srt" in names

    by_name = {e["name"]: e for e in data["entries"]}
    assert by_name["[SubsPlease] Frieren - 02 (1080p).mkv"]["routing"]["model_id"] == (
        "animejanai-hd-v3-ultracompact"
    )
    assert by_name["[BD 1080p] Fate_Ep22.mkv"]["routing"]["model_id"] == "animevideov3"
    assert by_name["[BD 1080p] Fate_Ep22.mkv"]["routing"]["dedup_policy"] == "dis"

    # Root listing must never leak .env
    r_root = client.get("/api/fs/browse", params={"path": str(root)})
    assert r_root.status_code == 200
    assert all(e["name"] != ".env" for e in r_root.json()["entries"])

    # Direct .env browse, null-byte, and missing directory errors
    assert client.get("/api/fs/browse", params={"path": ".env"}).status_code == 403
    assert client.get("/api/fs/browse", params={"path": "bad\x00path"}).status_code == 400
    assert client.get("/api/fs/browse", params={"path": "non_existent_folder"}).status_code == 404

    # expand_inputs skips _4K outputs and raises on empty directory
    (root / "anime_season" / "Already_4K.mkv").write_bytes(b"1234")
    fs = SafeLocalFSBrowser(root)
    expanded = fs.expand_inputs(["anime_season"])
    assert len(expanded) == 3
    assert all(not p.endswith("Already_4K.mkv") for p in expanded)

    empty_dir = root / "empty_folder"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="không chứa file video gốc"):
        fs.expand_inputs([str(empty_dir)])


# =====================================================================
# 3. LIVE MONITOR, SSE 1HZ SNAPSHOT & STALE PID CHECKPOINT RECOVERY
# =====================================================================

def test_status_sse_stream_and_dead_pid_checkpoint_detection(studio_env):
    client: TestClient = studio_env["client"]
    root: Path = studio_env["root"]

    # Create a resumable checkpoint + stale live_status from a dead PID
    cp_dir = root / "out" / ".temp_Ep05"
    (cp_dir / "raw_segs").mkdir(parents=True)
    (cp_dir / "upscaled_segs").mkdir(parents=True)
    for i in range(4):
        (cp_dir / "raw_segs" / f"seg_{i:04d}.mp4").write_bytes(b"a" * 512)
    for i in range(2):
        (cp_dir / "upscaled_segs" / f"up_{i:04d}.mp4").write_bytes(b"b" * 1024)
    (cp_dir / "manifest.json").write_text(json.dumps({"model_id": "animevideov3"}), encoding="utf-8")

    (root / "logs" / "live_status.json").write_text(
        json.dumps(
            {
                "video_name": "Ep05",
                "pid": 99999991,
                "stage": "upscaling",
                "total_chunks": 4,
                "completed_chunks": 2,
            }
        ),
        encoding="utf-8",
    )
    (root / "logs" / "latest.log").write_text("[INFO] Booting test log\n[CHUNK 02/04] Done\n", encoding="utf-8")

    r_status = client.get("/api/status")
    assert r_status.status_code == 200
    assert r_status.headers.get("X-Content-Type-Options") == "nosniff"
    assert r_status.headers.get("X-Frame-Options") == "DENY"
    snap = r_status.json()
    assert snap["telemetry"]["gpu_util_pct"] == 93.0
    assert snap["pipeline"]["effective_stage"] == "paused"
    assert len(snap["pipeline"]["checkpoints"]) == 1
    assert snap["pipeline"]["checkpoints"][0]["completed_chunks"] == 2
    assert snap["pipeline"]["checkpoints"][0]["total_chunks"] == 4
    assert any("CHUNK 02/04" in ln for ln in snap["logs"])

    # SSE stream (?once=1)
    with client.stream("GET", "/api/events?once=1") as sse_resp:
        assert sse_resp.status_code == 200
        assert "text/event-stream" in sse_resp.headers["content-type"]
        body = "".join(sse_resp.iter_text())
    assert "event: snapshot" in body
    data_line = next(ln for ln in body.splitlines() if ln.startswith("data: "))
    parsed_sse = json.loads(data_line[len("data: "):])
    assert parsed_sse["telemetry"]["power_w"] == 156.4


# =====================================================================
# 4. QUEUE SCHEDULING & 24/7 UPSCALE / PREVIEW / SUBTRANS CONTROLS
# =====================================================================

def test_queue_schedule_preview_and_job_lifecycle(studio_env):
    client: TestClient = studio_env["client"]
    root: Path = studio_env["root"]
    spawned = studio_env["spawned"]
    next_running = studio_env["next_proc_running"]

    # Empty schedule request returns clean zero summary
    r_empty = client.post("/api/queue/preview-schedule", json={"inputs": []})
    assert r_empty.status_code == 200
    assert r_empty.json()["total_episodes"] == 0

    # Invalid deadline rejected with 400
    r_bad_dl = client.post(
        "/api/queue/preview-schedule",
        json={"inputs": ["anime_season"], "deadline": "99:75"},
    )
    assert r_bad_dl.status_code == 400

    # Mark one episode as already completed (> 1 MiB _4K_animejanai.mkv)
    (root / "out" / "[SubsPlease] Frieren - 02 (1080p)_4K_animejanai.mkv").write_bytes(
        b"\x00" * (1024 * 1024 + 2048)
    )

    r_sched = client.post(
        "/api/queue/preview-schedule",
        json={
            "inputs": ["anime_season"],
            "deadline": "23:59",
            "allow_partial": True,
            "zero_disk": True,
            "use_tensorrt": True,
        },
    )
    assert r_sched.status_code == 200
    sched = r_sched.json()
    assert sched["total_episodes"] == 3
    assert sched["already_done_count"] == 1

    # Launch multi-episode upscale queue
    r_start = client.post(
        "/api/jobs/start",
        json={
            "inputs": ["anime_season"],
            "batch_size": 8,
            "deadline": "07:00",
            "allow_partial": True,
            "dry_run": True,
        },
    )
    assert r_start.status_code == 200
    cmd_upscale = spawned[-1]["cmd"]
    assert "--queue" in cmd_upscale
    assert "--deadline" in cmd_upscale and "07:00" in cmd_upscale
    assert "--allow-partial" in cmd_upscale
    assert "--dry-run" in cmd_upscale

    # Launch 10s preview job (verifies UPSCALER_NO_AUTO_OPEN=1)
    r_prev = client.post(
        "/api/jobs/preview",
        json={"video_path": "anime_season/[BD 1080p] Fate_Ep22.mkv", "seconds": 10},
    )
    assert r_prev.status_code == 200
    assert "--preview" in spawned[-1]["cmd"]
    assert spawned[-1]["kwargs"]["env"].get("UPSCALER_NO_AUTO_OPEN") == "1"

    # Launch Subtrans directly on MKV AND on standalone SRT (works 24/7, no daytime block)
    r_sub_mkv = client.post(
        "/api/jobs/subtrans",
        json={
            "input_path": "anime_season/[BD 1080p] Fate_Ep22.mkv",
            "subs_quality": "high",
            "subs_upgrade": True,
            "dry_run": True,
        },
    )
    assert r_sub_mkv.status_code == 200
    assert "--subs-only" in spawned[-1]["cmd"]
    assert "--subs-upgrade" in spawned[-1]["cmd"]

    r_sub_srt = client.post(
        "/api/jobs/subtrans",
        json={
            "input_path": "anime_season/Frieren_02.ja.srt",
            "subs_quality": "normal",
            "glossary": str(root / "glossaries" / "frieren.yaml"),
            "dry_run": True,
        },
    )
    assert r_sub_srt.status_code == 200
    assert "run" in spawned[-1]["cmd"]
    assert "--glossary" in spawned[-1]["cmd"]

    # Conflict protection (409) when a Web Job is actively running + Pause / Abort
    next_running["value"] = True
    r_live = client.post(
        "/api/jobs/subs-pending",
        json={"subs_quality": "normal", "dry_run": True},
    )
    assert r_live.status_code == 200

    r_conflict = client.post(
        "/api/jobs/subs-pending",
        json={"subs_quality": "normal", "dry_run": True},
    )
    assert r_conflict.status_code == 409

    assert client.post("/api/jobs/pause-graceful").json()["graceful_pause_pending"] is True
    assert (root / "logs" / ".graceful_pause.flag").is_file()
    assert client.post("/api/jobs/clear-pause").json()["graceful_pause_pending"] is False
    assert not (root / "logs" / ".graceful_pause.flag").exists()

    r_abort = client.post("/api/jobs/abort")
    assert r_abort.status_code == 200
    spawned[-1]["proc"].terminate()
    next_running["value"] = False


# =====================================================================
# 5. SPLIT-COMPARE LIGHTBOX & MEDIA PLAYER LAUNCHER
# =====================================================================

def test_preview_candidates_compare_frames_and_open_media(studio_env):
    client: TestClient = studio_env["client"]
    root: Path = studio_env["root"]
    opened_media = studio_env["opened_media"]

    src_vid = root / "Ep01.mkv"
    up_vid = root / "out" / "Ep01_preview_10s_4K.mkv"
    src_vid.write_bytes(b"\x00" * 4096)
    up_vid.write_bytes(b"\x00" * 8192)

    r_cand = client.get("/api/preview/candidates")
    assert r_cand.status_code == 200
    cands = r_cand.json()["candidates"]
    matched = next(c for c in cands if c["name"] == "Ep01_preview_10s_4K.mkv")
    assert matched["is_preview"] is True
    assert matched["preview_seconds"] == 10
    assert Path(matched["source_path"]).resolve() == src_vid.resolve()

    r_cmp = client.post(
        "/api/preview/compare-frames",
        json={
            "source_path": str(src_vid),
            "upscaled_path": str(up_vid),
            "timestamp_sec": 2.5,
        },
    )
    assert r_cmp.status_code == 200
    cmp_data = r_cmp.json()
    r_png = client.get(cmp_data["upscaled_frame_url"])
    assert r_png.status_code == 200
    assert r_png.headers["content-type"] == "image/png"
    assert r_png.content.startswith(b"\x89PNG")

    # Reject invalid frame filename traversal
    assert client.get("/api/preview/frame/bad_name.png").status_code == 400

    # Open media security checks
    assert client.post("/api/open-media", json={"path": ".env"}).status_code == 403
    assert client.post("/api/open-media", json={"path": "shows.yaml"}).status_code == 400
    r_open = client.post("/api/open-media", json={"path": str(up_vid), "prefer_mpv": True})
    assert r_open.status_code == 200
    assert r_open.json()["launcher"] == "mpv-stub"
    assert len(opened_media) == 1


# =====================================================================
# 6. MODELS, SHOWS.YAML VALIDATION, SUBTRANS VAULT & AUDITS
# =====================================================================

def test_models_shows_yaml_validation_subtrans_keys_and_audits(studio_env):
    client: TestClient = studio_env["client"]
    root: Path = studio_env["root"]

    # Models & shows.yaml
    r_ms = client.get("/api/config/models-and-shows")
    assert r_ms.status_code == 200
    assert len(r_ms.json()["models"]) == 2

    # Test routing with custom unsaved YAML
    custom_yaml = """shows:
  - pattern: "*Kimetsu*"
    model: animevideov3
    dedup_policy: sakuga
"""
    r_match = client.post(
        "/api/config/test-match",
        json={"sample_path": "Kimetsu_no_Yaiba_01.mkv", "yaml_content": custom_yaml},
    )
    assert r_match.status_code == 200
    assert r_match.json()["model_id"] == "animevideov3"
    assert r_match.json()["dedup_policy"] == "sakuga"

    # Reject invalid model ID or invalid dedup_policy on save
    assert (
        client.put(
            "/api/config/shows",
            json={"yaml_content": "shows:\n  - pattern: '*A*'\n    model: unknown_model\n"},
        ).status_code
        == 400
    )
    assert (
        client.put(
            "/api/config/shows",
            json={"yaml_content": "shows:\n  - pattern: '*A*'\n    dedup_policy: invalid_pol\n"},
        ).status_code
        == 400
    )

    # Save valid shows.yaml
    r_save = client.put("/api/config/shows", json={"yaml_content": custom_yaml})
    assert r_save.status_code == 200
    assert r_save.json()["rules_count"] == 1
    assert "Kimetsu" in (root / "shows.yaml").read_text(encoding="utf-8")

    # Seed isolated SQLite DB in tmp_path workspace and verify read-only overview + masked keys
    db_path = root / "subtrans" / ".cache" / "translations.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE translations (sig TEXT, pt_date TEXT)")
        conn.execute(
            "CREATE TABLE api_requests (model TEXT, status TEXT, cues_count INT, total_tokens INT, pt_date TEXT)"
        )
        conn.execute("CREATE TABLE key_exhaustion (key_id TEXT, model TEXT, reason TEXT, pt_date TEXT)")
        conn.execute("CREATE TABLE model_exhaustion (model TEXT, exhausted_until_pt TEXT, reason TEXT)")
        conn.execute("INSERT INTO translations VALUES ('s1', '2026-11-15')")
        conn.commit()
    finally:
        conn.close()

    r_sub = client.get("/api/subtrans/overview")
    assert r_sub.status_code == 200
    sub_ov = r_sub.json()
    assert sub_ov["configured_keys_count"] == 1
    # Must NEVER expose raw secret
    raw_json_str = json.dumps(sub_ov)
    assert "AIzaSyTestSecretKeySlotOne9999" not in raw_json_str
    assert sub_ov["key_slots"][0]["masked"] == "AIzaSy...9999"
    assert sub_ov["cache"]["total_cached_cues"] == 1

    # Update API key slot 2 while keeping slot 1 (None) and preserving CUSTOM_FLAG=keep_me
    r_keys = client.put(
        "/api/subtrans/api-keys",
        json={"key_1": None, "key_2": "AIzaSyNewSlotTwoKey8888", "key_3": None},
    )
    assert r_keys.status_code == 200
    assert "AIzaSyNewSlotTwoKey8888" not in json.dumps(r_keys.json())
    env_text = (root / ".env").read_text(encoding="utf-8")
    assert "CUSTOM_FLAG=keep_me" in env_text
    assert "GEMINI_API_KEY_1=AIzaSyTestSecretKeySlotOne9999" in env_text
    assert "GEMINI_API_KEY_2=AIzaSyNewSlotTwoKey8888" in env_text

    # Audits list & detail security checks
    audit_json = root / "out" / "Fate_Ep22_4K.audit.json"
    audit_md = root / "out" / "audit" / "Fate_Ep22.vi.audit.md"
    audit_json.write_text(
        json.dumps(
            {
                "output_file": str(root / "out" / "Fate_Ep22_4K.mkv"),
                "source_file": str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"),
                "status": "PASS",
                "output_sanity": {"status": "PASS"},
                "model_id": "animevideov3",
                "average_pipeline_fps": 10.13,
                "source_frames": 1478,
                "output_frames": 1478,
                "frames_matched": True,
                "audio_tracks_count": 2,
                "subtitle_tracks_count": 2,
                "fonts_count": 5,
                "output_file_size_mb": 124.5,
            }
        ),
        encoding="utf-8",
    )
    audit_md.write_text("# Subtrans Audit PASS\n", encoding="utf-8")

    r_audits = client.get("/api/audits")
    assert r_audits.status_code == 200
    items = r_audits.json()["audits"]
    assert len(items) == 1
    assert items[0]["status"] == "PASS"
    assert items[0]["sub_audit_md_path"] != ""

    r_det_json = client.get("/api/audits/detail", params={"path": str(audit_json)})
    assert r_det_json.status_code == 200
    assert r_det_json.json()["data"]["source_frames"] == 1478

    r_det_md = client.get("/api/audits/detail", params={"path": str(audit_md)})
    assert r_det_md.status_code == 200
    assert "Subtrans Audit PASS" in r_det_md.json()["content"]

    # Reject reading non-audit files or outside-workspace files
    assert client.get("/api/audits/detail", params={"path": ".env"}).status_code == 400
    outside_audit = root.parent / "outside.audit.json"
    outside_audit.write_text("{}", encoding="utf-8")
    assert client.get("/api/audits/detail", params={"path": str(outside_audit)}).status_code == 403


# =====================================================================
# 7. STATIC FRONTEND ASSETS & DOM SINK HYGIENE
# =====================================================================

def test_static_frontend_assets_and_zero_unsafe_dom_sinks(studio_env):
    client: TestClient = studio_env["client"]

    r_html = client.get("/")
    assert r_html.status_code == 200
    html = r_html.text
    for required_token in (
        'meta name="color-scheme" content="light dark"',
        'focusgroup="tablist nomemory"',
        'id="panel-monitor"',
        'id="panel-queue"',
        'id="panel-compare"',
        'id="panel-models"',
        'id="panel-subtrans"',
        'closedby="any"',
        '/static/js/app.js',
    ):
        assert required_token in html

    assert client.get("/static/styles.css").status_code == 200

    js_dir = Path(__file__).resolve().parents[1] / "web" / "static" / "js"
    js_files = sorted(js_dir.glob("*.js"))
    assert len(js_files) >= 6

    forbidden_sinks = (".innerHTML", ".outerHTML", "setHTMLUnsafe", "document.write(", "eval(")
    for jf in js_files:
        r_js = client.get(f"/static/js/{jf.name}")
        assert r_js.status_code == 200
        src_text = jf.read_text(encoding="utf-8")
        for sink in forbidden_sinks:
            assert sink not in src_text, f"Unsafe DOM sink '{sink}' found in {jf.name}"

    for panel_suffix in ("monitor", "queue", "compare", "models", "subtrans"):
        assert f'id="tab-{panel_suffix}" aria-controls="panel-{panel_suffix}"' in html


# =====================================================================
# 8. REGRESSION TESTS FOR REVIEWER-DISCOVERED EDGE CASES
# =====================================================================

def test_graceful_pause_exit_code_one_finalizes_as_paused_and_subtrans_keeps_running_status(tmp_path):
    from web.job_manager import JobManager

    root = _seed_workspace(tmp_path / "pause_ws")
    holder: Dict[str, FakePopenProcess] = {}

    def running_factory(cmd: List[str], **_kwargs: Any) -> FakePopenProcess:
        proc = FakePopenProcess(cmd, pid=51510, running=True, return_code=1)
        holder["proc"] = proc
        return proc

    jm = JobManager(root, popen_factory=running_factory)
    src_ep = str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")
    job = jm.start_upscale_job({"inputs": [src_ep], "dry_run": True})
    assert job["status"] == "running"

    pause_res = jm.request_graceful_pause()
    assert pause_res["graceful_pause_pending"] is True
    assert jm.get_state()["status"] == "pausing"
    assert (root / "logs" / ".graceful_pause.flag").is_file()

    # Simulate upscale.py pausing cleanly at chunk boundary and exiting 1 (unfinished_videos)
    (root / "logs" / "live_status.json").write_text(
        json.dumps({"pid": 51510, "stage": "paused", "video_name": "Fate_Ep22"}),
        encoding="utf-8",
    )
    holder["proc"]._running = False
    holder["proc"]._return_code = 1
    holder["proc"]._done.set()

    for _ in range(40):
        state = jm.get_state()
        if state["status"] not in {"running", "pausing"}:
            break
        threading.Event().wait(0.02)

    state = jm.get_state()
    assert state["status"] == "paused"
    assert state["history"][0]["status"] == "paused"
    assert not (root / "logs" / ".graceful_pause.flag").exists()

    # Real pipeline crash (stage == 'error', exit 1) while pausing must still finalize as 'failed'
    jm.start_upscale_job({"inputs": [src_ep], "dry_run": True})
    jm.request_graceful_pause()
    (root / "logs" / "live_status.json").write_text(
        json.dumps({"pid": 51510, "stage": "error", "last_error": "CUDA OOM"}),
        encoding="utf-8",
    )
    holder["proc"]._running = False
    holder["proc"]._return_code = 1
    holder["proc"]._done.set()
    for _ in range(40):
        if jm.get_state()["status"] not in {"running", "pausing"}:
            break
        threading.Event().wait(0.02)
    assert jm.get_state()["status"] == "failed"

    # Subtrans Web Job is non-chunked: setting graceful pause for external CLI must NOT flip subtrans into 'pausing'
    jm.start_subtrans_job({"input_path": src_ep, "dry_run": True})
    jm.request_graceful_pause()
    assert jm.get_state()["status"] == "running"
    jm.clear_graceful_pause()
    jm.abort_job()


def test_sse_log_cursor_no_duplicate_on_first_tick_and_holds_partial_line(tmp_path):
    from web.sse_hub import TelemetryHub

    root = _seed_workspace(tmp_path / "sse_logs_ws")
    latest_log = root / "logs" / "latest.log"
    seed_lines = [f"[21:10:{i:02d}] [INFO] Seed log #{i + 1}" for i in range(12)]
    latest_log.write_text("\n".join(seed_lines) + "\n", encoding="utf-8")

    hub = TelemetryHub(
        root,
        gpu_query_fn=lambda: {
            "gpu_util": 0.0,
            "mem_util": 0.0,
            "vram_used_mib": 250.0,
            "vram_total_mib": 12288.0,
            "power_w": 15.0,
            "temp_c": 40.0,
            "pstate": "P8",
            "clock_sm_mhz": 300.0,
            "clock_mem_mhz": 405.0,
            "nvenc_util": 0.0,
            "nvdec_util": 0.0,
            "pcie_tx_kb": 0.0,
            "pcie_rx_kb": 0.0,
        },
    )
    snap = hub.build_snapshot()
    assert snap["logs"] == seed_lines
    assert snap["log_cursor"]["offset"] > 0

    # Immediate 1Hz tick must NOT re-emit the 12 lines already delivered in snapshot
    inc_empty, cur_after = hub._read_incremental_logs_with_cursor()
    assert inc_empty == []
    assert cur_after["offset"] == snap["log_cursor"]["offset"]

    # Partial line without trailing '\n' is held until '\n' arrives (never split into 2 lines)
    with latest_log.open("a", encoding="utf-8") as f:
        f.write("[21:10:59] Đang nén HEVC ")
    inc_partial, _ = hub._read_incremental_logs_with_cursor()
    assert inc_partial == []

    with latest_log.open("a", encoding="utf-8") as f:
        f.write("chunk 01/10 (Zero-Disk)\n")
    inc_complete, _ = hub._read_incremental_logs_with_cursor()
    assert inc_complete == ["[21:10:59] Đang nén HEVC chunk 01/10 (Zero-Disk)"]

    # Non-Admin Windows fallback: when latest.log is stale but live_status['log_file'] has newer lines
    ep_log = root / "logs" / "Ep02_20261115_220000.log"
    ep_log.write_text("[22:00:01] Live CLI chunk 1/20\n", encoding="utf-8")
    os.utime(ep_log, (latest_log.stat().st_mtime + 10.0, latest_log.stat().st_mtime + 10.0))
    (root / "logs" / "live_status.json").write_text(
        json.dumps({"pid": os.getpid(), "stage": "upscaling", "log_file": ep_log.name}),
        encoding="utf-8",
    )
    assert hub.read_recent_logs(10) == ["[22:00:01] Live CLI chunk 1/20"]


def test_stale_live_status_isolated_during_subtrans_and_new_upscale_boot(tmp_path):
    from web.sse_hub import TelemetryHub

    root = _seed_workspace(tmp_path / "stale_live_ws")
    (root / "logs" / "live_status.json").write_text(
        json.dumps(
            {
                "pid": 999999,
                "video_name": "OldCompletedEpisode",
                "stage": "completed",
                "stage_detail": "Hoàn tất 4K",
                "total_chunks": 18,
                "completed_chunks": 18,
            }
        ),
        encoding="utf-8",
    )

    state_box: Dict[str, Any] = {
        "status": "running",
        "active_job": {
            "job_id": "sub123",
            "job_type": "subtrans",
            "status": "running",
            "pid": 60601,
            "started_at": "2026-11-15T22:00:00",
            "queue_items": [str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")],
            "options": {"subs_quality": "high"},
        },
        "history": [],
    }
    hub = TelemetryHub(
        root,
        gpu_query_fn=lambda: {
            "gpu_util": 0.0,
            "mem_util": 0.0,
            "vram_used_mib": 200.0,
            "vram_total_mib": 12288.0,
            "power_w": 14.0,
            "temp_c": 39.0,
            "pstate": "P8",
            "clock_sm_mhz": 300.0,
            "clock_mem_mhz": 405.0,
            "nvenc_util": 0.0,
            "nvdec_util": 0.0,
            "pcie_tx_kb": 0.0,
            "pcie_rx_kb": 0.0,
        },
        job_state_provider=lambda: state_box,
    )

    sub_pipe = hub.read_pipeline_state()
    assert sub_pipe["active_source"] == "web"
    assert sub_pipe["effective_stage"] == "subtrans"
    assert sub_pipe["live"]["video_name"] == "[BD 1080p] Fate_Ep22.mkv"
    assert sub_pipe["live"]["total_chunks"] == 0

    # Newly spawned upscale job (PID 60602 != stale PID 999999) before RunLogger initializes
    state_box["active_job"] = {
        "job_id": "up124",
        "job_type": "upscale",
        "status": "running",
        "pid": 60602,
        "started_at": "2026-11-15T22:01:00",
        "queue_items": [str(root / "anime_season" / "[BD 1080p] Gintama_01.mkv")],
        "options": {"model": "animevideov3", "zero_disk": True},
    }
    boot_pipe = hub.read_pipeline_state()
    assert boot_pipe["effective_stage"] == "initializing"
    assert boot_pipe["live"]["pid"] == 60602
    assert boot_pipe["live"]["video_name"] == "[BD 1080p] Gintama_01.mkv"


def test_run_logger_stage_guard_records_error_and_paused_states(tmp_path, monkeypatch):
    logs_dir = tmp_path / "guard_logs"
    monkeypatch.setattr(logger_mod, "LOGS_DIR", str(logs_dir))

    rl = logger_mod.RunLogger("GuardTestEp")
    rl.set_stage("upscaling", "Chunk 3/10", active_chunk=2)

    with pytest.raises(RuntimeError, match="FFmpeg pipe broken"):
        with rl.stage_guard():
            raise RuntimeError("FFmpeg pipe broken")

    saved = json.loads(Path(logger_mod.get_live_status_path()).read_text(encoding="utf-8"))
    assert saved["stage"] == "error"
    assert "FFmpeg pipe broken" in saved["last_error"]

    class FailedResumableError(RuntimeError):
        pass

    rl2 = logger_mod.RunLogger("GuardResumableEp")
    with pytest.raises(FailedResumableError):
        with rl2.stage_guard():
            raise FailedResumableError("RAMDisk ENOSPC at chunk 4")

    saved2 = json.loads(Path(logger_mod.get_live_status_path()).read_text(encoding="utf-8"))
    assert saved2["stage"] == "paused"
    assert "ENOSPC" in saved2["last_error"]


def test_real_ffmpeg_bt709_compare_frame_extraction(tmp_path):
    import shutil

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg binary not found in PATH")

    root = _seed_workspace(tmp_path / "real_ffmpeg_ws")
    src_vid = root / "anime_season" / "clip_src.mkv"
    up_vid = root / "out" / "clip_src_4K.mkv"

    for target, dims in ((src_vid, "160x120"), (up_vid, "320x240")):
        res = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={dims}:rate=24:duration=1",
                "-pix_fmt",
                "yuv420p10le",
                "-c:v",
                "libx265",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-color_range",
                "tv",
                str(target),
            ],
            capture_output=True,
            timeout=25.0,
            check=False,
        )
        if res.returncode != 0:
            pytest.skip(f"ffmpeg libx265 testsrc unavailable: {res.stderr.decode(errors='ignore')[:120]}")

    # Instantiate create_app WITHOUT frame_extract_fn mock so _default_extract_frame & PyAV probe run for real
    app = create_app(
        workspace_root=root,
        gpu_query_fn=lambda: {
            "gpu_util": 0.0,
            "mem_util": 0.0,
            "vram_used_mib": 200.0,
            "vram_total_mib": 12288.0,
            "power_w": 12.0,
            "temp_c": 38.0,
            "pstate": "P8",
            "clock_sm_mhz": 300.0,
            "clock_mem_mhz": 405.0,
            "nvenc_util": 0.0,
            "nvdec_util": 0.0,
            "pcie_tx_kb": 0.0,
            "pcie_rx_kb": 0.0,
        },
    )
    with TestClient(app) as client:
        r_cmp = client.post(
            "/api/preview/compare-frames",
            json={
                "source_path": str(src_vid),
                "upscaled_path": str(up_vid),
                "timestamp_sec": 0.25,
                "source_offset_sec": 0.0,
            },
        )
        assert r_cmp.status_code == 200, r_cmp.text
        payload = r_cmp.json()
        assert payload["source_meta"]["width"] == 160
        assert payload["upscaled_meta"]["width"] == 320

        r_png_src = client.get(payload["source_frame_url"])
        r_png_up = client.get(payload["upscaled_frame_url"])
        assert r_png_src.status_code == 200
        assert r_png_up.status_code == 200
        assert r_png_src.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert r_png_up.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_tabpanel_pointer_hit_testing_css_and_dialog_return_value_hygiene(studio_env):
    """Guard against hidden='until-found' tabpanels or Top-Layer popovers occluding pointer hits,
    and ensure modal <dialog> close handlers clear stale returnValue across sessions.
    """
    import re

    client: TestClient = studio_env["client"]
    css = client.get("/static/styles.css").text
    fs_picker_js = client.get("/static/js/fs_picker.js").text
    panels_js = client.get("/static/js/panels.js").text
    ui_js = client.get("/static/js/ui.js").text

    # 1. .tabpanels-stack must align items to start so hidden="until-found" boxes do not stretch full height
    stack_block = re.search(r"\.tabpanels-stack\s*\{([^}]*)\}", css, re.DOTALL)
    assert stack_block is not None
    assert "align-items: start" in stack_block.group(1)

    # 2. Hidden tabpanels must disable pointer-events without breaking Find-in-Page (no display:none / visibility:hidden)
    hidden_panel_block = re.search(r'\[role="tabpanel"\]\[hidden\]\s*\{([^}]*)\}', css, re.DOTALL)
    assert hidden_panel_block is not None
    hidden_panel_decl = hidden_panel_block.group(1)
    assert "pointer-events: none" in hidden_panel_decl
    assert "display:" not in hidden_panel_decl
    assert "visibility:" not in hidden_panel_decl

    # Active tabpanel must stack strictly above hidden siblings in grid-area 1 / 1
    active_panel_block = re.search(
        r'\[role="tabpanel"\]:not\(\[hidden\]\)\s*\{([^}]*)\}', css, re.DOTALL
    )
    assert active_panel_block is not None
    assert "z-index: 1" in active_panel_block.group(1)

    # 3. Boolean [hidden] must override author display (.btn, etc.) while excluding [hidden="until-found"]
    bool_hidden_block = re.search(
        r'\[hidden\]:not\(\[hidden="until-found"\]\)\s*\{([^}]*)\}', css, re.DOTALL
    )
    assert bool_hidden_block is not None
    assert "display: none" in bool_hidden_block.group(1)

    # 4. Top-Layer [popover] toasts must use fixed viewport coordinates + --toast-offset (never static at 0,0)
    toast_block = re.search(r"\.toast-item\[popover\]\s*\{([^}]*)\}", css, re.DOTALL)
    assert toast_block is not None
    assert "position: fixed" in toast_block.group(1)
    assert "--toast-offset" in toast_block.group(1)
    assert "relayoutToastStack" in ui_js

    # 5. Both modal dialogs must clear dialog.returnValue = "" so Esc / backdrop light-dismiss never leaks confirm
    assert 'dialog.returnValue = ""' in fs_picker_js
    assert 'auditDialog.returnValue = ""' in panels_js
    assert "pointerdown" in ui_js and "pointerup" in ui_js


def test_audits_mux_frame_count_fallback_and_subdir_discovery(tmp_path):
    """Ensure /api/audits falls back to mux_audit.video_frame_count when top-level frame counts are 0,
    and both /api/audits and /api/preview/candidates discover 1-level out/<subdir>/ outputs.
    """
    root = _seed_workspace(tmp_path / "subdir_audit_ws")
    season_out = root / "out" / "Season1"
    season_out.mkdir(parents=True, exist_ok=True)

    src_ep = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    out_ep = season_out / "Climax_Run.mkv"
    out_ep.write_bytes(b"\x00" * 4096)

    audit_path = season_out / "Climax_Run.audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "output_file": str(out_ep),
                "source_file": str(src_ep),
                "status": "PASS",
                "source_frames": 0,
                "output_frames": 0,
                "frames_matched": False,
                "mux_audit": {
                    "status": "PASS",
                    "video_frame_count": {
                        "passed": False,
                        "source_frames": 4374,
                        "output_frames": 4372,
                    },
                    "audio_streams": {"source_count": 1, "output_count": 1},
                    "subtitle_streams": {"source_count": 2, "output_count": 2},
                    "font_attachments": {"source_count": 4, "output_count": 4},
                },
                "output_sanity": {"status": "PASS"},
            }
        ),
        encoding="utf-8",
    )

    app = create_app(
        workspace_root=root,
        gpu_query_fn=lambda: {
            "gpu_util": 0.0,
            "mem_util": 0.0,
            "vram_used_mib": 200.0,
            "vram_total_mib": 12288.0,
            "power_w": 12.0,
            "temp_c": 38.0,
            "pstate": "P8",
            "clock_sm_mhz": 300.0,
            "clock_mem_mhz": 405.0,
            "nvenc_util": 0.0,
            "nvdec_util": 0.0,
            "pcie_tx_kb": 0.0,
            "pcie_rx_kb": 0.0,
        },
    )
    with TestClient(app) as client:
        r_aud = client.get("/api/audits")
        assert r_aud.status_code == 200
        audits = r_aud.json()["audits"]
        assert len(audits) == 1
        item = audits[0]
        assert item["episode"] == "Season1/Climax_Run"
        assert item["source_frames"] == 4374
        assert item["output_frames"] == 4372
        assert item["frames_matched"] is False
        assert item["audio_tracks_count"] == 1
        assert item["subtitle_tracks_count"] == 2
        assert item["fonts_count"] == 4

        r_cand = client.get("/api/preview/candidates")
        assert r_cand.status_code == 200
        cands = r_cand.json()["candidates"]
        matched = next(c for c in cands if c["name"] == "Season1/Climax_Run.mkv")
        assert Path(matched["source_path"]).resolve() == src_ep.resolve()


def test_dialog_reflow_async_races_checkpoint_focus_and_toast_clamping_guards(studio_env):
    """Guard against compact-viewport dialog clipping, stale async modal overwrites,
    right-click backdrop dismiss, 1Hz checkpoint focus loss, and Top-Layer toast overflow.
    """
    import re

    client: TestClient = studio_env["client"]
    css = client.get("/static/styles.css").text
    fs_picker_js = client.get("/static/js/fs_picker.js").text
    panels_js = client.get("/static/js/panels.js").text
    ui_js = client.get("/static/js/ui.js").text
    app_js = client.get("/static/js/app.js").text

    # 1. Modal dialog[open] must allow vertical scroll fallback (never overflow: clip that cuts off footer buttons)
    dlg_open_block = re.search(r"\.studio-dialog\[open\]\s*\{([^}]*)\}", css, re.DOTALL)
    assert dlg_open_block is not None
    assert "overflow-y: auto" in dlg_open_block.group(1)
    assert "overflow: clip" not in dlg_open_block.group(1)
    assert "#fs-picker-dialog[open]" in css

    # 2. Async browse() and loadDoc() sequence guards + aria-pressed state on audit switcher
    assert "browseSeq" in fs_picker_js
    assert "isCurrent()" in fs_picker_js
    assert "auditLoadSeq" in panels_js
    assert '"aria-pressed"' in panels_js

    # 3. 1Hz SSE tick must guard checkpoint-list rebuilds via signature so focus/clicks survive ticks
    assert "lastCheckpointSignature" in app_js

    # 4. Light-dismiss fallback must ignore non-primary mouse buttons (right-click) & toast stack must cap height
    assert "ev.button !== 0" in ui_js
    assert "--toast-max-h" in css and "--toast-max-h" in ui_js
    assert "onbeforematch" in ui_js



