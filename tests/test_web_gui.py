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


def _idle_gpu_metrics() -> Dict[str, Any]:
    return {
        "pstate": "P8",
        "gpu_util_pct": 0.0,
        "mem_util_pct": 0.0,
        "enc_util_pct": 0.0,
        "dec_util_pct": 0.0,
        "mem_clock_mhz": 405.0,
        "core_clock_mhz": 300.0,
        "power_w": 14.0,
        "temp_c": 39.0,
        "vram_used_mib": 200.0,
        "vram_free_mib": 12088.0,
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

    # Dynamic dis-<thr> and drift-<thr> dedup policies are accepted everywhere
    dynamic_yaml = """shows:
  - pattern: "*Frieren*"
    model: animevideov3
    dedup_policy: dis-0.25
  - pattern: "*Sakamoto*"
    model: animejanai-hd-v3-ultracompact
    dedup_policy: drift-0.6
"""
    r_dyn_match = client.post(
        "/api/config/test-match",
        json={"sample_path": "Frieren_Ep01.mkv", "yaml_content": dynamic_yaml},
    )
    assert r_dyn_match.status_code == 200
    assert r_dyn_match.json()["dedup_policy"] == "dis-0.25"

    r_dyn_save = client.put("/api/config/shows", json={"yaml_content": dynamic_yaml})
    assert r_dyn_save.status_code == 200
    assert r_dyn_save.json()["rules_count"] == 2

    from web.fs_browser import SafeLocalFSBrowser
    browser = SafeLocalFSBrowser(root)
    routed_drift = browser.resolve_routing_for_path("Sakamoto_Ep02.mkv")
    assert routed_drift["model_id"] == "animejanai-hd-v3-ultracompact"
    assert routed_drift["dedup_policy"] == "drift-0.6"

    # Reject invalid model ID or invalid dedup_policy on save (including non-positive / non-finite thresholds)
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
            json={"yaml_content": "shows:\n  - pattern: '*A*'\n    dedup_policy: dis-0\n"},
        ).status_code
        == 400
    )
    assert (
        client.put(
            "/api/config/shows",
            json={"yaml_content": "shows:\n  - pattern: '*A*'\n    dedup_policy: drift-nan\n"},
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
        gpu_query_fn=_idle_gpu_metrics,
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
        gpu_query_fn=_idle_gpu_metrics,
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
        gpu_query_fn=_idle_gpu_metrics,
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
        gpu_query_fn=_idle_gpu_metrics,
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


# =====================================================================
# 8. DEEP AUDIT REGRESSION COVERAGE (ALL 5 DEFECT CATEGORIES)
# =====================================================================


def test_env_api_keys_crlf_injection_key1_sync_and_precedence(tmp_path: Path, monkeypatch):
    from web.app import _write_dotenv_keys

    monkeypatch.delenv("SUBTRANS_DISABLE_DOTENV", raising=False)
    for var in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        monkeypatch.delenv(var, raising=False)

    root = _seed_workspace(tmp_path / "env_sync_ws")
    (root / ".env").write_text(
        "CUSTOM_FLAG=keep_me\n"
        "GEMINI_API_KEY=AIzaSyLegacySingleKey0000\n"
        "GEMINI_API_KEYS=AIzaSyListSlotOne1111;AIzaSyListSlotTwo2222\n"
        "GEMINI_API_KEY_1=AIzaSyOldSlotOne0001\n",
        encoding="utf-8",
    )

    # Direct writer must reject CR/LF/NUL injection
    with pytest.raises(ValueError):
        _write_dotenv_keys(root / ".env", {"GEMINI_API_KEY_1": "AIzaSyEvil\nINJECTED=1"})

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        # Pydantic validator must return 422 on CR/LF/NUL
        for bad_val in ("AIza\nBAD=1", "AIza\rBAD=1", "AIza\x00BAD"):
            r_bad = client.put("/api/subtrans/api-keys", json={"key_1": bad_val})
            assert r_bad.status_code == 422

        # Updating key_1 synchronizes legacy GEMINI_API_KEY and unwraps GEMINI_API_KEYS cleanly
        r_upd = client.put("/api/subtrans/api-keys", json={"key_1": "AIzaSyFreshSlotOne9999"})
        assert r_upd.status_code == 200
        env_after_upd = (root / ".env").read_text(encoding="utf-8")
        assert "AIzaSyLegacySingleKey0000" not in env_after_upd
        assert "GEMINI_API_KEYS=" not in env_after_upd
        assert "GEMINI_API_KEY=AIzaSyFreshSlotOne9999" in env_after_upd
        assert "GEMINI_API_KEY_1=AIzaSyFreshSlotOne9999" in env_after_upd
        assert "GEMINI_API_KEY_2=AIzaSyListSlotTwo2222" in env_after_upd
        assert "CUSTOM_FLAG=keep_me" in env_after_upd

        # Clearing key_1 deletes both GEMINI_API_KEY_1 and synced GEMINI_API_KEY without touching slot 2
        r_clr = client.put("/api/subtrans/api-keys", json={"key_1": ""})
        assert r_clr.status_code == 200
        assert r_clr.json()["configured_count"] == 1
        env_after_clr = (root / ".env").read_text(encoding="utf-8")
        assert "GEMINI_API_KEY=" not in env_after_clr
        assert "GEMINI_API_KEY_1=" not in env_after_clr
        assert "GEMINI_API_KEY_2=AIzaSyListSlotTwo2222" in env_after_clr

        # os.environ takes precedence over .env (load_dotenv override=False parity) + comma/semicolon list
        monkeypatch.setenv("GEMINI_API_KEYS", "AIzaSyEnvCommaOne1111,AIzaSyEnvCommaTwo2222")
        r_ov = client.get("/api/subtrans/overview")
        assert r_ov.status_code == 200
        ov = r_ov.json()
        raw_dump = json.dumps(ov)
        assert "AIzaSyEnvCommaOne1111" not in raw_dump
        assert "AIzaSyEnvCommaTwo2222" not in raw_dump
        assert "AIzaSyListSlotTwo2222" not in raw_dump
        slots = {s["slot"]: s for s in ov["key_slots"]}
        assert slots[1]["configured"] is True
        assert slots[1]["source"] == "env"
        assert slots[1]["masked"] == "AIzaSy...1111"
        assert slots[2]["configured"] is True
        assert slots[2]["source"] == "env"
        assert slots[2]["masked"] == "AIzaSy...2222"
        # Pool holds 2 keys from env GEMINI_API_KEYS + distinct slot-2 key still configured in .env/env
        assert ov["configured_keys_count"] == 3

        # Clearing slot 2 leaves strictly the 2 keys from os.environ GEMINI_API_KEYS
        assert client.put("/api/subtrans/api-keys", json={"key_2": ""}).status_code == 200
        assert client.get("/api/subtrans/overview").json()["configured_keys_count"] == 2


def test_env_api_keys_preserves_distinct_key1_when_editing_key3(tmp_path: Path, monkeypatch):
    """Verify editing key_3 preserves distinct GEMINI_API_KEY and GEMINI_API_KEY_1 entries."""
    monkeypatch.delenv("SUBTRANS_DISABLE_DOTENV", raising=False)
    for var in (
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
    ):
        monkeypatch.delenv(var, raising=False)

    root = _seed_workspace(tmp_path / "env_distinct_key1_ws")
    (root / ".env").write_text(
        "GEMINI_API_KEY=AIzaSyPrimaryKey0001\n"
        "GEMINI_API_KEY_1=AIzaSySecondKey0002\n",
        encoding="utf-8",
    )

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        r_put = client.put("/api/subtrans/api-keys", json={"key_3": "AIzaSyThirdKey0003"})
        assert r_put.status_code == 200
        assert r_put.json()["configured_count"] == 3

        r_ov = client.get("/api/subtrans/overview")
        assert r_ov.status_code == 200
        assert r_ov.json()["configured_keys_count"] == 3

    env_text = (root / ".env").read_text(encoding="utf-8")
    assert "AIzaSyPrimaryKey0001" in env_text
    assert "AIzaSySecondKey0002" in env_text
    assert "AIzaSyThirdKey0003" in env_text


def test_dns_rebinding_host_origin_csp_multidrive_nan_nul_and_corrupt_audit(tmp_path: Path):
    import base64
    import hashlib
    import re

    root = _seed_workspace(tmp_path / "sec_ws")

    # Localhost-only mode (allow_lan=False)
    app_local = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, allow_lan=False)
    with TestClient(app_local) as client:
        # Strict anti-DNS-rebinding Host/Origin checks
        assert client.get("/api/status", headers={"Host": "evil.attacker.com:8765"}).status_code == 403
        assert (
            client.get(
                "/api/status",
                headers={"Host": "127.0.0.1:8765", "Origin": "https://evil.attacker.com"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/config/test-match",
                json={"sample_path": "a.mkv"},
                headers={"Host": "127.0.0.1:8765", "Origin": "null"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/config/test-match",
                json={"sample_path": "a.mkv"},
                headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:9999"},
            ).status_code
            == 403
        )
        assert (
            client.get(
                "/api/status",
                headers={"Host": "localhost:8765", "Origin": "http://localhost:8765"},
            ).status_code
            == 200
        )

        # Strict CSP + SHA-256 hash matches exact inline FOUC script and zero inline style= attrs
        r_html = client.get("/")
        assert r_html.status_code == 200
        csp = r_html.headers.get("content-security-policy", "")
        assert "default-src 'self'" in csp
        assert "style-src 'self'" in csp
        assert "'unsafe-inline'" not in csp
        assert "img-src 'self' data:" in csp
        script_match = re.search(r"<script>([\s\S]*?)</script>", r_html.text)
        assert script_match is not None
        digest = base64.b64encode(hashlib.sha256(script_match.group(1).encode("utf-8")).digest()).decode("ascii")
        assert f"'sha256-{digest}'" in csp
        assert re.search(r"\sstyle\s*=", r_html.text) is None

        # Multi-drive unrestricted Windows browsing / open-media / compare-frames outside workspace_root
        ext_dir = tmp_path / "external_drive_D" / "Anime"
        ext_dir.mkdir(parents=True, exist_ok=True)
        ext_src = ext_dir / "External_Ep01.mkv"
        ext_4k = ext_dir / "External_Ep01_4K.mkv"
        ext_src.write_bytes(b"\x00" * 1024)
        ext_4k.write_bytes(b"\x00" * 1024)
        (ext_dir / ".env").write_text("SECRET=hidden\n", encoding="utf-8")

        r_browse = client.get("/api/fs/browse", params={"path": str(ext_dir)})
        assert r_browse.status_code == 200
        names = [e["name"] for e in r_browse.json()["entries"]]
        assert "External_Ep01.mkv" in names
        assert ".env" not in names

        # Null byte & NaN/Infinity guards -> 400/422 cleanly (never 500)
        assert client.get("/api/fs/browse", params={"path": "C:/anime/\x00bad"}).status_code == 400
        assert client.get("/api/audits/detail", params={"path": "out/x\x00.audit.json"}).status_code == 400
        assert client.post("/api/config/test-match", json={"sample_path": "a\x00b.mkv"}).status_code == 422
        r_nan = client.post(
            "/api/preview/compare-frames",
            content=json.dumps({"source_path": str(ext_src), "upscaled_path": str(ext_4k), "timestamp_sec": float("nan")}),
            headers={"Content-Type": "application/json"},
        )
        assert r_nan.status_code == 422
        r_inf = client.post(
            "/api/preview/compare-frames",
            content=json.dumps({"source_path": str(ext_src), "upscaled_path": str(ext_4k), "source_offset_sec": float("inf")}),
            headers={"Content-Type": "application/json"},
        )
        assert r_inf.status_code == 422

        # Corrupt / non-decodable MKV returns clean 422 diagnostic instead of HTTP 500
        r_corrupt_vid = client.post(
            "/api/preview/compare-frames",
            json={"source_path": str(ext_src), "upscaled_path": str(ext_4k), "timestamp_sec": 0.0},
        )
        assert r_corrupt_vid.status_code == 422

        # Corrupt .audit.json surfaced cleanly in list & 422 raw_preview in detail
        bad_audit = root / "out" / "Broken_Ep09_4K.audit.json"
        bad_audit.write_text("{corrupt_json_missing_quote: [1, 2", encoding="utf-8")
        r_audits = client.get("/api/audits")
        assert r_audits.status_code == 200
        corrupt_items = [a for a in r_audits.json()["audits"] if a["episode"] == "Broken_Ep09"]
        assert len(corrupt_items) == 1
        assert corrupt_items[0]["status"] == "CORRUPT"
        assert corrupt_items[0]["corrupt"] is True
        assert corrupt_items[0]["error"]

        r_det = client.get("/api/audits/detail", params={"path": str(bad_audit)})
        assert r_det.status_code == 422
        det_json = r_det.json()
        assert det_json["corrupt"] is True
        assert "{corrupt_json_missing_quote" in det_json["raw_preview"]

    # Open-media and compare-frames outside workspace_root succeed with stubbed handlers
    opened: List[Path] = []
    app_ext = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        open_media_fn=lambda p, _m: opened.append(p) or "mpv",
        frame_extract_fn=lambda _v, _t, out: out.parent.mkdir(parents=True, exist_ok=True) or out.write_bytes(PNG_1X1_BYTES),
    )
    with TestClient(app_ext) as client_ext:
        assert client_ext.post("/api/open-media", json={"path": str(ext_src)}).status_code == 200
        assert opened and opened[0] == ext_src.resolve()
        assert (
            client_ext.post(
                "/api/preview/compare-frames",
                json={"source_path": str(ext_src), "upscaled_path": str(ext_4k), "timestamp_sec": 0.0},
            ).status_code
            == 200
        )

    # LAN mode (--lan -> allow_lan=True): allows private LAN IP & .local but still blocks public attacker host
    lan_args = web_gui.build_parser().parse_args(["--lan"])
    assert lan_args.lan is True
    app_lan = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, allow_lan=True)
    with TestClient(app_lan) as client_lan:
        assert (
            client_lan.get(
                "/api/status",
                headers={"Host": "192.168.1.42:8765", "Origin": "http://192.168.1.42:8765"},
            ).status_code
            == 200
        )
        assert (
            client_lan.get(
                "/api/status",
                headers={"Host": "studio-box.local:8765", "Origin": "http://studio-box.local:8765"},
            ).status_code
            == 200
        )
        assert client_lan.get("/api/status", headers={"Host": "rebind.evil.org:8765"}).status_code == 403


def test_job_manager_concurrency_drain_validation_cli_parity_and_abort(tmp_path: Path, monkeypatch):
    import time
    import upscale
    from web.job_manager import JobManager

    root = _seed_workspace(tmp_path / "jm_parity_ws")
    ep1 = root / "anime_season" / "[SubsPlease] Frieren - 02 (1080p).mkv"
    ep2 = root / "anime_season" / "[SubsPlease] Frieren - 10 (1080p).mkv"

    # 1. TOCTOU concurrent launch lock: only 1 process spawns under concurrent start calls
    spawn_calls: List[List[str]] = []
    barrier = threading.Barrier(2)

    def slow_popen(cmd: List[str], **_kwargs: Any) -> FakePopenProcess:
        time.sleep(0.08)
        proc = FakePopenProcess(cmd, pid=71000 + len(spawn_calls), running=True)
        spawn_calls.append(cmd)
        return proc

    jm_race = JobManager(root, popen_factory=slow_popen)
    results: List[str] = []

    def worker() -> None:
        barrier.wait(timeout=3.0)
        try:
            jm_race.start_upscale_job({"inputs": [str(ep1)], "dry_run": True})
            results.append("ok")
        except RuntimeError:
            results.append("conflict")

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert sorted(results) == ["conflict", "ok"]
    assert len(spawn_calls) == 1
    jm_race.abort_job()

    # 2. Failed popen_factory / None stdout pipe must never leave job stuck in 'running'
    def failing_popen(_cmd: List[str], **_kwargs: Any) -> FakePopenProcess:
        raise OSError("simulated spawn error")

    jm_fail = JobManager(root, popen_factory=failing_popen)
    with pytest.raises(OSError):
        jm_fail.start_upscale_job({"inputs": [str(ep1)]})
    assert jm_fail.get_state()["status"] in {"idle", "failed", "error"}

    class _NoStdoutProc(FakePopenProcess):
        def __init__(self, cmd: List[str]) -> None:
            super().__init__(cmd, pid=71999, running=False, return_code=2)
            self.stdout = None  # type: ignore[assignment]

    jm_nostdout = JobManager(root, popen_factory=lambda c, **_kw: _NoStdoutProc(c))
    jm_nostdout.start_upscale_job({"inputs": [str(ep1)], "dry_run": True})
    for _ in range(40):
        if jm_nostdout.get_state()["status"] != "running":
            break
        time.sleep(0.025)
    assert jm_nostdout.get_state()["status"] == "failed"

    # 3. CLI parity in start_subtrans_job: --glossary on video container & multi-video directory queue
    captured_cmds: List[List[str]] = []
    jm_cli = JobManager(
        root,
        popen_factory=lambda cmd, **_kw: captured_cmds.append(cmd) or FakePopenProcess(cmd, pid=72000, running=False),
    )
    jm_cli.start_subtrans_job({"input_path": str(ep1), "glossary": "glossaries/frieren.yaml", "dry_run": True})
    assert "--subs-only" in captured_cmds[-1]
    assert "--glossary" in captured_cmds[-1]
    time.sleep(0.05)

    # Multi-video directory must include multiple mkvs via --queue (never silently truncated to items[0])
    dir_only = root / "multi_vid_dir"
    dir_only.mkdir()
    (dir_only / "Ep01.mkv").write_bytes(b"\x00" * 512)
    (dir_only / "Ep02.mkv").write_bytes(b"\x00" * 512)
    jm_cli.start_subtrans_job({"input_path": str(dir_only), "dry_run": True})
    cmd_dir = captured_cmds[-1]
    assert "--subs-only" in cmd_dir and "--queue" in cmd_dir
    assert str((dir_only / "Ep01.mkv").resolve()) in cmd_dir
    assert str((dir_only / "Ep02.mkv").resolve()) in cmd_dir
    time.sleep(0.05)

    # upscale.py CLI parser accepts --glossary cleanly
    parsed = upscale.build_arg_parser().parse_args(["--subs-only", "--glossary", "glossaries/frieren.yaml", str(ep1)])
    assert parsed.subs_only == "__INPUT__"
    assert parsed.glossary == "glossaries/frieren.yaml"

    # 4. abort_job() external .gpu_active.lock status reporting (non-misleading ok=False)
    assert jm_cli.abort_job()["status"] == "idle"
    lock_file = root / ".gpu_active.lock"
    lock_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        ext_res = jm_cli.abort_job()
        assert ext_res["ok"] is False
        assert ext_res["status"] == "external_lock"
        assert ext_res["aborted_pid"] is None
        assert ext_res["external_lock_pid"] == os.getpid()
    finally:
        lock_file.unlink(missing_ok=True)

    # 5. compute_schedule_preview: workspace_root relative output_dir, explicit .mkv target, zero_disk=False FPS penalty
    alt_cwd = tmp_path / "elsewhere_cwd"
    alt_cwd.mkdir()
    monkeypatch.chdir(alt_cwd)

    custom_out_dir = root / "custom_rel_out"
    custom_out_dir.mkdir()
    (custom_out_dir / f"{ep1.stem}_4K_animejanai.mkv").write_bytes(b"\x00" * (1024 * 1024 + 64))

    sched_rel = jm_cli.compute_schedule_preview(
        inputs=[str(ep1)],
        output_dir="custom_rel_out",
        zero_disk=True,
        use_tensorrt=False,
    )
    assert sched_rel["episodes"][0]["already_done"] is True

    explicit_out = root / "custom_rel_out" / "explicit_target_4K.mkv"
    explicit_out.write_bytes(b"\x00" * (1024 * 1024 + 64))
    sched_explicit = jm_cli.compute_schedule_preview(
        inputs=[str(ep2)],
        output_dir=str(explicit_out),
        zero_disk=True,
        use_tensorrt=False,
    )
    assert sched_explicit["episodes"][0]["already_done"] is True

    sched_zd_on = jm_cli.compute_schedule_preview(inputs=[str(ep2)], zero_disk=True, use_tensorrt=False)
    sched_zd_off = jm_cli.compute_schedule_preview(inputs=[str(ep2)], zero_disk=False, use_tensorrt=False)
    assert sched_zd_off["episodes"][0]["ref_fps"] < sched_zd_on["episodes"][0]["ref_fps"]

    # 6. Validation of subs_quality and deadline across job API endpoints
    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, popen_factory=lambda c, **_kw: FakePopenProcess(c))
    with TestClient(app) as client:
        assert client.post("/api/jobs/start", json={"inputs": [str(ep1)], "subs_quality": "ultra"}).status_code == 400
        assert client.post("/api/jobs/start", json={"inputs": [str(ep1)], "deadline": "29:99"}).status_code == 400
        assert client.post("/api/jobs/subtrans", json={"input_path": str(ep1), "subs_quality": "invalid"}).status_code == 400
        assert client.post("/api/jobs/subs-pending", json={"subs_quality": "bad"}).status_code == 400


def test_telemetry_small_logs_multitab_sse_checkpoints_sqlite_uri_and_shows_parity(tmp_path: Path, monkeypatch):
    import asyncio
    import src.dedup as core_dedup_mod
    import src.models_resolver as core_models_resolver
    from web.sse_hub import TelemetryHub, scan_temp_checkpoints

    # 1. Special characters '#' and '%' in SQLite DB path + Telemetry workspace
    special_root = _seed_workspace(tmp_path / "Shows #1 100% Folder")
    db_path = special_root / "subtrans" / ".cache" / "translations.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE translations (sig TEXT, pt_date TEXT)")
        conn.execute("INSERT INTO translations VALUES ('k1', '2026-11-16')")
        conn.commit()
    finally:
        conn.close()

    # 2. Custom -o directory checkpoint discovery
    custom_out = special_root / "custom_exports"
    custom_tp = custom_out / ".temp_Ep07"
    custom_tp.mkdir(parents=True, exist_ok=True)
    (custom_tp / "up_0000.mp4").write_bytes(b"\x00" * 512)
    (special_root / "logs" / "live_status.json").write_text(
        json.dumps(
            {
                "pid": 999999,
                "stage": "completed",
                "output_path": str(custom_out / "Ep07_4K.mkv"),
                "stage_detail": "Hoàn tất",
                "total_chunks": 10,
                "completed_chunks": 10,
            }
        ),
        encoding="utf-8",
    )
    found_tps = scan_temp_checkpoints(special_root)
    assert any(cp["name"] == ".temp_Ep07" for cp in found_tps)

    # 3. <96B log growth never re-emits earlier lines
    latest_log = special_root / "logs" / "latest.log"
    latest_log.write_text("[01:00:01] Short line 1\n", encoding="utf-8")
    state_box: Dict[str, Any] = {"status": "idle", "active_job": None, "last_finished_job": None, "history": []}
    hub = TelemetryHub(special_root, gpu_query_fn=_idle_gpu_metrics, job_state_provider=lambda: state_box)
    snap = hub.build_snapshot()
    assert snap["logs"] == ["[01:00:01] Short line 1"]
    cursor = snap["log_cursor"]
    with latest_log.open("a", encoding="utf-8") as f:
        f.write("[01:00:02] Short line 2\n")
    inc1, cursor = hub._read_incremental_logs_from_cursor(cursor)
    assert inc1 == ["[01:00:02] Short line 2"]
    with latest_log.open("a", encoding="utf-8") as f:
        f.write("[01:00:03] Longer third line crossing the ninety six byte boundary cleanly for verification\n")
    inc2, cursor = hub._read_incremental_logs_from_cursor(cursor)
    assert inc2 == ["[01:00:03] Longer third line crossing the ninety six byte boundary cleanly for verification"]

    # 4. No false 'subtrans' when live_status.json is completed without Stage B active;
    #    finished subtrans Web Job emits 'completed'
    pipe_done = hub.read_pipeline_state()
    assert pipe_done["effective_stage"] == "completed"
    state_box["last_finished_job"] = {
        "job_id": "sub_done_1",
        "job_type": "subtrans",
        "status": "completed",
        "pid": 80001,
        "exit_code": 0,
        "finished_at": "2026-11-16T01:05:00",
        "queue_items": [str(special_root / "anime_season" / "Frieren_02.ja.srt")],
        "options": {"subs_quality": "high"},
    }
    pipe_sub_done = hub.read_pipeline_state()
    assert pipe_sub_done["effective_stage"] == "completed"

    # 5. Multi-subscriber SSE log cursor isolation & non-mutating /api/status while broadcast loop runs
    def _parse_sse_json(raw_chunk: str) -> Dict[str, Any]:
        for ln in raw_chunk.splitlines():
            if ln.startswith("data: "):
                return json.loads(ln[len("data: "):])
        raise AssertionError(f"Missing SSE data line in {raw_chunk!r}")

    async def _verify_multi_tab_sse() -> None:
        gen1 = hub.stream_events()
        gen2 = hub.stream_events()
        try:
            ev1_init = _parse_sse_json(await anext(gen1))
            ev2_init = _parse_sse_json(await anext(gen2))
            assert ev1_init["logs"][-1].startswith("[01:00:03]")
            assert ev2_init["logs"][-1].startswith("[01:00:03]")

            with latest_log.open("a", encoding="utf-8") as f:
                f.write("[01:00:04] Multi-tab isolated log event\n")

            ev1_tick = _parse_sse_json(await anext(gen1))
            ev2_tick = _parse_sse_json(await anext(gen2))
            assert "[01:00:04] Multi-tab isolated log event" in ev1_tick["logs"]
            assert "[01:00:04] Multi-tab isolated log event" in ev2_tick["logs"]

            # Once _broadcast_loop is active, GET /api/status must not mutate shared hub.history
            assert hub._poll_task is not None and not hub._poll_task.done()
            hist_len_before = len(hub.history)
            hub.build_snapshot()
            assert len(hub.history) == hist_len_before
        finally:
            await gen1.aclose()
            await gen2.aclose()
            if hub._poll_task is not None and not hub._poll_task.done():
                hub._poll_task.cancel()
                try:
                    await hub._poll_task
                except asyncio.CancelledError:
                    pass

    asyncio.run(_verify_multi_tab_sse())

    # 6. SQLite #/% URI endpoint & 2-pass shows.yaml parity with mixed-case dedup_policy & override_dedup
    split_shows = (
        "shows:\n"
        "  - pattern: '*[SubDir]*'\n"
        "    model: animejanai-hd-v3-ultracompact\n"
        "  - pattern: '*Special*'\n"
        "    dedup_policy: 'DIS'\n"
    )
    (special_root / "shows.yaml").write_text(split_shows, encoding="utf-8")
    sample_vid = r"D:\[SubDir]\Ep01_Special.mkv"
    monkeypatch.chdir(special_root)
    monkeypatch.setattr(core_dedup_mod, "ROOT_DIR", str(special_root))
    core_model = core_models_resolver.resolve_model_for_video(sample_vid)
    core_dedup = core_dedup_mod.resolve_dedup_policy(sample_vid)
    fs_special = SafeLocalFSBrowser(special_root)
    web_routing = fs_special.resolve_routing_for_path(sample_vid)
    assert web_routing["model_id"] == "animejanai-hd-v3-ultracompact"
    assert core_model["name"] == "AnimeJaNai V3 HD UltraCompact"
    assert web_routing["dedup_policy"] == core_dedup == "dis"
    assert fs_special.resolve_routing_for_path(sample_vid, override_dedup="SAKUGA")["dedup_policy"] == "sakuga"

    app = create_app(workspace_root=special_root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        r_ov = client.get("/api/subtrans/overview")
        assert r_ov.status_code == 200
        assert r_ov.json()["cache"]["total_cached_cues"] == 1


def test_real_1080p_4k_ffmpeg_compare_and_frontend_regression_guards(studio_env, tmp_path: Path):
    import shutil

    client: TestClient = studio_env["client"]
    panels_js = client.get("/static/js/panels.js").text
    charts_js = client.get("/static/js/charts.js").text
    compare_js = client.get("/static/js/compare_stage.js").text
    app_js = client.get("/static/js/app.js").text
    css = client.get("/static/styles.css").text

    assert "setSelectValueSafely" in panels_js
    assert "clearKeySlot" in panels_js
    assert "clear-api-key-1" in panels_js
    assert "raw_preview" in panels_js
    assert "CORRUPT" in panels_js
    assert "prevFps.length > 0" in charts_js
    assert 'srcInput.value = item.source_path || ""' in compare_js
    assert "lastActivePipelineStep" in app_js
    assert "cur.reset" in app_js
    assert "td:not([colspan])" in css
    assert '[data-tone="warn"]' in css

    if shutil.which("ffmpeg"):
        vid_dir = tmp_path / "fhd_4k_pair"
        vid_dir.mkdir(parents=True, exist_ok=True)
        src_1080 = vid_dir / "pair_1080p.mkv"
        up_4k = vid_dir / "pair_4K.mkv"
        for target, dims in ((src_1080, "1920x1080"), (up_4k, "3840x2160")):
            res = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"testsrc=size={dims}:rate=24:duration=0.2",
                    "-frames:v",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:v",
                    "libx265",
                    "-preset",
                    "ultrafast",
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
                timeout=30.0,
                check=False,
            )
            if res.returncode != 0:
                pytest.skip("ffmpeg ultrafast 4K encode unavailable")
        app = create_app(workspace_root=studio_env["root"], gpu_query_fn=_idle_gpu_metrics)
        with TestClient(app) as real_client:
            r = real_client.post(
                "/api/preview/compare-frames",
                json={"source_path": str(src_1080), "upscaled_path": str(up_4k), "timestamp_sec": 0.0},
            )
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["source_meta"]["width"] == 1920
            assert data["source_meta"]["height"] == 1080
            assert data["upscaled_meta"]["width"] == 3840
            assert data["upscaled_meta"]["height"] == 2160
            assert real_client.get(data["source_frame_url"]).content.startswith(b"\x89PNG\r\n\x1a\n")
            assert real_client.get(data["upscaled_frame_url"]).content.startswith(b"\x89PNG\r\n\x1a\n")


def test_review_regression_fixes_and_edge_cases(tmp_path: Path, monkeypatch):
    """Kiểm tra hồi quy các lỗi thực nghiệm: failed jobs, logger step recovery, dotenv env/list/4th-key, deadline 7:xx, audit non-dict & FE guards."""
    import src.logger as logger_mod
    from web.app import _resolve_subtrans_key_slots, create_app
    from web.sse_hub import TelemetryHub

    root = tmp_path / "studio_review"
    for d in ("logs", "out", "models", "subtrans"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "models" / "models.yaml").write_text(
        "models:\n  animevideov3:\n    id: animevideov3\n    name: AnimeVideoV3\n    default: true\n",
        encoding="utf-8",
    )
    (root / "shows.yaml").write_text("shows: []\n", encoding="utf-8")

    # 1. Failed subtrans/upscale Web Jobs (status='failed', NOT 'error') & stale completed live_status.json
    (root / "logs" / "live_status.json").write_text(
        json.dumps({"pid": 999999, "stage": "completed", "stage_detail": "Tập trước đã xong"}),
        encoding="utf-8",
    )
    failed_sub = {
        "job_id": "sub_fail_1",
        "job_type": "subtrans",
        "status": "failed",
        "pid": 12345,
        "return_code": 2,
        "finished_at": "2099-01-01T00:00:00",
        "queue_items": [str(root / "ep01.ja.srt")],
    }
    state_box: Dict[str, Any] = {"status": "idle", "active_job": None, "history": [failed_sub]}
    hub = TelemetryHub(root, gpu_query_fn=_idle_gpu_metrics, job_state_provider=lambda: state_box)

    ps_sub = hub.read_pipeline_state()
    assert ps_sub["effective_stage"] == "error"
    assert ps_sub["live"]["failed_stage"] == "subtrans"

    state_box["history"] = [{**failed_sub, "job_type": "upscale", "queue_items": [str(root / "ep01.mkv")]}]
    ps_up = hub.read_pipeline_state()
    assert ps_up["effective_stage"] == "error"
    assert ps_up["live"]["failed_stage"] == "probing"

    # 2. RunLogger.set_stage preserves failed_stage / last_stage across crash & pause reloads
    monkeypatch.setattr(logger_mod, "LOGS_DIR", str(root / "logs"))
    lg = logger_mod.RunLogger("Ep01")
    lg.set_stage("muxing", "Đang mux...")
    lg.set_stage("error", "Mux lỗi")
    st_err = json.loads((root / "logs" / "live_status.json").read_text(encoding="utf-8"))
    assert st_err["stage"] == "error"
    assert st_err["failed_stage"] == "muxing"
    assert st_err["last_stage"] == "muxing"

    lg.set_stage("upscaling", "Chunk 2/5")
    lg.set_stage("paused", "Tạm dừng")
    st_pause = json.loads((root / "logs" / "live_status.json").read_text(encoding="utf-8"))
    assert st_pause["stage"] == "paused"
    assert st_pause["last_stage"] == "upscaling"
    assert "failed_stage" not in st_pause

    # 3. .env vs process os.environ precedence under SUBTRANS_DISABLE_DOTENV=1, [LIST] badge, 4th+ key preservation & deduplication
    dotenv_file = root / ".env"
    dotenv_file.write_text("GEMINI_API_KEY_1=dotenv_k1_secret\n", encoding="utf-8")
    monkeypatch.setenv("SUBTRANS_DISABLE_DOTENV", "1")
    monkeypatch.setenv("GEMINI_API_KEY_1", "proc_env_k1_secret")
    slots_env, cnt_env = _resolve_subtrans_key_slots(dotenv_file)
    assert cnt_env == 1
    assert slots_env[0]["source"] == "env"
    assert "proc_e" in slots_env[0]["masked"]

    monkeypatch.delenv("SUBTRANS_DISABLE_DOTENV", raising=False)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_1", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_3", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)

    dotenv_file.write_text("GEMINI_API_KEYS=key_one_111,key_two_222,key_three_333,key_four_444\n", encoding="utf-8")
    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        ov = client.get("/api/subtrans/overview").json()
        assert ov["configured_keys_count"] == 4
        assert ov["key_slots"][0]["env_name"] == "GEMINI_API_KEYS"
        r_upd = client.put("/api/subtrans/api-keys", json={"key_1": "new_k1_secret"})
        assert r_upd.status_code == 200
        assert r_upd.json()["key_slots"][0]["source"] == "dotenv"
        assert r_upd.json()["key_slots"][1]["source"] == "dotenv"
        assert r_upd.json()["key_slots"][2]["source"] == "dotenv"
        env_txt = dotenv_file.read_text(encoding="utf-8")
        assert "GEMINI_API_KEY_4=key_four_444" in env_txt
        assert "GEMINI_API_KEY_1=new_k1_secret" in env_txt
        assert "GEMINI_API_KEY=new_k1_secret" in env_txt

        # Clearing key_1 from GEMINI_API_KEYS with SUBTRANS_DISABLE_DOTENV unset keeps slots 2 & 3 as source='dotenv'
        dotenv_file.write_text("GEMINI_API_KEYS=key_one_111,key_two_222,key_three_333,key_four_444\n", encoding="utf-8")
        r_clr = client.put("/api/subtrans/api-keys", json={"key_1": ""})
        assert r_clr.status_code == 200
        clr_slots = r_clr.json()["key_slots"]
        assert clr_slots[0]["configured"] is False
        assert clr_slots[1]["configured"] is True and clr_slots[1]["source"] == "dotenv"
        assert clr_slots[2]["configured"] is True and clr_slots[2]["source"] == "dotenv"
        assert "GEMINI_API_KEY_4=key_four_444" in dotenv_file.read_text(encoding="utf-8")
        assert not any(
            k in os.environ
            for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4")
        )

        # Duplicate key lines deduplicated on write
        dotenv_file.write_text("GEMINI_API_KEY_1=dup1\nOTHER_VAR=keep\nGEMINI_API_KEY_1=dup2\n", encoding="utf-8")
        client.put("/api/subtrans/api-keys", json={"key_1": "dedup_k1_secret"})
        dup_lines = [ln for ln in dotenv_file.read_text(encoding="utf-8").splitlines() if ln.startswith("GEMINI_API_KEY_1=")]
        assert len(dup_lines) == 1
        assert "OTHER_VAR=keep" in dotenv_file.read_text(encoding="utf-8")

        # 4. Schedule deadline '7:xx' -> clean HH:MM message; non-dict audit '[1,2,3]' -> 422 corrupt
        dummy_vid = root / "out" / "ep01.mkv"
        dummy_vid.write_bytes(b"\x00" * 100)
        r_dl = client.post("/api/queue/preview-schedule", json={"inputs": [str(dummy_vid)], "deadline": "7:xx"})
        assert r_dl.status_code == 400
        assert "HH:MM" in r_dl.json()["detail"]

        bad_list_audit = root / "out" / "badlist_4K.audit.json"
        bad_list_audit.write_text("[1, 2, 3]", encoding="utf-8")
        r_aud = client.get("/api/audits").json()
        assert any(a["corrupt"] and a["status"] == "CORRUPT" for a in r_aud["audits"])
        r_det = client.get("/api/audits/detail", params={"path": str(bad_list_audit)})
        assert r_det.status_code == 422
        assert r_det.json()["corrupt"] is True

        # 5. Frontend regression guards present in served static files
        assert "priorSlots > 0 ? prevFps.slice(-priorSlots) : []" in client.get("/static/js/charts.js").text
        panels_txt = client.get("/static/js/panels.js").text
        assert "Number.isFinite(batchVal)" in panels_txt
        assert "Number.isFinite(offsetMs)" in panels_txt
        assert "a.corrupt && a.error" in panels_txt
        compare_txt = client.get("/static/js/compare_stage.js").text
        assert "Number.isFinite(tsSec)" in compare_txt
        assert ".text-error" in client.get("/static/styles.css").text


def test_external_output_dir_discovery_preview_source_resolution_and_audits_auth(tmp_path: Path):
    """Verify external -o directory discovery across candidates/audits, .preview_clip_* skip + VIDEO_EXTENSIONS
    source resolution, /api/audits/detail authorization order (400 vs 200 vs 403), and pending_subs.json priority.
    """
    import shutil

    root = _seed_workspace(tmp_path / "ext_out_ws")
    shutil.rmtree(root / "out")

    ext_out = tmp_path / "External_Upscaled_4K"
    (ext_out / "audit").mkdir(parents=True)
    (ext_out / "cast").mkdir(parents=True)
    raw_dir = tmp_path / "Raw_Sources" / "FateZero"
    raw_dir.mkdir(parents=True)

    src_ep10 = raw_dir / "FateZero - S01E10 - Rin's Adventure.mov"
    src_ep11 = raw_dir / "FateZero - S01E11 - The Grail Dialogue.mkv"
    src_ep10.write_bytes(b"\x00" * 2048)
    src_ep11.write_bytes(b"\x00" * 2048)

    up_ep10_prev = ext_out / "FateZero - S01E10 - Rin's Adventure_preview_10s_4K.mkv"
    up_ep11 = ext_out / "FateZero - S01E11 - The Grail Dialogue_4K.mkv"
    up_ep10_prev.write_bytes(b"\x00" * 4096)
    up_ep11.write_bytes(b"\x00" * 4096)

    # Ep10 preview audit recorded deleted .preview_clip_* as source_file (no original_source_file)
    audit_ep10 = ext_out / "FateZero - S01E10 - Rin's Adventure_preview_10s_4K.audit.json"
    audit_ep10.write_text(
        json.dumps(
            {
                "output_file": str(up_ep10_prev),
                "source_file": str(ext_out / ".preview_clip_FateZero - S01E10 - Rin's Adventure.mkv"),
                "status": "PASS",
                "output_sanity": {"status": "PASS"},
            }
        ),
        encoding="utf-8",
    )

    # Ep11 audit points to real raw_dir and has sibling .audit.md and .cast.yaml
    audit_ep11 = ext_out / "FateZero - S01E11 - The Grail Dialogue_4K.audit.json"
    audit_ep11.write_text(
        json.dumps(
            {
                "output_file": str(up_ep11),
                "source_file": str(src_ep11),
                "status": "PASS",
                "output_sanity": {"status": "PASS"},
            }
        ),
        encoding="utf-8",
    )
    audit_ep11_md = ext_out / "audit" / "FateZero - S01E11 - The Grail Dialogue.vi.audit.md"
    audit_ep11_md.write_text("# External Subtrans PASS\n", encoding="utf-8")
    cast_ep11 = ext_out / "cast" / "FateZero - S01E11 - The Grail Dialogue.cast.yaml"
    cast_ep11.write_text("characters: []\n", encoding="utf-8")

    # Test original_source_file preference in a query-param extra output dir
    extra_out = tmp_path / "Query_Out_4K"
    extra_out.mkdir(parents=True)
    up_orig = extra_out / "Ep12_preview_10s_4K.mkv"
    src_orig = raw_dir / "Ep12.ts"
    up_orig.write_bytes(b"\x00" * 4096)
    src_orig.write_bytes(b"\x00" * 2048)
    (extra_out / "Ep12_preview_10s_4K.audit.json").write_text(
        json.dumps(
            {
                "output_file": str(up_orig),
                "source_file": str(extra_out / ".preview_clip_Ep12.mkv"),
                "original_source_file": str(src_orig),
                "status": "PASS",
            }
        ),
        encoding="utf-8",
    )

    # Register ext_out via logs/live_status.json output_path
    (root / "logs" / "live_status.json").write_text(
        json.dumps({"pid": 999999, "stage": "completed", "output_path": str(up_ep11)}),
        encoding="utf-8",
    )

    # pending_subs.json at root takes priority over stale out/pending_subs.json
    (root / "pending_subs.json").write_text(
        json.dumps([{"video": "root_priority.mkv"}]), encoding="utf-8"
    )
    (root / "out").mkdir(exist_ok=True)
    (root / "out" / "pending_subs.json").write_text(
        json.dumps([{"video": "stale_out_item.mkv"}]), encoding="utf-8"
    )
    shutil.rmtree(root / "out")

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        # 1. /api/preview/candidates scans ext_out even without root/out, skips .preview_clip_*, resolves .mov across raw_dir
        r_cands = client.get("/api/preview/candidates")
        assert r_cands.status_code == 200
        cands = r_cands.json()["candidates"]
        ep10_item = next(c for c in cands if "S01E10" in c["upscaled_path"])
        assert ep10_item["is_preview"] is True
        assert Path(ep10_item["source_path"]).resolve() == src_ep10.resolve()

        # Optional output_dir query param discovers extra_out & honors original_source_file
        r_q_cands = client.get("/api/preview/candidates", params={"output_dir": str(extra_out)})
        assert r_q_cands.status_code == 200
        ep12_item = next(c for c in r_q_cands.json()["candidates"] if "Ep12" in c["upscaled_path"])
        assert Path(ep12_item["source_path"]).resolve() == src_orig.resolve()

        # 2. /api/audits discovers external -o audits + audit.md + cast.yaml, skips .preview_clip_* in source_file
        r_aud = client.get("/api/audits")
        assert r_aud.status_code == 200
        auds = r_aud.json()["audits"]
        aud10 = next(a for a in auds if "S01E10" in a["episode"])
        assert aud10["source_file"] == ""
        aud11 = next(a for a in auds if "S01E11" in a["episode"])
        assert Path(aud11["sub_audit_md_path"]).resolve() == audit_ep11_md.resolve()
        assert Path(aud11["cast_yaml_path"]).resolve() == cast_ep11.resolve()

        # Optional output_dir query param also prefers original_source_file in /api/audits
        r_q_aud = client.get("/api/audits", params={"output_dir": str(extra_out)})
        assert r_q_aud.status_code == 200
        aud12 = next(a for a in r_q_aud.json()["audits"] if "Ep12" in a["episode"])
        assert Path(aud12["source_file"]).resolve() == src_orig.resolve()

        # 3. /api/audits/detail allows external known output_dir files (200), validates suffix FIRST (400), rejects outside (403)
        assert client.get("/api/audits/detail", params={"path": str(audit_ep11)}).status_code == 200
        assert client.get("/api/audits/detail", params={"path": str(audit_ep11_md)}).status_code == 200
        assert client.get("/api/audits/detail", params={"path": str(cast_ep11)}).status_code == 200

        outside_secret = root.parent / "outside_secret.txt"
        outside_secret.write_text("secret", encoding="utf-8")
        assert client.get("/api/audits/detail", params={"path": str(outside_secret)}).status_code == 400

        outside_audit = root.parent / "outside.audit.json"
        outside_audit.write_text("{}", encoding="utf-8")
        assert client.get("/api/audits/detail", params={"path": str(outside_audit)}).status_code == 403

        # 4. /api/subtrans/overview checks root/pending_subs.json before out/pending_subs.json
        r_ov = client.get("/api/subtrans/overview")
        assert r_ov.status_code == 200
        assert r_ov.json()["pending_subs"] == [{"video": "root_priority.mkv"}]


def test_lan_readonly_default_vs_lan_control_and_cli_flag(tmp_path: Path):
    """Verify --lan defaults to read-only monitoring for non-loopback peers while preserving local loopback
    control, and --lan-control enables remote mutations.
    """
    parser = web_gui.build_parser()
    lan_only_args = parser.parse_args(["--lan"])
    assert lan_only_args.lan is True
    assert lan_only_args.lan_control is False
    lan_ctrl_args = parser.parse_args(["--lan", "--lan-control"])
    assert lan_ctrl_args.lan_control is True

    root = _seed_workspace(tmp_path / "lan_ro_ws")
    ep1 = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    audit_json = root / "out" / "Fate_Ep22_4K.audit.json"
    audit_json.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")

    # Default --lan (allow_lan=True, allow_lan_control=False)
    app_ro = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=81000, running=False),
        allow_lan=True,
        allow_lan_control=False,
    )
    lan_headers = {"Host": "192.168.1.42:8765", "Origin": "http://192.168.1.42:8765"}
    local_headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}

    with TestClient(app_ro) as client_lan:
        # Monitoring GETs allowed from LAN
        assert client_lan.get("/", headers=lan_headers).status_code == 200
        assert client_lan.get("/api/status", headers=lan_headers).status_code == 200
        assert client_lan.get("/api/logs/recent", headers=lan_headers).status_code == 200
        assert client_lan.get("/api/preview/candidates", headers=lan_headers).status_code == 200
        assert client_lan.get("/api/config/models-and-shows", headers=lan_headers).status_code == 200
        assert client_lan.get("/api/subtrans/overview", headers=lan_headers).status_code == 200
        assert client_lan.get("/api/audits", headers=lan_headers).status_code == 200
        assert (
            client_lan.get("/api/audits/detail", params={"path": str(audit_json)}, headers=lan_headers).status_code
            == 200
        )

        # FS browse, mutating, spawn, and key-edit routes rejected with 403 for LAN peer
        assert client_lan.get("/api/fs/browse", headers=lan_headers).status_code == 403
        assert client_lan.post("/api/queue/preview-schedule", json={"inputs": [str(ep1)]}, headers=lan_headers).status_code == 403
        assert client_lan.post("/api/jobs/start", json={"inputs": [str(ep1)], "dry_run": True}, headers=lan_headers).status_code == 403
        assert client_lan.post("/api/jobs/preview", json={"video_path": str(ep1)}, headers=lan_headers).status_code == 403
        assert client_lan.post("/api/jobs/subtrans", json={"input_path": str(ep1), "dry_run": True}, headers=lan_headers).status_code == 403
        assert client_lan.post("/api/jobs/subs-pending", json={"dry_run": True}, headers=lan_headers).status_code == 403
        assert client_lan.post("/api/jobs/pause-graceful", headers=lan_headers).status_code == 403
        assert client_lan.post("/api/jobs/clear-pause", headers=lan_headers).status_code == 403
        assert client_lan.post("/api/jobs/abort", headers=lan_headers).status_code == 403
        assert client_lan.post("/api/open-media", json={"path": str(ep1)}, headers=lan_headers).status_code == 403
        assert client_lan.put("/api/config/shows", json={"yaml_content": "shows: []\n"}, headers=lan_headers).status_code == 403
        assert client_lan.put("/api/subtrans/api-keys", json={"key_1": "x"}, headers=lan_headers).status_code == 403

        # Loopback client on the same server retains full control
        assert client_lan.get("/api/fs/browse", headers=local_headers).status_code == 200
        assert client_lan.post("/api/jobs/start", json={"inputs": [str(ep1)], "dry_run": True}, headers=local_headers).status_code == 200

    # Also verify non-loopback client socket IP even with local Host header cannot bypass --lan read-only
    with TestClient(app_ro, client=("192.168.1.99", 54321)) as remote_client:
        assert remote_client.get("/api/status", headers=lan_headers).status_code == 200
        assert remote_client.get("/api/fs/browse", headers=lan_headers).status_code == 403
        assert remote_client.post("/api/jobs/abort", headers=lan_headers).status_code == 403

    # --lan --lan-control enables full LAN control
    app_ctrl = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=81001, running=False),
        allow_lan=True,
        allow_lan_control=True,
    )
    with TestClient(app_ctrl, client=("192.168.1.99", 54321)) as ctrl_client:
        assert ctrl_client.get("/api/fs/browse", headers=lan_headers).status_code == 200
        assert ctrl_client.post("/api/jobs/start", json={"inputs": [str(ep1)], "dry_run": True}, headers=lan_headers).status_code == 200


def test_telemetry_shutdown_log_pinning_live_pid_conflict_and_schedule_optimizations(
    tmp_path: Path, monkeypatch
):
    """Verify TelemetryHub.shutdown() cleanly breaks SSE subscribers, active Web jobs pin logs/web_job.log,
    alive live_status.json PIDs block conflicting GPU launches, and compute_schedule_preview respects FAIL
    audits, already_done probe skip, and force=True.
    """
    import asyncio
    import time
    import web.job_manager as jm_mod
    from web.job_manager import JobManager
    from web.sse_hub import TelemetryHub

    root = _seed_workspace(tmp_path / "sse_pin_ws")

    # 1. Active Web upscale job pins deterministically to logs/web_job.log even if RunLogger log is newer
    web_log = root / "logs" / "web_job.log"
    run_log = root / "logs" / "Ep01_20261118.log"
    latest_log = root / "logs" / "latest.log"
    web_log.write_text("# Web Studio Job\n[INFO] Running\n", encoding="utf-8")
    run_log.write_text("[CHUNK 01/04] RunLogger line\n", encoding="utf-8")
    latest_log.write_text("[CHUNK 01/04] Latest line\n", encoding="utf-8")
    now_t = time.time()
    os.utime(web_log, (now_t - 5, now_t - 5))
    os.utime(run_log, (now_t + 10, now_t + 10))
    os.utime(latest_log, (now_t + 10, now_t + 10))
    (root / "logs" / "live_status.json").write_text(
        json.dumps({"pid": 999999, "stage": "upscaling", "log_file": run_log.name}),
        encoding="utf-8",
    )

    state_box: Dict[str, Any] = {
        "status": "running",
        "active_job": {"job_id": "j1", "job_type": "upscale", "status": "running", "pid": 60001},
        "history": [],
    }
    hub = TelemetryHub(root, gpu_query_fn=_idle_gpu_metrics, job_state_provider=lambda: state_box)
    assert hub._select_active_log_file() == web_log.resolve() or hub._select_active_log_file() == web_log

    # 2. TelemetryHub.shutdown() breaks active stream_events() subscribers cleanly
    async def _check_shutdown() -> None:
        gen = hub.stream_events()
        snap_ev = await anext(gen)
        assert "event: snapshot" in snap_ev
        tick_waiter = asyncio.create_task(anext(gen))
        await asyncio.sleep(0.05)
        assert hub._poll_task is not None and not hub._poll_task.done()
        assert len(hub._subscribers) == 1
        await hub.shutdown()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(tick_waiter, timeout=2.0)
        assert hub._poll_task is None
        assert hub._shutting_down is True

    asyncio.run(_check_shutdown())

    # 3. live_status.json alive PID in pre-GPU stage blocks new GPU launch even without .gpu_active.lock
    ep1 = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    (root / "logs" / "live_status.json").write_text(
        json.dumps({"pid": os.getpid(), "stage": "preflight_dedup"}),
        encoding="utf-8",
    )
    jm = JobManager(root, popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=73000, running=False))
    with pytest.raises(RuntimeError, match="preflight_dedup"):
        jm.start_upscale_job({"inputs": [str(ep1)], "dry_run": False})
    with pytest.raises(RuntimeError, match="preflight_dedup"):
        jm.start_preview_job({"video_path": str(ep1)})
    # dry_run does not require GPU lock
    dry_job = jm.start_upscale_job({"inputs": [str(ep1)], "dry_run": True})
    assert dry_job["status"] == "running"
    jm.abort_job()

    # 4. compute_schedule_preview: already_done skips PyAV probe, FAIL audit resets already_done, force=True overrides
    out_4k = root / "out" / f"{ep1.stem}_4K.mkv"
    out_4k.write_bytes(b"\x00" * (1024 * 1024 + 4096))
    probe_calls: List[str] = []

    def spy_probe(path_str: str) -> int:
        probe_calls.append(path_str)
        return 12345

    monkeypatch.setattr(jm_mod, "_probe_frames_lightweight", spy_probe)
    sched_done = jm.compute_schedule_preview(inputs=[str(ep1)], force=False)
    assert sched_done["episodes"][0]["already_done"] is True
    assert probe_calls == []

    # Sibling FAIL audit -> NOT already_done, probes & schedules
    (root / "out" / f"{ep1.stem}_4K.audit.json").write_text(
        json.dumps({"status": "FAIL"}), encoding="utf-8"
    )
    sched_fail = jm.compute_schedule_preview(inputs=[str(ep1)], force=False)
    assert sched_fail["episodes"][0]["already_done"] is False
    assert sched_fail["episodes"][0]["can_run"] is True
    assert len(probe_calls) == 1

    # Restore PASS audit and verify force=True overrides already_done
    (root / "out" / f"{ep1.stem}_4K.audit.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    sched_forced = jm.compute_schedule_preview(inputs=[str(ep1)], force=True)
    assert sched_forced["episodes"][0]["already_done"] is False
    assert sched_forced["episodes"][0]["can_run"] is True


def test_frontend_a11y_mobile_and_perf_guards(studio_env):
    """Verify A11y, mobile reflow, and rAF performance fixes remain intact in served static assets."""
    client: TestClient = studio_env["client"]
    html = client.get("/").text
    css = client.get("/static/styles.css").text
    ui_js = client.get("/static/js/ui.js").text
    charts_js = client.get("/static/js/charts.js").text
    compare_js = client.get("/static/js/compare_stage.js").text
    panels_js = client.get("/static/js/panels.js").text
    app_js = client.get("/static/js/app.js").text

    assert 'id="models-catalog-heading" class="section-heading" tabindex="-1"' in html
    assert 'id="fs-btn-drives" class="btn btn-secondary" autofocus' in html
    assert html.count('autocomplete="off"') >= 3
    assert 'id="shows-yaml-error"' in html
    assert 'aria-describedby="opt-deadline-error"' in html

    assert "flex-wrap: balance" not in css
    assert "touch-action: pan-y;" in css
    assert "touch-action: none;" not in css
    assert "@media (min-width: 860px)" in css
    assert "@media (max-width: 680px)" in css
    assert "z-index: 3;" in css
    assert ':root:has(> head > meta[name="color-scheme"][content])' in css
    assert "accent-color: var(--accent);" in css
    assert "flex: 1 1 8.5rem;" in css

    assert "hidingFocusedPanel" in ui_js
    assert "requestAnimationFrame" in charts_js
    assert "${base}-${suffix}" in charts_js and "forced-colors: active" in charts_js and "light-dark(" in charts_js
    assert "scheduleRender" in compare_js
    assert 'hasAttribute("hidden")' in compare_js
    assert "aria-busy" in panels_js
    assert "checkValidity()" in panels_js
    assert 'addEventListener("blur"' in panels_js
    assert "setShowsYamlError" in panels_js
    assert "REFRESH_TTL_MS" in panels_js
    assert "requestAnimationFrame" in app_js


def test_web_gui_mock_mode_routes_and_sse(tmp_path):
    """Verify web_gui.py --mock and create_app(..., mock_mode=True) serve deterministic telemetry, jobs, and SSE."""
    parsed = web_gui.build_parser().parse_args(["--mock", "--no-browser"])
    assert parsed.mock is True
    assert parsed.no_browser is True

    app = create_app(workspace_root=tmp_path, mock_mode=True)
    assert app.state.mock_mode is True

    with TestClient(app) as client:
        status_resp = client.get("/api/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert "Mock" in status_data["gpu_info"]["gpu_name"]
        assert status_data["telemetry"]["vram_used_mib"] >= 6144.0
        assert status_data["pipeline"]["effective_stage"] == "upscaling"
        assert status_data["pipeline"]["mock_mode"] is True
        assert status_data["pipeline"]["live"]["steady_fps"] > 0
        live_mock = status_data["pipeline"]["live"]
        completed_mock = live_mock["completed_chunks"]
        assert len(live_mock["chunks"]) == completed_mock
        assert [c["chunk_idx"] for c in live_mock["chunks"]] == list(range(completed_mock))
        assert live_mock["active_chunk"] == completed_mock
        assert live_mock["resumed_chunks"] == [0, 1]
        assert status_data["pipeline"]["job"]["active_job"]["job_type"] == "upscale"
        history_types = {h["job_type"] for h in status_data["pipeline"]["job"]["history"]}
        assert {"upscale", "subtrans"}.issubset(history_types)
        assert len(status_data["logs"]) >= 2

        sse_resp = client.get("/api/events?once=1")
        assert sse_resp.status_code == 200
        assert "event: snapshot" in sse_resp.text
        assert "data: " in sse_resp.text
        assert "mock-upscale-ep02" in sse_resp.text

        sub_resp = client.get("/api/subtrans/overview")
        assert sub_resp.status_code == 200
        sub_data = sub_resp.json()
        assert sub_data["mock_mode"] is True
        assert len(sub_data["pending_subs"]) >= 1

        audit_resp = client.get("/api/audits")
        assert audit_resp.status_code == 200
        audit_data = audit_resp.json()
        assert len(audit_data["audits"]) >= 1
        assert audit_data["audits"][0]["status"] == "PASS"

        # --mock without custom popen_factory simulates job start without spawning subprocesses
        sim_job = client.post("/api/jobs/start", json={"inputs": ["samples/ep01.mkv"], "dry_run": False})
        assert sim_job.status_code == 200
        assert sim_job.json()["mock"] is True
        assert sim_job.json()["job"]["mock"] is True


def test_batch4_security_env_parity_scheduler_and_cli_hardening(tmp_path: Path):
    """Verify Batch 4 Web Studio security, .env slot splitting, per-job log isolation, and scheduler/CLI parity."""
    import sqlite3
    import upscale
    from src.scheduler import compute_overnight_schedule
    from web.job_manager import JobManager
    from web.sse_hub import TelemetryHub

    root = _seed_workspace(tmp_path / "b4_ws")
    ep1 = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    ep2 = root / "anime_season" / "[SubsPlease] Frieren - 01 (1080p).mkv"

    # Seed SQLite cache with success/ok/dry_run rows + 1 active & 1 expired exhaustion row
    sub_dir = root / "subtrans"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / ".env").write_text(
        "GEMINI_API_KEY=AIzaSySlotOne111111,AIzaSySlotTwo222222;AIzaSySlotThree333333\n",
        encoding="utf-8",
    )
    cache_dir = sub_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "translations.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE translations (id INTEGER PRIMARY KEY, pt_date TEXT)")
        conn.execute(
            "CREATE TABLE api_requests (id INTEGER PRIMARY KEY, pt_date TEXT, model TEXT, cues_count INTEGER, total_tokens INTEGER, status TEXT)"
        )
        conn.execute("CREATE TABLE key_exhaustion (key_id TEXT, model TEXT, reason TEXT, pt_date TEXT)")
        conn.execute(
            "CREATE TABLE model_exhaustion (model TEXT, exhausted_until_pt TEXT, reason TEXT)"
        )
        from datetime import datetime
        from zoneinfo import ZoneInfo

        today_pt = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
        conn.executemany(
            "INSERT INTO api_requests (pt_date, model, cues_count, total_tokens, status) VALUES (?, ?, ?, ?, ?)",
            [
                (today_pt, "gemini-3.5-flash-lite", 100, 1200, "success"),
                (today_pt, "gemini-3.5-flash-lite", 50, 600, "ok"),
                (today_pt, "gemini-3.5-flash-lite", 999, 0, "dry_run"),
            ],
        )
        conn.executemany(
            "INSERT INTO model_exhaustion (model, exhausted_until_pt, reason) VALUES (?, ?, ?)",
            [
                ("gemini-3.8-flash", "2099-12-31T00:00:00-08:00", "RPD"),
                ("gemini-old-expired", "2020-01-01T00:00:00-08:00", "stale"),
            ],
        )
        conn.commit()

    app_ro = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=88001, running=True),
        allow_lan=True,
        allow_lan_control=False,
    )
    lan_headers = {"Host": "192.168.1.50:8765"}
    local_headers = {"Host": "127.0.0.1:8765"}

    with TestClient(app_ro) as local_client:
        # 1. /api/docs disabled (no broken external-CDN Swagger blank page under strict CSP)
        assert local_client.get("/api/docs", headers=local_headers).status_code == 404

        # 2. Loopback sees masked keys, excludes dry_run rows, and filters expired exhaustion rows
        sub_local = local_client.get("/api/subtrans/overview", headers=local_headers).json()
        assert [s["configured"] for s in sub_local["key_slots"]] == [True, True, True]
        assert sub_local["key_slots"][0]["masked"].startswith("AIza")
        reqs = sub_local["cache"]["today_requests"]
        assert all(r["status"] != "dry_run" for r in reqs)
        assert all(r["healthy"] is True for r in reqs)
        assert sum(r["cues"] for r in reqs) == 150
        ex_models = [m["model"] for m in sub_local["cache"]["exhausted_models"]]
        assert ex_models == ["gemini-3.8-flash"]

        # 3. Updating slot 2 splits singular GEMINI_API_KEY and syncs subtrans/.env
        upd = local_client.put(
            "/api/subtrans/api-keys",
            json={"key_2": "AIzaSyUpdatedTwo999999"},
            headers=local_headers,
        )
        assert upd.status_code == 200
        sub_env_txt = (sub_dir / ".env").read_text(encoding="utf-8")
        assert "GEMINI_API_KEY_1=AIzaSySlotOne111111" in sub_env_txt
        assert "GEMINI_API_KEY_2=AIzaSyUpdatedTwo999999" in sub_env_txt
        assert "GEMINI_API_KEY_3=AIzaSySlotThree333333" in sub_env_txt

        # 4. UNC paths rejected across endpoints
        assert local_client.get("/api/fs/browse", params={"path": r"\\192.168.1.9\share"}, headers=local_headers).status_code == 400
        assert local_client.get("/api/audits", params={"output_dir": "//192.168.1.9/share"}, headers=local_headers).status_code == 400
        assert local_client.post(
            "/api/preview/compare-frames",
            json={"source_path": r"\\evil\share\a.mkv", "upscaled_path": str(ep1)},
            headers=local_headers,
        ).status_code == 422

    with TestClient(app_ro, client=("192.168.1.99", 54321)) as lan_client:
        # 5. Non-loopback read-only --lan rejects custom ?output_dir= and omits masked API key substrings
        assert lan_client.get("/api/preview/candidates", params={"output_dir": str(root)}, headers=lan_headers).status_code == 403
        assert lan_client.get("/api/audits", params={"output_dir": str(root)}, headers=lan_headers).status_code == 403
        sub_lan = lan_client.get("/api/subtrans/overview", headers=lan_headers).json()
        assert all(s["masked"] == "" for s in sub_lan["key_slots"])

    # 6. Per-job log isolation + _finalize_locked idempotency
    jm = JobManager(root, popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=89001, running=False))
    job_rec = jm.start_upscale_job({"inputs": [str(ep1)], "dry_run": True})
    per_job_log = root / "logs" / f"web_job_{job_rec['job_id']}.log"
    assert per_job_log.is_file()
    hub = TelemetryHub(root, gpu_query_fn=_idle_gpu_metrics, job_state_provider=lambda: {"active_job": job_rec})
    assert hub._select_active_log_file() == per_job_log.resolve()
    with jm._lock:
        jm._finalize_locked(0)
        jm._finalize_locked(0)
    assert len(jm.get_state()["history"]) == 1

    # 7. Scheduler parity (no double-subtracted 300s subtitles_overhead) & dynamic --dedup-policy CLI
    sched_cli = compute_overnight_schedule(
        queue=[str(ep1), str(ep2)],
        deadline_str="23:59",
        fps_median=10.5,
        fps_p5=8.5,
        video_overhead_seconds=420.0,
        subtitles_overhead_seconds=300.0,
    )
    ep1_dec = sched_cli["episodes"][0]["decision_eta_seconds"]
    ep2_dec = sched_cli["episodes"][1]["decision_eta_seconds"]
    assert abs((ep1_dec + ep2_dec) - (sched_cli["episodes"][0]["eta_p5_seconds"] + sched_cli["episodes"][1]["eta_p5_seconds"]) * 1.07) < 0.5

    parser = upscale.build_arg_parser()
    assert parser.parse_args(["ep01.mkv", "--dedup-policy", "dis-0.25"]).dedup_policy == "dis-0.25"
    assert parser.parse_args(["ep01.mkv", "--dedup-policy", "drift-0.6"]).dedup_policy == "drift-0.6"


def test_sse_nan_in_live_status_strict_json_contract(tmp_path: Path):
    """Lock non-RFC8259 NaN/Infinity JSON emission in /api/events SSE snapshot."""
    root = _seed_workspace(tmp_path / "sse_nan")
    live_file = root / "logs" / "live_status.json"
    live_file.parent.mkdir(parents=True, exist_ok=True)
    live_file.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "stage": "upscaling",
                "steady_fps": float("nan"),
                "latest_fps": float("inf"),
            },
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    def _reject_nan(tok: str):
        raise ValueError(f"Non-RFC8259 JSON constant: {tok}")

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        resp = client.get("/api/events?once=1")
    assert resp.status_code == 200

    data_line = next(
        (ln[len("data: "):] for ln in resp.text.splitlines() if ln.startswith("data: ")),
        None,
    )
    assert data_line is not None
    parsed = json.loads(data_line, parse_constant=_reject_nan)
    assert parsed["pipeline"]["live"]["steady_fps"] is None


def test_pre_runlogger_upscale_failure_does_not_leak_prior_live_status(tmp_path: Path):
    """Lock stale prior-episode live_status.json bleed when a new upscale job fails before RunLogger."""
    from datetime import datetime
    import time as _time
    from web.sse_hub import TelemetryHub

    root = _seed_workspace(tmp_path / "pre_runlogger_leak")
    live_file = root / "logs" / "live_status.json"
    live_file.parent.mkdir(parents=True, exist_ok=True)
    live_file.write_text(
        json.dumps(
            {
                "video_name": "OldCompletedEpisode.mkv",
                "pid": 999999,
                "stage": "completed",
                "total_chunks": 18,
                "completed_chunks": 18,
                "chunks": [{"chunk_idx": 0, "frames": 192}],
            }
        ),
        encoding="utf-8",
    )
    old_ts = _time.time() - 30.0
    os.utime(live_file, (old_ts, old_ts))

    hub = TelemetryHub(
        root,
        gpu_query_fn=_idle_gpu_metrics,
        job_state_provider=lambda: {
            "active_job": None,
            "history": [
                {
                    "job_id": "fail1",
                    "job_type": "upscale",
                    "status": "failed",
                    "pid": 60602,
                    "finished_at": datetime.now().isoformat(),
                    "return_code": 1,
                    "queue_items": [str(root / "anime_season" / "NewFailedEp.mkv")],
                    "options": {},
                }
            ],
        },
    )
    pipe = hub.read_pipeline_state()
    assert pipe["effective_stage"] == "error"
    assert pipe["live"]["video_name"] == "NewFailedEp.mkv"
    assert pipe["live"]["pid"] == 60602
    assert pipe["live"]["total_chunks"] == 0
    assert pipe["live"]["chunks"] == []


def test_mock_pause_graceful_does_not_mutate_disk_flag(tmp_path: Path):
    """Lock --mock mode mutating workspace logs/.graceful_pause.flag on POST /api/jobs/pause-graceful."""
    app = create_app(workspace_root=tmp_path, mock_mode=True)
    with TestClient(app) as client:
        resp = client.post("/api/jobs/pause-graceful")
    assert resp.status_code == 200
    assert not (tmp_path / "logs" / ".graceful_pause.flag").exists()


def test_mock_get_status_is_idempotent_and_monotonic(tmp_path: Path):
    """Lock GET /api/status server state mutation and backwards chunk wraparound in --mock mode."""
    app = create_app(workspace_root=tmp_path, mock_mode=True)
    with TestClient(app) as client:
        snaps = [client.get("/api/status").json()["pipeline"]["live"] for _ in range(6)]
    assert all(s["completed_chunks"] == snaps[0]["completed_chunks"] for s in snaps[1:])
    assert all(snaps[i]["completed_chunks"] >= snaps[i - 1]["completed_chunks"] for i in range(1, 6))


def test_job_manager_schedule_preview_ignores_stale_policy_plan_json(tmp_path: Path):
    """Lock JobManager.compute_schedule_preview serving stale plan.json across dedup_override changes."""
    from web.job_manager import JobManager

    root = _seed_workspace(tmp_path / "jm_pol")
    ep = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    plan_file = root / "out" / ".temp_[BD 1080p] Fate_Ep22" / "plan.json"
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(
        json.dumps(
            {
                "source_file": ep.name,
                "dedup_policy": "dis",
                "total_frames": 24000,
                "total_kept_frames": 9600,
                "dedup_ratio_percent": 60.0,
                "chunks": [{"total_frames": 24000}],
            }
        ),
        encoding="utf-8",
    )

    jm = JobManager(root)
    sched = jm.compute_schedule_preview([str(ep)], dedup_override="off")
    ep_info = sched["episodes"][0]
    assert ep_info["dedup_policy"] == "off"
    assert ep_info["n_unique"] != 9600
    assert ep_info["dedup_ratio_percent"] == 0.0


def test_job_manager_subtrans_srt_forwards_subs_upgrade(tmp_path: Path):
    """Lock JobManager.start_subtrans_job dropping subs_upgrade=True for standalone .srt files."""
    from web.job_manager import JobManager

    root = _seed_workspace(tmp_path / "jm_sub")
    srt_path = root / "anime_season" / "Frieren_02.ja.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nテスト\n", encoding="utf-8")

    captured_cmds: List[List[str]] = []

    def capture_popen(cmd: List[str], **_kwargs: Any) -> FakePopenProcess:
        captured_cmds.append(list(cmd))
        return FakePopenProcess(cmd, pid=70001, running=False)

    jm = JobManager(root, popen_factory=capture_popen)
    jm.start_subtrans_job({"input_path": str(srt_path), "subs_upgrade": True, "dry_run": True})
    assert len(captured_cmds) == 1
    assert "--subs-upgrade" in captured_cmds[0]
    assert "--yes" not in captured_cmds[0]


def test_models_and_shows_exposes_auto_pareto_and_nvenc_multipass_backend_contract(studio_env):
    """Verify backend /api/config/models-and-shows and /api/jobs/start support auto-pareto and nvenc_multipass."""
    client = studio_env["client"]
    resp = client.get("/api/config/models-and-shows")
    assert resp.status_code == 200
    data = resp.json()
    assert "auto-pareto" in {m["id"] for m in data["routing_modes"]}
    assert data["nvenc_multipass_policies"] == ["auto", "fullres", "qres"]

    ep1 = studio_env["root"] / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    start_resp = client.post(
        "/api/jobs/start",
        json={
            "inputs": [str(ep1)],
            "model": "auto-pareto",
            "nvenc_multipass": "fullres",
            "dry_run": True,
        },
    )
    assert start_resp.status_code == 200
    cmd = studio_env["spawned"][-1]["cmd"]
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "auto-pareto"
    assert "--nvenc-multipass" in cmd and cmd[cmd.index("--nvenc-multipass") + 1] == "fullres"


def test_frontend_wires_auto_pareto_and_nvenc_multipass_controls(studio_env):
    """Lock missing auto-pareto routing_modes and nvenc_multipass selector in Web Studio UI."""
    client = studio_env["client"]
    html = client.get("/").text
    panels_js = client.get("/static/js/panels.js").text
    app_js = client.get("/static/js/app.js").text
    assert "routing_modes" in panels_js
    assert ("nvenc_multipass" in html or "nvenc-multipass" in html) and "nvenc_multipass" in panels_js
    assert "opt-nvenc-multipass" in html
    assert "24s" in html and "24s" in app_js
    assert "8s" not in html and "8s" not in app_js


def test_job_manager_prunes_web_job_logs_to_20_preserving_active_and_latest(tmp_path: Path):
    """Verify _spawn_command retains newest N=20 web_job_*.log files without deleting active or latest.log."""
    import time as _time
    from web.job_manager import JobManager, _MAX_RETAINED_WEB_JOB_LOGS

    assert _MAX_RETAINED_WEB_JOB_LOGS == 20
    root = _seed_workspace(tmp_path / "jm_prune")
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    latest_log = logs_dir / "latest.log"
    latest_log.write_text("keep latest\n", encoding="utf-8")

    seeded: List[Path] = []
    base_ts = _time.time() - 500.0
    for idx in range(25):
        p = logs_dir / f"web_job_{idx:010d}.log"
        p.write_text(f"log {idx}\n", encoding="utf-8")
        ts = base_ts + idx * 10.0
        os.utime(p, (ts, ts))
        seeded.append(p)

    srt_path = root / "anime_season" / "PruneTest.ja.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nテスト\n", encoding="utf-8")

    jm = JobManager(
        root,
        popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=71001, running=False),
    )
    job = jm.start_subtrans_job({"input_path": str(srt_path), "dry_run": True})
    active_log = Path(job["log_path"])

    remaining = list(logs_dir.glob("web_job_*.log"))
    assert len(remaining) == 20
    assert active_log.is_file()
    assert latest_log.is_file()
    # Oldest 6 seeded logs (0..5) must be pruned; newest 19 (6..24) + new active_log remain
    for old_p in seeded[:6]:
        assert not old_p.exists()
    for keep_p in seeded[6:]:
        assert keep_p.is_file()


def test_lifespan_cleans_web_compare_pngs_on_startup_and_shutdown(tmp_path: Path):
    """Verify FastAPI _lifespan removes scratch/web_compare/*.png on startup and shutdown."""
    root = _seed_workspace(tmp_path / "lifespan_cleanup")
    compare_dir = root / "scratch" / "web_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    stale_png = compare_dir / "stale_prev.png"
    stale_png.write_bytes(PNG_1X1_BYTES)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        assert not stale_png.exists()
        session_png = compare_dir / "session_temp.png"
        session_png.write_bytes(PNG_1X1_BYTES)
        assert client.get("/api/status").status_code == 200
        assert session_png.is_file()

    assert not session_png.exists()


def test_api_audits_stat_guarded_ttl_cache_and_invalidation(tmp_path: Path, monkeypatch):
    """Verify /api/audits caches within 1.5s TTL and invalidates immediately when audit files change."""
    import web.app as web_app_mod

    root = _seed_workspace(tmp_path / "audits_cache")
    audit1 = root / "out" / "Ep01_4K.audit.json"
    audit1.write_text(json.dumps({"status": "PASS", "source_frames": 100, "output_frames": 100}), encoding="utf-8")

    scan_calls = {"n": 0}
    orig_scan = web_app_mod.scan_temp_checkpoints

    def _counting_scan(ws_root: Path, extra_dirs=None):
        scan_calls["n"] += 1
        return orig_scan(ws_root, extra_dirs=extra_dirs)

    monkeypatch.setattr(web_app_mod, "scan_temp_checkpoints", _counting_scan)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        r1 = client.get("/api/audits")
        r2 = client.get("/api/audits")
        r3 = client.get("/api/audits")
        assert r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 200
        assert len(r1.json()["audits"]) == 1
        assert scan_calls["n"] == 1

        # Writing a new audit.json within the 1.5s TTL window must invalidate cache immediately
        audit2 = root / "out" / "Ep02_4K.audit.json"
        audit2.write_text(json.dumps({"status": "PASS", "source_frames": 200, "output_frames": 200}), encoding="utf-8")
        r4 = client.get("/api/audits")
        assert r4.status_code == 200
        assert len(r4.json()["audits"]) == 2
        assert scan_calls["n"] == 2

        # Creating a .temp_* dir must also invalidate immediately
        (root / "out" / ".temp_Ep03").mkdir(parents=True, exist_ok=True)
        r5 = client.get("/api/audits")
        assert r5.status_code == 200
        assert scan_calls["n"] == 3


def test_dotenv_key_priority_parity_and_midrun_vs_prerunlogger_crash(tmp_path: Path, monkeypatch):
    """Verify .env key priority matches subtrans/config.py and mid-run crashes retain chunk telemetry."""
    from datetime import datetime
    import time as _time
    import subtrans.config as subtrans_cfg
    from web.app import _resolve_subtrans_key_slots
    from web.sse_hub import TelemetryHub

    for var in ("SUBTRANS_DISABLE_DOTENV", "PYTHON_DOTENV_DISABLED", "GEMINI_API_KEYS", "GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        monkeypatch.delenv(var, raising=False)

    root = _seed_workspace(tmp_path / "env_parity")
    dotenv_file = root / ".env"
    dotenv_file.write_text("GEMINI_API_KEY=dotenv_primary_secret_1\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY_1", "env_slot1_secret_2")

    slots, count = _resolve_subtrans_key_slots(dotenv_file)
    assert count == 2
    assert slots[0]["env_name"] == "GEMINI_API_KEY"
    assert slots[0]["source"] == "dotenv"

    # Cross-check exact order parity with subtrans/config.py:get_gemini_api_keys()
    monkeypatch.setenv("GEMINI_API_KEY", "dotenv_primary_secret_1")
    cfg_keys = subtrans_cfg.get_gemini_api_keys()
    assert cfg_keys[0] == "dotenv_primary_secret_1"
    assert cfg_keys[1] == "env_slot1_secret_2"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_1", raising=False)

    # Unwrapping GEMINI_API_KEYS while updating key_2 syncs over stale single-token GEMINI_API_KEY
    dotenv_file.write_text(
        "GEMINI_API_KEY=stale_single_key\nGEMINI_API_KEYS=list_slot1_key;list_slot2_key\n",
        encoding="utf-8",
    )
    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        r_upd = client.put("/api/subtrans/api-keys", json={"key_2": "new_slot2_key"})
        assert r_upd.status_code == 200
        upd_slots = r_upd.json()["key_slots"]
        assert "list_s" in upd_slots[0]["masked"]
        assert "new_sl" in upd_slots[1]["masked"]

    # Mid-run crash (matching pid + recent mtime) retains chunk telemetry; prior-episode error does not leak
    live_file = root / "logs" / "live_status.json"
    live_file.write_text(
        json.dumps(
            {
                "video_name": "MidRunEp.mkv",
                "pid": 61616,
                "stage": "upscaling",
                "total_chunks": 10,
                "completed_chunks": 3,
                "chunks": [{"chunk_idx": 0}, {"chunk_idx": 1}, {"chunk_idx": 2}],
            }
        ),
        encoding="utf-8",
    )
    now_iso = datetime.now().isoformat()
    hub = TelemetryHub(
        root,
        gpu_query_fn=_idle_gpu_metrics,
        job_state_provider=lambda: {
            "active_job": None,
            "history": [
                {
                    "job_id": "mid1",
                    "job_type": "upscale",
                    "status": "failed",
                    "pid": 61616,
                    "started_at": now_iso,
                    "finished_at": now_iso,
                    "return_code": 1,
                    "queue_items": [str(root / "anime_season" / "MidRunEp.mkv")],
                    "options": {},
                }
            ],
        },
    )
    mid_pipe = hub.read_pipeline_state()
    assert mid_pipe["effective_stage"] == "error"
    assert mid_pipe["live"]["total_chunks"] == 10
    assert len(mid_pipe["live"]["chunks"]) == 3

    # Pre-RunLogger crash when prior live_status.json already had stage='error' must NOT leak old chunks
    live_file.write_text(
        json.dumps(
            {
                "video_name": "OldErrorEp.mkv",
                "pid": 999999,
                "stage": "error",
                "total_chunks": 12,
                "completed_chunks": 4,
                "chunks": [{"chunk_idx": 0}],
            }
        ),
        encoding="utf-8",
    )
    old_ts = _time.time() - 30.0
    os.utime(live_file, (old_ts, old_ts))
    hub_pre = TelemetryHub(
        root,
        gpu_query_fn=_idle_gpu_metrics,
        job_state_provider=lambda: {
            "active_job": None,
            "history": [
                {
                    "job_id": "fail_new",
                    "job_type": "upscale",
                    "status": "failed",
                    "pid": 62626,
                    "started_at": now_iso,
                    "finished_at": now_iso,
                    "return_code": 1,
                    "queue_items": [str(root / "anime_season" / "FreshFail.mkv")],
                    "options": {},
                }
            ],
        },
    )
    pre_pipe = hub_pre.read_pipeline_state()
    assert pre_pipe["effective_stage"] == "error"
    assert pre_pipe["live"]["video_name"] == "FreshFail.mkv"
    assert pre_pipe["live"]["total_chunks"] == 0
    assert pre_pipe["live"]["chunks"] == []


def test_sse_o1_log_polling_and_mock_control_disk_isolation(tmp_path: Path, monkeypatch):
    """Verify O(1) SSE log polling across 3 tabs and --mock clear-pause/abort disk isolation & monotonic ticks."""
    import asyncio
    from web.sse_hub import TelemetryHub

    root = _seed_workspace(tmp_path / "sse_o1")
    latest_log = root / "logs" / "latest.log"
    latest_log.write_text("[00:00:01] Boot\n", encoding="utf-8")

    hub = TelemetryHub(root, gpu_query_fn=_idle_gpu_metrics)
    select_calls = {"n": 0}
    orig_select = hub._select_active_log_file

    def _counted_select():
        select_calls["n"] += 1
        return orig_select()

    monkeypatch.setattr(hub, "_select_active_log_file", _counted_select)

    def _parse_data(raw: str) -> Dict[str, Any]:
        for ln in raw.splitlines():
            if ln.startswith("data: "):
                return json.loads(ln[len("data: "):])
        raise AssertionError(raw)

    async def _run_three_tabs() -> None:
        g1 = hub.stream_events()
        g2 = hub.stream_events()
        g3 = hub.stream_events()
        try:
            await anext(g1)
            await anext(g2)
            await anext(g3)
            select_calls["n"] = 0

            with latest_log.open("a", encoding="utf-8") as f:
                f.write("[00:00:02] Shared O(1) tick line 1\n")

            t1, t2, t3 = await asyncio.gather(anext(g1), anext(g2), anext(g3))
            for raw in (t1, t2, t3):
                assert "[00:00:02] Shared O(1) tick line 1" in _parse_data(raw)["logs"]
            # 1 tick shared across 3 subscribers -> 1 _select_active_log_file call (not 3)
            assert select_calls["n"] == 1
        finally:
            await g1.aclose()
            await g2.aclose()
            await g3.aclose()
            await hub.shutdown()

    asyncio.run(_run_three_tabs())

    # --mock mode: clear-pause / abort never touch disk flags or .gpu_active.lock, and ticks stay monotonic
    mock_ws = tmp_path / "mock_iso"
    (mock_ws / "logs").mkdir(parents=True, exist_ok=True)
    app = create_app(workspace_root=mock_ws, mock_mode=True)
    with TestClient(app) as client:
        assert client.post("/api/jobs/pause-graceful").status_code == 200
        assert client.get("/api/status").json()["pipeline"]["graceful_pause_pending"] is True
        assert not (mock_ws / "logs" / ".graceful_pause.flag").exists()

        assert client.post("/api/jobs/clear-pause").status_code == 200
        assert client.get("/api/status").json()["pipeline"]["graceful_pause_pending"] is False
        assert not (mock_ws / "logs" / ".graceful_pause.flag").exists()

        # Simulate stale lock file on disk: --mock abort must NOT unlink real workspace lock
        lock_path = mock_ws / ".gpu_active.lock"
        lock_path.write_text('{"pid": 999999}', encoding="utf-8")
        abort_res = client.post("/api/jobs/abort")
        assert abort_res.status_code == 200
        assert abort_res.json()["mock"] is True
        assert lock_path.is_file()

        # Verify SSE ticks advance monotonically without wraparound across >6 ticks
        completed_seq: List[int] = []
        for _ in range(9):
            pkt = app.state.telemetry_hub._poll_tick_sync()
            completed_seq.append(pkt["pipeline"]["live"]["completed_chunks"])
        assert completed_seq == sorted(completed_seq)
        assert completed_seq[-1] == 11


def test_dotenv_key_slots_alignment_empty_blocking_and_put_edits(tmp_path: Path, monkeypatch):
    """Verify _resolve_subtrans_key_slots and PUT /api/subtrans/api-keys handle mixed vars, empty overrides, and edits."""
    from dotenv import dotenv_values
    import subtrans.config as subtrans_cfg
    from web.app import _resolve_subtrans_key_slots

    for var in (
        "SUBTRANS_DISABLE_DOTENV",
        "PYTHON_DOTENV_DISABLED",
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
        "GEMINI_API_KEY_4",
    ):
        monkeypatch.delenv(var, raising=False)

    def _get_keys_from_dotenv(path: Path) -> List[str]:
        for k in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"):
            monkeypatch.delenv(k, raising=False)
        for k, v in dotenv_values(path).items():
            if v is not None:
                monkeypatch.setenv(k, v)
        return subtrans_cfg.get_gemini_api_keys()

    root = _seed_workspace(tmp_path / "env_slots_align")
    dotenv_file = root / ".env"

    # 1. Mixed GEMINI_API_KEYS + GEMINI_API_KEY + GEMINI_API_KEY_2 aligns slots 1..3 in pool order
    dotenv_file.write_text(
        "GEMINI_API_KEYS=keyAlpha111111\nGEMINI_API_KEY=keyBeta222222\nGEMINI_API_KEY_2=keyDelta444444\n",
        encoding="utf-8",
    )
    slots_mix, cnt_mix = _resolve_subtrans_key_slots(dotenv_file)
    assert cnt_mix == 3
    assert [s["env_name"] for s in slots_mix] == ["GEMINI_API_KEYS", "GEMINI_API_KEY", "GEMINI_API_KEY_2"]
    assert "keyAlp" in slots_mix[0]["masked"]
    assert "keyBet" in slots_mix[1]["masked"]
    assert "keyDel" in slots_mix[2]["masked"]

    # 2. Empty subtrans/.env or os.environ entry blocks fallback to lower-precedence .env (load_dotenv parity)
    sub_env = root / "subtrans" / ".env"
    sub_env.parent.mkdir(parents=True, exist_ok=True)
    dotenv_file.write_text("GEMINI_API_KEY=rootFallback999\n", encoding="utf-8")
    sub_env.write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    slots_sub_empty, cnt_sub_empty = _resolve_subtrans_key_slots(dotenv_file)
    assert cnt_sub_empty == 0
    assert all(not s["configured"] for s in slots_sub_empty)
    sub_env.unlink()

    monkeypatch.setenv("GEMINI_API_KEY", "")
    slots_env_empty, cnt_env_empty = _resolve_subtrans_key_slots(dotenv_file)
    assert cnt_env_empty == 0
    assert all(not s["configured"] for s in slots_env_empty)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # 3. Edit B: GEMINI_API_KEYS (2 keys) + GEMINI_API_KEY (3rd key) preserves slot 3 on key_2 update
    dotenv_file.write_text(
        "GEMINI_API_KEYS=k1_alpha11,k2_beta22\nGEMINI_API_KEY=k3_gamma33\n",
        encoding="utf-8",
    )
    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        r_b = client.put("/api/subtrans/api-keys", json={"key_2": "k2_updated22"})
        assert r_b.status_code == 200
        assert r_b.json()["configured_count"] == 3
    assert _get_keys_from_dotenv(dotenv_file) == ["k1_alpha11", "k2_updated22", "k3_gamma33"]

    # 4. Edit C: GEMINI_API_KEY + stale GEMINI_API_KEY_1 + GEMINI_API_KEY_2 unwraps without k_old1 shadowing key_2
    for var in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        monkeypatch.delenv(var, raising=False)
    dotenv_file.write_text(
        "GEMINI_API_KEY=k1_alpha11\nGEMINI_API_KEY_1=k_old_shadow\nGEMINI_API_KEY_2=k2_beta22\n",
        encoding="utf-8",
    )
    with TestClient(app) as client:
        r_c = client.put("/api/subtrans/api-keys", json={"key_2": "k2_updated22"})
        assert r_c.status_code == 200
        assert r_c.json()["configured_count"] == 2
    assert _get_keys_from_dotenv(dotenv_file) == ["k1_alpha11", "k2_updated22"]


def test_read_pipeline_state_subtrans_aborted_and_pid_none_crash_isolation(tmp_path: Path):
    """Verify subtrans aborted resets stale live_status.json and pid=None crash never matches prior file via mtime."""
    from datetime import datetime
    from web.sse_hub import TelemetryHub

    root = _seed_workspace(tmp_path / "pipe_edge")
    live_file = root / "logs" / "live_status.json"
    live_file.parent.mkdir(parents=True, exist_ok=True)
    live_file.write_text(
        json.dumps(
            {
                "video_name": "old_failed.mkv",
                "pid": 999999,
                "stage": "error",
                "total_chunks": 8,
                "completed_chunks": 3,
                "chunks": [{"chunk_idx": 0}, {"chunk_idx": 1}, {"chunk_idx": 2}],
            }
        ),
        encoding="utf-8",
    )
    now_iso = datetime.now().isoformat()

    hub_sub_aborted = TelemetryHub(
        root,
        gpu_query_fn=_idle_gpu_metrics,
        job_state_provider=lambda: {
            "active_job": None,
            "history": [
                {
                    "job_id": "sub_ab1",
                    "job_type": "subtrans",
                    "status": "aborted",
                    "pid": 77778,
                    "started_at": now_iso,
                    "finished_at": now_iso,
                    "queue_items": [str(root / "anime_season" / "AbortedSub.srt")],
                    "options": {"subs_quality": "high"},
                }
            ],
        },
    )
    sub_pipe = hub_sub_aborted.read_pipeline_state()
    assert sub_pipe["effective_stage"] == "idle"
    assert sub_pipe["live"]["video_name"] == "AbortedSub.srt"
    assert sub_pipe["live"]["stage"] == "idle"
    assert sub_pipe["live"]["total_chunks"] == 0
    assert sub_pipe["live"]["chunks"] == []

    hub_pid_none = TelemetryHub(
        root,
        gpu_query_fn=_idle_gpu_metrics,
        job_state_provider=lambda: {
            "active_job": None,
            "history": [
                {
                    "job_id": "fail_none_pid",
                    "job_type": "upscale",
                    "status": "failed",
                    "pid": None,
                    "started_at": now_iso,
                    "finished_at": now_iso,
                    "return_code": 1,
                    "queue_items": [str(root / "anime_season" / "NoPidCrash.mkv")],
                    "options": {},
                }
            ],
        },
    )
    fail_pipe = hub_pid_none.read_pipeline_state()
    assert fail_pipe["effective_stage"] == "error"
    assert fail_pipe["live"]["video_name"] == "NoPidCrash.mkv"
    assert fail_pipe["live"]["total_chunks"] == 0
    assert fail_pipe["live"]["chunks"] == []


def test_api_audits_invalidates_on_checkpoint_upscaled_seg_and_cast_artifacts(tmp_path: Path):
    """Verify /api/audits invalidates immediately when up_0000.mp4 or cast YAML is created within TTL."""
    root = _seed_workspace(tmp_path / "audits_ckpt_inval")
    audit1 = root / "out" / "Ep01_4K.audit.json"
    audit1.write_text(json.dumps({"status": "PASS", "source_frames": 100, "output_frames": 100}), encoding="utf-8")
    up_dir = root / "out" / ".temp_Ep01" / "upscaled_segs"
    up_dir.mkdir(parents=True, exist_ok=True)
    cast_dir = root / "out" / "cast"
    cast_dir.mkdir(parents=True, exist_ok=True)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        r1 = client.get("/api/audits").json()
        assert r1["checkpoints"][0]["completed_chunks"] == 0
        assert r1["audits"][0]["cast_yaml_path"] == ""

        (up_dir / "up_0000.mp4").write_bytes(b"\x00" * 16)
        r2 = client.get("/api/audits").json()
        assert r2["checkpoints"][0]["completed_chunks"] == 1

        (cast_dir / "Ep01.cast.yaml").write_text("characters: []\n", encoding="utf-8")
        r3 = client.get("/api/audits").json()
        assert r3["audits"][0]["cast_yaml_path"].endswith("Ep01.cast.yaml")


def test_mock_sse_o1_cursor_resets_existing_disk_log_and_aligns_prefix_len(tmp_path: Path, monkeypatch):
    """Verify --mock mode resets _last_log_cursor over existing logs/latest.log and keeps O(1) log sharing."""
    import asyncio

    mock_ws = tmp_path / "mock_with_disk_log"
    (mock_ws / "models").mkdir(parents=True, exist_ok=True)
    (mock_ws / "out").mkdir(parents=True, exist_ok=True)
    (mock_ws / "logs").mkdir(parents=True, exist_ok=True)
    (mock_ws / "logs" / "latest.log").write_text("existing disk line\n" * 400, encoding="utf-8")

    app = create_app(workspace_root=mock_ws, mock_mode=True)
    hub = app.state.telemetry_hub

    inc_calls = {"n": 0}
    orig_inc = hub._read_incremental_logs_from_cursor

    def _counted_inc(cursor):
        inc_calls["n"] += 1
        return orig_inc(cursor)

    monkeypatch.setattr(hub, "_read_incremental_logs_from_cursor", _counted_inc)

    def _parse_data(raw: str) -> Dict[str, Any]:
        for ln in raw.splitlines():
            if ln.startswith("data: "):
                return json.loads(ln[len("data: "):])
        raise AssertionError(raw)

    async def _verify_mock_o1() -> None:
        g1 = hub.stream_events()
        g2 = hub.stream_events()
        try:
            s1 = _parse_data(await anext(g1))
            s2 = _parse_data(await anext(g2))
            assert s1["log_cursor"]["path"] == "mock://telemetry.log"
            assert s1["log_cursor"]["prefix_len"] == 1
            assert s2["log_cursor"]["prefix_len"] == 1
            inc_calls["n"] = 0

            t1_1, t1_2 = await asyncio.gather(anext(g1), anext(g2))
            p1_1, p1_2 = _parse_data(t1_1), _parse_data(t1_2)
            assert any("[mock] SSE tick #1" in ln for ln in p1_1["logs"])
            assert p1_1["logs"] == p1_2["logs"]
            # Delayed subscriber joins after Tick 1
            g3 = hub.stream_events()
            try:
                _parse_data(await anext(g3))
                t2_1, t2_2, t2_3 = await asyncio.gather(anext(g1), anext(g2), anext(g3))
                for raw in (t2_1, t2_2, t2_3):
                    assert any("[mock] SSE tick #2" in ln for ln in _parse_data(raw)["logs"])
                # Exactly 1 incremental read per tick across 2 ticks (O(1), not per-subscriber)
                assert inc_calls["n"] == 2
            finally:
                await g3.aclose()
        finally:
            await g1.aclose()
            await g2.aclose()
            await hub.shutdown()

    asyncio.run(_verify_mock_o1())


def test_mixed_var_dotenv_key_edits_preserve_extra_and_avoid_slot_leakage(tmp_path: Path, monkeypatch):
    """Verify mixed GEMINI_API_KEYS/GEMINI_API_KEY/GEMINI_API_KEY_2 edits keep all keys in canonical order."""
    from dotenv import dotenv_values
    import subtrans.config as subtrans_cfg

    env_vars = (
        "SUBTRANS_DISABLE_DOTENV",
        "PYTHON_DOTENV_DISABLED",
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
        "GEMINI_API_KEY_4",
    )

    def _clear_env() -> None:
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)

    def _get_keys_from_dotenv(path: Path) -> List[str]:
        _clear_env()
        for k, v in dotenv_values(path).items():
            if v is not None:
                monkeypatch.setenv(k, v)
        keys = subtrans_cfg.get_gemini_api_keys()
        _clear_env()
        return keys

    _clear_env()
    root = _seed_workspace(tmp_path / "mixed_env_ws")
    dotenv_file = root / ".env"
    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)

    # Case A: GEMINI_API_KEY=k1,k2 + GEMINI_API_KEY_2=k_extra
    dotenv_file.write_text("GEMINI_API_KEY=k1_alpha,k2_beta\nGEMINI_API_KEY_2=k_extra\n", encoding="utf-8")
    with TestClient(app) as client:
        r1 = client.put("/api/subtrans/api-keys", json={"key_2": "k2_upd"})
        assert r1.status_code == 200
        assert r1.json()["configured_count"] == 3
        assert _get_keys_from_dotenv(dotenv_file) == ["k1_alpha", "k2_upd", "k_extra"]

        r2 = client.put("/api/subtrans/api-keys", json={"key_1": "k1_upd"})
        assert r2.status_code == 200
        assert r2.json()["configured_count"] == 3
        assert _get_keys_from_dotenv(dotenv_file) == ["k1_upd", "k2_upd", "k_extra"]

    # Case B: GEMINI_API_KEYS=k1,k2 + GEMINI_API_KEY_2=k_extra
    dotenv_file.write_text("GEMINI_API_KEYS=k1_alpha,k2_beta\nGEMINI_API_KEY_2=k_extra\n", encoding="utf-8")
    with TestClient(app) as client:
        r3 = client.put("/api/subtrans/api-keys", json={"key_2": "k2_upd"})
        assert r3.status_code == 200
        assert r3.json()["configured_count"] == 3
        assert _get_keys_from_dotenv(dotenv_file) == ["k1_alpha", "k2_upd", "k_extra"]

        r4 = client.put("/api/subtrans/api-keys", json={"key_1": "k1_upd"})
        assert r4.status_code == 200
        assert r4.json()["configured_count"] == 3
        assert _get_keys_from_dotenv(dotenv_file) == ["k1_upd", "k2_upd", "k_extra"]

    # Case C: GEMINI_API_KEYS=k1 + GEMINI_API_KEY=k2,k3 (replacing slot 2 must not leak k2 into slot 3)
    dotenv_file.write_text("GEMINI_API_KEYS=k1_alpha\nGEMINI_API_KEY=k2_beta,k3_gamma\n", encoding="utf-8")
    with TestClient(app) as client:
        r5 = client.put("/api/subtrans/api-keys", json={"key_2": "k2_upd"})
        assert r5.status_code == 200
        assert r5.json()["configured_count"] == 3
        assert _get_keys_from_dotenv(dotenv_file) == ["k1_alpha", "k2_upd", "k3_gamma"]


def test_lan_control_api_keys_put_masks_non_loopback_responses(tmp_path: Path, monkeypatch):
    """Verify --lan --lan-control strips masked API key previews on PUT /api/subtrans/api-keys for non-loopback peers."""
    for var in (
        "SUBTRANS_DISABLE_DOTENV",
        "PYTHON_DOTENV_DISABLED",
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
    ):
        monkeypatch.delenv(var, raising=False)

    root = _seed_workspace(tmp_path / "lan_ctrl_mask_ws")
    (root / ".env").write_text(
        "GEMINI_API_KEY_1=AIzaSySlotOne111111\nGEMINI_API_KEY_2=AIzaSySlotTwo222222\n",
        encoding="utf-8",
    )
    app_ctrl = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        allow_lan=True,
        allow_lan_control=True,
    )

    lan_headers = {"Host": "192.168.1.50:8765", "Origin": "http://192.168.1.50:8765"}
    with TestClient(app_ctrl, client=("192.168.1.99", 54321)) as lan_client:
        ov = lan_client.get("/api/subtrans/overview", headers=lan_headers)
        assert ov.status_code == 200
        ov_data = ov.json()
        assert ov_data["configured_keys_count"] == 2
        assert all(s["masked"] == "" for s in ov_data["key_slots"])

        put_res = lan_client.put(
            "/api/subtrans/api-keys",
            json={"key_3": "AIzaSySlotThree3333"},
            headers=lan_headers,
        )
        assert put_res.status_code == 200
        put_json = put_res.json()
        assert put_json["configured_count"] == 3
        assert [s["configured"] for s in put_json["key_slots"]] == [True, True, True]
        assert all(s["masked"] == "" for s in put_json["key_slots"])
        dumped = json.dumps(put_json)
        for secret in ("AIzaSySlotOne111111", "AIzaSySlotTwo222222", "AIzaSySlotThree3333"):
            assert secret not in dumped

        spoof_headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        spoof_res = lan_client.put(
            "/api/subtrans/api-keys",
            json={"key_2": "AIzaSySlotTwoUpdated"},
            headers=spoof_headers,
        )
        assert spoof_res.status_code == 200
        assert all(s["masked"] == "" for s in spoof_res.json()["key_slots"])

    with TestClient(app_ctrl) as tc_lan_host:
        non_lb_host_res = tc_lan_host.put(
            "/api/subtrans/api-keys",
            json={"key_1": "AIzaSySlotOneUpdated"},
            headers=lan_headers,
        )
        assert non_lb_host_res.status_code == 200
        assert all(s["masked"] == "" for s in non_lb_host_res.json()["key_slots"])

    with TestClient(app_ctrl) as local_client:
        local_headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        local_res = local_client.put(
            "/api/subtrans/api-keys",
            json={"key_1": "AIzaSySlotOne111111"},
            headers=local_headers,
        )
        assert local_res.status_code == 200
        local_slots = local_res.json()["key_slots"]
        assert local_slots[0]["masked"] == "AIzaSy...1111"
        assert local_slots[1]["masked"] == "AIzaSy...ated"
        assert local_slots[2]["masked"] == "AIzaSy...3333"


def test_sse_bool_pid_rejection_and_subtrans_failed_baseline_keys(tmp_path: Path, monkeypatch):
    """Verify bool PID values are rejected and subtrans failed/error states expose all baseline keys."""
    import psutil
    from web.job_manager import JobManager
    from web.sse_hub import TelemetryHub

    pid_checks: List[Any] = []

    def _forbid_bool_or_one_pid(pid: Any) -> bool:
        pid_checks.append(pid)
        assert not isinstance(pid, bool), f"psutil.pid_exists called with bool: {pid!r}"
        assert pid != 1, "bool True coerced to PID 1"
        return False

    monkeypatch.setattr(psutil, "pid_exists", _forbid_bool_or_one_pid)

    root = _seed_workspace(tmp_path / "bool_pid_ws")
    (root / "logs" / "live_status.json").write_text(
        json.dumps(
            {
                "video_name": "Ep01.mkv",
                "pid": True,
                "stage": "upscaling",
                "updated_at": "2026-11-22T10:00:00",
            }
        ),
        encoding="utf-8",
    )

    state_holder: Dict[str, Any] = {
        "status": "idle",
        "active_job": None,
        "history": [
            {
                "job_id": "job_prev",
                "job_type": "upscale",
                "status": "failed",
                "pid": True,
                "started_at": "2026-11-22T09:59:00",
                "finished_at": "2026-11-22T09:59:30",
                "queue_items": ["Ep01.mkv"],
                "options": {},
            }
        ],
    }
    hub = TelemetryHub(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        job_state_provider=lambda: state_holder,
    )
    snap_idle = hub.read_pipeline_state()
    assert snap_idle["pid_alive"] is False
    assert snap_idle["active_source"] == "idle"

    # Active job with bool pid must not treat live_pid (None) == active_pid (True) as matching
    state_holder["status"] = "running"
    state_holder["active_job"] = {
        "job_id": "job_bool",
        "job_type": "upscale",
        "pid": True,
        "started_at": "2026-11-22T10:01:00",
        "queue_items": ["Ep02.mkv"],
        "options": {"model": "animevideov3", "zero_disk": True},
    }
    snap_active = hub.read_pipeline_state()
    assert snap_active["effective_stage"] == "upscaling"

    # JobManager._ensure_ready_to_launch_locked must ignore bool pid in live_status.json
    jm = JobManager(workspace_root=root)
    with jm._lock:
        jm._ensure_ready_to_launch_locked(requires_gpu=True)

    # subtrans failed and subs_pending error synthetic live_data must contain all 10 baseline keys
    expected_baseline = {
        "model_id",
        "zero_disk",
        "total_frames",
        "unique_frames",
        "total_chunks",
        "completed_chunks",
        "steady_fps",
        "latest_fps",
        "effective_fps",
        "eta_seconds",
    }
    from datetime import datetime, timedelta
    fin_ts = (datetime.now() + timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%S")
    start_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    for job_type, status_val in (("subtrans", "failed"), ("subs_pending", "error")):
        state_holder["status"] = "idle"
        state_holder["active_job"] = None
        state_holder["history"] = [
            {
                "job_id": f"sub_{status_val}",
                "job_type": job_type,
                "status": status_val,
                "pid": 999999,
                "return_code": 1,
                "started_at": start_ts,
                "finished_at": fin_ts,
                "queue_items": ["Ep01.ja.srt"] if job_type == "subtrans" else [],
                "options": {"subs_quality": "high"},
            }
        ]
        snap_sub = hub.read_pipeline_state()
        assert snap_sub["effective_stage"] == "error"
        live_sub = snap_sub["live"]
        assert expected_baseline.issubset(live_sub.keys())
        assert live_sub["model_id"] == "subtrans (high)"
        assert live_sub["zero_disk"] is False
        assert live_sub["total_chunks"] == 0
        assert live_sub["completed_chunks"] == 0
        assert live_sub["eta_seconds"] is None
    assert pid_checks == []


def test_schedule_preview_nvenc_multipass_and_concurrent_api_keys_put(tmp_path: Path, monkeypatch):
    """Verify SchedulePreviewRequest validates nvenc_multipass and concurrent PUT /api/subtrans/api-keys is race-free."""
    from concurrent.futures import ThreadPoolExecutor

    for var in (
        "SUBTRANS_DISABLE_DOTENV",
        "PYTHON_DOTENV_DISABLED",
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
    ):
        monkeypatch.delenv(var, raising=False)

    root = _seed_workspace(tmp_path / "sched_concurrent_ws")
    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)

    with TestClient(app) as client:
        for valid_mp in ("auto", "fullres", "qres", ""):
            res_ok = client.post(
                "/api/queue/preview-schedule",
                json={"inputs": [], "nvenc_multipass": valid_mp},
            )
            assert res_ok.status_code == 200
        res_bad = client.post(
            "/api/queue/preview-schedule",
            json={"inputs": [], "nvenc_multipass": "bogus"},
        )
        assert res_bad.status_code == 422

        payloads = [
            {"key_1": "AIzaSyConcOne111111"},
            {"key_2": "AIzaSyConcTwo222222"},
            {"key_3": "AIzaSyConcThree3333"},
            {"key_1": "AIzaSyConcOne111111"},
            {"key_2": "AIzaSyConcTwo222222"},
            {"key_3": "AIzaSyConcThree3333"},
        ]

        def _put_slot(body: Dict[str, str]) -> int:
            return client.put("/api/subtrans/api-keys", json=body).status_code

        with ThreadPoolExecutor(max_workers=6) as pool:
            statuses = list(pool.map(_put_slot, payloads))
        assert statuses == [200] * 6

        ov = client.get("/api/subtrans/overview")
        assert ov.status_code == 200
        assert ov.json()["configured_keys_count"] == 3
        assert list(root.glob(".env.*.tmp")) == []
        env_text = (root / ".env").read_text(encoding="utf-8")
        assert "GEMINI_API_KEY_1=AIzaSyConcOne111111" in env_text
        assert "GEMINI_API_KEY_2=AIzaSyConcTwo222222" in env_text
        assert "GEMINI_API_KEY_3=AIzaSyConcThree3333" in env_text


def test_env_api_keys_editing_key3_preserves_key1_with_empty_or_populated_key2(tmp_path: Path, monkeypatch):
    """Verify editing key_3 preserves distinct GEMINI_API_KEY_1 with empty GEMINI_API_KEY_2 or 3 populated keys."""
    for var in (
        "SUBTRANS_DISABLE_DOTENV",
        "PYTHON_DOTENV_DISABLED",
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
    ):
        monkeypatch.delenv(var, raising=False)

    # Case A: GEMINI_API_KEY + GEMINI_API_KEY_1 + empty GEMINI_API_KEY_2=, PUT {"key_3": ...}
    root_a = _seed_workspace(tmp_path / "env_key3_empty_k2_ws")
    (root_a / ".env").write_text(
        "GEMINI_API_KEY=AIzaSyPrimaryKey0001\n"
        "GEMINI_API_KEY_1=AIzaSySecondKey0002\n"
        "GEMINI_API_KEY_2=\n",
        encoding="utf-8",
    )
    app_a = create_app(workspace_root=root_a, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app_a) as client_a:
        r_put_a = client_a.put("/api/subtrans/api-keys", json={"key_3": "AIzaSyThirdKey0003"})
        assert r_put_a.status_code == 200
        assert r_put_a.json()["configured_count"] == 3

        r_ov_a = client_a.get("/api/subtrans/overview")
        assert r_ov_a.status_code == 200
        ov_a = r_ov_a.json()
        assert ov_a["configured_keys_count"] == 3
        slots_a = {s["slot"]: s for s in ov_a["key_slots"]}
        assert slots_a[1]["masked"] == "AIzaSy...0001"
        assert slots_a[2]["masked"] == "AIzaSy...0002"
        assert slots_a[3]["masked"] == "AIzaSy...0003"

    env_a = (root_a / ".env").read_text(encoding="utf-8")
    assert "AIzaSyPrimaryKey0001" in env_a
    assert "AIzaSySecondKey0002" in env_a
    assert "AIzaSyThirdKey0003" in env_a

    # Case B: 3 distinct keys (GEMINI_API_KEY, GEMINI_API_KEY_1, GEMINI_API_KEY_2), PUT {"key_3": ...}
    root_b = _seed_workspace(tmp_path / "env_key3_three_keys_ws")
    (root_b / ".env").write_text(
        "GEMINI_API_KEY=AIzaSyPrimaryKey0001\n"
        "GEMINI_API_KEY_1=AIzaSySecondKey0002\n"
        "GEMINI_API_KEY_2=AIzaSyThirdKey0003\n",
        encoding="utf-8",
    )
    app_b = create_app(workspace_root=root_b, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app_b) as client_b:
        r_put_b = client_b.put("/api/subtrans/api-keys", json={"key_3": "AIzaSyUpdatedThree0004"})
        assert r_put_b.status_code == 200
        assert r_put_b.json()["configured_count"] == 3

        r_ov_b = client_b.get("/api/subtrans/overview")
        assert r_ov_b.status_code == 200
        ov_b = r_ov_b.json()
        assert ov_b["configured_keys_count"] == 3
        slots_b = {s["slot"]: s for s in ov_b["key_slots"]}
        assert slots_b[1]["masked"] == "AIzaSy...0001"
        assert slots_b[2]["masked"] == "AIzaSy...0002"
        assert slots_b[3]["masked"] == "AIzaSy...0004"

    env_b = (root_b / ".env").read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=AIzaSyPrimaryKey0001" in env_b
    assert "GEMINI_API_KEY_2=AIzaSySecondKey0002" in env_b
    assert "GEMINI_API_KEY_3=AIzaSyUpdatedThree0004" in env_b
    assert "AIzaSyThirdKey0003" not in env_b


def test_env_api_keys_tier5_preserves_distinct_key1_and_dedups_mirrored(tmp_path: Path, monkeypatch):
    """Verify ADV3-ADV6 and ADV10-ADV12 preserve distinct GEMINI_API_KEY_1 and deduplicate mirrored keys."""
    for var in (
        "SUBTRANS_DISABLE_DOTENV",
        "PYTHON_DOTENV_DISABLED",
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_1",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
        "GEMINI_API_KEY_4",
    ):
        monkeypatch.delenv(var, raising=False)

    def _run_put_and_overview(name: str, initial_env: str, payload: dict) -> tuple[int, dict, str]:
        root = _seed_workspace(tmp_path / name)
        (root / ".env").write_text(initial_env, encoding="utf-8")
        app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
        with TestClient(app) as client:
            r_put = client.put("/api/subtrans/api-keys", json=payload)
            assert r_put.status_code == 200
            r_ov = client.get("/api/subtrans/overview")
            assert r_ov.status_code == 200
            return r_put.json()["configured_count"], r_ov.json(), (root / ".env").read_text(encoding="utf-8")

    # ADV3: 3 distinct keys (GEMINI_API_KEY, GEMINI_API_KEY_1, GEMINI_API_KEY_2), PUT key_1 preserves Slot 2 & 3
    cnt3, ov3, env3 = _run_put_and_overview(
        "adv3_ws",
        "GEMINI_API_KEY=AIzaSyPrimaryKey0001\n"
        "GEMINI_API_KEY_1=AIzaSySecondKey0002\n"
        "GEMINI_API_KEY_2=AIzaSyThirdKey0003\n",
        {"key_1": "AIzaSyUpdatedOne8888"},
    )
    assert cnt3 == 3
    assert ov3["configured_keys_count"] == 3
    assert {s["slot"]: s["masked"] for s in ov3["key_slots"]} == {
        1: "AIzaSy...8888",
        2: "AIzaSy...0002",
        3: "AIzaSy...0003",
    }
    assert "AIzaSySecondKey0002" in env3
    assert "AIzaSyThirdKey0003" in env3

    # ADV4: 2 distinct keys (GEMINI_API_KEY, GEMINI_API_KEY_1), PUT key_1 preserves Slot 2
    cnt4, ov4, env4 = _run_put_and_overview(
        "adv4_ws",
        "GEMINI_API_KEY=AIzaSyPrimaryKey0001\nGEMINI_API_KEY_1=AIzaSySecondKey0002\n",
        {"key_1": "AIzaSyUpdatedOne8888"},
    )
    assert cnt4 == 2
    assert {s["slot"]: s["masked"] for s in ov4["key_slots"]} == {
        1: "AIzaSy...8888",
        2: "AIzaSy...0002",
        3: "",
    }
    assert "AIzaSySecondKey0002" in env4

    # ADV5: 2 distinct keys (GEMINI_API_KEY, GEMINI_API_KEY_1), PUT key_1="" clears Slot 1 & preserves Slot 2
    cnt5, ov5, env5 = _run_put_and_overview(
        "adv5_ws",
        "GEMINI_API_KEY=AIzaSyPrimaryKey0001\nGEMINI_API_KEY_1=AIzaSySecondKey0002\n",
        {"key_1": ""},
    )
    assert cnt5 == 1
    assert ov5["configured_keys_count"] == 1
    assert {s["slot"]: s["masked"] for s in ov5["key_slots"]} == {
        1: "",
        2: "AIzaSy...0002",
        3: "",
    }
    assert "GEMINI_API_KEY_2=AIzaSySecondKey0002" in env5
    assert "AIzaSyPrimaryKey0001" not in env5

    # ADV10: GEMINI_API_KEY + GEMINI_API_KEY_1 + empty GEMINI_API_KEY_2=, PUT key_1="" preserves Slot 2
    cnt10, ov10, env10 = _run_put_and_overview(
        "adv10_ws",
        "GEMINI_API_KEY=AIzaSyPrimaryKey0001\n"
        "GEMINI_API_KEY_1=AIzaSySecondKey0002\n"
        "GEMINI_API_KEY_2=\n",
        {"key_1": ""},
    )
    assert cnt10 == 1
    assert {s["slot"]: s["masked"] for s in ov10["key_slots"]} == {
        1: "",
        2: "AIzaSy...0002",
        3: "",
    }
    assert "GEMINI_API_KEY_2=AIzaSySecondKey0002" in env10

    # ADV6: GEMINI_API_KEYS (2 keys) + distinct GEMINI_API_KEY_1, PUT key_1 preserves Slot 3
    cnt6, ov6, env6 = _run_put_and_overview(
        "adv6_ws",
        "GEMINI_API_KEYS=AIzaSyMultiOne1111,AIzaSyMultiTwo2222\n"
        "GEMINI_API_KEY_1=AIzaSyThirdKey0003\n",
        {"key_1": "AIzaSyUpdatedOne8888"},
    )
    assert cnt6 == 3
    assert {s["slot"]: s["masked"] for s in ov6["key_slots"]} == {
        1: "AIzaSy...8888",
        2: "AIzaSy...2222",
        3: "AIzaSy...0003",
    }
    assert "AIzaSyThirdKey0003" in env6

    # ADV11: GEMINI_API_KEYS (2 keys) + GEMINI_API_KEY_1 + GEMINI_API_KEY_2, PUT key_2 preserves Slot 3 & 4
    cnt11, ov11, env11 = _run_put_and_overview(
        "adv11_ws",
        "GEMINI_API_KEYS=AIzaSyMultiOne1111,AIzaSyMultiTwo2222\n"
        "GEMINI_API_KEY_1=AIzaSyThirdKey0003\n"
        "GEMINI_API_KEY_2=AIzaSyFourthKey0004\n",
        {"key_2": "AIzaSyUpdatedTwo9999"},
    )
    assert cnt11 == 4
    assert ov11["configured_keys_count"] == 4
    assert {s["slot"]: s["masked"] for s in ov11["key_slots"]} == {
        1: "AIzaSy...1111",
        2: "AIzaSy...9999",
        3: "AIzaSy...0003",
    }
    assert "GEMINI_API_KEY_3=AIzaSyThirdKey0003" in env11
    assert "GEMINI_API_KEY_4=AIzaSyFourthKey0004" in env11

    # ADV12: mirrored GEMINI_API_KEY == GEMINI_API_KEY_1 + GEMINI_API_KEY_2, PUT key_1="" deduplicates & clears Slot 1
    cnt12, ov12, env12 = _run_put_and_overview(
        "adv12_ws",
        "GEMINI_API_KEY=AIzaSyPrimaryKey0001\n"
        "GEMINI_API_KEY_1=AIzaSyPrimaryKey0001\n"
        "GEMINI_API_KEY_2=AIzaSySecondKey0002\n",
        {"key_1": ""},
    )
    assert cnt12 == 1
    assert ov12["configured_keys_count"] == 1
    assert {s["slot"]: s["masked"] for s in ov12["key_slots"]} == {
        1: "",
        2: "AIzaSy...0002",
        3: "",
    }
    assert "AIzaSyPrimaryKey0001" not in env12
    assert "GEMINI_API_KEY_2=AIzaSySecondKey0002" in env12
