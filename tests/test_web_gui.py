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


def test_windows_venv_child_pid_live_telemetry_not_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify UPSCALER_WEB_JOB_ID and ppid lineage stream live upscaling telemetry and survive abort."""
    import asyncio
    import time
    import psutil
    from web.sse_hub import TelemetryHub, live_matches_web_job

    root = _seed_workspace(tmp_path / "pid_lineage_ws")
    stale_live = root / "logs" / "live_status.json"
    stale_live.write_text(
        json.dumps({"video_name": "OldCrash.mkv", "pid": 11111, "stage": "error", "last_error": "stale"}),
        encoding="utf-8",
    )
    stale_ts = time.time() - 30.0
    os.utime(stale_live, (stale_ts, stale_ts))

    captured_spawn: Dict[str, Any] = {}
    procs: List[FakePopenProcess] = []
    alive_pids = {25000, 27000}

    def fake_popen(cmd: List[str], **kwargs: Any) -> FakePopenProcess:
        proc = FakePopenProcess(cmd, pid=25000, running=True)
        orig_terminate = proc.terminate

        def _term() -> None:
            alive_pids.discard(25000)
            alive_pids.discard(27000)
            orig_terminate()

        proc.terminate = _term  # type: ignore[method-assign]
        captured_spawn["cmd"] = cmd
        captured_spawn["env"] = dict(kwargs.get("env") or {})
        captured_spawn["proc"] = proc
        procs.append(proc)
        return proc

    monkeypatch.setattr(logger_mod, "LOGS_DIR", str(root / "logs"))
    app = create_app(workspace_root=root, gpu_query_fn=_fake_gpu_metrics, popen_factory=fake_popen)
    with TestClient(app) as client:
        start_resp = client.post(
            "/api/jobs/start",
            json={"inputs": ["anime_season/[SubsPlease] Frieren - 10 (1080p).mkv"]},
        )
        assert start_resp.status_code == 200
        job = start_resp.json()["job"]
        assert job["pid"] == 25000
        assert captured_spawn["env"]["UPSCALER_WEB_JOB_ID"] == job["job_id"]

        monkeypatch.setattr(os, "getpid", lambda: 27000)
        monkeypatch.setattr(os, "getppid", lambda: 25000)
        monkeypatch.setattr(psutil, "pid_exists", lambda p: p in alive_pids)
        monkeypatch.setenv("UPSCALER_WEB_JOB_ID", job["job_id"])

        rl = logger_mod.RunLogger("Frieren_10")
        live_disk = json.loads(stale_live.read_text(encoding="utf-8"))
        assert live_disk["pid"] == 27000
        assert live_disk["ppid"] == 25000
        assert live_disk["job_id"] == job["job_id"]

        rl.update_live_context(total_chunks=4, total_frames=400, unique_frames=240)
        rl.set_stage("upscaling", "Chunk 2/4", active_chunk=1)
        rl.log_chunk_done(
            chunk_idx=0,
            total_chunks=4,
            frames=100,
            ai_time=5.0,
            ai_fps=20.0,
            enc_time=0.2,
            total_chunk_time=5.2,
            output_seg=str(root / "out" / "seg_0000.mp4"),
            steady_fps=21.4,
        )

        st_resp = client.get("/api/status")
        assert st_resp.status_code == 200
        pipe = st_resp.json()["pipeline"]
        assert pipe["effective_stage"] == "upscaling"
        assert pipe["live"]["stage"] == "upscaling"
        assert pipe["live"]["pid"] == 27000
        assert pipe["live"]["ppid"] == 25000
        assert pipe["live"]["job_id"] == job["job_id"]
        assert pipe["live"]["completed_chunks"] == 1
        assert pipe["live"]["active_chunk"] == 1
        assert pipe["live"]["steady_fps"] > 0

        assert live_matches_web_job(
            {"pid": 27000, "ppid": 25000, "job_id": None},
            {"pid": 25000, "job_id": job["job_id"], "started_at": job["started_at"]},
            time.time(),
        )
        assert TelemetryHub._live_matches_web_job(
            {"pid": 27000, "ppid": 25000, "job_id": None},
            {"pid": 25000, "job_id": job["job_id"], "started_at": job["started_at"]},
            time.time(),
        )

        async def _read_one_tick() -> Dict[str, Any]:
            gen = app.state.telemetry_hub.stream_events()
            try:
                snap_raw = await anext(gen)
                assert snap_raw.startswith("event: snapshot\n")
                rl.log_chunk_done(
                    chunk_idx=1,
                    total_chunks=4,
                    frames=100,
                    ai_time=4.5,
                    ai_fps=22.2,
                    enc_time=0.2,
                    total_chunk_time=4.7,
                    output_seg=str(root / "out" / "seg_0001.mp4"),
                    steady_fps=22.1,
                )
                for _ in range(5):
                    ev_raw = await anext(gen)
                    if ev_raw.startswith("event: tick\n"):
                        for ln in ev_raw.splitlines():
                            if ln.startswith("data: "):
                                return json.loads(ln[len("data: "):])
                raise AssertionError("Did not receive SSE event: tick")
            finally:
                await gen.aclose()

        tick_payload = asyncio.run(_read_one_tick())
        tick_pipe = tick_payload["pipeline"]
        assert tick_pipe["effective_stage"] == "upscaling"
        assert tick_pipe["live"]["completed_chunks"] == 2
        assert tick_pipe["live"]["active_chunk"] == 1
        assert tick_pipe["live"]["steady_fps"] > 0

        abort_resp = client.post("/api/jobs/abort")
        assert abort_resp.status_code == 200
        snap: Dict[str, Any] = {}
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            snap = app.state.job_manager.get_state()
            active = snap.get("active_job")
            if (not active or active.get("finished_at") is not None) and snap.get("history"):
                break
            time.sleep(0.02)
        assert snap.get("status") == "aborted"
        assert snap.get("history"), f"Aborted job did not finalize before deadline: {snap}"

        post_abort = client.get("/api/status").json()["pipeline"]
        assert post_abort["effective_stage"] == "paused"
        assert post_abort["live"]["completed_chunks"] == 2
        assert post_abort["live"]["active_chunk"] == 1
        assert post_abort["live"]["steady_fps"] > 0


def test_preview_job_honors_output_dir_batch_size_and_force_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify preview -o/batch_size/force=False forwarding, 1..32 validation, and .preview_clip_* isolation."""
    import contextlib
    import src.dedup as dedup_mod
    import src.models_resolver as mr_mod
    import src.pipeline as pipeline_mod
    import web.job_manager as jm_mod

    root = _seed_workspace(tmp_path / "preview_api_ws")
    spawned: List[Dict[str, Any]] = []

    def fake_popen(cmd: List[str], **kwargs: Any) -> FakePopenProcess:
        proc = FakePopenProcess(cmd, pid=52000 + len(spawned), running=False)
        spawned.append({"cmd": cmd, "env": dict(kwargs.get("env") or {})})
        return proc

    app = create_app(workspace_root=root, gpu_query_fn=_fake_gpu_metrics, popen_factory=fake_popen)
    custom_out = tmp_path / "custom_preview_out"
    custom_out.mkdir(parents=True, exist_ok=True)
    ep_rel = "anime_season/[BD 1080p] Fate_Ep22.mkv"

    with TestClient(app) as client:
        r_default = client.post(
            "/api/jobs/preview",
            json={"video_path": ep_rel, "seconds": 12, "output_dir": str(custom_out), "batch_size": 16},
        )
        assert r_default.status_code == 200
        cmd_def = spawned[-1]["cmd"]
        assert "-o" in cmd_def and cmd_def[cmd_def.index("-o") + 1] == str(custom_out)
        assert "--batch-size" in cmd_def and cmd_def[cmd_def.index("--batch-size") + 1] == "16"
        assert "--force" not in cmd_def
        assert spawned[-1]["env"]["UPSCALER_NO_AUTO_OPEN"] == "1"

        r_force = client.post(
            "/api/jobs/preview",
            json={"video_path": ep_rel, "seconds": 10, "force": True, "batch_size": 4},
        )
        assert r_force.status_code == 200
        assert "--force" in spawned[-1]["cmd"]

        for bad_bs in (0, 33, 99, True, False, 8.0, 8.5, "abc"):
            assert client.post("/api/jobs/preview", json={"video_path": ep_rel, "batch_size": bad_bs}).status_code == 422
            assert client.post("/api/jobs/start", json={"inputs": [ep_rel], "batch_size": bad_bs}).status_code == 422
            assert client.post(
                "/api/queue/preview-schedule",
                json={"inputs": [], "batch_size": bad_bs},
            ).status_code == 422

        for good_bs in (1, 32, "8", "32"):
            assert client.post(
                "/api/queue/preview-schedule",
                json={"inputs": [], "batch_size": good_bs},
            ).status_code == 200

        assert client.post(
            "/api/jobs/preview",
            json={"video_path": ep_rel, "output_dir": ".env"},
        ).status_code == 403
        assert client.post(
            "/api/jobs/preview",
            json={"video_path": ep_rel, "output_dir": str(root / ".env")},
        ).status_code == 403

    for bad_jm in (True, False, 8.0, "8.0", "8.5", 0, 33):
        with pytest.raises(ValueError):
            jm_mod._validate_batch_size(bad_jm)
    for ok_jm in (None, "", 1, "16", 32):
        jm_mod._validate_batch_size(ok_jm)

    iso_root = tmp_path / "pipeline_preview_iso"
    src_dir = iso_root / "Raw"
    src_dir.mkdir(parents=True, exist_ok=True)
    src_vid = src_dir / "[BD 1080p] Fate_Ep22.mkv"
    src_vid.write_bytes(b"\x00" * 4096)
    (iso_root / "shows.yaml").write_text(
        'shows:\n  - pattern: "*[BD*"\n    model: animevideov3\n    dedup_policy: dis-auto\n',
        encoding="utf-8",
    )
    (iso_root / "models").mkdir(parents=True, exist_ok=True)
    (iso_root / "models" / "models.yaml").write_text(
        (root / "models" / "models.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    target_out = iso_root / "custom_out"
    target_out.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(dedup_mod, "ROOT_DIR", str(iso_root))
    monkeypatch.setattr(mr_mod, "ROOT_DIR", str(iso_root))
    mr_mod._YAML_STAT_CACHE.clear()

    class _StopAfterDedup(RuntimeError):
        pass

    recorded_cut_cmds: List[List[str]] = []
    recorded_dedup_calls: List[tuple[str, Any]] = []
    orig_resolve_dedup = pipeline_mod.resolve_dedup_policy

    def spy_resolve_dedup(vid_path: str, cli_pol: Any = None) -> str:
        res = orig_resolve_dedup(vid_path, cli_pol)
        recorded_dedup_calls.append((vid_path, res))
        return res

    def fake_subprocess_run(cmd: List[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        recorded_cut_cmds.append(list(cmd))
        out_clip = Path(cmd[-1])
        out_clip.parent.mkdir(parents=True, exist_ok=True)
        out_clip.write_bytes(b"\x00" * 2048)
        return subprocess.CompletedProcess(cmd, 0)

    @contextlib.contextmanager
    def fake_ramdisk(**_kwargs: Any):
        ram_dir = iso_root / "ram"
        ram_dir.mkdir(parents=True, exist_ok=True)
        yield str(ram_dir)

    monkeypatch.setattr(pipeline_mod, "validate_environment", lambda **_kw: None)
    monkeypatch.setattr(pipeline_mod, "acquire_gpu_lock", lambda: None)
    monkeypatch.setattr(pipeline_mod, "release_gpu_lock", lambda: None)
    monkeypatch.setattr(pipeline_mod, "run_preflight_checks", lambda **_kw: None)
    monkeypatch.setattr(pipeline_mod, "RAMDisk", fake_ramdisk)
    monkeypatch.setattr(pipeline_mod, "resolve_dedup_policy", spy_resolve_dedup)
    monkeypatch.setattr(pipeline_mod.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        pipeline_mod,
        "probe_video",
        lambda _p: {
            "duration": 120.0,
            "nb_frames": 2880,
            "fps": 23.976,
            "fps_num": 24000,
            "fps_den": 1001,
            "width": 1920,
            "height": 1080,
            "size_bytes": 4096,
            "audio_count": 1,
            "subtitle_count": 1,
            "attachment_count": 0,
            "chapter_count": 0,
        },
    )
    monkeypatch.setattr(
        pipeline_mod,
        "split_video_segments",
        lambda _in, segs_d, _st: [str(Path(segs_d) / "seg_0000.mp4")],
    )
    monkeypatch.setattr(pipeline_mod, "get_segments_frame_counts", lambda _segs: [240])
    monkeypatch.setattr(pipeline_mod, "is_valid_completed_4k_output", lambda _p: False)

    def stop_at_plan(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise _StopAfterDedup("stop")

    monkeypatch.setattr(pipeline_mod, "load_or_create_plan", stop_at_plan)
    monkeypatch.setattr(pipeline_mod, "RollingDedupPlanner", stop_at_plan)

    with pytest.raises(_StopAfterDedup):
        pipeline_mod.run_upscale_pipeline(
            input_path=str(src_vid),
            output_path=str(target_out),
            preview_seconds=10,
            force=False,
            dedup=True,
            anti_sleep=False,
        )

    expected_temp = target_out / ".preview_clip_[BD 1080p] Fate_Ep22.mkv"
    assert recorded_cut_cmds and Path(recorded_cut_cmds[-1][-1]).resolve() == expected_temp.resolve()
    assert list(src_dir.glob(".preview_clip_*")) == []
    assert list(target_out.glob(".preview_clip_*")) == []
    assert recorded_dedup_calls == [(str(src_vid.resolve()), "dis-auto")]
    live_data = json.loads(Path(logger_mod.get_live_status_path()).read_text(encoding="utf-8"))
    assert live_data["dedup_policy"] == "dis-auto"

    expected_prev_4k = target_out / "[BD 1080p] Fate_Ep22_preview_10s_4K.mkv"
    expected_prev_4k.write_bytes(b"\x00" * 16)
    monkeypatch.setattr(
        pipeline_mod,
        "is_valid_completed_4k_output",
        lambda p: Path(p).resolve() == expected_prev_4k.resolve(),
    )
    ret_path = pipeline_mod.run_upscale_pipeline(
        input_path=str(src_vid),
        output_path=str(target_out),
        model="animevideov3",
        preview_seconds=10,
        force=False,
        dedup=True,
        anti_sleep=False,
    )
    assert Path(ret_path).resolve() == expected_prev_4k.resolve()
    assert list(target_out.glob(".preview_clip_*")) == []


def test_checkpoint_resume_preserves_full_source_path_and_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify manifest.json, plan.json, and scan_temp_checkpoints preserve full source_file and output_dir."""
    import contextlib
    import src.dedup as dedup_mod
    import src.pipeline as pipeline_mod
    from web.sse_hub import scan_temp_checkpoints

    src_ep = tmp_path / "Raw" / "Season 1" / "Ep11.mkv"
    src_ep.parent.mkdir(parents=True, exist_ok=True)
    src_ep.write_bytes(b"\x00" * 1024)

    cp_plan_dir = tmp_path / "planner_cp"
    cp_plan_dir.mkdir(parents=True, exist_ok=True)
    plan_file = cp_plan_dir / "plan.json"
    monkeypatch.setattr(
        dedup_mod,
        "scan_single_segment_actions",
        lambda **_kw: {
            "chunk_meta": {
                "chunk_idx": 0,
                "start_frame": 0,
                "end_frame": 4,
                "total_frames": 4,
                "kept_count": 2,
                "kept_indices": [0, 2],
                "runs": [{"frame_idx": 0, "count": 2}, {"frame_idx": 2, "count": 2}],
                "scene_cut": True,
            },
            "raw_actions": ["keep", "drop", "keep", "drop"],
            "first_gray": None,
            "last_gray": None,
        },
    )
    planner = dedup_mod.RollingDedupPlanner(
        input_path=str(src_ep),
        seg_paths=[str(cp_plan_dir / "seg_0000.mp4")],
        seg_frame_counts=[4],
        plan_path=str(plan_file),
        policy="default",
    )
    planner.finalize_plan()
    saved_plan = json.loads(plan_file.read_text(encoding="utf-8"))
    assert saved_plan["source_file"] == "Ep11.mkv"
    assert saved_plan["source_path"] == os.path.abspath(str(src_ep))

    ws = _seed_workspace(tmp_path / "cp_manifest_ws")
    custom_out = tmp_path / "custom_out_dir"
    custom_out.mkdir(parents=True, exist_ok=True)

    class _StopAfterManifest(RuntimeError):
        pass

    @contextlib.contextmanager
    def stop_ramdisk(**_kwargs: Any):
        raise _StopAfterManifest("stop")
        yield ""

    monkeypatch.setattr(pipeline_mod, "validate_environment", lambda **_kw: None)
    monkeypatch.setattr(pipeline_mod, "acquire_gpu_lock", lambda: None)
    monkeypatch.setattr(pipeline_mod, "release_gpu_lock", lambda: None)
    monkeypatch.setattr(pipeline_mod, "run_preflight_checks", lambda **_kw: None)
    monkeypatch.setattr(pipeline_mod, "RAMDisk", stop_ramdisk)
    monkeypatch.setattr(pipeline_mod, "is_valid_completed_4k_output", lambda _p: False)
    monkeypatch.setattr(
        pipeline_mod,
        "resolve_model_for_video",
        lambda _p, _m=None: {"id": "animevideov3", "name": "AnimeVideo-v3", "sha256": "abc123", "arch": {"upscale": 4}},
    )
    monkeypatch.setattr(
        pipeline_mod,
        "probe_video",
        lambda _p: {
            "duration": 24.0,
            "nb_frames": 576,
            "fps": 24.0,
            "fps_num": 24,
            "fps_den": 1,
            "width": 1920,
            "height": 1080,
            "size_bytes": 1024,
            "audio_count": 1,
            "subtitle_count": 0,
            "attachment_count": 0,
            "chapter_count": 0,
        },
    )

    with pytest.raises(_StopAfterManifest):
        pipeline_mod.run_upscale_pipeline(
            input_path=str(src_ep),
            output_path=str(custom_out),
            anti_sleep=False,
        )

    manifest_path = custom_out / ".temp_Ep11" / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["source_file"] == os.path.abspath(str(src_ep))
    assert manifest_data["output_dir"] == os.path.abspath(str(custom_out))

    default_cp = ws / "out" / ".temp_DefaultEp"
    (default_cp / "raw_segs").mkdir(parents=True, exist_ok=True)
    (default_cp / "upscaled_segs").mkdir(parents=True, exist_ok=True)
    default_src = ws / "anime_season" / "DefaultEp.mkv"
    (default_cp / "manifest.json").write_text(
        json.dumps({"model_id": "animevideov3", "source_file": str(default_src), "output_dir": str(ws / "out")}),
        encoding="utf-8",
    )

    fallback_cp = ws / "custom_scan" / ".temp_Ep11"
    (fallback_cp / "raw_segs").mkdir(parents=True, exist_ok=True)
    (fallback_cp / "upscaled_segs").mkdir(parents=True, exist_ok=True)
    (fallback_cp / "manifest.json").write_text(
        json.dumps({"model_id": "animevideov3", "source_file": str(ws / "custom_scan" / ".preview_clip_Ep11.mkv")}),
        encoding="utf-8",
    )
    (fallback_cp / "plan.json").write_text(
        json.dumps({"source_file": "Ep11.mkv", "source_path": os.path.abspath(str(src_ep)), "chunks": [{"chunk_idx": 0}]}),
        encoding="utf-8",
    )

    preview_temp_cp = ws / "out" / ".temp_.preview_clip_Ep11"
    preview_temp_cp.mkdir(parents=True, exist_ok=True)
    (preview_temp_cp / "manifest.json").write_text(
        json.dumps({"model_id": "animevideov3", "source_file": str(ws / "out" / ".preview_clip_Ep11.mkv")}),
        encoding="utf-8",
    )

    scanned = scan_temp_checkpoints(ws, extra_dirs=[ws / "custom_scan"])
    by_ep = {item["episode_name"]: item for item in scanned}
    assert by_ep[".preview_clip_Ep11"]["source_file"] == ""
    assert ".preview_clip_" not in by_ep["Ep11"]["source_file"]
    assert by_ep["DefaultEp"]["source_file"] == str(default_src)
    assert by_ep["DefaultEp"]["output_dir"] == str(ws / "out")
    assert by_ep["DefaultEp"]["is_custom_output_dir"] is False
    assert by_ep["Ep11"]["source_file"] == os.path.abspath(str(src_ep))
    assert by_ep["Ep11"]["output_dir"] == str((ws / "custom_scan").resolve())
    assert by_ep["Ep11"]["is_custom_output_dir"] is True


def test_abort_job_cleans_orphan_descendant_gpu_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify abort_job removes orphan descendant .gpu_active.lock while preserving external CLI locks."""
    from types import SimpleNamespace
    import psutil
    from web.job_manager import JobManager

    root1 = _seed_workspace(tmp_path / "abort_child_ws")
    jm1 = JobManager(root1, popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=25000, running=True))
    start1 = jm1.start_upscale_job({"inputs": ["anime_season/[SubsPlease] Frieren - 10 (1080p).mkv"]})
    assert start1["pid"] == 25000
    (root1 / ".gpu_active.lock").write_text("27000\n", encoding="utf-8")

    child_checks = {"n": 0}

    class _FakeParentProc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def children(self, recursive: bool = True) -> List[Any]:
            return [SimpleNamespace(pid=27000)] if self.pid == 25000 and recursive else []

    def fake_pid_exists_delay(pid: int) -> bool:
        if pid == 27000:
            child_checks["n"] += 1
            return child_checks["n"] <= 2
        return False

    monkeypatch.setattr(psutil, "Process", _FakeParentProc)
    monkeypatch.setattr(psutil, "pid_exists", fake_pid_exists_delay)
    res1 = jm1.abort_job()
    assert res1["ok"] is True
    assert res1["status"] == "aborted"
    assert res1["aborted_pid"] == 25000
    assert not (root1 / ".gpu_active.lock").exists()

    root2 = _seed_workspace(tmp_path / "abort_live_fallback_ws")
    jm2 = JobManager(root2, popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=25001, running=True))
    start2 = jm2.start_upscale_job({"inputs": ["anime_season/[SubsPlease] Frieren - 10 (1080p).mkv"]})
    job2 = start2
    (root2 / "logs" / "live_status.json").write_text(
        json.dumps({"job_id": job2["job_id"], "pid": 27001, "stage": "upscaling"}),
        encoding="utf-8",
    )
    (root2 / ".gpu_active.lock").write_text("27001\n", encoding="utf-8")

    def raising_process(_pid: int) -> Any:
        raise RuntimeError("Process lookup unavailable")

    monkeypatch.setattr(psutil, "Process", raising_process)
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: pid == 27001)
    res2 = jm2.abort_job()
    assert res2["ok"] is True
    assert not (root2 / ".gpu_active.lock").exists()

    root3 = _seed_workspace(tmp_path / "abort_external_preserve_ws")
    jm3 = JobManager(root3, popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=25002, running=True))
    jm3.start_upscale_job({"inputs": ["anime_season/[SubsPlease] Frieren - 10 (1080p).mkv"]})
    (root3 / ".gpu_active.lock").write_text("99001\n", encoding="utf-8")

    class _FakeParentProc3:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def children(self, recursive: bool = True) -> List[Any]:
            return [SimpleNamespace(pid=27002)] if self.pid == 25002 and recursive else []

    monkeypatch.setattr(psutil, "Process", _FakeParentProc3)
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: pid == 99001)
    res3 = jm3.abort_job()
    assert res3["ok"] is True
    assert (root3 / ".gpu_active.lock").read_text(encoding="utf-8").strip() == "99001"


def test_fs_browse_hides_windows_system_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify SafeLocalFSBrowser hides Windows system/hidden/secret folders and enforces 403 on secrets."""
    import stat as stat_mod
    import web.fs_browser as fs_browser_mod

    drive_root = tmp_path / "fake_drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    for d_name in ("$RECYCLE.BIN", "System Volume Information", "Recovery", "Normal_Show"):
        (drive_root / d_name).mkdir(exist_ok=True)
    (drive_root / "Normal_Ep01.mkv").write_bytes(b"\x00" * 64)
    (drive_root / ".env").write_text("SECRET=1\n", encoding="utf-8")

    browser = SafeLocalFSBrowser(tmp_path)
    listed = browser.browse(str(drive_root))
    names = {e["name"] for e in listed["entries"]}
    assert names == {"Normal_Show", "Normal_Ep01.mkv"}

    class _FakeDirEntry:
        def __init__(self, name: str, is_dir_flag: bool, attrs: int = 0, stat_err: bool = False) -> None:
            self.name = name
            self.path = str(drive_root / name)
            self._is_dir = is_dir_flag
            self._attrs = attrs
            self._stat_err = stat_err

        def is_dir(self, follow_symlinks: bool = False) -> bool:
            return self._is_dir

        def is_file(self, follow_symlinks: bool = False) -> bool:
            return not self._is_dir

        def stat(self, follow_symlinks: bool = False) -> Any:
            if self._stat_err:
                raise OSError("Access denied")

            class _FakeStat:
                st_size = 128
                st_mtime = 1700000000
                st_file_attributes = self._attrs

            return _FakeStat()

    class _FakeScandirCtx:
        def __init__(self, entries: List[Any]) -> None:
            self._entries = entries

        def __iter__(self) -> Any:
            return iter(self._entries)

        def __enter__(self) -> List[Any]:
            return self._entries

        def __exit__(self, *_args: Any) -> bool:
            return False

    monkeypatch.setattr(fs_browser_mod.sys, "platform", "win32")
    monkeypatch.setattr(stat_mod, "FILE_ATTRIBUTE_SYSTEM", 0x4, raising=False)
    monkeypatch.setattr(stat_mod, "FILE_ATTRIBUTE_HIDDEN", 0x2, raising=False)
    fake_entries = [
        _FakeDirEntry("HiddenFolder", True, attrs=0x2),
        _FakeDirEntry("SystemFolder", True, attrs=0x4),
        _FakeDirEntry("BrokenStatFolder", True, stat_err=True),
        _FakeDirEntry("VisibleFolder", True, attrs=0),
        _FakeDirEntry("HiddenVideo.mkv", False, attrs=0x2),
        _FakeDirEntry("VisibleVideo.mkv", False, attrs=0),
    ]
    monkeypatch.setattr(fs_browser_mod.os, "scandir", lambda _p: _FakeScandirCtx(fake_entries))
    win_listed = browser.browse(str(drive_root))
    assert [e["name"] for e in win_listed["entries"]] == ["VisibleFolder", "VisibleVideo.mkv"]

    monkeypatch.undo()
    app = create_app(workspace_root=drive_root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        assert client.get("/api/fs/browse", params={"path": ".env"}).status_code == 403
        assert client.get("/api/fs/browse", params={"path": str(drive_root / ".env")}).status_code == 403

        def _deny_scandir(_p: str) -> Any:
            raise PermissionError("Permission denied")

        monkeypatch.setattr(fs_browser_mod.os, "scandir", _deny_scandir)
        assert client.get("/api/fs/browse", params={"path": str(drive_root)}).status_code == 403


def test_dry_run_completion_does_not_resurrect_stale_live_status_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify dry-run completion suppresses stale error state, prefers newer log, and skips preview schedule."""
    from datetime import datetime
    import time
    import src.scheduler as sched_mod
    import upscale as upscale_mod
    from web.sse_hub import TelemetryHub

    ws = _seed_workspace(tmp_path / "dry_run_ws")
    ep = ws / "anime_season" / "[SubsPlease] Frieren - 10 (1080p).mkv"
    logs_dir = ws / "logs"
    stale_live = logs_dir / "live_status.json"
    old_log = logs_dir / "old_run.log"
    web_log = logs_dir / "web_job_dry01.log"
    old_log.write_text("[00:00:01] old error\n", encoding="utf-8")
    web_log.write_text("[00:00:02] dry-run decision table\n", encoding="utf-8")

    now_ts = time.time()
    os.utime(old_log, (now_ts - 40.0, now_ts - 40.0))
    os.utime(web_log, (now_ts, now_ts))

    stale_live.write_text(
        json.dumps({
            "video_name": "OldCrash.mkv",
            "pid": 999991,
            "stage": "error",
            "last_error": "CUDA OOM",
            "log_file": "old_run.log",
            "total_chunks": 10,
            "completed_chunks": 3,
            "chunks": [{"chunk_idx": 0}],
        }),
        encoding="utf-8",
    )
    os.utime(stale_live, (now_ts - 30.0, now_ts - 30.0))

    now_iso = datetime.fromtimestamp(now_ts).strftime("%Y-%m-%dT%H:%M:%S")
    job_state = {
        "status": "idle",
        "active_job": None,
        "history": [
            {
                "job_id": "dry01",
                "job_type": "upscale",
                "status": "completed",
                "pid": 81000,
                "started_at": now_iso,
                "finished_at": now_iso,
                "return_code": 0,
                "log_path": str(web_log),
                "queue_items": [str(ep)],
                "options": {"dry_run": True, "model": "animevideov3"},
            }
        ],
    }
    hub = TelemetryHub(ws, gpu_query_fn=_idle_gpu_metrics, job_state_provider=lambda: job_state)
    pipe = hub.read_pipeline_state()
    assert pipe["effective_stage"] == "completed"
    assert pipe["live"]["stage"] == "completed"
    assert pipe["live"]["dry_run"] is True
    assert "last_error" not in pipe["live"]
    assert pipe["live"]["chunks"] == []
    assert hub._select_active_log_file().resolve() == web_log.resolve()

    sched_calls: List[str] = []
    monkeypatch.setattr(
        sched_mod,
        "compute_overnight_schedule",
        lambda **kw: (sched_calls.append("compute") or {"summary": {}, "episodes": [], "decision": "RUN"}),
    )
    monkeypatch.setattr(
        sched_mod,
        "format_schedule_decision_table",
        lambda _s: (sched_calls.append("format") or "table"),
    )
    monkeypatch.setattr(upscale_mod, "is_graceful_pause_requested", lambda: False)
    monkeypatch.setattr(upscale_mod, "check_deadline_and_eta", lambda *_a, **_kw: (True, False, None))
    preview_out_file = ws / "out" / "ep_preview_10s_4K.mkv"
    preview_out_file.parent.mkdir(parents=True, exist_ok=True)
    preview_out_file.write_bytes(b"\x00" * 64)
    monkeypatch.setattr(
        upscale_mod,
        "run_upscale_pipeline",
        lambda **_kw: str(preview_out_file),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["upscale.py", str(ep), "--preview", "10", "--no-subs", "--no-organize-audit", "-o", str(ws / "out")],
    )
    with pytest.raises(SystemExit) as exc1:
        upscale_mod.main()
    assert exc1.value.code == 0
    assert sched_calls == []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upscale.py",
            str(ep),
            "--preview",
            "10",
            "--deadline",
            "23:59",
            "--no-subs",
            "--no-organize-audit",
            "-o",
            str(ws / "out"),
        ],
    )
    with pytest.raises(SystemExit) as exc2:
        upscale_mod.main()
    assert exc2.value.code == 0
    assert "compute" in sched_calls


def test_save_shows_yaml_and_dotenv_preserve_lf_newlines(studio_env: Dict[str, Any]):
    """Verify PUT /api/config/shows and PUT /api/subtrans/api-keys write LF-only bytes without CRLF."""
    root: Path = studio_env["root"]
    client: TestClient = studio_env["client"]

    shows_resp = client.put(
        "/api/config/shows",
        json={"yaml_content": "shows:\r\n  - pattern: '*TestShow*'\r\n    model: animevideov3\r\n"},
    )
    assert shows_resp.status_code == 200
    raw_shows = (root / "shows.yaml").read_bytes()
    assert b"\r\n" not in raw_shows and b"\r" not in raw_shows
    assert raw_shows.endswith(b"\n")

    (root / ".env").write_bytes(b"CUSTOM_FLAG=keep_me\r\nGEMINI_API_KEY_1=AIzaSyOldKey111111111\r\n")
    env_resp = client.put(
        "/api/subtrans/api-keys",
        json={"key_2": "AIzaSyLfCheckKey222222"},
    )
    assert env_resp.status_code == 200
    raw_env = (root / ".env").read_bytes()
    assert b"\r\n" not in raw_env and b"\r" not in raw_env
    assert raw_env.endswith(b"\n")


def test_compare_frames_uses_preview_start_sec_from_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify /api/preview/compare-frames resolves preview_start_sec by precedence, clamps, and honors override."""
    import web.app as web_app_mod

    ws = _seed_workspace(tmp_path / "compare_preview_start_ws")
    src_vid = ws / "anime_season" / "Ep01.mkv"
    src_vid.write_bytes(b"\x00" * 128)
    out_dir = ws / "out"
    audit_sub = out_dir / "audit"
    audit_sub.mkdir(parents=True, exist_ok=True)
    up_1 = out_dir / "Ep01_preview_10s_4K.mkv"
    up_1.write_bytes(b"\x00" * 128)

    sibling_audit = out_dir / "Ep01_preview_10s_4K.audit.json"
    sub_audit = audit_sub / "Ep01_preview_10s_4K.audit.json"
    live_file = ws / "logs" / "live_status.json"

    sibling_audit.write_text(json.dumps({"preview_start_sec": 120.5}), encoding="utf-8")
    sub_audit.write_text(json.dumps({"preview_start_sec": 90.0}), encoding="utf-8")
    live_file.write_text(
        json.dumps({
            "is_preview": True,
            "preview_start_sec": 45.0,
            "output_path": str(up_1),
            "video_name": "Ep01_preview_10s",
        }),
        encoding="utf-8",
    )

    durations = {"src": 1400.0, "up": 10.0}

    def fake_probe(p: Path) -> Dict[str, Any]:
        if p.resolve() == src_vid.resolve():
            return {"duration": durations["src"], "width": 1920, "height": 1080}
        return {"duration": durations["up"], "width": 3840, "height": 2160}

    monkeypatch.setattr(web_app_mod, "_probe_duration_and_dims", fake_probe)

    def fake_extract(_video_path: Path, _ts: float, out_png: Path) -> None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(PNG_1X1_BYTES)

    app = create_app(workspace_root=ws, gpu_query_fn=_idle_gpu_metrics, frame_extract_fn=fake_extract)
    payload_base = {
        "source_path": str(src_vid),
        "upscaled_path": str(up_1),
        "timestamp_sec": 2.0,
    }
    with TestClient(app) as client:
        r1 = client.post("/api/preview/compare-frames", json=payload_base).json()
        assert r1["source_offset_sec"] == 120.5
        assert r1["source_timestamp_sec"] == 122.5

        sibling_audit.unlink()
        r2 = client.post("/api/preview/compare-frames", json=payload_base).json()
        assert r2["source_offset_sec"] == 90.0
        assert r2["source_timestamp_sec"] == 92.0

        sub_audit.unlink()
        r3 = client.post("/api/preview/compare-frames", json=payload_base).json()
        assert r3["source_offset_sec"] == 45.0

        live_file.write_text(
            json.dumps({"is_preview": False, "preview_start_sec": 45.0, "output_path": str(up_1)}),
            encoding="utf-8",
        )
        r_fallback = client.post("/api/preview/compare-frames", json=payload_base).json()
        assert r_fallback["source_offset_sec"] == 420.0

        sibling_audit.write_text(json.dumps({"preview_start_sec": 500.0}), encoding="utf-8")
        durations["src"] = 100.0
        r_clamped = client.post("/api/preview/compare-frames", json=payload_base).json()
        assert r_clamped["source_offset_sec"] == 90.0

        durations["src"] = 0.0
        r_zero_dur = client.post("/api/preview/compare-frames", json=payload_base).json()
        assert r_zero_dur["source_offset_sec"] == 500.0

        durations["src"] = 1400.0
        r_explicit = client.post(
            "/api/preview/compare-frames",
            json={**payload_base, "source_offset_sec": 33.25},
        ).json()
        assert r_explicit["source_offset_sec"] == 33.25
        assert r_explicit["source_timestamp_sec"] == 35.25


def test_list_audits_deduplicates_paired_4k_entries(tmp_path: Path):
    """Verify /api/audits deduplicates paired raw/_4K audits preferring _4K before [:60] slicing."""
    import time

    ws = _seed_workspace(tmp_path / "dedup_audits_ws")
    out_dir = ws / "out"
    base_ts = time.time() - 500.0

    for idx in range(1, 66):
        stem = f"Ep{idx:03d}"
        raw_p = out_dir / f"{stem}.audit.json"
        up_p = out_dir / f"{stem}_4K.audit.json"
        common = {
            "status": "PASS",
            "output_file": f"{stem}_4K.mkv",
            "completed_at": "2026-11-22T01:00:00",
        }
        up_p.write_text(json.dumps(common), encoding="utf-8")
        raw_p.write_text(json.dumps(common), encoding="utf-8")
        os.utime(up_p, (base_ts + idx, base_ts + idx))
        # Give raw_p a newer mtime than up_p to confirm _4K stem still wins.
        os.utime(raw_p, (base_ts + idx + 1000.0, base_ts + idx + 1000.0))

    corrupt_p = out_dir / "ZCorrupt_4K.audit.json"
    corrupt_p.write_text("{bad json", encoding="utf-8")
    os.utime(corrupt_p, (base_ts + 2000.0, base_ts + 2000.0))

    app = create_app(workspace_root=ws, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        resp = client.get("/api/audits")
        assert resp.status_code == 200
        items = resp.json()["audits"]

    assert len(items) == 60
    assert any(it.get("corrupt") for it in items)
    ep_rows = [it for it in items if not it.get("corrupt")]
    assert len(ep_rows) == 59
    for row in ep_rows:
        assert row["audit_json_path"].endswith("_4K.audit.json")
    output_files = [row["output_file"] for row in ep_rows]
    assert len(output_files) == len(set(output_files))


def test_night_report_endpoint_and_workspace_isolation(tmp_path: Path):
    """Verify GET /api/night-report supports workspace-isolated real mode, mock mode, and 422 validation."""
    ws = _seed_workspace(tmp_path / "night_report_iso_ws")
    logs_dir = ws / "logs"
    (logs_dir / "soak_telemetry_night1.csv").write_text(
        "timestamp,elapsed_s,gpu_util_pct,vram_used_mib,power_w,temp_c,cpu_pct,ram_used_gb,"
        "ssd_read_mb_s,ssd_write_mb_s,pstate\n"
        "2026-11-22T01:00:00,0.0,88.0,6144,160.0,65.0,25.0,12.0,10.0,20.0,P0\n"
        "2026-11-22T01:00:10,10.0,90.0,6244,165.0,66.0,26.0,12.5,12.0,22.0,P0\n",
        encoding="utf-8",
    )
    (logs_dir / "ep_IsolatedShow_01_20261122_010000.log").write_text(
        "# AI Video Upscaler 4K - Execution Log\n"
        "# Started: 2026-11-22 01:00:00\n"
        "# Video: IsolatedShow_01.mkv\n"
        "[2026-11-22 01:00:01] [CHUNK 1/1] Hoàn tất 24 frames (AI CUDA: 1.0s @ 24.0 fps | "
        "Encode: 0.2s | Tổng: 1.2s | File: up_0000.mp4 1.0 MB)\n"
        "[2026-11-22 01:00:10] Mux Audit: PASS\n",
        encoding="utf-8",
    )
    db_path = ws / "subtrans" / ".cache" / "translations.db"
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE translations (id INTEGER PRIMARY KEY, model TEXT)")
        con.execute(
            "CREATE TABLE api_requests (id INTEGER PRIMARY KEY, created_at TEXT, model TEXT, status TEXT, error_type TEXT)"
        )
        con.execute("INSERT INTO translations (model) VALUES ('gemini-3.5-flash-lite')")
        con.execute(
            "INSERT INTO api_requests (created_at, model, status, error_type) "
            "VALUES ('2026-11-21 18:00:05', 'gemini-3.5-flash-lite', 'success', NULL)"
        )
        con.commit()
    finally:
        con.close()

    real_app = create_app(workspace_root=ws, gpu_query_fn=_idle_gpu_metrics, mock_mode=False)
    with TestClient(real_app) as client:
        r_ok = client.get("/api/night-report", params={"session_name": "night1"})
        assert r_ok.status_code == 200
        body = r_ok.json()
        assert body["ok"] is True
        assert body["mock_mode"] is False
        assert body["session_name"] == "night1"
        assert "NIGHT1" in body["markdown"]
        assert "IsolatedShow_01.mkv" in body["markdown"]
        assert "gemini-3.5-flash-lite" in body["markdown"]

        for bad_name in ("../evil", "bad space", "", "a" * 65, "night;1"):
            assert client.get("/api/night-report", params={"session_name": bad_name}).status_code == 422

    mock_app = create_app(workspace_root=tmp_path / "mock_nr_ws", mock_mode=True)
    with TestClient(mock_app) as mock_client:
        r_mock = mock_client.get("/api/night-report", params={"session_name": "soak_2"})
        assert r_mock.status_code == 200
        mbody = r_mock.json()
        assert mbody["ok"] is True
        assert mbody["mock_mode"] is True
        assert mbody["session_name"] == "soak_2"
        assert "SOAK_2" in mbody["markdown"]


def test_conftest_isolates_logs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify conftest _isolate_logs_dir isolates RunLogger into tmp_path even without PYTEST_CURRENT_TEST."""
    repo_logs = Path(__file__).resolve().parents[1] / "logs"
    before_logs = {p.name for p in repo_logs.iterdir()} if repo_logs.is_dir() else set()

    env_logs = os.environ.get("UPSCALER_LOGS_DIR", "")
    assert env_logs
    assert Path(env_logs).resolve().is_relative_to(tmp_path.resolve())

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    resolved_logs = Path(logger_mod.get_logs_dir()).resolve()
    resolved_live_dir = Path(logger_mod.get_live_status_path()).resolve().parent
    assert resolved_logs.is_relative_to(tmp_path.resolve())
    assert resolved_live_dir.is_relative_to(tmp_path.resolve())

    rl = logger_mod.RunLogger("conftest_isolation_probe")
    assert Path(rl.log_filepath).resolve().is_relative_to(tmp_path.resolve())
    assert Path(rl.live_status_path).resolve().is_relative_to(tmp_path.resolve())

    custom_dyn = tmp_path / "dynamic_env_logs"
    monkeypatch.setattr(logger_mod, "LOGS_DIR", logger_mod._CONFIG_LOGS_DIR)
    monkeypatch.setenv("UPSCALER_LOGS_DIR", str(custom_dyn))
    assert Path(logger_mod.get_logs_dir()).resolve() == custom_dyn.resolve()

    after_logs = {p.name for p in repo_logs.iterdir()} if repo_logs.is_dir() else set()
    assert after_logs == before_logs


def test_frontend_ui_upgrade_contracts_issue_02_to_20(studio_env: Dict[str, Any]):
    """Verify HTML, CSS, and ES module contracts across ISSUE-02 through ISSUE-20 remain intact."""
    import re

    client: TestClient = studio_env["client"]
    html = client.get("/").text
    css = client.get("/static/styles.css").text
    app_js = client.get("/static/js/app.js").text
    panels_js = client.get("/static/js/panels.js").text
    fs_picker_js = client.get("/static/js/fs_picker.js").text
    compare_js = client.get("/static/js/compare_stage.js").text

    assert 'id="opt-force"' in html
    assert re.search(r'<input[^>]*id="opt-force"[^>]*\bchecked\b', html) is None
    assert 'id="opt-subs-after"' in html
    assert re.search(r'<input[^>]*id="opt-subs-after"[^>]*\bchecked\b', html) is None
    assert 'aria-describedby="manual-queue-path-error"' in html
    assert 'id="manual-queue-path-error" class="field-error" role="alert" hidden' in html
    for token in (
        'id="night-report-details"',
        'id="btn-refresh-night-report"',
        'id="btn-copy-night-report"',
        'id="night-report-pre"',
        'id="night-report-code"',
    ):
        assert token in html
    assert "8s" not in html and "8s" not in app_js

    fs_list_blk = re.search(r"\.fs-entries-list\s*\{([^}]*)\}", css)
    assert fs_list_blk and "align-content: start;" in fs_list_blk.group(1)
    assert '.fs-entries-list[aria-busy="true"]' in css
    assert ".fs-error-banner" in css

    m900 = re.search(r"@media\s*\(min-width:\s*900px\)\s*\{([\s\S]*?)\n\}", css)
    assert m900 and "--fade-size: 0px;" in m900.group(1) and "flex: 1 1 auto;" in m900.group(1)
    m899 = re.search(r"@media\s*\(max-width:\s*899px\)\s*\{([\s\S]*?)\n\}", css)
    assert m899 and "--fade-size: 1.25rem;" in m899.group(1) and "scrollbar-width: thin;" in m899.group(1)

    assert '.selected-input-item[data-error="true"]' in css
    table_blk = re.search(r"\.studio-table th,\s*\.studio-table td\s*\{([^}]*)\}", css)
    assert table_blk and "padding-block: 0.52rem;" in table_blk.group(1) and "padding-inline: 0.55rem;" in table_blk.group(1)
    models_th = re.search(r"#models-table tbody th\s*\{([^}]*)\}", css)
    assert models_th and "word-break: keep-all;" in models_th.group(1)
    models_code = re.search(r"#models-table tbody th code\s*\{([^}]*)\}", css)
    assert models_code and "white-space: nowrap;" in models_code.group(1) and "word-break: keep-all;" in models_code.group(1)

    assert "text-overflow: ellipsis;" in css and "white-space: nowrap;" in css
    assert ".compare-controls-grid" in css
    assert "grid-template-rows: minmax(2.65rem, auto) auto auto;" in css
    assert "grid-row: 2;" in css and "grid-row: 3;" in css
    assert "@media (max-width: 520px)" in css

    stepper_blk = re.search(r"\.pipeline-stepper\s*\{([^}]*)\}", css)
    assert stepper_blk and "repeat(2, minmax(0, 1fr))" in stepper_blk.group(1)
    m600 = re.search(r"@media\s*\(min-width:\s*600px\)\s*\{([\s\S]*?)\n\}", css)
    assert m600 and "repeat(3, minmax(0, 1fr))" in m600.group(1)
    m1080 = re.search(r"@media\s*\(min-width:\s*1080px\)\s*\{([\s\S]*?)\n\}", css)
    assert m1080 and "repeat(6, minmax(0, 1fr))" in m1080.group(1)

    m640 = re.search(r"@media\s*\(max-width:\s*640px\)\s*\{([\s\S]*?)\n\}", css)
    assert m640
    m640_body = m640.group(1)
    assert ".studio-subtitle" in m640_body and "display: none;" in m640_body
    assert ".header-pills" in m640_body and "repeat(2, minmax(0, 1fr))" in m640_body
    assert ".status-pill > span:not(.pill-icon)" in m640_body and "text-overflow: ellipsis;" in m640_body

    btn_dis = re.search(r"\.btn-primary:disabled\s*\{([^}]*)\}", css)
    assert btn_dis
    for prop in (
        "background-color: var(--bg-elevated);",
        "border-color: var(--border-subtle);",
        "color: var(--fg-muted);",
        "opacity: 0.65;",
    ):
        assert prop in btn_dis.group(1)

    assert fs_picker_js.count("ev.stopPropagation()") >= 4
    assert 'entriesList.setAttribute("aria-busy", "true")' in fs_picker_js
    assert 'entriesList.removeAttribute("aria-busy")' in fs_picker_js
    assert '"fs-error-banner"' in fs_picker_js
    assert 'role: "alert"' in fs_picker_js
    assert "showToast" not in fs_picker_js
    browse_pos = fs_picker_js.find("async browse(")
    fetch_pos = fs_picker_js.find("await apiFetch", browse_pos)
    clear_pos = fs_picker_js.find("clearChildren(entriesList)", browse_pos)
    assert 0 <= browse_pos < fetch_pos < clear_pos

    assert "manual-queue-path-error" in panels_js
    assert "handleManualAdd" in panels_js
    assert 'val.includes("\\0")' in panels_js
    manual_pos = panels_js.find("handleManualAdd")
    preval_pos = panels_js.find('apiFetch("/api/queue/preview-schedule"', manual_pos)
    add_pos = panels_js.find("addInputs([val])", manual_pos)
    assert 0 <= manual_pos < preval_pos < add_pos
    assert "queueErrorPaths" in panels_js

    assert "validateBatchSizeInput" in panels_js
    assert "Number.isFinite(batchVal)" in panels_js
    assert "output_dir: optOutDir?.value.trim() || null" in panels_js
    assert "force: Boolean(optForce?.checked)" in panels_js

    assert "isActiveUpscale" in app_js and "isCpActive" in app_js
    assert "▶ Đang xử lý…" in app_js
    assert "isCustomOutputDir" in app_js and "focusWithPath" in panels_js
    assert "scheduleDirty = true" in panels_js

    assert "syncStartUpscaleLabel" in panels_js
    assert "Boolean(optSubsAfter?.checked)" in panels_js
    assert "+ Dịch VI" in panels_js

    assert "queueCtrl?.setSystemBusy?.(isRunning, busyReason)" in app_js
    assert "subtransCtrl?.setSystemBusy?.(isRunning, busyReason)" in app_js
    assert "syncQueueLaunchButtons" in panels_js and "syncSubtransLaunchButtons" in panels_js
    assert "pendingSubsCount === 0" in panels_js

    assert "tiết kiệm" in app_js and "dedup)" in app_js
    assert "tiết kiệm" in panels_js and "dedup)" in panels_js
    assert "job.last_job" in app_js and "job.last_finished_job" in app_js
    assert "lastJobMatchesLive" in app_js
    assert "parseElapsedSeconds(startedIso, endedIso)" in app_js

    assert "isProgrammaticLogScroll" in app_js
    assert "distanceFromBottom > 36" in app_js
    assert "distanceFromBottom <= 12" in app_js
    assert "logAutoscroll.checked = false" in app_js
    assert "logAutoscroll.checked = true" in app_js

    assert "extractSeq" in compare_js
    assert "isCurrentExtract()" in compare_js
    assert "++extractSeq" in compare_js
    assert 'btnExtract.setAttribute("aria-busy", "true")' in compare_js

    assert "{ showUserWarning = false } = {}" in panels_js
    assert "showUserWarning: true" in panels_js
    assert "if (showUserWarning)" in panels_js

    assert '"aria-disabled": "true"' in panels_js
    assert '"aria-pressed"' in panels_js
    assert "if (!docPath)" in panels_js
    assert "/api/night-report" in panels_js
    assert "nightReportSeq" in panels_js
    assert "nightReportCode.textContent" in panels_js
    assert "res && res.ok === false" in app_js
    assert 'err?.status === 409 ? "warn" : "error"' in app_js
    assert "innerHTML" not in app_js


def test_abort_job_before_drain_finalization_preserves_paused_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Ensure post-abort finalization and TelemetryHub fallback survive blocked _drain_output()."""
    import psutil
    from web.job_manager import JobManager
    from web.sse_hub import TelemetryHub

    class _BlockingStdout:
        def __init__(self, started: threading.Event, release: threading.Event) -> None:
            self._started = started
            self._release = release

        def __iter__(self):
            yield "[INFO] Booting...\n"
            self._started.set()
            assert self._release.wait(timeout=5.0), "_drain_output release event timed out"
            yield "[INFO] Drain complete\n"

    class _AbortRaceProc(FakePopenProcess):
        def __init__(
            self,
            cmd: List[str],
            pid: int,
            alive_set: set[int],
            started: threading.Event,
            release: threading.Event,
            *,
            delay_exit_on_terminate: bool = False,
        ) -> None:
            super().__init__(cmd, pid=pid, running=True)
            self._alive_set = alive_set
            self._delay_exit_on_terminate = delay_exit_on_terminate
            self.stdout = _BlockingStdout(started, release)  # type: ignore[assignment]

        def wait(self, timeout: float | None = None) -> int:
            if self._running:
                if timeout is not None:
                    raise subprocess.TimeoutExpired(self.cmd, timeout)
                self._done.wait(timeout=5.0)
            self._running = False
            return self._return_code

        def terminate(self) -> None:
            self._alive_set.discard(25000)
            self._alive_set.discard(27000)
            if not self._delay_exit_on_terminate:
                self._running = False
                self._return_code = -1
                self._done.set()

        def mark_exited(self, rc: int = -1) -> None:
            self._running = False
            self._return_code = rc
            self._done.set()

    def _idle_gpu_metrics() -> Dict[str, Any]:
        return {
            "pstate": "P8",
            "gpu_util_pct": 0.0,
            "mem_util_pct": 0.0,
            "enc_util_pct": 0.0,
            "dec_util_pct": 0.0,
            "mem_clock_mhz": 405.0,
            "core_clock_mhz": 210.0,
            "power_w": 12.0,
            "temp_c": 38.0,
            "vram_used_mib": 256.0,
            "vram_free_mib": 12032.0,
            "vram_total_mib": 12288.0,
        }

    def _write_live_status(root_dir: Path, job_dict: Dict[str, Any]) -> None:
        (root_dir / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": "Frieren_10",
                    "pid": 27000,
                    "ppid": 25000,
                    "job_id": job_dict["job_id"],
                    "started_at": job_dict["started_at"],
                    "stage": "upscaling",
                    "total_chunks": 4,
                    "completed_chunks": 2,
                    "active_chunk": 1,
                    "steady_fps": 21.5,
                    "chunks": [
                        {"idx": 0, "status": "done"},
                        {"idx": 1, "status": "done"},
                        {"idx": 2, "status": "running"},
                        {"idx": 3, "status": "pending"},
                    ],
                }
            ),
            encoding="utf-8",
        )

    # Case 1: abort_job() completes while _drain_output() is parked in stdout iteration.
    root1 = _seed_workspace(tmp_path / "abort_race_ws1")
    alive_pids: set[int] = {25000, 27000}
    drain_started1 = threading.Event()
    drain_release1 = threading.Event()
    procs1: List[_AbortRaceProc] = []

    def popen1(cmd: List[str], **kwargs: Any) -> _AbortRaceProc:
        p = _AbortRaceProc(cmd, pid=25000, alive_set=alive_pids, started=drain_started1, release=drain_release1)
        procs1.append(p)
        return p

    monkeypatch.setattr(psutil, "pid_exists", lambda p: p in alive_pids)
    jm1 = JobManager(root1, popen_factory=popen1)
    job1 = jm1.start_upscale_job({"inputs": ["anime_season/[SubsPlease] Frieren - 10 (1080p).mkv"]})
    assert drain_started1.wait(timeout=5.0)
    _write_live_status(root1, job1)

    abort_res = jm1.abort_job()
    assert abort_res["ok"] is True
    assert abort_res["status"] == "aborted"
    assert procs1[0].poll() == -1

    # Verify TelemetryHub unfinalized "aborted" fallback directly (empty history and prior-job history).
    unfinalized_active = {**job1, "status": "aborted", "finished_at": None, "return_code": None}
    hub_empty = TelemetryHub(
        root1,
        gpu_query_fn=_idle_gpu_metrics,
        job_state_provider=lambda: {"status": "aborted", "active_job": dict(unfinalized_active), "history": []},
    )
    pipe_empty = hub_empty.read_pipeline_state()
    assert pipe_empty["effective_stage"] == "paused"
    assert pipe_empty["live"]["stage"] == "paused"
    assert pipe_empty["live"]["last_stage"] == "upscaling"
    assert pipe_empty["live"]["completed_chunks"] == 2
    assert pipe_empty["live"]["steady_fps"] == 21.5
    assert hub_empty._select_active_log_file() == Path(job1["log_path"])

    prior_job = {
        "job_id": "prior0001",
        "job_type": "upscale",
        "status": "completed",
        "started_at": "2026-11-22T00:00:00",
        "finished_at": "2026-11-22T00:05:00",
        "return_code": 0,
    }
    hub_prior = TelemetryHub(
        root1,
        gpu_query_fn=_idle_gpu_metrics,
        job_state_provider=lambda: {
            "status": "aborted",
            "active_job": dict(unfinalized_active),
            "history": [prior_job],
        },
    )
    pipe_prior = hub_prior.read_pipeline_state()
    assert pipe_prior["effective_stage"] == "paused"
    assert pipe_prior["live"]["completed_chunks"] == 2
    assert pipe_prior["live"]["steady_fps"] == 21.5

    # Verify jm1.get_state() finalized cleanly once while _drain_output() is still blocked.
    hub_jm1 = TelemetryHub(root1, gpu_query_fn=_idle_gpu_metrics, job_state_provider=jm1.get_state)
    state1 = jm1.get_state()
    assert state1["status"] == "aborted"
    assert state1["active_job"]["finished_at"] is not None
    assert state1["active_job"]["return_code"] == -1
    assert len(state1["history"]) == 1
    assert state1["history"][0]["job_id"] == job1["job_id"]
    assert hub_jm1.read_pipeline_state()["effective_stage"] == "paused"

    # Release _drain_output() and join thread to prove zero duplicate history entries.
    drain_thread1 = next(t for t in threading.enumerate() if t.name == f"WebJobDrain-{job1['job_id']}")
    drain_release1.set()
    drain_thread1.join(timeout=5.0)
    assert not drain_thread1.is_alive()

    state1_after = jm1.get_state()
    assert len(state1_after["history"]) == 1
    assert state1_after["status"] == "aborted"
    assert hub_jm1.read_pipeline_state()["effective_stage"] == "paused"

    # Case 2: proc finishes exiting AFTER abort_job() returns while _drain_output() is still blocked;
    # TelemetryHub preserves "paused" pre-finalization and jm2.get_state() finalizes cleanly once.
    root2 = _seed_workspace(tmp_path / "abort_race_ws2")
    alive_pids2: set[int] = {25000, 27000}
    drain_started2 = threading.Event()
    drain_release2 = threading.Event()
    procs2: List[_AbortRaceProc] = []

    def popen2(cmd: List[str], **kwargs: Any) -> _AbortRaceProc:
        p = _AbortRaceProc(
            cmd,
            pid=25000,
            alive_set=alive_pids2,
            started=drain_started2,
            release=drain_release2,
            delay_exit_on_terminate=True,
        )
        procs2.append(p)
        return p

    monkeypatch.setattr(psutil, "pid_exists", lambda p: p in alive_pids2)
    jm2 = JobManager(root2, popen_factory=popen2)
    job2 = jm2.start_upscale_job({"inputs": ["anime_season/[SubsPlease] Frieren - 10 (1080p).mkv"]})
    assert drain_started2.wait(timeout=5.0)
    _write_live_status(root2, job2)

    abort_res2 = jm2.abort_job()
    assert abort_res2["ok"] is True
    assert procs2[0].poll() is None

    hub_jm2 = TelemetryHub(root2, gpu_query_fn=_idle_gpu_metrics, job_state_provider=jm2.get_state)
    pre_exit_state = jm2.get_state()
    assert pre_exit_state["status"] == "aborted"
    assert pre_exit_state["active_job"]["finished_at"] is None
    assert pre_exit_state["history"] == []
    assert hub_jm2.read_pipeline_state()["effective_stage"] == "paused"

    procs2[0].mark_exited(-1)
    assert procs2[0].poll() == -1
    post_exit_state = jm2.get_state()
    assert post_exit_state["status"] == "aborted"
    assert post_exit_state["active_job"]["finished_at"] is not None
    assert post_exit_state["active_job"]["return_code"] == -1
    assert len(post_exit_state["history"]) == 1
    assert hub_jm2.read_pipeline_state()["effective_stage"] == "paused"

    drain_thread2 = next(t for t in threading.enumerate() if t.name == f"WebJobDrain-{job2['job_id']}")
    drain_release2.set()
    drain_thread2.join(timeout=5.0)
    assert not drain_thread2.is_alive()
    assert len(jm2.get_state()["history"]) == 1
    assert hub_jm2.read_pipeline_state()["effective_stage"] == "paused"


def test_concurrent_graceful_pause_and_clear_zero_permission_error_or_desync(tmp_path: Path):
    """Concurrent pause/clear calls must never raise PermissionError or desync status vs flag."""
    from web.job_manager import JobManager

    root = _seed_workspace(tmp_path / "pause_clear_race_ws")
    flag_path = root / "logs" / ".graceful_pause.flag"
    holder: Dict[str, FakePopenProcess] = {}

    def running_factory(cmd: List[str], **_kwargs: Any) -> FakePopenProcess:
        proc = FakePopenProcess(cmd, pid=52520, running=True, return_code=0)
        holder["proc"] = proc
        return proc

    jm = JobManager(root, popen_factory=running_factory)
    src_ep = str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")
    job = jm.start_upscale_job({"inputs": [src_ep], "dry_run": True})
    assert job["status"] == "running"

    rounds = 40
    start_barrier = threading.Barrier(5)
    end_barrier = threading.Barrier(5)
    errors: List[BaseException] = []

    def pause_worker() -> None:
        try:
            for _ in range(rounds):
                start_barrier.wait(timeout=5.0)
                res = jm.request_graceful_pause()
                assert res["ok"] is True
                end_barrier.wait(timeout=5.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def clear_worker() -> None:
        try:
            for _ in range(rounds):
                start_barrier.wait(timeout=5.0)
                res = jm.clear_graceful_pause()
                assert res["ok"] is True
                end_barrier.wait(timeout=5.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=pause_worker, daemon=True),
        threading.Thread(target=pause_worker, daemon=True),
        threading.Thread(target=pause_worker, daemon=True),
        threading.Thread(target=clear_worker, daemon=True),
    ]
    for t in threads:
        t.start()

    for _ in range(rounds):
        start_barrier.wait(timeout=5.0)
        end_barrier.wait(timeout=5.0)
        cur_status = jm.get_state()["status"]
        assert cur_status in {"running", "pausing"}
        assert (cur_status == "pausing") == flag_path.is_file()
        assert not list((root / "logs").glob(".graceful_pause.flag.*.tmp"))

    for t in threads:
        t.join(timeout=15.0)
        assert not t.is_alive()
    assert not errors

    # Clear pause cleanly back to running, then complete normally; external-CLI pause remains allowed
    assert jm.clear_graceful_pause()["graceful_pause_pending"] is False
    assert jm.get_state()["status"] == "running"
    assert not flag_path.exists()
    holder["proc"]._running = False
    holder["proc"]._return_code = 0
    holder["proc"]._done.set()
    for _ in range(40):
        if jm.get_state()["status"] == "completed":
            break
        threading.Event().wait(0.01)
    assert jm.get_state()["status"] == "completed"
    ext_after_done = jm.request_graceful_pause()
    assert ext_after_done["ok"] is True
    assert ext_after_done["graceful_pause_pending"] is True
    assert flag_path.is_file()
    assert jm.clear_graceful_pause()["graceful_pause_pending"] is False
    assert not flag_path.exists()

    root_cli = _seed_workspace(tmp_path / "pause_cli_compat_ws")
    jm_cli = JobManager(root_cli, popen_factory=running_factory)
    ext_res = jm_cli.request_graceful_pause()
    assert ext_res["ok"] is True
    assert ext_res["graceful_pause_pending"] is True
    assert (root_cli / "logs" / ".graceful_pause.flag").is_file()
    clr_res = jm_cli.clear_graceful_pause()
    assert clr_res["ok"] is True
    assert clr_res["graceful_pause_pending"] is False
    assert not (root_cli / "logs" / ".graceful_pause.flag").exists()


def test_graceful_pause_racing_abort_job_never_orphans_flag(tmp_path: Path):
    """Racing request_graceful_pause() against abort_job() must never leave .graceful_pause.flag orphaned."""
    from web.job_manager import JobManager

    root = _seed_workspace(tmp_path / "pause_abort_race_ws")
    flag_path = root / "logs" / ".graceful_pause.flag"
    src_ep = str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")

    def running_factory(cmd: List[str], **_kwargs: Any) -> FakePopenProcess:
        return FakePopenProcess(cmd, pid=53530, running=True, return_code=-1)

    def _pause(b: threading.Barrier, out: List[Dict[str, Any]]) -> None:
        b.wait(timeout=5.0)
        out.append(jm.request_graceful_pause())

    def _abort(b: threading.Barrier, out: List[Dict[str, Any]]) -> None:
        b.wait(timeout=5.0)
        out.append(jm.abort_job())

    jm = JobManager(root, popen_factory=running_factory)
    for _ in range(30):
        jm.start_upscale_job({"inputs": [src_ep], "dry_run": True})
        barrier = threading.Barrier(2)
        pause_results: List[Dict[str, Any]] = []
        abort_results: List[Dict[str, Any]] = []

        t1 = threading.Thread(target=_pause, args=(barrier, pause_results), daemon=True)
        t2 = threading.Thread(target=_abort, args=(barrier, abort_results), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert not t1.is_alive() and not t2.is_alive()
        assert abort_results[0]["ok"] is True
        if pause_results[0]["ok"]:
            assert pause_results[0]["graceful_pause_pending"] is True
        else:
            assert pause_results[0]["status"] == "aborted"
            assert pause_results[0]["graceful_pause_pending"] is False

        state = jm.get_state()
        assert state["status"] == "aborted"
        assert state["active_job"]["finished_at"] is not None
        assert not flag_path.exists()
        assert not list((root / "logs").glob(".graceful_pause.flag.*.tmp"))

    # Deterministic mid-abort teardown test: request_graceful_pause() during terminate()/wait() window
    mid_started = threading.Event()
    mid_release = threading.Event()

    class _MidAbortProc(FakePopenProcess):
        def terminate(self) -> None:
            mid_started.set()
            assert mid_release.wait(timeout=5.0)
            super().terminate()

    jm_mid = JobManager(root, popen_factory=lambda cmd, **kw: _MidAbortProc(cmd, pid=54540, running=True))
    jm_mid.start_upscale_job({"inputs": [src_ep], "dry_run": True})
    abort_done: List[Dict[str, Any]] = []
    t_abort = threading.Thread(target=lambda: abort_done.append(jm_mid.abort_job()), daemon=True)
    t_abort.start()
    assert mid_started.wait(timeout=5.0)

    mid_pause = jm_mid.request_graceful_pause()
    assert mid_pause["ok"] is False
    assert mid_pause["status"] == "aborted"
    assert mid_pause["graceful_pause_pending"] is False
    assert not flag_path.exists()

    mid_release.set()
    t_abort.join(timeout=5.0)
    assert not t_abort.is_alive()
    assert abort_done[0]["ok"] is True
    assert jm_mid.get_state()["status"] == "aborted"
    assert not flag_path.exists()

    # Post-abort call after finalization must also refuse writing .graceful_pause.flag
    post_pause = jm_mid.request_graceful_pause()
    assert post_pause["ok"] is False
    assert post_pause["status"] == "aborted"
    assert post_pause["graceful_pause_pending"] is False
    assert not flag_path.exists()


def test_pause_graceful_after_aborted_web_job_refuses_without_external_cli_with_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """POST /api/jobs/pause-graceful returns HTTP 409 and writes no flag after abort when idle."""
    import psutil

    root = _seed_workspace(tmp_path / "pause_abort_409_ws")
    flag_path = root / "logs" / ".graceful_pause.flag"
    src_ep = str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")
    alive_pids: set[int] = {61610}

    def running_factory(cmd: List[str], **_kwargs: Any) -> FakePopenProcess:
        return FakePopenProcess(cmd, pid=61610, running=True, return_code=-1)

    monkeypatch.setattr(psutil, "pid_exists", lambda p: p in alive_pids)
    app = create_app(workspace_root=root, popen_factory=running_factory)
    with TestClient(app) as client:
        start_resp = client.post("/api/jobs/start", json={"inputs": [src_ep], "dry_run": True})
        assert start_resp.status_code == 200
        job = start_resp.json()["job"]

        # Simulate live_status.json written by the aborted Web Job before teardown
        (root / "logs" / "live_status.json").write_text(
            json.dumps({"job_id": job["job_id"], "pid": 61610, "stage": "upscaling"}),
            encoding="utf-8",
        )

        abort_resp = client.post("/api/jobs/abort")
        assert abort_resp.status_code == 200
        assert abort_resp.json()["ok"] is True
        alive_pids.discard(61610)

        pause_resp = client.post("/api/jobs/pause-graceful")
        assert pause_resp.status_code == 409
        payload = pause_resp.json()
        assert payload["ok"] is False
        assert payload["status"] == "aborted"
        assert payload["graceful_pause_pending"] is False
        assert payload["flag_path"] == str(flag_path)
        assert isinstance(payload.get("message"), str) and payload["message"]
        assert not flag_path.exists()

        # Even if a stale live_status.json has a terminal stage for another PID, refuse with 409
        alive_pids.add(os.getpid())
        (root / "logs" / "live_status.json").write_text(
            json.dumps({"pid": os.getpid(), "stage": "paused"}),
            encoding="utf-8",
        )
        stale_resp = client.post("/api/jobs/pause-graceful")
        assert stale_resp.status_code == 409
        assert stale_resp.json()["ok"] is False
        assert not flag_path.exists()


def test_pause_graceful_after_aborted_web_job_allows_live_external_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """POST /api/jobs/pause-graceful returns HTTP 200 and creates flag for live external CLI post-abort."""
    import psutil

    root = _seed_workspace(tmp_path / "pause_abort_ext_cli_ws")
    flag_path = root / "logs" / ".graceful_pause.flag"
    src_ep = str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")
    ext_pid = os.getpid()
    alive_pids: set[int] = {62620, ext_pid}

    def running_factory(cmd: List[str], **_kwargs: Any) -> FakePopenProcess:
        return FakePopenProcess(cmd, pid=62620, running=True, return_code=-1)

    monkeypatch.setattr(psutil, "pid_exists", lambda p: p in alive_pids)
    app = create_app(workspace_root=root, popen_factory=running_factory)
    with TestClient(app) as client:
        start_resp = client.post("/api/jobs/start", json={"inputs": [src_ep], "dry_run": True})
        assert start_resp.status_code == 200
        abort_resp = client.post("/api/jobs/abort")
        assert abort_resp.status_code == 200
        assert abort_resp.json()["ok"] is True
        alive_pids.discard(62620)
        assert app.state.job_manager.get_state()["status"] == "aborted"

        # Case 1: external CLI holds .gpu_active.lock + live_status.json
        lock_file = root / ".gpu_active.lock"
        lock_file.write_text(f"{ext_pid}\n", encoding="utf-8")
        (root / "logs" / "live_status.json").write_text(
            json.dumps({"pid": ext_pid, "stage": "upscaling"}),
            encoding="utf-8",
        )
        pipe_before = app.state.telemetry_hub.read_pipeline_state()
        assert pipe_before["active_source"] == "cli"
        assert pipe_before["gpu_lock"]["locked"] is True

        pause_resp = client.post("/api/jobs/pause-graceful")
        assert pause_resp.status_code == 200
        pause_json = pause_resp.json()
        assert pause_json["ok"] is True
        assert pause_json["graceful_pause_pending"] is True
        assert flag_path.is_file()
        assert app.state.job_manager.get_state()["status"] == "aborted"
        pipe_paused = app.state.telemetry_hub.read_pipeline_state()
        assert pipe_paused["active_source"] == "cli"
        assert pipe_paused["graceful_pause_pending"] is True

        clear_resp = client.post("/api/jobs/clear-pause")
        assert clear_resp.status_code == 200
        assert clear_resp.json()["graceful_pause_pending"] is False
        assert not flag_path.exists()

        # Case 2: external CLI in active pre-lock stage (live_status.json only, no .gpu_active.lock)
        lock_file.unlink()
        (root / "logs" / "live_status.json").write_text(
            json.dumps({"pid": ext_pid, "stage": "probing"}),
            encoding="utf-8",
        )
        pause_resp_2 = client.post("/api/jobs/pause-graceful")
        assert pause_resp_2.status_code == 200
        assert pause_resp_2.json()["ok"] is True
        assert pause_resp_2.json()["graceful_pause_pending"] is True
        assert flag_path.is_file()
        assert client.post("/api/jobs/clear-pause").status_code == 200
        assert not flag_path.exists()


def test_pause_graceful_during_abort_teardown_dying_child_lock_refuses_with_409(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Dying child holding .gpu_active.lock during or after abort_job() teardown must return HTTP 409."""
    import psutil

    root = _seed_workspace(tmp_path / "pause_abort_dying_child_ws")
    flag_path = root / "logs" / ".graceful_pause.flag"
    lock_file = root / ".gpu_active.lock"
    src_ep = str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")

    launcher_pid = 91000
    child_pid = 91001
    teardown_entered = threading.Event()
    teardown_release = threading.Event()
    child_alive = {"value": True}

    orig_process = psutil.Process

    class _FakeChildProc:
        pid = child_pid

        def ppid(self) -> int:
            return launcher_pid

    class _FakeLauncherProc:
        pid = launcher_pid

        def children(self, recursive: bool = True) -> List[Any]:
            return [_FakeChildProc()]

    def fake_process(pid: int | None = None) -> Any:
        if pid == launcher_pid:
            return _FakeLauncherProc()
        if pid == child_pid:
            return _FakeChildProc()
        return orig_process(pid)

    def fake_pid_exists(pid: int) -> bool:
        if pid == child_pid:
            return child_alive["value"]
        return False

    class _TeardownRaceProc(FakePopenProcess):
        def wait(self) -> int:
            teardown_entered.set()
            assert teardown_release.wait(timeout=5.0)
            return super().wait()

    monkeypatch.setattr(psutil, "Process", fake_process)
    monkeypatch.setattr(psutil, "pid_exists", fake_pid_exists)

    app = create_app(
        workspace_root=root,
        popen_factory=lambda cmd, **_kw: _TeardownRaceProc(cmd, pid=launcher_pid, running=True, return_code=-1),
    )
    with TestClient(app) as client:
        start_resp = client.post("/api/jobs/start", json={"inputs": [src_ep], "dry_run": True})
        assert start_resp.status_code == 200
        job = start_resp.json()["job"]

        lock_file.write_text(f"{child_pid}\n", encoding="utf-8")
        (root / "logs" / "live_status.json").write_text(
            json.dumps({"job_id": job["job_id"], "pid": child_pid, "ppid": launcher_pid, "stage": "upscaling"}),
            encoding="utf-8",
        )

        abort_holder: List[Any] = []
        t_abort = threading.Thread(target=lambda: abort_holder.append(client.post("/api/jobs/abort")), daemon=True)
        t_abort.start()
        assert teardown_entered.wait(timeout=5.0)

        # Even if get_state() observes launcher exit and finalizes _active_job during teardown,
        # pause-graceful must still refuse with 409 and never orphan .graceful_pause.flag.
        assert client.get("/api/status").status_code == 200
        pause_during = client.post("/api/jobs/pause-graceful")
        assert pause_during.status_code == 409
        assert pause_during.json()["ok"] is False
        assert pause_during.json()["status"] == "aborted"
        assert pause_during.json()["graceful_pause_pending"] is False
        assert not flag_path.exists()

        child_alive["value"] = False
        teardown_release.set()
        t_abort.join(timeout=5.0)
        assert not t_abort.is_alive()
        assert abort_holder[0].status_code == 200
        assert abort_holder[0].json()["ok"] is True
        assert not lock_file.exists()
        assert not flag_path.exists()
        assert app.state.job_manager.get_state()["status"] == "aborted"

        # A lingering descendant holding .gpu_active.lock or live_status.json after abort_job() returns must still return 409.
        child_alive["value"] = True
        lock_file.write_text(f"{child_pid}\n", encoding="utf-8")
        pause_lingering = client.post("/api/jobs/pause-graceful")
        assert pause_lingering.status_code == 409
        assert pause_lingering.json()["ok"] is False
        assert not flag_path.exists()

        lock_file.unlink()
        (root / "logs" / "live_status.json").write_text(
            json.dumps({"pid": child_pid, "stage": "upscaling"}),
            encoding="utf-8",
        )
        pause_lingering_live = client.post("/api/jobs/pause-graceful")
        assert pause_lingering_live.status_code == 409
        assert pause_lingering_live.json()["ok"] is False
        assert not flag_path.exists()


def test_pause_graceful_after_abort_rejects_foreign_job_id_and_aborted_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Post-abort external CLI detection rejects non-empty foreign job_id and 'aborted' stage in live_status.json."""
    import psutil

    root = _seed_workspace(tmp_path / "pause_abort_foreign_edge_ws")
    flag_path = root / "logs" / ".graceful_pause.flag"
    src_ep = str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")
    ext_pid = os.getpid()
    alive_pids: set[int] = {63630, ext_pid}

    monkeypatch.setattr(psutil, "pid_exists", lambda p: p in alive_pids)
    app = create_app(
        workspace_root=root,
        popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=63630, running=True, return_code=-1),
    )
    with TestClient(app) as client:
        assert client.post("/api/jobs/start", json={"inputs": [src_ep], "dry_run": True}).status_code == 200
        assert client.post("/api/jobs/abort").status_code == 200
        alive_pids.discard(63630)

        # Case 1: Foreign non-empty job_id in live_status.json (not an external CLI) -> HTTP 409
        (root / "logs" / "live_status.json").write_text(
            json.dumps({"job_id": "prev_web_job_99", "pid": ext_pid, "stage": "upscaling"}),
            encoding="utf-8",
        )
        foreign_resp = client.post("/api/jobs/pause-graceful")
        assert foreign_resp.status_code == 409
        assert foreign_resp.json()["ok"] is False
        assert not flag_path.exists()

        # Case 2: Terminal 'aborted' stage in live_status.json -> HTTP 409
        (root / "logs" / "live_status.json").write_text(
            json.dumps({"pid": ext_pid, "stage": "aborted"}),
            encoding="utf-8",
        )
        aborted_stage_resp = client.post("/api/jobs/pause-graceful")
        assert aborted_stage_resp.status_code == 409
        assert aborted_stage_resp.json()["ok"] is False
        assert not flag_path.exists()

        # Case 3: Genuine external CLI (blank job_id + active stage) -> HTTP 200
        (root / "logs" / "live_status.json").write_text(
            json.dumps({"job_id": "   ", "pid": ext_pid, "stage": "upscaling"}),
            encoding="utf-8",
        )
        cli_resp = client.post("/api/jobs/pause-graceful")
        assert cli_resp.status_code == 200
        assert cli_resp.json()["ok"] is True
        assert flag_path.is_file()
        assert client.post("/api/jobs/clear-pause").status_code == 200
        assert not flag_path.exists()


@pytest.mark.parametrize("dry_run", [True, False])
def test_abort_teardown_preserves_newer_job_pausing_flag(
    tmp_path: Path, dry_run: bool
):
    """Job 1 abort_job() finally must never unlink .graceful_pause.flag owned by a newer Job 2."""
    root = _seed_workspace(tmp_path / f"abort_cross_job_flag_{dry_run}_ws")
    flag_path = root / "logs" / ".graceful_pause.flag"
    src_ep = str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")
    teardown_entered = threading.Event()
    teardown_release = threading.Event()
    spawn_idx = {"n": 0}
    procs: List[FakePopenProcess] = []

    class _Job1TeardownHoldProc(FakePopenProcess):
        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None:
                teardown_entered.set()
                assert teardown_release.wait(timeout=5.0)
            return super().wait()

    def factory(cmd: List[str], **_kwargs: Any) -> FakePopenProcess:
        spawn_idx["n"] += 1
        if spawn_idx["n"] == 1:
            p = _Job1TeardownHoldProc(cmd, pid=71001, running=True, return_code=-1)
        else:
            p = FakePopenProcess(cmd, pid=71002, running=True, return_code=0)
        procs.append(p)
        return p

    app = create_app(workspace_root=root, popen_factory=factory)
    with TestClient(app) as client:
        start1 = client.post("/api/jobs/start", json={"inputs": [src_ep], "dry_run": dry_run})
        assert start1.status_code == 200
        job1_id = start1.json()["job"]["job_id"]

        abort_holder: List[Dict[str, Any]] = []
        t_abort = threading.Thread(
            target=lambda: abort_holder.append(app.state.job_manager.abort_job()),
            daemon=True,
        )
        t_abort.start()
        try:
            assert teardown_entered.wait(timeout=5.0)

            start2 = client.post("/api/jobs/start", json={"inputs": [src_ep], "dry_run": dry_run})
            assert start2.status_code == 200
            job2_id = start2.json()["job"]["job_id"]
            assert job2_id != job1_id

            pause2 = client.post("/api/jobs/pause-graceful")
            assert pause2.status_code == 200
            assert pause2.json()["ok"] is True
            assert pause2.json()["graceful_pause_pending"] is True
            assert flag_path.is_file()
            assert app.state.job_manager.get_state()["status"] == "pausing"
        finally:
            teardown_release.set()
            t_abort.join(timeout=5.0)

        assert not t_abort.is_alive()
        assert abort_holder[0]["ok"] is True

        state_after = app.state.job_manager.get_state()
        assert state_after["status"] == "pausing"
        assert state_after["active_job"]["job_id"] == job2_id
        assert flag_path.is_file()

        assert client.post("/api/jobs/clear-pause").status_code == 200
        procs[1].terminate()


def test_abort_subtrans_job_preserves_external_cli_gpu_lock_and_pause_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Aborting a non-GPU subtrans Web Job must preserve an active external CLI's GPU lock and pause flag."""
    import psutil

    root = _seed_workspace(tmp_path / "abort_subtrans_ext_cli_ws")
    flag_path = root / "logs" / ".graceful_pause.flag"
    lock_file = root / ".gpu_active.lock"
    src_ep = str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")
    ext_pid = os.getpid()
    sub_pid = 72001
    alive_pids: set[int] = {ext_pid, sub_pid}

    monkeypatch.setattr(psutil, "pid_exists", lambda p: p in alive_pids)

    def sub_factory(cmd: List[str], **_kwargs: Any) -> FakePopenProcess:
        return FakePopenProcess(cmd, pid=sub_pid, running=True, return_code=-1)

    app = create_app(workspace_root=root, popen_factory=sub_factory)
    with TestClient(app) as client:
        # External CLI is active and already requested graceful pause before subtrans starts & aborts.
        lock_file.write_text(f"{ext_pid}\n", encoding="utf-8")
        (root / "logs" / "live_status.json").write_text(
            json.dumps({"pid": ext_pid, "stage": "upscaling"}),
            encoding="utf-8",
        )
        pause_before = client.post("/api/jobs/pause-graceful")
        assert pause_before.status_code == 200
        assert flag_path.is_file()

        sub_resp = client.post("/api/jobs/subtrans", json={"input_path": src_ep, "dry_run": True})
        assert sub_resp.status_code == 200
        assert sub_resp.json()["job"]["job_type"] == "subtrans"
        assert flag_path.is_file()

        abort_resp = client.post("/api/jobs/abort")
        assert abort_resp.status_code == 200
        assert abort_resp.json()["ok"] is True
        alive_pids.discard(sub_pid)

        assert app.state.job_manager.get_state()["status"] == "aborted"
        assert lock_file.is_file()
        assert flag_path.is_file()

        # External CLI pause/clear controls continue working after subtrans abort.
        pause_after = client.post("/api/jobs/pause-graceful")
        assert pause_after.status_code == 200
        assert pause_after.json()["ok"] is True
        assert flag_path.is_file()
        clear_after = client.post("/api/jobs/clear-pause")
        assert clear_after.status_code == 200
        assert not flag_path.exists()


def test_abort_job_discards_aborting_token_on_unexpected_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Unexpected exception inside abort_job() lock block must still discard abort_token in finally."""
    from web.job_manager import JobManager

    root = _seed_workspace(tmp_path / "abort_token_exc_ws")
    flag_path = root / "logs" / ".graceful_pause.flag"
    src_ep = str(root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv")
    jm = JobManager(
        root,
        popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=73001, running=True, return_code=-1),
    )
    jm.start_upscale_job({"inputs": [src_ep], "dry_run": True})
    assert jm.request_graceful_pause()["ok"] is True
    assert flag_path.is_file()

    def _boom(_aborted_job: Dict[str, Any]) -> bool:
        raise RuntimeError("injected pause-check failure")

    monkeypatch.setattr(jm, "_has_live_external_cli_locked", _boom)
    with pytest.raises(RuntimeError, match="injected pause-check failure"):
        jm.abort_job()

    assert jm._aborting_tokens == set()
    monkeypatch.undo()
    # Subsequent pause/clear or status polling remains clean after exception recovery.
    assert jm.get_state()["status"] == "aborted"
    assert jm.clear_graceful_pause()["ok"] is True


def test_qa_remediation_intra_chunk_progress_and_log_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify 1Hz intra-chunk progress updates without mutating completed_chunks and run-log pruning."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(logger_mod, "LOGS_DIR", str(log_dir))

    import time as _time_mod

    old_ts = _time_mod.time() - 600.0
    for i in range(46):
        old_log = log_dir / f"upscale_20260101_0000{i:02d}_ep.log"
        old_log.write_text("old\n", encoding="utf-8")
        os.utime(old_log, (old_ts + i, old_ts + i))

    rl = logger_mod.RunLogger("Mushoku_S01E21.mkv")
    rl.update_live_context(
        model_id="realcugan-pro-conservative-2x",
        total_frames=1138,
        unique_frames=700,
        total_chunks=2,
        zero_disk=True,
    )
    assert len(list(log_dir.glob("upscale_*.log"))) <= 41

    rl.set_stage("upscaling", "Đang nâng cấp chunk 1/2", active_chunk=0)
    rl._last_intra_flush_ts = 0.0
    rl.update_active_chunk_progress(
        chunk_idx=0,
        done_distinct=175,
        total_distinct=350,
        chunk_elapsed_sec=20.0,
        chunk_total_frames=576,
        force=True,
    )
    snap = logger_mod.read_live_status()
    assert snap["completed_chunks"] == 0
    assert snap["chunks"] == []
    assert snap["active_chunk"] == 0
    assert snap["active_chunk_fraction"] == pytest.approx(0.5, abs=0.02)
    assert snap["latest_fps"] > 0.0
    assert snap["effective_fps"] > snap["latest_fps"]
    assert snap["eta_seconds"] is not None and snap["eta_seconds"] > 0

    dummy_seg = tmp_path / "up_0000.mp4"
    dummy_seg.write_bytes(b"x" * 16384)
    rl.log_chunk_done(
        chunk_idx=0,
        total_chunks=2,
        frames=576,
        ai_time=40.0,
        ai_fps=8.75,
        enc_time=0.5,
        total_chunk_time=40.5,
        output_seg=str(dummy_seg),
    )
    snap_done = logger_mod.read_live_status()
    assert snap_done["completed_chunks"] == 1
    assert snap_done["active_chunk_fraction"] == 0.0
    assert len(snap_done["chunks"]) == 1


def test_qa_remediation_checkpoint_metadata_schedule_deduction_and_safe_delete(tmp_path: Path):
    """Verify full checkpoint metadata, completed-chunk ETA deduction, and safe /api/checkpoints/delete guards."""
    root = _seed_workspace(tmp_path / "qa_cp_ws")
    ep_path = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    cp_dir = root / "out" / ".temp_[BD 1080p] Fate_Ep22"
    up_segs = cp_dir / "upscaled_segs"
    up_segs.mkdir(parents=True, exist_ok=True)
    (cp_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source_file": str(ep_path),
                "model_id": "realcugan-pro-conservative-2x",
                "scale": 2,
                "out_w": 3840,
                "out_h": 2160,
                "dedup_policy": "dis-auto",
                "dedup_policy_requested": "auto",
                "grain_restore": "off",
                "nvenc_multipass_policy": "qres",
                "zero_disk": True,
                "use_tensorrt": True,
                "batch_size": 4,
                "total_frames": 2400,
                "unique_frames": 1200,
                "total_chunks": 4,
            }
        ),
        encoding="utf-8",
    )
    (cp_dir / "plan.json").write_text(
        json.dumps(
            {
                "version": 1,
                "dedup_policy": "dis-auto",
                "total_frames": 2400,
                "total_kept_frames": 1200,
                "chunks": [
                    {"chunk_idx": 0, "total_frames": 600, "kept_count": 300},
                    {"chunk_idx": 1, "total_frames": 600, "kept_count": 300},
                    {"chunk_idx": 2, "total_frames": 600, "kept_count": 300},
                    {"chunk_idx": 3, "total_frames": 600, "kept_count": 300},
                ],
            }
        ),
        encoding="utf-8",
    )
    (up_segs / "up_0000.mp4").write_bytes(b"0" * 16384)
    (up_segs / "up_0001.mp4").write_bytes(b"1" * 16384)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        st = client.get("/api/status").json()
        cps = st["pipeline"]["checkpoints"]
        target_cp = next(c for c in cps if c["name"] == cp_dir.name)
        assert target_cp["completed_chunks"] == 2
        assert target_cp["total_chunks"] == 4
        assert target_cp["model_id"] == "realcugan-pro-conservative-2x"
        assert target_cp["dedup_policy"] == "dis-auto"
        assert target_cp["dedup_policy_requested"] == "auto"
        assert target_cp["grain_restore"] == "off"
        assert target_cp["nvenc_multipass_policy"] == "qres"
        assert target_cp["batch_size"] == 4
        assert target_cp["zero_disk"] is True
        assert target_cp["use_tensorrt"] is True

        sched_resume = client.post(
            "/api/queue/preview-schedule",
            json={"inputs": [str(ep_path)], "model": "realcugan-pro-conservative-2x", "force": False},
        )
        assert sched_resume.status_code == 200
        ep_res = sched_resume.json()["episodes"][0]
        assert "Tiếp tục checkpoint 2/4" in ep_res["routing_source"]

        sched_force = client.post(
            "/api/queue/preview-schedule",
            json={"inputs": [str(ep_path)], "model": "realcugan-pro-conservative-2x", "force": True},
        )
        assert sched_force.status_code == 200
        ep_force = sched_force.json()["episodes"][0]
        assert ep_res["eta_seconds"] < ep_force["eta_seconds"]

        # Reject deleting arbitrary non-checkpoint folder
        bad_del = client.post("/api/checkpoints/delete", json={"path": str(root / "out")})
        assert bad_del.status_code == 400

        # Reject deleting unknown .temp_ folder lacking checkpoint markers
        stray_dir = root / "out" / ".temp_not_a_checkpoint"
        stray_dir.mkdir(parents=True, exist_ok=True)
        (stray_dir / "notes.txt").write_text("hi", encoding="utf-8")
        untrusted_del = client.post("/api/checkpoints/delete", json={"path": str(stray_dir)})
        assert untrusted_del.status_code in (400, 403)
        assert stray_dir.is_dir()

        # Reject deleting when GPU lock is active
        lock_file = root / ".gpu_active.lock"
        lock_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
        try:
            locked_del = client.post("/api/checkpoints/delete", json={"path": str(cp_dir)})
            assert locked_del.status_code == 409
            assert cp_dir.is_dir()
        finally:
            lock_file.unlink(missing_ok=True)

        # Cleanly delete known idle checkpoint
        ok_del = client.post("/api/checkpoints/delete", json={"path": str(cp_dir)})
        assert ok_del.status_code == 200
        assert ok_del.json()["ok"] is True
        assert not cp_dir.exists()


def test_qa_remediation_input_filtering_folder_routing_dryrun_compare_and_cli_flags(tmp_path: Path):
    """Verify .srt / _4K rejection on upscale, subtrans dry_run status, dominant folder_routing, blank compare, and UI tokens."""
    root = _seed_workspace(tmp_path / "qa_misc_ws")
    season_dir = root / "anime_season"
    srt_file = season_dir / "Fate_Ep22.ja.srt"
    srt_file.write_text("1\n00:00:01,000 --> 00:00:02,000\nテスト\n", encoding="utf-8")
    already_4k = root / "out" / "[BD 1080p] Fate_Ep22_4K.mkv"
    already_4k.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 512)
    src_ep = season_dir / "[BD 1080p] Fate_Ep22.mkv"

    spawned_cmds: List[List[str]] = []

    def capture_factory(cmd: List[str], **_kw: Any) -> FakePopenProcess:
        spawned_cmds.append(list(cmd))
        return FakePopenProcess(cmd, pid=81234, running=True, return_code=0)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, popen_factory=capture_factory)
    with TestClient(app) as client:
        # Reject .srt and *_4K.mkv in upscale / preview / schedule
        for bad_input in (str(srt_file), str(already_4k)):
            assert client.post("/api/queue/preview-schedule", json={"inputs": [bad_input]}).status_code == 400
            assert client.post("/api/jobs/start", json={"inputs": [bad_input], "dry_run": True}).status_code == 400
            assert client.post("/api/jobs/preview", json={"video_path": bad_input}).status_code == 400

        # Allow both .srt and *_4K.mkv in subtrans, and reflect dry_run in live status
        sub_ok = client.post("/api/jobs/subtrans", json={"input_path": str(srt_file), "dry_run": True})
        assert sub_ok.status_code == 200
        st_sub = client.get("/api/status").json()
        assert st_sub["pipeline"]["live"]["dry_run"] is True
        assert "Dry-Run" in st_sub["pipeline"]["live"]["stage_detail"]
        assert client.post("/api/jobs/abort").status_code == 200

        # --no-subs omits redundant subtitle flags
        spawned_cmds.clear()
        start_nosubs = client.post(
            "/api/jobs/start",
            json={"inputs": [str(src_ep)], "subs_after_upscale": False, "dry_run": True},
        )
        assert start_nosubs.status_code == 200
        cmd_used = spawned_cmds[-1]
        assert "--no-subs" in cmd_used
        assert "--subs-quality" not in cmd_used
        assert "--auto-align-en" not in cmd_used
        assert client.post("/api/jobs/abort").status_code == 200

        # Folder routing reflects actual episodes inside folder
        browse_res = client.get("/api/fs/browse", params={"path": str(season_dir)}).json()
        assert browse_res["folder_routing"] is not None
        assert browse_res["folder_routing"]["model_id"] != ""

        # Blank compare-frames paths return HTTP 400 (not 403)
        cmp_blank = client.post(
            "/api/preview/compare-frames",
            json={"source_path": "", "upscaled_path": "   ", "timestamp_sec": 1.0},
        )
        assert cmp_blank.status_code == 400

        # Frontend token checks for new UX/QA features
        html = client.get("/").text
        panels_js = client.get("/static/js/panels.js").text
        compare_js = client.get("/static/js/compare_stage.js").text
        fs_js = client.get("/static/js/fs_picker.js").text
        app_js = client.get("/static/js/app.js").text

        assert "btn-browse-opt-outdir" in html and "btn-browse-sub-outdir" in html
        assert "opt-preview-seconds" in html
        assert "fs-jump-input" in html and "fs-filter-input" in html
        assert "btn-wipe-src" in html and "compare-pan-mode" in html
        assert "QUEUE_CONFIG_STORAGE_KEY" in panels_js
        assert "validity.badInput" in panels_js and "validity.badInput" in compare_js
        assert "foldersOnly" in fs_js and "foldersOnly" in panels_js
        assert "resumeCheckpointNow" in panels_js and "resumeCheckpointNow" in app_js
        assert "/api/checkpoints/delete" in app_js
        assert "active_chunk_fraction" in app_js


def test_eleven_keys_natural_slot_order_across_put_and_overview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify 11-key .env preserves natural numeric slot order across PUT /api/subtrans/api-keys and overview."""
    monkeypatch.delenv("SUBTRANS_DISABLE_DOTENV", raising=False)
    for env_k in list(os.environ.keys()):
        if env_k.startswith("GEMINI_API_KEY"):
            monkeypatch.delenv(env_k, raising=False)

    root = _seed_workspace(tmp_path / "eleven_keys_ws")
    env_lines = [
        f"GEMINI_API_KEY_{idx}=AQ.Ab8RN6SlotKey{idx:02d}_{idx:04d}"
        for idx in range(1, 12)
    ]
    (root / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
    with TestClient(app) as client:
        ov_init = client.get("/api/subtrans/overview").json()
        assert ov_init["configured_keys_count"] == 11
        assert {s["slot"]: s["masked"] for s in ov_init["key_slots"]} == {
            1: "AQ.Ab8...0001",
            2: "AQ.Ab8...0002",
            3: "AQ.Ab8...0003",
        }

        r_put = client.put("/api/subtrans/api-keys", json={"key_2": "AIzaSyUpdatedSlotTwo2222"})
        assert r_put.status_code == 200
        put_data = r_put.json()
        assert put_data["configured_count"] == 11
        assert {s["slot"]: s["masked"] for s in put_data["key_slots"]} == {
            1: "AQ.Ab8...0001",
            2: "AIzaSy...2222",
            3: "AQ.Ab8...0003",
        }

        ov_after = client.get("/api/subtrans/overview").json()
        assert ov_after["configured_keys_count"] == 11
        assert {s["slot"]: s["masked"] for s in ov_after["key_slots"]} == {
            1: "AQ.Ab8...0001",
            2: "AIzaSy...2222",
            3: "AQ.Ab8...0003",
        }


def test_reviewer_fixes_symlink_checkpoint_guard_unicode_quotes_preview_start_and_subs_meta(tmp_path: Path):
    """Verify symlink delete rejection, Kanji/double-bracket checkpoints, quoted paths, preview_start_sec, and subs metadata."""
    root = _seed_workspace(tmp_path / "reviewer_qa_ws")
    kanji_ep = root / "anime_season" / "[[1080p]] 葬送のフリーレン 第01話.mkv"
    kanji_ep.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 512)

    cp_dir = root / "out" / ".temp_[[1080p]] 葬送のフリーレン 第01話"
    (cp_dir / "upscaled_segs").mkdir(parents=True, exist_ok=True)
    (cp_dir / "upscaled_segs" / "up_0000.mp4").write_bytes(b"0" * 16384)
    (cp_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_file": str(kanji_ep),
                "model_id": "animevideov3",
                "total_chunks": 3,
                "segment_time": 24,
                "subs_after_upscale": False,
                "subs_quality": "high",
                "prefer_whisper": True,
                "async_subs": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    spawned_cmds: List[List[str]] = []

    def capture_factory(cmd: List[str], **_kw: Any) -> FakePopenProcess:
        spawned_cmds.append(list(cmd))
        return FakePopenProcess(cmd, pid=91234, running=False, return_code=0)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, popen_factory=capture_factory)
    with TestClient(app) as client:
        # 1. Verify checkpoint subtitle metadata & total_chunks round-trip
        st = client.get("/api/status").json()
        cp = next(c for c in st["pipeline"]["checkpoints"] if c["name"] == cp_dir.name)
        assert cp["total_chunks"] == 3
        assert cp["subs_after_upscale"] is False
        assert cp["subs_quality"] == "high"
        assert cp["prefer_whisper"] is True
        assert cp["async_subs"] is False

        # 2. Verify symlink/junction pointing to real checkpoint is rejected BEFORE resolve() without deleting target
        sym_path = root / "out" / ".temp_symlink_attack"
        created_link = False
        try:
            os.symlink(str(cp_dir), str(sym_path), target_is_directory=True)
            created_link = True
        except OSError:
            if sys.platform == "win32":
                import subprocess as _sp
                r_mk = _sp.run(["cmd", "/c", "mklink", "/J", str(sym_path), str(cp_dir)], capture_output=True)
                created_link = r_mk.returncode == 0
        if created_link:
            res_sym = client.post("/api/checkpoints/delete", json={"path": str(sym_path)})
            assert res_sym.status_code == 400
            assert cp_dir.is_dir(), "Real checkpoint target must NOT be deleted through symlink/junction!"
            try:
                os.rmdir(sym_path) if sym_path.is_dir() and not sym_path.is_symlink() else sym_path.unlink()
            except OSError:
                pass

        # 3. Verify Windows Explorer 'Copy as path' wrapped quotes work in browse & schedule
        quoted_dir = f'"{kanji_ep.parent}"'
        r_browse = client.get("/api/fs/browse", params={"path": quoted_dir})
        assert r_browse.status_code == 200
        assert any("葬送のフリーレン" in e["name"] for e in r_browse.json()["entries"])

        quoted_file = f'"{kanji_ep}"'
        r_sched = client.post("/api/queue/preview-schedule", json={"inputs": [quoted_file]})
        assert r_sched.status_code == 200

        # 4. Verify preview_start_sec ('14:20' -> 860.0) forwards --preview-start 860.0
        spawned_cmds.clear()
        r_prev = client.post(
            "/api/jobs/preview",
            json={"video_path": str(kanji_ep), "seconds": 15, "preview_start_sec": "14:20"},
        )
        assert r_prev.status_code == 200
        prev_cmd = spawned_cmds[-1]
        assert "--preview-start" in prev_cmd
        assert float(prev_cmd[prev_cmd.index("--preview-start") + 1]) == 860.0

        # 5. Cleanly delete the Unicode/double-bracket checkpoint
        r_del_ok = client.post("/api/checkpoints/delete", json={"path": str(cp_dir)})
        assert r_del_ok.status_code == 200
        assert not cp_dir.exists()


def test_deep_external_symlink_ancestor_guard_strict_preview_start_and_stale_live_reset(tmp_path: Path):
    """Verify deep symlink ancestors outside workspace_root, strict MM:SS/HH:MM:SS validation, and dead-PID live reset."""
    from web.job_manager import parse_preview_start_value

    root = _seed_workspace(tmp_path / "deep_guard_ws")
    ep_file = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"

    # Real custom -o directory outside workspace_root nested 2+ levels below a directory target
    ext_real_root = tmp_path / "external_real_drive"
    ext_real_out = ext_real_root / "batch_01" / "season_02"
    ext_cp = ext_real_out / ".temp_[BD 1080p] Fate_Ep22"
    (ext_cp / "upscaled_segs").mkdir(parents=True, exist_ok=True)
    (ext_cp / "upscaled_segs" / "up_0000.mp4").write_bytes(b"1" * 16384)
    (ext_cp / "manifest.json").write_text(
        json.dumps({"source_file": str(ep_file), "model_id": "animevideov3", "total_chunks": 2}),
        encoding="utf-8",
    )

    spawned_cmds: List[List[str]] = []

    def capture_factory(cmd: List[str], **_kw: Any) -> FakePopenProcess:
        spawned_cmds.append(list(cmd))
        return FakePopenProcess(cmd, pid=92345, running=False, return_code=0)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, popen_factory=capture_factory)
    with TestClient(app) as client:
        # 1. Stale dead-PID live_status.json in 'probing' with 0 checkpoints resets live.stage and stage_detail
        (root / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": "StaleDeadRun",
                    "pid": 99999992,
                    "stage": "probing",
                    "stage_detail": "Đang kiểm tra video nguồn & môi trường...",
                    "active_chunk": 0,
                    "active_chunk_fraction": 0.45,
                    "eta_seconds": 120.0,
                }
            ),
            encoding="utf-8",
        )
        st_idle = client.get("/api/status").json()["pipeline"]
        assert st_idle["effective_stage"] == "idle"
        assert st_idle["live"]["stage"] == "idle"
        assert st_idle["live"]["last_stage"] == "probing"
        assert st_idle["live"]["active_chunk"] is None
        assert st_idle["live"]["active_chunk_fraction"] == 0.0
        assert st_idle["live"]["eta_seconds"] is None
        assert "Đang kiểm tra" not in st_idle["live"]["stage_detail"]
        (root / "logs" / "live_status.json").unlink(missing_ok=True)

        # Register ext_real_out in job history so ext_cp is a known external -o checkpoint
        r_dry = client.post(
            "/api/jobs/start",
            json={"inputs": [str(ep_file)], "output_dir": str(ext_real_out), "dry_run": True},
        )
        assert r_dry.status_code == 200

        # 2. Deep symlink/junction ancestor (3 levels above .temp_*) outside workspace_root MUST be rejected
        ext_link_container = tmp_path / "external_attack_mount"
        ext_link_container.mkdir(parents=True, exist_ok=True)
        deep_link_root = ext_link_container / "linked_drive"
        created_deep_link = False
        try:
            os.symlink(str(ext_real_root), str(deep_link_root), target_is_directory=True)
            created_deep_link = True
        except OSError:
            if sys.platform == "win32":
                import subprocess as _sp

                r_mk = _sp.run(
                    ["cmd", "/c", "mklink", "/J", str(deep_link_root), str(ext_real_root)],
                    capture_output=True,
                )
                created_deep_link = r_mk.returncode == 0
        assert created_deep_link, "Failed to create symlink or junction for deep ancestor checkpoint test"
        try:
            deep_attack_cp = deep_link_root / "batch_01" / "season_02" / ext_cp.name
            res_deep = client.post("/api/checkpoints/delete", json={"path": str(deep_attack_cp)})
            assert res_deep.status_code == 400
            assert ext_cp.is_dir(), "Real external checkpoint must NOT be deleted via deep junction ancestor!"

            dotdot_attack = f"{deep_link_root}{os.sep}..{os.sep}{ext_real_root.name}{os.sep}batch_01{os.sep}season_02{os.sep}{ext_cp.name}"
            res_dotdot = client.post("/api/checkpoints/delete", json={"path": dotdot_attack})
            assert res_dotdot.status_code == 400
            assert ext_cp.is_dir()
        finally:
            try:
                os.rmdir(deep_link_root) if deep_link_root.is_dir() and not deep_link_root.is_symlink() else deep_link_root.unlink()
            except OSError:
                pass

        # Quote-wrapped UNC path must be rejected before filesystem traversal
        res_unc_quoted = client.post(
            "/api/checkpoints/delete",
            json={"path": '"\\\\192.0.2.1\\share\\.temp_attack"'},
        )
        assert res_unc_quoted.status_code in (400, 422)

        # Legitimate direct external -o checkpoint deletes cleanly
        res_ext_ok = client.post("/api/checkpoints/delete", json={"path": str(ext_cp)})
        assert res_ext_ok.status_code == 200
        assert not ext_cp.exists()

        # 3. Strict preview_start_sec validation rejects SS >= 60, MM >= 60 in HH:MM:SS, and non-integer HH/MM
        for bad_ts in (
            "14:60",
            "14:75.5",
            "01:60:00",
            "01:02:60",
            "1.5:20",
            "1.5:02:03",
            "01:2.5:03",
            "-1:20",
            "14: 20",
            "25:00:00",
            True,
        ):
            r_bad = client.post(
                "/api/jobs/preview",
                json={"video_path": str(ep_file), "seconds": 10, "preview_start_sec": bad_ts},
            )
            assert r_bad.status_code == 422, f"Expected 422 for invalid preview_start_sec={bad_ts!r}, got {r_bad.status_code}"
            with pytest.raises(ValueError):
                parse_preview_start_value(bad_ts)

        # Valid 2-part movie minute >= 60 ('85:30' == 5130.0) and sub-second '14:20.375' == 860.375 succeed
        for valid_ts, expected_sec in (("85:30", 5130.0), ("14:20.375", 860.375), ("01:02:03.25", 3723.25)):
            spawned_cmds.clear()
            r_ok = client.post(
                "/api/jobs/preview",
                json={"video_path": str(ep_file), "seconds": 10, "preview_start_sec": valid_ts},
            )
            assert r_ok.status_code == 200
            cmd_p = spawned_cmds[-1]
            assert "--preview-start" in cmd_p
            assert float(cmd_p[cmd_p.index("--preview-start") + 1]) == pytest.approx(expected_sec)


def test_mixed_slash_unc_quoted_output_dirs_subtrans_secret_guard_and_post_delete_idle(tmp_path: Path):
    """Verify mixed-slash UNC blocks, quoted -o paths, subtrans .env guards, strict float preview_start, and post-delete idle reset."""
    import shutil
    import time
    from web.fs_browser import is_unc_path
    from web.sse_hub import scan_temp_checkpoints

    root = _seed_workspace(tmp_path / "hardened_web_ws")
    ep_file = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    srt_file = root / "anime_season" / "Frieren_02.ja.srt"
    custom_out = tmp_path / "Custom Output Space"
    custom_out.mkdir(parents=True, exist_ok=True)

    spawned_cmds: List[List[str]] = []

    next_running = {"value": False}

    def capture_factory(cmd: List[str], **_kw: Any) -> FakePopenProcess:
        spawned_cmds.append(list(cmd))
        is_run = next_running["value"]
        return FakePopenProcess(cmd, pid=93456, running=is_run, return_code=None if is_run else 0)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, popen_factory=capture_factory)
    with TestClient(app) as client:
        # 1. Mixed-slash, curly-quoted, double-quoted, and NT-namespace UNC strings must all be rejected
        for unc_variant in (
            "\\/192.0.2.1/share/.temp_attack",
            "/\\192.0.2.1\\share\\.temp_attack",
            '""\\\\192.0.2.1\\share\\.temp_attack""',
            "“\\\\192.0.2.1\\share\\.temp_attack”",
            "\\??\\UNC\\192.0.2.1\\share\\.temp_attack",
        ):
            assert is_unc_path(unc_variant) is True
            assert client.post("/api/checkpoints/delete", json={"path": unc_variant}).status_code in (400, 422)
            assert client.get("/api/fs/browse", params={"path": unc_variant}).status_code == 400

        # Windows drive-relative path ('E:sub\..\.temp_attack') must be rejected with HTTP 400
        if sys.platform == "win32":
            assert client.post("/api/checkpoints/delete", json={"path": "E:sub\\..\\.temp_attack"}).status_code == 400
            assert client.get("/api/fs/browse", params={"path": "E:sub\\.."}).status_code == 400

        # 2. Explorer 'Copy as path' quotes on output_dir (-o) and test-match work cleanly without literal quote artifacts
        ext_q_cp = custom_out / f".temp_{ep_file.stem}"
        (ext_q_cp / "upscaled_segs").mkdir(parents=True, exist_ok=True)
        (ext_q_cp / "upscaled_segs" / "up_0000.mp4").write_bytes(b"q" * 8192)
        (ext_q_cp / "manifest.json").write_text(
            json.dumps({"source_file": str(ep_file), "model_id": "animevideov3", "total_chunks": 4}),
            encoding="utf-8",
        )
        quoted_out = f'"{custom_out}"'
        r_sched = client.post(
            "/api/queue/preview-schedule",
            json={"inputs": [str(ep_file)], "output_dir": quoted_out},
        )
        assert r_sched.status_code == 200
        assert r_sched.json()["episodes"][0]["resumed_done_chunks"] == 1
        if sys.platform == "win32":
            import ctypes

            buf = ctypes.create_unicode_buffer(1024)
            n_ch = ctypes.windll.kernel32.GetShortPathNameW(str(custom_out), buf, 1024)
            short_parent = Path(buf.value) if n_ch > 0 and buf.value else custom_out
            cps_dedup = scan_temp_checkpoints(root, extra_dirs=[short_parent, custom_out])
            assert sum(1 for c in cps_dedup if c["name"] == ext_q_cp.name) == 1
        spawned_cmds.clear()
        r_start_q = client.post(
            "/api/jobs/start",
            json={"inputs": [str(ep_file)], "output_dir": f"“{custom_out}”", "dry_run": True},
        )
        assert r_start_q.status_code == 200
        cmd_q = spawned_cmds[-1]
        assert cmd_q[cmd_q.index("-o") + 1] == str(custom_out)
        assert any(c["name"] == ext_q_cp.name for c in client.get("/api/status").json()["pipeline"]["checkpoints"])
        shutil.rmtree(ext_q_cp)

        r_match_q = client.post("/api/config/test-match", json={"sample_path": f'"{ep_file}"'})
        assert r_match_q.status_code == 200
        assert r_match_q.json()["sample"] == str(ep_file)

        # 3. Subtrans endpoint blocks .env secret paths on output_dir and glossary with HTTP 403 and rejects bool sub_offset_ms
        assert (
            client.post(
                "/api/jobs/subtrans",
                json={"input_path": str(srt_file), "output_dir": ".env", "dry_run": True},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/jobs/subtrans",
                json={"input_path": str(srt_file), "glossary": '".env.production"', "dry_run": True},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/jobs/subtrans",
                json={"input_path": str(srt_file), "sub_offset_ms": True, "dry_run": True},
            ).status_code
            == 422
        )

        # 4. Plain-seconds preview_start_sec rejects underscore/scientific/signed/Unicode float quirks matching panels.js
        for bad_plain in ("1_00", "1e2", "+10", ".5", "١٢"):
            r_bad_p = client.post(
                "/api/jobs/preview",
                json={"video_path": str(ep_file), "seconds": 10, "preview_start_sec": bad_plain},
            )
            assert r_bad_p.status_code == 422, f"Expected 422 for {bad_plain!r}, got {r_bad_p.status_code}"

        # 5. Reconcile paused checkpoint completed_chunks and transition cleanly to idle after deleting last checkpoint
        time.sleep(0.01)
        cp_dir = root / "out" / f".temp_{ep_file.stem}"
        (cp_dir / "upscaled_segs").mkdir(parents=True, exist_ok=True)
        (cp_dir / "upscaled_segs" / "up_0000.mp4").write_bytes(b"0" * 8192)
        (cp_dir / "upscaled_segs" / "up_0001.mp4").write_bytes(b"1" * 8192)
        (cp_dir / "manifest.json").write_text(
            json.dumps({"source_file": str(ep_file), "model_id": "animevideov3", "total_chunks": 6}),
            encoding="utf-8",
        )
        # First verify sub-second CLI live_status immediately overrides a just-finished dry_run via finished_epoch
        (root / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": ep_file.stem,
                    "input_path": str(ep_file),
                    "pid": 99999991,
                    "stage": "paused",
                    "stage_detail": "Đã dừng êm tại phân đoạn 1/6.",
                    "completed_chunks": 1,
                    "total_chunks": 6,
                }
            ),
            encoding="utf-8",
        )
        st_cli_paused = client.get("/api/status").json()["pipeline"]
        assert st_cli_paused["effective_stage"] == "paused"
        assert st_cli_paused["live"]["completed_chunks"] == 2

        # Next launch & abort a Web upscale job, verify completed_chunks reconciles to 2, then delete checkpoint -> idle
        next_running["value"] = True
        r_live_job = client.post("/api/jobs/start", json={"inputs": [str(ep_file)], "subs_after_upscale": False})
        assert r_live_job.status_code == 200
        live_jid = r_live_job.json()["job"]["job_id"]
        (root / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": ep_file.stem,
                    "input_path": str(ep_file),
                    "pid": 93456,
                    "job_id": live_jid,
                    "stage": "upscaling",
                    "stage_detail": "Đang suy luận phân đoạn 2/6...",
                    "completed_chunks": 1,
                    "total_chunks": 6,
                }
            ),
            encoding="utf-8",
        )
        assert client.post("/api/jobs/abort").status_code == 200
        next_running["value"] = False

        st_paused = client.get("/api/status").json()["pipeline"]
        assert st_paused["effective_stage"] == "paused"
        assert st_paused["live"]["completed_chunks"] == 2

        r_del_last = client.post("/api/checkpoints/delete", json={"path": str(cp_dir)})
        assert r_del_last.status_code == 200
        st_after_del = client.get("/api/status").json()["pipeline"]
        assert st_after_del["effective_stage"] == "idle"
        assert st_after_del["live"]["completed_chunks"] == 0
        assert st_after_del["checkpoints"] == []


def test_m1_p0_srgb_compare_parity_video_tail_clamp_outdir_hygiene_subtrans_batch_and_checkpoint_isolation(
    tmp_path: Path,
):
    """Verify M1 P0 fixes: 8-bit sRGB PNG parity, orphan cleanup, video-stream tail clamp, output_dir hygiene, subtrans batch routing, and isolated checkpoint resume."""
    import struct
    import cv2
    import numpy as np
    import upscale
    from web.app import _resolve_ffmpeg_binary
    from web.sse_hub import scan_temp_checkpoints

    root = _seed_workspace(tmp_path / "m1_p0_ws")
    ff_bin = _resolve_ffmpeg_binary()

    # 1. Real FFmpeg 8-bit untagged vs 10-bit bt709 dark frame parity + longer-audio MKV tail clamp
    src_vid = root / "anime_season" / "dark_ep.mkv"
    up_vid = root / "out" / "dark_ep_preview_10s_4K.mkv"
    r_src = subprocess.run(
        [
            ff_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x161820:s=160x90:r=24:d=1.0,format=yuv420p,setparams=range=tv:colorspace=unknown:color_primaries=unknown:color_trc=unknown",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src_vid),
        ],
        capture_output=True,
        timeout=25.0,
        check=False,
    )
    r_up = subprocess.run(
        [
            ff_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x161820:s=320x180:r=24:d=1.0,format=yuv420p10le,setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo:d=2.4",
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-c:a",
            "aac",
            str(up_vid),
        ],
        capture_output=True,
        timeout=25.0,
        check=False,
    )
    if r_src.returncode == 0 and r_up.returncode == 0:
        app_real = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)
        with TestClient(app_real) as client_real:
            # Clamping timestamp_sec=9999 uses video stream duration (~1.0s), avoiding empty audio tail at 2.3s
            r_cmp = client_real.post(
                "/api/preview/compare-frames",
                json={
                    "source_path": str(src_vid),
                    "upscaled_path": str(up_vid),
                    "timestamp_sec": 9999.0,
                    "source_offset_sec": 0.0,
                },
            )
            assert r_cmp.status_code == 200, r_cmp.text
            cmp_data = r_cmp.json()
            assert cmp_data["upscaled_meta"]["duration"] <= 1.2
            assert cmp_data["upscaled_meta"]["fps"] >= 23.0

            b_src = client_real.get(cmp_data["source_frame_url"]).content
            b_up = client_real.get(cmp_data["upscaled_frame_url"]).content
            # Both PNGs must be 8-bit RGB24 (IHDR bit_depth=8, color_type=2) with no cICP mismatch
            assert struct.unpack(">BB", b_src[24:26]) == (8, 2)
            assert struct.unpack(">BB", b_up[24:26]) == (8, 2)
            assert b"sRGB" in b_src and b"sRGB" in b_up
            assert b"cICP" not in b_up

            arr_src = cv2.imdecode(np.frombuffer(b_src, dtype=np.uint8), cv2.IMREAD_COLOR)
            arr_up = cv2.imdecode(np.frombuffer(b_up, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert arr_src is not None and arr_up is not None
            assert abs(float(arr_src.mean()) - float(arr_up.mean())) < 1.0

    # 2. Orphan src_*.png cleanup when upscaled frame extraction raises RuntimeError
    def _fail_on_up(vpath: Path, _ts: float, out_png: Path) -> None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        if out_png.name.startswith("up_"):
            raise RuntimeError("Simulated upscaled frame decode error")
        out_png.write_bytes(PNG_1X1_BYTES)

    app_orphan = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, frame_extract_fn=_fail_on_up)
    with TestClient(app_orphan) as client_orphan:
        ep_src = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
        ep_up = root / "out" / "Fate_Ep22_4K.mkv"
        ep_up.write_bytes(b"\x00" * 512)
        r_fail_cmp = client_orphan.post(
            "/api/preview/compare-frames",
            json={"source_path": str(ep_src), "upscaled_path": str(ep_up), "timestamp_sec": 1.0},
        )
        assert r_fail_cmp.status_code == 422
        assert list((root / "scratch" / "web_compare").glob("src_*.png")) == []

    # 3. Preview seconds range (3..60), output_dir validation, and subtrans batch routing
    spawned_cmds: List[List[str]] = []
    app_jobs = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        popen_factory=lambda cmd, **_kw: spawned_cmds.append(list(cmd)) or FakePopenProcess(cmd, running=False),
    )
    ep1 = root / "anime_season" / "[SubsPlease] Frieren - 02 (1080p).mkv"
    ep2 = root / "anime_season" / "[SubsPlease] Frieren - 10 (1080p).mkv"
    existing_mkv = root / "out" / "existing_file_4K.mkv"
    existing_mkv.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 256)

    with TestClient(app_jobs) as client_jobs:
        for bad_sec in (0, 2, 61, 999):
            assert (
                client_jobs.post("/api/jobs/preview", json={"video_path": str(ep1), "seconds": bad_sec}).status_code
                == 422
            )
        assert client_jobs.post("/api/jobs/preview", json={"video_path": str(ep1), "seconds": 3}).status_code == 200
        assert client_jobs.post("/api/jobs/preview", json={"video_path": str(ep1), "seconds": 60}).status_code == 200

        bad_out_candidates = [str(existing_mkv), str(root / "scratch" / "bad<>dir")]
        if sys.platform == "win32":
            unmounted = next(
                (f"{ch}:\\nonexistent_upscaler_out" for ch in "ZYXWVUTSRQPONM" if not Path(f"{ch}:\\").exists()),
                None,
            )
            if unmounted:
                bad_out_candidates.append(unmounted)

        for bad_out in bad_out_candidates:
            assert (
                client_jobs.post(
                    "/api/jobs/start",
                    json={"inputs": [str(ep1)], "output_dir": bad_out, "dry_run": True},
                ).status_code
                == 400
            )
            assert (
                client_jobs.post(
                    "/api/jobs/preview",
                    json={"video_path": str(ep1), "output_dir": bad_out},
                ).status_code
                == 400
            )

        # CLI upscale.py _validate_cli_output_path rejects --queue with single .mkv -o and bad chars
        with pytest.raises(SystemExit) as exc_q:
            upscale._validate_cli_output_path(str(existing_mkv), multi_queue=True)
        assert exc_q.value.code == 1
        if sys.platform == "win32":
            with pytest.raises(SystemExit) as exc_ch:
                upscale._validate_cli_output_path(str(root / "bad<>out"), multi_queue=False)
            assert exc_ch.value.code == 1

        # Subtrans directory of .srt files routes to `subtrans.py batch <dir>`
        subs_dir = root / "subs_only_folder"
        subs_dir.mkdir()
        (subs_dir / "Ep01.ja.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nA\n", encoding="utf-8")
        (subs_dir / "Ep02.ja.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nB\n", encoding="utf-8")
        spawned_cmds.clear()
        r_batch = client_jobs.post(
            "/api/jobs/subtrans",
            json={"input_path": str(subs_dir), "subs_quality": "high", "dry_run": True},
        )
        assert r_batch.status_code == 200
        sub_cmd = spawned_cmds[-1]
        assert any(part.endswith("subtrans.py") for part in sub_cmd)
        assert "batch" in sub_cmd
        assert str(subs_dir.resolve()) in sub_cmd
        assert "--subs-only" not in sub_cmd

        # Mixed video + .srt in JobManager.start_subtrans_job raises ValueError
        jm_mix = app_jobs.state.job_manager
        with pytest.raises(ValueError, match="trộn lẫn"):
            jm_mix.start_subtrans_job(
                {"input_path": str(ep1), "inputs": [str(ep1), str(subs_dir / "Ep01.ja.srt")], "dry_run": True}
            )

        # 4. Checkpoint model_requested="" preservation and isolated resumeCheckpointNow contract
        cp_auto = root / "out" / f".temp_{ep2.stem}"
        (cp_auto / "upscaled_segs").mkdir(parents=True)
        (cp_auto / "upscaled_segs" / "up_0000.mp4").write_bytes(b"\x00" * 256)
        (cp_auto / "manifest.json").write_text(
            json.dumps(
                {
                    "source_file": str(ep2),
                    "output_dir": str(root / "out"),
                    "model_id": "animejanai-hd-v3-ultracompact",
                    "model_requested": "",
                    "dedup_policy": "dis-auto",
                    "dedup_policy_requested": "",
                    "resolution_source": "shows.yaml (*[SubsPlease]*)",
                    "total_chunks": 4,
                }
            ),
            encoding="utf-8",
        )
        cps = scan_temp_checkpoints(root)
        matched_cp = next(c for c in cps if c["episode_name"] == ep2.stem)
        assert matched_cp["model_id"] == "animejanai-hd-v3-ultracompact"
        assert matched_cp["model_requested"] == ""
        assert matched_cp["dedup_policy_requested"] == ""

        panels_js = client_jobs.get("/static/js/panels.js").text
        resume_idx = panels_js.index("async resumeCheckpointNow(cp)")
        resume_body = panels_js[resume_idx:]
        assert "applyCheckpointOptions" not in resume_body
        assert "model: cpModelReq || null" in resume_body


def test_m2_p1_dynamic_keys_queue_summary_resume_init_abort_teardown_and_zero_chunk_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify M2 P1 features: 11+ dynamic .env key slots, queue_summary, non-zero resume init, zero CLI flash on abort, and 0-chunk queue cleanup."""
    import time
    import psutil
    from web.sse_hub import TelemetryHub

    monkeypatch.delenv("SUBTRANS_DISABLE_DOTENV", raising=False)
    for env_k in list(os.environ.keys()):
        if env_k.startswith("GEMINI_API_KEY"):
            monkeypatch.delenv(env_k, raising=False)

    root = _seed_workspace(tmp_path / "m2_p1_ws")
    env_lines = [f"GEMINI_API_KEY_{idx}=AQ.Ab8RN6DynSlot{idx:02d}_{idx:04d}" for idx in range(1, 12)]
    (root / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    next_running = {"val": True}
    app = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        popen_factory=lambda cmd, **_kw: FakePopenProcess(cmd, pid=88400, running=next_running["val"]),
    )
    ep1 = root / "anime_season" / "[SubsPlease] Frieren - 02 (1080p).mkv"
    ep2 = root / "anime_season" / "[SubsPlease] Frieren - 10 (1080p).mkv"
    ep3 = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"

    with TestClient(app) as client:
        # 1. Dynamic 1..11+ key slots via all_key_slots and PUT slots map
        ov = client.get("/api/subtrans/overview").json()
        assert len(ov["key_slots"]) == 3
        assert len(ov["all_key_slots"]) == 11
        assert ov["all_key_slots"][10]["slot"] == 11
        assert ov["all_key_slots"][10]["masked"] == "AQ.Ab8...0011"

        r_upd = client.put(
            "/api/subtrans/api-keys",
            json={"slots": {"5": "", "12": "AIzaSyAddedSlotTwelve1212"}},
        )
        assert r_upd.status_code == 200
        upd_json = r_upd.json()
        assert upd_json["configured_count"] == 11  # 11 - 1 cleared + 1 added = 11
        by_slot = {s["slot"]: s for s in upd_json["all_key_slots"]}
        assert by_slot[5]["configured"] is False
        assert by_slot[12]["configured"] is True
        assert by_slot[12]["masked"] == "AIzaSy...1212"

        for bad_slots in ({"0": "AIzaBad"}, {"33": "AIzaBad"}, {"8": "AIza\nCR"}):
            assert client.put("/api/subtrans/api-keys", json={"slots": bad_slots}).status_code == 422

        # 2. Resume initializing preserves checkpoint chunks (no 40% -> 0% -> 40% drop) & queue_summary
        cp_ep1 = root / "out" / f".temp_{ep1.stem}"
        (cp_ep1 / "upscaled_segs").mkdir(parents=True)
        (cp_ep1 / "upscaled_segs" / "up_0000.mp4").write_bytes(b"\x00" * 256)
        (cp_ep1 / "upscaled_segs" / "up_0001.mp4").write_bytes(b"\x00" * 256)
        (cp_ep1 / "manifest.json").write_text(
            json.dumps({"source_file": str(ep1), "total_chunks": 5, "model_id": "animejanai-hd-v3-ultracompact"}),
            encoding="utf-8",
        )
        cp_ep2_zero = root / "out" / f".temp_{ep2.stem}"
        (cp_ep2_zero / "raw_segs").mkdir(parents=True)
        (cp_ep2_zero / "raw_segs" / "seg_0000.mp4").write_bytes(b"\x00" * 256)

        r_q = client.post(
            "/api/jobs/start",
            json={"inputs": [str(ep1), str(ep2), str(ep3)], "subs_after_upscale": False},
        )
        assert r_q.status_code == 200
        q_job = r_q.json()["job"]

        # While initializing (before RunLogger writes live_status.json for q_job), completed_chunks stays 2/5
        st_init = client.get("/api/status").json()["pipeline"]
        assert st_init["effective_stage"] == "initializing"
        assert st_init["live"]["completed_chunks"] == 2
        assert st_init["live"]["total_chunks"] == 5
        assert st_init["live"]["resumed_chunks"] == [0, 1]
        assert st_init["queue_summary"] == {
            "total_items": 3,
            "current_index": 1,
            "completed_items": 0,
            "current_name": ep1.name,
            "remaining_count": 2,
            "remaining_names": [ep2.name, ep3.name],
        }

        # When RunLogger advances to ep2, queue_summary updates to 2/3
        (root / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": ep2.stem,
                    "input_path": str(ep2),
                    "pid": 88400,
                    "job_id": q_job["job_id"],
                    "stage": "upscaling",
                    "completed_chunks": 0,
                    "total_chunks": 6,
                }
            ),
            encoding="utf-8",
        )
        st_ep2 = client.get("/api/status").json()["pipeline"]
        assert st_ep2["queue_summary"]["current_index"] == 2
        assert st_ep2["queue_summary"]["completed_items"] == 1
        assert st_ep2["queue_summary"]["remaining_names"] == [ep3.name]

        # 3. Aborting multi-episode queue cleans 0-chunk .temp_ep2 while preserving 2-chunk .temp_ep1
        r_abort = client.post("/api/jobs/abort")
        assert r_abort.status_code == 200
        assert cp_ep1.is_dir()
        assert not cp_ep2_zero.exists()

        # 4. Dying child PID from aborted Web Job never flashes active_source == "cli"
        dying_pid = 88999
        monkeypatch.setattr(psutil, "pid_exists", lambda p: p == dying_pid)
        (root / ".gpu_active.lock").write_text(f"{dying_pid}\n", encoding="utf-8")
        (root / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": ep1.stem,
                    "input_path": str(ep1),
                    "pid": dying_pid,
                    "job_id": q_job["job_id"],
                    "stage": "upscaling",
                    "completed_chunks": 2,
                    "total_chunks": 5,
                }
            ),
            encoding="utf-8",
        )
        try:
            hub_dying = TelemetryHub(
                root,
                gpu_query_fn=_idle_gpu_metrics,
                job_state_provider=lambda: {
                    "status": "idle",
                    "active_job": None,
                    "history": [{**q_job, "status": "aborted", "finished_epoch": time.time()}],
                    "aborting": False,
                    "aborted_pids": [88400, dying_pid],
                },
            )
            pipe_dying = hub_dying.read_pipeline_state()
            assert pipe_dying["active_source"] == "idle"
            assert pipe_dying["effective_stage"] == "paused"
            assert "giải phóng VRAM" in pipe_dying["live"]["stage_detail"]
        finally:
            (root / ".gpu_active.lock").unlink(missing_ok=True)

        # 5. Frontend M2 UI contracts
        html = client.get("/").text
        app_js = client.get("/static/js/app.js").text
        compare_js = client.get("/static/js/compare_stage.js").text
        fs_js = client.get("/static/js/fs_picker.js").text
        panels_js = client.get("/static/js/panels.js").text

        assert 'id="queue-progress-strip"' in html
        assert 'id="btn-compare-prev-frame"' in html and 'id="btn-compare-next-frame"' in html
        assert 'id="compare-mismatch-warning"' in html
        assert 'id="btn-add-api-key-slot"' in html
        assert "⏳ Chờ tới lượt" in app_js
        assert "TRT10 FP16" in app_js and "PyTorch CUDA" in app_js
        assert "stepCompareFrame" in compare_js
        assert "this.currentPath !== prevPath" in fs_js
        assert "Xác nhận xóa?" in panels_js


def test_m3_p2_trt_sidecar_build_stage_b_flags_glossary_editor_pending_delete_and_clean_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify M3 TRT .sha256 sidecar & build-trt, Stage B CLI parity, pointer-safe glossary editor, pending_subs delete, --format ass, and --clean-temp."""
    import hashlib
    import psutil

    root = _seed_workspace(tmp_path / "m3_p2_ws")
    ep_file = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    srt_file = root / "anime_season" / "Frieren_02.ja.srt"

    # Seed subtrans/glossary.yaml + base.yaml pointer + canonical & alias series glossaries
    (root / "subtrans").mkdir(parents=True, exist_ok=True)
    (root / "subtrans" / "glossary.yaml").write_text(
        "version: 2\nterms:\n  - ja: 魔法\n    vi: ma thuật\n",
        encoding="utf-8",
    )
    g_dir = root / "glossaries"
    g_dir.mkdir(parents=True, exist_ok=True)
    (g_dir / "base.yaml").write_text("../subtrans/glossary.yaml\n", encoding="utf-8")
    (g_dir / "glossary.frieren.yaml").write_text(
        "version: 2\nseries: Frieren\nterms:\n  - ja: 勇者\n    vi: Dũng giả\n",
        encoding="utf-8",
    )
    (g_dir / "glossary.aliasfrieren.yaml").write_text(
        "glossary.frieren.yaml\n", encoding="utf-8", newline="\n"
    )

    # Seed state/pending_subs.json with 2 pending items
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "pending_subs.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-11-24T17:00:00",
                "items": [
                    {
                        "base_name": "Frieren_Ep01",
                        "video_path": str(root / "out" / "Frieren_Ep01_4K.mkv"),
                        "subs_quality": "normal",
                    },
                    {
                        "base_name": "Fate_Ep22",
                        "video_path": str(root / "out" / "Fate_Ep22_4K.mkv"),
                        "subs_quality": "high",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    spawned_cmds: List[List[str]] = []

    def capture_factory(cmd: List[str], **_kw: Any) -> FakePopenProcess:
        spawned_cmds.append(list(cmd))
        return FakePopenProcess(cmd, pid=77123, running=False, return_code=0)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, popen_factory=capture_factory)
    with TestClient(app) as client:
        # 1. TRT .engine without .engine.sha256 is flagged stale (trt_cached=False, trt_ready=False)
        trt_dir = root / ".cache" / "trt_engines"
        trt_dir.mkdir(parents=True, exist_ok=True)
        eng_file = trt_dir / "animevideov3_aa2x_RTX3060_trt10_1080x1920_b1-2.engine"
        eng_bytes = b"TRT10_FP16_SERIALIZED_PLAN_" * 64
        eng_file.write_bytes(eng_bytes)

        ms_stale = client.get("/api/config/models-and-shows").json()
        av3_stale = next(m for m in ms_stale["models"] if m["id"] == "animevideov3")
        assert av3_stale["trt_cached"] is False
        assert eng_file.name in av3_stale["stale_trt_engines"]

        sched_stale = client.post(
            "/api/queue/preview-schedule",
            json={"inputs": [str(ep_file)], "model": "animevideov3", "use_tensorrt": True},
        ).json()
        assert sched_stale["episodes"][0]["trt_ready"] is False

        # Sidecar missing weights_sha256 is still rejected when model has registered sha256 (build_trt_engine parity)
        eng_digest = hashlib.sha256(eng_bytes).hexdigest()
        sidecar_file = Path(f"{eng_file}.sha256")
        sidecar_file.write_text(
            json.dumps({"engine_sha256": eng_digest}),
            encoding="utf-8",
        )
        av3_no_wsha = next(
            m for m in client.get("/api/config/models-and-shows").json()["models"] if m["id"] == "animevideov3"
        )
        assert av3_no_wsha["trt_cached"] is False

        # Write matching .engine.sha256 sidecar with weights_sha256 -> trt_cached=True & trt_ready=True
        sidecar_file.write_text(
            json.dumps(
                {
                    "engine_sha256": eng_digest,
                    "weights_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                }
            ),
            encoding="utf-8",
        )
        ms_valid = client.get("/api/config/models-and-shows").json()
        av3_valid = next(m for m in ms_valid["models"] if m["id"] == "animevideov3")
        assert av3_valid["trt_cached"] is True
        assert av3_valid["stale_trt_engines"] == []

        sched_ready = client.post(
            "/api/queue/preview-schedule",
            json={"inputs": [str(ep_file)], "model": "animevideov3", "use_tensorrt": True},
        ).json()
        assert sched_ready["episodes"][0]["trt_ready"] is True

        # Corrupt .engine bytes -> sidecar mismatch caught
        eng_file.write_bytes(eng_bytes + b"CORRUPT")
        av3_corrupt = next(
            m for m in client.get("/api/config/models-and-shows").json()["models"] if m["id"] == "animevideov3"
        )
        assert av3_corrupt["trt_cached"] is False
        assert eng_file.name in av3_corrupt["stale_trt_engines"]

        # POST /api/jobs/build-trt rejects unknown model and spawns src/build_trt_engine.py for valid model
        assert client.post("/api/jobs/build-trt", json={"model_id": "nonexistent_sr"}).status_code == 400
        spawned_cmds.clear()
        r_trt = client.post("/api/jobs/build-trt", json={"model_id": "animevideov3", "force": True})
        assert r_trt.status_code == 200
        assert any("build_trt_engine.py" in part for part in spawned_cmds[-1])
        assert "--model" in spawned_cmds[-1] and "animevideov3" in spawned_cmds[-1]
        assert "--force" in spawned_cmds[-1]

        # 2. Stage B CLI flags forwarded on POST /api/jobs/start and secret/enum guards enforced
        assert (
            client.post(
                "/api/jobs/start",
                json={"inputs": [str(ep_file)], "glossary": ".env", "dry_run": True},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/jobs/start",
                json={"inputs": [str(ep_file)], "sound_cues": "invalid_mode", "dry_run": True},
            ).status_code
            == 422
        )
        spawned_cmds.clear()
        g_frieren_path = str(g_dir / "glossary.frieren.yaml")
        r_stage_b = client.post(
            "/api/jobs/start",
            json={
                "inputs": [str(ep_file)],
                "subs_after_upscale": True,
                "subs_quality": "high",
                "glossary": g_frieren_path,
                "subs_upgrade": True,
                "no_embed": True,
                "sound_cues": "translate",
                "dry_run": True,
            },
        )
        assert r_stage_b.status_code == 200
        cmd_sb = spawned_cmds[-1]
        assert cmd_sb[cmd_sb.index("--glossary") + 1] == g_frieren_path
        assert "--subs-upgrade" in cmd_sb
        assert "--no-embed" in cmd_sb
        assert cmd_sb[cmd_sb.index("--sound-cues") + 1] == "translate"

        # 3. Glossary Editor: base.yaml read-only (403 on PUT), traversal blocked, pointer-safe LF write on alias
        r_base = client.get("/api/subtrans/glossary", params={"name": "base.yaml"})
        assert r_base.status_code == 200
        assert r_base.json()["read_only"] is True
        assert r_base.json()["is_pointer"] is True
        assert r_base.json()["resolved_name"] == "glossary.yaml"
        assert (
            client.put(
                "/api/subtrans/glossary",
                json={"name": "base.yaml", "yaml_content": "version: 2\nterms: []\n"},
            ).status_code
            == 403
        )
        for bad_gname in ("../.env", "..\\models\\models.yaml", "sub/nested.yaml"):
            assert client.get("/api/subtrans/glossary", params={"name": bad_gname}).status_code in (400, 403)

        # Invalid YAML / missing 'vi' in terms rejected with 400
        assert (
            client.put(
                "/api/subtrans/glossary",
                json={"name": "glossary.frieren.yaml", "yaml_content": "terms:\n  - ja: 勇者\n"},
            ).status_code
            == 400
        )

        # Saving through 1-line alias pointer updates canonical target with LF newlines and keeps pointer intact
        assert (
            client.put(
                "/api/subtrans/glossary",
                json={
                    "name": "glossary.aliasfrieren.yaml",
                    "yaml_content": "characters:\n  フリーレン:\n    name_vi: Frieren\n",
                },
            ).status_code
            == 400
        )
        r_save_alias = client.put(
            "/api/subtrans/glossary",
            json={
                "name": "glossary.aliasfrieren.yaml",
                "yaml_content": (
                    "meta:\r\n  schema: 2\r\n  pair: ja-vi\r\n"
                    "characters:\r\n  - name_ja: フリーレン\r\n    name_vi: Frieren\r\n"
                    "terms:\r\n  - ja: 勇者\r\n    vi: Dũng giả\r\n  - ja: 魔族\r\n    vi: Ma tộc\r\n"
                ),
            },
        )
        assert r_save_alias.status_code == 200
        save_payload = r_save_alias.json()
        assert save_payload["is_pointer"] is True
        assert save_payload["resolved_name"] == "glossary.frieren.yaml"
        assert save_payload["terms_count"] == 2
        assert save_payload["characters_count"] == 1
        assert (g_dir / "glossary.aliasfrieren.yaml").read_bytes() == b"glossary.frieren.yaml\n"
        saved_canon_bytes = (g_dir / "glossary.frieren.yaml").read_bytes()
        assert b"\r\n" not in saved_canon_bytes
        assert "Ma tộc".encode("utf-8") in saved_canon_bytes

        # 4. Pending Subs DELETE & standalone subtrans --format ass
        assert client.delete("/api/subtrans/pending", params={"target": "Unknown_Ep99"}).status_code == 404
        r_del1 = client.delete("/api/subtrans/pending", params={"target": "Frieren_Ep01"})
        assert r_del1.status_code == 200
        assert r_del1.json()["remaining_count"] == 1
        ov_after_del = client.get("/api/subtrans/overview").json()
        assert len(ov_after_del["pending_subs"]) == 1
        assert ov_after_del["pending_subs"][0]["base_name"] == "Fate_Ep22"

        assert (
            client.post(
                "/api/jobs/subtrans",
                json={"input_path": str(srt_file), "output_format": "vtt", "dry_run": True},
            ).status_code
            == 422
        )
        spawned_cmds.clear()
        r_ass = client.post(
            "/api/jobs/subtrans",
            json={"input_path": str(srt_file), "output_format": "ass", "dry_run": True},
        )
        assert r_ass.status_code == 200
        assert spawned_cmds[-1][spawned_cmds[-1].index("--format") + 1] == "ass"

        # 5. Global Temp Cleanup refuses when GPU lock is active (409) and runs upscale.py --clean-temp when idle
        live_gpu_pid = 66554
        monkeypatch.setattr(psutil, "pid_exists", lambda p: p == live_gpu_pid)
        (root / ".gpu_active.lock").write_text(f"{live_gpu_pid}\n", encoding="utf-8")
        try:
            assert client.post("/api/jobs/clean-temp", json={}).status_code == 409
        finally:
            (root / ".gpu_active.lock").unlink(missing_ok=True)

        spawned_cmds.clear()
        r_clean = client.post("/api/jobs/clean-temp", json={})
        assert r_clean.status_code == 200
        assert "--clean-temp" in spawned_cmds[-1]

        # 6. Frontend M3 DOM & JS contracts
        html = client.get("/").text
        panels_js = client.get("/static/js/panels.js").text
        assert 'id="opt-glossary"' in html and 'id="opt-sound-cues"' in html
        assert 'id="opt-subs-upgrade"' in html and 'id="opt-no-embed"' in html
        assert 'id="btn-open-glossary-editor"' in html and 'id="sub-format-select"' in html
        assert 'id="pending-subs-box"' in html and 'id="btn-clean-temp"' in html
        assert 'id="glossary-editor-dialog"' in html and 'closedby="any"' in html
        assert "/api/jobs/build-trt" in panels_js
        assert "/api/jobs/clean-temp" in panels_js
        assert "/api/subtrans/glossary" in panels_js
        assert "/api/subtrans/pending" in panels_js


def test_review_regressions_glossary_size_trt_blended_hw_resume_probe_key_clear_subtrans_files_pending_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify fixes for glossary size_kb/schema/pointer bounds, blended TRT & resolution readiness, probing resume chunks, Slot 2 clear, explicit subtrans --file batching, and path-scoped pending_subs delete."""
    import hashlib
    from web.app import _apply_subtrans_key_edits, _resolve_ffmpeg_binary

    monkeypatch.delenv("SUBTRANS_DISABLE_DOTENV", raising=False)
    for env_k in list(os.environ.keys()):
        if env_k.startswith("GEMINI_API_KEY"):
            monkeypatch.delenv(env_k, raising=False)

    root = _seed_workspace(tmp_path / "review_reg_ws")
    # Add UpCunet2x entries to models.yaml to verify exact --grain-restore blended _ps2x_<8hex>_ digest verification
    from web.fs_browser import compute_upcunet2x_blended_sha256_stdlib

    cons_sha = "b8ae522511111111222222223333333344444444555555556666666677777777"
    dn3x_sha = "8157aa5e8888888899999999aaaaaaaabbbbbbbbccccccccddddddddeeeeeeee"
    models_yaml = root / "models" / "models.yaml"
    models_yaml.write_text(
        models_yaml.read_text(encoding="utf-8")
        + f"""  realcugan-pro-conservative-2x:
    name: Real-CUGAN Pro Conservative 2x
    arch:
      type: UpCunet2x
      upscale: 2
    downsample: none
    weights_path: models_bench/cugan_pro_cons_2x.pth
    sha256: "{cons_sha}"
    license: MIT
    default: false
  realcugan-pro-denoise3x-2x:
    name: Real-CUGAN Pro Denoise3x 2x
    arch:
      type: UpCunet2x
      upscale: 2
    downsample: none
    weights_path: models_bench/cugan_pro_dn3x_2x.pth
    sha256: "{dn3x_sha}"
    license: MIT
    default: false
""",
        encoding="utf-8",
    )

    # 1. Glossary overview includes numeric size_kb, enforces pointer bounds, and rejects malformed speaks_to / address_rules
    (root / "subtrans").mkdir(parents=True, exist_ok=True)
    (root / "subtrans" / "glossary.yaml").write_text("terms:\n  - ja: 魔法\n    vi: ma thuật\n", encoding="utf-8")
    g_dir = root / "glossaries"
    (g_dir / "base.yaml").write_text("../subtrans/glossary.yaml\n", encoding="utf-8")
    (g_dir / "glossary.frieren.yaml").write_text("terms:\n  - ja: 勇者\n    vi: Dũng giả\n" * 40, encoding="utf-8")
    (g_dir / "glossary.escape.yaml").write_text("../shows.yaml\n", encoding="utf-8")

    # 2. Clearing Slot 2 when GEMINI_API_KEY != GEMINI_API_KEY_1 preserves Slot 3
    upd_slot2 = _apply_subtrans_key_edits(
        {"GEMINI_API_KEY": "keyA_1111", "GEMINI_API_KEY_1": "keyB_2222", "GEMINI_API_KEY_2": "keyC_3333"},
        None,
        None,
        None,
        slots={2: ""},
    )
    assert upd_slot2["GEMINI_API_KEY_1"] == "keyA_1111"
    assert upd_slot2["GEMINI_API_KEY_2"] == ""
    assert upd_slot2["GEMINI_API_KEY_3"] == "keyC_3333"

    spawned_cmds: List[List[str]] = []
    next_running = {"val": False}
    app = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        popen_factory=lambda cmd, **_kw: spawned_cmds.append(list(cmd))
        or FakePopenProcess(cmd, pid=99112, running=next_running["val"], return_code=0),
    )

    with TestClient(app) as client:
        ov = client.get("/api/subtrans/overview").json()
        by_gname = {g["name"]: g for g in ov["glossaries"]}
        assert isinstance(by_gname["base.yaml"]["size_kb"], (int, float))
        assert by_gname["base.yaml"]["read_only"] is True
        assert by_gname["glossary.frieren.yaml"]["size_kb"] >= 1.0
        # Pointer escaping glossaries/ to ../shows.yaml must be blocked with 403
        assert client.get("/api/subtrans/glossary", params={"name": "glossary.escape.yaml"}).status_code == 403
        # Malformed speaks_to (null/str) or address_rules.relations (str) rejected with 400
        for bad_gyaml in (
            "characters:\n  - name_ja: A\n    speaks_to: null\n",
            "address_rules:\n  relations: not_a_list\n",
        ):
            assert (
                client.put(
                    "/api/subtrans/glossary",
                    json={"name": "glossary.frieren.yaml", "yaml_content": bad_gyaml},
                ).status_code
                == 400
            )

        # 3. Blended --grain-restore UpCunet2x engine (_ps2x_<8hex>_) matches exact blended SHA-256, and schedule checks resolution
        trt_dir = root / ".cache" / "trt_engines"
        blend_bytes = b"BLENDED_CUGAN_PS2X_ENGINE" * 16
        blend_sha = compute_upcunet2x_blended_sha256_stdlib(cons_sha, dn3x_sha, 0.45)
        blend_eng = trt_dir / f"realcugan-pro-conservative-2x_ps2x_{blend_sha[:8]}_RTX3060_trt10_1080x1920_b1-1.engine"
        blend_eng.write_bytes(blend_bytes)
        Path(f"{blend_eng}.sha256").write_text(
            json.dumps({"engine_sha256": hashlib.sha256(blend_bytes).hexdigest(), "weights_sha256": blend_sha}),
            encoding="utf-8",
        )
        ms = client.get("/api/config/models-and-shows").json()
        cugan_m = next(m for m in ms["models"] if m["id"] == "realcugan-pro-conservative-2x")
        assert cugan_m["trt_cached"] is True
        assert cugan_m["stale_trt_engines"] == []

        ff_bin = _resolve_ffmpeg_binary()
        scope_vid = root / "anime_season" / "CinemaScope_1920x800.mkv"
        r_scope = subprocess.run(
            [
                ff_bin,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1920x800:r=24:d=0.2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(scope_vid),
            ],
            capture_output=True,
            timeout=20.0,
            check=False,
        )
        if r_scope.returncode == 0:
            # Engine cached is 1080x1920 only -> 800x1920 episode must report trt_ready=False
            sched_scope = client.post(
                "/api/queue/preview-schedule",
                json={"inputs": [str(scope_vid)], "model": "realcugan-pro-conservative-2x", "use_tensorrt": True},
            ).json()
            assert sched_scope["episodes"][0]["trt_ready"] is False

        # 4. Multi-file standalone subtrans selection passes explicit --file entries without translating whole folder
        subs_5_dir = root / "subs_five"
        subs_5_dir.mkdir()
        for idx in range(1, 6):
            (subs_5_dir / f"ep{idx:02d}.srt").write_text(
                f"1\n00:00:01,000 --> 00:00:02,000\n台詞{idx}\n", encoding="utf-8"
            )
        spawned_cmds.clear()
        r_multi_sub = client.post(
            "/api/jobs/subtrans",
            json={
                "inputs": [str(subs_5_dir / "ep01.srt"), str(subs_5_dir / "ep03.srt")],
                "dry_run": True,
            },
        )
        assert r_multi_sub.status_code == 200
        cmd_msub = spawned_cmds[-1]
        assert "batch" in cmd_msub
        file_flag_vals = [cmd_msub[i + 1] for i, tok in enumerate(cmd_msub[:-1]) if tok == "--file"]
        assert len(file_flag_vals) == 2
        assert str((subs_5_dir / "ep01.srt").resolve()) in file_flag_vals
        assert str((subs_5_dir / "ep03.srt").resolve()) in file_flag_vals

        # 5. Path-scoped DELETE /api/subtrans/pending removes only matching show directory when bare stem '01' collides
        (root / "state").mkdir(parents=True, exist_ok=True)
        (root / "state" / "pending_subs.json").write_text(
            json.dumps(
                [
                    {"base_name": "01", "video_path": str(root / "ShowA" / "01.mkv")},
                    {"base_name": "01", "video_path": str(root / "ShowB" / "01.mkv")},
                ]
            ),
            encoding="utf-8",
        )
        r_del_a = client.delete("/api/subtrans/pending", params={"target": str(root / "ShowA" / "01.mkv")})
        assert r_del_a.status_code == 200
        assert r_del_a.json()["removed"] == 1
        assert r_del_a.json()["remaining_count"] == 1
        assert "ShowB" in r_del_a.json()["pending_subs"][0]["video_path"]

        # 6. Resume checkpoint chunks stay populated during stage='probing' / 'splitting' after RunLogger writes live_status.json
        ep_res = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
        cp_res = root / "out" / f".temp_{ep_res.stem}"
        (cp_res / "upscaled_segs").mkdir(parents=True, exist_ok=True)
        (cp_res / "upscaled_segs" / "up_0000.mp4").write_bytes(b"1234" * 32)
        (cp_res / "upscaled_segs" / "up_0002.mp4").write_bytes(b"1234" * 32)
        (cp_res / "manifest.json").write_text(
            json.dumps({"source_file": str(ep_res), "total_chunks": 5, "model_id": "animevideov3"}),
            encoding="utf-8",
        )
        next_running["val"] = True
        r_start_res = client.post("/api/jobs/start", json={"inputs": [str(ep_res)], "subs_after_upscale": False})
        assert r_start_res.status_code == 200
        res_job = r_start_res.json()["job"]
        (root / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": ep_res.name,
                    "input_path": str(ep_res),
                    "pid": 99112,
                    "job_id": res_job["job_id"],
                    "stage": "probing",
                    "total_chunks": 0,
                    "completed_chunks": 0,
                    "chunks": [],
                    "resumed_chunks": [],
                }
            ),
            encoding="utf-8",
        )
        st_probing = client.get("/api/status").json()["pipeline"]
        assert st_probing["effective_stage"] == "probing"
        assert st_probing["live"]["completed_chunks"] == 2
        assert st_probing["live"]["total_chunks"] == 5
        assert st_probing["live"]["resumed_chunks"] == [0, 2]
        client.post("/api/jobs/abort")

        # 7. Compare Stage syncs clamped timestamp back into #compare-ts-input & panels.js uses sizeLabel
        compare_js = client.get("/static/js/compare_stage.js").text
        panels_js_upd = client.get("/static/js/panels.js").text
        assert "tsInput.value = String(res.upscaled_timestamp_sec)" in compare_js
        assert "typeof g.size_kb === \"number\"" in panels_js_upd


def test_round2_clean_temp_checkpoint_safety_and_win32_path_hygiene(tmp_path: Path, monkeypatch):
    """Verify --clean-temp checkpoint preservation, non-existent -o no-op, reparse guards, and Win32 path hygiene."""
    import upscale
    from src.cleanup import clean_all_temporary_data
    from web.fs_browser import is_unc_path, validate_win32_path_hygiene
    from web.job_manager import _resolve_output_target, _validate_output_dir

    root = _seed_workspace(tmp_path / "r2_clean_ws")
    out_dir = root / "out"
    cp_dir = out_dir / ".temp_Ep01"
    (cp_dir / "upscaled_segs").mkdir(parents=True, exist_ok=True)
    (cp_dir / "upscaled_segs" / "up_0000.mp4").write_bytes(b"DATA" * 64)
    (cp_dir / "manifest.json").write_text(json.dumps({"source_file": "Ep01.mkv", "total_chunks": 4}), encoding="utf-8")
    (out_dir / "Ep01_4K.mux.audit.json").write_text('{"status":"PASS"}\n', encoding="utf-8")

    # 1. clean_all_temporary_data with include_checkpoints=False preserves .temp_Ep01 while moving loose audit JSON
    stats_keep = clean_all_temporary_data(str(out_dir), include_checkpoints=False)
    assert cp_dir.is_dir()
    assert (cp_dir / "upscaled_segs" / "up_0000.mp4").is_file()
    assert stats_keep["preserved_checkpoints"] == [".temp_Ep01"]
    assert (out_dir / "audit" / "Ep01_4K.mux.audit.json").is_file()

    # 2. CLI upscale.py --clean-temp -o <non_existent_dir> is a safe no-op (never wipes DEFAULT_OUTPUT_DIR out/)
    monkeypatch.setattr("src.config.DEFAULT_OUTPUT_DIR", str(out_dir))
    missing_explicit = tmp_path / "does_not_exist_yet_out"
    monkeypatch.setattr(sys, "argv", ["upscale.py", "--clean-temp", "-o", str(missing_explicit)])
    with pytest.raises(SystemExit) as exc_noop:
        upscale.main()
    assert exc_noop.value.code == 0
    assert cp_dir.is_dir(), "Explicit non-existent -o on --clean-temp must NEVER fall back to wiping DEFAULT_OUTPUT_DIR!"

    # 3. CLI _validate_cli_output_path runs BEFORE --clean-temp / --subs-pending and rejects DOS stems, trailing spaces/dots, drive-less roots
    for bad_cli_out in ("out/CON", "out/LPT1.txt", "out/trailing ", "out/trailing.", "\\out\\driveless"):
        monkeypatch.setattr(sys, "argv", ["upscale.py", "--clean-temp", "-o", bad_cli_out])
        with pytest.raises(SystemExit) as exc_info:
            upscale.main()
        assert exc_info.value.code == 1

    # 4. Extended-length local prefix \\?\<Drive>:\... is NOT treated as UNC and validates drive root properly
    ext_local = f"\\\\?\\{str(out_dir.resolve())}"
    assert is_unc_path(ext_local) is False
    validate_win32_path_hygiene(Path(ext_local), field="output_dir", raw_str=ext_local)
    if sys.platform == "win32" and not os.path.exists("Q:\\"):
        with pytest.raises(ValueError):
            validate_win32_path_hygiene(Path("\\\\?\\Q:\\nonexistent"), field="output_dir", raw_str="\\\\?\\Q:\\nonexistent")

    # 5. _validate_output_dir rejects trailing space inside quotes, DOS stems, drive-less root, and video/subtitle suffixes
    for bad_web_out in ('"out/space_end "', "out/dot_end.", "out/AUX", "out/COM9", "out/custom.mkv", "out/sub.srt"):
        with pytest.raises(ValueError):
            _validate_output_dir(root, bad_web_out, field="output_dir")
    if sys.platform == "win32":
        with pytest.raises(ValueError):
            _validate_output_dir(root, "\\out\\x", field="output_dir")
        with pytest.raises(ValueError):
            _validate_output_dir(root, "/out/x", field="output_dir")

    # _resolve_output_target converts p.resolve() OSError into ValueError cleanly
    orig_resolve = Path.resolve

    def _boom_resolve(self_p: Path, *a: Any, **kw: Any) -> Path:
        if "trigger_oserror_dir" in str(self_p):
            raise OSError("Simulated Win32 resolve error")
        return orig_resolve(self_p, *a, **kw)

    monkeypatch.setattr(Path, "resolve", _boom_resolve)
    with pytest.raises(ValueError, match="không hợp lệ"):
        _resolve_output_target(root, "trigger_oserror_dir")
    monkeypatch.setattr(Path, "resolve", orig_resolve)

    # 6. Web /api/queue/preview-schedule validates output_dir & /api/jobs/clean-temp defaults to --keep-checkpoints + blocks reparse
    spawned_cmds: List[List[str]] = []
    app = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        popen_factory=lambda cmd, **_kw: spawned_cmds.append(list(cmd)) or FakePopenProcess(cmd, pid=55110),
    )
    ep_in = root / "anime_season" / "[SubsPlease] Frieren - 02 (1080p).mkv"
    with TestClient(app) as client:
        for bad_sched_out in ("out/custom.mkv", "out/CON", "out/bad "):
            assert (
                client.post(
                    "/api/queue/preview-schedule",
                    json={"inputs": [str(ep_in)], "output_dir": bad_sched_out},
                ).status_code
                == 400
            )
        r_ct_safe = client.post("/api/jobs/clean-temp", json={"output_dir": str(out_dir)})
        assert r_ct_safe.status_code == 200
        assert "--keep-checkpoints" in spawned_cmds[-1]

        r_ct_wipe = client.post(
            "/api/jobs/clean-temp",
            json={"output_dir": str(out_dir), "include_checkpoints": True},
        )
        assert r_ct_wipe.status_code == 200
        assert "--keep-checkpoints" not in spawned_cmds[-1]

        # Simulate Win32 reparse point / NTFS junction on output_dir -> blocked with 400
        reparse_target = tmp_path / "junction_outside_target"
        reparse_target.mkdir()
        reparse_dir = root / "reparse_out"
        mk_j = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(reparse_dir), str(reparse_target)],
            capture_output=True,
            check=False,
        )
        if mk_j.returncode != 0:
            reparse_dir.mkdir(exist_ok=True)
            monkeypatch.setattr(os.path, "isjunction", lambda p: os.path.normcase(str(p)) == os.path.normcase(str(reparse_dir)))
        assert client.post("/api/jobs/clean-temp", json={"output_dir": str(reparse_dir)}).status_code in {400, 403}


def test_round2_subtrans_semicolons_en_sidecar_batch_parity_and_trt_strict_readiness(tmp_path: Path, capsys):
    """Verify Steins;Gate semicolon paths, .en sidecar pairing/exclusion, .ssa/.vtt batch parity, and strict TRT readiness."""
    import hashlib
    from src.build_trt_engine import main as build_trt_main
    from subtrans.cli import _collect_batch_input_files, _match_en_subtitle
    from web.fs_browser import compute_upcunet2x_blended_sha256_stdlib

    root = _seed_workspace(tmp_path / "r2_sub_trt_ws")
    cons_sha = "b8ae522511111111222222223333333344444444555555556666666677777777"
    dn3x_sha = "8157aa5e8888888899999999aaaaaaaabbbbbbbbccccccccddddddddeeeeeeee"
    compact_sha = "1111111122222222333333334444444455555555666666667777777788888888"
    ultra_sha = "99999999aaaaaaaabbbbbbbbccccccccddddddddeeeeeeeeffffffff00000000"
    models_yaml = root / "models" / "models.yaml"
    models_yaml.write_text(
        models_yaml.read_text(encoding="utf-8")
        + f"""  realcugan-pro-conservative-2x:
    name: Real-CUGAN Pro Conservative 2x
    arch:
      type: UpCunet2x
      upscale: 2
    downsample: none
    weights_path: models_bench/cugan_pro_cons_2x.pth
    sha256: "{cons_sha}"
    license: MIT
    default: false
  realcugan-pro-denoise3x-2x:
    name: Real-CUGAN Pro Denoise3x 2x
    arch:
      type: UpCunet2x
      upscale: 2
    downsample: none
    weights_path: models_bench/cugan_pro_dn3x_2x.pth
    sha256: "{dn3x_sha}"
    license: MIT
    default: false
  animejanai-hd-v3-compact:
    name: AnimeJaNai HD V3 Compact 2x
    arch:
      type: SRVGGNetCompact
      upscale: 2
    downsample: none
    weights_path: models_bench/aj_compact.pth
    sha256: "{compact_sha}"
    license: MIT
    default: false
  animejanai-hd-v3-ultracompact:
    name: AnimeJaNai HD V3 UltraCompact 2x
    arch:
      type: SRVGGNetCompact
      upscale: 2
    downsample: none
    weights_path: models_bench/aj_ultra.pth
    sha256: "{ultra_sha}"
    license: MIT
    default: false
""",
        encoding="utf-8",
    )

    # 1. Subtrans single-file .en sidecar auto-discovery & .ssa/.vtt/.en exclusion in _collect_batch_input_files
    sg_dir = root / "[BD] Steins;Gate;0"
    sg_dir.mkdir(parents=True)
    ja_srt = sg_dir / "Steins;Gate - 01.ja.srt"
    en_srt = sg_dir / "Steins;Gate - 01.en.srt"
    vi_art = sg_dir / "Steins;Gate - 01.vi.srt"
    partial_art = sg_dir / "Steins;Gate - 01.vi.partial.srt"
    ssa_in = sg_dir / "Chaos;Child - 02.ssa"
    vtt_in = sg_dir / "Robotics;Notes - 03.vtt"
    en_vtt = sg_dir / "Robotics;Notes - 03.en.vtt"
    for sf in (ja_srt, en_srt, vi_art, partial_art, ssa_in, vtt_in, en_vtt):
        sf.write_text("1\n00:00:01,000 --> 00:00:02,000\nテスト\n", encoding="utf-8")

    from web.fs_browser import filter_paired_subtitle_paths

    assert _match_en_subtitle(ja_srt, ja_srt.parent) == en_srt
    assert _match_en_subtitle(vtt_in, vtt_in.parent) == en_vtt
    assert _match_en_subtitle(ssa_in, ssa_in.parent) is None
    batch_files = _collect_batch_input_files(sg_dir)
    assert set(batch_files) == {ja_srt, ssa_in, vtt_in}
    # Explicit file list also filters paired .en.* and .vi.* output artifacts
    explicit_filtered = [Path(p) for p in filter_paired_subtitle_paths([str(p) for p in (ja_srt, en_srt, vi_art, partial_art, vtt_in, en_vtt)])]
    assert set(explicit_filtered) == {ja_srt, vtt_in}

    # 2. build_trt_engine CLI accepts --model auto-pareto in --dry-run mode
    rc_trt = build_trt_main(
        ["--model", "auto-pareto", "--width", "1920", "--height", "800", "--cache-dir", str(root / ".cache" / "trt_engines"), "--dry-run"]
    )
    trt_out = capsys.readouterr().out
    assert rc_trt == 0
    assert "model=animejanai-hd-v3-compact" in trt_out
    assert "model=animejanai-hd-v3-ultracompact" in trt_out
    assert "800x1920" in trt_out

    spawned_cmds: List[List[str]] = []
    app = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        popen_factory=lambda cmd, **_kw: spawned_cmds.append(list(cmd)) or FakePopenProcess(cmd, pid=66120),
    )
    with TestClient(app) as client:
        # 3. Semicolon path in standalone /api/jobs/subtrans is NOT split into fragments, and paired .en.srt is excluded on folder expansion
        r_sg = client.post("/api/jobs/subtrans", json={"input_path": str(ja_srt), "dry_run": True})
        assert r_sg.status_code == 200
        assert str(ja_srt.resolve()) in spawned_cmds[-1]

        r_sg_folder = client.post("/api/jobs/subtrans", json={"input_path": str(sg_dir), "dry_run": True})
        assert r_sg_folder.status_code == 200
        cmd_sg_f = spawned_cmds[-1]
        assert str(en_srt.resolve()) not in cmd_sg_f
        assert str(vi_art.resolve()) not in cmd_sg_f

        # Video input with --no-embed and output_format="ass" forwards --sub-format ass; without --no-embed rejects ass with 400
        ep_vid = root / "anime_season" / "[SubsPlease] Frieren - 02 (1080p).mkv"
        assert (
            client.post(
                "/api/jobs/subtrans",
                json={"input_path": str(ep_vid), "no_embed": False, "output_format": "ass", "dry_run": True},
            ).status_code
            == 400
        )
        r_vid_ass = client.post(
            "/api/jobs/subtrans",
            json={"input_path": str(ep_vid), "no_embed": True, "output_format": "ass", "dry_run": True},
        )
        assert r_vid_ass.status_code == 200
        assert "--sub-format" in spawned_cmds[-1] and "ass" in spawned_cmds[-1]

        # 4. Strict TRT sidecar verification: forged _ps2x_, missing _aa2x_ on 4x SRVGGNetCompact, transposed dims, grain_restore tier, auto-pareto, width/height build
        trt_dir = root / ".cache" / "trt_engines"
        trt_dir.mkdir(parents=True, exist_ok=True)
        payload_b = b"TRT_PLAN_BYTES_" * 32
        payload_h = hashlib.sha256(payload_b).hexdigest()

        # 4a. Forged _ps2x_deadbeef_ rejected
        forged_cugan = trt_dir / "realcugan-pro-conservative-2x_ps2x_deadbeef_RTX3060_trt10_1080x1920_b1-1.engine"
        forged_cugan.write_bytes(payload_b)
        Path(f"{forged_cugan}.sha256").write_text(
            json.dumps({"engine_sha256": payload_h, "weights_sha256": "deadbeef" * 8}), encoding="utf-8"
        )
        # 4b. 4x animevideov3 without _aa2x_ rejected
        no_aa2x = trt_dir / "animevideov3_RTX3060_trt10_1080x1920_b1-2.engine"
        no_aa2x.write_bytes(payload_b)
        Path(f"{no_aa2x}.sha256").write_text(
            json.dumps({"engine_sha256": payload_h, "weights_sha256": "abcdef0123456789" * 4}), encoding="utf-8"
        )
        ms_bad = client.get("/api/config/models-and-shows").json()
        assert next(m for m in ms_bad["models"] if m["id"] == "realcugan-pro-conservative-2x")["trt_cached"] is False
        assert next(m for m in ms_bad["models"] if m["id"] == "animevideov3")["trt_cached"] is False
        assert ms_bad["routing_modes"][0]["id"] == "auto-pareto"
        assert ms_bad["routing_modes"][0]["trt_cached"] is False

        # 4c. Real light blend (0.45) satisfies grain_restore='light' but NOT grain_restore='heavy' (1.0)
        forged_cugan.unlink()
        Path(f"{forged_cugan}.sha256").unlink()
        light_sha = compute_upcunet2x_blended_sha256_stdlib(cons_sha, dn3x_sha, 0.45)
        light_eng = trt_dir / f"realcugan-pro-conservative-2x_ps2x_{light_sha[:8]}_RTX3060_trt10_1080x1920_b1-1.engine"
        light_eng.write_bytes(payload_b)
        Path(f"{light_eng}.sha256").write_text(
            json.dumps({"engine_sha256": payload_h, "weights_sha256": light_sha}), encoding="utf-8"
        )
        # Seed real 1080p clip for dimension & grain_restore schedule checks
        from web.app import _resolve_ffmpeg_binary

        fhd_vid = root / "anime_season" / "FHD_1920x1080.mkv"
        subprocess.run(
            [_resolve_ffmpeg_binary(), "-y", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=24:d=0.2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(fhd_vid)],
            capture_output=True,
            timeout=20.0,
            check=True,
        )
        sched_light = client.post(
            "/api/queue/preview-schedule",
            json={"inputs": [str(fhd_vid)], "model": "realcugan-pro-conservative-2x", "grain_restore": "light", "use_tensorrt": True},
        ).json()
        sched_heavy = client.post(
            "/api/queue/preview-schedule",
            json={"inputs": [str(fhd_vid)], "model": "realcugan-pro-conservative-2x", "grain_restore": "heavy", "use_tensorrt": True},
        ).json()
        assert sched_light["episodes"][0]["trt_ready"] is True
        assert sched_light["episodes"][0]["width"] == 1920 and sched_light["episodes"][0]["height"] == 1080
        assert sched_heavy["episodes"][0]["trt_ready"] is False

        # 4d. auto-pareto requires BOTH compact and ultracompact engines at target resolution
        comp_eng = trt_dir / "animejanai-hd-v3-compact_RTX3060_trt10_800x1920_b1-4.engine"
        comp_eng.write_bytes(payload_b)
        Path(f"{comp_eng}.sha256").write_text(
            json.dumps({"engine_sha256": payload_h, "weights_sha256": compact_sha}), encoding="utf-8"
        )
        assert client.get("/api/config/models-and-shows").json()["routing_modes"][0]["trt_cached"] is False
        ultra_eng = trt_dir / "animejanai-hd-v3-ultracompact_RTX3060_trt10_800x1920_b1-4.engine"
        ultra_eng.write_bytes(payload_b)
        Path(f"{ultra_eng}.sha256").write_text(
            json.dumps({"engine_sha256": payload_h, "weights_sha256": ultra_sha}), encoding="utf-8"
        )
        rm_ready = client.get("/api/config/models-and-shows").json()["routing_modes"][0]
        assert rm_ready["trt_cached"] is True
        assert "800x1920" in rm_ready["trt_hw_profiles"]

        # 4e. POST /api/jobs/build-trt forwards --width, --height, and auto-pareto
        r_btrt = client.post(
            "/api/jobs/build-trt",
            json={"model_id": "auto-pareto", "width": 1920, "height": 800, "force": False},
        )
        assert r_btrt.status_code == 200
        cmd_btrt = spawned_cmds[-1]
        assert cmd_btrt[cmd_btrt.index("--model") + 1] == "auto-pareto"
        assert cmd_btrt[cmd_btrt.index("--width") + 1] == "1920"
        assert cmd_btrt[cmd_btrt.index("--height") + 1] == "800"
        assert client.post("/api/jobs/build-trt", json={"model_id": "animevideov3", "width": 32, "height": 800}).status_code == 422


def test_round2_queue_leakage_deadline_checkpoint_resume_and_abort_safety(tmp_path: Path, monkeypatch):
    """Verify mid-queue completed coercion, Episode 1 deadline paused status, batch_size=None resume, preflight_dedup seeding, and abort_job safety."""
    from web.sse_hub import scan_temp_checkpoints

    root = _seed_workspace(tmp_path / "r2_queue_cp_ws")
    ep1 = root / "anime_season" / "[SubsPlease] Frieren - 02 (1080p).mkv"
    ep2 = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"

    # 1. Legacy & explicit dedup_policy='off' preserved in scan_temp_checkpoints
    cp_off = root / "out" / f".temp_{ep1.stem}"
    (cp_off / "upscaled_segs").mkdir(parents=True, exist_ok=True)
    (cp_off / "upscaled_segs" / "up_0000.mp4").write_bytes(b"CHUNK0" * 32)
    (cp_off / "manifest.json").write_text(
        json.dumps({"source_file": str(ep1), "total_chunks": 3, "dedup_policy": "off", "model_id": "animevideov3"}),
        encoding="utf-8",
    )
    cps = scan_temp_checkpoints(root)
    assert cps[0]["dedup_policy"] == "off"

    running_flag = {"val": True}
    proc_rc = {"val": 0}
    last_proc: Dict[str, FakePopenProcess] = {}

    def queue_popen(cmd: List[str], **_kw: Any) -> FakePopenProcess:
        p = FakePopenProcess(cmd, pid=88210, running=running_flag["val"], return_code=proc_rc["val"])
        last_proc["p"] = p
        return p

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, popen_factory=queue_popen)
    with TestClient(app) as client:
        # 2. StartUpscaleRequest accepts batch_size=None (omitting --batch-size) & mid-queue completed coerces to initializing
        r_start = client.post(
            "/api/jobs/start",
            json={"inputs": [str(ep1), str(ep2)], "batch_size": None, "subs_after_upscale": False},
        )
        assert r_start.status_code == 200
        q_job = r_start.json()["job"]
        assert "--batch-size" not in q_job["command"]

        # Between Ep1 finishing (stage='completed' in live_status.json) and Ep2 starting while web_job_running is True -> initializing!
        (root / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": ep1.name,
                    "input_path": str(ep1),
                    "pid": 88210,
                    "job_id": q_job["job_id"],
                    "stage": "completed",
                    "queue_stats": {"completed_items": 1, "skipped_items": 1, "failed_items": 0, "stage_b": False},
                }
            ),
            encoding="utf-8",
        )
        st_mid = client.get("/api/status").json()["pipeline"]
        assert st_mid["effective_stage"] == "initializing"
        assert st_mid["queue_summary"]["skipped_items"] == 1

        # 3. preflight_dedup stage seeds resumed_chunks from checkpoint!
        (root / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": ep1.name,
                    "input_path": str(ep1),
                    "pid": 88210,
                    "job_id": q_job["job_id"],
                    "stage": "preflight_dedup",
                    "total_chunks": 0,
                    "completed_chunks": 0,
                    "chunks": [],
                    "resumed_chunks": [],
                }
            ),
            encoding="utf-8",
        )
        st_dedup = client.get("/api/status").json()["pipeline"]
        assert st_dedup["effective_stage"] == "preflight_dedup"
        assert st_dedup["live"]["completed_chunks"] == 1
        assert st_dedup["live"]["resumed_chunks"] == [0]

        # Stage B active queue_summary surfaces phase='stage_b' & failed_items while running
        (root / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": ep1.name,
                    "input_path": str(ep1),
                    "pid": 88210,
                    "job_id": q_job["job_id"],
                    "stage": "subtrans",
                    "queue_stats": {"completed_items": 1, "skipped_items": 0, "failed_items": 1, "stage_b": True},
                }
            ),
            encoding="utf-8",
        )
        st_stage_b = client.get("/api/status").json()["pipeline"]
        assert st_stage_b["queue_summary"]["phase"] == "stage_b"
        assert st_stage_b["queue_summary"]["failed_items"] == 1

        # 4. Episode 1 --deadline exit (return_code=1 + stage='paused' in live_status.json) stays effective_stage='paused'
        (root / "logs" / "live_status.json").write_text(
            json.dumps(
                {
                    "video_name": ep1.name,
                    "input_path": str(ep1),
                    "pid": 88210,
                    "job_id": q_job["job_id"],
                    "stage": "paused",
                    "stage_detail": "Tạm hoãn theo --deadline (còn 2 tập chưa chạy)",
                    "queue_stats": {"completed_items": 0, "skipped_items": 0, "failed_items": 2, "stage_b": False},
                }
            ),
            encoding="utf-8",
        )
        last_proc["p"]._running = False
        last_proc["p"]._return_code = 1
        st_dl_paused = client.get("/api/status").json()["pipeline"]
        assert st_dl_paused["effective_stage"] == "paused"
        assert st_dl_paused["live"]["queue_stats"]["failed_items"] == 2

        # 5. abort_job 0-chunk cleanup honors keep_temp=True AND treats os.scandir OSError as has_done_up=True (never deleting)
        cp_ep2 = root / "out" / f".temp_{ep2.stem}"
        (cp_ep2 / "upscaled_segs").mkdir(parents=True, exist_ok=True)
        (cp_ep2 / "manifest.json").write_text(json.dumps({"source_file": str(ep2)}), encoding="utf-8")

        running_flag["val"] = True
        r_keep = client.post(
            "/api/jobs/start",
            json={"inputs": [str(ep2)], "keep_temp": True, "subs_after_upscale": False},
        )
        assert r_keep.status_code == 200
        client.post("/api/jobs/abort")
        assert cp_ep2.is_dir(), "abort_job must preserve 0-chunk .temp_* when keep_temp=True!"

        # Now with keep_temp=False, simulate transient Win32 OSError on os.scandir(upscaled_segs) -> must NOT delete!
        r_lock = client.post(
            "/api/jobs/start",
            json={"inputs": [str(ep2)], "keep_temp": False, "subs_after_upscale": False},
        )
        assert r_lock.status_code == 200
        orig_scandir = os.scandir

        def _locked_scandir(p_arg: Any):
            if "upscaled_segs" in str(p_arg):
                raise PermissionError("Simulated Win32 sharing violation")
            return orig_scandir(p_arg)

        monkeypatch.setattr(os, "scandir", _locked_scandir)
        try:
            client.post("/api/jobs/abort")
        finally:
            monkeypatch.setattr(os, "scandir", orig_scandir)
        assert cp_ep2.is_dir(), "abort_job must NEVER delete .temp_* when os.scandir hits transient OSError!"

        # Dead PIDs in _aborted_owned_pids_by_job are pruned automatically
        jm = app.state.job_manager
        with jm._lock:
            jm._aborted_owned_pids_by_job["dead-tok"] = {99999999}
            jm._prune_dead_aborted_pids_locked()
            assert "dead-tok" not in jm._aborted_owned_pids_by_job


def test_round2_compare_atomic_tail_clamp_env_slots_glossary_and_pending_parity(tmp_path: Path, monkeypatch):
    """Verify Compare atomic PNG/TimeoutExpired/mutual tail clamp, .env unmirrored slots, strict Glossary schema, and pending_subs parity."""
    from subtrans.glossary import load_glossary
    from web.app import (
        _apply_subtrans_key_edits,
        _default_extract_frame,
        _probe_duration_and_dims,
        _resolve_ffmpeg_binary,
        _resolve_subtrans_key_slots_all,
    )

    monkeypatch.delenv("SUBTRANS_DISABLE_DOTENV", raising=False)
    for env_k in list(os.environ.keys()):
        if env_k.startswith("GEMINI_API_KEY"):
            monkeypatch.delenv(env_k, raising=False)

    root = _seed_workspace(tmp_path / "r2_cmp_env_ws")

    # 1. Compare _default_extract_frame converts TimeoutExpired -> sanitized RuntimeError & leaves no .tmp.png
    orig_sub_run = subprocess.run

    def _timeout_run(cmd_args: Any, *a: Any, **kw: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=cmd_args, timeout=20.0)

    monkeypatch.setattr(subprocess, "run", _timeout_run)
    out_png = tmp_path / "timeout_frame.png"
    with pytest.raises(RuntimeError, match="Quá thời gian"):
        _default_extract_frame(Path("dummy.mkv"), 1.0, out_png)
    assert not out_png.exists()
    monkeypatch.setattr(subprocess, "run", orig_sub_run)

    # 2. Mutual tail clamp when src_dur < up_dur + offset_s, plus 0-byte cache invalidation
    ff_bin = _resolve_ffmpeg_binary()
    src_short = root / "anime_season" / "ShortSrc_1s.mkv"
    up_longer = root / "out" / "ShortSrc_1s_4K.mkv"
    for clip_p, dur_s, res_s in ((src_short, "1.0", "320x180"), (up_longer, "3.0", "640x360")):
        subprocess.run(
            [ff_bin, "-y", "-f", "lavfi", "-i", f"color=c=navy:s={res_s}:r=24:d={dur_s}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip_p)],
            capture_output=True,
            timeout=20.0,
            check=True,
        )
    # _probe_duration_and_dims ignores stale DURATION tags > container_dur + 1.0 when mocked
    probed_src = _probe_duration_and_dims(src_short)
    assert 0.8 <= probed_src["duration"] <= 1.2

    running_state = {"val": False}
    spawned_cmds: List[List[str]] = []
    app = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        popen_factory=lambda cmd, **_kw: spawned_cmds.append(list(cmd))
        or FakePopenProcess(cmd, pid=44556, running=running_state["val"]),
    )
    with TestClient(app) as client:
        r_cmp = client.post(
            "/api/preview/compare-frames",
            json={
                "source_path": str(src_short),
                "upscaled_path": str(up_longer),
                "timestamp_sec": 2.5,
                "source_offset_sec": 0.2,
            },
        )
        assert r_cmp.status_code == 200
        cmp_json = r_cmp.json()
        # src_ts clamped near ~0.917s -> up_ts MUST be mutually recomputed to src_ts - 0.2 (~0.717s), NOT left at 2.5s!
        assert abs((cmp_json["source_timestamp_sec"] - cmp_json["upscaled_timestamp_sec"]) - 0.2) < 0.01
        assert cmp_json["upscaled_timestamp_sec"] < 1.0

        # Zero-byte cached PNG returns 404 on GET /api/preview/frame/... and re-extracts on POST /api/preview/compare-frames
        frame_name = cmp_json["source_frame_url"].rsplit("/", 1)[-1]
        cached_png = root / "scratch" / "web_compare" / frame_name
        cached_png.write_bytes(b"")
        assert client.get(f"/api/preview/frame/{frame_name}").status_code == 404
        r_reextract = client.post(
            "/api/preview/compare-frames",
            json={
                "source_path": str(src_short),
                "upscaled_path": str(up_longer),
                "timestamp_sec": 2.5,
                "source_offset_sec": 0.2,
            },
        )
        assert r_reextract.status_code == 200
        assert cached_png.stat().st_size > 0

        # 3. Dynamic .env slot indexing alignment when GEMINI_API_KEY is unmirrored in GEMINI_API_KEY_1..2
        dotenv_p = root / ".env"
        dotenv_p.write_text(
            "GEMINI_API_KEY=AIzaSyUnmirroredA1111\nGEMINI_API_KEY_1=AIzaSySlotTwoB2222\nGEMINI_API_KEY_2=AIzaSySlotThreeC3333\n",
            encoding="utf-8",
        )
        _, cnt_before, all_before = _resolve_subtrans_key_slots_all(dotenv_p)
        assert cnt_before == 3
        assert [s["masked"] for s in all_before[:3]] == [
            "AIzaSy...1111",
            "AIzaSy...2222",
            "AIzaSy...3333",
        ]
        # Editing Slot 2 updates AIzaSySlotTwoB2222 -> NewSlotTwoX9999 while keeping Slot 1 (1111) and Slot 3 (3333)!
        r_put_s2 = client.put("/api/subtrans/api-keys", json={"slots": {"2": "AIzaSyNewSlotTwoX9999"}})
        assert r_put_s2.status_code == 200
        after_slots = r_put_s2.json()["all_key_slots"]
        assert [s["masked"] for s in after_slots[:3]] == [
            "AIzaSy...1111",
            "AIzaSy...9999",
            "AIzaSy...3333",
        ]
        # os.environ secret is never leaked to disk when editing another slot, and overwriting an os.environ slot returns 422
        monkeypatch.setenv("GEMINI_API_KEY_5", "AIzaSyEnvSecretFive5555")
        assert client.put("/api/subtrans/api-keys", json={"slots": {"5": "AIzaSyTryOverwrite"}}).status_code == 422
        assert client.put("/api/subtrans/api-keys", json={"slots": {"4": "AIzaSyDiskFour4444"}}).status_code == 200
        assert "AIzaSyEnvSecretFive5555" not in dotenv_p.read_text(encoding="utf-8")
        monkeypatch.delenv("GEMINI_API_KEY_5", raising=False)

        # 4. Strict Glossary validation (rejects explicit null sections, bad meta/aliases/alt/strings) + *.yml support + real glossaries/*.yaml pass!
        g_dir = root / "glossaries"
        (g_dir / "glossary.custom.yml").write_text("terms:\n  - ja: 先生\n    vi: thầy\n", encoding="utf-8")
        ov_g = client.get("/api/subtrans/overview").json()["glossaries"]
        assert any(g["name"] == "glossary.custom.yml" for g in ov_g)

        for invalid_g in (
            "terms: null\n",
            "characters: null\n",
            "style_rules: null\n",
            "difficulty_markers: null\n",
            "meta: [1, 2]\n",
            "terms:\n  - ja: A\n    vi: B\n    aliases: not_a_list\n",
            "characters:\n  - name_ja: A\n    alt: 123\n",
            "characters:\n  - name_ja: A\n    role: [bad]\n",
        ):
            assert (
                client.put(
                    "/api/subtrans/glossary",
                    json={"name": "glossary.custom.yml", "yaml_content": invalid_g},
                ).status_code
                == 400
            )
        # subtrans/glossary.py:load_glossary also tolerates hand-edited null sections without TypeError
        null_g_file = tmp_path / "hand_edited_nulls.yaml"
        null_g_file.write_text("meta: null\nterms: null\ncharacters: null\nstyle_rules: null\n", encoding="utf-8")
        loaded_g = load_glossary(null_g_file)
        assert loaded_g.terms == [] and loaded_g.characters == []

        # Validate every real glossary YAML in the live repository against PUT /api/subtrans/glossary validator!
        live_repo = Path(__file__).resolve().parent.parent
        for real_gf in sorted((live_repo / "glossaries").glob("*.yaml")):
            raw_txt = real_gf.read_text(encoding="utf-8")
            if "\n" not in raw_txt.strip() and raw_txt.strip().endswith((".yaml", ".yml")):
                continue
            r_real = client.put(
                "/api/subtrans/glossary",
                json={"name": "glossary.custom.yml", "yaml_content": raw_txt},
            )
            assert r_real.status_code == 200, f"Real glossary {real_gf.name} failed validation: {r_real.text}"

        # 5. DELETE /api/subtrans/pending locked (409) while busy, and --subs-pending omits --subs-quality when subs_quality=None
        (root / "state").mkdir(parents=True, exist_ok=True)
        (root / "state" / "pending_subs.json").write_text(
            json.dumps([{"episode": "Ep09", "video_path": str(up_longer), "subs_quality": "high", "no_embed": True, "subs_upgrade": True}]),
            encoding="utf-8",
        )
        r_pend_preserve = client.post("/api/jobs/subs-pending", json={"subs_quality": None, "dry_run": True})
        assert r_pend_preserve.status_code == 200
        assert "--subs-quality" not in spawned_cmds[-1]

        running_state["val"] = True
        client.post("/api/jobs/subs-pending", json={"subs_quality": "high", "dry_run": True})
        assert client.delete("/api/subtrans/pending", params={"target": "Ep09"}).status_code == 409
        client.post("/api/jobs/abort")
        running_state["val"] = False
        assert client.delete("/api/subtrans/pending", params={"target": "Ep09"}).status_code == 200

        # Static CSS & JS regression checks for Round 2 fixes
        css_txt = client.get("/static/styles.css").text
        panels_txt = client.get("/static/js/panels.js").text
        app_txt = client.get("/static/js/app.js").text
        cmp_txt = client.get("/static/js/compare_stage.js").text
        assert ".span-full," in css_txt
        assert ".input-with-btn > select," in css_txt
        assert "#pending-subs-list .badge" in css_txt
        assert "overflow-wrap: anywhere;" in css_txt
        assert 'target.split(";")' not in panels_txt
        assert "(validateBatchSizeInput({ focusOnError: false, toastOnError: false }) ?? 8)" not in panels_txt
        assert "chk-clean-include-checkpoints" in panels_txt
        assert "confirmDiscardGlossaryChanges" in panels_txt
        assert 'activeEl.type === "range"' in cmp_txt
        assert "hasCheckpoints" in app_txt

    # 6. Mock mode locks DELETE /api/subtrans/pending (409) while simulated job is running, then deletes (200) once aborted
    mock_app = create_app(workspace_root=_seed_workspace(tmp_path / "r2_mock_ws"), mock_mode=True)
    with TestClient(mock_app) as mclient:
        ov_m1 = mclient.get("/api/subtrans/overview").json()
        assert len(ov_m1["pending_subs"]) == 1
        assert mclient.delete("/api/subtrans/pending", params={"target": "[Demo] Anime_EP02 (1080p)"}).status_code == 409
        assert mclient.post("/api/jobs/abort").status_code == 200
        r_mdel = mclient.delete("/api/subtrans/pending", params={"target": "[Demo] Anime_EP02 (1080p)"})
        assert r_mdel.status_code == 200
        assert mclient.get("/api/subtrans/overview").json()["pending_subs"] == []


def test_round2_cli_bare_resume_sub_meta_deadline_ep1_and_pending_subs_fields(tmp_path: Path, monkeypatch):
    """Verify upscale.main() Episode 1 pre-RunLogger deadline paused status, bare CLI resume Stage B metadata carry-forward, and --subs-pending field forwarding."""
    from datetime import datetime
    import time
    import upscale
    import src.logger as logger_mod

    root = _seed_workspace(tmp_path / "r2_cli_parity_ws")
    ep1 = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    out_dir = root / "out"
    live_file = root / "logs" / "live_status.json"
    monkeypatch.setattr(logger_mod, "LOGS_DIR", str(root / "logs"))
    monkeypatch.setattr("src.config.DEFAULT_OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("UPSCALER_WEB_JOB_ID", "webjob-dl-ep1")

    # 1. Episode 1 --deadline miss BEFORE RunLogger starts writes logs/live_status.json with stage='paused' and job_id!
    now_plus_1m = datetime.fromtimestamp(time.time() + 60.0).strftime("%H:%M")
    monkeypatch.setattr(upscale, "check_deadline_and_eta", lambda *_a, **_kw: (False, False, None))
    monkeypatch.setattr(
        sys,
        "argv",
        ["upscale.py", str(ep1), "-o", str(out_dir), "--deadline", now_plus_1m, "--no-subs"],
    )
    with pytest.raises(SystemExit) as exc_dl:
        upscale.main()
    assert exc_dl.value.code == 1
    assert live_file.is_file()
    dl_live = json.loads(live_file.read_text(encoding="utf-8"))
    assert dl_live["stage"] == "paused"
    assert dl_live["job_id"] == "webjob-dl-ep1"
    assert dl_live["pid"] == os.getpid()

    # 2. Bare CLI resume (`upscale.py <ep1> -o <out_dir>`) preserves saved Stage B metadata from checkpoint manifest!
    cp_dir = out_dir / f".temp_{ep1.stem}"
    cp_dir.mkdir(parents=True, exist_ok=True)
    saved_glossary = str(root / "glossaries" / "frieren.yaml")
    (cp_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_file": str(ep1),
                "total_chunks": 2,
                "subs_after_upscale": True,
                "subs_quality": "high",
                "glossary": saved_glossary,
                "subs_upgrade": True,
                "no_embed": True,
                "sound_cues": "translate",
                "prefer_whisper": True,
            }
        ),
        encoding="utf-8",
    )
    captured_stage_b_cmds: List[List[str]] = []
    captured_extra_meta: List[Dict[str, Any]] = []

    def _fake_upscale_pipe(*_a: Any, **kw: Any) -> str:
        captured_extra_meta.append(dict(kw.get("manifest_extra_meta") or {}))
        out_p = str(out_dir / f"{ep1.stem}_4K.mkv")
        Path(out_p).write_bytes(b"\x00" * (1024 * 1024 + 64))
        return out_p

    orig_subproc_run = subprocess.run

    def _fake_subproc_run(cmd_a: Any, *a: Any, **kw: Any) -> Any:
        if isinstance(cmd_a, list) and "--subs-only" in cmd_a:
            captured_stage_b_cmds.append(list(cmd_a))
            return subprocess.CompletedProcess(args=cmd_a, returncode=0)
        return orig_subproc_run(cmd_a, *a, **kw)

    monkeypatch.setattr("src.pipeline.run_upscale_pipeline", _fake_upscale_pipe)
    monkeypatch.setattr(upscale, "run_upscale_pipeline", _fake_upscale_pipe, raising=False)
    monkeypatch.setattr(subprocess, "run", _fake_subproc_run)
    monkeypatch.setattr(sys, "argv", ["upscale.py", str(ep1), "-o", str(out_dir)])
    with pytest.raises(SystemExit) as exc_ok:
        upscale.main()
    assert exc_ok.value.code == 0

    assert len(captured_stage_b_cmds) == 1
    sb_cmd = captured_stage_b_cmds[0]
    assert sb_cmd[sb_cmd.index("--subs-quality") + 1] == "high"
    assert sb_cmd[sb_cmd.index("--glossary") + 1] == saved_glossary
    assert "--subs-upgrade" in sb_cmd
    assert "--no-embed" in sb_cmd
    assert sb_cmd[sb_cmd.index("--sound-cues") + 1] == "translate"
    assert "--prefer-whisper" in sb_cmd
    # Bare CLI resume must NOT overwrite manifest_extra_meta with default CLI flags
    assert "subs_quality" not in captured_extra_meta[0]

    # 3. --subs-pending forwards per-item subs_quality, subs_upgrade, no_embed, and sound_cues when --subs-quality is NOT passed on CLI
    ja_dummy = root / "anime_season" / "Frieren_02.ja.srt"
    captured_pending_calls: List[Dict[str, Any]] = []

    class _FakeTransPipeline:
        def __init__(self, **init_kw: Any) -> None:
            self._init_kw = init_kw

        def run(self, **run_kw: Any) -> Dict[str, Any]:
            captured_pending_calls.append({**self._init_kw, **run_kw})
            return {"subtitle_path": None}

    monkeypatch.setattr("subtrans.pipeline.TranslationPipeline", _FakeTransPipeline)
    monkeypatch.setattr(
        upscale,
        "load_pending_subs",
        lambda: [
            {
                "base_name": ep1.stem,
                "video_path": str(out_dir / f"{ep1.stem}_4K.mkv"),
                "ja_path": str(ja_dummy),
                "output_dir": str(out_dir),
                "subs_quality": "high",
                "glossary": saved_glossary,
                "subs_upgrade": True,
                "no_embed": True,
                "sound_cues": "translate",
                "prefer_whisper": False,
                "sub_offset_ms": 120,
            }
        ],
    )
    monkeypatch.setattr(sys, "argv", ["upscale.py", "--subs-pending"])
    with pytest.raises(SystemExit) as exc_sp:
        upscale.main()
    assert exc_sp.value.code == 0
    assert len(captured_pending_calls) == 1
    sp = captured_pending_calls[0]
    assert sp["subs_quality"] == "high"
    assert sp["subs_upgrade"] is True
    assert sp["no_embed"] is True
    assert sp["sound_cues"] == "translate"
    assert sp["sub_offset_ms"] == 120
