"""Opaque-box requirement-driven E2E acceptance and performance tests for P0-P2."""

from __future__ import annotations

import asyncio
import base64
from collections import deque
import hashlib
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient
import httpx
import pysubs2
import pytest
from typer.testing import CliRunner

from subtrans.cache import TranslationCache
import subtrans.cli as subtrans_cli
from subtrans.config import load_models_config
from subtrans.dynamic_cast import HysteresisCastManager
from subtrans.eval.runner import load_yaml as load_eval_yaml, verify_pronoun_cue
from subtrans.gemini_backend import GeminiTranslator
from subtrans.glossary import load_glossary
from subtrans.linter import RULE_PRONOUN_CONSISTENCY, lint_cue
from subtrans.models import Cue
from subtrans.parser import detect_auto_offset
import subtrans.throttle as subtrans_throttle
from subtrans.throttle import (
    RateLimiter,
    clear_rate_limiter_registry,
    get_shared_rate_limiter,
)
from web.app import create_app
from web.job_manager import JobManager
from web.sse_hub import TelemetryHub


pytestmark = [pytest.mark.e2e]

REPO_ROOT = Path(__file__).resolve().parents[1]

PNG_1X1_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SECRET_ENV_TOKENS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


class _FakePopenProcess:
    """Minimal in-memory Popen double for offline JobManager tests."""

    def __init__(self, cmd: List[str], pid: int = 61001) -> None:
        self.cmd = list(cmd)
        self.pid = pid
        self._return_code = 0
        self.stdout = io.StringIO("[INFO] Dry-run stub job finished\n")

    def poll(self) -> int:
        return self._return_code

    def wait(self) -> int:
        return self._return_code

    def terminate(self) -> None:
        self._return_code = -15

    def kill(self) -> None:
        self._return_code = -9


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


def _seed_web_workspace(root: Path) -> Path:
    """Create an isolated Studio workspace tree inside tmp_path."""
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "subtrans" / ".cache").mkdir(parents=True, exist_ok=True)
    (root / "glossaries").mkdir(parents=True, exist_ok=True)
    (root / "out" / "audit").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / ".cache" / "trt_engines").mkdir(parents=True, exist_ok=True)
    (root / "anime_season").mkdir(parents=True, exist_ok=True)

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
    status: active
""",
        encoding="utf-8",
    )
    (root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv").write_bytes(b"\x00" * 4096)
    return root


def _write_nonperiodic_srt_pair(
    tmp_dir: Path,
    count: int,
    split_idx: Optional[int],
    offset_before_ms: int,
    offset_after_ms: int,
) -> tuple[Path, Path]:
    """Generate non-periodic JA/EN .srt files to avoid false periodic cue aliasing."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ja_subs = pysubs2.SSAFile()
    en_subs = pysubs2.SSAFile()
    cursor_ms = 5000
    for idx in range(count):
        dur_ms = 1800 + (idx * 173) % 900
        gap_ms = 1400 + (idx * 617) % 2300
        start_ms = cursor_ms
        end_ms = start_ms + dur_ms
        en_shift = offset_before_ms if (split_idx is None or idx < split_idx) else offset_after_ms
        ja_subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=f"セリフ {idx}"))
        en_subs.append(pysubs2.SSAEvent(start=start_ms + en_shift, end=end_ms + en_shift, text=f"Line {idx}"))
        cursor_ms = end_ms + gap_ms

    ja_path = tmp_dir / "episode.ja.srt"
    en_path = tmp_dir / "episode.en.srt"
    ja_subs.save(str(ja_path))
    en_subs.save(str(en_path))
    return ja_path, en_path


def _clean_subprocess_env(*, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    keep_keys = (
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "PATH",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "CUDA_VISIBLE_DEVICES",
    )
    env = {k: os.environ[k] for k in keep_keys if k in os.environ}
    env["PYTHONUTF8"] = "1"
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["SUBTRANS_DISABLE_DOTENV"] = "1"
    if extra:
        env.update(extra)
    return env


def _reject_non_finite_json_constant(token: str) -> Any:
    raise ValueError(f"Non-RFC8259 JSON constant emitted: {token}")


def _parse_sse_data_line(raw_chunk: str) -> Dict[str, Any]:
    for line in raw_chunk.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :], parse_constant=_reject_non_finite_json_constant)
    raise AssertionError(f"Missing SSE data line in {raw_chunk!r}")


def _write_audit_json(out_dir: Path, idx: int) -> Path:
    ep = f"Ep_{idx:03d}_4K"
    payload = {
        "output_file": str(out_dir / f"{ep}.mkv"),
        "source_file": f"Ep_{idx:03d}.mkv",
        "status": "SUCCESS",
        "output_sanity": {"status": "PASS"},
        "model_id": "animevideov3",
        "average_pipeline_fps": 18.4,
        "ai_steady_fps": 22.1,
        "source_frames": 34560,
        "output_frames": 34560,
        "frames_matched": True,
        "audio_tracks_count": 1,
        "subtitle_tracks_count": 1,
        "fonts_count": 0,
        "output_file_size_mb": 412.5,
        "completed_at": "2026-11-22T01:00:00",
    }
    target = out_dir / f"{ep}.audit.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


# Domain E1: Performance & Resource Bounds


def test_e1_detect_auto_offset_piecewise_perf_and_commercial_break(tmp_path: Path) -> None:
    """Verify detect_auto_offset meets <300ms piecewise bound at N=1200 and recovers >60s breaks."""
    ja_pw, en_pw = _write_nonperiodic_srt_pair(
        tmp_path / "piecewise",
        count=1200,
        split_idx=400,
        offset_before_ms=0,
        offset_after_ms=10000,
    )
    # Pre-warm newly written temp files so NTFS creation flush does not skew cold I/O.
    ja_pw.read_bytes()
    en_pw.read_bytes()

    # Bound up to 3 trials to reject single-shot 6-worker OS scheduling spikes while preserving strict <300ms.
    wall_trials_ms: List[float] = []
    cpu_trials_ms: List[float] = []
    for _ in range(3):
        c0 = time.process_time()
        t0 = time.perf_counter()
        _offset_ms, is_applied, _reason, stats = detect_auto_offset(str(ja_pw), str(en_pw))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        cpu_ms = (time.process_time() - c0) * 1000.0
        wall_trials_ms.append(elapsed_ms)
        cpu_trials_ms.append(cpu_ms)
        assert is_applied is True
        segs = stats.get("piecewise_segments") or []
        assert len(segs) == 2
        assert abs(int(segs[0]["offset_ms"])) <= 500
        assert abs(int(segs[1]["offset_ms"]) - 10000) <= 500
        if elapsed_ms < 300.0 and cpu_ms < 300.0:
            break

    best_wall_ms = min(wall_trials_ms)
    best_cpu_ms = min(cpu_trials_ms)
    assert best_wall_ms < 300.0, (
        f"detect_auto_offset N=1200 piecewise took min {best_wall_ms:.2f} ms "
        f"(wall={[round(x, 2) for x in wall_trials_ms]}, cpu={[round(x, 2) for x in cpu_trials_ms]}, limit 300 ms)"
    )
    assert best_cpu_ms < 300.0, (
        f"detect_auto_offset N=1200 piecewise CPU time {best_cpu_ms:.2f} ms (limit 300 ms)"
    )

    ja_brk, en_brk = _write_nonperiodic_srt_pair(
        tmp_path / "break_gt_60s",
        count=80,
        split_idx=None,
        offset_before_ms=75000,
        offset_after_ms=75000,
    )
    brk_offset_ms, brk_applied, _brk_reason, _brk_stats = detect_auto_offset(str(ja_brk), str(en_brk))
    assert brk_applied is True
    assert abs(brk_offset_ms - 75000) <= 500


def test_e1_dtw_alignment_plan_peak_ram_under_35mb() -> None:
    """Verify windowed Sakoe-Chiba DTW alignment at N=1200 stays below 35 MB peak working-set RAM."""
    probe_code = """
import gc
import json
import psutil
from subtrans.dtw_auto_align import SubCue, get_dtw_alignment_plan

ja_cues = []
en_cues = []
cursor_ms = 2000
for idx in range(1200):
    dur_ms = 1800 + (idx * 173) % 900
    gap_ms = 1400 + (idx * 617) % 2300
    start_ms = cursor_ms
    end_ms = start_ms + dur_ms
    ja_cues.append(SubCue(start_ms=start_ms, end_ms=end_ms, text=f"フラン、テスト発話 {idx}"))
    en_cues.append(SubCue(start_ms=start_ms, end_ms=end_ms, text=f"Fran test dialogue {idx}"))
    cursor_ms = end_ms + gap_ms

proc = psutil.Process()
gc.collect()
before_info = proc.memory_info()
before_peak_wset = getattr(before_info, "peak_wset", before_info.rss)
cue_map, _split_map = get_dtw_alignment_plan(
    ja_cues,
    en_cues,
    cast_names_ja_en=[("フラン", "Fran"), ("師匠", "Master")],
)
after_info = proc.memory_info()
after_peak_wset = getattr(after_info, "peak_wset", after_info.rss)
peak_delta_mb = (after_peak_wset - before_peak_wset) / (1024 * 1024)
matched_entries = [(idx, val) for idx, val in cue_map.items() if val is not None]
exact_text_matches = sum(
    1
    for idx, (s_ms, e_ms, en_text) in matched_entries
    if en_text == en_cues[idx].text and 0 <= s_ms < e_ms
)
print(
    json.dumps(
        {
            "peak_delta_mb": peak_delta_mb,
            "matched": len(matched_entries),
            "exact_text_matches": exact_text_matches,
        }
    )
)
""".strip()

    proc = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=str(REPO_ROOT),
        env=_clean_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"DTW subprocess failed:\n{proc.stderr}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    matched = int(payload["matched"])
    exact_text_matches = int(payload["exact_text_matches"])
    peak_delta_mb = float(payload["peak_delta_mb"])

    assert matched >= 1100, f"Expected >= 1100 aligned cues, got {matched}"
    assert exact_text_matches >= 1100, f"Expected >= 1100 exact aligned EN cues, got {exact_text_matches}"
    assert peak_delta_mb < 35.0, f"DTW N=1200 peak working-set RAM delta was {peak_delta_mb:.2f} MB (limit 35 MB)"


def test_e1_src_config_cold_import_lazy_torch_and_under_80ms() -> None:
    """Verify cold import src.config stays under 80 ms median and never imports torch eagerly."""
    env = _clean_subprocess_env(extra={"UPSCALER_HEVC_ENCODER": "libx265"})
    probe_code = (
        "import json, sys, time; "
        "t0 = time.perf_counter(); "
        "import src.config; "
        "elapsed_ms = (time.perf_counter() - t0) * 1000.0; "
        "print(json.dumps({'import_ms': elapsed_ms, 'has_torch': 'torch' in sys.modules}))"
    )

    latencies_ms: List[float] = []
    for run_idx in range(4):
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert payload["has_torch"] is False, "import src.config eagerly imported torch"
        if run_idx > 0:
            latencies_ms.append(float(payload["import_ms"]))

    median_ms = statistics.median(latencies_ms)
    assert median_ms < 80.0, f"Median cold import src.config took {median_ms:.2f} ms (limit 80 ms)"

    enc_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, src.config; assert 'torch' not in sys.modules; print(src.config.HEVC_ENCODER)",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert enc_check.returncode == 0, enc_check.stderr
    assert enc_check.stdout.strip().splitlines()[-1] == "libx265"


@pytest.mark.web
def test_e1_api_audits_p50_under_3ms_and_invalidates_on_new_file(tmp_path: Path) -> None:
    """Verify cached /api/audits handler p50 < 3 ms, ASGI roundtrip, and invalidation on 31st file."""
    root = _seed_web_workspace(tmp_path / "audits_perf")
    out_dir = root / "out"
    for idx in range(30):
        _write_audit_json(out_dir, idx)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics)

    audits_route = next((r for r in app.routes if getattr(r, "path", None) == "/api/audits"), None)
    assert audits_route is not None
    audits_endpoint = getattr(audits_route, "endpoint", None)
    assert callable(audits_endpoint)

    for _ in range(5):
        warm_payload = audits_endpoint(output_dir=None)
        assert len(warm_payload["audits"]) == 30

    window_p50s: List[float] = []
    for attempt in range(5):
        handler_latencies_ms: List[float] = []
        for _ in range(31):
            t0 = time.perf_counter()
            payload = audits_endpoint(output_dir=None)
            handler_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            assert len(payload["audits"]) == 30
        win_p50 = statistics.median(handler_latencies_ms)
        window_p50s.append(win_p50)
        if win_p50 < 3.0:
            break
        if attempt < 4:
            time.sleep(0.02)
    handler_p50_ms = min(window_p50s)
    assert handler_p50_ms < 3.0, (
        f"GET /api/audits handler p50 was {handler_p50_ms:.2f} ms across {len(window_p50s)} windows (limit 3.0 ms)"
    )

    async def _measure() -> tuple[float, int, int]:
        transport = httpx.ASGITransport(app=app, client=("testclient", 50000))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for _ in range(5):
                warm = await client.get("/api/audits")
                assert warm.status_code == 200

            latencies_ms: List[float] = []
            initial_count = 0
            for _ in range(31):
                t0 = time.perf_counter()
                resp = await client.get("/api/audits")
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                assert resp.status_code == 200
                initial_count = len(resp.json()["audits"])

            _write_audit_json(out_dir, 30)
            after_resp = await client.get("/api/audits")
            assert after_resp.status_code == 200
            after_count = len(after_resp.json()["audits"])
            return statistics.median(latencies_ms), initial_count, after_count

    asgi_p50_ms, count_before, count_after = asyncio.run(_measure())
    assert count_before == 30
    assert count_after == 31
    assert asgi_p50_ms < 15.0, f"GET /api/audits ASGI p50 was {asgi_p50_ms:.2f} ms (limit 15.0 ms)"


# Domain E2: CLI, Dry-Run, Eval & Throttle Behavioral Parity


def test_e2_cli_run_dry_run_writes_zero_cascade_telemetry_jsonl(tmp_path: Path) -> None:
    """Verify subtrans.py run --dry-run (with cascade enabled) writes zero cascade_telemetry_*.jsonl files."""
    ja_srt = tmp_path / "ep01.ja.srt"
    ja_srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,500\nフラン、大丈夫？\n\n"
        "2\n00:00:04,000 --> 00:00:06,500\n師匠、危ない！\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    cache_db = tmp_path / "cache.db"
    repo_logs_dir = REPO_ROOT / "logs"
    before_repo_telemetry = (
        set(repo_logs_dir.glob("cascade_telemetry_*.jsonl")) if repo_logs_dir.is_dir() else set()
    )

    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k != "PYTEST_CURRENT_TEST" and not any(tok in k.upper() for tok in _SECRET_ENV_TOKENS)
    }
    clean_env["PYTHONUTF8"] = "1"
    clean_env["PYTHON_DOTENV_DISABLED"] = "1"
    clean_env["SUBTRANS_DISABLE_DOTENV"] = "1"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "subtrans.py"),
            "run",
            str(ja_srt),
            "--dry-run",
            "--no-rag",
            "--out-dir",
            str(out_dir),
            "--cache-db",
            str(cache_db),
        ],
        cwd=str(tmp_path),
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"subtrans.py run --dry-run failed:\n{proc.stdout}\n{proc.stderr}"
    assert (out_dir / "ep01.vi.srt").is_file()
    assert list((tmp_path / "logs").glob("cascade_telemetry_*.jsonl")) == []

    after_repo_telemetry = (
        set(repo_logs_dir.glob("cascade_telemetry_*.jsonl")) if repo_logs_dir.is_dir() else set()
    )
    assert after_repo_telemetry == before_repo_telemetry


def test_e2_cli_batch_parity_and_dead_flag_rejection(tmp_path: Path) -> None:
    """Verify dead --yes/-y and --review/-r flags are rejected and batch exposes full run parity."""
    runner = CliRunner()
    dummy_ja = tmp_path / "dummy.ja.srt"
    dummy_ja.write_text("1\n00:00:01,000 --> 00:00:03,000\nフラン、大丈夫？\n", encoding="utf-8")
    in_dir = tmp_path / "batch_in"
    in_dir.mkdir()
    (in_dir / "ep01.ja.srt").write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n師匠、行こう。\n",
        encoding="utf-8",
    )

    dead_out = tmp_path / "dead_out"
    dead_db = tmp_path / "dead.db"
    for cmd_args, dead_flag in (
        (
            ["run", str(dummy_ja), "--dry-run", "--no-rag", "--cache-db", str(dead_db), "-o", str(dead_out), "--yes"],
            "--yes",
        ),
        (["run", str(dummy_ja), "--dry-run", "--no-rag", "--cache-db", str(dead_db), "-o", str(dead_out), "-y"], "-y"),
        (
            [
                "run",
                str(dummy_ja),
                "--dry-run",
                "--no-rag",
                "--cache-db",
                str(dead_db),
                "-o",
                str(dead_out),
                "--review",
            ],
            "--review",
        ),
        (["run", str(dummy_ja), "--dry-run", "--no-rag", "--cache-db", str(dead_db), "-o", str(dead_out), "-r"], "-r"),
        (["batch", str(in_dir), "--dry-run", "--cache-db", str(dead_db), "-o", str(dead_out), "--yes"], "--yes"),
        (["batch", str(in_dir), "--dry-run", "--cache-db", str(dead_db), "-o", str(dead_out), "-y"], "-y"),
        (["batch", str(in_dir), "--dry-run", "--cache-db", str(dead_db), "-o", str(dead_out), "--review"], "--review"),
        (["batch", str(in_dir), "--dry-run", "--cache-db", str(dead_db), "-o", str(dead_out), "-r"], "-r"),
    ):
        res = runner.invoke(subtrans_cli.app, cmd_args)
        plain_out = _ANSI_ESCAPE_RE.sub("", res.output).lower()
        assert res.exit_code == 2 and "no such option" in plain_out and dead_flag in plain_out, (
            f"Expected exit_code=2 & 'No such option' for {dead_flag} in {cmd_args}, got {res.exit_code}: {plain_out}"
        )

    help_res = runner.invoke(subtrans_cli.app, ["batch", "--help"])
    assert help_res.exit_code == 0
    help_plain = _ANSI_ESCAPE_RE.sub("", help_res.output)
    for flag in (
        "--subs-quality",
        "--subs-upgrade",
        "--no-rag",
        "--no-cascade",
        "--auto-offset",
        "--cache-db",
    ):
        assert flag in help_plain, f"Missing {flag} in subtrans batch --help"

    batch_out = tmp_path / "batch_out"
    batch_db = tmp_path / "batch.db"
    run_res = runner.invoke(
        subtrans_cli.app,
        [
            "batch",
            str(in_dir),
            "--dry-run",
            "--subs-quality",
            "high",
            "--subs-upgrade",
            "--no-rag",
            "--no-cascade",
            "--auto-offset",
            "--cache-db",
            str(batch_db),
            "-o",
            str(batch_out),
        ],
    )
    assert run_res.exit_code == 0, run_res.output
    vi_srt = batch_out / "ep01.vi.srt"
    audit_md = batch_out / "ep01.audit.md"
    assert vi_srt.is_file()
    assert "[DRY-RUN VI]" in vi_srt.read_text(encoding="utf-8")
    assert batch_db.is_file(), f"Expected batch --cache-db at {batch_db}"
    assert audit_md.is_file(), f"Expected batch QA audit report at {audit_md}"
    audit_text = audit_md.read_text(encoding="utf-8")
    assert "DRY-RUN" in audit_text and "ep01.ja.srt" in audit_text


def test_e2_eval_pronoun_verification_via_cast_manager_and_linter() -> None:
    """Verify subtrans eval CLI, verify_pronoun_cue, and HysteresisCastManager + lint_cue."""
    eval_res = CliRunner().invoke(subtrans_cli.app, ["eval"])
    assert eval_res.exit_code == 0, f"subtrans eval failed:\n{eval_res.output}"

    suite = load_eval_yaml("pronoun_suite.yaml")
    cues_by_id = {c["id"]: c for c in suite.get("cues", [])}
    for cid in ("tenken_01", "tenken_03", "tenken_05", "dmmtv_cc_01"):
        ok, msg = verify_pronoun_cue(cues_by_id[cid])
        assert ok is True, f"verify_pronoun_cue failed for {cid}: {msg}"

    mismatched_spec = {
        "id": "neg_tenken_comm_mismatch",
        "series": "tenken",
        "speaker": "Fran",
        "addressee": "Shishou",
        "mode": "comm",
        "ja": "師匠、大丈夫？",
        "allowed_pronouns": ["ông"],
        "expected_names": [],
    }
    bad_ok, bad_msg = verify_pronoun_cue(mismatched_spec)
    assert bad_ok is False, f"Expected verify_pronoun_cue to reject pronoun mismatch in mode=comm, got: {bad_msg}"

    tenken_glossary = load_glossary(REPO_ROOT / "glossaries" / "glossary.tenken.yaml")
    assert tenken_glossary is not None

    mgr = HysteresisCastManager.from_glossary(tenken_glossary)
    assert mgr.resolve_pronouns("Fran", "Shishou") == ("Fran", "Sư phụ")
    assert mgr.resolve_pronouns("Sư phụ", "Fran") == ("ta", "Fran")

    for mode in ("spoken", "comm"):
        valid_cue = Cue(
            id=1,
            start=0,
            end=2500,
            ja="師匠、行く。",
            vi="Sư phụ, Fran đi đây.",
            speaker="Fran",
            addressee="Shishou",
            p1="Fran",
            p2="Sư phụ",
            mode=mode,
        )
        valid_violations = lint_cue(valid_cue, glossary=tenken_glossary)
        assert not any(v.rule_id == RULE_PRONOUN_CONSISTENCY for v in valid_violations)

        bad_cue = Cue(
            id=2,
            start=2600,
            end=5000,
            ja="師匠、行く。",
            vi="Ông ơi, tôi đi đây.",
            speaker="Fran",
            addressee="Shishou",
            p1="tôi",
            p2="ông",
            mode=mode,
        )
        bad_violations = lint_cue(bad_cue, glossary=tenken_glossary)
        assert any(v.rule_id == RULE_PRONOUN_CONSISTENCY for v in bad_violations), (
            f"Expected {RULE_PRONOUN_CONSISTENCY} in mode={mode}"
        )

    shift_stats = mgr.observe_batch_cues(
        [
            Cue(
                id=3,
                start=5200,
                end=7000,
                ja="くそっ！",
                vi="Mày dám à, tao không tha đâu!",
                speaker="Fran",
                addressee="Shishou",
                p1="tao",
                p2="mày",
                mode="spoken",
            )
        ]
    )
    assert shift_stats["situational_shifts"] == 1
    assert mgr.resolve_pronouns("Fran", "Shishou") == ("Fran", "Sư phụ")


def test_e2_rate_limiter_thread_safe_rpm_and_models_yaml_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify models.yaml RPMs, concurrent RateLimiter windowing, and shared registry contract."""
    models_cfg = load_models_config().get("models", {})
    assert models_cfg["gemini-3.8-flash"]["rpm"] == 5
    assert models_cfg["gemini-3.5-flash-lite"]["rpm"] == 15

    clock_lock = threading.Lock()
    virtual_state = {"now": 1000.0}
    sleep_durations: List[float] = []
    orig_sleep = time.sleep

    def _fake_monotonic() -> float:
        with clock_lock:
            return virtual_state["now"]

    def _fake_sleep(seconds: float) -> None:
        with clock_lock:
            dt = max(0.0, float(seconds))
            sleep_durations.append(dt)
            virtual_state["now"] += dt
        orig_sleep(0.0005)

    monkeypatch.setattr(subtrans_throttle.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(subtrans_throttle.time, "sleep", _fake_sleep)

    acquired_times: List[float] = []

    class _RecordingDeque(deque):
        def append(self, val: float) -> None:
            with clock_lock:
                acquired_times.append(float(val))
            super().append(val)

    limiter = RateLimiter(max_rpm=5)
    limiter.timestamps = _RecordingDeque()
    barrier = threading.Barrier(10)
    errors: List[BaseException] = []

    def _worker() -> None:
        try:
            barrier.wait(timeout=2.0)
            limiter.wait_if_needed()
        except BaseException as exc:
            errors.append(exc)

    wall_start = time.perf_counter()
    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=2.0)
    wall_elapsed = time.perf_counter() - wall_start

    assert not errors
    assert wall_elapsed < 0.5, f"Virtual-clock concurrency check took {wall_elapsed:.3f}s"
    assert len(acquired_times) == 10
    assert sum(sleep_durations) >= 60.0
    sorted_times = sorted(acquired_times)
    for idx in range(5):
        assert sorted_times[idx + 5] - sorted_times[idx] >= 60.0

    assert callable(clear_rate_limiter_registry)
    assert callable(get_shared_rate_limiter)
    clear_rate_limiter_registry()
    try:
        rl_flash_a = get_shared_rate_limiter("slot1", "gemini-3.8-flash")
        rl_flash_b = get_shared_rate_limiter("slot1", "gemini-3.8-flash")
        rl_flash_other = get_shared_rate_limiter("slot2", "gemini-3.8-flash")
        rl_lite = get_shared_rate_limiter("slot1", "gemini-3.5-flash-lite")
        assert rl_flash_a is rl_flash_b
        assert rl_flash_a is not rl_flash_other
        assert rl_flash_a is not rl_lite
        assert rl_flash_a.max_rpm <= int(models_cfg["gemini-3.8-flash"]["rpm"])
        assert rl_lite.max_rpm <= int(models_cfg["gemini-3.5-flash-lite"]["rpm"])

        cache = TranslationCache(db_path=tmp_path / "throttle_shared.db")
        key_val = "fake-dry-run-key-001"
        tr_a = GeminiTranslator(api_keys=[key_val], model="gemini-3.8-flash", cache=cache, dry_run=True)
        tr_b = GeminiTranslator(api_keys=[key_val], model="gemini-3.8-flash", cache=cache, dry_run=True)
        tr_lite = GeminiTranslator(api_keys=[key_val], model="gemini-3.5-flash-lite", cache=cache, dry_run=True)
        kid = tr_a.key_ids[0]
        assert tr_a.rate_limiters[kid] is tr_b.rate_limiters[kid]
        assert tr_a.rate_limiters[kid] is not tr_lite.rate_limiters[kid]
        assert tr_a.rate_limiters[kid].max_rpm <= int(models_cfg["gemini-3.8-flash"]["rpm"])
        assert tr_lite.rate_limiters[kid].max_rpm <= int(models_cfg["gemini-3.5-flash-lite"]["rpm"])
    finally:
        clear_rate_limiter_registry()


# Domain E3: Web Studio & P2.3 Operational Scenarios


@pytest.mark.web
def test_e3_web_static_csp_dom_hygiene_and_pareto_multipass_24s(tmp_path: Path) -> None:
    """Verify strict CSP/DOM hygiene, auto-pareto + nvenc_multipass wiring, and 24s Zero-Disk labels."""
    root = _seed_web_workspace(tmp_path / "web_ui")
    spawned: List[List[str]] = []

    def _fake_popen(cmd: List[str], **_kwargs: Any) -> _FakePopenProcess:
        spawned.append(list(cmd))
        return _FakePopenProcess(cmd)

    app = create_app(workspace_root=root, gpu_query_fn=_idle_gpu_metrics, popen_factory=_fake_popen)
    with TestClient(app) as client:
        index_resp = client.get("/")
        assert index_resp.status_code == 200
        csp = index_resp.headers.get("content-security-policy", "")
        assert "'unsafe-inline'" not in csp
        html = index_resp.text

        script_tags = re.findall(r"<script\b([^>]*)>([\s\S]*?)</script>", html, re.IGNORECASE)
        src_scripts = [attrs for attrs, _body in script_tags if re.search(r"\bsrc\s*=", attrs, re.IGNORECASE)]
        inline_scripts = [body for attrs, body in script_tags if not re.search(r"\bsrc\s*=", attrs, re.IGNORECASE)]
        assert len(src_scripts) == 1 and "/static/js/app.js" in src_scripts[0]
        assert all(re.search(r"""\bsrc\s*=\s*["']/static/""", attrs, re.IGNORECASE) for attrs in src_scripts)
        assert len(inline_scripts) == 1
        script_sha = base64.b64encode(hashlib.sha256(inline_scripts[0].encode("utf-8")).digest()).decode("ascii")
        assert f"'sha256-{script_sha}'" in csp
        assert not re.search(r"\sstyle\s*=", html, re.IGNORECASE)
        assert re.search(r"""id\s*=\s*["']opt-nvenc-multipass["']""", html)
        assert not re.search(r"\b8s\b", html)

        js_bodies: Dict[str, str] = {}
        for js_name in ("app.js", "charts.js", "compare_stage.js", "fs_picker.js", "panels.js", "ui.js"):
            js_resp = client.get(f"/static/js/{js_name}")
            assert js_resp.status_code == 200, f"Failed loading {js_name}"
            body = js_resp.text
            js_bodies[js_name] = body
            for sink in (
                ".innerHTML",
                ".outerHTML",
                "insertAdjacentHTML",
                "setHTMLUnsafe",
                "document.write(",
                "eval(",
            ):
                assert sink not in body, f"Forbidden DOM sink {sink} in {js_name}"
            assert not re.search(r"\b8s\b", body), f"Stale 8s chunk string in {js_name}"

        panels_js = js_bodies["panels.js"]
        assert "routing_modes" in panels_js
        assert "nvenc_multipass" in panels_js
        assert "24s" in html
        assert "24s" in js_bodies["app.js"]

        cfg_resp = client.get("/api/config/models-and-shows")
        assert cfg_resp.status_code == 200
        cfg_data = cfg_resp.json()
        assert "auto-pareto" in {m["id"] for m in cfg_data["routing_modes"]}
        assert cfg_data["nvenc_multipass_policies"] == ["auto", "fullres", "qres"]

        ep = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
        start_resp = client.post(
            "/api/jobs/start",
            json={
                "inputs": [str(ep)],
                "model": "auto-pareto",
                "nvenc_multipass": "fullres",
                "dry_run": True,
            },
        )
        assert start_resp.status_code == 200
        cmd = spawned[-1]
        assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "auto-pareto"
        assert "--nvenc-multipass" in cmd and cmd[cmd.index("--nvenc-multipass") + 1] == "fullres"


@pytest.mark.web
def test_e3_web_sse_strict_json_rejects_nan_inf(tmp_path: Path) -> None:
    """Verify SSE snapshot and tick serialize NaN/Inf/-Inf telemetry values as strict RFC 8259 null."""
    root = _seed_web_workspace(tmp_path / "sse_strict")
    live_file = root / "logs" / "live_status.json"
    live_file.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "stage": "upscaling",
                "steady_fps": float("nan"),
                "latest_fps": float("inf"),
                "eta_seconds": float("-inf"),
            },
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    def _nan_gpu_metrics() -> Dict[str, Any]:
        metrics = _idle_gpu_metrics()
        metrics["gpu_util_pct"] = float("nan")
        metrics["temp_c"] = float("inf")
        return metrics

    app = create_app(workspace_root=root, gpu_query_fn=_nan_gpu_metrics)
    with TestClient(app) as client:
        resp = client.get("/api/events?once=1")
    assert resp.status_code == 200
    parsed = _parse_sse_data_line(resp.text)
    assert parsed["pipeline"]["live"]["steady_fps"] is None
    assert parsed["pipeline"]["live"]["latest_fps"] is None
    assert parsed["pipeline"]["live"]["eta_seconds"] is None
    assert parsed["telemetry"]["gpu_util_pct"] is None
    assert parsed["telemetry"]["temp_c"] is None

    async def _check_tick() -> None:
        hub = TelemetryHub(root, gpu_query_fn=_nan_gpu_metrics, history_size=8)
        gen = hub.stream_events(once=False)
        try:
            snap_ev = _parse_sse_data_line(await anext(gen))
            tick_ev = _parse_sse_data_line(await anext(gen))
            assert snap_ev["telemetry"]["gpu_util_pct"] is None
            assert tick_ev["telemetry"]["temp_c"] is None
            assert tick_ev["pipeline"]["live"]["steady_fps"] is None
        finally:
            await gen.aclose()
            await hub.shutdown()

    asyncio.run(_check_tick())


@pytest.mark.web
def test_e3_web_mock_isolation_and_status_idempotency(tmp_path: Path) -> None:
    """Verify --mock mode never writes .graceful_pause.flag and /api/status is idempotent."""
    app = create_app(workspace_root=tmp_path, mock_mode=True)
    with TestClient(app) as client:
        pause_resp = client.post("/api/jobs/pause-graceful")
        assert pause_resp.status_code == 200
        assert not (tmp_path / "logs" / ".graceful_pause.flag").exists()

        lives = [client.get("/api/status").json()["pipeline"]["live"] for _ in range(6)]

    baseline = lives[0]["completed_chunks"]
    for live in lives:
        assert live["completed_chunks"] == baseline
        assert [c["chunk_idx"] for c in live["chunks"]] == list(range(baseline))


@pytest.mark.web
def test_e3_p23_job_log_retention_and_compare_frames_cleanup(tmp_path: Path) -> None:
    """Verify P2.3 web_job_*.log retention (N=20) and per-session scratch/web_compare cleanup."""
    root = _seed_web_workspace(tmp_path / "p23_ops")
    logs_dir = root / "logs"
    base_time = time.time() - 1000.0
    for idx in range(25):
        old_log = logs_dir / f"web_job_old_{idx:02d}.log"
        old_log.write_text(f"old log {idx}\n", encoding="utf-8")
        ts = base_time + idx * 10.0
        os.utime(old_log, (ts, ts))
    latest_log = logs_dir / "latest.log"
    latest_log.write_text("keep latest\n", encoding="utf-8")

    jm = JobManager(root, popen_factory=lambda cmd, **_kw: _FakePopenProcess(cmd))
    ep = root / "anime_season" / "[BD 1080p] Fate_Ep22.mkv"
    jm.start_upscale_job({"inputs": [str(ep)], "dry_run": True})
    deadline = time.perf_counter() + 2.0
    while jm.get_state()["status"] == "running" and time.perf_counter() < deadline:
        time.sleep(0.01)

    remaining_job_logs = sorted(p.name for p in logs_dir.glob("web_job_*.log"))
    assert len(remaining_job_logs) == 20
    for pruned_idx in range(6):
        assert f"web_job_old_{pruned_idx:02d}.log" not in remaining_job_logs
    assert latest_log.read_text(encoding="utf-8") == "keep latest\n"

    up_mkv = root / "out" / "[BD 1080p] Fate_Ep22_4K.mkv"
    up_mkv.write_bytes(b"\x00" * 4096)

    def _fake_extract_frame(_video_path: Path, _ts: float, out_png: Path) -> None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(PNG_1X1_BYTES)

    app = create_app(
        workspace_root=root,
        gpu_query_fn=_idle_gpu_metrics,
        frame_extract_fn=_fake_extract_frame,
    )
    compare_dir = root / "scratch" / "web_compare"
    with TestClient(app) as client:
        cmp_resp = client.post(
            "/api/preview/compare-frames",
            json={
                "source_path": str(ep),
                "upscaled_path": str(up_mkv),
                "timestamp_sec": 1.0,
            },
        )
        assert cmp_resp.status_code == 200
        cmp_data = cmp_resp.json()
        assert client.get(cmp_data["source_frame_url"]).status_code == 200
        assert client.get(cmp_data["upscaled_frame_url"]).status_code == 200
        assert len(list(compare_dir.glob("*.png"))) == 2

    assert list(compare_dir.glob("*.png")) == []
