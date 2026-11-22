"""Safety contracts for source-snapshot validation (no production imports)."""
import io
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import verify_offline
from scripts.verify_offline import allowed_source, isolated_environment, resolve_impacted_tests, snapshot_sources, GUARD


@pytest.mark.parametrize("name", [".env", "subtrans/.env", "subtrans/.cache/knowledge.db",
                                 "out/episode.srt", ".temp_episode/manifest.json",
                                 "pending_subs.json", "state/pending_subs.json",
                                 "models_bench/model.pth", "../src/a.py",
                                 "src/__pycache__/a.py", "samples/video.mkv"])
def test_private_runtime_files_are_excluded(name):
    assert not allowed_source(Path(name))


@pytest.mark.parametrize("name", ["src/config.py", "subtrans/models.yaml",
                                 "subtrans/samples/sample_30s_jpn.ass", "pytest.ini",
                                 "ruff.toml", ".antigravityignore", ".gitignore", ".env.example",
                                 "tests/test_verify_offline.py", "docs/agents/STATE.md"])
def test_source_and_fixtures_are_included(name):
    assert allowed_source(Path(name))


def test_child_environment_drops_credentials_without_mutating_parent():
    parent = {"GEMINI_API_KEY_2": "private", "GOOGLE_API_KEY": "private",
              "GOOGLE_APPLICATION_CREDENTIALS": "credentials.json", "PATH": "tools",
              "PYTHONPATH": "live-repo", "PYTEST_ADDOPTS": "-m gpu", "RUN_GPU_TESTS": "1"}
    child = isolated_environment(parent)
    assert parent["GEMINI_API_KEY_2"] == "private"
    assert not any("KEY" in key for key in child)
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in child
    assert "PYTHONPATH" not in child and "PYTEST_ADDOPTS" not in child
    assert child["CUDA_VISIBLE_DEVICES"] == "-1" and child["RUN_GPU_TESTS"] == "0"
    assert child["OMP_NUM_THREADS"] == "1"
    assert child["MKL_NUM_THREADS"] == "1"
    assert child["OPENBLAS_NUM_THREADS"] == "1"
    assert child["NUMEXPR_NUM_THREADS"] == "1"
    assert child["PATH"] == "tools"


def test_snapshot_copies_working_changes_and_untracked_source_not_state(tmp_path):
    root, dest = tmp_path / "source", tmp_path / "copy"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "src").mkdir()
    source = root / "src" / "example.py"
    source.write_text("before", encoding="utf-8")
    (root / "ruff.toml").write_text("[lint]\n", encoding="utf-8")
    (root / ".antigravityignore").write_text("out/\n", encoding="utf-8")
    (root / ".gitignore").write_text(".env\n", encoding="utf-8")
    (root / ".env.example").write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "src/example.py", "ruff.toml", ".antigravityignore", ".gitignore", ".env.example"], check=True)
    source.write_text("after", encoding="utf-8")
    (root / "src" / "new.py").write_text("new", encoding="utf-8")
    (root / ".env").write_text("never copy", encoding="utf-8")
    (root / "subtrans").mkdir()
    (root / "subtrans" / ".env").write_text("never copy", encoding="utf-8")
    (root / "pending_subs.json").write_text("[]", encoding="utf-8")
    assert snapshot_sources(root, dest) == 6
    assert (dest / "src" / "example.py").read_text(encoding="utf-8") == "after"
    assert (dest / "ruff.toml").is_file()
    assert (dest / ".antigravityignore").is_file()
    assert (dest / ".env.example").is_file()
    assert not (dest / ".env").exists()
    assert not (dest / "subtrans" / ".env").exists()
    assert not (dest / "pending_subs.json").exists()


def test_resolve_impacted_tests_uses_2hop_and_cross_boundary_contracts(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "web").mkdir()
    (tmp_path / "tests" / "e2e").mkdir(parents=True)
    (tmp_path / "src/logger.py").write_text("def log_msg():\n    pass\n", encoding="utf-8")
    (tmp_path / "web/sse_hub.py").write_text("from src import logger\n", encoding="utf-8")
    (tmp_path / "web/app.py").write_text("import web.sse_hub\n", encoding="utf-8")
    (tmp_path / "tests/test_web_gui.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (tmp_path / "tests/test_edited.py").write_text("def test_ed(): pass\n", encoding="utf-8")
    (tmp_path / "tests/e2e/conftest.py").write_text("import src.logger\n", encoding="utf-8")

    impacted = resolve_impacted_tests(tmp_path, ["src/logger.py", "tests/test_edited.py", "tests/e2e/conftest.py"])
    assert "tests/test_web_gui.py" in impacted
    assert "tests/test_edited.py" in impacted
    assert "tests/e2e/" in impacted
    assert not any(p.endswith("conftest.py") for p in impacted)


def test_verify_offline_target_impacted_flags_and_cleanup(tmp_path, monkeypatch, capsys):
    # Initialize an isolated git repository because verify_offline sets GIT_CEILING_DIRECTORIES
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    # Minimal source tree for snapshot_sources
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests/test_mod.py").write_text("import src.mod\n", encoding="utf-8")
    monkeypatch.setattr(verify_offline, "ROOT", tmp_path)

    # 1. Clean working tree + --impacted short-circuits cleanly
    monkeypatch.setattr(verify_offline, "_detect_dirty_paths", lambda root: [])
    assert verify_offline.main(["--impacted"]) == 0
    assert "0 impacted tests" in capsys.readouterr().out

    recorded: list[tuple[list[str], Path]] = []
    orig_popen = subprocess.Popen

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            self._real = None
            cmd_list = list(cmd)
            if "-m" in cmd_list and "pytest" in cmd_list:
                recorded.append((cmd_list, Path(kwargs["cwd"])))
                self.stdout = iter(["1 passed\n"])
            else:
                self._real = orig_popen(cmd, **kwargs)
                self.stdout = self._real.stdout

        def __enter__(self):
            return self._real.__enter__() if self._real is not None else self

        def __exit__(self, *args):
            return self._real.__exit__(*args) if self._real is not None else False

        def poll(self):
            return self._real.poll() if self._real is not None else 0

        def communicate(self, *args, **kwargs):
            return self._real.communicate(*args, **kwargs)

        def wait(self, timeout=None):
            if self._real is not None:
                return self._real.wait(timeout=timeout)
            return 1 if "--fail-sim" in recorded[-1][0] else 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    # 2. --impacted sets default --maxfail 3 and --show-capture=no, then custom overrides apply
    monkeypatch.setattr(verify_offline, "_detect_dirty_paths", lambda root: ["src/mod.py"])
    assert verify_offline.main(["--impacted"]) == 0
    cmd1, snap1 = recorded[-1]
    assert "--show-capture=no" in cmd1
    assert "--maxfail" in cmd1 and cmd1[cmd1.index("--maxfail") + 1] == "3"
    assert "tests/test_mod.py" in cmd1
    assert not snap1.exists()

    assert verify_offline.main(["--impacted", "--maxfail", "5", "--capture"]) == 0
    cmd2, _ = recorded[-1]
    assert "--show-capture=no" not in cmd2
    assert cmd2[cmd2.index("--maxfail") + 1] == "5"

    # 3. --target cleans heavy repo/ snapshot even on failure
    class FailingPopen(FakePopen):
        def wait(self):
            return 1

    monkeypatch.setattr(subprocess, "Popen", FailingPopen)
    assert verify_offline.main(["--target", "tests/test_mod.py"]) == 1
    _, snap_target_fail = recorded[-1]
    assert not snap_target_fail.exists()

    # 4. Full run preserves repo/ snapshot for failure triage
    assert verify_offline.main([]) == 1
    _, snap_full_fail = recorded[-1]
    assert snap_full_fail.exists()


def test_guard_rejects_network_before_connection(tmp_path, monkeypatch):
    import socket
    # Register restoration before executing the standalone pytest plugin source.
    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, getattr(socket.socket, name))
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, getattr(socket, name))
    exec(GUARD, {})
    with pytest.raises(RuntimeError, match="Network disabled"):
        socket.create_connection(("example.invalid", 443))

    # Child Python subprocesses also inherit the guard via sitecustomize.py on PYTHONPATH.
    guard_dir = tmp_path / "guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(GUARD, encoding="utf-8")
    child_env = isolated_environment(dict(os.environ), tmp_dir=tmp_path)
    child_env["PYTHONPATH"] = str(guard_dir)
    res = subprocess.run(
        [sys.executable, "-c", "import socket; socket.create_connection(('example.invalid', 443))"],
        capture_output=True,
        text=True,
        env=child_env,
    )
    assert res.returncode != 0
    assert "Network disabled by verify_offline.py" in (res.stderr or "")


def test_partition_test_shards_covers_all_files_without_shard0_pinning():
    """Verify 6-worker LPT sharding covers 100% of test files with no duplicates or 31-file Shard-0 bottleneck."""
    import ast

    repo_root = Path(__file__).resolve().parents[1]
    shards = verify_offline._partition_test_shards(repo_root, 6)
    assert len(shards) == 6
    assert all(len(s) > 0 for s in shards)

    all_targets = [t for s in shards for t in s]
    assert len(all_targets) == len(set(all_targets))

    expected_files = sorted(
        p.relative_to(repo_root).as_posix()
        for base in ("tests", "subtrans/tests")
        for p in (repo_root / base).rglob("test_*.py")
    )
    covered_files = sorted({t.split("::", 1)[0] for t in all_targets})
    assert covered_files == expected_files

    for split_rel, parts in verify_offline._SPLIT_MONOLITH_PARTS.items():
        fn_count = sum(
            1
            for n in ast.parse((repo_root / split_rel).read_text(encoding="utf-8")).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
        )
        nodeids = [t for t in all_targets if t.startswith(f"{split_rel}::")]
        assert len(nodeids) == fn_count
        if split_rel in verify_offline._SPLIT_MONOLITH_SLICE_WEIGHTS_MS:
            weights = verify_offline._SPLIT_MONOLITH_SLICE_WEIGHTS_MS[split_rel]
            assert len(weights) == parts
            assert all(w > 0 for w in weights)
    assert set(verify_offline._SPLIT_MONOLITH_SLICE_WEIGHTS_MS) <= set(verify_offline._SPLIT_MONOLITH_PARTS)

    shard0_files = {t.split("::", 1)[0] for t in shards[0]}
    assert len(shard0_files) < 31
    assert any(not f.startswith("subtrans/tests/") for f in shard0_files) or any(
        not all(t.startswith("subtrans/tests/") for t in s) for s in shards
    )


def test_sharded_run_assigns_distinct_cwd_and_temp_per_worker(tmp_path, monkeypatch):
    """Verify every shard in a 6-worker run receives a unique working directory and TEMP dir."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for rel_dir in ("tests", "subtrans/tests"):
        (tmp_path / rel_dir).mkdir(parents=True, exist_ok=True)
    for i in range(6):
        (tmp_path / "tests" / f"test_u{i}.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    for i in range(4):
        (tmp_path / "subtrans/tests" / f"test_s{i}.py").write_text("def test_ok(): pass\n", encoding="utf-8")

    monkeypatch.setattr(verify_offline, "ROOT", tmp_path)
    recorded_cwds: list[Path] = []
    recorded_temps: list[str] = []
    orig_popen = subprocess.Popen

    class FakeShardPopen:
        def __init__(self, cmd, **kwargs):
            self._real = None
            cmd_list = list(cmd)
            if "-m" in cmd_list and "pytest" in cmd_list:
                recorded_cwds.append(Path(kwargs["cwd"]))
                recorded_temps.append(kwargs["env"]["TEMP"])
                junit_path = Path(cmd_list[cmd_list.index("--junitxml") + 1])
                junit_path.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>', encoding="utf-8")
                self.stdout = io.StringIO("1 passed\n")
            else:
                self._real = orig_popen(cmd, **kwargs)
                self.stdout = self._real.stdout

        def __enter__(self):
            return self._real.__enter__() if self._real is not None else self

        def __exit__(self, *args):
            return self._real.__exit__(*args) if self._real is not None else False

        def poll(self):
            return self._real.poll() if self._real is not None else 0

        def communicate(self, *args, **kwargs):
            return self._real.communicate(*args, **kwargs)

        def wait(self, timeout=None):
            if self._real is not None:
                return self._real.wait(timeout=timeout)
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakeShardPopen)
    assert verify_offline.main(["--workers", "6"]) == 0
    assert len(recorded_cwds) == 6
    assert len(set(recorded_cwds)) == 6
    assert len(set(recorded_temps)) == 6
    assert any(p.name == "repo" for p in recorded_cwds)
    assert sum(1 for p in recorded_cwds if p.name.startswith("repo_w")) == 5


def test_is_pid_alive_handles_live_exited_handle_and_invalid_pids():
    """Verify _is_pid_alive distinguishes live, zombie-handle exited, and invalid PIDs."""
    assert verify_offline._is_pid_alive(os.getpid()) is True
    assert verify_offline._is_pid_alive(0) is False
    assert verify_offline._is_pid_alive(-1) is False

    dead_proc = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_proc.wait(timeout=10)
    assert verify_offline._is_pid_alive(dead_proc.pid) is False
    dead_pid = dead_proc.pid
    del dead_proc
    assert verify_offline._is_pid_alive(dead_pid) is False

    live_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert verify_offline._is_pid_alive(live_proc.pid) is True
    finally:
        live_proc.terminate()
        live_proc.wait(timeout=10)
    assert verify_offline._is_pid_alive(live_proc.pid) is False


def test_verify_offline_retention_skips_active_pid_and_prunes_stale_dirs(tmp_path, monkeypatch):
    """Verify retention keeps active-PID dirs across >=3 runs and always cleans .active.pid."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests/test_mod.py").write_text("import src.mod\n", encoding="utf-8")
    monkeypatch.setattr(verify_offline, "ROOT", tmp_path)

    dead_proc = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_proc.wait(timeout=10)
    dead_pid = dead_proc.pid
    del dead_proc

    live_1 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    live_2 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        scratch = tmp_path / "scratch"
        scratch.mkdir()

        def _make_dir(name: str, mtime: float, pid_text: str | None = None) -> Path:
            d = scratch / name
            d.mkdir()
            if pid_text is not None:
                (d / ".active.pid").write_text(pid_text, encoding="utf-8")
            os.utime(d, (mtime, mtime))
            return d

        old_absent = _make_dir("agent-validation-old-absent", 100.0)
        old_corrupt = _make_dir("agent-validation-old-corrupt", 200.0, "corrupt\n")
        old_dead = _make_dir("agent-validation-old-dead", 300.0, f"{dead_pid}\n")
        active_1 = _make_dir("agent-validation-active-1", 400.0, f"{live_1.pid}\n")
        active_2 = _make_dir("agent-validation-active-2", 500.0, f"{live_2.pid}\n")
        recent_done = _make_dir("agent-validation-recent-done", 600.0)

        orig_popen = subprocess.Popen
        seen_run_dirs: list[Path] = []

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                self._real = None
                cmd_list = list(cmd)
                if "-m" in cmd_list and "pytest" in cmd_list:
                    run_dir = Path(kwargs["cwd"]).parent
                    seen_run_dirs.append(run_dir)
                    assert (run_dir / ".active.pid").read_text(encoding="utf-8").strip() == str(os.getpid())
                    self.stdout = iter(["1 passed\n"])
                else:
                    self._real = orig_popen(cmd, **kwargs)
                    self.stdout = self._real.stdout

            def __enter__(self):
                return self._real.__enter__() if self._real is not None else self

            def __exit__(self, *args):
                return self._real.__exit__(*args) if self._real is not None else False

            def wait(self, timeout=None):
                return self._real.wait(timeout=timeout) if self._real is not None else 0

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        assert verify_offline.main(["--target", "tests/test_mod.py"]) == 0
        assert len(seen_run_dirs) == 1
        new_run_dir = seen_run_dirs[0]

        assert active_1.is_dir()
        assert active_2.is_dir()
        assert recent_done.is_dir()
        assert new_run_dir.is_dir()
        assert not (new_run_dir / ".active.pid").exists()
        assert not old_absent.exists()
        assert not old_corrupt.exists()
        assert not old_dead.exists()

        before_abort = set(scratch.glob("agent-validation-*"))

        class AbortPopen:
            def __init__(self, cmd, **kwargs):
                cmd_list = list(cmd)
                if "-m" in cmd_list and "pytest" in cmd_list:
                    raise KeyboardInterrupt("simulated interrupt")
                self._real = orig_popen(cmd, **kwargs)
                self.stdout = self._real.stdout

            def __enter__(self):
                return self._real.__enter__()

            def __exit__(self, *args):
                return self._real.__exit__(*args)

            def wait(self, timeout=None):
                return self._real.wait(timeout=timeout)

        monkeypatch.setattr(subprocess, "Popen", AbortPopen)
        with pytest.raises(KeyboardInterrupt):
            verify_offline.main(["--target", "tests/test_mod.py"])

        aborted_dirs = set(scratch.glob("agent-validation-*")) - before_abort
        assert len(aborted_dirs) == 1
        aborted_dir = next(iter(aborted_dirs))
        assert not (aborted_dir / ".active.pid").exists()
        assert not (aborted_dir / ".active.pid.tmp").exists()
    finally:
        for proc in (live_1, live_2):
            proc.terminate()
            proc.wait(timeout=10)
