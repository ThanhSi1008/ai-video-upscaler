"""Safety contracts for source-snapshot validation (no production imports)."""
from pathlib import Path
import subprocess

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
    import os
    import socket
    import sys
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
