"""Safety contracts for source-snapshot validation (no production imports)."""
from pathlib import Path
import subprocess

import pytest

from scripts.verify_offline import allowed_source, isolated_environment, snapshot_sources, GUARD


@pytest.mark.parametrize("name", [".env", "subtrans/.env", "subtrans/.cache/knowledge.db",
                                 "out/episode.srt", ".temp_episode/manifest.json",
                                 "pending_subs.json", "models_bench/model.pth", "../src/a.py",
                                 "src/__pycache__/a.py", "samples/video.mkv"])
def test_private_runtime_files_are_excluded(name):
    assert not allowed_source(Path(name))


@pytest.mark.parametrize("name", ["src/config.py", "subtrans/models.yaml",
                                 "subtrans/samples/sample_30s_jpn.ass", "pytest.ini",
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
    subprocess.run(["git", "-C", str(root), "add", "src/example.py"], check=True)
    source.write_text("after", encoding="utf-8")
    (root / "src" / "new.py").write_text("new", encoding="utf-8")
    (root / ".env").write_text("never copy", encoding="utf-8")
    (root / "pending_subs.json").write_text("[]", encoding="utf-8")
    assert snapshot_sources(root, dest) == 2
    assert (dest / "src" / "example.py").read_text(encoding="utf-8") == "after"
    assert not (dest / ".env").exists()
    assert not (dest / "pending_subs.json").exists()


def test_guard_rejects_network_before_connection(monkeypatch):
    import socket
    # Register restoration before executing the standalone pytest plugin source.
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, getattr(socket.socket, name))
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, getattr(socket, name))
    exec(GUARD, {})
    with pytest.raises(RuntimeError, match="Network disabled"):
        socket.create_connection(("example.invalid", 443))
