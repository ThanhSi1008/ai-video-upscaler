"""Run repository tests in a disposable source snapshot, without runtime imports.

Uses the invoking interpreter. Keeps logs under scratch/agent-validation-*.
This is isolation for the reviewed test suite, not a sandbox for untrusted code.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {"src", "subtrans", "tests", "scripts", "models", "glossaries", "docs"}
ROOT_FILES = {"upscale.py", "subtrans.py", "pytest.ini", "requirements.txt", "shows.yaml",
              "AGENTS.md", "GEMINI.md", "CLAUDE.md", "README.md"}
SUFFIXES = {".py", ".yaml", ".yml", ".ini", ".md", ".txt", ".json", ".ass", ".srt"}
GUARD = '''"""Offline guard loaded before test collection in the source snapshot."""
import socket

def deny_network(*args, **kwargs):
    raise RuntimeError("Network disabled by verify_offline.py; use a stub backend")

socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network
socket.getaddrinfo = deny_network
'''


def allowed_source(relative: Path) -> bool:
    """Only source/config/fixtures; never copy production state or credentials."""
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return False
    if any(part.startswith(".") or part in {"__pycache__", "out", "scratch", "models_bench"}
           for part in relative.parts):
        return False
    if len(relative.parts) == 1:
        return relative.name in ROOT_FILES
    return relative.parts[0] in SOURCE_DIRS and relative.suffix.lower() in SUFFIXES


def isolated_environment(environ: dict[str, str]) -> dict[str, str]:
    clean = {key: value for key, value in environ.items()
             if not any(token in key.upper() for token in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
             and key.upper() not in {"GOOGLE_APPLICATION_CREDENTIALS", "PYTHONPATH",
                                     "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}}
    clean.update(CUDA_VISIBLE_DEVICES="-1", RUN_GPU_TESTS="0", PYTHON_DOTENV_DISABLED="1",
                 PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    return clean


def snapshot_sources(root: Path, destination: Path) -> int:
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached",
                             "--others", "--exclude-standard"], check=True, capture_output=True)
    paths = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    count = 0
    for name in paths:
        relative = Path(name)
        if not allowed_source(relative):
            continue
        source = root / relative
        if not source.exists():  # tracked deletion in working tree
            continue
        # Refuse links/junctions escaping the source tree.
        if source.is_symlink() or not source.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Refusing linked source: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count


def main() -> int:
    if len(sys.argv) > 1:
        print("Usage: python scripts/verify_offline.py (runs the full non-GPU suite)")
        return 2
    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="agent-validation-", dir=scratch))
    snapshot = run_dir / "repo"
    snapshot.mkdir()
    count = snapshot_sources(ROOT, snapshot)
    (snapshot / "agent_offline_guard.py").write_text(GUARD, encoding="utf-8")
    env = isolated_environment(dict(os.environ))
    # Pytest may inspect Git for tests of the context helper; snapshot has no parent repository.
    env["GIT_CEILING_DIRECTORIES"] = str(run_dir)
    command = [sys.executable, "-m", "pytest", "-m", "not gpu", "-q", "--tb=short",
               "-p", "agent_offline_guard", "--basetemp", str(run_dir / "tmp"),
               "--junitxml", str(run_dir / "junit.xml")]
    print(f"Snapshot: {snapshot}\nSource files: {count}\nCPU only; Python sockets blocked; no production cache/weights copied.", flush=True)
    with (run_dir / "pytest.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=snapshot, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace") as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
    print(f"Evidence: {run_dir}\nPytest exit code: {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
