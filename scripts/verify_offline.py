"""Run repository tests in a disposable source snapshot, without runtime imports.

Uses the invoking interpreter. Keeps logs under scratch/agent-validation-*.
This is isolation for the reviewed test suite, not a sandbox for untrusted code.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {"src", "subtrans", "tests", "scripts", "models", "glossaries", "docs", "web"}
SAFE_ROOT_DOTFILES = {".antigravityignore", ".gitignore", ".editorconfig", ".env.example"}
ROOT_FILES = {
    "upscale.py", "upscale.bat", "upscale_turbo.bat", "subtrans.py", "subtrans.bat",
    "web_gui.py", "web_gui.bat", "pytest.ini", "ruff.toml", "requirements.txt",
    "shows.yaml", "GEMINI.md", "README.md",
}
SUFFIXES = {
    ".py", ".bat", ".yaml", ".yml", ".toml", ".ini", ".md", ".txt", ".json",
    ".ass", ".srt", ".html", ".css", ".js", ".svg",
}
GUARD = '''"""Offline guard loaded before test collection in the source snapshot."""
import _socket
import socket
import sys

def deny_network(*args, **kwargs):
    if hasattr(socket, "socketpair") and sys._getframe(1).f_code is socket.socketpair.__code__:
        return _socket.socket.connect(*args, **kwargs)
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
    if len(relative.parts) == 1 and relative.name in SAFE_ROOT_DOTFILES:
        return True
    if relative.parts[0] == ".agents":
        return (
            not any(part in {"__pycache__", "out", "scratch"} for part in relative.parts)
            and relative.suffix.lower() in SUFFIXES | {".svg"}
        )
    if any(part.startswith(".") or part in {"__pycache__", "out", "scratch", "models_bench"}
           for part in relative.parts):
        return False
    if len(relative.parts) == 1:
        return relative.name in ROOT_FILES
    return relative.parts[0] in SOURCE_DIRS and relative.suffix.lower() in SUFFIXES


def isolated_environment(
    environ: dict[str, str],
    tmp_dir: Path | str | None = None,
) -> dict[str, str]:
    clean = {key: value for key, value in environ.items()
             if not any(token in key.upper() for token in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
             and key.upper() not in {"GOOGLE_APPLICATION_CREDENTIALS", "PYTHONPATH",
                                     "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
                                     "TEMP", "TMP", "TMPDIR"}}
    resolved_tmp = str(Path(tmp_dir) if tmp_dir is not None else (ROOT / "scratch" / "tmp"))
    clean.update(CUDA_VISIBLE_DEVICES="-1", RUN_GPU_TESTS="0", PYTHON_DOTENV_DISABLED="1",
                 SUBTRANS_DISABLE_DOTENV="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                 PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                 TEMP=resolved_tmp, TMP=resolved_tmp, TMPDIR=resolved_tmp)
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


def _rmtree_force(path: Path) -> None:
    """Force-remove a temporary directory including read-only files (e.g., .git/objects on Windows)."""
    if not path.exists():
        return

    def _clear_readonly(func, target, _exc_info):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_clear_readonly)


def _detect_dirty_paths(root: Path) -> list[str]:
    """Return modified and untracked repository-relative POSIX paths."""
    paths: set[str] = set()
    diff_res = subprocess.run(
        ["git", "--no-optional-locks", "diff", "--name-only", "HEAD"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if diff_res.returncode == 0:
        for ln in diff_res.stdout.splitlines():
            p = ln.strip().replace("\\", "/")
            if p:
                paths.add(p)
    st_res = subprocess.run(
        ["git", "--no-optional-locks", "status", "--porcelain"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if st_res.returncode == 0:
        for raw_line in st_res.stdout.splitlines():
            if len(raw_line) < 4:
                continue
            status = raw_line[:2]
            if "D" in status:
                continue
            rel = raw_line[3:].strip()
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1].strip()
            rel = rel.strip('"').replace("\\", "/")
            if rel:
                paths.add(rel)
    return sorted(paths)


def resolve_impacted_tests(root: Path, changed_paths: list[str] | None = None) -> list[str]:
    """Resolve impacted test files/directories from modified paths via agent_context 2-hop + contracts."""
    dirty = changed_paths if changed_paths is not None else _detect_dirty_paths(root)
    if not dirty:
        return []
    root_str = str(root)
    added_sys_path = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added_sys_path = True
    try:
        from scripts.agent_context import compute_impact

        _, impacted = compute_impact(root, dirty)
    finally:
        if added_sys_path and sys.path and sys.path[0] == root_str:
            sys.path.pop(0)

    candidates: set[str] = set(impacted)
    for rel in dirty:
        norm = rel.replace("\\", "/")
        if (norm.startswith("tests/") or norm.startswith("subtrans/tests/")) and norm.endswith(".py"):
            if Path(norm).name.startswith("test_") and (root / norm).is_file():
                candidates.add(norm)

    # Deduplicate files already covered by an impacted directory (e.g. subtrans/tests/)
    dirs = [c.rstrip("/") + "/" for c in candidates if c.endswith("/") or (root / c).is_dir()]
    resolved: list[str] = []
    for cand in sorted(candidates):
        norm = cand.rstrip("/") + "/" if (root / cand).is_dir() else cand
        if any(norm != d and norm.startswith(d) for d in dirs):
            continue
        if (root / cand).exists():
            resolved.append(cand)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--target",
        nargs="+",
        metavar="PATH_OR_NODEID",
        help="Run only specified test file(s) or pytest node ID(s) inside the isolated snapshot",
    )
    mode_group.add_argument(
        "--impacted",
        action="store_true",
        help="Run only tests impacted by current working-tree modifications/untracked files",
    )
    parser.add_argument(
        "--maxfail",
        type=int,
        default=None,
        help="Stop after N test failures (defaults to 3 in --impacted mode)",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Enable pytest failure stdout/stderr capture (overrides default --show-capture=no)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    is_fast_mode = bool(args.target or args.impacted)
    test_targets: list[str] = []
    if args.target:
        test_targets = list(args.target)
    elif args.impacted:
        dirty = _detect_dirty_paths(ROOT)
        if not dirty:
            print("No modified or untracked files detected; 0 impacted tests.")
            return 0
        test_targets = resolve_impacted_tests(ROOT, dirty)
        if not test_targets:
            print("No impacted tests matched modified files; skipping pytest.")
            return 0
        print(f"Impacted targets ({len(test_targets)}): {', '.join(test_targets)}")

    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="agent-validation-", dir=scratch))
    snapshot = run_dir / "repo"
    snapshot.mkdir()
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    code = 1
    try:
        count = snapshot_sources(ROOT, snapshot)
        (snapshot / "agent_offline_guard.py").write_text(GUARD, encoding="utf-8")
        env = isolated_environment(dict(os.environ), tmp_dir=tmp_dir)
        env["TEMP"] = env["TMP"] = env["TMPDIR"] = str(tmp_dir)
        # Pytest may inspect Git for tests of the context helper; snapshot has no parent repository.
        env["GIT_CEILING_DIRECTORIES"] = str(run_dir)

        command = [
            sys.executable, "-m", "pytest", "-m", "not gpu", "-q", "--tb=short",
            "-p", "agent_offline_guard", "--basetemp", str(tmp_dir / "pytest"),
            "--junitxml", str(run_dir / "junit.xml"),
        ]
        if not args.capture:
            command.append("--show-capture=no")
        maxfail = args.maxfail if args.maxfail is not None else (3 if args.impacted else None)
        if maxfail is not None and maxfail > 0:
            command.extend(["--maxfail", str(maxfail)])
        if test_targets:
            command.extend(test_targets)

        print(
            f"Snapshot: {snapshot}\nSource files: {count}\n"
            "CPU only; Python sockets blocked; no production cache/weights copied.",
            flush=True,
        )
        with (run_dir / "pytest.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(
                command, cwd=snapshot, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace",
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
        print(f"Evidence: {run_dir}\nPytest exit code: {code}")
    finally:
        if is_fast_mode or code == 0:
            # Always clean heavy snapshot & pytest temp files on fast modes and passing full runs
            _rmtree_force(snapshot)
            _rmtree_force(tmp_dir / "pytest")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            if code == 0:
                validation_dirs = sorted(
                    (p for p in scratch.glob("agent-validation-*") if p.is_dir()),
                    key=lambda p: (p.stat().st_mtime_ns, p.name),
                    reverse=True,
                )
                for old_dir in validation_dirs[2:]:
                    if old_dir != run_dir:
                        _rmtree_force(old_dir)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
