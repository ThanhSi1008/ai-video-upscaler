"""The bootstrap must remain offline, bounded and useful after a fresh clone."""

from pathlib import Path
import subprocess

import pytest

from scripts import agent_context


def make_memory(root: Path) -> None:
    for relative in agent_context.REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Fixture\n", encoding="utf-8")


def test_valid_memory_accepts_relative_links_and_ignores_remote_and_code(tmp_path):
    make_memory(tmp_path)
    (tmp_path / "docs/agents/STATE.md").write_text(
        "[rules](../../GEMINI.md#rules)\n[map](PROJECT_MAP.md)\n"
        "[remote](https://example.invalid/not-fetched)\n"
        "```md\n[example](missing-example.md)\n```\n",
        encoding="utf-8",
    )
    # Historical research docs are outside the active memory validation scope.
    (tmp_path / "docs/old.md").write_text("[old](gone.md)", encoding="utf-8")
    assert agent_context.check_memory(tmp_path) == []


def test_check_reports_missing_required_file_and_broken_target(tmp_path):
    make_memory(tmp_path)
    (tmp_path / "GEMINI.md").unlink()
    (tmp_path / "docs/agents/STATE.md").write_text("[next](missing.md)", encoding="utf-8")
    errors = agent_context.check_memory(tmp_path)
    assert any("required file: GEMINI.md" in e for e in errors)
    assert any("STATE.md:1: missing local target: missing.md" in e for e in errors)


def test_check_rejects_links_outside_repo_and_nonportable_file_urls(tmp_path):
    make_memory(tmp_path)
    (tmp_path / "GEMINI.md").write_text(
        "[outside](../outside.md)\n[absolute](file:///C:/outside.md)", encoding="utf-8"
    )
    errors = agent_context.check_memory(tmp_path)
    assert any("link leaves repository" in e for e in errors)
    assert any("nonportable local link" in e for e in errors)


def test_byte_budgets_and_line_limits_use_utf8_and_combined_scope(tmp_path):
    make_memory(tmp_path)
    (tmp_path / "GEMINI.md").write_text("ệ" * (agent_context.ROOT_BUDGET // 3 + 1), encoding="utf-8")
    (tmp_path / agent_context.SUBTRANS_SKILL_REL).write_text("x" * agent_context.COMBINED_BUDGET, encoding="utf-8")
    errors = agent_context.check_memory(tmp_path)
    assert any("GEMINI.md exceeds" in e and "bytes" in e for e in errors)
    assert any("Root + subtrans skill exceed" in e for e in errors)

    # Line limit enforcement (<= MAX_ROOT_LINES)
    make_memory(tmp_path)
    (tmp_path / "GEMINI.md").write_text("\n".join(["line"] * (agent_context.MAX_ROOT_LINES + 1)), encoding="utf-8")
    line_errors = agent_context.check_memory(tmp_path)
    assert any(f"GEMINI.md exceeds {agent_context.MAX_ROOT_LINES} lines" in e for e in line_errors)


def test_bootstrap_reads_only_curated_memory_and_git_metadata(tmp_path, monkeypatch, capsys):
    make_memory(tmp_path)
    (tmp_path / ".env").write_text("SECRET_SENTINEL", encoding="utf-8")
    (tmp_path / "docs/agents/PROJECT_MAP.md").write_text("MAP_SENTINEL", encoding="utf-8")
    calls = []

    def fake_git(root, *args):
        calls.append(args)
        return {("branch", "--show-current"): "main", ("rev-parse", "--short", "HEAD"): "abc1234", ("status", "--short"): ""}[args]

    monkeypatch.setattr(agent_context, "git_output", fake_git)
    assert agent_context.main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Branch: main" in out and "HEAD: abc1234" in out and "(clean)" in out
    assert "SECRET_SENTINEL" not in out and "MAP_SENTINEL" not in out
    assert len(calls) == 3


def test_inventory_only_counts_tracked_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent_context, "git_output", lambda *a: "README.md\0src/a.py\0src/b.py\0docs/a.md\0")
    assert agent_context.main(["--root", str(tmp_path), "--inventory"]) == 0
    out = capsys.readouterr().out
    assert "Tracked paths: 4" in out and "src: 2" in out
    assert "a.py" not in out


def test_git_unavailable_is_reported_without_shell_or_crash(tmp_path, monkeypatch, capsys):
    def missing_git(*args, **kwargs):
        assert kwargs.get("shell", False) is False
        assert args[0][0:2] == ["git", "--no-optional-locks"]
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(subprocess, "run", missing_git)
    assert agent_context.main(["--root", str(tmp_path), "--inventory"]) == 1
    assert "Inventory unavailable" in capsys.readouterr().out


def test_check_exit_code_is_nonzero_for_incomplete_memory(tmp_path, capsys):
    assert agent_context.main(["--root", str(tmp_path), "--check"]) == 1
    assert "ERROR:" in capsys.readouterr().out


def test_symbols_extracts_ast_classes_and_functions(tmp_path, capsys):
    sample_file = tmp_path / "sample.py"
    sample_file.write_text(
        '"""Module docstring."""\n\n'
        'class Alpha:\n'
        '    def beta(self, x: int) -> str:\n'
        '        return str(x)\n\n'
        'def gamma(val: bool = True) -> None:\n'
        '    pass\n',
        encoding="utf-8",
    )
    assert agent_context.main(["--root", str(tmp_path), "--symbols", "sample.py"]) == 0
    out = capsys.readouterr().out
    assert "class Alpha: [L3-L5]" in out
    assert "def beta(self, x: int) -> str: ... [L4-L5]" in out
    assert "def gamma(val: bool=True) -> None: ... [L7-L8]" in out


def test_find_symbol_locates_definitions_across_files(tmp_path, capsys):
    sample_dir = tmp_path / "src"
    sample_dir.mkdir()
    (sample_dir / "worker.py").write_text(
        "class ProcessingEngine:\n"
        "    def run_pipeline(self, mode: str) -> bool:\n"
        "        return True\n",
        encoding="utf-8",
    )
    assert agent_context.main(["--root", str(tmp_path), "--find", "run_pipeline"]) == 0
    out = capsys.readouterr().out
    assert "Found 1 symbol definition(s) matching 'run_pipeline':" in out
    assert "src/worker.py:2:   def run_pipeline(self, mode: str) -> bool:" in out


def test_install_git_hooks_creates_pre_commit_hook(tmp_path, capsys):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    hooks_dir = tmp_path / ".git" / "hooks"
    assert agent_context.main(["--root", str(tmp_path), "--install-hooks"]) == 0
    out = capsys.readouterr().out
    assert "Pre-commit hook successfully installed" in out
    hook_file = hooks_dir / "pre-commit"
    assert hook_file.is_file()
    assert "python scripts/agent_context.py --check" in hook_file.read_text(encoding="utf-8")


def test_ast_prunes_private_trees_and_rejects_explicit_private_targets(tmp_path, monkeypatch, capsys):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/worker.py").write_text("def visible(): pass\n", encoding="utf-8")
    private = [".venv/library.py", "venv/library.py", "out/result.py",
               "subtrans/.cache/cache.py", "src/.temp_episode/checkpoint.py"]
    for relative in private:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def private_sentinel(): pass\n", encoding="utf-8")
    walked = []
    real_walk = agent_context.os.walk

    def observe_walk(*args, **kwargs):
        for entry in real_walk(*args, **kwargs):
            walked.append(Path(entry[0]).relative_to(tmp_path).as_posix())
            yield entry

    monkeypatch.setattr(agent_context.os, "walk", observe_walk)
    assert agent_context.find_symbol(tmp_path, "visible")
    assert not any(name.startswith((".venv", "venv", "out", "subtrans/.cache", "src/.temp")) for name in walked)
    for relative in private:
        assert not agent_context.show_symbols(tmp_path, relative)
    assert "private_sentinel" not in capsys.readouterr().out


def test_ast_rejects_source_symlink_to_outside(tmp_path, capsys):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("def external_sentinel(): pass\n", encoding="utf-8")
    linked = root / "src/linked.py"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer mode or privilege")
    assert not agent_context.show_symbols(root, "src/linked.py")
    assert not agent_context.find_symbol(root, "external_sentinel")
    assert "def external_sentinel" not in capsys.readouterr().out


def test_existing_custom_and_modified_managed_hooks_are_preserved(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    hook = tmp_path / ".git/hooks/pre-commit"
    for contents in (b"#!/bin/sh\necho custom\n", agent_context.MANAGED_HOOK.encode() + b"echo custom\n"):
        hook.write_bytes(contents)
        assert not agent_context.install_git_hooks(tmp_path)
        assert hook.read_bytes() == contents


def test_managed_hook_is_idempotent_for_original_windows_newlines(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    hook = tmp_path / ".git/hooks/pre-commit"
    original = agent_context.MANAGED_HOOK.replace("\n", "\r\n").encode("utf-8")
    hook.write_bytes(original)
    before = hook.stat().st_mtime_ns
    assert agent_context.install_git_hooks(tmp_path)
    assert hook.read_bytes() == original
    assert hook.stat().st_mtime_ns == before


def test_custom_hooks_directory_outside_git_is_not_written(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "core.hooksPath", ".githooks"], check=True)
    assert not agent_context.install_git_hooks(tmp_path)
    assert not (tmp_path / ".githooks").exists()


def test_worktree_uses_common_git_hooks_directory(tmp_path):
    main, worktree = tmp_path / "main", tmp_path / "worktree"
    subprocess.run(["git", "init", "-q", str(main)], check=True)
    subprocess.run(["git", "-C", str(main), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "--allow-empty", "-qm", "fixture"], check=True)
    subprocess.run(["git", "-C", str(main), "worktree", "add", "--detach", "-q", str(worktree)], check=True)
    assert (worktree / ".git").is_file()
    assert agent_context.install_git_hooks(worktree)
    assert (main / ".git/hooks/pre-commit").read_bytes() == agent_context.MANAGED_HOOK.encode("utf-8")


def test_freshness_warns_only_when_runtime_code_drifts_past_state_baseline(tmp_path, monkeypatch):
    make_memory(tmp_path)
    (tmp_path / "docs/agents/STATE.md").write_text("> **Commit Baseline:** `6dae441`\n", encoding="utf-8")

    # Matching HEAD -> no warning
    assert agent_context.check_freshness(tmp_path, head_short="6dae441") is None

    # Docs-only change since baseline -> no warning
    monkeypatch.setattr(agent_context, "git_output", lambda root, *args: "")
    assert agent_context.check_freshness(tmp_path, head_short="0d6e3b6") is None

    # Runtime code changed since baseline -> actionable warning
    monkeypatch.setattr(
        agent_context,
        "git_output",
        lambda root, *args: "scripts/oneoff/embed_fate_subtitles.py\nsrc/pipeline.py",
    )
    warn = agent_context.check_freshness(tmp_path, head_short="0d6e3b6")
    assert warn is not None
    assert "6dae441" in warn and "0d6e3b6" in warn and "2 runtime path(s)" in warn

