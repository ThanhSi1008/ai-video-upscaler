"""The bootstrap must remain offline, bounded and useful after a fresh clone."""

from pathlib import Path
import subprocess

from scripts import agent_context


def make_memory(root: Path) -> None:
    for relative in agent_context.REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Fixture\n", encoding="utf-8")


def test_valid_memory_accepts_relative_links_and_ignores_remote_and_code(tmp_path):
    make_memory(tmp_path)
    (tmp_path / "docs/agents/START_HERE.md").write_text(
        "[rules](../../AGENTS.md#rules)\n[map](PROJECT_MAP.md)\n"
        "[remote](https://example.invalid/not-fetched)\n"
        "```md\n[example](missing-example.md)\n```\n",
        encoding="utf-8",
    )
    # Old research documentation is deliberately outside the validation scope.
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
    (tmp_path / "AGENTS.md").write_text(
        "[outside](../outside.md)\n[absolute](file:///C:/outside.md)", encoding="utf-8"
    )
    errors = agent_context.check_memory(tmp_path)
    assert any("link leaves repository" in e for e in errors)
    assert any("nonportable local link" in e for e in errors)


def test_byte_budgets_use_utf8_bytes_and_combined_scope(tmp_path):
    make_memory(tmp_path)
    (tmp_path / "AGENTS.md").write_text("ệ" * (agent_context.ROOT_BUDGET // 3 + 1), encoding="utf-8")
    (tmp_path / "subtrans/AGENTS.md").write_text("x" * agent_context.COMBINED_BUDGET, encoding="utf-8")
    errors = agent_context.check_memory(tmp_path)
    assert any("AGENTS.md exceeds" in e for e in errors)
    assert any("Root + subtrans AGENTS exceed" in e for e in errors)


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
