"""Tests for scripts/export_review_bundle.py tiered bundle generation and exclusions."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest

from scripts import export_review_bundle


pytestmark = pytest.mark.unit


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _seed_bundle_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)

    # Multi-byte UTF-8 payload crossing byte 1024 to test encoding boundaries.
    wide_comment = "# " + ("Tiếng Việt kiểm tra ranh giới 1KB — " * 40) + "\n"
    files = {
        "GEMINI.md": "# GEMINI Anchor\n",
        "docs/agents/DECISIONS.md": "# Decisions Anchor\n",
        "docs/agents/PROJECT_MAP.md": "# Project Map Anchor\n",
        "docs/agents/VALIDATION.md": "# Validation Anchor\n",
        ".agents/rules/video-gpu.md": "---\ntrigger: glob\n---\n# GPU Rule\n",
        "docs/agents/STATE.md": "> **Commit Baseline:** `abc1234`\n",
        "src/helper.py": (
            '"""\n\nReal helper semantic summary line.\n"""\n'
            + wide_comment
            + "def compute_metric(x: int) -> int:\n    return x + 1\n"
        ),
        "src/pipeline.py": (
            '"""src/pipeline.py\nReal pipeline summary line.\n"""\n'
            "from src.helper import compute_metric\n\n"
            "def run_stage(n: int) -> int:\n    return compute_metric(n)\n"
        ),
        "subtrans/core.py": '"""Subtitle core."""\n\ndef translate() -> str:\n    return "vi"\n',
        "web/app.py": '"""Web monitor."""\n\ndef status() -> dict:\n    return {}\n',
        "upscale.py": '"""Upscale entrypoint."""\n',
        "scripts/tool.py": '"""Tool helper."""\n\ndef ping() -> str:\n    return "ok"\n',
        "glossaries/base.yaml": "version: 2\n",
        ".env": "GEMINI_API_KEY=SECRET_DO_NOT_LEAK\n",
        "subtrans/.env": "GEMINI_API_KEY_2=SECRET_SUBTRANS_LEAK\n",
        "benchmarks/upscale/ab_test/KEY_ANSWER.json": '{"answer": "SECRET_KEY_ANSWER"}\n',
        "docs/archive/old_notes.md": "ARCHIVED_DOC_SENTINEL\n",
        "scripts/oneoff/legacy_fix.py": "ONEOFF_SCRIPT_SENTINEL = 1\n",
        "tests/test_sample.py": "TEST_DEFAULT_EXCLUDED = 1\n",
        "subtrans/tests/test_sub.py": "SUBTRANS_TEST_EXCLUDED = 1\n",
        ".venv/lib.py": "VENV_SECRET_SENTINEL = 1\n",
        "out/ep01.srt": "OUT_SRT_SENTINEL\n",
        "models_bench/model.pth": "BINARY_MODEL_SENTINEL",
    }
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "initial fixture commit")

    # Second commit changing src/helper.py so latest_diff pulls src/pipeline.py skeleton
    (root / "src/helper.py").write_text(
        '"""\n\nReal helper semantic summary line.\n"""\n'
        + wide_comment
        + "def compute_metric(x: int) -> int:\n    return x + 2\n",
        encoding="utf-8",
    )
    _git(root, "add", "src/helper.py")
    _git(root, "commit", "-qm", "update compute_metric offset")


def test_sync_latest_creates_and_atomically_overwrites_six_fixed_bundles(tmp_path):
    repo = tmp_path / "repo"
    _seed_bundle_repo(repo)

    assert export_review_bundle.main(["--root", str(repo), "--sync-latest"]) == 0
    bundle_dir = repo / "scratch" / "review_bundles"
    expected_files = (
        "latest_diff.txt",
        "focus_video.txt",
        "focus_subtrans.txt",
        "focus_web.txt",
        "focus_agents.txt",
        "focus_full.txt",
    )
    for name in expected_files:
        assert (bundle_dir / name).is_file()
    assert sorted(p.name for p in bundle_dir.glob("*.txt")) == sorted(expected_files)
    assert not list(bundle_dir.glob("*.tmp*"))

    # Modify tracked source without committing and confirm atomic overwrite works cleanly
    (repo / "src/helper.py").write_text(
        '"""\n\nReal helper semantic summary line.\n"""\n'
        "def compute_metric(x: int) -> int:\n    return x + 999\n",
        encoding="utf-8",
    )
    assert export_review_bundle.main(["--root", str(repo), "--sync-latest"]) == 0
    assert not list(bundle_dir.glob("*.tmp*"))
    diff_text = (bundle_dir / "latest_diff.txt").read_text(encoding="utf-8")
    full_text = (bundle_dir / "focus_full.txt").read_text(encoding="utf-8")
    assert "return x + 999" in diff_text
    assert "return x + 999" in full_text


def test_bundle_prefix_cache_order_utf8_skeletons_and_exclusions(tmp_path):
    repo = tmp_path / "repo"
    _seed_bundle_repo(repo)
    assert export_review_bundle.main(["--root", str(repo), "--sync-latest"]) == 0

    bundle_dir = repo / "scratch" / "review_bundles"
    diff_bundle = (bundle_dir / "latest_diff.txt").read_text(encoding="utf-8")

    # Prefix-cache deterministic section ordering: static headers/anchors/skeletons before volatile state/diff
    section_markers = [
        "## 1. REPOSITORY TREE",
        "## 2. STATIC CORE ANCHORS",
        "## 3. AST SKELETON MODULES",
        "## 4. VOLATILE STATE",
        "## 5. FOCUSED / CHANGED FULL FILES",
        "## 6. GIT LOG & DIFF",
        "## 7. BUNDLE METADATA",
    ]
    positions = [diff_bundle.index(m) for m in section_markers]
    assert positions == sorted(positions)

    # Numbered full block + multi-byte UTF-8 line preserved cleanly
    assert "[FULL] src/helper.py" in diff_bundle
    assert "L0001: " in diff_bundle
    assert "Tiếng Việt kiểm tra ranh giới 1KB" in diff_bundle

    # Neighboring pipeline.py included as [AST_SKELETON] with stripped filename echo
    assert "[AST_SKELETON] src/pipeline.py" in diff_bundle
    assert "Real pipeline summary line." in diff_bundle
    assert '"""src/pipeline.py' not in diff_bundle

    # Focus bundles isolate domain FULL files while keeping cross-domain skeletons
    video_bundle = (bundle_dir / "focus_video.txt").read_text(encoding="utf-8")
    assert "[FULL] src/helper.py" in video_bundle
    assert "[FULL] subtrans/core.py" not in video_bundle
    assert "[AST_SKELETON] subtrans/core.py" in video_bundle

    # Strict exclusions verified across generated bundles
    for forbidden in (
        "SECRET_DO_NOT_LEAK",
        "SECRET_SUBTRANS_LEAK",
        "SECRET_KEY_ANSWER",
        "ARCHIVED_DOC_SENTINEL",
        "ONEOFF_SCRIPT_SENTINEL",
        "TEST_DEFAULT_EXCLUDED",
        "SUBTRANS_TEST_EXCLUDED",
        "VENV_SECRET_SENTINEL",
        "OUT_SRT_SENTINEL",
        "BINARY_MODEL_SENTINEL",
    ):
        assert forbidden not in video_bundle
        assert forbidden not in diff_bundle


def test_include_tests_opt_in_and_refuse_tracked_overwrite(tmp_path):
    repo = tmp_path / "repo"
    _seed_bundle_repo(repo)

    full_default = repo / "scratch" / "default_full.txt"
    assert export_review_bundle.main(["--root", str(repo), "--full", "-o", str(full_default)]) == 0
    default_text = full_default.read_text(encoding="utf-8")
    assert "TEST_DEFAULT_EXCLUDED" not in default_text
    assert "SUBTRANS_TEST_EXCLUDED" not in default_text

    full_with_tests = repo / "scratch" / "tests_full.txt"
    assert export_review_bundle.main(
        ["--root", str(repo), "--full", "--include-tests", "-o", str(full_with_tests)]
    ) == 0
    with_tests_text = full_with_tests.read_text(encoding="utf-8")
    assert "TEST_DEFAULT_EXCLUDED" in with_tests_text
    assert "SUBTRANS_TEST_EXCLUDED" in with_tests_text

    # Must refuse writing over a tracked repo file
    before_gemini = (repo / "GEMINI.md").read_text(encoding="utf-8")
    rc = export_review_bundle.main(["--root", str(repo), "--full", "-o", str(repo / "GEMINI.md")])
    assert rc == 1
    assert (repo / "GEMINI.md").read_text(encoding="utf-8") == before_gemini


def test_format_full_file_numbered_and_unnumbered():
    assert export_review_bundle._format_full_file("demo.py", "a\nb\n") == (
        "### [FULL] demo.py (2 lines)\nL0001: a\nL0002: b\n"
    )
    assert export_review_bundle._format_full_file("demo.py", "a\nb\n", line_numbers=False) == (
        "### [FULL] demo.py (2 lines)\na\nb\n"
    )
    assert export_review_bundle._format_full_file("empty.txt", "", line_numbers=False) == (
        "### [FULL] empty.txt (0 lines)\n"
    )


def test_focus_full_deduplicated_unnumbered_and_preserves_split_bundles(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _seed_bundle_repo(repo)

    assert export_review_bundle.main(["--root", str(repo), "--sync-latest"]) == 0
    bundle_dir = repo / "scratch" / "review_bundles"
    for split_name in ("focus_video.txt", "focus_agents.txt"):
        split_text = (bundle_dir / split_name).read_text(encoding="utf-8")
        assert "## 2. STATIC CORE ANCHORS" in split_text
        assert "## 3. AST SKELETON MODULES" in split_text
        assert "### [AST_SKELETON]" in split_text
        assert "L0001: " in split_text

    def _fail_skeletons(*_args, **_kwargs):
        raise AssertionError("AST skeletons must be skipped for standalone --focus full")

    monkeypatch.setattr(export_review_bundle, "_format_py_skeleton", _fail_skeletons)
    assert export_review_bundle.main(["--root", str(repo), "--focus", "full"]) == 0

    content = (bundle_dir / "focus_full.txt").read_text(encoding="utf-8")
    section_markers = [
        "# AI-VIDEO-UPSCALER REVIEW BUNDLE",
        "## 1. REPOSITORY TREE",
        "## 2. FULL REPOSITORY FILES",
        "## 3. GIT LOG & DIFF",
        "## 4. BUNDLE METADATA",
    ]
    positions = [content.index(marker) for marker in section_markers]
    assert positions == sorted(positions)

    allowed_files = (
        "GEMINI.md",
        "docs/agents/DECISIONS.md",
        "docs/agents/PROJECT_MAP.md",
        "docs/agents/VALIDATION.md",
        "docs/agents/STATE.md",
        ".agents/rules/video-gpu.md",
        "src/helper.py",
        "src/pipeline.py",
        "subtrans/core.py",
        "web/app.py",
        "upscale.py",
        "scripts/tool.py",
        "glossaries/base.yaml",
    )
    for rel in allowed_files:
        assert content.count(f"### [FULL] {rel} ") == 1

    assert "STATIC CORE ANCHORS" not in content
    assert "AST SKELETON MODULES" not in content
    assert "### [AST_SKELETON]" not in content
    assert "VOLATILE STATE" not in content
    assert "L0001: " not in content
    assert re.search(r"^L\d{4}: ", content, re.MULTILINE) is None

    assert "Tiếng Việt kiểm tra ranh giới 1KB" in content
    assert "def compute_metric(x: int) -> int:\n    return x + 2\n" in content

    for forbidden in (
        "SECRET_DO_NOT_LEAK",
        "SECRET_SUBTRANS_LEAK",
        "SECRET_KEY_ANSWER",
        "ARCHIVED_DOC_SENTINEL",
        "ONEOFF_SCRIPT_SENTINEL",
        "TEST_DEFAULT_EXCLUDED",
        "SUBTRANS_TEST_EXCLUDED",
        "VENV_SECRET_SENTINEL",
        "OUT_SRT_SENTINEL",
        "BINARY_MODEL_SENTINEL",
    ):
        assert forbidden not in content

    assert "- Mode: focus:full\n" in content
    assert f"- Full Files: {len(allowed_files)} | AST Skeletons: 0\n" in content
    meta_match = re.search(r"^- Characters: (\d+) \| Est\. Tokens: (\d+)$", content, re.MULTILINE)
    assert meta_match is not None
    assert int(meta_match.group(1)) == len(content)
    assert int(meta_match.group(2)) == len(content) // 4


def test_focus_full_include_tests_and_rejects_combined_tokens(tmp_path, capsys):
    repo = tmp_path / "repo"
    _seed_bundle_repo(repo)

    default_out = repo / "scratch" / "focus_full_default.txt"
    assert export_review_bundle.main(["--root", str(repo), "--focus", "full", "-o", str(default_out)]) == 0
    default_text = default_out.read_text(encoding="utf-8")
    assert "TEST_DEFAULT_EXCLUDED" not in default_text
    assert "SUBTRANS_TEST_EXCLUDED" not in default_text

    tests_out = repo / "scratch" / "focus_full_tests.txt"
    assert export_review_bundle.main(
        ["--root", str(repo), "--focus", "full", "--include-tests", "-o", str(tests_out)]
    ) == 0
    tests_text = tests_out.read_text(encoding="utf-8")
    assert tests_text.count("### [FULL] tests/test_sample.py ") == 1
    assert tests_text.count("### [FULL] subtrans/tests/test_sub.py ") == 1
    assert "TEST_DEFAULT_EXCLUDED" in tests_text
    assert "SUBTRANS_TEST_EXCLUDED" in tests_text
    meta_match = re.search(r"^- Characters: (\d+) \| Est\. Tokens: (\d+)$", tests_text, re.MULTILINE)
    assert meta_match is not None
    assert int(meta_match.group(1)) == len(tests_text)

    assert export_review_bundle.main(["--root", str(repo), "--focus", "full,video"]) == 1
    err = capsys.readouterr().err
    assert "full" in err.lower()
    assert not (repo / "scratch" / "review_bundles" / "focus_full_video.txt").exists()

    with pytest.raises(ValueError, match="full"):
        export_review_bundle._render_bundle(
            repo,
            "focus:full,video",
            {"GEMINI.md": "# GEMINI\n"},
            {},
            {},
            "main",
            "abc1234",
            "subject",
            focus_tokens=["full", "video"],
        )
