"""Read-only, standard-library bootstrap for a new agent in this repository.

Reads only the curated Markdown memory and Git metadata. It deliberately does
not import the application, inspect credentials/media/checkpoints, install
packages, or run tests. All paths in the output are relative to the repository.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


REQUIRED_FILES = (
    "AGENTS.md",
    "GEMINI.md",
    "CLAUDE.md",
    "subtrans/AGENTS.md",
    "docs/agents/START_HERE.md",
    "docs/agents/PROJECT_MAP.md",
    "docs/agents/STATE.md",
    "docs/agents/DECISIONS.md",
    "docs/agents/VALIDATION.md",
    "docs/agents/tasks/TEMPLATE.md",
    "docs/agents/tasks/2026-10-29-agent-memory.md",
)
BOOTSTRAP_FILES = ("docs/agents/START_HERE.md", "docs/agents/STATE.md")
ROOT_BUDGET = 12 * 1024
COMBINED_BUDGET = 32 * 1024
LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+[^)]+)?\)")


def _inside(root: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _read_memory(root: Path, relative: str) -> str:
    path = root / relative
    if not _inside(root, path):
        raise ValueError(f"Memory path leaves repository: {relative}")
    return path.read_text(encoding="utf-8-sig")


def git_output(root: Path, *args: str) -> str | None:
    """Run only the caller's fixed read-only Git queries; never a shell."""
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def memory_files(root: Path) -> list[str]:
    """Bound validation to the memory system, not old research reports."""
    files = {"AGENTS.md", "GEMINI.md", "CLAUDE.md", "subtrans/AGENTS.md"}
    memory_dir = root / "docs" / "agents"
    if memory_dir.is_dir() and _inside(root, memory_dir):
        files.update(p.relative_to(root).as_posix() for p in memory_dir.rglob("*.md"))
    return sorted(files)


def _links(markdown: str):
    """Yield ordinary inline links outside fenced code blocks.

    This is intentionally a small Markdown check, not a full renderer. Anchors,
    reference-style links and remote targets are not fetched or validated.
    """
    fence = None
    for line_no, line in enumerate(markdown.splitlines(), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        for match in LINK_RE.finditer(line):
            yield line_no, match.group(1) or match.group(2)


def check_memory(root: Path) -> list[str]:
    """Return actionable errors without changing files or following externals."""
    errors = []
    for relative in REQUIRED_FILES:
        path = root / relative
        if not _inside(root, path) or not path.is_file():
            errors.append(f"Missing or unsafe required file: {relative}")

    sizes = {}
    for relative in ("AGENTS.md", "subtrans/AGENTS.md"):
        path = root / relative
        if _inside(root, path) and path.is_file():
            sizes[relative] = path.stat().st_size
    if sizes.get("AGENTS.md", 0) > ROOT_BUDGET:
        errors.append(f"AGENTS.md exceeds {ROOT_BUDGET} bytes: {sizes['AGENTS.md']}")
    if sum(sizes.values()) > COMBINED_BUDGET:
        errors.append(f"Root + subtrans AGENTS exceed {COMBINED_BUDGET} bytes: {sum(sizes.values())}")

    for relative in memory_files(root):
        path = root / relative
        if not path.is_file():
            continue  # Required files already have a clearer error above.
        try:
            content = _read_memory(root, relative)
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"Cannot read {relative}: {exc}")
            continue
        for line_no, target in _links(content):
            parsed = urlsplit(target)
            if parsed.scheme in {"https", "http", "mailto"} or target.startswith("#"):
                continue
            if parsed.scheme or parsed.netloc:
                errors.append(f"{relative}:{line_no}: nonportable local link: {target}")
                continue
            linked = path.parent / unquote(parsed.path)
            if not _inside(root, linked):
                errors.append(f"{relative}:{line_no}: link leaves repository: {target}")
            elif not linked.exists():
                errors.append(f"{relative}:{line_no}: missing local target: {target}")
    return errors


def show_bootstrap(root: Path) -> None:
    print("AGENT CONTEXT - read-only; no runtime/API/GPU initialization")
    print(f"Branch: {git_output(root, 'branch', '--show-current') or '(detached or unavailable)'}")
    print(f"HEAD: {git_output(root, 'rev-parse', '--short', 'HEAD') or '(unavailable)'}")
    status = git_output(root, 'status', '--short')
    print(f"Working tree:\n{status if status else '(clean)' if status == '' else '(Git unavailable)'}")
    for relative in BOOTSTRAP_FILES:
        print(f"\n--- {relative} ---")
        try:
            print(_read_memory(root, relative).rstrip())
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"Unavailable: {exc}")
    print("\nRead root AGENTS.md and the task-relevant nested AGENTS.md.")
    print("Further context: docs/agents/PROJECT_MAP.md, DECISIONS.md, VALIDATION.md, tasks/.")
    print("Reconcile the saved state against current Git and the latest user request.")


def show_inventory(root: Path) -> bool:
    listing = git_output(root, "ls-files", "-z")
    if listing is None:
        print("Inventory unavailable: Git repository or executable not found.")
        return False
    paths = [p for p in listing.split("\0") if p]
    groups = Counter(p.split("/", 1)[0] if "/" in p else "(root files)" for p in paths)
    print(f"Tracked paths: {len(paths)} (names only; file contents not scanned)")
    for group, count in sorted(groups.items()):
        print(f"  {group}: {count}")
    return True


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root (default: parent of scripts/)")
    parser.add_argument("--check", action="store_true", help="Check required memory files, local links and instruction byte budgets")
    parser.add_argument("--inventory", action="store_true", help="Count tracked paths by top-level directory without reading contents")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Repository root is not a directory: {root}")
    exit_code = 0
    if args.check:
        errors = check_memory(root)
        for error in errors:
            print(f"ERROR: {error}")
        if errors:
            exit_code = 1
        else:
            print(f"PASS: required memory files, local links, root <= {ROOT_BUDGET} bytes, root + subtrans <= {COMBINED_BUDGET} bytes")
    if args.inventory and not show_inventory(root):
        exit_code = 1
    if not args.check and not args.inventory:
        show_bootstrap(root)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
