"""Read-only, standard-library bootstrap for a new agent in this repository.

Reads only the curated Markdown memory and Git metadata. It deliberately does
not import the application, inspect credentials/media/checkpoints, install
packages, or run tests. All paths in the output are relative to the repository.
"""

from __future__ import annotations

import argparse
import ast
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


def _extract_py_symbols(file_path: Path) -> list[str]:
    """Extract AST classes and functions with signatures and line ranges."""
    try:
        content = file_path.read_text(encoding="utf-8-sig")
        tree = ast.parse(content, filename=file_path.name)
    except Exception as exc:
        return [f"  (failed to parse AST: {exc})"]

    results = []
    doc = ast.get_docstring(tree)
    if doc:
        first_line = doc.strip().splitlines()[0]
        results.append(f'  """{first_line}"""')

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            header = f"  class {node.name}({bases}):" if bases else f"  class {node.name}:"
            results.append(f"{header} [L{node.lineno}-L{node.end_lineno}]")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = ast.unparse(sub.args)
                    ret = f" -> {ast.unparse(sub.returns)}" if sub.returns else ""
                    results.append(f"    def {sub.name}({args}){ret}: ... [L{sub.lineno}-L{sub.end_lineno}]")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ast.unparse(node.args)
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            results.append(f"  def {node.name}({args}){ret}: ... [L{node.lineno}-L{node.end_lineno}]")
    return results


def show_symbols(root: Path, target: str | None = None) -> bool:
    """Print AST symbol maps for a specific file or directory of Python files."""
    target_path = (root / target) if target else root
    if not _inside(root, target_path) or not target_path.exists():
        print(f"Target path does not exist or leaves repo: {target}")
        return False

    if target_path.is_file():
        py_files = [target_path] if target_path.suffix == ".py" else []
    else:
        skip_dirs = {".venv", ".git", "__pycache__", "scratch", ".pytest_cache"}
        py_files = []
        for p in sorted(target_path.rglob("*.py")):
            if any(part in skip_dirs for part in p.parts):
                continue
            py_files.append(p)

    if not py_files:
        print(f"No Python files found in {target or '.'}")
        return False

    for p in py_files:
        rel = p.relative_to(root).as_posix()
        print(f"\n--- {rel} ---")
        symbols = _extract_py_symbols(p)
        if not symbols:
            print("  (no top-level classes or functions)")
        else:
            for s in symbols:
                print(s)
    return True


def find_symbol(root: Path, symbol_name: str) -> bool:
    """Search AST definitions across python files for exact or partial symbol name."""
    skip_dirs = {".venv", ".git", "__pycache__", "scratch", ".pytest_cache"}
    matches = []
    target_lower = symbol_name.lower()

    for p in sorted(root.rglob("*.py")):
        if any(part in skip_dirs for part in p.parts):
            continue
        try:
            content = p.read_text(encoding="utf-8-sig")
            tree = ast.parse(content, filename=p.name)
        except Exception:
            continue

        rel = p.relative_to(root).as_posix()
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if target_lower in node.name.lower():
                    bases = ", ".join(ast.unparse(b) for b in node.bases)
                    header = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
                    matches.append(f"{rel}:{node.lineno}: {header} [L{node.lineno}-L{node.end_lineno}]")
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if target_lower in sub.name.lower():
                            args = ast.unparse(sub.args)
                            ret = f" -> {ast.unparse(sub.returns)}" if sub.returns else ""
                            matches.append(f"{rel}:{sub.lineno}:   def {sub.name}({args}){ret}: ... [L{sub.lineno}-L{sub.end_lineno}]")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if target_lower in node.name.lower():
                    args = ast.unparse(node.args)
                    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                    matches.append(f"{rel}:{node.lineno}: def {node.name}({args}){ret}: ... [L{node.lineno}-L{node.end_lineno}]")

    if not matches:
        print(f"No AST definitions found matching '{symbol_name}'")
        return False

    print(f"Found {len(matches)} symbol definition(s) matching '{symbol_name}':")
    for m in matches:
        print(f"  {m}")
    return True


def install_git_hooks(root: Path) -> bool:
    """Install or update .git/hooks/pre-commit to enforce memory & link checks."""
    hooks_dir = root / ".git" / "hooks"
    if not hooks_dir.is_dir():
        print("Git hooks directory not found (.git/hooks). Is this a valid git repo?")
        return False
    hook_path = hooks_dir / "pre-commit"
    hook_content = (
        "#!/bin/sh\n"
        "# Antigravity agent context & memory integrity check\n"
        "python scripts/agent_context.py --check\n"
        "status=$?\n"
        "if [ $status -ne 0 ]; then\n"
        "    echo \"[PRE-COMMIT ERROR] Aborting commit due to agent_context check errors.\"\n"
        "    exit 1\n"
        "fi\n"
    )
    hook_path.write_text(hook_content, encoding="utf-8")
    try:
        hook_path.chmod(0o755)
    except OSError:
        pass
    print(f"Pre-commit hook successfully installed at: {hook_path}")
    return True


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root (default: parent of scripts/)")
    parser.add_argument("--check", action="store_true", help="Check required memory files, local links and instruction byte budgets")
    parser.add_argument("--inventory", action="store_true", help="Count tracked paths by top-level directory without reading contents")
    parser.add_argument("--symbols", nargs="?", const="", type=str, help="Show AST classes and functions for a path or entire repo")
    parser.add_argument("--find", type=str, help="Search AST definitions across python files for a class or function name")
    parser.add_argument("--install-hooks", action="store_true", help="Install git pre-commit hook to verify memory integrity before commit")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"Repository root is not a directory: {root}")
    exit_code = 0
    ran_action = False

    if args.check:
        ran_action = True
        errors = check_memory(root)
        for error in errors:
            print(f"ERROR: {error}")
        if errors:
            exit_code = 1
        else:
            print(f"PASS: required memory files, local links, root <= {ROOT_BUDGET} bytes, root + subtrans <= {COMBINED_BUDGET} bytes")

    if args.inventory:
        ran_action = True
        if not show_inventory(root):
            exit_code = 1

    if args.symbols is not None:
        ran_action = True
        target = args.symbols if args.symbols else None
        if not show_symbols(root, target):
            exit_code = 1

    if args.find:
        ran_action = True
        if not find_symbol(root, args.find):
            exit_code = 1

    if args.install_hooks:
        ran_action = True
        if not install_git_hooks(root):
            exit_code = 1

    if not ran_action:
        show_bootstrap(root)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
