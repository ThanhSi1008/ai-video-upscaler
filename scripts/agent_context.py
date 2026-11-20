"""Standard-library bootstrap for a new agent in this repository.

Reads only the curated Markdown memory and Git metadata. It deliberately does
not import the application, inspect credentials/media/checkpoints, install
packages, or run tests. AST commands read bounded source files. The explicit
--install-hooks action is the only command that writes a file.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import os
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
BASELINE_RE = re.compile(r"Commit Baseline:\*{0,2}\s*`([0-9a-fA-F]{7,40})`")
RUNTIME_PATHS = ("upscale.py", "subtrans.py", "web_gui.py", "src", "subtrans", "scripts", "web")
SOURCE_DIRS = frozenset({"src", "subtrans", "tests", "scripts", "benchmarks", "web"})
PRIVATE_DIRS = frozenset({"venv", "env", "bin", "__pycache__", "scratch", "out", "models_bench", "samples", "test_frames"})
MANAGED_HOOK = (
    "#!/bin/sh\n"
    "# Antigravity agent context & memory integrity check\n"
    "if [ -x .venv/Scripts/python.exe ]; then\n"
    "    .venv/Scripts/python.exe scripts/agent_context.py --check\n"
    "elif [ -x .venv/bin/python ]; then\n"
    "    .venv/bin/python scripts/agent_context.py --check\n"
    "else\n"
    "    python scripts/agent_context.py --check\n"
    "fi\n"
    "status=$?\n"
    "if [ $status -ne 0 ]; then\n"
    "    echo \"[PRE-COMMIT ERROR] Aborting commit due to agent_context check errors.\"\n"
    "    exit 1\n"
    "fi\n"
)


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


def check_freshness(root: Path, head_short: str | None = None) -> str | None:
    """Return a non-blocking warning when runtime code drifted past STATE.md baseline."""
    state_path = root / "docs" / "agents" / "STATE.md"
    if not _inside(root, state_path) or not state_path.is_file():
        return None
    try:
        content = _read_memory(root, "docs/agents/STATE.md")
    except (OSError, UnicodeError, ValueError):
        return None
    match = BASELINE_RE.search(content)
    if not match:
        return None
    baseline = match.group(1).lower()
    head = (head_short or git_output(root, "rev-parse", "--short", "HEAD") or "").lower()
    if not head or head.startswith(baseline) or baseline.startswith(head):
        return None
    changed_raw = git_output(root, "diff", "--name-only", f"{baseline}..HEAD", "--", *RUNTIME_PATHS)
    if changed_raw is None:
        return f"STATE.md baseline `{baseline}` differs from HEAD `{head}` (commit range unavailable)."
    changed = [line for line in changed_raw.splitlines() if line.strip()]
    if not changed:
        return None
    preview = ", ".join(changed[:4]) + ("..." if len(changed) > 4 else "")
    return f"STATE.md baseline `{baseline}` trails HEAD `{head}` ({len(changed)} runtime path(s) changed: {preview})"


def show_bootstrap(root: Path) -> None:
    print("AGENT CONTEXT - read-only; no runtime/API/GPU initialization")
    print(f"Branch: {git_output(root, 'branch', '--show-current') or '(detached or unavailable)'}")
    head = git_output(root, "rev-parse", "--short", "HEAD")
    print(f"HEAD: {head or '(unavailable)'}")
    status = git_output(root, 'status', '--short')
    print(f"Working tree:\n{status if status else '(clean)' if status == '' else '(Git unavailable)'}")
    freshness_warn = check_freshness(root, head_short=head)
    if freshness_warn:
        print(f"WARN: {freshness_warn}")
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


def _source_path_allowed(root: Path, path: Path) -> bool:
    """Reject private directories, links and resolved escapes before any read."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if not relative.parts:
        return _inside(root, path)
    if any(part.startswith(".") or part.lower() in PRIVATE_DIRS for part in relative.parts):
        return False
    if len(relative.parts) > 1 or path.is_dir():
        if relative.parts[0] not in SOURCE_DIRS:
            return False
    current = root
    for part in relative.parts:
        current /= part
        # A link to another path inside the repo can also expose runtime data.
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            return False
        if current.resolve() != current.absolute():
            return False
    return _inside(root, path)


def source_files(root: Path, target: str | None = None) -> list[Path]:
    """Prune before traversal; never enumerate virtualenv/cache/output trees."""
    root = root.resolve()
    target_path = root / target if target else root
    if not _source_path_allowed(root, target_path) or not target_path.exists():
        return []
    if target_path.is_file():
        return [target_path] if target_path.suffix.lower() == ".py" else []
    found = []
    for directory, dirs, files in os.walk(target_path, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(name for name in dirs if _source_path_allowed(root, parent / name))
        for name in sorted(files):
            path = parent / name
            if path.suffix.lower() == ".py" and _source_path_allowed(root, path):
                found.append(path)
    return sorted(found)


def show_symbols(root: Path, target: str | None = None) -> bool:
    """Print AST symbol maps for safe source files without importing them."""
    py_files = source_files(root, target)

    if not py_files:
        print(f"No allowed Python source files found in {target or '.'}")
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
    matches = []
    target_lower = symbol_name.lower()

    for p in source_files(root):
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
    """Install the managed hook without replacing user hooks or external paths."""
    common_dir = git_output(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    effective_hooks = git_output(root, "rev-parse", "--path-format=absolute", "--git-path", "hooks")
    if not common_dir or not effective_hooks:
        print("Git hooks directory unavailable. Is this a valid git repo?")
        return False
    common_path = Path(common_dir).resolve()
    hooks_dir = Path(effective_hooks)
    if not hooks_dir.is_absolute():
        hooks_dir = root / hooks_dir
    if not hooks_dir.resolve().is_relative_to(common_path):
        print("Refusing custom hooks path outside the repository Git directory.")
        return False
    hook_path = hooks_dir / "pre-commit"
    try:
        if hook_path.is_symlink() or not hook_path.resolve().is_relative_to(common_path):
            print("Refusing linked pre-commit hook.")
            return False
        if hook_path.exists():
            original_variants = {
                MANAGED_HOOK.encode("utf-8"),
                MANAGED_HOOK.replace("\n", "\r\n").encode("utf-8"),
            }
            if hook_path.read_bytes() in original_variants:
                print("Managed pre-commit hook already installed; unchanged.")
                return True
            print("Existing pre-commit hook preserved; integrate the memory check manually.")
            return False
        hooks_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects a hook created concurrently.
        with hook_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(MANAGED_HOOK)
        hook_path.chmod(0o755)
    except OSError as exc:
        print(f"Could not install managed hook: {exc}")
        return False
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
            freshness_warn = check_freshness(root)
            if freshness_warn:
                print(f"WARN: {freshness_warn}")

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
