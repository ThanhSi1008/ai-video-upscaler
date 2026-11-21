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


SUBTRANS_SKILL_REL = ".agents/skills/subtrans-anime-ops/SKILL.md"
EXPECTED_RULES = ("video-gpu.md", "subtrans.md", "web-ui.md")
EXPECTED_SUBAGENTS = ("test-verifier.md", "subtrans-evaluator.md", "repo-scout.md")
REQUIRED_FILES = (
    "GEMINI.md",
    "docs/agents/PROJECT_MAP.md",
    "docs/agents/STATE.md",
    "docs/agents/DECISIONS.md",
    "docs/agents/VALIDATION.md",
    "docs/agents/tasks/TEMPLATE.md",
    SUBTRANS_SKILL_REL,
)
BOOTSTRAP_FILES = ("docs/agents/STATE.md",)
ROOT_BUDGET = 4500
MAX_ROOT_LINES = 45
RULE_BUDGET = 4096
MAX_RULE_LINES = 60
COMBINED_BUDGET = 32 * 1024
LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+[^)]+)?\)")
BASELINE_RE = re.compile(r"Commit Baseline:\*{0,2}\s*`([0-9a-fA-F]{7,40})`")
DOC_ECHO_RE = re.compile(r"^(?:Module:\s*)?[\w./\\-]+(?:\.py)?\s*$")
SAFE_ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})
BLOCKED_RELATIVE_PATHS = frozenset({"benchmarks/upscale/ab_test/key_answer.json"})
RUNTIME_PATHS = ("upscale.py", "subtrans.py", "web_gui.py", "src", "subtrans", "scripts", "web")
SOURCE_DIRS = frozenset({"src", "subtrans", "tests", "scripts", "benchmarks", "web"})
PRIVATE_DIRS = frozenset({"venv", "env", "bin", "__pycache__", "scratch", "out", "models_bench", "samples", "test_frames"})
MAX_SLICE_DEFAULT = 220
MAX_PACK_SLICE_LINES = 160
MAX_PACK_TARGETS = 4
CROSS_BOUNDARY_CONTRACTS: tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], bool], ...] = (
    # (side_a, side_b, coupled_tests, bidirectional)
    (("src/logger.py",), ("web/sse_hub.py", "web/app.py", "src/night_report.py"), ("tests/test_web_gui.py",), True),
    (("src/dedup.py", "shows.yaml", "models/models.yaml"), ("web/fs_browser.py", "web/app.py"), ("tests/test_web_gui.py",), True),
    (("upscale.py", "subtrans.py", "subtrans/cli.py"), ("web/job_manager.py", "scripts/batch_translate_all.py"), ("tests/test_web_gui.py",), True),
    (("web/static/*",), ("web/app.py",), ("tests/test_web_gui.py", "tests/test_architecture_boundaries.py"), False),
    (("glossaries/*", "subtrans/*.yaml"), ("subtrans/glossary.py", "subtrans/config.py"), ("subtrans/tests/",), True),
    (("GEMINI.md", ".agents/*", "docs/agents/*", ".antigravityignore", ".gitignore"), ("scripts/agent_context.py",), ("tests/test_agent_context.py", "tests/test_agent_setup.py"), True),
    (("subtrans/cache.py",), ("web/app.py", "src/night_report.py"), ("tests/test_web_gui.py", "subtrans/tests/"), True),
    (("src/config.py", "src/pending_queue.py"), ("web/sse_hub.py", "web/job_manager.py", "web/app.py"), ("tests/test_web_gui.py",), True),
)
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
MANAGED_POST_COMMIT_HOOK = (
    "#!/bin/sh\n"
    "# Antigravity automatic review bundle sync (non-blocking)\n"
    "if [ -x .venv/Scripts/python.exe ]; then\n"
    "    .venv/Scripts/python.exe scripts/export_review_bundle.py --sync-latest >/dev/null 2>&1 || true\n"
    "elif [ -x .venv/bin/python ]; then\n"
    "    .venv/bin/python scripts/export_review_bundle.py --sync-latest >/dev/null 2>&1 || true\n"
    "else\n"
    "    python scripts/export_review_bundle.py --sync-latest >/dev/null 2>&1 || true\n"
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
    """Bound validation to the active memory system, rules, subagents, and workspace skills."""
    files = {"GEMINI.md"}
    memory_dir = root / "docs" / "agents"
    if memory_dir.is_dir() and _inside(root, memory_dir):
        files.update(p.relative_to(root).as_posix() for p in memory_dir.rglob("*.md"))
    rules_dir = root / ".agents" / "rules"
    if rules_dir.is_dir() and _inside(root, rules_dir):
        for rule_md in rules_dir.glob("*.md"):
            if _inside(root, rule_md):
                files.add(rule_md.relative_to(root).as_posix())
    agents_dir = root / ".agents" / "agents"
    if agents_dir.is_dir() and _inside(root, agents_dir):
        for agent_md in agents_dir.glob("*.md"):
            if _inside(root, agent_md):
                files.add(agent_md.relative_to(root).as_posix())
    skills_dir = root / ".agents" / "skills"
    if skills_dir.is_dir() and _inside(root, skills_dir):
        for skill_md in skills_dir.glob("*/SKILL.md"):
            if _inside(root, skill_md):
                files.add(skill_md.relative_to(root).as_posix())
    return sorted(files)


def _parse_frontmatter(markdown: str) -> dict[str, str] | None:
    """Parse flat YAML key/value frontmatter from the start of a Markdown file."""
    norm = markdown.replace("\r\n", "\n")
    if not norm.startswith("---\n"):
        return None
    end_idx = norm.find("\n---\n", 4)
    if end_idx == -1:
        return None
    block = norm[4:end_idx]
    parsed: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip()
        if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
            val = val[1:-1]
        parsed[key.strip()] = val.strip()
    return parsed


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
    for relative in ("GEMINI.md", SUBTRANS_SKILL_REL):
        path = root / relative
        if _inside(root, path) and path.is_file():
            sizes[relative] = path.stat().st_size
    if sizes.get("GEMINI.md", 0) > ROOT_BUDGET:
        errors.append(f"GEMINI.md exceeds {ROOT_BUDGET} bytes: {sizes['GEMINI.md']}")
    gemini_path = root / "GEMINI.md"
    if _inside(root, gemini_path) and gemini_path.is_file():
        try:
            gemini_lines = len(_read_memory(root, "GEMINI.md").splitlines())
            if gemini_lines > MAX_ROOT_LINES:
                errors.append(f"GEMINI.md exceeds {MAX_ROOT_LINES} lines: {gemini_lines}")
        except (OSError, UnicodeError, ValueError):
            pass  # Read failure will be reported in the link-validation loop below.
    if sum(sizes.values()) > COMBINED_BUDGET:
        errors.append(f"Root + subtrans skill exceed {COMBINED_BUDGET} bytes: {sum(sizes.values())}")

    has_workspace_agents_tree = any(
        (root / ".agents" / sub).exists() for sub in ("rules", "agents", "plugins")
    )
    if has_workspace_agents_tree:
        rules_dir = root / ".agents" / "rules"
        if not rules_dir.is_dir() or not _inside(root, rules_dir):
            errors.append("Missing or unsafe rules directory: .agents/rules")
        else:
            nested_dirs = [p.name for p in rules_dir.iterdir() if p.is_dir()]
            if nested_dirs:
                errors.append(f".agents/rules must remain flat (found subdirectories: {', '.join(sorted(nested_dirs))})")
            for rule_name in EXPECTED_RULES:
                if not (rules_dir / rule_name).is_file():
                    errors.append(f"Missing required rule file: .agents/rules/{rule_name}")
            for rule_path in sorted(rules_dir.glob("*.md")):
                rel_rule = rule_path.relative_to(root).as_posix()
                try:
                    raw_rule = _read_memory(root, rel_rule)
                except (OSError, UnicodeError, ValueError) as exc:
                    errors.append(f"Cannot read {rel_rule}: {exc}")
                    continue
                if rule_path.stat().st_size > RULE_BUDGET:
                    errors.append(f"{rel_rule} exceeds {RULE_BUDGET} bytes: {rule_path.stat().st_size}")
                rule_lines = len(raw_rule.splitlines())
                if rule_lines > MAX_RULE_LINES:
                    errors.append(f"{rel_rule} exceeds {MAX_RULE_LINES} lines: {rule_lines}")
                fm = _parse_frontmatter(raw_rule)
                if fm is None:
                    errors.append(f"{rel_rule}: missing or malformed YAML frontmatter")
                else:
                    for key in ("trigger", "globs", "description"):
                        if not fm.get(key):
                            errors.append(f"{rel_rule}: frontmatter missing '{key}'")
                    if fm.get("trigger") and fm["trigger"] != "glob":
                        errors.append(f"{rel_rule}: expected trigger: glob, got '{fm['trigger']}'")

        agents_dir = root / ".agents" / "agents"
        if not agents_dir.is_dir() or not _inside(root, agents_dir):
            errors.append("Missing or unsafe custom subagents directory: .agents/agents")
        else:
            for agent_name in EXPECTED_SUBAGENTS:
                if not (agents_dir / agent_name).is_file():
                    errors.append(f"Missing required subagent spec: .agents/agents/{agent_name}")
            for agent_path in sorted(agents_dir.glob("*.md")):
                rel_agent = agent_path.relative_to(root).as_posix()
                try:
                    raw_agent = _read_memory(root, rel_agent)
                except (OSError, UnicodeError, ValueError) as exc:
                    errors.append(f"Cannot read {rel_agent}: {exc}")
                    continue
                fm = _parse_frontmatter(raw_agent)
                if fm is None:
                    errors.append(f"{rel_agent}: missing or malformed YAML frontmatter")
                else:
                    if not fm.get("name") or not fm.get("description"):
                        errors.append(f"{rel_agent}: frontmatter missing 'name' or 'description'")
                    if fm.get("subagent", "").lower() != "true":
                        errors.append(f"{rel_agent}: subagent frontmatter must set 'subagent: true'")
                    if fm.get("inheritMcp", "").lower() != "false":
                        errors.append(f"{rel_agent}: subagent frontmatter must set 'inheritMcp: false'")

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
    print("\nRead root GEMINI.md and task-relevant skills under .agents/skills/*/SKILL.md.")
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


def _first_meaningful_doc_line(doc: str | None) -> str:
    """Return the first semantic summary line, skipping empty or module-path echo lines."""
    if not doc:
        return ""
    for raw_line in doc.strip().splitlines():
        line = raw_line.strip()
        if not line or DOC_ECHO_RE.match(line):
            continue
        return line
    return ""


def _format_decorator_tags(decorators: list[ast.expr]) -> str:
    if not decorators:
        return ""
    tags: list[str] = []
    for dec in decorators[:3]:
        tag = f"@{ast.unparse(dec)}"
        if len(tag) > 64:
            tag = tag[:61] + "..."
        tags.append(tag)
    if len(decorators) > 3:
        tags.append(f"+{len(decorators) - 3}")
    return " ".join(tags) + " "


def _is_model_or_dataclass(node: ast.ClassDef) -> bool:
    for base in node.bases:
        base_str = ast.unparse(base)
        if base_str == "BaseModel" or base_str.endswith(".BaseModel"):
            return True
    for dec in node.decorator_list:
        dec_str = ast.unparse(dec)
        if "dataclass" in dec_str:
            return True
    return False


def _iter_lexical_nested_defs(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """Collect nested functions/classes lexically enclosed by func_node without entering their bodies."""
    nested: list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = []

    def _walk_stmts(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nested.append(stmt)
                continue
            if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                _walk_stmts(stmt.body)
                _walk_stmts(stmt.orelse)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                _walk_stmts(stmt.body)
            elif isinstance(stmt, ast.Try):
                _walk_stmts(stmt.body)
                for handler in stmt.handlers:
                    _walk_stmts(handler.body)
                _walk_stmts(stmt.orelse)
                _walk_stmts(stmt.finalbody)
            elif hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar):
                _walk_stmts(stmt.body)
                for handler in stmt.handlers:
                    _walk_stmts(handler.body)
                _walk_stmts(stmt.orelse)
                _walk_stmts(stmt.finalbody)
            elif hasattr(ast, "Match") and isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    _walk_stmts(case.body)

    _walk_stmts(func_node.body)
    return nested


def _format_py_func_header(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    dec = _format_decorator_tags(node.decorator_list)
    end_ln = node.end_lineno or node.lineno
    return f"{dec}{kw} {node.name}({args}){ret}: ... [L{node.lineno}-L{end_ln}]"


def _format_py_class_header(node: ast.ClassDef) -> str:
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    dec = _format_decorator_tags(node.decorator_list)
    header = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
    end_ln = node.end_lineno or node.lineno
    return f"{dec}{header} [L{node.lineno}-L{end_ln}]"


def _slice_start_for_ast(node: ast.AST) -> int:
    decs = getattr(node, "decorator_list", None) or []
    lines = [getattr(node, "lineno", 1)] + [getattr(d, "lineno", 1) for d in decs]
    return min(lines)


def _collect_py_symbol_records(file_path: Path, tree: ast.Module | None = None) -> list[dict]:
    """Collect structured AST symbol records (depth <= 2) for a Python file."""
    if tree is None:
        content = file_path.read_text(encoding="utf-8-sig")
        tree = ast.parse(content, filename=file_path.name)

    records: list[dict] = []

    def _add_nested_in_func(
        parent_fn: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_qual: str,
        indent: str,
    ) -> None:
        for sub in _iter_lexical_nested_defs(parent_fn):
            end_ln = sub.end_lineno or sub.lineno
            span = end_ln - sub.lineno + 1
            if span < 4 and not getattr(sub, "decorator_list", None):
                continue
            if isinstance(sub, ast.ClassDef):
                disp = _format_py_class_header(sub)
                kind = "class"
            else:
                disp = _format_py_func_header(sub)
                kind = "def"
            records.append({
                "kind": kind,
                "name": sub.name,
                "qualname": f"{parent_qual}.{sub.name}",
                "indent": indent,
                "display": disp,
                "lineno": sub.lineno,
                "slice_start": _slice_start_for_ast(sub),
                "end_lineno": end_ln,
                "node": sub,
            })

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            end_ln = node.end_lineno or node.lineno
            records.append({
                "kind": "class",
                "name": node.name,
                "qualname": node.name,
                "indent": "  ",
                "display": _format_py_class_header(node),
                "lineno": node.lineno,
                "slice_start": _slice_start_for_ast(node),
                "end_lineno": end_ln,
                "node": node,
            })
            is_model = _is_model_or_dataclass(node)
            for sub in node.body:
                if is_model and isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    ann = ast.unparse(sub.annotation)
                    if sub.value is not None:
                        val_str = ast.unparse(sub.value)
                        if len(val_str) > 48:
                            val_str = val_str[:45] + "..."
                        field_disp = f"{sub.target.id}: {ann} = {val_str}"
                    else:
                        field_disp = f"{sub.target.id}: {ann}"
                    sub_end = getattr(sub, "end_lineno", sub.lineno) or sub.lineno
                    records.append({
                        "kind": "field",
                        "name": sub.target.id,
                        "qualname": f"{node.name}.{sub.target.id}",
                        "indent": "    ",
                        "display": field_disp,
                        "lineno": sub.lineno,
                        "slice_start": sub.lineno,
                        "end_lineno": sub_end,
                        "node": sub,
                    })
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sub_end = sub.end_lineno or sub.lineno
                    m_qual = f"{node.name}.{sub.name}"
                    records.append({
                        "kind": "def",
                        "name": sub.name,
                        "qualname": m_qual,
                        "indent": "    ",
                        "display": _format_py_func_header(sub),
                        "lineno": sub.lineno,
                        "slice_start": _slice_start_for_ast(sub),
                        "end_lineno": sub_end,
                        "node": sub,
                    })
                    _add_nested_in_func(sub, m_qual, "      ")
                elif isinstance(sub, ast.ClassDef):
                    sub_end = sub.end_lineno or sub.lineno
                    records.append({
                        "kind": "class",
                        "name": sub.name,
                        "qualname": f"{node.name}.{sub.name}",
                        "indent": "    ",
                        "display": _format_py_class_header(sub),
                        "lineno": sub.lineno,
                        "slice_start": _slice_start_for_ast(sub),
                        "end_lineno": sub_end,
                        "node": sub,
                    })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_ln = node.end_lineno or node.lineno
            records.append({
                "kind": "def",
                "name": node.name,
                "qualname": node.name,
                "indent": "  ",
                "display": _format_py_func_header(node),
                "lineno": node.lineno,
                "slice_start": _slice_start_for_ast(node),
                "end_lineno": end_ln,
                "node": node,
            })
            _add_nested_in_func(node, node.name, "    ")
    return records


def _extract_py_symbols(file_path: Path) -> list[str]:
    """Extract AST classes, model fields, functions, and depth-2 nested defs with line ranges."""
    try:
        content = file_path.read_text(encoding="utf-8-sig")
        tree = ast.parse(content, filename=file_path.name)
    except Exception as exc:
        return [f"  (failed to parse AST: {exc})"]

    results = []
    summary = _first_meaningful_doc_line(ast.get_docstring(tree))
    if summary:
        results.append(f'  """{summary}"""')

    for rec in _collect_py_symbol_records(file_path, tree):
        results.append(f"{rec['indent']}{rec['display']}")
    return results


def _brace_block_end(lines: list[str], start_idx: int, start_col: int = 0) -> int:
    """Find 1-based inclusive end line for a brace-delimited JS/CSS block starting at (start_idx, start_col)."""
    depth = 0
    seen_open = False
    in_block_comment = False
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        i = start_col if idx == start_idx else 0
        n = len(line)
        in_str: str | None = None
        while i < n:
            ch = line[i]
            nxt = line[i + 1] if i + 1 < n else ""
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue
            if in_str is not None:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch == "/" and nxt == "/":
                break
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue
            if ch in ("'", '"', "`"):
                in_str = ch
                i += 1
                continue
            if ch == "{":
                depth += 1
                seen_open = True
            elif ch == "}" and seen_open:
                depth -= 1
                if depth <= 0:
                    return idx + 1
            i += 1
    return start_idx + 1


def _find_js_func_body_brace(lines: list[str], start_idx: int) -> tuple[int, int]:
    """Skip parameter list (including destructured object params) to locate the body opening '{'."""
    paren_depth = 0
    seen_paren = False
    for idx in range(start_idx, min(len(lines), start_idx + 12)):
        line = lines[idx]
        in_str: str | None = None
        i = 0
        while i < len(line):
            ch = line[i]
            nxt = line[i + 1] if i + 1 < len(line) else ""
            if in_str is not None:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch == "/" and nxt == "/":
                break
            if ch in ("'", '"', "`"):
                in_str = ch
                i += 1
                continue
            if ch == "(":
                paren_depth += 1
                seen_paren = True
            elif ch == ")" and seen_paren:
                paren_depth -= 1
            elif ch == "{" and seen_paren and paren_depth <= 0:
                return idx, i
            i += 1
    return start_idx, max(0, lines[start_idx].rfind("{"))


JS_FUNC_RE = re.compile(
    r"^(export\s+(?:default\s+)?)?(async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\((.*)$"
)
JS_CLASS_RE = re.compile(
    r"^(export\s+(?:default\s+)?)?class\s+([A-Za-z_$][\w$]*)(?:\s+extends\s+([\w.$]+))?"
)
JS_METHOD_RE = re.compile(
    r"^  (?:static\s+)?(async\s+)?([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{"
)
JS_METHOD_SKIP = frozenset({"if", "for", "while", "switch", "catch", "else"})


def _collect_js_symbol_records(file_path: Path) -> list[dict]:
    """Extract top-level JS functions, classes, and class methods via stdlib regex + brace matching."""
    lines = file_path.read_text(encoding="utf-8-sig").splitlines()
    records: list[dict] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        ln = idx + 1
        m_cls = JS_CLASS_RE.match(line)
        if m_cls:
            name = m_cls.group(2)
            ext = f" extends {m_cls.group(3)}" if m_cls.group(3) else ""
            exp = m_cls.group(1) or ""
            open_col = max(0, line.find("{"))
            end_ln = _brace_block_end(lines, idx, start_col=open_col)
            records.append({
                "kind": "class",
                "name": name,
                "qualname": name,
                "indent": "  ",
                "display": f"{exp}class {name}{ext} [L{ln}-L{end_ln}]",
                "lineno": ln,
                "slice_start": ln,
                "end_lineno": end_ln,
            })
            for sub_idx in range(idx + 1, min(end_ln, len(lines))):
                m_meth = JS_METHOD_RE.match(lines[sub_idx])
                if m_meth and m_meth.group(2) not in JS_METHOD_SKIP:
                    m_name = m_meth.group(2)
                    m_async = m_meth.group(1) or ""
                    m_params = m_meth.group(3).strip()
                    m_ln = sub_idx + 1
                    b_idx, b_col = _find_js_func_body_brace(lines, sub_idx)
                    m_end = _brace_block_end(lines, b_idx, start_col=b_col)
                    records.append({
                        "kind": "def",
                        "name": m_name,
                        "qualname": f"{name}.{m_name}",
                        "indent": "    ",
                        "display": f"{m_async}{m_name}({m_params}) [L{m_ln}-L{m_end}]",
                        "lineno": m_ln,
                        "slice_start": m_ln,
                        "end_lineno": m_end,
                    })
            idx = end_ln
            continue
        m_fn = JS_FUNC_RE.match(line)
        if m_fn:
            exp = m_fn.group(1) or ""
            is_async = m_fn.group(2) or ""
            name = m_fn.group(3)
            raw_tail = m_fn.group(4)
            params = raw_tail.split(")", 1)[0].strip() if ")" in raw_tail else "..."
            b_idx, b_col = _find_js_func_body_brace(lines, idx)
            end_ln = _brace_block_end(lines, b_idx, start_col=b_col)
            records.append({
                "kind": "def",
                "name": name,
                "qualname": name,
                "indent": "  ",
                "display": f"{exp}{is_async}function {name}({params}) [L{ln}-L{end_ln}]",
                "lineno": ln,
                "slice_start": ln,
                "end_lineno": end_ln,
            })
            idx = end_ln
            continue
        idx += 1
    return records


CSS_SINGLE_SECTION_RE = re.compile(r"^/\*\s*={3,}\s*(.+?)\s*={3,}\s*\*/\s*$")
CSS_BANNER_START_RE = re.compile(r"^/\*\s*={5,}\s*$")
CSS_BANNER_END_RE = re.compile(r"^={5,}\s*\*/\s*$")
CSS_AT_BLOCK_RE = re.compile(r"^(@(?:media|container)\b[^{]+)\{")


def _collect_css_symbol_records(file_path: Path) -> list[dict]:
    """Extract CSS section headers and @media/@container blocks with line ranges."""
    lines = file_path.read_text(encoding="utf-8-sig").splitlines()
    raw_sections: list[tuple[int, str]] = []
    at_blocks: list[dict] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        m_single = CSS_SINGLE_SECTION_RE.match(stripped)
        if m_single:
            raw_sections.append((idx + 1, m_single.group(1).strip()))
            idx += 1
            continue
        if CSS_BANNER_START_RE.match(stripped):
            title = ""
            j = idx + 1
            while j < len(lines) and j <= idx + 8:
                cand = lines[j].strip()
                if CSS_BANNER_END_RE.match(cand) or "*/" in cand:
                    break
                if not title and cand and not set(cand) <= {"=", "-", "*"}:
                    title = cand
                j += 1
            if title:
                raw_sections.append((idx + 1, title))
            idx = j + 1
            continue
        m_at = CSS_AT_BLOCK_RE.match(stripped)
        if m_at:
            end_ln = _brace_block_end(lines, idx)
            hdr = " ".join(m_at.group(1).split())
            at_blocks.append({
                "kind": "css_block",
                "name": hdr,
                "qualname": hdr,
                "indent": "  ",
                "display": f"{hdr} [L{idx + 1}-L{end_ln}]",
                "lineno": idx + 1,
                "slice_start": idx + 1,
                "end_lineno": end_ln,
            })
        idx += 1

    records: list[dict] = []
    for pos, (start_ln, title) in enumerate(raw_sections):
        end_ln = (raw_sections[pos + 1][0] - 1) if pos + 1 < len(raw_sections) else len(lines)
        records.append({
            "kind": "css_section",
            "name": title,
            "qualname": title,
            "indent": "  ",
            "display": f"/* {title} */ [L{start_ln}-L{end_ln}]",
            "lineno": start_ln,
            "slice_start": start_ln,
            "end_lineno": end_ln,
        })
    records.extend(at_blocks)
    records.sort(key=lambda r: (r["lineno"], 0 if r["kind"] == "css_section" else 1))
    return records


def _extract_file_symbol_records(file_path: Path) -> list[dict]:
    suffix = file_path.suffix.lower()
    if suffix == ".py":
        return _collect_py_symbol_records(file_path)
    if suffix == ".js":
        return _collect_js_symbol_records(file_path)
    if suffix == ".css":
        return _collect_css_symbol_records(file_path)
    return []


def _source_path_allowed(root: Path, path: Path) -> bool:
    """Reject private directories, secrets, benchmark answers, links, and resolved escapes before any read."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if not relative.parts:
        return _inside(root, path)
    rel_posix = relative.as_posix().lower()
    if rel_posix in BLOCKED_RELATIVE_PATHS:
        return False
    leaf = relative.parts[-1].lower()
    if (leaf == ".env" or leaf.startswith(".env.")) and leaf not in SAFE_ENV_TEMPLATES:
        return False
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


def frontend_source_files(root: Path, target: str | None = None) -> list[Path]:
    """Return allowed .js and .css files under web/static/."""
    root = root.resolve()
    static_root = root / "web" / "static"
    if not static_root.is_dir() or not _source_path_allowed(root, static_root):
        return []
    target_path = (root / target) if target else static_root
    if not _source_path_allowed(root, target_path) or not target_path.exists():
        return []
    if target_path.is_file():
        if target_path.resolve().is_relative_to(static_root.resolve()) and target_path.suffix.lower() in {".js", ".css"}:
            return [target_path]
        return []
    if not target_path.resolve().is_relative_to(static_root.resolve()):
        if not static_root.resolve().is_relative_to(target_path.resolve()):
            return []
        target_path = static_root
    found = []
    for directory, dirs, files in os.walk(target_path, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(name for name in dirs if _source_path_allowed(root, parent / name))
        for name in sorted(files):
            path = parent / name
            if path.suffix.lower() in {".js", ".css"} and _source_path_allowed(root, path):
                found.append(path)
    return sorted(found)


def _resolve_symbol_targets(root: Path, targets: str | list[str] | None) -> list[Path]:
    """Collect .py plus frontend .js/.css when a web/static target is explicitly requested."""
    if targets is None or targets == []:
        return source_files(root)
    if isinstance(targets, str):
        targets_list = [targets]
    else:
        targets_list = list(targets)
    collected: list[Path] = []
    seen: set[Path] = set()
    for t in targets_list:
        for p in source_files(root, t):
            if p not in seen:
                seen.add(p)
                collected.append(p)
        for p in frontend_source_files(root, t):
            if p not in seen:
                seen.add(p)
                collected.append(p)
    return sorted(collected)


def show_symbols(root: Path, target: str | list[str] | None = None) -> bool:
    """Print AST/regex symbol maps for safe source files without importing them."""
    root = root.resolve()
    files = _resolve_symbol_targets(root, target)

    if not files:
        label = ", ".join(target) if isinstance(target, list) else (target or ".")
        print(f"No allowed Python/JS/CSS source files found in {label}")
        return False

    for p in files:
        rel = p.relative_to(root).as_posix()
        print(f"\n--- {rel} ---")
        suffix = p.suffix.lower()
        if suffix == ".py":
            symbols = _extract_py_symbols(p)
        else:
            symbols = [f"{r['indent']}{r['display']}" for r in _extract_file_symbol_records(p)]
        if not symbols:
            print("  (no top-level classes or functions)")
        else:
            for s in symbols:
                print(s)
    return True


def find_symbol(root: Path, symbol_name: str) -> bool:
    """Search Python AST and web/static JS/CSS definitions for exact or partial symbol name."""
    root = root.resolve()
    matches = []
    target_lower = symbol_name.lower()
    all_files = sorted(set(source_files(root)) | set(frontend_source_files(root)))

    for p in all_files:
        try:
            records = _extract_file_symbol_records(p)
        except Exception:
            continue
        rel = p.relative_to(root).as_posix()
        for rec in records:
            if target_lower in rec["name"].lower() or target_lower in rec["qualname"].lower():
                prefix = rec["indent"][2:] if len(rec["indent"]) >= 2 else ""
                matches.append(f"{rel}:{rec['lineno']}: {prefix}{rec['display']}")

    if not matches:
        print(f"No AST definitions found matching '{symbol_name}'")
        return False

    print(f"Found {len(matches)} symbol definition(s) matching '{symbol_name}':")
    for m in matches:
        print(f"  {m}")
    return True


SLICE_RANGE_RE = re.compile(r"^L?(\d+)-L?(\d+)$", re.IGNORECASE)


def _resolve_symbol_in_records(records: list[dict], symbol_query: str) -> dict | None:
    """Resolve a symbol target (Class, Class.method, nested name, or JS/CSS symbol) inside a file."""
    q_low = symbol_query.lower()
    for key in ("qualname", "name"):
        exact = [r for r in records if r[key] == symbol_query and r["kind"] != "field"]
        if exact:
            return exact[0]
    for key in ("qualname", "name"):
        ci = [r for r in records if r[key].lower() == q_low and r["kind"] != "field"]
        if ci:
            return ci[0]
    partial = [r for r in records if q_low in r["qualname"].lower() and r["kind"] != "field"]
    if len(partial) == 1:
        return partial[0]
    if partial:
        return partial[0]
    return None


def _render_file_slice(
    root: Path,
    file_path: Path,
    start_line: int,
    end_line: int,
    max_lines: int = MAX_SLICE_DEFAULT,
) -> list[str]:
    rel = file_path.relative_to(root).as_posix()
    lines = file_path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        return [f"# {rel} [L0-L0]", "  (empty file)"]
    s = max(1, start_line)
    e = min(len(lines), end_line)
    if s > e:
        return [f"# {rel} [L{s}-L{e}] (invalid empty range)"]
    truncated = False
    shown_end = e
    if e - s + 1 > max_lines:
        shown_end = s + max_lines - 1
        truncated = True
    out = [f"# {rel} [L{s}-L{shown_end}]"]
    for ln in range(s, shown_end + 1):
        out.append(f"L{ln:04d}: {lines[ln - 1]}")
    if truncated:
        out.append(
            f"# ... [truncated {e - shown_end} lines; use --slice {rel} L{shown_end + 1}-L{e}]"
        )
    return out


def show_slice(
    root: Path,
    raw_path: str,
    target: str,
    max_lines: int = MAX_SLICE_DEFAULT,
) -> bool:
    """Print bounded numbered lines for a validated .py, .js, or .css file slice or symbol."""
    root = root.resolve()
    cand = Path(raw_path)
    target_path = cand if cand.is_absolute() else (root / cand)
    try:
        target_path = target_path.resolve()
    except OSError:
        print(f"Invalid slice path: {raw_path}")
        return False
    if (
        not _source_path_allowed(root, target_path)
        or not target_path.is_file()
        or target_path.suffix.lower() not in {".py", ".js", ".css"}
    ):
        print(f"Refused or unsupported slice target: {raw_path}")
        return False

    m_range = SLICE_RANGE_RE.match(target.strip())
    if m_range:
        start_ln, end_ln = int(m_range.group(1)), int(m_range.group(2))
        if start_ln < 1 or end_ln < start_ln:
            print(f"Invalid line range '{target}' (expected 1-based start <= end).")
            return False
    else:
        records = _extract_file_symbol_records(target_path)
        rec = _resolve_symbol_in_records(records, target.strip())
        if rec is None:
            print(f"Symbol '{target}' not found in {target_path.relative_to(root).as_posix()}")
            return False
        start_ln = rec["slice_start"]
        end_ln = rec["end_lineno"]

    for ln in _render_file_slice(root, target_path, start_ln, end_ln, max_lines=max_lines):
        print(ln)
    return True


def _contract_matches(pattern: str, rel_posix: str) -> bool:
    if pattern.endswith("/*"):
        prefix = pattern[:-1]
        return rel_posix.startswith(prefix)
    if "*" in pattern:
        return re.fullmatch(re.escape(pattern).replace(r"\*", r"[^/]*"), rel_posix) is not None
    return rel_posix == pattern


def _is_test_rel(rel_posix: str) -> bool:
    return rel_posix.startswith("tests/") or rel_posix.startswith("subtrans/tests/")


def _build_repo_ast_index(root: Path) -> dict:
    """Parse all allowed .py files once and build import graph + symbol index."""
    py_files = source_files(root)
    mod_to_rel: dict[str, str] = {}
    parsed: dict[str, tuple[Path, ast.Module, list[dict]]] = {}
    for p in py_files:
        rel = p.relative_to(root).as_posix()
        parts = list(p.relative_to(root).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            mod_name = ".".join(parts[:-1])
        else:
            mod_name = ".".join(parts)
        if mod_name:
            mod_to_rel[mod_name] = rel
        try:
            content = p.read_text(encoding="utf-8-sig")
            tree = ast.parse(content, filename=p.name)
            records = _collect_py_symbol_records(p, tree)
            parsed[rel] = (p, tree, records)
        except Exception:
            continue

    reverse_deps: dict[str, set[str]] = {}
    for rel, (_, tree, _) in parsed.items():
        pkg_parts = list(Path(rel).with_suffix("").parts)[:-1]
        imported_rels: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    cand = alias.name
                    while cand:
                        if cand in mod_to_rel:
                            imported_rels.add(mod_to_rel[cand])
                            break
                        if "." not in cand:
                            break
                        cand = cand.rsplit(".", 1)[0]
            elif isinstance(node, ast.ImportFrom):
                base_mod = node.module or ""
                if node.level > 0:
                    trim = max(0, len(pkg_parts) - (node.level - 1))
                    prefix_parts = pkg_parts[:trim]
                    if base_mod:
                        prefix_parts.extend(base_mod.split("."))
                    resolved_base = ".".join(prefix_parts)
                else:
                    resolved_base = base_mod
                candidates = [resolved_base] if resolved_base else []
                for alias in node.names:
                    if alias.name != "*":
                        candidates.append(f"{resolved_base}.{alias.name}" if resolved_base else alias.name)
                for cand in candidates:
                    cur = cand
                    while cur:
                        if cur in mod_to_rel:
                            imported_rels.add(mod_to_rel[cur])
                            break
                        if "." not in cur:
                            break
                        cur = cur.rsplit(".", 1)[0]
        for target_rel in imported_rels:
            if target_rel != rel:
                reverse_deps.setdefault(target_rel, set()).add(rel)

    return {
        "parsed": parsed,
        "reverse_deps": reverse_deps,
    }


def _symbol_ref_callers(index: dict, symbol_name: str) -> set[str]:
    """Find files referencing bare symbol_name via Name or Attribute nodes."""
    hits: set[str] = set()
    for rel, (_, tree, _) in index["parsed"].items():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name) and node.id == symbol_name) or (
                isinstance(node, ast.Attribute) and node.attr == symbol_name
            ):
                hits.add(rel)
                break
    return hits


def compute_impact(
    root: Path,
    targets: list[str],
    index: dict | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve 2-hop AST reverse imports plus cross-boundary contracts to (operational_files, test_files)."""
    root = root.resolve()
    idx = index or _build_repo_ast_index(root)
    parsed = idx["parsed"]
    reverse_deps = idx["reverse_deps"]

    seed_files: set[str] = set()
    seed_symbols: set[str] = set()
    for raw in targets:
        cleaned = raw.strip().replace("\\", "/")
        if not cleaned:
            continue
        if ":" in cleaned and not Path(cleaned).drive:
            left, sym = cleaned.rsplit(":", 1)
            cleaned = left.strip()
            if sym.strip():
                seed_symbols.add(sym.strip().split(".")[-1])
        cand_path = (root / cleaned)
        if _inside(root, cand_path) and cand_path.exists():
            seed_files.add(cand_path.resolve().relative_to(root).as_posix())
            continue
        # Treat as path prefix/dir if trailing / or matching directory, else symbol lookup
        sym_leaf = cleaned.split(".")[-1]
        found_sym = False
        for rel, (_, _, recs) in parsed.items():
            for rec in recs:
                if rec["qualname"] == cleaned or rec["name"] == cleaned or rec["name"] == sym_leaf:
                    seed_files.add(rel)
                    seed_symbols.add(rec["name"])
                    found_sym = True
        for jf in frontend_source_files(root):
            jrel = jf.relative_to(root).as_posix()
            for rec in _extract_file_symbol_records(jf):
                if rec["qualname"] == cleaned or rec["name"] == cleaned or rec["name"] == sym_leaf:
                    seed_files.add(jrel)
                    found_sym = True
        if not found_sym:
            seed_files.add(cleaned.lstrip("./"))

    ops_hits: set[str] = set()
    test_hits: set[str] = set()

    def _record_hit(rel_posix: str) -> None:
        if rel_posix in seed_files:
            return
        if _is_test_rel(rel_posix):
            test_hits.add(rel_posix)
        else:
            ops_hits.add(rel_posix)

    for sf in seed_files:
        if _is_test_rel(sf):
            test_hits.add(sf)

    # Hop 1 + symbol-level callers
    hop1_ops: set[str] = set()
    for sf in seed_files:
        for caller in reverse_deps.get(sf, ()):
            _record_hit(caller)
            if not _is_test_rel(caller):
                hop1_ops.add(caller)
    for sym in seed_symbols:
        for caller in _symbol_ref_callers(idx, sym):
            _record_hit(caller)
            if not _is_test_rel(caller):
                hop1_ops.add(caller)

    # Hop 2 transitive runtime callers -> runtime + test callers
    hop2_ops: set[str] = set()
    for h1 in hop1_ops:
        for caller in reverse_deps.get(h1, ()):
            _record_hit(caller)
            if not _is_test_rel(caller):
                hop2_ops.add(caller)

    # Cross-boundary contracts against directly targeted/modified seeds
    active_set = seed_files
    coupled_ops: set[str] = set()
    for side_a, side_b, contract_tests, bidirectional in CROSS_BOUNDARY_CONTRACTS:
        hit_a = any(_contract_matches(pat, p) for pat in side_a for p in active_set)
        hit_b = any(_contract_matches(pat, p) for pat in side_b for p in active_set)
        if hit_a:
            for item in side_b:
                if "*" not in item:
                    coupled_ops.add(item)
                    _record_hit(item)
            for t in contract_tests:
                test_hits.add(t)
        if bidirectional and hit_b:
            for item in side_a:
                if "*" not in item:
                    coupled_ops.add(item)
                    _record_hit(item)
            for t in contract_tests:
                test_hits.add(t)

    # Also pull 1-hop test files for contract-coupled operational files
    for c_op in coupled_ops:
        for caller in reverse_deps.get(c_op, ()):
            if _is_test_rel(caller):
                test_hits.add(caller)

    return sorted(ops_hits), sorted(test_hits)


def show_impact(root: Path, target: str) -> bool:
    """Print 2-hop operational callers/contracts and test files impacted by path or symbol."""
    root = root.resolve()
    ops, tests = compute_impact(root, [target])
    print(f"Impact for '{target}':")
    print(f"Operational callers/coupled ({len(ops)}):")
    if ops:
        for item in ops:
            print(f"  - {item}")
    else:
        print("  (none)")
    print(f"Tests to run ({len(tests)}):")
    if tests:
        for item in tests:
            print(f"  - {item}")
    else:
        print("  (none)")
    return True


def _resolve_pack_target(
    root: Path,
    raw_target: str,
    idx: dict,
    frontend_cache: dict[str, tuple[Path, list[dict]]],
) -> list[tuple[Path, dict, list[dict]]]:
    """Resolve <SYM> or <PATH>:<SYM> into (file_path, matched_record, all_file_records)."""
    parsed = idx["parsed"]
    raw_norm = raw_target.strip()
    if ":" in raw_norm and not Path(raw_norm).drive:
        path_part, sym_part = raw_norm.rsplit(":", 1)
        rel_path = path_part.strip().replace("\\", "/").lstrip("./")
        cand = (root / rel_path).resolve()
        if not _source_path_allowed(root, cand) or not cand.is_file():
            return []
        rel = cand.relative_to(root).as_posix()
        if rel in parsed:
            recs = parsed[rel][2]
        elif rel in frontend_cache:
            recs = frontend_cache[rel][1]
        else:
            recs = _extract_file_symbol_records(cand)
        rec = _resolve_symbol_in_records(recs, sym_part.strip())
        return [(cand, rec, recs)] if rec is not None else []

    matches: list[tuple[Path, dict, list[dict]]] = []
    for rel in sorted(parsed):
        fp, _, recs = parsed[rel]
        rec = _resolve_symbol_in_records(recs, raw_norm)
        if rec is not None and (rec["qualname"] == raw_norm or rec["name"] == raw_norm):
            matches.append((fp, rec, recs))
    for rel in sorted(frontend_cache):
        fp, recs = frontend_cache[rel]
        rec = _resolve_symbol_in_records(recs, raw_norm)
        if rec is not None and (rec["qualname"] == raw_norm or rec["name"] == raw_norm):
            matches.append((fp, rec, recs))
    if not matches:
        for rel in sorted(parsed):
            fp, _, recs = parsed[rel]
            rec = _resolve_symbol_in_records(recs, raw_norm)
            if rec is not None:
                matches.append((fp, rec, recs))
    return matches[:2]


def _same_file_referenced_helpers(file_path: Path, rec: dict, all_records: list[dict]) -> list[str]:
    """Return 1-line signatures of same-file helper symbols referenced inside rec's body."""
    target_qual = rec["qualname"]
    if file_path.suffix.lower() == ".py" and "node" in rec:
        used_names: set[str] = set()
        for n in ast.walk(rec["node"]):
            if isinstance(n, ast.Name):
                used_names.add(n.id)
            elif isinstance(n, ast.Attribute):
                used_names.add(n.attr)
        helpers = [
            f"{r['indent'].strip()} {r['display']}".strip()
            for r in all_records
            if r["qualname"] != target_qual
            and r["kind"] in {"def", "class"}
            and r["name"] in used_names
            and r["lineno"] != rec["lineno"]
        ]
        return helpers[:12]

    lines = file_path.read_text(encoding="utf-8-sig").splitlines()
    body = "\n".join(lines[max(0, rec["slice_start"] - 1):rec["end_lineno"]])
    called = set(re.findall(r"\b([A-Za-z_$][\w$]*)\s*\(", body))
    return [
        r["display"]
        for r in all_records
        if r["qualname"] != target_qual and r["name"] in called and r["lineno"] != rec["lineno"]
    ][:12]


def show_pack(root: Path, targets: list[str]) -> bool:
    """Emit batched slice + same-file helper signatures + 2-hop impact for up to 4 targets."""
    root = root.resolve()
    if not targets or len(targets) > MAX_PACK_TARGETS:
        print(f"ERROR: --pack accepts 1 to {MAX_PACK_TARGETS} targets per call (got {len(targets)}).")
        return False

    idx = _build_repo_ast_index(root)
    frontend_cache: dict[str, tuple[Path, list[dict]]] = {}
    for jf in frontend_source_files(root):
        try:
            frontend_cache[jf.relative_to(root).as_posix()] = (jf, _extract_file_symbol_records(jf))
        except Exception:
            continue

    impact_seeds: list[str] = []
    any_found = False
    for target in targets:
        resolved = _resolve_pack_target(root, target, idx, frontend_cache)
        if not resolved:
            print(f"ERROR: Could not resolve --pack target '{target}'")
            return False
        any_found = True
        for fp, rec, all_recs in resolved:
            rel = fp.relative_to(root).as_posix()
            impact_seeds.append(f"{rel}:{rec['qualname']}")
            for ln in _render_file_slice(
                root, fp, rec["slice_start"], rec["end_lineno"], max_lines=MAX_PACK_SLICE_LINES
            ):
                print(ln)
            helpers = _same_file_referenced_helpers(fp, rec, all_recs)
            if helpers:
                print("# Same-file helpers referenced:")
                for h in helpers:
                    print(f"#   {h}")
            print()

    if not any_found:
        return False
    ops, tests = compute_impact(root, impact_seeds, index=idx)
    print(f"# Impact operational ({len(ops)}): {', '.join(ops) if ops else '(none)'}")
    print(f"# Impact tests ({len(tests)}): {', '.join(tests) if tests else '(none)'}")
    return True


def _install_single_hook(
    hook_path: Path,
    common_path: Path,
    content: str,
    label: str,
) -> bool:
    """Install or verify one managed hook without overwriting custom user hooks."""
    try:
        if hook_path.is_symlink() or not hook_path.resolve().is_relative_to(common_path):
            print(f"Refusing linked {label} hook.")
            return False
        if hook_path.exists():
            original_variants = {
                content.encode("utf-8"),
                content.replace("\n", "\r\n").encode("utf-8"),
            }
            if hook_path.read_bytes() in original_variants:
                print(f"Managed {label} hook already installed; unchanged.")
                return True
            print(f"Existing {label} hook preserved; integrate manually.")
            return False
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        with hook_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        hook_path.chmod(0o755)
    except OSError as exc:
        print(f"Could not install managed {label} hook: {exc}")
        return False
    print(f"{label.capitalize()} hook successfully installed at: {hook_path}")
    return True


def install_git_hooks(root: Path) -> bool:
    """Install managed pre-commit and post-commit hooks independently without replacing user hooks."""
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

    ok_pre = _install_single_hook(hooks_dir / "pre-commit", common_path, MANAGED_HOOK, "pre-commit")
    ok_post = _install_single_hook(
        hooks_dir / "post-commit", common_path, MANAGED_POST_COMMIT_HOOK, "post-commit"
    )
    return ok_pre and ok_post


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root (default: parent of scripts/)")
    parser.add_argument("--check", action="store_true", help="Check required memory files, local links and instruction byte budgets")
    parser.add_argument("--inventory", action="store_true", help="Count tracked paths by top-level directory without reading contents")
    parser.add_argument("--symbols", nargs="*", type=str, help="Show AST/JS/CSS symbol maps for one or more paths (or entire repo)")
    parser.add_argument("--find", type=str, help="Search AST and frontend symbol definitions for a class or function name")
    parser.add_argument("--slice", nargs=2, metavar=("PATH", "TARGET"), help="Print bounded numbered slice for line range (L10-L50) or symbol")
    parser.add_argument("--max-slice-lines", type=int, default=MAX_SLICE_DEFAULT, help=f"Maximum lines emitted by --slice (default: {MAX_SLICE_DEFAULT})")
    parser.add_argument("--impact", type=str, help="Show 2-hop reverse imports, cross-boundary couplings, and impacted tests")
    parser.add_argument("--pack", nargs="+", metavar="TARGET", help=f"Batched one-shot context pack (<= {MAX_PACK_TARGETS} targets): slice + helpers + impact")
    parser.add_argument("--install-hooks", action="store_true", help="Install git pre-commit and post-commit hooks")
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
            print(
                f"PASS: required memory files, local links, GEMINI.md <= {MAX_ROOT_LINES} lines / {ROOT_BUDGET} bytes, "
                f"root + subtrans skill <= {COMBINED_BUDGET} bytes"
            )
            freshness_warn = check_freshness(root)
            if freshness_warn:
                print(f"WARN: {freshness_warn}")

    if args.inventory:
        ran_action = True
        if not show_inventory(root):
            exit_code = 1

    if args.symbols is not None:
        ran_action = True
        targets = args.symbols if args.symbols else None
        if not show_symbols(root, targets):
            exit_code = 1

    if args.find:
        ran_action = True
        if not find_symbol(root, args.find):
            exit_code = 1

    if args.slice:
        ran_action = True
        if not show_slice(root, args.slice[0], args.slice[1], max_lines=max(1, args.max_slice_lines)):
            exit_code = 1

    if args.impact:
        ran_action = True
        if not show_impact(root, args.impact):
            exit_code = 1

    if args.pack:
        ran_action = True
        if not show_pack(root, args.pack):
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
