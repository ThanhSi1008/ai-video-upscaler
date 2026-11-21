"""Export tiered, prefix-cache-friendly review bundles into scratch/review_bundles/."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys


BUNDLE_DIR_REL = Path("scratch") / "review_bundles"
DOC_ECHO_RE = re.compile(r"^(?:Module:\s*)?[\w./\\-]+(?:\.py)?\s*$")

STATIC_ANCHOR_PREFIXES = (
    "GEMINI.md",
    "docs/agents/DECISIONS.md",
    "docs/agents/PROJECT_MAP.md",
    "docs/agents/VALIDATION.md",
)
STATE_FILE_REL = "docs/agents/STATE.md"

EXCLUDED_DIRS = frozenset({
    ".git",
    ".venv",
    "venv",
    "env",
    "bin",
    ".cache",
    ".pytest_cache",
    ".ruff_cache",
    "logs",
    "out",
    "scratch",
    "state",
    "models_bench",
    "samples",
    "test_frames",
    "archive_subtitles",
    "__pycache__",
    "benchmarks",
})

EXCLUDED_PATH_PREFIXES = (
    "docs/archive/",
    "scripts/oneoff/",
    ".agents/skills/modern-web-guidance/guides/",
    "subtrans/samples/",
    "subtrans/.cache/",
)

EXCLUDED_EXACT = frozenset({
    "pending_subs.json",
    "requirements.lock",
    "skills-lock.json",
    "benchmarks/upscale/ab_test/key_answer.json",
    ".gpu_active.lock",
})

SAFE_ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})

TEXT_SUFFIXES = frozenset({
    ".py", ".bat", ".ps1", ".sh", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".md", ".txt", ".json", ".ass", ".srt", ".html", ".css", ".js",
})

SAFE_DOTFILES = frozenset({".antigravityignore", ".gitignore", ".editorconfig", ".env.example"})

BINARY_SUFFIXES = frozenset({
    ".pth", ".safetensors", ".bin", ".param", ".onnx", ".engine", ".trt", ".plan", ".nip",
    ".mkv", ".mp4", ".webm", ".mov", ".avi", ".wav", ".flac", ".db", ".sqlite", ".sqlite3",
    ".pyc", ".pyo", ".pyd", ".exe", ".dll", ".so", ".whl", ".zip", ".7z", ".png", ".jpg",
    ".jpeg", ".webp", ".gif", ".ico", ".pdf", ".ttf", ".otf", ".woff", ".woff2",
})

FOCUS_PRESETS: dict[str, tuple[str, ...]] = {
    "video": ("upscale.py", "upscale.bat", "upscale_turbo.bat", "shows.yaml", "models/", "src/"),
    "subtrans": ("subtrans.py", "subtrans.bat", "subtrans/", "glossaries/"),
    "web": ("web_gui.py", "web_gui.bat", "web/"),
    "agents": ("GEMINI.md", "pytest.ini", "ruff.toml", "requirements.txt", ".antigravityignore", "docs/", ".agents/"),
    "scripts": ("scripts/",),
}

SYNC_BUNDLES: tuple[tuple[str, str, list[str]], ...] = (
    ("focus_video.txt", "focus:video", ["video"]),
    ("focus_subtrans.txt", "focus:subtrans", ["subtrans"]),
    ("focus_web.txt", "focus:web", ["web"]),
    ("focus_agents.txt", "focus:agents,scripts", ["agents", "scripts"]),
    ("focus_full.txt", "focus:full", ["full"]),
)


def _git_output(root: Path, *args: str) -> str | None:
    try:
        res = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        return res.stdout if res.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _first_meaningful_doc_line(doc: str | None) -> str:
    if not doc:
        return ""
    for raw_line in doc.strip().splitlines():
        line = raw_line.strip()
        if not line or DOC_ECHO_RE.match(line):
            continue
        return line
    return ""


def _is_excluded_rel(rel_posix: str, include_tests: bool = False) -> bool:
    low = rel_posix.lower()
    if low in EXCLUDED_EXACT:
        return True
    parts = Path(rel_posix).parts
    if not parts:
        return True
    leaf = parts[-1].lower()
    if (leaf == ".env" or leaf.startswith(".env.")) and leaf not in SAFE_ENV_TEMPLATES:
        return True
    if any(p.startswith(".temp_") for p in parts):
        return True
    if any(p.lower() in EXCLUDED_DIRS for p in parts):
        return True
    if any(low.startswith(pref) for pref in EXCLUDED_PATH_PREFIXES):
        return True
    if not include_tests and (low.startswith("tests/") or low.startswith("subtrans/tests/")):
        return True
    suffix = Path(rel_posix).suffix.lower()
    if suffix in BINARY_SUFFIXES:
        return True
    if parts[0].startswith(".") and len(parts) == 1 and leaf not in SAFE_DOTFILES:
        return True
    if suffix and suffix not in TEXT_SUFFIXES:
        return False if leaf in SAFE_DOTFILES else True
    return False


def _is_static_anchor(rel_posix: str) -> bool:
    if rel_posix in STATIC_ANCHOR_PREFIXES:
        return True
    return rel_posix.startswith(".agents/rules/") and rel_posix.endswith(".md")


def _matches_focus(rel_posix: str, focus_tokens: list[str]) -> bool:
    for tok in focus_tokens:
        prefixes = FOCUS_PRESETS.get(tok)
        if prefixes is None:
            prefixes = (tok,)
        for pref in prefixes:
            if pref.endswith("/"):
                if rel_posix.startswith(pref):
                    return True
            elif rel_posix == pref:
                return True
    return False


def _iter_lexical_nested(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    nested: list[ast.AST] = []

    def _walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nested.append(stmt)
                continue
            if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                _walk(stmt.body)
                _walk(stmt.orelse)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                _walk(stmt.body)
            elif isinstance(stmt, ast.Try):
                _walk(stmt.body)
                for h in stmt.handlers:
                    _walk(h.body)
                _walk(stmt.orelse)
                _walk(stmt.finalbody)
            elif hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar):
                _walk(stmt.body)
                for h in stmt.handlers:
                    _walk(h.body)
                _walk(stmt.orelse)
                _walk(stmt.finalbody)
            elif hasattr(ast, "Match") and isinstance(stmt, ast.Match):
                for c in stmt.cases:
                    _walk(c.body)

    _walk(func_node.body)
    return nested


def _format_decorators(decs: list[ast.expr]) -> list[str]:
    tags: list[str] = []
    for d in decs[:3]:
        s = f"@{ast.unparse(d)}"
        tags.append(s[:61] + "..." if len(s) > 64 else s)
    return tags


def _is_model_or_dataclass(node: ast.ClassDef) -> bool:
    for b in node.bases:
        bs = ast.unparse(b)
        if bs == "BaseModel" or bs.endswith(".BaseModel"):
            return True
    return any("dataclass" in ast.unparse(d) for d in node.decorator_list)


def _format_py_skeleton(rel_posix: str, text: str) -> str:
    try:
        tree = ast.parse(text, filename=rel_posix)
    except SyntaxError as exc:
        return f"### [AST_SKELETON] {rel_posix}\n# (SyntaxError: {exc})\n"

    lines = [f"### [AST_SKELETON] {rel_posix}"]
    summary = _first_meaningful_doc_line(ast.get_docstring(tree))
    if summary:
        lines.append(f'"""{summary}"""')

    for node in tree.body:
        if isinstance(node, ast.Import):
            names = ", ".join(
                f"{a.name} as {a.asname}" if a.asname else a.name for a in node.names
            )
            lines.append(f"import {names}")
        elif isinstance(node, ast.ImportFrom):
            dots = "." * node.level
            mod = node.module or ""
            names = ", ".join(
                f"{a.name} as {a.asname}" if a.asname else a.name for a in node.names
            )
            if len(names) > 100:
                names = names[:97] + "..."
            lines.append(f"from {dots}{mod} import {names}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            stmt_src = " ".join(ast.unparse(node).splitlines())
            if len(stmt_src) > 120:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    stmt_src = f"{ast.unparse(node.targets[0])} = <{type(node.value).__name__} ...>"
                elif isinstance(node, ast.AnnAssign):
                    stmt_src = f"{ast.unparse(node.target)}: {ast.unparse(node.annotation)} = ..."
                else:
                    stmt_src = stmt_src[:117] + "..."
            lines.append(stmt_src)
        elif isinstance(node, ast.ClassDef):
            for tag in _format_decorators(node.decorator_list):
                lines.append(tag)
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            hdr = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
            lines.append(f"{hdr}  # [L{node.lineno}-L{node.end_lineno or node.lineno}]")
            is_model = _is_model_or_dataclass(node)
            for sub in node.body:
                if is_model and isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    ann = ast.unparse(sub.annotation)
                    if sub.value is not None:
                        v = ast.unparse(sub.value)
                        if len(v) > 48:
                            v = v[:45] + "..."
                        lines.append(f"    {sub.target.id}: {ann} = {v}")
                    else:
                        lines.append(f"    {sub.target.id}: {ann}")
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for tag in _format_decorators(sub.decorator_list):
                        lines.append(f"    {tag}")
                    kw = "async def" if isinstance(sub, ast.AsyncFunctionDef) else "def"
                    ret = f" -> {ast.unparse(sub.returns)}" if sub.returns else ""
                    lines.append(
                        f"    {kw} {sub.name}({ast.unparse(sub.args)}){ret}: ...  # [L{sub.lineno}-L{sub.end_lineno or sub.lineno}]"
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for tag in _format_decorators(node.decorator_list):
                lines.append(tag)
            kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            lines.append(
                f"{kw} {node.name}({ast.unparse(node.args)}){ret}: ...  # [L{node.lineno}-L{node.end_lineno or node.lineno}]"
            )
            for sub in _iter_lexical_nested(node):
                sub_end = getattr(sub, "end_lineno", sub.lineno) or sub.lineno
                if (sub_end - sub.lineno + 1) < 4 and not getattr(sub, "decorator_list", None):
                    continue
                for tag in _format_decorators(getattr(sub, "decorator_list", [])):
                    lines.append(f"    {tag}")
                if isinstance(sub, ast.ClassDef):
                    lines.append(f"    class {sub.name}: ...  # [L{sub.lineno}-L{sub_end}]")
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    skw = "async def" if isinstance(sub, ast.AsyncFunctionDef) else "def"
                    sret = f" -> {ast.unparse(sub.returns)}" if sub.returns else ""
                    lines.append(
                        f"    {skw} {sub.name}({ast.unparse(sub.args)}){sret}: ...  # [L{sub.lineno}-L{sub_end}]"
                    )
    return "\n".join(lines) + "\n"


def _format_full_file(rel_posix: str, text: str, *, line_numbers: bool = True) -> str:
    src_lines = text.splitlines()
    header = f"### [FULL] {rel_posix} ({len(src_lines)} lines)\n"
    if not line_numbers:
        return header + ("\n".join(src_lines) + "\n" if src_lines else "")
    numbered = [f"L{idx:04d}: {ln}" for idx, ln in enumerate(src_lines, 1)]
    return header + "\n".join(numbered) + "\n"


def _collect_repo_snapshot(
    root: Path,
    include_tests: bool,
    build_skeletons: bool = True,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]], set[str]]:
    """Read allowed repo files once and prebuild AST skeletons + import edges."""
    tracked_raw = _git_output(root, "ls-files", "-z")
    tracked_set: set[str] = set()
    candidates: set[str] = set()
    if tracked_raw is not None:
        for p in tracked_raw.split("\0"):
            if p:
                tracked_set.add(p)
                candidates.add(p)
        others_raw = _git_output(root, "ls-files", "-z", "--others", "--exclude-standard")
        if others_raw:
            for p in others_raw.split("\0"):
                if p:
                    candidates.add(p)
    else:
        for p in root.rglob("*"):
            if p.is_file():
                candidates.add(p.relative_to(root).as_posix())

    texts: dict[str, str] = {}
    skeletons: dict[str, str] = {}
    mod_to_rel: dict[str, str] = {}
    parsed_py: dict[str, ast.Module] = {}

    for rel in sorted(candidates):
        if _is_excluded_rel(rel, include_tests=include_tests):
            continue
        fp = root / rel
        try:
            if not fp.is_file() or fp.is_symlink() or not fp.resolve().is_relative_to(root.resolve()):
                continue
            content = fp.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue
        if "\x00" in content:
            continue
        texts[rel] = content
        if build_skeletons and rel.endswith(".py"):
            skeletons[rel] = _format_py_skeleton(rel, content)
            parts = list(Path(rel).with_suffix("").parts)
            mod_name = ".".join(parts[:-1] if parts and parts[-1] == "__init__" else parts)
            if mod_name:
                mod_to_rel[mod_name] = rel
            try:
                parsed_py[rel] = ast.parse(content, filename=fp.name)
            except SyntaxError:
                pass

    # Build bidirectional 1-hop import neighbors for latest_diff skeletons
    neighbors: dict[str, set[str]] = {}
    if build_skeletons:
        for rel, tree in parsed_py.items():
            pkg_parts = list(Path(rel).with_suffix("").parts)[:-1]
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        cand = alias.name
                        while cand:
                            if cand in mod_to_rel:
                                imported.add(mod_to_rel[cand])
                                break
                            if "." not in cand:
                                break
                            cand = cand.rsplit(".", 1)[0]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level > 0:
                        trim = max(0, len(pkg_parts) - (node.level - 1))
                        pref = pkg_parts[:trim] + (base.split(".") if base else [])
                        base = ".".join(pref)
                    cands = [base] if base else []
                    for alias in node.names:
                        if alias.name != "*":
                            cands.append(f"{base}.{alias.name}" if base else alias.name)
                    for cand in cands:
                        cur = cand
                        while cur:
                            if cur in mod_to_rel:
                                imported.add(mod_to_rel[cur])
                                break
                            if "." not in cur:
                                break
                            cur = cur.rsplit(".", 1)[0]
            for target in imported:
                if target != rel:
                    neighbors.setdefault(rel, set()).add(target)
                    neighbors.setdefault(target, set()).add(rel)

    return texts, skeletons, neighbors, tracked_set


def _compute_diff_context(root: Path, diff_ref: str | None) -> tuple[list[str], str, str]:
    """Return (changed_rel_paths, log_header, unified_diff_text)."""
    status_out = (_git_output(root, "status", "--porcelain") or "").strip()
    is_dirty = bool(status_out)
    log_stat = (_git_output(root, "log", "-1", "--stat") or "").strip()

    if diff_ref and ".." in diff_ref:
        names_raw = _git_output(root, "diff", "--name-only", diff_ref) or ""
        diff_raw = _git_output(root, "diff", diff_ref) or ""
        changed = [ln.strip() for ln in names_raw.splitlines() if ln.strip()]
        return changed, log_stat, diff_raw.strip()

    if diff_ref and diff_ref != "HEAD":
        names_raw = _git_output(root, "diff", "--name-only", diff_ref) or ""
        diff_raw = _git_output(root, "diff", diff_ref) or ""
        changed = [ln.strip() for ln in names_raw.splitlines() if ln.strip()]
        return changed, log_stat, diff_raw.strip()

    if is_dirty:
        names_raw = _git_output(root, "diff", "--name-only", "HEAD") or ""
        untracked_raw = _git_output(root, "ls-files", "--others", "--exclude-standard") or ""
        diff_raw = _git_output(root, "diff", "HEAD") or ""
        changed = sorted(set(
            [ln.strip() for ln in names_raw.splitlines() if ln.strip()]
            + [ln.strip() for ln in untracked_raw.splitlines() if ln.strip()]
        ))
        return changed, log_stat, diff_raw.strip()

    has_parent = _git_output(root, "rev-parse", "--verify", "HEAD~1") is not None
    if has_parent:
        names_raw = _git_output(root, "diff", "--name-only", "HEAD~1..HEAD") or ""
        diff_raw = _git_output(root, "diff", "HEAD~1..HEAD") or ""
    else:
        names_raw = _git_output(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD") or ""
        diff_raw = _git_output(root, "show", "--format=", "HEAD") or ""
    changed = [ln.strip() for ln in names_raw.splitlines() if ln.strip()]
    return changed, log_stat, diff_raw.strip()


def _render_bundle(
    root: Path,
    mode: str,
    texts: dict[str, str],
    skeletons: dict[str, str],
    neighbors: dict[str, set[str]],
    branch: str,
    head_short: str,
    commit_subject: str,
    focus_tokens: list[str] | None = None,
    changed_paths: list[str] | None = None,
    log_stat: str = "",
    diff_text: str = "",
) -> tuple[str, int, int]:
    """Assemble bundle in strictly deterministic static-first, volatile-last order."""
    all_paths = sorted(texts.keys())
    if mode.startswith("focus:"):
        raw_tokens = [t.strip().lower() for t in mode.split(":", 1)[1].split(",") if t.strip()]
        tok_list = focus_tokens if focus_tokens is not None else raw_tokens
        if ("full" in raw_tokens and len(raw_tokens) > 1) or ("full" in tok_list and len(tok_list) > 1):
            raise ValueError("'full' cannot be combined with other --focus tokens")

    if mode == "focus:full":
        parts: list[str] = [
            "# AI-VIDEO-UPSCALER REVIEW BUNDLE\n",
            "## 1. REPOSITORY TREE\n" + "\n".join(all_paths) + "\n",
            "## 2. FULL REPOSITORY FILES\n",
        ]
        for p in all_paths:
            parts.append(_format_full_file(p, texts[p], line_numbers=False))

        parts.append("## 3. GIT LOG & DIFF\n")
        if log_stat:
            parts.append("### git log -1 --stat\n" + log_stat + "\n")
        else:
            parts.append("(git metadata unavailable)\n")

        full_count = len(all_paths)
        skel_count = 0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts.append(
            "## 4. BUNDLE METADATA\n"
            f"- Branch / HEAD: {branch}@{head_short}\n"
            f"- Commit Subject: {commit_subject}\n"
            f"- Timestamp: {ts}\n"
            f"- Mode: {mode}\n"
            f"- Full Files: {full_count} | AST Skeletons: {skel_count}\n"
        )
        body_so_far = "\n".join(parts)
        est = len(body_so_far) + 42
        for _ in range(2):
            tail = f"- Characters: {est} | Est. Tokens: {est // 4}\n"
            est = len(body_so_far) + len(tail)
        final_text = body_so_far + f"- Characters: {est} | Est. Tokens: {est // 4}\n"
        return final_text, full_count, skel_count

    anchors = [p for p in all_paths if _is_static_anchor(p)]
    anchor_set = set(anchors)

    if mode == "full":
        full_set = {p for p in all_paths if p != STATE_FILE_REL and p not in anchor_set}
        skel_set: set[str] = set()
    elif mode.startswith("focus:"):
        tokens = focus_tokens or []
        full_set = {
            p for p in all_paths
            if p != STATE_FILE_REL and p not in anchor_set and _matches_focus(p, tokens)
        }
        skel_set = {p for p in all_paths if p in skeletons and p not in full_set and p not in anchor_set}
    else:  # latest_diff
        changed_valid = {p for p in (changed_paths or []) if p in texts}
        full_set = {p for p in changed_valid if p != STATE_FILE_REL and p not in anchor_set}
        dep_cands: set[str] = set()
        for cp in changed_valid:
            dep_cands.update(neighbors.get(cp, ()))
        skel_set = {
            p for p in sorted(dep_cands)
            if p in skeletons and p not in full_set and p not in anchor_set
        }

    parts: list[str] = []
    # 1. Static deterministic header + directory tree
    parts.append("# AI-VIDEO-UPSCALER REVIEW BUNDLE\n")
    parts.append("## 1. REPOSITORY TREE\n" + "\n".join(all_paths) + "\n")

    # 2. Static [FULL] core anchors
    parts.append("## 2. STATIC CORE ANCHORS\n")
    for p in anchors:
        parts.append(_format_full_file(p, texts[p]))

    # 3. Deterministic [AST_SKELETON] context modules
    parts.append("## 3. AST SKELETON MODULES\n")
    if skel_set:
        for p in sorted(skel_set):
            parts.append(skeletons[p])
    else:
        parts.append("(none)\n")

    # 4. Volatile operational state
    parts.append("## 4. VOLATILE STATE\n")
    if STATE_FILE_REL in texts:
        parts.append(_format_full_file(STATE_FILE_REL, texts[STATE_FILE_REL]))

    # 5. Volatile focused / changed [FULL] files
    parts.append("## 5. FOCUSED / CHANGED FULL FILES\n")
    for p in sorted(full_set):
        parts.append(_format_full_file(p, texts[p]))

    # 6. Volatile Git commit metadata & unified diff
    parts.append("## 6. GIT LOG & DIFF\n")
    if log_stat:
        parts.append("### git log -1 --stat\n" + log_stat + "\n")
    if diff_text:
        parts.append("### unified diff\n" + diff_text + "\n")
    elif not log_stat:
        parts.append("(git metadata unavailable)\n")

    full_count = len(anchors) + (1 if STATE_FILE_REL in texts else 0) + len(full_set)
    skel_count = len(skel_set)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 7. Volatile bundle metadata summary
    meta_placeholder = (
        "## 7. BUNDLE METADATA\n"
        f"- Branch / HEAD: {branch}@{head_short}\n"
        f"- Commit Subject: {commit_subject}\n"
        f"- Timestamp: {ts}\n"
        f"- Mode: {mode}\n"
        f"- Full Files: {full_count} | AST Skeletons: {skel_count}\n"
    )
    parts.append(meta_placeholder)
    body_so_far = "\n".join(parts)
    # Two-pass exact length calculation so Characters / Est. Tokens match output byte-for-byte
    est = len(body_so_far) + 42
    for _ in range(2):
        tail = f"- Characters: {est} | Est. Tokens: {est // 4}\n"
        est = len(body_so_far) + len(tail)
    final_text = body_so_far + f"- Characters: {est} | Est. Tokens: {est // 4}\n"
    return final_text, full_count, skel_count


def _atomic_write_bundle(root: Path, target_path: Path, content: str, tracked_set: set[str]) -> None:
    root_res = root.resolve()
    resolved = target_path.resolve()
    if resolved.is_symlink():
        raise ValueError(f"Refusing to write through symlink: {target_path}")
    if resolved.is_relative_to(root_res):
        rel_posix = resolved.relative_to(root_res).as_posix()
        if rel_posix in tracked_set:
            raise ValueError(f"Refusing to overwrite Git-tracked file: {rel_posix}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(content, encoding="utf-8", newline="\n")
    os.replace(tmp_path, target_path)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sync-latest", action="store_true", help="Atomically refresh all 6 fixed review bundles")
    group.add_argument("--focus", type=str, help="Comma-separated focus preset(s) or path prefixes")
    group.add_argument("--diff", nargs="?", const="HEAD", default=None, help="Export diff bundle (optional git ref)")
    group.add_argument("--full", action="store_true", help="Export full repository text bundle")
    parser.add_argument("--include-tests", action="store_true", help="Include tests/ and subtrans/tests/")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Custom output path (never overwrites tracked files)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: Repository root not found: {root}", file=sys.stderr)
        return 1

    try:
        focus_tokens_cli: list[str] | None = None
        if args.focus:
            focus_tokens_cli = [t.strip().lower() for t in args.focus.split(",") if t.strip()]
            if "full" in focus_tokens_cli and len(focus_tokens_cli) > 1:
                raise ValueError("'full' cannot be combined with other --focus tokens")

        build_skeletons = focus_tokens_cli != ["full"]
        texts, skeletons, neighbors, tracked_set = _collect_repo_snapshot(
            root,
            include_tests=args.include_tests,
            build_skeletons=build_skeletons,
        )
        branch = (_git_output(root, "branch", "--show-current") or "detached").strip()
        head_short = (_git_output(root, "rev-parse", "--short", "HEAD") or "unknown").strip()
        commit_subject = (_git_output(root, "log", "-1", "--pretty=%s") or "(no commits)").strip()
        bundle_dir = root / BUNDLE_DIR_REL

        if args.sync_latest:
            changed, log_stat, diff_text = _compute_diff_context(root, None)
            written: list[str] = []
            diff_bundle, _, _ = _render_bundle(
                root, "diff", texts, skeletons, neighbors, branch, head_short, commit_subject,
                changed_paths=changed, log_stat=log_stat, diff_text=diff_text,
            )
            _atomic_write_bundle(root, bundle_dir / "latest_diff.txt", diff_bundle, tracked_set)
            written.append("latest_diff.txt")
            for filename, mode, tokens in SYNC_BUNDLES:
                content, _, _ = _render_bundle(
                    root, mode, texts, skeletons, neighbors, branch, head_short, commit_subject,
                    focus_tokens=tokens, log_stat=log_stat,
                )
                _atomic_write_bundle(root, bundle_dir / filename, content, tracked_set)
                written.append(filename)
            print(f"Synced {len(written)} review bundles in {bundle_dir} ({branch}@{head_short}): {', '.join(written)}")
            return 0

        if args.focus:
            tokens = focus_tokens_cli or []
            joined = "_".join(re.sub(r"[^a-z0-9]+", "_", t).strip("_") for t in tokens) or "custom"
            out_path = args.output or (bundle_dir / f"focus_{joined}.txt")
            _, log_stat, _ = _compute_diff_context(root, None)
            content, full_c, skel_c = _render_bundle(
                root, f"focus:{','.join(tokens)}", texts, skeletons, neighbors, branch, head_short, commit_subject,
                focus_tokens=tokens, log_stat=log_stat,
            )
        elif args.diff is not None:
            out_path = args.output or (bundle_dir / "latest_diff.txt")
            changed, log_stat, diff_text = _compute_diff_context(root, args.diff)
            content, full_c, skel_c = _render_bundle(
                root, "diff", texts, skeletons, neighbors, branch, head_short, commit_subject,
                changed_paths=changed, log_stat=log_stat, diff_text=diff_text,
            )
        else:
            out_path = args.output or (bundle_dir / "full_repo.txt")
            _, log_stat, _ = _compute_diff_context(root, None)
            content, full_c, skel_c = _render_bundle(
                root, "full", texts, skeletons, neighbors, branch, head_short, commit_subject,
                log_stat=log_stat,
            )

        _atomic_write_bundle(root, out_path, content, tracked_set)
        print(
            f"Wrote {out_path} ({branch}@{head_short}, full={full_c}, skeletons={skel_c}, est_tokens={len(content) // 4})"
        )
        return 0
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
