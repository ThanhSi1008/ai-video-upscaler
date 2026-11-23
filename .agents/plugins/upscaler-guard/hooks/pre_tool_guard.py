"""PreToolUse safety, context-firewall, and autonomous local-hardware hook for e:\\ai-video-upscaler.

Enforces GEMINI.md, .agents/rules/*.md, and VALIDATION.md boundaries automatically:
1. Allows free local .env reading and editing (free-tier keys); hard-denies git add/commit of .env.
2. Enforces Context-Window & Full-Rewrite Firewall: blocks reading scratch/review_bundles/, blocks full-file
   write_file/write_to_file on existing > 250-line files, requires <= 280-line bounded ranges on read_file/view_file for
   > 350-line files, and blocks unbounded shell dumps on > 350-line files.
3. Hard-denies unapproved environment mutations (pip/uv/conda install/uninstall) protecting locked .venv ABI (and local untracked snapshot requirements.lock).
4. Prompts (force_ask) before deleting .temp_* checkpoints.
5. Grants full autonomous authority for local CPU/RAM/NVMe, TensorRT 10 FP16 builds (src/build_trt_engine.py),
   PyTorch CUDA FP16, 10-bit HEVC NVENC full-episode upscales, and hardware benchmarks whenever .gpu_active.lock is idle.
6. Prompts (force_ask) before invoking live paid Gemini API commands or colliding with an active .gpu_active.lock.
"""

from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path
import re
import sys

SAFE_ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})
MAX_REWRITE_LINES = 250
MAX_VIEW_UNBOUNDED_LINES = 350
MAX_VIEW_SPAN = 280

READ_TOOLS = frozenset({"read_file", "view_file"})
FULL_WRITE_TOOLS = frozenset({"write_file", "write_to_file"})
EDIT_TOOLS = frozenset({"edit_file", "replace_file_content", "multi_replace_file_content"})
ALL_FILE_TOOLS = READ_TOOLS | FULL_WRITE_TOOLS | EDIT_TOOLS

BUNDLE_SHELL_READ_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:cat|type|get-content|gc|more|head|tail|sed)\b[^;&|\n]*scratch[/\\]review_bundles\b",
    re.IGNORECASE,
)
SHELL_DUMP_CMD_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:cat|type|get-content|gc)\s+(.+)$",
    re.IGNORECASE,
)
SHELL_BOUND_RE = re.compile(
    r"(?:-TotalCount\b|-Head\b|-Tail\b|-First\b|-Last\b|\bSelect-Object\b|\bhead\b|\btail\b|\bsed\s+-n\b)",
    re.IGNORECASE,
)
ENV_MUTATION_RE = re.compile(
    r"(?:\bpip(?:\d+(?:\.\d+)*)?(?:\.exe)?\s+(?:install|uninstall)\b|\buv\s+pip\b|\bconda\s+(?:install|remove|update)\b)",
    re.IGNORECASE,
)
TEMP_DELETE_RE = re.compile(
    r"\b(?:rm|del|erase|rmdir|rd|remove-item|ri|shutil\.rmtree)\b.*\.temp_",
    re.IGNORECASE,
)
_ENV_PREFIX_RE = r"(?:[A-Za-z_]\w*=(?:\"[^\"]*\"|'[^']*'|[^\s;&|]*)\s+)*"
_CMD_START_RE = r"^\s*(?:&\s*)?" + _ENV_PREFIX_RE
_PY_EXE_RE = r"(?:\"[^\"]*python[\w.-]*\"|'[^']*python[\w.-]*'|[\w./\\:-]*python[\w.-]*)"
_PY_RUNNER_RE = rf"(?:{_PY_EXE_RE}\s+(?:-[uBEOs]+\s+)*(?:-m\s+)?)?"
_PATH_PREFIX_RE = r"(?:[\w./\\:-]*[/\\])?"
_UPSCALE_INVOC_RE = (
    _CMD_START_RE
    + r"(?:"
    + _PY_RUNNER_RE
    + rf"['\"]?{_PATH_PREFIX_RE}upscale(?:_turbo)?(?:\.py|\.bat)['\"]?"
    + r"|"
    + rf"{_PY_EXE_RE}\s+(?:-[uBEOs]+\s+)*-m\s+['\"]?upscale['\"]?"
    + r")(?:\s|$)"
)
UPSCALE_CLEAN_RE = re.compile(
    _UPSCALE_INVOC_RE + r".*(?:--clean-temp|--cleanup|--force)\b",
    re.IGNORECASE,
)
BUILD_TRT_EXEC_RE = re.compile(
    _CMD_START_RE
    + _PY_RUNNER_RE
    + rf"['\"]?(?:{_PATH_PREFIX_RE}build_trt_engine(?:\.py)?|src\.build_trt_engine)['\"]?(?:\s|$)",
    re.IGNORECASE,
)
SUBTRANS_EXEC_RE = re.compile(
    _CMD_START_RE
    + _PY_RUNNER_RE
    + rf"['\"]?(?:{_PATH_PREFIX_RE}subtrans(?:\.py|\.bat|\.exe|\.cli)?|subtrans\.cli|{_PATH_PREFIX_RE}scripts[/\\]batch_translate_all\.py|scripts\.batch_translate_all|{_PATH_PREFIX_RE}benchmarks[/\\]subtrans[/\\](?!validate_pronouns_regex\.py)[\w.-]+\.py|benchmarks\.subtrans\.(?!validate_pronouns_regex\b)[\w.]+)['\"]?(?:\s|$)",
    re.IGNORECASE,
)
LOCAL_SUBTRANS_SUBCMD_RE = re.compile(
    r"\bsubtrans(?:\.py|\.bat|\.exe|\.cli)?['\"]?\s+(?:eval\b|compare\b|rag\s+stats\b)",
    re.IGNORECASE,
)
UPSCALE_SUBS_EXEC_RE = re.compile(
    _UPSCALE_INVOC_RE
    + r".*(?:--subs-only|--subs-upgrade|--subs-pending|--subs-after-upscale|--async-subs)\b",
    re.IGNORECASE,
)
UPSCALE_GPU_EXEC_RE = re.compile(
    r"(?:"
    + _UPSCALE_INVOC_RE
    + r"|"
    + _CMD_START_RE
    + _PY_RUNNER_RE
    + rf"['\"]?(?:{_PATH_PREFIX_RE}scripts[/\\](?:benchmark_pipeline|silicon_roofline_probe|upscale_x4_single)\.py|scripts\.(?:benchmark_pipeline|silicon_roofline_probe|upscale_x4_single)|{_PATH_PREFIX_RE}benchmarks[/\\](?:upscale|research)[/\\][\w.-]+\.py|benchmarks\.(?:upscale|research)\.[\w.]+|{_PATH_PREFIX_RE}tests[/\\]test_gpu_throttling_overlap\.py|tests\.test_gpu_throttling_overlap)['\"]?(?:\s|$)"
    + r")",
    re.IGNORECASE,
)
BARE_PYTEST_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:&\s*)?"
    + _ENV_PREFIX_RE
    + rf"(?:{_PY_EXE_RE}\s+(?:-[uBEOs]+\s+)*-m\s+)?['\"]?[\w./\\:-]*pytest(?:\.exe)?['\"]?(?:\s|$)",
    re.IGNORECASE,
)

SAFE_BUILD_TRT_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:--help|-h|--dry-run)(?:\s|$)",
    re.IGNORECASE,
)
NON_GPU_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:--help|-h|--list-models|--dry-run|--subs-only|--night-report|--version|--engine(?:=|\s+)['\"]?cpu['\"]?)(?:\s|$)",
    re.IGNORECASE,
)
RUN_GPU_TESTS_RE = re.compile(
    r"(?:^|[\s;$])(?:env:)?RUN_GPU_TESTS\s*=\s*['\"]?1['\"]?(?:\s|$)",
    re.IGNORECASE,
)
CUDA_HIDDEN_RE = re.compile(
    r"(?:^|[\s;$])(?:env:)?CUDA_VISIBLE_DEVICES\s*=\s*(?:'-1'|\"-1\"|-1|''|\"\")(?:\s|;|$)",
    re.IGNORECASE,
)
SAFE_SUBTRANS_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:--help|-h|--dry-run|--version|--mock|--test)(?:\s|$)",
    re.IGNORECASE,
)
SAFE_DIRECT_PYTEST_FILES = frozenset({
    "tests/test_agent_context.py",
    "tests/test_agent_setup.py",
    "tests/test_verify_offline.py",
    "tests/test_smoke_pipeline.py",
    "tests/test_web_gui.py",
    "tests/test_architecture_boundaries.py",
    "tests/test_export_review_bundle.py",
})
_PYTEST_VALUE_OPTS = frozenset({
    "-k", "-m", "-o", "-c", "-p", "-r", "--maxfail", "--basetemp", "--rootdir",
    "--confcutdir", "--import-mode", "--tb", "--capture", "--junitxml", "--durations",
})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _is_secret_env_path(raw_path: str) -> bool:
    if not raw_path:
        return False
    name = Path(raw_path.strip().strip("'\"").replace("\\", "/")).name.lower()
    if name in SAFE_ENV_TEMPLATES:
        return False
    return name == ".env" or name.startswith(".env.")


def _resolve_repo_path(root: Path, raw_path: str) -> tuple[Path | None, str]:
    cleaned = str(raw_path or "").strip()
    if not cleaned:
        return None, ""
    cand = Path(cleaned)
    if not cand.is_absolute():
        cand = root / cand
    try:
        resolved = cand.resolve()
        root_res = root.resolve()
    except OSError:
        return None, ""
    if not resolved.is_relative_to(root_res):
        return None, ""
    return resolved, resolved.relative_to(root_res).as_posix()


def _is_review_bundle_path(rel_posix: str, raw_path: str) -> bool:
    norm_raw = str(raw_path or "").replace("\\", "/").lower()
    norm_rel = rel_posix.lower()
    return (
        norm_rel == "scratch/review_bundles"
        or norm_rel.startswith("scratch/review_bundles/")
        or "/scratch/review_bundles/" in norm_raw
        or norm_raw.startswith("scratch/review_bundles/")
    )


def _count_lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def _split_shell_segments(cmd: str) -> list[str]:
    """Split a command string on unquoted ;, &&, ||, |, or newlines."""
    segments: list[str] = []
    cur: list[str] = []
    in_quote: str | None = None
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if in_quote is not None:
            cur.append(ch)
            if ch == "\\" and in_quote == '"' and i + 1 < n:
                cur.append(cmd[i + 1])
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = ch
            cur.append(ch)
            i += 1
            continue
        if ch in (";", "\n", "|"):
            seg = "".join(cur).strip()
            if seg:
                segments.append(seg)
            cur = []
            if ch == "|" and i + 1 < n and cmd[i + 1] == "|":
                i += 2
            else:
                i += 1
            continue
        if ch == "&" and i + 1 < n and cmd[i + 1] == "&":
            seg = "".join(cur).strip()
            if seg:
                segments.append(seg)
            cur = []
            i += 2
            continue
        cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        segments.append(tail)
    return segments


def _split_command_chains(cmd: str) -> list[str]:
    """Split a command string on unquoted ;, &&, ||, or newlines while keeping single | pipes intact."""
    chains: list[str] = []
    cur: list[str] = []
    in_quote: str | None = None
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if in_quote is not None:
            cur.append(ch)
            if ch == "\\" and in_quote == '"' and i + 1 < n:
                cur.append(cmd[i + 1])
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = ch
            cur.append(ch)
            i += 1
            continue
        if ch in (";", "\n"):
            seg = "".join(cur).strip()
            if seg:
                chains.append(seg)
            cur = []
            i += 1
            continue
        if ch == "|" and i + 1 < n and cmd[i + 1] == "|":
            seg = "".join(cur).strip()
            if seg:
                chains.append(seg)
            cur = []
            i += 2
            continue
        if ch == "&" and i + 1 < n and cmd[i + 1] == "&":
            seg = "".join(cur).strip()
            if seg:
                chains.append(seg)
            cur = []
            i += 2
            continue
        cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        chains.append(tail)
    return chains


def _shlex_tokens(seg: str) -> list[str]:
    """Tokenize a shell segment preserving Windows backslashes while stripping quote delimiters."""
    tokens: list[str] = []
    cur: list[str] = []
    has_token = False
    in_quote: str | None = None
    i = 0
    n = len(seg)
    while i < n:
        ch = seg[i]
        if in_quote is not None:
            if ch == "\\" and in_quote == '"' and i + 1 < n and seg[i + 1] == '"':
                cur.append('"')
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
            else:
                cur.append(ch)
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = ch
            has_token = True
            i += 1
            continue
        if ch.isspace():
            if has_token:
                tokens.append("".join(cur))
                cur = []
                has_token = False
            i += 1
            continue
        cur.append(ch)
        has_token = True
        i += 1
    if has_token:
        tokens.append("".join(cur))
    return tokens


_GIT_GLOBAL_VALUE_OPTS = frozenset({"-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env"})
_GIT_COMMIT_VALUE_OPTS = frozenset({
    "-m", "--message", "-f", "--file", "-c", "--reedit-message", "--reuse-message",
    "--author", "--date", "-t", "--template", "-s", "--gpg-sign", "--cleanup",
    "--fixup", "--squash", "-u", "--untracked-files", "--trailer", "--pathspec-from-file",
})


def _git_cmd_targets_secret_env(cmd: str) -> bool:
    """Return True if any git add/commit segment passes a secret .env path argument."""
    for seg in _split_shell_segments(cmd):
        tokens = _shlex_tokens(seg)
        i = 0
        while i < len(tokens) and (tokens[i] == "&" or ("=" in tokens[i] and not tokens[i].startswith("-"))):
            i += 1
        if i >= len(tokens):
            continue
        if Path(tokens[i].replace("\\", "/")).name.lower() not in {"git", "git.exe"}:
            continue
        i += 1
        while i < len(tokens) and tokens[i].startswith("-"):
            opt = tokens[i]
            if opt.lower() in _GIT_GLOBAL_VALUE_OPTS or opt in {"-C", "-c"}:
                i += 2
            else:
                i += 1
        if i >= len(tokens):
            continue
        subcmd = tokens[i].lower()
        if subcmd not in {"add", "commit"}:
            continue
        i += 1
        after_ddash = False
        while i < len(tokens):
            tok = tokens[i]
            if not after_ddash and tok == "--":
                after_ddash = True
                i += 1
                continue
            if not after_ddash and tok.startswith("-"):
                low_opt = tok.lower()
                if subcmd == "commit" and (low_opt in _GIT_COMMIT_VALUE_OPTS or tok in {"-F", "-C"}):
                    i += 2
                    continue
                i += 1
                continue
            if _is_secret_env_path(tok):
                return True
            i += 1
    return False


def _first_present(args: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if args.get(key) is not None:
            return args.get(key)
    return None


def _powerset(items: list[str]):
    for r in range(len(items) + 1):
        yield from itertools.combinations(items, r)


def _eval_marker_ast(node: ast.AST, active: frozenset[str]) -> bool:
    if isinstance(node, ast.Expression):
        return _eval_marker_ast(node.body, active)
    if isinstance(node, ast.Name):
        low = node.id.lower()
        if low in {"true", "1"}:
            return True
        if low in {"false", "0"}:
            return False
        return low in active
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return bool(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_marker_ast(node.operand, active)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval_marker_ast(v, active) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_eval_marker_ast(v, active) for v in node.values)
    raise ValueError("Unsupported marker AST node")


def _is_strict_gpu_or_smoke_marker(expr: str) -> tuple[bool, bool]:
    """Return (is_safe_marker, may_use_gpu) for a pytest -m boolean expression via AST truth-table analysis."""
    cleaned = expr.strip().strip("'\"")
    if not cleaned:
        return False, False
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return False, False
    names = {n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)} - {"true", "false"}
    if not names or len(names) > 8:
        return False, False
    other_vars = sorted(names - {"gpu", "smoke"})
    all_vars = sorted(names)
    try:
        leaks_non_gpu_smoke = any(
            _eval_marker_ast(tree, frozenset(sub)) for sub in _powerset(other_vars)
        )
        satisfiable = any(
            _eval_marker_ast(tree, frozenset(sub)) for sub in _powerset(all_vars)
        )
        non_gpu_vars = sorted(names - {"gpu"})
        may_use_gpu = ("gpu" in names) and any(
            _eval_marker_ast(tree, frozenset(sub) | {"gpu"}) for sub in _powerset(non_gpu_vars)
        )
    except ValueError:
        return False, False
    return (satisfiable and not leaks_non_gpu_smoke), may_use_gpu


def _expr_has_gpu_mark(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and sub.attr == "gpu"
            and isinstance(sub.value, ast.Attribute)
            and sub.value.attr == "mark"
        ):
            return True
    return False


def _inspect_test_file_gpu_marks(root: Path, file_part: str) -> tuple[bool, bool]:
    """Return (is_dedicated_gpu_module, has_any_gpu_mark) for a test path via AST inspection."""
    resolved, rel_posix = _resolve_repo_path(root, file_part)
    norm = (rel_posix or file_part.replace("\\", "/").lstrip("./")).lower()
    if norm in SAFE_DIRECT_PYTEST_FILES:
        return False, False
    if norm == "tests/test_gpu_throttling_overlap.py" and (resolved is None or not resolved.exists()):
        return True, True
    if resolved is None or not resolved.is_file() or resolved.suffix.lower() != ".py":
        return False, False
    if not norm.startswith("tests/"):
        return False, False
    try:
        tree = ast.parse(resolved.read_text(encoding="utf-8-sig", errors="ignore"), filename=str(resolved))
    except (OSError, SyntaxError):
        return False, False

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets
        ):
            if _expr_has_gpu_mark(stmt.value):
                return True, True
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "pytestmark"
            and stmt.value is not None
            and _expr_has_gpu_mark(stmt.value)
        ):
            return True, True

    total_tests = 0
    gpu_tests = 0
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name.startswith("test_"):
            total_tests += 1
            if any(_expr_has_gpu_mark(d) for d in stmt.decorator_list):
                gpu_tests += 1
        elif isinstance(stmt, ast.ClassDef):
            cls_gpu = any(_expr_has_gpu_mark(d) for d in stmt.decorator_list) or any(
                (
                    isinstance(cs, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in cs.targets)
                    and _expr_has_gpu_mark(cs.value)
                )
                for cs in stmt.body
            )
            for cs in stmt.body:
                if isinstance(cs, (ast.FunctionDef, ast.AsyncFunctionDef)) and cs.name.startswith("test_"):
                    total_tests += 1
                    if cls_gpu or any(_expr_has_gpu_mark(d) for d in cs.decorator_list):
                        gpu_tests += 1

    has_any = gpu_tests > 0
    return (total_tests > 0 and gpu_tests == total_tests), has_any


def _extract_pytest_tail_info(seg: str) -> tuple[bool, list[str], list[str], list[str]]:
    """Return (found_pytest, tail_tokens, marker_exprs, target_args) for a shell segment."""
    tokens = _shlex_tokens(seg)
    p_idx = -1
    for idx, tok in enumerate(tokens):
        if tok == "&" or tok.startswith("-"):
            continue
        base = Path(tok.replace("\\", "/")).name.lower()
        if base in {"pytest", "pytest.exe"}:
            p_idx = idx
            break
    if p_idx < 0:
        return False, [], [], []
    tail = tokens[p_idx + 1 :]
    markers: list[str] = []
    targets: list[str] = []
    i = 0
    while i < len(tail):
        tok = tail[i]
        if tok == "--":
            targets.extend(tail[i + 1 :])
            break
        if tok in {"-m", "--markexpr"} and i + 1 < len(tail):
            markers.append(tail[i + 1])
            i += 2
            continue
        if tok.startswith("--markexpr="):
            markers.append(tok.split("=", 1)[1])
            i += 1
            continue
        if tok.startswith("-m=") or (
            tok.startswith("-m") and len(tok) > 2 and not tok.startswith("--")
        ):
            markers.append(tok[2:].lstrip("="))
            i += 1
            continue
        if tok.startswith("-"):
            opt_name = tok.split("=", 1)[0]
            if opt_name in _PYTEST_VALUE_OPTS and "=" not in tok:
                i += 2
            else:
                i += 1
            continue
        targets.append(tok)
        i += 1
    return True, tail, markers, targets


def _segment_uses_gpu_pytest(seg: str, cmd: str, root: Path) -> bool:
    """Return True when a pytest segment may select @pytest.mark.gpu tests without CUDA hidden."""
    if not BARE_PYTEST_RE.search(seg) or CUDA_HIDDEN_RE.search(cmd):
        return False
    found, tail, markers, targets = _extract_pytest_tail_info(seg)
    if not found:
        return False
    if any(t in {"--help", "-h", "--version", "--markers"} for t in tail) and not targets:
        return False
    if markers:
        return _is_strict_gpu_or_smoke_marker(markers[-1])[1]
    return any(_inspect_test_file_gpu_marks(root, raw_t.strip("'\"").split("::", 1)[0])[1] for raw_t in targets)


def _is_safe_pytest_invocation(cmd: str, root: Path) -> bool:
    """Return True when every pytest segment uses isolated snapshot flags, safe allowlist files, or GPU/smoke benchmark scopes."""
    saw_pytest = False
    segments = _split_shell_segments(cmd)
    has_gpu_env = any(RUN_GPU_TESTS_RE.search(s) for s in segments)
    cuda_hidden = bool(CUDA_HIDDEN_RE.search(cmd))
    for seg in segments:
        if not BARE_PYTEST_RE.search(seg):
            continue
        saw_pytest = True
        found, tail, markers, targets = _extract_pytest_tail_info(seg)
        if not found:
            return False
        if any(t in {"--help", "-h", "--version", "--markers"} for t in tail) and not any(
            not t.startswith("-") and (".py" in t or "/" in t or "\\" in t) for t in tail
        ):
            continue
        has_guard_plugin = any(
            t == "agent_offline_guard" and i > 0 and tail[i - 1] == "-p"
            for i, t in enumerate(tail)
        ) or ("-pagent_offline_guard" in tail)
        if has_guard_plugin and cuda_hidden:
            continue
        has_strict_safe_marker = bool(markers) and _is_strict_gpu_or_smoke_marker(markers[-1])[0]
        if has_strict_safe_marker and not targets:
            continue
        if not targets:
            return False
        for raw_t in targets:
            file_part = raw_t.strip("'\"").split("::", 1)[0]
            resolved, rel_posix = _resolve_repo_path(root, file_part)
            norm = (rel_posix or file_part.replace("\\", "/").lstrip("./")).rstrip("/").lower()
            if norm in SAFE_DIRECT_PYTEST_FILES:
                continue
            has_parent_traversal = ".." in Path(file_part.replace("\\", "/")).parts
            in_repo_or_rel = (resolved is not None or not Path(file_part).is_absolute()) and not has_parent_traversal
            if has_strict_safe_marker and in_repo_or_rel and (norm == "tests" or norm.startswith("tests/")):
                continue
            if has_gpu_env and _inspect_test_file_gpu_marks(root, file_part)[0]:
                continue
            return False
    return saw_pytest


def _coerce_optional_int(val: object) -> tuple[bool, int | None]:
    if val is None:
        return True, None
    if isinstance(val, bool):
        return False, None
    if isinstance(val, int):
        return True, val
    if isinstance(val, str) and val.strip():
        try:
            return True, int(val.strip())
        except ValueError:
            return False, None
    return False, None


def evaluate_tool_call(payload: dict, repo_root: Path | None = None) -> dict:
    root = (repo_root or _repo_root()).resolve()
    tool_call = payload.get("toolCall") or {}
    tool_name = (tool_call.get("name") or "").lower()
    args = tool_call.get("args") or {}

    # 1. File read / write / edit checks
    if tool_name in ALL_FILE_TOOLS:
        target = str(
            args.get("Path")
            or args.get("AbsolutePath")
            or args.get("TargetFile")
            or args.get("FilePath")
            or ""
        )
        resolved, rel_posix = _resolve_repo_path(root, target)

        # 1a. Block reading generated review bundles into agent context
        if tool_name in READ_TOOLS and _is_review_bundle_path(rel_posix, target):
            return {
                "decision": "deny",
                "reason": "Blocked (Context Firewall): Do not read scratch/review_bundles/ into context; use scripts/agent_context.py --pack / --slice instead.",
            }

        # 1b. Local .env read and edit are allowed freely; git add/commit of .env remains hard-denied
        if _is_secret_env_path(target):
            return {"decision": "allow"}

        # 1c. Full-file overwrite firewall (> 250 lines)
        if (
            tool_name in FULL_WRITE_TOOLS
            and resolved is not None
            and resolved.is_file()
            and not rel_posix.lower().startswith("scratch/")
        ):
            total_lines = _count_lines(resolved)
            if total_lines > MAX_REWRITE_LINES:
                return {
                    "decision": "deny",
                    "reason": (
                        f"Blocked (Rewrite Firewall): Refusing full-file {tool_name} on existing {total_lines}-line file "
                        f"'{rel_posix}' (> {MAX_REWRITE_LINES} lines). Use surgical replace_file_content or multi_replace_file_content."
                    ),
                }

        # 1d. Unbounded read_file / view_file firewall (> 350 lines requires <= 280-line bounded range)
        if tool_name in READ_TOOLS and resolved is not None and resolved.is_file():
            total_lines = _count_lines(resolved)
            if total_lines > MAX_VIEW_UNBOUNDED_LINES:
                raw_start = _first_present(args, ("StartLine", "start_line", "startLine"))
                raw_end = _first_present(args, ("EndLine", "end_line", "endLine"))
                raw_offset = _first_present(args, ("offset", "Offset"))
                raw_limit = _first_present(args, ("limit", "Limit"))
                suffix = resolved.suffix.lower()
                hint = (
                    f"Run scripts/agent_context.py --pack / --symbols / --slice first for .py/.js/.css, or pass bounded StartLine..EndLine (<= {MAX_VIEW_SPAN} lines) on {tool_name}."
                    if suffix in {".py", ".js", ".css"}
                    else f"Pass bounded StartLine..EndLine (<= {MAX_VIEW_SPAN} lines) directly on {tool_name}."
                )
                eff_start: int | None = None
                eff_end: int | None = None
                if raw_start is not None or raw_end is not None:
                    ok_s, s_val = _coerce_optional_int(raw_start)
                    ok_e, e_val = _coerce_optional_int(raw_end)
                    if ok_s and ok_e and not (s_val is None and e_val is None):
                        eff_start = max(1, s_val if s_val is not None else 1)
                        eff_end = min(total_lines, e_val if e_val is not None else total_lines)
                elif raw_offset is not None or raw_limit is not None:
                    ok_o, o_val = _coerce_optional_int(raw_offset)
                    ok_l, l_val = _coerce_optional_int(raw_limit)
                    if ok_o and ok_l and not (o_val is None and l_val is None):
                        if l_val is not None and l_val <= 0:
                            eff_start, eff_end = 2, 1
                        else:
                            eff_start = max(1, o_val if (o_val is not None and o_val >= 1) else 1)
                            eff_end = min(total_lines, (eff_start + l_val - 1) if l_val is not None else total_lines)

                if eff_start is None or eff_end is None:
                    return {
                        "decision": "deny",
                        "reason": (
                            f"Blocked (Context Firewall): '{rel_posix}' has {total_lines} lines (> {MAX_VIEW_UNBOUNDED_LINES}). "
                            f"Missing bounded line range. {hint}"
                        ),
                    }
                if eff_end < eff_start or (eff_end - eff_start + 1) > MAX_VIEW_SPAN:
                    span = max(0, eff_end - eff_start + 1)
                    return {
                        "decision": "deny",
                        "reason": (
                            f"Blocked (Context Firewall): '{rel_posix}' has {total_lines} lines (> {MAX_VIEW_UNBOUNDED_LINES}); "
                            f"effective span L{eff_start}-L{eff_end} ({span} lines) exceeds {MAX_VIEW_SPAN} lines or is empty. {hint}"
                        ),
                    }

        return {"decision": "allow"}

    # 2. Shell command checks (including unwrapped PowerShell/cmd wrappers)
    if tool_name == "run_command":
        cmd = str(args.get("CommandLine") or "")
        best: dict = {"decision": "allow"}
        for candidate_cmd in _expand_wrapped_commands(cmd):
            res = _evaluate_single_command(candidate_cmd, root)
            if _DECISION_PRIORITY.get(res.get("decision", "allow"), 0) > _DECISION_PRIORITY.get(
                best.get("decision", "allow"), 0
            ):
                best = res
                if best.get("decision") == "deny":
                    break
        return best

    return {"decision": "allow"}


_START_PROC_SWITCH_PARAMS = frozenset({
    "-nonewwindow", "-wait", "-passthru", "-loaduserprofile", "-usenewenvironment",
})
_START_PROC_VALUE_PARAMS = frozenset({
    "-workingdirectory", "-windowstyle", "-verb", "-credential",
    "-redirectstandardoutput", "-redirectstandarderror", "-redirectstandardinput",
})
_DECISION_PRIORITY = {"allow": 0, "ask": 1, "force_ask": 2, "deny": 3}


def _unwrap_single_level_wrappers(cmd: str) -> list[str]:
    """Extract inner command strings from PowerShell/cmd shell wrappers in cmd."""
    inner: list[str] = []
    for seg in _split_shell_segments(cmd):
        tokens = _shlex_tokens(seg)
        idx = 0
        while idx < len(tokens) and (
            tokens[idx] == "&" or ("=" in tokens[idx] and not tokens[idx].startswith("-"))
        ):
            idx += 1
        if idx >= len(tokens):
            continue
        head = Path(tokens[idx].replace("\\", "/")).name.lower()
        if head in {"invoke-expression", "iex"}:
            j = idx + 1
            while j < len(tokens):
                low = tokens[j].lower()
                if low in {"-command", "-c"} and j + 1 < len(tokens):
                    inner.append(tokens[j + 1])
                    j += 2
                    continue
                if not tokens[j].startswith("-"):
                    inner.append(tokens[j])
                j += 1
        elif head in {"cmd", "cmd.exe"}:
            for j in range(idx + 1, len(tokens)):
                if tokens[j].lower() in {"/c", "/k"} and j + 1 < len(tokens):
                    inner.append(" ".join(tokens[j + 1 :]))
                    break
        elif head in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
            for j in range(idx + 1, len(tokens)):
                low = tokens[j].lower()
                if low in {"-file", "-f"}:
                    break
                if low in {"-command", "-c"} and j + 1 < len(tokens):
                    inner.append(" ".join(tokens[j + 1 :]))
                    break
        elif head in {"start-process", "saps"}:
            file_path: str | None = None
            arg_parts: list[str] = []
            in_arg_list = False
            j = idx + 1
            while j < len(tokens):
                tok = tokens[j]
                low = tok.lower()
                if low in {"-filepath", "-pspath"} and j + 1 < len(tokens):
                    file_path = tokens[j + 1]
                    in_arg_list = False
                    j += 2
                    continue
                if low in {"-argumentlist", "-args"}:
                    in_arg_list = True
                    j += 1
                    continue
                if low in _START_PROC_SWITCH_PARAMS:
                    in_arg_list = False
                    j += 1
                    continue
                if low in _START_PROC_VALUE_PARAMS:
                    in_arg_list = False
                    j += 2
                    continue
                if in_arg_list:
                    if "," in tok:
                        arg_parts.extend(p.strip() for p in tok.split(",") if p.strip())
                    else:
                        arg_parts.append(tok)
                elif file_path is None and not tok.startswith("-"):
                    file_path = tok
                else:
                    arg_parts.append(tok)
                j += 1
            if file_path:
                exe_fmt = f'"{file_path}"' if " " in file_path else file_path
                rebuilt = f"{exe_fmt} {' '.join(arg_parts)}".strip()
                inner.append(rebuilt)
    return inner


def _expand_wrapped_commands(cmd: str, max_depth: int = 2) -> list[str]:
    """Return cmd followed by any unwrapped shell-wrapper inner commands up to max_depth."""
    seen: list[str] = [cmd]
    frontier: list[str] = [cmd]
    for _ in range(max_depth):
        next_frontier: list[str] = []
        for item in frontier:
            for sub in _unwrap_single_level_wrappers(item):
                if sub and sub not in seen:
                    seen.append(sub)
                    next_frontier.append(sub)
        if not next_frontier:
            break
        frontier = next_frontier
    return seen


def _evaluate_single_command(cmd: str, root: Path) -> dict:
    shell_segs = _split_shell_segments(cmd)
    chain_segs = _split_command_chains(cmd)

    # 2a. Block shell reads targeting scratch/review_bundles/ (allow export_review_bundle.py execution)
    if any(BUNDLE_SHELL_READ_RE.search(seg) and "export_review_bundle.py" not in seg for seg in shell_segs):
        return {
            "decision": "deny",
            "reason": "Blocked (Context Firewall): Do not dump scratch/review_bundles/ via shell; use scripts/agent_context.py --pack / --slice.",
        }

    # 2b. Block unbounded shell dumps on existing repo files > 350 lines
    for chain_seg in chain_segs:
        pipe_parts = _split_shell_segments(chain_seg)
        for idx, part in enumerate(pipe_parts):
            if any(SHELL_BOUND_RE.search(p) for p in pipe_parts[idx:]):
                continue
            m_dump = SHELL_DUMP_CMD_RE.match(part.strip())
            if not m_dump:
                continue
            tail_str = m_dump.group(1)
            tokens = _shlex_tokens(tail_str)
            for tok in tokens:
                cleaned = tok.strip("'\"")
                if not cleaned or cleaned.startswith("-"):
                    continue
                resolved, rel_posix = _resolve_repo_path(root, cleaned)
                if resolved is not None and resolved.is_file():
                    total_lines = _count_lines(resolved)
                    if total_lines > MAX_VIEW_UNBOUNDED_LINES:
                        return {
                            "decision": "deny",
                            "reason": (
                                f"Blocked (Context Firewall): Unbounded shell dump on {total_lines}-line file '{rel_posix}' "
                                f"(> {MAX_VIEW_UNBOUNDED_LINES}). Use scripts/agent_context.py --slice / --pack or bounded view_file."
                            ),
                        }

    # 2c. Hard-deny git add / git commit targeting .env or subtrans/.env
    if _git_cmd_targets_secret_env(cmd):
        return {
            "decision": "deny",
            "reason": "Blocked (GEMINI.md §4): Never stage or commit .env secret files with git add/commit.",
        }

    if ENV_MUTATION_RE.search(cmd):
        return {
            "decision": "deny",
            "reason": "Blocked (GEMINI.md §2): Environment mutations (pip/uv/conda install) are prohibited to protect locked .venv ABI (local untracked snapshot requirements.lock).",
        }

    if any(TEMP_DELETE_RE.search(seg) or UPSCALE_CLEAN_RE.search(seg) for seg in shell_segs):
        return {
            "decision": "force_ask",
            "reason": "Checkpoint Guard (GEMINI.md §4): Deleting or overwriting .temp_* checkpoints requires confirmation.",
        }

    # Live Gemini API quota guard (allow purely local subcommands eval / compare / rag stats)
    if any(
        (SUBTRANS_EXEC_RE.search(seg) or UPSCALE_SUBS_EXEC_RE.search(seg))
        and not (bool(LOCAL_SUBTRANS_SUBCMD_RE.search(seg)) and not UPSCALE_SUBS_EXEC_RE.search(seg))
        and not SAFE_SUBTRANS_FLAGS_RE.search(seg)
        for seg in shell_segs
    ):
        return {
            "decision": "force_ask",
            "reason": "Gemini Quota Guard (GEMINI.md §4 & subtrans.md): Command may consume live Gemini API RPD; confirm intent.",
        }

    # GPU exclusivity lock guard (single RTX 3060 12GB owner; autonomous when .gpu_active.lock is absent)
    if (root / ".gpu_active.lock").exists():
        if any(
            (
                (bool(BUILD_TRT_EXEC_RE.search(seg)) and not SAFE_BUILD_TRT_FLAGS_RE.search(seg))
                or (bool(UPSCALE_GPU_EXEC_RE.search(seg)) and not NON_GPU_FLAGS_RE.search(seg))
                or bool(RUN_GPU_TESTS_RE.search(seg))
                or _segment_uses_gpu_pytest(seg, cmd, root)
            )
            for seg in shell_segs
        ):
            return {
                "decision": "force_ask",
                "reason": "GPU Exclusivity Guard (video-gpu.md): .gpu_active.lock exists; confirm no other GPU job is running.",
            }

    # Bare pytest on production repo guard
    if BARE_PYTEST_RE.search(cmd) and not _is_safe_pytest_invocation(cmd, root):
        return {
            "decision": "ask",
            "reason": "Test Isolation Guard (VALIDATION.md): Run .\\.venv\\Scripts\\python.exe scripts/verify_offline.py instead of bare pytest.",
        }

    return {"decision": "allow"}


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        decision = evaluate_tool_call(payload)
        sys.stdout.write(json.dumps(decision, ensure_ascii=False))
    except Exception:
        sys.stdout.write(json.dumps({"decision": "allow"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
