"""PreToolUse safety & context-firewall hook for e:\\ai-video-upscaler.

Enforces GEMINI.md, .agents/rules/*.md, and VALIDATION.md boundaries automatically:
1. Allows free local .env reading; prompts (force_ask) on .env writes/edits; hard-denies git add/commit of .env.
2. Enforces Context-Window & Full-Rewrite Firewall: blocks reading scratch/review_bundles/, blocks full-file
   write_file/write_to_file on existing > 250-line files, requires <= 280-line bounded ranges on read_file/view_file for
   > 350-line files, and blocks unbounded shell dumps on > 350-line files.
3. Hard-denies unapproved environment mutations (pip/uv/conda install/uninstall) protecting locked .venv ABI (and local untracked snapshot requirements.lock).
4. Prompts (force_ask) before deleting .temp_* checkpoints.
5. Hard-denies unapproved build_trt_engine.py or unflagged full-episode upscale/benchmark runs.
6. Prompts (force_ask) before invoking live paid Gemini API commands or concurrent GPU runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
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
UPSCALE_CLEAN_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:[\w./\\:-]*python(?:\.exe)?\s+)?[.\\/]*upscale(?:\.py|\.bat)\b.*(?:--clean-temp|--cleanup|--force)\b",
    re.IGNORECASE,
)
_ENV_PREFIX_RE = r"(?:[A-Za-z_]\w*=(?:\"[^\"]*\"|'[^']*'|[^\s;&|]*)\s+)*"
BUILD_TRT_EXEC_RE = re.compile(
    r"(?:^|[;&|]\s*)" + _ENV_PREFIX_RE + r"(?:[\w./\\:-]*python(?:\.exe)?\s+(?:-m\s+)?)?(?:[\w./\\:-]*[/\\])?(?:src[/\\.]build_trt_engine|build_trt_engine(?:\.py)?)\b",
    re.IGNORECASE,
)
SUBTRANS_EXEC_RE = re.compile(
    r"(?:^|[;&|]\s*)" + _ENV_PREFIX_RE + r"(?:[\w./\\:-]*python(?:\.exe)?\s+(?:-m\s+)?)?(?:[.\\/]*subtrans(?:\.py|\.bat|\.cli)|scripts[/\\]batch_translate_all\.py|benchmarks[/\\]subtrans[/\\](?!validate_pronouns_regex\.py)[\w.-]+\.py)\b",
    re.IGNORECASE,
)
LOCAL_SUBTRANS_SUBCMD_RE = re.compile(
    r"\bsubtrans(?:\.py|\.bat|\.cli)\s+(?:eval\b|compare\b|rag\s+stats\b)",
    re.IGNORECASE,
)
UPSCALE_SUBS_EXEC_RE = re.compile(
    r"(?:^|[;&|]\s*)" + _ENV_PREFIX_RE + r"(?:[\w./\\:-]*python(?:\.exe)?\s+)?[.\\/]*upscale(?:\.py|\.bat)\b.*(?:--subs-only|--subs-upgrade)\b",
    re.IGNORECASE,
)
UPSCALE_GPU_EXEC_RE = re.compile(
    r"(?:^|[;&|]\s*)" + _ENV_PREFIX_RE + r"(?:[\w./\\:-]*python(?:\.exe)?\s+)?(?:[.\\/]*upscale(?:_turbo)?(?:\.py|\.bat)|scripts[/\\](?:benchmark_pipeline|silicon_roofline_probe|upscale_x4_single)\.py)\b",
    re.IGNORECASE,
)
BARE_PYTEST_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:[\w./\\:-]*python(?:\.exe)?\s+-m\s+)?pytest(?:\.exe)?\b",
    re.IGNORECASE,
)

SAFE_BUILD_TRT_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:--help|-h|--dry-run)(?:\s|$)",
    re.IGNORECASE,
)
NON_GPU_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:--help|-h|--list-models|--dry-run|--subs-only)(?:\s|$)",
    re.IGNORECASE,
)
RUN_GPU_TESTS_RE = re.compile(
    r"(?:^|[\s;$])(?:env:)?RUN_GPU_TESTS\s*=\s*['\"]?1['\"]?(?:\s|$)",
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
SAFE_UPSCALE_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:--preview|--dry-run|--help|-h|--list-models|--clean-temp|--cleanup|--version|--subs-only|--subs-upgrade|--smoke)(?:\s|$)",
    re.IGNORECASE,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _is_secret_env_path(raw_path: str) -> bool:
    if not raw_path:
        return False
    name = Path(raw_path.strip().strip("'\"")).name.lower()
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
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        try:
            return shlex.split(seg, posix=False)
        except ValueError:
            return seg.split()


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
        while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
            i += 1
        if i >= len(tokens):
            continue
        if Path(tokens[i]).name.lower() not in {"git", "git.exe"}:
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


_SHELL_WRITE_CMDS = frozenset({"set-content", "sc", "add-content", "ac", "out-file", "tee"})


def _shell_write_targets_secret_env(cmd: str) -> bool:
    """Return True if an unquoted shell redirection or file-writing cmdlet targets secret .env."""
    for seg in _split_shell_segments(cmd):
        # Scan unquoted > / >> redirection targets inside the segment
        in_q: str | None = None
        i = 0
        n = len(seg)
        while i < n:
            ch = seg[i]
            if in_q is not None:
                if ch == "\\" and in_q == '"' and i + 1 < n:
                    i += 2
                    continue
                if ch == in_q:
                    in_q = None
                i += 1
                continue
            if ch in ("'", '"'):
                in_q = ch
                i += 1
                continue
            if ch == ">":
                j = i + 1 + (1 if i + 1 < n and seg[i + 1] == ">" else 0)
                while j < n and seg[j].isspace():
                    j += 1
                k = j
                while k < n and not seg[k].isspace() and seg[k] not in "&|;":
                    k += 1
                target = seg[j:k].strip("'\"")
                if _is_secret_env_path(target):
                    return True
                i = k
                continue
            i += 1

        tokens = _shlex_tokens(seg)
        if not tokens:
            continue
        cmd_name = Path(tokens[0]).name.lower()
        if cmd_name in _SHELL_WRITE_CMDS:
            for tok in tokens[1:]:
                if not tok.startswith("-") and _is_secret_env_path(tok):
                    return True
        elif cmd_name == "sed" and any(t == "-i" or t.startswith("-i") for t in tokens[1:]):
            for tok in tokens[1:]:
                if not tok.startswith("-") and _is_secret_env_path(tok):
                    return True
    return False


def _first_present(args: dict, keys: tuple[str, ...]) -> object:
    for key in keys:
        if args.get(key) is not None:
            return args.get(key)
    return None


def _is_safe_pytest_invocation(cmd: str, root: Path) -> bool:
    """Return True only when every pytest segment uses isolated snapshot flags or strictly safe test files."""
    saw_pytest = False
    for seg in _split_shell_segments(cmd):
        if not BARE_PYTEST_RE.search(seg):
            continue
        saw_pytest = True
        tokens = _shlex_tokens(seg)
        p_idx = -1
        for idx, tok in enumerate(tokens):
            base = Path(tok.strip("'\"")).name.lower()
            if base in {"pytest", "pytest.exe"}:
                p_idx = idx
                break
        if p_idx < 0:
            return False
        tail = tokens[p_idx + 1:]
        if any(t in {"--help", "-h", "--version"} for t in tail) and not any(
            not t.startswith("-") and (".py" in t or "/" in t or "\\" in t) for t in tail
        ):
            continue
        has_guard_plugin = any(
            t == "agent_offline_guard" and i > 0 and tail[i - 1] == "-p"
            for i, t in enumerate(tail)
        ) or ("-pagent_offline_guard" in tail)
        if has_guard_plugin and "CUDA_VISIBLE_DEVICES=-1" in cmd:
            continue
        targets: list[str] = []
        i = 0
        while i < len(tail):
            tok = tail[i]
            if tok == "--":
                targets.extend(tail[i + 1:])
                break
            if tok.startswith("-"):
                opt_name = tok.split("=", 1)[0]
                if opt_name in _PYTEST_VALUE_OPTS and "=" not in tok:
                    i += 2
                else:
                    i += 1
                continue
            targets.append(tok)
            i += 1
        if not targets:
            return False
        for raw_t in targets:
            file_part = raw_t.strip("'\"").split("::", 1)[0]
            _, rel_posix = _resolve_repo_path(root, file_part)
            norm = (rel_posix or file_part.replace("\\", "/").lstrip("./")).lower()
            if norm not in SAFE_DIRECT_PYTEST_FILES:
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

        # 1b. Local .env read is allowed; writes/edits prompt via force_ask
        if _is_secret_env_path(target):
            if tool_name in READ_TOOLS:
                return {"decision": "allow"}
            return {
                "decision": "force_ask",
                "reason": "Env Write Guard (GEMINI.md §4): Modifying .env secrets requires user confirmation.",
            }

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

    # 2. Shell command checks
    if tool_name == "run_command":
        cmd = str(args.get("CommandLine") or "")
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
                try:
                    tokens = shlex.split(tail_str, posix=False)
                except ValueError:
                    tokens = tail_str.split()
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

        # 2d. Prompt (force_ask) on shell redirection / writes targeting .env
        if _shell_write_targets_secret_env(cmd):
            return {
                "decision": "force_ask",
                "reason": "Env Write Guard (GEMINI.md §4): Shell write targeting .env secrets requires user confirmation.",
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

        if any(BUILD_TRT_EXEC_RE.search(seg) and not SAFE_BUILD_TRT_FLAGS_RE.search(seg) for seg in shell_segs):
            return {
                "decision": "deny",
                "reason": "Blocked (video-gpu.md): Running build_trt_engine.py requires explicit user approval.",
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

        # GPU exclusivity lock guard
        if (root / ".gpu_active.lock").exists():
            if any(
                (bool(UPSCALE_GPU_EXEC_RE.search(seg)) or bool(RUN_GPU_TESTS_RE.search(seg)))
                and not NON_GPU_FLAGS_RE.search(seg)
                for seg in shell_segs
            ):
                return {
                    "decision": "force_ask",
                    "reason": "GPU Exclusivity Guard (video-gpu.md): .gpu_active.lock exists; confirm no other GPU job is running.",
                }

        # Unflagged full-episode upscale / benchmark guard
        if any(UPSCALE_GPU_EXEC_RE.search(seg) and not SAFE_UPSCALE_FLAGS_RE.search(seg) for seg in shell_segs):
            return {
                "decision": "deny",
                "reason": "Blocked (video-gpu.md): Full-episode upscale/benchmark runs require --preview, --dry-run, or scripts/smoke_test_pipeline.py.",
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
