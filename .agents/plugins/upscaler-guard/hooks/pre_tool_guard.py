"""PreToolUse safety hook for e:\\ai-video-upscaler.

Enforces AGENTS.md & VALIDATION.md boundaries automatically:
1. Hard-denies reading/writing/printing .env or GEMINI_API_KEY values.
2. Requires explicit confirmation (force_ask) before deleting .temp_* checkpoints.
3. Requires confirmation (force_ask) before invoking live Gemini API commands (subtrans check/models/run without --dry-run).
4. Warns/asks before running bare pytest on the production repo or starting GPU inference when .gpu_active.lock is present.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

SAFE_ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})

ENV_CAT_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:cat|type|get-content|gc|more|head|tail)\s+['\"]?(?:[\w./\\:-]+[/\\])?(\.env(?:\.[\w-]+)?)(?:['\"]|\s|$)",
    re.IGNORECASE,
)
KEY_PRINT_RE = re.compile(
    r"\b(?:echo|write-output|write-host|printf|printenv|dir\s+env:|get-childitem\s+env:)\b.*GEMINI_API_KEY",
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
SUBTRANS_EXEC_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:[\w./\\:-]*python(?:\.exe)?\s+(?:-m\s+)?)?(?:[.\\/]*subtrans(?:\.py|\.bat|\.cli)|scripts[/\\]batch_translate_all\.py)\b",
    re.IGNORECASE,
)
UPSCALE_SUBS_EXEC_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:[\w./\\:-]*python(?:\.exe)?\s+)?[.\\/]*upscale(?:\.py|\.bat)\b.*(?:--subs-only|--subs-upgrade)\b",
    re.IGNORECASE,
)
UPSCALE_GPU_EXEC_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:[\w./\\:-]*python(?:\.exe)?\s+)?[.\\/]*upscale(?:\.py|\.bat)\b",
    re.IGNORECASE,
)
BARE_PYTEST_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:[\w./\\:-]*python(?:\.exe)?\s+-m\s+)?pytest\b",
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


def evaluate_tool_call(payload: dict, repo_root: Path | None = None) -> dict:
    root = repo_root or _repo_root()
    tool_call = payload.get("toolCall") or {}
    tool_name = (tool_call.get("name") or "").lower()
    args = tool_call.get("args") or {}

    # 1. File read/write/edit checks for .env
    if tool_name in {"read_file", "write_file", "edit_file", "view_file"}:
        target = (
            args.get("Path")
            or args.get("AbsolutePath")
            or args.get("TargetFile")
            or ""
        )
        if _is_secret_env_path(str(target)):
            return {
                "decision": "deny",
                "reason": (
                    "Blocked by upscaler-guard (AGENTS.md §5): Never read, edit, or expose "
                    ".env files containing GEMINI_API_KEY. Use .env.example for templates."
                ),
            }
        return {"decision": "allow"}

    # 2. Shell command checks
    if tool_name == "run_command":
        cmd = str(args.get("CommandLine") or "")

        env_match = ENV_CAT_RE.search(cmd)
        if env_match and _is_secret_env_path(env_match.group(1)):
            return {
                "decision": "deny",
                "reason": (
                    "Blocked by upscaler-guard (AGENTS.md §5): Printing .env contents "
                    "in terminal output is prohibited to protect GEMINI_API_KEY."
                ),
            }

        if KEY_PRINT_RE.search(cmd):
            return {
                "decision": "deny",
                "reason": (
                    "Blocked by upscaler-guard (AGENTS.md §5): Printing GEMINI_API_KEY "
                    "to stdout/logs is strictly prohibited."
                ),
            }

        if TEMP_DELETE_RE.search(cmd) or UPSCALE_CLEAN_RE.search(cmd):
            return {
                "decision": "force_ask",
                "reason": (
                    "Checkpoint Guard (AGENTS.md §5): Deleting or overwriting .temp_<basename> "
                    "resume checkpoints requires explicit user confirmation."
                ),
            }

        # Live Gemini API quota guard
        is_subtrans_cmd = bool(SUBTRANS_EXEC_RE.search(cmd) or UPSCALE_SUBS_EXEC_RE.search(cmd))
        if is_subtrans_cmd:
            safe_flags = ("--help", "-h", "--dry-run", "--version")
            if not any(flag in cmd for flag in safe_flags):
                return {
                    "decision": "force_ask",
                    "reason": (
                        "Gemini Quota Guard (AGENTS.md §5 & subtrans/AGENTS.md): Command may "
                        "make live Gemini API requests (check/models/run without --dry-run). "
                        "Approve only if live RPD consumption is intended."
                    ),
                }

        # GPU exclusivity lock guard
        if (root / ".gpu_active.lock").exists():
            is_gpu_cmd = bool(UPSCALE_GPU_EXEC_RE.search(cmd)) or ("RUN_GPU_TESTS=1" in cmd)
            non_gpu_flags = ("--help", "-h", "--list-models", "--dry-run", "--subs-only")
            if is_gpu_cmd and not any(flag in cmd for flag in non_gpu_flags):
                return {
                    "decision": "force_ask",
                    "reason": (
                        "GPU Exclusivity Guard (AGENTS.md §4.6): .gpu_active.lock exists in "
                        "workspace root. Confirm no other GPU upscale job is active."
                    ),
                }

        # Bare pytest on production repo guard
        if BARE_PYTEST_RE.search(cmd):
            safe_pytest_markers = (
                "verify_offline.py",
                "agent_offline_guard",
                "test_agent_context.py",
                "test_agent_setup.py",
                "--version",
                "--help",
            )
            if not any(tok in cmd for tok in safe_pytest_markers):
                return {
                    "decision": "ask",
                    "reason": (
                        "Test Isolation Guard (docs/agents/VALIDATION.md): Bare pytest on the "
                        "production repo can touch .gpu_active.lock, SQLite caches, or models.yaml. "
                        "Prefer: .\\.venv\\Scripts\\python.exe scripts/verify_offline.py"
                    ),
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
