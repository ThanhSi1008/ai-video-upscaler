"""PreToolUse safety hook for e:\\ai-video-upscaler.

Enforces GEMINI.md, .agents/rules/*.md, and VALIDATION.md boundaries automatically:
1. Hard-denies reading/editing/printing .env or GEMINI_API_KEY values.
2. Hard-denies unapproved environment mutations (pip/uv/conda install/uninstall) protecting requirements.lock.
3. Prompts (force_ask) before deleting .temp_* checkpoints.
4. Hard-denies unapproved build_trt_engine.py or unflagged full-episode upscale runs.
5. Prompts (force_ask) before invoking live paid Gemini API commands.
6. Prompts (ask / force_ask) before running bare pytest or concurrent GPU jobs.
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
BUILD_TRT_EXEC_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:[\w./\\:-]*python(?:\.exe)?\s+(?:-m\s+)?)?(?:[\w./\\:-]*[/\\])?(?:src[/\\.]build_trt_engine|build_trt_engine(?:\.py)?)\b",
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

SAFE_SUBTRANS_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:--help|-h|--dry-run|--offline|--version|--mock|--test)(?:\s|$)",
    re.IGNORECASE,
)
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
                "reason": "Blocked (GEMINI.md §4): Never read, edit, or expose .env secrets; use .env.example.",
            }
        return {"decision": "allow"}

    # 2. Shell command checks
    if tool_name == "run_command":
        cmd = str(args.get("CommandLine") or "")

        env_match = ENV_CAT_RE.search(cmd)
        if env_match and _is_secret_env_path(env_match.group(1)):
            return {
                "decision": "deny",
                "reason": "Blocked (GEMINI.md §4): Printing .env contents to stdout is prohibited.",
            }

        if KEY_PRINT_RE.search(cmd):
            return {
                "decision": "deny",
                "reason": "Blocked (GEMINI.md §4): Printing GEMINI_API_KEY to stdout/logs is prohibited.",
            }

        if ENV_MUTATION_RE.search(cmd):
            return {
                "decision": "deny",
                "reason": "Blocked (GEMINI.md §2): Environment mutations (pip/uv/conda install) are prohibited to protect locked ABI (requirements.lock).",
            }

        if TEMP_DELETE_RE.search(cmd) or UPSCALE_CLEAN_RE.search(cmd):
            return {
                "decision": "force_ask",
                "reason": "Checkpoint Guard (GEMINI.md §4): Deleting or overwriting .temp_* checkpoints requires confirmation.",
            }

        if BUILD_TRT_EXEC_RE.search(cmd) and not any(flag in cmd for flag in ("--help", "-h", "--dry-run")):
            return {
                "decision": "deny",
                "reason": "Blocked (video-gpu.md): Running build_trt_engine.py requires explicit user approval.",
            }

        # Live Gemini API quota guard
        is_subtrans_cmd = bool(SUBTRANS_EXEC_RE.search(cmd) or UPSCALE_SUBS_EXEC_RE.search(cmd))
        if is_subtrans_cmd and not SAFE_SUBTRANS_FLAGS_RE.search(cmd):
            return {
                "decision": "force_ask",
                "reason": "Gemini Quota Guard (GEMINI.md §4 & subtrans.md): Command may consume live Gemini API RPD; confirm intent.",
            }

        # GPU exclusivity lock guard
        if (root / ".gpu_active.lock").exists():
            is_gpu_cmd = bool(UPSCALE_GPU_EXEC_RE.search(cmd)) or ("RUN_GPU_TESTS=1" in cmd)
            non_gpu_flags = ("--help", "-h", "--list-models", "--dry-run", "--subs-only")
            if is_gpu_cmd and not any(flag in cmd for flag in non_gpu_flags):
                return {
                    "decision": "force_ask",
                    "reason": "GPU Exclusivity Guard (video-gpu.md): .gpu_active.lock exists; confirm no other GPU job is running.",
                }

        # Unflagged full-episode upscale guard
        if UPSCALE_GPU_EXEC_RE.search(cmd) and not SAFE_UPSCALE_FLAGS_RE.search(cmd):
            return {
                "decision": "deny",
                "reason": "Blocked (video-gpu.md): Full-episode upscale runs require --preview, --dry-run, or scripts/smoke_test_pipeline.py.",
            }

        # Bare pytest on production repo guard
        if BARE_PYTEST_RE.search(cmd):
            safe_pytest_markers = (
                "verify_offline.py",
                "smoke_test_pipeline.py",
                "agent_offline_guard",
                "test_agent_context.py",
                "test_agent_setup.py",
                "test_verify_offline.py",
                "test_smoke_pipeline.py",
                "test_web_gui.py",
                "--version",
                "--help",
            )
            if not any(tok in cmd for tok in safe_pytest_markers):
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
