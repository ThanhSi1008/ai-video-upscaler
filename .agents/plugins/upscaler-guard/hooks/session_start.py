"""SessionStart lifecycle hook for e:\\ai-video-upscaler.

Injects a compact (<300 chars) live operational status line into new agent sessions:
Git branch/HEAD, GPU lock status, active .temp_* checkpoints, daytime Gemini quota guard,
and latest task handoff path. Stdlib-only; never reads .env or imports app runtime.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


def _repo_root() -> Path:
    # .agents/plugins/upscaler-guard/hooks/session_start.py -> parents[4] == repo root
    return Path(__file__).resolve().parents[4]


def _git(root: Path, *args: str) -> str:
    try:
        res = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
        return res.stdout.strip() if res.returncode == 0 else ""
    except Exception:
        return ""


def build_status_message(root: Path, now: datetime | None = None) -> str:
    now = now or datetime.now()
    branch = _git(root, "branch", "--show-current") or "detached"
    head = _git(root, "rev-parse", "--short", "HEAD") or "unknown"
    gpu_locked = (root / ".gpu_active.lock").exists()
    checkpoints = sorted(p.name for p in root.glob(".temp_*") if p.is_dir())
    quota_window = "GUARD_ACTIVE(>=14:00)" if now.hour >= 14 else "normal(<14:00)"

    tasks_dir = root / "docs" / "agents" / "tasks"
    latest_task = "none"
    if tasks_dir.is_dir():
        candidates = sorted(
            p.name for p in tasks_dir.glob("*.md") if p.name.upper() != "TEMPLATE.MD"
        )
        if candidates:
            latest_task = f"docs/agents/tasks/{candidates[-1]}"

    return (
        f"[upscaler-guard] Git={branch}@{head} | "
        f"GPU_Lock={'ACTIVE' if gpu_locked else 'idle'} | "
        f"Checkpoints(.temp_*)={len(checkpoints)} | "
        f"Gemini_Quota={quota_window} | "
        f"Latest_Task={latest_task}"
    )


def main() -> int:
    try:
        _ = sys.stdin.read()
        root = _repo_root()
        msg = build_status_message(root)
        out = {
            "injectSteps": [
                {
                    "systemMessage": {
                        "systemMessage": msg,
                    }
                }
            ]
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
    except Exception:
        sys.stdout.write("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
