"""SessionStart lifecycle hook for e:\\ai-video-upscaler.

Injects a compact (<250 chars) live operational status line into new agent sessions:
Git branch/HEAD, GPU lock status, active .temp_* checkpoints, daytime Gemini quota guard,
and active task handoff path when present. Stdlib-only; never reads .env or app runtime.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

BASELINE_RE = re.compile(r"Commit Baseline:\*{0,2}\s*`([0-9a-fA-F]{7,40})`")
RUNTIME_PATHS = ("upscale.py", "subtrans.py", "web_gui.py", "src", "subtrans", "scripts", "web")


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


def _state_freshness(root: Path, head_short: str) -> str:
    state_path = root / "docs" / "agents" / "STATE.md"
    if not state_path.is_file() or not head_short or head_short == "unknown":
        return "stale"
    try:
        content = state_path.read_text(encoding="utf-8-sig")
    except OSError:
        return "stale"
    m = BASELINE_RE.search(content)
    if not m:
        return "stale"
    baseline = m.group(1).lower()
    head = head_short.lower()
    if head.startswith(baseline) or baseline.startswith(head):
        return "fresh"
    try:
        res = subprocess.run(
            ["git", "--no-optional-locks", "diff", "--name-only", f"{baseline}..HEAD", "--", *RUNTIME_PATHS],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
        if res.returncode != 0:
            return "stale"
        return "fresh" if not res.stdout.strip() else "stale"
    except Exception:
        return "stale"


def _discover_checkpoints(root: Path) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for base in (root, root / "out"):
        if not base.is_dir():
            continue
        for p in sorted(base.glob(".temp_*")):
            if p.is_dir():
                res = p.resolve()
                if res not in seen:
                    seen.add(res)
                    found.append(p)
    return found


def build_status_message(root: Path, now: datetime | None = None) -> str:
    now = now or datetime.now()
    branch = _git(root, "branch", "--show-current") or "detached"
    head = _git(root, "rev-parse", "--short", "HEAD") or "unknown"
    porcelain = _git(root, "status", "--porcelain")
    dirty_count = len([ln for ln in porcelain.splitlines() if ln.strip()]) if porcelain else 0
    state_status = _state_freshness(root, head)
    gpu_locked = (root / ".gpu_active.lock").exists()
    checkpoints = _discover_checkpoints(root)
    quota_window = "GUARD_ACTIVE(>=14:00)" if now.hour >= 14 else "normal(<14:00)"

    parts = [
        f"[upscaler-guard] Git={branch}@{head}",
        f"GPU_Lock={'ACTIVE' if gpu_locked else 'idle'}",
        f"Checkpoints(.temp_*)={len(checkpoints)}",
        f"Dirty={dirty_count}",
        f"STATE={state_status}",
        f"Gemini_Quota={quota_window}",
    ]

    tasks_dir = root / "docs" / "agents" / "tasks"
    if tasks_dir.is_dir():
        candidates = sorted(
            p.name for p in tasks_dir.glob("*.md") if p.name.upper() != "TEMPLATE.MD"
        )
        if candidates:
            parts.append(f"Task=docs/agents/tasks/{candidates[-1]}")

    return " | ".join(parts)


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
