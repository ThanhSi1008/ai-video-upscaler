"""Stop lifecycle audit hook for e:\\ai-video-upscaler.

Fast (< 2s) pre-completion check:
1. Checks `git status --porcelain` for stray root temporary files.
2. Runs `ruff check` ONLY on modified/untracked `.py` files.
3. Runs `scripts/agent_context.py` `check_memory()` if memory/agent files changed.

Never runs `pytest` or `verify_offline.py`. Returns `{}` with zero token overhead
when clean, or `{"decision": "continue", "reason": ...}` on actionable failures.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

STRAY_ROOT_EXACT = frozenset({"temp_sub.ass", "vid_probe.json"})
SAFE_ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})


def _is_secret_env_rel(rel_posix: str) -> bool:
    name = Path(rel_posix).name.lower()
    if name in SAFE_ENV_TEMPLATES:
        return False
    return name == ".env" or name.startswith(".env.")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ruff_command(root: Path) -> list[str]:
    for cand in (
        root / ".venv" / "Scripts" / "ruff.exe",
        Path(sys.executable).resolve().parent / "ruff.exe",
    ):
        if cand.is_file():
            return [str(cand)]
    return [sys.executable, "-m", "ruff"]


def _validate_yaml_file(root: Path, resolved: Path) -> str:
    text = resolved.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        yaml.safe_load(text)
        return ""
    except ImportError:
        for cand in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
            if cand.is_file():
                try:
                    res = subprocess.run(
                        [str(cand), "-c", "import sys, yaml; yaml.safe_load(open(sys.argv[1], encoding='utf-8-sig'))", str(resolved)],
                        cwd=str(root),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=0.8,
                        check=False,
                    )
                    if res.returncode == 0:
                        return ""
                    err = (res.stderr or res.stdout or "YAML parse error").strip().splitlines()[-1]
                    return f"YAML ({resolved.relative_to(root).as_posix()}): {err[:160]}"
                except (OSError, subprocess.TimeoutExpired):
                    return ""
        return ""
    except Exception as exc:
        return f"YAML ({resolved.relative_to(root).as_posix()}): {str(exc).splitlines()[0][:160]}"


def _is_stray_root_temp(rel_posix: str) -> bool:
    p = Path(rel_posix)
    if p.parent != Path("."):
        return False
    name = p.name
    return (
        name.endswith(".tmp")
        or name.startswith("_tmp_")
        or name.startswith(".preview_clip_")
        or name in STRAY_ROOT_EXACT
    )


def _parse_porcelain_paths(stdout: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw_line in stdout.splitlines():
        if len(raw_line) < 4:
            continue
        status = raw_line[:2]
        rel = raw_line[3:].strip()
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1].strip()
        rel = rel.strip('"')
        if rel:
            entries.append((status, rel.replace("\\", "/")))
    return entries


def evaluate_stop(payload: dict, repo_root: Path | None = None) -> dict[str, str]:
    """Run bounded fast repository hygiene checks before finalizing the agent turn."""
    if payload.get("continuationActive"):
        return {}
    term_reason = str(payload.get("terminationReason") or "NO_TOOL_CALL")
    if term_reason != "NO_TOOL_CALL":
        return {}

    root = (repo_root or _repo_root()).resolve()
    issues: list[str] = []

    try:
        staged_res = subprocess.run(
            ["git", "--no-optional-locks", "diff", "--cached", "--name-only"],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=0.6,
            check=False,
        )
        if staged_res.returncode == 0 and staged_res.stdout.strip():
            staged_secrets = [
                ln.strip().replace("\\", "/")
                for ln in staged_res.stdout.splitlines()
                if ln.strip() and _is_secret_env_rel(ln.strip())
            ]
            if staged_secrets:
                issues.append(f"Staged .env secret(s): {', '.join(staged_secrets[:3])}")
    except (OSError, subprocess.TimeoutExpired):
        pass

    try:
        status_res = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=0.6,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        status_res = None

    if status_res is not None and status_res.returncode == 0 and status_res.stdout.strip():
        changed = _parse_porcelain_paths(status_res.stdout)

        stray = [
            rel
            for status, rel in changed
            if "D" not in status and _is_stray_root_temp(rel) and (root / rel).exists()
        ]
        if stray:
            issues.append(f"Stray root temp file(s): {', '.join(stray[:3])}")

        for status, rel in changed:
            if "D" in status:
                continue
            target = root / rel
            if not target.is_file():
                continue
            low = rel.lower()
            if low.endswith(".json"):
                try:
                    json.loads(target.read_text(encoding="utf-8-sig"))
                except Exception as exc:
                    issues.append(f"JSON ({rel}): {str(exc)[:160]}")
            elif low.endswith((".yaml", ".yml")):
                yaml_err = _validate_yaml_file(root, target)
                if yaml_err:
                    issues.append(yaml_err)

        modified_py = [
            str(root / rel)
            for status, rel in changed
            if "D" not in status and rel.lower().endswith(".py") and (root / rel).is_file()
        ]
        if modified_py:
            try:
                ruff_res = subprocess.run(
                    [*_ruff_command(root), "check", "--output-format=concise", *modified_py],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=0.9,
                    check=False,
                )
                if ruff_res.returncode != 0:
                    raw_err = ruff_res.stdout.strip() or ruff_res.stderr.strip()
                    first_line = raw_err.splitlines()[0].strip() if raw_err else "lint errors"
                    issues.append(f"Ruff: {first_line[:180]}")
            except (OSError, subprocess.TimeoutExpired):
                pass

        memory_touched = any(
            rel == "GEMINI.md" or rel.startswith("docs/agents/") or rel.startswith(".agents/")
            for _, rel in changed
        )
        if memory_touched:
            root_str = str(root)
            added_sys_path = False
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
                added_sys_path = True
            try:
                from scripts.agent_context import check_memory

                mem_errors = check_memory(root)
                if mem_errors:
                    issues.append(f"agent_context: {mem_errors[0][:160]}")
            except Exception:
                pass
            finally:
                if added_sys_path and sys.path and sys.path[0] == root_str:
                    sys.path.pop(0)

    if not issues:
        return {}
    return {
        "decision": "continue",
        "reason": "[upscaler-guard Stop] " + "; ".join(issues[:3]),
    }


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        decision = evaluate_stop(payload)
        sys.stdout.write(json.dumps(decision, ensure_ascii=False))
    except Exception:
        sys.stdout.write("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
