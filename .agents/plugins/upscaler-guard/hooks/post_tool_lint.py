"""PostToolUse auto-lint hook for e:\\ai-video-upscaler.

Runs Ruff `--fix` quickly (< 1.5s) on modified `.py` files and validates `.json` / `.yaml` / `.yml` syntax (< 50ms).
Emits `{}` on stdout with 0 extra token overhead when clean; reports concise diagnostics to stderr on failure.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

FILE_MUTATION_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "write_to_file",
    "replace_file_content",
    "multi_replace_file_content",
})


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


def _validate_yaml_file(root: Path, resolved: Path) -> tuple[bool, str]:
    text = resolved.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        yaml.safe_load(text)
        return True, ""
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
                        timeout=1.0,
                        check=False,
                    )
                    if res.returncode == 0:
                        return True, ""
                    err = (res.stderr or res.stdout or "YAML parse error").strip().splitlines()[-1]
                    return False, f"YAML: {err[:300]}"
                except (OSError, subprocess.TimeoutExpired):
                    return True, ""
        return True, ""
    except Exception as exc:
        return False, f"YAML: {str(exc).splitlines()[0][:300]}"


def run_post_tool_lint(payload: dict, repo_root: Path | None = None) -> dict[str, object]:
    """Run targeted Ruff fix on `.py` or fast syntax check on `.json`/`.yaml`/`.yml` inside repo_root."""
    root = (repo_root or _repo_root()).resolve()
    tool_call = payload.get("toolCall") or {}
    tool_name = str(tool_call.get("name") or "").lower()
    if tool_name not in FILE_MUTATION_TOOLS or payload.get("error"):
        return {"ok": True, "lintErrors": ""}

    args = tool_call.get("args") or {}
    raw_target = str(
        args.get("Path")
        or args.get("AbsolutePath")
        or args.get("TargetFile")
        or args.get("FilePath")
        or ""
    ).strip()
    if not raw_target:
        return {"ok": True, "lintErrors": ""}

    target_path = Path(raw_target)
    if not target_path.is_absolute():
        target_path = root / target_path
    try:
        resolved = target_path.resolve()
    except OSError:
        return {"ok": True, "lintErrors": ""}
    if not resolved.is_file() or not resolved.is_relative_to(root):
        return {"ok": True, "lintErrors": ""}

    suffix = resolved.suffix.lower()
    if suffix == ".json":
        try:
            json.loads(resolved.read_text(encoding="utf-8-sig"))
            return {"ok": True, "lintErrors": ""}
        except Exception as exc:
            return {"ok": False, "lintErrors": f"JSON: {str(exc)[:300]}"}

    if suffix in {".yaml", ".yml"}:
        ok, err = _validate_yaml_file(root, resolved)
        return {"ok": ok, "lintErrors": err}

    if suffix != ".py":
        return {"ok": True, "lintErrors": ""}

    cmd = [*_ruff_command(root), "check", "--fix", "--output-format=concise", str(resolved)]
    try:
        res = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1.4,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"ok": True, "lintErrors": ""}

    if res.returncode == 0:
        return {"ok": True, "lintErrors": ""}

    raw_out = "\n".join(part.strip() for part in (res.stdout, res.stderr) if part and part.strip())
    diag_lines = [
        line.strip()
        for line in raw_out.splitlines()
        if line.strip() and not line.strip().startswith("Found ")
    ][:5]
    concise = " | ".join(diag_lines) or f"ruff check exited with code {res.returncode}"
    return {"ok": False, "lintErrors": concise[:400]}


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        result = run_post_tool_lint(payload)
        if not result.get("ok") and result.get("lintErrors"):
            sys.stderr.write(f"[upscaler-guard PostToolUse ruff] {result['lintErrors']}\n")
    except Exception:
        pass
    sys.stdout.write("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
