"""Read-only Stdio MCP Server (JSON-RPC 2.0) for e:\\ai-video-upscaler.

Provides bounded, zero-side-effect codebase and operational inspection tools:
- upscaler_memory_check: Validate memory files, budgets, and Git status.
- upscaler_find_symbol: Search Python AST classes/functions across repo.
- upscaler_file_symbols: List AST signatures and line ranges for one Python file.
- upscaler_list_checkpoints: Inspect .temp_* checkpoints and .gpu_active.lock state.
- upscaler_list_glossaries: List anime glossaries in glossaries/.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import agent_context  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
MAX_OUTPUT_CHARS = 4000
MAX_OUTPUT_LINES = 80

TOOLS = [
    {
        "name": "upscaler_memory_check",
        "description": "Check agent memory files, links, budgets, and Git HEAD.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "upscaler_find_symbol",
        "description": "Find Python AST class/function definitions across repo.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol substring.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "upscaler_file_symbols",
        "description": "List AST classes, functions, and line ranges for a Python file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative .py path.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "upscaler_list_checkpoints",
        "description": "Read-only summary of .temp_* checkpoints and .gpu_active.lock.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "upscaler_list_glossaries",
        "description": "List anime glossary files in glossaries/ with sizes.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


def _cap_output(
    text: str,
    max_chars: int = MAX_OUTPUT_CHARS,
    max_lines: int = MAX_OUTPUT_LINES,
) -> str:
    """Bound tool return payloads so raw output never floods the context window."""
    if not text:
        return text
    lines = text.splitlines()
    truncated = False
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines])
        truncated = True
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
        truncated = True
    return f"{text}\n... [truncated]" if truncated else text


def _capture(fn, *args) -> str:
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        fn(*args)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue().strip()


def call_tool(name: str, arguments: dict) -> tuple[str, bool]:
    if name == "upscaler_memory_check":
        errors = agent_context.check_memory(REPO_ROOT)
        branch = agent_context.git_output(REPO_ROOT, "branch", "--show-current") or "unknown"
        head = agent_context.git_output(REPO_ROOT, "rev-parse", "--short", "HEAD") or "unknown"
        if errors:
            return f"Branch={branch}@{head}\nFAIL:\n" + "\n".join(errors), True
        return f"Branch={branch}@{head}\nPASS: All memory files, links, and budgets valid.", False

    if name == "upscaler_find_symbol":
        query = str(arguments.get("query") or "").strip()
        if not query:
            return "Error: 'query' must be non-empty.", True
        out = _capture(agent_context.find_symbol, REPO_ROOT, query)
        return out or f"No symbols matching '{query}'.", False

    if name == "upscaler_file_symbols":
        rel = str(arguments.get("path") or "").strip()
        if not rel:
            return "Error: 'path' must be non-empty.", True
        out = _capture(agent_context.show_symbols, REPO_ROOT, rel)
        return out or f"No allowed Python file at '{rel}'.", False

    if name == "upscaler_list_checkpoints":
        gpu_lock = (REPO_ROOT / ".gpu_active.lock").exists()
        rows = [f"gpu_active_lock: {gpu_lock}"]
        seen_cps: set[Path] = set()
        checkpoint_dirs: list[Path] = []
        for base in (REPO_ROOT, REPO_ROOT / "out"):
            if not base.is_dir():
                continue
            for cp in sorted(base.glob(".temp_*")):
                if cp.is_dir():
                    res = cp.resolve()
                    if res not in seen_cps:
                        seen_cps.add(res)
                        checkpoint_dirs.append(cp)
        for cp in checkpoint_dirs:
            manifest_file = cp / "manifest.json"
            up_segs = list((cp / "upscaled_segs").glob("up_*.mp4")) if (cp / "upscaled_segs").is_dir() else []
            model_id = "unknown"
            if manifest_file.is_file():
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    model_id = str(data.get("model_id", "unknown"))
                except Exception:
                    model_id = "unreadable"
            rel_cp = cp.relative_to(REPO_ROOT).as_posix()
            rows.append(f"- {rel_cp}: model={model_id}, completed_segments={len(up_segs)}")
        if len(rows) == 1:
            rows.append("No .temp_* checkpoints present.")
        return "\n".join(rows), False

    if name == "upscaler_list_glossaries":
        g_dir = REPO_ROOT / "glossaries"
        if not g_dir.is_dir():
            return "glossaries/ directory not found.", True
        items = [
            f"- {p.name} ({p.stat().st_size} bytes)"
            for p in sorted(g_dir.glob("*.yaml"))
        ]
        return "\n".join(items) if items else "No YAML glossaries found.", False

    return f"Unknown tool: {name}", True


def handle_request(req: dict) -> dict | None:
    method = req.get("method", "")
    req_id = req.get("id")

    # Notifications do not have an id and expect no response.
    if req_id is None:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "upscaler-inspector",
                    "version": "1.0.0",
                },
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = req.get("params") or {}
        t_name = str(params.get("name") or "")
        t_args = params.get("arguments") or {}
        text, is_err = call_tool(t_name, t_args)
        bounded_text = _cap_output(text)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": bounded_text}],
                "isError": is_err,
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            sys.stderr.write(f"[upscaler-mcp] error: {exc}\n")
            sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
