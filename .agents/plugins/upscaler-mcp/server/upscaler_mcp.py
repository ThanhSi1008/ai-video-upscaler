"""Read-only Stdio MCP Server (JSON-RPC 2.0) for e:\\ai-video-upscaler.

Provides safe, zero-side-effect codebase & operational inspection tools:
- upscaler_memory_check: Run agent_context memory/budget/freshness check.
- upscaler_find_symbol: Locate Python classes/functions across the repo via AST.
- upscaler_file_symbols: List AST signatures & line ranges for a specific Python file.
- upscaler_list_checkpoints: Inspect .temp_* checkpoint manifests read-only (never modifies/deletes).
- upscaler_list_glossaries: List available per-series anime glossaries in glossaries/.
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

TOOLS = [
    {
        "name": "upscaler_memory_check",
        "description": "Verify agent memory files, relative links, instruction byte budgets, and Git baseline status (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "upscaler_find_symbol",
        "description": "Search Python AST definitions across src/, subtrans/, scripts/, tests/ by class or function name.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Partial or exact class/function name to locate.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "upscaler_file_symbols",
        "description": "Extract AST classes, methods, functions, and line ranges for a relative Python path (e.g. 'src/pipeline.py').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to a Python source file.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "upscaler_list_checkpoints",
        "description": "Read-only summary of any .temp_<basename> video upscale checkpoints and .gpu_active.lock state.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "upscaler_list_glossaries",
        "description": "List per-series anime glossaries under glossaries/ with file sizes.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


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
        return f"Branch={branch}@{head}\nPASS: All memory files, links, and byte budgets valid.", False

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
        for cp in sorted(REPO_ROOT.glob(".temp_*")):
            if not cp.is_dir():
                continue
            manifest_file = cp / "manifest.json"
            up_segs = list((cp / "upscaled_segs").glob("up_*.mp4")) if (cp / "upscaled_segs").is_dir() else []
            model_id = "unknown"
            if manifest_file.is_file():
                try:
                    data = json.loads(manifest_file.read_text(encoding="utf-8"))
                    model_id = str(data.get("model_id", "unknown"))
                except Exception:
                    model_id = "unreadable"
            rows.append(f"- {cp.name}: model={model_id}, completed_segments={len(up_segs)}")
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
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": text}],
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
