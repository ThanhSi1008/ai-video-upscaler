"""Tests for workspace AI Agent customizations (.agents/skills, .agents/plugins, hooks, MCP)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / ".agents"

EXPECTED_SKILLS = {
    "safe-offline-test",
    "video-upscale-ops",
    "subtrans-anime-ops",
    "agent-handoff-commit",
}

ALLOWED_PLUGIN_KEYS = {
    "name",
    "description",
    "logo",
    "suggestedPrompts",
    "displayName",
    "version",
    "author",
    "homepage",
    "repository",
    "disabled",
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_workspace_skills_have_valid_frontmatter():
    skills_dir = AGENTS_DIR / "skills"
    assert skills_dir.is_dir(), "Missing .agents/skills directory"
    discovered = {p.name for p in skills_dir.iterdir() if p.is_dir()}
    assert EXPECTED_SKILLS.issubset(discovered)

    for skill_name in EXPECTED_SKILLS:
        skill_md = skills_dir / skill_name / "SKILL.md"
        assert skill_md.is_file(), f"Missing SKILL.md for {skill_name}"
        text = skill_md.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{skill_name}/SKILL.md must begin with YAML frontmatter"
        match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
        assert match is not None, f"Malformed YAML frontmatter in {skill_name}/SKILL.md"
        fm = match.group(1)
        assert f"name: {skill_name}" in fm
        assert "description:" in fm


def test_workspace_plugins_manifests_and_logos():
    plugins_dir = AGENTS_DIR / "plugins"
    for plugin_name in ("upscaler-guard", "upscaler-mcp"):
        p_dir = plugins_dir / plugin_name
        manifest_path = p_dir / "plugin.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["name"] == plugin_name
        assert set(manifest.keys()).issubset(ALLOWED_PLUGIN_KEYS)
        assert 1 <= len(manifest.get("suggestedPrompts", [])) <= 3
        logo = p_dir / manifest["logo"]
        assert logo.is_file()


def test_session_start_hook_emits_valid_inject_steps(tmp_path):
    mod = _load_module(
        AGENTS_DIR / "plugins" / "upscaler-guard" / "hooks" / "session_start.py",
        "upscaler_session_start",
    )
    (tmp_path / ".env").write_text("GEMINI_API_KEY=TOP_SECRET_VALUE", encoding="utf-8")
    (tmp_path / ".temp_ep01").mkdir()
    msg = mod.build_status_message(tmp_path)
    assert "[upscaler-guard]" in msg
    assert "Checkpoints(.temp_*)=1" in msg
    assert "TOP_SECRET_VALUE" not in msg


def test_pre_tool_guard_enforces_project_boundaries(tmp_path):
    mod = _load_module(
        AGENTS_DIR / "plugins" / "upscaler-guard" / "hooks" / "pre_tool_guard.py",
        "upscaler_pre_tool_guard",
    )

    # 1. Block reading .env, allow reading .env.example
    res_env = mod.evaluate_tool_call(
        {"toolCall": {"name": "read_file", "args": {"Path": "e:/ai-video-upscaler/.env"}}},
        repo_root=tmp_path,
    )
    assert res_env["decision"] == "deny"

    res_example = mod.evaluate_tool_call(
        {"toolCall": {"name": "read_file", "args": {"Path": "e:/ai-video-upscaler/.env.example"}}},
        repo_root=tmp_path,
    )
    assert res_example["decision"] == "allow"

    # 2. Block shell printing .env or GEMINI_API_KEY
    for bad_cmd in ("Get-Content .env", "cat subtrans/.env", "echo $env:GEMINI_API_KEY"):
        res = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": bad_cmd}}},
            repo_root=tmp_path,
        )
        assert res["decision"] == "deny", f"Expected deny for: {bad_cmd}"

    # 3. Guard deleting .temp_* checkpoints
    for cp_cmd in ("Remove-Item -Recurse -Force .temp_ep01", "python upscale.py --clean-temp"):
        res = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": cp_cmd}}},
            repo_root=tmp_path,
        )
        assert res["decision"] == "force_ask", f"Expected force_ask for: {cp_cmd}"

    # 4. Guard live Gemini API calls vs dry-run / linter commands
    for live_cmd in ("python subtrans.py check", ".\\subtrans.bat run ep.ja.srt -o out/"):
        res = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": live_cmd}}},
            repo_root=tmp_path,
        )
        assert res["decision"] == "force_ask", f"Expected force_ask for: {live_cmd}"

    for safe_cmd in (
        "python subtrans.py run ep.ja.srt -o scratch/dry/ --dry-run",
        "python subtrans.py --help",
        "ruff check upscale.py subtrans.py src subtrans",
        ".\\.venv\\Scripts\\python.exe scripts/verify_offline.py",
    ):
        res = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": safe_cmd}}},
            repo_root=tmp_path,
        )
        assert res["decision"] == "allow", f"Expected allow for: {safe_cmd}"

    # 5. Bare pytest asks; GPU lock triggers force_ask
    res_pytest = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": "pytest"}}},
        repo_root=tmp_path,
    )
    assert res_pytest["decision"] == "ask"

    (tmp_path / ".gpu_active.lock").write_text("1234", encoding="utf-8")
    res_gpu = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": "python upscale.py ep1.mkv"}}},
        repo_root=tmp_path,
    )
    assert res_gpu["decision"] == "force_ask"


def test_upscaler_mcp_server_jsonrpc_handlers():
    mod = _load_module(
        AGENTS_DIR / "plugins" / "upscaler-mcp" / "server" / "upscaler_mcp.py",
        "upscaler_mcp_server",
    )

    init_resp = mod.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "upscaler-inspector"

    # Notification returns None
    assert mod.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    list_resp = mod.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_resp is not None
    tool_names = {t["name"] for t in list_resp["result"]["tools"]}
    assert "upscaler_find_symbol" in tool_names
    assert "upscaler_list_glossaries" in tool_names

    call_resp = mod.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "upscaler_list_glossaries", "arguments": {}},
        }
    )
    assert call_resp is not None
    assert call_resp["result"]["isError"] is False
    assert "glossary." in call_resp["result"]["content"][0]["text"]
