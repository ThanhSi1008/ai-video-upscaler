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
    "modern-web-guidance",
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

    guides_dir = skills_dir / "modern-web-guidance" / "guides"
    assert guides_dir.is_dir()
    guide_files = list(guides_dir.rglob("*.md"))
    assert 1 <= len(guide_files) <= 45, f"Expected pruned guides <= 45, got {len(guide_files)}"


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

    # Empty tasks dir omits Latest_Task=none noise
    msg_no_task = mod.build_status_message(tmp_path)
    assert "[upscaler-guard]" in msg_no_task
    assert "Checkpoints(.temp_*)=1" in msg_no_task
    assert "TOP_SECRET_VALUE" not in msg_no_task
    assert "Latest_Task=none" not in msg_no_task

    # Active task file appends Task=...
    tasks_dir = tmp_path / "docs" / "agents" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "TEMPLATE.md").write_text("# Template\n", encoding="utf-8")
    (tasks_dir / "2026-11-21-sample-task.md").write_text("# Active\n", encoding="utf-8")
    msg_with_task = mod.build_status_message(tmp_path)
    assert "Task=docs/agents/tasks/2026-11-21-sample-task.md" in msg_with_task


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
    assert "GEMINI.md" in res_env["reason"]
    assert "AGENTS.md" not in res_env["reason"]

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

    # 5. Bare pytest asks; safe targeted pytest allowed; GPU lock triggers force_ask
    res_pytest = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": "pytest"}}},
        repo_root=tmp_path,
    )
    assert res_pytest["decision"] == "ask"

    res_smoke = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": ".\\.venv\\Scripts\\python.exe scripts/smoke_test_pipeline.py"}}},
        repo_root=tmp_path,
    )
    assert res_smoke["decision"] == "allow"

    (tmp_path / ".gpu_active.lock").write_text("1234", encoding="utf-8")
    res_gpu = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": "python upscale.py ep1.mkv"}}},
        repo_root=tmp_path,
    )
    assert res_gpu["decision"] == "force_ask"
    (tmp_path / ".gpu_active.lock").unlink()

    # 6. Environment mutation guard (requirements.lock locked ABI)
    for mut_cmd in (
        "pip install torch",
        ".\\.venv\\Scripts\\python.exe -m pip uninstall -y numpy",
        "uv pip install fastapi",
        "conda install ffmpeg",
    ):
        res_mut = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": mut_cmd}}},
            repo_root=tmp_path,
        )
        assert res_mut["decision"] == "deny", f"Expected deny for: {mut_cmd}"

    res_freeze = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": ".\\.venv\\Scripts\\python.exe -m pip freeze"}}},
        repo_root=tmp_path,
    )
    assert res_freeze["decision"] == "allow"

    # 7. Unapproved TensorRT engine build & unflagged full-episode upscale guards
    res_trt = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": ".\\.venv\\Scripts\\python.exe src/build_trt_engine.py"}}},
        repo_root=tmp_path,
    )
    assert res_trt["decision"] == "deny"

    res_trt_inspect = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": ".\\.venv\\Scripts\\python.exe scripts/agent_context.py --symbols src/build_trt_engine.py"}}},
        repo_root=tmp_path,
    )
    assert res_trt_inspect["decision"] == "allow"

    res_full = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": "python upscale.py ep1.mkv -o out/"}}},
        repo_root=tmp_path,
    )
    assert res_full["decision"] == "deny"

    for bounded_up in (
        "python upscale.py ep1.mkv --preview 10",
        "python upscale.py ep1.mkv --dry-run",
        "python scripts/batch_translate_all.py --dry-run",
    ):
        res_ok = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": bounded_up}}},
            repo_root=tmp_path,
        )
        assert res_ok["decision"] == "allow", f"Expected allow for: {bounded_up}"

    res_live_batch = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": ".\\.venv\\Scripts\\python.exe scripts/batch_translate_all.py"}}},
        repo_root=tmp_path,
    )
    assert res_live_batch["decision"] == "force_ask"


def test_workspace_scoped_rules_and_custom_subagents():
    rules_dir = AGENTS_DIR / "rules"
    agents_dir = AGENTS_DIR / "agents"
    assert rules_dir.is_dir() and agents_dir.is_dir()

    for rule_name in ("video-gpu.md", "subtrans.md", "web-ui.md"):
        rule_path = rules_dir / rule_name
        assert rule_path.is_file(), f"Missing scoped rule: {rule_name}"
        text = rule_path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "trigger: glob" in text
        assert "globs:" in text and "description:" in text

    for agent_name in ("test-verifier.md", "subtrans-evaluator.md"):
        agent_path = agents_dir / agent_name
        assert agent_path.is_file(), f"Missing subagent spec: {agent_name}"
        text = agent_path.read_text(encoding="utf-8")
        assert "subagent: true" in text
        assert "inheritMcp: false" in text
        assert "enableWriteTools: false" in text


def test_hooks_json_and_lifecycle_post_tool_and_stop(tmp_path):
    hooks_cfg = json.loads((AGENTS_DIR / "plugins" / "upscaler-guard" / "hooks.json").read_text(encoding="utf-8"))
    assert "upscaler-post-tool-lint" in hooks_cfg
    assert "upscaler-stop-audit" in hooks_cfg

    post_mod = _load_module(
        AGENTS_DIR / "plugins" / "upscaler-guard" / "hooks" / "post_tool_lint.py",
        "upscaler_post_tool_lint",
    )
    clean_py = tmp_path / "clean_mod.py"
    clean_py.write_text('"""Clean."""\n\nVALUE = 1\n', encoding="utf-8")
    res_clean = post_mod.run_post_tool_lint(
        {"toolCall": {"name": "write_file", "args": {"Path": str(clean_py)}}},
        repo_root=tmp_path,
    )
    assert res_clean == {"ok": True, "lintErrors": ""}

    bad_py = tmp_path / "bad_mod.py"
    bad_py.write_text("def broken(:\n    pass\n", encoding="utf-8")
    res_bad = post_mod.run_post_tool_lint(
        {"toolCall": {"name": "edit_file", "args": {"Path": str(bad_py)}}},
        repo_root=tmp_path,
    )
    assert res_bad["ok"] is False
    assert res_bad["lintErrors"]

    stop_mod = _load_module(
        AGENTS_DIR / "plugins" / "upscaler-guard" / "hooks" / "stop_audit.py",
        "upscaler_stop_audit",
    )
    import subprocess
    stop_repo = tmp_path / "stop_repo"
    stop_repo.mkdir()
    subprocess.run(["git", "init", "-q", str(stop_repo)], check=True)
    # Clean repo -> empty dict (0 token overhead)
    assert stop_mod.evaluate_stop({"terminationReason": "NO_TOOL_CALL"}, repo_root=stop_repo) == {}
    # Modified broken Python triggers Stop block
    (stop_repo / "broken_stop.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    res_ruff = stop_mod.evaluate_stop({"terminationReason": "NO_TOOL_CALL"}, repo_root=stop_repo)
    assert res_ruff.get("decision") == "continue"
    assert "Ruff:" in res_ruff.get("reason", "")
    (stop_repo / "broken_stop.py").unlink()
    # Continuation loop protection
    (stop_repo / "stray.tmp").write_text("x", encoding="utf-8")
    assert stop_mod.evaluate_stop({"continuationActive": True}, repo_root=stop_repo) == {}
    # Stray root temp file detected when stopping normally
    res_stray = stop_mod.evaluate_stop({"terminationReason": "NO_TOOL_CALL"}, repo_root=stop_repo)
    assert res_stray.get("decision") == "continue"
    assert "stray.tmp" in res_stray.get("reason", "")


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

    # Oversized payload truncation guard
    huge_lines = "\n".join(f"line {i}: " + ("x" * 100) for i in range(mod.MAX_OUTPUT_LINES + 20))
    capped = mod._cap_output(huge_lines)
    assert capped.endswith("... [truncated]")
    assert len(capped) <= mod.MAX_OUTPUT_CHARS + 32


def test_first_1kb_header_utf8_decode():
    """Verify that the first 1 KB (1024 bytes) of critical modules decodes cleanly as UTF-8."""
    target_files = [
        "src/logger.py",
        "src/trt_zero_copy_upscaler.py",
        "subtrans/dtw_auto_align.py",
    ]
    for rel_path in target_files:
        p = REPO_ROOT / rel_path
        assert p.is_file(), f"Target file missing: {rel_path}"
        raw_bytes = p.read_bytes()
        decoded = raw_bytes[:1024].decode("utf-8")
        assert len(decoded) > 0
        assert all(ord(c) < 128 for c in decoded), (
            f"Non-ASCII character found in the first 1024 bytes of {rel_path}"
        )
