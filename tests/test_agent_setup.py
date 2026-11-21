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
        expected_fm_name = "local-web-guidance" if skill_name == "modern-web-guidance" else skill_name
        assert f"name: {expected_fm_name}" in fm
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


def test_session_start_hook_emits_valid_inject_steps(tmp_path, monkeypatch):
    mod = _load_module(
        AGENTS_DIR / "plugins" / "upscaler-guard" / "hooks" / "session_start.py",
        "upscaler_session_start",
    )
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    (tmp_path / ".env").write_text("GEMINI_API_KEY=TOP_SECRET_VALUE", encoding="utf-8")
    (tmp_path / ".temp_ep01").mkdir()
    (tmp_path / "out" / ".temp_ep02").mkdir(parents=True)

    # Empty tasks dir omits Latest_Task=none noise; scans both root and out/.temp_* + STATE/Dirty
    msg_no_task = mod.build_status_message(tmp_path)
    assert "[upscaler-guard]" in msg_no_task
    assert "Checkpoints(.temp_*)=2" in msg_no_task
    assert "Dirty=0" in msg_no_task
    assert "STATE=stale" in msg_no_task
    assert "TOP_SECRET_VALUE" not in msg_no_task
    assert "Latest_Task=none" not in msg_no_task

    # Matching STATE baseline in git repo emits STATE=fresh and Dirty=count
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "docs" / "agents" / "tasks").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "agents" / "tasks" / "TEMPLATE.md").write_text("# Template\n", encoding="utf-8")
    (tmp_path / "docs" / "agents" / "STATE.md").write_text("Baseline\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "docs/agents/STATE.md", "docs/agents/tasks/TEMPLATE.md"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=T", "-c", "user.email=t@e", "-c", "commit.gpgsign=false", "commit", "-qm", "init"],
        check=True,
    )
    head_sha = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "--short", "HEAD"], text=True).strip()
    (tmp_path / "docs" / "agents" / "STATE.md").write_text(f"> **Commit Baseline:** `{head_sha}`\n", encoding="utf-8")
    (tmp_path / "docs" / "agents" / "tasks" / "2026-11-21-sample-task.md").write_text("# Active\n", encoding="utf-8")
    msg_with_task = mod.build_status_message(tmp_path)
    assert "STATE=fresh" in msg_with_task
    assert "Dirty=" in msg_with_task
    assert "Task=docs/agents/tasks/2026-11-21-sample-task.md" in msg_with_task


def test_pre_tool_guard_enforces_project_boundaries(tmp_path):
    mod = _load_module(
        AGENTS_DIR / "plugins" / "upscaler-guard" / "hooks" / "pre_tool_guard.py",
        "upscaler_pre_tool_guard",
    )

    # 1. Local .env reads are allowed; writes prompt (force_ask); git add/commit on .env is denied
    for read_tool in ("read_file", "view_file"):
        res_env_read = mod.evaluate_tool_call(
            {"toolCall": {"name": read_tool, "args": {"Path": "e:/ai-video-upscaler/.env"}}},
            repo_root=tmp_path,
        )
        assert res_env_read["decision"] == "allow"

    for write_tool in ("write_file", "edit_file", "replace_file_content"):
        res_env_write = mod.evaluate_tool_call(
            {"toolCall": {"name": write_tool, "args": {"TargetFile": ".env"}}},
            repo_root=tmp_path,
        )
        assert res_env_write["decision"] == "force_ask"

    for bad_git in ("git add .env", "git commit subtrans/.env -m 'oops'"):
        res = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": bad_git}}},
            repo_root=tmp_path,
        )
        assert res["decision"] == "deny", f"Expected deny for: {bad_git}"

    res_safe_commit = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": 'git commit -m "Document .env policy"'}}},
        repo_root=tmp_path,
    )
    assert res_safe_commit["decision"] == "allow"

    res_shell_env_write = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": "Set-Content .env 'KEY=1'"}}},
        repo_root=tmp_path,
    )
    assert res_shell_env_write["decision"] == "force_ask"

    # 2. Context-Window & Rewrite Firewall checks
    big_py = tmp_path / "src" / "big_mod.py"
    big_py.parent.mkdir(parents=True, exist_ok=True)
    big_py.write_text("\n".join(f"x_{i} = {i}" for i in range(380)) + "\n", encoding="utf-8")

    res_bundle_read = mod.evaluate_tool_call(
        {"toolCall": {"name": "read_file", "args": {"Path": "scratch/review_bundles/latest_diff.txt"}}},
        repo_root=tmp_path,
    )
    assert res_bundle_read["decision"] == "deny"

    res_bundle_cat = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": "cat scratch/review_bundles/focus_web.txt"}}},
        repo_root=tmp_path,
    )
    assert res_bundle_cat["decision"] == "deny"

    res_bundle_gen = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": ".\\.venv\\Scripts\\python.exe scripts/export_review_bundle.py --sync-latest"}}},
        repo_root=tmp_path,
    )
    assert res_bundle_gen["decision"] == "allow"

    for full_w in ("write_file", "write_to_file"):
        res_rw = mod.evaluate_tool_call(
            {"toolCall": {"name": full_w, "args": {"Path": "src/big_mod.py"}}},
            repo_root=tmp_path,
        )
        assert res_rw["decision"] == "deny"

    for r_tool in ("read_file", "view_file"):
        res_view_unbounded = mod.evaluate_tool_call(
            {"toolCall": {"name": r_tool, "args": {"AbsolutePath": str(big_py)}}},
            repo_root=tmp_path,
        )
        assert res_view_unbounded["decision"] == "deny"

        res_view_eof_span = mod.evaluate_tool_call(
            {"toolCall": {"name": r_tool, "args": {"AbsolutePath": str(big_py), "StartLine": 50}}},
            repo_root=tmp_path,
        )
        assert res_view_eof_span["decision"] == "deny"

        res_view_bounded = mod.evaluate_tool_call(
            {"toolCall": {"name": r_tool, "args": {"AbsolutePath": str(big_py), "StartLine": 100, "EndLine": 250}}},
            repo_root=tmp_path,
        )
        assert res_view_bounded["decision"] == "allow"

        res_offset_limit = mod.evaluate_tool_call(
            {"toolCall": {"name": r_tool, "args": {"Path": str(big_py), "offset": 20, "limit": 200}}},
            repo_root=tmp_path,
        )
        assert res_offset_limit["decision"] == "allow"

    res_dump = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": "Get-Content src/big_mod.py"}}},
        repo_root=tmp_path,
    )
    assert res_dump["decision"] == "deny"

    res_dump_bounded = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": "Get-Content src/big_mod.py -TotalCount 50"}}},
        repo_root=tmp_path,
    )
    assert res_dump_bounded["decision"] == "allow"

    # 3. Guard deleting .temp_* checkpoints
    for cp_cmd in ("Remove-Item -Recurse -Force .temp_ep01", "python upscale.py --clean-temp"):
        res = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": cp_cmd}}},
            repo_root=tmp_path,
        )
        assert res["decision"] == "force_ask", f"Expected force_ask for: {cp_cmd}"

    # 4. Guard live Gemini API calls vs dry-run / local commands
    for live_cmd in (
        "python subtrans.py check",
        ".\\subtrans.bat run ep.ja.srt -o out/",
        "python benchmarks/subtrans/eval_golden_pronouns.py",
        "python subtrans.py run ep.ja.srt --offline",
    ):
        res = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": live_cmd}}},
            repo_root=tmp_path,
        )
        assert res["decision"] == "force_ask", f"Expected force_ask for: {live_cmd}"

    for safe_cmd in (
        "python subtrans.py run ep.ja.srt -o scratch/dry/ --dry-run",
        "python subtrans.py eval",
        "python subtrans.py compare a.srt b.srt",
        "python subtrans.py rag stats",
        "python benchmarks/subtrans/eval_golden_pronouns.py --dry-run",
        "python benchmarks/subtrans/validate_pronouns_regex.py",
        "python subtrans.py --help",
        "ruff check upscale.py subtrans.py src subtrans",
        ".\\.venv\\Scripts\\python.exe scripts/verify_offline.py",
    ):
        res = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": safe_cmd}}},
            repo_root=tmp_path,
        )
        assert res["decision"] == "allow", f"Expected allow for: {safe_cmd}"

    # 5. Bare or mixed-unsafe pytest asks; strictly safe targeted pytest allowed; GPU lock triggers force_ask
    for ask_pytest in (
        "pytest",
        "pytest tests/test_agent_context.py tests/test_subtitle_guardrails.py",
    ):
        res_pytest = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": ask_pytest}}},
            repo_root=tmp_path,
        )
        assert res_pytest["decision"] == "ask"

    res_safe_pytest = mod.evaluate_tool_call(
        {"toolCall": {"name": "run_command", "args": {"CommandLine": "pytest tests/test_agent_context.py tests/test_agent_setup.py -q"}}},
        repo_root=tmp_path,
    )
    assert res_safe_pytest["decision"] == "allow"

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

    for extra_gpu_cmd in (
        ".\\upscale_turbo.bat ep1.mkv",
        "python scripts/benchmark_pipeline.py",
        "python scripts/silicon_roofline_probe.py",
        "python scripts/upscale_x4_single.py ep1.mkv",
    ):
        res_gpu_extra = mod.evaluate_tool_call(
            {"toolCall": {"name": "run_command", "args": {"CommandLine": extra_gpu_cmd}}},
            repo_root=tmp_path,
        )
        assert res_gpu_extra["decision"] == "deny", f"Expected deny for: {extra_gpu_cmd}"

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

    for agent_name in ("test-verifier.md", "subtrans-evaluator.md", "repo-scout.md"):
        agent_path = agents_dir / agent_name
        assert agent_path.is_file(), f"Missing subagent spec: {agent_name}"
        text = agent_path.read_text(encoding="utf-8")
        assert "subagent: true" in text
        assert "inheritMcp: false" in text
        assert "enableWriteTools: true" in text
        for forbidden_tool in (
            "write_file",
            "edit_file",
            "write_to_file",
            "replace_file_content",
            "multi_replace_file_content",
        ):
            assert forbidden_tool in text, f"{agent_name} must prohibit {forbidden_tool}"

    subtrans_rule = (rules_dir / "subtrans.md").read_text(encoding="utf-8")
    assert "subtrans/**/*" in subtrans_rule and "glossaries/*.yaml" in subtrans_rule


def test_hooks_json_and_lifecycle_post_tool_and_stop(tmp_path):
    hooks_cfg = json.loads((AGENTS_DIR / "plugins" / "upscaler-guard" / "hooks.json").read_text(encoding="utf-8"))
    assert "upscaler-post-tool-lint" in hooks_cfg
    assert "upscaler-stop-audit" in hooks_cfg
    pre_hook = hooks_cfg["upscaler-pre-tool-guard"]["PreToolUse"][0]
    assert "view_file" in pre_hook["matcher"] and "write_to_file" in pre_hook["matcher"]
    assert "..\\..\\..\\.venv\\Scripts\\python.exe" in pre_hook["hooks"][0]["command"]
    assert hooks_cfg["upscaler-post-tool-lint"]["PostToolUse"][0]["hooks"][0]["timeout"] == 3
    assert hooks_cfg["upscaler-stop-audit"]["Stop"][0]["timeout"] == 3

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

    # Fast JSON and YAML validation via FilePath argument
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    res_bad_json = post_mod.run_post_tool_lint(
        {"toolCall": {"name": "write_to_file", "args": {"FilePath": str(bad_json)}}},
        repo_root=tmp_path,
    )
    assert res_bad_json["ok"] is False and "JSON:" in str(res_bad_json["lintErrors"])

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("shows:\n  - [unclosed\n", encoding="utf-8")
    res_bad_yaml = post_mod.run_post_tool_lint(
        {"toolCall": {"name": "replace_file_content", "args": {"FilePath": str(bad_yaml)}}},
        repo_root=tmp_path,
    )
    assert res_bad_yaml["ok"] is False and "YAML:" in str(res_bad_yaml["lintErrors"])

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
    # Staged .env secret triggers Stop block
    (stop_repo / ".env").write_text("GEMINI_API_KEY=SECRET\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(stop_repo), "add", "-f", ".env"], check=True)
    res_env_staged = stop_mod.evaluate_stop({"terminationReason": "NO_TOOL_CALL"}, repo_root=stop_repo)
    assert res_env_staged.get("decision") == "continue"
    assert "Staged .env secret" in res_env_staged.get("reason", "")
    subprocess.run(["git", "-C", str(stop_repo), "rm", "--cached", "-f", "-q", ".env"], check=True)
    (stop_repo / ".env").unlink()
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
    (stop_repo / "stray.tmp").unlink()

    # Modified broken YAML or JSON triggers Stop block
    (stop_repo / "shows.yaml").write_text("shows:\n  - [unclosed\n", encoding="utf-8")
    res_yaml_stop = stop_mod.evaluate_stop({"terminationReason": "NO_TOOL_CALL"}, repo_root=stop_repo)
    assert res_yaml_stop.get("decision") == "continue"
    assert "YAML" in res_yaml_stop.get("reason", "")
    (stop_repo / "shows.yaml").unlink()


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
