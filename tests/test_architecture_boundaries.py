"""Architectural boundary, CSP integrity, mock status schema, and hygiene tests."""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path
import re

from fastapi.testclient import TestClient
import pytest

import web.app as web_app


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_WEB_MODULES = frozenset({
    "torch",
    "torchvision",
    "tensorrt",
    "triton",
    "spandrel",
    "src.config",
    "src.pipeline",
    "src.cuda_engine",
    "src.build_trt_engine",
    "src.trt_zero_copy_upscaler",
})
FORBIDDEN_SRC_ATTRS = frozenset({
    "config",
    "pipeline",
    "cuda_engine",
    "build_trt_engine",
    "trt_zero_copy_upscaler",
})
CODING_HEADER_RE = re.compile(r"^#.*coding[:=]\s*[-\w.]+")
DOC_ECHO_RE = re.compile(r"^(?:Module:\s*)?[\w./\\-]+(?:\.py)?\s*$")


def _iter_runtime_python_files() -> list[Path]:
    files: list[Path] = []
    for name in ("upscale.py", "subtrans.py", "web_gui.py"):
        p = REPO_ROOT / name
        if p.is_file():
            files.append(p)
    for rel_dir in ("src", "subtrans", "web", "scripts"):
        base = REPO_ROOT / rel_dir
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(REPO_ROOT).as_posix()
            if rel.startswith(("subtrans/tests/", "scripts/oneoff/", "subtrans/.cache/")):
                continue
            if "__pycache__" in p.parts:
                continue
            files.append(p)
    return files


def test_web_zero_vram_ast_boundaries():
    """Verify web/ and web_gui.py never import GPU, TensorRT, or pipeline runtime modules."""
    targets = [REPO_ROOT / "web_gui.py", *sorted((REPO_ROOT / "web").rglob("*.py"))]
    violations: list[str] = []

    for path in targets:
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        rel = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN_WEB_MODULES):
                        violations.append(f"{rel}:{node.lineno} imports {mod}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if any(mod == f or mod.startswith(f + ".") for f in FORBIDDEN_WEB_MODULES):
                    violations.append(f"{rel}:{node.lineno} imports from {mod}")
                elif mod == "src":
                    for alias in node.names:
                        if alias.name in FORBIDDEN_SRC_ATTRS:
                            violations.append(f"{rel}:{node.lineno} imports src.{alias.name}")

    assert not violations, "Web zero-VRAM boundary violated:\n" + "\n".join(violations)


def test_csp_inline_theme_script_sha256_matches_index_html():
    """Verify CSP header hash in web/app.py matches web/static/index.html inline theme script."""
    html = (REPO_ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
    assert len(scripts) == 1, f"Expected 1 inline <script> tag in index.html, found {len(scripts)}"
    digest_b64 = base64.b64encode(hashlib.sha256(scripts[0].encode("utf-8")).digest()).decode("ascii")
    expected_csp_hash = f"sha256-{digest_b64}"
    assert web_app._INLINE_THEME_SCRIPT_SHA256 == expected_csp_hash
    assert expected_csp_hash in web_app._CSP_HEADER_VALUE


def test_mock_status_chunk_indices_are_zero_based_and_completed_only(tmp_path):
    """Verify mock /api/status emits 0-based completed-only chunk entries matching active_chunk."""
    app = web_app.create_app(workspace_root=tmp_path, mock_mode=True)
    with TestClient(app) as client:
        resp = client.get("/api/status")
    assert resp.status_code == 200
    live = resp.json()["pipeline"]["live"]
    completed = live["completed_chunks"]
    chunks = live["chunks"]
    assert isinstance(completed, int) and completed > 0
    assert len(chunks) == completed
    assert [c["chunk_idx"] for c in chunks] == list(range(completed))
    assert 0 <= live["active_chunk"] < live["total_chunks"]
    assert live["active_chunk"] == completed
    assert live["resumed_chunks"] == [0, 1]


def test_runtime_and_test_python_hygiene():
    """Verify UTF-8 integrity, zero BOM/NUL, no coding banners, and non-echo module docstrings."""
    runtime_files = _iter_runtime_python_files()
    runtime_set = set(runtime_files)
    all_py = list(runtime_files)
    for test_dir in (REPO_ROOT / "tests", REPO_ROOT / "subtrans" / "tests"):
        if test_dir.is_dir():
            for p in sorted(test_dir.rglob("*.py")):
                if "__pycache__" not in p.parts:
                    all_py.append(p)

    for path in all_py:
        raw = path.read_bytes()
        assert b"\x00" not in raw, f"NUL byte in {path}"
        assert not raw.startswith((b"\xff\xfe", b"\xfe\xff")), f"UTF-16 BOM in {path}"
        text = raw.decode("utf-8")
        if path in runtime_set:
            top_lines = text.splitlines()[:2]
            assert not any(CODING_HEADER_RE.match(ln) for ln in top_lines), (
                f"Legacy coding header present in {path}"
            )
            tree = ast.parse(text, filename=str(path))
            mod_doc = ast.get_docstring(tree, clean=False)
            if mod_doc:
                first_line = next((ln.strip() for ln in mod_doc.splitlines() if ln.strip()), "")
                assert not DOC_ECHO_RE.match(first_line), (
                    f"Filename-echo module docstring in {path}: {first_line!r}"
                )
