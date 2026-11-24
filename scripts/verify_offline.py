"""Run repository tests in a disposable source snapshot, without runtime imports.

Uses the invoking interpreter. Keeps logs under scratch/agent-validation-*.
This is isolation for the reviewed test suite, not a sandbox for untrusted code.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {"src", "subtrans", "tests", "scripts", "models", "glossaries", "docs", "web"}
SAFE_ROOT_DOTFILES = {".antigravityignore", ".gitignore", ".editorconfig", ".env.example"}
ROOT_FILES = {
    "upscale.py", "upscale.bat", "upscale_turbo.bat", "subtrans.py", "subtrans.bat",
    "web_gui.py", "web_gui.bat", "pytest.ini", "ruff.toml", "requirements.txt",
    "shows.yaml", "GEMINI.md", "README.md",
}
SUFFIXES = {
    ".py", ".bat", ".yaml", ".yml", ".toml", ".ini", ".md", ".txt", ".json",
    ".ass", ".srt", ".html", ".css", ".js", ".svg",
}
GUARD = '''"""Offline guard loaded before test collection and inherited by child Python processes."""
import _socket
import os
import socket
import sys

if sys.platform == "win32" and not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        import ctypes
        from ctypes import wintypes

        _k32 = ctypes.windll.kernel32
        _k32.GetCurrentProcess.restype = wintypes.HANDLE
        _k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        _k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        _k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]

        class _PwrState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        _hp = _k32.GetCurrentProcess()
        _st = _PwrState(Version=1, ControlMask=0x1, StateMask=0)
        _k32.SetProcessInformation(_hp, 4, ctypes.byref(_st), ctypes.sizeof(_st))
        if int(_k32.GetPriorityClass(_hp)) in (0, 0x00000020, 0x00004000, 0x00000040):
            _k32.SetPriorityClass(_hp, 0x00008000)
    except Exception:
        pass

def deny_network(*args, **kwargs):
    if hasattr(socket, "socketpair") and sys._getframe(1).f_code is socket.socketpair.__code__:
        return _socket.socket.connect(*args, **kwargs)
    raise RuntimeError("Network disabled by verify_offline.py; use a stub backend")

socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.socket.sendto = deny_network
socket.create_connection = deny_network
socket.getaddrinfo = deny_network
'''


def allowed_source(relative: Path) -> bool:
    """Only source/config/fixtures; never copy production state or credentials."""
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return False
    if len(relative.parts) == 1 and relative.name in SAFE_ROOT_DOTFILES:
        return True
    if relative.parts[0] == ".agents":
        return (
            not any(part in {"__pycache__", "out", "scratch"} for part in relative.parts)
            and relative.suffix.lower() in SUFFIXES | {".svg"}
        )
    if any(part.startswith(".") or part in {"__pycache__", "out", "scratch", "models_bench"}
           for part in relative.parts):
        return False
    if len(relative.parts) == 1:
        return relative.name in ROOT_FILES
    return relative.parts[0] in SOURCE_DIRS and relative.suffix.lower() in SUFFIXES


def isolated_environment(
    environ: dict[str, str],
    tmp_dir: Path | str | None = None,
) -> dict[str, str]:
    clean = {key: value for key, value in environ.items()
             if not any(token in key.upper() for token in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
             and key.upper() not in {"GOOGLE_APPLICATION_CREDENTIALS", "PYTHONPATH",
                                     "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
                                     "TEMP", "TMP", "TMPDIR"}}
    resolved_tmp = str(Path(tmp_dir) if tmp_dir is not None else (ROOT / "scratch" / "tmp"))
    clean.update(CUDA_VISIBLE_DEVICES="-1", RUN_GPU_TESTS="0", PYTHON_DOTENV_DISABLED="1",
                 SUBTRANS_DISABLE_DOTENV="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                 PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", UPSCALER_HEVC_ENCODER="libx265",
                 OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                 OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                 TEMP=resolved_tmp, TMP=resolved_tmp, TMPDIR=resolved_tmp)
    return clean


def snapshot_sources(root: Path, destination: Path) -> int:
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached",
                             "--others", "--exclude-standard"], check=True, capture_output=True)
    paths = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    count = 0
    for name in paths:
        relative = Path(name)
        if not allowed_source(relative):
            continue
        source = root / relative
        if not source.exists():  # tracked deletion in working tree
            continue
        # Refuse links/junctions escaping the source tree.
        if source.is_symlink() or not source.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Refusing linked source: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        count += 1
    return count


def _rmtree_force(path: Path) -> None:
    """Force-remove a temporary directory including read-only files (e.g., .git/objects on Windows)."""
    if not path.exists():
        return

    def _clear_readonly(func, target, _exc_info):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_clear_readonly)


def _is_pid_alive(pid: int) -> bool:
    """Return True if a positive OS process ID is currently running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        STILL_ACTIVE = 259
        WAIT_OBJECT_0 = 0x00000000
        WAIT_TIMEOUT = 0x00000102
        ERROR_ACCESS_DENIED = 5

        # Fresh WinDLL + explicit HANDLE restype avoids global windll mutation and 64-bit handle truncation.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
        if not handle:
            if ctypes.get_last_error() == ERROR_ACCESS_DENIED:
                handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if not handle:
                    return ctypes.get_last_error() == ERROR_ACCESS_DENIED
                try:
                    exit_code = wintypes.DWORD()
                    return bool(
                        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                        and exit_code.value == STILL_ACTIVE
                    )
                finally:
                    kernel32.CloseHandle(handle)
            return False
        try:
            # WaitForSingleObject distinguishes still-running processes from exited zombie handles.
            wait_res = kernel32.WaitForSingleObject(handle, 0)
            if wait_res == WAIT_TIMEOUT:
                return True
            if wait_res == WAIT_OBJECT_0:
                return False
            exit_code = wintypes.DWORD()
            return bool(
                kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                and exit_code.value == STILL_ACTIVE
            )
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _write_active_pid(pid_file: Path, pid: int) -> None:
    """Atomically write the active-run PID marker inside a validation directory."""
    tmp_file = pid_file.with_name(f"{pid_file.name}.tmp")
    tmp_file.write_text(f"{pid}\n", encoding="utf-8")
    os.replace(tmp_file, pid_file)


def _is_active_run_dir(run_dir: Path) -> bool:
    """Return True if run_dir holds a live .active.pid marker (or is locked mid-write)."""
    pid_file = run_dir / ".active.pid"
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    if not raw:
        return False
    try:
        pid = int(raw.splitlines()[0].strip())
    except ValueError:
        return False
    if pid == os.getpid():
        return True
    return _is_pid_alive(pid)


def _dir_sort_key(path: Path) -> tuple[int, str]:
    """Return (st_mtime_ns, name) without crashing if a concurrent runner prunes path."""
    try:
        return (path.stat().st_mtime_ns, path.name)
    except OSError:
        return (-1, path.name)


def _detect_dirty_paths(root: Path) -> list[str]:
    """Return modified and untracked repository-relative POSIX paths."""
    paths: set[str] = set()
    diff_res = subprocess.run(
        ["git", "--no-optional-locks", "diff", "--name-only", "HEAD"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if diff_res.returncode == 0:
        for ln in diff_res.stdout.splitlines():
            p = ln.strip().replace("\\", "/")
            if p:
                paths.add(p)
    st_res = subprocess.run(
        ["git", "--no-optional-locks", "status", "--porcelain"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if st_res.returncode == 0:
        for raw_line in st_res.stdout.splitlines():
            if len(raw_line) < 4:
                continue
            status = raw_line[:2]
            if "D" in status:
                continue
            rel = raw_line[3:].strip()
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1].strip()
            rel = rel.strip('"').replace("\\", "/")
            if rel:
                paths.add(rel)
    return sorted(paths)


def resolve_impacted_tests(root: Path, changed_paths: list[str] | None = None) -> list[str]:
    """Resolve impacted test files/directories from modified paths via agent_context 2-hop + contracts."""
    dirty = changed_paths if changed_paths is not None else _detect_dirty_paths(root)
    if not dirty:
        return []
    root_str = str(root)
    added_sys_path = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added_sys_path = True
    try:
        from scripts.agent_context import compute_impact

        _, impacted = compute_impact(root, dirty)
    finally:
        if added_sys_path and sys.path and sys.path[0] == root_str:
            sys.path.pop(0)

    candidates: set[str] = set()
    for item in impacted:
        norm_item = item.replace("\\", "/")
        if norm_item.endswith("/") or (root / norm_item).is_dir():
            candidates.add(norm_item)
        elif norm_item.endswith(".py"):
            if Path(norm_item).name.startswith("test_"):
                candidates.add(norm_item)
            else:
                parent_dir = Path(norm_item).parent.as_posix()
                if parent_dir and parent_dir != "." and (root / parent_dir).is_dir():
                    candidates.add(parent_dir.rstrip("/") + "/")
    for rel in dirty:
        norm = rel.replace("\\", "/")
        if (norm.startswith("tests/") or norm.startswith("subtrans/tests/")) and norm.endswith(".py"):
            if Path(norm).name.startswith("test_") and (root / norm).is_file():
                candidates.add(norm)
            elif (root / norm).is_file():
                parent_dir = Path(norm).parent.as_posix()
                if parent_dir and parent_dir != "." and (root / parent_dir).is_dir():
                    candidates.add(parent_dir.rstrip("/") + "/")

    # Deduplicate files already covered by an impacted directory (e.g. subtrans/tests/)
    dirs = [c.rstrip("/") + "/" for c in candidates if c.endswith("/") or (root / c).is_dir()]
    resolved: list[str] = []
    for cand in sorted(candidates):
        norm = cand.rstrip("/") + "/" if (root / cand).is_dir() else cand
        if any(norm != d and norm.startswith(d) for d in dirs):
            continue
        if (root / cand).exists():
            resolved.append(cand)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--target",
        nargs="+",
        metavar="PATH_OR_NODEID",
        help="Run only specified test file(s) or pytest node ID(s) inside the isolated snapshot",
    )
    mode_group.add_argument(
        "--impacted",
        action="store_true",
        help="Run only tests impacted by current working-tree modifications/untracked files",
    )
    parser.add_argument(
        "--maxfail",
        type=int,
        default=None,
        help="Stop after N test failures (defaults to 3 in --impacted mode)",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Enable pytest failure stdout/stderr capture (overrides default --show-capture=no)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel isolated snapshot workers for full-suite runs (default: 6 P-cores; 1 disables sharding)",
    )
    return parser


_MEASURED_FILE_WEIGHT_MS: dict[str, int] = {
    "tests/e2e/test_tier1_features.py": 17600,
    "tests/e2e/test_e2e_group_b.py": 15820,
    "tests/test_web_gui.py": 13510,
    "tests/e2e/test_tier2_boundaries.py": 9670,
    "tests/test_e2e_acceptance_p0_p2.py": 7280,
    "tests/test_export_review_bundle.py": 7230,
    "tests/test_rtx3060_unlocks.py": 6470,
    "tests/e2e/test_tier3_pairwise.py": 6130,
    "tests/test_nvme_cache_and_input_triton.py": 5560,
    "tests/test_dedup_policy.py": 4840,
    "tests/test_nextgen_cugan_grain_and_cadence.py": 4800,
    "tests/e2e/test_tier4_scenarios.py": 4620,
    "tests/test_zero_disk_and_trt_export.py": 4270,
    "tests/e2e/test_e2e_integration_t3_t4.py": 4170,
    "tests/test_algorithmic_optimizations.py": 3980,
    "tests/test_mux_audit_regression.py": 3860,
    "tests/test_benchmark_infra.py": 3680,
    "tests/test_agent_context.py": 2950,
    "subtrans/tests/test_throttle.py": 2830,
    "tests/test_output_sanity.py": 2820,
    "subtrans/tests/test_cache.py": 2720,
    "tests/test_pareto_o1_o6.py": 2720,
    "subtrans/tests/test_audit_fixes_subtrans.py": 2660,
    "subtrans/tests/test_dtw_and_dynamic_cast.py": 1990,
    "tests/test_architecture_boundaries.py": 1960,
    "tests/test_pipeline.py": 1940,
    "tests/test_allow_partial.py": 1720,
    "tests/test_verify_offline.py": 1680,
    "tests/test_model_spandrel_and_whisper.py": 1560,
    "subtrans/tests/test_subtrans_resilience.py": 1560,
    "tests/test_agent_setup.py": 1500,
    "subtrans/tests/test_subtrans_deep_upgrades.py": 1410,
    "tests/test_smoke_pipeline.py": 980,
    "subtrans/tests/test_pipeline_dynamic_batch.py": 920,
    "tests/test_audit_fixes_video.py": 860,
    "tests/test_cli_lock_and_pipeline_regressions.py": 820,
    "tests/e2e/test_e2e_group_a.py": 800,
    "subtrans/tests/test_subtrans_v2_max_quality.py": 750,
    "tests/test_ffmpeg_ops.py": 630,
    "tests/test_gop_rle_and_3stage_overlap.py": 630,
    "tests/test_nvenc_config_and_vram.py": 520,
    "subtrans/tests/test_cli_batch.py": 520,
    "subtrans/tests/test_subtrans_v2_5_ceiling_polish.py": 490,
    "subtrans/tests/test_rag.py": 440,
    "subtrans/tests/test_parser.py": 430,
    "subtrans/tests/test_eval_runner.py": 420,
    "subtrans/tests/test_cascade.py": 390,
    "subtrans/tests/test_key_rotation_stub.py": 340,
    "subtrans/tests/test_subtrans_stub_pipeline.py": 280,
    "subtrans/tests/test_gemini.py": 250,
    "tests/test_model_selection_and_safety.py": 200,
    "tests/test_deadline_math_realignment.py": 180,
    "subtrans/tests/test_fts5_kanji.py": 170,
    "subtrans/tests/test_whisper_noise_and_speaker.py": 140,
    "tests/test_dedup.py": 110,
    "subtrans/tests/test_gemini_quota_management.py": 110,
    "tests/test_enospc_backpressure_retry.py": 100,
    "subtrans/tests/test_glossary.py": 90,
    "subtrans/tests/test_sample_fixture.py": 70,
    "tests/test_cli_and_shows_polish.py": 70,
    "subtrans/tests/test_gemini_bisect.py": 70,
    "subtrans/tests/test_quota_wrapper.py": 70,
    "subtrans/tests/test_ja_morph_tagger.py": 60,
    "subtrans/tests/test_harness.py": 50,
    "subtrans/tests/test_writer.py": 40,
    "tests/test_cleanup.py": 40,
    "tests/test_zero_disk_ring_buffers_and_cel.py": 30,
    "subtrans/tests/test_linter.py": 30,
    "tests/test_e_drive_isolation.py": 30,
    "tests/test_bd_gop_desync.py": 30,
    "tests/test_subtitle_guardrails.py": 20,
    "tests/test_ramdisk_worst_case.py": 20,
    "subtrans/tests/test_auto_offset.py": 20,
    "tests/test_set_track_defaults.py": 20,
    "tests/test_triton_p010le_kernel.py": 20,
    "tests/test_model_regression.py": 20,
    "tests/test_gpu_throttling_overlap.py": 20,
}


def _split_e2e_into_n(
    file_path: Path,
    rel: str,
    base_weight: int,
    parts: int,
    first_ratio: float | None = None,
    slice_weights: tuple[int, ...] | None = None,
) -> list[tuple[int, list[str]]]:
    """Split pure-function E2E tier files into N balanced nodeid slices."""
    import ast

    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except Exception:
        return [(base_weight, [rel])]
    has_test_classes = any(isinstance(n, ast.ClassDef) and n.name.startswith("Test") for n in tree.body)
    test_fns = [
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
    ]
    if has_test_classes or len(test_fns) < parts * 2:
        return [(base_weight, [rel])]
    use_slice_weights = slice_weights is not None and len(slice_weights) == parts
    if parts == 2 and first_ratio is not None:
        mid = max(1, min(len(test_fns) - 1, int(round(len(test_fns) * first_ratio))))
        w_first = max(1, slice_weights[0] if use_slice_weights else (base_weight // 2))
        w_second = max(1, slice_weights[1] if use_slice_weights else (base_weight // 2))
        return [
            (w_first, [f"{rel}::{fn}" for fn in test_fns[:mid]]),
            (w_second, [f"{rel}::{fn}" for fn in test_fns[mid:]]),
        ]
    slices: list[tuple[int, list[str]]] = []
    n_total = len(test_fns)
    for idx in range(parts):
        lo = (idx * n_total) // parts
        hi = ((idx + 1) * n_total) // parts
        fn_slice = test_fns[lo:hi]
        if fn_slice:
            w_part = max(1, slice_weights[idx] if use_slice_weights else (base_weight // parts))
            slices.append((w_part, [f"{rel}::{fn}" for fn in fn_slice]))
    return slices


def _lpt_pack(units: list[tuple[int, list[str]]], k: int) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(k)]
    loads = [0] * k
    for weight, targets in sorted(units, key=lambda item: (-item[0], item[1][0])):
        idx = min(range(k), key=lambda i: loads[i])
        buckets[idx].extend(targets)
        loads[idx] += weight
    return [sorted(b) for b in buckets if b]


_SPLIT_MONOLITH_PARTS: dict[str, int] = {
    "tests/e2e/test_tier1_features.py": 3,
    "tests/e2e/test_tier2_boundaries.py": 3,
    "tests/e2e/test_e2e_group_b.py": 3,
    "tests/test_web_gui.py": 3,
    "tests/e2e/test_tier3_pairwise.py": 2,
    "tests/e2e/test_tier4_scenarios.py": 2,
}

_SPLIT_MONOLITH_SLICE_WEIGHTS_MS: dict[str, tuple[int, ...]] = {
    "tests/e2e/test_tier1_features.py": (4120, 8080, 5400),
    "tests/e2e/test_tier2_boundaries.py": (3330, 3180, 3160),
    "tests/e2e/test_e2e_group_b.py": (13860, 210, 1750),
    "tests/test_web_gui.py": (3510, 5390, 4610),
    "tests/e2e/test_tier3_pairwise.py": (3490, 2650),
    "tests/e2e/test_tier4_scenarios.py": (1220, 3400),
}


def _partition_test_shards(snapshot: Path, num_workers: int) -> list[list[str]]:
    """Split heavy pure-function test monoliths and LPT-pack all units evenly across P-core shards."""
    all_units: list[tuple[int, list[str]]] = []
    file_count = 0
    for base_rel in ("tests", "subtrans/tests"):
        base = snapshot / base_rel
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("test_*.py")):
            file_count += 1
            rel = p.relative_to(snapshot).as_posix()
            weight = _MEASURED_FILE_WEIGHT_MS.get(rel, max(200, p.stat().st_size // 20))
            parts = _SPLIT_MONOLITH_PARTS.get(rel, 1)
            if parts > 1:
                slice_weights = _SPLIT_MONOLITH_SLICE_WEIGHTS_MS.get(rel)
                all_units.extend(_split_e2e_into_n(p, rel, weight, parts=parts, slice_weights=slice_weights))
            else:
                all_units.append((weight, [rel]))
    if file_count < 8 or num_workers <= 1:
        return []
    return _lpt_pack(all_units, min(num_workers, len(all_units)))


def _merge_junit_files(shard_xmls: list[Path], dest_xml: Path) -> tuple[int, int, int, int]:
    """Combine per-shard JUnit XML files into a single root <testsuites> report."""
    root = ET.Element("testsuites")
    total_tests = total_failures = total_errors = total_skipped = 0
    for xml_path in shard_xmls:
        if not xml_path.is_file():
            continue
        suite_root = ET.parse(xml_path).getroot()
        suites = [suite_root] if suite_root.tag == "testsuite" else list(suite_root.findall("testsuite"))
        for st in suites:
            root.append(st)
            total_tests += int(st.attrib.get("tests", 0))
            total_failures += int(st.attrib.get("failures", 0))
            total_errors += int(st.attrib.get("errors", 0))
            total_skipped += int(st.attrib.get("skipped", 0))
    ET.ElementTree(root).write(dest_xml, encoding="utf-8", xml_declaration=True)
    return total_tests, total_failures, total_errors, total_skipped


def _configure_win32_pcore_qos() -> int | None:
    """Disable Win11 EcoQoS throttling so sharded pytest workers stay on P-cores."""
    if sys.platform != "win32" or os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        h_proc = kernel32.GetCurrentProcess()

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        state = _PowerThrottlingState(Version=1, ControlMask=0x1, StateMask=0)
        kernel32.SetProcessInformation(h_proc, 4, ctypes.byref(state), ctypes.sizeof(state))
        prev_prio = int(kernel32.GetPriorityClass(h_proc))
        if prev_prio in (0, 0x00000020, 0x00004000, 0x00000040):
            kernel32.SetPriorityClass(h_proc, 0x00008000)
            return prev_prio or 0x00000020
    except Exception:
        pass
    return None


def _restore_win32_pcore_qos(prev_prio: int | None) -> None:
    """Restore prior Windows process priority class after sharded validation completes."""
    if prev_prio is None or sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), prev_prio)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    is_fast_mode = bool(args.target or args.impacted)
    test_targets: list[str] = []
    if args.target:
        test_targets = list(args.target)
    elif args.impacted:
        dirty = _detect_dirty_paths(ROOT)
        if not dirty:
            print("No modified or untracked files detected; 0 impacted tests.")
            return 0
        test_targets = resolve_impacted_tests(ROOT, dirty)
        if not test_targets:
            print("No impacted tests matched modified files; skipping pytest.")
            return 0
        print(f"Impacted targets ({len(test_targets)}): {', '.join(test_targets)}")

    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="agent-validation-", dir=scratch))
    active_pid_file = run_dir / ".active.pid"
    snapshot = run_dir / "repo"
    tmp_dir = run_dir / "tmp"

    code = 1
    extra_snapshots: list[Path] = []
    prev_prio = _configure_win32_pcore_qos()
    try:
        _write_active_pid(active_pid_file, os.getpid())
        snapshot.mkdir()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        count = snapshot_sources(ROOT, snapshot)
        guard_dir = run_dir / "guard"
        guard_dir.mkdir(parents=True, exist_ok=True)
        (guard_dir / "sitecustomize.py").write_text(GUARD, encoding="utf-8")
        (snapshot / "agent_offline_guard.py").write_text(GUARD, encoding="utf-8")
        base_env = isolated_environment(dict(os.environ), tmp_dir=tmp_dir)
        ffmpeg_bin_dir = ROOT / "bin" / "ffmpeg"
        if ffmpeg_bin_dir.is_dir():
            cur_path = base_env.get("PATH", "")
            if str(ffmpeg_bin_dir) not in cur_path.split(os.pathsep):
                base_env["PATH"] = f"{ffmpeg_bin_dir}{os.pathsep}{cur_path}" if cur_path else str(ffmpeg_bin_dir)
        base_env["TEMP"] = base_env["TMP"] = base_env["TMPDIR"] = str(tmp_dir)
        # Propagate socket guard across child Python subprocesses without exposing live repo paths.
        base_env["PYTHONPATH"] = str(guard_dir)
        # Pytest may inspect Git for tests of the context helper; snapshot has no parent repository.
        base_env["GIT_CEILING_DIRECTORIES"] = str(run_dir)

        base_flags = [sys.executable, "-m", "pytest", "-m", "not gpu", "-q", "--tb=short", "-p", "agent_offline_guard"]
        if not args.capture:
            base_flags.append("--show-capture=no")
        maxfail = args.maxfail if args.maxfail is not None else (3 if args.impacted else None)
        if maxfail is not None and maxfail > 0:
            base_flags.extend(["--maxfail", str(maxfail)])

        shards = _partition_test_shards(snapshot, int(args.workers)) if not test_targets else []

        print(
            f"Snapshot: {snapshot}\nSource files: {count}\n"
            "CPU only; Python sockets blocked; no production cache/weights copied.",
            flush=True,
        )
        if not shards:
            command = [
                *base_flags,
                "--basetemp", str(tmp_dir / "pytest"),
                "--junitxml", str(run_dir / "junit.xml"),
                *test_targets,
            ]
            with (run_dir / "pytest.log").open("w", encoding="utf-8") as log:
                with subprocess.Popen(
                    command, cwd=snapshot, env=base_env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace",
                ) as process:
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    code = process.wait()
        else:
            shard_dirs: list[Path] = []
            for idx, _s_targets in enumerate(shards):
                if idx == 0:
                    shard_dirs.append(snapshot)
                else:
                    w_snap = run_dir / f"repo_w{idx}"
                    shard_dirs.append(w_snap)
                    extra_snapshots.append(w_snap)

            def _link_py_or_copy(src_f: str, dst_f: str) -> None:
                if src_f.endswith(".py"):
                    try:
                        os.link(src_f, dst_f)
                        return
                    except OSError:
                        pass
                shutil.copyfile(src_f, dst_f)

            def _run_shard(idx_targets: tuple[int, list[str]]) -> tuple[int, int, str, Path]:
                idx, targets = idx_targets
                if shard_dirs[idx] != snapshot:
                    shutil.copytree(snapshot, shard_dirs[idx], copy_function=_link_py_or_copy)
                w_tmp = tmp_dir / f"w{idx}"
                w_tmp.mkdir(parents=True, exist_ok=True)
                w_env = dict(base_env)
                w_env["TEMP"] = w_env["TMP"] = w_env["TMPDIR"] = str(w_tmp)
                w_xml = run_dir / f"junit_w{idx}.xml"
                cmd = [*base_flags, "--basetemp", str(w_tmp / "pytest"), "--junitxml", str(w_xml), *targets]
                with subprocess.Popen(
                    cmd, cwd=shard_dirs[idx], env=w_env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                ) as proc:
                    assert proc.stdout is not None
                    out = proc.stdout.read()
                    rc = proc.wait()
                return idx, rc, out, w_xml

            with ThreadPoolExecutor(max_workers=len(shards)) as pool:
                shard_results = sorted(pool.map(_run_shard, enumerate(shards)), key=lambda r: r[0])

            shard_xmls: list[Path] = []
            codes: list[int] = []
            with (run_dir / "pytest.log").open("w", encoding="utf-8") as log:
                for idx, rc, out, w_xml in shard_results:
                    header = f"=== Shard {idx + 1}/{len(shards)} (exit={rc}) ===\n"
                    log.write(header + out + "\n")
                    if rc != 0:
                        print(header + out, end="", flush=True)
                    codes.append(rc)
                    shard_xmls.append(w_xml)
                tot, fails, errs, skips = _merge_junit_files(shard_xmls, run_dir / "junit.xml")
                passed = max(0, tot - fails - errs - skips)
                summary = f"Shards ({len(shards)} workers): {passed} passed, {skips} skipped, {fails} failed, {errs} errors\n"
                print(summary, end="", flush=True)
                log.write(summary)
            for w_xml in shard_xmls:
                w_xml.unlink(missing_ok=True)
            code = next((rc for rc in codes if rc != 0), 0)
        print(f"Evidence: {run_dir}\nPytest exit code: {code}")
    finally:
        try:
            cleanup_paths: list[Path] = []
            if code == 0:
                cleanup_paths.extend(extra_snapshots)
            if is_fast_mode or code == 0:
                cleanup_paths.extend([snapshot, run_dir / "guard", tmp_dir])
            if cleanup_paths:
                with ThreadPoolExecutor(max_workers=min(8, len(cleanup_paths))) as clean_pool:
                    list(clean_pool.map(_rmtree_force, cleanup_paths))
            if is_fast_mode or code == 0:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                if code == 0:
                    validation_dirs = sorted(
                        (p for p in scratch.glob("agent-validation-*") if p.is_dir()),
                        key=_dir_sort_key,
                        reverse=True,
                    )
                    inactive_dirs = [
                        d for d in validation_dirs
                        if d == run_dir or not _is_active_run_dir(d)
                    ]
                    for old_dir in inactive_dirs[2:]:
                        if old_dir != run_dir and not _is_active_run_dir(old_dir):
                            _rmtree_force(old_dir)
        finally:
            try:
                active_pid_file.unlink(missing_ok=True)
                active_pid_file.with_name(f"{active_pid_file.name}.tmp").unlink(missing_ok=True)
            except OSError:
                pass
            _restore_win32_pcore_qos(prev_prio)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
