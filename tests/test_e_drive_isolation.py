# -*- coding: utf-8 -*-
"""
tests/test_e_drive_isolation.py
Kiểm thử cô lập 100% đường dẫn binary, biến môi trường cache và TEMP/TMP/TMPDIR
về thư mục dự án (ổ E:), cũng như tự dọn snapshot ở scripts/verify_offline.py.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import time

import scripts.verify_offline as verify_offline
from src.env_isolation import configure_e_drive_isolation


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_configure_e_drive_isolation_routes_caches_path_and_cross_drive_temp(tmp_path: Path):
    proj_root = tmp_path / "upscaler_proj"
    (proj_root / "bin" / "ffmpeg").mkdir(parents=True)
    (proj_root / ".venv" / "Scripts").mkdir(parents=True)

    # Giả lập TEMP trỏ sang ổ đĩa khác với proj_root
    cross_drive_tmp = r"Z:\OtherDrive\Temp" if proj_root.drive.upper() == "C:" else r"C:\Users\Admin\AppData\Local\Temp"
    fake_env: dict[str, str] = {
        "Temp": cross_drive_tmp,
        "tmp": cross_drive_tmp,
        "PATH": r"C:\Windows\System32;C:\OldFFmpeg\bin",
    }

    resolved = configure_e_drive_isolation(
        project_root=proj_root,
        environ=fake_env,
        create_dirs=True,
    )

    expected_cache = proj_root / ".cache"
    expected_tmp = str(proj_root / "scratch" / "tmp")
    assert resolved["HF_HOME"] == str(expected_cache / "huggingface")
    assert resolved["HUGGINGFACE_HUB_CACHE"] == str(expected_cache / "huggingface" / "hub")
    assert resolved["TORCH_HOME"] == str(expected_cache / "torch")
    assert resolved["XDG_CACHE_HOME"] == str(expected_cache)
    assert resolved["PIP_CACHE_DIR"] == str(expected_cache / "pip")
    assert resolved["MPLCONFIGDIR"] == str(expected_cache / "matplotlib")
    assert resolved["NUMBA_CACHE_DIR"] == str(expected_cache / "numba")
    assert resolved["RUFF_CACHE_DIR"] == str(expected_cache / "ruff_cache")

    assert "Temp" not in fake_env and "tmp" not in fake_env
    assert fake_env["TEMP"] == expected_tmp
    assert fake_env["TMP"] == expected_tmp
    assert fake_env["TMPDIR"] == expected_tmp
    assert Path(expected_tmp).is_dir()

    path_parts = fake_env["PATH"].split(os.pathsep)
    assert path_parts[0] == str(proj_root / "bin" / "ffmpeg")
    assert path_parts[1] == str(proj_root / ".venv" / "Scripts")
    # Gọi lần 2 phải idempotent, không lặp phần tử PATH
    configure_e_drive_isolation(project_root=proj_root, environ=fake_env, create_dirs=True)
    assert fake_env["PATH"].split(os.pathsep) == path_parts


def test_configure_e_drive_isolation_preserves_same_drive_temp_and_pytest_env(tmp_path: Path):
    proj_root = tmp_path / "proj"
    proj_root.mkdir()
    same_drive_tmp = tmp_path / "same_drive_tmp"
    same_drive_tmp.mkdir()

    # 1. Cùng ổ đĩa -> giữ nguyên
    env_same = {"TEMP": str(same_drive_tmp), "TMP": str(same_drive_tmp)}
    configure_e_drive_isolation(project_root=proj_root, environ=env_same, create_dirs=False)
    assert env_same["TEMP"] == str(same_drive_tmp)

    # 2. Đang trong pytest (PYTEST_CURRENT_TEST) -> giữ nguyên dù khác ổ đĩa
    env_pytest = {
        "PYTEST_CURRENT_TEST": "tests/test_dummy.py::test_fn (call)",
        "TEMP": r"C:\Temp",
        "TMP": r"C:\Temp",
    }
    configure_e_drive_isolation(project_root=proj_root, environ=env_pytest, create_dirs=False)
    assert env_pytest["TEMP"] == r"C:\Temp"


def test_configure_e_drive_isolation_swallows_oserror_on_readonly_fs(tmp_path: Path, monkeypatch):
    def deny_mkdir(self, *args, **kwargs):
        raise PermissionError("Read-only filesystem")

    monkeypatch.setattr(Path, "mkdir", deny_mkdir)
    fake_env: dict[str, str] = {}
    resolved = configure_e_drive_isolation(project_root=tmp_path, environ=fake_env, create_dirs=True)
    assert resolved["PIP_CACHE_DIR"].endswith("pip")
    assert resolved["TEMP"] == str(tmp_path / "scratch" / "tmp")


def test_windows_batch_launchers_use_crlf_and_export_e_drive_envs():
    bat_names = ("upscale.bat", "upscale_turbo.bat", "subtrans.bat", "web_gui.bat")
    for name in bat_names:
        bat_path = REPO_ROOT / name
        assert bat_path.is_file(), f"Missing {name}"
        raw = bat_path.read_bytes()
        assert b"\r\n" in raw, f"{name} must use Windows CRLF line endings"
        bare_lf = raw.replace(b"\r\n", b"")
        assert b"\n" not in bare_lf, f"{name} contains non-CRLF bare LF line endings"

        text = raw.decode("utf-8")
        assert r"bin\ffmpeg" in text
        assert r"TEMP=%SCRIPT_DIR%scratch\tmp" in text
        assert r"HF_HOME=%SCRIPT_DIR%.cache\huggingface" in text
        assert r"PIP_CACHE_DIR=%SCRIPT_DIR%.cache\pip" in text


def test_verify_offline_main_cleans_snapshot_on_pass_and_prunes_old_runs(tmp_path: Path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True)

    # Tạo sẵn 2 run cũ để kiểm tra cắt tỉa giữ tối đa 2 thư mục gần nhất sau pass
    old1 = scratch / "agent-validation-old1"
    old2 = scratch / "agent-validation-old2"
    old1.mkdir()
    old2.mkdir()
    os.utime(old1, (time.time() - 200, time.time() - 200))
    os.utime(old2, (time.time() - 100, time.time() - 100))

    class FakePopen:
        def __init__(self, command, cwd=None, env=None, **kwargs):
            self.cwd = Path(cwd)
            self.env = dict(env)
            # Giả lập pytest tạo cả file read-only (như .git/objects) trong basetemp
            pytest_sub = Path(self.env["TEMP"]) / "pytest" / ".git" / "objects"
            pytest_sub.mkdir(parents=True, exist_ok=True)
            ro_file = pytest_sub / "readonly.obj"
            ro_file.write_text("temp", encoding="utf-8")
            os.chmod(ro_file, stat.S_IREAD)
            self.stdout = iter(["10 passed in 0.10s\n"])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def wait(self):
            return 0

    def fake_snapshot(root, dest):
        (dest / "dummy.py").write_text("x = 1\n", encoding="utf-8")
        return 1

    monkeypatch.setattr(verify_offline, "ROOT", tmp_path)
    monkeypatch.setattr(verify_offline, "snapshot_sources", fake_snapshot)
    monkeypatch.setattr(verify_offline.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(sys, "argv", ["scripts/verify_offline.py"])

    rc = verify_offline.main()
    assert rc == 0

    remaining = sorted(scratch.glob("agent-validation-*"), key=lambda p: p.stat().st_mtime_ns)
    assert len(remaining) == 2
    assert not old1.exists()
    newest = remaining[-1]
    assert (newest / "pytest.log").is_file()
    assert not (newest / "repo").exists()
    assert (newest / "tmp").is_dir()
    assert not (newest / "tmp" / "pytest").exists()
