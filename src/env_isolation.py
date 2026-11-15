# -*- coding: utf-8 -*-
"""
src/env_isolation.py
Cô lập 100% đường dẫn binary (FFmpeg/Python), thư mục tạm (TEMP/TMP/TMPDIR)
và toàn bộ biến môi trường cache về thư mục dự án trên ổ E: (`E:\\ai-video-upscaler`),
tránh mọi phụ thuộc hoặc rò rỉ dữ liệu tạm sang ổ hệ thống C:.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Mapping, MutableMapping


def _get_env_ci(environ: Mapping[str, str], key: str) -> str | None:
    """Tra cứu biến môi trường không phân biệt hoa/thường."""
    if key in environ:
        return environ[key]
    target = key.upper()
    for k, v in environ.items():
        if k.upper() == target:
            return v
    return None


def _set_env_ci(environ: MutableMapping[str, str], key: str, value: str) -> None:
    """Gán biến môi trường chuẩn chữ hoa, loại bỏ khóa lệch hoa/thường cũ (nếu là dict thường)."""
    canonical = key.upper()
    if type(environ) is dict:
        for existing in list(environ.keys()):
            if existing.upper() == canonical and existing != canonical:
                environ.pop(existing, None)
    environ[canonical] = value


def _norm_dir(path_str: str) -> str:
    return os.path.normcase(os.path.normpath(path_str))


def configure_e_drive_isolation(
    project_root: str | os.PathLike[str] | None = None,
    environ: MutableMapping[str, str] | None = None,
    create_dirs: bool = True,
) -> dict[str, str]:
    """
    Định tuyến PATH, cache AI/Python/pip/Ruff và thư mục TEMP/TMP/TMPDIR
    về nội bộ `project_root` (mặc định là thư mục gốc repo trên ổ E:).

    - Không ghi đè TEMP/TMP/TMPDIR nếu đang chạy trong pytest (`PYTEST_CURRENT_TEST`)
      hoặc nếu TEMP hiện tại đã nằm cùng ổ đĩa với `project_root`.
    - Luôn giữ `OSError` an toàn trên môi trường sandbox/chỉ đọc.
    """
    base_root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    target_env: MutableMapping[str, str] = os.environ if environ is None else environ

    # 1. Ưu tiên tuyệt đối bin/ffmpeg và .venv/Scripts cục bộ trên PATH
    priority_dirs: list[str] = []
    for candidate in (base_root / "bin" / "ffmpeg", base_root / ".venv" / "Scripts"):
        if candidate.is_dir():
            priority_dirs.append(str(candidate))

    if priority_dirs:
        existing_path = _get_env_ci(target_env, "PATH") or ""
        priority_norms = {_norm_dir(p) for p in priority_dirs}
        tail_parts = [
            part for part in existing_path.split(os.pathsep)
            if part and _norm_dir(part) not in priority_norms
        ]
        _set_env_ci(target_env, "PATH", os.pathsep.join(priority_dirs + tail_parts))

    # 2. Định tuyến toàn bộ cache AI / Python / package / linter vào project_root/.cache
    cache_base = base_root / ".cache"
    cache_defaults = {
        "HF_HOME": str(cache_base / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(cache_base / "huggingface" / "hub"),
        "TORCH_HOME": str(cache_base / "torch"),
        "XDG_CACHE_HOME": str(cache_base),
        "PIP_CACHE_DIR": str(cache_base / "pip"),
        "MPLCONFIGDIR": str(cache_base / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache_base / "numba"),
        "RUFF_CACHE_DIR": str(cache_base / "ruff_cache"),
    }

    resolved: dict[str, str] = {}
    dirs_to_create: list[Path] = []
    for env_key, default_dir in cache_defaults.items():
        current_val = _get_env_ci(target_env, env_key)
        if not current_val:
            _set_env_ci(target_env, env_key, default_dir)
            chosen = default_dir
        else:
            chosen = current_val
        resolved[env_key] = chosen
        if chosen:
            dirs_to_create.append(Path(chosen))

    # 3. Định tuyến TEMP/TMP/TMPDIR về scratch/tmp nếu chưa trỏ cùng ổ đĩa (trừ khi trong pytest)
    default_tmp = str(base_root / "scratch" / "tmp")
    in_pytest = bool(_get_env_ci(target_env, "PYTEST_CURRENT_TEST"))
    current_temp = _get_env_ci(target_env, "TEMP") or _get_env_ci(target_env, "TMP")
    should_redirect_tmp = False
    if not in_pytest:
        if not current_temp:
            should_redirect_tmp = True
        elif base_root.drive:
            temp_drive = Path(current_temp).drive
            if temp_drive.upper() != base_root.drive.upper():
                should_redirect_tmp = True

    if should_redirect_tmp:
        for tmp_key in ("TEMP", "TMP", "TMPDIR"):
            _set_env_ci(target_env, tmp_key, default_tmp)
        if environ is None:
            # Xóa cache nội bộ của module tempfile để tempfile.gettempdir() nhận ngay đường dẫn mới
            tempfile.tempdir = None
        chosen_tmp = default_tmp
    else:
        chosen_tmp = current_temp or default_tmp
        for tmp_key in ("TEMP", "TMP", "TMPDIR"):
            if not _get_env_ci(target_env, tmp_key):
                _set_env_ci(target_env, tmp_key, chosen_tmp)

    resolved["TEMP"] = chosen_tmp
    resolved["TMP"] = chosen_tmp
    resolved["TMPDIR"] = _get_env_ci(target_env, "TMPDIR") or chosen_tmp
    if priority_dirs:
        resolved["PATH"] = _get_env_ci(target_env, "PATH") or ""
    if chosen_tmp:
        dirs_to_create.append(Path(chosen_tmp))

    if create_dirs:
        for d in dirs_to_create:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    return resolved
