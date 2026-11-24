"""FastAPI application and REST/SSE routes for Local Real-Time Web Studio."""

from __future__ import annotations

from contextlib import asynccontextmanager, suppress
from datetime import datetime
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence
import urllib.parse
import uuid
import yaml

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from subtrans.cache import compute_key_id, get_next_pt_reset, get_now_pt
from web.fs_browser import (
    AUTO_PARETO_MODEL_ID,
    AUTO_PARETO_MODEL_NAME,
    DEDUP_POLICY_PARAMS,
    SUBTITLE_EXTENSIONS,
    VALID_GRAIN_RESTORE_POLICIES,
    VALID_NVENC_MULTIPASS_POLICIES,
    VIDEO_EXTENSIONS,
    SafeLocalFSBrowser,
    extract_trt_hw_profiles,
    has_reparse_or_symlink_in_chain,
    inspect_model_trt_cache,
    is_secret_filename,
    is_unc_path,
    is_upscaled_output_stem,
    is_valid_dedup_policy,
    match_show_pattern,
    match_show_rule,
    natural_sort_key,
    resolve_allowed_trt_weights_shas,
    strip_wrapping_quotes,
    unwrap_outer_quotes_preserving_trailing_space,
    validate_win32_path_hygiene,
)
from web.job_manager import MODEL_REFERENCE_FPS, JobManager, _validate_output_dir, parse_preview_start_value
from web.sse_hub import TelemetryHub, _coerce_pid, check_active_gpu_lock, scan_temp_checkpoints

ALLOWED_OPEN_MEDIA_EXTS = VIDEO_EXTENSIONS | SUBTITLE_EXTENSIONS
STATIC_DIR = Path(__file__).resolve().parent / "static"


# PYDANTIC PAYLOAD SCHEMAS

_DEADLINE_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _validate_no_nul_str(val: Optional[str], field_name: str = "path", reject_unc: bool = True) -> Optional[str]:
    if val is not None:
        if "\x00" in val:
            raise ValueError(f"Trường '{field_name}' chứa ký tự NUL (\\x00) không hợp lệ.")
        if reject_unc and is_unc_path(val):
            raise ValueError(f"Trường '{field_name}' không cho phép đường dẫn mạng UNC (\\\\... hoặc //...).")
    return val


def _reject_bool_or_float_int(val: Any, field_name: str) -> Any:
    if isinstance(val, (bool, float)):
        raise ValueError(f"{field_name} phải là số nguyên hợp lệ (không nhận boolean/số thực).")
    return val


def _validate_deadline_str(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = val.strip()
    if not s:
        return None
    m = _DEADLINE_RE.match(s)
    if not m:
        raise ValueError(f"Định dạng deadline '{val}' không hợp lệ (cần HH:MM, ví dụ 07:00).")
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Giờ deadline '{val}' nằm ngoài phạm vi 00:00..23:59.")
    return f"{hh:02d}:{mm:02d}"


class SchedulePreviewRequest(BaseModel):
    inputs: List[str] = Field(default_factory=list)
    deadline: Optional[str] = None
    allow_partial: bool = False
    model: Optional[str] = None
    dedup_policy: Optional[str] = None
    nvenc_multipass: Optional[str] = None
    grain_restore: Optional[str] = None
    batch_size: int = Field(default=8, ge=1, le=32)
    zero_disk: bool = True
    use_tensorrt: bool = True
    subs_after_upscale: bool = True
    output_dir: Optional[str] = None
    force: bool = False

    @field_validator("batch_size", mode="before")
    @classmethod
    def _validate_batch_size_type(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return _reject_bool_or_float_int(v, "batch_size")

    @field_validator("inputs")
    @classmethod
    def _validate_inputs_no_nul(cls, v: List[str]) -> List[str]:
        for item in v:
            _validate_no_nul_str(item, "inputs")
        return v

    @field_validator("output_dir")
    @classmethod
    def _validate_outdir_no_nul(cls, v: Optional[str]) -> Optional[str]:
        return _validate_no_nul_str(v, "output_dir")

    @field_validator("nvenc_multipass")
    @classmethod
    def _validate_mp(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not str(v).strip():
            return None
        norm = str(v).strip().lower()
        if norm not in VALID_NVENC_MULTIPASS_POLICIES:
            raise ValueError("nvenc_multipass phải là 'auto', 'fullres' hoặc 'qres'.")
        return norm

    @field_validator("grain_restore")
    @classmethod
    def _validate_gr(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not str(v).strip():
            return None
        norm = str(v).strip().lower()
        if norm not in VALID_GRAIN_RESTORE_POLICIES:
            raise ValueError("grain_restore phải là 'auto', 'off', 'light' hoặc 'heavy'.")
        return norm


class StartUpscaleRequest(BaseModel):
    inputs: List[str] = Field(default_factory=list)
    output_dir: Optional[str] = None
    model: Optional[str] = None
    dedup_policy: Optional[str] = None
    nvenc_multipass: Optional[str] = None
    grain_restore: Optional[str] = None
    batch_size: Optional[int] = Field(default=None, ge=1, le=32)
    deadline: Optional[str] = None
    allow_partial: bool = False
    zero_disk: bool = True
    use_tensorrt: bool = True
    subs_after_upscale: bool = True
    subs_quality: str = "normal"
    glossary: Optional[str] = None
    subs_upgrade: bool = False
    no_embed: bool = False
    sound_cues: str = "drop"
    async_subs: bool = False
    prefer_whisper: bool = False
    auto_align_en: bool = True
    sub_offset_ms: int = 0
    keep_temp: bool = False
    force: bool = False
    dry_run: bool = False

    @field_validator("inputs")
    @classmethod
    def _validate_inputs_no_nul(cls, v: List[str]) -> List[str]:
        for item in v:
            _validate_no_nul_str(item, "inputs")
        return v

    @field_validator("batch_size", mode="before")
    @classmethod
    def _validate_batch_size_type(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return _reject_bool_or_float_int(v, "batch_size")

    @field_validator("sub_offset_ms", mode="before")
    @classmethod
    def _validate_sub_offset_type(cls, v: Any) -> Any:
        return _reject_bool_or_float_int(v, "sub_offset_ms")

    @field_validator("output_dir", "glossary")
    @classmethod
    def _validate_outdir_no_nul(cls, v: Optional[str]) -> Optional[str]:
        return _validate_no_nul_str(v, "path")

    @field_validator("sound_cues")
    @classmethod
    def _validate_sc(cls, v: Optional[str]) -> str:
        norm = str(v or "drop").strip().lower() or "drop"
        if norm not in {"drop", "translate"}:
            raise ValueError("sound_cues phải là 'drop' hoặc 'translate'.")
        return norm

    @field_validator("nvenc_multipass")
    @classmethod
    def _validate_mp(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not str(v).strip():
            return None
        norm = str(v).strip().lower()
        if norm not in VALID_NVENC_MULTIPASS_POLICIES:
            raise ValueError("nvenc_multipass phải là 'auto', 'fullres' hoặc 'qres'.")
        return norm

    @field_validator("grain_restore")
    @classmethod
    def _validate_gr(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not str(v).strip():
            return None
        norm = str(v).strip().lower()
        if norm not in VALID_GRAIN_RESTORE_POLICIES:
            raise ValueError("grain_restore phải là 'auto', 'off', 'light' hoặc 'heavy'.")
        return norm


class StartPreviewRequest(BaseModel):
    video_path: str
    seconds: int = 10
    preview_start_sec: Optional[float] = None
    output_dir: Optional[str] = None
    model: Optional[str] = None
    dedup_policy: Optional[str] = None
    nvenc_multipass: Optional[str] = None
    grain_restore: Optional[str] = None
    batch_size: int = Field(default=8, ge=1, le=32)
    zero_disk: bool = True
    use_tensorrt: bool = True
    force: bool = False
    auto_open_player: bool = False

    @field_validator("preview_start_sec", mode="before")
    @classmethod
    def _validate_preview_start_type(cls, v: Any) -> Any:
        return parse_preview_start_value(v)

    @field_validator("preview_start_sec")
    @classmethod
    def _validate_preview_start(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not math.isfinite(v) or v < 0.0 or v > 86400.0:
            raise ValueError("preview_start_sec phải nằm trong khoảng 0..86400 giây.")
        return round(float(v), 3)

    @field_validator("video_path")
    @classmethod
    def _validate_vp(cls, v: str) -> str:
        return _validate_no_nul_str(v, "video_path") or ""

    @field_validator("output_dir")
    @classmethod
    def _validate_outdir_no_nul(cls, v: Optional[str]) -> Optional[str]:
        return _validate_no_nul_str(v, "output_dir")

    @field_validator("batch_size", mode="before")
    @classmethod
    def _validate_batch_size_type(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return _reject_bool_or_float_int(v, "batch_size")

    @field_validator("seconds", mode="before")
    @classmethod
    def _validate_sec_type(cls, v: Any) -> Any:
        return _reject_bool_or_float_int(v, "seconds")

    @field_validator("seconds")
    @classmethod
    def _validate_sec(cls, v: int) -> int:
        if v < 3 or v > 60:
            raise ValueError("seconds phải nằm trong khoảng 3..60 giây.")
        return v

    @field_validator("nvenc_multipass")
    @classmethod
    def _validate_mp(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not str(v).strip():
            return None
        norm = str(v).strip().lower()
        if norm not in VALID_NVENC_MULTIPASS_POLICIES:
            raise ValueError("nvenc_multipass phải là 'auto', 'fullres' hoặc 'qres'.")
        return norm

    @field_validator("grain_restore")
    @classmethod
    def _validate_gr(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not str(v).strip():
            return None
        norm = str(v).strip().lower()
        if norm not in VALID_GRAIN_RESTORE_POLICIES:
            raise ValueError("grain_restore phải là 'auto', 'off', 'light' hoặc 'heavy'.")
        return norm


class StartSubtransRequest(BaseModel):
    input_path: Optional[str] = None
    inputs: Optional[List[str]] = None
    output_dir: Optional[str] = None
    subs_quality: str = "normal"
    subs_upgrade: bool = False
    prefer_whisper: bool = False
    auto_align_en: bool = True
    no_embed: bool = False
    sub_offset_ms: int = 0
    glossary: Optional[str] = None
    output_format: str = "srt"
    dry_run: bool = False

    @field_validator("input_path", "output_dir", "glossary")
    @classmethod
    def _validate_paths_no_nul(cls, v: Optional[str]) -> Optional[str]:
        return _validate_no_nul_str(v, "path")

    @field_validator("inputs")
    @classmethod
    def _validate_sub_inputs_no_nul(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        for item in v:
            _validate_no_nul_str(item, "inputs")
        return v

    @field_validator("output_format")
    @classmethod
    def _validate_sub_fmt(cls, v: Optional[str]) -> str:
        norm = str(v or "srt").strip().lower() or "srt"
        if norm not in {"srt", "ass"}:
            raise ValueError("output_format phải là 'srt' hoặc 'ass'.")
        return norm

    @field_validator("sub_offset_ms", mode="before")
    @classmethod
    def _validate_sub_offset_type(cls, v: Any) -> Any:
        return _reject_bool_or_float_int(v, "sub_offset_ms")


class StartSubsPendingRequest(BaseModel):
    subs_quality: Optional[str] = None
    dry_run: bool = False


class BuildTrtRequest(BaseModel):
    model_id: str
    width: Optional[int] = Field(default=None, ge=64, le=4096)
    height: Optional[int] = Field(default=None, ge=64, le=2304)
    grain_restore: Optional[str] = None
    force: bool = False
    dry_run: bool = False

    @field_validator("model_id")
    @classmethod
    def _validate_mid(cls, v: str) -> str:
        return (_validate_no_nul_str(v, "model_id", reject_unc=False) or "").strip()

    @field_validator("width", "height", mode="before")
    @classmethod
    def _validate_dims_type(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return _reject_bool_or_float_int(v, "width/height")

    @field_validator("width", "height")
    @classmethod
    def _validate_even_dims(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v % 2 != 0:
            raise ValueError("width và height phải là số nguyên chẵn (chia hết cho 2) cho chuẩn 4:2:0.")
        return v

    @field_validator("grain_restore")
    @classmethod
    def _validate_trt_gr(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        norm = _validate_no_nul_str(v, "grain_restore", reject_unc=False)
        clean = (norm or "").strip().lower()
        if not clean:
            return None
        if clean not in (*VALID_GRAIN_RESTORE_POLICIES, "clean"):
            raise ValueError(f"grain_restore '{v}' không hợp lệ (chỉ chấp nhận auto, off, clean, light, heavy).")
        return clean


class CleanTempRequest(BaseModel):
    output_dir: Optional[str] = None
    include_checkpoints: bool = False
    dry_run: bool = False

    @field_validator("output_dir")
    @classmethod
    def _validate_clean_out(cls, v: Optional[str]) -> Optional[str]:
        return _validate_no_nul_str(v, "output_dir")


class SaveGlossaryRequest(BaseModel):
    name: str
    yaml_content: str

    @field_validator("name")
    @classmethod
    def _validate_gname(cls, v: str) -> str:
        return (_validate_no_nul_str(v, "name") or "").strip()

    @field_validator("yaml_content")
    @classmethod
    def _validate_gyaml(cls, v: str) -> str:
        return _validate_no_nul_str(v, "yaml_content", reject_unc=False) or ""


class OpenMediaRequest(BaseModel):
    path: str
    prefer_mpv: bool = True

    @field_validator("path")
    @classmethod
    def _validate_p(cls, v: str) -> str:
        return _validate_no_nul_str(v, "path") or ""


class DeleteCheckpointRequest(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def _validate_cp_path(cls, v: str) -> str:
        return _validate_no_nul_str(v, "path") or ""


class CompareFramesRequest(BaseModel):
    source_path: str
    upscaled_path: str
    timestamp_sec: float = 2.0
    source_offset_sec: Optional[float] = None

    @field_validator("source_path", "upscaled_path")
    @classmethod
    def _reject_null_paths(cls, v: str) -> str:
        if "\x00" in v:
            raise ValueError("Đường dẫn không được chứa ký tự NUL (\\x00).")
        if is_unc_path(v):
            raise ValueError("Từ chối đường dẫn mạng UNC (\\\\... hoặc //...).")
        return v

    @field_validator("timestamp_sec", "source_offset_sec")
    @classmethod
    def _reject_non_finite_floats(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not math.isfinite(v):
            raise ValueError("Giá trị thời gian phải là số hữu hạn (không nhận NaN/Infinity).")
        return v


class TestShowsMatchRequest(BaseModel):
    sample_path: str
    yaml_content: Optional[str] = None

    @field_validator("sample_path")
    @classmethod
    def _reject_nul_sample(cls, v: Optional[str]) -> Optional[str]:
        return _validate_no_nul_str(v, "sample_path", reject_unc=True)

    @field_validator("yaml_content")
    @classmethod
    def _reject_nul(cls, v: Optional[str]) -> Optional[str]:
        return _validate_no_nul_str(v, "yaml_content", reject_unc=False)


class SaveShowsYamlRequest(BaseModel):
    yaml_content: str

    @field_validator("yaml_content")
    @classmethod
    def _reject_nul(cls, v: str) -> str:
        return _validate_no_nul_str(v, "yaml_content", reject_unc=False) or ""


class UpdateApiKeysRequest(BaseModel):
    key_1: Optional[str] = None
    key_2: Optional[str] = None
    key_3: Optional[str] = None
    slots: Optional[Dict[int, Optional[str]]] = None

    @field_validator("key_1", "key_2", "key_3")
    @classmethod
    def _reject_crlf_and_null(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and any(ch in v for ch in ("\r", "\n", "\x00")):
            raise ValueError("API key không được chứa ký tự xuống dòng (CR/LF) hoặc NUL.")
        return v

    @field_validator("slots", mode="before")
    @classmethod
    def _validate_slots_map(cls, v: Any) -> Any:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("slots phải là từ điển ánh xạ số khe (1..32) sang chuỗi API key.")
        normalized: Dict[int, Optional[str]] = {}
        for raw_k, raw_val in v.items():
            if isinstance(raw_k, bool):
                raise ValueError("Số khe khóa không được là kiểu boolean.")
            try:
                slot_idx = int(str(raw_k).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Số khe khóa '{raw_k}' không hợp lệ (phải là số nguyên 1..32).") from exc
            if slot_idx < 1 or slot_idx > 32:
                raise ValueError("Số khe khóa phải nằm trong khoảng 1..32.")
            if raw_val is not None:
                val_str = str(raw_val)
                if any(ch in val_str for ch in ("\r", "\n", "\x00")):
                    raise ValueError("API key không được chứa ký tự xuống dòng (CR/LF) hoặc NUL.")
                normalized[slot_idx] = val_str
            else:
                normalized[slot_idx] = None
        return normalized


# HELPER UTILITIES

def _mask_secret(raw: str) -> str:
    val = (raw or "").strip()
    if not val:
        return ""
    if len(val) <= 10:
        return f"{val[:2]}***{val[-2:]}"
    return f"{val[:6]}...{val[-4:]}"


_DOTENV_LOCK = threading.Lock()


def _read_dotenv_map(dotenv_path: Path) -> Dict[str, str]:
    """Read KEY=VALUE entries from a local .env file without logging secret values."""
    result: Dict[str, str] = {}
    if not dotenv_path.is_file():
        return result
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                v = v[1:-1]
            if k:
                result[k] = v
    except Exception:
        pass
    return result


def _validate_dotenv_value(raw_val: str, key_name: str = "API key") -> str:
    """Reject CR/LF newlines or NUL bytes before writing .env entries."""
    if any(ch in raw_val for ch in ("\r", "\n", "\x00")):
        raise ValueError(f"Giá trị '{key_name}' chứa ký tự xuống dòng (CR/LF) hoặc NUL không hợp lệ.")
    return raw_val.strip()


def _write_dotenv_keys(dotenv_path: Path, updates: Dict[str, Optional[str]]) -> None:
    """Safely update specified keys in local .env while preserving unrelated entries."""
    sanitized_updates: Dict[str, Optional[str]] = {}
    for key_name, raw_val in updates.items():
        if raw_val is None:
            sanitized_updates[key_name] = None
        else:
            sanitized_updates[key_name] = _validate_dotenv_value(str(raw_val), key_name=key_name)

    existing_lines: List[str] = []
    if dotenv_path.is_file():
        existing_lines = dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines()

    handled: set[str] = set()
    new_lines: List[str] = []

    for raw_line in existing_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(raw_line)
            continue
        check_line = stripped[len("export "):].strip() if stripped.startswith("export ") else stripped
        key_name = check_line.split("=", 1)[0].strip()
        if key_name in sanitized_updates and sanitized_updates[key_name] is not None:
            if key_name in handled:
                continue
            val = sanitized_updates[key_name] or ""
            handled.add(key_name)
            if val:
                new_lines.append(f"{key_name}={val}")
            # Empty string deletes the existing key line
        else:
            new_lines.append(raw_line)

    for key_name, val in sanitized_updates.items():
        if val is None or key_name in handled:
            continue
        if val:
            new_lines.append(f"{key_name}={val}")

    tmp_path = dotenv_path.parent / f".env.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:6]}.tmp"
    content = "\n".join(new_lines).rstrip("\n") + ("\n" if new_lines else "")
    try:
        tmp_path.write_text(content, encoding="utf-8", newline="\n")
        os.replace(tmp_path, dotenv_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _split_gemini_key_tokens(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tok.strip() for tok in re.split(r"[;,]", raw) if tok.strip()]


_SLOT_ENV_KEY_RE = re.compile(r"^GEMINI_API_KEY_([1-9]\d*)$")


def _resolve_subtrans_key_slots_all(
    dotenv_path: Path,
    key_id_fn: Optional[Callable[[str], str]] = None,
) -> tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """Resolve 3-slot summary and dynamic 1..N API key slot statuses matching _apply_subtrans_key_edits."""
    sub_dotenv_path = dotenv_path.parent / "subtrans" / ".env"
    dotenv_map = _read_dotenv_map(dotenv_path)
    if sub_dotenv_path.is_file():
        for sk, sv in _read_dotenv_map(sub_dotenv_path).items():
            # Match load_dotenv(..., override=False): subtrans/.env presence blocks root .env even when empty
            dotenv_map[sk] = sv

    def _effective_var(name: str) -> tuple[str, str]:
        if name in os.environ:
            return os.environ[name].strip(), "env"
        if name in dotenv_map:
            return dotenv_map[name].strip(), "dotenv"
        return "", ""

    prefix_entries: List[tuple[str, str, str]] = []
    seen_keys: set[str] = set()

    for p_name in ("GEMINI_API_KEYS", "GEMINI_API_KEY"):
        p_raw, p_src = _effective_var(p_name)
        for tok in _split_gemini_key_tokens(p_raw):
            if tok not in seen_keys:
                seen_keys.add(tok)
                prefix_entries.append((tok, p_src, p_name))

    slot_table: Dict[int, tuple[str, str, str]] = {}
    for idx, entry in enumerate(prefix_entries, start=1):
        slot_table[idx] = entry

    all_var_names = set(dotenv_map.keys()) | set(os.environ.keys())
    numbered_names: List[tuple[int, str]] = []
    alpha_names: List[str] = []
    for k in all_var_names:
        if not k.startswith("GEMINI_API_KEY_"):
            continue
        m_num = _SLOT_ENV_KEY_RE.match(k)
        if m_num:
            numbered_names.append((int(m_num.group(1)), k))
        else:
            alpha_names.append(k)

    for n_idx, var_name in sorted(numbered_names, key=lambda x: x[0]):
        e_val, e_src = _effective_var(var_name)
        for tok in _split_gemini_key_tokens(e_val):
            if tok in seen_keys:
                continue
            seen_keys.add(tok)
            target = n_idx
            while target in slot_table:
                target += 1
            if target <= 32:
                slot_table[target] = (tok, e_src, var_name)

    for var_name in sorted(alpha_names):
        e_val, e_src = _effective_var(var_name)
        for tok in _split_gemini_key_tokens(e_val):
            if tok in seen_keys:
                continue
            seen_keys.add(tok)
            target = max(slot_table.keys(), default=0) + 1
            if target <= 32:
                slot_table[target] = (tok, e_src, var_name)

    total_slot_count = max(3, min(32, max(slot_table.keys(), default=0)))

    all_slots_summary: List[Dict[str, Any]] = []
    for idx in range(1, total_slot_count + 1):
        default_var = f"GEMINI_API_KEY_{idx}"
        if idx in slot_table:
            chosen_val, chosen_src, chosen_var = slot_table[idx]
        else:
            chosen_val, chosen_src, chosen_var = "", "", default_var

        all_slots_summary.append(
            {
                "slot": idx,
                "env_name": chosen_var,
                "configured": bool(chosen_val),
                "masked": _mask_secret(chosen_val) if chosen_val else "",
                "key_id": key_id_fn(chosen_val) if (chosen_val and key_id_fn) else ("configured" if chosen_val else "none"),
                "source": chosen_src,
            }
        )

    return all_slots_summary[:3], len(seen_keys), all_slots_summary


def _resolve_subtrans_key_slots(
    dotenv_path: Path,
    key_id_fn: Optional[Callable[[str], str]] = None,
) -> tuple[List[Dict[str, Any]], int]:
    slots_3, active_cnt, _ = _resolve_subtrans_key_slots_all(dotenv_path, key_id_fn=key_id_fn)
    return slots_3, active_cnt


def _apply_subtrans_key_edits(
    existing: Dict[str, str],
    key_1: Optional[str],
    key_2: Optional[str],
    key_3: Optional[str],
    slots: Optional[Dict[int, Optional[str]]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[str]]:
    """Transform merged .env key entries into normalized GEMINI_API_KEY_1..N slots plus synced GEMINI_API_KEY."""
    requested_edits: Dict[int, str] = {}
    for base_idx, base_val in ((1, key_1), (2, key_2), (3, key_3)):
        if base_val is not None:
            requested_edits[base_idx] = base_val
    if slots:
        for dyn_idx, dyn_val in slots.items():
            if dyn_val is not None and 1 <= int(dyn_idx) <= 32:
                requested_edits[int(dyn_idx)] = dyn_val
    if not requested_edits:
        return {}

    env_map = {k: str(v).strip() for k, v in (env_overrides or {}).items() if k.startswith("GEMINI_API_KEY")}

    def _eff_entry(var_name: str) -> tuple[str, str]:
        if var_name in env_map:
            return env_map[var_name], "env"
        if var_name in existing:
            return str(existing[var_name]).strip(), "dotenv"
        return "", ""

    eff_key_1 = requested_edits.get(1)
    eff_key_2 = requested_edits.get(2)
    multi_raw, multi_src = _eff_entry("GEMINI_API_KEYS")
    prim_raw, prim_src = _eff_entry("GEMINI_API_KEY")
    old_multi = _split_gemini_key_tokens(multi_raw)
    old_prim_orig = _split_gemini_key_tokens(prim_raw)
    old_prim = list(old_prim_orig)
    s1_raw, _ = _eff_entry("GEMINI_API_KEY_1")
    legacy_triple_coexist = bool(
        old_multi and len(old_prim_orig) == 1 and s1_raw and prim_src != "env" and multi_src != "env"
    )

    if legacy_triple_coexist and eff_key_1 is not None:
        old_prim = []

    prefix_tokens: List[tuple[str, str]] = []
    seen_tokens: set[str] = set()
    for tok in old_multi:
        if tok not in seen_tokens:
            seen_tokens.add(tok)
            prefix_tokens.append((tok, multi_src or "dotenv"))
    for tok in old_prim:
        if tok not in seen_tokens:
            seen_tokens.add(tok)
            prefix_tokens.append((tok, prim_src or "dotenv"))
    prefix_len = len(prefix_tokens)

    all_keys = set(existing.keys()) | set(env_map.keys())
    numbered_vars: Dict[int, tuple[List[str], str]] = {}
    alpha_vars: List[tuple[str, List[str], str]] = []
    for k in all_keys:
        if not k.startswith("GEMINI_API_KEY_"):
            continue
        m = _SLOT_ENV_KEY_RE.match(k)
        raw_v, src_v = _eff_entry(k)
        if m:
            numbered_vars[int(m.group(1))] = (_split_gemini_key_tokens(raw_v), src_v or "dotenv")
        else:
            alpha_vars.append((k, _split_gemini_key_tokens(raw_v), src_v or "dotenv"))

    # Phase 1: Build canonical 1-based slot_table matching _resolve_subtrans_key_slots_all().
    slot_table: Dict[int, tuple[str, str]] = {}
    for idx, entry_pair in enumerate(prefix_tokens, start=1):
        slot_table[idx] = entry_pair

    for n in sorted(numbered_vars.keys()):
        toks_n, src_n = numbered_vars[n]
        if (
            prefix_len > 0
            and n == 1
            and src_n != "env"
            and (
                (eff_key_1 is not None and legacy_triple_coexist)
                or (
                    bool(key_2 and key_2.strip())
                    and not slots
                    and not old_multi
                    and len(old_prim_orig) == 1
                    and prim_src != "env"
                    and bool(numbered_vars.get(2, ([], ""))[0])
                )
            )
        ):
            continue
        for tok in toks_n:
            if tok in seen_tokens:
                continue
            seen_tokens.add(tok)
            target = n
            while target in slot_table:
                target += 1
            if target <= 32:
                slot_table[target] = (tok, src_n)

    for _alpha_name, toks_a, src_a in sorted(alpha_vars, key=lambda x: x[0]):
        for tok in toks_a:
            if tok in seen_tokens:
                continue
            seen_tokens.add(tok)
            target = max(slot_table.keys(), default=0) + 1
            if target <= 32:
                slot_table[target] = (tok, src_a)

    # Phase 2: Apply requested Slot 1..32 edits or explicit clears.
    for slot_idx in sorted(requested_edits.keys()):
        raw_val = requested_edits[slot_idx]
        cleaned = _validate_dotenv_value(raw_val, key_name=f"GEMINI_API_KEY_{slot_idx}")
        cur_pair = slot_table.get(slot_idx)
        if cleaned:
            if cur_pair and cur_pair[1] == "env":
                raise ValueError(
                    f"Khe {slot_idx} đang bị khóa bởi biến môi trường hệ thống (os.environ); hãy chỉnh biến môi trường trực tiếp."
                )
            slot_table[slot_idx] = (cleaned, "dotenv")
        else:
            if cur_pair and cur_pair[1] == "env":
                shadowed_dotenv_toks = set(
                    _split_gemini_key_tokens(str(existing.get(f"GEMINI_API_KEY_{slot_idx}") or ""))
                )
                if shadowed_dotenv_toks:
                    for k_sl in [
                        ks for ks, (ktok, ksrc) in slot_table.items()
                        if ksrc == "dotenv" and ktok in shadowed_dotenv_toks
                    ]:
                        slot_table.pop(k_sl, None)
                else:
                    raise ValueError(
                        f"Khe {slot_idx} đến từ biến môi trường hệ thống (os.environ), không thể xóa từ tệp .env."
                    )
            else:
                slot_table.pop(slot_idx, None)

    # Phase 3: Materialize normalized .env updates without leaking os.environ secrets onto disk.
    slot1_dotenv = slot_table[1][0] if (1 in slot_table and slot_table[1][1] == "dotenv") else ""
    updates: Dict[str, Optional[str]] = {
        "GEMINI_API_KEYS": "",
        "GEMINI_API_KEY": "" if "GEMINI_API_KEY" in env_map else slot1_dotenv,
    }
    for k_alpha in existing:
        if k_alpha.startswith("GEMINI_API_KEY_") and not _SLOT_ENV_KEY_RE.match(k_alpha):
            updates[k_alpha] = ""
    all_indices = (
        {1, 2, 3}
        | set(slot_table.keys())
        | {int(m.group(1)) for k in existing if (m := _SLOT_ENV_KEY_RE.match(k))}
        | set(requested_edits.keys())
    )
    for idx in sorted(all_indices):
        pair = slot_table.get(idx)
        updates[f"GEMINI_API_KEY_{idx}"] = pair[0] if (pair and pair[1] == "dotenv") else ""
    return updates


_HMS_DURATION_RE = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")


def _parse_hms_duration_tag(raw_val: Any) -> float:
    m = _HMS_DURATION_RE.match(str(raw_val or "").strip())
    if not m:
        return 0.0
    try:
        return int(m.group(1)) * 3600.0 + int(m.group(2)) * 60.0 + float(m.group(3))
    except ValueError:
        return 0.0


def _resolve_ffmpeg_binary() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    bundled = repo_root / "bin" / "ffmpeg" / exe_name
    if bundled.is_file():
        return str(bundled)
    return "ffmpeg"


def _probe_duration_and_dims(video_path: Path) -> Dict[str, Any]:
    """Read video stream duration, FPS, and pixel dimensions via PyAV, returning zeros on failure."""
    try:
        import av
        with av.open(str(video_path)) as container:
            if container.streams.video:
                vs = container.streams.video[0]
                rate = vs.average_rate or vs.base_rate
                fps_val = float(rate) if rate and float(rate) > 0 else 23.976
                container_dur = float(container.duration / av.time_base) if container.duration else 0.0
                stream_dur = 0.0
                if vs.duration is not None and vs.time_base is not None:
                    cand_sd = float(vs.duration * vs.time_base)
                    if cand_sd > 0.0 and (container_dur <= 0.0 or cand_sd <= container_dur + 1.0):
                        stream_dur = cand_sd
                if stream_dur <= 0.0 and vs.metadata:
                    for mk, mv in vs.metadata.items():
                        if str(mk).upper().startswith("DURATION"):
                            parsed_tag = _parse_hms_duration_tag(mv)
                            if parsed_tag > 0.0 and (container_dur <= 0.0 or parsed_tag <= container_dur + 1.0):
                                stream_dur = parsed_tag
                                break
                if stream_dur <= 0.0 and vs.frames and int(vs.frames) > 0 and fps_val > 0.0:
                    cand_fd = float(vs.frames) / fps_val
                    if container_dur <= 0.0 or cand_fd <= container_dur + 1.0:
                        stream_dur = cand_fd
                if stream_dur <= 0.0 and container_dur > 0.0:
                    try:
                        c_start_s = float(container.start_time / av.time_base) if container.start_time is not None else 0.0
                        max_end_s = 0.0
                        for tail_window_s in (15.0, 60.0, container_dur):
                            seek_s = max(0.0, container_dur - tail_window_s) + c_start_s
                            if vs.time_base is not None and float(vs.time_base) > 0.0:
                                container.seek(int(seek_s / float(vs.time_base)), backward=True, any_frame=False, stream=vs)
                            else:
                                container.seek(int(seek_s * av.time_base), backward=True, any_frame=False)
                            for pkt in container.demux(vs):
                                if pkt.pts is None or getattr(pkt, "is_discard", False):
                                    continue
                                tb = pkt.time_base or vs.time_base
                                if tb is not None:
                                    pkt_s = float(pkt.pts * tb) - c_start_s
                                    pkt_d = float(pkt.duration * tb) if pkt.duration else (1.0 / fps_val)
                                    if pkt_s + pkt_d > max_end_s:
                                        max_end_s = pkt_s + pkt_d
                            if max_end_s > 0.0 or tail_window_s >= container_dur:
                                break
                        if max_end_s > 0.0:
                            stream_dur = max_end_s
                    except Exception:
                        pass
                eff_dur = (
                    min(stream_dur, container_dur)
                    if (stream_dur > 0.0 and container_dur > 0.0)
                    else (stream_dur if stream_dur > 0.0 else container_dur)
                )
                return {
                    "duration": round(max(0.0, eff_dur), 3),
                    "width": int(vs.width or 0),
                    "height": int(vs.height or 0),
                    "fps": round(fps_val, 3),
                }
    except Exception:
        pass
    return {"duration": 0.0, "width": 0, "height": 0, "fps": 23.976}


def _default_extract_frame(video_path: Path, timestamp_sec: float, out_png: Path) -> None:
    """Extract an 8-bit rgb24 PNG frame atomically with normalized BT.709->sRGB color tags."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp_png = (
        out_png.parent
        / f".{out_png.stem}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp.png"
    )
    cmd = [
        _resolve_ffmpeg_binary(),
        "-y",
        "-ss",
        f"{max(0.0, float(timestamp_sec)):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        (
            "scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc:flags=bicubic+accurate_rnd+full_chroma_int,"
            "format=rgb24,"
            "setparams=colorspace=gbr:color_primaries=bt709:color_trc=iec61966-2-1:range=pc"
        ),
        str(tmp_png),
    ]
    startupinfo = None
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=25.0,
            check=False,
            startupinfo=startupinfo,
        )
    except subprocess.TimeoutExpired as exc:
        with suppress(OSError):
            tmp_png.unlink(missing_ok=True)
        raise RuntimeError(
            f"Quá thời gian chờ (25s) khi trích xuất khung hình tại {timestamp_sec:.2f}s từ '{video_path.name}'."
        ) from exc

    try:
        valid_tmp = res.returncode == 0 and tmp_png.is_file() and tmp_png.stat().st_size > 0
    except OSError:
        valid_tmp = False
    if not valid_tmp:
        with suppress(OSError):
            tmp_png.unlink(missing_ok=True)
        raise RuntimeError(
            f"FFmpeg không thể trích xuất khung hình tại {timestamp_sec:.2f}s từ '{video_path.name}'."
        )
    try:
        os.replace(tmp_png, out_png)
    except OSError as exc:
        with suppress(OSError):
            tmp_png.unlink(missing_ok=True)
        raise RuntimeError(f"Không thể lưu khung hình trích xuất '{out_png.name}': {exc}") from exc


def _default_open_media(target_path: Path, prefer_mpv: bool = True) -> str:
    """Open local media file with mpv if installed or via Windows os.startfile fallback."""
    if prefer_mpv:
        import shutil
        mpv_bin = shutil.which("mpv")
        if mpv_bin:
            subprocess.Popen([mpv_bin, str(target_path)])
            return "mpv"
    if sys.platform == "win32" and hasattr(os, "startfile"):
        os.startfile(str(target_path))  # type: ignore[attr-defined]
        return "os.startfile"
    raise RuntimeError("Môi trường hiện tại không hỗ trợ mở trình phát media trực tiếp.")


# APPLICATION FACTORY

_INLINE_THEME_SCRIPT_SHA256 = "sha256-uwuIZqV9D8DaFdBXlW9a6ICGWbLeDFMqwRUM1xF0W04="
_CSP_HEADER_VALUE = (
    f"default-src 'self'; "
    f"script-src 'self' '{_INLINE_THEME_SCRIPT_SHA256}'; "
    f"style-src 'self'; "
    f"img-src 'self' data:; "
    f"connect-src 'self'; "
    f"font-src 'self'; "
    f"object-src 'none'; "
    f"base-uri 'self'; "
    f"frame-ancestors 'none'; "
    f"form-action 'self'"
)
_SINGLE_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def _is_loopback_ip(raw_host: str) -> bool:
    h = (raw_host or "").strip().strip("[]")
    if "%" in h:
        h = h.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    return bool(mapped is not None and mapped.is_loopback)


def _is_loopback_request(client_host: str, host_name: str) -> bool:
    if client_host == "testclient":
        return (
            host_name in {"testserver", "localhost"}
            or host_name.endswith(".localhost")
            or _is_loopback_ip(host_name)
        )
    return _is_loopback_ip(client_host)


def _resolve_dir_from_raw(
    root: Path,
    raw_val: Optional[str],
    strict_secret: bool = False,
) -> Optional[Path]:
    if not raw_val or not str(raw_val).strip():
        return None
    _validate_no_nul_str(str(raw_val), "output_dir")
    raw_unwrapped = unwrap_outer_quotes_preserving_trailing_space(str(raw_val))
    cleaned = strip_wrapping_quotes(str(raw_val))
    if not cleaned:
        return None
    _validate_no_nul_str(cleaned, "output_dir")
    raw_path = Path(os.path.expanduser(cleaned))
    if (raw_path.drive and not raw_path.is_absolute()) or (sys.platform == "win32" and not raw_path.drive and raw_path.root):
        if strict_secret:
            raise ValueError("Đường dẫn tương đối theo ổ đĩa hoặc thiếu ký tự ổ đĩa không hợp lệ.")
        return None
    if (
        is_secret_filename(cleaned)
        or is_secret_filename(raw_path.name)
        or any(is_secret_filename(part) for part in raw_path.parts)
    ):
        if strict_secret:
            raise PermissionError("Từ chối truy cập tệp cấu hình nhạy cảm.")
        return None
    if strict_secret:
        validate_win32_path_hygiene(
            Path(os.path.expanduser(raw_unwrapped)),
            field="output_dir",
            raw_str=raw_unwrapped,
        )
    p = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        resolved = p.resolve()
    except OSError:
        return None
    if is_secret_filename(resolved.name) or any(is_secret_filename(part) for part in resolved.parts):
        if strict_secret:
            raise PermissionError("Từ chối truy cập tệp cấu hình nhạy cảm.")
        return None
    if resolved.suffix.lower() in VIDEO_EXTENSIONS:
        return resolved.parent
    return resolved


def _read_live_status_dict(root: Path) -> Dict[str, Any]:
    live_file = root / "logs" / "live_status.json"
    if not live_file.is_file():
        return {}
    try:
        parsed = json.loads(live_file.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _parse_valid_preview_start_sec(raw_val: Any) -> Optional[float]:
    if raw_val is None or isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
        return None
    val = float(raw_val)
    if not math.isfinite(val) or val < 0.0:
        return None
    return val


def _lookup_preview_start_sec(root: Path, src_p: Path, up_p: Path) -> Optional[float]:
    for audit_cand in (
        up_p.with_suffix(".audit.json"),
        up_p.parent / "audit" / f"{up_p.stem}.audit.json",
    ):
        if not audit_cand.is_file():
            continue
        try:
            adata = json.loads(audit_cand.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(adata, dict):
            parsed = _parse_valid_preview_start_sec(adata.get("preview_start_sec"))
            if parsed is not None:
                return parsed

    live_data = _read_live_status_dict(root)
    if not live_data or live_data.get("is_preview") is False:
        return None
    live_parsed = _parse_valid_preview_start_sec(live_data.get("preview_start_sec"))
    if live_parsed is None:
        return None

    def _norm_resolved(raw_path: str) -> Optional[str]:
        if not raw_path:
            return None
        cand_p = Path(os.path.expanduser(raw_path))
        if not cand_p.is_absolute():
            cand_p = root / cand_p
        try:
            return os.path.normcase(str(cand_p.resolve()))
        except OSError:
            return os.path.normcase(os.path.normpath(str(cand_p)))

    norm_up = os.path.normcase(str(up_p))
    norm_src = os.path.normcase(str(src_p))
    raw_out = str(live_data.get("output_path") or "").strip()
    if raw_out:
        if _norm_resolved(raw_out) == norm_up:
            return live_parsed
        return None

    live_vname = str(live_data.get("video_name") or "").strip()
    up_base_stem = re.sub(r"_4K(?:_animejanai)?$", "", up_p.stem, flags=re.IGNORECASE)
    if not live_vname or live_vname.lower() not in {up_p.stem.lower(), up_base_stem.lower()}:
        return None
    raw_in = str(live_data.get("input_path") or "").strip()
    if raw_in and _norm_resolved(raw_in) != norm_src:
        return None
    return live_parsed


def _collect_known_output_dirs(
    root: Path,
    job_manager: JobManager,
    extra_dir: Optional[str] = None,
) -> List[Path]:
    """Collect deduplicated existing output directories from out/, live_status.json, and JobManager."""
    candidates: List[Path] = []
    if extra_dir is not None and str(extra_dir).strip():
        resolved_extra = _resolve_dir_from_raw(root, extra_dir, strict_secret=True)
        if resolved_extra is not None:
            candidates.append(resolved_extra)

    candidates.append(root / "out")

    live_data = _read_live_status_dict(root)
    if live_data.get("output_path"):
        d_live = _resolve_dir_from_raw(root, str(live_data.get("output_path")))
        if d_live is not None:
            candidates.append(d_live)

    try:
        job_state = job_manager.get_state()
    except Exception:
        job_state = {}
    job_records: List[Dict[str, Any]] = []
    if isinstance(job_state.get("active_job"), dict):
        job_records.append(job_state["active_job"])
    for hj in job_state.get("history") or []:
        if isinstance(hj, dict):
            job_records.append(hj)
    for job in job_records:
        opts = job.get("options") if isinstance(job.get("options"), dict) else {}
        for opt_key in ("output_dir", "output_path"):
            raw_opt = opts.get(opt_key)
            if raw_opt:
                try:
                    d_job = _resolve_dir_from_raw(root, str(raw_opt))
                except ValueError:
                    d_job = None
                if d_job is not None:
                    candidates.append(d_job)

    unique: List[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
            if not resolved.is_dir():
                continue
            norm = os.path.normcase(str(resolved))
        except OSError:
            continue
        if norm not in seen:
            seen.add(norm)
            unique.append(resolved)
    return unique


def _collect_known_source_dirs(
    root: Path,
    out_dirs: Sequence[Path],
    job_manager: JobManager,
) -> List[Path]:
    """Collect known source video directories from workspace, out_dirs, live_status.json, and JobManager."""
    candidates: List[Path] = [root, *out_dirs]

    live_data = _read_live_status_dict(root)
    if live_data.get("input_path"):
        raw_in = str(live_data.get("input_path")).strip()
        if raw_in and ".preview_clip_" not in raw_in and "\x00" not in raw_in:
            p_in = Path(os.path.expanduser(raw_in))
            if not p_in.is_absolute():
                p_in = root / p_in
            try:
                rp_in = p_in.resolve()
                candidates.append(rp_in if rp_in.is_dir() else rp_in.parent)
            except OSError:
                pass

    try:
        job_state = job_manager.get_state()
    except Exception:
        job_state = {}
    job_records: List[Dict[str, Any]] = []
    if isinstance(job_state.get("active_job"), dict):
        job_records.append(job_state["active_job"])
    for hj in job_state.get("history") or []:
        if isinstance(hj, dict):
            job_records.append(hj)
    for job in job_records:
        for qi in job.get("queue_items") or []:
            raw_qi = str(qi or "").strip()
            if not raw_qi or "\x00" in raw_qi or ".preview_clip_" in raw_qi:
                continue
            p_qi = Path(os.path.expanduser(raw_qi))
            if not p_qi.is_absolute():
                p_qi = root / p_qi
            try:
                rp_qi = p_qi.resolve()
                candidates.append(rp_qi if rp_qi.is_dir() else rp_qi.parent)
            except OSError:
                pass

    unique: List[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
            if not resolved.is_dir():
                continue
            norm = os.path.normcase(str(resolved))
        except OSError:
            continue
        if norm not in seen:
            seen.add(norm)
            unique.append(resolved)
    return unique


def _parse_host_header(raw_host: Optional[str]) -> tuple[str, Optional[int], bool]:
    if not raw_host:
        return "", None, False
    h = raw_host.strip()
    if not h or "," in h or "@" in h or any(c.isspace() for c in h):
        return "", None, False
    try:
        parsed = urllib.parse.urlsplit(f"//{h}")
        port = parsed.port
    except ValueError:
        return "", None, False
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname:
        return "", None, False
    return hostname, port, True


def _is_allowed_hostname(hostname: str, allow_lan: bool, client_host: str) -> bool:
    if not hostname:
        return False
    if hostname == "testserver" and client_host == "testclient":
        return True
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None
    if ip is not None:
        if ip.is_loopback:
            return True
        if allow_lan and (ip.is_private or ip.is_link_local) and not ip.is_unspecified and not ip.is_multicast:
            return True
        return False
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    if allow_lan:
        local_name = socket.gethostname().rstrip(".").lower()
        if hostname == local_name:
            return True
        if hostname.endswith(".local"):
            stem = hostname[:-6]
            if stem and _SINGLE_LABEL_RE.match(stem):
                return True
        if "." not in hostname and _SINGLE_LABEL_RE.match(hostname):
            return True
    return False


def _validate_origin_header(
    raw_origin: Optional[str],
    host_name: str,
    host_port: Optional[int],
    allow_lan: bool,
    client_host: str,
) -> bool:
    if raw_origin is None:
        return True
    o_val = raw_origin.strip()
    if not o_val or o_val.lower() == "null":
        return False
    try:
        parsed = urllib.parse.urlsplit(o_val)
        o_port = parsed.port
    except ValueError:
        return False
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or parsed.username or parsed.password:
        return False
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        return False
    o_host = (parsed.hostname or "").rstrip(".").lower()
    if not _is_allowed_hostname(o_host, allow_lan=allow_lan, client_host=client_host):
        return False
    if o_host != host_name:
        return False
    if host_port is not None:
        eff_o_port = o_port if o_port is not None else (443 if scheme == "https" else 80)
        if eff_o_port != host_port:
            return False
    elif o_port is not None:
        return False
    return True


def _build_mock_gpu_query() -> Callable[[], Dict[str, Any]]:
    """Return deterministic simulated RTX 3060 12GB GPU metrics without invoking nvidia-smi."""
    tick_state = {"step": 0}

    def _query() -> Dict[str, Any]:
        step = tick_state["step"]
        tick_state["step"] += 1
        wave = step % 6
        vram_used = 6144.0 + wave * 48.0
        return {
            "pstate": "P0",
            "gpu_util_pct": float(76 + (wave * 3) % 15),
            "mem_util_pct": float(58 + (wave * 2) % 10),
            "enc_util_pct": float(44 + (wave * 4) % 16),
            "dec_util_pct": float(30 + (wave * 3) % 12),
            "mem_clock_mhz": 7500.0,
            "core_clock_mhz": 1867.0 + wave * 15.0,
            "power_w": 138.0 + wave * 2.5,
            "temp_c": 62.0 + float(wave % 3),
            "vram_used_mib": vram_used,
            "vram_free_mib": max(0.0, 12288.0 - vram_used),
        }

    return _query


def _enable_mock_studio_state(
    job_manager: JobManager,
    telemetry_hub: TelemetryHub,
) -> Dict[str, Callable[[], Dict[str, Any]]]:
    """Attach deterministic simulated active/completed jobs, telemetry ticks, and SSE logs."""
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    mock_active_job: Dict[str, Any] = {
        "job_id": "mock-upscale-ep02",
        "job_type": "upscale",
        "status": "running",
        "pid": 90420,
        "started_at": now_str,
        "finished_at": None,
        "return_code": None,
        "command": ["python", "upscale.py", "[Demo] Anime_EP02.mkv", "-o", "out/"],
        "queue_items": ["[Demo] Anime_EP02 (1080p).mkv"],
        "options": {"model": "animevideov3", "zero_disk": True, "dedup_policy": "dis"},
    }
    mock_history: List[Dict[str, Any]] = [
        {
            "job_id": "mock-subtrans-ep01",
            "job_type": "subtrans",
            "status": "completed",
            "pid": 90412,
            "started_at": now_str,
            "finished_at": now_str,
            "return_code": 0,
            "command": ["python", "subtrans.py", "run", "Anime_EP01.ja.srt", "-o", "out/"],
            "queue_items": ["[Demo] Anime_EP01 (1080p).mkv"],
            "options": {"subs_quality": "high", "dry_run": True},
        },
        {
            "job_id": "mock-upscale-ep01",
            "job_type": "upscale",
            "status": "completed",
            "pid": 90401,
            "started_at": now_str,
            "finished_at": now_str,
            "return_code": 0,
            "command": ["python", "upscale.py", "[Demo] Anime_EP01.mkv", "-o", "out/"],
            "queue_items": ["[Demo] Anime_EP01 (1080p).mkv"],
            "options": {"model": "animevideov3", "zero_disk": True, "dedup_policy": "default"},
        },
    ]

    mock_lock = threading.Lock()
    tick_counter = {"n": 0}
    mock_runtime = {"pause_pending": False, "aborted": False}
    total_chunks = 12
    total_frames = 34560

    orig_get_state = job_manager.get_state

    def _mock_get_state() -> Dict[str, Any]:
        real_state = orig_get_state()
        if real_state.get("active_job") is not None:
            return real_state
        with mock_lock:
            pause_pending = mock_runtime["pause_pending"]
            aborted = mock_runtime["aborted"]
        combined_history = list(real_state.get("history") or []) + [dict(h) for h in mock_history]
        if aborted:
            return {
                "status": "idle",
                "active_job": None,
                "history": combined_history[:25],
            }
        active_copy = dict(mock_active_job)
        active_copy["status"] = "pausing" if pause_pending else "running"
        return {
            "status": active_copy["status"],
            "active_job": active_copy,
            "history": combined_history[:25],
        }

    job_manager.get_state = _mock_get_state  # type: ignore[method-assign]
    telemetry_hub.gpu_static = {
        "gpu_name": "NVIDIA GeForce RTX 3060 12GB (Mock)",
        "driver_version": "Mock 572.x",
        "vram_total_mib": 12288.0,
    }

    def _mock_read_pipeline_state() -> Dict[str, Any]:
        with mock_lock:
            n = tick_counter["n"]
            pause_pending = mock_runtime["pause_pending"]
            aborted = mock_runtime["aborted"]
        completed_chunks = min(total_chunks - 1, 4 + n)
        chunk_frames = total_frames // total_chunks
        done_frames = min(total_frames, completed_chunks * chunk_frames)
        # Mirror RunLogger: emit 0-based entries solely for completed chunks.
        chunks = [
            {
                "chunk_idx": idx,
                "frames": chunk_frames,
                "ai_fps": round(40.8 + (idx % 3) * 0.4, 2),
                "steady_fps": 41.2,
                "effective_fps": round(41.1 + (idx % 3) * 0.3, 2),
                "size_mb": round(28.5 + (idx % 4) * 1.8, 2),
            }
            for idx in range(completed_chunks)
        ]
        latest_fps = round(41.2 + (n % 4) * 0.4, 1)
        stage_val = "paused" if aborted else "upscaling"
        return {
            "active_source": "web",
            "gpu_lock": {"locked": not aborted, "pid": 90420, "mock": True},
            "graceful_pause_pending": pause_pending,
            "effective_stage": stage_val,
            "pid_alive": not aborted,
            "mock_mode": True,
            "live": {
                "video_name": "[Demo] Anime_EP02 (1080p).mkv",
                "pid": 90420,
                "job_id": "mock-upscale-ep02",
                "job_type": "upscale",
                "started_at": now_str,
                "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "stage": stage_val,
                "stage_detail": f"Mock TensorRT 10.16 FP16 Zero-Disk 4K ({completed_chunks + 1}/{total_chunks} chunks)",
                "model_id": "animevideov3",
                "runtime_backend": "tensorrt",
                "zero_disk": True,
                "total_frames": total_frames,
                "unique_frames": 18240,
                "done_frames": done_frames,
                "total_chunks": total_chunks,
                "completed_chunks": completed_chunks,
                "chunks": chunks,
                "resumed_chunks": [0, 1],
                "active_chunk": completed_chunks,
                "steady_fps": 41.2,
                "latest_fps": latest_fps,
                "effective_fps": 41.5,
                "eta_seconds": max(15, 620 - n * 5),
            },
            "checkpoints": [],
            "job": _mock_get_state(),
        }

    def _mock_logs_with_cursor(max_lines: int = 120) -> tuple[List[str], Dict[str, Any]]:
        with mock_lock:
            step = tick_counter["n"]
        chunk_num = min(total_chunks, 4 + step)
        lines = [
            "[mock] Loaded TensorRT 10.16 FP16 engine: animevideov3 (batch=8, zero_disk=True)",
            "[mock] Dedup DIS plan: 18240/34560 unique frames (-47.2% savings)",
            f"[mock] Chunk {chunk_num}/12: 41.2 fps | p010le Main10 | VRAM 6.0/12.0 GB",
            "[mock] Subtrans EP01 completed offline QA: pronoun_consistency=100% (0 CJK leaks)",
        ][-max_lines:]
        return lines, {"path": "mock://telemetry.log", "prefix_len": 1, "sig": "mock", "offset": step}

    def _mock_incremental_logs(cursor: Optional[Dict[str, Any]]) -> tuple[List[str], Dict[str, Any]]:
        with mock_lock:
            step = tick_counter["n"]
        cur = cursor if isinstance(cursor, dict) else {}
        prev_off = int(cur.get("offset") or 0) if cur.get("path") == "mock://telemetry.log" else 0
        if step <= prev_off:
            return [], {
                "path": "mock://telemetry.log",
                "prefix_len": 1,
                "sig": "mock",
                "offset": prev_off,
                "reset": False,
            }
        chunk_num = min(total_chunks, 4 + step)
        line = f"[mock] SSE tick #{step}: chunk {chunk_num}/12 @ 41.{step % 6} fps | temp 62C"
        return [line], {"path": "mock://telemetry.log", "prefix_len": 1, "sig": "mock", "offset": step, "reset": False}

    orig_poll_tick_sync = telemetry_hub._poll_tick_sync

    def _mock_poll_tick_sync() -> Dict[str, Any]:
        with mock_lock:
            tick_counter["n"] += 1
        return orig_poll_tick_sync()

    def _mock_pause_graceful() -> Dict[str, Any]:
        with mock_lock:
            mock_runtime["pause_pending"] = True
        return {"ok": True, "graceful_pause_pending": True, "mock": True}

    def _mock_clear_pause() -> Dict[str, Any]:
        with mock_lock:
            mock_runtime["pause_pending"] = False
        return {"ok": True, "graceful_pause_pending": False, "mock": True}

    def _mock_abort_job() -> Dict[str, Any]:
        with mock_lock:
            mock_runtime["aborted"] = True
            mock_runtime["pause_pending"] = False
        return {
            "ok": True,
            "aborted_pid": mock_active_job["pid"],
            "status": "aborted",
            "mock": True,
            "message": f"Đã dừng tiến trình Mock Web Job (PID={mock_active_job['pid']}).",
        }

    with telemetry_hub._log_cursor_lock:
        telemetry_hub._last_log_cursor = {
            "path": "mock://telemetry.log",
            "prefix_len": 1,
            "sig": "mock",
            "offset": 0,
        }
    telemetry_hub.read_pipeline_state = _mock_read_pipeline_state  # type: ignore[method-assign]
    telemetry_hub.read_recent_logs_with_cursor = _mock_logs_with_cursor  # type: ignore[method-assign]
    telemetry_hub._read_incremental_logs_from_cursor = _mock_incremental_logs  # type: ignore[method-assign]
    telemetry_hub._poll_tick_sync = _mock_poll_tick_sync  # type: ignore[method-assign]
    return {
        "pause_graceful": _mock_pause_graceful,
        "clear_pause": _mock_clear_pause,
        "abort": _mock_abort_job,
    }


def create_app(
    workspace_root: Optional[Path]=None,
    gpu_query_fn: Optional[Callable[[], Dict[str, Any]]]=None,
    popen_factory: Optional[Callable[..., Any]]=None,
    open_media_fn: Optional[Callable[[Path, bool], str]]=None,
    frame_extract_fn: Optional[Callable[[Path, float, Path], None]]=None,
    allow_lan: bool=False,
    allow_lan_control: bool=False,
    mock_mode: bool=False,
) -> FastAPI:
    """Create the FastAPI monitoring app bound to workspace_root."""
    root = Path(workspace_root or Path(__file__).resolve().parent.parent).resolve()
    is_mock = bool(mock_mode)
    effective_gpu_query = gpu_query_fn if gpu_query_fn is not None else (_build_mock_gpu_query() if is_mock else None)
    fs_browser = SafeLocalFSBrowser(root)
    job_manager = JobManager(root, popen_factory=popen_factory)
    telemetry_hub = TelemetryHub(
        root,
        gpu_query_fn=effective_gpu_query,
        job_state_provider=job_manager.get_state,
        history_size=120,
    )
    mock_controls = _enable_mock_studio_state(job_manager, telemetry_hub) if is_mock else None
    media_opener = open_media_fn or _default_open_media
    extractor = frame_extract_fn or _default_extract_frame

    def _cleanup_web_compare_dir() -> None:
        compare_dir = root / "scratch" / "web_compare"
        if not compare_dir.is_dir():
            return
        try:
            for png in compare_dir.glob("*.png"):
                try:
                    png.unlink()
                except OSError:
                    pass
        except OSError:
            pass

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        _cleanup_web_compare_dir()
        try:
            yield
        finally:
            _cleanup_web_compare_dir()
            await telemetry_hub.shutdown()

    if not is_mock:
        _cleanup_web_compare_dir()

    app = FastAPI(
        title="AI Video Upscaler & Subtrans Studio",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=_lifespan,
    )
    app.state.workspace_root = root
    app.state.fs_browser = fs_browser
    app.state.job_manager = job_manager
    app.state.telemetry_hub = telemetry_hub
    app.state.allow_lan = bool(allow_lan)
    app.state.allow_lan_control = bool(allow_lan_control)
    app.state.mock_mode = is_mock

    def _apply_security_headers(resp: Any) -> Any:
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        resp.headers.setdefault("Content-Security-Policy", _CSP_HEADER_VALUE)
        return resp

    def _sanitize_json_value(val: Any) -> Any:
        if isinstance(val, float) and not math.isfinite(val):
            return str(val)
        if isinstance(val, dict):
            return {str(k): _sanitize_json_value(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_sanitize_json_value(item) for item in val]
        if isinstance(val, BaseException):
            return str(val)
        return val

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(_request: Any, exc: RequestValidationError) -> JSONResponse:
        return _apply_security_headers(
            JSONResponse(status_code=422, content={"detail": _sanitize_json_value(exc.errors())})
        )

    @app.middleware("http")
    async def _security_headers_middleware(request: Any, call_next: Callable[[Any], Any]) -> Any:
        client_obj = getattr(request, "client", None)
        client_host = (getattr(client_obj, "host", "") or "").lower()
        host_name, host_port, host_ok = _parse_host_header(request.headers.get("host"))
        if not host_ok or not _is_allowed_hostname(host_name, allow_lan=allow_lan, client_host=client_host):
            return _apply_security_headers(
                JSONResponse(
                    status_code=403,
                    content={"detail": "Từ chối kết nối: Host header không thuộc localhost hoặc mạng LAN hợp lệ."},
                )
            )
        if not _validate_origin_header(
            request.headers.get("origin"),
            host_name=host_name,
            host_port=host_port,
            allow_lan=allow_lan,
            client_host=client_host,
        ):
            return _apply_security_headers(
                JSONResponse(
                    status_code=403,
                    content={"detail": "Từ chối kết nối: Origin không khớp hoặc không hợp lệ (anti-DNS-rebinding)."},
                )
            )
        for _, raw_qv in request.query_params.multi_items():
            if "\x00" in raw_qv:
                return _apply_security_headers(
                    JSONResponse(
                        status_code=400,
                        content={"detail": "Tham số truy vấn không được chứa ký tự NUL (\\x00)."},
                    )
                )
        is_loopback = _is_loopback_request(client_host, host_name)
        if allow_lan and not allow_lan_control and not is_loopback:
            req_path = request.url.path or ""
            has_custom_outdir = bool((request.query_params.get("output_dir") or "").strip()) and req_path in {
                "/api/preview/candidates",
                "/api/audits",
            }
            if (
                request.method not in {"GET", "HEAD", "OPTIONS"}
                or req_path == "/api/fs/browse"
                or req_path.startswith("/api/fs/")
                or has_custom_outdir
            ):
                return _apply_security_headers(
                    JSONResponse(
                        status_code=403,
                        content={
                            "detail": (
                                "Chế độ --lan mặc định chỉ cho phép theo dõi (read-only). "
                                "Khởi chạy máy chủ với --lan-control để cho phép điều khiển từ mạng LAN."
                            )
                        },
                    )
                )
        response = await call_next(request)
        return _apply_security_headers(response)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        index_file = STATIC_DIR / "index.html"
        if not index_file.is_file():
            raise HTTPException(status_code=404, detail="Chưa tìm thấy giao diện web/static/index.html.")
        return FileResponse(str(index_file), media_type="text/html; charset=utf-8")

    # 1. LIVE MONITOR & TELEMETRY SSE
    @app.get("/api/status")
    def api_status() -> Dict[str, Any]:
        return telemetry_hub.build_snapshot()

    @app.get("/api/events")
    async def api_events(once: int = Query(default=0, ge=0, le=1)) -> StreamingResponse:
        headers = {
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            telemetry_hub.stream_events(once=bool(once)),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.get("/api/logs/recent")
    def api_recent_logs(lines: int = Query(default=120, ge=10, le=1000)) -> Dict[str, Any]:
        return {"lines": telemetry_hub.read_recent_logs(max_lines=lines)}

    # 2. FILE SYSTEM BROWSER, QUEUE & JOB CONTROLS
    @app.get("/api/fs/browse")
    def api_fs_browse(
        path: Optional[str] = Query(default=None),
        include_subtitles: bool = Query(default=False),
    ) -> Dict[str, Any]:
        try:
            return fs_browser.browse(raw_path=path, include_subtitles=include_subtitles)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/queue/preview-schedule")
    def api_preview_schedule(req: SchedulePreviewRequest) -> Dict[str, Any]:
        try:
            _resolve_dir_from_raw(root, req.output_dir, strict_secret=True)
            _validate_output_dir(root, req.output_dir, field="output_dir")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not req.inputs:
            try:
                norm_dl = _validate_deadline_str(req.deadline)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "total_episodes": 0,
                "accepted_count": 0,
                "already_done_count": 0,
                "skipped_count": 0,
                "total_eta_minutes": 0.0,
                "deadline_str": norm_dl or "",
                "episodes": [],
            }
        try:
            return job_manager.compute_schedule_preview(
                inputs=req.inputs,
                deadline_str=req.deadline,
                allow_partial=req.allow_partial,
                model_override=req.model,
                dedup_override=req.dedup_policy,
                grain_restore=req.grain_restore,
                zero_disk=req.zero_disk,
                use_tensorrt=req.use_tensorrt,
                subs_after_upscale=req.subs_after_upscale,
                output_dir=req.output_dir,
                force=req.force,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _build_mock_job_response(job_type: str, queue_items: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": True,
            "mock": True,
            "job": {
                "job_id": f"mock-{job_type}",
                "job_type": job_type,
                "status": "running",
                "pid": 90420,
                "started_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "finished_at": None,
                "return_code": None,
                "queue_items": queue_items,
                "options": options,
                "mock": True,
            },
        }

    @app.post("/api/jobs/start")
    def api_start_upscale(req: StartUpscaleRequest) -> Dict[str, Any]:
        try:
            _resolve_dir_from_raw(root, req.output_dir, strict_secret=True)
            _resolve_dir_from_raw(root, req.glossary, strict_secret=True)
            _validate_output_dir(root, req.output_dir, field="output_dir")
            _validate_output_dir(root, req.glossary, field="glossary")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if is_mock and popen_factory is None:
            if not req.inputs:
                raise HTTPException(status_code=400, detail="Vui lòng chọn ít nhất một file video hoặc thư mục nguồn.")
            return _build_mock_job_response("upscale", list(req.inputs), req.model_dump())
        try:
            job = job_manager.start_upscale_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/preview")
    def api_start_preview(req: StartPreviewRequest) -> Dict[str, Any]:
        try:
            _resolve_dir_from_raw(root, req.output_dir, strict_secret=True)
            _validate_output_dir(root, req.output_dir, field="output_dir")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if is_mock and popen_factory is None:
            if not (req.video_path or "").strip():
                raise HTTPException(status_code=400, detail="Thiếu video_path.")
            return _build_mock_job_response("preview", [req.video_path], req.model_dump())
        try:
            job = job_manager.start_preview_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/subtrans")
    def api_start_subtrans(req: StartSubtransRequest) -> Dict[str, Any]:
        """Start standalone subtitle translation job."""
        try:
            _resolve_dir_from_raw(root, req.output_dir, strict_secret=True)
            _resolve_dir_from_raw(root, req.glossary, strict_secret=True)
            _validate_output_dir(root, req.output_dir, field="output_dir")
            _validate_output_dir(root, req.glossary, field="glossary")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if is_mock and popen_factory is None:
            mock_targets = [strip_wrapping_quotes(str(x)) for x in (req.inputs or []) if strip_wrapping_quotes(str(x))]
            if not mock_targets and req.input_path:
                cleaned_single = strip_wrapping_quotes(req.input_path)
                if cleaned_single:
                    mock_targets = [cleaned_single]
            if not mock_targets:
                raise HTTPException(status_code=400, detail="Thiếu input_path.")
            return _build_mock_job_response("subtrans", mock_targets, req.model_dump())
        try:
            job = job_manager.start_subtrans_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/subs-pending")
    def api_start_subs_pending(req: StartSubsPendingRequest) -> Dict[str, Any]:
        if is_mock and popen_factory is None:
            return _build_mock_job_response("subs_pending", ["state/pending_subs.json"], req.model_dump())
        try:
            job = job_manager.start_subs_pending_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/build-trt")
    def api_start_build_trt(req: BuildTrtRequest) -> Dict[str, Any]:
        models_reg, _, _ = fs_browser._load_routing_tables()
        mid = req.model_id.strip()
        if not mid or (mid.lower() not in {"all", AUTO_PARETO_MODEL_ID} and mid not in models_reg):
            raise HTTPException(
                status_code=400,
                detail=f"Mô hình '{mid}' không hợp lệ hoặc không có trong models/models.yaml.",
            )
        if (req.width is None) != (req.height is None):
            raise HTTPException(
                status_code=400,
                detail="Cần cung cấp đồng thời cả 'width' và 'height' khi chỉ định độ phân giải biên dịch TRT.",
            )
        if is_mock and popen_factory is None:
            target_lbl = f"{mid} ({req.width}×{req.height})" if (req.width and req.height) else mid
            return _build_mock_job_response("build_trt", [target_lbl], req.model_dump())
        try:
            job = job_manager.start_build_trt_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/clean-temp")
    def api_start_clean_temp(req: CleanTempRequest) -> Dict[str, Any]:
        if req.output_dir is not None and str(req.output_dir).strip():
            pre_raw = Path(os.path.expanduser(strip_wrapping_quotes(str(req.output_dir))))
            if has_reparse_or_symlink_in_chain(pre_raw, workspace_root=root):
                raise HTTPException(
                    status_code=403,
                    detail="Từ chối dọn tệp tạm qua thư mục symlink hoặc junction reparse point.",
                )
        elif has_reparse_or_symlink_in_chain(root / "out", workspace_root=root):
            raise HTTPException(
                status_code=403,
                detail="Từ chối dọn tệp tạm qua thư mục symlink hoặc junction reparse point.",
            )
        try:
            _resolve_dir_from_raw(root, req.output_dir, strict_secret=True)
            _validate_output_dir(root, req.output_dir, field="output_dir")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if is_mock and popen_factory is None:
            return _build_mock_job_response("clean_temp", [req.output_dir or "out/"], req.model_dump())
        try:
            job = job_manager.start_clean_temp_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/pause-graceful")
    def api_pause_graceful() -> Any:
        if is_mock and popen_factory is None and mock_controls is not None:
            return mock_controls["pause_graceful"]()
        try:
            result = job_manager.request_graceful_pause()
            if not result.get("ok", True):
                return JSONResponse(status_code=409, content=result)
            return result
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/clear-pause")
    def api_clear_pause() -> Dict[str, Any]:
        if is_mock and popen_factory is None and mock_controls is not None:
            return mock_controls["clear_pause"]()
        return job_manager.clear_graceful_pause()

    @app.post("/api/jobs/abort")
    def api_abort_job() -> Dict[str, Any]:
        if is_mock and popen_factory is None and mock_controls is not None:
            return mock_controls["abort"]()
        return job_manager.abort_job()

    @app.post("/api/checkpoints/delete")
    def api_delete_checkpoint(req: DeleteCheckpointRequest) -> Dict[str, Any]:
        cleaned = (req.path or "").strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="Thiếu đường dẫn thư mục checkpoint (.temp_*).")
        if is_mock and popen_factory is None:
            return {"ok": True, "deleted": cleaned, "freed_mb": 0.0, "mock": True}

        jstate = job_manager.get_state()
        if jstate.get("status") in {"running", "pausing"}:
            raise HTTPException(
                status_code=409,
                detail="Không thể xóa checkpoint khi đang có tiến trình chạy. Hãy tạm dừng hoặc dừng tác vụ trước.",
            )
        lock_state = check_active_gpu_lock(root)
        if lock_state.get("locked"):
            raise HTTPException(
                status_code=409,
                detail=f"GPU đang bận xử lý tiến trình khác (PID={lock_state.get('pid')}); từ chối xóa checkpoint.",
            )

        cleaned = strip_wrapping_quotes(cleaned)
        if not cleaned or "\x00" in cleaned or is_unc_path(cleaned):
            raise HTTPException(status_code=400, detail="Đường dẫn thư mục checkpoint (.temp_*) không hợp lệ.")

        user_cand = Path(os.path.expanduser(cleaned))
        if user_cand.drive and not user_cand.is_absolute():
            raise HTTPException(status_code=400, detail="Đường dẫn tương đối theo ổ đĩa không hợp lệ.")
        orig_was_abs = user_cand.is_absolute()
        raw_cand = user_cand if orig_was_abs else (root / user_cand)
        raw_abs = Path(os.path.abspath(str(raw_cand)))

        def _is_link_or_reparse(p: Path) -> bool:
            if p.is_symlink() or os.path.islink(str(p)):
                return True
            try:
                st = p.lstat()
                return bool(getattr(st, "st_file_attributes", 0) & 0x0400)
            except OSError:
                return False

        root_norm = os.path.normcase(os.path.abspath(str(root)))

        def _is_root_or_root_ancestor(p: Path) -> bool:
            pn = os.path.normcase(os.path.abspath(str(p))).rstrip(os.sep)
            rn = root_norm.rstrip(os.sep)
            return pn == rn or rn.startswith(pn + os.sep)

        has_link_ancestor = False
        if ".." in user_cand.parts:
            acc = Path(user_cand.anchor) if orig_was_abs else root
            rel_parts = user_cand.parts[1:] if orig_was_abs else user_cand.parts
            for part in rel_parts:
                if part == "..":
                    acc = acc.parent
                elif part != ".":
                    acc = acc / part
                    if not _is_root_or_root_ancestor(acc) and _is_link_or_reparse(acc):
                        has_link_ancestor = True
                        break
        cur_par = raw_abs.parent
        while not has_link_ancestor:
            cur_norm = os.path.normcase(str(cur_par))
            if cur_norm == root_norm or cur_par.parent == cur_par or _is_root_or_root_ancestor(cur_par):
                break
            if _is_link_or_reparse(cur_par):
                has_link_ancestor = True
                break
            cur_par = cur_par.parent

        if (
            _is_link_or_reparse(raw_abs)
            or has_link_ancestor
            or not raw_abs.name.startswith(".temp_")
            or len(raw_abs.name) <= len(".temp_")
        ):
            raise HTTPException(status_code=400, detail="Chỉ cho phép xóa thư mục checkpoint .temp_* hợp lệ.")
        cand = raw_abs.resolve()
        if (
            _is_link_or_reparse(cand)
            or not cand.name.startswith(".temp_")
            or len(cand.name) <= len(".temp_")
        ):
            raise HTTPException(status_code=400, detail="Chỉ cho phép xóa thư mục checkpoint .temp_* hợp lệ.")
        if not cand.is_dir():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy thư mục checkpoint: {cand}")

        known_outs = _collect_known_output_dirs(root, job_manager)
        known_cps = scan_temp_checkpoints(root, extra_dirs=known_outs)
        norm_cand = os.path.normcase(str(cand))

        def _norm_cp_path(raw_cp_p: str) -> str:
            if not raw_cp_p:
                return ""
            try:
                return os.path.normcase(str(Path(raw_cp_p).resolve()))
            except OSError:
                return os.path.normcase(os.path.abspath(raw_cp_p))

        matched_cp = next(
            (cp for cp in known_cps if _norm_cp_path(str(cp.get("path") or "")) == norm_cand),
            None,
        )
        has_cp_markers = any(
            (cand / marker).exists() for marker in ("raw_segs", "upscaled_segs", "manifest.json", "plan.json")
        )
        if matched_cp is None or not has_cp_markers:
            raise HTTPException(
                status_code=403,
                detail="Từ chối xóa: thư mục không thuộc danh sách checkpoint .temp_* hợp lệ của dự án.",
            )

        freed_mb = float(matched_cp.get("size_mb") or 0.0)
        try:
            shutil.rmtree(cand)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Không thể xóa checkpoint: {exc}") from exc

        try:
            remaining_cps = scan_temp_checkpoints(root, extra_dirs=known_outs)
            live_file = root / "logs" / "live_status.json"
            if not remaining_cps and live_file.is_file() and not check_active_gpu_lock(root).get("locked"):
                ldata = json.loads(live_file.read_text(encoding="utf-8"))
                if isinstance(ldata, dict):
                    lpid = _coerce_pid(ldata.get("pid"))
                    lpid_alive = False
                    if lpid is not None:
                        import psutil

                        lpid_alive = psutil.pid_exists(lpid)
                    if not lpid_alive and str(ldata.get("stage") or "") not in {"idle", "completed", "error"}:
                        ldata = dict(ldata)
                        ldata["stage"] = "idle"
                        ldata["stage_detail"] = "Đã dọn sạch checkpoint — hệ thống sẵn sàng chờ lệnh."
                        ldata["completed_chunks"] = 0
                        ldata["total_chunks"] = 0
                        ldata["chunks"] = []
                        ldata["resumed_chunks"] = []
                        ldata["active_chunk"] = None
                        ldata["active_chunk_fraction"] = 0.0
                        ldata["eta_seconds"] = None
                        tmp_lf = live_file.with_suffix(".json.tmp")
                        tmp_lf.write_text(json.dumps(ldata, ensure_ascii=False, indent=2), encoding="utf-8")
                        os.replace(tmp_lf, live_file)
        except Exception:
            pass
        return {"ok": True, "deleted": str(cand), "freed_mb": freed_mb}

    @app.post("/api/open-media")
    def api_open_media(req: OpenMediaRequest) -> Dict[str, Any]:
        raw_unwrapped = unwrap_outer_quotes_preserving_trailing_space(req.path)
        cleaned = strip_wrapping_quotes(req.path)
        if not cleaned or "\x00" in cleaned or is_unc_path(cleaned):
            raise HTTPException(status_code=400, detail="Đường dẫn không hợp lệ.")
        cand = Path(os.path.expanduser(cleaned))
        if cand.drive and not cand.is_absolute():
            raise HTTPException(status_code=400, detail="Đường dẫn tương đối theo ổ đĩa không hợp lệ.")
        if sys.platform == "win32" and not cand.drive and cand.root:
            raise HTTPException(status_code=400, detail="Đường dẫn gốc thiếu ký tự ổ đĩa không hợp lệ.")
        try:
            validate_win32_path_hygiene(Path(os.path.expanduser(raw_unwrapped)), field="path", raw_str=raw_unwrapped)
            cand = (cand if cand.is_absolute() else (root / cand)).resolve()
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if is_secret_filename(cand.name):
            raise HTTPException(status_code=403, detail="Từ chối truy cập tệp nhạy cảm.")
        if not cand.is_file():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy tệp media: {cand}")
        if cand.suffix.lower() not in ALLOWED_OPEN_MEDIA_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Chỉ cho phép mở tệp video hoặc phụ đề hợp lệ (nhận được '{cand.suffix}').",
            )
        try:
            launcher = media_opener(cand, req.prefer_mpv)
            return {"ok": True, "path": str(cand), "launcher": launcher}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def _iter_out_search_dirs(out_dir: Path) -> List[Path]:
        """Collect out_dir, out_dir/audit, and valid level-1 subdirectories."""
        if not out_dir.is_dir():
            return []
        dirs: List[Path] = [out_dir]
        audit_sub = out_dir / "audit"
        if audit_sub.is_dir():
            dirs.append(audit_sub)
        try:
            for sub in sorted(out_dir.iterdir(), key=lambda p: p.name.lower()):
                if (
                    sub.is_dir()
                    and not sub.name.startswith(".")
                    and sub.name.lower() not in {"audit", "cast"}
                ):
                    dirs.append(sub)
        except OSError:
            pass
        return dirs

    # 3. PREVIEW 1080p vs 4K SPLIT-COMPARE
    @app.get("/api/preview/candidates")
    def api_preview_candidates(output_dir: Optional[str] = Query(default=None)) -> Dict[str, Any]:
        try:
            out_dirs = _collect_known_output_dirs(root, job_manager, extra_dir=output_dir)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        pairs: List[Dict[str, Any]] = []
        if not out_dirs:
            return {"candidates": pairs}

        default_out = (root / "out").resolve()
        out_dir_set = {os.path.normcase(str(d.resolve())) for d in out_dirs}
        known_source_dirs = _collect_known_source_dirs(root, out_dirs, job_manager)
        seen_src_dirs = {os.path.normcase(str(d.resolve())) for d in known_source_dirs}

        def _register_source_dir(cand_dir: Path) -> None:
            try:
                rp = cand_dir.resolve()
                if not rp.is_dir():
                    return
                norm = os.path.normcase(str(rp))
            except OSError:
                return
            if norm not in seen_src_dirs:
                seen_src_dirs.add(norm)
                known_source_dirs.append(rp)

        # Index audit records to resolve canonical source_file and output_file across all out_dirs
        audit_source_map: Dict[str, str] = {}
        audited_outputs: Dict[str, Path] = {}
        for base_out in out_dirs:
            for audit_dir in _iter_out_search_dirs(base_out):
                for af in audit_dir.glob("*.audit.json"):
                    try:
                        adata = json.loads(af.read_text(encoding="utf-8"))
                        if not isinstance(adata, dict):
                            continue
                        out_f = str(adata.get("output_file") or "").strip()
                        raw_orig = str(adata.get("original_source_file") or "").strip()
                        raw_src = str(adata.get("source_file") or "").strip()

                        valid_src_str = ""
                        for raw_cand in (raw_orig, raw_src):
                            if not raw_cand or ".preview_clip_" in raw_cand:
                                continue
                            src_p = Path(os.path.expanduser(raw_cand))
                            if not src_p.is_absolute():
                                src_p = (root / src_p).resolve()
                            else:
                                src_p = src_p.resolve()
                            if src_p.parent.is_dir():
                                _register_source_dir(src_p.parent)
                            if src_p.is_file():
                                valid_src_str = str(src_p)
                                break

                        if out_f:
                            out_p = Path(os.path.expanduser(out_f))
                            if not out_p.is_absolute():
                                out_p = (root / out_p).resolve()
                            else:
                                out_p = out_p.resolve()
                            if out_p.is_file() and out_p.suffix.lower() in VIDEO_EXTENSIONS:
                                norm_out = os.path.normcase(str(out_p))
                                audited_outputs[norm_out] = out_p
                                if valid_src_str:
                                    audit_source_map[norm_out] = valid_src_str
                        if valid_src_str:
                            audit_source_map[af.name[:-len(".audit.json")].lower()] = valid_src_str
                    except Exception:
                        pass

        candidate_map: Dict[str, Path] = dict(audited_outputs)
        for base_out in out_dirs:
            scan_video_dirs = [
                d for d in _iter_out_search_dirs(base_out) if d.name.lower() not in {"audit", "cast"}
            ]
            for vdir in scan_video_dirs:
                try:
                    for p in vdir.iterdir():
                        if (
                            p.is_file()
                            and p.suffix.lower() in VIDEO_EXTENSIONS
                            and not p.name.startswith(".")
                            and ("_4k" in p.stem.lower())
                        ):
                            rp = p.resolve()
                            candidate_map[os.path.normcase(str(rp))] = rp
                except OSError:
                    continue

        def _safe_mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        upscaled_files = sorted(
            candidate_map.values(),
            key=_safe_mtime,
            reverse=True,
        )

        ext_order = [".mkv", ".mp4", ".webm", ".avi", ".mov", ".ts", ".m2ts"]
        all_exts = [*ext_order, *(e for e in sorted(VIDEO_EXTENSIONS) if e not in ext_order)]

        for up_file in upscaled_files[:40]:
            try:
                up_st = up_file.stat()
            except OSError:
                continue
            stem = up_file.stem
            up_resolved = up_file.resolve()
            norm_up = os.path.normcase(str(up_resolved))
            src_path = audit_source_map.get(norm_up) or audit_source_map.get(stem.lower()) or ""

            # Infer base stem when output filename is a preview clip
            preview_match = re.match(r"^(.*)_preview_(\d+)s_4K(?:_animejanai)?$", stem, flags=re.IGNORECASE)
            is_preview_clip = bool(preview_match)
            preview_seconds = int(preview_match.group(2)) if preview_match else 0
            raw_base = (
                preview_match.group(1)
                if preview_match
                else re.sub(r"_4K(?:_animejanai)?$", "", stem, flags=re.IGNORECASE)
            )

            if (not src_path or ".preview_clip_" in src_path or not Path(src_path).is_file()) and raw_base:
                src_path = ""
                ep_search_dirs: List[Path] = list(known_source_dirs)
                parent_norm = os.path.normcase(str(up_resolved.parent))
                if parent_norm not in seen_src_dirs:
                    ep_search_dirs.append(up_resolved.parent)
                for search_base in ep_search_dirs:
                    for ext in all_exts:
                        cand_src = search_base / f"{raw_base}{ext}"
                        if cand_src.is_file() and cand_src.resolve() != up_resolved:
                            src_path = str(cand_src.resolve())
                            break
                    if src_path:
                        break

            up_parent = up_resolved.parent
            if up_parent == default_out:
                rel_label = up_file.name
            elif os.path.normcase(str(up_parent)) in out_dir_set:
                rel_label = f"{up_parent.name}/{up_file.name}"
            else:
                rel_label = f"{up_parent.name}/{up_file.name}"

            pairs.append({
                "name": rel_label,
                "upscaled_path": str(up_resolved),
                "source_path": src_path if (src_path and Path(src_path).is_file()) else "",
                "is_preview": is_preview_clip,
                "preview_seconds": preview_seconds,
                "size_mb": round(up_st.st_size / (1024 * 1024), 2),
                "mtime": datetime.fromtimestamp(up_st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })

        return {"candidates": pairs}

    @app.post("/api/preview/compare-frames")
    def api_compare_frames(req: CompareFramesRequest) -> Dict[str, Any]:
        raw_src_unw = unwrap_outer_quotes_preserving_trailing_space(req.source_path)
        raw_up_unw = unwrap_outer_quotes_preserving_trailing_space(req.upscaled_path)
        raw_src = strip_wrapping_quotes(req.source_path)
        raw_up = strip_wrapping_quotes(req.upscaled_path)
        if not raw_src or not raw_up or is_unc_path(raw_src) or is_unc_path(raw_up):
            raise HTTPException(
                status_code=400,
                detail="Vui lòng cung cấp đầy đủ đường dẫn video gốc (1080p) và video thành phẩm (4K).",
            )
        src_p = Path(os.path.expanduser(raw_src))
        up_p = Path(os.path.expanduser(raw_up))
        if (src_p.drive and not src_p.is_absolute()) or (up_p.drive and not up_p.is_absolute()):
            raise HTTPException(status_code=400, detail="Đường dẫn tương đối theo ổ đĩa không hợp lệ.")
        if sys.platform == "win32" and ((not src_p.drive and src_p.root) or (not up_p.drive and up_p.root)):
            raise HTTPException(status_code=400, detail="Đường dẫn gốc thiếu ký tự ổ đĩa không hợp lệ.")
        try:
            validate_win32_path_hygiene(Path(os.path.expanduser(raw_src_unw)), field="source_path", raw_str=raw_src_unw)
            validate_win32_path_hygiene(Path(os.path.expanduser(raw_up_unw)), field="upscaled_path", raw_str=raw_up_unw)
            src_p = (src_p if src_p.is_absolute() else (root / src_p)).resolve()
            up_p = (up_p if up_p.is_absolute() else (root / up_p)).resolve()
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if is_secret_filename(src_p.name) or is_secret_filename(up_p.name):
            raise HTTPException(status_code=403, detail="Từ chối truy cập tệp nhạy cảm.")
        if not src_p.is_file():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy video gốc 1080p: {src_p}")
        if not up_p.is_file():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy video 4K đối chiếu: {up_p}")
        if src_p.suffix.lower() not in VIDEO_EXTENSIONS or up_p.suffix.lower() not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Định dạng file video không hợp lệ.")

        src_meta = _probe_duration_and_dims(src_p)
        up_meta = _probe_duration_and_dims(up_p)

        up_fps = max(1.0, float(up_meta.get("fps") or 23.976))
        up_tail_guard = max(0.12, round(1.5 / up_fps, 3))
        up_ts = max(0.0, float(req.timestamp_sec))
        if up_meta["duration"] > 0 and up_ts > max(0.0, up_meta["duration"] - up_tail_guard):
            up_ts = max(0.0, round(up_meta["duration"] - up_tail_guard, 3))

        if req.source_offset_sec is not None:
            offset_s = float(req.source_offset_sec)
        else:
            m_prev = re.match(r"^(.*)_preview_(\d+)s_4K(?:_animejanai)?$", up_p.stem, flags=re.IGNORECASE)
            if m_prev:
                prev_dur = int(m_prev.group(2))
                recorded_start = _lookup_preview_start_sec(root, src_p, up_p)
                if recorded_start is not None:
                    if src_meta["duration"] > 0:
                        offset_s = min(
                            max(0.0, recorded_start),
                            max(0.0, float(src_meta["duration"]) - float(prev_dur)),
                        )
                    else:
                        offset_s = max(0.0, recorded_start)
                elif src_meta["duration"] > 0:
                    max_dur = max(1, int(round(src_meta["duration"])))
                    offset_s = (
                        0.0
                        if prev_dur >= max_dur
                        else float(max(0, min(int(src_meta["duration"] * 0.3), max_dur - prev_dur)))
                    )
                else:
                    offset_s = 0.0
            else:
                offset_s = 0.0

        src_ts = max(0.0, up_ts + offset_s)
        src_fps = max(1.0, float(src_meta.get("fps") or up_fps))
        src_tail_guard = max(0.12, round(1.5 / src_fps, 3))
        if src_meta["duration"] > 0 and src_ts > max(0.0, src_meta["duration"] - src_tail_guard):
            src_ts = max(0.0, round(src_meta["duration"] - src_tail_guard, 3))
            up_ts = max(0.0, round(src_ts - offset_s, 3))
            if up_meta["duration"] > 0 and up_ts > max(0.0, up_meta["duration"] - up_tail_guard):
                up_ts = max(0.0, round(up_meta["duration"] - up_tail_guard, 3))

        compare_dir = root / "scratch" / "web_compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            f"{src_p}|{src_p.stat().st_mtime_ns}|{up_p}|{up_p.stat().st_mtime_ns}|{src_ts:.3f}|{up_ts:.3f}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]

        src_png = compare_dir / f"src_{digest}.png"
        up_png = compare_dir / f"up_{digest}.png"

        def _has_valid_png(png_p: Path) -> bool:
            try:
                return png_p.is_file() and png_p.stat().st_size > 0
            except OSError:
                return False

        created_src_png = False
        created_up_png = False
        try:
            if not _has_valid_png(src_png):
                created_src_png = True
                extractor(src_p, src_ts, src_png)
                if not _has_valid_png(src_png):
                    raise RuntimeError(f"Tệp khung hình trích xuất rỗng: '{src_png.name}'.")
            if not _has_valid_png(up_png):
                created_up_png = True
                extractor(up_p, up_ts, up_png)
                if not _has_valid_png(up_png):
                    raise RuntimeError(f"Tệp khung hình trích xuất rỗng: '{up_png.name}'.")
        except subprocess.TimeoutExpired as exc:
            if created_src_png:
                with suppress(OSError):
                    src_png.unlink(missing_ok=True)
            if created_up_png:
                with suppress(OSError):
                    up_png.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail="Quá thời gian chờ khi trích xuất khung hình đối chiếu.",
            ) from exc
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            if created_src_png:
                with suppress(OSError):
                    src_png.unlink(missing_ok=True)
            if created_up_png:
                with suppress(OSError):
                    up_png.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            if created_src_png:
                with suppress(OSError):
                    src_png.unlink(missing_ok=True)
            if created_up_png:
                with suppress(OSError):
                    up_png.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Retain at most the 12 most recent finalized temporary comparison frames (6 pairs)
        def _final_png_mtime(p_cand: Path) -> float:
            try:
                return p_cand.stat().st_mtime
            except OSError:
                return 0.0

        try:
            pngs = sorted(
                (
                    p_f
                    for p_f in compare_dir.glob("*.png")
                    if not p_f.name.startswith(".") and not p_f.name.endswith(".tmp.png")
                ),
                key=_final_png_mtime,
                reverse=True,
            )
            for old_f in pngs[12:]:
                with suppress(OSError):
                    old_f.unlink()
        except OSError:
            pass

        return {
            "ok": True,
            "source_frame_url": f"/api/preview/frame/{src_png.name}",
            "upscaled_frame_url": f"/api/preview/frame/{up_png.name}",
            "source_timestamp_sec": round(src_ts, 3),
            "upscaled_timestamp_sec": round(up_ts, 3),
            "source_offset_sec": round(offset_s, 3),
            "source_meta": src_meta,
            "upscaled_meta": up_meta,
        }

    @app.get("/api/preview/frame/{frame_name}")
    def api_get_preview_frame(frame_name: str) -> FileResponse:
        if not re.fullmatch(r"(?:src|up)_[a-f0-9]{12}\.png", frame_name):
            raise HTTPException(status_code=400, detail="Tên khung hình không hợp lệ.")
        frame_file = (root / "scratch" / "web_compare" / frame_name).resolve()
        try:
            is_valid_frame = frame_file.is_file() and frame_file.stat().st_size > 0
        except OSError:
            is_valid_frame = False
        if not is_valid_frame:
            raise HTTPException(status_code=404, detail="Không tìm thấy khung hình trích xuất.")
        return FileResponse(str(frame_file), media_type="image/png")

    # 4. MODELS REGISTRY & SHOWS.YAML CONFIGURATOR
    @app.get("/api/config/models-and-shows")
    def api_models_and_shows() -> Dict[str, Any]:
        models_file = root / "models" / "models.yaml"
        shows_file = root / "shows.yaml"
        trt_cache_dir = root / ".cache" / "trt_engines"

        models_list: List[Dict[str, Any]] = []
        reg: Dict[str, Any] = {}
        if models_file.is_file():
            try:
                raw_m = yaml.safe_load(models_file.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                raw_m = {}
            reg = raw_m.get("models", {}) or {}
            for m_id, info in reg.items():
                if not isinstance(info, dict):
                    continue
                rel_w = str(info.get("weights_path") or "")
                w_path = (root / rel_w).resolve() if rel_w else None
                w_exists = bool(w_path and w_path.is_file())
                w_size_mb = round(w_path.stat().st_size / (1024 * 1024), 2) if w_exists and w_path else 0.0

                arch = info.get("arch") or {}
                arch_type_str = str(arch.get("type") or "Unknown")
                up_scale_int = int(arch.get("upscale", 2) or 2)
                allowed_shas = resolve_allowed_trt_weights_shas(str(m_id), reg, grain_restore="auto")
                trt_engines, stale_trt_engines = inspect_model_trt_cache(
                    trt_cache_dir,
                    str(m_id),
                    expected_weights_sha256=str(info.get("sha256") or ""),
                    allowed_weights_shas=allowed_shas if allowed_shas else None,
                    arch_type=arch_type_str,
                    native_upscale=up_scale_int,
                )
                trt_hw_profiles = extract_trt_hw_profiles(trt_engines)

                med_fps, p5_fps = MODEL_REFERENCE_FPS.get(str(m_id), (10.5, 8.5))
                models_list.append({
                    "id": str(m_id),
                    "name": str(info.get("name") or m_id),
                    "arch_type": arch_type_str,
                    "num_feat": arch.get("num_feat"),
                    "num_conv": arch.get("num_conv"),
                    "native_upscale": up_scale_int,
                    "downsample": str(info.get("downsample") or "none"),
                    "weights_path": rel_w,
                    "weights_ready": w_exists,
                    "weights_size_mb": w_size_mb,
                    "trt_cached": len(trt_engines) > 0,
                    "trt_engine_files": trt_engines,
                    "trt_hw_profiles": trt_hw_profiles,
                    "stale_trt_engines": stale_trt_engines,
                    "sha256_prefix": str(info.get("sha256") or "")[:12],
                    "license": str(info.get("license") or ""),
                    "default": bool(info.get("default", False)),
                    "ref_median_fps": med_fps,
                    "ref_p5_fps": p5_fps,
                })

        shows_text = shows_file.read_text(encoding="utf-8") if shows_file.is_file() else "shows: []\n"
        shows_rules: List[Dict[str, Any]] = []
        try:
            parsed_s = yaml.safe_load(shows_text) or {}
            shows_rules = [r for r in (parsed_s.get("shows") or []) if isinstance(r, dict)]
        except Exception:
            shows_rules = []

        dedup_policies = [
            {"id": k, **v} for k, v in DEDUP_POLICY_PARAMS.items()
        ]
        by_model_id = {m["id"]: m for m in models_list}
        pareto_detail = by_model_id.get("animejanai-hd-v3-compact")
        pareto_fast = by_model_id.get("animejanai-hd-v3-ultracompact")
        pareto_detail_trt = bool(pareto_detail and pareto_detail.get("trt_cached"))
        pareto_fast_trt = bool(pareto_fast and pareto_fast.get("trt_cached"))
        pareto_shared_profiles = sorted(
            set((pareto_detail or {}).get("trt_hw_profiles") or [])
            & set((pareto_fast or {}).get("trt_hw_profiles") or []),
            key=natural_sort_key,
        )
        pareto_med_fps, pareto_p5_fps = MODEL_REFERENCE_FPS.get(AUTO_PARETO_MODEL_ID, (16.5, 13.5))
        routing_modes = [
            {
                "id": AUTO_PARETO_MODEL_ID,
                "name": AUTO_PARETO_MODEL_NAME,
                "detail_model_id": "animejanai-hd-v3-compact",
                "fast_model_id": "animejanai-hd-v3-ultracompact",
                "weights_ready": bool(
                    pareto_detail and pareto_detail.get("weights_ready") and pareto_fast and pareto_fast.get("weights_ready")
                ),
                "trt_cached": bool(pareto_detail_trt and pareto_fast_trt and pareto_shared_profiles),
                "detail_trt_cached": pareto_detail_trt,
                "fast_trt_cached": pareto_fast_trt,
                "trt_hw_profiles": pareto_shared_profiles,
                "ref_median_fps": pareto_med_fps,
                "ref_p5_fps": pareto_p5_fps,
            }
        ]

        return {
            "models": models_list,
            "routing_modes": routing_modes,
            "nvenc_multipass_policies": list(VALID_NVENC_MULTIPASS_POLICIES),
            "grain_restore_policies": list(VALID_GRAIN_RESTORE_POLICIES),
            "shows_yaml": shows_text,
            "shows_rules": shows_rules,
            "dedup_policies": dedup_policies,
        }

    def _validate_shows_yaml_text(yaml_content: str) -> List[Dict[str, Any]]:
        try:
            parsed = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            raise ValueError(f"Lỗi cú pháp YAML trong shows.yaml: {exc}") from exc

        if parsed is None:
            parsed = {"shows": []}
        if not isinstance(parsed, dict) or "shows" not in parsed:
            raise ValueError("Cấu trúc shows.yaml phải chứa khóa gốc 'shows:' dạng danh sách.")

        rules = parsed.get("shows")
        if rules is None:
            return []
        if not isinstance(rules, list):
            raise ValueError("Trường 'shows' phải là một danh sách các rule '- pattern: ...'.")

        models_file = root / "models" / "models.yaml"
        valid_models: set[str] = set()
        if models_file.is_file():
            try:
                raw_m = yaml.safe_load(models_file.read_text(encoding="utf-8")) or {}
                valid_models = {str(k).lower() for k in (raw_m.get("models") or {}).keys()}
            except yaml.YAMLError:
                valid_models = set()

        validated: List[Dict[str, Any]] = []
        for idx, item in enumerate(rules, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Rule thứ #{idx} không hợp lệ (phải là dictionary).")
            pat = str(item.get("pattern") or "").strip()
            if not pat:
                raise ValueError(f"Rule thứ #{idx} thiếu trường 'pattern'.")
            mod = str(item.get("model") or "").strip()
            if mod and valid_models and mod.lower() not in valid_models and mod.lower() != AUTO_PARETO_MODEL_ID:
                raise ValueError(
                    f"Rule '{pat}' trỏ tới model '{mod}' không tồn tại trong models/models.yaml."
                )
            pol = str(item.get("dedup_policy") or "").strip()
            if pol and not is_valid_dedup_policy(pol):
                raise ValueError(
                    f"Rule '{pat}' dùng dedup_policy '{pol}' không hợp lệ "
                    f"(chấp nhận: {', '.join(DEDUP_POLICY_PARAMS.keys())}, dis-<float>, drift-<float>)."
                )
            validated.append(item)
        return validated

    @app.post("/api/config/test-match")
    def api_test_shows_match(req: TestShowsMatchRequest) -> Dict[str, Any]:
        sample = strip_wrapping_quotes(req.sample_path)
        if not sample or is_unc_path(sample):
            raise HTTPException(status_code=400, detail="Vui lòng nhập tên file hoặc đường dẫn mẫu để kiểm thử.")

        if req.yaml_content is not None:
            try:
                rules = _validate_shows_yaml_text(req.yaml_content)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            models_file = root / "models" / "models.yaml"
            reg: Dict[str, Any] = {}
            default_id = "animevideov3"
            if models_file.is_file():
                try:
                    raw_m = yaml.safe_load(models_file.read_text(encoding="utf-8")) or {}
                    reg = raw_m.get("models") or {}
                except yaml.YAMLError:
                    reg = {}
                for k, v in reg.items():
                    if isinstance(v, dict) and v.get("default"):
                        default_id = str(k)
                        break
            lower_to_key = {str(k).lower(): str(k) for k in reg.keys()}

            matched_pat = ""
            chosen_mod = default_id
            chosen_pol = "default"
            src_desc = "Mặc định hệ thống (models.yaml)"
            matched_model_pat = ""
            matched_dedup_pat = ""
            for r in rules:
                if not isinstance(r, dict):
                    continue
                pat = str(r.get("pattern") or "").strip()
                if not pat:
                    continue
                if match_show_rule(sample, pat):
                    m_target = str(r.get("model") or "").strip().lower()
                    if m_target == AUTO_PARETO_MODEL_ID or (m_target and m_target in lower_to_key):
                        chosen_mod = AUTO_PARETO_MODEL_ID if m_target == AUTO_PARETO_MODEL_ID else lower_to_key[m_target]
                        src_desc = f"shows.yaml ('{pat}')"
                        matched_model_pat = pat
                        break
            for r in rules:
                if not isinstance(r, dict):
                    continue
                pat = str(r.get("pattern") or "").strip()
                if not pat:
                    continue
                if match_show_rule(sample, pat):
                    p_target = str(r.get("dedup_policy") or "").strip().lower()
                    if p_target and is_valid_dedup_policy(p_target):
                        chosen_pol = p_target
                        matched_dedup_pat = pat
                        break
            matched_pat = matched_model_pat or matched_dedup_pat

            med_fps, _ = MODEL_REFERENCE_FPS.get(chosen_mod, (10.5, 8.5))
            default_mod_name = AUTO_PARETO_MODEL_NAME if chosen_mod == AUTO_PARETO_MODEL_ID else chosen_mod
            return {
                "sample": sample,
                "model_id": chosen_mod,
                "model_name": str((reg.get(chosen_mod) or {}).get("name") or default_mod_name),
                "dedup_policy": chosen_pol,
                "matched_pattern": matched_pat,
                "resolution_source": src_desc,
                "ref_median_fps": med_fps,
            }

        res = fs_browser.resolve_routing_for_path(sample)
        med_fps, _ = MODEL_REFERENCE_FPS.get(res["model_id"], (10.5, 8.5))
        return {"sample": sample, **res, "ref_median_fps": med_fps}

    @app.put("/api/config/shows")
    def api_save_shows(req: SaveShowsYamlRequest) -> Dict[str, Any]:
        try:
            rules = _validate_shows_yaml_text(req.yaml_content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        shows_path = root / "shows.yaml"
        tmp_path = root / f".shows.yaml.{os.getpid()}.tmp"
        text = req.yaml_content.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
        tmp_path.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp_path, shows_path)
        return {"ok": True, "rules_count": len(rules)}

    # 5. SUBTRANS STUDIO & AUDIT VAULT
    mock_pending_subs: List[Dict[str, Any]] = (
        [{"episode": "[Demo] Anime_EP02 (1080p)", "status": "queued", "subs_quality": "high"}]
        if is_mock
        else []
    )

    def _is_reparse_or_symlink(p_check: Path) -> bool:
        try:
            if (
                p_check.is_symlink()
                or os.path.islink(str(p_check))
                or (hasattr(os.path, "isjunction") and os.path.isjunction(str(p_check)))
            ):
                return True
            if os.path.lexists(str(p_check)):
                st_chk = p_check.lstat()
                if bool(getattr(st_chk, "st_file_attributes", 0) & 0x0400):
                    return True
        except OSError:
            return False
        return False

    @app.get("/api/subtrans/overview")
    def api_subtrans_overview(request: Request) -> Dict[str, Any]:
        now_pt = get_now_pt()
        next_pt, next_vn = get_next_pt_reset(now_pt)
        today_pt_str = now_pt.strftime("%Y-%m-%d")
        reset_seconds = max(0, int((next_pt - now_pt).total_seconds()))

        # Masked API key slot statuses mirroring subtrans/config.py precedence
        with _DOTENV_LOCK:
            key_slots, active_key_count, all_key_slots = _resolve_subtrans_key_slots_all(
                root / ".env", key_id_fn=compute_key_id
            )
        req_client = getattr(request, "client", None)
        req_client_host = (getattr(req_client, "host", "") or "").lower()
        req_host_name, _, _ = _parse_host_header(request.headers.get("host"))
        if not _is_loopback_request(req_client_host, req_host_name):
            for s_entry in all_key_slots:
                s_entry["masked"] = ""
            for s_entry in key_slots:
                s_entry["masked"] = ""

        sub_models_path = root / "subtrans" / "models.yaml"
        sub_models: List[Dict[str, Any]] = []
        chain: List[str] = []
        if sub_models_path.is_file():
            try:
                sm_raw = yaml.safe_load(sub_models_path.read_text(encoding="utf-8")) or {}
                chain = [str(x) for x in (sm_raw.get("chain") or [])]
                for m_name, m_cfg in (sm_raw.get("models") or {}).items():
                    if isinstance(m_cfg, dict):
                        sub_models.append({
                            "model": str(m_name),
                            "rpm": m_cfg.get("rpm", 0),
                            "rpd": m_cfg.get("rpd", 0),
                            "tpm": m_cfg.get("tpm", 0),
                            "status": str(m_cfg.get("status") or "active"),
                        })
            except Exception:
                pass
        # Query SQLite strictly read-only (?mode=ro) so GET never creates or mutates production DBs.
        cache_stats: Dict[str, Any] = {
            "db_exists": False,
            "total_cached_cues": 0,
            "today_cached_cues": 0,
            "today_requests": [],
            "exhausted_keys": [],
            "exhausted_models": [],
        }
        db_file = root / "subtrans" / ".cache" / "translations.db"
        if db_file.is_file():
            cache_stats["db_exists"] = True
            try:
                escaped_db_uri = db_file.resolve().as_uri()
                ro_uri = f"{escaped_db_uri}?mode=ro"
                conn = sqlite3.connect(ro_uri, uri=True, timeout=3.0)
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM translations")
                    cache_stats["total_cached_cues"] = int((cur.fetchone() or [0])[0])

                    cur.execute("SELECT COUNT(*) FROM translations WHERE pt_date = ?", (today_pt_str,))
                    cache_stats["today_cached_cues"] = int((cur.fetchone() or [0])[0])

                    cur.execute(
                        """
                        SELECT model, status, COUNT(*), COALESCE(SUM(cues_count), 0), COALESCE(SUM(total_tokens), 0)
                        FROM api_requests
                        WHERE pt_date = ? AND LOWER(COALESCE(status, '')) != 'dry_run'
                        GROUP BY model, status
                        ORDER BY COUNT(*) DESC
                        """,
                        (today_pt_str,),
                    )
                    cache_stats["today_requests"] = [
                        {
                            "model": str(r[0] or ""),
                            "status": str(r[1] or ""),
                            "healthy": str(r[1] or "").strip().lower() in {"success", "ok"},
                            "requests": int(r[2] or 0),
                            "cues": int(r[3] or 0),
                            "tokens": int(r[4] or 0),
                        }
                        for r in cur.fetchall()
                    ]

                    cur.execute(
                        "SELECT key_id, model, reason FROM key_exhaustion WHERE pt_date = ?",
                        (today_pt_str,),
                    )
                    cache_stats["exhausted_keys"] = [
                        {"key_id": str(r[0]), "model": str(r[1]), "reason": str(r[2] or "")}
                        for r in cur.fetchall()
                    ]

                    cur.execute("SELECT model, exhausted_until_pt, reason FROM model_exhaustion")
                    active_exhausted_models: List[Dict[str, str]] = []
                    for r in cur.fetchall():
                        until_raw = str(r[1] or "").strip()
                        if not until_raw:
                            continue
                        is_active_ex = False
                        try:
                            until_dt = datetime.fromisoformat(until_raw)
                            if until_dt.tzinfo is None:
                                until_dt = until_dt.replace(tzinfo=now_pt.tzinfo)
                            is_active_ex = until_dt > now_pt
                        except ValueError:
                            is_active_ex = until_raw > now_pt.isoformat()
                        if is_active_ex:
                            active_exhausted_models.append(
                                {"model": str(r[0]), "until_pt": until_raw, "reason": str(r[2] or "")}
                            )
                    cache_stats["exhausted_models"] = active_exhausted_models
                finally:
                    conn.close()
            except Exception:
                pass
        elif is_mock:
            cache_stats = {
                "db_exists": True,
                "mock_mode": True,
                "total_cached_cues": 1420,
                "today_cached_cues": 318,
                "today_requests": [
                    {"model": "gemini-3.5-flash-lite", "status": "success", "requests": 14, "cues": 318, "tokens": 24800}
                ],
                "exhausted_keys": [],
                "exhausted_models": [],
            }

        glossaries_dir = root / "glossaries"
        sub_base_raw_ov = root / "subtrans" / "glossary.yaml"
        sub_base_for_ov = sub_base_raw_ov.resolve() if not _is_reparse_or_symlink(sub_base_raw_ov) else sub_base_raw_ov
        gloss_root_for_ov = (
            glossaries_dir.resolve()
            if (glossaries_dir.is_dir() and not _is_reparse_or_symlink(glossaries_dir))
            else glossaries_dir
        )
        glossaries: List[Dict[str, Any]] = []
        if glossaries_dir.is_dir() and not _is_reparse_or_symlink(glossaries_dir):
            cand_gloss_files = sorted(
                {*glossaries_dir.glob("*.yaml"), *glossaries_dir.glob("*.yml")},
                key=lambda p: p.name.lower(),
            )
            for gf in cand_gloss_files:
                try:
                    if _is_reparse_or_symlink(gf) or is_secret_filename(gf.name):
                        continue
                    gf_resolved = gf.resolve()
                    curr_g = gf_resolved
                    seen_ov: set[Path] = set()
                    while curr_g.is_file() and curr_g not in seen_ov:
                        seen_ov.add(curr_g)
                        raw_ov = curr_g.read_text(encoding="utf-8").strip()
                        if "\n" not in raw_ov and raw_ov.endswith((".yaml", ".yml")):
                            nxt_ov_raw = curr_g.parent / raw_ov
                            if _is_reparse_or_symlink(nxt_ov_raw):
                                break
                            nxt_ov = nxt_ov_raw.resolve()
                            if ((nxt_ov.is_relative_to(gloss_root_for_ov) and nxt_ov.parent == gloss_root_for_ov) or nxt_ov == sub_base_for_ov) and nxt_ov.is_file() and nxt_ov != curr_g:
                                curr_g = nxt_ov
                                continue
                        break
                    sz_kb = round(curr_g.stat().st_size / 1024.0, 1)
                    ro_flag = (
                        gf.name.lower() in {"base.yaml", "base.yml"}
                        or curr_g == sub_base_for_ov
                        or not curr_g.is_relative_to(gloss_root_for_ov)
                    )
                except OSError:
                    continue
                glossaries.append({
                    "name": gf.name,
                    "path": str(gf_resolved),
                    "resolved_name": curr_g.name,
                    "is_pointer": curr_g != gf_resolved,
                    "read_only": ro_flag,
                    "size_kb": sz_kb,
                })

        # Pending subtitles queue (check state/, legacy root, then out/)
        pending_items: List[Dict[str, Any]] = []
        found_disk_pending = False
        for p_cand in (root / "state" / "pending_subs.json", root / "pending_subs.json", root / "out" / "pending_subs.json"):
            if p_cand.is_file():
                found_disk_pending = True
                try:
                    p_raw = json.loads(p_cand.read_text(encoding="utf-8"))
                    if isinstance(p_raw, list):
                        pending_items = [x for x in p_raw if isinstance(x, dict)]
                    elif isinstance(p_raw, dict) and isinstance(p_raw.get("items"), list):
                        pending_items = [x for x in p_raw["items"] if isinstance(x, dict)]
                    break
                except Exception:
                    pass

        return {
            "pt_date": today_pt_str,
            "next_reset_vn": next_vn.strftime("%Y-%m-%d %H:%M:%S"),
            "reset_in_seconds": reset_seconds,
            "key_slots": key_slots,
            "all_key_slots": all_key_slots,
            "configured_keys_count": active_key_count,
            "models": sub_models,
            "chain": chain,
            "cache": cache_stats,
            "glossaries": glossaries,
            "pending_subs": (
                pending_items
                if (found_disk_pending or not is_mock)
                else list(mock_pending_subs)
            ),
            "mock_mode": is_mock,
        }

    @app.put("/api/subtrans/api-keys")
    def api_update_subtrans_keys(request: Request, req: UpdateApiKeysRequest) -> Dict[str, Any]:
        """Update or clear GEMINI_API_KEY_1..3 in local .env without logging or returning raw keys."""
        dotenv_path = root / ".env"
        sub_dotenv_path = root / "subtrans" / ".env"

        with _DOTENV_LOCK:
            existing = _read_dotenv_map(dotenv_path)
            if sub_dotenv_path.is_file():
                for sk, sv in _read_dotenv_map(sub_dotenv_path).items():
                    existing[sk] = sv
            env_overrides = {k: v for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY")}

            try:
                updates = _apply_subtrans_key_edits(
                    existing,
                    req.key_1,
                    req.key_2,
                    req.key_3,
                    slots=req.slots,
                    env_overrides=env_overrides,
                )
                if updates:
                    _write_dotenv_keys(dotenv_path, updates)
                    if sub_dotenv_path.is_file():
                        sub_existing = _read_dotenv_map(sub_dotenv_path)
                        has_gemini_in_sub = any(k.startswith("GEMINI_API_KEY") for k in sub_existing)
                        sub_updates = (
                            updates
                            if has_gemini_in_sub
                            else {k: v for k, v in updates.items() if k in sub_existing and v is not None}
                        )
                        if sub_updates:
                            _write_dotenv_keys(sub_dotenv_path, sub_updates)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))

            # Leave Web Studio os.environ untouched so subprocesses load .env cleanly and UI slot controls stay accurate
            slots_summary, active_key_count, all_slots_summary = _resolve_subtrans_key_slots_all(
                dotenv_path, key_id_fn=compute_key_id
            )

        req_client = getattr(request, "client", None)
        req_client_host = (getattr(req_client, "host", "") or "").lower()
        req_host_name, _, _ = _parse_host_header(request.headers.get("host"))
        if not _is_loopback_request(req_client_host, req_host_name):
            for s_entry in all_slots_summary:
                s_entry["masked"] = ""
            for s_entry in slots_summary:
                s_entry["masked"] = ""
        return {
            "ok": True,
            "key_slots": slots_summary,
            "all_key_slots": all_slots_summary,
            "configured_count": active_key_count,
        }

    def _resolve_glossary_entry(raw_name: str) -> tuple[Path, Path, bool, bool]:
        """Resolve glossary file strictly inside glossaries/, following single-line text pointers."""
        cleaned = strip_wrapping_quotes(raw_name)
        if not cleaned or "\x00" in cleaned or is_unc_path(cleaned):
            raise HTTPException(status_code=400, detail="Tên tệp Glossary không hợp lệ.")
        raw_gloss_dir = root / "glossaries"
        raw_sub_base = root / "subtrans" / "glossary.yaml"
        if _is_reparse_or_symlink(raw_gloss_dir) or _is_reparse_or_symlink(raw_sub_base):
            raise HTTPException(status_code=403, detail="Từ chối truy cập symlink/junction Glossary.")
        gloss_root = raw_gloss_dir.resolve()
        sub_base_glossary = raw_sub_base.resolve()

        cand_in = Path(os.path.expanduser(cleaned))
        if cand_in.is_absolute():
            entry_raw = cand_in
        elif "/" in cleaned or "\\" in cleaned:
            entry_raw = root / cand_in
        else:
            entry_raw = gloss_root / cand_in

        if _is_reparse_or_symlink(entry_raw):
            raise HTTPException(status_code=403, detail="Từ chối truy cập symlink/junction Glossary.")
        try:
            entry_path = entry_raw.resolve()
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Đường dẫn Glossary không hợp lệ: {exc}") from exc

        if entry_path == sub_base_glossary:
            if not entry_path.is_file():
                raise HTTPException(status_code=404, detail="Không tìm thấy từ điển gốc subtrans/glossary.yaml.")
            return entry_path, entry_path, True, False

        if not entry_path.is_relative_to(gloss_root) or entry_path.parent != gloss_root:
            raise HTTPException(status_code=403, detail="Chỉ cho phép truy cập các tệp *.yaml nằm trong thư mục glossaries/.")
        if entry_path.suffix.lower() not in {".yaml", ".yml"} or is_secret_filename(entry_path.name):
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ tệp từ điển định dạng .yaml/.yml trong glossaries/.")
        if not entry_path.is_file():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy tệp Glossary: {entry_path.name}")

        curr = entry_path
        seen_ptrs: set[Path] = set()
        while curr.is_file() and curr not in seen_ptrs:
            seen_ptrs.add(curr)
            try:
                raw_txt = curr.read_text(encoding="utf-8").strip()
            except Exception:
                break
            if "\n" not in raw_txt and raw_txt.endswith((".yaml", ".yml")):
                nxt_raw = curr.parent / raw_txt
                if _is_reparse_or_symlink(nxt_raw):
                    raise HTTPException(status_code=403, detail="Từ chối con trỏ trỏ tới symlink/junction.")
                nxt = nxt_raw.resolve()
                if not ((nxt.is_relative_to(gloss_root) and nxt.parent == gloss_root) or nxt == sub_base_glossary):
                    raise HTTPException(
                        status_code=403,
                        detail="Từ chối con trỏ Glossary trỏ ra ngoài thư mục glossaries/.",
                    )
                if nxt.is_file() and nxt != curr:
                    curr = nxt
                    continue
            break

        is_pointer = curr != entry_path
        is_read_only = (
            entry_path.name.lower() in {"base.yaml", "base.yml"}
            or curr == sub_base_glossary
            or not curr.is_relative_to(gloss_root)
        )
        return entry_path, curr, is_read_only, is_pointer

    def _validate_glossary_yaml_text(yaml_content: str) -> Dict[str, Any]:
        stripped = yaml_content.strip()
        if "\n" not in stripped and stripped.endswith((".yaml", ".yml")):
            raise ValueError("Nội dung YAML không được là dòng con trỏ đơn lẻ (.yaml).")
        try:
            parsed = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            raise ValueError(f"Lỗi cú pháp YAML trong Glossary: {exc}") from exc
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ValueError("Cấu trúc gốc của Glossary phải là một bản đồ (mapping) YAML.")
        for non_null_k in ("meta", "terms", "characters", "style_rules", "difficulty_markers", "address_rules"):
            if non_null_k in parsed and parsed[non_null_k] is None:
                raise ValueError(
                    f"Mục '{non_null_k}' không được để giá trị null — hãy dùng danh sách rỗng [] hoặc bản đồ {{}}."
                )
        if "meta" in parsed:
            if not isinstance(parsed["meta"], dict):
                raise ValueError("Mục 'meta' phải là một bản đồ (mapping).")
            if "schema" in parsed["meta"]:
                sch_v = parsed["meta"]["schema"]
                if isinstance(sch_v, bool) or not isinstance(sch_v, int) or sch_v < 1:
                    raise ValueError("Mục 'meta.schema' phải là số nguyên dương (ví dụ: 2).")
        if "terms" in parsed:
            if not isinstance(parsed["terms"], list):
                raise ValueError("Mục 'terms' phải là danh sách (list).")
            for idx, t in enumerate(parsed["terms"], start=1):
                if (
                    not isinstance(t, dict)
                    or not isinstance(t.get("ja"), str)
                    or not t["ja"].strip()
                    or not isinstance(t.get("vi"), str)
                    or not t["vi"].strip()
                ):
                    raise ValueError(f"Mục terms #{idx} phải có đầy đủ chuỗi 'ja' và 'vi' khác rỗng.")
                for list_fld in ("aliases", "alt"):
                    if list_fld in t:
                        lv = t[list_fld]
                        if not isinstance(lv, list) or any(not isinstance(item_l, str) or not item_l.strip() for item_l in lv):
                            raise ValueError(f"Mục terms #{idx}.{list_fld} phải là danh sách chuỗi khác rỗng.")
                for opt_str_fld in ("en", "zh", "note", "genre"):
                    if opt_str_fld in t and t[opt_str_fld] is not None and not isinstance(t[opt_str_fld], str):
                        raise ValueError(f"Mục terms #{idx}.{opt_str_fld} phải là chuỗi văn bản.")
        if "characters" in parsed:
            if not isinstance(parsed["characters"], list):
                raise ValueError("Mục 'characters' phải là danh sách (list).")
            for idx, c in enumerate(parsed["characters"], start=1):
                if not isinstance(c, dict):
                    raise ValueError(f"Mục characters #{idx} phải là đối tượng từ điển.")
                for char_str_fld in (
                    "name_ja",
                    "name_vi",
                    "name_en",
                    "role",
                    "gender",
                    "register",
                    "speaking_style",
                    "first_person",
                    "second_person",
                    "note",
                ):
                    if char_str_fld in c and c[char_str_fld] is not None and not isinstance(c[char_str_fld], str):
                        raise ValueError(f"Mục characters #{idx}.{char_str_fld} phải là chuỗi văn bản.")
                for char_list_fld in ("aliases", "alt"):
                    if char_list_fld in c:
                        cal = c[char_list_fld]
                        if not isinstance(cal, list) or any(not isinstance(x_al, str) or not x_al.strip() for x_al in cal):
                            raise ValueError(f"Mục characters #{idx}.{char_list_fld} phải là danh sách chuỗi khác rỗng.")
                if "speaks_to" in c:
                    if not isinstance(c["speaks_to"], list) or any(not isinstance(s, dict) for s in c["speaks_to"]):
                        raise ValueError(f"Mục characters #{idx}.speaks_to phải là danh sách các đối tượng từ điển.")
                    for s_obj in c["speaks_to"]:
                        for s_fld in ("listener", "self", "other", "note"):
                            if s_fld in s_obj and s_obj[s_fld] is not None and not isinstance(s_obj[s_fld], str):
                                raise ValueError(f"Mục characters #{idx}.speaks_to.{s_fld} phải là chuỗi văn bản.")
        for str_list_k in ("style_rules", "difficulty_markers"):
            if str_list_k in parsed:
                sl_val = parsed[str_list_k]
                if not isinstance(sl_val, list) or any(not isinstance(x_s, str) or not x_s.strip() for x_s in sl_val):
                    raise ValueError(f"Mục '{str_list_k}' phải là danh sách chuỗi văn bản khác rỗng.")
        if "address_rules" in parsed:
            if not isinstance(parsed["address_rules"], dict):
                raise ValueError("Mục 'address_rules' phải là bản đồ (mapping).")
            if (
                "principle" in parsed["address_rules"]
                and parsed["address_rules"]["principle"] is not None
                and not isinstance(parsed["address_rules"]["principle"], str)
            ):
                raise ValueError("Mục 'address_rules.principle' phải là chuỗi văn bản.")
            for ar_k in ("relations", "first_person", "second_person", "honorifics"):
                if ar_k in parsed["address_rules"]:
                    ar_v = parsed["address_rules"][ar_k]
                    if not isinstance(ar_v, list) or any(not isinstance(item_ar, dict) for item_ar in ar_v):
                        raise ValueError(f"Mục address_rules.{ar_k} phải là danh sách các đối tượng từ điển.")
                    for idx_ar, item_ar in enumerate(ar_v, start=1):
                        if ar_k == "relations":
                            for rel_fld in ("case", "self", "other", "note"):
                                if rel_fld in item_ar and item_ar[rel_fld] is not None and not isinstance(item_ar[rel_fld], str):
                                    raise ValueError(f"Mục address_rules.relations #{idx_ar}.{rel_fld} phải là chuỗi văn bản.")
                        else:
                            for pr_fld in ("ja", "vi", "note"):
                                if pr_fld in item_ar and item_ar[pr_fld] is not None and not isinstance(item_ar[pr_fld], str):
                                    raise ValueError(f"Mục address_rules.{ar_k} #{idx_ar}.{pr_fld} phải là chuỗi văn bản.")
                            if "alt" in item_ar:
                                alt_v = item_ar["alt"]
                                if not isinstance(alt_v, list) or any(not isinstance(x_alt, str) or not x_alt.strip() for x_alt in alt_v):
                                    raise ValueError(f"Mục address_rules.{ar_k} #{idx_ar}.alt phải là danh sách chuỗi khác rỗng.")
        return {
            "terms_count": len(parsed.get("terms") or []),
            "characters_count": len(parsed.get("characters") or []),
        }

    @app.get("/api/subtrans/glossary")
    def api_get_glossary(name: str = Query(...)) -> Dict[str, Any]:
        entry_p, resolved_p, is_ro, is_ptr = _resolve_glossary_entry(name)
        text = resolved_p.read_text(encoding="utf-8")
        return {
            "ok": True,
            "name": entry_p.name,
            "resolved_name": resolved_p.name,
            "path": str(entry_p),
            "resolved_path": str(resolved_p),
            "read_only": is_ro,
            "is_pointer": is_ptr,
            "yaml_content": text,
        }

    @app.put("/api/subtrans/glossary")
    def api_save_glossary(req: SaveGlossaryRequest) -> Dict[str, Any]:
        entry_p, resolved_p, is_ro, is_ptr = _resolve_glossary_entry(req.name)
        if is_ro:
            raise HTTPException(
                status_code=403,
                detail="Từ điển gốc (base.yaml / subtrans/glossary.yaml) được khóa chỉ-đọc để bảo vệ quy tắc nền.",
            )
        try:
            stats = _validate_glossary_yaml_text(req.yaml_content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        tmp_g = (
            resolved_p.parent
            / f".{resolved_p.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:6]}.tmp"
        )
        norm_text = req.yaml_content.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
        try:
            tmp_g.write_text(norm_text, encoding="utf-8", newline="\n")
            os.replace(tmp_g, resolved_p)
        except Exception:
            with suppress(OSError):
                tmp_g.unlink(missing_ok=True)
            raise
        return {
            "ok": True,
            "name": entry_p.name,
            "resolved_name": resolved_p.name,
            "is_pointer": is_ptr,
            **stats,
        }

    @app.delete("/api/subtrans/pending")
    def api_delete_pending_sub(target: str = Query(..., min_length=1)) -> Dict[str, Any]:
        cleaned_target = strip_wrapping_quotes(target)
        if not cleaned_target or "\x00" in cleaned_target:
            raise HTTPException(status_code=400, detail="Mục cần xóa khỏi pending_subs.json không hợp lệ.")
        jstate = job_manager.get_state()
        if jstate.get("status") in {"running", "pausing"}:
            raise HTTPException(
                status_code=409,
                detail="Không thể xóa mục pending_subs.json khi đang có tiến trình chạy. Hãy chờ hoặc dừng tác vụ trước.",
            )
        if not is_mock:
            lock_state = check_active_gpu_lock(root)
            if lock_state.get("locked"):
                raise HTTPException(
                    status_code=409,
                    detail=f"GPU đang bận xử lý tiến trình khác (PID={lock_state.get('pid')}); tạm khóa xóa pending_subs.json.",
                )
            pstate = telemetry_hub.read_pipeline_state()
            if pstate.get("pid_alive") and str(pstate.get("effective_stage") or "") in {
                "initializing",
                "probing",
                "preflight_dedup",
                "upscaling",
                "muxing",
                "auditing",
                "subtrans",
            }:
                raise HTTPException(
                    status_code=409,
                    detail="Đang có tiến trình CLI hoạt động — tạm khóa xóa pending_subs.json để tránh ghi đè dữ liệu.",
                )

        has_path_sep = ("/" in cleaned_target) or ("\\" in cleaned_target)
        stem_t = os.path.splitext(os.path.basename(cleaned_target))[0]
        stem_variants = {stem_t, f"{stem_t}_4K", f"{stem_t}_4K_animejanai"}
        if stem_t.lower().endswith("_4k_animejanai"):
            stem_variants.add(stem_t[:-14])
        elif stem_t.lower().endswith("_4k"):
            stem_variants.add(stem_t[:-3])
        cand_keys = {cleaned_target} | stem_variants
        norm_target_dir = (
            os.path.normcase(os.path.dirname(os.path.abspath(cleaned_target)))
            if has_path_sep
            else ""
        )
        norm_target_abs = os.path.normcase(os.path.abspath(cleaned_target)) if has_path_sep else ""

        def _matches_pending(it: Any) -> bool:
            if not isinstance(it, dict):
                return False
            if has_path_sep:
                for path_k in ("video_path", "ja_path", "base_name", "episode"):
                    vp_str = str(it.get(path_k) or "").strip()
                    if not vp_str or (("/" not in vp_str) and ("\\" not in vp_str)):
                        continue
                    vp_abs = os.path.normcase(os.path.abspath(vp_str))
                    if vp_abs == norm_target_abs:
                        return True
                    vp_dir = os.path.normcase(os.path.dirname(vp_abs))
                    vp_stem = os.path.splitext(os.path.basename(vp_abs))[0]
                    if vp_dir == norm_target_dir and vp_stem in stem_variants:
                        return True
                return False
            for field_k in ("base_name", "episode", "video_path", "ja_path"):
                v_str = str(it.get(field_k) or "").strip()
                if not v_str:
                    continue
                if v_str in cand_keys or os.path.basename(v_str) in cand_keys:
                    return True
                v_stem = os.path.splitext(os.path.basename(v_str))[0]
                if v_stem in cand_keys:
                    return True
            return False

        pending_file: Optional[Path] = None
        raw_obj: Any = None
        for p_cand in (root / "state" / "pending_subs.json", root / "pending_subs.json", root / "out" / "pending_subs.json"):
            if p_cand.is_file() and not _is_reparse_or_symlink(p_cand):
                pending_file = p_cand
                try:
                    raw_obj = json.loads(p_cand.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Không thể đọc pending_subs.json: {exc}") from exc
                break

        if pending_file is None:
            if is_mock:
                remaining_mock = [it for it in mock_pending_subs if not _matches_pending(it)]
                removed_mock = len(mock_pending_subs) - len(remaining_mock)
                if removed_mock == 0:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Không tìm thấy mục '{cleaned_target}' trong pending_subs.json.",
                    )
                mock_pending_subs[:] = remaining_mock
                return {
                    "ok": True,
                    "removed": removed_mock,
                    "remaining_count": len(mock_pending_subs),
                    "pending_subs": list(mock_pending_subs),
                    "mock": True,
                }
            raise HTTPException(status_code=404, detail="Không có tệp pending_subs.json nào đang tồn tại.")

        is_wrapped_dict = isinstance(raw_obj, dict) and isinstance(raw_obj.get("items"), list)
        items_list = list(raw_obj["items"] if is_wrapped_dict else (raw_obj if isinstance(raw_obj, list) else []))

        remaining = [it for it in items_list if not _matches_pending(it)]
        removed_count = len(items_list) - len(remaining)
        if removed_count == 0:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy mục '{cleaned_target}' trong pending_subs.json.")

        out_obj: Any = {**raw_obj, "items": remaining} if is_wrapped_dict else remaining
        tmp_pf = pending_file.parent / f".{pending_file.name}.{os.getpid()}.tmp"
        try:
            tmp_pf.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
            os.replace(tmp_pf, pending_file)
        except Exception:
            tmp_pf.unlink(missing_ok=True)
            raise

        return {
            "ok": True,
            "removed": removed_count,
            "remaining_count": len(remaining),
            "pending_subs": [x for x in remaining if isinstance(x, dict)],
        }

    audits_cache_lock = threading.Lock()
    audits_cache: Dict[str, Any] = {
        "key": None,
        "stat_sig": None,
        "expires_at": 0.0,
        "payload": None,
    }

    def _temp_dir_sig(scope_label: str, temp_path: Path, dir_mtime_ns: int) -> tuple[Any, ...]:
        sub_parts: List[Any] = []
        try:
            with os.scandir(temp_path) as t_it:
                for t_entry in t_it:
                    tname = t_entry.name
                    if tname in ("upscaled_segs", "raw_segs"):
                        try:
                            seg_st = t_entry.stat(follow_symlinks=False)
                            seg_count = 0
                            with os.scandir(t_entry.path) as seg_it:
                                for _ in seg_it:
                                    seg_count += 1
                            sub_parts.append((tname, seg_st.st_mtime_ns, seg_count))
                        except OSError:
                            pass
                    elif tname in ("manifest.json", "plan.json"):
                        try:
                            mst = t_entry.stat(follow_symlinks=False)
                            sub_parts.append((tname, mst.st_mtime_ns, mst.st_size))
                        except OSError:
                            pass
        except OSError:
            pass
        sub_parts.sort()
        return (scope_label, temp_path.name, dir_mtime_ns, tuple(sub_parts))

    def _collect_audits_stat_sig(
        out_dirs: Sequence[Path],
    ) -> tuple[Any, ...]:
        """Collect lightweight os.scandir stat signature for /api/audits TTL cache invalidation."""
        dir_sigs: List[Any] = []
        audit_sigs: List[Any] = []
        temp_sigs: List[Any] = []
        try:
            with os.scandir(root) as it:
                for entry in it:
                    if entry.name.startswith(".temp_") and entry.is_dir(follow_symlinks=False):
                        try:
                            temp_sigs.append(_temp_dir_sig("root", Path(entry.path), entry.stat(follow_symlinks=False).st_mtime_ns))
                        except OSError:
                            temp_sigs.append(("root", entry.name, 0, ()))
        except OSError:
            pass

        for base_out in out_dirs:
            search_targets: List[Path] = [base_out]
            try:
                base_st = base_out.stat()
                dir_sigs.append((str(base_out), base_st.st_mtime_ns))
                with os.scandir(base_out) as it:
                    for entry in it:
                        ename = entry.name
                        elow = ename.lower()
                        try:
                            est = entry.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if ename.startswith(".temp_"):
                                temp_sigs.append(_temp_dir_sig(str(base_out), Path(entry.path), est.st_mtime_ns))
                            elif elow in {"audit", "cast"} or not ename.startswith("."):
                                dir_sigs.append((entry.path, est.st_mtime_ns))
                                search_targets.append(Path(entry.path))
                        elif elow.endswith((".audit.json", ".audit.md", ".cast.yaml")):
                            audit_sigs.append((entry.path, est.st_mtime_ns, est.st_size))
            except OSError:
                continue

            for sub_dir in search_targets[1:]:
                try:
                    with os.scandir(sub_dir) as sub_it:
                        for sub_entry in sub_it:
                            sname = sub_entry.name
                            slow = sname.lower()
                            try:
                                sst = sub_entry.stat(follow_symlinks=False)
                            except OSError:
                                continue
                            if sub_entry.is_dir(follow_symlinks=False):
                                if sname.startswith(".temp_"):
                                    temp_sigs.append(_temp_dir_sig(str(sub_dir), Path(sub_entry.path), sst.st_mtime_ns))
                            elif slow.endswith((".audit.json", ".audit.md", ".cast.yaml")):
                                audit_sigs.append((sub_entry.path, sst.st_mtime_ns, sst.st_size))
                except OSError:
                    continue

        dir_sigs.sort()
        audit_sigs.sort()
        temp_sigs.sort()
        return (tuple(dir_sigs), tuple(audit_sigs), tuple(temp_sigs))

    @app.get("/api/audits")
    def api_list_audits(output_dir: Optional[str] = Query(default=None)) -> Dict[str, Any]:
        try:
            out_dirs = _collect_known_output_dirs(root, job_manager, extra_dir=output_dir)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            job_state = job_manager.get_state()
        except Exception:
            job_state = {}
        active_j = job_state.get("active_job") if isinstance(job_state.get("active_job"), dict) else {}
        hist_list = job_state.get("history") if isinstance(job_state.get("history"), list) else []
        latest_h = hist_list[0] if (hist_list and isinstance(hist_list[0], dict)) else {}
        job_sig = (
            job_state.get("status"),
            active_j.get("job_id"),
            latest_h.get("job_id"),
            latest_h.get("finished_at"),
        )
        cache_key = (
            (output_dir or "").strip(),
            is_mock,
            tuple(str(d) for d in out_dirs),
            job_sig,
        )
        stat_sig = _collect_audits_stat_sig(out_dirs)
        now_mono = time.monotonic()
        with audits_cache_lock:
            cached_payload = audits_cache["payload"]
            if (
                cached_payload is not None
                and now_mono < audits_cache["expires_at"]
                and audits_cache["key"] == cache_key
                and audits_cache["stat_sig"] == stat_sig
            ):
                return {
                    "audits": list(cached_payload["audits"]),
                    "checkpoints": list(cached_payload["checkpoints"]),
                }

        audits: List[Dict[str, Any]] = []
        discovered_entries: List[tuple[Path, Path]] = []
        seen_keys: set[tuple[str, str]] = set()
        seen_paths: set[str] = set()

        for base_out in out_dirs:
            norm_base = os.path.normcase(str(base_out.resolve()))
            base_audit_res = (base_out / "audit").resolve()
            for search_dir in _iter_out_search_dirs(base_out):
                search_res = search_dir.resolve()
                for af in search_dir.glob("*.audit.json"):
                    try:
                        norm_af = os.path.normcase(str(af.resolve()))
                    except OSError:
                        continue
                    sub_key = (
                        af.name.lower()
                        if search_res in {base_out.resolve(), base_audit_res}
                        else f"{search_dir.name.lower()}/{af.name.lower()}"
                    )
                    if norm_af in seen_paths or (norm_base, sub_key) in seen_keys:
                        continue
                    seen_paths.add(norm_af)
                    seen_keys.add((norm_base, sub_key))
                    discovered_entries.append((af, base_out))

        def _safe_audit_mtime(pair: tuple[Path, Path]) -> float:
            try:
                return pair[0].stat().st_mtime
            except OSError:
                return 0.0

        discovered_entries.sort(key=_safe_audit_mtime, reverse=True)

        uncollapsed_items: List[tuple[tuple[float, int], Dict[str, Any]]] = []
        deduped_items: Dict[tuple[str, str, str], tuple[tuple[bool, float, int], tuple[float, int], Dict[str, Any]]] = {}

        for entry_idx, (af, base_out) in enumerate(discovered_entries):
            try:
                st = af.stat()
                mtime_val = st.st_mtime
                size_kb_val = round(st.st_size / 1024.0, 1)
            except OSError:
                mtime_val = 0.0
                size_kb_val = 0.0
            raw_stem = af.name[:-len(".audit.json")] if af.name.endswith(".audit.json") else af.stem
            corrupt_ep_label = re.sub(r"_4K(?:_animejanai)?$", "", raw_stem, flags=re.IGNORECASE)
            sort_key = (mtime_val, -entry_idx)
            try:
                adata = json.loads(af.read_text(encoding="utf-8"))
            except Exception as exc:
                uncollapsed_items.append((sort_key, {
                    "episode": corrupt_ep_label,
                    "audit_json_path": str(af.resolve()),
                    "output_file": "",
                    "source_file": "",
                    "status": "CORRUPT",
                    "corrupt": True,
                    "error": f"JSON lỗi ({type(exc).__name__}): {exc}",
                    "sanity_status": "CORRUPT",
                    "model_id": "",
                    "average_pipeline_fps": 0.0,
                    "ai_steady_fps": 0.0,
                    "source_frames": 0,
                    "output_frames": 0,
                    "frames_matched": False,
                    "audio_tracks_count": 0,
                    "subtitle_tracks_count": 0,
                    "fonts_count": 0,
                    "output_file_size_mb": size_kb_val / 1024.0,
                    "completed_at": "",
                    "sub_audit_md_path": "",
                    "cast_yaml_path": "",
                    "mtime": mtime_val,
                }))
                continue
            if not isinstance(adata, dict):
                uncollapsed_items.append((sort_key, {
                    "episode": corrupt_ep_label,
                    "audit_json_path": str(af.resolve()),
                    "output_file": "",
                    "source_file": "",
                    "status": "CORRUPT",
                    "corrupt": True,
                    "error": "Cấu trúc audit JSON không phải object",
                    "sanity_status": "CORRUPT",
                    "model_id": "",
                    "average_pipeline_fps": 0.0,
                    "ai_steady_fps": 0.0,
                    "source_frames": 0,
                    "output_frames": 0,
                    "frames_matched": False,
                    "audio_tracks_count": 0,
                    "subtitle_tracks_count": 0,
                    "fonts_count": 0,
                    "output_file_size_mb": size_kb_val / 1024.0,
                    "completed_at": "",
                    "sub_audit_md_path": "",
                    "cast_yaml_path": "",
                    "mtime": mtime_val,
                }))
                continue

            stem_4k = raw_stem
            stem_src = re.sub(r"_4K(?:_animejanai)?$", "", stem_4k, flags=re.IGNORECASE)

            sub_audit_md = ""
            cast_yaml = ""
            raw_doc_dirs = [af.parent, base_out, base_out / "audit", base_out / "cast"]
            doc_dirs: List[Path] = []
            seen_doc_dirs: set[str] = set()
            for dd in raw_doc_dirs:
                try:
                    dd_res = dd.resolve()
                    norm_dd = os.path.normcase(str(dd_res))
                except OSError:
                    continue
                if norm_dd not in seen_doc_dirs and dd_res.is_dir():
                    seen_doc_dirs.add(norm_dd)
                    doc_dirs.append(dd_res)

            for doc_dir in doc_dirs:
                for cand_stem in (stem_4k, stem_src, f"{stem_src}.vi", f"{stem_4k}.vi"):
                    md_p = doc_dir / f"{cand_stem}.audit.md"
                    if not sub_audit_md and md_p.is_file():
                        sub_audit_md = str(md_p.resolve())
                    cy_p = doc_dir / f"{cand_stem}.cast.yaml"
                    if not cast_yaml and cy_p.is_file():
                        cast_yaml = str(cy_p.resolve())

            mux_info = adata.get("mux_audit") if isinstance(adata.get("mux_audit"), dict) else {}
            sanity_info = adata.get("output_sanity") if isinstance(adata.get("output_sanity"), dict) else {}
            vfc = (
                mux_info.get("video_frame_count")
                if isinstance(mux_info.get("video_frame_count"), dict)
                else {}
            )
            src_frames = int(adata.get("source_frames") or vfc.get("source_frames") or 0)
            out_frames = int(adata.get("output_frames") or vfc.get("output_frames") or 0)
            if "frames_matched" in adata and (adata.get("source_frames") or adata.get("output_frames")):
                frames_matched = bool(adata.get("frames_matched"))
            elif "passed" in vfc and (src_frames or out_frames):
                frames_matched = bool(vfc.get("passed"))
            else:
                frames_matched = bool(src_frames > 0 and src_frames == out_frames)

            aud_streams = mux_info.get("audio_streams") if isinstance(mux_info.get("audio_streams"), dict) else {}
            sub_streams = (
                mux_info.get("subtitle_streams") if isinstance(mux_info.get("subtitle_streams"), dict) else {}
            )
            font_att = mux_info.get("font_attachments") if isinstance(mux_info.get("font_attachments"), dict) else {}

            audio_count = int(adata.get("audio_tracks_count") or aud_streams.get("output_count") or 0)
            sub_count = int(adata.get("subtitle_tracks_count") or sub_streams.get("output_count") or 0)
            fonts_count = int(adata.get("fonts_count") or font_att.get("output_count") or 0)

            episode_label = (
                f"{af.parent.name}/{stem_4k}"
                if af.parent.resolve() not in {base_out.resolve(), (base_out / "audit").resolve()}
                else stem_4k
            )

            raw_orig_src = str(adata.get("original_source_file") or "").strip()
            raw_audit_src = str(adata.get("source_file") or "").strip()
            resolved_audit_src = ""
            for src_cand in (raw_orig_src, raw_audit_src):
                if src_cand and ".preview_clip_" not in src_cand:
                    resolved_audit_src = src_cand
                    break

            raw_output_file = str(adata.get("output_file") or "")
            completed_at_val = str(adata.get("completed_at") or "")
            item = {
                "episode": episode_label,
                "audit_json_path": str(af.resolve()),
                "output_file": raw_output_file,
                "source_file": resolved_audit_src,
                "status": str(adata.get("status") or mux_info.get("status") or "UNKNOWN"),
                "sanity_status": str(sanity_info.get("status") or "N/A"),
                "model_id": str(adata.get("model_id") or ""),
                "average_pipeline_fps": adata.get("average_pipeline_fps", 0.0),
                "ai_steady_fps": adata.get("ai_steady_fps", 0.0),
                "source_frames": src_frames,
                "output_frames": out_frames,
                "frames_matched": frames_matched,
                "audio_tracks_count": audio_count,
                "subtitle_tracks_count": sub_count,
                "fonts_count": fonts_count,
                "output_file_size_mb": adata.get("output_file_size_mb", 0.0),
                "completed_at": completed_at_val,
                "sub_audit_md_path": sub_audit_md,
                "cast_yaml_path": cast_yaml,
            }

            try:
                norm_parent_dir = os.path.normcase(str(af.parent.resolve()))
            except OSError:
                norm_parent_dir = os.path.normcase(os.path.normpath(str(af.parent)))
            stripped_out = raw_output_file.strip()
            if stripped_out:
                out_cand_p = Path(os.path.expanduser(stripped_out))
                if not out_cand_p.is_absolute():
                    out_cand_p = root / out_cand_p
                try:
                    norm_output_file = os.path.normcase(str(out_cand_p.resolve()))
                except OSError:
                    norm_output_file = os.path.normcase(os.path.normpath(str(out_cand_p)))
            else:
                norm_output_file = ""
            stripped_completed = completed_at_val.strip()

            if norm_parent_dir and norm_output_file and stripped_completed:
                dedup_key = (norm_parent_dir, norm_output_file, stripped_completed)
                pref_key = (is_upscaled_output_stem(raw_stem), mtime_val, -entry_idx)
                existing = deduped_items.get(dedup_key)
                if existing is None or pref_key > existing[0]:
                    deduped_items[dedup_key] = (pref_key, sort_key, item)
            else:
                uncollapsed_items.append((sort_key, item))

        combined_audits: List[tuple[tuple[float, int], Dict[str, Any]]] = list(uncollapsed_items)
        for _pref_key, sort_key, item in deduped_items.values():
            combined_audits.append((sort_key, item))
        combined_audits.sort(key=lambda pair: pair[0], reverse=True)
        audits = [pair[1] for pair in combined_audits[:60]]

        if is_mock and not audits:
            audits.append({
                "episode": "Demo_EP01_4K",
                "audit_json_path": "",
                "output_file": "out/Demo_EP01_4K.mkv",
                "source_file": "[Demo] Anime_EP01 (1080p).mkv",
                "status": "PASS",
                "sanity_status": "PASS",
                "model_id": "animevideov3",
                "average_pipeline_fps": 41.2,
                "ai_steady_fps": 41.2,
                "source_frames": 34560,
                "output_frames": 34560,
                "frames_matched": True,
                "audio_tracks_count": 2,
                "subtitle_tracks_count": 2,
                "fonts_count": 12,
                "output_file_size_mb": 642.5,
                "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "sub_audit_md_path": "",
                "cast_yaml_path": "",
                "mock_mode": True,
            })

        all_cps = scan_temp_checkpoints(root, extra_dirs=out_dirs)
        clean_target_dir = _resolve_dir_from_raw(root, output_dir, strict_secret=True) if (output_dir and output_dir.strip()) else (root / "out").resolve()
        target_cps: List[Dict[str, Any]] = []
        if clean_target_dir and clean_target_dir.is_dir():
            for cp_it in all_cps:
                try:
                    if Path(str(cp_it.get("path") or "")).resolve().is_relative_to(clean_target_dir):
                        target_cps.append(cp_it)
                except OSError:
                    pass
        result_payload = {
            "audits": audits,
            "checkpoints": all_cps,
            "target_checkpoints": target_cps,
        }
        with audits_cache_lock:
            audits_cache["key"] = cache_key
            audits_cache["stat_sig"] = stat_sig
            audits_cache["expires_at"] = time.monotonic() + 1.5
            audits_cache["payload"] = result_payload
        return {
            "audits": list(result_payload["audits"]),
            "checkpoints": list(result_payload["checkpoints"]),
            "target_checkpoints": list(result_payload["target_checkpoints"]),
        }

    @app.get("/api/audits/detail")
    def api_audit_detail(path: str = Query(...)) -> Any:
        if "\x00" in path or is_unc_path(path):
            raise HTTPException(status_code=400, detail="Đường dẫn file audit không hợp lệ (chứa NUL hoặc UNC).")
        cand = Path(os.path.expanduser(path.strip()))
        if not cand.is_absolute():
            cand = (root / cand).resolve()
        else:
            cand = cand.resolve()

        low_name = cand.name.lower()
        if not (
            low_name.endswith(".audit.json")
            or low_name.endswith(".audit.md")
            or low_name.endswith(".cast.yaml")
        ):
            raise HTTPException(status_code=400, detail="Loại tệp báo cáo không hợp lệ.")

        # Permit reading audit artifacts within workspace_root or any known output directory
        allowed_roots = [root, *_collect_known_output_dirs(root, job_manager)]
        is_allowed = False
        for ar in allowed_roots:
            try:
                cand.relative_to(ar)
                is_allowed = True
                break
            except ValueError:
                continue
        if not is_allowed:
            raise HTTPException(
                status_code=403,
                detail="Chỉ cho phép xem báo cáo kiểm định thuộc dự án hoặc thư mục xuất đã cấu hình.",
            )

        if not cand.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy tệp báo cáo kiểm định.")

        raw_text = cand.read_text(encoding="utf-8", errors="replace")
        if low_name.endswith(".audit.json"):
            try:
                parsed_json = json.loads(raw_text)
            except (json.JSONDecodeError, ValueError) as exc:
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": f"File audit JSON bị hỏng cú pháp ({exc}).",
                        "corrupt": True,
                        "path": str(cand),
                        "raw_preview": raw_text[:4000],
                    },
                )
            if not isinstance(parsed_json, dict):
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": "Cấu trúc audit JSON không phải object",
                        "corrupt": True,
                        "path": str(cand),
                        "raw_preview": raw_text[:4000],
                    },
                )
            return {"type": "json", "path": str(cand), "data": parsed_json}
        return {"type": "text", "path": str(cand), "content": raw_text}

    @app.get("/api/night-report")
    def api_night_report(
        session_name: str = Query(default="night1", pattern=r"^[A-Za-z0-9_-]{1,64}$"),
    ) -> Dict[str, Any]:
        if is_mock:
            mock_md = "\n".join([
                f"# BÁO CÁO TỔNG KẾT CA CHẠY QUA ĐÊM ({session_name.upper()})",
                "",
                "## 1. Thông Tin Phiên Chạy & Cửa Sổ Giám Sát",
                "* **Chế độ:** Mock Studio Telemetry (Offline Demo)",
                "",
                "## 2. Viễn Trắc Hệ Thống (Phần Cứng & Ổ Cứng)",
                "* **GPU Utilization (TB):** **82.0%** (P-State: P0 | Mẫu: 120)",
                "* **Đỉnh VRAM (RTX 3060):** **6,384 MiB**",
                "",
                "## 3. Trạng Thái & Hiệu Năng Từng Tập Video",
                "### Tập 1: [Demo] Anime_EP01 (1080p).mkv",
                "* **Trạng thái:** ✅ HOÀN TẤT TRỌN VẸN (100%)",
                "* **Tốc độ AI:** Trung vị **41.20 fps** | P5 **39.80 fps**",
                "",
                "## 4. Thống Kê Phụ Đề & Hạn Mức Gemini API",
                "* **gemini-3.5-flash-lite:** Tổng 14 requests | Thành công: 14 (100.0%) | 503 Spike: 0 (0.0%)",
                "",
                "## 5. Danh Sách Việc Cần Làm Bằng Tay (Action Items)",
                "1. **Kiểm tra phụ đề tiếng Việt:** Đã sẵn sàng trong chế độ mô phỏng.",
                "",
            ])
            mock_sessions = ["night1"]
            if session_name not in mock_sessions:
                mock_sessions.append(session_name)
            return {
                "ok": True,
                "session_name": session_name,
                "available_sessions": mock_sessions,
                "markdown": mock_md,
                "mock_mode": True,
            }
        from src.night_report import generate_night_report_text, list_night_report_sessions
        try:
            markdown = generate_night_report_text(session_name=session_name, workspace_root=root)
            sessions = list_night_report_sessions(workspace_root=root)
            if session_name not in sessions:
                sessions.append(session_name)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Không thể tạo báo cáo ca đêm: {exc}") from exc
        return {
            "ok": True,
            "session_name": session_name,
            "available_sessions": sessions,
            "markdown": markdown,
            "mock_mode": False,
        }

    return app
