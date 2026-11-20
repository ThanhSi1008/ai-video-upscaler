# -*- coding: utf-8 -*-
"""
web/app.py
Ứng dụng FastAPI trung tâm cho AI Video Upscaler & Subtrans Real-Time Studio:
- Phục vụ giao diện Web hiện đại (web/static/) và luồng SSE 1.0 Hz (/api/events).
- Cung cấp đầy đủ REST API cho Live Monitor, Queue & Scheduler, Preview 1080p vs 4K,
  Models & shows.yaml Configurator, Subtrans & Audit Vault.
- Cho phép vận hành bình thường mọi tác vụ (Upscale, Preview, Subtrans, Subs-Pending,
  cấu hình shows.yaml & quản lý API Key cục bộ) ở bất kỳ khung giờ nào.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import socket
import sqlite3
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence
import urllib.parse
import yaml

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from subtrans.cache import compute_key_id, get_next_pt_reset, get_now_pt
from web.fs_browser import (
    DEDUP_POLICY_PARAMS,
    SUBTITLE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    SafeLocalFSBrowser,
    is_secret_filename,
    match_show_pattern,
)
from web.job_manager import MODEL_REFERENCE_FPS, JobManager
from web.sse_hub import TelemetryHub, scan_temp_checkpoints

ALLOWED_OPEN_MEDIA_EXTS = VIDEO_EXTENSIONS | SUBTITLE_EXTENSIONS
STATIC_DIR = Path(__file__).resolve().parent / "static"


# =====================================================================
# PYDANTIC PAYLOAD SCHEMAS
# =====================================================================

_DEADLINE_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _validate_no_nul_str(val: Optional[str], field_name: str = "path") -> Optional[str]:
    if val is not None and "\x00" in val:
        raise ValueError(f"Trường '{field_name}' chứa ký tự NUL (\\x00) không hợp lệ.")
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
    zero_disk: bool = True
    use_tensorrt: bool = True
    subs_after_upscale: bool = True
    output_dir: Optional[str] = None
    force: bool = False

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


class StartUpscaleRequest(BaseModel):
    inputs: List[str] = Field(default_factory=list)
    output_dir: Optional[str] = None
    model: Optional[str] = None
    dedup_policy: Optional[str] = None
    batch_size: int = 8
    deadline: Optional[str] = None
    allow_partial: bool = False
    zero_disk: bool = True
    use_tensorrt: bool = True
    subs_after_upscale: bool = True
    subs_quality: str = "normal"
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

    @field_validator("output_dir")
    @classmethod
    def _validate_outdir_no_nul(cls, v: Optional[str]) -> Optional[str]:
        return _validate_no_nul_str(v, "output_dir")


class StartPreviewRequest(BaseModel):
    video_path: str
    seconds: int = 10
    model: Optional[str] = None
    dedup_policy: Optional[str] = None
    zero_disk: bool = True
    use_tensorrt: bool = True
    force: bool = True
    auto_open_player: bool = False

    @field_validator("video_path")
    @classmethod
    def _validate_vp(cls, v: str) -> str:
        return _validate_no_nul_str(v, "video_path") or ""

    @field_validator("seconds")
    @classmethod
    def _validate_sec(cls, v: int) -> int:
        if v < 1 or v > 3600:
            raise ValueError("seconds phải nằm trong khoảng 1..3600.")
        return v


class StartSubtransRequest(BaseModel):
    input_path: str
    output_dir: Optional[str] = None
    subs_quality: str = "normal"
    subs_upgrade: bool = False
    prefer_whisper: bool = False
    auto_align_en: bool = True
    no_embed: bool = False
    sub_offset_ms: int = 0
    glossary: Optional[str] = None
    dry_run: bool = False

    @field_validator("input_path", "output_dir", "glossary")
    @classmethod
    def _validate_paths_no_nul(cls, v: Optional[str]) -> Optional[str]:
        return _validate_no_nul_str(v, "path")


class StartSubsPendingRequest(BaseModel):
    subs_quality: str = "normal"
    dry_run: bool = False


class OpenMediaRequest(BaseModel):
    path: str
    prefer_mpv: bool = True

    @field_validator("path")
    @classmethod
    def _validate_p(cls, v: str) -> str:
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

    @field_validator("sample_path", "yaml_content")
    @classmethod
    def _reject_nul(cls, v: Optional[str]) -> Optional[str]:
        return _validate_no_nul_str(v, "sample_path/yaml_content")


class SaveShowsYamlRequest(BaseModel):
    yaml_content: str

    @field_validator("yaml_content")
    @classmethod
    def _reject_nul(cls, v: str) -> str:
        return _validate_no_nul_str(v, "yaml_content") or ""


class UpdateApiKeysRequest(BaseModel):
    key_1: Optional[str] = None
    key_2: Optional[str] = None
    key_3: Optional[str] = None

    @field_validator("key_1", "key_2", "key_3")
    @classmethod
    def _reject_crlf_and_null(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and any(ch in v for ch in ("\r", "\n", "\x00")):
            raise ValueError("API key không được chứa ký tự xuống dòng (CR/LF) hoặc NUL.")
        return v


# =====================================================================
# HELPER UTILITIES
# =====================================================================

def _mask_secret(raw: str) -> str:
    val = (raw or "").strip()
    if not val:
        return ""
    if len(val) <= 10:
        return f"{val[:2]}***{val[-2:]}"
    return f"{val[:6]}...{val[-4:]}"


def _read_dotenv_map(dotenv_path: Path) -> Dict[str, str]:
    """Đọc các cặp KEY=VALUE từ file .env cục bộ mà không bao giờ in hoặc log giá trị thô."""
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
    """Chống chèn dòng mới (CR/LF) hoặc ký tự NUL khi ghi tệp .env."""
    if any(ch in raw_val for ch in ("\r", "\n", "\x00")):
        raise ValueError(f"Giá trị '{key_name}' chứa ký tự xuống dòng (CR/LF) hoặc NUL không hợp lệ.")
    return raw_val.strip()


def _write_dotenv_keys(dotenv_path: Path, updates: Dict[str, Optional[str]]) -> None:
    """Cập nhật an toàn các biến API key vào .env cục bộ, chống CR/LF injection và bảo toàn mọi dòng khác."""
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
            # Nếu chuỗi rỗng -> xóa dòng khóa cũ đó
        else:
            new_lines.append(raw_line)

    for key_name, val in sanitized_updates.items():
        if val is None or key_name in handled:
            continue
        if val:
            new_lines.append(f"{key_name}={val}")

    tmp_path = dotenv_path.parent / f".env.{os.getpid()}.tmp"
    content = "\n".join(new_lines).rstrip("\n") + ("\n" if new_lines else "")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, dotenv_path)


def _split_gemini_key_tokens(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tok.strip() for tok in re.split(r"[;,]", raw) if tok.strip()]


def _resolve_subtrans_key_slots(
    dotenv_path: Path,
    key_id_fn: Optional[Callable[[str], str]] = None,
) -> tuple[List[Dict[str, Any]], int]:
    """
    Giải mã trạng thái 3 khe API Key & tổng số khóa hoạt động khớp logic ưu tiên của subtrans/config.py:
    1) os.environ luôn thắng file .env cục bộ của workspace (hành vi load_dotenv(..., override=False)).
       Lưu ý: luôn đọc file workspace dotenv_path truyền vào (kể cả khi pytest cô lập đặt SUBTRANS_DISABLE_DOTENV=1
       để chặn python-dotenv tự đọc .env gốc của repo).
    2) Hỗ trợ GEMINI_API_KEYS (phân tách ',' hoặc ';'), GEMINI_API_KEY và GEMINI_API_KEY_1..N.
    """
    dotenv_map = _read_dotenv_map(dotenv_path)

    def _effective_var(name: str) -> tuple[str, str]:
        if os.environ.get(name) and os.environ[name].strip():
            return os.environ[name].strip(), "env"
        if dotenv_map.get(name) and dotenv_map[name].strip():
            return dotenv_map[name].strip(), "dotenv"
        return "", ""

    multi_raw, multi_src = _effective_var("GEMINI_API_KEYS")
    multi_tokens = _split_gemini_key_tokens(multi_raw)

    prim_raw, prim_src = _effective_var("GEMINI_API_KEY")
    prim_tokens = _split_gemini_key_tokens(prim_raw)

    slots_summary: List[Dict[str, Any]] = []
    for idx in (1, 2, 3):
        slot_key_name = f"GEMINI_API_KEY_{idx}"
        slot_val, slot_src = _effective_var(slot_key_name)
        slot_tokens = _split_gemini_key_tokens(slot_val)

        chosen_val = ""
        chosen_src = ""
        chosen_var = slot_key_name

        # Ưu tiên tuyệt đối: biến từ process env (override=False) theo thứ tự GEMINI_API_KEYS -> GEMINI_API_KEY -> GEMINI_API_KEY_N
        if multi_src == "env" and idx - 1 < len(multi_tokens):
            chosen_val, chosen_src, chosen_var = multi_tokens[idx - 1], "env", "GEMINI_API_KEYS"
        elif idx == 1 and prim_src == "env" and prim_tokens:
            chosen_val, chosen_src, chosen_var = prim_tokens[0], "env", "GEMINI_API_KEY"
        elif slot_src == "env" and slot_tokens:
            chosen_val, chosen_src, chosen_var = slot_tokens[0], "env", slot_key_name
        elif idx >= 2 and prim_src == "env" and idx - 1 < len(prim_tokens):
            chosen_val, chosen_src, chosen_var = prim_tokens[idx - 1], "env", "GEMINI_API_KEY"
        # Kế đến: biến từ .env theo đúng thứ tự ưu tiên của subtrans/config.py
        elif multi_src == "dotenv" and idx - 1 < len(multi_tokens):
            chosen_val, chosen_src, chosen_var = multi_tokens[idx - 1], "dotenv", "GEMINI_API_KEYS"
        elif idx == 1 and prim_src == "dotenv" and prim_tokens:
            chosen_val, chosen_src, chosen_var = prim_tokens[0], "dotenv", "GEMINI_API_KEY"
        elif slot_src == "dotenv" and slot_tokens:
            chosen_val, chosen_src, chosen_var = slot_tokens[0], "dotenv", slot_key_name
        elif idx >= 2 and prim_src == "dotenv" and idx - 1 < len(prim_tokens):
            chosen_val, chosen_src, chosen_var = prim_tokens[idx - 1], "dotenv", "GEMINI_API_KEY"

        slots_summary.append(
            {
                "slot": idx,
                "env_name": chosen_var,
                "configured": bool(chosen_val),
                "masked": _mask_secret(chosen_val) if chosen_val else "",
                "key_id": key_id_fn(chosen_val) if (chosen_val and key_id_fn) else ("configured" if chosen_val else "none"),
                "source": chosen_src,
            }
        )

    ordered_pool: List[str] = []
    seen_keys: set[str] = set()

    def _add_tokens(toks: List[str]) -> None:
        for t in toks:
            if t not in seen_keys:
                seen_keys.add(t)
                ordered_pool.append(t)

    _add_tokens(multi_tokens)
    _add_tokens(prim_tokens)
    all_var_names = set(dotenv_map.keys()) | set(os.environ.keys())
    for extra_name in sorted(k for k in all_var_names if k.startswith("GEMINI_API_KEY_")):
        e_val, _ = _effective_var(extra_name)
        _add_tokens(_split_gemini_key_tokens(e_val))

    return slots_summary, len(ordered_pool)


def _probe_duration_and_dims(video_path: Path) -> Dict[str, Any]:
    """Đọc nhanh thời lượng (giây) và kích thước khung hình qua PyAV (hoặc fallback mặc định)."""
    try:
        import av
        with av.open(str(video_path)) as container:
            if container.streams.video:
                vs = container.streams.video[0]
                dur = float(container.duration / av.time_base) if container.duration else 0.0
                return {
                    "duration": round(dur, 3),
                    "width": int(vs.width or 0),
                    "height": int(vs.height or 0),
                }
    except Exception:
        pass
    return {"duration": 0.0, "width": 0, "height": 0}


def _default_extract_frame(video_path: Path, timestamp_sec: float, out_png: Path) -> None:
    """Trích xuất 1 khung hình PNG chuẩn ma trận màu BT.709 (YUV Limited -> RGB Full)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, float(timestamp_sec)):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc",
        str(out_png),
    ]
    startupinfo = None
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=25.0,
        check=False,
        startupinfo=startupinfo,
    )
    if res.returncode != 0 or not out_png.is_file():
        raise RuntimeError(
            f"FFmpeg không thể trích xuất khung hình tại {timestamp_sec:.2f}s từ '{video_path.name}'."
        )


def _default_open_media(target_path: Path, prefer_mpv: bool = True) -> str:
    """Mở file media trên máy cục bộ Windows qua mpv (nếu có) hoặc trình phát mặc định."""
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


# =====================================================================
# APPLICATION FACTORY
# =====================================================================

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
    cleaned = str(raw_val).strip()
    _validate_no_nul_str(cleaned, "output_dir")
    p = Path(os.path.expanduser(cleaned))
    if not p.is_absolute():
        p = root / p
    try:
        resolved = p.resolve()
    except OSError:
        return None
    if is_secret_filename(resolved.name):
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


def _collect_known_output_dirs(
    root: Path,
    job_manager: JobManager,
    extra_dir: Optional[str] = None,
) -> List[Path]:
    """Gom và khử trùng lặp mọi thư mục xuất (-o) hiện hữu từ out/, live_status.json, JobManager và query param."""
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
    """Gom các thư mục video nguồn đã biết từ workspace, out_dirs, live_status.json và hàng đợi JobManager."""
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


def _enable_mock_studio_state(job_manager: JobManager, telemetry_hub: TelemetryHub) -> None:
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

    orig_get_state = job_manager.get_state

    def _mock_get_state() -> Dict[str, Any]:
        real_state = orig_get_state()
        if real_state.get("active_job") is not None:
            return real_state
        combined_history = list(real_state.get("history") or []) + [dict(h) for h in mock_history]
        return {
            "status": "running",
            "active_job": dict(mock_active_job),
            "history": combined_history[:25],
        }

    job_manager.get_state = _mock_get_state  # type: ignore[method-assign]
    telemetry_hub.gpu_static = {
        "gpu_name": "NVIDIA GeForce RTX 3060 12GB (Mock)",
        "driver_version": "Mock 572.x",
        "vram_total_mib": 12288.0,
    }

    tick_counter = {"n": 0}
    total_chunks = 12
    total_frames = 34560

    def _mock_read_pipeline_state() -> Dict[str, Any]:
        n = tick_counter["n"]
        tick_counter["n"] += 1
        completed_chunks = min(total_chunks - 1, 4 + (n % 5))
        done_frames = min(total_frames, completed_chunks * (total_frames // total_chunks) + (n % 6) * 120)
        chunks = []
        for idx in range(total_chunks):
            if idx < completed_chunks:
                st = "done"
            elif idx == completed_chunks:
                st = "running"
            else:
                st = "pending"
            chunks.append({"idx": idx, "chunk_id": idx + 1, "status": st})
        latest_fps = round(41.2 + (n % 4) * 0.4, 1)
        return {
            "active_source": "web",
            "gpu_lock": {"locked": True, "pid": 90420, "mock": True},
            "graceful_pause_pending": False,
            "effective_stage": "upscaling",
            "pid_alive": True,
            "mock_mode": True,
            "live": {
                "video_name": "[Demo] Anime_EP02 (1080p).mkv",
                "pid": 90420,
                "job_id": "mock-upscale-ep02",
                "job_type": "upscale",
                "started_at": now_str,
                "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "stage": "upscaling",
                "stage_detail": f"Mock TensorRT 10.16 FP16 Zero-Disk 4K ({completed_chunks + 1}/{total_chunks} chunks)",
                "model_id": "animevideov3",
                "zero_disk": True,
                "total_frames": total_frames,
                "unique_frames": 18240,
                "done_frames": done_frames,
                "total_chunks": total_chunks,
                "completed_chunks": completed_chunks,
                "chunks": chunks,
                "resumed_chunks": [1, 2],
                "active_chunk": completed_chunks + 1,
                "steady_fps": 41.2,
                "latest_fps": latest_fps,
                "effective_fps": 41.5,
                "eta_seconds": max(15, 620 - n * 5),
            },
            "checkpoints": [],
            "job": _mock_get_state(),
        }

    def _mock_logs_with_cursor(max_lines: int = 120) -> tuple[List[str], Dict[str, Any]]:
        step = tick_counter["n"]
        lines = [
            f"[mock] Loaded TensorRT 10.16 FP16 engine: animevideov3 (batch=8, zero_disk=True)",
            f"[mock] Dedup DIS plan: 18240/34560 unique frames (-47.2% savings)",
            f"[mock] Chunk {4 + (step % 5)}/12: 41.2 fps | p010le Main10 | VRAM 6.0/12.0 GB",
            f"[mock] Subtrans EP01 completed offline QA: pronoun_consistency=100% (0 CJK leaks)",
        ][-max_lines:]
        return lines, {"path": "mock://telemetry.log", "prefix_len": len(lines), "sig": "mock", "offset": step}

    def _mock_incremental_logs(cursor: Optional[Dict[str, Any]]) -> tuple[List[str], Dict[str, Any]]:
        step = tick_counter["n"]
        prev_off = int((cursor or {}).get("offset") or 0)
        if step <= prev_off:
            return [], dict(cursor or {"path": "mock://telemetry.log", "prefix_len": 0, "sig": "mock", "offset": step})
        line = f"[mock] SSE tick #{step}: chunk {4 + (step % 5)}/12 @ 41.{step % 6} fps | temp 62C"
        return [line], {"path": "mock://telemetry.log", "prefix_len": 1, "sig": "mock", "offset": step, "reset": False}

    telemetry_hub.read_pipeline_state = _mock_read_pipeline_state  # type: ignore[method-assign]
    telemetry_hub.read_recent_logs_with_cursor = _mock_logs_with_cursor  # type: ignore[method-assign]
    telemetry_hub._read_incremental_logs_from_cursor = _mock_incremental_logs  # type: ignore[method-assign]


def create_app(
    workspace_root: Optional[Path] = None,
    gpu_query_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    popen_factory: Optional[Callable[..., Any]] = None,
    open_media_fn: Optional[Callable[[Path, bool], str]] = None,
    frame_extract_fn: Optional[Callable[[Path, float, Path], None]] = None,
    allow_lan: bool = False,
    allow_lan_control: bool = False,
    mock_mode: bool = False,
) -> FastAPI:
    """Khởi tạo FastAPI app gắn với thư mục gốc dự án (hỗ trợ cô lập hoàn toàn khi test)."""
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
    if is_mock:
        _enable_mock_studio_state(job_manager, telemetry_hub)
    media_opener = open_media_fn or _default_open_media
    extractor = frame_extract_fn or _default_extract_frame

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await telemetry_hub.shutdown()

    app = FastAPI(
        title="AI Video Upscaler & Subtrans Studio",
        version="1.0.0",
        docs_url="/api/docs",
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
            if (
                request.method not in {"GET", "HEAD", "OPTIONS"}
                or req_path == "/api/fs/browse"
                or req_path.startswith("/api/fs/")
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

    # -----------------------------------------------------------------
    # 1. LIVE MONITOR & TELEMETRY SSE
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # 2. FILE SYSTEM BROWSER, QUEUE & JOB CONTROLS
    # -----------------------------------------------------------------
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

    @app.post("/api/jobs/start")
    def api_start_upscale(req: StartUpscaleRequest) -> Dict[str, Any]:
        try:
            job = job_manager.start_upscale_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/preview")
    def api_start_preview(req: StartPreviewRequest) -> Dict[str, Any]:
        try:
            job = job_manager.start_preview_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/subtrans")
    def api_start_subtrans(req: StartSubtransRequest) -> Dict[str, Any]:
        """Khởi chạy dịch phụ đề trực tiếp ở mọi khung giờ không bị khóa cứng."""
        try:
            job = job_manager.start_subtrans_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/subs-pending")
    def api_start_subs_pending(req: StartSubsPendingRequest) -> Dict[str, Any]:
        try:
            job = job_manager.start_subs_pending_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/pause-graceful")
    def api_pause_graceful() -> Dict[str, Any]:
        try:
            return job_manager.request_graceful_pause()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/clear-pause")
    def api_clear_pause() -> Dict[str, Any]:
        return job_manager.clear_graceful_pause()

    @app.post("/api/jobs/abort")
    def api_abort_job() -> Dict[str, Any]:
        return job_manager.abort_job()

    @app.post("/api/open-media")
    def api_open_media(req: OpenMediaRequest) -> Dict[str, Any]:
        cleaned = (req.path or "").strip()
        if not cleaned or "\x00" in cleaned:
            raise HTTPException(status_code=400, detail="Đường dẫn không hợp lệ.")
        cand = Path(os.path.expanduser(cleaned))
        if not cand.is_absolute():
            cand = (root / cand).resolve()
        else:
            cand = cand.resolve()

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
        """Quét out/, out/audit và các thư mục con cấp 1 hợp lệ (vd: out/Season1, out/benchmark_*)."""
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

    # -----------------------------------------------------------------
    # 3. PREVIEW 1080p vs 4K SPLIT-COMPARE
    # -----------------------------------------------------------------
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

        # Lập bản đồ từ audit.json để biết chính xác source_file & output_file gốc across all out_dirs
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

            # Đoán tên file gốc nếu là bản preview (vd: Ep01_preview_10s_4K -> Ep01)
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
        src_p = Path(os.path.expanduser(req.source_path.strip()))
        up_p = Path(os.path.expanduser(req.upscaled_path.strip()))
        if not src_p.is_absolute():
            src_p = (root / src_p).resolve()
        else:
            src_p = src_p.resolve()
        if not up_p.is_absolute():
            up_p = (root / up_p).resolve()
        else:
            up_p = up_p.resolve()

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

        up_ts = max(0.0, float(req.timestamp_sec))
        if up_meta["duration"] > 0 and up_ts > max(0.0, up_meta["duration"] - 0.08):
            up_ts = max(0.0, round(up_meta["duration"] - 0.1, 3))

        if req.source_offset_sec is not None:
            offset_s = float(req.source_offset_sec)
        else:
            m_prev = re.match(r"^(.*)_preview_(\d+)s_4K(?:_animejanai)?$", up_p.stem, flags=re.IGNORECASE)
            if m_prev and src_meta["duration"] > 0:
                prev_dur = int(m_prev.group(2))
                max_dur = max(1, int(round(src_meta["duration"])))
                offset_s = (
                    0.0
                    if prev_dur >= max_dur
                    else float(max(0, min(int(src_meta["duration"] * 0.3), max_dur - prev_dur)))
                )
            else:
                offset_s = 0.0

        src_ts = max(0.0, up_ts + offset_s)

        compare_dir = root / "scratch" / "web_compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            f"{src_p}|{src_p.stat().st_mtime_ns}|{up_p}|{up_p.stat().st_mtime_ns}|{src_ts:.3f}|{up_ts:.3f}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]

        src_png = compare_dir / f"src_{digest}.png"
        up_png = compare_dir / f"up_{digest}.png"

        try:
            if not src_png.is_file():
                extractor(src_p, src_ts, src_png)
            if not up_png.is_file():
                extractor(up_p, up_ts, up_png)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Giữ tối đa 24 ảnh tạm mới nhất để tiết kiệm dung lượng ổ đĩa
        try:
            pngs = sorted(compare_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old_f in pngs[24:]:
                try:
                    old_f.unlink()
                except OSError:
                    pass
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
        if not frame_file.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy khung hình trích xuất.")
        return FileResponse(str(frame_file), media_type="image/png")

    # -----------------------------------------------------------------
    # 4. MODELS REGISTRY & SHOWS.YAML CONFIGURATOR
    # -----------------------------------------------------------------
    @app.get("/api/config/models-and-shows")
    def api_models_and_shows() -> Dict[str, Any]:
        models_file = root / "models" / "models.yaml"
        shows_file = root / "shows.yaml"
        trt_cache_dir = root / ".cache" / "trt_engines"

        models_list: List[Dict[str, Any]] = []
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

                trt_engines: List[str] = []
                if trt_cache_dir.is_dir():
                    safe_id = str(m_id).replace(" ", "_").replace("/", "_")
                    prefixes = {safe_id}
                    if m_id == "animevideov3":
                        prefixes.add("realesr-animevideov3")
                    for ef in trt_cache_dir.glob("*.engine"):
                        if any(ef.name.startswith(f"{pfx}_") for pfx in prefixes):
                            trt_engines.append(ef.name)

                med_fps, p5_fps = MODEL_REFERENCE_FPS.get(str(m_id), (10.5, 8.5))
                arch = info.get("arch") or {}
                models_list.append({
                    "id": str(m_id),
                    "name": str(info.get("name") or m_id),
                    "arch_type": str(arch.get("type") or "Unknown"),
                    "num_feat": arch.get("num_feat"),
                    "num_conv": arch.get("num_conv"),
                    "native_upscale": arch.get("upscale", 2),
                    "downsample": str(info.get("downsample") or "none"),
                    "weights_path": rel_w,
                    "weights_ready": w_exists,
                    "weights_size_mb": w_size_mb,
                    "trt_cached": len(trt_engines) > 0,
                    "trt_engine_files": trt_engines,
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

        return {
            "models": models_list,
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
            if mod and valid_models and mod.lower() not in valid_models:
                raise ValueError(
                    f"Rule '{pat}' trỏ tới model '{mod}' không tồn tại trong models/models.yaml."
                )
            pol = str(item.get("dedup_policy") or "").strip()
            if pol and pol.lower() not in DEDUP_POLICY_PARAMS:
                raise ValueError(
                    f"Rule '{pat}' dùng dedup_policy '{pol}' không hợp lệ "
                    f"(chấp nhận: {', '.join(DEDUP_POLICY_PARAMS.keys())})."
                )
            validated.append(item)
        return validated

    @app.post("/api/config/test-match")
    def api_test_shows_match(req: TestShowsMatchRequest) -> Dict[str, Any]:
        sample = (req.sample_path or "").strip()
        if not sample:
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

            vid_name = os.path.basename(sample.rstrip("\\/"))
            parent_dir = os.path.basename(os.path.dirname(sample)) if ("/" in sample or "\\" in sample) else ""
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
                if match_show_pattern(vid_name, pat) or (parent_dir and match_show_pattern(parent_dir, pat)):
                    m_target = str(r.get("model") or "").strip().lower()
                    if m_target and m_target in lower_to_key:
                        chosen_mod = lower_to_key[m_target]
                        src_desc = f"shows.yaml ('{pat}')"
                        matched_model_pat = pat
                        break
            for r in rules:
                if not isinstance(r, dict):
                    continue
                pat = str(r.get("pattern") or "").strip()
                if not pat:
                    continue
                if match_show_pattern(vid_name, pat) or (parent_dir and match_show_pattern(parent_dir, pat)):
                    p_target = str(r.get("dedup_policy") or "").strip().lower()
                    if p_target and p_target in DEDUP_POLICY_PARAMS:
                        chosen_pol = p_target
                        matched_dedup_pat = pat
                        break
            matched_pat = matched_model_pat or matched_dedup_pat

            med_fps, _ = MODEL_REFERENCE_FPS.get(chosen_mod, (10.5, 8.5))
            return {
                "sample": sample,
                "model_id": chosen_mod,
                "model_name": str((reg.get(chosen_mod) or {}).get("name") or chosen_mod),
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
        text = req.yaml_content.replace("\r\n", "\n").rstrip("\n") + "\n"
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, shows_path)
        return {"ok": True, "rules_count": len(rules)}

    # -----------------------------------------------------------------
    # 5. SUBTRANS STUDIO & AUDIT VAULT
    # -----------------------------------------------------------------
    @app.get("/api/subtrans/overview")
    def api_subtrans_overview() -> Dict[str, Any]:
        now_pt = get_now_pt()
        next_pt, next_vn = get_next_pt_reset(now_pt)
        today_pt_str = now_pt.strftime("%Y-%m-%d")
        reset_seconds = max(0, int((next_pt - now_pt).total_seconds()))

        # Kiểm tra trạng thái các khe API Key khớp 100% subtrans/config.py (chỉ trả dạng che, tuyệt đối không lộ khóa thật)
        key_slots, active_key_count = _resolve_subtrans_key_slots(root / ".env", key_id_fn=compute_key_id)

        # Đọc subtrans/models.yaml
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
        elif is_mock:
            cache_stats = {
                "db_exists": True,
                "mock_mode": True,
                "total_cached_cues": 1420,
                "today_cached_cues": 318,
                "today_requests": [
                    {"model": "gemini-3.5-flash-lite", "status": "ok", "requests": 14, "cues": 318, "tokens": 24800}
                ],
                "exhausted_keys": [],
                "exhausted_models": [],
            }

        # Đọc thống kê SQLite ở chế độ CHỈ ĐỌC (không bao giờ tạo/ghi vào DB production khi GET)
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
                        WHERE pt_date = ?
                        GROUP BY model, status
                        ORDER BY COUNT(*) DESC
                        """,
                        (today_pt_str,),
                    )
                    cache_stats["today_requests"] = [
                        {
                            "model": str(r[0] or ""),
                            "status": str(r[1] or ""),
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
                    cache_stats["exhausted_models"] = [
                        {"model": str(r[0]), "until_pt": str(r[1] or ""), "reason": str(r[2] or "")}
                        for r in cur.fetchall()
                    ]
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
                    {"model": "gemini-3.5-flash-lite", "status": "ok", "requests": 14, "cues": 318, "tokens": 24800}
                ],
                "exhausted_keys": [],
                "exhausted_models": [],
            }

        # Danh sách glossaries/*.yaml
        glossaries_dir = root / "glossaries"
        glossaries: List[Dict[str, str]] = []
        if glossaries_dir.is_dir():
            for gf in sorted(glossaries_dir.glob("*.yaml")):
                glossaries.append({"name": gf.name, "path": str(gf.resolve())})

        # Pending subtitles queue (check state/, legacy root, then out/)
        pending_items: List[Dict[str, Any]] = []
        for p_cand in (root / "state" / "pending_subs.json", root / "pending_subs.json", root / "out" / "pending_subs.json"):
            if p_cand.is_file():
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
            "configured_keys_count": active_key_count,
            "models": sub_models,
            "chain": chain,
            "cache": cache_stats,
            "glossaries": glossaries,
            "pending_subs": (
                pending_items
                if pending_items or not is_mock
                else [{"episode": "[Demo] Anime_EP02 (1080p)", "status": "queued", "subs_quality": "high"}]
            ),
            "mock_mode": is_mock,
        }

    @app.put("/api/subtrans/api-keys")
    def api_update_subtrans_keys(req: UpdateApiKeysRequest) -> Dict[str, Any]:
        """Cập nhật hoặc xóa các khe GEMINI_API_KEY_1..3 vào .env cục bộ (không bao giờ trả hay log khóa thô)."""
        dotenv_path = root / ".env"
        existing = _read_dotenv_map(dotenv_path)

        updates: Dict[str, Optional[str]] = {
            "GEMINI_API_KEY_1": req.key_1,
            "GEMINI_API_KEY_2": req.key_2,
            "GEMINI_API_KEY_3": req.key_3,
        }

        # Nếu .env cũ đang dùng GEMINI_API_KEYS gộp (comma/semicolon-separated) và người dùng sửa/xóa slot,
        # tách các khe còn lại sang GEMINI_API_KEY_1..3 và gỡ GEMINI_API_KEYS để không che bóng bản cập nhật mới.
        has_any_edit = any(v is not None for v in (req.key_1, req.key_2, req.key_3))
        if has_any_edit and existing.get("GEMINI_API_KEYS"):
            old_multi = _split_gemini_key_tokens(existing.get("GEMINI_API_KEYS"))
            for idx in range(1, max(4, len(old_multi) + 1)):
                k_name = f"GEMINI_API_KEY_{idx}"
                if idx <= 3:
                    if updates[k_name] is None and not existing.get(k_name) and idx - 1 < len(old_multi):
                        updates[k_name] = old_multi[idx - 1]
                elif k_name not in updates and not existing.get(k_name) and idx - 1 < len(old_multi):
                    updates[k_name] = old_multi[idx - 1]
            updates["GEMINI_API_KEYS"] = ""

        # Luôn đồng bộ GEMINI_API_KEY chính với slot 1 khi slot 1 được cập nhật hoặc xóa
        # để subtrans/config.py (vốn ưu tiên GEMINI_API_KEY trước GEMINI_API_KEY_1) không bị khóa cũ che mất.
        if req.key_1 is not None:
            clean_1 = req.key_1.strip()
            updates["GEMINI_API_KEY"] = clean_1

        try:
            _write_dotenv_keys(dotenv_path, updates)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # Không chép khóa từ .env vào os.environ của tiến trình Web Studio: tiến trình con upscale/subtrans
        # tự gọi load_dotenv(override=False) khi khởi chạy, còn giữ os.environ sạch giúp các khe còn lại
        # không bị gắn nhãn nhầm [ENV] và vẫn giữ nguyên nút Xóa / chỉnh sửa trực tiếp trong UI.
        slots_summary, active_key_count = _resolve_subtrans_key_slots(dotenv_path, key_id_fn=compute_key_id)
        return {
            "ok": True,
            "key_slots": slots_summary,
            "configured_count": active_key_count,
        }

    @app.get("/api/audits")
    def api_list_audits(output_dir: Optional[str] = Query(default=None)) -> Dict[str, Any]:
        try:
            out_dirs = _collect_known_output_dirs(root, job_manager, extra_dir=output_dir)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

        for af, base_out in discovered_entries[:60]:
            try:
                st = af.stat()
                mtime_val = st.st_mtime
                size_kb_val = round(st.st_size / 1024.0, 1)
            except OSError:
                mtime_val = 0.0
                size_kb_val = 0.0
            raw_stem = af.name[:-len(".audit.json")] if af.name.endswith(".audit.json") else af.stem
            corrupt_ep_label = re.sub(r"_4K(?:_animejanai)?$", "", raw_stem, flags=re.IGNORECASE)
            try:
                adata = json.loads(af.read_text(encoding="utf-8"))
            except Exception as exc:
                audits.append({
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
                })
                continue
            if not isinstance(adata, dict):
                audits.append({
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
                })
                continue

            stem_4k = af.name[:-len(".audit.json")]
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

            audits.append({
                "episode": episode_label,
                "audit_json_path": str(af.resolve()),
                "output_file": str(adata.get("output_file") or ""),
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
                "completed_at": str(adata.get("completed_at") or ""),
                "sub_audit_md_path": sub_audit_md,
                "cast_yaml_path": cast_yaml,
            })

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

        return {
            "audits": audits,
            "checkpoints": scan_temp_checkpoints(root, extra_dirs=out_dirs),
        }

    @app.get("/api/audits/detail")
    def api_audit_detail(path: str = Query(...)) -> Any:
        if "\x00" in path:
            raise HTTPException(status_code=400, detail="Đường dẫn file audit chứa ký tự NUL (\\x00) không hợp lệ.")
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

        # Cho phép đọc biên bản trong workspace_root hoặc bất kỳ thư mục xuất (-o) đã biết
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

    return app
