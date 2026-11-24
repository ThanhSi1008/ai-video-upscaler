"""Safe local filesystem browser and show-routing resolver for Web Studio."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import string
import sys
from typing import Any, Dict, List, Optional, Sequence
import yaml

from src.show_routing import match_show_pattern, match_show_rule

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".webm", ".avi", ".mov", ".ts", ".m2ts"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}
_WINDOWS_PROTECTED_DIR_NAMES = frozenset({
    "$recycle.bin",
    "system volume information",
    "recovery",
    "config.msi",
    "msocache",
    "documents and settings",
})

# Pure-stdlib mirror of src.dedup.DEDUP_POLICY_PARAMS to avoid importing torch in the web process.
_DEDUP_HI = 768
_DEDUP_LO = 320
_DEDUP_FRAC = 0.33
DEDUP_POLICY_PARAMS: Dict[str, Dict[str, Any]] = {
    "default": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC},
    "sensitive": {"hi": 512, "lo": 160, "frac": 0.15},
    "sensitive-a": {"hi": 512, "lo": 160, "frac": 0.15},
    "sensitive-b": {"hi": 384, "lo": 128, "frac": 0.10},
    "motion": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "tau": 0.3},
    "drift": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "drift_delta": 0.75, "min_response": 0.10},
    "drift-0.5": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "drift_delta": 0.5, "min_response": 0.10},
    "drift-0.75": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "drift_delta": 0.75, "min_response": 0.10},
    "drift-1.0": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "drift_delta": 1.0, "min_response": 0.10},
    "dis": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "dis_thresh": 0.30},
    "dis-auto": {
        "hi": _DEDUP_HI,
        "lo": _DEDUP_LO,
        "frac": _DEDUP_FRAC,
        "dis_thresh": 0.30,
        "auto_noise_floor": True,
        "lip_flap_guard": True,
    },
    "dis-0.2": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "dis_thresh": 0.20},
    "dis-0.3": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "dis_thresh": 0.30},
    "dis-0.4": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "dis_thresh": 0.40},
    "dis-0.5": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "dis_thresh": 0.50},
    "dis-sakuga": {"hi": 1024, "lo": 448, "frac": 0.38, "dis_thresh": 0.35, "local_thresh": 1.50, "min_moving_area_pct": 12.0},
    "sakuga": {"hi": 1024, "lo": 448, "frac": 0.38, "dis_thresh": 0.35, "local_thresh": 1.50, "min_moving_area_pct": 12.0},
    "off": {"all_keep": True},
}

AUTO_PARETO_MODEL_ID = "auto-pareto"
AUTO_PARETO_MODEL_NAME = "AnimeJaNai HD V3 Auto-Pareto (Compact 16c + UltraCompact 8c)"
VALID_NVENC_MULTIPASS_POLICIES = ("auto", "fullres", "qres")
VALID_GRAIN_RESTORE_POLICIES = ("auto", "off", "light", "heavy")


def is_valid_dedup_policy(policy: str) -> bool:
    """Validate static dedup policies and dynamic dis-auto-<float>/dis-<float>/drift-<float> variants."""
    pol = str(policy or "").strip().lower()
    if not pol:
        return False
    if pol in DEDUP_POLICY_PARAMS:
        return True
    for prefix in ("dis-auto-", "dis-", "drift-"):
        if pol.startswith(prefix):
            val_str = pol[len(prefix):]
            try:
                val = float(val_str)
            except ValueError:
                return False
            return math.isfinite(val) and val > 0.0
    return False


def natural_sort_key(name: str) -> List[Any]:
    """Return a numeric-aware sort key so episode numbers sort naturally."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(name))]


_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "\u201c": "\u201d",
    "\u201d": "\u201c",
    "\u2018": "\u2019",
    "\u2019": "\u2018",
}


def strip_wrapping_quotes(raw: Optional[str]) -> str:
    """Strip matching ASCII or smart outer quote pairs around a pasted filesystem path."""
    s = str(raw or "").strip()
    while len(s) >= 2:
        first, last = s[0], s[-1]
        if (first in ('"', "'") and last == first) or _QUOTE_PAIRS.get(first) == last:
            s = s[1:-1].strip()
        else:
            break
    return s


def unwrap_outer_quotes_preserving_trailing_space(raw: Optional[str]) -> str:
    """Unwrap outer quote pairs without stripping trailing spaces on path segments before validation."""
    if raw is None:
        return ""
    raw_s = str(raw)
    trimmed = raw_s.strip()
    if not trimmed:
        return ""
    if len(trimmed) >= 2 and (
        (trimmed[0] in ('"', "'") and trimmed[-1] == trimmed[0])
        or _QUOTE_PAIRS.get(trimmed[0]) == trimmed[-1]
    ):
        cur = trimmed
        while len(cur) >= 2 and (
            (cur[0] in ('"', "'") and cur[-1] == cur[0])
            or _QUOTE_PAIRS.get(cur[0]) == cur[-1]
        ):
            cur = cur[1:-1]
        return cur
    return raw_s.lstrip()


def is_secret_filename(name: str) -> bool:
    """Block access to secret .env files over HTTP/LAN while permitting templates."""
    low = strip_wrapping_quotes(name).lower()
    if low in {".env.example", ".env.sample", ".env.template"}:
        return False
    return low == ".env" or low.startswith(".env.")


def is_unc_path(raw: Optional[str]) -> bool:
    """Return True if raw string targets a Windows/POSIX UNC share or raw device namespace."""
    s = strip_wrapping_quotes(raw)
    if not s:
        return False
    norm = s.replace("/", "\\")
    if re.match(r"^\\\\\?\\[A-Za-z]:(?:\\|$)", norm):
        return False
    return norm.startswith("\\\\") or norm.lower().startswith("\\??\\unc\\")


def is_upscaled_output_stem(stem: str) -> bool:
    """Return True if filename stem already denotes a rendered 4K output."""
    low = stem.lower()
    return low.endswith("_4k") or low.endswith("_4k_animejanai")


_FORBIDDEN_WIN32_CHARS = frozenset('<>:"|?*')
_RESERVED_WIN32_STEMS = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
        "com\u00b9",
        "com\u00b2",
        "com\u00b3",
        "lpt\u00b9",
        "lpt\u00b2",
        "lpt\u00b3",
    }
)


def is_single_path_reparse_or_symlink(p: Path) -> bool:
    """Return True before resolve() if path entry itself is a symlink, Win32 junction, or reparse point."""
    try:
        s = str(p)
        if p.is_symlink() or os.path.islink(s) or (hasattr(os.path, "isjunction") and os.path.isjunction(s)):
            return True
        if os.path.lexists(s):
            return bool(getattr(p.lstat(), "st_file_attributes", 0) & 0x0400)
    except OSError:
        return False
    return False


def has_reparse_or_symlink_in_chain(raw_path: Path, workspace_root: Optional[Path] = None) -> bool:
    """Return True before resolve() if raw_path or any non-system ancestor is a symlink/reparse point."""
    import tempfile

    if is_single_path_reparse_or_symlink(raw_path):
        return True
    safe_roots: set[str] = set()
    for r_cand in (str(workspace_root) if workspace_root else None, tempfile.gettempdir()):
        if r_cand:
            cur_r = os.path.normcase(os.path.abspath(r_cand)).rstrip(os.sep)
            while cur_r:
                safe_roots.add(cur_r)
                par_r = os.path.dirname(cur_r).rstrip(os.sep)
                if par_r == cur_r:
                    break
                cur_r = par_r

    orig_abs = raw_path.is_absolute()
    if ".." in raw_path.parts:
        acc = Path(raw_path.anchor) if orig_abs else (workspace_root or Path.cwd())
        rel_parts = raw_path.parts[1:] if orig_abs else raw_path.parts
        for part in rel_parts:
            if part == "..":
                acc = acc.parent
            elif part != ".":
                acc = acc / part
                acc_norm = os.path.normcase(os.path.abspath(str(acc))).rstrip(os.sep)
                if acc_norm not in safe_roots and is_single_path_reparse_or_symlink(acc):
                    return True

    target_raw = raw_path if orig_abs else ((workspace_root / raw_path) if workspace_root else raw_path)
    cur = Path(os.path.abspath(str(target_raw)))
    while True:
        cur_norm = os.path.normcase(str(cur)).rstrip(os.sep)
        if cur_norm in safe_roots:
            break
        if is_single_path_reparse_or_symlink(cur):
            return True
        if cur.parent == cur:
            break
        cur = cur.parent
    return False
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_BLENDED_PS2X_SHA8_RE = re.compile(r"_ps2x_([0-9a-f]{8})_")
_TRT_HW_PROFILE_RE = re.compile(r"_(\d{2,5}x\d{2,5})_b1-\d+\.engine$", re.IGNORECASE)
_SUBTITLE_ARTIFACT_SUFFIXES = tuple(
    f"{mid}{ext}"
    for mid in (".vi", ".vi.partial", ".partial", ".review")
    for ext in (".srt", ".ass", ".ssa", ".vtt")
)
_SUB_JA_STEM_RE = re.compile(r"(?:\[cc\])?(?:[._]whisper)?(?:[._](?:ja|jpn))(?:\[cc\])?$", re.IGNORECASE)
_SUB_EN_STEM_RE = re.compile(r"(?:\[cc\])?(?:[._](?:en|eng))(?:\[cc\])?$", re.IGNORECASE)
_TRT_SIDECAR_CACHE: Dict[tuple[str, int, int, int, tuple[str, ...]], bool] = {}


def filter_paired_subtitle_paths(sub_paths: Sequence[str]) -> List[str]:
    """Exclude output artifacts and English sidecars that have a sibling non-EN subtitle in the same folder."""
    cleaned = [
        p for p in sub_paths if not os.path.basename(p).lower().endswith(_SUBTITLE_ARTIFACT_SUFFIXES)
    ]
    non_en_keys = {
        (
            os.path.normcase(os.path.dirname(os.path.abspath(p))),
            _SUB_JA_STEM_RE.sub("", os.path.splitext(os.path.basename(p))[0]).lower(),
        )
        for p in cleaned
        if not _SUB_EN_STEM_RE.search(os.path.splitext(os.path.basename(p))[0])
    }
    selected: List[str] = []
    for p in cleaned:
        stem = os.path.splitext(os.path.basename(p))[0]
        if _SUB_EN_STEM_RE.search(stem):
            p_dir = os.path.normcase(os.path.dirname(os.path.abspath(p)))
            en_base = _SUB_EN_STEM_RE.sub("", stem).lower()
            if (p_dir, en_base) in non_en_keys:
                continue
        selected.append(p)
    return selected


def compute_upcunet2x_blended_sha256_stdlib(cons_sha: str, dn3x_sha: str, alpha: float) -> str:
    """Stdlib mirror of src/cuda_engine.py:compute_blended_weights_sha256 for zero-VRAM TRT checks."""
    a = round(max(0.0, min(1.0, float(alpha))), 4)
    c_clean = str(cons_sha or "").strip().lower()
    d_clean = str(dn3x_sha or "").strip().lower()
    if a <= 0.0:
        return c_clean
    if a >= 1.0:
        return d_clean
    token = f"upcunet2x_blend_v2:{c_clean}:{d_clean}:{a:.4f}".encode("utf-8")
    return hashlib.sha256(token).hexdigest().lower()


def resolve_allowed_trt_weights_shas(
    model_id: str,
    models_reg: Dict[str, Any],
    grain_restore: str = "auto",
) -> set[str]:
    """Return exact allowed 64-hex weights SHA-256 digests for model_id and grain_restore policy."""
    m_info = models_reg.get(model_id) if isinstance(models_reg, dict) else None
    if not isinstance(m_info, dict):
        return set()
    base_sha = str(m_info.get("sha256") or "").strip().lower()
    if model_id != "realcugan-pro-conservative-2x":
        return {base_sha} if base_sha else set()
    dn3x_info = models_reg.get("realcugan-pro-denoise3x-2x") if isinstance(models_reg, dict) else None
    dn3x_sha = str((dn3x_info or {}).get("sha256") or "").strip().lower() if isinstance(dn3x_info, dict) else ""
    light_sha = (
        compute_upcunet2x_blended_sha256_stdlib(base_sha, dn3x_sha, 0.45)
        if (base_sha and dn3x_sha)
        else base_sha
    )
    tier = str(grain_restore or "auto").strip().lower()
    if tier in ("off", "clean"):
        return {base_sha} if base_sha else set()
    if tier == "light":
        return {light_sha} if light_sha else set()
    if tier == "heavy":
        chosen_heavy = dn3x_sha or base_sha
        return {chosen_heavy} if chosen_heavy else set()
    return {s for s in (base_sha, light_sha, dn3x_sha) if s}


def extract_trt_hw_profiles(engine_names: Sequence[str]) -> List[str]:
    """Extract sorted unique '{H}x{W}' profile tags from verified TensorRT .engine filenames."""
    profiles: set[str] = set()
    for name in engine_names:
        m = _TRT_HW_PROFILE_RE.search(str(name))
        if m:
            profiles.add(m.group(1).lower())
    return sorted(profiles, key=natural_sort_key)


def _hash_file_sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().lower()


def verify_trt_engine_sidecar_stdlib(
    engine_file: Path,
    expected_weights_sha256: Optional[str] = None,
    allowed_weights_shas: Optional[Sequence[str] | set[str]] = None,
) -> bool:
    """Verify .engine file has a valid .sha256 sidecar matching exact engine & registered/blended weights digests."""
    try:
        if not engine_file.is_file():
            return False
        ef_st = engine_file.stat()
        if ef_st.st_size <= 0:
            return False
        sidecar = Path(f"{engine_file}.sha256")
        if not sidecar.is_file():
            return False
        sc_st = sidecar.stat()
        if sc_st.st_size <= 0:
            return False
        allowed_tuple: tuple[str, ...]
        if allowed_weights_shas is not None:
            allowed_tuple = tuple(sorted({str(s).strip().lower() for s in allowed_weights_shas if str(s).strip()}))
        else:
            exp_single = str(expected_weights_sha256 or "").strip().lower()
            allowed_tuple = (exp_single,) if exp_single else ()
        cache_key = (str(engine_file.resolve()), ef_st.st_mtime_ns, ef_st.st_size, sc_st.st_mtime_ns, allowed_tuple)
        cached = _TRT_SIDECAR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        raw = sidecar.read_text(encoding="utf-8").strip()
        if not raw:
            return False
        if raw.startswith("{"):
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return False
            rec_eng = str(parsed.get("engine_sha256") or "").strip().lower()
            rec_w = str(parsed.get("weights_sha256") or "").strip().lower()
        else:
            rec_eng = raw.split()[0].strip().lower()
            rec_w = ""
        if not _HEX64_RE.fullmatch(rec_eng):
            return False
        m_ps2x = _BLENDED_PS2X_SHA8_RE.search(engine_file.name.lower())
        if m_ps2x and (not _HEX64_RE.fullmatch(rec_w) or not rec_w.startswith(m_ps2x.group(1))):
            return False
        weights_ok = (rec_w in allowed_tuple) if allowed_tuple else True
        ok = weights_ok and (_hash_file_sha256_hex(engine_file) == rec_eng)
        if len(_TRT_SIDECAR_CACHE) >= 512:
            _TRT_SIDECAR_CACHE.clear()
        _TRT_SIDECAR_CACHE[cache_key] = ok
        return ok
    except Exception:
        return False


def inspect_model_trt_cache(
    trt_cache_dir: Path,
    model_id: str,
    expected_weights_sha256: Optional[str] = None,
    allowed_weights_shas: Optional[Sequence[str] | set[str]] = None,
    arch_type: str = "",
    native_upscale: int = 2,
) -> tuple[List[str], List[str]]:
    """Return (verified_engine_names, stale_unverified_engine_names) for model_id in trt_cache_dir."""
    valid_engines: List[str] = []
    stale_engines: List[str] = []
    if not trt_cache_dir.is_dir():
        return valid_engines, stale_engines
    safe_id = str(model_id).replace(" ", "_").replace("/", "_")
    prefixes = {safe_id}
    if model_id == "animevideov3":
        prefixes.add("realesr-animevideov3")
    req_aa2x = (str(arch_type) == "SRVGGNetCompact") and (int(native_upscale or 2) > 2)
    req_ps2x = str(arch_type) == "UpCunet2x"
    for ef in sorted(trt_cache_dir.glob("*.engine"), key=lambda p: p.name.lower()):
        if not any(ef.name.startswith(f"{pfx}_") for pfx in prefixes):
            continue
        low_ef = ef.name.lower()
        arch_tag_ok = True
        if req_aa2x and not any(low_ef.startswith(f"{pfx.lower()}_aa2x_") for pfx in prefixes):
            arch_tag_ok = False
        elif not req_aa2x and "_aa2x_" in low_ef:
            arch_tag_ok = False
        if req_ps2x and "_ps2x_" not in low_ef:
            arch_tag_ok = False
        elif not req_ps2x and "_ps2x_" in low_ef:
            arch_tag_ok = False
        if arch_tag_ok and verify_trt_engine_sidecar_stdlib(
            ef,
            expected_weights_sha256=expected_weights_sha256,
            allowed_weights_shas=allowed_weights_shas,
        ):
            valid_engines.append(ef.name)
        else:
            stale_engines.append(ef.name)
    return valid_engines, stale_engines


def validate_win32_path_hygiene(p: Path, field: str = "path", raw_str: Optional[str] = None) -> None:
    """Validate Win32 drive existence and reject illegal chars, trailing dots/spaces, and DOS stems."""
    if sys.platform != "win32":
        return
    if p.drive and not p.is_absolute():
        raise ValueError(f"Đường dẫn tương đối theo ổ đĩa tại '{field}' không hợp lệ.")
    if not p.drive and p.root:
        raise ValueError(f"Đường dẫn gốc thiếu ký tự ổ đĩa tại '{field}' không hợp lệ.")
    drive_norm = p.drive[4:] if p.drive.startswith("\\\\?\\") else p.drive
    if drive_norm and len(drive_norm) == 2 and drive_norm[0].isalpha() and drive_norm[1] == ":":
        if not Path(f"{drive_norm}\\").exists():
            raise ValueError(f"Ổ đĩa đích '{drive_norm}\\' tại '{field}' không tồn tại.")
    raw_segments: List[str] = []
    if raw_str:
        norm_bs = str(raw_str).replace("/", "\\")
        if re.match(r"^\\\\\?\\[A-Za-z]:", norm_bs):
            norm_bs = norm_bs[6:]
        elif re.match(r"^[A-Za-z]:", norm_bs):
            norm_bs = norm_bs[2:]
        raw_segments = [seg for seg in norm_bs.split("\\") if seg]
    non_drive_parts = list(p.parts[1:] if (p.drive or p.root) else p.parts)
    for part in [*raw_segments, *non_drive_parts]:
        if part in (".", ".."):
            continue
        if any(ch in _FORBIDDEN_WIN32_CHARS or ord(ch) < 32 for ch in part):
            raise ValueError(
                f"Đường dẫn tại '{field}' chứa ký tự cấm của Windows (<>:\"|?*)."
            )
        if part.endswith(".") or part.endswith(" "):
            raise ValueError(
                f"Thành phần đường dẫn tại '{field}' không được kết thúc bằng dấu chấm '.' hoặc khoảng trắng."
            )
        stem0 = part.split(".", 1)[0].strip().lower()
        if stem0 in _RESERVED_WIN32_STEMS:
            raise ValueError(
                f"Đường dẫn tại '{field}' chứa tên thiết bị hệ thống dành riêng của Windows ({stem0.upper()})."
            )


class SafeLocalFSBrowser:
    """Sandboxed local directory browser and routing resolver for Web Studio."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def _load_routing_tables(self) -> tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
        models_path = self.workspace_root / "models" / "models.yaml"
        shows_path = self.workspace_root / "shows.yaml"
        registry: Dict[str, Any] = {}
        default_id = "animevideov3"
        if models_path.is_file():
            try:
                raw = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
                registry = raw.get("models", {}) or {}
                for m_id, m_info in registry.items():
                    if isinstance(m_info, dict) and m_info.get("default"):
                        default_id = str(m_id)
                        break
            except Exception:
                pass

        shows_rules: List[Dict[str, Any]] = []
        if shows_path.is_file():
            try:
                s_raw = yaml.safe_load(shows_path.read_text(encoding="utf-8")) or {}
                shows_rules = [r for r in (s_raw.get("shows") or []) if isinstance(r, dict)]
            except Exception:
                pass
        return registry, default_id, shows_rules

    def resolve_routing_for_path(
        self,
        video_path: str | Path,
        override_model: Optional[str] = None,
        override_dedup: Optional[str] = None,
        routing_tables: Optional[tuple[Dict[str, Any], str, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, str]:
        """Resolve model_id, model_name, dedup_policy, and matched rule metadata for a video path."""
        registry, default_id, shows_rules = (
            routing_tables if routing_tables is not None else self._load_routing_tables()
        )
        lower_to_key = {str(k).lower(): str(k) for k in registry.keys()}

        vid_str = str(video_path)
        if "\x00" in vid_str:
            raise ValueError("Đường dẫn chứa ký tự NUL (\\x00) không hợp lệ.")

        chosen_model = None
        matched_rule_pattern = None
        rule_dedup = None

        if override_model and override_model.strip():
            cand = override_model.strip().lower()
            if cand == "realesr-animevideov3":
                cand = "animevideov3"
            if cand == AUTO_PARETO_MODEL_ID:
                chosen_model = AUTO_PARETO_MODEL_ID
                source_desc = f"Chỉ định trực tiếp ({chosen_model})"
            elif cand in lower_to_key:
                chosen_model = lower_to_key[cand]
                source_desc = f"Chỉ định trực tiếp ({chosen_model})"
            else:
                chosen_model = override_model.strip()
                source_desc = f"Chỉ định trực tiếp ({chosen_model})"
        else:
            source_desc = "Mặc định hệ thống (models.yaml)"

        matched_model_pattern = ""
        matched_dedup_pattern = ""
        dedup_source = "Mặc định hệ thống (default)"

        # Match model and dedup_policy in two passes to mirror src/models_resolver.py + src/dedup.py
        # (a show may declare model and dedup_policy on separate shows.yaml rules).
        for rule in shows_rules:
            if not isinstance(rule, dict):
                continue
            pat = str(rule.get("pattern", "")).strip()
            if not pat:
                continue
            if match_show_rule(vid_str, pat):
                m_target = str(rule.get("model", "")).strip().lower()
                if m_target == AUTO_PARETO_MODEL_ID or (m_target and m_target in lower_to_key):
                    if not chosen_model:
                        chosen_model = AUTO_PARETO_MODEL_ID if m_target == AUTO_PARETO_MODEL_ID else lower_to_key[m_target]
                        source_desc = f"shows.yaml ('{pat}')"
                    matched_model_pattern = pat
                    break

        for rule in shows_rules:
            if not isinstance(rule, dict):
                continue
            pat = str(rule.get("pattern", "")).strip()
            if not pat:
                continue
            if match_show_rule(vid_str, pat):
                pol_target = str(rule.get("dedup_policy", "")).strip().lower()
                if pol_target and is_valid_dedup_policy(pol_target):
                    rule_dedup = pol_target
                    dedup_source = f"shows.yaml ('{pat}')"
                    matched_dedup_pattern = pat
                    break

        matched_rule_pattern = matched_model_pattern or matched_dedup_pattern

        if not chosen_model:
            chosen_model = default_id

        norm_override_dedup = override_dedup.strip().lower() if override_dedup else ""
        if norm_override_dedup and is_valid_dedup_policy(norm_override_dedup):
            chosen_dedup = norm_override_dedup
            dedup_source = f"Chỉ định thủ công ({norm_override_dedup})"
        elif rule_dedup:
            chosen_dedup = rule_dedup
        else:
            chosen_dedup = "default"

        model_meta = registry.get(chosen_model, {}) if isinstance(registry, dict) else {}
        default_name = AUTO_PARETO_MODEL_NAME if chosen_model == AUTO_PARETO_MODEL_ID else chosen_model
        return {
            "model_id": chosen_model,
            "model_name": str(model_meta.get("name") or default_name),
            "dedup_policy": chosen_dedup,
            "dedup_source": dedup_source,
            "matched_pattern": matched_rule_pattern or "",
            "matched_model_pattern": matched_model_pattern or "",
            "matched_dedup_pattern": matched_dedup_pattern or "",
            "resolution_source": source_desc,
        }

    def list_drives(self) -> List[Dict[str, Any]]:
        """Enumerate available local drives with free/total GiB metadata."""
        drives: List[Dict[str, Any]] = []
        if sys.platform == "win32":
            for letter in string.ascii_uppercase:
                drive_root = f"{letter}:\\"
                if os.path.exists(drive_root):
                    free_gb = 0.0
                    total_gb = 0.0
                    try:
                        usage = shutil.disk_usage(drive_root)
                        free_gb = round(usage.free / (1024 ** 3), 2)
                        total_gb = round(usage.total / (1024 ** 3), 2)
                    except OSError:
                        pass
                    is_workspace_drive = (
                        str(self.workspace_root).upper().startswith(f"{letter}:")
                    )
                    drives.append({
                        "name": f"Ổ đĩa {letter}:",
                        "path": drive_root,
                        "type": "drive",
                        "free_gb": free_gb,
                        "total_gb": total_gb,
                        "is_workspace": is_workspace_drive,
                    })
        else:
            usage = shutil.disk_usage("/")
            drives.append({
                "name": "/",
                "path": "/",
                "type": "drive",
                "free_gb": round(usage.free / (1024 ** 3), 2),
                "total_gb": round(usage.total / (1024 ** 3), 2),
                "is_workspace": True,
            })
        return drives

    def browse(self, raw_path: Optional[str] = None, include_subtitles: bool = False) -> Dict[str, Any]:
        """List directory contents or drive roots while blocking NUL bytes and secret files."""
        if not raw_path or raw_path.strip().lower() == "drives":
            return {
                "current_path": "",
                "parent_path": None,
                "workspace_root": str(self.workspace_root),
                "entries": self.list_drives(),
                "source_video_count": 0,
            }

        cleaned = strip_wrapping_quotes(raw_path)
        if not cleaned or cleaned.lower() == "drives":
            return {
                "current_path": "",
                "parent_path": None,
                "workspace_root": str(self.workspace_root),
                "entries": self.list_drives(),
                "source_video_count": 0,
            }
        if "\x00" in cleaned:
            raise ValueError("Đường dẫn chứa ký tự NUL (\\x00) không hợp lệ.")
        if is_unc_path(cleaned):
            raise ValueError("Từ chối truy cập đường dẫn mạng UNC (\\\\... hoặc //...).")

        cand_raw = unwrap_outer_quotes_preserving_trailing_space(raw_path)
        cand = Path(os.path.expanduser(cleaned))
        if cand.drive and not cand.is_absolute():
            raise ValueError("Đường dẫn tương đối theo ổ đĩa (vd: E:folder) không hợp lệ.")
        if sys.platform == "win32" and not cand.drive and cand.root:
            raise ValueError("Đường dẫn gốc thiếu ký tự ổ đĩa không hợp lệ.")
        validate_win32_path_hygiene(cand, field="path", raw_str=cand_raw)
        try:
            if not cand.is_absolute():
                cand = (self.workspace_root / cand).resolve()
            else:
                cand = cand.resolve()
        except OSError as exc:
            raise ValueError(f"Đường dẫn không hợp lệ: {exc}") from exc

        if is_secret_filename(cand.name):
            raise PermissionError("Từ chối truy cập tệp cấu hình nhạy cảm.")

        if not cand.exists():
            raise FileNotFoundError(f"Không tìm thấy đường dẫn: {cand}")

        if cand.is_file():
            cand = cand.parent

        parent_path: Optional[str]
        if cand.parent == cand:
            parent_path = "drives"
        else:
            parent_path = str(cand.parent)

        dirs: List[Dict[str, Any]] = []
        files: List[Dict[str, Any]] = []
        source_video_count = 0
        out_dir = self.workspace_root / "out"
        routing_tables = self._load_routing_tables()

        try:
            dir_iter = list(os.scandir(cand))
        except PermissionError as exc:
            raise PermissionError(f"Không có quyền đọc thư mục: {cand}") from exc

        for entry in sorted(dir_iter, key=lambda e: natural_sort_key(e.name)):
            name = entry.name
            if is_secret_filename(name):
                continue
            if name.startswith(".") or name in {"__pycache__", ".venv", "venv", "node_modules"}:
                continue
            if name.lower() in _WINDOWS_PROTECTED_DIR_NAMES:
                continue
            if sys.platform == "win32":
                try:
                    attrs = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
                    if attrs & (stat.FILE_ATTRIBUTE_SYSTEM | stat.FILE_ATTRIBUTE_HIDDEN):
                        continue
                except OSError:
                    continue

            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                continue

            if is_dir:
                dirs.append({
                    "name": name,
                    "path": entry.path,
                    "type": "dir",
                })
            elif is_file:
                ext = os.path.splitext(name)[1].lower()
                stem = os.path.splitext(name)[0]
                if ext in VIDEO_EXTENSIONS:
                    is_4k_out = is_upscaled_output_stem(stem)
                    try:
                        st = entry.stat()
                        size_mb = round(st.st_size / (1024 * 1024), 2)
                        mtime = int(st.st_mtime)
                    except OSError:
                        size_mb = 0.0
                        mtime = 0

                    routing = self.resolve_routing_for_path(entry.path, routing_tables=routing_tables)
                    already_done = False
                    if not is_4k_out and out_dir.is_dir():
                        for suffix in ("_4K.mkv", "_4K_animejanai.mkv"):
                            out_cand = out_dir / f"{stem}{suffix}"
                            if out_cand.is_file() and out_cand.stat().st_size > 1024 * 1024:
                                already_done = True
                                break
                    if not is_4k_out:
                        source_video_count += 1
                    files.append({
                        "name": name,
                        "path": entry.path,
                        "type": "video_4k" if is_4k_out else "video",
                        "size_mb": size_mb,
                        "mtime": mtime,
                        "already_upscaled": already_done,
                        "routing": routing,
                    })
                elif include_subtitles and ext in SUBTITLE_EXTENSIONS:
                    try:
                        size_mb = round(entry.stat().st_size / (1024 * 1024), 3)
                    except OSError:
                        size_mb = 0.0
                    files.append({
                        "name": name,
                        "path": entry.path,
                        "type": "subtitle",
                        "size_mb": size_mb,
                    })

        source_files = [
            f for f in files if f.get("type") == "video" and isinstance(f.get("routing"), dict)
        ]
        if source_files:
            pair_counts: Dict[tuple[str, str], int] = {}
            pair_sample: Dict[tuple[str, str], Dict[str, Any]] = {}
            distinct_models: set[str] = set()
            for sf in source_files:
                rt = sf["routing"]
                key = (str(rt.get("model_id") or ""), str(rt.get("dedup_policy") or "default"))
                pair_counts[key] = pair_counts.get(key, 0) + 1
                pair_sample.setdefault(key, rt)
                if key[0]:
                    distinct_models.add(key[0])
            dominant_key = max(pair_counts.items(), key=lambda kv: kv[1])[0]
            folder_routing = dict(pair_sample[dominant_key])
            if len(pair_counts) > 1:
                folder_routing["mixed_routing"] = True
                folder_routing["distinct_models_count"] = len(distinct_models)
                folder_routing["resolution_source"] = (
                    f"Hỗn hợp {len(pair_counts)} cấu hình theo từng tệp (phổ biến: {dominant_key[0]})"
                )
        else:
            folder_routing = self.resolve_routing_for_path(cand / "sample.mkv", routing_tables=routing_tables)

        return {
            "current_path": str(cand),
            "parent_path": parent_path,
            "workspace_root": str(self.workspace_root),
            "source_video_count": source_video_count,
            "folder_routing": folder_routing,
            "entries": dirs + files,
        }

    def expand_inputs(
        self,
        raw_inputs: Sequence[str],
        allow_subtitles: bool = True,
        allow_upscaled_4k: bool = True,
    ) -> List[str]:
        """Expand file/directory inputs in natural order, skipping hidden and *_4K outputs."""
        expanded: List[str] = []
        seen: set[str] = set()

        def _add(p_str: str) -> None:
            abs_p = os.path.abspath(os.path.expanduser(p_str))
            norm = os.path.normcase(abs_p)
            if norm not in seen:
                seen.add(norm)
                expanded.append(abs_p)

        for raw in raw_inputs:
            if not raw or not str(raw).strip():
                continue
            raw_unwrapped = unwrap_outer_quotes_preserving_trailing_space(str(raw))
            item = strip_wrapping_quotes(str(raw))
            if not item:
                continue
            if "\x00" in item:
                raise ValueError("Đường dẫn chứa ký tự không hợp lệ.")
            if is_unc_path(item):
                raise ValueError("Từ chối truy cập đường dẫn mạng UNC (\\\\... hoặc //...).")
            cand = Path(os.path.expanduser(item))
            if cand.drive and not cand.is_absolute():
                raise ValueError("Đường dẫn tương đối theo ổ đĩa (vd: E:folder) không hợp lệ.")
            if sys.platform == "win32" and not cand.drive and cand.root:
                raise ValueError("Đường dẫn gốc thiếu ký tự ổ đĩa không hợp lệ.")
            validate_win32_path_hygiene(cand, field="inputs", raw_str=raw_unwrapped)
            try:
                if not cand.is_absolute():
                    cand = (self.workspace_root / cand).resolve()
                else:
                    cand = cand.resolve()
            except OSError as exc:
                raise ValueError(f"Đường dẫn đầu vào không hợp lệ: {exc}") from exc

            if is_secret_filename(cand.name):
                raise PermissionError("Từ chối truy cập tệp nhạy cảm.")

            if cand.is_dir():
                found: List[str] = []
                found_4k: List[str] = []
                found_subs_raw: List[str] = []
                for entry_name in sorted(os.listdir(cand), key=natural_sort_key):
                    if entry_name.startswith("."):
                        continue
                    full_p = cand / entry_name
                    if not full_p.is_file():
                        continue
                    stem, ext = os.path.splitext(entry_name)
                    ext_low = ext.lower()
                    if ext_low in VIDEO_EXTENSIONS:
                        if is_upscaled_output_stem(stem):
                            if allow_upscaled_4k:
                                found_4k.append(str(full_p))
                        else:
                            found.append(str(full_p))
                    elif allow_subtitles and ext_low in SUBTITLE_EXTENSIONS:
                        low_name = entry_name.lower()
                        if low_name.endswith(_SUBTITLE_ARTIFACT_SUFFIXES):
                            continue
                        found_subs_raw.append(str(full_p))
                found_subs = filter_paired_subtitle_paths(found_subs_raw)
                chosen = found or found_4k or found_subs
                if not chosen:
                    sub_hint = " hoặc phụ đề .srt/.ass" if allow_subtitles else ""
                    raise ValueError(
                        f"Thư mục '{cand}' không chứa file video gốc{sub_hint} hợp lệ nào (hoặc tất cả đã là bản _4K)."
                    )
                for fp in chosen:
                    _add(fp)
            elif cand.is_file():
                ext = cand.suffix.lower()
                allowed_exts = (VIDEO_EXTENSIONS | SUBTITLE_EXTENSIONS) if allow_subtitles else VIDEO_EXTENSIONS
                if ext not in allowed_exts:
                    if ext in SUBTITLE_EXTENSIONS and not allow_subtitles:
                        raise ValueError(
                            f"Tệp phụ đề '{cand.name}' không phải là video nguồn để nâng cấp 4K (hãy dùng Tab Subtrans)."
                        )
                    raise ValueError(f"Định dạng file không được hỗ trợ: {cand.name}")
                if ext in SUBTITLE_EXTENSIONS and cand.name.lower().endswith(_SUBTITLE_ARTIFACT_SUFFIXES):
                    raise ValueError(
                        f"Tệp '{cand.name}' là phụ đề thành phẩm/tạm (.vi/.partial/.review) — không nạp làm đầu vào."
                    )
                if (not allow_upscaled_4k) and (ext in VIDEO_EXTENSIONS) and is_upscaled_output_stem(cand.stem):
                    raise ValueError(
                        f"Tệp '{cand.name}' đã là bản thành phẩm 4K (_4K) — từ chối nạp làm nguồn đầu vào để tránh tràn VRAM 8K."
                    )
                _add(str(cand))
            else:
                raise FileNotFoundError(f"Không tìm thấy đường dẫn: {cand}")

        return expanded
