"""Safe local filesystem browser and show-routing resolver for Web Studio."""

from __future__ import annotations

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


def is_secret_filename(name: str) -> bool:
    """Block access to secret .env files over HTTP/LAN while permitting templates."""
    low = name.strip().lower()
    if low in {".env.example", ".env.sample", ".env.template"}:
        return False
    return low == ".env" or low.startswith(".env.")


def is_unc_path(raw: Optional[str]) -> bool:
    """Return True if raw string targets a Windows/POSIX UNC network share (\\\\... or //...)." """
    if not raw:
        return False
    s = str(raw).strip()
    return s.startswith("\\\\") or s.startswith("//")


def is_upscaled_output_stem(stem: str) -> bool:
    """Return True if filename stem already denotes a rendered 4K output."""
    low = stem.lower()
    return low.endswith("_4k") or low.endswith("_4k_animejanai")


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

        cleaned = raw_path.strip()
        if "\x00" in cleaned:
            raise ValueError("Đường dẫn chứa ký tự NUL (\\x00) không hợp lệ.")
        if is_unc_path(cleaned):
            raise ValueError("Từ chối truy cập đường dẫn mạng UNC (\\\\... hoặc //...).")

        cand = Path(os.path.expanduser(cleaned))
        if not cand.is_absolute():
            cand = (self.workspace_root / cand).resolve()
        else:
            cand = cand.resolve()

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
            item = str(raw).strip()
            if "\x00" in item:
                raise ValueError("Đường dẫn chứa ký tự không hợp lệ.")
            if is_unc_path(item):
                raise ValueError("Từ chối truy cập đường dẫn mạng UNC (\\\\... hoặc //...).")
            cand = Path(os.path.expanduser(item))
            if not cand.is_absolute():
                cand = (self.workspace_root / cand).resolve()
            else:
                cand = cand.resolve()

            if is_secret_filename(cand.name):
                raise PermissionError("Từ chối truy cập tệp nhạy cảm.")

            if cand.is_dir():
                found: List[str] = []
                for entry_name in sorted(os.listdir(cand), key=natural_sort_key):
                    if entry_name.startswith("."):
                        continue
                    full_p = cand / entry_name
                    if not full_p.is_file():
                        continue
                    stem, ext = os.path.splitext(entry_name)
                    if ext.lower() not in VIDEO_EXTENSIONS:
                        continue
                    if is_upscaled_output_stem(stem):
                        continue
                    found.append(str(full_p))
                if not found:
                    raise ValueError(
                        f"Thư mục '{cand}' không chứa file video gốc hợp lệ nào (hoặc tất cả đã là bản _4K)."
                    )
                for fp in found:
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
                if (not allow_upscaled_4k) and (ext in VIDEO_EXTENSIONS) and is_upscaled_output_stem(cand.stem):
                    raise ValueError(
                        f"Tệp '{cand.name}' đã là bản thành phẩm 4K (_4K) — từ chối nạp làm nguồn đầu vào để tránh tràn VRAM 8K."
                    )
                _add(str(cand))
            else:
                raise FileNotFoundError(f"Không tìm thấy đường dẫn: {cand}")

        return expanded
