# -*- coding: utf-8 -*-
"""
web/fs_browser.py
Trình duyệt ổ đĩa & thư mục nội bộ an toàn cho Web Studio trên Windows:
- Liệt kê các ổ đĩa hiện hữu (C:, D:, E:...) kèm dung lượng trống.
- Duyệt thư mục, lọc file video/phụ đề, sắp xếp tự nhiên (Tập 2 trước Tập 10).
- Tự động phân giải trước Model SR & Dedup Policy theo shows.yaml cho từng file/thư mục.
- Tuyệt đối ẩn và chặn truy cập các file nhạy cảm (.env, .env.*, khóa bí mật).
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import string
import sys
from typing import Any, Dict, List, Optional, Sequence
import yaml

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".webm", ".avi", ".mov", ".ts", ".m2ts"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}

# Bản sao thuần stdlib của DEDUP_POLICY_PARAMS (tránh nạp src.config -> torch._C._cuda_init() trong tiến trình Web)
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
    "dis-0.2": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "dis_thresh": 0.20},
    "dis-0.3": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "dis_thresh": 0.30},
    "dis-0.4": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "dis_thresh": 0.40},
    "dis-0.5": {"hi": _DEDUP_HI, "lo": _DEDUP_LO, "frac": _DEDUP_FRAC, "dis_thresh": 0.50},
    "dis-sakuga": {"hi": 1024, "lo": 448, "frac": 0.38, "dis_thresh": 0.35, "local_thresh": 1.50, "min_moving_area_pct": 12.0},
    "sakuga": {"hi": 1024, "lo": 448, "frac": 0.38, "dis_thresh": 0.35, "local_thresh": 1.50, "min_moving_area_pct": 12.0},
    "off": {"all_keep": True},
}


def match_show_pattern(text: str, pattern: str) -> bool:
    """Khớp mẫu tên file/thư mục anime từ shows.yaml không phân biệt hoa thường (an toàn với '[SubsPlease]')."""
    if not text or not pattern:
        return False
    t_low = str(text).lower()
    p_low = str(pattern).strip().lower()
    if not p_low:
        return False
    if "*" not in p_low and "?" not in p_low:
        return p_low in t_low
    regex_parts = ["^"]
    for ch in p_low:
        if ch == "*":
            regex_parts.append(".*")
        elif ch == "?":
            regex_parts.append(".")
        else:
            regex_parts.append(re.escape(ch))
    regex_parts.append("$")
    return re.match("".join(regex_parts), t_low, flags=re.DOTALL) is not None


def natural_sort_key(name: str) -> List[Any]:
    """Khóa sắp xếp tự nhiên theo số tập (Tập 2 đứng trước Tập 10)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(name))]


def is_secret_filename(name: str) -> bool:
    """Chặn tuyệt đối mọi file .env chứa khóa API (cho phép .env.example)."""
    low = name.strip().lower()
    if low in {".env.example", ".env.sample", ".env.template"}:
        return False
    return low == ".env" or low.startswith(".env.")


def is_upscaled_output_stem(stem: str) -> bool:
    """Kiểm tra tên file đã là bản thành phẩm 4K để tránh đưa ngược vào hàng đợi nguồn."""
    low = stem.lower()
    return low.endswith("_4k") or low.endswith("_4k_animejanai")


class SafeLocalFSBrowser:
    """Bộ duyệt file cục bộ phục vụ chọn video và thư mục anime trên giao diện Web."""

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
    ) -> Dict[str, str]:
        """Phân giải model_id, model_name, dedup_policy và nguồn khớp cho một đường dẫn video."""
        registry, default_id, shows_rules = self._load_routing_tables()
        lower_to_key = {str(k).lower(): str(k) for k in registry.keys()}

        vid_str = str(video_path)
        vid_name = os.path.basename(vid_str.rstrip("\\/"))
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(vid_str)))

        chosen_model = None
        matched_rule_pattern = None
        rule_dedup = None

        if override_model and override_model.strip():
            cand = override_model.strip().lower()
            if cand == "realesr-animevideov3":
                cand = "animevideov3"
            if cand in lower_to_key:
                chosen_model = lower_to_key[cand]
                source_desc = f"Chỉ định trực tiếp ({chosen_model})"
            else:
                chosen_model = override_model.strip()
                source_desc = f"Chỉ định trực tiếp ({chosen_model})"
        else:
            source_desc = "Mặc định hệ thống (models.yaml)"

        for rule in shows_rules:
            pat = str(rule.get("pattern", "")).strip()
            if not pat:
                continue
            if match_show_pattern(vid_name, pat) or match_show_pattern(parent_dir, pat):
                matched_rule_pattern = pat
                m_target = str(rule.get("model", "")).strip().lower()
                if not chosen_model and m_target in lower_to_key:
                    chosen_model = lower_to_key[m_target]
                    source_desc = f"shows.yaml ('{pat}')"
                pol_target = str(rule.get("dedup_policy", "")).strip().lower()
                if pol_target in DEDUP_POLICY_PARAMS:
                    rule_dedup = pol_target
                break

        if not chosen_model:
            chosen_model = default_id

        if override_dedup and override_dedup.strip() in DEDUP_POLICY_PARAMS:
            chosen_dedup = override_dedup.strip()
        elif rule_dedup:
            chosen_dedup = rule_dedup
        else:
            chosen_dedup = "default"

        model_meta = registry.get(chosen_model, {}) if isinstance(registry, dict) else {}
        return {
            "model_id": chosen_model,
            "model_name": str(model_meta.get("name") or chosen_model),
            "dedup_policy": chosen_dedup,
            "matched_pattern": matched_rule_pattern or "",
            "resolution_source": source_desc,
        }

    def list_drives(self) -> List[Dict[str, Any]]:
        """Liệt kê các ổ đĩa khả dụng trên Windows (hoặc gốc '/' trên POSIX)."""
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
        """Duyệt nội dung thư mục hoặc danh sách ổ đĩa."""
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
            raise ValueError("Đường dẫn không hợp lệ.")

        # Hỗ trợ đường dẫn tương đối tính từ workspace_root
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

                    routing = self.resolve_routing_for_path(entry.path)
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

        return {
            "current_path": str(cand),
            "parent_path": parent_path,
            "workspace_root": str(self.workspace_root),
            "source_video_count": source_video_count,
            "folder_routing": self.resolve_routing_for_path(cand / "sample.mkv"),
            "entries": dirs + files,
        }

    def expand_inputs(self, raw_inputs: Sequence[str]) -> List[str]:
        """
        Mở rộng danh sách file/thư mục đầu vào theo thứ tự tập tự nhiên,
        bỏ qua file ẩn và bản thành phẩm *_4K / *_4K_animejanai.
        """
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
                if ext not in (VIDEO_EXTENSIONS | SUBTITLE_EXTENSIONS):
                    raise ValueError(f"Định dạng file không được hỗ trợ: {cand.name}")
                _add(str(cand))
            else:
                raise FileNotFoundError(f"Không tìm thấy đường dẫn: {cand}")

        return expanded
