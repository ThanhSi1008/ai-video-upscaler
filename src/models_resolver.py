"""Super-resolution model registry loader, shows.yaml matcher, and SHA-256 verifier."""

import copy
import hashlib
import json
import os
import re
import sys
import yaml
from typing import Dict, Any, Optional

from src.config import ROOT_DIR, DEFAULT_OUTPUT_DIR
from src.show_routing import match_show_pattern, match_show_rule

MODELS_YAML = "models/models.yaml"
SHOWS_YAML = "shows.yaml"
_YAML_STAT_CACHE: dict[tuple[str, int, int], Any] = {}


def _load_yaml_cached(path_str: str) -> Any:
    """Load YAML file with stat-tuple memoization when yaml.safe_load/open are unmocked."""
    if not hasattr(yaml.safe_load, "_mock_name") and not hasattr(open, "_mock_name"):
        try:
            abs_p = os.path.abspath(path_str)
            st = os.stat(abs_p)
            key = (abs_p, st.st_mtime_ns, st.st_size)
            if key in _YAML_STAT_CACHE:
                return copy.deepcopy(_YAML_STAT_CACHE[key])
            with open(abs_p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if len(_YAML_STAT_CACHE) >= 64:
                _YAML_STAT_CACHE.clear()
            _YAML_STAT_CACHE[key] = copy.deepcopy(data)
            return data
        except OSError:
            pass
    with open(path_str, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_repo_path(path_str: str) -> str:
    """Ưu tiên đường dẫn hiện tại (phục vụ test override), fallback về ROOT_DIR khi chạy khác thư mục."""
    if os.path.exists(path_str) or os.path.isabs(path_str):
        return path_str
    candidate = os.path.join(ROOT_DIR, path_str)
    return candidate if os.path.exists(candidate) else path_str


def verify_weights_sha256(weights_path: str, expected_sha256: Optional[str]) -> bool:
    """Kiểm tra toàn vẹn SHA-256 của file trọng số mô hình."""
    if not expected_sha256:
        return True
    h = hashlib.sha256()
    with open(weights_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower() == expected_sha256.strip().lower()


def load_models_registry() -> Dict[str, Any]:
    """Nạp danh mục mô hình từ models/models.yaml."""
    models_yaml_path = _resolve_repo_path(MODELS_YAML)
    if not os.path.exists(models_yaml_path):
        raise FileNotFoundError(f"Không tìm thấy file danh mục mô hình: {MODELS_YAML}")
    data = _load_yaml_cached(models_yaml_path) or {}
    return data.get("models", {})


def get_default_model_id(registry: Optional[Dict[str, Any]] = None) -> str:
    """Trả về ID của mô hình mặc định trong models.yaml."""
    reg = registry if registry is not None else load_models_registry()
    for m_id, m_info in reg.items():
        if m_info.get("default", False):
            return m_id
    return "animevideov3"


def list_models_formatted() -> str:
    """Định dạng danh sách các mô hình khả dụng để in ra CLI (--list-models)."""
    registry = load_models_registry()
    lines = [
        "=" * 96,
        "📋 DANH MỤC MÔ HÌNH SUPER-RESOLUTION ĐĂNG KÝ (models/models.yaml)",
        "=" * 96,
        f"{'ID':<40} | {'Tên hiển thị':<34} | {'Kiến trúc':<12} | {'Mặc định'}",
        "-" * 96
    ]
    for m_id, m in registry.items():
        is_def = "⭐ [Mặc định]" if m.get("default", False) else ""
        arch_desc = f"{m['arch']['num_feat']}f{m['arch']['num_conv']}c (x{m['arch']['upscale']})"
        lines.append(f"{m_id:<40} | {m['name']:<34} | {arch_desc:<12} | {is_def}")
    lines.append("=" * 96)
    lines.append("Ghi chú: Để ánh xạ tự động cho từng bộ phim, cấu hình mẫu trong shows.yaml.")
    return "\n".join(lines)


def resolve_model_for_video(video_path: str, cli_model_arg: Optional[str] = None) -> Dict[str, Any]:
    """
    Phân giải mô hình cho video theo thứ tự ưu tiên:
    CLI --model > shows.yaml > Default (animevideov3)
    """
    registry = load_models_registry()
    default_id = get_default_model_id(registry)
    lower_to_key = {str(k).lower(): k for k in registry.keys()}

    chosen_id = None
    resolution_source = None

    # 1. Ưu tiên cao nhất: CLI arg (hỗ trợ khớp không phân biệt hoa thường cho ID như 2x_AnimeJaNai_V3)
    # Chấp nhận cả alias cũ 'realesr-animevideov3' -> 'animevideov3'
    if cli_model_arg:
        arg_clean = cli_model_arg.strip().lower()
        if arg_clean == "realesr-animevideov3":
            arg_clean = "animevideov3"

        if arg_clean in lower_to_key:
            chosen_id = lower_to_key[arg_clean]
            resolution_source = f"Tham số dòng lệnh (--model {cli_model_arg})"
        else:
            raise ValueError(
                f"Mô hình '{cli_model_arg}' không tồn tại trong models/models.yaml.\n"
                f"Danh sách mô hình hợp lệ: {list(registry.keys())}"
            )

    # 2. Ưu tiên thứ hai: shows.yaml (nếu chưa chỉ định qua CLI)
    shows_yaml_path = _resolve_repo_path(SHOWS_YAML)
    if not chosen_id and os.path.exists(shows_yaml_path):
        try:
            shows_data = _load_yaml_cached(shows_yaml_path) or {}
            shows_rules = shows_data.get("shows", []) or []

            for rule in shows_rules:
                pat = rule.get("pattern", "")
                m_target = str(rule.get("model", "")).strip().lower()
                if match_show_rule(video_path, pat):
                    if m_target in lower_to_key:
                        chosen_id = lower_to_key[m_target]
                        resolution_source = f"shows.yaml (Khớp mẫu '{pat}')"
                        break
        except Exception as exc:
            print(
                f"⚠️ Cảnh báo: Không thể đọc cấu hình '{shows_yaml_path}' ({exc}); dùng mô hình mặc định.",
                file=sys.stderr,
            )

    # 3. Mặc định
    if not chosen_id:
        chosen_id = default_id
        resolution_source = "Mặc định hệ thống (models/models.yaml)"

    model_info = dict(registry[chosen_id])
    if model_info.get("weights_path"):
        model_info["weights_path"] = _resolve_repo_path(model_info["weights_path"])
    model_info["resolution_source"] = resolution_source
    return model_info


def _read_sibling_audit_status(output_file_path: str) -> Optional[str]:
    """Đọc trạng thái kiểm định ('PASS' / 'FAIL') từ file .audit.json cùng thư mục hoặc audit/."""
    stem = os.path.splitext(output_file_path)[0]
    direct_audit = f"{stem}.audit.json"
    subdir_audit = os.path.join(
        os.path.dirname(os.path.abspath(output_file_path)),
        "audit",
        os.path.basename(direct_audit),
    )
    for audit_cand in (direct_audit, subdir_audit):
        if os.path.isfile(audit_cand):
            try:
                with open(audit_cand, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("status"):
                    return str(data["status"]).strip().upper()
            except Exception:
                return None
    return None


def is_valid_completed_4k_output(candidate_path: str, min_size_bytes: int = 1024 * 1024) -> bool:
    """
    Xác minh file 4K thành phẩm tồn tại, dung lượng vượt ngưỡng (> 1 MiB)
    và không bị đánh dấu status == 'FAIL' trong file .audit.json đi kèm.
    """
    if not candidate_path or not os.path.isfile(candidate_path):
        return False
    try:
        if os.path.getsize(candidate_path) <= min_size_bytes:
            return False
    except OSError:
        return False
    status = _read_sibling_audit_status(candidate_path)
    if status and status.startswith("FAIL"):
        return False
    return True


def resolve_expected_4k_output_path(
    input_path: str,
    output_arg: Optional[str] = None,
    model_id: Optional[str] = None,
) -> str:
    """
    Tính toán đường dẫn file 4K đích kỳ vọng theo chuẩn hậu tố mô hình:
    - Nếu output_arg là đường dẫn file .mkv/.mp4 tường minh: giữ nguyên đường dẫn đó.
    - Nếu output_arg là thư mục (hoặc None -> DEFAULT_OUTPUT_DIR):
      ghép <base_name>_4K_animejanai.mkv (nếu model_id bắt đầu bằng 'animejanai-') hoặc <base_name>_4K.mkv.
    """
    if output_arg:
        expanded_out = os.path.abspath(os.path.expanduser(output_arg))
        if output_arg.strip().lower().endswith((".mkv", ".mp4")) or os.path.isfile(expanded_out):
            return expanded_out
        out_dir = expanded_out
    else:
        out_dir = DEFAULT_OUTPUT_DIR

    eff_model_id = model_id
    try:
        eff_model_id = resolve_model_for_video(input_path, model_id).get("id")
    except Exception:
        eff_model_id = model_id or ""

    out_suffix = (
        "_4K_animejanai.mkv"
        if str(eff_model_id or "").strip().lower().startswith("animejanai-")
        else "_4K.mkv"
    )
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(out_dir, f"{base_name}{out_suffix}")


def find_existing_4k_output_path(
    input_path: str,
    output_arg: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Optional[str]:
    """
    Tìm file 4K thành phẩm hợp lệ đã tồn tại (> 1 MiB, sibling .audit.json không FAIL),
    hỗ trợ cả hậu tố _4K_animejanai.mkv lẫn _4K.mkv và đường dẫn -o file tường minh.
    """
    expected_path = resolve_expected_4k_output_path(
        input_path, output_arg=output_arg, model_id=model_id
    )
    if output_arg:
        expanded_out = os.path.abspath(os.path.expanduser(output_arg))
        if output_arg.strip().lower().endswith((".mkv", ".mp4")) or os.path.isfile(expanded_out):
            return expected_path if is_valid_completed_4k_output(expected_path) else None

    out_dir = os.path.dirname(expected_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    raw_candidates = [
        expected_path,
        os.path.join(out_dir, f"{base_name}_4K_animejanai.mkv"),
        os.path.join(out_dir, f"{base_name}_4K.mkv"),
    ]
    seen: set[str] = set()
    for cand in raw_candidates:
        norm = os.path.normcase(os.path.abspath(cand))
        if norm in seen:
            continue
        seen.add(norm)
        if is_valid_completed_4k_output(cand):
            return cand
    return None
