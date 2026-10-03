# -*- coding: utf-8 -*-
"""
src/models_resolver.py
Hệ thống quản lý và phân giải mô hình Super-Resolution cho từng bộ anime:
Thứ tự ưu tiên (D2):
1. Tham số CLI: --model <id>
2. Cấu hình shows.yaml (khớp mẫu regex/glob tên file hoặc thư mục do người dùng khai báo)
3. Mặc định trong models/models.yaml (animevideov3)
"""

import os
import yaml
import fnmatch
from typing import Dict, Any, Optional, Tuple

MODELS_YAML = "models/models.yaml"
SHOWS_YAML = "shows.yaml"


def load_models_registry() -> Dict[str, Any]:
    """Nạp danh mục mô hình từ models/models.yaml."""
    if not os.path.exists(MODELS_YAML):
        raise FileNotFoundError(f"Không tìm thấy file danh mục mô hình: {MODELS_YAML}")
    with open(MODELS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


def get_default_model_id() -> str:
    """Trả về ID của mô hình mặc định trong models.yaml."""
    registry = load_models_registry()
    for m_id, m_info in registry.items():
        if m_info.get("default", False):
            return m_id
    return "animevideov3"


def list_models_formatted() -> str:
    """Định dạng danh sách các mô hình khả dụng để in ra CLI (--list-models)."""
    registry = load_models_registry()
    lines = [
        "=" * 80,
        "📋 DANH MỤC MÔ HÌNH SUPER-RESOLUTION ĐĂNG KÝ (models/models.yaml)",
        "=" * 80,
        f"{'ID':<25} | {'Tên hiển thị':<30} | {'Kiến trúc':<12} | {'Mặc định'}",
        "-" * 80
    ]
    for m_id, m in registry.items():
        is_def = "⭐ [Mặc định]" if m.get("default", False) else ""
        arch_desc = f"{m['arch']['num_feat']}f{m['arch']['num_conv']}c (x{m['arch']['upscale']})"
        lines.append(f"{m_id:<25} | {m['name']:<30} | {arch_desc:<12} | {is_def}")
    lines.append("=" * 80)
    lines.append("Ghi chú: Để ánh xạ tự động cho từng bộ phim, cấu hình mẫu trong shows.yaml.")
    return "\n".join(lines)


def resolve_model_for_video(video_path: str, cli_model_arg: Optional[str] = None) -> Dict[str, Any]:
    """
    Phân giải mô hình cho video theo thứ tự ưu tiên:
    CLI --model > shows.yaml > Default (animevideov3)
    """
    registry = load_models_registry()
    default_id = get_default_model_id()

    chosen_id = None
    resolution_source = None

    # 1. Ưu tiên cao nhất: CLI arg
    # Chấp nhận cả alias cũ 'realesr-animevideov3' -> 'animevideov3'
    if cli_model_arg:
        arg_clean = cli_model_arg.strip().lower()
        if arg_clean == "realesr-animevideov3":
            arg_clean = "animevideov3"

        if arg_clean in registry:
            chosen_id = arg_clean
            resolution_source = f"Tham số dòng lệnh (--model {cli_model_arg})"
        else:
            raise ValueError(
                f"Mô hình '{cli_model_arg}' không tồn tại trong models/models.yaml.\n"
                f"Danh sách mô hình hợp lệ: {list(registry.keys())}"
            )

    # 2. Ưu tiên thứ hai: shows.yaml (nếu chưa chỉ định qua CLI)
    if not chosen_id and os.path.exists(SHOWS_YAML):
        try:
            with open(SHOWS_YAML, "r", encoding="utf-8") as f:
                shows_data = yaml.safe_load(f) or {}
            shows_rules = shows_data.get("shows", []) or []

            vid_name = os.path.basename(video_path)
            vid_dir = os.path.basename(os.path.dirname(os.path.abspath(video_path)))

            for rule in shows_rules:
                pat = rule.get("pattern", "")
                m_target = rule.get("model", "")
                # Khớp với tên file hoặc tên thư mục chứa
                if fnmatch.fnmatch(vid_name, pat) or fnmatch.fnmatch(vid_dir, pat) or (pat and pat.lower() in vid_name.lower()):
                    if m_target in registry:
                        chosen_id = m_target
                        resolution_source = f"shows.yaml (Khớp mẫu '{pat}')"
                        break
        except Exception:
            pass

    # 3. Mặc định
    if not chosen_id:
        chosen_id = default_id
        resolution_source = "Mặc định hệ thống (models/models.yaml)"

    model_info = dict(registry[chosen_id])
    model_info["resolution_source"] = resolution_source
    return model_info
