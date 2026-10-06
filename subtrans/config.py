# -*- coding: utf-8 -*-
"""
subtrans/config.py
Configuration loader and validator for subtrans.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load .env from workspace root or subtrans/
load_dotenv()
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

CONFIG_PATH = Path(__file__).parent / "config.yaml"
MODELS_CONFIG_PATH = Path(__file__).parent / "models.yaml"


def load_config(custom_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads configuration from yaml file."""
    path = custom_path or CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: Dict[str, Any], custom_path: Optional[Path] = None):
    """Saves configuration back to yaml file."""
    path = custom_path or CONFIG_PATH
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def load_models_config(custom_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads models and rate limits configuration from models.yaml."""
    path = custom_path or MODELS_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_models_config(cfg: Dict[str, Any], custom_path: Optional[Path] = None):
    """Saves models configuration back to models.yaml."""
    path = custom_path or MODELS_CONFIG_PATH
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def update_model_quota_in_yaml(model_name: str, quota_id: str, quota_value: Any, rpd_source: str = "đọc từ phản hồi 429"):
    """
    Cập nhật tự động hạn mức quotaValue vào models.yaml khi nhận được lỗi 429 thực tế.
    """
    try:
        cfg = load_models_config()
        if "models" not in cfg:
            cfg["models"] = {}
        if model_name not in cfg["models"]:
            cfg["models"][model_name] = {}

        if "perday" in quota_id.lower() or "requestsperday" in quota_id.lower() or "daily" in quota_id.lower():
            try:
                cfg["models"][model_name]["rpd"] = int(quota_value)
            except Exception:
                cfg["models"][model_name]["rpd"] = quota_value
            cfg["models"][model_name]["rpd_source"] = f"{rpd_source} ({quota_id})"
        elif "perminute" in quota_id.lower() or "requestsperminute" in quota_id.lower() or "rpm" in quota_id.lower():
            try:
                cfg["models"][model_name]["rpm"] = int(quota_value)
            except Exception:
                cfg["models"][model_name]["rpm"] = quota_value
            cfg["models"][model_name]["rpm_source"] = f"{rpd_source} ({quota_id})"

        save_models_config(cfg)
    except Exception as e:
        import logging
        logging.getLogger("subtrans.config").warning(f"Could not update quota in models.yaml for {model_name}: {e}")


def get_gemini_api_keys() -> list[str]:
    """
    Thu thập danh sách tất cả các GEMINI_API_KEY từ môi trường và .env:
    - Hỗ trợ GEMINI_API_KEYS (phân tách dấu phẩy hoặc khoảng trắng)
    - Hỗ trợ GEMINI_API_KEY (phân tách dấu phẩy hoặc đơn lẻ)
    - Hỗ trợ các biến GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3,...
    Trả về danh sách các key không trùng lặp và không rỗng.
    """
    keys = []

    # 1. Từ GEMINI_API_KEYS
    raw_keys = os.environ.get("GEMINI_API_KEYS", "")
    if raw_keys:
        for k in raw_keys.replace(";", ",").split(","):
            k_clean = k.strip()
            if k_clean and k_clean not in keys:
                keys.append(k_clean)

    # 2. Từ GEMINI_API_KEY (hỗ trợ cả dấu phẩy)
    raw_key = os.environ.get("GEMINI_API_KEY", "")
    if raw_key:
        for k in raw_key.replace(";", ",").split(","):
            k_clean = k.strip()
            if k_clean and k_clean not in keys:
                keys.append(k_clean)

    # 3. Quét các biến GEMINI_API_KEY_*
    for env_k, env_v in sorted(os.environ.items()):
        if env_k.startswith("GEMINI_API_KEY_") and env_v:
            for k in env_v.replace(";", ",").split(","):
                k_clean = k.strip()
                if k_clean and k_clean not in keys:
                    keys.append(k_clean)

    return keys


def get_gemini_api_key() -> Optional[str]:
    """Retrieves primary GEMINI_API_KEY from environment."""
    keys = get_gemini_api_keys()
    return keys[0] if keys else None

