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


def get_gemini_api_key() -> Optional[str]:
    """Retrieves GEMINI_API_KEY from environment."""
    return os.environ.get("GEMINI_API_KEY")

