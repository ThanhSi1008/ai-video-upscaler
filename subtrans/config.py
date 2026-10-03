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


def get_gemini_api_key() -> Optional[str]:
    """Retrieves GEMINI_API_KEY from environment."""
    return os.environ.get("GEMINI_API_KEY")
