"""Configuration loader, model metadata, and 3-key rotation settings for subtrans."""

import os
import threading
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    from src.env_isolation import configure_e_drive_isolation
    configure_e_drive_isolation(project_root=_REPO_ROOT)
except Exception:
    pass

# Load .env explicitly from subtrans/ or repo root unless disabled in isolated offline tests
if (
    os.environ.get("PYTHON_DOTENV_DISABLED") != "1"
    and os.environ.get("SUBTRANS_DISABLE_DOTENV") != "1"
):
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CONFIG_PATH = Path(__file__).parent / "config.yaml"
MODELS_CONFIG_PATH = Path(__file__).parent / "models.yaml"
_DEFAULT_RUNTIME_QUOTA_PATH = Path(__file__).parent / ".cache" / "models_runtime_quota.yaml"
RUNTIME_QUOTA_PATH = _DEFAULT_RUNTIME_QUOTA_PATH


def _atomic_write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Writes YAML atomically via a same-directory temp file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


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


def _parse_positive_int(value: Any) -> Optional[int]:
    """Parses value as an int only if it is strictly > 0; returns None otherwise."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def load_models_config(custom_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads models and rate limits configuration from models.yaml, overlaying untracked runtime quota metadata."""
    path = custom_path or MODELS_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if custom_path is None and not os.environ.get("PYTEST_CURRENT_TEST") and RUNTIME_QUOTA_PATH.exists():
        try:
            with open(RUNTIME_QUOTA_PATH, "r", encoding="utf-8") as rf:
                rt_cfg = yaml.safe_load(rf) or {}
            rt_models = rt_cfg.get("models")
            if isinstance(rt_models, dict) and isinstance(cfg.get("models"), dict):
                for m_name, m_meta in rt_models.items():
                    if not isinstance(m_meta, dict) or m_name not in cfg["models"]:
                        continue
                    target_meta = cfg["models"][m_name]
                    if not isinstance(target_meta, dict):
                        continue
                    rpd_val = _parse_positive_int(m_meta.get("rpd"))
                    if rpd_val is not None:
                        target_meta["rpd"] = rpd_val
                        if m_meta.get("rpd_source"):
                            target_meta["rpd_source"] = m_meta["rpd_source"]
                    rpm_val = _parse_positive_int(m_meta.get("rpm"))
                    if rpm_val is not None:
                        target_meta["rpm"] = rpm_val
                        if m_meta.get("rpm_source"):
                            target_meta["rpm_source"] = m_meta["rpm_source"]
        except Exception:
            pass

    return cfg


def save_models_config(cfg: Dict[str, Any], custom_path: Optional[Path] = None):
    """Saves models configuration back to YAML atomically."""
    path = custom_path or MODELS_CONFIG_PATH
    _atomic_write_yaml(path, cfg)


def update_model_quota_in_yaml(
    model_name: str,
    quota_id: str,
    quota_value: Any,
    rpd_source: str = "đọc từ phản hồi 429",
    custom_path: Optional[Path] = None,
):
    """
    Cập nhật hạn mức quotaValue quan sát được khi gặp lỗi 429 thực tế:
    - Khi custom_path được truyền tường minh: ghi nguyên tử vào custom_path.
    - Khi custom_path is None: ghi vào file untracked subtrans/.cache/models_runtime_quota.yaml
      (không làm bẩn git status của subtrans/models.yaml).
    - Bỏ qua giá trị <= 0 hoặc rỗng để không phá cấu hình hạn mức gốc.
    """
    parsed_val = _parse_positive_int(quota_value)
    if parsed_val is None:
        return
    if (
        custom_path is None
        and RUNTIME_QUOTA_PATH == _DEFAULT_RUNTIME_QUOTA_PATH
        and (os.environ.get("PYTEST_CURRENT_TEST") or model_name == "unit-test-model")
    ):
        return
    try:
        target_path = custom_path if custom_path is not None else RUNTIME_QUOTA_PATH
        if custom_path is not None:
            cfg = load_models_config(custom_path)
        elif target_path.exists():
            with open(target_path, "r", encoding="utf-8") as rf:
                cfg = yaml.safe_load(rf) or {}
        else:
            cfg = {}

        if "models" not in cfg or not isinstance(cfg["models"], dict):
            cfg["models"] = {}
        if model_name not in cfg["models"] or not isinstance(cfg["models"][model_name], dict):
            cfg["models"][model_name] = {}

        src_label = rpd_source if (not quota_id or quota_id in rpd_source) else f"{rpd_source} ({quota_id})"
        qid_lower = (quota_id or "").lower()
        if "perday" in qid_lower or "requestsperday" in qid_lower or "daily" in qid_lower:
            cfg["models"][model_name]["rpd"] = parsed_val
            cfg["models"][model_name]["rpd_source"] = src_label
        elif "perminute" in qid_lower or "requestsperminute" in qid_lower or "rpm" in qid_lower:
            cfg["models"][model_name]["rpm"] = parsed_val
            cfg["models"][model_name]["rpm_source"] = src_label

        _atomic_write_yaml(target_path, cfg)
    except Exception as e:
        import logging
        logging.getLogger("subtrans.config").warning(f"Could not update runtime quota metadata for {model_name}: {e}")


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

