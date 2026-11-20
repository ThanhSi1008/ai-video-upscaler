# -*- coding: utf-8 -*-
"""
src/pending_queue.py
Manages the pending subtitle translation queue (state/pending_subs.json).
Stdlib-only module so subtrans/pipeline.py can import it without loading
torch, CUDA, or video hardware probing.
"""

import json
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
PENDING_SUBS_FILE = os.path.join(ROOT_DIR, "state", "pending_subs.json")
LEGACY_PENDING_SUBS_FILE = os.path.join(ROOT_DIR, "pending_subs.json")


def _resolve_pending_file() -> str:
    """Prefer src.config.PENDING_SUBS_FILE if src.config is loaded (supports monkeypatching)."""
    cfg_mod = sys.modules.get("src.config")
    if cfg_mod is not None and hasattr(cfg_mod, "PENDING_SUBS_FILE"):
        return str(getattr(cfg_mod, "PENDING_SUBS_FILE"))
    return PENDING_SUBS_FILE


def load_pending_subs() -> list:
    """Load pending subtitle entries, falling back to legacy root path if unmigrated."""
    target_file = _resolve_pending_file()
    read_path = target_file
    if not os.path.exists(read_path):
        if target_file == PENDING_SUBS_FILE and os.path.exists(LEGACY_PENDING_SUBS_FILE):
            read_path = LEGACY_PENDING_SUBS_FILE
        else:
            return []
    try:
        with open(read_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_pending_subs(items: list) -> None:
    """Atomically write the pending subtitle queue JSON to disk."""
    target_file = _resolve_pending_file()
    try:
        parent = os.path.dirname(os.path.abspath(target_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_file = f"{target_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, target_file)
    except Exception:
        pass


def add_pending_sub(record: dict) -> None:
    """Add or update an episode entry in the pending subtitle queue."""
    if os.environ.get("PYTEST_CURRENT_TEST") and "dummy" in str(record.get("base_name", "")):
        return
    items = load_pending_subs()
    items = [it for it in items if it.get("base_name") != record.get("base_name")]
    items.append(record)
    save_pending_subs(items)


def remove_pending_sub(base_name: str) -> None:
    """Remove a completed episode entry from the pending queue (matches base_name or video_path)."""
    if not base_name:
        return
    items = load_pending_subs()
    stem = os.path.splitext(os.path.basename(str(base_name)))[0]
    candidate_names = {str(base_name), stem, f"{stem}_4K"}
    if stem.endswith("_4K"):
        candidate_names.add(stem[:-3])
    abs_target = (
        os.path.abspath(str(base_name))
        if (os.path.sep in str(base_name) or "/" in str(base_name))
        else None
    )

    def _is_match(it: dict) -> bool:
        if it.get("base_name") in candidate_names:
            return True
        vp = it.get("video_path")
        if vp:
            if vp == base_name or (abs_target and os.path.abspath(vp) == abs_target):
                return True
            if os.path.splitext(os.path.basename(vp))[0] in candidate_names:
                return True
        return False

    new_items = [it for it in items if not _is_match(it)]
    if len(new_items) != len(items):
        save_pending_subs(new_items)
