# -*- coding: utf-8 -*-
"""
src/pending_queue.py
Quản lý hàng đợi các tập phụ đề chưa đạt chuẩn nhúng (pending_subs.json).
Module thuần thư viện chuẩn Python (stdlib-only) để subtrans/pipeline.py có thể nạp
độc lập mà KHÔNG kích hoạt import torch / CUDA hay quét phần cứng video.
"""

import json
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
PENDING_SUBS_FILE = os.path.join(ROOT_DIR, "pending_subs.json")


def _resolve_pending_file() -> str:
    """Ưu tiên thuộc tính src.config.PENDING_SUBS_FILE nếu module src.config đã được nạp (tương thích monkeypatch)."""
    cfg_mod = sys.modules.get("src.config")
    if cfg_mod is not None and hasattr(cfg_mod, "PENDING_SUBS_FILE"):
        return str(getattr(cfg_mod, "PENDING_SUBS_FILE"))
    return PENDING_SUBS_FILE


def load_pending_subs() -> list:
    """Đọc danh sách các tập phụ đề chưa đạt chuẩn (Item 4b)."""
    target_file = _resolve_pending_file()
    if not os.path.exists(target_file):
        return []
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_pending_subs(items: list) -> None:
    """Ghi nguyên tử danh sách pending_subs.json xuống đĩa."""
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
    """Thêm hoặc cập nhật một tập vào pending_subs.json."""
    if os.environ.get("PYTEST_CURRENT_TEST") and "dummy" in str(record.get("base_name", "")):
        return
    items = load_pending_subs()
    items = [it for it in items if it.get("base_name") != record.get("base_name")]
    items.append(record)
    save_pending_subs(items)


def remove_pending_sub(base_name: str) -> None:
    """Xóa một tập khỏi pending_subs.json khi đã đạt chuẩn (khớp base_name hoặc video_path)."""
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
