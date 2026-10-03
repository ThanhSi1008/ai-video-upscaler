# -*- coding: utf-8 -*-
"""
subtrans/validation.py
Validation rules shared across all translation backends.
"""

import re
from typing import List, Dict, Any, Tuple

# Regex to detect leftover Japanese characters:
# Hiragana: \u3040-\u309F, Katakana: \u30A0-\u30FF, Kanji: \u4E00-\u9FAF
JAPANESE_CHAR_REGEX = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]")
CHATTER_REGEX = re.compile(r"(?i)(here is the translation|translation:|vietnamese:|dịch:|^```)", re.IGNORECASE)


def validate_translations(expected_ids: List[int], translated_items: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validates translated batch items:
    1. Count and IDs match exactly and in order.
    2. No empty translation.
    3. No leftover Japanese characters.
    4. No conversational chatter or markdown remnants.
    Returns (is_valid, error_reason).
    """
    if len(translated_items) != len(expected_ids):
        return False, f"Count mismatch: expected {len(expected_ids)} items, got {len(translated_items)}"

    for exp_id, item in zip(expected_ids, translated_items):
        item_id = item.get("id")
        if item_id != exp_id:
            return False, f"ID order mismatch: expected id {exp_id}, got {item_id}"

        vi = item.get("vi", "")
        if not vi or not vi.strip():
            return False, f"Empty translation for id {exp_id}"

        vi_clean = vi.strip()
        if JAPANESE_CHAR_REGEX.search(vi_clean):
            return False, f"Leftover Japanese characters in id {exp_id}: '{vi_clean}'"

        if CHATTER_REGEX.search(vi_clean):
            return False, f"Chatter/markdown artifact detected in id {exp_id}: '{vi_clean}'"

    return True, ""
