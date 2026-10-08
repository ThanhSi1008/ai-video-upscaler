# -*- coding: utf-8 -*-
"""
subtrans/eval/metrics.py
Deterministic metrics for evaluation harness:
- Characters Per Second (CPS)
- CJK character leakage
- Romaji name retention in vocatives
- Pronoun relationship conformance
- Strict zero-overlap & gap compliance
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Set

# Regex nhận diện ký tự CJK (Kanji, Hiragana, Katakana, Bopomofo, Hangul)
CJK_REGEX = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]"
)


def normalize_nfc(text: str) -> str:
    """Chuẩn hóa Unicode sang dạng NFC."""
    return unicodedata.normalize("NFC", text or "")


def calculate_cps(text: str, duration_ms: int) -> float:
    """
    Tính tốc độ đọc Characters Per Second (CPS).
    CPS = số ký tự hiển thị / (duration_ms / 1000).
    """
    if duration_ms <= 0:
        return 0.0
    clean = re.sub(r"\s+", " ", text or "").strip()
    return round(len(clean) / (duration_ms / 1000.0), 2)


def detect_cjk_leakage(text: str) -> List[str]:
    """Tìm tất cả các ký tự CJK còn sót lại trong bản dịch tiếng Việt."""
    if not text:
        return []
    return CJK_REGEX.findall(text)


def check_romaji_retention(vi_text: str, expected_names: List[str]) -> bool:
    """
    Kiểm tra xem ít nhất một trong các tên Romaji mong đợi có xuất hiện trong câu tiếng Việt hay không.
    So khớp không phân biệt hoa thường với ranh giới từ.
    """
    if not expected_names or not vi_text:
        return False
    norm_vi = normalize_nfc(vi_text).lower()
    for name in expected_names:
        norm_name = normalize_nfc(name).strip().lower()
        if not norm_name:
            continue
        # Ranh giới từ hoặc cụm từ
        pattern = r"\b" + re.escape(norm_name) + r"\b"
        if re.search(pattern, norm_vi):
            return True
    return False


def check_pronoun_match(vi_text: str, allowed_pronouns: List[str]) -> bool:
    """
    Kiểm tra xem câu phụ đề có chứa bất kỳ đại từ nào trong tập đại từ được phép hay không.
    (Ví dụ: Fran gọi Sư phụ thì được phép chứa: 'sư phụ', 'ngài').
    """
    if not allowed_pronouns or not vi_text:
        return False
    norm_vi = normalize_nfc(vi_text).lower()
    for p in allowed_pronouns:
        norm_p = normalize_nfc(p).strip().lower()
        if not norm_p:
            continue
        pattern = r"\b" + re.escape(norm_p) + r"\b"
        if re.search(pattern, norm_vi):
            return True
    return False


def check_zero_overlap_violation(end_ms: int, next_start_ms: int, min_gap_ms: int = 40) -> bool:
    """
    Kiểm tra xem hai mốc thời gian liên tiếp có vi phạm khoảng cách an toàn (gap < 40ms) hoặc overlap không.
    Trả về True nếu VI PHẠM (end_ms > next_start_ms - min_gap_ms).
    """
    return end_ms > (next_start_ms - min_gap_ms)


def compute_harness_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tổng hợp kết quả kiểm thử đánh giá:
    - Tổng số case
    - Tỷ lệ đỗ (%)
    - Tỷ lệ vi phạm CPS (> 20 CPS)
    - Tỷ lệ sót CJK
    - Tỷ lệ giữ tên Romaji
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "pass_rate_pct": 100.0}

    passed = sum(1 for r in results if r.get("passed", False))
    cjk_fails = sum(1 for r in results if r.get("cjk_leakage"))
    cps_high = sum(1 for r in results if r.get("cps", 0.0) > 20.0)
    romaji_ok = sum(1 for r in results if r.get("romaji_retained") is True)
    romaji_total = sum(1 for r in results if r.get("expected_romaji"))

    return {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate_pct": round((passed / total) * 100.0, 2),
        "cjk_leak_count": cjk_fails,
        "cjk_leak_rate_pct": round((cjk_fails / total) * 100.0, 2),
        "high_cps_count": cps_high,
        "high_cps_rate_pct": round((cps_high / total) * 100.0, 2),
        "romaji_retention_rate_pct": round((romaji_ok / romaji_total * 100.0), 2) if romaji_total > 0 else 100.0,
    }
