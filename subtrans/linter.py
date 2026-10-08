# -*- coding: utf-8 -*-
"""
subtrans/linter.py
Linter V1 for Vietnamese anime subtitle quality assurance:
- CJK character leakage detection (Kanji, Hiragana, Katakana)
- Reading speed (CPS) validation & duration stretching
- Unicode NFC normalization
- Romaji proper name retention in vocatives
- Banned machine translationese / Sino-Vietnamese calques
- Strict zero-overlap timing compliance (40ms inter-cue gap)
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Literal

from subtrans.models import Cue, Glossary
from subtrans.eval.metrics import (
    calculate_cps,
    detect_cjk_leakage,
    normalize_nfc,
    check_zero_overlap_violation,
)

# Rule IDs
RULE_CJK_LEAK = "LINT_CJK_LEAK"
RULE_HIGH_CPS = "LINT_HIGH_CPS"
RULE_UNICODE_NOT_NFC = "LINT_UNICODE_NOT_NFC"
RULE_ROMAJI_MISSING = "LINT_ROMAJI_MISSING"
RULE_BANNED_TERM = "LINT_BANNED_TERM"
RULE_EMPTY_TEXT = "LINT_EMPTY_TEXT"
RULE_ZERO_OVERLAP = "LINT_ZERO_OVERLAP"

# Banned translationese / calques -> recommended replacement
BANNED_TERMS_MAP: Dict[str, str] = {
    "Pháp Lạp": "Fran",
    "Ô Lỗ Thi": "Urushi",
    "Linh Lâm": "Reirin",
    "Huệ Nguyệt": "Keigetsu",
    "Tuệ Nguyệt": "Keigetsu",
    "thế giới quan": "thế giới",
}

# Standard Anime Vocative Patterns (Japanese name calling -> Romaji expected)
VOCATIVE_MAPPINGS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"フラン"), "Fran"),
    (re.compile(r"ジャン"), "Jean"),
    (re.compile(r"ネル"), "Nell"),
    (re.compile(r"玲琳"), "Reirin"),
    (re.compile(r"慧月"), "Keigetsu"),
    (re.compile(r"ウルシ"), "Urushi"),
    (re.compile(r"師匠"), "Sư phụ"),
]


@dataclass
class LintViolation:
    rule_id: str
    severity: Literal["ERROR", "WARNING", "INFO"]
    cue_id: int
    message: str
    suggestion: Optional[str] = None


def lint_cue(
    cue: Cue,
    glossary: Optional[Glossary] = None,
    cast_sheet: Optional[Any] = None,
    max_safe_cps: float = 20.0,
    max_hard_cps: float = 25.0,
) -> List[LintViolation]:
    """Inspects an individual subtitle cue and returns a list of lint violations."""
    violations: List[LintViolation] = []
    text = (cue.vi or "").strip()

    # 1. Empty or untranslated placeholder check
    if not text or text in ["[Chưa dịch]", "(Chưa dịch)", "[Untranslated]", "[DRY-RUN]"]:
        violations.append(
            LintViolation(
                rule_id=RULE_EMPTY_TEXT,
                severity="ERROR",
                cue_id=cue.id,
                message=f"Cue {cue.id} has empty or placeholder translation: '{text}'",
                suggestion="Translate line to Vietnamese.",
            )
        )
        return violations

    # 2. CJK Leakage Check
    leaked_cjk = detect_cjk_leakage(text)
    if leaked_cjk:
        unique_cjk = "".join(dict.fromkeys(leaked_cjk))
        violations.append(
            LintViolation(
                rule_id=RULE_CJK_LEAK,
                severity="ERROR",
                cue_id=cue.id,
                message=f"Cue {cue.id} contains leftover CJK characters: '{unique_cjk}'",
                suggestion="Translate or remove Japanese/Chinese characters.",
            )
        )

    # 3. Unicode NFC Normalization Check
    if not unicodedata.is_normalized("NFC", text):
        violations.append(
            LintViolation(
                rule_id=RULE_UNICODE_NOT_NFC,
                severity="WARNING",
                cue_id=cue.id,
                message=f"Cue {cue.id} text is not normalized to Unicode NFC form.",
                suggestion="Normalize using unicodedata.normalize('NFC', text).",
            )
        )

    # 4. Characters Per Second (CPS) Check
    cps = calculate_cps(text, cue.duration_ms)
    if cps > max_hard_cps:
        violations.append(
            LintViolation(
                rule_id=RULE_HIGH_CPS,
                severity="ERROR",
                cue_id=cue.id,
                message=f"Cue {cue.id} reading speed is critically high ({cps:.1f} CPS > hard cap {max_hard_cps:.1f}).",
                suggestion="Extend duration into adjacent gap or condense wording.",
            )
        )
    elif cps > max_safe_cps:
        violations.append(
            LintViolation(
                rule_id=RULE_HIGH_CPS,
                severity="WARNING",
                cue_id=cue.id,
                message=f"Cue {cue.id} reading speed is high ({cps:.1f} CPS > safe cap {max_safe_cps:.1f}).",
                suggestion="Extend duration or shorten phrasing.",
            )
        )

    # 5. Banned Translationese / Sino-Vietnamese calques check
    norm_text = normalize_nfc(text)
    for banned, repl in BANNED_TERMS_MAP.items():
        pattern = r"\b" + re.escape(banned) + r"\b"
        if re.search(pattern, norm_text, flags=re.IGNORECASE):
            violations.append(
                LintViolation(
                    rule_id=RULE_BANNED_TERM,
                    severity="ERROR",
                    cue_id=cue.id,
                    message=f"Cue {cue.id} contains banned translationese/calque '{banned}'.",
                    suggestion=f"Replace with '{repl}'.",
                )
            )

    # 6. Romaji Proper Name Retention in Vocatives
    # Check vocative patterns in Japanese dialogue
    norm_text_lower = norm_text.lower()
    ja_text = cue.ja or ""
    expected_romaji_names: List[str] = []

    # From built-in vocative mappings
    for regex_ja, romaji in VOCATIVE_MAPPINGS:
        if regex_ja.search(ja_text):
            expected_romaji_names.append(romaji)

    # From Glossary characters
    if glossary and glossary.characters:
        for char in glossary.characters:
            if char.name_ja and char.name_ja in ja_text:
                expected_romaji_names.append(char.name_vi)

    # From CastSheet characters if provided
    if cast_sheet and hasattr(cast_sheet, "characters"):
        for char in cast_sheet.characters:
            char_ja = getattr(char, "name_ja", "")
            char_vi = getattr(char, "name_vi", "")
            if char_ja and char_ja in ja_text and char_vi:
                expected_romaji_names.append(char_vi)

    for expected_name in set(expected_romaji_names):
        norm_expected = normalize_nfc(expected_name).strip().lower()
        if not norm_expected:
            continue
        pattern = r"\b" + re.escape(norm_expected) + r"\b"
        if not re.search(pattern, norm_text_lower):
            violations.append(
                LintViolation(
                    rule_id=RULE_ROMAJI_MISSING,
                    severity="ERROR",
                    cue_id=cue.id,
                    message=f"Cue {cue.id} called '{expected_name}' in Japanese, but the Romaji name was omitted or changed.",
                    suggestion=f"Keep Romaji proper name '{expected_name}' directly in the Vietnamese dialogue.",
                )
            )

    return violations


def lint_subtitles(
    cues: List[Cue],
    glossary: Optional[Glossary] = None,
    cast_sheet: Optional[Any] = None,
    min_gap_ms: int = 40,
    max_safe_cps: float = 20.0,
    max_hard_cps: float = 25.0,
) -> List[LintViolation]:
    """
    Runs full linter checks on a sequence of subtitle cues, including inter-cue timing.
    """
    all_violations: List[LintViolation] = []
    kept_cues = [c for c in cues if not getattr(c, "is_dropped_sound", False) and not getattr(c, "is_skipped", False)]

    # 1. Per-cue checks
    for cue in kept_cues:
        all_violations.extend(
            lint_cue(
                cue=cue,
                glossary=glossary,
                cast_sheet=cast_sheet,
                max_safe_cps=max_safe_cps,
                max_hard_cps=max_hard_cps,
            )
        )

    # 2. Sequence inter-cue checks (Zero-overlap & safe gap)
    for i in range(len(kept_cues) - 1):
        c1 = kept_cues[i]
        c2 = kept_cues[i + 1]
        if check_zero_overlap_violation(c1.end, c2.start, min_gap_ms=min_gap_ms):
            all_violations.append(
                LintViolation(
                    rule_id=RULE_ZERO_OVERLAP,
                    severity="ERROR",
                    cue_id=c1.id,
                    message=f"Cue {c1.id} (end: {c1.end}ms) violates {min_gap_ms}ms gap before Cue {c2.id} (start: {c2.start}ms).",
                    suggestion=f"Set Cue {c1.id} end <= {c2.start - min_gap_ms}ms.",
                )
            )

    return all_violations


def auto_fix_cues(
    cues: List[Cue],
    min_gap_ms: int = 40,
    max_safe_cps: float = 20.0,
) -> Tuple[List[Cue], int]:
    """
    Applies deterministic auto-fixes to subtitle cues:
    1. Normalizes text to Unicode NFC form.
    2. Collapses redundant spaces and standardizes ellipsis.
    3. Stretches cue duration when CPS > max_safe_cps and an adjacent gap is available.
    Returns (fixed_cues, fixes_applied_count).
    """
    fixes_count = 0
    kept = [c for c in cues if not getattr(c, "is_dropped_sound", False) and not getattr(c, "is_skipped", False)]
    if not kept:
        return cues, 0

    for i, cue in enumerate(kept):
        if not cue.vi:
            continue

        raw = cue.vi
        # 1. Unicode NFC
        norm = normalize_nfc(raw)
        # 2. Ellipsis & whitespace standardization
        norm = norm.replace("…", "...").replace("\r\n", "\n")
        norm = re.sub(r"[ \t]+", " ", norm).strip()

        if norm != raw:
            cue.vi = norm
            fixes_count += 1

        # 3. Duration stretching to lower CPS when a gap exists
        cps = calculate_cps(cue.vi, cue.duration_ms)
        if cps > max_safe_cps:
            clean_len = len(re.sub(r"\s+", " ", cue.vi).strip())
            needed_ms = int((clean_len / max_safe_cps) * 1000)
            if i + 1 < len(kept):
                next_start = kept[i + 1].start
                available_end = next_start - min_gap_ms
            else:
                available_end = cue.start + needed_ms

            if available_end > cue.end:
                new_end = min(cue.start + needed_ms, available_end)
                if new_end > cue.end:
                    cue.end = new_end
                    fixes_count += 1

    return cues, fixes_count


def format_lint_report(violations: List[LintViolation]) -> str:
    """Formats list of violations into a human-readable markdown summary."""
    if not violations:
        return "✅ [Linter V1] Tất cả kiểm tra chất lượng đều ĐẠT (PASS - 0 violations).\n"

    errors = [v for v in violations if v.severity == "ERROR"]
    warnings = [v for v in violations if v.severity == "WARNING"]
    infos = [v for v in violations if v.severity == "INFO"]

    lines = [
        f"### 📋 Báo Cáo Kiểm Định Phụ Đề (Linter V1)",
        f"- **Tổng số vi phạm:** {len(violations)} (Lỗi nghiêm trọng: {len(errors)}, Cảnh báo: {len(warnings)}, Góp ý: {len(infos)})",
        "",
        "| ID Cue | Mức độ | Quy tắc | Chi tiết lỗi | Gợi ý khắc phục |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for v in violations:
        badge = "🔴 ERROR" if v.severity == "ERROR" else ("🟡 WARNING" if v.severity == "WARNING" else "🔵 INFO")
        lines.append(f"| #{v.cue_id} | {badge} | `{v.rule_id}` | {v.message} | {v.suggestion or '-'} |")

    return "\n".join(lines) + "\n"
