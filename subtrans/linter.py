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
from subtrans.ja_morph_tagger import strip_ja_furigana, has_cjk_characters
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
RULE_GLOSSARY_TERM = "LINT_GLOSSARY_TERM"
RULE_PRONOUN_CONSISTENCY = "LINT_PRONOUN_CONSISTENCY"

ALL_RULE_IDS = {
    RULE_CJK_LEAK,
    RULE_HIGH_CPS,
    RULE_UNICODE_NOT_NFC,
    RULE_ROMAJI_MISSING,
    RULE_BANNED_TERM,
    RULE_EMPTY_TEXT,
    RULE_ZERO_OVERLAP,
    RULE_GLOSSARY_TERM,
    RULE_PRONOUN_CONSISTENCY,
}

CRITICAL_RULES = {
    RULE_CJK_LEAK,
    RULE_BANNED_TERM,
}

# Banned translationese / calques -> recommended replacement
BANNED_TERMS_MAP: Dict[str, str] = {
    "Pháp Lạp": "Fran",
    "Ô Lỗ Thi": "Urushi",
    "Linh Lâm": "Reirin",
    "Huệ Nguyệt": "Keigetsu",
    "Tuệ Nguyệt": "Keigetsu",
    "A Mạn Đạt": "Amanda",
    "Khắc Lâm Mỗ Đặc": "Klimt",
    "thế giới quan": "thế giới",
}

# Standard Anime Vocative Patterns (Japanese name calling -> Romaji expected)
# Short Katakana names use non-Katakana lookaround boundaries so loanwords like プログラム / パネル never false-positive.
VOCATIVE_MAPPINGS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?<![\u30A0-\u30FFー])フラン(?![\u30A0-\u30FFー])"), "Fran"),
    (re.compile(r"(?<![\u30A0-\u30FFー])ジャン(?![\u30A0-\u30FFー])"), "Jean"),
    (re.compile(r"(?<![\u30A0-\u30FFー])ネル(?![\u30A0-\u30FFー])"), "Nell"),
    (re.compile(r"玲琳"), "Reirin"),
    (re.compile(r"慧月"), "Keigetsu"),
    (re.compile(r"(?<![\u30A0-\u30FFー])ウルシ(?![\u30A0-\u30FFー])"), "Urushi"),
    (re.compile(r"師匠"), "Sư phụ"),
]

_PURE_KATAKANA_RE = re.compile(r"^[\u30A0-\u30FFー]+$")
_HOSTILE_JA_MARKERS = ("貴様", "きさま", "てめえ", "てめぇ", "お前", "この野郎", "くそ")


def _ja_contains_character_name(ja_raw: str, ja_clean: str, char_ja: str) -> bool:
    """Checks if Japanese dialogue contains a character name without Katakana loanword false positives."""
    c_clean = strip_ja_furigana(char_ja or "").strip()
    if not c_clean:
        return False
    if _PURE_KATAKANA_RE.match(c_clean) and len(c_clean) <= 3:
        katakana_pat = r"(?<![\u30A0-\u30FFー])" + re.escape(c_clean) + r"(?![\u30A0-\u30FFー])"
        return bool(re.search(katakana_pat, ja_clean) or re.search(katakana_pat, ja_raw))
    return bool(c_clean in ja_clean or (char_ja and char_ja in ja_raw))


def _expand_romaji_variants(display_name: str, aliases: Optional[List[str]] = None) -> List[str]:
    """Builds valid Romaji forms for a character (full name, explicit aliases, and >=3-char given/family tokens)."""
    forms: List[str] = []
    if display_name:
        forms.append(display_name.strip())
        # If full name like "Jean Doubee" or "Subaru Natsuki", accept "Jean" or "Subaru" in dialogue
        parts = [p.strip() for p in re.split(r"[\s・\-]+", display_name.strip()) if len(p.strip()) >= 3]
        if len(parts) >= 2:
            for p in parts:
                if p[0].isupper():
                    forms.append(p)
    for al in aliases or []:
        if isinstance(al, str) and al.strip():
            forms.append(al.strip())
    return list(dict.fromkeys(forms))


@dataclass
class LintViolation:
    rule_id: str
    severity: Literal["ERROR", "WARNING", "INFO"]
    cue_id: int
    message: str
    suggestion: Optional[str] = None
    is_hard_ceiling: bool = False
    span: Optional[Tuple[int, int]] = None


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
                is_hard_ceiling=True,
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

    # 6. Romaji Proper Name Retention in Vocatives (furigana-transparent & Katakana-boundary safe)
    norm_text_lower = norm_text.lower()
    ja_text = cue.ja or ""
    ja_clean = strip_ja_furigana(ja_text)
    expected_romaji_groups: Dict[str, List[str]] = {}

    # From built-in vocative mappings
    for regex_ja, romaji in VOCATIVE_MAPPINGS:
        if regex_ja.search(ja_text) or regex_ja.search(ja_clean):
            expected_romaji_groups.setdefault(romaji, _expand_romaji_variants(romaji))

    # From Glossary characters
    if glossary and glossary.characters:
        for char in glossary.characters:
            if char.name_ja and char.name_vi and _ja_contains_character_name(ja_text, ja_clean, char.name_ja):
                variants = _expand_romaji_variants(char.name_vi, getattr(char, "aliases", None))
                existing = expected_romaji_groups.get(char.name_vi, [])
                expected_romaji_groups[char.name_vi] = list(dict.fromkeys(existing + variants))

    # From CastSheet characters if provided
    if cast_sheet and hasattr(cast_sheet, "characters"):
        for char in cast_sheet.characters:
            char_ja = getattr(char, "name_ja", "")
            char_vi = getattr(char, "name_vi", "")
            raw_aliases = getattr(char, "aliases", None)
            char_aliases = raw_aliases if isinstance(raw_aliases, (list, tuple)) else None
            if char_ja and char_vi and _ja_contains_character_name(ja_text, ja_clean, char_ja):
                variants = _expand_romaji_variants(char_vi, char_aliases)
                existing = expected_romaji_groups.get(char_vi, [])
                expected_romaji_groups[char_vi] = list(dict.fromkeys(existing + variants))

    for expected_name, valid_variants in expected_romaji_groups.items():
        matched_any = False
        for variant in valid_variants:
            norm_var = normalize_nfc(variant).strip().lower()
            if not norm_var:
                continue
            pattern = r"\b" + re.escape(norm_var) + r"\b"
            if re.search(pattern, norm_text_lower):
                matched_any = True
                break
        if not matched_any:
            violations.append(
                LintViolation(
                    rule_id=RULE_ROMAJI_MISSING,
                    severity="ERROR",
                    cue_id=cue.id,
                    message=f"Cue {cue.id} called '{expected_name}' in Japanese, but the Romaji name was omitted or changed.",
                    suggestion=f"Keep Romaji proper name '{expected_name}' directly in the Vietnamese dialogue.",
                )
            )

    # 7. Glossary Canonical Term Compliance (furigana-transparent, multi-char CJK terms)
    if glossary and glossary.terms and has_cjk_characters(ja_clean):
        for term in glossary.terms:
            t_ja_clean = strip_ja_furigana(term.ja or "").strip()
            if len(t_ja_clean) < 2:
                continue
            if t_ja_clean in ja_clean or (term.ja and term.ja in ja_text):
                valid_forms = [normalize_nfc(term.vi).strip().lower()] + [
                    normalize_nfc(a).strip().lower() for a in (term.alt or []) if a
                ]
                valid_forms = [vf for vf in valid_forms if vf]
                if valid_forms and not any(vf in norm_text_lower for vf in valid_forms):
                    violations.append(
                        LintViolation(
                            rule_id=RULE_GLOSSARY_TERM,
                            severity="WARNING",
                            cue_id=cue.id,
                            message=(
                                f"Cue {cue.id} contains glossary term '{term.ja}' "
                                f"but expected translation '{term.vi}' was not found."
                            ),
                            suggestion=f"Use glossary term '{term.vi}'.",
                        )
                    )

    # 8. Speaker -> Addressee Pronoun Consistency Check (when explicit CoT p1/p2 is present)
    p1_val = normalize_nfc(getattr(cue, "p1", None) or "").strip().lower()
    p2_val = normalize_nfc(getattr(cue, "p2", None) or "").strip().lower()
    if (
        (p1_val or p2_val)
        and (cue.mode or "spoken") == "spoken"
        and not cue.uncertain
        and glossary
        and glossary.characters
        and cue.speaker
        and cue.addressee
        and cue.speaker.strip().lower() != "unknown"
        and cue.addressee.strip().lower() != "unknown"
    ):
        is_hostile_shift = (
            p1_val in {"tao"}
            or p2_val in {"mày"}
            or any(hm in ja_clean for hm in _HOSTILE_JA_MARKERS)
        )
        if not is_hostile_shift:
            spk_low = cue.speaker.strip().lower()
            spk_ja_clean = strip_ja_furigana(cue.speaker.strip())
            addr_low = cue.addressee.strip().lower()
            for char in glossary.characters:
                c_vi_low = (char.name_vi or "").strip().lower()
                c_ja_clean = strip_ja_furigana(char.name_ja or "").strip()
                c_aliases_low = {a.strip().lower() for a in getattr(char, "aliases", []) if a}
                if spk_low == c_vi_low or spk_low in c_aliases_low or (c_ja_clean and spk_ja_clean == c_ja_clean):
                    matched_rule = None
                    for st in char.speaks_to or []:
                        if st.listener.strip().lower() == addr_low:
                            matched_rule = st
                            break
                    if matched_rule:
                        allowed_self = {
                            normalize_nfc(tok).strip().lower()
                            for tok in re.split(r"[/|,]", matched_rule.self or "")
                            if tok.strip()
                        }
                        allowed_other = {
                            normalize_nfc(tok).strip().lower()
                            for tok in re.split(r"[/|,]", matched_rule.other or "")
                            if tok.strip()
                        }
                        mismatch_parts = []
                        if p1_val and allowed_self and p1_val not in allowed_self:
                            mismatch_parts.append(f"p1='{p1_val}' (expected '{matched_rule.self}')")
                        if p2_val and allowed_other and p2_val not in allowed_other:
                            mismatch_parts.append(f"p2='{p2_val}' (expected '{matched_rule.other}')")
                        if mismatch_parts:
                            violations.append(
                                LintViolation(
                                    rule_id=RULE_PRONOUN_CONSISTENCY,
                                    severity="WARNING",
                                    cue_id=cue.id,
                                    message=(
                                        f"Cue {cue.id} ({char.name_vi} -> {matched_rule.listener}) pronoun drift: "
                                        + ", ".join(mismatch_parts)
                                    ),
                                    suggestion=f"Use canonical pronouns '{matched_rule.self} / {matched_rule.other}'.",
                                )
                            )
                    break

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
