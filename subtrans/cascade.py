# -*- coding: utf-8 -*-
"""
subtrans/cascade.py
Stage P2: Smart Cloud Cascade (100% Cloud API: Flash-Lite -> Flash).
- Difficulty scoring function S(cue) in [0.0, 15.0] with series-aware calibration
- Two-tier pipeline:
  1. Draft Pass: high-throughput, low-quota Gemini Flash-Lite (2,000 RPD)
  2. Repair/Editor Pass: high-tier Gemini Flash on difficult/uncertain cues (only 5-15% of episode)
- Guardrails:
  + Cổng Re-lint & Accept/Reject: Chạy lại linter sau repair, chỉ nhận bản sửa nếu không tạo lỗi mới
  + Fail-Open: Tự động giữ nguyên bản draft Lite khi gặp 429, timeout, bad JSON schema
  + Đầy đủ telemetry: Phân phối S(cue), edit rate, rejection rate, provenance reason_code
"""

import re
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel

from subtrans.models import Cue, Glossary
from subtrans.linter import lint_cue, LintViolation
from subtrans.gemini_backend import GeminiTranslator, filter_relevant_glossary

logger = logging.getLogger("subtrans.cascade")


class RepairOutputItem(BaseModel):
    id: int
    vi: str
    fix_rationale: Optional[str] = None


REPAIR_SYSTEM_INSTRUCTION = (
    "You are the Senior Chief Anime Subtitle Editor. "
    "Your job is to review draft Vietnamese subtitles for difficult, ambiguous, or error-flagged anime dialogue cues.\n"
    "Strict Editing Principles:\n"
    "1. Resolve any naming errors, pronoun ambiguities, or mistranslations described in 'issues'.\n"
    "2. CRITICAL VOCATIVE RULE: When Japanese dialogue calls a character by proper name (e.g. フラン -> Fran, ジャン -> Jean, ネル -> Nell, 玲琳 -> Reirin), "
    "ALWAYS KEEP the Romaji proper name directly in the Vietnamese subtitle. Never replace proper names with generic standalone pronouns ('anh/em/cậu/cô/mày').\n"
    "3. NEVER use literal Sino-Vietnamese calques for names (no 'Pháp Lạp', 'Ô Lỗ Thi', 'Huệ Nguyệt', 'Tuệ Nguyệt', 'Linh Lâm').\n"
    "4. Natural spoken Vietnamese: Output exactly one polished, natural spoken dialogue line per cue.\n"
    "Output must be a valid JSON array matching the provided schema."
)


def score_cue_difficulty(
    cue: Cue,
    violations: Optional[List[LintViolation]] = None,
    series: Optional[str] = None,
) -> float:
    """
    Computes difficulty score S(cue) for a subtitle cue in [0.0, 15.0].
    Calibrated weights:
    - Linter ERROR (+4.0), Linter WARNING (+1.5)
    - LLM uncertain flag (+4.0), specific reason (+2.0)
    - EN reference conflict (+3.0)
    - Dialogue length (>35 JA or >70 EN chars: +1.5)
    - Archaic/noble Japanese markers tuned by series (+2.0)
    """
    score = 0.0

    # 1. Linter violations (độ ưu tiên chất lượng cao nhất)
    if violations:
        for v in violations:
            if v.severity == "ERROR":
                score += 4.0
            elif v.severity == "WARNING":
                score += 1.5

    # 2. Uncertainty flag from draft model
    if cue.uncertain:
        score += 4.0
    if cue.uncertain_reason and cue.uncertain_reason.lower() not in ("none", "", "no"):
        score += 2.0

    # 3. English reference conflict
    if getattr(cue, "en_conflict", False):
        score += 3.0

    # 4. Long or multi-clause lines (nguy cơ tràn CPS hoặc cắt vụn)
    ja_len = len(cue.ja or "")
    en_len = len(cue.en or "")
    if ja_len > 35 or en_len > 70:
        score += 1.5

    # 5. Archaic, divine, or noble speech in Japanese (tách theo series để tránh dương tính giả)
    ja_text = cue.ja or ""
    series_clean = (series or "").lower()
    if series is None:
        # v0 compatibility / unstated series
        if re.search(r"(妾|汝|主殿|お主|貴様|陛下)", ja_text):
            score += 2.0
    elif "futsutsuka" in series_clean or "cung đình" in series_clean:
        # Series cung đình: bệ hạ, khanh, điện hạ, thiếp
        if re.search(r"(妾|陛下|卿|殿|主殿|閣下|汝|お主)", ja_text):
            score += 2.0
    else:
        # Series hành động / thông thường: bỏ 貴様 và 陛下 để tránh false positive
        if re.search(r"(妾|汝|主殿|お主)", ja_text):
            score += 2.0

    final_score = round(min(15.0, score), 2)
    cue.difficulty_score = final_score
    return final_score


def find_cues_needing_repair(
    cues: List[Cue],
    glossary: Optional[Glossary] = None,
    difficulty_threshold: float = 4.0,
    series: Optional[str] = None,
) -> List[Tuple[Cue, float, List[str]]]:
    """
    Filters cues that require high-tier editor review.
    Returns list of (cue, score, issues_list).
    """
    candidates: List[Tuple[Cue, float, List[str]]] = []

    for cue in cues:
        if getattr(cue, "is_dropped_sound", False) or getattr(cue, "is_skipped", False):
            continue
        # Skip untranslated cues (cannot review non-existent draft)
        if getattr(cue, "backend_used", None) == "untranslated" or (cue.vi or "").strip() in ["[Chưa dịch]", "(Chưa dịch)", "[Untranslated]", ""]:
            continue

        violations = lint_cue(cue, glossary=glossary)
        score = score_cue_difficulty(cue, violations=violations, series=series)

        issues: List[str] = []
        if cue.uncertain:
            issues.append(f"draft_uncertain: {cue.uncertain_reason or 'yes'}")
        if getattr(cue, "en_conflict", False):
            issues.append("en_reference_conflict")
        for v in violations:
            issues.append(f"{v.rule_id} ({v.severity}): {v.message}")

        has_error = any(v.severity == "ERROR" for v in violations)
        if score >= difficulty_threshold or has_error:
            candidates.append((cue, score, issues))

    return candidates


def accept_or_reject_repaired_cue(
    cue: Cue,
    fixed_vi: str,
    glossary: Optional[Glossary] = None,
) -> Tuple[bool, str, List[LintViolation]]:
    """
    Guardrail: Re-lints the cue after Flash repair pass and decides to accept or reject.
    Returns: (is_accepted, reason_code, new_violations)
    """
    clean_fixed = fixed_vi.strip()
    if not clean_fixed:
        return False, "rejected_empty", []

    # If Flash didn't change anything
    if clean_fixed == (cue.draft_vi or cue.vi or "").strip():
        return True, "accepted_identical", []

    # Run linter on draft and proposed fix
    orig_violations = lint_cue(cue, glossary=glossary)
    temp_cue = cue.model_copy(update={"vi": clean_fixed})
    new_violations = lint_cue(temp_cue, glossary=glossary)

    orig_errors = sum(1 for v in orig_violations if v.severity == "ERROR")
    new_errors = sum(1 for v in new_violations if v.severity == "ERROR")

    # 1. Reject if introduced critical violations (CJK leakage or banned Sino-Vietnamese)
    critical_rules = {"LINT_CJK_LEAK", "LINT_BANNED_TERM", "no_cjk_leakage", "no_sino_vietnamese_names"}
    orig_crit = {v.rule_id for v in orig_violations if v.rule_id in critical_rules}
    new_crit = {v.rule_id for v in new_violations if v.rule_id in critical_rules}
    if new_crit - orig_crit:
        return False, "rejected_critical_violation", new_violations

    # 2. Reject if new ERROR violations introduced
    if new_errors > orig_errors:
        return False, "rejected_new_error", new_violations

    # 3. Check extreme CPS overflow (> 25 CPS ceiling)
    cps_overflow = any(v.rule_id == "LINT_HIGH_CPS" and "vượt trần cứng" in v.message for v in new_violations)
    if cps_overflow and not any(v.rule_id == "LINT_HIGH_CPS" and "vượt trần cứng" in v.message for v in orig_violations):
        return False, "rejected_cps_overflow", new_violations

    # 4. Accepted
    if new_errors < orig_errors or len(new_violations) < len(orig_violations):
        return True, "accepted_improved", new_violations
    return True, "accepted_valid", new_violations


def run_cascade_repair_pass(
    cues: List[Cue],
    repair_translator: GeminiTranslator,
    glossary: Optional[Glossary] = None,
    difficulty_threshold: float = 4.0,
    series: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the Cloud Cascade Repair/Editor Pass on cues needing review.
    Uses the provided high-tier repair_translator (e.g. gemini-3.8-flash).
    Features:
    - Fail-open resilience (429, timeout, invalid JSON -> retains Lite draft, flags unrepaired)
    - Re-lint accept/reject gate
    - Telemetry tracking (S-distribution, edit rate, rejection rate)
    """
    candidates = find_cues_needing_repair(
        cues,
        glossary=glossary,
        difficulty_threshold=difficulty_threshold,
        series=series
    )
    if not candidates:
        logger.info("✨ [CASCADE REPAIR] Không có cue nào cần chỉnh sửa (tất cả đều đạt chuẩn).")
        return {
            "cues_evaluated": len(cues),
            "candidates_count": 0,
            "repaired_count": 0,
            "rejected_count": 0,
            "edited_count": 0,
            "repair_ratio_pct": 0.0,
            "s_distribution": {"p50": 0.0, "p90": 0.0, "max": 0.0},
        }

    cues_to_fix = [item[0] for item in candidates]
    scores = [item[1] for item in candidates]
    s_dist = {
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "max": float(np.max(scores)),
    }

    logger.info(
        f"🛠️ [CASCADE REPAIR] Gom {len(cues_to_fix)}/{len(cues)} cues khó ({len(cues_to_fix)/len(cues)*100:.1f}%) "
        f"[S p50={s_dist['p50']:.1f}, p90={s_dist['p90']:.1f}, max={s_dist['max']:.1f}] "
        f"để biên tập chất lượng cao qua model '{repair_translator.model}'..."
    )

    # Preserve original draft text
    for c in cues_to_fix:
        if not c.draft_vi:
            c.draft_vi = c.vi

    # Dry-run check
    if getattr(repair_translator, "dry_run", False):
        for c in cues_to_fix:
            c.uncertain = False
            c.model_used = f"{repair_translator.model} (dry-run repair)"
            c.repair_reason_code = "dry_run_accepted"
        return {
            "cues_evaluated": len(cues),
            "candidates_count": len(cues_to_fix),
            "repaired_count": len(cues_to_fix),
            "rejected_count": 0,
            "edited_count": 0,
            "repair_ratio_pct": round(len(cues_to_fix) / max(1, len(cues)) * 100, 2),
            "s_distribution": s_dist,
        }

    # Prepare repair payload
    repair_items = []
    for cue, score, issues in candidates:
        item = {
            "id": cue.id,
            "ja": cue.ja,
            "en": cue.en,
            "draft_vi": cue.draft_vi or cue.vi,
            "speaker": cue.speaker or "unknown",
            "addressee": cue.addressee,
            "mode": cue.mode or "spoken",
            "issues": issues,
        }
        repair_items.append(item)

    payload = {"cues_to_review": repair_items}
    rel_glossary = filter_relevant_glossary(cues_to_fix, glossary)
    if rel_glossary:
        payload["glossary"] = rel_glossary

    prompt_lines = [
        "Review and repair the following difficult/ambiguous Vietnamese anime subtitle drafts.",
        "Fix all issues listed in 'issues' while ensuring natural phrasing and authentic character voice.",
        f"\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    ]
    prompt_str = "\n".join(prompt_lines)

    # Call repair API with Fail-Open Guardrail
    try:
        from google.genai import types
        config_kwargs = {
            "system_instruction": REPAIR_SYSTEM_INSTRUCTION,
            "response_mime_type": "application/json",
            "response_schema": list[RepairOutputItem],
            "temperature": 0.0,
        }
        response = repair_translator.client.models.generate_content(
            model=repair_translator.model,
            contents=prompt_str,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        data = json.loads(response.text.strip())
        id_to_fixed = {it["id"]: it["vi"] for it in data if "id" in it and "vi" in it}

        repaired_count = 0
        rejected_count = 0
        edited_count = 0

        for c in cues_to_fix:
            if c.id in id_to_fixed and id_to_fixed[c.id].strip():
                fixed_vi = id_to_fixed[c.id].strip()
                if fixed_vi != c.draft_vi:
                    edited_count += 1

                # Re-lint Accept / Reject Gate
                accepted, reason_code, _ = accept_or_reject_repaired_cue(c, fixed_vi, glossary=glossary)
                c.repair_reason_code = reason_code

                if accepted:
                    c.vi = fixed_vi
                    c.uncertain = False
                    c.uncertain_reason = "repaired"
                    c.model_used = f"{repair_translator.model} (repair)"
                    repaired_count += 1
                else:
                    # Rejected: preserve draft
                    c.vi = c.draft_vi
                    rejected_count += 1
                    logger.info(f"🛡️ [CASCADE REJECT] Từ chối bản sửa cue #{c.id} ({reason_code}), giữ draft: '{c.draft_vi}'")
            else:
                c.repair_reason_code = "unrepaired_missing_id"

        edit_rate = round(edited_count / max(1, len(cues_to_fix)) * 100, 1)
        rejection_rate = round(rejected_count / max(1, len(cues_to_fix)) * 100, 1)
        logger.info(
            f"✅ [CASCADE REPAIR] Hoàn tất: nhận {repaired_count}/{len(cues_to_fix)} cues "
            f"(Flash sửa: {edited_count} [{edit_rate}%], từ chối: {rejected_count} [{rejection_rate}%])."
        )
        return {
            "cues_evaluated": len(cues),
            "candidates_count": len(cues_to_fix),
            "repaired_count": repaired_count,
            "rejected_count": rejected_count,
            "edited_count": edited_count,
            "repair_ratio_pct": round(repaired_count / max(1, len(cues)) * 100, 2),
            "s_distribution": s_dist,
        }

    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
            fail_code = "unrepaired_quota_exceeded"
        elif "time" in err_msg or "timeout" in err_msg or "timed out" in err_msg or isinstance(e, TimeoutError):
            fail_code = "unrepaired_timeout"
        elif "json" in err_msg or "schema" in err_msg or "expecting value" in err_msg or isinstance(e, (json.JSONDecodeError, ValueError)):
            fail_code = "unrepaired_bad_schema"
        else:
            fail_code = "unrepaired_api_error"

        logger.warning(f"⚠️ [CASCADE FAIL-OPEN] Lượt repair thất bại ({e}). Tự động giữ nguyên bản draft Lite ({fail_code}).")
        for c in cues_to_fix:
            c.vi = c.draft_vi
            c.repair_reason_code = fail_code

        return {
            "cues_evaluated": len(cues),
            "candidates_count": len(cues_to_fix),
            "repaired_count": 0,
            "rejected_count": 0,
            "edited_count": 0,
            "repair_ratio_pct": 0.0,
            "error": str(e),
            "fail_open_code": fail_code,
            "s_distribution": s_dist,
        }
