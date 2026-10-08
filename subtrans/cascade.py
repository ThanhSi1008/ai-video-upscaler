# -*- coding: utf-8 -*-
"""
subtrans/cascade.py
Stage P2: Smart Cloud Cascade (100% Cloud API: Flash-Lite -> Flash).
- Difficulty scoring function S(cue) combining uncertainty, Linter violations, and linguistic markers
- Two-tier pipeline:
  1. Draft Pass: high-throughput, low-quota Gemini Flash-Lite (2,000 RPD)
  2. Repair/Editor Pass: high-tier Gemini Flash on difficult/uncertain cues (only 5-15% of episode)
"""

import re
import json
import logging
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
) -> float:
    """
    Computes difficulty score S(cue) for a subtitle cue in [0.0, 15.0].
    Signals evaluated:
    - LLM uncertain flag and reason (+4.0 / +2.0)
    - EN reference conflict (+3.0)
    - Linter violations (ERROR +4.0, WARNING +1.5)
    - Dialogue length and complexity (+1.5)
    - Archaic/divine/noble Japanese markers (+2.0)
    """
    score = 0.0

    # 1. Uncertainty flag from draft model
    if cue.uncertain:
        score += 4.0
    if cue.uncertain_reason and cue.uncertain_reason not in ("none", ""):
        score += 2.0

    # 2. English reference conflict
    if getattr(cue, "en_conflict", False):
        score += 3.0

    # 3. Linter violations
    if violations:
        for v in violations:
            if v.severity == "ERROR":
                score += 4.0
            elif v.severity == "WARNING":
                score += 1.5

    # 4. Long or multi-clause lines
    ja_len = len(cue.ja or "")
    en_len = len(cue.en or "")
    if ja_len > 35 or en_len > 70:
        score += 1.5

    # 5. Archaic, divine, or noble speech in Japanese
    if re.search(r"(妾|貴様|我|汝|主殿|閣下|陛下|卿|殿|お主)", cue.ja or ""):
        score += 2.0

    return round(score, 2)


def find_cues_needing_repair(
    cues: List[Cue],
    glossary: Optional[Glossary] = None,
    difficulty_threshold: float = 4.0,
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
        score = score_cue_difficulty(cue, violations=violations)

        issues: List[str] = []
        if cue.uncertain:
            issues.append(f"draft_uncertain: {cue.uncertain_reason or 'yes'}")
        if getattr(cue, "en_conflict", False):
            issues.append("en_reference_conflict")
        for v in violations:
            issues.append(f"{v.rule_id} ({v.severity}): {v.message}")

        # If score >= threshold or has ERROR violation or uncertain
        has_error = any(v.severity == "ERROR" for v in violations)
        if score >= difficulty_threshold or cue.uncertain or has_error:
            candidates.append((cue, score, issues))

    return candidates


def run_cascade_repair_pass(
    cues: List[Cue],
    repair_translator: GeminiTranslator,
    glossary: Optional[Glossary] = None,
    difficulty_threshold: float = 4.0,
) -> Dict[str, Any]:
    """
    Runs the Cloud Cascade Repair/Editor Pass on cues needing review.
    Uses the provided high-tier repair_translator (e.g. gemini-3.8-flash).
    """
    candidates = find_cues_needing_repair(cues, glossary=glossary, difficulty_threshold=difficulty_threshold)
    if not candidates:
        logger.info("✨ [CASCADE REPAIR] Không có cue nào cần chỉnh sửa (tất cả đều đạt chuẩn).")
        return {
            "cues_evaluated": len(cues),
            "repaired_count": 0,
            "repair_ratio_pct": 0.0,
        }

    cues_to_fix = [item[0] for item in candidates]
    logger.info(
        f"🛠️ [CASCADE REPAIR] Gom {len(cues_to_fix)}/{len(cues)} cues khó ({len(cues_to_fix)/len(cues)*100:.1f}%) "
        f"để biên tập chất lượng cao qua model '{repair_translator.model}'..."
    )

    # Dry-run check
    if repair_translator.dry_run:
        for c in cues_to_fix:
            c.uncertain = False
            c.model_used = f"{repair_translator.model} (dry-run repair)"
        return {
            "cues_evaluated": len(cues),
            "repaired_count": len(cues_to_fix),
            "repair_ratio_pct": round(len(cues_to_fix) / max(1, len(cues)) * 100, 2),
        }

    # Prepare repair payload
    repair_items = []
    for cue, score, issues in candidates:
        item = {
            "id": cue.id,
            "ja": cue.ja,
            "en": cue.en,
            "draft_vi": cue.vi,
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

    # Call repair API
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
        for c in cues_to_fix:
            if c.id in id_to_fixed and id_to_fixed[c.id].strip():
                c.vi = id_to_fixed[c.id].strip()
                c.uncertain = False
                c.uncertain_reason = "repaired"
                c.model_used = f"{repair_translator.model} (repair)"
                repaired_count += 1

        logger.info(f"✅ [CASCADE REPAIR] Hoàn tất biên tập {repaired_count} cues khó.")
        return {
            "cues_evaluated": len(cues),
            "repaired_count": repaired_count,
            "repair_ratio_pct": round(repaired_count / max(1, len(cues)) * 100, 2),
        }
    except Exception as e:
        logger.warning(f"⚠️ [CASCADE REPAIR] Thất bại trong lượt repair ({e}), giữ nguyên bản draft.")
        return {
            "cues_evaluated": len(cues),
            "repaired_count": 0,
            "repair_ratio_pct": 0.0,
            "error": str(e),
        }
