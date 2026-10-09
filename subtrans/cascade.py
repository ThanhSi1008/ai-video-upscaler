# -*- coding: utf-8 -*-
"""
subtrans/cascade.py
Stage P2: Smart Cloud Cascade (100% Cloud API: Flash-Lite -> Flash).
- Difficulty scoring function S(cue) in [0.0, 15.0] on post-auto-fixed cues
- Two-tier pipeline:
  1. Draft Pass: high-throughput, low-quota Gemini Flash-Lite (2,000 RPD)
  2. Repair/Editor Pass: high-tier Gemini Flash on difficult/uncertain cues (strictly capped <= 20% of episode, max 2 requests)
- Guardrails:
  + Cổng Re-lint & Accept/Reject: So sánh danh tính vi phạm _ident, chốt chặn trôi dạt (edit distance > 0.5, số, metadata disagreement)
  + Fail-Open: Tự động giữ nguyên bản draft Lite khi gặp 429, timeout, bad JSON schema
  + Đầy đủ telemetry per-cue JSONL: Phân phối S(cue), edit rate, rejection rate, provenance reason_code
"""

import re
import json
import logging
import difflib
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field

from subtrans.models import Cue, Glossary
from subtrans.linter import lint_cue, LintViolation, CRITICAL_RULES, ALL_RULE_IDS, RULE_HIGH_CPS
from subtrans.gemini_backend import GeminiTranslator, filter_relevant_glossary

logger = logging.getLogger("subtrans.cascade")

MAX_REPAIR_PERCENT = 0.20
MAX_CUES_PER_REQUEST = 25
MAX_FLASH_REQUESTS = 2


def normalized_char_edit_distance(s1: str, s2: str) -> float:
    """Calculates normalized Levenshtein-like distance in [0.0, 1.0] using difflib.SequenceMatcher."""
    if s1 == s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
    return round(1.0 - ratio, 3)


def _ident(v: LintViolation) -> Tuple[str, Any]:
    """Generates identity tuple for lint violation to track exact occurrences instead of raw counts."""
    return (v.rule_id, getattr(v, "span", None) or v.message)


class RepairOutputItem(BaseModel):
    id: int
    vi: str
    changed: bool = False
    change_reason: Optional[str] = "none"
    speaker: Optional[str] = None
    addressee: Optional[str] = None
    mode: Optional[str] = None


REPAIR_SYSTEM_INSTRUCTION = (
    "You post-edit Vietnamese anime subtitles. Edit ONLY the listed cues and ONLY to fix "
    "their listed issues; if a draft is acceptable, return it unchanged. JA is the source of truth; "
    "EN only resolves omitted subjects/objects. Keep each speaker→addressee pronoun pair exactly as in "
    "<cast>. Use names exactly as in <glossary>. Do not add, drop or reorder information; keep numbers. "
    "Keep length within ±30% of the draft unless the issue is 'too_long'."
)


def score_cue_difficulty(
    cue: Cue,
    violations: Optional[List[LintViolation]] = None,
    series: Optional[str] = None,
    difficulty_markers: Optional[List[str]] = None,
) -> float:
    """
    Computes difficulty score S(cue) for a subtitle cue in [0.0, 15.0] on post-auto-fixed violations.
    Primary calibrated weights (evidence-backed):
    - Linter ERROR: +4.0 (highest priority quality defect)
    - LLM uncertain flag: +4.0 (explicit draft uncertainty)
    - LLM uncertain specific reason: +2.0
    
    Logging features (weight 0.0 until regression weights calibrated from logs):
    - Linter WARNING: 0.0 (tracked in log)
    - EN reference conflict: 0.0 (tracked in log)
    - Dialogue length (>35 JA or >70 EN chars): 0.0 (tracked in log)
    - Series-specific difficulty markers: +2.0 (only if explicitly matched in custom difficulty_markers list)
    
    Returns: final_score (float) and populates cue.difficulty_score & cue.difficulty_features
    """
    score = 0.0
    features: Dict[str, Any] = {
        "linter_errors": 0,
        "linter_warnings": 0,
        "uncertain": False,
        "uncertain_reason": "none",
        "en_conflict": False,
        "long_line": False,
        "marker_matched": False,
    }

    if violations:
        for v in violations:
            if v.severity == "ERROR":
                score += 4.0
                features["linter_errors"] += 1
            elif v.severity == "WARNING":
                features["linter_warnings"] += 1

    if cue.uncertain:
        score += 4.0
        features["uncertain"] = True
    if cue.uncertain_reason and cue.uncertain_reason.lower() not in ("none", "", "no"):
        score += 2.0
        features["uncertain_reason"] = cue.uncertain_reason

    if getattr(cue, "en_conflict", False):
        features["en_conflict"] = True

    ja_len = len(cue.ja or "")
    en_len = len(cue.en or "")
    if ja_len > 35 or en_len > 70:
        features["long_line"] = True

    # Series-specific difficulty markers from YAML config (NOT hardcoded, no lone 殿)
    ja_text = cue.ja or ""
    if difficulty_markers:
        for m in difficulty_markers:
            if m and m in ja_text:
                score += 2.0
                features["marker_matched"] = True
                features["matched_marker"] = m
                break

    final_score = round(min(15.0, score), 2)
    cue.difficulty_score = final_score
    cue.difficulty_features = features
    return final_score


def find_cues_needing_repair(
    cues: List[Cue],
    glossary: Optional[Glossary] = None,
    difficulty_threshold: float = 4.0,
    series: Optional[str] = None,
    difficulty_markers: Optional[List[str]] = None,
) -> List[Tuple[Cue, float, List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Filters cues that require high-tier editor review.
    Returns list of (cue, score, structured_issues, features).
    """
    candidates: List[Tuple[Cue, float, List[Dict[str, Any]], Dict[str, Any]]] = []

    # Use difficulty markers from glossary if not passed directly
    if difficulty_markers is None and glossary:
        difficulty_markers = glossary.difficulty_markers

    for cue in cues:
        if getattr(cue, "is_dropped_sound", False) or getattr(cue, "is_skipped", False):
            continue
        # Skip untranslated cues (cannot review non-existent draft)
        if getattr(cue, "backend_used", None) == "untranslated" or (cue.vi or "").strip() in ["[Chưa dịch]", "(Chưa dịch)", "[Untranslated]", ""]:
            continue

        violations = lint_cue(cue, glossary=glossary)
        score = score_cue_difficulty(
            cue,
            violations=violations,
            series=series,
            difficulty_markers=difficulty_markers
        )
        features = cue.difficulty_features

        structured_issues: List[Dict[str, Any]] = []
        if cue.uncertain:
            structured_issues.append({
                "code": "draft_uncertain",
                "message": cue.uncertain_reason or "uncertainty flagged by draft model",
                "expected": "verify speaker/pronoun against context",
            })
        if getattr(cue, "en_conflict", False):
            structured_issues.append({
                "code": "en_reference_conflict",
                "message": "Japanese text and English reference conflict",
                "expected": "follow Japanese dialogue as source of truth",
            })
        for v in violations:
            structured_issues.append({
                "code": v.rule_id,
                "message": v.message,
                "expected": v.suggestion or "adhere to subtitling rule",
            })

        has_error = any(v.severity == "ERROR" for v in violations)
        if score >= difficulty_threshold or has_error:
            candidates.append((cue, score, structured_issues, features))

    return candidates


def accept_or_reject_repaired_cue(
    cue: Cue,
    fixed_vi: str,
    glossary: Optional[Glossary] = None,
) -> Tuple[bool, str, List[LintViolation]]:
    """
    Guardrail: Re-lints the cue after Flash repair pass and decides to accept or reject.
    - Identity comparison: tracks exact rule_id + span/message occurrences
    - Shared base: identical baseline for comparison and linting
    - Structured CPS check via is_hard_ceiling
    - Semantic drift guards: edit distance > 0.5, preserved numbers/digits
    Returns: (is_accepted, reason_code, new_violations)
    """
    clean_fixed = fixed_vi.strip()
    if not clean_fixed:
        return False, "rejected_empty", []

    base = (cue.draft_vi or cue.vi or "").strip()
    if clean_fixed == base:
        return True, "accepted_identical", []

    orig = {_ident(v): v for v in lint_cue(cue.model_copy(update={"vi": base}), glossary=glossary)}
    new = {_ident(v): v for v in lint_cue(cue.model_copy(update={"vi": clean_fixed}), glossary=glossary)}
    introduced = [v for k, v in new.items() if k not in orig]

    # 1. Critical violations check (from linter.CRITICAL_RULES)
    if any(v.rule_id in CRITICAL_RULES for v in introduced):
        return False, "rejected_critical_violation", list(new.values())

    # 2. Structured extreme CPS overflow check (> 25 CPS ceiling)
    if any(v.is_hard_ceiling for v in introduced):
        return False, "rejected_cps_overflow", list(new.values())

    # 3. New error violations check
    if any(v.severity == "ERROR" for v in introduced):
        return False, "rejected_new_error", list(new.values())

    # 4. Semantic drift guards
    # (a) Large edit distance (> 0.5) -> accepted but flagged for human review queue
    edit_dist = normalized_char_edit_distance(base, clean_fixed)
    if edit_dist > 0.5:
        cue.is_large_edit = True
        return True, "accepted_large_edit", list(new.values())

    # (b) Number / digit check
    base_numbers = re.findall(r"\d+", base)
    fixed_numbers = re.findall(r"\d+", clean_fixed)
    if base_numbers and base_numbers != fixed_numbers:
        cue.is_digit_mismatch = True
        return True, "accepted_digit_mismatch", list(new.values())

    # 5. Accepted improved / valid
    orig_errors = sum(1 for v in orig.values() if v.severity == "ERROR")
    new_errors = sum(1 for v in new.values() if v.severity == "ERROR")
    if new_errors < orig_errors or len(new) < len(orig):
        return True, "accepted_lint_clean", list(new.values())

    return True, "accepted_valid", list(new.values())


def run_cascade_repair_pass(
    cues: List[Cue],
    repair_translator: GeminiTranslator,
    glossary: Optional[Glossary] = None,
    difficulty_threshold: float = 4.0,
    series: Optional[str] = None,
    episode_basename: str = "",
) -> Dict[str, Any]:
    """
    Runs the Cloud Cascade Repair/Editor Pass on cues needing review.
    Uses high-tier repair_translator (gemini-3.8-flash).
    Features:
    - Evidence-backed S(cue) and Top-K budget cap (<= 20% of episode, max 2 requests)
    - Affirmative dynamic cast/glossary in prompt
    - Symmetrical surrounding context (prev_vi, next_vi)
    - Re-lint accept/reject gate with identity checking and drift guards
    - Per-cue telemetry logging in JSONL
    - Fail-open resilience
    """
    all_candidates = find_cues_needing_repair(
        cues,
        glossary=glossary,
        difficulty_threshold=difficulty_threshold,
        series=series,
        difficulty_markers=glossary.difficulty_markers if glossary else None
    )

    if not all_candidates:
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

    # Top-K Budget Capping (Preserving 20 RPD Flash limit)
    total_cues_count = max(1, len(cues))
    max_budget_cues = min(
        max(1, int(total_cues_count * MAX_REPAIR_PERCENT)),
        MAX_CUES_PER_REQUEST * MAX_FLASH_REQUESTS
    )

    # Sort descending by (has_error, score)
    def _candidate_rank(item):
        cue, score, issues, _ = item
        has_err = any(iss.get("code") in ("LINT_CJK_LEAK", "LINT_BANNED_TERM") or "ERROR" in iss.get("message", "") for iss in issues)
        return (1 if has_err else 0, score)

    all_candidates.sort(key=_candidate_rank, reverse=True)
    budgeted_candidates = all_candidates[:max_budget_cues]

    scores = [item[1] for item in budgeted_candidates]
    s_dist = {
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "max": float(np.max(scores)),
    }

    logger.info(
        f"🛠️ [CASCADE REPAIR] Gom {len(budgeted_candidates)}/{len(cues)} cues khó "
        f"({len(budgeted_candidates)/len(cues)*100:.1f}%, ngân sách tối đa: {max_budget_cues}) "
        f"[S p50={s_dist['p50']:.1f}, p90={s_dist['p90']:.1f}, max={s_dist['max']:.1f}] "
        f"để biên tập qua '{repair_translator.model}'..."
    )

    cues_to_fix = [item[0] for item in budgeted_candidates]
    for c in cues_to_fix:
        if not c.draft_vi:
            c.draft_vi = c.vi

    # Telemetry records
    telemetry_records: List[Dict[str, Any]] = []

    # Dry-run check
    if getattr(repair_translator, "dry_run", False):
        for c, score, issues, features in budgeted_candidates:
            c.uncertain = False
            c.model_used = f"{repair_translator.model} (dry-run repair)"
            c.repair_reason_code = "dry_run_accepted"
            telemetry_records.append({
                "cue_id": c.id,
                "s_score": score,
                "s_features": features,
                "was_repaired": True,
                "flash_changed": False,
                "reason_code": "dry_run_accepted",
                "metadata_disagreement": False,
                "orig_vi": c.draft_vi,
                "fixed_vi": c.vi,
                "issues": issues,
            })
        _write_telemetry_jsonl(episode_basename, telemetry_records)
        return {
            "cues_evaluated": len(cues),
            "candidates_count": len(budgeted_candidates),
            "repaired_count": len(budgeted_candidates),
            "rejected_count": 0,
            "edited_count": 0,
            "repair_ratio_pct": round(len(budgeted_candidates) / max(1, len(cues)) * 100, 2),
            "s_distribution": s_dist,
        }

    # Map cues to index for surrounding context lookup
    cue_idx_map = {c.id: idx for idx, c in enumerate(cues)}

    repaired_count = 0
    rejected_count = 0
    edited_count = 0

    last_fail_code: Optional[str] = None

    # Process in chunks of MAX_CUES_PER_REQUEST
    chunk_size = MAX_CUES_PER_REQUEST
    for chunk_start in range(0, len(budgeted_candidates), chunk_size):
        chunk = budgeted_candidates[chunk_start:chunk_start + chunk_size]
        chunk_cues = [item[0] for item in chunk]

        # Prepare repair items with context and structured issues
        repair_items = []
        for cue, score, issues, _ in chunk:
            idx = cue_idx_map.get(cue.id, -1)
            prev_cue = cues[idx - 1] if idx > 0 else None
            next_cue = cues[idx + 1] if 0 <= idx < len(cues) - 1 else None

            item = {
                "id": cue.id,
                "ja": cue.ja,
                "en": cue.en,
                "draft_vi": cue.draft_vi or cue.vi,
                "prev_final_vi": (prev_cue.vi if prev_cue else "") or "",
                "next_final_vi": (next_cue.vi if next_cue else "") or "",
                "speaker": cue.speaker or "unknown",
                "addressee": cue.addressee,
                "mode": cue.mode or "spoken",
                "issues": issues,
            }
            repair_items.append(item)

        # Dynamic affirmative cast & glossary blocks
        rel_glossary = filter_relevant_glossary(chunk_cues, glossary)
        glossary_block = ""
        if rel_glossary:
            gloss_lines = [f"- {term} → {trans}" for term, trans in rel_glossary.items()]
            glossary_block = "<glossary>\n" + "\n".join(gloss_lines) + "\n</glossary>\n\n"

        cast_block = ""
        if glossary and glossary.characters:
            hit_chars = [c for c in glossary.characters if any(c.name_ja in (cue.ja or "") for cue in chunk_cues)]
            if hit_chars:
                cast_lines = [f"- {c.name_ja} → {c.name_vi}" for c in hit_chars]
                cast_block = "<cast>\n" + "\n".join(cast_lines) + "\n</cast>\n\n"

        prompt_str = (
            f"{cast_block}{glossary_block}"
            f"<cues>\n{json.dumps({'cues_to_review': repair_items}, ensure_ascii=False, indent=2)}\n</cues>"
        )

        try:
            from google.genai import types
            from subtrans.cache import execute_and_log_api_request

            config_kwargs = {
                "system_instruction": REPAIR_SYSTEM_INSTRUCTION,
                "response_mime_type": "application/json",
                "response_schema": list[RepairOutputItem],
                "temperature": 0.0,
            }

            def _send():
                return repair_translator.client.models.generate_content(
                    model=repair_translator.model,
                    contents=prompt_str,
                    config=types.GenerateContentConfig(**config_kwargs),
                )

            curr_kid = repair_translator.key_ids[repair_translator.current_key_idx] if repair_translator.key_ids else "none"
            response = execute_and_log_api_request(
                _send,
                model=repair_translator.model,
                backend="gemini",
                cues_count=len(chunk_cues),
                request_type="cascade_repair",
                cache=repair_translator.cache,
                key_id=curr_kid,
            )

            raw_text = response.text.strip()
            data = json.loads(raw_text)
            id_to_fixed = {it["id"]: it for it in data if "id" in it and "vi" in it}

            for cue, score, issues, features in chunk:
                if cue.id in id_to_fixed:
                    resp_item = id_to_fixed[cue.id]
                    fixed_vi = str(resp_item.get("vi", "")).strip()
                    resp_changed = resp_item.get("changed", False) or (fixed_vi != cue.draft_vi)

                    if fixed_vi != cue.draft_vi:
                        edited_count += 1

                    # Check metadata disagreement
                    it_speaker = resp_item.get("speaker")
                    it_addressee = resp_item.get("addressee")
                    it_mode = resp_item.get("mode")
                    disagreement = False
                    if (it_speaker and cue.speaker and it_speaker.strip().lower() != cue.speaker.strip().lower()) or \
                       (it_addressee and cue.addressee and it_addressee.strip().lower() != cue.addressee.strip().lower()) or \
                       (it_mode and cue.mode and it_mode.strip().lower() != cue.mode.strip().lower()):
                        disagreement = True
                    cue.metadata_disagreement = disagreement

                    # Re-lint Accept / Reject Gate
                    accepted, reason_code, _ = accept_or_reject_repaired_cue(cue, fixed_vi, glossary=glossary)
                    cue.repair_reason_code = reason_code

                    if accepted:
                        cue.vi = fixed_vi
                        cue.uncertain = False
                        cue.uncertain_reason = "repaired"
                        cue.model_used = f"{repair_translator.model} (repair)"
                        repaired_count += 1
                    else:
                        cue.vi = cue.draft_vi
                        rejected_count += 1
                        logger.info(f"🛡️ [CASCADE REJECT] Từ chối bản sửa cue #{cue.id} ({reason_code}), giữ draft: '{cue.draft_vi}'")

                    telemetry_records.append({
                        "cue_id": cue.id,
                        "s_score": score,
                        "s_features": features,
                        "was_repaired": accepted,
                        "flash_changed": resp_changed,
                        "reason_code": reason_code,
                        "metadata_disagreement": disagreement,
                        "orig_vi": cue.draft_vi,
                        "fixed_vi": fixed_vi if accepted else cue.draft_vi,
                        "issues": issues,
                    })
                else:
                    cue.repair_reason_code = "unrepaired_missing_id"
                    telemetry_records.append({
                        "cue_id": cue.id,
                        "s_score": score,
                        "s_features": features,
                        "was_repaired": False,
                        "flash_changed": False,
                        "reason_code": "unrepaired_missing_id",
                        "metadata_disagreement": False,
                        "orig_vi": cue.draft_vi,
                        "fixed_vi": cue.draft_vi,
                        "issues": issues,
                    })

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

            last_fail_code = fail_code
            logger.warning(f"⚠️ [CASCADE FAIL-OPEN] Lượt repair thất bại ({e}). Tự động giữ nguyên bản draft Lite ({fail_code}).")
            for cue, score, issues, features in chunk:
                cue.vi = cue.draft_vi
                cue.repair_reason_code = fail_code
                telemetry_records.append({
                    "cue_id": cue.id,
                    "s_score": score,
                    "s_features": features,
                    "was_repaired": False,
                    "flash_changed": False,
                    "reason_code": fail_code,
                    "metadata_disagreement": False,
                    "orig_vi": cue.draft_vi,
                    "fixed_vi": cue.draft_vi,
                    "issues": issues,
                    "high_priority_review": score >= 4.0,
                })

    edit_rate = round(edited_count / max(1, len(budgeted_candidates)) * 100, 1)
    rejection_rate = round(rejected_count / max(1, len(budgeted_candidates)) * 100, 1)
    logger.info(
        f"✅ [CASCADE REPAIR] Hoàn tất: nhận {repaired_count}/{len(budgeted_candidates)} cues "
        f"(Flash sửa: {edited_count} [{edit_rate}%], từ chối: {rejected_count} [{rejection_rate}%])."
    )

    _write_telemetry_jsonl(episode_basename, telemetry_records)

    ret_stats = {
        "cues_evaluated": len(cues),
        "candidates_count": len(budgeted_candidates),
        "repaired_count": repaired_count,
        "rejected_count": rejected_count,
        "edited_count": edited_count,
        "repair_ratio_pct": round(repaired_count / max(1, len(cues)) * 100, 2),
        "s_distribution": s_dist,
    }
    if last_fail_code:
        ret_stats["fail_open_code"] = last_fail_code
    return ret_stats


def _write_telemetry_jsonl(episode_basename: str, records: List[Dict[str, Any]]) -> None:
    """Writes detailed per-cue telemetry JSONL for calibration and logistic regression training."""
    if not records:
        return
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    bname = episode_basename or "session"
    target_file = log_dir / f"cascade_telemetry_{bname}.jsonl"
    try:
        with open(target_file, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Could not write cascade telemetry JSONL: {e}")
