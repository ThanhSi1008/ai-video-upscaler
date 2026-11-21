"""Two-tier cloud translation cascade with difficulty scoring and digit quarantine."""

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
from subtrans.gemini_backend import (
    GeminiTranslator,
    build_episode_system_prompt,
    filter_relevant_glossary,
    get_anime_safety_settings,
    extract_genai_response_text,
    get_thinking_config,
)
from subtrans.ja_morph_tagger import strip_ja_furigana, extract_ja_linguistic_hints

logger = logging.getLogger("subtrans.cascade")

MAX_REPAIR_PERCENT = 0.20
MAX_CUES_PER_REQUEST = 15
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


def build_repair_system_prompt(glossary: Optional[Glossary] = None) -> str:
    """
    Xây dựng System Instruction tĩnh (> 4,096 tokens) cho lượt sửa Cascade bằng cách kết hợp
    REPAIR_SYSTEM_INSTRUCTION với cẩm nang dịch thuật & bảng nhân vật tĩnh của tập, kích hoạt
    Gemini Implicit Context Caching giảm 75-90% chi phí input token ở cả Stage 2c.
    """
    base_episode_prompt = build_episode_system_prompt(glossary=glossary)
    return f"{REPAIR_SYSTEM_INSTRUCTION}\n\n{base_episode_prompt}"


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
                score += 1.5
                features["linter_warnings"] += 1

    if cue.uncertain:
        score += 4.0
        features["uncertain"] = True
    if cue.uncertain_reason and cue.uncertain_reason.lower() not in ("none", "", "no"):
        score += 2.0
        features["uncertain_reason"] = cue.uncertain_reason

    if getattr(cue, "en_conflict", False):
        score += 4.0
        features["en_conflict"] = True

    ja_len = len(cue.ja or "")
    en_len = len(cue.en or "")
    if ja_len > 35 or en_len > 70:
        features["long_line"] = True

    # Series-specific difficulty markers from YAML config (furigana-transparent)
    ja_text = cue.ja or ""
    ja_clean = strip_ja_furigana(ja_text)
    if difficulty_markers:
        for m in difficulty_markers:
            if m and (m in ja_text or m in ja_clean):
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
        base_violations = lint_cue(cue.model_copy(update={"vi": clean_fixed}), glossary=glossary)
        return True, "accepted_identical", base_violations

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
    # (a) Number / digit check (evaluated first so digit drift is never masked by large edits)
    base_numbers = re.findall(r"\d+", base)
    fixed_numbers = re.findall(r"\d+", clean_fixed)
    if base_numbers and base_numbers != fixed_numbers:
        cue.is_digit_mismatch = True
        return True, "accepted_digit_mismatch", list(new.values())

    # (b) Large edit distance (> 0.5) -> accepted but flagged for human review queue
    edit_dist = normalized_char_edit_distance(base, clean_fixed)
    if edit_dist > 0.5:
        cue.is_large_edit = True
        return True, "accepted_large_edit", list(new.values())

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
    dynamic_cast_context: Optional[str] = None,
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
            ja_hints = extract_ja_linguistic_hints(cue.ja) if cue.ja else None
            if ja_hints:
                item["ja_hints"] = ja_hints
            if cue.speaker and cue.speaker.strip().lower() != "unknown":
                item["speaker_cc"] = strip_ja_furigana(cue.speaker.strip())
            if getattr(cue, "p1", None):
                item["p1"] = cue.p1
            if getattr(cue, "p2", None):
                item["p2"] = cue.p2
            if getattr(cue, "mode_hint", None):
                item["mode_hint"] = cue.mode_hint
            repair_items.append(item)

        # Dynamic affirmative cast & glossary blocks
        rel_glossary = filter_relevant_glossary(chunk_cues, glossary)
        glossary_block = ""
        if rel_glossary and rel_glossary.get("terms"):
            gloss_lines = []
            for t in rel_glossary["terms"]:
                line = f"- {t.get('ja', '')} → {t.get('vi', '')}"
                extras = []
                if t.get("alt"):
                    alt_val = ", ".join(t["alt"]) if isinstance(t["alt"], list) else str(t["alt"])
                    extras.append(f"alt: {alt_val}")
                if t.get("note"):
                    extras.append(f"note: {t['note']}")
                if extras:
                    line += f" ({'; '.join(extras)})"
                gloss_lines.append(line)
            glossary_block = "<glossary>\n" + "\n".join(gloss_lines) + "\n</glossary>\n\n"

        cast_block = ""
        if glossary and glossary.characters:
            cast_lines = []
            for c in glossary.characters:
                meta_parts = []
                if getattr(c, "gender", None):
                    meta_parts.append(str(c.gender))
                if getattr(c, "speaking_style", None):
                    meta_parts.append(str(c.speaking_style))
                if getattr(c, "note", None):
                    meta_parts.append(str(c.note))
                meta_str = f" [{', '.join(meta_parts)}]" if meta_parts else ""
                speaks_str = ""
                if getattr(c, "speaks_to", None):
                    pairs = [f"{s.self}/{s.other} -> {s.listener}" for s in c.speaks_to]
                    speaks_str = f" | speaks_to: {'; '.join(pairs)}"
                cast_lines.append(f"- {c.name_ja} → {c.name_vi}{meta_str}{speaks_str}")
            if cast_lines:
                cast_block = "<cast>\n" + "\n".join(cast_lines) + "\n</cast>\n\n"

        dyn_cast_block = (
            f"<dynamic_cast>\n{dynamic_cast_context.strip()}\n</dynamic_cast>\n\n"
            if dynamic_cast_context and dynamic_cast_context.strip()
            else ""
        )
        addr_block = ""
        if glossary and glossary.address_rules and glossary.address_rules.relations:
            rel_lines = [
                f"- {r.case}: {r.self} / {r.other}" + (f" ({r.note})" if r.note else "")
                for r in glossary.address_rules.relations
            ]
            addr_block = "<address_rules>\n" + "\n".join(rel_lines) + "\n</address_rules>\n\n"

        prompt_str = (
            f"{cast_block}{dyn_cast_block}{addr_block}{glossary_block}"
            f"<cues>\n{json.dumps({'cues_to_review': repair_items}, ensure_ascii=False, separators=(',', ':'))}\n</cues>"
        )

        try:
            import time as _time
            from google.genai import types
            from subtrans.cache import execute_and_log_api_request
            from subtrans.throttle import (
                call_with_retry,
                DailyQuotaExhaustedError,
                MinuteRateLimitError,
                QuotaExhaustedError,
            )

            config_kwargs: Dict[str, Any] = {
                "system_instruction": build_repair_system_prompt(glossary),
                "response_mime_type": "application/json",
                "response_schema": list[RepairOutputItem],
                "safety_settings": get_anime_safety_settings(),
                "temperature": 0.0,
                "max_output_tokens": 16384,
            }
            if getattr(repair_translator, "http_options", None) is not None:
                config_kwargs["http_options"] = repair_translator.http_options
            thinking_cfg = get_thinking_config(repair_translator.model)
            if thinking_cfg is not None:
                config_kwargs["thinking_config"] = thinking_cfg

            api_keys_list = getattr(repair_translator, "api_keys", []) or []
            repair_cache = getattr(repair_translator, "cache", None)
            rate_limiters = getattr(repair_translator, "rate_limiters", {}) or {}
            rpm_attempts = 0
            max_rpm_attempts = max(4, len(api_keys_list) * 2)

            while True:
                curr_kid = (
                    repair_translator.key_ids[repair_translator.current_key_idx]
                    if getattr(repair_translator, "key_ids", None)
                    else "none"
                )
                if len(api_keys_list) > 1 and repair_cache and repair_cache.is_key_model_exhausted(curr_kid, repair_translator.model):
                    if not repair_translator.rotate_key(reason="Cascade pre-check PerDay exhausted"):
                        raise DailyQuotaExhaustedError(
                            model=repair_translator.model,
                            quota_id="AllKeysExhausted",
                            quota_value="",
                            reset_pt_str="",
                            reset_vn_str="",
                        )
                    curr_kid = repair_translator.key_ids[repair_translator.current_key_idx]

                curr_limiter = rate_limiters.get(curr_kid)
                if (
                    not getattr(repair_translator, "explicit_caching", False)
                    and len(api_keys_list) > 1
                    and curr_limiter is not None
                    and getattr(curr_limiter, "max_rpm", 60) <= 5
                    and curr_limiter.available_slots() == 0
                    and hasattr(repair_translator, "select_next_round_robin_key")
                ):
                    has_free_other = any(
                        idx != getattr(repair_translator, "current_key_idx", 0)
                        and not (repair_cache and repair_cache.is_key_model_exhausted(kid, repair_translator.model))
                        and (rate_limiters.get(kid) is None or rate_limiters[kid].available_slots() > 0)
                        for idx, kid in enumerate(getattr(repair_translator, "key_ids", []))
                    )
                    if has_free_other and repair_translator.select_next_round_robin_key(reason="Cascade low-RPM slot overflow"):
                        curr_kid = repair_translator.key_ids[repair_translator.current_key_idx]
                        curr_limiter = rate_limiters.get(curr_kid)
                if curr_limiter is not None:
                    curr_limiter.wait_if_needed()

                def _single_cascade_call():
                    return execute_and_log_api_request(
                        lambda: repair_translator.client.models.generate_content(
                            model=repair_translator.model,
                            contents=prompt_str,
                            config=types.GenerateContentConfig(**config_kwargs),
                        ),
                        model=repair_translator.model,
                        backend="gemini",
                        cues_count=len(chunk_cues),
                        request_type="cascade_repair",
                        cache=repair_cache,
                        key_id=curr_kid,
                    )

                try:
                    response = call_with_retry(
                        fn=_single_cascade_call,
                        model_name=repair_translator.model,
                        max_retries=2,
                        cache=repair_cache,
                        key_id=curr_kid,
                        all_key_ids=getattr(repair_translator, "key_ids", None),
                        raise_on_per_minute=(len(api_keys_list) > 1),
                    )
                    if getattr(repair_translator, "round_robin", False) and len(api_keys_list) > 1:
                        repair_translator.select_next_round_robin_key(reason="Cascade repair round-robin")
                    break
                except MinuteRateLimitError as mre:
                    if rpm_attempts >= max_rpm_attempts:
                        raise QuotaExhaustedError(
                            f"Cascade 429 PerMinute retry limit ({max_rpm_attempts}) exceeded on '{repair_translator.model}'"
                        ) from mre
                    rpm_attempts += 1
                    wait_sec = min(float(mre.retry_delay or 10.0), 60.0)
                    if curr_kid in rate_limiters:
                        rate_limiters[curr_kid].record_cooldown(wait_sec)
                    prev_idx = getattr(repair_translator, "current_key_idx", 0)
                    if (
                        len(api_keys_list) > 1
                        and hasattr(repair_translator, "select_next_round_robin_key")
                        and repair_translator.select_next_round_robin_key(reason=f"Cascade 429 PerMinute ({mre.quota_id})")
                        and getattr(repair_translator, "current_key_idx", 0) != prev_idx
                    ):
                        next_kid = repair_translator.key_ids[repair_translator.current_key_idx]
                        next_limiter = rate_limiters.get(next_kid)
                        if not next_limiter or next_limiter.available_slots() > 0:
                            continue
                    logger.warning(f"⏳ [CASCADE 429 PER-MINUTE] Waiting {wait_sec:.1f}s before retrying...")
                    _time.sleep(wait_sec)
                    active_kid = (
                        repair_translator.key_ids[repair_translator.current_key_idx]
                        if getattr(repair_translator, "key_ids", None)
                        else "none"
                    )
                    if active_kid in rate_limiters:
                        rate_limiters[active_kid].clear_cooldown()
                    continue
                except DailyQuotaExhaustedError as dqe:
                    if len(api_keys_list) > 1 and repair_cache:
                        repair_cache.mark_key_model_exhausted(
                            key_id=curr_kid,
                            model=repair_translator.model,
                            quota_id=dqe.quota_id or "PerDay",
                            quota_value=dqe.quota_value or "",
                            reason=f"Cascade 429 PerDay: {dqe.quota_id}",
                        )
                        if repair_translator.rotate_key(reason=f"Cascade 429 PerDay ({dqe.quota_id})"):
                            continue
                    raise

            raw_text = extract_genai_response_text(response)
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
                    accepted, reason_code, repaired_violations = accept_or_reject_repaired_cue(
                        cue, fixed_vi, glossary=glossary
                    )
                    cue.repair_reason_code = reason_code

                    if accepted:
                        source_digits = set(
                            re.findall(r"\d+", f"{cue.ja or ''} {cue.en or ''} {cue.draft_vi or ''}")
                        )
                        fixed_digits = set(re.findall(r"\d+", fixed_vi))
                        hallucinated_digits = bool(fixed_digits - source_digits)
                        if hallucinated_digits:
                            cue.is_digit_mismatch = True

                        cue.vi = fixed_vi
                        has_remaining_errors = any(
                            getattr(v, "severity", "") == "ERROR" for v in (repaired_violations or [])
                        )
                        if reason_code == "accepted_digit_mismatch" or hallucinated_digits:
                            cue.uncertain = True
                            cue.uncertain_reason = "digit_mismatch"
                        elif has_remaining_errors:
                            cue.uncertain = True
                            if not cue.uncertain_reason or cue.uncertain_reason == "none":
                                cue.uncertain_reason = "lint_error"
                        else:
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
    import os
    if not records or os.environ.get("PYTEST_CURRENT_TEST"):
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
