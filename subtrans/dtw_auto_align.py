"""Anchor-guided monotonic DTW alignment between Whisper JA cues and studio EN timing.

Supports 1:1 direct snap, 1:2 proportional pause-aware splitting, 2:1 clause-level
EN splitting, OP/ED >45s gap quarantine, and semantic Cast/punctuation anchor bonuses.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

from subtrans.ja_morph_tagger import strip_ja_furigana


@dataclass
class SubCue:
    start_ms: int
    end_ms: int
    text: str


def estimate_ja_mora(text: str) -> int:
    """Estimate Japanese mora count for dialogue text."""
    clean = re.sub(r"[\s\W_]+", "", text)
    return max(1, len(clean))


def _compute_monotonic_dtw_backpointers(
    ja_cues: List[SubCue],
    en_cues: List[SubCue],
    cast_names_ja_en: List[Tuple[str, str]] = (),
    max_window_ms: int = 2500,
) -> List[Tuple[str, int, int]]:
    """Run windowed Sakoe-Chiba monotonic DP with 3 rolling rows and return traced ops."""
    N, M = len(ja_cues), len(en_cues)
    if N == 0 or M == 0:
        return []

    INF = 1e9
    SKIP_JA_PENALTY = 0.85
    SKIP_EN_PENALTY = 0.85
    SPLIT_MERGE_PENALTY = 0.12

    safe_win = max(1, max_window_ms)
    inv_2w = 1.0 / (2.0 * safe_win)

    cast_pairs: List[Tuple[str, str]] = []
    for ja_name, en_name in cast_names_ja_en:
        norm_ja = strip_ja_furigana(ja_name)
        if norm_ja:
            cast_pairs.append((norm_ja, en_name.lower()))

    ja_s = [c.start_ms for c in ja_cues]
    ja_e = [c.end_ms for c in ja_cues]
    ja_mora = [0] * N
    ja_q = [False] * N
    ja_ex = [False] * N
    ja_cmask = [0] * N
    for i, c in enumerate(ja_cues):
        txt = c.text
        jc = strip_ja_furigana(txt)
        ja_mora[i] = estimate_ja_mora(jc)
        ja_q[i] = ("？" in txt) or ("?" in txt)
        ja_ex[i] = ("！" in txt) or ("!" in txt)
        if cast_pairs:
            mask = 0
            for k, (jn, _) in enumerate(cast_pairs):
                if jn in jc:
                    mask |= 1 << k
            ja_cmask[i] = mask

    en_s = [c.start_ms for c in en_cues]
    en_e = [c.end_ms for c in en_cues]
    en_words = [0] * M
    en_q = [False] * M
    en_ex = [False] * M
    en_cmask = [0] * M
    for j, c in enumerate(en_cues):
        txt = c.text
        en_words[j] = estimate_en_words(txt)
        en_q[j] = "?" in txt
        en_ex[j] = "!" in txt
        if cast_pairs:
            el = txt.lower()
            mask = 0
            for k, (_, en_n) in enumerate(cast_pairs):
                if en_n in el:
                    mask |= 1 << k
            en_cmask[j] = mask

    ja_p_ok = [False] * max(0, N - 1)
    ja_p_mora = [0] * max(0, N - 1)
    ja_p_q = [False] * max(0, N - 1)
    ja_p_ex = [False] * max(0, N - 1)
    ja_p_cmask = [0] * max(0, N - 1)
    for i in range(N - 1):
        if (ja_s[i + 1] - ja_e[i]) <= 600:
            ja_p_ok[i] = True
            jc = strip_ja_furigana(ja_cues[i].text + " " + ja_cues[i + 1].text)
            ja_p_mora[i] = estimate_ja_mora(jc)
            ja_p_q[i] = ja_q[i] or ja_q[i + 1]
            ja_p_ex[i] = ja_ex[i] or ja_ex[i + 1]
            if cast_pairs:
                mask = 0
                for k, (jn, _) in enumerate(cast_pairs):
                    if jn in jc:
                        mask |= 1 << k
                ja_p_cmask[i] = mask

    en_p_ok = [False] * max(0, M - 1)
    en_p_words = [0] * max(0, M - 1)
    en_p_q = [False] * max(0, M - 1)
    en_p_ex = [False] * max(0, M - 1)
    en_p_cmask = [0] * max(0, M - 1)
    for j in range(M - 1):
        if (en_s[j + 1] - en_e[j]) <= 600:
            en_p_ok[j] = True
            comb_e = en_cues[j].text + " " + en_cues[j + 1].text
            en_p_words[j] = estimate_en_words(comb_e)
            en_p_q[j] = en_q[j] or en_q[j + 1]
            en_p_ex[j] = en_ex[j] or en_ex[j + 1]
            if cast_pairs:
                el = comb_e.lower()
                mask = 0
                for k, (_, en_n) in enumerate(cast_pairs):
                    if en_n in el:
                        mask |= 1 << k
                en_p_cmask[j] = mask

    # Ensure band width covers asymmetric length skew so (0, 0) -> (N, M) is reachable.
    band_w = max(40, abs(N - M) + 10)
    j_lows = [max(0, round(i * M / N) - band_w) for i in range(N + 1)]
    j_highs = [min(M, round(i * M / N) + band_w) for i in range(N + 1)]

    OP_NONE = 0
    OP_SKIP_JA = 1
    OP_SKIP_EN = 2
    OP_1_1 = 3
    OP_1_2 = 4
    OP_2_1 = 5

    back_rows = [bytearray(j_highs[i] - j_lows[i] + 1) for i in range(N + 1)]
    dp_curr = [INF] * (M + 1)
    dp_next1 = [INF] * (M + 1)
    dp_next2 = [INF] * (M + 1)
    dp_curr[0] = 0.0

    for i in range(N + 1):
        jl = j_lows[i]
        jh = j_highs[i]
        has_next1 = i < N
        has_next2 = i + 1 < N
        jl1 = j_lows[i + 1] if has_next1 else 0
        jh1 = j_highs[i + 1] if has_next1 else -1
        jl2 = j_lows[i + 2] if has_next2 else 0
        jh2 = j_highs[i + 2] if has_next2 else -1

        if has_next1:
            js1 = ja_s[i]
            je1 = ja_e[i]
            jm1 = ja_mora[i]
            jq1 = ja_q[i]
            jex1 = ja_ex[i]
            jcm1 = ja_cmask[i]
        if has_next2 and ja_p_ok[i]:
            jp_ok = True
            jp_s = ja_s[i]
            jp_e = ja_e[i + 1]
            jp_m = ja_p_mora[i]
            jp_q = ja_p_q[i]
            jp_ex = ja_p_ex[i]
            jp_cm = ja_p_cmask[i]
        else:
            jp_ok = False

        for j in range(jl, jh + 1):
            d_ij = dp_curr[j]
            if d_ij >= INF:
                continue

            if has_next1 and jl1 <= j <= jh1:
                cand = d_ij + SKIP_JA_PENALTY
                if cand < dp_next1[j]:
                    dp_next1[j] = cand
                    back_rows[i + 1][j - jl1] = OP_SKIP_JA

            if j < M and j + 1 <= jh:
                cand = d_ij + SKIP_EN_PENALTY
                if cand < dp_curr[j + 1]:
                    dp_curr[j + 1] = cand
                    back_rows[i][j + 1 - jl] = OP_SKIP_EN

            if has_next1 and j < M and jl1 <= j + 1 <= jh1:
                es1 = en_s[j]
                ee1 = en_e[j]
                if je1 + max_window_ms >= es1 and ee1 + max_window_ms >= js1:
                    gap = max(0, (js1 if js1 > es1 else es1) - (je1 if je1 < ee1 else ee1))
                    if gap <= max_window_ms:
                        inter = max(0, (je1 if je1 < ee1 else ee1) - (js1 if js1 > es1 else es1))
                        union = max(1, (je1 if je1 > ee1 else ee1) - (js1 if js1 < es1 else es1))
                        iou_cost = 1.0 - (inter / union)
                        boundary_err = (abs(js1 - es1) + abs(je1 - ee1)) * inv_2w
                        ew1 = en_words[j]
                        dp_pen = 0.35 if ((jm1 <= 3 and ew1 >= 10) or (ew1 <= 1 and jm1 >= 22)) else 0.0
                        bonus = 0.0
                        if jq1 and en_q[j]:
                            bonus += 0.20
                        if jex1 and en_ex[j]:
                            bonus += 0.15
                        if jcm1 & en_cmask[j]:
                            bonus += 0.35
                        c11 = 0.65 * iou_cost + 0.55 * boundary_err + dp_pen - bonus
                        if c11 < 0.0:
                            c11 = 0.0
                        cand = d_ij + c11
                        if cand < dp_next1[j + 1]:
                            dp_next1[j + 1] = cand
                            back_rows[i + 1][j + 1 - jl1] = OP_1_1

            if has_next1 and j + 1 < M and en_p_ok[j] and jl1 <= j + 2 <= jh1:
                ep_s = en_s[j]
                ep_e = en_e[j + 1]
                if je1 + max_window_ms >= ep_s and ep_e + max_window_ms >= js1:
                    gap = max(0, (js1 if js1 > ep_s else ep_s) - (je1 if je1 < ep_e else ep_e))
                    if gap <= max_window_ms:
                        inter = max(0, (je1 if je1 < ep_e else ep_e) - (js1 if js1 > ep_s else ep_s))
                        union = max(1, (je1 if je1 > ep_e else ep_e) - (js1 if js1 < ep_s else ep_s))
                        iou_cost = 1.0 - (inter / union)
                        boundary_err = (abs(js1 - ep_s) + abs(je1 - ep_e)) * inv_2w
                        ewp = en_p_words[j]
                        dp_pen = 0.35 if ((jm1 <= 3 and ewp >= 10) or (ewp <= 1 and jm1 >= 22)) else 0.0
                        bonus = 0.0
                        if jq1 and en_p_q[j]:
                            bonus += 0.20
                        if jex1 and en_p_ex[j]:
                            bonus += 0.15
                        if jcm1 & en_p_cmask[j]:
                            bonus += 0.35
                        c12 = 0.65 * iou_cost + 0.55 * boundary_err + dp_pen - bonus
                        if c12 < 0.0:
                            c12 = 0.0
                        cand = d_ij + c12 + SPLIT_MERGE_PENALTY
                        if cand < dp_next1[j + 2]:
                            dp_next1[j + 2] = cand
                            back_rows[i + 1][j + 2 - jl1] = OP_1_2

            if jp_ok and j < M and jl2 <= j + 1 <= jh2:
                es1 = en_s[j]
                ee1 = en_e[j]
                if jp_e + max_window_ms >= es1 and ee1 + max_window_ms >= jp_s:
                    gap = max(0, (jp_s if jp_s > es1 else es1) - (jp_e if jp_e < ee1 else ee1))
                    if gap <= max_window_ms:
                        inter = max(0, (jp_e if jp_e < ee1 else ee1) - (jp_s if jp_s > es1 else es1))
                        union = max(1, (jp_e if jp_e > ee1 else ee1) - (jp_s if jp_s < es1 else es1))
                        iou_cost = 1.0 - (inter / union)
                        boundary_err = (abs(jp_s - es1) + abs(jp_e - ee1)) * inv_2w
                        ew1 = en_words[j]
                        dp_pen = 0.35 if ((jp_m <= 3 and ew1 >= 10) or (ew1 <= 1 and jp_m >= 22)) else 0.0
                        bonus = 0.0
                        if jp_q and en_q[j]:
                            bonus += 0.20
                        if jp_ex and en_ex[j]:
                            bonus += 0.15
                        if jp_cm & en_cmask[j]:
                            bonus += 0.35
                        c21 = 0.65 * iou_cost + 0.55 * boundary_err + dp_pen - bonus
                        if c21 < 0.0:
                            c21 = 0.0
                        cand = d_ij + c21 + SPLIT_MERGE_PENALTY
                        if cand < dp_next2[j + 1]:
                            dp_next2[j + 1] = cand
                            back_rows[i + 2][j + 1 - jl2] = OP_2_1

        for jj in range(jl, jh + 1):
            dp_curr[jj] = INF
        dp_curr, dp_next1, dp_next2 = dp_next1, dp_next2, dp_curr

    ops: List[Tuple[str, int, int]] = []
    ci, cj = N, M
    while ci > 0 or cj > 0:
        jl = j_lows[ci]
        jh = j_highs[ci]
        if cj < jl or cj > jh:
            break
        code = back_rows[ci][cj - jl]
        if code == OP_NONE:
            break
        if code == OP_SKIP_JA:
            pi, pj = ci - 1, cj
            ops.append(("SKIP_JA", pi, pj))
        elif code == OP_SKIP_EN:
            pi, pj = ci, cj - 1
            ops.append(("SKIP_EN", pi, pj))
        elif code == OP_1_1:
            pi, pj = ci - 1, cj - 1
            ops.append(("1:1", pi, pj))
        elif code == OP_1_2:
            pi, pj = ci - 1, cj - 2
            ops.append(("1:2", pi, pj))
        elif code == OP_2_1:
            pi, pj = ci - 2, cj - 1
            ops.append(("2:1", pi, pj))
        else:
            break
        ci, cj = pi, pj

    return ops


def _split_2_to_1_timing(
    ja_first: SubCue,
    ja_second: SubCue,
    en_cue: SubCue,
) -> Tuple[Tuple[int, int, str], Tuple[int, int, str]]:
    """Split a merged EN timing window and reference text across two consecutive JA cues."""
    dur1 = max(1, ja_first.end_ms - ja_first.start_ms)
    dur2 = max(1, ja_second.end_ms - ja_second.start_ms)
    ratio = dur1 / (dur1 + dur2)
    en_start = en_cue.start_ms
    en_end = max(en_start, en_cue.end_ms)
    total_en_dur = en_end - en_start

    if total_en_dur <= 1:
        c1_start, c1_end = en_start, en_start
        c2_start, c2_end = en_start, en_end
    elif total_en_dur == 2:
        c1_start, c1_end = en_start, en_start + 1
        c2_start, c2_end = en_start + 1, en_end
    else:
        gap = min(50, total_en_dur // 3)
        avail = total_en_dur - gap
        min_part = 200 if avail >= 400 else 1
        part1_dur = max(min_part, min(avail - min_part, int(avail * ratio)))
        c1_start = en_start
        c1_end = c1_start + part1_dur
        c2_start = c1_end + gap
        c2_end = en_end

    en_part1, en_part2 = split_en_reference_for_2_to_1(en_cue.text, ratio)
    return (c1_start, c1_end, en_part1), (c2_start, c2_end, en_part2)


def estimate_en_words(text: str) -> int:
    """Estimate English word count after stripping SDH audio tags."""
    clean = re.sub(r"\[.*?\]|\(.*?\)|♪.*?♪", "", text).strip()
    return max(1, len(clean.split()))


def split_ja_text_proportionally(ja_text: str, ratio_first: float) -> Tuple[str, str]:
    """
    Split Whisper Japanese text into 2 parts when matching 2 consecutive EN cues (1:2 Match).
    Prioritize splitting at punctuation marks, spaces, or particles.
    """
    n = len(ja_text)
    if n <= 4:
        return ja_text, ""
    target_pos = max(1, min(n - 1, int(round(n * ratio_first))))

    # Find natural breakpoint (punctuation or whitespace) within +-35% radius around target_pos
    best_idx = None
    best_dist = n
    for m in re.finditer(r"[、。！？…\s ]+", ja_text):
        cut = m.end()
        if 1 <= cut < n:
            dist = abs(cut - target_pos)
            if dist < best_dist:
                best_dist = dist
                best_idx = cut

    if best_idx is not None and best_dist <= max(4, int(n * 0.35)):
        return ja_text[:best_idx].strip(), ja_text[best_idx:].strip()
    return ja_text[:target_pos].strip(), ja_text[target_pos:].strip()


def split_en_reference_for_2_to_1(en_text: str, ratio_first: float) -> Tuple[str, str]:
    """
    Split English reference text at natural clause boundary (comma, semicolon, dash,
    or conjunction) when 2 consecutive Whisper Japanese cues map to 1 long EN cue (2:1 Match).
    Prevents Gemini from repeating identical EN reference across 2 consecutive cues.
    """
    clean = (en_text or "").strip()
    n = len(clean)
    if n < 12 or len(clean.split()) < 4:
        return clean, clean

    target_pos = max(1, min(n - 1, int(round(n * ratio_first))))
    candidates: List[Tuple[int, int, int]] = []

    # Priority 1: Clause punctuation marks (, ; : - -- ! ?)
    for m in re.finditer(r"(?:[,;:—]|--|\.\.\.|[!?])\s+", clean):
        cut_end = m.end()
        if 3 <= cut_end <= n - 3:
            candidates.append((abs(cut_end - target_pos), m.end(), m.end()))

    # Priority 2: Coordinating conjunctions (and, but, so, while, when, because)
    if not candidates:
        for m in re.finditer(r"\s+(?:and|but|so|while|when|because|though)\s+", clean, flags=re.IGNORECASE):
            cut_pos = m.start()
            if 4 <= cut_pos <= n - 4:
                candidates.append((abs(cut_pos - target_pos) + 4, cut_pos, cut_pos + 1))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_dist, p1_end, p2_start = candidates[0]
        if best_dist <= max(8, int(n * 0.45)):
            part1 = clean[:p1_end].strip()
            part2 = clean[p2_start:].strip()
            if part1 and part2:
                return part1, part2

    return clean, clean


def _compute_dtw_pair_cost(
    j_start: int,
    j_end: int,
    e_start: int,
    e_end: int,
    j_txt: str,
    e_txt: str,
    cast_names_ja_en: List[Tuple[str, str]],
    max_window_ms: int,
) -> float:
    """Compute DTW alignment cost combining temporal IoU, boundary error, density mismatch, and anchor bonuses."""
    gap = max(0, max(j_start, e_start) - min(j_end, e_end))
    if gap > max_window_ms:
        return 1e6

    inter = max(0, min(j_end, e_end) - max(j_start, e_start))
    union = max(1, max(j_end, e_end) - min(j_start, e_start))
    iou_cost = 1.0 - (inter / union)
    boundary_err = (abs(j_start - e_start) + abs(j_end - e_end)) / (2.0 * max(1, max_window_ms))

    j_clean = strip_ja_furigana(j_txt)
    mora = estimate_ja_mora(j_clean)
    en_words = estimate_en_words(e_txt)

    # Penalize extreme length mismatches (e.g., short exclamations stealing long EN dialogue).
    density_penalty = 0.0
    if (mora <= 3 and en_words >= 10) or (en_words <= 1 and mora >= 22):
        density_penalty = 0.35

    anchor_bonus = 0.0
    if ("？" in j_txt or "?" in j_txt) and "?" in e_txt:
        anchor_bonus += 0.20
    if ("！" in j_txt or "!" in j_txt) and "!" in e_txt:
        anchor_bonus += 0.15
    for ja_name, en_name in cast_names_ja_en:
        norm_ja_name = strip_ja_furigana(ja_name)
        if norm_ja_name and norm_ja_name in j_clean and en_name.lower() in e_txt.lower():
            anchor_bonus += 0.35
            break

    return max(0.0, 0.65 * iou_cost + 0.55 * boundary_err + density_penalty - anchor_bonus)


def align_ja_to_en_monotonic_dtw(
    ja_cues: List[SubCue],
    en_cues: List[SubCue],
    cast_names_ja_en: List[Tuple[str, str]] = (),
    max_window_ms: int = 2500
) -> List[Tuple[SubCue, str]]:
    """Align Whisper JA cues to studio EN cues supporting 1:1, 1:2, 2:1, and OP/ED quarantine."""
    N, M = len(ja_cues), len(en_cues)
    if N == 0 or M == 0:
        return [(c, "") for c in ja_cues]

    ops = _compute_monotonic_dtw_backpointers(ja_cues, en_cues, cast_names_ja_en, max_window_ms)

    aligned_output: List[Tuple[SubCue, str]] = []
    for op, pi, pj in ops:
        if op == "1:1":
            aligned_output.append((
                SubCue(en_cues[pj].start_ms, en_cues[pj].end_ms, ja_cues[pi].text),
                en_cues[pj].text
            ))
        elif op == "1:2":
            dur1 = max(1, en_cues[pj].end_ms - en_cues[pj].start_ms)
            dur2 = max(1, en_cues[pj + 1].end_ms - en_cues[pj + 1].start_ms)
            ja_part1, ja_part2 = split_ja_text_proportionally(ja_cues[pi].text, dur1 / (dur1 + dur2))
            aligned_output.append((
                SubCue(en_cues[pj + 1].start_ms, en_cues[pj + 1].end_ms, ja_part2 or ja_cues[pi].text),
                en_cues[pj + 1].text
            ))
            aligned_output.append((
                SubCue(en_cues[pj].start_ms, en_cues[pj].end_ms, ja_part1 or ja_cues[pi].text),
                en_cues[pj].text
            ))
        elif op == "2:1":
            (c1_start, c1_end, en_part1), (c2_start, c2_end, en_part2) = _split_2_to_1_timing(
                ja_cues[pi], ja_cues[pi + 1], en_cues[pj]
            )
            # Push in reverse order so final .reverse() preserves pi, pi+1 ordering.
            aligned_output.append((SubCue(c2_start, c2_end, ja_cues[pi + 1].text), en_part2))
            aligned_output.append((SubCue(c1_start, c1_end, ja_cues[pi].text), en_part1))
        elif op == "SKIP_JA":
            aligned_output.append((ja_cues[pi], ""))

    aligned_output.reverse()
    return aligned_output


def get_dtw_alignment_map(
    ja_cues: List[SubCue],
    en_cues: List[SubCue],
    cast_names_ja_en: List[Tuple[str, str]] = (),
    max_window_ms: int = 2500
) -> Dict[int, Optional[Tuple[int, int, str]]]:
    """Map JA cue index -> (start_ms, end_ms, en_text) or None for unaligned OP/ED lines."""
    N, M = len(ja_cues), len(en_cues)
    if N == 0 or M == 0:
        return {}

    ops = _compute_monotonic_dtw_backpointers(ja_cues, en_cues, cast_names_ja_en, max_window_ms)

    cue_map: Dict[int, Optional[Tuple[int, int, str]]] = {}
    for op, pi, pj in ops:
        if op == "1:1":
            cue_map[pi] = (en_cues[pj].start_ms, en_cues[pj].end_ms, en_cues[pj].text)
        elif op == "1:2":
            cue_map[pi] = (en_cues[pj].start_ms, en_cues[pj + 1].end_ms, en_cues[pj].text + " " + en_cues[pj + 1].text)
        elif op == "2:1":
            (c1_start, c1_end, en_part1), (c2_start, c2_end, en_part2) = _split_2_to_1_timing(
                ja_cues[pi], ja_cues[pi + 1], en_cues[pj]
            )
            cue_map[pi] = (c1_start, c1_end, en_part1)
            cue_map[pi + 1] = (c2_start, c2_end, en_part2)
        elif op == "SKIP_JA":
            cue_map[pi] = None

    return cue_map


def get_dtw_alignment_plan(
    ja_cues: List[SubCue],
    en_cues: List[SubCue],
    cast_names_ja_en: List[Tuple[str, str]] = (),
    max_window_ms: int = 2500,
    min_pause_for_1_to_2_split_ms: int = 250,
) -> Tuple[Dict[int, Optional[Tuple[int, int, str]]], Dict[int, List[Tuple[int, int, str, str]]]]:
    """Return (cue_map, split_1_to_2_map) splitting 1:2 matches across scene pauses >= threshold."""
    N, M = len(ja_cues), len(en_cues)
    if N == 0 or M == 0:
        return {}, {}

    ops = _compute_monotonic_dtw_backpointers(ja_cues, en_cues, cast_names_ja_en, max_window_ms)

    cue_map: Dict[int, Optional[Tuple[int, int, str]]] = {}
    split_1_to_2_map: Dict[int, List[Tuple[int, int, str, str]]] = {}
    for op, pi, pj in ops:
        if op == "1:1":
            cue_map[pi] = (en_cues[pj].start_ms, en_cues[pj].end_ms, en_cues[pj].text)
        elif op == "1:2":
            en_gap = en_cues[pj + 1].start_ms - en_cues[pj].end_ms
            dur1 = max(1, en_cues[pj].end_ms - en_cues[pj].start_ms)
            dur2 = max(1, en_cues[pj + 1].end_ms - en_cues[pj + 1].start_ms)
            ja_part1, ja_part2 = split_ja_text_proportionally(ja_cues[pi].text, dur1 / (dur1 + dur2))
            if en_gap >= min_pause_for_1_to_2_split_ms and ja_part1 and ja_part2:
                split_1_to_2_map[pi] = [
                    (en_cues[pj].start_ms, en_cues[pj].end_ms, ja_part1, en_cues[pj].text),
                    (en_cues[pj + 1].start_ms, en_cues[pj + 1].end_ms, ja_part2, en_cues[pj + 1].text),
                ]
                cue_map[pi] = (en_cues[pj].start_ms, en_cues[pj].end_ms, en_cues[pj].text)
            else:
                cue_map[pi] = (
                    en_cues[pj].start_ms,
                    en_cues[pj + 1].end_ms,
                    en_cues[pj].text + " " + en_cues[pj + 1].text,
                )
        elif op == "2:1":
            (c1_start, c1_end, en_part1), (c2_start, c2_end, en_part2) = _split_2_to_1_timing(
                ja_cues[pi], ja_cues[pi + 1], en_cues[pj]
            )
            cue_map[pi] = (c1_start, c1_end, en_part1)
            cue_map[pi + 1] = (c2_start, c2_end, en_part2)
        elif op == "SKIP_JA":
            cue_map[pi] = None

    return cue_map, split_1_to_2_map
