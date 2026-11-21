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
) -> List[List[Optional[Tuple[str, int, int]]]]:
    """Run monotonic 1:1, 1:2, 2:1, and skip DP transitions and return backpointer table."""
    N, M = len(ja_cues), len(en_cues)
    INF = 1e9
    dp = [[INF] * (M + 1) for _ in range(N + 1)]
    back: List[List[Optional[Tuple[str, int, int]]]] = [[None] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0

    SKIP_JA_PENALTY = 0.85
    SKIP_EN_PENALTY = 0.85
    SPLIT_MERGE_PENALTY = 0.12

    for i in range(N + 1):
        for j in range(M + 1):
            if dp[i][j] >= INF:
                continue
            if i < N and dp[i][j] + SKIP_JA_PENALTY < dp[i + 1][j]:
                dp[i + 1][j] = dp[i][j] + SKIP_JA_PENALTY
                back[i + 1][j] = ("SKIP_JA", i, j)
            if j < M and dp[i][j] + SKIP_EN_PENALTY < dp[i][j + 1]:
                dp[i][j + 1] = dp[i][j] + SKIP_EN_PENALTY
                back[i][j + 1] = ("SKIP_EN", i, j)
            if (
                i < N
                and j < M
                and ja_cues[i].end_ms + max_window_ms >= en_cues[j].start_ms
                and en_cues[j].end_ms + max_window_ms >= ja_cues[i].start_ms
            ):
                c11 = _compute_dtw_pair_cost(
                    ja_cues[i].start_ms, ja_cues[i].end_ms,
                    en_cues[j].start_ms, en_cues[j].end_ms,
                    ja_cues[i].text, en_cues[j].text,
                    cast_names_ja_en, max_window_ms,
                )
                if c11 < 1e5 and dp[i][j] + c11 < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = dp[i][j] + c11
                    back[i + 1][j + 1] = ("1:1", i, j)
            if (
                i < N
                and j + 1 < M
                and (en_cues[j + 1].start_ms - en_cues[j].end_ms) <= 600
                and ja_cues[i].end_ms + max_window_ms >= en_cues[j].start_ms
                and en_cues[j + 1].end_ms + max_window_ms >= ja_cues[i].start_ms
            ):
                c12 = _compute_dtw_pair_cost(
                    ja_cues[i].start_ms, ja_cues[i].end_ms,
                    en_cues[j].start_ms, en_cues[j + 1].end_ms,
                    ja_cues[i].text, en_cues[j].text + " " + en_cues[j + 1].text,
                    cast_names_ja_en, max_window_ms,
                ) + SPLIT_MERGE_PENALTY
                if c12 < 1e5 and dp[i][j] + c12 < dp[i + 1][j + 2]:
                    dp[i + 1][j + 2] = dp[i][j] + c12
                    back[i + 1][j + 2] = ("1:2", i, j)
            if (
                i + 1 < N
                and j < M
                and (ja_cues[i + 1].start_ms - ja_cues[i].end_ms) <= 600
                and ja_cues[i + 1].end_ms + max_window_ms >= en_cues[j].start_ms
                and en_cues[j].end_ms + max_window_ms >= ja_cues[i].start_ms
            ):
                c21 = _compute_dtw_pair_cost(
                    ja_cues[i].start_ms, ja_cues[i + 1].end_ms,
                    en_cues[j].start_ms, en_cues[j].end_ms,
                    ja_cues[i].text + " " + ja_cues[i + 1].text, en_cues[j].text,
                    cast_names_ja_en, max_window_ms,
                ) + SPLIT_MERGE_PENALTY
                if c21 < 1e5 and dp[i][j] + c21 < dp[i + 2][j + 1]:
                    dp[i + 2][j + 1] = dp[i][j] + c21
                    back[i + 2][j + 1] = ("2:1", i, j)
    return back


def _split_2_to_1_timing(
    ja_first: SubCue,
    ja_second: SubCue,
    en_cue: SubCue,
) -> Tuple[Tuple[int, int, str], Tuple[int, int, str]]:
    """Split a merged EN timing window and reference text across two consecutive JA cues."""
    dur1 = max(1, ja_first.end_ms - ja_first.start_ms)
    dur2 = max(1, ja_second.end_ms - ja_second.start_ms)
    total_en_dur = max(600, en_cue.end_ms - en_cue.start_ms)
    gap = 50
    ratio = dur1 / (dur1 + dur2)
    part1_dur = max(200, int((total_en_dur - gap) * ratio))
    c1_start = en_cue.start_ms
    c1_end = c1_start + part1_dur
    c2_start = c1_end + gap
    c2_end = max(c2_start + 200, en_cue.end_ms)
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
    boundary_err = (abs(j_start - e_start) + abs(j_end - e_end)) / (2.0 * max_window_ms)

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

    back = _compute_monotonic_dtw_backpointers(ja_cues, en_cues, cast_names_ja_en, max_window_ms)

    aligned_output: List[Tuple[SubCue, str]] = []
    ci, cj = N, M
    while ci > 0 or cj > 0:
        step = back[ci][cj]
        if step is None:
            break
        op, pi, pj = step
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
        ci, cj = pi, pj

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

    back = _compute_monotonic_dtw_backpointers(ja_cues, en_cues, cast_names_ja_en, max_window_ms)

    cue_map: Dict[int, Optional[Tuple[int, int, str]]] = {}
    ci, cj = N, M
    while ci > 0 or cj > 0:
        step = back[ci][cj]
        if step is None:
            break
        op, pi, pj = step
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
        ci, cj = pi, pj

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

    back = _compute_monotonic_dtw_backpointers(ja_cues, en_cues, cast_names_ja_en, max_window_ms)

    cue_map: Dict[int, Optional[Tuple[int, int, str]]] = {}
    split_1_to_2_map: Dict[int, List[Tuple[int, int, str, str]]] = {}
    ci, cj = N, M
    while ci > 0 or cj > 0:
        step = back[ci][cj]
        if step is None:
            break
        op, pi, pj = step
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
        ci, cj = pi, pj

    return cue_map, split_1_to_2_map
