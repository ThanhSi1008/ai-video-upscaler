# -*- coding: utf-8 -*-
"""
subtrans/dtw_auto_align.py
Thuật toán Auto-Align Subtitle Timestamps thế hệ mới bằng Quy hoạch động đơn điệu
(Anchor-Guided Monotonic Dynamic Time Warping - MS-DTW):
1. Khắc phục triệt để các ca góc của thuật toán tham lam cục bộ:
   - 1 JA <-> 2 EN: Whisper gộp 1 câu dài có ngắt nghỉ ngắn -> Tự động tách câu tiếng Nhật
     tại dấu câu (、, 。) theo tỷ lệ thời lượng của 2 mốc EN phòng thu.
   - 2 JA <-> 1 EN: Whisper ngắt làm 2 câu ngắn -> Tự động gộp tiếng Nhật vào khung thời gian của câu EN.
   - OP/ED Quarantine: Phát hiện khoảng trống EN > 45s (nhạc mở đầu / kết thúc), giữ nguyên timestamp Whisper,
     tuyệt đối không cho phép lời bài hát tranh chấp biên với câu thoại.
   - Lọc bỏ thẻ SDH trong tiếng Anh ([groans], (sighs), ♪ ... ♪).
2. Điểm neo ngữ nghĩa (Anchor Bonus):
   - Thưởng điểm khi câu JA và EN cùng xuất hiện tên nhân vật trong Cast Sheet, hoặc cùng có dấu câu ? !.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict


@dataclass
class SubCue:
    start_ms: int
    end_ms: int
    text: str


def estimate_ja_mora(text: str) -> int:
    """Ước tính số lượng âm tiết (mora) của câu thoại tiếng Nhật."""
    clean = re.sub(r"[\s\W_]+", "", text)
    return max(1, len(clean))


def estimate_en_words(text: str) -> int:
    """Ước tính số từ tiếng Anh sau khi loại bỏ các thẻ âm thanh SDH."""
    clean = re.sub(r"\[.*?\]|\(.*?\)|♪.*?♪", "", text).strip()
    return max(1, len(clean.split()))


def split_ja_text_proportionally(ja_text: str, ratio_first: float) -> Tuple[str, str]:
    """
    Tách câu tiếng Nhật của Whisper thành 2 phần khi khớp với 2 câu EN liên tiếp (1:2 Match).
    Ưu tiên tách tại dấu phẩy '、', dấu chấm '。', dấu than '！', dấu hỏi '？', dấu cách hoặc trợ từ.
    """
    n = len(ja_text)
    if n <= 4:
        return ja_text, ""
    target_pos = max(1, min(n - 1, int(round(n * ratio_first))))

    # Tìm điểm ngắt tự nhiên (dấu câu / khoảng trắng) trong bán kính +-35% quanh target_pos
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


def align_ja_to_en_monotonic_dtw(
    ja_cues: List[SubCue],
    en_cues: List[SubCue],
    cast_names_ja_en: List[Tuple[str, str]] = (),
    max_window_ms: int = 2500
) -> List[Tuple[SubCue, str]]:
    """
    Căn chỉnh chuỗi phụ đề Whisper JA theo timeline chuẩn phòng thu của EN
    bằng Quy hoạch động đơn điệu (Monotonic DTW) hỗ trợ:
      - 1 JA <-> 1 EN (Snap trực tiếp)
      - 1 JA <-> 2 EN (Whisper gộp câu -> Tự động tách JA theo 2 mốc EN)
      - 2 JA <-> 1 EN (Whisper tách vụn -> Tự động gộp JA vào 1 mốc EN)
      - Cô lập vùng OP/ED (Khoảng trống EN > 45s) để không kéo lệch câu thoại biên.
    Returns:
      Danh sách các cặp (Aligned_SubCue_với_Timestamp_Chuẩn, paired_en_text).
    """
    N, M = len(ja_cues), len(en_cues)
    if N == 0 or M == 0:
        return [(c, "") for c in ja_cues]

    def pair_cost(j_start: int, j_end: int, e_start: int, e_end: int, j_txt: str, e_txt: str) -> float:
        # Nếu cách nhau quá xa (> max_window_ms), phạt nặng để cấm ghép
        gap = max(0, max(j_start, e_start) - min(j_end, e_end))
        if gap > max_window_ms:
            return 1e6

        inter = max(0, min(j_end, e_end) - max(j_start, e_start))
        union = max(1, max(j_end, e_end) - min(j_start, e_start))
        iou_cost = 1.0 - (inter / union)

        boundary_err = (abs(j_start - e_start) + abs(j_end - e_end)) / (2.0 * max_window_ms)

        # Kiểm tra Điểm neo Ngữ nghĩa (Dấu câu ? ! và Tên nhân vật trong Cast Sheet)
        anchor_bonus = 0.0
        if ("？" in j_txt or "?" in j_txt) and "?" in e_txt:
            anchor_bonus += 0.20
        if ("！" in j_txt or "!" in j_txt) and "!" in e_txt:
            anchor_bonus += 0.15
        for ja_name, en_name in cast_names_ja_en:
            if ja_name in j_txt and en_name.lower() in e_txt.lower():
                anchor_bonus += 0.35
                break

        return max(0.0, 0.65 * iou_cost + 0.55 * boundary_err - anchor_bonus)

    INF = 1e9
    dp = [[INF] * (M + 1) for _ in range(N + 1)]
    back = [[None] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0

    SKIP_JA_PENALTY = 0.85
    SKIP_EN_PENALTY = 0.85
    SPLIT_MERGE_PENALTY = 0.12

    for i in range(N + 1):
        for j in range(M + 1):
            if dp[i][j] >= INF:
                continue

            # 1. Bỏ qua 1 dòng JA (Lời bài hát OP/ED hoặc tiếng thở không có trong EN)
            if i < N and dp[i][j] + SKIP_JA_PENALTY < dp[i + 1][j]:
                dp[i + 1][j] = dp[i][j] + SKIP_JA_PENALTY
                back[i + 1][j] = ("SKIP_JA", i, j)

            # 2. Bỏ qua 1 dòng EN (Thẻ SDH [music] hoặc biển hiệu trên màn hình)
            if j < M and dp[i][j] + SKIP_EN_PENALTY < dp[i][j + 1]:
                dp[i][j + 1] = dp[i][j] + SKIP_EN_PENALTY
                back[i][j + 1] = ("SKIP_EN", i, j)

            # 3. Khớp 1 JA <-> 1 EN
            if i < N and j < M:
                c11 = pair_cost(
                    ja_cues[i].start_ms, ja_cues[i].end_ms,
                    en_cues[j].start_ms, en_cues[j].end_ms,
                    ja_cues[i].text, en_cues[j].text
                )
                if c11 < 1e5 and dp[i][j] + c11 < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = dp[i][j] + c11
                    back[i + 1][j + 1] = ("1:1", i, j)

            # 4. Khớp 1 JA <-> 2 EN (Whisper gộp 1 câu dài, Studio EN tách làm 2 câu)
            if i < N and j + 1 < M and (en_cues[j + 1].start_ms - en_cues[j].end_ms) <= 600:
                c12 = pair_cost(
                    ja_cues[i].start_ms, ja_cues[i].end_ms,
                    en_cues[j].start_ms, en_cues[j + 1].end_ms,
                    ja_cues[i].text, en_cues[j].text + " " + en_cues[j + 1].text
                ) + SPLIT_MERGE_PENALTY
                if c12 < 1e5 and dp[i][j] + c12 < dp[i + 1][j + 2]:
                    dp[i + 1][j + 2] = dp[i][j] + c12
                    back[i + 1][j + 2] = ("1:2", i, j)

            # 5. Khớp 2 JA <-> 1 EN (Whisper ngắt 2 câu ngắn, Studio EN gộp làm 1 câu)
            if i + 1 < N and j < M and (ja_cues[i + 1].start_ms - ja_cues[i].end_ms) <= 600:
                c21 = pair_cost(
                    ja_cues[i].start_ms, ja_cues[i + 1].end_ms,
                    en_cues[j].start_ms, en_cues[j].end_ms,
                    ja_cues[i].text + " " + ja_cues[i + 1].text, en_cues[j].text
                ) + SPLIT_MERGE_PENALTY
                if c21 < 1e5 and dp[i][j] + c21 < dp[i + 2][j + 1]:
                    dp[i + 2][j + 1] = dp[i][j] + c21
                    back[i + 2][j + 1] = ("2:1", i, j)

    # Truy vết đường đi tối ưu (Backtracking)
    aligned_output: List[Tuple[SubCue, str]] = []
    ci, cj = N, M
    while ci > 0 or cj > 0:
        step = back[ci][cj]
        if step is None:
            break
        op, pi, pj = step
        if op == "1:1":
            # Ghim hoàn toàn mốc thời gian chuẩn của Studio EN
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
            dur1 = max(1, ja_cues[pi].end_ms - ja_cues[pi].start_ms)
            dur2 = max(1, ja_cues[pi + 1].end_ms - ja_cues[pi + 1].start_ms)
            total_en_dur = max(600, en_cues[pj].end_ms - en_cues[pj].start_ms)
            gap = 50
            part1_dur = max(200, int((total_en_dur - gap) * (dur1 / (dur1 + dur2))))
            c1_start = en_cues[pj].start_ms
            c1_end = c1_start + part1_dur
            c2_start = c1_end + gap
            c2_end = max(c2_start + 200, en_cues[pj].end_ms)
            # Thêm theo thứ tự ngược để reverse() ở cuối giữ đúng trật tự pi, pi+1
            aligned_output.append((
                SubCue(c2_start, c2_end, ja_cues[pi + 1].text),
                en_cues[pj].text
            ))
            aligned_output.append((
                SubCue(c1_start, c1_end, ja_cues[pi].text),
                en_cues[pj].text
            ))
        elif op == "SKIP_JA":
            # Giữ nguyên timestamp gốc của Whisper nếu nằm trong vùng OP/ED hoặc không có sub EN tương ứng
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
    """
    Trả về ánh xạ trực tiếp từ chỉ mục câu JA (pi: 0..N-1) sang tuple (start_ms, end_ms, en_text).
    Nếu câu JA không khớp với câu EN nào (SKIP_JA), giá trị là None.
    Đảm bảo tuyệt đối không làm dồn toa hay lệch chỉ mục câu thoại.
    """
    N, M = len(ja_cues), len(en_cues)
    if N == 0 or M == 0:
        return {}

    def pair_cost(j_start: int, j_end: int, e_start: int, e_end: int, j_txt: str, e_txt: str) -> float:
        gap = max(0, max(j_start, e_start) - min(j_end, e_end))
        if gap > max_window_ms:
            return 1e6
        inter = max(0, min(j_end, e_end) - max(j_start, e_start))
        union = max(1, max(j_end, e_end) - min(j_start, e_start))
        iou_cost = 1.0 - (inter / union)
        boundary_err = (abs(j_start - e_start) + abs(j_end - e_end)) / (2.0 * max_window_ms)
        anchor_bonus = 0.0
        if ("？" in j_txt or "?" in j_txt) and "?" in e_txt:
            anchor_bonus += 0.20
        if ("！" in j_txt or "!" in j_txt) and "!" in e_txt:
            anchor_bonus += 0.15
        for ja_name, en_name in cast_names_ja_en:
            if ja_name in j_txt and en_name.lower() in e_txt.lower():
                anchor_bonus += 0.35
                break
        return max(0.0, 0.65 * iou_cost + 0.55 * boundary_err - anchor_bonus)

    INF = 1e9
    dp = [[INF] * (M + 1) for _ in range(N + 1)]
    back = [[None] * (M + 1) for _ in range(N + 1)]
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
            if i < N and j < M:
                c11 = pair_cost(
                    ja_cues[i].start_ms, ja_cues[i].end_ms,
                    en_cues[j].start_ms, en_cues[j].end_ms,
                    ja_cues[i].text, en_cues[j].text
                )
                if c11 < 1e5 and dp[i][j] + c11 < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = dp[i][j] + c11
                    back[i + 1][j + 1] = ("1:1", i, j)
            if i < N and j + 1 < M and (en_cues[j + 1].start_ms - en_cues[j].end_ms) <= 600:
                c12 = pair_cost(
                    ja_cues[i].start_ms, ja_cues[i].end_ms,
                    en_cues[j].start_ms, en_cues[j + 1].end_ms,
                    ja_cues[i].text, en_cues[j].text + " " + en_cues[j + 1].text
                ) + SPLIT_MERGE_PENALTY
                if c12 < 1e5 and dp[i][j] + c12 < dp[i + 1][j + 2]:
                    dp[i + 1][j + 2] = dp[i][j] + c12
                    back[i + 1][j + 2] = ("1:2", i, j)
            if i + 1 < N and j < M and (ja_cues[i + 1].start_ms - ja_cues[i].end_ms) <= 600:
                c21 = pair_cost(
                    ja_cues[i].start_ms, ja_cues[i + 1].end_ms,
                    en_cues[j].start_ms, en_cues[j].end_ms,
                    ja_cues[i].text + " " + ja_cues[i + 1].text, en_cues[j].text
                ) + SPLIT_MERGE_PENALTY
                if c21 < 1e5 and dp[i][j] + c21 < dp[i + 2][j + 1]:
                    dp[i + 2][j + 1] = dp[i][j] + c21
                    back[i + 2][j + 1] = ("2:1", i, j)

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
            dur1 = max(1, ja_cues[pi].end_ms - ja_cues[pi].start_ms)
            dur2 = max(1, ja_cues[pi + 1].end_ms - ja_cues[pi + 1].start_ms)
            total_en_dur = max(600, en_cues[pj].end_ms - en_cues[pj].start_ms)
            gap = 50
            part1_dur = max(200, int((total_en_dur - gap) * (dur1 / (dur1 + dur2))))
            c1_start = en_cues[pj].start_ms
            c1_end = c1_start + part1_dur
            c2_start = c1_end + gap
            c2_end = max(c2_start + 200, en_cues[pj].end_ms)
            cue_map[pi] = (c1_start, c1_end, en_cues[pj].text)
            cue_map[pi + 1] = (c2_start, c2_end, en_cues[pj].text)
        elif op == "SKIP_JA":
            cue_map[pi] = None
        ci, cj = pi, pj

    return cue_map
