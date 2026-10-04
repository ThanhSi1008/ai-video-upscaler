# -*- coding: utf-8 -*-
"""
subtrans/report.py
Stage C: Audit report generation (out/<name>.audit.md) and optional [?] review file.
Adheres to ADDENDUM 6 specifications:
- Audits uncertain cues ([?])
- Audits single-line compliance and line lengths (soft cap 80, max 107)
- Audits leftover Japanese characters
- Audits glossary compliance
- Produces review file when --review-file is requested
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pysubs2

from subtrans.models import Cue, Glossary
from subtrans.validation import JAPANESE_CHAR_REGEX
from subtrans.writer import format_timestamp_srt


def format_timestamp(ms: int) -> str:
    """Formats milliseconds as mm:ss.mmm."""
    seconds = ms // 1000
    m = seconds // 60
    s = seconds % 60
    rem = ms % 1000
    return f"{m:02d}:{s:02d}.{rem:03d}"


def generate_qa_report(
    cues: List[Cue],
    ja_path: Path,
    en_path: Optional[Path],
    output_subtitle_path: Path,
    report_path: Path,
    elapsed_time_sec: float,
    stats: Dict[str, Any],
    glossary: Optional[Glossary] = None,
    review_file: bool = False,
    stage_times: Optional[Dict[str, float]] = None,
) -> Path:
    """
    Generates a structured Markdown audit report (out/<name>.audit.md)
    and optional review file for uncertain cues.
    """
    total_cues = len(cues)
    dialogue_cues = [c for c in cues if not c.is_skipped]
    skipped_cues = [c for c in cues if c.is_skipped]

    gemini_count = stats.get("gemini", 0)
    local_count = stats.get("local", 0)
    cache_hits = stats.get("cache_hits", 0)

    # QA checks
    uncertain_cues = [c for c in dialogue_cues if c.uncertain]
    leftover_japanese = []
    multiline_cues = []
    lines_over_80 = []
    lines_over_107 = []
    missing_en = []
    glossary_violations = []

    for c in dialogue_cues:
        vi_text = c.vi or ""
        ja_text = c.ja or ""

        # Multiline check
        if "\n" in vi_text or "\\N" in vi_text:
            multiline_cues.append(c)

        # Length checks
        line_len = len(vi_text)
        if line_len > 107:
            lines_over_107.append((c, line_len))
        elif line_len > 80:
            lines_over_80.append((c, line_len))

        # Check leftover Japanese
        if JAPANESE_CHAR_REGEX.search(vi_text):
            leftover_japanese.append(c)

        # Check missing EN match if EN was provided
        if en_path and not c.en:
            missing_en.append(c)

        # Check glossary terms (allow vi or any alt)
        if glossary and glossary.terms:
            for term in glossary.terms:
                if len(term.ja) == 1:
                    # Single character terms require English context or exact match
                    if c.en and term.en and re.search(r"\b" + re.escape(term.en) + r"\b", c.en, re.IGNORECASE):
                        valid_translations = [term.vi.lower()] + [a.lower() for a in term.alt]
                        if not any(v in vi_text.lower() for v in valid_translations):
                            glossary_violations.append((c, term))
                elif term.ja in ja_text:
                    valid_translations = [term.vi.lower()] + [a.lower() for a in term.alt]
                    if not any(v in vi_text.lower() for v in valid_translations):
                        glossary_violations.append((c, term))

    lines = [
        f"# Subtitle Translation Audit Report: `{output_subtitle_path.name}`",
        "",
        "## 1. Summary Overview",
        "",
        f"- **Source (JA)**: `{ja_path.name}`",
        f"- **Aligned Reference (EN)**: `{en_path.name if en_path else 'None'}`",
        f"- **Output File**: `{output_subtitle_path.name}`",
        f"- **Processing Time**: {elapsed_time_sec:.1f}s ({len(dialogue_cues) / max(0.1, elapsed_time_sec):.2f} cues/s)",
        "",
        "| Metric | Count | Percentage |",
        "| :--- | :--- | :--- |",
        f"| **Total Cues** | {total_cues} | 100.0% |",
        f"| **Dialogue Cues Translated** | {len(dialogue_cues)} | {len(dialogue_cues)/max(1, total_cues)*100:.1f}% |",
        f"| **Uncertain Lines Flagged [?]** | {len(uncertain_cues)} | {len(uncertain_cues)/max(1, len(dialogue_cues))*100:.1f}% |",
        f"| **Single-line Compliant (0 hard wraps)** | {len(dialogue_cues) - len(multiline_cues)} | {(len(dialogue_cues)-len(multiline_cues))/max(1, len(dialogue_cues))*100:.1f}% |",
        f"| **Lines > 80 chars (Soft cap)** | {len(lines_over_80)} | {len(lines_over_80)/max(1, len(dialogue_cues))*100:.1f}% |",
        f"| **Lines > 107 chars (Hard threshold)** | {len(lines_over_107)} | {len(lines_over_107)/max(1, len(dialogue_cues))*100:.1f}% |",
        f"| **Cache Hits (Zero Latency)** | {cache_hits} | {cache_hits/max(1, len(dialogue_cues))*100:.1f}% |",
        "",
    ]

    if stage_times:
        lines.extend([
            "### Stage Timing Breakdown",
            "",
            "| Stage | Duration (s) | Overhead / Share |",
            "| :--- | :--- | :--- |",
            f"| Stage 1: Parse & Align | {stage_times.get('stage_1_parse', 0):.2f}s | {stage_times.get('stage_1_parse', 0)/max(0.1, elapsed_time_sec)*100:.1f}% |",
            f"| Stage A: Cast Sheet Analysis | {stage_times.get('stage_a_cast', 0):.2f}s | {stage_times.get('stage_a_cast', 0)/max(0.1, elapsed_time_sec)*100:.1f}% |",
            f"| Stage 2: Translation Batches | {stage_times.get('stage_2_translate', 0):.2f}s | {stage_times.get('stage_2_translate', 0)/max(0.1, elapsed_time_sec)*100:.1f}% |",
            f"| Stage Review (Optional) | {stage_times.get('stage_review', 0):.2f}s | {stage_times.get('stage_review', 0)/max(0.1, elapsed_time_sec)*100:.1f}% |",
            f"| Stage 4: Subtitle Write | {stage_times.get('stage_4_write', 0):.2f}s | {stage_times.get('stage_4_write', 0)/max(0.1, elapsed_time_sec)*100:.1f}% |",
            f"| Stage C: Audit Generation | {stage_times.get('stage_c_report', 0):.2f}s | {stage_times.get('stage_c_report', 0)/max(0.1, elapsed_time_sec)*100:.1f}% |",
            f"| **Total Pipeline** | **{elapsed_time_sec:.2f}s** | 100.0% |",
            "",
        ])

    model_counts = stats.get("model_counts", {})
    total_diag = max(1, len(dialogue_cues))
    local_cue_count = sum(cnt for mod, cnt in model_counts.items() if "local" in mod.lower() or "translategemma" in mod.lower() or "ollama" in mod.lower())
    local_pct = (local_cue_count / total_diag) * 100.0

    if model_counts:
        lines.extend([
            "### Model Usage Breakdown",
            "",
            "| Model / Source | Cues Translated | Share |",
            "| :--- | :--- | :--- |",
        ])
        for mod, cnt in model_counts.items():
            lines.append(f"| `{mod}` | {cnt} | {cnt/total_diag*100:.1f}% |")
        lines.append("")

        if local_pct > 30.0:
            lines.extend([
                "> [!WARNING]",
                f"> **CẢNH BÁO QUOTA HẠN MỨC**: Số lượng câu thoại do mô hình cục bộ (Local Ollama) dịch chiếm **{local_pct:.1f}%** (> 30% tổng số câu thoại).",
                "> Cần kiểm tra lại các câu thoại do mô hình cục bộ sinh ra để đảm bảo không bị lỗi xưng hô hoặc hiểu sai ngữ cảnh!",
                ""
            ])

    # Section: Batch Efficiency & Request Metrics (Target <= 5 requests/episode)
    total_batches = stats.get("total_batches", 0)
    split_batches = stats.get("split_batches", 0)
    split_rate = stats.get("split_rate_pct", 0.0)
    ep_reqs = stats.get("requests_this_episode", 0)
    batch_sz = stats.get("batch_size_configured", 100)
    dry_run = stats.get("dry_run", False)

    target_status = "✅ ĐẠT (≤ 5 requests)" if ep_reqs <= 5 else "⚠️ VƯỢT (> 5 requests)"
    if dry_run:
        target_status += " [DRY-RUN - 0 Quota Used]"

    lines.extend([
        "### Batch Efficiency & Request Minimization (Mục tiêu: ≤ 5 requests/tập)",
        "",
        "| Chỉ số | Giá trị đo được | Ghi chú / Mục tiêu |",
        "| :--- | :--- | :--- |",
        f"| **Kích thước Batch cấu hình** | {batch_sz} cues | Khoảng tối ưu: 100–150 cues |",
        f"| **Tổng số Batch gửi đi** | {total_batches} | Đã gom cụm thoại |",
        f"| **Số Batch phải tách đôi (Bisect)** | {split_batches} | Tách đôi tự động khi sai lệch ID/số cue |",
        f"| **Tỷ lệ Batch phải tách (Split Rate)** | {split_rate:.1f}% | Càng thấp hiệu suất càng cao |",
        f"| **Tổng số API Request thực tế của tập** | **{ep_reqs} requests** | **{target_status}** |",
        "",
    ])

    # Section: Pacific Time Quota & Usage
    pt_cues = stats.get("pacific_time_cues", {})
    pt_reqs = stats.get("pacific_time_requests", {})
    from subtrans.config import load_models_config
    from subtrans.cache import get_next_pt_reset, get_now_pt, TranslationCache
    models_cfg = load_models_config()
    configured_models = models_cfg.get("models", {})
    next_pt, next_vn = get_next_pt_reset()
    now_pt = get_now_pt()
    cache_inst = TranslationCache()

    lines.extend([
        f"### Hạn Mức Gemini API Theo Ngày Thái Bình Dương (Pacific Time - PT)",
        "",
        f"- **Giờ hiện tại PT**: `{now_pt.strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        f"- **Mốc Reset 00:00 PT kế tiếp**: `{next_pt.strftime('%Y-%m-%d 00:00:00 PT')}` (`{next_vn.strftime('%H:%M ngày %d/%m/%Y')} giờ VN`)",
        "",
        "| Model | Requests Hôm Nay (PT) | Hạn Mức Ngày (RPD) | Hạn Mức Còn Lại | Nguồn Hạn Mức | Trạng Thái |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    # Hiển thị tất cả model trong chuỗi models.yaml
    tracked_models = list(configured_models.keys())
    for mod in tracked_models:
        r_cnt = pt_reqs.get(mod, 0)
        mod_info = configured_models.get(mod, {})
        rpd_val = mod_info.get("rpd", "chưa biết")
        rpd_source = mod_info.get("rpd_source", "chưa xác minh")
        is_exhausted = cache_inst.is_model_exhausted(mod)

        rem_str = "—"
        if isinstance(rpd_val, int):
            rem_reqs = max(0, rpd_val - r_cnt)
            rem_str = f"**{rem_reqs}**"
            if rem_reqs == 0 or is_exhausted:
                st = f"🛑 Hết hạn mức (Khóa đến {next_vn.strftime('%H:%M %d/%m')})"
            else:
                st = "🟢 Khả dụng"
        else:
            rem_str = str(rpd_val)
            st = "🟢 Không giới hạn (Local)"

        lines.append(f"| `{mod}` | {r_cnt} | {rpd_val} | {rem_str} | {rpd_source} | {st} |")
    lines.append("")

    # Cảnh báo nếu model chính sắp hết
    primary_m = "gemini-3.8-flash"
    primary_reqs_used = pt_reqs.get(primary_m, 0)
    primary_rpd = configured_models.get(primary_m, {}).get("rpd", 20)
    if isinstance(primary_rpd, int) and (primary_rpd - primary_reqs_used) < 5:
        lines.extend([
            "> [!WARNING]",
            f"> **CẢNH BÁO HẠN MỨC GẦN HẾT**: Model `{primary_m}` chỉ còn **{max(0, primary_rpd - primary_reqs_used)} requests** trong ngày PT.",
            f"> Tập tiếp theo nếu thiếu hạn mức sẽ tự động kích hoạt chuyển sang `{models_cfg.get('chain', [])[1]}`.",
            ""
        ])

    # Section: Phân loại dữ liệu minh bạch
    lines.extend([
        "### 3. Phân Loại Dữ Liệu Báo Cáo (Data Transparency Breakdown)",
        "",
        "| Danh mục | Hạng mục dữ liệu | Trạng thái xác thực & Nguồn |",
        "| :--- | :--- | :--- |",
        f"| **SỐ ĐO (MEASURED)** | Số request thực tế trong tập: `{ep_reqs}` | Đo đếm trực tiếp qua SQLite `api_requests` |",
        f"| **SỐ ĐO (MEASURED)** | Số request hôm nay theo giờ PT | Bộ đếm theo ngày Thái Bình Dương trong DB |",
        f"| **SỐ ĐO (MEASURED)** | Tỷ lệ batch phải tách: `{split_rate:.1f}%` | Đo lường thực tế qua bộ kiểm tra ID & số cue |",
        f"| **SỐ ĐO (MEASURED)** | Thời gian chạy pipeline: `{elapsed_time_sec:.1f}s` | Đo đạc chính xác qua `time.monotonic()` |",
        f"| **ƯỚC LƯỢNG (ESTIMATED)** | Số request cho tập 300 cues (~3 requests) | Tính toán lý thuyết: 300 ÷ 100 cues/batch = 3 |",
        f"| **ƯỚC LƯỢNG (ESTIMATED)** | Thời điểm reset quota: `{next_vn.strftime('%H:%M %d/%m')}` | Tính toán qua `zoneinfo` (PDT 14:00 / PST 15:00) |",
        f"| **CHƯA XÁC MINH (UNVERIFIED)** | Hạn mức RPD = 20 (Flash) / 500 (Lite) | Chưa xác minh (nguồn bên thứ ba, chờ đối soát AI Studio) |",
        f"| **CHƯA XÁC MINH (UNVERIFIED)** | Giới hạn RPM/TPM của từng model | Đang để trống (chưa có số liệu chính thức từ dashboard) |",
        "",
    ])

    # Section: Cues generated by Local or Lite models (B4)
    local_or_lite_cues = [
        c for c in dialogue_cues
        if any(tag in (getattr(c, "model_used", "") or "").lower() for tag in ["local", "translategemma", "ollama", "lite", "flash-lite"])
    ]
    if local_or_lite_cues:
        lines.extend([
            f"### 📋 Danh sách câu thoại do Local / Lite sinh ra ({len(local_or_lite_cues)} cues)",
            "",
            "| ID | Time | Model | Japanese | Vietnamese |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for c in local_or_lite_cues[:50]: # Hiển thị tối đa 50 cues tiêu biểu
            t_str = format_timestamp(c.start)
            mod_str = getattr(c, 'model_used', 'N/A')
            lines.append(f"| {c.id} | {t_str} | `{mod_str}` | {c.ja} | {c.vi} |")
        if len(local_or_lite_cues) > 50:
            lines.append(f"*... và {len(local_or_lite_cues) - 50} câu khác.*")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 2. Quality Assurance Audit",
        "",
    ])

    # Section: Uncertain Cues
    if uncertain_cues:
        lines.append(f"### ❓ Uncertain Lines Flagged for Review ({len(uncertain_cues)} cues)")
        lines.append("| ID | Time | Japanese | English Context | Draft Vietnamese |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for c in uncertain_cues[:25]:
            t_str = format_timestamp(c.start)
            en_str = c.en or "—"
            lines.append(f"| {c.id} | {t_str} | {c.ja} | {en_str} | {c.vi} |")
        lines.append("")
    else:
        lines.append("### ✅ No Uncertain / Guesswork Lines Flagged\n")

    # Section: Leftover Japanese
    if leftover_japanese:
        lines.append(f"### ⚠️ Leftover Japanese Characters ({len(leftover_japanese)} cues)")
        lines.append("| ID | Time | Japanese | Vietnamese |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for c in leftover_japanese[:20]:
            t_str = format_timestamp(c.start)
            lines.append(f"| {c.id} | {t_str} | {c.ja} | {c.vi} |")
        lines.append("")
    else:
        lines.append("### ✅ No Leftover Japanese Characters Detected\n")

    # Section: Glossary Violations
    if glossary_violations:
        lines.append(f"### ⚠️ Potential Glossary Mismatches ({len(glossary_violations)} occurrences)")
        lines.append("| ID | Time | Expected Term | Found Text |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for c, term in glossary_violations[:20]:
            t_str = format_timestamp(c.start)
            lines.append(f"| {c.id} | {t_str} | `{term.ja} -> {term.vi}` | {c.vi} |")
        lines.append("")
    elif glossary and glossary.terms:
        lines.append("### ✅ All Detected Glossary Terms Match Expected Translations\n")

    # Section: Long lines audit
    if lines_over_107:
        lines.append(f"### ℹ️ Lines Exceeding 107 Characters ({len(lines_over_107)} cues)")
        lines.append("| ID | Time | Length | Text |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for c, l_len in lines_over_107[:15]:
            t_str = format_timestamp(c.start)
            lines.append(f"| {c.id} | {t_str} | {l_len} | {c.vi} |")
        lines.append("")

    # Section: Sample Dialogue Inspection
    lines.append("---")
    lines.append("## 3. Sample Dialogue Inspection (First 10 Translated Cues)")
    lines.append("")
    lines.append("| ID | Time | Japanese (Source) | English (Context) | Vietnamese (Final) | Backend |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for c in dialogue_cues[:10]:
        t_str = format_timestamp(c.start)
        en_str = c.en or "—"
        vi_str = (c.vi or "").replace("\n", " ").replace(r"\N", " ")
        lines.append(f"| {c.id} | {t_str} | {c.ja} | {en_str} | {vi_str} | {c.backend_used or 'gemini'} |")

    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Produce review file if requested
    if review_file and uncertain_cues:
        review_path = report_path.with_name(f"{output_subtitle_path.stem}.review.srt")
        review_blocks = []
        for idx, c in enumerate(uncertain_cues, start=1):
            t_start = format_timestamp_srt(c.start)
            t_end = format_timestamp_srt(c.end)
            review_blocks.append(
                f"{idx}\n{t_start} --> {t_end}\n[?] {c.vi}\n# JA: {c.ja}\n# EN: {c.en or '—'}"
            )
        review_path.write_text("\n\n".join(review_blocks) + "\n", encoding="utf-8")
        print(f"  • Review file generated: {review_path}")

    return report_path
