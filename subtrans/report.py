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
    if model_counts:
        lines.extend([
            "### Model Usage Breakdown",
            "",
            "| Model / Source | Cues Translated | Share |",
            "| :--- | :--- | :--- |",
        ])
        for mod, cnt in model_counts.items():
            lines.append(f"| `{mod}` | {cnt} | {cnt/max(1, len(dialogue_cues))*100:.1f}% |")
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
