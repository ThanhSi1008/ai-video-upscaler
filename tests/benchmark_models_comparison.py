# -*- coding: utf-8 -*-
"""
tests/benchmark_models_comparison.py
Executes Requirement 6 of ADDENDUM 8:
Re-runs the 43-cue comparison across:
1. gemini-3.8-flash
2. gemini-3.5-flash-lite
3. gemini-2.5-flash
4. local translategemma:12b (Fair run: with Cast Sheet, +/-10 context, quote stripping)

Measures:
- Latency / execution time
- Output quote rate (should be 0% with regex)
- Uncertain flags
- Pronoun consistency on controversial lines
Saves raw outputs under out/compare_raw/
"""

import os
import sys
import time
import json
import re
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from subtrans.pipeline import TranslationPipeline
from subtrans.cache import DB_PATH
import sqlite3
import pysubs2

SAMPLE_JA = Path("subtrans/samples/sample_3min_jpn.ass")
SAMPLE_EN = Path("subtrans/samples/sample_3min_eng.ass")
GLOSSARY_PATH = Path("glossaries/glossary.mushoku.yaml")
OUT_RAW = Path("out/compare_raw")
OUT_RAW.mkdir(parents=True, exist_ok=True)


def clear_cache_for_model(model_name: str):
    """Deletes cached entries for a specific model to ensure true latency measurement."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM translations WHERE model = ?", (model_name,))
            conn.commit()
    except Exception as e:
        print(f"Warning clearing cache for {model_name}: {e}")


def run_benchmark():
    models_to_test = [
        {"name": "gemini-3.8-flash", "local": False, "out_sub": OUT_RAW / "gemini_38"},
        {"name": "gemini-3.5-flash-lite", "local": False, "out_sub": OUT_RAW / "gemini_35_lite"},
        {"name": "gemini-2.5-flash", "local": False, "out_sub": OUT_RAW / "gemini_25"},
        {"name": "translategemma:12b", "local": True, "out_sub": OUT_RAW / "local_12b"},
    ]

    results = []

    print("======================================================================")
    print("🚀 BẮT ĐẦU BENCHMARK SO SÁNH 4 MÔ HÌNH TRÊN 43 CUE (ADDENDUM 8)")
    print("======================================================================")

    for m_info in models_to_test:
        mod_name = m_info["name"]
        is_local = m_info["local"]
        out_dir = m_info["out_sub"]
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[RUNNING] Đang chạy mô hình: {mod_name} (Local: {is_local})...")
        clear_cache_for_model(mod_name)

        pipeline = TranslationPipeline(
            local_only=is_local,
            model_override=mod_name if not is_local else None,
            output_dir=out_dir,
            output_format="srt",
        )

        t0 = time.monotonic()
        run_res = pipeline.run(
            ja_path=SAMPLE_JA,
            en_path=SAMPLE_EN,
            glossary_path=GLOSSARY_PATH,
        )
        elapsed = time.monotonic() - t0

        srt_path = run_res["output_path"]
        # Copy to compare_raw root with model name for convenience
        target_copy = OUT_RAW / f"{mod_name.replace(':', '_')}.srt"
        target_copy.write_text(Path(srt_path).read_text(encoding="utf-8"), encoding="utf-8")

        # Load cues from SRT to inspect metrics
        subs = pysubs2.load(str(srt_path))
        dialogue_texts = [e.text.strip() for e in subs]

        # Audit metrics
        quoted_lines = sum(1 for t in dialogue_texts if t.startswith('"') or t.endswith('"') or t.startswith('“') or t.endswith('”'))
        uncertain_lines = run_res.get("stats", {}).get("uncertain", 0)
        # Parse uncertain lines from audit file
        audit_file = run_res.get("report_path")
        uncertain_count = 0
        if audit_file and Path(audit_file).exists():
            audit_text = Path(audit_file).read_text(encoding="utf-8")
            m_unc = re.search(r"Uncertain Lines Flagged \[.*?\]\s*\|\s*(\d+)", audit_text)
            if m_unc:
                uncertain_count = int(m_unc.group(1))

        # Check key controversial cues (1-indexed, so index = cue_id - 1)
        cue_10 = dialogue_texts[9] if len(dialogue_texts) >= 10 else ""  # 結ばれた
        cue_15_16 = " ".join(dialogue_texts[14:16]) if len(dialogue_texts) >= 16 else ""  # Rudeus thư
        cue_21 = dialogue_texts[20] if len(dialogue_texts) >= 21 else ""  # Cú đấm
        cue_42 = dialogue_texts[41] if len(dialogue_texts) >= 42 else ""  # Eris đi

        results.append({
            "model": mod_name,
            "is_local": is_local,
            "elapsed_seconds": round(elapsed, 2),
            "cues_per_second": round(len(dialogue_texts) / max(0.1, elapsed), 2),
            "quote_rate_percent": round(quoted_lines / len(dialogue_texts) * 100, 1),
            "uncertain_flags": uncertain_count,
            "raw_srt": str(target_copy),
            "stage_times": run_res.get("stage_times", {}),
            "sample_cues": {
                "cue_10": cue_10,
                "cue_15_16": cue_15_16,
                "cue_21": cue_21,
                "cue_42": cue_42,
            }
        })
        print(f"  • Hoàn thành trong {elapsed:.2f}s ({len(dialogue_texts)/max(0.1, elapsed):.2f} cues/s)!")

    # Summary Table
    print("\n======================================================================")
    print("📊 BẢNG TỔNG HỢP SO SÁNH 4 MÔ HÌNH (ADDENDUM 8 BENCHMARK)")
    print("======================================================================")
    headers = ["Model", "Time (s)", "Speed (cue/s)", "Quote Rate", "Uncertain [?]", "結ばれた (Cue 10)", "Xưng thư (Cue 15-16)", "Eris xưng (Cue 42)"]
    print(f"{'Model':<24} | {'Time':<8} | {'Speed':<10} | {'Quotes':<7} | {'Uncert':<7} | {'Cue 10':<30} | {'Cue 42'}")
    print("-" * 110)
    for r in results:
        m = r["model"]
        t = f"{r['elapsed_seconds']}s"
        spd = f"{r['cues_per_second']} c/s"
        q = f"{r['quote_rate_percent']}%"
        u = str(r['uncertain_flags'])
        c10 = (r["sample_cues"]["cue_10"][:28] + "..") if len(r["sample_cues"]["cue_10"]) > 28 else r["sample_cues"]["cue_10"]
        c42 = r["sample_cues"]["cue_42"]
        print(f"{m:<24} | {t:<8} | {spd:<10} | {q:<7} | {u:<7} | {c10:<30} | {c42}")

    # Write summary JSON and Markdown
    summary_path = OUT_RAW / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Đã lưu kết quả chi tiết tại: {summary_path}")


if __name__ == "__main__":
    run_benchmark()
