"""
Batch Subtitle Translation Orchestrator for Raw Sources
- Sequentially translates anime episodes using TriSub AI (upscale.py --subs-only)
- Idempotent: Skips files that already have an embedded Vietnamese subtitle track
- Fault-tolerant: Retries on temporary API glitches, logs errors, and continues next file
- Tracks progress in logs/batch_translation_state.json
"""

import os
import sys
import re
import json
import time
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(r"E:\ai-video-upscaler")
PYTHON_BIN = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
UPSCALE_PY = PROJECT_ROOT / "upscale.py"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = LOGS_DIR / "batch_translation_state.json"
SUB_OUT_DIR = Path(r"E:\Films\Subtitles")
RAW_SOURCES_DIR = Path(r"E:\Films\Raw_Sources")
INCOMPLETE_SUBSTRING = "fate.strange.fake"

SKIP_PATTERNS = ["nced", "ncop", "trailer", "sample", "pv", ".temp_"]

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def has_vietnamese_sub(fpath: str) -> bool:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index:stream_tags=language,title",
            "-select_streams", "s",
            "-of", "json",
            fpath
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            return False
        data = json.loads(res.stdout)
        for s in data.get("streams", []):
            lang = s.get("tags", {}).get("language", "").lower()
            title = s.get("tags", {}).get("title", "").lower()
            if lang in ("vie", "vi") or "vietnamese" in title or "tiếng việt" in title or "tieng viet" in title:
                return True
        return False
    except Exception:
        return False

def collect_queue() -> list:
    all_files = []
    for root, dirs, files in os.walk(str(RAW_SOURCES_DIR)):
        for f in files:
            if not f.lower().endswith(".mkv"):
                continue
            if INCOMPLETE_SUBSTRING in f.lower():
                continue
            if any(k in f.lower() for k in SKIP_PATTERNS):
                continue
            all_files.append(os.path.join(root, f))
            
    # Sort by relative folder and then natural sort by filename
    def sort_order(p):
        rel = os.path.relpath(p, str(RAW_SOURCES_DIR))
        parts = rel.split(os.sep)
        # Order priorities for series
        folder = parts[0].lower()
        priority = 50
        if "frieren" in folder:
            priority = 10
        elif "fate" in folder:
            priority = 20
        elif "violet" in folder:
            priority = 30
        elif "jujutsu" in folder:
            priority = 40
        elif "demon" in folder or "kimetsu" in folder:
            priority = 60
        return (priority, folder, natural_sort_key(p))

    all_files.sort(key=sort_order)
    return all_files

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "start_time": time.time(),
        "total": 0,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "history": {}
    }

def save_state(state: dict):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = STATE_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    temp_file.replace(STATE_FILE)

def run_translation_for_file(fpath: str, max_retries: int = 3) -> dict:
    cmd = [
        str(PYTHON_BIN),
        str(UPSCALE_PY),
        "--subs-only", fpath,
        "-o", str(SUB_OUT_DIR),
        "--subs-quality", "normal"
    ]
    
    last_err = ""
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.time() - t0
        
        if res.returncode == 0 and has_vietnamese_sub(fpath):
            return {
                "status": "SUCCESS",
                "duration_seconds": round(elapsed, 1),
                "attempt": attempt,
                "error": None
            }
            
        last_err = res.stderr[-500:] if res.stderr else res.stdout[-500:]
        print(f"⚠️ [ATTEMPT {attempt}/{max_retries} FAILED] {os.path.basename(fpath)}: {last_err.strip()[-200:]}", flush=True)
        if attempt < max_retries:
            time.sleep(10 * attempt)
            
    return {
        "status": "FAILED",
        "duration_seconds": 0,
        "attempt": max_retries,
        "error": last_err.strip()
    }

def main():
    print("=" * 80, flush=True)
    print("🎬 TRI-SUB AI BATCH TRANSLATOR - SEQUENTIAL RUNNER", flush=True)
    print(f"📁 Target Folder: {RAW_SOURCES_DIR}", flush=True)
    print(f"📁 Output Subtitles: {SUB_OUT_DIR}", flush=True)
    print("=" * 80, flush=True)

    SUB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    queue = collect_queue()
    state = load_state()
    state["total"] = len(queue)
    save_state(state)

    print(f"📋 Total episodes in queue: {len(queue)}", flush=True)
    
    for idx, fpath in enumerate(queue, 1):
        fname = os.path.basename(fpath)
        rel_path = os.path.relpath(fpath, str(RAW_SOURCES_DIR))
        
        # Check if already processed in state or already has VI sub
        if rel_path in state["history"] and state["history"][rel_path].get("status") == "SUCCESS":
            print(f"[{idx}/{len(queue)}] ⏭️ ĐÃ DỊCH XONG TRƯỚC ĐÓ: {rel_path}", flush=True)
            continue
            
        if has_vietnamese_sub(fpath):
            print(f"[{idx}/{len(queue)}] ⏭️ CONTAINER ĐÃ CÓ PHỤ ĐỀ TIẾNG VIỆT: {rel_path}", flush=True)
            state["history"][rel_path] = {
                "status": "SUCCESS",
                "duration_seconds": 0,
                "timestamp": time.time(),
                "note": "Pre-existing VI track"
            }
            state["skipped"] += 1
            save_state(state)
            continue

        print(f"\n[{idx}/{len(queue)}] 🚀 BẮT ĐẦU DỊCH: {rel_path}...", flush=True)
        res = run_translation_for_file(fpath)
        
        state["history"][rel_path] = {
            "status": res["status"],
            "duration_seconds": res.get("duration_seconds", 0),
            "timestamp": time.time(),
            "error": res.get("error")
        }
        
        if res["status"] == "SUCCESS":
            state["completed"] += 1
            print(f"[{idx}/{len(queue)}] ✅ HOÀN THÀNH ({res['duration_seconds']}s): {fname}", flush=True)
        else:
            state["failed"] += 1
            print(f"[{idx}/{len(queue)}] ❌ THẤT BẠI: {fname} (Lỗi: {res.get('error')})", flush=True)
            
        save_state(state)
        # Small cooldown between episodes to stay safely within rate limits
        time.sleep(2)

    total_done = state["completed"] + state["skipped"]
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 HOÀN TẤT TOÀN BỘ HÀNG ĐỢI!", flush=True)
    print(f"📊 Tổng số tập: {state['total']} | Đã hoàn tất: {total_done} | Thành công mới: {state['completed']} | Bỏ qua (đã có): {state['skipped']} | Thất bại: {state['failed']}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
