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
from typing import Optional

PROJECT_ROOT = Path(r"E:\ai-video-upscaler")
PYTHON_BIN = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
UPSCALE_PY = PROJECT_ROOT / "upscale.py"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_FILE = LOGS_DIR / "batch_translation_state.json"
SUB_OUT_DIR = Path(r"E:\Films\Subtitles")
RAW_SOURCES_DIR = Path(r"E:\Films\Raw_Sources")
INCOMPLETE_SUBSTRING = "fate.strange.fake"

SKIP_FILENAME_PATTERNS = ["nced", "ncop", "trailer", "sample", "pv", "special ending", ".temp_"]
SKIP_DIR_PATTERNS = [r"\other", "/other", r"\trailers", "/trailers"]

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
        if any(d in root.lower() for d in SKIP_DIR_PATTERNS):
            continue
        for f in files:
            if not f.lower().endswith(".mkv"):
                continue
            if INCOMPLETE_SUBSTRING in f.lower():
                continue
            if any(k in f.lower() for k in SKIP_FILENAME_PATTERNS):
                continue
            all_files.append(os.path.join(root, f))
            
    def sort_order(p):
        rel = os.path.relpath(p, str(RAW_SOURCES_DIR))
        parts = rel.split(os.sep)
        folder = parts[0].lower()
        subfolder = parts[1].lower() if len(parts) > 1 else ""
        
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
            
        sub_priority = 10
        if "special" in subfolder:
            sub_priority = 20
            
        return (priority, folder, sub_priority, natural_sort_key(p))

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

QUOTA_EXHAUSTION_MARKERS = (
    "cạn kiệt hạn mức ngày",
    "toàn bộ các model gemini trong chuỗi đã hết hạn mức",
    "dailyquotaexhaustederror",
)

PARTIAL_SUB_MARKERS = (
    "chưa đủ điều kiện nhúng",
    "chưa đạt chuẩn nhúng",
)


def is_daily_quota_exhausted_output(output_text: str) -> bool:
    low = (output_text or "").lower()
    return any(m in low for m in QUOTA_EXHAUSTION_MARKERS)


def is_partial_subtitle_output(output_text: str) -> bool:
    low = (output_text or "").lower()
    return any(m in low for m in PARTIAL_SUB_MARKERS)


def prefetch_episode_stage_a(
    fpath: str,
    output_dir: Path = SUB_OUT_DIR,
    subs_quality: str = "normal",
    dry_run: bool = False,
    skip_if_series_cast_missing: bool = False,
) -> dict:
    """
    Background Stage A prefetch (text subtitles + Cast Sheet) for episode N+1
    while episode N finishes final translation/muxing:
    - Never invokes Whisper STT (`prefer_whisper_when_no_ja=False`) to avoid CPU/GPU contention.
    - Writes `<base_name>.cast.yaml` into `output_dir` so episode N+1 reads it in ~0 ms.
    """
    vid_dir = os.path.dirname(os.path.abspath(fpath))
    base_name = os.path.splitext(os.path.basename(fpath))[0]
    extracted_used = False
    cleanup_fn = None
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.cleanup import cleanup_subtitle_extracts
        from src.subtitles import find_or_extract_subtitles
        from subtrans.cache import TranslationCache
        from subtrans.cast_sheet import (
            extract_series_name,
            get_series_cast_path,
            load_or_generate_series_cast_sheet,
        )
        from subtrans.config import load_config, load_models_config
        from subtrans.parser import parse_and_align_subtitles

        cleanup_fn = cleanup_subtitle_extracts
        audit_cast = output_dir / "audit" / f"{base_name}.cast.yaml"
        direct_cast = output_dir / f"{base_name}.cast.yaml"
        if audit_cast.exists() or direct_cast.exists():
            return {"status": "ALREADY_READY", "episode": base_name}

        if skip_if_series_cast_missing:
            series_name = extract_series_name(base_name)
            series_cast = get_series_cast_path(series_name, output_dir)
            audit_series_cast = output_dir / "audit" / series_cast.name
            if not series_cast.exists() and not audit_series_cast.exists():
                return {"status": "SKIPPED_WAIT_SERIES_CAST", "episode": base_name}

        extracted_used = True
        ja_sub, en_sub = find_or_extract_subtitles(fpath, prefer_whisper_when_no_ja=False)
        if not ja_sub:
            return {"status": "SKIPPED_NO_TEXT_SUB", "episode": base_name}

        en_arg = None if ja_sub == en_sub else en_sub
        cues, _, _ = parse_and_align_subtitles(
            ja_path=str(ja_sub),
            en_path=str(en_arg) if en_arg else None,
            auto_align_whisper_to_en=False,
        )
        if not cues:
            return {"status": "SKIPPED_EMPTY_CUES", "episode": base_name}

        cfg = load_config()
        models_cfg = load_models_config()
        raw_chain = models_cfg.get("chain", ["gemini-3.5-flash-lite", "gemini-3.8-flash"])
        gemini_chain = [m for m in raw_chain if m != "local" and "preview" not in m.lower()]
        lite_models = [m for m in gemini_chain if "lite" in m.lower()]
        flash_models = [m for m in gemini_chain if "lite" not in m.lower()]
        ordered = (flash_models + lite_models) if subs_quality == "high" else (lite_models + flash_models)
        primary_model = ordered[0] if ordered else "gemini-3.5-flash-lite"
        fallbacks = [m for m in ordered if m != primary_model]
        timeout_ms = int(cfg.get("gemini", {}).get("request_timeout_ms", 45000))

        output_dir.mkdir(parents=True, exist_ok=True)
        cache_inst = TranslationCache()
        cast_glossary = load_or_generate_series_cast_sheet(
            cues=cues,
            source_name=base_name,
            output_dir=output_dir,
            episode_cast_path=direct_cast,
            gemini_model=primary_model,
            fallback_models=fallbacks,
            timeout_ms=timeout_ms,
            dry_run=dry_run,
            cache=cache_inst,
        )
        chars_n = len(cast_glossary.characters) if (cast_glossary and cast_glossary.characters) else 0
        return {"status": "PREFETCHED", "episode": base_name, "characters": chars_n}
    except Exception as e:
        return {"status": "ERROR", "episode": os.path.basename(fpath), "error": str(e)}
    finally:
        if extracted_used and cleanup_fn is not None:
            try:
                cleanup_fn(vid_dir, base_name)
            except Exception:
                pass


class StageAPrefetchWorker:
    """Single-worker background Stage A prefetcher for episode N+1; cancels cleanly on quota exhaustion."""

    def __init__(self, prefetch_fn=None):
        from concurrent.futures import ThreadPoolExecutor
        import threading

        self._prefetch_fn = prefetch_fn or prefetch_episode_stage_a
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage_a_prefetch")
        self._cancel_event = threading.Event()
        self._futures: dict = {}

    def submit_next(
        self,
        next_fpath: str,
        current_fpath: Optional[str] = None,
        output_dir: Path = SUB_OUT_DIR,
        subs_quality: str = "normal",
        same_series_headstart_sec: float = 8.0,
    ):
        if not next_fpath or next_fpath in self._futures or self._cancel_event.is_set():
            return None

        same_folder = False
        if current_fpath:
            same_folder = os.path.dirname(os.path.abspath(current_fpath)) == os.path.dirname(os.path.abspath(next_fpath))

        def _task():
            if same_folder:
                if same_series_headstart_sec > 0:
                    # Give episode N time to emit <series>.cast.yaml before prefetching same-series episodes
                    if self._cancel_event.wait(timeout=same_series_headstart_sec):
                        return {"status": "CANCELLED"}
                if self._cancel_event.is_set():
                    return {"status": "CANCELLED"}
                sys.path.insert(0, str(PROJECT_ROOT))
                from subtrans.cast_sheet import extract_series_name, get_series_cast_path

                next_base = os.path.splitext(os.path.basename(next_fpath))[0]
                series_name = extract_series_name(next_base)
                series_cast = get_series_cast_path(series_name, output_dir)
                audit_series_cast = output_dir / "audit" / series_cast.name
                if not series_cast.exists() and not audit_series_cast.exists():
                    return {"status": "SKIPPED_WAIT_SERIES_CAST", "episode": next_base}
            if self._cancel_event.is_set():
                return {"status": "CANCELLED"}
            return self._prefetch_fn(next_fpath, output_dir=output_dir, subs_quality=subs_quality)

        fut = self._executor.submit(_task)
        self._futures[next_fpath] = fut
        return fut

    def wait_if_inflight(self, fpath: str, timeout: float = 20.0) -> Optional[dict]:
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        fut = self._futures.pop(fpath, None)
        if fut is None:
            return None
        try:
            return fut.result(timeout=timeout)
        except (TimeoutError, FuturesTimeoutError):
            if fut.cancel():
                return None
            # Do not swallow TimeoutError while worker is still writing files; wait to prevent races.
            try:
                return fut.result()
            except Exception:
                return None
        except Exception:
            return None

    def shutdown(self, cancel: bool = False) -> None:
        if cancel:
            self._cancel_event.set()
            for fut in self._futures.values():
                fut.cancel()
        self._executor.shutdown(wait=not cancel)


def _has_external_vi_output(fpath: str, out_dir: Path, started_at: float) -> bool:
    base_name = os.path.splitext(os.path.basename(fpath))[0]
    vid_dir = Path(os.path.abspath(fpath)).parent
    candidates = [
        vid_dir / f"{base_name}.vi.srt",
        out_dir / f"{base_name}.vi.srt",
        out_dir / f"{base_name}.srt",
    ]
    for cand in candidates:
        try:
            if cand.exists() and cand.stat().st_size > 0 and cand.stat().st_mtime >= (started_at - 1.0):
                return True
        except Exception:
            pass
    return False


def run_translation_for_file(
    fpath: str,
    max_retries: int = 3,
    auto_offset: bool = True,
    no_embed: bool = False,
    prefer_whisper: bool = False,
    subs_quality: str = "normal",
) -> dict:
    cmd = [
        str(PYTHON_BIN),
        str(UPSCALE_PY),
        "--subs-only", fpath,
        "-o", str(SUB_OUT_DIR),
        "--subs-quality", subs_quality,
    ]
    if auto_offset:
        cmd.append("--auto-offset")
    if no_embed:
        cmd.append("--no-embed")
    if prefer_whisper:
        cmd.append("--prefer-whisper")

    last_err = ""
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.time() - t0
        combined_out = f"{res.stdout or ''}\n{res.stderr or ''}"
        low_out = combined_out.lower()

        if res.returncode == 0 and (no_embed or has_vietnamese_sub(fpath)):
            return {
                "status": "SUCCESS",
                "duration_seconds": round(elapsed, 1),
                "attempt": attempt,
                "error": None
            }

        if is_daily_quota_exhausted_output(combined_out):
            return {
                "status": "QUOTA_EXHAUSTED",
                "duration_seconds": round(elapsed, 1),
                "attempt": attempt,
                "error": "All Gemini models exhausted daily RPD quota; stopping queue until 00:00 PT reset.",
            }

        if res.returncode == 0 and is_partial_subtitle_output(combined_out):
            return {
                "status": "PARTIAL_PENDING",
                "duration_seconds": round(elapsed, 1),
                "attempt": attempt,
                "error": "Subtitle emitted as .partial.srt (saved to pending queue; skipping retry).",
            }

        translation_done = (
            "không thể nhúng phụ đề vào container" in low_out
            or "file phụ đề ngoài vẫn sẵn sàng" in low_out
            or "hoàn tất dịch phụ đề độc lập" in low_out
            or _has_external_vi_output(fpath, SUB_OUT_DIR, t0)
        )
        if res.returncode == 0 and translation_done:
            return {
                "status": "EMBED_FAILED",
                "duration_seconds": round(elapsed, 1),
                "attempt": attempt,
                "error": "Translation succeeded (sidecar available) but MKV embed/audit failed; skipping re-translation.",
            }

        last_err = res.stderr[-500:] if res.stderr else res.stdout[-500:]
        print(f"[ATTEMPT {attempt}/{max_retries} FAILED] {os.path.basename(fpath)}: {last_err.strip()[-200:]}", flush=True)
        if attempt < max_retries:
            time.sleep(10 * attempt)

    return {
        "status": "FAILED",
        "duration_seconds": 0,
        "attempt": max_retries,
        "error": last_err.strip()
    }

def build_cli_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Batch Subtitle Translation Orchestrator for Raw Sources")
    parser.add_argument("--no-auto-offset", action="store_true", help="Disable automatic JA-EN subtitle offset detection")
    parser.add_argument("--no-embed", action="store_true", help="Do not embed subtitles into source MKV containers")
    parser.add_argument("--prefer-whisper", action="store_true", help="Prefer Whisper STT + DTW when JA track is missing")
    parser.add_argument("--subs-quality", choices=["normal", "fast", "high"], default="normal", help="Translation quality tier")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries on transient network failures")
    parser.add_argument("--no-prefetch", action="store_true", help="Disable Stage A Cast Sheet prefetching for episode N+1")
    return parser


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = build_cli_parser().parse_args(argv)
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
    quality_mode = "normal" if args.subs_quality == "fast" else args.subs_quality
    prefetcher = None if args.no_prefetch else StageAPrefetchWorker()

    def _find_next_eligible(start_zero_idx: int) -> str | None:
        for cand_p in queue[start_zero_idx:]:
            cand_rel = os.path.relpath(cand_p, str(RAW_SOURCES_DIR))
            if cand_rel in state["history"] and state["history"][cand_rel].get("status") == "SUCCESS":
                continue
            if not args.no_embed and has_vietnamese_sub(cand_p):
                continue
            return cand_p
        return None

    try:
        for idx, fpath in enumerate(queue, 1):
            fname = os.path.basename(fpath)
            rel_path = os.path.relpath(fpath, str(RAW_SOURCES_DIR))

            # Check if already processed in state or already has VI sub
            if rel_path in state["history"] and state["history"][rel_path].get("status") == "SUCCESS":
                print(f"[{idx}/{len(queue)}] [SKIP] Already translated: {rel_path}", flush=True)
                continue

            if not args.no_embed and has_vietnamese_sub(fpath):
                print(f"[{idx}/{len(queue)}] [SKIP] Container already has VI subtitle: {rel_path}", flush=True)
                state["history"][rel_path] = {
                    "status": "SUCCESS",
                    "duration_seconds": 0,
                    "timestamp": time.time(),
                    "note": "Pre-existing VI track"
                }
                state["skipped"] += 1
                save_state(state)
                continue

            if prefetcher is not None:
                pf_res = prefetcher.wait_if_inflight(fpath)
                if pf_res and pf_res.get("status") == "PREFETCHED":
                    print(
                        f"  [STAGE-A PREFETCH HIT] Prepared cast sheet ({pf_res.get('characters', 0)} chars) for '{fname}'.",
                        flush=True,
                    )
                next_ep = _find_next_eligible(idx)
                if next_ep:
                    prefetcher.submit_next(
                        next_ep,
                        current_fpath=fpath,
                        output_dir=SUB_OUT_DIR,
                        subs_quality=quality_mode,
                    )

            print(f"\n[{idx}/{len(queue)}] Translating: {rel_path}...", flush=True)
            res = run_translation_for_file(
                fpath,
                max_retries=args.max_retries,
                auto_offset=not args.no_auto_offset,
                no_embed=args.no_embed,
                prefer_whisper=args.prefer_whisper,
                subs_quality=quality_mode,
            )

            state["history"][rel_path] = {
                "status": res["status"],
                "duration_seconds": res.get("duration_seconds", 0),
                "timestamp": time.time(),
                "error": res.get("error")
            }

            if res["status"] == "SUCCESS":
                state["completed"] += 1
                print(f"[{idx}/{len(queue)}] DONE ({res['duration_seconds']}s): {fname}", flush=True)
            elif res["status"] == "QUOTA_EXHAUSTED":
                if prefetcher is not None:
                    prefetcher.shutdown(cancel=True)
                    prefetcher = None
                print(
                    f"\n[{idx}/{len(queue)}] DAILY RPD QUOTA EXHAUSTED across Gemini chain on '{fname}'.\n"
                    "   - Saved partial translations in SQLite cache & state/pending_subs.json.\n"
                    "   - Stopping queue cleanly until 00:00 PT (14:00-15:00 VN) reset.",
                    flush=True,
                )
                save_state(state)
                break
            elif res["status"] == "PARTIAL_PENDING":
                state["failed"] += 1
                print(f"[{idx}/{len(queue)}] PARTIAL SAVED: {fname} ({res.get('error')})", flush=True)
            elif res["status"] == "EMBED_FAILED":
                state["failed"] += 1
                print(f"[{idx}/{len(queue)}] TRANSLATED (EMBED FAILED): {fname} ({res.get('error')})", flush=True)
            else:
                state["failed"] += 1
                print(f"[{idx}/{len(queue)}] FAILED: {fname} (Error: {res.get('error')})", flush=True)

            save_state(state)
            time.sleep(2)
    finally:
        if prefetcher is not None:
            prefetcher.shutdown(cancel=False)

    total_done = state["completed"] + state["skipped"]
    print("\n" + "=" * 80, flush=True)
    print("BATCH QUEUE COMPLETED", flush=True)
    print(f"Total: {state['total']} | Completed: {total_done} | New: {state['completed']} | Skipped: {state['skipped']} | Failed: {state['failed']}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
