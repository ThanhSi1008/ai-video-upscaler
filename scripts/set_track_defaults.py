"""
Batch Track Configuration Utility for MKV Media
- Sets target audio (default: Japanese) to default (flag-default=1) and other audio tracks to flag-default=0.
- Sets target subtitle (default: Vietnamese or English full dialogue) to default (disabling Signs & Songs).
- Uses mkvpropedit for instantaneous in-place header modification.
- Supports CLI flags: targets, --audio-lang, --sub-lang, --skip-pattern, --dry-run, --mkvmerge, --mkvpropedit.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

# Default legacy constants for backward compatibility
raw_dir = r"E:\Films\Raw_Sources"
incomplete = "Fate.Strange.Fake"
mkvmerge_path = shutil.which("mkvmerge") or r"C:\Program Files\MKVToolNix\mkvmerge.exe"
mkvpropedit_path = shutil.which("mkvpropedit") or r"C:\Program Files\MKVToolNix\mkvpropedit.exe"


def resolve_tools(
    custom_mkvmerge: Optional[str] = None,
    custom_mkvpropedit: Optional[str] = None
) -> Tuple[str, str]:
    """Dynamically resolve mkvmerge and mkvpropedit paths with system and default fallbacks."""
    resolved_merge = (
        custom_mkvmerge
        or shutil.which("mkvmerge")
        or r"C:\Program Files\MKVToolNix\mkvmerge.exe"
    )
    resolved_propedit = (
        custom_mkvpropedit
        or shutil.which("mkvpropedit")
        or r"C:\Program Files\MKVToolNix\mkvpropedit.exe"
    )
    return resolved_merge, resolved_propedit


def is_japanese_audio(track: Dict[str, Any]) -> bool:
    """Check if audio track is Japanese."""
    props = track.get("properties", {}) or {}
    lang = props.get("language", "").lower()
    lang_ietf = props.get("language_ietf", "").lower()
    name = props.get("track_name", "").lower()
    if lang in ["jpn", "ja"] or lang_ietf.startswith("ja"):
        return True
    if any(k in name for k in ["japanese", "jpn", "nihongo"]):
        return True
    return False


def matches_audio_lang(track: Dict[str, Any], preferred_langs: List[str]) -> bool:
    """Check if audio track matches any of the preferred language codes/names."""
    props = track.get("properties", {}) or {}
    lang = props.get("language", "").lower()
    lang_ietf = props.get("language_ietf", "").lower()
    name = props.get("track_name", "").lower()

    for pref in preferred_langs:
        pref = pref.strip().lower()
        if not pref:
            continue
        if pref in ["jpn", "ja"]:
            if is_japanese_audio(track):
                return True
        elif pref in ["vie", "vi"]:
            if lang in ["vie", "vi"] or lang_ietf.startswith("vi") or any(k in name for k in ["vietnamese", "tiếng việt", "tieng viet"]):
                return True
        elif pref in ["eng", "en"]:
            if lang in ["eng", "en"] or lang_ietf.startswith("en") or any(k in name for k in ["english", "eng"]):
                return True
        else:
            if lang == pref or lang_ietf.startswith(pref) or pref in name:
                return True
    return False


def is_vietnamese_sub(track: Dict[str, Any]) -> bool:
    """Check if subtitle track is Vietnamese."""
    props = track.get("properties", {}) or {}
    lang = props.get("language", "").lower()
    lang_ietf = props.get("language_ietf", "").lower()
    name = props.get("track_name", "").lower()
    if lang in ["vie", "vi"] or lang_ietf.startswith("vi"):
        return True
    if any(k in name for k in ["vietnamese", "tiếng việt", "tieng viet"]):
        return True
    return False


def is_signs_or_songs(track: Dict[str, Any]) -> bool:
    """Check if subtitle track contains only signs or songs."""
    props = track.get("properties", {}) or {}
    name = props.get("track_name", "").lower()
    if any(k in name for k in ["s&s", "signs", "songs", "[signs]", "s/s"]):
        return True
    return False


def is_dialogue_sub(track: Dict[str, Any]) -> bool:
    """Check if subtitle track is full English dialogue."""
    if is_signs_or_songs(track):
        return False
    props = track.get("properties", {}) or {}
    name = props.get("track_name", "").lower()
    lang = props.get("language", "").lower()
    lang_ietf = props.get("language_ietf", "").lower()
    if lang in ["eng", "en"] or lang_ietf.startswith("en"):
        if "[cc]" in name:
            return False
        return True
    return False


def matches_sub_lang(track: Dict[str, Any], pref: str) -> bool:
    """Check if subtitle track matches a specific language preference."""
    pref = pref.strip().lower()
    if not pref:
        return False
    if pref in ["vie", "vi"]:
        return is_vietnamese_sub(track)
    elif pref in ["eng", "en"]:
        return is_dialogue_sub(track)
    else:
        props = track.get("properties", {}) or {}
        lang = props.get("language", "").lower()
        lang_ietf = props.get("language_ietf", "").lower()
        name = props.get("track_name", "").lower()
        if lang == pref or lang_ietf.startswith(pref) or pref in name:
            return not is_signs_or_songs(track)
    return False


def select_target_sub(sub_tracks: List[Dict[str, Any]], sub_langs: List[str]) -> Optional[Dict[str, Any]]:
    """Select the best matching subtitle track based on preference priority."""
    if not sub_tracks:
        return None

    # Check each preference in order
    for pref in sub_langs:
        pref = pref.strip().lower()
        if not pref:
            continue
        if pref in ["vie", "vi"]:
            vi_subs = [t for t in sub_tracks if is_vietnamese_sub(t)]
            if vi_subs:
                return vi_subs[0]
        elif pref in ["eng", "en"]:
            dialogue_subs = [t for t in sub_tracks if is_dialogue_sub(t)]
            if dialogue_subs:
                explicit_dialog = [t for t in dialogue_subs if "dialog" in t.get("properties", {}).get("track_name", "").lower()]
                return explicit_dialog[0] if explicit_dialog else dialogue_subs[0]
        else:
            cand = [t for t in sub_tracks if matches_sub_lang(t, pref)]
            if cand:
                return cand[0]

    # Fallback to any non-signs track, or first track
    non_signs = [t for t in sub_tracks if not is_signs_or_songs(t)]
    return non_signs[0] if non_signs else sub_tracks[0]


def process_file(
    fpath: str,
    mkvmerge_bin: Optional[str] = None,
    mkvpropedit_bin: Optional[str] = None,
    audio_langs: Optional[List[str]] = None,
    sub_langs: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Inspect and adjust default audio and subtitle tracks for an MKV file.
    When dry_run=True, simulates the changes without executing mkvpropedit.
    """
    merge_exe = mkvmerge_bin or mkvmerge_path
    propedit_exe = mkvpropedit_bin or mkvpropedit_path
    preferred_audio = audio_langs or ["ja", "jpn"]
    preferred_subs = sub_langs or ["vi", "vie", "en", "eng"]

    cmd = [merge_exe, "-J", fpath]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        return {"status": "error", "msg": f"Failed to inspect with mkvmerge: {res.stderr}"}

    try:
        data = json.loads(res.stdout)
    except Exception as e:
        return {"status": "error", "msg": f"JSON decode error: {e}"}

    tracks = data.get("tracks", [])
    audio_tracks = [t for t in tracks if t.get("type") == "audio"]
    sub_tracks = [t for t in tracks if t.get("type") == "subtitles"]

    propedit_args = []
    changes = []

    # 1. Audio selection
    if audio_tracks:
        matched_audios = [t for t in audio_tracks if matches_audio_lang(t, preferred_audio)]
        target_audio = None
        if matched_audios:
            default_audio = [t for t in matched_audios if t.get("properties", {}).get("default_track", False)]
            if default_audio:
                target_audio = default_audio[0]
            else:
                surround = [t for t in matched_audios if "5.1" in t.get("properties", {}).get("track_name", "").lower()]
                target_audio = surround[0] if surround else matched_audios[0]
        elif len(audio_tracks) == 1:
            target_audio = audio_tracks[0]

        if target_audio:
            target_uid = target_audio.get("properties", {}).get("uid")
            for t in audio_tracks:
                uid = t.get("properties", {}).get("uid")
                is_def = t.get("properties", {}).get("default_track", False)
                if uid == target_uid:
                    if not is_def:
                        propedit_args.extend(["--edit", f"track:={uid}", "--set", "flag-default=1"])
                        changes.append(f"Audio Target (UID={uid}) -> default=1")
                else:
                    if is_def:
                        propedit_args.extend(["--edit", f"track:={uid}", "--set", "flag-default=0"])
                        changes.append(f"Audio Non-Target (UID={uid}) -> default=0")

    # 2. Subtitle selection
    if sub_tracks:
        target_sub = select_target_sub(sub_tracks, preferred_subs)
        if target_sub:
            target_sub_uid = target_sub.get("properties", {}).get("uid")
            for t in sub_tracks:
                uid = t.get("properties", {}).get("uid")
                is_def = t.get("properties", {}).get("default_track", False)
                if uid == target_sub_uid:
                    if not is_def:
                        propedit_args.extend(["--edit", f"track:={uid}", "--set", "flag-default=1"])
                        changes.append(f"Sub (UID={uid}, '{target_sub.get('properties',{}).get('track_name','')}') -> default=1")
                else:
                    if is_def:
                        propedit_args.extend(["--edit", f"track:={uid}", "--set", "flag-default=0"])
                        changes.append(f"Sub (UID={uid}, '{t.get('properties',{}).get('track_name','')}') -> default=0")

    if not propedit_args:
        return {"status": "unchanged", "changes": []}

    if dry_run:
        return {"status": "simulated", "changes": changes, "dry_run": True}

    cmd_edit = [propedit_exe, fpath] + propedit_args
    res_edit = subprocess.run(cmd_edit, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res_edit.returncode != 0:
        return {"status": "error", "msg": f"mkvpropedit failed: {res_edit.stderr}"}

    return {"status": "modified", "changes": changes}


def collect_files(targets: List[str], skip_patterns: List[str]) -> List[str]:
    """Collect all target MKV files from file paths or directories, filtering out skip patterns."""
    all_files = []
    for target in targets:
        if os.path.isfile(target):
            if target.lower().endswith(".mkv"):
                fname = os.path.basename(target)
                if any(p.lower() in fname.lower() for p in skip_patterns):
                    print(f"[SKIP] Matches skip pattern: {fname}")
                    continue
                all_files.append(os.path.abspath(target))
        elif os.path.isdir(target):
            for root, dirs, fnames in os.walk(target):
                for f in fnames:
                    if f.lower().endswith(".mkv"):
                        if any(p.lower() in f.lower() for p in skip_patterns):
                            print(f"[SKIP] Matches skip pattern: {f}")
                            continue
                        all_files.append(os.path.abspath(os.path.join(root, f)))
        else:
            print(f"[WARN] Target path not found: {target}")
    return sorted(list(set(all_files)))


def _build_arg_parser(targets_default: Any = argparse.SUPPRESS) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch Track Configuration Utility for MKV Media"
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=targets_default,
        help="Files or directories to process (requires explicit target, --dry-run, or --yes-default-target)",
    )
    parser.add_argument(
        "--audio-lang",
        default="ja,jpn",
        help="Comma-separated audio language priority (default: ja,jpn)",
    )
    parser.add_argument(
        "--sub-lang",
        default="vi,vie,en,eng",
        help="Comma-separated subtitle language priority (default: vi,vie,en,eng)",
    )
    parser.add_argument(
        "--skip-pattern",
        action="append",
        default=None,
        help="Patterns to skip (can be specified multiple times; default: Fate.Strange.Fake)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate modifications without modifying files",
    )
    parser.add_argument(
        "--yes-default-target",
        action="store_true",
        help="Explicitly allow processing the default E:\\Films\\Raw_Sources target when no targets are passed",
    )
    parser.add_argument(
        "--mkvmerge",
        default=None,
        help="Path to mkvmerge executable",
    )
    parser.add_argument(
        "--mkvpropedit",
        default=None,
        help="Path to mkvpropedit executable",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for track configuration."""
    return _build_arg_parser(targets_default=[raw_dir]).parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    raw_args = _build_arg_parser(targets_default=argparse.SUPPRESS).parse_args(argv)
    has_explicit_target = hasattr(raw_args, "targets") and bool(raw_args.targets)
    if not has_explicit_target and not raw_args.dry_run and not raw_args.yes_default_target:
        print(
            "Error: No target path provided. Pass explicit target path(s), --dry-run, or --yes-default-target "
            f"to process default directory ({raw_dir}).",
            file=sys.stderr,
        )
        return 2

    targets = list(raw_args.targets) if has_explicit_target else [raw_dir]
    mkvmerge_bin, mkvpropedit_bin = resolve_tools(raw_args.mkvmerge, raw_args.mkvpropedit)
    skip_patterns = raw_args.skip_pattern if raw_args.skip_pattern is not None else ["Fate.Strange.Fake"]
    audio_langs = [x.strip().lower() for x in raw_args.audio_lang.split(",") if x.strip()]
    sub_langs = [x.strip().lower() for x in raw_args.sub_lang.split(",") if x.strip()]

    all_files = collect_files(targets, skip_patterns)
    mode_tag = " [DRY-RUN SIMULATION]" if raw_args.dry_run else ""
    print(f"Total completed MKV files to process{mode_tag}: {len(all_files)}")
    start_time = time.time()

    modified_count = 0
    unchanged_count = 0
    error_count = 0
    by_folder_stats: Dict[str, Dict[str, int]] = {}

    for idx, fpath in enumerate(all_files, 1):
        folder_key = os.path.dirname(fpath)
        fname = os.path.basename(fpath)
        stats = by_folder_stats.setdefault(folder_key, {"modified": 0, "unchanged": 0, "error": 0})

        result = process_file(
            fpath,
            mkvmerge_bin=mkvmerge_bin,
            mkvpropedit_bin=mkvpropedit_bin,
            audio_langs=audio_langs,
            sub_langs=sub_langs,
            dry_run=raw_args.dry_run,
        )
        if result["status"] in ("modified", "simulated"):
            modified_count += 1
            stats["modified"] += 1
            prefix = "[DRY-RUN] " if raw_args.dry_run else ""
            print(f"{prefix}Applied changes to {fname}: {', '.join(result.get('changes', []))}")
        elif result["status"] == "unchanged":
            unchanged_count += 1
            stats["unchanged"] += 1
        else:
            error_count += 1
            stats["error"] += 1
            print(f"ERROR on {fname}: {result.get('msg')}")

        if idx % 50 == 0 or idx == len(all_files):
            elapsed = time.time() - start_time
            print(f"Progress: [{idx}/{len(all_files)}] files processed ({elapsed:.1f}s) - Modified: {modified_count}, Unchanged: {unchanged_count}, Errors: {error_count}")

    total_elapsed = time.time() - start_time
    print("\n==========================================")
    print(f"SUMMARY{mode_tag} (Completed in {total_elapsed:.1f}s)")
    print(f"Total files: {len(all_files)}")
    print(f"Modified/Simulated: {modified_count}")
    print(f"Unchanged: {unchanged_count}")
    print(f"Errors: {error_count}")
    print("==========================================")
    for folder, s in sorted(by_folder_stats.items()):
        print(f"  {folder}: {s['modified']} modified, {s['unchanged']} unchanged, {s['error']} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
