"""
Batch Track Configuration Utility for MKV Media
- Sets Japanese audio to default (flag-default=1) and other audio tracks to flag-default=0.
- Sets Vietnamese subtitle to default if available, or activates full dialogue track (disabling Signs & Songs).
- Uses mkvpropedit for instantaneous in-place header modification.
"""

import os
import sys
import json
import subprocess
import time

sys.stdout.reconfigure(encoding='utf-8')

raw_dir = r'E:\Films\Raw_Sources'
incomplete = 'Fate.Strange.Fake'
mkvmerge_path = r'C:\Program Files\MKVToolNix\mkvmerge.exe'
mkvpropedit_path = r'C:\Program Files\MKVToolNix\mkvpropedit.exe'

def is_japanese_audio(track):
    props = track.get('properties', {})
    lang = props.get('language', '').lower()
    lang_ietf = props.get('language_ietf', '').lower()
    name = props.get('track_name', '').lower()
    if lang in ['jpn', 'ja'] or lang_ietf.startswith('ja'):
        return True
    if any(k in name for k in ['japanese', 'jpn', 'nihongo']):
        return True
    return False

def is_vietnamese_sub(track):
    props = track.get('properties', {})
    lang = props.get('language', '').lower()
    lang_ietf = props.get('language_ietf', '').lower()
    name = props.get('track_name', '').lower()
    if lang in ['vie', 'vi'] or lang_ietf.startswith('vi'):
        return True
    if any(k in name for k in ['vietnamese', 'tiếng việt', 'tieng viet']):
        return True
    return False

def is_signs_or_songs(track):
    props = track.get('properties', {})
    name = props.get('track_name', '').lower()
    if any(k in name for k in ['s&s', 'signs', 'songs', '[signs]', 's/s']):
        return True
    return False

def is_dialogue_sub(track):
    if is_signs_or_songs(track):
        return False
    props = track.get('properties', {})
    name = props.get('track_name', '').lower()
    lang = props.get('language', '').lower()
    lang_ietf = props.get('language_ietf', '').lower()
    if lang in ['eng', 'en'] or lang_ietf.startswith('en'):
        if '[cc]' in name:
            return False
        return True
    return False

def process_file(fpath):
    cmd = [mkvmerge_path, '-J', fpath]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if res.returncode != 0:
        return {'status': 'error', 'msg': 'Failed to inspect with mkvmerge'}
    
    try:
        data = json.loads(res.stdout)
    except Exception as e:
        return {'status': 'error', 'msg': f'JSON decode error: {e}'}
    
    tracks = data.get('tracks', [])
    audio_tracks = [t for t in tracks if t.get('type') == 'audio']
    sub_tracks = [t for t in tracks if t.get('type') == 'subtitles']
    
    propedit_args = []
    changes = []
    
    # 1. Audio selection
    if audio_tracks:
        ja_audios = [t for t in audio_tracks if is_japanese_audio(t)]
        target_ja = None
        if ja_audios:
            default_ja = [t for t in ja_audios if t.get('properties', {}).get('default_track', False)]
            if default_ja:
                target_ja = default_ja[0]
            else:
                surround_ja = [t for t in ja_audios if '5.1' in t.get('properties', {}).get('track_name', '').lower()]
                target_ja = surround_ja[0] if surround_ja else ja_audios[0]
        elif len(audio_tracks) == 1:
            target_ja = audio_tracks[0]
        
        if target_ja:
            target_uid = target_ja.get('properties', {}).get('uid')
            for t in audio_tracks:
                uid = t.get('properties', {}).get('uid')
                is_def = t.get('properties', {}).get('default_track', False)
                if uid == target_uid:
                    if not is_def:
                        propedit_args.extend(['--edit', f'track:={uid}', '--set', 'flag-default=1'])
                        changes.append(f"Audio JA (UID={uid}) -> default=1")
                else:
                    if is_def:
                        propedit_args.extend(['--edit', f'track:={uid}', '--set', 'flag-default=0'])
                        changes.append(f"Audio Non-JA (UID={uid}) -> default=0")

    # 2. Subtitle selection
    if sub_tracks:
        vi_subs = [t for t in sub_tracks if is_vietnamese_sub(t)]
        target_sub = None
        if vi_subs:
            target_sub = vi_subs[0]
        else:
            dialogue_subs = [t for t in sub_tracks if is_dialogue_sub(t)]
            if dialogue_subs:
                explicit_dialog = [t for t in dialogue_subs if 'dialog' in t.get('properties', {}).get('track_name', '').lower()]
                target_sub = explicit_dialog[0] if explicit_dialog else dialogue_subs[0]
            else:
                non_signs = [t for t in sub_tracks if not is_signs_or_songs(t)]
                target_sub = non_signs[0] if non_signs else sub_tracks[0]
        
        if target_sub:
            target_sub_uid = target_sub.get('properties', {}).get('uid')
            for t in sub_tracks:
                uid = t.get('properties', {}).get('uid')
                is_def = t.get('properties', {}).get('default_track', False)
                if uid == target_sub_uid:
                    if not is_def:
                        propedit_args.extend(['--edit', f'track:={uid}', '--set', 'flag-default=1'])
                        changes.append(f"Sub (UID={uid}, '{target_sub.get('properties',{}).get('track_name','')}') -> default=1")
                else:
                    if is_def:
                        propedit_args.extend(['--edit', f'track:={uid}', '--set', 'flag-default=0'])
                        changes.append(f"Sub (UID={uid}, '{t.get('properties',{}).get('track_name','')}') -> default=0")
                        
    if not propedit_args:
        return {'status': 'unchanged', 'changes': []}
    
    cmd_edit = [mkvpropedit_path, fpath] + propedit_args
    res_edit = subprocess.run(cmd_edit, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if res_edit.returncode != 0:
        return {'status': 'error', 'msg': f'mkvpropedit failed: {res_edit.stderr}'}
    
    return {'status': 'modified', 'changes': changes}

def main():
    all_files = []
    for root, dirs, fnames in os.walk(raw_dir):
        for f in fnames:
            if f.lower().endswith('.mkv'):
                if incomplete.lower() in f.lower():
                    print(f"[SKIP] Incomplete file downloading: {f}")
                    continue
                all_files.append(os.path.join(root, f))
                
    print(f"Total completed MKV files to process: {len(all_files)}")
    start_time = time.time()
    
    modified_count = 0
    unchanged_count = 0
    error_count = 0
    
    by_folder_stats = {}
    
    for idx, fpath in enumerate(all_files, 1):
        rel_folder = os.path.relpath(os.path.dirname(fpath), raw_dir)
        fname = os.path.basename(fpath)
        stats = by_folder_stats.setdefault(rel_folder, {'modified': 0, 'unchanged': 0, 'error': 0})
        
        result = process_file(fpath)
        if result['status'] == 'modified':
            modified_count += 1
            stats['modified'] += 1
        elif result['status'] == 'unchanged':
            unchanged_count += 1
            stats['unchanged'] += 1
        else:
            error_count += 1
            stats['error'] += 1
            print(f"ERROR on {fname}: {result.get('msg')}")
            
        if idx % 50 == 0 or idx == len(all_files):
            elapsed = time.time() - start_time
            print(f"Progress: [{idx}/{len(all_files)}] files processed ({elapsed:.1f}s) - Modified: {modified_count}, Unchanged: {unchanged_count}, Errors: {error_count}")
            
    total_elapsed = time.time() - start_time
    print(f"\n==========================================")
    print(f"SUMMARY (Completed in {total_elapsed:.1f}s)")
    print(f"Total files: {len(all_files)}")
    print(f"Modified: {modified_count}")
    print(f"Unchanged: {unchanged_count}")
    print(f"Errors: {error_count}")
    print(f"==========================================")
    for folder, s in sorted(by_folder_stats.items()):
        print(f"  {folder}: {s['modified']} modified, {s['unchanged']} unchanged, {s['error']} errors")

if __name__ == '__main__':
    main()
