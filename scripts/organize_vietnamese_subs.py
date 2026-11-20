# -*- coding: utf-8 -*-
"""
scripts/organize_vietnamese_subs.py
Tự động đồng bộ và đặt các file phụ đề tiếng Việt vào đúng từng thư mục phim và từng tập video:
1. Fate Zero (25 tập + 2 bản v2)
2. Fate stay night Unlimited Blade Works (26 tập: 00 -> 25)
3. Fate stay night Heaven's Feel I, II, III (3 movie)
4. Fate Apocrypha (25 tập)
5. Fate Grand Order - Absolute Demonic Front Babylonia (22 tập: 00 -> 21)
6. FGO Camelot Movie 1 & 2 (Wandering & Paladin)
"""

import argparse
import os
import re
import shutil
import sys
from typing import Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.env_isolation import configure_e_drive_isolation


def _list_mkvs(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted([f for f in os.listdir(dir_path) if f.endswith(".mkv") and not f.startswith(".")])


def _list_subs(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted([f for f in os.listdir(dir_path) if f.endswith((".srt", ".ass")) and not f.startswith(".")])


def _copy_if_real(src_path: str, dst_path: str, dry_run: bool) -> None:
    if dry_run:
        return
    shutil.copy2(src_path, dst_path)


def organize(source_dir: str, dry_run: bool = False) -> Dict[str, int]:
    base_dir = os.path.abspath(source_dir)
    if not os.path.isdir(base_dir):
        print(f"⚠️ Thư mục nguồn không tồn tại: {base_dir}")
        return {}

    stats: Dict[str, int] = {}

    # 1. Fate Zero
    fz_vid_dir = os.path.join(base_dir, "1-Fate Zero [Ma10p_1080p]")
    fz_sub_dir = os.path.join(base_dir, "fatezero-season-1-2011_vietnamese_10087549-[SubSource]")
    count_fz = 0
    if os.path.isdir(fz_vid_dir) and os.path.isdir(fz_sub_dir):
        vids = _list_mkvs(fz_vid_dir)
        subs = _list_subs(fz_sub_dir)
        for v in vids:
            # Nhận diện tập [01] hoặc [11v2]
            m = re.search(r"\[(\d{2})(?:v\d+)?\]", v)
            if m:
                ep = int(m.group(1))
                matched_sub = next((s for s in subs if f"S01E{ep:02d}" in s or f"E{ep:02d}" in s), None)
                if matched_sub:
                    ext = os.path.splitext(matched_sub)[1]
                    target_name = os.path.splitext(v)[0] + ".vi" + ext
                    src_path = os.path.join(fz_sub_dir, matched_sub)
                    dst_path = os.path.join(fz_vid_dir, target_name)
                    _copy_if_real(src_path, dst_path, dry_run)
                    count_fz += 1
    stats["Fate Zero"] = count_fz

    # 2. Fate stay night Unlimited Blade Works
    ubw_vid_dir = os.path.join(base_dir, "2-Fate stay night Unlimited Blade Works [Ma10p_1080p]")
    s1_sub_dir = os.path.join(base_dir, "fatestay-night-unlimited-blade-works-season-1_vietnamese_3304772-[SubSource]")
    s2_sub_dir = os.path.join(base_dir, "fatestay-night-unlimited-blade-works-season-2_vietnamese_3304773-[SubSource]")
    count_ubw = 0
    if os.path.isdir(ubw_vid_dir):
        vids = _list_mkvs(ubw_vid_dir)
        s1_subs = _list_subs(s1_sub_dir)
        s2_subs = _list_subs(s2_sub_dir)
        for v in vids:
            m = re.search(r"\[(\d{2})\]", v)
            if m:
                ep = int(m.group(1))
                if ep == 0:
                    matched_sub = next((s for s in s1_subs if "S00E01" in s), None)
                    s_dir = s1_sub_dir
                elif 1 <= ep <= 12:
                    matched_sub = next((s for s in s1_subs if f"S01E{ep:02d}" in s), None)
                    s_dir = s1_sub_dir
                else:
                    s2_ep = ep - 12
                    matched_sub = next(
                        (
                            s for s in s2_subs
                            if f"S02E{s2_ep:02d}" in s or f"S01E{ep:02d}" in s or f"E{ep:02d}" in s
                        ),
                        None,
                    )
                    s_dir = s2_sub_dir
                if matched_sub and os.path.isdir(s_dir):
                    ext = os.path.splitext(matched_sub)[1]
                    target_name = os.path.splitext(v)[0] + ".vi" + ext
                    src_path = os.path.join(s_dir, matched_sub)
                    dst_path = os.path.join(ubw_vid_dir, target_name)
                    _copy_if_real(src_path, dst_path, dry_run)
                    count_ubw += 1
    stats["Fate stay night UBW"] = count_ubw

    # 3. Heaven's Feel I, II, III
    hf_list = [
        ("3.1-Fate stay night Heaven's Feel I - presage flower [Ma10p_1080p]", "fatestay-night-heavens-feel-i-presage-flower-2017_vietnamese_3304665-[SubSource]"),
        ("3.2-Fate stay night Heaven's Feel II - lost butterfly [Ma10p_1080p]", "fatestay-night-heavens-feel-ii-lost-butterfly-2019_vietnamese_3304669-[SubSource]"),
        ("3.3-Fate stay night Heaven's Feel III - spring song [Ma10p_1080p]", "fatestay-night-heavens-feel-iii-spring-song-2020_vietnamese_3304670-[SubSource]"),
    ]
    count_hf = 0
    for v_sub, s_sub in hf_list:
        v_dir = os.path.join(base_dir, v_sub)
        if not os.path.isdir(v_dir) and "Heaven's Feel III" in v_sub:
            alt_v_dir = os.path.join(base_dir, "3.3-Fate \u2044 stay night Heaven's Feel III - spring song [Ma10p_1080p]")
            if os.path.isdir(alt_v_dir):
                v_dir = alt_v_dir
        s_dir = os.path.join(base_dir, s_sub)
        if os.path.isdir(v_dir) and os.path.isdir(s_dir):
            vids = [f for f in _list_mkvs(v_dir) if "Heaven" in f]
            subs = _list_subs(s_dir)
            if vids and subs:
                ext = os.path.splitext(subs[0])[1]
                target_name = os.path.splitext(vids[0])[0] + ".vi" + ext
                _copy_if_real(os.path.join(s_dir, subs[0]), os.path.join(v_dir, target_name), dry_run)
                count_hf += 1
    stats["Heaven's Feel Trilogy"] = count_hf

    # 4. Fate Apocrypha
    apo_vid_dir = os.path.join(base_dir, "Fate Apocrypha")
    apo_sub_dir = os.path.join(base_dir, "fateapocrypha-season-1_vietnamese_3305298-[SubSource]")
    count_apo = 0
    if os.path.isdir(apo_vid_dir) and os.path.isdir(apo_sub_dir):
        vids = _list_mkvs(apo_vid_dir)
        subs = _list_subs(apo_sub_dir)
        for v in vids:
            m = re.search(r"-\s*(\d{2})\b", v)
            if m:
                ep = int(m.group(1))
                matched_sub = next((s for s in subs if f"S01E{ep:02d}" in s), None)
                if matched_sub:
                    ext = os.path.splitext(matched_sub)[1]
                    target_name = os.path.splitext(v)[0] + ".vi" + ext
                    _copy_if_real(os.path.join(apo_sub_dir, matched_sub), os.path.join(apo_vid_dir, target_name), dry_run)
                    count_apo += 1
    stats["Fate Apocrypha"] = count_apo

    # 5. Fate Grand Order - Absolute Demonic Front Babylonia
    bab_vid_dir = os.path.join(base_dir, "Fate Grand Order - Absolute Demonic Front Babylonia")
    bab_sub_dir = os.path.join(base_dir, "fategrand-order-absolute-demonic-front-babylonia-season-1_vietnamese_2543067-[SubSource]")
    count_bab = 0
    if os.path.isdir(bab_vid_dir) and os.path.isdir(bab_sub_dir):
        vids = _list_mkvs(bab_vid_dir)
        subs = _list_subs(bab_sub_dir)
        for v in vids:
            m = re.search(r"S01E(\d{2})", v)
            if m:
                ep = int(m.group(1))
                matched_sub = next((s for s in subs if f"S01E{ep:02d}" in s), None)
                if matched_sub:
                    ext = os.path.splitext(matched_sub)[1]
                    target_name = os.path.splitext(v)[0] + ".vi" + ext
                    _copy_if_real(os.path.join(bab_sub_dir, matched_sub), os.path.join(bab_vid_dir, target_name), dry_run)
                    count_bab += 1
    stats["FGO Babylonia"] = count_bab

    # 6. Camelot Movies
    c1_vid = os.path.join(base_dir, "[Salieri] Fate - Grand Order - Divine Realm of the Round Table - Camelot Wandering; Agateram 1080p BD (HDR) [Dual Audio].mkv")
    c1_sub = os.path.join(base_dir, "Fate.Grand.Order.Shinsei.Entaku.Ryouiki.Camelot.1.-.Wandering.Agateram.2020.1080p.BluRay.x264.AAC5.1-[YTS.MX].srt")
    count_cam = 0
    if os.path.exists(c1_vid) and os.path.exists(c1_sub):
        target_name = os.path.splitext(os.path.basename(c1_vid))[0] + ".vi.srt"
        dst = os.path.join(base_dir, target_name)
        _copy_if_real(c1_sub, dst, dry_run)
        count_cam += 1

    c2_vid = os.path.join(base_dir, "[Salieri] Fate - Grand Order - Divine Realm of the Round Table - Camelot Paladin; Agateram 1080p BD (HDR) [Dual Audio].mkv")
    c2_sub = os.path.join(base_dir, "Camelot Paladin.srt")
    if os.path.exists(c2_vid) and os.path.exists(c2_sub):
        target_name = os.path.splitext(os.path.basename(c2_vid))[0] + ".vi.srt"
        dst = os.path.join(base_dir, target_name)
        _copy_if_real(c2_sub, dst, dry_run)
        count_cam += 1
    stats["FGO Camelot Movies"] = count_cam

    mode_prefix = "[DRY-RUN] " if dry_run else ""
    print("=========================================================================")
    print(f"{mode_prefix}HOÀN TẤT ĐỒNG BỘ PHỤ ĐỀ TIẾNG VIỆT VÀO TỪNG PHẦN TỪNG TẬP:")
    print("=========================================================================")
    for k, v in stats.items():
        print(f"  • {k:30s}: Đã đặt {v:2d} file phụ đề .vi tương ứng")
    total_synced = sum(stats.values())
    print(f"\n🎯 TỔNG CỘNG: {total_synced} FILE PHỤ ĐỀ ĐÃ ĐƯỢC ĐẶT KHỚP CHÍNH XÁC VÀO TẬP PHIM!")
    print("=========================================================================")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    configure_e_drive_isolation(project_root=_REPO_ROOT)

    parser = argparse.ArgumentParser(
        description="Tự động đồng bộ và đặt phụ đề tiếng Việt vào từng thư mục Fate Series"
    )
    parser.add_argument("source_dir", help="Thư mục gốc chứa video và phụ đề (ví dụ: E:\\Films\\Raw_Sources)")
    parser.add_argument("--dry-run", action="store_true", help="Mô phỏng đối chiếu mà không copy file")
    args = parser.parse_args(argv)
    if not os.path.isdir(args.source_dir):
        print(f"❌ Thư mục nguồn không tồn tại: {args.source_dir}", file=sys.stderr)
        return 2
    organize(args.source_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
