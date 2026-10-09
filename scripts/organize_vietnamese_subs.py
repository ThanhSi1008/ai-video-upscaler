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

import os
import re
import shutil

BASE_DIR = "/Users/xis108/Documents/RawSources"

def organize():
    stats = {}
    copied = []

    # 1. Fate Zero
    fz_vid_dir = os.path.join(BASE_DIR, "1-Fate Zero [Ma10p_1080p]")
    fz_sub_dir = os.path.join(BASE_DIR, "fatezero-season-1-2011_vietnamese_10087549-[SubSource]")
    vids = sorted([f for f in os.listdir(fz_vid_dir) if f.endswith(".mkv") and not f.startswith(".")])
    subs = sorted([f for f in os.listdir(fz_sub_dir) if f.endswith((".srt", ".ass"))])
    
    count_fz = 0
    for v in vids:
        # Nhận diện tập [01] hoặc [11v2]
        m = re.search(r'\[(\d{2})(?:v\d+)?\]', v)
        if m:
            ep = int(m.group(1))
            matched_sub = next((s for s in subs if f"S01E{ep:02d}" in s), None)
            if matched_sub:
                ext = os.path.splitext(matched_sub)[1]
                target_name = os.path.splitext(v)[0] + ".vi" + ext
                src_path = os.path.join(fz_sub_dir, matched_sub)
                dst_path = os.path.join(fz_vid_dir, target_name)
                shutil.copy2(src_path, dst_path)
                count_fz += 1
                copied.append(("Fate Zero", v, target_name))
    stats["Fate Zero"] = count_fz

    # 2. Fate stay night Unlimited Blade Works
    ubw_vid_dir = os.path.join(BASE_DIR, "2-Fate stay night Unlimited Blade Works [Ma10p_1080p]")
    s1_sub_dir = os.path.join(BASE_DIR, "fatestay-night-unlimited-blade-works-season-1_vietnamese_3304772-[SubSource]")
    s2_sub_dir = os.path.join(BASE_DIR, "fatestay-night-unlimited-blade-works-season-2_vietnamese_3304773-[SubSource]")
    vids = sorted([f for f in os.listdir(ubw_vid_dir) if f.endswith(".mkv") and not f.startswith(".")])
    s1_subs = sorted([f for f in os.listdir(s1_sub_dir) if f.endswith((".srt", ".ass"))])
    s2_subs = sorted([f for f in os.listdir(s2_sub_dir) if f.endswith((".srt", ".ass"))])
    
    count_ubw = 0
    for v in vids:
        m = re.search(r'\[(\d{2})\]', v)
        if m:
            ep = int(m.group(1))
            if ep == 0:
                matched_sub = next((s for s in s1_subs if "S00E01" in s), None)
                s_dir = s1_sub_dir
            elif 1 <= ep <= 12:
                matched_sub = next((s for s in s1_subs if f"S01E{ep:02d}" in s), None)
                s_dir = s1_sub_dir
            else:
                matched_sub = next((s for s in s2_subs if f"S01E{ep:02d}" in s), None)
                s_dir = s2_sub_dir
            if matched_sub:
                ext = os.path.splitext(matched_sub)[1]
                target_name = os.path.splitext(v)[0] + ".vi" + ext
                src_path = os.path.join(s_dir, matched_sub)
                dst_path = os.path.join(ubw_vid_dir, target_name)
                shutil.copy2(src_path, dst_path)
                count_ubw += 1
                copied.append(("Fate UBW", v, target_name))
    stats["Fate stay night UBW"] = count_ubw

    # 3. Heaven's Feel I, II, III
    hf_list = [
        ("3.1-Fate stay night Heaven's Feel I - presage flower [Ma10p_1080p]", "fatestay-night-heavens-feel-i-presage-flower-2017_vietnamese_3304665-[SubSource]"),
        ("3.2-Fate stay night Heaven's Feel II - lost butterfly [Ma10p_1080p]", "fatestay-night-heavens-feel-ii-lost-butterfly-2019_vietnamese_3304669-[SubSource]"),
        ("3.3-Fate ⁄ stay night Heaven's Feel III - spring song [Ma10p_1080p]", "fatestay-night-heavens-feel-iii-spring-song-2020_vietnamese_3304670-[SubSource]")
    ]
    count_hf = 0
    for v_sub, s_sub in hf_list:
        v_dir = os.path.join(BASE_DIR, v_sub)
        s_dir = os.path.join(BASE_DIR, s_sub)
        if os.path.exists(v_dir) and os.path.exists(s_dir):
            vids = [f for f in os.listdir(v_dir) if f.endswith(".mkv") and "Heaven" in f and not f.startswith(".")]
            subs = [f for f in os.listdir(s_dir) if f.endswith((".srt", ".ass")) and not f.startswith(".")]
            if vids and subs:
                ext = os.path.splitext(subs[0])[1]
                target_name = os.path.splitext(vids[0])[0] + ".vi" + ext
                shutil.copy2(os.path.join(s_dir, subs[0]), os.path.join(v_dir, target_name))
                count_hf += 1
                copied.append(("Heaven's Feel", vids[0], target_name))
    stats["Heaven's Feel Trilogy"] = count_hf

    # 4. Fate Apocrypha
    apo_vid_dir = os.path.join(BASE_DIR, "Fate Apocrypha")
    apo_sub_dir = os.path.join(BASE_DIR, "fateapocrypha-season-1_vietnamese_3305298-[SubSource]")
    vids = sorted([f for f in os.listdir(apo_vid_dir) if f.endswith(".mkv") and not f.startswith(".")])
    subs = sorted([f for f in os.listdir(apo_sub_dir) if f.endswith((".srt", ".ass"))])
    count_apo = 0
    for v in vids:
        m = re.search(r'-\s*(\d{2})\b', v)
        if m:
            ep = int(m.group(1))
            matched_sub = next((s for s in subs if f"S01E{ep:02d}" in s), None)
            if matched_sub:
                ext = os.path.splitext(matched_sub)[1]
                target_name = os.path.splitext(v)[0] + ".vi" + ext
                shutil.copy2(os.path.join(apo_sub_dir, matched_sub), os.path.join(apo_vid_dir, target_name))
                count_apo += 1
                copied.append(("Fate Apocrypha", v, target_name))
    stats["Fate Apocrypha"] = count_apo

    # 5. Fate Grand Order - Absolute Demonic Front Babylonia
    bab_vid_dir = os.path.join(BASE_DIR, "Fate Grand Order - Absolute Demonic Front Babylonia")
    bab_sub_dir = os.path.join(BASE_DIR, "fategrand-order-absolute-demonic-front-babylonia-season-1_vietnamese_2543067-[SubSource]")
    vids = sorted([f for f in os.listdir(bab_vid_dir) if f.endswith(".mkv") and not f.startswith(".")])
    subs = sorted([f for f in os.listdir(bab_sub_dir) if f.endswith((".srt", ".ass"))])
    count_bab = 0
    for v in vids:
        m = re.search(r'S01E(\d{2})', v)
        if m:
            ep = int(m.group(1))
            matched_sub = next((s for s in subs if f"S01E{ep:02d}" in s), None)
            if matched_sub:
                ext = os.path.splitext(matched_sub)[1]
                target_name = os.path.splitext(v)[0] + ".vi" + ext
                shutil.copy2(os.path.join(bab_sub_dir, matched_sub), os.path.join(bab_vid_dir, target_name))
                count_bab += 1
                copied.append(("FGO Babylonia", v, target_name))
    stats["FGO Babylonia"] = count_bab

    # 6. Camelot Movies
    c1_vid = os.path.join(BASE_DIR, "[Salieri] Fate - Grand Order - Divine Realm of the Round Table - Camelot Wandering; Agateram 1080p BD (HDR) [Dual Audio].mkv")
    c1_sub = os.path.join(BASE_DIR, "Fate.Grand.Order.Shinsei.Entaku.Ryouiki.Camelot.1.-.Wandering.Agateram.2020.1080p.BluRay.x264.AAC5.1-[YTS.MX].srt")
    count_cam = 0
    if os.path.exists(c1_vid) and os.path.exists(c1_sub):
        target_name = os.path.splitext(os.path.basename(c1_vid))[0] + ".vi.srt"
        dst = os.path.join(BASE_DIR, target_name)
        shutil.copy2(c1_sub, dst)
        count_cam += 1
        copied.append(("Camelot 1", os.path.basename(c1_vid), target_name))

    c2_vid = os.path.join(BASE_DIR, "[Salieri] Fate - Grand Order - Divine Realm of the Round Table - Camelot Paladin; Agateram 1080p BD (HDR) [Dual Audio].mkv")
    c2_sub = os.path.join(BASE_DIR, "Camelot Paladin.srt")
    if os.path.exists(c2_vid) and os.path.exists(c2_sub):
        target_name = os.path.splitext(os.path.basename(c2_vid))[0] + ".vi.srt"
        dst = os.path.join(BASE_DIR, target_name)
        shutil.copy2(c2_sub, dst)
        count_cam += 1
        copied.append(("Camelot 2", os.path.basename(c2_vid), target_name))
    stats["FGO Camelot Movies"] = count_cam

    print("=========================================================================")
    print("HOÀN TẤT ĐỒNG BỘ PHỤ ĐỀ TIẾNG VIỆT VÀO TỪNG PHẦN TỪNG TẬP:")
    print("=========================================================================")
    for k, v in stats.items():
        print(f"  • {k:30s}: Đã đặt {v:2d} file phụ đề .vi tương ứng")
    total_synced = sum(stats.values())
    print(f"\n🎯 TỔNG CỘNG: {total_synced} FILE PHỤ ĐỀ ĐÃ ĐƯỢC ĐẶT KHỚP CHÍNH XÁC VÀO TẬP PHIM!")
    print("=========================================================================")

if __name__ == "__main__":
    organize()
