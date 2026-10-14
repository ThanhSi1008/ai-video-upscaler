#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dedup import scan_dedup_plan

def test_windows(video_path):
    print(f"=== ĐO ĐẠC CỬA SỔ TRÊN: {video_path} ===")
    
    # 1. Quét plan cho default, sensitive-b, dis-0.3, off
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p_def = scan_dedup_plan(video_path, os.path.join(td, "def.json"), policy="default")
        p_sb = scan_dedup_plan(video_path, os.path.join(td, "sb.json"), policy="sensitive-b")
        p_dis = scan_dedup_plan(video_path, os.path.join(td, "dis.json"), policy="dis-0.3")
        p_drift = scan_dedup_plan(video_path, os.path.join(td, "drift.json"), policy="drift-0.5")
        p_off = scan_dedup_plan(video_path, os.path.join(td, "off.json"), policy="off")
        
        # Lấy actions chuỗi frame từ chunks
        def get_actions(p):
            acts = []
            for c in p["chunks"]:
                for r in c["runs"]:
                    acts.append("keep")
                    for _ in range(r["count"] - 1):
                        acts.append("drop")
            return acts

        acts_def = get_actions(p_def)
        acts_sb = get_actions(p_sb)
        acts_dis = get_actions(p_dis)
        acts_drift = get_actions(p_drift)
        acts_off = get_actions(p_off)

        print(f"Tổng frame: {len(acts_def)}")
        print(f"Tổng kept default: {sum(1 for a in acts_def if a == 'keep')}")
        print(f"Tổng kept sensitive-b: {sum(1 for a in acts_sb if a == 'keep')}")
        print(f"Tổng kept dis-0.3: {sum(1 for a in acts_dis if a == 'keep')}")
        print(f"Tổng kept drift-0.5: {sum(1 for a in acts_drift if a == 'keep')}")
        print(f"Tổng kept off: {sum(1 for a in acts_off if a == 'keep')}")

        windows = [
            ("Window 1 (zoom/lia cũ theo báo cáo dis)", 965, 1048),
            ("Window 2 (khựng người dùng thấy 44-46s)", 1055, 1103),
            ("Cửa sổ rộng bao cả hai (36-50s)", int(round(36.0 * 24000/1001)), int(round(50.0 * 24000/1001))),
        ]

        for name, start_f, end_f in windows:
            total_w = end_f - start_f + 1
            k_def = sum(1 for a in acts_def[start_f:end_f+1] if a == "keep")
            k_sb = sum(1 for a in acts_sb[start_f:end_f+1] if a == "keep")
            k_dis = sum(1 for a in acts_dis[start_f:end_f+1] if a == "keep")
            k_drift = sum(1 for a in acts_drift[start_f:end_f+1] if a == "keep")
            k_off = sum(1 for a in acts_off[start_f:end_f+1] if a == "keep")

            print(f"\n--- {name} [Frames {start_f} -> {end_f}, Tổng {total_w} frames] ---")
            print(f"  default    : {k_def}/{total_w} kept ({k_def/total_w*100:.1f}%), bỏ {total_w - k_def}")
            print(f"  sensitive-b: {k_sb}/{total_w} kept ({k_sb/total_w*100:.1f}%), bỏ {total_w - k_sb}")
            print(f"  dis-0.3    : {k_dis}/{total_w} kept ({k_dis/total_w*100:.1f}%), bỏ {total_w - k_dis}")
            print(f"  drift-0.5  : {k_drift}/{total_w} kept ({k_drift/total_w*100:.1f}%), bỏ {total_w - k_drift}")
            print(f"  off        : {k_off}/{total_w} kept ({k_off/total_w*100:.1f}%), bỏ {total_w - k_off}")

if __name__ == "__main__":
    test_windows("samples/sample_1min_bd.mkv")
