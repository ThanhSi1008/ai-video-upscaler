import os
import sys
import json
import time
import hashlib
import random
import subprocess
import cv2
import numpy as np
from pathlib import Path

# Paths
out_default = "/Users/xis108/Desktop/upscale/scratch/out_stutter_default/stutter_10s_bd_4K.mkv"
out_drift = "/Users/xis108/Desktop/upscale/scratch/out_stutter_drift-0_5/stutter_10s_bd_4K.mkv"
out_off = "/Users/xis108/Desktop/upscale/scratch/out_stutter_off/stutter_10s_bd_4K.mkv"

pack_dir = os.path.expanduser("~/bd_model_test/blind_stutter_pack")
key_dir = os.path.expanduser("~/bd_model_test/.key")
os.makedirs(pack_dir, exist_ok=True)
os.makedirs(key_dir, exist_ok=True)

# 1. Tính MAE cho cả 3 file
def analyze_mae(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray = None
    mae_list = []
    
    # Cảnh lia máy / zoom chậm rơi vào mốc 8s-10s (frame 192 đến 240) của clip 10s này
    zoom_pairs = []
    
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            mae = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            mae_list.append(mae)
            if 192 <= idx < 240:
                zoom_pairs.append(mae)
        prev_gray = gray
        idx += 1
    cap.release()
    
    identical_all = sum(1 for m in mae_list if m < 0.05)
    identical_zoom = sum(1 for m in zoom_pairs if m < 0.05)
    
    return {
        "total_frames": total_frames,
        "n_pairs": len(mae_list),
        "identical_all": identical_all,
        "pct_identical_all": round(identical_all / max(1, len(mae_list)) * 100.0, 2),
        "n_zoom_pairs": len(zoom_pairs),
        "identical_zoom": identical_zoom,
        "pct_identical_zoom": round(identical_zoom / max(1, len(zoom_pairs)) * 100.0, 2),
    }

print("Đang đo MAE cho default...")
mae_default = analyze_mae(out_default)
print("Đang đo MAE cho drift-0.5...")
mae_drift = analyze_mae(out_drift)
print("Đang đo MAE cho off...")
mae_off = analyze_mae(out_off)

# 2. Đọc thông số audit & thời gian
def get_stats(audit_path):
    with open(audit_path) as f:
        data = json.load(f)
    return {
        "file_size_mb": data.get("output_file_size_mb"),
        "total_processing_seconds": data.get("total_processing_seconds"),
        "pipeline_fps": data.get("average_pipeline_fps"),
        "ai_steady_fps": data.get("ai_steady_fps"),
        "delta_y": data.get("output_sanity", {}).get("checks", [{}])[0].get("delta_y"),
        "corr": data.get("output_sanity", {}).get("checks", [{}])[0].get("corr"),
    }

audit_default = get_stats("/Users/xis108/Desktop/upscale/scratch/out_stutter_default/audit/stutter_10s_bd_4K.audit.json")
audit_drift = get_stats("/Users/xis108/Desktop/upscale/scratch/out_stutter_drift-0_5/audit/stutter_10s_bd_4K.audit.json")
audit_off = get_stats("/Users/xis108/Desktop/upscale/scratch/out_stutter_off/audit/stutter_10s_bd_4K.audit.json")

# Kế hoạch dedup
plan_default_kept = 124
plan_drift_kept = 125
plan_off_kept = 240

b2_summary = {
    "default": {
        "policy": "default",
        "description": "mpdecimate chuẩn (hi=768, lo=320, frac=0.33)",
        "frames_infer": plan_default_kept,
        "pct_frames_vs_off": round(plan_default_kept / 240.0 * 100.0, 2),
        "gpu_time_s": audit_default["total_processing_seconds"],
        "pipeline_fps": audit_default["pipeline_fps"],
        "ai_steady_fps": audit_default["ai_steady_fps"],
        "file_size_mb": audit_default["file_size_mb"],
        "output_sanity": {"delta_y": audit_default["delta_y"], "corr": audit_default["corr"]},
        "mae_all": mae_default,
    },
    "drift-0.5": {
        "policy": "drift-0.5",
        "description": "mpdecimate + phase correlation drift (delta=0.5, min_response=0.10)",
        "frames_infer": plan_drift_kept,
        "pct_frames_vs_off": round(plan_drift_kept / 240.0 * 100.0, 2),
        "gpu_time_s": audit_drift["total_processing_seconds"],
        "pipeline_fps": audit_drift["pipeline_fps"],
        "ai_steady_fps": audit_drift["ai_steady_fps"],
        "file_size_mb": audit_drift["file_size_mb"],
        "output_sanity": {"delta_y": audit_drift["delta_y"], "corr": audit_drift["corr"]},
        "mae_all": mae_drift,
    },
    "off": {
        "policy": "off",
        "description": "Không dedup (100% 240 frames infer, baseline mượt tuyệt đối)",
        "frames_infer": plan_off_kept,
        "pct_frames_vs_off": 100.0,
        "gpu_time_s": audit_off["total_processing_seconds"],
        "pipeline_fps": audit_off["pipeline_fps"],
        "ai_steady_fps": audit_off["ai_steady_fps"],
        "file_size_mb": audit_off["file_size_mb"],
        "output_sanity": {"delta_y": audit_off["delta_y"], "corr": audit_off["corr"]},
        "mae_all": mae_off,
    }
}

with open("/Users/xis108/Desktop/upscale/scratch/b2_stutter_results.json", "w") as f:
    json.dump(b2_summary, f, indent=2)

print("\n=== B2 SUMMARY RESULTS ===")
print(json.dumps(b2_summary, indent=2))

# 3. Tạo Blind Pack B3: Clip_P, Clip_Q, Clip_R
# Mã hóa ngẫu nhiên bảo mật
candidates = [
    ("default", out_default),
    ("drift-0.5", out_drift),
    ("off", out_off),
]

# Random shuffle với seed bảo mật dựa trên urandom
rng = random.Random(os.urandom(32))
shuffled = list(candidates)
rng.shuffle(shuffled)

letters = ["P", "Q", "R"]
key_mapping = {}

print("\n=== B3. XUẤT XƯỞNG BLIND PACK PACK_STUTTER ===")
for letter, (pol_name, src_file) in zip(letters, shuffled):
    dst_clip = os.path.join(pack_dir, f"Clip_{letter}.mkv")
    # Copy bằng hardlink hoặc file copy
    if os.path.exists(dst_clip):
        os.remove(dst_clip)
    subprocess.run(["cp", src_file, dst_clip], check=True)
    
    # Tính sha256
    with open(dst_clip, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    
    key_mapping[f"Clip_{letter}"] = {
        "policy": pol_name,
        "source_path": src_file,
        "sha256": sha,
        "frames_infer": b2_summary[pol_name]["frames_infer"],
        "identical_pairs_zoom": b2_summary[pol_name]["mae_all"]["identical_zoom"],
        "identical_pairs_all": b2_summary[pol_name]["mae_all"]["identical_all"],
    }
    print(f"✅ Gói mù Clip_{letter}.mkv -> {dst_clip} (SHA256: {sha[:16]}...)")

# Lưu key bí mật
key_file = os.path.join(key_dir, "stutter_key.json")
with open(key_file, "w") as f:
    json.dump({
        "pack_name": "PACK_STUTTER",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mapping": key_mapping
    }, f, indent=2)

os.chmod(key_file, 0o600)
print(f"\n🔒 Đã lưu file chìa khóa tại {key_file} (Quyền hạn: 600 - Chỉ chủ sở hữu đọc).")
