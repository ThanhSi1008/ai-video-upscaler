import os, glob, json, numpy as np
from PIL import Image
import torch
from src.mps_engine import upscale_frames_with_fallback
from src.models_resolver import resolve_model_for_video

def measure_dataset(name, input_dir, output_dir, model_name="animevideov3"):
    os.makedirs(output_dir, exist_ok=True)
    model_info = resolve_model_for_video("", cli_model_arg=model_name)
    
    print(f"--- Running upscale for {name} ({len(os.listdir(input_dir))} frames) ---")
    upscale_frames_with_fallback(
        input_dir=input_dir,
        output_dir=output_dir,
        model_info=model_info,
        engine="mps",
        precision="fp16",
        fmt="png"
    )
    
    # Measure file sizes
    png_files = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    sizes_bytes = [os.path.getsize(f) for f in png_files]
    sizes_mb = [s / (1024 * 1024) for s in sizes_bytes]
    
    stats = {
        "dataset": name,
        "n_frames": len(sizes_bytes),
        "bytes_p50": float(np.percentile(sizes_bytes, 50)),
        "bytes_p95": float(np.percentile(sizes_bytes, 95)),
        "bytes_p99": float(np.percentile(sizes_bytes, 99)),
        "bytes_max": float(np.max(sizes_bytes)),
        "mb_p50": float(np.percentile(sizes_mb, 50)),
        "mb_p95": float(np.percentile(sizes_mb, 95)),
        "mb_p99": float(np.percentile(sizes_mb, 99)),
        "mb_max": float(np.max(sizes_mb)),
    }
    return stats

def main():
    scratch_dir = "/Users/xis108/Desktop/upscale/scratch/c1_test"
    bd_in = os.path.join(scratch_dir, "bd_in")
    bd_out = os.path.join(scratch_dir, "bd_out")
    web_in = os.path.join(scratch_dir, "web_in")
    web_out = os.path.join(scratch_dir, "web_out")
    
    bd_stats = measure_dataset("BD_420_120f", bd_in, bd_out)
    web_stats = measure_dataset("WEB_DL_120f", web_in, web_out)
    
    out = {
        "BD_420": bd_stats,
        "WEB_DL": web_stats,
        "floor_safe_16mb": {
            "floor_value_mb": 16.0,
            "bd_exceeds": bool(bd_stats["mb_max"] > 16.0),
            "web_exceeds": bool(web_stats["mb_max"] > 16.0)
        }
    }
    
    res_path = "/Users/xis108/Desktop/upscale/scratch/results_c1.json"
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved C1 results to", res_path)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
