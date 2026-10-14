import os, sys, json, time, subprocess, shutil, secrets

def run_cmd(cmd, cwd="/Users/xis108/Desktop/upscale"):
    print(f"\n>>> Running: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    dt = time.time() - t0
    print(f">>> Finished in {dt:.2f}s (code: {res.returncode})")
    if res.returncode != 0:
        print("STDERR tail:\n", "\n".join(res.stderr.splitlines()[-20:]))
        print("STDOUT tail:\n", "\n".join(res.stdout.splitlines()[-20:]))
    return res.returncode, dt, res.stdout, res.stderr

def parse_log_stats(stdout, audit_path):
    stats = {
        "n_unique": None,
        "ai_fps": None,
        "pipeline_fps": None,
        "wall_time_sec": None,
        "audit_pass": False,
        "output_size_mb": None
    }
    # Parse stdout for unique frames
    for line in stdout.splitlines():
        if "unique frames" in line.lower():
            # example: Giữ 529 unique frames
            parts = line.split()
            for p in parts:
                if p.isdigit():
                    stats["n_unique"] = int(p)
                    break
        if "tốc độ suy luận" in line.lower() or "fps" in line.lower():
            pass

    if os.path.exists(audit_path):
        try:
            with open(audit_path, "r") as f:
                audit = json.load(f)
            stats["audit_pass"] = audit.get("frames_matched", False)
            stats["ai_fps"] = audit.get("ai_steady_fps")
            stats["pipeline_fps"] = audit.get("average_pipeline_fps")
            stats["output_size_mb"] = audit.get("output_file_size_mb")
            stats["wall_time_sec"] = audit.get("total_processing_seconds")
        except Exception as e:
            print("Audit read error:", e)

    return stats

def main():
    sample_video = "/Users/xis108/Desktop/upscale/samples/sample_1min_bd_420.mkv"
    base_scratch = "/Users/xis108/Desktop/upscale/scratch/c2_runs"
    os.makedirs(base_scratch, exist_ok=True)
    
    policies = [
        {"name": "default", "args": ["--dedup-policy", "default"]},
        {"name": "sensitive-b", "args": ["--dedup-policy", "sensitive-b"]},
        {"name": "drift-0.5", "args": ["--dedup-policy", "drift", "--drift-delta", "0.5"]},
        {"name": "off", "args": ["--dedup-policy", "off"]}
    ]
    
    results = {}
    
    for p in policies:
        p_name = p["name"]
        print(f"\n==================================================")
        print(f"=== EXECUTING C2 RUN: {p_name.upper()} ===")
        print(f"==================================================")
        
        run_dir = os.path.join(base_scratch, p_name.replace(".", "_"))
        os.makedirs(run_dir, exist_ok=True)
        out_file = os.path.join(run_dir, f"sample_1min_{p_name.replace('.', '_')}_4K.mkv")
        
        cmd = [
            "./venv/bin/python", "upscale.py",
            sample_video,
            "-o", out_file,
            "--no-subs", "--no-embed"
        ] + p["args"]
        
        code, duration, stdout, stderr = run_cmd(cmd)
        if code != 0:
            print(f"FAILED RUN FOR {p_name}!")
            results[p_name] = {"status": "FAIL", "code": code}
            continue
            
        audit_file = os.path.splitext(out_file)[0] + ".audit.json"
        # If organized into audit/ subfolder:
        if not os.path.exists(audit_file):
            alt_audit = os.path.join(run_dir, "audit", os.path.basename(audit_file))
            if os.path.exists(alt_audit):
                audit_file = alt_audit
                
        stats = parse_log_stats(stdout, audit_file)
        if not stats["wall_time_sec"]:
            stats["wall_time_sec"] = duration
        stats["out_file"] = out_file
        stats["audit_file"] = audit_file
        results[p_name] = stats
        print(f"Result for {p_name}: n_unique={stats['n_unique']}, wall_time={stats['wall_time_sec']:.1f}s, ai_fps={stats['ai_fps']}, audit_pass={stats['audit_pass']}")

    # Save C2 measurements
    res_path = "/Users/xis108/Desktop/upscale/scratch/results_c2.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved C2 measurements to {res_path}")

    # ==================================================
    # C3: BLIND TEST PACK CREATION
    # ==================================================
    print(f"\n==================================================")
    print(f"=== EXECUTING C3: CREATING BLIND TEST PACK ===")
    print(f"==================================================")
    
    pack_dir = "/Users/xis108/Documents/BDModelTest/PACK_DEDUP"
    key_dir = "/Users/xis108/bd_model_test/.key"
    os.makedirs(pack_dir, exist_ok=True)
    os.makedirs(key_dir, exist_ok=True)
    
    # Shuffle labels
    labels = ["Clip_A", "Clip_B", "Clip_C", "Clip_D"]
    rng = secrets.SystemRandom()
    shuffled_labels = list(labels)
    rng.shuffle(shuffled_labels)
    
    key_mapping = {}
    label_to_policy = {}
    for i, p in enumerate(policies):
        p_name = p["name"]
        assigned_label = shuffled_labels[i]
        key_mapping[p_name] = assigned_label
        label_to_policy[assigned_label] = p_name
        
    key_file = os.path.join(key_dir, "dedup_key.json")
    with open(key_file, "w") as f:
        json.dump({
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "key_mapping_policy_to_label": key_mapping,
            "label_to_policy": label_to_policy
        }, f, indent=2)
    os.chmod(key_file, 0o600)
    print(f"Secret key written to {key_file} (chmod 600, not displayed)")

    # Copy stripped full 4K clips
    for label, p_name in label_to_policy.items():
        src_clip = results[p_name]["out_file"]
        dst_clip = os.path.join(pack_dir, f"{label}.mkv")
        print(f"Exporting {label}.mkv (stripped metadata)...")
        strip_cmd = [
            "ffmpeg", "-y", "-i", src_clip,
            "-c", "copy",
            "-map_metadata", "-1",
            dst_clip
        ]
        subprocess.run(strip_cmd, check=True, capture_output=True)

        # 1. Đoạn zoom 10s: 00:38 -> 00:48 (10s)
        zoom_dst = os.path.join(pack_dir, f"{label}_01_zoom10s.mkv")
        subprocess.run([
            "ffmpeg", "-y", "-ss", "00:00:38", "-i", src_clip,
            "-t", "10", "-c", "copy", "-map_metadata", "-1",
            zoom_dst
        ], check=True, capture_output=True)

        # 2. Đoạn motion lia camera: 00:20 -> 00:30 (10s)
        pan_dst = os.path.join(pack_dir, f"{label}_02_pan10s.mkv")
        subprocess.run([
            "ffmpeg", "-y", "-ss", "00:00:20", "-i", src_clip,
            "-t", "10", "-c", "copy", "-map_metadata", "-1",
            pan_dst
        ], check=True, capture_output=True)

        # 3. Đoạn climax chuyển động: 00:50 -> 01:00 (10s)
        action_dst = os.path.join(pack_dir, f"{label}_03_action10s.mkv")
        subprocess.run([
            "ffmpeg", "-y", "-ss", "00:00:50", "-i", src_clip,
            "-t", "10", "-c", "copy", "-map_metadata", "-1",
            action_dst
        ], check=True, capture_output=True)

    # Create evaluation template CSV
    csv_path = os.path.join(pack_dir, "scores_dedup.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Clip,Smoothness_1_10,Zoom_Pan_Quality_1_10,Micro_Stutter_1_10,Overall_Score_1_10,Comments\n")
        for lbl in sorted(labels):
            f.write(f"{lbl},,,,, \n")
            f.write(f"{lbl}_01_zoom10s,,,,, \n")
            f.write(f"{lbl}_02_pan10s,,,,, \n")
            f.write(f"{lbl}_03_action10s,,,,, \n")
    print(f"Created scores_dedup.csv at {csv_path}")
    print("\n>>> ALL C2 AND C3 RUNS COMPLETED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    main()
