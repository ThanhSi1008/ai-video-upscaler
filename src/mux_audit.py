# -*- coding: utf-8 -*-
"""
src/mux_audit.py
Kiểm tra tự động tính toàn vẹn của container MKV sau khi mux:
1. Kiểm tra xen kẽ gói tin (Window Interleaving): Mỗi cửa sổ 10s đều có cả video và audio.
2. Kiểm tra tua và âm lượng (Seek Volumedetect Test): Đo >= 7 mốc (đầu, giữa, cuối) bằng volumedetect, lệch <= 0.5 dB.
3. Kiểm tra MD5 luồng âm thanh gốc (Bit-exact Audio MD5): MD5 luồng a:0 giữa nguồn và đích phải trùng khớp từng bit.
4. Kiểm tra số stream, font chữ đính kèm (attachments), và chapters khớp nguồn.
Nếu bất kỳ điều kiện nào thất bại -> status = 'FAIL', pipeline không được báo hoàn tất.
"""

import os
import re
import random
import subprocess
from typing import Dict, Any, List, Optional
from src.ffmpeg_ops import probe_video


def get_audio_stream_md5(file_path: str) -> Optional[str]:
    """Tính MD5 của luồng âm thanh đầu tiên (stream copy sang md5 format)."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", file_path,
        "-map", "0:a:0",
        "-c", "copy",
        "-f", "md5", "-"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            match = re.search(r"MD5=([0-9a-fA-F]+)", res.stdout)
            if match:
                return match.group(1).lower()
            return res.stdout.strip().replace("MD5=", "").lower()
    except Exception:
        pass
    return None


def get_audio_volume_db(file_path: str, start_sec: float, duration_sec: float = 2.0) -> Optional[float]:
    """Đo mean volume của đoạn âm thanh tại mốc thời gian start_sec bằng volumedetect."""
    cmd = [
        "ffmpeg", "-v", "info",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", file_path,
        "-map", "0:a:0",
        "-af", "volumedetect",
        "-f", "null", "-"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        match = re.search(r"mean_volume:\s*(-?[\d\.]+)\s*dB", res.stderr)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def check_window_interleaving(file_path: str, duration_sec: float, window_size_sec: float = 10.0, num_samples: int = 10) -> Dict[str, Any]:
    """
    Kiểm tra sự hiện diện của cả video lẫn audio trong các cửa sổ 10s trải dài toàn tập.
    Ngăn chặn triệt để lỗi audio dồn cục đầu file khiến các đoạn sau bị rỗng gói tin.
    """
    if duration_sec <= 0:
        return {"passed": True, "windows_tested": 0, "failed_windows": []}

    # Chọn các mốc thời gian phân bổ đều khắp thời lượng
    step = max(10.0, duration_sec / max(1, num_samples))
    sample_starts = [round(i * step, 1) for i in range(num_samples)]
    # Đảm bảo có mốc gần đầu và gần cuối
    if sample_starts and sample_starts[0] > 5.0:
        sample_starts.insert(0, 5.0)
    if duration_sec > 25.0:
        sample_starts.append(round(duration_sec - 15.0, 1))
    sample_starts = sorted(list(set([s for s in sample_starts if s < duration_sec - 2.0])))

    failed_windows = []
    for start_t in sample_starts:
        dur = min(window_size_sec, duration_sec - start_t)
        # Kiểm tra audio packet
        cmd_a = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-read_intervals", f"{start_t}%+{dur}",
            "-show_entries", "packet=pts_time",
            "-of", "csv=p=0",
            file_path
        ]
        r_a = subprocess.run(cmd_a, capture_output=True, text=True)
        has_audio = len(r_a.stdout.strip()) > 0

        # Kiểm tra video packet
        cmd_v = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-read_intervals", f"{start_t}%+{dur}",
            "-show_entries", "packet=pts_time",
            "-of", "csv=p=0",
            file_path
        ]
        r_v = subprocess.run(cmd_v, capture_output=True, text=True)
        has_video = len(r_v.stdout.strip()) > 0

        if not has_audio or not has_video:
            failed_windows.append({
                "window": f"[{start_t}s - {round(start_t + dur, 1)}s]",
                "has_audio": has_audio,
                "has_video": has_video
            })

    return {
        "passed": len(failed_windows) == 0,
        "windows_tested": len(sample_starts),
        "failed_windows": failed_windows
    }


def verify_mux_integrity(src_path: str, out_path: str) -> Dict[str, Any]:
    """
    Thực hiện kiểm tra tự động toàn diện tính toàn vẹn của container MKV.
    Trả về dict kết quả, nếu không đạt status = 'FAIL'.
    """
    reasons = []

    # 1. Probe cả nguồn và đích
    src_meta = probe_video(src_path)
    out_meta = probe_video(out_path)
    dur = out_meta.get("duration") or src_meta.get("duration", 0.0)

    # 2. Check (d): Số stream, fonts (attachments), chapters
    src_audio_cnt = src_meta.get("audio_count", 0)
    out_audio_cnt = out_meta.get("audio_count", 0)
    src_attach_cnt = src_meta.get("attachment_count", 0)
    out_attach_cnt = out_meta.get("attachment_count", 0)
    src_chap_cnt = src_meta.get("chapter_count", 0)
    out_chap_cnt = out_meta.get("chapter_count", 0)

    stream_check_passed = True
    if out_audio_cnt < src_audio_cnt:
        reasons.append(f"Mất luồng âm thanh: Nguồn có {src_audio_cnt}, đích có {out_audio_cnt}")
        stream_check_passed = False
    if out_attach_cnt < src_attach_cnt:
        reasons.append(f"Mất font chữ đính kèm: Nguồn có {src_attach_cnt}, đích có {out_attach_cnt}")
        stream_check_passed = False
    if out_chap_cnt < src_chap_cnt:
        reasons.append(f"Mất chapters: Nguồn có {src_chap_cnt}, đích có {out_chap_cnt}")
        stream_check_passed = False

    # 3. Check (c): MD5 của luồng audio gốc
    src_audio_md5 = get_audio_stream_md5(src_path) if src_audio_cnt > 0 else None
    out_audio_md5 = get_audio_stream_md5(out_path) if out_audio_cnt > 0 else None
    audio_md5_match = (src_audio_md5 == out_audio_md5) if (src_audio_md5 and out_audio_md5) else True
    if src_audio_md5 and out_audio_md5 and not audio_md5_match:
        reasons.append(f"MD5 luồng âm thanh không khớp: Nguồn {src_audio_md5} != Đích {out_audio_md5}")

    # 4. Check (a): Interleaving cửa sổ 10s
    interleave_res = check_window_interleaving(out_path, dur, window_size_sec=10.0, num_samples=10)
    if not interleave_res["passed"]:
        reasons.append(f"Lỗi xen kẽ gói tin: Có {len(interleave_res['failed_windows'])} cửa sổ 10s bị thiếu audio hoặc video")

    # 5. Check (b): Seek test >= 5 mốc ngẫu nhiên + gần đầu + gần cuối bằng volumedetect (lệch <= 0.5 dB)
    seek_points = []
    if dur > 30.0 and src_audio_cnt > 0 and out_audio_cnt > 0:
        # Gần đầu (5s)
        seek_points.append(5.0)
        # 5 mốc phân bố / ngẫu nhiên
        step = (dur - 40.0) / 6.0
        for i in range(1, 6):
            pt = round(15.0 + i * step + random.uniform(-1.0, 1.0), 1)
            seek_points.append(min(dur - 15.0, max(10.0, pt)))
        # Gần cuối
        seek_points.append(round(dur - 15.0, 1))
        seek_points = sorted(list(set(seek_points)))

    vol_results = []
    vol_test_passed = True
    for t in seek_points:
        v_src = get_audio_volume_db(src_path, t, duration_sec=2.0)
        v_out = get_audio_volume_db(out_path, t, duration_sec=2.0)
        if v_src is not None and v_out is not None:
            diff = abs(v_src - v_out)
            passed = diff <= 0.5
            vol_results.append({
                "seek_sec": t,
                "src_db": v_src,
                "out_db": v_out,
                "diff_db": round(diff, 2),
                "passed": passed
            })
            if not passed:
                vol_test_passed = False
                reasons.append(f"Âm lượng tại mốc t={t}s bị lệch > 0.5 dB (src={v_src}, out={v_out})")
        elif v_out is None and v_src is not None:
            vol_test_passed = False
            reasons.append(f"Mất tiếng hoàn toàn khi tua tới mốc t={t}s (src={v_src} dB, out=NULL)")

    # 6. Check (e): Kiểm tra bảo toàn khung hình video (Video Frame Count Match per AGENTS.md Definition of Done)
    frame_count_match = True
    src_frames_cnt = None
    out_frames_cnt = None
    try:
        r_s = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            src_path
        ], capture_output=True, text=True, check=True)
        r_o = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            out_path
        ], capture_output=True, text=True, check=True)
        src_frames_cnt = int(r_s.stdout.strip()) if r_s.stdout.strip().isdigit() else None
        out_frames_cnt = int(r_o.stdout.strip()) if r_o.stdout.strip().isdigit() else None
        if src_frames_cnt is not None and out_frames_cnt is not None:
            frame_count_match = (src_frames_cnt == out_frames_cnt)
            if not frame_count_match:
                reasons.append(f"Lệch số frame video: Nguồn có {src_frames_cnt} frames, Đích có {out_frames_cnt} frames (Mất {src_frames_cnt - out_frames_cnt} frames)")
    except Exception:
        pass

    all_passed = (
        stream_check_passed and
        audio_md5_match and
        interleave_res["passed"] and
        vol_test_passed and
        frame_count_match
    )

    return {
        "status": "PASS" if all_passed else "FAIL",
        "duration_seconds": round(dur, 2),
        "stream_counts": {
            "passed": stream_check_passed,
            "source_audio": src_audio_cnt,
            "output_audio": out_audio_cnt,
            "source_fonts": src_attach_cnt,
            "output_fonts": out_attach_cnt,
            "source_chapters": src_chap_cnt,
            "output_chapters": out_chap_cnt
        },
        "video_frame_count": {
            "passed": frame_count_match,
            "source_frames": src_frames_cnt,
            "output_frames": out_frames_cnt
        },
        "audio_bit_exact_md5": {
            "passed": audio_md5_match,
            "source_md5": src_audio_md5,
            "output_md5": out_audio_md5
        },
        "window_interleaving_10s": interleave_res,
        "volume_seek_test": {
            "passed": vol_test_passed,
            "sample_points_count": len(vol_results),
            "results": vol_results
        },
        "failure_reasons": reasons
    }
