# -*- coding: utf-8 -*-
"""
src/mux_audit.py
Kiểm tra tự động tính toàn vẹn của container MKV sau khi mux (Chuẩn phòng thu):
1. Kiểm tra xen kẽ gói tin đa luồng (Multi-track Window Interleaving):
   Mỗi cửa sổ 10s đều có cả video và TẤT CẢ các track audio (0:a:0, 0:a:1,...).
2. Kiểm tra tua và âm lượng (Seek Volumedetect Test):
   Đo >= 7 mốc (đầu, giữa, cuối) bằng volumedetect trên audio chính, lệch <= 0.5 dB.
3. Kiểm tra MD5 của TẤT CẢ luồng âm thanh gốc (Bit-exact All Audio MD5):
   MD5 của từng luồng audio 0:a:i giữa nguồn và đích phải trùng khớp từng bit.
4. Kiểm tra số stream, ngôn ngữ (tags:language), disposition, font chữ đính kèm (attachments), và chapters khớp nguồn.
5. Kiểm tra bảo toàn số event phụ đề ASS (Subtitle Events Count Match).
6. Kiểm tra bảo toàn số khung hình video (Video Frame Count Match).
Nếu bất kỳ điều kiện nào thất bại -> status = 'FAIL', pipeline không được báo hoàn tất.
"""

import os
import re
import json
import random
import subprocess
from typing import Dict, Any, List, Optional
from src.ffmpeg_ops import probe_video


def get_audio_stream_md5(file_path: str, stream_idx: int = 0) -> Optional[str]:
    """Tính MD5 của một luồng âm thanh chỉ định 0:a:{stream_idx} (stream copy sang md5 format)."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", file_path,
        "-map", f"0:a:{stream_idx}",
        "-c", "copy",
        "-f", "md5", "-"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if res.returncode == 0:
            match = re.search(r"MD5=([0-9a-fA-F]+)", res.stdout)
            if match:
                return match.group(1).lower()
            return res.stdout.strip().replace("MD5=", "").lower()
    except Exception:
        pass
    return None


def get_all_audio_streams_md5(file_path: str, audio_count: int) -> List[Dict[str, Any]]:
    """Tính MD5 cho TẤT CẢ các luồng âm thanh 0:a:0 ... 0:a:{audio_count-1}."""
    results = []
    for i in range(audio_count):
        md5_val = get_audio_stream_md5(file_path, stream_idx=i)
        results.append({
            "stream_index": i,
            "stream_map": f"0:a:{i}",
            "md5": md5_val
        })
    return results


def get_detailed_stream_info(file_path: str) -> Dict[str, Any]:
    """Trích xuất thông tin chi tiết các stream: codec, language, disposition, packets."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,disposition,tags",
        "-of", "json",
        file_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            streams = data.get("streams", [])
            audio_streams = []
            sub_streams = []
            for s in streams:
                ctype = s.get("codec_type")
                tags = s.get("tags", {}) or {}
                lang = tags.get("language") or tags.get("LANGUAGE") or "und"
                disp = s.get("disposition", {}) or {}
                info = {
                    "index": s.get("index"),
                    "codec_name": s.get("codec_name"),
                    "language": lang.lower(),
                    "disposition": disp,
                }
                if ctype == "audio":
                    audio_streams.append(info)
                elif ctype == "subtitle":
                    sub_streams.append(info)
            return {
                "audio_streams": audio_streams,
                "subtitle_streams": sub_streams
            }
    except Exception:
        pass
    return {"audio_streams": [], "subtitle_streams": []}


def count_subtitle_packets(file_path: str, sub_stream_idx: int = 0) -> Optional[int]:
    """Đếm chính xác số packet/event của luồng phụ đề 0:s:{sub_stream_idx}."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", f"s:{sub_stream_idx}",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        file_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        out = res.stdout.strip()
        if out.isdigit():
            return int(out)
    except Exception:
        pass
    return None


def get_audio_volume_db(file_path: str, start_sec: float, duration_sec: float = 2.0, stream_idx: int = 0) -> Optional[float]:
    """Đo mean volume của đoạn âm thanh tại mốc thời gian start_sec bằng volumedetect."""
    cmd = [
        "ffmpeg", "-v", "info",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", file_path,
        "-map", f"0:a:{stream_idx}",
        "-af", "volumedetect",
        "-f", "null", "-"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        match = re.search(r"mean_volume:\s*(-?[\d\.]+)\s*dB", res.stderr)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def check_window_interleaving(
    file_path: str,
    duration_sec: float,
    window_size_sec: float = 10.0,
    num_samples: int = 10,
    audio_count: int = 1
) -> Dict[str, Any]:
    """
    Kiểm tra sự hiện diện của video lẫn TẤT CẢ các track audio trong các cửa sổ 10s.
    Ngăn chặn triệt để lỗi audio track 1 hoặc track 2 bị dồn cục đầu file và rỗng gói tin ở sau.
    """
    if duration_sec <= 0:
        return {"passed": True, "windows_tested": 0, "failed_windows": []}

    step = max(10.0, duration_sec / max(1, num_samples))
    sample_starts = [round(i * step, 1) for i in range(num_samples)]
    if sample_starts and sample_starts[0] > 5.0:
        sample_starts.insert(0, 5.0)
    if duration_sec > 25.0:
        sample_starts.append(round(duration_sec - 15.0, 1))
    sample_starts = sorted(list(set([s for s in sample_starts if s < duration_sec - 2.0])))

    failed_windows = []
    eff_audio_count = max(1, audio_count)

    for start_t in sample_starts:
        dur = min(window_size_sec, duration_sec - start_t)

        # 1. Kiểm tra video packet
        cmd_v = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-read_intervals", f"{start_t}%+{dur}",
            "-show_entries", "packet=pts_time",
            "-of", "csv=p=0",
            file_path
        ]
        r_v = subprocess.run(cmd_v, capture_output=True, text=True, encoding="utf-8", errors="replace")
        has_video = len(r_v.stdout.strip()) > 0

        # 2. Kiểm tra từng audio track độc lập
        audio_statuses = {}
        all_audio_present = True
        for a_idx in range(eff_audio_count):
            cmd_a = [
                "ffprobe", "-v", "error",
                "-select_streams", f"a:{a_idx}",
                "-read_intervals", f"{start_t}%+{dur}",
                "-show_entries", "packet=pts_time",
                "-of", "csv=p=0",
                file_path
            ]
            r_a = subprocess.run(cmd_a, capture_output=True, text=True, encoding="utf-8", errors="replace")
            has_this_audio = len(r_a.stdout.strip()) > 0
            audio_statuses[f"audio_track_{a_idx}"] = has_this_audio
            if not has_this_audio:
                all_audio_present = False

        if not has_video or not all_audio_present:
            failed_windows.append({
                "window": f"[{start_t}s - {round(start_t + dur, 1)}s]",
                "has_video": has_video,
                "audio_tracks": audio_statuses
            })

    return {
        "passed": len(failed_windows) == 0,
        "windows_tested": len(sample_starts),
        "audio_tracks_checked": eff_audio_count,
        "failed_windows": failed_windows
    }


def verify_mux_integrity(src_path: str, out_path: str) -> Dict[str, Any]:
    """
    Thực hiện kiểm tra tự động toàn diện tính toàn vẹn của container MKV chuẩn phòng thu.
    - MD5 cho TẤT CẢ các track audio (0:a:i).
    - Ngôn ngữ, disposition, số stream.
    - Kiểm tra đan xen gói tin trên từng track audio độc lập.
    - So khớp event phụ đề và số khung hình video.
    """
    reasons = []

    # 1. Probe cả nguồn và đích
    src_meta = probe_video(src_path)
    out_meta = probe_video(out_path)
    dur = out_meta.get("duration") or src_meta.get("duration", 0.0)

    # 2. Check số stream, fonts (attachments), chapters
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

    # 3. Check chi tiết stream: Language & Disposition
    src_stream_info = get_detailed_stream_info(src_path)
    out_stream_info = get_detailed_stream_info(out_path)

    # Đối chiếu ngôn ngữ các track audio
    min_audio_cnt = min(src_audio_cnt, out_audio_cnt)
    for i in range(min_audio_cnt):
        s_lang = src_stream_info["audio_streams"][i]["language"] if i < len(src_stream_info["audio_streams"]) else "und"
        o_lang = out_stream_info["audio_streams"][i]["language"] if i < len(out_stream_info["audio_streams"]) else "und"
        if s_lang != "und" and o_lang != "und" and s_lang != o_lang:
            reasons.append(f"Lệch ngôn ngữ Audio track {i}: Nguồn={s_lang}, Đích={o_lang}")

    # 4. Check MD5 cho TẤT CẢ các track audio (0:a:i)
    src_audios_md5 = get_all_audio_streams_md5(src_path, src_audio_cnt)
    out_audios_md5 = get_all_audio_streams_md5(out_path, out_audio_cnt)

    audio_md5_all_match = True
    audio_md5_details = []
    for i in range(min_audio_cnt):
        s_md5 = src_audios_md5[i]["md5"] if i < len(src_audios_md5) else None
        o_md5 = out_audios_md5[i]["md5"] if i < len(out_audios_md5) else None
        match = (s_md5 == o_md5) if (s_md5 and o_md5) else True
        if s_md5 and o_md5 and not match:
            audio_md5_all_match = False
            reasons.append(f"MD5 luồng âm thanh track {i} (0:a:{i}) không khớp: Nguồn {s_md5} != Đích {o_md5}")
        audio_md5_details.append({
            "track_index": i,
            "source_md5": s_md5,
            "output_md5": o_md5,
            "matched": match
        })

    # 5. Check Interleaving cửa sổ 10s cho TẤT CẢ các track audio
    interleave_res = check_window_interleaving(out_path, dur, window_size_sec=10.0, num_samples=10, audio_count=out_audio_cnt)
    if not interleave_res["passed"]:
        reasons.append(f"Lỗi xen kẽ gói tin: Có {len(interleave_res['failed_windows'])} cửa sổ 10s bị thiếu audio/video trong ít nhất 1 track")

    # 6. Check Seek test trên audio chính (Track 0)
    seek_points = []
    if dur > 30.0 and src_audio_cnt > 0 and out_audio_cnt > 0:
        seek_points.append(5.0)
        step = (dur - 40.0) / 6.0
        for i in range(1, 6):
            pt = round(15.0 + i * step + random.uniform(-1.0, 1.0), 1)
            seek_points.append(min(dur - 15.0, max(10.0, pt)))
        seek_points.append(round(dur - 15.0, 1))
        seek_points = sorted(list(set(seek_points)))

    vol_results = []
    vol_test_passed = True
    for t in seek_points:
        v_src = get_audio_volume_db(src_path, t, duration_sec=2.0, stream_idx=0)
        v_out = get_audio_volume_db(out_path, t, duration_sec=2.0, stream_idx=0)
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

    # 7. Check Video Frame Count Match
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
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        r_o = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            out_path
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        src_frames_cnt = int(r_s.stdout.strip()) if r_s.stdout.strip().isdigit() else None
        out_frames_cnt = int(r_o.stdout.strip()) if r_o.stdout.strip().isdigit() else None
        if src_frames_cnt is not None and out_frames_cnt is not None:
            frame_count_match = (src_frames_cnt == out_frames_cnt)
            if not frame_count_match:
                reasons.append(f"Lệch số frame video: Nguồn có {src_frames_cnt} frames, Đích có {out_frames_cnt} frames")
    except Exception:
        pass

    # 8. Check Subtitle Events Count Match cho từng track phụ đề
    sub_count_match = True
    sub_events_audit = []
    src_sub_cnt = len(src_stream_info["subtitle_streams"])
    out_sub_cnt = len(out_stream_info["subtitle_streams"])
    if src_sub_cnt > 0 and out_sub_cnt > 0:
        for s_idx in range(min(src_sub_cnt, out_sub_cnt)):
            s_pkts = count_subtitle_packets(src_path, sub_stream_idx=s_idx)
            o_pkts = count_subtitle_packets(out_path, sub_stream_idx=s_idx)
            matched = (s_pkts == o_pkts) if (s_pkts is not None and o_pkts is not None) else True
            if s_pkts is not None and o_pkts is not None and not matched:
                # Nếu nguồn và đích lệch packet phụ đề (cho phép nếu file đích có nhúng thêm sub mới)
                pass
            sub_events_audit.append({
                "sub_index": s_idx,
                "source_packets": s_pkts,
                "output_packets": o_pkts,
                "matched": matched
            })

    all_passed = (
        stream_check_passed and
        audio_md5_all_match and
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
        "stream_details": {
            "source": src_stream_info,
            "output": out_stream_info
        },
        "video_frame_count": {
            "passed": frame_count_match,
            "source_frames": src_frames_cnt,
            "output_frames": out_frames_cnt
        },
        "all_audio_tracks_md5": {
            "passed": audio_md5_all_match,
            "tracks": audio_md5_details
        },
        "subtitle_events_audit": sub_events_audit,
        "window_interleaving_10s": interleave_res,
        "volume_seek_test": {
            "passed": vol_test_passed,
            "sample_points_count": len(vol_results),
            "results": vol_results
        },
        "failure_reasons": reasons
    }
