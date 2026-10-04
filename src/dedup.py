# -*- coding: utf-8 -*-
"""
src/dedup.py
Module quản lý cơ chế Deduplication & Concat Re-expansion cho Anime Video Upscale:
- Phân tích toàn tập bằng bộ lọc mpdecimate (Preflight Scan lập plan.json 1 lần duy nhất)
- Bắt buộc ép giữ frame đầu mỗi chunk (bảo toàn 100% tính độc lập và Checkpoint Resume)
- Lọc bỏ khung hình trùng lặp trước khi đưa vào NCNN Vulkan Metal (giảm ~50% tải GPU)
- Tái tạo khung hình 4K (Re-expansion) bằng FFmpeg Concat Demuxer không tốn RAM/Disk
- Xác minh giao dịch phân đoạn (Transactional Segment verification) chuẩn xác từng frame
"""

import os
import re
import json
import shutil
import subprocess
from typing import List, Dict, Any, Optional

from src.config import (
    DEFAULT_DEDUP_HI,
    DEFAULT_DEDUP_LO,
    DEFAULT_DEDUP_FRAC,
    HEVC_ENCODER,
    HEVC_PIX_FMT,
    HEVC_COLOR_RANGE,
    HEVC_COLORSPACE,
    HEVC_COLOR_PRIMARIES,
    HEVC_COLOR_TRC,
    HEVC_TAG
)


def scan_dedup_plan(
    input_path: str,
    plan_path: str,
    segment_seconds: int = 8,
    fps_num: int = 24000,
    fps_den: int = 1001,
    hi: int = DEFAULT_DEDUP_HI,
    lo: int = DEFAULT_DEDUP_LO,
    frac: float = DEFAULT_DEDUP_FRAC
) -> Dict[str, Any]:
    """
    Quét phân tích toàn tập bằng filter mpdecimate để lập plan.json.
    Chỉ chạy 1 lần duy nhất trước khi upscale, overhead cực nhỏ (~1-2 phút CPU).
    """
    # 1. Ghim phiên bản FFmpeg để đảm bảo tính tương thích log debug
    ver_proc = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ffmpeg_ver = ver_proc.stdout.splitlines()[0] if ver_proc.stdout else "unknown"

    # 2. Đếm tổng số packet/frame chính xác từ luồng video bằng ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    expected_frames: Optional[int] = None
    if probe_res.returncode == 0 and probe_res.stdout.strip().isdigit():
        expected_frames = int(probe_res.stdout.strip())

    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", f"mpdecimate=hi={hi}:lo={lo}:frac={frac}",
        "-v", "debug",
        "-f", "null", "-"
    ]

    pattern = re.compile(r"\[Parsed_mpdecimate_0.*?\]\s+.*?\b(keep|drop)\s+pts:")
    frame_actions: List[str] = []

    # Thực thi ffmpeg và bắt dòng log debug
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    if proc.stderr:
        for line in proc.stderr:
            m = pattern.search(line)
            if m:
                frame_actions.append(m.group(1))

    proc.wait()
    if proc.returncode != 0 and not frame_actions:
        raise RuntimeError(f"Lỗi khi quét mpdecimate trên file: {input_path}")

    total_frames = len(frame_actions)
    if total_frames == 0:
        raise ValueError("Không quét được frame nào từ file video.")

    # 3. Assertions bất biến: tổng keep + drop phải bằng len(frame_actions) và khớp ffprobe
    raw_keep = sum(1 for a in frame_actions if a == "keep")
    raw_drop = sum(1 for a in frame_actions if a == "drop")
    assert raw_keep + raw_drop == total_frames, (
        f"Parse log debug không khớp: keep ({raw_keep}) + drop ({raw_drop}) != tổng actions ({total_frames})"
    )
    if expected_frames is not None:
        assert total_frames == expected_frames, (
            f"Số frame quét từ debug log ({total_frames}) không khớp số frame ffprobe đếm được ({expected_frames})!"
        )

    # Số frame lý tưởng mỗi chunk 8s
    fps_val = fps_num / fps_den
    frames_per_chunk = int(round(segment_seconds * fps_val))

    # BẮT BUỘC ÉP GIỮ FRAME ĐẦU MỖI CHUNK per AGENTS.md Invariant:
    # kept = sorted(set(kept) | set(range(0, total, frames_per_chunk)))
    chunk_start_indices = set(range(0, total_frames, frames_per_chunk))
    for c_start in chunk_start_indices:
        if c_start < total_frames:
            frame_actions[c_start] = "keep"

    # Chia các frame thành từng chunk
    chunks_meta = []
    total_kept_count = 0
    num_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk

    for c_idx in range(num_chunks):
        c_start = c_idx * frames_per_chunk
        c_end = min((c_idx + 1) * frames_per_chunk, total_frames)
        c_frames = frame_actions[c_start:c_end]
        c_len = len(c_frames)

        # Xây dựng danh sách runs cho chunk này
        runs: List[Dict[str, int]] = []
        curr_kept = 0
        curr_count = 0
        kept_indices = []

        for idx, act in enumerate(c_frames):
            if act == "keep":
                if curr_count > 0:
                    runs.append({"frame_idx": curr_kept, "count": curr_count})
                curr_kept = idx
                curr_count = 1
                kept_indices.append(idx)
            else:
                curr_count += 1

        if curr_count > 0:
            runs.append({"frame_idx": curr_kept, "count": curr_count})

        total_kept_count += len(kept_indices)

        chunks_meta.append({
            "chunk_idx": c_idx,
            "start_frame": c_start,
            "end_frame": c_end,
            "total_frames": c_len,
            "kept_count": len(kept_indices),
            "kept_indices": kept_indices,
            "runs": runs
        })

    dedup_ratio = (1.0 - (total_kept_count / total_frames)) * 100.0 if total_frames > 0 else 0.0

    plan_data = {
        "source_file": os.path.basename(input_path),
        "ffmpeg_version": ffmpeg_ver,
        "total_frames": total_frames,
        "raw_kept_frames": raw_keep,
        "raw_dropped_frames": raw_drop,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "fps": round(fps_val, 3),
        "segment_seconds": segment_seconds,
        "frames_per_chunk": frames_per_chunk,
        "total_chunks": num_chunks,
        "total_kept_frames": total_kept_count,
        "dedup_ratio_percent": round(dedup_ratio, 2),
        "chunks": chunks_meta
    }

    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_data, f, indent=2, ensure_ascii=False)

    return plan_data


def load_or_create_plan(
    input_path: str,
    plan_path: str,
    segment_seconds: int = 8,
    fps_num: int = 24000,
    fps_den: int = 1001,
    resume: bool = True
) -> Dict[str, Any]:
    """
    Nạp plan.json đã có từ phiên chạy trước nếu bật resume;
    nếu chưa có thì tự động quét preflight và lưu plan.json.
    """
    if resume and os.path.exists(plan_path):
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("total_frames", 0) > 0 and len(data.get("chunks", [])) > 0:
                return data
        except Exception:
            pass

    return scan_dedup_plan(
        input_path=input_path,
        plan_path=plan_path,
        segment_seconds=segment_seconds,
        fps_num=fps_num,
        fps_den=fps_den
    )


def prune_redundant_frames(
    frames_dir: str,
    kept_indices: List[int],
    fmt: str = "png"
) -> int:
    """
    Xóa ngay lập tức các khung hình trùng lặp khỏi RAM Disk trước khi gọi AI,
    chỉ giữ lại các file tương ứng với kept_indices (0-based -> frame_%04d 1-based).
    Trả về số lượng frame thực tế được giữ lại để upscale.
    """
    kept_set = set(kept_indices)
    removed_count = 0

    all_files = sorted(os.listdir(frames_dir))
    for fname in all_files:
        if not fname.endswith(f".{fmt}"):
            continue
        # Tách số thứ tự (frame_0001.png -> idx 0)
        m = re.search(r"(\d+)", fname)
        if m:
            frame_1based = int(m.group(1))
            frame_0based = frame_1based - 1
            if frame_0based not in kept_set:
                file_path = os.path.join(frames_dir, fname)
                try:
                    os.remove(file_path)
                    removed_count += 1
                except OSError:
                    pass

    return len(kept_set)


def generate_concat_demuxer_script(
    runs: List[Dict[str, int]],
    frames_out_dir: str,
    concat_script_path: str,
    fps_num: int = 24000,
    fps_den: int = 1001,
    fmt: str = "png"
) -> str:
    """
    Tạo file script Concat Demuxer cho FFmpeg để tái tạo 4K CFR:
    Mỗi file ảnh 4K được nạp duy nhất 1 lần vào bộ nhớ và nhân bản bằng duration.
    """
    frame_dur = fps_den / fps_num

    with open(concat_script_path, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for r in runs:
            # 0-based frame_idx -> frame%08d.png (đồng bộ với extract_frames)
            img_name = f"frame{r['frame_idx'] + 1:08d}.{fmt}"
            img_path = os.path.join(frames_out_dir, img_name)
            dur = r["count"] * frame_dur
            f.write(f"file '{img_path}'\n")
            f.write(f"duration {dur:.6f}\n")

        # Quy chuẩn FFmpeg Concat Demuxer: dòng cuối cùng lặp lại file mà không có duration
        last_img_name = f"frame{runs[-1]['frame_idx'] + 1:08d}.{fmt}"
        last_img_path = os.path.join(frames_out_dir, last_img_name)
        f.write(f"file '{last_img_path}'\n")

    return concat_script_path


def encode_chunk_reexpanded(
    concat_script_path: str,
    output_path: str,
    expected_frames: int,
    fps_num: int = 24000,
    fps_den: int = 1001,
    bitrate: str = "28M"
) -> None:
    """
    Mã hóa phân đoạn 4K qua FFmpeg Concat Demuxer + VideoToolbox HEVC 10-bit:
    - Chốt cờ CFR: -video_track_timescale fps_num -frames:v expected_frames
    - Chuỗi ma trận màu sắc BT.709 Full RGB ➔ Limited YUV p010le
    - Giao dịch an toàn: ghi ra .tmp.mp4, xác minh đủ số frame rồi mới rename
    """
    tmp_output = f"{output_path}.tmp.mp4"
    if os.path.exists(tmp_output):
        os.remove(tmp_output)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_script_path,
        "-vf", (
            f"scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv,"
            f"fps={fps_num}/{fps_den}"
        ),
        "-pix_fmt", HEVC_PIX_FMT,
        "-c:v", HEVC_ENCODER,
        "-b:v", bitrate,
        "-color_range", HEVC_COLOR_RANGE,
        "-colorspace", HEVC_COLORSPACE,
        "-color_primaries", HEVC_COLOR_PRIMARIES,
        "-color_trc", HEVC_COLOR_TRC,
        "-tag:v", HEVC_TAG,
        "-video_track_timescale", str(fps_num),
        "-frames:v", str(expected_frames),
        tmp_output
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr or "Không có thông tin stderr"
        raise RuntimeError(f"Lỗi mã hóa FFmpeg Concat Re-expansion (exit {e.returncode}):\n{err_msg}")

    # Xác minh giao dịch (Transactional verification)
    cmd_verify = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        tmp_output
    ]
    probe_res = subprocess.run(cmd_verify, capture_output=True, text=True, check=True)
    actual_frames = int(probe_res.stdout.strip())

    if actual_frames != expected_frames:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        raise RuntimeError(
            f"Lỗi giao dịch phân đoạn: Số frame thực tế ({actual_frames}) "
            f"không khớp số frame kỳ vọng ({expected_frames})!"
        )

    # Di chuyển nguyên tử sang file hoàn tất
    shutil.move(tmp_output, output_path)


def spawn_background_preflight(
    video_path: str,
    output_dir: Optional[str] = None,
    segment_seconds: int = 8,
    logger=None
) -> Optional[subprocess.Popen]:
    """
    Kích hoạt tiến trình quét mpdecimate lập plan.json trước cho tập kế tiếp.
    Chạy với độ ưu tiên thấp (nice -n 10) trong nền lúc GPU đang bận xử lý tập trước.
    """
    import sys
    if os.environ.get("BENCHMARK_MODE") == "1":
        if logger:
            logger.warning("⚡ [BENCHMARK MODE] Từ chối kích hoạt spawn_background_preflight.", to_console=True)
        return None
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = output_dir or "out"
        work_dir = os.path.join(out_dir, f".temp_{base_name}")
        os.makedirs(work_dir, exist_ok=True)
        plan_file = os.path.join(work_dir, "plan.json")
        if os.path.exists(plan_file) and os.path.getsize(plan_file) > 100:
            return None

        # Gọi python chạy preflight với nice -n 10
        cmd = [
            "nice", "-n", "10",
            sys.executable, "-c",
            f"import sys; from src.dedup import load_or_create_plan; from src.ffmpeg_ops import probe_video; "
            f"meta = probe_video('{video_path}'); "
            f"load_or_create_plan('{video_path}', '{plan_file}', {segment_seconds}, meta['fps_num'], meta['fps_den'])"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        msg = f"⚡ [Multi-Episode Queue] Đã kích hoạt Preflight nền (nice -n 10) cho tập kế tiếp: {os.path.basename(video_path)}"
        if logger:
            logger.info(msg, to_console=True)
        else:
            print(msg)
        return proc
    except Exception:
        return None

