#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 AI Video Upscaler 4K - CLI Entry Point
Công cụ Upscale Video Anime lên 4K chuyên dụng trên Terminal cho macOS (Apple Silicon M3 Pro).
- Tăng độ phân giải 2x bằng NCNN Vulkan Metal API (realesr-animevideov3).
- Mã hóa phần cứng 10-bit HEVC qua Apple Media Engine (VideoToolbox).
- Bảo vệ tối đa ổ cứng NVMe SSD với bộ nhớ đệm RAM Disk ảo.
- Bảo toàn 100% âm thanh đa kênh, phụ đề đa ngôn ngữ, font chữ và chapters gốc.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

from src.config import (
    DEFAULT_MODEL,
    DEFAULT_ENGINE,
    DEFAULT_PRECISION,
    DEFAULT_BITRATE,
    DEFAULT_SEGMENT_TIME,
    DEFAULT_FRAME_FORMAT,
    load_pending_subs,
    remove_pending_sub,
)
from src.pipeline import run_upscale_pipeline, FailedResumableError
from src.ffmpeg_ops import probe_video
from src.dedup import spawn_background_preflight
from src.models_resolver import list_models_formatted
from src.cleanup import (
    cleanup_checkpoint_dir,
    cleanup_subtitle_extracts,
    organize_output_artifacts,
    clean_all_temporary_data,
)
from pathlib import Path


def check_deadline_and_eta(
    video_path: str,
    deadline_str: str,
    fps_median: float = 1.61,
    fps_p5: float = 1.62,
    video_overhead: float = 930.0,
    subtitles_overhead: float = 300.0,
    eta_margin_pct: float = 0.07,
    allow_partial: bool = False
) -> Tuple[bool, bool, Optional[datetime]]:
    """
    Kiểm tra thời gian hoàn thành dự toán của tập video so với mốc deadline.
    Quyết định dựa trên P5 (1.62 fps) + overhead (video ~930s [n=2] + 300s subs) + 7% ETA margin.
    Trung vị fps: 1.61 (cỡ mẫu n=5 tập).
    Returns: (can_run, is_partial, deadline_dt)
    """
    now = datetime.now()
    try:
        dh, dm = map(int, deadline_str.split(":"))
        deadline_dt = now.replace(hour=dh, minute=dm, second=0, microsecond=0)
        if deadline_dt <= now:
            deadline_dt += timedelta(days=1)
    except Exception as e:
        print(f"⚠️ Định dạng deadline không hợp lệ '{deadline_str}' (cần HH:MM): {e}")
        return True, False, None

    seconds_left = (deadline_dt - now).total_seconds()

    # Đọc số frame độc nhất từ plan.json nếu có
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    candidate_plans = [
        os.path.splitext(video_path)[0] + ".plan.json",
        f"out/{base_name}.plan.json",
        f"out/.temp_{base_name}/plan.json"
    ]
    n_unique = None
    is_measured = False
    for p in candidate_plans:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                    n_unique = pdata.get("total_kept_frames")
                    if n_unique:
                        is_measured = True
                        break
            except Exception:
                pass

    if n_unique is None:
        # Ước lượng thận trọng 60.0% frame giữ nếu chưa quét (thay thế 39.1% cũ, đo n=4: 39.1%, 46.0%, 54.4%, 61.3%)
        try:
            meta = probe_video(video_path)
            total_f = meta.get("nb_frames") or int(meta.get("duration", 1440) * 23.976)
            n_unique = int(total_f * 0.60)
        except Exception:
            n_unique = int(34000 * 0.60)

    total_overhead = video_overhead + subtitles_overhead
    eta_p5 = (n_unique / max(0.5, fps_p5)) + total_overhead
    decision_eta = eta_p5 * (1.0 + eta_margin_pct)
    eta_med = (n_unique / max(0.5, fps_median)) + total_overhead

    source_tag = "đo (plan.json)" if is_measured else "ước lượng thận trọng (n=4)"

    if decision_eta > seconds_left:
        if allow_partial:
            print(f"\n⏳ [ALLOW PARTIAL] Tập '{os.path.basename(video_path)}':")
            print(f"   • Cần: ~{eta_med/60:.1f}m (kỳ vọng trung vị) | Quyết định hạn chót: {decision_eta/60:.1f}m (P5 1.62 fps + 5m subs + 7% margin) cho {n_unique} unique frames [{source_tag}]")
            print(f"   • Thời gian còn lại trước {deadline_str}: {seconds_left/60:.1f} phút ({seconds_left/3600:.2f}h)")
            print(f"   -> Chế độ --allow-partial đang BẬT: Sẽ chạy đến hạn chót {deadline_str}, dừng sạch ở ranh giới chunk và giữ segment!\n")
            return True, True, deadline_dt
        else:
            print(f"\n⏰ [DEADLINE SKIP] Tập '{os.path.basename(video_path)}':")
            print(f"   • Dự toán cần: {decision_eta/60:.1f} phút (P5 1.62 fps + 5m subs + 7% margin) cho ~{n_unique} unique frames [{source_tag}]")
            print(f"   • Thời gian còn lại trước {deadline_str}: {seconds_left/60:.1f} phút ({seconds_left/3600:.2f}h)")
            print(f"   -> Bỏ qua tập này để không bị dở dang ca qua đêm! (Dùng --allow-partial nếu muốn chạy một phần)\n")
            return False, False, deadline_dt

    print(f"⏳ [DEADLINE OK] Tập '{os.path.basename(video_path)}' ({n_unique} unique frames [{source_tag}]): Cần ~{eta_med/60:.1f}m (Quyết định P5+7%: {decision_eta/60:.1f}m), còn {seconds_left/60:.1f}m trước hạn chót {deadline_str}.")
    return True, False, deadline_dt


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="🎬 AI Video Upscaler 4K - Tối ưu hóa cho Apple Silicon M3 Pro (PyTorch MPS / NCNN Metal + VideoToolbox 10-bit)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input",
        nargs="*",
        default=[],
        help="Đường dẫn đến một hoặc nhiều file video nguồn (.mkv, .mp4, .webm)"
    )
    parser.add_argument(
        "--queue",
        nargs="+",
        default=None,
        help="Danh sách nhiều tập video cần chạy tuần tự trong ca qua đêm"
    )
    parser.add_argument(
        "--deadline",
        default=None,
        metavar="HH:MM",
        help="Mốc hạn chót dừng hàng đợi (ví dụ: 07:00). Nếu tập tiếp theo dự kiến không kịp thì không bắt đầu."
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Nếu tập tiếp theo không kịp hạn chót --deadline, vẫn cho phép chạy đến hạn chót, dừng sạch ở ranh giới chunk, giữ các phân đoạn up_*.mp4 để tiếp tục lần sau."
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Chế độ benchmark thuần túy: Bộ lập lịch và preflight nền từ chối chạy để tránh gây nhiễu tài nguyên hệ thống."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Đường dẫn file hoặc thư mục xuất 4K (Mặc định: ~/Documents/Upscaled/<tên>_4K.mkv)"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"Mô hình AI upscale (Mặc định: tự động chọn qua shows.yaml hoặc mặc định animevideov3)"
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        choices=["mps", "ncnn"],
        help=f"Động cơ suy luận AI (Mặc định: {DEFAULT_ENGINE} - PyTorch MPS FP16)"
    )
    parser.add_argument(
        "--precision",
        default=DEFAULT_PRECISION,
        choices=["fp16", "fp32"],
        help=f"Độ chính xác suy luận (Mặc định: {DEFAULT_PRECISION})"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Liệt kê danh sách các mô hình Super-Resolution đã đăng ký trong models/models.yaml (D1, D2)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bắt buộc chạy và xóa phân đoạn cũ nếu phát hiện xung đột mô hình (D7)"
    )
    parser.add_argument(
        "-b", "--bitrate",
        default=DEFAULT_BITRATE,
        help=f"Bitrate video 4K HEVC 10-bit (Mặc định: {DEFAULT_BITRATE})"
    )
    parser.add_argument(
        "-s", "--segment",
        type=int,
        default=DEFAULT_SEGMENT_TIME,
        help=f"Độ dài phân đoạn cuốn chiếu tính bằng giây (Mặc định: {DEFAULT_SEGMENT_TIME}s - tối ưu cho RAM Disk)"
    )
    parser.add_argument(
        "-f", "--format",
        default=DEFAULT_FRAME_FORMAT,
        choices=["jpg", "png"],
        help=f"Định dạng khung hình trung gian (Mặc định: {DEFAULT_FRAME_FORMAT})"
    )
    parser.add_argument(
        "--no-ramdisk",
        action="store_true",
        help="Không sử dụng RAM Disk ảo mà dùng thư mục tạm trên ổ cứng SSD"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Không tiếp tục từ phân đoạn đã làm dở mà chạy lại từ đầu"
    )
    parser.add_argument(
        "--no-caffeinate",
        action="store_true",
        help="Không tự động kích hoạt caffeinate để chống sleep máy trong khi chạy"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Không sử dụng bộ lọc frame trùng (Deduplication) mà tính toán 100%% tất cả các khung hình"
    )
    parser.add_argument(
        "--preview",
        type=int,
        nargs="?",
        const=10,
        default=None,
        metavar="SECONDS",
        help="Chế độ xem trước: trích xuất nhanh N giây (mặc định: 10s) ở giữa phim, upscale và tự mở bằng IINA"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chế độ chạy thử lập lịch (Dry Run): Quét preflight hàng đợi, tính toán ETA từng tập và xuất bảng quyết định theo --deadline mà không chạy GPU."
    )
    parser.add_argument(
        "--subs-after-upscale",
        action="store_true",
        default=True,
        help="Tự động chạy dịch phụ đề (subtrans) sau khi toàn bộ hàng đợi upscale hoàn tất (Mặc định: Bật)"
    )
    parser.add_argument(
        "--no-subs", "--no-subs-after-upscale",
        dest="subs_after_upscale",
        action="store_false",
        help="Tắt bước dịch phụ đề sau upscale"
    )
    parser.add_argument(
        "--subs-only",
        nargs="?",
        const="__INPUT__",
        default=None,
        metavar="VIDEO",
        help="Chỉ chạy riêng bước dịch phụ đề (subtrans) cho tập video chỉ định trong tiến trình riêng (không chạy AI upscale)"
    )
    parser.add_argument(
        "--subs-upgrade",
        action="store_true",
        help="Dịch nâng cấp lại các câu thoại trước đó được dịch bằng Lite/Local bằng model chất lượng cao hơn khi còn hạn mức"
    )
    parser.add_argument(
        "--sub-offset-ms",
        type=int,
        default=0,
        help="Độ lệch thời gian dịch chuyển phụ đề tiếng Nhật / tiếng Việt (mili-giây, ví dụ: -11154 cho bản Crunchyroll lệch intro)"
    )
    parser.add_argument(
        "--sub-offset-target",
        type=str,
        default="",
        help="Tên file hoặc chuỗi nhận diện tập cụ thể áp dụng cờ --sub-offset-ms trong hàng đợi"
    )
    parser.add_argument(
        "--auto-offset",
        action="store_true",
        default=None,
        help="Tự động phát hiện và áp dụng bù lệch thời gian JA-EN cho phụ đề (Mặc định: Bật trong hàng đợi, Tắt khi chạy 1 tập đơn lẻ)"
    )
    parser.add_argument(
        "--no-auto-offset",
        action="store_true",
        default=False,
        help="Tắt cơ chế tự động bù lệch thời gian phụ đề trong hàng đợi"
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        default=False,
        help="Không nhúng track phụ đề vào container MKV 4K, chỉ xuất và giữ file .vi.srt ngoài"
    )
    parser.add_argument(
        "--night-report",
        nargs="?",
        const="night1",
        default=None,
        metavar="SESSION",
        help="Sinh báo cáo tổng kết phiên chạy đêm từ log, telemetry CSV, audit.json và api_requests (Chỉ đọc, không gọi GPU/API)"
    )
    parser.add_argument(
        "--sound-cues",
        choices=["drop", "translate"],
        default="drop",
        help="Xử lý chú thích âm thanh thuần ngoặc đơn: 'drop' (bỏ qua, mặc định) hoặc 'translate' (dịch)"
    )
    parser.add_argument(
        "--subs-quality",
        choices=["normal", "high"],
        default="normal",
        help="Chất lượng dịch phụ đề: 'normal' (Lite trước, tối ưu quota) hoặc 'high' (Flash trước)"
    )
    parser.add_argument(
        "--subs-pending",
        action="store_true",
        help="Chạy lại quy trình dịch phụ đề cho các tập ghi nhận trong pending_subs.json"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        default=False,
        help="Giữ lại thư mục tạm và checkpoint (.temp_<basename>) sau khi upscale thành công (Mặc định: Tự động dọn dẹp sạch sẽ để tiết kiệm dung lượng SSD)"
    )
    parser.add_argument(
        "--clean-temp", "--cleanup",
        action="store_true",
        default=False,
        help="Quét và dọn dẹp sạch sẽ toàn bộ thư mục tạm (.temp_*), symlinks mồ côi và gom báo cáo audit trong thư mục xuất xưởng"
    )
    parser.add_argument(
        "--no-organize-audit",
        action="store_true",
        default=False,
        help="Không tự động gom các file báo cáo kiểm định (.audit.json, .audit.md, .cast.yaml) vào thư mục con audit/"
    )

    return parser
 
 
def parse_args(args=None):
    return build_arg_parser().parse_args(args)


def find_or_extract_subtitles(video_path: str) -> tuple[Optional[str], Optional[str]]:
    """Tìm hoặc trích xuất phụ đề tiếng Nhật (ja) và tiếng Anh (en) cho video."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    vid_dir = os.path.dirname(os.path.abspath(video_path))

    ja_sub = None
    en_sub = None

    # 1. Tìm các file phụ đề sẵn có cùng thư mục
    candidates_ja = [
        os.path.join(vid_dir, f"{base_name}.ja.ass"),
        os.path.join(vid_dir, f"{base_name}_ja.ass"),
        os.path.join(vid_dir, f"{base_name}_jpn.ass"),
        os.path.join(vid_dir, f"{base_name}.ass"),
        os.path.join(vid_dir, f"{base_name}.ja.srt"),
        os.path.join(vid_dir, f"{base_name}.ja[cc].srt"),
        os.path.join(vid_dir, f"{base_name}_ja.srt"),
        os.path.join(vid_dir, f"{base_name}_jpn.srt"),
        os.path.join(vid_dir, f"{base_name}.srt"),
        f"subtrans/samples/{base_name}_jpn.ass",
        f"subtrans/samples/{base_name}.ass",
        f"subtrans/samples/{base_name}.ja[cc].srt",
    ]
    for c in candidates_ja:
        if os.path.exists(c):
            ja_sub = c
            break

    candidates_en = [
        os.path.join(vid_dir, f"{base_name}.en.ass"),
        os.path.join(vid_dir, f"{base_name}_en.ass"),
        os.path.join(vid_dir, f"{base_name}_eng.ass"),
        os.path.join(vid_dir, f"{base_name}.en.srt"),
        os.path.join(vid_dir, f"{base_name}_en.srt"),
        os.path.join(vid_dir, f"{base_name}_eng.srt"),
        f"subtrans/samples/{base_name}_eng.ass",
        f"subtrans/samples/{base_name}.en.ass",
    ]
    for c in candidates_en:
        if os.path.exists(c):
            en_sub = c
            break

    # 2. Kiểm tra container video qua ffprobe để trích xuất nếu thiếu file rời
    tmp_dir = os.path.join(vid_dir, f".temp_{base_name}")
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index:stream_tags=language,title",
            "-select_streams", "s",
            "-of", "json",
            video_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        sub_info = json.loads(r.stdout).get("streams", [])

        if sub_info:
            os.makedirs(tmp_dir, exist_ok=True)

            # 2a. Trích xuất stream EN từ container video nếu chưa có file EN rời
            if not en_sub:
                en_idx = None
                for s in sub_info:
                    lang = s.get("tags", {}).get("language", "").lower()
                    title = s.get("tags", {}).get("title", "").lower()
                    if lang in ("eng", "en") or "english" in title or "eng" in title:
                        en_idx = s["index"]
                        break
                # Nếu không gắn cờ ngôn ngữ rõ ràng nhưng có stream subtitle, lấy stream đầu tiên làm EN tham chiếu
                if en_idx is None and len(sub_info) > 0:
                    en_idx = sub_info[0]["index"]

                if en_idx is not None:
                    extracted_en = os.path.join(tmp_dir, "extracted_en.ass")
                    ext_cmd = ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{en_idx}", "-c:s", "copy", extracted_en]
                    sub_res = subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if sub_res.returncode == 0 and os.path.exists(extracted_en) and os.path.getsize(extracted_en) > 100:
                        en_sub = extracted_en

            # 2b. Trích xuất stream JA từ container video nếu chưa có file JA rời
            if not ja_sub:
                ja_idx = None
                for s in sub_info:
                    lang = s.get("tags", {}).get("language", "").lower()
                    title = s.get("tags", {}).get("title", "").lower()
                    if lang in ("jpn", "ja") or "japanese" in title or "jpn" in title:
                        ja_idx = s["index"]
                        break
                if ja_idx is not None:
                    extracted_ja = os.path.join(tmp_dir, "extracted_ja.ass")
                    ext_cmd = ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{ja_idx}", "-c:s", "copy", extracted_ja]
                    sub_res = subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if sub_res.returncode == 0 and os.path.exists(extracted_ja) and os.path.getsize(extracted_ja) > 100:
                        ja_sub = extracted_ja
    except Exception:
        pass

    return ja_sub, en_sub


def main():
    args = parse_args()

    # Xử lý cờ --clean-temp / --cleanup (Dọn dẹp nhanh thư mục tạm và gom gọn audit)
    if getattr(args, "clean_temp", False):
        from src.config import DEFAULT_OUTPUT_DIR
        from src.logger import format_bytes
        target_clean_dir = args.output if (args.output and os.path.isdir(args.output)) else DEFAULT_OUTPUT_DIR
        print(f"\n🧹 Bắt đầu quét và dọn dẹp thư mục: {target_clean_dir}...")
        report = clean_all_temporary_data(target_clean_dir)
        print(f"✅ Đã dọn dẹp {len(report['cleaned_dirs'])} thư mục/symlink tạm:")
        for cd in report['cleaned_dirs']:
            print(f"   • {cd}")
        print(f"📁 Đã gom {len(report['organized_files'])} file báo cáo kỹ thuật vào 'audit/':")
        for of in report['organized_files']:
            print(f"   • {of}")
        print(f"🛡️  Tổng dung lượng SSD đã giải phóng: {format_bytes(report['bytes_freed'])}\n")
        sys.exit(0)

    # Xử lý cờ --night-report (Sinh báo cáo tổng kết phiên đêm, chỉ đọc)
    if args.night_report:
        from src.night_report import generate_night_report_text
        report = generate_night_report_text(args.night_report)
        print(report)
        sys.exit(0)

    # Xử lý cờ --subs-pending (Chạy lại phụ đề cho các tập dở dang trong pending_subs.json)
    if args.subs_pending:
        pending_items = load_pending_subs()
        if not pending_items:
            print("ℹ️ Không có tập nào trong danh sách phụ đề dở dang (pending_subs.json).")
            sys.exit(0)
        print(f"\n📋 [SUBS PENDING] Tìm thấy {len(pending_items)} tập cần xử lý lại phụ đề:")
        for it in pending_items:
            vid_p = it.get("video_path", "")
            print(f"  • {os.path.basename(vid_p)} (Tỷ lệ thật: {it.get('real_rate', 0)*100:.1f}%, uncertain: {it.get('uncertain_rate', 0)*100:.1f}%)")

        from subtrans.pipeline import TranslationPipeline
        for it in pending_items:
            vid_p = it.get("video_path")
            if not vid_p or not os.path.exists(vid_p):
                continue
            ja_s, en_s = find_or_extract_subtitles(vid_p)
            if not ja_s:
                continue
            base_vid = os.path.splitext(os.path.basename(vid_p))[0]
            out_d = Path(args.output) if args.output and os.path.isdir(args.output) else Path(os.path.dirname(os.path.abspath(vid_p)))
            sub_pipe = TranslationPipeline(
                local_only=False,
                output_dir=out_d,
                output_format="srt",
                require_approved_cast=False,
                allow_latest_alias=False,
                subs_quality=getattr(args, "subs_quality", "normal"),
            )
            res = sub_pipe.run(
                ja_path=Path(ja_s),
                en_path=Path(en_s) if en_s else None,
                output_basename=f"{base_vid}_4K",
                subs_upgrade=True,
                sub_offset_ms=args.sub_offset_ms,
                auto_offset=True,
                sound_cues=args.sound_cues,
            )
            srt_p = res.get('subtitle_path') or res.get('output_path')
            if srt_p and os.path.exists(srt_p) and not str(srt_p).endswith(".partial.srt"):
                remove_pending_sub(vid_p)
                target_4k = os.path.join(out_d, f"{base_vid}_4K.mkv")
                if os.path.exists(target_4k) and not getattr(args, "no_embed", False):
                    from src.ffmpeg_ops import embed_subtitle_track
                    embed_subtitle_track(target_4k, str(srt_p), original_source_path=vid_p)
        sys.exit(0)

    # Xử lý cờ --subs-only (Chạy riêng phụ đề trong tiến trình con độc lập)
    if args.subs_only:
        target_subs_vid = args.input if args.subs_only == "__INPUT__" else args.subs_only
        if not target_subs_vid:
            print("❌ Lỗi: Cần cung cấp đường dẫn video khi dùng cờ --subs-only (ví dụ: ./upscale.py --subs-only <tập.mkv>)")
            sys.exit(1)

        print(f"\n{'='*75}")
        print(f"📝 [SUBS-ONLY] BẮT ĐẦU DỊCH PHỤ ĐỀ ĐỘC LẬP CHO: {os.path.basename(target_subs_vid)}")
        print(f"{'='*75}")
        ja_sub, en_sub = find_or_extract_subtitles(target_subs_vid)
        if not ja_sub:
            print(f"❌ Không tìm thấy hoặc không thể trích xuất phụ đề tiếng Nhật cho '{os.path.basename(target_subs_vid)}'.")
            sys.exit(1)

        print(f"   • File gốc JA: {ja_sub}")
        if en_sub:
            print(f"   • File tham chiếu EN: {en_sub}")

        from subtrans.pipeline import TranslationPipeline
        vid_dir = os.path.dirname(os.path.abspath(target_subs_vid))
        sub_out_dir = Path(args.output) if args.output and os.path.isdir(args.output) else Path(vid_dir)
        sub_pipeline = TranslationPipeline(
            local_only=False,
            output_dir=sub_out_dir,
            output_format="srt",
            require_approved_cast=False,
            allow_latest_alias=False,
            subs_quality=getattr(args, "subs_quality", "normal"),
        )
        base_vid_name = os.path.splitext(os.path.basename(target_subs_vid))[0]
        use_auto_offset = bool(args.auto_offset) if getattr(args, "auto_offset", None) is not None else False
        res = sub_pipeline.run(
            ja_path=Path(ja_sub),
            en_path=Path(en_sub) if en_sub else None,
            output_basename=f"{base_vid_name}_4K",
            subs_upgrade=args.subs_upgrade,
            sub_offset_ms=args.sub_offset_ms,
            auto_offset=use_auto_offset,
            sound_cues=args.sound_cues,
        )
        srt_path = res.get('subtitle_path') or res.get('output_path')
        print(f"\n✅ Hoàn tất dịch phụ đề độc lập cho '{os.path.basename(target_subs_vid)}'. File: {srt_path}")

        # Đồng bộ file SRT và nhúng trực tiếp vào container MKV 4K nếu đã có file 4K
        if srt_path and os.path.exists(srt_path):
            is_partial_sub = str(srt_path).endswith(".partial.srt")
            if is_partial_sub:
                print(f"⚠️ [CHƯA ĐỦ ĐIỀU KIỆN NHÚNG] File phụ đề là bản một phần ({os.path.basename(str(srt_path))}). Không nhúng vào MKV 4K.")
            else:
                from src.ffmpeg_ops import embed_subtitle_track
                target_4k = os.path.join(sub_out_dir, f"{base_vid_name}_4K.mkv")
                if os.path.exists(target_4k):
                    if not getattr(args, "no_embed", False):
                        print(f"🎬 Đang nhúng track phụ đề tiếng Việt vào container MKV: {os.path.basename(target_4k)}...")
                        ok = embed_subtitle_track(target_4k, str(srt_path), original_source_path=target_subs_vid)
                        if ok:
                            print("✅ Đã nhúng phụ đề tiếng Việt (mặc định) trực tiếp vào container 4K MKV!")
                        else:
                            print("⚠️ Không thể nhúng phụ đề vào container 4K (audit không đạt hoặc lỗi), file phụ đề ngoài vẫn sẵn sàng.")
                    else:
                        print(f"ℹ️ Cờ --no-embed được bật: Giữ nguyên container 4K, chỉ dùng file phụ đề ngoài '{os.path.basename(str(srt_path))}'.")

        # Dọn dẹp tệp phụ đề tạm và gom gọn báo cáo audit
        cleanup_subtitle_extracts(vid_dir, base_vid_name)
        if not getattr(args, "no_organize_audit", False):
            organize_output_artifacts(str(sub_out_dir), base_name=base_vid_name)

        sys.exit(0)

    # Kiểm tra cờ liệt kê mô hình (--list-models)
    if args.list_models:
        print(list_models_formatted())
        sys.exit(0)

    # Xác định danh sách video cần xử lý
    videos = []
    if args.queue:
        videos.extend(args.queue)
    if isinstance(args.input, list):
        videos.extend(args.input)
    elif args.input:
        videos.append(args.input)

    if not videos:
        print("❌ Lỗi: Vui lòng cung cấp ít nhất một file video đầu vào hoặc dùng cờ --queue.")
        sys.exit(1)

    fps_median = 1.61 if args.engine == "mps" else 1.35
    fps_p5 = 1.62 if args.engine == "mps" else 1.25
    video_overhead = 930.0  # Hiệu chỉnh theo 2 tập đêm 1 [691s, 1166s], cỡ mẫu n=2
    subtitles_overhead = 300.0

    if args.benchmark_mode:
        os.environ["BENCHMARK_MODE"] = "1"
        print("\n⚡ [BENCHMARK MODE] Đã kích hoạt chế độ benchmark thuần túy: Bộ lập lịch và preflight nền từ chối chạy để bảo toàn môi trường đo đạc!")

    # 1. Chế độ Dry Run (--dry-run): Preflight toàn bộ hàng đợi và in bảng quyết định, không chạy GPU
    if args.dry_run:
        from src.scheduler import compute_overnight_schedule, format_schedule_decision_table
        print(f"\n🔍 [DRY RUN] BẮT ĐẦU KIỂM TRA LẬP LỊCH CHO {len(videos)} TẬP TRONG HÀNG ĐỢI...")
        sched = compute_overnight_schedule(
            queue=videos,
            deadline_str=args.deadline,
            fps_median=fps_median,
            fps_p5=fps_p5,
            video_overhead_seconds=video_overhead,
            subtitles_overhead_seconds=subtitles_overhead,
            allow_partial=args.allow_partial,
            output_dir=args.output,
            segment_seconds=args.segment,
            preflight_now=False
        )
        print(format_schedule_decision_table(sched))
        sys.exit(0)

    # 2. Phiên chạy thật: Lập lịch và in bảng quyết định ngay đầu phiên (nếu không ở benchmark mode)
    if not args.benchmark_mode:
        from src.scheduler import compute_overnight_schedule, format_schedule_decision_table, BackgroundQueuePreflightManager
        sched = compute_overnight_schedule(
            queue=videos,
            deadline_str=args.deadline,
            fps_median=fps_median,
            fps_p5=fps_p5,
            video_overhead_seconds=video_overhead,
            subtitles_overhead_seconds=subtitles_overhead,
            allow_partial=args.allow_partial,
            output_dir=args.output,
            segment_seconds=args.segment,
            preflight_now=False
        )
        print(format_schedule_decision_table(sched))

        # Kích hoạt preflight nền cho toàn bộ các tập tiếp theo trong hàng đợi (nice -n 10)
        bg_preflight = None
        if len(videos) > 1 and not args.no_dedup:
            def _on_bg_plan_ready(scanned_vid, pdata):
                try:
                    updated_sched = compute_overnight_schedule(
                        queue=videos,
                        deadline_str=args.deadline,
                        fps_median=fps_median,
                        fps_p5=fps_p5,
                        video_overhead_seconds=video_overhead,
                        subtitles_overhead_seconds=subtitles_overhead,
                        allow_partial=args.allow_partial,
                        output_dir=args.output,
                        segment_seconds=args.segment,
                        preflight_now=False
                    )
                    print(f"\n🔄 [BẢNG LẬP LỊCH CẬP NHẬT] Đã nạp số đo n_unique thật cho '{os.path.basename(scanned_vid)}':")
                    print(format_schedule_decision_table(updated_sched))
                except Exception as e_bg:
                    pass

            bg_preflight = BackgroundQueuePreflightManager(
                queue=videos,
                output_dir=args.output,
                segment_seconds=args.segment,
                skip_first=True,
                benchmark_mode=args.benchmark_mode,
                on_plan_ready=_on_bg_plan_ready
            )
            bg_preflight.start()

    completed_videos = []
    failed_resumable_videos = {}  # vid -> (up_segs_dir, chunk_idx)
    unfinished_videos = {}        # vid -> reason

    for idx, vid in enumerate(videos):
        print(f"\n{'='*75}")
        print(f"▶ [{idx+1}/{len(videos)}] Bắt đầu xử lý: {os.path.basename(vid)}")
        print(f"{'='*75}")

        # Kiểm tra deadline nếu có (A6)
        deadline_dt = None
        if args.deadline:
            can_finish, is_partial, raw_deadline_dt = check_deadline_and_eta(
                vid,
                args.deadline,
                fps_median=fps_median,
                fps_p5=fps_p5,
                video_overhead=video_overhead,
                subtitles_overhead=subtitles_overhead,
                eta_margin_pct=0.07,
                allow_partial=args.allow_partial
            )
            if not can_finish:
                print(f"⏹️ Dừng hàng đợi tại tập {idx+1}/{len(videos)} do chạm ngưỡng hạn chót {args.deadline}.")
                unfinished_videos[vid] = f"Bỏ qua do chạm hạn chót {args.deadline}"
                break

            # (a) Dành thời gian cho giai đoạn phụ đề của các tập đã xong:
            # Trừ đi thời gian phụ đề cần thiết cho các tập đã hoàn tất trước đó trong queue
            subs_reserved_sec = (len(completed_videos) * subtitles_overhead) if args.subs_after_upscale else 0.0
            deadline_dt = (raw_deadline_dt - timedelta(seconds=subs_reserved_sec)) if raw_deadline_dt else None

        try:
            out_file = run_upscale_pipeline(
                input_path=vid,
                output_path=args.output,
                model=args.model,
                engine=args.engine,
                precision=args.precision,
                force=args.force,
                bitrate=args.bitrate,
                segment_time=args.segment,
                fmt=args.format,
                use_ramdisk=not args.no_ramdisk,
                resume=not args.no_resume,
                caffeinate=not args.no_caffeinate,
                preview_seconds=args.preview,
                dedup=not args.no_dedup,
                allow_partial=args.allow_partial,
                deadline_dt=deadline_dt,
                keep_temp=args.keep_temp
            )
            # (b) Tập dang dở (do allow-partial) hoặc không hợp lệ bị loại khỏi mux/phụ đề:
            if out_file and os.path.isfile(out_file) and out_file.endswith((".mkv", ".mp4")):
                completed_videos.append((vid, out_file))
                if vid in unfinished_videos:
                    del unfinished_videos[vid]
                if vid in failed_resumable_videos:
                    del failed_resumable_videos[vid]
            else:
                unfinished_videos[vid] = "Dừng sạch ranh giới phân đoạn theo --allow-partial"
                print(f"\n⏸️ [PARTIAL] Tập '{os.path.basename(vid)}' chưa hoàn tất 4K, tự động loại khỏi danh sách dịch phụ đề (Stage B) và không mux thành phẩm.")

            # Cập nhật fps từ audit.json nếu có
            audit_path = os.path.splitext(out_file)[0] + ".audit.json" if out_file else None
            if audit_path and not os.path.exists(audit_path):
                alt_audit = os.path.join(os.path.dirname(out_file), "audit", os.path.basename(audit_path))
                if os.path.exists(alt_audit):
                    audit_path = alt_audit
            if audit_path and os.path.exists(audit_path):
                try:
                    with open(audit_path, "r", encoding="utf-8") as f:
                        adata = json.load(f)
                    measured_fps = adata.get("ai_steady_fps")
                    if measured_fps and measured_fps > 0:
                        fps_median = float(measured_fps)
                except Exception:
                    pass

        except FailedResumableError as fre:
            print(f"\n⚠️ [FAILED-RESUMABLE] Tập '{os.path.basename(vid)}' dừng do ENOSPC tại chunk {fre.chunk_idx + 1}. Đánh dấu để thử lại cuối hàng đợi.")
            failed_resumable_videos[vid] = (fre.up_segs_dir, fre.chunk_idx)
            unfinished_videos[vid] = f"Dừng ở chunk {fre.chunk_idx + 1} do ENOSPC (failed-resumable)"
            if len(videos) > 1:
                print("Chuyển sang tập tiếp theo trong hàng đợi...\n")
                continue
            else:
                pass
        except KeyboardInterrupt:
            print("\n\n⚠️ Đã tạm dừng quá trình theo yêu cầu (Ctrl+C).")
            print("🛡️  RAM Disk và các file đệm đã được tự động giải phóng an toàn.")
            print("💡 Lần sau khi chạy lại cùng video này, hệ thống sẽ tự động tiếp tục từ phân đoạn đang dở!\n")
            sys.exit(130)
        except Exception as e:
            print(f"\n❌ Đã xảy ra lỗi khi xử lý tập '{vid}': {e}\n")
            unfinished_videos[vid] = f"Lỗi: {e}"
            if len(videos) > 1:
                print("Chuyển sang tập tiếp theo trong hàng đợi...\n")
                continue
            else:
                sys.exit(1)

    # -------------------------------------------------------------------------
    # TỰ ĐỘNG THỬ LẠI CÁC TẬP FAILED-RESUMABLE MỘT LẦN Ở CUỐI HÀNG ĐỢI
    # -------------------------------------------------------------------------
    if failed_resumable_videos:
        print(f"\n{'='*75}")
        print(f"🔄 [RETRY PASS] TỰ ĐỘNG THỬ LẠI {len(failed_resumable_videos)} TẬP FAILED-RESUMABLE MỘT LẦN")
        print(f"{'='*75}")
        for vid, (up_dir, last_c_idx) in list(failed_resumable_videos.items()):
            print(f"▶ Thử lại: {os.path.basename(vid)} (tiếp tục từ phân đoạn {last_c_idx + 1})...")
            try:
                out_file = run_upscale_pipeline(
                    input_path=vid,
                    output_path=args.output,
                    model=args.model,
                    engine=args.engine,
                    precision=args.precision,
                    force=args.force,
                    bitrate=args.bitrate,
                    segment_time=args.segment,
                    fmt=args.format,
                    use_ramdisk=not args.no_ramdisk,
                    resume=not args.no_resume,
                    caffeinate=not args.no_caffeinate,
                    preview_seconds=args.preview,
                    dedup=not args.no_dedup,
                    allow_partial=args.allow_partial,
                    deadline_dt=None,
                    keep_temp=args.keep_temp
                )
                if out_file and os.path.isfile(out_file) and out_file.endswith((".mkv", ".mp4")):
                    completed_videos.append((vid, out_file))
                    del failed_resumable_videos[vid]
                    if vid in unfinished_videos:
                        del unfinished_videos[vid]
                    print(f"✅ Thử lại thành công: '{os.path.basename(vid)}' đã hoàn tất 4K!")
            except Exception as retry_err:
                print(f"❌ Thử lại vẫn thất bại cho '{os.path.basename(vid)}': {retry_err}")
                unfinished_videos[vid] = f"Thử lại thất bại (chunk {last_c_idx + 1}): {retry_err}"

    # -------------------------------------------------------------------------
    # STAGE B: DỊCH PHỤ ĐỀ (CHỈ CHẠY CHO CÁC TẬP ĐÃ HOÀN TẤT)
    # CHẠY TRONG TIẾN TRÌNH CON ĐỘC LẬP (SUBPROCESS), KHÔNG CHẶN HAY HỎNG VIDEO 4K
    # -------------------------------------------------------------------------
    if args.subs_after_upscale and completed_videos:
        print(f"\n{'='*75}")
        print("📝 [STAGE B] BẮT ĐẦU DỊCH PHỤ ĐỀ TRONG TIẾN TRÌNH CON ĐỘC LẬP (SAU KHI CẢ HÀNG ĐỢI XONG)")
        print(f"{'='*75}")

        sub_timeout = 1560  # Watchdog 25 phút + 60s buffer
        queue_auto_offset = not getattr(args, "no_auto_offset", False)

        for src_vid, out_vid in completed_videos:
            sub_out_dir = os.path.dirname(os.path.abspath(out_vid))
            sub_cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--subs-only",
                src_vid,
                "-o", sub_out_dir,
                "--sound-cues", args.sound_cues,
                "--subs-quality", args.subs_quality,
            ]
            if args.subs_upgrade:
                sub_cmd.append("--subs-upgrade")
            if getattr(args, "no_embed", False):
                sub_cmd.append("--no-embed")

            # Xử lý cờ --sub-offset-ms thủ công trong hàng đợi / tập đơn lẻ:
            manual_offset_applied = False
            if args.sub_offset_ms:
                if len(videos) == 1 or (args.sub_offset_target and args.sub_offset_target.lower() in os.path.basename(src_vid).lower()):
                    manual_offset_applied = True
                    print(f"  • [OFFSET] Áp dụng cờ thủ công --sub-offset-ms={args.sub_offset_ms}ms cho: '{os.path.basename(src_vid)}'")
                    sub_cmd.extend(["--sub-offset-ms", str(args.sub_offset_ms)])
                elif not args.sub_offset_target:
                    print(f"  ⚠️ [QUEUE OFFSET WARNING] Cờ thủ công --sub-offset-ms={args.sub_offset_ms}ms được truyền trong hàng đợi nhiều tập nhưng KHÔNG có --sub-offset-target. Để bảo toàn đồng bộ của các tập khác, offset thủ công KHÔNG áp dụng toàn cục cho '{os.path.basename(src_vid)}' (dùng --auto-offset tính riêng).")

            # Nếu không áp dụng offset thủ công cho tập này, dùng auto_offset nếu được bật
            if not manual_offset_applied and queue_auto_offset:
                sub_cmd.append("--auto-offset")
            print(f"\n🚀 Khởi chạy tiến trình con dịch phụ đề cho tập '{os.path.basename(src_vid)}'...")
            try:
                proc = subprocess.run(sub_cmd, timeout=sub_timeout, check=False)
                if proc.returncode == 0:
                    print(f"✅ Tiến trình con phụ đề hoàn tất thành công cho '{os.path.basename(src_vid)}'.")
                else:
                    print(f"⚠️ Tiến trình con phụ đề thoát với mã lỗi {proc.returncode}. Video 4K và audit.json vẫn được bảo toàn nguyên vẹn!")
            except subprocess.TimeoutExpired:
                print(f"⏱️ Tiến trình con phụ đề vượt quá thời gian tối đa ({sub_timeout}s) và đã bị ngắt an toàn. Video 4K không bị ảnh hưởng!")
            except Exception as se:
                print(f"⚠️ Lỗi khởi chạy tiến trình con phụ đề: {se}")

    # -------------------------------------------------------------------------
    # DỌN DẸP PHỤ ĐỀ TẠM & GOM GỌN CÁC FILE KỸ THUẬT (AUDIT/CAST) SAU KHI HOÀN TẤT
    # -------------------------------------------------------------------------
    if not getattr(args, "no_organize_audit", False) and completed_videos:
        for src_vid, out_vid in completed_videos:
            sub_out_dir = os.path.dirname(os.path.abspath(out_vid))
            base_vid_name = os.path.splitext(os.path.basename(src_vid))[0]
            cleanup_subtitle_extracts(os.path.dirname(os.path.abspath(src_vid)), base_vid_name)
            organize_output_artifacts(sub_out_dir, base_name=base_vid_name)

    # -------------------------------------------------------------------------
    # KIỂM ĐỊNH TẤT CẢ CÁC TẬP: NẾU CÒN TẬP CHƯA XONG THÌ THOÁT MÃ ≠ 0 VÀ IN BẢNG
    # -------------------------------------------------------------------------
    unfinished_list = [v for v in videos if v not in [cv[0] for cv in completed_videos]]
    if unfinished_list:
        print(f"\n{'='*75}")
        print("🚨 BẢNG TỔNG KẾT HÀNG ĐỢI: CÁC TẬP CHƯA XONG")
        print(f"{'='*75}")
        print(f"{'STT':<4} | {'Tập Video':<45} | {'Trạng Thái / Ghi Chú'}")
        print("-" * 75)
        for u_idx, u_vid in enumerate(unfinished_list):
            reason = unfinished_videos.get(u_vid, "Chưa hoàn tất")
            print(f"[{u_idx+1:02d}] | {os.path.basename(u_vid)[:45]:<45} | {reason}")
        print(f"{'='*75}")
        print(f"⚠️ Có {len(unfinished_list)}/{len(videos)} tập chưa hoàn tất. Hàng đợi kết thúc với trạng thái CHƯA XONG.\n")
        sys.exit(1)

    print("\n🎉 Toàn bộ công việc trong hàng đợi đã hoàn tất thành công!")
    sys.exit(0)


if __name__ == "__main__":
    main()

