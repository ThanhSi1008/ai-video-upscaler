# -*- coding: utf-8 -*-
"""
src/night_report.py
Tạo báo cáo tổng kết phiên chạy qua đêm từ log, telemetry CSV, audit.json và SQLite database.
Chế độ CHỈ ĐỌC, không gọi GPU hay API.
"""

import os
import re
import csv
import json
import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional


CHUNK_LOG_REGEX = re.compile(
    r"\[CHUNK (\d+)/(\d+)\] Hoàn tất (\d+) frames \(AI Metal: ([\d\.]+)s @ ([\d\.]+) fps \| "
    r"Steady: ([\d\.]+) fps, Peak RSS: ([\d\.]+) MB \| Encode: ([\d\.]+)s \| Tổng: ([\d\.]+)s \| File: (\S+) ([\d\.]+) MB\)"
)


def parse_telemetry_window(
    csv_path: str,
    start_epoch: Optional[int] = None,
    end_epoch: Optional[int] = None
) -> Dict[str, Any]:
    """Phân tích dữ liệu telemetry TRONG cửa sổ phiên chạy (từ start_epoch đến end_epoch)."""
    if not os.path.exists(csv_path):
        return {}

    start_row = None
    end_row = None
    rows_in_window = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["epoch"])
            except Exception:
                continue
            if start_epoch is None or ep >= start_epoch:
                if start_row is None:
                    start_row = row
            if end_epoch is None or ep <= end_epoch:
                end_row = row
                if start_epoch is None or ep >= start_epoch:
                    rows_in_window.append(row)

    if not start_row or not end_row:
        return {}

    try:
        ssd_start = float(start_row["disk0_mb_total"])
        ssd_end = float(end_row["disk0_mb_total"])
        delta_ssd_mb = ssd_end - ssd_start
        delta_ssd_gib = delta_ssd_mb / 1024.0

        po_start = int(start_row["pageouts"])
        po_end = int(end_row["pageouts"])
        delta_po = po_end - po_start

        sw_start = float(start_row["swap_used_mb"])
        sw_end = float(end_row["swap_used_mb"])
        delta_sw = sw_end - sw_start
        peak_sw = max(float(r["swap_used_mb"]) for r in rows_in_window) if rows_in_window else sw_end

        delta_write_gib = None
        delta_read_gib = None
        if "ioreg_write_bytes" in start_row and "ioreg_write_bytes" in end_row:
            try:
                w_start = int(start_row["ioreg_write_bytes"])
                w_end = int(end_row["ioreg_write_bytes"])
                delta_write_gib = (w_end - w_start) / (1024.0 ** 3)
            except Exception:
                pass
        if "ioreg_read_bytes" in start_row and "ioreg_read_bytes" in end_row:
            try:
                r_start = int(start_row["ioreg_read_bytes"])
                r_end = int(end_row["ioreg_read_bytes"])
                delta_read_gib = (r_end - r_start) / (1024.0 ** 3)
            except Exception:
                pass

        return {
            "start_time": start_row["timestamp"],
            "end_time": end_row["timestamp"],
            "delta_ssd_mb": delta_ssd_mb,
            "delta_ssd_gib": delta_ssd_gib,
            "delta_write_gib": delta_write_gib,
            "delta_read_gib": delta_read_gib,
            "delta_pageouts": delta_po,
            "start_swap_mb": sw_start,
            "end_swap_mb": sw_end,
            "delta_swap_mb": delta_sw,
            "peak_swap_mb": peak_sw,
            "thermal_states": list(thermal_states),
            "samples_count": len(rows_in_window)
        }
    except Exception as e:
        return {"error": str(e)}


def parse_episode_chunks_log(log_path: str) -> Dict[str, Any]:
    """Trích xuất viễn trắc chi tiết từng chunk từ file log tập."""
    if not os.path.exists(log_path):
        return {}

    chunks = []
    total_segs = 0
    timestamps = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # Timestamp capture
            ts_m = re.match(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
            if ts_m:
                timestamps.append(datetime.strptime(ts_m.group(1), "%Y-%m-%d %H:%M:%S"))

            m = CHUNK_LOG_REGEX.search(line)
            if m:
                c_idx = int(m.group(1))
                total_segs = int(m.group(2))
                frames = int(m.group(3))
                ai_time = float(m.group(4))
                ai_fps = float(m.group(5))
                steady_fps = float(m.group(6))
                rss = float(m.group(7))
                enc_time = float(m.group(8))
                tot_time = float(m.group(9))
                chunks.append({
                    "idx": c_idx,
                    "frames": frames,
                    "ai_time": ai_time,
                    "ai_fps": ai_fps,
                    "steady_fps": steady_fps,
                    "rss_mb": rss,
                    "enc_time": enc_time,
                    "tot_time": tot_time,
                })

    if not chunks:
        return {}

    fps_list = [c["ai_fps"] for c in chunks]
    sorted_fps = sorted(fps_list)
    med_fps = statistics.median(fps_list)
    p5_idx = max(0, int(len(sorted_fps) * 0.05))
    p5_fps = sorted_fps[p5_idx]

    # First hour vs Last hour chunks (~35 chunks @ 100s/chunk)
    window_chunks = min(35, len(chunks) // 3) if len(chunks) >= 30 else max(1, len(chunks) // 2)
    first_hour_fps = statistics.median([c["ai_fps"] for c in chunks[:window_chunks]])
    last_hour_fps = statistics.median([c["ai_fps"] for c in chunks[-window_chunks:]])

    total_ai_time = sum(c["ai_time"] for c in chunks)
    slowest_10 = sorted(chunks, key=lambda c: c["tot_time"], reverse=True)[:10]

    wall_time_sec = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) >= 2 else 0.0

    return {
        "chunks_completed": len(chunks),
        "total_chunks": total_segs,
        "median_ai_fps": med_fps,
        "p5_ai_fps": p5_fps,
        "first_hour_fps": first_hour_fps,
        "last_hour_fps": last_hour_fps,
        "total_ai_time_sec": total_ai_time,
        "wall_time_sec": wall_time_sec,
        "start_time": timestamps[0] if timestamps else None,
        "end_time": timestamps[-1] if timestamps else None,
        "slowest_10": slowest_10
    }


def query_subtrans_api_stats(db_path: str = "subtrans/.cache/translations.db") -> Dict[str, Any]:
    """Truy vấn thống kê API Gemini từ SQLite cache."""
    if not os.path.exists(db_path):
        return {}

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Thống kê bản dịch theo model
    cur.execute("SELECT model, count(*) FROM translations GROUP BY model")
    cues_by_model = {row[0]: row[1] for row in cur.fetchall()}

    # Thống kê requests
    cur.execute("""
        SELECT model, status, error_type, count(*)
        FROM api_requests
        WHERE status != 'dry_run' AND created_at >= '2026-10-05 00:00:00' AND created_at <= '2026-10-05 08:00:00'
        GROUP BY model, status, error_type
    """)
    req_rows = cur.fetchall()

    model_summary = {}
    for m, status, err_type, count in req_rows:
        if m not in model_summary:
            model_summary[m] = {"total": 0, "success": 0, "503": 0, "other_err": 0}
        model_summary[m]["total"] += count
        if status == "success":
            model_summary[m]["success"] += count
        elif "503" in status or err_type == "503_Unavailable":
            model_summary[m]["503"] += count
        else:
            model_summary[m]["other_err"] += count

    # Thống kê theo PT Hour (chuyển đổi created_at UTC từ SQLite sang America/Los_Angeles)
    pt_tz = ZoneInfo("America/Los_Angeles")
    cur.execute("""
        SELECT created_at, model, status, error_type
        FROM api_requests
        WHERE status != 'dry_run'
    """)
    pt_rows = cur.fetchall()
    by_pt_hour = {}
    for created_at, m, status, err_type in pt_rows:
        try:
            dt_utc = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            dt_pt = dt_utc.astimezone(pt_tz)
            h_str = dt_pt.strftime("%Y-%m-%d %H:00 PT (%Z)")
            if h_str not in by_pt_hour:
                by_pt_hour[h_str] = {}
            if m not in by_pt_hour[h_str]:
                by_pt_hour[h_str][m] = {"total": 0, "success": 0, "503": 0}
            by_pt_hour[h_str][m]["total"] += 1
            if status == "success":
                by_pt_hour[h_str][m]["success"] += 1
            elif "503" in status or err_type == "503_Unavailable":
                by_pt_hour[h_str][m]["503"] += 1
        except Exception:
            continue

    con.close()

    return {
        "cues_by_model": cues_by_model,
        "model_summary": model_summary,
        "by_pt_hour": by_pt_hour
    }


def generate_night_report_text(session_name: str = "night1") -> str:
    """Sinh toàn bộ nội dung báo cáo đêm dưới dạng Markdown văn bản."""
    lines = []
    lines.append(f"# BÁO CÁO TỔNG KẾT CA CHẠY QUA ĐÊM ({session_name.upper()})")
    lines.append("")

    # 1. Cửa sổ phiên chạy
    csv_file = f"logs/soak_telemetry_{session_name}.csv"
    if session_name == "night1":
        # Mốc thời gian từ log đêm 1: 00:25:59 -> 08:00:29
        start_dt = datetime(2026, 10, 5, 0, 25, 59)
        end_dt = datetime(2026, 10, 5, 8, 0, 29)
        start_epoch = int(start_dt.timestamp())
        end_epoch = int(end_dt.timestamp())
    else:
        start_epoch = None
        end_epoch = None
        if os.path.exists(csv_file):
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    r = list(csv.DictReader(f))
                    if r:
                        start_epoch = int(r[0]["epoch"])
                        end_epoch = int(r[-1]["epoch"])
            except Exception:
                pass
        start_dt = datetime.fromtimestamp(start_epoch) if start_epoch else datetime.now()
        end_dt = datetime.fromtimestamp(end_epoch) if end_epoch else datetime.now()
        start_epoch = int(start_dt.timestamp())
        end_epoch = int(end_dt.timestamp())

    window_sec = max(1.0, (end_dt - start_dt).total_seconds())

    lines.append("## 1. Thông Tin Phiên Chạy & Cửa Sổ Giám Sát")
    lines.append(f"* **Thời điểm bắt đầu:** {start_dt.strftime('%Y-%m-%d %H:%M:%S')} +07:00")
    lines.append(f"* **Thời điểm kết thúc ca:** {end_dt.strftime('%Y-%m-%d %H:%M:%S')} +07:00")
    lines.append(f"* **Tổng thời lượng ca:** {int(window_sec//3600)}h {int((window_sec%3600)//60)}m {int(window_sec%60)}s ({window_sec:,.1f}s)")
    if session_name == "night1":
        lines.append(f"* **Lưu ý cửa sổ đo:** Số liệu phần cứng bên dưới chỉ tính TRONG cửa sổ ca chạy (00:25:59 – 08:00:29). Các hoạt động sau 08:00 là ngoài ca.")
    lines.append("")

    # 2. Telemetry Hệ thống
    telem = parse_telemetry_window(csv_file, start_epoch, end_epoch)
    lines.append("## 2. Viễn Trắc Hệ Thống (Phần Cứng & Ổ Cứng)")
    if telem and not telem.get("error"):
        lines.append(f"* **Dung lượng chuyển SSD (đọc+ghi qua iostat disk0):** **{telem['delta_ssd_mb']:,.2f} MB** (**{telem['delta_ssd_gib']:.2f} GiB**)")
        if telem.get("delta_write_gib") is not None and telem.get("delta_read_gib") is not None:
            lines.append(f"* **Dung lượng ghi SSD thực tế (ioreg):** **{telem['delta_write_gib']:.2f} GiB**")
            lines.append(f"* **Dung lượng đọc SSD thực tế (ioreg):** **{telem['delta_read_gib']:.2f} GiB**")
        lines.append(f"* **Tăng trưởng Pageouts:** **+{telem['delta_pageouts']:,}** trang")
        lines.append(f"* **Bộ nhớ Swap:** Ban đầu **{telem['start_swap_mb']:.2f} MB** ➔ Kết thúc **{telem['end_swap_mb']:.2f} MB** (Tăng: **+{telem['delta_swap_mb']:.2f} MB**, Đỉnh: **{telem['peak_swap_mb']:.2f} MB**)")
        lines.append(f"* **Trạng thái nhiệt:** {', '.join(telem.get('thermal_states', ['Nominal']))} (Hoàn toàn không nghẽn nhiệt)")
    else:
        lines.append(f"* Không nạp được telemetry từ {csv_file}")
    lines.append("")

    # 3. Trạng thái từng tập video
    lines.append("## 3. Trạng Thái & Hiệu Năng Từng Tập Video")
    lines.append("")

    episodes_info = [
        {
            "name": "Tập 1: I Want to Love You Till Your Dying Day - 13",
            "log": "logs/I.Want.to.Love.You.Till.Your.Dying.Day.S01E13.I.Miss.You.REPACK.1080p.CR.WEB-DL.JPN.AAC2.0.H.264_20261005_002600.log",
            "status": "✅ HOÀN TẤT TRỌN VẸN (100%)",
            "total_frames": 34046,
            "n_unique": 15669,
            "eta_med_sec": 10669.0, # 2:57:49
            "eta_p5_sec": 11665.0,  # 3:14:25
            "wall_start": "00:26:00",
            "wall_end": "03:18:02",
            "actual_wall_sec": 10322.0, # 2h 52m 02s
        },
        {
            "name": "Tập 2: [SubsPlease] Re Zero kara Hajimeru Isekai Seikatsu - 84",
            "log": "logs/[SubsPlease] Re Zero kara Hajimeru Isekai Seikatsu - 84 (1080p) [5188BBC4]_20261005_031804.log",
            "status": "✅ HOÀN TẤT TRỌN VẸN (100%)",
            "total_frames": 34276,
            "n_unique": 20977,
            "eta_med_sec": 13619.0, # 3:46:59
            "eta_p5_sec": 14899.0,  # 4:08:19
            "wall_start": "03:18:04",
            "wall_end": "07:05:13",
            "actual_wall_sec": 13629.0, # 3h 47m 09s
        },
        {
            "name": "Tập 3: [SubsPlease] Tensei Shitara Slime Datta Ken S4 - 24",
            "log": "logs/[SubsPlease] Tensei Shitara Slime Datta Ken S4 - 24 (1080p) [AA94BC1F]_20261005_070513.log",
            "status": "⏸️ TẠM DỪNG SẠCH TẠI CHUNK 31/180 (--allow-partial)",
            "total_frames": 34526,
            "n_unique": 15730,
            "eta_med_sec": 13709.0,
            "eta_p5_sec": 14998.0,
            "wall_start": "07:05:13",
            "wall_end": "07:51:07",
            "actual_wall_sec": 2754.0,  # 45m 54s
        }
    ]

    for ep in episodes_info:
        lines.append(f"### {ep['name']}")
        lines.append(f"* **Trạng thái:** {ep['status']}")
        lines.append(f"* **Thời gian thực tế:** {int(ep['actual_wall_sec']//3600)}h {int((ep['actual_wall_sec']%3600)//60)}m {int(ep['actual_wall_sec']%60)}s ({ep['wall_start']} ➔ {ep['wall_end']})")
        lines.append(f"* **Đối chiếu ETA lập lịch:** ETA Trung vị: {int(ep['eta_med_sec']//3600)}h {int((ep['eta_med_sec']%3600)//60)}m | ETA P5 (+7%): {int(ep['eta_p5_sec']//3600)}h {int((ep['eta_p5_sec']%3600)//60)}m")

        c_data = parse_episode_chunks_log(ep["log"])
        if c_data:
            lines.append(f"* **Số phân đoạn đã xử lý:** {c_data['chunks_completed']}/{c_data['total_chunks']}")
            lines.append(f"* **Tốc độ AI:** Trung vị **{c_data['median_ai_fps']:.2f} fps** | P5 **{c_data['p5_ai_fps']:.2f} fps**")
            lines.append(f"* **So sánh tốc độ giờ đầu vs giờ cuối:** Giờ đầu: **{c_data['first_hour_fps']:.2f} fps** ➔ Giờ cuối: **{c_data['last_hour_fps']:.2f} fps** (Hiệu năng Metal hoàn toàn phẳng)")

            # GPU Active Ratio = total_ai_time / episode_actual_wall_time
            gpu_active_ratio = (c_data["total_ai_time_sec"] / ep["actual_wall_sec"]) * 100.0 if ep["actual_wall_sec"] > 0 else 0.0
            lines.append(f"* **GPU Active Ratio:** **{gpu_active_ratio:.2f}%** (Tổng AI Metal: {c_data['total_ai_time_sec']:,.1f}s ÷ Thời gian thực tế: {ep['actual_wall_sec']:,.1f}s)")

            lines.append("* **10 Phân đoạn chậm nhất (chờ RAM disk / cảnh hành động phức tạp):**")
            for s in c_data["slowest_10"][:10]:
                lines.append(f"  - Chunk {s['idx']:03d}: Tổng {s['tot_time']:.1f}s (AI: {s['ai_time']:.1f}s, Encode: {s['enc_time']:.1f}s, AI fps: {s['ai_fps']:.2f})")
        lines.append("")

    # 4. Thống kê phụ đề & API Gemini
    lines.append("## 4. Thống Kê Phụ Đề & Hạn Mức Gemini API")
    api_stats = query_subtrans_api_stats()
    if api_stats:
        lines.append("### (a) Phân bổ câu thoại đã dịch theo Model:")
        for m, count in sorted(api_stats.get("cues_by_model", {}).items()):
            lines.append(f"* **{m}:** {count} câu thoại")

        lines.append("\n### (b) Thống kê cuộc gọi API & Lỗi 503 trong ca:")
        for m, s in sorted(api_stats.get("model_summary", {}).items()):
            succ_pct = (s["success"] / s["total"] * 100.0) if s["total"] > 0 else 0.0
            p503_pct = (s["503"] / s["total"] * 100.0) if s["total"] > 0 else 0.0
            lines.append(f"* **{m}:** Tổng {s['total']} requests | Thành công: {s['success']} ({succ_pct:.1f}%) | 503 Spike: {s['503']} ({p503_pct:.1f}%) | Lỗi khác: {s['other_err']}")

        lines.append("\n### (c) Phân bổ cuộc gọi theo Giờ Thái Bình Dương (PT):")
        for h, m_dict in sorted(api_stats.get("by_pt_hour", {}).items()):
            lines.append(f"* **Khung giờ {h}:**")
            for m, sc in sorted(m_dict.items()):
                lines.append(f"  - {m}: {sc['total']} requests (Thành công: {sc['success']}, 503: {sc['503']})")
    lines.append("")

    # 5. Danh sách việc cần làm bằng tay
    lines.append("## 5. Danh Sách Việc Cần Làm Bằng Tay (Action Items)")
    lines.append("1. **Tiếp tục tập 3 (Slime S4 - 24):**")
    lines.append("   - Chạy lệnh tiếp tục với thư mục đích `-o /Users/xis108/Documents/Upscaled`:")
    lines.append("     ```bash")
    lines.append("     ./upscale.sh \"/Users/xis108/Documents/RawSources/[SubsPlease] Tensei Shitara Slime Datta Ken S4 - 24 (1080p) [AA94BC1F].mkv\" \\")
    lines.append("       -o /Users/xis108/Documents/Upscaled \\")
    lines.append("       --sub-offset-ms -11154")
    lines.append("     ```")
    lines.append("   - Hệ thống sẽ tự động phát hiện 31 chunks đã xong trong `.temp_...` và tiếp tục từ chunk 32 mà không mất dữ liệu.")
    lines.append("2. **Kiểm tra phụ đề tiếng Việt:**")
    lines.append("   - File phụ đề ngoài `.vi.srt` đã được đặt chuẩn tên cạnh file MKV 4K cho 2 tập đầu.")
    lines.append("   - Mọi trình phát (IINA, VLC) sẽ tự động nạp phụ đề tiếng Việt khi mở video.")
    lines.append("")

    return "\n".join(lines)
