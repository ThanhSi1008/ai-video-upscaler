"""Read-only overnight run reporter aggregating logs, telemetry CSV, audits, and SQLite."""

import glob
import os
import re
import csv
import json
import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional


CHUNK_LOG_REGEX = re.compile(
    r"\[CHUNK (?P<idx>\d+)/(?P<total>\d+)\] Hoàn tất (?P<frames>\d+) frames "
    r"\(AI (?:CUDA|GPU): (?P<ai_time>[\d\.]+)s @ (?P<ai_fps>[\d\.]+) fps"
    r"(?:\s*\|\s*(?:T0:\s*[\d\.]+s(?:,\s*)?)?"
    r"(?:Steady:\s*(?P<steady_fps>[\d\.]+)\s*fps)?(?:,\s*)?"
    r"(?:Peak RSS:\s*(?P<rss_mb>[\d\.]+)\s*MB)?)?"
    r"\s*\|\s*Encode: (?P<enc_time>[\d\.]+)s \| Tổng: (?P<tot_time>[\d\.]+)s \| "
    r"File: (?P<file>\S+) (?P<size>[\d\.]+) (?:MB|GB)\)"
)


def _parse_telemetry_ts(raw_ts: Any) -> Optional[datetime]:
    """Parse an ISO or space-separated telemetry timestamp into a datetime."""
    text = str(raw_ts or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_optional_float(val: Any) -> Optional[float]:
    """Parse numeric telemetry values while treating N/A and empty fields as None."""
    if val is None:
        return None
    cleaned = str(val).strip()
    if not cleaned or cleaned.upper() in {"N/A", "[NOT SUPPORTED]", "UNKNOWN", "ERR"}:
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _safe_telem_float(val: Any, default: float = 0.0) -> float:
    """Convert a telemetry cell to float with a safe fallback."""
    parsed = _parse_optional_float(val)
    return parsed if parsed is not None else default


def _row_epoch_seconds(row: Dict[str, Any]) -> Optional[int]:
    """Extract integer epoch seconds from legacy `epoch` or ISO `timestamp` rows."""
    raw_epoch = str(row.get("epoch") or "").strip()
    if raw_epoch:
        try:
            return int(float(raw_epoch))
        except (ValueError, TypeError):
            pass
    dt = _parse_telemetry_ts(row.get("timestamp"))
    if dt is not None:
        return int(dt.timestamp())
    return None


def _parse_windows_telemetry_window(
    rows: List[Dict[str, Any]],
    start_epoch: Optional[int],
    end_epoch: Optional[int],
) -> Dict[str, Any]:
    """Aggregate Windows 17-column TelemetryCollector CSV metrics inside [start_epoch, end_epoch]."""
    parsed_rows = []
    for row in rows:
        dt = _parse_telemetry_ts(row.get("timestamp"))
        ep = _row_epoch_seconds(row)
        if ep is None:
            continue
        if start_epoch is not None and ep < start_epoch:
            continue
        if end_epoch is not None and ep > end_epoch:
            continue
        exact_ts = dt.timestamp() if dt is not None else float(ep)
        parsed_rows.append((ep, exact_ts, row))

    if not parsed_rows:
        return {}

    rows_in_window = [r for _, _, r in parsed_rows]
    total_read_mb = 0.0
    total_write_mb = 0.0
    prev_elapsed: Optional[float] = None
    prev_ts: Optional[float] = None

    for idx, (_, exact_ts, row) in enumerate(parsed_rows):
        cur_elapsed = _parse_optional_float(row.get("elapsed_s"))
        if len(parsed_rows) == 1:
            dt_sec = 1.0
        elif idx == 0:
            dt_sec = 0.0
        elif cur_elapsed is not None and prev_elapsed is not None and cur_elapsed >= prev_elapsed:
            dt_sec = max(0.0, cur_elapsed - prev_elapsed)
        elif exact_ts is not None and prev_ts is not None and exact_ts >= prev_ts:
            dt_sec = max(0.0, exact_ts - prev_ts)
        else:
            dt_sec = 1.0

        if dt_sec > 0.0:
            total_read_mb += max(0.0, _safe_telem_float(row.get("ssd_read_mb_s"))) * dt_sec
            total_write_mb += max(0.0, _safe_telem_float(row.get("ssd_write_mb_s"))) * dt_sec

        prev_elapsed = cur_elapsed
        prev_ts = exact_ts

    gpu_utils = [_safe_telem_float(r.get("gpu_util_pct")) for r in rows_in_window]
    vrams = [_safe_telem_float(r.get("vram_used_mib")) for r in rows_in_window]
    powers = [_safe_telem_float(r.get("power_w")) for r in rows_in_window]
    temps = [_safe_telem_float(r.get("temp_c")) for r in rows_in_window]
    cpus = [_safe_telem_float(r.get("cpu_pct")) for r in rows_in_window]
    rams = [_safe_telem_float(r.get("ram_used_gb")) for r in rows_in_window]
    pstates = sorted({
        str(r.get("pstate")).strip()
        for r in rows_in_window
        if r.get("pstate") and str(r.get("pstate")).strip().upper() != "N/A"
    }) or ["N/A"]

    peak_temp_c = max(temps) if temps else 0.0
    thermal_states = ["Nominal"] if peak_temp_c < 83.0 else ["Thermal Throttle (>=83C)"]
    delta_read_gib = total_read_mb / 1024.0
    delta_write_gib = total_write_mb / 1024.0
    delta_ssd_mb = total_read_mb + total_write_mb
    delta_ssd_gib = delta_ssd_mb / 1024.0
    n_samples = len(rows_in_window)

    return {
        "schema": "windows_telemetry_v1",
        "start_time": rows_in_window[0].get("timestamp", ""),
        "end_time": rows_in_window[-1].get("timestamp", ""),
        "samples_count": n_samples,
        "delta_ssd_mb": delta_ssd_mb,
        "delta_ssd_gib": delta_ssd_gib,
        "delta_read_gib": delta_read_gib,
        "delta_write_gib": delta_write_gib,
        "avg_gpu_util_pct": sum(gpu_utils) / n_samples,
        "peak_vram_mib": max(vrams) if vrams else 0.0,
        "avg_power_w": sum(powers) / n_samples,
        "peak_power_w": max(powers) if powers else 0.0,
        "peak_temp_c": peak_temp_c,
        "avg_cpu_pct": sum(cpus) / n_samples,
        "peak_ram_used_gb": max(rams) if rams else 0.0,
        "pstates": pstates,
        "thermal_states": thermal_states,
    }


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
        fieldnames = set(reader.fieldnames or [])
        is_legacy_schema = "epoch" in fieldnames and "disk0_mb_total" in fieldnames
        is_windows_schema = not is_legacy_schema and (
            "gpu_util_pct" in fieldnames or "vram_used_mib" in fieldnames
        )
        if is_windows_schema:
            return _parse_windows_telemetry_window(list(reader), start_epoch, end_epoch)

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
        thermal_states = sorted({
            str(r["thermal_state"]).strip()
            for r in rows_in_window
            if r.get("thermal_state")
        }) or ["Nominal"]

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
                c_idx = int(m.group("idx"))
                total_segs = int(m.group("total"))
                frames = int(m.group("frames"))
                ai_time = float(m.group("ai_time"))
                ai_fps = float(m.group("ai_fps"))
                steady_fps = float(m.group("steady_fps") or ai_fps)
                rss = float(m.group("rss_mb") or 0.0)
                enc_time = float(m.group("enc_time"))
                tot_time = float(m.group("tot_time"))
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


def query_subtrans_api_stats(
    db_path: str = "subtrans/.cache/translations.db",
    start_utc: Optional[str] = None,
    end_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Queries Gemini API stats from SQLite cache in read-only mode."""
    if not os.path.isfile(db_path):
        return {}

    ro_uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    con = sqlite3.connect(ro_uri, uri=True, timeout=30.0)
    try:
        con.execute("PRAGMA busy_timeout=5000")
        cur = con.cursor()

        try:
            tables = {
                row[0]
                for row in cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "translations" not in tables:
                return {}

            # Thống kê bản dịch theo model
            cur.execute("SELECT model, count(*) FROM translations GROUP BY model")
            cues_by_model = {row[0]: row[1] for row in cur.fetchall()}

            req_rows = []
            pt_rows = []
            if "api_requests" in tables:
                # Tự động xác định cửa sổ thời gian mới nhất nếu không truyền start_utc / end_utc
                eff_start = start_utc
                eff_end = end_utc
                if eff_start is None or eff_end is None:
                    cur.execute("SELECT MAX(created_at) FROM api_requests WHERE status != 'dry_run'")
                    row_max = cur.fetchone()
                    max_ts = row_max[0] if row_max else None
                    if max_ts:
                        try:
                            dt_max = datetime.strptime(max_ts, "%Y-%m-%d %H:%M:%S")
                            if eff_end is None:
                                eff_end = max_ts
                            if eff_start is None:
                                eff_start = (dt_max - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            pass

                if eff_start is not None and eff_end is not None:
                    cur.execute(
                        """
                        SELECT model, status, error_type, count(*)
                        FROM api_requests
                        WHERE status != 'dry_run' AND created_at >= ? AND created_at <= ?
                        GROUP BY model, status, error_type
                        """,
                        (eff_start, eff_end),
                    )
                else:
                    cur.execute(
                        """
                        SELECT model, status, error_type, count(*)
                        FROM api_requests
                        WHERE status != 'dry_run'
                        GROUP BY model, status, error_type
                        """
                    )
                req_rows = cur.fetchall()

                # Thống kê theo PT Hour (chuyển đổi created_at UTC từ SQLite sang America/Los_Angeles)
                if eff_start is not None and eff_end is not None:
                    cur.execute(
                        """
                        SELECT created_at, model, status, error_type
                        FROM api_requests
                        WHERE status != 'dry_run' AND created_at >= ? AND created_at <= ?
                        """,
                        (eff_start, eff_end),
                    )
                else:
                    cur.execute(
                        """
                        SELECT created_at, model, status, error_type
                        FROM api_requests
                        WHERE status != 'dry_run'
                        """
                    )
                pt_rows = cur.fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return {}
            raise
    finally:
        con.close()

    model_summary = {}
    for m, status, err_type, count in req_rows:
        if m not in model_summary:
            model_summary[m] = {"total": 0, "success": 0, "503": 0, "other_err": 0}
        model_summary[m]["total"] += count
        if status == "success":
            model_summary[m]["success"] += count
        elif "503" in str(status) or err_type == "503_Unavailable":
            model_summary[m]["503"] += count
        else:
            model_summary[m]["other_err"] += count

    pt_tz = ZoneInfo("America/Los_Angeles")
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
            elif "503" in str(status) or err_type == "503_Unavailable":
                by_pt_hour[h_str][m]["503"] += 1
        except Exception:
            continue

    return {
        "cues_by_model": cues_by_model,
        "model_summary": model_summary,
        "by_pt_hour": by_pt_hour,
    }


def _inspect_episode_log_summary(log_path: str) -> Dict[str, Any]:
    """Đọc tóm tắt siêu dữ liệu tập từ file log (tên tập, khung hình, trạng thái hoàn tất)."""
    ep_name = os.path.basename(log_path)
    total_frames = 0
    n_unique = 0
    status = "⏳ ĐANG XỬ LÝ / CHƯA HOÀN TẤT"
    started_dt = None
    ended_dt = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        m_vid = re.search(r"^# Video:\s*(.+)$", text, flags=re.MULTILINE)
        if m_vid:
            ep_name = m_vid.group(1).strip()
        m_started = re.search(r"^# Started:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text, flags=re.MULTILINE)
        if m_started:
            started_dt = datetime.strptime(m_started.group(1), "%Y-%m-%d %H:%M:%S")
        m_dedup = re.search(r"Kế hoạch Deduplication:\s*Giữ\s*(\d+)/(\d+)", text)
        if m_dedup:
            n_unique = int(m_dedup.group(1))
            total_frames = int(m_dedup.group(2))
        ts_matches = re.findall(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", text, flags=re.MULTILINE)
        if ts_matches:
            if started_dt is None:
                started_dt = datetime.strptime(ts_matches[0], "%Y-%m-%d %H:%M:%S")
            ended_dt = datetime.strptime(ts_matches[-1], "%Y-%m-%d %H:%M:%S")
        if "Mux Audit: PASS" in text or "HOÀN TẤT 4K" in text.upper() or "HOÀN TẤT THÀNH CÔNG" in text.upper():
            status = "✅ HOÀN TẤT TRỌN VẸN (100%)"
        elif "ALLOW-PARTIAL DEADLINE" in text or "GRACEFUL PAUSE" in text:
            status = "⏸️ TẠM DỪNG SẠCH (--allow-partial / pause)"
        elif "[ERROR]" in text:
            status = "❌ LỖI TRONG QUÁ TRÌNH XỬ LÝ"
    except Exception:
        pass
    actual_wall = (
        max(0.0, (ended_dt - started_dt).total_seconds())
        if (started_dt and ended_dt)
        else 0.0
    )
    return {
        "name": ep_name,
        "log": log_path,
        "status": status,
        "total_frames": total_frames,
        "n_unique": n_unique,
        "started_dt": started_dt,
        "ended_dt": ended_dt,
        "wall_start": started_dt.strftime("%H:%M:%S") if started_dt else "N/A",
        "wall_end": ended_dt.strftime("%H:%M:%S") if ended_dt else "N/A",
        "actual_wall_sec": actual_wall,
    }


def _discover_session_logs(
    logs_dir: str = "logs",
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Tìm tự động danh sách file log tập thuộc cửa sổ ca chạy (không hardcode ngày cố định)."""
    if not os.path.isdir(logs_dir):
        return []
    candidates = [
        p for p in sorted(glob.glob(os.path.join(logs_dir, "*.log")))
        if os.path.basename(p).lower() != "latest.log" and not os.path.islink(p)
    ]
    if not candidates:
        return []

    parsed = [_inspect_episode_log_summary(p) for p in candidates]
    if start_dt is not None and end_dt is not None:
        win_start = start_dt - timedelta(minutes=5)
        win_end = end_dt + timedelta(minutes=5)
        selected = [
            ep for ep in parsed
            if ep["started_dt"] is not None and win_start <= ep["started_dt"] <= win_end
        ]
        return sorted(selected, key=lambda ep: ep["started_dt"] or datetime.min)

    with_ts = [ep for ep in parsed if ep["started_dt"] is not None]
    if not with_ts:
        return []
    latest_ts = max(ep["ended_dt"] or ep["started_dt"] for ep in with_ts)
    cutoff = latest_ts - timedelta(hours=12)
    selected = [ep for ep in with_ts if (ep["ended_dt"] or ep["started_dt"]) >= cutoff]
    return sorted(selected, key=lambda ep: ep["started_dt"])


def generate_night_report_text(
    session_name: str = "night1",
    workspace_root: Optional[Path | str] = None,
) -> str:
    """Generate Markdown overnight run summary from session logs, telemetry CSV, and SQLite stats."""
    base_root = Path(workspace_root).resolve() if workspace_root is not None else None
    lines = []
    lines.append(f"# BÁO CÁO TỔNG KẾT CA CHẠY QUA ĐÊM ({session_name.upper()})")
    lines.append("")

    # 1. Cửa sổ phiên chạy (ưu tiên đọc từ CSV telemetry của session; fallback từ log mới nhất)
    if base_root is not None:
        logs_dir_path = base_root / "logs"
        bench_dir_path = base_root / "out" / "benchmarks"
        candidate_csvs = [
            str(logs_dir_path / f"soak_telemetry_{session_name}.csv"),
            str(logs_dir_path / f"{session_name}_telemetry.csv"),
            str(logs_dir_path / f"telemetry_{session_name}.csv"),
            str(bench_dir_path / f"{session_name}_telemetry.csv"),
        ]
        logs_dir_arg = str(logs_dir_path)
    else:
        candidate_csvs = [
            f"logs/soak_telemetry_{session_name}.csv",
            f"logs/{session_name}_telemetry.csv",
            f"logs/telemetry_{session_name}.csv",
            f"out/benchmarks/{session_name}_telemetry.csv",
        ]
        logs_dir_arg = "logs"
    csv_file = next((p for p in candidate_csvs if os.path.exists(p)), candidate_csvs[0])
    start_epoch: Optional[int] = None
    end_epoch: Optional[int] = None
    if os.path.exists(csv_file):
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                epochs = [ep for ep in (_row_epoch_seconds(r) for r in rows) if ep is not None]
                if epochs:
                    start_epoch = epochs[0]
                    end_epoch = epochs[-1]
        except Exception:
            pass

    csv_start_dt = datetime.fromtimestamp(start_epoch) if start_epoch is not None else None
    csv_end_dt = datetime.fromtimestamp(end_epoch) if end_epoch is not None else None
    episodes_info = _discover_session_logs(logs_dir_arg, start_dt=csv_start_dt, end_dt=csv_end_dt)

    if csv_start_dt is not None and csv_end_dt is not None:
        start_dt = csv_start_dt
        end_dt = csv_end_dt
    elif episodes_info:
        starts = [ep["started_dt"] for ep in episodes_info if ep["started_dt"] is not None]
        ends = [ep["ended_dt"] or ep["started_dt"] for ep in episodes_info if (ep["ended_dt"] or ep["started_dt"]) is not None]
        start_dt = min(starts) if starts else datetime.now()
        end_dt = max(ends) if ends else start_dt
        start_epoch = int(start_dt.timestamp())
        end_epoch = int(end_dt.timestamp())
    else:
        start_dt = datetime.now()
        end_dt = start_dt

    window_sec = max(1.0, (end_dt - start_dt).total_seconds())

    lines.append("## 1. Thông Tin Phiên Chạy & Cửa Sổ Giám Sát")
    lines.append(f"* **Thời điểm bắt đầu:** {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"* **Thời điểm kết thúc ca:** {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"* **Tổng thời lượng ca:** {int(window_sec//3600)}h {int((window_sec%3600)//60)}m {int(window_sec%60)}s ({window_sec:,.1f}s)")
    lines.append("")

    # 2. Telemetry Hệ thống
    telem = parse_telemetry_window(csv_file, start_epoch, end_epoch)
    lines.append("## 2. Viễn Trắc Hệ Thống (Phần Cứng & Ổ Cứng)")
    if telem and not telem.get("error"):
        if telem.get("schema") == "windows_telemetry_v1":
            pstates_str = ", ".join(telem.get("pstates", ["N/A"]))
            lines.append(
                f"* **GPU Utilization (TB):** **{telem['avg_gpu_util_pct']:.1f}%** "
                f"(P-State: {pstates_str} | Mẫu: {telem['samples_count']})"
            )
            lines.append(f"* **Đỉnh VRAM (RTX 3060):** **{telem['peak_vram_mib']:,.0f} MiB**")
            lines.append(
                f"* **Công suất & Nhiệt độ GPU:** TB **{telem['avg_power_w']:.1f} W** "
                f"(Đỉnh: **{telem['peak_power_w']:.1f} W**) | Đỉnh nhiệt: **{telem['peak_temp_c']:.1f}°C** "
                f"({', '.join(telem.get('thermal_states', ['Nominal']))})"
            )
            lines.append(
                f"* **CPU & RAM Hệ thống:** CPU TB **{telem['avg_cpu_pct']:.1f}%** | "
                f"Đỉnh RAM: **{telem['peak_ram_used_gb']:.2f} GB**"
            )
            lines.append(
                f"* **Băng thông NVMe SSD (tích phân):** Đọc **{telem['delta_read_gib']:.2f} GiB** + "
                f"Ghi **{telem['delta_write_gib']:.2f} GiB** = Tổng **{telem['delta_ssd_mb']:,.2f} MB** "
                f"(**{telem['delta_ssd_gib']:.2f} GiB**)"
            )
        else:
            lines.append(f"* **Dung lượng chuyển SSD (đọc+ghi qua iostat disk0):** **{telem['delta_ssd_mb']:,.2f} MB** (**{telem['delta_ssd_gib']:.2f} GiB**)")
            if telem.get("delta_write_gib") is not None and telem.get("delta_read_gib") is not None:
                lines.append(f"* **Dung lượng ghi SSD thực tế (ioreg):** **{telem['delta_write_gib']:.2f} GiB**")
                lines.append(f"* **Dung lượng đọc SSD thực tế (ioreg):** **{telem['delta_read_gib']:.2f} GiB**")
            lines.append(f"* **Tăng trưởng Pageouts:** **+{telem['delta_pageouts']:,}** trang")
            lines.append(f"* **Bộ nhớ Swap:** Ban đầu **{telem['start_swap_mb']:.2f} MB** ➔ Kết thúc **{telem['end_swap_mb']:.2f} MB** (Tăng: **+{telem['delta_swap_mb']:.2f} MB**, Đỉnh: **{telem['peak_swap_mb']:.2f} MB**)")
            lines.append(f"* **Trạng thái nhiệt:** {', '.join(telem.get('thermal_states', ['Nominal']))}")
    else:
        lines.append(f"* Không nạp được telemetry từ {csv_file}")
    lines.append("")

    # 3. Trạng thái từng tập video
    lines.append("## 3. Trạng Thái & Hiệu Năng Từng Tập Video")
    lines.append("")

    if not episodes_info:
        lines.append("* Không tìm thấy file log tập video nào trong cửa sổ phiên chạy.")
        lines.append("")
    else:
        for idx, ep in enumerate(episodes_info, 1):
            lines.append(f"### Tập {idx}: {ep['name']}")
            lines.append(f"* **Trạng thái:** {ep['status']}")
            lines.append(
                f"* **Thời gian thực tế:** {int(ep['actual_wall_sec']//3600)}h {int((ep['actual_wall_sec']%3600)//60)}m "
                f"{int(ep['actual_wall_sec']%60)}s ({ep['wall_start']} ➔ {ep['wall_end']})"
            )

            c_data = parse_episode_chunks_log(ep["log"])
            if c_data:
                lines.append(f"* **Số phân đoạn đã xử lý:** {c_data['chunks_completed']}/{c_data['total_chunks']}")
                lines.append(f"* **Tốc độ AI:** Trung vị **{c_data['median_ai_fps']:.2f} fps** | P5 **{c_data['p5_ai_fps']:.2f} fps**")
                lines.append(f"* **So sánh tốc độ giờ đầu vs giờ cuối:** Giờ đầu: **{c_data['first_hour_fps']:.2f} fps** ➔ Giờ cuối: **{c_data['last_hour_fps']:.2f} fps**")

                gpu_active_ratio = (c_data["total_ai_time_sec"] / ep["actual_wall_sec"]) * 100.0 if ep["actual_wall_sec"] > 0 else 0.0
                lines.append(f"* **GPU Active Ratio:** **{gpu_active_ratio:.2f}%** (Tổng AI GPU: {c_data['total_ai_time_sec']:,.1f}s ÷ Thời gian thực tế: {ep['actual_wall_sec']:,.1f}s)")

                lines.append("* **10 Phân đoạn chậm nhất:**")
                for s in c_data["slowest_10"][:10]:
                    lines.append(f"  - Chunk {s['idx']:03d}: Tổng {s['tot_time']:.1f}s (AI: {s['ai_time']:.1f}s, Encode: {s['enc_time']:.1f}s, AI fps: {s['ai_fps']:.2f})")
            lines.append("")

    # 4. Thống kê phụ đề & API Gemini
    lines.append("## 4. Thống Kê Phụ Đề & Hạn Mức Gemini API")

    def _utc_sql_bound(dt_val: datetime, ep_val: Optional[int]) -> Optional[str]:
        try:
            ts_val = float(ep_val) if ep_val is not None else dt_val.timestamp()
            return datetime.fromtimestamp(ts_val, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            return None

    start_utc_str = _utc_sql_bound(start_dt, start_epoch)
    end_utc_str = _utc_sql_bound(end_dt, end_epoch)
    db_kwargs = (
        {"db_path": str(base_root / "subtrans" / ".cache" / "translations.db")}
        if base_root is not None
        else {}
    )
    api_stats = (
        query_subtrans_api_stats(start_utc=start_utc_str, end_utc=end_utc_str, **db_kwargs)
        if ((start_epoch is not None or episodes_info) and start_utc_str and end_utc_str)
        else query_subtrans_api_stats(**db_kwargs)
    )
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
    lines.append("1. **Tiếp tục tập dở dang (Resume Checkpoint):**")
    lines.append("   - Chạy lệnh tiếp tục với thư mục đích `-o out/`:")
    lines.append("     ```powershell")
    lines.append("     .\\upscale.bat \"<duong_dan_video.mkv>\" -o out\\")
    lines.append("     python upscale.py \"<duong_dan_video.mkv>\" -o out/")
    lines.append("     ```")
    lines.append("   - Hệ thống sẽ tự động phát hiện các chunks đã xong trong `.temp_...` và tiếp tục mà không làm mất dữ liệu.")
    lines.append("2. **Kiểm tra phụ đề tiếng Việt:**")
    lines.append("   - File phụ đề ngoài `.vi.srt` được đặt chuẩn tên cạnh file MKV 4K để mọi trình phát tự động nạp.")
    lines.append("")

    return "\n".join(lines)
