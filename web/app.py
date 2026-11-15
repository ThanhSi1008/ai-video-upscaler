# -*- coding: utf-8 -*-
"""
web/app.py
Ứng dụng FastAPI trung tâm cho AI Video Upscaler & Subtrans Real-Time Studio:
- Phục vụ giao diện Web hiện đại (web/static/) và luồng SSE 1.0 Hz (/api/events).
- Cung cấp đầy đủ REST API cho Live Monitor, Queue & Scheduler, Preview 1080p vs 4K,
  Models & shows.yaml Configurator, Subtrans & Audit Vault.
- Cho phép vận hành bình thường mọi tác vụ (Upscale, Preview, Subtrans, Subs-Pending,
  cấu hình shows.yaml & quản lý API Key cục bộ) ở bất kỳ khung giờ nào.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional
import yaml

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from subtrans.cache import compute_key_id, get_next_pt_reset, get_now_pt
from web.fs_browser import (
    DEDUP_POLICY_PARAMS,
    SUBTITLE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    SafeLocalFSBrowser,
    is_secret_filename,
    match_show_pattern,
)
from web.job_manager import MODEL_REFERENCE_FPS, JobManager
from web.sse_hub import TelemetryHub, scan_temp_checkpoints

ALLOWED_OPEN_MEDIA_EXTS = VIDEO_EXTENSIONS | SUBTITLE_EXTENSIONS
STATIC_DIR = Path(__file__).resolve().parent / "static"


# =====================================================================
# PYDANTIC PAYLOAD SCHEMAS
# =====================================================================

class SchedulePreviewRequest(BaseModel):
    inputs: List[str] = Field(default_factory=list)
    deadline: Optional[str] = None
    allow_partial: bool = False
    model: Optional[str] = None
    dedup_policy: Optional[str] = None
    zero_disk: bool = True
    use_tensorrt: bool = True
    subs_after_upscale: bool = True
    output_dir: Optional[str] = None


class StartUpscaleRequest(BaseModel):
    inputs: List[str] = Field(default_factory=list)
    output_dir: Optional[str] = None
    model: Optional[str] = None
    dedup_policy: Optional[str] = None
    batch_size: int = 8
    deadline: Optional[str] = None
    allow_partial: bool = False
    zero_disk: bool = True
    use_tensorrt: bool = True
    subs_after_upscale: bool = True
    subs_quality: str = "normal"
    async_subs: bool = False
    prefer_whisper: bool = False
    auto_align_en: bool = True
    sub_offset_ms: int = 0
    keep_temp: bool = False
    force: bool = False
    dry_run: bool = False


class StartPreviewRequest(BaseModel):
    video_path: str
    seconds: int = 10
    model: Optional[str] = None
    dedup_policy: Optional[str] = None
    zero_disk: bool = True
    use_tensorrt: bool = True
    force: bool = True
    auto_open_player: bool = False


class StartSubtransRequest(BaseModel):
    input_path: str
    output_dir: Optional[str] = None
    subs_quality: str = "normal"
    subs_upgrade: bool = False
    prefer_whisper: bool = False
    auto_align_en: bool = True
    no_embed: bool = False
    sub_offset_ms: int = 0
    glossary: Optional[str] = None
    dry_run: bool = False


class StartSubsPendingRequest(BaseModel):
    subs_quality: str = "normal"
    dry_run: bool = False


class OpenMediaRequest(BaseModel):
    path: str
    prefer_mpv: bool = True


class CompareFramesRequest(BaseModel):
    source_path: str
    upscaled_path: str
    timestamp_sec: float = 2.0
    source_offset_sec: Optional[float] = None


class TestShowsMatchRequest(BaseModel):
    sample_path: str
    yaml_content: Optional[str] = None


class SaveShowsYamlRequest(BaseModel):
    yaml_content: str


class UpdateApiKeysRequest(BaseModel):
    key_1: Optional[str] = None
    key_2: Optional[str] = None
    key_3: Optional[str] = None


# =====================================================================
# HELPER UTILITIES
# =====================================================================

def _mask_secret(raw: str) -> str:
    val = (raw or "").strip()
    if not val:
        return ""
    if len(val) <= 10:
        return f"{val[:2]}***{val[-2:]}"
    return f"{val[:6]}...{val[-4:]}"


def _read_dotenv_map(dotenv_path: Path) -> Dict[str, str]:
    """Đọc các cặp KEY=VALUE từ file .env cục bộ mà không bao giờ in hoặc log giá trị thô."""
    result: Dict[str, str] = {}
    if not dotenv_path.is_file():
        return result
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                v = v[1:-1]
            if k:
                result[k] = v
    except Exception:
        pass
    return result


def _write_dotenv_keys(dotenv_path: Path, updates: Dict[str, Optional[str]]) -> None:
    """Cập nhật an toàn các biến GEMINI_API_KEY_1..3 vào .env cục bộ, bảo toàn mọi dòng khác."""
    existing_lines: List[str] = []
    if dotenv_path.is_file():
        existing_lines = dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines()

    handled: set[str] = set()
    new_lines: List[str] = []

    for raw_line in existing_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(raw_line)
            continue
        check_line = stripped[len("export "):].strip() if stripped.startswith("export ") else stripped
        key_name = check_line.split("=", 1)[0].strip()
        if key_name in updates and updates[key_name] is not None:
            val = (updates[key_name] or "").strip()
            handled.add(key_name)
            if val:
                new_lines.append(f"{key_name}={val}")
            # Nếu chuỗi rỗng -> xóa dòng khóa cũ đó
        else:
            new_lines.append(raw_line)

    for key_name, raw_val in updates.items():
        if raw_val is None or key_name in handled:
            continue
        val = raw_val.strip()
        if val:
            new_lines.append(f"{key_name}={val}")

    tmp_path = dotenv_path.parent / f".env.{os.getpid()}.tmp"
    content = "\n".join(new_lines).rstrip("\n") + ("\n" if new_lines else "")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, dotenv_path)


def _probe_duration_and_dims(video_path: Path) -> Dict[str, Any]:
    """Đọc nhanh thời lượng (giây) và kích thước khung hình qua PyAV (hoặc fallback mặc định)."""
    try:
        import av
        with av.open(str(video_path)) as container:
            if container.streams.video:
                vs = container.streams.video[0]
                dur = float(container.duration / av.time_base) if container.duration else 0.0
                return {
                    "duration": round(dur, 3),
                    "width": int(vs.width or 0),
                    "height": int(vs.height or 0),
                }
    except Exception:
        pass
    return {"duration": 0.0, "width": 0, "height": 0}


def _default_extract_frame(video_path: Path, timestamp_sec: float, out_png: Path) -> None:
    """Trích xuất 1 khung hình PNG chuẩn ma trận màu BT.709 (YUV Limited -> RGB Full)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, float(timestamp_sec)):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc",
        str(out_png),
    ]
    startupinfo = None
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=25.0,
        check=False,
        startupinfo=startupinfo,
    )
    if res.returncode != 0 or not out_png.is_file():
        raise RuntimeError(
            f"FFmpeg không thể trích xuất khung hình tại {timestamp_sec:.2f}s từ '{video_path.name}'."
        )


def _default_open_media(target_path: Path, prefer_mpv: bool = True) -> str:
    """Mở file media trên máy cục bộ Windows qua mpv (nếu có) hoặc trình phát mặc định."""
    if prefer_mpv:
        import shutil
        mpv_bin = shutil.which("mpv")
        if mpv_bin:
            subprocess.Popen([mpv_bin, str(target_path)])
            return "mpv"
    if sys.platform == "win32" and hasattr(os, "startfile"):
        os.startfile(str(target_path))  # type: ignore[attr-defined]
        return "os.startfile"
    raise RuntimeError("Môi trường hiện tại không hỗ trợ mở trình phát media trực tiếp.")


# =====================================================================
# APPLICATION FACTORY
# =====================================================================

def create_app(
    workspace_root: Optional[Path] = None,
    gpu_query_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    popen_factory: Optional[Callable[..., Any]] = None,
    open_media_fn: Optional[Callable[[Path, bool], str]] = None,
    frame_extract_fn: Optional[Callable[[Path, float, Path], None]] = None,
) -> FastAPI:
    """Khởi tạo FastAPI app gắn với thư mục gốc dự án (hỗ trợ cô lập hoàn toàn khi test)."""
    root = Path(workspace_root or Path(__file__).resolve().parent.parent).resolve()
    fs_browser = SafeLocalFSBrowser(root)
    job_manager = JobManager(root, popen_factory=popen_factory)
    telemetry_hub = TelemetryHub(
        root,
        gpu_query_fn=gpu_query_fn,
        job_state_provider=job_manager.get_state,
        history_size=120,
    )
    media_opener = open_media_fn or _default_open_media
    extractor = frame_extract_fn or _default_extract_frame

    app = FastAPI(
        title="AI Video Upscaler & Subtrans Studio",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.workspace_root = root
    app.state.fs_browser = fs_browser
    app.state.job_manager = job_manager
    app.state.telemetry_hub = telemetry_hub

    @app.middleware("http")
    async def _security_headers_middleware(request: Any, call_next: Callable[[Any], Any]) -> Any:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        index_file = STATIC_DIR / "index.html"
        if not index_file.is_file():
            raise HTTPException(status_code=404, detail="Chưa tìm thấy giao diện web/static/index.html.")
        return FileResponse(str(index_file), media_type="text/html; charset=utf-8")

    # -----------------------------------------------------------------
    # 1. LIVE MONITOR & TELEMETRY SSE
    # -----------------------------------------------------------------
    @app.get("/api/status")
    def api_status() -> Dict[str, Any]:
        return telemetry_hub.build_snapshot()

    @app.get("/api/events")
    async def api_events(once: int = Query(default=0, ge=0, le=1)) -> StreamingResponse:
        headers = {
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            telemetry_hub.stream_events(once=bool(once)),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.get("/api/logs/recent")
    def api_recent_logs(lines: int = Query(default=120, ge=10, le=1000)) -> Dict[str, Any]:
        return {"lines": telemetry_hub.read_recent_logs(max_lines=lines)}

    # -----------------------------------------------------------------
    # 2. FILE SYSTEM BROWSER, QUEUE & JOB CONTROLS
    # -----------------------------------------------------------------
    @app.get("/api/fs/browse")
    def api_fs_browse(
        path: Optional[str] = Query(default=None),
        include_subtitles: bool = Query(default=False),
    ) -> Dict[str, Any]:
        try:
            return fs_browser.browse(raw_path=path, include_subtitles=include_subtitles)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/queue/preview-schedule")
    def api_preview_schedule(req: SchedulePreviewRequest) -> Dict[str, Any]:
        if not req.inputs:
            return {
                "total_episodes": 0,
                "accepted_count": 0,
                "already_done_count": 0,
                "skipped_count": 0,
                "total_eta_minutes": 0.0,
                "deadline_str": req.deadline or "",
                "episodes": [],
            }
        try:
            return job_manager.compute_schedule_preview(
                inputs=req.inputs,
                deadline_str=req.deadline,
                allow_partial=req.allow_partial,
                model_override=req.model,
                dedup_override=req.dedup_policy,
                zero_disk=req.zero_disk,
                use_tensorrt=req.use_tensorrt,
                subs_after_upscale=req.subs_after_upscale,
                output_dir=req.output_dir,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/start")
    def api_start_upscale(req: StartUpscaleRequest) -> Dict[str, Any]:
        try:
            job = job_manager.start_upscale_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/preview")
    def api_start_preview(req: StartPreviewRequest) -> Dict[str, Any]:
        try:
            job = job_manager.start_preview_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/subtrans")
    def api_start_subtrans(req: StartSubtransRequest) -> Dict[str, Any]:
        """Khởi chạy dịch phụ đề trực tiếp ở mọi khung giờ không bị khóa cứng."""
        try:
            job = job_manager.start_subtrans_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/subs-pending")
    def api_start_subs_pending(req: StartSubsPendingRequest) -> Dict[str, Any]:
        try:
            job = job_manager.start_subs_pending_job(req.model_dump())
            return {"ok": True, "job": job}
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/pause-graceful")
    def api_pause_graceful() -> Dict[str, Any]:
        try:
            return job_manager.request_graceful_pause()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/clear-pause")
    def api_clear_pause() -> Dict[str, Any]:
        return job_manager.clear_graceful_pause()

    @app.post("/api/jobs/abort")
    def api_abort_job() -> Dict[str, Any]:
        return job_manager.abort_job()

    @app.post("/api/open-media")
    def api_open_media(req: OpenMediaRequest) -> Dict[str, Any]:
        cleaned = (req.path or "").strip()
        if not cleaned or "\x00" in cleaned:
            raise HTTPException(status_code=400, detail="Đường dẫn không hợp lệ.")
        cand = Path(os.path.expanduser(cleaned))
        if not cand.is_absolute():
            cand = (root / cand).resolve()
        else:
            cand = cand.resolve()

        if is_secret_filename(cand.name):
            raise HTTPException(status_code=403, detail="Từ chối truy cập tệp nhạy cảm.")
        if not cand.is_file():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy tệp media: {cand}")
        if cand.suffix.lower() not in ALLOWED_OPEN_MEDIA_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Chỉ cho phép mở tệp video hoặc phụ đề hợp lệ (nhận được '{cand.suffix}').",
            )
        try:
            launcher = media_opener(cand, req.prefer_mpv)
            return {"ok": True, "path": str(cand), "launcher": launcher}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def _iter_out_search_dirs(out_dir: Path) -> List[Path]:
        """Quét out/, out/audit và các thư mục con cấp 1 hợp lệ (vd: out/Season1, out/benchmark_*)."""
        if not out_dir.is_dir():
            return []
        dirs: List[Path] = [out_dir]
        audit_sub = out_dir / "audit"
        if audit_sub.is_dir():
            dirs.append(audit_sub)
        try:
            for sub in sorted(out_dir.iterdir(), key=lambda p: p.name.lower()):
                if (
                    sub.is_dir()
                    and not sub.name.startswith(".")
                    and sub.name.lower() not in {"audit", "cast"}
                ):
                    dirs.append(sub)
        except OSError:
            pass
        return dirs

    # -----------------------------------------------------------------
    # 3. PREVIEW 1080p vs 4K SPLIT-COMPARE
    # -----------------------------------------------------------------
    @app.get("/api/preview/candidates")
    def api_preview_candidates() -> Dict[str, Any]:
        out_dir = root / "out"
        pairs: List[Dict[str, Any]] = []
        if not out_dir.is_dir():
            return {"candidates": pairs}

        # Lập bản đồ từ audit.json để biết chính xác source_file & output_file gốc
        audit_source_map: Dict[str, str] = {}
        audited_outputs: Dict[str, Path] = {}
        for audit_dir in _iter_out_search_dirs(out_dir):
            for af in audit_dir.glob("*.audit.json"):
                try:
                    adata = json.loads(af.read_text(encoding="utf-8"))
                    out_f = str(adata.get("output_file") or "")
                    src_f = str(adata.get("source_file") or "")
                    if out_f:
                        out_p = Path(out_f)
                        if not out_p.is_absolute():
                            out_p = (root / out_p).resolve()
                        else:
                            out_p = out_p.resolve()
                        if out_p.is_file() and out_p.suffix.lower() in VIDEO_EXTENSIONS:
                            norm_out = os.path.normcase(str(out_p))
                            audited_outputs[norm_out] = out_p
                            if src_f and Path(src_f).is_file():
                                audit_source_map[norm_out] = src_f
                    if src_f and Path(src_f).is_file():
                        audit_source_map[af.name[:-len(".audit.json")].lower()] = src_f
                except Exception:
                    pass

        candidate_map: Dict[str, Path] = dict(audited_outputs)
        scan_video_dirs = [d for d in _iter_out_search_dirs(out_dir) if d.name.lower() != "audit"]
        for vdir in scan_video_dirs:
            try:
                for p in vdir.iterdir():
                    if (
                        p.is_file()
                        and p.suffix.lower() in VIDEO_EXTENSIONS
                        and not p.name.startswith(".")
                        and ("_4k" in p.stem.lower())
                    ):
                        rp = p.resolve()
                        candidate_map[os.path.normcase(str(rp))] = rp
            except OSError:
                continue

        upscaled_files = sorted(
            candidate_map.values(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for up_file in upscaled_files[:40]:
            stem = up_file.stem
            norm_up = os.path.normcase(str(up_file.resolve()))
            src_path = audit_source_map.get(norm_up) or audit_source_map.get(stem.lower()) or ""

            # Đoán tên file gốc nếu là bản preview (vd: Ep01_preview_10s_4K -> Ep01)
            preview_match = re.match(r"^(.*)_preview_(\d+)s_4K(?:_animejanai)?$", stem, flags=re.IGNORECASE)
            is_preview_clip = bool(preview_match)
            preview_seconds = int(preview_match.group(2)) if preview_match else 0
            raw_base = (
                preview_match.group(1)
                if preview_match
                else re.sub(r"_4K(?:_animejanai)?$", "", stem, flags=re.IGNORECASE)
            )

            if (not src_path or ".preview_clip_" in src_path) and raw_base:
                for search_base in (root, out_dir, up_file.parent):
                    for ext in (".mkv", ".mp4", ".webm", ".avi"):
                        cand_src = search_base / f"{raw_base}{ext}"
                        if cand_src.is_file() and cand_src.resolve() != up_file.resolve():
                            src_path = str(cand_src.resolve())
                            break
                    if src_path and ".preview_clip_" not in src_path:
                        break

            rel_label = (
                f"{up_file.parent.name}/{up_file.name}"
                if up_file.parent.resolve() != out_dir.resolve()
                else up_file.name
            )
            pairs.append({
                "name": rel_label,
                "upscaled_path": str(up_file.resolve()),
                "source_path": src_path if (src_path and Path(src_path).is_file()) else "",
                "is_preview": is_preview_clip,
                "preview_seconds": preview_seconds,
                "size_mb": round(up_file.stat().st_size / (1024 * 1024), 2),
                "mtime": datetime.fromtimestamp(up_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })

        return {"candidates": pairs}

    @app.post("/api/preview/compare-frames")
    def api_compare_frames(req: CompareFramesRequest) -> Dict[str, Any]:
        src_p = Path(os.path.expanduser(req.source_path.strip()))
        up_p = Path(os.path.expanduser(req.upscaled_path.strip()))
        if not src_p.is_absolute():
            src_p = (root / src_p).resolve()
        else:
            src_p = src_p.resolve()
        if not up_p.is_absolute():
            up_p = (root / up_p).resolve()
        else:
            up_p = up_p.resolve()

        if is_secret_filename(src_p.name) or is_secret_filename(up_p.name):
            raise HTTPException(status_code=403, detail="Từ chối truy cập tệp nhạy cảm.")
        if not src_p.is_file():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy video gốc 1080p: {src_p}")
        if not up_p.is_file():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy video 4K đối chiếu: {up_p}")
        if src_p.suffix.lower() not in VIDEO_EXTENSIONS or up_p.suffix.lower() not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Định dạng file video không hợp lệ.")

        src_meta = _probe_duration_and_dims(src_p)
        up_meta = _probe_duration_and_dims(up_p)

        up_ts = max(0.0, float(req.timestamp_sec))
        if up_meta["duration"] > 0 and up_ts > max(0.0, up_meta["duration"] - 0.08):
            up_ts = max(0.0, round(up_meta["duration"] - 0.1, 3))

        if req.source_offset_sec is not None:
            offset_s = float(req.source_offset_sec)
        else:
            m_prev = re.match(r"^(.*)_preview_(\d+)s_4K(?:_animejanai)?$", up_p.stem, flags=re.IGNORECASE)
            if m_prev and src_meta["duration"] > 0:
                prev_dur = int(m_prev.group(2))
                max_dur = max(1, int(round(src_meta["duration"])))
                offset_s = (
                    0.0
                    if prev_dur >= max_dur
                    else float(max(0, min(int(src_meta["duration"] * 0.3), max_dur - prev_dur)))
                )
            else:
                offset_s = 0.0

        src_ts = max(0.0, up_ts + offset_s)

        compare_dir = root / "scratch" / "web_compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            f"{src_p}|{src_p.stat().st_mtime_ns}|{up_p}|{up_p.stat().st_mtime_ns}|{src_ts:.3f}|{up_ts:.3f}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]

        src_png = compare_dir / f"src_{digest}.png"
        up_png = compare_dir / f"up_{digest}.png"

        try:
            if not src_png.is_file():
                extractor(src_p, src_ts, src_png)
            if not up_png.is_file():
                extractor(up_p, up_ts, up_png)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Giữ tối đa 24 ảnh tạm mới nhất để tiết kiệm dung lượng ổ đĩa
        try:
            pngs = sorted(compare_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old_f in pngs[24:]:
                try:
                    old_f.unlink()
                except OSError:
                    pass
        except OSError:
            pass

        return {
            "ok": True,
            "source_frame_url": f"/api/preview/frame/{src_png.name}",
            "upscaled_frame_url": f"/api/preview/frame/{up_png.name}",
            "source_timestamp_sec": round(src_ts, 3),
            "upscaled_timestamp_sec": round(up_ts, 3),
            "source_offset_sec": round(offset_s, 3),
            "source_meta": src_meta,
            "upscaled_meta": up_meta,
        }

    @app.get("/api/preview/frame/{frame_name}")
    def api_get_preview_frame(frame_name: str) -> FileResponse:
        if not re.fullmatch(r"(?:src|up)_[a-f0-9]{12}\.png", frame_name):
            raise HTTPException(status_code=400, detail="Tên khung hình không hợp lệ.")
        frame_file = (root / "scratch" / "web_compare" / frame_name).resolve()
        if not frame_file.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy khung hình trích xuất.")
        return FileResponse(str(frame_file), media_type="image/png")

    # -----------------------------------------------------------------
    # 4. MODELS REGISTRY & SHOWS.YAML CONFIGURATOR
    # -----------------------------------------------------------------
    @app.get("/api/config/models-and-shows")
    def api_models_and_shows() -> Dict[str, Any]:
        models_file = root / "models" / "models.yaml"
        shows_file = root / "shows.yaml"
        trt_cache_dir = root / ".cache" / "trt_engines"

        models_list: List[Dict[str, Any]] = []
        if models_file.is_file():
            raw_m = yaml.safe_load(models_file.read_text(encoding="utf-8")) or {}
            reg = raw_m.get("models", {}) or {}
            for m_id, info in reg.items():
                if not isinstance(info, dict):
                    continue
                rel_w = str(info.get("weights_path") or "")
                w_path = (root / rel_w).resolve() if rel_w else None
                w_exists = bool(w_path and w_path.is_file())
                w_size_mb = round(w_path.stat().st_size / (1024 * 1024), 2) if w_exists and w_path else 0.0

                trt_engines: List[str] = []
                if trt_cache_dir.is_dir():
                    safe_id = str(m_id).replace(" ", "_").replace("/", "_")
                    prefixes = {safe_id}
                    if m_id == "animevideov3":
                        prefixes.add("realesr-animevideov3")
                    for ef in trt_cache_dir.glob("*.engine"):
                        if any(ef.name.startswith(f"{pfx}_") for pfx in prefixes):
                            trt_engines.append(ef.name)

                med_fps, p5_fps = MODEL_REFERENCE_FPS.get(str(m_id), (10.5, 8.5))
                arch = info.get("arch") or {}
                models_list.append({
                    "id": str(m_id),
                    "name": str(info.get("name") or m_id),
                    "arch_type": str(arch.get("type") or "Unknown"),
                    "num_feat": arch.get("num_feat"),
                    "num_conv": arch.get("num_conv"),
                    "native_upscale": arch.get("upscale", 2),
                    "downsample": str(info.get("downsample") or "none"),
                    "weights_path": rel_w,
                    "weights_ready": w_exists,
                    "weights_size_mb": w_size_mb,
                    "trt_cached": len(trt_engines) > 0,
                    "trt_engine_files": trt_engines,
                    "sha256_prefix": str(info.get("sha256") or "")[:12],
                    "license": str(info.get("license") or ""),
                    "default": bool(info.get("default", False)),
                    "ref_median_fps": med_fps,
                    "ref_p5_fps": p5_fps,
                })

        shows_text = shows_file.read_text(encoding="utf-8") if shows_file.is_file() else "shows: []\n"
        shows_rules: List[Dict[str, Any]] = []
        try:
            parsed_s = yaml.safe_load(shows_text) or {}
            shows_rules = [r for r in (parsed_s.get("shows") or []) if isinstance(r, dict)]
        except Exception:
            shows_rules = []

        dedup_policies = [
            {"id": k, **v} for k, v in DEDUP_POLICY_PARAMS.items()
        ]

        return {
            "models": models_list,
            "shows_yaml": shows_text,
            "shows_rules": shows_rules,
            "dedup_policies": dedup_policies,
        }

    def _validate_shows_yaml_text(yaml_content: str) -> List[Dict[str, Any]]:
        try:
            parsed = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            raise ValueError(f"Lỗi cú pháp YAML trong shows.yaml: {exc}") from exc

        if parsed is None:
            parsed = {"shows": []}
        if not isinstance(parsed, dict) or "shows" not in parsed:
            raise ValueError("Cấu trúc shows.yaml phải chứa khóa gốc 'shows:' dạng danh sách.")

        rules = parsed.get("shows")
        if rules is None:
            return []
        if not isinstance(rules, list):
            raise ValueError("Trường 'shows' phải là một danh sách các rule '- pattern: ...'.")

        models_file = root / "models" / "models.yaml"
        valid_models: set[str] = set()
        if models_file.is_file():
            raw_m = yaml.safe_load(models_file.read_text(encoding="utf-8")) or {}
            valid_models = {str(k).lower() for k in (raw_m.get("models") or {}).keys()}

        validated: List[Dict[str, Any]] = []
        for idx, item in enumerate(rules, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Rule thứ #{idx} không hợp lệ (phải là dictionary).")
            pat = str(item.get("pattern") or "").strip()
            if not pat:
                raise ValueError(f"Rule thứ #{idx} thiếu trường 'pattern'.")
            mod = str(item.get("model") or "").strip()
            if mod and valid_models and mod.lower() not in valid_models:
                raise ValueError(
                    f"Rule '{pat}' trỏ tới model '{mod}' không tồn tại trong models/models.yaml."
                )
            pol = str(item.get("dedup_policy") or "").strip()
            if pol and pol.lower() not in DEDUP_POLICY_PARAMS:
                raise ValueError(
                    f"Rule '{pat}' dùng dedup_policy '{pol}' không hợp lệ "
                    f"(chấp nhận: {', '.join(DEDUP_POLICY_PARAMS.keys())})."
                )
            validated.append(item)
        return validated

    @app.post("/api/config/test-match")
    def api_test_shows_match(req: TestShowsMatchRequest) -> Dict[str, Any]:
        sample = (req.sample_path or "").strip()
        if not sample:
            raise HTTPException(status_code=400, detail="Vui lòng nhập tên file hoặc đường dẫn mẫu để kiểm thử.")

        if req.yaml_content is not None:
            try:
                rules = _validate_shows_yaml_text(req.yaml_content)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            models_file = root / "models" / "models.yaml"
            reg: Dict[str, Any] = {}
            default_id = "animevideov3"
            if models_file.is_file():
                raw_m = yaml.safe_load(models_file.read_text(encoding="utf-8")) or {}
                reg = raw_m.get("models") or {}
                for k, v in reg.items():
                    if isinstance(v, dict) and v.get("default"):
                        default_id = str(k)
                        break
            lower_to_key = {str(k).lower(): str(k) for k in reg.keys()}

            vid_name = os.path.basename(sample.rstrip("\\/"))
            parent_dir = os.path.basename(os.path.dirname(sample)) if ("/" in sample or "\\" in sample) else ""
            matched_pat = ""
            chosen_mod = default_id
            chosen_pol = "default"
            src_desc = "Mặc định hệ thống (models.yaml)"

            for r in rules:
                pat = str(r.get("pattern") or "").strip()
                if match_show_pattern(vid_name, pat) or (parent_dir and match_show_pattern(parent_dir, pat)):
                    matched_pat = pat
                    m_target = str(r.get("model") or "").strip().lower()
                    if m_target in lower_to_key:
                        chosen_mod = lower_to_key[m_target]
                        src_desc = f"shows.yaml ('{pat}')"
                    p_target = str(r.get("dedup_policy") or "").strip().lower()
                    if p_target in DEDUP_POLICY_PARAMS:
                        chosen_pol = p_target
                    break

            med_fps, _ = MODEL_REFERENCE_FPS.get(chosen_mod, (10.5, 8.5))
            return {
                "sample": sample,
                "model_id": chosen_mod,
                "model_name": str((reg.get(chosen_mod) or {}).get("name") or chosen_mod),
                "dedup_policy": chosen_pol,
                "matched_pattern": matched_pat,
                "resolution_source": src_desc,
                "ref_median_fps": med_fps,
            }

        res = fs_browser.resolve_routing_for_path(sample)
        med_fps, _ = MODEL_REFERENCE_FPS.get(res["model_id"], (10.5, 8.5))
        return {"sample": sample, **res, "ref_median_fps": med_fps}

    @app.put("/api/config/shows")
    def api_save_shows(req: SaveShowsYamlRequest) -> Dict[str, Any]:
        try:
            rules = _validate_shows_yaml_text(req.yaml_content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        shows_path = root / "shows.yaml"
        tmp_path = root / f".shows.yaml.{os.getpid()}.tmp"
        text = req.yaml_content.replace("\r\n", "\n").rstrip("\n") + "\n"
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, shows_path)
        return {"ok": True, "rules_count": len(rules)}

    # -----------------------------------------------------------------
    # 5. SUBTRANS STUDIO & AUDIT VAULT
    # -----------------------------------------------------------------
    @app.get("/api/subtrans/overview")
    def api_subtrans_overview() -> Dict[str, Any]:
        now_pt = get_now_pt()
        next_pt, next_vn = get_next_pt_reset(now_pt)
        today_pt_str = now_pt.strftime("%Y-%m-%d")
        reset_seconds = max(0, int((next_pt - now_pt).total_seconds()))

        # Kiểm tra trạng thái các khe API Key (chỉ trả về dạng đã che, tuyệt đối không lộ khóa thật)
        dotenv_map = _read_dotenv_map(root / ".env")
        allow_proc_env = os.environ.get("SUBTRANS_DISABLE_DOTENV") != "1"

        def _slot_val(slot_key: str, fallback_primary: bool = False) -> str:
            if slot_key in dotenv_map and dotenv_map[slot_key]:
                return dotenv_map[slot_key]
            if fallback_primary and dotenv_map.get("GEMINI_API_KEY"):
                return dotenv_map["GEMINI_API_KEY"]
            if allow_proc_env:
                if os.environ.get(slot_key):
                    return os.environ[slot_key]
                if fallback_primary and os.environ.get("GEMINI_API_KEY"):
                    return os.environ["GEMINI_API_KEY"]
            return ""

        key_slots: List[Dict[str, Any]] = []
        for idx, env_name in enumerate(
            ("GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"), start=1
        ):
            raw_v = _slot_val(env_name, fallback_primary=(idx == 1))
            key_slots.append({
                "slot": idx,
                "env_name": env_name,
                "configured": bool(raw_v),
                "masked": _mask_secret(raw_v),
                "key_id": compute_key_id(raw_v) if raw_v else "none",
            })

        # Đọc subtrans/models.yaml
        sub_models_path = root / "subtrans" / "models.yaml"
        sub_models: List[Dict[str, Any]] = []
        chain: List[str] = []
        if sub_models_path.is_file():
            try:
                sm_raw = yaml.safe_load(sub_models_path.read_text(encoding="utf-8")) or {}
                chain = [str(x) for x in (sm_raw.get("chain") or [])]
                for m_name, m_cfg in (sm_raw.get("models") or {}).items():
                    if isinstance(m_cfg, dict):
                        sub_models.append({
                            "model": str(m_name),
                            "rpm": m_cfg.get("rpm", 0),
                            "rpd": m_cfg.get("rpd", 0),
                            "tpm": m_cfg.get("tpm", 0),
                            "status": str(m_cfg.get("status") or "active"),
                        })
            except Exception:
                pass

        # Đọc thống kê SQLite ở chế độ CHỈ ĐỌC (không bao giờ tạo/ghi vào DB production khi GET)
        cache_stats: Dict[str, Any] = {
            "db_exists": False,
            "total_cached_cues": 0,
            "today_cached_cues": 0,
            "today_requests": [],
            "exhausted_keys": [],
            "exhausted_models": [],
        }
        db_file = root / "subtrans" / ".cache" / "translations.db"
        if db_file.is_file():
            cache_stats["db_exists"] = True
            try:
                ro_uri = f"file:{db_file.as_posix()}?mode=ro"
                conn = sqlite3.connect(ro_uri, uri=True, timeout=3.0)
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM translations")
                    cache_stats["total_cached_cues"] = int((cur.fetchone() or [0])[0])

                    cur.execute("SELECT COUNT(*) FROM translations WHERE pt_date = ?", (today_pt_str,))
                    cache_stats["today_cached_cues"] = int((cur.fetchone() or [0])[0])

                    cur.execute(
                        """
                        SELECT model, status, COUNT(*), COALESCE(SUM(cues_count), 0), COALESCE(SUM(total_tokens), 0)
                        FROM api_requests
                        WHERE pt_date = ?
                        GROUP BY model, status
                        ORDER BY COUNT(*) DESC
                        """,
                        (today_pt_str,),
                    )
                    cache_stats["today_requests"] = [
                        {
                            "model": str(r[0] or ""),
                            "status": str(r[1] or ""),
                            "requests": int(r[2] or 0),
                            "cues": int(r[3] or 0),
                            "tokens": int(r[4] or 0),
                        }
                        for r in cur.fetchall()
                    ]

                    cur.execute(
                        "SELECT key_id, model, reason FROM key_exhaustion WHERE pt_date = ?",
                        (today_pt_str,),
                    )
                    cache_stats["exhausted_keys"] = [
                        {"key_id": str(r[0]), "model": str(r[1]), "reason": str(r[2] or "")}
                        for r in cur.fetchall()
                    ]

                    cur.execute("SELECT model, exhausted_until_pt, reason FROM model_exhaustion")
                    cache_stats["exhausted_models"] = [
                        {"model": str(r[0]), "until_pt": str(r[1] or ""), "reason": str(r[2] or "")}
                        for r in cur.fetchall()
                    ]
                finally:
                    conn.close()
            except Exception:
                pass

        # Danh sách glossaries/*.yaml
        glossaries_dir = root / "glossaries"
        glossaries: List[Dict[str, str]] = []
        if glossaries_dir.is_dir():
            for gf in sorted(glossaries_dir.glob("*.yaml")):
                glossaries.append({"name": gf.name, "path": str(gf.resolve())})

        # Danh sách pending_subs.json
        pending_items: List[Dict[str, Any]] = []
        for p_cand in (root / "out" / "pending_subs.json", root / "pending_subs.json"):
            if p_cand.is_file():
                try:
                    p_raw = json.loads(p_cand.read_text(encoding="utf-8"))
                    if isinstance(p_raw, list):
                        pending_items = [x for x in p_raw if isinstance(x, dict)]
                    elif isinstance(p_raw, dict) and isinstance(p_raw.get("items"), list):
                        pending_items = [x for x in p_raw["items"] if isinstance(x, dict)]
                    break
                except Exception:
                    pass

        return {
            "pt_date": today_pt_str,
            "next_reset_vn": next_vn.strftime("%Y-%m-%d %H:%M:%S"),
            "reset_in_seconds": reset_seconds,
            "key_slots": key_slots,
            "configured_keys_count": sum(1 for s in key_slots if s["configured"]),
            "models": sub_models,
            "chain": chain,
            "cache": cache_stats,
            "glossaries": glossaries,
            "pending_subs": pending_items,
        }

    @app.put("/api/subtrans/api-keys")
    def api_update_subtrans_keys(req: UpdateApiKeysRequest) -> Dict[str, Any]:
        """Cập nhật các khe GEMINI_API_KEY_1..3 vào .env cục bộ (không bao giờ trả hay log khóa thô)."""
        updates: Dict[str, Optional[str]] = {
            "GEMINI_API_KEY_1": req.key_1,
            "GEMINI_API_KEY_2": req.key_2,
            "GEMINI_API_KEY_3": req.key_3,
        }
        # Nếu người dùng đặt slot 1, đồng bộ cả GEMINI_API_KEY chính nếu chưa có
        dotenv_path = root / ".env"
        existing = _read_dotenv_map(dotenv_path)
        if req.key_1 is not None and req.key_1.strip() and not existing.get("GEMINI_API_KEY"):
            updates["GEMINI_API_KEY"] = req.key_1.strip()

        _write_dotenv_keys(dotenv_path, updates)

        # Chỉ cập nhật os.environ khi không chạy trong môi trường cô lập cấm dotenv của pytest
        if os.environ.get("SUBTRANS_DISABLE_DOTENV") != "1":
            for env_k, raw_v in updates.items():
                if raw_v is None:
                    continue
                clean_v = raw_v.strip()
                if clean_v:
                    os.environ[env_k] = clean_v
                else:
                    os.environ.pop(env_k, None)

        refreshed = _read_dotenv_map(dotenv_path)
        slots_summary = []
        for idx, env_name in enumerate(
            ("GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"), start=1
        ):
            val = refreshed.get(env_name) or (refreshed.get("GEMINI_API_KEY") if idx == 1 else "") or ""
            slots_summary.append({
                "slot": idx,
                "env_name": env_name,
                "configured": bool(val),
                "masked": _mask_secret(val),
                "key_id": compute_key_id(val) if val else "none",
            })

        return {"ok": True, "key_slots": slots_summary}

    @app.get("/api/audits")
    def api_list_audits() -> Dict[str, Any]:
        out_dir = root / "out"
        audits: List[Dict[str, Any]] = []
        discovered_files: List[Path] = []
        seen_keys: set[str] = set()

        for search_dir in _iter_out_search_dirs(out_dir):
            for af in search_dir.glob("*.audit.json"):
                dedup_key = (
                    af.name.lower()
                    if search_dir.resolve() in {out_dir.resolve(), (out_dir / "audit").resolve()}
                    else f"{search_dir.name.lower()}/{af.name.lower()}"
                )
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                discovered_files.append(af)

        discovered_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for af in discovered_files[:60]:
            try:
                adata = json.loads(af.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(adata, dict):
                continue

            stem_4k = af.name[:-len(".audit.json")]
            stem_src = re.sub(r"_4K(?:_animejanai)?$", "", stem_4k, flags=re.IGNORECASE)

            sub_audit_md = ""
            cast_yaml = ""
            doc_dirs = [af.parent, out_dir, out_dir / "audit", out_dir / "cast"]
            for doc_dir in doc_dirs:
                if not doc_dir.is_dir():
                    continue
                for cand_stem in (stem_4k, stem_src, f"{stem_src}.vi", f"{stem_4k}.vi"):
                    md_p = doc_dir / f"{cand_stem}.audit.md"
                    if not sub_audit_md and md_p.is_file():
                        sub_audit_md = str(md_p.resolve())
                    cy_p = doc_dir / f"{cand_stem}.cast.yaml"
                    if not cast_yaml and cy_p.is_file():
                        cast_yaml = str(cy_p.resolve())

            mux_info = adata.get("mux_audit") if isinstance(adata.get("mux_audit"), dict) else {}
            sanity_info = adata.get("output_sanity") if isinstance(adata.get("output_sanity"), dict) else {}
            vfc = (
                mux_info.get("video_frame_count")
                if isinstance(mux_info.get("video_frame_count"), dict)
                else {}
            )
            src_frames = int(adata.get("source_frames") or vfc.get("source_frames") or 0)
            out_frames = int(adata.get("output_frames") or vfc.get("output_frames") or 0)
            if "frames_matched" in adata and (adata.get("source_frames") or adata.get("output_frames")):
                frames_matched = bool(adata.get("frames_matched"))
            elif "passed" in vfc and (src_frames or out_frames):
                frames_matched = bool(vfc.get("passed"))
            else:
                frames_matched = bool(src_frames > 0 and src_frames == out_frames)

            aud_streams = mux_info.get("audio_streams") if isinstance(mux_info.get("audio_streams"), dict) else {}
            sub_streams = (
                mux_info.get("subtitle_streams") if isinstance(mux_info.get("subtitle_streams"), dict) else {}
            )
            font_att = mux_info.get("font_attachments") if isinstance(mux_info.get("font_attachments"), dict) else {}

            audio_count = int(adata.get("audio_tracks_count") or aud_streams.get("output_count") or 0)
            sub_count = int(adata.get("subtitle_tracks_count") or sub_streams.get("output_count") or 0)
            fonts_count = int(adata.get("fonts_count") or font_att.get("output_count") or 0)

            episode_label = (
                f"{af.parent.name}/{stem_4k}"
                if af.parent.resolve() not in {out_dir.resolve(), (out_dir / "audit").resolve()}
                else stem_4k
            )

            audits.append({
                "episode": episode_label,
                "audit_json_path": str(af.resolve()),
                "output_file": str(adata.get("output_file") or ""),
                "source_file": str(adata.get("source_file") or ""),
                "status": str(adata.get("status") or mux_info.get("status") or "UNKNOWN"),
                "sanity_status": str(sanity_info.get("status") or "N/A"),
                "model_id": str(adata.get("model_id") or ""),
                "average_pipeline_fps": adata.get("average_pipeline_fps", 0.0),
                "ai_steady_fps": adata.get("ai_steady_fps", 0.0),
                "source_frames": src_frames,
                "output_frames": out_frames,
                "frames_matched": frames_matched,
                "audio_tracks_count": audio_count,
                "subtitle_tracks_count": sub_count,
                "fonts_count": fonts_count,
                "output_file_size_mb": adata.get("output_file_size_mb", 0.0),
                "completed_at": str(adata.get("completed_at") or ""),
                "sub_audit_md_path": sub_audit_md,
                "cast_yaml_path": cast_yaml,
            })

        return {
            "audits": audits,
            "checkpoints": scan_temp_checkpoints(root),
        }

    @app.get("/api/audits/detail")
    def api_audit_detail(path: str = Query(...)) -> Dict[str, Any]:
        cand = Path(os.path.expanduser(path.strip()))
        if not cand.is_absolute():
            cand = (root / cand).resolve()
        else:
            cand = cand.resolve()

        # Chỉ cho phép đọc các báo cáo kiểm định nằm bên trong workspace_root
        try:
            cand.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Chỉ cho phép xem báo cáo kiểm định thuộc dự án.") from exc

        low_name = cand.name.lower()
        if not (
            low_name.endswith(".audit.json")
            or low_name.endswith(".audit.md")
            or low_name.endswith(".cast.yaml")
        ):
            raise HTTPException(status_code=400, detail="Loại tệp báo cáo không hợp lệ.")
        if not cand.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy tệp báo cáo kiểm định.")

        raw_text = cand.read_text(encoding="utf-8", errors="replace")
        if low_name.endswith(".audit.json"):
            return {"type": "json", "path": str(cand), "data": json.loads(raw_text)}
        return {"type": "text", "path": str(cand), "content": raw_text}

    return app
