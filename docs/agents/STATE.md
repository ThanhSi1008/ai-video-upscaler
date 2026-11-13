# Trạng Thái Hệ Thống & Ngữ Cảnh Hoạt Động (STATE)

> **Cập nhật ngày:** 2026-11-13
>
> **Commit Baseline:** `bb646e2` (`perf(project): trien khai toan dien 3 goi toi uu Pareto cho Video Pipeline va Subtrans`)
>
> **Tài liệu liên quan:** [START_HERE](START_HERE.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md) | [Root AGENTS](../../AGENTS.md) | [Subtrans AGENTS](../../subtrans/AGENTS.md)

---

## 1. Trạng Thái Hiện Tại Của Codebase

- **Bộ kiểm thử tự động:** Đạt **PASS 100%** (`288 passed, 11 skipped, 1 deselected`, `0 errors` với `ruff check`) qua runner cô lập `scripts/verify_offline.py` trên Windows 11.
- **Trạng thái Git:** Sạch sẽ, kỷ luật commit riêng biệt cho từng yêu cầu kỹ thuật; đã trang bị đầy đủ `.agents/skills/`, `.agents/plugins/`, `.vscode/` và `ruff.toml`.

---

## 2. Danh Mục Các Thành Phần Đã Kiểm Chứng (Verified & Production Ready)

| Phân hệ | Module / File | Trạng thái kiểm chứng | Chi tiết kỹ thuật |
| :--- | :--- | :--- | :--- |
| **PyTorch CUDA / TensorRT 10 FP16** | `src/cuda_engine.py`, `src/build_trt_engine.py`, `src/config.py` | PASS | `RealESRAnimeVideoV3` & `AnimeJaNaiCompact2x` (cắt lát 2-frame đuôi 8K tiết kiệm ~4.5 GB VRAM), `configure_cuda_fast_math` (`HMMA.F16` 51 TFLOPS), TensorRT 10.16 (`execute_async_v3` + `_PureFP16TRTExportModule`), Pinned `uint8` H2D & 1-sync D2H. |
| **Zero-Disk Pipe & Gen4 NVMe Buffer** | `src/trt_zero_copy_upscaler.py`, `src/ffmpeg_ops.py`, `src/ramdisk.py` | PASS | Giải mã `extract_frames_raw_rgb24` (`-hwaccel cuda` BT.709 pipe) -> TensorRT/CUDA FP16 -> `p010le` pipe -> NVENC (`upscale_turbo.bat`); tự định tuyến đệm cuốn chiếu về NVMe Gen4 `WD_BLACK SN7100` (`E:\scratch\rolling_buffer`). |
| **Dedup, DIS Flow & Alder Lake P-Core** | `src/dedup.py`, `src/scheduler.py`, `src/preflight.py` | PASS | Split theo GOP, `mpdecimate`, DIS optical flow (`--dis-thresh 0.30`), `configure_alder_lake_pcore_qos` tắt EcoQoS giữ tiến trình trên P-Cores `i5-12600K` + script Admin `scripts/unlock_rtx3060_admin.ps1`. |
| **10-bit NVENC BT.709** | `src/ffmpeg_ops.py` | PASS | `hevc_nvenc` Main 10 (`p010le`, preset `p6`, multipass `fullres`/`qres`), ma trận màu BT.709 hai chiều Full $\leftrightarrow$ Limited. |
| **Bit-exact Mux & Audit** | `src/ffmpeg_ops.py`, `src/mux_audit.py` | PASS | Bắt buộc cờ `-max_interleave_delta 0`, audit MD5 âm thanh, font đính kèm và số lượng frame. |
| **Output Sanity & PID GPU Lock** | `src/output_sanity.py`, `src/config.py` | PASS | Kiểm tra sau chunk 0 và mỗi 10 chunks ($|\Delta Y| \le 6/255$, corr $\ge 0.98$); khóa `.gpu_active.lock` kiểm tra PID sống (`psutil.pid_exists`) chống xóa nhầm khi import. |
| **Gemini 3-Key Subtrans (36 RPM)** | `subtrans/gemini_backend.py`, `subtrans/pipeline.py` | PASS | Luân phiên 3 API key mở khóa trọn vẹn 36 RPM (Lite) / 12 RPM (Flash), compact JSON payload, log RPD chuẩn 1x, hỗ trợ alias `--subs-quality fast`. |
| **SQLite WAL Batch Cache & Glossary** | `subtrans/cache.py`, `subtrans/glossary.py` | PASS | SQLite `WAL` + `get_cues_batch`/`set_cues_batch` kèm CoT metadata, băm SHA-256 nội dung Glossary, giải tham chiếu text-pointer `.yaml` trên Windows, truyền đủ `gender`/`register`/`note`. |
| **Smart Cascade & RAG Series Filter** | `subtrans/cascade.py`, `subtrans/rag.py` | PASS | Format chuẩn `<glossary>` & `<cast>` đầy đủ `speaks_to`, xoay vòng đa key cho Repair Pass, lọc `error_bank` theo đúng `series`, cô lập hoàn toàn `--dry-run`. |
| **Whisper 5.1 FC & Banded DTW** | `src/subtitles.py`, `subtrans/parser.py`, `subtrans/dtw_auto_align.py` | PASS | Trích xuất phụ đề theo `codec_name` (bỏ qua bitmap PGS), lùi CPU khi GPU đang giữ lock, ghi đè chuẩn `c.en` sau DTW + truyền `cast_names_ja_en` + cửa sổ nhanh DTW. |
| **Multi-Speaker & Dual Dialogue**| `subtrans/parser.py`, `subtrans/writer.py` | PASS | Bóc tách người nói vào `Cue.speaker`, Smart Semantic Wrap (>72 chars), bảo toàn timeline Dual Dialogue 2 dòng (`- Lời 1\n- Lời 2`). |
| **Agent Tooling, Hooks & MCP** | `scripts/agent_context.py`, `.agents/` | PASS | AST lookup (`--symbols`, `--find`), kiểm định byte budget & độ mới baseline (`--check`), 4 skills, plugin bảo vệ `upscaler-guard` và MCP read-only `upscaler-mcp`. |
| **Anti-Sleep & Dry-Run Safety** | `src/preflight.py`, `upscale.py` | PASS | Win32 API native `SetThreadExecutionState`, chốt chặn `--dry-run` trước mọi nhánh xóa tạm/gọi API, khôi phục `--subs-pending` cho cả `video_path` và `ja_path`. |

---

## 3. Các Thành Phần Thử Nghiệm / Cần Lưu Ý (Unverified / Experimental)

1. **Zero-Disk Pipe V2 & TensorRT 10.16 Engine (`src/trt_zero_copy_upscaler.py`, `src/build_trt_engine.py`):**
   - Đã cài đặt `tensorrt-cu12==10.16.1.11` + `triton-windows==3.2.0.post21` trong `.venv`, hoàn thiện toàn trình NVDEC raw RGB24 pipe -> TensorRT 10 FP16 (`execute_async_v3`) -> `p010le` pipe -> NVENC (`upscale_turbo.bat` / `--zero-disk`, `--no-tensorrt`) và vượt qua toàn bộ unit test offline; giữ `--zero-disk` mặc định `False` trên CLI cho đến khi người dùng xác nhận trên video thật.
2. **RAMDisk & Bộ Đệm NVMe Gen4 (`src/ramdisk.py`):**
   - Tự dò ổ đĩa RAM vật lý (`UPSCALER_RAMDISK_DIR` hoặc nhãn `"RAM"`), mặc định định tuyến về `E:\ai-video-upscaler\scratch\rolling_buffer` trên NVMe Gen4 `WD_BLACK SN7100` (nhanh gấp ~3 lần ổ hệ thống `C:` Gen3).
3. **Bảo vệ Gemini Quota ban ngày:**
   - Tuyệt đối không chạy lệnh gọi API Gemini thật trong khoảng từ **14:00 đến đêm chạy**. Mọi kiểm thử chức năng bắt buộc dùng cờ `--dry-run` hoặc mock backend.
4. **Không chạy kiểm thử GPU khi pipeline đang chạy:**
   - Hệ thống chỉ hỗ trợ 1 tiến trình chiếm dụng GPU duy nhất tại một thời điểm.
