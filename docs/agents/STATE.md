# Trạng Thái Hệ Thống & Ngữ Cảnh Hoạt Động (STATE)

> **Cập nhật ngày:** 2026-11-13
>
> **Commit Baseline:** `a535105` (`feat(agents): thiet lap ha tang AI Agents, plugins va quy chuan IDE toan dien`)
>
> **Tài liệu liên quan:** [START_HERE](START_HERE.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md) | [Root AGENTS](../../AGENTS.md) | [Subtrans AGENTS](../../subtrans/AGENTS.md)

---

## 1. Trạng Thái Hiện Tại Của Codebase

- **Bộ kiểm thử tự động:** Đạt **PASS 100%** (`265 passed, 11 skipped, 1 deselected`, `0 errors` với `ruff check`) qua runner cô lập `scripts/verify_offline.py` trên Windows 11.
- **Trạng thái Git:** Sạch sẽ, kỷ luật commit riêng biệt cho từng yêu cầu kỹ thuật; đã trang bị đầy đủ `.agents/skills/`, `.agents/plugins/`, `.vscode/` và `ruff.toml`.

---

## 2. Danh Mục Các Thành Phần Đã Kiểm Chứng (Verified & Production Ready)

| Phân hệ | Module / File | Trạng thái kiểm chứng | Chi tiết kỹ thuật |
| :--- | :--- | :--- | :--- |
| **PyTorch CUDA FP16 & Pinned I/O** | `src/cuda_engine.py`, `src/models_resolver.py` | PASS | `RealESRAnimeVideoV3`, `AnimeJaNaiCompact2x` (`janai-fix`), truyền BGR `uint8` qua Pinned Memory + chuẩn hóa FP16 trên GPU (giảm 75% PCIe H2D/D2H), kiểm chứng SHA-256 weights. |
| **Dedup, DIS Flow & Win32 Preflight** | `src/dedup.py`, `src/scheduler.py` | PASS | Split theo GOP, `mpdecimate`, DIS optical flow (`--dis-thresh 0.30`), Preflight nền chuẩn Windows (`BELOW_NORMAL_PRIORITY_CLASS`, truyền tham số qua `sys.argv`). |
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

1. **Zero-Disk Pipe V2 & TensorRT Engine (`src/trt_zero_copy_upscaler.py`, `src/build_trt_engine.py`):**
   - Đã hoàn thiện kiến trúc V2 (hỗ trợ `scale=2.0`, độ phân giải động `ensure_resolution`, tái sử dụng Pinned Buffer xuyên suốt tập, ghi `.tmp.mp4` nguyên tử) và vượt qua unit test CPU; cần người dùng duyệt chạy thử nghiệm thực tế trên GPU trước khi đặt làm mặc định sản xuất.
2. **RAMDisk Context Manager (`src/ramdisk.py`):**
   - Hiện sử dụng `tempfile.TemporaryDirectory` trên SSD/NVMe thông thường với dọn dẹp cuốn chiếu, chưa mount đĩa RAM ảo vật lý trên Windows.
3. **Bảo vệ Gemini Quota ban ngày:**
   - Tuyệt đối không chạy lệnh gọi API Gemini thật trong khoảng từ **14:00 đến đêm chạy**. Mọi kiểm thử chức năng bắt buộc dùng cờ `--dry-run` hoặc mock backend.
4. **Không chạy kiểm thử GPU khi pipeline đang chạy:**
   - Hệ thống chỉ hỗ trợ 1 tiến trình chiếm dụng GPU duy nhất tại một thời điểm.
