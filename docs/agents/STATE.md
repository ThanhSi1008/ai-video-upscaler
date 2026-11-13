# Trạng Thái Hệ Thống & Ngữ Cảnh Hoạt Động (STATE)

> **Cập nhật ngày:** 2026-11-13
>
> **Commit Baseline:** `a496ab7` (`feat(subtrans): trien khai toan dien lo trinh 6 goi nang cap toi da chat luong dich phu de AI V2`)
>
> **Tài liệu liên quan:** [START_HERE](START_HERE.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md) | [Root AGENTS](../../AGENTS.md) | [Subtrans AGENTS](../../subtrans/AGENTS.md)

---

## 1. Trạng Thái Hiện Tại Của Codebase

- **Bộ kiểm thử tự động:** Đạt **PASS 100%** (`301 passed, 11 skipped, 1 deselected`, `0 errors` với `ruff check`, `6/6` ca hồi quy & `22/22` cues benchmark trong `subtrans.py eval`) qua runner cô lập `scripts/verify_offline.py` trên Windows 11.
- **Trạng thái Git:** Sạch sẽ, kỷ luật commit riêng biệt cho từng yêu cầu kỹ thuật; đã trang bị đầy đủ `.agents/skills/`, `.agents/plugins/`, `.vscode/` và `ruff.toml`.

---

## 2. Danh Mục Các Thành Phần Đã Kiểm Chứng (Verified & Production Ready)

| Phân hệ | Module / File | Trạng thái kiểm chứng | Chi tiết kỹ thuật |
| :--- | :--- | :--- | :--- |
| **PyTorch CUDA / TensorRT 10 FP16** | `src/cuda_engine.py`, `src/build_trt_engine.py`, `src/config.py` | PASS | `CUDA_MODULE_LOADING=LAZY`, `CUDA_CACHE_MAXSIZE=2GB` trên NVMe Gen4, `configure_cuda_fast_math` (`HMMA.F16` 51 TFLOPS), TensorRT 10.16 (`builder_optimization_level=5`, `TimingCache`, `execute_async_v3`), Ping-Pong Pinned D2H 0-copy RAM. |
| **Zero-Disk Pipe & Gen4 NVMe Buffer** | `src/trt_zero_copy_upscaler.py`, `src/ffmpeg_ops.py`, `src/ramdisk.py` | PASS | Giải mã `extract_frames_raw_rgb24` (`-hwaccel cuda` BT.709 pipe) -> NHWC 0-transpose H2D -> TensorRT 10 FP16 -> Triton/`addcmul_` `p010le` pipe -> NVENC (`upscale_turbo.bat`); đệm cuốn chiếu trên NVMe Gen4 `WD_BLACK SN7100`. |
| **Dedup, DIS Flow & Alder Lake P-Core** | `src/dedup.py`, `src/scheduler.py`, `src/preflight.py` | PASS | Split theo GOP, `mpdecimate` (64KB pipe buffer), DIS optical flow tiền tính toán `1/r^2` & `mag_sq` (0 `hypot`), `configure_alder_lake_pcore_qos` giữ P-Cores `i5-12600K` + `unlock_rtx3060_admin.ps1`. |
| **10-bit NVENC BT.709** | `src/ffmpeg_ops.py` | PASS | `hevc_nvenc` Main 10 (`p010le`, preset `p6`, multipass `fullres`/`qres`), ma trận màu BT.709 hai chiều Full $\leftrightarrow$ Limited. |
| **Bit-exact Mux & Audit** | `src/ffmpeg_ops.py`, `src/mux_audit.py` | PASS | Bắt buộc cờ `-max_interleave_delta 0`, audit MD5 âm thanh, font đính kèm và số lượng frame. |
| **Output Sanity & PID GPU Lock** | `src/output_sanity.py`, `src/config.py` | PASS | Kiểm tra sau chunk 0 và mỗi 10 chunks ($|\Delta Y| \le 6/255$, corr $\ge 0.98$); khóa `.gpu_active.lock` kiểm tra PID sống (`psutil.pid_exists`) chống xóa nhầm khi import. |
| **Gemini 3-Key Subtrans V2 (36 RPM)** | `subtrans/gemini_backend.py`, `subtrans/pipeline.py`, `subtrans/cast_sheet.py` | PASS | Xoay vòng 3-Key modulo + ít tải nhất + đổi key khi `429 PerMinute`, theo dõi cạn RPD theo `(model, key_hash)`, tiền tố tĩnh `>4.096` tokens Implicit Context Caching (`cached_tokens`), `ThinkingConfig(MINIMAL/LOW)`, CoT `p1`/`p2` tường minh trong `HysteresisCastManager` và cắt ngữ cảnh tại điểm chuyển cảnh (`>=4500ms`). |
| **Contextual Cache (`v2.2`) & Glossary** | `subtrans/cache.py`, `subtrans/glossary.py`, `subtrans/ja_morph_tagger.py` | PASS | Khóa cache ngữ cảnh `compute_contextual_cue_sig`, `inline_cached_context` khi partial hit, ghi ngược `Cascade` vào cache, deep-merge `AddressRules`, `strip_ja_furigana` xuyên suốt, nhận diện Kansai-ben & `soregashi`. |
| **Smart Cascade, Linter V2 & Hybrid RAG** | `subtrans/cascade.py`, `subtrans/linter.py`, `subtrans/rag.py` | PASS | Ranh giới từ Katakana chống báo giả, `Character.aliases`, `RULE_GLOSSARY_TERM` & `RULE_PRONOUN_CONSISTENCY`, cách ly lệch con số trong Cascade, Hybrid BM25 RAG + tự động tuyển chọn mẫu `silver`, `22/22` cues & `6/6` ca hồi quy trong `eval/runner.py`. |
| **Faster-Whisper, DTW 1:2 & Piecewise** | `src/subtitles.py`, `subtrans/parser.py`, `subtrans/dtw_auto_align.py` | PASS | Hỗ trợ `faster-whisper` + Silero VAD + `initial_prompt` từ Glossary, lọc âm thanh CC `(筆記音)`, gắn `mode_hint="inner"` cho `＜...＞`/`Italics`, tách `1:2` theo khoảng lặng `>=250ms` và `Piecewise 2-Segment Auto-Offset`. |
| **Playback Writer, Async Stage B & Queue**| `subtrans/writer.py`, `src/pending_queue.py`, `upscale.py` | PASS | Bảo toàn `<Rudeus!>`, cặp nháy giữa câu, lời gọi cảm xúc lặp lại & Typesetting `.ass`; tách `src/pending_queue.py` (không nạp `torch`), cô lập `PYTHON_DOTENV_DISABLED`, hỗ trợ `--async-subs` & `--target-4k`. |
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
