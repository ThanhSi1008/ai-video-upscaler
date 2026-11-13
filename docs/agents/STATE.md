# Trạng Thái Hệ Thống & Ngữ Cảnh Hoạt Động (STATE)

> **Cập nhật ngày:** 2026-11-13
>
> **Commit Baseline:** `d258b19` (`perf(pipeline): khoa so kiem toan cuoi 3-Stage Overlap, GOP-RLE Queue Preflight va TensorRT Zero-Alloc Ring`)
>
> **Tài liệu liên quan:** [START_HERE](START_HERE.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md) | [Root AGENTS](../../AGENTS.md) | [Subtrans AGENTS](../../subtrans/AGENTS.md)

---

## 1. Trạng Thái Hiện Tại Của Codebase

- **Bộ kiểm thử tự động:** Đạt **PASS 100%** (`319 passed, 11 skipped, 1 deselected`, `0 errors` với `ruff check`, `6/6` ca hồi quy & `22/22` cues benchmark trong `subtrans.py eval`) qua runner cô lập `scripts/verify_offline.py` trên Windows 11.
- **Trạng thái Git:** Sạch sẽ, kỷ luật commit riêng biệt cho từng yêu cầu kỹ thuật; đã trang bị đầy đủ `.agents/skills/`, `.agents/plugins/`, `.vscode/` và `ruff.toml`.

---

## 2. Danh Mục Các Thành Phần Đã Kiểm Chứng (Verified & Production Ready)

| Phân hệ | Module / File | Trạng thái kiểm chứng | Chi tiết kỹ thuật |
| :--- | :--- | :--- | :--- |
| **PyTorch CUDA / TensorRT 10 FP16** | `src/cuda_engine.py`, `src/build_trt_engine.py`, `src/config.py` | PASS | `CUDA_MODULE_LOADING=LAZY`, `CUDA_CACHE_MAXSIZE=2GB`, TensorRT 10.16 (`builder_optimization_level=5`, `_out_ring` 2-slot 0-alloc, shape/address cache, `execute_async_v3`), kế thừa `UPSCALER_HEVC_ENCODER`. |
| **Zero-Disk Pipe & 3-Stage Overlap** | `src/trt_zero_copy_upscaler.py`, `src/pipeline.py`, `src/ffmpeg_ops.py` | PASS | Gối đầu 3 tầng thực sự (`Decode N+1` + `GPU N` + `worker_finalize_zero_disk`), `uint8` NCHW transpose giảm 50% băng thông VRAM, Belady's OPT Cyclic Memoization + 0-reinference Sanity (`upscale_turbo.bat`). |
| **Dedup, GOP-RLE & Queue Preflight** | `src/dedup.py`, `src/scheduler.py`, `src/preflight.py` | PASS | `deduplicate_cyclic_frames` (`A->B->A`), `raw_actions_rle` tái phân hoạch GOP `< 1 ms`, `run_preflight_worker_cli` tách sẵn `raw_segs` + GOP trong `--queue`, 1-Pass fused `mpdecimate+gray`, Early-Exit DIS Flow. |
| **10-bit NVENC BT.709** | `src/ffmpeg_ops.py`, `src/config.py` | PASS | `hevc_nvenc` Main 10 (`p010le`, preset `p6`, `CQ 18`, `maxrate 45M`, multipass `fullres`, lookahead `32`, Spatial/Temporal AQ), ma trận màu BT.709 hai chiều Full $\leftrightarrow$ Limited. |
| **Bit-exact 1-Pass Mux & Parallel Audit** | `src/ffmpeg_ops.py`, `src/mux_audit.py` | PASS | `mux_segments_directly` (1-Pass Concat+Mux, khử 8 GB `merged_4k.mp4`), `-max_interleave_delta 0`, 1-Pass `-f streamhash` MD5 và `ThreadPoolExecutor` audit đa luồng. |
| **Output Sanity & PID GPU Lock** | `src/output_sanity.py`, `src/config.py` | PASS | Kiểm tra sau chunk 0 và mỗi 10 chunks ($|\Delta Y| \le 6/255$, corr $\ge 0.98$, 0 infer lặp trên Zero-Disk); khóa `.gpu_active.lock` kiểm tra PID sống. |
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
