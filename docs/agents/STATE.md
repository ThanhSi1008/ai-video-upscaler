# Trạng Thái Hệ Thống & Ngữ Cảnh Hoạt Động (STATE)

> **Cập nhật ngày:** 2026-11-14
>
> **Commit Baseline:** `101aced` (`docs(benchmarks): compile comprehensive final optimization report and safety audit`)
>
> **Tài liệu liên quan:** [START_HERE](START_HERE.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md) | [Root AGENTS](../../AGENTS.md) | [Subtrans AGENTS](../../subtrans/AGENTS.md)

---

## 1. Trạng Thái Hiện Tại Của Codebase

- **Bộ kiểm thử tự động:** Đạt **PASS 100%** (`488 passed, 11 skipped, 1 deselected`, bao gồm `115/115` ca kiểm thử hộp đen `tests/e2e/` và `30/30` ca `tests/test_benchmark_infra.py`, `0 errors` với `ruff check`, `6/6` ca hồi quy & `22/22` cues benchmark trong `subtrans.py eval`) qua runner cô lập `scripts/verify_offline.py` trên Windows 11.
- **Thực nghiệm phần cứng & Bão hòa Pareto:** Đã đo đạc đối chứng trên mẫu đại cảnh **Fate/Apocrypha Tập 22** (`H.264 Hi10P`, `95.06%` unique frames) trên RTX 3060 12GB + i5-12600K: **Preview `3.61 -> 8.05 fps` (`+122.99%`)**, **Climax `8.53 -> 9.47 fps` (`+11.02%`, AI Steady-State `10.20 fps` = `92.85%` trần vật lý GA106)**, `Mux Audit PASS` bit-exact (`benchmarks/final_optimization_report.md`).
- **Trạng thái Git:** Sạch sẽ, kỷ luật commit riêng biệt cho từng yêu cầu kỹ thuật; đã trang bị đầy đủ `.agents/skills/`, `.agents/plugins/`, `.vscode/` và `ruff.toml`.

---

## 2. Danh Mục Các Thành Phần Đã Kiểm Chứng (Verified & Production Ready)

| Phân hệ | Module / File | Trạng thái kiểm chứng | Chi tiết kỹ thuật |
| :--- | :--- | :--- | :--- |
| **PyTorch CUDA / TensorRT 10 FP16** | `src/cuda_engine.py`, `src/build_trt_engine.py`, `src/config.py` | PASS | `CUDA_MODULE_LOADING=LAZY`, `CUDA_CACHE_MAXSIZE=2GB`, TensorRT 10.16 (`resolve_trt_profile_max_batch`, `_out_ring` 2-slot 0-alloc, `execute_async_v3`), kế thừa `UPSCALER_HEVC_ENCODER`. |
| **Zero-Disk Pipe & 3-Stage Overlap** | `src/trt_zero_copy_upscaler.py`, `src/pipeline.py`, `src/ffmpeg_ops.py` | PASS | Gối đầu 3 tầng thực sự, `_PipeDrainWorker` bất đồng bộ `8 MiB` zero-copy + chống deadlock, `CUDA Event` micro-timings phi phong tỏa, bù `Open-GOP CRA discard` & `verify_segment_resume_frames`. |
| **Dedup, GOP-RLE & Sakuga DIS** | `src/dedup.py`, `src/scheduler.py`, `src/preflight.py` | PASS | Băm `Cyclic Cel` 3 tầng (`64x36` thumbnail + BLAKE2b + SIMD uint8, tăng tốc `27.02x`), `raw_actions_rle`, phân tách ngưỡng `DIS` toàn cục (`0.35px`) / cục bộ (`1.50px` @ `12%`) cho policy `sakuga`/`dis-sakuga`. |
| **Benchmark, Telemetry & E2E Suite** | `scripts/telemetry_collector.py`, `scripts/benchmark_pipeline.py`, `scripts/compare_benchmarks.py`, `tests/e2e/` | PASS | Thu thập 17 chỉ số phần cứng chu kỳ `1.0s`, đối chiếu delta tự động, bộ 115 ca kiểm thử hộp đen 4 tầng (`tests/e2e/`) bảo vệ toàn bộ bất biến kỹ thuật. |
| **10-bit NVENC BT.709** | `src/ffmpeg_ops.py`, `src/config.py` | PASS | `hevc_nvenc` Main 10 (`p010le`, preset `p6`, `CQ 18`, `maxrate 45M`, multipass `fullres`, lookahead `32`, Spatial/Temporal AQ), ma trận màu BT.709 hai chiều Full $\leftrightarrow$ Limited. |
| **Bit-exact 1-Pass Mux & Parallel Audit** | `src/ffmpeg_ops.py`, `src/mux_audit.py` | PASS | `mux_segments_directly` (1-Pass Concat+Mux, khử 8 GB `merged_4k.mp4`), `-max_interleave_delta 0`, 1-Pass `-f streamhash` MD5 và `ThreadPoolExecutor` audit đa luồng. |
| **Output Sanity & PID GPU Lock** | `src/output_sanity.py`, `src/config.py` | PASS | Kiểm tra sau chunk 0 và mỗi 10 chunks ($|\Delta Y| \le 6/255$, corr $\ge 0.98$, 0 infer lặp trên Zero-Disk); khóa `.gpu_active.lock` kiểm tra PID sống. |
| **Gemini 3-Key Subtrans V2.5 (36 RPM)** | `subtrans/gemini_backend.py`, `subtrans/pipeline.py`, `subtrans/audio_diarization.py` | PASS | Xoay vòng 3-Key modulo + ít tải nhất + đổi key `429 PerMinute`, tiền tố tĩnh `>4.096` tokens Implicit & `--explicit-cache`, `audio_diarization.py` (5.1 FC 8kHz CMS timbre + F0 `audio_turn`), CoT `p1`/`p2` trong `HysteresisCastManager` và `StageAPrefetchWorker`. |
| **Contextual Cache (`v2.2`), MeCab & Glossary** | `subtrans/cache.py`, `subtrans/glossary.py`, `subtrans/ja_morph_tagger.py` | PASS | Khóa cache ngữ cảnh `compute_contextual_cue_sig`, ghi ngược `Cascade`, tách từ hình thái học MeCab `fugashi` + `unidic-lite` (0 báo giả từ ghép Kanji, phân biệt `君` vs `-君`), tự dò Glossary theo thư mục cha (`frieren`, `violet`, `jujutsu`, `kimetsu`, `rezero`). |
| **Smart Cascade, Linter V2 & Hybrid RAG** | `subtrans/cascade.py`, `subtrans/linter.py`, `subtrans/rag.py` | PASS | Tiền tố tĩnh `>4.096` tokens cho Cascade, `RULE_GLOSSARY_TERM` & `RULE_PRONOUN_CONSISTENCY`, cách ly lệch con số, `KnowledgeBase` 1 lần/tập + `BUILTIN_GOLD_EXEMPLARS` (`rag seed`), `22/22` cues & `6/6` ca hồi quy PASS. |
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
