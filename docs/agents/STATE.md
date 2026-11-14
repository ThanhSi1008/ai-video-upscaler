# Trạng Thái Hệ Thống & Ngữ Cảnh Hoạt Động (STATE)

> **Cập nhật ngày:** 2026-11-15
>
> **Commit Baseline:** `bd33075` (`fix(group-a-b): clamp Sakuga RGB overshoots in P010LE & harden auto-download/Whisper/TRT safety`)
>
> **Tài liệu liên quan:** [START_HERE](START_HERE.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md) | [Root AGENTS](../../AGENTS.md) | [Subtrans AGENTS](../../subtrans/AGENTS.md)

---

## 1. Trạng Thái Hiện Tại Của Codebase

- **Bộ kiểm thử tự động:** Đạt **PASS 100%** (716+ ca kiểm thử tự động không GPU qua runner cô lập `scripts/verify_offline.py` trên Windows 11, bao gồm 169/169 ca E2E Acceptance `tests/test_e2e_*.py`, 141/141 ca `subtrans/tests/`, 30/30 ca `tests/test_benchmark_infra.py`, `2/2` ca đối chiếu độc lập `tests/test_model_regression.py` PSNR >= 50 dB và 115/115 ca `tests/e2e/`).
- **Thực nghiệm phần cứng Nhóm A & Nhóm B trên mẫu đại cảnh `Fate/Apocrypha Tập 22` (`BD 1080p H.264 Hi10P`, `95.06%` unique frames ở Climax `13:30`) trên RTX 3060 12GB + i5-12600K (`--zero-disk` TensorRT 10 FP16):**
  - **Mô hình mặc định `animevideov3` (Hậu Nhóm A):** Preview `3.61 -> 8.05 -> 9.81 fps` (AI Steady `11.14 fps`), Climax `8.53 -> 9.47 fps` (AI Steady `10.20 fps`).
  - **Mô hình Native `2x` sắc nét `animejanai-hd-v3-sharp1-ultracompact` (`64f8c`):** Preview **`13.58 fps`** (AI Steady **`20.18 fps`**, `19.88s`), Climax 60s **`18.27 fps` (`2.14x` vs Baseline / `1.93x` vs `animevideov3` Zero-Disk, AI Steady `19.99 fps`, Peak `20.81 fps`, `80.89s`)**, Peak VRAM **`6.68 GB` (`6838 MiB` < 9.5 GB)**.
  - **Mô hình Native `2x` siêu tốc `animejanai-hd-v3-sharp1-superultracompact` (`24f8c`):** Climax 60s **`21.13 fps` (`2.48x` vs Baseline / `2.23x` vs `animevideov3` Zero-Disk, AI Steady `24.32 fps`, Peak `28.03 fps`, `69.94s`)**, Peak VRAM **`6.07 GB` (`6212 MiB`)**.
  - **Kiểm định chất lượng 12 khung hình Sakuga 4K Main 10 (`out/benchmark_group_b/empirical_quality_audit.json`):** Output Sanity PASS ($|\Delta Y| = 0.32 \le 6.0$, Pearson $r = 0.99992 \ge 0.98$), Cycle PSNR vs BD 1080p tăng **`39.04 -> 39.68..39.78 dB`**, Cycle SSIM tăng **`0.98471 -> 0.98876`**, độ sắc nét nét vẽ Sobel P95 tăng **`+23.17%` (`16.23 -> 19.99`)**, độ trôi màu BT.709 giảm `3.75x`, khớp `1478/1478` frames và `Mux Audit PASS` bit-exact 100% (2 FLAC MD5, 2 ASS, 5 Fonts, 1 Chapter).
- **Trạng thái Git:** Sạch sẽ, hoàn thành trọn vẹn Nhóm A & Nhóm B kèm trọng số Native `2x` chính thức đã xác thực SHA-256 trong `models_bench/` và `models/models.yaml` (giữ nguyên `animevideov3` là `default: true` theo `AGENTS.md` §5).

---

## 2. Danh Mục Các Thành Phần Đã Kiểm Chứng (Verified & Production Ready)

| Phân hệ | Module / File | Trạng thái kiểm chứng | Chi tiết kỹ thuật |
| :--- | :--- | :--- | :--- |
| **PyAV 18.1.0 In-Process Probing & Audit** | `src/ffmpeg_ops.py`, `src/mux_audit.py` | PASS | Thay thế ~230-290 `ffprobe` subprocesses/tập bằng PyAV 18.1.0 in-process, giữ nguyên tuyệt đối Concat-Mux `-max_interleave_delta 0` và Single-Pass `streamhash` MD5 subprocesses. |
| **Track Defaults CLI** | `scripts/set_track_defaults.py` | PASS | Tham số hóa `argparse` CLI, tùy biến đường dẫn video/mkvpropedit, xử lý đường dẫn tương đối và hỗ trợ cờ `--dry-run`. |
| **Triton P010LE Kernel & Pinned Ring** | `src/trt_zero_copy_upscaler.py` | PASS | Kernel `@triton.jit` BT.709 TV-range + clamp RGB `[0,1]` trên thanh ghi trước ma trận có dấu (0 LSB rò rỉ sắc độ vùng chói Sakuga), khử cold-start <50ms & B=1 dynamic recompile, nạp H2D trực tiếp vào `gpu_nhwc_ring`. |
| **Unique Inference FPS Deadline Realignment** | `upscale.py`, `src/scheduler.py` | PASS | Căn chỉnh công thức ETA `--deadline` theo Unique Inference FPS ($N_{\text{unique}} / \text{FPS}_{\text{unique}}$), loại bỏ triệt để sai lệch dự toán ~2.1x khi có dedup. |
| **SQLite Connection Context Manager** | `subtrans/cache.py`, `subtrans/rag.py` | PASS | Bọc kết nối SQLite bằng `contextlib.contextmanager` đóng sạch handle, triệt tiêu hoàn toàn lỗi khóa file `WinError 32` trên Windows 11. |
| **SQLite FTS5 2-Gram/3-Gram Kanji RAG** | `subtrans/rag.py` | PASS | Nâng cấp chỉ mục RAG lên SQLite FTS5 với tokenizer character 2-gram + 3-gram, tra cứu chính xác từ khóa Kanji 2 chữ (`魔術`, `聖杯`, `令呪`). |
| **Spandrel 0.4.2 Native 2x & Auto-Verify** | `src/cuda_engine.py`, `src/build_trt_engine.py`, `src/models_resolver.py`, `models/models.yaml` | PASS | Nạp kiến trúc Native 2x qua `spandrel 0.4.2` & `AnimeJaNaiCompact2x(num_feat, num_conv)`, phân giải ID không phân biệt hoa thường, tự động tải `.pth`/`.zip` bắt buộc xác thực SHA-256 64-hex + HTTPS, singleton `_get_trt_logger()` & `get_gpu_worker` theo `device`/`weights_path`, giới hạn `max_batch=2` khi `VRAM <15 GB` giúp 8/8 model biên dịch sạch và Native `2x` đạt `18.27–21.13 fps` Climax (`0` OOM, hỗ trợ cả khung lẻ `1920x804`). |
| **Faster-Whisper Large-V3-Turbo STT** | `src/subtitles.py` | PASS | Mặc định model STT `large-v3-turbo` trên CUDA float16 với fallback CPU int8, không ghi `.ja.srt` rỗng khi 0 cue và luôn thu hồi `.temp.wav` + VRAM trong `finally:`, tương thích 100% với 5.1 FC mono & ghim mốc DTW. |
| **E2E Acceptance Suite** | `tests/test_e2e_group_a.py`, `tests/test_e2e_group_b.py`, `tests/test_e2e_integration_t3_t4.py` | PASS | 169 ca kiểm thử tự động 4 tầng (Tier 1 Isolated, Tier 2 BVA, Tier 3 Cross-Feature, Tier 4 Workload Scenarios) bảo vệ toàn diện 14 tính năng của Nhóm A & Nhóm B. |
| **PyTorch CUDA / TensorRT 10 FP16** | `src/cuda_engine.py`, `src/build_trt_engine.py`, `src/config.py` | PASS | `CUDA_MODULE_LOADING=LAZY`, `CUDA_CACHE_MAXSIZE=2GB`, TensorRT 10.16 (`resolve_trt_profile_max_batch`, `_out_ring` 2-slot 0-alloc, `execute_async_v3`), kế thừa `UPSCALER_HEVC_ENCODER`. |
| **Zero-Disk Pipe & 3-Stage Overlap** | `src/trt_zero_copy_upscaler.py`, `src/pipeline.py`, `src/ffmpeg_ops.py` | PASS | Gối đầu 3 tầng + Background Cel trên `dec_exec`, 2-slot `_p010_ring` & `gpu_in_ring` (0 cấp phát VRAM động, 100% TensorRT pointer hit), FMA Scalar Add Fusion, Triton warmup, `_PipeDrainWorker` `8 MiB` + chống deadlock, Dynamic NVDEC & cắt tỉa `CRA ffprobe`. |
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
