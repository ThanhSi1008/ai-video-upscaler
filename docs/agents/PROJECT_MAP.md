# Bản đồ project cho agent

Đối chiếu tĩnh ngày **2026-11-15**, từ nhánh `main` (Nhóm A, Nhóm B & Hiệu chuẩn Silicon GA106 R3 hoàn tất). Đây là bản đồ tìm code, không phải chứng nhận pipeline đã chạy thành công. Trạng thái mới nhất: [STATE](STATE.md); khác biệt giữa ý định và code: [DECISIONS](DECISIONS.md); cách kiểm thử: [VALIDATION](VALIDATION.md). Bắt đầu bằng [START_HERE](START_HERE.md) và [AGENTS gốc](../../AGENTS.md).

## Điểm vào và luồng chính

- [`upscale.py`](../../upscale.py): `build_arg_parser`/`parse_args` nhận CLI (`--zero-disk` mặc định `True` kèm `--no-zero-disk`/`--disk`, đồng bộ đủ `sakuga`/`dis-sakuga`); `expand_video_inputs` mở rộng thư mục thành hàng đợi sắp xếp tự nhiên (bỏ qua file ẩn và bản `*_4K`); `main` điều phối queue, deadline, retry, mode phụ đề và cleanup. `check_deadline_and_eta` kiểm tra ETA trước mỗi tập.
- [`upscale.bat`](../../upscale.bat) / [`upscale_turbo.bat`](../../upscale_turbo.bat): chạy từ root (chuẩn CRLF trên `cmd.exe`); chọn Python trong `.venv`, `venv`, rồi PATH (cả hai chạy mặc định Zero-Disk TensorRT 10 FP16).
- [`subtrans.py`](../../subtrans.py) / [`subtrans.bat`](../../subtrans.bat) → [`subtrans/cli.py`](../../subtrans/cli.py): Typer CLI với subcommand `run`, `batch`, `eval`, `rag` và các công cụ khác. Đọc [AGENTS Subtrans](../../subtrans/AGENTS.md) trước sửa.
- [`web_gui.py`](../../web_gui.py) / [`web_gui.bat`](../../web_gui.bat) → [`web/app.py`](../../web/app.py): Giao diện Web cục bộ thời gian thực trên `http://127.0.0.1:8765` (FastAPI + Uvicorn + SSE 1Hz), điều phối subprocess `upscale.py` / `subtrans.py` không chiếm VRAM tĩnh, kết nối cầu nối `logs/live_status.json` và `logs/.graceful_pause.flag`.

```text
upscale.py:main
  → pipeline.run_upscale_pipeline
  → probe + resolve model + preflight (Alder Lake P-Core QoS + AntiSleep) + manifest
  → split video theo GOP (hoặc kế thừa raw_segs từ Queue Preflight) + dedup plan (raw_actions_rle)
  → 3-Stage Overlapped Loop:
      [Zero-Disk] NVDEC RGB24 pipe (N+1) ‖ TensorRT 10 FP16 + Triton P010LE (N) ‖ Async NVENC flush (N-1)
      [Chuẩn]     FFmpeg PNG extract (N+1) ‖ TensorRT/PyTorch FP16 + Cyclic Hardlink (N) ‖ NVENC + Rolling Cleanup (N-1)
  → 1-Pass Concat-Mux (mux_segments_directly) → Parallel Mux Audit (streamhash MD5) → cleanup khi PASS
  → Stage B phụ đề trong subprocess (đồng bộ sau queue hoặc song song qua --async-subs)

subtrans.cli → TranslationPipeline
  → parse/align → cast + glossary → Gemini translation/cache
  → lint/cascade repair → chuẩn hóa playback/write → QA report/quality gate
```

Đọc code nhánh thực tế trước khi thay đổi: các nhánh `--clean-temp`, `--subs-pending`, `--subs-only` ở `upscale.main` đều đã có chốt bảo vệ `--dry-run`; đường `--zero-disk` (mặc định `True`) hỗ trợ độ phân giải động và có cờ đối ứng `--no-zero-disk` / `--disk` để lùi về lưu PNG khi cần đối soát.

## Các cụm module video

| File trong `src/` | Trách nhiệm và symbol để tìm |
| --- | --- |
| [`config.py`](../../src/config.py) | Hằng số runtime, `CACHE_ROOT_DIR`, `configure_nvme_jit_caches` (định tuyến `CUDA_CACHE_PATH`/`TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` về `.cache` trên NVMe Gen4 `WD_BLACK SN7100`), `configure_cuda_fast_math` (Ampere FP16 `HMMA.F16` + cuDNN benchmark), kế thừa `UPSCALER_HEVC_ENCODER`, `get_hevc_rate_args`, `validate_environment`; `gpu_exclusivity_lock` và khóa `.gpu_active.lock` kiểm tra PID sống. |
| [`pipeline.py`](../../src/pipeline.py) | `run_upscale_pipeline`, `FailedResumableError`; gối đầu 3 tầng + Background Cel trên `dec_exec`, tích lũy `micro_timings` toàn cục, xác thực khung hình khi resume (`verify_segment_resume_frames`), Zero-Disk TensorRT 10 FP16, 1-Pass Concat-Mux. |
| [`ffmpeg_ops.py`](../../src/ffmpeg_ops.py) | `probe_video`, `split_video_segments`, `get_segments_frame_counts`, `_count_leading_discard_packets` (cắt tỉa non-HEVC), `verify_segment_resume_frames`, `extract_frames`, `extract_frames_raw_rgb24` (Dynamic NVDEC 8-bit + `1 MiB` slice), `mux_segments_directly`. |
| [`models_resolver.py`](../../src/models_resolver.py) | `load_models_registry`, `get_default_model_id`, `match_show_pattern` (khớp glob/substring an toàn với ngoặc vuông `[SubsPlease]`, `[BD 1080p]`), `resolve_model_for_video` (khớp ID không phân biệt hoa thường cho CLI & `shows.yaml`), `verify_weights_sha256`, `list_models_formatted`; neo đường dẫn tương đối về `ROOT_DIR`. |
| [`cuda_engine.py`](../../src/cuda_engine.py) | `RealESRAnimeVideoV3` (vi phân lô đuôi 8K 2-frame), `AnimeJaNaiCompact2x` (tham số hóa `num_feat`/`num_conv`), `SpandrelModelWrapper`, `GPUPersistentWorker` (tự động tải `.pth`/`.zip` có query-string + bắt buộc xác thực SHA-256 64-hex, Cyclic Cel Dedup + NTFS Hardlink, TensorRT 10 auto-runner), `get_gpu_worker` (kiểm tra `weights_path` & `device`), `get_cuda_inference_engine`, `upscale_frames_with_fallback`. |
| [`ai_engine.py`](../../src/ai_engine.py) | `upscale_frames`: runner NCNN Vulkan dự phòng; không phải engine PyTorch chính. |
| [`dedup.py`](../../src/dedup.py) | `DEDUP_POLICY_PARAMS` (`sakuga`, `dis-sakuga`), `resolve_dedup_policy`, `deduplicate_cyclic_frames` (băm 3 tầng `64x36` thumbnail + BLAKE2b + SIMD uint8, `27x`), `raw_actions_rle`, phân tách ngưỡng DIS toàn cục/cục bộ. |
| [`output_sanity.py`](../../src/output_sanity.py) | `check_frame_sanity_arrays` (nhận trực tiếp ảnh thu nhỏ GPU `270x480`), `check_frame_sanity`, `OutputSanityError`, `write_failed_sanity_audit`; ΔY/correlation guard. |
| [`mux_audit.py`](../../src/mux_audit.py) | `verify_mux_integrity` (đa luồng `ThreadPoolExecutor`, tái sử dụng metadata), `get_all_audio_streams_md5` (1-Pass `-f streamhash`), `check_window_interleaving` song song. |
| [`ramdisk.py`](../../src/ramdisk.py) | `RAMDisk` tự dò đĩa RAM vật lý/`UPSCALER_RAMDISK_DIR` hoặc định tuyến về `scratch/rolling_buffer` trên NVMe Gen4 (`E:`); `cleanup_orphan_ramdisks` dọn thư mục mồ côi. |
| [`cleanup.py`](../../src/cleanup.py) | `cleanup_checkpoint_dir`, `cleanup_subtitle_extracts`, `organize_output_artifacts`, `clean_all_temporary_data`, `get_path_size`; có xóa file/symlink target, không dùng để khảo sát. |
| [`preflight.py`](../../src/preflight.py) | `AntiSleepContext`, `configure_alder_lake_pcore_qos` (tắt EcoQoS giữ tiến trình trên P-Cores i5-12600K), `check_power_source`, `check_disk_space`, `run_preflight_checks`. |
| [`scheduler.py`](../../src/scheduler.py) | `find_cached_plan`, `preflight_video_plan`, `BackgroundQueuePreflightManager`, `compute_overnight_schedule`, `check_gemini_quota_for_queue`, `format_schedule_decision_table`. |
| [`logger.py`](../../src/logger.py) | `RunLogger`, format/progress helpers, `get_swap_usage`, `get_thermal_state`; ghi log theo tập, cập nhật `latest.log`, ghi nguyên tử `logs/live_status.json` (heatmap chunk, FPS, micro-timings) và kiểm tra cờ dừng êm `logs/.graceful_pause.flag`. |
| [`night_report.py`](../../src/night_report.py) | `parse_telemetry_window`, `parse_episode_chunks_log`, `query_subtrans_api_stats`, `generate_night_report_text`; còn dữ liệu lịch sử hardcoded, không mặc nhiên là đo đạc mới. |
| [`subtitles.py`](../../src/subtitles.py) | `find_or_extract_subtitles`, `transcribe_japanese_audio`, `_build_whisper_initial_prompt`; hỗ trợ `faster-whisper` + Silero VAD (fallback `openai-whisper`), ưu tiên EN 0ms lệch mặc định hoặc `--prefer-whisper`. |
| [`pending_queue.py`](../../src/pending_queue.py) | `load_pending_subs`, `save_pending_subs`, `add_pending_sub`, `remove_pending_sub`; module stdlib-only quản lý `pending_subs.json` không kích hoạt import `torch`/CUDA. |
| [`trt_zero_copy_upscaler.py`](../../src/trt_zero_copy_upscaler.py) | `_nhwc_u8_to_nchw_fp16_norm_kernel` & `_nhwc_u8_to_nchw_fp16_norm` (`@triton.jit` chuẩn hóa đầu vào 1-pass `0.51 ms/b`), `_rgb_to_p010le_triton_kernel` (`@triton.jit` BT.709 TV-range + clamp RGB `[0,1]` trước ma trận), `GPUColorConverterP010LE`, `SRVGGNetCompactTRTWrapper`, `_PipeDrainWorker` (`8 MiB` zero-copy + chống deadlock), `DoubleBufferedPipeUpscaler` (`_p010_ring` + `gpu_in_ring` + `gpu_nhwc_ring` 0-alloc, `_ensure_cuda_event_pools` tách `avg_h2d_dma_ms` / `avg_in_norm_ms` trong `micro_timings`). |
| [`build_trt_engine.py`](../../src/build_trt_engine.py) | `_get_trt_logger` (singleton `ILogger`), `_PureFP16TRTExportModule`, `_GenericTRTExportModule`, `resolve_trt_profile_max_batch` (giới hạn `max_batch=2` theo VRAM `<15 GB` tránh OOM 12.15 GiB), ONNX export (`dynamo=False`), `TRTRuntimeEngineWrapper` (`_out_ring` 2-slot + `_batch_out_ring` cắt lát zero-copy), `get_hybrid_engine_or_wrapper`. |

`src/__init__.py` là package marker. Symbol nhỏ/nested không liệt kê hết: dùng `rg -n '^(class |def |    def )' src/<file>.py` khi đi sâu.

## Web Studio thời gian thực (`web/`)

| File trong `web/` | Trách nhiệm và symbol để tìm |
| --- | --- |
| [`app.py`](../../web/app.py) | `create_app`: REST API (`/api/status`, `/api/events`, `/api/logs/recent`, `/api/fs/browse`, `/api/queue/preview-schedule`, `/api/jobs/*`, `/api/open-media`, `/api/preview/*`, `/api/config/*`, `/api/subtrans/*`, `/api/audits*`), trích xuất PNG BT.709 đối chiếu `1080p vs 4K`, đọc chỉ-đọc SQLite cache Subtrans, quản lý 3-key `.env` che mặt nạ và security headers (`nosniff`, `DENY`). |
| [`sse_hub.py`](../../web/sse_hub.py) | `TelemetryHub`, `check_active_gpu_lock`, `scan_temp_checkpoints`: vòng lặp 1.0 Hz thu thập phần cứng (GPU/VRAM/NVENC/Power/Temp/CPU/RAM/NVMe), đọc `logs/live_status.json`, `.gpu_active.lock`, `.temp_*` checkpoints và tail `logs/latest.log` / `logs/web_job.log`. |
| [`job_manager.py`](../../web/job_manager.py) | `JobManager`: quản lý subprocess `upscale.py` / `subtrans.py` không qua shell, lập lịch `compute_schedule_preview`, chạy hàng đợi/preview/subtrans mọi khung giờ, kiểm tra khóa GPU cho tác vụ video, đặt/xóa cờ dừng êm `.graceful_pause.flag` và hủy khẩn cấp `abort_job`. |
| [`fs_browser.py`](../../web/fs_browser.py) | `SafeLocalFSBrowser`, `natural_sort_key`, `is_secret_filename`: liệt kê ổ đĩa cố định/RAMDisk trên Windows, duyệt thư mục/file media sắp xếp tự nhiên, chặn file nhạy cảm (`.env`, `.git`) và đối chiếu quy tắc `shows.yaml`. |
| [`static/`](../../web/static) | Giao diện chuẩn `modern-web-guidance` (`0` lỗi `legacy-scan`, `0` `innerHTML`): `index.html`, `styles.css` (`oklch()` + `light-dark()`), `js/ui.js`, `js/charts.js`, `js/fs_picker.js`, `js/compare_stage.js`, `js/panels.js`, `js/app.js`. |

## Subtrans và dữ liệu thuật ngữ

| Cụm | Điểm đọc |
| --- | --- |
| Điều phối/cấu hình/schema | `subtrans/pipeline.py` (scene-cut clamp `>=4500ms`, inline cached context, Cascade write-back), `cli.py`, `config.py` (`PYTHON_DOTENV_DISABLED` guard), `config.yaml`, `models.yaml`, `models.py` (`p1`, `p2`, `mode_hint`, `Character.aliases`). |
| Nhập và căn thời gian | `parser.py` (CC sound prefix filter, `detect_dialogue_mode_hint`, Piecewise 2-Segment Auto-Offset), `dtw_auto_align.py` (`get_dtw_alignment_plan` 1:2 pause split, `split_en_reference_for_2_to_1`); model `Cue` xuyên pipeline. |
| Ngữ cảnh nhân vật/ngôn ngữ | `cast_sheet.py` (3-key rotation, `CastSheetResponse`), `dynamic_cast.py` (`observe_batch_cues` ưu tiên CoT `p1`/`p2`), `audio_diarization.py` (5.1 FC 8kHz CMS timbre + F0 `audio_turn`), `ja_morph_tagger.py` (MeCab `fugashi` + `unidic-lite`, 0 báo giả Kanji ghép), `glossary.py` (tự dò theo thư mục cha). |
| Dịch và độ bền API | `gemini_backend.py` (`build_episode_system_prompt` `>4.096` tokens Implicit & `--explicit-cache`, `ThinkingConfig` MINIMAL/LOW, 3-key modulo & RPM failover), `translator.py`, `throttle.py` (per-key RPD), `cache.py` (`compute_contextual_cue_sig`, `cached_tokens`). |
| Hậu kiểm/tri thức | `linter.py` (Katakana boundary, `RULE_GLOSSARY_TERM`, `RULE_PRONOUN_CONSISTENCY`), `validation.py`, `cascade.py` (`build_repair_system_prompt` `>4.096` tokens, digit quarantine), `rag.py` (Hybrid BM25, `BUILTIN_GOLD_EXEMPLARS` `rag seed`, mẫu `silver`), `writer.py`, `report.py`. |
| Đánh giá | `subtrans/eval/runner.py` (`verify_regression_case`, `verify_pronoun_cue`), `metrics.py`, `regression_suite.yaml` (6 ca), `pronoun_suite.yaml` (22 cues); `subtrans/tests/`, `tests/test_subtrans_deep_upgrades.py`, `tests/test_subtrans_v2_max_quality.py`. |
| Thuật ngữ theo series | [`glossaries/README.md`](../../glossaries/README.md), `glossaries/base.yaml`, `glossary.*.yaml`; phân biệt với base glossary trong `subtrans/`. |

## Nguồn sự thật và dữ liệu vận hành

- **Cấu hình video:** `src/config.py`; model metadata/default: `models/models.yaml`; override theo series: `shows.yaml`; CLI override: `upscale.py`. Resolver model dùng đường dẫn tương đối, nên chạy từ root repo.
- **Cấu hình dịch:** `subtrans/config.yaml` và `subtrans/models.yaml`, áp dụng bởi `config.py`, `pipeline.py` và CLI. Model ID/quota là cấu hình lưu trong repo, chưa được xác minh qua dịch vụ trong phiên khảo sát.
- **Checkpoint video:** `.temp_<basename>/manifest.json`, `plan.json`, `raw_segs/`, `upscaled_segs/`; không phải agent memory, không sửa/xóa để dọn workspace.
- **Dữ liệu runtime khác:** `out/`, `logs/`, cache/SQLite, cast, audit, `pending_subs.json`, `.gpu_active.lock`, weights. Không đọc hàng loạt hay đưa vào context/commit. `.env` và key luôn ngoài memory.
- **Tài liệu chuyên sâu:** [Video](../video-pipeline.md), [Dedup](../dedup-architecture.md), [Audit](../audit-and-safety.md), [Engineering log](../engineering-log.md), [Roofline](../roofline_theoretical_model.md), [Empirical](../empirical_benchmark_report.md), [Gap Audit](../benchmarks_gap_analysis_audit.md). Các con số benchmark là lịch sử nếu thiếu phép đo mới.

## Script, công cụ Agent, benchmark và test

- [`.agents/`](../../.agents):
  - `skills/`: 5 workspace skills nạp theo ngữ cảnh (`safe-offline-test`, `video-upscale-ops`, `subtrans-anime-ops`, `agent-handoff-commit`, `modern-web-guidance`).
  - `plugins/upscaler-guard/`: lifecycle hooks (`SessionStart`, `PreToolUse`) bảo vệ `.env`, `.temp_*`, Gemini quota và khóa GPU.
  - `plugins/upscaler-mcp/`: MCP server Stdio read-only (mặc định tắt để tiết kiệm token).
- [`.vscode/`](../../.vscode), [`.editorconfig`](../../.editorconfig), [`ruff.toml`](../../ruff.toml), [`.env.example`](../../.env.example): cấu hình IDE/linter/mẫu biến môi trường dùng chung cho dev và AI Agents.
- [`scripts/`](../../scripts):
  - `agent_context.py`: Bootstrap Agent, kiểm tra tính toàn vẹn bộ nhớ (`--check`), trích xuất cấu trúc code AST (`--symbols`), tìm kiếm symbol (`--find`), và cài đặt pre-commit hook (`--install-hooks`).
  - `verify_offline.py`: Kiểm thử an toàn sandbox offline ngắt kết nối mạng, định tuyến `TEMP`/`TMP`/`--basetemp` về `scratch/test_runs/<stamp>/tmp` trên NVMe Gen4 (`E:`).
  - `silicon_roofline_probe.py`: Đo trần vật lý silicon GA106 cô lập (`run_silicon_probe`: `cublasGemmEx` `16F`/`32F`, GDDR6 STREAM Triad, PCIe 4.0 DMA, Triton input/output kernels và trần 7th-Gen NVENC `-multipass fullres`).
  - `telemetry_collector.py`, `benchmark_pipeline.py`, `compare_benchmarks.py`: Hạ tầng thu thập 17 chỉ số phần cứng chu kỳ `1.0s`, điều phối chạy benchmark tự động (Preview/Climax) và đối chiếu delta/bất biến.
  - `unlock_rtx3060_admin.ps1`: Kích hoạt HAGS (`HwSchMode=2`), xuất/nạp profile tắt `CUDA - Force P2 State` và chẩn đoán RAM XMP / Resizable BAR / TensorRT.
  - `batch_translate_all.py` điều phối dịch tuần tự; `set_track_defaults.py` đổi default track MKV; `organize_vietnamese_subs.py` sắp xếp phụ đề; `upscale_x4_single.py` tiện ích upscale riêng. Đọc tham số/đường dẫn và side effect trước chạy.
- [`benchmarks/README.md`](../../benchmarks/README.md): nhóm `upscale/` đo model, dedup, latency và chaos resume; `subtrans/` đánh giá pronoun; `research/` hiệu chuẩn DIS/so sánh hình ảnh; [`baseline_summary.md`](../../benchmarks/baseline_summary.md) & [`final_optimization_report.md`](../../benchmarks/final_optimization_report.md) lưu đối chứng Golden Baseline và báo cáo tối ưu hóa bão hòa Pareto + hiệu chuẩn Silicon GA106 R3.
- [`pytest.ini`](../../pytest.ini): thu thập `tests/` (bao gồm `tests/e2e/` 115 ca kiểm thử hộp đen 4 tầng, `tests/test_e2e_*.py` 169 ca Nhóm A/B, `tests/test_web_gui.py` cho Web Studio & SSE/REST, `tests/test_benchmark_infra.py` và `tests/test_nvme_cache_and_input_triton.py` 13 ca cache NVMe + Triton input norm) và `subtrans/tests/`. Video tests chia nhóm dedup/GOP, resume/storage, model/color, mux, E2E; subtitle tests chia parser/writer/glossary, API stub/quota/cache, cascade/RAG/eval.
- `-m "not gpu"` loại test gắn marker GPU, nhưng vài test không gắn marker vẫn tự chọn CUDA. Chỉ một tiến trình test, kiểm tra production idle và cách ly dữ liệu theo [VALIDATION](VALIDATION.md).
- Test mô phỏng/unit PASS không thay thế nghiệm thu video thực, frame count hay audit trên tập thành phẩm. Phạm vi kiểm chứng hiện tại phải được ghi trong [STATE](STATE.md).
