# Bản đồ project cho agent

Đối chiếu tĩnh ngày **2026-11-13**, từ baseline Git `13b5183`. Đây là bản đồ tìm code, không phải chứng nhận pipeline đã chạy thành công. Trạng thái mới nhất: [STATE](STATE.md); khác biệt giữa ý định và code: [DECISIONS](DECISIONS.md); cách kiểm thử: [VALIDATION](VALIDATION.md). Bắt đầu bằng [START_HERE](START_HERE.md) và [AGENTS gốc](../../AGENTS.md).

## Điểm vào và luồng chính

- [`upscale.py`](../../upscale.py): `build_arg_parser`/`parse_args` nhận CLI; `main` điều phối queue, deadline, retry, mode phụ đề và cleanup. `check_deadline_and_eta` kiểm tra ETA trước mỗi tập.
- [`upscale.bat`](../../upscale.bat) / [`upscale_turbo.bat`](../../upscale_turbo.bat): chạy từ root; chọn Python trong `.venv`, `venv`, rồi PATH (`upscale_turbo.bat` bật `--zero-disk`).
- [`subtrans.py`](../../subtrans.py) / [`subtrans.bat`](../../subtrans.bat) → [`subtrans/cli.py`](../../subtrans/cli.py): Typer CLI với subcommand `run`, `batch`, `eval`, `rag` và các công cụ khác. Đọc [AGENTS Subtrans](../../subtrans/AGENTS.md) trước sửa.

```text
upscale.py:main
  → pipeline.run_upscale_pipeline
  → probe + resolve model + preflight + manifest
  → split video theo GOP + đếm packet từng segment + dedup plan
  → extract frames → PyTorch inference → encode/re-expand → rolling cleanup
  → concat → mux stream copy → mux audit → cleanup khi đạt và không keep-temp
  → Stage B phụ đề trong subprocess sau khi queue video kết thúc (nếu bật)

subtrans.cli → TranslationPipeline
  → parse/align → cast + glossary → Gemini translation/cache
  → lint/cascade repair → chuẩn hóa playback/write → QA report/quality gate
```

Đọc code nhánh thực tế trước khi thay đổi: các nhánh `--clean-temp`, `--subs-pending`, `--subs-only` ở `upscale.main` đều đã có chốt bảo vệ `--dry-run`; đường `--zero-disk` hỗ trợ độ phân giải động nhưng vẫn là cờ tùy chọn cần người dùng kích hoạt.

## Các cụm module video

| File trong `src/` | Trách nhiệm và symbol để tìm |
| --- | --- |
| [`config.py`](../../src/config.py) | Hằng số runtime, `configure_cuda_fast_math` (Ampere FP16 `HMMA.F16` + cuDNN benchmark), kế thừa `UPSCALER_HEVC_ENCODER`, `get_hevc_rate_args`, `validate_environment`; khóa `.gpu_active.lock` kiểm tra PID sống. |
| [`pipeline.py`](../../src/pipeline.py) | `run_upscale_pipeline`, `FailedResumableError`; gối đầu 3 tầng thực sự (`Decode N+1` + `GPU N` + `worker_finalize_zero_disk` / `Encode N-1`), Zero-Disk NVDEC RGB24 pipe, TensorRT 10 FP16, 0-reinference Sanity, 1-Pass Concat-Mux. |
| [`ffmpeg_ops.py`](../../src/ffmpeg_ops.py) | `probe_video`, `split_video_segments`, `get_segments_frame_counts`, `extract_frames`, `extract_frames_raw_rgb24` (`readinto` 0-copy), `encode_chunk_hevc`, `concat_segments`, `mux_final_container`, `mux_segments_directly`, `embed_subtitle_track`. |
| [`models_resolver.py`](../../src/models_resolver.py) | `load_models_registry`, `get_default_model_id`, `resolve_model_for_video`, `verify_weights_sha256`, `list_models_formatted`; neo đường dẫn tương đối về `ROOT_DIR`. |
| [`cuda_engine.py`](../../src/cuda_engine.py) | `RealESRAnimeVideoV3` (vi phân lô đuôi 8K 2-frame), `AnimeJaNaiCompact2x`, `GPUPersistentWorker` (Cyclic Cel Dedup + NTFS Hardlink, TensorRT 10 auto-runner + Pinned D2H), `get_cuda_inference_engine`, `upscale_frames_with_fallback`. |
| [`ai_engine.py`](../../src/ai_engine.py) | `upscale_frames`: runner NCNN Vulkan dự phòng; không phải engine PyTorch chính. |
| [`dedup.py`](../../src/dedup.py) | `DEDUP_POLICY_PARAMS`, `resolve_dedup_policy`, `deduplicate_cyclic_frames`, `raw_actions_rle` tái phân hoạch GOP `< 1 ms`, `run_preflight_worker_cli` (tách sẵn `raw_segs` + GOP), 1-Pass fused `mpdecimate+gray`, Early-Exit DIS Flow. |
| [`output_sanity.py`](../../src/output_sanity.py) | `check_frame_sanity_arrays` (nhận trực tiếp ảnh thu nhỏ GPU `270x480`), `check_frame_sanity`, `OutputSanityError`, `write_failed_sanity_audit`; ΔY/correlation guard. |
| [`mux_audit.py`](../../src/mux_audit.py) | `verify_mux_integrity` (đa luồng `ThreadPoolExecutor`, tái sử dụng metadata), `get_all_audio_streams_md5` (1-Pass `-f streamhash`), `check_window_interleaving` song song. |
| [`ramdisk.py`](../../src/ramdisk.py) | `RAMDisk` tự dò đĩa RAM vật lý/`UPSCALER_RAMDISK_DIR` hoặc định tuyến về `scratch/rolling_buffer` trên NVMe Gen4 (`E:`); `cleanup_orphan_ramdisks` dọn thư mục mồ côi. |
| [`cleanup.py`](../../src/cleanup.py) | `cleanup_checkpoint_dir`, `cleanup_subtitle_extracts`, `organize_output_artifacts`, `clean_all_temporary_data`, `get_path_size`; có xóa file/symlink target, không dùng để khảo sát. |
| [`preflight.py`](../../src/preflight.py) | `AntiSleepContext`, `configure_alder_lake_pcore_qos` (tắt EcoQoS giữ tiến trình trên P-Cores i5-12600K), `check_power_source`, `check_disk_space`, `run_preflight_checks`. |
| [`scheduler.py`](../../src/scheduler.py) | `find_cached_plan`, `preflight_video_plan`, `BackgroundQueuePreflightManager`, `compute_overnight_schedule`, `check_gemini_quota_for_queue`, `format_schedule_decision_table`. |
| [`logger.py`](../../src/logger.py) | `RunLogger`, format/progress helpers, `get_swap_usage`, `get_thermal_state`; ghi log theo tập và cập nhật `latest.log`. |
| [`night_report.py`](../../src/night_report.py) | `parse_telemetry_window`, `parse_episode_chunks_log`, `query_subtrans_api_stats`, `generate_night_report_text`; còn dữ liệu lịch sử hardcoded, không mặc nhiên là đo đạc mới. |
| [`subtitles.py`](../../src/subtitles.py) | `find_or_extract_subtitles`, `transcribe_japanese_audio`, `_build_whisper_initial_prompt`; hỗ trợ `faster-whisper` + Silero VAD (fallback `openai-whisper`), ưu tiên EN 0ms lệch mặc định hoặc `--prefer-whisper`. |
| [`pending_queue.py`](../../src/pending_queue.py) | `load_pending_subs`, `save_pending_subs`, `add_pending_sub`, `remove_pending_sub`; module stdlib-only quản lý `pending_subs.json` không kích hoạt import `torch`/CUDA. |
| [`trt_zero_copy_upscaler.py`](../../src/trt_zero_copy_upscaler.py) | `GPUColorConverterP010LE`, `SRVGGNetCompactTRTWrapper` (cắt lát 2-frame giảm 75% đỉnh VRAM 8K), `DoubleBufferedPipeUpscaler` (transpose `uint8` giảm 50% băng thông VRAM, Belady's OPT Cyclic Memoization + 0-reinference Sanity). |
| [`build_trt_engine.py`](../../src/build_trt_engine.py) | `_PureFP16TRTExportModule`, ONNX export (`dynamo=False`), `TRTRuntimeEngineWrapper` (`_out_ring` 2-slot khử cấp phát động + cache `set_input_shape`/`set_tensor_address`), `get_hybrid_engine_or_wrapper`. |

`src/__init__.py` là package marker. Symbol nhỏ/nested không liệt kê hết: dùng `rg -n '^(class |def |    def )' src/<file>.py` khi đi sâu.

## Subtrans và dữ liệu thuật ngữ

| Cụm | Điểm đọc |
| --- | --- |
| Điều phối/cấu hình/schema | `subtrans/pipeline.py` (scene-cut clamp `>=4500ms`, inline cached context, Cascade write-back), `cli.py`, `config.py` (`PYTHON_DOTENV_DISABLED` guard), `config.yaml`, `models.yaml`, `models.py` (`p1`, `p2`, `mode_hint`, `Character.aliases`). |
| Nhập và căn thời gian | `parser.py` (CC sound prefix filter, `detect_dialogue_mode_hint`, Piecewise 2-Segment Auto-Offset), `dtw_auto_align.py` (`get_dtw_alignment_plan` 1:2 pause split, `split_en_reference_for_2_to_1`); model `Cue` xuyên pipeline. |
| Ngữ cảnh nhân vật/ngôn ngữ | `cast_sheet.py` (3-key rotation, `CastSheetResponse`), `dynamic_cast.py` (`observe_batch_cues` ưu tiên CoT `p1`/`p2`), `ja_morph_tagger.py` (`strip_ja_furigana`, Kansai-ben, `soregashi`), `glossary.py` (`_merge_address_rules`). |
| Dịch và độ bền API | `gemini_backend.py` (`build_episode_system_prompt` `>4.096` tokens Implicit Caching, `ThinkingConfig` MINIMAL/LOW, 3-key modulo & RPM failover), `translator.py`, `throttle.py` (per-key RPD), `cache.py` (`compute_contextual_cue_sig`, `cached_tokens`). |
| Hậu kiểm/tri thức | `linter.py` (Katakana boundary, `RULE_GLOSSARY_TERM`, `RULE_PRONOUN_CONSISTENCY`), `validation.py`, `cascade.py` (digit quarantine, contextual repair), `rag.py` (Hybrid BM25, `promote_cascade_repairs` silver), `writer.py` (ASS sign preservation), `report.py`. |
| Đánh giá | `subtrans/eval/runner.py` (`verify_regression_case`, `verify_pronoun_cue`), `metrics.py`, `regression_suite.yaml` (6 ca), `pronoun_suite.yaml` (22 cues); `subtrans/tests/`, `tests/test_subtrans_deep_upgrades.py`, `tests/test_subtrans_v2_max_quality.py`. |
| Thuật ngữ theo series | [`glossaries/README.md`](../../glossaries/README.md), `glossaries/base.yaml`, `glossary.*.yaml`; phân biệt với base glossary trong `subtrans/`. |

## Nguồn sự thật và dữ liệu vận hành

- **Cấu hình video:** `src/config.py`; model metadata/default: `models/models.yaml`; override theo series: `shows.yaml`; CLI override: `upscale.py`. Resolver model dùng đường dẫn tương đối, nên chạy từ root repo.
- **Cấu hình dịch:** `subtrans/config.yaml` và `subtrans/models.yaml`, áp dụng bởi `config.py`, `pipeline.py` và CLI. Model ID/quota là cấu hình lưu trong repo, chưa được xác minh qua dịch vụ trong phiên khảo sát.
- **Checkpoint video:** `.temp_<basename>/manifest.json`, `plan.json`, `raw_segs/`, `upscaled_segs/`; không phải agent memory, không sửa/xóa để dọn workspace.
- **Dữ liệu runtime khác:** `out/`, `logs/`, cache/SQLite, cast, audit, `pending_subs.json`, `.gpu_active.lock`, weights. Không đọc hàng loạt hay đưa vào context/commit. `.env` và key luôn ngoài memory.
- **Tài liệu chuyên sâu:** [Video](../video-pipeline.md), [Dedup](../dedup-architecture.md), [Audit](../audit-and-safety.md), [Engineering log](../engineering-log.md). Các con số benchmark là lịch sử nếu thiếu phép đo mới.

## Script, công cụ Agent, benchmark và test

- [`.agents/`](../../.agents):
  - `skills/`: 4 workspace skills nạp theo ngữ cảnh (`safe-offline-test`, `video-upscale-ops`, `subtrans-anime-ops`, `agent-handoff-commit`).
  - `plugins/upscaler-guard/`: lifecycle hooks (`SessionStart`, `PreToolUse`) bảo vệ `.env`, `.temp_*`, Gemini quota và khóa GPU.
  - `plugins/upscaler-mcp/`: MCP server Stdio read-only (mặc định tắt để tiết kiệm token).
- [`.vscode/`](../../.vscode), [`.editorconfig`](../../.editorconfig), [`ruff.toml`](../../ruff.toml), [`.env.example`](../../.env.example): cấu hình IDE/linter/mẫu biến môi trường dùng chung cho dev và AI Agents.
- [`scripts/`](../../scripts):
  - `agent_context.py`: Bootstrap Agent, kiểm tra tính toàn vẹn bộ nhớ (`--check`), trích xuất cấu trúc code AST (`--symbols`), tìm kiếm symbol (`--find`), và cài đặt pre-commit hook (`--install-hooks`).
  - `verify_offline.py`: Kiểm thử an toàn sandbox offline ngắt kết nối mạng.
  - `unlock_rtx3060_admin.ps1`: Kích hoạt HAGS (`HwSchMode=2`), xuất/nạp profile tắt `CUDA - Force P2 State` và chẩn đoán RAM XMP / Resizable BAR / TensorRT.
  - `batch_translate_all.py` điều phối dịch tuần tự; `set_track_defaults.py` đổi default track MKV; `organize_vietnamese_subs.py` sắp xếp phụ đề; `upscale_x4_single.py` tiện ích upscale riêng. Đọc tham số/đường dẫn và side effect trước chạy.
- [`benchmarks/README.md`](../../benchmarks/README.md): nhóm `upscale/` đo model, dedup, latency và chaos resume; `subtrans/` đánh giá pronoun; `research/` hiệu chuẩn DIS/so sánh hình ảnh. Không thuộc test mặc định và không phải lệnh bootstrap.
- [`pytest.ini`](../../pytest.ini): thu thập `tests/` và `subtrans/tests/`. Video tests chia nhóm dedup/GOP, resume/storage, model/color, mux; subtitle tests chia parser/writer/glossary, API stub/quota/cache, cascade/RAG/eval.
- `-m "not gpu"` loại test gắn marker GPU, nhưng vài test không gắn marker vẫn tự chọn CUDA. Chỉ một tiến trình test, kiểm tra production idle và cách ly dữ liệu theo [VALIDATION](VALIDATION.md).
- Test mô phỏng/unit PASS không thay thế nghiệm thu video thực, frame count hay audit trên tập thành phẩm. Phạm vi kiểm chứng hiện tại phải được ghi trong [STATE](STATE.md).
