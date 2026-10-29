# Bản đồ project cho agent

Đối chiếu tĩnh ngày **2026-10-29**, từ baseline Git `8db5f9b`. Đây là bản đồ tìm code, không phải chứng nhận pipeline đã chạy thành công. Trạng thái mới nhất: [STATE](STATE.md); khác biệt giữa ý định và code: [DECISIONS](DECISIONS.md); cách kiểm thử: [VALIDATION](VALIDATION.md). Bắt đầu bằng [START_HERE](START_HERE.md) và [AGENTS gốc](../../AGENTS.md).

## Điểm vào và luồng chính

- [`upscale.py`](../../upscale.py): `build_arg_parser`/`parse_args` nhận CLI; `main` điều phối queue, deadline, retry, mode phụ đề và cleanup. `check_deadline_and_eta` kiểm tra ETA trước mỗi tập.
- [`upscale.bat`](../../upscale.bat): chạy từ root; chọn Python trong `.venv`, `venv`, rồi PATH.
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

Đọc code nhánh thực tế trước khi thay đổi: mode đặc biệt ở `upscale.main` có thể chạy trước nhánh `--dry-run`; đường `--zero-disk` còn hạn chế đã ghi trong DECISIONS.

## Các cụm module video

| File trong `src/` | Trách nhiệm và symbol để tìm |
| --- | --- |
| [`config.py`](../../src/config.py) | Hằng số runtime, `get_hevc_rate_args`, `validate_environment`; pending subtitles và GPU lock file. Import có side effect; xem VALIDATION. |
| [`pipeline.py`](../../src/pipeline.py) | `run_upscale_pipeline`, `FailedResumableError`; worker extract/encode, checkpoint, backpressure, deadline và quality gates. |
| [`ffmpeg_ops.py`](../../src/ffmpeg_ops.py) | `probe_video`, `split_video_segments`, `get_segments_frame_counts`, `extract_frames`, `encode_chunk_hevc`, `concat_segments`, `mux_final_container`, `embed_subtitle_track`. |
| [`models_resolver.py`](../../src/models_resolver.py) | `load_models_registry`, `get_default_model_id`, `resolve_model_for_video`, `list_models_formatted`; chọn metadata model, không tự xác minh hash weights. |
| [`cuda_engine.py`](../../src/cuda_engine.py) | `RealESRAnimeVideoV3`, `AnimeJaNaiCompact2x`, `GPUPersistentWorker`, `get_gpu_worker`, `get_cuda_inference_engine`, `upscale_frames_with_fallback`, `GPUError`; load weights, batch, precision fallback cùng model. |
| [`ai_engine.py`](../../src/ai_engine.py) | `upscale_frames`: runner NCNN Vulkan dự phòng; không phải engine PyTorch chính. |
| [`dedup.py`](../../src/dedup.py) | `DEDUP_POLICY_PARAMS`, `resolve_dedup_policy`, `detect_global_motion_frames`, `detect_drift_policy_actions`, `detect_dis_policy_actions`; `scan_dedup_plan`/`load_or_create_plan`, `prune_redundant_frames`, concat script/re-expansion và background preflight. |
| [`output_sanity.py`](../../src/output_sanity.py) | `check_frame_sanity_arrays`, `check_frame_sanity`, `OutputSanityError`, `write_failed_sanity_audit`; ΔY/correlation guard. |
| [`mux_audit.py`](../../src/mux_audit.py) | `verify_mux_integrity` và helpers MD5 audio, probe stream, subtitle packets, seek volume, window interleaving; xem giới hạn gate trong DECISIONS. |
| [`ramdisk.py`](../../src/ramdisk.py) | `RAMDisk` quản lý thư mục `tempfile` và exit hooks; `cleanup_orphan_ramdisks` là placeholder. Không có mount RAM disk trong bản hiện tại. |
| [`cleanup.py`](../../src/cleanup.py) | `cleanup_checkpoint_dir`, `cleanup_subtitle_extracts`, `organize_output_artifacts`, `clean_all_temporary_data`, `get_path_size`; có xóa file/symlink target, không dùng để khảo sát. |
| [`preflight.py`](../../src/preflight.py) | `AntiSleepContext`, `check_power_source`, `check_disk_space`, `run_preflight_checks`, `send_system_notification`; Win32 anti-sleep và kiểm tra tài nguyên. |
| [`scheduler.py`](../../src/scheduler.py) | `find_cached_plan`, `preflight_video_plan`, `BackgroundQueuePreflightManager`, `compute_overnight_schedule`, `check_gemini_quota_for_queue`, `format_schedule_decision_table`. |
| [`logger.py`](../../src/logger.py) | `RunLogger`, format/progress helpers, `get_swap_usage`, `get_thermal_state`; ghi log theo tập và cập nhật `latest.log`. |
| [`night_report.py`](../../src/night_report.py) | `parse_telemetry_window`, `parse_episode_chunks_log`, `query_subtrans_api_stats`, `generate_night_report_text`; còn dữ liệu lịch sử hardcoded, không mặc nhiên là đo đạc mới. |
| [`subtitles.py`](../../src/subtitles.py) | `find_or_extract_subtitles`, `transcribe_japanese_audio`; sidecar/container JA/EN, ưu tiên EN khi thiếu JA theo mặc định; Whisper tùy điều kiện. |
| [`trt_zero_copy_upscaler.py`](../../src/trt_zero_copy_upscaler.py) | `GPUColorConverterP010LE`, `SRVGGNetCompactTRTWrapper`, `DoubleBufferedPipeUpscaler`; đường pipe thử nghiệm với kích thước cố định. |
| [`build_trt_engine.py`](../../src/build_trt_engine.py) | ONNX export, TensorRT build/cache/runtime wrapper, `get_hybrid_engine_or_wrapper`; không suy ra CLI đã tích hợp TensorRT chỉ vì module tồn tại. |

`src/__init__.py` là package marker. Symbol nhỏ/nested không liệt kê hết: dùng `rg -n '^(class |def |    def )' src/<file>.py` khi đi sâu.

## Subtrans và dữ liệu thuật ngữ

| Cụm | Điểm đọc |
| --- | --- |
| Điều phối/cấu hình/schema | `subtrans/pipeline.py`, `cli.py`, `config.py`, `config.yaml`, `models.yaml`, `models.py`. |
| Nhập và căn thời gian | `parser.py`, `dtw_auto_align.py`; model `Cue` là dữ liệu xuyên pipeline. |
| Ngữ cảnh nhân vật/ngôn ngữ | `cast_sheet.py`, `dynamic_cast.py`, `ja_morph_tagger.py`, `glossary.py`, `glossary.yaml`. |
| Dịch và độ bền API | `gemini_backend.py`, `translator.py`, `throttle.py`, `cache.py`; không dùng lớp legacy như bằng chứng cho backend được phép. |
| Hậu kiểm/tri thức | `linter.py`, `validation.py`, `cascade.py`, `rag.py`, `writer.py`, `report.py`. |
| Đánh giá | `subtrans/eval/runner.py`, `metrics.py`, các YAML suites; `subtrans/tests/` và `subtrans/samples/`. |
| Thuật ngữ theo series | [`glossaries/README.md`](../../glossaries/README.md), `glossaries/base.yaml`, `glossary.*.yaml`; phân biệt với base glossary trong `subtrans/`. |

## Nguồn sự thật và dữ liệu vận hành

- **Cấu hình video:** `src/config.py`; model metadata/default: `models/models.yaml`; override theo series: `shows.yaml`; CLI override: `upscale.py`. Resolver model dùng đường dẫn tương đối, nên chạy từ root repo.
- **Cấu hình dịch:** `subtrans/config.yaml` và `subtrans/models.yaml`, áp dụng bởi `config.py`, `pipeline.py` và CLI. Model ID/quota là cấu hình lưu trong repo, chưa được xác minh qua dịch vụ trong phiên khảo sát.
- **Checkpoint video:** `.temp_<basename>/manifest.json`, `plan.json`, `raw_segs/`, `upscaled_segs/`; không phải agent memory, không sửa/xóa để dọn workspace.
- **Dữ liệu runtime khác:** `out/`, `logs/`, cache/SQLite, cast, audit, `pending_subs.json`, `.gpu_active.lock`, weights. Không đọc hàng loạt hay đưa vào context/commit. `.env` và key luôn ngoài memory.
- **Tài liệu chuyên sâu:** [Video](../video-pipeline.md), [Dedup](../dedup-architecture.md), [Audit](../audit-and-safety.md), [Engineering log](../engineering-log.md). Các con số benchmark là lịch sử nếu thiếu phép đo mới.

## Script, benchmark và test

- [`scripts/`](../../scripts): `batch_translate_all.py` điều phối dịch tuần tự; `set_track_defaults.py` đổi default track MKV; `organize_vietnamese_subs.py` sắp xếp phụ đề; `upscale_x4_single.py` tiện ích upscale riêng. Đọc tham số/đường dẫn và side effect trước chạy.
- [`benchmarks/README.md`](../../benchmarks/README.md): nhóm `upscale/` đo model, dedup, latency và chaos resume; `subtrans/` đánh giá pronoun; `research/` hiệu chuẩn DIS/so sánh hình ảnh. Không thuộc test mặc định và không phải lệnh bootstrap.
- [`pytest.ini`](../../pytest.ini): thu thập `tests/` và `subtrans/tests/`. Video tests chia nhóm dedup/GOP, resume/storage, model/color, mux; subtitle tests chia parser/writer/glossary, API stub/quota/cache, cascade/RAG/eval.
- `-m "not gpu"` loại test gắn marker GPU, nhưng vài test không gắn marker vẫn tự chọn CUDA. Chỉ một tiến trình test, kiểm tra production idle và cách ly dữ liệu theo [VALIDATION](VALIDATION.md).
- Test mô phỏng/unit PASS không thay thế nghiệm thu video thực, frame count hay audit trên tập thành phẩm. Phạm vi kiểm chứng hiện tại phải được ghi trong [STATE](STATE.md).
