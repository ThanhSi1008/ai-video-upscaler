# Project Map (Codebase Navigation Index)

> **Updated:** 2026-11-21 | **Related:** [GEMINI](../../GEMINI.md) | [STATE](STATE.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md)

## 1. Entrypoints & Data Flows

- [`upscale.py`](../../upscale.py) / [`upscale.bat`](../../upscale.bat) / [`upscale_turbo.bat`](../../upscale_turbo.bat): CLI (`--zero-disk=True` default, `--no-zero-disk`/`--disk`, `--queue`, `--preview`, `--dedup-policy` incl. `sakuga`/`dis-sakuga` and dynamic `dis-<float>`/`drift-<float>`, `--subs-*`). `expand_video_inputs` natural-sorts folders.
- [`subtrans.py`](../../subtrans.py) / [`subtrans.bat`](../../subtrans.bat) -> [`subtrans/cli.py`](../../subtrans/cli.py): Typer CLI (`run`, `batch`, `eval`, `rag`, `check`, `models`). Read [subtrans-anime-ops](../../.agents/skills/subtrans-anime-ops/SKILL.md) before editing.
- [`web_gui.py`](../../web_gui.py) / [`web_gui.bat`](../../web_gui.bat) -> [`web/app.py`](../../web/app.py): Local Real-Time Web Studio (`http://127.0.0.1:8765`, `--mock` implies `--no-browser` & simulated job launches, `docs_url=None`, UNC path rejection, FastAPI + Uvicorn + SSE 1Hz), spawns subprocesses without holding static VRAM; bridges `logs/live_status.json`, `logs/web_job_<id>.log` & `logs/.graceful_pause.flag`.

```text
upscale.py:main -> pipeline.run_upscale_pipeline
  -> probe + model resolve + preflight (P-Core QoS + AntiSleep) + manifest
  -> GOP split + dedup plan (raw_actions_rle)
  -> 3-Stage Overlap:
      [Zero-Disk] NVDEC RGB24 pipe (N+1) || TRT 10 FP16 + Triton P010LE (N) || Async NVENC (N-1)
      [Fallback]  FFmpeg PNG extract (N+1) || TRT/CUDA FP16 + Hardlink Cel (N) || NVENC + cleanup (N-1)
  -> 1-Pass Concat-Mux (mux_segments_directly) -> Parallel Mux Audit (streamhash MD5) -> cleanup
  -> Stage B subtrans subprocess (sequential or --async-subs)

subtrans.cli -> TranslationPipeline
  -> parse/align -> Stage A cast + glossary -> Gemini 3-key translation/cache
  -> Linter V2 / Cascade repair -> playback writer -> QA audit / quality gate
```

## 2. Config File Disambiguation

| File | Scope | Role |
| :--- | :--- | :--- |
| [`models/models.yaml`](../../models/models.yaml) | Video SR | Super-Resolution model registry (`animevideov3` default, AnimeJaNai Native 2x, SPAN, SHA-256) |
| [`shows.yaml`](../../shows.yaml) | Video SR | Per-series/release-tag SR model & `dedup_policy` overrides |
| [`subtrans/models.yaml`](../../subtrans/models.yaml) | Subtrans LLM | Gemini Cloud model metadata (`gemini-3.5-flash-lite`, `gemini-3.8-flash`) & fallback `chain` |
| [`subtrans/config.yaml`](../../subtrans/config.yaml) | Subtrans LLM | Batch size, timeouts, RPM/RPD config, and translation defaults |
| [`subtrans/glossary.yaml`](../../subtrans/glossary.yaml) | Subtrans Glossary | Canonical base Schema v2 glossary (230+ anime terms, `style_rules`, `address_rules`) |
| [`glossaries/base.yaml`](../../glossaries/base.yaml) | Subtrans Glossary | Cross-platform text pointer (`100644`) aliasing `../subtrans/glossary.yaml` |

## 3. Video Core (`src/`)

| Module | Key Responsibilities & Symbols |
| :--- | :--- |
| [`config.py`](../../src/config.py) | Runtime constants, NVMe JIT cache routing, `configure_cuda_fast_math`, `get_hevc_rate_args`, `PENDING_SUBS_FILE` (`state/pending_subs.json`), PID `.gpu_active.lock` |
| [`env_isolation.py`](../../src/env_isolation.py) | `configure_e_drive_isolation`: routes `PATH`, caches (`HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, etc.), and `TEMP`/`TMP`/`TMPDIR` onto E: drive |
| [`pipeline.py`](../../src/pipeline.py) | `run_upscale_pipeline`, `FailedResumableError`, 3-stage overlap, Background Cel on `dec_exec`, resume frame check, Zero-Disk & 1-Pass Mux |
| [`ffmpeg_ops.py`](../../src/ffmpeg_ops.py) | PyAV probing + FFmpeg: `probe_video`, `split_video_segments`, `_count_leading_discard_packets`, `extract_frames_raw_rgb24`, `mux_segments_directly` |
| [`models_resolver.py`](../../src/models_resolver.py) | `load_models_registry`, `match_show_pattern`, `resolve_model_for_video`, `verify_weights_sha256` |
| [`show_routing.py`](../../src/show_routing.py) | Stdlib-only `shows.yaml` matcher (`match_show_pattern`, `candidate_show_names`, `match_show_rule`) shared across `src/` and `web/` |
| [`cuda_engine.py`](../../src/cuda_engine.py) | `RealESRAnimeVideoV3` (8K 2-frame slicing), `AnimeJaNaiCompact2x`, `SpandrelModelWrapper`, `GPUPersistentWorker`, `get_gpu_worker` |
| [`ai_engine.py`](../../src/ai_engine.py) | `upscale_frames`: NCNN Vulkan fallback runner |
| [`trt_zero_copy_upscaler.py`](../../src/trt_zero_copy_upscaler.py) | `@triton.jit` `_nhwc_u8_to_nchw_fp16_norm_kernel` & `_rgb_to_p010le_triton_kernel`, `_PipeDrainWorker`, `DoubleBufferedPipeUpscaler` (0-alloc rings, Belady `OPT`) |
| [`build_trt_engine.py`](../../src/build_trt_engine.py) | `_get_trt_logger`, `resolve_trt_profile_max_batch`, ONNX export, `TRTRuntimeEngineWrapper` (2-slot `_out_ring`, `execute_async_v3`) |
| [`dedup.py`](../../src/dedup.py) | `DEDUP_POLICY_PARAMS`, `deduplicate_cyclic_frames` (`BLAKE2b` + SIMD), `raw_actions_rle`, 1-pass `mpdecimate+DIS`, `run_preflight_worker_cli` |
| [`output_sanity.py`](../../src/output_sanity.py) | `check_frame_sanity_arrays`, `check_frame_sanity`, `OutputSanityError`, `write_failed_sanity_audit` ($|\Delta Y| \le 6/255$, $r \ge 0.98$) |
| [`mux_audit.py`](../../src/mux_audit.py) | `verify_mux_integrity`, `get_all_audio_streams_md5` (in-process PyAV demux + `-f streamhash` fallback), parallel `check_window_interleaving` |
| [`ramdisk.py`](../../src/ramdisk.py) | `RAMDisk` (physical RAM disk / `scratch/rolling_buffer` on NVMe Gen4), `cleanup_orphan_ramdisks` |
| [`cleanup.py`](../../src/cleanup.py) | `cleanup_checkpoint_dir`, `cleanup_subtitle_extracts`, `organize_output_artifacts`, `clean_all_temporary_data` |
| [`preflight.py`](../../src/preflight.py) | `AntiSleepContext`, `configure_alder_lake_pcore_qos`, AC power & disk checks |
| [`scheduler.py`](../../src/scheduler.py) | `BackgroundQueuePreflightManager`, `compute_overnight_schedule`, `check_gemini_quota_for_queue` |
| [`logger.py`](../../src/logger.py) | `RunLogger`, atomic `logs/live_status.json`, `logs/.graceful_pause.flag` check |
| [`night_report.py`](../../src/night_report.py) | `parse_telemetry_window`, `parse_episode_chunks_log`, `query_subtrans_api_stats`, `generate_night_report_text` |
| [`subtitles.py`](../../src/subtitles.py) | `find_or_extract_subtitles`, `transcribe_japanese_audio` (`faster-whisper` + 5.1 FC mono + Silero VAD + Glossary prompt) |
| [`pending_queue.py`](../../src/pending_queue.py) | Stdlib-only queue manager for `state/pending_subs.json` (with legacy root fallback; avoids importing `torch`) |

## 4. Web Studio (`web/`) & Subtrans (`subtrans/`)

- **`web/`:** [`app.py`](../../web/app.py) (REST/SSE/compare/audit/config endpoints), [`sse_hub.py`](../../web/sse_hub.py) (1Hz telemetry), [`job_manager.py`](../../web/job_manager.py) (subprocess queue, `.graceful_pause.flag`), [`fs_browser.py`](../../web/fs_browser.py) (`SafeLocalFSBrowser`), [`static/`](../../web/static) (Vanilla HTML/CSS/JS, `oklch()`, strict CSP `sha256`, zero `innerHTML`).
- **`subtrans/`:** `pipeline.py`, `cli.py`, `config.py`, `models.py`, `parser.py`, `dtw_auto_align.py`, `cast_sheet.py`, `dynamic_cast.py`, `audio_diarization.py`, `ja_morph_tagger.py`, `glossary.py`, `prompts.py`, `gemini_backend.py`, `translator.py`, `throttle.py`, `cache.py`, `linter.py`, `validation.py`, `cascade.py`, `rag.py`, `writer.py`, `report.py`, and `eval/`. See [subtrans-anime-ops](../../.agents/skills/subtrans-anime-ops/SKILL.md) and [glossaries/README.md](../../glossaries/README.md).

## 5. Tooling, Benchmarks, Tests & Archives

- **[`.agents/`](../../.agents):** `rules/` (glob-scoped `video-gpu.md`, `subtrans.md`, `web-ui.md`), `agents/` (`repo-scout.md`, `test-verifier.md`, `subtrans-evaluator.md` with `inheritMcp: false`), `skills/` (`agent-handoff-commit`, `safe-offline-test`, `video-upscale-ops`, `subtrans-anime-ops`, pruned 39-guide `modern-web-guidance`), `plugins/upscaler-guard/` (`SessionStart`, `PreToolUse`, `PostToolUse`, `Stop`), `plugins/upscaler-mcp/` (`disabled: true`).
- **[`scripts/`](../../scripts):** `agent_context.py` (`--pack`/`--symbols`/`--slice`/`--impact`/`--install-hooks`), `export_review_bundle.py` (`--sync-latest` 6 bundles incl. `--focus full`), `verify_offline.py` (`--target`/`--impacted`), `smoke_test_pipeline.py` (`< 5s` 24f CPU pipeline smoke test), `silicon_roofline_probe.py`, `telemetry_collector.py`, `benchmark_pipeline.py`, `compare_benchmarks.py`, `batch_translate_all.py`, `set_track_defaults.py`, `upscale_x4_single.py`, `unlock_rtx3060_admin.ps1`, `unlock_rtx3060_admin.bat`, and `scripts/oneoff/` (`embed_fate_subtitles.py`, `organize_vietnamese_subs.py`).
- **[`benchmarks/README.md`](../../benchmarks/README.md):** `upscale/`, `subtrans/`, `research/`, and summary reports.
- **Tests (`pytest.ini` markers: `unit`, `subtrans`, `e2e`, `gpu`, `web`, `smoke`):** `tests/` (unit/capability suites incl. `test_architecture_boundaries.py`, `test_export_review_bundle.py`, `test_smoke_pipeline.py`, `test_web_gui.py`, `test_nvenc_config_and_vram.py`, `test_zero_disk_and_trt_export.py`, `test_cli_lock_and_pipeline_regressions.py`, `test_gop_rle_and_3stage_overlap.py`, `test_zero_disk_ring_buffers_and_cel.py`), `tests/e2e/` (Group A/B + Tier 1–4 suites), and `subtrans/tests/` (all subtrans tests consolidated).
- **Runtime & Archives (excluded via [`.antigravityignore`](../../.antigravityignore)):** `docs/archive/` (20 historical task logs + 7 legacy reports), `state/pending_subs.json`, `logs/`, `out/`, `scratch/` (incl. `scratch/review_bundles/`), `models_bench/`, `subtrans/.cache/`, `.cache/` (`.env` / `subtrans/.env` are locally readable, gitignored).
