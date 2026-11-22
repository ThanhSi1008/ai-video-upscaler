# Project Map (Codebase Navigation Index)

> **Updated:** 2026-11-22 | **Related:** [GEMINI](../../GEMINI.md) | [STATE](STATE.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md)

## 1. Entrypoints & Data Flows

- [`upscale.py`](../../upscale.py) / [`upscale.bat`](../../upscale.bat) / [`upscale_turbo.bat`](../../upscale_turbo.bat): CLI (`--zero-disk=True` default, `--no-zero-disk`/`--disk`, `--model` incl. `auto-pareto`, `--nvenc-multipass auto|fullres|qres`, `--queue`, `--preview`, `--dedup-policy` incl. `dis-auto`/`sakuga`/`dis-sakuga` and dynamic `dis-auto-<float>`/`dis-<float>`/`drift-<float>`, `--subs-*`). `expand_video_inputs` natural-sorts folders.
- [`subtrans.py`](../../subtrans.py) / [`subtrans.bat`](../../subtrans.bat) -> [`subtrans/cli.py`](../../subtrans/cli.py): Typer CLI (`run`, `batch`, `eval`, `rag`, `check`, `models`). Read [subtrans-anime-ops](../../.agents/skills/subtrans-anime-ops/SKILL.md) before editing.
- [`web_gui.py`](../../web_gui.py) / [`web_gui.bat`](../../web_gui.bat) -> [`web/app.py`](../../web/app.py): Local Real-Time Web Studio (`http://127.0.0.1:8765`, `--mock` implies `--no-browser` & simulated job launches, `docs_url=None`, UNC path rejection, FastAPI + Uvicorn + SSE 1Hz), spawns subprocesses without holding static VRAM; bridges `logs/live_status.json`, `logs/web_job_<id>.log` (`N=20` retention) & `logs/.graceful_pause.flag`, and cleans `scratch/web_compare/*.png` in `_lifespan`.

```text
upscale.py:main -> pipeline.run_upscale_pipeline
  -> probe + model resolve (incl. auto-pareto) + preflight (P-Core QoS + AntiSleep) + manifest
  -> GOP split (24s super-chunks in CUDA Zero-Disk) + RollingDedupPlanner (max_workers=2) / raw_actions_rle
  -> 3-Stage Overlap:
      [Zero-Disk] FFmpeg p010le/nv12 pipe (N+1) || Triton YUV2RGB + TRT 10 FP16 + Triton P010LE (N) || PyAV/Pipe NVENC (N-1)
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
| [`config.py`](../../src/config.py) | PEP 562 lazy `DEFAULT_ENGINE`/`DEVICE_NAME`/`DEFAULT_BATCH_SIZE`/`HEVC_ENCODER`, `VALID_NVENC_MULTIPASS_POLICIES`, NVMe JIT cache routing, `configure_cuda_fast_math`, `resolve_episode_nvenc_multipass`, `get_hevc_rate_args`, `PENDING_SUBS_FILE` (`state/pending_subs.json`), ownership-gated `_atexit_release_gpu_lock` |
| [`env_isolation.py`](../../src/env_isolation.py) | `configure_e_drive_isolation`: routes `PATH`, caches (`HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, etc.), and `TEMP`/`TMP`/`TMPDIR` onto E: drive |
| [`pipeline.py`](../../src/pipeline.py) | `run_upscale_pipeline`, `FailedResumableError`, `RollingDedupPlanner(..., sync_initial_chunks=3)` & `finalize_plan` (`episode_multipass`), 3-stage overlap, `24s` Zero-Disk super-chunks, `auto-pareto` scene-cut hysteresis, Zero-Disk & 1-Pass Mux |
| [`ffmpeg_ops.py`](../../src/ffmpeg_ops.py) | PyAV probing + FFmpeg: `probe_video`, `split_video_segments`, `_count_leading_discard_packets`, `_recover_open_gop_leading_frames`, `extract_frames_raw_yuv420`, `yuv420_to_rgb_small_u8`, `mux_segments_directly` |
| [`models_resolver.py`](../../src/models_resolver.py) | `load_models_registry`, `AUTO_PARETO_MODEL_ID`, `resolve_model_for_video`, `verify_weights_sha256` |
| [`show_routing.py`](../../src/show_routing.py) | Stdlib-only `shows.yaml` matcher (`match_show_pattern`, `candidate_show_names`, `match_show_rule`) shared across `src/` and `web/` |
| [`cuda_engine.py`](../../src/cuda_engine.py) | `build_collapsed_4x_to_2x_buffers`, `is_srvgg_compact_model`, `RealESRAnimeVideoV3` (Catmull-Rom collapsed 2x), `AnimeJaNaiCompact2x`, `SpandrelModelWrapper`, `GPUPersistentWorker` (`weights_only=True`) |
| [`ai_engine.py`](../../src/ai_engine.py) | `upscale_frames`: NCNN Vulkan fallback runner |
| [`trt_zero_copy_upscaler.py`](../../src/trt_zero_copy_upscaler.py) | `@triton.jit` `_yuv420_to_nchw_fp16_bt709_kernel` & `_rgb_to_p010le_triton_kernel`, `_bound_live_cyclic_memo`, `spawn_nvenc_pipe_writer(..., chroma_location="left")`, `_InProcessPyAVNVENCEncoder`, `DoubleBufferedPipeUpscaler` |
| [`build_trt_engine.py`](../../src/build_trt_engine.py) | `_PureFP16TRTExportModule` (`_aa2x`), `resolve_trt_profile_max_batch`, `verify_trt_engine_sha256_sidecar`, ONNX export, `TRTRuntimeEngineWrapper` (2-slot `_out_ring`, `execute_async_v3`) |
| [`dedup.py`](../../src/dedup.py) | `DEDUP_POLICY_PARAMS` (`dis-auto`, `sakuga`, `dis-sakuga`), `_check_8x8_lip_flap_gate`, `_estimate_textured_lk_pan_zoom`, `RollingDedupPlanner`, `encode_chunk_reexpanded(..., multipass=...)`, `deduplicate_cyclic_frames`, `raw_actions_rle` |
| [`output_sanity.py`](../../src/output_sanity.py) | `check_frame_sanity_arrays`, `check_frame_sanity`, `OutputSanityError`, `write_failed_sanity_audit` ($|\Delta Y| \le 6/255$, $r \ge 0.98$) |
| [`mux_audit.py`](../../src/mux_audit.py) | `verify_mux_integrity` (incl. 10-bit & BT.709 `tv` `video_vui_audit`), `get_all_audio_streams_md5` (PyAV demux + `-f streamhash` fallback), parallel `check_window_interleaving` |
| [`ramdisk.py`](../../src/ramdisk.py) | `RAMDisk` (physical RAM disk / `scratch/rolling_buffer` on NVMe Gen4), `cleanup_orphan_ramdisks` |
| [`cleanup.py`](../../src/cleanup.py) | `cleanup_checkpoint_dir`, `cleanup_subtitle_extracts`, `organize_output_artifacts`, `clean_all_temporary_data` |
| [`preflight.py`](../../src/preflight.py) | `AntiSleepContext`, `configure_alder_lake_pcore_qos`, AC power & disk checks |
| [`scheduler.py`](../../src/scheduler.py) | `BackgroundQueuePreflightManager`, `compute_overnight_schedule`, `check_gemini_quota_for_queue` |
| [`logger.py`](../../src/logger.py) | `RunLogger`, atomic `logs/live_status.json`, `logs/.graceful_pause.flag` check |
| [`night_report.py`](../../src/night_report.py) | `parse_telemetry_window` (legacy + Windows 17-col `TelemetryCollector` CSV), `parse_episode_chunks_log`, `query_subtrans_api_stats`, `generate_night_report_text` |
| [`subtitles.py`](../../src/subtitles.py) | `find_or_extract_subtitles`, `transcribe_japanese_audio` (`faster-whisper` + 5.1 FC mono + Silero VAD + Glossary prompt) |
| [`pending_queue.py`](../../src/pending_queue.py) | Stdlib-only queue manager for `state/pending_subs.json` (with legacy root fallback; avoids importing `torch`) |

## 4. Web Studio (`web/`) & Subtrans (`subtrans/`)

- **`web/`:** [`app.py`](../../web/app.py) (REST/SSE/compare/audit/config endpoints, `/api/audits` 1.5s stat-guarded TTL cache, `_resolve_subtrans_key_slots`/`_apply_subtrans_key_edits`/`_DOTENV_LOCK`, `_lifespan` `scratch/web_compare/*.png` cleanup), [`sse_hub.py`](../../web/sse_hub.py) (1Hz O(1) `_poll_tick_sync`, `_sanitize_json_payload` `allow_nan=False`, `_coerce_pid`), [`job_manager.py`](../../web/job_manager.py) (subprocess queue, `_prune_old_web_job_logs`, `_find_cached_plan(..., expected_policy=...)`, `.graceful_pause.flag`), [`fs_browser.py`](../../web/fs_browser.py) (`SafeLocalFSBrowser`), [`static/`](../../web/static) (Vanilla HTML/CSS/JS, hex tokens + `light-dark()` + `@media (prefers-color-scheme: dark)`, `auto-pareto`/`#opt-nvenc-multipass`/`24s` UI, strict CSP `sha256`, zero `innerHTML`).
- **`subtrans/`:** `pipeline.py` (`piecewise_ja_offsets` in `pending_subs.json`), `cli.py` (`batch` parity, `--review`/`--yes`/`ollama_*` removed), `config.py`, `models.py`, `parser.py` (banded `detect_auto_offset`), `dtw_auto_align.py` (windowed Sakoe-Chiba rolling DTW), `cast_sheet.py`, `dynamic_cast.py`, `audio_diarization.py`, `ja_morph_tagger.py`, `glossary.py`, `prompts.py`, `gemini_backend.py`, `translator.py`, `throttle.py` (`get_shared_rate_limiter`, `clear_rate_limiter_registry`), `cache.py` (`compute_cue_cache_key` conditional `gloss_skill_names`/`en_only_source`), `linter.py`, `validation.py`, `cascade.py` (dual Flash+Lite keys), `rag.py` (FTS5 purge), `writer.py`, `report.py`, and `eval/runner.py` (`verify_pronoun_cue` via `HysteresisCastManager` + `lint_cue`). See [subtrans-anime-ops](../../.agents/skills/subtrans-anime-ops/SKILL.md) and [glossaries/README.md](../../glossaries/README.md).

## 5. Tooling, Benchmarks, Tests & Archives

- **[`.agents/`](../../.agents):** `rules/` (glob-scoped `video-gpu.md`, `subtrans.md`, `web-ui.md`), `agents/` (`repo-scout.md`, `test-verifier.md`, `subtrans-evaluator.md` with `inheritMcp: false`), `skills/` (`agent-handoff-commit`, `safe-offline-test`, `video-upscale-ops`, `subtrans-anime-ops`, pruned 39-guide `modern-web-guidance`), `plugins/upscaler-guard/` (`SessionStart`, `PreToolUse`, `PostToolUse`, `Stop`), `plugins/upscaler-mcp/` (`disabled: true`).
- **[`scripts/`](../../scripts):** `agent_context.py` (`--check`/`--pack`/`--symbols`/`--slice`/`--impact`/`--install-hooks`, 10 `CROSS_BOUNDARY_CONTRACTS`), `export_review_bundle.py` (`--sync-latest` 6 bundles incl. `--focus full`), `verify_offline.py` (`--target`/`--impacted`, 6-shard LPT + Win11 P-core QoS), `smoke_test_pipeline.py` (`< 5s` 24f CPU pipeline smoke test), `silicon_roofline_probe.py`, `telemetry_collector.py`, `benchmark_pipeline.py`, `compare_benchmarks.py`, `batch_translate_all.py` (`--source-dir`/`-o`/`--state-file`/`--dry-run`), `set_track_defaults.py`, `upscale_x4_single.py`, `unlock_rtx3060_admin.ps1`, `unlock_rtx3060_admin.bat`, and `scripts/oneoff/` (`embed_fate_subtitles.py`, `organize_vietnamese_subs.py`).
- **[`benchmarks/README.md`](../../benchmarks/README.md):** `upscale/`, `subtrans/`, `research/`, and summary reports.
- **Tests (`pytest.ini` markers: `unit`, `subtrans`, `e2e`, `gpu`, `web`, `smoke`):** `tests/` (unit/capability suites incl. `test_e2e_acceptance_p0_p2.py`, `test_pareto_o1_o6.py`, `test_architecture_boundaries.py`, `test_export_review_bundle.py`, `test_smoke_pipeline.py`, `test_web_gui.py`, `test_allow_partial.py`, `test_dedup_policy.py`, `test_enospc_backpressure_retry.py`, `test_ffmpeg_ops.py`, `test_nvenc_config_and_vram.py`, `test_zero_disk_and_trt_export.py`, `test_cli_lock_and_pipeline_regressions.py`, `test_gop_rle_and_3stage_overlap.py`, `test_zero_disk_ring_buffers_and_cel.py`), `tests/e2e/` (Group A/B + Tier 1–4 suites), and `subtrans/tests/` (`test_cli_batch.py`, `test_throttle.py`, `test_eval_runner.py`, `test_cache.py`, `test_cascade.py`, etc.).
- **Runtime & Archives (excluded via [`.antigravityignore`](../../.antigravityignore)):** `docs/archive/` (20 historical task logs + 7 legacy reports), `state/pending_subs.json`, `logs/`, `out/`, `scratch/` (incl. `scratch/review_bundles/`), `models_bench/`, `subtrans/.cache/`, `.cache/` (`.env` / `subtrans/.env` are locally readable, gitignored).
