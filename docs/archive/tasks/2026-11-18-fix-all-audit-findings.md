# Handoff: Khắc phục toàn bộ điểm yếu kiểm định P0, P1, P2

Tạo khi bắt đầu tác vụ nhiều bước; cập nhật trong lúc làm, trước compact/bàn giao/dừng. Mỗi tác vụ một file.

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-18 -> 2026-11-20 (+07:00)
- Nguồn yêu cầu và mục tiêu: Yêu cầu trực tiếp từ Người dùng (`/swarm /browser khắc phục TOÀN BỘ các vấn đề đã xác minh trong báo cáo kiểm định ở trên và kiểm tra lại sau khi đã hoàn thành để đảm bảo`). Vá toàn bộ vấn đề P0, P1, P2 đã xác minh trên Video Core (`upscale.py`, `src/`), Subtrans + Scripts (`subtrans/`, `scripts/`), Web Studio (`web_gui.py`, `web/`, `web/static/`) và tests/docs; kiểm định lại toàn diện bằng `ruff.exe`, `git diff --check`, `agent_context.py --check`, `scripts/verify_offline.py` và live Chrome browser (`/browser`).
- Kết quả cần bàn giao:
  - Vá toàn bộ nhóm lỗi P0, P1, P2 kèm bộ unit/regression tests mới (+42 ca).
  - `.\.venv\Scripts\ruff.exe check upscale.py subtrans.py web_gui.py src subtrans web scripts tests` đạt `0` lỗi.
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py` đạt `100% PASS` (`0` failed).
  - Live Web Studio verification (`1440x900`, `390x844`, `320x568`) qua `/browser` xác nhận toàn bộ sửa lỗi UI/UX/API hoạt động đúng, `0` lỗi console, tiến trình server nền đã dừng sạch.
  - Cập nhật `docs/agents/STATE.md` và commit Git chi tiết theo `AGENTS.md` §5.
- Được phép / chưa được phép: Được phép sửa mã nguồn, test, tài liệu; không chạy Gemini API thật, không chạy tác vụ GPU dài (>100 frames), không sửa/xóa `.env`, `.temp_*`, media production.
- Ràng buộc quan trọng: 100% cục bộ Windows 11; tuân thủ Gemini daytime guard; bảo toàn bất biến `animevideov3`, `hevc_nvenc` `main10`/`p010le`/`p6`/`CQ 18`/`maxrate 45M`/`fullres`/`lookahead 32`/Spatial+Temporal AQ, BT.709 hai chiều, `-max_interleave_delta 0`, CSP `sha256`, 0 `.innerHTML`.
- Trạng thái: hoàn tất

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, `main`, `5271cd2`
- File đã sửa (54 file sửa + 3 file mới, kèm cập nhật `docs/agents/STATE.md`):
  - Video Core & CLI (`impl-video`): `upscale.py`, `requirements.txt`, `src/build_trt_engine.py`, `src/cleanup.py`, `src/config.py`, `src/cuda_engine.py`, `src/dedup.py`, `src/ffmpeg_ops.py`, `src/logger.py`, `src/models_resolver.py`, `src/mux_audit.py`, `src/night_report.py`, `src/pipeline.py`, `src/ramdisk.py`, `src/scheduler.py`, `src/subtitles.py`, `src/trt_zero_copy_upscaler.py`, `tests/test_e2e_integration_t3_t4.py`, `tests/test_audit_fixes_video.py` (mới).
  - Subtrans, Scripts & Docs (`impl-subtrans`): `subtrans/audio_diarization.py`, `subtrans/cache.py`, `subtrans/cascade.py`, `subtrans/cast_sheet.py`, `subtrans/cli.py`, `subtrans/config.py`, `subtrans/gemini_backend.py`, `subtrans/glossary.yaml`, `subtrans/linter.py`, `subtrans/parser.py`, `subtrans/pipeline.py`, `subtrans/rag.py`, `subtrans/report.py`, `subtrans/throttle.py`, `subtrans/writer.py`, `glossaries/README.md`, `scripts/agent_context.py`, `scripts/batch_translate_all.py`, `scripts/benchmark_pipeline.py`, `scripts/embed_fate_subtitles.py`, `scripts/organize_vietnamese_subs.py`, `scripts/set_track_defaults.py`, `scripts/upscale_x4_single.py`, `subtrans/tests/test_audit_fixes_subtrans.py` (mới).
  - Web Studio (`impl-web`): `web_gui.py`, `web/app.py`, `web/fs_browser.py`, `web/job_manager.py`, `web/sse_hub.py`, `web/static/index.html`, `web/static/styles.css`, `web/static/js/app.js`, `web/static/js/charts.js`, `web/static/js/compare_stage.js`, `web/static/js/panels.js`, `web/static/js/ui.js`, `tests/test_web_gui.py`.
- Chủ sở hữu từng phần / subagent / kết quả đã nhận: `impl-video` (#29, #31), `impl-subtrans` (#24), `impl-web` (#20, #22, #37), `impl-verifier` (#30, #34, #38) — đã hoàn tất và giải phóng (`kill`); `browser` (`ff7d6feb-8df4-403a-a43c-7a958b4fed0b`) xác minh sống 2 vòng PASS.
- Tiến trình/session đang chạy và nơi xem kết quả: `0` tiến trình nền đang chạy (`web_gui.py` task-442 đã dừng sạch).
- File dữ liệu production cần giữ: `out/.temp_FateZero - S01E10 - Rin's Adventure`, `E:\Films\Raw_Sources`, `E:\Films\Upscaled_4K`, `subtrans/.cache/*.db`, `.cache/trt_engines/*.engine`, `.env` (nguyên vẹn 100%).

## Đã làm và bằng chứng
- **Video Core & CLI:** Bảo toàn `original_input_path` + `try/finally` xóa `.preview_clip_*`; sửa Zero-Disk `slot_drain_events` loop wait + `BrokenPipeError` khi `written <= 0` + xả `stderr` daemon + `PipeDrainWorker.abort()` + NaN/Inf guard; đồng bộ `BackgroundQueuePreflightManager` (`mark_main_started`/`wait_for_video`) + `.split_complete` + `plan.json` atomic; helper dùng chung `resolve_expected_4k_output_path`/`find_existing_4k_output_path` cho `_4K_animejanai.mkv`; mux staging `*.mux_tmp.mkv` chỉ `os.replace` khi `Mux Audit == PASS`; sửa `mux_audit.py` (clip không audio, đếm sub/event, MD5/frame `None`, ưu tiên demux packet); khóa `.gpu_active.lock` nguyên tử sớm + `UPSCALER_FORCE_WHISPER_CPU=1` cho `--async-subs`; ring buffer Open-GOP `deque(maxlen=n_discard + 1)`; `RAMDisk` `SIGINT -> KeyboardInterrupt`; `get_dedup_params()` hỗ trợ `dis-<float>`/`drift-<float>`; sidecar `.engine.sha256`; sửa `F821` trong `tests/test_e2e_integration_t3_t4.py`; cập nhật `night_report.py`, `logger.py`, `requirements.txt`.
- **Subtrans, Scripts & Docs:** Chặn vòng lặp `429 PerMinute` (`max_rpm_retries` + `stage_deadline`) + thêm `500/502/504`; sửa `StageAPrefetchWorker` + `.cast.yaml` atomic + dọn file tạm; clamp timestamp âm ở `parser.py` & `writer.py`; SQLite WAL `busy_timeout=10000` + `BEGIN IMMEDIATE`; khử log đúp `429 PerDay` + luân phiên key trong `cast_sheet.py`/`cascade.py`; chuyển runtime quota overlay sang `subtrans/.cache/models_runtime_quota.yaml` (không làm bẩn `subtrans/models.yaml`); sửa `--subs-upgrade` ưu tiên Flash & Cascade chọn Flash chưa cạn; bỏ `師匠 -> Sư phụ` toàn cục & lọc `error_bank`; chặn `--local` bằng lỗi thoát khác `0`; vector hóa `audio_diarization.py`; cập nhật `agent_context.py`, `set_track_defaults.py` (`main([])` thoát `2`), `benchmark_pipeline.py` (`zero_disk=True`), dọn đường dẫn cứng `C:`/`G:` trên các script phụ.
- **Web Studio & Live Browser:** Phát hiện thư mục đích/nguồn ngoài `-o` (`E:\Films\Upscaled_4K` & `E:\Films\Raw_Sources`) cho `/api/preview/candidates`, `/api/audits`, `/api/audits/detail` (trả HTTP `200`, vẫn chặn `403` ngoài phạm vi); offload `nvidia-smi`/I/O khỏi `asyncio` event loop bằng `asyncio.to_thread` + `_counter_lock`; ghim `logs/web_job.log` trong Web job; thêm FastAPI `lifespan` -> `TelemetryHub.shutdown()`; chuyển `--lan` sang mặc định chỉ đọc kèm `--lan-control`; chặn double-launch theo `live_status.json` PID sống; tăng tốc `preview-schedule` (parse YAML 1 lần, kiểm `already_done` trước PyAV + cache); sửa độ tương phản Canvas 2D Auto Dark Mode (`charts.js`), lưới `2x2`, scroll đáy `#live-log-pre`, defer `preview-schedule`, validate `blur` `#opt-deadline`, phản hồi YAML lỗi Tab 4, `autocomplete="off"`, wrap `.fs-entry-row` & `.wipe-slider-group` ở `320x568`, gỡ scroll trap `.table-wrapper` mobile, và tuân thủ `modern-web-guidance` (`tabs`, `dialog`, `responsive-table`, `dark-mode-toggle`, `component-specific-light-dark-theme`, `accessibility`, `forms`).
- **Lệnh kiểm tra & bằng chứng (`.venv\Scripts\python.exe`, Python 3.11.9):**
  - `.\.venv\Scripts\ruff.exe check upscale.py subtrans.py web_gui.py src subtrans web scripts tests` -> **exit 0 (`All checks passed!`)**
  - `git --no-optional-locks diff --check` -> **exit 0**
  - `.\.venv\Scripts\python.exe scripts/agent_context.py --check` -> **exit 0 (`PASS`, 0 errors, 0 warnings sau khi cập nhật `STATE.md`)**
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py` -> **exit 0: `803 passed, 14 skipped, 1 deselected` (`0 failed`) trong `266.88s`**, evidence `E:\ai-video-upscaler\scratch\agent-validation-s10t20cf` (+42 ca mới so với mốc `761`: +17 video, +21 subtrans, +4 web).
  - Live Browser (`http://127.0.0.1:8765/`, `1440x900`, `390x844`, `320x568`, non-destructive): **PASS toàn bộ**, `0` lỗi console JS/DOM, xác minh hình ảnh trên 9 ảnh chụp (gồm `step_68_shot_1_charts_dark_fresh.png`, `step_72_shot_1_fs_picker_320_fresh.png`, `step_17_shot_1.png`, `final_step_38.png`).
- **Mux/frame/fps:** N/A — không tạo thành phẩm video mới trong phiên sửa lỗi offline & Web UI.

## Bước tiếp theo
1. Toàn bộ tác vụ đã hoàn tất và sẵn sàng cho kế hoạch refactor tiếp theo nếu người dùng yêu cầu.
- Câu hỏi/approval còn chờ: Không.
- Rủi ro/giới hạn cần báo người dùng: Các nhánh GPU thật, biên dịch TensorRT engine, chạy `hevc_nvenc` thực tế và gọi Gemini API sống chưa chạy trong phiên này (theo quy định bảo vệ GPU & hạn mức ban ngày); khi chạy thực tế tập đầu tiên sau đợt vá này nên kiểm tra nhanh bằng `--preview 10`. Ngoài ra trình duyệt có thể giữ cache tĩnh đĩa cũ sau khi cập nhật `web/static/*` (cần Ctrl+F5 lần đầu).

## Kết thúc
- Kết quả và giới hạn đã thông báo: Báo cáo đầy đủ trong câu trả lời cuối cùng của phiên làm việc.
- Commit/tag: Tra `git log -n 1 -- docs/agents/tasks/2026-11-18-fix-all-audit-findings.md` sau commit hoàn tất.
- Tài liệu cập nhật: [STATE](../STATE.md), [AGENTS](../../../AGENTS.md)
