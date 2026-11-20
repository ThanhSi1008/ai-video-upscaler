# Handoff: Xây dựng Web GUI thời gian thực & tích hợp `modern-web-guidance`

Tạo khi bắt đầu tác vụ nhiều bước; cập nhật trong lúc làm, trước compact/bàn giao/dừng. Mỗi tác vụ một file.

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-15 (Asia/Ho_Chi_Minh, UTC+7)
- Nguồn yêu cầu và mục tiêu: Người dùng phê duyệt thực thi toàn bộ kế hoạch xây dựng Web GUI thời gian thực cho `e:\ai-video-upscaler` kèm cài đặt `npx modern-web-guidance@latest install`, với điều chỉnh trực tiếp: **bỏ ràng buộc "Zero-VRAM Local Studio Control Room" và bỏ "Khóa an toàn API Key & Quota Ban Ngày", để người dùng sử dụng toàn bộ tính năng một cách bình thường** (dịch phụ đề, quản lý cấu hình/API key cục bộ, chạy hàng đợi/preview bất kỳ lúc nào mà không bị khóa theo giờ).
- Kết quả cần bàn giao:
  1. Cài đặt Node.js LTS & `npx modern-web-guidance@latest install` trong workspace, đạt `0` lỗi khi quét `legacy-scan`.
  2. Cầu nối viễn trắc thời gian thực `logs/live_status.json` + dừng êm `logs/.graceful_pause.flag` trong `src/logger.py` & `src/pipeline.py` (kèm `UPSCALER_NO_AUTO_OPEN=1` trong `upscale.py`).
  3. Backend `web_gui.py`, `web_gui.bat` (CRLF), `web/` (`FastAPI` + `Uvicorn`, SSE 1Hz phần cứng/pipeline/log/queue, duyệt file/thư mục nội bộ, điều khiển chạy queue/preview/subtrans bình thường mọi khung giờ, quản lý `shows.yaml`, đối chiếu khung hình `1080p vs 4K`, thống kê Subtrans & Audit).
  4. Giao diện Web hiện đại `web/static/` tuân thủ `modern-web-guidance` (5 tab trực quan: Live Monitor, Queue & Scheduler, Preview Split-Compare, Models & `shows.yaml`, Subtrans & Audit Vault).
  5. Bộ kiểm thử tự động `tests/test_web_gui.py`, cập nhật `scripts/verify_offline.py` & `scripts/agent_context.py`, cập nhật tài liệu dự án để bàn giao kiểm chứng trình duyệt & commit Git.
- Được phép / chưa được phép:
  - Được phép: Cài đặt Node.js LTS, `modern-web-guidance`, `fastapi`, `uvicorn`; sửa/thêm code trong `src/`, `upscale.py`, `web/`, `web_gui.py`, `web_gui.bat`, `scripts/`, `tests/`, `docs/`, `AGENTS.md`, `README.md`.
  - Ràng buộc Agent khi kiểm thử (`AGENTS.md` §5): Agent không in/commit `.env`, không gọi Gemini API thật trong bài test tự động của Agent (test dùng mock/`--dry-run` trong `tmp_path`). Trên ứng dụng Web GUI dành cho người dùng, không áp đặt khóa ban ngày 14:00 hay khóa cấu hình để người dùng vận hành bình thường.
- Ràng buộc quan trọng: Giữ nguyên 100% bất biến chất lượng video/mux (`AGENTS.md` §4), `.bat` chuẩn CRLF, `AGENTS.md <= 12 KiB`.
- Trạng thái: hoàn thành.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, nhánh `main`, `HEAD = 2196175`.
- File đã sửa trong lượt hoàn thiện hậu kiểm (`DeepInvestigator` + vòng rà soát độc lập, chưa commit để điều phối viên soát và commit):
  - `upscale.py`, `src/logger.py`
  - `web_gui.py`, `web/app.py`, `web/fs_browser.py`, `web/job_manager.py`, `web/sse_hub.py`
  - `web/static/index.html`, `web/static/styles.css`, `web/static/js/{ui.js,charts.js,compare_stage.js,panels.js,app.js}`
  - `tests/test_web_gui.py`, `docs/agents/tasks/2026-11-15-realtime-web-gui-modern-web-guidance.md`
- Chủ sở hữu từng phần / subagent / kết quả đã nhận: Đã hoàn tất sửa lỗi 5 nhóm từ `DeepInvestigator`, vá thêm các lỗi hồi quy phát hiện qua rà soát độc lập + Chrome CDP thực tế; giữ nguyên uncommitted cho agent điều phối (`parent`) rà soát và commit Git.
- Tiến trình/session đang chạy và nơi xem kết quả: Không còn tiến trình nền đang chạy; thư mục kiểm định Headless Chrome tạm đã dọn sạch.
- File dữ liệu production cần giữ: `.env`, `subtrans/.cache/`, `models_bench/`, `out/`.

## Đã làm và bằng chứng
- Thay đổi vòng đầu & vòng hoàn thiện hậu kiểm (`DeepInvestigator`):
  - Cài đặt Node.js LTS, `modern-web-guidance` skill, `fastapi` và `uvicorn`.
  - Bổ sung cầu nối `logs/live_status.json` + `logs/.graceful_pause.flag` (`src/logger.py`, `src/pipeline.py`, `upscale.py`), thêm `--glossary` và vòng lặp đa tập cho `--subs-only` cùng cập nhật Stage B `"subtrans"` -> `"completed"` trong `upscale.py`.
  - `.env` API Keys & Subtrans (`web/app.py`, `web/static/index.html`, `web/static/js/panels.js`): chặn CR/LF/NUL trong `_write_dotenv_keys` và Pydantic validator, khử trùng lặp nhiều dòng cùng key trong `.env`, đồng bộ/xóa `GEMINI_API_KEY` + tách `GEMINI_API_KEYS` khi sửa/xóa slot đồng thời bảo toàn key thứ 4+ (`GEMINI_API_KEY_4...`), **không** chép `.env` key vào `os.environ` tiến trình Web Studio (tránh gắn nhầm `[ENV]` và làm mất nút `✕ Xóa` của các khe còn lại), chuẩn hóa ưu tiên `os.environ` > workspace `.env` (`override=False`) + phân tách `,`/`;` và `env_name` chính xác cho badge `[LIST]`, bổ sung nút `✕ Xóa` từng khe và checkbox `#clear-api-key-1..3` trên UI.
  - Bảo mật cục bộ & xác thực đầu vào (`web/app.py`, `web_gui.py`): bổ sung middleware chống DNS-rebinding `Host`/`Origin` (`allow_lan=False` chỉ cho loopback; `allow_lan=True` cho IP riêng/`.local`/tên miền LAN đơn nhãn nhưng không khóa duyệt media/compare đa ổ đĩa `C:`/`D:`/`E:` ngoài workspace), áp đặt CSP nghiêm ngặt (`style-src 'self'`, `script-src 'self' 'sha256-...'`, `img-src 'self' data:`), handler `RequestValidationError` khử `NaN`/`Infinity` tránh HTTP `500`, chặn `\x00` trên query/path (HTTP `400`/`422`), trả `422` khi trích xuất frame đối chiếu thất bại trên tệp video lỗi, hiển thị rõ `.audit.json` hỏng (`status: "CORRUPT"`, `corrupt: True`, `422 raw_preview`), và xử lý `.audit.json` không phải object (vd. `[1, 2, 3]`) thành HTTP `422 corrupt: True`.
  - `JobManager` (`web/job_manager.py`): giữ `self._lock` xuyên suốt kiểm tra + spawn + đăng ký chống TOCTOU, bọc `proc is None`/`stdout=None` vào `try...finally`, kiểm tra `subs_quality` & `deadline` (HTTP `400` kể cả `compute_schedule_preview` trước `split(':')`) trên mọi endpoint, truyền `--glossary` và hàng đợi đa tập trong `start_subtrans_job`, phân biệt `abort_job()` trả `status: "external_lock"` (`ok: False`) khi còn `.gpu_active.lock` ngoài, chuẩn hóa `-o` tương đối theo `workspace_root`, hỗ trợ đích `.mkv` tường minh và giữ hệ số `zero_disk=False` khi `use_tensorrt=False`.
  - `TelemetryHub`, `RunLogger`, SQLite URI & `shows.yaml` (`web/sse_hub.py`, `src/logger.py`, `web/app.py`, `web/fs_browser.py`): sửa phát hiện log `<96B` qua `_file_prefix_sig` + cờ `reset`, cô lập `sub_cursor` cho từng kết nối SSE đa tab, không mutate `TelemetryHub.history` trên `GET /api/status` khi `_broadcast_loop` đang chạy, xử lý đúng `status in {"failed", "error"}` & `aborted -> paused` & fallback `last_finished_job` (không bị `live_status.json` `completed` cũ che khuất lỗi mới), lưu `last_stage`/`failed_stage` trong `RunLogger.set_stage` trước khi chuyển `error`/`paused`, phát `completed` khi job `subtrans`/`subs_pending` kết thúc mã `0`, quét thư mục `.temp_*` theo `-o`, escape `#`/`%`/`?` cho SQLite RO URI bằng `Path.as_uri()`, khớp `shows.yaml` 2 lượt độc lập (`model` & `dedup_policy`) và chuẩn hóa chữ hoa/thường `override_dedup`.
  - Frontend (`web/static/`): sửa `charts.js` `priorSlots > 0 ? prevFps.slice(-priorSlots) : []` (tránh `slice(-0)` xóa mất mẫu đơn lẻ khi `arr.length === 1`), xóa `#compare-src-input` khi candidate không có `source_path`, giữ bước hoạt động cuối trên `#pipeline-stepper` khi `error`/`paused`/`subtrans`, xóa `logRing` khi đổi tệp log/`cur.reset`, guard `setSelectValueSafely` cho `<select>`, bổ sung kiểm tra `Number.isFinite` + `aria-invalid` + `.focus()` cho `optBatch`, `subOffset`, `compare-ts-input`, `compare-offset-input` cùng khóa nút submit trong `try...finally`, hiển thị chi tiết `a.corrupt && a.error` trên `#audits-tbody` kèm `.text-error` & `input[aria-invalid="true"]`, loại `td[colspan]` khỏi nhãn dọc mobile `<680px`, và bổ sung `.toast-item[data-tone="warn"]`.
- Lệnh kiểm tra vòng rà soát cuối (`HEAD = 2196175` + working tree):
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py` -> **`756 passed, 14 skipped, 1 deselected in 452.42s (0:07:32)`** (`exit 0`, gồm `22/22` ca trong `tests/test_web_gui.py`, có kiểm thử BT.709 `ffmpeg` `1920x1080` -> `3840x2160` và hồi quy toàn diện), evidence: `E:\ai-video-upscaler\scratch\agent-validation-hzq5gau7`.
  - `npx.cmd -y modern-web-guidance@latest legacy-scan web/static --json` -> **`totalFindings: 0`** (33 guides / 35 rules).
  - `ruff check upscale.py subtrans.py web_gui.py src subtrans scripts web tests/test_web_gui.py` -> **`All checks passed!`**.
  - Kiểm thử trình duyệt thực `chrome.exe --headless=new` qua WebSocket CDP + `uvicorn` cô lập trên desktop `1440×900` và mobile `390×844` (SSE 2 tab đồng thời + `<96B` log growth + đổi log, `#pipeline-stepper` `muxing -> error`, `TelemetryChartsController` `arr.length === 1`, `compare-frames` bằng MKV `ffmpeg` ngoài workspace + orphan clear + validation `aria-invalid`, split `shows.yaml` & `opt-batch-size`, `[LIST]` & `✕ Xóa` slot 1 giữ nguyên slots 2–3 ở trạng thái `dotenv` & key thứ 4+, `.audit.json` `CORRUPT` & `[1,2,3]` `422 raw_preview`, theme `light`/`dark`, `warn` toast, mobile `td[colspan]::before == "none"` & `scrollWidth == innerWidth == 390`, `--lan` `allow_lan=True`/`False`) -> **`ALL_CDP_E2E_CHECKS_PASSED`**, `0` console errors, `0` uncaught JS exceptions, `0` CSP violations.
  - `.\.venv\Scripts\python.exe scripts/agent_context.py --check` -> **`PASS`**; tất cả 5 file `*.bat` chuẩn `CRLF`; `AGENTS.md` = `12,205` bytes (`<= 12,288` bytes); `git diff --check` sạch lỗi khoảng trắng.
- Mux/frame/fps: N/A — không chạy suy luận GPU hay tạo video thành phẩm trong phiên này.

## Bước tiếp theo
1. Điều phối viên (`parent`) rà soát diff làm việc hiện tại và thực hiện `git commit` theo quy trình.

## Kết thúc
- Kết quả và giới hạn đã thông báo: Đã khắc phục toàn bộ lỗi hậu kiểm và lỗi bổ sung từ rà soát độc lập, vượt qua 756 ca kiểm thử offline (`PASS 100%`), `modern-web-guidance legacy-scan` (`0` lỗi), `ruff check`, và kiểm chứng Headless Chrome CDP thực tế ở cả desktop & mobile; chưa chạy upscale GPU dài, chưa bind socket `--lan` từ IP máy khác trong LAN, và chưa gọi Gemini API trực tuyến trong kiểm thử tự động.
- Commit/tag: Chưa commit (giữ nguyên thay đổi trên working tree để điều phối viên xử lý).
- Tài liệu cập nhật: [STATE](../STATE.md), [AGENTS](../../../AGENTS.md), [PROJECT_MAP](../PROJECT_MAP.md), [README](../../../README.md).
