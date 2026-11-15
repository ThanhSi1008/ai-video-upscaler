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
- Root, branch, baseline commit: `e:\ai-video-upscaler`, nhánh `main`, `HEAD = 705516a`.
- File đã sửa/tạo (chưa commit):
  - `requirements.txt`, `upscale.py`, `src/logger.py`, `src/pipeline.py`
  - `web_gui.py`, `web_gui.bat`, `web/__init__.py`, `web/app.py`, `web/sse_hub.py`, `web/job_manager.py`, `web/fs_browser.py`
  - `web/static/index.html`, `web/static/styles.css`, `web/static/js/{ui.js,charts.js,fs_picker.js,compare_stage.js,panels.js,app.js}`
  - `.agents/skills/modern-web-guidance/`, `skills-lock.json`
  - `scripts/agent_context.py`, `scripts/verify_offline.py`, `tests/test_web_gui.py`
  - `AGENTS.md`, `README.md`, `docs/agents/PROJECT_MAP.md`, `docs/agents/STATE.md`, `docs/agents/tasks/2026-11-15-realtime-web-gui-modern-web-guidance.md`
- Chủ sở hữu từng phần / subagent / kết quả đã nhận: Coding worker hoàn thiện toàn bộ mã nguồn, kiểm thử offline & tài liệu; Agent điều phối (`parent`) thực hiện kiểm tra trực quan `/browser` và commit Git cuối cùng.
- Tiến trình/session đang chạy và nơi xem kết quả: Không còn tiến trình nền đang chạy.
- File dữ liệu production cần giữ: `.env`, `subtrans/.cache/`, `models_bench/`, `out/`.

## Đã làm và bằng chứng
- Thay đổi:
  - Cài đặt Node.js LTS, `modern-web-guidance` skill, `fastapi` và `uvicorn`.
  - Bổ sung cầu nối `logs/live_status.json` (ghi nguyên tử) + cờ dừng êm `logs/.graceful_pause.flag` trong `src/logger.py` và tích hợp điểm dừng chunk an toàn trong cả hai nhánh Async/Zero-Disk của `src/pipeline.py` (đồng bộ `prev_c_idx = c_idx` khi dừng đồng bộ).
  - Xây dựng backend `web_gui.py`, `web_gui.bat` (chuẩn CRLF), `web/app.py`, `web/sse_hub.py`, `web/job_manager.py`, `web/fs_browser.py` không nạp `torch`/CUDA trong tiến trình Web (giữ trọn 12GB VRAM cho subprocess `upscale.py`, khởi động tức thì, có test đối chiếu đồng bộ `DEDUP_POLICY_PARAMS` và `match_show_pattern` với `src/`), kèm đồng bộ con trỏ byte `log_cursor` chống lặp/xé dòng log, cô lập `live_status.json` cũ khi chạy Subtrans/khởi động tập mới, `RunLogger.stage_guard()` ghi nhận lỗi/tạm dừng và phân loại đúng mã thoát `1` khi dừng êm `upscale.py`.
  - Xây dựng giao diện `web/static/` (`index.html`, `styles.css`, 6 ES modules trong `web/static/js/`) tuân thủ đầy đủ `modern-web-guidance` (`focusgroup="tablist nomemory"` + `aria-controls`, `hidden="until-found"` + `beforematch` kèm fallback boolean `hidden` khi thiếu `onbeforematch`, `.tabpanels-stack { align-items: start }`, `[role="tabpanel"][hidden] { pointer-events: none }`, `[role="tabpanel"]:not([hidden]) { position: relative; z-index: 1 }` chống che khuất con trỏ chuột trên mọi tab, `[hidden]:not([hidden="until-found"]) { display: none !important }`, `<dialog closedby="any">` cuộn an toàn ở viewport hẹp `320×256` + xóa sạch `returnValue = ""` + chốt chặn phiên/thứ tự bất đồng bộ `browseSeq`/`auditLoadSeq`/`dialog.open` + fallback `pointerdown`/`pointerup` chỉ nhận chuột trái `button === 0`, `popover="manual"` toast định vị cố định góc dưới phải `--toast-offset` + giới hạn `--toast-max-h` theo chiều cao màn hình + `ariaNotify`, bảo toàn tiêu điểm `#checkpoint-list` qua chu kỳ SSE 1Hz `lastCheckpointSignature`, FOUC-free `<meta name="color-scheme">` + `:has()`, IME-safe `isComposing || keyCode === 229`, căn tọa độ rèm kéo theo khung hình `drawX`/`drawW` + bọc an toàn `PointerCapture`, 0 `innerHTML`, security headers `nosniff`/`DENY`).
  - Cập nhật `web/app.py` để `/api/audits` kế thừa số khung hình và luồng từ `mux_audit.video_frame_count` khi cấp cao nhất bằng `0`, đồng thời tự động quét thêm các thư mục con cấp 1 hợp lệ trong `out/` cho cả Kho Kiểm định và Đối chiếu 4K.
  - Cập nhật `scripts/verify_offline.py` cho phép duy nhất `socket.socketpair` nội bộ của `asyncio.ProactorEventLoop` trên Windows trong khi vẫn khóa chặt mọi kết nối mạng/TCP bên ngoài.
- Quyết định đã chốt:
  - Không áp đặt khóa cứng dịch phụ đề sau 14:00 hay giới hạn vô hiệu hóa tác vụ trên Web GUI theo đúng chỉ đạo người dùng; cho phép vận hành bình thường 24/7 mọi tính năng.
  - Tách bản sao hằng số nhẹ `DEDUP_POLICY_PARAMS` & `match_show_pattern` sang `web/fs_browser.py` (kèm unit test đối soát 1:1 với `src/`) để tiến trình Web không kích hoạt `src.config` -> `torch._C._cuda_init()`.
- Lệnh kiểm tra, interpreter, số pass/fail/skip, thời gian, evidence path:
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py` -> **`750 passed, 14 skipped, 1 deselected in 457.36s`** (`exit 0`), evidence: `E:\ai-video-upscaler\scratch\agent-validation-6lm55hl1` (bao gồm `16/16` ca mới trong `tests/test_web_gui.py`, có kiểm thử hồi quy hit-testing `[role="tabpanel"][hidden]`/`returnValue`, reflow hộp thoại/toast, bất đồng bộ modal, `mux_audit.video_frame_count` & `out/<subdir>/`, và trích xuất PNG BT.709 thực bằng `ffmpeg` 10-bit HEVC).
  - `npx.cmd -y modern-web-guidance@latest legacy-scan web/static --json` -> **`totalFindings: 0`** (33 guides / 35 rules).
  - `ruff check upscale.py subtrans.py web_gui.py src subtrans scripts web tests/test_web_gui.py` -> **`All checks passed!`**.
  - `cmd.exe /c "web_gui.bat --help"` -> **`exit 0`**.
  - `.\.venv\Scripts\python.exe scripts/agent_context.py --check` -> **`PASS`**.
  - Kiểm thử trình duyệt thực Headless Chrome CDP (`Input.dispatchMouseEvent` tọa độ thực + `document.elementFromPoint` + `Input.dispatchKeyEvent`) + `uvicorn` trên không gian làm việc cô lập (cả 5 tab ở viewport `1440×980`, `580×860`, `320×256` WCAG 1.4.10 Reflow, 7 chu kỳ mở/duyệt ổ đĩa/chọn nhiều file/chọn cả thư mục/`Esc`/click backdrop `closedby="any"`/`Hủy`/`✕ Đóng` + đua bất đồng bộ giữa 2 phiên của `#fs-picker-dialog`, 3 chu kỳ mở/chuyển tài liệu `.json`/`.md`/`.yaml` có `aria-pressed` + đua bất đồng bộ của `#audit-detail-dialog`, bảo toàn focus nút `#checkpoint-list` qua SSE 1Hz, kéo rèm chuột thực trên `#compare-canvas`, xếp chồng 4 toast Top-Layer ở `320×256` nằm trọn trong viewport không đè nhau, giả lập trình duyệt thiếu `closedBy` & `onbeforematch` bỏ qua click chuột phải vào backdrop, `beforematch` Find-in-Page và `ArrowRight` chuyển tab) -> **`0` console errors, `0` uncaught JS exceptions**.
  - Kiểm chứng tương tác trình duyệt thực tế qua `/browser` trên `http://127.0.0.1:8765/` (`visited_origins: http://127.0.0.1:8765`, `0` console errors): xác nhận SSE 1.0 Hz phần cứng RTX 3060 trực tiếp, cả 5 tab hiển thị chuẩn, click chuột thực mở/duyệt ổ đĩa (`C:`, `D:`, `E:`, `G:`, `H:`)/đóng hộp thoại `#fs-picker-dialog` không bị che khuất, thử quy tắc `[SubsPlease]* -> animejanai-hd-v3-ultracompact`, vận hành Subtrans 24/7 + quản lý 3 khe khóa `.env`, và mở biên bản `#audit-detail-dialog` hiển thị khớp khung hình `1.478 / 1.478`.
- Mux/frame/fps: N/A — không chạy suy luận GPU hay tạo video thành phẩm trong phiên này.

## Bước tiếp theo
1. Đã hoàn tất toàn bộ yêu cầu; không còn bước tồn đọng.

## Kết thúc
- Kết quả và giới hạn đã thông báo: Hoàn tất triển khai toàn trình Web GUI thời gian thực & tích hợp `modern-web-guidance`, vượt qua 750 ca kiểm thử cô lập (`PASS 100%`), quét `modern-web-guidance legacy-scan` (`0` lỗi), và kiểm chứng tương tác trực tiếp trên trình duyệt (`0` lỗi console); chưa chạy upscale GPU dài hay gọi Gemini API trực tuyến trong kiểm thử tự động.
- Commit/tag: Commit trực tiếp trên nhánh `main` kèm đồng bộ `STATE.md`.
- Tài liệu cập nhật: [STATE](../STATE.md), [AGENTS](../../../AGENTS.md), [PROJECT_MAP](../PROJECT_MAP.md), [README](../../../README.md).
