# Handoff: Chuẩn hóa mặc định Zero-Disk, Tự động phân luồng shows.yaml, CRLF .bat & ADR-010

Tạo khi bắt đầu tác vụ nhiều bước; cập nhật trong lúc làm, trước compact/bàn giao/dừng. Mỗi tác vụ một file.

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-15T14:51+07:00 – 2026-11-15T16:38+07:00
- Nguồn yêu cầu và mục tiêu: Người dùng ủy quyền toàn phần (`/teamwork /boost /swarm /plan`) để phân tích kỹ lưỡng từ nguyên lý đầu tiên, quyết định và thực thi trọn gói phương án tối ưu nhất cho cấu hình mặc định pipeline (`--zero-disk`, `shows.yaml`) và xử lý dứt điểm định hướng `Cross-Episode OP/ED Cache` mà không cần hỏi lại hay chờ duyệt.
- Kết quả cần bàn giao:
  1. Bật mặc định `--zero-disk=True` trên toàn bộ `upscale.py`, `src/pipeline.py`, `upscale.bat` kèm cờ dự phòng `--no-zero-disk` / `--disk` tương thích ngược 100%.
  2. Kích hoạt sẵn quy tắc tự động phân luồng trong `shows.yaml`: `[SubsPlease]` / `[Erai-raws]` Web-DL -> `animejanai-hd-v3-ultracompact` (`default` dedup); Blu-ray (`*[BD*`, `*BluRay*`, `*BDRip*`) -> `animevideov3` (`dis` dedup); các file không khớp mẫu giữ nguyên mặc định `animevideov3`.
  3. Ban hành quyết định kiến trúc **ADR-010** trong `docs/agents/DECISIONS.md` bác bỏ `Cross-Episode OP/ED Cache` dựa trên 5 chứng minh kỹ thuật bất biến (thay đổi chữ credit ở 100% ED & >70% OP, lệch GOP/PTS, thoại/SFX đè nhạc, nguy cơ trôi phụ đề 20 phút còn lại, và nghịch lý Pareto chỉ tiết kiệm ~1.7%–2.2% ca đêm).
  4. Chuẩn hóa CRLF + `.gitattributes` (`*.bat text eol=crlf`) cho toàn bộ 4 file `.bat` trên Windows `cmd.exe`.
- Được phép / chưa được phép: Người dùng cấp toàn quyền quyết định và triển khai ngay cấu hình tối ưu nhất cho dự án.
- Ràng buộc quan trọng: Bảo vệ Gemini API quota ban ngày (>= 14:00, 0 cuộc gọi API thật), kiểm thử cô lập qua `scripts/verify_offline.py`, giữ `AGENTS.md <= 12,288 B`.
- Trạng thái: hoàn tất.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, `main`, `1a9dc9c`
- File đã sửa: `.gitattributes`, `AGENTS.md`, `README.md`, `docs/agents/DECISIONS.md`, `docs/agents/PROJECT_MAP.md`, `docs/agents/STATE.md`, `docs/video-pipeline.md`, `.agents/skills/video-upscale-ops/SKILL.md`, `shows.yaml`, `src/pipeline.py`, `tests/e2e/test_tier1_features.py`, `tests/test_cli_and_shows_polish.py`, `tests/test_phase2_zero_disk_and_trt.py`, `tests/test_rtx3060_unlocks.py`, `unlock_rtx3060_admin.bat`, `upscale.py`, `upscale_turbo.bat`.
- Chủ sở hữu từng phần / subagent / kết quả đã nhận: Đội ngũ đa tác tử `teamwork` (`84ec4c58-d201-48e9-ac64-1306bcffaa60`, qua 2 vòng Gate + Independent Victory Audit `VICTORY CONFIRMED`) và kiểm tra đối chứng độc lập bởi Agent điều phối chính.
- Tiến trình/session đang chạy và nơi xem kết quả: Đã kết thúc toàn bộ tiến trình con.
- File dữ liệu production cần giữ: `.env`, `subtrans/.cache/*.db`, `models_bench/`.

## Đã làm và bằng chứng
- Thay đổi:
  - `upscale.py`, `src/pipeline.py`: `--zero-disk` mặc định `True`, bổ sung `--no-zero-disk` / `--disk`, đồng bộ dự toán overhead ca đêm.
  - `shows.yaml`: Kích hoạt 7 rule tự động phân luồng Web-DL (`[SubsPlease]`, `[Erai-raws]`) và Blu-ray (`*[BD*`, `*BluRay*`, `*BDRip*`).
  - `docs/agents/DECISIONS.md`: Ghi nhận `ADR-010` bác bỏ Cross-Episode OP/ED Cache.
  - `.gitattributes`, `upscale_turbo.bat`, `unlock_rtx3060_admin.bat`: Khóa chuẩn `eol=crlf` trên Windows `cmd.exe`.
  - Đồng bộ test suite & tài liệu (`AGENTS.md = 12,200 B <= 12,288 B`, `PROJECT_MAP.md`, `STATE.md`, `README.md`, `docs/video-pipeline.md`, `SKILL.md`).
- Lệnh kiểm tra, interpreter, số pass/fail/skip, thời gian, evidence path:
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py` -> **`734 passed, 14 skipped, 1 deselected, 0 failed in 447.57s`**, exit code `0`.
  - `.\.venv\Scripts\python.exe scripts/agent_context.py --check` -> **`PASS`** (`AGENTS.md = 12,200 B <= 12,288 B`; `root + subtrans = 31,616 B <= 32,768 B`).
  - `ruff check` & `git diff --check` & `cmd.exe /c upscale.bat --help` & `cmd.exe /c upscale_turbo.bat --help` -> **`0` lỗi**.
- Mux/frame/fps: **`N/A — không tạo thành phẩm`** (chuẩn hóa cấu hình mặc định, phân luồng model, `.bat` CRLF và ADR-010; số đo phần cứng R3 trước đó giữ nguyên hiệu lực).

## Bước tiếp theo
1. Không còn hạng mục mở — dự án đã hoàn tất khép kín toàn bộ phần cứng, thuật toán, phần mềm, công cụ và cấu hình vận hành tự động.
- Câu hỏi/approval còn chờ: Không có.
- Rủi ro/giới hạn cần báo người dùng: Nếu cần giữ PNG trung gian để debug từng ảnh rời, dùng cờ `--no-zero-disk` (hoặc `--disk`).

## Kết thúc
- Kết quả và giới hạn đã thông báo: Báo cáo đầy đủ kèm cập nhật `walkthrough.md`.
- Commit/tag: Tra `git log -n 1 -- docs/agents/tasks/2026-11-15-autonomous-pipeline-defaults-and-adr010.md`.
- Tài liệu cập nhật: [STATE](../STATE.md), [AGENTS](../../../AGENTS.md), [PROJECT_MAP](../PROJECT_MAP.md), [DECISIONS](../DECISIONS.md).
