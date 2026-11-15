# Handoff: Hoàn thiện nhanh các tối ưu Phần mềm, CLI & Cấu hình (Items 1, 3a, 5)

Tạo khi bắt đầu tác vụ nhiều bước; cập nhật trong lúc làm, trước compact/bàn giao/dừng. Mỗi tác vụ một file.

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-15T14:22+07:00
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu thực hiện trước các điểm hoàn thiện nhanh chóng, dễ dàng và an toàn trong 5 điểm còn lại ở tầng phần mềm/CLI/cấu hình, để lại các phần khó/lâu hoặc cần quyết định chính sách cho người dùng tự quyết.
- Kết quả cần bàn giao:
  1. Đồng bộ cờ CLI `--dedup-policy` hỗ trợ đầy đủ `sakuga` và `dis-sakuga` trong `upscale.py`.
  2. Khắc phục lỗi `fnmatch` nuốt dấu ngoặc vuông `[...]` khi khớp mẫu `shows.yaml` (`src/models_resolver.py`, `src/dedup.py`), bổ sung tính năng tự động mở rộng đường dẫn thư mục thành hàng đợi video sắp xếp tự nhiên (`expand_video_inputs` trong `upscale.py`, tránh lỗi ký tự `[` `]` của PowerShell), và cập nhật mẫu cấu hình chuẩn cho `SubsPlease` / `Blu-ray` trong `shows.yaml`.
  3. Dọn sạch các mục cấu hình thừa lịch sử (`ollama`, `translategemma:12b`, `- local`) trong `subtrans/config.yaml` và `subtrans/models.yaml` (kèm cập nhật bài kiểm thử `tests/test_gemini_quota_management.py` và thêm `tests/test_cli_and_shows_polish.py`).
  4. Trình bày rõ ràng 2 điểm để lại cho người dùng quyết định (bật mặc định `--zero-disk=True` trên `upscale.py` / mở khóa rule `shows.yaml` tự động, và thuật toán phức tạp Cross-Episode OP/ED Cache).
- Được phép / chưa được phép:
  - Được phép sửa các điểm nhanh/dễ/an toàn; giữ nguyên model SR mặc định toàn cục (`animevideov3`) và mặc định `--zero-disk=False` của `upscale.py` cho đến khi có xác nhận trực tiếp.
- Ràng buộc quan trọng: Bảo vệ Gemini quota ban ngày (sau 14:00 không gọi API thật), chạy kiểm thử qua `scripts/verify_offline.py`, giữ `AGENTS.md <= 12,288 B`, commit ngay khi xong yêu cầu.
- Trạng thái: hoàn tất.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, `main`, `b8fb0ea`
- File đã sửa: `upscale.py`, `src/models_resolver.py`, `src/dedup.py`, `shows.yaml`, `subtrans/config.yaml`, `subtrans/models.yaml`, `subtrans/pipeline.py`, `subtrans/AGENTS.md`, `tests/test_gemini_quota_management.py`, `tests/test_cli_and_shows_polish.py`, `docs/agents/STATE.md`, `docs/agents/PROJECT_MAP.md`.
- Chủ sở hữu từng phần / subagent / kết quả đã nhận: Agent chính thực hiện trực tiếp (Solo).
- Tiến trình/session đang chạy và nơi xem kết quả: Không có tiến trình chạy nền.
- File dữ liệu production cần giữ: `.env`, `subtrans/.cache/*.db`, `models_bench/`.

## Đã làm và bằng chứng
- Thay đổi:
  - `upscale.py:build_arg_parser` & `upscale.py:expand_video_inputs`: Đồng bộ `sakuga`/`dis-sakuga` vào `choices` của `--dedup-policy`; bổ sung quét thư mục tự động theo thứ tự tập tự nhiên (`_natural_sort_key`), bỏ qua file ẩn và thành phẩm `*_4K` / `*_4K_animejanai`.
  - `src/models_resolver.py:match_show_pattern`, `resolve_model_for_video` & `src/dedup.py:resolve_dedup_policy`: Thay thế `fnmatch` bằng bộ khớp glob/substring escape dấu ngoặc vuông `[SubsPlease]`, `[BD 1080p]`.
  - `shows.yaml`: Cập nhật tài liệu 8 model hiện hành và sẵn 3 mẫu cấu hình cho Sakuga, `[SubsPlease]*` Web-DL và `*[BD*` Blu-ray.
  - `subtrans/config.yaml`, `subtrans/models.yaml`, `subtrans/pipeline.py`, `subtrans/AGENTS.md`, `tests/test_gemini_quota_management.py`: Gỡ bỏ hoàn toàn khai báo `ollama` / `translategemma:12b` / `local` thừa lịch sử.
  - `tests/test_cli_and_shows_polish.py`: Thêm 5 ca kiểm thử tự động khóa bất biến đồng bộ CLI/engine, ngoặc vuông `shows.yaml`, quét thư mục và cấu hình Cloud thuần.
- Lệnh kiểm tra, interpreter, số pass/fail/skip, thời gian, evidence path:
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py` -> **`734 passed, 14 skipped, 1 deselected in 252.18s (0:04:12)`**, exit code `0`, evidence tại `E:\ai-video-upscaler\scratch\agent-validation-y_oojcf9`.
  - `.\.venv\Scripts\python.exe scripts/agent_context.py --check` -> **`PASS`** (`0` lỗi).
  - `ruff check upscale.py src/models_resolver.py src/dedup.py subtrans/pipeline.py tests/test_gemini_quota_management.py tests/test_cli_and_shows_polish.py` -> **`All checks passed!`**.
- Mux/frame/fps: **`N/A — không tạo thành phẩm`** (tác vụ hoàn thiện CLI, khớp mẫu cấu hình và kiểm thử offline).

## Bước tiếp theo
1. Người dùng xem xét quyết định 2 mục còn lại:
   - Mục chính sách cấu hình mặc định: Có bật `--zero-disk` thành mặc định `True` trên `upscale.py` và/hoặc kích hoạt sẵn rule `[SubsPlease]*` / `*[BD*` trong `shows.yaml` hay không.
   - Mục thuật toán phức tạp/lâu: Có đầu tư xây dựng **Cross-Episode OP/ED Cache** (~10–12% tiết kiệm thời gian mùa phim nhưng nhiều biên lệch GOP/SFX) hay không.
- Câu hỏi/approval còn chờ: Chờ quyết định của người dùng trên 2 mục trên.
- Rủi ro/giới hạn cần báo người dùng: Không có rủi ro đối với cấu hình sản xuất hiện tại.

## Kết thúc
- Kết quả và giới hạn đã thông báo: Đã hoàn thành và kiểm thử 100% 3 hạng mục nhanh/an toàn, bàn giao rõ 2 hạng mục chờ quyết định.
- Commit/tag: Tra `git log -n 1 -- docs/agents/tasks/2026-11-15-software-cli-quick-optimizations.md`.
- Tài liệu cập nhật: [STATE](../STATE.md), [AGENTS](../../../AGENTS.md), [PROJECT_MAP](../PROJECT_MAP.md), [Subtrans AGENTS](../../../subtrans/AGENTS.md).
