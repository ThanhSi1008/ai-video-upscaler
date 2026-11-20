# Handoff: Upscale Fate/Zero Tập 10, 11, 12 lên 4K qua Web Studio GUI

Tạo khi bắt đầu tác vụ nhiều bước; cập nhật trong lúc làm, trước compact/bàn giao/dừng. Mỗi tác vụ một file.

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-16 05:38 (+07:00)
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu dùng `/browser` (Web Studio GUI `http://127.0.0.1:8765/`) để thao tác, chạy và theo dõi quá trình upscale 3 tập Fate/Zero (tập 10, 11, 12) từ `E:\Films\Raw_Sources` và xuất kết quả ra `E:\Films\Upscaled_4K`, sau đó báo cáo kết quả.
- Kết quả cần bàn giao:
  - `E:\Films\Upscaled_4K\FateZero - S01E10 - Rin's Adventure_4K.mkv`
  - `E:\Films\Upscaled_4K\FateZero - S01E11 - The Grail Dialogue_4K.mkv`
  - `E:\Films\Upscaled_4K\FateZero - S01E12 - Invitation Of The Holy Grail_4K.mkv`
  - Kèm `audit/*_4K.audit.json` (`Mux Audit PASS`, `Output Sanity PASS`, `frames_matched: True`), bằng chứng ảnh chụp Web GUI, kiểm chứng `ffprobe`/PyAV và báo cáo tổng hợp.
- Được phép / chưa được phép: Được phép chạy Web GUI `web_gui.py` và điều phối toàn trình 3 tập Fate/Zero trên GPU cục bộ qua `browser` subagent. Không probe port/API qua CLI thay cho trình duyệt; không ghi đè phụ đề Việt sẵn có (`#opt-subs-after` tắt vì cả 3 MKV đã nhúng `vie` SRT + 2 track `eng` SSA + Dual Audio).
- Ràng buộc quan trọng: Bảo toàn BT.709 two-way matrix, 10-bit `hevc_nvenc`, bit-exact stream copy `-max_interleave_delta 0`, 1:1 frame count parity, giữ `AGENTS.md <= 12,288` bytes, commit Git khi hoàn tất.
- Trạng thái: tạm dừng — đã tắt daemon Web GUI theo yêu cầu người dùng (`tạm dừng hết đi, khi nào tôi ngồi vào bàn PC rồi làm tiếp`), giữ nguyên 3 file nguồn `E:\Films\Raw_Sources` và thư mục đích `E:\Films\Upscaled_4K` để chạy lại khi người dùng ngồi vào bàn PC.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, `main`, `53bfdda` (working tree sạch).
- File đã sửa (kể cả chưa commit):
  - Mới tạo: `docs/agents/tasks/2026-11-16-fate-zero-ep10-12-web-gui.md`
- Chủ sở hữu từng phần / subagent / kết quả đã nhận:
  - Agent chính: Đã tắt daemon Web GUI `--lan` (`task-1021`), xác nhận 0 tiến trình GPU/Web/FFmpeg còn chạy.
  - Subagent `browser` (`a7c87738-fe67-45ff-8714-e2bb9fd50f6b`): Nhàn rỗi sau báo cáo `BLOCKED` do hộp thoại Chrome `"Allow remote debugging?"` trên PC khi người dùng đang điều khiển từ điện thoại.
- Tiến trình/session đang chạy và nơi xem kết quả:
  - Đích xuất 4K: `E:\Films\Upscaled_4K\`
  - Đường dẫn nguồn đã xác minh:
    1. `E:\Films\Raw_Sources\[Tenrai-Sensei] Fate Zero [BD][1080p][HEVC 10bit x265][Dual Audio]\Season 1\FateZero - S01E10 - Rin's Adventure.mkv` (`434.13 MB`, `1920x1080`, `24000/1001` fps, `1429.44s`, 2 audio, 3 subtitles incl. `vie`)
    2. `E:\Films\Raw_Sources\[Tenrai-Sensei] Fate Zero [BD][1080p][HEVC 10bit x265][Dual Audio]\Season 1\FateZero - S01E11 - The Grail Dialogue.mkv` (`481.74 MB`, `1920x1080`, `24000/1001` fps, `1772.78s`, 2 audio, 3 subtitles incl. `vie`)
    3. `E:\Films\Raw_Sources\[Tenrai-Sensei] Fate Zero [BD][1080p][HEVC 10bit x265][Dual Audio]\Season 1\FateZero - S01E12 - Invitation Of The Holy Grail.mkv` (`409.36 MB`, `1920x1080`, `24000/1001` fps, `1710.72s`, 2 audio, 3 subtitles incl. `vie`)
- File dữ liệu production cần giữ: Toàn bộ file `.mkv` và `.vi.srt` trong `E:\Films\Raw_Sources\`, thành phẩm `_4K.mkv` và `audit/` trong `E:\Films\Upscaled_4K\`, không xóa `.temp_*` khi đang chạy/dở dang.

## Đã làm và bằng chứng
- Xác minh đĩa & stream đầu vào bằng PyAV: ổ `E:` còn `70.52 GB` trống; GPU idle (không có `.gpu_active.lock`); cả 3 tập Fate/Zero 10, 11, 12 đều là `1920x1080`, `24000/1001` fps, 2 audio tracks, 3 subtitle tracks (`vie` SRT, `eng` S&S SSA, `eng` Dialog - ENG SSA) và có sidecar `.vi.srt` sẵn.
- Đang chuẩn bị khởi chạy `web_gui.py --port 8765 --no-browser` và thao tác qua `browser` subagent (`#opt-model=""` -> `animevideov3`, `#opt-dedup="dis"`, `#opt-batch-size=8`, `#opt-zero-disk=checked`, `#opt-tensorrt=checked`, `#opt-subs-after=unchecked`, `#opt-output-dir=E:\Films\Upscaled_4K`).

## Bước tiếp theo
1. Khởi chạy daemon `.\.venv\Scripts\python.exe web_gui.py --port 8765 --no-browser`.
2. Gọi `browser` subagent mở `http://127.0.0.1:8765/`, cấu hình Tab 2 (`#panel-queue`), tính dự báo ETA và nhấn `Bắt đầu Nâng cấp 4K`.
3. Theo dõi định kỳ bằng `schedule` + `send_message` tới cùng `browser` subagent cho đến khi 3 tập hoàn tất 100%, kiểm tra Tab 3 đối chiếu + audit, dọn daemon và commit Git.

## Kết thúc
- Kết quả và giới hạn đã thông báo: (chưa hoàn tất)
- Commit/tag: (chưa hoàn tất)
- Tài liệu cập nhật: `docs/agents/tasks/2026-11-16-fate-zero-ep10-12-web-gui.md`
