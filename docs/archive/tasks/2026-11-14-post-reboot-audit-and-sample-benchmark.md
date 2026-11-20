# Handoff: Hậu khởi động lại Windows 11 — Tối ưu dọn tiến trình nền, trích xuất Sample hoành tráng & Đo đạc thực nghiệm Full Pipeline

## Yêu cầu và phạm vi
- Ngày/múi giờ: `2026-11-14T12:00:00+07:00`, Asia/Bangkok (+07:00).
- Nguồn yêu cầu và mục tiêu: Người dùng vừa khởi động lại máy để kích hoạt `HAGS = 2` & `CUDA P0 State` và yêu cầu trực tiếp:
  1. Kiểm tra các tiến trình vô nghĩa, tốn tài nguyên, không cần thiết đang ăn RAM/CPU/VRAM của máy để tắt vĩnh viễn (Startup/Autorun) hoặc tạm thời tùy trường hợp.
  2. Tìm một tập phim hoành tráng trong `"E:\Films\Raw_Sources"` lấy ra làm sample để chạy test thử tốc độ, độ ổn định, VRAM, xung nhịp, chất lượng.
  3. Ghi logs, số liệu viễn trắc (telemetry) và tài liệu liên quan đầy đủ để sau khi chạy xong có cơ sở dữ liệu thực tế tiếp tục cải tiến, nâng cấp và tối ưu hóa thêm.
- Trạng thái: đang chạy lượt benchmark toàn trình số 2 (sau khi vá triệt để 3 nút thắt phần cứng phát hiện ở lượt 1).

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, branch `main`, baseline `9aec6e5`.
- Các file mã nguồn đã cập nhật:
  - `src/build_trt_engine.py`
  - `src/ffmpeg_ops.py`
  - `src/trt_zero_copy_upscaler.py`
  - `src/pipeline.py`
  - `tests/test_final_ceiling_audit.py`

## Đã làm và bằng chứng
1. **Xác nhận mở khóa phần cứng hậu Restart:**
   - `HKLM\...\GraphicsDrivers\HwSchMode = 2` (**HAGS ĐÃ BẬT**).
   - Xung thực thi CUDA FP16 đạt **`P0, 7501 MHz / 7501 MHz VRAM, Core 1942 MHz, 158.96 W`** (đã gỡ khóa `P2 @ 7301 MHz`).
2. **Dọn sạch Startup rác (vĩnh viễn) & Tiến trình nền ngốn RAM/VRAM (tạm thời):**
   - Sao lưu Registry tại `logs/startup_hkcu_backup_20261114.json`.
   - Gỡ vĩnh viễn 8 mục tự khởi động thừa trong `HKCU\...\Run` (`MicrosoftEdgeAutoLaunch`, `GoogleChromeAutoLaunch`, `Adobe Acrobat Synchronizer`, `Figma Agent`, `Gemini`, `Upwork`, `HYP_1_0` HoYoPlay, `GoogleUpdaterTaskUser`), khóa `StartupBoostEnabled=0` & `BackgroundModeEnabled=0` của Edge và tắt `\GoogleUserPEH\*`. Chỉ giữ lại `UniKey`, `GoogleDriveFS`, `Zalo`.
   - Tạm dừng 14 tiến trình nền rảnh rỗi (`NVIDIA Overlay` ngốn `151.5 MB` VRAM, `Microsoft.CmdPal.UI` ngốn `104.7 MB` VRAM, `msedge` ngầm, `DeskIn` 0 kết nối, `MSPCManager`, `IntelGraphicsSoftware`, `CrossDeviceResume`, `figma_agent`, 4 tiện ích phụ `PowerToys`, `AdobeNotificationClient`, `CCXProcess`), giải phóng **`+1.08 GB` RAM** (trống `17.39 GB`) và nâng VRAM trống lên **`9.634 MiB / 12.288 MiB`**.
3. **Trích xuất Sample đại cảnh chiến đấu hoành tráng:**
   - Nguồn: `Demon Slayer: Entertainment District Arc - S03E10 - Never Give Up.mkv` (`15:30 -> 18:30`, trường đoạn đỉnh cao Âm Trụ Uzui Tengen khai mở Phổ Nhạc tử chiến Gyutaro).
   - File mẫu: `samples/sample_kimetsu_s03e10_climax_3min.mkv` (`146.68 MiB`, `182.569s`, **`4.374` frames**, `1920x1080` `hevc Main 10 yuv420p10le`, 2 luồng Opus Audio, 3 luồng phụ đề ASS, 10 font chữ, 3 chương).
4. **Phát hiện & xử lý triệt để 3 lỗi phần cứng thực tế từ Lượt chạy 1 (`logs/benchmark_kimetsu_s03e10_20261114_run1_before_fix_*`):**
   - **Lỗi 1 — TensorRT 10.16 OOM `12.74 GB` & kẹt 33 phút (`src/build_trt_engine.py`):** `.float()` trên nhánh trung gian 8K (`4320x7680`) ở `max_batch=8` tạo tensor `4 x 3.185 GB = 12.740.198.400 bytes` vượt trần 12GB VRAM, kèm lỗi `ForeignNode` ở `builder_optimization_level=5`, khiến engine fallback ngốn ~8 GB VRAM và chỉ đạt `5.72 fps`. **Đã sửa:** Giữ đồ thị FP16 thuần khiết trên CUDA, giới hạn profile `trt_max_batch = 2` cho mô hình `PixelShuffle(4)` (`resolve_trt_profile_max_batch`), đặt `builder_optimization_level = 3` và cắt lát zero-copy qua `_batch_out_ring` — đo thực tế đạt **`11.55 fps` (`+114%` so với PyTorch `5.39 fps`), đỉnh VRAM chỉ `1.331 MB` (`PSNR = 54.37 dB`, `MAD = 0.3737/255`)**.
   - **Lỗi 2 — Khoảng trống GPU `P8` `99–112s` giữa các chunk do Windows `NtReadFile` khóa trang MDL $O(N^2)$ (`src/ffmpeg_ops.py`):** Truyền `mv[offset:total_target]` (`1.36 GB` = `334.125` trang 4KB) vào pipe ẩn danh 4KB của Windows khiến kernel khóa/mở `334.125` trang lặp lại sau mỗi 4KB (`112.97s` cho 220 frames!). **Đã sửa:** Đặt `bufsize=0`, đọc từng lát `1 MiB` (`1048576` bytes), bỏ `select='...'` khi giữ 100% frame và giải mã đa luồng CPU ở `DecodeWorker` (`0 MB` VRAM, khớp pixel tuyệt đối `max_diff = 0`) — thời gian giải mã giảm từ **`112.97s` xuống `1.05s` (nhanh gấp `107.6x`)**!
   - **Lỗi 3 — `[Errno 22] Invalid argument` khi NVENC thiếu VRAM (`src/trt_zero_copy_upscaler.py`, `src/pipeline.py`):** Đổi `spawn_nvenc_pipe_writer` sang `bufsize=0` + lát ghi `4 MiB` + thu `stderr`, gắn `record_stream` chống thu hồi sớm qua 3 CUDA streams, và thêm chốt kiểm tra VRAM trống `< 2.2 GB` trước khi mở phiên `hevc_nvenc` song song.

## Bước tiếp theo
1. Hoàn tất Lượt chạy 2 toàn trình (`run_benchmark_with_telemetry.py`) trên toàn bộ `23/23` chunks (`4.374` frames).
2. Xác minh độc lập `Output Sanity`, `Mux Audit` (`streamhash` MD5), `ffprobe -count_frames == 4374`.
3. Chạy `scripts/verify_offline.py` + `scripts/agent_context.py --check`, ghi đầy đủ nhật ký So sánh Trước/Sau + Lộ trình cải tiến vào `docs/engineering-log.md`, cập nhật `STATE.md` & `PROJECT_MAP.md`, và commit sạch sẽ.
