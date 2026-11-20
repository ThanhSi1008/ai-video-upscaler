# Handoff: Tối ưu hóa thuật toán kịch trần cho Video Upscale Pipeline (Algorithmic Ceiling)

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-13 (Asia/Bangkok, UTC+7)
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu kiểm toán lại toàn bộ giới hạn phần cứng & phần mềm xem đã kịch trần chưa, và nếu đã kịch trần thì chuyển sang tối ưu tầng **thuật toán (Algorithmic Complexity)** để vắt kiệt tốc độ mà không giảm chất lượng anime.
- Kết quả cần bàn giao:
  1. Intra-chunk Non-Consecutive / Cyclic Anime Cel Memoization (`A -> B -> A` lip-flap / blink / cut reuse + Belady's Exact Liveness Eviction trên Zero-Disk & NTFS Hardlink trên Standard Disk).
  2. Triệt tiêu lượt suy luận GPU lặp lại (`B=1`) khi kiểm tra `OutputSanity` ở chế độ `--zero-disk` bằng `F.avg_pool2d` bội số nguyên ngay ở Batch 0.
  3. Hợp nhất `Concat + Mux` một lượt (`mux_segments_directly`), loại bỏ ~8 GB đọc/ghi file trung gian `merged_4k.mp4`.
  4. Giải mã `extract_frames_raw_rgb24` đọc thẳng `readinto` vào vùng nhớ NumPy tiền cấp phát (khử bản sao `b"".join()` 1.2 GB trên DDR4-2133).
  5. Hợp nhất 1 lượt giải mã FFmpeg `split=2` (`mpdecimate` + `480x270 gray`) và thác cắt sớm (Early-Exit Cascade) cho `DIS Optical Flow` trong `src/dedup.py`.
  6. Băm đa luồng âm thanh 1 lượt đọc (`-f streamhash -hash md5`) và song song hóa `ThreadPoolExecutor` cho `verify_mux_integrity` (`src/mux_audit.py`), tái sử dụng metadata trong `src/pipeline.py`.
- Được phép / chưa được phép:
  - Được phép: Tối ưu thuật toán bảo toàn 100% chất lượng nét vẽ anime và toàn bộ bất biến `AGENTS.md` §4; viết unit test offline và commit riêng.
  - Chưa được phép: Không đổi model mặc định (`realesr-animevideov3`), không hạ thông số NVENC, không chạy GPU > 100 frames, không chạm các file `subtrans/*` hay `src/subtitles.py` của phiên song song.
- Ràng buộc quan trọng: Tôn trọng tuyệt đối quyền sở hữu file với phiên Subtrans song song; chỉ stage đúng 12 file thuộc phạm vi Video Upscale.
- Trạng thái: hoàn tất.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, nhánh `main`, commit nền `b896ec1`.
- File đã sửa/thêm (thuộc phạm vi tác vụ này):
  - `src/dedup.py`
  - `src/trt_zero_copy_upscaler.py`
  - `src/cuda_engine.py`
  - `src/ffmpeg_ops.py`
  - `src/mux_audit.py`
  - `src/output_sanity.py`
  - `src/pipeline.py`
  - `tests/test_algorithmic_optimizations.py`
  - `AGENTS.md`, `docs/agents/PROJECT_MAP.md`, `docs/agents/STATE.md`
- File dữ liệu production cần giữ: `.env`, `subtrans/.cache/*`, `models_bench/*`, `.temp_*`.

## Đã làm và bằng chứng
- Thay đổi chi tiết:
  - `src/dedup.py:deduplicate_cyclic_frames`: Băm `blake2b` 128-bit + kiểm chứng `cv2.absdiff` toàn khung (`max_diff <= 2`, `MAE <= 0.20`, `PSNR >= 57 dB`) bắt chu kỳ nhép miệng `A->B->A` mà giữ nguyên 100% dịch chuyển 1-pixel/grain.
  - `src/dedup.py:_is_dis_motion_early_exit`, `_scan_fused_mpdecimate_and_dis`: Thác cắt sớm DIS (`np.array_equal` -> `count_nonzero` -> `pan` -> `zoom` -> `p90`) + giải mã 1 lượt `[0:v]split=2`.
  - `src/trt_zero_copy_upscaler.py:DoubleBufferedPipeUpscaler.process_frames_to_pipe`: Giải thuật thu hồi tối ưu Belady (`OPT`) chỉ lưu frame còn xuất hiện ở batch sau + thu mẫu Sanity `F.avg_pool2d` ở Batch 0 (0 lượt infer lặp).
  - `src/cuda_engine.py:GPUPersistentWorker.upscale_directory`: Khử trùng lặp chu kỳ + tạo `os.link` NTFS hardlink (0 lần nén PNG thừa, 0 byte đĩa phụ).
  - `src/ffmpeg_ops.py:extract_frames_raw_rgb24`, `mux_segments_directly`: Luồng `readinto` 0-copy RAM và ghép nối + đóng gói MKV 1 lượt (khử 8 GB I/O `merged_4k.mp4`).
  - `src/mux_audit.py:get_all_audio_streams_md5`, `check_window_interleaving`, `verify_mux_integrity` & `src/pipeline.py`: Băm đa track 1 lượt `-f streamhash` và kiểm định song song trên `ThreadPoolExecutor`.
- Quyết định bác bỏ: Loại hướng đóng băng nền ngoài hộp giới hạn ROI (ROI Delta Background Freeze) vì trên nguồn BD có hạt nhiễu phim (film grain) sẽ tạo ranh giới nhiễu tĩnh/động quanh nhân vật; chọn khử trùng lặp chu kỳ toàn khung (`max_diff <= 2` trên 100% pixel) không để lại đường nối không gian.
- Kiểm thử & Bằng chứng:
  - `.\.venv\Scripts\python.exe scripts/agent_context.py --check` -> `PASS` (`AGENTS.md = 11,984 bytes <= 12,288 bytes`).
  - `ruff check src/dedup.py src/trt_zero_copy_upscaler.py src/cuda_engine.py src/ffmpeg_ops.py src/mux_audit.py src/output_sanity.py src/pipeline.py tests/test_algorithmic_optimizations.py` -> `All checks passed!`.
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py` -> **`314 passed, 11 skipped, 1 deselected in 82.74s`**, evidence tại `E:\ai-video-upscaler\scratch\agent-validation-fai06vlr`.
- Mux/frame/fps: Đã kiểm chứng `verify_mux_integrity == PASS` (`144/144` frames, khớp bit-exact MD5 2 track audio) trên clip mẫu sinh bởi FFmpeg trong `test_mux_segments_directly_and_streamhash_multi_audio_md5`; tốc độ tập phim thực tế: `N/A — không chạy benchmark GPU dài trên tập phim thật khi chưa có yêu cầu`.

## Kết thúc
- Kết quả và giới hạn đã thông báo: Báo cáo rõ phần cứng/driver đã kịch trần (chỉ còn Resizable BAR trong BIOS và Restart Windows để nạp HAGS/P0), chi tiết 6 thuật toán vừa triển khai, và các lựa chọn đánh đổi mô hình (`animejanai-hd-v3-compact` native 2x) cần người dùng cho phép nếu muốn tăng thêm ~2x tốc độ.
- Commit/tag: Tra `git log -n 1 -- docs/agents/tasks/2026-11-13-algorithmic-ceiling-optimizations.md` ngay sau bước commit của phiên này.
- Tài liệu cập nhật: [STATE](../STATE.md), [PROJECT_MAP](../PROJECT_MAP.md), [AGENTS](../../../AGENTS.md).
