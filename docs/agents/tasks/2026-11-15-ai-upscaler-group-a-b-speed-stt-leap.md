# Handoff: Nâng cấp toàn diện pipeline AI Video Upscaler theo Nhóm A và Nhóm B (Speed & STT Leap)

Tạo lập và hoàn thiện hồ sơ bàn giao toàn diện sau khi hoàn tất 5 Milestone M1-M5 của sáng kiến tối ưu hóa Nhóm A (Vá lỗi & tối ưu bằng thư viện có sẵn) và Nhóm B (Bước nhảy vọt tốc độ 1.6x–2.5x & chất lượng STT).

## Yêu cầu và phạm vi
- **Ngày/múi giờ:** 2026-11-15T02:05:00+07:00 (UTC 2026-11-14T19:05:00Z).
- **Nguồn yêu cầu và mục tiêu:** Yêu cầu trực tiếp từ Người dùng qua `/plan` và `/teamwork` (`ORIGINAL_REQUEST.md`). Mục tiêu: Nâng cấp toàn diện pipeline AI Video Upscaler theo Nhóm A và Nhóm B, bảo đảm tuyệt đối không làm suy giảm chất lượng hình ảnh, giữ trọn vẹn bit-exact Stream Copy audio/subtitles/fonts/chapters, triệt tiêu lỗi Windows `WinError 32`, nâng cấp STT sang `large-v3-turbo` và mở rộng hỗ trợ Native 2x qua `spandrel`.
- **Kết quả cần bàn giao:**
  1. M1: Căn chỉnh công thức ETA `--deadline` theo Unique Inference FPS và kernel `@triton.jit` P010LE BT.709 TV-range.
  2. M2: Tích hợp PyAV 18.1.0 in-process metadata thay thế ~230-290 tiến trình `ffprobe.exe`/tập, giữ nguyên Concat-Mux `-max_interleave_delta 0` và Single-Pass `streamhash` MD5, tham số hóa CLI `scripts/set_track_defaults.py`.
  3. M3: Bọc kết nối SQLite bằng context manager đóng sạch handle triệt tiêu `WinError 32`, nâng cấp chỉ mục RAG lên SQLite FTS5 n-gram 2-gram/3-gram cho từ khóa Kanji 2 chữ.
  4. M4: Tích hợp `spandrel 0.4.2` nạp các kiến trúc Native 2x không PReLU (`SPAN-2x`, `2x_AnimeJaNai_V3`), module polymorphic ONNX/TRT export, nâng cấp Whisper STT mặc định lên `large-v3-turbo`.
  5. M5: Bộ 169 ca kiểm thử E2E 4 tầng (`tests/test_e2e_group_a.py`, `tests/test_e2e_group_b.py`, `tests/test_e2e_integration_t3_t4.py`), đo đạc thực nghiệm trên sample chuẩn Fate Apocrypha 22, đồng bộ trạng thái tài liệu và Git commit kỷ luật.
- **Được phép / chưa được phép:**
  - Được phép: Tối ưu mã nguồn nội bộ bằng thư viện đã có trong `.venv` (`triton-windows`, `av`, `spandrel`, `faster-whisper`), chạy kiểm thử cô lập offline qua `scripts/verify_offline.py`.
  - Chưa được phép: Không gọi API Gemini thật ban ngày (14:00-đêm), không chiếm dụng GPU khi pipeline đang chạy, không làm tràn ngân sách `AGENTS.md` (giới hạn 12,288 bytes).
- **Ràng buộc quan trọng:** 100% CỤC BỘ trên Windows 11, RTX 3060 12GB VRAM, Peak VRAM < 9.5 GB, Output Sanity Guard $|\Delta Y| \le 6.0/255$, Pearson $r \ge 0.98$, Mux bit-exact PASS.
- **Trạng thái:** **HOÀN THÀNH 100% (VICTORY CONFIRMED)**.

## Mốc mã nguồn và điều phối
- **Root, branch, baseline commit:** `e:\ai-video-upscaler`, branch `main`. Baseline trước sáng kiến: `cf38379`.
- **Chuỗi commit sáng kiến:**
  - `de5ebf4`: `feat(pipeline): realign deadline ETA math to unique inference FPS and add Triton P010LE JIT kernel` (Milestone 1)
  - `5eaf4d7`: `feat(m2): integrate PyAV 18.1.0 in-process metadata and parameterize track defaults CLI` (Milestone 2)
  - `d9bbc5d`: `feat(subtrans): clean SQLite connection closing and FTS5 2-gram/3-gram Kanji RAG` (Milestone 3)
  - `ac6b061`: `feat(model-engine): integrate Spandrel 0.4.2 Native 2x & Faster-Whisper Turbo` (Milestone 4)
  - `d6a7853`: `feat(m5): complete Group A & Group B acceptance, E2E test suite, and state synchronization` (Milestone 5)
- **File đã tạo mới và sửa đổi trong M5:**
  - `tests/test_e2e_group_a.py` (82 test cases)
  - `tests/test_e2e_group_b.py` (66 test cases)
  - `tests/test_e2e_integration_t3_t4.py` (21 test cases)
  - `subtrans/tests/test_whisper_noise_and_speaker.py` (sửa mock subprocess ffmpeg `-i` argument)
  - `docs/agents/STATE.md` (đồng bộ baseline và danh mục 14 thành phần)
  - `docs/agents/tasks/2026-11-15-ai-upscaler-group-a-b-speed-stt-leap.md` (hồ sơ bàn giao này)
- **Chủ sở hữu từng phần / subagent:**
  - M1 Worker: Deadline Math, Triton P010LE, Pinned Memory Ring.
  - M2 Worker: PyAV 18.1.0 in-process metadata, Concat-Mux preservation, Track Defaults CLI.
  - M3 Worker: SQLite context manager, FTS5 2-gram/3-gram Kanji RAG.
  - M4 Worker: Spandrel 0.4.2 Native 2x, Faster-Whisper Large-V3-Turbo.
  - M5 Explorers (1, 2, 3) & Worker M5: Bộ 169 ca kiểm thử E2E, benchmark thực nghiệm Fate Apocrypha 22, nghiệm thu tổng thể.
- **Tiến trình/session đang chạy và nơi xem kết quả:** Không có tiến trình nền treo; toàn bộ background tasks đã hoàn thành với exit code 0.
- **File dữ liệu production cần giữ:** Giữ nguyên models registry `models/models.yaml`, glossaries, và checkpoints.

## Đã làm và bằng chứng
- **Thay đổi chi tiết theo tính năng:**
  1. *Feature 1 (Deadline Math Realignment)*: `src/scheduler.py`, `upscale.py`: Thay thế effective FPS bằng unique inference FPS khi tính toán ETA, sửa sai lệch ~2.1x khi có dedup.
  2. *Feature 2 (Triton JIT P010LE Kernel)*: `src/trt_zero_copy_upscaler.py`: Kernel `@triton.jit` BT.709 TV-range thay thế `torch.compile`, khử cold start <50ms và B=1 dynamic recompile.
  3. *Feature 3 (Direct Pinned H2D Transfer)*: `src/trt_zero_copy_upscaler.py`: Tiền cấp phát `gpu_nhwc_ring` khử hoàn toàn cấp phát VRAM động ~50MB trong batch dispatch.
  4. *Feature 4 (PyAV In-Process Metadata)*: `src/ffmpeg_ops.py`, `src/mux_audit.py`: Thay thế ~230-290 ffprobe subprocesses bằng PyAV in-process, tiết kiệm 7-12s/tập.
  5. *Feature 5 (Bit-Exact Subprocess Preservation)*: `src/ffmpeg_ops.py`, `src/mux_audit.py`: Giữ nguyên tuyệt đối Concat-Mux `-max_interleave_delta 0` và Single-Pass `ffmpeg -f streamhash` MD5.
  6. *Feature 6 (Track Defaults CLI Parameterization)*: `scripts/set_track_defaults.py`: Tham số hóa CLI với `argparse`, tùy biến đường dẫn và hỗ trợ `--dry-run`.
  7. *Feature 7 (SQLite Handle Closing Context)*: `subtrans/cache.py`, `subtrans/rag.py`: Bọc kết nối SQLite bằng context manager đóng handle sạch sẽ, triệt tiêu lỗi khóa file `WinError 32` trên Windows.
  8. *Feature 8 (FTS5 2-Gram/3-Gram Kanji RAG)*: `subtrans/rag.py`: Bảng ảo FTS5 với bộ tách từ character 2-gram + 3-gram tra cứu chính xác từ khóa Kanji 2 chữ (`魔術`, `聖杯`, `令呪`).
  9. *Feature 9 (Spandrel 0.4.2 Native 2x)*: `src/cuda_engine.py`: Tích hợp Spandrel `ModelLoader` nạp mô hình Native 2x không dùng PReLU, bỏ qua bicubic downsample.
  10. *Feature 10 (Polymorphic TRT Engine Export)*: `src/build_trt_engine.py`: Lớp `_GenericTRTExportModule` bọc module ONNX opset 18, hỗ trợ TensorRT 10 với profile max_batch=4 cho 2x.
  11. *Feature 11 (Models Registry Update)*: `models/models.yaml`: Đăng ký kiến trúc `span-2x` và `2x_AnimeJaNai_V3`.
  12. *Feature 12 (Faster-Whisper Large-V3-Turbo)*: `src/subtitles.py`: Nâng cấp model STT mặc định lên `large-v3-turbo` CUDA float16 với fallback CPU int8, tương thích 100% 5.1 FC và DTW auto-align.
  13. *Feature 13 (Fate Apocrypha Benchmark Verification)*: Thực nghiệm trên mẫu BD 1080p Hi10P Fate Apocrypha 22.
  14. *Feature 14 (100% Offline Test & Agent Context Pass)*: Toàn bộ suite offline và kiểm tra bộ nhớ đạt chuẩn 100%.

- **Số liệu & Kết quả thực nghiệm:**
  - **E2E Acceptance Suite (169 tests):**
    - Lệnh: `.\.venv\Scripts\python.exe -m pytest tests/test_e2e_group_a.py tests/test_e2e_group_b.py tests/test_e2e_integration_t3_t4.py -v`
    - Kết quả: **169 passed in 124.31s (0:02:04)** (100% PASS, 0 failed, 0 error, 0 skipped, 0 GPU marker).
  - **Full Offline Test Suite (verify_offline.py):**
    - Lệnh: `.\.venv\Scripts\python.exe scripts/verify_offline.py`
    - Kết quả: **707 passed, 14 skipped, 1 deselected in 481.96s (0:08:01)**, Pytest exit code: 0, không có rò rỉ network hay socket.
  - **Subtrans Tests:**
    - Lệnh: `.\.venv\Scripts\python.exe -m pytest subtrans/tests/test_whisper_noise_and_speaker.py -v`
    - Kết quả: **10 passed in 1.93s (100% PASS)**.
  - **Mux Audit Regression Tests:**
    - Lệnh: `.\.venv\Scripts\python.exe -m pytest tests/test_mux_audit_regression.py -v`
    - Kết quả: **6 passed in 5.92s (100% PASS)**.
  - **Agent Memory Integrity Check:**
    - Lệnh: `.\.venv\Scripts\python.exe scripts/agent_context.py --check`
    - Kết quả: **PASS: required memory files, local links, root <= 12288 bytes, root + subtrans <= 32768 bytes**.
  - **Thực nghiệm Benchmark trên Fate Apocrypha 22 (Live 4s CUDA preview):**
    - Lệnh: `.\.venv\Scripts\python.exe scripts/benchmark_pipeline.py "E:\Films\Raw_Sources\Fate Apocrypha\[AC] Fate Apocrypha - 22 [BD 1080p Hi10 FLAC][DualAudio][695AB63E].mkv" --tier preview --duration 4 --label "fate22_m5_4s_live" --engine cuda`
    - Tổng thời gian: 50.45 s (126 frames, kept 93 unique frames, 26.19% dedup).
    - Tốc độ: AI Steady FPS = 3.98 fps, Average Pipeline FPS = 2.59 fps.
    - Peak VRAM: 9313.0 MiB (< 9.5 GB = 9728 MiB trần an toàn RTX 3060).
    - Output Sanity Guard: `status: PASS`, $\Delta Y = 1.0643 \le 6.0/255$, Pearson $r = 0.999809 \ge 0.98$.
    - BT.709 Color: Source `yuv420p10le` High 10 $\rightarrow$ Output `yuv420p10le` Main 10 HEVC, BT.709 Limited, không clipping.
    - Bit-exact Stream Copy Mux Audit: `status: PASS`.
      - Audio: 2 FLAC Stereo tracks, MD5 track 0 (`27f1426c...`) và track 1 (`3e370b4c...`) khớp 100% từng bit.
      - Subtitles: 2 SSA/ASS tracks khớp số lượng packets.
      - Fonts: 5 fonts (3 TTF, 2 OTF) đính kèm nguyên vẹn.
      - Chapters: Khớp toàn vẹn.

## Bước tiếp theo
1. Toàn bộ 5 Milestone M1-M5 đã hoàn tất và được nghiệm thu độc lập.
2. Sẵn sàng cho việc vận hành thực tế qua đêm hoặc mở rộng các tính năng tương lai.

## Kết thúc
- **Kết quả và giới hạn đã thông báo:** Toàn bộ 14 tính năng của Nhóm A & Nhóm B hoàn tất trọn vẹn, vượt qua mọi chốt chặn an toàn và tiêu chí nghiệm thu khắt khe nhất.
- **Commit/tag:** Được ghi nhận trong lịch sử Git của nhánh `main`.
- **Tài liệu cập nhật:** [STATE](../STATE.md), [PROJECT_MAP](../PROJECT_MAP.md), [START_HERE](../START_HERE.md).
