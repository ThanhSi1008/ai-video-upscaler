# Handoff: Khảo sát mẫu Sakuga khắc nghiệt nhất, Đo đạc Baseline & Vòng lặp Tối ưu Bão hòa Pareto (/teamwork)

## Yêu cầu và phạm vi
- Ngày/múi giờ: `2026-11-14T18:48:00+07:00`, Asia/Bangkok (+07:00).
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu qua `/plan /teamwork` khảo sát toàn bộ `E:\Films\Raw_Sources` chọn ra tập hoành tráng nhất để đo chỉ số thực nghiệm, ghi logs/telemetry, thực hiện vòng lặp tối ưu hóa toàn diện `ai-video-upscaler`, tái kiểm thử sau mỗi lần tối ưu cho đến khi đạt điểm bão hòa Pareto hoàn hảo và báo cáo tổng kết.
- Trạng thái: **HOÀN THÀNH 100% (`VICTORY CONFIRMED`)**.

## Mốc mã nguồn và điều phối
- Root, branch, commits: `e:\ai-video-upscaler`, branch `main`, chuỗi commit triển khai `028f0ff` -> `d786a80` -> `101aced`.
- Mẫu chuẩn tuyển chọn từ 261 video (`160.4 GB`): `E:\Films\Raw_Sources\Fate Apocrypha\[AC] Fate Apocrypha - 22 [BD 1080p Hi10 FLAC][DualAudio][695AB63E].mkv` (`H.264 Hi10P` CPU decode, đại cảnh Karna vs Sieg đạt `95.06%` unique frames ở Climax).

## Đã làm và bằng chứng
1. **Hạ tầng Benchmark & Kiểm thử Hộp đen (R3):**
   - Triển khai `scripts/telemetry_collector.py` (17 chỉ số phần cứng chu kỳ `1.0s`), `scripts/benchmark_pipeline.py`, `scripts/compare_benchmarks.py` và `tests/test_benchmark_infra.py` (`30/30` PASS).
   - Xây dựng bộ kiểm thử hộp đen 4 tầng `tests/e2e/` (`115/115` PASS) bảo vệ toàn bộ bất biến kỹ thuật.
2. **Thiết lập Golden Baseline & 5 Can thiệp Kiến trúc (R1, R2):**
   - Lưu hồ sơ chuẩn tại `benchmarks/baseline_summary.md` và `out/benchmark_baseline/`.
   - **Target 1 (`src/ffmpeg_ops.py`, `src/pipeline.py`):** Bù gói tin `Open-GOP CRA discard` (`_count_leading_discard_packets`) và xác thực số khung hình khi resume (`verify_segment_resume_frames`).
   - **Target 2 (`src/trt_zero_copy_upscaler.py`):** Mở rộng ghi pipe `P010LE` lên `8 MiB` `memoryview` zero-copy và `_PipeDrainWorker` bất đồng bộ kèm khóa chống deadlock (`d786a80`).
   - **Target 3 (`src/dedup.py`):** Chỉ mục `Cyclic Cel` 3 tầng (`64x36` thumbnail + BLAKE2b + SIMD uint8) tăng tốc tra cứu `27.02x` (`3612.6ms -> 133.7ms`).
   - **Target 4 (`src/trt_zero_copy_upscaler.py`):** Đo vi thời gian GPU phi phong tỏa bằng `torch.cuda.Event(enable_timing=True)`.
   - **Target 5 (`src/dedup.py`):** Phân tách ngưỡng `DIS Optical Flow` toàn cục (`0.35px`) và cục bộ (`1.50px` @ `12%`) cùng chính sách `sakuga` / `dis-sakuga`.
3. **Số liệu Đối chứng & Bão hòa Pareto (R4 — `benchmarks/final_optimization_report.md`):**
   - **Tier Preview (`10s`, `270` frames):** `3.61 -> 8.05 fps` (**`+122.99%`**), thời gian `74.69s -> 33.52s` (`-55.12%`), CPU `33.2% -> 18.5%`, Mux Audit `PASS`.
   - **Tier Climax (`60s`, `1478` frames):** `8.53 -> 9.47 fps` (**`+11.02%`**), **AI Steady-State `8.90 -> 10.20 fps` (`+14.61%`)**, đạt **`92.85%` trần vật lý TensorRT FP16 GA106 (`10.985 fps`)**, Mux Audit `PASS` bit-exact.
   - Kiểm thử tổng thể: `verify_offline.py` đạt **`488 passed, 11 skipped, 1 deselected`** (`0` lỗi), `tests/e2e/` đạt **`115/115 passed`**.
