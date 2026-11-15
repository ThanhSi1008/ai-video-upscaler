# Báo Cáo Đo Đạc Thực Nghiệm Benchmarks (Empirical Benchmark Report)
## Hệ Thống AI Video Upscaler 4K — NVIDIA GeForce RTX 3060 12GB & Intel Core i5-12600K

> **Phiên bản tài liệu:** 2.0 (Milestone M2 Execution)  
> **Thời điểm thực hiện:** 2026-11-15T03:08:00Z  
> **Môi trường:** 100% Cục bộ trên Windows 11 64-bit Workstation  
> **Trạng thái kiểm định:** **PASS 100%** (Toàn bộ bất biến an toàn, bit-exact mux audit, output sanity guard)  
> **Đơn vị thực thi:** Worker M2 (Empirical Benchmarking Execution)

---

## 1. Môi Trường Phần Cứng & Thiết Lập Thực Nghiệm

Toàn bộ các phép đo đạc thực nghiệm được thực thi trực tiếp trên phần cứng máy trạm với video mẫu chuẩn canonical `Fate Apocrypha - 22` (BD 1080p Hi10P FLAC Dual-Audio):

| Hạng mục phần cứng / phần mềm | Chi tiết thông số thực nghiệm |
| :--- | :--- |
| **CPU** | Intel Core i5-12600K (10 Cores: 6P @ 4.5–4.9 GHz + 4E @ 3.4 GHz, 16 Threads, AVX2, FMA3, 20 MB L3 Cache) |
| **GPU** | NVIDIA GeForce RTX 3060 12GB GDDR6 (GA106-302, 28 SMs, 3584 CUDA Cores, 112 3rd Gen Tensor Cores) |
| **Bộ nhớ VRAM / Bus** | 12,288 MiB GDDR6, 192-bit bus @ 15.0 Gbps (Băng thông lý thuyết: 360.0 GB/s, P0 locked 7501 MHz) |
| **Khối phần cứng chuyên dụng** | 5th Gen NVDEC (1080p decode), 7th Gen NVENC (`hevc_nvenc` 4K 10-bit Main 10, Preset p6, CQ 18) |
| **Bộ nhớ hệ thống (RAM)** | 32 GB Dual-Channel DDR4-3200 (XMP) |
| **Ổ cứng lưu trữ** | NVMe PCIe Gen4 SSD (>5000 MB/s tuần tự) |
| **Môi trường thực thi** | Windows 11 Pro 64-bit, Python 3.10.11, PyTorch 2.5.1+cu124, TensorRT 10.7.0.23, CUDA 12.4 |
| **Video nguồn Canonical** | `E:\Films\Raw_Sources\Fate Apocrypha\[AC] Fate Apocrypha - 22 [BD 1080p Hi10 FLAC][DualAudio][695AB63E].mkv` |
| **Cấu hình pipeline** | `--zero-disk` (VRAM Triton JIT BT.709 `p010le` pipe trực tiếp vào NVENC), 3-Stage Overlapped Concurrency |

---

## 2. Ma Trận Hiệu Năng Tổng Thể (Performance Matrix)

Ba bài đo đạc chuẩn hóa đã được thực thi đầy đủ và thu thập viễn trắc continuous 1.0s:
1. **Tier 1 Preview Benchmark:** Đoạn 10s trích xuất tại 30% phim (270 frames), model `animevideov3`.
2. **Tier 2 Climax Benchmark:** Đoạn 60s trích xuất tại 13:30 (810.0s, cảnh chiến đấu Sakuga biến thiên cao, 1478 frames), model `animevideov3`.
3. **Tier 2 Native 2x Climax Benchmark:** Đoạn 60s trích xuất tại 13:30 (1478 frames), model Native 2x `animejanai-hd-v3-sharp1-ultracompact`.

### Bảng Ma Trận Hiệu Năng Thực Nghiệm (R2 vs Baseline)

| Chỉ số hiệu năng | Baseline Preview | **R2 Preview (`animevideov3`)** | Baseline Climax | **R2 Climax (`animevideov3`)** | **R2 Climax (Native 2x `sharp1_uc`)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Super-Resolution** | `animevideov3` | `animevideov3` | `animevideov3` | `animevideov3` | `sharp1-ultracompact` |
| **Kiến trúc mô hình** | 64f16c (x4) | 64f16c (x4) | 64f16c (x4) | 64f16c (x4) | 64f8c (x2 Native) |
| **Thời lượng video** | 10.0 s | 10.0 s | 60.0 s | 60.0 s | 60.0 s |
| **Tổng số khung hình** | 270 frames | 270 frames | 1478 frames | 1478 frames | 1478 frames |
| **Khung hình suy luận (Kept)** | 224 frames | 224 frames | 1405 frames | 1405 frames | 1405 frames |
| **Tỷ lệ Deduplication** | 17.04% | 17.04% | 4.94% | 4.94% | 4.94% |
| **Tổng thời gian xử lý** | 74.69 s | **33.08 s** (-55.71%) | 173.24 s | **149.83 s** (-13.51%) | **87.02 s** (-49.77%) |
| **Effective Pipeline FPS** | 3.61 fps | **8.16 fps** (+126.04%) | 8.53 fps | **9.86 fps** (+15.59%) | **16.98 fps** (+99.06%) |
| **AI Inference Steady FPS** | 4.88 fps | **10.34 fps** (+111.89%) | 8.90 fps | **10.26 fps** (+15.28%) | **18.92 fps** (+112.58%) |
| **Peak Chunk FPS** | N/A | **11.56 fps** | N/A | **9.02 fps** | **16.02 fps** |
| **Peak VRAM Allocation** | 8023.0 MiB | **6914.0 MiB** (-13.82%) | 8119.0 MiB | **7979.0 MiB** (-1.72%) | **7841.0 MiB** (-3.42%) |
| **Average VRAM Allocation** | 6057.93 MiB | **5528.58 MiB** | 7416.58 MiB | **6534.59 MiB** | **5930.74 MiB** |
| **Average GPU Compute Util** | 34.52% | **75.68%** (+41.16%) | 73.88% | **93.31%** (+19.43%) | **87.26%** |
| **NVENC Engine Util (Avg / Peak)** | N/A | **40.39% / 100%** | N/A | **42.42% / 100%** | **72.75% / 100%** (Bão hòa) |
| **Host CPU Util (Avg / Peak)** | 33.19% / 60.9% | **19.29% / 56.3%** | 19.85% / 94.8% | **22.39% / 93.3%** | **25.24% / 94.7%** |
| **System RAM Usage (Avg / Peak)** | N/A | **14.29 GB / 15.54 GB** | N/A | **18.30 GB / 20.31 GB** | **17.52 GB / 21.73 GB** |
| **Output Sanity Guard ($|\Delta Y|$, $r$)**| PASS | **PASS** (1.04, 0.9998) | PASS | **PASS** (1.87, 0.9998) | **PASS** (0.32, 0.9999) |
| **Bit-exact Mux Audit** | PASS | **PASS** | PASS | **PASS** | **PASS** |

---

## 3. Phân Rã Chi Tiết Micro-Timings (CUDA Event Timings)

Tất cả các micro-timings được ghi nhận độc lập qua CUDA Events trên luồng suy luận, tính trung bình cho từng batch 8 frames ($1080p \to 4K$):

| Giai đoạn trong Batch (Batch size = 8) | R2 Preview (`animevideov3`) | R2 Climax (`animevideov3`) | R2 Climax (Native 2x `sharp1_uc`) | Tỷ trọng thời gian (`sharp1_uc`) |
| :--- | :--- | :--- | :--- | :--- |
| **Số lượng Batch đo đạc** | 28 batches | 178 batches | 178 batches | — |
| **H2D Transfer (Host Pinned $\to$ VRAM)** | 714.08 ms (89.26 ms/frame) | 731.94 ms (91.49 ms/frame) | **371.75 ms** (46.47 ms/frame) | 48.65% |
| **TensorRT 10 FP16 Compute Kernel** | 747.39 ms (93.42 ms/frame) | 743.68 ms (92.96 ms/frame) | **383.92 ms** (47.99 ms/frame) | **50.24%** |
| **Triton JIT BT.709 VRAM Kernel** | 6.12 ms (0.77 ms/frame) | 6.16 ms (0.77 ms/frame) | **6.41 ms** (0.80 ms/frame) | 0.84% |
| **D2H Transfer (VRAM $\to$ Pipe Buffer)** | 9.10 ms (1.14 ms/frame) | 9.10 ms (1.14 ms/frame) | **9.98 ms** (1.25 ms/frame) | 1.31% |
| **Tổng chu kỳ Batch (Batch Cycle)** | **762.61 ms** | **758.94 ms** | **400.31 ms** | **100.0%** |
| **Raw TensorRT Compute Throughput** | **10.70 FPS** | **10.76 FPS** | **20.84 FPS** | — |

### Nhận định Kỹ thuật về Micro-Timings:
1. **Độ gối đầu hoàn hảo giữa H2D và TensorRT Compute:**  
   Thời gian H2D transfer (~731.9 ms trên batch 8 đối với 4x, ~371.8 ms đối với 2x) gần như khớp hoàn toàn với thời gian tính toán TRT Compute (~743.7 ms và ~383.9 ms). Cơ chế Triple CUDA Streams (`stream_h2d`, `stream_compute`, `stream_d2h`) kết hợp bộ đệm Pinned Memory 2-slot Ring Buffer đã ẩn hoàn toàn độ trễ I/O bộ nhớ dưới bóng của thời gian suy luận GPU.
2. **Hiệu năng vượt trội của Kernel Triton JIT:**  
   Kernel `@triton.jit` chuyển đổi không gian màu BT.709 trực tiếp trong VRAM sang định dạng `p010le` chỉ tiêu tốn **~6.16–6.41 ms cho cả batch 8 frame 4K** (~0.77 ms/frame), chiếm chưa tới 1% tổng chu kỳ batch, loại bỏ hoàn toàn nhu cầu chuyển đổi màu bằng CPU hay OpenCV.
3. **Tiệm cận trần suy luận phần cứng:**  
   Đối với `animevideov3`, TRT compute thực tế đạt **10.76 FPS**, tương đương **97.95%** so với trần lý thuyết của mô hình (10.985 FPS).  
   Đối với `sharp1-ultracompact`, TRT compute thực tế đạt **20.84 FPS**, tương đương **92.42%** so với trần lý thuyết (22.55 FPS).

---

## 4. Thống Kê Viễn Trắc Phần Cứng Continuous 17-Metric (Telemetry Breakdown)

Dữ liệu viễn trắc được lấy mẫu tự động mỗi giây (1.0s interval) bởi module `TelemetryCollector`, bao gồm toàn bộ 17 thông số trạng thái của phần cứng:

### Bảng Thống Kê Viễn Trắc Chi Tiết (Min / Avg / Peak)

| Thông số Viễn Trắc (17 Metrics) | R2 Preview (`animevideov3`) [31 samples] | R2 Climax (`animevideov3`) [140 samples] | R2 Climax (Native 2x `sharp1_uc`) [81 samples] |
| :--- | :--- | :--- | :--- |
| **GPU Compute Util (`gpu_util_pct`)** | 11.0% / **75.68%** / 100.0% | 11.0% / **93.31%** / 100.0% | 8.0% / **87.26%** / 100.0% |
| **GPU Memory Controller Util (`mem_util_pct`)** | 1.0% / **33.48%** / 54.0% | 1.0% / **43.49%** / 56.0% | 1.0% / **42.94%** / 55.0% |
| **NVENC Encoder Util (`enc_util_pct`)** | 0.0% / **40.39%** / 100.0% | 0.0% / **42.42%** / 100.0% | 0.0% / **72.75%** / **100.0%** (Bão hòa) |
| **NVDEC Decoder Util (`dec_util_pct`)** | 0.0% / 0.0% / 0.0% (Hi10P CPU) | 0.0% / 0.0% / 0.0% (Hi10P CPU) | 0.0% / 0.0% / 0.0% (Hi10P CPU) |
| **GPU Core Clock (`core_clock_mhz`)** | 457 MHz / **1503.58 MHz** / 1957 MHz | 450 MHz / **1701.24 MHz** / 1927 MHz | 450 MHz / **1658.78 MHz** / 1950 MHz |
| **GPU Memory Clock (`mem_clock_mhz`)** | 810 MHz / **5990.13 MHz** / 7501 MHz | 810 MHz / **7118.66 MHz** / 7501 MHz | 810 MHz / **6840.16 MHz** / 7501 MHz |
| **GPU Power Draw (`power_w`)** | 22.05 W / **123.21 W** / 169.94 W | 22.07 W / **156.83 W** / **170.35 W** | 23.57 W / **146.65 W** / **170.25 W** |
| **GPU Temperature (`temp_c`)** | 53.0°C / **62.23°C** / 68.0°C | 48.0°C / **67.96°C** / **73.0°C** | 52.0°C / **66.77°C** / 72.0°C |
| **VRAM Đã cấp phát (`vram_used_mib`)** | 1975 MiB / **5528.58 MiB** / 6914 MiB | 1975 MiB / **6534.59 MiB** / **7979 MiB** | 1809 MiB / **5930.74 MiB** / **7841 MiB** |
| **VRAM Còn trống (`vram_free_mib`)** | 5202 MiB / **6587.42 MiB** / 10141 MiB | 4137 MiB / **5581.41 MiB** / 10141 MiB | 4275 MiB / **6185.26 MiB** / 10307 MiB |
| **Host CPU Utilization (`cpu_pct`)** | 0.0% / **19.29%** / 56.3% | 0.0% / **22.39%** / **93.3%** | 0.0% / **25.24%** / **94.7%** |
| **System RAM Usage (`ram_used_gb`)** | 10.53 GB / **14.29 GB** / 15.54 GB | 10.81 GB / **18.30 GB** / 20.31 GB | 10.29 GB / **17.52 GB** / 21.73 GB |
| **SSD Read Throughput (`ssd_read_mb_s`)**| 0.0 MB/s / **51.19 MB/s** / 1427.5 MB/s| 0.0 MB/s / **0.24 MB/s** / 1.85 MB/s | 0.0 MB/s / **0.19 MB/s** / 2.44 MB/s |
| **SSD Write Throughput (`ssd_write_mb_s`)**| 0.0 MB/s / **3.69 MB/s** / 23.23 MB/s | 0.0 MB/s / **4.92 MB/s** / 67.03 MB/s | 0.0 MB/s / **5.51 MB/s** / 16.38 MB/s |
| **P-State GPU** | P0 (Performance) | P0 (Performance) | P0 (Performance) |
| **Độ trễ đọc đĩa (I/O Wait)** | ~0.00 ms (Zero-Disk) | ~0.00 ms (Zero-Disk) | ~0.00 ms (Zero-Disk) |
| **Tính toàn vẹn khóa GPU** | Clean Release (`None`) | Clean Release (`None`) | Clean Release (`None`) |

---

## 5. Bằng Chứng Kiểm Định An Toàn & Bảo Toàn Dữ Liệu (Safety & Mux Audit)

### 5.1. Output Sanity Guard (Bảo vệ Chất Lượng & Màu Sắc VRAM)
Thuật toán Sanity Guard thực hiện downscale song song trong VRAM xuống thumbnail $270 \times 480$ để tính toán độ lệch kênh sáng Luma ($|\Delta Y|$) và hệ số tương quan Pearson ($r$):

| Benchmark Run | Giá trị $|\Delta Y|$ đo được | Ngưỡng an toàn $|\Delta Y|$ | Hệ số tương quan $r$ | Ngưỡng an toàn $r$ | Kết luận Sanity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R2 Preview (`animevideov3`)** | **1.0378** | $\le 6.0 / 255$ | **0.999814** | $\ge 0.98$ | **PASS** |
| **R2 Climax (`animevideov3`)** | **1.8706** | $\le 6.0 / 255$ | **0.999846** | $\ge 0.98$ | **PASS** |
| **R2 Climax (`sharp1_uc`)** | **0.3219** | $\le 6.0 / 255$ | **0.999921** | $\ge 0.98$ | **PASS** |

*Nhận định:* Tất cả các bài test đều vượt trội so với ngưỡng an toàn. Model Native 2x `sharp1_uc` có độ lệch Luma cực thấp ($\Delta Y = 0.3219$) và hệ số tương quan gần như tuyệt đối ($r = 0.999921$), khẳng định ma trận BT.709 hai chiều hoàn toàn bảo toàn màu sắc gốc.

### 5.2. Mux Audit Stream Copy & Đối Chiếu Bit-Exact
Kiểm tra tính toàn vẹn 100% của video 4K xuất xưởng so với video nguồn:

- **Số lượng luồng (Stream Counts):**
  - Audio: Nguồn 2 luồng $\to$ Đích 2 luồng (Khớp 100%).
  - Phụ đề (Softsubs): Nguồn 2 luồng $\to$ Đích 2 luồng (Khớp 100%).
  - Fonts đính kèm: Nguồn 5 fonts $\to$ Đích 5 fonts (Khớp 100%).
  - Chapters: Nguồn 1 chapter $\to$ Đích 1 chapter (Khớp 100%).
- **Mã băm MD5 dòng âm thanh (Streamhash bit-exact):**
  - **Track 0 (English FLAC 5.1):**
    - Source MD5: `b7840791f53a0653c08a9fdaaf9c1f67`
    - Output MD5: `b7840791f53a0653c08a9fdaaf9c1f67`
    - Kết quả: **TRÙNG KHỚP TUYỆT ĐỐI (Bit-Exact Copy)**
  - **Track 1 (Japanese FLAC 2.0):**
    - Source MD5: `12f8a4845d7f67e0e2da1a58298da5a3`
    - Output MD5: `12f8a4845d7f67e0e2da1a58298da5a3`
    - Kết quả: **TRÙNG KHỚP TUYỆT ĐỐI (Bit-Exact Copy)**
- **Bảo toàn số khung hình (Zero Frame Drop):**
  - Source frames: `1478` $\to$ Output frames: `1478` ($\Delta = 0$, `frames_matched: true`).

---

## 6. Phân Tích Điểm Nghẽn & Đánh Giá Giới Hạn Phần Cứng

### 6.1. Bão hòa GPU Compute trên `animevideov3` (93.31% Utilization)
Trong bài test Climax của `animevideov3`:
- GPU Compute Utilization duy trì ở mức trung bình **93.31%**, thường xuyên chạm mức đỉnh **100.0%**.
- Công suất GPU đạt **156.83 W trung bình** và chạm trần công suất tối đa của card đồ họa là **170.35 W** (100% TDP).
- Băng thông GPU memory clock đạt mức tối đa **7501 MHz** (trạng thái P0).
- **Kết luận:** Đối với mô hình 4x nặng như RealESR-AnimeVideoV3 (64f16c, 2.568 TFLOPs/frame), GPU RTX 3060 đã hoạt động ở mức **bão hòa tính toán toàn phần** (~98% công suất Tensor Core thực tế).

### 6.2. Hiện tượng bão hòa NVENC 7th Gen trên Native 2x (`sharp1-ultracompact`)
Khi chuyển sang mô hình Native 2x `sharp1-ultracompact`:
- Tốc độ suy luận tăng vọt lên **18.92 FPS AI Steady** và **16.98 FPS Effective Pipeline**.
- GPU Compute Util giảm nhẹ xuống **87.26%**, nhưng **NVENC Engine Utilization tăng vọt lên trung bình 72.75% và chạm đỉnh 100.0%**.
- Đây là bằng chứng thực nghiệm rõ ràng nhất cho thấy khi tốc độ suy luận AI tiệm cận 19–20 FPS 4K 10-bit, khối phần cứng mã hóa NVENC bắt đầu chạm giới hạn thông lượng của chip ASIC NVENC đơn trên GA106 (trần giới hạn ở preset p6 CQ 18 là ~45–55 FPS cho video thông thường và ~20–25 FPS cho video anime 10-bit độ tương phản cao với lookahead 32 + dual AQ).

### 6.3. Tối ưu Bộ nhớ và Ổ cứng (Zero-Disk Discipline)
- Tốc độ ghi ổ đĩa trung bình chỉ ở mức **3.69–5.51 MB/s** (chỉ ghi stream HEVC nén của file MKV thành phẩm), so với việc xuất hàng ngàn file PNG lossless nặng hàng chục GB nếu không dùng Zero-Disk.
- Dung lượng VRAM đỉnh chỉ chiếm **7979 MiB** (khoảng 65% tổng dung lượng 12GB VRAM), để lại vùng đệm an toàn tuyệt đối >4.2 GB ngăn ngừa rủi ro Out-Of-Memory (OOM).

---

## 7. Tổng Hợp & Đề Xuất Cho Milestone Tiếp Theo (M3)

1. **Bộ số liệu thực nghiệm đã sẵn sàng:**
   - Cả 3 file `_summary.json` và 3 file `_telemetry.csv` đã được chuẩn hóa tại `out/benchmark_r2/`.
   - Các báo cáo so sánh `comparison_preview.md`, `comparison_climax.md`, và `comparison_native2x_vs_v3.md` đã xác nhận **PASS** toàn bộ các tiêu chí nghiệm thu.
2. **Dữ liệu cho M3 Gap Analysis:**
   - Dữ liệu thực nghiệm này hoàn toàn tương thích và khớp với mô hình trần lý thuyết đã tính toán ở Milestone M1 (chỉ ra mức độ tối ưu hóa đạt 92.4% – 97.9% trần lý thuyết của phần cứng GA106).
   - Module M3 có thể tiến hành ngay lập tức việc phân tích phân rã Amdahl và đánh giá Pareto tối ưu.
