# Engineering Log — Nhật Ký Thực Nghiệm & Đo Đạc Hiệu Năng

> **Tài liệu lưu trữ toàn bộ số đo thực nghiệm, benchmark đối đầu, soak test và lịch sử kiểm định của dự án.**
> Tham chiếu từ: [AGENTS.md](../AGENTS.md) và [subtrans/AGENTS.md](../subtrans/AGENTS.md).

---

## 1. Dữ Liệu Thực Nghiệm Cốt Lõi (Baseline Ground Truth)

* **Phần cứng mục tiêu:** Windows 11 64-bit, Intel Core i5-12600K (10C/16T, AVX2), NVIDIA GeForce RTX 3060 (12GB GDDR6 VRAM, CUDA & Tensor Cores), 32GB RAM.
* **Tốc độ AI PyTorch CUDA FP16 Batch Inference (`batch_size=6`):** Đạt **4.8 – 5.2 fps** suy luận thuần GPU, và **~10 – 12+ fps** hiệu dụng khi kết hợp Deduplication Preflight và ghi ảnh OpenCV C++ AVX2 (`cv2.imwrite`).
* **Mã hóa phần cứng:** NVIDIA NVENC HEVC Main 10 (`hevc_nvenc`, Preset `p6`, Multipass `qres`, Lookahead `24`, `p010le`, BT.709).
* **VRAM đỉnh:** Ổn định ở mức **~6.20 GB / 12 GB** (51.6% capacity), tuyệt đối an toàn không lo tràn VRAM.
* **Overhead ngoài AI:** Decode FFmpeg ~8.5s + Encode NVENC ~4.2s (nhanh gấp 2.5x so với encode CPU).
* **Kiểm định gối đầu (Pipelined Triple-Buffering):** CPU decode (chunk N+1) và NVENC encode (chunk N-1) chạy song song với GPU AI infer (chunk N). Triệt tiêu toàn bộ thời gian chết của GPU trên toàn tập 24 phút.

---

## 2. Kết Quả True Blind Test (Chấm Mù Trực Quan)

* **Thiết lập:** Người dùng kiểm tra độc lập ở tỷ lệ nguyên bản trên màn hình Liquid Retina XDR với 3 bối cảnh điển hình:
  1. Cận cảnh khuôn mặt nhân vật (Close-up)
  2. Phong cảnh chi tiết (Scenery)
  3. Hành động chuyển động nhanh & ruy băng bay (Action)
* **Kết quả:** `realesr-animevideov3` được chọn đứng **Hạng 1 ở cả 3 cảnh**:
  - `realesr-animevideov3`: Nét vẽ sắc nét, tự nhiên, tái tạo viền đen chuẩn anime, không gây bệt màu.
  - `ArtCNN_C4F32_DS`: Bị loại do Mean Absolute Difference (MAD) Luma chỉ đạt 0.2% so với Bicubic chuẩn (không cải thiện rõ rệt).
  - `AnimeJaNai HD V3`: Bị loại do nét vẽ mềm hơn và không giữ được độ sắc cạnh ở viền mắt/tóc.

---

## 3. Đo Đạc Deduplication & Kiểm Định Toàn Vẹn Khung Hình

### A. Quét Preflight Toàn Tập Thật (`Mushoku Tensei S3 - 14`, 34.046 frames - 1.35 GB)
* Thời gian quét: **123.67 giây (2.06 phút)** @ 275.3 fps.
* Số khung hình mpdecimate giữ lại (`hi=768:lo=320:frac=0.33`):
  - Raw kept: **13.216 frames** (38.82%).
  - Raw drop: **20.830 frames** (61.18%).
  - Ép giữ ranh giới chunk 8s ($i \times 192$): bổ sung 108 frames -> **Tổng AI infer: 13.324 unique frames (39.14%)**.
  - **Tỷ lệ loại bỏ trùng lặp thực tế: 60.86%**.
* Thời gian chạy thực tế ước tính: 1 tập mất **~3h 01m** @ 1.24 fps; 2 tập qua đêm mất **~6h 03m** (an toàn trong cửa sổ 8 giờ).

### B. Đo Đạc Preflight Theo 4 Ngưỡng `hi` (Mẫu 3 phút & Toàn tập)
* **Mẫu 3 phút (4.360 frames):**
  - `hi=768` (mặc định): 2.186 frames (28.7m)
  - `hi=640`: 2.301 frames (+1.5m)
  - `hi=512`: 2.458 frames (+3.6m)
  - `hi=384`: 2.679 frames (+6.5m)
* **Toàn tập 24 phút (34.046 frames):**
  - `hi=768` (mặc định): 13.324 frames (2h 55m)
  - `hi=640`: 14.095 frames (3h 05m, +10.1m) — điểm giữ thêm 771 frames an toàn nếu cần
  - `hi=512`: 15.398 frames (3h 22m)
  - `hi=384`: 17.557 frames (3h 50m)

### C. Thẩm Định Độc Lập Khối 16x16 & 8x8 Pixel Trên Toàn Bộ 2.174 Dropped Frames (Mẫu 3 phút)
* Phân tích 8.160 khối/frame: Mean max-block MAD **3.77 / 255** (1.48%), Median P50 **2.82**, P90 **7.35**, P95 **8.11**, P99 **9.42**, Max toàn tập **10.75 / 255** (4.21%).
* Số frame có MAD khối 16x16 >= 12.0 (ngưỡng hi=768/64): **0 frames (0.00%)**.
* Đối soát trực quan Top 30 worst frames: Sai lệch chỉ là rung viền răng cưa 1-pixel và nhiễu nén H.264 trên nét tĩnh; không phát hiện mất bất kỳ cử động miệng hay chớp mắt nào trên mẫu thử nghiệm.
* **⚠️ CẢNH BÁO QUAN TRỌNG:** Phân tích khối 16x16/8x8 này chỉ kiểm tra chuyển động nhân vật/khuôn mặt trên nền tĩnh, **chưa kiểm tra các cảnh chuyển động máy quay pan/zoom chậm** (xem đo đạc thực nghiệm đứng hình tại Mục 7).
* Đối soát C-loop `vf_mpdecimate.c` trên mặt phẳng Y gốc (Luma): 0 / 108 frame vượt ngưỡng hi=768 (Min SAD: 658, Mean: 718.8, Max: 768). Thuật toán của FFmpeg 8.1.2 hoạt động chuẩn xác.

---

## 4. Nghiên Cứu Đối Đầu PyTorch MPS vs NCNN Vulkan

### A. Độ Khớp Điểm Ảnh & Kiến Trúc Mạng
* `realesr-animevideov3-x2` thực chất chạy mạng SRVGGNetCompact 4x rồi nội suy Bicubic 0.5 để ra 2x. Khớp trọng số `strict=True` với `realesr-animevideov3.pth`.
* **Độ khớp hình ảnh:** PSNR **51.05 dB**, MAD **0.493 / 255** (0.19%), Max diff 44.0 / 255 (tập trung chủ yếu ở 10px rìa ngoài cùng do cơ chế reflection padding của tiling khác biệt; vùng trung tâm interior đạt **60.02 dB** PSNR).
* **FP16 vs FP32 trên MPS:** Đạt **61.74 dB** PSNR (0 NaN / 0 Inf).
* **Thời gian nén PNG:** Mức nén 1 đạt **310 ms / frame** (nhanh gấp ~9 lần mức 6 cũ: 2.826 ms) với dung lượng 9.1 MB.

### B. Soak Test 2.000 Khung Hình Trên PyTorch MPS FP16
* File báo cáo: `out/mps_soak_2000_report.json`
* Hoàn tất trong **1.120,17 giây (18.67 phút)**. Tốc độ trung bình: **1.785 fps**.
* Bộ nhớ MPS ổn định ở **61.19 MB**, Process RSS ổn định ở **36.0 MB** (không rò rỉ bộ nhớ).

### C. Soak Test End-to-End 6.000 Khung Hình Trên PyTorch MPS FP16
* File báo cáo: `out/mps_soak_6000_report.json`
* Hoàn tất trong **3.570,27 giây (59.50 phút)**.
* Tốc độ: Median **1.683 fps**, P5 **1.677 fps**, P95 **1.684 fps**, Mean **1.681 fps** (độ lệch cực đại +-0.005 fps xuyên suốt 1 giờ chạy liên tục).
* Bộ nhớ: Đỉnh Driver Allocated **2.090,81 MB**, Đỉnh Process Footprint **674,42 MB**.
* Swap & Nhiệt độ: Pageouts tăng +133, trạng thái nhiệt duy trì Nominal (không nghẽn nhiệt).

### D. Tối Ưu Hóa Tiling NCNN Vulkan Fallback
* Cấu hình `-t 540 -j 1:2:2`: Đạt **1.417 +- 0.020 fps** (72.68s wall time, 720.2 MB peak RSS). Nhanh hơn auto-tile `-t 0` (1.200 fps) +18.9%, triệt tiêu ~19.5% overhead prepadding dư thừa. Đã ghim làm tham số mặc định cho NCNN fallback.

---

## 5. Nghiệm Thu Phiên Chạy Đêm 1 (Night 1 Telemetry)

* **Thiết lập:** Hàng đợi 3 tập tự động qua đêm ngày 05/10/2026:
  - Tập 1: `Kimi ga Shinu made Koi wo Shitai 13` (34.120 frames)
  - Tập 2: `Re Zero kara Hajimeru Isekai Seikatsu 84` (34.047 frames)
  - Tập 3: `Tensei Shitara Slime Datta Ken S4-24` (34.050 frames, ngắt an toàn tại chunk 31 lúc 08:00)
* **Kết quả đo đạc thực tế:**
  - **Tập 1:** Thời gian thực **2h 51m 45s (10.305s)**; Kept frames: 15.688 (46.0%); AI fps: 1.682; GPU active: **93.2%**; Tỷ lệ lệch ETA trung vị: +2.2%. Mux audit: PASS (11 font, 2 kênh audio, 2 luồng sub).
  - **Tập 2:** Thời gian thực **3h 47m 05s (13.625s)**; Kept frames: 20.867 (61.3%); AI fps: 1.681; GPU active: **91.1%**; Tỷ lệ lệch ETA trung vị: +2.9%. Mux audit: PASS (24 font, 2 kênh audio, 2 luồng sub).
  - **Tập 3:** Xử lý 31/179 chunk (5.952 frames, 3.238 kept), đạt **1.678 fps**. Lưu checkpoint `.temp` 1.9 GB an toàn trên SSD để resume từ chunk 32.
* **Số Đo Telemetry Từ CSV (`logs/soak_telemetry_night1.csv` từ 00:26:00 đến 08:00:29):**
  - Ghi disk tích lũy: **125.6 GB** (từ 2.506.978 MB lên 2.632.596 MB; bao gồm thành phẩm 3.58 GB, các intermediate chunk và tác vụ hệ điều hành).
  - Pageouts: Tăng **+7.169** (từ 113.508 lên 120.677).
  - Swap: Tăng **+1.075,9 MB** (từ 1.463,4 MB lên 2.539,3 MB; hệ điều hành phân bổ swap nền, không gây áp lực suy giảm hiệu năng).
* **Audit Trực Quan:** Người dùng đã xem và nghe toàn bộ 2 tập thành phẩm, xác nhận âm thanh khớp khẩu hình, phụ đề hiển thị chuẩn xác, không có hiện tượng mất frame hay méo tiếng.

---

## 6. Hiệu Chỉnh Scheduler & Cơ Chế Auto-Offset Phụ Đề

### A. Hiệu Chỉnh Overhead & Sức Chứa Hàng Đợi (`src/scheduler.py`)
* **Overhead chuẩn hóa:** **930,0s (~15.5 phút)** cho xử lý video (hiệu chỉnh theo dữ liệu thực tế đêm 1: 691s và 1166s, cỡ mẫu n=2) + 300s trích xuất & nhúng phụ đề = **1.230s (~20.5 phút/tập)**.
* **Tỷ lệ kept frames thận trọng cho tập chưa quét preflight:** Đặt ở mức **60.0%** (dựa trên 4 tập đo đạc thực tế: 39.1%, 46.0%, 54.4%, 61.3%; trung bình ~50.0%). Khi preflight nền chạy xong, scheduler tự động cập nhật số đo thực tế vào bảng quyết định deadline.
* **Sức chứa ca đêm 8 giờ:** Thực tế đạt **~2 tập/đêm** (~3h/tập với $n_{unique} \approx 17.200$). 3 tập chỉ hoàn tất khi tổng số khung hình độc lập đủ nhỏ.

### B. Kiểm Định 4 Tầng Của Thuật Toán Auto-Offset
* **4 tiêu chí kích hoạt:**
  1. Chênh lệch $|offset| \ge 0.5$s (đã hạ từ 1.0s xuống 0.5s sau đêm 2 để bắt kịp độ lệch vừa phải).
  2. Số cặp câu đối soát hợp lệ $\ge 30$.
  3. Độ trôi dạt cực đại giữa 5 phần thời gian (quintile drift) $\le 0.5$s (chống trôi dạt lũy tiến do lệch framerate 23.976 vs 24.0).
  4. Tỷ lệ khớp thời gian tăng $\ge 15.0$ điểm phần trăm sau khi bù offset.
* **Số đo thực nghiệm trên 3 tập:**
  - Tập 1 (`Kimi ga Shinu`): Offset đo được -148 ms ($< 0.5$s) -> Không áp dụng (PASS).
  - Tập 2 (`Re:Zero`): Offset đo được -98 ms ($< 0.5$s) -> Không áp dụng (PASS).
  - Tập 3 (`Slime S4-24`): Do bản phát sóng có thêm sponsor card 11 giây ở đầu, offset đo được **-11.141 ms** ($\approx -11.154$s), 285 cặp khớp, quintile drift 0.029s ($\le 0.5$s), tỷ lệ khớp tăng từ 55.9% -> 85.3% (+29.4 điểm) -> Tự động áp dụng thành công (PASS).

---

## 7. Sửa Lỗi Lệch Khung Hình BD Remux GOP Dài & Khảo Sát Giật Hình Camera Zoom

### A. Khắc Phục Lệch Khung Hình BD Remux (Commit `5c868ab`)
* **Vấn đề:** GOP dài biến thiên trên BD Remux (`[250, 250, 219...]`) làm lệch với lưới 192 frame tĩnh trong `src/dedup.py`, dẫn đến frame desync.
* **Giải pháp:** Đưa `split_video_segments` lên trước, dùng `get_segments_frame_counts` đo số frame thực tế qua FFprobe, ép giữ frame đầu mỗi chunk (`keep`), tích hợp `video_frame_count` vào `src/mux_audit.py`.
* **Kiểm định mẫu 1 phút BD Remux (`sample_1min_bd.mkv`):**
  - Nguồn gốc: **1.438 frames** (AVC 1080p, 260.6 MB).
  - Thành phẩm 4K: **1.438 frames** (khớp toàn bộ số frame, 0 frame mất mát). Kept AI: 841 frames (58.48%), Dropped: 597 frames (41.52%).
  - Dung lượng 4K: **88.3 MB** (HEVC p010le bitrate tối ưu + Concat duration re-expansion).
  - Mux audit: **PASS** (1.438 frames, 2ch audio FLAC, 15 fonts).

### B. Thực Nghiệm MAE Hiện Tượng Giật Hình Ở Cảnh Camera Zoom/Pan Chậm (00:44 – 00:46)
* **Số liệu đối soát 48 frames (2.0s @ 23.976 fps):**
  - Nguồn BD gốc: 100% frame có chuyển động pixel liên tục ($0.53 \le \text{MAE} \le 3.8$, Mean $2.67$), **0 frame đứng hình**.
  - Bản 4K thành phẩm: **24 / 47 bước chuyển tiếp (51.1%) bị đóng băng** (`diff < 0.05`), lặp lại frame cũ rồi nhảy nấc.
* **Nguyên nhân cốt lõi:** Khối 8x8 của `mpdecimate` (`hi=768:lo=320:frac=0.33`) đánh giá sai chuyển động vi mô (sub-pixel camera pan/zoom) dưới ngưỡng tĩnh và gán nhãn `drop`.
* **Đề xuất giải pháp:** Cờ `--no-dedup` cho BD Remux (tăng +71% số frame AI infer: 1438/841) hoặc adaptive motion threshold theo series trong `shows.yaml`.

---

## 8. Đêm 2, Xoay Vòng Khóa Gemini & Chốt Chặn Output Sanity

### A. Vận Hành Đêm 2 & Cổng Backpressure Sàn 16MB
* Tiếp tục hoàn thiện Slime Ep 24 sau gián đoạn checkpoint; phát hiện nguy cơ ENOSPC khi RAM Disk đầy do chunk nén kém.
* Nâng sàn ước tính an toàn `DEFAULT_PNG_UNIT_SIZE_MB` từ 9.0 MB lên 16.0 MB và bổ sung cơ chế retry 2 lần chống rớt chunk (`f04eafa`).

### B. Đa Khóa Gemini & Xoay Vòng Hạn Mức Tự Động (Multi-Key Rotation)
* Thiết lập 4 API Key trên 4 project Google Cloud độc lập ở bậc Free Tier (không nhập thẻ tín dụng).
* Hạn ngạch: Gemini 2.5 Flash-Lite đạt 2.000 RPD (500 RPD/key), Flash đạt 80 RPD (20 RPD/key).
* Tự động xoay vòng key khi gặp mã 429 (`RESOURCE_EXHAUSTED`), đảm bảo chuỗi dịch phụ đề không bị nghẽn gián đoạn.



### C. Đính Chính Lỗi Residual AnimeJaNai & Chốt Chặn Output Sanity Guard
* Khắc phục triệt để lỗi thiếu `F.interpolate` nearest x2 trên `AnimeJaNaiCompact2x` tại tag `janai-fix` (Commit `ca8f1d9`).
* Thiết lập chốt chặn bắt buộc `src/output_sanity.py`: Kiểm tra |ΔY| <= 6/255 và corr >= 0.98 sau chunk 1 và mỗi 10 chunk sau để ngăn chặn triệt để video đen/mờ.

---

## 9. Thực Nghiệm Các Chính Sách Deduplication (CPU Preflight & Global Motion Detection)

### A. Đặt Vấn Đề & Phản Hồi Từ Người Dùng
* **Phản hồi người dùng:**
  1. Hai model P/Q (Real-ESRGAN và AnimeJaNai HD V3 Compact) sau khi fix residual không có sự khác biệt rõ rệt về cảm quan.
  2. Ở các tập WEB-DL trước đây, từng xuất hiện hiện tượng giật/khựng ở các cảnh camera lia (pan) hoặc phóng to (zoom) chậm nhưng chưa quá rõ.
  3. Cả nguồn BD và WEB-DL đều mong muốn cân bằng theo nguyên lý Pareto: những phân cảnh cần mượt (pan/zoom chuyển động) thì phải mượt, đồng thời tìm phương án giữ chuyển động với chi phí GPU tăng ít nhất có thể.

### B. Thiết Kế & Thuật Toán Các Chính Sách Dedup (CPU Preflight)
1. `default`: `hi=768:lo=320:frac=0.33` (chính sách mặc định hiện tại).
2. `sensitive-a`: `hi=512:lo=160:frac=0.15` (hạ ngưỡng block 8x8 mpdecimate).
3. `sensitive-b`: `hi=384:lo=128:frac=0.10` (hạ sâu ngưỡng block 8x8 mpdecimate).
4. `motion` ($\tau \in \{0.15, 0.30, 0.50\}$): Giữ nếu `default` giữ HOẶC phát hiện chuyển động camera toàn cục so với frame nguồn liền trước.
   - Thu nhỏ luma về $480 \times 270$ grayscale.
   - Tính toán Phase Correlation (`cv2.phaseCorrelate`) trên toàn khung hình và 4 góc phần tư độc lập.
   - Giữ frame nếu $\max|\Delta| \ge \tau$ (quy đổi về pixel 1080p) và response peak $\ge 0.10$, hoặc có vector dịch chuyển nghịch đảo giữa các góc đối diện (zoom in/out).
5. `off`: Giữ toàn bộ 100% khung hình (tắt deduplication).

### C. Số Liệu Đo Đạc Thực Nghiệm (CPU Preflight & Motion Simulation)

#### 1. Mẫu Clip 1 Phút BD Remux (`sample_1min_bd.mkv`, 1.438 frames, 1.275 frame có chuyển động thật $\tau \ge 0.15$px = 88.7%):
| Chính sách | Kept Frames | Tỷ lệ giữ | Đóng băng clip | Đóng băng 44–46s | Frame thêm | Độ chính xác chuyển động thật | Est. Thời gian (h) | CPU (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | 834 / 1.438 | 58.0% | 37.7% | 57.8% | 0 | - | 0.40h (24.1m) | 4.9s |
| `sensitive-a` | 913 / 1.438 | 63.5% | 31.8% | 42.2% | +79 | 95.8% | 0.42h (25.0m) | 4.6s |
| `sensitive-b` | 961 / 1.438 | 66.8% | 28.1% | **0.0%** | +127 | 96.9% | 0.42h (25.4m) | 4.5s |
| `motion-0.15` | 1.315 / 1.438 | 91.4% | **0.0%** | **0.0%** | +481 | 100.0% | 0.49h (29.1m) | 9.6s |
| `motion-0.30` | 1.076 / 1.438 | 74.8% | 18.9% | **0.0%** | +242 | 99.2% | 0.44h (26.6m) | 9.8s |
| `motion-0.50` | 923 / 1.438 | 64.2% | 30.8% | 22.2% | +89 | 98.9% | 0.42h (25.1m) | 9.9s |
| `off` | 1.438 / 1.438 | 100.0% | **0.0%** | **0.0%** | +604 | 79.6% | 0.51h (30.4m) | 0.2s |

#### 2. Mẫu 3 Phút WEB-DL (`sample_3min.mkv`, 4.360 frames, 3.328 frame có chuyển động thật $\tau \ge 0.15$px = 76.3%):
| Chính sách | Kept Frames | Tỷ lệ giữ | Đóng băng clip | Frame thêm | Độ chính xác chuyển động thật | Est. Thời gian (h) | CPU (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | 2.186 / 4.360 | 50.1% | 43.6% | 0 | - | 0.64h (38.1m) | 14.1s |
| `sensitive-a` | 2.713 / 4.360 | 62.2% | 32.7% | +527 | 69.1% | 0.73h (43.6m) | 12.3s |
| `sensitive-b` | 2.903 / 4.360 | 66.6% | 28.9% | +717 | 68.3% | 0.76h (45.6m) | 11.3s |
| `motion-0.15` | 3.637 / 4.360 | 83.4% | **0.0%** | +1.451 | 100.0% | 0.89h (53.2m) | 25.9s |
| `motion-0.30` | 2.754 / 4.360 | 63.2% | 27.7% | +568 | 93.3% | 0.73h (44.0m) | 26.7s |
| `motion-0.50` | 2.252 / 4.360 | 51.7% | 41.7% | +66 | 93.9% | 0.65h (38.8m) | 27.0s |
| `off` | 4.360 / 4.360 | 100.0% | **0.0%** | +2.174 | 66.7% | 1.01h (60.6m) | 0.2s |

#### 3. Toàn Tập BD Remux Đầy Đủ (`S01E02`, 34.069 frames, ~23.7 phút):
| Chính sách | Kept Frames | Tỷ lệ giữ | Ước lượng 1 tập (h) | Chênh lệch so với default | CPU Preflight Scan |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | 22.135 / 34.069 | 64.97% | 4.08h (4h 05m) | Baseline (0) | 101.1s (~1.7m) |
| `sensitive-a` | 23.271 / 34.069 | 68.31% | 4.27h (4h 16m) | +1.136 frames (+5.1%, +11m) | 97.9s (~1.6m) |
| `sensitive-b` | 23.568 / 34.069 | 69.18% | 4.32h (4h 19m) | +1.433 frames (+6.5%, +14m) | 96.3s (~1.6m) |
| `motion-0.30` | 26.613 / 34.069 | 78.12% | 4.85h (4h 51m) | +4.478 frames (+20.2%, +46m) | 209.4s (~3.5m) |
| `off` | 34.069 / 34.069 | 100.0% | 6.14h (6h 08m) | +11.934 frames (+53.9%, +123m) | 1.5s |

### D. Nhận Định Kỹ Thuật & Đề Xuất Ứng Viên Pareto
* **Giữ nguyên mặc định:** `default` tiếp tục giữ vai trò mặc định toàn hệ thống để bảo đảm tốc độ và sức chứa 2 tập/đêm cho người dùng phổ thông.
* **Đề xuất Ứng viên 1 (`motion-0.30`):** Ứng viên cao cấp xử lý triệt để hiện tượng giật camera pan/zoom. Xóa sạch 100% hiện tượng đóng băng tại cửa sổ 44–46s; 93.3% - 99.2% số frame giữ thêm là chuyển động thật. Phù hợp cấu hình qua `shows.yaml` cho các bộ phim anime có nhiều cảnh lia máy nghệ thuật.
* **Đề xuất Ứng viên 2 (`sensitive-b`):** Ứng viên hiệu năng/chi phí thấp. Chỉ cần chỉnh tham số `mpdecimate` mà không cần thêm bước tính toán OpenCV; giảm hiện tượng đứng hình tại cảnh 44–46s về 0.0% với chi phí tăng chỉ ~14 phút/tập trên BD Remux.

---

## 10. Khai Thác Tri-Lingual Corpus Season 1 Tensei Shitara Ken Deshita & Dịch Thuật Season 2

### A. Khai Thác Dữ Liệu Phụ Đề Chuẩn B-Global (Season 1)
* **Quy mô tập mẫu:** 11 tập Season 1 (`S01E02` đến `S01E12`, BD Remux 1080p).
* **Trích xuất & Lưu trữ đa ngữ:** Đã trích xuất và bảo toàn 100% ba luồng phụ đề nguyên bản (Tiếng Việt B-Global ASS, Tiếng Nhật SDH ASS, Tiếng Anh Kaleido-subs ASS) vào thư mục `Subtitles_Archive/Season_01/` và `archive_subtitles/Tensei_Shitara_Ken_Deshita/Season_01/` (tổng 33 file phụ đề).
* **Xây dựng từ điển Schema v2 (`glossary.tenken.yaml`):**
  - Khai thác 11 nhân vật cốt lõi (Fran, Sư phụ, Urushi, Jean Doubee, Bernard, Andy, Amanda, Nell, Klimt, Garrus, Donadrond).
  - 253 thuật ngữ ma pháp, kỹ năng, địa danh, tước hiệu và bang hội chuẩn phòng thu.
  - 6 style rules chuyên sâu định hình tính cách nhân vật: Fran ít nói/câu ngắn, Sư phụ bảo bọc/chu đáo, sói Urushi (tuyệt đối không dịch thành Jet), Jean Doubee kịch tính/quý tộc.
  - Thiết lập symlink đa hình thức: `glossary.tenken.yaml`, `glossary.tenseishitarakendeshita.yaml`, `glossary.reincarnatedasasword.yaml`.

### B. Khắc Phục Lỗi Cache Bẩn Dry-Run Trong `subtrans/cache.py`
* **Vấn đề phát hiện:** Chế độ thử nghiệm `--dry-run` trước đây vô tình ghi các bản ghi giả lập `[DRY-RUN VI] English text...` vào cơ sở dữ liệu SQLite `translations.db`. Khi người dùng thực thi dịch thật, pipeline kích hoạt Cache Hit (47.2%) và trả về chuỗi tiếng Anh giả lập thay vì gọi AI dịch tiếng Việt.
* **Giải pháp:** Bổ sung chốt chặn bắt buộc tại `get_cue` và `set_cue` trong `subtrans/cache.py`: Tuyệt đối từ chối lưu và đọc các chuỗi chứa tiền tố `[DRY-RUN]`. Dọn sạch toàn bộ 622 bản ghi rác trong cơ sở dữ liệu.

### C. Dịch Thuật Season 2 (S02E01 & S02E02)
* **Đầu vào:** Phụ đề tiếng Nhật ABEMA WebRip (`.ja.srt`) kết hợp phụ đề tiếng Anh đối chiếu SubsPlease (`.en.ass`) với bù độ lệch thời gian `--en-offset-ms -846` (E01) và `-844` (E02).
* **Số liệu thực thi:**
  - **Tập 1 (S02E01, 439 thoại):** Hoàn thành trong **73.1 giây** (5 requests Gemini Flash-Lite). Tỷ lệ dịch thật 100%, 0 lỗi format, 0 câu `[Chưa dịch]`.
  - **Tập 2 (S02E02, 451 thoại):** Hoàn thành trong **113.4 giây** (kích hoạt thành công cơ chế Self-Healing Bisect khi phát hiện ký tự sót `魔石`, tự động tách đôi 100 cues -> 50 + 50 cues để dịch lại hoàn hảo).
* **Bàn giao thành phẩm:**
  - Tự động tạo và đặt 2 file `.vi.srt` ngay bên cạnh file video MKV gốc trong `RawSources/Tensei_Shitara_Ken_Deshita/` để trình phát phát hiện tức thì.
  - Lưu trữ đầy đủ bộ phụ đề 3 thứ tiếng vào `Subtitles_Archive/Season_02/`.

---

## 11. Nâng Cấp Toàn Diện Phân Hệ Dịch Thuật Subtrans (P0–P3) & Mở Rộng Bộ Kiểm Thử (155 Unit Tests PASS)

### A. Triển Khai 4 Giai Đoạn Nâng Cấp Kiến Trúc (Roadmap P0 ➔ P3)
1. **P0 — Evaluation Harness & Định Lượng Độc Lập (`subtrans/eval/`):**
   * Xây dựng bộ công cụ tính chỉ số tất định: CPS, CJK leakage rate, Romaji vocative retention, zero-overlap compliance, pronoun matching.
   * Đóng băng `regression_suite.yaml` (6 ca kiểm thử biên kinh điển) và `pronoun_suite.yaml` (13 ca đối thoại vai vế đại từ chuẩn).
   * Runner CLI: `python3 subtrans.py eval` trực quan hóa tiến độ bằng bảng Rich formatting.
   * **Ràng buộc chốt:** 100% sử dụng Cloud API (Gemini Flash-Lite, Flash), tuyệt đối không dùng local model.
2. **P1 — Schema CoT & Linter V1 (`subtrans/linter.py`):**
   * Đưa các trường suy luận ngữ cảnh (`speaker`, `addressee`, `mode`, `en_conflict`, `uncertain`) lên trước chuỗi tiếng Việt `vi` trong schema Gemini output, tạo Chain-of-Thought ngầm giải quyết triệt để xưng hô.
   * Linter V1 kiểm tra 6 tiêu chuẩn phòng thu: CJK leakage, reading speed CPS (ngưỡng an toàn 20.0, trần cứng 25.0), Unicode NFC, Romaji vocatives, cấm calque Hán-Việt thô, và khoảng cách an toàn 40ms giữa 2 câu thoại.
   * Cơ chế tự động sửa lỗi (`auto_fix_cues`): chuẩn hóa NFC, thu gọn khoảng trắng, tự động kéo giãn thời lượng câu thoại vào khoảng trống để hạ CPS.
3. **P2 — Smart Cloud Cascade Model (`subtrans/cascade.py`):**
   * Hàm tính độ khó $S(cue) \in [0.0, 15.0]$ tổng hợp từ cờ `uncertain`, `en_conflict`, lỗi linter, độ dài câu và từ vựng cổ ngữ.
   * Mô hình biên tập 2 tầng 100% Cloud: Draft Pass dùng `gemini-3.5-flash-lite` (tiết kiệm quota), tự động lọc 5–15% cues khó ($S \ge 4.0$) chuyển sang Repair Pass dùng `gemini-3.8-flash` với vai trò biên tập viên trưởng.
4. **P3 — Lightweight Subtitle RAG & Error Bank (`subtrans/rag.py`):**
   * Cơ sở dữ liệu SQLite `.cache/knowledge.db` lưu trữ các đơn vị phụ đề mẫu đối chiếu chuẩn (`gold_units`) và bẫy lỗi kinh điển (`error_bank`).
   * Tự động căn chỉnh và nạp thành công **4.144 câu song ngữ mẫu** từ 11 tập Tenken Season 1 qua `import_subtitles_triplet`.
   * Truy vấn trích xuất tự động vài ví dụ mẫu (few-shot) và quy tắc phòng ngừa rủi ro tương ứng để tiêm vào payload prompt gửi tới Gemini API.

### B. Mở Rộng Toàn Diện Bộ Kiểm Thử (155 Tests PASS)
* **`subtrans/tests/test_rag.py` (13 tests):** Kiểm tra khởi tạo DB, từ chối chuỗi rỗng/khoảng trắng, tự động chuẩn hoá NFC, nạp & truy vấn few-shot, cô lập series (`series="tenken"` vs `futsutsuka`), không crash khi thiếu từ khóa kanji, chống trùng lặp, cô lập cautions theo series, quy tắc chung `series=None`, quy trình bóc tách triplet ASS (lọc tags `{\an8}`, lọc comment `is_comment=True`, lọc overlap < 300ms), xử lý thiếu file nguồn, import chỉ có JA + VI.
* **`subtrans/tests/test_linter.py` (10 tests):** Kiểm tra bắt placeholder/rỗng (`[Chưa dịch]`, `[DRY-RUN]`), CJK leakage, ngưỡng biên CPS (0ms, 20.0 CPS an toàn, 20.1 CPS cảnh báo, 25.0 CPS cảnh báo, 25.1 CPS lỗi), chuẩn hóa NFC, từ cấm không phân biệt hoa thường, bảo tồn vocative với Glossary và CastSheet, ranh giới khoảng nghỉ zero-overlap (40ms vs 39ms vs overlap âm), tự động kéo giãn thời lượng câu cuối/khoảng trống hẹp, và định dạng báo cáo markdown.
* **`subtrans/tests/test_gemini.py` (10 tests):** Kiểm tra lọc glossary liên quan, dịch thành công, chế độ dry-run không tốn quota, tự động retry & chia đôi (bisect), hướng dẫn xưng hô tên riêng trong prompt, gán metadata suy luận CoT, tiêm payload RAG few-shot & cautions, fallback câu thoại âm thanh rỗng `（ため息）`, phân loại danh mục `uncertain`, cấu hình thinking budget.
* **`subtrans/tests/test_cascade.py` (10 tests):** Kiểm tra tính điểm độ khó baseline, các thành phần trọng số, lọc ứng viên cần sửa, ngưỡng lọc tùy chỉnh, bỏ qua khi toàn bộ câu đạt chuẩn, dry-run repair pass, API repair thành công, dự phòng khi API lỗi, tích hợp glossary trong repair, và phản hồi ID cục bộ.
* **`subtrans/tests/test_eval_runner.py` (6 tests):** Kiểm tra nạp file YAML, tính toàn vẹn của 6 regression cases, tính toàn vẹn của 13 pronoun benchmark cues, so khớp đại từ & tên Romaji, tổng kết rỗng và thực thi runner.
* **Tổng số kiểm thử đạt chuẩn:** **155 passed, 2 skipped** (thời gian chạy: 44.31 giây).
