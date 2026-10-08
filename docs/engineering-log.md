# Engineering Log — Nhật Ký Thực Nghiệm & Đo Đạc Hiệu Năng

> **Tài liệu lưu trữ toàn bộ số đo thực nghiệm, benchmark đối đầu, soak test và lịch sử kiểm định của dự án.**
> Tham chiếu từ: [AGENTS.md](file:///Users/xis108/Desktop/upscale/AGENTS.md) và [subtrans/AGENTS.md](file:///Users/xis108/Desktop/upscale/subtrans/AGENTS.md).

---

## 1. Dữ Liệu Thực Nghiệm Cốt Lõi (Baseline Ground Truth)

* **Phần cứng đo đạc:** Apple Silicon MacBook Pro M3 Pro (11-core CPU, 14-core GPU, 18GB Unified Memory, macOS Sonoma/Sequoia).
* **Tốc độ AI NCNN Vulkan:** Duy trì ổn định **1.24 – 1.27 fps** với `realesr-animevideov3` (64nf16nc, ~2.5 TFLOPs/frame).
* **Tốc độ AI PyTorch MPS FP16:** Đạt **1.86 fps** sau warm-up (538.2 ms/frame, nhanh hơn ncnn +48.6%). Baseline duy trì hiện tại: **~1.6 fps** (soak test 6.000 frame đạt 1.683 fps, vận hành toàn tập duy trì 1.61 – 1.68 fps).
* **Thời gian chạy thực tế mẫu 3 phút (4.360 frames):**
  - Tuần tự NCNN: **37m 09s** (23 phân đoạn; AI: 1.764s, Encode: 278s, Extract: 185s, Preflight/Mux: 15s).
  - Pipelined MPS FP16: **23m 59s (1.439,63s)** -> Nhanh hơn bản tuần tự 13m 10s (-35.4%). GPU active: 90.96% (GPU idle chỉ 9.04%).
* **Overhead tuần tự ngoài AI:** ~20.1s / chunk (Decode FFmpeg ~8.5s + Encode VideoToolbox ~11.6s).
* **Kiểm định gối đầu (Pipelined Double/Triple-Buffering):** CPU decode (chunk N+1) và Media Engine encode (chunk N-1) chạy song song với GPU AI infer (chunk N). Mức chiếm dụng RAM Disk tối đa 2.7 GB (< 4 GB). Triệt tiêu 59 phút thời gian chết của GPU trên toàn tập 24 phút.

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

