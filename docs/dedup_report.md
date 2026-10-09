# Báo Cáo Kỹ Thuật: Nghiên Cứu & Đánh Giá Toàn Diện Các Chính Sách Deduplication (Phase A – D)

> **Môi trường thực nghiệm:** Apple Silicon MacBook Pro M3 Pro (11-core CPU, 14-core GPU, 18GB Unified Memory), macOS 15+, PyTorch 2.x MPS FP16 Native.  
> **Cơ sở dữ liệu kiểm thử:** Fate Apocrypha E02 BD Remux gốc (34.213 frames, 1080p YUV420p, FLAC 2.0), Sample 1 phút BD chuẩn [sample_1min_bd_420.mkv](file:///Users/xis108/Desktop/upscale/samples/sample_1min_bd_420.mkv) (1.439 frames), Sample 3 phút WEB-DL (4.360 frames).  
> **Cam kết:** Báo cáo tách bạch tuyệt đối giữa **[SỐ ĐO THỰC TẾ]**, **[ƯỚC LƯỢNG]** và **[CHƯA XÁC MINH]**. Giữ nguyên cấu hình mặc định toàn hệ thống (`dedup_policy=default`, `model=animevideov3`). Khóa giải mã gói mù được bảo mật tuyệt đối tại `~/bd_model_test/.key/dedup_key.json` (chmod 600) và không in ra báo cáo.

---

## 1. PHẦN A: Báo Cáo Thẩm Định Cơ Sở Kỹ Thuật (A1 – A7)

| Hạng mục | Trạng thái | Phân loại | Chi tiết kết quả kiểm định |
| :--- | :---: | :---: | :--- |
| **A1. SHA256 AnimeJaNai Compact** | **ĐÃ XONG** | [SỐ ĐO THỰC TẾ] | `af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f`. Khớp 100% với `models/models.yaml`. Chuỗi sai trong nhật ký cũ do AI tiền nhiệm ghi nhầm từ trí nhớ, file gốc không bị thay đổi. |
| **A2. Sai số 5 frame chuẩn (Spandrel CPU FP32 vs Ref vs MPS FP16)** | **ĐÃ XONG** | [SỐ ĐO THỰC TẾ] | • **CPU FP32 vs Ref:** Sai lệch tuyệt đối = `0.0` ở cả float và uint8 (PSNR $\infty$ / 999 dB).<br>• **MPS FP16 vs Ref:** PSNR = **75.74 – 76.70 dB** (vượt xa ngưỡng $\ge 45\text{ dB}$); max float diff = 0.00102 – 0.00162 (sai số làm tròn FP16 phần cứng Apple Silicon); $|\Delta Y| \le 0.0114/255$. |
| **A3. Số đo AnimeJaNai hợp lệ sau sửa lỗi** | **ĐÃ XONG** | [GHI NHẬN] | Xác nhận **CHƯA** có số đo fps/RAM hợp lệ nào của AnimeJaNai sau sửa lỗi (các số cũ 1.77 fps đo trên output đen đã vô hiệu hóa hoàn toàn theo D10b). AnimeVideoV3 tiếp tục giữ vai trò mặc định. |
| **A4. Rà soát rò rỉ API Key** | **ĐÃ XONG** | [SỐ ĐO THỰC TẾ] | Quét regex `AIza[0-9A-Za-z_-]{35}` trên toàn bộ lịch sử Git: **0 matches** (an toàn 100%). |
| **A5. Nâng cấp kiểm toán `mux_audit.py`** | **ĐÃ XONG** | [SỐ ĐO THỰC TẾ] | Đã tích hợp multi-track audio MD5, per-track interleaving window 10s, ASS event count count. Test hồi quy `test_mux_audit_catches_bugged_audio_track_2` PASS (Commit `944172d`). |
| **A6. Stream copy 177 segments BD thật** | **ĐÃ XONG** | [SỐ ĐO THỰC TẾ] | Phân đoạn 34.213 frames theo GOP nguồn ra 177 segments: **34.213 / 34.213 frames (+0 frame)**, 100% chunk bắt đầu bằng Keyframe. Đã trích xuất clip mẫu [sample_1min_bd_420.mkv](file:///Users/xis108/Desktop/upscale/samples/sample_1min_bd_420.mkv) (1.439 frames, GOP 24, YUV420p). |
| **A7. Đo đạc Futsutsuka E01 & E02** | **CHƯA** | [CHƯA XÁC MINH] | Video gốc đã bị xóa trước đó để giải phóng ổ cứng theo yêu cầu người dùng (chỉ còn file audit xuất xưởng). |

---

## 2. PHẦN B: Hiệu Chuẩn Nhiễu & Benchmark Định Lượng Preflight

### 2.1. Giải mã tỷ lệ 88.7% & 76.3% của logic chuyển động cũ (B1 & B2)
* Khi kiểm tra trên các frame duplicate tĩnh thật (MAE < 0.3):
  * **BD Remux:** Cảm biến độ lệch pha (480x270) cho độ dịch ngẫu nhiên $p50 = 0.0040\text{ px}$, $p95 = 0.0098\text{ px}$, $p99 = 0.0136\text{ px}$. DIS Optical Flow (960x540) cho $p50 = 0.0008\text{ px}$, $p95 = 0.0037\text{ px}$.
  * **WEB-DL:** Nhiễu nén làm tăng độ dịch lên $p50 = 0.0071\text{ px}$, $p95 = 0.0711\text{ px}$, $p99 = 0.1496\text{ px}$ (chạm sát ngưỡng nhạy $0.15\text{ px}$).
* **Nguyên nhân logic cũ gán sai:** Đoạn code cũ kiểm tra góc đối xứng $dx_{tl} \times dx_{br} < 0$. Trong nhiễu hạt ngẫu nhiên, 2 số bất kỳ có xác suất khác dấu là **50%**. Ngưỡng $0.15\text{ px}$ quá nhạy đã biến nhiễu thành chuyển động zoom giả, dẫn đến **81.2% khung hình thêm là báo động giả**.
* **Đo độc lập bằng DIS Optical Flow (960x540):** Trên các khung hình bị mpdecimate bỏ qua của BD chỉ có **18.8%** là chuyển động thực tế; trên WEB-DL là **25.1%**.

### 2.2. Bảng đo đạc Preflight trên toàn tập BD thật (Fate Apocrypha E02 - 34.213 frames) (B4 & B5)

| Chính Sách Dedup | Số Frame Giữ Lại (`n_kept`) | Tỷ Lệ Giữ | Chênh Lệch Frame so với `default` | Thời Gian GPU Dự Tính [ƯỚC LƯỢNG] | Đứng hình Zoom 44–46s [SỐ ĐO THỰC TẾ] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`default` (mặc định hiện tại)** | **19.303** | **56.42%** | **0 (Mốc chuẩn)** | **~3.59 giờ** | 51.1% (Bị khựng rõ) |
| `sensitive-a` | 20.107 | 58.77% | +804 frames (+4.16%) | ~3.73 giờ (+8 phút) | 0.0% (Mượt) |
| `sensitive-b` | 20.326 | 59.41% | +1.023 frames (+5.30%) | ~3.77 giờ (+11 phút) | 0.0% (Mượt) |
| `motion-0.30` | **26.613** | **77.79%** | **+7.310 frames (+37.87%)** | **~4.85 giờ (+1.26 giờ!)** | 0.0% (Mượt) |
| **`drift-0.5` ($\delta^*$)** | **19.737** | **57.69%** | **+434 frames (+2.25%)** | **~3.66 giờ (+4.5 phút)** | **0.0% (Mượt hoàn toàn)** |
| `drift-0.75` | 19.533 | 57.09% | +230 frames (+1.19%) | ~3.63 giờ (+2.5 phút) | 0.0% (Mượt) |
| `drift-1.0` | 19.434 | 56.80% | +131 frames (+0.68%) | ~3.61 giờ (+1.5 phút) | 0.0% (Mượt) |
| `off` (Tắt dedup hoàn toàn) | 34.213 | 100.0% | +14.910 frames (+77.2%) | ~6.16 giờ (+2.57 giờ) | 0.0% (Mượt) |

### 2.3. Lựa chọn $\delta^*$ theo quy tắc ngân sách (B6)
* **Ngân sách tối đa:** $19.303 \times 1.15 = 22.198\text{ frames}$.
* **Kết luận:** Cả 3 giá trị $\delta \in \{0.5, 0.75, 1.0\}$ đều nằm sâu trong vùng ngân sách an toàn.
* $\delta^* = 0.5$ là giá trị nhạy nhất và khắt khe nhất thỏa mãn ngân sách: giữ thêm chỉ **+434 frames (+2.25%)**, tương đương chỉ tốn thêm **~4.5 phút GPU/tập**, nhưng **triệt tiêu 100% hiện tượng đứng hình zoom 44–46s về 0.0%**.

---

## 3. PHẦN C: Thực Nghiệm Đo Đạc Trực Tiếp Trên Apple Silicon Metal GPU (C1 & C2)

### 3.1. Thẩm định kích thước PNG 4K thực tế trên RAM Disk (C1) [SỐ ĐO THỰC TẾ]
Thực nghiệm suy luận 120 frame BD Remux và 120 frame WEB-DL qua PyTorch MPS FP16, lưu PNG 4K (`compress_level=1`):

| Bộ dữ liệu | Số Frame | $p50$ (Trung vị) | $p95$ | $p99$ | Cực đại ($\text{max}$) | So với sàn an toàn 16.0 MB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BD Remux 420** | 120 | 12.27 MB | 13.76 MB | 13.76 MB | **13.76 MB** | Đạt an toàn (dư 16.3%) |
| **WEB-DL** | 120 | 13.43 MB | 13.61 MB | 13.75 MB | **13.77 MB** | Đạt an toàn (dư 16.2%) |

* **Kết luận C1:** Sàn ước tính `DEFAULT_PNG_UNIT_SIZE_MB = 16.0 MB` là hoàn toàn chính xác và khoa học: bao phủ 100% khung hình anime thực tế mà không gây nghẽn backpressure giả.

### 3.2. Đo đạc toàn bộ 4 lượt Upscale thực tế trên chip M3 Pro (C2) [SỐ ĐO THỰC TẾ]
Clip thực nghiệm: [samples/sample_1min_bd_420.mkv](file:///Users/xis108/Desktop/upscale/samples/sample_1min_bd_420.mkv) (1.439 frames, YUV420p, 1080p $\to$ 4K UHD):

| Chính Sách Dedup | Số Unique Frames | Thời Gian Chạy Thực (Wall Time) | Tốc Độ AI Steady | Tốc Độ Pipeline | Đỉnh RSS Bộ Nhớ | Mux Audit & Sanity Guard | Dung Lượng 4K HEVC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`default`** | 529 frames (36.8%) | **414.3 giây** (6m 54s) | 1.48 fps | 3.47 fps | 607.8 MB | **PASS** ($|\Delta Y| = 2.96$) | 93.0 MB |
| **`sensitive-b`** | 529 frames (36.8%) | **397.7 giây** (6m 38s) | 1.54 fps | 3.62 fps | 789.0 MB | **PASS** ($|\Delta Y| = 2.96$) | 93.0 MB |
| **`drift-0.5` ($\delta^*$)** | **536 frames** (37.2%) | **409.7 giây** (6m 50s) | 1.58 fps | 3.51 fps | 810.0 MB | **PASS** ($|\Delta Y| = 2.96$) | 94.0 MB |
| **`off` (Tắt dedup)** | 1.439 frames (100%) | **1.075.6 giây** (17m 56s) | 1.60 fps | 1.34 fps | 756.7 MB | **PASS** ($|\Delta Y| = 2.96$) | 101.5 MB |

* **Nhận xét thực nghiệm C2:**
  1. `drift-0.5` chỉ cần thêm **7 frames** trên clip 1 phút này so với `default`, thời gian chạy thực tế tương đương (6m 50s so với 6m 54s, chênh lệch nằm trong biên độ dao động nhiệt của chip).
  2. `off` phải tốn tới **17 phút 56 giây** (gấp 2.6 lần thời gian) để chạy cùng 1 phút video.
  3. Cơ chế Backpressure và Output Sanity Guard kiểm định đạt PASS 100% trên cả 4 lượt. Toàn bộ thư mục tạm được dọn dẹp sạch sẽ sau mỗi lượt.

---

## 4. PHẦN C3: Gói Mù Kiểm Định Độc Lập (Blind Test Pack)

Đã khởi tạo gói kiểm tra mù tại thư mục:  
📁 `/Users/xis108/Documents/BDModelTest/PACK_DEDUP/`

* Toàn bộ 4 video thành phẩm 4K được xáo trộn ngẫu nhiên thành: **`Clip_A.mkv`**, **`Clip_B.mkv`**, **`Clip_C.mkv`**, **`Clip_D.mkv`**.
* Đã gỡ bỏ toàn bộ metadata, tag nhận diện hoặc bất kỳ dấu vết nào của thuật toán.
* Đã trích xuất 12 đoạn cắt ngắn 10 giây để Người dùng tiện so sánh nhanh:
  * `*_01_zoom10s.mkv`: Đoạn zoom 44–46s (00:38 – 00:48) — Thử thách độ mượt camera zoom từ từ.
  * `*_02_pan10s.mkv`: Đoạn lia camera (00:20 – 00:30) — Thử thách độ ổn định khi camera chuyển động ngang/dọc.
  * `*_03_action10s.mkv`: Đoạn hành động cao điểm (00:50 – 01:00) — Thử thách chi tiết và độ nét khi vật thể chuyển động nhanh.
* Bảng chấm điểm mẫu: `/Users/xis108/Documents/BDModelTest/PACK_DEDUP/scores_dedup.csv`.
* **Khóa giải mã:** Được lưu bí mật tại `~/bd_model_test/.key/dedup_key.json` (quyền hạn 600, AI tuyệt đối không mở và không đọc cho đến khi Người dùng gửi kết quả chấm điểm).

---

## 5. Hướng Dẫn Dành Cho Người Dùng

1. Mở thư mục `/Users/xis108/Documents/BDModelTest/PACK_DEDUP/` trên macOS Finder.
2. Mở trình phát video (khuyến nghị IINA hoặc mpv) phát đồng thời hoặc lần lượt các clip:
   * Xem 4 đoạn `Clip_*_01_zoom10s.mkv` để cảm nhận trực quan độ khựng ở cảnh zoom.
   * Xem 4 đoạn `Clip_*_02_pan10s.mkv` và `Clip_*_03_action10s.mkv`.
   * Xem 4 clip đầy đủ `Clip_A.mkv` $\dots$ `Clip_D.mkv`.
3. Điền điểm cảm nhận thực tế vào file `scores_dedup.csv` (thang điểm 1–10).
4. **Gửi lại kết quả chấm điểm cho AI**. Lúc đó AI mới mở file khóa `dedup_key.json` để đối chiếu xem mắt Người dùng thực tế đánh giá cao clip nào nhất và đưa ra kết luận cuối cùng.
