# DEDUP-ARCHITECTURE.MD — Kiến Trúc Deduplication Engine & Dòng Chảy Quang Học DIS

> **Chuyên đề phân hệ loại bỏ trùng lặp khung hình (Deduplication Preflight) & Tái tạo khung hình 4K.**  
> **Tham chiếu chính:** [AGENTS.md](../AGENTS.md) | **Đo đạc thực nghiệm:** [docs/engineering-log.md](engineering-log.md)

---

## 1. Trách Nhiệm Module & Các File Mã Nguồn Cốt Lõi

| File mã nguồn | Vai trò & Trách nhiệm chính |
| :--- | :--- |
| [`src/dedup.py`](../src/dedup.py) | Engine Deduplication: Quét Preflight mpdecimate, DIS Optical Flow, Phase Correlation, tạo `plan.json`, tỉa frame RAM Disk |
| [`upscale.py`](../upscale.py) | CLI options: tiếp nhận các cờ `--dedup-policy`, `--dis-thresh`, `--drift-delta` và truyền vào pipeline |
| [`shows.yaml`](../shows.yaml) | Cấu hình cấp series: ánh xạ chính sách `dedup_policy` tự động theo từng bộ anime |
| [`tests/test_dedup_policy.py`](../tests/test_dedup_policy.py) | Bộ kiểm thử tự động toàn diện kiểm tra schema, tương thích ngược, resume rejection và DIS scan |

---

## 2. Bản Đồ Chính Sách Deduplication (`--dedup-policy`)

Bảng tra cứu tham số trong [`src/dedup.py:DEDUP_POLICY_PARAMS`](../src/dedup.py):

| Chính sách | Tham số cốt lõi | Đặc điểm & Đối tượng phù hợp |
| :--- | :--- | :--- |
| **`default`** (Mặc định) | `hi=768 : lo=320 : frac=0.33` | Cân bằng Pareto cho toàn hệ thống: giảm 55%–62% thời gian GPU; bảo đảm chuẩn 2 tập/đêm |
| **`sensitive-a`** | `hi=512 : lo=160 : frac=0.15` | Nhạy hơn với chuyển động nhỏ của nhân vật/khuôn mặt |
| **`sensitive-b`** | `hi=384 : lo=128 : frac=0.10` | Khắc phục chuyển động camera pan/zoom với chi phí preflight cực thấp (~96s CPU/tập) |
| **`motion`** | `hi=768 : lo=320 : frac=0.33`, $\tau = 0.30$ | Bổ sung tương quan pha Phase Correlation trên 4 góc phần tư độc lập |
| **`drift` / `drift-0.5`** | `drift_delta = 0.5 – 1.0 px` | Đo độ dịch tích lũy Phase Correlation từ frame giữ gần nhất; bắt lia máy tịnh tiến nhưng **thất bại trên zoom** do triệt tiêu vector |
| **`dis` / `dis-0.3`** | `dis_thresh = 0.30 px` (1080p) | **Dòng chảy quang học DIS 3 tầng**: Bắt các phân đoạn zoom chậm, lia máy pan/tilt và chuyển động mịn |
| **`off`** | Giữ 100% tất cả khung hình | Tắt hoàn toàn deduplication, chạy AI SR trên toàn bộ frame nguồn |

---

## 3. Khắc Phục Lệch Khung Hình BD Remux GOP Dài (Dynamic GOP Chunk Alignment)

### A. Vấn Đề Lưới 192 Frames Tĩnh Trước Đây
Trước đây hệ thống áp đặt giả định lưới 192 frames tĩnh ($8\text{s} \times 24\text{ fps}$). Tuy nhiên, các bản BD Remux gốc có ranh giới Keyframe GOP biến thiên (ví dụ: `[250, 250, 219...]`), dẫn đến việc cắt ngang GOP khiến FFmpeg phải re-encode hoặc lệch số frame giữa các chunk.

### B. Cơ Chế Giải Quyết Đã Chốt
1. **Đưa Phân Đoạn Lên Trước:** [`src/pipeline.py`](../src/pipeline.py) thực thi `split_video_segments` (Stream Copy nguyên bản) **TRƯỚC** khi quét preflight.
2. **Đo Frame Thực Tế Từng Segment:** Hàm `get_segments_frame_counts` sử dụng FFprobe đo chính xác độ dài thực tế của từng segment theo ranh giới GOP nguồn.
3. **Ép Giữ Frame Đầu Mỗi Chunk:** Hàm `scan_dedup_plan` tự động thêm `chunk_start_indices` vào danh sách bắt buộc giữ (`keep`), bảo đảm mỗi segment luôn có khung hình khởi đầu hợp lệ.
4. **Tự Động Hủy Bỏ Plan Cũ:** Nếu phát hiện `plan.json` có sẵn không khớp với `chunk_frame_counts` thực tế, hệ thống tự động vô hiệu hóa cache và quét lại từ đầu.
5. **Kiểm Định Khung Hình Thành Phẩm:** `src/mux_audit.py` đối soát `video_frame_count` so với video gốc, đánh `FAIL` ngay lập tức nếu thiếu dù chỉ 1 khung hình.

---

## 4. Dòng Chảy Quang Học DIS (Dense Inverse Search Optical Flow)

### A. Cơ Chế Toán Học 3 Tầng Nhận Diện Chuyển Động
Được cài đặt tại [`src/dedup.py:detect_dis_policy_actions`](../src/dedup.py), thuật toán đọc luồng grayscale $480 \times 270$ qua FFmpeg pipe không chạm đĩa:

1. **Lia máy (Pan / Tilt Translation):**
   $$dx = \text{median}(u), \quad dy = \text{median}(v) \implies \text{mag}_{pan} = \sqrt{dx^2 + dy^2}$$
2. **Thu phóng Xuyên tâm (Radial Divergence Zoom In/Out):**
   - Tọa độ từ tâm quang học: $r_x = x - W/2, \; r_y = y - H/2, \; r = \sqrt{r_x^2 + r_y^2}$.
   - Thành phần vận tốc xuyên tâm: $v_{rad} = \frac{r_x u + r_y v}{r}$.
   - Tốc độ dãn nở xuyên tâm: $k_{radial} = \text{median}\left(\frac{v_{rad}}{r}\right)$.
   - Biên độ dịch mép ảnh: $zoom_{edge} = |k_{radial}| \times r_{max}$.
3. **Chuyển động Cục bộ / Phân tán (Local / Diffuse Motion):**
   - Phân vị 90 độ lớn trường dòng chảy: $P_{90} = \text{percentile}(\|\vec{V}\|, 90)$.
   - Diện tích chuyển động: $\text{Area}_{moving} = \% \text{ pixels có } \|\vec{V}\| \ge \tau$.

**Điều kiện Kích hoạt:**
$$\text{Score} = \max(\text{mag}_{pan}, \; zoom_{edge}, \; 0.7 \times P_{90})$$
Nếu $\text{Score} \ge \tau$ (mặc định 0.30 px 1080p) hoặc $\text{Area}_{moving} \ge 5.0\% \implies$ **Giữ lại khung hình và cập nhật frame tham chiếu**.

### B. Cổng Sàng Lọc Candidate Drop Gating
* DIS **chỉ tính toán** trên các khung hình mà `mpdecimate` muốn bỏ (`drop`).
* Các khung hình `mpdecimate` đã giữ (`keep`) được chấp nhận ngay lập tức, bỏ qua tính toán dòng chảy.
* Nhờ đó, tốc độ đo đạc đạt **$\approx 430$ fps trên CPU Apple Silicon M3 Pro**, chỉ thêm ~1.5–2 phút CPU cho toàn bộ tập anime 24 phút.

---

## 5. Tái Tạo Khung Hình 4K (Concat Demuxer Re-expansion)

* Sau khi GPU upscale các khung hình độc lập, hệ thống không dùng hardlinks trên ổ cứng (tiết kiệm 100% dung lượng SSD).
* Thay vào đó, FFmpeg Concat Demuxer sử dụng danh sách `runs` từ `plan.json`:
  ```text
  file '/Volumes/RAMDisk/frame_0001.png'
  duration 0.0834167
  file '/Volumes/RAMDisk/frame_0003.png'
  duration 0.0417083
  ```
* Kết hợp với bộ lọc `-vf fps=24000/1001` khi nạp vào Apple Media Engine, video 4K được tái tạo chính xác từng millisecond và khớp hoàn toàn số khung hình của video nguồn.

---

## 6. Chốt Chặn Resume Cache Trong `plan.json`

Hàm [`src/dedup.py:load_or_create_plan`](../src/dedup.py) kiểm định nghiêm ngặt trước khi tái sử dụng cache:
1. Từ chối resume nếu `dedup_policy` trong cache khác với chính sách yêu cầu (`ValueError: Từ chối resume`).
2. Từ chối resume nếu chính sách `drift` có `drift_delta` trong cache khác với giá trị yêu cầu.
3. Từ chối resume nếu chính sách `dis` có `dis_thresh` trong cache khác với giá trị yêu cầu.
4. Từ chối resume nếu số lượng và phân bố `chunk_frame_counts` không khớp với các segment thực tế.
