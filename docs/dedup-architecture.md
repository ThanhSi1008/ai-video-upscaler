# DEDUP-ARCHITECTURE.MD — Kiến Trúc Deduplication Engine, Cyclic Cel Memoization & DIS Flow

> **Chuyên đề phân hệ loại bỏ trùng lặp khung hình (Deduplication Preflight), khử chu kỳ hoạt họa `A->B->A` & tái tạo khung hình 4K.**
>
> **Tham chiếu chính:** [AGENTS.md](../AGENTS.md) | **Đo đạc thực nghiệm:** [docs/engineering-log.md](engineering-log.md)

---

## 1. Trách Nhiệm Module & Các File Mã Nguồn Cốt Lõi

| File mã nguồn | Vai trò & Trách nhiệm chính |
| :--- | :--- |
| [`src/dedup.py`](../src/dedup.py) | Engine Deduplication: 1-Pass fused `mpdecimate+DIS`, Early-Exit DIS Flow, `deduplicate_cyclic_frames` (`A->B->A`), nén `raw_actions_rle`, `run_preflight_worker_cli` |
| [`src/scheduler.py`](../src/scheduler.py) | `BackgroundQueuePreflightManager`: tách sẵn `raw_segs` theo GOP và quét nền `plan.json` khớp `dedup_policy` cho các tập tiếp theo trong `--queue` |
| [`upscale.py`](../upscale.py) | CLI options: tiếp nhận các cờ `--dedup-policy`, `--dis-thresh`, `--drift-delta`, `--no-dedup` và điều phối hàng đợi |
| [`shows.yaml`](../shows.yaml) | Cấu hình cấp series: ánh xạ chính sách `dedup_policy` tự động theo từng bộ anime |
| [`tests/test_dedup_policy.py`](../tests/test_dedup_policy.py) | Bộ kiểm thử tự động kiểm tra schema, resume rejection, DIS scan và tái phân hoạch RLE |

---

## 2. Bản Đồ Chính Sách Deduplication (`--dedup-policy`)

Bảng tra cứu tham số trong [`src/dedup.py:DEDUP_POLICY_PARAMS`](../src/dedup.py):

| Chính sách | Tham số cốt lõi | Đặc điểm & Đối tượng phù hợp |
| :--- | :--- | :--- |
| **`default`** (Mặc định) | `hi=768 : lo=320 : frac=0.33` | Cân bằng Pareto cho toàn hệ thống: giảm 55%–62% khung hình trùng liên tiếp |
| **`sensitive-a`** | `hi=512 : lo=160 : frac=0.15` | Nhạy hơn với chuyển động nhỏ của nhân vật/khuôn mặt |
| **`sensitive-b`** | `hi=384 : lo=128 : frac=0.10` | Giữ mượt chuyển động camera pan/zoom với chi phí CPU preflight cực thấp |
| **`motion`** | `hi=768 : lo=320 : frac=0.33`, $\tau = 0.30$ | Bổ sung tương quan pha Phase Correlation trên 4 góc phần tư độc lập |
| **`drift` / `drift-0.5`** | `drift_delta = 0.5 – 1.0 px` | Đo độ dịch tích lũy Phase Correlation từ frame giữ gần nhất |
| **`dis` / `dis-0.3`** | `dis_thresh = 0.30 px` (1080p) | **Dòng chảy quang học DIS 1-Pass + Early-Exit**: Bắt trọn vẹn cảnh zoom chậm, lia máy pan/tilt và chuyển động mịn |
| **`off`** | Giữ 100% tất cả khung hình | Tắt hoàn toàn deduplication, chạy AI SR trên toàn bộ frame nguồn |

---

## 3. Đồng Bộ GOP Động, Nén `raw_actions_rle` & Preflight Hàng Đợi 0 Giây Chờ

1. **Phân Đoạn Theo Keyframe GOP Nguồn:** [`src/pipeline.py`](../src/pipeline.py) thực thi `split_video_segments` (Stream Copy) và đo chính xác số frame thực tế từng segment qua `get_segments_frame_counts` (`[250, 250, 219...]`).
2. **Ép Giữ Frame Đầu Mỗi Chunk:** `scan_dedup_plan` cưỡng bức frame đầu tiên của mọi chunk = `keep`, bảo đảm mọi phân đoạn luôn có khung hình gốc độc lập.
3. **Tái Phân Hoạch GOP `< 1 ms` Nhờ `raw_actions_rle`:**
   - `scan_dedup_plan` lưu thêm chuỗi nén Run-Length Encoding (`raw_actions_rle`, chỉ ~25 KB/tập) của danh sách `keep`/`drop` thô đối với các chính sách độc lập ranh giới (`default`, `sensitive*`, `motion`, `off`).
   - Khi `load_or_create_plan` gặp `plan.json` có lưới chunk cũ khác với `chunk_frame_counts` GOP thực tế, hệ thống giải mã `raw_actions_rle` và phân hoạch lại sang lưới GOP mới ngay trong RAM chỉ mất **`< 1 ms`** (không cần giải mã lại video bằng FFmpeg; các file `plan.json` thế hệ cũ thiếu `raw_actions_rle` vẫn tự động quét lại an toàn).
4. **Background Queue Preflight Khớp GOP 100%:** Trong hàng đợi `--queue`, `run_preflight_worker_cli` chạy nền ở mức ưu tiên thấp (`BELOW_NORMAL_PRIORITY_CLASS`), tách sẵn `raw_segs` theo GOP và lập `plan.json` đúng `dedup_policy` cho các tập kế tiếp trong lúc GPU đang xử lý tập hiện tại -> chuyển tập với **0 giây chờ preflight**.

---

## 4. Khử Trùng Lặp Chu Kỳ Hoạt Họa `A->B->A` (Cyclic Cel Memoization)

Bộ lọc `mpdecimate` chỉ loại bỏ được các frame trùng **liền kề** (`A, A, A`), nhưng hoàn toàn bỏ lọt các vòng lặp hoạt họa **xen kẽ** rất phổ biến trong anime như nhép miệng thoại (`A -> B -> A -> C -> A`), chớp mắt, tóc/áo bay theo chu kỳ 3–4 bản vẽ hoặc cắt góc máy qua lại trong hội thoại:

1. **Nhận diện hai tầng tốc độ cao (`deduplicate_cyclic_frames`):**
   - **Khớp chính xác bit-exact:** Băm mật mã `BLAKE2b` 128-bit ở tốc độ C native (> 3 GB/s).
   - **Chốt chặn nhiễu lượng tử H.264/HEVC:** Lọc thô trên mẫu lấy cách quãng `24 KB`, sau đó kiểm tra toàn khung hình bằng OpenCV C++ SIMD (`cv2.absdiff`), chỉ chấp nhận tái sử dụng khi sai lệch tối đa toàn ảnh `max_diff <= 2/255` và `MAE <= 0.20/255` (`PSNR >= 57 dB` — bảo vệ tuyệt đối 100% chi tiết mắt/miệng nhỏ nhất).
2. **Tái sử dụng kết quả 4K không suy luận lại:**
   - **Chế độ Zero-Disk ([`src/trt_zero_copy_upscaler.py`](../src/trt_zero_copy_upscaler.py)):** Áp dụng giải thuật tối ưu lý thuyết **Belady's Optimal (`OPT`)** để chỉ lưu trên RAM đúng những khung hình `p010le` chắc chắn còn xuất hiện lại ở các batch sau và giải phóng ngay sau lần tham chiếu cuối cùng.
   - **Chế độ Standard ([`src/cuda_engine.py`](../src/cuda_engine.py)):** Chỉ chạy GPU trên các khung hình phân biệt thực sự và tạo liên kết cứng nguyên tử NTFS (`os.link`) cho các vị trí lặp chu kỳ với 0 byte dung lượng đĩa tăng thêm.

---

## 5. Dòng Chảy Quang Học DIS 1-Pass & Early-Exit Cascade

Được cài đặt tại [`src/dedup.py:_scan_fused_mpdecimate_and_dis`](../src/dedup.py) và `detect_dis_policy_actions`:

1. **Giải mã hợp nhất 1-Pass (`[0:v]split=2`):** Một tiến trình FFmpeg duy nhất vừa xuất log `mpdecimate` qua `stderr` vừa bơm ảnh xám `480x270` qua `stdout` pipe, cắt bỏ hoàn toàn lượt giải mã video lần thứ hai.
2. **Thác Thoát Sớm (Early-Exit Cascade):**
   - Kiểm tra `np.array_equal` bỏ qua trong `0 ms` các khung hình tĩnh giống hệt nhau.
   - Đánh giá tuần tự 3 điều kiện (chỉ tính căn bậc hai / tích vô hướng nặng khi bước nhẹ phía trước chưa vượt ngưỡng $\tau$):
     1. **Lia máy (Pan / Tilt):** $\text{mag}_{pan} = \sqrt{\text{median}(u)^2 + \text{median}(v)^2} \ge \tau \implies$ giữ ngay.
     2. **Thu phóng Xuyên tâm (Radial Zoom):** $zoom_{edge} = |\text{median}(v_{rad}/r)| \times r_{max} \ge \tau \implies$ giữ ngay.
     3. **Chuyển động Cục bộ / Phân tán:** $0.7 \times P_{90}(\|\vec{V}\|) \ge \tau$ hoặc diện tích chuyển động $\ge 5.0\% \implies$ giữ khung hình.

---

## 6. Chốt Chặn Toàn Vẹn & Tái Tạo Khung Hình 4K

* **Chế độ Zero-Disk:** Nhân bản trực tiếp `memoryview` của khung hình `p010le` vào đường ống NVENC theo đúng `repeat_counts` từ `runs` của `plan.json`.
* **Chế độ Standard:** Tái tạo thời lượng chính xác từng khung hình qua FFmpeg Concat Demuxer (`generate_concat_demuxer_script` + `encode_chunk_reexpanded`).
* **Chốt chặn Resume nghiêm ngặt (`load_or_create_plan`):** Tự động từ chối resume (`ValueError`) nếu `dedup_policy`, `drift_delta`, hoặc `dis_thresh` trong `plan.json` khác với cấu hình phiên chạy hiện tại.
