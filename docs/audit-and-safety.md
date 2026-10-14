# AUDIT-AND-SAFETY.MD — Kiểm Định Toàn Vẹn, Output Sanity Guard & Vận Hành An Toàn

> **Chuyên đề phân hệ kiểm định chất lượng xuất xưởng, chốt chặn an toàn và tự động dọn dẹp.**  
> **Tham chiếu chính:** [AGENTS.md](../AGENTS.md) | **Đo đạc thực nghiệm:** [docs/engineering-log.md](engineering-log.md)

---

## 1. Trách Nhiệm Module & Các File Mã Nguồn Cốt Lõi

| File mã nguồn | Vai trò & Trách nhiệm chính |
| :--- | :--- |
| [`src/output_sanity.py`](../src/output_sanity.py) | Output Sanity Guard: Chốt chặn kiểm định độ sáng $|\Delta Y| \le 6/255$ và tương quan $\ge 0.98$, chống video đen/mờ |
| [`src/mux_audit.py`](../src/mux_audit.py) | Stream Copy Integrity Audit: Đối soát bit-exact audio, sub, font đính kèm và tổng số khung hình video |
| [`src/cleanup.py`](../src/cleanup.py) | Auto Cleanup: Dọn sạch `.temp_<basename>` (thu hồi ~6.6GB/tập) và gom file báo cáo vào thư mục con `audit/` |
| [`src/preflight.py`](../src/preflight.py) | Preflight Checks: Kiểm tra nguồn sạc AC, dung lượng đĩa trống, Win32 SetThreadExecutionState chống sleep máy, âm báo Windows |
| [`src/night_report.py`](../src/night_report.py) | Night Telemetry Reporter: Tổng hợp số đo ca đêm, phân tích GPU active, tỷ lệ dedup và xuất báo cáo Markdown |

---

## 2. Output Sanity Guard ([`src/output_sanity.py`](../src/output_sanity.py))

### A. Vấn Đề Lịch Sử Cần Triệt Tiêu
Trong các thử nghiệm model trước đây (AnimeJaNai thiếu phép cộng residual nearest $\times 2$), video 4K đầu ra bị đen toàn bộ (luma chỉ đạt 0.04–0.35) nhưng pipeline vẫn chạy hết và xuất xưởng file hỏng.

### B. Tiêu Chuẩn Kiểm Định Bắt Buộc
Hàm `verify_output_sanity` được kích hoạt tự động:
1. **Thời điểm kiểm tra:** Ngay sau khi hoàn tất **Chunk 1**, và kiểm tra định kỳ sau mỗi **10 chunks** tiếp theo.
2. **Quy trình đo đạc:**
   - Trích xuất 1 frame 4K đã upscale và 1 frame 1080p nguồn tương ứng.
   - Downscale frame 4K về 1080p bằng bicubic và chuyển sang không gian màu xám (Luma Y).
   - Đo độ lệch độ sáng tuyệt đối trung bình: $|\Delta Y| = |\text{mean}(Y_{4K}) - \text{mean}(Y_{1080p})|$.
   - Đo hệ số tương quan Pearson: $r(Y_{4K}, Y_{1080p})$.
3. **Chốt chặn an toàn:**
   - $|\Delta Y| \le 6.0 / 255$ (độ lệch sáng không quá ~2.35%).
   - Hệ số tương quan $r \ge 0.98$.
4. **Hành vi khi vi phạm:**
   - Dừng khẩn cấp toàn bộ pipeline ngay lập tức (`raise RuntimeError`).
   - Ghi nhận `audit.json: {"sanity": "FAIL (OUTPUT_SANITY)"}`.
   - **Nghiêm cấm bước Mux và nghiêm cấm chuyển model**.

---

## 3. Cổng Kiểm Toán Xuất Xưởng Mux Audit ([`src/mux_audit.py`](../src/mux_audit.py))

Sau khi mux xong file video 4K thành phẩm (`.mkv`), hàm `verify_mux_integrity` quét toàn bộ container:
1. **Khớp Số Lượng Khung Hình Video (`video_frame_count`):**
   - Đếm số lượng frame thực tế của video 4K thành phẩm qua FFprobe.
   - Bắt buộc phải **bằng chính xác 100%** số lượng khung hình của video nguồn gốc.
   - Bất kỳ sự lệch frame nào (kể cả $\pm 1$ frame) đều bị đánh rớt `FAIL` ngay lập tức.
2. **Toàn Vẹn Luồng Âm Thanh (Audio Streams):**
   - Đối soát số lượng track audio (tiếng Nhật 2.0 FLAC, thuyết minh, bình luận).
   - Xác thực mã băm MD5 độc lập từng track audio để đảm bảo không bị câm tiếng do lỗi interleaving.
3. **Toàn Vẹn Phụ Đề & Font Chữ Đính Kèm (Subtitles & Fonts):**
   - Kiểm tra số lượng track phụ đề gốc (ASS/SRT).
   - Đếm số lượng font chữ nhúng (attachments TTF/OTF), đối soát khớp 100% với file nguồn (ví dụ: 15 fonts trên bản BD).
4. **Ghi Nhận Kết Quả:**
   - Xuất file `<basename>.audit.json` và `<basename>.audit.md`. Chỉ khi `status == "PASS"`, file thành phẩm mới được bàn giao.

---

## 4. Tự Động Dọn Dẹp & Thu Hồi Ổ Cứng SSD ([`src/cleanup.py`](../src/cleanup.py))

Kích hoạt tự động ngay sau khi Mux Audit xác nhận `PASS`:
1. **Xóa Sạch Thư Mục Tạm `.temp_<basename>`:**
   - Thu hồi sạch sẽ **~6.6 GB/tập** dung lượng SSD (các file HEVC segment `up_*.mp4`, concat demuxer lists).
   - Quét sạch cả symlinks, thư mục con thực tế và các thư mục rỗng.
2. **Gom Gọn Thư Mục Xuất Xưởng:**
   - Tự động di chuyển các file báo cáo kỹ thuật (`.audit.json`, `.audit.md`, `.cast.yaml`) vào thư mục con `audit/`.
   - Giữ thư mục xuất xưởng hoàn toàn gọn gàng, chỉ chứa duy nhất video `.mkv` và file phụ đề `.vi.srt`.
3. **Cờ Điều Khiển:**
   - `--keep-temp`: Giữ lại thư mục tạm và checkpoint khi người dùng cần debug.
   - `--clean-temp` / `--cleanup`: Quét và dọn sạch toàn bộ thư mục `.temp_*` cũ còn sót lại trong thư mục xuất xưởng rồi thoát ngay.

---

## 5. Cổng Backpressure & Chống Lỗi Tràn Đĩa ENOSPC

Trong [`src/pipeline.py`](../src/pipeline.py):
* **Sàn An Toàn Kích Thước PNG:** `DEFAULT_PNG_UNIT_SIZE_MB = 16.0 MB` (đo đạc thực tế frame 4K phức tạp đạt tối đa 13.77 MB).
* **Công thức Ước Tính Dung Lượng Cần Thiết:**
  $$\text{need} = n_{unique} \times \max(\text{recent}, \; P_{95}, \; 16.0) \times 1.2$$
* **Kiểm Tra Giữa Chunk:** Trước khi trích xuất khung hình mới lên bộ đệm, hệ thống kiểm tra dung lượng còn trống. Nếu không đủ:
  - Tạm dừng và đợi chunk trước đó hoàn tất encode HEVC để dọn dẹp frame giải phóng dung lượng đệm.
  - Cơ chế retry tối đa **2 lần**; nếu vẫn thiếu tài nguyên, đánh dấu `failed-resumable` và chuyển sang tập tiếp theo, sau đó tự thử lại ở cuối hàng đợi.
  - Thoát mã non-zero và in rõ bảng báo cáo chưa hoàn tất.

---

## 6. Vòng Đời Tiến Trình & Quản Trị Tài Nguyên

1. **Độc Quyền GPU NVIDIA:**
   - Chỉ duy nhất 1 tiến trình infer GPU NVIDIA được phép hoạt động tại một thời điểm.
   - Giải phóng bộ nhớ VRAM sau khi hoàn tất bước AI SR của video, cho phép phân hệ dịch phụ đề (subtrans) có thể sử dụng tài nguyên mà không gây xung đột.
2. **Chống Sleep Máy Tuyệt Đối (`AntiSleepContext`):**
   - [`src/preflight.py`](../src/preflight.py) gọi Win32 API native `SetThreadExecutionState` trong Context Manager.
   - Tự động hoàn trả trạng thái ban đầu của hệ thống kể cả khi tiến trình bị ngắt bởi `SIGINT` (`Ctrl+C`).
