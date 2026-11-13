# AUDIT-AND-SAFETY.MD — Kiểm Định Toàn Vẹn, Output Sanity Guard & Vận Hành An Toàn

> **Chuyên đề phân hệ kiểm định chất lượng xuất xưởng, chốt chặn an toàn và tự động dọn dẹp.**  
> **Tham chiếu chính:** [AGENTS.md](../AGENTS.md) | **Đo đạc thực nghiệm:** [docs/engineering-log.md](engineering-log.md)

---

## 1. Trách Nhiệm Module & Các File Mã Nguồn Cốt Lõi

| File mã nguồn | Vai trò & Trách nhiệm chính |
| :--- | :--- |
| [`src/output_sanity.py`](../src/output_sanity.py) | Output Sanity Guard: Kiểm định độ sáng $|\Delta Y| \le 6/255$ và tương quan Pearson $r \ge 0.98$ (hỗ trợ cả file ảnh lẫn mảng GPU `270x480` không cần suy luận lặp) |
| [`src/mux_audit.py`](../src/mux_audit.py) | Parallel Stream Copy Integrity Audit: Đối soát đa luồng frame count, MD5 toàn bộ luồng audio trong 1 lượt `-f streamhash`, phụ đề, font chữ và interleaving |
| [`src/cleanup.py`](../src/cleanup.py) | Auto Cleanup: Dọn sạch `.temp_<basename>` sau khi Mux Audit `PASS` và gom báo cáo kỹ thuật vào thư mục con `audit/` |
| [`src/preflight.py`](../src/preflight.py) | Preflight Checks: Khóa luồng trên P-Cores (`configure_alder_lake_pcore_qos`), chống sleep Win32 `SetThreadExecutionState`, kiểm tra nguồn sạc AC & đĩa trống |
| [`src/config.py`](../src/config.py) | Khóa độc quyền GPU theo PID sống (`.gpu_active.lock`) và tự động thu hồi qua `atexit` |
| [`src/night_report.py`](../src/night_report.py) | Night Telemetry Reporter: Tổng hợp số đo ca đêm, tỷ lệ dedup và báo cáo kiểm định |

---

## 2. Output Sanity Guard ([`src/output_sanity.py`](../src/output_sanity.py))

### A. Mục Đích Bảo Vệ
Ngăn chặn triệt để rủi ro mô hình xuất video đen, trắng xóa hoặc sai lệch cấu trúc/màu sắc nghiêm trọng trước khi tốn hàng giờ mã hóa toàn tập.

### B. Cơ Chế Kiểm Định Tức Thì (0 Lượt Suy Luận Phụ Ở Zero-Disk)
Các hàm `check_frame_sanity` (trên đĩa) và `check_frame_sanity_arrays` (trực tiếp trong RAM) tự động kích hoạt:
1. **Thời điểm kiểm tra:** Ngay tại **Chunk đầu tiên (`c_idx == 0`)**, và lặp lại định kỳ sau mỗi **10 chunks** tiếp theo.
2. **Tối ưu 0-Reinference ở chế độ `--zero-disk`:** Ngay trong Batch 0 trên GPU, `DoubleBufferedPipeUpscaler` dùng `F.avg_pool2d` thu nhỏ trực tiếp khung hình 4K đầu tiên về `480x270` và rút bất đồng bộ về CPU cùng luồng `stream_d2h`, giúp kiểm tra Sanity hoàn tất trước cả khi chờ NVENC đóng gói xong mà không tốn thêm lượt gọi TensorRT nào.
3. **Ngưỡng chốt chặn cứng:**
   - Độ lệch sáng trung bình tuyệt đối: $|\Delta Y| \le 6.0 / 255$ (~2.35%).
   - Hệ số tương quan cấu trúc Pearson: $r(Y_{4K}, Y_{1080p}) \ge 0.98$.
4. **Hành vi khi vi phạm:**
   - Huỷ ngay tiến trình mã hóa phân đoạn hiện tại, ghi biên bản lỗi qua `write_failed_sanity_audit` và ném ngoại lệ `OutputSanityError` để dừng khẩn cấp toàn pipeline (nghiêm cấm Mux thành phẩm lỗi).

---

## 3. Kiểm Toán Xuất Xưởng Đa Luồng Mux Audit ([`src/mux_audit.py`](../src/mux_audit.py))

Sau khi ghép nối container 4K `.mkv` bằng `mux_segments_directly`, hàm `verify_mux_integrity` thực thi kiểm toán song song bằng `ThreadPoolExecutor`:
1. **Khớp Tuyệt Đối Số Khung Hình (`nb_frames`):**
   - Tái sử dụng metadata nguồn đã probe để tránh quét lặp và đối soát số lượng khung hình của file 4K thành phẩm. Lệch dù chỉ $\pm 1$ frame lập tức đánh `FAIL`.
2. **Đối Soát MD5 Toàn Bộ Luồng Audio Trong 1 Lượt Đọc (`get_all_audio_streams_md5`):**
   - Sử dụng muxer `-map 0:a -c copy -f streamhash -hash md5 -` của FFmpeg để băm đồng thời tất cả các track âm thanh (Nhật, Anh, bình luận...) chỉ trong **1 lượt đọc tuần tự container duy nhất**.
3. **Kiểm Tra Đan Xen Packet Đa Cửa Sổ Song Song (`check_window_interleaving`):**
   - Quét song song 5 cửa sổ thời gian trải dọc tập phim để xác nhận cờ `-max_interleave_delta 0` hoạt động chuẩn xác, không xảy ra hiện tượng gom cục audio gây câm tiếng cục bộ.
4. **Bảo Toàn Phụ Đề, Chapters & Font Chữ Đính Kèm:**
   - Đối soát khớp 100% số lượng luồng phụ đề (ASS/SRT), Chapters và toàn bộ font chữ nhúng (TTF/OTF). Chỉ khi `status == "PASS"` mới cho phép bước dọn dẹp tự động chạy.

---

## 4. Tự Động Dọn Dẹp & Thu Hồi Ổ Cứng SSD ([`src/cleanup.py`](../src/cleanup.py))

1. **Xóa Sạch Thư Mục Tạm `.temp_<basename>` Khi Mux Audit `PASS`:**
   - Thu hồi toàn bộ dung lượng các phân đoạn `raw_segs/` và `upscaled_segs/` (trừ khi bật `--keep-temp`).
2. **Gom Gọn Thư Mục Xuất Xưởng (`organize_output_artifacts`):**
   - Tự động chuyển `.audit.json`, `.audit.md`, `.cast.yaml` vào thư mục con `audit/`, giữ thư mục đích chỉ chứa file xem phim `.mkv` và `.vi.srt`.
   - Hỗ trợ quét dọn độc lập qua cờ `--clean-temp` / `--cleanup` (có chốt bảo vệ không xóa file khi đi kèm `--dry-run`).

---

## 5. Cổng Backpressure & Chống Lỗi Tràn Đĩa ENOSPC

Trong [`src/pipeline.py`](../src/pipeline.py):
* **Sàn An Toàn Kích Thước PNG (Chế độ Standard):** `DEFAULT_PNG_UNIT_SIZE_MB = 16.0 MB`, ước tính nhu cầu:
  $$\text{need} = n_{\text{unique}} \times \max(\text{recent}, \; P_{95}, \; 16.0) \times 1.2$$
* **Chế độ Zero-Disk:** Không sinh PNG trung gian trên đĩa nên chỉ yêu cầu biên an toàn nhỏ (`max(256 MB, n_unique * 0.5 MB)`) cho phân đoạn MP4 nén.
* **Tự phục hồi ENOSPC:** Khi thiếu không gian đệm, tự động đồng bộ chờ `EncodeWorker` giải phóng chunk trước, hủy tạm prefetch PNG nếu cần và thử lại tối đa **2 lần**; nếu vẫn cạn đĩa thì ném `FailedResumableError` để bảo toàn các `up_*.mp4` đã xong và tự động thử lại ở cuối hàng đợi `--queue`.

---

## 6. Độc Quyền GPU Theo PID, Khóa P-Core & Chống Sleep Máy

1. **Khóa Độc Quyền GPU Kiểm Chứng PID (`.gpu_active.lock`):**
   - `acquire_gpu_lock()` ghi PID hiện tại và kiểm tra trạng thái sống thực tế của tiến trình giữ khóa (tự động thu hồi khóa mồ côi nếu phiên trước bị mất điện), ngăn chặn xung đột VRAM giữa Video Upscale và Whisper STT.
2. **Khóa Nhân Hiệu Năng Cao Alder Lake P-Core (`configure_alder_lake_pcore_qos`):**
   - Tắt cờ `PROCESS_POWER_THROTTLING_EXECUTION_SPEED` (EcoQoS) và đặt `ABOVE_NORMAL_PRIORITY_CLASS` trên Windows 11 để Thread Director không đẩy luồng điều phối CUDA xuống nhân tiết kiệm điện E-Core khi cửa sổ Terminal thu nhỏ chạy qua đêm.
3. **Chống Sleep Máy An Toàn (`AntiSleepContext`):**
   - Gọi Win32 API native `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)` và khôi phục nguyên trạng qua Context Manager + `atexit` kể cả khi bấm `Ctrl+C`.
