# AGENTS.md — AI Agent Guidelines & Architecture Compass

> **Kim chỉ nam dành cho các AI Agent khi đọc, bảo trì và phát triển codebase này.**
> **Điểm khởi đầu bắt buộc:** [docs/agents/START_HERE.md](docs/agents/START_HERE.md) | [docs/agents/STATE.md](docs/agents/STATE.md) | [docs/agents/PROJECT_MAP.md](docs/agents/PROJECT_MAP.md) | [docs/agents/DECISIONS.md](docs/agents/DECISIONS.md) | [docs/agents/VALIDATION.md](docs/agents/VALIDATION.md)
> **Quy tắc bắt buộc:** Ngắn gọn, chuẩn xác, không suy diễn; cập nhật file này sau mỗi lần tái cấu trúc hoặc thay đổi kiến trúc code; **BẮT BUỘC commit chi tiết sau MỖI yêu cầu** (nêu rõ nguồn yêu cầu, việc đã làm, lỗi đã fix, số liệu thực nghiệm).

---

## 1. Mục Tiêu & Ràng Buộc Cốt Lõi (Core Invariants)

* **Thiết bị vận hành mục tiêu:**
  * **Windows Workstation:** Windows 11 64-bit, Intel Core i5-12600K (10C/16T AVX2), NVIDIA GeForce RTX 3060 (12GB VRAM GDDR6, CUDA & Tensor Cores), 32GB RAM, FFmpeg `hevc_nvenc`.
* **Môi trường bắt buộc:** **100% CỤC BỘ trên máy người dùng (Windows 11)**. Tuyệt đối không đề xuất Cloud GPU, Colab, Kaggle.
* **Tôn chỉ Pareto:** Cân bằng chất lượng anime và thời gian. Không hy sinh nét vẽ lấy tốc độ rẻ tiền; không lãng phí tài nguyên cho cải thiện vô hình.
* **Sức chứa thực tế:**
  * RTX 3060 (PyTorch CUDA FP16 `batch_size=6` ~4.8–5.2 fps suy luận, ~10–12+ fps hiệu dụng với Dedup & `cv2.imwrite`): **~6–8 tập/đêm** (~45–60 phút/tập ~17.000 unique frames).

---

## 2. Lệnh CLI Cơ Bản & Bảng Cờ (Commands & Flags)

### A. Lệnh Vận Hành Thường Dùng (PowerShell / CMD)

```powershell
# Nâng cấp 1 tập video đơn lẻ:
python upscale.py path/to/source.mkv -o out/
.\upscale.bat path/to/source.mkv

# Xem trước nhanh 10 giây ở 30% phim:
python upscale.py path/to/source.mkv --preview 10

# Chạy hàng đợi nhiều tập qua đêm (auto-offset và deadline):
python upscale.py --queue ep1.mkv ep2.mkv -o out/ --deadline 07:00 --allow-partial

# Tiếp tục tập bị gián đoạn (tự nhận checkpoint .temp):
python upscale.py path/to/source.mkv -o out/

# Dịch phụ đề riêng lẻ:
python subtrans.py path/to/source.mkv -o out/
.\subtrans.bat path/to/source.mkv

# Kiểm tra bộ nhớ Agent và chạy kiểm thử an toàn offline:
python scripts/agent_context.py --check
python scripts/verify_offline.py
pytest
```

### B. Bảng Cờ CLI Chính (`upscale.py`)
| Cờ CLI | Mặc định | Ý nghĩa & Hành vi |
| :--- | :--- | :--- |
| `-o`, `--output` | `out/<tên>_4K.mkv` | Thư mục hoặc file đích xuất video 4K |
| `--queue` | `None` | Danh sách file video chạy nối tiếp qua đêm |
| `--engine` | `cuda` | Engine suy luận: `cuda` (NVIDIA CUDA FP16) hoặc `cpu` |
| `--model` | `animevideov3` | Model Super-Resolution đăng ký trong `models/models.yaml` |
| `--batch-size` | `6` (RTX 3060) | Kích thước batch cho AI Inference GPU (Tensor Cores) |
| `--subs-quality` | `fast` | Chất lượng dịch: `fast` (`gemini-3.5-flash-lite`) hoặc `high` (`gemini-3.8-flash`) |
| `--subs-only` | `False` | Chỉ trích xuất và dịch phụ đề, bỏ qua video upscale |
| `--subs-upgrade` | `False` | Nâng cấp phụ đề VI hiện có lên bản dịch Gemini |
| `--auto-align-en` | `True` | Ghim mốc thời gian phụ đề Whisper/JA theo track EN phòng thu |
| `--no-auto-align-en` | `False` | Tắt tính năng tự động ghim mốc thời gian phụ đề |
| `--allow-partial` | `False` | Cho phép dừng tập an toàn ở chunk 8s khi chạm deadline ca đêm |
| `--auto-offset` | Bật trong `--queue` | Tự động đo và bù độ lệch thời gian phụ đề JA - EN |
| `--no-auto-offset` | `False` | Ép tắt tính năng auto-offset trong hàng đợi |
| `--sub-offset-ms` | `0` | Bù offset phụ đề thủ công theo ms (luôn ưu tiên cao nhất) |
| `--no-embed` | `False` | Tắt bước nhúng phụ đề vào container MKV thành phẩm |
| `--keep-temp` | `False` | Giữ lại thư mục tạm `.temp_<basename>` (mặc định xóa khi Mux PASS) |
| `--clean-temp`, `--cleanup` | `False` | Quét dọn toàn bộ thư mục tạm và gom audit/cast rồi thoát |
| `--dedup-policy` | `default` | Chính sách dedup: `default`, `sensitive-a/b`, `motion`, `drift`, `dis`, `off` |
| `--dis-thresh` | `0.30` | Ngưỡng dòng chảy quang học DIS tính bằng pixel 1080p |
| `--dry-run` | `False` | Chạy mô phỏng toàn bộ quy trình không gọi AI hay API |

---

## 3. Bản Đồ Mã Nguồn & Trách Nhiệm Module

> **Chi tiết vị trí từng file, hàm và cấu trúc module:** Xem [`docs/agents/PROJECT_MAP.md`](docs/agents/PROJECT_MAP.md).

- **`upscale.py` / `src/`:** Pipeline phóng to video (CUDA FP16 Native, NVENC 10-bit, Dedup DIS Flow, Rolling Segments).
- **`subtrans.py` / `subtrans/`:** Engine dịch thuật phụ đề (Gemini 3-Key Round-Robin, Whisper STT Fallback, Multi-Speaker CC).
- **`scripts/`:** Tiện ích vận hành (`agent_context.py`, `verify_offline.py`, `batch_translate_all.py`, `set_track_defaults.py`).
- **`models/` & `shows.yaml`:** Registry mô hình Super-Resolution và ánh xạ riêng từng bộ anime.
- **`tests/` & `subtrans/tests/`:** 100% Pytest tự động hóa kiểm thử đơn vị và tích hợp.

---

## 4. Những Bất Biến Kỹ Thuật (Do NOT Break)

1. **Bảo vệ Bộ Nhớ & Ổ Cứng (Storage & RAM Discipline):**
   * Khung hình trung gian của từng chunk 8s phải được xóa cuốn chiếu (Rolling Segments Cleanup) ngay khi chunk được mã hóa xong sang file HEVC `.mp4`.
   * Thư mục SSD chỉ lưu trữ các segment HEVC đã nén (`up_*.mp4`) trong `.temp_<basename>` để phục vụ resume.
   * Tối ưu I/O khung hình qua OpenCV C++ `cv2.imwrite` tận dụng AVX2 của CPU Core i5-12600K.
2. **Bit-exact Stream Copy & Cờ Mux Phòng Thu:**
   * Video 4K thành phẩm **bắt buộc Stream Copy nguyên bản**: Audio đa kênh, Phụ đề mềm (ASS/SRT), Font chữ nhúng (TTF/OTF) và Chapters.
   * Lệnh mux bắt buộc dùng cờ:
     `-map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 -c copy -max_interleave_delta 0`
   * Cờ `-max_interleave_delta 0` là bắt buộc để ngăn lỗi tràn buffer và mất tiếng cục bộ sau vài phút phát lại.
3. **Chuỗi ma trận màu sắc chuẩn xác (BT.709 Two-way Matrix):**
   * Trích xuất: `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc` (YUV Limited -> RGB24 Full).
   * Mã hóa: `scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv` (RGB24 Full -> YUV Limited `p010le`).
   * Metadata gắn cờ: `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`.
4. **Mã hóa phần cứng 10-bit NVENC:**
   * Bắt buộc dùng `hevc_nvenc` với profile `main10`, định dạng pixel `p010le`, preset `p6` (chất lượng cao), multipass `qres`, lookahead `24` và `spatial-aq 1`.
5. **Vòng đời an toàn (Process Lifecycle & Anti-Sleep):**
   * Sử dụng Win32 API native `SetThreadExecutionState` ngăn hệ thống đi vào chế độ ngủ (sleep/suspend) trong suốt quá trình chạy.
   * Tất cả tài nguyên đệm tạm và tiến trình con phải nằm trong Context Manager với `atexit` để thu hồi sạch sẽ kể cả khi bị `Ctrl+C` (SIGINT).
6. **Độc quyền GPU & Điều phối tài nguyên:**
   * Tại một thời điểm chỉ duy nhất 1 tiến trình infer GPU được hoạt động.
   * Pipeline chạy gối đầu 3 tầng: CPU Decode (chunk N+1) và NVENC Encoder (chunk N-1) chạy song song với GPU Infer (chunk N).
   * Tuyệt đối không nạp mô hình cục bộ LLM khi tiến trình Upscale GPU đang hoạt động.

---

## 5. Ranh Giới Hành Động Cho AI Agent (Agent Boundaries)

* **Kỷ Luật Commit Từng Yêu Cầu (Mandatory Commit Per Request):**
  * Sau khi hoàn thành **MỖI** yêu cầu (dù từ Người dùng trực tiếp hay từ AI nghiên cứu), **bắt buộc phải commit Git ngay lập tức**. Tuyệt đối không dồn gộp nhiều yêu cầu hoặc nhiều đợt sửa đổi vào một commit chung.
  * Thông điệp commit (commit message & body) phải trình bày rõ ràng 3 yếu tố:
    1. **Nguồn yêu cầu:** Do Người dùng yêu cầu hay AI nghiên cứu đề xuất.
    2. **Hành động cụ thể:** Đã làm gì, đã fix những lỗi gì, đã tối ưu hoặc tái cấu trúc những module nào.
    3. **Số liệu & Kết quả:** Đính kèm số liệu kiểm chứng (kết quả `pytest`, dung lượng bộ nhớ giải phóng, fps, trạng thái audit, v.v.).
* **Bảo vệ Gemini API Key:** Key lấy từ biến môi trường `GEMINI_API_KEY` hoặc file `.env`. Nghiêm cấm in, ghi log, hoặc commit API key ra màn hình và file lưu trữ.
* **Bảo vệ Quota Ban Ngày:** Tuyệt đối không chạy lệnh gọi API Gemini thật trong khoảng từ **14:00 đến đêm chạy**. Mọi kiểm thử chức năng bắt buộc dùng cờ `--dry-run` hoặc stub backend.
* **Bảo toàn Checkpoint Resume:** Tuyệt đối không xóa hay sửa đổi các file trong thư mục `.temp_<basename>` trên SSD khi chưa có sự xác nhận của người dùng.
* **Hỏi Ý Kiến Trước Khi Thực Hiện:**
  * Chạy các tác vụ GPU/AI dài (> 100 frames).
  * Thay đổi model SR mặc định hoặc các tham số chất lượng mã hóa NVENC.

---

## 6. Chỉ Mục Tài Liệu Chuyên Sâu & Quyết Định Kỹ Thuật (ADR)

> **Hồ sơ lý do kỹ thuật chi tiết:** Xem [`docs/agents/DECISIONS.md`](docs/agents/DECISIONS.md).

| Phân hệ chuyên sâu | Tài liệu tham chiếu | Trọng tâm kỹ thuật |
| :--- | :--- | :--- |
| **GPU & Video Pipeline** | [`docs/video-pipeline.md`](docs/video-pipeline.md) | PyTorch CUDA FP16 (`src/cuda_engine.py`), batch size 6–8, `janai-fix` residual +2x, NVENC 10-bit p010le, ma trận BT.709. |
| **Dedup & DIS Flow** | [`docs/dedup-architecture.md`](docs/dedup-architecture.md) | Split segment theo GOP, scan `mpdecimate` (768:320:0.33), dòng chảy quang học DIS lia/zoom, tái tạo Concat Demuxer. |
| **Dịch Thuật Subtrans** | [`subtrans/AGENTS.md`](subtrans/AGENTS.md) | Gemini 3-key round-robin, trích xuất 5.1 FC mono + bandpass, regex lọc ảo giác Whisper & tiếng hét lặp token, Monotonic DTW auto-align. |
| **Audit & An Toàn** | [`docs/audit-and-safety.md`](docs/audit-and-safety.md) | Output Sanity Guard ($|\Delta Y| \le 6$, corr $\ge 0.98$), Mux Audit bit-exact stream copy, auto-cleanup, AntiSleep native Win32. |
| **Nhật Ký Thực Nghiệm** | [`docs/engineering-log.md`](docs/engineering-log.md) | Lịch sử đo đạc benchmark, VRAM, fps, soak tests và phân tích viễn trắc. |

---

## 7. Tiêu Chí Nghiệm Thu (Definition of Done)

Một phiên làm việc hoặc mã nguồn được coi là hoàn tất khi thỏa mãn đầy đủ các điều kiện sau:
1. **Kiểm thử tự động:** Chạy `pytest` đạt **PASS 100%** cho toàn bộ các bài test không đánh dấu `@pytest.mark.gpu`.
2. **Kiểm tra bộ nhớ Agent:** Chạy `python scripts/agent_context.py --check` đạt **0 lỗi** (toàn bộ liên kết hợp lệ, nằm trong byte budget).
3. **Kiểm định Mux:** Hàm `verify_mux_integrity` trả về `status: "PASS"` (khớp số kênh audio, số luồng sub, và số lượng font chữ đính kèm).
4. **Bảo toàn khung hình:** `ffprobe -count_frames` trên video 4K thành phẩm khớp chính xác từng khung hình so với video nguồn gốc.
5. **Trạng thái Git & Kỷ luật Commit:** Mỗi yêu cầu hoàn thành đều được commit riêng biệt với thông điệp rõ nguồn, nội dung làm/fix và số liệu; `git status` sạch hoàn toàn, không có file tạm bị bỏ sót; tag phiên bản được cập nhật tương ứng.
