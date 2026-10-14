# AGENTS.md — AI Agent Guidelines & Architecture Compass

> **Kim chỉ nam dành cho các AI Agent khi đọc, bảo trì và phát triển codebase này.**
> **Quy tắc bắt buộc:** Ngắn gọn, chuẩn xác, không suy diễn; cập nhật file này sau mỗi lần tái cấu trúc hoặc thay đổi kiến trúc code; **BẮT BUỘC commit chi tiết sau MỖI yêu cầu** (nêu rõ nguồn yêu cầu, việc đã làm, lỗi đã fix, số liệu thực nghiệm).
> **Chỉ mục tài liệu chuyên sâu:** [Video Pipeline](docs/video-pipeline.md) | [Dedup & DIS](docs/dedup-architecture.md) | [Subtrans](subtrans/AGENTS.md) | [Audit & Safety](docs/audit-and-safety.md) | [Engineering Log](docs/engineering-log.md).

---

## 1. Mục Tiêu & Ràng Buộc Cốt Lõi (Core Invariants)

* **Thiết bị vận hành mục tiêu:**
  * **Windows Workstation:** Windows 11 64-bit, Intel Core i5-12600K (10C/16T, tăng tốc AVX2), NVIDIA GeForce RTX 3060 (12GB VRAM GDDR6, CUDA Cores & Tensor Cores), 32GB RAM, FFmpeg hỗ trợ `hevc_nvenc`.
* **Môi trường bắt buộc:** **100% CỤC BỘ trên máy người dùng (Windows 11)**. Tuyệt đối không đề xuất Cloud GPU, Colab, Kaggle.
* **Tôn chỉ Pareto:** Chất lượng hình ảnh cân bằng với thời gian. Không hy sinh nét vẽ anime lấy tốc độ rẻ tiền; không lãng phí tài nguyên cho cải thiện vô hình.
* **Sức chứa thực tế:**
  * Trên RTX 3060 (PyTorch CUDA FP16 Batch Inference `batch_size=6` đạt ~4.8–5.2 fps suy luận thuần, ~10–12+ fps hiệu dụng với Dedup & `cv2.imwrite`): **~6–8 tập mỗi đêm** (~45 phút – 1 giờ/tập với ~17.000 unique frames).

---

## 2. Lệnh CLI Cơ Bản & Bảng Cờ (Commands & Flags)

### A. Lệnh Vận Hành Thường Dùng (PowerShell / CMD)

```powershell
# Nâng cấp 1 tập video đơn lẻ:
python upscale.py path/to/source.mkv -o out/

# Chạy nhanh qua launcher tiện ích Windows:
.\upscale.bat path/to/source.mkv

# Xem trước nhanh 10 giây ở 30% phim:
python upscale.py path/to/source.mkv --preview 10

# Chạy hàng đợi nhiều tập qua đêm (tự động bật auto-offset và lập lịch deadline):
python upscale.py --queue ep1.mkv ep2.mkv -o out/ --deadline 07:00 --allow-partial

# Tiếp tục tập bị gián đoạn (tự động nhận checkpoint .temp):
python upscale.py path/to/source.mkv -o out/

# Dịch phụ đề riêng lẻ:
python subtrans.py path/to/source.mkv -o out/
.\subtrans.bat path/to/source.mkv

# Chạy kiểm thử tự động toàn diện:
pytest
```

### B. Bảng Cờ CLI Chính (`upscale.py`)
| Cờ CLI | Mặc định | Ý nghĩa & Hành vi |
| :--- | :--- | :--- |
| `-o`, `--output` | `out/<tên>_4K.mkv` | Thư mục hoặc file đích xuất video 4K |
| `--queue` | `None` | Danh sách các file video chạy nối tiếp qua đêm |
| `--engine` | `cuda` | Engine suy luận: `cuda` (NVIDIA CUDA FP16) hoặc `cpu` |
| `--model` | `animevideov3` | Model Super-Resolution đăng ký trong `models/models.yaml` |
| `--batch-size` | `6` (CUDA RTX 3060) | Kích thước batch cho AI Inference GPU (tối ưu hóa Tensor Cores) |
| `--subs-quality` | `fast` | Chất lượng dịch: `fast` (`gemini-3.5-flash-lite` 500 RPD) hoặc `high` (`gemini-3.8-flash`) |
| `--subs-only` | `False` | Chỉ trích xuất và dịch phụ đề, bỏ qua giai đoạn video upscale |
| `--subs-upgrade` | `False` | Nâng cấp phụ đề tiếng Việt hiện có lên bản dịch chất lượng cao (Gemini) |
| `--auto-align-en` | `True` | Ghim mốc thời gian phụ đề Whisper/tiếng Nhật theo track tiếng Anh phòng thu |
| `--no-auto-align-en` | `False` | Tắt tính năng tự động ghim mốc thời gian phụ đề theo tiếng Anh |
| `--allow-partial` | `False` | Cho phép dừng tập an toàn ở ranh giới chunk 8s khi chạm deadline ca đêm |
| `--auto-offset` | Bật trong `--queue` | Tự động đo và bù độ lệch thời gian phụ đề tiếng Nhật - Anh |
| `--no-auto-offset` | `False` | Ép tắt tính năng auto-offset trong hàng đợi |
| `--sub-offset-ms` | `0` | Bù offset phụ đề thủ công theo mili-giây (luôn có độ ưu tiên cao nhất) |
| `--no-embed` | `False` | Tắt bước nhúng phụ đề vào container MKV thành phẩm |
| `--keep-temp` | `False` | Giữ lại thư mục tạm `.temp_<basename>` (mặc định: tự động dọn dẹp sạch khi Mux PASS) |
| `--clean-temp`, `--cleanup` | `False` | Quét dọn toàn bộ thư mục tạm và gom file audit/cast trong thư mục xuất xưởng rồi thoát |
| `--dedup-policy` | `default` | Chính sách dedup: `default` (768:320:0.33), `sensitive-a` (512:160:0.15), `sensitive-b` (384:128:0.10), `motion`, `drift`, `dis` (dòng chảy quang học DIS), `off` |
| `--dis-thresh` | `0.30` | Ngưỡng dòng chảy quang học DIS tính bằng pixel 1080p (`0.20` - `0.50`) |
| `--dry-run` | `False` | Chạy mô phỏng toàn bộ quy trình mà không gọi AI hay API |

---

## 3. Kiến Trúc Thư Mục & Trách Nhiệm Module

```text
ai-video-upscaler/
├── upscale.py              # CLI Entrypoint (Argparse, UTF-8 console, Queue orchestrator, Checkpoint Resume)
├── upscale.bat             # Windows batch launcher (One-click chạy trên Windows)
├── subtrans.py             # Subtitle Translation CLI Runner
├── subtrans.bat            # Windows batch launcher cho Subtrans (UTF-8, Multi-Key Round-Robin)
├── shows.yaml              # Ánh xạ mô hình SR theo từng bộ anime (người dùng cấu hình)
├── pytest.ini              # Cấu hình Pytest (testpaths = tests subtrans/tests)
├── requirements.txt        # Danh sách thư viện Python phụ thuộc
├── README.md               # Hướng dẫn sử dụng & CLI options (Windows 11 & NVIDIA RTX)
├── LICENSE                 # Giấy phép mã nguồn mở
├── AGENTS.md               # Kim chỉ nam kiến trúc & các bất biến kỹ thuật
├── src/                    # Pipeline phóng to video (CUDA FP16, NVENC 10-bit, FFmpeg)
│   ├── config.py           # Single source of truth: cấu hình Windows, NVENC, batch size, paths
│   ├── subtitles.py        # Trích xuất phụ đề rời/container, Whisper STT fallback
│   ├── ramdisk.py          # RAMDisk Context Manager: Quản lý bộ đệm khung hình và rolling cleanup
│   ├── ffmpeg_ops.py       # Thao tác FFmpeg/FFprobe chuẩn phòng thu (NVENC BT.709 10-bit, Mux, Embed Subs)
│   ├── mps_engine.py       # PyTorch GPU Engine: Hỗ trợ CUDA FP16/FP32 Native, Tensor Cores, batch inference
│   ├── ai_engine.py        # Super-Resolution Interface abstraction
│   ├── models_resolver.py  # SR Models Registry resolver & weights loader (SHA-256 verification)
│   ├── mux_audit.py        # Stream copy integrity audit & font count verifier
│   ├── dedup.py            # Deduplication Engine: Preflight mpdecimate scan, DIS Optical Flow & Concat Re-expansion
│   ├── preflight.py        # Win32 SetThreadExecutionState (AntiSleep), kiểm tra sạc & dung lượng đĩa trống
│   ├── pipeline.py         # Orchestrator: Triple-buffering, Rolling Segments, Checkpoint Resume
│   ├── scheduler.py        # Dynamic overnight queue scheduler & ETA predictor
│   ├── night_report.py     # Báo cáo viễn trắc tổng kết phiên chạy qua đêm Markdown
│   ├── output_sanity.py    # Output Sanity Guard: Chốt chặn kiểm định đầu ra (|ΔY| <= 6, corr >= 0.98)
│   ├── cleanup.py          # Auto-cleanup: Dọn dẹp thư mục tạm (.temp_*), symlinks & gom gọn audit/
│   └── logger.py           # Structured telemetry logging (logs/*.log & logs/latest.log symlink)
├── subtrans/               # Engine dịch thuật phụ đề (xem subtrans/AGENTS.md)
├── benchmarks/             # Kịch bản đo lường hiệu năng, Soak Tests & Khảo sát thuật toán
│   ├── upscale/            # Benchmarks hiệu năng GPU, VRAM, tốc độ suy luận, Dedup
│   ├── subtrans/           # Đánh giá độ ổn định xưng hô (Golden Pronoun Evaluations)
│   └── research/           # Kịch bản hiệu chuẩn và nghiên cứu chuyên sâu
├── scripts/                # Tiện ích vận hành (organize subs, batch launchers)
├── models/                 # Super-Resolution models registry (models.yaml)
├── models_bench/           # Trọng số mô hình AI PyTorch (.pth)
├── glossaries/             # Bảng thuật ngữ riêng từng bộ anime (glossary.<anime>.yaml)
├── samples/                # Video mẫu phục vụ kiểm thử (Mushoku, Slime, Sword)
├── test_frames/            # 3 frame ảnh mẫu chuẩn (Action, Close-up, Scenery)
├── tests/                  # Bộ kiểm thử tự động (100% Pytest Unit & Integration Tests)
├── docs/                   # Tài liệu chuyên sâu: video-pipeline.md, dedup-architecture.md, audit-and-safety.md...
└── logs/                   # Log chi tiết có cấu trúc & báo cáo tổng kết đêm
```

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

## 6. Bảng Quyết Định Kỹ Thuật Đã Chốt & Chỉ Mục Tài Liệu Chuyên Sâu

### A. Phân Hệ Phóng To Video AI & GPU Engine — [docs/video-pipeline.md](docs/video-pipeline.md)
* **Mô hình Super-Resolution:** Chọn `realesr-animevideov3` (kiến trúc mạng `SRVGGNetCompact` 4x kết hợp Bicubic 0.5 để xuất 2x).
* **Động cơ suy luận chính:** PyTorch CUDA FP16 Native (`src/mps_engine.py` hỗ trợ CUDA device, FP16 half precision, `torch.cuda.synchronize()`). Hỗ trợ Batch Inference `batch_size=6` tận dụng Tensor Cores: **4.8 – 5.2 fps** thuần GPU, **~10–12+ fps** hiệu dụng với Dedup và ghi frame `cv2.imwrite` (VRAM ~6.2GB).
* **Đính chính mô hình AnimeJaNai:** Thêm phép cộng nearest $\times 2$ residual tại tag `janai-fix` (`F.interpolate(x, scale_factor=2.0, mode='nearest')`), khớp bit-exact 0.0 với spandrel CPU chuẩn (PSNR 75–158 dB).
* **Mã hóa phần cứng 10-bit:** `hevc_nvenc` Main 10 Profile (`p010le`, preset `p6`, multipass `qres`, lookahead `24`, spatial AQ).
* **Màu sắc & Stream Copy:** Ma trận BT.709 hai chiều (Limited YUV $\leftrightarrow$ Full RGB24), gắn cờ `-tag:v hvc1`, cờ mux bắt buộc `-max_interleave_delta 0` copy nguyên bản audio, subtitle, fonts và chapters.

### B. Phân Hệ Deduplication Preflight & Dòng Chảy Quang Học DIS — [docs/dedup-architecture.md](docs/dedup-architecture.md)
* **Ngưỡng mặc định:** `hi=768:lo=320:frac=0.33` (giảm ~60% thời gian GPU; 0% lỗi khẩu hình/chớp mắt trên anime).
* **Khắc phục lệch khung hình BD Remux GOP dài:** `split_video_segments` stream copy trước, `get_segments_frame_counts` đo frame thực tế theo Keyframe GOP, ép giữ frame đầu mỗi chunk, tái tạo `runs` khớp 100% số frame.
* **Nhận Diện Chuyển Động Qua DIS Optical Flow:** Thuật toán dòng chảy quang học DIS (Pan/Tilt translation, Radial expansion zoom in/out, Local/diffuse $P_{90}$) kết hợp Candidate Drop Gating. Khắc phục các đoạn lia máy và zoom mà `mpdecimate` bỏ sót. Hỗ trợ cờ `--dedup-policy dis` và `--dis-thresh 0.30`.
* **Tái tạo khung hình 4K (Re-expansion):** FFmpeg Concat Demuxer theo thời lượng `runs` đại diện + `fps=24000/1001` (triệt tiêu toàn bộ chi phí đĩa của hardlinks).

### C. Phân Hệ Dịch Thuật Phụ Đề Subtrans — [subtrans/AGENTS.md](subtrans/AGENTS.md)
* **100% Cloud API & Phân Phối Tải 3 Key:** Chạy 100% Gemini qua Google GenAI SDK với cơ chế luân phiên Round-Robin 3 API key (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`) chống nghẽn RPM. Mặc định dùng `gemini-3.5-flash-lite` (500 RPD/key, tốc độ cực nhanh); chế độ chất lượng cao `--subs-quality high` dùng `gemini-3.8-flash`.
* **Cơ Chế Whisper STT Fallback & Auto-Align:** Tự động dùng Whisper trích xuất tiếng Nhật khi video thiếu track JA (`src/subtitles.py`), ghim chuẩn xác mốc thời gian theo track tiếng Anh phòng thu qua 3-pass snapping (`--auto-align-en`).
* **Đồng nhất xưng hô & Bảo tồn Romaji:** Stage A Cast Sheet (`out/<name>.cast.yaml`) nạp vô điều kiện vào mọi batch; giữ nguyên tên Romaji khi đối thoại gọi tên.
* **Hiển thị tối ưu & CC Splitting:** Tự động gộp 2 người nói cùng lúc thành Dual Dialogue 2 dòng (`- Lời 1\n- Lời 2`), gộp câu ngắn (<1.2s), thời lượng đọc tối thiểu $\ge 1.2\text{s} - 1.4\text{s}$, khoảng nghỉ 40ms chống flicker.
* **Smart Cloud Cascade & Subtitle RAG:** Điểm khó $S(cue)$, Draft Pass Lite kết hợp Repair Pass Flash; kho tri thức SQLite `.cache/knowledge.db`.

### D. Phân Hệ Kiểm Định Toàn Vẹn, Sanity Guard & An Toàn — [docs/audit-and-safety.md](docs/audit-and-safety.md)
* **Output Sanity Guard:** `src/output_sanity.py` kiểm tra sau chunk 1 và mỗi 10 chunks sau ($|\Delta Y| \le 6/255$, tương quan $\ge 0.98$ so với bicubic); dừng khẩn cấp và cấm mux nếu rớt.
* **Cổng Kiểm Toán Mux Audit:** `src/mux_audit.py` đối soát bit-exact stream copy, MD5 track âm thanh, số lượng phụ đề, font chữ đính kèm và tổng số khung hình `video_frame_count`.
* **Tự Động Dọn Dẹp:** `src/cleanup.py` tự động kích hoạt khi Mux Audit PASS; xóa sạch `.temp_<basename>`, gom file báo cáo vào thư mục con `audit/`.
* **Vòng Đời An Toàn:** Win32 API native `SetThreadExecutionState` chống sleep máy trong suốt tiến trình.

### E. Nhật Ký Kỹ Thuật & Số Đo Benchmarks Chi Tiết — [docs/engineering-log.md](docs/engineering-log.md)
* Lưu trữ chi tiết các thử nghiệm kiểm chứng, benchmark soak test và phân tích viễn trắc.

---

## 7. Tiêu Chí Nghiệm Thu (Definition of Done)

Một phiên làm việc hoặc mã nguồn được coi là hoàn tất khi thỏa mãn đầy đủ các điều kiện sau:
1. **Kiểm thử tự động:** Chạy `pytest` đạt **PASS 100%** cho toàn bộ các bài test không đánh dấu `@pytest.mark.gpu`.
2. **Kiểm định Mux:** Hàm `verify_mux_integrity` trả về `status: "PASS"` (khớp số kênh audio, số luồng sub, và số lượng font chữ đính kèm).
3. **Bảo toàn khung hình:** `ffprobe -count_frames` trên video 4K thành phẩm khớp chính xác từng khung hình so với video nguồn gốc.
4. **Trạng thái Git & Kỷ luật Commit:** Mỗi yêu cầu hoàn thành đều được commit riêng biệt với thông điệp rõ nguồn, nội dung làm/fix và số liệu; `git status` sạch hoàn toàn, không có file tạm bị bỏ sót; tag phiên bản được cập nhật tương ứng.
