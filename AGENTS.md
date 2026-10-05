# AGENTS.md — AI Agent Guidelines & Architecture Compass

> **Kim chỉ nam dành cho các AI Agent khi đọc, bảo trì và phát triển codebase này.**
> **Quy tắc bắt buộc:** Ngắn gọn, chuẩn xác, không suy diễn; cập nhật file này sau mỗi lần tái cấu trúc hoặc thay đổi kiến trúc code.
> Chi tiết thực nghiệm & số đo: xem [docs/engineering-log.md](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md) | Phân hệ phụ đề: xem [subtrans/AGENTS.md](file:///Users/xis108/Desktop/upscale/subtrans/AGENTS.md).

---

## 1. Mục Tiêu & Ràng Buộc Cốt Lõi (Core Invariants)

* **Thiết bị vận hành:** Apple Silicon MacBook Pro M3 Pro (11-core CPU, 14-core GPU, 18GB Unified Memory).
* **Môi trường bắt buộc:** **100% CỤC BỘ trên macOS**. Tuyệt đối không đề xuất Cloud GPU, Colab, Kaggle.
* **Tôn chỉ Pareto:** Chất lượng hình ảnh cân bằng với thời gian. Không hy sinh nét vẽ anime lấy tốc độ rẻ tiền; không lãng phí tài nguyên cho cải thiện vô hình.
* **Sức chứa thực tế:** **~2 tập mỗi đêm** (~3 giờ/tập với ~17.000 unique frames @ 1.68 fps sustained sau deduplication).

---

## 2. Lệnh CLI Cơ Bản & Bảng Cờ (Commands & Flags)

### A. Lệnh Vận Hành Thường Dùng
```bash
# Nâng cấp 1 tập video đơn lẻ (tự động nhận track phụ đề cùng thư mục)
./upscale.sh path/to/source.mkv -o ~/Documents/Upscaled/

# Chạy hàng đợi nhiều tập qua đêm (tự động bật auto-offset và lập lịch deadline 8h)
./upscale.sh --queue ep1.mkv ep2.mkv -o ~/Documents/Upscaled/

# Tiếp tục tập bị gián đoạn (tự động nhận thư mục checkpoint .temp)
./upscale.sh path/to/ep3.mkv -o ~/Documents/Upscaled/

# Dịch phụ đề độc lập (không chạy AI video)
./upscale.sh --subs-only path/to/source.mkv -o ~/Documents/Upscaled/

# Chạy kiểm thử tự động toàn diện
./venv/bin/pytest
```

### B. Bảng Cờ CLI Chính (`upscale.py`)
| Cờ CLI | Mặc định | Ý nghĩa & Hành vi |
| :--- | :--- | :--- |
| `-o`, `--output` | `~/Documents/Upscaled/` | Thư mục hoặc file đích xuất video 4K |
| `--queue` | `None` | Danh sách các file video chạy nối tiếp qua đêm |
| `--engine` | `mps` | Engine suy luận: `mps` (PyTorch Metal FP16) hoặc `ncnn` (Vulkan MoltenVK) |
| `--model` | `animevideov3` | Model Super-Resolution đăng ký trong `models/models.yaml` |
| `--allow-partial` | `False` | Cho phép dừng tập an toàn ở ranh giới chunk 8s khi chạm deadline ca đêm |
| `--auto-offset` | Bật trong `--queue` | Tự động đo và bù độ lệch thời gian phụ đề tiếng Nhật - Anh |
| `--no-auto-offset` | `False` | Ép tắt tính năng auto-offset trong hàng đợi |
| `--sub-offset-ms` | `0` | Bù offset phụ đề thủ công theo mili-giây (luôn có độ ưu tiên cao nhất) |
| `--no-embed` | `False` | Tắt bước nhúng phụ đề vào container MKV thành phẩm |
| `--dry-run` | `False` | Chạy mô phỏng toàn bộ quy trình mà không gọi AI hay API |

---

## 3. Kiến Trúc Thư Mục & Trách Nhiệm Module

```text
upscale/
├── upscale.py              # CLI Entrypoint (Argparse, Queue orchestrator, Checkpoint Resume)
├── upscale.sh              # Shell launcher native (tự nhận virtualenv hoặc python3)
├── subtrans.py             # Subtitle Translation CLI Runner
├── shows.yaml              # Ánh xạ mô hình SR theo từng bộ anime (người dùng cấu hình)
├── pytest.ini              # Cấu hình Pytest (testpaths = tests subtrans/tests)
├── requirements.txt        # Danh sách thư viện Python phụ thuộc
├── README.md               # Hướng dẫn sử dụng & CLI options
├── LICENSE                 # Giấy phép mã nguồn mở
├── AGENTS.md               # Kim chỉ nam kiến trúc & các bất biến kỹ thuật
├── src/                    # Pipeline phóng to video (Triple-buffering, Metal GPU, FFmpeg, RAM Disk)
│   ├── config.py           # Single source of truth: Hằng số, paths, params VideoToolbox & RAM Disk
│   ├── ramdisk.py          # RAMDisk Context Manager: mount/unmount HFS+ RAM Disk ảo (Zero SSD Wear)
│   ├── ffmpeg_ops.py       # Thao tác FFmpeg/FFprobe chuẩn phòng thu (BT.709 10-bit, Mux, Embed Subtitles)
│   ├── mps_engine.py       # PyTorch MPS FP16 Native Apple Silicon inference worker (Engine chính)
│   ├── ai_engine.py        # NCNN Vulkan via MoltenVK (Metal API) inference worker (Engine fallback)
│   ├── models_resolver.py  # SR Models Registry resolver & weights loader
│   ├── mux_audit.py        # Stream copy integrity audit & font count verifier
│   ├── dedup.py            # Deduplication Engine: Preflight mpdecimate scan & Concat Re-expansion
│   ├── preflight.py        # CaffeinateContext (chống sleep), Check AC/Pin, Check đĩa trống, macOS Notifs
│   ├── pipeline.py         # Orchestrator: Triple-buffering, Rolling Segments, Checkpoint Resume
│   ├── scheduler.py        # Dynamic overnight queue scheduler & ETA predictor
│   ├── night_report.py     # Báo cáo viễn trắc tổng kết phiên chạy qua đêm Markdown
│   └── logger.py           # Structured telemetry logging (logs/*.log & logs/latest.log symlink)
├── subtrans/               # Engine dịch thuật phụ đề (xem subtrans/AGENTS.md)
├── bin/realesrgan/         # Universal Mach-O binary & NCNN models (.param / .bin)
├── models/                 # Super-Resolution models registry (models.yaml)
├── glossaries/             # Bảng thuật ngữ riêng từng bộ anime (glossary.<anime>.yaml)
├── samples/                # Video mẫu phục vụ kiểm thử (sample_30s.mkv, sample_3min.mkv)
├── test_frames/            # 3 frame ảnh mẫu chuẩn (Action, Close-up, Scenery)
├── tests/                  # Bộ kiểm thử tự động, benchmark hiệu năng, soak test, chaos test
├── docs/                   # Tài liệu kỹ thuật chi tiết (engineering-log.md)
└── logs/                   # Log chi tiết có cấu trúc & báo cáo tổng kết đêm
```

---

## 4. Những Bất Biến Kỹ Thuật (Do NOT Break)

1. **Bảo vệ NVMe SSD qua RAM Disk ảo:**
   * Mọi khung hình trung gian dạng PNG (cả trích xuất 1080p và xuất 4K AI) **bắt buộc phải ghi trên RAM Disk ảo HFS+** (`/Volumes/RAMDisk`).
   * Sau khi mỗi phân đoạn (chunk 8s) mã hóa xong, phải dọn sạch frame PNG trên RAM Disk ngay lập tức.
   * SSD chỉ lưu trữ các segment HEVC đã nén (`up_*.mp4`) trong thư mục `.temp_<basename>` để bảo toàn khả năng resume.
2. **Bit-exact Stream Copy & Cờ Mux Phòng Thu:**
   * Video 4K thành phẩm **bắt buộc Stream Copy nguyên bản**: Audio đa kênh, Phụ đề mềm (ASS/SRT), Font chữ nhúng (TTF/OTF) và Chapters.
   * Lệnh mux bắt buộc dùng cờ:
     `-map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 -c copy -max_interleave_delta 0`
   * Cờ `-max_interleave_delta 0` là bắt buộc để ngăn lỗi tràn buffer và mất tiếng cục bộ sau vài phút phát lại.
3. **Chuỗi ma trận màu sắc chuẩn xác (BT.709 Two-way Matrix):**
   * Trích xuất: `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc` (YUV Limited -> RGB24 Full).
   * Mã hóa: `scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv` (RGB24 Full -> YUV Limited `p010le`).
   * Metadata gắn cờ: `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`.
4. **Vòng đời an toàn (Process Lifecycle & Auto Cleanup):**
   * `caffeinate` và `hdiutil` (RAM Disk) phải luôn nằm trong `try...finally` hoặc Context Manager với `atexit` để thu hồi sạch sẽ kể cả khi bị `Ctrl+C` (SIGINT).
5. **Độc quyền GPU & Điều phối tài nguyên:**
   * Tại một thời điểm chỉ duy nhất 1 tiến trình infer GPU Metal được hoạt động.
   * Pipeline chạy gối đầu 3 tầng: CPU Decode (chunk N+1) và Apple Media Engine Encode (chunk N-1) chạy song song với GPU Infer (chunk N).
   * Tuyệt đối không nạp mô hình cục bộ Ollama khi tiến trình Upscale GPU đang hoạt động.

---

## 5. Ranh Giới Hành Động Cho AI Agent (Agent Boundaries)

* **Bảo vệ Gemini API Key:** Key lấy từ biến môi trường `GEMINI_API_KEY` hoặc file `.env`. Nghiêm cấm in, ghi log, hoặc commit API key ra màn hình và file lưu trữ.
* **Bảo vệ Quota Ban Ngày:** Tuyệt đối không chạy lệnh gọi API Gemini thật trong khoảng từ **14:00 đến đêm chạy**. Mọi kiểm thử chức năng bắt buộc dùng cờ `--dry-run` hoặc stub backend.
* **Bảo toàn Checkpoint Resume:** Tuyệt đối không xóa hay sửa đổi các file trong thư mục `.temp_<basename>` trên SSD khi chưa có sự xác nhận của người dùng.
* **Hỏi Ý Kiến Trước Khi Thực Hiện:**
  * Chạy các tác vụ GPU/AI dài (> 100 frames).
  * Thao tác thủ công mount/unmount ổ đĩa hệ thống bằng `hdiutil`.
  * Thay đổi model SR mặc định hoặc các tham số chất lượng mã hóa VideoToolbox.

---

## 6. Bảng Quyết Định Kỹ Thuật Đã Chốt (Decisions Summary)

* **Mô hình Super-Resolution:** Chọn `realesr-animevideov3` (thắng 3/3 cảnh trong True Blind Test trên Liquid Retina XDR; chi tiết: [docs/engineering-log.md#2](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#2)).
* **Engine suy luận chính:** Chọn **PyTorch MPS FP16 Native** (`src/mps_engine.py`, đạt 1.86 fps warm-up, 1.68 fps sustained soak; NCNN Vulkan đóng vai trò fallback; chi tiết: [docs/engineering-log.md#4](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#4)).
* **Ngưỡng Deduplication:** Chọn `hi=768:lo=320:frac=0.33` (giảm 60.86% thời gian GPU; đối soát 16x16/8x8 block đạt 0% lỗi khẩu hình/chớp mắt trên mẫu kiểm tra; chi tiết: [docs/engineering-log.md#3](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#3)).
* **Tái tạo khung hình 4K (Re-expansion):** Chọn **FFmpeg Concat Demuxer** theo thời lượng run đại diện + `fps=24000/1001` (triệt tiêu 100% chi phí đĩa của hardlinks).
* **Đồng nhất xưng hô:** Stage A Cast Sheet (`out/<name>.cast.yaml`) nạp vô điều kiện vào mọi batch; chuỗi Gemini ưu tiên bảo toàn đại từ (chi tiết: [subtrans/AGENTS.md](file:///Users/xis108/Desktop/upscale/subtrans/AGENTS.md)).
* **Căn chỉnh phụ đề tự động:** Thuật toán `--auto-offset` kiểm định 4 tầng (ngưỡng 1.0s, số cặp >= 30, quintile drift <= 0.5s, độ khớp tăng >= 15 điểm; chi tiết: [docs/engineering-log.md#6](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#6)).
* **Cổng kiểm định Mux:** `src/mux_audit.py` xác minh toàn vẹn luồng stream copy và khớp số lượng font chữ nhúng trước khi xuất xưởng.

---

## 7. Tiêu Chí Nghiệm Thu (Definition of Done)

Một phiên làm việc hoặc mã nguồn được coi là hoàn tất khi thỏa mãn đầy đủ các điều kiện sau:
1. **Kiểm thử tự động:** Chạy `pytest` đạt **PASS 100%** cho toàn bộ các bài test không đánh dấu `@pytest.mark.gpu`.
2. **Kiểm định Mux:** Hàm `verify_mux_integrity` trả về `status: "PASS"` (khớp số kênh audio, số luồng sub, và số lượng font chữ đính kèm).
3. **Bảo toàn khung hình:** `ffprobe -count_frames` trên video 4K thành phẩm khớp chính xác từng khung hình so với video nguồn gốc.
4. **Trạng thái Git:** `git status` sạch hoàn toàn, không có file tạm bị bỏ sót; tag phiên bản được cập nhật tương ứng.
