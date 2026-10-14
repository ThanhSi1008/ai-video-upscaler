# AI Video Upscaler 4K & AI Subtitle Translator (Windows 11 & NVIDIA RTX)

Công cụ tối ưu hóa chuyên sâu để **Upscale Video Anime lên 4K (2x Super-Resolution)** và **Dịch phụ đề Anime bằng AI (TriSub AI)** chạy hoàn toàn trên Terminal / PowerShell Windows.

Hệ thống được thiết kế và tối ưu hóa tuyệt đối cho cấu hình phần cứng:
* **Hệ điều hành:** Windows 11 64-bit.
* **CPU:** Intel Core i5-12600K (10 nhân / 16 luồng, tập lệnh tăng tốc AVX2 cho I/O OpenCV C++ `cv2.imwrite`).
* **GPU:** NVIDIA GeForce RTX 3060 (12GB GDDR6 VRAM, CUDA Cores & Tensor Cores).
* **Mã hóa phần cứng:** NVIDIA NVENC HEVC Main 10 (`hevc_nvenc`, Preset `p6`, Multipass `qres`, Lookahead `24`, Spatial AQ, `p010le`, BT.709).
* **Chống sleep máy:** Win32 API native `SetThreadExecutionState` (`ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED`).

---

## ⚡ 1. Công Cụ Upscale Video Anime 4K (`upscale.py` / `upscale.bat`)

Sử dụng trực tiếp nhân GPU NVIDIA Tensor Cores kết hợp bộ mã hóa phần cứng NVENC để đạt chất lượng chuẩn tham chiếu phòng thu:

* **Mô hình AI:** `realesr-animevideov3` (RealESR-AnimeVideoV3) chuyên dụng cho video anime 2x (khử nhiễu nén, giữ nét vẽ tự nhiên, không bị hiệu ứng viền sáp).
* **Mã hóa phần cứng 10-bit:** NVIDIA NVENC `hevc_nvenc` Main 10 Profile (`p010le`, Preset `p6`, Multipass `qres`, Lookahead `24`, Spatial AQ, Tag `hvc1`, BT.709 Limited Range TV).
* **Quản lý bộ nhớ đệm khung hình:** Tự động đệm và dọn sạch phân đoạn cuốn chiếu (Rolling Segments Cleanup) ngay sau khi từng chunk mã hóa xong, hoặc hỗ trợ RAM Disk ảo (ImDisk), bảo vệ tuyệt đối tuổi thọ ổ cứng NVMe SSD.
* **Tối ưu hóa I/O khung hình:** Sử dụng OpenCV C++ (`cv2.imwrite`) tận dụng tập lệnh AVX2 của CPU Core i5-12600K, nhanh hơn PIL từ 2x đến 3x.
* **Bảo toàn nguyên bản dữ liệu gốc & Đồng bộ âm thanh:** Tự động **Stream Copy** toàn bộ audio đa kênh (FLAC/AAC/TrueHD), phụ đề mềm đa ngôn ngữ (ASS/SRT), font chữ nhúng (TrueType/OpenType) và chapters từ file gốc với cờ `-max_interleave_delta 0` ép đan xen đều đặn từng packet theo DTS (loại bỏ hoàn toàn lỗi audio dồn khối gây mất tiếng; không dùng file audio tạm).
* **Checkpoint Resume:** Hỗ trợ tiếp tục chạy ngay từ phân đoạn đang dở nếu bị tạm dừng (Ctrl+C hoặc tắt máy).
* **Cơ chế chống ngủ máy (Anti-Sleep):** Sử dụng Win32 API native `SetThreadExecutionState` gắn chặt với vòng đời tiến trình.
* **Kiểm tra an toàn trước khi chạy (Pre-flight Checks):** Cảnh báo nguồn điện AC vs Pin, kiểm tra dung lượng ổ cứng trống tối thiểu an toàn.
* **Output Sanity Guard & Mux Audit:** Tự động đo độ lệch sáng $|\Delta Y| \le 6/255$ và tương quan Pearson $r \ge 0.98$ sau chunk 1 để triệt tiêu nguy cơ video đen/hỏng, đồng thời đối soát bit-exact frame count, audio tracks và fonts trước khi bàn giao.

---

## 💻 2. Yêu Cầu Hệ Thống & Cài Đặt

### Cấu Hình Phần Cứng Windows (NVIDIA RTX)
* **Hệ điều hành:** Windows 10/11 64-bit.
* **GPU:** NVIDIA GeForce RTX 3060 (12GB VRAM) hoặc dòng RTX 20/30/40/50 series với Driver NVIDIA mới nhất.
* **CPU & RAM:** Intel Core i5/i7/i9 (khuyến nghị $\ge 16\text{ GB}$ RAM, tối ưu AVX2).
* **Công cụ bổ trợ:** FFmpeg hỗ trợ `hevc_nvenc` (khuyến nghị bản Gyan.dev Full Build, đã thêm vào PATH).

**Cài đặt trên Windows (PowerShell):**
```powershell
# 1. Clone repo và tạo virtual environment
git clone https://github.com/ThanhSi1008/ai-video-upscaler.git
cd ai-video-upscaler
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt PyTorch với hỗ trợ CUDA 12.4+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 4. Tải trọng số mô hình realesr-animevideov3
# (Tự động tải khi chạy lần đầu hoặc đặt file .pth vào models_bench/)
```

---

## 🚀 3. Hướng Dẫn Sử Dụng CLI (`upscale.py` / `upscale.bat`)

### Lệnh Vận Hành Trên Windows (PowerShell / CMD):
```powershell
# 1. Xem trước nhanh 10 giây (tự động cắt tại 30% phim, upscale và mở bằng trình phát mặc định):
python upscale.py samples/sample_sword_3min.mkv --preview 10

# 2. Chạy nhanh bằng launcher tiện ích Windows (.bat):
.\upscale.bat samples/sample_sword_3min.mkv

# 3. Chạy đầy đủ 1 tập video với bitrate 28M và phân đoạn 8 giây:
python upscale.py samples/sample_sword_3min.mkv -b 28M -s 8 -o out/sword_4K.mkv

# 4. Chạy hàng đợi nhiều tập qua đêm với deadline dừng an toàn:
python upscale.py --queue ep1.mkv ep2.mkv -o out/ --deadline 07:00 --allow-partial
```

### Bảng Tham Số Tùy Chọn:
* `-o`, `--output`: Đường dẫn file hoặc thư mục xuất (Mặc định: `out/<tên>_4K.mkv`).
* `-m`, `--model`: Mô hình AI (Mặc định: tự động chọn qua `shows.yaml` hoặc mặc định `animevideov3`).
* `--engine`: Động cơ suy luận (`cuda` cho NVIDIA GPU hoặc `cpu`; mặc định: `cuda`).
* `--precision`: Độ chính xác suy luận (`fp16` hoặc `fp32`, mặc định: `fp16`).
* `--batch-size`: Batch size cho AI Inference GPU (Mặc định: `6` trên RTX 3060 12GB).
* `--list-models`: Liệt kê các mô hình đã đăng ký trong `models/models.yaml`.
* `--force`: Bắt buộc chạy và ghi đè/xóa phân đoạn cũ nếu phát hiện xung đột mô hình.
* `--subs-after-upscale`: Tự động chạy dịch phụ đề sau khi toàn bộ hàng đợi upscale hoàn tất (Mặc định: Bật).
* `--no-subs`: Tắt giai đoạn dịch phụ đề sau upscale.
* `--subs-only`: Chỉ trích xuất và dịch phụ đề, bỏ qua giai đoạn upscale video.
* `--subs-upgrade`: Nâng cấp phụ đề tiếng Việt hiện có lên bản dịch chất lượng cao (Gemini).
* `--subs-quality {fast,high}`: Chất lượng dịch phụ đề (`fast`: `gemini-3.5-flash-lite` 500 RPD mặc định; `high`: `gemini-3.8-flash`).
* `--auto-align-en` / `--no-auto-align-en`: Tự động ghim mốc thời gian của Whisper / tiếng Nhật theo track tiếng Anh phòng thu (Mặc định: Bật).
* `--auto-offset` / `--no-auto-offset`: Tự động nhận diện và bù trừ độ lệch thời gian (time-offset) giữa track JA và EN.
* `--sub-offset-ms [ms]`: Bù trừ độ lệch thời gian phụ đề thủ công theo mili-giây.
* `--queue`: Danh sách nhiều tập video cần chạy tuần tự trong ca qua đêm.
* `--deadline HH:MM`: Mốc hạn chót dừng hàng đợi an toàn nếu tập kế tiếp dự kiến không kịp.
* `--allow-partial`: Cho phép tập không kịp deadline chạy đến hạn chót, dừng sạch ở ranh giới chunk và bảo toàn segment để chạy tiếp lần sau.
* `--dry-run`: Chạy thử nghiệm mô phỏng toàn bộ tiến trình lập lịch hoặc dịch phụ đề mà không suy luận GPU hay gọi API thật.
* `--benchmark-mode`: Bật chế độ đo đạc thuần túy.
* `-b`, `--bitrate`: Bitrate video 4K (Mặc định: `28M` hoặc `35M`).
* `-s`, `--segment`: Độ dài phân đoạn cuốn chiếu (Mặc định: `8` giây).
* `-f`, `--format`: Định dạng khung hình trung gian (`png` lossless BT.709 hoặc `jpg`).
* `--no-ramdisk`: Buộc dùng thư mục tạm trên ổ cứng nếu không tạo RAM Disk ảo.
* `--no-resume`: Buộc chạy lại từ đầu thay vì tiếp tục phân đoạn cũ.
* `--no-caffeinate`: Không kích hoạt cơ chế chống sleep máy trong khi chạy.
* `--preview [giây]`: Chế độ xem trước N giây ở giữa phim và mở trình phát xem ngay.

---

## 📊 4. Số Liệu Đo Đạc Hiệu Năng Thực Tế (NVIDIA RTX 3060 12GB)

| Thông số kỹ thuật | Cấu hình Tiêu chuẩn (Standalone) | Cấu hình Tối ưu hóa (Dedup + Pipelined) |
| :--- | :--- | :--- |
| **Phần cứng** | Intel Core i5-12600K, NVIDIA RTX 3060 12GB | Intel Core i5-12600K, NVIDIA RTX 3060 12GB |
| **Engine suy luận** | PyTorch CUDA FP16 (`batch_size=1`) | PyTorch CUDA FP16 Batch Inference (`batch_size=6`) |
| **Tốc độ AI Inference** | **1.37 fps** (RealESR-AnimeVideoV3) | **4.8 – 5.2 fps** thuần GPU \| **~10 – 12+ fps** hiệu dụng (Dedup) |
| **Encoder phần cứng** | `hevc_nvenc` Main 10 (Preset `p6`, CQ 19) | `hevc_nvenc` Main 10 (`p6`, Multipass `qres`, Lookahead `24`, `p010le`) |
| **VRAM tiêu thụ** | ~2.4 GB / 12 GB VRAM | **~6.20 GB** / 12 GB VRAM (51.6% capacity - Đạt đỉnh Tensor Cores, an toàn) |
| **I/O Khung hình** | PIL Image.save (~45 ms/frame) | OpenCV C++ `cv2.imwrite` AVX2 (~14 ms/frame, nhanh hơn 3x) |
| **Output Sanity Guard** | $\|\Delta Y\| \approx 2.80 / 255$, Correlation $\approx 0.9998$ (PASS) | $\|\Delta Y\| \approx 2.80 / 255$, Correlation $\approx 0.9998$ (PASS) |
| **Mux Audit** | Khớp 100% audio, subtitle, font, frame count (PASS) | Khớp 100% audio, subtitle, font, frame count (PASS) |
| **Anti-Sleep** | Win32 `SetThreadExecutionState` | Win32 `SetThreadExecutionState` |

---

## 🌐 5. Dịch Phụ Đề Anime Bằng AI (`subtrans/` & `trisub_ai/`)

Phân hệ dịch phụ đề anime chuyên sâu bằng mô hình ngôn ngữ lớn (Gemini Cloud API):

```powershell
# Chạy trên Windows (sử dụng launcher tiện ích):
.\subtrans.bat check                   # Kiểm tra sức khỏe kết nối & 3 API keys
.\subtrans.bat models                  # Liệt kê danh sách Gemini models
.\subtrans.bat samples/sample_sword_3min.mkv -o out/
```

* **100% Cloud API:** Chạy bằng Gemini Flash-Lite (`gemini-3.5-flash-lite`, 500 RPD/key) / Flash (`gemini-3.8-flash`) với Schema "Suy luận trước, dịch sau".
* **Đồng bộ UTF-8:** Hiển thị mượt mà trên PowerShell / CMD, không lỗi charmap.
* **Cơ chế Whisper STT Fallback:** Tự động trích xuất âm thanh và chạy Whisper để tạo phụ đề `.ja.srt` nếu video gốc thiếu track tiếng Nhật.
* **Auto-Align Subtitle Timestamps:** Tự động ghim mốc thời gian của Whisper / tiếng Nhật theo track tiếng Anh phòng thu (`--auto-align-en`).
* **Luân phiên Round-Robin 3 Key:** Phân bổ đều request giữa 3 API keys để không bao giờ bị nghẽn RPM.
* **Đồng nhất xưng hô:** Stage A Cast Sheet (`out/<name>.cast.yaml`) nạp vô điều kiện vào mọi batch; giữ nguyên tên Romaji khi gọi tên.
* **Linter V1 & Dual Dialogue:** Tự động gộp câu nói đồng thời, loại bỏ flicker, kiểm tra tốc độ đọc CPS và chống rò rỉ CJK.

---

## 📁 6. Cấu Trúc Thư Mục Chuẩn Hóa

```text
ai-video-upscaler/
├── upscale.py                  # CLI Entrypoint cho AI Video Upscaler 4K
├── upscale.bat                 # Windows batch launcher (One-click chạy trên Windows)
├── subtrans.py                 # CLI Entrypoint cho Anime Subtitle Translator
├── subtrans.bat                # Windows batch launcher cho Subtrans (UTF-8, multi-key)
├── shows.yaml                  # Cấu hình model SR theo từng bộ anime
├── pytest.ini                  # Cấu hình Pytest
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
├── README.md                   # Tài liệu hướng dẫn sử dụng
├── AGENTS.md                   # Kim chỉ nam kiến trúc & bất biến kỹ thuật
├── src/                        # Core Engine: AI Video Upscaling (PyTorch CUDA, NVENC 10-bit)
│   ├── config.py               # Quản lý cấu hình tập trung, phần cứng & RAM Disk
│   ├── ai_engine.py            # PyTorch Super-Resolution Inference Engine
│   ├── subtitles.py            # Trích xuất phụ đề rời/container, Whisper STT fallback
│   ├── ffmpeg_ops.py           # FFmpeg / FFprobe chuẩn tham chiếu (NVENC 10-bit)
│   ├── preflight.py            # Chống sleep (Win32 SetThreadExecutionState), kiểm tra sạc & đĩa trống
│   ├── pipeline.py             # Bộ điều phối Rolling Segments Pipeline + Checkpoint Resume
│   ├── dedup.py                # Deduplication Engine (mpdecimate, DIS Optical Flow)
│   ├── mux_audit.py            # Stream copy integrity audit & font count verifier
│   ├── output_sanity.py        # Output Sanity Guard kiểm định độ sáng & tương quan đầu ra
│   ├── ramdisk.py              # Quản lý RAM Disk ảo (Windows ImDisk / Rolling Cleanup)
│   ├── scheduler.py            # Bộ lập lịch hàng đợi qua đêm & preflight nền
│   ├── cleanup.py              # Dọn dẹp thư mục tạm, symlinks & gom file audit
│   └── logger.py               # Telemetry logging & thanh tiến trình Rich
├── subtrans/                   # Core Engine: Anime Subtitle Translation (Google GenAI Gemini)
│   ├── cli.py                  # CLI Typer cho subtrans
│   ├── pipeline.py             # Pipeline dịch phụ đề 4 giai đoạn
│   ├── parser.py               # Stage 1: Parse ASS/SRT, Auto-Align Whisper timestamps, Auto-Offset
│   ├── cast_sheet.py           # Stage A: Khai thác bảng nhân vật & danh xưng
│   ├── gemini_backend.py       # Gemini API client (Round-Robin 3 key, Retry, Backoff)
│   ├── cascade.py              # Stage P2: Smart Cloud Cascade Repair pass
│   ├── rag.py                  # RAG few-shot retrieval & cautions
│   ├── linter.py               # Bộ quy tắc kiểm định CPS, CJK leakage, ngoặc kép
│   ├── writer.py               # Xuất phụ đề SRT/ASS chuẩn single-line
│   ├── report.py               # Báo cáo kiểm định chất lượng (QA Report .audit.md/.json)
│   └── tests/                  # Unit tests riêng của phân hệ dịch phụ đề
├── tests/                      # Automated Test Suite (100% Pytest Unit & Integration Tests)
├── benchmarks/                 # Kịch bản đo lường hiệu năng & Soak Tests
│   ├── upscale/                # Benchmarks hiệu năng GPU, VRAM, tốc độ suy luận, Dedup
│   ├── subtrans/               # Đánh giá độ ổn định xưng hô (Golden Pronoun Evaluations)
│   └── research/               # Kịch bản hiệu chuẩn và nghiên cứu chuyên sâu
├── scripts/                    # Các tiện ích vận hành (organize subs, subtrans batch utilities)
├── glossaries/                 # Từ điển thuật ngữ riêng theo từng series anime
├── models/                     # Danh mục đăng ký mô hình Super-Resolution (models.yaml)
├── models_bench/               # Trọng số mô hình AI PyTorch (.pth)
├── docs/                       # Tài liệu kỹ thuật chuyên sâu & engineering logs
├── samples/                    # Video mẫu 3 phút phục vụ kiểm thử (Mushoku, Slime, Sword)
├── logs/                       # Nhật ký telemetry CSV và báo cáo phiên chạy đêm
└── out/                        # Thư mục xuất video 4K, phụ đề hoàn thiện và báo cáo audit
```
