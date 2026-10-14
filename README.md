# AI Video Upscaler 4K & AI Subtitle Translator (Cross-Platform: Windows RTX & Apple Silicon)

Công cụ tối ưu hóa chuyên sâu để **Upscale Video Anime lên 4K (2x Super-Resolution)** và **Dịch phụ đề Anime bằng AI (TriSub AI)** chạy hoàn toàn cục bộ trên Terminal / PowerShell.

Hỗ trợ tối ưu hóa phần cứng trên cả 2 nền tảng:
* **Windows 11 + NVIDIA GeForce RTX:** Tận dụng nhân **NVIDIA CUDA Cores / Tensor Cores (PyTorch CUDA FP16)** kết hợp **NVIDIA NVENC (HEVC Main 10 Profile)**.
* **macOS (Apple Silicon M-Series):** Tận dụng nhân **Apple Silicon GPU Metal (PyTorch MPS FP16)** kết hợp **Apple Media Engine (VideoToolbox)** và RAM Disk ảo chống hao mòn SSD.

---

## ⚡ 1. Công Cụ Upscale Video Anime 4K (`upscale.py` / `upscale.sh`)

Sử dụng trực tiếp nhân GPU chuyên dụng kết hợp bộ mã hóa phần cứng để đạt chất lượng chuẩn tham chiếu phòng thu:

* **Mô hình AI:** `realesr-animevideov3` chuyên dụng cho video anime 2x (khử nhiễu nén, giữ nét vẽ tự nhiên, không bị hiệu ứng viền sáp).
* **Mã hóa phần cứng 10-bit:**
  * **Windows (NVIDIA):** `hevc_nvenc` Main 10 Profile (`p010le`, Preset `p5`, Spatial AQ, Tag `hvc1`, BT.709 Limited Range TV).
  * **macOS (Apple Silicon):** `hevc_videotoolbox` Main 10 Profile (`p010le`, Spatial AQ, Tag `hvc1`, BT.709 Limited Range TV).
* **Quản lý bộ nhớ đệm khung hình:**
  * **macOS:** Sử dụng **RAM Disk ảo HFS+** đệm khung hình trung gian (0 byte ảnh tĩnh ghi xuống SSD).
  * **Windows:** Tự động đệm và dọn sạch phân đoạn cuốn chiếu (Rolling Segments Cleanup) ngay sau khi từng chunk mã hóa xong, hoặc hỗ trợ RAM Disk ảo (ImDisk).
* **Bảo toàn nguyên bản dữ liệu gốc & Đồng bộ âm thanh:** Tự động **Stream Copy** toàn bộ audio đa kênh (FLAC/AAC/TrueHD), phụ đề mềm đa ngôn ngữ (ASS/SRT), font chữ nhúng (TrueType/OpenType) và chapters từ file gốc với cờ `-max_interleave_delta 0` ép đan xen đều đặn từng packet theo DTS (loại bỏ hoàn toàn lỗi audio dồn khối gây mất tiếng; không dùng file audio tạm).
* **Checkpoint Resume:** Hỗ trợ tiếp tục chạy ngay từ phân đoạn đang dở nếu bị tạm dừng (Ctrl+C hoặc tắt máy).
* **Cơ chế chống ngủ máy (Anti-Sleep):**
  * **Windows:** Sử dụng Win32 API native `SetThreadExecutionState` (`ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED`).
  * **macOS:** Sử dụng `caffeinate -dims` gắn chặt với PID của phiên chạy.
* **Kiểm tra an toàn trước khi chạy (Pre-flight Checks):** Cảnh báo sạc AC vs Pin (chống sập nguồn giữa chừng), kiểm tra dung lượng ổ cứng trống tối thiểu an toàn.
* **Output Sanity Guard & Mux Audit:** Tự động đo độ lệch sáng $|\Delta Y| \le 6/255$ và tương quan Pearson $r \ge 0.98$ sau chunk 1 để triệt tiêu nguy cơ video đen/hỏng, đồng thời đối soát bit-exact frame count, audio tracks và fonts trước khi bàn giao.

---

## 💻 2. Yêu Cầu Hệ Thống & Cài Đặt

### A. Cấu Hình Windows (NVIDIA RTX)
* **Hệ điều hành:** Windows 10/11 64-bit.
* **GPU:** NVIDIA GeForce RTX 3060 (12GB VRAM khuyến nghị) hoặc dòng RTX 20/30/40/50 series với Driver NVIDIA mới nhất.
* **CPU & RAM:** Intel Core i5/i7/i9 hoặc AMD Ryzen (khuyến nghị $\ge 16\text{ GB}$ RAM).
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

### B. Cấu Hình macOS (Apple Silicon)
* **Phần cứng:** Apple Silicon Mac (M1/M2/M3/M4 Series). Khuyến nghị $\ge 16\text{ GB}$ Unified Memory.
* **Công cụ bổ trợ:** `brew install ffmpeg`.

**Cài đặt trên macOS (Terminal):**
```bash
git clone https://github.com/ThanhSi1008/ai-video-upscaler.git
cd ai-video-upscaler
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 3. Hướng Dẫn Sử Dụng CLI (`upscale.py`)

### Lệnh Vận Hành Trên Windows (PowerShell):
```powershell
# 1. Xem trước nhanh 10 giây (tự động cắt tại 30% phim, upscale và mở bằng trình phát mặc định):
python upscale.py samples/sample_sword_3min.mkv --preview 10

# 2. Chạy đầy đủ 1 tập video với bitrate 28M và phân đoạn 8 giây:
python upscale.py samples/sample_sword_3min.mkv -b 28M -s 8 -o out/sword_4K.mkv

# 3. Chạy hàng đợi nhiều tập qua đêm:
python upscale.py --queue ep1.mkv ep2.mkv -o out/
```

### Lệnh Vận Hành Trên macOS (Terminal):
```bash
# 1. Chạy nhanh bằng shell launcher (mặc định phân đoạn 8s + RAM Disk + Caffeinate):
./upscale.sh samples/sample_3min.mkv

# 2. Xem trước nhanh 10 giây:
python3 upscale.py samples/sample_3min.mkv --preview 10

# 3. Chạy đầy đủ với tùy biến bitrate và đường dẫn xuất:
python3 upscale.py samples/sample_3min.mkv -b 28M -s 8 -o ~/Documents/Upscaled/output_4K.mkv
```

### Bảng Tham Số Tùy Chọn:
* `-o`, `--output`: Đường dẫn file hoặc thư mục xuất (Mặc định: `~/Documents/Upscaled/<tên>_4K.mkv` hoặc thư mục `out/`).
* `-m`, `--model`: Mô hình AI (Mặc định: tự động chọn qua `shows.yaml` hoặc mặc định `animevideov3`).
* `--engine`: Động cơ suy luận (`cuda` cho NVIDIA GPU, `mps` cho Apple Silicon Metal, `ncnn` cho NCNN Vulkan, hoặc `cpu`; mặc định: tự động nhận diện).
* `--precision`: Độ chính xác suy luận (`fp16` hoặc `fp32`, mặc định: `fp16`).
* `--list-models`: Liệt kê các mô hình đã đăng ký trong `models/models.yaml`.
* `--force`: Bắt buộc chạy và ghi đè/xóa phân đoạn cũ nếu phát hiện xung đột mô hình.
* `--subs-after-upscale`: Tự động chạy dịch phụ đề sau khi toàn bộ hàng đợi upscale hoàn tất (Mặc định: Bật).
* `--no-subs`: Tắt giai đoạn dịch phụ đề sau upscale.
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

## 📊 4. Số Liệu Đo Đạc Thực Tế

| Thông số / Nền tảng | NVIDIA GeForce RTX 3060 12GB (Windows 11) | Apple Silicon M3 Pro (macOS) |
| :--- | :--- | :--- |
| **Engine suy luận** | PyTorch CUDA FP16 Native | PyTorch MPS FP16 Native |
| **Tốc độ AI Inference** | **2.2 – 2.5 fps** (Single-frame, chưa batch) | **1.68 – 1.70 fps** |
| **Encoder phần cứng** | `hevc_nvenc` Main 10 (Preset p5, p010le) | `hevc_videotoolbox` Main 10 (p010le) |
| **VRAM tiêu thụ** | **~1.1 GB** / 12 GB VRAM (Dư dồi dào cho batch) | **~2.1 GB** / 18 GB Unified Memory |
| **Output Sanity Guard** | $|\Delta Y| \approx 1.84 / 255$, Correlation $\approx 0.9996$ (PASS) | $|\Delta Y| \le 6.0 / 255$, Correlation $\ge 0.98$ (PASS) |
| **Mux Audit** | Khớp 100% audio, subtitle, frame count (PASS) | Khớp 100% audio, sub, font, frame count (PASS) |
| **Anti-Sleep** | Win32 `SetThreadExecutionState` | macOS `caffeinate -dims` |

---

## 🌐 5. Dịch Phụ Đề Anime Bằng AI (`subtrans/` & `trisub_ai/`)

Phân hệ dịch phụ đề anime chuyên sâu bằng mô hình ngôn ngữ lớn (Gemini Cloud API):

```bash
# Dịch phụ đề độc lập:
python subtrans.py samples/sample_sword_3min.mkv -o out/
```

* **100% Cloud API:** Chạy bằng Gemini Flash-Lite / Flash với Schema "Suy luận trước, dịch sau".
* **Đồng nhất xưng hô:** Stage A Cast Sheet (`out/<name>.cast.yaml`) nạp vô điều kiện vào mọi batch; giữ nguyên tên Romaji khi gọi tên.
* **Linter V1 & Dual Dialogue:** Tự động gộp câu nói đồng thời, loại bỏ flicker, kiểm tra tốc độ đọc CPS và chống rò rỉ CJK.

---

## 📁 6. Cấu Trúc Thư Mục Chuẩn Hóa

```text
ai-video-upscaler/
├── upscale.py                  # CLI Entrypoint với đầy đủ tùy chọn (Windows & macOS)
├── upscale.sh                  # Shell launcher tiện lợi cho macOS/Linux
├── subtrans.py                 # Subtitle Translation CLI Runner
├── shows.yaml                  # Cấu hình model SR theo từng bộ anime
├── pytest.ini                  # Cấu hình Pytest
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
├── README.md                   # Tài liệu hướng dẫn sử dụng
├── AGENTS.md                   # Kim chỉ nam kiến trúc & bất biến kỹ thuật
├── src/                        # Pipeline phóng to video (CUDA/MPS, NVENC/VideoToolbox, FFmpeg)
│   ├── config.py               # Thiết lập cấu hình, auto-detect platform/hardware
│   ├── mps_engine.py           # Multi-platform PyTorch Engine (CUDA FP16 / MPS / CPU)
│   ├── ffmpeg_ops.py           # FFmpeg / FFprobe chuẩn tham chiếu (NVENC/VideoToolbox 10-bit)
│   ├── preflight.py            # Chống sleep (Win32 / Caffeinate), kiểm tra sạc & đĩa trống
│   ├── pipeline.py             # Bộ điều phối Rolling Segments Pipeline + Checkpoint Resume
│   ├── dedup.py                # Deduplication Engine (mpdecimate, DIS Optical Flow)
│   ├── mux_audit.py            # Stream copy integrity audit & font count verifier
│   ├── output_sanity.py        # Output Sanity Guard kiểm định độ sáng & tương quan đầu ra
│   ├── ramdisk.py              # Quản lý RAM Disk ảo (macOS HFS+ / Windows fallback)
│   └── logger.py               # Telemetry logging & thanh tiến trình Rich
├── samples/                    # Video mẫu phục vụ kiểm thử
├── models_bench/               # Trọng số mô hình AI PyTorch (.pth)
├── docs/                       # Tài liệu kỹ thuật chuyên sâu
└── out/                        # Thư mục xuất video thành phẩm
```

