# AI Video Upscaler 4K & AI Subtitle Translator (Apple Silicon M3 Pro)

Công cụ tối ưu hóa chuyên sâu cho macOS để **Upscale Video Anime lên 4K (2x Super-Resolution)** và **Dịch phụ đề Anime bằng AI (TriSub AI)** chạy hoàn toàn cục bộ trên Terminal.

---

## ⚡ 1. Công Cụ Upscale Video Anime 4K (`upscale.py` / `upscale.sh`)

Sử dụng trực tiếp nhân **Apple Silicon GPU Metal / Vulkan** kết hợp **Apple Media Engine (VideoToolbox)** để đạt chất lượng chuẩn tham chiếu phòng thu:

* **Mô hình AI:** `realesr-animevideov3` chuyên dụng cho video anime 2x (khử nhiễu nén, giữ nét tự nhiên).
* **Mã hóa phần cứng 10-bit:** `hevc_videotoolbox` Main 10 Profile (`p010le`, BT.709 Limited Range TV, Tag `hvc1`, Spatial AQ).
* **Bảo vệ tuyệt đối NVMe SSD:** Sử dụng **RAM Disk ảo HFS+ trên macOS** để đệm khung hình trung gian. Khung hình thô và 4K chỉ sinh ra trong RAM và xóa ngay lập tức khi chunk được encode xong ➔ **0 byte ảnh tĩnh ghi xuống SSD!**
* **Bảo toàn 100% dữ liệu gốc:** Tự động **Stream Copy** toàn bộ audio đa kênh, phụ đề mềm đa ngôn ngữ (ASS/SRT), font chữ nhúng (TrueType) và chapters từ file gốc.
* **Checkpoint Resume:** Hỗ trợ tiếp tục chạy ngay từ phân đoạn đang dở nếu bị tạm dừng (Ctrl+C hoặc tắt máy).
* **Cơ chế chống ngủ máy (caffeinate):** Tự động gọi `caffeinate -dims` gắn chặt với PID của phiên chạy; tự động dọn dẹp khi kết thúc hoặc bị gián đoạn.
* **Kiểm tra an toàn trước khi chạy (Pre-flight Checks):** Cảnh báo sạc AC vs Pin (chống sập nguồn giữa chừng), kiểm tra dung lượng ổ cứng trống tối thiểu an toàn.
* **Thông báo macOS:** Tự động bắn thông báo Notification Center và chuông âm thanh (`Glass`) ngay khi hoàn tất.

### Cách sử dụng trên Terminal:

```bash
# 1. Chạy nhanh bằng shell launcher (mặc định phân đoạn 8s + RAM Disk + Caffeinate):
./upscale.sh samples/sample_3min.mkv

# 2. Xem trước nhanh 10 giây (tự động cắt tại 30% phim, upscale và mở bằng IINA):
python3 upscale.py samples/sample_3min.mkv --preview 10

# 3. Chạy đầy đủ với tùy biến bitrate và đường dẫn xuất:
python3 upscale.py samples/sample_3min.mkv -b 28M -s 8 -o ~/Documents/Upscaled/output_4K.mkv
```

**Các tham số tùy chọn:**
* `-o`, `--output`: Đường dẫn file hoặc thư mục xuất (Mặc định: `~/Documents/Upscaled/<tên>_4K.mkv`).
* `-m`, `--model`: Mô hình AI (`realesr-animevideov3`, `realesrgan-x4plus-anime`, `realesrgan-x4plus`).
* `-b`, `--bitrate`: Bitrate video 4K (Mặc định: `28M`).
* `-s`, `--segment`: Độ dài phân đoạn cuốn chiếu (Mặc định: `8` giây - tối ưu vừa vặn trong RAM Disk).
* `-f`, `--format`: Định dạng khung hình trung gian (`png` lossless BT.709 hoặc `jpg`).
* `--no-ramdisk`: Buộc dùng thư mục tạm trên SSD nếu máy thiếu RAM.
* `--no-resume`: Buộc chạy lại từ đầu thay vì tiếp tục phân đoạn cũ.
* `--no-caffeinate`: Không tự động kích hoạt caffeinate để chống sleep máy trong khi chạy.
* `--preview [giây]`: Chế độ xem trước N giây ở giữa phim, sau đó tự mở bằng IINA.

---

## 📊 2. Số Liệu Đo Đạc Thực Tế Trên Apple Silicon M3 Pro

* **Tốc độ suy luận AI (1080p ➔ 4K):** ~**1.1 – 1.2 fps** trên GPU Metal / MoltenVK (đáp ứng đúng giới hạn phần cứng tính toán TFLOPS thực tế của M3 Pro).
* **Thời gian xử lý video 30 giây (~720 frames):** ~10 – 11 phút.
* **Thời gian xử lý video 3 phút (~4.360 frames):** ~60 – 65 phút.
* **Thời gian xử lý tập phim 24 phút (~34.000 frames):** ~7.5 – 8 giờ (chạy qua đêm với Checkpoint Resume & Caffeinate an toàn).
* **Mức độ tiêu thụ ổ cứng:** **0 GB ghi frame trung gian vào SSD** khi bật RAM Disk ảo.

---

## 🌐 3. Dịch Phụ Đề Anime Bằng AI (`trisub_ai/`)

Thư mục `trisub_ai/` chứa module dịch phụ đề anime chuyên sâu bằng LLM (Gemini / Claude / OpenAI):

```bash
cd trisub_ai
python3 tri_sub_translate.py --help
```

---

## 📁 4. Cấu Trúc Thư Mục Chuẩn Hóa

```text
upscale/
├── upscale.sh                  # Launcher tiện lợi chạy nhanh trên Terminal
├── upscale.py                  # CLI Entrypoint với đầy đủ tùy chọn
├── bin/
│   └── realesrgan/             # Binary ncnn-vulkan Mach-O & Models NCNN
├── src/
│   ├── config.py               # Thiết lập cấu hình, hằng số, encoder params
│   ├── ramdisk.py              # Quản lý RAM Disk ảo (Zero NVMe wear, Auto Eject)
│   ├── ffmpeg_ops.py           # FFmpeg / FFprobe chuẩn tham chiếu (10-bit BT.709, Stream Copy)
│   ├── ai_engine.py            # Quản lý suy luận NCNN Vulkan trên GPU M3 Pro
│   ├── preflight.py            # Kiểm tra nguồn điện, đĩa trống & Caffeinate chống ngủ
│   ├── pipeline.py             # Bộ điều phối Rolling Segments Pipeline + Checkpoint Resume
│   └── logger.py               # Hiển thị thanh tiến trình & Báo cáo Benchmark thực tế
├── samples/
│   ├── sample_30s.mkv          # Clip mẫu 32s (Crunchyroll WEB-DL 1080p)
│   ├── sample_3min.mkv         # Clip mẫu 3 phút (Mushoku Tensei S3 Ep 14 climax)
│   └── README.md
├── logs/                       # Log có cấu trúc chi tiết từng phiên chạy
├── trisub_ai/                  # Module dịch phụ đề anime bằng AI
└── requirements.txt            # Dependencies tối giản
```
