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

### Cách sử dụng trên Terminal:

```bash
# 1. Chạy nhanh bằng shell launcher (mặc định phân đoạn 15s + RAM Disk):
./upscale.sh samples/sample_30s.mkv

# 2. Hoặc chạy qua Python CLI với đầy đủ tùy chọn:
python3 upscale.py samples/sample_30s.mkv -b 28M -s 15 -o ~/Documents/Upscaled/output_4K.mkv
```

**Các tham số tùy chọn:**
* `-o`, `--output`: Đường dẫn file hoặc thư mục xuất (Mặc định: `~/Documents/Upscaled/<tên>_4K.mkv`).
* `-m`, `--model`: Mô hình AI (`realesr-animevideov3`, `realesrgan-x4plus-anime`, `realesrgan-x4plus`).
* `-b`, `--bitrate`: Bitrate video 4K (Mặc định: `28M`).
* `-s`, `--segment`: Độ dài phân đoạn cuốn chiếu (Mặc định: `15` giây - tối ưu vừa vặn trong RAM Disk).
* `-f`, `--format`: Định dạng khung hình trung gian (`jpg` chất lượng cao hoặc `png` lossless).
* `--no-ramdisk`: Buộc dùng thư mục tạm trên SSD nếu máy thiếu RAM.
* `--no-resume`: Buộc chạy lại từ đầu thay vì tiếp tục phân đoạn cũ.

---

## 📊 2. Số Liệu Đo Đạc Thực Tế Trên Apple Silicon M3 Pro

* **Tốc độ suy luận AI (1080p ➔ 4K):** ~**1.1 – 1.2 fps** trên GPU Metal / MoltenVK (đáp ứng đúng giới hạn phần cứng tính toán TFLOPS thực tế của M3 Pro).
* **Thời gian xử lý video 30 giây (~720 frames):** ~10 – 11 phút.
* **Thời gian xử lý tập phim 24 phút (~34.000 frames):** ~7.5 – 8 giờ (chạy qua đêm với Checkpoint Resume an toàn).
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
│   ├── pipeline.py             # Bộ điều phối Rolling Segments Pipeline + Checkpoint Resume
│   └── logger.py               # Hiển thị thanh tiến trình & Báo cáo Benchmark thực tế
├── samples/
│   ├── sample_30s.mkv          # File mẫu WEB-DL 1080p Crunchyroll
│   └── README.md
├── trisub_ai/                  # Module dịch phụ đề anime bằng AI
└── requirements.txt            # Dependencies tối giản
```
