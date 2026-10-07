# AI Video Upscaler 4K & AI Subtitle Translator (Apple Silicon M3 Pro)

Công cụ tối ưu hóa chuyên sâu cho macOS để **Upscale Video Anime lên 4K (2x Super-Resolution)** và **Dịch phụ đề Anime bằng AI (TriSub AI)** chạy hoàn toàn cục bộ trên Terminal.

---

## ⚡ 1. Công Cụ Upscale Video Anime 4K (`upscale.py` / `upscale.sh`)

Sử dụng trực tiếp nhân **Apple Silicon GPU Metal / Vulkan** kết hợp **Apple Media Engine (VideoToolbox)** để đạt chất lượng chuẩn tham chiếu phòng thu:

* **Mô hình AI:** `realesr-animevideov3` chuyên dụng cho video anime 2x (khử nhiễu nén, giữ nét tự nhiên).
* **Mã hóa phần cứng 10-bit:** `hevc_videotoolbox` Main 10 Profile (`p010le`, BT.709 Limited Range TV, Tag `hvc1`, Spatial AQ).
* **Bảo vệ tuyệt đối NVMe SSD:** Sử dụng **RAM Disk ảo HFS+ trên macOS** để đệm khung hình trung gian. Khung hình thô và 4K chỉ sinh ra trong RAM và xóa ngay lập tức khi chunk được encode xong ➔ **0 byte ảnh tĩnh ghi xuống SSD!**
* **Bảo toàn nguyên bản dữ liệu gốc & Đồng bộ âm thanh:** Tự động **Stream Copy** toàn bộ audio đa kênh, phụ đề mềm đa ngôn ngữ (ASS/SRT), font chữ nhúng (TrueType) và chapters từ file gốc với cờ `-max_interleave_delta 0` ép đan xen đều đặn từng packet theo DTS (loại bỏ hoàn toàn lỗi audio dồn khối gây mất tiếng sau vài phút phát; không dùng file audio tạm).
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
* `-m`, `--model`: Mô hình AI (Mặc định: tự động chọn qua `shows.yaml` hoặc mặc định `animevideov3`).
* `--engine`: Động cơ suy luận (`mps` cho PyTorch MPS FP16 hoặc `ncnn` cho NCNN Vulkan, mặc định: `mps`).
* `--precision`: Độ chính xác suy luận (`fp16` hoặc `fp32`, mặc định: `fp16`).
* `--list-models`: Liệt kê các mô hình đã đăng ký trong `models/models.yaml`.
* `--force`: Bắt buộc chạy và ghi đè/xóa phân đoạn cũ nếu phát hiện xung đột mô hình.
* `--subs-after-upscale`: Tự động chạy dịch phụ đề sau khi toàn bộ hàng đợi upscale hoàn tất (Mặc định: Bật).
* `--no-subs`: Tắt giai đoạn dịch phụ đề sau upscale.
* `--queue`: Danh sách nhiều tập video cần chạy tuần tự trong ca qua đêm.
* `--deadline HH:MM`: Mốc hạn chót dừng hàng đợi an toàn nếu tập kế tiếp dự kiến không kịp.
* `--allow-partial`: Cho phép tập không kịp deadline chạy đến hạn chót, dừng sạch ở ranh giới chunk và bảo toàn segment để chạy tiếp lần sau.
* `--dry-run`: Chạy thử nghiệm mô phỏng toàn bộ tiến trình lập lịch hoặc dịch phụ đề mà không suy luận GPU hay gọi API thật.
* `--benchmark-mode`: Bật chế độ đo đạc thuần túy (Scheduler và Preflight nền tự động từ chối chạy để tránh làm bẩn dữ liệu).
* `--auto-offset`: Tự động nhận diện và áp dụng bù lệch thời gian JA-EN (Mặc định: Bật trong hàng đợi, Tắt khi chạy 1 tập đơn).
* `--no-auto-offset`: Tắt tính năng tự động bù lệch thời gian phụ đề trong hàng đợi.
* `--sub-offset-ms [ms]`: Bù lệch thời gian phụ đề thủ công theo mili-giây (ví dụ `-11154`). Trong hàng đợi, cờ này chỉ áp dụng khi có `--sub-offset-target`.
* `--sub-offset-target [tên]`: Chỉ định tên tập cụ thể áp dụng offset thủ công trong hàng đợi.
* `--no-embed`: Không nhúng track phụ đề vào container MKV 4K, chỉ xuất và giữ file .vi.srt ngoài.
* `--night-report [phiên]`: Sinh báo cáo tổng kết phiên chạy đêm từ log, telemetry CSV, audit.json và DB (Chế độ chỉ đọc, không gọi GPU/API).
* `-b`, `--bitrate`: Bitrate video 4K (Mặc định: `28M`).
* `-s`, `--segment`: Độ dài phân đoạn cuốn chiếu (Mặc định: `8` giây - tối ưu vừa vặn trong RAM Disk).
* `-f`, `--format`: Định dạng khung hình trung gian (`png` lossless BT.709 hoặc `jpg`).
* `--no-ramdisk`: Buộc dùng thư mục tạm trên SSD nếu máy thiếu RAM.
* `--no-resume`: Buộc chạy lại từ đầu thay vì tiếp tục phân đoạn cũ.
* `--no-caffeinate`: Không tự động kích hoạt caffeinate để chống sleep máy trong khi chạy.
* `--preview [giây]`: Chế độ xem trước N giây ở giữa phim, sau đó tự mở bằng IINA.

---

## 📊 2. Số Liệu Đo Đạc Thực Tế Trên Apple Silicon M3 Pro

* **Tốc độ suy luận AI (1080p ➔ 4K):** Đạt ổn định **1.68 – 1.70 fps** trên GPU Metal với PyTorch MPS FP16 (`realesr-animevideov3`). *(Lưu ý: Số đo 1.79 fps trước đây của AnimeJaNai Compact bị vô hiệu do lỗi lớp forward thiếu phép cộng residual dẫn đến video đen; đã được khắc phục tại tag `janai-fix` kèm chốt chặn đầu ra Output Sanity Guard).*
* **Soak Test 6.000 frame liên tục (1 giờ):** Median **1.683 fps**, P5 **1.677 fps**, Đỉnh driver VRAM **2.090,8 MB**, Footprint RSS **674,4 MB**, Pageouts: **+133**.
* **Thời gian xử lý video 3 phút (~4.360 frames, 2.186 unique frames):** Hoàn tất toàn bộ trong **23 phút 59 giây** (Pipelined Double-Buffering trên MPS), nhanh hơn bản tuần tự 13 phút 10 giây (-35,4%).
* **Sức chứa ca chạy qua đêm (Overnight Batch):** Sức chứa thực tế là **~2 tập mỗi đêm** (mỗi tập 24 phút cần ~3 giờ thực tế ở mức $n_{unique}$ trung bình ~17.200 frames kết hợp overhead video ~15.5m và phụ đề ~5m). Hàng đợi 3 tập chỉ hoàn tất trọn vẹn trong ca 7.5–8 giờ khi tổng $n_{unique}$ của 3 tập đủ nhỏ (ví dụ các tập nhiều cảnh tĩnh với $n_{unique} \le 13.000$). Với các tập chưa đo preflight, hệ thống tự động áp dụng ước lượng thận trọng 60.0% kept frames ($n=4$) để đảm bảo an toàn hạn chót.
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
