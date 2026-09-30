# 📋 KẾ HOẠCH KỸ THUẬT TOÀN DIỆN: NÂNG CẤP HỆ THỐNG AI VIDEO UPSCALER SANG REAL-CUGAN PRO NATIVE 2x (4K ANIME)

---

## 1. ĐÁNH GIÁ HIỆN TRẠNG MÃ NGUỒN HIỆN TẠI (CODEBASE AUDIT & GAP ANALYSIS)

Sau khi đọc và phân tích toàn bộ mã nguồn (`upscale.py`, `app.py`, `check_native.py`, `Dockerfile`, `requirements.txt`, `README.md`), hệ thống hiện tại có các ưu điểm và lỗ hổng kỹ thuật sau:

### 1.1. Các ưu điểm sẵn có (Giữ nguyên và kế thừa)
* **Kiến trúc Multi-Processing Dual GPU (`_gpu_segment_worker`):** Đã chia video thành 2 phân đoạn độc lập và xử lý song song trên `cuda:0` và `cuda:1` bằng `torch.multiprocessing` (spawn mode).
* **Streaming qua RAM Pipe không ghi đĩa:** Giải mã và nén trực tiếp qua `stdin`/`stdout` pipe của FFmpeg, tránh tạo hàng vạn file ảnh tạm trên ổ cứng.
* **Hỗ trợ phần cứng đa nền tảng:** Tự động nhận diện NVIDIA CUDA, Apple Silicon MPS (`hevc_videotoolbox`), và CPU `libx264`.

### 1.2. Các hạn chế kỹ thuật nghiêm trọng cần khắc phục
1. **Thiếu mô hình Native 2x (Nguyên nhân cốt lõi gây mất nét):**
   * Trong `upscale.py` chỉ có `SRVGGNetCompact` (AnimeVideoV3) và `RRDBNet` (Real-ESRGAN x4plus 6B). **Cả hai đều là mô hình 4x!**
   * Khi upscale 1080p lên 4K, code hiện tại (dòng 273-275 và 713-715) phóng đại lên 4x rồi dùng:
     ```python
     output = F.interpolate(output, size=(target_h, target_w), mode='bilinear', align_corners=False)
     ```
     $\rightarrow$ Việc nội suy `bilinear` từ 8K nén về 4K làm **mờ nét vẽ (blur)**, phá hủy hoàn toàn vi chi tiết mà AI dày công tái tạo!
2. **Không gian màu 8-bit gây Color Banding trên 4K:**
   * Trong thiết lập NVENC (dòng 206, 208, 651), code đang ép định dạng `-pix_fmt yuv420p` (8-bit).
   * Khi chiếu video anime 4K trên màn hình lớn, chuẩn 8-bit lộ rõ các vệt nứt màu (banding) ở vùng tối và bầu trời.
3. **Mất phụ đề mềm (Subtitle) và Font chữ khi Remux:**
   * Lệnh ghép âm thanh hiện tại (dòng 571-576 và 797-802) chỉ copy audio:
     ```python
     mux_cmd = ['ffmpeg', '-y', '-i', temp_v, '-i', video_input, '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a?', video_output]
     ```
     $\rightarrow$ Khi đưa file SubsPlease `.mkv` vào, **toàn bộ phụ đề `.ass` và fonts đính kèm bị xóa sổ** khỏi video đầu ra!
4. **Chưa hỗ trợ tải Magnet Link trực tiếp trên giao diện/CLI:**
   * Chỉ hỗ trợ upload file và link YouTube qua `yt-dlp`. Với file torrent/magnet SubsPlease, người dùng phải tải thủ công.

---

## 2. MỤC TIÊU KỸ THUẬT (CORE OBJECTIVES)

1. **Tích hợp Real-CUGAN Pro Native 2x:**
   * Bổ sung định nghĩa mạng Cascaded U-Net (CUNet 2x) thuần túy bằng PyTorch vào `upscale.py`.
   * Hỗ trợ tự động tải trọng số chính thức:
     * **`up2x-pro-conservative.pth`** (Mặc định tối ưu cho nguồn SubsPlease / Web-DL).
     * **`up2x-pro-no-denoise.pth`** (Dành cho nguồn Blu-ray Remux sạch).
     * **`up2x-pro-denoise3x.pth`** (Dành cho video cũ có nhiễu nặng).
2. **Chuẩn hóa tỷ lệ nội suy Native 2x (1080p $\rightarrow$ 3840x2160):**
   * Khi đầu vào là 1080p và đầu ra là 4K UHD ($2\times$), mô hình xuất trực tiếp tensor $(3840, 2160)$, **bỏ qua hoàn toàn bước resize `bilinear`**.
3. **Nâng cấp chuẩn nén NVENC sang HEVC 10-bit (`yuv420p10le`):**
   * Ép FFmpeg NVENC xuất ra `-pix_fmt yuv420p10le -profile:v main10 -cq 17` để khử triệt để hiện tượng nứt dải màu.
4. **Bảo tồn $100\%$ Subtitle (.ass), Audio (FLAC/AAC) và Attachments (Fonts):**
   * Remux đầy đủ tất cả stream: video 4K mới + audio gốc + phụ đề gốc + font đính kèm vào container `.mkv`.
5. **Tích hợp công cụ tải Magnet Link siêu tốc với `aria2c`:**
   * Bổ sung tiện ích dòng lệnh và tab UI để tải trực tiếp 1 tập SubsPlease qua Magnet chỉ trong 30 giây trên Kaggle.
6. **Bảo đảm tương thích tuyệt đối:**
   * Vận hành mượt mà trên **Kaggle Dual NVIDIA T4 (FP16)** và **MacBook Pro M3 Pro (`Mac15,6`)**.

---

## 3. LỘ TRÌNH TRIỂN KHAI CHI TIẾT (5 GIAI ĐOẠN)

### 🔹 Giai đoạn 1: Hiện thực hóa cấu trúc mạng Real-CUGAN 2x trong `upscale.py`
* **Công việc cụ thể:**
  1. Viết class kiến trúc `CUNet2x` (Cascaded U-Net) bằng PyTorch chuẩn:
     * Gồm 2 tầng U-Net tuần tự với các khối trích xuất đặc trưng `UNetConv`, skip connections và tầng mở rộng pixel `PixelShuffle(2)`.
  2. Xây dựng hàm `load_realcugan_model(model_variant, device)`:
     * Tự động nhận diện checkpoint (.pth) từ kho chính thức của Bilibili AI Lab.
     * Tự động tải weights về thư mục gốc nếu chưa có.
     * Cấu hình chế độ inference `model.half().eval()`, bật `channels_last` và hỗ trợ `torch.compile` khi chạy trên CUDA.

### 🔹 Giai đoạn 2: Cập nhật quy trình suy luận Native 2x & Quản lý bộ nhớ
* **Công việc cụ thể:**
  1. Điều chỉnh hàm `_gpu_segment_worker` và luồng GPU đơn:
     * Kiểm tra hệ số scale: Nếu mô hình là Real-CUGAN 2x và mục tiêu là 4K từ 1080p, bypass hoàn toàn hàm `F.interpolate`.
     * Tối ưu VRAM: Real-CUGAN Pro 2x trên 1080p chiếm ~4.2 GB VRAM $\rightarrow$ Đặt `batch_size = 1` hoặc `2` để giữ VRAM an toàn dưới 8 GB trên mỗi T4 (vốn có 15 GB).
  2. Tích hợp bộ lọc làm nét vi mô Laplacian 5x5 và Color Boost tương thích với Real-CUGAN.

### 🔹 Giai đoạn 3: Nâng cấp luồng mã hóa FFmpeg & Bảo toàn Metadata (Muxing)
* **Công việc cụ thể:**
  1. **Nâng cấp NVENC/VideoToolbox sang 10-bit:**
     * Với NVIDIA GPU: `-c:v hevc_nvenc -preset p7 -tune hq -cq 17 -pix_fmt yuv420p10le -profile:v main10`.
     * Với Apple Silicon M3 Pro: `-c:v hevc_videotoolbox -q:v 65 -pix_fmt yuv420p10le`.
  2. **Cập nhật lệnh ghép Remux cuối cùng:**
     * Trích xuất nguyên vẹn: `-map 0:v -map 1:a? -map 1:s? -map 1:t? -c:v copy -c:a copy -c:s copy -c:t copy`.
     * Đổi container đầu ra mặc định thành `.mkv` (vì container MKV hỗ trợ tốt nhất cho phụ đề ASS và font chữ anime).

### 🔹 Giai đoạn 4: Cập nhật giao diện Gradio (`app.py`) & Bổ sung công cụ Magnet
* **Công việc cụ thể:**
  1. Thêm các lựa chọn mô hình Real-CUGAN vào `MODEL_MAP`:
     * `Real-CUGAN Pro 2x (Khuyên Dùng cho SubsPlease Web-DL - Cực Nét Vector)` $\rightarrow$ `realcugan_pro_conservative`
     * `Real-CUGAN Pro 2x (Bản Không Khử Nhiễu - Cho Blu-ray Remux)` $\rightarrow$ `realcugan_pro_no_denoise`
     * `Real-CUGAN Pro 2x (Khử Nhiễu Mạnh Denoise3x - Cho Video Cũ)` $\rightarrow$ `realcugan_pro_denoise3x`
     * `AnimeVideoV3 (Siêu Tốc 25+ FPS)`
     * `Real-ESRGAN x4Plus Anime 6B (4x RRDBNet)`
  2. Bổ sung chức năng nạp link Magnet:
     * Viết hàm hỗ trợ gọi `aria2c` tải nhanh 1 tập anime vào thư mục `/kaggle/working/input`.
     * Thêm ô nhập Magnet Link trên giao diện Gradio cạnh ô YouTube URL.

### 🔹 Giai đoạn 5: Cập nhật Dockerfile, Requirements & Hướng dẫn Kaggle
* **Công việc cụ thể:**
  1. Bổ sung `aria2` vào `Dockerfile` và script hướng dẫn chạy trên Kaggle.
  2. Viết tài liệu hướng dẫn quy trình 1 tập 30 phút trên Kaggle Notebook trong `README.md`.

---

## 4. KẾ HOẠCH KIỂM THỬ & TIÊU CHÍ CHẤP NHẬN (ACCEPTANCE CRITERIA)

| Hạng mục kiểm tra | Phương pháp kiểm thử | Tiêu chuẩn thành công |
| :--- | :--- | :--- |
| **Kiểm tra tải Weights** | Chạy khởi động mô hình Real-CUGAN Pro | Tự động tải `up2x-pro-conservative.pth` (~60MB) thành công, MD5 hash hợp lệ. |
| **VRAM Footprint trên GPU** | Chạy 1 đoạn video 1080p sample trên GPU | VRAM không vượt quá **6.0 GB / 15 GB** trên mỗi card T4. Không xảy ra CUDA OOM. |
| **Bảo tồn Phụ đề & Font** | Dùng file sample MKV có softsub `.ass` và font | File 4K đầu ra mở trên VLC/MPV hiển thị đầy đủ sub tiếng Việt/Anh chuẩn kiểu dáng, font chữ không bị lỗi. |
| **Kiểm tra dải màu 10-bit** | Dùng `ffprobe` soi file thành phẩm | Stream video hiển thị: `pix_fmt: yuv420p10le`, `profile: Main 10`, không còn vệt nứt màu (banding). |
| **Tốc độ Dual T4** | Đo thông lượng trên Kaggle | Tốc độ kết hợp đạt từ **14 đến 18 FPS** (1 tập 24 phút hoàn thành trong 35–38 phút). |
| **Tương thích Mac15,6** | Chạy thử nghiệm trên MacBook Pro M3 Pro | Xử lý được qua MPS/CPU không báo lỗi syntax hay thiếu thư viện CUDA. |

---

## 5. KẾT LUẬN & SẴN SÀNG THỰC THI

Kế hoạch này giải quyết triệt để vấn đề mất nét do downscale của mô hình cũ, đưa hệ thống lên chuẩn **Master Quality Real-CUGAN Pro Native 2x**, tối ưu 100% cho nhu cầu xem anime Web-DL SubsPlease của bạn.
