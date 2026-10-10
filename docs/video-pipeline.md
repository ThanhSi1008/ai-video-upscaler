# VIDEO-PIPELINE.MD — Hướng Dẫn Kỹ Thuật Phân Hệ Phóng To Video AI & Metal GPU Engine

> **Chuyên đề kiến trúc phân hệ xử lý video siêu phân giải (Super-Resolution Pipeline).**  
> **Tham chiếu chính:** [AGENTS.md](file:///Users/xis108/Desktop/upscale/AGENTS.md) | **Đo đạc thực nghiệm:** [docs/engineering-log.md](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md)

---

## 1. Trách Nhiệm Module & Các File Mã Nguồn Cốt Lõi

| File mã nguồn | Vai trò & Trách nhiệm chính |
| :--- | :--- |
| [`src/pipeline.py`](file:///Users/xis108/Desktop/upscale/src/pipeline.py) | Orchestrator điều phối toàn bộ luồng: gối đầu 3 tầng (triple-buffering), rolling segments, checkpoint resume |
| [`src/mps_engine.py`](file:///Users/xis108/Desktop/upscale/src/mps_engine.py) | Engine suy luận chính: PyTorch MPS FP16 Native Apple Silicon, nạp trọng số PyTorch và chạy inference |
| [`src/ai_engine.py`](file:///Users/xis108/Desktop/upscale/src/ai_engine.py) | Engine fallback: NCNN Vulkan thông qua MoltenVK (Metal API), tối ưu hóa tiling `-t 540 -j 1:2:2` |
| [`src/models_resolver.py`](file:///Users/xis108/Desktop/upscale/src/models_resolver.py) | Bộ phân giải model registry (`models/models.yaml`), xác thực mã băm SHA256 và tải trọng số |
| [`src/ramdisk.py`](file:///Users/xis108/Desktop/upscale/src/ramdisk.py) | Context Manager quản lý RAM Disk ảo HFS+ 4GB (`/Volumes/RAMDisk`), triệt tiêu 100% hao mòn ổ cứng SSD |
| [`src/ffmpeg_ops.py`](file:///Users/xis108/Desktop/upscale/src/ffmpeg_ops.py) | Thao tác FFmpeg/FFprobe chuẩn phòng thu: ma trận màu BT.709 10-bit, VideoToolbox HEVC, Stream Copy Mux |
| [`src/config.py`](file:///Users/xis108/Desktop/upscale/src/config.py) | Single Source of Truth: định nghĩa hằng số độ dài chunk 8s, bitrate 4K, dung lượng RAM Disk |

---

## 2. Kiến Trúc Điều Phối Gối Đầu 3 Tầng (Triple-Buffering)

Hệ thống loại bỏ hoàn toàn thời gian chết của GPU thông qua mô hình chạy song song 3 tầng trong [`src/pipeline.py`](file:///Users/xis108/Desktop/upscale/src/pipeline.py):

```text
Thời gian ──────►
Chunk N-1: [Media Engine HEVC Encode (CPU/AME)] ──► Lưu temp HEVC segment trên SSD (.temp/)
Chunk N:   [PyTorch MPS FP16 AI Inference (GPU)] ──► Đọc PNG 1080p, xuất PNG 4K trên RAM Disk
Chunk N+1: [FFmpeg Trích xuất Frame (CPU)]       ──► Giải mã GOP từ source video ra RAM Disk
```

* **Hiệu quả thực nghiệm:** Tỷ lệ hoạt động GPU (GPU Active) đạt **90.9% – 93.2%** (thời gian GPU rảnh rỗi chỉ còn ~7–9%). Giảm 35.4% tổng thời gian chạy so với pipeline tuần tự cũ, tiết kiệm ~59 phút trên mỗi tập anime 24 phút.
* **Bộ nhớ RAM Disk đỉnh:** Chỉ lưu trữ tối đa 3 chunk khung hình song song (đỉnh sử dụng ~2.7 GB trên tổng 4.0 GB của RAM Disk ảo).

---

## 3. Engine Suy Luận: PyTorch MPS FP16 Native & Fallback NCNN

### A. PyTorch MPS FP16 Native ([`src/mps_engine.py`](file:///Users/xis108/Desktop/upscale/src/mps_engine.py))
* **Mô hình chính:** `realesr-animevideov3` (kiến trúc mạng `SRVGGNetCompact` 4x kết hợp nội suy Bicubic 0.5 để xuất 2x).
* **Đặc tính kỹ thuật:**
  - Chạy native trên Apple Silicon Metal Performance Shaders (MPS) với định dạng nửa độ chính xác **FP16** (`torch.float16`).
  - Tốc độ đo đạc: **1.86 fps warm-up**, **1.68 fps sustained** xuyên suốt soak test 6.000 frame liên tục.
  - Bộ nhớ MPS cấp phát cực kỳ ổn định ở mức **~61 MB**, process RSS ~36 MB, triệt tiêu 100% rò rỉ bộ nhớ (memory leak).
  - Độ trung thực hình ảnh so với bản FP32 chuẩn: PSNR đạt **61.74 dB** (0 NaN / 0 Inf).
* **Đính chính mô hình AnimeJaNai (`AnimeJaNaiCompact2x`):**
  - Đã khắc phục triệt để lỗi thiếu phép cộng phần dư nearest $\times 2$ tại tag `janai-fix`:
    ```python
    residual = F.interpolate(x, scale_factor=2.0, mode='nearest')
    out = self.body(x) + residual
    ```
  - Khớp bit-exact 0.0 với spandrel CPU chuẩn (PSNR 75–158 dB, $|\Delta Y| \le 0.0114/255$).

### B. Fallback NCNN Vulkan ([`src/ai_engine.py`](file:///Users/xis108/Desktop/upscale/src/ai_engine.py))
* Đóng vai trò dự phòng thông qua Mach-O binary `./bin/realesrgan/realesrgan-ncnn-vulkan` liên kết MoltenVK.
* Tham số tiling tối ưu hóa chuẩn phòng thu: `-t 540 -j 1:2:2` (tốc độ 1.417 fps, nhanh hơn 18.9% so với auto-tiling `-t 0`).

---

## 4. Bảo Vệ NVMe SSD Qua RAM Disk Ảo HFS+ ([`src/ramdisk.py`](file:///Users/xis108/Desktop/upscale/src/ramdisk.py))

* **Nguyên tắc bất biến:** Tuyệt đối không ghi khung hình trung gian dạng PNG (cả trích xuất 1080p và xuất 4K AI) lên ổ cứng SSD.
* **Cơ chế vận hành:**
  1. Tự động tạo RAM Disk 4GB trên bộ nhớ hợp nhất (Unified Memory) qua lệnh `hdiutil attach -nomount ram://8388608` và định dạng HFS+ với cờ `noatime`.
  2. Gắn điểm gắn kết tại `/Volumes/RAMDisk`.
  3. Sau khi mỗi phân đoạn (chunk 8s) được VideoToolbox mã hóa thành file segment HEVC (`up_*.mp4`), toàn bộ frame PNG trung gian của phân đoạn đó được xóa sạch ngay lập tức trên RAM Disk.
  4. Vòng đời được bảo vệ bằng `try...finally` kết hợp `atexit` và bắt tín hiệu SIGINT (`Ctrl+C`), tự động gỡ unmount sạch sẽ khi tiến trình kết thúc.

---

## 5. Chuỗi Ma Trận Màu Sắc & Mã Hóa VideoToolbox ([`src/ffmpeg_ops.py`](file:///Users/xis108/Desktop/upscale/src/ffmpeg_ops.py))

### A. Chuỗi Chuyển Đổi Màu BT.709 Hai Chiều (Two-Way Matrix)
Để triệt tiêu hiện tượng lệch màu (color shift) và bợt màu đen khi chuyển đổi qua lại giữa YUV và RGB:
* **Trích xuất 1080p ra PNG (YUV Limited ➔ RGB24 Full):**
  `-vf scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc`
* **Mã hóa 4K từ PNG (RGB24 Full ➔ YUV Limited 10-bit `p010le`):**
  `-vf scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv,format=p010le`
* **Gắn metadata màu sắc chuẩn xác:**
  `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`

### B. Tham Số Mã Hóa Phần Cứng Apple Media Engine
* Codec: `hevc_videotoolbox` (mã hóa HEVC phần cứng trên Apple Silicon).
* Bitrate mặc định: `35M` (tối ưu cho 4K anime 60fps/24fps, cân bằng độ nét chi tiết và dung lượng).
* Profile: Main 10 (`p010le`), tương thích tối đa với Apple TV 4K, QuickTime Player, IINA và TV OLED.

---

## 6. Muxing & Stream Copy Nguyên Bản

Lệnh mux ghép nối cuối cùng ([`src/ffmpeg_ops.py:mux_final_video`](file:///Users/xis108/Desktop/upscale/src/ffmpeg_ops.py#L320)) bắt buộc tuân thủ:
```bash
ffmpeg -y -i "up_concat.mp4" -i "source.mkv" \
  -map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 \
  -c copy -max_interleave_delta 0 "output.mkv"
```
* **Bảo toàn Bit-Exact:** Toàn bộ âm thanh đa kênh (FLAC/AAC/TrueHD), phụ đề mềm (ASS/SRT), font chữ đính kèm (TTF/OTF) và mục lục phân đoạn (Chapters) được copy nguyên bản 100%.
* **Bắt buộc dùng `-max_interleave_delta 0`:** Ngăn chặn hiện tượng tràn buffer xen kẽ (interleaving buffer overflow) gây mất tiếng hoặc lệch đồng bộ âm thanh - hình ảnh sau vài phút phát lại.
