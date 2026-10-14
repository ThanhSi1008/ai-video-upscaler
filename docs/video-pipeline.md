# VIDEO-PIPELINE.MD — Hướng Dẫn Kỹ Thuật Phân Hệ Phóng To Video AI & GPU Engine

> **Chuyên đề kiến trúc phân hệ xử lý video siêu phân giải (Super-Resolution Pipeline).**  
> **Tham chiếu chính:** [AGENTS.md](../AGENTS.md) | **Đo đạc thực nghiệm:** [engineering-log.md](engineering-log.md)

---

## 1. Trách Nhiệm Module & Các File Mã Nguồn Cốt Lõi

| File mã nguồn | Vai trò & Trách nhiệm chính |
| :--- | :--- |
| `src/pipeline.py` | Orchestrator điều phối toàn bộ luồng: gối đầu 3 tầng (triple-buffering), rolling segments, checkpoint resume |
| `src/mps_engine.py` | Multi-platform PyTorch Engine: Hỗ trợ CUDA FP16 (NVIDIA), MPS FP16 (Apple Silicon) và CPU |
| `src/ai_engine.py` | Engine fallback: NCNN Vulkan thông qua MoltenVK (Metal API trên Mac), tối ưu hóa tiling `-t 540 -j 1:2:2` |
| `src/models_resolver.py` | Bộ phân giải model registry (`models/models.yaml`), xác thực mã băm SHA256 và tải trọng số |
| `src/ramdisk.py` | Quản lý bộ đệm khung hình: RAM Disk ảo HFS+ 4GB trên macOS; cơ chế rolling cleanup trên Windows |
| `src/ffmpeg_ops.py` | Thao tác FFmpeg/FFprobe chuẩn phòng thu: ma trận màu BT.709 10-bit, NVENC/VideoToolbox HEVC, Stream Copy Mux |
| `src/config.py` | Single Source of Truth: định nghĩa hằng số độ dài chunk 8s, bitrate 4K, tự động phát hiện hardware/encoder |

---

## 2. Kiến Trúc Điều Phối Gối Đầu 3 Tầng (Triple-Buffering)

Hệ thống loại bỏ hoàn toàn thời gian chết của GPU thông qua mô hình chạy song song 3 tầng trong `src/pipeline.py`:

```text
Thời gian ──────►
Chunk N-1: [Phần cứng HEVC Encode (NVENC / VideoToolbox)] ──► Lưu temp HEVC segment trên SSD (.temp/)
Chunk N:   [PyTorch AI Inference (CUDA / MPS FP16)]        ──► Đọc PNG 1080p, xuất PNG 4K
Chunk N+1: [FFmpeg Trích xuất Frame (CPU)]                 ──► Giải mã GOP từ source video ra thư mục đệm
```

* **Tỷ lệ hoạt động GPU:** Duy trì ở mức cao nhờ pipeline gối đầu.
* **Bộ nhớ đệm đỉnh:** Chỉ lưu trữ tối đa 3 chunk khung hình song song, tự động dọn dẹp cuốn chiếu ngay khi mã hóa xong.

---

## 3. Engine Suy Luận: PyTorch CUDA FP16 & MPS FP16 Native

### A. PyTorch CUDA FP16 Native (`src/mps_engine.py` trên Windows / NVIDIA RTX)
* **Phần cứng mục tiêu:** NVIDIA GeForce RTX 3060 (12GB VRAM GDDR6) và các dòng RTX Ampere/Ada.
* **Mô hình chính:** `realesr-animevideov3` (kiến trúc mạng `SRVGGNetCompact` 4x kết hợp nội suy Bicubic 0.5 để xuất 2x).
* **Đặc tính kỹ thuật:**
  - Chạy native trên CUDA với định dạng nửa độ chính xác **FP16** (`torch.float16`).
  - Hỗ trợ **Batch Inference (`batch_size=6`)**: Tận dụng tối đa băng thông Tensor Cores của kiến trúc Ampere, nâng tốc độ suy luận thuần GPU từ 2.2 fps (single frame) lên **4.8 – 5.2 fps** (và 10–12+ fps hiệu dụng khi kết hợp Dedup Preflight).
  - VRAM chiếm dụng ổn định ở mức **~6.2 GB / 12 GB** (an toàn tuyệt đối, không gây OOM).
  - Sử dụng **OpenCV `cv2.imwrite` (tối ưu hóa C++ AVX2 của Core i5-12600K)** để ghi khung hình trung gian, nhanh hơn PIL từ 2x đến 3x.
  - Đồng bộ phần cứng chuẩn xác: Gọi `torch.cuda.synchronize()` đảm bảo tiến trình inference hoàn tất trọn vẹn trước khi ghi dữ liệu.
  - Tương quan đầu ra Output Sanity: $r \ge 0.9996$, $|\Delta Y| \approx 1.84/255$ (PASS tuyệt đối).

### B. PyTorch MPS FP16 Native (`src/mps_engine.py` trên Apple Silicon)
* **Phần cứng mục tiêu:** Apple Silicon (M1/M2/M3/M4 Series).
* **Đặc tính kỹ thuật:**
  - Chạy native trên Apple Silicon Metal Performance Shaders (MPS) với định dạng FP16.
  - Tốc độ đo đạc: **1.86 fps warm-up**, **1.68 fps sustained** xuyên suốt soak test 6.000 frame liên tục.
  - Bộ nhớ MPS cấp phát cực kỳ ổn định ở mức **~61 MB**, process RSS ~36 MB, triệt tiêu 100% rò rỉ bộ nhớ.
  - Độ trung thực hình ảnh so với bản FP32 chuẩn: PSNR đạt **61.74 dB**.

### C. Đính chính mô hình AnimeJaNai (`AnimeJaNaiCompact2x`)
* Đã khắc phục triệt để lỗi thiếu phép cộng phần dư nearest $\times 2$ tại tag `janai-fix`:
  ```python
  residual = F.interpolate(x, scale_factor=2.0, mode='nearest')
  out = self.body(x) + residual
  ```
* Khớp bit-exact 0.0 với spandrel CPU chuẩn (PSNR 75–158 dB, $|\Delta Y| \le 0.0114/255$).

---

## 4. Bảo Vệ Bộ Nhớ & Lưu Trữ Đệm Khung Hình (`src/ramdisk.py`)

* **Trên macOS:**
  1. Tự động tạo RAM Disk 4GB trên bộ nhớ hợp nhất (Unified Memory) qua lệnh `hdiutil attach -nomount ram://8388608` và định dạng HFS+ với cờ `noatime`.
  2. Gắn điểm gắn kết tại `/Volumes/RAMDisk`.
  3. Xóa sạch frame PNG trung gian của từng chunk 8s ngay khi VideoToolbox mã hóa xong.
* **Trên Windows:**
  1. Áp dụng cơ chế Rolling Segments Cleanup: Khung hình 1080p và 4K của từng chunk 8s được dọn sạch ngay lập tức khi NVENC encode xong chunk đó.
  2. Không để tích tụ hàng nghìn file PNG trên ổ cứng SSD, giảm thiểu 95%+ hao mòn ghi đĩa.

---

## 5. Chuỗi Ma Trận Màu Sắc & Mã Hóa Phần Cứng 10-bit (`src/ffmpeg_ops.py`)

### A. Chuỗi Chuyển Đổi Màu BT.709 Hai Chiều (Two-Way Matrix)
Để triệt tiêu hiện tượng lệch màu (color shift) và bợt màu đen khi chuyển đổi qua lại giữa YUV và RGB:
* **Trích xuất 1080p ra PNG (YUV Limited ➔ RGB24 Full):**
  `-vf scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc`
* **Mã hóa 4K từ PNG (RGB24 Full ➔ YUV Limited 10-bit `p010le`):**
  `-vf scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv,format=p010le`
* **Gắn metadata màu sắc chuẩn xác:**
  `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`

### B. Tham Số Mã Hóa Phần Cứng Theo Nền Tảng
* **Windows (NVIDIA NVENC):**
  - Codec: `hevc_nvenc`
  - Profile: Main 10 (`p010le`)
  - Preset: `p6` (chất lượng cao)
  - Kiến trúc Ampere: Multipass `qres` (Full resolution two-pass rate control), Lookahead `24` frames
  - Flags: `-spatial-aq 1 -b:v 28M` (hoặc `-cq 19`)
* **macOS (Apple Media Engine):**
  - Codec: `hevc_videotoolbox`
  - Profile: Main 10 (`p010le`)
  - Bitrate mặc định: `35M`
  - Flags: Spatial AQ enabled

---

## 6. Muxing & Stream Copy Nguyên Bản

Lệnh mux ghép nối cuối cùng (`src/ffmpeg_ops.py:mux_final_video`) bắt buộc tuân thủ:
```bash
ffmpeg -y -i "up_concat.mp4" -i "source.mkv" \
  -map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 \
  -c copy -max_interleave_delta 0 "output.mkv"
```
* **Bảo toàn Bit-Exact:** Toàn bộ âm thanh đa kênh (FLAC/AAC/TrueHD), phụ đề mềm (ASS/SRT), font chữ đính kèm (TTF/OTF) và mục lục phân đoạn (Chapters) được copy nguyên bản 100%.
* **Bắt buộc dùng `-max_interleave_delta 0`:** Ngăn chặn hiện tượng tràn buffer xen kẽ (interleaving buffer overflow) gây mất tiếng hoặc lệch đồng bộ âm thanh - hình ảnh sau vài phút phát lại.

