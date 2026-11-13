# VIDEO-PIPELINE.MD — Hướng Dẫn Kỹ Thuật Phân Hệ Phóng To Video AI & GPU Engine

> **Chuyên đề kiến trúc phân hệ xử lý video siêu phân giải (Super-Resolution Pipeline).**  
> **Tham chiếu chính:** [AGENTS.md](../AGENTS.md) | **Đo đạc thực nghiệm:** [engineering-log.md](engineering-log.md)

---

## 1. Trách Nhiệm Module & Các File Mã Nguồn Cốt Lõi

| File mã nguồn | Vai trò & Trách nhiệm chính |
| :--- | :--- |
| [`src/pipeline.py`](../src/pipeline.py) | Orchestrator điều phối toàn bộ luồng: gối đầu 3 tầng thực sự (`Decode N+1` $\parallel$ `GPU N` $\parallel$ `Encode N-1` / `worker_finalize_zero_disk`), 1-Pass Concat-Mux, checkpoint resume |
| [`src/build_trt_engine.py`](../src/build_trt_engine.py) | Trình biên dịch & thực thi **TensorRT 10.16 FP16** (`builder_optimization_level=5`, `execute_async_v3`, vòng đệm VRAM `_out_ring` 2-slot khử cấp phát động, cache shape/address) |
| [`src/trt_zero_copy_upscaler.py`](../src/trt_zero_copy_upscaler.py) | Zero-Disk Pipe Engine: Truyền tuần tự `NHWC` qua Pinned Memory, transpose `uint8`, Cyclic Cel Memoization (Belady's OPT), kernel **Triton JIT** `p010le` & thu mẫu Sanity Batch 0 |
| [`src/cuda_engine.py`](../src/cuda_engine.py) | PyTorch GPU Engine: `RealESRAnimeVideoV3` (cắt lát đuôi 8K 2-frame), `AnimeJaNaiCompact2x`, `batch_size=8`, tự động nạp TensorRT hoặc fallback CUDA FP16 `channels_last` |
| [`src/models_resolver.py`](../src/models_resolver.py) | Bộ phân giải model registry (`models/models.yaml`, `shows.yaml`), xác thực mã băm SHA-256 |
| [`src/ramdisk.py`](../src/ramdisk.py) | Quản lý bộ đệm khung hình: Ưu tiên RAM Disk vật lý (`UPSCALER_RAMDISK_DIR` / ImDisk), tiếp đến định tuyến tự động sang NVMe Gen4 `scratch/rolling_buffer` và xóa cuốn chiếu |
| [`src/ffmpeg_ops.py`](../src/ffmpeg_ops.py) | Thao tác FFmpeg/FFprobe chuẩn phòng thu: `extract_frames_raw_rgb24` (`readinto` 0-copy), ma trận màu BT.709 10-bit, NVENC HEVC, `mux_segments_directly` (1-Pass Concat-Mux) |
| [`src/config.py`](../src/config.py) | Single Source of Truth: cấu hình `HMMA.F16` (`allow_fp16_reduced_precision_reduction`), `CUDA_MODULE_LOADING=LAZY`, tham số NVENC `CQ 18` `fullres` và khóa PID `.gpu_active.lock` |

---

## 2. Kiến Trúc Điều Phối Gối Đầu 3 Tầng (True 3-Stage Overlap)

Hệ thống loại bỏ hoàn toàn thời gian chết của GPU thông qua mô hình chạy song song 3 tầng trong [`src/pipeline.py`](../src/pipeline.py) cho cả hai chế độ vận hành:

### A. Chế Độ Turbo Zero-Disk (`--zero-disk` / `upscale_turbo.bat`)
```text
Thời gian ──────►
Chunk N-1: [EncodeWorker: Xả 32-frame lookahead NVENC & đóng gói MP4] ──► up_{N-1}.mp4
Chunk N:   [GPU: H2D NHWC -> TensorRT 10 FP16 -> Triton p010le Pipe]  ──► Bơm thẳng vào stdin NVENC
Chunk N+1: [DecodeWorker: NVDEC giải mã GOP -> readinto RGB24 RAM]    ──► Prefetch sẵn trên RAM 32GB (0 PNG đĩa)
```

### B. Chế Độ Chuẩn Đệm Cuốn Chiếu (Standard Rolling Buffer — `upscale.bat`)
```text
Thời gian ──────►
Chunk N-1: [EncodeWorker: Concat Demuxer + NVENC HEVC 10-bit]         ──► Xuất up_{N-1}.mp4 & xóa sạch PNG
Chunk N:   [GPU: TensorRT 10 / CUDA FP16 + NTFS Hardlink Cyclic Cel]  ──► Đọc PNG 1080p, xuất PNG 4K
Chunk N+1: [DecodeWorker: FFmpeg trích xuất PNG 1080p đã lọc Dedup]   ──► Prefetch sớm khi đệm trống >= 1600 MB
```

---

## 3. Engine Suy Luận: TensorRT 10 FP16 & PyTorch CUDA FP16

### A. TensorRT 10.16 FP16 & PyTorch Hybrid (`src/build_trt_engine.py`, `src/cuda_engine.py`)
* **Phần cứng mục tiêu:** NVIDIA GeForce RTX 3060 12GB (GA106 Ampere, mở khóa xung VRAM **P0 `7500 MHz`** & **HAGS `HwSchMode=2`** qua `unlock_rtx3060_admin.bat`) + Intel Core i5-12600K (khóa luồng trên P-Cores qua `configure_alder_lake_pcore_qos`).
* **Mô hình chính:** `realesr-animevideov3` (`SRVGGNetCompact` 4x kết hợp nội suy Area/Bicubic 0.5 để xuất 2x).
* **Đặc tính kỹ thuật cốt lõi:**
  - **TensorRT 10 FP16 (`HMMA.16816.F16`):** Tự động xuất ONNX và biên dịch `.engine` với `builder_optimization_level=5` + `TimingCache`, gộp `Conv2d + PReLU + PixelShuffle` mở khóa đỉnh **~51 TFLOPS** nhân Tensor (gấp đôi mặc định FP32 accumulate).
  - **Zero-Alloc VRAM Ring (`_out_ring`):** Tái sử dụng vòng đệm đôi 2-slot trên VRAM (~199 MB/slot ở `batch_size=8`), khử hoàn toàn ~2.100 lần gọi `torch.empty` mỗi tập và lưu vết cache cho `set_input_shape` / `set_tensor_address`.
  - **Vi phân lô đuôi 8K (`2-frame micro-slicing`):** Ở tầng `PixelShuffle(4)` trung gian (`7680x4320`), tự động cắt lát mỗi 2 khung hình trước khi hạ mẫu về 4K, giảm 75% đỉnh bộ nhớ đệm 8K.
  - **Không lượng tử hóa `INT8 PTQ`:** Bảo toàn trọn vẹn 65.536 mức biểu diễn của `FP16` qua 33 lớp `PReLU`, ngăn chặn triệt để hiện tượng vân bậc thang màu (`color banding`) trên dải xuất 10-bit.

### B. Đính chính mô hình AnimeJaNai (`AnimeJaNaiCompact2x`)
* Khắc phục triệt để lỗi thiếu phép cộng phần dư nearest $\times 2$ tại tag `janai-fix`:
  ```python
  residual = F.interpolate(x, scale_factor=2.0, mode="nearest")
  out = self.body(x) + residual
  ```
* Khớp bit-exact với spandrel CPU chuẩn (PSNR 75–158 dB, $|\Delta Y| \le 0.0114/255$).

---

## 4. Bảo Vệ Bộ Nhớ & Lưu Trữ Đệm Khung Hình (`src/ramdisk.py`)

* **Thứ tự ưu tiên định tuyến bộ đệm tự động:**
  1. **Zero-Disk Pipe (`--zero-disk`):** Triệt tiêu 100% file ảnh trung gian trên ổ đĩa.
  2. **RAM Disk vật lý:** Tự động phát hiện ổ RAM thật qua `UPSCALER_RAMDISK_DIR` hoặc ổ `R:`/`Z:` mang nhãn `"RAM"` (ImDisk / OSFMount).
  3. **NVMe PCIe Gen4 tốc độ cao:** Định tuyến về `scratch/rolling_buffer` trên ổ workspace NVMe Gen4 (`WD_BLACK SN7100 ~7.25 GB/s`), tránh nghẽn `%TEMP%` trên ổ hệ thống Gen3 và xóa cuốn chiếu ngay sau mỗi chunk 8s.

---

## 5. Chuỗi Ma Trận Màu Sắc & Mã Hóa Phần Cứng 10-bit (`src/ffmpeg_ops.py`, `src/config.py`)

### A. Chuỗi Chuyển Đổi Màu BT.709 Hai Chiều (Two-Way Matrix)
* **Trích xuất 1080p (YUV Limited -> RGB24 Full):**
  `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc`
* **Mã hóa 4K (RGB24 Full -> YUV Limited 10-bit `p010le`):**
  - Chế độ Zero-Disk: Kernel **Triton JIT** / `addcmul_` thực hiện ma trận ITU-R BT.709 10-bit TV Range (`Y: 64..940`, `UV: 64..960`, dịch trái `<< 6`, `-chroma_sample_location center`) trực tiếp trên VRAM.
  - Chế độ Standard: `-vf scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv` với `-pix_fmt p010le`.
* **Metadata màu sắc bắt buộc:**
  `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`

### B. Tham Số Mã Hóa Phần Cứng NVIDIA NVENC (`get_hevc_rate_args`)
* **Codec & Profile:** `hevc_nvenc`, `main10` (`p010le`), Preset `p6`, Tune `hq`.
* **Kiểm soát chất lượng thông minh cho Anime:** VBR `CQ 18` (`-cq 18 -b:v 0 -maxrate 45M -bufsize 90M`), Multipass `fullres`, Lookahead `32` frames, B-frames `4` (`-b_ref_mode middle`).
* **Adaptive Quantization kép:** `-spatial-aq 1 -aq-strength 6` (bảo vệ nét vẽ đen sắc mảnh) kết hợp `-temporal-aq 1` (khóa chất lượng nền tĩnh anime, giảm 35–45% dung lượng cảnh tĩnh).

---

## 6. Ghép Nối 1 Lượt (1-Pass Concat-Mux) & Stream Copy Nguyên Bản

Hàm [`src/ffmpeg_ops.py:mux_segments_directly`](../src/ffmpeg_ops.py) ghép trực tiếp danh sách phân đoạn `up_*.mp4` với container nguồn chỉ trong **1 lượt I/O duy nhất** (loại bỏ hoàn toàn việc ghi/đọc ~8 GB file trung gian `merged_4k.mp4`):
```bash
ffmpeg -y -f concat -safe 0 -i "concat_4k_list.txt" -i "source.mkv" \
  -map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 \
  -c copy -max_interleave_delta 0 "output_4K.mkv"
```
* **Bảo toàn Bit-Exact:** Toàn bộ âm thanh đa kênh (FLAC/AAC/TrueHD), phụ đề mềm (ASS/SRT), font chữ đính kèm (TTF/OTF) và Chapters được copy nguyên bản 100% với cờ bắt buộc `-max_interleave_delta 0`.
