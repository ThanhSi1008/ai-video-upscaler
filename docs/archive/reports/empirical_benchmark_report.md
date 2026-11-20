# Báo Cáo Đo Đạc Thực Nghiệm Benchmarks & Vi Thăm Dò Silicon (Empirical Benchmark & Silicon Probe Report)
## Hệ Thống AI Video Upscaler 4K — NVIDIA GeForce RTX 3060 12GB (`GA106-302-A1`) & Intel Core i5-12600K

> - **Phiên bản tài liệu:** 3.0 (Calibrated R3 Final — Silicon Probe + 3-Regime Live Climax Execution)
> - **Thời điểm thực hiện:** 2026-11-15T12:15:09+07:00 – 2026-11-15T12:29:27+07:00
> - **Môi trường:** 100% Cục bộ trên Windows 11 Pro 64-bit (`Build 26300`)
> - **Trạng thái kiểm định:** **PASS 100%** (Khớp `1478/1478` frames, Bit-Exact `streamhash` MD5 Mux Audit, Output Sanity Guard, `.gpu_active.lock` giải phóng sạch)
> - **Kho dữ liệu gốc:** `out/benchmark_r3_final/` (`silicon_probe.json`, `r3_climax_*_summary.json`, `r3_climax_*.audit.json`) & `scratch/hw_and_model_exact_audit.json`

---

## 1. Môi Trường Phần Cứng & Toolchain Đã Kiểm Toán (Zero-Hallucination Ground Truth)

Toàn bộ cấu hình máy trạm và `.venv` được truy vấn trực tiếp từ WMI/CIM (`Win32_Processor`, `Win32_PhysicalMemory`, `Get-PhysicalDisk`), `nvidia-smi -q`, `torch.cuda.get_device_properties`, và metadata gói Python (xóa bỏ triệt để các thông số ghi nhầm ở bản nháp cũ như `Python 3.10.11`, `TRT 10.7`, `XMP 3200`, `SN770 5150 MB/s`, hay `FLAC 5.1`):

| Hạng mục phần cứng / phần mềm | Thông số thực tế đã kiểm toán (`hw_and_model_exact_audit.json`) |
| :--- | :--- |
| **Hệ điều hành & Bo mạch chủ** | Microsoft Windows 11 Pro 64-bit `10.0.26300` (**Build `26300`**, WDDM) trên **Gigabyte `B760M G P WIFI DDR4`** (LGA1700, PCIe 5.0 CPU root) |
| **CPU** | `12th Gen Intel(R) Core(TM) i5-12600K` (10C/16T: `6P` Golden Cove @ `3.7–4.9 GHz` [`4.5 GHz` all-core] + `4E` Gracemont @ `2.8–3.6 GHz`, `L2 = 9,728 KB` [`9.5 MiB`], `L3 = 20,480 KB` [`20.0 MiB`], AVX2/FMA3/VNNI) |
| **GPU & Bộ đệm L2** | `NVIDIA GeForce RTX 3060` (`GA106-302-A1`, `sm_86`, VBIOS `94.06.2f.00.9a`, Driver `617.42`, `28 SMs`, `3,584 CUDA Cores`, `112` 3rd-Gen Tensor Cores, **`L2 Cache = 2,359,296 B` (`2.25 MiB`)**, **`TGP = 170.0 W`**) |
| **VRAM & Giao tiếp PCIe** | `12,288 MiB` (`12,115 MiB` WDDM khả dụng) GDDR6, `192-bit` @ `7,501 MHz` P0 (`15.0 Gbps` = **`360.0 GB/s`** lý thuyết, **`302.61 GB/s`** STREAM Triad thực tế); PCIe `4.0 x16` (`31.51 GB/s`), **`BAR1 = 256 MiB` (`Resizable BAR`: Tắt)** |
| **Bộ nhớ hệ thống (RAM)** | **`32 GiB` Dual-Channel DDR4** (`G.Skill F4-3200C16-16GVK` 16 GiB + `PNY 16GF2X16QFHH36-135-K` 16 GiB) đang chạy ở **`ConfiguredClockSpeed = 2,133 MT/s` @ `1.20 V` JEDEC (`34.13 GB/s` lý thuyết)** do lắp 2 thanh khác mã không bật XMP |
| **Hệ thống ổ cứng NVMe** | **Disk 0 (`C:` `378.21 GiB` OS + `D:`):** `OCPC MFL-300 512GB` (NVMe PCIe Gen3 x4) <br> **Disk 1 (`E:` `464.67 GiB` Workspace, `.cache`, `scratch/`, `Raw_Sources`):** **`WD_BLACK SN7100 500GB`** (NVMe PCIe Gen4 x4, FW `7615M0WD`, datasheet **`6,800 MB/s` đọc / `5,800 MB/s` ghi**) |
| **Toolchain `.venv` chính xác** | **Python `3.11.9`**, **PyTorch `2.6.0+cu124`** (CUDA `12.4`, cuDNN `90100`), **TensorRT `10.16.1.11`**, **Triton `3.2.0`** (`triton-windows 3.2.0.post21`), **PyAV `18.1.0`**, **`faster-whisper` `1.2.1`** (`ctranslate2 4.9.0`), **`spandrel` `0.4.2`**, **NumPy `2.4.6`**, **OpenCV `5.0.0`** (`5.0.0.93`), **FFmpeg `8.1.2-full_build-www.gyan.dev`** |
| **Video nguồn Canonical** | `E:\Films\Raw_Sources\Fate Apocrypha\[AC] Fate Apocrypha - 22 [BD 1080p Hi10 FLAC][DualAudio][695AB63E].mkv` (`1920x1080` H.264 Hi10P `23.976 fps`; `2x` luồng **FLAC `2.0` Stereo** 24-bit/48kHz ENG + JPN; `2x` ASS; `5` Fonts; `9` chapters cả tập / `1` chapter trên clip 60s Climax) |

---

## 2. Kết Quả Vi Thăm Dò Trần Vật Lý Silicon (`out/benchmark_r3_final/silicon_probe.json`)

Script độc lập `scripts/silicon_roofline_probe.py` đã đo trực tiếp các trần vật lý cô lập của chiếc card `RTX 3060 12GB` (`GA106-302-A1`) trong `41.58s`:

| Phép đo vi thăm dò cô lập (`silicon_probe.json`) | Trần Lý Thuyết GA106 (`1,777 MHz` / `P0`) | Kết Quả Thực Nghiệm Đo Trực Tiếp | Hiệu Suất Silicon & Ý Nghĩa Vật Lý |
| :--- | :--- | :--- | :--- |
| **1. IEEE FP32 GEMM (`4096^3`, `allow_tf32=False`)** | `12.74 TFLOPS` (`12.20 T` @ `1,702 MHz` live) | **`6.702 TFLOPS`** (`20.5058 ms`, `128.05 W`) | **`54.93%`** (Datapath thứ 2 trên `sm_86` chia sẻ lệnh FP32/INT32) |
| **2. Dense TC FP16 (`FP32-Accum` `HMMA.16816.F32`)** | **`25.48 TFLOPS`** (**`27.96 T`** @ `1,950 MHz` live) | **`23.211 TFLOPS`** (`cublasGemmEx 32F`, `5.9214 ms`, `148.44 W`, `cos=1.0`) | **`83.03%`** trần xung thực; `torch.matmul` FP16 luôn gọi `32F` (`23.178–23.344 TFLOPS`) |
| **3. Dense TC FP16 (`FP16-Accum` `HMMA.16816.F16`)** | **`50.95 TFLOPS`** (`49.78 T` @ `1,736.25 MHz` TDP; `101.90 T` chỉ khi 2:4 sparse!) | **`38.844 TFLOPS`** (`cublasGemmEx 16F`, `3.5382 ms`, `158.47 W`, `cos=0.999995`; burst `42.63 T`) | **`78.03%`** trần xung TDP (**`1.6735x`** nhanh hơn `FP32-accum`), bác bỏ hoàn toàn con số `101.9 TFLOPS` cũ |
| **4. Eager cuDNN `Conv2d(64->64, 3x3)` `8x1080p`** | `AI = 288.0 FLOPs/B`, `4.25 GB/batch` DRAM | **`12.140 TFLOPS`** (`100.7421 ms/layer`, `168.13 W`) | Chứng minh TensorRT 10 `NHWC` (`28.27–28.77 TFLOPS`) nhanh gấp **`2.33x–2.37x`** PyTorch cuDNN thuần |
| **5. GDDR6 Bandwidth (`256 MiB` Copy / STREAM Triad)** | `360.0 GB/s` (`192-bit` @ `7,501 MHz` P0) | Copy: **`295.69 GB/s`** (`1.8157 ms`) <br> Triad: **`302.61 GB/s`** (`2.6612 ms`, `97%` `mem_util`) | **`84.06%`** băng thông lý thuyết; xác lập điểm gãy thực nghiệm **`128.36 FLOPs/B`** (`HMMA.F16`) & **`76.70 FLOPs/B`** (`HMMA.F32`) |
| **6. PCIe 4.0 x16 Pinned DMA (`H2D` & `D2H`)** | `31.51 GB/s` PCIe / `34.13 GB/s` DDR4-2133 | H2D (`49.77 MB`): **`2.1822 ms`** (**`22.81 GB/s`**) <br> D2H (`199.07 MB`): **`8.7918 ms`** (**`22.64 GB/s`**) | **`72.38%` / `71.86%`** trần vật lý PCIe 4.0 x16 (`0.273 ms/f` H2D, `1.099 ms/f` D2H) |
| **7. Fused R3 `@triton.jit` Input Norm (`8x1080p`)** | `149.30 MB/batch` (1-pass VRAM) vs `348.36 MB` cũ | **`0.5133 ms/batch`** (`0.0642 ms/f`, **`290.88 GB/s`**) vs `1.6429 ms` PyTorch 2-kernel | Đạt **`80.80%`** đỉnh `360 GB/s`, **nhanh gấp `3.20x`** đường chuẩn hóa PyTorch cũ |
| **8. Output `@triton.jit` BT.709 `rgb_to_p010le` (`8x4K`)**| `597.20 MB/batch` (`RGB FP16 -> P010LE`) | **`5.6683 ms/batch`** (`0.7085 ms/f`, **`105.36 GB/s`**) | Kèm clamp thanh ghi `[0,1]` chống rò rỉ sắc độ Sakuga (`0 LSB` sai lệch) |
| **9. 7th-Gen NVENC 4K Main10 `p6 CQ 18 lookahead 32`** | `hevc_nvenc` raw `stdin` pipe (`0` AI inference) | **`-multipass fullres`:** `18.96 FPS` (120f), **`20.67 FPS`** (240f), **`22.70 FPS` (`44.053 ms/f`) steady!** <br> *1-pass (`disabled`):* **`29.50 FPS`** (120f) | **Phát hiện quyết định:** `-multipass fullres` mã hóa 2 lượt full 4K trên 1 chip NVENC, khóa cứng trần E2E ở **`~21–24.6 FPS`**! |

---

## 3. Các Tối Ưu Hóa Vi Cấu Trúc R3 & Ma Trận Hiệu Năng 3 Chế Độ (3-Regime Matrix)

### 3.1 Ba Tối Ưu Hóa Mới Trong R3 (`src/config.py`, `src/trt_zero_copy_upscaler.py`, `scripts/verify_offline.py`)
1. **Định tuyến Cache GPU/JIT/Triton & Test `TEMP` sang `WD_BLACK SN7100` Gen4 (`E:\ai-video-upscaler\.cache` & `scratch/`):** Cấu hình `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` sang ổ NVMe Gen4 `E:` (`6,800/5,800 MB/s`) và cô lập `TEMP`/`TMP`/`TMPDIR` của `verify_offline.py` khỏi ổ hệ thống `C:` (`OCPC MFL-300` Gen3).
2. **Kernel `@triton.jit` Chuẩn Hóa Đầu Vào Gộp 1 Lượt (`_nhwc_u8_to_nchw_fp16_norm_kernel`):** Chuyển `(B, H, W, 3) uint8` sang `(B, 3, H, W) float16 * (1/255)` trực tiếp vào địa chỉ cố định `self.gpu_in_ring[slot]` trong 1 lượt đọc/ghi VRAM (`149.30 MB/batch` thay vì `348.36 MB`), giảm độ trễ từ `1.6429 ms` xuống **`0.5133 ms/batch` (`3.20x`)** và được warm-up sẵn trước khi bấm giờ.
3. **Hồ Sự Kiện CUDA Tiền Cấp Phát (`_ensure_cuda_event_pools`) & Tách Bạch `avg_h2d_dma_ms` / `avg_in_norm_ms`:** Khử 100% lệnh gọi driver `cudaEventCreate` trong vòng lặp nóng và đặt mốc `evt_h2d_dma_e` ngay sau `gpu_nhwc.copy_(..., non_blocking=True)`, tách riêng thời gian thực thi kernel chuẩn hóa đầu vào (`avg_in_norm_ms = 0.593–0.724 ms/batch`) khỏi thời gian xếp hàng Copy-Engine dưới Windows WDDM `HwSchMode=2`.

### 3.2 Bảng Ma Trận Hiệu Năng Thực Nghiệm Toàn Trình (Baseline $\to$ R2 $\to$ R3 Final)

Tất cả 3 chế độ R3 đều chạy tuần tự trên đoạn Climax `60s @ 13:30` (`810s`, `1478` frames, `1405` unique = `95.06%`, `178` batches / `8` chunks, `--zero-disk`, `--no-subs`), làm nguội GPU xuống **`< 50°C` (`42°C–46°C`)** trước mỗi lượt chạy:

| Chỉ số hiệu năng | Baseline Climax (`PNG`) | R2 Climax (`animevideov3`) | **R3 Regime 1: `animevideov3` (`64f16c`)** | **R3 Regime 2: `sharp1-uc` (`64f8c` 2x)** | **R3 Regime 3: `sharp1-suc` (`24f8c` 2x)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Khối lượng tính toán / frame** | `2.571 TFLOPs` | `2.571 TFLOPs` | **`2.571 TFLOPs`** (`621K` params) | **`1.260 TFLOPs`** (`305K` params) | **`0.1859 TFLOPs`** (`45.2K` params) |
| **Nhiệt độ GPU trước khi chạy** | — | `48.0 °C` | **`46.0 °C`** | **`42.0 °C`** | **`43.0 °C`** |
| **Tổng thời gian xử lý (`total_time_s`)** | `173.24 s` | `149.83 s` | **`145.95 s`** (`-15.75%` vs Base) | **`79.79 s`** (`-53.94%` vs Base) | **`71.05 s`** (`-58.99%` vs Base) |
| **Effective Pipeline FPS** | `8.53 fps` | `9.86 fps` | **`10.13 fps`** (`+18.76%` vs Base) | **`18.52 fps`** (`2.17x` vs Base) | **`20.80 fps`** (`2.44x` vs Base) |
| **AI / Pipe Steady-State FPS** | `8.90 fps` | `10.26 fps` | **`10.52 fps`** (`+18.20%` vs Base) | **`20.69 fps`** (`2.32x` vs Base) | **`24.56 fps`** (`2.76x` vs Base) |
| **Pure TensorRT Isolated FPS (`8000/avg_trt_ms`)**| — | `10.76 fps` | **`11.00 fps`** (**`28.27 TFLOPS`**) | **`22.83 fps`** (**`28.77 TFLOPS`**) | **`44.91 fps`** (`8.35 TFLOPS`) |
| **Peak / Avg VRAM (`MiB`)** | `8119 / 7417` | `7979 / 6535` | **`8156.0 / 6586.6 MiB`** | **`8061.0 / 6081.9 MiB`** | **`7425.0 / 5533.8 MiB`** |
| **Active GPU Compute Util (`avg`)** | `73.88%` | `93.31%` | **`94.90%`** (`94.20%` toàn run) | **`94.24%`** (`84.28%` toàn run) | **`64.20%`** (`55.52%` toàn run) |
| **Active GDDR6 Mem Controller Util** | — | `43.49%` | **`44.53%`** | **`50.70%`** | **`31.27%`** |
| **Active NVENC ASIC Util (`avg / peak`)**| — | `42.42% / 100%` | **`40.88% / 100.0%`** | **`89.06% / 100.0%`** (Bão hòa kép) | **`96.36% / 100.0%`** (Kịch trần ASIC) |
| **Active GPU Power (`avg / peak`)** | — | `156.83 / 170.35 W`| **`157.61 / 170.38 W`** (`73°C`) | **`161.24 / 170.93 W`** (`70°C`) | **`121.53 / 167.42 W`** (`65°C`) |
| **Output Sanity (`|\Delta Y|`, `r`)** | `PASS` | `PASS` (`1.87`, `0.9998`) | **`PASS`** (`1.8706`, `0.999846`) | **`PASS`** (`0.3219`, `0.999921`) | **`PASS`** (`0.3101`, `0.999923`) |
| **Mux Audit & Frame Count** | `1478/1478` | `1478/1478` | **`PASS` (`1478/1478`, `247.86 MB`)**| **`PASS` (`1478/1478`, `255.98 MB`)**| **`PASS` (`1478/1478`, `258.46 MB`)**|

*(Ghi chú tham chiếu Preview 10s `270` frames: Baseline `3.61 fps` / `74.69s` $\to$ R2 `animevideov3` `8.16 fps` / `33.08s` $\to$ Hậu Nhóm A `animevideov3` `9.81 fps` / `27.53s` $\to$ Native 2x `sharp1-uc` `13.58 fps` / `19.88s`, AI Steady `20.18 fps`.)*

---

## 4. Phân Rã Vi Thời Gian CUDA Events R3 (`micro_timings`, 178 Batches $\times$ 8 Frames)

| Pha vi thời gian GPU (`Batch = 8`) | Cô lập (`silicon_probe.json`) | R3 Regime 1: `animevideov3` (`avg` / `steady`) | R3 Regime 2: `sharp1-uc` (`avg` / `steady`) | R3 Regime 3: `sharp1-suc` (`avg` / `steady`) | Giải thích vật lý & cơ chế chồng lấp |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pure PCIe H2D DMA (`49.77 MB`)** | **`2.1822 ms`** (`22.81 GB/s`) | `715.001` / `711.140 ms`* | `335.505` / `339.600 ms`* | `66.339` / `79.059 ms`* | *Trong pipeline, `evt_h2d_s` đóng dấu sớm trên `stream_h2d` rảnh trong khi WDDM `HwSchMode=2` xếp gói DMA sau batch `stream_compute` đang chạy; DMA vật lý thật chỉ tốn **`2.18 ms`** và ẩn **100%**! |
| **Fused Input Norm (`avg_in_norm_ms`)** | **`0.5133 ms`** (`290.88 GB/s`) | **`0.724`** / **`0.721 ms`** | **`0.593`** / **`0.590 ms`** | **`0.634`** / **`0.628 ms`** | Kernel `@triton.jit` 1-pass mới giảm 3.2x độ trễ chuẩn hóa đầu vào, chiếm **`< 0.17%`** chu kỳ batch. |
| **Tổng `stream_h2d` (`avg_h2d_ms`)** | `2.6955 ms` | `715.725 ms` | `336.097 ms` | `66.973 ms` | Luôn nhỏ hơn `avg_trt_ms` (`715.7 < 727.5`, `336.1 < 350.4`, `67.0 < 178.1`), chứng minh batch $b+1$ luôn nạp xong trước khi TensorRT xong batch $b$! |
| **TensorRT 10 FP16 (`avg_trt_ms`)** | — | **`727.477`** / **`722.069 ms`** | **`350.358`** / **`350.531 ms`** | **`178.148`** / **`178.410 ms`** | `90.93 ms/f` (`v3`), `43.80 ms/f` (`uc`), `22.27 ms/f` (`suc`). |
| **Output Triton P010LE (`avg_triton_ms`)**| **`5.6683 ms`** (`105.36 GB/s`) | **`5.939`** / **`6.020 ms`** | **`5.724`** / **`5.717 ms`** | **`5.504`** / **`5.508 ms`** | Biến đổi màu BT.709 TV-range 10-bit + clamp `[0,1]` cho 8 khung 4K trong `~0.71 ms/f`. |
| **PCIe D2H P010LE (`avg_d2h_ms`)** | **`8.7918 ms`** (`22.64 GB/s`) | **`8.836`** / **`8.666 ms`** | **`9.535`** / **`9.299 ms`** | **`14.542`** / **`14.418 ms`** | Khớp sát `8.79 ms` cô lập trên `v3`/`uc`; tăng lên `14.54 ms` ở `suc` do áp lực ngược từ ống NVENC đầy. |

---

## 5. Thống Kê Viễn Trắc Continuous 17-Metric Trên 3 Chế Độ R3 (`active_telemetry`)

| Thông số Viễn Trắc Phần Cứng | R3 Regime 1: `animevideov3` [136 active samples] | R3 Regime 2: `sharp1-uc` [66 active samples] | R3 Regime 3: `sharp1-suc` [56 active samples] |
| :--- | :---: | :---: | :---: |
| **GPU Compute Util (`avg_gpu_util_pct`)** | **`94.90%`** (Toàn run: `94.20%`, Peak `100%`) | **`94.24%`** (Toàn run: `84.28%`, Peak `100%`) | **`64.20%`** (Toàn run: `55.52%`, do chờ NVENC) |
| **GDDR6 Mem Controller (`avg_mem_util_pct`)**| **`44.53%`** (`~111.8 GB/s` + L2 halo) | **`50.70%`** (`~105.5 GB/s` + NVENC 2-pass) | **`31.27%`** |
| **NVENC ASIC Util (`avg / peak_enc_util`)** | `40.88%` / `100.0%` | **`89.06%` / `100.0%`** (Bão hòa kép) | **`96.36%` / `100.0%`** (Nghẽn cứng ASIC) |
| **NVDEC Decoder Util (`avg_dec_util_pct`)** | `0.0%` (H.264 Hi10P giải mã trên P-Cores) | `0.0%` (H.264 Hi10P giải mã trên P-Cores) | `0.0%` (H.264 Hi10P giải mã trên P-Cores) |
| **GPU Core Clock (`avg / peak_core_clock`)**| **`1,701.11 MHz`** / `1,957.0 MHz` | **`1,785.74 MHz`** / `1,957.0 MHz` | **`1,918.48 MHz`** / `1,965.0 MHz` |
| **GPU Memory Clock (`avg_mem_clock_mhz`)** | `7,107.41 MHz` (Active P0 `7,501 MHz`) | **`7,501.00 MHz`** (Khóa cứng P0) | **`7,501.00 MHz`** (Khóa cứng P0) |
| **GPU Power Draw (`avg / peak_power_w`)** | **`157.61 W` / `170.38 W`** (`100.2%` TGP) | **`161.24 W` / `170.93 W`** (`100.5%` TGP) | **`121.53 W` / `167.42 W`** |
| **GPU Temperature (`pre -> peak_temp_c`)** | `46.0 °C` $\to$ **`73.0 °C`** | `42.0 °C` $\to$ **`70.0 °C`** | `43.0 °C` $\to$ **`65.0 °C`** |
| **VRAM Allocation (`avg / peak_vram_mib`)** | `6,586.6 MiB` / **`8,156.0 MiB`** | `6,081.9 MiB` / **`8,061.0 MiB`** | `5,533.8 MiB` / **`7,425.0 MiB`** |
| **Host CPU Util (`avg / peak_cpu_pct`)** | `19.91%` / `94.0%` | `23.36%` / `91.4%` | `22.76%` / `93.0%` |
| **System RAM Usage (`avg / peak_ram_gb`)** | `18.42 GB` / `21.70 GB` (trên `32 GiB`) | `18.80 GB` / `21.85 GB` | `19.15 GB` / `22.30 GB` |
| **SSD Read / Write (`WD_BLACK SN7100` E:)** | `0.29 MB/s` / **`3.36 MB/s`** (Zero-Disk) | `0.02 MB/s` / **`5.10 MB/s`** (Zero-Disk) | `0.15 MB/s` / **`5.49 MB/s`** (Zero-Disk) |

---

## 6. Bằng Chứng Kiểm Định An Toàn & Bit-Exact Mux Audit (`r3_climax_*.audit.json`)

1. **Output Sanity Guard (`270x480` Luma & Pearson $r$):**
   - `r3_climax_animevideov3`: $|\Delta Y| = \mathbf{1.8706} \le 6.0$, $r = \mathbf{0.999846} \ge 0.98$ $\to$ **`PASS`**.
   - `r3_climax_sharp1_uc`: $|\Delta Y| = \mathbf{0.3219} \le 6.0$, $r = \mathbf{0.999921} \ge 0.98$ $\to$ **`PASS`**.
   - `r3_climax_sharp1_suc`: $|\Delta Y| = \mathbf{0.3101} \le 6.0$, $r = \mathbf{0.999923} \ge 0.98$ $\to$ **`PASS`**.
2. **Bit-Exact Stream Copy & Bảo Toàn Khung Hình Trên Cả 3 Run R3:**
   - Khung hình video (`ffprobe -count_frames`): Nguồn **`1478`** $\to$ Đích **`1478`** (`frames_matched: true`, độ trôi thời lượng `0.001s`).
   - Mã băm `streamhash` MD5 âm thanh trùng khớp 100% trên cả 3 thành phẩm:
     - **Track 0 (`English 2.0 FLAC` stereo, 24-bit/48kHz):** `b7840791f53a0653c08a9fdaaf9c1f67` == `b7840791f53a0653c08a9fdaaf9c1f67` (**PASS**).
     - **Track 1 (`Japanese 2.0 FLAC` stereo, 24-bit/48kHz):** `12f8a4845d7f67e0e2da1a58298da5a3` == `12f8a4845d7f67e0e2da1a58298da5a3` (**PASS**).
   - Bảo toàn `2/2` luồng phụ đề ASS (`0` và `11` packets trong cửa sổ 60s), `5/5` font TTF/OTF, `1/1` chapter, `7/7` cửa sổ interleaving 10s (`-max_interleave_delta 0`) và `7/7` điểm kiểm tra âm lượng `0.0 dB` sai lệch.

---

## 7. Kết Luận Thực Nghiệm: Ba Chế Độ Chuyển Dịch Điểm Nghẽn Vật Lý

1. **Chế độ 1 (`animevideov3`, `2.571 TFLOPs/f`):** Bão hòa hoàn toàn **Tensor Cores & trần nhiệt điện `170W` TGP** (`94.90%` GPU util, `170.38W` đỉnh, đạt `10.13 FPS` E2E / `11.00 FPS` TRT = `28.27 TFLOPS`), trong khi NVENC mới tải `40.88%`.
2. **Chế độ 2 (`sharp1-ultracompact`, `1.260 TFLOPs/f`):** Chạm **Điểm Cân Bằng Bão Hòa Kép Pareto Hoàn Hảo** — tốc độ suy luận TensorRT (`22.83 FPS`) khớp chính xác với trần ổn định của chip 7th-Gen NVENC `-multipass fullres` (`22.70 FPS`), đồng thời vắt kiệt **`94.24%` GPU Compute VÀ `89.06%` NVENC ASIC** ở `18.52 FPS` E2E (`20.69 FPS` steady).
3. **Chế độ 3 (`sharp1-superultracompact`, `0.1859 TFLOPs/f`):** TensorRT vọt lên **`44.91 FPS`**, vượt xa khả năng mã hóa 2 lượt full-4K của 7th-Gen NVENC (`22.70 FPS`), đẩy NVENC chạm trần cứng **`96.36%` trung bình / `100.0%` đỉnh** và ghìm pipeline ở **`20.80 FPS` E2E (`24.56 FPS` steady)** do áp lực ngược từ bộ mã hóa phần cứng.
