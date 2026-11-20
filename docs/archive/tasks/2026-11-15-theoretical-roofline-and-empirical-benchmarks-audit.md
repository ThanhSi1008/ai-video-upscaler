# Handoff: Theoretical Roofline, Empirical Benchmarking & Gap Analysis Optimization Audit (R3 GA106 Silicon Calibration)

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-15 (UTC+7)
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu tính toán chi tiết toàn bộ các thông số kỹ thuật trên lý thuyết (R1), chạy thực tế đo chính xác benchmarks (R2), sau đó so sánh đối chiếu Gap Analysis (R3) để đánh giá xem Project đã thật sự tối ưu nhất có thể so với các số liệu trên lý thuyết và thực tế; kèm đợt kiểm toán hiệu chuẩn vật lý Silicon Ampere GA106 tiêu dùng, vi tối ưu hóa R3 và đo đạc 3 chế độ Climax trên máy trạm (i5-12600K, RTX 3060 12GB `GA106-302-A1`, `32 GiB` DDR4 `2,133 MT/s`, `WD_BLACK SN7100 500GB` NVMe Gen4, NVENC 10-bit).
- Kết quả cần bàn giao:
  1. [docs/roofline_theoretical_model.md](../../roofline_theoretical_model.md) — Mô hình trần lý thuyết Consumer Ampere GA106 Roofline, độ phức tạp chính xác 3 model chuẩn, định lý ẩn trễ 3 tầng, giới hạn Amdahl `-multipass fullres` & script Python tự kiểm chứng (v3.0).
  2. [docs/empirical_benchmark_report.md](../../empirical_benchmark_report.md) — Báo cáo đo trần silicon cô lập (`silicon_probe.json`) & thực nghiệm 3 chế độ Climax R3 trên `Fate/Apocrypha - 22` (v3.0).
  3. [docs/benchmarks_gap_analysis_audit.md](../../benchmarks_gap_analysis_audit.md) — Báo cáo kiểm toán khoảng trống vi cấu trúc & giải quyết trọn vẹn 3 nghịch lý lịch sử từ nguyên lý đầu tiên (v3.1).
  4. [benchmarks/final_optimization_report.md](../../../benchmarks/final_optimization_report.md) — Hiệu chuẩn thông số phần cứng/`.venv` (Mục 2) & bổ sung Mục 13 tổng hợp R3.
  5. Vi tối ưu hóa mã nguồn & công cụ đo trần silicon: [src/config.py](../../../src/config.py), [src/trt_zero_copy_upscaler.py](../../../src/trt_zero_copy_upscaler.py), [scripts/verify_offline.py](../../../scripts/verify_offline.py), [scripts/silicon_roofline_probe.py](../../../scripts/silicon_roofline_probe.py), [tests/test_nvme_cache_and_input_triton.py](../../../tests/test_nvme_cache_and_input_triton.py).
  6. Bằng chứng runtime tại `scratch/hw_and_model_exact_audit.json` và `out/benchmark_r3_final/` (`silicon_probe.json` + 3 bộ `r3_climax_*_summary.json`, `*_telemetry.csv`, `*.audit.json`).
- Trạng thái: Đã tổng hợp và kiểm định tài liệu/bộ nhớ (`docs-synth` -> `@verifier`).

## Mốc mã nguồn và điều phối
- Root: `e:\ai-video-upscaler`
- Branch: `main`
- Baseline commit: `aa421b0`
- File đã sửa/tạo trong đợt R3:
  - `src/config.py` (`CACHE_ROOT_DIR`, `configure_nvme_jit_caches` định tuyến `.cache` về `WD_BLACK SN7100` NVMe Gen4, `gpu_exclusivity_lock`)
  - `src/trt_zero_copy_upscaler.py` (`_nhwc_u8_to_nchw_fp16_norm_kernel` 1-pass `@triton.jit`, `_ensure_cuda_event_pools`, tách `avg_h2d_dma_ms` / `avg_in_norm_ms`)
  - `scripts/verify_offline.py` (cô lập `TEMP`/`TMP`/`TMPDIR` và `--basetemp` trên `E:\ai-video-upscaler\scratch\`)
  - `scripts/silicon_roofline_probe.py` (đo trần silicon GA106 cô lập có khóa GPU PID)
  - `tests/test_nvme_cache_and_input_triton.py` (13 ca kiểm thử đơn vị offline mới)
  - `docs/roofline_theoretical_model.md`, `docs/empirical_benchmark_report.md`, `docs/benchmarks_gap_analysis_audit.md`, `benchmarks/final_optimization_report.md`
  - `docs/agents/STATE.md`, `docs/agents/PROJECT_MAP.md`, `docs/agents/tasks/2026-11-15-theoretical-roofline-and-empirical-benchmarks-audit.md`
- Chủ sở hữu điều phối Swarm: `@coordinator` -> `@hw-silicon-probe` (#7), `@io-kernel-opt` (#9), `@gpu-bench` (#12), `@docs-synth`, `@verifier`.

## Đã làm và bằng chứng
- **Kiểm toán phần cứng & `.venv` thực tế (`scratch/hw_and_model_exact_audit.json`):**
  - Windows 11 Pro `10.0.26300` (Build `26300`), Gigabyte `B760M G P WIFI DDR4`, Intel Core i5-12600K (`6P+4E`, `9.5 MiB` L2, `20.0 MiB` L3), RAM `32 GiB` Dual-Channel DDR4 đang chạy ở **`2,133 MT/s` @ `1.20 V` JEDEC (`34.13 GB/s`)**, Disk 0 (`C:/D:`) `OCPC MFL-300 512GB` Gen3, Disk 1 (`E:`) **`WD_BLACK SN7100 500GB` Gen4 (`6,800/5,800 MB/s`)**, RTX 3060 `GA106-302-A1` (`28 SMs`, `2.25 MiB` L2, `360.0 GB/s` GDDR6, `BAR1 = 256 MiB` Resizable BAR tắt, `170.0 W` TGP); Python `3.11.9`, PyTorch `2.6.0+cu124`, TensorRT `10.16.1.11`, Triton `3.2.0` (`3.2.0.post21`), PyAV `18.1.0`, FFmpeg `8.1.2`.
- **Hiệu chuẩn Mô hình Trần Lý Thuyết Consumer GA106 & `silicon_probe.json` (R1):**
  - Sửa hằng số A100/2:4-sparse cũ sang kiến trúc **GeForce Ampere GA106 đặc (`0%` sparsity)**: Dense FP16 `HMMA.16816.F16` = `512 FMAs/SM/clk` (`1,024 FLOPs/SM/clk`) = **`50.95 TFLOPS` @ `1,777 MHz`** (`38.844 TFLOPS` đo thực tế `cublasGemmEx(16F)`); Dense FP16 `HMMA.16816.F32` (GeForce half-rate) = `256 FMAs/SM/clk` (`512 FLOPs/SM/clk`) = **`25.48 TFLOPS` @ `1,777 MHz` / `27.96 TFLOPS` @ `1,950 MHz`** (`23.211 TFLOPS` đo thực tế `cublasGemmEx(32F)`); STREAM Triad = **`302.61 GB/s`** (`I_knee,emp = 128.36 FLOPs/B`); PCIe 4.0 DMA = `22.81 / 22.64 GB/s`; 7th-Gen NVENC `-multipass fullres` hội tụ biên = **`22.70 FPS` (`44.053 ms/f`)**.
  - Độ phức tạp chính xác 3 model chuẩn: `animevideov3` (`64f16c`, `2.571 TFLOPs/f`, `I_realized = 252.70 FLOPs/B`), `sharp1-ultracompact` (`64f8c`, `1.260 TFLOPs/f`, `I_realized = 272.45 FLOPs/B`), `sharp1-superultracompact` (`24f8c`, `0.1859 TFLOPs/f`, `I_realized = 94.32 FLOPs/B`).
- **Vi tối ưu hóa R3 & Đo đạc Thực nghiệm 3 Chế Độ Climax (`out/benchmark_r3_final/`, `1,478` frames, `95.06%` unique) (R2 & R3):**
  - Kernel `@triton.jit` chuẩn hóa đầu vào 1-pass rút thời gian từ `1.6429 ms` xuống **`0.5133 ms/batch` (`290.88 GB/s`, nhanh gấp `3.20x`)**; pool `CUDA Event` tiền cấp phát tách rõ `avg_in_norm_ms` (`0.59–0.72 ms`) khỏi hàng đợi WDDM `avg_h2d_dma_ms`.
  - **Regime 1 (`animevideov3`):** `145.95 s`, **`10.13 fps` E2E**, **`10.52 fps` steady**, **`11.00 fps` TRT (`727.48 ms/b`, `28.27 TFLOPS` — gấp `2.33x` cuDNN eager)**, GPU Active `94.90%`, Power `157.61 / 170.38 W`.
  - **Regime 2 (`sharp1-ultracompact`):** `79.79 s`, **`18.52 fps` E2E (`1.83x`)**, **`20.69 fps` steady**, **`22.83 fps` TRT (`350.36 ms/b`, `28.77 TFLOPS`)**, NVENC Active `89.06%` (`100%` pk).
  - **Regime 3 (`sharp1-superultracompact`):** `71.05 s`, **`20.80 fps` E2E (`2.05x`)**, **`24.56 fps` steady**, **`44.91 fps` TRT (`178.15 ms/b`)**, NVENC Active **`96.36%` (`100%` pk)** — chạm sát trần phần cứng NVENC `-multipass fullres` (`22.70 fps` unique $\approx 23.88$ fps với `4.94%` dedup).
  - Bất biến an toàn: Khớp `1478/1478` frames trên cả 3 chế độ, Output Sanity PASS (`|\Delta Y| = 0.31..1.87 <= 6.0`, `r = 0.99985..0.99992 >= 0.98`), Bit-exact Mux Audit PASS (2x Stereo `2.0` FLAC MD5 `b7840791f53a0653c08a9fdaaf9c1f67` / `12f8a4845d7f67e0e2da1a58298da5a3`, 2 ASS, 5 Fonts, 1 Chapter).

## Việc còn lại / kiểm định bàn giao
- Chuyển giao sang `@verifier` để chạy kiểm định độc lập toàn bộ tài liệu, script Python nhúng trong `docs/roofline_theoretical_model.md`, `scripts/agent_context.py --check`, `scripts/verify_offline.py` (`729+` tests) và `git diff --check`, sau đó `@coordinator` thực hiện commit Git theo kỷ luật `AGENTS.md` §5.
