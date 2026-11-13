# Handoff: Mở khóa toàn diện hiệu năng PC & RTX 3060 cho AI Video Upscaler

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-13, Asia/Bangkok (+07:00).
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu thực hiện tối đa mọi biện pháp có thể làm được (trong phạm vi quyền hiện tại, môi trường `.venv` và codebase) để mở khóa toàn bộ hiệu năng của bộ PC (`Intel Core i5-12600K` + `MSI RTX 3060 12GB GA106-302-A1` + `WD_BLACK SN7100 Gen4 NVMe`) cho dự án `ai-video-upscaler`, sau đó liệt kê chi tiết những bước cuối chỉ người dùng mới thực hiện được (BIOS / Admin UAC / MSI Afterburner / đóng app VRAM).
- Phạm vi bảo vệ:
  - Bảo toàn nguyên trạng (không đụng chạm, không stage) các file đang dở của tác vụ Subtrans khác trong working tree (`subtrans/*`, `scripts/batch_translate_all.py`, `tests/test_subtrans_deep_upgrades.py`, `docs/agents/tasks/2026-11-13-subtrans-deep-audit-and-upgrade.md` và các hunk `--prefer-whisper` trong `upscale.py`).
  - Không chạy GPU dài `> 100 frames`, không đổi model mặc định `realesr-animevideov3` hay hạ chất lượng mã hóa NVENC (`p6`, `cq 18`, `fullres`), không gọi Gemini API thật, không đóng `blender.exe` (PID 2616) của người dùng.

## Đã triển khai hoàn tất
1. **Cấu hình Windows 11 User-Space (`HKCU` & Power Plan):**
   - Nhân bản và kích hoạt **Ultimate Performance** (`GUID: 36766665-39c4-4fcc-b947-fa8f88634596`).
   - Cấu hình `HKCU:\Software\Microsoft\DirectX\UserGpuPreferences`: chuyển toàn bộ ứng dụng nền (`Zalo.exe`, `ZaloCall.exe`, `Cloudflare WARP.exe`, `AdobeNotificationClient.exe`, `IntelGraphicsSoftware.exe`, `msedgewebview2.exe`, `msedge.exe`) sang iGPU `Intel UHD Graphics 770` (`GpuPreference=1;`) và ghim cứng `.venv\Scripts\python.exe` + `ffmpeg.exe` (`8.1.2`) vào `NVIDIA GeForce RTX 3060` (`GpuPreference=2;`).
   - Tắt Xbox GameDVR background capture trong `HKCU` (`GameDVR_Enabled=0`, `AppCaptureEnabled=0`).
2. **Nâng cấp Runtime AI trong `.venv` (0 xung đột `torch 2.6.0+cu124`):**
   - Cài đặt thành công `tensorrt-cu12==10.16.1.11`, `tensorrt-cu12-bindings==10.16.1.11`, `tensorrt-cu12-libs==10.16.1.11` (chỉ nhánh CUDA 12, tiết kiệm ~1.91 GB so với gói gộp CU13), `triton-windows==3.2.0.post21`, `onnxscript==0.7.2`, `onnx_ir==1.1.0`.
3. **Mở khóa kiến trúc Codebase (`src/`, `upscale.py`, `upscale_turbo.bat`, `scripts/`):**
   - `src/config.py`: Bổ sung `configure_cuda_fast_math()` mở khóa lệnh Tensor Core `HMMA.16816.F16` full-rate 51 TFLOPS (`allow_fp16_reduced_precision_reduction=True`), `allow_tf32=True`, `cudnn.benchmark=True`.
   - `src/preflight.py`: Bổ sung `configure_alder_lake_pcore_qos()` tắt `EcoQoS` (`ProcessPowerThrottling=4`) + nâng `ABOVE_NORMAL_PRIORITY_CLASS` trong `AntiSleepContext`, giữ tiến trình trên 6 nhân P-Core của `i5-12600K` khi chạy ngầm qua đêm.
   - `src/build_trt_engine.py`: Viết lại hoàn chỉnh với `_PureFP16TRTExportModule` (gộp Conv2d+PReLU trong 2MB L2 SRAM), hỗ trợ song song TensorRT 10 (`set_tensor_address` + `execute_async_v3`) & 8.x, ép `.contiguous()` NCHW trước `data_ptr()`, cache `.engine` theo `(model_id, hw, batch, gpu, trt_ver)`, tự fallback PyTorch khi gặp độ phân giải lạ.
   - `src/cuda_engine.py` & `src/trt_zero_copy_upscaler.py`: Cắt lát vi mô 2-frame ở đuôi 8K (`RealESRAnimeVideoV3`, `SRVGGNetCompactTRTWrapper`) và `GPUColorConverterP010LE` giảm ~75% đỉnh đệm 8K (~4.5 GB VRAM tiết kiệm, khớp bit-exact `max_diff == 0.0`); gộp kiểm tra `isfinite` trên GPU + Pinned Output Buffer chỉ đồng bộ 1 lần mỗi batch.
   - `src/ffmpeg_ops.py` & `src/pipeline.py`: Bổ sung `extract_frames_raw_rgb24()` giải mã trực tiếp `-hwaccel cuda` BT.709 TV->PC qua stdout pipe vào RAM (0 PNG ghi đĩa cho `--zero-disk`), tích hợp TensorRT 10 xuyên suốt cả đường chuẩn lẫn đường Zero-Disk.
   - `src/ramdisk.py`: Tự nhận diện RAMDisk (`UPSCALER_RAMDISK_DIR` / nhãn `"RAM"`), mặc định định tuyến bộ đệm về `E:\ai-video-upscaler\scratch\rolling_buffer` trên NVMe Gen4 `WD_BLACK SN7100` thay vì `%TEMP%` trên ổ `C:` Gen3 chậm.
   - `upscale.py`, `upscale_turbo.bat`, `scripts/unlock_rtx3060_admin.ps1`: Thêm cờ `--no-tensorrt`, launcher 1-click `upscale_turbo.bat`, và script Admin tự động bật HAGS + nạp profile `.nip` tắt `CUDA - Force P2 State` + chẩn đoán phần cứng trực quan.

## Kiểm chứng
- `ruff check upscale.py subtrans.py src subtrans scripts tests`: `All checks passed!` (`0 errors`).
- `.\.venv\Scripts\python.exe scripts/agent_context.py --check`: `PASS: required memory files, local links, root <= 12288 bytes, root + subtrans <= 32768 bytes`.
- `.\.venv\Scripts\python.exe scripts/verify_offline.py`: **`288 passed, 11 skipped, 1 deselected in 83.48s`** (`Pytest exit code: 0`, Evidence: `E:\ai-video-upscaler\scratch\agent-validation-6d7bm6a7`).
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\unlock_rtx3060_admin.ps1 -CheckOnly`: `PASS` (`Torch=2.6.0+cu124 | TensorRT=True | Triton=True`, Power Scheme `Ultimate Performance`, đã cài sẵn `Orbmu2k.nvidiaProfileInspector` v3.0.2.3 và bổ sung `-silent -mergeImport` + launcher `unlock_rtx3060_admin.bat`).
- Mux Audit / Frame Count / FPS trên thành phẩm: `N/A — không tạo thành phẩm` (tuân thủ giới hạn không chạy GPU dài khi chưa có xác nhận và đang mở `blender.exe`).
- Trạng thái: hoàn tất.
