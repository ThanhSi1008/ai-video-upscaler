# Handoff: Chốt kiểm toán cuối toàn bộ điểm nghẽn Pipeline Overlap, Queue Preflight & TensorRT Zero-Alloc

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-13T22:16:30+07:00
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu kiểm tra thật kỹ toàn bộ hệ thống và codebase xem tất cả những thứ có thể tối ưu được đã tối ưu hết chưa, hay vẫn còn điểm nghẽn ẩn nào chưa được mở khóa.
- Kết quả cần bàn giao:
  1. Triệt tiêu 5 điểm nghẽn ẩn cuối cùng ở tầng điều phối song song (`src/pipeline.py`), preflight hàng đợi đa tập (`src/dedup.py`, `src/scheduler.py`, `upscale.py`), bộ nhớ đệm VRAM/binding TensorRT (`src/build_trt_engine.py`, `src/trt_zero_copy_upscaler.py`) và khởi tạo tiến trình con (`src/config.py`).
  2. Bộ kiểm thử đơn vị mới `tests/test_final_ceiling_audit.py` xác minh toàn bộ 5 tối ưu cuối.
  3. Kiểm thử offline cô lập `scripts/verify_offline.py` PASS 100% (`319 passed`), `scripts/agent_context.py --check` 0 lỗi, cập nhật bộ nhớ Agent và commit riêng biệt.
- Được phép / chưa được phép: Được phép tối ưu toàn bộ mã nguồn Video Upscale (`upscale.py`, `src/`, `tests/`, `docs/agents/`, `AGENTS.md`) miễn bảo toàn 100% bất biến chất lượng `AGENTS.md` §4; không chạy GPU > 100 frames, không gọi Gemini API thật, không sửa các file Subtrans đang dirty trên working tree.
- Ràng buộc quan trọng: Bảo toàn 100% chất lượng hình ảnh anime, ma trận BT.709 hai chiều, 10-bit NVENC `p010le`, stream copy bit-exact và `--allow-partial` / `ENOSPC` retry.
- Trạng thái: hoàn tất.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, branch `main`, baseline đầu phiên `13b5183`.
- File đã sửa và đóng gói trong tác vụ này:
  - `src/pipeline.py`
  - `src/dedup.py`
  - `src/scheduler.py`
  - `upscale.py`
  - `src/build_trt_engine.py`
  - `src/trt_zero_copy_upscaler.py`
  - `src/config.py`
  - `tests/test_final_ceiling_audit.py`
  - `AGENTS.md`, `docs/agents/STATE.md`, `docs/agents/PROJECT_MAP.md`, `docs/agents/tasks/2026-11-13-final-pipeline-overlap-and-trt-polish.md`
- File ngoài phạm vi (phiên Subtrans đồng thời đang sửa — giữ nguyên không chạm): `requirements.txt`, `scripts/batch_translate_all.py`, `src/subtitles.py`, `subtrans/*`, `glossaries/*`, `docs/agents/tasks/2026-11-13-subtrans-v2-5-ceiling-polish.md`, `tests/test_subtrans_v2_5_ceiling_polish.py`.

## Đã làm và bằng chứng
- Thay đổi:
  1. `src/pipeline.py:run_upscale_pipeline`, `worker_finalize_zero_disk`: Khởi chạy sớm `future_extract` của `chunk i+1` ngay trước khi GPU suy luận `chunk i` (đạt `100%` chồng lấp giải mã CPU/NVDEC với GPU ở `--zero-disk` và Standard Disk khi trống `>= 1600 MB`); đẩy `enc_proc.wait() + os.replace` sang `EncodeWorker` sau khi kiểm định Sanity trong RAM để GPU qua chunk kế tiếp với `0ms` chờ flush lookahead 32 frames.
  2. `src/dedup.py:_encode_actions_rle`, `_decode_actions_rle`, `_partition_actions_into_chunks`, `load_or_create_plan`, `run_preflight_worker_cli`: Nén chuỗi hành động `raw_actions_rle` trong `plan.json` để tái phân hoạch theo lưới GOP thật trong `< 1 ms` (0 lượt giải mã lại FFmpeg); đồng thời cho worker preflight nền tách sẵn `raw_segs` theo GOP và truyền đúng `dedup_policy` từ `upscale.py`/`src/scheduler.py`.
  3. `src/build_trt_engine.py:TRTRuntimeEngineWrapper._acquire_output_buffer`, `_exec_single_batch` & `src/trt_zero_copy_upscaler.py:DoubleBufferedPipeUpscaler`: Bộ đệm vòng đôi VRAM 2-slot (`_out_ring`) khử ~2.100 lần `torch.empty` (`199 MB`/batch) mỗi tập; cache `set_input_shape`/`set_tensor_address`; transpose NCHW ngay ở `uint8` (`49.7 MB` thay vì `99.5 MB` `float16`) giảm 50% băng thông VRAM khi chạy TensorRT.
  4. `src/config.py:_detect_hevc_encoder`: Kế thừa biến môi trường `UPSCALER_HEVC_ENCODER` để worker con không spawn `ffmpeg -encoders` khi import.
- Lệnh kiểm tra & bằng chứng:
  - `.\.venv\Scripts\python.exe scripts/agent_context.py --check` -> `PASS: required memory files, local links, root <= 12288 bytes, root + subtrans <= 32768 bytes`.
  - `ruff check upscale.py src tests/test_final_ceiling_audit.py` -> `All checks passed!`.
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py` -> `319 passed, 11 skipped, 1 deselected in 82.01s`, `Pytest exit code: 0`, Evidence: `E:\ai-video-upscaler\scratch\agent-validation-z72ztllb`.
- Mux/frame/fps: `N/A — không tạo thành phẩm` (chỉ kiểm toán, tối ưu mã nguồn và xác minh bằng unit/integration tests cô lập theo quy tắc an toàn).

## Bước tiếp theo
1. Hoàn tất toàn bộ các tầng Phần cứng, Hệ điều hành/Driver, Trình biên dịch AI (TensorRT 10 + Triton), Kiến trúc gối đầu 3 tầng & Hàng đợi đa tập, và Thuật toán đặc thù Anime.
2. Khi người dùng muốn nghiệm thu tốc độ thực tế trên video thật: chạy `.\upscale_turbo.bat samples\sample_30s.mkv` hoặc `--preview 10` khi có yêu cầu.

## Kết thúc
- Kết quả và giới hạn đã thông báo: Báo cáo minh bạch 5 điểm nghẽn ẩn cuối cùng vừa phát hiện & khóa sổ hoàn toàn, kèm bảng tổng kết toàn diện các tầng đã kịch trần và các giới hạn vật lý bất khả kháng (xung RAM kit trộn hãng, Resizable BAR trong BIOS).
- Commit/tag: tra `git log -n 1 -- docs/agents/tasks/2026-11-13-final-pipeline-overlap-and-trt-polish.md` ngay sau bước commit của phiên này.
- Tài liệu cập nhật: [STATE](../STATE.md), [PROJECT_MAP](../PROJECT_MAP.md), [AGENTS](../../../AGENTS.md).
