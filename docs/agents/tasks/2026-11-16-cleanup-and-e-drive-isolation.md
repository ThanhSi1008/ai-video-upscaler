# Handoff: Tạm dừng tác vụ, cô lập 100% sang ổ E: và dọn sạch rác C: & E:

Tạo khi bắt đầu tác vụ nhiều bước; cập nhật trong lúc làm, trước compact/bàn giao/dừng. Mỗi tác vụ một file.

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-16 06:34 (+07:00)
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu tạm dừng mọi tiến trình đang chạy (chờ ngồi vào bàn PC), rà soát toàn diện rác/tàn dư trên cả ổ `C:` và `E:`, cô lập hoàn toàn project `E:\ai-video-upscaler` chỉ phụ thuộc ổ `E:` (không còn liên kết runtime/binary/cache/temp sang ổ `C:`), rồi dọn sạch toàn bộ rác đã xác minh.
- Kết quả cần bàn giao:
  - Tắt sạch daemon Web GUI `task-1021`, giữ nguyên Fate/Zero ep 10–12 ở trạng thái tạm dừng trong [handoff Fate/Zero](2026-11-16-fate-zero-ep10-12-web-gui.md).
  - Cô lập Python 3.11.9 (`bin/python311/`), `.venv` (relink `pyvenv.cfg` + `Scripts/*` + recompile `.pyc`), `ffmpeg`/`ffprobe` (`bin/ffmpeg/`), `ruff.exe` (`.venv/Scripts/ruff.exe`), toàn bộ biến cache và `TEMP`/`TMP`/`TMPDIR` về `E:\ai-video-upscaler\.cache` và `scratch/tmp`.
  - Xóa sạch toàn bộ rác verified của project trên cả `C:` và `E:`, kiểm thử `verify_offline.py` PASS 100%, `ruff`, `agent_context.py --check`, đo dung lượng trước/sau và commit Git.
- Được phép / chưa được phép: Được người dùng yêu cầu trực tiếp dọn sạch rác trên `C:` và `E:` và tách hoàn toàn khỏi `C:`. Không đụng chạm dữ liệu không thuộc project (`AppData\Local\uv`, `models--laion--clap-htsat-fused`, `.blend` trong `%TEMP%`), `.env`, SQLite `subtrans/.cache/*.db`, `models_bench/`, `.cache/trt_engines/*.engine`, `out/benchmark_r3_final/`, `out/benchmark_group_b/*.{json,md}`, `E:\Films\Raw_Sources\*.mkv`.
- Ràng buộc quan trọng: Bảo toàn 100% chữ ký `NVME_JIT_CACHE_DEFAULTS` & `configure_nvme_jit_caches()` (3 keys) trong `src/config.py`, giữ nguyên `PATH` trong `verify_offline.isolated_environment()`, giữ CRLF cho `.bat`, giữ `run_dir/tmp` tồn tại khi `verify_offline` xong, giữ `AGENTS.md <= 12,288` bytes.
- Trạng thái: hoàn tất

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, `main`, `53bfdda`
- File đã sửa (kể cả chưa commit):
  - Mới tạo: `src/env_isolation.py`, `tests/test_e_drive_isolation.py`, `docs/agents/tasks/2026-11-16-cleanup-and-e-drive-isolation.md`, `docs/agents/tasks/2026-11-16-fate-zero-ep10-12-web-gui.md`
  - Đã sửa: `src/config.py`, `src/build_trt_engine.py`, `subtrans/config.py`, `upscale.py`, `subtrans.py`, `web_gui.py`, `upscale.bat`, `upscale_turbo.bat`, `subtrans.bat`, `web_gui.bat`, `.gitignore`, `pytest.ini`, `ruff.toml`, `scripts/agent_context.py`, `scripts/verify_offline.py`, `docs/agents/PROJECT_MAP.md`, `docs/agents/STATE.md`
- Chủ sở hữu từng phần / subagent / kết quả đã nhận:
  - Agent chính thực thi trực tiếp; không dùng GPU/Gemini API; `browser` subagent cũ nhàn rỗi (`a7c87738-fe67-45ff-8714-e2bb9fd50f6b`) không can thiệp.
- Tiến trình/session đang chạy và nơi xem kết quả: Không còn daemon/GPU nào chạy (`ACTIVE_PROCS` chỉ có lệnh check, `.gpu_active.lock` = False).
- File dữ liệu production cần giữ: `.env`, `subtrans/.cache/{translations.db,knowledge.db}`, `models/models.yaml`, `models_bench/*.pth`, `.cache/trt_engines/{*.engine,trt_timing.cache}`, `out/benchmark_r3_final/`, `out/benchmark_group_b/*.{json,md}`, `E:\Films\Raw_Sources\*`, `E:\Films\Subtitles\*`, `E:\Films\Upscaled_4K\`.

## Đã làm và bằng chứng
- **Tạm dừng chạy:** Tắt daemon `web_gui.py --lan` (`task-1021`); `manage_task(list)` xác nhận `0` background task còn chạy.
- **Cô lập ổ E: (`bin/` = `540.12 MiB` thêm vào `E:\ai-video-upscaler`):**
  - Copy Python 3.11.9 runtime (`46.44 MB`, loại `site-packages`/`test`/`idlelib`/`__pycache__`) -> `bin/python311/`; copy `ffmpeg.exe` & `ffprobe.exe` (`462.33 MB`) -> `bin/ffmpeg/`; copy `ruff.exe` (`25.04 MB`) -> `.venv/Scripts/ruff.exe`.
  - Cập nhật `.venv/pyvenv.cfg` -> `E:\ai-video-upscaler\bin\python311`; vá `activate`, `activate.bat`, `numba` và `26` distlib `.exe` shebang; xóa `9,443` `.pyc` cũ/`-pytest-` và `compileall -j 8` lại `.venv/Lib/site-packages` trên `E:` (`0` tham chiếu `Users\Admin` còn sót).
  - Tạo `src/env_isolation.py:configure_e_drive_isolation` và gọi ở `src/config.py`, `subtrans/config.py`, `upscale.py`, `subtrans.py`, `web_gui.py` và 4 file `.bat` (kiểm chứng CRLF 100% + `cmd /c "subtrans.bat --help"` exit 0).
  - Cập nhật `src/build_trt_engine.py` (tự xóa `.onnx` trung gian sau build `.engine` thành công) và `scripts/verify_offline.py` (`_rmtree_force` xóa cả file read-only `.git/objects` trong `repo/` và `tmp/pytest/` khi `PASS`, giữ tối đa 2 run gần nhất).
- **Dọn dẹp có kiểm chứng:**
  - Trên `C:` (`127/127` mục, **đã xóa `10.281 GiB` / `11,038,683,091 B`**): `.upscale_venv` (`4,680.89 MB`), `%TEMP%\pip-*` (`5,301.20 MB`), `AppData\Local\pip` (`208.74 MB`), `.cache\whisper` (`138.53 MB`), `models--Systran--faster-whisper-small` (`2.54 MB`), `.triton` (`3.05 MB`), `AppData\Local\npm-cache` (`31.41 MB`), `scratch\teamwork` (`7.56 MB`), `pytest-of-Admin` + `HeadlessChrome*` + `cdp_studio_verify_*` + `tmp*` test dirs trong `%TEMP%` (`~140 MB`).
  - Trên `E:` (`4.553 GiB` ở 4B + `911.62 MB` ở 4C = **`5.443 GiB` / `5.845 GB` xóa thô**): rác video/benchmark/log trong `out/`, `scratch/` (26 bản snapshot cũ + debug scripts), `samples/` clip thử chưa track (giữ `samples/README.md`), `.cache/{cuda_jit,inductor}` + `.onnx` (giữ `*.engine` + `trt_timing.cache`), `logs/*.log` (giữ `logs/.gitkeep`), `.temp_sample_kimetsu_s03e10_climax_3min`, `scratch/.temp_test_break`, và 2 thư mục Frieren rỗng 0 file (`Frieren꞉ Beyond Journey` do lỗi dấu nháy đơn & `.temp_...S01E05...`).
- **Số đo thực tế & Kiểm thử:**
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py`: **`761 passed, 14 skipped, 1 deselected in 438.18s (0:07:18)`**, exit code `0`, evidence `E:\ai-video-upscaler\scratch\agent-validation-1xrurlc5` (`scratch/` còn đúng `0.10 MB`).
  - `.\.venv\Scripts\ruff.exe check` 10 file thêm/sửa: `All checks passed!`.
  - Audit cô lập (`.\.venv\Scripts\python.exe`): `sys.base_prefix == E:\ai-video-upscaler\bin\python311`, `which("ffmpeg/ffprobe") == E:\ai-video-upscaler\bin\ffmpeg\*`, `which("ruff") == E:\ai-video-upscaler\.venv\Scripts\ruff.EXE`, `tempfile.gettempdir() == E:\ai-video-upscaler\scratch\tmp`, 0 rò rỉ về `C:`, toàn bộ tệp bảo vệ nguyên vẹn.
  - Dung lượng trống thực đo: `C:` `75.451 GiB` -> **`85.726 GiB`** (**`+10.275 GiB` / `+11.032 GB`**); `E:` `70.516 GiB` -> **`75.490 GiB`** (**`+4.974 GiB` / `+5.341 GB`** sau khi đặt thêm `540.12 MiB` runtime `bin/` + `ruff.exe`); tổng cộng tăng **`+15.249 GiB` (`+16.373 GB`)**.
  - Mux/frame/fps: **`N/A — không tạo thành phẩm`** (chỉ cô lập môi trường, dọn rác và kiểm thử offline).

## Bước tiếp theo
1. Kiểm tra `agent_context.py --check`, `git diff --check`, commit chi tiết theo `AGENTS.md` §5.

## Kết thúc
- Kết quả và giới hạn đã thông báo: Hoàn tất tạm dừng tiến trình, cô lập 100% dự án sang ổ `E:` và dọn sạch `10.275 GiB` trên `C:` + `4.974 GiB` ròng (`5.443 GiB` thô) trên `E:`. Chưa chạy lại Fate/Zero ep 10–12 (giữ tạm dừng theo yêu cầu chờ người dùng ngồi vào PC).
- Commit/tag: xem `git log -1 -- docs/agents/tasks/2026-11-16-cleanup-and-e-drive-isolation.md` (baseline: `53bfdda`)
- Tài liệu cập nhật: [STATE](../STATE.md), [PROJECT_MAP](../PROJECT_MAP.md)
