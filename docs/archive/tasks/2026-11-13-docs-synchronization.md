# Handoff: Đồng bộ toàn diện hệ thống tài liệu kỹ thuật & README theo kiến trúc TensorRT 10 Zero-Disk hiện hành

Tạo và hoàn tất cho yêu cầu kiểm tra, đối chiếu và cập nhật toàn bộ tài liệu dự án khớp với mã nguồn thực thi hiện tại.

## Yêu cầu và phạm vi
- Ngày/múi giờ: `2026-11-13T22:48:00+07:00`
- Nguồn yêu cầu và mục tiêu: Người dùng hỏi trực tiếp (*"trong phần tài liệu có cần chỉnh sửa lại gì cho phù hợp với hiện tại không?"*) — rà soát toàn diện các file tài liệu đối chiếu với runtime `upscale.py` / `src/*` (tại commit `d258b19`) và cập nhật mọi thông tin lỗi thời.
- Kết quả cần bàn giao: Đồng bộ đầy đủ `README.md`, `AGENTS.md`, `.agents/skills/video-upscale-ops/SKILL.md`, `docs/video-pipeline.md`, `docs/dedup-architecture.md`, `docs/audit-and-safety.md`, `docs/engineering-log.md`, `docs/agents/DECISIONS.md`, `docs/agents/STATE.md`, `docs/agents/PROJECT_MAP.md` và hồ sơ handoff này; kiểm tra byte budget/links đạt PASS 0 cảnh báo và commit riêng biệt.
- Được phép / chưa được phép:
  - Được phép: Chỉnh sửa các tài liệu gốc và tài liệu phân hệ Video/Agent Memory thuộc quyền sở hữu của phiên này.
  - Chưa được phép: Tuyệt đối không chạm, ghi đè hoặc `git add` vào các file đang có thay đổi dở dang của phiên Subtrans song song (`subtrans/AGENTS.md`, `subtrans/*`, `src/subtitles.py`, `requirements.txt`, `scripts/batch_translate_all.py`, `glossaries/*`, `docs/agents/tasks/2026-11-13-subtrans-v2-5-ceiling-polish.md`, `tests/test_subtrans_v2_5_ceiling_polish.py`).
- Ràng buộc quan trọng: Bảo toàn số đo lịch sử trong `docs/engineering-log.md` Mục 1–11 và `README.md` (không bịa số đo fps mới chưa chạy thực nghiệm trên video dài); giữ `AGENTS.md <= 12,288 bytes` và `AGENTS.md + subtrans/AGENTS.md <= 32,768 bytes`.
- Trạng thái: hoàn tất.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, branch `main`, baseline code commit `d258b19`.
- File đã sửa trong tác vụ này (11 file tài liệu):
  - `README.md`
  - `AGENTS.md`
  - `.agents/skills/video-upscale-ops/SKILL.md`
  - `docs/video-pipeline.md`
  - `docs/dedup-architecture.md`
  - `docs/audit-and-safety.md`
  - `docs/engineering-log.md`
  - `docs/agents/DECISIONS.md`
  - `docs/agents/STATE.md`
  - `docs/agents/PROJECT_MAP.md`
  - `docs/agents/tasks/2026-11-13-docs-synchronization.md`
- Chủ sở hữu từng phần / tiến trình song song: Phiên Subtrans song song đang sở hữu `subtrans/AGENTS.md` và các module dịch thuật; không đụng chạm chéo.
- File dữ liệu production cần giữ: `.env`, `.cache/`, `out/`, `logs/`, `models_bench/`.

## Đã làm và bằng chứng
- Phát hiện và khắc phục 6 nhóm lệch tài liệu lớn giữa bản viết Tháng 10/2026 và runtime Tháng 11/2026 (`d258b19`):
  1. **Tham số mã hóa NVENC (`src/config.py:275-316`):** Tài liệu cũ vẫn ghi `Multipass qres`, `Lookahead 24`, `CQ 19 / 28M`; đã đồng bộ sang `CQ 18` (`maxrate 45M`, `bufsize 90M`), `Multipass fullres`, `Lookahead 32`, AQ kép (`spatial-aq 1`, `aq-strength 6`, `temporal-aq 1`), `bf 4`, `b_ref_mode middle`.
  2. **Engine suy luận & Đường ống Zero-Disk (`src/build_trt_engine.py`, `src/trt_zero_copy_upscaler.py`, `src/pipeline.py`):** Tài liệu cũ chỉ mô tả PyTorch `batch_size=6` qua PNG; đã bổ sung `DEFAULT_BATCH_SIZE = 8`, TensorRT 10.16 FP16 (`_out_ring` 2-slot 0-alloc), Triton JIT `p010le`, cắt lát 2-frame 8K, True 3-Stage Overlap, `upscale_turbo.bat`, `unlock_rtx3060_admin.bat`, `--zero-disk`, `--no-tensorrt` và `--subs-quality` mặc định `normal`.
  3. **Dedup, GOP-RLE & Queue Preflight (`src/dedup.py`, `src/scheduler.py`):** Tài liệu cũ ghi lệch GOP luôn hủy `plan.json`; đã cập nhật `raw_actions_rle` tái phân hoạch GOP `< 1 ms`, `run_preflight_worker_cli` tách sẵn `raw_segs` trong `--queue`, 1-Pass fused `mpdecimate+gray` + Early-Exit DIS, và Cyclic Cel Memoization (`A->B->A` `BLAKE2b` + Belady's `OPT` / NTFS `os.link`).
  4. **Mux 1-Pass, Sanity 0-Reinference & Audit Đa Luồng (`src/ffmpeg_ops.py`, `src/output_sanity.py`, `src/mux_audit.py`, `src/preflight.py`):** Thay thế tên hàm không tồn tại `verify_output_sanity` và quy trình mux 2 bước cũ bằng `mux_segments_directly` (1-Pass Concat-Mux), `check_frame_sanity` / `check_frame_sanity_arrays`, 1-Pass `-f streamhash` MD5 đa luồng âm thanh, khóa `.gpu_active.lock` kiểm tra PID sống và `configure_alder_lake_pcore_qos` + `ES_DISPLAY_REQUIRED`.
  5. **Cấu trúc thư mục & Subtrans V2 trong `README.md`:** Xóa tham chiếu thư mục không tồn tại `trisub_ai/`, sửa mô tả nhầm `src/ai_engine.py` (NCNN Vulkan dự phòng), bổ sung đầy đủ các module mới và kiến trúc Subtrans V2.
  6. **ADR & Nhật ký kỹ thuật (`docs/agents/DECISIONS.md`, `docs/engineering-log.md`, `docs/agents/STATE.md`, `docs/agents/PROJECT_MAP.md`):** Bổ sung **ADR-008** (giữ `FP16` thay vì `INT8 PTQ` và giữ `NVENC p6 fullres CQ 18` thay vì hạ `p1–p3`), **ADR-009**, **Mục 12** trong `engineering-log.md`, và nâng baseline `STATE.md` / `PROJECT_MAP.md` từ `13b5183` lên `d258b19`.
- Kiểm định tự động:
  - Lệnh: `.\.venv\Scripts\python.exe scripts/agent_context.py --check` -> `PASS` (0 lỗi link, 0 cảnh báo baseline, nằm gọn trong ngân sách byte `AGENTS.md <= 12 KiB` & tổng `<= 32 KiB`).
  - Kiểm tra whitespace diff: `git diff --cached --check` -> `0 errors`.
- Mux/frame/fps: `N/A` (tác vụ chỉ cập nhật tài liệu `.md`, không thay đổi mã nguồn thực thi sau commit `d258b19`).

## Bước tiếp theo
1. Không còn việc tồn đọng cho yêu cầu đồng bộ tài liệu này.
2. Khi người dùng chạy thực nghiệm tập anime 24 phút tiếp theo bằng `upscale_turbo.bat` (`--zero-disk`), có thể ghi nhận thêm số đo wall-clock thực tế vào `docs/engineering-log.md`.
- Câu hỏi/approval còn chờ: Không.
- Rủi ro/giới hạn cần báo người dùng: File `subtrans/AGENTS.md` đang được phiên làm việc Subtrans song song cập nhật trực tiếp nên được giữ nguyên không đụng chạm để tránh xung đột Git.

## Kết thúc
- Kết quả và giới hạn đã thông báo: Đã báo cáo chi tiết 10 tài liệu được cập nhật và các điểm lệch đã khắc phục cho người dùng.
- Commit/tag: Xem `git log -n 1 -- docs/agents/tasks/2026-11-13-docs-synchronization.md` (commit ngay trên nền `d258b19`).
- Tài liệu cập nhật: [STATE](../STATE.md), [PROJECT_MAP](../PROJECT_MAP.md), [DECISIONS](../DECISIONS.md), [AGENTS](../../../AGENTS.md), [README](../../../README.md).
