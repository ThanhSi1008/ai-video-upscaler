# Handoff: Thực thi tối ưu hóa toàn diện dự án theo lộ trình Pareto (Gói 1 + Gói 2 + Gói 3)

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-13, Asia/Bangkok (+07:00).
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu thực hiện việc tối ưu hóa toàn diện dự án dựa trên các đề xuất kiểm toán chi tiết sau khi khảo sát toàn bộ codebase.
- Kết quả đã bàn giao:
  1. **Gói 1 (Bugfix ngầm, Cô lập Side-Effects & Đồng bộ CLI):** Sửa `F823 UnboundLocalError hashlib` (`subtrans/pipeline.py`), `F821 NameError thermal_states` (`src/night_report.py`), khôi phục `--subs-pending` cho cả bản ghi `video_path` và `ja_path`/`base_name`, đưa chốt `--dry-run` bảo vệ `--clean-temp` / `--subs-pending` / `--subs-only` trong `upscale.py`, hỗ trợ alias `--subs-quality fast -> normal`, chuyển tiếp `batch_size` + `zero_disk` ở Retry Pass, khóa `.gpu_active.lock` kiểm tra PID sống (`is_gpu_lock_held_by_other`), cô lập hoàn toàn `--cache-db` & `--dry-run` (không ghi bẩn `.cast.yaml` thật, `pending_subs.json`, `error_bank`, `subtrans/models.yaml`, `logs/cascade_telemetry_*.jsonl`), làm sạch `subtrans/models.yaml`, bổ sung đầy đủ `requirements.txt`.
  2. **Gói 2 (Tối ưu Chất lượng, Thông lượng 3-Key & SQLite/Payload cho Subtrans):** Gỡ nút cổ chai 5 RPM toàn cục (mở khóa 36 RPM trên 3 API keys Flash-Lite) & sửa log đúp RPD 2x, tái sử dụng kết quả RAG mỗi batch (`rag_bundle`), sửa prompt `<glossary>` + `<cast>` đầy đủ `speaks_to` + xoay vòng đa key cho `subtrans/cascade.py`, truyền đầy đủ `gender`/`register`/`note` từ Glossary, sửa khởi tạo `HysteresisCastManager` từ `glossary`, sửa ghi đè `c.en = paired_en` sau DTW + truyền `cast_names_ja_en` + cửa sổ nhanh DTW (`subtrans/dtw_auto_align.py`), trích xuất phụ đề theo `codec_name` (bỏ qua bitmap PGS, giữ chuẩn `.srt`/`.ass`) và lùi Whisper về CPU khi GPU đang giữ lock (`src/subtitles.py`), giải tham chiếu text-pointer glossary Windows (`resolve_glossary_target`), bật SQLite `WAL` + batch `get_cues_batch`/`set_cues_batch` + lưu metadata CoT + băm SHA-256 nội dung Glossary (`compute_glossary_hash`) + compact JSON payload.
  3. **Gói 3 (Tối ưu Hiệu năng Video Pipeline, Windows Preflight & Zero-Disk V2):** Tối ưu `src/cuda_engine.py` đọc BGR `uint8` vào Pinned Memory tái sử dụng + đổi kênh/chuẩn hóa FP16 trực tiếp trên GPU (giảm 75% băng thông PCIe H2D & D2H), xác minh SHA-256 weights + neo đường dẫn `ROOT_DIR` (`src/models_resolver.py`), sửa Background Queue Preflight chạy native Windows (`BELOW_NORMAL_PRIORITY_CLASS` thay `nice -n 10`, truyền tham số qua `sys.argv`), hoàn thiện `--zero-disk` V2 (`SRVGGNetCompactTRTWrapper(scale=2.0)`, `ensure_resolution` động, tái sử dụng Pinned Buffer xuyên suốt tập, ghi `.tmp.mp4` nguyên tử, ghi `write_failed_sanity_audit`), sửa `test_d10_b` tránh giả lập FP16 bằng phần mềm trên CPU khi chạy offline.
- Trạng thái: hoàn tất.

## Mốc mã nguồn và bằng chứng kiểm định
- Root, branch, baseline trước commit: `e:\ai-video-upscaler`, branch `main`, HEAD `a535105`.
- Các file đã cập nhật/tạo mới:
  - Core & Video: `requirements.txt`, `upscale.py`, `src/config.py`, `src/cuda_engine.py`, `src/dedup.py`, `src/models_resolver.py`, `src/night_report.py`, `src/pipeline.py`, `src/scheduler.py`, `src/subtitles.py`, `src/trt_zero_copy_upscaler.py`.
  - Subtrans: `subtrans/cache.py`, `subtrans/cascade.py`, `subtrans/cast_sheet.py`, `subtrans/config.py`, `subtrans/dtw_auto_align.py`, `subtrans/gemini_backend.py`, `subtrans/glossary.py`, `subtrans/glossary.yaml`, `subtrans/models.py`, `subtrans/models.yaml`, `subtrans/parser.py`, `subtrans/pipeline.py`, `subtrans/throttle.py`.
  - Kiểm thử & Tài liệu: `tests/test_pareto_optimizations.py`, `tests/test_model_selection_and_safety.py`, `docs/agents/STATE.md`, `docs/agents/PROJECT_MAP.md`.
- Bằng chứng kiểm định thực tế:
  - `ruff check upscale.py subtrans.py src subtrans scripts tests .agents`: **PASS (0 errors)**.
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py`: **PASS 100% (`265 passed, 11 skipped, 1 deselected in 157.33s`)**, evidence tại `scratch/agent-validation-oanuvaq9`.
  - `.\.venv\Scripts\python.exe scripts/agent_context.py --check`: **PASS (0 errors)**, nằm gọn trong ngân sách byte.
  - Mux Audit / Frame Count / FPS GPU thực tế: **N/A — không tạo thành phẩm video trong phiên tối ưu offline** (tuân thủ quy tắc hỏi ý kiến trước tác vụ GPU dài và bảo vệ quota ban ngày sau 14:00).

## Bước tiếp theo (Tùy chọn khi người dùng phê duyệt)
- Chạy benchmark ngắn (`--preview 5` trên 1 tập mẫu) để đo tốc độ thực tế của Pinned `uint8` H2D/D2H, `--zero-disk` V2 và cân nhắc kích hoạt Gói 4 (`--engine tensorrt` FP16 Kernel Fusion trên RTX 3060).
