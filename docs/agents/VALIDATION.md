# Sổ Tay Kiểm Thử & Xác Minh An Toàn (VALIDATION)

> **Cập nhật ngày:** 2026-10-29  
> **Tài liệu liên quan:** [START_HERE](START_HERE.md) | [PROJECT_MAP](PROJECT_MAP.md) | [STATE](STATE.md) | [DECISIONS](DECISIONS.md) | [Root AGENTS](../../AGENTS.md)

---

## 1. Ba Nguyên Tắc Kiểm Thử Bất Biến (Safety Invariants)

1. **Tuyệt đối không rò rỉ hoặc tiêu tốn Quota API thật:**
   - Mọi bài kiểm thử tự động (`pytest`) phải sử dụng mock/stub backend (`subtrans/gemini_backend.py` hoặc `tests/test_key_rotation_stub.py`).
   - Tuyệt đối không gọi API thật trong giờ làm việc (14:00 đến đêm).
2. **Tuyệt đối không xóa hoặc ghi đè Checkpoint sản xuất:**
   - Thư mục `.temp_<basename>` trên SSD chứa các chunk đã upscale phục vụ resume qua đêm. Mọi bài test dùng fixture `tmp_path` do pytest quản lý riêng biệt.
3. **Độc quyền tài nguyên phần cứng (GPU & RAM):**
   - Không chạy các bài test gắn tag `@pytest.mark.gpu` khi pipeline upscale chính đang hoạt động để tránh gây crash hoặc OOM.

---

## 2. Các Lệnh Kiểm Thử Thường Dùng (Test Playbook)

### A. Kiểm tra tính toàn vẹn bộ nhớ ngữ cảnh Agent (Memory Integrity)
Chạy script kiểm tra để đảm bảo toàn bộ tài liệu agent đều nằm trong ngân sách byte và không có link chết:
```powershell
python scripts/agent_context.py --check
```

### B. Kiểm thử cô lập nguồn hoàn toàn offline (Offline Snapshot)
Công cụ `scripts/verify_offline.py` sao chép mã nguồn sang thư mục tạm, ngắt kết nối socket mạng và lọc sạch các biến môi trường nhạy cảm (`GEMINI_API_KEY`, credential files) trước khi chạy test:
```powershell
python scripts/verify_offline.py
```

### C. Chạy kiểm thử theo từng phân hệ (Fast Feedback)
```powershell
# 1. Kiểm thử phân hệ tiền xử lý và lọc ảo giác Whisper (< 2s):
.\.venv\Scripts\python.exe -m pytest subtrans/tests/test_whisper_noise_and_speaker.py subtrans/tests/test_parser.py

# 2. Kiểm thử toàn bộ phân hệ dịch thuật Subtrans (~30s):
.\.venv\Scripts\python.exe -m pytest subtrans/tests/

# 3. Kiểm thử phân hệ Video Pipeline, Dedup và Chốt chặn Sanity:
.\.venv\Scripts\python.exe -m pytest tests/test_output_sanity.py tests/test_dedup_policy.py tests/test_mux_audit_regression.py

# 4. Kiểm thử toàn bộ test suite dự án (bỏ qua GPU tests):
.\.venv\Scripts\python.exe -m pytest -m "not gpu"
```

---

## 3. Tiêu Chí Đánh Giá Hoàn Tất (Acceptance Checklist)

Trước khi commit mã nguồn hoặc kết thúc lượt làm việc:
- [ ] Lệnh `python scripts/agent_context.py --check` trả về 0 lỗi.
- [ ] Toàn bộ bài test liên quan trong `tests/` và `subtrans/tests/` đạt 100% PASS.
- [ ] Không có file tạm, checkpoint giả hoặc credential rơi rớt trong thư mục dự án.
- [ ] Tài liệu [`STATE.md`](STATE.md) và hồ sơ handoff trong `docs/agents/tasks/` được cập nhật.
