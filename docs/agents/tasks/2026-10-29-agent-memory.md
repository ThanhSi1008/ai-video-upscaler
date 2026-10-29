# Thiết Lập Hệ Thống Bộ Nhớ Bền Vững & Chống Quên Ngữ Cảnh Cho AI Agent

---

## 1. Thông Tin Nhiệm Vụ
- **Ngày:** 2026-10-29
- **Nguồn yêu cầu:** Người dùng yêu cầu tối ưu hóa môi trường làm việc của AI Agent Coder chống quên ngữ cảnh (context window management) dựa trên các bài đăng uy tín của các hãng AI lớn (Anthropic, OpenAI, Google DeepMind).
- **Người thực hiện:** AI Agent Coder (Senior Python Engineer)
- **Commit liên quan:** `<pending>`
- **Trạng thái:** HOÀN TẤT

---

## 2. Mục Tiêu & Ràng Buộc
- **Mục tiêu:**
  - Thiết lập kiến trúc bộ nhớ phân tầng (Hierarchical Context Architecture) và công cụ bootstrap nhẹ (`scripts/agent_context.py`), cô lập môi trường kiểm thử (`scripts/verify_offline.py`).
  - Thiết lập ngân sách byte nghiêm ngặt (`ROOT_BUDGET <= 12KB`, `COMBINED_BUDGET <= 32KB`) để ngăn chặn việc instruction bloat nuốt trọn 30-40% context window của mô hình.
  - Xây dựng mạng lưới tài liệu hỗ trợ định hướng nhanh: [`START_HERE.md`](../START_HERE.md), [`PROJECT_MAP.md`](../PROJECT_MAP.md), [`STATE.md`](../STATE.md), [`DECISIONS.md`](../DECISIONS.md), [`VALIDATION.md`](../VALIDATION.md).
- **Ràng buộc:**
  - Không import nặng (PyTorch, CUDA, OpenCV) khi đọc ngữ cảnh agent.
  - Không in hay lưu API keys, không gọi network socket khi chạy offline verification.
  - 100% relative links hợp lệ bên trong repository.

---

## 3. Nhật Ký Thay Đổi Cụ Thể
- **Thư mục & File thiết lập:**
  - `docs/agents/START_HERE.md`: Điểm bắt đầu duy nhất cho mọi agent.
  - `docs/agents/STATE.md`: Trạng thái thực tế của codebase, các module đã verified và unverified.
  - `docs/agents/DECISIONS.md`: Hồ sơ quyết định kiến trúc (ADRs) giải thích "tại sao không làm X" để chống agent đề xuất lại phương án sai.
  - `docs/agents/VALIDATION.md`: Sổ tay hướng dẫn kiểm thử cô lập offline và an toàn.
  - `docs/agents/tasks/TEMPLATE.md` & `2026-10-29-agent-memory.md`: Chuẩn hóa handoff phiên làm việc.
  - `GEMINI.md` & `CLAUDE.md`: File kim chỉ nam ở root tương thích công cụ agent từ Google và Anthropic.
  - `AGENTS.md`: Tái cấu trúc tinh gọn (dưới 12KB) bằng cách liên kết trực tiếp với các tài liệu chuyên sâu `docs/`.
  - `scripts/agent_context.py`: Tiện ích tiêu chuẩn dòng lệnh kiểm tra tính toàn vẹn bộ nhớ (`--check`) và in tóm tắt (`--bootstrap`).
  - `scripts/verify_offline.py`: Công cụ snapshot mã nguồn và chạy test hoàn toàn offline chống leak key và quota.

---

## 4. Số Liệu & Kết Quả Kiểm Chứng
- `python scripts/agent_context.py --check`: PASS 0 errors (toàn bộ liên kết hợp lệ, ngân sách dung lượng đạt chuẩn).
- `pytest tests/test_agent_context.py tests/test_verify_offline.py`: PASS 27/27 bài test.
- `pytest`: PASS 100% toàn bộ test suite.
