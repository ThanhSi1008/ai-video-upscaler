# Mẫu ghi chép nhiệm vụ (Task Handoff Template)

> **Mỗi khi thực hiện một yêu cầu mới, tạo một file `docs/agents/tasks/YYYY-MM-DD-<tên-nhiệm-vụ>.md` theo mẫu này.**

---

## 1. Thông Tin Nhiệm Vụ
- **Ngày:** YYYY-MM-DD
- **Nguồn yêu cầu:** Người dùng yêu cầu / AI nghiên cứu đề xuất
- **Người thực hiện:** AI Agent Coder (Senior Python Engineer)
- **Commit liên quan:** `<hash>`
- **Trạng thái:** HOÀN TẤT / ĐANG THỰC HIỆN / TẠM DỪNG

---

## 2. Mục Tiêu & Ràng Buộc
- **Mục tiêu chính:** [Mô tả ngắn gọn 1-2 câu]
- **Ràng buộc bất biến:** [Ví dụ: 100% cục bộ, không gọi API thật ban ngày, bảo vệ checkpoint .temp, không làm gãy test]

---

## 3. Nhật Ký Thay Đổi Cụ Thể
- **File / Module thay đổi:**
  - `path/to/file1.py`: [Mô tả thay đổi]
  - `path/to/file2.py`: [Mô tả thay đổi]
- **Lỗi đã sửa / Tính năng đã thêm:**
  1. [Lỗi 1 / Tính năng 1]
  2. [Lỗi 2 / Tính năng 2]

---

## 4. Số Liệu & Kết Quả Kiểm Chứng (Verification)
- **Kiểm thử tự động:** `pytest` [Số test pass, thời gian chạy]
- **Kiểm định Mux:** [Khớp số luồng audio/sub/font, bit-exact]
- **Chỉ số hiệu năng:** [FPS, VRAM giải phóng, thời lượng]

---

## 5. Handoff Cho Agent Kế Tiếp
- **Điểm cần lưu ý khi bắt đầu tiếp theo:** [Những gì chưa làm hoặc ca biên cần quan sát]
- **Tài liệu đã cập nhật:** [`AGENTS.md`](../../../AGENTS.md), [`STATE.md`](../STATE.md)
