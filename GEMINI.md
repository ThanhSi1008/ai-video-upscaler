# GEMINI.md — Chỉ Dẫn Dành Cho Google Gemini / Antigravity AI Agent

> **Tài liệu khởi động nhanh cho Google Gemini & Antigravity IDE Agent.**
> **Điểm khởi đầu chi tiết:** [docs/agents/START_HERE.md](docs/agents/START_HERE.md) | **Kim chỉ nam đầy đủ:** [AGENTS.md](AGENTS.md).

---

## 1. Bất Biến Cốt Lõi (Invariants)

1. **100% CỤC BỘ trên Windows 11 Workstation:** (Intel Core i5-12600K + NVIDIA RTX 3060 12GB). Tuyệt đối không đề xuất Colab, Cloud GPU hay môi trường Linux/Kaggle.
2. **Kỷ luật Commit MỖI Yêu Cầu:** Bắt buộc commit Git ngay sau khi hoàn thành MỖI yêu cầu (nêu rõ nguồn yêu cầu, hành động cụ thể, và số liệu thực nghiệm).
3. **Bảo vệ API Key & Quota Ban Ngày:** Không in, log hoặc commit `GEMINI_API_KEY`. Không gọi API thật từ 14:00 đến đêm; test bắt buộc dùng stub hoặc `--dry-run`.
4. **Bảo vệ Checkpoint:** Tuyệt đối không xóa hay sửa đổi thư mục `.temp_<basename>` trên SSD khi chưa có sự xác nhận của người dùng.
5. **Độc quyền GPU:** Chỉ duy nhất 1 tiến trình infer GPU hoạt động tại một thời điểm; không nạp Local LLM khi GPU đang upscale.

---

## 2. Bản Đồ Ngữ Cảnh Bền Vững (Persistent Memory Bank)

- **Điểm bắt đầu:** [`docs/agents/START_HERE.md`](docs/agents/START_HERE.md)
- **Bản đồ mã nguồn:** [`docs/agents/PROJECT_MAP.md`](docs/agents/PROJECT_MAP.md)
- **Trạng thái & Hạn chế:** [`docs/agents/STATE.md`](docs/agents/STATE.md)
- **Hồ sơ quyết định kiến trúc (ADR):** [`docs/agents/DECISIONS.md`](docs/agents/DECISIONS.md)
- **Sổ tay kiểm thử an toàn:** [`docs/agents/VALIDATION.md`](docs/agents/VALIDATION.md)
- **Phân hệ Subtrans:** [`subtrans/AGENTS.md`](subtrans/AGENTS.md)

---

## 3. Lệnh Thao Tác Chuẩn

```powershell
# Kiểm tra bộ nhớ agent (không tràn context window):
python scripts/agent_context.py
python scripts/agent_context.py --check

# Kiểm thử an toàn hoàn toàn offline:
python scripts/verify_offline.py

# Chạy kiểm thử tự động toàn diện:
pytest
```
