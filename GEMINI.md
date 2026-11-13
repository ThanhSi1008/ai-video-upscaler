# Gemini / Antigravity — project entry point

Đọc [AGENTS.md](AGENTS.md), sau đó [START_HERE](docs/agents/START_HERE.md), [STATE](docs/agents/STATE.md) và hồ sơ tác vụ đang tiếp tục. Các file đó là nguồn chung cho mọi agent; không duy trì một bản policy khác tại đây.

Sau compact/chuyển phiên, kiểm tra Git và đọc lại handoff trước hành động. Khi sửa Subtrans, đọc [subtrans/AGENTS.md](subtrans/AGENTS.md). Cập nhật handoff ngay trong lúc làm, giữ quyền sở hữu file rõ ràng với subagent.

Bootstrap: `python scripts/agent_context.py` (thư viện chuẩn, không gọi API).
Kiểm thử: xem [VALIDATION](docs/agents/VALIDATION.md); dùng Python trong .venv và runner cô lập.
Nếu công cụ không tự nạp chỉ dẫn, yêu cầu nó đọc các file trên thủ công.

