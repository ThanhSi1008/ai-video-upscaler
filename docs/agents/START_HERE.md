# Khởi động và khôi phục ngữ cảnh Agent

Bộ nhớ nằm trong Git, dùng chung cho Codex, subagent và công cụ khác. Nó giúp khôi phục công việc sau khi rút gọn ngữ cảnh; không tăng context window hoặc bảo đảm AI không quên. Không cần dịch vụ/vector database, API hay cấu hình tài khoản riêng.

## Khi bắt đầu, tiếp tục hoặc nhận bàn giao

1. Đọc [AGENTS gốc](../../AGENTS.md), [STATE](STATE.md) và **hồ sơ tác vụ được chỉ định** trong [tasks](tasks). Đối chiếu yêu cầu mới nhất của người dùng với mục tiêu đã lưu.
2. Chạy từ root repo:
   `.\.venv\Scripts\python.exe scripts/agent_context.py`
   Kiểm tra branch, HEAD, diff và cảnh báo baseline cũ. Trạng thái Git trực tiếp đáng tin hơn dòng “sạch” trong tài liệu cũ.
3. Đọc [PROJECT_MAP](PROJECT_MAP.md), [DECISIONS](DECISIONS.md) theo phần liên quan; sửa Subtrans phải đọc [subtrans/AGENTS](../../subtrans/AGENTS.md). Đọc [VALIDATION](VALIDATION.md) trước chạy test.
4. Đọc source của đúng symbol và test liên quan. Code hiện tại mô tả hành vi; chính sách người dùng quy định điều được phép. Khi hai thứ khác nhau, ghi nhận và giữ nguyên cấu hình nhạy cảm cho đến khi có yêu cầu thay đổi.

Không nạp toàn bộ lịch sử, logs, corpus, media, weights, cache hoặc `.env` vào context. Tài liệu benchmark cũ là dữ liệu lịch sử, không phải số đo mới.

## Trong lúc làm việc: lưu trước khi mất context

Với tác vụ nhiều bước, tạo `tasks/YYYY-MM-DD-<slug>.md` từ [TEMPLATE](tasks/TEMPLATE.md) **ngay khi bắt đầu**. Agent chính cập nhật mục đang làm trong [STATE](STATE.md). Mỗi tác vụ có file riêng; không dùng một scratchpad chung cho nhiều agent.

Cập nhật hồ sơ sau mốc quan trọng, kết quả test, quyết định/đổi phạm vi, trước khi bàn giao hoặc compact, và trước khi dừng:
- mục tiêu, phạm vi đã được phép, điều còn chờ người dùng;
- kết quả đã xác minh kèm lệnh và file chứng cứ, tách giả thuyết;
- branch/base commit, file đã sửa, việc dở dang, tiến trình còn chạy;
- bước tiếp theo cụ thể để một agent mới tiếp tục được ngay.

Không đợi đến cuối phiên mới viết handoff. Sau compact, đọc lại hồ sơ và Git; không thực hiện lại bước đã xong. File nhớ là dữ liệu bàn giao, không cấp thêm quyền và không thay thế chỉ dẫn mới nhất của người dùng. Không lưu key hoặc toàn bộ hội thoại.

## Phối hợp subagent

Agent chính gửi cho mỗi subagent: mục tiêu, root/branch, file handoff, file được sửa, invariant, tiêu chí trả kết quả. Chia quyền sở hữu file để tránh ghi đè; subagent phải đọc chỉ dẫn áp dụng dù lịch sử chat có được truyền hay không.

Subagent trả: file/symbol đã đọc hoặc sửa, phát hiện + bằng chứng, test đã chạy, điều chưa biết. Agent chính tổng hợp vào handoff; không dựa vào mailbox tạm làm bộ nhớ lâu dài. Agent chính commit cho yêu cầu đã tích hợp; không để nhiều agent cùng stage/commit hoặc chạy GPU.

## Lệnh nhỏ để tìm đúng code

```powershell
.\.venv\Scripts\python.exe scripts/agent_context.py --check
.\.venv\Scripts\python.exe scripts/agent_context.py --inventory
.\.venv\Scripts\python.exe scripts/agent_context.py --symbols src/subtitles.py
.\.venv\Scripts\python.exe scripts/agent_context.py --find snap_whisper
.\.venv\Scripts\python.exe scripts/verify_offline.py
```

Bootstrap dùng thư viện chuẩn, không import ứng dụng. `--install-hooks` là thao tác ghi tùy chọn; không tự chạy khi bootstrap. Kiểm tra link/byte không xác nhận nội dung tài liệu đúng với runtime.

## Duy trì và dùng trên nền tảng khác

- Giữ root AGENTS ngắn (ngân sách repo 12 KiB); chi tiết đặt trong docs và chỉ đọc theo nhu cầu. Codex có giới hạn chỉ dẫn project mặc định 32 KiB theo [OpenAI Docs](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
- Antigravity có cơ chế đọc AGENTS/GEMINI theo workspace và thư mục; xem [Google Rules](https://antigravity.google/docs/rules/). [GEMINI](../../GEMINI.md) và [CLAUDE](../../CLAUDE.md) chỉ trỏ về cùng nguồn, không sao chép policy.
- Kiểm tra ở phiên mới bằng yêu cầu: “Đọc AGENTS.md và docs/agents/START_HERE.md, nêu baseline, task đang làm và bước tiếp theo.” Nếu công cụ không tự nạp file, gửi yêu cầu này thủ công.
- Khi source đổi: cập nhật bản đồ/quyết định liên quan, kiểm thử, cập nhật STATE với **baseline đã kiểm tra**, handoff và commit chi tiết. Không sửa hash sau mỗi commit chỉ đổi tài liệu; lịch sử Git xác định commit hoàn tất.

