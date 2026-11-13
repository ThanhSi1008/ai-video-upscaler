# Handoff: <tên tác vụ>

Tạo khi bắt đầu tác vụ nhiều bước; cập nhật trong lúc làm, trước compact/bàn giao/dừng. Mỗi tác vụ một file.

## Yêu cầu và phạm vi
- Ngày/múi giờ:
- Nguồn yêu cầu và mục tiêu:
- Kết quả cần bàn giao:
- Được phép / chưa được phép:
- Ràng buộc quan trọng:
- Trạng thái: đang làm / chờ thông tin / hoàn tất.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit:
- File đã sửa (kể cả chưa commit):
- Chủ sở hữu từng phần / subagent / kết quả đã nhận:
- Tiến trình/session đang chạy và nơi xem kết quả:
- File dữ liệu production cần giữ:

## Đã làm và bằng chứng
- Thay đổi: file:symbol, lý do.
- Quyết định đã chốt / giả thuyết / hướng thử bị loại và lý do.
- Lệnh kiểm tra, interpreter, số pass/fail/skip, thời gian, evidence path.
- Mux/frame/fps: số đo thật hoặc N/A + lý do.

## Bước tiếp theo
1. Hành động cụ thể tiếp theo, file/symbol hoặc lệnh.
2. Việc còn lại và điều kiện hoàn tất.
- Câu hỏi/approval còn chờ (không coi thời gian trôi qua là đồng ý):
- Rủi ro/giới hạn cần báo người dùng:

## Kết thúc
- Kết quả và giới hạn đã thông báo:
- Commit/tag: điền hash có sẵn hoặc tra `git log -- <file-handoff>` sau commit, không dùng <pending> cho task đã xong.
- Tài liệu cập nhật: [STATE](../STATE.md), [AGENTS](../../../AGENTS.md), bản đồ/quyết định liên quan.

