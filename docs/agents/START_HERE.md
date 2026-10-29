# Hướng Dẫn Khởi Đầu Dành Cho AI Agent (START_HERE)

> **Chào bạn, AI Agent Coder! Đây là điểm xuất phát bắt buộc trước khi thực hiện bất kỳ thay đổi nào trong repository này.**
> **Mục tiêu:** Giúp bạn nắm bắt ngữ cảnh cốt lõi trong < 1 phút, không bị trôi/quên ngữ cảnh (context rot), và tránh lặp lại các giải pháp sai đã bị loại trừ trong quá khứ.

---

## 1. Quy Trình Khởi Động 4 Bước (Bootstrap Workflow)

1. **Xem trạng thái hiện tại của codebase tại [`STATE.md`](STATE.md):**
   - Xác định commit baseline gần nhất, nhánh đang chạy, và danh sách các tính năng đã kiểm chứng (verified) vs chưa hoàn thiện (unverified).
2. **Nắm vững các bất biến cốt lõi tại [Root `AGENTS.md`](../../AGENTS.md):**
   - Môi trường bắt buộc: 100% CỤC BỘ trên Windows 11 (Intel Core i5-12600K + NVIDIA RTX 3060 12GB).
   - Tuyệt đối không đề xuất Cloud GPU, Colab, Kaggle.
   - Kỷ luật commit: Bắt buộc commit sau MỖI yêu cầu (kèm nguồn yêu cầu, việc đã làm, số liệu thực nghiệm).
   - Tuyệt đối không in, log hay commit API keys; không gọi API thật trong khoảng 14:00 đến đêm.
3. **Tra cứu vị trí code tại [`PROJECT_MAP.md`](PROJECT_MAP.md):**
   - Không tự quét toàn bộ thư mục bằng lệnh duyệt file gây tràn context window. Tra cứu bảng chỉ mục để tìm đúng module cần sửa.
4. **Đối chiếu quyết định kỹ thuật tại [`DECISIONS.md`](DECISIONS.md):**
   - Đọc các ADRs (Architectural Decision Records) để hiểu lý do tại sao kiến trúc được thiết kế như hiện tại và những phương án cấm triển khai.

---

## 2. Các Phân Hệ Cốt Lõi Cần Chú Ý

- **AI Video Upscaler (`src/`):**
  - Động cơ chính: PyTorch CUDA FP16 Native (`batch_size=6` hoặc `8`) kết hợp OpenCV AVX2 I/O.
  - Phân hệ Dedup & DIS Optical Flow ([`docs/dedup-architecture.md`](../dedup-architecture.md)): Cắt video theo Keyframe GOP, scan mpdecimate + DIS, tái tạo 4K bằng Concat Demuxer.
  - Chuỗi ma trận màu BT.709 hai chiều ([`docs/video-pipeline.md`](../video-pipeline.md)): Giữ nguyên Limited YUV <-> Full RGB.
  - Mux an toàn: Bắt buộc cờ `-max_interleave_delta 0` và đối soát stream copy ([`docs/audit-and-safety.md`](../audit-and-safety.md)).
- **Subtitle Translation Engine (`subtrans/`):**
  - Tham khảo chi tiết tại [Subtrans AGENTS](../../subtrans/AGENTS.md).
  - Dùng 100% Cloud API Gemini qua Google GenAI SDK (luân phiên 3 API key).
  - Whisper STT Fallback & Auto-Align: Trộn âm thanh trung tâm 5.1 FC, lọc ảo giác và tiếng hét lặp ký tự, ghim timeline theo EN phòng thu qua Monotonic DTW.

---

## 3. Lệnh Thao Tác Nhanh Cho Agent

```powershell
# 1. Khởi động nhanh và kiểm tra tính toàn vẹn bộ nhớ ngữ cảnh:
python scripts/agent_context.py
python scripts/agent_context.py --check

# 2. Tra cứu cấu trúc code AST tiết kiệm token (không đọc full file):
python scripts/agent_context.py --symbols src/subtitles.py   # Xem classes/functions/line-numbers
python scripts/agent_context.py --symbols subtrans/         # Xem API surface của cả package
python scripts/agent_context.py --find snap_whisper         # Tìm nơi định nghĩa symbol trong 0.05s

# 3. Chạy test cô lập hoàn toàn offline (chống leak API key & socket):
python scripts/verify_offline.py

# 4. Chạy kiểm thử tự động toàn diện:
pytest
```

---

## 4. Bàn Giao Nhiệm Vụ (Task Handoff)

Sau khi hoàn thành công việc:
1. Chạy `pytest` bảo đảm 100% PASS.
2. Ghi chép nhật ký theo mẫu [`tasks/TEMPLATE.md`](tasks/TEMPLATE.md) lưu vào `docs/agents/tasks/`.
3. Cập nhật trạng thái mới nhất vào [`STATE.md`](STATE.md).
4. Commit Git riêng biệt với thông điệp rõ ràng theo đúng quy định tại [Root `AGENTS.md`](../../AGENTS.md).
