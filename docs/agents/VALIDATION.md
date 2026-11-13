# Kiểm thử và bằng chứng

Cập nhật 2026-10-31. Xem [STATE](STATE.md), [bản đồ](PROJECT_MAP.md), [quy tắc](../../AGENTS.md).

## Lệnh mặc định tại root Windows

```powershell
.\.venv\Scripts\python.exe scripts/agent_context.py --check
.\.venv\Scripts\python.exe scripts/verify_offline.py
git diff --check
```

Dùng Python trong `.venv`; Python trên PATH có thể không có pytest/dependencies. Không tự cài hoặc nâng phiên bản môi trường đang dùng cho production chỉ để bootstrap.

`verify_offline.py` dùng interpreter đang gọi, snapshot source/config/fixtures sang `scratch/agent-validation-*/repo`, chạy `pytest -m "not gpu"`. Giữ `pytest.log` và `junit.xml` trong thư mục của lần chạy để đối chiếu. Không copy cache, checkpoint, weights, .env, video hoặc pending queue.

Process test ẩn CUDA, loại biến key/token, tắt dotenv và plugin autoload, chặn socket Python trước collection. Đây là cách ly dành cho suite đã đọc; **không phải sandbox hệ điều hành cho code không tin cậy**, không chặn mạng của mọi subprocess. Test mới có gọi subprocess/API phải review trước khi chạy. Các test cần weights/video thật sẽ skip khi snapshot không có fixture; ghi rõ số lượng.

## Vì sao không chạy bare pytest trên repo production?

- `src/config.py` đăng ký atexit xóa lock; test subtitle guardrails thao tác lock thật nếu chạy trực tiếp.
- `TranslationCache()`, `KnowledgeBase()` mặc định có thể ghi cache production dù API bị mock; `throttle.call_with_retry` có thể sửa `subtrans/models.yaml`.
- `-m "not gpu"` chỉ loại marker; một số test còn lại vẫn tự chọn CUDA nếu nhìn thấy GPU.
- GPU soak test có thao tác xóa checkpoint cố định. Chỉ chạy khi phạm vi đã được người dùng cho phép.
- `subtrans.py check/models` gọi API thật. `upscale.py --dry-run` không bảo vệ các nhánh cleanup/subs-only/subs-pending chạy trước nó. Dry-run dịch còn ghi cast/output/cache/telemetry.

Nếu cần test nhanh, dùng snapshot và lệnh từ log, hoặc chỉ chạy trực tiếp file test đã xác minh hoàn toàn độc lập runtime (ví dụ test công cụ agent). Không áp dụng blanket “mọi test đều dùng tmp_path”.

## Nghiệm thu và báo cáo

Ghi nguyên lệnh, interpreter, baseline, phạm vi, số passed/failed/skipped/deselected, thời gian, đường dẫn bằng chứng. Test fail phải ghi đúng nguyên nhân; không đổi assertion chỉ để xanh.

Thay đổi bộ nhớ/tooling không tạo video: mux/frame-count/fps ghi **N/A — không tạo thành phẩm**, không ghi PASS. Khi sửa pipeline và có chạy video trong phạm vi được phép, phải đo mux integrity và `ffprobe -count_frames` trên source/output theo AGENTS; unit tests không thay thế kiểm chứng này.

`agent_context.py --check` kiểm tra file bắt buộc, link nội bộ dạng inline (bỏ qua anchors/remote/reference links), ngân sách byte. Cảnh báo freshness so sánh source hiện tại với baseline lưu trong STATE; không thay kiểm tra ngữ nghĩa bằng source/test. Hook kiểm tra working tree, không phải chứng nhận nội dung staged.

Kết thúc: cập nhật [handoff](tasks/TEMPLATE.md), STATE, kiểm tra diff/staged, commit riêng của yêu cầu. Giữ số đo cũ là lịch sử; không gọi API/GPU để tạo một báo cáo “PASS” ngoài phạm vi được phép.

