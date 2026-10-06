# Subtitle Translation Engine (subtrans) — AI Agent Guidelines

> **Chỉ dẫn kiến trúc và quy chuẩn phát triển cho phân hệ dịch phụ đề anime (`subtrans/`).**
> Tham chiếu từ: [AGENTS.md](file:///Users/xis108/Desktop/upscale/AGENTS.md) | Đo đạc: [docs/engineering-log.md](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md).

---

## 1. Mục Tiêu & Thứ Tự Ưu Tiên Cốt Lõi

1. **Không dịch sai / không bịa đặt:** Khi nghi vấn hoặc thiếu ngữ cảnh, chọn phương án an toàn nhất và gắn cờ `"uncertain": true`.
2. **Văn phong tiếng Việt tự nhiên khi đọc to:** Tránh dịch thô theo từng từ (word-for-word); giữ nhịp điệu đàm thoại tự nhiên của anime.
3. **Đồng nhất đại từ xưng hô và tên riêng toàn tập:** Bắt buộc tuân theo bảng nhân vật Stage A Cast Sheet (`out/<name>.cast.yaml`).
4. **Bảo tồn chuẩn mẫu SRT:** Xuất chuẩn định dạng phụ đề tương thích cao nhất với các trình phát video hiện đại (IINA, mpv, VLC).
5. **Tiếng Nhật là chân lý (Source of Truth):** Tiếng Anh chỉ dùng để bổ trợ xác định chủ ngữ, ngôi xưng hô, đối tượng khi câu thoại tiếng Nhật lược bỏ (zero-pronouns).

---

## 2. Quy Chuẩn Định Dạng SRT (Sample Fidelity)

* **Số dòng văn bản:** Mặc định xuất **chính xác 1 dòng văn bản / cue** (không tự ý ngắt 2 dòng; độ dài có thể lên tới 107 ký tự). Giới hạn mềm: 80 ký tự. Chỉ ngắt dòng khi người dùng chỉ định rõ `--wrap N`.
* **Dấu câu chuẩn hóa:**
  * Dấu ba chấm: Sử dụng đúng 3 chấm ASCII `...` (không dùng ký tự unicode `…`).
  * Ngắt lời / dở câu: Sử dụng gạch ngang `—` ở cuối câu.
  * Cảm thán / câu hỏi: Tối đa 2 dấu `!` hoặc `?` liên tiếp (ví dụ: `!!`, `??`).
* **Dấu ngoặc kép / trích dẫn:**
  * Chỉ gỡ dấu ngoặc kép khi nó bao trùm **toàn bộ câu thoại** (ví dụ: `"Xin chào!"` -> `Xin chào!`).
  * Không gỡ dấu ngoặc đơn của các từ/cụm từ trích dẫn ở đầu câu (ví dụ: `'Sư phụ' của cô...` phải được giữ nguyên vẹn).
* **Câu nói ngắt qua nhiều cue:** Phân tách tại ranh giới ngữ đoạn tự nhiên (cue trước kết thúc bằng `,` hoặc `...`, cue kế tiếp bắt đầu bằng chữ thường).

---

## 3. Kiến Trúc Phân Hệ `subtrans/`

```text
subtrans/
├── cli.py                  # Typer CLI entrypoint: check, models, run, compare, batch
├── pipeline.py             # Orchestrator: Stage A (Cast sheet) -> Stage B (Gemini/Local) -> Stage C (QA Audit)
├── config.py / config.yaml # Cấu hình RPM, RPD, chuỗi model, batch size (mặc định 100)
├── cast_sheet.py           # Stage A: Khám phá nhân vật, đại từ xưng hô, xuất out/<name>.cast.yaml
├── models.py               # Pydantic models: Cue, GlossaryTerm, AddressRules, Glossary (Schema v2)
├── parser.py               # pysubs2: lọc OP/ED/Signs, gỡ tag ASS, align time overlap JA-EN (>=50%)
├── gemini_backend.py       # Primary: Google GenAI SDK (JSON Schema, uncertain flag, Context +/-10)
├── local_backend.py        # Fallback: Ollama (translategemma:12b, keep_alive: 0 khi giải phóng)
├── review.py               # Optional Post-edit: Ollama (gemma3:12b)
├── throttle.py             # RateLimiter, Exponential Backoff + Jitter, phân loại lỗi 429
├── cache.py                # SQLite cache (.cache/translations.db) keyed by SHA-256
├── glossary.py             # Loader & Merger: Nạp Base glossary + Per-series Override
├── glossary.yaml           # Base glossary chuẩn: 12 style rules, address rules, 230+ anime terms
├── writer.py               # Xuất file .srt chuẩn mẫu (1 dòng/cue, soft cap 80)
└── report.py               # Stage C: Báo cáo QA audit Markdown (out/<name>.audit.md & --review-file)
```

---

## 4. Chính Sách Quota & Chuỗi Fallback Gemini

### A. Chuỗi Model Cấu Hình
* **Mặc định:** `gemini-3.8-flash`
* **Fallback tự động:** `["gemini-3.7-flash", "gemini-3.6-flash"]`
* **Lite model (cuối chuỗi Gemini):** `gemini-3.5-flash-lite`
* **Quy tắc Thinking:** `thinking_budget=0` cho dòng 3.8/3.7/3.6; `thinking_level=MINIMAL` cho flash-lite.
* **Cấm sử dụng:** Tuyệt đối không dùng dòng `gemini-2.*` cũ làm mặc định. Không dùng alias `gemini-flash-latest` nếu thiếu cờ `--allow-latest-alias`.

### B. Phân Loại Lỗi 429 & Cơ Chế Chuyển Model
1. **Lỗi 429 PerMinute (Rate Limit RPM) hoặc 503 (Spike tải tạm thời):**
   * Tạm dừng (pause) theo `retryDelay` hoặc exponential backoff + jitter (tối đa 5 lần thử lại).
   * **Bắt buộc retry trên CÙNG model** để bảo toàn tính nhất quán tuyệt đối của đại từ xưng hô toàn tập.
2. **Lỗi 429 PerDay (Daily Quota RPD cạn kiệt ngày):**
   * **Xoay vòng Key trước khi chuyển model (Multi-Key Rotation):** Nếu cấu hình nhiều key trong `.env` (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`), hệ thống tự động xoay sang key tiếp theo để tiếp tục dịch trên CÙNG model trước khi phải hạ cấp xuống model khác.
   * Khi TẤT CẢ các keys đều cạn kiệt trên model đó: Ngay lập tức ghi DB đánh dấu model cạn kiệt cho tới mốc reset giờ Thái Bình Dương (00:00 PT = 14:00 VN) và **chuyển sang model kế tiếp trong chuỗi**.
3. **Fallback sang Ollama Local (`translategemma:12b`):**
   * Chỉ kích hoạt khi TOÀN BỘ chuỗi Gemini cạn kiệt VÀ **không có tiến trình Upscale GPU nào đang chạy**.
   * Nếu GPU đang bận upscale, hệ thống đánh dấu các câu là `[Chưa dịch]` để bảo vệ VRAM và checkpoint, chờ người dùng tiếp tục sau.
   * Khi nạp Ollama, luôn giải phóng bộ nhớ bằng `keep_alive: 0` ngay khi dịch xong.

### C. Cơ Chế Stage A Cast Sheet Unconditional Inclusion
* Bảng quan hệ nhân vật và xưng hô sinh ra từ Stage A luôn được truyền vào prompt của mọi batch (kể cả khi câu thoại trong batch đó không nhắc tên nhân vật), giúp model giải quyết triệt để vấn đề zero-pronouns của tiếng Nhật.

---

## 5. Lệnh CLI Cơ Bản

```bash
# Kiểm tra môi trường, API keys và danh sách model khả dụng
python3 subtrans.py check

# Dịch một tập phụ đề đơn lẻ (tự động Stage A -> B -> C)
python3 subtrans.py run path/to/ja.srt --en path/to/en.ass -o out/

# Chạy thử nghiệm KHÔNG tốn quota API (Mock Dry-Run)
python3 subtrans.py run path/to/ja.srt --en path/to/en.ass -o out/ --dry-run

# Chạy test suite chuyên biệt của subtrans
./venv/bin/pytest subtrans/tests/
```
