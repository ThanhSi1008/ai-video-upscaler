# Subtitle Translation Engine (subtrans) — AI Agent Guidelines

> **Chỉ dẫn kiến trúc và quy chuẩn phát triển cho phân hệ dịch phụ đề anime (`subtrans/`).**
> Tham chiếu từ: [AGENTS.md](file:///Users/xis108/Desktop/upscale/AGENTS.md) | Đo đạc: [docs/engineering-log.md](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md).

---

## 1. Mục Tiêu & Thứ Tự Ưu Tiên Cốt Lõi

1. **Không dịch sai / không bịa đặt:** Khi nghi vấn hoặc thiếu ngữ cảnh, chọn phương án an toàn nhất và gắn cờ `"uncertain": true`.
2. **Văn phong tiếng Việt tự nhiên khi đọc to:** Tránh dịch thô theo từng từ (word-for-word); giữ nhịp điệu đàm thoại tự nhiên của anime.
3. **Đồng nhất đại từ xưng hô và bảo tồn tên riêng Romaji khi gọi tên:** Bắt buộc tuân theo bảng nhân vật Stage A Cast Sheet (`out/<name>.cast.yaml`). Khi lời thoại tiếng Nhật gọi đích danh ai đó bằng tên riêng (vocative, e.g. `フラン` -> Fran, `ジャン` -> Jean, `ネルさん` -> chị Nell, `玲琳` -> Reirin), **bắt buộc giữ nguyên tên riêng Romaji trong câu phụ đề tiếng Việt**. Tuyệt đối không thay thế tên riêng bằng đại từ xưng hô chung chung (anh/em/cậu/cô/mày) để triệt tiêu lỗi đoán sai đại từ và giữ đồng bộ âm thanh - chữ viết.
4. **Bảo tồn chuẩn mẫu SRT:** Xuất chuẩn định dạng phụ đề tương thích cao nhất với các trình phát video hiện đại (IINA, mpv, VLC).
5. **Tiếng Nhật là chân lý (Source of Truth):** Tiếng Anh chỉ dùng để bổ trợ xác định chủ ngữ, ngôi xưng hô, đối tượng khi câu thoại tiếng Nhật lược bỏ (zero-pronouns).

---

## 2. Quy Chuẩn Định Dạng SRT & Xưng Hô (Sample Fidelity)

* **Bảo tồn tên riêng Romaji khi xưng hô gọi tên (Vocative Name Retention):**
  * Khi lời thoại tiếng Nhật gọi tên nhân vật, phụ đề tiếng Việt phải giữ nguyên tên Romaji (e.g. "Fran, cẩn thận!", "Jean, chạy đi!", "Cảm ơn chị Nell", "Reirin, chuyện gì vậy?").
  * Có thể kèm từ xưng hô thân mật/kính ngữ tự nhiên ("chị Nell", "bác Garrus"), nhưng **tên riêng Romaji bắt buộc phải xuất hiện**, không được triệt tiêu thành các đại từ chung chung ("cậu", "anh", "em", "mày").
* **Chống dịch đúp tiếng cười & Triệt tiêu dị bản trùng lặp trong Dual Dialogue:**
  * Các từ tượng thanh tiếng cười anime (`ふはは+`, `ははは+`, `フハハ+`, `ククク+`, `ふふふ+`...) được nhận diện chính xác là interjection/sound cue trong `subtrans/parser.py`, tránh bị gán nhầm câu thoại tiếng Anh overlap của nhân vật khác.
  * Bộ `optimize_cues_for_playback` trong `subtrans/writer.py` tự động phát hiện các câu xuất hiện cùng lúc chia sẻ tiền tố chung dài (common prefix >= 15 chars) hoặc độ trùng lặp từ vựng >= 60% để chỉ giữ lại 1 câu hoàn chỉnh duy nhất, tuyệt đối không gộp thành 2 dòng Dual Dialogue lặp lại nhau.
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
├── eval/                   # Evaluation Harness: metrics, regression_suite.yaml, pronoun_suite.yaml, runner
├── cascade.py              # Smart Cloud Cascade: S(cue) difficulty scoring & Flash repair pass
├── rag.py                  # Lightweight Subtitle RAG & Error Bank: gold_units & error_bank (.cache/knowledge.db)
├── linter.py               # Linter V1: CJK leak, CPS reading speed, Unicode NFC, Romaji vocatives, banned calques
├── cli.py                  # Typer CLI entrypoint: check, models, run, compare, batch, eval, rag
├── pipeline.py             # Orchestrator: Stage A (Cast) -> Stage B (Gemini Cloud) -> Stage 2c (Cascade) -> Stage 4 (Linter/Write) -> Stage C (QA)
├── config.py / config.yaml # Cấu hình RPM, RPD, chuỗi model, batch size (mặc định 100)
├── cast_sheet.py           # Stage A: Khám phá nhân vật, đại từ xưng hô, xuất out/<name>.cast.yaml
├── models.py               # Pydantic models: Cue (với CoT metadata), GlossaryTerm, AddressRules, Glossary
├── parser.py               # pysubs2: lọc OP/ED/Signs, gỡ tag ASS, align time overlap JA-EN (>=50%)
├── gemini_backend.py       # Primary: Google GenAI SDK (Schema "Suy luận trước, dịch sau", RAG context, CoT)
├── local_backend.py        # [DEPRECATED theo chỉ đạo Người dùng: 100% Cloud API, không dùng model local]
├── throttle.py             # RateLimiter, Exponential Backoff + Jitter, phân loại lỗi 429
├── cache.py                # SQLite cache (.cache/translations.db) keyed by SHA-256
├── glossary.py             # Loader & Merger: Nạp Base glossary + Per-series Override
├── glossary.yaml           # Base glossary chuẩn: 13 style rules, address rules, 230+ anime terms
├── writer.py               # Xuất file .srt chuẩn mẫu (1 dòng/cue, soft cap 80, CPS auto-fix, zero-overlap)
└── report.py               # Stage C: Báo cáo QA audit Markdown (out/<name>.audit.md & --review-file)
```

> **Quy định LLM Cốt Lõi:** **100% sử dụng Cloud API (Gemini Cloud: Flash-Lite, Flash), TUYỆT ĐỐI BỎ QUA MODEL LOCAL (Ollama / TranslateGemma)** theo chỉ định trực tiếp từ Người dùng.
> **Schema "Suy luận trước, dịch sau":** `SubtitleOutputItem` đặt các trường `speaker`, `addressee`, `mode`, `en_conflict` lên TRƯỚC trường bản dịch `vi`, giúp mô hình giải quyết ngữ cảnh nhân vật và zero-pronouns trước khi sinh bản dịch tiếng Việt.
> **Linter V1 Hậu Kiểm:** Tự động bắt lỗi sót ký tự CJK, vi phạm CPS > 20, lỗi Unicode chưa chuẩn hóa NFC, mất tên Romaji khi gọi tên, và các từ Hán-Việt dịch thô cấm dùng.
> **Smart Cloud Cascade (P2):** Draft Pass bằng `gemini-3.5-flash-lite` (tiết kiệm quota, 2000 RPD), tự động tính điểm độ khó $S(cue)$ và gom 5–15% số cue khó/nghi vấn gửi vào Repair Pass của `gemini-3.8-flash` dưới dạng biên tập viên trưởng.
> **Lightweight RAG & Error Bank (P3):** Lưu trữ hơn 4.100+ câu mẫu song ngữ (`gold_units`) và 7 quy tắc bẫy lỗi (`error_bank`) trong SQLite để tự động trích xuất Few-shot và cảnh báo vào prompt dịch theo từng series.


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

### D. Chốt Chặn Toàn Vẹn Cache & Cấm Cache Chuỗi Thử Nghiệm Dry-Run
* Tuyệt đối không lưu các bản ghi giả lập `[DRY-RUN]` hoặc chuỗi `[Chưa dịch]` vào SQLite cache `subtrans/.cache/translations.db`. Hàm `get_cue` và `set_cue` tự động loại bỏ các chuỗi này để ngăn chặn hiện tượng ô nhiễm cache khi người dùng chạy lệnh dịch thật sau lệnh dry-run.

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
