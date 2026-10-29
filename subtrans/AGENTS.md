# Subtitle Translation Engine (subtrans) — AI Agent Guidelines

> **Chỉ dẫn kiến trúc và quy chuẩn phát triển cho phân hệ dịch phụ đề anime (`subtrans/`).**
> Tham chiếu từ: [AGENTS.md](../AGENTS.md) | Đo đạc: [docs/engineering-log.md](../docs/engineering-log.md).

## 0. Nạp lại ngữ cảnh và nguồn kiểm chứng

- Bắt đầu/resume/subagent: đọc [START_HERE](../docs/agents/START_HERE.md), handoff của task và file này; đối chiếu code trước khi sửa. Không dùng lịch sử chat làm bộ nhớ duy nhất.
- Đối chiếu ngày **2026-10-29**: `TranslationPipeline.__init__` lấy thứ tự model từ `models.yaml:chain`, lọc `local`/`preview`, rồi sắp Lite trước (`normal`) hoặc Flash trước (`high`). `config.yaml` giữ batch, timeout và tùy chọn; CLI override trong `cli.py`. Model/quota ở đây là cấu hình repo, chưa xác minh khả dụng hay hạn mức nhà cung cấp trong phiên này.
- Khởi động offline: dùng `--help`; **không chạy `check`/`models`**, vì chúng gọi API thật. `check` còn hiển thị mảnh key. Không đọc/in `.env` hay API key; mọi test dùng stub.
- `--dry-run` không gọi API nhưng vẫn ghi cast giả, subtitle/audit, telemetry và có thể sửa `pending_subs.json`. Dùng output riêng. `--cache-db` chỉ đổi cache pipeline; backend/RAG còn tạo DB mặc định. Khi test phải cách ly `cache.DB_PATH`, `rag.KNOWLEDGE_DB_PATH` và file quota; không sửa/xóa dữ liệu vận hành để kiểm thử.
- `--local`/`--review` và cấu hình Ollama là dấu vết cũ, không phải backend được phép dùng. Cổng chất lượng tại `pipeline.py`: bản dịch thật >=95%, nghi vấn <10%; còn lại xuất `.vi.partial.*` và ghi hàng đợi pending.

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
* **Ghim mốc thời gian theo phụ đề tiếng Anh phòng thu (Auto-Align):**
  * `--auto-align-en` mặc định bật, nhưng `parse_and_align_subtitles` chỉ ghim khi tên file chứa `whisper` hoặc `is_whisper=True`; track JA phòng thu giữ timeline nguồn sau offset.
  * `snap_whisper_cues_to_en` dùng `dtw_auto_align.get_dtw_alignment_map`: DTW đơn điệu toàn cục (1:1, 1:2, 2:1), tự động phát hiện và loại bỏ các cue Whisper mồ côi (`is_dropped_sound=True`, `is_skipped=True`) khi unmapped và cách câu EN gần nhất > 2500ms hoặc cue ngắn <= 6 ký tự / chứa từ khóa âm thanh.
  * Pass 3 đảm bảo tính đơn điệu và khoảng cách tối thiểu 50ms được cô lập trên `active_cues` (bỏ qua các cue đã bị drop/skip để không đẩy lùi mốc thời gian hợp lệ), đồng thời bảo toàn mốc thời gian Dual Dialogue khi 2 cue đồng thời (`abs(start diff) <= 15ms` và cùng end).
* **Bộ lọc chống ảo giác Whisper & tiếng hét lặp token:**
  * `WHISPER_HALLUCINATION_REGEX` nhận diện và triệt tiêu các mẫu ảo giác phổ biến của Whisper tiếng Nhật ở đoạn nhạc OP/ED hoặc khoảng lặng (`ご視聴ありがとうございました`, `チャンネル登録`, `高評価`, `字幕`, `提供`, `ご覧いただきありがとうございました`).
  * `SCREAM_OR_REPEAT_LOOP_REGEX` nhận diện các vòng lặp token hoặc tiếng hét kéo dài (1 ký tự lặp >= 6 lần, cụm 1-3 ký tự lặp >= 4 lần không chứa Kanji) để tự động đánh dấu sound cue loại bỏ khi `--sound-cues drop`.
* **Tối ưu hóa hiển thị phụ đề đa thoại & chống trôi chữ (Dual Dialogue & Playback Optimization):**
  * `writer.optimize_cues_for_playback` dùng độ chênh start và tỷ lệ overlap để gộp thoại song song 2 dòng (`- Lời 1\n- Lời 2`), mục tiêu tối thiểu 2.2s trước bước khử chồng lấn. Ngưỡng cụ thể đọc trong code.
  * Tự động gộp các mảnh câu liên tiếp nói quá nhanh (< 1.2s/mảnh) thành một câu liền mạch; bảo đảm thời lượng đọc tối thiểu (>= 1.2s - 1.4s) và khoảng nghỉ 40ms chống giật flicker trên các trình phát video (mpv, VLC, v.v.).
* **Bóc tách phụ đề đa người nói Closed Captions (Multi-Speaker CC Splitting):**
  * Bóc tách các event CC dạng gộp `-（Người 1）Thoại 1\N-（Người 2）Thoại 2` thành các event độc lập có cùng timeline qua `split_multi_speaker_cc_events` trong `subtrans/parser.py`.
  * Bóc tiền tố gạch ngang đầu dòng `^[-–—]\s*` và tên người nói qua `strip_speaker_prefix`, lưu vào trường `Cue.speaker`. Mỗi câu được dịch riêng lẻ và tự động hợp nhất thành phụ đề song song 2 dòng qua bộ Dual Dialogue tại `subtrans/writer.py`. Chỉ dịch chuyển +10ms cho các cue không bị skip.
* **Chống dịch đúp tiếng cười & Triệt tiêu dị bản trùng lặp trong Dual Dialogue:**
  * Các từ tượng thanh tiếng cười anime (`ふはは+`, `ははは+`, `フハハ+`, `ククク+`, `ふふふ+`...) được nhận diện chính xác là interjection/sound cue trong `subtrans/parser.py`, tránh bị gán nhầm câu thoại tiếng Anh overlap của nhân vật khác.
  * Bộ `optimize_cues_for_playback` trong `subtrans/writer.py` tự động phát hiện các câu xuất hiện cùng lúc chia sẻ tiền tố chung dài (common prefix >= 15 chars) hoặc độ trùng lặp từ vựng >= 60% để chỉ giữ lại 1 câu hoàn chỉnh duy nhất, tuyệt đối không gộp thành 2 dòng Dual Dialogue lặp lại nhau.
* **Số dòng văn bản:** Thoại đơn mặc định 1 dòng/cue; ngoại lệ là Dual Dialogue ở trên hoặc `--wrap N`. Giới hạn mềm 80 ký tự, báo cáo thêm các câu vượt 107 ký tự.
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
├── config.py / config.yaml # Loader, batch size 100, timeout và tùy chọn pipeline
├── models.yaml            # Chuỗi model và số liệu quota đã lưu; không phải tra cứu API hiện thời
├── cast_sheet.py           # Stage A: Khám phá nhân vật, đại từ xưng hô, xuất out/<name>.cast.yaml
├── models.py               # Pydantic models: Cue (với CoT metadata), GlossaryTerm, AddressRules, Glossary
├── parser.py               # pysubs2: lọc OP/ED/Signs, gỡ tag ASS, align time overlap JA-EN (>=50%), Auto-Align EN
├── dtw_auto_align.py       # DTW đơn điệu và ánh xạ timeline JA-EN
├── dynamic_cast.py         # Hysteresis: baseline/arc overrides, ghi nhận thay đổi nhất thời
├── ja_morph_tagger.py      # Regex nhận diện đại từ, kính ngữ, sắc thái JA đưa vào prompt
├── gemini_backend.py       # Primary: Google GenAI SDK (Schema "Suy luận trước, dịch sau", RAG context, CoT)
├── throttle.py             # RateLimiter, Exponential Backoff + Jitter, phân loại lỗi 429
├── cache.py                # SQLite cache (.cache/translations.db) keyed by SHA-256 (hỗ trợ rag_hash và stage)
├── glossary.py             # Loader & Merger: Nạp Base glossary + Per-series Override
├── glossary.yaml           # Base glossary chuẩn: 13 style rules, address rules, 230+ anime terms
├── writer.py               # Xuất file .srt chuẩn mẫu (1 dòng/cue, soft cap 80, CPS auto-fix, zero-overlap)
└── report.py               # Stage C: Báo cáo QA audit Markdown (out/<name>.audit.md & --review-file)
```

> **Quy định LLM Cốt Lõi:** **100% sử dụng Cloud API (Gemini Cloud: Flash-Lite, Flash), TUYỆT ĐỐI BỎ QUA MODEL LOCAL (Ollama / TranslateGemma)** theo chỉ định trực tiếp từ Người dùng.  
> **Schema "Suy luận trước, dịch sau":** `SubtitleOutputItem` đặt các trường `speaker`, `addressee`, `mode`, `en_conflict` lên TRƯỚC trường bản dịch `vi`, giúp mô hình giải quyết ngữ cảnh nhân vật và zero-pronouns trước khi sinh bản dịch tiếng Việt.  
> **Linter V1 Hậu Kiểm:** Tự động bắt lỗi sót ký tự CJK, vi phạm CPS > 20 (trần cứng > 25), lỗi Unicode chưa chuẩn hóa NFC, mất tên Romaji khi gọi tên, và các từ Hán-Việt dịch thô cấm dùng.  
> **Smart Cloud Cascade (P2):** Draft Pass bằng `gemini-3.5-flash-lite` (tiết kiệm quota, 500 RPD/key), tự động tính điểm độ khó $S(cue)$ (sau khi chạy auto-fix) và gom tối đa 20% số cue khó gửi vào Repair Pass của `gemini-3.8-flash`. Cổng Re-lint so sánh theo danh tính vi phạm `_ident`, chặn trôi dạt ngữ nghĩa và lệch chữ số.  
> **Lightweight RAG & Error Bank (P3):** Lưu trữ câu song ngữ (`gold_units` có status `draft`, `reviewed`, `gold`) và quy tắc bẫy lỗi (`error_bank`). Chỉ nạp `gold`/`reviewed` vào prompt, loại trừ tập đang đo (`exclude_episode`), cách ly cache nháp bằng `rag_hash`.

---

## 4. Chính Sách Quota & Chuỗi Fallback Gemini

### A. Chuỗi Model Cấu Hình
* **Mặc định cấu hình (`--subs-quality normal`, tài liệu cũ gọi Fast):** `gemini-3.5-flash-lite`; 500 RPD/key là số liệu lưu trong repo, không bảo đảm quota hiện thời.
* **Chế độ Chất lượng cao (`--subs-quality high`):** `gemini-3.8-flash`.
* **Fallback cấu hình:** `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`; xem `models.yaml:chain`.
* **Quy tắc Thinking:** `thinking_level=MINIMAL` cho dòng `gemini-3.5-flash-lite`; `thinking_budget=0` cho dòng 3.8/3.7/3.6.
* **Cấm sử dụng:** Tuyệt đối không dùng dòng `gemini-2.*` cũ làm mặc định. Không dùng alias `gemini-flash-latest` nếu thiếu cờ `--allow-latest-alias`.

### B. Phân Phối Tải & Xử Lý Lỗi 429
1. **Phân bổ Luân phiên Round-Robin 3 Key:**
   * Tự động luân phiên đều các request giữa 3 API key (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`) cấu hình trong `.env`, chống nghẽn RPM trong các tập anime có nhiều câu thoại dồn dập.
2. **Lỗi 429 PerMinute (Rate Limit RPM) hoặc 503 (Spike tải tạm thời):**
   * Retry cùng model theo `retryDelay` hoặc backoff+jitter; giới hạn từng lớp nằm trong `gemini_backend._call_api` và `throttle.call_with_retry`. Sau khi hết retry, pipeline hiện có thể chuyển model; không suy ra tính nhất quán từ việc retry.
3. **Lỗi 429 PerDay (Daily Quota RPD cạn kiệt ngày):**
   * **Xoay vòng Key trong nhóm:** Khi 1 key cạn hạn ngạch ngày, hệ thống tự động loại bỏ key đó và tiếp tục với các key còn lại.
   * Khi TẤT CẢ các keys đều cạn kiệt trên model đó: Ghi DB tới mốc reset 00:00 PT (14:00 hoặc 15:00 VN tùy DST), rồi **chuyển model kế tiếp**; tính giờ bằng `cache.get_next_pt_reset`.
4. **Cơ Chế Dịch Dự Phòng Whisper (Speech-to-Text Fallback):**
   * Nếu file video nguồn không có track tiếng Nhật (chỉ có audio tiếng Nhật và phụ đề tiếng Anh), hệ thống tự động kích hoạt Whisper để chuyển đổi âm thanh thành track `.ja.srt` tạm thời (`src/subtitles.py`), sau đó căn chỉnh khớp mốc thời gian qua `--auto-align-en` và tiến hành dịch EN+JA -> VI trọn vẹn.
5. **Không dùng Model Local & Dừng Chạy Êm (Graceful Exit):**
   * **Tuyệt đối không dùng model local (Ollama / TranslateGemma):** Bỏ toàn bộ fallback cục bộ.
   * Khi TOÀN BỘ chuỗi Gemini cạn kiệt hạn mức ngày (RPD): Hệ thống tự động dừng êm (Graceful Exit), in rõ thời gian reset (14:00 giờ VN PDT hoặc 15:00 giờ VN PST sau ngày 1/11). Toàn bộ câu thoại đã dịch được bảo toàn an toàn trong cache SQLite để phiên chạy kế tiếp tiếp tục ngay lập tức mà không tiêu tốn lại bất kỳ request nào.

### C. Cơ Chế Stage A Cast Sheet Unconditional Inclusion
* Bảng quan hệ nhân vật và xưng hô sinh ra từ Stage A luôn được truyền vào prompt của mọi batch (kể cả khi câu thoại trong batch đó không nhắc tên nhân vật), giúp model giải quyết triệt để vấn đề zero-pronouns của tiếng Nhật.

### D. Chốt Chặn Toàn Vẹn Cache & Cấm Cache Chuỗi Thử Nghiệm Dry-Run
* Tuyệt đối không lưu các bản ghi giả lập `[DRY-RUN]` hoặc chuỗi `[Chưa dịch]` vào SQLite cache `subtrans/.cache/translations.db`. Hàm `get_cue` và `set_cue` tự động loại bỏ các chuỗi này để ngăn chặn hiện tượng ô nhiễm cache khi người dùng chạy lệnh dịch thật sau lệnh dry-run.

---

## 5. Lệnh CLI Cơ Bản

```powershell
# Khởi động offline; CLI cần subcommand run và đầu vào subtitle, không phải video:
python subtrans.py --help
.\subtrans.bat run path/to/ja.srt --en path/to/en.ass -o out/

# Chạy trực tiếp qua Python CLI:
python subtrans.py run path/to/ja.srt --en path/to/en.ass -o out/ --auto-align-en

# Chạy thử nghiệm KHÔNG tốn quota API (Mock Dry-Run):
python subtrans.py run path/to/ja.srt --en path/to/en.ass -o out/dry-run/ --dry-run

# Chạy test suite chuyên biệt của subtrans:
python -m pytest subtrans/tests/ -m "not gpu"
```
