# BÁO CÁO KỸ THUẬT TOÀN DIỆN: PHÂN HỆ DỊCH THUẬT PHỤ ĐỀ ANIME TIẾNG VIỆT (`subtrans`)
**Dự án:** AI Video Upscaler & Subtitle Translation Engine (`ai-video-upscaler`)  
**Thiết bị vận hành:** Apple Silicon MacBook Pro M3 Pro (11-core CPU, 14-core GPU, 18GB Unified Memory)  
**Môi trường:** 100% CỤC BỘ trên macOS (Python 3.12, PyTorch Metal MPS, FFmpeg, Google GenAI SDK, SQLite)  
**Mục tiêu tài liệu:** Cung cấp đầy đủ bối cảnh hệ thống, kiến trúc thực thi, toàn bộ các bài toán kỹ thuật đã giải quyết, các lỗi thực nghiệm đã gặp và phương án khắc phục, cùng các vấn đề mở để AI Nghiên cứu mới đề xuất lộ trình tối ưu hóa chuyên sâu.

---

## 1. Bối Cảnh Hệ Thống & Triết Lý Thiết Kế Cốt Lõi

### 1.1. Mục Tiêu của Phân Hệ `subtrans`
Phân hệ `subtrans` có nhiệm vụ tự động hóa toàn bộ quy trình dịch phụ đề anime từ tiếng Nhật (JA) sang tiếng Việt (VI) với độ chuẩn xác cao, văn phong tự nhiên khi đọc to, đồng nhất tuyệt đối về đại từ xưng hô vai vế, và bảo toàn tương thích định dạng hiển thị với các trình phát video hiện đại (IINA, mpv, VLC, Apple TV).

### 1.2. Triết Lý 2 Nguồn Đầu Vào (Dual-track Input Paradigm)
Dịch phụ đề anime có một đặc thù ngôn ngữ cực kỳ phức tạp: **Hiện tượng lược bỏ đại từ và chủ ngữ trong tiếng Nhật (Zero-Pronouns)**.
* **Tiếng Nhật (Source of Truth - Chân lý ngữ nghĩa):** Cung cấp sắc thái giọng điệu (nghiêm túc, bông đùa, khinh miệt), đại từ nhân xưng gốc (`watashi`, `boku`, `ore`, `atashi`), kính ngữ (`-sama`, `-san`, `-kun`, `-chan`), và cảm xúc thực sự của nhân vật.
* **Tiếng Anh (Disambiguation Reference - Tham chiếu đối soát):** Bản dịch tiếng Anh chính thức từ nhà phát hành (Crunchyroll, B-Global, Netflix) đã được đội ngũ bản địa hóa bổ sung chủ ngữ, tân ngữ, và bối cảnh đối tượng giao tiếp.
* **Quy tắc phối hợp:** Hệ thống lấy câu tiếng Nhật làm gốc để dịch, nhưng truyền song song câu tiếng Anh tương ứng để LLM xác định chính xác "ai đang nói với ai", triệt tiêu lỗi suy diễn sai đại từ do tiếng Nhật lược bỏ.

### 1.3. Hạ Tầng & Ràng Buộc Kỹ Thuật
1. **Thiết bị:** Chạy trên macOS M3 Pro 18GB RAM. Không dùng Cloud GPU, không dùng Colab.
2. **Tiến trình độc quyền tài nguyên:** Khi tiến trình Video Upscale đang chiếm dụng GPU Metal (`/tmp/upscale_gpu.lock`), phân hệ dịch phụ đề ưu tiên gọi Gemini API đám mây (qua mạng) hoặc tạm dừng Ollama Local để tránh tràn VRAM.
3. **Bảo toàn Quota API ban ngày:** Sử dụng cơ chế xoay vòng nhiều API Key (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`,...) để mở rộng hạn ngạch, kèm bộ nhớ đệm SQLite SHA-256 (`translations.db`) để không bao giờ dịch lại một câu đã có trong kho.

---

## 2. Kiến Trúc Pipeline 3 Giai Đoạn (3-Stage Translation Pipeline)

```text
               [File JA.srt / JA.ass]   +   [File EN.ass / EN.srt]
                          │                           │
                          ▼                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: PARSE, CLEAN, SPLIT & ALIGN (subtrans/parser.py)                 │
│  • Gỡ bỏ mã màu/style ASS, lọc OP/ED/Signs, gỡ dấu ngoặc đơn Closed Captions │
│  • Bóc tách sự kiện Closed Captions gộp người nói (split_multi_speaker)  │
│  • Lọc từ tượng thanh/tiếng cười/tiếng thở thuần túy (is_pure_sound_cue) │
│  • Căn chỉnh lệch mốc thời gian tự động (--auto-offset 4-tier check)     │
│  • So khớp thời gian chồng lấn (Time Overlap Matching >= 50%)            │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE A: CAST SHEET DISCOVERY (subtrans/cast_sheet.py)                   │
│  • Quét toàn bộ tập, nhận diện dàn nhân vật xuất hiện                     │
│  • Xác định giới tính, vai vế, quan hệ xưng hô cặp đôi (speaks_to)       │
│  • Xuất file out/<name>.cast.yaml (nạp vô điều kiện vào mọi batch dịch) │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE B: BATCH TRANSLATION & RESILIENCE (subtrans/gemini_backend.py)     │
│  • Cửa sổ ngữ cảnh trượt (Rolling Context Window: +/- 10 cues)           │
│  • Base Glossary Schema v2 (glossary.yaml) + Per-series Override YAML    │
│  • Structured JSON Output (Pydantic schema: id, vi, uncertain)           │
│  • Multi-Key Rotation khi gặp lỗi 429 Quota Exceeded                     │
│  • Cơ chế Tách đôi Batch (Bisect) khi validation hoặc parse JSON thất bại │
│  • Fallback sang Ollama Local (translategemma:12b) khi cạn kiệt API     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE C: PLAYBACK OPTIMIZATION & QA AUDIT (subtrans/writer.py & report.py)│
│  • Chuẩn hóa dấu câu (ASCII 3 dots '...', ngắt lời '—', gỡ ngoặc kép)   │
│  • Anti-Duplicate & Roll-up: Triệt tiêu câu lặp lại và dị bản trùng lặp  │
│  • Dual Dialogue: Gộp 2 người nói cùng lúc (- Lời 1\n- Lời 2) >= 2.2s    │
│  • Rapid Fragments: Gộp các mảnh câu nói nhanh liên tiếp (< 1.2s/mảnh)   │
│  • Strict Zero-Overlap: Ép c.end <= next.start - 40ms (chống giật màn)  │
│  • QA Audit Report Markdown: Kiểm định tỷ lệ dịch thật >= 95%           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
                        [File Thành Phẩm .vi.srt]
```

---

## 3. Toàn Bộ Những Gì Đã Triển Khai & Hoàn Thiện

### 3.1. Hệ Thống Thuật Ngữ Đa Tầng (Glossary Schema v2)
* **Base Glossary (`subtrans/glossary.yaml`):** Bộ quy tắc nền tảng áp dụng chung cho mọi anime (hơn 230 thuật ngữ anime kinh điển, 13 quy tắc phong cách `style_rules`, bảng xưng hô ngôi 1, ngôi 2, kính ngữ `address_rules`).
* **Per-series Override (`glossaries/glossary.<anime>.yaml`):** Bảng thuật ngữ chuyên biệt cho từng tác phẩm:
  * `glossary.futsutsuka.yaml`: Hệ thống danh xưng hậu cung cổ trang hoàng gia (Hoàng đế, Quý phi, Thục phi, Thuần phi, nữ quan), xưng hô tráo đổi linh hồn (Reirin x Keigetsu).
  * `glossary.tenken.yaml`: Hệ thống RPG isekai, kỹ năng ma pháp, xưng hô Sư phụ - Fran - Urushi - Jean - Nell - Amanda.
* **Bộ lọc động (Dynamic Term Filtering):** Chỉ gửi những thuật ngữ thực sự xuất hiện trong batch tiếng Nhật / tiếng Anh hiện tại để tiết kiệm token và tránh làm loãng sự tập trung của LLM.

### 3.2. Căn Chỉnh Thời Gian Tự Động 4 Tầng (`--auto-offset`)
Giải quyết hiện tượng lệch mốc thời gian giữa bản phụ đề tiếng Nhật (từ truyền hình NHK/Tokyo MX) và bản phụ đề tiếng Anh (từ web streaming):
* Quét phân vị 5 đoạn (Quintiles) trên toàn bộ video.
* Kiểm tra 4 điều kiện nghiêm ngặt:
  1. Độ lệch tuyệt đối `|offset| <= 500ms`.
  2. Số cặp câu đối ứng `>= 30`.
  3. Độ trôi phân vị (Quintile drift) `<= 500ms` (chứng minh không có đoạn chèn quảng cáo làm lệch lũy tiến).
  4. Tỷ lệ khớp thời gian tăng tối thiểu `>= 15 điểm phần trăm`.
* Nếu thỏa mãn, tự động bù `shift(ms=offset)` cho phụ đề tiếng Nhật trước khi căn chỉnh ngữ nghĩa.

### 3.3. Bảo Tồn Tên Riêng Romaji trong Xưng Hô (Vocative Name Retention)
* **Quy chuẩn:** Bất cứ khi nào lời thoại tiếng Nhật gọi đích danh tên nhân vật (`フラン` -> Fran, `ジャン` -> Jean, `ネルさん` -> chị Nell, `玲琳` -> Reirin), phụ đề tiếng Việt **bắt buộc phải giữ nguyên tên Romaji trong câu thoại**.
* **Ý nghĩa:** Tiếng Nhật thường dùng tên riêng thay cho ngôi thứ hai. Việc giữ tên Romaji triệt tiêu 100% lỗi đoán mò đại từ (anh/em/cậu/cô/mày) và đảm bảo sự đồng bộ tuyệt đối giữa âm thanh người xem nghe thấy từ diễn viên lồng tiếng và chữ hiển thị trên màn hình.

### 3.4. Bộ Tối Ưu Hiển Thị Playback Toàn Diện (`optimize_cues_for_playback`)
Phụ đề trích xuất từ Closed Captions (CC) thường bị chẻ vụn, nhấp nháy hoặc đè chữ. Module `subtrans/writer.py` áp dụng 5 tầng chuẩn hóa:
1. **Chuẩn 1 dòng văn bản / cue:** Không tự ý ngắt dòng, soft cap 80 ký tự.
2. **Kéo dài thời lượng tối thiểu:** Câu đơn thoại tối thiểu 1.2s – 1.4s; câu song thoại tối thiểu 2.2s để mắt người kịp đọc.
3. **Gộp mảnh câu nói nhanh (Rapid Fragments):** Các câu ngắt dở dang (< 1.2s/mảnh) được gộp lại thành một câu liền mạch tự nhiên.
4. **Hợp nhất đối thoại song song (Dual Dialogue):** Hai người nói cùng lúc hoặc chênh nhau <= 800ms với overlap >= 400ms được gộp thành 2 dòng có gạch đầu dòng (`- Lời 1\n- Lời 2`).
5. **Khoảng nghỉ chống giật (Strict Zero-Overlap & 40ms Gap):** Ép khoảng cách giữa 2 block độc lập tối thiểu 40ms (`c.end <= next.start - 40ms`), ngăn chặn triệt để hiện tượng trình phát video (IINA, mpv, VLC) hiển thị đè 3–4 dòng che kín màn hình.

---

## 4. Các Lỗi Kỹ Thuật Đã Gặp Trong Thực Tế & Cách Khắc Phục

| STT | Tên Lỗi / Hiện Tượng | Nguyên Nhân Kỹ Thuật Gốc Rễ | Giải Pháp Kỹ Thuật Đã Triển Khai | Kết Quả Thực Nghiệm |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Mất câu thoại của người nói trước trong Closed Captions** | Phụ đề CC tiếng Nhật gộp 2 người nói vào cùng 1 cue: `-（Người 1）Thoại 1\N-（Người 2）Thoại 2`. Gemini chỉ dịch câu của người thứ 2 và nuốt mất câu của người thứ 1. | Triển khai `split_multi_speaker_cc_events` trong `parser.py` bóc tách thành 2 event độc lập cùng start/end trước khi gửi đi dịch; sau đó `writer.py` hợp nhất lại thành Dual Dialogue. | Khắc phục 100% câu thoại bị mất (đã đối soát thành công trên tập 07 cảnh Kenshuu gọi "Thuần phi Kin"). |
| **2** | **Tràn màn hình 3–4 dòng trên IINA/mpv (Sub Stacking)** | Các cue thoại có timeline gối đầu nhẹ (100–300ms) do thoại đuổi nhau. IINA giữ lại cue cũ và render đồng thời cue mới. | Thuật toán Strict Zero-Overlap: Tự động co ngắn end time của cue trước: `c.end = min(c.end, next.start - 40ms)`. | Triệt tiêu 100% hiện tượng đè 3-4 dòng; màn hình luôn tối đa 2 dòng sạch sẽ. |
| **3** | **Cạn kiệt Quota API ban ngày (429 Resource Exhausted)** | Dự án chỉ dùng gói API miễn phí; 1 project bị giới hạn 500 RPD (Requests Per Day) với Lite và 20 RPD với Flash. | Kiến trúc Đa khóa (Multi-Key Rotation) kết nối 4 project độc lập. Khi gặp lỗi 429 PerDay, tự động đổi key ngay lập tức trong batch dịch mà không ngắt quãng pipeline. | Nâng dung lượng thực tế lên 2.000 RPD (Lite) và 80 RPD (Flash), dịch mượt mà 2–3 tập mỗi đêm. |
| **4** | **Ô nhiễm Cache SQLite do chuỗi thử nghiệm Dry-Run** | Lệnh test với cờ `--dry-run` sinh ra chuỗi `[DRY-RUN VI]...` và bị lưu vào database cache. Khi chạy thật, cache hit trả về chuỗi dry-run giả lập. | Thêm bộ lọc `UNTRANSLATED_STRINGS` và kiểm tra regex chặn hoàn toàn việc lưu các chuỗi `[DRY-RUN]`, `[Chưa dịch]` vào SQLite database. | Cache sạch hoàn toàn, bảo toàn khả năng tái sử dụng giữa các lần chạy. |
| **5** | **Suy diễn sai đại từ vai vế khi xưng hô bằng tên riêng** | Tiếng Nhật gọi tên nhân vật thay cho ngôi thứ hai. AI cố ép dịch tên riêng thành đại từ "anh/em/cậu/cô/mày", dẫn đến đoán sai giới tính/quan hệ. | Thiết lập quy chuẩn bắt buộc *Vocative Name Retention*: Giữ nguyên tên Romaji trong phụ đề tiếng Việt (Fran, Jean, Nell, Reirin...). | Giảm 90% lỗi sai đại từ xưng hô trong các cảnh đối thoại gọi tên. |
| **6** | **Dịch đúp tiếng cười & Nhân đôi câu thoại trong Dual Dialogue** *(Gặp tại Tenken S2-02)* | Cue JA là tiếng cười của Jean `ふはははは!` kéo dài 4.4s. Do time overlap, câu thoại EN của cô bé bị gán nhầm vào cue tiếng cười. AI dịch câu EN đó thành câu khen ngợi Jean. Cue tiếp theo của cô bé cũng dịch thành câu khen ngợi. Bộ Dual Dialogue thấy 2 câu khác chữ phần đuôi nên gộp thành 2 dòng lặp lại: `- Đúng là pháp sư chiêu hồn vĩ đại thời đại chúng ta! Ngài đỉnh quá!\n- Đúng là pháp sư chiêu hồn vĩ đại, tuyệt vời quá!`. | **1.** Mở rộng `PURE_INTERJECTION_REGEX` nhận diện toàn bộ các từ tượng thanh tiếng cười anime (`ふはは+`, `ははは+`, `フハハ+`, `ククク+`, `ふふふ+`...).<br>**2.** Nâng cấp bộ `Anti-Duplicate` trong `writer.py`: Tự động phát hiện 2 câu xuất hiện cùng lúc chia sẻ tiền tố chung dài (common prefix >= 15 chars) hoặc độ trùng lặp từ vựng >= 60% để chỉ giữ lại 1 câu duy nhất. | Triệt tiêu hoàn toàn câu dịch đúp; phụ đề tại 14:54 chỉ còn 1 dòng duy nhất chuẩn xác: `Đúng là pháp sư chiêu hồn vĩ đại, tuyệt vời quá!`. |

---

## 5. Những Điểm Nghẽn Kỹ Thuật Còn Tồn Tại (Current Bottlenecks & Open Challenges)

Mặc dù hệ thống đã vận hành rất tốt, phân hệ dịch thuật hiện vẫn còn một số điểm nghẽn tiềm ẩn cần được tối ưu hóa sâu hơn:

### 5.1. Bất Đối Xứng Ngữ Nghĩa giữa JA và EN (Semantic Asymmetry)
* **Vấn đề:** Các nhóm fansub hoặc nhà phát hành phương Tây thường "bản địa hóa tự do" (localization). Có những đoạn tiếng Nhật chỉ là một câu cảm thán ngắn (1–2 từ), nhưng bản tiếng Anh lại thêm thắt cả một câu dài, hoặc gộp 2 câu liên tiếp vào một cue. Khi đó, nếu hệ thống dựa quá nhiều vào câu tiếng Anh tham chiếu, AI sẽ bị "tiếng Anh dắt mũi" và dịch sai lệch hoàn toàn ý đồ của đạo diễn anime.

### 5.2. Nhận Diện Độc Thoại Nội Tâm (Monologue) vs Lời Nói Ra Miệng (Spoken Dialogue)
* **Vấn đề:** Trong anime, nhân vật suy nghĩ trong đầu thường xưng hô là *"tôi/mình"*, nhưng khi nói ra miệng với bạn bè/người yêu lại xưng là *"tớ - cậu"*, *"anh - em"*, hoặc *"ta - ngươi"*. Hiện tại hệ thống chưa có bước phân loại tự động câu thoại đó là suy nghĩ nội tâm (Monologue) hay lời nói phát ra ngoài, khiến AI đôi khi áp dụng đại từ xưng hô của lời nói vào dòng suy nghĩ nội tâm.

### 5.3. Rò Rỉ Ngữ Cảnh Qua Ranh Giới Phân Cảnh (Scene Boundary Context Bleed)
* **Vấn đề:** Cửa sổ ngữ cảnh trượt hiện tại là cố định `+/- 10 cues`. Nếu cue thứ 5 nằm ở cuối cảnh 1 (trong rừng giữa 2 nhân vật A và B), và cue thứ 6 chuyển cảnh sang một thị trấn mới hoàn toàn (nhân vật C và D xuất hiện), AI vẫn nhìn thấy 5 câu của cảnh cũ và có xu hướng "nhiễm" đại từ hoặc chủ ngữ của A và B sang cho C và D.

### 5.4. Độ Thông Minh của Model & Khả Năng Hiểu Ẩn Ý (Model Reasoning Ceiling)
* **Vấn đề:** Để đảm bảo tốc độ và tiết kiệm chi phí/quota, `gemini-3.5-flash-lite` được chọn làm model mặc định. Với các câu giao tiếp hàng ngày, Flash-Lite hoạt động rất tốt. Tuy nhiên với các thể loại anime có:
  * Thơ ca, câu đối, thư từ cung đình cổ trang.
  * Lời giải thích triết lý, ma pháp trừu tượng dài dòng.
  * Chơi chữ (puns), từ lóng (slang), hoặc câu nói mỉa mai ngầm (sarcasm).
  Flash-Lite đôi khi dịch quá thô hoặc hiểu theo nghĩa đen.

---

## 6. Các Câu Hỏi & Định Hướng Cần AI Nghiên Cứu Tư Vấn

Để đưa chất lượng dịch phụ đề tiếng Việt của phân hệ này lên chuẩn mực cao nhất (tương đương với các dịch giả anime chuyên nghiệp hàng đầu), chúng tôi cần AI Nghiên cứu tập trung phản biện và đề xuất phương án chi tiết cho các câu hỏi sau:

1. **Bộ lọc Bất Đối Xứng Ngữ Nghĩa JA-EN (Semantic Asymmetry Gate):**
   * Làm thế nào để tự động đo lường độ lệch ngữ nghĩa/độ dài giữa cue JA và cue EN trước khi gửi vào prompt?
   * Khi phát hiện bất đối xứng, nên có cơ chế fallback như thế nào (bỏ EN, chỉ gửi JA, hay gửi prompt cảnh báo AI)?
2. **Nhận thức Phân Cảnh Tự Động (Automated Scene Boundary Detection):**
   * Làm sao để phát hiện điểm ngắt phân cảnh chính xác mà không tốn tài nguyên (dựa vào khoảng lặng timeline phụ đề > 2.5s, hay phân tích video scene cuts)?
   * Cơ chế làm mới bộ nhớ đệm ngữ cảnh (Context Reset) tại ranh giới phân cảnh nên được thiết kế ra sao?
3. **Phân Loại Trạng Thái Thoại (Utterance Tagging):**
   * Giải pháp hiệu quả nhất để gắn nhãn từng câu thoại: `[Spoken]`, `[Inner Monologue]`, `[Spell/Skill]`, `[Flashback]`, `[Telepathy]` trước khi dịch?
4. **Cơ Chế Nâng Cấp Model Động (Dynamic Model Escalation):**
   * Thiết kế thuật toán chấm điểm độ phức tạp của câu thoại (Dựa trên độ dài, từ vựng hiếm, cờ `uncertain: true`, số lượng mệnh đề phụ) để tự động quyết định câu nào dịch bằng Lite (nhanh/rẻ) và câu nào điều hướng sang Flash/Thinking model (suy luận sâu)?
5. **Khai Thác Dữ Liệu Tập Trước (Few-shot Prompting từ Season 1 / Ep trước):**
   * Với các bộ anime đã có dữ liệu phụ đề chính thức tập trước (như Season 1 của *Tensei Shitara Ken Deshita*), làm thế nào để xây dựng cơ chế RAG (Retrieval-Augmented Generation) nhẹ cục bộ, tự động truy vấn 3–5 cặp câu thoại mẫu có ngữ cảnh tương đồng nhất để nhúng vào prompt dịch của Season 2?
6. **Hậu Kiểm Độc Lập Dựa Trên Quy Tắc (Deterministic Subtitle Linter Pass):**
   * Đề xuất danh mục các bộ quy tắc linter (sau khi dịch xong và trước khi xuất file `.srt`) để bắt tự động 100% các lỗi: lật đại từ trong tích tắc, sót tên riêng Romaji, từ cấm Hán-Việt, và câu quá dài vượt mắt nhìn.
