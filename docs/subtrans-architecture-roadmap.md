# LỘ TRÌNH KIẾN TRÚC & ĐÁNH GIÁ ĐỊNH LƯỢNG (`subtrans`)
**Dự án:** Subtitle Translation Engine (`subtrans`)  
**Triết lý vận hành:** **100% Cloud API (Gemini Cloud), KHÔNG SỬ DỤNG MODEL LOCAL (Bỏ qua Ollama / Local LLMs)** theo chỉ định trực tiếp từ Người dùng.  
**Thứ tự ưu tiên chuẩn mực:** **Đo (Evaluation Harness) ➔ Đổi Schema & Ngữ cảnh ➔ Linter V1 ➔ Cascade Model (Flash/Lite) ➔ Dữ liệu RAG & Error Bank**.

---

## 1. Giai Đoạn P0: Nền Móng & Bộ Thước Đo Định Lượng (Evaluation Harness)

### 1.1. Ba Lớp Kiểm Thử
1. **Regression Test Suite:** Đóng băng 6 ca lỗi kinh điển (mỗi ca gồm JA, EN, Expected VI, Bad Pattern):
   - Ca 1: Multi-speaker CC splitting (câu Kenshuu gọi "Thuần phi Kin" Futsutsuka Ep 07).
   - Ca 2: Tràn 3–4 dòng do gối đầu thời gian (Strict zero-overlap 40ms gap).
   - Ca 3: Ô nhiễm SQLite cache do chuỗi dry-run `[DRY-RUN]` hoặc `[Chưa dịch]`.
   - Ca 4: Bảo tồn tên riêng Romaji trong xưng hô gọi tên (Vocative name retention).
   - Ca 5: Tiếng cười anime bị gán nhầm câu thoại tiếng Anh (Laughter sound cue detection: `ふはははは!`).
   - Ca 6: Triệt tiêu dị bản lặp lại trong Dual Dialogue (Anti-duplicate variant: `>= 15 chars` chung hoặc `>= 60%` từ vựng).
2. **Pronoun & Relationship Benchmark Suite:**
   - 150–200 cue mẫu từ *Futsutsuka* và *Tenken* được gắn nhãn tay `(speaker, addressee, mode, allowed_pronouns)`.
   - Đây là thước đo trực tiếp đo tỷ lệ chính xác đại từ (Pronoun Accuracy %).
3. **Holdout Episode Test:**
   - 1–2 tập người dùng tự sửa toàn bộ phụ đề tiếng Việt làm tập kiểm thử độc lập (không đưa vào RAG/Prompt).

### 1.2. Bộ Chỉ Số Tất Định (Deterministic Metrics)
* **Pronoun Accuracy (%):** Tỷ lệ câu xưng hô đúng theo cặp `(speaker, addressee, mode)`.
* **Romaji Retention (%):** Tỷ lệ giữ nguyên tên riêng Romaji khi tiếng Nhật gọi tên.
* **CJK Leakage (%):** Tỷ lệ phát hiện ký tự Kanji/Hiragana/Katakana bị lọt vào phụ đề tiếng Việt.
* **CPS Violation (%):** Tỷ lệ câu có tốc độ đọc vượt quá ngưỡng an toàn:
  - Mục tiêu (Netflix Standard): `<= 17 CPS`.
  - Trần tối đa (Anime Fansub): `<= 22–25 CPS`.
* **Zero-Overlap Compliance (%):** Tỷ lệ 100% tuân thủ `c.end <= next.start - 40ms`.

---

## 2. Giai Đoạn P1: Schema "Suy Luận Trước, Dịch Sau" & Linter V1

### 2.1. Nâng Cấp Pydantic Schema cho Gemini
Tận dụng tính năng giữ nguyên thứ tự thuộc tính trong Schema từ tháng 11/2025 của Gemini API để tạo Chain-of-Thought ngầm (suy luận bối cảnh trước khi sinh ra bản dịch `vi`):

```python
class CueOutputItem(BaseModel):
    id: int
    speaker: str                 # Nhân vật nói (từ cast sheet hoặc "unknown")
    addressee: Optional[str]     # Nhân vật nghe (nếu xác định được)
    mode: Literal[
        "spoken", "inner", "narration", "spell", 
        "flashback", "comm", "lyrics", "onscreen"
    ]
    en_conflict: bool            # Cờ đánh dấu EN mâu thuẫn/thêm thắt so với JA
    vi: str                      # Bản dịch tiếng Việt hoàn chỉnh
    uncertain: Literal[
        "none", "pronoun", "referent", "wordplay", 
        "archaic", "term", "other"
    ]
```

### 2.2. Linter Hậu Kiểm V1
Kiểm tra tự động trước khi xuất file `.srt`:
1. **Kiểm tra CPS:** Kéo dài end time nếu còn gap, nén từ ngữ nếu vượt trần.
2. **Kiểm tra CJK & Unicode NFC:** Tự động chuẩn hóa hoặc báo lỗi.
3. **Kiểm tra Tên riêng & Đại từ:** So khớp với bảng Cast Sheet của tập.

---

## 3. Giai Đoạn P2: Cascade Model Thông Minh (Chỉ Dùng Cloud API)
* Không dùng local model.
* Chiến lược Cascade:
  - **Draft Pass:** Sử dụng `gemini-3.5-flash-lite` theo từng batch.
  - **Review/Repair Pass:** Chỉ gom các câu có điểm khó $S(cue)$ cao hoặc bị cắm cờ `uncertain != "none"` gửi vào 1–2 request `gemini-3.8-flash` dưới dạng biên tập/chỉnh sửa (`editing review format`), vừa tiết kiệm RPD vừa đạt chất lượng cao nhất.

---

## 4. Giai Đoạn P3: Dữ Liệu RAG Nhẹ & Error Bank
* Xây dựng kho tri thức cục bộ:
  - Bảng `units`: Các cặp câu đã được người dùng duyệt (`gold`).
  - Bảng `error_bank`: Các lỗi từng xảy ra kèm cách sửa đúng và quy tắc.
  - Tự động gợi ý Few-shot mẫu cho các tập kế tiếp.
