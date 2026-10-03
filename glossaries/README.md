# Anime Subtitle Translation Glossaries

Thư mục lưu trữ các file thuật ngữ (glossary) dùng cho công cụ `subtrans`:

1. **`base.yaml`** (symlink trỏ tới `subtrans/glossary.yaml`):
   - Bảng thuật ngữ nền chung (Schema v2) áp dụng cho mọi bộ anime:
     - 12 quy tắc dịch (`style_rules`): nguồn JA chuẩn, văn phong nói tự nhiên, không word-by-word, 42 ký tự/dòng.
     - Quy chuẩn đại từ xưng hô (`address_rules`): ngôi 1, ngôi 2, kính ngữ, quan hệ nhân vật.
     - 230+ thuật ngữ anime phổ biến: xưng hô, chức danh, trường học, romcom, isekai, shounen, võ thuật, tu tiên (xianxia), ẩm thực.
   - **Tự động kích hoạt làm lớp nền cho mọi lượt dịch.**

2. **File thuật ngữ theo từng bộ (Series-specific glossary)**:
   - Đặt tên theo dạng `glossary.<tên-bộ>.yaml` (ví dụ: `glossary.rezero.yaml`).
   - Các thuật ngữ và nhân vật trong file riêng sẽ **được ưu tiên ghi đè** lên bảng nền.
   - Sử dụng cờ `-g` hoặc `--glossary`:
     ```bash
     python subtrans.py run anime_ep01.ass -g glossaries/glossary.rezero.yaml
     ```
