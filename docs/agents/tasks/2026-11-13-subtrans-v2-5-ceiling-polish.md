# Handoff: Triển khai Gói V2.5 — Chạm Trần 100% Phân hệ Dịch Phụ đề AI (Subtrans Ceiling Polish)

## Yêu cầu và phạm vi
- Ngày/múi giờ: `2026-11-13T22:57:00+07:00`, Asia/Bangkok (+07:00).
- Nguồn yêu cầu và mục tiêu: Người dùng hỏi *"nó đã tối ưu nhất có thể chưa?"*, trực tiếp yêu cầu *"triển khai luôn Gói V2.5 (Chạm trần 100%) để xử lý trọn vẹn các điểm trên"* và xác nhận *"xong hết chưa? xong hết rồi thì commit và push lên git đi"*.
- Kết quả cần bàn giao:
  1. Cài đặt `faster-whisper` (kèm `onnxruntime` cho Silero VAD) và `fugashi` + `unidic-lite` (MeCab C++ morphological tokenizer) vào `.venv` mà không đụng chạm phiên bản `torch`/`tensorrt_cu12`/`numpy`; cập nhật `requirements.txt`.
  2. Nâng cấp `subtrans/ja_morph_tagger.py` lên bộ tách từ hình thái học MeCab (`fugashi` + `unidic-lite`, `<0.5ms`/câu, 0 VRAM) kết hợp lớp chặn ranh giới Kanji ghép ở nhánh fallback Regex, triệt tiêu 100% báo giả đại từ do trùng chữ Hán (`私立`, `我慢`, `下僕`, `君臨`, `余裕`, `手前味噌`, `某所`) và phân biệt chính xác đại từ `君` (`2P:kimi`) với hậu tố tên `-君` (`HON:kun`).
  3. Tối ưu vòng đời SQLite `KnowledgeBase` (khởi tạo 1 lần mỗi tập thay vì mỗi batch), sắp xếp khóa tĩnh trước khóa động trong `filter_relevant_glossary` & `GeminiTranslator.translate_batch`, nâng cấp `build_repair_system_prompt` (`> 4.096` tokens tĩnh) để kích hoạt Implicit Context Caching cho cả lượt sửa `Cascade`, và bổ sung cơ chế tùy chọn **Explicit Context Caching** (`--explicit-cache` / `explicit_context_caching` TTL 15 phút kèm fallback mượt).
  4. Biên soạn bộ Glossary chuyên biệt chuẩn fan-sub cho các series ưu tiên hàng đầu (`glossary.frieren.yaml`, `glossary.violet.yaml`, `glossary.jujutsu.yaml`, `glossary.kimetsu.yaml`, `glossary.rezero.yaml` + pointer `glossary.demonslayer.yaml`), hỗ trợ tự dò Glossary theo cả đường dẫn thư mục cha (`Frieren/S01E01.mkv`), và bổ sung kho câu mẫu `BUILTIN_GOLD_EXEMPLARS` + lệnh `subtrans.py rag seed` xóa bỏ trạng thái cold-start của Hybrid RAG.
  5. Xây dựng module `subtrans/audio_diarization.py`: DSP thuần CPU (`ffmpeg` pipe kênh thoại trung tâm `5.1 FC` ở `8kHz` mono in-memory + đặc trưng âm sắc phổ log-band CMS & cao độ `F0` cepstral bằng `numpy`, 0 file tạm, 0 VRAM) tự động gắn nhãn `audio_turn_hint` (`speaker_changed` / `same_speaker`) cho các file `.srt` trần khuyết tên người nói.
  6. Bổ sung cơ chế gối đầu chuẩn bị phụ đề + `StageAPrefetchWorker` cho tập `N+1` trong `scripts/batch_translate_all.py` (kèm cờ `--no-prefetch` và khóa an toàn chống trùng `<series>.cast.yaml` / hủy ngay khi `QUOTA_EXHAUSTED`).
- Được phép / chưa được phép: 100% Cloud Gemini API (`gemini-3.5-flash-lite`, `gemini-3.8-flash`), tuyệt đối không dùng local LLM/Ollama, không đọc/in `.env`, không gọi Gemini API thật trong kiểm thử.
- Trạng thái: hoàn tất.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, branch `main`, tích hợp trên nền `04c8b4e`.
- File đã sửa/tạo mới:
  - `requirements.txt`, `src/subtitles.py`, `scripts/batch_translate_all.py`
  - `subtrans/AGENTS.md`, `subtrans/audio_diarization.py`, `subtrans/ja_morph_tagger.py`, `subtrans/models.py`, `subtrans/glossary.py`, `subtrans/rag.py`, `subtrans/gemini_backend.py`, `subtrans/cascade.py`, `subtrans/pipeline.py`, `subtrans/cli.py`, `subtrans/config.yaml`
  - `glossaries/glossary.{frieren,violet,jujutsu,kimetsu,demonslayer,rezero}.yaml`
  - `tests/test_subtrans_v2_5_ceiling_polish.py`
- File dữ liệu production giữ nguyên vẹn: `.env`, `subtrans/.cache/*`, `pending_subs.json`, `.temp_*`, `out/*`, `logs/*`.

## Đã làm và bằng chứng
- Đã hoàn tất trọn vẹn cả 6 hạng mục Gói V2.5 và viết bộ kiểm thử đơn vị/tích hợp đầy đủ `tests/test_subtrans_v2_5_ceiling_polish.py` (7 bài kiểm thử chuyên sâu bao phủ toàn bộ 6 hạng mục).
- Kiểm định cô lập:
  - Lệnh: `.\.venv\Scripts\python.exe scripts/verify_offline.py`
  - Kết quả: **`319 passed, 11 skipped, 1 deselected in 180.61s`** (Pytest exit code `0`, Evidence: `E:\ai-video-upscaler\scratch\agent-validation-g6gx0o5v`).
  - Linter mã nguồn: `ruff check upscale.py subtrans.py src subtrans scripts tests` -> `All checks passed!` (`0 errors`).
  - Bộ nhớ Agent: `.\.venv\Scripts\python.exe scripts/agent_context.py --check` -> `PASS`.
- Mux/frame/fps: `N/A — không tạo thành phẩm video trong tác vụ nâng cấp Subtrans V2.5`.

## Bước tiếp theo
1. Không còn hạng mục tồn đọng.
- Câu hỏi/approval còn chờ: Không.
- Rủi ro/giới hạn cần báo người dùng: Không gọi API thật ban ngày theo quy định bảo vệ quota; chế độ `--explicit-cache` tự động fallback êm về `system_instruction` Implicit Caching nếu gói API Free Tier từ chối `caches.create`.

## Kết thúc
- Kết quả và giới hạn đã thông báo: Đã báo cáo hoàn tất kiểm thử 319 tests PASS, commit và push toàn bộ lên GitHub `origin/main`.
- Commit/tag: Xem `git log -n 1 -- docs/agents/tasks/2026-11-13-subtrans-v2-5-ceiling-polish.md`.
- Tài liệu cập nhật: [subtrans/AGENTS.md](../../../subtrans/AGENTS.md), [STATE](../STATE.md), [PROJECT_MAP](../PROJECT_MAP.md), [README](../../../README.md).
