# Handoff: Triển khai Lộ trình 6 Gói Nâng cấp Tối đa Chất lượng Phân hệ Dịch Phụ đề AI (Subtrans V2)

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-13, Asia/Bangkok (+07:00).
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu đọc toàn bộ codebase chi tiết, đánh giá chuyên sâu phân hệ dịch phụ đề bằng AI (`subtrans/`, `src/subtitles.py`, `upscale.py`, `glossaries/`, `scripts/`), đề xuất giải pháp cải thiện và phê duyệt thực thi toàn diện lộ trình 6 gói nâng cấp (`P0-A`, `P0-B`, `P1-A`, `P1-B`, `P2`, `P3`).
- Kết quả bàn giao: Hoàn thiện 100% cả 6 gói giải pháp (`P0-A`, `P0-B`, `P1-A`, `P1-B`, `P2`, `P3`), mở rộng bộ kiểm thử đơn vị/tích hợp offline và đồng bộ tài liệu kiến trúc.
- Được phép / chưa được phép: 100% Cloud Gemini API (`gemini-3.5-flash-lite`, `gemini-3.8-flash`), tuyệt đối không bật local Ollama, không đọc/in `.env`, không gọi Gemini API thật sau 14:00 (Daytime Quota Guard).
- Trạng thái: hoàn tất.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, branch `main`, baseline `c4918d6`.
- File đã sửa/tạo mới: `upscale.py`, `src/pending_queue.py`, `src/subtitles.py`, `src/cleanup.py`, `src/scheduler.py`, `scripts/verify_offline.py`, `scripts/batch_translate_all.py`, `subtrans/{models.py,models.yaml,config.py,validation.py,cache.py,throttle.py,dynamic_cast.py,gemini_backend.py,ja_morph_tagger.py,dtw_auto_align.py,parser.py,glossary.py,cast_sheet.py,linter.py,cascade.py,writer.py,pipeline.py,report.py,rag.py,AGENTS.md}`, `subtrans/eval/{pronoun_suite.yaml,runner.py}`, `subtrans/tests/test_gemini.py`, `tests/{test_subtrans_deep_upgrades.py,test_subtrans_v2_max_quality.py}`, `docs/agents/{STATE.md,PROJECT_MAP.md}`.
- Chủ sở hữu từng phần / subagent: Coordinator thực thi toàn bộ code/test; subagent khảo sát read-only `b4ed235a-0aba-485b-83b1-b6ea17592d46` đã kết thúc từ đầu phiên.
- File dữ liệu production giữ nguyên vẹn: `.env`, `subtrans/.cache/*`, `pending_subs.json`, `.temp_*`, `out/*`, `logs/*`.

## Đã làm và bằng chứng
- Thay đổi cốt lõi theo 6 gói:
  - **P0-A (`subtrans/cache.py`, `subtrans/pipeline.py`, `subtrans/dynamic_cast.py`, `subtrans/gemini_backend.py`):** Khóa cache ngữ cảnh `compute_contextual_cue_sig` (`prompt_version="v2.2"`), giữ `inline_cached_context` khi partial cache hit, ghi ngược kết quả sửa `Cascade` vào `TranslationCache`, bổ sung cặp đại từ tường minh `(p1, p2)` trong `SubtitleOutputItem` CoT và chặn từ ghép tiếng Việt giả dương tính (`con rồng`, `một mình`, `cô gái`).
  - **P0-B (`subtrans/gemini_backend.py`, `subtrans/models.yaml`, `subtrans/throttle.py`, `subtrans/validation.py`, `subtrans/report.py`):** Bổ sung `STATIC_ANIME_TRANSLATION_HANDBOOK` vào `build_episode_system_prompt` (> `16.000` ký tự / > `4.096` tokens tĩnh) kích hoạt Gemini Implicit Context Caching + ghi `cached_tokens` vào SQLite & báo cáo QA; `ThinkingConfig` (`MINIMAL` cho `3.5-flash-lite`, `LOW` cho `3.8/3.7/3.6/3.5-flash`), `max_output_tokens=16384`, `BLOCK_NONE` safety, mở rộng `JAPANESE_CHAR_REGEX` (CJK Ext-A & `「」`); sửa xoay vòng 3-Key modulo từ Key #3 về Key #1, theo dõi cạn RPD theo `(model, key_id)` và đổi key ngay khi `429 PerMinute`.
  - **P1-A (`src/subtitles.py`, `subtrans/parser.py`, `subtrans/dtw_auto_align.py`):** Hỗ trợ `faster-whisper` + Silero VAD + `initial_prompt` từ `Glossary` (fallback mượt sang `openai-whisper`); đọc `event.name` `.ass`, loại `event.is_comment`, chặn `\org|\clip|\iclip|\p[1-9]`, chặn nhận nhầm âm thanh CC `(筆記音)` làm người nói, gắn `mode_hint="inner"` cho `＜...＞`/`Italics`, tách `1:2` theo khoảng lặng `>= 250ms` (`get_dtw_alignment_plan`) và bù lệch `Piecewise 2-Segment Auto-Offset`.
  - **P1-B (`subtrans/glossary.py`, `subtrans/ja_morph_tagger.py`, `subtrans/linter.py`, `subtrans/cascade.py`, `subtrans/writer.py`):** Cắt ngữ cảnh tại điểm chuyển cảnh (`gap >= 4500ms`), deep-merge `AddressRules`, bắt phương ngữ Kansai (`なんでやねん`) & `某` (`soregashi`), ranh giới từ Katakana chống báo giả (`プログラム` vs `ラム`), `Character.aliases`, `RULE_GLOSSARY_TERM` & `RULE_PRONOUN_CONSISTENCY`, cách ly lệch con số trong Cascade, đảo thứ tự `optimize_cues_for_playback -> lint_subtitles -> report`, bảo toàn `<Rudeus!>`, cặp nháy giữa câu, lời gọi lặp lại cảm xúc và Typesetting `.ass`.
  - **P2 (`subtrans/rag.py`, `subtrans/eval/pronoun_suite.yaml`, `subtrans/eval/runner.py`):** Hybrid BM25 RAG + điểm thưởng cặp vai `(speaker, addressee)` + phân tầng `gold > reviewed > silver`, chống tự nhiễm độc `error_bank`, tự động thăng hạng câu sửa sạch (`promote_cascade_repairs`), mở rộng `pronoun_suite.yaml` lên `22` cues và kiểm chứng sống toàn diện trong `eval/runner.py` (`6/6` regression + `22/22` pronoun cues).
  - **P3 (`src/pending_queue.py`, `src/cleanup.py`, `subtrans/config.py`, `scripts/verify_offline.py`, `src/scheduler.py`, `scripts/batch_translate_all.py`, `upscale.py`):** Tách `src/pending_queue.py` (stdlib-only, không kéo `torch`), dọn sạch `*.whisper_ja.srt`, cô lập tuyệt đối `PYTHON_DOTENV_DISABLED=1`, tính đúng model + 3-key RPD trong `scheduler.py`, và hỗ trợ `--target-4k` + `--async-subs` trong `upscale.py`.
- Kiểm định thực nghiệm (100% offline/isolated):
  - `.\.venv\Scripts\python.exe scripts/verify_offline.py`: **`298 passed, 0 failed, 11 skipped, 1 deselected`** trong `83.87s`, evidence tại `E:\ai-video-upscaler\scratch\agent-validation-xkofqckq`.
  - `.\.venv\Scripts\python.exe scripts/agent_context.py --check`: **`PASS`** (`0` errors, `0` warnings).
  - `ruff check upscale.py subtrans.py src subtrans scripts tests`: **`All checks passed!`** (`0` errors).
- Mux/frame/fps: **`N/A — không tạo thành phẩm`** (tác vụ nâng cấp mã nguồn & kiểm thử phụ đề offline theo quy định Daytime Quota Guard).

## Bước tiếp theo
1. Vận hành thực tế vào ca đêm (sau mốc reset hạn mức hoặc ngoài khung bảo vệ ban ngày) bằng `python upscale.py ... --async-subs` hoặc `python subtrans.py run ...`.
2. Theo dõi bảng tỷ lệ `% Cached Input Tokens` trong `out/<episode>.audit.md` sau các tập chạy thật đầu tiên.

## Kết thúc
- Kết quả và giới hạn đã thông báo: Đã kiểm chứng toàn bộ bằng unit/integration tests offline; không gọi Gemini API thật trong giờ bảo vệ ban ngày.
- Commit/tag: tra bằng `git log -n 1 -- docs/agents/tasks/2026-11-13-subtrans-v2-max-quality-roadmap.md`.
- Tài liệu cập nhật: [STATE](../STATE.md), [PROJECT_MAP](../PROJECT_MAP.md), [subtrans/AGENTS.md](../../../subtrans/AGENTS.md).
