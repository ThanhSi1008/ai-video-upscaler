# Hồ Sơ Quyết Định Kiến Trúc (Architectural Decision Records - ADR)

> **Cập nhật ngày:** 2026-10-29  
> **Tài liệu liên quan:** [START_HERE](START_HERE.md) | [PROJECT_MAP](PROJECT_MAP.md) | [STATE](STATE.md) | [VALIDATION](VALIDATION.md) | [Root AGENTS](../../AGENTS.md) | [Video Pipeline](../video-pipeline.md) | [Dedup Architecture](../dedup-architecture.md) | [Audit & Safety](../audit-and-safety.md) | [Subtrans AGENTS](../../subtrans/AGENTS.md)

---

## Danh Mục Các Quyết Định Kiến Trúc

### ADR-001: 100% Cục Bộ cho Video Upscale & 100% Cloud API cho Dịch Thuật
- **Bối cảnh:** Thiết bị mục tiêu là Windows 11 Workstation với RTX 3060 (12GB VRAM). Vừa upscale video 4K vừa chạy LLM cục bộ (Ollama / Local LLM) gây cạn kiệt VRAM nghiêm trọng (OOM).
- **Quyết định:**
  - AI Video Upscale chạy 100% cục bộ trên PyTorch CUDA FP16 Native.
  - Phụ đề chạy 100% qua Google Gemini Cloud API (3 API key luân phiên).
  - Tuyệt đối cấm nạp Local LLM khi tiến trình GPU Upscale đang chạy.

### ADR-002: Bắt Buộc Cờ `-max_interleave_delta 0` Khi Mux Stream Copy
- **Bối cảnh:** Khi mux container MKV chứa nhiều stream âm thanh đa kênh (5.1, FLAC, AC3) và nhiều phụ đề mềm, FFmpeg mặc định có thể tạo ra độ lệch interleave giữa audio và video.
- **Quyết định:** Lệnh mux bắt buộc dùng:
  `-map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 -c copy -max_interleave_delta 0`
- **Lý do:** Ngăn chặn triệt để lỗi tràn bộ đệm và mất tiếng cục bộ sau vài phút phát lại trên TV/trình phát video.

### ADR-003: Chuỗi Ma Trận Màu Sắc BT.709 Hai Chiều Chuẩn Xác
- **Bối cảnh:** Anime sản xuất theo chuẩn BT.709 TV Limited Range. Nếu chuyển đổi sai giữa RGB Full Range và YUV Limited Range, hình ảnh 4K sẽ bị bợt màu (gamma wash-out) hoặc bết đen.
- **Quyết định:**
  - Trích xuất: `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc` (YUV Limited -> RGB24 Full).
  - Mã hóa: `scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv` (RGB24 Full -> YUV Limited `p010le`).
  - Gắn cờ metadata: `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`.

### ADR-004: Deduplication Trước Suy Luận Bằng Keyframe GOP + DIS Optical Flow
- **Bối cảnh:** Anime chứa ~60% khung hình tĩnh hoặc trùng lặp do kỹ thuật hoạt họa giới hạn (Limited Animation 8–12 fps). Upscale từng frame trùng lặp gây lãng phí 60% thời gian GPU.
- **Quyết định:**
  - Cắt video theo Keyframe GOP (`split_video_segments`).
  - Quét `mpdecimate` với ngưỡng mặc định `768:320:0.33`.
  - Bổ sung phát hiện chuyển động lia máy/zoom qua dòng chảy quang học DIS (`--dedup-policy dis`, `--dis-thresh 0.30`).
  - Tái tạo frame 4K hoàn chỉnh qua FFmpeg Concat Demuxer (triệt tiêu 100% dung lượng đĩa của hardlinks).

### ADR-005: Đính Chính Mô Hình AnimeJaNai (`janai-fix`)
- **Bối cảnh:** Kiến trúc `AnimeJaNaiCompact2x` nguyên bản học residual trên không gian ảnh nội suy bicubic/nearest. Khi chạy suy luận PyTorch thuần nếu thiếu phép cộng nearest $\times 2$, ảnh đầu ra sẽ bị tối đen hoặc sai lệch cấu trúc nghiêm trọng.
- **Quyết định:** Bổ sung phép cộng residual nearest $\times 2$ trong `src/cuda_engine.py`:
  `F.interpolate(x, scale_factor=2.0, mode='nearest')`. Đạt khớp bit-exact 0.0 với spandrel CPU chuẩn (PSNR 75–158 dB).

### ADR-006: Khử Ảo Giác Whisper & Trích Xuất Kênh Trung Tâm Thoại (FC)
- **Bối cảnh:** Whisper khi gặp nhạc nền OP/ED hoặc tiếng hét kéo dài trong anime thường phát sinh ảo giác (lặp câu YouTube `ご視聴ありがとうございました`, lặp token `ああああああああ`). Track âm thanh đa kênh 5.1 có tiếng nhạc lớn ở kênh bên làm át giọng nói.
- **Quyết định:**
  - Dùng `ffprobe` chọn track tiếng Nhật; với âm thanh 5.1 (`channels >= 6`), trộn ưu tiên kênh trung tâm Front Center (FC):
    `pan=mono|c0=0.7*FC+0.15*FL+0.15*FR,highpass=f=80,lowpass=f=8000,afftdn=nf=-25`.
  - Bộ lọc Regex `WHISPER_HALLUCINATION_REGEX` & `SCREAM_OR_REPEAT_LOOP_REGEX`.
  - Tự động drop các câu Whisper mồ côi (`is_dropped_sound=True`) nằm giữa khoảng lặng nhạc EN (> 2500ms).
  - Pass 3 chuẩn hóa mốc thời gian chỉ chạy trên `active_cues` và bảo toàn Dual Dialogue.

### ADR-007: Bảo Toàn Người Nói Multi-Speaker & Hiển Thị Song Song
- **Bối cảnh:** Phụ đề Closed Captions Nhật thường gộp 2 người nói vào cùng 1 event (ví dụ `-（絹秀）... \N -（清佳）...`).
- **Quyết định:**
  - Tách thành 2 event độc lập trong `subtrans/parser.py`, lưu người nói vào `Cue.speaker`.
  - Dịch độc lập từng câu để không bỏ sót thoại của nhân vật phụ.
  - Gom lại thành Dual Dialogue 2 dòng (`- Lời 1\n- Lời 2`) ở `subtrans/writer.py` khi thời gian đồng thời.
