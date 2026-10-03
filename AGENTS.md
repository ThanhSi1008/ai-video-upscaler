# AGENTS.md — AI Agent Guidelines & Architecture Compass

> **Kim chỉ nam dành cho các AI Agent khi đọc, bảo trì và phát triển codebase này.**
> **Quy tắc bắt buộc:** Ngắn gọn, chuẩn xác, không suy diễn; cập nhật file này sau mỗi lần tái cấu trúc hoặc thay đổi kiến trúc code.

---

## 1. Mục Tiêu & Ràng Buộc Cốt Lõi (Core Invariants)

* **Thiết bị:** Apple Silicon MacBook Pro M3 Pro (11-core CPU, 14-core GPU, 18GB Unified Memory).
* **Môi trường bắt buộc:** **100% CỤC BỘ trên macOS**. Tuyệt đối không đề xuất Cloud GPU, Colab, Kaggle.
* **Tôn chỉ Pareto (Điểm ngọt kỹ thuật):** Chất lượng hình ảnh phải cân bằng với thời gian. Không hy sinh nét vẽ anime lấy tốc độ rẻ tiền, không lãng phí tài nguyên cho cải thiện vô hình.
* **Mục tiêu thời gian:** **~4 giờ / tập 24 phút** (~17.000 unique frames sau deduplication @ 1.25 fps).

---

## 2. Kiến Trúc Thư Mục & Trách Nhiệm Module (`src/`)

```text
upscale/
├── upscale.py              # CLI Entrypoint (Argparse, Error handling, Ctrl+C trap)
├── upscale.sh              # Shell launcher (tự nhận virtualenv hoặc python3)
├── src/
│   ├── config.py           # Single source of truth: Hằng số, Paths, Model choices, VideoToolbox params
│   ├── ramdisk.py          # RAMDisk Context Manager: mount/unmount HFS+ RAM Disk ảo (Zero SSD Wear)
│   ├── ffmpeg_ops.py       # Thao tác FFmpeg/FFprobe chuẩn phòng thu (BT.709 10-bit, Stream Copy gốc)
│   ├── ai_engine.py        # NCNN Vulkan via MoltenVK (Metal API) suy luận mô hình anime
│   ├── dedup.py            # Deduplication Engine: Preflight mpdecimate scan & Concat Re-expansion
│   ├── preflight.py        # CaffeinateContext (chống sleep), Check AC/Pin, Check đĩa trống, macOS Notifs
│   ├── pipeline.py         # Orchestrator: Rolling Segments, Checkpoint Resume, Preview mode
│   └── logger.py           # Structured telemetry logging (logs/*.log & logs/latest.log symlink)
├── bin/realesrgan/         # Universal Mach-O binary & NCNN models (.param / .bin)
├── samples/                # Video mẫu test (sample_30s.mkv, sample_3min.mkv)
├── tests/                  # Kiểm tra tự động, A/B test blind samples
└── logs/                   # Log chi tiết có cấu trúc từng phiên chạy
```

---

## 3. Những Bất Biến Kỹ Thuật Tuyệt Đối Không Được Phá Vỡ (Do NOT Break)

1. **Bảo vệ 100% NVMe SSD (Zero SSD Wear):**
   * Mọi frame trung gian (trích xuất và 4K AI) **bắt buộc phải ghi trên RAM Disk ảo**.
   * Sau khi một chunk encode xong, phải `rmtree` dọn sạch RAM Disk ngay lập tức.
2. **Bảo toàn 100% luồng gốc (Bit-exact Stream Copy):**
   * Video 4K thành phẩm **bắt buộc Stream Copy nguyên bản**: Audio đa kênh, Phụ đề mềm (ASS/SRT), Font chữ nhúng (TTF/OTF) và Chapters (`-map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 -c copy`).
   * Tuyệt đối không re-encode âm thanh hay làm mất font phụ đề.
3. **Chuỗi ma trận màu sắc chuẩn xác (BT.709 Two-way Matrix):**
   * Trích xuất: `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc` (YUV Limited ➔ RGB24 Full).
   * Mã hóa: `scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv` (RGB24 Full ➔ YUV Limited `p010le`).
   * Metadata gắn cờ: `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`.
4. **Vòng đời an toàn (Process Lifecycle & Auto Cleanup):**
   * `caffeinate` và `hdiutil` (RAM Disk) phải luôn nằm trong `try...finally` hoặc Context Manager với `atexit` để thu hồi sạch sẽ kể cả khi bị `Ctrl+C` (SIGINT).

---

## 4. Dữ Liệu Thực Nghiệm Chuẩn Đã Đo Đạc (Ground Truth Numbers)

* **Tốc độ AI M3 Pro (GPU 14-core):** Duy trì ổn định **1.24 – 1.27 fps** với `realesr-animevideov3` (64nf16nc, ~2.5 TFLOPs/frame).
* **Kết quả True Blind Test (3/3 chọn bởi người dùng):** Người dùng mù nhãn kiểm tra trực tiếp ở độ phân giải 100% trên màn hình Liquid Retina XDR và chọn `realesr-animevideov3` đứng 1st ở cả 3 cảnh (Cận cảnh, Phong cảnh, Hành động). ArtCNN bị loại vì MAD Luma chỉ 0.2% so với Bicubic; AnimeJaNai bị loại vì nét mềm hơn.
* **Thời gian chạy thực tế mẫu 3 phút (4.360 frames):** Hoàn tất toàn bộ 23 phân đoạn trong **37m 09s** (AI: 1.764s, Encode: 278s, Extract: 185s, Preflight/Mux: 15s).
* **Overhead tuần tự ngoài AI:** **~20.1s / chunk** (Decode FFmpeg ~8.5s + Encode VideoToolbox ~11.6s).
* **Giải pháp Pipelined Double-Buffering:** Chạy gối đầu CPU decode (chunk N+1) và Media Engine encode (chunk N-1) song song với GPU AI infer (chunk N). Mức chiếm dụng RAM Disk tối đa 2.7 GB (< 4 GB). Triệt tiêu toàn bộ 59 phút thời gian chết của GPU, đưa thời gian chạy toàn tập 24 phút về **~3h48m** (< 4 giờ).
* **Kiểm định độc lập Deduplication (4.360 frames):** `ffmpeg mpdecimate` lọc ra chính xác **2.174 raw kept frames** (+ 12 frame ép ranh giới chunk = 2.186 unique frames).
* **Audit cục bộ Local Patch SSIM (Lưới 8x8):** Global SSIM trung bình 0.98061 (min 0.93286); Local Patch SSIM trung bình 0.96996 (min 0.89905); MAD điểm ảnh chỉ 0.93/255 (0.36%). Khẳng định 100% không mất cử động môi hay chớp mắt.

---

## 5. Quy Chuẩn Khi Viết & Sửa Code Cho AI Agent

* **Không dùng thư viện nặng không cần thiết:** Dự án giữ dependencies tối giản (FFmpeg CLI + NCNN binary + Python stdlib + Pillow). Không cài PyTorch / Torchvision / TensorFlow nếu không có yêu cầu đặc biệt.
* **Bảo toàn khả năng Resume:** Thư mục `.temp_<basename>` lưu các phân đoạn `up_*.mp4` trên SSD cho phép dừng và chạy tiếp bất cứ lúc nào.
* **Cập nhật tài liệu song hành:** Bất cứ khi nào thêm tham số CLI mới vào `upscale.py`, phải cập nhật `README.md` và `AGENTS.md` tương ứng.

---

## 6. Thiết Kế Kiến Trúc Deduplication & Vận Hành Qua Đêm (~4 Giờ/Tập)

### A. Quy Trình Deduplication (`plan.json` ➔ Concat Re-expansion)
1. **Phân tích Preflight toàn tập (1 lần duy nhất):**
   * Dùng `mpdecimate=hi=768:lo=320:frac=0.33,showinfo` quét toàn tập để lập `plan.json`.
   * **Bắt buộc ép giữ frame đầu mỗi chunk:** `kept = sorted(set(kept) | set(range(0, total, 192)))` để các phân đoạn 8s hoàn toàn độc lập, bảo toàn 100% khả năng Checkpoint Resume.
2. **Suy luận AI có chọn lọc:**
   * Chỉ giải mã và đưa các frame `kept` (~96 frame/chunk) vào NCNN Vulkan $\rightarrow$ Giảm 50% thời gian GPU.
3. **Tái tạo khung hình 4K (Re-expansion) không tốn RAM/Disk:**
   * **Tuyệt đối không dùng hardlink PNG:** RAM Disk HFS+ không hỗ trợ clonefile và FFmpeg sẽ decode lặp lại PNG.
   * **Giải pháp chuẩn:** Dùng **FFmpeg Concat Demuxer** với thuộc tính `duration` theo độ dài run của frame đại diện + chuỗi filter `fps=24000/1001`. FFmpeg chỉ decode frame 4K một lần và nhân bản bằng tham chiếu buffer trong bộ nhớ.
   * Chốt cờ CFR: `-video_track_timescale 24000 -frames:v 192` và kiểm tra giao dịch qua `ffprobe -count_frames`.

### B. 8 Nguyên Tắc Vận Hành Tác Vụ Chạy Qua Đêm (Overnight Batch)
1. **Hàng đợi có chặn trên (Bounded Queue):** Giữ tối đa 2 batch trong RAM Disk (ngăn tràn bộ nhớ 4GB).
2. **Độc quyền GPU:** Chỉ một tiến trình NCNN chạy tại một thời điểm; decode (CPU) và encode (Media Engine) chạy gối đầu song song.
3. **Giao dịch phân đoạn (Transactional Segment):** Ghi ra file `.tmp`, xác minh đủ 192 frames bằng `ffprobe` rồi mới `mv` sang `up_*.mp4`.
4. **Dung sai lỗi (Fault Tolerance):** Frame lỗi retry 1 lần; nếu thất bại fallback sang Lanczos upscale và ghi log cảnh báo, tuyệt đối không để lọt frame đen làm hỏng phim.
5. **Môi trường qua đêm:** `caffeinate -dimsu`, cắm sạc MagSafe, mở nắp máy (tránh sleep của màn hình phụ), đặt máy trên mặt phẳng cứng thoáng khí.
6. **Giám sát viễn trắc:** Ghi nhận fps và nhiệt độ GPU theo từng chunk vào file log.
7. **Kiểm định cuối tập (End-of-Episode Audit):** Đảm bảo số frame 4K khớp 100% nguồn, độ lệch A/V drift < 1 frame.
8. **Bảo vệ SSD:** Toàn bộ frame PNG trung gian sinh và diệt trên RAM Disk; SSD chỉ nhận các file segment HEVC thành phẩm (~5GB/tập).

---

## 7. Kiến Trúc Subtitle Translation CLI (`subtrans/`)

```text
subtrans/
├── cli.py                  # Typer CLI entrypoint: check, models, run, compare, batch
├── pipeline.py             # Orchestrator: Stage A (Cast sheet) -> 1 -> 5 (Audit), Context +/-10
├── config.py / config.yaml # Cấu hình RPM, RPD, Model choices, Batch size (50)
├── cast_sheet.py           # Stage A: Khám phá nhân vật, đại từ xưng hô, xuất out/<name>.cast.yaml
├── models.py               # Pydantic models: Cue, GlossaryTerm, AddressRules, Glossary (Schema v2)
├── parser.py               # pysubs2: lọc OP/ED/Signs, gỡ tag ASS, align time overlap JA-EN (>=50%)
├── gemini_backend.py       # Primary: Google GenAI SDK (JSON Schema, uncertain flag, Context +/-10)
├── local_backend.py        # Fallback: Ollama (translategemma:12b, keep_alive: 0 khi giải phóng)
├── review.py               # Optional Post-edit: Ollama (gemma3:12b)
├── throttle.py             # RateLimiter (15 RPM sliding window), Exponential Backoff + Jitter
├── cache.py                # SQLite cache (.cache/translations.db) keyed by SHA-256
├── glossary.py             # Loader & Merger: Nạp Base glossary + Per-series Override
├── glossary.yaml           # Base glossary chuẩn: 12 style rules, address rules, 230+ anime terms
├── writer.py               # Xuất .srt chuẩn mẫu (1 dòng/cue, soft cap 80, max 107; --wrap N khi cần)
└── report.py               # Stage C: Báo cáo QA audit Markdown (out/<name>.audit.md & --review-file)
glossaries/                 # Thư mục chứa bảng thuật ngữ riêng cho từng bộ (glossary.<anime>.yaml)
subtrans.py                 # Root CLI runner executable (./subtrans.py ...)
```

### Bất biến kỹ thuật của `subtrans` (Cập nhật ADDENDUM 6 & 8):
1. **Thứ tự ưu tiên cốt lõi:**
   - (1) Không dịch sai / không bịa đặt: khi nghi vấn, dịch phương án an toàn và gắn cờ `"uncertain": true`.
   - (2) Văn phong tiếng Việt nói tự nhiên khi đọc to (không dịch thô/word-for-word).
   - (3) Đồng nhất tên riêng và đại từ xưng hô trong toàn tập (Stage A Cast Sheet).
   - (4) Chuẩn định dạng mẫu SRT `ReZERO_S04E19_Vietsub_Chuan.srt`.
2. **Quy chuẩn định dạng SRT (Sample Fidelity):**
   - Mặc định xuất **chính xác 1 dòng văn bản / cue** (không ép ngắt 2 dòng; chấp nhận độ dài lên tới 107 ký tự). Giới hạn mềm: 80 ký tự. Chỉ ngắt 2 dòng khi người dùng chỉ định `--wrap N`.
   - Dấu câu: dùng đúng 3 chấm ASCII `...` (không dùng ký tự unicode `…`), gạch ngang `—` khi bị ngắt lời, tối đa 2 dấu `!` hoặc `?` liên tiếp.
   - Câu ngắt qua 2 cue: câu dịch được phân chia tại ranh giới ngữ đoạn tự nhiên (cue trước kết thúc bằng `,` hoặc `...`, cue sau bắt đầu bằng chữ thường).
3. **Tiếng Nhật là chân lý (Source of truth):** Tiếng Anh chỉ dùng để phân biệt chủ ngữ, ngôi xưng hô, đối tượng và thành ngữ khi tiếng Nhật lược bỏ.
4. **Kế thừa Glossary Schema v2 & Stage A Cast Sheet:** Bảng nền `subtrans/glossary.yaml` luôn tự động nạp làm lớp cơ sở. Stage A tự động phân tích và tạo cast sheet `out/<name>.cast.yaml` một lần mỗi tập để đồng bộ đại từ xưng hô cho toàn bộ các batch.
5. **Bộ nhớ cục bộ an toàn:** Khi dùng mô hình cục bộ qua Ollama, chỉ duy nhất 1 mô hình được nạp vào VRAM/Unified Memory tại một thời điểm; sau khi xong lập tức gọi `keep_alive: 0` để giải phóng RAM.
6. **Bảo toàn giao dịch & Caching:** Từng câu thoại được lưu cache SHA-256 trong SQLite; nếu ngắt quãng hoặc gặp lỗi quota, có thể tiếp tục ngay lập tức với độ trễ 0s cho các câu đã dịch.
7. **Lựa chọn mô hình Gemini Cloud (Cập nhật ADDENDUM 8 - 2026-10-01):**
   - Mặc định: `gemini.model: "gemini-3.8-flash"`.
   - Fallback tự động: `gemini.fallback_models: ["gemini-3.7-flash", "gemini-3.6-flash"]`.
   - Lite model: `gemini.lite_model: "gemini-3.5-flash-lite"`.
   - Tuyệt đối không dùng `gemini-2.5-*` hay `gemini-2.0-*` làm mặc định (legacy / ngừng hỗ trợ).
   - Tuyệt đối không dùng alias `gemini-flash-latest` nếu không có cờ `--allow-latest-alias`.
   - Cấu hình Thinking: Dùng `thinking_budget=0` cho dòng 3.8/3.7/3.6 và `thinking_level=MINIMAL` cho dòng flash-lite để triệt tiêu độ trễ và tiết kiệm quota token đầu ra.
   - Chuỗi fallback & Bảo toàn đại từ: Mặc định `allow_silent_fallback: false`. Khi gặp 429 (Quota) hoặc 503 (Spike tải), pipeline tạm dừng (pause 30s) và retry trên model chính để bảo đảm 100% tính đồng nhất của đại từ toàn tập, tuyệt đối không tự ý hạ cấp xuống Lite ("tớ-cậu") hoặc Local giữa chừng. Chỉ chuyển model khi gặp lỗi vĩnh viễn (404/410) hoặc người dùng chỉ định rõ `--allow-fallback`.
   - Cast Sheet Unconditional Inclusion: Bảng nhân vật và quan hệ xưng hô của Stage A luôn được truyền vào mọi batch, không bị lọc bỏ khi câu thoại lược bỏ tên riêng (zero-pronouns).

---

## 8. Kết Quả Đo Đạc Kiểm Định Thực Tế Mới Nhất (Ground Truth Audit)

* **FFmpeg Version Pinning:** `ffmpeg version 8.1.2` (ghim và assert `raw_keep + raw_drop == total_frames`).
* **Kiểm tra độc lập số frame Deduplication (`sample_3min.mkv` - 4.360 frames):**
  - Dòng lệnh `ffmpeg -vf mpdecimate ... -f null -` xuất ra chính xác **2.174 frames** (49.86% raw kept).
  - Ép giữ ranh giới chunk 8s ($i \times 192$) bổ sung 12 frames $\rightarrow$ Tổng đưa vào AI: **2.186 unique frames**.
* **Audit Toàn Bộ 4.360 Frames (Local Patch SSIM Lưới 8x8 = 64 ô/frame):**
  - Global SSIM trung bình: **0.98061** (P1: 0.94340, Min toàn tập: 0.93286).
  - Local Patch SSIM (vùng tệ nhất mỗi frame): Trung bình **0.96996**, Min tuyệt đối **0.89905** (Frame #2723, Patch 6,1).
  - Sai lệch điểm ảnh: Global MAD trung bình **0.93 / 255** (0.36%), Max Patch MAD trung bình **1.38 / 255** (0.54%). Khẳng định 100% không mất chớp mắt hay khẩu hình miệng.
* **Thời gian thực tế chạy trọn vẹn 3 phút:** **37m 09s** (AI: 1.764s, Encode: 278s, Extract: 185s, Mux/Preflight: 15s). Khớp hoàn hảo với dự đoán ~36 phút của phân tích lý thuyết.
* **Kiểm tra luồng thành phẩm 4K:** Video 4.360 packets (lệch 0 frame, 0.000s drift), Audio 7.830 packets (khớp 100%), 18 fonts nhúng nguyên vẹn.
* **Kết quả True Blind Test (Người dùng chấm mù 3/3):** `realesr-animevideov3` đạt vị trí 1st tuyệt đối ở cả 3 cảnh, loại bỏ hoàn toàn ArtCNN và AnimeJaNai.


