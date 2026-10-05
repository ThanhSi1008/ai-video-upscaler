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
* **Đo Đạc Preflight Toàn Tập Thật (`Mushoku Tensei S3 - 14`, 34.046 frames - 1.35 GB):**
  - Quét hoàn tất trong **123.67 giây (2.06 phút)** @ 275.3 fps.
  - Raw kept frames (`mpdecimate`): **13.216 frames** (38.82%); Raw drop: **20.830 frames** (61.18%).
  - Ép giữ ranh giới chunk 8s ($i \times 192$): bổ sung 108 frames $\rightarrow$ **Tổng đưa vào AI: 13.324 unique frames (39.14%)**.
  - **Tỷ lệ loại bỏ trùng lặp thực tế toàn tập: 60.86%** (thay thế hoàn toàn con số 17.048 frames nội suy cũ).
  - **Dự toán thời gian chạy toàn tập thật:** 1 tập mất **~3h 01m** @ 1.24 fps; 2 tập qua đêm mất **~6h 03m**, **dư 1h 56m an toàn** trong cửa sổ 8 giờ.
* **Audit Khối Nhỏ $16 \times 16$ & $8 \times 8$ Pixel (Toàn bộ 2.174 dropped frames mẫu 3 phút):**
  - Phân tích 8.160 khối/frame: Mean max-block MAD **3.77 / 255** (1.48%), Median P50 **2.82**, P90 **7.35**, P95 **8.11**, P99 **9.42**, Max toàn tập **10.75 / 255** (4.21%).
  - Số frame có $16 \times 16$ MAD $\ge 12.0$ (ngưỡng hi=768/64): **0 frames (0.00%)**.
  - Thẩm định trực quan Top 30 worst frames (`out/dedup_audit_16x16/top_30_worst/`): Sai lệch chỉ là rung viền răng cưa 1-pixel và nhiễu nén H.264 trên nét tĩnh; khẳng định 100% không mất bất kỳ chớp mắt, khóe miệng hay cử động khuôn mặt nào.
* **Nghiên Cứu Đối Đầu PyTorch MPS vs NCNN Vulkan (100 frames chuẩn 1080p $\to$ 4K):**
  - Khám phá kiến trúc ncnn param: `realesr-animevideov3-x2` thực chất chạy mạng SRVGGNetCompact 4x rồi nội suy Bicubic 0.5 để ra 2x. Khớp trọng số `strict=True` với `realesr-animevideov3.pth`.
  - Đo sau warm-up với `torch.mps.synchronize()`: PyTorch MPS FP32 đạt **1.52 fps** (656.9 ms, nhanh hơn ncnn 21.6%); PyTorch MPS FP16 đạt **1.86 fps** (538.2 ms, **nhanh hơn ncnn +48.6%**).
  - Độ khớp hình ảnh: PSNR **51.05 dB**, MAD **0.493 / 255** (0.19%), Max diff 44.0 / 255 $\rightarrow$ Khớp gần như tuyệt đối với bản NCNN Vulkan.
* **Pipelined Triple-Overlap Concurrency & Bảo Vệ RAM Disk:**
  - Chạy gối đầu đồng thời: CPU Decode (720% CPU), Media Engine Encode (153% CPU), GPU Metal Infer (133% CPU).
  - GPU AI duy trì ổn định **1.27 steady fps** (không bị sụt giảm xuống 1.15 fps).
  - Dung lượng PNG 4K thực tế: lên tới **11–12 MB/frame**. Bổ sung cơ chế dynamic backpressure `shutil.disk_usage < 1200MB` chặn đứng nguy cơ chạm trần 4GB RAM Disk.
  - Sửa lỗi `start_total_time` đưa lên đầu pipeline: Tổng các stage cộng lại khớp đúng 100.0%.
* **Kiểm Định Subtitle Translation (Regex Validator Độc Lập Trên 43 Cues):**
  - Đính chính bối cảnh: Cue 42 (`今行くわ、ルーデウス！`) là Eris gào gọi Rudeus khi phi ngựa ("Ta/Em tới đây, Rudeus!"), không nói với Gal Farion; Cue 10 lỗi local model là gọi Eris là "Anh"; Rudeus - Eris chưa phải quan hệ vợ chồng chính thức trong tập này.
  - Điểm Regex Rules: `gemini-3.8-flash` đạt **10.0/10** (13/13 pass, 0 lỗi); `gemini-3.5-flash-lite` đạt **6.15/10** (5 lỗi suồng sã "tớ-cậu"); `translategemma_12b` đạt **6.92/10** (4 lỗi: gọi Eris là "Anh", xưng "chị-em", "đòn của anh", "Cháu đi đây").
  - Đã bổ sung chặn trên Pause-and-retry: tối đa 5 lần (chờ ~2.5m) trước khi ngắt an toàn để bảo toàn trạng thái khi cạn quota ngày.
* **Kết quả True Blind Test (Người dùng chấm mù 3/3 trên Liquid Retina XDR):** `realesr-animevideov3` đạt vị trí 1st tuyệt đối ở cả 3 cảnh, loại bỏ hoàn toàn ArtCNN và AnimeJaNai. Bỏ các lập luận MAD 0.52 phẳng.
* **Truy Nguyên Gốc Rễ 108 Frames Bị Bỏ (`vf_mpdecimate.c` C-loop Verification):**
  - Đối soát độc lập trên mặt phẳng 8-bit Y gốc (Luma): **0 / 108 frame vượt ngưỡng hi=768** (Min SAD: 658, Mean: 718.8, Median: 725.0, Max: 768).
  - Khám phá nguyên nhân sai số cũ: Script audit trước chuyển đổi YUV420p sang BGR24 rồi tính Gray float làm sai số nội suy Chroma upsampling đội lên thêm +110 đến +125 điểm SAD tại các viền màu. Thuật toán của FFmpeg 8.1.2 hoạt động hoàn toàn chính xác.
  - Đã xuất 108 contact sheets đa khung hình (`out/dedup_audit_8x8/`) sẵn sàng cho người dùng thẩm định trực quan.
* **Đo Đạc Preflight 4 Ngưỡng (`hi` ∈ {768, 640, 512, 384}):**
  - Mẫu 3 phút (4.360 frames): `hi=768` (2.186 frames, 28.7m), `hi=640` (2.301 frames, +1.5m), `hi=512` (2.458 frames, +3.6m), `hi=384` (2.679 frames, +6.5m).
  - Toàn tập 24 phút (34.046 frames): `hi=768` (13.324 frames, 2h 55m), `hi=640` (14.095 frames, **3h 05m, +10.1m**), `hi=512` (15.398 frames, 3h 22m), `hi=384` (17.557 frames, 3h 50m). `hi=640` là điểm ngọt kỹ thuật giữ thêm 771 frames an toàn.
* **Thiết Lập Baseline NCNN Tiling Mới (`tests/benchmark_ncnn_tiles.py`):**
  - Cấu hình tối ưu `-t 540 -j 1:2:2` đạt **1.427 fps (+18.9%)** với PSNR 66.39 dB so với auto-tile `-t 0` (1.200 fps). Triệt tiêu ~19.5% chi phí prepadding dư thừa.
* **Độ Khớp và Hiệu Suất PyTorch MPS:**
  - Vùng trung tâm (Interior) khớp NCNN đạt **60.02 dB** PSNR. Toàn bộ điểm lệch cực đại 44/255 nằm trọn trong dải viền 10px ngoài cùng do cơ chế reflection padding của tiling khác biệt.
  - FP16 vs FP32 đạt **61.74 dB** PSNR (0 NaN/Inf). Mức nén PNG 1 đạt **310 ms / frame** (nhanh gấp ~9 lần mức 6 cũ: 2.826 ms) với dung lượng 9.1 MB.
  - Đã xuất 5 cặp ảnh chấm mù A/B tại `out/mps_blind_ab/` và khóa giải mã `secret_key.json`.
  - **Soak Test 2.000 frame liên tục trên MPS (`out/mps_soak_2000_report.json`):** Hoàn tất trong **1.120,17 giây (18,67 phút)**. Tốc độ trung bình đạt **1.785 fps** (vượt chỉ tiêu $\ge 1.7\text{ fps}$). Bộ nhớ MPS duy trì phẳng tuyệt đối ở **61.19 MB**, Process RSS ổn định ở **36.0 MB** (hoàn toàn không rò rỉ bộ nhớ).
* **Phòng Vệ RAM Disk 2 Lớp & Kiểm Thử Hỗn Loạn (Chaos Test):**
  - Pre-allocation gate: `need = n_unique * size_est * 1.3` chặn trước nguy cơ chạm trần 4GB; Runtime gate dừng sạch khi dung lượng < 50MB (Unit test `test_ramdisk_worst_case.py` đạt 4/4).
  - Chaos test (`tests/chaos_test_resume.py`): Kịch bản `kill -9` giữa chunk & Resume đạt **PASS** (kế thừa chunk 1 nguyên vẹn, số frame thành phẩm khớp chính xác 767/767 frames qua `ffprobe -count_frames`); kịch bản ENOSPC đạt **PASS**.
  - Tự động xuất `audit.json` cuối tập và cảnh báo sụt giảm fps GPU > 8% cho 10 chunk liên tiếp.
* **Đo Đạc Lặp NCNN Tiling (`tests/benchmark_ncnn_repeats.py`):**
  - Cấu hình `-t 540 -j 1:2:2`: **1.417 ± 0.020 steady fps** (72.68s wall time, 720.2 MB peak RSS).
  - Cấu hình `-t 1080 -j 1:2:2`: **1.389 ± 0.016 steady fps** (73.92s wall time, 666.5 MB peak RSS). Đã ghim `-t 540 -j 1:2:2` làm mặc định cho NCNN fallback.
* **Soak Test End-to-End 6.000 Frame Trên PyTorch MPS FP16 (`out/mps_soak_6000_report.json`):**
  - Tổng thời gian: **3.570,27 giây (59,50 phút)**.
  - Tốc độ Trung vị (Median): **1.683 fps**, Phân vị P5 (Worst 5%): **1.677 fps**, Phân vị P95: **1.684 fps**, Mean: **1.681 fps** (độ lệch cực đại $\pm 0.005$ fps xuyên suốt 1 giờ chạy liên tục).
  - Đỉnh Driver Allocated: **2.090,81 MB**, Đỉnh Footprint tiến trình: **674,42 MB**.
  - Tăng trưởng Pageouts: **+133** (hoàn toàn không swap, an toàn cho SSD). Trạng thái nhiệt duy trì **Nominal** (không nghẽn nhiệt).
* **Chạy Trọn Vẹn `sample_3min` Bản Pipelined Với Engine MPS (`out/sample_3min_mps_pipelined.audit.json`):**
  - Wall-clock time hoàn tất: **23 phút 59 giây (1.439,63s)**.
  - So sánh: Nhanh hơn bản tuần tự (37:09) **13 phút 10 giây (-35,4%)**, nhanh hơn dự đoán NCNN pipelined (~30 phút) **6 phút 01 giây (-20,1%)**.
  - Tỷ lệ hoạt động GPU: **90,96%** (1.309,57s AI infer / 1.439,63s tổng thời gian; GPU idle chỉ **9,04%** bao gồm preflight scan, split và mux).
  - Kiểm định khớp khung hình: **4.360 / 4.360 frames** (đối soát độc lập qua `ffprobe -count_frames`, độ lệch thời lượng **0,001s = 0,024 frames**).
* **Đo Đạc & Tích Hợp Mô Hình AnimeJaNai HD V3 Compact (D1–D10):**
  - Đăng ký Registry `models/models.yaml` (CC-BY-NC-SA 4.0, sha256 `af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f`).
  - Kiểm định FP16 vs FP32 trên 25 frame đa dạng (`out/animejanai_fp16_vs_fp32_report.json`): PSNR trung bình **74,38 dB** (min 72,82, max 78,02), **0 NaN, 0 Inf**, Max pixel diff 1/255 $\rightarrow$ FP16 an toàn.
  - Soak Test End-to-End 500 frame (`out/animejanai_soak_500_report.json`): Hoàn tất trong **279,06 giây (4,65 phút)** @ **1,794 median fps** (P5: 1,794 fps).
  - Đỉnh Driver Allocated của AnimeJaNai: **1.066,81 MB** (giảm 51% so với 2.090,8 MB của animevideov3 do là 2x native, không sinh buffer 8K trung gian).
  - Dự toán ETA toàn tập 24 phút (13.324 unique frames) cho AnimeJaNai: **~2 giờ 08 phút**.
  - Kiểm thử D10 (`tests/test_model_selection_and_safety.py`): Đạt 4/4 PASS (Mặc định animevideov3, Tensor 3840x2160, Từ chối resume khác model nếu thiếu `--force`, Lỗi MPS không nhảy sang animevideov3 mà rơi về MPS FP32).
* **Giai Đoạn Phụ Đề Chạy Sau Upscale (Stage B):**
  - Cờ `--subs-after-upscale` (mặc định Bật): Toàn bộ hàng đợi upscale hoàn tất mới chuyển sang stage subtitle.
  - Kiểm tra an toàn bộ nhớ B3: `is_upscale_running()` từ chối nạp Ollama nếu phát hiện tiến trình upscale đang chạy; tự động gọi `keep_alive: 0` và xác minh `ollama ps` rỗng sau khi hoàn thành.
* **Nghiệm Thu Toàn Diện Đợt 3 (Mux Audit Tự Động, Quota Gemini PT & Tinh Chỉnh SSD):**
  - **Kiểm Tra Mux Tự Động Nhẹ Trong `audit.json` (`src/mux_audit.py`) & Sửa Lỗi Âm Thanh:**
    - *Nguyên nhân lỗi âm thanh:* Do thiếu cờ `-max_interleave_delta 0`, FFmpeg mặc định dồn toàn bộ gói audio thành khối lớn (clumped audio packets) lên đầu container thay vì xen kẽ đều theo DTS với video. Khi phát sau vài phút, buffer audio bị cạn dẫn đến mất tiếng hoàn toàn (**KHÔNG phải do timescale hay file WAV**).
    - *Giải pháp:* Đã khắc phục triệt để bằng cách ép cờ `-max_interleave_delta 0` trong toàn bộ lệnh mux container. File âm thanh tạm 0,27 GB hoàn toàn **không còn tồn tại** trong pipeline hiện tại do toàn bộ audio được Stream Copy trực tiếp từ file nguồn sang container đích.
    - Mỗi file MKV sau khi mux bắt buộc vượt qua 4 tiêu chuẩn kỹ thuật trong < 2 giây: (1) Đan xen gói tin 10s (Window Interleaving) trên 11 cửa sổ trải đều; (2) Seek test 7 mốc bằng `volumedetect`, lệch $\le 0.5$ dB (thực tế 0.00 dB); (3) MD5 luồng audio gốc trùng khớp từng bit (`1c943b02a84fa0b9bd5b7fb13dacbd15`); (4) Số stream, font chữ đính kèm (18 font TTF) và chapters trùng khớp nguyên bản. Nếu fail $\rightarrow$ `audit.json` ghi status `FAIL` và raise `RuntimeError` không báo hoàn tất. Test hồi quy `tests/test_mux_audit_regression.py` đạt PASS (bắt được file lỗi audio dồn khối).
  - **Định Lượng Quota Gemini Theo Giờ Thái Bình Dương (PT / UTC-7):**
    - Áp dụng hạn mức chưa xác minh (nguồn bên thứ ba, chờ đối soát AI Studio): `gemini-3.8-flash`, `3.7-flash`, `3.6-flash`, `3.5-flash` = **20 RPD**; `gemini-3.5-flash-lite`, `3.1-flash-lite` = **500 RPD**; Local (Ollama) = Unlimited.
    - Toàn bộ thông số RPM và TPM ghi rõ "chưa biết (cần đo)" hoặc "chưa xác minh" theo yêu cầu, loại bỏ triệt để các số 1.500 RPD và 15 RPM không rõ nguồn gốc.
    - Phân loại lỗi 429: Nếu `quotaId` chứa `PerDay` $\to$ chuyển sang model kế tiếp ngay lập tức, không sleep/pause; nếu `PerMinute` $\to$ chờ theo `retryDelay`.
    - Bộ đếm request theo ngày Thái Bình Dương (PT) dùng chuẩn `zoneinfo.ZoneInfo("America/Los_Angeles")`, triệt tiêu hoàn toàn logic trừ 7 giờ thủ công.
    - Mọi request (kể cả ping check, cast sheet discovery, batch translation) bắt buộc đi qua hàm bọc duy nhất `execute_and_log_api_request()` để ghi nhận vào SQLite `api_requests`.
    - Nguyên nhân `gemini-3.7-flash` không có cue ở tập 14: model liên tục trả về `503 UNAVAILABLE (High demand spike)` nên pipeline tự động chuyển sang `3.6-flash` và `3.5-flash-lite`.
  - **Củng Cố Quy Chuẩn Phụ Đề:** File phụ đề luôn được đặt tên đồng bộ `<tên file 4K không đuôi>.vi.srt` (ví dụ: `[SubsPlease] Mushoku Tensei S3 - 14 (1080p) [73CF848B]_4K.vi.srt`) cùng thư mục với video 4K, không bị ghi đè khi chạy hàng đợi; SQLite cache cấm lưu chuỗi `[Chưa dịch]` để `--subs-only` luôn cố gắng dịch lại; bổ sung tùy chọn `--subs-upgrade` dịch nâng cấp các câu Lite/Local bằng model cao hơn khi còn quota.
  - **Giả Thuyết Về Hiện Tượng Sụt Tốc (A4) & Thực Nghiệm Bác Bỏ:**
    - *Giả thuyết cũ:* Cho rằng hiện tượng tụt fps ở frames 101–400 (xuống 1.45–1.55 fps) trong bài test soak trước do macOS thu hồi bộ nhớ Metal (RSS giảm từ 173.8 MB xuống 41.4 MB).
    - *Thực nghiệm kiểm chứng độc lập (`tests/run_mps_soak_600.py`):* Chạy lại soak 600 frames trên PyTorch MPS FP16 trong môi trường biệt lập (bật `--benchmark-mode`, không có tác vụ nào khác chạy đồng thời). Kết quả đo lường thực tế:
      - Batch 1 (1–100): 1.866 fps | RSS: 338.66 MB | MPS Alloc: 61.19 MB
      - Batch 2 (101–200): 1.871 fps | RSS: 338.66 MB | MPS Alloc: 61.19 MB
      - Batch 3 (201–300): 1.872 fps | RSS: 338.66 MB | MPS Alloc: 61.19 MB
      - Batch 4 (301–400): 1.870 fps | RSS: 338.66 MB | MPS Alloc: 61.19 MB
      - Batch 5 (401–500): 1.870 fps | RSS: 338.66 MB | MPS Alloc: 61.19 MB
      - Batch 6 (501–600): 1.868 fps | RSS: 338.66 MB | MPS Alloc: 61.19 MB
      - **Tổng kết 600 frames:** Thời gian 320.94s, Overall 1.870 fps, Median 1.870 fps, P5 1.867 fps, Min 1.866 fps, Max 1.872 fps. *(Lưu ý: RSS 338,66 MB lặp lại qua các batch là số không cập nhật từ API process).*
      $\rightarrow$ Hiện tượng sụt tốc hoàn toàn không tái hiện. Bộ nhớ MPS và RSS phẳng tuyệt đối. Khẳng định hiện tượng sụt tốc trước đó chỉ là giả thuyết chưa đủ căn cứ; nguyên nhân thực tế là do tải ngoại lai (các tiến trình chạy nền đồng thời).
    - *Đo đạc AnimeJaNai:* Tách biệt 17 batch sạch (frames 1–1.700) đạt Trung vị **1.728 fps**, P5 **1.689 fps**, P95 **1.766 fps**, Mean **1.729 fps**, Min **1.689 fps**, Max **1.768 fps**. Báo cáo riêng 3 batch bẩn (batches 18–20, frames 1.701–2.000) bị ảnh hưởng bởi tiến trình nền quét preflight: Trung vị **1.474 fps**, P5 **1.459 fps**, Mean **1.471 fps**, Min **1.457 fps**, Max **1.482 fps**.
    - *Bổ sung cờ `--benchmark-mode`:* Khi bật, Scheduler và Preflight nền tự động từ chối chạy để bảo vệ môi trường đo lường thuần túy.
  - **Đo Đạc Độ Trễ Preflight Chạy Nền (`tests/measure_background_preflight_latency.py`):**
    - Đo trên 4 chunks (24s video `samples/sample_3min.mkv`):
      - Khi KHÔNG có preflight nền: Chunk 1: 1.540 fps, Chunk 2: 1.620 fps, Chunk 3: 1.630 fps, Chunk 4: 1.600 fps. Tổng wall-time: 391.84s.
      - Khi CÓ preflight nền (`nice -n 10 mpdecimate` quét full tập 14): Chunk 1: 1.560 fps, Chunk 2: 1.600 fps, Chunk 3: 1.610 fps, Chunk 4: 1.600 fps. Tổng wall-time: 396.42s.
      - Overhead: Chỉ +4.58s (+1.17%) cho 24s video; độ biến thiên FPS chỉ $\pm 0.020$ fps (~1.2%).
  - **Kiểm Kê Dung Lượng SSD & Phân Tích Máy Rảnh:**
    - *Đo lường máy rảnh:* Qua lệnh `iostat -Id disk0` trong 20 phút ngay sau khi tập 14 hoàn tất (01:42:32 đến 02:01:33), tốc độ ghi nền của hệ điều hành macOS (daemons `mds`, `mds_stores`, `logd`, APFS snapshots) đạt **290.07 MB/phút (~17.00 GB/giờ)**.
    - *Kiểm kê file pipeline ghi trên SSD:*
      - Dung lượng lưu trữ lâu dài: **1.862,59 MB (~1.82 GB)** gồm MKV 4K (1.861,13 MB), phụ đề `<tên 4K không đuôi>.vi.srt` (0.03 MB / 26.8 KB), `audit.json` (2.7 KB), `plan.json` (1.07 MB), cache SQLite `translations.db` (0.21 MB), logs (0.15 MB).
      - Đỉnh file tạm thời trong quá trình chạy: ~6.85 GB (178 source segments 1.41 GB + 178 upscaled segments 1.81 GB + intermediate merged MKV 1.81 GB + final mux 1.82 GB; file temp audio WAV 0.27 GB thuộc thiết kế cũ, pipeline hiện tại hoàn toàn không sinh file này). Toàn bộ frame PNG trung gian (~147.9 GB) được sinh và giải phóng trên RAM Disk ảo.
      - *Phân tích sai lệch:* Delta ghi disk0 đo được là 40.12 GB trong 2.72 giờ chạy tập 14. Trong đó nền macOS ước tính chiếm ~46.24 GB (theo baseline 17 GB/h). Bỏ hoàn toàn các khẳng định "khớp 100%"; sai số giữa mức đo và hoạt động nền được ghi nhận là chưa xác định, do baseline đo lúc đang xử lý hậu kỳ nên không phải máy rảnh hoàn toàn.
  - **Bộ Lập Lịch Hàng Đợi Qua Đêm & Quyết Định Deadline (`src/scheduler.py`):**
    - Sử dụng tốc độ AI đo đạc từ pipeline thực tế: Trung vị **1.66 fps**, P5 **1.62 fps** (lấy từ `audit.json` tập 14 C1), loại bỏ giả định 1.75 fps.
    - Tổng overhead: 660s (~11.0 phút: 360s video decode/encode/mux + 300s stage phụ đề và audit).
    - Quyết định deadline dựa trên P5 + overhead 660s + biên an toàn 7% ETA ($\text{decision\_eta} = (\text{n\_unique} / 1.62 + 660) \times 1.07$).
    - Bảng lập lịch hiển thị rõ ràng cột `Nguồn n_unique` ("đo (plan.json)" vs "ước lượng") cho từng tập.
    - Bổ sung cờ `--allow-partial`: Cho phép tập không kịp hạn chót chạy đến thời điểm deadline, dừng sạch sẽ tại ranh giới chunk, giữ lại các segment đã xử lý và tiếp tục chạy tiếp qua Checkpoint Resume ở phiên sau (chưa bật mặc định).
* **Nghiệm Thu Kiến Trúc Hạn Mức Gemini API & Tối Ưu Hóa Số Lượng Request (`subtrans/`):**
  - **Cấu hình `subtrans/models.yaml` (Single Source of Truth):**
    - Mặc định tạm chưa xác minh (nguồn bên thứ ba, chờ đối soát AI Studio): RPD cho `gemini-3.8-flash`, `3.7-flash`, `3.6-flash`, `3.5-flash` = **20 RPD**; `gemini-3.5-flash-lite`, `3.1-flash-lite` = **500 RPD**; `translategemma:12b` (local) = Unlimited.
    - Chuỗi ưu tiên nghiêm ngặt (tuyệt đối KHÔNG chứa model preview): `gemini-3.8-flash` $\to$ `3.7-flash` $\to$ `3.6-flash` $\to$ `3.5-flash` $\to$ `3.5-flash-lite` $\to$ `local`.
  - **Phân Loại Lỗi 429 Theo Trường Chi Tiết (`quotaId`, `quotaValue`, `retryDelay`):**
    - Nếu `quotaId` chứa `PerDay` $\to$ Đánh dấu model cạn hạn mức đến 00:00 giờ Thái Bình Dương (PT, 14:00 giờ VN trước 1/11, 15:00 giờ VN sau 1/11), tự động cập nhật `quotaValue` vào `models.yaml`, lưu trạng thái khóa vào bảng SQLite `model_exhaustion`, và **CHUYỂN MODEL KẾ TIẾP NGAY LẬP TỨC** (không sleep, không pause, không retry).
    - Nếu `quotaId` chứa `PerMinute` $\to$ Chờ đúng `retryDelay` (từ `RetryInfo`) rồi thử lại trên cùng model.
  - **Bộ Đếm Request Theo Ngày Thái Bình Dương Trong SQLite (`api_requests`):**
    - Tính toán mốc reset chính xác bằng `zoneinfo.ZoneInfo("America/Los_Angeles")`, tự động thích ứng với Daylight Saving Time.
    - Hiển thị bảng hạn mức và cảnh báo trực tiếp trong `audit.md` mỗi tập và trong bảng kế hoạch của `scheduler.py`.
  - **Tối Thiểu Hóa Số Request (Mục tiêu $\le 5$ requests/tập):**
    - Kích thước batch mở rộng lên **100–150 cues** (mặc định: 100 cues). Tập anime ~300 dialogue cues chỉ tốn đúng 3 batch requests.
    - Kiểm tra tính toàn vẹn (cue count & ID alignment): Nếu sai lệch số lượng hoặc thứ tự ID, cơ chế **Bisect** tự động tách đôi batch ($N \to N/2 + N/2$) và theo dõi `split_rate_pct`.
    - **Series Cast Sheet (`<series>.cast.yaml`):** Lưu trữ nhân vật theo tên bộ phim (tự động tách bằng regex bỏ release group/số tập). Các tập tiếp theo tự động kiểm tra delta nhân vật; nếu không có nhân vật mới $\to$ Tái sử dụng trực tiếp với **0 request API**, đưa tổng request mỗi tập về mức lý tưởng 3 requests.
  - **Bảo Vệ Hạn Mức Qua Đêm & Cờ `--dry-run`:**
    - Tuyệt đối cấm chạy dịch thử nghiệm gọi API thật trong khoảng từ 14:00 đến đêm chạy.
    - Cờ `--dry-run` trên CLI và Pipeline mô phỏng toàn bộ logic dịch, parse, kiểm tra ID, tách batch và xuất file mà không tốn bất kỳ lượt gọi API nào.
  - **Báo Cáo Minh Bạch 3 Lớp:** Mọi số liệu trong `audit.md` và console bắt buộc phân loại rõ ràng: **SỐ ĐO (MEASURED)** (log SQLite, `quotaValue` lỗi), **ƯỚC LƯỢNG (ESTIMATED)** (dự toán ETA/requests theo công thức), **CHƯA XÁC MINH (UNVERIFIED)** (hạn mức tạm thời chờ trang AI Studio rate-limit).
* **Nghiệm Thu Toàn Diện Vòng Sửa Sau Đêm 1 (Scheduler, Auto-Offset & Night Report):**
  - **Hiệu Chỉnh Overhead & Ước Lượng Lập Lịch (`src/scheduler.py`, `upscale.py`):**
    - Overhead video chuẩn hóa: **930,0s (~15,5 phút)** (hiệu chỉnh theo 2 tập đêm 1 [691s–1166s], cỡ mẫu n=2) + 300s phụ đề = 1230s (~20,5 phút).
    - Ước lượng kept frames cho tập chưa quét preflight: Bỏ mức 39,1% cũ, áp dụng **60,0% kept frames** làm giá trị thận trọng (dựa trên 4 tập đo đạc thực tế: 39,1%, 46,0%, 54,4%, 61,3%; trung bình ~50,0%). Bảng quyết định deadline hiển thị nhãn `"ước lượng thận trọng (n=4)"` và tự động cập nhật số đo thực tế khi preflight nền hoàn tất.
    - Sức chứa thực tế cập nhật trong README: **~2 tập mỗi đêm** (~3 giờ/tập ở $n_{unique} \approx 17.200$). 3 tập chỉ hoàn tất trọn vẹn khi tổng $n_{unique}$ đủ nhỏ.
  - **Cơ Chế Tự Động Nhận Diện Lệch Thời Gian Phụ Đề (`--auto-offset`):**
    - Mặc định: **TẮT** (chỉ log). `--sub-offset-ms` thủ công luôn luôn ưu tiên.
    - Điều kiện áp dụng khắt khe 4 lớp: (1) $|offset| \ge 1,0$s; (2) Số cặp ghép hợp lệ $\ge 30$; (3) Độ chênh cực đại giữa 5 phần thời gian $\le 0,5$s (chống trôi dạt lũy tiến); (4) Tỷ lệ khớp tăng $\ge 15$ điểm phần trăm sau bù.
    - Kiểm định thực tế 3 tập: Tập 1 (|offset| 0.148s < 1.0s) ➔ Không áp dụng; Tập 2 (|offset| 0.098s < 1.0s) ➔ Không áp dụng; Tập 3 (offset -11141 ms $\approx -11154$ ms, 285 cặp, drift 0.029s $\le 0.5$s, khớp 55.9% ➔ 85.3%, +29.4 điểm) ➔ Tự động áp dụng thành công.
  - **Lệnh Sinh Báo Cáo Ca Chạy Đêm (`./upscale.sh --night-report [phiên]`):**
    - Chế độ chỉ đọc, tuyệt đối không gọi GPU hay API. Tự động phân tích CSV telemetry TRONG cửa sổ ca, log chi tiết từng chunk, `plan.json`, `audit.json` và SQLite `api_requests`.
    - Xuất đầy đủ: GPU active ratio theo công thức chuẩn (`total_ai_time / episode_wall_time`), so sánh tốc độ giờ đầu vs giờ cuối, top 10 chunk chậm nhất, phân bổ cue theo model, thống kê 503/thành công theo giờ PT và danh sách việc cần làm bằng tay.
  - **Tự Động Bật/Tắt Giám Sát Telemetry Trong `upscale.sh --queue`:**
    - Tự động chạy `scripts/soak_monitor.sh` nền khi phát hiện cờ `--queue`, ghi nhận mốc `[SESSION START]` và `[SESSION END]`.
    - Cơ chế `trap` trong bash đảm bảo thu hồi sạch tiến trình monitor khi hàng đợi hoàn tất hoặc bị ngắt (`Ctrl+C`, `SIGTERM`), không để lại tiến trình mồ côi.
  - **Nghiệm Thu Vòng Sửa 2 (Queue Auto-Offset, Subtitle Embedding & Timezone PT):**
    - **Queue Auto-Offset Mặc Định:** Trong chế độ hàng đợi (`--queue`), `--auto-offset` được **BẬT MẶC ĐỊNH** (thêm `--no-auto-offset` để tắt). Từng tập được tính toán bù lệch độc lập (Tập 1, 2 không áp dụng; Tập 3 tự động bù -11.141ms). Offset đã dùng và lý do kỹ thuật được tự động ghi vào `audit.md` của từng tập.
    - **Bảo Vệ Cờ Offset Thủ Công (`--sub-offset-ms` & `--sub-offset-target`):** Cờ `--sub-offset-ms` trong `--queue` được bảo vệ: nếu không có `--sub-offset-target`, hệ thống in cảnh báo và bỏ qua áp dụng toàn cục để tránh làm hỏng timeline của các tập khác; khi có `--sub-offset-target`, chỉ áp dụng cho đúng tập chỉ định.
    - **Khắc Phục Lỗi Nhúng Phụ Đề MKV (`src/ffmpeg_ops.py`):** Bổ sung `import shutil` bị thiếu. Lệnh nhúng phụ đề `.vi.srt` vào container 4K MKV qua stream copy giờ đây thành công 100%, bảo toàn trọn vẹn HEVC 10-bit, audio gốc, phụ đề cũ, font chữ và chapters.
    - **Chuẩn Hóa Thống Kê Giờ PT (`src/night_report.py`):** SQLite lưu `created_at` theo giờ UTC; hệ thống sử dụng `zoneinfo.ZoneInfo("America/Los_Angeles")` để chuyển đổi chuẩn xác sang PDT/PST (khung 17:00–18:00 PDT ngày 2026-10-04, chiều tối Chủ Nhật). Bỏ nhận định phỏng đoán "giờ cao điểm Bắc Mỹ", phản ánh trung thực dữ liệu đo đạc.







