# AGENTS.md — AI Agent Guidelines & Architecture Compass

> **Kim chỉ nam dành cho các AI Agent khi đọc, bảo trì và phát triển codebase này.**
> **Quy tắc bắt buộc:** Ngắn gọn, chuẩn xác, không suy diễn; cập nhật file này sau mỗi lần tái cấu trúc hoặc thay đổi kiến trúc code; **BẮT BUỘC commit chi tiết sau MỖI yêu cầu** (nêu rõ nguồn yêu cầu, việc đã làm, lỗi đã fix, số liệu thực nghiệm).
> **Chỉ mục tài liệu chuyên sâu:** [Video Pipeline](file:///Users/xis108/Desktop/upscale/docs/video-pipeline.md) | [Dedup & DIS](file:///Users/xis108/Desktop/upscale/docs/dedup-architecture.md) | [Subtrans](file:///Users/xis108/Desktop/upscale/subtrans/AGENTS.md) | [Audit & Safety](file:///Users/xis108/Desktop/upscale/docs/audit-and-safety.md) | [Engineering Log](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md).

---

## 1. Mục Tiêu & Ràng Buộc Cốt Lõi (Core Invariants)

* **Thiết bị vận hành:** Apple Silicon MacBook Pro M3 Pro (11-core CPU, 14-core GPU, 18GB Unified Memory).
* **Môi trường bắt buộc:** **100% CỤC BỘ trên macOS**. Tuyệt đối không đề xuất Cloud GPU, Colab, Kaggle.
* **Tôn chỉ Pareto:** Chất lượng hình ảnh cân bằng với thời gian. Không hy sinh nét vẽ anime lấy tốc độ rẻ tiền; không lãng phí tài nguyên cho cải thiện vô hình.
* **Sức chứa thực tế:** **~2 tập mỗi đêm** (~3 giờ/tập với ~17.000 unique frames @ 1.68 fps sustained sau deduplication).

---

## 2. Lệnh CLI Cơ Bản & Bảng Cờ (Commands & Flags)

### A. Lệnh Vận Hành Thường Dùng
```bash
# Nâng cấp 1 tập video đơn lẻ (tự động nhận track phụ đề cùng thư mục)
./upscale.sh path/to/source.mkv -o ~/Documents/Upscaled/

# Chạy hàng đợi nhiều tập qua đêm (tự động bật auto-offset và lập lịch deadline 8h)
./upscale.sh --queue ep1.mkv ep2.mkv -o ~/Documents/Upscaled/

# Tiếp tục tập bị gián đoạn (tự động nhận thư mục checkpoint .temp)
./upscale.sh path/to/ep3.mkv -o ~/Documents/Upscaled/

# Dịch phụ đề độc lập (không chạy AI video)
./upscale.sh --subs-only path/to/source.mkv -o ~/Documents/Upscaled/

# Chạy kiểm thử tự động toàn diện
./venv/bin/pytest
```

### B. Bảng Cờ CLI Chính (`upscale.py`)
| Cờ CLI | Mặc định | Ý nghĩa & Hành vi |
| :--- | :--- | :--- |
| `-o`, `--output` | `~/Documents/Upscaled/` | Thư mục hoặc file đích xuất video 4K |
| `--queue` | `None` | Danh sách các file video chạy nối tiếp qua đêm |
| `--engine` | `mps` | Engine suy luận: `mps` (PyTorch Metal FP16) hoặc `ncnn` (Vulkan MoltenVK) |
| `--model` | `animevideov3` | Model Super-Resolution đăng ký trong `models/models.yaml` |
| `--allow-partial` | `False` | Cho phép dừng tập an toàn ở ranh giới chunk 8s khi chạm deadline ca đêm |
| `--auto-offset` | Bật trong `--queue` | Tự động đo và bù độ lệch thời gian phụ đề tiếng Nhật - Anh |
| `--no-auto-offset` | `False` | Ép tắt tính năng auto-offset trong hàng đợi |
| `--sub-offset-ms` | `0` | Bù offset phụ đề thủ công theo mili-giây (luôn có độ ưu tiên cao nhất) |
| `--no-embed` | `False` | Tắt bước nhúng phụ đề vào container MKV thành phẩm |
| `--keep-temp` | `False` | Giữ lại thư mục tạm `.temp_<basename>` (mặc định: tự động dọn dẹp sạch khi Mux PASS) |
| `--clean-temp`, `--cleanup` | `False` | Quét dọn toàn bộ thư mục tạm và gom file audit/cast trong thư mục xuất xưởng rồi thoát |
| `--dedup-policy` | `default` | Chính sách dedup: `default` (768:320:0.33), `sensitive-a` (512:160:0.15), `sensitive-b` (384:128:0.10), `motion` (kèm phase correlation), `drift` (tích lũy), `dis` (dòng chảy quang học DIS bắt 100% zoom/lia chậm), `off` (tắt) |
| `--dis-thresh` | `0.30` | Ngưỡng dòng chảy quang học DIS tính bằng pixel 1080p (`0.20` - `0.50`) |
| `--dry-run` | `False` | Chạy mô phỏng toàn bộ quy trình mà không gọi AI hay API |

---

## 3. Kiến Trúc Thư Mục & Trách Nhiệm Module

```text
upscale/
├── upscale.py              # CLI Entrypoint (Argparse, Queue orchestrator, Checkpoint Resume)
├── upscale.sh              # Shell launcher native (tự nhận virtualenv hoặc python3)
├── subtrans.py             # Subtitle Translation CLI Runner
├── shows.yaml              # Ánh xạ mô hình SR theo từng bộ anime (người dùng cấu hình)
├── pytest.ini              # Cấu hình Pytest (testpaths = tests subtrans/tests)
├── requirements.txt        # Danh sách thư viện Python phụ thuộc
├── README.md               # Hướng dẫn sử dụng & CLI options
├── LICENSE                 # Giấy phép mã nguồn mở
├── AGENTS.md               # Kim chỉ nam kiến trúc & các bất biến kỹ thuật
├── src/                    # Pipeline phóng to video (Triple-buffering, Metal GPU, FFmpeg, RAM Disk)
│   ├── config.py           # Single source of truth: Hằng số, paths, params VideoToolbox & RAM Disk
│   ├── ramdisk.py          # RAMDisk Context Manager: mount/unmount HFS+ RAM Disk ảo (Zero SSD Wear)
│   ├── ffmpeg_ops.py       # Thao tác FFmpeg/FFprobe chuẩn phòng thu (BT.709 10-bit, Mux, Embed Subtitles)
│   ├── mps_engine.py       # PyTorch MPS FP16 Native Apple Silicon inference worker (Engine chính)
│   ├── ai_engine.py        # NCNN Vulkan via MoltenVK (Metal API) inference worker (Engine fallback)
│   ├── models_resolver.py  # SR Models Registry resolver & weights loader
│   ├── mux_audit.py        # Stream copy integrity audit & font count verifier
│   ├── dedup.py            # Deduplication Engine: Preflight mpdecimate scan & Concat Re-expansion
│   ├── preflight.py        # CaffeinateContext (chống sleep), Check AC/Pin, Check đĩa trống, macOS Notifs
│   ├── pipeline.py         # Orchestrator: Triple-buffering, Rolling Segments, Checkpoint Resume
│   ├── scheduler.py        # Dynamic overnight queue scheduler & ETA predictor
│   ├── night_report.py     # Báo cáo viễn trắc tổng kết phiên chạy qua đêm Markdown
│   ├── output_sanity.py    # Output Sanity Guard: Chốt chặn kiểm định đầu ra (|ΔY| <= 6, corr >= 0.98)
│   ├── cleanup.py          # Auto-cleanup: Dọn dẹp thư mục tạm (.temp_*), symlinks & gom gọn audit/
│   └── logger.py           # Structured telemetry logging (logs/*.log & logs/latest.log symlink)
├── subtrans/               # Engine dịch thuật phụ đề (xem subtrans/AGENTS.md)
├── bin/realesrgan/         # Universal Mach-O binary & NCNN models (.param / .bin)
├── models/                 # Super-Resolution models registry (models.yaml)
├── glossaries/             # Bảng thuật ngữ riêng từng bộ anime (glossary.<anime>.yaml)
├── samples/                # Video mẫu phục vụ kiểm thử (sample_30s.mkv, sample_3min.mkv)
├── test_frames/            # 3 frame ảnh mẫu chuẩn (Action, Close-up, Scenery)
├── tests/                  # Bộ kiểm thử tự động, benchmark hiệu năng, soak test, chaos test
├── docs/                   # Tài liệu chuyên sâu: video-pipeline.md, dedup-architecture.md, audit-and-safety.md, engineering-log.md
└── logs/                   # Log chi tiết có cấu trúc & báo cáo tổng kết đêm
```

---

## 4. Những Bất Biến Kỹ Thuật (Do NOT Break)

1. **Bảo vệ NVMe SSD qua RAM Disk ảo:**
   * Mọi khung hình trung gian dạng PNG (cả trích xuất 1080p và xuất 4K AI) **bắt buộc phải ghi trên RAM Disk ảo HFS+** (`/Volumes/RAMDisk`).
   * Sau khi mỗi phân đoạn (chunk 8s) mã hóa xong, phải dọn sạch frame PNG trên RAM Disk ngay lập tức.
   * SSD chỉ lưu trữ các segment HEVC đã nén (`up_*.mp4`) trong thư mục `.temp_<basename>` để bảo toàn khả năng resume.
2. **Bit-exact Stream Copy & Cờ Mux Phòng Thu:**
   * Video 4K thành phẩm **bắt buộc Stream Copy nguyên bản**: Audio đa kênh, Phụ đề mềm (ASS/SRT), Font chữ nhúng (TTF/OTF) và Chapters.
   * Lệnh mux bắt buộc dùng cờ:
     `-map 0:v:0 -map 1:a? -map 1:s? -map 1:t? -map_chapters 1 -c copy -max_interleave_delta 0`
   * Cờ `-max_interleave_delta 0` là bắt buộc để ngăn lỗi tràn buffer và mất tiếng cục bộ sau vài phút phát lại.
3. **Chuỗi ma trận màu sắc chuẩn xác (BT.709 Two-way Matrix):**
   * Trích xuất: `scale=in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=pc` (YUV Limited -> RGB24 Full).
   * Mã hóa: `scale=in_color_matrix=bt709:in_range=pc:out_color_matrix=bt709:out_range=tv` (RGB24 Full -> YUV Limited `p010le`).
   * Metadata gắn cờ: `-color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -tag:v hvc1`.
4. **Vòng đời an toàn (Process Lifecycle & Auto Cleanup):**
   * `caffeinate` và `hdiutil` (RAM Disk) phải luôn nằm trong `try...finally` hoặc Context Manager với `atexit` để thu hồi sạch sẽ kể cả khi bị `Ctrl+C` (SIGINT).
5. **Độc quyền GPU & Điều phối tài nguyên:**
   * Tại một thời điểm chỉ duy nhất 1 tiến trình infer GPU Metal được hoạt động.
   * Pipeline chạy gối đầu 3 tầng: CPU Decode (chunk N+1) và Apple Media Engine Encode (chunk N-1) chạy song song với GPU Infer (chunk N).
   * Tuyệt đối không nạp mô hình cục bộ Ollama khi tiến trình Upscale GPU đang hoạt động.

---

## 5. Ranh Giới Hành Động Cho AI Agent (Agent Boundaries)

* **Kỷ Luật Commit Từng Yêu Cầu (Mandatory Commit Per Request):**
  * Sau khi hoàn thành **MỖI** yêu cầu (dù từ Người dùng trực tiếp hay từ AI nghiên cứu), **bắt buộc phải commit Git ngay lập tức**. Tuyệt đối không dồn gộp nhiều yêu cầu hoặc nhiều đợt sửa đổi vào một commit chung.
  * Thông điệp commit (commit message & body) phải trình bày rõ ràng 3 yếu tố:
    1. **Nguồn yêu cầu:** Do Người dùng yêu cầu hay AI nghiên cứu đề xuất.
    2. **Hành động cụ thể:** Đã làm gì, đã fix những lỗi gì, đã tối ưu hoặc tái cấu trúc những module nào.
    3. **Số liệu & Kết quả:** Đính kèm số liệu kiểm chứng (kết quả `pytest`, dung lượng bộ nhớ giải phóng, fps, trạng thái audit, v.v.).
  * Điều này đảm bảo lịch sử Git phản ánh chính xác timeline tiến độ, giúp trích xuất báo cáo kỹ thuật tức thì mà không cần lục tìm phức tạp.
* **Bảo vệ Gemini API Key:** Key lấy từ biến môi trường `GEMINI_API_KEY` hoặc file `.env`. Nghiêm cấm in, ghi log, hoặc commit API key ra màn hình và file lưu trữ.
* **Bảo vệ Quota Ban Ngày:** Tuyệt đối không chạy lệnh gọi API Gemini thật trong khoảng từ **14:00 đến đêm chạy**. Mọi kiểm thử chức năng bắt buộc dùng cờ `--dry-run` hoặc stub backend.
* **Bảo toàn Checkpoint Resume:** Tuyệt đối không xóa hay sửa đổi các file trong thư mục `.temp_<basename>` trên SSD khi chưa có sự xác nhận của người dùng.
* **Hỏi Ý Kiến Trước Khi Thực Hiện:**
  * Chạy các tác vụ GPU/AI dài (> 100 frames).
  * Thao tác thủ công mount/unmount ổ đĩa hệ thống bằng `hdiutil`.
  * Thay đổi model SR mặc định hoặc các tham số chất lượng mã hóa VideoToolbox.

---

## 6. Bảng Quyết Định Kỹ Thuật Đã Chốt & Chỉ Mục Tài Liệu Chuyên Sâu

> **Lưu ý AI Agent:** Để tiết kiệm context token, chi tiết triển khai và số đo thực nghiệm được tách vào các tài liệu chuyên biệt bên dưới. Khi cần bảo trì hoặc phát triển phân hệ nào, hãy đọc tài liệu tương ứng qua đường dẫn file.

### A. Phân Hệ Phóng To Video AI & Metal GPU Engine — [docs/video-pipeline.md](file:///Users/xis108/Desktop/upscale/docs/video-pipeline.md)
* **Mô hình Super-Resolution:** Chọn `realesr-animevideov3` (thắng 3/3 cảnh trong True Blind Test trên Liquid Retina XDR; chi tiết: [docs/engineering-log.md#2](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#2)).
* **Engine suy luận chính:** PyTorch MPS FP16 Native (`src/mps_engine.py`, 1.68 fps sustained soak; NCNN Vulkan đóng vai trò fallback; chi tiết: [docs/engineering-log.md#4](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#4)).
* **Đính chính mô hình AnimeJaNai:** Thêm phép cộng nearest $\times 2$ residual tại tag `janai-fix` (`F.interpolate(x, scale_factor=2.0, mode='nearest')`), khớp bit-exact 0.0 với spandrel CPU chuẩn (PSNR 75–158 dB).
* **Bảo vệ SSD & Triple-buffering:** RAM Disk ảo HFS+ 4GB (`/Volumes/RAMDisk`), dọn sạch frame sau từng chunk 8s. CPU Decode (chunk N+1) và Media Engine Encode (chunk N-1) chạy song song với GPU Infer (chunk N).
* **Màu sắc & Stream Copy:** Ma trận BT.709 hai chiều (Limited YUV $\leftrightarrow$ Full RGB24), gắn cờ `-tag:v hvc1`, cờ mux bắt buộc `-max_interleave_delta 0` copy nguyên bản audio, subtitle, fonts và chapters.
* **Khóa GPU:** Dùng file khóa `/tmp/upscale_gpu.lock` bảo đảm độc quyền GPU theo từng chunk infer.

### B. Phân Hệ Deduplication Preflight & Dòng Chảy Quang Học DIS — [docs/dedup-architecture.md](file:///Users/xis108/Desktop/upscale/docs/dedup-architecture.md)
* **Ngưỡng mặc định:** `hi=768:lo=320:frac=0.33` (giảm 60.86% thời gian GPU; 0% lỗi khẩu hình/chớp mắt trên mẫu kiểm tra; chi tiết: [docs/engineering-log.md#3](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#3)).
* **Khắc phục lệch khung hình BD Remux GOP dài:** `split_video_segments` stream copy trước, `get_segments_frame_counts` đo frame thực tế theo Keyframe GOP, ép giữ frame đầu mỗi chunk, tái tạo `runs` khớp 100% số frame.
* **Thất Bại Của Drift Trên Zoom:** Chính sách `drift` (dựa trên Phase Correlation) **thất bại trên chuyển động thu phóng (zoom)** do các vector dãn nở xuyên tâm đối xứng qua tâm triệt tiêu lẫn nhau về 0; con số báo cáo 0% đứng hình trước đây là sai lệch do đo nhầm cửa sổ hoặc kích hoạt nhầm trên nhiễu.
* **Nhận Diện Chuyển Động Qua DIS Optical Flow:** Thuật toán dòng chảy quang học DIS (Pan/Tilt translation, Radial expansion zoom in/out, Local/diffuse $P_{90}$ và $\ge 5\%$ diện tích) kết hợp Candidate Drop Gating đạt ~430 fps trên CPU M3 Pro. Khắc phục các đoạn lia máy và zoom mà `mpdecimate` bỏ sót. Hỗ trợ cờ `--dedup-policy dis` và `--dis-thresh 0.30`.
* **Tái tạo khung hình 4K (Re-expansion):** FFmpeg Concat Demuxer theo thời lượng `runs` đại diện + `fps=24000/1001` (triệt tiêu toàn bộ chi phí đĩa của hardlinks).
* **Chốt chặn Resume Cache:** `src/dedup.py:load_or_create_plan` từ chối resume nếu chính sách (`dedup_policy`), `dis_thresh`, `drift_delta` hoặc phân bố chunk trong cache không khớp với phiên chạy hiện tại.

### C. Phân Hệ Dịch Thuật Phụ Đề Subtrans — [subtrans/AGENTS.md](file:///Users/xis108/Desktop/upscale/subtrans/AGENTS.md)
* **100% Cloud API:** Bỏ hoàn toàn model local (Ollama/TranslateGemma), chạy 100% Gemini Flash-Lite / Flash qua Google GenAI SDK.
* **Đồng nhất xưng hô & Bảo tồn Romaji:** Stage A Cast Sheet (`out/<name>.cast.yaml`) nạp vô điều kiện vào mọi batch; giữ nguyên tên Romaji khi đối thoại gọi tên (vocative: Fran, Jean, Nell, Reirin).
* **Hiển thị tối ưu & CC Splitting:** Tự động gộp 2 người nói cùng lúc thành Dual Dialogue 2 dòng (`- Lời 1\n- Lời 2`), gộp câu ngắn (<1.2s), thời lượng đọc tối thiểu $\ge 1.2\text{s} - 1.4\text{s}$, khoảng nghỉ 40ms chống flicker. Tách Closed Captions đa người nói `- (Người 1)\N- (Người 2)` thành các event độc lập. Lọc sạch sound cue tiếng cười anime.
* **Căn chỉnh tự động:** Thuật toán `--auto-offset` kiểm định 4 tầng (ngưỡng 0.5s, pairs $\ge 30$, quintile drift $\le 0.5$s, match rate tăng $\ge 15\%$).
* **Evaluation Harness P0 & Schema CoT / Linter V1 P1:** Chỉ số tất định (`subtrans.py eval`), schema suy luận bối cảnh trước khi dịch (`speaker`, `addressee`, `mode`, `en_conflict`), Linter V1 kiểm tra 6 tiêu chuẩn phòng thu.
* **Smart Cloud Cascade P2 & Subtitle RAG P3:** Điểm khó $S(cue)$, Draft Pass Lite kết hợp Repair Pass Flash (cổng Re-lint so sánh theo danh tính `_ident`, trần ngân sách Top-K, `thinking_budget=0`). Kho tri thức SQLite `.cache/knowledge.db` lưu 4.144 câu triplet, phân cấp `gold`/`reviewed`/`draft`, cách ly `rag_hash`, loại trừ tập đang đo (`exclude_episode`).
* **Multi-Key Rotation & Dừng Chạy Êm:** Nạp song song nhiều API Keys xoay vòng khi gặp 429 RPD; Graceful Exit khi cạn hạn ngạch ngày, bảo toàn 100% bản dịch trong cache SQLite.

### D. Phân Hệ Kiểm Định Toàn Vẹn, Sanity Guard & An Toàn — [docs/audit-and-safety.md](file:///Users/xis108/Desktop/upscale/docs/audit-and-safety.md)
* **Output Sanity Guard:** `src/output_sanity.py` kiểm tra sau chunk 1 và mỗi 10 chunks sau ($|\Delta Y| \le 6/255$, tương quan $\ge 0.98$ so với bicubic); dừng khẩn cấp và cấm mux nếu rớt, triệt tiêu nguy cơ video đen/mờ.
* **Cổng Kiểm Toán Mux Audit:** `src/mux_audit.py` đối soát bit-exact stream copy, MD5 track âm thanh, số lượng phụ đề, font chữ đính kèm và tổng số khung hình `video_frame_count`.
* **Tự Động Dọn Dẹp & Thu Hồi SSD:** `src/cleanup.py` tự động kích hoạt khi Mux Audit PASS; xóa sạch `.temp_<basename>` (thu hồi ~6.6 GB/tập), gom file báo cáo vào thư mục con `audit/`. Cờ `--keep-temp` và `--clean-temp`.
* **Cổng Backpressure & Chống ENOSPC:** Sàn an toàn `DEFAULT_PNG_UNIT_SIZE_MB = 16.0 MB`, công thức `need = n_unique \times \max(...) \times 1.2`, retry 2 lần, đánh dấu `failed-resumable`.
* **Vòng Đời An Toàn:** `CaffeinateContext` chống sleep máy, tự động thu hồi RAM Disk khi `Ctrl+C` hoặc kết thúc tiến trình.

### E. Nhật Ký Kỹ Thuật & Số Đo Benchmarks Chi Tiết — [docs/engineering-log.md](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md)
* Lưu trữ chi tiết: True Blind Test Liquid Retina XDR, soak test MPS FP16 2.000 / 6.000 frames, viễn trắc ca đêm (Night 1 & Night 2), số đo MAE đứng hình zoom 44–46s, bảng so sánh các chính sách dedup, và nghiệm thu đối đầu 4 cấu hình C0–C3 subtrans.
---

## 7. Tiêu Chí Nghiệm Thu (Definition of Done)

Một phiên làm việc hoặc mã nguồn được coi là hoàn tất khi thỏa mãn đầy đủ các điều kiện sau:
1. **Kiểm thử tự động:** Chạy `pytest` đạt **PASS 100%** cho toàn bộ các bài test không đánh dấu `@pytest.mark.gpu`.
2. **Kiểm định Mux:** Hàm `verify_mux_integrity` trả về `status: "PASS"` (khớp số kênh audio, số luồng sub, và số lượng font chữ đính kèm).
3. **Bảo toàn khung hình:** `ffprobe -count_frames` trên video 4K thành phẩm khớp chính xác từng khung hình so với video nguồn gốc.
4. **Trạng thái Git & Kỷ luật Commit:** Mỗi yêu cầu hoàn thành đều được commit riêng biệt với thông điệp rõ nguồn, nội dung làm/fix và số liệu; `git status` sạch hoàn toàn, không có file tạm bị bỏ sót; tag phiên bản được cập nhật tương ứng.
