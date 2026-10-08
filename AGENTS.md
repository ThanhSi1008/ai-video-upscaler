# AGENTS.md — AI Agent Guidelines & Architecture Compass

> **Kim chỉ nam dành cho các AI Agent khi đọc, bảo trì và phát triển codebase này.**
> **Quy tắc bắt buộc:** Ngắn gọn, chuẩn xác, không suy diễn; cập nhật file này sau mỗi lần tái cấu trúc hoặc thay đổi kiến trúc code; **BẮT BUỘC commit chi tiết sau MỖI yêu cầu** (nêu rõ nguồn yêu cầu, việc đã làm, lỗi đã fix, số liệu thực nghiệm).
> Chi tiết thực nghiệm & số đo: xem [docs/engineering-log.md](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md) | Phân hệ phụ đề: xem [subtrans/AGENTS.md](file:///Users/xis108/Desktop/upscale/subtrans/AGENTS.md).

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
| `--no-organize-audit` | `False` | Tắt bước gom các file audit (.audit.json, .audit.md, .cast.yaml) vào thư mục con `audit/` |
| `--dedup-policy` | `default` | Chính sách dedup: `default` (768:320:0.33), `sensitive-a` (512:160:0.15), `sensitive-b` (384:128:0.10), `motion` (kèm phase correlation), `off` (tắt) |
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
├── docs/                   # Tài liệu kỹ thuật chi tiết (engineering-log.md)
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

## 6. Bảng Quyết Định Kỹ Thuật Đã Chốt (Decisions Summary)

* **Mô hình Super-Resolution:** Chọn `realesr-animevideov3` (thắng 3/3 cảnh trong True Blind Test trên Liquid Retina XDR; chi tiết: [docs/engineering-log.md#2](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#2)).
* **Engine suy luận chính:** Chọn **PyTorch MPS FP16 Native** (`src/mps_engine.py`, đạt 1.86 fps warm-up, 1.68 fps sustained soak; NCNN Vulkan đóng vai trò fallback; chi tiết: [docs/engineering-log.md#4](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#4)).
* **Ngưỡng Deduplication:** Chọn `hi=768:lo=320:frac=0.33` (giảm 60.86% thời gian GPU; đối soát 16x16/8x8 block đạt 0% lỗi khẩu hình/chớp mắt trên mẫu kiểm tra; chi tiết: [docs/engineering-log.md#3](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#3)).
* **Tái tạo khung hình 4K (Re-expansion):** Chọn **FFmpeg Concat Demuxer** theo thời lượng run đại diện + `fps=24000/1001` (triệt tiêu 100% chi phí đĩa của hardlinks).
* **Đồng nhất xưng hô:** Stage A Cast Sheet (`out/<name>.cast.yaml`) nạp vô điều kiện vào mọi batch; chuỗi Gemini ưu tiên bảo toàn đại từ (chi tiết: [subtrans/AGENTS.md](file:///Users/xis108/Desktop/upscale/subtrans/AGENTS.md)).
* **Căn chỉnh phụ đề tự động:** Thuật toán `--auto-offset` kiểm định 4 tầng (ngưỡng 0.5s [cập nhật từ 1.0s sau đêm 2], số cặp >= 30, quintile drift <= 0.5s, độ khớp tăng >= 15 điểm; chi tiết: [docs/engineering-log.md#6](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#6)).
* **Cổng Backpressure & Chống ENOSPC:** Công thức `need = n_unique × max(recent, p95, 9.0) × 1.2` kèm kiểm tra giữa chunk; cơ chế retry tối đa 2 lần, đánh dấu `failed-resumable`, tự thử lại cuối hàng đợi; thoát mã non-zero và in bảng CHƯA XONG nếu chưa hoàn tất.
* **Chốt chặn phụ đề & Idempotency:** Tuyệt đối không ghi `[Chưa dịch]` vào file SRT (dùng text EN fallback); chỉ nhúng khi bản dịch thật ≥ 95% và uncertain < 10%; cờ `--subs-pending`; bóc tách track cũ trước khi nhúng mới.
* **Xử lý chú thích âm thanh:** Cờ `--sound-cues {drop,translate}` (mặc định `drop` tự động lọc bỏ cue thuần ngoặc đơn không có phụ đề tiếng Anh đối ứng).
* **Khóa GPU độc quyền:** Dùng file khóa `/tmp/upscale_gpu.lock` kiểm soát GPU theo từng chunk infer, cho phép nạp Ollama ở stage phụ đề sau khi hoàn tất upscale GPU.
* **Cổng kiểm định Mux:** `src/mux_audit.py` xác minh toàn vẹn luồng stream copy và khớp số lượng font chữ nhúng trước khi xuất xưởng.
* **Tự động Dọn Dẹp Checkpoint & Gom Gọn Audit:** `src/cleanup.py` kích hoạt tự động sau khi `verify_mux_integrity` trả về PASS; xóa sạch `.temp_<basename>` (kể cả symlinks, thư mục con thực tế và thư mục rỗng) giúp thu hồi ~6.6 GB/tập; tự động gom các file `.audit.json`, `.audit.md`, `.cast.yaml` vào subfolder `audit/` để thư mục xuất xưởng chỉ còn video `.mkv` và `.vi.srt`. Cờ `--keep-temp` cho phép giữ lại khi cần debug.
* **Đa Khóa Gemini & Xoay Vòng Hạn Mức (Multi-Key Rotation):** `subtrans/config.py` và `subtrans/gemini_backend.py` hỗ trợ nạp song song nhiều API Key (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`,...). Khi gặp lỗi 429 (`RESOURCE_EXHAUSTED` / PerDay), hệ thống tự động xoay sang key kế tiếp của cùng model trong batch dịch, nâng hạn ngạch Lite lên 2.000 RPD (500 RPD/key) và Flash lên 80 RPD/model (20 RPD/key) từ 4 project độc lập mà không ngắt quãng pipeline.
* **Đính chính số đo AnimeJaNai & Chốt chặn đầu ra Output Sanity Guard:** Mọi kết quả đo đạc AnimeJaNai trước đây trên MPS (D4 PSNR 74 dB đối chiếu sai giữa FP16 và FP32 đều thiếu residual, D8 fps 1.79, D10b) đều **KHÔNG CÓ GIÁ TRỊ** về độ đúng của đầu ra do lớp `AnimeJaNaiCompact2x` thiếu phép cộng phần dư nearest ×2 dẫn đến video đầu ra bị đen (luma 0.04–0.35). Đã sửa triệt để tại tag `janai-fix` với phép cộng `F.interpolate(x, scale_factor=2.0, mode='nearest')` (khớp bit-exact 0.0 với spandrel CPU, PSNR 75–158 dB). Đồng thời thiết lập chốt chặn bắt buộc `src/output_sanity.py` (kiểm tra sau chunk 1 và mỗi 10 chunk sau: |ΔY| ≤ 6/255, corr ≥ 0.98; nếu sai dừng ngay lập tức, ghi `audit.json: FAIL (OUTPUT_SANITY)`, cấm mux và cấm chuyển model theo D6).
* **Tối ưu hóa hiển thị phụ đề đa thoại & chống trôi chữ (Dual Dialogue & Playback Optimization):** Loại bỏ triệt để đoạn cắt cụt timeline trong `subtrans/parser.py`; bổ sung `optimize_cues_for_playback` trong `subtrans/writer.py`. Tự động phát hiện 2 người nói cùng lúc / gần cùng lúc (start chênh <= 800ms, overlap >= 400ms) để gộp thành phụ đề song song 2 dòng (`- Lời 1\n- Lời 2`) kéo dài tối thiểu 2.2s; tự động gộp các mảnh câu liên tiếp nói quá nhanh (< 1.2s/mảnh) thành một câu liền mạch; bảo đảm thời lượng đọc tối thiểu (>= 1.2s - 1.4s) và khoảng nghỉ 40ms chống giật flicker trên IINA, mpv, VLC và Apple TV.
* **Bóc tách phụ đề đa người nói Closed Captions (Multi-Speaker CC Splitting):** Khắc phục triệt để hiện tượng mất câu thoại của người nói trước trong các event CC dạng gộp `-（Người 1）Thoại 1\N-（Người 2）Thoại 2` (như câu gọi "Thuần phi Kin" của Kenshuu tại 14:18 Tập 07). Triển khai `split_multi_speaker_cc_events` và `is_multi_speaker_or_multi_event_text` trong `subtrans/parser.py` để tách thành các event độc lập có cùng timeline; nâng cấp `strip_speaker_prefix` bóc tiền tố gạch ngang đầu dòng `^[-–—]\s*`. Mỗi câu được dịch riêng lẻ và tự động hợp nhất thành phụ đề song song 2 dòng qua bộ Dual Dialogue tại `subtrans/writer.py`. Đã áp dụng và cập nhật thành công 100% cho series Futsutsuka (Tập 04–11).
* **Khắc phục lệch khung hình BD Remux GOP dài (Dynamic GOP Chunk Alignment & Frame Audit):** Loại bỏ triệt để giả định lưới cố định 192 frames trong `src/dedup.py`. Đưa bước phân đoạn `split_video_segments` (Stream Copy) lên trước trong `src/pipeline.py` và đo chính xác số frame thực tế từng segment bằng `get_segments_frame_counts` theo ranh giới Keyframe GOP nguồn (ví dụ GOP 250 frames trên BD Remux tạo ra `[250, 250, 219]` frames thay vì bị cắt cụt ở 192 frames). Tự động ép giữ frame đầu mỗi segment (`keep`) và xây dựng danh sách `runs` tái tạo khớp bit-exact 100% số frame. Tự động vô hiệu hóa và quét lại các file `plan.json` cũ nếu phát hiện không khớp độ dài phân đoạn. Tích hợp kiểm tra số frame video vào `src/mux_audit.py` (`video_frame_count`), tự động đánh rớt `FAIL` nếu video thành phẩm mất bất kỳ khung hình nào so với nguồn gốc.
* **Hiện tượng đứng hình khi camera zoom/pan chậm & Các chính sách Dedup mới (Camera Motion Freeze & Dedup Policies):** Thực nghiệm MAE trên 48 frames (00:44–00:46) phát hiện 51.1% transitions bị đứng hình do mpdecimate gán nhãn sai các chuyển động vi mô của camera pan/zoom. Đã triển khai hỗ trợ cờ `--dedup-policy {default,sensitive,sensitive-a,sensitive-b,motion,off}` và cấu hình qua `shows.yaml` (`dedup_policy`). Thực nghiệm CPU trên BD Remux và WEB-DL xác định 2 ứng viên xuất sắc: `motion-0.30` (giảm đứng hình zoom 44–46s về 0.0%, 93–99% frame thêm là chuyển động thật) và `sensitive-b` (giảm đứng hình zoom về 0.0%, preflight cực nhanh ~96s/tập, chỉ tăng ~14 phút/tập). Mặc định toàn hệ thống tiếp tục giữ nguyên `default` để bảo đảm chuẩn 2 tập/đêm (chi tiết: [docs/engineering-log.md#9](file:///Users/xis108/Desktop/upscale/docs/engineering-log.md#9)).

---

## 7. Tiêu Chí Nghiệm Thu (Definition of Done)

Một phiên làm việc hoặc mã nguồn được coi là hoàn tất khi thỏa mãn đầy đủ các điều kiện sau:
1. **Kiểm thử tự động:** Chạy `pytest` đạt **PASS 100%** cho toàn bộ các bài test không đánh dấu `@pytest.mark.gpu`.
2. **Kiểm định Mux:** Hàm `verify_mux_integrity` trả về `status: "PASS"` (khớp số kênh audio, số luồng sub, và số lượng font chữ đính kèm).
3. **Bảo toàn khung hình:** `ffprobe -count_frames` trên video 4K thành phẩm khớp chính xác từng khung hình so với video nguồn gốc.
4. **Trạng thái Git & Kỷ luật Commit:** Mỗi yêu cầu hoàn thành đều được commit riêng biệt với thông điệp rõ nguồn, nội dung làm/fix và số liệu; `git status` sạch hoàn toàn, không có file tạm bị bỏ sót; tag phiên bản được cập nhật tương ứng.
