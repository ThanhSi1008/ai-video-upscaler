# AI Video Upscaler 4K & AI Subtitle Translator (Windows 11 & NVIDIA RTX)

Công cụ tối ưu hóa chuyên sâu để **Upscale Video Anime lên 4K (2x Super-Resolution)** và **Dịch phụ đề Anime bằng AI (Subtrans V2)** chạy 100% cục bộ trên Terminal / PowerShell và **Giao diện Web thời gian thực (`web_gui.bat`)** trên Windows 11.

Hệ thống được thiết kế và tối ưu hóa kịch trần cho cấu hình phần cứng mục tiêu:
* **Hệ điều hành:** Windows 11 64-bit (`HAGS = 2`, Power Plan `Ultimate Performance`, PCIe ASPM Off).
* **CPU:** Intel Core i5-12600K (10 nhân / 16 luồng; tự động ghim `HighQoS` giữ luồng điều phối trên P-Cores Golden Cove + tăng tốc AVX2 cho OpenCV C++ `cv2.imwrite` & SIMD).
* **GPU:** NVIDIA GeForce RTX 3060 (12GB GDDR6 VRAM @ P0 `7500 MHz`, Ampere Tensor Cores `HMMA.16816.F16`, NVIDIA TensorRT 10.16 FP16 & Triton JIT).
* **Mã hóa phần cứng:** NVIDIA NVENC HEVC Main 10 (`hevc_nvenc`, Preset `p6`, `CQ 18`, `maxrate 45M`, Multipass `fullres`, Lookahead `32`, Spatial AQ `strength 6` + Temporal AQ, `bf 4`, `b_ref_mode middle`, `p010le`, BT.709).
* **Chống sleep máy:** Win32 API native `SetThreadExecutionState` (`ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED`).

---

## ⚡ 1. Công Cụ Upscale Video Anime 4K (`upscale.py` / `upscale.bat` / `upscale_turbo.bat`)

Khai thác trực tiếp nhân NVIDIA Tensor Cores (TensorRT 10 FP16 / PyTorch CUDA FP16) kết hợp bộ mã hóa phần cứng ASIC NVENC để đạt chất lượng chuẩn tham chiếu phòng thu với thông lượng tối đa:

* **Mô hình AI:** `realesr-animevideov3` (`SRVGGNetCompact` 2x chuyên dụng cho anime, khử nhiễu nén, giữ nét vẽ sắc bén không bệt sáp) hoặc `animejanai-compact-2x`, cấu hình linh hoạt theo từng series trong `shows.yaml` và `models/models.yaml`.
* **TensorRT 10.16 FP16 & Zero-Disk Turbo Pipe (`--zero-disk`):**
  * Tự động xuất ONNX và biên dịch engine TensorRT 10 FP16 (`builder_optimization_level=5`, hợp nhất `Conv3x3 + PReLU` trong SRAM, vòng đệm VRAM `_out_ring` 2-slot không cấp phát động).
  * Đường ống **Zero-Disk (`upscale_turbo.bat`)** giải mã `NVDEC raw RGB24 pipe` $\rightarrow$ `TensorRT 10 FP16` (cắt lát vi mô 2-frame hạ 75% đỉnh VRAM 8K) $\rightarrow$ kernel **Triton JIT `p010le`** trên GPU $\rightarrow$ `hevc_nvenc`, **triệt tiêu 100% file PNG trung gian trên SSD**.
* **Gối đầu 3 tầng thực sự (True 3-Stage Overlap):** Giải mã chunk `N+1` trên CPU/NVDEC và xả mã hóa `NVENC` của chunk `N-1` chạy song song hoàn toàn với suy luận GPU của chunk `N`.
* **Khử trùng lặp kép & Cyclic Cel Memoization (`A->B->A`):**
  * Quét 1-Pass hợp nhất `mpdecimate + gray` kết hợp dòng chảy quang học `DISOpticalFlow` thoát sớm bắt trọn cảnh camera pan/zoom chậm, lưu chuỗi `raw_actions_rle` tái phân hoạch GOP trong `< 1 ms`.
  * Phát hiện chu kỳ hoạt họa lặp lại (`A->B->A` mấp máy môi, chớp mắt, tóc bay) bằng `BLAKE2b` + kiểm chứng SIMD pixel-exact, tái sử dụng kết quả qua giải thuật tối ưu **Belady's `OPT`** (Zero-Disk) hoặc **NTFS Hardlink `os.link`** 0ms/0-byte (Chuẩn).
* **1-Pass Concat-Mux & Bảo toàn Bit-Exact dữ liệu gốc:** Ghép trực tiếp danh sách phân đoạn `up_*.mp4` cùng toàn bộ Audio đa kênh (FLAC/AAC/TrueHD), Phụ đề mềm (ASS/SRT), Font chữ nhúng (TTF/OTF) và Chapters từ MKV gốc trong **1 lượt duy nhất** (`mux_segments_directly` với `-max_interleave_delta 0`, xóa bỏ ~8 GB ghi tạm `merged_4k.mp4`).
* **Output Sanity Guard & Parallel `streamhash` Mux Audit:** Kiểm tra độ lệch sáng $|\Delta Y| \le 6/255$ và tương quan Pearson $r \ge 0.98$ ngay trên ảnh thu nhỏ GPU (0 suy luận lặp) sau chunk 0 và mỗi 10 chunks; đối soát bit-exact frame count, MD5 đa luồng âm thanh 1-Pass (`-f streamhash`) và đan xen packet song song trước khi dọn thư mục tạm.
* **Checkpoint Resume & Độc quyền GPU:** Tự động tiếp tục từ phân đoạn `up_*.mp4` đang dở nếu dừng giữa chừng (`Ctrl+C`), bảo vệ độc quyền VRAM bằng khóa `.gpu_active.lock` có kiểm chứng PID sống.

---

## 💻 2. Yêu Cầu Hệ Thống & Cài Đặt

### Cấu Hình Phần Cứng Windows (NVIDIA RTX)
* **Hệ điều hành:** Windows 10/11 64-bit.
* **GPU:** NVIDIA GeForce RTX 3060 (12GB VRAM) hoặc dòng RTX 20/30/40/50 series với Driver NVIDIA mới nhất.
* **CPU & RAM:** Intel Core i5/i7/i9 (khuyến nghị $\ge 16\text{ GB}$ RAM, tối ưu AVX2).
* **Công cụ bổ trợ:** FFmpeg & FFprobe hỗ trợ `hevc_nvenc` (khuyến nghị bản Gyan.dev Full Build, đã thêm vào `PATH`).

**Cài đặt trên Windows (PowerShell):**
```powershell
# 1. Clone repo và tạo virtual environment
git clone https://github.com/ThanhSi1008/ai-video-upscaler.git
cd ai-video-upscaler
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt PyTorch với hỗ trợ CUDA 12.4+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Cài đặt các thư viện phụ thuộc (bao gồm TensorRT 10, Triton Windows, Google GenAI, Faster-Whisper)
pip install -r requirements.txt

# 4. (Khuyến nghị chạy 1 lần với quyền Admin) Mở khóa HAGS=2 & xung VRAM CUDA P0 (7500 MHz):
.\unlock_rtx3060_admin.bat

# 5. Kiểm tra sức khỏe bộ nhớ dự án & chạy bộ kiểm thử an toàn cô lập offline:
.\.venv\Scripts\python.exe scripts/agent_context.py --check
.\.venv\Scripts\python.exe scripts/verify_offline.py
```

---

## 🚀 3. Hướng Dẫn Sử Dụng CLI (`upscale.py` / `upscale.bat` / `upscale_turbo.bat`)

### Lệnh Vận Hành Trên Windows (PowerShell / CMD):
```powershell
# 1. Xem trước nhanh 10 giây (tự động cắt tại 30% phim, upscale và mở bằng trình phát mặc định):
python upscale.py samples/sample_sword_3min.mkv --preview 10

# 2. Nâng cấp 1 tập video bằng launcher Chuẩn hoặc Turbo Zero-Disk (TensorRT 10 + 0 PNG đĩa):
.\upscale.bat samples/sample_sword_3min.mkv
.\upscale_turbo.bat samples/sample_sword_3min.mkv

# 3. Chạy đầy đủ 1 tập video qua CLI Python (mặc định Turbo Zero-Disk CQ 18 fullres trên hevc_nvenc):
python upscale.py samples/sample_sword_3min.mkv -o out/sword_4K.mkv

# 4. Chạy hàng đợi nhiều tập qua đêm (preflight nền tách sẵn GOP, auto-offset và deadline an toàn):
python upscale.py --queue ep1.mkv ep2.mkv -o out/ --deadline 07:00 --allow-partial

# 5. Mở giao diện Web Studio thời gian thực trên trình duyệt (http://127.0.0.1:8765):
.\web_gui.bat
python web_gui.py --port 8765
```

### Bảng Tham Số Tùy Chọn Chính:
* `-o`, `--output`: Đường dẫn file hoặc thư mục xuất (Mặc định: `out/<tên>_4K.mkv`).
* `-m`, `--model`: Mô hình AI (Mặc định: tự động chọn qua `shows.yaml` hoặc mặc định `animevideov3`).
* `--engine`: Động cơ suy luận (`cuda` cho NVIDIA GPU hoặc `cpu`; mặc định: `cuda`).
* `--precision`: Độ chính xác suy luận (`fp16` hoặc `fp32`, mặc định: `fp16`).
* `--batch-size`: Kích thước lô xử lý khung hình trên GPU (Mặc định: `8` trên RTX 3060 12GB).
* `--zero-disk`: Kích hoạt pipeline Zero-Disk (Mặc định: Bật; `NVDEC RGB24 pipe -> TensorRT/CUDA FP16 -> Triton p010le pipe -> NVENC`, 0 file PNG trung gian).
* `--no-zero-disk`: Tắt chế độ Zero-Disk và dùng pipeline truyền thống lưu file PNG trung gian trên đĩa (alias `--disk`).
* `--no-tensorrt`: Tắt tự động biên dịch/chạy TensorRT 10 FP16 và dùng thuần PyTorch CUDA FP16.
* `--dedup-policy`: Chính sách khử trùng lặp (`default`, `sensitive-a`, `sensitive-b`, `motion`, `drift`, `dis`, `off`; mặc định: `default` hoặc theo `shows.yaml`).
* `--dis-thresh`: Ngưỡng dòng chảy quang học $\tau$ tính bằng pixel 1080p cho chính sách `dis` (Mặc định: `0.30`).
* `--drift-delta`: Ngưỡng dịch tích lũy $\delta$ tính bằng pixel 1080p cho chính sách `drift` (Mặc định: `0.75`).
* `--no-dedup`: Tắt bộ lọc khử trùng lặp, suy luận 100% khung hình.
* `--list-models`: Liệt kê các mô hình đã đăng ký trong `models/models.yaml`.
* `--force`: Bắt buộc chạy và ghi đè/xóa phân đoạn cũ nếu phát hiện xung đột mô hình.
* `--queue`: Danh sách nhiều tập video cần chạy tuần tự trong ca qua đêm.
* `--deadline HH:MM`: Mốc hạn chót dừng hàng đợi an toàn nếu tập kế tiếp dự kiến không kịp.
* `--allow-partial`: Cho phép tập không kịp deadline chạy đến hạn chót, dừng sạch ở ranh giới chunk và bảo toàn segment `up_*.mp4` để chạy tiếp lần sau.
* `--subs-after-upscale` / `--no-subs`: Tự động chạy dịch phụ đề sau khi hàng đợi upscale hoàn tất (Mặc định: Bật).
* `--async-subs`: Chạy bất đồng bộ tiến trình dịch phụ đề Cloud Gemini của tập `N` song song khi GPU đang upscale tập `N+1`.
* `--subs-only [VIDEO]`: Chỉ trích xuất và dịch phụ đề trong tiến trình riêng, bỏ qua giai đoạn upscale video.
* `--subs-pending`: Chạy lại quy trình dịch phụ đề cho các tập ghi nhận trong `pending_subs.json`.
* `--subs-upgrade`: Nâng cấp các câu thoại trước đó lên model chất lượng cao hơn khi còn hạn mức.
* `--subs-quality {normal,fast,high}`: Chất lượng dịch phụ đề (`normal`/`fast`: ưu tiên `gemini-3.5-flash-lite` tiết kiệm quota; `high`: ưu tiên `gemini-3.8-flash`; mặc định: `normal`).
* `--prefer-whisper`: Khi MKV thiếu phụ đề JA (chỉ có audio JA + sub EN), chạy `faster-whisper` STT + DTW align EN để giữ đầy đủ đại từ/kính ngữ tiếng Nhật.
* `--auto-align-en` / `--no-auto-align-en`: Tự động ghim mốc thời gian Whisper/JA theo track tiếng Anh phòng thu (Mặc định: Bật).
* `--auto-offset` / `--no-auto-offset`: Tự động nhận diện và bù độ lệch thời gian giữa track JA và EN (Mặc định: Bật trong `--queue`).
* `--sub-offset-ms [ms]`: Bù trừ độ lệch thời gian phụ đề thủ công theo mili-giây (luôn ưu tiên cao nhất).
* `--no-embed`: Không nhúng phụ đề vào container MKV 4K, chỉ xuất và giữ file `.vi.srt` rời.
* `--target-4k`: Chỉ định container MKV 4K đích cụ thể để nhúng phụ đề trong chế độ `--subs-only`.
* `-b`, `--bitrate`: Bitrate dự phòng cho encoder không phải NVENC (trên `hevc_nvenc`, hệ thống mặc định dùng `CQ 18` VBR với `maxrate 45M`).
* `-s`, `--segment`: Độ dài phân đoạn cuốn chiếu tính bằng giây (Mặc định: `8` giây).
* `-f`, `--format`: Định dạng khung hình trung gian khi chạy chế độ đệm đĩa `--no-zero-disk` (`png` lossless BT.709 mặc định hoặc `jpg`).
* `--keep-temp`: Giữ lại thư mục tạm `.temp_<basename>` sau khi Mux Audit PASS (Mặc định: tự động xóa sạch).
* `--clean-temp`, `--cleanup`: Quét dọn toàn bộ thư mục tạm `.temp_*`, symlinks mồ côi và gom báo cáo vào `audit/` rồi thoát.
* `--no-ramdisk`: Buộc dùng thư mục tạm trên ổ cứng thay vì RAM Disk / rolling buffer.
* `--no-resume`: Buộc chạy lại từ đầu thay vì tiếp tục phân đoạn cũ.
* `--no-anti-sleep`: Không kích hoạt cơ chế chống sleep máy (`SetThreadExecutionState`).
* `--dry-run`: Chạy mô phỏng toàn bộ quy trình lập lịch hoặc dịch phụ đề mà không gọi GPU hay API thật.
* `--benchmark-mode`: Chế độ đo đạc thuần túy (tắt preflight nền để tránh nhiễu tài nguyên).
* `--preview [giây]`: Chế độ xem trước `N` giây (mặc định `10s`) ở 30% phim và mở trình phát ngay.

---

## 📊 4. Số Liệu Thực Nghiệm Lịch Sử & Kiến Trúc Hiện Hành (NVIDIA RTX 3060 12GB)

> **Ghi chú minh bạch số liệu:** Hai cột đầu lưu giữ số đo thực nghiệm lịch sử từ các phiên chạy chuẩn PyTorch PNG (`docs/engineering-log.md` Mục 1–11). Cột thứ ba mô tả cấu hình sản xuất hiện hành sau khi mở khóa TensorRT 10.16 FP16, Zero-Disk Pipe và 6 thuật toán tối ưu kịch trần (`docs/engineering-log.md` Mục 12).

| Thông số kỹ thuật | Mốc cơ sở lịch sử (`batch_size=1`) | Mốc PyTorch PNG lịch sử (`batch_size=6`) | Kiến trúc hiện hành (`TensorRT 10 + Zero-Disk`, `batch_size=8`) |
| :--- | :--- | :--- | :--- |
| **Phần cứng & Xung VRAM** | Core i5-12600K, RTX 3060 12GB (P2) | Core i5-12600K, RTX 3060 12GB (P2) | Core i5-12600K (`HighQoS` P-Core), RTX 3060 12GB (`HAGS=2`, P0 `7500 MHz`) |
| **Engine suy luận** | PyTorch CUDA FP16 (`batch_size=1`) | PyTorch CUDA FP16 (`batch_size=6`) | **TensorRT 10.16 FP16** (`opt=5`, `_out_ring` 2-slot) / CUDA FP16 (`batch_size=8`) |
| **Thông lượng đã đo / Cơ chế** | **1.37 fps** thuần GPU | **4.8 – 5.2 fps** thuần GPU \| **~10 – 12+ fps** hiệu dụng (Dedup) | Khử ~58 lượt đọc/ghi GDDR6 + Cyclic Cel (`A->B->A`) + 3-Stage Overlap |
| **Đường truyền khung hình** | PIL `Image.save` (~45 ms/frame) | OpenCV C++ `cv2.imwrite` AVX2 (~14 ms/frame) | **Zero-Disk Pipe** (`NVDEC RGB24` $\rightarrow$ **Triton JIT `p010le`**, **0 PNG SSD**) hoặc Hardlink `os.link` |
| **Encoder phần cứng** | `hevc_nvenc` Main 10 (`p6`, `CQ 19`) | `hevc_nvenc` Main 10 (`p6`, `qres`, Lookahead `24`) | `hevc_nvenc` Main 10 (`p6`, **`CQ 18`**, `maxrate 45M`, **`fullres`**, Lookahead **`32`**, Dual AQ) |
| **Đóng gói MKV & Audit** | Mux 2 bước (`merged_4k.mp4` + MKV) | Mux 2 bước + kiểm tra tuần tự | **1-Pass `mux_segments_directly`** (khử ~8 GB ghi tạm) + **1-Pass `streamhash` MD5** song song |
| **Output Sanity Guard** | $\|\Delta Y\| \approx 2.80 / 255$, $r \approx 0.9998$ (PASS) | $\|\Delta Y\| \approx 2.80 / 255$, $r \approx 0.9998$ (PASS) | `avg_pool2d` trực tiếp trên GPU Batch 0 (0 suy luận lặp, $\|\Delta Y\| \le 6/255$, $r \ge 0.98$) |

---

## 🌐 5. Dịch Phụ Đề Anime Bằng AI — Subtrans V2 (`subtrans/`)

Phân hệ dịch phụ đề anime chuyên sâu bằng mô hình ngôn ngữ lớn trên đám mây (Google GenAI Gemini SDK):

```powershell
# Chạy trên Windows (sử dụng launcher tiện ích UTF-8):
.\subtrans.bat check                   # Kiểm tra sức khỏe kết nối & các Gemini API keys
.\subtrans.bat models                  # Liệt kê danh sách Gemini models & hạn mức
.\subtrans.bat eval                    # Chạy bộ kiểm định hồi quy (6 ca) & đại từ (22 cues) offline
.\subtrans.bat run samples/sample_sword_3min.mkv -o out/
```

* **100% Cloud Gemini API & Xoay vòng 3-Key thông minh:** Chạy bằng `gemini-3.5-flash-lite` (Draft) và `gemini-3.8-flash` (Cascade Repair) với xoay vòng 3 API keys modulo + chọn key ít tải nhất + tự động đổi key khi gặp `429 PerMinute` và theo dõi cạn RPD riêng từng `(model, key_hash)`.
* **Implicit Context Caching & Contextual Cache `v2.2`:** Ghim tiền tố hệ thống tĩnh `> 4.096` tokens để kích hoạt bộ nhớ đệm ngữ cảnh của Gemini (`cached_tokens`), kết hợp SQLite Contextual Cache (`compute_contextual_cue_sig`, tiêm ngữ cảnh tại chỗ khi partial hit và ghi ngược kết quả Cascade Repair).
* **Đồng nhất xưng hô đa tầng (`CastSheet` + `HysteresisCastManager`):** Khai thác bảng nhân vật Stage A (`.cast.yaml`), kết hợp bộ gắn nhãn hình thái tiếng Nhật (`ja_morph_tagger.py` nhận diện Kansai-ben, `soregashi`, đại từ ngôi 1/2) và trường suy luận Chain-of-Thought `p1`/`p2` tường minh trước khi sinh câu tiếng Việt.
* **Smart Cloud Cascade, Linter V2 & Hybrid BM25 RAG:** Linter V2 kiểm định CPS, rò rỉ CJK, nhất quán thuật ngữ Glossary (`RULE_GLOSSARY_TERM`), nhất quán đại từ (`RULE_PRONOUN_CONSISTENCY`) và biên từ Katakana; tự động lọc 5–15% câu khó sang `gemini-3.8-flash` biên tập lại và tuyển chọn mẫu `silver` vào kho tri thức Hybrid BM25 RAG (`.cache/knowledge.db`).
* **Faster-Whisper 5.1 FC + Silero VAD & Monotonic DTW Auto-Align:** Khi video thiếu phụ đề tiếng Nhật, tự động trích xuất kênh thoại trung tâm 5.1 FC + lọc thông dải, chạy `faster-whisper` với `initial_prompt` từ Glossary, sau đó căn chỉnh mốc thời gian theo track tiếng Anh phòng thu bằng quy hoạch động Monotonic DTW (hỗ trợ tách `1:2` theo khoảng lặng `>= 250ms` và bù lệch `Piecewise 2-Segment Auto-Offset`).

---

## 📁 6. Cấu Trúc Thư Mục Chuẩn Hóa

```text
ai-video-upscaler/
├── upscale.py                  # CLI Entrypoint cho AI Video Upscaler 4K
├── upscale.bat                 # Windows launcher chế độ Chuẩn (.venv auto-detect)
├── upscale_turbo.bat           # Windows launcher chế độ Turbo Zero-Disk (TensorRT 10 + 0 PNG)
├── unlock_rtx3060_admin.bat    # Tiện ích Admin mở khóa HAGS=2, PCIe ASPM Off & CUDA P0 (7500 MHz)
├── subtrans.py                 # CLI Entrypoint cho Anime Subtitle Translator (Subtrans V2)
├── subtrans.bat                # Windows launcher cho Subtrans (UTF-8, multi-key)
├── web_gui.py                  # Entrypoint khởi chạy Local Real-Time Web Studio (FastAPI + Uvicorn)
├── web_gui.bat                 # Windows launcher mở nhanh Web GUI trên trình duyệt mặc định
├── web/                        # Backend SSE 1Hz, duyệt ổ đĩa cục bộ, đối chiếu 1080p/4K & giao diện web/static/
├── shows.yaml                  # Ánh xạ model SR & chính sách dedup riêng theo từng bộ anime
├── pytest.ini                  # Cấu hình Pytest (tests/ & subtrans/tests/)
├── ruff.toml                   # Cấu hình trình kiểm tra mã nguồn tĩnh Ruff
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
├── README.md                   # Tài liệu hướng dẫn sử dụng tổng quan
├── AGENTS.md                   # Kim chỉ nam kiến trúc & bất biến kỹ thuật cho AI Agent
├── .agents/                    # Kỹ năng chuyên biệt (skills/), guard hooks & MCP server cục bộ
├── src/                        # Core Engine: AI Video Upscaling (TensorRT 10 / CUDA FP16, NVENC 10-bit)
│   ├── config.py               # Hằng số runtime, CUDA Fast Math, NVENC CQ 18 fullres & PID GPU Lock
│   ├── pipeline.py             # Điều phối 3-Stage Overlap (Chuẩn & Zero-Disk), Resume & 1-Pass Mux
│   ├── build_trt_engine.py     # Xuất ONNX & biên dịch TensorRT 10.16 FP16 (_out_ring 2-slot 0-alloc)
│   ├── trt_zero_copy_upscaler.py # Zero-Disk Pipe: NVDEC RGB24 -> TRT/CUDA -> Triton JIT p010le -> NVENC
│   ├── cuda_engine.py          # PyTorch CUDA FP16 Engine, 8K 2-frame slicing, Cyclic Hardlink Worker
│   ├── ai_engine.py            # NCNN Vulkan runner dự phòng khi không dùng CUDA
│   ├── models_resolver.py      # Đọc models/models.yaml, kiểm tra SHA-256 trọng số & khớp shows.yaml
│   ├── ffmpeg_ops.py           # FFmpeg / FFprobe: BT.709 2 chiều, NVENC Main 10, 1-Pass Concat-Mux
│   ├── dedup.py                # 1-Pass mpdecimate+DIS, Cyclic Cel (A->B->A) & raw_actions_rle GOP
│   ├── scheduler.py            # Lập lịch hàng đợi qua đêm, kiểm tra deadline & Queue Preflight nền
│   ├── preflight.py            # Anti-Sleep Win32, Alder Lake P-Core HighQoS, kiểm tra nguồn & đĩa trống
│   ├── output_sanity.py        # Output Sanity Guard (Batch-0 GPU 0-reinference |ΔY| & Pearson corr)
│   ├── mux_audit.py            # Kiểm định Mux đa luồng & băm MD5 đa luồng âm thanh 1-Pass (streamhash)
│   ├── subtitles.py            # Trích xuất phụ đề, Faster-Whisper 5.1 FC + Silero VAD fallback
│   ├── pending_queue.py        # Quản lý hàng đợi phụ đề tồn đọng (stdlib-only, không nạp torch)
│   ├── ramdisk.py              # Tự dò RAM Disk vật lý hoặc định tuyến sang NVMe Gen4 rolling_buffer
│   ├── cleanup.py              # Dọn dẹp thư mục tạm, symlinks mồ côi & gom báo cáo vào audit/
│   ├── night_report.py         # Tổng hợp báo cáo phiên chạy đêm từ log & telemetry CSV
│   └── logger.py               # Telemetry logging & thanh tiến trình Rich
├── subtrans/                   # Core Engine: Anime Subtitle Translation V2 (Google GenAI Gemini)
│   ├── cli.py                  # Giao diện dòng lệnh Typer (run, batch, eval, rag, check, models)
│   ├── pipeline.py             # Pipeline dịch 4 giai đoạn (Implicit Caching, Contextual Cache, Cascade)
│   ├── parser.py               # Đọc ASS/SRT, lọc CC, gắn mode_hint & Piecewise Auto-Offset
│   ├── dtw_auto_align.py       # Quy hoạch động Monotonic DTW ghim mốc thời gian Whisper theo EN
│   ├── cast_sheet.py           # Stage A: Khai thác bảng nhân vật & danh xưng cố định
│   ├── dynamic_cast.py         # HysteresisCastManager theo dõi xưng hô động qua CoT p1/p2
│   ├── audio_diarization.py    # DSP CPU 5.1 FC 8kHz nhận diện đổi lượt nói (speaker_changed) cho .srt trần
│   ├── ja_morph_tagger.py      # Tách từ hình thái học MeCab (fugashi/UniDic), Kansai-ben & kính ngữ
│   ├── gemini_backend.py       # Gemini client xoay vòng 3-key, ThinkingConfig & retry/bisect
│   ├── cascade.py              # Stage P2: Smart Cloud Cascade Repair bằng gemini-3.8-flash
│   ├── rag.py                  # Hybrid BM25 RAG few-shot retrieval & tự động thăng hạng mẫu silver
│   ├── linter.py               # Linter V2 kiểm định CPS, CJK, thuật ngữ Glossary & đại từ
│   ├── writer.py               # Xuất SRT/ASS bảo toàn typesetting, lời gọi lặp & dấu ngoặc hội thoại
│   ├── eval/                   # Bộ kiểm định tất định offline (6 ca hồi quy & 22 cues đại từ)
│   └── tests/                  # Unit tests riêng của phân hệ dịch phụ đề
├── tests/                      # Bộ kiểm thử tự động toàn hệ thống (Pytest Unit & Integration Tests)
├── benchmarks/                 # Kịch bản đo lường hiệu năng GPU, Dedup, DIS & đánh giá đại từ
├── scripts/                    # Tiện ích vận hành (agent_context.py, verify_offline.py, unlock_rtx3060_admin.ps1)
├── glossaries/                 # Từ điển thuật ngữ & quy tắc xưng hô riêng theo từng series anime
├── models/                     # Danh mục đăng ký mô hình Super-Resolution (models/models.yaml)
├── models_bench/               # Trọng số mô hình AI PyTorch (.pth) & cache TensorRT (.engine)
├── docs/                       # Tài liệu kiến trúc chuyên sâu, ADR (docs/agents/) & engineering logs
├── samples/                    # Video mẫu 3 phút phục vụ kiểm thử nhanh
├── logs/                       # Nhật ký telemetry CSV và báo cáo phiên chạy đêm
└── out/                        # Thư mục xuất video 4K, phụ đề hoàn thiện và báo cáo kiểm định (audit/)
```
