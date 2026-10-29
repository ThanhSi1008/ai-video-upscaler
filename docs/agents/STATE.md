# Trạng Thái Hệ Thống & Ngữ Cảnh Hoạt Động (STATE)

> **Cập nhật ngày:** 2026-10-29  
> **Commit Baseline:** `2cb4514` (`feat(agent): tich hop AST Symbol Map, Symbol Finder va Git Pre-commit hook`)  
> **Tài liệu liên quan:** [START_HERE](START_HERE.md) | [PROJECT_MAP](PROJECT_MAP.md) | [DECISIONS](DECISIONS.md) | [VALIDATION](VALIDATION.md) | [Root AGENTS](../../AGENTS.md) | [Subtrans AGENTS](../../subtrans/AGENTS.md)

---

## 1. Trạng Thái Hiện Tại Của Codebase

- **Bộ kiểm thử tự động:** Đạt **PASS 100%** (232 passed, 10 skipped) trên môi trường Python 3.11 Windows 11.
- **Trạng thái Git:** Sạch sẽ, kỷ luật commit riêng biệt cho từng yêu cầu kỹ thuật.

---

## 2. Danh Mục Các Thành Phần Đã Kiểm Chứng (Verified & Production Ready)

| Phân hệ | Module / File | Trạng thái kiểm chứng | Chi tiết kỹ thuật |
| :--- | :--- | :--- | :--- |
| **PyTorch CUDA FP16** | `src/cuda_engine.py` | PASS | `RealESRAnimeVideoV3`, `AnimeJaNaiCompact2x` (có tag `janai-fix`). Batch size 6–8, ~5 fps thuần, ~10–12 fps hiệu dụng. |
| **Dedup & DIS Flow** | `src/dedup.py` | PASS | Cắt video theo Keyframe GOP, scan `mpdecimate`, phát hiện lia/zoom qua dòng chảy quang học DIS (`--dis-thresh 0.30`), tái tạo frame qua Concat Demuxer. |
| **10-bit NVENC BT.709** | `src/ffmpeg_ops.py` | PASS | `hevc_nvenc` Main 10 (`p010le`, preset `p6`, multipass `qres`), ma trận màu BT.709 hai chiều Full $\leftrightarrow$ Limited. |
| **Bit-exact Mux & Audit** | `src/ffmpeg_ops.py`, `src/mux_audit.py` | PASS | Bắt buộc cờ `-max_interleave_delta 0`, audit MD5 âm thanh, font đính kèm và số lượng frame. |
| **Output Sanity Guard** | `src/output_sanity.py` | PASS | Kiểm tra sau chunk 1 và mỗi 10 chunks ($|\Delta Y| \le 6/255$, tương quan $\ge 0.98$ so với bicubic); chặn đứng màn hình đen. |
| **Gemini 3-Key Subtrans** | `subtrans/gemini_backend.py` | PASS | Luân phiên 3 API key (`GEMINI_API_KEY`, `_2`, `_3`), chống nghẽn RPM. Mặc định `gemini-3.5-flash-lite`, high mode `gemini-3.8-flash`. |
| **Whisper Audio Preprocessing**| `src/subtitles.py` | PASS | Trộn kênh trung tâm 5.1 FC (`pan=mono|c0=0.7*FC...`), lọc tần số giọng nói 80Hz–8000Hz, lọc segment rác (`no_speech_prob > 0.65`, `compression_ratio > 2.2`). |
| **Whisper Anti-Hallucination** | `subtrans/parser.py` | PASS | Regex `WHISPER_HALLUCINATION_REGEX` & `SCREAM_OR_REPEAT_LOOP_REGEX`, drop orphan Whisper cues (>2500ms từ EN gần nhất), cô lập Pass 3 trên `active_cues`. |
| **Multi-Speaker & Dual Dialogue**| `subtrans/parser.py`, `subtrans/writer.py` | PASS | Bóc tách người nói vào `Cue.speaker`, bảo toàn timeline Dual Dialogue 2 dòng (`- Lời 1\n- Lời 2`). |
| **Agent AST Code Intelligence**| `scripts/agent_context.py` | PASS | Trích xuất AST classes/functions (`--symbols`), định vị symbol (`--find`), kiểm định ngân sách byte (`--check`), Git pre-commit hook tự động (`--install-hooks`). |
| **Anti-Sleep & Quota Guard** | `src/preflight.py`, `src/scheduler.py` | PASS | Win32 API native `SetThreadExecutionState`, kiểm tra dung lượng ổ đĩa và lập lịch hàng đợi qua đêm. |

---

## 3. Các Thành Phần Thử Nghiệm / Cần Lưu Ý (Unverified / Experimental)

1. **TensorRT Zero-Copy Pipeline (`src/trt_zero_copy_upscaler.py`, `src/build_trt_engine.py`):**
   - Đang ở mức độ thử nghiệm kiến trúc đường ống kích thước cố định, chưa được tích hợp vào CLI chính `upscale.py`. Không tự ý kích hoạt cho các tác vụ sản xuất.
2. **RAMDisk Context Manager (`src/ramdisk.py`):**
   - Hiện sử dụng `tempfile.TemporaryDirectory` trên SSD/NVMe thông thường với dọn dẹp cuốn chiếu, chưa mount đĩa RAM ảo vật lý trên Windows.
3. **Bảo vệ Gemini Quota ban ngày:**
   - Tuyệt đối không chạy lệnh gọi API Gemini thật trong khoảng từ **14:00 đến đêm chạy**. Mọi kiểm thử chức năng bắt buộc dùng cờ `--dry-run` hoặc mock backend.
4. **Không chạy kiểm thử GPU khi pipeline đang chạy:**
   - Hệ thống chỉ hỗ trợ 1 tiến trình chiếm dụng GPU duy nhất tại một thời điểm.
