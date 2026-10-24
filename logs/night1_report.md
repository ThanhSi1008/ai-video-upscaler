# BÁO CÁO GIÁM SÁT CA CHẠY QUA ĐÊM (NIGHT 1)

> [!NOTE] Lưu trữ Lịch sử (Historical Archive - Initial macOS Prototype Run)
> Báo cáo này ghi lại kết quả thử nghiệm ban đầu trên thiết bị macOS cũ trước khi chuyển giao.
> Toàn bộ dự án hiện tại đã chính thức chuyển dịch 100% sang nền tảng **Windows 11 (64-bit)** với cấu hình phần cứng:
> **Intel Core i5-12600K (10 nhân / 16 luồng, AVX2)** | **NVIDIA GeForce RTX 3060 12GB (Ampere GA106, Tensor Cores FP16 & TensorRT)** | **HEVC NVENC 10-bit hardware encoder**.


* **Thời điểm bắt đầu ca:** 2026-10-05 00:25:59 +07:00
* **Commit hash:** `82ea864a56ca0afa09740e143efeab56c5cfb050` (`git status` sạch hoàn toàn)
* **Nguồn điện:** AC Power (`pmset -g batt`: AC attached, 80%)
* **Môi trường tiến trình trước chạy:**
  * `pgrep -fl "upscale.py|realesrgan|ffmpeg"`: Rỗng
  * `ollama ps`: Rỗng (0 models loaded in Unified Memory)
  * `ollama serve`: Daemon Homebrew chạy nền ở trạng thái nghỉ (0.0% CPU)
* **Dung lượng đĩa khả dụng:** 36 GiB trên `/System/Volumes/Data` (ngưỡng tối thiểu yêu cầu ≥ 25 GiB: ĐẠT)
* **Trạng thái RAM Disk:** Sẵn sàng cấp phát mới (đã mount thành công 4.0 GB HFS+ RAM Disk ảo)
* **Kiểm tra file nguồn:** 3/3 file nguồn tồn tại đầy đủ trong `/Users/xis108/Documents/RawSources/`
* **Mốc đo lường hệ thống ban đầu (Baseline - 00:25:26):**
  * `iostat -Id disk0`: **2.504.141,15 MB** đã ghi từ lúc boot
  * `sysctl vm.swapusage`: `total = 3072.00M  used = 1463.38M  free = 1608.62M`
  * `vm_stat` Pageouts: **113.487**
  * Dung lượng trống thư mục đích: **36 GiB**

---

## 1. Khởi Chạy & Xác Thực B2 (00:26 – 00:29)

* **Lệnh khởi chạy hàng đợi:**
  ```bash
  nohup ./upscale.sh --queue \
    "/Users/xis108/Documents/RawSources/I.Want.to.Love.You.Till.Your.Dying.Day.S01E13.I.Miss.You.REPACK.1080p.CR.WEB-DL.JPN.AAC2.0.H.264.MSubs-ToonsHub.mkv" \
    "/Users/xis108/Documents/RawSources/[SubsPlease] Re Zero kara Hajimeru Isekai Seikatsu - 84 (1080p) [5188BBC4].mkv" \
    "/Users/xis108/Documents/RawSources/[SubsPlease] Tensei Shitara Slime Datta Ken S4 - 24 (1080p) [AA94BC1F].mkv" \
    -o /Users/xis108/Documents/Upscaled \
    --deadline 08:00 \
    --allow-partial \
    > logs/night1_stdout.log 2>&1 &
  ```
* **PID tiến trình chính:** `51394` (Upscale Queue Runner)
* **PID tiến trình giám sát:** `51395` (`scripts/soak_monitor.sh` -> file CSV: `logs/soak_telemetry_night1.csv`, chu kỳ 60s)
* **(a) Xác nhận khẳng định `caffeinate` (`pmset -g assertions`):** ĐẠT
  * PID 51400 (bọc từ `upscale.sh` `-ims`): `PreventUserIdleSystemSleep`, `PreventSystemSleep`, `PreventDiskIdle`
  * PID 51450 (bọc từ `src/preflight.py` `CaffeinateContext`): `PreventUserIdleSystemSleep`, `PreventUserIdleDisplaySleep`, `PreventSystemSleep`, `PreventDiskIdle`
  * Cả hai tầng bảo vệ chống sleep máy và màn hình đều đang hoạt động tích cực.
* **(b) Xác nhận mốc hạn chót (`--deadline`):** ĐẠT
  * Mốc hạn chót in ra: `08:00 (2026-10-05 08:00:00)` — **Đúng ngày hôm nay (2026-10-05)**, không bị nhảy sang ngày kế tiếp.
* **Bảng quyết định lập lịch ban đầu (00:25:59):**
```text
==============================================================================================================================
📊 BẢNG QUYẾT ĐỊNH LẬP LỊCH HÀNG ĐỢI QUA ĐÊM (OVERNIGHT BATCH SCHEDULER)
==============================================================================================================================
• Thời điểm bắt đầu    : 2026-10-05 00:25:59
• Mốc hạn chót          : 08:00 (2026-10-05 08:00:00)
• Quỹ thời gian có sẵn : 7:34:00
• Tốc độ AI cơ sở       : Trung vị 1.66 fps | P5 1.62 fps (đo thực tế trong pipeline M3 Pro C1)
• Overhead mỗi tập      : 660s (~11.0 phút: 360s video + 300s phụ đề)
• Biên an toàn ETA      : +7.0% trên ETA P5 để ra quyết định hạn chót
• Cho phép dở dang      : --allow-partial = BẬT
------------------------------------------------------------------------------------------------------------------------------
STT  | Tập Video                          | Tổng Frames | n_unique  | Nguồn n_unique   | Dedup % | ETA (Med) | ETA P5 (+7%) | Kết Thúc | Quyết Định
------------------------------------------------------------------------------------------------------------------------------
[01] | I.Want.to.Love.You.Till.Your....   | 34046       | 15669     | đo (plan.json)   | 54.0  % | 2:48:19   | 3:04:15      | 03:14:19 | ✅ KỊP
[02] | [SubsPlease] Re Zero kara Haj...   | 34276       | 13415     | ước lượng        | 60.9  % | 2:25:41   | 2:39:27      | 05:40:00 | ✅ KỊP
[03] | [SubsPlease] Tensei Shitara S...   | 34526       | 13513     | ước lượng        | 60.9  % | 2:26:40   | 2:40:31      | 08:06:40 | ⏸️ CHẠY 1 PHẦN (HẠN CHÓT)
------------------------------------------------------------------------------------------------------------------------------
🎯 TỔNG KẾT: Duyệt thực thi 3/3 tập (gồm 1 tập chạy 1 phần) | Tổng thời lượng chạy dự kiến: 6:54:18.
==============================================================================================================================
```
* **Bảng quyết định cập nhật sau khi Preflight nền quét xong Tập 2 (00:27:39):**
  * `[Background Preflight]` quét xong `[SubsPlease] Re Zero - 84`: **`n_unique` đo được = 20.876 frames** (Tỷ lệ Dedup thật: **38.7%** so với 60.9% ước lượng).
  * Bộ lập lịch tự động tái tính toán và cập nhật lại ETA bảo thủ cho Tập 2 lên 4h 01m 35s:
```text
==============================================================================================================================
📊 BẢNG QUYẾT ĐỊNH LẬP LỊCH HÀNG ĐỢI QUA ĐÊM (OVERNIGHT BATCH SCHEDULER)
==============================================================================================================================
• Thời điểm bắt đầu    : 2026-10-05 00:27:39
• Mốc hạn chót          : 08:00 (2026-10-05 08:00:00)
• Quỹ thời gian có sẵn : 7:32:20
• Tốc độ AI cơ sở       : Trung vị 1.66 fps | P5 1.62 fps (đo thực tế trong pipeline M3 Pro C1)
• Overhead mỗi tập      : 660s (~11.0 phút: 360s video + 300s phụ đề)
• Biên an toàn ETA      : +7.0% trên ETA P5 để ra quyết định hạn chót
• Cho phép dở dang      : --allow-partial = BẬT
------------------------------------------------------------------------------------------------------------------------------
STT  | Tập Video                          | Tổng Frames | n_unique  | Nguồn n_unique   | Dedup % | ETA (Med) | ETA P5 (+7%) | Kết Thúc | Quyết Định
------------------------------------------------------------------------------------------------------------------------------
[01] | I.Want.to.Love.You.Till.Your....   | 34046       | 15669     | đo (plan.json)   | 54.0  % | 2:48:19   | 3:04:15      | 03:15:58 | ✅ KỊP
[02] | [SubsPlease] Re Zero kara Haj...   | 34046       | 20876     | đo (plan.json)   | 38.7  % | 3:40:36   | 4:01:35      | 06:56:34 | ✅ KỊP
[03] | [SubsPlease] Tensei Shitara S...   | 34526       | 13513     | ước lượng        | 60.9  % | 2:26:40   | 2:40:31      | 09:23:15 | ⏸️ CHẠY 1 PHẦN (HẠN CHÓT)
------------------------------------------------------------------------------------------------------------------------------
🎯 TỔNG KẾT: Duyệt thực thi 3/3 tập (gồm 1 tập chạy 1 phần) | Tổng thời lượng chạy dự kiến: 6:45:25.
==============================================================================================================================
```

---

* **Bảng quyết định cập nhật sau khi Preflight nền quét xong Tập 3 (00:29:36):**
  * `[Background Preflight]` quét xong `[SubsPlease] Slime S4 - 24`: **`n_unique` đo được = 18.772 frames** (Tỷ lệ Dedup thật: **45.6%** so với 60.9% ước lượng).
  * Toàn bộ 3/3 tập đã có số đo chính xác từ `plan.json` (không còn số ước lượng):
```text
==============================================================================================================================
📊 BẢNG QUYẾT ĐỊNH LẬP LỊCH HÀNG ĐỢI QUA ĐÊM (OVERNIGHT BATCH SCHEDULER)
==============================================================================================================================
• Thời điểm bắt đầu    : 2026-10-05 00:29:36
• Mốc hạn chót          : 08:00 (2026-10-05 08:00:00)
• Quỹ thời gian có sẵn : 7:30:23
• Tốc độ AI cơ sở       : Trung vị 1.66 fps | P5 1.62 fps (đo thực tế trong pipeline M3 Pro C1)
• Overhead mỗi tập      : 660s (~11.0 phút: 360s video + 300s phụ đề)
• Biên an toàn ETA      : +7.0% trên ETA P5 để ra quyết định hạn chót
• Cho phép dở dang      : --allow-partial = BẬT
------------------------------------------------------------------------------------------------------------------------------
STT  | Tập Video                          | Tổng Frames | n_unique  | Nguồn n_unique   | Dedup % | ETA (Med) | ETA P5 (+7%) | Kết Thúc | Quyết Định
------------------------------------------------------------------------------------------------------------------------------
[01] | I.Want.to.Love.You.Till.Your....   | 34046       | 15669     | đo (plan.json)   | 54.0  % | 2:48:19   | 3:04:15      | 03:17:55 | ✅ KỊP
[02] | [SubsPlease] Re Zero kara Haj...   | 34046       | 20876     | đo (plan.json)   | 38.7  % | 3:40:36   | 4:01:35      | 06:58:31 | ✅ KỊP
[03] | [SubsPlease] Tensei Shitara S...   | 34525       | 18772     | đo (plan.json)   | 45.6  % | 3:19:28   | 3:38:25      | 10:18:00 | ⏸️ CHẠY 1 PHẦN (HẠN CHÓT)
------------------------------------------------------------------------------------------------------------------------------
Ghi chú: Cột 'Nguồn n_unique' phân định rõ giữa số đo thực tế từ plan.json và số ước lượng từ metadata (~39.1% kept).
         Quyết định hạn chót dựa trên P5 (1.62 fps) + 5 phút subtitles + biên an toàn 7% ETA.
🎯 TỔNG KẾT: Duyệt thực thi 3/3 tập (gồm 1 tập chạy 1 phần) | Tổng thời lượng chạy dự kiến: 6:43:28.
==============================================================================================================================
```

---

## 2. Nhật Ký Giám Sát Định Kỳ B3 (~20 phút/lần)

| Thời điểm | PID | Tập đang chạy / Chunk | Đĩa trống | RAM Disk trống | Swap Used / Free | Nhiệt độ (`pmset`) | Trạng thái ghi nhận |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **00:29** | 51394 | Tập 1: Đoạn 01/178 (Metal AI) | 36.08 GiB | 3.355 MB | 1463 MB / 1608 MB | Normal | Preflight nền Tập 2 xong (20.876 kept). Preflight Tập 3 đang chạy. |
| **00:50** | 51394 | Tập 1: Đoạn 24/178 (12.9%) | 33 GiB | 2.7 GiB | 2820 MB / 1276 MB | Normal | AI 1.61–1.62 fps ổn định; Preflight Tập 3 xong (18.772 kept, 45.6% dedup); Cả 3 tập đã có số đo thực. |
| **01:11** | 51394 | Tập 1: Đoạn 50/178 (27.5%) | 33 GiB | 2.9 GiB | 2788 MB / 1308 MB | Normal | AI 1.58–1.60 fps ổn định; Swap giảm nhẹ (-32 MB); RAM Disk giải phóng sạch sau từng chunk. |
| **01:31** | 51394 | Tập 1: Đoạn 76/178 (42.1%) | 33 GiB | 3.0 GiB | 2851 MB / 1245 MB | Normal | AI 1.59–1.63 fps ổn định; Tiến độ tăng +26 chunk/20m; RAM Disk giải phóng đều; Nhiệt độ bình thường. |
| **01:51** | 51394 | Tập 1: Đoạn 101/178 (56.2%) | 33 GiB | 3.4 GiB | 2843 MB / 1253 MB | Normal | AI 1.57–1.62 fps ổn định; Đạt >50% tập 1 (+25 chunk/20m); RAM Disk trống 3.4 GiB; Nhiệt độ bình thường. |

---

## 3. Báo Cáo Kết Thúc Hàng Đợi (B5)

### 1. Thời Gian Thực Thi & Trạng Thái Từng Tập
* **Thời điểm bắt đầu phiên:** 2026-10-05 00:25:59 +07:00 *(SỐ ĐO)*
* **Thời điểm kết thúc phiên:** 2026-10-05 08:00:02 +07:00 *(SỐ ĐO)*
* **Tổng thời gian phiên chạy:** 7 giờ 34 phút 03 giây (27.243 giây) *(SỐ ĐO)*
* **Trạng thái từng tập trong hàng đợi:**
  * **Tập 1 (`I.Want.to.Love.You...S01E13`):**
    * Trạng thái: **Xong trọn vẹn 178/178 chunks** (4K MKV + Phụ đề `.vi.srt`) *(SỐ ĐO)*
    * Thời gian thực tế: **2 giờ 51 phút 57 giây** (10.317s) *(SỐ ĐO)*
    * So với ETA Trung vị (2:48:19): chậm hơn +3m 38s (+2.1%)
    * So với ETA P5 bảo thủ (+7%) (3:04:15): nhanh hơn 12m 18s (-6.7%)
  * **Tập 2 (`[SubsPlease] Re Zero - 84`):**
    * Trạng thái: **Xong trọn vẹn 178/178 chunks** (4K MKV + Phụ đề `.vi.srt`) *(SỐ ĐO)*
    * Thời gian thực tế: **3 giờ 47 phút 01 giây** (13.621s) *(SỐ ĐO)*
    * So với ETA Trung vị (3:40:36): chậm hơn +6m 25s (+2.9%)
    * So với ETA P5 bảo thủ (+7%) (4:01:35): nhanh hơn 14m 34s (-6.0%)
  * **Tập 3 (`[SubsPlease] Slime S4 - 24`):**
    * Trạng thái: **Dừng sạch tại ranh giới phân đoạn 31/180** theo cơ chế `--allow-partial` lúc 07:51:23 *(SỐ ĐO)*
    * Giữ an toàn 31 file phân đoạn `up_*.mp4` (tổng 560 MB) trong `.temp_[SubsPlease] Tensei Shitara Slime Datta Ken S4 - 24 (1080p) [AA94BC1F]/upscaled_segs/` *(SỐ ĐO)*
    * Tự động loại khỏi mux thành phẩm và Stage B phụ đề để nhường đúng 10 phút an toàn cho việc hoàn tất phụ đề của Tập 1 và Tập 2 trước 08:00.

### 2. Kiểm Định Tự Động (`audit.json`)
* **Tập 1 (`I.Want.to.Love.You...S01E13_4K.audit.json`):**
  * Status: **PASS** *(SỐ ĐO)*
  * Audio Bit-exact MD5: Khớp nguồn (`17054753e3187237bd4e5b5dbfdf8881` vs `17054753e3187237bd4e5b5dbfdf8881`) *(SỐ ĐO)*
  * Độ lệch thời lượng: `0.0s`, Độ lệch A/V: `0.0 frames` *(SỐ ĐO)*
  * Luồng mux: 1 Audio, 8 Subtitles, 11 Fonts, 3 Chapters khớp nguồn *(SỐ ĐO)*
  * Kiểm tra đan xen 10s (Window Interleaving): 11/11 cửa sổ đạt chuẩn *(SỐ ĐO)*
  * Kiểm tra âm lượng tại 7 mốc seek: Lệch 0.0 dB trên cả 7 điểm *(SỐ ĐO)*
* **Tập 2 (`[SubsPlease] Re Zero - 84_4K.audit.json`):**
  * Status: **PASS** *(SỐ ĐO)*
  * Audio Bit-exact MD5: Khớp nguồn (`a3f0a36fcffcc258656fc340e3d45fa4` vs `a3f0a36fcffcc258656fc340e3d45fa4`) *(SỐ ĐO)*
  * Độ lệch thời lượng: `0.0s`, Độ lệch A/V: `0.0 frames` *(SỐ ĐO)*
  * Luồng mux: 1 Audio, 1 Subtitle, 24 Fonts, 0 Chapters khớp nguồn *(SỐ ĐO)*
  * Kiểm tra đan xen 10s: 11/11 cửa sổ đạt chuẩn *(SỐ ĐO)*
  * Kiểm tra âm lượng tại 7 mốc seek: Lệch 0.0 dB trên cả 7 điểm *(SỐ ĐO)*
* **Tập 3:** Chưa mux thành phẩm do dừng dở dang `--allow-partial` (không sinh file audit.json).

### 3. Giai Đoạn Phụ Đề (Stage B Subtitles)
* **Tập 1:** Đã dịch thành công ra `I.Want.to.Love.You..._4K.vi.srt` (424 cues) trong 228.5s *(SỐ ĐO)*.
  * Tỷ lệ cue theo model: `gemini-3.6-flash`: 421 cues (99.3%), `cache`: 3 cues (0.7%) *(SỐ ĐO)*.
  * Số cue gắn cờ `uncertain`: **0 cues (0.0%)** *(SỐ ĐO)*.
* **Tập 2:** Đã dịch thành công ra `[SubsPlease] Re Zero - 84_4K.vi.srt` (322 cues) trong 281.1s *(SỐ ĐO)*.
  * Tỷ lệ cue theo model: `gemini-3.6-flash`: 200 cues (62.1%), `gemini-3.5-flash`: 122 cues (37.9%) *(SỐ ĐO)*.
  * Số cue gắn cờ `uncertain`: **0 cues (0.0%)** *(SỐ ĐO)*.
* **Cảnh báo hạn mức & Lỗi mạng:**
  * Lỗi 429 Quota Exhaustion (`PerDay`): **0 lần** *(SỐ ĐO)*.
  * Lỗi 503 UNAVAILABLE (quá tải tạm thời Google Cloud): 21 lần ghi nhận, pipeline đã tự động retry và hạ cấp mô hình thông suốt theo chuỗi 3.8 ➔ 3.7 ➔ 3.6 ➔ 3.5-flash *(SỐ ĐO)*.
* **Bộ nhớ mô hình cục bộ (`ollama ps`):** Rỗng (0 mô hình chiếm RAM) *(SỐ ĐO)*.

### 4. Mốc Đo Lường Cuối Phiên (Hệ Thống & Phần Cứng)
* **GB ghi SSD trong phiên:**
  * `iostat -Id disk0` ban đầu (00:25:26): 2.504.141,15 MB *(SỐ ĐO)*
  * `iostat -Id disk0` cuối phiên (13:55:44): 2.689.767,30 MB *(SỐ ĐO)*
  * Lượng dữ liệu ghi SSD tích lũy: **185.626 MB (~181,28 GiB)** *(SỐ ĐO)* (Bao gồm 2 file MKV 4K ~3.4 GB, 31 chunks dở dang ~560 MB, swap I/O và các tiến trình nền macOS).
* **Pageouts (`vm_stat`):**
  * Ban đầu: 113.487 ➔ Cuối phiên: 120.731 ➔ Tăng thêm **7.244 pageouts** trên tổng 13,5 giờ hoạt động *(SỐ ĐO)*.
### 4. Mốc Đo Lường Cuối Phiên (Hệ Thống & Phần Cứng)
* **GB ghi SSD TRONG CA (tại mốc kết thúc 08:00:29):**
  * `iostat -Id disk0` ban đầu (00:25:59): 2.504.159,84 MB *(SỐ ĐO từ soak CSV)*
  * `iostat -Id disk0` kết thúc ca (08:00:29): 2.632.595,68 MB *(SỐ ĐO từ soak CSV)*
  * Lượng dữ liệu ghi SSD tích lũy TRONG CA: **128.435,84 MB (~125,43 GiB / 128,44 GB)** *(SỐ ĐO)*.
  * *(Lưu ý tách bạch: Khoảng thời gian từ 08:00:29 đến 13:55:44 là NGOÀI CA làm việc của hàng đợi, phát sinh thêm 55,83 GiB do hệ điều hành macOS và môi trường IDE Antigravity, nâng tổng số đo lúc 13:55 lên 181,28 GiB).*
* **Pageouts (`vm_stat`) TRONG CA:**
  * Ban đầu: 113.487 ➔ Kết thúc ca (08:00:29): 120.677 ➔ Tăng thêm **7.190 pageouts** trong 7,5 giờ ca chạy *(SỐ ĐO)*. (Thời gian ngoài ca 08:00–13:55 chỉ tăng thêm 54 pageouts).
* **Swap (`sysctl vm.swapusage`) TRONG CA:**
  * Ban đầu: `used = 1463.38 MB` ➔ Kết thúc ca (08:00:29): `used = 2539.31 MB` (mức tăng +1.075,93 MB, đỉnh ghi nhận 2.715,31 MB lúc 07:54) *(SỐ ĐO)*. (Đến 13:55 macOS tự thu hồi còn 2.499,31 MB).
* **Dung lượng đĩa trống:** **30 GiB** trên `/System/Volumes/Data` (vẫn vượt xa ngưỡng an toàn 8 GiB) *(SỐ ĐO)*.
* **Cảnh báo nhiệt (`pmset -g therm`):** **0 lần** bất thường (giữ trạng thái Normal xuyên suốt phiên) *(SỐ ĐO)*.

### 5. Việc Cần Thao Tác Bằng Tay
1. **Kiểm tra nghe thử 3 điểm:** Mở IINA nghe thử 3 điểm (đầu, giữa, cuối) trên cả 2 file MKV 4K để thẩm định đồng bộ âm thanh:
   * `/Users/xis108/Documents/Upscaled/I.Want.to.Love.You.Till.Your.Dying.Day.S01E13.I.Miss.You.REPACK.1080p.CR.WEB-DL.JPN.AAC2.0.H.264.MSubs-ToonsHub_4K.mkv`
   * `/Users/xis108/Documents/Upscaled/[SubsPlease] Re Zero kara Hajimeru Isekai Seikatsu - 84 (1080p) [5188BBC4]_4K.mkv`
2. **Kiểm tra phụ đề IINA:** Kéo video vào IINA, xác nhận IINA tự nạp file phụ đề `.vi.srt` nằm cạnh và khớp khẩu hình ở 3 mốc (đầu, giữa, cuối).
3. **Lệnh Resume tiếp tục Tập 3 (KHÔNG tự chạy):**
   ```bash
   ./upscale.sh "/Users/xis108/Documents/RawSources/[SubsPlease] Tensei Shitara Slime Datta Ken S4 - 24 (1080p) [AA94BC1F].mkv" -o "/Users/xis108/Documents/Upscaled/[SubsPlease] Tensei Shitara Slime Datta Ken S4 - 24 (1080p) [AA94BC1F]_4K.mkv"
   ```
   *Hệ thống sẽ tự nhận diện 31 phân đoạn đã upscale có sẵn và tiếp tục ngay từ phân đoạn 32.*
4. **Lệnh tạo phụ đề Tập 3 sau khi upscale xong (KHÔNG tự chạy):**
   ```bash
   ./upscale.sh --subs-only "/Users/xis108/Documents/RawSources/[SubsPlease] Tensei Shitara Slime Datta Ken S4 - 24 (1080p) [AA94BC1F].mkv" -o "/Users/xis108/Documents/Upscaled" --sub-offset-ms -11154
   ```

### 6. Quan Sát Để Cải Thiện (Chỉ Số Đo, Không Sửa Mã)
* **a. Phân bổ thời gian theo giai đoạn & Tỷ lệ GPU Active / Idle (Chuẩn hóa):**
  * *Nguyên lý tính toán đúng:* `GPU Active Ratio = Tổng thời gian AI Infer ÷ Thời gian thực của tập`. Tuyệt đối không dùng tổng thời gian chunk cộng dồn vì các stage extract (CPU), infer (Metal GPU), encode (Media Engine) chạy gối đầu song song (triple-overlap concurrency), khiến tổng thời gian các chunk cộng lại vượt thời gian thực tế.
  * **Tập 1:** Thời gian thực tập 10.316,9s; Tổng thời gian AI Metal Infer: 9.626,0s ➔ **GPU Active Ratio: 93.30% (GPU Idle: 6.70%)** *(SỐ ĐO)*.
  * **Tập 2:** Thời gian thực tập 13.620,6s; Tổng thời gian AI Metal Infer: 12.454,5s ➔ **GPU Active Ratio: 91.44% (GPU Idle: 8.56%)** *(SỐ ĐO)*.
  * **Tập 3 (31 chunks dở dang):** Thời gian thực 2.770,0s; Tổng thời gian AI Metal Infer: 2.437,3s ➔ **GPU Active Ratio: 87.99% (GPU Idle: 12.01%)** *(SỐ ĐO)*.
  * **10 chunk chậm nhất Tập 1 (Wall time):** Chunk 50 (179.0s), Chunk 17 (169.0s), Chunk 132 (168.1s), Chunk 13 (167.9s), Chunk 95 (166.1s), Chunk 107 (165.6s), Chunk 138 (157.8s), Chunk 169 (155.0s), Chunk 137 (154.2s), Chunk 114 (152.0s) *(SỐ ĐO)*.
  * **So sánh FPS giờ đầu vs giờ cuối:**
    * Tập 1: Giờ đầu 1.608 fps vs Giờ cuối 1.653 fps (Mức tụt: **-2.83%**, tăng tốc nhẹ, không sụt giảm) *(SỐ ĐO)*.
    * Tập 2: Giờ đầu 1.654 fps vs Giờ cuối 1.672 fps (Mức tụt: **-1.06%**, không sụt giảm) *(SỐ ĐO)*.
    * Không có bất kỳ cảnh báo sụt giảm FPS > 8% nào trên toàn bộ 3 tập.
* **b. So sánh ETA dự đoán so với thực tế:**
  * Tập 1: Dự đoán P5+7% là 3:04:15, Thực tế chạy mất 2:51:57 (Sai số dự đoán: +12m 18s bảo thủ an toàn) *(SỐ ĐO)*.
  * Tập 2: Dự đoán P5+7% là 4:01:35, Thực tế chạy mất 3:47:01 (Sai số dự đoán: +14m 34s bảo thủ an toàn) *(SỐ ĐO)*.
  * Số frames unique (`n_unique`):
    * Tập 1: Ước lượng 13.312 frames ➔ Đo thực `plan.json`: **15.669 frames** (+17.7%) *(SỐ ĐO)*.
    * Tập 2: Ước lượng 13.415 frames ➔ Đo thực `plan.json`: **20.876 frames** (+55.6% do nhiều phân cảnh hành động liên tục) *(SỐ ĐO)*.
    * Tập 3: Ước lượng 13.513 frames ➔ Đo thực `plan.json`: **18.772 frames** (+38.9%) *(SỐ ĐO)*.
* **c. Tác động của Preflight nền lên FPS Metal:**
  * Khi Preflight nền quét tập 2 & 3 (Chunk 1 và 2 Tập 1): Tốc độ AI đạt trung bình **1.605 fps** *(SỐ ĐO)*.
  * Sau khi Preflight nền hoàn tất (Chunk 3 đến 10 Tập 1): Tốc độ AI đạt trung bình **1.608 fps** *(SỐ ĐO)*.
  * Độ chênh lệch: chỉ **0.003 fps (0.19%)**, chứng minh việc gán `nice -n 10` cho preflight hoàn toàn triệt tiêu ảnh hưởng tới GPU Metal.
* **d. Thống kê Hạn mức Gemini API theo ngày Thái Bình Dương (2026-10-04 PT):**
  * `gemini-3.6-flash`: 14 thành công, 11 lỗi 503, 1 lỗi mạng.
  * `gemini-3.5-flash`: 5 thành công, 0 lỗi.
  * `gemini-3.7-flash`: 1 thành công, 9 lỗi 503, 2 lỗi mạng.
  * `gemini-3.8-flash`: 0 thành công, 10 lỗi 503, 2 lỗi mạng.
  * Số lỗi 429 (`PerDay` / `PerMinute`): **0**.
  * Giá trị `quotaValue`: **Không có** (do chưa từng chạm trần 429).
* **e. Hiệu suất Batch phụ đề:**
  * Kích thước batch: 100 cues.
  * Tập 1: 17 batches, 0 batch phải tách đôi (Split rate 0.0%) *(SỐ ĐO)*.
  * Tập 2: 18 batches, 0 batch phải tách đôi (Split rate 0.0%) *(SỐ ĐO)*.
* **f. Phân loại Cảnh báo / Lỗi trong Log:**
  * Tổng số lỗi nghiêm trọng (ERROR / ❌): **0** *(SỐ ĐO)*.
  * Cảnh báo quá tải máy chủ (503 UNAVAILABLE): 21 lần.
  * Cảnh báo hạ cấp mô hình (Fallback): 5 lần.
  * Cảnh báo không thể nhúng phụ đề trực tiếp vào container MKV: 2 lần (file `.vi.srt` ngoài vẫn toàn vẹn).
  * Cảnh báo kiểm soát dòng chảy RAM Disk (Backpressure pause 10-20s): 11 lần (bảo vệ tuyệt đối không tràn 4.0 GB RAM Disk).
* **g. Dữ liệu ghi SSD:**
  * Ghi thêm ~125.43 GiB trong thời gian ca chạy (00:25:59 đến 08:00:29). Sau ca chạy đến 13:55 phát sinh thêm 55.83 GiB ngoài ca do OS macOS và Antigravity IDE.

---

## 4. Đo Lường Độ Lệch Phụ Đề (B6)

* **Script thực thi:** `/Users/xis108/night1_analysis/subtitle_drift.py` (đặt độc lập ngoài repo)
* **Lệnh chạy:** `./venv/bin/python3 /Users/xis108/night1_analysis/subtitle_drift.py`
* **File dữ liệu kết quả:** `/Users/xis108/night1_analysis/subtitle_drift_results.json`

### 1. Bảng Độ Lệch Trung Vị & Tỷ Lệ Khớp Theo 5 Ngũ Phân Vị (Quintiles)

#### Tập 1: `I.Want.to.Love.You.Till.Your.Dying.Day.S01E13` (Gợi ý offset: -111 ms trên JA)
| Ngũ Phân Vị (Khoảng thời gian) | Số Cues JA | Khớp (Chưa bù) | Tỷ lệ khớp | Offset Trung Vị (Chưa bù) | Offset Trung Vị (Đã bù -111ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Q1 (0.0s – 282.2s)** | 97 | 57 | 58.8% | +0.136s | +0.025s |
| **Q2 (282.2s – 564.5s)** | 56 | 42 | 75.0% | +0.164s | +0.053s |
| **Q3 (564.5s – 846.7s)** | 90 | 64 | 71.1% | +0.155s | +0.044s |
| **Q4 (846.7s – 1128.9s)** | 82 | 61 | 74.4% | +0.163s | +0.052s |
| **Q5 (1128.9s – 1411.2s)** | 99 | 71 | 71.7% | +0.156s | +0.045s |
* *Tỷ lệ khớp toàn tập (Chưa bù):* **70.3%**; *Đã bù (-111 ms):* **71.5%** *(SỐ ĐO)*.
* *Offset trung vị toàn tập (Chưa bù):* **+0.152s**; *Đã bù (-111 ms):* **+0.045s** *(SỐ ĐO)*.
* *Chênh lệch trung vị cực đại giữa các phần:* **0.027s** (<= 0.5s: ✅ Ổn định, không có hiện tượng trôi dạt tích lũy) *(SỐ ĐO)*.

#### Tập 2: `[SubsPlease] Re Zero kara Hajimeru Isekai Seikatsu - 84` (Gợi ý offset: -60 ms trên JA)
| Ngũ Phân Vị (Khoảng thời gian) | Số Cues JA | Khớp (Chưa bù) | Tỷ lệ khớp | Offset Trung Vị (Chưa bù) | Offset Trung Vị (Đã bù -60ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Q1 (0.0s – 284.0s)** | 71 | 54 | 76.1% | +0.084s | +0.024s |
| **Q2 (284.0s – 568.0s)** | 74 | 58 | 78.4% | +0.099s | +0.039s |
| **Q3 (568.0s – 852.0s)** | 65 | 53 | 81.5% | +0.103s | +0.043s |
| **Q4 (852.0s – 1136.0s)** | 60 | 42 | 70.0% | +0.107s | +0.047s |
| **Q5 (1136.0s – 1420.0s)** | 52 | 38 | 73.1% | +0.114s | +0.054s |
* *Tỷ lệ khớp toàn tập (Chưa bù):* **77.0%**; *Đã bù (-60 ms):* **77.0%** *(SỐ ĐO)*.
* *Offset trung vị toàn tập (Chưa bù):* **+0.100s**; *Đã bù (-60 ms):* **+0.040s** *(SỐ ĐO)*.
* *Chênh lệch trung vị cực đại giữa các phần:* **0.036s** (<= 0.5s: ✅ Ổn định, không có hiện tượng trôi dạt tích lũy) *(SỐ ĐO)*.

#### Tập 3: `[SubsPlease] Slime S4 - 24` (Gợi ý offset: -11.154 s trên JA / +11,154 s trên EN)
| Ngũ Phân Vị (Khoảng thời gian) | Số Cues JA | Khớp (Chưa bù) | Tỷ lệ khớp | Offset Trung Vị (Chưa bù) | Offset Trung Vị (Đã bù -11.154s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Q1 (0.0s – 283.2s)** | 56 | 46 | 82.1% | -0.009s | -0.009s |
| **Q2 (283.2s – 566.4s)** | 98 | 69 | 70.4% | -0.001s | +0.001s |
| **Q3 (566.4s – 849.5s)** | 71 | 56 | 78.9% | -0.018s | -0.018s |
| **Q4 (849.5s – 1132.7s)** | 59 | 50 | 84.7% | -0.031s | -0.031s |
| **Q5 (1132.7s – 1415.9s)** | 83 | 60 | 72.3% | +0.006s | +0.006s |
* *Tỷ lệ khớp toàn tập (Chưa bù):* **48.2%** (chỉ 205 cues); *Đã bù (-11.154 s):* **76.6%** (tăng vọt lên 313 cues, +28.4 điểm phần trăm) *(SỐ ĐO)*.
* *Offset trung vị toàn tập sau bù:* **-0.007s** (giảm sát về 0) *(SỐ ĐO)*.
* *Chênh lệch trung vị cực đại giữa các phần:* **0.035s** (<= 0.5s: ✅ Cực kỳ phẳng và đồng nhất trên toàn bộ tập) *(SỐ ĐO)*.

### 2. Danh Sách 5 Cues Có Độ Lệch Tuyệt Đối Lớn Nhất Từng Tập
* **Tập 1:**
  1. Cue #245 (855.5s): $\Delta$ -4.608s — JA: `1608 1614 1706……` / EN: `I didn't get called.`
  2. Cue #41 (134.2s): $\Delta$ +2.557s — JA: `もしかして〜って思ったんだけどな〜` / EN: `But the mood between them...`
  3. Cue #213 (766.9s): $\Delta$ +2.536s — JA: `こんなに濡れてたら ろうか歩けない……` / EN: `But I'm not sure what to do...`
  4. Cue #208 (752.4s): $\Delta$ +2.372s — JA: `前髪も服もまるで気にしないのね` / EN: `She's so beautiful, but she doesn't...`
  5. Cue #147 (544.8s): $\Delta$ +2.364s — JA: `先パイに対して あんなたいど...` / EN: `I'm the one who should be apologizing...`
* **Tập 2:**
  1. Cue #171 (698.4s): $\Delta$ +2.731s — JA: `これも新しい価値観の発見になるのかなァってね` / EN: `Perhaps you killing yourself...`
  2. Cue #130 (497.7s): $\Delta$ +2.404s — JA: `何もあなただけじゃないのよん？` / EN: `you're not the only one...`
  3. Cue #174 (714.8s): $\Delta$ -2.376s — JA: `記憶を利用する月食と 存在そのものを利用する日食` / EN: `while Solar Eclipse uses...`
  4. Cue #25 (112.9s): $\Delta$ -2.288s — JA: `古の盟約により 頂へ至る者の志を问わん` / EN: `I ask the will of the one...`
  5. Cue #215 (891.4s): $\Delta$ +1.954s — JA: `見つけました` / EN: `Ah, we found you!`
* **Tập 3:**
  1. Cue #208 (780.3s): $\Delta$ +13.970s — JA: `世界など滅んでしまえばよいのです……` / EN: `may as well fall at the hands...`
  2. Cue #225 (849.8s): $\Delta$ +13.886s — JA: `あの娘に 託したく……` / EN: `Give this—my hope—to her.`
  3. Cue #194 (722.4s): $\Delta$ +13.839s — JA: `クロエの『ヨグ・ソトース』の中に...` / EN: `but her soul—her Preserver...`
  4. Cue #163 (599.6s): $\Delta$ +13.281s — JA: `一緒に旅をした勇者はあなただったんだね` / EN: `So you were the Hero who saved me...`
  5. Cue #273 (1068.6s): $\Delta$ +13.208s — JA: `ううん ルミナスと『未来』で会えて嬉しい！` / EN: `I'm glad I got to meet you here...`

### 3. Đánh Giá Biến Thiên Giữa Các Phần
* Mức chênh lệch cực đại giữa 5 phần của cả 3 tập đều nằm trong khoảng **0.027s – 0.036s**, nhỏ hơn rất nhiều so với ngưỡng cảnh báo **0.500s**.
* Khẳng định: **Không có hiện tượng trôi dạt lũy tiến theo thời gian (drift)** do lệch fps hay timescale.

### 4. Kết Luận Khuyến Nghị
* **Tập 1 (`I.Want.to.Love.You...S01E13`):** **Khớp tự nhiên**. Độ lệch thô chỉ +0.152s, tỷ lệ khớp tự nhiên 70.3%, không cần bất kỳ offset nào.
* **Tập 2 (`Re Zero - 84`):** **Khớp tự nhiên**. Độ lệch thô chỉ +0.100s, tỷ lệ khớp tự nhiên 77.0%, không cần bất kỳ offset nào.
* **Tập 3 (`Slime S4 - 24`):** **Cần một offset cố định duy nhất**: `--sub-offset-ms -11154` (bù đúng 11,154 s thời lượng thẻ tài trợ mở đầu của bản ABEMA). Sau khi bù offset này, độ lệch trung vị giảm về xấp xỉ `0.000s` (-0.007s) và tỷ lệ khớp đạt `76.6%` (tăng +28.4 điểm phần trăm) đồng đều từ đầu đến cuối tập. Không cần offset theo đoạn.

---

## 9. Phân Tích Chỉ Đọc Từ `api_requests` (Subtitles & Gemini API Usage)

* Dữ liệu truy vấn trực tiếp từ SQLite cache `subtrans/.cache/translations.db` (55 requests thực tế trong ca chạy 00:25:59 – 08:00:29 ngày 2026-10-05, không tính dry-run):

| Model | Tổng Requests | Thành Công | Tỷ Lệ Thành Công | Lỗi 503 (Demand Spike) | Tỷ Lệ 503 | Lỗi Khác (Chuyển Fallback) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gemini-3.8-flash`** | 12 | 0 | **0.0%** | 10 | **83.3%** | 2 |
| **`gemini-3.7-flash`** | 12 | 1 | **8.3%** | 9 | **75.0%** | 2 |
| **`gemini-3.6-flash`** | 26 | 14 | **53.8%** | 11 | **42.3%** | 1 |
| **`gemini-3.5-flash`** | 5 | 5 | **100.0%** | 0 | **0.0%** | 0 |

### Phân Bổ Theo Giờ Thái Bình Dương (Pacific Time - PT):
* Ca chạy dịch phụ đề diễn ra lúc 07:51 – 08:00 (giờ VN, UTC+7), tức **00:51 – 01:00 UTC ngày 2026-10-05**, tương ứng **17:51 – 18:00 Pacific Daylight Time (PDT, UTC-7) ngày 2026-10-04 (Chủ nhật)**.
  - **Khung 17:00 – 18:00 PT (PDT):**
    * `gemini-3.8-flash`: 12 requests (0 thành công, 10 lần 503 UNAVAILABLE, 2 lỗi khác).
    * `gemini-3.7-flash`: 12 requests (1 thành công, 9 lần 503 UNAVAILABLE, 2 lỗi khác).
    * `gemini-3.6-flash`: 26 requests (14 thành công [53.8%], 11 lần 503, 1 lỗi khác).
    * `gemini-3.5-flash`: 3 requests (3 thành công 100%, 0 lần 503).
  - **Khung 18:00 – 19:00 PT (PDT):**
    * `gemini-3.5-flash`: 2 requests (2 thành công 100%, 0 lần 503).
* **Nhận định & Khuyến nghị:**
  - Toàn bộ 55 requests diễn ra trong một cửa sổ ~9 phút (07:51 – 08:00 VN, tương ứng 17:51 – 18:00 PDT). Trong cửa sổ ngắn này, `gemini-3.8-flash` và `gemini-3.7-flash` bị lỗi 503 tạm thời. Do chỉ là quan sát trong một khoảng thời gian hẹp ~9 phút, không thể suy diễn thành quy luật theo khung giờ.
  - Chuỗi fallback tự động đa tầng (`3.8 -> 3.7 -> 3.6 -> 3.5`) đã phát huy tác dụng: khi 3.8 và 3.7 gặp lỗi 503 trong cửa sổ đó, 3.6 và 3.5 đã tự động hoàn thành các câu thoại mà không làm gián đoạn tiến trình.
  - **Quy tắc bất biến:** Không thay đổi chuỗi model (giữ nguyên `3.8 -> 3.7 -> 3.6 -> 3.5 -> local`).

---

## 10. Phân Rã Thành Phần Thời Gian Không Chồng Lấp (Non-overlapping Phase Breakdown)

* Phân tích chỉ đọc từ log chi tiết của 2 tập đã hoàn tất trọn vẹn:
  - **Tập 1 (`I.Want.to.Love.You - 13`, 178 chunks):** Thời gian thực tế **10.322,0s** (~2h 52m 02s), tổng thời gian AI Metal **9.626,0s**. Thành phần không chồng lấp ngoài AI: **696,0s (~11,6 phút)**.
  - **Tập 2 (`Re Zero - 84`, 178 chunks):** Thời gian thực tế **13.629,0s** (~3h 47m 09s), tổng thời gian AI Metal **12.454,5s**. Thành phần không chồng lấp ngoài AI: **1.174,5s (~19,6 phút)**.

| Thành Phần / Pha | Tập 1 (Giây) | Tập 1 (%) | Tập 2 (Giây) | Tập 2 (%) | Bản Chất Kỹ Thuật & Phân Loại |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Chờ RAM Disk Backpressure** | **540,0s** | 77.6% | **1.145,0s** | 97.5% | **SỐ ĐO**: RAM Disk 4GB bị đầy (`shutil.disk_usage < need`), CPU decode & AI infer phải tạm dừng chờ Media Engine encode xong và dọn sạch PNG chunk trước. |
| **2. Khởi Động Phiên (Startup Phase)** | **~25,0s** | 3.6% | **~15,0s** | 1.3% | **SỐ ĐO**: Mount RAM Disk, probe metadata, chia chunk 8s, nạp model PyTorch MPS vào VRAM, warm-up inference và trích xuất chunk 1. |
| **3. Đuôi Cuối Tập (Tail Phase)** | **~25,0s** | 3.6% | **~21,0s** | 1.8% | **SỐ ĐO**: Encode phần cứng cho chunk cuối cùng (chunk 178), ghép nối Concat Demuxer, Bit-exact Stream Copy Muxing (audio, subs, fonts, chapters) và Mux Audit. |
| **4. Làm Mới Worker (MPS Restarts)** | **~4,0s** | 0.6% | **~4,0s** | 0.3% | **SỐ ĐO**: 17 lần tái khởi tạo worker định kỳ sau mỗi 10 chunks (~0.23s/lần) để triệt tiêu tích tụ cache MPS và giữ footprint VRAM phẳng. |
| **5. Đồng Bộ Luồng & Handshake** | **~102,0s** | 14.6% | **-10,5s** | — | **PHẦN DƯ (Residual)**: Tính toán từ $\text{Thời gian ngoài AI} - \sum(\text{Pha đo được})$. Phản ánh độ lệch đồng bộ giữa các luồng (Tập 2 âm nhẹ -10.5s do các pha gối đầu nhẹ lên nhau). |
| **TỔNG CỘNG** | **696,0s** | 100% | **1.174,5s** | 100% | **Khớp chính xác với độ lệch đo đạc thực tế.** |

### Đối Chiếu Số Liệu Khung Hình Giữ Lại ($n_{unique}$) & Tỷ Lệ Dedup:
* **Tập 1 (`I.Want.to.Love.You...S01E13`, 34.046 frames):**
  * Giữ lại: **15.669 frames (46.0% kept)** | Loại bỏ: **18.377 frames (54.0% dropped)** *(SỐ ĐO)*.
* **Tập 2 (`[SubsPlease] Re Zero - 84`, 34.046 frames):**
  * Số đo thô từ `mpdecimate`: **20.876 frames (61.3% kept)** | Loại bỏ: **13.170 frames (38.7% dropped)** *(SỐ ĐO)*.
  * Số thực tế đưa vào AI sau ép 101 frames ranh giới chunk: **20.977 frames (61.6% kept)** | Loại bỏ: **13.069 frames (38.4% dropped)** *(SỐ ĐO)*.
* **Tập 3 (`[SubsPlease] Slime S4 - 24`, 34.046 frames):**
  * Giữ lại: **18.527 frames (54.4% kept)** | Loại bỏ: **15.519 frames (45.6% dropped)** *(SỐ ĐO)*.
* **Kết luận:** Bảng lập lịch chỉ cần dùng $n_{unique}$ sau ép ranh giới (**20.977 frames** cho Tập 2) để tính toán ETA.

### Nguyên Nhân Khác Biệt Thời Gian Chờ Giữa 2 Tập:
* Tập 2 có số frame đưa vào AI ($n_{unique} = 20.977$, 61.6% kept) cao hơn nhiều so với Tập 1 ($n_{unique} = 15.669$, 46.0% kept).
* Do Tập 2 có nhiều phân đoạn hành động nặng (170–192 frames/chunk thay vì 40–80 frames/chunk như Tập 1), dung lượng ảnh PNG 4K sinh ra vượt ngưỡng 2.5–3.0 GB nhanh hơn, kích hoạt cơ chế an toàn backpressure **66 lần** (so với **29 lần** ở Tập 1), dẫn đến thời gian chờ gộp lên tới 1.145s (~19 phút).

### Đánh Giá 3 Cấu Hình Backpressure & Khả Năng Tối Ưu Hóa:

| Cấu Hình | Định Nghĩa Kỹ Thuật | Thời Gian Chờ (Đo / Ước Lượng) | Phân Loại & Đánh Giá Rủi Ro |
| :--- | :--- | :---: | :--- |
| **(a) Hiện tại** | RAM disk 4GB, cổng `need = n_unique * 12.0MB * 1.3` | **540s** (Tập 1), **1.145s** (Tập 2) | **SỐ ĐO THẬT**: 29 lần chờ (Tập 1) và 66 lần chờ (Tập 2). An toàn tuyệt đối, 0 tràn RAM disk, 0 SSD swap. |
| **(b) Cổng kích thước động** | RAM disk 4GB, cổng `need = n_unique * png_size * 1.2` | Giảm chờ ~50–70% *(ƯỚC LƯỢNG chưa đo)* | **ƯỚC LƯỢNG (chưa đo)**. Lưu ý kỹ thuật: Cổng **bắt buộc phải dùng p95 hoặc max kích thước PNG gần đây × 1.2**, tuyệt đối KHÔNG dùng trung bình (~9.8 MB) vì các frame nặng trong cảnh hành động có thể đạt 11–12 MB, nếu dùng trung bình sẽ cho qua quá sớm làm tràn RAM Disk. |
| **(c) Nâng RAM Disk 5GB + Cổng động** | RAM disk 5GB, cổng `need = n_unique * png_size * 1.2` | Triệt tiêu phần lớn thời gian chờ *(ƯỚC LƯỢNG chưa đo)* | **ƯỚC LƯỢNG (chưa đo)**. Chiếm 5GB / 18GB Unified Memory. Có nguy cơ tăng memory pressure khi gối đầu CPU + Metal + Media Engine. |

* **Đối chiếu chéo với GPU idle:**
  * GPU idle trong ca Đêm 1 là **6.70%** (Tập 1) và **8.56%** (Tập 2), gần bằng tỷ lệ thời gian chờ backpressure trên tổng thời gian chạy (**5.2%** và **8.4%**).
  * Điều này chứng minh: Lợi ích tối đa về mặt thời gian nếu triệt tiêu hoàn toàn việc chờ backpressure chỉ là khoảng **5% – 8%**, không hơn. Vì vậy, việc tinh chỉnh này không đáng làm trước ca Đêm 2 để bảo toàn tuyệt đối tính ổn định của hệ thống.

---

## 11. Thẩm Định Thực Tế Của Người Dùng (Human QA Verification)

* **Phản hồi trực tiếp từ người dùng:**
  > *«tôi đã xem và nghe phụ đề toàn bộ tập 1 và 2, tất cả đều ổn không có gì sai sót hay lỗi».*

---

## 12. Điều Tra & Khắc Phục Nhúng Phụ Đề Vào Container MKV

* **Hiện tượng trong ca Đêm 1:** 
  Khi kết thúc Stage B độc lập cho Tập 1 và Tập 2, pipeline in cảnh báo: `⚠️ Không thể nhúng phụ đề vào container 4K, file phụ đề ngoài vẫn sẵn sàng.`
* **Nguyên nhân cốt lõi (Root Cause):**
  - Trong `src/ffmpeg_ops.py`, tại hàm `embed_subtitle_track()`, sau khi `ffmpeg` chạy thành công tạo file tạm `tmp_out`, lệnh `shutil.move(tmp_out, mkv_path)` được gọi.
  - Tuy nhiên, module `shutil` **chưa được import** ở đầu file `src/ffmpeg_ops.py` (chỉ có `os`, `json`, `subprocess`, `glob`, `typing`).
  - Lệnh `ffmpeg` thực tế đã tạo file thành công với mã thoát 0, nhưng Python ném ra `NameError: name 'shutil' is not defined`. Khối `try...except` đã bắt lỗi này và xóa file tạm, dẫn đến cảnh báo hiển thị sai lệch.
* **Đính chính số lượng Font chữ đính kèm:**
  - Con số "16 font" ở log cũ là do lệnh ffprobe kiểm tra nhanh bị giới hạn hiển thị `[:25]`.
  - Kiểm tra toàn diện trên file nguồn và file 4K bằng `probe_video()` xác nhận chính xác:
    * **Tập 1 (`I.Want.to.Love.You...S01E13`):** Đúng **11 font attachments** (TTF/OTF).
    * **Tập 2 (`[SubsPlease] Re Zero - 84`):** Đúng **24 font attachments** (TTF/OTF).
* **Quy trình nhúng phụ đề an toàn mới:**
  1. Ghi luồng phụ đề ra file tạm `mkv_path.embed_sub.tmp.mkv` với đầy đủ map `-map 0:v:0 -map 0:a? -map 1:0 -map 0:s? -map 0:t? -map_chapters 0 -c copy -max_interleave_delta 0`.
  2. Tự động chạy `verify_mux_integrity()` so sánh file tạm với video nguồn gốc:
     - Kiểm tra bit-exact MD5 của toàn bộ luồng audio gốc.
     - Kiểm tra đan xen gói tin 10s (Window Interleaving).
     - Kiểm tra âm lượng tại 7 mốc thời gian seek bằng `volumedetect` (lệch $\le 0.5$ dB).
     - Kiểm tra số luồng audio, số font đính kèm (phải $\ge$ nguồn: 11 font cho Tập 1, 24 font cho Tập 2) và chapters.
  3. **Chỉ khi Mux Audit đạt PASS 100%:** Mới thực hiện `shutil.move` nguyên tử ghi đè lên file 4K thành phẩm.
  4. **Nếu Mux Audit trả về FAIL hoặc gặp ngoại lệ:** Giữ nguyên file 4K gốc không chỉnh sửa, xóa sạch file tạm, in cảnh báo chi tiết lý do và giữ file `.vi.srt` độc lập bên ngoài.
  5. Bổ sung cờ `--no-embed` (mặc định False) cho phép người dùng chủ động tắt bước nhúng và chỉ giữ file phụ đề ngoài nếu muốn.
* **Kết quả nghiệm thu trên bản sao APFS clone (`cp -c`):**
  - **Tập 1:** Embed thành công trong 4.25s (dung lượng tạm 1756.78 MB), Audit status **PASS**, font attachments **11/11**, subs: 9, audio: 1.
  - **Tập 2:** Embed thành công trong 4.95s (dung lượng tạm 1655.12 MB), Audit status **PASS**, font attachments **24/24**, subs: 2, audio: 1.
  - **Failure test:** Cố tình đưa file lệch nguồn để kích hoạt lỗi MD5 audio $\to$ hàm trả về `False`, file 4K được giữ nguyên vẹn 100% byte, file tạm được dọn sạch.



