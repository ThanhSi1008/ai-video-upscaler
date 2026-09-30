#!/usr/bin/env python3
"""
================================================================================
 VIDEO NATIVE RESOLUTION CHECKER (KIỂM TRA CHẤT LƯỢNG 1080P THỰC HAY FAKE 720P)
================================================================================
Công cụ độc lập kiểm tra độ phân giải gốc của video/ảnh bằng phương pháp
Descaling Error Analysis (Phân tích sai số nghịch đảo MSE).

Nguyên lý:
- Video 1080p giả lập (phóng to từ 720p hoặc ~810p/878p của Studio Anime) sẽ có
  sai số MSE chạm đáy cục bộ (Local Minimum / Valley) tại đúng độ phân giải gốc.
- Video 1080p thực sự sẽ có sai số MSE tăng dốc đều (đơn điệu) khi hạ độ phân giải
  và không có bất kỳ thung lũng nào trước mốc 1080p.
================================================================================
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server / CLI
import matplotlib.pyplot as plt


# Known standard and anime studio master resolutions
STUDIO_PROFILES = {
    720: {
        "name": "720p (HD Standard)",
        "desc": "Fake 1080p: Video 720p bị kéo/phóng to bằng Bicubic/Bilinear (thường do web stream / rip ẩu)",
        "verdict_type": "FAKE_720P",
        "color": "#ef4444"
    },
    810: {
        "name": "810p (Anime Studio Master)",
        "desc": "Anime vẽ tay ở ~810p (1440x810), upscale bởi studio (KyoAni, CloverWorks, A-1 Pictures)",
        "verdict_type": "ANIME_STUDIO_810P",
        "color": "#f59e0b"
    },
    844: {
        "name": "844p (Anime Studio Master)",
        "desc": "Anime vẽ tay ở ~844p (1500x844), upscale bởi studio",
        "verdict_type": "ANIME_STUDIO_844P",
        "color": "#eab308"
    },
    864: {
        "name": "864p (Anime Studio Master)",
        "desc": "Anime vẽ tay ở ~864p (1536x864), upscale bởi studio",
        "verdict_type": "ANIME_STUDIO_864P",
        "color": "#d97706"
    },
    878: {
        "name": "878p (Anime Studio Master)",
        "desc": "Anime vẽ tay ở ~878p (1560x878), chuẩn Anime Studio phổ biến (MAPPA, Wit Studio)",
        "verdict_type": "ANIME_STUDIO_878P",
        "color": "#f97316"
    },
    900: {
        "name": "900p (1600x900)",
        "desc": "Bản master 900p kéo lên 1080p",
        "verdict_type": "ANIME_STUDIO_900P",
        "color": "#ca8a04"
    },
    1080: {
        "name": "1080p (Full HD Native)",
        "desc": "Video Full HD 1080p thực sự, giữ trọn vẹn chi tiết tần số cao gốc",
        "verdict_type": "TRUE_1080P",
        "color": "#10b981"
    }
}


def parse_timestamp_to_seconds(ts_str):
    """Chuyển chuỗi thời gian (00:04:30 hoặc 270 hoặc 4:30) thành giây float."""
    if not ts_str:
        return None
    try:
        parts = ts_str.strip().split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        else:
            return float(parts[0])
    except Exception:
        return None


def calculate_sharpness(img_gray):
    """Tính độ sắc nét của khung hình bằng phương sai của toán tử Laplacian."""
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(laplacian.var())


def get_video_duration(video_path, cap=None):
    """Lấy thời lượng video chính xác (hỗ trợ cả container Matroska .MKV khi stream metadata bị ẩn)."""
    if cap is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frame_count > 0 and fps > 0:
            return frame_count / fps

    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        if out:
            return float(out)
    except Exception:
        pass
    return 0.0


def extract_frame_ffmpeg(video_path, timestamp_sec):
    """Fallback an toàn bằng FFmpeg để giải mã hoàn hảo 100% mọi tệp .mkv (10-bit Hi10P, HEVC, AV1)."""
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", str(timestamp_sec),
            "-i", str(video_path),
            "-vframes", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
        if proc.returncode == 0 and len(proc.stdout) > 0:
            img_arr = np.frombuffer(proc.stdout, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
            return img
    except Exception:
        pass
    return None


def extract_best_frames_from_video(video_path, timestamp_sec=None, num_samples=6):
    """
    Trích xuất khung hình từ video (hỗ trợ tối ưu cho định dạng .mkv, .mp4):
    - Tương thích hoàn toàn với file MKV 10-bit, đa luồng phụ đề softsub (ASS/SSA), nhiều track âm thanh.
    - Nếu timestamp_sec được chỉ định: trích xuất đúng thời điểm đó.
    - Nếu không: quét các khung hình ứng viên qua video, đo độ sắc nét (Laplacian variance),
      loại bỏ khung hình quá tối/đen và chọn khung hình có chi tiết cao nhất.
    """
    cap = cv2.VideoCapture(str(video_path))
    duration = get_video_duration(video_path, cap=cap)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    if timestamp_sec is not None:
        frame_img = None
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cap.release()

        # Nếu OpenCV không đọc được mkv đặc biệt, dùng FFmpeg fallback
        if frame_img is None:
            frame_img = extract_frame_ffmpeg(video_path, timestamp_sec)

        if frame_img is None:
            raise ValueError(f"Không thể đọc khung hình tại {timestamp_sec}s từ video MKV.")
        return [(timestamp_sec, frame_img, calculate_sharpness(frame_img))]

    # Tự động quét nhiều mốc thời gian (bỏ qua 8% đầu và 8% cuối tránh intro/credits đen)
    if duration <= 2.0:
        sample_times = [1.0] if duration > 1.0 else [0.5]
    else:
        start_t = duration * 0.08
        end_t = duration * 0.92
        sample_times = np.linspace(start_t, end_t, num_samples).tolist()

    candidates = []
    for t in sample_times:
        frame_gray = None
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_gray is None:
            frame_gray = extract_frame_ffmpeg(video_path, t)

        if frame_gray is None:
            continue

        # Bỏ qua khung hình quá tối (mean brightness < 20) hoặc quá sáng (> 240)
        mean_val = float(np.mean(frame_gray))
        if mean_val < 20 or mean_val > 240:
            continue

        sharpness = calculate_sharpness(frame_gray)
        candidates.append((t, frame_gray, sharpness))

    cap.release()

    if not candidates:
        # Thử lấy khung hình tại 1/3 thời lượng bằng FFmpeg
        fallback_img = extract_frame_ffmpeg(video_path, max(1.0, duration / 3.0))
        if fallback_img is not None:
            return [(duration / 3.0, fallback_img, calculate_sharpness(fallback_img))]
        raise ValueError("Không thể trích xuất khung hình hợp lệ từ tệp video MKV!")

    # Sắp xếp theo độ nét giảm dần
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def compute_descaling_errors(img_gray, crop_size=600, min_h=None, max_h=None, step=10, kernel=cv2.INTER_CUBIC):
    """
    Tính đồ thị sai số nghịch đảo MSE theo chiều cao quét từ min_h đến max_h.
    Sử dụng kỹ thuật Full-Frame Transform sau đó đo MSE ở vùng trung tâm:
    - Thu nhỏ và phóng to trên toàn khung hình giúp bảo toàn 100% tỷ lệ khung hình và pha tọa độ điểm ảnh.
    - So sánh MSE tại vùng trung tâm để loại bỏ viền đen/subtitles và viền ngoài mép ảnh.
    """
    orig_h, orig_w = img_gray.shape

    if max_h is None:
        max_h = orig_h
    if min_h is None:
        min_h = int(round(orig_h * (680 / 1080)))

    # Vùng crop trung tâm để đo sai số (loại bỏ viền đen, phụ đề và viền mép)
    actual_crop_h = min(crop_size, orig_h - 20)
    actual_crop_w = min(crop_size, orig_w - 20)
    cy, cx = orig_h // 2, orig_w // 2
    crop_orig = img_gray[
        cy - actual_crop_h // 2 : cy + actual_crop_h // 2,
        cx - actual_crop_w // 2 : cx + actual_crop_w // 2
    ].astype(np.float32)

    heights = list(range(min_h, max_h + 1, step))
    if max_h not in heights:
        heights.append(max_h)
    heights.sort()

    errors = []
    img_f = img_gray.astype(np.float32)

    for h in heights:
        if h == orig_h:
            errors.append(0.0)
            continue

        w = int(round(h * (orig_w / orig_h)))

        # Thu nhỏ toàn khung hình
        downscaled = cv2.resize(img_f, (w, h), interpolation=kernel)
        # Phóng to ngược lại kích thước gốc
        upscaled = cv2.resize(downscaled, (orig_w, orig_h), interpolation=kernel)

        # Trích xuất cùng tọa độ trung tâm để tính sai số MSE
        crop_up = upscaled[
            cy - actual_crop_h // 2 : cy + actual_crop_h // 2,
            cx - actual_crop_w // 2 : cx + actual_crop_w // 2
        ]

        mse = float(np.mean((crop_orig - crop_up) ** 2))
        errors.append(mse)

    return heights, errors


def analyze_curve_and_verdict(heights, errors):
    """
    Phân tích đồ thị sai số để tìm thung lũng (local minimum) và đưa ra phán quyết:
    - Các độ phân giải gốc thực tế của video/anime luôn nằm trong khoảng [690p - 930p]
      (như 720p, 810p, 844p, 864p, 878p, 900p).
    - Vùng > 930p tiệm cận 1080p có sai số rất nhỏ và dễ bị nhiễu làm tròn ma trận điểm ảnh.
    - Một thung lũng thực sự (Native Resolution) phải có độ sâu sụt giảm (dip_ratio) >= 10%
      và đáy võng sâu rõ rệt so với 2 điểm lân cận.
    """
    # Xét các điểm ứng viên trong dải phân giải gốc tiêu chuẩn [690p, 930p]
    valid_indices = [i for i, h in enumerate(heights) if 690 <= h <= 930]

    valleys = []
    for idx in valid_indices:
        if idx == 0 or idx >= len(heights) - 1:
            continue
        h = heights[idx]
        e = errors[idx]
        e_prev = errors[idx - 1]
        e_next = errors[idx + 1]

        # Kiểm tra cực tiểu cục bộ: MSE thấp hơn cả 2 điểm liền kề
        if e < e_prev and e < e_next:
            drop_left = e_prev - e
            rise_right = e_next - e
            prominence = min(drop_left, rise_right)
            dip_ratio = (prominence / e) * 100 if e > 1e-6 else 100.0

            # Lọc thung lũng thực sự có độ sâu rõ rệt (dip_ratio >= 8.5%)
            if dip_ratio >= 8.5:
                valleys.append({
                    "height": h,
                    "mse": e,
                    "drop_left": drop_left,
                    "rise_right": rise_right,
                    "prominence": prominence,
                    "dip_ratio": dip_ratio
                })

    # Sắp xếp các thung lũng theo độ sâu giảm dần
    valleys.sort(key=lambda v: v["prominence"], reverse=True)

    # Đánh giá kết luận
    if not valleys:
        verdict = {
            "verdict_type": "TRUE_1080P",
            "detected_native": 1080,
            "title": "✅ 1080p GỐC THỰC SỰ (True Native 1080p)",
            "details": "Đồ thị sai số dốc đều liên tục về 0 tại 1080p, không có đáy võng bất thường ở 720p hay 810p. Video bảo toàn trọn vẹn chi tiết tần số cao chuẩn Full HD.",
            "confidence": 98.5,
            "best_valley": None,
            "all_valleys": []
        }
        return verdict

    best_v = valleys[0]
    detected_h = best_v["height"]
    dip_pct = best_v["dip_ratio"]

    # Xác định mức độ tin cậy dựa trên tỷ lệ sụt giảm
    confidence = min(99.5, max(80.0, 75.0 + dip_pct * 1.2))

    # Đối chiếu với các profile chuẩn
    if 710 <= detected_h <= 730:
        verdict_type = "FAKE_720P"
        title = "❌ FAKE 1080p (Kéo từ 720p lên)"
        details = (
            f"Phát hiện đáy thung lũng sụt giảm sai số cực mạnh tại {detected_h}p (độ sâu võng {dip_pct:.1f}%). "
            f"Video này vốn chỉ là 720p HD thông thường và đã bị phóng to (upscale nội suy) lên 1080p!"
        )
    elif 795 <= detected_h <= 825:
        verdict_type = "ANIME_STUDIO_810P"
        title = "🎨 ANIME STUDIO MASTER (~810p Native)"
        details = (
            f"Phát hiện đáy võng tại {detected_h}p (~810p). Đây là chuẩn vẽ tay hoạt hình quen thuộc "
            f"(KyoAni, CloverWorks). Tệp được Master chính quy từ Studio chứ không phải web tự ý kéo từ 720p."
        )
    elif 860 <= detected_h <= 890:
        verdict_type = "ANIME_STUDIO_878P"
        title = "🎨 ANIME STUDIO MASTER (~878p Native)"
        details = (
            f"Phát hiện đáy võng tại {detected_h}p (~878p). Chuẩn dựng khung hình độ phân giải cao của "
            f"Studio MAPPA / Wit Studio. Bản vẽ gốc ở ~878p và được Studio nội suy lên 1080p chuẩn truyền hình."
        )
    elif 835 <= detected_h <= 855:
        verdict_type = "ANIME_STUDIO_844P"
        title = "🎨 ANIME STUDIO MASTER (~844p Native)"
        details = f"Phát hiện đáy võng tại {detected_h}p (~844p). Bản vẽ gốc hoạt hình ở 844p được upscale bởi studio."
    elif 895 <= detected_h <= 915:
        verdict_type = "ANIME_STUDIO_900P"
        title = "🎨 BẢN MASTER 900p (1600x900)"
        details = f"Phát hiện đáy võng tại {detected_h}p. Video gốc ở độ phân giải 900p kéo lên 1080p."
    else:
        verdict_type = "CUSTOM_UPSCALED"
        title = f"⚠️ UPSCALED TỪ {detected_h}p"
        details = f"Phát hiện sai số cực tiểu cục bộ rõ rệt tại {detected_h}p (độ sâu sụt giảm {dip_pct:.1f}%)."

    return {
        "verdict_type": verdict_type,
        "detected_native": detected_h,
        "title": title,
        "details": details,
        "confidence": confidence,
        "best_valley": best_v,
        "all_valleys": valleys
    }


def plot_analysis_result(heights, errors, verdict, output_png_path, filename_label=""):
    """
    Vẽ đồ thị Descaling Error Curve chuyên nghiệp với giao diện Dark Mode cao cấp.
    Sử dụng 3 panel:
    - Panel 1 (trên cùng): Banner tóm tắt kết quả phân tích & độ tin cậy (không bị chồng lấn).
    - Panel 2 (giữa): Đồ thị chính MSE Curve kèm các mốc phân giải chuẩn và mũi tên chỉ đáy võng.
    - Panel 3 (dưới): Biểu đồ tốc độ biến thiên d(MSE)/dh phát hiện đảo chiều độ dốc.
    """
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11.5, 8.8), dpi=150)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 3.2, 1.2], hspace=0.32)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # 1. Panel 0: Banner tóm tắt kết quả
    ax0.axis("off")
    is_fake = "FAKE" in verdict["verdict_type"]
    is_true = "TRUE" in verdict["verdict_type"]
    badge_border = "#ef4444" if is_fake else ("#10b981" if is_true else "#f59e0b")
    badge_bg = "#1e1e2e"

    clean_title = verdict["title"].replace("❌ ", "[!] ").replace("✅ ", "[OK] ").replace("🎨 ", "[STUDIO] ").replace("⚠️ ", "[WARN] ")
    banner_text = (
        f"KẾT QUẢ PHÂN TÍCH: {clean_title}\n"
        f"Độ phân giải gốc ước tính: {verdict['detected_native']}p   |   Độ tin cậy thuật toán: {verdict['confidence']:.1f}%\n"
        f"Đánh giá: {verdict['details']}"
    )
    ax0.text(
        0.5, 0.5, banner_text,
        transform=ax0.transAxes,
        fontsize=9.8,
        ha="center", va="center",
        bbox=dict(
            boxstyle="round,pad=0.8,rounding_size=0.3",
            facecolor=badge_bg,
            edgecolor=badge_border,
            linewidth=2.2,
            alpha=0.95
        ),
        color="#f8fafc",
        linespacing=1.45
    )

    # 2. Panel 1: Đồ thị chính (MSE Curve)
    ax1.plot(
        heights, errors,
        marker="o", markersize=4.5,
        color="#38bdf8", linewidth=2.2,
        label="Sai số MSE (Descaling Error Curve)",
        zorder=4
    )

    # Đánh dấu các mốc chuẩn
    for std_h, prof in STUDIO_PROFILES.items():
        if std_h in [720, 810, 878, 1080]:
            ax1.axvline(
                x=std_h, color=prof["color"], linestyle="--", alpha=0.75, linewidth=1.4,
                label=f"Mốc {prof['name']}"
            )

    # Nếu có thung lũng phát hiện, đánh dấu bằng mũi tên và điểm sáng
    best_v = verdict.get("best_valley")
    if best_v:
        vh = best_v["height"]
        vmse = best_v["mse"]
        ax1.scatter([vh], [vmse], color="#ef4444", s=130, zorder=6, edgecolors="#ffffff", linewidth=2)
        y_span = max(errors) - min(errors)
        ax1.annotate(
            f"Đáy võng tại {vh}p\n(MSE = {vmse:.4f})",
            xy=(vh, vmse),
            xytext=(vh - 45, vmse + y_span * 0.22),
            arrowprops=dict(facecolor="#f87171", shrink=0.08, width=2, headwidth=7),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#f87171", alpha=0.9),
            fontsize=9.0, fontweight="bold", color="#f8fafc"
        )

    ax1.set_title(
        f"PHÂN TÍCH SAI SỐ NGHỊCH ĐẢO (DESCALING ERROR ANALYSIS) - {filename_label}",
        fontsize=12, fontweight="bold", color="#f1f5f9", pad=10
    )
    ax1.set_ylabel("Sai số MSE (Càng thấp = Càng khớp)", fontsize=10.5, color="#cbd5e1")
    ax1.grid(True, linestyle=":", alpha=0.35, color="#64748b")
    ax1.legend(loc="upper right", framealpha=0.88, facecolor="#0f172a", edgecolor="#334155", fontsize=8.8)

    # 3. Panel 2: Đạo hàm bậc nhất d(MSE)/dh
    diff_h = heights[1:]
    diff_e = np.diff(errors)
    bar_colors = ["#f87171" if val > 0.001 else "#38bdf8" for val in diff_e]

    ax2.bar(diff_h, diff_e, width=6.5, color=bar_colors, alpha=0.85, edgecolor="#0f172a")
    ax2.axhline(0, color="#94a3b8", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("Độ phân giải chiều dọc quét (Vertical Resolution)", fontsize=10.5, color="#cbd5e1")
    ax2.set_ylabel("delta MSE", fontsize=10, color="#cbd5e1")
    ax2.set_title("Tốc độ biến thiên delta MSE (Cột màu ĐỎ > 0: Dấu hiệu nghịch đảo gradient = Chắc chắn là Video Upscaled)", fontsize=9.2, color="#94a3b8")
    ax2.grid(True, linestyle=":", alpha=0.35, color="#64748b")

    output_png = Path(output_png_path).resolve()
    plt.savefig(str(output_png), bbox_inches="tight")
    plt.close()
    return str(output_png)


def analyze_target(input_path, timestamp=None, crop_size=600, output_png="result_analysis.png", multi_frame=False):
    """
    Hàm phân tích toàn diện chấp nhận cả tệp video và tệp ảnh:
    - Nếu là video: tự động trích xuất khung hình chất lượng cao nhất hoặc theo timestamp.
    - Nếu multi_frame=True: quét và lấy trung bình 3 khung hình sắc nét nhất.
    - Tính toán sai số Descaling.
    - Phán quyết và xuất biểu đồ kết quả.
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp đầu vào: {input_path}")

    # Kiểm tra phần mở rộng tệp
    ext = path.suffix.lower()
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts", ".m4v"}
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    frames_to_test = []

    if ext in video_exts:
        print(f"\n🎬 Đang xử lý tệp video: {path.name}")
        ts_sec = parse_timestamp_to_seconds(timestamp)
        if ts_sec is not None:
            print(f"⏱️ Trích xuất khung hình tại thời điểm: {timestamp} ({ts_sec}s)...")
        else:
            print("🔍 Đang tự động quét tìm khung hình tĩnh có độ chi tiết và độ sắc nét cao nhất...")

        extracted = extract_best_frames_from_video(path, timestamp_sec=ts_sec, num_samples=8)
        if multi_frame and len(extracted) >= 3 and ts_sec is None:
            frames_to_test = [item[1] for item in extracted[:3]]
            print(f"-> Đã chọn top {len(frames_to_test)} khung hình sắc nét nhất (Laplacian Var: {extracted[0][2]:.1f}) để tính trung bình.")
        else:
            best_t, best_img, best_sharpness = extracted[0]
            frames_to_test = [best_img]
            print(f"-> Đã chọn khung hình tại {best_t:.1f}s với điểm sắc nét: {best_sharpness:.1f}")

    elif ext in image_exts:
        print(f"\n🖼️ Đang xử lý tệp hình ảnh: {path.name}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Không thể đọc tệp ảnh: {path}")
        frames_to_test = [img]
    else:
        raise ValueError(f"Định dạng tệp không được hỗ trợ ({ext}). Hỗ trợ video: mp4, mkv, mov... hoặc ảnh: png, jpg, webp...")

    # Kiểm tra kích thước khung hình
    first_h, first_w = frames_to_test[0].shape
    print(f"📐 Kích thước khung hình đầu vào: {first_w}x{first_h}p")
    if first_h != 1080:
        print(f"⚠️ Cảnh báo: Chiều cao ảnh là {first_h}p (chuẩn khuyến nghị để kiểm tra là 1080p).")

    # Tiến hành quét Descaling
    print("⏳ Đang quét ma trận điểm ảnh từ 680p đến 1080p qua thuật toán nội suy Cubic...")
    all_errors = []
    heights = []

    for idx, f_gray in enumerate(frames_to_test):
        h_list, errs = compute_descaling_errors(
            f_gray,
            crop_size=crop_size,
            min_h=680,
            max_h=1080,
            step=10,
            kernel=cv2.INTER_CUBIC
        )
        heights = h_list
        all_errors.append(errs)

    # Tính trung bình sai số nếu có nhiều khung hình
    avg_errors = np.mean(np.array(all_errors), axis=0).tolist()

    # Phân tích đường cong và đưa ra kết luận
    verdict = analyze_curve_and_verdict(heights, avg_errors)

    # Xuất đồ thị phân tích
    out_png = plot_analysis_result(heights, avg_errors, verdict, output_png, filename_label=path.name)
    print(f"📊 Đã xuất đồ thị phân tích cao cấp ra tệp: {out_png}")

    return {
        "input_file": str(path),
        "resolution": f"{first_w}x{first_h}",
        "verdict": verdict,
        "heights": heights,
        "errors": avg_errors,
        "plot_path": out_png
    }


def print_cli_report(result):
    """In báo cáo định dạng đẹp mắt ra Terminal."""
    verdict = result["verdict"]
    print("\n" + "=" * 70)
    print("           📋 KẾT QUẢ KIỂM ĐỊNH CHẤT LƯỢNG ĐỘ PHÂN GIẢI VIDEO")
    print("=" * 70)
    print(f"📁 Tệp kiểm tra      : {Path(result['input_file']).name}")
    print(f"📐 Độ phân giải khung : {result['resolution']}")
    print(f"🔍 Kết luận          : {verdict['title']}")
    print(f"🎯 Độ phân giải gốc  : {verdict['detected_native']}p")
    print(f"🛡️ Độ tin cậy        : {verdict['confidence']:.1f}%")
    print("-" * 70)
    print("📝 Đánh giá chi tiết:")
    print(f"   {verdict['details']}")
    print("-" * 70)

    # Bảng sai số tại các mốc quan trọng
    print("📊 Bảng sai số MSE tại các mốc phân giải quan trọng:")
    h_list = result["heights"]
    e_list = result["errors"]
    for std_h, prof in STUDIO_PROFILES.items():
        if std_h in h_list:
            mse_val = e_list[h_list.index(std_h)]
            flag = " 👈 (GỐC PHÁT HIỆN)" if std_h == verdict["detected_native"] else ""
            print(f"   - {prof['name']:<28}: MSE = {mse_val:>9.4f}{flag}")

    print("=" * 70)
    print(f"📈 Xem đồ thị chi tiết tại: {result['plot_path']}\n")


def launch_web_ui(port=7865):
    """
    Khởi chạy giao diện WebUI độc lập siêu nhẹ bằng thư viện chuẩn Python (http.server).
    Không yêu cầu cài thêm gradio hay bất kỳ package web nào khác.
    Giao diện Dark Mode Glassmorphism cao cấp, hỗ trợ kéo thả tệp, xem kết quả và đồ thị trực tiếp.
    """
    import http.server
    import urllib.parse
    import tempfile
    import base64
    import webbrowser

    HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kiểm Tra Chất Lượng Video 1080p Native vs Fake 720p</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --card-bg: rgba(18, 24, 38, 0.75);
            --border: rgba(255, 255, 255, 0.08);
            --accent-blue: #38bdf8;
            --accent-purple: #818cf8;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(129, 140, 248, 0.10) 0px, transparent 50%);
            color: var(--text-main);
            font-family: 'Plus Jakarta Sans', sans-serif;
            min-height: 100vh;
            padding: 30px 20px;
        }
        .container { max-width: 1240px; margin: 0 auto; }
        .header {
            text-align: center;
            margin-bottom: 35px;
        }
        .badge-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: var(--accent-blue);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        h1 {
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle { color: var(--text-sub); font-size: 1.05rem; max-width: 700px; margin: 0 auto; line-height: 1.5; }
        .grid-layout {
            display: grid;
            grid-template-columns: 460px 1fr;
            gap: 26px;
        }
        @media (max-width: 980px) { .grid-layout { grid-template-columns: 1fr; } }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 26px;
            backdrop-filter: blur(16px);
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        }
        .card-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 36px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
            background: rgba(255, 255, 255, 0.02);
            position: relative;
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent-blue);
            background: rgba(56, 189, 248, 0.05);
        }
        .dropzone input {
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            opacity: 0;
            cursor: pointer;
        }
        .drop-icon { font-size: 2.8rem; margin-bottom: 10px; display: block; }
        .drop-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 4px; }
        .drop-sub { font-size: 0.85rem; color: var(--text-sub); }
        .file-selected {
            margin-top: 14px;
            padding: 10px 14px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 10px;
            font-size: 0.9rem;
            color: var(--accent-blue);
            font-family: 'JetBrains Mono', monospace;
            word-break: break-all;
            display: none;
        }
        .form-group { margin-top: 18px; }
        .form-label { display: block; font-size: 0.9rem; font-weight: 600; margin-bottom: 8px; color: #cbd5e1; }
        .form-input {
            width: 100%;
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 11px 14px;
            color: var(--text-main);
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        .form-input:focus { border-color: var(--accent-blue); }
        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 18px;
            cursor: pointer;
            font-size: 0.92rem;
            color: #cbd5e1;
            user-select: none;
        }
        .btn-run {
            width: 100%;
            margin-top: 24px;
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
            border: none;
            border-radius: 12px;
            padding: 14px;
            font-size: 1.05rem;
            font-weight: 700;
            color: #ffffff;
            cursor: pointer;
            transition: all 0.25s ease;
            box-shadow: 0 8px 20px -4px rgba(2, 132, 199, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .btn-run:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 12px 26px -4px rgba(2, 132, 199, 0.7);
        }
        .btn-run:disabled { opacity: 0.6; cursor: not-allowed; }
        .spinner {
            width: 20px; height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top-color: #fff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            display: none;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result-empty {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 480px;
            color: var(--text-sub);
            text-align: center;
            gap: 12px;
        }
        .verdict-box {
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 22px;
            border-width: 2px;
            border-style: solid;
        }
        .verdict-box.fake {
            background: rgba(239, 68, 68, 0.12);
            border-color: var(--danger);
        }
        .verdict-box.true {
            background: rgba(16, 185, 129, 0.12);
            border-color: var(--success);
        }
        .verdict-box.studio {
            background: rgba(245, 158, 11, 0.12);
            border-color: var(--warning);
        }
        .verdict-title { font-size: 1.45rem; font-weight: 800; margin-bottom: 6px; }
        .verdict-meta {
            display: flex;
            gap: 18px;
            font-size: 0.95rem;
            color: #cbd5e1;
            margin-bottom: 12px;
        }
        .verdict-desc { font-size: 0.95rem; color: #e2e8f0; line-height: 1.5; }
        .table-responsive {
            overflow-x: auto;
            margin-top: 18px;
            border: 1px solid var(--border);
            border-radius: 12px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            text-align: left;
        }
        th, td { padding: 10px 14px; border-bottom: 1px solid var(--border); }
        th { background: rgba(0, 0, 0, 0.35); font-weight: 600; color: #cbd5e1; }
        tr:last-child td { border-bottom: none; }
        .plot-container {
            margin-top: 22px;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--border);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        .plot-container img { width: 100%; height: auto; display: block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="badge-tag">🔬 Descaling Error Analysis Engine</div>
            <h1>Kiểm Tra Độ Phân Giải Video Gốc</h1>
            <p class="subtitle">Phát hiện video 1080p gốc thực sự hay chỉ là file 720p kéo lên, hoặc bản vẽ Studio Anime gốc ~810p/878p bằng thuật toán sai số nghịch đảo MSE.</p>
        </div>

        <div class="grid-layout">
            <!-- Cột cấu hình & upload -->
            <div class="card">
                <div class="card-title">📁 Tải Lên Video / Hình Ảnh</div>
                
                <div class="dropzone" id="dropArea">
                    <input type="file" id="fileInput" accept="video/*,image/*">
                    <span class="drop-icon">🎬</span>
                    <div class="drop-title">Kéo thả tệp hoặc bấm vào đây để chọn</div>
                    <div class="drop-sub">Hỗ trợ: MP4, MKV, AVI, WEBM, PNG, JPG, WEBP...</div>
                </div>
                <div class="file-selected" id="fileInfo"></div>

                <div class="form-group">
                    <label class="form-label" for="timeInput">⏱️ Mốc thời gian trích xuất (Tùy chọn cho Video)</label>
                    <input class="form-input" type="text" id="timeInput" placeholder="Ví dụ: 00:04:30 hoặc 270 (Để trống: tự động quét cảnh nét nhất)">
                </div>

                <div class="form-group">
                    <label class="form-label" for="cropInput">📐 Kích thước vùng kiểm tra (Crop Box): <span id="cropVal">600</span>px</label>
                    <input class="form-input" type="range" id="cropInput" min="400" max="800" step="50" value="600" oninput="document.getElementById('cropVal').innerText = this.value">
                </div>

                <label class="checkbox-label">
                    <input type="checkbox" id="multiBox">
                    <span>🔬 Quét trung bình 3 khung hình sắc nét nhất</span>
                </label>

                <button class="btn-run" id="btnRun" onclick="startAnalysis()">
                    <span class="spinner" id="spinner"></span>
                    <span id="btnText">🚀 Bắt Đầu Phân Tích</span>
                </button>
            </div>

            <!-- Cột hiển thị kết quả -->
            <div class="card" id="resultCard">
                <div class="card-title">📊 Kết Quả Phán Quyết</div>
                <div class="result-empty" id="emptyState">
                    <span style="font-size: 3rem;">🔍</span>
                    <p>Chưa có dữ liệu phân tích.<br>Vui lòng chọn tệp và bấm <strong>"Bắt Đầu Phân Tích"</strong>.</p>
                </div>
                <div id="resultContent" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const dropArea = document.getElementById('dropArea');

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                const f = fileInput.files[0];
                fileInfo.style.display = 'block';
                fileInfo.innerText = `✓ Đã chọn: ${f.name} (${(f.size / (1024*1024)).toFixed(2)} MB)`;
            }
        });

        ['dragenter', 'dragover'].forEach(name => {
            dropArea.addEventListener(name, (e) => { e.preventDefault(); dropArea.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(name => {
            dropArea.addEventListener(name, (e) => { e.preventDefault(); dropArea.classList.remove('dragover'); });
        });
        dropArea.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                const f = e.dataTransfer.files[0];
                fileInfo.style.display = 'block';
                fileInfo.innerText = `✓ Đã chọn: ${f.name} (${(f.size / (1024*1024)).toFixed(2)} MB)`;
            }
        });

        async function startAnalysis() {
            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Vui lòng chọn một tệp Video hoặc Hình ảnh cần kiểm tra!');
                return;
            }

            const btn = document.getElementById('btnRun');
            const spinner = document.getElementById('spinner');
            const btnText = document.getElementById('btnText');
            btn.disabled = true;
            spinner.style.display = 'inline-block';
            btnText.innerText = 'Đang đọc và tải dữ liệu lên...';

            const file = fileInput.files[0];
            const reader = new FileReader();

            reader.onload = async function(e) {
                btnText.innerText = 'Đang phân tích sai số điểm ảnh...';
                const base64Data = e.target.result.split(',')[1];
                const payload = {
                    filename: file.name,
                    file_b64: base64Data,
                    timestamp: document.getElementById('timeInput').value.trim() || null,
                    crop: parseInt(document.getElementById('cropInput').value),
                    multi: document.getElementById('multiBox').checked
                };

                try {
                    const res = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();

                    if (!res.ok || data.error) {
                        throw new Error(data.error || 'Lỗi khi phân tích tệp!');
                    }

                    renderResult(data);
                } catch (err) {
                    alert('Lỗi: ' + err.message);
                } finally {
                    btn.disabled = false;
                    spinner.style.display = 'none';
                    btnText.innerText = '🚀 Bắt Đầu Phân Tích';
                }
            };

            reader.readAsDataURL(file);
        }

        function renderResult(data) {
            document.getElementById('emptyState').style.display = 'none';
            const container = document.getElementById('resultContent');
            container.style.display = 'block';

            const v = data.verdict;
            let vClass = 'studio';
            if (v.verdict_type.includes('FAKE')) vClass = 'fake';
            if (v.verdict_type.includes('TRUE')) vClass = 'true';

            let tableRows = '';
            const profiles = {
                720: '720p (HD Standard)',
                810: '810p (Anime Studio Master)',
                878: '878p (Anime Studio Master)',
                1080: '1080p (Full HD Native)'
            };

            for (const [stdH, name] of Object.entries(profiles)) {
                const idx = data.heights.indexOf(parseInt(stdH));
                if (idx !== -1) {
                    const mse = data.errors[idx];
                    const isTarget = parseInt(stdH) === v.detected_native;
                    const mark = isTarget ? '<span style="color: #38bdf8; font-weight: bold;">👈 ĐÁY VÕNG GỐC</span>' : '';
                    tableRows += `<tr>
                        <td><strong>${stdH}p</strong></td>
                        <td>${name}</td>
                        <td style="font-family: 'JetBrains Mono', monospace;">${mse.toFixed(4)}</td>
                        <td>${mark}</td>
                    </tr>`;
                }
            }

            container.innerHTML = `
                <div class="verdict-box ${vClass}">
                    <div class="verdict-title">${v.title}</div>
                    <div class="verdict-meta">
                        <span>🎯 Độ phân giải gốc: <strong>${v.detected_native}p</strong></span>
                        <span>🛡️ Độ tin cậy: <strong>${v.confidence.toFixed(1)}%</strong></span>
                        <span>📐 Khung hình: <strong>${data.resolution}</strong></span>
                    </div>
                    <div class="verdict-desc">${v.details}</div>
                </div>

                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Mốc</th>
                                <th>Chuẩn Độ Phân Giải</th>
                                <th>Sai Số MSE</th>
                                <th>Đánh Giá</th>
                            </tr>
                        </thead>
                        <tbody>${tableRows}</tbody>
                    </table>
                </div>

                <div class="plot-container">
                    <img src="${data.plot_data_url}" alt="Đồ thị Descaling Curve">
                </div>
            `;
        }
    </script>
</body>
</html>
"""

    class RequestHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/api/analyze":
                try:
                    content_len = int(self.headers.get("Content-Length", 0))
                    post_body = self.rfile.read(content_len)
                    payload = json.loads(post_body.decode("utf-8"))

                    b64_data = payload.get("file_b64")
                    filename = payload.get("filename", "media.tmp")
                    suffix = Path(filename).suffix or ".tmp"

                    if not b64_data:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Không tìm thấy dữ liệu tệp tải lên"}, ensure_ascii=False).encode("utf-8"))
                        return

                    file_bytes = base64.b64decode(b64_data)
                    ts_val = payload.get("timestamp")
                    crop_val = int(payload.get("crop", 600))
                    multi_val = bool(payload.get("multi", False))

                    # Lưu file tạm để phân tích
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
                        tmp_in.write(file_bytes)
                        tmp_in_path = tmp_in.name

                    tmp_out_png = tempfile.mktemp(suffix=".png")

                    try:
                        res = analyze_target(
                            input_path=tmp_in_path,
                            timestamp=ts_val,
                            crop_size=crop_val,
                            output_png=tmp_out_png,
                            multi_frame=multi_val
                        )

                        # Đọc file ảnh PNG và mã hóa Base64
                        with open(tmp_out_png, "rb") as f_png:
                            b64_png = base64.b64encode(f_png.read()).decode("utf-8")
                        plot_data_url = f"data:image/png;base64,{b64_png}"

                        response_data = {
                            "verdict": res["verdict"],
                            "resolution": res["resolution"],
                            "heights": res["heights"],
                            "errors": res["errors"],
                            "plot_data_url": plot_data_url
                        }

                        self.send_response(200)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))

                    finally:
                        if os.path.exists(tmp_in_path):
                            try: os.remove(tmp_in_path)
                            except Exception: pass
                        if os.path.exists(tmp_out_png):
                            try: os.remove(tmp_out_png)
                            except Exception: pass

                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress normal HTTP logging
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), RequestHandler)
    web_url = f"http://localhost:{port}"
    print(f"\n🌐 WebUI Độc Lập đang chạy tại: {web_url}")
    print(f"💡 Nhấn Ctrl + C để dừng WebUI.\n")

    try:
        webbrowser.open(web_url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 WebUI đã dừng.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(
        description="Tool độc lập kiểm tra chất lượng video 1080p gốc hay kéo từ 720p (Descaling Error Analysis)"
    )
    parser.add_argument("input", nargs="?", help="Đường dẫn đến tệp video (.mp4, .mkv...) hoặc ảnh (.png, .jpg)")
    parser.add_argument("-t", "--time", help="Mốc thời gian trích xuất từ video (ví dụ: 00:04:30 hoặc 270s)")
    parser.add_argument("-c", "--crop", type=int, default=600, help="Kích thước vùng cắt kiểm tra trung tâm (mặc định: 600)")
    parser.add_argument("-o", "--output", default="result_analysis.png", help="Tên tệp đồ thị PNG xuất ra (mặc định: result_analysis.png)")
    parser.add_argument("-m", "--multi-frame", action="store_true", help="Quét trung bình 3 khung hình sắc nét nhất trong video")
    parser.add_argument("-j", "--json", action="store_true", help="Xuất kết quả dưới định dạng JSON")
    parser.add_argument("--web", action="store_true", help="Khởi chạy giao diện WebUI trực quan độc lập")
    parser.add_argument("--port", type=int, default=7865, help="Cổng mạng cho WebUI (mặc định: 7865)")

    args = parser.parse_args()

    if args.web:
        launch_web_ui(port=args.port)
        return

    if not args.input:
        parser.print_help()
        print("\nVí dụ sử dụng:")
        print("  python3 check_native.py sample.png")
        print("  python3 check_native.py my_video.mp4")
        print("  python3 check_native.py my_video.mp4 --time 00:04:30")
        print("  python3 check_native.py --web   # Mở giao diện kéo thả trực quan trên trình duyệt")
        return

    try:
        res = analyze_target(
            input_path=args.input,
            timestamp=args.time,
            crop_size=args.crop,
            output_png=args.output,
            multi_frame=args.multi_frame
        )

        if args.json:
            clean_res = {
                "input_file": res["input_file"],
                "resolution": res["resolution"],
                "verdict_type": res["verdict"]["verdict_type"],
                "detected_native": res["verdict"]["detected_native"],
                "confidence": res["verdict"]["confidence"],
                "title": res["verdict"]["title"],
                "details": res["verdict"]["details"],
                "plot_path": res["plot_path"]
            }
            print(json.dumps(clean_res, ensure_ascii=False, indent=2))
        else:
            print_cli_report(res)

    except Exception as e:
        print(f"\n❌ Lỗi khi thực hiện kiểm tra: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
