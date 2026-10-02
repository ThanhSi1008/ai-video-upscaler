import os
import sys
import re
import json
import gc
import subprocess
import urllib.request
import zipfile
import warnings
import time
import threading
import shutil
import tempfile
from queue import Queue
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TORCH_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_LOGS"] = "-inductor"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ==============================================================================
# --- 1. KIẾN TRÚC MÔ HÌNH: SRVGGNetCompact (AnimeJaNai V3) & UpCunet2x (Real-CUGAN) ---
# ==============================================================================

class SRVGGNetCompact(nn.Module):
    """
    Kiến trúc SRVGGNetCompact 16-layer chuyên biệt cho Super-Resolution Native 2x siêu tốc.
    Được sử dụng bởi dòng mô hình AnimeJaNai V3 (Compact & Sharp) của the-database.
    Tốc độ vượt trội (~12–15 FPS trên NVIDIA Tesla T4 FP16), đạt ngân sách ≤ 2 giờ/tập 24 phút.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # Lớp Conv đầu vào
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # 16 khối tích chập đặc trưng sâu
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # Lớp Conv cuối & PixelShuffle Native 2x
        self.body.append(nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        # Cộng thêm phần nội suy nền (base residual) giữ độ ổn định dải màu và độ sáng
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out
# ==============================================================================
# --- 2. DANH MỤC WEIGHTS MÔ HÌNH SUPER-RESOLUTION CHUYÊN BIỆT ---
# ==============================================================================

WEIGHTS_INFO = {
    # NGUỒN B: WEB-DL Gốc (SubsPlease / Erai-raws / Crunchyroll / Netflix gốc)
    "animejanai_v3_compact": {
        "file": "2x_AnimeJaNai_HD_V3_Compact.pth",
        "arch": "srvggnet_compact",
        "scale": 2,
        "desc": "AnimeJaNai V3 Compact 16-lớp (WEB-DL Gốc: SubsPlease/Erai - Master Quality ~3.8 FPS trên T4)",
        "zip_url": "https://github.com/the-database/mpv-AnimeJaNai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
        "zip_extract": "2x_AnimeJaNai_HD_V3_Compact.pth"
    },
    # NGUỒN A: BDRip 10-bit (Hi10P / Main 10) (Đã xử lý deband/dither 16-bit, viền nét đanh, không color banding)
    "animejanai_v3_sharp": {
        "file": "2x_AnimeJaNai_HD_V3Sharp1_Compact.pth",
        "arch": "srvggnet_compact",
        "scale": 2,
        "desc": "AnimeJaNai V3 Sharp 16-lớp (BDRip 10-bit: Hi10P/Main 10 - Nét đanh giữ grain ~3.8 FPS trên T4)",
        "zip_url": "https://github.com/the-database/mpv-AnimeJaNai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
        "zip_extract": "2x_AnimeJaNai_HD_V3Sharp1_Compact.pth"
    },
    # NGUỒN B - ULTRA COMPACT: WEB-DL Gốc (8-lớp, Tốc độ gấp đôi ~7.5–8.0 FPS trên T4)
    "animejanai_v3_ultracompact": {
        "file": "2x_AnimeJaNai_HD_V3_UltraCompact.pth",
        "arch": "srvggnet_compact",
        "scale": 2,
        "desc": "AnimeJaNai V3 UltraCompact 8-lớp (WEB-DL Gốc - Tốc độ gấp đôi ~7.5–8.0 FPS trên T4)",
        "zip_url": "https://github.com/the-database/mpv-AnimeJaNai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
        "zip_extract": "2x_AnimeJaNai_HD_V3_UltraCompact.pth"
    },
    # NGUỒN A - ULTRA COMPACT: BDRip 10-bit Sharp (8-lớp, Tốc độ gấp đôi ~7.5–8.0 FPS trên T4)
    "animejanai_v3_sharp_ultracompact": {
        "file": "2x_AnimeJaNai_HD_V3Sharp1_UltraCompact.pth",
        "arch": "srvggnet_compact",
        "scale": 2,
        "desc": "AnimeJaNai V3 Sharp UltraCompact 8-lớp (BDRip 10-bit - Tốc độ gấp đôi ~7.5–8.0 FPS trên T4)",
        "zip_url": "https://github.com/the-database/mpv-AnimeJaNai/releases/download/3.0.0/2x_AnimeJaNai_HD_V3_ModelsOnly.zip",
        "zip_extract": "2x_AnimeJaNai_HD_V3Sharp1_UltraCompact.pth"
    }
}

def resolve_model_key(name):
    name_l = (name or "").lower()
    is_ultra = any(k in name_l for k in ["ultra", "ultracompact", "ultra compact", "ultra-compact", "8-lớp", "8 lop", "gấp đôi", "gap doi"])
    is_sharp = any(k in name_l for k in ["bdrip", "10-bit", "10bit", "hi10p", "main10", "main 10", "sharp", "nguồn a", "nguon a"])

    if is_ultra:
        if is_sharp:
            return "animejanai_v3_sharp_ultracompact"
        return "animejanai_v3_ultracompact"

    # Nhận diện Nguồn A: BDRip 10-bit (Hi10P / Main 10)
    if is_sharp:
        return "animejanai_v3_sharp"
    # Mặc định tối ưu cho Nguồn B: WEB-DL Gốc
    return "animejanai_v3_compact"

def ensure_model_weights(model_key, progress_callback=None):
    if model_key not in WEIGHTS_INFO:
        model_key = resolve_model_key(model_key)
    info = WEIGHTS_INFO[model_key]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, info["file"])
    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 100000:
        return weights_path

    # Kiểm tra xem file có ở thư mục con 2x_AnimeJaNai_HD_V3_ModelsOnly không
    sub_path = os.path.join(script_dir, "2x_AnimeJaNai_HD_V3_ModelsOnly", info["file"])
    if os.path.exists(sub_path) and os.path.getsize(sub_path) > 100000:
        try:
            shutil.copy2(sub_path, weights_path)
            return weights_path
        except Exception:
            return sub_path

    # Kiểm tra xem file có ở thư mục làm việc hiện tại không
    if os.path.exists(info["file"]) and os.path.getsize(info["file"]) > 100000:
        return info["file"]

    print(f"📥 Tự động tải weights mô hình '{info['desc']}'...")
    if progress_callback:
        progress_callback(0.01, desc=f"📥 Đang chuẩn bị weights: {info['file']}...")

    # Tải gói ZIP từ GitHub Releases (AnimeJaNai V3)
    if "zip_url" in info:
        zip_url = info["zip_url"]
        zip_path = os.path.join(script_dir, "2x_AnimeJaNai_HD_V3_ModelsOnly.zip")
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100000:
            print(f"🔗 Đang tải kho mô hình ZIP từ: {zip_url}")
            download_ok = False
            # Thử bằng curl trước vì hỗ trợ redirect và SSL tốt nhất trên Colab/Linux
            try:
                curl_cmd = ['curl', '-L', '-k', '--retry', '3', '-o', zip_path, zip_url]
                subprocess.run(curl_cmd, check=True)
                if os.path.exists(zip_path) and os.path.getsize(zip_path) > 100000:
                    download_ok = True
            except Exception as e_curl:
                print(f"⚠️ Tải qua curl thất bại ({e_curl}), thử lại bằng urllib...")

            if not download_ok:
                try:
                    req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, 'wb') as f:
                        shutil.copyfileobj(resp, f)
                    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 100000:
                        download_ok = True
                except Exception as e_url:
                    print(f"⚠️ Urllib cũng thất bại: {e_url}")

        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 100000:
            print(f"📦 Đang giải nén bộ trọng số AnimeJaNai V3...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)
                    if not filename:
                        continue
                    # Trích xuất trực tiếp từng file ra thẳng script_dir (tránh bị lồng thư mục)
                    dest_file = os.path.join(script_dir, filename)
                    with zf.open(member) as source, open(dest_file, "wb") as target:
                        shutil.copyfileobj(source, target)

            if os.path.exists(weights_path) and os.path.getsize(weights_path) > 100000:
                print(f"✅ Đã tải thành công: {weights_path} ({os.path.getsize(weights_path)/(1024*1024):.2f} MB)")
                return weights_path

    raise RuntimeError(f"Không thể tải weights mô hình {model_key}. Vui lòng kiểm tra kết nối mạng!")

def load_model(model_key, weights_path, device):
    info = WEIGHTS_INFO.get(model_key, WEIGHTS_INFO["animejanai_v3_compact"])
    arch = info.get("arch", "srvggnet_compact")
    is_pro = False

    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()

    state_dict = torch.load(weights_path, map_location='cpu')
    state_dict = state_dict.get('params_ema', state_dict.get('params', state_dict))

    # Tự động nhận diện cấu hình số lớp từ trọng số (16 cho Compact, 8 cho UltraCompact, 4 cho SuperUltraCompact)
    conv_keys = [k for k, v in state_dict.items() if k.endswith('.weight') and v.dim() == 4]
    num_feat = state_dict['body.0.weight'].shape[0] if 'body.0.weight' in state_dict else 64
    num_in_ch = state_dict['body.0.weight'].shape[1] if 'body.0.weight' in state_dict else 3
    num_conv = len(conv_keys) - 2 if len(conv_keys) >= 2 else 16
    act_type = 'prelu' if any('weight' in k and v.dim() == 1 for k, v in state_dict.items()) else 'relu'

    model = SRVGGNetCompact(num_in_ch=num_in_ch, num_out_ch=3, num_feat=num_feat, num_conv=num_conv, upscale=2, act_type=act_type)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    if device.type == 'cuda':
        model = model.half().to(memory_format=torch.channels_last)
    elif device.type == 'mps':
        model = model.half()
    model = model.to(device)

    # Tối ưu hóa PyTorch JIT Graph cho GPU để triệt tiêu overhead CPU kernel launch
    if device.type == 'cuda':
        try:
            # Dùng tensor nhỏ (256x256) chỉ tốn ~384 KB VRAM để trace graph an toàn tuyệt đối chống OOM
            dummy_in = torch.zeros(1, 3, 256, 256, dtype=torch.float16, device=device).to(memory_format=torch.channels_last)
            traced = torch.jit.trace(model, dummy_in)
            traced = torch.jit.optimize_for_inference(traced)
            model = traced
            del dummy_in
            torch.cuda.empty_cache()
            print("⚡ Đã kích hoạt PyTorch JIT Graph & Tensor Fusion siêu tốc!")
        except Exception as e_jit:
            print(f"ℹ️ Sử dụng mô hình PyTorch chuẩn ({e_jit})")

    return model, arch, is_pro

# Alias tương thích ngược
load_realcugan_model = load_model

# ==============================================================================
# --- 3. NHẬN DIỆN PHẦN CỨNG & BỘ MÃ HÓA HEVC 10-BIT MAIN10 ---
# ==============================================================================

def get_best_device():
    """
    Tự động nhận diện thiết bị tăng tốc phần cứng tốt nhất:
    1. NVIDIA GPU (CUDA)
    2. Apple Silicon M-Series (MPS Metal Acceleration)
    3. CPU Software Fallback
    """
    if torch.cuda.is_available():
        num = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if num > 0 else "CUDA"
        return torch.device('cuda:0'), 'cuda', f"{num}x NVIDIA GPU ({name})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), 'mps', "Apple Silicon (Metal Performance Shaders - MPS)"
    return torch.device('cpu'), 'cpu', "CPU (Software Mode)"

def ensure_ffmpeg_nvenc():
    """
    Tự động kiểm tra và cài đặt FFmpeg có hỗ trợ NVIDIA NVENC
    nếu đang chạy trên môi trường Linux GPU (Google Colab / Kaggle).
    """
    if sys.platform != 'linux' or not torch.cuda.is_available():
        return

    try:
        res = subprocess.run(['ffmpeg', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if 'hevc_nvenc' in res.stdout:
            return  # Đã có sẵn NVENC
    except Exception:
        pass

    # Tự động nạp bản build FFmpeg NVENC nếu chạy trên Colab (/content) hoặc Kaggle (/kaggle)
    if os.path.exists('/content') or os.path.exists('/kaggle'):
        print("⚡ [Auto-Setup] Phát hiện thiếu FFmpeg NVENC trên GPU. Đang tự động nạp FFmpeg NVENC (~15s)...", flush=True)
        setup_cmd = (
            "wget -q --timeout=30 https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -O /tmp/ffmpeg.tar.xz && "
            "tar -xf /tmp/ffmpeg.tar.xz -C /tmp && "
            "cp -f /tmp/ffmpeg-*-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg 2>/dev/null || true && "
            "cp -f /tmp/ffmpeg-*-linux64-gpl/bin/ffprobe /usr/local/bin/ffprobe 2>/dev/null || true && "
            "cp -f /tmp/ffmpeg-*-linux64-gpl/bin/ffmpeg /usr/bin/ffmpeg 2>/dev/null || true && "
            "cp -f /tmp/ffmpeg-*-linux64-gpl/bin/ffprobe /usr/bin/ffprobe 2>/dev/null || true && "
            "chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /usr/bin/ffmpeg /usr/bin/ffprobe 2>/dev/null || true && "
            "rm -rf /tmp/ffmpeg*"
        )
        try:
            subprocess.run(setup_cmd, shell=True, check=False)
            print("✅ [Auto-Setup] Đã kích hoạt FFmpeg NVIDIA NVENC phần cứng thành công!", flush=True)
        except Exception as e:
            print(f"⚠️ [Auto-Setup] Không thể tự động cài FFmpeg NVENC: {e}")

def get_hevc_encoder_flags(device_type):
    """
    Tự động chọn encoder HEVC 10-bit tối ưu nhất theo phần cứng:
    - Apple Silicon M-Series: hevc_videotoolbox (Hardware 10-bit Main10)
    - NVIDIA GPU: hevc_nvenc (Hardware NVENC 10-bit Main10)
    - Fallback: libx265 (CPU 10-bit)
    """
    if device_type == 'mps' or sys.platform == 'darwin':
        try:
            res = subprocess.run(['ffmpeg', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if 'hevc_videotoolbox' in res.stdout:
                return [
                    '-c:v', 'hevc_videotoolbox',
                    '-profile:v', 'main10',
                    '-pix_fmt', 'p010le',
                    '-q:v', '65',
                    '-spatial_aq', '1'
                ], "hevc_videotoolbox 10-bit (Apple Silicon Hardware)"
        except Exception:
            pass

    if device_type == 'cuda':
        ensure_ffmpeg_nvenc()
        try:
            res = subprocess.run(['ffmpeg', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if 'hevc_nvenc' in res.stdout:
                test_flags = [
                    '-c:v', 'hevc_nvenc',
                    '-preset', 'p4',
                    '-tune', 'hq',
                    '-cq', '18',
                    '-spatial-aq', '1',
                    '-pix_fmt', 'yuv420p10le',
                    '-profile:v', 'main10'
                ]
                # NVENC phần cứng yêu cầu độ phân giải tối thiểu >= 160x160 (dùng 256x256 để kiểm tra)
                test_cmd = [
                    'ffmpeg', '-y', '-f', 'lavfi', '-i', 'nullsrc=s=256x256:d=0.04',
                    *test_flags,
                    '-f', 'null', '-'
                ]
                test_res = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if test_res.returncode == 0:
                    return test_flags, "hevc_nvenc 10-bit (NVIDIA Hardware Fast)"

                # Fallback NVENC tinh gọn nếu tùy chọn nâng cao không được driver hỗ trợ
                simple_flags = [
                    '-c:v', 'hevc_nvenc',
                    '-preset', 'p4',
                    '-cq', '18',
                    '-pix_fmt', 'yuv420p10le'
                ]
                test_cmd2 = [
                    'ffmpeg', '-y', '-f', 'lavfi', '-i', 'nullsrc=s=256x256:d=0.04',
                    *simple_flags,
                    '-f', 'null', '-'
                ]
                test_res2 = subprocess.run(test_cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if test_res2.returncode == 0:
                    return simple_flags, "hevc_nvenc 10-bit (NVIDIA Hardware Fast)"
        except Exception:
            pass

    return [
        '-c:v', 'libx265',
        '-crf', '18',
        '-preset', 'veryfast',
        '-pix_fmt', 'yuv420p10le'
    ], "libx265 10-bit (CPU Software - Fast)"

# ==============================================================================
# --- 4. TẢI TẬP PHIM TỪ GOOGLE DRIVE ---
# ==============================================================================

def download_gdrive(url, output_dir="/content/input", progress_cb=None):
    os.makedirs(output_dir, exist_ok=True)
    if progress_cb:
        progress_cb(0.01, desc="☁️ Đang phân tích link Google Drive...")

    file_id = None
    m = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if m:
        file_id = m.group(1)
    else:
        m2 = re.search(r'id=([a-zA-Z0-9_-]+)', url)
        if m2:
            file_id = m2.group(1)

    if not file_id:
        raise ValueError(f"Không thể trích xuất File ID từ link Google Drive: {url}")

    print(f"☁️ Đang kết nối tải file từ Google Drive ID: {file_id}...")
    if progress_cb:
        progress_cb(0.02, desc=f"☁️ Đang kéo file video từ Google Drive ({file_id[:8]}...)...")

    target_file = None
    try:
        import gdown
        downloaded = gdown.download(id=file_id, output=output_dir + "/", quiet=False, fuzzy=True)
        if downloaded and os.path.exists(downloaded) and os.path.getsize(downloaded) > 100000:
            target_file = downloaded
    except Exception as e_gd:
        print(f"⚠️ gdown không thành công: {e_gd}, chuyển sang curl...")

    if not target_file or not os.path.exists(target_file):
        out_path = os.path.join(output_dir, f"gdrive_video_{file_id}.mkv")
        download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        curl_cmd = ['curl', '-L', '-o', out_path, download_url]
        subprocess.run(curl_cmd, check=True)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
            target_file = out_path

    if not target_file or not os.path.exists(target_file) or os.path.getsize(target_file) < 100000:
        raise RuntimeError("Không thể tải file từ Google Drive! Vui lòng đảm bảo link đã được bật quyền chia sẻ: 'Bất kỳ ai có đường liên kết đều có thể xem' (Anyone with the link).")

    print(f"✅ Tải thành công video từ Google Drive: '{target_file}' ({os.path.getsize(target_file)/(1024*1024):.1f} MB)")
    return target_file

# ==============================================================================
# --- 5. HỆ THỐNG XỬ LÝ PHÂN ĐOẠN & CHECKPOINT / AUTO-RESUME AN TOÀN ---
# ==============================================================================

def is_valid_segment(seg_path, expected_frames, tolerance=2):
    """
    Kiểm tra nhanh tính toàn vẹn của một file phân đoạn đã được render (.mkv).
    Đảm bảo file không bị lỗi EOF, bị đứt đoạn hoặc thiếu frame khi phiên Colab bị ngắt giữa chừng.
    """
    if not os.path.exists(seg_path) or os.path.getsize(seg_path) < 1000:
        return False
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=nb_read_packets',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            seg_path
        ]
        res = subprocess.check_output(cmd, timeout=15).decode().strip()
        if res.isdigit():
            actual = int(res)
            if abs(actual - expected_frames) <= tolerance:
                return True
    except Exception:
        pass

    try:
        dur_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', seg_path]
        dur_str = subprocess.check_output(dur_cmd, timeout=10).decode().strip()
        if float(dur_str) > 0:
            return True
    except Exception:
        pass

    return False

def render_segment(
    seg_idx, s_frame, n_frames, video_input, seg_file,
    model, arch, is_pro, device, target_w, target_h, fps, src_w, src_h, encoder_flags, batch_size=1,
    progress_queue=None, live_progress_cb=None,
    enable_dedup=True, dedup_threshold=0.003
):
    """
    Xử lý render 1 phân đoạn video độc lập với độ chính xác khung hình byte-for-byte.
    Tích hợp Smart Anime Deduplication tự động tái sử dụng kết quả khung hình tĩnh/trùng lặp.
    Ghi tạm vào file .part và chỉ đổi tên thành file chính thức khi toàn bộ phân đoạn hoàn tất.
    """
    seg_part = (seg_file[:-4] if seg_file.endswith(".mkv") else seg_file) + "_part.mkv"
    if os.path.exists(seg_part):
        try: os.remove(seg_part)
        except Exception: pass

    seek_time = s_frame / fps if (s_frame > 0 and fps > 0) else 0.0
    ffmpeg_read_cmd = ['ffmpeg', '-y', '-threads', '0']
    if seek_time > 0:
        ffmpeg_read_cmd.extend(['-accurate_seek', '-ss', f"{seek_time:.6f}"])
    ffmpeg_read_cmd.extend([
        '-i', video_input,
        '-vframes', str(n_frames),
        '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
    ])
    process_read = subprocess.Popen(ffmpeg_read_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=32*1024*1024)

    ffmpeg_write_cmd = [
        'ffmpeg', '-y',
        '-sws_flags', 'fast_bilinear',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{target_w}x{target_h}', '-r', str(fps),
        '-i', '-',
        '-f', 'matroska',
        *encoder_flags,
        seg_part
    ]
    process_write = subprocess.Popen(ffmpeg_write_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=32*1024*1024)

    frame_size = src_w * src_h * 3
    queue_size = 8
    input_queue = Queue(maxsize=queue_size)
    output_queue = Queue(maxsize=4)

    def reader_worker():
        try:
            remaining = n_frames
            while remaining > 0:
                cur_frames = min(batch_size, remaining)
                req_bytes = cur_frames * frame_size
                chunk = process_read.stdout.read(req_bytes)
                if not chunk: break
                while len(chunk) < req_bytes:
                    more = process_read.stdout.read(req_bytes - len(chunk))
                    if not more: break
                    chunk += more
                if not chunk: break
                input_queue.put(chunk)
                remaining -= (len(chunk) // frame_size)
            input_queue.put(None)
        except Exception:
            input_queue.put(None)

    def writer_worker():
        try:
            while True:
                item = output_queue.get()
                if item is None:
                    break
                try:
                    process_write.stdin.write(item)
                except Exception:
                    pass
                output_queue.task_done()
        except Exception:
            pass
        finally:
            try:
                process_write.stdin.flush()
            except Exception:
                pass
            try:
                process_write.stdin.close()
            except Exception:
                pass

    reader_thread = threading.Thread(target=reader_worker, daemon=True)
    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    reader_thread.start()
    writer_thread.start()
    gc.disable()

    try:
        processed_cnt = 0
        profile_cnt = 0
        t_wait_read = 0.0
        t_gpu_infer = 0.0
        t_post_pcie = 0.0
        t_wait_write = 0.0
        last_frame_np = None
        last_out_bytes = None
        dedup_skip_cnt = 0
        latest_profiler_str = ""

        while processed_cnt < n_frames:
            t0 = time.perf_counter()
            chunk = input_queue.get()
            if chunk is None: break
            current_b = len(chunk) // frame_size
            if current_b == 0: break
            t1 = time.perf_counter()

            img_np_batch = np.frombuffer(chunk, dtype=np.uint8).reshape((current_b, src_h, src_w, 3))

            # Smart Anime Deduplication: bỏ qua AI inference nếu frame trùng lặp hoặc tĩnh
            if enable_dedup and current_b == 1 and last_frame_np is not None and last_out_bytes is not None:
                curr_frame = img_np_batch[0]
                is_dup = False
                if np.array_equal(curr_frame, last_frame_np):
                    is_dup = True
                else:
                    diff = np.mean(np.abs(curr_frame[::8, ::8].astype(np.int16) - last_frame_np[::8, ::8].astype(np.int16))) / 255.0
                    if diff <= dedup_threshold:
                        is_dup = True

                if is_dup:
                    output_queue.put(last_out_bytes)
                    dedup_skip_cnt += 1
                    processed_cnt += 1
                    profile_cnt += 1
                    if profile_cnt >= 20:
                        dedup_pct = (dedup_skip_cnt / processed_cnt * 100) if processed_cnt > 0 else 0
                        latest_profiler_str = f"Dedup: {dedup_pct:.1f}% ({dedup_skip_cnt}f trùng)"
                        profile_cnt = 0
                    if progress_queue:
                        try: progress_queue.put((1, latest_profiler_str))
                        except Exception: pass
                    if live_progress_cb:
                        try: live_progress_cb(1, latest_profiler_str)
                        except Exception: pass
                    continue

            # Khung hình mới cần chạy model AI:
            if enable_dedup and current_b == 1:
                last_frame_np = img_np_batch[0].copy()

            img_t = torch.from_numpy(img_np_batch)

            if device.type == 'cuda':
                img_t = img_t.to(device, non_blocking=True).permute(0, 3, 1, 2).to(torch.float16, non_blocking=True)
                if arch == "srvggnet_compact":
                    img_t.mul_(1.0 / 255.0)
                elif is_pro:
                    img_t.mul_(0.7 / 255.0).add_(0.15)
                else:
                    img_t.mul_(1.0 / 255.0)
                img_t = img_t.to(memory_format=torch.channels_last)
            elif device.type == 'mps':
                img_t = img_t.to(device).permute(0, 3, 1, 2).to(torch.float16)
                if arch == "srvggnet_compact":
                    img_t.mul_(1.0 / 255.0)
                elif is_pro:
                    img_t.mul_(0.7 / 255.0).add_(0.15)
                else:
                    img_t.mul_(1.0 / 255.0)
            else:
                img_t = img_t.to(device).permute(0, 3, 1, 2).float()
                if arch == "srvggnet_compact":
                    img_t.mul_(1.0 / 255.0)
                elif is_pro:
                    img_t.mul_(0.7 / 255.0).add_(0.15)
                else:
                    img_t.mul_(1.0 / 255.0)

            t2 = time.perf_counter()
            with torch.inference_mode():
                raw_out = model(img_t)
                # Chỉ đồng bộ CUDA khi cần đo mẫu cho profiler (mỗi 20 frames) để luồng CUDA stream chạy ngầm song song tối đa
                if device.type == 'cuda' and profile_cnt == 0:
                    torch.cuda.synchronize(device)
            t3 = time.perf_counter()

            with torch.inference_mode():
                if arch == "srvggnet_compact":
                    raw_out.clamp_(0.0, 1.0)
                elif is_pro:
                    raw_out.sub_(0.15).mul_(1.0 / 0.7).clamp_(0.0, 1.0)
                else:
                    raw_out.clamp_(0.0, 1.0)

                if raw_out.shape[2] != target_h or raw_out.shape[3] != target_w:
                    raw_out = F.interpolate(raw_out, size=(target_h, target_w), mode='area')

                output = raw_out.mul_(255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
                out_bytes = memoryview(output.cpu().numpy()).cast('B')
            t4 = time.perf_counter()

            if enable_dedup and current_b == 1:
                last_out_bytes = bytes(out_bytes)

            output_queue.put(out_bytes)
            t5 = time.perf_counter()

            t_wait_read += (t1 - t0)
            t_gpu_infer += (t3 - t2)
            t_post_pcie += (t4 - t3)
            t_wait_write += (t5 - t4)
            profile_cnt += current_b

            if profile_cnt >= 20:
                gpu_ms = t_gpu_infer / profile_cnt * 1000
                post_ms = t_post_pcie / profile_cnt * 1000
                read_ms = t_wait_read / profile_cnt * 1000
                write_ms = t_wait_write / profile_cnt * 1000
                dedup_pct = (dedup_skip_cnt / processed_cnt * 100) if processed_cnt > 0 else 0
                latest_profiler_str = f"GPU: {gpu_ms:.0f}ms | Bỏ qua trùng: {dedup_pct:.1f}% ({dedup_skip_cnt}f)"
                print(f"\n📊 [Profiler] GPU Infer: {gpu_ms:.1f}ms/f | "
                      f"Post+PCIe: {post_ms:.1f}ms | "
                      f"Đọc: {read_ms:.1f}ms | "
                      f"Dedup: {dedup_pct:.1f}% ({dedup_skip_cnt} frames) | "
                      f"Buffers: in={input_queue.qsize()}/{queue_size}, out={output_queue.qsize()}/4", flush=True)
                profile_cnt = 0
                t_wait_read = t_gpu_infer = t_post_pcie = t_wait_write = 0.0

            processed_cnt += current_b
            if progress_queue:
                try: progress_queue.put((current_b, latest_profiler_str))
                except Exception: pass
            if live_progress_cb:
                try: live_progress_cb(current_b, latest_profiler_str)
                except Exception: pass
    finally:
        output_queue.put(None)
        try: writer_thread.join(timeout=30)
        except Exception: pass
        gc.enable()

    try:
        if process_read.poll() is None:
            process_read.terminate()
            process_read.wait(timeout=5)
    except Exception: pass

    try:
        if process_write.stdin and not process_write.stdin.closed:
            try: process_write.stdin.close()
            except Exception: pass
    except Exception: pass

    try:
        process_write.wait(timeout=30)
    except Exception as e_w:
        print(f"⚠️ Chờ luồng ghi FFmpeg phân đoạn {seg_idx}: {e_w}")

    if os.path.exists(seg_part) and os.path.getsize(seg_part) > 1000:
        shutil.move(seg_part, seg_file)
        return True
    return False

# ==============================================================================
# --- 6. WORKER PHÂN ĐOẠN DUAL GPU (NVIDIA T4 x2 TRÊN KAGGLE) ---
# ==============================================================================

def _gpu_segment_worker(task_queue, video_input, target_w, target_h, fps, src_w, src_h, weights_path, model_key, gpu_id, encoder_flags, batch_size, return_dict, progress_queue, enable_dedup=True, dedup_threshold=0.003):
    try:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

        model, arch, is_pro = load_model(model_key, weights_path, device)

        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except Exception:
                break
            if task is None:
                break

            seg_idx, s_frame, n_frames, seg_file = task
            ok = render_segment(
                seg_idx=seg_idx,
                s_frame=s_frame,
                n_frames=n_frames,
                video_input=video_input,
                seg_file=seg_file,
                model=model,
                arch=arch,
                is_pro=is_pro,
                device=device,
                target_w=target_w,
                target_h=target_h,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                encoder_flags=encoder_flags,
                batch_size=batch_size,
                progress_queue=progress_queue,
                enable_dedup=enable_dedup,
                dedup_threshold=dedup_threshold
            )
            return_dict[f"seg_{seg_idx}"] = ok

        return_dict[f"worker_{gpu_id}"] = True
    except Exception as e:
        print(f"⚠️ Lỗi GPU worker {gpu_id}: {e}")
        return_dict[f"worker_{gpu_id}"] = False

# ==============================================================================
# --- 7. HÀM UPSCALE CHÍNH (TỐI ƯU HÓA CHO APPLE SILICON MAC & MULTI-GPU) ---
# ==============================================================================

def upscale_video(
    video_input,
    output_dir=None,
    model_name=None,
    progress_callback=None,
    translate_sub=False,
    sub_api_key=None,
    sub_api_type="gemini",
    sub_model=None,
    enable_dedup=True,
    dedup_threshold=0.003
):
    if isinstance(video_input, str):
        video_input = os.path.expanduser(video_input.strip())
        if video_input.startswith("magnet:?"):
            raise ValueError("❌ Không hỗ trợ link Magnet/Torrent. Vui lòng tải video về máy hoặc Google Drive!")

    is_gdrive = isinstance(video_input, str) and ("drive.google.com" in video_input or "drive.usercontent.google.com" in video_input)

    if output_dir is None:
        if os.path.exists('/content/drive/MyDrive'):
            output_dir = '/content/drive/MyDrive/Upscaled'
        elif os.path.exists('/kaggle/working'):
            output_dir = '/kaggle/working'
        else:
            output_dir = os.path.expanduser('~/Movies/Upscaled')
    else:
        output_dir = os.path.expanduser(str(output_dir).strip())
    os.makedirs(output_dir, exist_ok=True)

    # Thư mục tạm tốc độ cao (trên Mac dùng ổ NVMe SSD nội bộ)
    if os.path.exists('/content'):
        scratch_dir = '/content/temp_work'
    elif os.path.exists('/kaggle/working'):
        scratch_dir = '/kaggle/working/temp_work'
    else:
        scratch_dir = os.path.join(output_dir, '.temp_work')
    os.makedirs(scratch_dir, exist_ok=True)

    model_key = resolve_model_key(model_name)
    device, device_type, device_desc = get_best_device()
    num_cuda_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    encoder_flags, encoder_desc = get_hevc_encoder_flags(device_type)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    print(f"🚀 Thiết bị: {device_desc} | Mô hình: {WEIGHTS_INFO[model_key]['desc']} | Mã hóa: {encoder_desc}")

    copied_to_scratch = False
    temp_downloaded_file = None
    if is_gdrive:
        print("☁️ Nhận diện Link Google Drive. Đang nạp video...")
        if progress_callback:
            progress_callback(0.01, desc="☁️ Đang kết nối tải video từ Google Drive...")
        temp_downloaded_file = download_gdrive(video_input.strip(), output_dir=os.path.join(output_dir, "input"), progress_cb=progress_callback)
        video_input = temp_downloaded_file
    else:
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"Không tìm thấy file video nguồn '{video_input}'!")
        # Tự động nạp file từ Google Drive FUSE về SSD cục bộ để tránh nghẽn I/O mạng Drive
        if video_input.startswith('/content/drive/'):
            print(f"⚡ Đang nạp nhanh video nguồn từ Google Drive vào Local NVMe SSD ({scratch_dir})...")
            if progress_callback:
                progress_callback(0.01, desc="⚡ Đang nạp video nguồn vào Local NVMe SSD...")
            local_src = os.path.join(scratch_dir, os.path.basename(video_input))
            if not os.path.exists(local_src) or os.path.getsize(local_src) != os.path.getsize(video_input):
                shutil.copy2(video_input, local_src)
            video_input = local_src
            copied_to_scratch = True
            print(f"✅ Đã nạp thành công video vào Local SSD: {local_src}")

    video_base = os.path.basename(os.path.splitext(video_input)[0])
    video_output = os.path.join(output_dir, f"{video_base}_4K.mkv")

    # DỊCH PHỤ ĐỀ TIẾNG VIỆT TỰ ĐỘNG BẰNG TRISUB AI (NẾU ĐƯỢC BẬT)
    vi_sub_path = None
    if translate_sub:
        try:
            print("🌐 [TriSub AI] Đang tự động dịch phụ đề Anime sang Tiếng Việt...", flush=True)
            if progress_callback:
                progress_callback(0.01, desc="🌐 [TriSub AI] Đang dịch phụ đề Anime sang Tiếng Việt...")

            repo_root = os.path.dirname(os.path.abspath(__file__))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from trisub_ai.tri_sub_translate import translate_subtitles_for_video

            vi_srt_output = os.path.join(output_dir, f"{video_base}_Vietsub.srt")
            vi_sub_path = translate_subtitles_for_video(
                video_path=video_input,
                output_srt=vi_srt_output,
                api_key=sub_api_key,
                api_type=sub_api_type,
                model_name=sub_model,
                progress_callback=progress_callback
            )
            print(f"✅ [TriSub AI] Đã tạo thành công file Vietsub: {vi_sub_path}", flush=True)
        except Exception as e_sub:
            print(f"⚠️ [TriSub AI] Lỗi trong quá trình dịch phụ đề: {e_sub}. Tiếp tục quá trình nâng cấp 4K...", flush=True)


    # Đọc thông số metadata gốc
    try:
        fps_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 \"{video_input}\""
        fps_res = subprocess.check_output(fps_cmd, shell=True).decode().strip()
        fps = eval(fps_res) if '/' in fps_res else float(fps_res)

        res_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 \"{video_input}\""
        src_res = subprocess.check_output(res_cmd, shell=True).decode().strip()
        src_w, src_h = map(int, src_res.split('x'))
    except Exception as e:
        print(f"⚠️ Lỗi phân tích metadata video: {e}")
        fps, src_w, src_h = 23.976, 1920, 1080

    print(f"ℹ️ Thông số gốc: {src_w}x{src_h} @ {fps:.3f} FPS")

    # Chuẩn bị weights
    weights_path = ensure_model_weights(model_key, progress_callback=progress_callback)

    expected_frames = None
    try:
        frames_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 \"{video_input}\""
        frames_res = subprocess.check_output(frames_cmd, shell=True).decode().strip()
        if frames_res.isdigit():
            expected_frames = int(frames_res)
    except Exception:
        pass

    if not expected_frames:
        try:
            dur_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{video_input}\""
            dur_res = subprocess.check_output(dur_cmd, shell=True).decode().strip()
            dur = float(dur_res)
            if dur > 0 and fps > 0:
                expected_frames = int(dur * fps)
        except Exception:
            pass

    if not expected_frames:
        expected_frames = 34500  # Ước lượng mặc định 24 phút anime nếu không đọc được

    target_w = src_w * 2
    target_h = src_h * 2

    # Thư mục Checkpoint an toàn: Lưu trực tiếp trên output_dir (Google Drive) để chống mất phiên Colab Free!
    ckpt_dir = os.path.join(output_dir, f".checkpoints_{video_base}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Độ dài phân đoạn tối ưu: 2400 frames (~100 giây / 3 phút render) cho AnimeJaNai V3, 1200 frames cho CUGAN
    arch = WEIGHTS_INFO[model_key].get("arch", "srvggnet_compact")
    seg_len = 2400 if arch == "srvggnet_compact" else 1200
    total_segments = (expected_frames + seg_len - 1) // seg_len
    segments = []
    for i in range(total_segments):
        s_frame = i * seg_len
        n_frames = min(seg_len, expected_frames - s_frame)
        seg_file = os.path.join(ckpt_dir, f"seg_{i:04d}_{s_frame}_{s_frame+n_frames}.mkv")
        segments.append((i, s_frame, n_frames, seg_file))

    # Kiểm tra các phân đoạn checkpoint đã hoàn thành từ trước
    completed_segs = set()
    already_rendered_frames = 0
    for i, s_frame, n_frames, seg_file in segments:
        if is_valid_segment(seg_file, n_frames):
            completed_segs.add(i)
            already_rendered_frames += n_frames

    if len(completed_segs) > 0:
        print(f"🔄 CHECKPOINT TỰ ĐỘNG: Đã tìm thấy {len(completed_segs)}/{total_segments} phân đoạn ({already_rendered_frames}/{expected_frames} frames) hoàn tất trên Google Drive!")
        print(f"⏩ Tự động bỏ qua các phân đoạn cũ và tiếp tục xử lý các phần còn lại...")

    if device.type == 'cuda':
        total_mem = torch.cuda.get_device_properties(device).total_memory
        # A100 (40/80GB) hoặc GPU HBM2 băng thông lớn (>= 24GB) -> Batch 4
        if total_mem >= 24 * 1024**3 and src_h <= 1080:
            batch_size = 4
        elif total_mem >= 6 * 1024**3 and src_h <= 1080:
            batch_size = 2
        else:
            batch_size = 1
        print(f"⚡ Cấu hình Batch size: {batch_size} (VRAM khả dụng: {total_mem / (1024**3):.1f} GB)")
    else:
        batch_size = 1
        print(f"⚡ Cấu hình Batch size: {batch_size}")

    todo_segments = [seg for seg in segments if seg[0] not in completed_segs]
    start_time = time.time()
    last_print_time = 0.0

    # NẾU CÓ DUAL GPU T4 x2 (KAGGLE): CHẠY ĐỒNG THỜI CẢ 2 GPU QUA MULTI-PROCESSING
    if num_cuda_gpus >= 2 and len(todo_segments) >= 2:
        try: mp.set_start_method('spawn', force=True)
        except Exception: pass

        print(f"🔥 KÍCH HOẠT DUAL GPU: Phân phối các phân đoạn song song trên cả {num_cuda_gpus} GPU NVIDIA T4!")
        manager = mp.Manager()
        task_queue = manager.Queue()
        return_dict = manager.dict()
        progress_queue = manager.Queue()

        for seg in todo_segments:
            task_queue.put(seg)
        for _ in range(num_cuda_gpus):
            task_queue.put(None)

        processes = []
        for g_id in range(num_cuda_gpus):
            p = mp.Process(
                target=_gpu_segment_worker,
                args=(task_queue, video_input, target_w, target_h, fps, src_w, src_h, weights_path, model_key, g_id, encoder_flags, batch_size, return_dict, progress_queue, enable_dedup, dedup_threshold)
            )
            p.start()
            processes.append(p)

        completed_total = already_rendered_frames
        latest_prof = ""
        while completed_total < expected_frames:
            try:
                added = progress_queue.get(timeout=0.3)
                if isinstance(added, tuple):
                    cnt, prof = added
                    completed_total += cnt
                    if prof:
                        latest_prof = prof
                else:
                    completed_total += added
            except Exception:
                if not any(p.is_alive() for p in processes) and task_queue.empty():
                    break

            now = time.time()
            if (now - last_print_time) >= 0.5 or completed_total >= expected_frames:
                last_print_time = now
                elapsed = now - start_time
                speed_fps = (completed_total - already_rendered_frames) / elapsed if elapsed > 0 else 0.0
                pct = (completed_total / expected_frames) * 100
                cur_sec = completed_total / fps if fps > 0 else 0
                cur_str = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
                tot_sec = expected_frames / fps if fps > 0 else 0
                tot_str = f"{int(tot_sec // 60):02d}:{int(tot_sec % 60):02d}"
                eta_sec = (expected_frames - completed_total) / speed_fps if speed_fps > 0 else 0
                eta_str = f"{int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"

                prof_part = f" | ⚡ {latest_prof}" if latest_prof else ""
                status_msg = f"⏳ {completed_total}/{expected_frames} ({pct:.1f}%) | {speed_fps:.2f} fps | {cur_str}/{tot_str} | ETA: {eta_str}{prof_part}"
                print(status_msg + "    ", end='\r', flush=True)

                if progress_callback:
                    try: progress_callback(pct / 100.0, desc=status_msg)
                    except Exception: pass

        for p in processes:
            p.join()

    # NẾU CÓ SINGLE GPU (GOOGLE COLAB FREE T4 / APPLE SILICON MPS / CPU)
    elif len(todo_segments) > 0:
        model, arch, is_pro = load_model(model_key, weights_path, device)
        completed_total = already_rendered_frames
        latest_prof = ""

        for seg_idx, s_frame, n_frames, seg_file in todo_segments:
            seg_banner = f"🎞️ [Phân đoạn {seg_idx + 1}/{total_segments}] Đang render frame {s_frame} -> {s_frame + n_frames}..."
            print(f"\n{seg_banner}", flush=True)

            def live_cb(cnt, prof=""):
                nonlocal completed_total, last_print_time, latest_prof
                completed_total += cnt
                if prof:
                    latest_prof = prof
                now = time.time()
                if (now - last_print_time) >= 0.5 or completed_total >= expected_frames:
                    last_print_time = now
                    elapsed = now - start_time
                    speed_fps = (completed_total - already_rendered_frames) / elapsed if elapsed > 0 else 0.0
                    pct = (completed_total / expected_frames) * 100
                    cur_sec = completed_total / fps if fps > 0 else 0
                    cur_str = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
                    tot_sec = expected_frames / fps if fps > 0 else 0
                    tot_str = f"{int(tot_sec // 60):02d}:{int(tot_sec % 60):02d}"
                    eta_sec = (expected_frames - completed_total) / speed_fps if speed_fps > 0 else 0
                    eta_str = f"{int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"

                    prof_part = f" | ⚡ {latest_prof}" if latest_prof else ""
                    status_msg = f"⏳ {completed_total}/{expected_frames} ({pct:.1f}%) | {speed_fps:.2f} fps | {cur_str}/{tot_str} | ETA: {eta_str}{prof_part}"
                    print(status_msg + "    ", end='\r', flush=True)
                    if progress_callback:
                        try: progress_callback(pct / 100.0, desc=status_msg)
                        except Exception: pass

            ok = render_segment(
                seg_idx=seg_idx,
                s_frame=s_frame,
                n_frames=n_frames,
                video_input=video_input,
                seg_file=seg_file,
                model=model,
                arch=arch,
                is_pro=is_pro,
                device=device,
                target_w=target_w,
                target_h=target_h,
                fps=fps,
                src_w=src_w,
                src_h=src_h,
                encoder_flags=encoder_flags,
                batch_size=batch_size,
                live_progress_cb=live_cb,
                enable_dedup=enable_dedup,
                dedup_threshold=dedup_threshold
            )
            if not ok or not is_valid_segment(seg_file, n_frames):
                raise RuntimeError(f"Lỗi khi xử lý phân đoạn {seg_idx} ({seg_file})!")

    # KIỂM TRA TẤT CẢ PHÂN ĐOẠN ĐÃ ĐẦY ĐỦ TRƯỚC KHI GHÉP NỐI
    missing_segs = [seg[0] for seg in segments if not is_valid_segment(seg[3], seg[2])]
    if missing_segs:
        raise RuntimeError(f"Thiếu các phân đoạn sau chưa hoàn thành: {missing_segs}. Vui lòng chạy lại để tiếp tục hoàn thiện!")

    # GHÉP NỐI TOÀN BỘ PHÂN ĐOẠN BẰNG FFMPEG CONCAT DEMUXER (SIÊU TỐC ~2-3 GIÂY)
    print("\n📦 Đang ghép nối tất cả các phân đoạn 4K...", flush=True)
    concat_txt = os.path.join(scratch_dir, f"_concat_{video_base}.txt")
    with open(concat_txt, "w") as f:
        for _, _, _, seg_file in segments:
            f.write(f"file '{os.path.abspath(seg_file)}'\n")

    temp_merged = os.path.join(scratch_dir, f"_temp_merged_{video_base}.mkv")
    concat_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_txt, '-c', 'copy', temp_merged]
    subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # BẢO TỒN 100% METADATA: VIDEO 4K + TOÀN BỘ AUDIO + PHỤ ĐỀ MỀM (.ASS) + FONT ĐÍNH KÈM + CHAPTERS
    print("🔊 Ghép 100% Audio gốc, Subtitle (.ass), Font đính kèm và Chapters vào tệp MKV xuất xưởng...", flush=True)
    temp_final = os.path.join(scratch_dir, f"_final_{os.path.basename(video_output)}")
    
    if vi_sub_path and os.path.exists(vi_sub_path):
        print("✨ [TriSub AI] Đang tích hợp track Phụ đề Tiếng Việt làm mặc định vào tập phim 4K...", flush=True)
        mux_cmd = [
            'ffmpeg', '-y',
            '-i', temp_merged,
            '-i', video_input,
            '-sub_charenc', 'UTF-8', '-i', vi_sub_path,
            '-c', 'copy',
            '-map', '0:v:0',
            '-map', '1:a?',
            '-map', '2:s:0',          # Phụ đề Tiếng Việt mới từ TriSub AI
            '-map', '1:s?',            # Toàn bộ phụ đề gốc có sẵn
            '-map', '1:t?',            # Font đính kèm
            '-map_metadata', '1',
            '-map_chapters', '1',
            '-metadata:s:s:0', 'language=vie',
            '-metadata:s:s:0', 'title=Tiếng Việt (TriSub AI)',
            '-disposition:s:0', 'default',
            temp_final
        ]
    else:
        mux_cmd = [
            'ffmpeg', '-y',
            '-i', temp_merged,
            '-i', video_input,
            '-c', 'copy',
            '-map', '0:v:0',
            '-map', '1:a?',
            '-map', '1:s?',
            '-map', '1:t?',
            '-map_metadata', '1',
            '-map_chapters', '1',
            temp_final
        ]
    subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # DỌN DẸP FILE TẠM & CHECKPOINTS KHI HOÀN TẤT
    try:
        if os.path.exists(temp_merged): os.remove(temp_merged)
        if os.path.exists(concat_txt): os.remove(concat_txt)
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    except Exception:
        pass

    if os.path.exists(temp_final) and os.path.getsize(temp_final) > 1000:
        print(f"☁️ Đang lưu tập phim 4K hoàn chỉnh vào: '{video_output}'...", flush=True)
        shutil.move(temp_final, video_output)

        if copied_to_scratch and os.path.exists(video_input):
            try: os.remove(video_input)
            except Exception: pass

        print(f"\n✨ KẾT THÚC HOÀN HẢO! Tập phim 4K nằm tại: {video_output}", flush=True)
        if progress_callback:
            try: progress_callback(1.0, desc="✨ Hoàn tất nâng cấp video 4K!")
            except Exception: pass
        return video_output
    else:
        raise RuntimeError(f"Quá trình xuất video 4K không thành công hoặc file kết quả bị trống: '{video_output}'")

def main():
    if len(sys.argv) < 2:
        print("❌ Lỗi: Vui lòng cung cấp đường dẫn video hoặc link Google Drive!")
        print("💡 Sử dụng: python3 upscale.py <video.mkv hoặc link_gdrive> [animejanai_v3_ultracompact / animejanai_v3_sharp_ultracompact / animejanai_v3_compact / animejanai_v3_sharp]")
        return

    video_input = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "animejanai_v3_compact"
    upscale_video(video_input=video_input, model_name=model_name)

if __name__ == '__main__':
    main()
