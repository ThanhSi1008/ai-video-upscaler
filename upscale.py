import os
import sys
import re
import json
import gc
import subprocess
import urllib.request
import warnings
import time
import threading
import shutil
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

# ==============================================================================
# --- 1. KIẾN TRÚC MÔ HÌNH CỐT LÕI: Real-CUGAN Native 2x (Cascaded U-Net 2x) ---
# ==============================================================================

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        x0 = torch.mean(x, dim=(2, 3), keepdim=True, dtype=torch.float32).to(x.dtype)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        return x.mul_(x0)

class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.seblock = SEBlock(out_channels, reduction=8, bias=True) if se else None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z

class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = x1[:, :, 4:-4, 4:-4]
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, alpha=1.0):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = x1[:, :, 16:-16, 16:-16]
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
        x2 = x2[:, :, 4:-4, 4:-4]
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x4 = self.conv4(x2 + x3)
        if alpha != 1.0:
            x4 = x4 * alpha
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)
        z = self.conv_bottom(x5)
        return z

class UpCunet2x(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet2x, self).__init__()
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)
        self.mode = "se"

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, alpha=1.0):
        # Chế độ Full-Frame (tile_mode=0) cho độ nét vector tối đa và không lỗi ghép mảnh
        n, c, h0, w0 = x.shape
        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), 'reflect')
        x = self.unet1(x)
        if self.mode == "se":
            x = x[:, :, 20:-20, 20:-20]
            if w0 != pw or h0 != ph:
                x = x[:, :, :h0 * 2, :w0 * 2]
            return x

        x0 = self.unet2(x, alpha)
        x = x[:, :, 20:-20, 20:-20]
        x = torch.add(x0, x)
        if w0 != pw or h0 != ph:
            x = x[:, :, :h0 * 2, :w0 * 2]
        return x

# ==============================================================================
# --- 4. DANH MỤC WEIGHTS REAL-CUGAN PRO NATIVE 2x ---
# ==============================================================================

WEIGHTS_INFO = {
    "cugan_conservative": {
        "file": "up2x-latest-conservative.pth",
        "desc": "Real-CUGAN 2x Conservative (Khuyên dùng cho SubsPlease Web-DL)",
        "urls": [
            "https://huggingface.co/spaces/mayhug/Real-CUGAN/resolve/main/weights/up2x-latest-conservative.pth",
            "https://raw.githubusercontent.com/bilibili/ailab/main/Real-CUGAN/weights_v3/up2x-latest-conservative.pth"
        ]
    },
    "cugan_no_denoise": {
        "file": "up2x-latest-no-denoise.pth",
        "desc": "Real-CUGAN 2x No-Denoise (Tối ưu cho Blu-ray Remux)",
        "urls": [
            "https://huggingface.co/spaces/mayhug/Real-CUGAN/resolve/main/weights/up2x-latest-no-denoise.pth",
            "https://raw.githubusercontent.com/bilibili/ailab/main/Real-CUGAN/weights_v3/up2x-latest-no-denoise.pth"
        ]
    },
    "cugan_denoise3x": {
        "file": "up2x-latest-denoise3x.pth",
        "desc": "Real-CUGAN 2x Denoise3x (Khử nhiễu nặng cho anime cũ)",
        "urls": [
            "https://huggingface.co/spaces/mayhug/Real-CUGAN/resolve/main/weights/up2x-latest-denoise3x.pth",
            "https://raw.githubusercontent.com/bilibili/ailab/main/Real-CUGAN/weights_v3/up2x-latest-denoise3x.pth"
        ]
    }
}

def resolve_model_info(name):
    name_l = (name or "").lower()
    # Mặc định là SE (siêu tốc ~7-9 FPS). Nếu người dùng chỉ định rõ "pro" thì mới chạy Pro (~1.6 FPS)
    mode = "pro" if ("_pro" in name_l or " pro" in name_l or name_l.endswith("pro")) else "se"
    if "no_denoise" in name_l or "no-denoise" in name_l:
        key = "cugan_no_denoise"
    elif "denoise3x" in name_l or ("denoise" in name_l and "3" in name_l):
        key = "cugan_denoise3x"
    else:
        key = "cugan_conservative"
    return key, mode

def resolve_model_key(name):
    key, _ = resolve_model_info(name)
    return key

def ensure_model_weights(model_key, progress_callback=None):
    info = WEIGHTS_INFO[model_key]
    weights_path = info["file"]
    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 100000:
        return weights_path

    print(f"📥 Tự động tải weights mô hình '{info['desc']}'...")
    if progress_callback:
        progress_callback(0.01, desc=f"📥 Đang tải weights: {info['file']}...")

    download_success = False
    for url in info["urls"]:
        try:
            print(f"🔗 Đang tải từ: {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp, open(weights_path, 'wb') as f:
                shutil.copyfileobj(resp, f)
            if os.path.exists(weights_path) and os.path.getsize(weights_path) > 100000:
                print(f"✅ Đã tải thành công: {weights_path} ({os.path.getsize(weights_path)/(1024*1024):.2f} MB)")
                download_success = True
                break
        except Exception as e:
            print(f"⚠️ Thất bại tải từ {url} qua urllib: {e}")
            try:
                print("🔄 Thử lại bằng curl --http1.1...")
                curl_cmd = ['curl', '--http1.1', '-L', '-o', weights_path, url]
                subprocess.run(curl_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(weights_path) and os.path.getsize(weights_path) > 100000:
                    print(f"✅ Đã tải thành công bằng curl: {weights_path} ({os.path.getsize(weights_path)/(1024*1024):.2f} MB)")
                    download_success = True
                    break
            except Exception as e2:
                print(f"⚠️ Curl cũng thất bại: {e2}")

    if not download_success:
        raise RuntimeError(f"Không thể tải weights mô hình {model_key}. Vui lòng kiểm tra kết nối mạng!")
    return weights_path

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
        return torch.device('mps'), 'mps', "Apple Silicon M3 Pro (Metal Performance Shaders - MPS)"
    return torch.device('cpu'), 'cpu', "CPU (Software Mode)"

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
        try:
            res = subprocess.run(['ffmpeg', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if 'hevc_nvenc' in res.stdout:
                return [
                    '-c:v', 'hevc_nvenc',
                    '-preset', 'p4',
                    '-tune', 'hq',
                    '-cq', '18',
                    '-spatial-aq', '1',
                    '-pix_fmt', 'yuv420p10le',
                    '-profile:v', 'main10'
                ], "hevc_nvenc 10-bit (NVIDIA Hardware Fast)"
        except Exception:
            pass

    return [
        '-c:v', 'libx265',
        '-crf', '18',
        '-preset', 'veryfast',
        '-pix_fmt', 'yuv420p10le'
    ], "libx265 10-bit (CPU Software - Fast)"

def load_realcugan_model(model_key, weights_path, device, mode="se"):
    model = UpCunet2x(in_channels=3, out_channels=3)
    model.set_mode(mode)
    state_dict = torch.load(weights_path, map_location='cpu')
    is_pro = ("pro" in state_dict)
    if is_pro:
        del state_dict["pro"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if device.type == 'cuda':
        model = model.half().to(memory_format=torch.channels_last)
    elif device.type == 'mps':
        model = model.half()

    model = model.to(device)
    return model, is_pro

# ==============================================================================
# ==============================================================================
# --- 5. TẢI TẬP PHIM TỪ GOOGLE DRIVE HOẶC MAGNET ---
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
    # 1. Thử dùng thư viện gdown
    try:
        import gdown
        downloaded = gdown.download(id=file_id, output=output_dir + "/", quiet=False, fuzzy=True)
        if downloaded and os.path.exists(downloaded) and os.path.getsize(downloaded) > 100000:
            target_file = downloaded
    except Exception as e_gd:
        print(f"⚠️ gdown không thành công: {e_gd}, chuyển sang curl...")

    # 2. Fallback dùng curl trực tiếp
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

def download_magnet(magnet_uri, output_dir="/content/input", progress_cb=None):
    os.makedirs(output_dir, exist_ok=True)
    if shutil.which("aria2c") is None:
        raise RuntimeError(
            "❌ Lệnh 'aria2c' chưa được cài đặt!\n"
            "Chạy lệnh sau trên ô code Kaggle Notebook:\n"
            "  !apt-get update -qq && apt-get install -y aria2 -qq"
        )

    if progress_cb:
        progress_cb(0.01, desc="🧲 Đang kết nối aria2c tải tập anime qua Magnet...")

    print(f"📥 Bắt đầu kéo Magnet link bằng aria2c vào: {output_dir}")
    cmd = [
        "aria2c",
        "--seed-time=0",
        "--disable-ipv6=true",
        "--max-connection-per-server=16",
        "--split=16",
        "--bt-stop-timeout=120",
        "--summary-interval=5",
        f"--dir={output_dir}",
        magnet_uri
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        line_clean = line.strip()
        if line_clean:
            print(f"[aria2] {line_clean}")
            if "%" in line_clean and progress_cb:
                try:
                    match = re.search(r'\((\d+)%\)', line_clean)
                    if match:
                        pct = int(match.group(1))
                        progress_cb(0.01 + 0.08 * (pct / 100.0), desc=f"🧲 Đang kéo torrent: {pct}%...")
                except Exception:
                    pass

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Lỗi khi tải torrent bằng aria2c (Mã lỗi: {proc.returncode})")

    video_exts = {".mkv", ".mp4", ".ts", ".m2ts"}
    downloaded_videos = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in video_exts:
                p = os.path.join(root, f)
                if os.path.getsize(p) > 50 * 1024 * 1024:
                    downloaded_videos.append(p)

    if not downloaded_videos:
        raise FileNotFoundError(f"Không tìm thấy file video anime trong thư mục '{output_dir}'!")

    downloaded_videos.sort(key=lambda x: os.path.getsize(x), reverse=True)
    selected_video = downloaded_videos[0]
    print(f"✅ Tải thành công tập phim: '{selected_video}' ({os.path.getsize(selected_video)/(1024*1024):.1f} MB)")
    return selected_video

# ==============================================================================
# --- 6. WORKER PHÂN ĐOẠN DUAL GPU (NVIDIA T4 x2 TRÊN KAGGLE) ---
# ==============================================================================

def _gpu_segment_worker(video_input, start_frame, total_frames_to_process, target_w, target_h, fps, src_w, src_h, weights_path, model_key, gpu_id, chunk_output_path, return_dict, progress_queue, model_mode="se"):
    try:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model, is_pro = load_realcugan_model(model_key, weights_path, device, mode=model_mode)

        seek_time = start_frame / fps if (start_frame > 0 and fps > 0) else 0.0
        
        ffmpeg_read_cmd = ['ffmpeg', '-y', '-threads', '0']
        if seek_time > 0:
            ffmpeg_read_cmd.extend(['-ss', f"{seek_time:.4f}"])
        ffmpeg_read_cmd.extend([
            '-i', video_input,
            '-vframes', str(total_frames_to_process),
            '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
        ])
        process_read = subprocess.Popen(ffmpeg_read_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=32*1024*1024)

        if os.path.exists(chunk_output_path):
            try: os.remove(chunk_output_path)
            except Exception: pass

        # MÃ HÓA PHẦN CỨNG HEVC 10-BIT (NVENC TURING CHUẨN 4K MASTER - FAST PRESET P4)
        ffmpeg_write_cmd = [
            'ffmpeg', '-y',
            '-threads', '0',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{target_w}x{target_h}', '-r', str(fps),
            '-i', '-',
            '-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq', '-cq', '18',
            '-spatial-aq', '1', '-pix_fmt', 'yuv420p10le', '-profile:v', 'main10',
            chunk_output_path
        ]
        process_write = subprocess.Popen(ffmpeg_write_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=32*1024*1024)

        frame_size = src_w * src_h * 3
        batch_size = 2 if (model_mode == "se" or src_h <= 720) else 1
        queue_size = 8

        input_queue = Queue(maxsize=queue_size)
        output_queue = Queue(maxsize=queue_size)

        def reader_worker():
            try:
                remaining = total_frames_to_process
                while remaining > 0:
                    cur_frames = min(batch_size, remaining)
                    req_bytes = cur_frames * frame_size
                    chunk = process_read.stdout.read(req_bytes)
                    if not chunk:
                        break
                    while len(chunk) < req_bytes:
                        more = process_read.stdout.read(req_bytes - len(chunk))
                        if not more:
                            break
                        chunk += more
                    if not chunk:
                        break
                    input_queue.put(chunk)
                    remaining -= (len(chunk) // frame_size)
                input_queue.put(None)
            except Exception:
                input_queue.put(None)

        def writer_worker():
            try:
                while True:
                    item = output_queue.get()
                    if item is None: break
                    try:
                        process_write.stdin.write(item)
                    except Exception: pass
                    output_queue.task_done()
            except Exception: pass

        reader_thread = threading.Thread(target=reader_worker, daemon=True)
        writer_thread = threading.Thread(target=writer_worker, daemon=True)
        reader_thread.start()
        writer_thread.start()
        gc.disable()

        processed_cnt = 0
        while processed_cnt < total_frames_to_process:
            chunk = input_queue.get()
            if chunk is None: break
            current_b = len(chunk) // frame_size
            if current_b == 0: break

            img_np_batch = np.frombuffer(chunk, dtype=np.uint8).reshape((current_b, src_h, src_w, 3))
            img_t = torch.from_numpy(img_np_batch).to(device, non_blocking=True)
            img_t = img_t.permute(0, 3, 1, 2).to(torch.float16, non_blocking=True)
            if is_pro:
                img_t.mul_(0.7 / 255.0).add_(0.15)
            else:
                img_t.mul_(1.0 / 255.0)
            img_t = img_t.to(memory_format=torch.channels_last)

            with torch.inference_mode():
                raw_out = model(img_t)
                if is_pro:
                    raw_out.sub_(0.15).mul_(1.0 / 0.7).clamp_(0.0, 1.0)
                else:
                    raw_out.clamp_(0.0, 1.0)

                # NATIVE 2x ĐÃ RA CHÍNH XÁC (3840x2160) NÊN HOÀN TOÀN BỎ QUA INTERPOLATE
                if raw_out.shape[2] != target_h or raw_out.shape[3] != target_w:
                    raw_out = F.interpolate(raw_out, size=(target_h, target_w), mode='area')

                output = raw_out.mul_(255.0).round_().to(torch.uint8).permute(0, 2, 3, 1).contiguous()

            output_queue.put(memoryview(output.cpu().numpy()).cast('B'))

            processed_cnt += current_b
            try: progress_queue.put(current_b)
            except Exception: pass

        output_queue.put(None)
        writer_thread.join(timeout=10)
        gc.enable()

        try:
            if process_read.poll() is None:
                process_read.terminate()
                process_read.wait(timeout=5)
        except Exception: pass

        try:
            if process_write.stdin and not process_write.stdin.closed:
                process_write.stdin.close()
            process_write.wait(timeout=30)
        except Exception as e_w:
            print(f"⚠️ Đóng luồng ghi FFmpeg worker {gpu_id}: {e_w}")

        return_dict[gpu_id] = True
    except Exception as e:
        print(f"⚠️ Lỗi GPU worker {gpu_id}: {e}")
        return_dict[gpu_id] = False

# ==============================================================================
# --- 7. HÀM UPSCALE CHÍNH CHO KAGGLE DUAL NVIDIA T4 ---
# ==============================================================================

def upscale_video(video_input, output_dir=None, model_name="cugan_conservative", progress_callback=None):
    is_magnet = isinstance(video_input, str) and video_input.strip().startswith("magnet:?")
    is_gdrive = isinstance(video_input, str) and ("drive.google.com" in video_input or "drive.usercontent.google.com" in video_input)
    
    if output_dir is None:
        if os.path.exists('/content/drive/MyDrive'):
            output_dir = '/content/drive/MyDrive/Upscaled'
        elif os.path.exists('/kaggle/working'):
            output_dir = '/kaggle/working'
        else:
            output_dir = os.path.expanduser('~/Movies/Upscaled')
    os.makedirs(output_dir, exist_ok=True)

    # Dùng ổ SSD cục bộ cho file tạm để ghi với tốc độ 500+ MB/s, tránh độ trễ I/O mạng của Google Drive
    if os.path.exists('/content'):
        scratch_dir = '/content/temp_work'
    elif os.path.exists('/kaggle/working'):
        scratch_dir = '/kaggle/working/temp_work'
    else:
        scratch_dir = os.path.join(tempfile.gettempdir(), 'ai_upscale_work')
    os.makedirs(scratch_dir, exist_ok=True)

    model_key, model_mode = resolve_model_info(model_name)
    device, device_type, device_desc = get_best_device()
    num_cuda_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    encoder_flags, encoder_desc = get_hevc_encoder_flags(device_type)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    mode_desc = "⚡ Bản SE Siêu Tốc (~7–9 FPS)" if model_mode == "se" else "👑 Bản Pro (~1.6 FPS)"
    print(f"🚀 Thiết bị: {device_desc} | Mô hình: {WEIGHTS_INFO[model_key]['desc']} [{mode_desc}] | Mã hóa: {encoder_desc}")

    temp_downloaded_file = None
    if is_gdrive:
        print("☁️ Nhận diện Link Google Drive. Đang tải video...")
        if progress_callback:
            progress_callback(0.01, desc="☁️ Đang kết nối tải video từ Google Drive...")
        temp_downloaded_file = download_gdrive(video_input.strip(), output_dir=os.path.join(output_dir, "input"), progress_cb=progress_callback)
        video_input = temp_downloaded_file
    elif is_magnet:
        print("🧲 Nhận diện Magnet link. Bắt đầu tải tập anime bằng aria2c...")
        if progress_callback:
            progress_callback(0.01, desc="🧲 Đang tải anime qua Magnet link (aria2c)...")
        temp_downloaded_file = download_magnet(video_input.strip(), output_dir=os.path.join(output_dir, "input"), progress_cb=progress_callback)
        video_input = temp_downloaded_file
    else:
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"Không tìm thấy file video nguồn '{video_input}'!")
        # Tự động nạp file từ Google Drive FUSE về SSD cục bộ để tránh nghẽn I/O mạng Drive
        if video_input.startswith('/content/drive/'):
            print(f"⚡ Đang nạp nhanh video nguồn từ Google Drive vào Local NVMe SSD ({scratch_dir})...")
            if progress_callback:
                progress_callback(0.01, desc="⚡ Đang nạp video nguồn vào Local NVMe SSD...")
            local_src = os.path.join(scratch_dir, f"source_{os.path.basename(video_input)}")
            if not os.path.exists(local_src) or os.path.getsize(local_src) != os.path.getsize(video_input):
                shutil.copy2(video_input, local_src)
            video_input = local_src
            print(f"✅ Đã nạp thành công video vào Local SSD: {local_src}")

    # LUÔN XUẤT RA ĐỊNH DẠNG .MKV ĐỂ BẢO TỒN NGUYÊN VẸN TOÀN BỘ PHỤ ĐỀ MỀM (.ASS) VÀ FONT ĐÍNH KÈM!
    video_base = os.path.basename(os.path.splitext(video_input)[0])
    video_output = os.path.join(output_dir, f"{video_base}_4K_RealCUGAN.mkv")

    if os.path.exists(video_output):
        try: os.remove(video_output)
        except Exception: pass

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

    # Tải weights Real-CUGAN
    weights_path = ensure_model_weights(model_key, progress_callback=progress_callback)

    expected_frames = None
    try:
        frames_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 \"{video_input}\""
        frames_res = subprocess.check_output(frames_cmd, shell=True).decode().strip()
        if frames_res.isdigit():
            expected_frames = int(frames_res)
    except Exception as e:
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

    # Native 2x: 1080p -> 4K Ultra-HD (3840x2160)
    target_w = src_w * 2
    target_h = src_h * 2

    # NẾU CÓ DUAL GPU T4 x2 TRÊN KAGGLE: KÍCH HOẠT MULTI-PROCESSING ĐỘC LẬP
    if num_cuda_gpus >= 2 and expected_frames and expected_frames > 100:
        try: mp.set_start_method('spawn', force=True)
        except Exception: pass

        print(f"🔥 KÍCH HOẠT DUAL GPU: Chạy song song cả {num_cuda_gpus} card NVIDIA T4 cùng lúc!")
        print(f"⚡ Tổng số frames: {expected_frames} | Độ phân giải mục tiêu 4K: {target_w}x{target_h} (Real-CUGAN Native 2x)")

        half_frames = expected_frames // 2
        segments = [
            (0, half_frames, 0, os.path.join(output_dir, "_part_gpu0.mkv")),
            (half_frames, expected_frames - half_frames, 1, os.path.join(output_dir, "_part_gpu1.mkv"))
        ]

        for _, _, _, chunk_p in segments:
            if os.path.exists(chunk_p):
                try: os.remove(chunk_p)
                except Exception: pass

        manager = mp.Manager()
        return_dict = manager.dict()
        progress_queue = manager.Queue()
        processes = []

        start_time = time.time()

        for s_frame, n_frames, g_id, chunk_path in segments:
            p = mp.Process(
                target=_gpu_segment_worker,
                args=(video_input, s_frame, n_frames, target_w, target_h, fps, src_w, src_h, weights_path, model_key, g_id, chunk_path, return_dict, progress_queue, model_mode)
            )
            p.start()
            processes.append(p)

        completed_total = 0
        last_print_t = 0.0

        while completed_total < expected_frames:
            try:
                added = progress_queue.get(timeout=0.3)
                completed_total += added
            except Exception:
                if not any(p.is_alive() for p in processes):
                    break

            now = time.time()
            if (now - last_print_t) >= 0.5 or completed_total >= expected_frames:
                last_print_t = now
                elapsed = now - start_time
                speed_fps = completed_total / elapsed if elapsed > 0 else 0.0
                pct = (completed_total / expected_frames) * 100
                cur_sec = completed_total / fps if fps > 0 else 0
                cur_str = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
                tot_sec = expected_frames / fps if fps > 0 else 0
                tot_str = f"{int(tot_sec // 60):02d}:{int(tot_sec % 60):02d}"
                eta_sec = (expected_frames - completed_total) / speed_fps if speed_fps > 0 else 0
                eta_str = f"{int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"

                status_msg = f"⏳ {completed_total}/{expected_frames} ({pct:.1f}%) | {speed_fps:.2f} fps | {cur_str}/{tot_str} | ETA: {eta_str}"
                print(status_msg + "    ", end='\r', flush=True)

                if progress_callback:
                    try: progress_callback(pct / 100.0, desc=status_msg)
                    except Exception: pass

        for p in processes:
            p.join()

        elapsed = time.time() - start_time
        effective_fps = expected_frames / elapsed if elapsed > 0 else 0
        print(f"\n⚡ HOÀN THÀNH XỬ LÝ DUAL T4! Thời gian: {elapsed:.2f}s | Tốc độ hiệu dụng: {effective_fps:.2f} FPS!", flush=True)

        chunk_files = [seg[3] for seg in segments if os.path.exists(seg[3]) and os.path.getsize(seg[3]) > 1000]

        if len(chunk_files) >= 1:
            print("📦 Đang nối 2 nửa video và sao chép 100% Audio, Subtitle (.ass), Fonts...", flush=True)
            concat_txt = os.path.join(output_dir, f"_concat_{int(time.time())}.txt")
            with open(concat_txt, "w") as f:
                for c_path in chunk_files:
                    f.write(f"file '{os.path.abspath(c_path)}'\n")

            temp_concat = os.path.join(output_dir, f"_temp_concat_{int(time.time())}.mkv")
            concat_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_txt, '-c', 'copy', temp_concat]
            subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # BẢO TỒN 100% METADATA (VIDEO 4K + AUDIO GỐC + PHỤ ĐỀ MỀM + ATTACHMENT FONTS)
            mux_cmd = [
                'ffmpeg', '-y',
                '-i', temp_concat,
                '-i', video_input,
                '-c', 'copy',
                '-map', '0:v:0',
                '-map', '1:a?',
                '-map', '1:s?',
                '-map', '1:t?',
                video_output
            ]
            subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            for f_clean in [concat_txt, temp_concat] + chunk_files:
                if os.path.exists(f_clean):
                    try: os.remove(f_clean)
                    except Exception: pass

            if os.path.exists(video_output) and os.path.getsize(video_output) > 1000:
                print(f"\n✨ KẾT THÚC HOÀN HẢO! Tập phim 4K nằm tại: {video_output}", flush=True)
                if progress_callback:
                    try: progress_callback(1.0, desc="✨ Hoàn tất nâng cấp video 4K!")
                    except Exception: pass
                return video_output

    # LUỒNG GPU ĐƠN (KHI CHỈ CÓ 1 GPU HOẶC CHẠY KIỂM THỬ)
    model, is_pro = load_realcugan_model(model_key, weights_path, device, mode=model_mode)
    batch_size = 2 if (model_mode == "se" or src_h <= 720) else 1
    queue_size = 8

    ffmpeg_read_cmd = [
        'ffmpeg', '-y', '-threads', '0', '-i', video_input,
        '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
    ]
    process_read = subprocess.Popen(ffmpeg_read_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=32*1024*1024)

    temp_video_only = os.path.join(scratch_dir, f"_temp_v_{os.path.basename(video_output)}")
    ffmpeg_write_cmd = [
        'ffmpeg', '-y',
        '-threads', '0',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{target_w}x{target_h}', '-r', str(fps),
        '-i', '-',
        *encoder_flags,
        temp_video_only
    ]
    process_write = subprocess.Popen(ffmpeg_write_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=32*1024*1024)

    frame_size = src_w * src_h * 3
    idx = 0
    input_queue = Queue(maxsize=queue_size)
    output_queue = Queue(maxsize=queue_size)
    batch_bytes_target = batch_size * frame_size

    def reader_worker():
        try:
            while True:
                chunk = process_read.stdout.read(batch_bytes_target)
                if not chunk:
                    input_queue.put(None)
                    break
                while len(chunk) % frame_size != 0:
                    more = process_read.stdout.read(frame_size - (len(chunk) % frame_size))
                    if not more:
                        break
                    chunk += more
                if not chunk:
                    input_queue.put(None)
                    break
                input_queue.put(chunk)
        except Exception:
            input_queue.put(None)

    def writer_worker():
        try:
            while True:
                item = output_queue.get()
                if item is None: break
                try:
                    process_write.stdin.write(item)
                except Exception: pass
                output_queue.task_done()
        except Exception: pass

    reader_thread = threading.Thread(target=reader_worker, daemon=True)
    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    start_time = time.time()
    last_print_time = 0.0
    reader_thread.start()
    writer_thread.start()
    gc.disable()

    try:
        while True:
            chunk = input_queue.get()
            if chunk is None: break
            current_b = len(chunk) // frame_size
            if current_b == 0: break

            img_np_batch = np.frombuffer(chunk, dtype=np.uint8).reshape((current_b, src_h, src_w, 3))
            img_t = torch.from_numpy(img_np_batch)
            if device.type == 'cuda':
                img_t = img_t.to(device, non_blocking=True)
                img_t = img_t.permute(0, 3, 1, 2).to(torch.float16, non_blocking=True)
                if is_pro:
                    img_t.mul_(0.7 / 255.0).add_(0.15)
                else:
                    img_t.mul_(1.0 / 255.0)
                img_t = img_t.to(memory_format=torch.channels_last)
            elif device.type == 'mps':
                img_t = img_t.to(device)
                img_t = img_t.permute(0, 3, 1, 2).to(torch.float16)
                if is_pro:
                    img_t.mul_(0.7 / 255.0).add_(0.15)
                else:
                    img_t.mul_(1.0 / 255.0)
            else:
                img_t = img_t.to(device)
                img_t = img_t.permute(0, 3, 1, 2).float()
                if is_pro:
                    img_t.mul_(0.7 / 255.0).add_(0.15)
                else:
                    img_t.mul_(1.0 / 255.0)

            with torch.inference_mode():
                raw_out = model(img_t)
                if is_pro:
                    raw_out.sub_(0.15).mul_(1.0 / 0.7).clamp_(0.0, 1.0)
                else:
                    raw_out.clamp_(0.0, 1.0)

                if raw_out.shape[2] != target_h or raw_out.shape[3] != target_w:
                    raw_out = F.interpolate(raw_out, size=(target_h, target_w), mode='area')

                output = raw_out.mul_(255.0).round_().to(torch.uint8).permute(0, 2, 3, 1).contiguous()

            output_queue.put(memoryview(output.cpu().numpy()).cast('B'))
            
            idx += current_b

            now = time.time()
            if (now - last_print_time) >= 0.5 or (expected_frames and idx >= expected_frames):
                last_print_time = now
                elapsed_time = now - start_time
                speed_fps = idx / elapsed_time if elapsed_time > 0 else 0
                pct = (idx / expected_frames) * 100 if expected_frames else 0
                cur_sec = idx / fps if fps > 0 else 0
                cur_str = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
                tot_sec = expected_frames / fps if (expected_frames and fps > 0) else 0
                tot_str = f"{int(tot_sec // 60):02d}:{int(tot_sec % 60):02d}"
                eta_sec = (expected_frames - idx) / speed_fps if (expected_frames and speed_fps > 0) else 0
                eta_str = f"{int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"

                status_msg = f"⏳ {idx}/{expected_frames} ({pct:.1f}%) | {speed_fps:.2f} fps | {cur_str}/{tot_str} | ETA: {eta_str}"
                print(status_msg + "    ", end='\r', flush=True)

                if progress_callback and expected_frames:
                    try: progress_callback(pct / 100.0, desc=status_msg)
                    except Exception: pass

    finally:
        gc.enable()
        try: output_queue.put(None); writer_thread.join(timeout=5)
        except Exception: pass
        try: process_read.terminate()
        except Exception: pass
        try:
            if process_write.stdin and not process_write.stdin.closed:
                process_write.stdin.close()
            process_write.wait(timeout=5)
        except Exception: pass

        temp_final_mkv = os.path.join(scratch_dir, f"_final_{os.path.basename(video_output)}")
        if os.path.exists(temp_video_only) and os.path.getsize(temp_video_only) > 0:
            print("🔊 Ghép 100% Audio gốc, Subtitle (.ass) và Fonts vào MKV...", flush=True)
            mux_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_only,
                '-i', video_input,
                '-c', 'copy',
                '-map', '0:v:0',
                '-map', '1:a?',
                '-map', '1:s?',
                '-map', '1:t?',
                temp_final_mkv
            ]
            subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try: os.remove(temp_video_only)
            except Exception: pass

            print(f"☁️ Đang lưu tập phim 4K hoàn chỉnh vào Google Drive: '{video_output}'...", flush=True)
            shutil.move(temp_final_mkv, video_output)

        print(f"\n✨ KẾT THÚC HOÀN HẢO! Tập phim 4K nằm tại: {video_output}", flush=True)

    return video_output

def main():
    if len(sys.argv) < 2:
        print("❌ Lỗi: Vui lòng cung cấp link Magnet hoặc đường dẫn file video!")
        print("💡 Sử dụng: python3 upscale.py <magnet:... hoặc video.mkv> [cugan_conservative/cugan_no_denoise/cugan_denoise3x]")
        return
    
    video_input = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "cugan_conservative"
    upscale_video(video_input=video_input, model_name=model_name)

if __name__ == '__main__':
    main()
