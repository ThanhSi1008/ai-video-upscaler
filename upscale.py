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

# Ẩn toàn bộ các dòng Warning hiển thị của Python & PyTorch Inductor
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TORCH_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_LOGS"] = "-inductor"

# --- 1. Thuật toán Sắp xếp Tự nhiên (Natural Sort) ---
def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

# --- 2. Kiến trúc SRVGGNetCompact chuyên dụng cho AnimeVideoV3 ---
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        return out + base

def get_device_and_codec(requested_codec="auto"):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        default_codec = 'h264_nvenc'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        default_codec = 'hevc_videotoolbox'
    else:
        device = torch.device('cpu')
        default_codec = 'libx264'

    if requested_codec == "auto" or not requested_codec:
        codec = default_codec
    else:
        codec = requested_codec
    return device, codec

# Worker tiến trình chạy phân luồng độc lập trên 1 GPU riêng biệt
def _gpu_segment_worker(video_input, start_frame, total_frames_to_process, target_w, target_h, fps, src_w, src_h, weights_path, encoder_codec, gpu_id, chunk_output_path, return_dict, progress_queue):
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.nn import functional as F
        from queue import Queue

        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4)
        state_dict = torch.load(weights_path, map_location='cpu')
        if 'params_ema' in state_dict: state_dict = state_dict['params_ema']
        elif 'params' in state_dict: state_dict = state_dict['params']
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        if device.type == 'cuda':
            model = model.half().to(memory_format=torch.channels_last)
            try: model = torch.compile(model, mode="default")
            except Exception: pass
        model = model.to(device)

        seek_time = start_frame / fps if (start_frame > 0 and fps > 0) else 0.0
        
        ffmpeg_read_cmd = ['ffmpeg', '-y']
        if seek_time > 0:
            ffmpeg_read_cmd.extend(['-ss', f"{seek_time:.4f}"])
        ffmpeg_read_cmd.extend([
            '-i', video_input,
            '-vframes', str(total_frames_to_process),
            '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
        ])
        process_read = subprocess.Popen(ffmpeg_read_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10*1024*1024)

        if os.path.exists(chunk_output_path):
            try: os.remove(chunk_output_path)
            except Exception: pass

        ffmpeg_write_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{target_w}x{target_h}', '-r', str(fps),
            '-i', '-'
        ]

        if "hevc" in encoder_codec:
            ffmpeg_write_cmd.extend(['-c:v', 'hevc_nvenc', '-preset', 'p1', '-pix_fmt', 'yuv420p'])
        elif "nvenc" in encoder_codec or encoder_codec == "auto":
            ffmpeg_write_cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'p1', '-pix_fmt', 'yuv420p'])
        elif "videotoolbox" in encoder_codec:
            ffmpeg_write_cmd.extend(['-c:v', encoder_codec, '-q:v', '65', '-pix_fmt', 'yuv420p'])
        else:
            ffmpeg_write_cmd.extend(['-c:v', 'libx264', '-crf', '18', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p'])

        ffmpeg_write_cmd.append(chunk_output_path)

        process_write = subprocess.Popen(ffmpeg_write_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10*1024*1024)

        frame_size = src_w * src_h * 3
        batch_size = 6 if src_h <= 720 else (4 if src_h <= 1080 else 2)
        queue_size = 12

        input_queue = Queue(maxsize=queue_size)
        output_queue = Queue(maxsize=queue_size)

        def reader_worker():
            try:
                for _ in range(total_frames_to_process):
                    in_bytes = process_read.stdout.read(frame_size)
                    if not in_bytes or len(in_bytes) != frame_size:
                        input_queue.put(None)
                        break
                    input_queue.put(in_bytes)
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
                        process_write.stdin.flush()
                    except Exception: pass
                    output_queue.task_done()
            except Exception: pass

        reader_thread = threading.Thread(target=reader_worker, daemon=True)
        writer_thread = threading.Thread(target=writer_worker, daemon=True)
        reader_thread.start()
        writer_thread.start()

        processed_cnt = 0
        while processed_cnt < total_frames_to_process:
            batch_bytes = []
            for _ in range(batch_size):
                item = input_queue.get()
                if item is None: break
                batch_bytes.append(item)
            if not batch_bytes: break

            current_b = len(batch_bytes)
            img_nps = [np.frombuffer(b, dtype=np.uint8).reshape((src_h, src_w, 3)) for b in batch_bytes]
            img_np_batch = np.stack(img_nps, axis=0)

            img_t = torch.from_numpy(img_np_batch).pin_memory().to(device, non_blocking=True)
            img_t = img_t.permute(0, 3, 1, 2).to(torch.float16, non_blocking=True).div(255.0)
            img_t = img_t.to(memory_format=torch.channels_last)

            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                output = model(img_t)
                if output.shape[2] != target_h or output.shape[3] != target_w:
                    output = F.interpolate(output, size=(target_h, target_w), mode='bilinear', align_corners=False)
                output = output.clamp(0, 1).mul(255.0).round().to(torch.uint8)

            output_np = output.cpu().numpy()
            output_np = np.transpose(output_np, (0, 2, 3, 1))

            for i in range(current_b):
                output_queue.put(output_np[i].tobytes())

            processed_cnt += current_b
            try: progress_queue.put(current_b)
            except Exception: pass

            if processed_cnt % 30 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        output_queue.put(None)
        writer_thread.join(timeout=10)

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
            print(f"⚠️ Cảnh báo đóng luồng ghi FFmpeg worker {gpu_id}: {e_w}")

        return_dict[gpu_id] = True
    except Exception as e:
        print(f"⚠️ Lỗi GPU worker {gpu_id}: {e}")
        return_dict[gpu_id] = False

# --- 5. Hàm xử lý upscale chính tích hợp Tự Động Dual GPU Multi-processing ---
def upscale_video(video_input, output_dir=None, encoder_codec="auto", keep_highest=False, progress_callback=None):
    is_youtube = "youtube.com" in video_input or "youtu.be" in video_input
    
    if output_dir is None:
        if os.path.exists('/kaggle/working'):
            output_dir = '/kaggle/working'
        else:
            output_dir = os.path.expanduser('~/Documents/mushoku-tensei')
    os.makedirs(output_dir, exist_ok=True)

    device, encoder_codec = get_device_and_codec(encoder_codec)
    print(f"🚀 Thiết bị tính toán được chọn: {device} | Codec: {encoder_codec}")

    temp_input_file = None
    if is_youtube:
        print("📥 Phát hiện liên kết YouTube. Bắt đầu tải video thô...")
        if progress_callback:
            progress_callback(0.01, desc="📥 Đang kết nối tải video từ YouTube...")
        try:
            import yt_dlp
            import glob
            
            for f in glob.glob('yt_temp_input*'):
                try: os.remove(f)
                except Exception: pass

            opts = {
                'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best/b',
                'outtmpl': 'yt_temp_input.%(ext)s',
                'merge_output_format': 'mp4',
                'quiet': True,
                'no_warnings': True,
                'noprogress': True,
                'geo_bypass': True,
                'geo_bypass_country': 'VN',
                'extractor_args': {'youtube': {'player_client': ['android']}},
                'socket_timeout': 10,
                'nocheckcertificate': True,
            }

            download_success = False
            video_title = "youtube_video"

            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(video_input, download=True)
                    video_title = info.get('title', 'youtube_video')
                    video_title = re.sub(r'[\\/*?:"<>|]', "", video_title)
                downloaded = glob.glob('yt_temp_input.*')
                if downloaded and os.path.getsize(downloaded[0]) > 100000:
                    download_success = True
                    temp_input_file = downloaded[0]
            except Exception as e1:
                print(f"⚠️ Thử lại với luồng mweb client: {e1}")
                opts['extractor_args'] = {'youtube': {'player_client': ['mweb', 'android']}}
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(video_input, download=True)
                        video_title = info.get('title', 'youtube_video')
                        video_title = re.sub(r'[\\/*?:"<>|]', "", video_title)
                    downloaded = glob.glob('yt_temp_input.*')
                    if downloaded and os.path.getsize(downloaded[0]) > 100000:
                        download_success = True
                        temp_input_file = downloaded[0]
                except Exception as e2:
                    print(f"❌ Không thể giải mã video YouTube: {e2}")

            if not download_success:
                raise Exception("Không thể tải video từ YouTube. Vui lòng tải file video trực tiếp từ máy của bạn.")

            video_input = temp_input_file
            video_output = os.path.join(output_dir, f"{video_title}_upscaled.mp4")
            print(f"✅ Tải thành công video: '{video_title}'. Bắt đầu chạy upscale...")
        except Exception as e:
            print(f"❌ Lỗi trong quá trình tải YouTube: {e}")
            raise Exception(f"Lỗi tải YouTube: {str(e)}")
    else:
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"Không tìm thấy file video nguồn '{video_input}'!")
        video_base = os.path.basename(os.path.splitext(video_input)[0])
        video_output = os.path.join(output_dir, f"{video_base}_upscaled.mp4")

    # XÓA FILE KẾT QUẢ CŨ NẾU TỒN TẠI ĐỂ LUÔN GHI ĐÈ (OVERRIDE) BẰNG FILE MỚI
    if os.path.exists(video_output):
        try: os.remove(video_output)
        except Exception: pass

    # Trích xuất siêu dữ liệu qua ffprobe
    try:
        fps_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 \"{video_input}\""
        fps_res = subprocess.check_output(fps_cmd, shell=True).decode().strip()
        fps = eval(fps_res) if '/' in fps_res else float(fps_res)
        
        res_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 \"{video_input}\""
        src_res = subprocess.check_output(res_cmd, shell=True).decode().strip()
        src_w, src_h = map(int, src_res.split('x'))
    except Exception as e:
        print(f"⚠️ Thất bại khi phân tích siêu dữ liệu video: {e}")
        fps, src_w, src_h = 30.0, 1920, 1080
        
    print(f"ℹ️ Cấu hình gốc phát hiện: {src_w}x{src_h} @ {fps:.3f} FPS")

    # Trọng số AI
    weights_path = "realesr-animevideov3.pth"
    if not os.path.exists(weights_path):
        print("📥 Tự động tải mô hình phục hồi chuyên dụng từ kho lưu trữ GitHub...")
        if progress_callback:
            progress_callback(0.02, desc="📥 Đang tải weights Real-ESRGAN...")
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        try:
            urllib.request.urlretrieve(url, weights_path)
            print("✅ Đã đồng bộ trọng số mạng Anime thành công!")
        except Exception as e:
            print(f"❌ Lỗi hạ tầng mạng: {e}")
            raise e

    expected_frames = None
    try:
        frames_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 \"{video_input}\""
        frames_res = subprocess.check_output(frames_cmd, shell=True).decode().strip()
        if frames_res.isdigit():
            expected_frames = int(frames_res)
    except Exception as e:
        print(f"⚠️ Không thể đọc số lượng frame dự kiến: {e}")

    # ĐỘ PHÂN GIẢI MỤC TIÊU: CHUẨN HÓA VỀ 4K ULTRA-HD (3840x2160) ĐỂ TRÁNH VƯỢT GIỚI HẠN PHẦN CỨNG NVENC H.264 (4096x4096)
    aspect_ratio = src_w / src_h
    if keep_highest and (src_w * 4 <= 4096) and (src_h * 4 <= 4096):
        target_w, target_h = src_w * 4, src_h * 4
    else:
        if aspect_ratio >= (16 / 9):
            target_w = 3840
            target_h = int(3840 / aspect_ratio)
        else:
            target_h = 2160
            target_w = int(2160 * aspect_ratio)
    target_w = (target_w // 2) * 2
    target_h = (target_h // 2) * 2

    num_cuda_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # NẾU CÓ DUAL GPU T4 x2 TRÊN KAGGLE: KÍCH HOẠT PHÂN LUỒNG ĐỘC LẬP TỐC ĐỘ 14 - 18 FPS + THANH TIẾN ĐỘ THỜI GIAN THỰC
    if num_cuda_gpus >= 2 and expected_frames and expected_frames > 100:
        try: mp.set_start_method('spawn', force=True)
        except Exception: pass

        print(f"🔥 BẮT ĐẦU CHẠY PHÂN LUỒNG ĐỘC LẬP DUAL GPU: KÍCH HOẠT CẢ {num_cuda_gpus} CARDS NVIDIA T4 CÙNG LÚC!")
        print(f"⚡ Tổng số frames: {expected_frames} | Độ phân giải mục tiêu 4K Ultra-HD: {target_w}x{target_h}")

        half_frames = expected_frames // 2
        segments = [
            (0, half_frames, 0, os.path.join(output_dir, "_part_gpu0.mp4")),
            (half_frames, expected_frames - half_frames, 1, os.path.join(output_dir, "_part_gpu1.mp4"))
        ]

        # Xóa tệp tạm cũ nếu có
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
                args=(video_input, s_frame, n_frames, target_w, target_h, fps, src_w, src_h, weights_path, encoder_codec, g_id, chunk_path, return_dict, progress_queue)
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
        print(f"\n⚡ HOÀN THÀNH XỬ LÝ SONG SONG DUAL GPU! Thời gian: {elapsed:.2f}s | Tốc độ hiệu dụng: {effective_fps:.2f} FPS!", flush=True)

        if progress_callback:
            try: progress_callback(0.98, desc="📦 Đang nối 2 đoạn video và ghép âm thanh gốc...")
            except Exception: pass

        chunk_files = [seg[3] for seg in segments if os.path.exists(seg[3]) and os.path.getsize(seg[3]) > 1000]

        if len(chunk_files) >= 1:
            print("📦 Đang nối các đoạn video và ghép âm thanh gốc...", flush=True)
            concat_txt = os.path.join(output_dir, f"_concat_{int(time.time())}.txt")
            with open(concat_txt, "w") as f:
                for c_path in chunk_files:
                    f.write(f"file '{os.path.abspath(c_path)}'\n")

            temp_concat = os.path.join(output_dir, f"_temp_concat_{int(time.time())}.mp4")
            if os.path.exists(temp_concat):
                try: os.remove(temp_concat)
                except Exception: pass

            concat_cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_txt, '-c', 'copy', temp_concat]
            subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if os.path.exists(video_output):
                try: os.remove(video_output)
                except Exception: pass

            mux_cmd = [
                'ffmpeg', '-y',
                '-i', temp_concat,
                '-i', video_input,
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-map', '0:v:0',
                '-map', '1:a?',
                video_output
            ]
            subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if not os.path.exists(video_output) or os.path.getsize(video_output) < 1000:
                print("⚠️ Đang sử dụng phương án sao chép trực tiếp...")
                if os.path.exists(temp_concat) and os.path.getsize(temp_concat) > 1000:
                    shutil.copy(temp_concat, video_output)
                elif chunk_files:
                    shutil.copy(chunk_files[0], video_output)

            for f_clean in [concat_txt, temp_concat] + chunk_files:
                if os.path.exists(f_clean):
                    try: os.remove(f_clean)
                    except Exception: pass

            if os.path.exists(video_output) and os.path.getsize(video_output) > 1000:
                print(f"\n✨ KẾT THÚC HOÀN HẢO! Video 4K nằm tại: {video_output}", flush=True)
                if progress_callback:
                    try: progress_callback(1.0, desc="✨ Hoàn tất nâng cấp video 4K!")
                    except Exception: pass
                return video_output

        print("⚠️ Cảnh báo: Luồng Dual GPU chưa tạo được file kết quả. Tự động chuyển sang luồng GPU đơn an toàn...")

    # LUỒNG CHẠY GPU ĐƠN THƯỜNG (KHI CHỈ CÓ 1 GPU HOẶC DUAL GPU CẦN DỰ PHÒNG AN TOÀN)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4)
    state_dict = torch.load(weights_path, map_location='cpu')
    if 'params_ema' in state_dict: state_dict = state_dict['params_ema']
    elif 'params' in state_dict: state_dict = state_dict['params']
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if device.type == 'cuda':
        model = model.half().to(memory_format=torch.channels_last)
        batch_size = 6 if src_h <= 720 else (4 if src_h <= 1080 else 2)
        queue_size = 12
        try: model = torch.compile(model, mode="default")
        except Exception: pass
    else:
        if device.type == 'mps': model = model.half()
        batch_size = 1
        queue_size = 2

    model = model.to(device)

    ffmpeg_read_cmd = [
        'ffmpeg', '-y', '-i', video_input,
        '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
    ]
    process_read = subprocess.Popen(ffmpeg_read_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10*1024*1024)

    temp_video_only = os.path.join(output_dir, f"_temp_v_{os.path.basename(video_output)}")
    if os.path.exists(temp_video_only):
        try: os.remove(temp_video_only)
        except Exception: pass

    ffmpeg_write_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{target_w}x{target_h}', '-r', str(fps),
        '-i', '-',
        '-c:v', encoder_codec
    ]

    if "videotoolbox" in encoder_codec:
        quality_opts = ['-q:v', '65']
    elif "nvenc" in encoder_codec:
        quality_opts = ['-preset', 'p1', '-tune', 'll', '-rc', 'constqp', '-qp', '20']
    else:
        quality_opts = ['-crf', '18', '-preset', 'ultrafast']

    ffmpeg_write_cmd.extend(quality_opts)
    ffmpeg_write_cmd.extend(['-pix_fmt', 'yuv420p', temp_video_only])
    
    process_write = subprocess.Popen(ffmpeg_write_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10*1024*1024)

    frame_size = src_w * src_h * 3
    idx = 0

    input_queue = Queue(maxsize=queue_size)
    output_queue = Queue(maxsize=queue_size)

    def reader_worker():
        try:
            while True:
                in_bytes = process_read.stdout.read(frame_size)
                if not in_bytes or len(in_bytes) != frame_size:
                    input_queue.put(None)
                    break
                input_queue.put(in_bytes)
        except Exception: input_queue.put(None)

    def writer_worker():
        try:
            while True:
                item = output_queue.get()
                if item is None: break
                try:
                    process_write.stdin.write(item)
                    process_write.stdin.flush()
                except Exception: pass
                output_queue.task_done()
        except Exception: pass

    reader_thread = threading.Thread(target=reader_worker, daemon=True)
    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    start_time = time.time()
    last_print_time = 0.0
    reader_thread.start()
    writer_thread.start()

    try:
        while True:
            batch_bytes = []
            for _ in range(batch_size):
                item = input_queue.get()
                if item is None: break
                batch_bytes.append(item)
            if not batch_bytes: break
                
            current_b = len(batch_bytes)
            img_nps = [np.frombuffer(b, dtype=np.uint8).reshape((src_h, src_w, 3)) for b in batch_bytes]
            img_np_batch = np.stack(img_nps, axis=0)
            
            if device.type == 'cuda':
                img_t = torch.from_numpy(img_np_batch).pin_memory().to(device, non_blocking=True)
                img_t = img_t.permute(0, 3, 1, 2).to(torch.float16, non_blocking=True).div(255.0)
                img_t = img_t.to(memory_format=torch.channels_last)
            else:
                img_t = torch.from_numpy(img_np_batch).to(device)
                dtype = torch.float16 if device.type == 'mps' else torch.float32
                img_t = img_t.permute(0, 3, 1, 2).to(dtype).div(255.0)

            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                output = model(img_t)
                if output.shape[2] != target_h or output.shape[3] != target_w:
                    output = F.interpolate(output, size=(target_h, target_w), mode='bilinear', align_corners=False)
                output = output.clamp(0, 1).mul(255.0).round().to(torch.uint8)

            output_np = output.cpu().numpy()
            output_np = np.transpose(output_np, (0, 2, 3, 1))
            
            for i in range(current_b):
                output_queue.put(output_np[i].tobytes())
            
            idx += current_b
            if idx % 30 == 0:
                gc.collect()
                if device.type == 'cuda': torch.cuda.empty_cache()
                elif device.type == 'mps': torch.mps.empty_cache()

            now = time.time()
            if (now - last_print_time) >= 1.0 or (expected_frames and idx >= expected_frames):
                last_print_time = now
                elapsed_time = now - start_time
                speed_fps = idx / elapsed_time if elapsed_time > 0 else 0
                current_video_time = idx / fps if fps > 0 else 0
                video_time_str = f"{int(current_video_time // 60):02d}:{int(current_video_time % 60):02d}"
                
                if expected_frames:
                    total_video_time = expected_frames / fps if fps > 0 else 0
                    total_video_time_str = f"{int(total_video_time // 60):02d}:{int(total_video_time % 60):02d}"
                    remaining_frames = expected_frames - idx
                    eta_time = remaining_frames / speed_fps if speed_fps > 0 else 0
                    eta_str = f"{int(eta_time // 60):02d}:{int(eta_time % 60):02d}"
                    pct = (idx / expected_frames) * 100
                    status_msg = f"⏳ {idx}/{expected_frames} ({pct:.1f}%) | {speed_fps:.2f} fps | {video_time_str}/{total_video_time_str} | ETA: {eta_str}"
                    print(status_msg + "    ", end='\r', flush=True)

                    if progress_callback:
                        try: progress_callback(pct / 100.0, desc=status_msg)
                        except Exception: pass
                else:
                    status_msg = f"⏳ {idx} frames | {speed_fps:.2f} fps | {video_time_str}"
                    print(status_msg + "    ", end='\r', flush=True)

    finally:
        print("\n", flush=True)
        print("🎬 Hoàn tất luồng xử lý khung hình...", flush=True)
        try: output_queue.put(None); writer_thread.join(timeout=5)
        except Exception: pass

        try:
            if process_read.poll() is None:
                process_read.terminate()
                process_read.wait(timeout=2)
        except Exception:
            try: process_read.kill()
            except Exception: pass

        try:
            if process_write.stdin and not process_write.stdin.closed:
                process_write.stdin.close()
            if process_write.poll() is None:
                process_write.wait(timeout=3)
        except Exception:
            try: process_write.kill()
            except Exception: pass

        if os.path.exists(temp_video_only) and os.path.getsize(temp_video_only) > 0:
            print("🔊 Đang ghép âm thanh gốc và xuất video 4K hoàn chỉnh 100%...", flush=True)
            if os.path.exists(video_output):
                try: os.remove(video_output)
                except Exception: pass
                
            mux_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_only,
                '-i', video_input,
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-map', '0:v:0',
                '-map', '1:a?',
                video_output
            ]
            subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if not os.path.exists(video_output) or os.path.getsize(video_output) < 1000:
                print("⚠️ Đang sử dụng phương án sao chép trực tiếp...")
                shutil.copy(temp_video_only, video_output)

            if os.path.exists(temp_video_only):
                try: os.remove(temp_video_only)
                except Exception: pass

        if is_youtube and temp_input_file and os.path.exists(temp_input_file):
            try: os.remove(temp_input_file)
            except Exception: pass

        print(f"\n✨ KẾT THÚC HOÀN HẢO! Video 4K nằm tại: {video_output}", flush=True)

    return video_output

def main():
    if len(sys.argv) < 2:
        print("❌ Lỗi: Vui lòng cung cấp đường dẫn video input!")
        print("💡 Sử dụng: python3 upscale.py <video_input.mp4/youtube_url> [auto/libx264/hevc_nvenc/hevc_videotoolbox] [keep/scale]")
        return
    
    video_input = sys.argv[1]
    encoder_codec = sys.argv[2] if len(sys.argv) > 2 else "auto"
    keep_highest = (sys.argv[3] == "keep") if len(sys.argv) > 3 else False
    
    upscale_video(video_input=video_input, encoder_codec=encoder_codec, keep_highest=keep_highest)

if __name__ == '__main__':
    main()
