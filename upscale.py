import os
import sys
import re
import subprocess
import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
import time
import threading
from queue import Queue

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

# --- 3. Giải thuật Tiling tối ưu luồng tính toán bất đồng bộ trên GPU ---
def upscale_tiled(model, img_t, device, tile=800, pad=16, scale=4):
    _, c, h, w = img_t.shape
    out = torch.zeros((1, c, h * scale, w * scale), device=device, dtype=img_t.dtype)

    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, py1 = max(y0 - pad, 0), min(y1 + pad, h)
            px0, px1 = max(x0 - pad, 0), min(x1 + pad, w)

            tile_out = model(img_t[:, :, py0:py1, px0:px1])

            ct, cl = (y0 - py0) * scale, (x0 - px0) * scale
            ch, cw = (y1 - y0) * scale, (x1 - x0) * scale
            
            out[:, :, y0*scale:y1*scale, x0*scale:x1*scale] = \
                tile_out[:, :, ct:ct+ch, cl:cl+cw]
    return out

def get_device_and_codec(requested_codec="auto"):
    if torch.cuda.is_available():
        device = torch.device('cuda')
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

# --- 4. Hàm xử lý upscale chính dùng được từ Python API & Gradio UI ---
def upscale_video(video_input, output_dir=None, encoder_codec="auto", keep_highest=False, progress_callback=None):
    is_youtube = "youtube.com" in video_input or "youtu.be" in video_input
    
    if output_dir is None:
        output_dir = os.path.expanduser('~/Documents/mushoku-tensei')
    os.makedirs(output_dir, exist_ok=True)

    device, encoder_codec = get_device_and_codec(encoder_codec)
    print(f"🚀 Thiết bị tính toán được chọn: {device} | Codec: {encoder_codec}")

    temp_input_file = None
    if is_youtube:
        print("📥 Phát hiện liên kết YouTube. Bắt đầu tải video thô...")
        if progress_callback:
            progress_callback(0.01, desc="📥 Đang tải video từ YouTube...")
        try:
            import yt_dlp
            import glob
            
            for f in glob.glob('yt_temp_input.*'):
                try:
                    os.remove(f)
                except Exception:
                    pass
                    
            ydl_opts = {
                'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
                'outtmpl': 'yt_temp_input.%(ext)s',
                'merge_output_format': 'mp4',
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_input, download=True)
                video_title = info.get('title', 'youtube_video')
                video_title = re.sub(r'[\\/*?:"<>|]', "", video_title)
                
            downloaded = glob.glob('yt_temp_input.*')
            if not downloaded:
                raise Exception("Không thể tải video từ YouTube!")
            temp_input_file = downloaded[0]
            video_input = temp_input_file
            video_output = os.path.join(output_dir, f"{video_title}_upscaled.mp4")
            print(f"✅ Tải thành công video: '{video_title}'. Bắt đầu chạy upscale...")
        except Exception as e:
            print(f"❌ Lỗi trong quá trình tải YouTube: {e}")
            raise e
    else:
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"Không tìm thấy file video nguồn '{video_input}'!")
        video_base = os.path.basename(os.path.splitext(video_input)[0])
        video_output = os.path.join(output_dir, f"{video_base}_upscaled.mp4")

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

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4)
    state_dict = torch.load(weights_path, map_location='cpu')
    
    if 'params_ema' in state_dict: 
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict: 
        state_dict = state_dict['params']
        
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if device.type in ['mps', 'cuda']:
        model = model.half()
    model = model.to(device)

    expected_frames = None
    try:
        frames_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 \"{video_input}\""
        frames_res = subprocess.check_output(frames_cmd, shell=True).decode().strip()
        if frames_res.isdigit():
            expected_frames = int(frames_res)
    except Exception as e:
        print(f"⚠️ Không thể đọc số lượng frame dự kiến: {e}")

    # Đọc ffmpeg pipes
    print("🎞️ Khởi tạo luồng giải mã video trực tiếp từ FFmpeg...")
    ffmpeg_read_cmd = ['ffmpeg', '-y']
    if device.type == 'mps':
        ffmpeg_read_cmd.extend(['-hwaccel', 'videotoolbox'])
    elif device.type == 'cuda':
        ffmpeg_read_cmd.extend(['-hwaccel', 'cuda'])
    
    ffmpeg_read_cmd.extend([
        '-i', video_input,
        '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
    ])
    
    try:
        process_read = subprocess.Popen(ffmpeg_read_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception:
        # Fallback không dùng -hwaccel nếu lỗi
        ffmpeg_read_cmd = [
            'ffmpeg', '-y', '-i', video_input,
            '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
        ]
        process_read = subprocess.Popen(ffmpeg_read_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    upscaled_w = src_w * 4
    upscaled_h = src_h * 4

    if keep_highest:
        target_w, target_h = upscaled_w, upscaled_h
    else:
        aspect_ratio = src_w / src_h
        if aspect_ratio >= (16 / 9):
            target_w = 3840
            target_h = int(3840 / aspect_ratio)
        else:
            target_h = 2160
            target_w = int(2160 * aspect_ratio)
        
        target_w = (target_w // 2) * 2
        target_h = (target_h // 2) * 2
        
        if target_w > upscaled_w or target_h > upscaled_h:
            target_w, target_h = upscaled_w, upscaled_h

    print(f"🎬 Khởi tạo luồng mã hóa video đầu ra ({target_w}x{target_h} @ {fps:.3f} FPS)...")
    ffmpeg_write_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{target_w}x{target_h}', '-r', str(fps),
        '-i', '-',
        '-i', video_input,
    ]
        
    if "videotoolbox" in encoder_codec:
        quality_opts = ['-q:v', '65']
    elif "nvenc" in encoder_codec:
        quality_opts = ['-cq', '20', '-preset', 'p4']
    else:
        quality_opts = ['-crf', '18']

    ffmpeg_write_cmd.extend(['-c:v', encoder_codec])
    ffmpeg_write_cmd.extend(quality_opts)
    ffmpeg_write_cmd.extend([
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a?', video_output
    ])
    
    try:
        process_write = subprocess.Popen(ffmpeg_write_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"⚠️ Codec {encoder_codec} không hỗ trợ, fallback sang libx264...")
        encoder_codec = "libx264"
        ffmpeg_write_cmd[ffmpeg_write_cmd.index('-c:v') + 1] = encoder_codec
        process_write = subprocess.Popen(ffmpeg_write_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_size = src_w * src_h * 3
    idx = 0
    print("🚀 Bắt đầu quá trình AI Super-Resolution...")
    
    input_queue = Queue(maxsize=4)
    output_queue = Queue(maxsize=4)

    def reader_worker():
        try:
            while True:
                in_bytes = process_read.stdout.read(frame_size)
                if not in_bytes or len(in_bytes) != frame_size:
                    input_queue.put(None)
                    break
                input_queue.put(in_bytes)
        except Exception:
            input_queue.put(None)

    def writer_worker():
        try:
            while True:
                item = output_queue.get()
                if item is None:
                    break
                process_write.stdin.write(item)
                output_queue.task_done()
        except Exception:
            pass

    reader_thread = threading.Thread(target=reader_worker, daemon=True)
    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    
    start_time = time.time()
    reader_thread.start()
    writer_thread.start()

    failed_log_path = "failed_frames.txt"

    try:
        while True:
            in_bytes = input_queue.get()
            if in_bytes is None:
                break
                
            idx += 1
            
            try:
                img_np = np.frombuffer(in_bytes, dtype=np.uint8).reshape((src_h, src_w, 3)).copy()
                img_t = torch.from_numpy(img_np).to(device)
                dtype = torch.float16 if (device.type in ['mps', 'cuda']) else torch.float32
                img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(dtype).div(255.0)

                with torch.inference_mode():
                    if src_h <= 1920 and src_w <= 1920:
                        output = model(img_t)
                    else:
                        output = upscale_tiled(model, img_t, device, tile=1920, pad=16, scale=4)
                        
                    if output.shape[2] != target_h or output.shape[3] != target_w:
                        output = F.interpolate(output, size=(target_h, target_w), mode='bicubic', align_corners=False)

                    output = output.squeeze(0).clamp(0, 1).mul(255.0).round().to(torch.uint8)

                output_np = output.cpu().numpy()
                output_np = np.transpose(output_np, (1, 2, 0))
                output_queue.put(output_np.tobytes())
                
            except Exception as e:
                with open(failed_log_path, "a") as log_file:
                    log_file.write(f"Frame_{idx} -> {str(e)}\n")
            
            elapsed_time = time.time() - start_time
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
                    progress_callback(pct / 100.0, desc=status_msg)
            else:
                status_msg = f"⏳ {idx} frames | {speed_fps:.2f} fps | {video_time_str}"
                print(status_msg + "    ", end='\r', flush=True)
                if progress_callback:
                    progress_callback(None, desc=status_msg)

            if device.type == 'mps' and idx % 30 == 0:
                torch.mps.empty_cache()
            elif device.type == 'cuda' and idx % 30 == 0:
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n⚠️ Quá trình chạy bị ngắt bởi người dùng!")
        
    finally:
        print("\n🎬 Đang hoàn tất đóng gói dữ liệu và đồng bộ âm thanh gốc...")
        try:
            output_queue.put(None)
            writer_thread.join(timeout=10)
        except Exception:
            pass
            
        try:
            process_read.stdout.close()
            process_read.wait()
        except Exception:
            pass
            
        try:
            if process_write.stdin:
                process_write.stdin.close()
            process_write.wait()
        except Exception:
            pass
        
        if is_youtube and temp_input_file and os.path.exists(temp_input_file):
            try:
                os.remove(temp_input_file)
                print("🧹 Đã dọn dẹp file YouTube tạm thời.")
            except Exception as e:
                print(f"⚠️ Cảnh báo không thể xóa file tạm: {e}")
        
        print(f"\n✨ KẾT THÚC HOÀN HẢO! Video Anime ở chuẩn {target_w}x{target_h} @ {fps:.3f} FPS nằm tại: {video_output}")

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
