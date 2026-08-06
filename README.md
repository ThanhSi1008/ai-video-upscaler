# AI Video Upscaler & 4K Super-Resolution

An AI-powered video resolution upscaler and enhancement web application using **Real-ESRGAN (SRVGGNetCompact)** and **FFmpeg**.

---

## 🌟 Features

- **AI Super-Resolution**: Upscale anime and general videos to 4K resolution using Real-ESRGAN (`realesr-animevideov3`).
- **Web UI**: Modern, minimal Gradio web interface for uploading video files or processing YouTube video links directly.
- **Hardware Acceleration**: Automatic hardware detection (NVIDIA CUDA `nvenc`, Apple Silicon `videotoolbox`, or CPU `libx264`).
- **Zero-Disk Frame Pipes**: High-performance streaming processing using FFmpeg memory pipes (no temporary image frame files created on disk).
- **Deploy Anywhere**: Pre-configured `Dockerfile` for containerized environments, Kaggle Notebooks, or local deployment.

---

## 🚀 Quick Start

### 1. Local Setup

```bash
# Clone the repository
git clone https://github.com/ThanhSi1008/ai-video-upscaler.git
cd ai-video-upscaler

# Install dependencies
pip install -r requirements.txt

# Launch Web UI
python3 app.py
```

Open your browser at `http://localhost:7860`.

### 2. Command Line Interface (CLI)

```bash
python3 upscale.py <video_input_or_youtube_url> [auto/libx264/hevc_nvenc] [keep/scale]
```

---

## ☁️ Free GPU Deployment (Kaggle)

Run the following cell in a free **Kaggle Notebook (GPU T4 x2)** to launch the Web UI:

```python
# 1. Cài đặt FFmpeg và các thư viện Python
!apt-get update -qq && apt-get install -y ffmpeg -qq
!pip install -q gradio yt-dlp torch torchvision pillow numpy

# 2. Tải/Cập nhật mã nguồn mới nhất từ GitHub
import os, sys, subprocess, time, importlib

repo_dir = "/kaggle/working/ai-video-upscaler"
if os.path.exists(repo_dir):
    %cd {repo_dir}
    !git pull
else:
    !git clone https://github.com/ThanhSi1008/ai-video-upscaler.git {repo_dir}
    %cd {repo_dir}

if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

# 3. Khởi chạy Web UI Server dưới nền
import upscale, app
importlib.reload(upscale)
importlib.reload(app)

app.app.queue().launch(server_name="0.0.0.0", server_port=7860, share=False, prevent_thread_lock=True)

# 4. Tạo đường dẫn Web UI Public an toàn 100% qua SSH Tunnel (Chống sập Kaggle session)
print("🌐 Đang tạo đường dẫn Web UI Public cho bạn...")
proc = subprocess.Popen(
    ['ssh', '-o', 'StrictHostKeyChecking=no', '-R', '80:localhost:7860', 'nokey@localhost.run'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

start = time.time()
while time.time() - start < 15:
    line = proc.stdout.readline()
    if not line:
        break
    if 'lhr.life' in line or 'lhrtunnel' in line or 'tunneled' in line:
        urls = [w for w in line.split() if 'https://' in w]
        if urls:
            print("\n🎉 MỞ WEB UI CỦA BẠN TẠI ĐƯỜNG DẪN BÊN DƯỚI:")
            print(f"👉 {urls[0]}\n")
            break
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
