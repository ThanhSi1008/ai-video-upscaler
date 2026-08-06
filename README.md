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

Run the following cell in a free **Kaggle Notebook (GPU T4 x2)**:

```python
# 1. Cài đặt FFmpeg và thư viện Python
!apt-get update -qq && apt-get install -y ffmpeg -qq
!pip install -q gradio yt-dlp torch torchvision pillow numpy

# 2. Tải mã nguồn dự án từ GitHub
import os
if not os.path.exists('/kaggle/working/ai-video-upscaler'):
    !git clone https://github.com/ThanhSi1008/ai-video-upscaler.git /kaggle/working/ai-video-upscaler

%cd /kaggle/working/ai-video-upscaler

# 3. Khởi chạy Web UI
import app
app.app.queue().launch(share=True)
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
