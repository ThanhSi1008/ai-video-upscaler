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

### 1. Web UI (Máy cá nhân Mac / Windows / Google Colab)

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

---

### 2. Form Native UI (Dành riêng cho Kaggle Cloud - Chống sập Session 100%)

Run the following cell in a free **Kaggle Notebook (GPU T4 x2)**:

```python
# @title 🎬 AI Video Upscaler 4K - Nhập Thông Tin Video
# @markdown Dán link YouTube hoặc đường dẫn tệp video của bạn bên dưới:

video_input = "https://www.youtube.com/watch?v=OpdeWENZhUY" # @param {type:"string"}
codec_choice = "auto" # @param ["auto", "hevc_nvenc", "h264_nvenc", "libx264"]
keep_highest = False # @param {type:"boolean"}

# ------------------------------------------------------------
import os, sys, importlib
!apt-get update -qq && apt-get install -y ffmpeg -qq
!pip install -q --no-cache-dir https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz gradio

repo_dir = "/kaggle/working/ai-video-upscaler"
if os.path.exists(repo_dir):
    %cd {repo_dir}
    !git pull
else:
    !git clone https://github.com/ThanhSi1008/ai-video-upscaler.git {repo_dir}
    %cd {repo_dir}

if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import upscale
importlib.reload(upscale)

print("🚀 Bắt đầu quá trình nâng cấp video AI 4K...")
output_file = upscale.upscale_video(
    video_input=video_input,
    encoder_codec=codec_choice,
    keep_highest=keep_highest
)
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
