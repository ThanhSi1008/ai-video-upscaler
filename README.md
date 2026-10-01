---
title: AI Video Upscaler 4K
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎬 AI Video Upscaler 4K & Super-Resolution Master Suite

[![Deploy to Spaces](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deploy-to-spaces-lg.svg)](https://huggingface.co/new-space?template=ThanhSi1008/ai-video-upscaler)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)

An AI-powered video super-resolution platform optimized for **Native 2x Super-Resolution (AnimeJaNai V3)**, **HEVC 10-bit Master Quality NVENC**, **Automatic Segment Checkpointing & Resilient Resume**, **High-speed Magnet Ingestion (`aria2c`)**, **Full Audio/Subtitle/Font Preservation**, and **NVIDIA Multi-GPU Parallel Processing**.

---

## 🌟 Key Features

- **⚡ NGUỒN B: AnimeJaNai V3 Compact (Native 2x)**: Specially trained for **WEB-DL Gốc** (SubsPlease / Erai-raws / Crunchyroll / Netflix 1080p). Cleans streaming ringing and 8-bit banding while keeping line-art crisp. Ultra-fast (~12–15 FPS on Colab T4, ~25–40 mins per 24-minute episode).
- **⚡ NGUỒN A: AnimeJaNai V3 Sharp (Native 2x)**: Specially tuned for **BDRip 10-bit (Hi10P / Main 10)**. Preserves razor-sharp line-art and clean gradients without watercolor artifacts (~12–15 FPS on Colab T4).
- **🛡️ Google Colab Free-Proof Auto-Checkpoint & Resume**: Video is processed in resilient 2,400-frame segments saved directly to Google Drive. If a Colab session disconnects, it resumes instantly from the exact missing segment without losing hours of work!
- **🎨 HEVC 10-Bit Color (`yuv420p10le`)**: Master Quality hardware encoding with `-cq 18 -profile:v main10 -spatial-aq 1` completely eliminating color banding on 4K HDR displays.
- **📝 100% Subtitle, Font & Audio Preservation**: Full bit-exact copying of all original Japanese/English audio tracks, styled ASS/SSA subtitles, embedded font attachments (.ttf), and chapters into `.mkv`.
- **🧲 Direct Magnet / Torrent Ingestion**: Powered by `aria2c` multi-connection downloading (downloading a 1.4 GB episode in ~30 seconds on Kaggle).
- **🚀 Dual-GPU Multi-Processing Acceleration**: Dynamically distributes video segments in parallel across multi-GPU setups (e.g. Dual NVIDIA T4 on Kaggle).
- **🔍 Native Resolution Quality Checker**: Built-in independent tool to verify whether a video is true native 1080p or upscaled from 720p / ~810p / ~878p (see [docs/README_CHECK_NATIVE.md](docs/README_CHECK_NATIVE.md)).

---

## 🔍 Native Resolution Quality Checker (Tools)

Check whether your video is genuine Native 1080p or just an upscaled 720p file before running AI super-resolution:

```bash
# Analyze video directly (auto-detects sharpest keyframe)
python3 check_native.py samples/my_video.mkv

# Or launch standalone Web UI on port 7865
python3 check_native.py --web
```
👉 Full documentation and mathematical details: [docs/README_CHECK_NATIVE.md](docs/README_CHECK_NATIVE.md).

---

## 🚀 Deployment Options

### Option A: 1-Click Deploy to Hugging Face Spaces (Always-On / Easy Demo)

Click the badge below to duplicate this app directly into your Hugging Face Spaces account:

[![Deploy to Spaces](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deploy-to-spaces-lg.svg)](https://huggingface.co/new-space?template=ThanhSi1008/ai-video-upscaler)

> 💡 **Note**: Free Spaces run on CPU. For maximum GPU acceleration, upgrade the Space hardware to T4 / A10G GPU in Space Settings.

### Option B: Free Google Colab Deployment (1-Click T4 GPU - No Phone Verification)

Run directly on Google Colab with free NVIDIA T4 GPU acceleration:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhSi1008/ai-video-upscaler/blob/main/notebooks/Google_Colab_RealCUGAN_4K.ipynb)

1. Click the badge above to open the official notebook in Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Mount Google Drive and run the cells. Upscaled 4K videos save directly to your Google Drive!

---

### Option C: Free Kaggle Notebook Deployment (Dual NVIDIA T4 GPUs ~16+ FPS)

Run the following cell inside a free **Kaggle GPU Notebook** (with Accelerator set to **GPU T4 x2**):

```python
# @title 🎬 AI Video Upscaler 4K - Ultra High Speed WebUI (Real-CUGAN Pro)
import os, sys

!apt-get update -qq && apt-get install -y ffmpeg aria2 -qq
!pip install -q --no-cache-dir gradio torch torchvision

repo_dir = "/kaggle/working/ai-video-upscaler"
if os.path.exists(repo_dir):
    %cd {repo_dir}
    !git pull
else:
    !git clone https://github.com/ThanhSi1008/ai-video-upscaler.git {repo_dir}
    %cd {repo_dir}

# Khởi chạy WebUI song song Dual GPU với TinyURL alias cố định
!python3 app.py --share --alias=4k-upscaler
```

---

### Option D: Apple Silicon Mac (M1/M2/M3 Pro/Max) & Local Offline

Optimized natively for Apple Silicon hardware using **MPS (Metal Performance Shaders)** and **VideoToolbox Hardware 10-bit HEVC (`hevc_videotoolbox`)**:

```bash
# 1. Clone the repository
git clone https://github.com/ThanhSi1008/ai-video-upscaler.git
cd ai-video-upscaler

# 2. Create Python 3.12 virtual environment & install requirements
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt

# 3. Ensure FFmpeg is installed (with VideoToolbox support)
brew install ffmpeg

# 4. Launch the Web UI
./venv/bin/python app.py

# Or run CLI directly on any video file
./venv/bin/python upscale.py /path/to/anime_episode.mkv
```

- Output videos are saved to `~/Movies/Upscaled` with bit-exact softsub `.ass` and font preservation.
- Open and enjoy directly in **IINA** (`/Applications/IINA.app`) with full HDR/Retina color fidelity.

---

### Option E: Docker Container Deployment

```bash
# Build Docker image
docker build -t ai-video-upscaler .

# Run with GPU support
docker run --gpus all -p 7860:7860 ai-video-upscaler
```

---

## 🛠️ Architecture & Technologies

- **Frontend / UI**: Gradio 4.x
- **Backend / Deep Learning**: PyTorch 2.x, TorchVision, Real-ESRGAN
- **Video I/O & Encoding**: FFmpeg, PyTorch IPC Multiprocessing, NVENC / VideoToolbox / libx264
- **Media Ingestion**: yt-dlp

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
