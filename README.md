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

An AI-powered video super-resolution platform optimized for **Real-CUGAN Pro (Native 2x Cascaded U-Net)**, **HEVC 10-bit Master Quality NVENC**, **High-speed Magnet Ingestion (`aria2c`)**, **Full Audio/Subtitle/Font Preservation**, and **NVIDIA Dual-GPU Parallel Processing**.

---

## 🌟 Key Features

- **👑 Real-CUGAN Pro Native 2x (Dedicated for Anime 1080p -> 4K)**:
  - **`cugan_conservative`**: Specifically tuned for clean Web-DL (SubsPlease / Crunchyroll) to eliminate mosquito noise while preserving vector line-art and cel-shading.
  - **`cugan_no_denoise`**: Maximum detail retention for pristine Blu-ray Remux sources.
  - **`cugan_denoise3x`**: Aggressive artifact removal for older or heavily compressed releases.
- **🎨 HEVC 10-Bit Color (`yuv420p10le`)**: Master Quality hardware encoding with `-cq 17 -profile:v main10` eliminating color banding on large 4K screens.
- **📝 100% Subtitle, Font & Audio Preservation**: Full bit-exact copying of Japanese/English audio tracks, styled ASS/SSA subtitles, and embedded font attachments into `.mkv`.
- **🧲 Direct Magnet / Torrent Ingestion**: Powered by `aria2c` multi-connection downloading (downloading a 1.4 GB episode in ~30 seconds on Kaggle).
- **⚡ PyTorch 5x5 Laplacian Pyramid GPU Filter**: High-pass micro-edge detail sharpening directly on PyTorch CUDA Tensors.
- **🚀 Dual-GPU Multi-Processing Acceleration**: Splits and processes video segments in parallel across multi-GPU setups (e.g. Dual NVIDIA T4 on Kaggle at ~16 FPS, ~35 mins for a 24-minute episode).
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

---

### Option B: Free Kaggle Notebook Deployment (Dual NVIDIA T4 GPUs ~16+ FPS)

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

### Option C: Local Machine Installation (Mac / Windows / Linux)

```bash
# 1. Clone the repository
git clone https://github.com/ThanhSi1008/ai-video-upscaler.git
cd ai-video-upscaler

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure FFmpeg is installed
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg

# 4. Launch the Web UI
python3 app.py
```

Open `http://localhost:7860` in your browser.

---

### Option D: Docker Container Deployment

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
