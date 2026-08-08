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

An AI-powered video super-resolution and restoration platform supporting **Real-ESRGAN (`realesr-animevideov3` & `RealESRGAN_x4plus_anime_6B`)**, **5x5 GPU Laplacian Detail Enhancement**, **Anime 4K HDR Color Boost**, and **NVIDIA Dual-GPU Parallel Processing**.

---

## 🌟 Key Features

- **🤖 Dual AI Model Architecture**:
  - **`AnimeVideoV3`**: Lightweight & ultra-fast (~16+ FPS on Dual T4 GPUs).
  - **`Real-ESRGAN x4Plus Anime 6B`**: Deep 6-block RRDBNet model for vector-sharp line art and micro-detail reconstruction.
- **⚡ PyTorch 5x5 Laplacian Pyramid GPU Filter**: High-pass micro-edge detail sharpening directly on PyTorch CUDA Tensors.
- **🎨 Anime 4K HDR Color & Dynamic Contrast Boost**: Automatic vibrancy and dynamic range enhancement for crisp, vibrant visuals.
- **🚀 Dual-GPU Multi-Processing Acceleration**: Splits and processes video segments in parallel across multi-GPU setups (e.g. Dual NVIDIA T4 on Kaggle).
- **🎬 NVIDIA NVENC Master Quality Quality**: Encoder settings tuned with Spatial & Temporal Adaptive Quantization (`-qp 14`, `-spatial-aq 1`, `-temporal-aq 1`).
- **⚡ Zero-Disk Streaming Memory Pipes**: High-performance FFmpeg stdin/stdout streaming without creating millions of temporary image files on disk.

---

## 🚀 Deployment Options

### Option A: 1-Click Deploy to Hugging Face Spaces (Always-On / Easy Demo)

Click the badge below to duplicate this app directly into your Hugging Face Spaces account:

[![Deploy to Spaces](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deploy-to-spaces-lg.svg)](https://huggingface.co/new-space?template=ThanhSi1008/ai-video-upscaler)

> 💡 **Note**: Free Spaces run on CPU. For maximum GPU acceleration, upgrade the Space hardware to T4 / A10G GPU in Space Settings.

---

### Option B: Free Kaggle Notebook Deployment (Dual NVIDIA T4 GPUs ~16+ FPS)

Run the following cell inside a free **Kaggle GPU Notebook**:

```python
# @title 🎬 AI Video Upscaler 4K - Ultra High Speed WebUI
import os, sys

!apt-get update -qq && apt-get install -y ffmpeg -qq
!pip install -q --no-cache-dir gradio torch torchvision yt-dlp

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
