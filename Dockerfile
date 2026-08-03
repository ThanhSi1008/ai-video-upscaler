# Sử dụng base image PyTorch chính thức với CUDA 11.8 và Ubuntu 22.04
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Đặt thư mục làm việc
WORKDIR /app

# Cài đặt FFmpeg và các thư viện hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn ứng dụng
COPY . .

# Expose cổng Gradio
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Khởi chạy Gradio App
CMD ["python3", "app.py"]
