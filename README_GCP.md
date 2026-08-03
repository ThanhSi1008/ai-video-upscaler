# Hướng Dẫn Deploy AI Video Upscaler Lên Google Cloud Platform (GCP)

Tài liệu này hướng dẫn chi tiết 2 phương án triển khai ứng dụng **AI Video Upscaler** kèm **Gradio Web UI** lên Google Cloud Platform (GCP).

---

## 🚀 Phương Án 1: GCP Compute Engine (VM với GPU NVIDIA T4/L4) - *Khuyên Dùng*

Đây là phương án tối ưu nhất cho tác vụ AI nâng cấp video nặng, cho phép tận dụng tối đa GPU NVIDIA CUDA với chi phí cố định.

### Bước 1: Tạo Compute Engine VM trên GCP

Chạy lệnh `gcloud` trong Google Cloud Shell hoặc cài đặt local CLI:

```bash
gcloud compute instances create upscale-gpu-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --maintenance-policy=TERMINATE
```

### Bước 2: Mở Cổng Tường Lửa 7860 Cho Web UI

```bash
gcloud compute firewall-rules create allow-gradio-ui \
    --allow=tcp:7860 \
    --target-tags=http-server,https-server \
    --description="Open port 7860 for Gradio Web UI"
```

### Bước 3: SSH Vào VM & Cài Đặt Môi Trường GPU

```bash
gcloud compute ssh upscale-gpu-vm --zone=us-central1-a
```

Trên VM, cài đặt Driver NVIDIA & Docker:

```bash
# 1. Cài đặt Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. Cài đặt NVIDIA Container Toolkit (Cho phép Docker dùng GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Bước 4: Clone Code & Khởi Chạy Container

```bash
# Clone repository của bạn
git clone <YOUR_GIT_REPOSITORY_URL>
cd upscale

# Build Docker image
sudo docker build -t upscale-app .

# Chạy container với GPU
sudo docker run -d --gpus all -p 7860:7860 --name upscale-web-app upscale-app
```

Bây giờ bạn có thể truy cập Web UI tại địa chỉ: `http://<IP_NGOẠI_MẠNG_CỦA_VM>:7860`

---

## ⚡ Phương Án 2: GCP Cloud Run (Serverless Container với GPU)

Nếu bạn muốn ứng dụng tự động mở rộng theo nhu cầu (Serverless), có thể deploy lên **Cloud Run GPU**.

### Bước 1: Push Container Image Lên Google Artifact Registry

```bash
# Đặt ID Dự Án GCP
PROJECT_ID=$(gcloud config get-value project)

# Tạo Repository trong Artifact Registry
gcloud artifacts repositories create upscale-repo \
    --repository-format=docker \
    --location=us-central1

# Configure Docker auth
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build & Push Image
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/upscale-repo/upscale-app:latest .
docker push us-central1-docker.pkg.dev/$PROJECT_ID/upscale-repo/upscale-app:latest
```

### Bước 2: Deploy Lên Cloud Run

```bash
gcloud run deploy upscale-service \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/upscale-repo/upscale-app:latest \
    --region us-central1 \
    --port 7860 \
    --cpu 4 \
    --memory 16Gi \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --max-instances 2 \
    --allow-unauthenticated
```

Sau khi hoàn tất, GCP sẽ cung cấp cho bạn một đường dẫn HTTPS truy cập Web UI trực tiếp.
