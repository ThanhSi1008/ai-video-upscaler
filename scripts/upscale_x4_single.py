# -*- coding: utf-8 -*-
"""
scripts/upscale_x4_single.py
Công cụ phóng to ảnh đơn lẻ x4 (Native 4x Super-Resolution) chuyên dụng:
- Sử dụng kiến trúc RealESR-AnimeVideoV3 gốc (64f16c, PixelShuffle 4x).
- KHÔNG qua bước nội suy hạ mẫu Bicubic 0.5x (giữ nguyên độ phóng đại x4).
- Tối ưu hóa suy luận trên NVIDIA Tensor Cores (FP16 NHWC / PyTorch CUDA).
- Hỗ trợ đường dẫn Unicode tiếng Việt có dấu an toàn trên Windows.

Cách sử dụng:
    python scripts/upscale_x4_single.py "path/to/image.png"
    python scripts/upscale_x4_single.py "path/to/image.png" --output "path/to/output_4x.png"
"""

import os
import sys
import time
import argparse
from typing import Optional

# Cấu hình UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class RealESRAnimeVideoV3_4x(nn.Module):
    """
    Mô hình RealESR-AnimeVideoV3 x4 gốc (Native 4x).
    Bỏ qua lớp F.interpolate(out_4x, scale_factor=0.5, mode='bicubic') để giữ nguyên x4.
    """
    def __init__(self):
        super().__init__()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(3, 64, 3, 1, 1))
        self.body.append(nn.PReLU(64))
        for _ in range(16):
            self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
            self.body.append(nn.PReLU(64))
        self.body.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(4)

    def forward(self, x):
        feat = x
        for layer in self.body:
            feat = layer(feat)
        out = self.upsampler(feat)
        base = F.interpolate(x, scale_factor=4.0, mode='nearest')
        return out + base


def resolve_weights_path(custom_path: Optional[str] = None) -> str:
    """Tự động tìm kiếm file trọng số realesr-animevideov3.pth trên hệ thống."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        custom_path,
        os.path.join(repo_root, "models_bench", "realesr-animevideov3.pth"),
        os.path.join(repo_root, "models", "realesr-animevideov3.pth"),
        os.path.join(repo_root, "bin", "realesrgan", "models", "realesr-animevideov3.pth"),
        os.path.expanduser("~/.cache/realesrgan/realesr-animevideov3.pth"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)

    # Nếu chưa có, tải tự động về models_bench/
    target_path = os.path.join(repo_root, "models_bench", "realesr-animevideov3.pth")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    print(f"📥 Đang tự động tải trọng số RealESR-AnimeVideoV3 từ {url}...")
    import urllib.request
    urllib.request.urlretrieve(url, target_path)
    return target_path


def upscale_single_image(
    input_path: str,
    output_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    precision: str = "fp16",
    device_str: Optional[str] = None
) -> str:
    """
    Thực hiện upscale ảnh x4 bằng RealESR-AnimeVideoV3 native 4x.
    Returns:
        Đường dẫn file ảnh 4K/4x đã lưu.
    """
    input_path = os.path.abspath(os.path.expanduser(input_path))
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh nguồn: {input_path}")

    if not output_path:
        base_dir = os.path.dirname(input_path)
        base_name, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(base_dir, f"{base_name}_4x{ext or '.png'}")
    else:
        output_path = os.path.abspath(os.path.expanduser(output_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    weights_file = resolve_weights_path(weights_path)

    print(f"\n{'='*70}")
    print(f"🖼️  File nguồn: {input_path}")
    img_pil = Image.open(input_path).convert("RGB")
    orig_w, orig_h = img_pil.size
    print(f"📐 Kích thước gốc: {orig_w} × {orig_h}")

    # Chuyển đổi tensor chuẩn hóa [0.0, 1.0]
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"⚙️  Thiết bị suy luận: {device} ({dev_name})")

    # Khởi tạo mô hình
    model = RealESRAnimeVideoV3_4x().to(device)

    # Nạp weights
    print(f"📦 Nạp trọng số: {os.path.basename(weights_file)}")
    sd = torch.load(weights_file, map_location="cpu", weights_only=False)
    weights = sd.get("params_ema", sd.get("params", sd))
    clean_weights = {k: v for k, v in weights.items() if k.startswith("body.")}
    model.load_state_dict(clean_weights, strict=True)
    model.eval()

    use_fp16 = precision.lower() == "fp16" and device.type == "cuda"
    if use_fp16:
        model = model.half().to(memory_format=torch.channels_last)
        tensor = tensor.to(device).half().to(memory_format=torch.channels_last)
    else:
        model = model.float()
        tensor = tensor.to(device).float()

    print(f"🚀 Đang suy luận AI x4 native (Precision: {'FP16 NHWC' if use_fp16 else 'FP32'})...")
    t0 = time.time()
    with torch.inference_mode():
        out = model(tensor)
        out = out.clamp(0.0, 1.0)
    elapsed = time.time() - t0

    # Chuyển về numpy uint8
    out_cpu = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    out_uint8 = (out_cpu * 255.0 + 0.5).astype(np.uint8)
    out_h, out_w, _ = out_uint8.shape

    print(f"✨ Hoàn tất suy luận trong {elapsed:.3f} giây!")
    print(f"📐 Kích thước sau upscale x4: {out_w} × {out_h} (Tỷ lệ: 4.0x)")

    # Lưu ảnh xuất ra
    print(f"💾 Đang lưu ảnh: {output_path}")
    out_pil = Image.fromarray(out_uint8)
    out_pil.save(output_path, format="PNG", compress_level=3)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Đã lưu thành công: {output_path} ({size_mb:.2f} MB)")
    print(f"{'='*70}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Upscale ảnh đơn lẻ x4 native bằng RealESR-AnimeVideoV3 (không hạ 0.5)"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=r"G:\My Drive\01_Hoc_tap_PTIT\Năm_4\YEAR_4_DO_AN_TOT_NGHIEP\02_PRE-PRODUCTION\04_STORY_BOARD\CNTT\SC01\1.png",
        help="Đường dẫn file ảnh đầu vào (mặc định: storyboard SC01/1.png)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Đường dẫn file ảnh đầu ra (mặc định: <tên_ảnh>_4x.png cùng thư mục)"
    )
    parser.add_argument(
        "-w", "--weights",
        default=None,
        help="Đường dẫn tùy chọn tới file trọng số realesr-animevideov3.pth"
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Độ chính xác dấu phẩy động (mặc định: fp16)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Thiết bị tính toán: cuda:0 hoặc cpu (mặc định: tự động chọn CUDA nếu có)"
    )

    args = parser.parse_args()
    upscale_single_image(
        input_path=args.input,
        output_path=args.output,
        weights_path=args.weights,
        precision=args.precision,
        device_str=args.device
    )


if __name__ == "__main__":
    main()
