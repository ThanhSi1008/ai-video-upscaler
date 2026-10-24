"""
tests/test_model_regression.py - Bộ kiểm thử hồi quy theo mô hình (C4)
Mỗi mô hình trong registry (models/models.yaml) phải vượt qua kiểm tra đối chiếu độc lập với spandrel.
"""

import os
import pytest
import torch
import spandrel
import numpy as np

from src.models_resolver import load_models_registry
from src.cuda_engine import RealESRAnimeVideoV3, AnimeJaNaiCompact2x


def calc_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 999.0
    return float(10.0 * np.log10(1.0 / mse))


@pytest.mark.parametrize("model_id", ["animevideov3", "animejanai-hd-v3-compact"])
def test_model_independent_regression(model_id):
    """
    So sánh đối chiếu độc lập mô hình trong src/cuda_engine.py với spandrel
    trên 1 crop nhỏ (128x128).
    Ngưỡng: FP32 PSNR >= 50 dB, chênh độ sáng |ΔY| <= 0.5/255.
    """
    models = load_models_registry()
    assert model_id in models, f"Model {model_id} không có trong registry"
    m_info = models[model_id]
    weights_path = m_info["weights_path"]
    if not os.path.exists(weights_path):
        pytest.skip(f"Chưa tải file trọng số {weights_path}, bỏ qua kiểm thử")

    # 1. Nạp mô hình tham chiếu độc lập qua Spandrel
    loader = spandrel.ModelLoader()
    ref_desc = loader.load_from_file(weights_path)
    ref_model = ref_desc.model.eval()

    # 2. Nạp mô hình nội bộ của pipeline
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = sd.get("params_ema", sd.get("params", sd))
    clean_weights = {k: v for k, v in weights.items() if k.startswith("body.")}

    if model_id == "animejanai-hd-v3-compact":
        our_model = AnimeJaNaiCompact2x().eval()
        our_model.load_state_dict(clean_weights, strict=True)
    else:
        our_model = RealESRAnimeVideoV3().eval()
        our_model.load_state_dict(clean_weights, strict=True)

    # 3. Chạy trên crop nhỏ 128x128
    torch.manual_seed(42)
    x = torch.rand(1, 3, 128, 128)

    with torch.no_grad():
        out_ref = torch.clamp(ref_model(x), 0.0, 1.0)
        out_our = torch.clamp(our_model(x), 0.0, 1.0)

    # Nếu animevideov3: spandrel cho ra 4x, còn RealESRAnimeVideoV3 downsample bicubic về 2x
    # Ta so sánh phần thân 4x trước khi downsample hoặc so sánh đầu ra của pipeline
    if model_id == "animevideov3":
        # So sánh đầu ra 4x của thân mạng
        feat = x
        for layer in our_model.body:
            feat = layer(feat)
        out_4x_raw = our_model.upsampler(feat) + torch.nn.functional.interpolate(x, scale_factor=4.0, mode='nearest')
        out_4x_our = torch.clamp(out_4x_raw, 0.0, 1.0)
        diff_max = (out_ref - out_4x_our).abs().max().item()
        assert diff_max < 1e-5, f"Lệch thân mạng 4x với spandrel: {diff_max}"
        assert out_our.shape == (1, 3, 256, 256)
    else:
        # AnimeJaNai: So sánh trực tiếp 2x
        arr_ref = out_ref.squeeze(0).permute(1, 2, 0).numpy()
        arr_our = out_our.squeeze(0).permute(1, 2, 0).numpy()

        psnr = calc_psnr(arr_ref, arr_our)
        delta_y = abs(float(np.mean(arr_ref) - np.mean(arr_our))) * 255.0

        assert psnr >= 50.0, f"PSNR quá thấp: {psnr:.2f} dB < 50 dB"
        assert delta_y <= 0.5, f"Chênh độ sáng quá lớn: {delta_y:.4f} > 0.5"
