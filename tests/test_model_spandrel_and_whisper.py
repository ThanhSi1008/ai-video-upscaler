# -*- coding: utf-8 -*-
"""
tests/test_model_spandrel_and_whisper.py
Kiểm thử toàn diện cho Milestone 4 (M4):
- Tích hợp Spandrel 0.4.2 & Native 2x (không downsample tail khi scale==2, downsample bicubic khi scale==4).
- Bộ bọc _GenericTRTExportModule và quy tắc resolve_trt_profile_max_batch (max_batch=4 cho 2x).
- Cấu hình Faster-Whisper (large-v3-turbo mặc định, override qua WHISPER_MODEL, compute_type CUDA/CPU).
- Đối chiếu hồi quy với realesr-animevideov3 và registry models.yaml.
"""

import os
import sys
import tempfile
import inspect
from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn
import spandrel

from src.cuda_engine import (
    SpandrelModelWrapper,
    RealESRAnimeVideoV3,
    AnimeJaNaiCompact2x,
    GPUPersistentWorker,
)
from src.build_trt_engine import (
    resolve_trt_profile_max_batch,
    _GenericTRTExportModule,
    _PureFP16TRTExportModule,
    export_model_to_onnx,
)
from src.models_resolver import load_models_registry, get_default_model_id
from src.subtitles import transcribe_japanese_audio


# =============================================================================
# 1. Spandrel Native 2x & 4x Model Loader Tests
# =============================================================================

def test_spandrel_loader_with_realesr_animevideov3():
    """Kiểm tra nạp realesr-animevideov3 qua Spandrel 0.4.2 và áp dụng bicubic 0.5x tail."""
    weights_path = "models_bench/realesr-animevideov3.pth"
    if not os.path.exists(weights_path):
        pytest.skip(f"Chưa có weights {weights_path}")

    loader = spandrel.ModelLoader()
    desc = loader.load_from_file(weights_path)
    assert desc.scale == 4
    assert "Compact" in desc.architecture.name

    wrapper = SpandrelModelWrapper(desc.model, scale=desc.scale, target_scale=2.0)
    assert wrapper.scale == 4
    assert wrapper.target_scale == 2.0
    assert hasattr(wrapper, "body")
    assert hasattr(wrapper, "upsampler")

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = wrapper(x)

    # Đầu vào 64x64 qua mô hình 4x -> 256x256, bicubic 0.5x tail -> 128x128 (đúng hệ số 2x)
    assert out.shape == (1, 3, 128, 128)


def test_spandrel_native_2x_span_no_downsample():
    """Kiểm tra mô hình SPAN Native 2x qua Spandrel: không downsample và cho ra đúng 2x."""
    from spandrel.architectures.SPAN.__arch.span import SPAN

    # Tạo mô hình SPAN 2x tối giản và lưu state_dict
    span_model = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=2)
    sd = span_model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_weights = f.name
    try:
        torch.save(sd, tmp_weights)

        loader = spandrel.ModelLoader()
        desc = loader.load_from_file(tmp_weights)
        assert desc.scale == 2
        assert desc.architecture.name == "SPAN"

        wrapper = SpandrelModelWrapper(desc.model, scale=desc.scale, target_scale=2.0)
        assert wrapper.scale == 2
        assert wrapper.target_scale == 2.0

        # Kiểm tra Native 2x không gọi _tail_4x_to_2x
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = wrapper(x)

        # 32x32 -> 64x64 trực tiếp không qua nội suy bicubic tail
        assert out.shape == (1, 3, 64, 64)
    finally:
        if os.path.exists(tmp_weights):
            os.remove(tmp_weights)


def test_spandrel_model_wrapper_micro_slicing():
    """Kiểm tra cơ chế micro-slicing khi batch lớn trên ảnh Full HD với mô hình scale 4."""
    class Dummy4xInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 4
        def forward(self, x):
            return nn.functional.interpolate(x, scale_factor=4.0, mode="nearest")

    inner = Dummy4xInner()
    wrapper = SpandrelModelWrapper(inner, scale=4, target_scale=2.0)

    # Batch 4 khung hình kích thước 540x960 (>= 540*960)
    x = torch.randn(4, 3, 540, 960)
    with torch.no_grad():
        out = wrapper(x)

    assert out.shape == (4, 3, 1080, 1920)


def test_gpu_persistent_worker_spandrel_integration():
    """Kiểm tra GPUPersistentWorker tự động nạp qua Spandrel trên CPU."""
    weights_path = "models_bench/realesr-animevideov3.pth"
    if not os.path.exists(weights_path):
        pytest.skip(f"Chưa có weights {weights_path}")

    worker = GPUPersistentWorker(
        model_id="animevideov3",
        weights_path=weights_path,
        precision="fp32",
        device="cpu",
    )
    assert isinstance(worker.model, SpandrelModelWrapper)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = worker.model(x)
    assert out.shape == (1, 3, 128, 128)


# =============================================================================
# 2. Generic TRT Export & Profile Max Batch Tests
# =============================================================================

def test_resolve_trt_profile_max_batch_rules():
    """Kiểm tra phân giải max_batch cho TRT profile: Native 2x cho phép batch=4, 4x giới hạn batch=2."""
    class MockModel2x(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 2

    class MockModel4x(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 4

    # 1. Native 2x model: max_batch <= 4
    m2 = MockModel2x()
    assert resolve_trt_profile_max_batch(m2, max_batch=8) == 4
    assert resolve_trt_profile_max_batch(m2, max_batch=2) == 2

    # 2. 4x model (như animevideov3): max_batch <= 2 để tránh OOM 8K buffer
    m4 = MockModel4x()
    assert resolve_trt_profile_max_batch(m4, max_batch=8) == 2
    assert resolve_trt_profile_max_batch(m4, max_batch=1) == 1

    # 3. Kích thước nhỏ (<540p): không bị giới hạn
    assert resolve_trt_profile_max_batch(m4, max_batch=8, input_hw=(270, 480)) == 8


def test_generic_trt_export_module():
    """Kiểm tra _GenericTRTExportModule cho kiến trúc không có .body (như SPAN)."""
    class SimpleSPAN2x(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 12, 3, 1, 1)
            self.ps = nn.PixelShuffle(2)
            self.scale = 2
        def forward(self, x):
            return self.ps(self.conv(x))

    span = SimpleSPAN2x()
    wrapped = _GenericTRTExportModule(span, target_scale=2.0)
    assert wrapped.model_scale == 2.0
    assert wrapped.target_scale == 2.0

    x = torch.randn(1, 3, 32, 32)
    out = wrapped(x)
    assert out.shape == (1, 3, 64, 64)

    # Thử xuất ONNX với _GenericTRTExportModule qua export_model_to_onnx
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        exported_path = export_model_to_onnx(
            span,
            onnx_path,
            max_batch=4,
            dummy_shape=(32, 32),
            target_scale=2.0,
        )
        assert os.path.exists(exported_path)
        assert os.path.getsize(exported_path) > 0
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)


def test_generic_trt_export_module_unwraps_spandrel_wrapper():
    """Kiểm tra _GenericTRTExportModule bóc tách chính xác SpandrelModelWrapper."""
    class DummyInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 12, 3, 1, 1)
            self.ps = nn.PixelShuffle(2)
        def forward(self, x):
            return self.ps(self.conv(x))

    inner = DummyInner()
    sp_wrapper = SpandrelModelWrapper(inner, scale=2, target_scale=2.0)
    trt_export_mod = _GenericTRTExportModule(sp_wrapper, target_scale=2.0)

    assert trt_export_mod.model is inner
    assert trt_export_mod.model_scale == 2.0


# =============================================================================
# 3. Model Registry Tests (models/models.yaml)
# =============================================================================

def test_models_registry_native_2x_entries():
    """Kiểm tra models/models.yaml có đầy đủ span-2x và 2x_AnimeJaNai_V3."""
    registry = load_models_registry()

    assert "animevideov3" in registry
    assert registry["animevideov3"]["default"] is True

    # 1. Kiểm tra span-2x
    assert "span-2x" in registry
    span_cfg = registry["span-2x"]
    assert span_cfg["arch"]["type"] == "SPAN"
    assert span_cfg["arch"]["upscale"] == 2
    assert span_cfg["downsample"] == "none"
    assert span_cfg["default"] is False

    # 2. Kiểm tra 2x_AnimeJaNai_V3
    assert "2x_AnimeJaNai_V3" in registry
    janai_cfg = registry["2x_AnimeJaNai_V3"]
    assert janai_cfg["arch"]["type"] == "SPAN"
    assert janai_cfg["arch"]["upscale"] == 2
    assert janai_cfg["downsample"] == "none"
    assert janai_cfg["default"] is False

    assert get_default_model_id() == "animevideov3"


# =============================================================================
# 4. Faster-Whisper Turbo Configuration Tests
# =============================================================================

def test_whisper_default_model_signature():
    """Kiểm tra chữ ký transcribe_japanese_audio có default model_size='large-v3-turbo'."""
    sig = inspect.signature(transcribe_japanese_audio)
    assert sig.parameters["model_size"].default == "large-v3-turbo"


def test_whisper_env_override(monkeypatch, tmp_path):
    """Kiểm tra biến môi trường WHISPER_MODEL ghi đè chính xác mô hình được nạp."""
    called_model = {}

    class MockSegment:
        start = 0.0
        end = 1.0
        text = "こんにちは"
        no_speech_prob = 0.0
        compression_ratio = 1.0
        avg_logprob = 0.0

    class MockWhisperModel:
        def __init__(self, model_size, device="cuda", compute_type="float16"):
            called_model["model_size"] = model_size
            called_model["device"] = device
            called_model["compute_type"] = compute_type

        def transcribe(self, audio_path, **kwargs):
            return [MockSegment()], MagicMock()

    monkeypatch.setitem(sys.modules, "faster_whisper", MagicMock(WhisperModel=MockWhisperModel))
    monkeypatch.setenv("WHISPER_MODEL", "kotoba-whisper-v2")

    # Giả lập file audio tạm tồn tại và probe hợp lệ
    def mock_run_impl(cmd, *args, **kwargs):
        if cmd[0] == "ffprobe":
            return MagicMock(returncode=0, stdout='{"streams": [{"index": 0, "channels": 2}]}')
        if "-vn" in cmd:
            out_file = cmd[-1]
            with open(out_file, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 40)
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", mock_run_impl)

    fake_video = str(tmp_path / "test_ep.mkv")
    with open(fake_video, "w") as f:
        f.write("mock")
    out_srt = str(tmp_path / "test_ep.ja.srt")

    ok = transcribe_japanese_audio(fake_video, out_srt)
    assert ok is True
    assert called_model.get("model_size") == "kotoba-whisper-v2"
    # Trên máy không có GPU lock hoặc fallback, compute_type phù hợp device
    assert called_model.get("compute_type") in ("float16", "int8")


def test_whisper_compute_type_selection(monkeypatch, tmp_path):
    """Kiểm tra compute_type: float16 trên CUDA, int8 trên CPU fallback."""
    recorded_kwargs = {}

    class MockSegment:
        start = 0.0
        end = 1.0
        text = "こんにちは"
        no_speech_prob = 0.0
        compression_ratio = 1.0
        avg_logprob = 0.0

    class MockWhisperModel:
        def __init__(self, model_size, device="cpu", compute_type="int8"):
            recorded_kwargs["device"] = device
            recorded_kwargs["compute_type"] = compute_type

        def transcribe(self, audio_path, **kwargs):
            return [MockSegment()], MagicMock()

    monkeypatch.setitem(sys.modules, "faster_whisper", MagicMock(WhisperModel=MockWhisperModel))
    # Ép GPU lock đang bận -> chuyển sang CPU
    monkeypatch.setattr("src.config.is_gpu_lock_held_by_other", lambda: 99999)

    def mock_run_impl(cmd, *args, **kwargs):
        if cmd[0] == "ffprobe":
            return MagicMock(returncode=0, stdout='{"streams": [{"index": 0, "channels": 2}]}')
        if "-vn" in cmd:
            out_file = cmd[-1]
            with open(out_file, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 40)
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", mock_run_impl)

    fake_video = str(tmp_path / "test_ep2.mkv")
    with open(fake_video, "w") as f:
        f.write("mock")
    out_srt = str(tmp_path / "test_ep2.ja.srt")

    ok = transcribe_japanese_audio(fake_video, out_srt)
    assert ok is True
    assert recorded_kwargs.get("device") == "cpu"
    assert recorded_kwargs.get("compute_type") == "int8"


# =============================================================================
# 5. Native 2x Registry Integrity, Mixed-Case Resolution & VRAM-Aware TRT Batch
# =============================================================================

def test_resolve_model_mixed_case_and_ultracompact_entries():
    """Kiểm tra phân giải không phân biệt hoa thường cho 2x_AnimeJaNai_V3 và các biến thể UltraCompact."""
    from src.models_resolver import resolve_model_for_video

    res_exact = resolve_model_for_video("ep22.mkv", "2x_AnimeJaNai_V3")
    assert res_exact["id"] == "2x_AnimeJaNai_V3"

    res_lower = resolve_model_for_video("ep22.mkv", "2x_animejanai_v3")
    assert res_lower["id"] == "2x_AnimeJaNai_V3"

    res_uc = resolve_model_for_video("ep22.mkv", "animejanai-hd-v3-sharp1-ultracompact")
    assert res_uc["id"] == "animejanai-hd-v3-sharp1-ultracompact"
    assert res_uc["arch"]["upscale"] == 2
    assert res_uc["downsample"] == "none"


def test_all_registered_models_have_sha256_and_https_source_url():
    """Mọi mô hình trong models/models.yaml phải có sha256 hợp lệ (64 ký tự hex) và source_url https://."""
    registry = load_models_registry()
    assert len(registry) >= 7
    for m_id, m_cfg in registry.items():
        sha = str(m_cfg.get("sha256") or "")
        url = str(m_cfg.get("source_url") or "")
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha.lower()), (
            f"Model {m_id} thiếu sha256 hợp lệ: {sha!r}"
        )
        assert url.startswith("https://"), f"Model {m_id} thiếu source_url https: {url!r}"


def test_resolve_trt_profile_max_batch_vram_aware_cap():
    """Trên GPU 12 GB VRAM (RTX 3060) ở 1080p, Native 2x giới hạn max_batch=2 tránh OOM 12.15 GiB."""
    m2 = AnimeJaNaiCompact2x(num_feat=64, num_conv=8)
    assert resolve_trt_profile_max_batch(m2, max_batch=8, input_hw=(1080, 1920)) == 4
    assert resolve_trt_profile_max_batch(m2, max_batch=8, input_hw=(1080, 1920), available_vram_gb=12.0) == 2
    assert resolve_trt_profile_max_batch(m2, max_batch=8, input_hw=(1080, 1920), available_vram_gb=24.0) == 4
    assert resolve_trt_profile_max_batch(m2, max_batch=8, input_hw=(360, 640), available_vram_gb=12.0) == 8


def test_animejanai_compact_parametric_variants():
    """Kiểm tra AnimeJaNaiCompact2x hỗ trợ cả UltraCompact (64f8c) và SuperUltraCompact (24f8c)."""
    x = torch.randn(1, 3, 32, 32)
    for nf, nc in ((64, 8), (24, 8)):
        m = AnimeJaNaiCompact2x(num_feat=nf, num_conv=nc).eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, 3, 64, 64)


def test_auto_download_zip_extraction_and_sha256_verification(monkeypatch, tmp_path):
    """Kiểm tra _auto_download_registered_weights giải nén đúng .pth từ gói .zip và xác thực SHA-256."""
    import hashlib
    import io
    import zipfile

    dummy_bytes = b"FAKE_PTH_WEIGHTS_PAYLOAD_12345"
    expected_sha = hashlib.sha256(dummy_bytes).hexdigest()

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("2x_AnimeJaNai_HD_V3_ModelsOnly/test_target.pth", dummy_bytes)
    zip_payload = zip_buf.getvalue()

    class FakeHTTPResp(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=120: FakeHTTPResp(zip_payload),
    )

    target_pth = tmp_path / "test_target.pth"
    dummy_worker = object.__new__(GPUPersistentWorker)
    dummy_worker.model_id = "test-zip-model"
    dummy_worker.weights_path = str(target_pth)

    dummy_worker._auto_download_registered_weights({
        "name": "Test Zip Model",
        "source_url": "https://example.com/release/models.zip",
        "sha256": expected_sha,
    })

    assert target_pth.exists()
    assert target_pth.read_bytes() == dummy_bytes

    # Trường hợp sai SHA-256 phải xóa file tạm và ném RuntimeError
    bad_pth = tmp_path / "test_target.pth"
    bad_pth.unlink()
    with pytest.raises(RuntimeError):
        dummy_worker._auto_download_registered_weights({
            "name": "Test Zip Model",
            "source_url": "https://example.com/release/models.zip",
            "sha256": "0" * 64,
        })
    assert not bad_pth.exists()


def test_p010le_clamps_out_of_gamut_rgb_overshoots_before_bt709():
    """Kiểm tra P010LE clamp RGB overshoots (<0 hoặc >1) trước ma trận BT.709 để không rò rỉ sắc độ Cb/Cr."""
    from src.trt_zero_copy_upscaler import GPUColorConverterP010LE

    conv = GPUColorConverterP010LE(device=torch.device("cpu"))
    # R > 1.0 khi G < 1.0 trước đây làm trôi U/V tới 15 codewords dù chưa chạm biên [64..960]
    overshoot = torch.tensor(
        [[[[1.08]], [[0.92]], [[0.80]]], [[[-0.06]], [[0.50]], [[0.45]]]],
        dtype=torch.float32,
    ).expand(2, 3, 8, 8).contiguous()
    pre_clamped = overshoot.clamp(0.0, 1.0)

    out_overshoot = conv.rgb_to_p010le(overshoot)
    out_clamped = conv._p010le_core(pre_clamped)
    assert torch.equal(out_overshoot, out_clamped), (
        "RGB overshoot ngoài [0, 1] phải khớp bit-exact với RGB đã clamp trước ma trận BT.709!"
    )


def test_auto_download_requires_valid_sha256_and_handles_querystring_appledouble_zip(monkeypatch, tmp_path):
    """Kiểm tra _auto_download_registered_weights từ chối thiếu SHA-256/HTTP và bỏ qua __MACOSX / ._ trong .zip?token=..."""
    import hashlib
    import io
    import zipfile

    dummy_worker = object.__new__(GPUPersistentWorker)
    dummy_worker.model_id = "test-sec-model"
    target_pth = tmp_path / "clean_target.pth"
    dummy_worker.weights_path = str(target_pth)

    network_called = {"count": 0}

    def fail_if_called(*args, **kwargs):
        network_called["count"] += 1
        raise AssertionError("Không được gọi mạng khi thiếu SHA-256 hoặc URL không phải HTTPS!")

    monkeypatch.setattr("urllib.request.urlopen", fail_if_called)

    # 1. Thiếu hoặc sai định dạng SHA-256 64-hex -> từ chối ngay trước khi tải
    for bad_sha in (None, "", "abc123", "z" * 64):
        with pytest.raises(RuntimeError):
            dummy_worker._auto_download_registered_weights({
                "name": "Missing SHA",
                "source_url": "https://example.com/weights.pth",
                "sha256": bad_sha,
            })
    # 2. HTTP không mã hóa -> từ chối ngay trước khi tải
    with pytest.raises(RuntimeError):
        dummy_worker._auto_download_registered_weights({
            "name": "Insecure HTTP",
            "source_url": "http://example.com/weights.pth",
            "sha256": "a" * 64,
        })
    assert network_called["count"] == 0

    # 3. URL .zip có query string + chứa thư mục và rác AppleDouble __MACOSX / ._clean_target.pth
    real_payload = b"REAL_VERIFIED_WEIGHTS_BYTES_98765"
    real_sha = hashlib.sha256(real_payload).hexdigest()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pack/", b"")
        zf.writestr("__MACOSX/pack/clean_target.pth", b"APPLEDOUBLE_JUNK_1")
        zf.writestr("pack/._clean_target.pth", b"APPLEDOUBLE_JUNK_2")
        zf.writestr("pack/clean_target.pth", real_payload)
    zip_bytes = zip_buf.getvalue()

    class FakeHTTPResp(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=120: FakeHTTPResp(zip_bytes))
    dummy_worker._auto_download_registered_weights({
        "name": "Signed Zip URL",
        "source_url": "https://example.com/releases/pack.zip?token=xyz&dl=1",
        "sha256": real_sha,
    })
    assert target_pth.exists()
    assert target_pth.read_bytes() == real_payload


def test_get_gpu_worker_recreates_on_device_or_weights_change(monkeypatch, tmp_path):
    """Kiểm tra singleton get_gpu_worker tạo mới worker khi đổi weights_path hoặc device."""
    import src.cuda_engine as ce

    created = []

    class DummyWorker:
        def __init__(self, model_id, weights_path, precision="fp16", max_chunks_before_restart=10, device=None, batch_size=None, use_tensorrt=True):
            self.model_id = model_id
            self.weights_path = weights_path
            self.precision = precision.lower()
            self.device = torch.device(device or "cpu")
            self.batch_size = batch_size or 1
            self.use_tensorrt = bool(use_tensorrt)
            created.append(self)

    monkeypatch.setattr(ce, "GPUPersistentWorker", DummyWorker)
    monkeypatch.setattr(ce, "_GLOBAL_GPU_WORKER", None)

    w1_path = str(tmp_path / "m1.pth")
    w2_path = str(tmp_path / "m2.pth")

    inst1 = ce.get_gpu_worker("custom-id", w1_path, precision="fp32", device="cpu", batch_size=2)
    inst1_reuse = ce.get_gpu_worker("custom-id", w1_path, precision="fp32", device="cpu", batch_size=2)
    assert inst1 is inst1_reuse
    assert len(created) == 1

    # Đổi weights_path trên cùng model_id -> bắt buộc khởi tạo mới
    inst2 = ce.get_gpu_worker("custom-id", w2_path, precision="fp32", device="cpu", batch_size=2)
    assert inst2 is not inst1
    assert len(created) == 2

    # Đổi device từ cpu -> meta (hoặc cuda giả lập) -> bắt buộc khởi tạo mới
    inst3 = ce.get_gpu_worker("custom-id", w2_path, precision="fp32", device="meta", batch_size=2)
    assert inst3 is not inst2
    assert inst3.device.type == "meta"
    assert len(created) == 3


def test_transcribe_japanese_audio_no_empty_srt_and_cleans_temp_wav(monkeypatch, tmp_path):
    """Khi 0 câu thoại hợp lệ hoặc lỗi giữa chừng: không tạo file .srt rỗng và luôn xóa .temp.wav."""
    def mock_run_impl(cmd, *args, **kwargs):
        if cmd[0] == "ffprobe":
            return MagicMock(returncode=0, stdout='{"streams": [{"index": 0, "channels": 2}]}')
        if "-vn" in cmd:
            with open(cmd[-1], "wb") as f:
                f.write(b"RIFF" + b"\x00" * 64)
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", mock_run_impl)
    monkeypatch.setattr("src.config.is_gpu_lock_held_by_other", lambda: 99999)

    fake_video = str(tmp_path / "silent_ep.mkv")
    with open(fake_video, "w", encoding="utf-8") as f:
        f.write("mock")

    # 1. Trường hợp 0 câu thoại -> trả về False, KHÔNG ghi file .ja.srt rỗng, xóa sạch .temp.wav
    class SilentWhisper:
        def __init__(self, *a, **kw):
            pass
        def transcribe(self, audio_path, **kw):
            return iter([]), MagicMock()

    monkeypatch.setitem(sys.modules, "faster_whisper", MagicMock(WhisperModel=SilentWhisper))
    out_srt_silent = tmp_path / "silent_ep.ja.srt"
    temp_wav_silent = tmp_path / "silent_ep.ja.temp.wav"
    assert transcribe_japanese_audio(fake_video, str(out_srt_silent)) is False
    assert not out_srt_silent.exists(), "Không được ghi file .srt rỗng 0 cue làm hỏng phát hiện track JA!"
    assert not temp_wav_silent.exists(), "Phải dọn sạch .temp.wav sau khi nhận diện 0 cue!"

    # 2. Trường hợp transcribe ném ngoại lệ -> trả về False và vẫn xóa sạch .temp.wav trong finally
    class CrashingWhisper:
        def __init__(self, *a, **kw):
            pass
        def transcribe(self, audio_path, **kw):
            raise RuntimeError("Simulated CTranslate2 failure")

    monkeypatch.setitem(sys.modules, "faster_whisper", MagicMock(WhisperModel=CrashingWhisper))
    out_srt_err = tmp_path / "err_ep.ja.srt"
    temp_wav_err = tmp_path / "err_ep.ja.temp.wav"
    assert transcribe_japanese_audio(fake_video, str(out_srt_err)) is False
    assert not out_srt_err.exists()
    assert not temp_wav_err.exists(), "Phải dọn sạch .temp.wav trong finally ngay cả khi Whisper ném lỗi!"
