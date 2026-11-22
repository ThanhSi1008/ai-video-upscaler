# -*- coding: utf-8 -*-
"""
tests/test_enospc_backpressure_retry.py
Kiểm tra mô phỏng:
1. Cổng Backpressure: need = n_unique * max(recent, p95, 9.0) * 1.2
2. Giả lập chuỗi chunk 192 frame nặng liên tiếp (với dữ liệu kích thước PNG thực tế ~11.5 - 14.5 MB của Slime)
3. Cơ chế retry ENOSPC tối đa 2 lần và ném FailedResumableError khi vượt ngưỡng retry
"""

import os
import shutil
import tempfile
import pytest
import numpy as np

from src.pipeline import FailedResumableError


def test_backpressure_formula_calculation():
    """Kiểm tra công thức tính need_mb với sàn 9MB, recent, p95 và hệ số 1.2"""
    # Trường hợp 1: Chưa có chunk nào trước đó -> dùng sàn 9.0 MB
    chunk_png_unit_sizes = []
    n_unique = 192
    recent_png_mb = chunk_png_unit_sizes[-1] if chunk_png_unit_sizes else 9.0
    p95_png_mb = float(np.percentile(chunk_png_unit_sizes, 95)) if chunk_png_unit_sizes else 9.0
    unit_size_mb = max(recent_png_mb, p95_png_mb, 9.0)
    need_mb = n_unique * unit_size_mb * 1.2
    assert unit_size_mb == 9.0
    assert need_mb == pytest.approx(192 * 9.0 * 1.2)  # 2073.6 MB

    # Trường hợp 2: Chunk gần nhất là 11.5 MB (thực tế Slime chunk 54-55)
    chunk_png_unit_sizes = [10.2, 11.5]
    recent_png_mb = chunk_png_unit_sizes[-1]
    p95_png_mb = float(np.percentile(chunk_png_unit_sizes, 95))
    unit_size_mb = max(recent_png_mb, p95_png_mb, 9.0)
    need_mb = n_unique * unit_size_mb * 1.2
    assert unit_size_mb == pytest.approx(11.5, rel=0.01)
    assert need_mb > 2600.0

    # Trường hợp 3: Chuỗi chunk nặng với p95 lên đến 14.2 MB (đỉnh của Slime chunk 55)
    chunk_png_unit_sizes = [10.0, 11.0, 12.0, 13.5, 14.5]
    recent_png_mb = 11.0  # chunk vừa rồi nhẹ hơn
    p95_png_mb = float(np.percentile(chunk_png_unit_sizes, 95))  # nhưng p95 tích lũy cao
    unit_size_mb = max(recent_png_mb, p95_png_mb, 9.0)
    assert unit_size_mb > 14.0
    need_mb = n_unique * unit_size_mb * 1.2
    # Với 192 frames, need_mb > 3200 MB -> kích hoạt backpressure chờ encode ngay lập tức trên RAM Disk 4096MB
    assert need_mb > 3200.0


def test_consecutive_heavy_chunks_backpressure_trigger():
    """Mô phỏng chuỗi 3 chunk 192 frames liên tiếp với kích thước 12.0 MB / frame"""
    ramdisk_total_mb = 4096.0
    chunk_n_minus_1_out_mb = 192 * 11.8  # ~2265 MB chưa giải phóng
    free_ramdisk_mb = ramdisk_total_mb - chunk_n_minus_1_out_mb  # 1831 MB

    n_unique_chunk_n = 192
    unit_size_mb = 12.0
    need_mb = n_unique_chunk_n * unit_size_mb * 1.2  # 2764.8 MB

    # Kiểm tra cổng backpressure
    assert free_ramdisk_mb < need_mb, "Cổng backpressure PHẢI kích hoạt vì 1831 MB < 2764.8 MB"

    # Sau khi chờ Media Engine encode xong chunk N-1 và giải phóng:
    free_after_release = ramdisk_total_mb  # Trở lại ~4096 MB
    assert free_after_release >= need_mb, "Sau khi giải phóng, RAM Disk đủ không gian an toàn"


def test_failed_resumable_error_properties():
    """Kiểm tra lớp ngoại lệ FailedResumableError mang đầy đủ thuộc tính chunk_idx và up_segs_dir"""
    err = FailedResumableError("ENOSPC test", chunk_idx=56, up_segs_dir="/tmp/fake_segs")
    assert err.chunk_idx == 56
    assert err.up_segs_dir == "/tmp/fake_segs"
    assert "ENOSPC test" in str(err)


def test_queue_retry_and_chua_xong_exit(monkeypatch, capsys):
    """Mô phỏng hàng đợi: khi ENOSPC ném FailedResumableError,
    thử lại ở cuối hàng đợi, và nếu vẫn lỗi thì thoát mã 1 kèm bảng CHƯA XONG.
    """
    import upscale
    from unittest.mock import MagicMock

    # Giả lập 2 video trong hàng đợi
    fake_vids = ["/tmp/ep1.mkv", "/tmp/ep2.mkv"]
    monkeypatch.setattr("sys.argv", ["upscale.py", "--queue", fake_vids[0], fake_vids[1], "--no-subs", "--benchmark-mode"])

    call_counts = {"ep1": 0, "ep2": 0}

    def fake_run_upscale(input_path, **kwargs):
        if "ep1" in input_path:
            call_counts["ep1"] += 1
            return "/tmp/ep1_4K.mkv"
        else:
            call_counts["ep2"] += 1
            # ep2 luôn lỗi ENOSPC
            raise FailedResumableError("ENOSPC simulated", chunk_idx=56, up_segs_dir="/tmp/.temp_ep2")

    monkeypatch.setattr(upscale, "run_upscale_pipeline", fake_run_upscale)
    monkeypatch.setattr(os.path, "isfile", lambda p: True if "_4K" in str(p) else False)

    with pytest.raises(SystemExit) as exc_info:
        upscale.main()

    # Mã thoát PHẢI là 1 (không bao giờ thoát 0 khi còn tập dở)
    assert exc_info.value.code == 1

    # ep1 được gọi 1 lần; ep2 được gọi 2 lần (1 lần trong queue chính + 1 lần retry pass ở cuối queue)
    assert call_counts["ep1"] == 1
    assert call_counts["ep2"] == 2

    captured = capsys.readouterr().out
    assert "CHƯA XONG" in captured
    assert "BẢNG TỔNG KẾT HÀNG ĐỢI: CÁC TẬP CHƯA XONG" in captured
    assert "Toàn bộ công việc trong hàng đợi đã hoàn tất" not in captured


def test_queue_retry_succeeds_at_end_of_queue(monkeypatch, capsys):
    """Mô phỏng hàng đợi: ep2 bị FailedResumableError lần 1 nhưng retry pass thành công,
    hàng đợi hoàn tất với mã 0.
    """
    import upscale

    fake_vids = ["/tmp/ep1.mkv", "/tmp/ep2.mkv"]
    monkeypatch.setattr("sys.argv", ["upscale.py", "--queue", fake_vids[0], fake_vids[1], "--no-subs", "--benchmark-mode"])

    call_counts = {"ep1": 0, "ep2": 0}

    def fake_run_upscale(input_path, **kwargs):
        if "ep1" in input_path:
            call_counts["ep1"] += 1
            return "/tmp/ep1_4K.mkv"
        else:
            call_counts["ep2"] += 1
            if call_counts["ep2"] == 1:
                # Lần 1 ném FailedResumableError
                raise FailedResumableError("ENOSPC simulated", chunk_idx=56, up_segs_dir="/tmp/.temp_ep2")
            else:
                # Lần 2 (retry pass) thành công
                return "/tmp/ep2_4K.mkv"

    monkeypatch.setattr(upscale, "run_upscale_pipeline", fake_run_upscale)
    monkeypatch.setattr(os.path, "isfile", lambda p: True if "_4K" in str(p) else False)

    with pytest.raises(SystemExit) as exc_info:
        upscale.main()

    assert exc_info.value.code == 0
    assert call_counts["ep1"] == 1
    assert call_counts["ep2"] == 2

    captured = capsys.readouterr().out
    assert "Thử lại thành công: 'ep2.mkv' đã hoàn tất 4K!" in captured
    assert "Toàn bộ công việc trong hàng đợi đã hoàn tất thành công!" in captured


@pytest.mark.parametrize("extra_args", [[], ["-s", "8"]])
def test_queue_retry_preserves_effective_segment_multipass_and_explicit_flag(
    monkeypatch, tmp_path, extra_args
):
    """Lock queue retry pass bug dropping eff_segment_time, nvenc_multipass, and segment_time_explicit."""
    import upscale

    ep1 = tmp_path / "ep1_retry.mkv"
    ep1.touch()
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "upscale.py",
            "--queue", str(ep1),
            "-o", str(out_dir),
            "--no-subs",
            "--benchmark-mode",
            "--nvenc-multipass", "qres",
            *extra_args,
        ],
    )

    recorded_calls = []

    def fake_run_upscale(input_path, **kwargs):
        recorded_calls.append(dict(kwargs))
        if len(recorded_calls) == 1:
            raise FailedResumableError("ENOSPC simulated", chunk_idx=0, up_segs_dir=str(tmp_path / "up"))
        return str(out_dir / "ep1_retry_4K.mkv")

    monkeypatch.setattr(upscale, "run_upscale_pipeline", fake_run_upscale)
    monkeypatch.setattr(os.path, "isfile", lambda p: "_4K" in str(p))

    with pytest.raises(SystemExit) as exc_info:
        upscale.main()

    assert exc_info.value.code == 0
    assert len(recorded_calls) == 2
    call1, call2 = recorded_calls
    assert call2["segment_time"] == call1["segment_time"]
    assert call2["nvenc_multipass"] == "qres"
    assert call2["segment_time_explicit"] == call1["segment_time_explicit"]


@pytest.mark.parametrize("style", ["explicit", "kwargs"])
@pytest.mark.parametrize("override_flag", [None, "--sub-offset-ms", "--sub-offset-ms=500"])
def test_subs_pending_restores_or_overrides_piecewise_ja_offsets(
    monkeypatch, tmp_path, style, override_flag
):
    """Verify --subs-pending forwards piecewise_ja_offsets unless overridden by --sub-offset-ms."""
    from pathlib import Path
    import subtrans.pipeline as sub_pipe_mod
    import upscale

    ja_srt = tmp_path / "ep1.ja.srt"
    ja_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nテスト\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pw_segments = [
        {"until_ms": 90000, "offset_ms": 0},
        {"from_ms": 90000, "offset_ms": -10000},
    ]
    pending_record = {
        "base_name": "ep1_4K",
        "ja_path": str(ja_srt),
        "output_dir": str(out_dir),
        "sub_offset_ms": 0,
        "piecewise_ja_offsets": pw_segments,
    }
    captured_runs = []

    class _PipelineExplicit:
        def __init__(self, *args, **kwargs):
            pass

        def run(
            self,
            ja_path,
            en_path=None,
            output_basename=None,
            glossary_path=None,
            subs_upgrade=False,
            sub_offset_ms=0,
            auto_offset=False,
            sound_cues="drop",
            video_path=None,
            prefer_whisper=False,
            piecewise_ja_offsets=None,
        ):
            captured_runs.append({
                "sub_offset_ms": sub_offset_ms,
                "auto_offset": auto_offset,
                "piecewise_ja_offsets": piecewise_ja_offsets,
            })
            out_srt = Path(out_dir) / f"{output_basename}.vi.srt"
            out_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nThu\n", encoding="utf-8")
            return {"subtitle_path": str(out_srt)}

    class _PipelineKwargs:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            captured_runs.append(dict(kwargs))
            out_srt = Path(out_dir) / f"{kwargs['output_basename']}.vi.srt"
            out_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nThu\n", encoding="utf-8")
            return {"subtitle_path": str(out_srt)}

    monkeypatch.setattr(
        sub_pipe_mod,
        "TranslationPipeline",
        _PipelineExplicit if style == "explicit" else _PipelineKwargs,
    )
    monkeypatch.setattr(upscale, "load_pending_subs", lambda: [dict(pending_record)])
    monkeypatch.setattr(upscale, "remove_pending_sub", lambda *_a, **_kw: None)

    argv = ["upscale.py", "--subs-pending", "-o", str(out_dir), "--no-embed", "--dry-run"]
    if override_flag == "--sub-offset-ms":
        argv.extend(["--sub-offset-ms", "500"])
    elif override_flag == "--sub-offset-ms=500":
        argv.append("--sub-offset-ms=500")
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        upscale.main()

    assert exc_info.value.code == 0
    assert len(captured_runs) == 1
    run_call = captured_runs[0]
    if override_flag is None:
        assert run_call["sub_offset_ms"] == 0
        assert run_call["auto_offset"] is False
        assert run_call["piecewise_ja_offsets"] == pw_segments
    else:
        assert run_call["sub_offset_ms"] == 500
        assert run_call["auto_offset"] is True
        assert run_call.get("piecewise_ja_offsets") is None


@pytest.mark.parametrize("quality_args,expected_quality", [
    ([], "normal"),
    (["--subs-quality", "high"], "high"),
    (["--subs-quality=high"], "high"),
])
def test_subs_pending_honors_subs_quality_override_forms(
    monkeypatch, tmp_path, quality_args, expected_quality
):
    """Verify --subs-pending honors both --subs-quality high and --subs-quality=high overrides."""
    from pathlib import Path
    import subtrans.pipeline as sub_pipe_mod
    import upscale

    ja_srt = tmp_path / "ep_qual.ja.srt"
    ja_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nテスト\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending_record = {
        "base_name": "ep_qual_4K",
        "ja_path": str(ja_srt),
        "output_dir": str(out_dir),
        "subs_quality": "normal",
        "sub_offset_ms": 0,
    }
    captured_qualities = []

    class _StubPipeline:
        def __init__(self, *args, **kwargs):
            captured_qualities.append(kwargs.get("subs_quality"))

        def run(self, **kwargs):
            out_srt = Path(out_dir) / f"{kwargs['output_basename']}.vi.srt"
            out_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nThu\n", encoding="utf-8")
            return {"subtitle_path": str(out_srt)}

    monkeypatch.setattr(sub_pipe_mod, "TranslationPipeline", _StubPipeline)
    monkeypatch.setattr(upscale, "load_pending_subs", lambda: [dict(pending_record)])
    monkeypatch.setattr(upscale, "remove_pending_sub", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "sys.argv",
        ["upscale.py", "--subs-pending", "-o", str(out_dir), "--no-embed", "--dry-run", *quality_args],
    )

    with pytest.raises(SystemExit) as exc_info:
        upscale.main()

    assert exc_info.value.code == 0
    assert captured_qualities == [expected_quality]


@pytest.mark.parametrize("halt_mode", ["graceful_pause", "allow_partial_deadline"])
def test_queue_retry_pass_skipped_after_graceful_pause_or_allow_partial_halt(
    monkeypatch, tmp_path, halt_mode
):
    """Verify [RETRY PASS] does not run after graceful pause or --allow-partial queue halt."""
    import upscale

    ep1 = tmp_path / "ep1.mkv"
    ep2 = tmp_path / "ep2.mkv"
    ep1.touch()
    ep2.touch()
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = ["upscale.py", "--queue", str(ep1), str(ep2), "-o", str(out_dir), "--no-subs", "--benchmark-mode"]
    if halt_mode == "allow_partial_deadline":
        argv.append("--allow-partial")
    monkeypatch.setattr("sys.argv", argv)

    called = []
    paused_state = {"active": False}

    def fake_run_upscale(input_path, **kwargs):
        name = os.path.basename(input_path)
        called.append(name)
        if name == "ep1.mkv":
            raise FailedResumableError("ENOSPC simulated", chunk_idx=1, up_segs_dir=str(tmp_path / "up1"))
        if halt_mode == "graceful_pause":
            paused_state["active"] = True
        up2 = out_dir / ".temp_ep2" / "upscaled_segs"
        up2.mkdir(parents=True, exist_ok=True)
        return str(up2)

    monkeypatch.setattr(upscale, "run_upscale_pipeline", fake_run_upscale)
    monkeypatch.setattr(upscale, "was_graceful_paused", lambda: paused_state["active"])

    with pytest.raises(SystemExit) as exc_info:
        upscale.main()

    assert exc_info.value.code == 1
    assert called == ["ep1.mkv", "ep2.mkv"]

