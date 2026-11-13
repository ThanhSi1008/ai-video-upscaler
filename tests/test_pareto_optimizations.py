# -*- coding: utf-8 -*-
"""
tests/test_pareto_optimizations.py
Bộ kiểm thử hồi quy cho toàn bộ 3 gói tối ưu hóa Pareto (Core/CLI, Subtrans, Video Pipeline).
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.config as app_config
from src.cuda_engine import GPUPersistentWorker
from src.dedup import build_preflight_subprocess_spec
from src.night_report import parse_telemetry_window
from src.subtitles import find_or_extract_subtitles
from src.trt_zero_copy_upscaler import DoubleBufferedPipeUpscaler, SRVGGNetCompactTRTWrapper
from subtrans.cache import TranslationCache
from subtrans.cascade import run_cascade_repair_pass
from subtrans.dtw_auto_align import SubCue, get_dtw_alignment_map
from subtrans.gemini_backend import GeminiTranslator, filter_relevant_glossary
from subtrans.glossary import compute_glossary_hash, load_glossary, resolve_glossary_target
from subtrans.models import Character, CharacterSpeaksTo, Cue, Glossary, GlossaryTerm
from subtrans.parser import snap_whisper_cues_to_en
from subtrans.pipeline import TranslationPipeline
import upscale as upscale_cli


def test_night_report_parse_telemetry_window_thermal_states(tmp_path: Path):
    csv_file = tmp_path / "telemetry.csv"
    csv_file.write_text(
        "epoch,timestamp,disk0_mb_total,pageouts,swap_used_mb,thermal_state\n"
        "1000,2026-11-13T01:00:00,100.0,10,64.0,Nominal\n"
        "1060,2026-11-13T01:01:00,124.0,12,96.0,Fair\n",
        encoding="utf-8",
    )
    res = parse_telemetry_window(str(csv_file), start_epoch=1000, end_epoch=1100)
    assert "error" not in res
    assert res["delta_ssd_mb"] == pytest.approx(24.0)
    assert res["thermal_states"] == ["Fair", "Nominal"]


def test_pid_aware_gpu_lock_and_pending_sub_matching(tmp_path: Path, monkeypatch):
    lock_path = tmp_path / ".gpu_active.lock"
    pending_path = tmp_path / "pending_subs.json"
    monkeypatch.setattr(app_config, "GPU_LOCK_FILE", str(lock_path))
    monkeypatch.setattr(app_config, "PENDING_SUBS_FILE", str(pending_path))
    monkeypatch.setattr(app_config, "_GPU_LOCK_OWNED", False)

    # 1. Stale lock từ PID không tồn tại phải được tự động thu hồi
    lock_path.write_text("99999999\n", encoding="utf-8")
    monkeypatch.setattr("psutil.pid_exists", lambda pid: False)
    assert app_config.is_gpu_lock_held_by_other() is None
    app_config.acquire_gpu_lock()
    assert app_config._GPU_LOCK_OWNED is True
    assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())
    app_config.release_gpu_lock()
    assert not lock_path.exists()

    # 2. Khóa thuộc PID khác đang sống: cấm ghi đè và cấm xóa khi atexit/release
    foreign_pid = os.getpid() + 4242
    lock_path.write_text(f"{foreign_pid}\n", encoding="utf-8")
    monkeypatch.setattr("psutil.pid_exists", lambda pid: pid == foreign_pid)
    assert app_config.is_gpu_lock_held_by_other() == foreign_pid
    with pytest.raises(RuntimeError, match="GPU đang bận"):
        app_config.acquire_gpu_lock()
    app_config._atexit_release_gpu_lock()
    app_config.release_gpu_lock()
    assert lock_path.exists()

    # 3. remove_pending_sub khớp được cả base_name, video_path và hậu tố _4K
    app_config.add_pending_sub({
        "base_name": "Ep01_Test",
        "video_path": str(tmp_path / "Ep01_Test.mkv"),
        "ja_path": str(tmp_path / "Ep01_Test.ja.srt"),
    })
    assert len(app_config.load_pending_subs()) == 1
    app_config.remove_pending_sub("Ep01_Test_4K")
    assert app_config.load_pending_subs() == []


def test_upscale_cli_fast_quality_alias_and_clean_temp_dry_run(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["upscale.py", "ep01.mkv", "--subs-quality", "fast", "--dry-run"])
    args = upscale_cli.parse_args()
    assert args.subs_quality == "normal"
    assert args.dry_run is True

    temp_dir = tmp_path / ".temp_Anime_Ep01"
    temp_dir.mkdir()
    (temp_dir / "up_0000.mp4").write_bytes(b"00" * 64)

    monkeypatch.setattr(sys, "argv", ["upscale.py", "--clean-temp", "-o", str(tmp_path), "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        upscale_cli.main()
    assert exc.value.code == 0
    assert temp_dir.exists(), "Chế độ --dry-run tuyệt đối không được xóa thư mục .temp_*"


def test_preflight_subprocess_spec_windows_safe_argv():
    tricky_video = r"E:\Anime Library\Fate's Stay [1080p]\Ep 01.mkv"
    tricky_plan = r"E:\Anime Library\out\.temp_Fate's Stay\plan.json"
    cmd, kwargs = build_preflight_subprocess_spec(tricky_video, tricky_plan, segment_seconds=8)

    assert cmd[0] == sys.executable
    assert cmd[1] == "-c"
    assert tricky_video in cmd[3:]
    assert tricky_plan in cmd[3:]
    assert "Fate's" not in cmd[2], "Không được nội suy trực tiếp đường dẫn chứa dấu nháy đơn vào chuỗi Python -c"
    if sys.platform == "win32":
        assert "creationflags" in kwargs
        assert cmd[0] != "nice"


class _TinyDummySRVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.ModuleList([nn.Conv2d(3, 12, 3, 1, 1)])
        self.upsampler = nn.PixelShuffle(2)
        self.upscale = 2


def test_trt_wrapper_scale_kwarg_and_dynamic_resolution_pipe():
    base = _TinyDummySRVGG().eval()
    wrapper = SRVGGNetCompactTRTWrapper(base, scale=2.0).eval()
    x = torch.rand(1, 3, 16, 24, dtype=torch.float32)
    with torch.inference_mode():
        y = wrapper(x)
    assert tuple(y.shape) == (1, 3, 32, 48)

    pipe_up = DoubleBufferedPipeUpscaler(
        model_or_engine=wrapper,
        batch_size=2,
        device=torch.device("cpu"),
        input_hw=(36, 64),
        scale=2,
    )
    assert pipe_up.output_hw == (72, 128)
    out_hw = pipe_up.ensure_resolution(40, 80)
    assert out_hw == (80, 160)
    assert tuple(pipe_up.host_in_pins[0].shape) == (2, 3, 40, 80)
    assert tuple(pipe_up.host_out_pins[0].shape) == (2, 80 * 3 // 2, 160)


def test_cuda_engine_upscale_directory_bgr_channel_fidelity(tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    # Tạo ảnh thuần màu Xanh Dương (Pure Blue trong BGR = [255, 0, 0])
    blue_bgr = np.zeros((16, 24, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255
    blue_bgr[:, :, 1] = 32
    blue_bgr[:, :, 2] = 0
    cv2.imwrite(str(in_dir / "frame_0001.png"), blue_bgr)

    worker = GPUPersistentWorker.__new__(GPUPersistentWorker)
    worker.device = torch.device("cpu")
    worker.precision = "fp32"
    worker.batch_size = 2
    worker.scale = 2
    worker.chunks_processed = 0
    worker.total_restarts = 0
    worker.max_chunks_before_restart = 0
    worker.model = lambda t: F.interpolate(t, scale_factor=2.0, mode="nearest")

    stats = worker.upscale_directory(str(in_dir), str(out_dir), fmt="png", batch_size=2)
    assert stats["num_frames"] == 1

    saved_bgr = cv2.imread(str(out_dir / "frame_0001.png"), cv2.IMREAD_COLOR)
    assert saved_bgr is not None
    assert saved_bgr.shape == (32, 48, 3)
    # Kênh Blue phải giữ nguyên 255, kênh Red phải giữ nguyên 0 (không bị đảo kênh BGR <-> RGB)
    assert int(saved_bgr[0, 0, 0]) == 255
    assert int(saved_bgr[0, 0, 1]) == 32
    assert int(saved_bgr[0, 0, 2]) == 0


def test_subtitles_skips_bitmap_pgs_and_preserves_srt_vs_ass(tmp_path: Path, monkeypatch):
    fake_mkv = tmp_path / "Show_Ep01.mkv"
    fake_mkv.write_bytes(b"fake_mkv")

    probe_json = {
        "streams": [
            {
                "index": 2,
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "eng", "title": "Blu-ray PGS"},
                "disposition": {"default": 1},
            },
            {
                "index": 3,
                "codec_name": "subrip",
                "tags": {"language": "eng", "title": "Full Dialogue"},
                "disposition": {"default": 0},
            },
            {
                "index": 4,
                "codec_name": "ass",
                "tags": {"language": "jpn", "title": "Japanese CC"},
                "disposition": {"default": 0},
            },
        ]
    }
    recorded_ffmpeg_cmds = []

    def _fake_run(cmd, *args, **kwargs):
        if cmd[0] == "ffprobe":
            res = MagicMock()
            res.returncode = 0
            res.stdout = json.dumps(probe_json)
            return res
        if cmd[0] == "ffmpeg":
            recorded_ffmpeg_cmds.append(cmd)
            out_file = Path(cmd[-1])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello dialogue line!\n", encoding="utf-8")
            res = MagicMock()
            res.returncode = 0
            return res
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    ja_sub, en_sub = find_or_extract_subtitles(str(fake_mkv))

    assert en_sub is not None and en_sub.endswith(".srt")
    assert ja_sub is not None and ja_sub.endswith(".ass")
    mapped_streams = [cmd[cmd.index("-map") + 1] for cmd in recorded_ffmpeg_cmds]
    assert "0:2" not in mapped_streams, "Phụ đề ảnh bitmap hdmv_pgs_subtitle phải bị bỏ qua"
    assert "0:3" in mapped_streams and "0:4" in mapped_streams


def test_glossary_pointer_resolution_hash_and_character_metadata(tmp_path: Path):
    target_yaml = tmp_path / "glossary.tenken.yaml"
    target_yaml.write_text(
        "terms:\n"
        "  - ja: 魔剣\n"
        "    vi: ma kiếm\n"
        "characters:\n"
        "  - name_ja: フラン\n"
        "    name_vi: Fran\n"
        "    gender: female\n"
        "    register: stoic cute\n"
        "    note: Black Cat girl\n",
        encoding="utf-8",
    )
    pointer_yaml = tmp_path / "glossary.reincarnatedasasword.yaml"
    pointer_yaml.write_text("glossary.tenken.yaml\n", encoding="utf-8")

    resolved = resolve_glossary_target(pointer_yaml)
    assert resolved == target_yaml.resolve()

    g1 = load_glossary(pointer_yaml)
    assert g1 is not None
    assert g1.characters[0].speaking_style == "stoic cute"
    assert g1.characters[0].note == "Black Cat girl"

    g2 = Glossary(
        terms=[GlossaryTerm(ja="魔剣", vi="thánh kiếm")],
        characters=g1.characters,
    )
    assert compute_glossary_hash(g1) != compute_glossary_hash(g2)

    rel = filter_relevant_glossary([Cue(id=1, start=0, end=1000, ja="魔剣よ")], g1)
    assert rel["characters"][0]["gender"] == "female"
    assert rel["characters"][0]["register"] == "stoic cute"
    assert rel["characters"][0]["note"] == "Black Cat girl"


def test_cascade_prompt_formats_glossary_terms_and_full_cast(tmp_path: Path):
    mock_translator = GeminiTranslator(api_key="mock_key", model="gemini-3.8-flash", dry_run=True)
    mock_translator.dry_run = False
    mock_translator.cache = TranslationCache(db_path=tmp_path / "cascade_cache.db")
    mock_translator.client = MagicMock()
    captured = {}

    def _fake_gen(model, contents, config):
        captured["contents"] = contents
        resp = MagicMock()
        resp.text = json.dumps([
            {"id": 1, "speaker": "Fran", "addressee": "Teacher", "mode": "spoken", "vi": "Sư phụ, ma kiếm sáng lên!", "changed": True}
        ])
        return resp

    mock_translator.client.models.generate_content.side_effect = _fake_gen

    cues = [
        Cue(
            id=1,
            start=0,
            end=2000,
            ja="魔剣が光った！",
            en="Master, the magic sword glowed!",
            vi="Kiếm phép sáng!",
            draft_vi="Kiếm phép sáng!",
            uncertain=True,
            uncertain_reason="term",
        )
    ]
    glossary = Glossary(
        terms=[GlossaryTerm(ja="魔剣", vi="ma kiếm", note="Không dịch là kiếm phép")],
        characters=[
            Character(
                name_ja="フラン",
                name_vi="Fran",
                gender="female",
                speaking_style="stoic",
                speaks_to=[CharacterSpeaksTo(listener="Teacher", self="con", other="Sư phụ")],
            )
        ],
    )
    stats = run_cascade_repair_pass(cues, mock_translator, glossary=glossary, difficulty_threshold=1.0)
    assert stats["repaired_count"] == 1
    assert "- 魔剣 → ma kiếm" in captured["contents"]
    assert "con/Sư phụ -> Teacher" in captured["contents"]
    assert "- terms →" not in captured["contents"]


def test_sqlite_cache_wal_batch_io_and_cot_metadata(tmp_path: Path):
    db_file = tmp_path / "cache_wal.db"
    cache = TranslationCache(db_path=db_file)

    with sqlite3.connect(db_file) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"

    written = cache.set_cues_batch([
        (
            "k1",
            "gemini",
            "gemini-3.5-flash-lite",
            "Chào Sư phụ!",
            {"speaker": "Fran", "addressee": "Teacher", "mode": "spoken", "uncertain": False, "en_conflict": True},
        ),
        ("k_dry", "gemini", "gemini-3.5-flash-lite", "[DRY-RUN VI] Hello", None),
        ("k_untrans", "untranslated", "none", "[Chưa dịch]", None),
    ])
    assert written == 1

    batch_res = cache.get_cues_batch(["k1", "k_dry", "k_untrans", "missing"])
    assert list(batch_res.keys()) == ["k1"]
    vi_text, meta = batch_res["k1"]
    assert vi_text == "Chào Sư phụ!"
    assert meta["speaker"] == "Fran"
    assert meta["addressee"] == "Teacher"
    assert meta["en_conflict"] is True


def test_snap_whisper_cues_overwrites_stale_en_and_banded_dtw_parity():
    cues = [
        Cue(id=1, start=1200, end=2900, ja="フラン、行くぞ！", en="WRONG OLD OVERLAP"),
        Cue(id=2, start=3200, end=4900, ja="うん、師匠！", en="WRONG SECOND"),
    ]
    en_events = [
        {"id": 0, "start": 1000, "end": 2800, "duration": 1800, "text": "Fran, let's go!"},
        {"id": 1, "start": 3100, "end": 4800, "duration": 1700, "text": "Yeah, Master!"},
    ]
    snapped = snap_whisper_cues_to_en(
        cues,
        en_events,
        cast_names_ja_en=[("フラン", "Fran"), ("師匠", "Master")],
    )
    assert snapped == 2
    assert cues[0].en == "Fran, let's go!"
    assert cues[1].en == "Yeah, Master!"

    ja_sc = [SubCue(1200, 2900, "フラン、行くぞ！"), SubCue(3200, 4900, "うん、師匠！")]
    en_sc = [SubCue(1000, 2800, "Fran, let's go!"), SubCue(3100, 4800, "Yeah, Master!")]
    m = get_dtw_alignment_map(ja_sc, en_sc, cast_names_ja_en=[("フラン", "Fran")])
    assert m[0] == (1000, 2800, "Fran, let's go!")


def test_pipeline_dry_run_isolation_and_hashlib_scope(tmp_path: Path, monkeypatch):
    import subtrans.rag as rag_mod

    pending_file = tmp_path / "pending_subs.json"
    monkeypatch.setattr(app_config, "PENDING_SUBS_FILE", str(pending_file))
    monkeypatch.setattr(rag_mod, "KNOWLEDGE_DB_PATH", tmp_path / "knowledge.db")

    ja_srt = tmp_path / "TestEp01.ja.srt"
    ja_srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nルーデウス、行くわよ！\n\n"
        "2\n00:00:03,500 --> 00:00:05,500\n分かった、シルフィ。\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out_dry"
    cache_db = tmp_path / "isolated_cache.db"

    pipe = TranslationPipeline(
        output_dir=out_dir,
        dry_run=True,
        enable_rag=True,
        enable_cascade=False,
        cache_db_path=cache_db,
    )
    # Khi chưa có cast_file sẵn trên đĩa và glossary có nhân vật -> kiểm chứng nhánh hashlib ở dòng 395
    with patch("subtrans.pipeline.load_or_generate_series_cast_sheet") as mock_cast:
        mock_cast.return_value = Glossary(
            characters=[Character(name_ja="ルーデウス", name_vi="Rudeus")]
        )
        res = pipe.run(ja_path=ja_srt)

    assert Path(res["output_path"]).exists()
    assert not pending_file.exists(), "--dry-run không được tạo hoặc ghi vào pending_subs.json"
