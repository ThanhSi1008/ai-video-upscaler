# -*- coding: utf-8 -*-
"""
tests/test_cli_and_shows_polish.py
Kiểm thử đồng bộ CLI, khớp mẫu ngoặc vuông shows.yaml, mở rộng hàng đợi theo thư mục
và cấu hình thuần Cloud của subtrans:
1. Mọi khóa trong DEDUP_POLICY_PARAMS (gồm 'sakuga', 'dis-sakuga') đều hợp lệ trên CLI upscale.py.
2. match_show_pattern / resolve_model_for_video / resolve_dedup_policy xử lý đúng dấu ngoặc vuông '[SubsPlease]', '[BD 1080p]'.
3. expand_video_inputs quét thư mục theo thứ tự tập tự nhiên, lọc bỏ file ẩn và bản thành phẩm *_4K.
4. subtrans/config.yaml & subtrans/models.yaml sạch cấu hình thừa ollama/translategemma/local.
"""

from pathlib import Path
from unittest.mock import patch
import pytest
import yaml

from upscale import build_arg_parser, expand_video_inputs
from src.dedup import DEDUP_POLICY_PARAMS, resolve_dedup_policy
from src.models_resolver import match_show_pattern, resolve_model_for_video
from subtrans.config import load_config, load_models_config


def test_cli_dedup_policy_choices_match_engine_registry():
    """Đảm bảo mọi policy trong src/dedup.py (bao gồm sakuga & dis-sakuga) đều gõ được trên CLI upscale.py."""
    parser = build_arg_parser()
    dedup_action = next(a for a in parser._actions if a.dest == "dedup_policy")
    assert set(dedup_action.choices) == set(DEDUP_POLICY_PARAMS.keys())

    for pol in ("sakuga", "dis-sakuga", "dis", "default", "off"):
        args = parser.parse_args(["ep01.mkv", "--dedup-policy", pol])
        assert args.dedup_policy == pol


def test_bracket_safe_show_pattern_matching(tmp_path: Path):
    """Kiểm tra khớp mẫu tên file chứa ngoặc vuông '[SubsPlease]', '[BD 1080p]' không bị lỗi fnmatch."""
    sp_file = "[SubsPlease] Sousou no Frieren - 02 (1080p) [A1B2C3D4].mkv"
    bd_file = "[AC] Fate Apocrypha - 22 [BD 1080p Hi10 FLAC][DualAudio][695AB63E].mkv"

    assert match_show_pattern(sp_file, "[SubsPlease]*") is True
    assert match_show_pattern(sp_file, "*[subsplease]*") is True
    assert match_show_pattern(bd_file, "*[BD*") is True
    assert match_show_pattern(bd_file, "[SubsPlease]*") is False

    shows_yaml = tmp_path / "shows.yaml"
    shows_yaml.write_text(
        """
shows:
  - pattern: "[SubsPlease]*"
    model: "animejanai-hd-v3-ultracompact"
    dedup_policy: "default"
  - pattern: "*[BD*"
    model: "animejanai-hd-v3-sharp1-ultracompact"
    dedup_policy: "sakuga"
""",
        encoding="utf-8",
    )

    with patch("src.models_resolver.SHOWS_YAML", str(shows_yaml)), patch(
        "src.dedup.ROOT_DIR", str(tmp_path)
    ):
        m_sp = resolve_model_for_video(sp_file, None)
        assert m_sp["id"] == "animejanai-hd-v3-ultracompact"
        assert resolve_dedup_policy(sp_file, None) == "default"

        m_bd = resolve_model_for_video(bd_file, None)
        assert m_bd["id"] == "animejanai-hd-v3-sharp1-ultracompact"
        assert resolve_dedup_policy(bd_file, None) == "sakuga"


def test_expand_video_inputs_directory_natural_sort_and_skip_4k(tmp_path: Path):
    """Quét thư mục sắp xếp đúng Tập 2 trước Tập 10, bỏ qua file ẩn và bản xuất _4K."""
    ep10 = tmp_path / "[SubsPlease] Show - 10 (1080p).mkv"
    ep2 = tmp_path / "[SubsPlease] Show - 2 (1080p).mkv"
    already_4k = tmp_path / "[SubsPlease] Show - 1 (1080p)_4K.mkv"
    already_aj = tmp_path / "[SubsPlease] Show - 1 (1080p)_4K_animejanai.mkv"
    hidden_clip = tmp_path / ".preview_clip_Show.mkv"
    notes = tmp_path / "notes.txt"

    for f in (ep10, ep2, already_4k, already_aj, hidden_clip, notes):
        f.write_bytes(b"dummy")

    result = expand_video_inputs([str(tmp_path), str(ep2)])
    assert result == [str(ep2), str(ep10)]


def test_expand_video_inputs_empty_or_only_4k_directory_raises(tmp_path: Path):
    """Thư mục chỉ chứa bản thành phẩm _4K phải báo lỗi rõ ràng thay vì chạy lặp đệ quy."""
    (tmp_path / "Ep01_4K.mkv").write_bytes(b"done")
    with pytest.raises(ValueError, match="không chứa file video gốc hợp lệ"):
        expand_video_inputs([str(tmp_path)])


def test_shows_yaml_valid_and_subtrans_configs_purged_of_ollama():
    """Đảm bảo shows.yaml hợp lệ và cấu hình subtrans không còn mục ollama/translategemma/local."""
    shows_data = yaml.safe_load(Path("shows.yaml").read_text(encoding="utf-8"))
    assert isinstance(shows_data, dict)
    assert shows_data.get("shows") == []

    sub_cfg = load_config()
    assert "ollama" not in sub_cfg
    assert "local_batch_size" not in sub_cfg.get("translation", {})

    mod_cfg = load_models_config()
    assert "translategemma:12b" not in mod_cfg.get("models", {})
    assert "local" not in mod_cfg.get("chain", [])
