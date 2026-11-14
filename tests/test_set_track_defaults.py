# -*- coding: utf-8 -*-
"""
tests/test_set_track_defaults.py
Unit tests for scripts/set_track_defaults.py CLI, track matching, and simulation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from scripts.set_track_defaults import (
    collect_files,
    is_dialogue_sub,
    is_japanese_audio,
    is_signs_or_songs,
    is_vietnamese_sub,
    matches_audio_lang,
    matches_sub_lang,
    parse_args,
    process_file,
    resolve_tools,
    select_target_sub,
)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.targets == [r"E:\Films\Raw_Sources"]
    assert args.audio_lang == "ja,jpn"
    assert args.sub_lang == "vi,vie,en,eng"
    assert args.skip_pattern is None
    assert args.dry_run is False
    assert args.mkvmerge is None
    assert args.mkvpropedit is None


def test_parse_args_custom():
    args = parse_args([
        "video1.mkv", "video2.mkv",
        "--audio-lang", "en,eng",
        "--sub-lang", "vi",
        "--skip-pattern", "SP1",
        "--skip-pattern", "SP2",
        "--dry-run",
        "--mkvmerge", "custom_merge",
        "--mkvpropedit", "custom_propedit",
    ])
    assert args.targets == ["video1.mkv", "video2.mkv"]
    assert args.audio_lang == "en,eng"
    assert args.sub_lang == "vi"
    assert args.skip_pattern == ["SP1", "SP2"]
    assert args.dry_run is True
    assert args.mkvmerge == "custom_merge"
    assert args.mkvpropedit == "custom_propedit"


def test_resolve_tools():
    with mock.patch("shutil.which", return_value="/usr/bin/mkvmerge"):
        merge, prop = resolve_tools(None, None)
        assert merge == "/usr/bin/mkvmerge"

    merge_custom, prop_custom = resolve_tools("custom_m", "custom_p")
    assert merge_custom == "custom_m"
    assert prop_custom == "custom_p"


def test_matches_audio_lang():
    ja_track = {"properties": {"language": "jpn", "track_name": "Stereo"}}
    assert matches_audio_lang(ja_track, ["ja", "jpn"]) is True
    assert matches_audio_lang(ja_track, ["en"]) is False

    en_track = {"properties": {"language": "eng", "track_name": "English 5.1"}}
    assert matches_audio_lang(en_track, ["en", "eng"]) is True
    assert matches_audio_lang(en_track, ["ja"]) is False


def test_subtitle_matching_and_filtering():
    signs_track = {"properties": {"language": "eng", "track_name": "Signs & Songs"}}
    assert is_signs_or_songs(signs_track) is True
    assert is_dialogue_sub(signs_track) is False

    dialogue_track = {"properties": {"language": "eng", "track_name": "Full Dialogue"}}
    assert is_signs_or_songs(dialogue_track) is False
    assert is_dialogue_sub(dialogue_track) is True

    vi_track = {"properties": {"language": "vie", "track_name": "Tieng Viet"}}
    assert is_vietnamese_sub(vi_track) is True


def test_select_target_sub():
    subs = [
        {"properties": {"uid": 1, "language": "eng", "track_name": "Signs/Songs"}},
        {"properties": {"uid": 2, "language": "eng", "track_name": "Full Dialogue"}},
        {"properties": {"uid": 3, "language": "vie", "track_name": "Vietnamese"}},
    ]
    # Priority: vi,vie,en,eng -> Should select Vietnamese track (uid 3)
    target = select_target_sub(subs, ["vi", "vie", "en", "eng"])
    assert target is not None
    assert target["properties"]["uid"] == 3

    # Priority: en,eng -> Should select Full Dialogue (uid 2)
    target_en = select_target_sub(subs, ["en", "eng"])
    assert target_en is not None
    assert target_en["properties"]["uid"] == 2


def test_collect_files(tmp_path: Path):
    d = tmp_path / "season1"
    d.mkdir()
    f1 = d / "ep01.mkv"
    f2 = d / "ep02.mkv"
    f3 = d / "Fate.Strange.Fake.mkv"
    f1.write_bytes(b"")
    f2.write_bytes(b"")
    f3.write_bytes(b"")

    files = collect_files([str(d)], ["Fate.Strange.Fake"])
    assert len(files) == 2
    assert any("ep01.mkv" in f for f in files)
    assert any("ep02.mkv" in f for f in files)
    assert not any("Fate.Strange.Fake" in f for f in files)


def test_process_file_dry_run_simulation():
    mock_json = {
        "tracks": [
            {
                "type": "audio",
                "properties": {"uid": 101, "language": "eng", "default_track": True},
            },
            {
                "type": "audio",
                "properties": {"uid": 102, "language": "jpn", "default_track": False},
            },
            {
                "type": "subtitles",
                "properties": {"uid": 201, "language": "vie", "default_track": False},
            },
        ]
    }

    with mock.patch("subprocess.run") as mock_run:
        # Inspection returns mock_json
        mock_run.return_value = mock.MagicMock(returncode=0, stdout=json.dumps(mock_json))
        res = process_file(
            "test.mkv",
            mkvmerge_bin="mkvmerge",
            mkvpropedit_bin="mkvpropedit",
            audio_langs=["ja", "jpn"],
            sub_langs=["vi", "vie"],
            dry_run=True,
        )

        assert res["status"] == "simulated"
        assert res.get("dry_run") is True
        assert len(res["changes"]) == 3
        # Should only call mkvmerge -J, never mkvpropedit
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "-J" in cmd


def test_process_file_already_default_unchanged():
    mock_json = {
        "tracks": [
            {
                "type": "audio",
                "properties": {"uid": 102, "language": "jpn", "default_track": True},
            },
            {
                "type": "subtitles",
                "properties": {"uid": 201, "language": "vie", "default_track": True},
            },
        ]
    }

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.MagicMock(returncode=0, stdout=json.dumps(mock_json))
        res = process_file(
            "test.mkv",
            mkvmerge_bin="mkvmerge",
            mkvpropedit_bin="mkvpropedit",
            audio_langs=["ja", "jpn"],
            sub_langs=["vi", "vie"],
            dry_run=False,
        )

        assert res["status"] == "unchanged"
        mock_run.assert_called_once()


def test_process_file_executes_mkvpropedit():
    mock_json = {
        "tracks": [
            {
                "type": "audio",
                "properties": {"uid": 101, "language": "eng", "default_track": True},
            },
            {
                "type": "audio",
                "properties": {"uid": 102, "language": "jpn", "default_track": False},
            },
        ]
    }

    with mock.patch("subprocess.run") as mock_run:
        # First call is mkvmerge -J, second call is mkvpropedit
        mock_run.side_effect = [
            mock.MagicMock(returncode=0, stdout=json.dumps(mock_json)),
            mock.MagicMock(returncode=0, stderr=""),
        ]
        res = process_file(
            "test.mkv",
            mkvmerge_bin="mkvmerge",
            mkvpropedit_bin="mkvpropedit",
            audio_langs=["ja", "jpn"],
            dry_run=False,
        )

        assert res["status"] == "modified"
        assert mock_run.call_count == 2
        edit_cmd = mock_run.call_args_list[1][0][0]
        assert "mkvpropedit" in edit_cmd[0]
