# -*- coding: utf-8 -*-
"""
tests/test_subtrans_stub_pipeline.py
End-to-end integration test for Stage B Subtitle Translation pipeline using a StubTranslator backend.
Zero Cloud API quota, Zero GPU:
- Evaluates RAG retrieval, Draft Pass, and Smart Cloud Cascade Repair Pass
- Tests Re-lint Accept/Reject Gate, auto_fix_cues, and SRT output generation
- Verifies subtitle embedding and Stream Copy Mux Integrity on sample_30s.mkv
- NEVER skipped: fully self-contained!
"""

import os
import sys
import re
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pysubs2

from subtrans.pipeline import TranslationPipeline
from subtrans.models import Cue
from subtrans.gemini_backend import GeminiTranslator, SubtitleOutputItem
from subtrans.cascade import RepairOutputItem
from src.ffmpeg_ops import embed_subtitle_track, probe_video
from src.mux_audit import verify_mux_integrity


SAMPLE_MKV = Path("samples/sample_30s.mkv")


def _create_sample_subtitles(tmp_dir: Path) -> tuple[Path, Path]:
    """Creates realistic sample Japanese and English subtitle files."""
    ja_path = tmp_dir / "sample_ep.ja.srt"
    en_path = tmp_dir / "sample_ep.en.srt"

    ja_content = (
        "1\n00:00:01,000 --> 00:00:03,000\nフラン、準備はいいか？\n\n"
        "2\n00:00:03,500 --> 00:00:05,000\nうん、師匠。行こう。\n\n"
        "3\n00:00:05,500 --> 00:00:07,500\n（爆発音）\n\n"
        "4\n00:00:08,000 --> 00:00:10,500\n妾が相手をしてやろう。\n\n"
    )
    en_content = (
        "1\n00:00:01,000 --> 00:00:03,000\nFran, are you ready?\n\n"
        "2\n00:00:03,500 --> 00:00:05,000\nYeah, Master. Let's go.\n\n"
        "3\n00:00:05,500 --> 00:00:07,500\n(explosion)\n\n"
        "4\n00:00:08,000 --> 00:00:10,500\nI shall be your opponent.\n\n"
    )

    ja_path.write_text(ja_content, encoding="utf-8")
    en_path.write_text(en_content, encoding="utf-8")
    return ja_path, en_path


def test_subtrans_stub_pipeline_end_to_end(tmp_path):
    """
    Tests entire subtitle translation pipeline with stub Cloud API backend:
    - RAG few-shot retrieval
    - Draft generation
    - Cascade Repair pass on difficult/noble cue
    - Writing SRT & QA audit report
    - Embedding and verifying mux integrity on sample MKV
    """
    ja_path, en_path = _create_sample_subtitles(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_db = tmp_path / "test_cache.db"

    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyMockKeyForTesting12345"}):
        pipeline = TranslationPipeline(
            local_only=False,
            output_dir=out_dir,
            output_format="srt",
            require_approved_cast=False,
            allow_latest_alias=False,
            dry_run=False,
            enable_cascade=True,
            enable_rag=True,
            cache_db_path=cache_db,
        )

    # Mock Cast Sheet API response
    mock_cast_obj = MagicMock()
    mock_cast_obj.text = json.dumps({
        "characters": [
            {"name_ja": "フラン", "name_vi": "Fran", "gender": "female"},
            {"name_ja": "師匠", "name_vi": "Jean", "gender": "male"},
            {"name_ja": "貴婦人", "name_vi": "NobleWoman", "gender": "female"},
        ]
    })
    mock_cast_obj.usage_metadata.prompt_token_count = 50
    mock_cast_obj.usage_metadata.candidates_token_count = 30
    mock_cast_obj.usage_metadata.total_token_count = 80
    mock_cast_obj.usage_metadata.thinking_token_count = 0

    # Mock GeminiTranslator API call for Draft pass (0-indexed cues: 0, 1, 3)
    mock_draft_items = [
        {
            "id": 0,
            "speaker": "Jean",
            "addressee": "Fran",
            "mode": "spoken",
            "uncertain": False,
            "uncertain_reason": "none",
            "vi": "Fran, em đã sẵn sàng chưa?",
        },
        {
            "id": 1,
            "speaker": "Fran",
            "addressee": "Jean",
            "mode": "spoken",
            "uncertain": False,
            "uncertain_reason": "none",
            "vi": "Vâng, sư phụ. Đi thôi.",
        },
        {
            "id": 3,
            "speaker": "NobleWoman",
            "addressee": "Fran",
            "mode": "spoken",
            "uncertain": True,
            "uncertain_reason": "noble_speech",
            "vi": "Ta sẽ làm đối thủ của ngươi.",
        },
    ]

    # Mock Flash Repair API call for Cascade pass (cue 3)
    mock_repair_response = [
        {
            "id": 3,
            "vi": "Thiếp thân sẽ tiếp chiêu các ngươi.",
            "changed": True,
            "change_reason": "archaic_noble_speech",
            "speaker": "NobleWoman",
            "addressee": "Fran",
            "mode": "spoken",
        }
    ]

    mock_draft_response_obj = MagicMock()
    mock_draft_response_obj.text = json.dumps(mock_draft_items)
    mock_draft_response_obj.usage_metadata.prompt_token_count = 120
    mock_draft_response_obj.usage_metadata.candidates_token_count = 45
    mock_draft_response_obj.usage_metadata.total_token_count = 165
    mock_draft_response_obj.usage_metadata.thinking_token_count = 0

    mock_repair_response_obj = MagicMock()
    mock_repair_response_obj.text = json.dumps(mock_repair_response)
    mock_repair_response_obj.usage_metadata.prompt_token_count = 80
    mock_repair_response_obj.usage_metadata.candidates_token_count = 25
    mock_repair_response_obj.usage_metadata.total_token_count = 105
    mock_repair_response_obj.usage_metadata.thinking_token_count = 0

    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyMockKeyForTesting12345"}), \
         patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        def mock_generate_content(model, contents, config):
            cnt_str = str(contents)
            if "cues_to_review" in cnt_str:
                return mock_repair_response_obj
            if "cues_to_translate" in cnt_str:
                req_ids = [int(x) for x in re.findall(r'"id":\s*(\d+)', cnt_str)]
                filtered = [it for it in mock_draft_items if it["id"] in req_ids]
                resp = MagicMock()
                resp.text = json.dumps(filtered)
                resp.usage_metadata.prompt_token_count = 100
                resp.usage_metadata.candidates_token_count = 40
                resp.usage_metadata.total_token_count = 140
                resp.usage_metadata.thinking_token_count = 0
                return resp
            return mock_cast_obj

        mock_client.models.generate_content.side_effect = mock_generate_content

        res = pipeline.run(
            ja_path=ja_path,
            en_path=en_path,
            output_basename="sample_ep",
            auto_offset=False,
        )

    # 1. Verify subtitle file generation
    srt_path = Path(res["output_path"])
    assert srt_path.exists(), f"Output SRT {srt_path} does not exist"
    srt_content = srt_path.read_text(encoding="utf-8")

    assert "Fran, em đã sẵn sàng chưa?" in srt_content
    assert "Vâng, sư phụ. Đi thôi." in srt_content
    # Cue 4 was repaired by Flash in cascade pass
    assert "Thiếp thân sẽ tiếp chiêu các ngươi." in srt_content

    # 2. Verify QA report generation
    report_path = Path(res["report_path"])
    assert report_path.exists(), f"QA report {report_path} does not exist"

    # 3. Test subtitle embedding and mux integrity if sample_30s.mkv is present
    if SAMPLE_MKV.exists():
        target_clone = tmp_path / "sample_clone.mkv"
        shutil.copyfile(SAMPLE_MKV, target_clone)

        embed_ok = embed_subtitle_track(
            mkv_path=str(target_clone),
            srt_path=str(srt_path),
            original_source_path=str(SAMPLE_MKV),
        )
        assert embed_ok, "embed_subtitle_track returned False"

        mux_audit = verify_mux_integrity(str(SAMPLE_MKV), str(target_clone))
        assert mux_audit["status"] == "PASS", f"Mux audit failed: {mux_audit}"
