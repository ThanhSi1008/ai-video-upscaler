"""Offline tests for subtrans CLI dead-flag rejection and batch parity with run."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from subtrans import cli as subtrans_cli


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def test_cli_rejects_dead_review_and_yes_flags(tmp_path: Path):
    """Verify --review/-r and --yes/-y are rejected with exit code 2 on run and batch."""
    runner = CliRunner()
    ja_srt = tmp_path / "ep01.ja.srt"
    ja_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nテスト\n", encoding="utf-8")
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "ep01.ja.srt").write_text("1\n00:00:01,000 --> 00:00:03,000\nテスト\n", encoding="utf-8")

    for args, flag in (
        (["run", str(ja_srt), "--yes"], "--yes"),
        (["run", str(ja_srt), "-y"], "-y"),
        (["run", str(ja_srt), "--review"], "--review"),
        (["run", str(ja_srt), "-r"], "-r"),
        (["batch", str(in_dir), "--yes"], "--yes"),
        (["batch", str(in_dir), "-y"], "-y"),
        (["batch", str(in_dir), "--review"], "--review"),
        (["batch", str(in_dir), "-r"], "-r"),
    ):
        res = runner.invoke(subtrans_cli.app, args)
        out_low = _plain(res.output).lower()
        assert res.exit_code == 2
        assert "no such option" in out_low
        assert flag in out_low


def test_cli_batch_exposes_run_parity_flags_and_forwards_options(tmp_path: Path):
    """Verify batch --help exposes run parity options and forwards them to TranslationPipeline."""
    runner = CliRunner()
    help_res = runner.invoke(subtrans_cli.app, ["batch", "--help"], env={"COLUMNS": "240"})
    assert help_res.exit_code == 0
    help_text = _plain(help_res.output)
    for expected_flag in (
        "--en-offset-ms",
        "--sub-offset-ms",
        "--model",
        "--allow-latest-alias",
        "--on-quota",
        "--limit",
        "--format",
        "--wrap",
        "--gloss-skill-names",
        "--require-approved-cast",
        "--review-file",
        "--subs-quality",
        "--subs-upgrade",
        "--auto-offset",
        "--no-rag",
        "--no-cascade",
        "--cache-db",
        "--auto-align-en",
        "--smart-wrap",
        "--explicit-cache",
    ):
        assert expected_flag in help_text

    in_dir = tmp_path / "in_parity"
    in_dir.mkdir()
    ja_srt = in_dir / "ep01.ja.srt"
    ja_srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nテスト\n", encoding="utf-8")
    out_dir = tmp_path / "out_parity"
    cache_db = tmp_path / "parity.db"

    captured_init = []
    captured_run = []

    class _StubPipeline:
        def __init__(self, **kwargs):
            captured_init.append(kwargs)

        def run(self, **kwargs):
            captured_run.append(kwargs)
            return {}

    with patch("subtrans.cli.TranslationPipeline", side_effect=_StubPipeline):
        res = runner.invoke(
            subtrans_cli.app,
            [
                "batch",
                str(in_dir),
                "--dry-run",
                "--subs-quality",
                "high",
                "--subs-upgrade",
                "--no-rag",
                "--no-cascade",
                "--auto-offset",
                "--format",
                "ass",
                "--wrap",
                "60",
                "--gloss-skill-names",
                "--require-approved-cast",
                "--review-file",
                "--no-smart-wrap",
                "--no-auto-align-en",
                "--explicit-cache",
                "--en-offset-ms",
                "120",
                "--sub-offset-ms",
                "-250",
                "--limit",
                "5",
                "--cache-db",
                str(cache_db),
                "-o",
                str(out_dir),
            ],
        )

    assert res.exit_code == 0, res.output
    assert len(captured_init) == 1
    init_kw = captured_init[0]
    assert init_kw["subs_quality"] == "high"
    assert init_kw["subs_upgrade"] is True
    assert init_kw["enable_rag"] is False
    assert init_kw["enable_cascade"] is False
    assert init_kw["output_format"] == "ass"
    assert init_kw["wrap_chars"] == 60
    assert init_kw["gloss_skill_names"] is True
    assert init_kw["require_approved_cast"] is True
    assert init_kw["review_file"] is True
    assert init_kw["smart_wrap"] is False
    assert init_kw["auto_align_whisper_to_en"] is False
    assert init_kw["explicit_caching"] is True
    assert init_kw["limit_cues"] == 5
    assert init_kw["cache_db_path"] == cache_db
    assert init_kw["output_dir"] == out_dir

    assert len(captured_run) == 1
    run_kw = captured_run[0]
    assert run_kw["ja_path"] == ja_srt
    assert run_kw["en_offset_ms"] == 120
    assert run_kw["sub_offset_ms"] == -250
    assert run_kw["subs_upgrade"] is True
    assert run_kw["auto_offset"] is True


def test_cli_batch_matches_en_subtitles_across_naming_variants(tmp_path: Path):
    """Verify _match_en_subtitle and _collect_batch_input_files handle naming variants and shared dirs."""
    in_dir = tmp_path / "subs_in"
    en_dir = tmp_path / "subs_en"
    in_dir.mkdir()
    en_dir.mkdir()

    ja1 = in_dir / "ep01.ja.srt"
    ja2 = in_dir / "ep02.ja[cc].srt"
    ja3 = in_dir / "sample_30s_jpn.ass"
    ja4 = in_dir / "ep04.srt"
    for p in (ja1, ja2, ja3, ja4):
        p.write_text("", encoding="utf-8")

    en1 = en_dir / "ep01.en.srt"
    en2 = en_dir / "ep02.en.ass"
    en3 = en_dir / "sample_30s_eng.ass"
    en4 = en_dir / "ep04.srt"
    for p in (en1, en2, en3, en4):
        p.write_text("", encoding="utf-8")

    assert subtrans_cli._match_en_subtitle(ja1, en_dir) == en1
    assert subtrans_cli._match_en_subtitle(ja2, en_dir) == en2
    assert subtrans_cli._match_en_subtitle(ja3, en_dir) == en3
    assert subtrans_cli._match_en_subtitle(ja4, en_dir) == en4

    # Shared input_dir / en_dir skips outputs and paired EN tracks without self-matching
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    s_ja = shared_dir / "ep01.ja.srt"
    s_en = shared_dir / "ep01.en.srt"
    s_vi = shared_dir / "ep01.vi.srt"
    s_rev = shared_dir / "ep01.review.srt"
    s_untagged = shared_dir / "ep02.srt"
    s_en_only = shared_dir / "ep03.en.srt"
    for p in (s_ja, s_en, s_vi, s_rev, s_untagged, s_en_only):
        p.write_text("", encoding="utf-8")

    assert subtrans_cli._match_en_subtitle(s_untagged, shared_dir) is None
    assert subtrans_cli._match_en_subtitle(s_ja, shared_dir) == s_en
    collected = subtrans_cli._collect_batch_input_files(shared_dir, shared_dir)
    assert collected == [s_ja, s_untagged, s_en_only]


def test_cli_batch_local_only_still_exits_1(tmp_path: Path):
    """Verify batch --local still exits 1 with Cloud Gemini guidance."""
    runner = CliRunner()
    res = runner.invoke(subtrans_cli.app, ["batch", str(tmp_path), "--local"])
    assert res.exit_code == 1
    assert "Cloud Gemini" in _plain(res.output)


def test_cli_batch_invalid_latest_model_rejected_without_allow_latest_alias(tmp_path: Path):
    """Verify batch rejects mutable gemini-flash-latest unless --allow-latest-alias is passed."""
    runner = CliRunner()
    in_dir = tmp_path / "latest_in"
    in_dir.mkdir()
    (in_dir / "ep01.ja.srt").write_text("1\n00:00:01,000 --> 00:00:03,000\nテスト\n", encoding="utf-8")
    out_dir = tmp_path / "latest_out"
    cache_db = tmp_path / "latest.db"

    res_bad = runner.invoke(
        subtrans_cli.app,
        [
            "batch",
            str(in_dir),
            "--dry-run",
            "--no-rag",
            "--model",
            "gemini-flash-latest",
            "--cache-db",
            str(cache_db),
            "-o",
            str(out_dir),
        ],
    )
    assert res_bad.exit_code == 1

    res_ok = runner.invoke(
        subtrans_cli.app,
        [
            "batch",
            str(in_dir),
            "--dry-run",
            "--no-rag",
            "--model",
            "gemini-flash-latest",
            "--allow-latest-alias",
            "--cache-db",
            str(cache_db),
            "-o",
            str(out_dir),
        ],
    )
    assert res_ok.exit_code == 0, res_ok.output


def test_cli_batch_whisper_ja_en_pairing_and_shared_dir_dedup(tmp_path: Path):
    """Verify .whisper_ja.srt and _whisper.ja.srt pair with .en.srt and suppress EN in shared dirs."""
    in_dir = tmp_path / "whisper_in"
    en_dir = tmp_path / "whisper_en"
    in_dir.mkdir()
    en_dir.mkdir()

    w1 = in_dir / "ep01.whisper_ja.srt"
    w2 = in_dir / "ep02_whisper.ja.srt"
    e1 = en_dir / "ep01.en.srt"
    e2 = en_dir / "ep02.en.srt"
    for p in (w1, w2, e1, e2):
        p.write_text("", encoding="utf-8")

    assert subtrans_cli._match_en_subtitle(w1, en_dir) == e1
    assert subtrans_cli._match_en_subtitle(w2, en_dir) == e2

    shared = tmp_path / "whisper_shared"
    shared.mkdir()
    sw1 = shared / "ep01.whisper_ja.srt"
    se1 = shared / "ep01.en.srt"
    sw2 = shared / "ep02_whisper.ja.srt"
    se2 = shared / "ep02.en.srt"
    se_only = shared / "ep03.en.srt"
    for p in (sw1, se1, sw2, se2, se_only):
        p.write_text("", encoding="utf-8")

    assert subtrans_cli._match_en_subtitle(sw1, shared) == se1
    assert subtrans_cli._match_en_subtitle(sw2, shared) == se2
    collected = subtrans_cli._collect_batch_input_files(shared, shared)
    assert collected == [sw1, sw2, se_only]


def test_cli_batch_partial_failure_continues_and_exits_1(tmp_path: Path):
    """Verify batch continues processing remaining files on per-file error and exits 1."""
    runner = CliRunner()
    in_dir = tmp_path / "partial_in"
    in_dir.mkdir()
    ep1 = in_dir / "ep01.ja.srt"
    ep2 = in_dir / "ep02.ja.srt"
    ep1.write_text("1\n00:00:01,000 --> 00:00:02,000\nあ\n", encoding="utf-8")
    ep2.write_text("1\n00:00:01,000 --> 00:00:02,000\nい\n", encoding="utf-8")

    processed = []

    class _PartialFailPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, ja_path, **_kwargs):
            processed.append(Path(ja_path).name)
            if Path(ja_path).name == "ep01.ja.srt":
                raise RuntimeError("simulated episode error")
            return {}

    with patch("subtrans.cli.TranslationPipeline", side_effect=_PartialFailPipeline):
        res = runner.invoke(subtrans_cli.app, ["batch", str(in_dir), "--dry-run", "-o", str(tmp_path / "out")])

    assert res.exit_code == 1
    assert processed == ["ep01.ja.srt", "ep02.ja.srt"]
    assert "ep01.ja.srt" in _plain(res.output)
