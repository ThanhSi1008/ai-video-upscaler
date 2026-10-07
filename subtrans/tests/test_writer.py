# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
import pysubs2
from subtrans.models import Cue
from subtrans.writer import wrap_subtitle_line, write_subtitles


def test_wrap_subtitle_line_short():
    text = "Xin chào các bạn!"
    assert wrap_subtitle_line(text, max_chars=42) == text


def test_wrap_subtitle_line_punctuation_split():
    # Long text with comma near the center
    text = "Nếu như ngày hôm đó cậu không đến, tôi nhất định sẽ không thể nào vượt qua được nỗi đau này."
    wrapped = wrap_subtitle_line(text, max_chars=42, delimiter="\\N")
    assert "\\N" in wrapped
    parts = wrapped.split("\\N")
    assert len(parts) == 2
    # Should break after comma
    assert parts[0].endswith(",")


def test_write_subtitles_preserves_styles(tmp_path):
    ssa = pysubs2.SSAFile()
    event = pysubs2.SSAEvent(start=1000, end=3000, text="元の日本語", style="Main")
    ssa.events.append(event)

    cues = [
        Cue(
            id=0,
            start=1000,
            end=3000,
            style="Main",
            ja="元の日本語",
            vi="Tiếng Việt dịch chuẩn",
            raw_text="元の日本語",
        )
    ]

    out_ass = tmp_path / "test.ass"
    written_path = write_subtitles(cues, ssa, out_ass)

    assert written_path.exists()
    loaded = pysubs2.load(str(written_path))
    assert len(loaded) == 1
    assert loaded[0].text == "Tiếng Việt dịch chuẩn"
    assert loaded[0].style == "Main"


def test_optimize_cues_dual_dialogue_simultaneous():
    """Kiểm tra 2 người nói cùng lúc/gần cùng lúc -> gộp thành 1 cue 2 dòng song song."""
    from subtrans.writer import optimize_cues_for_playback
    cues = [
        Cue(id=0, start=10000, end=13000, ja="あ", vi="Không được đâu!"),
        Cue(id=1, start=10200, end=13200, ja="い", vi="Đi nhanh lên!"),
    ]
    optimized = optimize_cues_for_playback(cues)
    assert len(optimized) == 1
    opt = optimized[0]
    assert opt.start == 10000
    assert opt.end >= 13200
    assert "- Không được đâu!" in opt.vi
    assert "- Đi nhanh lên!" in opt.vi
    assert "\n" in opt.vi


def test_optimize_cues_rapid_fragments_merging():
    """Kiểm tra 1 câu bị tách làm 2 mảnh nói quá nhanh -> gộp thành 1 câu liền mạch."""
    from subtrans.writer import optimize_cues_for_playback
    cues = [
        Cue(id=0, start=5000, end=5800, ja="あ", vi="Nếu có ba ngày..."),
        Cue(id=1, start=5900, end=7200, ja="い", vi="ta thừa sức điều tra ra chân tướng!"),
    ]
    optimized = optimize_cues_for_playback(cues)
    assert len(optimized) == 1
    opt = optimized[0]
    assert opt.start == 5000
    assert opt.end >= 7200
    assert "Nếu có ba ngày... ta thừa sức điều tra ra chân tướng!" in opt.vi


def test_optimize_cues_minimum_duration():
    """Kiểm tra câu ngắn đơn lẻ được kéo dài tối thiểu >= 1200ms nếu không đụng câu sau."""
    from subtrans.writer import optimize_cues_for_playback
    cues = [
        Cue(id=0, start=1000, end=1500, ja="はい", vi="Vâng ạ."),
        Cue(id=1, start=4000, end=6000, ja="あ", vi="Chúng ta bắt đầu thôi."),
    ]
    optimized = optimize_cues_for_playback(cues)
    assert len(optimized) == 2
    assert optimized[0].end >= 2200  # Kéo dài ít nhất 1.2s


def test_write_srt_subtitles_preserves_dual_dialogue(tmp_path):
    """Kiểm tra file SRT xuất ra có cả 2 dòng đối thoại khi 2 người nói cùng lúc."""
    from subtrans.writer import write_srt_subtitles
    cues = [
        Cue(id=0, start=10000, end=13500, ja="あ", vi="Nàng có sao không?"),
        Cue(id=1, start=10100, end=13000, ja="い", vi="Thần không sao ạ."),
    ]
    out_srt = tmp_path / "dual.srt"
    write_srt_subtitles(cues, out_srt, optimize_playback=True)

    content = out_srt.read_text(encoding="utf-8")
    assert "- Nàng có sao không?" in content
    assert "- Thần không sao ạ." in content
    blocks = [b.strip() for b in content.strip().split("\n\n") if b.strip()]
    assert len(blocks) == 1
    assert "00:00:10,000 --> 00:00:13,500" in blocks[0]

