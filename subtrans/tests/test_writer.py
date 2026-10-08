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


def test_optimize_cues_anti_duplicate_roll_up():
    """Kiểm tra hai cue CC lặp nội dung hoặc một câu bao hàm câu kia -> gộp thành 1 câu, không tạo dual."""
    from subtrans.writer import optimize_cues_for_playback
    cues = [
        Cue(id=0, start=1000, end=3000, ja="あ", vi="Leelee, chúng ta hãy đi tìm nữ quan đó thôi."),
        Cue(id=1, start=2000, end=5000, ja="い", vi="Leelee, chúng ta hãy đi tìm nữ quan đó thôi."),
    ]
    optimized = optimize_cues_for_playback(cues)
    assert len(optimized) == 1
    assert optimized[0].vi == "Leelee, chúng ta hãy đi tìm nữ quan đó thôi."
    assert "\n" not in optimized[0].vi


def test_optimize_cues_strict_zero_overlap():
    """Kiểm tra mọi block độc lập không bao giờ chồng lấn thời gian (c.end <= next.start - 40ms)."""
    from subtrans.writer import optimize_cues_for_playback
    cues = [
        Cue(id=0, start=1000, end=4000, ja="あ", vi="Câu thoại đầu tiên kéo dài."),
        Cue(id=1, start=3500, end=7000, ja="い", vi="Câu thoại thứ hai bắt đầu sớm."),
    ]
    optimized = optimize_cues_for_playback(cues)
    for k in range(len(optimized) - 1):
        assert optimized[k].end <= optimized[k + 1].start - 40


def test_optimize_cues_banquet_scene_no_stacking():
    """Kiểm tra phân đoạn phức tạp phòng tiệc (6 cues) không bao giờ bị đè 3-4 dòng trên IINA."""
    from subtrans.writer import optimize_cues_for_playback
    cues = [
        Cue(id=188, start=717049, end=721178, ja="あらまあ 随分 見事な品ですこと", vi="Ôi chao, đây quả là một món đồ lộng lẫy làm sao."),
        Cue(id=189, start=720761, end=723889, ja="莉莉 例の女官を探しましょう", vi="Leelee, chúng ta hãy đi tìm nữ quan đó thôi."),
        Cue(id=190, start=721262, end=724515, ja="身ひとつで 謹慎されていたはずでは？", vi="Chẳng phải ả ta đáng lẽ phải đang chịu cấm túc và không mang theo gì sao?"),
        Cue(id=191, start=723973, end=724515, ja="うん", vi="Ừm."),
        Cue(id=192, start=724598, end=728018, ja="それが どうして こんなものを お持ちなのやら", vi="Vậy mà cớ sao ả lại có thể sở hữu một thứ thế này cơ chứ?"),
        Cue(id=193, start=725599, end=728018, ja="雅容様は どこにいるんだろう", vi="Gayou-sama đang ở đâu nhỉ?"),
    ]
    optimized = optimize_cues_for_playback(cues)

    # 1. Tuyệt đối không chồng lấn thời gian
    for k in range(len(optimized) - 1):
        assert optimized[k].end <= optimized[k + 1].start - 40, f"Overlap at {k}: {optimized[k].end} > {optimized[k+1].start - 40}"

    # 2. Không có block nào chứa > 2 dòng
    for c in optimized:
        lines = [l for l in c.vi.split("\n") if l.strip()]
        assert len(lines) <= 2, f"Too many lines in block: {c.vi}"


def test_optimize_cues_echoed_keywords_preserved():
    """Kiểm tra khi nhân vật nhại lại/nhắc lại từ khóa của câu trước (echoing dialogue) không bị nuốt mất."""
    from subtrans.writer import optimize_cues_for_playback
    cues = [
        Cue(id=82, ja="a", start=312228, end=316149, vi="lại có phần hơi bất thường thế này khiến tớ khá bất ngờ đấy."),
        Cue(id=83, ja="b", start=316232, end=318234, vi="Hơi bất thường..."),
        Cue(id=84, ja="c", start=318317, end=319444, vi="(Tousetsu) Hừm... -(Leelee) Hả?"),
    ]
    optimized = optimize_cues_for_playback(cues)
    assert len(optimized) == 3
    assert "lại có phần hơi bất thường" in optimized[0].vi
    assert optimized[1].vi == "Hơi bất thường..."
    assert "(Tousetsu) Hừm..." in optimized[2].vi


def test_optimize_cues_duplicate_variant_dialogue():
    """Kiểm tra hai cue trùng lặp/dị bản câu thoại gần nhau không bị biến thành Dual Dialogue 2 dòng."""
    from subtrans.writer import optimize_cues_for_playback
    cues = [
        Cue(id=303, ja="ふはははは!", start=893685, end=898148, vi="Đúng là pháp sư chiêu hồn vĩ đại thời đại chúng ta! Ngài đỉnh quá!"),
        Cue(id=304, ja="まさに偉大な死霊術師 素敵ですー!", start=894227, end=898148, vi="Đúng là pháp sư chiêu hồn vĩ đại, tuyệt vời quá!"),
    ]
    optimized = optimize_cues_for_playback(cues)
    assert len(optimized) == 1
    # Không được có dấu gạch ngang đầu dòng hay ngắt 2 dòng
    assert "\n" not in optimized[0].vi
    assert "Đúng là pháp sư chiêu hồn vĩ đại" in optimized[0].vi




