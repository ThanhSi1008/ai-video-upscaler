# -*- coding: utf-8 -*-
"""
subtrans/ja_morph_tagger.py
Bộ quét đặc trưng hình thái học và sắc thái ngữ dụng tiếng Nhật (Zero-Latency Regex Tagger):
- Trích xuất tự động các dấu chỉ đại từ xưng hô (Pronoun Markers: ore, boku, kisama, temee, anta, omae...)
- Trích xuất trợ từ cuối câu (Sentence-Final Particles: desuwa, daze, kashira...)
- Trích xuất kính ngữ & quan hệ thân tộc (senpai, sensei, sama, onii...)
- Đính kèm trực tiếp vào payload dịch thuật của LLM nhằm giải quyết triệt để hiện tượng lược bỏ đại từ (Zero-Pronouns)
  và triệt tiêu hiện tượng mỏ neo tiếng Anh (English-Anchoring Bias) mà không phát sinh thêm chi phí API hay độ trễ.
"""

import re
from typing import List, Tuple, Optional

# Regex nhận diện Furigana nội dòng đứng ngay sau Kanji: VD 金(きん)淑(しゅく)妃(ひ), 魔剣(まけん)
_FURIGANA_INLINE_REGEX = re.compile(
    r"(?<=[\u4E00-\u9FFF\u3400-\u4DBF\u3005\u30F5\u30F6])"
    r"[（\(][\u3040-\u309F\u30A0-\u30FFー・\s]{1,24}[）\)]"
)
_CJK_DETECTION_REGEX = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]")


def strip_ja_furigana(text: str) -> str:
    """
    Loại bỏ chú thích cách đọc Furigana/Ruby trong ngoặc đơn đứng ngay sau Kanji
    (ví dụ: '金(きん)淑(しゅく)妃(ひ)' -> '金淑妃', '珊瑚(さんご)' -> '珊瑚')
    mà KHÔNG làm ảnh hưởng tới chú thích âm thanh đầu câu như '（ため息）'.
    """
    if not text:
        return ""
    return _FURIGANA_INLINE_REGEX.sub("", text)


def has_cjk_characters(text: str) -> bool:
    """Kiểm tra nhanh chuỗi có chứa ký tự tiếng Nhật/CJK hay không."""
    if not text:
        return False
    return bool(_CJK_DETECTION_REGEX.search(text))


# Danh mục mẫu nhận diện đại từ xưng hô, ngữ khí và kính ngữ tiếng Nhật
JA_REGISTER_PATTERNS: List[Tuple[str, str]] = [
    # 2nd Person Pronouns (Ngôi 2 - Sắc thái đối đầu / suồng sã / số nhiều)
    (r"貴様ら|てめぇら|てめえら", "2P_PLURAL:kisamara/temeera (căm phẫn số nhiều -> các ngươi/chúng mày/lũ khốn)"),
    (r"お前ら|おまえら|あんたら", "2P_PLURAL:omaera (ngang hàng/bề trên số nhiều -> các ngươi/tụi bây/các cậu)"),
    (r"貴様|きさま", "2P:kisama (căm phẫn/khinh miệt -> ngươi/mày)"),
    (r"てめぇ|てめえ|手前", "2P:temee (thô bạo/đối đầu -> mày/tên khốn)"),
    (r"あんた|アンタ", "2P:anta (suồng sã/trách móc/tsundere)"),
    (r"お前|おまえ", "2P:omae (ngang hàng/bề trên/thân mật)"),
    (r"君|きみ", "2P:kimi (thân mật/trên bảo dưới -> cậu/em)"),

    # 1st Person Pronouns (Ngôi 1 - Sắc thái giới tính / vai vế)
    (r"わたくし|ワタクシ", "1P:watakushi (trang trọng/quý tộc/hầu cận -> tôi/thiếp/thần)"),
    (r"俺|オレ", "1P:ore (nam tính/mạnh mẽ -> tôi/ta/tao/anh)"),
    (r"僕|ボク", "1P:boku (nam lịch sự/trẻ -> tôi/tớ/em)"),
    (r"私|ワタシ", "1P:watashi (chuẩn mực/lịch sự -> tôi/em/mình)"),
    (r"あたし|アタシ", "1P:atashi (nữ thân mật -> em/tớ/mình)"),
    (r"あたい|アタイ", "1P:atai (nữ cá tính/bụi đời -> tôi/em/bà đây)"),
    (r"わし|ワシ", "1P:washi (lão niên/uy quyền -> lão/ta/ông)"),
    (r"わらわ|妾", "1P:warawa (cổ trang quý tộc nữ -> thiếp/ta)"),
    (r"拙者|せっしゃ", "1P:sessha (võ sĩ/samurai -> tại hạ)"),
    (r"それがし|某", "1P:soregashi (võ sĩ cổ trang -> tại hạ)"),
    (r"(?:^|[、。\s「『（])(?:余|予)(?:は|が|の|を|に|も)", "1P:yo (đế vương/quý tộc cổ trang -> ta/trẫm)"),
    (r"我|吾|われ", "1P:ware (cổ trang/ngạo mạn -> ta)"),
    (r"自分", "1P/REFL:jibun (bản thân/tự mình/kỷ luật)"),

    # 2nd Person Archaic & Formal (Ngôi 2 - Cổ trang / Kính trọng)
    (r"そなた|其方", "2P:sonata (cổ trang quý tộc/bề trên -> nàng/ngươi/khanh)"),
    (r"お主|おぬし", "2P:onushi (bề trên gọi kẻ dưới/cổ trang -> ngươi)"),
    (r"貴殿|きでん", "2P:kiden (trang trọng kính trọng -> các hạ/ngài)"),

    # Honorifics & Kinship (Kính ngữ & Thân tộc)
    (r"陛下|へいか", "HON:heika (bệ hạ)"),
    (r"閣下|かっか", "HON:kakka (các hạ/ngài)"),
    (r"お嬢様|おじょうさま", "HON:ojousama (tiểu thư)"),
    (r"先輩|せんぱい", "HON:senpai (tiền bối/anh/chị)"),
    (r"先生|せんせい", "HON:sensei (thầy/cô/tiên sinh)"),
    (r"様|さま", "HON:sama (ngài/đại nhân/tiểu thư)"),
    (r"殿下|でんか", "HON:denka (điện hạ)"),
    (r"兄上|姉上|父上|母上", "KIN:noble_kinship (huynh trưởng/tỷ tỷ/phụ thân/mẫu thân)"),
    (r"兄さん|にいさん|お兄ちゃん|兄貴", "KIN:onii (anh trai/huynh)"),
    (r"姉さん|ねえさん|お姉ちゃん|姉貴", "KIN:onee (chị gái/tỷ)"),

    # Verb Honorifics & Humble Forms (Giải quyết Zero-Pronouns khi tỉnh lược chủ ngữ)
    (r"で(?:は)?ございます|で(?:は)?ござ[るらりれ]|ございます", "REGISTER:keigo_formal (trang trọng/hầu cận/lịch thiệp)"),
    (r"であります", "REGISTER:military (quân đội/trang nghiêm)"),
    (r"(?:いらっしゃる|おいでになる)", "REGISTER:sonkeigo (tôn kính người nghe/bề trên)"),
    (r"(?:申します|申し上げます|参ります)", "REGISTER:kenjougo (khiêm nhường xưng hô bản thân)"),

    # Verb Imperatives, Commands & Requests (hỗ trợ cả cuối câu lẫn mệnh đề giữa câu đa mệnh đề)
    (r"やが[るれらり]|やがっ", "VULGAR:yagare (khinh bỉ/rủa xả -> tên khốn/chết tiệt)"),
    (r"くたばれ|失せろ", "HOSTILE:chết tiệt/biến đi"),
    (r"(?:なさい|たまえ|おくれ)(?:[。！？…\s」』＞\)]|$)", "CMD:senior_to_junior (bề trên lệnh bề dưới -> hãy...)"),
    (r"(?:てくれ|てよ|ちょうだい)(?:[。！？…\s」』＞\)]|$)", "REQ:peer_casual (nhờ vả thân mật/ngang hàng)"),
    (r"(?:てください|ていただけますか|ください(?:[。！？…\s」』＞\)]|$))", "REQ:polite (yêu cầu lịch sự)"),

    # Sentence-Final Particles & Dialect Markers (hỗ trợ câu đa mệnh đề ngăn bởi 。！？)
    (r"(?:だわ|かしら|ですわ|ますわ|ですの|わよ)(?:[。！？…\s」』＞\)]|$)", "SFP:nữ_tính_hoặc_tiểu_thư"),
    (r"(?:[ぜぞ]|だな|かよ)(?:[。！？…\s」』＞\)]|$)", "SFP:nam_tính_mạnh"),
    (r"(?:っす|ッス)(?:[。！？…\s」』＞\)]|$)", "SFP:hậu_bối_kính_cẩn"),
    (r"(?:じゃ|じゃな|じゃぞ|じゃろう|かのう|わい)(?:[。！？…\s」』＞\)]|$)", "SFP:lão_niên_cổ_trang"),
    (r"(?:なんでやねん|ちゃうねん|やねん|やな|やで|やんか|へんわ|あかん|せやな|せやかて|ほんま|さかい)(?:[。！？…\s」』＞\)]|$)", "DIALECT:kansai_ben (phương ngữ Kansai thân mật)"),
]

_COMPILED_PATTERNS = [(re.compile(pat), desc) for pat, desc in JA_REGISTER_PATTERNS]


def extract_ja_linguistic_hints(ja_text: str) -> Optional[str]:
    """
    Trích xuất chuỗi gợi ý hình thái học từ câu tiếng Nhật (quét cả bản gốc có Furigana/Ruby
    lẫn bản đã gộp Furigana như '貴(き)様(さま)' -> '貴様'):
    Ví dụ: 'てめぇ、何者だ！' -> '2P:temee (thô bạo/đối đầu -> mày/tên khốn)'
    Trả về None nếu không phát hiện tín hiệu đặc biệt (câu trung tính).
    """
    if not ja_text or not has_cjk_characters(ja_text):
        return None

    stripped = strip_ja_furigana(ja_text)
    hints = []
    for pattern, desc in _COMPILED_PATTERNS:
        if pattern.search(ja_text) or (stripped != ja_text and pattern.search(stripped)):
            hints.append(desc)

    return " | ".join(hints) if hints else None
