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

# Danh mục mẫu nhận diện đại từ xưng hô, ngữ khí và kính ngữ tiếng Nhật
JA_REGISTER_PATTERNS: List[Tuple[str, str]] = [
    # 2nd Person Pronouns (Ngôi 2 - Sắc thái đối đầu / suồng sã)
    (r"貴様|きさま", "2P:kisama (căm phẫn/khinh miệt -> ngươi/mày)"),
    (r"てめぇ|てめえ|手前", "2P:temee (thô bạo/đối đầu -> mày/tên khốn)"),
    (r"あんた|アンタ", "2P:anta (suồng sã/trách móc/tsundere)"),
    (r"お前|おまえ", "2P:omae (ngang hàng/bề trên/thân mật)"),
    (r"君|きみ", "2P:kimi (thân mật/trên bảo dưới -> cậu/em)"),

    # 1st Person Pronouns (Ngôi 1 - Sắc thái giới tính / vai vế)
    (r"俺|オレ", "1P:ore (nam tính/mạnh mẽ -> tôi/ta/tao/anh)"),
    (r"僕|ボク", "1P:boku (nam lịch sự/trẻ -> tôi/tớ/em)"),
    (r"私|ワタシ", "1P:watashi (chuẩn mực/lịch sự -> tôi/em/mình)"),
    (r"あたし|アタシ", "1P:atashi (nữ thân mật -> em/tớ/mình)"),
    (r"わし|ワシ", "1P:washi (lão niên/uy quyền -> lão/ta/ông)"),
    (r"わらわ|妾", "1P:warawa (cổ trang quý tộc nữ -> thiếp/ta)"),
    (r"自分", "1P/REFL:jibun (bản thân/tự mình/kỷ luật)"),

    # Honorifics & Kinship (Kính ngữ & Thân tộc)
    (r"先輩|せんぱい", "HON:senpai (tiền bối/anh/chị)"),
    (r"先生|せんせい", "HON:sensei (thầy/cô/tiên sinh)"),
    (r"様|さま", "HON:sama (ngài/đại nhân/tiểu thư)"),
    (r"兄さん|にいさん|お兄ちゃん|兄貴", "KIN:onii (anh trai/huynh)"),
    (r"姉さん|ねえさん|お姉ちゃん|姉貴", "KIN:onee (chị gái/tỷ)"),

    # Sentence-Final Particles (Trợ từ cuối câu - Ngữ khí)
    (r"(?:だわ|かしら|ですわ|わよ)[。！？…]*$", "SFP:nữ_tính_hoặc_tiểu_thư"),
    (r"(?:[ぜぞ]|だな|かよ)[。！？…]*$", "SFP:nam_tính_mạnh"),
    (r"(?:っす|ッス)[。！？…]*$", "SFP:hậu_bối_kính_cẩn"),
]

_COMPILED_PATTERNS = [(re.compile(pat), desc) for pat, desc in JA_REGISTER_PATTERNS]


def extract_ja_linguistic_hints(ja_text: str) -> Optional[str]:
    """
    Trích xuất chuỗi gợi ý hình thái học từ câu tiếng Nhật:
    Ví dụ: 'てめぇ、何者だ！' -> '2P:temee (thô bạo/đối đầu -> mày/tên khốn)'
    Trả về None nếu không phát hiện tín hiệu đặc biệt (câu trung tính).
    """
    if not ja_text:
        return None

    hints = []
    for pattern, desc in _COMPILED_PATTERNS:
        if pattern.search(ja_text):
            hints.append(desc)

    return " | ".join(hints) if hints else None
