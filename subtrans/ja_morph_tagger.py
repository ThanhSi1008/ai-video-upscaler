"""MeCab/UniDic morphological tagger with boundary-safe regex fallback for Japanese hints."""

from __future__ import annotations

import re
import threading
from typing import Any, List, Optional, Tuple

# Regex nhận diện Furigana nội dòng đứng ngay sau Kanji: VD 金(きん)淑(しゅく)妃(ひ), 魔剣(まけん)
_FURIGANA_INLINE_REGEX = re.compile(
    r"(?<=[\u4E00-\u9FFF\u3400-\u4DBF\u3005\u30F5\u30F6])"
    r"[（\(][\u3040-\u309F\u30A0-\u30FFー・\s]{1,24}[）\)]"
)
_CJK_DETECTION_REGEX = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]")

_TAGGER_LOCK = threading.Lock()
_MECAB_TAGGER: Any = None
_MECAB_UNAVAILABLE: bool = False


def _get_mecab_tagger() -> Any:
    """Khởi tạo lười (lazy singleton) bộ tách từ MeCab C++ qua fugashi + unidic-lite."""
    global _MECAB_TAGGER, _MECAB_UNAVAILABLE
    if _MECAB_TAGGER is not None:
        return _MECAB_TAGGER
    if _MECAB_UNAVAILABLE:
        return None
    with _TAGGER_LOCK:
        if _MECAB_TAGGER is not None:
            return _MECAB_TAGGER
        if _MECAB_UNAVAILABLE:
            return None
        try:
            import fugashi  # type: ignore

            _MECAB_TAGGER = fugashi.Tagger()
            return _MECAB_TAGGER
        except Exception:
            _MECAB_UNAVAILABLE = True
            return None


def is_morphological_backend_active() -> bool:
    """Trả về True nếu bộ tách từ hình thái học C++ (`fugashi` + `unidic-lite`) sẵn sàng."""
    return _get_mecab_tagger() is not None


def tokenize_ja_morphemes(text: str) -> List[Tuple[str, str, str]]:
    """
    Tách câu tiếng Nhật thành danh sách bộ ba `(surface, pos1, pos2)` bằng MeCab/UniDic.
    Trả về danh sách rỗng nếu chuỗi rỗng hoặc không có `fugashi`.
    """
    if not text:
        return []
    tagger = _get_mecab_tagger()
    if tagger is None:
        return []
    try:
        out: List[Tuple[str, str, str]] = []
        for w in tagger(text):
            feat = getattr(w, "feature", None)
            pos1 = getattr(feat, "pos1", "*") if feat is not None else "*"
            pos2 = getattr(feat, "pos2", "*") if feat is not None else "*"
            out.append((str(w.surface), str(pos1 or "*"), str(pos2 or "*")))
        return out
    except Exception:
        return []


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


# Rào chắn biên Kanji cho nhánh fallback Regex: không đứng ngay sau Kanji khác,
# và không đứng trước Kanji khác (ngoại trừ hậu tố số nhiều ら/達/等 hoặc 我が/我が軍).
_KANJI_CLS = r"\u3400-\u4DBF\u4E00-\u9FFF"
_NO_PREV_KANJI = rf"(?<![{_KANJI_CLS}])"
_NO_NEXT_KANJI_UNLESS_PLURAL = rf"(?![{_KANJI_CLS}])|(?=[達等])"

# Các mẫu cụm từ / trợ từ / kính ngữ / phương ngữ không bị nhập nhằng từ ghép Kanji đơn
JA_REGISTER_PATTERNS: List[Tuple[str, str]] = [
    # 2nd Person Plural & Kana Pronouns
    (r"貴様ら|てめぇら|てめえら", "2P_PLURAL:kisamara/temeera (căm phẫn số nhiều -> các ngươi/chúng mày/lũ khốn)"),
    (r"お前ら|おまえら|あんたら", "2P_PLURAL:omaera (ngang hàng/bề trên số nhiều -> các ngươi/tụi bây/các cậu)"),
    (r"貴様|きさま", "2P:kisama (căm phẫn/khinh miệt -> ngươi/mày)"),
    (r"てめぇ|てめえ", "2P:temee (thô bạo/đối đầu -> mày/tên khốn)"),
    (r"あんた|アンタ", "2P:anta (suồng sã/trách móc/tsundere)"),
    (r"お前|おまえ", "2P:omae (ngang hàng/bề trên/thân mật)"),
    (r"(?<![ァ-ヶー])きみ(?:ら|たち|達)?(?=[はがのをにもへとや、。！？…\s」』＞\)]|$)", "2P:kimi (thân mật/trên bảo dưới -> cậu/em)"),

    # 1st Person Kana & Multi-Kanji Pronouns
    (r"わたくし|ワタクシ", "1P:watakushi (trang trọng/quý tộc/hầu cận -> tôi/thiếp/thần)"),
    (r"オレ(?:ら|たち|達)?", "1P:ore (nam tính/mạnh mẽ -> tôi/ta/tao/anh)"),
    (r"ボク(?:ら|たち|達)?", "1P:boku (nam lịch sự/trẻ -> tôi/tớ/em)"),
    (r"ワタシ(?:ら|たち|達)?", "1P:watashi (chuẩn mực/lịch sự -> tôi/em/mình)"),
    (r"あたし|アタシ", "1P:atashi (nữ thân mật -> em/tớ/mình)"),
    (r"あたい|アタイ", "1P:atai (nữ cá tính/bụi đời -> tôi/em/bà đây)"),
    (r"わし|ワシ", "1P:washi (lão niên/uy quyền -> lão/ta/ông)"),
    (r"わらわ", "1P:warawa (cổ trang quý tộc nữ -> thiếp/ta)"),
    (r"拙者|せっしゃ", "1P:sessha (võ sĩ/samurai -> tại hạ)"),
    (r"それがし", "1P:soregashi (võ sĩ cổ trang -> tại hạ)"),
    (r"われ(?:ら|われ)?(?=[はがのをにもへとや、。！？…\s」』＞\)]|$)", "1P:ware (cổ trang/ngạo mạn -> ta)"),
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
    (r"(?<!お疲れ)(?<!ご苦労)(?<!お陰)(?<!おかげ)様|さま", "HON:sama (ngài/đại nhân/tiểu thư)"),
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

# Các mẫu đại từ Kanji đơn dễ trùng từ ghép — dùng khi môi trường KHÔNG có fugashi (có rào chắn biên Kanji)
_FALLBACK_KANJI_PRONOUN_PATTERNS: List[Tuple[str, str]] = [
    (rf"{_NO_PREV_KANJI}手前(?:ら|達)?(?!(?:味噌|どもの?[{_KANJI_CLS}]|[{_KANJI_CLS}]))(?=[ら達たち、。！？…\s」』＞\)]|(?:は|が|の|を|に|も|と))", "2P:temee (thô bạo/đối đầu -> mày/tên khốn)"),
    (rf"{_NO_PREV_KANJI}君(?:{_NO_NEXT_KANJI_UNLESS_PLURAL})", "2P:kimi (thân mật/trên bảo dưới -> cậu/em)"),
    (rf"{_NO_PREV_KANJI}俺(?:{_NO_NEXT_KANJI_UNLESS_PLURAL})", "1P:ore (nam tính/mạnh mẽ -> tôi/ta/tao/anh)"),
    (rf"{_NO_PREV_KANJI}僕(?:{_NO_NEXT_KANJI_UNLESS_PLURAL})", "1P:boku (nam lịch sự/trẻ -> tôi/tớ/em)"),
    (rf"{_NO_PREV_KANJI}私(?:{_NO_NEXT_KANJI_UNLESS_PLURAL})", "1P:watashi (chuẩn mực/lịch sự -> tôi/em/mình)"),
    (rf"{_NO_PREV_KANJI}妾(?:{_NO_NEXT_KANJI_UNLESS_PLURAL})", "1P:warawa (cổ trang quý tộc nữ -> thiếp/ta)"),
    (rf"{_NO_PREV_KANJI}某(?![{_KANJI_CLS}])(?=は|が|の|を|に|も|、|。)", "1P:soregashi (võ sĩ cổ trang -> tại hạ)"),
    (r"(?:^|[、。\s「『（])(?:余|予)(?:は|が|の|を|に|も)", "1P:yo (đế vương/quý tộc cổ trang -> ta/trẫm)"),
    (rf"{_NO_PREV_KANJI}(?:我(?:が)?|吾)(?:{_NO_NEXT_KANJI_UNLESS_PLURAL}|(?=が[{_KANJI_CLS}]))", "1P:ware (cổ trang/ngạo mạn -> ta)"),
]

_COMPILED_PATTERNS = [(re.compile(pat), desc) for pat, desc in JA_REGISTER_PATTERNS]
_COMPILED_FALLBACK_KANJI = [(re.compile(pat), desc) for pat, desc in _FALLBACK_KANJI_PRONOUN_PATTERNS]


def _extract_morphological_kanji_hints(morphemes: List[Tuple[str, str, str]]) -> List[str]:
    """
    Trích xuất gợi ý đại từ & hậu tố Kanji bằng hình vị MeCab/UniDic chính xác:
    loại bỏ hoàn toàn dương tính giả từ ghép (`私立`, `我慢`, `下僕`, `君臨`, `余裕`, `某所`, `手前味噌`).
    """
    found: List[str] = []

    def _add(label: str) -> None:
        if label not in found:
            found.append(label)

    n = len(morphemes)
    for i, (surf, pos1, _pos2) in enumerate(morphemes):
        next_surf = morphemes[i + 1][0] if i + 1 < n else ""
        prev_surf = morphemes[i - 1][0] if i > 0 else ""

        if surf == "私" and pos1 == "代名詞":
            _add("1P:watashi (chuẩn mực/lịch sự -> tôi/em/mình)")
        elif surf == "僕" and pos1 == "代名詞":
            _add("1P:boku (nam lịch sự/trẻ -> tôi/tớ/em)")
        elif surf == "俺" and pos1 == "代名詞":
            _add("1P:ore (nam tính/mạnh mẽ -> tôi/ta/tao/anh)")
        elif surf == "妾" and pos1 in ("代名詞", "名詞"):
            _add("1P:warawa (cổ trang quý tộc nữ -> thiếp/ta)")
        elif surf in ("我", "吾", "我が", "我ら", "我々") and (pos1 in ("代名詞", "連体詞") or surf in ("我ら", "我々")):
            _add("1P:ware (cổ trang/ngạo mạn -> ta)")
        elif surf in ("余", "予") and pos1 == "代名詞" and next_surf in ("は", "が", "の", "を", "に", "も", "と", "こそ"):
            _add("1P:yo (đế vương/quý tộc cổ trang -> ta/trẫm)")
        elif surf == "某" and pos1 in ("代名詞", "名詞") and next_surf in ("は", "が", "の", "を", "に", "も", "、"):
            _add("1P:soregashi (võ sĩ cổ trang -> tại hạ)")
        elif surf == "君":
            if pos1 == "代名詞":
                _add("2P:kimi (thân mật/trên bảo dưới -> cậu/em)")
            elif pos1 == "接尾辞":
                _add("HON:kun (hậu tố tên -kun ngang hàng/thân mật)")
        elif surf == "手前":
            # Loại trừ `手前味噌` và vị trí không gian `駅の少し手前`, `一歩手前`
            if next_surf != "味噌" and prev_surf not in ("の", "少し", "ちょっと", "一歩", "寸前", "すぐ", "ずっと"):
                if next_surf in ("ら", "達", "ども", "は", "が", "を", "に", "の", "も", "、", "！", "？", ""):
                    _add("2P:temee (thô bạo/đối đầu -> mày/tên khốn)")

    return found


def extract_ja_linguistic_hints(ja_text: str) -> Optional[str]:
    """
    Trích xuất chuỗi gợi ý hình thái học từ câu tiếng Nhật (quét cả bản gốc lẫn bản đã gỡ Furigana
    như '貴(き)様(さま)' -> '貴様'), ưu tiên tách từ MeCab C++ (`fugashi` + `unidic-lite`) để chặn 100%
    báo giả đại từ trên từ ghép Hán-Nhật.
    Trả về None nếu không phát hiện tín hiệu đặc biệt (câu trung tính).
    """
    if not ja_text or not has_cjk_characters(ja_text):
        return None

    stripped = strip_ja_furigana(ja_text)
    hints: List[str] = []

    def _push(desc: str) -> None:
        if desc not in hints:
            hints.append(desc)

    morphemes = tokenize_ja_morphemes(stripped)
    if morphemes:
        for mh in _extract_morphological_kanji_hints(morphemes):
            _push(mh)
    else:
        for pattern, desc in _COMPILED_FALLBACK_KANJI:
            if pattern.search(stripped) or (stripped != ja_text and pattern.search(ja_text)):
                _push(desc)

    for pattern, desc in _COMPILED_PATTERNS:
        if pattern.search(stripped) or (stripped != ja_text and pattern.search(ja_text)):
            _push(desc)

    return " | ".join(hints) if hints else None
