"""Hybrid BM25 and SQLite FTS5 2/3-gram subtitle RAG and error bank manager."""

import os
import re
import sqlite3
import hashlib
import logging
import contextlib
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import pysubs2

from subtrans.models import Cue
from subtrans.eval.metrics import normalize_nfc
from subtrans.ja_morph_tagger import strip_ja_furigana

logger = logging.getLogger("subtrans.rag")

KNOWLEDGE_DB_PATH = Path(__file__).parent / ".cache" / "knowledge.db"


def _generate_ngram_string(text: str) -> str:
    """
    Strips furigana and non-word characters, producing space-separated character 2-grams and 3-grams.
    If length == 1: returns the single character.
    If length >= 2: adds all 2-character slices.
    If length >= 3: adds all 3-character slices.
    """
    if not text:
        return ""
    # Strip furigana: handles (ふりがな), （ふりがな）, and [ふりがな]
    furi_stripped = strip_ja_furigana(text)
    furi_stripped = re.sub(r"\[[^\]]+\]", "", furi_stripped)
    cleaned = re.sub(r"[\s\d\W]+", "", furi_stripped)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned
    tokens = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
    if len(cleaned) >= 3:
        tokens.extend(cleaned[i : i + 3] for i in range(len(cleaned) - 2))
    return " ".join(tokens)

# Canonical Error Bank Seeds
INITIAL_ERROR_BANK_SEEDS = [
    {
        "category": "vocative_name",
        "ja_pattern": "フラン",
        "bad_pattern": "em, |cậu, |mày, ",
        "good_pattern": "Fran",
        "explanation": "Khi tiếng Nhật gọi đích danh 'Fran' (フラン), bắt buộc giữ nguyên Romaji 'Fran', tuyệt đối không thay bằng đại từ chung chung.",
        "series": "tenken",
    },
    {
        "category": "vocative_name",
        "ja_pattern": "玲琳",
        "bad_pattern": "cô, |em, |nàng, ",
        "good_pattern": "Reirin",
        "explanation": "Khi tiếng Nhật gọi đích danh 'Reirin' (玲琳), bắt buộc giữ nguyên Romaji 'Reirin'.",
        "series": "futsutsuka",
    },
    {
        "category": "banned_calque",
        "ja_pattern": "世界観",
        "bad_pattern": "thế giới quan",
        "good_pattern": "thế giới",
        "explanation": "Không dịch máy móc calque Hán-Việt 'thế giới quan' khi ngữ cảnh đang nói về thế giới hiện thực/thế giới xung quanh.",
        "series": None,
    },
    {
        "category": "banned_transliteration",
        "ja_pattern": "フラン",
        "bad_pattern": "Pháp Lạp",
        "good_pattern": "Fran",
        "explanation": "Tuyệt đối không dùng âm Hán-Việt 'Pháp Lạp' để dịch tên nhân vật Fran.",
        "series": "tenken",
    },
    {
        "category": "banned_transliteration",
        "ja_pattern": "ウルシ",
        "bad_pattern": "Ô Lỗ Thi",
        "good_pattern": "Urushi",
        "explanation": "Tuyệt đối không dùng âm Hán-Việt 'Ô Lỗ Thi' cho nhân vật Urushi.",
        "series": "tenken",
    },
    {
        "category": "banned_transliteration",
        "ja_pattern": "慧月|Tuệ Nguyệt|Huệ Nguyệt",
        "bad_pattern": "Huệ Nguyệt|Tuệ Nguyệt",
        "good_pattern": "Keigetsu",
        "explanation": "Tuyệt đối không dịch tên Keigetsu thành Huệ Nguyệt hoặc Tuệ Nguyệt.",
        "series": "futsutsuka",
    },
    {
        "category": "anime_laughter",
        "ja_pattern": "ふはは|ははは|フハハ|ククク|ふふふ",
        "bad_pattern": "[A-Za-z ]{10,}",
        "good_pattern": "(tiếng cười) hoặc (âm thanh)",
        "explanation": "Không gán nhầm câu thoại tiếng Anh overlap của nhân vật khác vào từ tượng thanh tiếng cười anime.",
        "series": None,
    },
]

# Kho câu mẫu chuẩn fan-sub tích hợp sẵn (Cold-Start Gold Exemplars) cho các series trọng điểm & hội thoại chung
BUILTIN_GOLD_EXEMPLARS: List[Dict[str, Optional[str]]] = [
    {
        "series": "frieren",
        "ja": "フリーレン様、そろそろ起きないと置いていきますよ。",
        "en": "Mistress Frieren, if you don't wake up soon, I will leave you behind.",
        "vi": "Cô Frieren, nếu cô không dậy ngay là em bỏ cô lại đấy nhé.",
        "speaker": "Fern",
        "addressee": "Frieren",
    },
    {
        "series": "frieren",
        "ja": "シュタルク様、朝からそんなに食べるとお腹を壊しますよ。",
        "en": "Master Stark, you'll get a stomachache if you eat that much in the morning.",
        "vi": "Cậu Stark, mới sáng sớm mà ăn nhiều thế là đau bụng đấy.",
        "speaker": "Fern",
        "addressee": "Stark",
    },
    {
        "series": "frieren",
        "ja": "ゾルトラークはもう人を殺す魔法じゃない。一般攻撃魔法だよ。",
        "en": "Zoltraak is no longer killing magic. It is ordinary offensive magic now.",
        "vi": "Zoltraak giờ không còn là Ma pháp giết người nữa. Nó đã trở thành Ma pháp tấn công cơ bản rồi.",
        "speaker": "Frieren",
        "addressee": "Qual",
    },
    {
        "series": "violet",
        "ja": "お客様がお望みなら、どこでも駆けつけます。自動手記人形サービス、ヴァイオレット・エヴァーガーデンです。",
        "en": "If it is your wish, I will travel anywhere to meet you. Auto Memory Doll service, Violet Evergarden.",
        "vi": "Nếu đó là tâm nguyện của quý khách, tôi xin sẵn sàng đi tới bất cứ nơi đâu. Dịch vụ Búp bê Ký ức Tự động, Violet Evergarden xin nghe.",
        "speaker": "Violet",
        "addressee": "Client",
    },
    {
        "series": "violet",
        "ja": "少佐が私にくださった「愛してる」という言葉の意味を、知りたいのです。",
        "en": "I want to understand the meaning of the words 'I love you' that the Major gave me.",
        "vi": "Tôi muốn hiểu được ý nghĩa của lời 'Anh yêu em' mà Thiếu tá đã dành cho tôi.",
        "speaker": "Violet",
        "addressee": "Hodgins",
    },
    {
        "series": "jujutsu",
        "ja": "大丈夫、僕最強だから。領域展開「無量空処」。",
        "en": "Don't worry, I'm the strongest. Domain Expansion: Unlimited Void.",
        "vi": "Yên tâm đi, vì thầy là kẻ mạnh nhất mà. Lãnh địa triển khai: Vô Lượng Không Xứ.",
        "speaker": "Gojo",
        "addressee": "Itadori",
    },
    {
        "series": "jujutsu",
        "ja": "頭が高いぞ小僧。伏魔御廚子で刻んでやる。",
        "en": "Know your place, brat. I'll slice you apart with Malevolent Shrine.",
        "vi": "Ngươi hơi ngạo mạn rồi đấy thằng nhóc. Ta sẽ băm xác ngươi bằng Phục Ma Ngự Trù Tử.",
        "speaker": "Sukuna",
        "addressee": "Itadori",
    },
    {
        "series": "kimetsu",
        "ja": "全集中、水の呼吸、壱ノ型・水面斬り！",
        "en": "Total Concentration, Water Breathing, First Form: Water Surface Slash!",
        "vi": "Toàn Tập Trung, Hơi thở của Nước, Thức thứ nhất: Thủy Diện Trảm!",
        "speaker": "Tanjiro",
        "addressee": "Demon",
    },
    {
        "series": "kimetsu",
        "ja": "お館様の御前だぞ。頭を下げろ、炭治郎。",
        "en": "You are in the presence of the Master. Bow your head, Tanjiro.",
        "vi": "Đang ở trước mặt Chúa công đấy. Cúi đầu xuống đi, Tanjiro.",
        "speaker": "Rengoku",
        "addressee": "Tanjiro",
    },
    {
        "series": "rezero",
        "ja": "バルス、朝から見苦しい顔ね。レムを見習いなさい。",
        "en": "Barusu, your face is unsightly first thing in the morning. Follow Rem's example.",
        "vi": "Barusu, mới sáng ra mà mặt mũi đã khó coi rồi đấy. Hãy học tập Rem đi.",
        "speaker": "Ram",
        "addressee": "Subaru",
    },
    {
        "series": "fate",
        "ja": "先輩、朝ごはんの準備ができましたよ。",
        "en": "Senpai, breakfast is ready.",
        "vi": "Senpai, bữa sáng em chuẩn bị xong rồi ạ.",
        "speaker": "Sakura",
        "addressee": "Shirou",
    },
    {
        "series": "tenken",
        "ja": "師匠、次はどいつを斬ればいい？黒猫族を馬鹿にする奴は許さない！",
        "en": "Master, who should I cut down next? I won't forgive anyone who looks down on the Black Cat Tribe!",
        "vi": "Sư phụ, tiếp theo con chém tên nào đây? Con sẽ không tha cho kẻ nào dám coi thường Hắc Miêu Tộc!",
        "speaker": "Fran",
        "addressee": "Shishou",
    },
    {
        "series": "futsutsuka",
        "ja": "慧月様、どうかお体を大切になさってくださいませ。",
        "en": "Lady Keigetsu, please take good care of your health.",
        "vi": "Tiểu thư Keigetsu, xin cô hãy giữ gìn ngọc thể thật tốt ạ.",
        "speaker": "Reirin",
        "addressee": "Keigetsu",
    },
    {
        "series": "mushoku",
        "ja": "ロキシー師匠、シルフィ、俺が必ず家族を守るから。",
        "en": "Master Roxy, Sylphy, I swear I will protect our family.",
        "vi": "Sư phụ Roxy, Sylphy, anh nhất định sẽ bảo vệ gia đình của chúng ta.",
        "speaker": "Rudeus",
        "addressee": "Roxy",
    },
    {
        "series": None,
        "ja": "先輩、今日の部活は休みですか？",
        "en": "Senpai, is club practice canceled today?",
        "vi": "Tiền bối ơi, buổi sinh hoạt câu lạc bộ hôm nay nghỉ ạ?",
        "speaker": "Kohai",
        "addressee": "Senpai",
    },
    {
        "series": None,
        "ja": "貴様ら、この魔王の城に足を踏み入れたことを後悔させてやろう！",
        "en": "You fools, I'll make you regret setting foot inside this Demon King's castle!",
        "vi": "Lũ các ngươi, ta sẽ bắt các ngươi phải hối hận vì dám đặt chân vào tòa thành của Ma Vương này!",
        "speaker": "DemonKing",
        "addressee": "Heroes",
    },
]


class KnowledgeBase:
    """SQLite-backed Knowledge Base managing gold subtitle units and error bank."""

    _generate_ngram_string = staticmethod(_generate_ngram_string)

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or KNOWLEDGE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._error_bank_cache: Dict[Optional[str], List[Tuple[str, str, str, str, str]]] = {}
        self._error_bank_lock = threading.Lock()
        self._init_db()

    def _invalidate_error_bank_cache(self) -> None:
        with self._error_bank_lock:
            self._error_bank_cache.clear()

    @contextlib.contextmanager
    def _connect(self, immediate: bool = False):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            with conn:
                if immediate:
                    conn.execute("BEGIN IMMEDIATE")
                yield conn
        finally:
            conn.close()

    def get_db_sha256(self) -> str:
        """Returns SHA-256 hash of the knowledge base file for experiment provenance tracking."""
        if not self.db_path.exists():
            return "empty"
        hasher = hashlib.sha256()
        with open(self.db_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        with self._connect(immediate=True) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gold_units (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    series TEXT,
                    episode TEXT,
                    ja_text TEXT NOT NULL,
                    ja_ngram TEXT,
                    en_text TEXT,
                    vi_text TEXT NOT NULL,
                    speaker TEXT,
                    addressee TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    lang_pair TEXT NOT NULL DEFAULT 'ja-vi',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gold_series ON gold_units(series)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gold_ja ON gold_units(ja_text)")

            # Schema migration and synchronization for existing DBs
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(gold_units)")
            cols = [r[1] for r in cursor.fetchall()]
            if "status" not in cols:
                conn.execute("ALTER TABLE gold_units ADD COLUMN status TEXT NOT NULL DEFAULT 'draft'")
            if "lang_pair" not in cols:
                conn.execute("ALTER TABLE gold_units ADD COLUMN lang_pair TEXT NOT NULL DEFAULT 'ja-vi'")
            if "ja_ngram" not in cols:
                conn.execute("ALTER TABLE gold_units ADD COLUMN ja_ngram TEXT")

            # SQLite FTS5 index with character 2-gram + 3-gram tokenization
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS gold_units_fts USING fts5(
                    ja_ngram,
                    content='gold_units',
                    content_rowid='id'
                )
            """)

            # Synchronize any unindexed rows into gold_units_fts
            cursor.execute("SELECT id, ja_text FROM gold_units WHERE ja_ngram IS NULL OR ja_ngram = ''")
            unindexed = cursor.fetchall()
            if unindexed:
                for r_id, r_ja in unindexed:
                    ngram_str = _generate_ngram_string(r_ja)
                    cursor.execute("UPDATE gold_units SET ja_ngram = ? WHERE id = ?", (ngram_str, r_id))
                    cursor.execute(
                        "INSERT OR REPLACE INTO gold_units_fts(rowid, ja_ngram) VALUES (?, ?)",
                        (r_id, ngram_str),
                    )

            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_bank (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    ja_pattern TEXT NOT NULL,
                    bad_pattern TEXT NOT NULL,
                    good_pattern TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    series TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_err_series ON error_bank(series)")

            # Seed error bank if empty
            cursor.execute("SELECT COUNT(*) FROM error_bank")
            if cursor.fetchone()[0] == 0:
                for seed in INITIAL_ERROR_BANK_SEEDS:
                    conn.execute("""
                        INSERT INTO error_bank (category, ja_pattern, bad_pattern, good_pattern, explanation, series)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        seed["category"],
                        seed["ja_pattern"],
                        seed["bad_pattern"],
                        seed["good_pattern"],
                        seed["explanation"],
                        seed["series"],
                    ))
            conn.commit()

    def _insert_gold_unit_conn(
        self,
        conn: sqlite3.Connection,
        ja_text: str,
        vi_text: str,
        en_text: Optional[str] = None,
        series: Optional[str] = None,
        episode: Optional[str] = None,
        speaker: Optional[str] = None,
        addressee: Optional[str] = None,
        status: str = "reviewed",
        lang_pair: str = "ja-vi",
    ) -> bool:
        ja_norm = normalize_nfc(ja_text).strip()
        vi_norm = normalize_nfc(vi_text).strip()
        if not ja_norm or not vi_norm:
            return False

        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM gold_units WHERE ja_text = ? AND vi_text = ? AND series IS ?",
            (ja_norm, vi_norm, series),
        )
        if cursor.fetchone():
            return False

        ngram_str = _generate_ngram_string(ja_norm)
        cursor.execute("""
            INSERT INTO gold_units (series, episode, ja_text, ja_ngram, en_text, vi_text, speaker, addressee, status, lang_pair)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            series,
            episode,
            ja_norm,
            ngram_str,
            normalize_nfc(en_text).strip() if en_text else None,
            vi_norm,
            speaker,
            addressee,
            status,
            lang_pair,
        ))
        new_id = cursor.lastrowid
        cursor.execute(
            "INSERT OR REPLACE INTO gold_units_fts(rowid, ja_ngram) VALUES (?, ?)",
            (new_id, ngram_str),
        )
        return True

    def add_gold_unit(
        self,
        ja_text: str,
        vi_text: str,
        en_text: Optional[str] = None,
        series: Optional[str] = None,
        episode: Optional[str] = None,
        speaker: Optional[str] = None,
        addressee: Optional[str] = None,
        status: str = "reviewed",
        lang_pair: str = "ja-vi",
    ) -> bool:
        """
        Inserts a verified subtitle unit if not already present.
        status: 'gold' (human verified), 'reviewed' (archive aligned), 'draft' (unverified pipeline output).
        Only 'gold' and 'reviewed' are ever retrieved as few-shots.
        """
        ja_norm = normalize_nfc(ja_text).strip()
        vi_norm = normalize_nfc(vi_text).strip()
        if not ja_norm or not vi_norm:
            return False

        with self._connect(immediate=True) as conn:
            inserted = self._insert_gold_unit_conn(
                conn,
                ja_text=ja_norm,
                vi_text=vi_norm,
                en_text=en_text,
                series=series,
                episode=episode,
                speaker=speaker,
                addressee=addressee,
                status=status,
                lang_pair=lang_pair,
            )
            if inserted:
                conn.commit()
            return inserted

    def import_subtitles_triplet(
        self,
        ja_path: Path,
        en_path: Optional[Path],
        vi_path: Path,
        series: str,
        episode: Optional[str] = None,
        status: str = "reviewed",
    ) -> int:
        """
        Imports and aligns a triplet of subtitle files (JA, EN, VI) into gold_units.
        Aligns cues based on timestamp overlap.
        """
        if not ja_path.exists() or not vi_path.exists():
            logger.warning(f"File missing for triplet: ja={ja_path}, vi={vi_path}")
            return 0

        ja_ssa = pysubs2.load(str(ja_path))
        vi_ssa = pysubs2.load(str(vi_path))
        en_ssa = pysubs2.load(str(en_path)) if (en_path and en_path.exists()) else []

        imported_count = 0
        ep_name = episode or ja_path.stem
        norm_series = series.lower()

        with self._connect(immediate=True) as conn:
            for ja_evt in ja_ssa:
                if ja_evt.is_comment:
                    continue
                ja_text = re.sub(r"\{[^\}]*\}", "", ja_evt.text).replace(r"\N", " ").strip()
                if not ja_text:
                    continue

                # Find matching VI event by highest overlap
                best_vi_match = None
                best_vi_overlap = 0
                for vi_evt in vi_ssa:
                    if vi_evt.is_comment:
                        continue
                    overlap = max(0, min(ja_evt.end, vi_evt.end) - max(ja_evt.start, vi_evt.start))
                    if overlap > best_vi_overlap:
                        best_vi_overlap = overlap
                        best_vi_match = vi_evt

                if not best_vi_match or best_vi_overlap < 300:
                    continue

                vi_text = re.sub(r"\{[^\}]*\}", "", best_vi_match.text).replace(r"\N", " ").strip()
                if not vi_text:
                    continue

                # Optional EN match
                en_text = None
                if en_ssa:
                    best_en_match = None
                    best_en_overlap = 0
                    for en_evt in en_ssa:
                        if en_evt.is_comment:
                            continue
                        overlap = max(0, min(ja_evt.end, en_evt.end) - max(ja_evt.start, en_evt.start))
                        if overlap > best_en_overlap:
                            best_en_overlap = overlap
                            best_en_match = en_evt
                    if best_en_match and best_en_overlap >= 300:
                        en_text = re.sub(r"\{[^\}]*\}", "", best_en_match.text).replace(r"\N", " ").strip()

                if self._insert_gold_unit_conn(
                    conn,
                    ja_text=ja_text,
                    vi_text=vi_text,
                    en_text=en_text,
                    series=norm_series,
                    episode=ep_name,
                    status=status,
                ):
                    imported_count += 1
            if imported_count:
                conn.commit()

        logger.info(f"Imported {imported_count} gold units from {ja_path.name}")
        return imported_count

    @staticmethod
    def _extract_bigrams(text: str) -> List[str]:
        """Extracts character 2-grams and 3-grams from Japanese text, ignoring whitespace/punctuation."""
        cleaned = re.sub(r"[\s\d\W]+", "", text)
        if len(cleaned) < 2:
            return [cleaned] if cleaned else []
        tokens = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
        if len(cleaned) >= 3:
            tokens.extend(cleaned[i : i + 3] for i in range(len(cleaned) - 2))
        return tokens

    def seed_default_gold_exemplars(self) -> int:
        """Nạp kho câu mẫu chuẩn `BUILTIN_GOLD_EXEMPLARS` vào bảng `gold_units` (idempotent)."""
        inserted = 0
        with self._connect(immediate=True) as conn:
            for ex in BUILTIN_GOLD_EXEMPLARS:
                ok = self._insert_gold_unit_conn(
                    conn,
                    ja_text=str(ex.get("ja") or ""),
                    vi_text=str(ex.get("vi") or ""),
                    en_text=ex.get("en"),
                    series=ex.get("series"),
                    episode="__builtin_seed__",
                    speaker=ex.get("speaker"),
                    addressee=ex.get("addressee"),
                    status="gold",
                )
                if ok:
                    inserted += 1
            if inserted:
                conn.commit()
        return inserted

    def retrieve_few_shots(
        self,
        cues: List[Cue],
        series: Optional[str] = None,
        max_examples: int = 3,
        min_score: float = 0.05,
        exclude_episode: Optional[str] = None,
        fallback_builtin: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Retrieves relevant few-shot examples from gold_units matching cues in the batch.
        Uses Hybrid BM25 scoring over character 2/3-grams + exact Kanji/Katakana term match
        + (speaker, addressee) relational pair bonus.
        Only retrieves units with status IN ('gold', 'reviewed', 'silver') (NEVER retrieves 'draft').
        Strictly isolates series and excludes the current episode being evaluated to prevent test leakage.
        """
        if not cues:
            return []

        # Extract Japanese keywords (Kanji >= 2 chars, Katakana >= 2 chars; furigana-transparent)
        batch_ja = " ".join(strip_ja_furigana(c.ja or "") for c in cues)
        keywords = set(re.findall(r"[\u3400-\u4DBF\u4E00-\u9FFF]{2,}|[\u30A0-\u30FF]{2,}", batch_ja))
        if not keywords and not fallback_builtin:
            return []

        batch_speakers = {
            strip_ja_furigana(c.speaker.strip()).lower()
            for c in cues
            if getattr(c, "speaker", None) and c.speaker.strip().lower() != "unknown"
        }
        batch_pairs = {
            (strip_ja_furigana(c.speaker.strip()).lower(), c.addressee.strip().lower())
            for c in cues
            if getattr(c, "speaker", None)
            and getattr(c, "addressee", None)
            and c.speaker.strip().lower() != "unknown"
            and c.addressee.strip().lower() != "unknown"
        }

        # Compute query n-grams
        query_bigrams: List[str] = []
        for kw in keywords:
            query_bigrams.extend(self._extract_bigrams(kw))
        if not query_bigrams and not fallback_builtin:
            return []

        candidates = []
        if query_bigrams:
            # Fetch candidate gold/reviewed/silver units (never 'draft') via FTS5 MATCH
            clean_tokens = [tok.replace('"', '""') for tok in set(query_bigrams) if tok]
            match_expr = " OR ".join(f'"{tok}"' for tok in clean_tokens)
            if match_expr:
                candidate_limit = max(50, max_examples * 10)
                with self._connect() as conn:
                    cursor = conn.cursor()
                    query = """
                        SELECT g.id, g.ja_text, g.en_text, g.vi_text, g.series, g.episode, g.speaker, g.addressee, g.status
                        FROM gold_units_fts f
                        JOIN gold_units g ON g.id = f.rowid
                        WHERE gold_units_fts MATCH ?
                          AND g.status IN ('reviewed', 'gold', 'silver')
                    """
                    params: List[Any] = [match_expr]
                    if series:
                        query += " AND (g.series = ? OR g.series IS NULL)"
                        params.append(series.lower())
                    if exclude_episode:
                        query += " AND (g.episode != ? OR g.episode IS NULL)"
                        params.append(exclude_episode)
                    query += " ORDER BY f.rank LIMIT ?"
                    params.append(candidate_limit)
                    cursor.execute(query, params)
                    candidates = cursor.fetchall()

        if fallback_builtin and len(candidates) < max_examples:
            seen_ja = {row[1] for row in candidates}
            series_norm = series.lower() if series else None
            next_neg_id = -1
            for ex in BUILTIN_GOLD_EXEMPLARS:
                ex_series = ex.get("series")
                if series_norm and ex_series is not None and ex_series != series_norm:
                    continue
                if not series_norm and ex_series is not None:
                    continue
                ex_ja = str(ex.get("ja") or "")
                if ex_ja and ex_ja not in seen_ja:
                    candidates.append((
                        next_neg_id,
                        ex_ja,
                        ex.get("en"),
                        str(ex.get("vi") or ""),
                        ex_series,
                        "__builtin_seed__",
                        ex.get("speaker"),
                        ex.get("addressee"),
                        "gold",
                    ))
                    seen_ja.add(ex_ja)
                    next_neg_id -= 1

        if not candidates:
            return []

        # Build candidate document n-gram representations (furigana-transparent)
        doc_bigrams: Dict[int, List[str]] = {}
        doc_clean_ja: Dict[int, str] = {}
        for cid, ja_text, en_text, vi_text, s_val, ep_val, spk_val, addr_val, st_val in candidates:
            c_ja = strip_ja_furigana(ja_text)
            doc_clean_ja[cid] = c_ja
            doc_bigrams[cid] = self._extract_bigrams(c_ja)

        # Compute document frequencies and BM25 IDF for query n-grams
        N = len(candidates)
        avgdl = sum(len(bg) for bg in doc_bigrams.values()) / max(1, N)
        k1 = 1.2
        b = 0.75

        unique_q_bigrams = set(query_bigrams)
        df: Dict[str, int] = {}
        for qb in unique_q_bigrams:
            df[qb] = sum(1 for cid in doc_bigrams if qb in doc_bigrams[cid])

        # Hybrid BM25 + Exact Term + Speaker/Addressee Pair Scoring
        scored_candidates: List[Tuple[float, Dict[str, str]]] = []
        import math
        for cid, ja_text, en_text, vi_text, s_val, ep_val, spk_val, addr_val, st_val in candidates:
            d_bgs = doc_bigrams[cid]
            dl = len(d_bgs)
            if dl == 0:
                continue

            tf: Dict[str, int] = {}
            for bg in d_bgs:
                tf[bg] = tf.get(bg, 0) + 1

            score = 0.0
            if query_bigrams:
                for qb in unique_q_bigrams:
                    if qb in tf:
                        n_q = df[qb]
                        idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
                        term_tf = tf[qb]
                        num = term_tf * (k1 + 1.0)
                        den = term_tf + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
                        score += idf * (num / den)
            elif fallback_builtin and ep_val == "__builtin_seed__":
                # Hiragana/EN-only fallback: prioritize series-matching built-in exemplars
                score += 0.15 if (series and s_val == series.lower()) else 0.06

            # Exact multi-char term match bonus
            c_ja = doc_clean_ja[cid]
            exact_hits = sum(1 for kw in keywords if kw in c_ja)
            if exact_hits:
                score += 0.65 * exact_hits

            # Speaker / Addressee relational match bonus
            c_spk = strip_ja_furigana((spk_val or "").strip()).lower()
            c_addr = (addr_val or "").strip().lower()
            if c_spk and c_addr and (c_spk, c_addr) in batch_pairs:
                score += 0.60
            elif c_spk and c_spk in batch_speakers:
                score += 0.25

            if score <= 0.0:
                continue

            # Prefer human gold / reviewed over auto-curated silver
            tier_weight = 1.20 if st_val == "gold" else (1.05 if st_val == "reviewed" else 0.90)
            score *= tier_weight

            if score >= min_score:
                item = {"ja": ja_text, "vi": vi_text}
                if en_text:
                    item["en"] = en_text
                    item["en_ref"] = en_text
                if spk_val:
                    item["speaker"] = spk_val
                if addr_val:
                    item["addressee"] = addr_val
                scored_candidates.append((score, item))

        # Sort by BM25 score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate and return top max_examples
        results: List[Dict[str, str]] = []
        for sc, item in scored_candidates:
            if item not in results:
                results.append(item)
            if len(results) >= max_examples:
                break

        if results:
            logger.info(
                f"📚 [RAG BM25] Đã chọn {len(results)}/{len(candidates)} ví dụ gold/reviewed "
                f"(top score={scored_candidates[0][0]:.2f}): {[r['ja'][:15] for r in results]}"
            )
        return results

    def search_gold_units(
        self,
        query_text: str,
        top_k: int = 5,
        series: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fast inverted index search via SQLite FTS5 for 2-gram/3-gram Kanji and glossary retrieval."""
        ngram_str = _generate_ngram_string(query_text)
        tokens = ngram_str.split()
        if not tokens:
            return []

        clean_tokens = [t.replace('"', '""') for t in set(tokens) if t]
        match_expr = " OR ".join(f'"{t}"' for t in clean_tokens)
        if not match_expr:
            return []

        with self._connect() as conn:
            cursor = conn.cursor()
            query = """
                SELECT g.id, g.ja_text, g.en_text, g.vi_text, g.speaker, g.addressee, g.series, g.episode, g.status, f.rank
                FROM gold_units_fts f
                JOIN gold_units g ON g.id = f.rowid
                WHERE gold_units_fts MATCH ?
                  AND g.status IN ('reviewed', 'gold', 'silver')
            """
            params: List[Any] = [match_expr]
            if series:
                query += " AND (g.series = ? OR g.series IS NULL)"
                params.append(series.lower())
            query += " ORDER BY f.rank LIMIT ?"
            params.append(top_k)

            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "ja": row[1],
                    "ja_text": row[1],
                    "en": row[2],
                    "en_text": row[2],
                    "vi": row[3],
                    "vi_text": row[3],
                    "speaker": row[4],
                    "addressee": row[5],
                    "series": row[6],
                    "episode": row[7],
                    "status": row[8],
                    "score": -float(row[9]) if row[9] is not None else 0.0,
                })
            return results

    def record_correction(
        self,
        ja_text: str,
        wrong_vi: str,
        corrected_vi: str,
        category: str = "human_correction",
        explanation: Optional[str] = None,
        series: Optional[str] = None,
        episode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Feedback Loop ('subtrans learn'):
        Records a human-corrected subtitle into both Error Bank and gold_units (status='gold').
        Prevents repeating the same mistake in future episodes.
        """
        ja_norm = normalize_nfc(ja_text).strip()
        wrong_norm = normalize_nfc(wrong_vi).strip()
        corr_norm = normalize_nfc(corrected_vi).strip()
        expl = explanation or f"Sửa '{wrong_norm}' thành '{corr_norm}'"

        # 1. Add to gold_units as gold
        gold_added = self.add_gold_unit(
            ja_text=ja_norm,
            vi_text=corr_norm,
            series=series,
            episode=episode,
            status="gold"
        )

        # 2. Add rule to error_bank
        with self._connect(immediate=True) as conn:
            conn.execute("""
                INSERT INTO error_bank (category, ja_pattern, bad_pattern, good_pattern, explanation, series)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                category,
                ja_norm,
                re.escape(wrong_norm),
                corr_norm,
                expl,
                series.lower() if series else None,
            ))
            conn.commit()
        self._invalidate_error_bank_cache()

        logger.info(f"💡 [LEARN] Đã nạp phản hồi sửa lỗi vào Error Bank và Gold Units: '{corr_norm}'")
        return {
            "status": "success",
            "gold_added": gold_added,
            "ja": ja_norm,
            "wrong": wrong_norm,
            "corrected": corr_norm,
            "series": series,
            "episode": episode,
        }

    def _insert_error_pattern_conn(
        self,
        conn: sqlite3.Connection,
        category: str,
        ja_pattern: str,
        bad_pattern: str,
        good_pattern: str,
        explanation: str,
        series: Optional[str] = None,
    ) -> bool:
        ja_norm = normalize_nfc(ja_pattern).strip()
        bad_norm = normalize_nfc(bad_pattern).strip()
        good_norm = normalize_nfc(good_pattern).strip()
        series_norm = series.lower().strip() if series else None

        if not ja_norm or not bad_norm:
            return False

        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM error_bank
            WHERE category = ? AND ja_pattern = ? AND bad_pattern = ? AND (series = ? OR (series IS NULL AND ? IS NULL))
        """, (category, ja_norm, bad_norm, series_norm, series_norm))
        if cursor.fetchone():
            return False

        conn.execute("""
            INSERT INTO error_bank (category, ja_pattern, bad_pattern, good_pattern, explanation, series)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (category, ja_norm, bad_norm, good_norm, explanation, series_norm))
        return True

    def add_error_pattern(
        self,
        category: str,
        ja_pattern: str,
        bad_pattern: str,
        good_pattern: str,
        explanation: str,
        series: Optional[str] = None,
    ) -> bool:
        """
        Inserts an error pattern into error_bank if not already present.
        Returns True if newly inserted, False if already exists.
        """
        ja_norm = normalize_nfc(ja_pattern).strip()
        bad_norm = normalize_nfc(bad_pattern).strip()
        if not ja_norm or not bad_norm:
            return False

        with self._connect(immediate=True) as conn:
            inserted = self._insert_error_pattern_conn(
                conn,
                category=category,
                ja_pattern=ja_norm,
                bad_pattern=bad_norm,
                good_pattern=good_pattern,
                explanation=explanation,
                series=series,
            )
            if inserted:
                conn.commit()
        if inserted:
            self._invalidate_error_bank_cache()
        return inserted

    def learn_from_linter_violations(
        self,
        violations: List[Any],
        cues: List[Cue],
        series: Optional[str] = None,
        episode: Optional[str] = None,
        glossary: Optional[Any] = None,
    ) -> int:
        """
        Extracts and stores error patterns from linter violations into error_bank.
        Captures:
        - RULE_BANNED_TERM: banned translationese/calques
        - RULE_CJK_LEAK: leftover Japanese/Chinese characters
        - RULE_ROMAJI_MISSING: missing character Romaji name when called in vocative
        - RULE_GLOSSARY_TERM: missing canonical glossary translation
        Returns count of newly learned patterns.
        """
        if not violations or not cues:
            return 0

        _known_banned_ja = {
            "thế giới quan": "世界観",
            "pháp lạp": "フラン",
            "ô lỗ thi": "ウルシ",
            "linh lâm": "玲琳",
            "huệ nguyệt": "慧月",
            "tuệ nguyệt": "慧月",
            "a mạn đạt": "アマンダ",
            "khắc lâm mỗ đặc": "クリムト",
        }
        _known_vocative_ja = {
            "fran": "フラン",
            "jean": "ジャン",
            "nell": "ネル",
            "reirin": "玲琳",
            "keigetsu": "慧月",
            "urushi": "ウルシ",
            "stephane": "ステファン",
            "fel": "フェル",
        }
        if glossary and getattr(glossary, "characters", None):
            for ch in glossary.characters:
                c_vi = (getattr(ch, "name_vi", "") or "").strip().lower()
                c_ja = strip_ja_furigana(getattr(ch, "name_ja", "") or "").strip()
                if c_vi and c_ja:
                    _known_vocative_ja.setdefault(c_vi, c_ja)

        cue_map = {c.id: c for c in cues}
        learned_count = 0

        with self._connect(immediate=True) as conn:
            for v in violations:
                cue = cue_map.get(getattr(v, "cue_id", -1))
                if not cue or not cue.ja or not cue.vi:
                    continue
                if getattr(cue, "used_en_fallback", False) or cue.vi.strip() in (
                    "[Chưa dịch]",
                    "(Chưa dịch)",
                    "[Untranslated]",
                    "[DRY-RUN]",
                ):
                    continue

                rule_id = getattr(v, "rule_id", "")
                ja_text = cue.ja.strip()
                ja_clean = strip_ja_furigana(ja_text)
                vi_text = cue.vi.strip()

                # Guard against error_bank self-poisoning: skip non-deterministic WARNING rules on uncertain drafts
                if rule_id == "LINT_GLOSSARY_TERM" and getattr(cue, "uncertain", False):
                    continue

                if rule_id == "LINT_BANNED_TERM":
                    match = re.search(r"['\"]([^'\"]+)['\"]", getattr(v, "message", ""))
                    bad_word = match.group(1) if match else ""
                    suggestion = getattr(v, "suggestion", "")
                    match_sugg = re.search(r"['\"]([^'\"]+)['\"]", suggestion)
                    good_word = match_sugg.group(1) if match_sugg else suggestion

                    if bad_word:
                        mapped_ja = _known_banned_ja.get(bad_word.lower())
                        ja_pat_src = mapped_ja if (mapped_ja and mapped_ja in ja_clean) else ja_clean[:30]
                        if self._insert_error_pattern_conn(
                            conn,
                            category="banned_calque",
                            ja_pattern=re.escape(ja_pat_src),
                            bad_pattern=bad_word,
                            good_pattern=good_word or "Không dùng từ cấm",
                            explanation=f"Tránh dùng cụm '{bad_word}', nên dùng '{good_word}'.",
                            series=series,
                        ):
                            learned_count += 1

                elif rule_id == "LINT_CJK_LEAK":
                    from subtrans.eval.metrics import detect_cjk_leakage
                    leaks = detect_cjk_leakage(vi_text)
                    if leaks:
                        leak_chars = "".join(dict.fromkeys(leaks))
                        ja_pat_src = leak_chars if leak_chars in ja_clean else ja_clean[:30]
                        if self._insert_error_pattern_conn(
                            conn,
                            category="cjk_leak",
                            ja_pattern=re.escape(ja_pat_src),
                            bad_pattern=re.escape(leak_chars),
                            good_pattern="[Dịch hoàn toàn sang tiếng Việt]",
                            explanation=f"Bản dịch sót ký tự CJK '{leak_chars}'. Dịch hoàn toàn sang tiếng Việt.",
                            series=series,
                        ):
                            learned_count += 1

                elif rule_id == "LINT_ROMAJI_MISSING":
                    match = re.search(r"called '([^']+)'", getattr(v, "message", ""))
                    char_name = match.group(1) if match else ""
                    if char_name:
                        mapped_ja = _known_vocative_ja.get(char_name.lower())
                        ja_pat_src = mapped_ja if (mapped_ja and mapped_ja in ja_clean) else ja_clean[:30]
                        if self._insert_error_pattern_conn(
                            conn,
                            category="vocative_name",
                            ja_pattern=re.escape(ja_pat_src),
                            bad_pattern=r"\b(anh|em|cậu|cô|mày|hắn|nó)\b",
                            good_pattern=char_name,
                            explanation=f"Khi câu tiếng Nhật gọi đích danh '{char_name}', bắt buộc giữ tên Romaji '{char_name}'.",
                            series=series,
                        ):
                            learned_count += 1

                elif rule_id == "LINT_GLOSSARY_TERM":
                    m_term = re.search(
                        r"glossary term '([^']+)' but expected translation '([^']+)'",
                        getattr(v, "message", ""),
                    )
                    if m_term:
                        term_ja, term_vi = m_term.group(1), m_term.group(2)
                        term_ja_clean = strip_ja_furigana(term_ja)
                        if self._insert_error_pattern_conn(
                            conn,
                            category="glossary_term",
                            ja_pattern=re.escape(term_ja_clean),
                            bad_pattern=f"Thiếu thuật ngữ '{term_vi}'",
                            good_pattern=term_vi,
                            explanation=f"Thuật ngữ '{term_ja}' bắt buộc dịch chuẩn theo Glossary là '{term_vi}'.",
                            series=series,
                        ):
                            learned_count += 1
            if learned_count:
                conn.commit()

        if learned_count:
            self._invalidate_error_bank_cache()
            logger.info(f"🧠 [RAG LEARN] Đã tự động nạp {learned_count} bẫy lỗi từ linter vào error_bank (series={series})")
        return learned_count

    def promote_cascade_repairs(
        self,
        cues: List[Cue],
        glossary: Optional[Any] = None,
        series: Optional[str] = None,
        episode: Optional[str] = None,
        max_per_episode: int = 12,
    ) -> int:
        """
        Auto-curates high-confidence Cascade Flash-repaired cues (0 lint violations, non-uncertain,
        non-large-edit, non-digit-mismatch) into gold_units with status='silver' for future episodes.
        """
        if not cues:
            return 0
        from subtrans.linter import lint_cue

        promoted = 0
        norm_series = series.lower().strip() if series else None
        for cue in cues:
            if promoted >= max_per_episode:
                break
            if (
                "(repair)" not in (getattr(cue, "model_used", "") or "")
                or getattr(cue, "uncertain", False)
                or getattr(cue, "is_large_edit", False)
                or getattr(cue, "is_digit_mismatch", False)
                or getattr(cue, "metadata_disagreement", False)
                or getattr(cue, "repair_reason_code", "") not in ("accepted_lint_clean", "accepted_valid")
            ):
                continue
            ja_clean = strip_ja_furigana(cue.ja or "").strip()
            vi_clean = (cue.vi or "").strip()
            if len(ja_clean) < 4 or not vi_clean:
                continue
            if lint_cue(cue, glossary=glossary):
                continue
            if self.add_gold_unit(
                ja_text=cue.ja or ja_clean,
                vi_text=vi_clean,
                en_text=cue.en,
                series=norm_series,
                episode=episode,
                speaker=cue.speaker if (cue.speaker and cue.speaker.lower() != "unknown") else None,
                addressee=cue.addressee,
                status="silver",
            ):
                promoted += 1
        if promoted:
            logger.info(f"🥈 [RAG SILVER] Đã tự động tuyển chọn {promoted} câu Cascade sạch làm mẫu silver (series={norm_series}).")
        return promoted

    def retrieve_cautions(
        self,
        cues: List[Cue],
        series: Optional[str] = None,
        max_cautions: int = 2,
    ) -> List[Dict[str, str]]:
        """
        Retrieves matching cautions from error_bank if patterns match current batch.
        """
        if not cues:
            return []

        batch_ja_raw = " ".join(c.ja or "" for c in cues)
        batch_ja_clean = " ".join(strip_ja_furigana(c.ja or "") for c in cues)
        cautions: List[Dict[str, str]] = []
        series_key = series.lower() if series else None

        with self._error_bank_lock:
            rows = self._error_bank_cache.get(series_key)

        if rows is None:
            with self._connect() as conn:
                cursor = conn.cursor()
                query = "SELECT category, ja_pattern, bad_pattern, good_pattern, explanation FROM error_bank"
                params: List[Any] = []
                if series_key:
                    query += " WHERE series = ? OR series IS NULL"
                    params.append(series_key)
                cursor.execute(query, params)
                rows = cursor.fetchall()
            with self._error_bank_lock:
                self._error_bank_cache[series_key] = rows

        for cat, ja_pat, bad_pat, good_pat, expl in rows:
            try:
                if re.search(ja_pat, batch_ja_clean) or re.search(ja_pat, batch_ja_raw):
                    cautions.append({
                        "category": cat,
                        "rule": expl,
                        "avoid": bad_pat,
                        "prefer": good_pat,
                    })
            except re.error:
                pass
            if len(cautions) >= max_cautions:
                break

        return cautions[:max_cautions]

    def get_stats(self) -> Dict[str, Any]:
        """Returns counts of gold units and error bank items."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM gold_units")
            gold_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT series) FROM gold_units")
            series_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM error_bank")
            error_count = cursor.fetchone()[0]
            return {
                "gold_units_count": gold_count,
                "distinct_series_count": series_count,
                "error_bank_rules_count": error_count,
            }
