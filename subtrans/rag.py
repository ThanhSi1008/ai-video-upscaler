# -*- coding: utf-8 -*-
"""
subtrans/rag.py
Stage P3: Lightweight Subtitle RAG & Error Bank.
- SQLite storage for verified gold subtitle units (.cache/knowledge.db)
- Auto-seeding of Error Bank from regression suite
- Triplet alignment importer for (JA, EN, VI) archive subtitles
- Contextual few-shot retrieval and error prevention prompt injection
"""

import os
import re
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import pysubs2

from subtrans.models import Cue
from subtrans.eval.metrics import normalize_nfc

logger = logging.getLogger("subtrans.rag")

KNOWLEDGE_DB_PATH = Path(__file__).parent / ".cache" / "knowledge.db"

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


class KnowledgeBase:
    """SQLite-backed Knowledge Base managing gold subtitle units and error bank."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or KNOWLEDGE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gold_units (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    series TEXT,
                    episode TEXT,
                    ja_text TEXT NOT NULL,
                    en_text TEXT,
                    vi_text TEXT NOT NULL,
                    speaker TEXT,
                    addressee TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gold_series ON gold_units(series)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gold_ja ON gold_units(ja_text)")

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
            cursor = conn.cursor()
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

    def add_gold_unit(
        self,
        ja_text: str,
        vi_text: str,
        en_text: Optional[str] = None,
        series: Optional[str] = None,
        episode: Optional[str] = None,
        speaker: Optional[str] = None,
        addressee: Optional[str] = None,
    ) -> bool:
        """Inserts a verified gold subtitle unit if not already present."""
        ja_norm = normalize_nfc(ja_text).strip()
        vi_norm = normalize_nfc(vi_text).strip()
        if not ja_norm or not vi_norm:
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM gold_units WHERE ja_text = ? AND vi_text = ? AND series IS ?",
                (ja_norm, vi_norm, series),
            )
            if cursor.fetchone():
                return False

            conn.execute("""
                INSERT INTO gold_units (series, episode, ja_text, en_text, vi_text, speaker, addressee)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                series,
                episode,
                ja_norm,
                normalize_nfc(en_text).strip() if en_text else None,
                vi_norm,
                speaker,
                addressee,
            ))
            conn.commit()
            return True

    def import_subtitles_triplet(
        self,
        ja_path: Path,
        en_path: Optional[Path],
        vi_path: Path,
        series: str,
        episode: Optional[str] = None,
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

            if self.add_gold_unit(
                ja_text=ja_text,
                vi_text=vi_text,
                en_text=en_text,
                series=series.lower(),
                episode=ep_name,
            ):
                imported_count += 1

        logger.info(f"Imported {imported_count} gold units from {ja_path.name}")
        return imported_count

    def retrieve_few_shots(
        self,
        cues: List[Cue],
        series: Optional[str] = None,
        max_examples: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Retrieves relevant few-shot examples from gold_units matching cues in the batch.
        Extracts key Kanji/names to perform targeted keyword lookup.
        """
        if not cues:
            return []

        # Extract Japanese keywords (Kanji words >= 2 chars, Katakana terms)
        batch_ja = " ".join(c.ja for c in cues)
        keywords = set(re.findall(r"[\u4E00-\u9FAF]{2,}|[\u30A0-\u30FF]{2,}", batch_ja))
        if not keywords:
            return []

        results: List[Dict[str, str]] = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for kw in list(keywords)[:8]:
                query = "SELECT ja_text, en_text, vi_text FROM gold_units WHERE ja_text LIKE ?"
                params = [f"%{kw}%"]
                if series:
                    query += " AND (series = ? OR series IS NULL)"
                    params.append(series.lower())
                query += " LIMIT 2"
                cursor.execute(query, params)
                for row in cursor.fetchall():
                    item = {"ja": row[0], "vi": row[2]}
                    if row[1]:
                        item["en"] = row[1]
                    if item not in results:
                        results.append(item)
                    if len(results) >= max_examples:
                        return results

        return results[:max_examples]

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

        batch_ja = " ".join(c.ja for c in cues)
        cautions: List[Dict[str, str]] = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = "SELECT category, ja_pattern, bad_pattern, good_pattern, explanation FROM error_bank"
            params = []
            if series:
                query += " WHERE series = ? OR series IS NULL"
                params.append(series.lower())
            cursor.execute(query, params)
            for cat, ja_pat, bad_pat, good_pat, expl in cursor.fetchall():
                try:
                    if re.search(ja_pat, batch_ja):
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
        with sqlite3.connect(self.db_path) as conn:
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
