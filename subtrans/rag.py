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
import hashlib
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
                    status TEXT NOT NULL DEFAULT 'draft',
                    lang_pair TEXT NOT NULL DEFAULT 'ja-vi',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gold_series ON gold_units(series)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gold_ja ON gold_units(ja_text)")

            # Schema migration for existing DBs
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(gold_units)")
            cols = [r[1] for r in cursor.fetchall()]
            if "status" not in cols:
                conn.execute("ALTER TABLE gold_units ADD COLUMN status TEXT NOT NULL DEFAULT 'draft'")
            if "lang_pair" not in cols:
                conn.execute("ALTER TABLE gold_units ADD COLUMN lang_pair TEXT NOT NULL DEFAULT 'ja-vi'")

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

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM gold_units WHERE ja_text = ? AND vi_text = ? AND series IS ?",
                (ja_norm, vi_norm, series),
            )
            if cursor.fetchone():
                return False

            conn.execute("""
                INSERT INTO gold_units (series, episode, ja_text, en_text, vi_text, speaker, addressee, status, lang_pair)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                series,
                episode,
                ja_norm,
                normalize_nfc(en_text).strip() if en_text else None,
                vi_norm,
                speaker,
                addressee,
                status,
                lang_pair,
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
                status=status,
            ):
                imported_count += 1

        logger.info(f"Imported {imported_count} gold units from {ja_path.name}")
        return imported_count

    @staticmethod
    def _extract_bigrams(text: str) -> List[str]:
        """Extracts character bigrams from Japanese text, ignoring whitespace/punctuation."""
        cleaned = re.sub(r"[\s\d\W]+", "", text)
        if len(cleaned) < 2:
            return [cleaned] if cleaned else []
        return [cleaned[i:i+2] for i in range(len(cleaned) - 1)]

    def retrieve_few_shots(
        self,
        cues: List[Cue],
        series: Optional[str] = None,
        max_examples: int = 3,
        min_score: float = 0.05,
        exclude_episode: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Retrieves relevant few-shot examples from gold_units matching cues in the batch.
        Uses BM25 scoring over character bigrams with IDF weighting.
        Only retrieves units with status IN ('gold', 'reviewed') (NEVER retrieves 'draft').
        Strictly isolates series and excludes the current episode being evaluated to prevent test leakage.
        """
        if not cues:
            return []

        # Extract Japanese keywords (Kanji >= 2 chars, Katakana >= 2 chars) to check eligibility
        batch_ja = " ".join(c.ja for c in cues)
        keywords = set(re.findall(r"[\u4E00-\u9FAF]{2,}|[\u30A0-\u30FF]{2,}", batch_ja))
        if not keywords:
            return []

        # Compute query bigrams
        query_bigrams: List[str] = []
        for kw in keywords:
            query_bigrams.extend(self._extract_bigrams(kw))
        if not query_bigrams:
            return []

        # Fetch candidate gold units (status must be 'reviewed' or 'gold', never 'draft')
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = """
                SELECT id, ja_text, en_text, vi_text, series, episode
                FROM gold_units
                WHERE status IN ('reviewed', 'gold')
            """
            params: List[Any] = []
            if series:
                query += " AND (series = ? OR series IS NULL)"
                params.append(series.lower())
            if exclude_episode:
                query += " AND (episode != ? OR episode IS NULL)"
                params.append(exclude_episode)
            cursor.execute(query, params)
            candidates = cursor.fetchall()

        if not candidates:
            return []

        # Build candidate document bigram representations
        doc_bigrams: Dict[int, List[str]] = {}
        for cid, ja_text, en_text, vi_text, s_val, ep_val in candidates:
            doc_bigrams[cid] = self._extract_bigrams(ja_text)

        # Compute document frequencies and BM25 IDF for query bigrams
        N = len(candidates)
        avgdl = sum(len(bg) for bg in doc_bigrams.values()) / max(1, N)
        k1 = 1.2
        b = 0.75

        unique_q_bigrams = set(query_bigrams)
        df: Dict[str, int] = {}
        for qb in unique_q_bigrams:
            df[qb] = sum(1 for cid in doc_bigrams if qb in doc_bigrams[cid])

        # BM25 Scoring
        scored_candidates: List[Tuple[float, Dict[str, str]]] = []
        import math
        for cid, ja_text, en_text, vi_text, s_val, ep_val in candidates:
            d_bgs = doc_bigrams[cid]
            dl = len(d_bgs)
            if dl == 0:
                continue

            # TF count for this doc
            tf: Dict[str, int] = {}
            for bg in d_bgs:
                tf[bg] = tf.get(bg, 0) + 1

            score = 0.0
            for qb in unique_q_bigrams:
                if qb in tf:
                    n_q = df[qb]
                    idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
                    term_tf = tf[qb]
                    num = term_tf * (k1 + 1.0)
                    den = term_tf + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
                    score += idf * (num / den)

            if score >= min_score:
                item = {"ja": ja_text, "vi": vi_text}
                if en_text:
                    item["en"] = en_text
                    item["en_ref"] = en_text
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
        with sqlite3.connect(self.db_path) as conn:
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
