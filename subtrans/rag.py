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
from subtrans.ja_morph_tagger import strip_ja_furigana

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
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
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
        """Extracts character 2-grams and 3-grams from Japanese text, ignoring whitespace/punctuation."""
        cleaned = re.sub(r"[\s\d\W]+", "", text)
        if len(cleaned) < 2:
            return [cleaned] if cleaned else []
        tokens = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
        if len(cleaned) >= 3:
            tokens.extend(cleaned[i : i + 3] for i in range(len(cleaned) - 2))
        return tokens

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
        if not keywords:
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
        if not query_bigrams:
            return []

        # Fetch candidate gold/reviewed/silver units (never 'draft')
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = """
                SELECT id, ja_text, en_text, vi_text, series, episode, speaker, addressee, status
                FROM gold_units
                WHERE status IN ('reviewed', 'gold', 'silver')
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
            for qb in unique_q_bigrams:
                if qb in tf:
                    n_q = df[qb]
                    idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
                    term_tf = tf[qb]
                    num = term_tf * (k1 + 1.0)
                    den = term_tf + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
                    score += idf * (num / den)

            if score <= 0.0:
                continue

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
        good_norm = normalize_nfc(good_pattern).strip()
        series_norm = series.lower().strip() if series else None

        if not ja_norm or not bad_norm:
            return False

        with sqlite3.connect(self.db_path) as conn:
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
            conn.commit()
            return True

    def learn_from_linter_violations(
        self,
        violations: List[Any],
        cues: List[Cue],
        series: Optional[str] = None,
        episode: Optional[str] = None,
    ) -> int:
        """
        Extracts and stores error patterns from linter violations into error_bank.
        Captures:
        - RULE_BANNED_TERM: banned translationese/calques
        - RULE_CJK_LEAK: leftover Japanese/Chinese characters
        - RULE_ROMAJI_MISSING: missing character Romaji name when called in vocative
        Returns count of newly learned patterns.
        """
        if not violations or not cues:
            return 0

        cue_map = {c.id: c for c in cues}
        learned_count = 0

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
                    if self.add_error_pattern(
                        category="banned_calque",
                        ja_pattern=re.escape(ja_text[:30]),
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
                    leak_chars = "".join(set(leaks))
                    if self.add_error_pattern(
                        category="cjk_leak",
                        ja_pattern=re.escape(ja_text[:30]),
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
                    ja_clean = strip_ja_furigana(ja_text)
                    if self.add_error_pattern(
                        category="vocative_name",
                        ja_pattern=re.escape(ja_clean[:30]),
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
                    if self.add_error_pattern(
                        category="glossary_term",
                        ja_pattern=re.escape(strip_ja_furigana(term_ja)),
                        bad_pattern=re.escape(vi_text[:40]),
                        good_pattern=term_vi,
                        explanation=f"Thuật ngữ '{term_ja}' bắt buộc dịch chuẩn theo Glossary là '{term_vi}'.",
                        series=series,
                    ):
                        learned_count += 1

        if learned_count:
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
