# -*- coding: utf-8 -*-
"""
subtrans/cache.py
Stage 3: SQLite cache for translated items.
Keyed by SHA-256 hash of input text, context, glossary, and model.
"""

import os
import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

CACHE_DIR = Path(__file__).parent / ".cache"
DB_PATH = CACHE_DIR / "translations.db"


def compute_cue_cache_key(model: str, ja: str, en: Optional[str] = None, glossary_hash: str = "") -> str:
    """Generates a SHA-256 cache key for an individual dialogue cue."""
    canonical = json.dumps({
        "model": model,
        "ja": ja.strip(),
        "en": (en or "").strip(),
        "glossary": glossary_hash,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TranslationCache:
    """Manages SQLite cache for translation responses."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    cache_key TEXT PRIMARY KEY,
                    backend TEXT,
                    model TEXT,
                    vi TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get_cue(self, cache_key: str) -> Optional[str]:
        """Retrieves cached translation for a cue."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vi FROM translations WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
            if row:
                return row[0]
        return None

    def set_cue(self, cache_key: str, backend: str, model: str, vi: str):
        """Saves translated cue into cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO translations (cache_key, backend, model, vi)
                VALUES (?, ?, ?, ?)
            """, (cache_key, backend, model, vi))
            conn.commit()

    def count(self) -> int:
        """Returns total cached items."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM translations")
            row = cursor.fetchone()
            return row[0] if row else 0
