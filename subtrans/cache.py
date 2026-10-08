# -*- coding: utf-8 -*-
"""
subtrans/cache.py
Stage 3: SQLite cache for translated items.
Keyed by SHA-256 hash of input text, context, glossary, and model.
Hỗ trợ đếm request và cue theo giờ Thái Bình Dương (Pacific Time - PT / UTC-7),
không bao giờ cache chuỗi [Chưa dịch].
"""

import os
import sqlite3
import hashlib
import json
import zoneinfo
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

CACHE_DIR = Path(__file__).parent / ".cache"
DB_PATH = CACHE_DIR / "translations.db"
UNTRANSLATED_STRINGS = {"[chưa dịch]", "(chưa dịch)", "[untranslated]", "chưa dịch"}

PT_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
VN_TZ = zoneinfo.ZoneInfo("Asia/Ho_Chi_Minh")


def get_now_pt() -> datetime.datetime:
    """Trả về thời gian hiện tại theo múi giờ Thái Bình Dương (PT)."""
    return datetime.datetime.now(PT_TZ)


def get_today_pt_str() -> str:
    """Trả về chuỗi ngày YYYY-MM-DD theo giờ Thái Bình Dương."""
    return get_now_pt().strftime("%Y-%m-%d")


def get_next_pt_reset() -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Tính toán chính xác thời điểm 00:00 ngày tiếp theo theo giờ Thái Bình Dương (PT)
    và thời điểm tương ứng theo giờ Việt Nam (VN: 14:00 khi daylight saving PDT, 15:00 sau 1/11 khi PST).
    """
    now_pt = get_now_pt()
    next_pt = (now_pt + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    next_vn = next_pt.astimezone(VN_TZ)
    return next_pt, next_vn


def compute_key_id(api_key: str) -> str:
    """Trả về 8 ký tự đầu của sha256(api_key). Tuyệt đối không log hay lưu key thật."""
    if not api_key:
        return "none"
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()[:8]


def compute_cue_cache_key(
    model: str,
    ja: str,
    en: Optional[str] = None,
    glossary_hash: str = "",
    cast_sheet_hash: str = "",
    prompt_version: str = "v2.0"
) -> str:
    """Generates a SHA-256 cache key for an individual dialogue cue keyed by model, text, glossary, cast sheet, and prompt."""
    canonical = json.dumps({
        "model": model,
        "ja": ja.strip(),
        "en": (en or "").strip(),
        "glossary": glossary_hash,
        "cast_sheet": cast_sheet_hash,
        "prompt_version": prompt_version
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
                    pt_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT,
                    backend TEXT,
                    cues_count INTEGER,
                    status TEXT,
                    error_type TEXT,
                    quota_id TEXT,
                    quota_value TEXT,
                    pt_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_exhaustion (
                    model TEXT PRIMARY KEY,
                    exhausted_until_pt TEXT,
                    quota_id TEXT,
                    quota_value TEXT,
                    reason TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS key_exhaustion (
                    key_id TEXT,
                    model TEXT,
                    pt_date TEXT,
                    exhausted_until_pt TEXT,
                    quota_id TEXT,
                    quota_value TEXT,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (key_id, model, pt_date)
                )
            """)
            # Ensure columns exist if table was created earlier
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(translations)")
            cols_tr = [r[1] for r in cursor.fetchall()]
            if "pt_date" not in cols_tr:
                conn.execute("ALTER TABLE translations ADD COLUMN pt_date TEXT")

            cursor.execute("PRAGMA table_info(api_requests)")
            cols = [r[1] for r in cursor.fetchall()]
            if "pt_date" not in cols:
                conn.execute("ALTER TABLE api_requests ADD COLUMN pt_date TEXT")
            if "quota_id" not in cols:
                conn.execute("ALTER TABLE api_requests ADD COLUMN quota_id TEXT")
            if "quota_value" not in cols:
                conn.execute("ALTER TABLE api_requests ADD COLUMN quota_value TEXT")
            if "error_type" not in cols:
                conn.execute("ALTER TABLE api_requests ADD COLUMN error_type TEXT")
            if "key_id" not in cols:
                conn.execute("ALTER TABLE api_requests ADD COLUMN key_id TEXT")

            # Backfill pt_date bằng ZoneInfo("America/Los_Angeles") cho các bản ghi cũ
            today_pt = get_today_pt_str()
            cursor.execute("SELECT id, created_at FROM api_requests WHERE pt_date IS NULL")
            for r_id, c_at in cursor.fetchall():
                p_date = today_pt
                if c_at:
                    try:
                        dt = datetime.datetime.fromisoformat(c_at.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=datetime.timezone.utc)
                        p_date = dt.astimezone(PT_TZ).strftime("%Y-%m-%d")
                    except Exception:
                        p_date = today_pt
                conn.execute("UPDATE api_requests SET pt_date = ? WHERE id = ?", (p_date, r_id))

            cursor.execute("SELECT cache_key, created_at FROM translations WHERE pt_date IS NULL")
            for c_key, c_at in cursor.fetchall():
                p_date = today_pt
                if c_at:
                    try:
                        dt = datetime.datetime.fromisoformat(c_at.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=datetime.timezone.utc)
                        p_date = dt.astimezone(PT_TZ).strftime("%Y-%m-%d")
                    except Exception:
                        p_date = today_pt
                conn.execute("UPDATE translations SET pt_date = ? WHERE cache_key = ?", (p_date, c_key))

            conn.commit()

    def get_cue(self, cache_key: str) -> Optional[str]:
        """Retrieves cached translation for a cue. Bỏ qua nếu là chuỗi chưa dịch hoặc dry-run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vi FROM translations WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
            if row and row[0]:
                val = row[0].strip()
                if val.lower() in UNTRANSLATED_STRINGS or val.startswith("[DRY-RUN") or "[dry-run]" in val.lower():
                    return None
                return val
        return None

    def set_cue(self, cache_key: str, backend: str, model: str, vi: str):
        """Saves translated cue into cache kèm ngày Thái Bình Dương (PT). Tuyệt đối KHÔNG cache chuỗi [Chưa dịch] hoặc [DRY-RUN]."""
        if not vi or vi.strip().lower() in UNTRANSLATED_STRINGS or vi.startswith("[DRY-RUN") or "[dry-run]" in vi.lower():
            return

        pt_date = get_today_pt_str()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO translations (cache_key, backend, model, vi, pt_date)
                VALUES (?, ?, ?, ?, ?)
            """, (cache_key, backend, model, vi, pt_date))
            conn.commit()

    def log_request(
        self,
        model: str,
        backend: str,
        cues_count: int,
        status: str = "success",
        error_type: str = "",
        quota_id: str = "",
        quota_value: str = "",
        key_id: str = ""
    ):
        """Ghi nhận một lời gọi API vào cơ sở dữ liệu để theo dõi hạn mức theo ngày Thái Bình Dương."""
        pt_date = get_today_pt_str()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_requests (model, backend, cues_count, status, error_type, quota_id, quota_value, pt_date, key_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (model, backend, cues_count, status, error_type, quota_id, str(quota_value), pt_date, key_id))
            conn.commit()

    def mark_key_model_exhausted(
        self,
        key_id: str,
        model: str,
        quota_id: str = "PerDay",
        quota_value: str = "",
        reason: str = ""
    ):
        """Đánh dấu (key_id, model) cạn hạn mức đến 00:00 PT ngày tiếp theo."""
        today_pt = get_today_pt_str()
        next_pt, _ = get_next_pt_reset()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO key_exhaustion (key_id, model, pt_date, exhausted_until_pt, quota_id, quota_value, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (key_id, model, today_pt, next_pt.isoformat(), quota_id, str(quota_value), reason))
            conn.commit()

    def is_key_model_exhausted(self, key_id: str, model: str) -> bool:
        """Kiểm tra xem (key_id, model) có bị cạn hạn mức trong ngày PT hiện tại không."""
        today_pt = get_today_pt_str()
        now_pt = get_now_pt()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT exhausted_until_pt FROM key_exhaustion
                WHERE key_id = ? AND model = ? AND pt_date = ?
            """, (key_id, model, today_pt))
            row = cursor.fetchone()
            if not row or not row[0]:
                return False
            try:
                exhausted_until = datetime.datetime.fromisoformat(row[0])
                if exhausted_until.tzinfo is None:
                    exhausted_until = exhausted_until.replace(tzinfo=PT_TZ)
                return now_pt < exhausted_until
            except Exception:
                return False

    def mark_model_exhausted(
        self,
        model: str,
        exhausted_until_pt: Optional[datetime.datetime] = None,
        quota_id: str = "",
        quota_value: str = "",
        reason: str = ""
    ):
        """Đánh dấu model đã hết hạn mức đến 00:00 ngày Thái Bình Dương (PT)."""
        if exhausted_until_pt is None:
            next_pt, _ = get_next_pt_reset()
            exhausted_until_pt = next_pt
        iso_val = exhausted_until_pt.isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_exhaustion (model, exhausted_until_pt, quota_id, quota_value, reason, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (model, iso_val, quota_id, str(quota_value), reason))
            conn.commit()

    def is_model_exhausted(self, model: str) -> bool:
        """Kiểm tra xem model có đang bị đánh dấu hết hạn mức trong ngày Thái Bình Dương hiện tại không."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT exhausted_until_pt FROM model_exhaustion WHERE model = ?", (model,))
            row = cursor.fetchone()
            if not row or not row[0]:
                return False
            try:
                exhausted_until = datetime.datetime.fromisoformat(row[0])
                now_pt = get_now_pt()
                if now_pt < exhausted_until:
                    return True
                else:
                    # Đã qua 00:00 PT, tự động dọn bản ghi
                    conn.execute("DELETE FROM model_exhaustion WHERE model = ?", (model,))
                    conn.commit()
                    return False
            except Exception:
                return False

    def get_model_exhaustion_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết về việc hết hạn mức của model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT exhausted_until_pt, quota_id, quota_value, reason FROM model_exhaustion WHERE model = ?", (model,))
            row = cursor.fetchone()
            if row:
                return {
                    "exhausted_until_pt": row[0],
                    "quota_id": row[1],
                    "quota_value": row[2],
                    "reason": row[3],
                }
        return None

    def get_cues_today_pt(self) -> Dict[str, int]:
        """
        Đếm số cue đã dịch theo từng model trong ngày hiện tại theo giờ Thái Bình Dương (PT).
        Sử dụng cột pt_date được tính qua zoneinfo.ZoneInfo('America/Los_Angeles') - KHÔNG dùng offset cố định -7.
        """
        pt_date = get_today_pt_str()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model, COUNT(*)
                FROM translations
                WHERE pt_date = ?
                GROUP BY model
            """, (pt_date,))
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_requests_today_pt(self) -> Dict[str, int]:
        """
        Đếm số lượt gọi request thực tế (loại trừ dry-run) theo từng model trong ngày hiện tại theo giờ Thái Bình Dương (PT).
        Sử dụng cột pt_date được tính qua zoneinfo.ZoneInfo('America/Los_Angeles') - KHÔNG dùng offset cố định -7.
        """
        pt_date = get_today_pt_str()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model, COUNT(*)
                FROM api_requests
                WHERE pt_date = ?
                  AND status != 'dry_run'
                GROUP BY model
            """, (pt_date,))
            return {row[0]: row[1] for row in cursor.fetchall()}

    def count(self) -> int:
        """Returns total cached items."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM translations")
            row = cursor.fetchone()
            return row[0] if row else 0


def execute_and_log_api_request(
    fn,
    model: str,
    backend: str = "gemini",
    cues_count: int = 0,
    request_type: str = "translation",
    cache: Optional[TranslationCache] = None,
    key_id: str = ""
):
    """
    HÀM BỌC DUY NHẤT (Single Unified Wrapper) cho TẤT CẢ các lời gọi API.
    Mọi request (kể cả test/benchmark/ping/cast_sheet/translation) đều được
    ghi nhận vào bảng `api_requests` với ngày Thái Bình Dương (PT) chuẩn xác.
    """
    c = cache or TranslationCache()
    try:
        res = fn()
        c.log_request(
            model=model,
            backend=backend,
            cues_count=cues_count,
            status="success",
            error_type="",
            quota_id="",
            quota_value="",
            key_id=key_id
        )
        return res
    except Exception as e:
        quota_id = None
        quota_value = None
        try:
            from subtrans.throttle import parse_429_details
            quota_id, quota_value, _ = parse_429_details(e)
        except Exception:
            pass

        err_str = str(e).lower()
        if "perday" in (quota_id or "").lower() or "requestsperday" in (quota_id or "").lower():
            status_str = "exhausted_429_perday"
            error_type = "429_PerDay"
        elif "429" in err_str or "resource_exhausted" in err_str:
            status_str = "error_429_perminute"
            error_type = "429_PerMinute"
        elif "503" in err_str or "unavailable" in err_str:
            status_str = "error_503_unavailable"
            error_type = "503_Unavailable"
        elif "timeout" in err_str or "timed out" in err_str:
            status_str = "error_timeout"
            error_type = "Timeout"
        else:
            status_str = "error"
            error_type = type(e).__name__

        c.log_request(
            model=model,
            backend=backend,
            cues_count=cues_count,
            status=status_str,
            error_type=error_type,
            quota_id=quota_id or "",
            quota_value=str(quota_value or ""),
            key_id=key_id
        )
        raise

