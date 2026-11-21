"""Stage A episode cast sheet generator for character names, registers, and pronouns."""

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from collections import Counter
import yaml
from pydantic import BaseModel
from google import genai
from google.genai import types

from subtrans.models import Cue, Character, CharacterSpeaksTo, Glossary
from subtrans.config import get_gemini_api_key, get_gemini_api_keys
from subtrans.gemini_backend import (
    get_thinking_config,
    get_anime_safety_settings,
    extract_genai_response_text,
)
from subtrans.ja_morph_tagger import strip_ja_furigana, has_cjk_characters
from subtrans.cache import execute_and_log_api_request, TranslationCache, compute_key_id
from subtrans.throttle import (
    call_with_retry,
    DailyQuotaExhaustedError,
    MinuteRateLimitError,
    QuotaExhaustedError,
    RateLimiter,
)

logger = logging.getLogger("subtrans.cast_sheet")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically writes raw bytes via same-directory temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp_path, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _atomic_write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Atomically writes YAML payload via same-directory temp file + os.replace."""
    dumped = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    _atomic_write_bytes(path, dumped.encode("utf-8"))


class CastSpeaksToItem(BaseModel):
    listener: str = "all"
    self: str = "tôi"
    other: str = "cậu"
    note: Optional[str] = None


class CastCharacterItem(BaseModel):
    name_ja: str = ""
    name_vi: str
    gender: Optional[str] = None
    speaking_style: Optional[str] = None
    speaks_to: List[CastSpeaksToItem] = []


class CastSheetResponse(BaseModel):
    characters: List[CastCharacterItem] = []


_COMMON_EN_STOPWORDS = {
    "the", "and", "but", "you", "your", "yours", "they", "them", "their", "this", "that",
    "these", "those", "what", "where", "when", "why", "how", "who", "whom", "well", "yes",
    "yeah", "nah", "nope", "hey", "look", "wait", "stop", "come", "here", "there", "just",
    "now", "then", "still", "even", "only", "really", "maybe", "perhaps", "please", "thank",
    "thanks", "sorry", "excuse", "hello", "goodbye", "right", "okay", "sure", "fine", "good",
    "great", "god", "gods", "heaven", "hell", "damn", "cursed", "guild", "magic", "mana",
    "skill", "skills", "sword", "shield", "king", "queen", "prince", "princess", "lord",
    "lady", "master", "sir", "miss", "brother", "sister", "father", "mother", "dad", "mom",
    "uncle", "aunt", "captain", "general", "doctor", "professor", "teacher", "everyone",
    "somebody", "someone", "anyone", "nobody", "nothing", "everything", "something",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "tomorrow", "yesterday", "morning", "night", "evening",
}

CAST_PROMPT_TEMPLATE = """
You are an expert anime subtitle director. Analyze the following anime dialogue lines and determine the cast members, their character relationships, speaking styles, and Vietnamese pronouns.

Rules:
1. Names: ALWAYS keep Romaji spelling exactly as in English subtitles (e.g. Subaru, Emilia, Ram, Rem, Reirin, Keigetsu, Kou, Shu, Leelee, Gayou).
   CRITICAL: NEVER translate character names or clan/family names into Sino-Vietnamese (Hán-Việt) or Vietnamese words (DO NOT use Linh Lâm, Huệ Nguyệt, Tuệ Nguyệt, Chu, Hoàng, Thư, Lợi Lợi, etc.). Always use exact Romaji matching the English subtitles!
2. Nicknames: Keep deliberate nicknames intact (e.g. "Emilia-tan", "Barusu").
3. Speaking Style / Register:
   - Archaic, divine, dragons, royalty, powerful elders: ta / ngươi / ngài / bệ hạ / tiểu thư / chàng.
   - Modern casual, peers, teens: tôi / cậu / tớ / mình / em / anh / mày / tao.
4. Output MUST be valid JSON matching this schema:
{
  "characters": [
    {
      "name_ja": "Japanese name if known or empty",
      "name_vi": "Romaji name",
      "gender": "male | female | other",
      "speaking_style": "archaic | divine | noble | modern casual",
      "speaks_to": [
        {
          "listener": "Target character name or 'all'",
          "self": "tôi | ta | tớ | em ...",
          "other": "cậu | ngươi | ngài | anh ...",
          "note": "Short rationale"
        }
      ]
    }
  ]
}

Dialogue samples from the episode:
"""


def extract_series_name(name_or_path: str) -> str:
    """Trích xuất tên series/bộ phim từ tên file video hoặc subtitle."""
    name = Path(name_or_path).stem
    s = re.sub(r'\.(mkv|mp4|ass|srt)$', '', name, flags=re.IGNORECASE)
    # Xóa release group tags như [SubsPlease], [Erai-raws], [HorribleSubs]
    s = re.sub(r'^\[.*?\]\s*', '', s)
    # Xóa checksums và tags độ phân giải: [73CF848B], (1080p), (4K)
    s = re.sub(r'\[[A-Fa-f0-9]{8}\]', '', s)
    s = re.sub(r'\((1080p|720p|4k|2160p|bdrip|web-dl)\)', '', s, flags=re.IGNORECASE)
    # Xóa số tập: ' - 14', ' E14', ' Ep.14', ' 14'
    s = re.sub(r'\s+-\s+\d+.*$', '', s)
    s = re.sub(r'\s+S\d+E\d+.*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+E\d+.*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+Episode\s+\d+.*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\[.*?\]', '', s)
    s = re.sub(r'\(.*?\)', '', s)
    s = s.strip(' -_')
    return s or name


def get_series_cast_path(series_name: str, out_dir: Path) -> Path:
    """Trả về đường dẫn file cast sheet của toàn bộ: out/<series>.cast.yaml (hoặc out/audit/<series>.cast.yaml)."""
    norm_series = re.sub(r'[\s/\\:]+', '_', series_name.strip())
    audit_file = out_dir / "audit" / f"{norm_series}.cast.yaml"
    if audit_file.exists():
        return audit_file
    return out_dir / f"{norm_series}.cast.yaml"


def check_for_new_characters(cues: List[Cue], existing_characters: List[Character]) -> Set[str]:
    """
    Quét đa tín hiệu trên lời thoại để phát hiện nhân vật mới xuất hiện trong tập:
    1. Nhãn người nói từ phụ đề CC tiếng Nhật (Cue.speaker đã khử Furigana, xuất hiện >= 2 lần).
    2. Tên đi kèm danh xưng tiếng Anh (Lord/Lady/Master/Sir/Miss/... <Name>).
    3. Tên gọi trực tiếp ở đầu/cuối câu có dấu phẩy vocative xuất hiện >= 2 lần.
    4. Danh từ riêng viết hoa giữa câu xuất hiện lặp lại >= 3 lần (hỗ trợ cả chế độ EN-only fallback).
    """
    existing_names: Set[str] = set()
    for c in existing_characters:
        if c.name_vi:
            existing_names.add(c.name_vi.strip().lower())
            for tok in re.findall(r"[A-Za-z]{3,}", c.name_vi):
                existing_names.add(tok.lower())
        if c.name_ja:
            existing_names.add(c.name_ja.strip().lower())
            existing_names.add(strip_ja_furigana(c.name_ja).strip().lower())

    potential_new: Set[str] = set()
    cc_speaker_counts: Counter = Counter()
    vocative_counts: Counter = Counter()
    mid_sentence_counts: Counter = Counter()

    honorific_re = re.compile(
        r"\b(?:lord|lady|master|mr\.|ms\.|miss|sir|brother|sister|prince|princess|king|queen|captain|general|doctor)\s+([A-Z][a-z]{2,15})\b",
        flags=re.IGNORECASE,
    )
    vocative_lead_re = re.compile(r"^\s*([A-Z][a-z]{2,15}),\s+")
    vocative_tail_re = re.compile(r",\s*([A-Z][a-z]{2,15})[!?.]+\s*$")
    mid_proper_re = re.compile(r"[a-z,;:]\s+([A-Z][a-z]{2,15})\b")

    for cue in cues:
        if cue.is_skipped:
            continue

        # 1. Tín hiệu nhãn người nói từ CC tiếng Nhật
        raw_spk = (cue.speaker or "").strip()
        if raw_spk and raw_spk.lower() != "unknown":
            clean_spk = strip_ja_furigana(raw_spk).strip()
            if clean_spk and clean_spk.lower() not in existing_names:
                cc_speaker_counts[clean_spk] += 1

        # 2. Tín hiệu tiếng Anh (từ cue.en hoặc cue.ja khi chạy EN-only fallback)
        en_source = cue.en
        if not en_source and cue.ja and not has_cjk_characters(cue.ja):
            en_source = cue.ja
        if not en_source:
            continue

        for m in honorific_re.finditer(en_source):
            cand = m.group(1)
            # Chỉ nhận từ thực sự viết hoa chữ cái đầu trong văn bản gốc
            if cand[0].isupper() and cand.lower() not in _COMMON_EN_STOPWORDS and cand.lower() not in existing_names:
                potential_new.add(cand)

        for rx in (vocative_lead_re, vocative_tail_re):
            m = rx.search(en_source)
            if m:
                cand = m.group(1)
                if cand.lower() not in _COMMON_EN_STOPWORDS and cand.lower() not in existing_names:
                    vocative_counts[cand] += 1

        for m in mid_proper_re.finditer(en_source):
            cand = m.group(1)
            if cand.lower() not in _COMMON_EN_STOPWORDS and cand.lower() not in existing_names:
                mid_sentence_counts[cand] += 1

    for spk, cnt in cc_speaker_counts.items():
        if cnt >= 2:
            potential_new.add(spk)
    for cand, cnt in vocative_counts.items():
        if cnt >= 2:
            potential_new.add(cand)
    for cand, cnt in mid_sentence_counts.items():
        if cnt >= 3:
            potential_new.add(cand)

    return potential_new


def _merge_cast_characters(existing_chars: List[Character], new_chars: List[Character]) -> List[Character]:
    """Additively merge newly discovered characters into an existing series cast list."""
    merged: List[Character] = [c.model_copy(deep=True) for c in existing_chars]
    index_by_key: Dict[str, int] = {}
    for idx, ch in enumerate(merged):
        if ch.name_vi:
            index_by_key[ch.name_vi.strip().lower()] = idx
        if ch.name_ja:
            index_by_key[strip_ja_furigana(ch.name_ja).strip().lower()] = idx

    for nc in new_chars:
        keys = []
        if nc.name_vi:
            keys.append(nc.name_vi.strip().lower())
        if nc.name_ja:
            keys.append(strip_ja_furigana(nc.name_ja).strip().lower())
        match_idx = next((index_by_key[k] for k in keys if k and k in index_by_key), None)
        if match_idx is None:
            new_idx = len(merged)
            merged.append(nc.model_copy(deep=True))
            for k in keys:
                if k:
                    index_by_key[k] = new_idx
        else:
            existing = merged[match_idx]
            if not existing.name_ja and nc.name_ja:
                existing.name_ja = nc.name_ja
            if not existing.gender and nc.gender:
                existing.gender = nc.gender
            if not existing.speaking_style and nc.speaking_style:
                existing.speaking_style = nc.speaking_style
            seen_listeners = {s.listener.strip().lower() for s in existing.speaks_to if s.listener}
            for sp in nc.speaks_to:
                l_key = (sp.listener or "").strip().lower()
                if l_key and l_key not in seen_listeners:
                    existing.speaks_to.append(sp.model_copy(deep=True))
                    seen_listeners.add(l_key)
    return merged


def load_or_generate_series_cast_sheet(
    cues: List[Cue],
    source_name: str,
    output_dir: Path,
    episode_cast_path: Optional[Path] = None,
    gemini_model: str = "gemini-3.8-flash",
    fallback_models: Optional[List[str]] = None,
    timeout_ms: int = 45000,
    dry_run: bool = False,
    force_update: bool = False,
    cache: Optional[Any] = None,
) -> Optional[Glossary]:
    """
    Stage A: Quản lý Cast Sheet theo bộ (<series>.cast.yaml).
    - Tự động nạp <series>.cast.yaml nếu đã có.
    - Chỉ gọi API khi chưa có Cast Sheet hoặc khi phát hiện nhân vật mới.
    - Tiết kiệm triệt để số lượng request (mục tiêu <= 5 request/tập).
    """
    series_name = extract_series_name(source_name)
    series_cast_file = get_series_cast_path(series_name, output_dir)
    reusable_existing_chars: List[Character] = []

    # 1. Nếu đã có Cast Sheet theo bộ, kiểm tra xem có cần cập nhật không
    if series_cast_file.exists() and not force_update:
        try:
            with open(series_cast_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            meta_desc = str(data.get("meta", {}).get("description", ""))
            is_dry_artifact = "dry-run" in meta_desc.lower()
            chars_data = data.get("characters", [])
            characters = []
            for cd in chars_data:
                speaks = [
                    CharacterSpeaksTo(
                        listener=s.get("listener", "all"),
                        self=s.get("self", "tôi"),
                        other=s.get("other", "cậu"),
                        note=s.get("note"),
                    )
                    for s in cd.get("speaks_to", [])
                ]
                characters.append(
                    Character(
                        name_ja=cd.get("name_ja", ""),
                        name_vi=cd.get("name_vi", ""),
                        aliases=[str(a) for a in cd.get("aliases", []) if a],
                        gender=cd.get("gender"),
                        speaking_style=cd.get("register") or cd.get("speaking_style"),
                        note=cd.get("note"),
                        speaks_to=speaks,
                    )
                )

            if characters:
                # Nếu đang chạy dry-run và đã có cast sheet thật trên đĩa, tuyệt đối không ghi đè cast thật!
                if dry_run and not is_dry_artifact:
                    return Glossary(characters=characters)

                # Nếu đang chạy thật nhưng file hiện có là mẫu dry-run (Rudeus/Sylphiette),
                # chỉ tái sử dụng khi lời thoại thực sự khớp nhân vật; tránh ô nhiễm bộ anime khác.
                can_reuse = True
                if not dry_run and is_dry_artifact:
                    can_reuse = any(
                        (c.name_ja and c.name_ja in (cue.ja or ""))
                        or (c.name_vi and c.name_vi.lower() in (cue.en or "").lower())
                        for c in characters
                        for cue in cues
                        if not cue.is_skipped
                    )

                if can_reuse:
                    new_names = check_for_new_characters(cues, characters)
                    if not new_names:
                        logger.info(
                            f"✅ Cast sheet của bộ '{series_name}' ({len(characters)} nhân vật) đã sẵn sàng. "
                            f"Không có nhân vật mới -> Tái sử dụng (0 requests API)."
                        )
                        # Ghi bản sao nguyên tử cho tập hiện tại nếu cần
                        if episode_cast_path and not episode_cast_path.exists():
                            try:
                                _atomic_write_bytes(episode_cast_path, series_cast_file.read_bytes())
                            except Exception:
                                pass
                        return Glossary(characters=characters)
                    else:
                        reusable_existing_chars = list(characters)
                        logger.info(f"Phát hiện nhân vật mới tiềm năng ({', '.join(new_names)}) trong '{series_name}'. Đang cập nhật Cast Sheet...")
                else:
                    logger.info(f"Bỏ qua Cast Sheet mẫu Dry-Run cũ của '{series_name}' để sinh Cast Sheet thật từ lời thoại.")
        except Exception as e:
            logger.warning(f"Lỗi khi đọc {series_cast_file}: {e}. Đang tạo lại...")

    # 2. Xử lý trường hợp Dry-Run: Không bao giờ gọi API
    if dry_run:
        logger.info(f"[DRY-RUN] Sinh Cast Sheet mẫu cho bộ '{series_name}' mà không gọi API...")
        mock_chars = [
            Character(name_ja="ルーデウス", name_vi="Rudeus", gender="male", speaks_to=[
                CharacterSpeaksTo(listener="all", self="tôi", other="cậu")
            ]),
            Character(name_ja="シルフィエット", name_vi="Sylphiette", gender="female", speaks_to=[
                CharacterSpeaksTo(listener="Rudeus", self="em", other="anh")
            ]),
        ]
        cast_glossary = Glossary(characters=mock_chars)
        _atomic_write_yaml(
            series_cast_file,
            {
                "meta": {"schema": 2, "series": series_name, "description": "Auto-generated Series Cast Sheet (Dry-Run)"},
                "characters": [c.model_dump(exclude_none=True) for c in mock_chars],
            },
        )
        if episode_cast_path:
            _atomic_write_yaml(
                episode_cast_path,
                {
                    "meta": {"schema": 2, "series": series_name, "description": "Auto-generated Episode Cast Sheet (Dry-Run)"},
                    "characters": [c.model_dump(exclude_none=True) for c in mock_chars],
                },
            )
        return cast_glossary

    # 3. Tạo mới qua Gemini API
    res_glossary = generate_cast_sheet(
        cues=cues,
        output_path=series_cast_file,
        gemini_model=gemini_model,
        fallback_models=fallback_models,
        timeout_ms=timeout_ms,
        cache=cache,
    )
    if res_glossary and reusable_existing_chars:
        merged_chars = _merge_cast_characters(reusable_existing_chars, list(res_glossary.characters))
        res_glossary = Glossary(characters=merged_chars)
        _atomic_write_yaml(
            series_cast_file,
            {
                "meta": {"schema": 2, "series": series_name, "description": "Auto-generated Series Cast Sheet"},
                "characters": [c.model_dump(exclude_none=True) for c in merged_chars],
            },
        )
    if res_glossary and episode_cast_path and series_cast_file.exists():
        try:
            _atomic_write_bytes(episode_cast_path, series_cast_file.read_bytes())
        except Exception:
            pass
    return res_glossary


def generate_cast_sheet(
    cues: List[Cue],
    output_path: Path,
    gemini_model: str = "gemini-3.8-flash",
    fallback_models: Optional[List[str]] = None,
    timeout_ms: int = 45000,
    cache: Optional[Any] = None,
    api_keys: Optional[List[str]] = None,
) -> Optional[Glossary]:
    """
    Stage A: Discovers characters and pronoun tables across the episode.
    Persists cast sheet to output_path and returns a Glossary object.
    Supports 3-key rotation on 429 PerDay, Pydantic structured output, and anime safety settings.
    """
    keys = list(api_keys) if api_keys is not None else get_gemini_api_keys()
    if not keys:
        single_key = get_gemini_api_key()
        if single_key:
            keys = [single_key]
    if not keys:
        logger.info("GEMINI_API_KEY not found; skipping automatic Cast Sheet generation.")
        return None

    key_ids = [compute_key_id(k) for k in keys]
    has_real_cache = isinstance(cache, TranslationCache)

    # Sample dialogue cues: up to 60 distinct dialogue cues across the episode
    dialogue_cues = [c for c in cues if not c.is_skipped and (c.ja or c.en)]
    step = max(1, len(dialogue_cues) // 60)
    sampled = dialogue_cues[::step][:60]

    sample_lines = []
    for c in sampled:
        spk_part = ""
        if c.speaker and c.speaker.strip().lower() != "unknown":
            spk_part = f" [SPEAKER_CC: {strip_ja_furigana(c.speaker.strip())}]"
        if not c.en and c.ja and not has_cjk_characters(c.ja):
            sample_lines.append(f"[{c.id}]{spk_part} EN: {c.ja}")
        else:
            ja_clean = strip_ja_furigana(c.ja) if c.ja else ""
            en_part = f" | EN: {c.en}" if c.en else ""
            sample_lines.append(f"[{c.id}]{spk_part} JA: {ja_clean}{en_part}")

    prompt = CAST_PROMPT_TEMPLATE + "\n".join(sample_lines)

    candidate_models = [gemini_model]
    if fallback_models:
        for m in fallback_models:
            if m not in candidate_models:
                candidate_models.append(m)

    http_options = types.HttpOptions(timeout=int(timeout_ms))
    clients = {idx: genai.Client(api_key=k, http_options=http_options) for idx, k in enumerate(keys)}

    raw_text: Optional[str] = None
    last_err: Optional[Exception] = None
    for model_name in candidate_models:
        rpm_limit = 12 if "lite" in model_name.lower() else 4
        rate_limiters = {kid: RateLimiter(max_rpm=rpm_limit) for kid in key_ids}
        cfg_args: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": CastSheetResponse,
            "safety_settings": get_anime_safety_settings(),
            "temperature": 0.1,
            "http_options": http_options,
        }
        thinking_cfg = get_thinking_config(model_name)
        if thinking_cfg:
            cfg_args["thinking_config"] = thinking_cfg

        # Vòng 1: xoay ngay sang key khác nếu gặp 429 PerMinute; vòng 2 (nếu cần): đợi cooldown của key
        max_rounds = 2 if len(key_ids) > 1 else 1
        for round_idx in range(max_rounds):
            should_try_next_model = False
            for key_idx, kid in enumerate(key_ids):
                if has_real_cache and len(keys) > 1 and cache.is_key_model_exhausted(kid, model_name):
                    continue

                limiter = rate_limiters[kid]
                client = clients[key_idx]
                rotate_on_rpm = len(key_ids) > 1 and round_idx == 0

                try:
                    def _single_api_call():
                        return execute_and_log_api_request(
                            lambda: client.models.generate_content(
                                model=model_name,
                                contents=prompt,
                                config=types.GenerateContentConfig(**cfg_args),
                            ),
                            model=model_name,
                            backend="gemini",
                            cues_count=0,
                            request_type="cast_sheet",
                            cache=cache,
                            key_id=kid,
                        )

                    res = call_with_retry(
                        fn=_single_api_call,
                        rate_limiter=limiter,
                        model_name=model_name,
                        max_retries=2,
                        cache=cache if has_real_cache else None,
                        key_id=kid,
                        all_key_ids=key_ids,
                        raise_on_per_minute=rotate_on_rpm,
                    )
                    raw_text = extract_genai_response_text(res)
                    if raw_text:
                        break
                except MinuteRateLimitError as mre:
                    last_err = mre
                    limiter.record_cooldown(mre.retry_delay)
                    logger.warning(
                        f"Stage A Cast Sheet key '{kid}' hit 429 PerMinute on '{model_name}'; rotating to next key..."
                    )
                    continue
                except DailyQuotaExhaustedError as dqe:
                    last_err = dqe
                    if has_real_cache:
                        cache.mark_key_model_exhausted(
                            key_id=kid,
                            model=model_name,
                            quota_id=dqe.quota_id or "PerDay",
                            quota_value=dqe.quota_value or "",
                            reason=f"Stage A 429 PerDay: {dqe.quota_id}",
                        )
                    logger.warning(
                        f"Stage A Cast Sheet key '{kid}' hit daily quota on '{model_name}'; rotating key/model..."
                    )
                    continue
                except QuotaExhaustedError as qe:
                    last_err = qe
                    logger.warning(
                        f"Stage A Cast Sheet key '{kid}' exhausted RPM retries on '{model_name}'; trying next key..."
                    )
                    continue
                except Exception as e:
                    last_err = e
                    logger.warning(f"Stage A Cast Sheet attempt with '{model_name}' (key={kid}) failed: {e}. Trying fallback...")
                    should_try_next_model = True
                    break

            if raw_text or should_try_next_model:
                break

        if raw_text:
            break

    if not raw_text:
        logger.warning(f"Failed to generate Stage A Cast Sheet across all candidate models: {last_err}")
        return None

    try:
        data = json.loads(raw_text)
        chars_data = data.get("characters", [])

        characters = []
        for cd in chars_data:
            speaks = [
                CharacterSpeaksTo(
                    listener=s.get("listener", "all"),
                    self=s.get("self", "tôi"),
                    other=s.get("other", "cậu"),
                    note=s.get("note"),
                )
                for s in cd.get("speaks_to", [])
            ]
            characters.append(
                Character(
                    name_ja=cd.get("name_ja", ""),
                    name_vi=cd.get("name_vi", ""),
                    gender=cd.get("gender"),
                    speaking_style=cd.get("register") or cd.get("speaking_style"),
                    note=cd.get("note"),
                    speaks_to=speaks,
                )
            )

        cast_glossary = Glossary(characters=characters)

        # Persist to YAML atomically
        _atomic_write_yaml(
            output_path,
            {
                "meta": {"schema": 2, "description": "Auto-generated Series Cast Sheet"},
                "characters": [c.model_dump(exclude_none=True) for c in characters],
            },
        )

        return cast_glossary

    except Exception as e:
        logger.warning(f"Failed to generate Stage A Cast Sheet: {e}")
        return None

