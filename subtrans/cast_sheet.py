# -*- coding: utf-8 -*-
"""
subtrans/cast_sheet.py
Stage A: Cast Sheet generation and persistence.
Analyzes episode dialogue once per episode to standardize character names,
nicknames (e.g. Emilia-tan, Barusu), register (archaic/divine vs modern casual),
and speaker-to-listener pronoun consistency.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import yaml
from google import genai
from google.genai import types

from subtrans.models import Cue, Character, CharacterSpeaksTo, Glossary
from subtrans.config import get_gemini_api_key
from subtrans.gemini_backend import get_thinking_config
from subtrans.cache import execute_and_log_api_request

logger = logging.getLogger("subtrans.cast_sheet")

CAST_PROMPT_TEMPLATE = """
You are an expert anime subtitle director. Analyze the following anime dialogue lines and determine the cast members, their character relationships, registers, and Vietnamese pronouns.

Rules:
1. Names: Keep Romaji spelling exactly as in English subtitles (e.g. Subaru, Emilia, Ram, Rem, Shaula, Reid, Volcanica, Farsale).
2. Nicknames: Keep deliberate nicknames intact (e.g. "Emilia-tan", "Barusu").
3. Register:
   - Archaic, divine, dragons, royalty, powerful elders: ta / ngươi / ngài / bệ hạ / tiểu thư / chàng.
   - Modern casual, peers, teens: tôi / cậu / tớ / mình / em / anh / mày / tao.
4. Output MUST be valid JSON matching this schema:
{
  "characters": [
    {
      "name_ja": "Japanese name if known or empty",
      "name_vi": "Romaji name",
      "gender": "male | female | other",
      "register": "archaic | divine | noble | modern casual",
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
    Quét nhanh các câu thoại để tìm xem có tên nhân vật mới xuất hiện trong tập không.
    """
    existing_names = {c.name_vi.lower() for c in existing_characters}
    for c in existing_characters:
        if c.name_ja:
            existing_names.add(c.name_ja.lower())

    potential_new = set()
    # Quét các từ viết hoa trong lời thoại tiếng Anh
    for cue in cues:
        if cue.is_skipped or not cue.en:
            continue
        # Tìm danh từ riêng viết hoa có thể là tên nhân vật
        words = re.findall(r'\b[A-Z][a-z]{2,15}\b', cue.en)
        for w in words:
            w_lower = w.lower()
            if w_lower in {"the", "and", "but", "you", "they", "this", "that", "what", "where", "when", "why", "how", "well", "yes", "hey", "look", "wait", "come", "here", "just"}:
                continue
            if w_lower not in existing_names:
                # Nếu từ xuất hiện nhiều lần hoặc đi kèm danh xưng/gọi tên
                if re.search(r'\b(lord|lady|master|mr\.|ms\.|miss|sir|brother|sister)\s+' + re.escape(w) + r'\b', cue.en, re.IGNORECASE):
                    potential_new.add(w)
    return potential_new


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
) -> Optional[Glossary]:
    """
    Stage A: Quản lý Cast Sheet theo bộ (<series>.cast.yaml).
    - Tự động nạp <series>.cast.yaml nếu đã có.
    - Chỉ gọi API khi chưa có Cast Sheet hoặc khi phát hiện nhân vật mới.
    - Tiết kiệm triệt để số lượng request (mục tiêu <= 5 request/tập).
    """
    series_name = extract_series_name(source_name)
    series_cast_file = get_series_cast_path(series_name, output_dir)

    # 1. Nếu đã có Cast Sheet theo bộ, kiểm tra xem có cần cập nhật không
    if series_cast_file.exists() and not force_update:
        try:
            with open(series_cast_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
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
                        speaks_to=speaks,
                    )
                )

            if characters:
                new_names = check_for_new_characters(cues, characters)
                if not new_names:
                    logger.info(
                        f"✅ Cast sheet của bộ '{series_name}' ({len(characters)} nhân vật) đã sẵn sàng. "
                        f"Không có nhân vật mới -> Tái sử dụng (0 requests API)."
                    )
                    # Ghi bản sao cho tập hiện tại nếu cần
                    if episode_cast_path and not episode_cast_path.exists():
                        try:
                            episode_cast_path.write_bytes(series_cast_file.read_bytes())
                        except Exception:
                            pass
                    return Glossary(characters=characters)
                else:
                    logger.info(f"Phát hiện nhân vật mới tiềm năng ({', '.join(new_names)}) trong '{series_name}'. Đang cập nhật Cast Sheet...")
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
        series_cast_file.parent.mkdir(parents=True, exist_ok=True)
        with open(series_cast_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "meta": {"schema": 2, "series": series_name, "description": "Auto-generated Series Cast Sheet (Dry-Run)"},
                    "characters": [c.model_dump() for c in mock_chars],
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )
        if episode_cast_path:
            episode_cast_path.parent.mkdir(parents=True, exist_ok=True)
            with open(episode_cast_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "meta": {"schema": 2, "series": series_name, "description": "Auto-generated Episode Cast Sheet (Dry-Run)"},
                        "characters": [c.model_dump() for c in mock_chars],
                    },
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                )
        return cast_glossary

    # 3. Tạo mới qua Gemini API
    res_glossary = generate_cast_sheet(
        cues=cues,
        output_path=series_cast_file,
        gemini_model=gemini_model,
        fallback_models=fallback_models,
        timeout_ms=timeout_ms,
    )
    if res_glossary and episode_cast_path and series_cast_file.exists():
        try:
            episode_cast_path.write_bytes(series_cast_file.read_bytes())
        except Exception:
            pass
    return res_glossary


def generate_cast_sheet(
    cues: List[Cue],
    output_path: Path,
    gemini_model: str = "gemini-3.8-flash",
    fallback_models: Optional[List[str]] = None,
    timeout_ms: int = 45000,
) -> Optional[Glossary]:
    """
    Stage A: Discovers characters and pronoun tables across the episode.
    Persists cast sheet to output_path and returns a Glossary object.
    Includes request timeout and candidate model fallback if initial model hits 503/429.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        logger.info("GEMINI_API_KEY not found; skipping automatic Cast Sheet generation.")
        return None

    # Sample dialogue cues: up to 60 distinct dialogue cues across the episode
    dialogue_cues = [c for c in cues if not c.is_skipped and (c.ja or c.en)]
    step = max(1, len(dialogue_cues) // 60)
    sampled = dialogue_cues[::step][:60]

    sample_lines = []
    for c in sampled:
        en_part = f" | EN: {c.en}" if c.en else ""
        sample_lines.append(f"[{c.id}] JA: {c.ja}{en_part}")

    prompt = CAST_PROMPT_TEMPLATE + "\n".join(sample_lines)

    candidate_models = [gemini_model]
    if fallback_models:
        for m in fallback_models:
            if m not in candidate_models:
                candidate_models.append(m)

    http_options = types.HttpOptions(timeout=int(timeout_ms))
    client = genai.Client(api_key=api_key, http_options=http_options)

    res = None
    last_err = None
    for model_name in candidate_models:
        try:
            cfg_args = {
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "http_options": http_options,
            }
            thinking_cfg = get_thinking_config(model_name)
            if thinking_cfg:
                cfg_args["thinking_config"] = thinking_cfg

            def _send():
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg_args),
                )

            res = execute_and_log_api_request(
                _send,
                model=model_name,
                backend="gemini",
                cues_count=0,
                request_type="cast_sheet"
            )
            if res and res.text:
                break
        except Exception as e:
            last_err = e
            logger.warning(f"Stage A Cast Sheet attempt with '{model_name}' failed: {e}. Trying fallback...")

    if not res or not res.text:
        logger.warning(f"Failed to generate Stage A Cast Sheet across all candidate models: {last_err}")
        return None

    try:
        data = json.loads(res.text.strip())
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
                    speaks_to=speaks,
                )
            )

        cast_glossary = Glossary(characters=characters)

        # Persist to YAML
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "meta": {"schema": 2, "description": "Auto-generated Series Cast Sheet"},
                    "characters": [c.model_dump() for c in characters],
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )

        return cast_glossary

    except Exception as e:
        logger.warning(f"Failed to generate Stage A Cast Sheet: {e}")
        return None

