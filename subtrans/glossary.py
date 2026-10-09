# -*- coding: utf-8 -*-
"""
subtrans/glossary.py
Loader and merger for Schema v2 anime translation glossaries.
Supports style_rules, address_rules, terms filtering, and per-series overriding.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import re
import yaml
import json

from subtrans.models import (
    Glossary,
    GlossaryTerm,
    Character,
    CharacterSpeaksTo,
    AddressRules,
    PronounRule,
    RelationRule,
)

DEFAULT_BASE_GLOSSARY_PATH = Path(__file__).parent / "glossary.yaml"
GLOSSARIES_DIR = Path(__file__).resolve().parent.parent / "glossaries"


def find_series_glossary(series_or_filename: str, glossaries_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Tự động tìm file glossary riêng của bộ anime trong thư mục glossaries/.
    Ví dụ: 'Futsutsuka_Na_Akujo_Dewa_Gozaimasu_Ga' -> glossaries/glossary.futsutsuka.yaml
    """
    g_dir = glossaries_dir or GLOSSARIES_DIR
    if not g_dir.exists() or not series_or_filename:
        return None

    # Chuẩn hoá tên chuỗi đầu vào (bỏ ký tự đặc biệt, lowercase)
    norm_name = re.sub(r'[^a-zA-Z0-9]', '', series_or_filename.lower())
    if not norm_name:
        return None

    candidates = (
        list(g_dir.glob("glossary.*.yaml"))
        + list(g_dir.glob("glossary.*.yml"))
        + list(g_dir.glob("glossary.*.json"))
    )
    for cand in candidates:
        parts = cand.name.split(".")
        if len(parts) >= 3 and parts[0] == "glossary":
            key = parts[1].lower()
            norm_key = re.sub(r'[^a-zA-Z0-9]', '', key)
            if norm_key and norm_key in norm_name:
                return cand

    return None


def _parse_pronoun_list(raw_list: List[Dict[str, Any]]) -> List[PronounRule]:
    res = []
    for item in raw_list or []:
        if item.get("ja") and item.get("vi"):
            res.append(
                PronounRule(
                    ja=str(item["ja"]),
                    vi=str(item["vi"]),
                    alt=[str(a) for a in item.get("alt", [])],
                    note=item.get("note"),
                )
            )
    return res


def load_glossary(glossary_path: Optional[Path]) -> Optional[Glossary]:
    """Loads a Schema v2 YAML or JSON glossary file."""
    if not glossary_path or not glossary_path.exists():
        return None

    try:
        with open(glossary_path, "r", encoding="utf-8") as f:
            if glossary_path.suffix.lower() in (".yaml", ".yml"):
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)

        meta = data.get("meta", {})
        schema_ver = meta.get("schema", 2)
        style_rules = [str(r) for r in data.get("style_rules", [])]
        difficulty_markers = [str(m) for m in data.get("difficulty_markers", [])]

        # Parse address_rules
        raw_addr = data.get("address_rules", {})
        addr_rules = None
        if raw_addr:
            relations = [
                RelationRule(
                    case=str(r.get("case", "")),
                    self=str(r.get("self", "")),
                    other=str(r.get("other", "")),
                    note=r.get("note"),
                )
                for r in raw_addr.get("relations", [])
                if r.get("case")
            ]
            addr_rules = AddressRules(
                principle=raw_addr.get("principle"),
                first_person=_parse_pronoun_list(raw_addr.get("first_person", [])),
                second_person=_parse_pronoun_list(raw_addr.get("second_person", [])),
                honorifics=_parse_pronoun_list(raw_addr.get("honorifics", [])),
                relations=relations,
            )

        # Parse terms
        terms = []
        for t in data.get("terms", []):
            if t.get("ja") and t.get("vi"):
                terms.append(
                    GlossaryTerm(
                        ja=str(t["ja"]),
                        en=str(t["en"]) if t.get("en") else None,
                        zh=str(t["zh"]) if t.get("zh") else None,
                        aliases=[str(a) for a in t.get("aliases", [])],
                        vi=str(t["vi"]),
                        alt=[str(a) for a in t.get("alt", [])],
                        note=t.get("note"),
                        genre=t.get("genre"),
                    )
                )

        # Parse characters
        characters = []
        for c in data.get("characters", []):
            speaks = [
                CharacterSpeaksTo(
                    listener=str(s.get("listener", "")),
                    self=str(s.get("self", "")),
                    other=str(s.get("other", "")),
                    note=s.get("note"),
                )
                for s in c.get("speaks_to", [])
            ]
            characters.append(
                Character(
                    name_ja=str(c.get("name_ja", "")),
                    name_vi=str(c.get("name_vi", "")),
                    gender=c.get("gender"),
                    speaks_to=speaks,
                )
            )

        return Glossary(
            schema_ver=schema_ver,
            style_rules=style_rules,
            difficulty_markers=difficulty_markers,
            address_rules=addr_rules,
            terms=terms,
            characters=characters,
        )
    except Exception as e:
        print(f"⚠️  Failed to load glossary from {glossary_path}: {e}")
        return None


def merge_glossaries(base: Optional[Glossary], override: Optional[Glossary]) -> Optional[Glossary]:
    """
    Merges base glossary with series-specific override glossary.
    Override terms and characters take precedence over base entries.
    """
    if not base and not override:
        return None
    if not base:
        return override
    if not override:
        return base

    # Combine style rules (deduplicated, keeping order)
    seen_rules = set()
    combined_rules = []
    for r in override.style_rules + base.style_rules:
        if r not in seen_rules:
            seen_rules.add(r)
            combined_rules.append(r)

    # Combine difficulty markers (deduplicated)
    diff_markers = list(dict.fromkeys(override.difficulty_markers + base.difficulty_markers))

    # Address rules: prefer override's address rules if present, otherwise base
    addr = override.address_rules or base.address_rules

    # Merge terms: override terms replace base terms with same ja
    term_dict: Dict[str, GlossaryTerm] = {t.ja: t for t in base.terms}
    for ot in override.terms:
        term_dict[ot.ja] = ot  # overrides matching ja
    merged_terms = list(term_dict.values())

    # Merge characters: override characters replace base characters by name_ja (or name_vi fallback)
    char_dict: Dict[str, Character] = {(c.name_ja or c.name_vi): c for c in base.characters}
    for oc in override.characters:
        char_dict[oc.name_ja or oc.name_vi] = oc
    merged_characters = list(char_dict.values())

    return Glossary(
        schema_ver=2,
        style_rules=combined_rules,
        difficulty_markers=diff_markers,
        address_rules=addr,
        terms=merged_terms,
        characters=merged_characters,
    )
