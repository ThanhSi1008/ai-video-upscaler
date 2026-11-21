"""Schema v2 anime glossary loader, pointer resolver, and per-series merger."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib
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
    Auto-detect a series-specific glossary file under glossaries/.
    Example: 'Futsutsuka_Na_Akujo_Dewa_Gozaimasu_Ga' -> glossaries/glossary.futsutsuka.yaml
    """
    g_dir = glossaries_dir or GLOSSARIES_DIR
    if not g_dir.exists() or not series_or_filename:
        return None

    # Normalize input string (supports parent folder names like 'Frieren/S01E01.mkv')
    raw_str = str(series_or_filename)
    path_obj = Path(raw_str)
    search_targets = [path_obj.name]
    if path_obj.parent and path_obj.parent.name and path_obj.parent.name not in (".", "raw_sources", "out", "samples", "subtitles"):
        search_targets.append(path_obj.parent.name)
    search_targets.append(raw_str)

    candidates = sorted(
        list(g_dir.glob("glossary.*.yaml"))
        + list(g_dir.glob("glossary.*.yml"))
        + list(g_dir.glob("glossary.*.json")),
        key=lambda p: len(p.name),
        reverse=True,
    )
    for target in search_targets:
        norm_name = re.sub(r"[^a-zA-Z0-9]", "", target.lower())
        if not norm_name:
            continue
        for cand in candidates:
            parts = cand.name.split(".")
            if len(parts) >= 3 and parts[0] == "glossary":
                key = parts[1].lower()
                norm_key = re.sub(r"[^a-zA-Z0-9]", "", key)
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


def resolve_glossary_target(glossary_path: Optional[Path]) -> Optional[Path]:
    """
    Resolve cross-platform plain-text alias pointers safely
    (e.g., glossaries/base.yaml -> ../subtrans/glossary.yaml,
    glossary.reincarnatedasasword.yaml -> glossary.tenken.yaml).
    """
    if not glossary_path or not glossary_path.exists():
        return glossary_path
    repo_root = Path(__file__).resolve().parent.parent
    seen = set()
    curr = glossary_path.resolve()
    while curr.exists() and curr not in seen:
        seen.add(curr)
        if curr.suffix.lower() not in (".yaml", ".yml"):
            break
        try:
            raw = curr.read_text(encoding="utf-8").strip()
            if "\n" not in raw and raw.endswith((".yaml", ".yml")):
                nxt = (curr.parent / raw).resolve()
                # Allow pointers targeting within the same directory or repository root.
                if (nxt.is_relative_to(repo_root) or nxt.is_relative_to(curr.parent.resolve())) and nxt.exists() and nxt != curr:
                    curr = nxt
                    continue
        except Exception:
            break
        break
    return curr


def compute_glossary_hash(glossary: Optional[Glossary]) -> str:
    """Compute a short 16-char SHA-256 hash of Glossary contents for cache busting."""
    if not glossary:
        return ""
    payload = {
        "style_rules": glossary.style_rules,
        "terms": [t.model_dump() for t in glossary.terms],
        "characters": [c.model_dump() for c in glossary.characters],
        "address_rules": glossary.address_rules.model_dump() if glossary.address_rules else None,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_glossary(glossary_path: Optional[Path]) -> Optional[Glossary]:
    """Loads a Schema v2 YAML or JSON glossary file."""
    if not glossary_path or not glossary_path.exists():
        return None

    try:
        resolved_path = resolve_glossary_target(glossary_path) or glossary_path
        with open(resolved_path, "r", encoding="utf-8") as f:
            if resolved_path.suffix.lower() in (".yaml", ".yml"):
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)

        # Handle cross-platform plain-text pointer aliases
        if isinstance(data, str) and data.strip().endswith((".yaml", ".yml")):
            target_path = (resolved_path.parent / data.strip()).resolve()
            if target_path.exists() and target_path != resolved_path:
                return load_glossary(target_path)

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
                    aliases=[str(a) for a in c.get("aliases", []) if a],
                    gender=c.get("gender"),
                    speaking_style=c.get("register") or c.get("speaking_style"),
                    note=c.get("note"),
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


def _merge_pronoun_rules(base_list: List[PronounRule], override_list: List[PronounRule]) -> List[PronounRule]:
    merged: Dict[str, PronounRule] = {r.ja: r for r in (base_list or [])}
    for r in override_list or []:
        merged[r.ja] = r
    return list(merged.values())


def _merge_address_rules(base_addr: Optional[AddressRules], override_addr: Optional[AddressRules]) -> Optional[AddressRules]:
    """Deep-merges base and series-override AddressRules without wiping base pronoun/honorific tables."""
    if not base_addr and not override_addr:
        return None
    if not base_addr:
        return override_addr
    if not override_addr:
        return base_addr

    rel_map: Dict[str, RelationRule] = {r.case: r for r in (base_addr.relations or [])}
    for r in override_addr.relations or []:
        rel_map[r.case] = r

    return AddressRules(
        principle=override_addr.principle or base_addr.principle,
        first_person=_merge_pronoun_rules(base_addr.first_person, override_addr.first_person),
        second_person=_merge_pronoun_rules(base_addr.second_person, override_addr.second_person),
        honorifics=_merge_pronoun_rules(base_addr.honorifics, override_addr.honorifics),
        relations=list(rel_map.values()),
    )


def merge_glossaries(base: Optional[Glossary], override: Optional[Glossary]) -> Optional[Glossary]:
    """
    Merges base glossary with series-specific override glossary.
    Override terms and characters take precedence over base entries, while AddressRules are deep-merged.
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

    # Deep-merge address rules so partial series overrides keep base honorifics & pronoun tables
    addr = _merge_address_rules(base.address_rules, override.address_rules)

    # Merge terms: override terms replace base terms with same ja
    term_dict: Dict[str, GlossaryTerm] = {t.ja: t for t in base.terms}
    for ot in override.terms:
        term_dict[ot.ja] = ot  # overrides matching ja
    merged_terms = list(term_dict.values())

    # Merge characters: override characters replace base characters by name_ja (or name_vi fallback), preserving aliases
    char_dict: Dict[str, Character] = {(c.name_ja or c.name_vi): c for c in base.characters}
    for oc in override.characters:
        key = oc.name_ja or oc.name_vi
        existing = char_dict.get(key)
        if existing and not oc.aliases and existing.aliases:
            oc = oc.model_copy(update={"aliases": list(existing.aliases)})
        char_dict[key] = oc
    merged_characters = list(char_dict.values())

    return Glossary(
        schema_ver=2,
        style_rules=combined_rules,
        difficulty_markers=diff_markers,
        address_rules=addr,
        terms=merged_terms,
        characters=merged_characters,
    )


# Backward-compatible helper aliases used by src/subtitles.py Whisper prompt builder
find_matching_glossary = find_series_glossary
load_glossary_file = load_glossary
