# -*- coding: utf-8 -*-
"""
subtrans/models.py
Data models for subtrans subtitle translation tool (Schema v2 support).
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Cue(BaseModel):
    """Represents a single subtitle cue."""
    id: int
    start: int  # Start time in milliseconds
    end: int    # End time in milliseconds
    style: str = "Default"
    ja: str     # Cleaned Japanese dialogue
    en: Optional[str] = None  # Aligned English dialogue (optional)
    vi: Optional[str] = None  # Translated Vietnamese dialogue
    backend_used: Optional[str] = None  # "gemini" | "local" | "cache" | "skipped"
    model_used: Optional[str] = None  # e.g. "gemini-3.8-flash", "translategemma:12b"
    is_skipped: bool = False
    is_dropped_sound: bool = False
    used_en_fallback: bool = False
    uncertain: bool = False
    raw_text: Optional[str] = None  # Original raw text for skipped cues


class GlossaryTerm(BaseModel):
    ja: str
    en: Optional[str] = None
    zh: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    vi: str
    alt: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    genre: Optional[str] = None


class PronounRule(BaseModel):
    ja: str
    vi: str
    alt: List[str] = Field(default_factory=list)
    note: Optional[str] = None


class RelationRule(BaseModel):
    case: str
    self: str
    other: str
    note: Optional[str] = None


class AddressRules(BaseModel):
    principle: Optional[str] = None
    first_person: List[PronounRule] = Field(default_factory=list)
    second_person: List[PronounRule] = Field(default_factory=list)
    honorifics: List[PronounRule] = Field(default_factory=list)
    relations: List[RelationRule] = Field(default_factory=list)


class CharacterSpeaksTo(BaseModel):
    listener: str
    self: str
    other: str
    note: Optional[str] = None


class Character(BaseModel):
    name_ja: str
    name_vi: str
    gender: Optional[str] = None
    speaks_to: List[CharacterSpeaksTo] = Field(default_factory=list)


class Glossary(BaseModel):
    schema_ver: int = 2
    style_rules: List[str] = Field(default_factory=list)
    address_rules: Optional[AddressRules] = None
    terms: List[GlossaryTerm] = Field(default_factory=list)
    characters: List[Character] = Field(default_factory=list)
