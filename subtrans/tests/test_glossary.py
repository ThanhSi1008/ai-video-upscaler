# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from subtrans.glossary import load_glossary, merge_glossaries, DEFAULT_BASE_GLOSSARY_PATH
from subtrans.models import Glossary, GlossaryTerm, Character


def test_load_default_base_glossary():
    assert DEFAULT_BASE_GLOSSARY_PATH.exists()
    glossary = load_glossary(DEFAULT_BASE_GLOSSARY_PATH)
    assert glossary is not None
    assert len(glossary.style_rules) == 12
    assert len(glossary.terms) >= 200
    assert glossary.address_rules is not None
    assert len(glossary.address_rules.first_person) > 0


def test_merge_glossaries():
    base = Glossary(
        style_rules=["Rule 1"],
        terms=[
            GlossaryTerm(ja="魔法", vi="ma pháp"),
            GlossaryTerm(ja="勇者", vi="anh hùng")
        ]
    )
    override = Glossary(
        style_rules=["Rule 2"],
        terms=[
            GlossaryTerm(ja="勇者", vi="Dũng giả")  # overrides base
        ],
        characters=[
            Character(name_ja="スバル", name_vi="Subaru")
        ]
    )

    merged = merge_glossaries(base, override)
    assert merged is not None
    assert "Rule 2" in merged.style_rules
    assert "Rule 1" in merged.style_rules
    term_map = {t.ja: t.vi for t in merged.terms}
    assert term_map["勇者"] == "Dũng giả"  # Override took precedence!
    assert term_map["魔法"] == "ma pháp"
    assert len(merged.characters) == 1
    assert merged.characters[0].name_vi == "Subaru"
