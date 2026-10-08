# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from subtrans.glossary import load_glossary, merge_glossaries, DEFAULT_BASE_GLOSSARY_PATH
from subtrans.models import Glossary, GlossaryTerm, Character


def test_load_default_base_glossary():
    assert DEFAULT_BASE_GLOSSARY_PATH.exists()
    glossary = load_glossary(DEFAULT_BASE_GLOSSARY_PATH)
    assert glossary is not None
    assert len(glossary.style_rules) >= 13
    assert any("Addressing by proper name" in r for r in glossary.style_rules)
    assert len(glossary.terms) >= 200
    assert glossary.address_rules is not None
    assert len(glossary.address_rules.first_person) > 0
    calling_by_name = [r for r in glossary.address_rules.second_person if r.ja == "(calling by name)"]
    assert len(calling_by_name) == 1
    assert "Romaji" in calling_by_name[0].vi or "Romaji" in (calling_by_name[0].note or "")


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


def test_find_series_glossary():
    from subtrans.glossary import find_series_glossary
    # Test matching Futsutsuka
    res = find_series_glossary("Futsutsuka_Na_Akujo_Dewa_Gozaimasu_Ga_S01E04")
    assert res is not None
    assert res.name == "glossary.futsutsuka.yaml"

    # Test matching Mushoku
    res_m = find_series_glossary("Mushoku_Tensei_S02E01")
    assert res_m is not None
    assert res_m.name == "glossary.mushoku.yaml"

    # Non-existent
    assert find_series_glossary("Some_Completely_Unknown_Anime_123") is None


def test_futsutsuka_glossary_content():
    from subtrans.glossary import find_series_glossary
    g_path = find_series_glossary("Futsutsuka")
    assert g_path is not None
    g = load_glossary(g_path)
    assert g is not None

    char_names = {c.name_vi for c in g.characters}
    assert "Reirin" in char_names
    assert "Keigetsu" in char_names
    assert "Leelee" in char_names
    assert "Gayou" in char_names

    # Check terms do not contain Han-Viet name translations
    term_vi_map = {t.ja: t.vi for t in g.terms}
    assert term_vi_map["玲琳"] == "Reirin"
    assert term_vi_map["慧月"] == "Keigetsu"
    assert term_vi_map["莉莉"] == "Leelee"
    assert "Linh Lâm" not in term_vi_map.values()
    assert "Huệ Nguyệt" not in term_vi_map.values()
    assert "Tuệ Nguyệt" not in term_vi_map.values()
    assert "Lợi Lợi" not in term_vi_map.values()

