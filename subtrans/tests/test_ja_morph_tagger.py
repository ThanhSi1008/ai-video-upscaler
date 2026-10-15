# -*- coding: utf-8 -*-
"""
subtrans/tests/test_ja_morph_tagger.py
Unit tests for Japanese morphological and register tagger (ja_morph_tagger.py):
- Pronoun identification (1P and 2P)
- Honorifics and kinship detection
- Sentence-final particle (SFP) register detection
- Neutral sentences returning None
- Complex multi-hint utterances
- Integration check in translate_batch payload
"""

import json
from unittest.mock import MagicMock
import pytest

from subtrans.ja_morph_tagger import extract_ja_linguistic_hints
from subtrans.models import Cue
from subtrans.gemini_backend import GeminiTranslator


def test_extract_1p_pronouns():
    """Verifies detection of 1st-person pronouns with correct register labels."""
    assert "1P:ore" in (extract_ja_linguistic_hints("俺の番だ。") or "")
    assert "1P:boku" in (extract_ja_linguistic_hints("僕は行かないよ。") or "")
    assert "1P:watashi" in (extract_ja_linguistic_hints("私がやります。") or "")
    assert "1P:atashi" in (extract_ja_linguistic_hints("あたしがやる！") or "")
    assert "1P:washi" in (extract_ja_linguistic_hints("わしの話を聞け。") or "")
    assert "1P:warawa" in (extract_ja_linguistic_hints("妾の前にひれ伏すがよい。") or "")
    assert "1P/REFL:jibun" in (extract_ja_linguistic_hints("自分の身は自分で守る。") or "")


def test_extract_2p_pronouns():
    """Verifies detection of 2nd-person pronouns with adversarial/casual register labels."""
    assert "2P:kisama" in (extract_ja_linguistic_hints("貴様、何者だ！") or "")
    assert "2P:temee" in (extract_ja_linguistic_hints("てめぇ、ふざけるな！") or "")
    assert "2P:anta" in (extract_ja_linguistic_hints("あんたのせいよ。") or "")
    assert "2P:omae" in (extract_ja_linguistic_hints("お前はどうする？") or "")
    assert "2P:kimi" in (extract_ja_linguistic_hints("君の名前は？") or "")


def test_extract_honorifics_and_kinship():
    """Verifies detection of honorifics and family titles."""
    assert "HON:senpai" in (extract_ja_linguistic_hints("先輩、お疲れ様です！") or "")
    assert "HON:sensei" in (extract_ja_linguistic_hints("先生、教えてください。") or "")
    assert "HON:sama" in (extract_ja_linguistic_hints("フリーレン様、こちらです。") or "")
    assert "KIN:onii" in (extract_ja_linguistic_hints("お兄ちゃん、待って！") or "")
    assert "KIN:onee" in (extract_ja_linguistic_hints("姉さん、大変だよ！") or "")


def test_extract_sentence_final_particles():
    """Verifies detection of sentence-final particles conveying gendered or social tone."""
    assert "SFP:nam_tính_mạnh" in (extract_ja_linguistic_hints("これで終わりだぜ！") or "")
    assert "SFP:nữ_tính_hoặc_tiểu_thư" in (extract_ja_linguistic_hints("ごきげんよう、ですわ。") or "")
    assert "SFP:hậu_bối_kính_cẩn" in (extract_ja_linguistic_hints("了解っす！") or "")


def test_neutral_utterance_returns_none():
    """Verifies that utterances without register markers return None."""
    assert extract_ja_linguistic_hints("今日はいい天気ですね。") is None
    assert extract_ja_linguistic_hints("駅はあちらです。") is None
    assert extract_ja_linguistic_hints("") is None


def test_multi_hint_combination():
    """Verifies that an utterance containing both a pronoun and an SFP produces combined hints."""
    res = extract_ja_linguistic_hints("てめぇをぶっ飛ばすぜ！")
    assert res is not None
    assert "2P:temee" in res
    assert "SFP:nam_tính_mạnh" in res


def test_translate_batch_injects_ja_hints():
    """Verifies that translate_batch properly injects ja_hints into cues_to_translate."""
    translator = GeminiTranslator(api_key="mock_key")
    translator.client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '[{"id": 1, "vi": "Mày chết chắc!"}, {"id": 2, "vi": "Chào buổi sáng."}]'
    translator.client.models.generate_content.return_value = mock_resp

    cues = [
        Cue(id=1, start=0, end=1000, ja="てめぇを倒す！", en="I'll defeat you!"),
        Cue(id=2, start=1000, end=2000, ja="おはようございます", en="Good morning"),
    ]
    translator.translate_batch(cues, [])

    # Lấy nội dung prompt được gửi đi
    call_args = translator.client.models.generate_content.call_args
    prompt_sent = call_args.kwargs.get("contents", "")
    assert "ja_hints" in prompt_sent
    assert "2P:temee" in prompt_sent
