"""
tests/test_vocab.py — Türkçe Kelime ve CTC Sözlük Testleri
"""

import pytest
import torch
from src.vocab.turkish_vocab import (
    VOCAB_SIZE,
    BLANK_IDX,
    normalize_turkish_text,
    text_to_indices,
    indices_to_text,
    ctc_greedy_decode,
    to_viseme_sequence,
)
from src.vocab.top500_words import (
    TOP_500_WORDS,
    is_top500_word,
    get_word_rank,
    get_word_viseme_signature,
)


def test_vocab_size_and_special_tokens():
    assert VOCAB_SIZE == 34  # 4 special + 29 letters + 1 apostrophe
    assert BLANK_IDX == 0


def test_turkish_normalization():
    # Türkçe I/ı ve İ/i testleri
    assert normalize_turkish_text("IŞIK") == "ışık"
    assert normalize_turkish_text("İSTANBUL") == "istanbul"
    assert normalize_turkish_text("Çığır, Açan!") == "çığır açan"
    assert normalize_turkish_text("Ahmet'in") == "ahmet'in"
    # Şapkalı (düzeltme işaretli) sesli harflerin normalizasyonu
    assert normalize_turkish_text("Hâlâ kâğıt ve rüzgâr") == "hala kağıt ve rüzgar"
    assert normalize_turkish_text("Mahkûm") == "mahkum"


def test_text_to_indices_and_back():
    sample = "merhaba dünya"
    indices = text_to_indices(sample)
    assert len(indices) == len(sample)
    recovered = indices_to_text(indices)
    assert recovered == sample


def test_ctc_greedy_decode():
    # Tekrarlanan harflerin ve blank'in doğru çözülmesi
    # "b" (idx for b), "b", blank(0), "e", "n"
    from src.vocab.turkish_vocab import CHAR_TO_IDX
    b = CHAR_TO_IDX["b"]
    e = CHAR_TO_IDX["e"]
    n = CHAR_TO_IDX["n"]
    
    # Batch = 1, Time = 6
    seq = torch.tensor([[b, b, 0, e, n, n]])
    decoded = ctc_greedy_decode(seq)
    assert decoded == ["ben"]


def test_viseme_mapping():
    # 'b' ve 'p' ikisi de çift dudaksı (viseme 0)
    assert to_viseme_sequence("b") == [0]
    assert to_viseme_sequence("p") == [0]
    assert to_viseme_sequence("f") == [1]  # diş-dudaksı
    assert get_word_viseme_signature("ben") == get_word_viseme_signature("pen")


def test_top500_words():
    assert len(TOP_500_WORDS) == 500
    assert len(set(TOP_500_WORDS)) == 500
    assert is_top500_word("bir")
    assert is_top500_word("bu")
    assert is_top500_word("ve")
    assert get_word_rank("bir") == 1
    assert get_word_rank("bu") >= 2
    assert not is_top500_word("süperkalifrajilistik")

    # Hiçbir kelime 1 harfli olmamalı (VSR false-positive engelleme)
    assert all(len(w) >= 2 for w in TOP_500_WORDS)

    # Parçalanmış ekler ve YouTube artefaktları elenmiş olmalı
    forbidden_artifacts = {"altyazı", "kamera", "video", "nın", "nin", "yı", "yi", "ca", "ra", "ba", "so", "na"}
    for bad in forbidden_artifacts:
        assert bad not in TOP_500_WORDS, f"Yasaklı artefakt {bad} sözlükte bulundu!"
