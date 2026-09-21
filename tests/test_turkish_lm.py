"""
tests/test_turkish_lm.py — Türkçe Dil Modeli (Unigram & Bigram) Birim Testleri
"""

import pytest
from src.vocab.turkish_lm import TurkishLanguageModel


def test_lm_unigram_scores():
    lm = TurkishLanguageModel()
    
    # En sık kelime 'bir', 'bu', 'de'
    score_bir = lm.score_unigram("bir")
    score_bu = lm.score_unigram("bu")
    score_rare = lm.score_unigram("buraya")
    
    # 'bir' log-olasılığı 'buraya'dan büyük olmalı
    assert score_bir > score_rare
    assert score_bir > -4.5
    assert score_rare < -7.0

    # Bilinmeyen / sözlük dışı kelime
    score_unk = lm.score_unigram("abrakadabraxyz")
    assert score_unk <= -8.5


def test_lm_bigram_transitions():
    lm = TurkishLanguageModel()

    # Sıkça birlikte kullanılan ikililer
    score_bu_bir = lm.score_transition("bu", "bir")
    score_bu_rare = lm.score_transition("bu", "buraya")
    assert score_bu_bir > score_bu_rare
    assert score_bu_bir > -1.0  # Bigram tablosundaki yüksek skor

    # Soru kalıpları
    score_var_mi = lm.score_transition("var", "mı")
    score_var_rare = lm.score_transition("var", "dakika")
    assert score_var_mi > score_var_rare
    assert score_var_mi > -0.8

    # Edat ve bağlaç kalıpları
    score_tabii_ki = lm.score_transition("tabii", "ki")
    assert score_tabii_ki > -0.7

    score_devam_ediyor = lm.score_transition("devam", "ediyor")
    assert score_devam_ediyor > -0.7


def test_lm_sentence_scoring():
    lm = TurkishLanguageModel()

    # Doğal Türkçe cümle vs rastgele sözcük dizisi
    natural_sentence = ["bu", "bir", "gerçekten", "çok", "büyük", "soru"]
    random_words = ["mar", "dakika", "ku", "kamera", "ak", "ayrı"]

    score_nat = lm.score_sentence(natural_sentence)
    score_rand = lm.score_sentence(random_words)

    # Doğal Türkçe cümle belirgin şekilde daha yüksek LM olasılığı almalı
    assert score_nat > score_rand


def test_lm_case_and_normalization():
    lm = TurkishLanguageModel()
    # Büyük harf ve Türkçe karakter normalizasyonu
    assert lm.score_unigram("BİR") == lm.score_unigram("bir")
    assert lm.score_transition("VAR", "MI") == lm.score_transition("var", "mı")
