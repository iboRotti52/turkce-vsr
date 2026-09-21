"""
tests/test_lexicon_decoder.py — LexiconBeamSearchDecoder Birim Testleri
"""

import pytest
import torch

from src.spotter.keyword_spotter import DetectedKeyword
from src.spotter.lexicon_decoder import LexiconBeamSearchDecoder, TrieNode
from src.vocab.turkish_vocab import CHAR_TO_IDX, SPACE_IDX, VOCAB_SIZE


def test_trie_construction():
    target_words = {"var", "ben", "bir", "türkiye", "kitap"}
    decoder = LexiconBeamSearchDecoder(target_vocab=target_words, beam_size=10)
    assert decoder.root is not None
    # 'var' kontrolü
    n = decoder.root
    for ch in "var":
        assert ch in n.children
        n = n.children[ch]
    assert n.is_word
    assert n.word == "var"

    # Sözlükte olmayan kelime
    n2 = decoder.root
    assert "x" not in n2.children


def test_viseme_distance():
    decoder = LexiconBeamSearchDecoder()
    # Aynı harf -> 0.0
    assert decoder._get_viseme_distance("a", "a") == 0.0
    # Aynı visem (Bilabial: b ve p) -> 0.35
    assert decoder._get_viseme_distance("b", "p") == 0.35
    # Çift dudaksı ve diş-dudaksı (b ve v) -> 0.55
    assert decoder._get_viseme_distance("b", "v") == 0.55
    # Alakasız (a ve k) -> 2.50
    assert decoder._get_viseme_distance("a", "k") == 2.50


def test_decode_synthetic_word():
    # 2 kelimelik sentetik dizi: "ben bir"
    # Harfler: b, e, n, space, b, i, r
    target_words = {"ben", "bir", "ve", "bu"}
    decoder = LexiconBeamSearchDecoder(target_vocab=target_words, beam_size=20, fps=25.0)

    seq = ["b", "e", "n", " ", "b", "i", "r"]
    T = len(seq) * 3  # Her harf 3 kare sürsün
    logits = torch.full((T, VOCAB_SIZE), -5.0)

    for i, ch in enumerate(seq):
        idx = SPACE_IDX if ch == " " else CHAR_TO_IDX[ch]
        for f in range(i * 3, (i + 1) * 3):
            logits[f, idx] = 10.0  # Yüksek logit

    decoded_text, keywords = decoder.decode(logits)
    assert "ben" in decoded_text
    assert "bir" in decoded_text
    assert len(keywords) == 2
    assert keywords[0].word == "ben"
    assert keywords[1].word == "bir"
    assert keywords[0].start_sec >= 0.0
    assert keywords[1].end_sec > keywords[0].start_sec


def test_lm_prior_disambiguation():
    """
    Akustik olarak eşit güçte yarışan iki hipotezden (örn. 'bu bir' vs 'bu mar')
    Türkçe dil modelinin (unigram+bigram) 'bu bir'i seçtiğini kanıtlar.
    """
    target_words = {"bu", "bir", "mar"}
    decoder = LexiconBeamSearchDecoder(
        target_vocab=target_words,
        beam_size=30,
        lm_alpha=0.8,
        lm_beta=0.5,
        use_lm=True,
    )

    # Dizi: "bu " ardından hem 'b-i-r' hem 'm-a-r' için eşit logitler verilsin
    # "b", "u", " "
    seq1 = ["b", "u", " "]
    # Sonraki harfler için hem b/i/r hem m/a/r'a eşit logit verelim
    T = len(seq1) * 3 + 3 * 3
    logits = torch.full((T, VOCAB_SIZE), -5.0)

    for i, ch in enumerate(seq1):
        idx = SPACE_IDX if ch == " " else CHAR_TO_IDX[ch]
        for f in range(i * 3, (i + 1) * 3):
            logits[f, idx] = 8.0

    offset = len(seq1) * 3
    # Harf 1: 'b' ve 'm' eşit (8.0)
    for f in range(offset, offset + 3):
        logits[f, CHAR_TO_IDX["b"]] = 8.0
        logits[f, CHAR_TO_IDX["m"]] = 8.0
    # Harf 2: 'i' ve 'a' eşit (8.0)
    for f in range(offset + 3, offset + 6):
        logits[f, CHAR_TO_IDX["i"]] = 8.0
        logits[f, CHAR_TO_IDX["a"]] = 8.0
    # Harf 3: 'r' (8.0)
    for f in range(offset + 6, offset + 9):
        logits[f, CHAR_TO_IDX["r"]] = 8.0

    text, kws = decoder.decode(logits)
    # Dil modeli sayesinde ('bu', 'bir') açık ara kazanmalı!
    assert "bir" in text
    assert "mar" not in text
    words = [k.word for k in kws]
    assert words == ["bu", "bir"]


def test_decode_blank_separated_words():
    """
    Sessiz videolarda açık <space> token'ı üretilmese ve kelimeler sadece
    <blank> (dudak kapanması / sessizlik) ile ayrılsa bile çözücünün
    her iki kelimeyi de eksiksiz yakaladığını doğrular.
    """
    target_words = {"ben", "bir"}
    decoder = LexiconBeamSearchDecoder(target_vocab=target_words, beam_size=20, fps=25.0)

    seq = ["b", "e", "n", "<blank>", "<blank>", "<blank>", "b", "i", "r"]
    from src.vocab.turkish_vocab import BLANK_IDX
    T = len(seq) * 2
    logits = torch.full((T, VOCAB_SIZE), -5.0)

    for i, ch in enumerate(seq):
        idx = BLANK_IDX if ch == "<blank>" else CHAR_TO_IDX[ch]
        for f in range(i * 2, (i + 1) * 2):
            logits[f, idx] = 8.0

    text, kws = decoder.decode(logits)
    assert "ben" in text
    assert "bir" in text
    assert [k.word for k in kws] == ["ben", "bir"]


def test_decode_coarticulation_words():
    """
    İki kelime arasında ne space ne de blank olmadan doğrudan ardışık
    geçiş (koartikülasyon) olduğunda Trie kök geçişinin çalıştığını doğrular.
    """
    target_words = {"ben", "bir"}
    decoder = LexiconBeamSearchDecoder(target_vocab=target_words, beam_size=20, fps=25.0)

    seq = ["b", "e", "n", "b", "i", "r"]
    T = len(seq) * 2
    logits = torch.full((T, VOCAB_SIZE), -5.0)

    for i, ch in enumerate(seq):
        idx = CHAR_TO_IDX[ch]
        for f in range(i * 2, (i + 1) * 2):
            logits[f, idx] = 8.0

    text, kws = decoder.decode(logits)
    assert "ben" in text
    assert "bir" in text
    assert [k.word for k in kws] == ["ben", "bir"]


def test_decode_realistic_word_boundaries():
    """
    Kelime bittikten sonra uzun sessizlik gelse bile kelimenin
    tüm video boyunca uzatılmadığını ve gerçekçi bitiş zamanına sahip olduğunu doğrular.
    """
    target_words = {"bir"}
    decoder = LexiconBeamSearchDecoder(target_vocab=target_words, beam_size=20, fps=25.0)

    # 4 kare "bir", ardından 20 kare blank (toplam 24 kare = ~1 saniye)
    from src.vocab.turkish_vocab import BLANK_IDX
    T = 24
    logits = torch.full((T, VOCAB_SIZE), -5.0)
    logits[:, BLANK_IDX] = 5.0  # Varsayılan blank

    # İlk 6 karede "bir" emisyonu
    logits[0:2, CHAR_TO_IDX["b"]] = 10.0
    logits[2:4, CHAR_TO_IDX["i"]] = 10.0
    logits[4:6, CHAR_TO_IDX["r"]] = 10.0

    text, kws = decoder.decode(logits)
    assert "bir" in text
    assert len(kws) == 1
    # Bitiş zamanı videonun sonu (24. kare) DEĞİL, harflerin bittiği aralık (~5-6. kare) olmalı
    assert kws[0].end_frame <= 8
    assert kws[0].end_sec < 0.40


def test_decode_repetition_penalty():
    """
    Aynı kelimenin peş peşe halüsinasyon olarak tekrarlanmasını (örn. 'bir bir bir')
    repetition penalty'nin engellediğini doğrular.
    """
    target_words = {"bir", "ben"}
    decoder_with_rep = LexiconBeamSearchDecoder(
        target_vocab=target_words,
        beam_size=20,
        repeat_penalty=5.0,
        lm_beta=0.5,
    )
    # 6 kare 'b-i-r', ardından 6 kare belirsiz harfler
    seq = ["b", "i", "r"]
    T = 16
    logits = torch.full((T, VOCAB_SIZE), -4.0)
    for i, ch in enumerate(seq):
        for f in range(i * 2, (i + 1) * 2):
            logits[f, CHAR_TO_IDX[ch]] = 8.0

    text, kws = decoder_with_rep.decode(logits)
    # 'bir' tek sefer yakalanmalı, arka arkaya 'bir bir' tekrarlanmamalı
    words = [k.word for k in kws]
    assert words.count("bir") == 1


def test_decode_min_duration():
    """
    Çok kısa (örneğin sadece 1 karelik gürültü emisyonu) sahte kelimelerin
    min_dur_factor kısıtlamasıyla reddedildiğini doğrular.
    """
    target_words = {"türkiye"}  # 7 harfli uzun kelime
    decoder = LexiconBeamSearchDecoder(
        target_vocab=target_words,
        beam_size=20,
        min_dur_factor=1.0,  # en az 7 kare sürmeli
    )
    # Sadece 2 kare süren sahte bir emisyon
    T = 10
    from src.vocab.turkish_vocab import BLANK_IDX
    logits = torch.full((T, VOCAB_SIZE), -5.0)
    logits[:, BLANK_IDX] = 2.0
    logits[1, CHAR_TO_IDX["t"]] = 5.0

    text, kws = decoder.decode(logits)
    # 7 harfli 'türkiye' 2 karede tamamlanamaz
    assert "türkiye" not in text


