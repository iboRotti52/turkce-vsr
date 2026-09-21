"""
tests/test_spotter.py — KeywordSpotter ve Evaluation Metrics Testleri
"""

import pytest
import torch
from src.evaluation.metrics import (
    compute_batch_cer,
    compute_batch_wer,
    compute_cer,
    compute_keyword_spotting_metrics,
    compute_wer,
    evaluate_predictions,
    levenshtein_distance,
)
from src.spotter.keyword_spotter import DetectedKeyword, KeywordSpotter
from src.vocab.turkish_vocab import CHAR_TO_IDX, VOCAB_SIZE


def test_levenshtein_distance():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "") == 3
    assert levenshtein_distance("", "def") == 3
    assert levenshtein_distance("kitap", "kitap") == 0
    assert levenshtein_distance("kitap", "katip") == 2
    assert levenshtein_distance(["bir", "gün"], ["bir", "gün"]) == 0
    assert levenshtein_distance(["bir", "gün"], ["bir"]) == 1


def test_cer_and_wer():
    # Mükemmel eşleşme
    assert compute_cer("merhaba dünya", "merhaba dünya") == 0.0
    assert compute_wer("merhaba dünya", "merhaba dünya") == 0.0

    # Büyük/küçük harf ve Türkçe karakter toleransı
    assert compute_cer("İSTANBUL", "istanbul") == 0.0
    assert compute_wer("IŞIK", "ışık") == 0.0

    # 1 karakter fark: "kitap" (5 karakter) -> "katap" (1 harf fark: CER = 1/5 = 0.2)
    assert pytest.approx(compute_cer("kitap", "katap"), 0.01) == 0.2

    # 1 kelime fark: "bu bir test" (3 kelime) -> "bu test" (1 eksik: WER = 1/3 = 0.333)
    assert pytest.approx(compute_wer("bu bir test", "bu test"), 0.01) == 0.3333

    # Batch testleri
    refs = ["bu bir test", "güzel bir gün"]
    hyps = ["bu bir test", "güzel gün"]
    assert compute_batch_cer(refs, refs) == 0.0
    assert compute_batch_wer(refs, refs) == 0.0
    assert compute_batch_wer(refs, hyps) > 0.0


def test_keyword_spotting_metrics_calculation():
    # Hedef sözlük
    target = {"bir", "bu", "ve", "çok", "de"}

    # Ref: "bir", "bu", "ve", "çok" (4 hedef kelime)
    # Hyp: "bir", "bu", "de" (3 hedef kelime: 2 TP ("bir", "bu"), 1 FP ("de"), 2 FN ("ve", "çok"))
    ref = ["bu bir test ve çok güzel"]
    hyp = ["bu bir de harika"]

    res = compute_keyword_spotting_metrics(ref, hyp, target_vocab=target)
    assert res["true_positives"] == 2
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 2
    assert pytest.approx(res["precision"], 0.01) == 2 / 3
    assert pytest.approx(res["recall"], 0.01) == 2 / 4
    expected_f1 = 2 * (2 / 3) * (2 / 4) / (2 / 3 + 2 / 4)
    assert pytest.approx(res["f1"], 0.01) == expected_f1


def test_spotter_from_text():
    spotter = KeywordSpotter()
    text = "Bugün hava çok güzel ama biraz soğuk gibi."
    keywords = spotter.spot_from_text(text)
    # çok, ama, biraz, gibi, güzel hepsi top500'dedir
    assert "çok" in keywords
    assert "ama" in keywords
    assert "güzel" in keywords


def test_spotter_from_logits_exact_match():
    spotter = KeywordSpotter(min_confidence=0.1)

    # "ben" kelimesini oluşturan karakterler: 'b', 'e', 'n'
    b_idx = CHAR_TO_IDX["b"]
    e_idx = CHAR_TO_IDX["e"]
    n_idx = CHAR_TO_IDX["n"]
    blank = 0

    # 15 karelik sahte logits: [blank, blank, b_idx, b_idx, blank, e_idx, e_idx, blank, n_idx, n_idx, blank, blank, blank, blank, blank]
    seq = [blank, blank, b_idx, b_idx, blank, e_idx, e_idx, blank, n_idx, n_idx, blank, blank, blank, blank, blank]
    T = len(seq)
    logits = torch.full((T, VOCAB_SIZE), -10.0)
    for t, idx in enumerate(seq):
        logits[t, idx] = 10.0  # çok yüksek olasılık

    detections = spotter.spot_from_logits(logits)
    assert len(detections) >= 1
    det = detections[0]
    assert det.word == "ben"
    assert not det.is_viseme_approx
    assert det.confidence > 0.8
    assert det.start_frame < det.end_frame
    assert det.rank_in_500 > 0


def test_spotter_viseme_tolerance():
    # "pen" kelimesi (viseme signature 'ben' ile aynıdır: b/p bilabial)
    spotter = KeywordSpotter(viseme_tolerance=True, min_confidence=0.1)
    
    p_idx = CHAR_TO_IDX["p"]
    e_idx = CHAR_TO_IDX["e"]
    n_idx = CHAR_TO_IDX["n"]
    blank = 0

    seq = [blank, p_idx, p_idx, blank, e_idx, blank, n_idx, blank]
    T = len(seq)
    logits = torch.full((T, VOCAB_SIZE), -10.0)
    for t, idx in enumerate(seq):
        logits[t, idx] = 10.0

    detections = spotter.spot_from_logits(logits)
    # 'pen' 500 kelimede değil, ancak 'ben' 500 kelimede ve aynı visem imzasına sahip ('0-7-2')
    words_spotted = [d.word for d in detections]
    assert "ben" in words_spotted
    match = [d for d in detections if d.word == "ben"][0]
    assert match.is_viseme_approx is True


def test_evaluate_predictions_all_in_one():
    refs = ["bu bir test", "çok güzel bir gün"]
    hyps = ["bu bir test", "çok güzel bir gün"]
    results = evaluate_predictions(refs, hyps)

    assert results["cer"] == 0.0
    assert results["wer"] == 0.0
    assert results["spotter_precision"] == 1.0
    assert results["spotter_recall"] == 1.0
    assert results["spotter_f1"] == 1.0


def test_spotter_pad_and_unk_handling():
    from src.vocab.turkish_vocab import PAD_IDX, UNK_IDX
    spotter = KeywordSpotter(min_confidence=0.1)

    b_idx = CHAR_TO_IDX["b"]
    e_idx = CHAR_TO_IDX["e"]
    n_idx = CHAR_TO_IDX["n"]

    # PAD ve UNK tokenları ile çevrili "ben"
    seq = [PAD_IDX, PAD_IDX, b_idx, e_idx, n_idx, UNK_IDX, PAD_IDX]
    logits = torch.full((len(seq), VOCAB_SIZE), -10.0)
    for t, idx in enumerate(seq):
        logits[t, idx] = 10.0

    detections = spotter.spot_from_logits(logits)
    assert len(detections) == 1
    assert detections[0].word == "ben"


def test_spotter_batch_logits():
    spotter = KeywordSpotter(min_confidence=0.1)
    batch_logits = torch.randn(3, 25, VOCAB_SIZE)
    results = spotter.spot_batch(batch_logits)

    assert len(results) == 3
    assert isinstance(results[0], list)


def test_spotter_unmapped_token_does_not_spot_w():
    # Apostrophe veya harita dışı sembollerin 'w' kelimesine eşleşmesini engelleme testi
    spotter = KeywordSpotter(viseme_tolerance=True, min_confidence=0.1)
    apos = CHAR_TO_IDX["'"]
    seq = [apos, apos]
    logits = torch.full((len(seq), VOCAB_SIZE), -10.0)
    for t, idx in enumerate(seq):
        logits[t, idx] = 10.0

    detections = spotter.spot_from_logits(logits)
    assert len(detections) == 0


def test_cer_wer_empty_references_with_hypotheses():
    # Boş referansa karşı halüsinasyon yapıldığında 0.0 değil hata oranı dönmeli
    cer = compute_batch_cer([""], ["merhaba dünya"])
    wer = compute_batch_wer([""], ["merhaba dünya"])
    assert cer == 1.0
    assert wer == 1.0


def test_evaluate_predictions_with_spotted_keywords():
    refs = ["bu bir test"]
    hyps = ["bu bir test"]
    spotted = [[DetectedKeyword(word="bu", start_frame=0, end_frame=5, start_sec=0.0, end_sec=0.2, confidence=0.9, rank_in_500=2)]]

    res = evaluate_predictions(refs, hyps, spotted_keywords=spotted)
    assert res["cer"] == 0.0
    assert res["wer"] == 0.0
    assert res["spotter_precision"] == 1.0
    assert res["spotter_recall"] == 0.5  # "bu" tespit edildi, "bir" tespit edilmedi (1/2)
    assert res["spotter_f1"] > 0.0


def test_spotter_continuous_no_spaces_text():
    spotter = KeywordSpotter()
    # Boşluksuz sürekli konuşma transkripti ("zaman" ilk 500'dedir)
    text = "bugecebirzaman"
    spotted = spotter.spot_from_text(text)
    assert "gece" in spotted
    assert "zaman" in spotted
    assert "bir" in spotted



def test_spotter_continuous_logits_substring():
    # Boşluksuz CTC karakter çıktısı (b-u-g-e-c-e) üzerinden substring yakalama
    spotter = KeywordSpotter(min_confidence=0.1)
    chars = ["b", "u", "g", "e", "c", "e"]
    indices = [CHAR_TO_IDX[c] for c in chars]

    logits = torch.full((len(indices), VOCAB_SIZE), -10.0)
    for t, idx in enumerate(indices):
        logits[t, idx] = 10.0

    detections = spotter.spot_from_logits(logits)
    words_spotted = [d.word for d in detections]
    assert "gece" in words_spotted


def test_spotter_sliding_window_ctc():
    from src.models.vsr_conformer import VSRConformerModel
    model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=64, num_layers=1, encoder_type="bigru")
    model.eval()

    spotter = KeywordSpotter(target_vocab={"bir", "ben", "zaman"})
    video = torch.randn(1, 1, 25, 88, 88)

    detections = spotter.spot_sliding_window_ctc(model, video, window_sizes=(10, 14), stride=4, min_score=-5.0)
    assert isinstance(detections, list)
    for d in detections:
        assert d.word in {"bir", "ben", "zaman"}
        assert d.start_frame <= d.end_frame
        assert d.match_type == "ctc_alignment"


