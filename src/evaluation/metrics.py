"""
src/evaluation/metrics.py — Türkçe VSR CER, WER ve 500 Kelime Avcısı (Spotter) Metrikleri
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from src.vocab.turkish_vocab import normalize_turkish_text
from src.vocab.top500_words import TOP_500_SET


def levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """
    İki dizi (karakter veya kelime) arasındaki Levenshtein düzenleme mesafesini hesaplar.
    O(min(N, M)) bellek karmaşıklığı ile dinamik programlama.
    """
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    if n > m:
        seq1, seq2 = seq2, seq1
        n, m = m, n

    current_row = list(range(n + 1))
    for i in range(1, m + 1):
        previous_row = current_row
        current_row = [i] + [0] * n
        char2 = seq2[i - 1]
        for j in range(1, n + 1):
            add = previous_row[j] + 1
            delete = current_row[j - 1] + 1
            change = previous_row[j - 1] + (0 if seq1[j - 1] == char2 else 1)
            current_row[j] = min(add, delete, change)

    return current_row[n]


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Karakter Hata Oranı (Character Error Rate - CER).
    CER = Levenshtein(ref_chars, hyp_chars) / len(ref_chars)
    """
    ref_norm = normalize_turkish_text(reference)
    hyp_norm = normalize_turkish_text(hypothesis)

    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0

    dist = levenshtein_distance(ref_norm, hyp_norm)
    return float(dist / len(ref_norm))


def compute_batch_cer(references: List[str], hypotheses: List[str]) -> float:
    """
    Toplu Karakter Hata Oranı.
    Toplam düzenleme mesafesi / toplam referans karakter sayısı.
    """
    assert len(references) == len(hypotheses), "Referans ve hipotez boyutları eşleşmeli"
    total_dist = 0
    total_chars = 0

    for ref, hyp in zip(references, hypotheses):
        ref_norm = normalize_turkish_text(ref)
        hyp_norm = normalize_turkish_text(hyp)
        total_dist += levenshtein_distance(ref_norm, hyp_norm)
        total_chars += len(ref_norm)

    if total_chars == 0:
        return 0.0 if total_dist == 0 else 1.0
    return float(total_dist / total_chars)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Kelime Hata Oranı (Word Error Rate - WER).
    WER = Levenshtein(ref_words, hyp_words) / len(ref_words)
    """
    ref_words = normalize_turkish_text(reference).split()
    hyp_words = normalize_turkish_text(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist / len(ref_words))


def compute_batch_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Toplu Kelime Hata Oranı.
    Toplam düzenleme mesafesi / toplam referans kelime sayısı.
    """
    assert len(references) == len(hypotheses), "Referans ve hipotez boyutları eşleşmeli"
    total_dist = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = normalize_turkish_text(ref).split()
        hyp_words = normalize_turkish_text(hyp).split()
        total_dist += levenshtein_distance(ref_words, hyp_words)
        total_words += len(ref_words)

    if total_words == 0:
        return 0.0 if total_dist == 0 else 1.0
    return float(total_dist / total_words)


def extract_keywords(
    item: Union[str, List[Any]], target_vocab: Set[str]
) -> List[str]:
    """Metin, kelime listesi veya DetectedKeyword listesinden hedef kelimeleri ayıklar."""
    if isinstance(item, str):
        words = normalize_turkish_text(item).split()
        return [w for w in words if w in target_vocab]
    elif isinstance(item, list):
        keywords = []
        for elem in item:
            if hasattr(elem, "word"):
                w = normalize_turkish_text(elem.word)
            else:
                w = normalize_turkish_text(str(elem))
            if w in target_vocab:
                keywords.append(w)
        return keywords
    return []


def compute_keyword_spotting_metrics(
    references: List[str],
    hypotheses: List[Union[str, List[Any]]],
    target_vocab: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Sürekli konuşma içinde hedef 500 kelimenin tespit doğruluğunu hesaplar.
    Micro-averaged Precision, Recall ve F1-Score döndürür.
    """
    if target_vocab is None:
        target_vocab = TOP_500_SET

    assert len(references) == len(hypotheses), "Referans ve hipotez uzunlukları eşleşmeli"

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0

    for ref, hyp in zip(references, hypotheses):
        ref_keywords = extract_keywords(ref, target_vocab)
        hyp_keywords = extract_keywords(hyp, target_vocab)

        gt_counts = Counter(ref_keywords)
        pred_counts = Counter(hyp_keywords)

        sample_tp = 0
        for word, count in pred_counts.items():
            gt_c = gt_counts.get(word, 0)
            sample_tp += min(count, gt_c)

        sample_fp = sum(pred_counts.values()) - sample_tp
        sample_fn = sum(gt_counts.values()) - sample_tp

        total_tp += sample_tp
        total_fp += sample_fp
        total_fn += sample_fn
        total_gt += len(ref_keywords)
        total_pred += len(hyp_keywords)

    precision = total_tp / max(total_tp + total_fp, 1) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / max(total_tp + total_fn, 1) if (total_tp + total_fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "true_positives": int(total_tp),
        "false_positives": int(total_fp),
        "false_negatives": int(total_fn),
        "total_ground_truth": int(total_gt),
        "total_predicted": int(total_pred),
    }


def evaluate_predictions(
    references: List[str],
    hypotheses: List[Union[str, List[Any]]],
    spotted_keywords: Optional[List[Union[str, List[Any]]]] = None,
    target_vocab: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Tüm metrikleri (CER, WER, Precision, Recall, F1) bir arada hesaplar.
    references: Referans Türkçe metinler.
    hypotheses: Modelin deşifre ettiği tam metinler (veya geriye uyumluluk için tespit nesneleri).
    spotted_keywords: (İsteğe bağlı) KeywordSpotter tarafından tespit edilmiş anahtar kelimeler listesi.
    """
    # Metin hallerini alarak CER ve WER hesapla
    str_hyps = []
    for h in hypotheses:
        if isinstance(h, str):
            str_hyps.append(h)
        elif isinstance(h, list):
            words = [getattr(elem, "word", str(elem)) for elem in h]
            str_hyps.append(" ".join(words))
        else:
            str_hyps.append(str(h))

    cer = compute_batch_cer(references, str_hyps)
    wer = compute_batch_wer(references, str_hyps)

    # Spotter kaynak verisi: eğer spotted_keywords verildiyse onu kullan, yoksa hypotheses'ı kullan
    spotter_input = spotted_keywords if spotted_keywords is not None else hypotheses
    spot_metrics = compute_keyword_spotting_metrics(references, spotter_input, target_vocab)

    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "spotter_precision": spot_metrics["precision"],
        "spotter_recall": spot_metrics["recall"],
        "spotter_f1": spot_metrics["f1"],
        "true_positives": spot_metrics["true_positives"],
        "false_positives": spot_metrics["false_positives"],
        "false_negatives": spot_metrics["false_negatives"],
    }
