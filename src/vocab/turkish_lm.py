"""
src/vocab/turkish_lm.py — Türkçe Dil Modeli (Unigram & Bigram Önsel Dağılımları)
Sürekli Türkçe Dudak Okuma (VSR) ve Prefix Beam Search için dilsel geçiş olasılıklarını sağlar.

Formülasyon:
  Score = log P_CTC + alpha * log P_LM(W) + beta * WordCount
  log P_LM(W) = sum_{k=1}^K log P(w_k | w_{k-1})
"""

import json
import math
import pathlib
from typing import Dict, List, Optional, Set, Tuple

from src.vocab.top500_words import TOP_500_SET, TOP_500_WORDS
from src.vocab.turkish_vocab import normalize_turkish_text

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
UNIGRAM_JSON_PATH = ROOT / "data" / "metadata" / "tokenizer" / "top500_unigrams.json"

# En yaygın Türkçe ikili sözcük geçişleri (Bigram Transisyonları) ve deneysel log-olasılıkları
# log(P(w2 | w1)) in [-3.0, -0.4]
TOP_BIGRAM_TRANSITIONS: Dict[Tuple[str, str], float] = {
    # İşaret Sıfatları & Belirteçler + bir
    ("bu", "bir"): -0.80,
    ("şu", "bir"): -1.30,
    ("o", "bir"): -1.10,
    ("her", "bir"): -1.20,
    ("başka", "bir"): -0.95,
    ("farklı", "bir"): -1.00,
    ("yeni", "bir"): -0.90,
    ("büyük", "bir"): -0.85,
    ("küçük", "bir"): -1.15,
    ("güzel", "bir"): -0.85,
    ("önemli", "bir"): -0.85,
    ("iyi", "bir"): -0.90,
    ("kötü", "bir"): -1.30,
    ("zor", "bir"): -1.15,
    ("kolay", "bir"): -1.25,
    ("nasıl", "bir"): -0.90,
    ("böyle", "bir"): -0.85,
    ("şöyle", "bir"): -1.10,
    ("öyle", "bir"): -0.95,
    ("birlikte", "bir"): -1.10,
    ("gibi", "bir"): -1.00,

    # bir + İsim / Zaman / Niteleme
    ("bir", "şey"): -0.80,
    ("bir", "tane"): -1.10,
    ("bir", "gün"): -1.15,
    ("bir", "de"): -0.95,
    ("bir", "iki"): -1.35,
    ("bir", "insan"): -1.25,
    ("bir", "kadın"): -1.35,
    ("bir", "adam"): -1.35,
    ("bir", "çocuk"): -1.30,
    ("bir", "dakika"): -1.45,
    ("bir", "saat"): -1.40,
    ("bir", "yıl"): -1.25,
    ("bir", "hafta"): -1.40,
    ("bir", "konu"): -1.25,
    ("bir", "yol"): -1.35,
    ("bir", "yer"): -1.20,
    ("bir", "zaman"): -1.25,
    ("bir", "arada"): -1.05,
    ("bir", "şekilde"): -0.90,
    ("bir", "araya"): -1.15,
    ("bir", "kısmı"): -1.35,
    ("bir", "biri"): -1.30,
    ("bir", "çok"): -1.05,

    # Derecelendirme Zarf ve Sıfatları (daha, çok, en)
    ("daha", "çok"): -0.70,
    ("daha", "fazla"): -0.85,
    ("daha", "iyi"): -0.80,
    ("daha", "büyük"): -0.95,
    ("daha", "kolay"): -1.15,
    ("daha", "zor"): -1.15,
    ("daha", "önce"): -0.90,
    ("daha", "sonra"): -0.80,
    ("daha", "az"): -1.20,
    ("daha", "doğru"): -1.25,
    ("çok", "daha"): -0.85,
    ("çok", "iyi"): -0.75,
    ("çok", "güzel"): -0.70,
    ("çok", "fazla"): -0.80,
    ("çok", "büyük"): -0.85,
    ("çok", "önemli"): -0.80,
    ("çok", "az"): -1.15,
    ("çok", "zor"): -1.05,
    ("çok", "kolay"): -1.20,
    ("çok", "teşekkür"): -0.95,
    ("çok", "farklı"): -1.05,
    ("çok", "doğru"): -1.15,
    ("en", "çok"): -0.75,
    ("en", "büyük"): -0.80,
    ("en", "iyi"): -0.75,
    ("en", "önemli"): -0.80,
    ("en", "fazla"): -0.95,
    ("en", "güzel"): -0.95,
    ("en", "son"): -0.90,
    ("en", "ilk"): -1.25,
    ("en", "zor"): -1.15,
    ("en", "kolay"): -1.20,
    ("en", "az"): -1.05,

    # Soru Edatları & Varlık/Yokluk
    ("var", "mı"): -0.55,
    ("yok", "mu"): -0.65,
    ("olur", "mu"): -0.75,
    ("olmaz", "mı"): -0.85,
    ("değil", "mi"): -0.45,
    ("öyle", "mi"): -0.70,
    ("böyle", "mi"): -0.80,
    ("musun", "mu"): -1.10,
    ("var", "dı"): -0.85,
    ("var", "dır"): -0.95,
    ("yok", "tu"): -0.95,
    ("yok", "tur"): -1.05,

    # İyelik / Zamir + için / Edat
    ("bunun", "için"): -0.65,
    ("onun", "için"): -0.70,
    ("benim", "için"): -0.75,
    ("senin", "için"): -0.85,
    ("bizim", "için"): -0.80,
    ("sizin", "için"): -0.90,
    ("onlar", "için"): -0.95,
    ("ne", "için"): -0.75,
    ("kendisi", "için"): -0.90,
    ("yapmak", "için"): -0.75,
    ("etmek", "için"): -0.80,
    ("olmak", "için"): -0.85,
    ("görmek", "için"): -0.95,
    ("almak", "için"): -1.05,
    ("vermek", "için"): -1.05,

    # Soru Kelimeleri (ne, nasıl)
    ("ne", "kadar"): -0.55,
    ("ne", "zaman"): -0.70,
    ("ne", "demek"): -0.75,
    ("ne", "oldu"): -0.80,
    ("ne", "yap"): -0.85,
    ("ne", "olur"): -0.95,
    ("ne", "var"): -0.85,
    ("nasıl", "oldu"): -0.90,
    ("nasıl", "yap"): -0.95,

    # Söylem Belirteçleri, Bağlaçlar & ki
    ("tabii", "ki"): -0.50,
    ("öyle", "ki"): -0.75,
    ("demek", "ki"): -0.60,
    ("diyor", "ki"): -0.65,
    ("dedi", "ki"): -0.70,
    ("biliyorum", "ki"): -0.85,
    ("aynı", "zamanda"): -0.55,
    ("ilk", "olarak"): -0.65,
    ("ilk", "defa"): -0.70,
    ("tam", "olarak"): -0.65,
    ("genel", "olarak"): -0.70,
    ("olarak", "da"): -0.75,
    ("olarak", "bir"): -0.80,
    ("dolayısıyla", "bu"): -0.75,
    ("dolayısıyla", "biz"): -0.85,
    ("dolayısıyla", "burada"): -0.95,
    ("veya", "başka"): -1.05,
    ("ya", "da"): -0.45,
    ("hem", "de"): -0.50,
    ("her", "zaman"): -0.65,
    ("her", "gün"): -0.80,
    ("her", "şey"): -0.70,
    ("her", "yerde"): -0.95,
    ("her", "hafta"): -1.05,
    ("hiç", "bir"): -0.55,
    ("hiç", "kimse"): -0.70,
    ("hiç", "yok"): -0.85,
    ("hiç", "değil"): -0.80,
    ("biraz", "daha"): -0.65,
    ("biraz", "önce"): -0.80,
    ("biraz", "sonra"): -0.80,
    ("çünkü", "bu"): -0.70,
    ("çünkü", "ben"): -0.75,
    ("çünkü", "o"): -0.80,
    ("çünkü", "biz"): -0.85,
    ("ama", "bu"): -0.65,
    ("ama", "ben"): -0.70,
    ("ama", "o"): -0.75,
    ("ama", "aslında"): -0.80,
    ("evet", "yani"): -0.75,
    ("evet", "tabii"): -0.80,
    ("hayır", "aslında"): -0.85,

    # Birleşik Fiil & Yardımcı Fiil Kalıpları
    ("devam", "ediyor"): -0.55,
    ("devam", "etti"): -0.65,
    ("devam", "edecek"): -0.75,
    ("kabul", "et"): -0.75,
    ("kabul", "etti"): -0.80,
    ("kabul", "etmek"): -0.85,
    ("teşekkür", "ederim"): -0.50,
    ("teşekkür", "ediyorum"): -0.55,
    ("rica", "ederim"): -0.55,
    ("fark", "etti"): -0.80,
    ("fark", "et"): -0.85,
    ("yardım", "et"): -0.80,
    ("merak", "etme"): -0.65,
    ("kendi", "kendini"): -0.60,
    ("kendi", "kendine"): -0.60,
    ("kendi", "hayatı"): -0.90,
    ("sahip", "olmak"): -0.65,
    ("sahip", "olduğu"): -0.70,
    ("sahip", "olan"): -0.75,
    ("sahip", "olduğunu"): -0.80,
    ("olduğu", "gibi"): -0.60,
    ("olduğu", "için"): -0.55,
    ("olduğu", "kadar"): -0.80,
    ("olduğu", "zaman"): -0.75,
    ("olduğunu", "söyledi"): -0.65,
    ("olduğunu", "düşünüyorum"): -0.70,
    ("olduğunu", "biliyorum"): -0.75,
    ("olduğunu", "gösteriyor"): -0.80,
    ("gerekiyor", "yani"): -0.75,
    ("yapması", "gerekiyor"): -0.70,
    ("olması", "gerekiyor"): -0.65,
    ("etmesi", "gerekiyor"): -0.75,
    ("türkiyede", "bunu"): -0.85,
    ("türkiyede", "ve"): -0.90,
    ("türkiyede", "çok"): -0.90,
    ("gerçekten", "çok"): -0.65,
    ("gerçekten", "büyük"): -0.75,
    ("gerçekten", "önemli"): -0.75,
    ("gerçekten", "iyi"): -0.80,
    ("gerçekten", "zor"): -0.85,
    ("düşünüyorum", "ki"): -0.65,
    ("düşünüyorum", "yani"): -0.70,
    ("biliyorum", "ki"): -0.70,
    ("biliyorum", "yani"): -0.75,
}


class TurkishLanguageModel:
    """
    Hedef 500 kelimelik Türkçe unigram ve bigram dil modeli.
    Prefix Beam Search çözücüsü için n-gram geçiş skorları üretir.
    """

    def __init__(
        self,
        unigram_dict: Optional[Dict[str, float]] = None,
        bigram_dict: Optional[Dict[Tuple[str, str], float]] = None,
        backoff_penalty: float = -1.2,
        default_unigram_score: float = -8.5,
    ):
        self.backoff_penalty = backoff_penalty
        self.default_unigram_score = default_unigram_score
        self.bigrams: Dict[Tuple[str, str], float] = dict(TOP_BIGRAM_TRANSITIONS)
        if bigram_dict:
            self.bigrams.update(bigram_dict)

        # Unigram sözlüğü yükleme (varsa dosyadan, yoksa parametreden)
        self.unigrams: Dict[str, float] = {}
        if unigram_dict:
            self.unigrams = dict(unigram_dict)
        elif UNIGRAM_JSON_PATH.exists():
            try:
                with open(UNIGRAM_JSON_PATH, "r", encoding="utf-8") as f:
                    self.unigrams = json.load(f)
            except Exception:
                self.unigrams = {}

        # Eğer unigram sözlüğü boşsa Zipf kuralına göre yaklaşık skorlar üret
        if not self.unigrams:
            self._init_fallback_unigrams()

    def _init_fallback_unigrams(self) -> None:
        """Dosya okunamadığında Zipf dağılımı log-olasılıkları."""
        for rank, w in enumerate(TOP_500_WORDS, start=1):
            # Rank 1: -3.74, Rank 500: ~ -8.50
            approx_score = -3.74 - math.log(rank) * 0.77
            self.unigrams[w] = round(approx_score, 4)

    def score_unigram(self, word: str) -> float:
        """Kelimenin unigram log-olasılığını döndürür."""
        clean = normalize_turkish_text(word)
        return self.unigrams.get(clean, self.default_unigram_score)

    def score_bigram(self, prev_word: str, curr_word: str) -> Optional[float]:
        """(prev_word, curr_word) ikilisinin bigram log-olasılığını döndürür, yoksa None döner."""
        w1 = normalize_turkish_text(prev_word)
        w2 = normalize_turkish_text(curr_word)
        return self.bigrams.get((w1, w2), None)

    def score_transition(self, prev_word: Optional[str], curr_word: str) -> float:
        """
        Geçiş skoru: log P(curr_word | prev_word).
        - prev_word None veya boş ise: unigram skoru.
        - (prev_word, curr_word) bilinen bir bigram ise: bigram skoru.
        - Aksi halde: unigram skoru + backoff cezası.
        """
        w2 = normalize_turkish_text(curr_word)
        if not prev_word:
            return self.score_unigram(w2)

        w1 = normalize_turkish_text(prev_word)
        bg = self.score_bigram(w1, w2)
        if bg is not None:
            return bg

        # Backoff: P(w2 | w1) ~ P(w2) * backoff_factor
        return self.score_unigram(w2) + self.backoff_penalty

    def score_sentence(self, words: List[str]) -> float:
        """Tüm kelime dizisinin toplam dil modeli log-skorunu hesaplar."""
        if not words:
            return 0.0
        total = 0.0
        prev = None
        for w in words:
            total += self.score_transition(prev, w)
            prev = w
        return total
