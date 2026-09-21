"""
src/vocab/top500_words.py — Genel Türkçede En Çok Kullanılan 500 Kelime ve Visem İmzaları
Genel Türkçe konuşma dilinde en yüksek frekansa sahip 500 otantik Türkçe kelimeyi içerir.
Spesifik veri kümesine özgü parçalanmış ekler veya video terimleri elenmiştir.
"""

from typing import Dict, List, Set
from src.vocab.turkish_vocab import normalize_turkish_text, to_viseme_sequence

# Genel Türkçede en sık kullanılan 500 tam kelime (len >= 2, kanonik sözlük sırası)
TOP_500_WORDS: List[str] = [
    "bir", "bu", "ne", "ve", "için", "mi", "de", "ben", "çok", "ama",
    "evet", "var", "da", "mı", "değil", "şey", "iyi", "hayır", "daha", "sen",
    "kadar", "bana", "gibi", "bunu", "yok", "onu", "tamam", "beni", "seni", "benim",
    "her", "sana", "ki", "neden", "sadece", "zaman", "senin", "burada", "olduğunu", "nasıl",
    "hiç", "sonra", "şimdi", "en", "öyle", "mu", "şu", "misin", "önce", "biraz",
    "hadi", "güzel", "musun", "oldu", "yani", "ona", "böyle", "işte", "onun", "bile",
    "lütfen", "bak", "eğer", "peki", "çünkü", "artık", "gerçekten", "istiyorum", "geri", "biliyorum",
    "kim", "başka", "iki", "olarak", "belki", "tek", "doğru", "büyük", "olan", "biri",
    "bay", "buraya", "olur", "adam", "ile", "olacak", "hiçbir", "biz", "demek", "sanırım",
    "yardım", "bilmiyorum", "bunun", "teşekkürler", "hala", "tüm", "gün", "yeni", "fazla", "ederim",
    "nerede", "tanrım", "merhaba", "efendim", "şeyi", "orada", "gece", "son", "kötü", "şeyler",
    "pekala", "biliyor", "harika", "sorun", "et", "bütün", "tam", "ilk", "gerek", "siz",
    "diye", "hemen", "ol", "üzgünüm", "küçük", "olabilir", "iş", "olsun", "size", "bayan",
    "aynı", "hakkında", "teşekkür", "tabii", "gel", "yüzden", "onları", "izin", "kız", "kendi",
    "devam", "bize", "ver", "oluyor", "bizi", "buna", "içinde", "selam", "anne", "vardı",
    "bizim", "geldi", "sizi", "baba", "dur", "aslında", "seninle", "asla", "göre", "kimse",
    "özür", "değilim", "önemli", "git", "tekrar", "yoksa", "benimle", "mısın", "mü", "yine",
    "bence", "haydi", "şekilde", "olmak", "bugün", "söyle", "lazım", "birlikte", "para", "ister",
    "olmaz", "uzun", "onunla", "bakalım", "zaten", "biliyorsun", "al", "geliyor", "dostum", "üç",
    "dakika", "onlar", "gerçek", "emin", "işe", "birkaç", "herkes", "saat", "söyledi", "istiyorsun",
    "pek", "yapıyorsun", "eve", "miyim", "yıl", "çocuk", "hep", "şunu", "neler", "yer",
    "dilerim", "sizin", "bekle", "bırak", "istemiyorum", "olduğu", "ilgili", "gidelim", "nedir", "yere",
    "etmek", "lanet", "olmalı", "kaç", "nereye", "hepsi", "buradan", "an", "insanlar", "burası",
    "az", "karşı", "çocuklar", "işi", "eski", "hazır", "kendini", "gerekiyor", "ediyorum", "fakat",
    "kesinlikle", "benden", "kadın", "söz", "yapmak", "anda", "tane", "zor", "diğer", "dinle",
    "bunlar", "görmek", "istiyor", "kişi", "niye", "insan", "kabul", "seviyorum", "aman", "misiniz",
    "yarın", "zorunda", "tanrı", "bundan", "kez", "sağ", "elbette", "gidip", "içeri", "falan",
    "tatlım", "yanlış", "hoş", "hem", "senden", "onlara", "doktor", "hafta", "özel", "yemek",
    "yapma", "kontrol", "oraya", "gereken", "yerde", "ediyor", "geç", "mutlu", "bazı", "dedim",
    "birini", "ondan", "eminim", "ayrıca", "dışarı", "bakın", "nereden", "ihtiyacım", "gitti", "beri",
    "edin", "etti", "yalnız", "erkek", "yalan", "merak", "gördüm", "değildi", "polis", "görüşürüz",
    "dün", "musunuz", "babam", "haber", "gidiyor", "olmuş", "akşam", "geçen", "sakin", "sanki",
    "nefret", "sabah", "dikkat", "beş", "karar", "umarım", "yaptım", "ay", "veya", "çabuk",
    "etme", "yerine", "istediğim", "farklı", "sonunda", "üzerinde", "olması", "gördün", "sence", "değilsin",
    "aptal", "kendimi", "ettim", "tamamen", "oldukça", "kendine", "fark", "yeter", "işin", "boyunca",
    "vardır", "bunları", "olduğumu", "tarafından", "kes", "iyiyim", "annem", "öldü", "neyse", "olursa",
    "açık", "dünya", "adı", "yap", "hangi", "söyledim", "yaptın", "dört", "aldım", "dolar",
    "olamaz", "görünüyor", "birisi", "yüzünden", "olsa", "kolay", "yoktu", "bende", "adamı", "beraber",
    "konuşmak", "bazen", "buldum", "dedi", "halde", "ise", "ancak", "yanında", "sahip", "olmadığını",
    "yol", "gelecek", "yapıyor", "başına", "gitmek", "bitti", "geldim", "tabi", "evde", "henüz",
    "onların", "anlıyorum", "cevap", "yolu", "zamanı", "genç", "öyleyse", "almak", "yardımcı", "kahretsin",
    "oldum", "konusunda", "kal", "günü", "adamın", "takip", "ateş", "durum", "söylemek", "azından",
    "adım", "eder", "düşündüm", "istedim", "neredeyse", "muhtemelen", "düşünüyorsun", "mükemmel", "yaptı", "ciddi",
    "sanmıyorum", "olun", "ev", "herhangi", "yeniden", "boş", "altında", "on", "kan", "ara",
    "göz", "düşünüyorum", "hızlı", "süre", "miyiz", "oğlum", "aç", "uzak", "duydum", "konuda",
    "sakın", "şeyin", "gelen", "çıktı", "saniye", "garip", "araba", "gidiyorum", "yakın", "arada",
    "olduğuna", "insanların", "hayatta", "anladım", "çalışıyorum", "hayal", "söylüyor", "hayatım", "galiba", "kaldı",
    "rahat", "canım", "sürü", "ikinci", "kimin", "bilirsin", "olacağım", "dersin", "hepimiz", "adamım",
    "bebeğim", "ortaya", "verdi", "başladı", "hayat", "gelip", "şeye", "çalışıyor", "sende", "yaptığını",
    "güçlü", "olay", "hatta", "durumda", "arkadaşım", "bebek", "çocuğu", "savaş", "şeyleri", "kesin",
]

TOP_500_SET: Set[str] = set(TOP_500_WORDS)
WORD_TO_RANK: Dict[str, int] = {w: idx + 1 for idx, w in enumerate(TOP_500_WORDS)}


def is_top500_word(word: str) -> bool:
    """Verilen kelimenin hedef 500 kelime listesinde olup olmadığını kontrol eder."""
    clean = normalize_turkish_text(word)
    return clean in TOP_500_SET


def get_word_rank(word: str) -> int:
    """Kelimenin frekans sırasını (1-500) döndürür, listede yoksa -1 döner."""
    clean = normalize_turkish_text(word)
    return WORD_TO_RANK.get(clean, -1)


def get_word_viseme_signature(word: str) -> str:
    """
    Kelimenin visem imzasını döndürür (örneğin "ben" -> "0-7-2").
    Dudakta aynı görünen kelimeleri (homophenes) eşleştirmek için kullanılır.
    """
    visemes = to_viseme_sequence(word)
    return "-".join(map(str, visemes))
