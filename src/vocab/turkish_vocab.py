"""
src/vocab/turkish_vocab.py — Türkçe CTC Alfabesi ve Visem Haritası
"""

from typing import Dict, List, Optional, Set, Tuple
import torch

# 29 Türkçe Harf
TURKISH_LETTERS = [
    "a", "b", "c", "ç", "d", "e", "f", "g", "ğ", "h",
    "ı", "i", "j", "k", "l", "m", "n", "o", "ö", "p",
    "r", "s", "ş", "t", "u", "ü", "v", "y", "z"
]

SPECIAL_TOKENS = ["<blank>", "<pad>", "<unk>", " "]
PUNCTUATION_TOKENS = ["'"]

# Vocab listesi (indeks 0 blank CTC içindir)
VOCAB_LIST = SPECIAL_TOKENS + TURKISH_LETTERS + PUNCTUATION_TOKENS
BLANK_IDX = 0
PAD_IDX = 1
UNK_IDX = 2
SPACE_IDX = 3

CHAR_TO_IDX: Dict[str, int] = {char: idx for idx, char in enumerate(VOCAB_LIST)}
IDX_TO_CHAR: Dict[int, str] = {idx: char for idx, char in enumerate(VOCAB_LIST)}
VOCAB_SIZE = len(VOCAB_LIST)

# 8 Temel Türkçe Visem Grubu (Homophenes)
VISEME_MAP: Dict[str, int] = {
    # 0: Çift dudaksı (Bilabial)
    "b": 0, "p": 0, "m": 0,
    # 1: Diş-dudaksı (Labiodental)
    "f": 1, "v": 1,
    # 2: Dişeti / Diş (Alveolar/Dental)
    "d": 2, "t": 2, "s": 2, "z": 2, "n": 2,
    # 3: Damaksı (Post-alveolar / Palatal)
    "c": 3, "ç": 3, "j": 3, "ş": 3, "y": 3,
    # 4: Damak / Boğaz (Velar/Glottal)
    "k": 4, "g": 4, "ğ": 4, "h": 4,
    # 5: Sıvı / Titrek (Liquids)
    "l": 5, "r": 5,
    # 6: Yuvarlak ünlüler (Rounded vowels)
    "o": 6, "ö": 6, "u": 6, "ü": 6,
    # 7: Düz ünlüler (Unrounded vowels)
    "a": 7, "e": 7, "ı": 7, "i": 7,
}


def normalize_turkish_text(text: str) -> str:
    """
    Python'un standart lower() fonksiyonunun I -> i hatasını çözen
    Türkçe uyumlu metin normalizasyonu.
    """
    if not text:
        return ""
    
    # Özel Türkçe büyük-küçük harf dönüşümleri
    replacements = {
        "I": "ı",
        "İ": "i",
        "â": "a",
        "Â": "a",
        "î": "i",
        "Î": "i",
        "û": "u",
        "Û": "u",
        "ô": "o",
        "Ô": "o",
    }
    for upper, lower in replacements.items():
        text = text.replace(upper, lower)
    
    text = text.lower()
    
    # İzin verilen karakterler haricindeki sembolleri temizle
    allowed = set(TURKISH_LETTERS + [" ", "'"])
    cleaned = []
    prev_space = False
    for ch in text:
        if ch in allowed:
            if ch == " ":
                if not prev_space:
                    cleaned.append(ch)
                    prev_space = True
            else:
                cleaned.append(ch)
                prev_space = False
        else:
            if not prev_space:
                cleaned.append(" ")
                prev_space = True
                
    return "".join(cleaned).strip()


def text_to_indices(text: str) -> List[int]:
    """Türkçe metni CTC token indekslerine dönüştürür."""
    norm_text = normalize_turkish_text(text)
    indices = []
    for ch in norm_text:
        indices.append(CHAR_TO_IDX.get(ch, UNK_IDX))
    return indices


def indices_to_text(indices: List[int], remove_special: bool = True) -> str:
    """İndeks listesini okunabilir Türkçe metne çevirir."""
    chars = []
    for idx in indices:
        if idx in IDX_TO_CHAR:
            ch = IDX_TO_CHAR[idx]
            if remove_special and ch in ["<blank>", "<pad>", "<unk>"]:
                continue
            chars.append(ch)
    return "".join(chars).strip()


def ctc_greedy_decode(
    logits_or_indices: torch.Tensor,
    blank_penalty: float = 0.0,
) -> List[str]:
    """
    CTC Greedy Decoding: Tekrarlayan ardışık indeksleri ve blank (0) tokenlarını kaldırır.
    blank_penalty > 0 ise blank token logit'inden çıkarılarak harf emisyonu teşvik edilir.
    Girdi: (Batch, Time) veya (Batch, Time, Vocab)
    Çıktı: Her batch elemanı için deşifre edilmiş metin listesi.
    """
    if logits_or_indices.is_floating_point():
        logits = logits_or_indices.clone()
        if logits.dim() == 1:
            logits = logits.unsqueeze(0).unsqueeze(0)  # (1, 1, Vocab)
        elif logits.dim() == 2:
            logits = logits.unsqueeze(0)  # (1, T, Vocab)
        elif logits.dim() > 3:
            raise ValueError(f"Beklenen logits boyutu azami 3, alınan: {logits.dim()}")
        if blank_penalty > 0.0:
            logits[:, :, BLANK_IDX] -= blank_penalty
        indices = torch.argmax(logits, dim=-1)
    else:
        if logits_or_indices.dim() == 1:
            indices = logits_or_indices.unsqueeze(0)  # (1, T)
        elif logits_or_indices.dim() == 2:
            indices = logits_or_indices
        else:
            raise ValueError(f"Beklenen indeks tensörü boyutu 1 veya 2, alınan: {logits_or_indices.dim()}")

    decoded_strings = []
    batch_size = indices.size(0)
    for b in range(batch_size):
        row = indices[b].tolist()
        collapsed = []
        prev = -1
        for idx in row:
            if idx != prev:
                if idx != BLANK_IDX:
                    collapsed.append(idx)
                prev = idx
        decoded_strings.append(indices_to_text(collapsed))
    return decoded_strings


def to_viseme_sequence(text: str) -> List[int]:
    """Verilen metindeki harfleri visem sınıf indekslerine (0-7) çevirir."""
    norm = normalize_turkish_text(text)
    return [VISEME_MAP[ch] for ch in norm if ch in VISEME_MAP]
