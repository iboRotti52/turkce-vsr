"""asr.py — platform-bağımsız ASR (kelime zaman damgalı).

Apple Silicon (M-serisi) -> mlx-whisper (Metal, en hızlı).
Windows / Linux / Intel Mac -> faster-whisper (CTranslate2; CUDA varsa GPU, yoksa CPU).

İkisi de aynı sözleşmeyi döner:  {"text": str, "words": [{"word","start","end"}, ...]}

Backend otomatik seçilir; test/zorlama için ASR_BACKEND=mlx|faster env değişkeni.
"""
from __future__ import annotations

import os
import platform

# Mantıksal model adı -> her backend'in kendi repo/ismi
_MLX = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}
_FW = {
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3": "large-v3",
}

_fw_model = None  # faster-whisper modelini süreç başına bir kez yükle
LAST_DEVICE = None  # son kullanılan cihaz ("cuda" / "cpu (...)") — teşhis için


def _backend() -> str:
    forced = os.environ.get("ASR_BACKEND")
    if forced:
        return forced
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "faster"


def transcribe_words(audio_path: str, model: str, language: str) -> dict:
    if _backend() == "mlx":
        return _mlx(audio_path, _MLX.get(model, model), language)
    return _faster(audio_path, _FW.get(model, model), language)


def _mlx(audio_path: str, repo: str, language: str) -> dict:
    import mlx_whisper
    r = mlx_whisper.transcribe(
        audio_path, path_or_hf_repo=repo, language=language, word_timestamps=True)
    words = [
        {"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
        for seg in r.get("segments", []) for w in seg.get("words", [])
    ]
    return {"text": r.get("text", ""), "words": words}


def _faster(audio_path: str, name: str, language: str) -> dict:
    global _fw_model
    from faster_whisper import WhisperModel
    if _fw_model is None:
        # CUDA varsa float16, yoksa CPU int8 (otomatik düşer).
        # Hangi cihaza düşüldüğü LAST_DEVICE'a yazılır — sessiz CPU fallback'i
        # (ör. cuDNN eksikse) fark edilmeden yavaş çalışmayı önlemek için.
        global LAST_DEVICE
        try:
            _fw_model = WhisperModel(name, device="cuda", compute_type="float16")
            LAST_DEVICE = "cuda"
        except Exception as e:
            _fw_model = WhisperModel(name, device="cpu", compute_type="int8")
            LAST_DEVICE = f"cpu (cuda basarisiz: {str(e)[:120]})"
    segments, _info = _fw_model.transcribe(
        audio_path, language=language, word_timestamps=True)
    text_parts, words = [], []
    for seg in segments:                       # generator -> burada koşar
        text_parts.append(seg.text)
        for w in (seg.words or []):
            words.append({"word": w.word, "start": float(w.start),
                          "end": float(w.end)})
    return {"text": "".join(text_parts), "words": words}
