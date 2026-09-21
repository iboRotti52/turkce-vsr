"""config.py — yol ve sabitler. M2 Max yerel kurulumu için."""
import os
from pathlib import Path

# --- Yollar -------------------------------------------------------------------
# LIPREADING_*_DIR ortam değişkenleri: bulut çalıştırmasında (Modal) her worker'ın
# WORK/MASTER'ı paylaşılan bir Volume'a, state.db'yi ise yerel/geçici bir yola
# yönlendirmesini sağlar (paralel worker'ların aynı SQLite dosyasına eşzamanlı
# yazmasını önlemek için — bkz. modal_pipeline.py). Ortam değişkeni yoksa davranış
# eskisiyle birebir aynıdır.
BASE = Path(__file__).resolve().parent / "data"
WORK = Path(os.environ.get("LIPREADING_WORK_DIR", BASE / "work"))
MASTER = Path(os.environ.get("LIPREADING_MASTER_DIR", BASE / "master"))
MODELS = BASE / "models"    # indirilen model varlıkları (FaceLandmarker .task)
DB_PATH = Path(os.environ.get("LIPREADING_DB_PATH", BASE / "state.db"))
SOURCES_FILE = Path(__file__).resolve().parent / "sources.txt"

for d in (WORK, MASTER, MODELS):
    d.mkdir(parents=True, exist_ok=True)

# --- Yüz landmark modeli (MediaPipe Tasks API — yeni, desteklenen yol) --------
# Not: mediapipe'ın yeni sürümleri (özellikle Py>=3.13 wheel'leri) eski
# `mp.solutions.face_mesh` API'sini KALDIRDI; yalnız Tasks API var. Bu yüzden
# FaceLandmarker + .task model varlığı kullanılır (ilk çalıştırmada otomatik iner).
FACE_MODEL = MODELS / "face_landmarker.task"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

# --- Video/ses formatı (LRS / Auto-AVSR uyumlu) ------------------------------
FPS = 25
AUDIO_SR = 16000
FACE_SIZE = 224             # master tam yüz (RGB). Model-agnostik ana çıktı.

# --- Auto-AVSR uyumlu ağız kırpması (mouth.mp4) ------------------------------
# Auto-AVSR fine-tuning'i 96x96 GRİ, 68-noktalı ortalama yüze hizalanmış ağız
# ROI'si ister; tam yüz kabul etmiyor. Bu çıktı face.mp4'e EK olarak üretilir.
# Kaynak: video25.mp4'ün tam çözünürlüklü kareleri (face.mp4'ten türetmek
# büyütme olurdu — ölçtük: 224x224 içinde ağız yalnız ~57x19 piksel).
# ÖNEMLİ: yalnız bu ayardan SONRA işlenen videolarda üretilir; mevcut veriye
# dokunulmaz (eski faces.json'larda gereken `stable4` landmark'ları yok).
MOUTH_CROP = True
MOUTH_SIZE = 96

# --- Yüz çıkarımı hız/yük ayarı ----------------------------------------------
# MediaPipe landmark'ları normalize (0..1) döner; küçültülmüş kareden de aynı
# oranlar gelir. Bu yüzden mesh'e küçük kare verip koordinatı ORİJİNAL w,h ile
# ölçeklemek hızı artırır, doğruluğu (pratik olarak) bozmaz. Yerel yükü düşürür.
FACE_MESH_LONG_EDGE = 480   # mesh'e verilen karenin uzun kenarı (px)

# --- ASR (platform-bağımsız; bkz. asr.py) ------------------------------------
# Mantıksal model: "turbo" (hızlı) veya "large-v3" (yüksek doğruluk).
# Backend otomatik: Apple Silicon -> mlx-whisper (Metal); Windows/Linux/Intel
# Mac -> faster-whisper (CUDA varsa GPU, yoksa CPU). Bkz. asr.py
ASR_MODEL = os.environ.get("ASR_MODEL", "turbo")
ASR_LANG = "tr"

# --- Segmentasyon (LRS3 kuralı) ----------------------------------------------
MAX_CLIP_SEC = 6.0
MAX_CLIP_CHARS = 100
MIN_CLIP_SEC = 1.0

# --- Kalite / dublaj filtresi -------------------------------------------------
MIN_FACE_PX = 100           # min yüz kenarı
# Dublaj kontrolü artık SEGMENT bazında: her segmentin kendi ağız-açıklığı <-> ses
# enerjisi korelasyonu. Eşik altı = o segment muhtemel dublaj/sessizlik -> ele
# (videoyu değil). Kısa+temiz segmentte korelasyon yükselir; eşiği pilot
# sonuçlarına (audit raporundaki dağılıma) bakarak kalibre et.
DUB_CORR_THRESHOLD = 0.15
MIN_SEG_DUB_FRAMES = 5      # korelasyon için segmentte gereken min kare

# --- Türkçe alfabe (normalizasyon) -------------------------------------------
TR_ALPHABET = set("abcçdefgğhıijklmnoöprsştuüvyz ")

# --- Konuşmacı kimliği (split sızıntısını önlemek için) -----------------------
# splits.py "aynı konuşmacı hem train hem test'te olmasın" garantisi verir.
# Varsayılan kural KANAL bazında gruplamaktır (muhafazakâr: aynı kanalın tüm
# videoları tek split'e gider) — çünkü kişisel kanallarda tüm videolar aynı
# kişidir (ör. Onur Tirpan, Eren Aktan).
#
# İSTİSNA: "toplayıcı" kanallarda her video FARKLI bir konuşmacıdır (TEDx gibi).
# Bunları kanal bazında gruplamak, 27 videoyu tek split'e tıkar ve veri setini
# kullanılamaz hale getirir. Bu yüzden aşağıdaki kanallarda video bazında ayrılır.
AGGREGATOR_CHANNELS = {
    "TEDx Talks",
}

# Aynı konuşmacı FARKLI kanal adları altında olabilir — o zaman kanal bazlı
# gruplama sızıntıyı yakalayamaz. Ölçtük: "AK PARTİ" ve "AK PARTi Genel Merkez"
# (büyük İ / küçük i farkı) aynı konuşmacının fahri doktora törenleri.
# Buraya eklenen kanallar tek bir konuşmacı kimliğinde birleştirilir.
CHANNEL_ALIASES = {
    "AK PARTİ": "spk:erdogan",
    "AK PARTi Genel Merkez": "spk:erdogan",
}
