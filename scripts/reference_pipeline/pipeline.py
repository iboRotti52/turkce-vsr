"""
pipeline.py — idempotent aşamalar. Her aşama çıktısı zaten varsa ATLAR.
Bu sayede worker/laptop/n8n çökse bile kaldığı yerden devam eder.

Aşamalar:  download -> normalize -> faces -> asr -> (dub check) -> segment -> crop -> done
Master çıktı:  data/master/<video_id>/<seg_id>/{face.mp4, audio.wav, align.json, meta.json}
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import state
from config import (
    ASR_LANG, ASR_MODEL, AUDIO_SR, DUB_CORR_THRESHOLD, FACE_MESH_LONG_EDGE,
    FACE_MODEL, FACE_MODEL_URL, FACE_SIZE, FPS, MASTER, MAX_CLIP_CHARS,
    MAX_CLIP_SEC, MIN_CLIP_SEC, MIN_FACE_PX, MIN_SEG_DUB_FRAMES, MOUTH_CROP, MOUTH_SIZE,
    TR_ALPHABET, WORK,
)


@contextmanager
def _timed(timings: dict, stage: str):
    """Aşama süresini saniye cinsinden biriktirir. Amaç: T4'ün gerçekten
    meşgul olduğu oranı ölçmek — yalnız `asr` GPU kullanıyor, kalan aşamalar
    (download/normalize/faces/crop) T4 kirada dururken CPU'da koşuyor.
    Atlanmış (idempotent) aşama 0.0 yazar, bu da bilgidir."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round(timings.get(stage, 0.0) + time.perf_counter() - t0, 2)


def video_id_from_url(url: str) -> str:
    """YouTube watch?v= veya youtu.be/ID'den video id çıkar (platform-bağımsız)."""
    from urllib.parse import parse_qs, urlparse
    u = urlparse(url)
    if u.hostname and "youtu.be" in u.hostname:
        return u.path.lstrip("/")
    return parse_qs(u.query).get("v", [url])[0]


def _run(cmd: list[str]):
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("\n" + "="*50)
        print(f"HATA YAPAN KOMUT: {' '.join(cmd)}")
        print(f"HATA DETAYI:\n{e.stderr}")
        print("="*50 + "\n")
        # CalledProcessError'in str()'i yalnizca exit kodunu tasir; sonuc
        # JSON'ina str(e)[:300] yazildigi icin GERCEK sebep kayboluyordu
        # (15 indirme hatasinin bot-tespiti mi, silinmis video mu oldugunu
        # bu yuzden anlayamadik). Sebebi mesaja goc ettir.
        kuyruk = (e.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError(f"{cmd[0]} basarisiz (kod {e.returncode}): "
                           + " | ".join(kuyruk)) from e


def vdir(video_id: str) -> Path:
    d = WORK / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 1) İNDİRME
# ---------------------------------------------------------------------------
def _video_akisi_var(f: Path) -> bool:
    """Dosyada gercekten VIDEO akisi var mi?

    yt-dlp `bv*+ba` ile video ve sesi ayri indirip birlestiriyor. Birlestirme
    yarida kalirsa (ornek: YouTube "page needs to be reloaded" engellemesi)
    elde sesi olan ama videosu olmayan bir dosya kaliyor. Boyut kontrolu bunu
    yakalamiyordu: dosya megabaytlarca, ama icinde kare yok.

    Zinciri sessizce zehirliyordu: ASR sesi bulup yuzlerce segment uretiyor,
    video25.mp4 bos cikiyor, yuz asamasi 3 saniyede bitiyor, her segment
    "yuz_yok" diye eleniyor -- ve bozuk dosya onbellekte KALICI oldugu icin
    sonraki her kosu ayni sonucu veriyor. (Olculdu 2026-08-23: 4dXrsnRYRwI,
    C4rqS1EgwDI, AtRVoB0McBs -- ucu de 0 kabul, faces asamasi ~3sn.)
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(f)],
            capture_output=True, text=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return False
    return "video" in r.stdout


def _sure(f: Path) -> float:
    """Medya suresi (sn). Okunamazsa -1."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(f)],
            capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        return -1.0


def download(video_id: str, url: str):
    d = vdir(video_id)
    src = next(d.glob("src.*"), None)
    if src and src.stat().st_size > 0:
        if _video_akisi_var(src):
            return  # resume: zaten indirilmiş
        # Bozuk kalinti: sil, yeniden indirilsin. Turetilmis dosyalar da
        # bu kaynaktan uretildigi icin onlar da gitmeli.
        print(f"BOZUK KALINTI siliniyor: {src}")
        src.unlink(missing_ok=True)
        for tur in ("video25.mp4", "audio16k.wav", "faces.json", "align.json"):
            (d / tur).unlink(missing_ok=True)
    cmd = ["yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
           "-o", str(d / "src.%(ext)s")]
    # YouTube'un JS "n-challenge"i icin cozucu script gerekiyor; yt-dlp bunu
    # varsayilan olarak indirmiyor ("Remote components ... were skipped").
    # Bu bayrak olmadan sadece gorseller listeleniyor -> "Requested format is
    # not available" hatasi (bulut IP'lerinde olcduk).
    if os.environ.get("YTDLP_REMOTE_COMPONENTS"):
        cmd += ["--remote-components", os.environ["YTDLP_REMOTE_COMPONENTS"]]
    # YTDLP_COOKIES_FILE ayarlıysa ekle (bulut IP'lerinde YouTube bot-tespitini
    # aşmak için gerekli — bkz. modal_pipeline.py). Yerelde ortam değişkeni
    # yoksa davranış eskisiyle birebir aynı.
    cookies_file = os.environ.get("YTDLP_COOKIES_FILE")
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    cmd.append(url)
    _run(cmd)


# ---------------------------------------------------------------------------
# 2b) TRİYAJ — pahalı aşamalardan ÖNCE "bu videoda okunabilir dudak var mı?"
# ---------------------------------------------------------------------------
# Olculdu (196 videoluk kosu): videolarin %48'i geniş plan konferans kaydi,
# yuz 1920px'lik karede 8-60px. Bunlar MIN_FACE_PX'i gecemedigi icin SIFIR
# segment veriyor -- ama once ASR (%34) ve yuz cikarma (%22) tam fiyat
# odeniyor. Yani butcenin yaklasik yarisi kullanilamaz videoya gidiyor.
# Birkac kare orneklemek bunu ~%1'e indiriyor.
TRIAJ_KARE = 40          # kac kare orneklenir
TRIAJ_PAY = 0.8          # MIN_FACE_PX'e tolerans (ornekleme sansini telafi eder)


def triaj(video_id: str) -> tuple[bool, float]:
    """(gecti_mi, olculen_en_buyuk_yuz_px). video25.mp4 uzerinde calisir.

    !!! BORU HATTINDA KULLANILMIYOR — bilerek. 196 videoda olculdu
    (triaj_degerlendir.py, triaj_tarama.json):
        20 kare  -> %14 tasarruf, %17.5 veri kaybi
        300 kare -> %14 tasarruf, %2.7 veri kaybi
    Tavan 27-32 video; ustune cikmiyor cunku seyrek yakin plan iceren
    videolar duzgun ornekleme ile de kaciriliyor. Esigin (40-100px) hicbir
    etkisi yok: ayrim "yuz buyuk mu" degil, "yuz BULUNDU mu". Yani %2.7
    kalici veri kaybi karsiliginda %14 hesaplama -- kotu takas, eklenmedi.
    Fonksiyon teshis icin duruyor; yeniden degerlendirmek istersen
    triaj_degerlendir.py'yi calistir.

    Kacirma riski: kullanilabilir kare orani p ise P(kacirma) = (1-p)^40.
    p=0.10 -> %1.5, p=0.05 -> %13. Yakin plani hic olmayan videoyu elemek
    istiyoruz zaten; kaybi olan, sadece %5'ten az yakin plan iceren video."""
    import cv2
    d = vdir(video_id)
    cap = cv2.VideoCapture(str(d / "video25.mp4"))
    toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if toplam <= 0:
        cap.release()
        return True, -1.0          # olcemedik -> ELEME (kapi sadece kesin eler)
    _ensure_face_model()
    lm, mp = _face_landmarker()
    en_buyuk, ts = 0.0, 0
    for k in range(TRIAJ_KARE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(toplam * (k + 0.5) / TRIAJ_KARE))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        rgb = cv2.cvtColor(_small(fr), cv2.COLOR_BGR2RGB)
        ts += 1000
        res = lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if res.face_landmarks:
            pts = res.face_landmarks[0]
            kenar = max(max(q.x for q in pts) - min(q.x for q in pts),
                        max(q.y for q in pts) - min(q.y for q in pts))
            en_buyuk = max(en_buyuk, kenar * max(h, w))
            if en_buyuk >= MIN_FACE_PX:      # yeterli kanit, erken cik
                break
    cap.release()
    return en_buyuk >= MIN_FACE_PX * TRIAJ_PAY, round(en_buyuk, 1)


# ---------------------------------------------------------------------------
# 2) NORMALİZE (25fps video + 16kHz mono ses)
# ---------------------------------------------------------------------------
def normalize(video_id: str):
    d = vdir(video_id)
    v25, a16 = d / "video25.mp4", d / "audio16k.wav"
    src = next(d.glob("src.*"))
    # Yarida kesilmis bir donusturmenin biraktigi gudUk video25.mp4, sadece
    # exists() bakildigi icin gecerli sayiliyordu. Sonucu: ASR sesi bulup
    # yuzlerce segment uretiyor ama yuz asamasi 3 saniyede bitiyor ve her
    # segment "yuz_yok" diye eleniyor -- kalici, cunku dosya onbellekte.
    # (Olculdu 2026-08-23: 4dXrsnRYRwI, AtRVoB0McBs, C4rqS1EgwDI.)
    # Kontrol: turetilmis dosyanin suresi kaynaginkiyle tutmali.
    if v25.exists():
        k, t = _sure(src), _sure(v25)
        if t < 0 or (k > 0 and t < 0.9 * k):
            print(f"BOZUK video25.mp4 siliniyor ({t:.1f}sn / kaynak {k:.1f}sn)")
            v25.unlink(missing_ok=True)
            (d / "faces.json").unlink(missing_ok=True)
    if not v25.exists():
        _run(["ffmpeg", "-y", "-i", str(src), "-r", str(FPS), "-an", str(v25)])
    if not a16.exists():
        _run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(AUDIO_SR),
              "-vn", str(a16)])


# ---------------------------------------------------------------------------
# 3) YÜZ + LANDMARK (MediaPipe, CPU — Mac'te rahat)
#    faces.json: {frame_idx: {t, bbox[x,y,w,h], mouth_open, mouth_cx, mouth_cy}}
#    YALNIZ segment zaman aralıklarındaki kareler işlenir (yerel yükü düşürür);
#    geri kalan kareler decode edilmeden grab() ile atlanır. Mesh'e küçültülmüş
#    kare verilir, koordinat ORİJİNAL w,h ile ölçeklenir (landmark'lar normalize).
# ---------------------------------------------------------------------------
def _needed_frames(segs: list[dict]) -> set[int]:
    needed = set()
    for s in segs:
        f0 = int(round(s["start"] * FPS))
        f1 = int(round(s["end"] * FPS))
        needed.update(range(f0, f1 + 1))
    return needed


def _small(frame):
    import cv2
    h, w = frame.shape[:2]
    scale = FACE_MESH_LONG_EDGE / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))


def _ensure_face_model():
    """FaceLandmarker .task varlığını ilk çalıştırmada indir (idempotent)."""
    if FACE_MODEL.exists() and FACE_MODEL.stat().st_size > 0:
        return
    import urllib.request
    FACE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)


def _face_landmarker():
    """MediaPipe Tasks FaceLandmarker (VIDEO modu). Eski solutions API'si
    yeni wheel'lerde (Py>=3.13) yok; bu desteklenen yoldur."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts), mp


def extract_faces(video_id: str, segs: list[dict]):
    d = vdir(video_id)
    out = d / "faces.json"
    import cv2

    needed = _needed_frames(segs)
    if not needed:
        out.write_text("{}")
        return
    # Kendini onaran idempotency: faces.json mevcut SEGMENTLERİN karelerini
    # gerçekten kapsıyorsa atla. Bayat/eksik/boş (crash veya re-segment sonrası
    # uyumsuz) ise yeniden çıkar. Eski "varsa atla" mantığı uyumsuz dosyayı asla
    # yenilemiyordu -> video 'done' ama 0 segment ("sorunlu") kalıyordu.
    if out.exists():
        try:
            mevcut = json.loads(out.read_text())
            have = {int(k) for k in mevcut}
        except Exception:
            mevcut, have = {}, set()
        kapsam_ok = have and len(needed & have) >= 0.95 * len(needed)
        # MOUTH_CROP açıkken `stable4` da ŞART: eski faces.json'larda bu alan yok
        # (ağız hizalaması sonradan eklendi). Yalnız kapsama bakıp atlarsak,
        # geriye dönük ağız kırpması hiç üretilemez. Yüz bulunan bir karede
        # stable4 yoksa dosya BAYAT sayılır ve yeniden çıkarılır.
        stable4_ok = True
        if MOUTH_CROP and kapsam_ok:
            yuzlu = [v for v in mevcut.values() if v.get("bbox")]
            stable4_ok = bool(yuzlu) and any(v.get("stable4") for v in yuzlu)
        if kapsam_ok and stable4_ok:
            return
    max_needed = max(needed)

    _ensure_face_model()
    landmarker, mp = _face_landmarker()
    cap = cv2.VideoCapture(str(d / "video25.mp4"))
    faces: dict[str, dict] = {}
    idx = 0
    while idx <= max_needed:
        if idx not in needed:
            if not cap.grab():        # konuşma dışı kare: decode etme, atla
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]         # ORİJİNAL çözünürlük (koordinat tabanı)
        rgb = cv2.cvtColor(_small(frame), cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_img, int(idx / FPS * 1000))
        rec = {"t": idx / FPS, "bbox": None, "mouth_open": 0.0,
               "mouth_cx": None, "mouth_cy": None}
        if res.face_landmarks:
            lm = res.face_landmarks[0]    # normalize landmark listesi (478 nokta)
            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            fh = max(1.0, y1 - y0)
            # iç dudak landmark'ları: 13 (üst), 14 (alt) — topoloji aynı
            mouth_open = abs(lm[14].y - lm[13].y) * h / fh
            rec["bbox"] = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
            rec["mouth_open"] = float(mouth_open)
            rec["mouth_cx"] = float((lm[13].x + lm[14].x) / 2 * w)
            rec["mouth_cy"] = float((lm[13].y + lm[14].y) / 2 * h)
            # Auto-AVSR ağız kırpması için 4 "stable" nokta — SIRA ÖNEMLİ:
            # sağ göz, sol göz, burun ucu, ağız merkezi. (video_process.py
            # bunları 68 noktalı ortalama-yüze affine hizalar; stable_points=(0,1,2,3))
            # ORİJİNAL video koordinatlarında saklanır ki tam çözünürlükten kırpabilelim.
            rec["stable4"] = [
                [float((lm[33].x + lm[133].x) / 2 * w),      # sağ göz (dış+iç köşe ort.)
                 float((lm[33].y + lm[133].y) / 2 * h)],
                [float((lm[362].x + lm[263].x) / 2 * w),     # sol göz
                 float((lm[362].y + lm[263].y) / 2 * h)],
                [float(lm[1].x * w), float(lm[1].y * h)],    # burun ucu
                [float((lm[13].x + lm[14].x) / 2 * w),       # ağız merkezi
                 float((lm[13].y + lm[14].y) / 2 * h)],
            ]
        faces[str(idx)] = rec
        idx += 1
    cap.release()
    landmarker.close()
    out.write_text(json.dumps(faces))


def _load_faces(video_id: str) -> dict[int, dict]:
    d = vdir(video_id)
    raw = json.loads((d / "faces.json").read_text())
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# 4) ASR + KELİME ZAMAN DAMGASI (mlx-whisper, Metal)
#    WhisperX'e gerek YOK — mlx-whisper kelime zamanını kendisi verir.
# ---------------------------------------------------------------------------
def transcribe(video_id: str):
    d = vdir(video_id)
    out = d / "align.json"
    if out.exists():
        return
    from asr import transcribe_words

    res = transcribe_words(str(d / "audio16k.wav"), ASR_MODEL, ASR_LANG)
    words = [{"word": _norm_tr(w["word"]),
              "start": float(w["start"]), "end": float(w["end"])}
             for w in res["words"]]
    out.write_text(json.dumps({"text": res["text"], "words": words},
                              ensure_ascii=False))


def _norm_tr(w: str) -> str:
    w = w.strip().lower()
    return "".join(ch for ch in w if ch in TR_ALPHABET or ch.isalpha())


# ---------------------------------------------------------------------------
# 5) DUBLAJ / VOICE-OVER KONTROLÜ — SEGMENT BAZINDA (TalkNet yerine hafif heuristik)
#    Her segmentin kendi ağız-açıklığı <-> ses-enerjisi korelasyonu. Eşik altı
#    segment elenir (videonun tamamı değil). Tüm-video korelasyonu çok-sahneli
#    kaynakta iyi konuşmayı da reddettiği için bu yaklaşım terk edildi.
# ---------------------------------------------------------------------------
def _corr(faces: dict[int, dict], audio: np.ndarray, sr: int,
          t0: float, t1: float) -> tuple[float, int]:
    """[t0,t1) aralığında ağız-açıklığı ile ses enerjisinin korelasyonu.
    (skor, kullanılan kare sayısı) döner. n'i de döndürüyoruz çünkü bir
    korelasyonun anlamlı olup olmadığı yalnız n ile birlikte söylenebilir."""
    f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    mouth = np.array(
        [faces[i]["mouth_open"] if (i in faces and faces[i]["bbox"]) else 0.0
         for i in range(f0, f1)], dtype=np.float32)
    # Yuz bulunamayan kareyi 0.0 sayip korelasyona SOKMAK, sesle hicbir
    # iliskisi olmayan sabit bir sinyal eklemek demek: korelasyonu sistematik
    # olarak sifira dogru seyreltir. Yuzun %50 karede goruldugu, dublajsiz bir
    # video bu yuzden esigin altina dusup "dublajli" damgasi yiyordu.
    # Dogrusu: eksik kareyi sifirlamak degil, hesabin disinda birakmak.
    gecerli = np.array(
        [bool(i in faces and faces[i]["bbox"]) for i in range(f0, f1)])

    seg_audio = audio[int(t0 * sr): int(t1 * sr)]
    hop = max(1, int(sr / FPS))
    n_kare = min(len(mouth), len(seg_audio) // hop)
    maske = gecerli[:n_kare]
    n = int(maske.sum())
    if n < MIN_SEG_DUB_FRAMES:
        return float("nan"), n
    rms = np.array([np.sqrt(np.mean(seg_audio[i*hop:(i+1)*hop]**2) + 1e-9)
                    for i in range(n_kare)], dtype=np.float32)[maske]
    m = mouth[:n_kare][maske]
    if m.std() < 1e-6 or rms.std() < 1e-6:
        # ponytail: nan = "hesaplanamadi", 0.0 = "korelasyon gercekten sifir".
        # Ayni degeri ikisine de dondurursek cagiran taraf ayirt edemez;
        # video seviyesinde tam bu yuzden 27 video yanlislikla "dublajli"
        # damgasi yedi (yuzu bulunamayan video, dublaj sanildi).
        return float("nan"), n
    return float(np.corrcoef(m, rms)[0, 1]), n


# Gercek konusmada olculen video-seviyesi korelasyon ~0.04. Bunu gurultuden
# 3σ ile ayirabilmek icin 3/sqrt(n) < 0.04, yani n > 5625 GECERLI kare gerekir.
# Daha az karede dusuk skor "dublaj kaniti" degil, "kanit yoklugu"dur: videoyu
# elemek yerine segment kapilarina birakiriz, onlar yuzsuz segmenti zaten eler.
DUB_MIN_N = 5625

def dub_esigi(n: int) -> float:
    """n örnekten hesaplanmış bir korelasyonun gürültüden ayrılma sınırı (3σ).

    Bağımsız iki seride korelasyonun standart sapması ≈ 1/sqrt(n). ÖLÇTÜK
    (2026-08-22): segmentler 1-6 sn = 25-150 kare, yani null σ ≈ 0.12 ve
    sabit 0.15 eşiği saf gürültünün üst %12'sini seçiyordu — TEDx gibi
    dublajsız içerikte bile segmentlerin %80'i "dublajlı" diye eleniyordu.
    Aynı hesap video boyunca (on binlerce kare) yapılınca σ ≈ 0.006'ya
    düşüyor ve karar güvenilir hâle geliyor. Sabit eşik yerine anlamlılık:
    örnek sayısı yetersizse eşik kendiliğinden erişilemez olur."""
    return 3.0 / np.sqrt(n) if n >= MIN_SEG_DUB_FRAMES else 1.0


def dub_karari(skor: float, n: int) -> str:
    """Video-seviyesi dublaj karari: "elendi" | "gecti" | "karar_yok".

    Uc ayri durum, uc ayri cevap. Bu ayrimi yapmamak bize pahaliya mal oldu:
      - skor hesaplanamadi (nan, hic yuz yok)            -> karar_yok
      - yeterli gecerli kare yok (n < DUB_MIN_N)         -> karar_yok
      - kanit var ve korelasyon gurultuden ayrilmiyor    -> elendi
    "karar_yok" ELEME DEGILDIR; segment kapilari isi devralir."""
    if np.isnan(skor) or n < DUB_MIN_N:
        return "karar_yok"
    return "elendi" if skor < dub_esigi(n) else "gecti"


def video_dub_score(faces: dict[int, dict], audio: np.ndarray,
                    sr: int) -> tuple[float, int]:
    """Dublaj kararı BURADA verilir — tüm video, n on binlerce kare."""
    return _corr(faces, audio, sr, 0.0, len(audio) / sr)


def seg_dub_score(faces: dict[int, dict], audio: np.ndarray, sr: int,
                  s: dict) -> float:
    """Geriye dönük uyumluluk / teşhis için. Kabul kararında KULLANILMAZ:
    segment n'i (25-150) bu korelasyonu anlamlı kılmaya yetmiyor."""
    return _corr(faces, audio, sr, s["start"], s["end"])[0]


# ---------------------------------------------------------------------------
# 6) CÜMLE SEGMENTASYONU (LRS3 kuralı: noktalama, <=6sn / <=100 karakter)
# ---------------------------------------------------------------------------
def segment(video_id: str):
    d = vdir(video_id)
    align = json.loads((d / "align.json").read_text())
    words = align["words"]
    if not words:
        return []
    segs, cur, t0 = [], [], words[0]["start"]
    for wd in words:
        cur.append(wd)
        text = " ".join(x["word"] for x in cur)
        dur = wd["end"] - t0
        if dur >= MAX_CLIP_SEC or len(text) >= MAX_CLIP_CHARS:
            segs.append(_mkseg(cur))
            cur, t0 = [], wd["end"]
    if cur:
        segs.append(_mkseg(cur))
    # DB'ye yaz
    out = []
    for i, s in enumerate(segs):
        if s is None or (s["end"] - s["start"]) < MIN_CLIP_SEC:
            continue
        seg_id = f"{video_id}_{i:04d}"
        state.add_segment(video_id, seg_id, s["text"], s["start"], s["end"])
        s["seg_id"] = seg_id
        out.append(s)
    return out


def _mkseg(words):
    if not words:
        return None
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["word"] for w in words).strip(),
        "words": words,
    }


# ---------------------------------------------------------------------------
# 7) KIRPMA + MASTER'A YAZMA (tam yüz 224x224 RGB + 16kHz wav + align + meta)
# ---------------------------------------------------------------------------
def crop_segments(video_id: str, segs: list[dict]) -> dict[str, int]:
    """Red sebeplerini sayıp döndürür. NEDEN: üç eleme kapısı da qc_pass=0
    yazıyordu, sıfır verimli bir videonun hangi kapıda öldüğü görünmüyordu
    (videoların %27'si böyle). Sayaç state.db'ye değil sonuç dict'ine gidiyor —
    Modal'da state.db geçici, master/'dan yeniden kuruluyor ve reddedilen
    segment master/'a hiç yazılmadığı için sebep orada yaşayamaz."""
    import cv2
    import soundfile as sf

    red: dict[str, int] = {}
    def _red(key):                       # sebep sayacı
        red[key] = red.get(key, 0) + 1

    d = vdir(video_id)
    faces = _load_faces(video_id)
    audio, sr = sf.read(str(d / "audio16k.wav"))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    cap = cv2.VideoCapture(str(d / "video25.mp4"))

    # DUBLAJ KARARI: video seviyesinde, bir kez. Segment seviyesinde
    # verilemez (bkz. dub_esigi) — 25-150 karelik korelasyon gürültüdür.
    v_score, v_n = video_dub_score(faces, audio, sr)
    v_esik = dub_esigi(v_n)
    # Kac karede yuz bulundu? Dusuk skorun sebebi dublaj mi, yoksa yuz
    # takibinin cokmesi mi — ayirt etmek icin gerekli, hesabi bedava.
    yuzlu = sum(1 for f in faces.values() if f.get("bbox"))
    red["video_yuz_orani"] = round(yuzlu / max(1, len(faces)), 4)
    red["video_dub_skor"] = None if np.isnan(v_score) else round(v_score, 4)
    red["video_dub_esik"] = round(float(v_esik), 4)
    red["video_dub_n"] = v_n
    # nan ise karar verilemiyor: videoyu topluca eleme, segment kapilari
    # (yuz_yok/yuz_kucuk) dogru etiketi zaten koyacak.
    red["video_dub_karar"] = karar = dub_karari(v_score, v_n)
    if karar == "elendi":
        # Video gerçekten dublajlı/uyumsuz: TAMAMI elenir, tek tek değil.
        for s in segs:
            state.mark_segment_qc(video_id, s["seg_id"], False)
        _red("video_dublajli")
        cap.release()
        return red

    for s in segs:
        mdir = MASTER / video_id / s["seg_id"]
        face_mp4 = mdir / "face.mp4"
        if face_mp4.exists():            # resume: bu segment kırpılmış
            _qc(video_id, s)
            _red("kabul")
            continue

        # KAPI SIRASI: önce YÜZ, sonra DUBLAJ. Tersi değil!
        # seg_dub_score() yüz yoksa mouth dizisini baştan sona 0 doldurur,
        # m.std()<1e-6 dalına düşer ve TAM 0.0 döner. Dublaj kapısı önce
        # geldiğinde "yüz bulunamadı" hatası "dublajlı" diye raporlanıyordu —
        # sıfır verimli videoların (%27) tamamı bu yüzden yanlış teşhis aldı.
        # Yan fayda: yüz yoksa korelasyon hiç hesaplanmıyor.
        f0 = int(round(s["start"] * FPS))
        f1 = int(round(s["end"] * FPS))
        # segment boyunca yüz bbox'larının birleşimi (sabit kare için)
        bxs = [faces[i]["bbox"] for i in range(f0, f1)
               if i in faces and faces[i]["bbox"]]
        if not bxs:
            state.mark_segment_qc(video_id, s["seg_id"], False)
            _red("yuz_yok")              # segment karelerinde hiç yüz bulunamadı
            continue
        x = min(b[0] for b in bxs); y = min(b[1] for b in bxs)
        x2 = max(b[0]+b[2] for b in bxs); y2 = max(b[1]+b[3] for b in bxs)
        side = max(x2 - x, y2 - y)
        if side < MIN_FACE_PX:
            state.mark_segment_qc(video_id, s["seg_id"], False)
            _red("yuz_kucuk")            # yüz var ama MIN_FACE_PX altında
            continue

        # Segment dublaj kapısı KALDIRILDI — dublaj kararı yukarıda, video
        # seviyesinde verildi. Skor yalnız meta.json'a ve histograma yazılıyor
        # (teşhis için); hiçbir segmenti elemiyor.
        score = seg_dub_score(faces, audio, sr, s)
        if np.isnan(score):
            _red("dub_hesaplanamadi")
        else:
            _red(f"dub_{min(int(score * 20), 19) * 5:02d}")   # 0.05'lik kovalar
        cx, cy = (x + x2) // 2, (y + y2) // 2
        half = side // 2

        mdir.mkdir(parents=True, exist_ok=True)
        with _video_writer(face_mp4, FACE_SIZE) as writer:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            for _ in range(f0, f1):
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                a = max(0, cy-half); b = min(h, cy+half)
                c = max(0, cx-half); e = min(w, cx+half)
                crop = frame[a:b, c:e]
                if crop.size == 0:
                    continue
                writer.write(cv2.resize(crop, (FACE_SIZE, FACE_SIZE)))

        # eşlenik ses kes
        _run(["ffmpeg", "-y", "-i", str(d / "audio16k.wav"),
              "-ss", str(s["start"]), "-to", str(s["end"]),
              "-ac", "1", "-ar", str(AUDIO_SR), str(mdir / "audio.wav")])

        # Auto-AVSR uyumlu 96x96 gri ağız kırpması (varsa)
        agiz = write_mouth_crop(cap, faces, f0, f1, mdir / "mouth.mp4")

        (mdir / "align.json").write_text(
            json.dumps({"text": s["text"], "words": s["words"]}, ensure_ascii=False))
        (mdir / "meta.json").write_text(json.dumps({
            "video_id": video_id, "seg_id": s["seg_id"],
            "start": s["start"], "end": s["end"], "fps": FPS, "face_size": FACE_SIZE,
            "dub_corr": None if np.isnan(score) else round(score, 3),
            "mouth_crop": agiz,
        }))
        _qc(video_id, s)
        _red("kabul")
    cap.release()
    return red


def _video_writer(path, size: int, gray: bool = False):
    """cv2.VideoWriter'i YEREL geçici dosyaya açar; kapanınca hedefe kopyalanır.

    NEDEN? cv2.VideoWriter mp4'ü sonlandırırken dosyada geri gidip (seek) başlık
    yazıyor. S3/R2 gibi nesne-depolama mount'ları rastgele yazmayı desteklemiyor
    -> yazma SESSİZCE başarısız oluyor, dosya hiç oluşmuyor. ÖLÇTÜK: S3'e geçişten
    sonra bulutta işlenen videoda face.mp4 hiç üretilmemişti (align.json ve
    ffmpeg'in yazdığı audio.wav vardı, çünkü onlar sıralı yazım).
    Çözüm: /tmp'ye yaz, bitince baytları hedefe kopyala (sıralı yazım -> güvenli).
    """
    import contextlib
    import tempfile
    import cv2

    @contextlib.contextmanager
    def _ctx():
        tmp = Path(tempfile.mkdtemp()) / Path(path).name
        w = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS, (size, size), isColor=not gray)
        try:
            yield w
        finally:
            w.release()
            if tmp.exists() and tmp.stat().st_size > 0:
                Path(path).write_bytes(tmp.read_bytes())
            import shutil
            shutil.rmtree(tmp.parent, ignore_errors=True)

    return _ctx()


def write_mouth_crop(cap, faces: dict, f0: int, f1: int, out_path) -> bool:
    """Auto-AVSR'ın beklediği 96x96 GRİ, ortalama-yüze hizalanmış ağız videosu.

    Neden ayrı bir çıktı? Auto-AVSR fine-tuning'i tam yüz 224x224 kabul etmiyor;
    68-noktalı ortalama yüze affine hizalanmış 96x96 ağız ROI istiyor
    (vendor/auto_avsr/video_process.py — orijinal referans kod).

    Neden ORİJİNAL karelerden? face.mp4'teki (224x224) ağız bölgesi gerçekte
    ~57x19 piksel; ondan 96x96 üretmek büyütme olur, detay kazandırmaz. Burada
    video25.mp4'ün tam çözünürlüklü kareleri kullanılır (ölçtük, bkz. B seçeneği).

    ESKİ VERİYE DOKUNMAZ: `stable4` landmark'ları yalnız bu değişiklikten sonra
    işlenen videoların faces.json'ında var. Yoksa sessizce atlanır ve False döner.
    """
    if not MOUTH_CROP:
        return False
    lms = [faces[i].get("stable4") if i in faces else None for i in range(f0, f1)]
    if not any(lm is not None for lm in lms):
        return False          # eski video (stable4 yok) veya yüz bulunamadı

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
    import cv2
    import numpy as np
    from auto_avsr.video_process import VideoProcess

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = []
    for _ in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    lms = [None if lm is None else np.array(lm, dtype=np.float32)
           for lm in lms[:len(frames)]]
    if not frames or not any(lm is not None for lm in lms):
        return False

    try:
        seq = VideoProcess(convert_gray=True)(frames, lms)
    except Exception:
        return False          # hizalama başarısız (aşırı profil/kadraj dışı)
    if seq is None or len(seq) == 0:
        return False

    with _video_writer(out_path, MOUTH_SIZE, gray=True) as w:
        for f in seq:
            w.write(f.astype(np.uint8))
    return True


def _qc(video_id, s):
    ok = (s["end"] - s["start"]) >= MIN_CLIP_SEC and len(s["text"]) > 0
    state.mark_segment_qc(video_id, s["seg_id"], ok)


# ---------------------------------------------------------------------------
# ORKESTRASYON — tek video, tam idempotent
# ---------------------------------------------------------------------------
def process_video(video_id: str) -> dict:
    v = state.get_video(video_id)
    if not v:
        return {"video_id": video_id, "status": "missing"}

    timings: dict[str, float] = {}
    try:
        # Sıra: download -> normalize -> ASR (Metal, hızlı) -> segment ->
        #       faces (yalnız segment kareleri) -> crop (segment-bazlı dub).
        with _timed(timings, "download"):
            if not state.stage_reached(video_id, "downloaded"):
                download(video_id, v["url"]); state.set_stage(video_id, "downloaded")
        with _timed(timings, "normalize"):
            if not state.stage_reached(video_id, "normalized"):
                normalize(video_id); state.set_stage(video_id, "normalized")
        with _timed(timings, "asr"):            # <- TEK GPU kullanan aşama
            if not state.stage_reached(video_id, "asr"):
                transcribe(video_id); state.set_stage(video_id, "asr")

        with _timed(timings, "segment"):
            segs = segment(video_id)        # idempotent: segments tablosunu yazar
            state.set_stage(video_id, "segmented")

        with _timed(timings, "faces"):
            if not state.stage_reached(video_id, "faces"):
                extract_faces(video_id, segs); state.set_stage(video_id, "faces")

        with _timed(timings, "crop"):
            red = crop_segments(video_id, segs)  # segment-bazlı dublaj kontrolü içeride
        state.set_stage(video_id, "done", status="done")
        return {"video_id": video_id, "status": "done",
                "segments": len(segs), "good_segments": state.good_count(video_id),
                "timings": timings, "gpu_share": _gpu_share(timings), "red": red}

    except Exception as e:  # hata -> 'error', resume bir sonraki turda devam eder
        state.set_stage(video_id, v["stage"], status="error", reason=str(e)[:300])
        return {"video_id": video_id, "status": "error", "error": str(e)[:300],
                "timings": timings}


def _gpu_share(timings: dict) -> float:
    """T4'ün gerçekten kullanıldığı sürenin oranı. Düşükse (~0.3 altı) pipeline'ı
    CPU-only + GPU-only iki fonksiyona bölmek maliyeti ciddi düşürür."""
    total = sum(timings.values())
    return round(timings.get("asr", 0.0) / total, 3) if total else 0.0
