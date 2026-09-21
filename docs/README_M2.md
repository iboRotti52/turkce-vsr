# Türkçe LRS — Resume'lu Veri Toplama (MacBook Pro M2 Max, native)

Tek-konuşmacı Türkçe videolardan, LRS/Auto-AVSR uyumlu, model-agnostik bir
"master" veri seti üretir. n8n orkestrasyon yapar; ağır iş native macOS worker'da
(Metal/Neural Engine) koşar. **Kaldığı yerden devam eder.**

## Resume nasıl çalışıyor?
Üç katman birlikte:
1. **Kalıcı durum** — `data/state.db` (SQLite). Her videonun hangi aşamada olduğu yazılı.
2. **Idempotent aşamalar** — her aşama çıktısı zaten varsa atlar (`src.*`, `video25.mp4`,
   `faces.json`, `align.json`, `master/.../face.mp4`). Video yarıda kalırsa kaldığı
   aşamadan devam eder.
3. **n8n yeniden sorgular** — her tetiklemede `pending` (yeni + yarıda kalmış + hatalı)
   videoları çeker. Worker/laptop/n8n çökse bile bir sonraki turda kalınan yerden sürer.

## Kritik: ML'i Docker'a KOYMA
Docker Desktop Mac'te bir Linux VM'de çalışır, Metal'e erişemez → whisper/mediapipe
CPU-only kalır, yavaşlar. Bu yüzden:
- **Worker → native macOS** (uvicorn ile, Docker değil).
- **n8n → Docker veya native fark etmez** (sadece HTTP atar, GPU istemez).
- Execute Command node'una hiç dokunmuyoruz (cloud'da yok, n8n 2.0'da varsayılan kapalı).

## Kurulum

```bash
# 1) Sistem araçları (native ARM64)
brew install yt-dlp ffmpeg

# 2) Python ortamı
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) Kaynakları doldur — sources.txt'e Türkçe, TEK-KONUŞMACI, frontal video URL'leri
#    (TEDx, haber sunucusu, monolog podcast). DUBLAJ İSTEME.

# 4) Worker'ı başlat (native, Metal)
uvicorn worker.app:app --host 127.0.0.1 --port 8077
```

Worker ayakta. Şimdi ya n8n ile sürekli işlet ya da elle tetikle:

```bash
# Elle (n8n olmadan da çalışır):
curl -X POST http://127.0.0.1:8077/sources/load
curl http://127.0.0.1:8077/queue/pending
curl -X POST http://127.0.0.1:8077/process/VIDEO_ID
curl http://127.0.0.1:8077/status
```

## n8n ile (resume'lu, otomatik)
`n8n_resume_workflow.json`'u içe aktar. Akış:
`Schedule (10 dk) → Load Sources → Get Pending → Split → Loop → POST /process/{id} → Loop`

- n8n Docker'da ise worker'a `http://host.docker.internal:8077` ile ulaşır (JSON'da hazır).
- n8n native ise URL'leri `http://127.0.0.1:8077` yap.
- Schedule her 10 dk'da bir `pending`'i yeniden çeker → otomatik resume.
- Worker uzun sürebilir; Process Video node'unda timeout 30 dk'ya ayarlı.

## Çıktı (model-agnostik master)
```
data/master/<video_id>/<seg_id>/
    face.mp4      # tam yüz 224x224 RGB @25fps   (her crop'u buradan türetirsin)
    audio.wav     # 16kHz mono
    align.json    # {text, words:[{word,start,end}]}
    meta.json
data/state.db     # SQLite durum (resume)
```

Buradan ince adaptörle istediğin formatı üretirsin:
- **MobiVSR**: kelime sınırlarından tek-kelime klipleri + kelime sınıfı etiketi.
- **Auto-AVSR**: cümle klipleri + tam metin; 96x96 gray ağız re-crop.

## Aşamalar (idempotent)
`download → normalize → asr → segment → faces → crop → done`

**Sıra neden böyle?** ASR (mlx-whisper, Metal) ucuzdur; önce koşar. Segmentler
çıkınca yüz çıkarımı **yalnız konuşma karelerinde** yapılır (tüm video değil) +
kare mesh'e küçültülerek verilir (`config.FACE_MESH_LONG_EDGE`). Bu, en ağır yerel
adımı (MediaPipe) ciddi hafifletir.

Dublaj kontrolü artık **segment bazında** (TalkNet yerine hafif heuristik):
her segmentin kendi ağız-açıklığı↔ses-enerjisi korelasyonu eşik altındaysa o
**segment** elenir — videonun tamamı değil (`config.DUB_CORR_THRESHOLD`). Tüm-video
korelasyonu çok-sahneli kaynakta iyi konuşmayı da reddettiği için terk edildi.
Tek-konuşmacı kaynakta bu yeterli; çok-konuşmacı sahnelerde TalkNet'e geç.

## Fizibilite araçları (yerel, hafif)
```bash
python audit.py          # data/feasibility_report.md: saat, konuşmacı, kelime-
                         # frekans (kapalı-küme adayı), vizem/homofen yoğunluğu
python splits.py         # KONUŞMACI-DISJOINT train/val/test (segments.split)
```
`audit.py` "verimiz fizibıl mi, hangi hedef (kapalı/açık küme) gerçekçi?" sorusunu
veriyle yanıtlar. `splits.py` aynı konuşmacının train+test'e sızmasını engeller
(yoksa model dudak değil kimlik öğrenir).

## Ortam notları
- **ASR modeli:** pilot için `whisper-large-v3-turbo` (küçük/hızlı); nihai yüksek-
  doğruluk toplama için `config.WHISPER_REPO`'yu `whisper-large-v3-mlx` yap.
- **Yüz landmark:** MediaPipe **Tasks API** `FaceLandmarker` (`.task` modeli ilk
  çalıştırmada `data/models/`'e otomatik iner). Yeni mediapipe wheel'lerinde
  (Py≥3.13) eski `mp.solutions.face_mesh` API'si KALDIRILDI; bu yüzden Tasks API.

## Bilinen sınırlar (ilk sürüm — dürüstçe)
- Yüz takibi tek yüz varsayar (kaynak küratörlüğüne güvenir); kalabalık sahne için zayıf.
- Dublaj kontrolü kaba; eşiği `data/`'daki sonuçlara bakıp kalibre et.
- `faces.json` büyük videolarda şişer; çok veri toplarsan `.npz`'ye geçir.
- ASR Türkçe ortografiyi normalize eder ama noktalama bilgisini düşürür; cümle
  segmentasyonu şu an süre/karakter sınırına dayanıyor (saf noktalama değil).
