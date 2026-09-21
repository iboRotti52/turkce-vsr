# GPU Konteyner Kurulum Politikaları & Kaynaklar

Ekiple paylaşım için: Auto-AVSR modelini serverless GPU'da (Modal) çalıştırmak üzere
kullandığımız imaj kurulum politikaları, GPU/snapshot ayarları ve kaynak linkleri.
Çalışan scriptler: [`benchmark/`](benchmark/).

---

## 1. Platform ve Hesap

- **Platform:** [Modal](https://modal.com) — serverless GPU, kart gerektirmeyen
  ücretsiz kademe (**$30/ay kredi**, aşımda iş durur, sürpriz fatura yok).
- Neden Modal: kod-öncelikli (Docker/YAML yazmadan `modal.Image` ile Python'da
  tanımlanıyor), cold-start/queue/exec sürelerini ayrı ayrı ölçen hazır gözlemlenebilirlik var.

## 2. İmaj Kurulum Politikası

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "pytorch-lightning", "sentencepiece", "av",
        "mediapipe==0.10.14",   # PIN ZORUNLU — bkz. not aşağıda
        "opencv-python-headless", "scikit-image", "numpy", "six",
    )
    .add_local_dir("/tmp/auto_avsr_check", remote_path="/root/auto_avsr")
)
```

**Kritik not — mediapipe versiyon pini:** mediapipe'ın yeni sürümleri (~0.10.30+),
Python 3.13+ wheel'lerinde eski `mp.solutions.face_detection` / `face_mesh` API'sini
**tamamen kaldırdı**, yalnız yeni Tasks API kaldı. Auto-AVSR'ın (ve bizim kendi
dataset pipeline'ımızın) kullandığı yüz-landmark kodu eski API'ye yazılmış. Python
3.11 + `mediapipe==0.10.14` pini ile bu sorunu aşıyoruz. **Yeni bir GPU imajı
kurarken bu pini değiştirmeyin** — değiştirirseniz yüz tespiti sessizce kırılır.

**`opencv-python-headless` tercih edildi** (`opencv-python` değil): konteynerde
GUI/display yok, headless sürüm daha küçük image + gereksiz sistem bağımlılığı
(libGL vs.) sorunlarını azaltıyor.

## 3. GPU ve Snapshot Politikası

```python
@app.cls(
    image=image,
    gpu="L4",                       # bkz. GPU seçimi notu
    volumes={"/ckpt": vol},
    scaledown_window=600,           # 10 dk boşta -> konteyner kapanır
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class VSR:
    @modal.enter(snap=True)         # SNAPSHOT'A DAHIL: model yükleme + CUDA ısınma çıkarımı
    def load_and_warm(self): ...

    @modal.enter(snap=False)        # SNAPSHOT DIŞI: mediapipe (tflite thread'leri restore'da riskli)
    def load_detector(self): ...
```

**GPU seçimi:** T4 / L4 / A10G karşılaştırdık (bkz. `LATENCY_RAPORU.md`) — **L4**
kazandı: A10G ile aynı hız, T4'ten ~%25-35 hızlı, daha ucuz.

**`enable_memory_snapshot` + `enable_gpu_snapshot`:** model GPU'da yüklenmiş ve
CUDA kernelleri ısınmış haldeyken süreci "donduruyor"; sonraki cold start bu
durumdan başlıyor. **Zorunlu ön koşul: `modal deploy` ile kalıcı deploy** —
`modal run` (geçici çalıştırma) snapshot'ı aktive etmiyor, bunu atlarsanız
optimizasyon sessizce devre dışı kalır.

**`@modal.enter(snap=True)` vs `snap=False` ayrımı önemli:** snapshot'a giren
kısımda CUDA/GPU state'i olmalı; tflite (mediapipe) gibi thread-havuzu kullanan
kütüphaneleri snapshot dışına almazsanız restore sırasında bozulma riski var.

**`scaledown_window=600`:** konteyner son istekten 10 dk sonra kapanıyor. Kısa
tutarsanız (biz ilk testte 60sn kullanmıştık) oturum-içi tekrar cold-start riski
artar; uzun tutarsanız boşta-bekleme maliyeti artar. 10 dk, pilot trafiği için
makul bir orta nokta.

## 4. Ölçülmüş Sonuç (politikanın etkisi)

| | Snapshot yok | Snapshot + ısınma |
|---|---|---|
| Cold start (uçtan uca) | 33.6 sn | **10.5 sn** (worker cache'liyken) — cache'siz 27-35sn |
| Sıcak istek | ~3.6-5.1 sn | ~3.6-5.1 sn (değişmedi) |

Detaylı metodoloji ve tüm ham veriler: [`LATENCY_RAPORU.md`](LATENCY_RAPORU.md) (Bölüm 9).

**Önemli sınırlama:** Snapshot kazancı **worker cache'ine bağlı** — garanti değil,
olasılık artışı. Modal'da hangi fiziksel makineye düşeceğinizi seçemiyorsunuz.
Gerçek trafik olmadan (düşük/pilot trafik) kazanç dalgalı; trafik arttıkça
tipikleşir. Bunu "cache'i garantiye alma" değil "worst-case'i iyileştirme"
stratejisi olarak görün.

## 5. Checkpoint / Model Dosyası Politikası

Ağır dosyalar (model checkpoint) **image içine gömülmüyor** — ayrı bir
`modal.Volume`'a bir kere yükleniyor, fonksiyonlar oradan mount ediyor:

```bash
modal volume create avsr-ckpt
modal volume put avsr-ckpt vsr_trlrs3_base.pth /vsr_trlrs3_base.pth
```

Neden: image her rebuild'de checkpoint'i yeniden indirmiyor/paketlemiyor —
build hızlanıyor, gereksiz veri transferi olmuyor. **İndirilen her checkpoint'in
MD5'i, kaynağın (repo README'sindeki model zoo tablosu) belirttiği değerle
doğrulanmalı** — bütünlük garantisi için.

## 6. Kaynaklar

**Modal dokümantasyonu:**
- [Memory Snapshots](https://modal.com/docs/guide/memory-snapshots)
- [Cold start performance](https://modal.com/docs/guide/cold-start)
- [GPU Memory Snapshots: Supercharging sub-second startup](https://modal.com/blog/gpu-mem-snapshots)
- [Endpoint metrics](https://modal.com/docs/guide/endpoint-metrics) (P50/P90/P99 gözlemlenebilirlik)
- [Pricing](https://modal.com/pricing)

**Model kaynağı (Auto-AVSR):**
- [Auto-AVSR GitHub — Imperial College London](https://github.com/mpc001/auto_avsr) (resmi repo, model zoo checkpoint linkleri README'de)
- [Auto-AVSR paper (arXiv)](https://arxiv.org/html/2303.14307v3)

**Genel serverless GPU karşılaştırması** (Modal'ı neden seçtiğimizin arka planı):
- [Cold start reduction techniques (Spheron)](https://www.spheron.network/blog/gpu-cold-start-llm-inference-2026/)
- [ServerlessLLM: Low-Latency Serverless Inference (arXiv)](https://arxiv.org/pdf/2401.14351)

## 7. Çalışan Scriptler (`benchmark/`)

| Dosya | Ne yapar |
|---|---|
| `vsr_benchmark.py` | 3 GPU × 4 klip uzunluğu × 3 tekrar + burst testi — tam karşılaştırma |
| `vsr_snapshot.py` | Snapshot + enter-ısınması politikası uygulanmış deploy edilebilir versiyon |
| `bench_T4.json` / `bench_L4.json` / `bench_A10G.json` | Ham ölçüm sonuçları |
| `snap_deploy.log` / `snap_extra.log` | Snapshot deneyinin ham logları |

**Kendi Modal hesabınızla tekrar çalıştırmak için:**
```bash
modal setup                              # hesap bağlama (tarayıcı OAuth)
modal volume create avsr-ckpt
modal volume put avsr-ckpt <checkpoint>.pth /vsr_trlrs3_base.pth
modal run vsr_benchmark.py               # tek seferlik ölçüm
modal deploy vsr_snapshot.py             # snapshot'ı aktive eden kalıcı deploy
```
