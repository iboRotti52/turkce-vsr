# 👄 Türkçe Dudak Okuma (Visual Speech Recognition - VSR)

Bu proje, video akışlarından ses olmadan yalnızca dudak hareketlerini analiz ederek konuşulan **Türkçe metni** deşifre eden uçtan uca bir Görsel Konuşma Tanıma (Visual Speech Recognition / Lip Reading) yapay zeka sistemidir.

Proje şu anda temiz araştırma başlangıcındadır. Mimari ve eğitim reçetesi önceden seçilmemiştir; bütün bileşenler veri/evaluation audit'i ve geniş kaynak araştırması sonrasında kanıtlarla belirlenecektir.

## Antigravity ile otonom araştırmayı başlatma

Antigravity'de `/goal` açarken [ANTIGRAVITY_GOAL.md](ANTIGRAVITY_GOAL.md) içindeki metni kullanın. Kalıcı araştırma kuralları `GEMINI.md`, tek yaşayan modelin güncel durumu `research/CANDIDATE.md`, çalıştırılabilir reçetesi ise `configs/research_candidate.yaml` içindedir.

Araştırma çok sayıda kalıcı pilot model üretmez. Tek kanonik modeli kanıta dayalı probe'larla olgunlaştırır; full training yalnız readiness kapılarının tamamı geçildikten sonra yapılır.

---

## 🚀 Araştırma kapsamı

1. **Gelişmiş Yüz ve Dudak Ön İşleme**:
   - `MediaPipe FaceMesh` ile 468 yüz referans noktası üzerinden dudak bölgesinin (ROI) tespiti.
   - Ani kafa hareketlerinde titremeyi önleyen **üstel hareketli ortalama (EMA) stabilizasyonu**.
   - 25 FPS sabitleme ve `88x88` piksel gri tonlamalı standart normalizasyon.
2. **Türkçe Fonetik ve Visem Haritalama**:
   - Türkçe alfabeye özgü 29 harf (`ç, ğ, ı, ö, ş, ü`) ve CTC blank token uyumlu sözlük.
   - Eşgörünümlü sesleri (homophenes) analiz eden **12 sınıflı Türkçe visem sınıflandırması** (çift dudaksı: b/p/m, diş-dudaksı: f/v vb.).
3. **Açık mimari araştırması**:
   - Görsel frontend, zamansal model, objective/tokenizer, loss, curriculum, optimizer, initializer, decoder ve keyword spotting henüz seçilmemiştir.
   - VSR, video, ASR, self-supervised öğrenme, multilingual transfer ve komşu alanlardan uygun her yaklaşım değerlendirilebilir.
4. **İnteraktif Web Arayüzü (Gradio)**:
   - Video yükleme veya web kamerasından canlı dudak okuma, görsel visem akışı ve kırpılmış dudak ROI önizlemesi.
5. **Hugging Face Entegrasyonu**:
   - Model ağırlıklarını ve veri setlerini Hugging Face Hub üzerinden yönetme, Trackio ile eğitim metriklerini gerçek zamanlı izleme.

---

## 🧠 Entegre Edilen Agent Skills Ekosistemi (`.agents/skills/`)

Projede, yapay zeka ajanlarının (Antigravity, Claude Code, Cursor, Codex) geliştirme sürecini disiplinli ve otonom yürütmesi için 4 temel depodan toplam **44 yetenek (skill)** entegre edilmiştir:

| Kaynak Depo | Yetenekler (Skills) | İşlev |
| :--- | :--- | :--- |
| **`obra/superpowers`** (Tamamı - 14 Skill) | `test-driven-development`, `systematic-debugging`, `verification-before-completion`, `brainstorming`, `executing-plans`, `subagent-driven-development`, `using-git-worktrees` vb. | Kodlamadan önce test yazma, sessiz tensör hatalarını önleme, sistematik hata ayıklama ve doğrulama. |
| **`nik1t7n/research-craft-skill`** (Tamamı) | `research-craft` | Bilimsel problem seçimi, arXiv/SOTA makale inceleme disiplini, deney günlüğü tutma ve hipotez testi. |
| **`huggingface/skills`** (Tamamı - 25 Skill) | `hf-cli`, `huggingface-datasets`, `huggingface-papers`, `huggingface-trackio`, `huggingface-gradio`, `huggingface-spaces`, `huggingface-vision-trainer`, `huggingface-zerogpu` vb. | HF Hub operasyonları, veri seti keşfi, model checkpoint saklama ve deney takibi. |
| **`LichAmnesia/awesome-antigravity-skills`** (Tamamı) | `conventional-commits`, `pr-description`, `skill-template` | Git standartları ve yeni agent skill şablonları. |
| **Özel Alan Yeteneği (Domain Skill)** | `turkish-lip-reading` | Türkçe dudak okuma ön işleme, fonem/visem, model mimarisi ve eğitim runbook'u. |

---

## 📂 Proje Dizin Yapısı

```text
delightful-hawking/
├── .agents/
│   └── skills/              # 44 adet Agent Skill (superpowers, HF, research-craft, vb.)
├── configs/
│   ├── baseline_ctc.yaml    # 3D-ResNet + BiGRU CTC eğitim konfigürasyonu
│   └── avhubert_tr.yaml     # Pretrained AV-HubERT transfer learning ayarları
├── docs/                    # Kapsamlı mimari, gecikme ve veri kılavuzları
│   ├── DATASET_GUIDE.md     # S3 Türkçe Veri Kümesi Kılavuzu (İndirme & Pipeline)
│   ├── TEKNIK_DOKUMANTASYON.md # Matematiksel ve kuramsal temeller
│   ├── MIMARI.md            # Uçtan uca model ve çıkarım mimarisi
│   └── LATENCY_RAPORU.md    # T4/L4/A10G gecikme ve kıyaslama raporu
├── data/
│   ├── master/              # S3'ten indirilen 96x96 dudak ROI'leri (mouth.mp4)
│   ├── raw/                 # Ham videolar (.mp4)
│   ├── processed/           # 25 FPS kırpılmış dudak tensörleri
│   └── metadata/            # Split haritası (split_map.json), tokenizer ve CSV'ler
│       └── tokenizer/       # 75k cümle ile eğitilmiş Türkçe SentencePiece modeli
├── scripts/
│   ├── download_s3_dataset.py   # Çoklu iş parçacıklı S3 veri indiricisi
│   ├── build_dataset_manifest.py # train/val/test CSV manifest üreticisi
│   └── reference_pipeline/  # Veri toplama ve doğrulama referans betikleri
├── vendor/
│   └── auto_avsr/           # 68-nokta ortalama yüz referansı ve afin hizalama
├── src/
│   ├── preprocessing/
│   │   ├── face_mesh.py     # MediaPipe yüz ve dudak landmark tespiti
│   │   ├── lip_cropper.py   # Dudak ROI kırpma ve 25 FPS video normalizasyonu
│   │   └── turkish_vocab.py # 29 harfli Türkçe CTC sözlüğü ve visem haritası
│   ├── dataset/
│   │   └── dataset.py       # PyTorch Dataset, veri artırma ve collate_fn
│   ├── models/
│   │   └── resnet3d_conformer.py # 3D-ResNet + BiGRU CTC Dudak Okuma Modeli
│   ├── training/
│   │   └── train.py         # Model eğitim ve değerlendirme motoru
│   └── evaluation/
│       ├── metrics.py       # CER, WER ve Viseme Error Rate hesaplayıcı
│       └── evaluate.py      # Test kümesi doğrulama scripti
├── tests/
│   ├── test_vocab.py        # Türkçe karakter ve CTC decode birim testleri
│   ├── test_metrics.py      # CER / WER metrik testleri
│   └── test_model.py        # Model forward pass ve CTC loss testleri
├── app/
│   └── app.py               # İnteraktif Gradio web arayüzü
├── requirements.txt         # Gerekli bağımlılıklar
└── README.md                # Proje dokümantasyonu
```

---

## 🛠️ Kurulum ve Başlangıç

### 1. Sanal Ortam Oluşturma ve Bağımlılıklar

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Türkçe Dudak Verilerini İndirme

> **Büyük-veri rejimi:** Araştırma artık Hugging Face'teki büyük veriyle devam ediyor:
> **https://huggingface.co/datasets/avsr-tr-ekip/avsr-tr-dataset**
> Kodun varsayılan indirme adresi bu datasettir (`src/data/hf_downloader.py`).
> Arkadaş devri için önce [HANDOVER.md](HANDOVER.md) dosyasını oku.

Projede kullanılan ~28.4 saatlik Türkçe dudak okuma verisini S3 deposundan çekmek için:

```bash
# Model eğitimi için gereken 96x96 dudak ROI klibi ve etiketleri indir (~4.7 GiB):
python scripts/download_s3_dataset.py --mode mouth

# (İsteğe bağlı) Transfer learning temel ağırlıklarını indir (~1 GiB):
python scripts/download_s3_dataset.py --mode ckpt

# İndirilen verilerden train.csv, val.csv ve test.csv manifestolarını üret:
python scripts/build_dataset_manifest.py
```
> Detaylı veri mimarisi ve S3 hiyerarşisi için [DATASET_GUIDE.md](docs/DATASET_GUIDE.md) belgesini inceleyin.

### 3. Birim Testlerini Çalıştırma (TDD)

```bash
python3 -m unittest discover tests
```

### 3. Gradio Web Uygulamasını Başlatma

Kullanıcı dostu web arayüzünü çalıştırmak için:

```bash
python3 -m app.app
```
Tarayıcınızda `http://localhost:7860` adresini açarak bir video yükleyebilir veya web kamerasından konuşarak dudak okuma sonucunu test edebilirsiniz.

---

## 📊 Değerlendirme Metrikleri

- **CER (Character Error Rate)**: Harf düzeyinde hata oranı: $\frac{S + D + I}{N_{char}}$
- **WER (Word Error Rate)**: Kelime düzeyinde hata oranı: $\frac{S + D + I}{N_{word}}$
- **Viseme Error Rate**: Görsel olarak ayırt edilemeyen sesleri (homophenes) tolere eden görsel fonetik hata oranı.

---

## 📄 Lisans

Bu proje MIT lisansı altında sunulmaktadır.
