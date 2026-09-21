---
name: turkish-lip-reading
description: Comprehensive guide and domain runbook for developing, training, and evaluating Turkish Lip Reading (Visual Speech Recognition - VSR) models. Covers video face/lip ROI preprocessing with MediaPipe, Turkish phoneme-to-viseme mapping, CTC vocabulary construction, 3D-CNN + Conformer / AV-HubERT architectures, dataset collection (YouTube/TRT), WER/CER evaluation, and Gradio deployment.
---

# Turkish Lip Reading (Görsel Konuşma Tanıma - VSR) Runbook

Bu skill, Türkçe dilinde video tabanlı dudak okuma (Visual Speech Recognition / Lip Reading) sistemleri tasarlamak, eğitmek ve doğrulamak için uçtan uca teknik yönergeleri içerir.

---

## 1. Türkçe Fonetik ve Visem (Görsel Fonem) Yapısı

Dudak okumada temel zorluk, ses tellerinde veya ağız boşluğunun arkasında oluşan ancak dudakta aynı görünen seslerdir (**Homophenes / Eşgörünümlü sesler**). 

### A. Türkçe Alfabe ve CTC Sözlüğü
Türkçe alfabede 29 harf bulunur. CTC (Connectionist Temporal Classification) için standart sözlük yapısı:
- `0`: `<blank>` (CTC için zorunlu boşluk etiketi)
- `1`: `<pad>`
- `2`: `<unk>`
- `3`: `<space>` (Kelimeleri ayıran boşluk)
- `4-32`: `a, b, c, ç, d, e, f, g, ğ, h, ı, i, j, k, l, m, n, o, ö, p, r, s, ş, t, u, ü, v, y, z`
- (İsteğe bağlı): Kesme işareti (`'`)

### B. Türkçe Visem Haritalaması (Viseme Classes)
İnsan veya model dudaktan bakarken aşağıdaki ses gruplarını birbirinden tek başına ayıramaz. Bu nedenle model kelime/cümle düzeyinde dil modeli (Language Model / KenLM / BERT) ile desteklenmelidir:
1. **Çift Dudaksı (Bilabial)**: `/b/, /p/, /m/` (Dudaklar tamamen kapanır ve açılır)
2. **Diş-Dudaksı (Labiodental)**: `/f/, /v/` (Üst dişler alt dudağa değer)
3. **Dişeti / Diş (Alveolar / Dental)**: `/d/, /t/, /s/, /z/, /n/` (Dudaklar hafif aralık, dişler birbirine yakın)
4. **Damaksı / Dişeti-Damaksı (Post-alveolar / Palatal)**: `/c/, /ç/, /j/, /ş/, /y/` (Dudaklar hafifçe öne yuvarlanır)
5. **Damak / Boğaz (Velar / Glottal)**: `/k/, /g/, /ğ/, /h/` (Ağız açıklığı orta, dudak hareketi minimum)
6. **Yanal / Titrek**: `/l/, /r/`
7. **Yuvarlak Ünlüler (Rounded Vowels)**: `/o/, /ö/, /u/, /ü/` (Dudaklar belirgin şekilde büzülür ve dairesel form alır)
8. **Düz Ünlüler (Unrounded Vowels)**: `/a/, /e/, /ı/, /i/` (Dudaklar yatay olarak gerilir veya düz açılır)

---

## 2. Video ve Dudak Ön İşleme Protokolü

Standart bir VSR modelinin başarılı olabilmesi için video ön işleme aşaması son derece katıdır:

1. **Sabit Kare Hızı (FPS Normalization)**:
   - Tüm eğitim ve çıkarım videoları **25 FPS**'e dönüştürülmelidir (`ffmpeg -r 25` veya OpenCV interpolasyonu).
2. **Yüz & Dudak Landmark Tespiti**:
   - `MediaPipe FaceMesh` modeli (468/478 landmark) kullanılır.
   - Dudak dış hat landmark indisleri: `[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]`.
   - Dudak merkez koordinatı (centroid): `(x_center, y_center) = (mean(x_landmarks), mean(y_landmarks))`.
3. **ROI (Region of Interest) Kırpma ve Sabitleme**:
   - Dudak merkezinden $W \times H$ boyutunda kare bir kutu kırpılır (standart: `88x88` veya `96x96`).
   - Ani kafa hareketlerinde titremeyi (jitter) önlemek için bounding box koordinatları hareketli ortalama (Moving Average / Kalman Filter) ile yumuşatılmalıdır.
4. **Normalizasyon**:
   - Genellikle tek kanallı gri tonlama (Grayscale) tercih edilir.
   - Piksel değerleri `[0, 1]` aralığına ve ardından `(pixel - 0.421) / 0.165` (LRW/LRS standart ortalama/standart sapma) ile normalize edilir.

---

## 3. Model Mimarisi Seçenekleri

### Seçenek 1: 3D-ResNet + Conformer + CTC (Önerilen Hafif Omurga)
- **Frontend (3D-CNN)**: `Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))` + BatchNorm3d + PReLU + MaxPool3d. Zamansal bilgiyi (temporal motion) ve mekansal özellikleri (spatial features) ilk katmanda birlikte yakalar.
- **ResNet 2D Omurga**: ResNet-18 benzeri residual bloklar ile 512 boyutlu görsel özellik vektörü çıkarılır.
- **Temporal Encoder**: 4-12 katmanlı Conformer veya 3 katmanlı Bidirectional GRU (BiGRU, hidden=512).
- **CTC Başlığı**: `Linear(512, vocab_size)` + `nn.CTCLoss(blank=0, zero_infinity=True)`.

### Seçenek 2: Pretrained Foundation Model (AV-HubERT / Auto-AVSR)
- Imperial College veya Meta AI tarafından eğitilmiş AV-HubERT ağırlıkları (`large-voxceleb2` veya `base`) alınır.
- Son CTC katmanı Türkçe sözlük boyutuna (33 token) uyarlanıp dondurulan görsel katmanlar kademeli olarak (gradual unfreezing) Türkçe veriyle fine-tune edilir.

---

## 4. Veri Seti Hazırlığı ve Veri Artırma (Data Augmentation)

### Türkçe Veri Kaynakları:
- TRT Haber / TBMM TV / YouTube Türkçe röportaj ve söyleşi kanalları.
- Net yüz açısı (frontal, $\pm 15^\circ$), iyi aydınlatma, minimum dudak örtünmesi (mikrofon, sakal vb.).
- Altyazı veya ses kaydı `whisper-large-v3-turbo` ile Türkçe kelime zaman damgalarına (word-level timestamps) ayrıştırılarak 2-5 saniyelik video kliplerine bölünür.

### Video Veri Artırma Yöntemleri:
- **Random Crop**: 96x96 kırpılmış videodan rastgele 88x88 pencere seçme.
- **Random Horizontal Flip**: Yatay çevirme (%50 olasılık).
- **Time Masking (SpecAugment for Video)**: Video dizisinde ardışık 2-5 kareyi sıfırlama (modeli kare kaybına karşı dayanıklı kılar).

---

## 5. Değerlendirme Metrikleri

- **CER (Character Error Rate)**: $(S + D + I) / N_{char}$ (Türkçe gibi eklemeli dillerde harf doğruluğunu ölçmek için birincil metriktir).
- **WER (Word Error Rate)**: $(S + D + I) / N_{word}$ (Kelime hata oranı).
- **Inference Latency**: Tek bir video karesi için milisaniye cinsinden gecikme süresi (Gradio arayüzünde canlı video için < 40 ms hedeflenir).

---

## 6. Doğrulama ve Test Rutini

1. Sentetik video tensörü `(Batch, Channels, Frames, Height, Width)` -> `(2, 1, 25, 88, 88)` ile forward pass test edilir.
2. CTC Loss'un `(T, B, C)` giriş formatında NaN veya sonsuz üretmediği doğrulanır.
3. Türkçe karakter dönüşümünde (`I-ı`, `İ-i`) Python `locale` hatalarına karşı Türkçe harf normalizasyonu fonksiyonu test edilir.

---

## 7. Çözümleme, Boşluk Cezası ve Halüsinasyon Kontrolü (Decoding Invariants)

1. **Hedef Sözlükte Asgari Uzunluk (`min_word_len >= 2`)**:
   - `unigram5000` veya diğer alt-kelime sözlüklerinde yer alan tek harfli simgeler (`a, e, i, b, c, d, ...`) CTC çözücülerinde devasa sahte alarm tufanına (False Positive > 10.000) yol açar. Anahtar kelime avcısında tek harfli simgeler sözlükten elenmeli veya en az 2 harfli tam sözcükler kullanılmalıdır.
2. **Uzunluk Ölçekli Kelime Primi (Length-Normalized Word Bonus)**:
   - Sabit kelime primi (`beta`) kısa 2 harfli kelimeleri ("da, de, en, ne") kayırırken uzun Türkçe sözcükleri cezalandırır. Kelime primi formülü `beta * len(completed_word)` olarak ölçeklenmelidir.
3. **Asgari Kelime Süresi (`min_dur`)**:
   - 25 FPS videoda hiçbir Türkçe kelime 1-2 karede telaffuz edilemez. Çözücüde her kelime için `min_dur = max(2, int(len(completed_word) * min_dur_factor))` süresi aranmalıdır.
4. **Ardışık Tekrar Cezası (`repeat_penalty`)**:
   - Belirsiz dudak hareketlerinde aynı kelimenin ardışık olarak tekrarlanmasını engellemek için son tespit edilen kelimeye ceza puanı (örn. `repeat_penalty=3.0`) uygulanmalıdır.
5. **Kalibre Edilmiş Boşluk Cezası (`blank_penalty`)**:
   - CTC tepe noktalarını açmak için boşluk cezası kullanılabilir; ancak 1.0 üzerindeki değerler sesli harf patlamasına neden olur. Önerilen güvenli aralık `0.2 - 0.4`tür.

---

## 8. Veri Kalitesi ve Bölümleme Standartları (Dataset & Split Invariants)

1. **Konuşmacı Ayrık Veri Bölümlemesi (Speaker-Disjoint Split)**:
   - Aynı konuşmacının farklı cümleleri asla train, val ve test kümeleri arasına dağıtılmamalıdır. `split_map.json` gibi video/konuşmacı bazlı harita üzerinden sıfır sızıntı garanti edilmelidir.
2. **Altyazı ve Jenerik Filtrelemesi (Level 1 Data Quality)**:
   - YouTube ve internet kaynaklı verilerde sessizlik/müzik anlarında ekranda beliren çevirmen/jenerik imzaları (`altyazı`, `m k altyazı`, vb.) ve 2 kelimeden kısa anlamsız transkriptler eğitim ve test kümelerinden ayıklanmalıdır.
3. **Müfredat Eğitimi (Curriculum Learning: Word -> Phrase -> Continuous)**:
   - Akustik model sıfırdan doğrudan sürekli cümlelerle eğitildiğinde CTC boşluk tuzağına düşer. Sırasıyla izole kelimeler (Word), 2-4 kelimelik ifadeler (Phrase) ve ardından sürekli cümleler (Continuous) aşamaları uygulanmalıdır.

