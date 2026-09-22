# Birincil Araştırma Kaynakları (Primary Literature)

**Candidate Sürümü:** `c0.4.0`  
**Son Güncelleme:** 2026-09-18  
**Kapsam:** Görsel Konuşma Tanıma (VSR), Zamansal Modelleme, CTC Çıkarımı, Düşük-Veri Öğrenmesi ve Türkçeye Özgü Morfolojik Yapı.

---

## 1. Görsel Ön Katman ve Ön Eğitim (Visual Frontend & Transfer Learning)

### 1.1. Auto-AVSR: Visual Speech Recognition with Hybrid CTC/Attention
- **Referans:** Ma, P., Petridis, S., & Pantic, M. (2023). *Visual Speech Recognition for Multiple Languages in the Wild*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / ICASSP 2023.
- **Odak / Katkı:** 3D-ResNet18 ön katman (3D conv + 2D ResNet blokları) ile dudak ROI piksellerinden $B \times T \times 512$ uzamsal-zamansal özellik çıkarımı. LRS3/VoxCeleb2 üzerinde eğitilmiş ön-eğitimli ağırlıkların transfer edilebilirliği.
- **Bu Projedeki Kanıt & Rol:**
  - `c0.4.0` modelinde `VisualFrontend` Auto-AVSR (`vsr_trlrs3_base.pth`) ön-eğitimli ağırlıklarından 114 tensörü birebir aktarır (D2, D4).
  - Sıfırdan eğitime kıyasla ön-eğitimli ağırlıklar olmadan modelin Türkçe veride tam blank çöküşüne uğradığı ispatlanmıştır.
  - BatchNorm katmanlarının eval modunda tutulması (`eval_mode=True`) az konuşmacılı düşük-veri rejiminde istatistiklerin bozulmasını önlemiştir.

### 1.3. AV-HubERT & Multilingual AV-HubERT (mAV-HubERT / MuAViC)
- **Referans:** Shi, B., Hsu, W. N., Toda, T., & Mohamed, A. (2022). *Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction*. ICLR 2022 / TPAMI 2023. Anand, P. et al. (2023). *MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Translation*. Interspeech 2023.
- **Odak / Katkı:** Görsel ve işitsel modalitelerin maskeli küme tahmini (masked cluster prediction) ile kendi kendini denetimli (self-supervised) ön eğitimi. 30 saat veya hatta 1 saatlik etiketli hedef dil verisinde dahi sıfırdan eğitime göre 20-30% mutlak WER iyileşmesi.
- **Bu Projedeki Kanıt & Rol:**
  - Mevcut 4.38 saatlik ve 5 eğitim konuşmacılı düşük-veri rejiminde, sıfırdan eğitilen temporal modellerin konuşmacı ezberleme (speaker memorization) tuzağına düşmesi literatürün bu bulgusuyla tam örtüşür.
  - mAV-HubERT tipi çok dilli görsel-işitsel ön eğitilmiş modeller veya Auto-AVSR'ın geniş veri (LRS3/VoxCeleb2) ön eğitimi, az verili rejimlerde görsel artikülasyon temsillerinin bozulmadan transferi için vazgeçilmez temel oluşturur.

### 1.4. Visual Transformer Pre-training (VTP) without Audio
- **Referans:** Kim, B. et al. (2023). *Visual Transformer Pre-training for Video-Only Visual Speech Recognition*. CVPR 2023.
- **Odak / Katkı:** Ses sinyali olmadan, yalnızca sessiz konuşan yüz videoları üzerinde maskeli oto-kodlayıcı (MAE) tabanlı zamansal-uzamsal ön eğitim.
- **Bu Projedeki Kanıt & Rol:**
  - Üretim kısıtı sessiz video olan VSR sistemlerinde, görsel katmanların konuşmacı kimliğine değil dudak artikülasyonuna kilitlenmesini sağlayan ön-eğitim mekanizması.

---

## 2. Zamansal Modelleme ve Düzenlileştirme (Temporal Modeling & Regularization)

### 2.1. Conformer: Convolution-Augmented Transformer for Speech Recognition
- **Referans:** Gulati, A., Qin, J., Chiu, C. C., Parmar, N., Zhang, Y., Yu, J., ... & Pang, R. (2020). *Conformer: Convolution-augmented Transformer for Speech Recognition*. Interspeech 2020.
- **Odak / Katkı:** Öz-dikkat (self-attention) mekanizmasının küresel bağlam yakalama gücü ile derinlemesine ayrılabilir evrişimlerin (depthwise separable convolution) yerel bağlam modelleme gücünün birleşimi.
- **Bu Projedeki Kanıt & Rol:**
  - BiGRU baseline'ına karşı Conformer mimarisi (4 katman, $d_{model}=512$, 8 kafa) 1.325 Türkçe klip üzerinde Val Loss'u 3.09'dan 2.7399'a düşürmüş ve Greedy CER'i %89'dan %73.49'a çekmiştir (D4, D8).
  - Az konuşmacılı (5 konuşmacı) rejimde 13.7M parametreli Conformer'ın 2. epoch'tan (~330 adım) sonra aşırı uyuma girdiği ve validasyon kaybının diverjans yaptığı ampirik olarak saptanmıştır (D8, D10, D12).
  - D19'daki kompakt conformer deneyinin teknik bir confound (25 dondurulmuş rastgele tensör) içerdiği D20'de tespit edilmiş; modelin `out_dim=512` ön katmanı koruyan `Linear(512, d_model)` köprüsü ile düzeltilmesi sağlanmıştır.

### 2.2. SpecAugment: A Simple Data Augmentation Method for ASR
- **Referans:** Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*. Interspeech 2019.
- **Odak / Katkı:** Zaman ve frekans/kanal maskelemesiyle akustik temsillerin düzenlileştirilmesi.
- **Bu Projedeki Kanıt & Rol:**
  - `REG-001` probe'unda 3D-ResNet çıktısı üzerinde feature-level SpecAugment (zaman maskelemesi $\le 15$ kare, kanal maskelemesi $\le 48$ kanal) denenmiş, ancak 5 konuşmacılı rejimde 2-epoch aşırı uyum diverjansını tek başına kıramamıştır (Best Val Loss 2.7481 vs baseline 2.7399). Literatürün öngördüğü regülarizasyon etkisi aşırı düşük konuşmacı varyansında tek başına yetersiz kalmıştır (D12).

### 2.3. Cross-Modal Audio-to-Visual Teacher Distillation
- **Referans:** Zhao, Y. et al. (2020). *Hearing Lips: Improving Lip Reading by Distilling Speech Recognizers*. AAAI 2020. Haliassos, A. et al. (2023). *Jointly Learning to See and Hear: Cross-Modal Alignment for Visual Speech Recognition*. CVPR 2023.
- **Odak / Katkı:** Eğitim sırasında konuşma sesinin (audio) mevcut olduğu ortamlarda, önceden eğitilmiş güçlü bir ASR öğretmen modelinin (ör. Whisper veya Wav2Vec2-XLSR) gizli katman veya posterior olasılıklarının sessiz VSR öğrenci modeline damıtılması (distillation).
- **Bu Projedeki Kanıt & Rol:**
  - Üretim girdisi sessiz video kalmak şartıyla, eğitim aşamasında sesli modaliteden görsel zamansal kodlayıcıya yumuşak hedef (soft target) aktarımı konuşmacı yüz ezberini radikal biçimde engeller. Mevcut veri havuzunda `mouth.mp4` dosyaları sessiz kırpılmış olduğundan ham videolardan ses geri kazanımı veya cross-lingual transfer alternatifleri araştırma planında tutulmalıdır.

---

## 3. CTC Kaybı, Hedef Tokenizer ve Deşifre (CTC, Targets & Decoding)

### 3.1. Connectionist Temporal Classification (CTC)
- **Referans:** Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. ICML 2006.
- **Odak / Katkı:** Önceden hizalanmamış girdi-çıktı dizilerinde boşluk (`<blank>`) sembolü ve dinamik programlama (Forward-Backward) ile hizalama gerektirmeyen eğitim.
- **Bu Projedeki Rol:**
  - Çıktı uzayı 31 karakterlik Türkçe alfabe + boşluk (`<blank>`) olarak yapılandırılmıştır (D2).

### 3.2. CTC Blank Penalty and Temperature Adjustment in Low-Resource Regimes
- **Referans:** Kanda, N., Wu, X., & Li, J. (2021). *Investigation of Blank Penalty for End-to-End Automatic Speech Recognition*. ICASSP 2021.
- **Odak / Katkı:** CTC modellerinin az veri rejiminde aşırı güvenli (over-confident) boşluk sembolüne kilitlenmesi ve silme hatalarının (deletions) patlaması. Çıkarım anında boşluk logitlerinden ceza düşülerek (`logits[:, blank_idx] -= blank_penalty`) karakter emisyonunun teşvik edilmesi.
- **Bu Projedeki Kanıt & Rol:**
  - `DEC-004` ve `D11` deneylerinde `blank_penalty = 1.2` kalibrasyonuyla silme hataları %62.7 düşürülmüş ($D=3463 \to 1290$), Greedy CER %89.87'den %73.49'a indirilmiştir.
  - Keyword Spotter modülünde `blank_penalty = 1.2` kalibrasyonu recall'ı ikiye katlayarak F1 skorunu 0.0747'ye taşımıştır (D15).

### 3.3. Hedef Birimleri: Fonem, Visem, Karakter ve Alt-Kelime (BPE) Karşılaştırması
- **Referans:** Fisher, C. G. (1968). *Confusions among visually perceived consonants*. JSHR. Hazen, T. J. (2006). *Visual model structures and input features for audio-visual speech recognition*. ICASSP 2006.
- **Odak / Katkı:**
  - **Visem:** Görsel olarak eşdeğer sesler (ör. bilabial /b, p, m/, labiodental /f, v/, alveolar /d, t, n, s, z/). Visem seviyesinde CTC kaybı akustik belirsizliği azaltır; ancak metne geri dönüş (inversion) birebir olmadığından güçlü bir dil modeli zorunludur.
  - **Karakter (Mevcut):** 31 harf. Türkçe fonetik imla dili olduğundan birebir harf-ses uyumu yüksektir ve OOV sorunu yoktur. Ancak sessiz videoda görsel homofenler (homophenes) harf bazında ayırt edilemediğinden silme ve yerine koyma hataları üretir.
  - **Alt-Kelime (BPE):** Sondan eklemeli morfolojide kök+ek ayrımı sağlasa da, <5 saatlik düşük veride token başına düşen örneklem sayısı çok küçüldüğünden CTC boşluk çöküşünü şiddetlendirir.
- **Bu Projedeki Kanıt & Rol:**
  - Kanonik modelde Karakter CTC eğitim hedefi korunmuş; Keyword Spotting katmanında visem toleransı (`viseme_tolerance=True`) uygulanarak bilabial eşdeğerlik çözülmüştür (D15'te TP 3'ten 31'e çıkmıştır).

---

## 4. Türkçe Dilbilimsel Yapı, Uzun Klipler ve Düşük-Veri Sınırları

### 4.1. Turkish Grammar & Agglutinative Morphology
- **Referans:** Göksel, A., & Kerslake, C. (2005). *Turkish: A Comprehensive Grammar*. Routledge.
- **Odak / Katkı:** Türkçenin sondan eklemeli yapısı, ses uyumu ve türetimsel/çekimsel ek zenginliği.
- **Bu Projedeki Kanıt & Rol:**
  - En sık kullanılan 500 Türkçe kök kelime kümesi, sürekli konuşma korpusundaki kelimelerin token yoğunluğu bakımından yalnızca **%43.1**'ini karşılayabilmektedir (Kelimelerin %56.9'u sözlük dışıdır / OOV).
  - Katı Trie beam search çözücüsünün WER'i %98'in altına indirememesinin temel nedeni bu morfolojik OOV tavanıdır (D14). CTC posterior spotter ise kelime sınırlarından bağımsız çalıştığı için hedef kelimeleri başarıyla yakalayabilmektedir (D15).

### 4.2. Düşük-Veri Konuşmacı Aşırı Uyumu ve Örneklem Verimliliği
- **Referans:** Haliassos, A., Ma, P., Petridis, S., & Pantic, M. (2023). *Jointly Learning to See and Hear: Cross-Modal Alignment for Visual Speech Recognition*. CVPR 2023.
- **Odak / Katkı:** Az konuşmacılı görsel konuşma korpuslarında zamansal modellerin yüz morfolojisi ve dudak devinim ritmini hızla ezberlemesi.
- **Bu Projedeki Kanıt & Rol:**
  - 1.63 saat ve 5 konuşmacılık mevcut korpusumuzda 2-epoch (~330 adım) tavanı oluşmuş; frontend dondurma, SpecAugment veya kompakt conformer mimarilerinin bu tavanı aşamadığı gözlemlenmiştir. Ancak D19'daki kompakt conformer deneyinin teknik confound taşıdığı D20'de gösterilmiş ve temiz başlangıçla yeniden probe edilmiştir.

### 4.3. Uzun Klip Kullanımı: Sıralama, Kovalamaca (Bucketing) ve Kayan Pencere
- **Referans:** Khoshfetrat Pakazad, P. et al. (2020). *Sequence Bucketing and Dynamic Batching for Memory-Efficient End-to-End Speech Recognition*. Interspeech 2020.
- **Odak / Katkı:** Değişken uzunluklu video dizilerinde sıfır-padding israfını önlemek ve VRAM sınırlarını aşmadan uzun cümleleri eğitmek için dizilerin süreye göre gruplanması (sequence bucketing) ve dinamik batch boyutlandırması.
- **Bu Projedeki Kanıt & Rol:**
  - `train.csv` içindeki 356 klip ($> 8.0$s) elendiğinde toplam eğitim süresinin **%41.52'si (1.15 saat)** devre dışı kalmaktadır.
  - Bu 1.15 saatlik veri aynı 5 konuşmacıya ait olsa dahi zengin Türkçe sözcük dağarcığı, hece dizilimleri ve koartikülasyon varyasyonu içerir. Sequence bucketing veya padding-verimli dinamik batching ile bu verinin eğitime dahil edilmesi yüksek bilgi değerine sahiptir.

## 2026-09-22 Large-data yön taraması

Aşağıdaki kaynaklar META-001 kararının ampirik kanıtı değildir; c0.5'te hangi
mekanizmaların açık tutulması gerektiğini belirleyen **literature context**'tir.

### Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels
- **ArXiv:** 2303.14307
- Daha büyük ve otomatik etiketlenmiş AVSpeech/VoxCeleb2 verisinin VSR/AVSR
  performansını iyileştirebildiğini gösterir.
- **Projeye etkisi:** D24'ü "data scaling işe yaramaz" diye yorumlama; asıl yeniden
  test edilmesi gereken şey **daha çeşitli data scaling**'dir.

### Multilingual Audio-Visual Speech Recognition with Hybrid CTC/RNN-T Fast Conformer
- **ArXiv:** 2405.12983
- Fast Conformer üzerinde hybrid CTC/RNN-T ve daha büyük multilingual AV data kullanır.
- **Projeye etkisi:** c0.5'te plain CTC objective kalıcı varsayım değildir. Ancak RNN-T
  ilk probe için yüksek-confound olabilir; önce daha kontrollü objective değişiklikleri
  tercih edilebilir.

### Audio-Visual Efficient Conformer for Robust Speech Recognition
- **ArXiv:** 2301.01456
- Intermediate CTC losses ve residual conditioning ile CTC'nin conditional-independence
  sınırlamasını azaltmaya çalışır.
- **Projeye etkisi:** Eğer c0.5 baseline'da META-001'deki loss→CER ayrışması sürerse,
  aynı Conformer ailesinde **Inter-CTC** düşük-confound ilk mekanizma probe'u olabilir.

### SwinLip: An Efficient Visual Speech Encoder for Lip Reading Using Swin Transformer
- **ArXiv:** 2505.04394
- Daha verimli visual speech encoder alternatifi sunar.
- **Projeye etkisi:** Large-data subgroup failure'ları visual representation'a işaret
  ederse frontend ailesi yeniden açılmalıdır; bugün için kabul edilmiş değişiklik değildir.

### VALLR: Visual ASR Language Model for Lip Reading
- **ArXiv:** 2503.21408
- Phoneme-centric visual recognition + language-model reconstruction yaklaşımı önerir.
- **Projeye etkisi:** Viseme/phoneme ambiguity için güçlü fakat daha büyük kavramsal
  değişikliktir; kontrollü c0.5 baseline/Inter-CTC sorularından sonra değerlendirilmesi
  daha doğru olur.
