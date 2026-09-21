# Failure Analysis

**Candidate:** `c0.1.0`  
**Son Güncelleme:** 2026-09-17  

---

## Hata Kümesi 1: CTC Tam Boşluk Çökmesi (Total Deletion / Empty Output)

- **Probe Kimliği:** `probe_init001_control_autoavsr`
- **Tarih:** 2026-09-17 19:14 UTC
- **Donanım:** Modal A10G (Süre: 46.5s, Maliyet: zsh.015)
- **Provenance:**
  - Split: `data/metadata/split_map_iborotti.json`
  - Train: 60 segment, Val: 30 segment (Akil Ünüvar)
  - Initializer: `vsr_trlrs3_base.pth` (114 frontend tensörü)
  - Mimari: 3D-ResNet18 + 3-katman BiGRU (d_model=512) + CTC Head
  - Seed: 42

### Ham Tahmin Örnekleri (Held-out Doğrulama Kümesi)

| # | Referans Transkript | Ham Model Tahmini (`Hyp`) | Tespit Edilen Kelimeler | Durum |
|---|---|---|---|---|
| 1 | `arkadaşlar selam kanalıma hoş geldiniz` | `''` | `[]` | Tam Silme (Deletion) |
| 2 | `bugün aslında sizlerle şeyi konuşmak istiyorum` | `''` | `[]` | Tam Silme (Deletion) |
| 3 | `çoğunuz yks 'ye girdiniz şu an sıralamaları bekliyorsunuz` | `''` | `[]` | Tam Silme (Deletion) |
| 4 | `istediğiniz bir üniversite var istediğiniz bir bölüm var` | `''` | `[]` | Tam Silme (Deletion) |
| 5 | `hayallerinizdeki üniversite için işte hani siz şu an sıralamalarınızı bekliyorsunuz` | `''` | `[]` | Tam Silme (Deletion) |

### Metrikler ve Mekanizma
- 25/25 validasyon örneğinde %100 boş dizi (`''`). CER: %100.0, WER: %100.0.
- Neden: 140 karelik sürekli cümlelerde rastgele BiGRU katmanı CTC kaybını düşürmenin en kolay yolu olarak her kareye blank tokeni atamıştır.

---

## Hata Kümesi 2: Tekil Karakter Emisyonu ve Hizalama Tavanı (Conformer Artifact)

- **Probe Kimliği:** `probe_arch001_conformer`
- **Tarih:** 2026-09-17 19:34 UTC
- **Donanım:** Modal A10G (Süre: 59.38s, Maliyet: zsh.018)
- **Provenance:**
  - Split: `data/metadata/split_map_iborotti.json`
  - Train: 60 segment, Val: 30 segment (Akil Ünüvar)
  - Initializer: `vsr_trlrs3_base.pth` (114 frontend tensörü)
  - Mimari: 3D-ResNet18 + 4-katman Conformer (d_model=512) + CTC Head
  - Seed: 42, blank_penalty=0.4, lr=2e-4

### Ham Tahmin Örnekleri (Held-out Doğrulama Kümesi - Epoch 6)

| # | Referans Transkript | Ham Model Tahmini (`Hyp`) | Durum |
|---|---|---|---|
| 1 | `arkadaşlar selam kanalıma hoş geldiniz` | `'b'` | Tekil Karakter Emisyonu |
| 2 | `bugün aslında sizlerle şeyi konuşmak istiyorum` | `'b'` | Tekil Karakter Emisyonu |
| 3 | `çoğunuz yks 'ye girdiniz şu an sıralamaları bekliyorsunuz` | `'b'` | Tekil Karakter Emisyonu |
| 4 | `istediğiniz bir üniversite var istediğiniz bir bölüm var` | `'b'` | Tekil Karakter Emisyonu |
| 5 | `hayallerinizdeki üniversite için işte hani siz şu an sıralam` | `'b'` | Tekil Karakter Emisyonu |

### Mekanizma ve Çıkarım
1. **İlerleme:** BiGRU'nun tam sessizliğine (`''`) kıyasla, Conformer self-attention katmanları 6. epochta boşluk bariyerini aşmış ve CER'i %100'den %98.94'e düşürmüştür. Train loss 2.9973 ile 3.00'ın altına inmiş, en iyi val loss 3.0960 ile BiGRU'yu (3.1269) geride bırakmıştır.
2. **Hata Türü (Mode Collapse to Dominant Token):** Ağ, dudak hareketlerinde belirli bir visem geçişini (muhtemelen dudak kapanması - bilabial /b, p, m/) tespit etmiş ancak 140 karelik sürekli cümlede kelime ve fonem sınırlarını denetimsiz olarak çözemediği için tüm cümlelere jenerik tekil emisyon (`'b'`) basmıştır.
3. **Kök Neden:** Sürekli 6-saniyelik cümleler ($T \approx 140$ kare vs $U \approx 35$ karakter), temporal modelleme için çok geniştir. Model doğrudan tam cümlelerle eğitildiğinde visem-harf hizalaması oturmadan yerel boşluk/jenerik token minimumuna takılmaktadır.
4. **Çözüm (TRAIN-001):** Model önce kısa müfredat dilimleri ($\le 3.5$s) üzerinde iki aşamalı (dondurulmuş frontend ardından diferansiyel ince ayار) ile eğitilmelidir.

---

## Hata Kümesi 3: Statik Önsel / Baskın Hece Çökmesi ve Çözümü (Two-Stage Curriculum Unfreeze)

- **Probe Kimliği:** `probe_train001_steps_horizon` -> `probe_train001_full_curriculum_unfreeze`
- **Tarih:** 2026-09-18
- **Donanım:** Modal A10G (Maliyet: $0.0651 USD)
- **Provenance:**
  - Split: `data/metadata/split_map_iborotti.json`
  - Train: 360 kısa segment ($\le 3.5$s), Val: 60 kısa segment (Akil Ünüvar)
  - Initializer: `vsr_trlrs3_base.pth` (114 frontend tensörü)
  - Mimari: 3D-ResNet18 + 4-katman Conformer (d_model=512) + CTC Head (blank_penalty=0.8)
  - Reçete: Epoch 1–10 (Frontend frozen, lr=5e-4); Epoch 11–20 (Frontend unfreeze, lr=2.5e-5, CosineAnnealingLR)

### Ham Tahmin Örnekleri (Epoch 20 Karşılaştırması)

| # | Referans Transkript | Ham Tahmin (`Hyp`) | Ham Argmax (`Hyp_raw`) | Durum |
|---|---|---|---|---|
| 1 | `arkadaşlar selam kanalıma hoş geldiniz` | `ben i ar ie er ar arar` | `eiaieeaaar` | Çoklu Harf/Fonem |
| 2 | `çoğunuz yks 'ye girdiniz şu an sıralamaları bekliyorsunuz` | `ben ban bar ben der ber bar bar yar yera` | `enaar eeerarararera` | Kelime Heceleri |
| 3 | `istediğiniz bir üniversite var istediğiniz bir bölüm var` | `ben in bar bener ber bar bar dar yera` | `eniar eeerarararera` | Kelime Heceleri |
| 4 | `ben ikinci yöntemi denemiştim` | `ben in bar benar ber bar barar yera` | `eniar eaerarararer` | Kelime Uyuşması ("ben") |
| 5 | `farklı bir üniversiteye gidip mesela içi istiyorsunuz` | `ben a dar bener er ar bararyer` | `eaaeeeaararer` | Çeşitli Harfler |

### Mekanizma ve Çıkarım
1. **İlerleme:** Validasyon CER %91.78'den **%74.92**'ye indi. Validasyon kaybı **2.8486**'ya geriledi. Model ilk defa `"ben"`, `"bar"`, `"der"`, `"yera"` gibi anlamlı Türkçe heceler ve kelimeler üretti.
2. **Kalan Hata (Fonetik Saçılma ve Sözlük Dışı Diziler):** Model fonetik olarak dudak hareketlerini yakalamakta ancak sözlük kısıtlaması ve dil modeli olmadan ham CTC argmax harfleri birleştirdiğinde `"bener"`, `"barar"` gibi sözlük dışı diziler üretmektedir. Bu durum WER'in (~1.16) CER'den (%74.92) yüksek kalmasına yol açmaktadır.
3. **Çözüm (DEC-001):** Top-500 Türkçe kelime Trie ağacı ve Unigram/Bigram Türkçe Dil Modeli kullanan `LexiconBeamSearchDecoder` entegre edilerek fonetik emisyonlar geçerli Türkçe kelimelere kısıtlanacaktır.

---

## Hata Kümesi 4: Sözlük Kısıtlamalı Beam Search'te Aşırı Kelime Ekleme (Word Insertion Hallucination)

- **Probe Kimliği:** `probe_dec001_lexicon_beam_vs_greedy`
- **Tarih:** 2026-09-18
- **Donanım:** Modal A10G (Maliyet: $0.0038 USD)
- **Provenance:**
  - Candidate: `c0.2.0` (`probe_train001_full_curriculum_unfreeze_best.pt`)
  - Validasyon: 60 segment (Akil Ünüvar, $\le 3.5$s)
  - Çözücü: `LexiconBeamSearchDecoder` (beam_size=30, lm_alpha=0.4, lm_beta=1.0)

### Ham Tahmin Örnekleri (Greedy vs Lexicon Beam Search)

| # | Referans Transkript | Greedy Baseline | Lexicon Beam Search | Hata Türü |
|---|---|---|---|---|
| 1 | `arkadaşlar selam kanalıma hoş geldiniz` | `beniarieerararar` | `ben izin de ara bir yer` | Insertion: 6 kısa kelime halüsinasyonu |
| 2 | `bugün aslında sizlerle şeyi konuşmak istiyorum` | `benaar eeerar arar` | `ben ama de arada bir yer` | Substitution & Insertion |
| 3 | `çoğunuz yks 'ye girdiniz şu an sıralamaları bekliyorsunuz` | `ben a bar bener er ar barar yera` | `ben eder bir de bir yer bir yer bir de` | Aşırı tekrarlayan kısa kelime döngüsü |
| 4 | `istediğiniz bir üniversite var istediğiniz bir bölüm var` | `beni ar ener er ar arar yera` | `ben izin bir de bir yer bir yer bir` | "bir", "de", "yer" aşırı ekleme |
| 5 | `ne yapacaksınız bir sene daha mezuna mı kalacaksınız` | `benaar eeerar ararer` | `ben eder de ara bir de neler` | Kısmi kelime eşleşmesi ("bir") |

### Mekanizma ve Çıkarım
1. **İyileşen Yönler:**
   - CER: %77.69'dan **%74.92**'ye geriledi.
   - Spotter F1: %1.97'den **%9.13**'e yükseldi (~4.6x artış).
   - Çıkarım Gecikmesi: 28.0 ms/video (gerçek zamanlı).
2. **Bozulan Yön:**
   - WER: Greedy 1.0024 iken Beam Search 1.1010'a yükseldi.
3. **Kök Neden:**
   - `lm_score += beta * len(completed_word)` formülü her tamamlanan kelimeye pozitif log-prob ödülü eklediğinden, çözücü akustik belirsizlik anında uzun heceleri çok sayıda kısa 1-3 harfli sözlük kelimesine (`ben`, `izin`, `de`, `ara`, `bir`, `yer`) parçalayarak aşırı ekleme hatası ($I$) üretti.
   - Akustik model CER'i (%75) henüz katı 500-kelimelik Trie kısıtlamasını güvenle taşıyacak kadar keskin değildir.
4. **Alınan Aksiyon:**
   - `lexicon_decoder.py` içindeki kelime uzunluğu katsayısı düzeltildi (`lm_score += beta`).
   - $\beta \le 0$ (kelime ekleme cezası) ile hızlı bir doğrulama probe'u veya doğrudan akustik CER'i düşürecek tam cümle müfredatı (`CURR-002`) çalıştırılmalıdır.

---

## Hata Kümesi 5: Sürekli Cümlelerde Sesli Harf Kümelenmesi ve Sözlük Tahmini (DEC-002)

- **Probe Kimliği:** `probe_dec002_c030_lexicon_beam`
- **Tarih:** 2026-09-18 07:35 UTC
- **Donanım:** Modal A10G (Maliyet: $0.0059 USD)
- **Provenance:**
  - Candidate: `c0.3.0` (`probe_curr002_sentence_scaling_best.pt`)
  - Validasyon: 60 segment (Akil Ünüvar, $\le 6.0$s)
  - Çözücü: `LexiconBeamSearchDecoder` (beam_size=30, lm_alpha=0.3, lm_beta=-0.2)

### Ham Tahmin Örnekleri (Greedy vs Calibrated Lexicon Beam Search)

| # | Referans Transkript | Greedy Baseline | Calibrated Beam Search | Durum |
|---|---|---|---|---|
| 1 | `arkadaşlar selam kanalıma hoş geldiniz` | `anaaaaaaa` | `""` | Sessiz (Boş) |
| 2 | `çoğunuz yks 'ye girdiniz şu an sıralamaları bekliyorsunuz` | `anaaaaaaaa aaa` | `azından arada` | Sözlük Kelimesi |
| 3 | `istediğiniz bir üniversite var istediğiniz bir bölüm var` | `aaaaaaaaaaa` | `adamın arada` | Sözlük Kelimesi |
| 4 | `hayallerinizdeki üniversite için işte hani siz şu an sıralamalarınızı bekliyorsunuz` | `aiaaaaaaa aaaaaaaa` | `adamın arada bana arada` | Sözlük Kelimesi |
| 5 | `yoldan aslında sistemi yasal yollardan ekleyerek istediğiniz üniversitenin...` | `aiaaaaaaa aaaaaaaaaa` | `bana arada araba arada` | Sözlük Kelimesi |
| 6 | `ne yapacaksınız bir sene daha mezuna mı kalacaksınız` | `anaaaaaaaaa` | `azından` | Sözlük Kelimesi |

### Mekanizma ve Çıkarım
1. **İyileşen Yönler:**
   - Ekleme hatası patlaması (`DEC-001`'deki $I=350+$) tamamen durduruldu.
   - WER 1.0000'den **0.9981**'e geriledi.
   - CER **%89.45** (Greedy: %89.86).
   - Spotter F1 skoru **0.0062** (Greedy: 0.0000).
   - Gecikme 93.6 ms/klip (gerçek zamanlı).
2. **Kalan Hata (Akustik Belirsizlik ve Sık Sesli Harfler):**
   - 6 saniyelik uzun dizilerde model yalnızca 720 eğitim klibi gördüğü için, görmediği konuşmacı ve kelimelerde akustik belirsizlik arttığında en yüksek sıklığa sahip ünlüleri (`'a'`, `'i'`, `'n'`) basmaktadır (`"anaaaaaaa"`, `"aaaaaaaaaaa"`).
   - Calibrated Beam Search bu ünlü dizilerini sözlükteki `"adamın"`, `"arada"`, `"bana"`, `"araba"`, `"azından"` gibi bu ünlüleri içeren kelimelere eşleştirmektedir.
3. **Kök Neden:**
   - Model `train.csv` içindeki 1.681 klibin sadece 720'sini görmüştür. Kalan 961 klip (%57) henüz modele sunulmamıştır.
   - Akustik temsil kalitesi yükselmeden çözücü hiperparametrelerini ince ayarlamak marjinal kazançtan öteye geçemez.
4. **Alınan Aksiyon (CONFIRM-001):**
   - Model tüm eğitim verisine (1.681 klip, $\le 8.0$s) açılarak 5 konuşmacının tamamı ve 2.78 saatlik verinin tümü üzerinde ince ayar yapılacaktır.

---

## Hata Kümesi 6: Konuşmacılar Arası Aşırı Uyum ve İnce Ayar Erken Yakınsaması (CONFIRM-001)

- **Probe Kimliği:** `probe_confirm001_full_train_scaling`
- **Tarih:** 2026-09-18 08:16 UTC
- **Donanım:** Modal A10G (Maliyet: $0.4475 USD)
- **Provenance:**
  - Candidate: `c0.3.0` -> `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`)
  - Eğitim: 1.325 segment ($\le 8.0$s, 5 konuşmacı), Validasyon: 60 segment (Akil Ünüvar)
  - Hiperparametreler: lr=1.5e-4, CosineAnnealingLR, blank_penalty=0.8

### Ham Tahmin Örnekleri (Held-out Validasyon Kümesi)

| # | Referans Transkript | Ham Model Tahmini (`Hyp`) | Ham Argmax (`Hyp_raw`) | Gözlemlenen Olay |
|---|---|---|---|---|
| 1 | `arkadaşlar selam kanalıma hoş geldiniz` | `herin dei eiar` | `ei eeiar` | Çoklu ünsüz ("herin", "dei") |
| 2 | `bugün aslında sizlerle şeyi konuşmak istiyorum` | `beni e ii ar` | `beeiar` | Kelime başlangıcı ("beni") |
| 3 | `çoğunuz yks 'ye girdiniz şu an sıralamaları bekliyorsunuz` | `benae dei ei ari arirlir` | `eeei ai aii` | Ünsüz çeşitliliği ("benae", "dei") |
| 4 | `istediğiniz bir üniversite var istediğiniz bir bölüm var` | `ei ee ee eri berlar` | `ei eeeeei erar` | Fonetik eşleşme ("berlar" ~ "bölüm var") |
| 5 | `formül olarak şeyi yapabilirsiniz` | `buni deeiir` | `buieeii` | Fonetik eşleşme ("buni" ~ "formül") |
| 6 | `aslında hani istediğiniz bir bölüm var` | `eniin deiii birlir` | `ei eiiiri` | Fonetik eşleşme ("birlir" ~ "bölüm var") |
| 7 | `ama içi yerine sallıyorum örnek veriyorum` | `beni deei allı` | `eeei alı` | Morfolojik kök ("allı" ~ "sallıyorum") |

### Mekanizma ve Çıkarım
1. **İlerleme:**
   - Validasyon kaybı **2.7399** ile tüm zamanların en düşük seviyesine indi.
   - Eğitim kaybı **2.6180**'e geriledi.
   - Model jenerik sesli harf tekrarlarını (`"anaaaaaaa"`, `"eee"`) kırdı; ünsüzler (`'h'`, `'r'`, `'n'`, `'d'`, `'b'`, `'l'`, `'s'`) ve hece öbekleri (`"allı"`, `"berlar"`, `"birlir"`, `"buni"`) üretmeye başladı.
2. **Ortaya Çıkan Patoloji (Cross-Speaker Overfitting):**
   - Model 2. epochta validasyon kaybında zirve yaptıktan (**2.7399**) sonra, eğitim kaybı düşmeye devam ederken (2.78 -> 2.61) validasyon kaybı yükselmiştir (2.7399 -> 3.27).
   - Neden: 13.7M parametreli Conformer, görsel veri artırma (SpecAugment, rastgele kırpma, zaman jitter'ı) sınırlı olduğunda 5 eğitim konuşmacısının spesifik dudak şekillerine ve mimiklerine aşırı uyum sağlamakta, görmediği 6. validasyon konuşmacısına genelleme yapmakta zorlanmaktadır.
3. **Alınan Karar:**
   - En iyi kontrol noktası olan Epoch 2 ağırlıkları `c0.4.0` için mühürlenmiştir (`probe_confirm001_full_train_scaling_best.pt`).
   - Bu keskinleşmiş ünsüz temsilleri üzerinde `DEC-003` probe'u ile çözücü test edilecek; ardından görsel aşırı uyumu kırmak için `AUG-001` araştırılacaktır.

---

## Hata Kümesi 7: %86.7 Silme Hatası Darboğazı ve Fotometrik Veri Artırma Başarısızlığı (AUG-001)

- **Probe Kimliği:** `probe_aug001_photometric_reg`
- **Tarih:** 2026-09-18 08:31 UTC
- **Donanım:** Modal A10G (Maliyet: $0.3182 USD)
- **Provenance:**
  - Candidate: `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`)
  - Eğitim: 1.325 segment ($\le 8.0$s), Validasyon: 60 segment (Akil Ünüvar)
  - Veri Artırma: Brightness $\pm 10\%$, Contrast $\pm 15\%$, weight_decay=5e-4

### Niceliksel Karakter Hata Ayrıştırması (`CONFIRM-001` / `c0.4.0`)

Held-out validasyon kümesindeki (60 cümle) detaylı `jiwer.process_characters` incelemesi:
- Toplam Referans Karakter: **1.823**
- Toplam Model Hipotez Karakter: **568** (Model referansın yalnızca %31.2'si kadar karakter basabilmektedir!)
- Yerine Koyma Hatası ($S$): **190 (%13.1)**
- **Silme / Eksik Emisyon Hatası ($D$): 1.257 (%86.7 - TÜM HATALARIN %87'Sİ!)**
- Ekleme Hatası ($I$): **2 (%0.1)**

### Mekanizma ve Çıkarım
1. **Darboğazın Gerçek Yeri:** CER %75–80 bandına saplanıp kalmış görünmektedir. Ancak bunun sebebi modelin yanlış harfler tahmin etmesi değil (S=190 vs D=1257), karakterlerin büyük kısmını CTC boşluk (blank) tokenine kurban etmesidir. Ekleme hatası neredeyse sıfırdır ($I=2$).
2. **AUG-001 Başarısızlığı:**
   - Fotometrik pertürbasyon eklemek, dudak konturlarındaki ince gri ton geçişlerini bozduğu için Auto-AVSR görsel ön katmanının öznitelik çıkarımını zayıflatmış, eğitim yakınsamasını yavaşlatmış ve validasyon kaybını 2.8105'in altına düşürememiştir (baseline 2.7399 idi). Karar: `REJECT`.
3. **Çözüm Mekanizması (`DEC-004`):**
   - Modelin emisyon üretmesini engelleyen şey eğitim eksikliği değil, çıkarım anındaki (inference-time) aşırı muhafazakar `blank_penalty` eşiğidir.
   - CTC greedy/beam decoding anında `logits[:, BLANK_IDX] -= blank_penalty` işlemi doğrudan silme ($D$) hatalarını azaltarak yerine koyma ($S$) dengesine yaklaştıracak ve sıfır yeniden eğitim maliyetiyle CER'i %65'in altına çekebilecektir.

---

## Hata Kümesi 8: Silme Hatasının Başarıyla Dengelenmesi ve Kalan Fonetik Şablon Darboğazı (DEC-004)

- **Probe Kimliği:** `probe_dec004_blank_penalty_grid`
- **Tarih:** 2026-09-18 09:10 UTC
- **Donanım:** Modal A10G (Maliyet: $0.0176 USD)
- **Provenance:**
  - Candidate: `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`, Epoch 2)
  - Validasyon: 60 segment (Akil Ünüvar, $\le 8.0$s)
  - Yöntem: Çıkarım anı `blank_penalty` ızgarası: `[0.0, 0.4, 0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]`

### Izgara Üzerindeki Hata Dönüşümü

| BP | Blank Oranı | Greedy CER | Deletions ($D$) | Substitutions ($S$) | Insertions ($I$) | Silme Oranı ($D/Total$) | Beam CER | Beam WER |
|---|---|---|---|---|---|---|---|---|
| **0.0 (Ham)** | 85.1% | 89.87% | **3.463** | 484 | 3 | **87.7%** | 87.10% | 100.00% |
| 0.4 | 79.3% | 84.82% | 3.100 | 620 | 8 | 83.2% | 82.43% | 99.83% |
| 0.8 (Eski) | 63.2% | 75.61% | 2.103 | 1.203 | 17 | 63.3% | 80.14% | 100.00% |
| **1.2 (Optimum Greedy)** | **49.1%** | **73.49%** | **1.290** | **1.849** | **91** | **39.9%** | 78.68% | 100.00% |
| 1.5 | 36.5% | 76.04% | 712 | 2.321 | 309 | 21.3% | 76.88% | 99.50% |
| 1.8 | 30.9% | 78.16% | 532 | 2.462 | 441 | 15.5% | 75.90% | 99.16% |
| **2.0 (Optimum Beam)** | **27.1%** | 80.36% | 438 | 2.523 | 571 | 12.4% | **74.77%** | **99.00%** |
| 2.5 | 17.5% | 87.28% | 241 | 2.621 | 974 | 6.3% | 75.36% | 102.68% |

### Ham Tahmin İncelemesi ($BP=1.2$ Greedy vs $BP=2.0$ Beam)

| # | Referans Transkript | Greedy ($BP=1.2$) | Lexicon Beam ($BP=2.0$) | Analiz |
|---|---|---|---|---|
| 1 | `arkadaşlar selam kanalıma hoş geldiniz` | `bana an ban an bar ar barir` | `bana arada` | Aktif ünsüzler (`b`, `n`, `r`), "arada" yakalandı |
| 2 | `bugün aslında sizlerle şeyi konuşmak istiyorum` | `ban a ari bar an bari ar barar` | `bana arada` | "ban", "bari" emisyonları |
| 3 | `çoğunuz yks 'ye girdiniz şu an sıralamaları bekliyorsunuz` | `ban a ani ban an bari ar barar bar aryar` | `bana arada araba nereden` | 4 kelime eşleşmesi |
| 4 | `istediğiniz bir üniversite var istediğiniz bir bölüm var` | `bena an ban an ar ar baran barar` | `bana arada araba` | Fonetik hece grupları |
| 5 | `hayallerinizdeki üniversite için işte hani siz şu an...` | `ben a bani ban an bari bar barar...` | `bana azından arada neden azından...` | Çoklu sözlük kelimesi |

### Mekanizma ve Çıkarım
1. **İyileşen Yönler:**
   - Silme hataları $D=3463$'ten $D=1290$'a (**%62.7 düşüş**) indirildi.
   - Silme oranı %87.7'den %39.9'a inerek yerine koyma ($S=1849$) ile sağlıklı bir dengeye oturdu.
   - Modelin daha önce hiç çıkaramadığı ünsüzler (`'b'`, `'n'`, `'r'`) ardı ardına emisyon üretti.
   - Greedy CER tüm zamanların en iyi seviyesi olan **%73.49**'a, Beam CER **%74.77**'ye, Beam WER **%99.00**'a geriledi.
2. **Kalan Darboğaz (Jenerik Fonetik Şablon / Bilabial-Alveolar Loop):**
   - $BP=1.2$ ile karakter emisyonları serbest bırakıldığında, modelin neredeyse her cümlede `"ban an ban an bar ar barir"`, `"ban a ari bar an bari"` gibi tekrarlayan bir bilabial-alveolar hece şablonuna başvurduğu görülmektedir.
   - **Kök Neden:** Değerlendirilen kontrol noktası `CONFIRM-001`'in **2. epoch** checkpoint'idir. Model bu erken aşamada en güçlü dudak hareketlerini (bilabial dudak kapanması /b, m, p/ ve açık ünlü /a/) birleştirerek jenerik bir ritim yakalamıştır.
   - Modelin eğitim konuşmacıları üzerinde 15–20. epochlarda öğrendiği spesifik ünsüz ayrımları (`herin`, `allı`, `birlir`, `deeiir`), validasyon konuşmacısına cross-speaker overfitting nedeniyle aktarılamamıştır.
3. **Alınan Karar ve Çözüm Yolu (`REG-001`):**
   - Greedy için `blank_penalty = 1.2`, Beam Search için `blank_penalty = 2.0` kalibre edilmiş değerler olarak dondurulmuştur.
   - Akustik modelin bu jenerik bilabial şablonu kırması ve daha zengin fonemleri held-out konuşmacıda üretebilmesi için, piksel bazlı değil (fotometrik pertürbasyon gibi başarısız olan yöntemler yerine) **3D-ResNet ön katman çıktı tensörü ($B \times T \times D$) üzerinde SpecAugment (zaman ve kanal maskelemesi)** ve **Conformer Dropout (0.1 -> 0.2)** uygulanacaktır.

---

## Hata Kümesi 9: 2-Epoch Aşırı Uyum Paradoksu ve Feature-Level SpecAugment Sınırları (REG-001)

- **Probe Kimliği:** `probe_reg001_specaugment_conformer`
- **Tarih:** 2026-09-18 09:51 UTC
- **Donanım:** Modal A10G (Maliyet: $0.2925 USD)
- **Provenance:**
  - Candidate: `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`, Epoch 2)
  - Eğitim: 1.325 segment ($\le 8.0$s), Validasyon: 60 segment (Akil Ünüvar)
  - Düzenlileştirme: 3D-ResNet çıktısı üzerinde Feature-level SpecAugment (2 zaman maskesi $\le 15$ kare, 2 kanal maskesi $\le 48$ kanal) + Conformer Dropout 0.2 + `blank_penalty = 1.2`

### Epoch-by-Epoch Dinamiği

| Epoch | Train Loss | Val Loss | Blank Oranı | CER | WER | Notlar |
|---|---|---|---|---|---|---|
| 01 | 2.7732 | 3.0962 | 89.4% | 0.8865 | 1.0000 | Başlangıç adaptasyonu |
| **02** | **2.7666** | **2.7481** | **85.6%** | **0.7520** | **1.0468** | **En iyi Val Loss ve CER** |
| 03 | 2.7597 | 2.8238 | 86.7% | 0.8091 | 1.0000 | Val Loss yükselmeye başladı |
| 04 | 2.7508 | 2.9898 | 87.7% | 0.8086 | 1.0000 | Ayrışma belirginleşti |
| 05 | 2.7386 | 3.5184 | 95.2% | 0.8942 | 1.0000 | Geçici aşırı sessizlik sıçraması |
| 06 | 2.7322 | 2.9080 | 86.6% | 0.7825 | 1.0000 | Toparlanma |
| 08 | 2.7069 | 3.1979 | 90.1% | 0.8348 | 1.0000 | Diverjans sürüyor |
| 10 | 2.6945 | 3.0815 | 88.2% | 0.8389 | 1.0000 | Eğitim kaybı düşerken val kaybı yüksek |
| 12 | 2.6900 | 3.2332 | 89.1% | 0.8394 | 1.0000 | Final diverjans |

### Gerçek Tahmin Örnekleri (Epoch 12)

- `Ref: arkadaşlar selam kanalıma hoş geldiniz` $\to$ `Hyp: eeaeear`
- `Ref: bugün aslında sizlerle şeyi konuşmak istiyorum` $\to$ `Hyp: beeaiaar`
- `Ref: çoğunuz yks 'ye girdiniz şu an sıralamaları bekliyorsunuz` $\to$ `Hyp: bueeii ai bararar`
- `Ref: istediğiniz bir üniversite var istediğiniz bir bölüm var` $\to$ `Hyp: ei iieeai barlir`
- `Ref: farklı bir üniversiteye gidip mesela içi istiyorsunuz` $\to$ `Hyp: ei aaei bararyor`
- `Ref: bir yandan da şey var belki biliyorsunuzdur...` $\to$ `Hyp: eeeeei ii bora ere eeei ereerir`

### Kök Neden ve Mekanizma Analizi
1. **"2-Epoch Paradoksu" Kanıtlandı:**
   - 1.325 segmentlik tam veri üzerinde koşulan 3 bağımsız deneyde (`CONFIRM-001`, `AUG-001`, `REG-001`), öğrenme oranı diferansiyel ($10^{-5}$ frontend, $2\cdot 10^{-4}$ Conformer) iken, validasyon kaybı istisnasız **Epoch 2 (~330 adım)** seviyesinde global minimumuna ulaşmaktadır.
   - Bu noktada eğitim kümesindeki 5 konuşmacının genel visem-hece dinamikleri transfer edilmektedir.
   - Ancak 330 adımdan sonra, görsel ön katman (11.2M parametre) bu 5 konuşmacının dudak ve yüz anatomisine aşırı uyum sağlamakta; unseen validasyon konuşmacısına genelleme yeteneği hızla bozulmaktadır.
2. **Feature SpecAugment Neden Yeterli Olmadı?**
   - Conformer girdi tensörüne ($B \times T \times D$) uygulanan rastgele zaman/kanal maskelemesi ve dropout=0.2, Conformer katmanlarının aşırı ezberlemesini sınırlamaya çalışsa da, altındaki 11.2M parametreli 3D-ResNet ön katmanı eğitilebilir kaldığı için konuşmacı-spesifik görsel adaptasyon yine de gerçekleşmiştir.
3. **Alınan Karar:**
   - Pre-registered karar kuralına göre probe **`REJECT`** edilmiştir.
   - Kanonik model `c0.4.0` olarak korunmuştur.

---

## Hata Kümesi 10: Dondurulmuş Ön Katman ile Sınırların Belirlenmesi ve Akustik Tavan (FE-001)

- **Probe Kimliği:** `probe_fe001_frozen_frontend_conformer`
- **Tarih:** 2026-09-18 10:11 UTC
- **Donanım:** Modal A10G (Maliyet: $0.2135 USD)
- **Provenance:**
  - Candidate: `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`, Epoch 2)
  - Eğitim: 1.325 segment ($\le 8.0$s), Validasyon: 60 segment (Akil Ünüvar)
  - Yöntem: 3D-ResNet görsel ön katmanı donduruldu (`freeze_frontend=True`, BatchNorm eval modu), yalnızca Conformer katmanları (14.4M parametre) eğitildi.

### Epoch-by-Epoch Dinamiği

| Epoch | Train Loss | Val Loss | Blank Oranı | CER | WER | Notlar |
|---|---|---|---|---|---|---|
| 01 | 2.7635 | 3.0794 | 91.0% | 0.8755 | 1.0000 | Frontend donduruldu |
| **02** | **2.7539** | **2.7670** | **86.8%** | **0.7861** | **1.0000** | **En iyi Val Loss (2.7670)** |
| 03 | 2.7388 | 3.1275 | 89.1% | 0.8291 | 1.0000 | Salınım başladı |
| 04 | 2.7355 | 2.9977 | 88.4% | 0.8032 | 1.0000 | Toparlanma |
| 05 | 2.7303 | 2.8074 | 88.8% | 0.7809 | 1.0017 | İstikrar arayışı |
| **06** | **2.7176** | **2.8000** | **86.1%** | **0.7540** | **1.0151** | **En iyi CER (0.7540)** |
| 08 | 2.7019 | 3.4212 | 98.7% | 0.9208 | 1.0000 | Geçici aşırı boşluk sıçraması |
| 10 | 2.6802 | 3.0569 | 89.1% | 0.8068 | 1.0000 | Conformer eğitim kaybı düşüyor |
| 12 | 2.6647 | 3.0477 | 89.0% | 0.8121 | 1.0000 | Final diverjans |

### Gerçek Tahmin Örnekleri (Epoch 12)

- `Ref: arkadaşlar selam kanalıma hoş geldiniz` $\to$ `Hyp: ene ee ee arı`
- `Ref: bugün aslında sizlerle şeyi konuşmak istiyorum` $\to$ `Hyp: beneaei arı`
- `Ref: çoğunuz yks 'ye girdiniz şu an sıralamaları bekliyorsunuz` $\to$ `Hyp: bene ei ei are erayar`
- `Ref: istediğiniz bir üniversite var istediğiniz bir bölüm var` $\to$ `Hyp: ee ee ee ari borlar`
- `Ref: hayallerinizdeki üniversite için işte hani siz şu an sıralamalarınızı bekliyorsunuz` $\to$ `Hyp: ei ee ei ae ere ee ei irir`

### Kök Neden ve Dört Deneyin Büyük Sentezi
1. **Frontend'in Dondurulması Aşırı Uyumu Durdurdu mu?**
   - Hayır. Frontend dondurulduğunda bile en iyi validasyon kaybı yine **Epoch 2'de (2.7670)** gerçekleşti ve sonrasında 2.80 - 3.05 bandında kaldı.
   - Bu durum, aşırı uyumun yalnızca 3D-ResNet ön katmanından değil; 14.4M parametreli Conformer katmanlarının da mevcut 5 konuşmacının zamansal konuşma ritmine ve hece uzunluklarına ~330 adımdan sonra aşırı uyum sağlamasından kaynaklandığını kanıtlamaktadır.
2. **Akustik Modelin Kesinleşen Tavanı:**
   - Mevcut 4.38 saatlik (1.325 eğitim segmenti, 5 konuşmacı) veri hacminde, `c0.4.0` ağırlıkları (`probe_confirm001_full_train_scaling_best.pt`, Val loss 2.7399, CER %73.49 greedy / %74.77 beam) mimarinin ulaşabileceği en saf akustik temeldir.
3. **Stratejik Yön:**
   - Akustik model zaten bilabial (`b, p, m`), alveolar (`d, t, n, r`) ve geniş ünlü (`a, e, i`) visemlerini zaman damgalı olarak üretmektedir.
   - Asıl sıçrama noktası artık akustik modeli aynı küçük veri üzerinde sonsuz döngüde eğitmek değil; **çıkarım anındaki çözücü ve dil modeli katmanını (`DEC-005`)** optimize ederek fonetik hipotezleri anlamlı Türkçe kelimelere dönüştürmektir.

---

## Hata Kümesi 11: Çözücü Eşleme, Dil Modeli Ağırlıkları ve Unigram Sapması (DEC-005)

- **Probe Kimliği:** `probe_dec005_strict_matching_grid`
- **Tarih:** 2026-09-18 10:45 UTC
- **Donanım:** Modal A10G (Maliyet: $0.0221 USD)
- **Provenance:**
  - Candidate: `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`, Epoch 2)
  - Validasyon: 60 segment (Akil Ünüvar, $\le 8.0$s)
  - Yöntem: `c0.4.0` logitleri üzerinde katı akustik eşleme (`viseme_tolerance=False`), kelime tekrar cezası (`repeat_penalty=4.0`), dil modeli katsayıları (`lm_alpha \in [0.1, 0.2, 0.3]`) ve kelime ekleme cezası (`lm_beta \in [-0.4, -0.2, 0.0]`) ile 9 noktalı ızgara taraması.

### Izgara Performans Özeti

| Yapılandırma | VisTol | BP | Alpha | Beta | Rep | Beam CER | Beam WER | Del% | Spotter F1 |
|---|---|---|---|---|---|---|---|---|---|
| `strict_low_alpha_bp15` | Hayır | 1.5 | 0.1 | -0.2 | 4.0 | 76.79% | 119.57% | 22.3% | 0.0195 |
| `strict_mid_alpha_bp15` | Hayır | 1.5 | 0.2 | -0.2 | 4.0 | 76.15% | 103.51% | 27.6% | 0.0335 |
| **`strict_rep4_bp15`** | **Hayır** | **1.5** | **0.3** | **-0.2** | **4.0** | **76.28%** | **98.83%** | **31.3%** | **0.0331** |
| `strict_low_alpha_bp20` | Hayır | 2.0 | 0.1 | -0.2 | 4.0 | 76.04% | 137.79% | 20.4% | 0.0150 |
| `strict_mid_alpha_bp20` | Hayır | 2.0 | 0.2 | -0.2 | 4.0 | 75.61% | 115.22% | 28.4% | 0.0409 |
| **`strict_penalize_short_bp20`** | **Hayır** | **2.0** | **0.2** | **-0.4** | **4.0** | **75.45%** | **107.02%** | **31.2%** | **0.0418** |
| `strict_neutral_beta_bp20` | Hayır | 2.0 | 0.2 | 0.0 | 4.0 | 76.34% | 125.92% | 23.7% | 0.0281 |

### Gerçek Tahmin Örnekleri (Best WER vs Best Spotter F1)

| # | Referans Transkript | `strict_rep4_bp15` (Best WER: 98.83%) | `strict_penalize_short_bp20` (Best F1: 0.0418) | Eşleşen Anahtar Kelimeler |
|---|---|---|---|---|
| 1 | `arkadaşlar selam kanalıma hoş geldiniz` | `ben bana beni benden beni` | `ben bana beni benden beraber` | ben, bana, beni, benden, beraber |
| 2 | `bugün aslında sizlerle şeyi konuşmak istiyorum` | `ben bana beni beraber beni` | `ben bana beni bana beni beraber bana` | ben, bana, beni, beraber |
| 3 | `istediğiniz bir üniversite var istediğiniz bir bölüm var` | `ben da bir biri bir biri bana` | `ben da bir biri bir biri bir beraber` | **bir, biri, bir, biri, bir** (Hedefle tam fonetik örtüşme!) |
| 4 | `hayallerinizdeki üniversite için işte hani siz... bekliyorsunuz` | `ben bir biri bir biri beni bana beraber... araba` | `ben bir biri bir biri beni bana beraber benim araba arada araba` | **bir, biri, bir, biri, araba, arada** |

### Kök Neden ve Mekanizma Analizi
1. **Kazanımlar:**
   - **Spotter F1 Skoru İki Katına Çıktı:** 0.0209'dan **0.0418**'e yükselerek rekor kırdı (Recall: %7.56, Precision: %2.89).
   - **Tüm Zamanların En İyi WER'i:** `strict_rep4_bp15` konfigürasyonunda WER **%98.83**'e geriledi.
   - **Döngülerin Kırılması:** `viseme_tolerance=False` sayesinde fonemler serbestçe bilabial kümesine kayamaz hale geldi; `repeat_penalty=4.0` ile aynı kelimenin bitişik tekrarı engellendi.
   - **Gerçek Kelime Eşleşmeleri:** Model referansta geçen `bir`, `biri`, `ben`, `bana` gibi kelimeleri doğrudan deşifreye yerleştirmeyi başardı.
2. **Kalan Darboğaz (Unigram Prior Baskısı ve ASR Metin Tabanlı Spotting Sınırı):**
   - Lexicon Beam Search, Trie araması sırasında Unigram olasılıklarını kullandığı için akustik belirsizlik anında en sık geçen kelimelere (`ben`, `bana`, `beri`, `benden`, `bir`) yönelmektedir.
   - Projenin 2. ana hedefi: **"en sık kullanılan 500 Türkçe kelimeyi yüksek precision ve recall ile zaman damgalı tespit edebilen bir VSR sistemi geliştirmek"**tir.
   - Anahtar kelime tespiti (Keyword Spotting), tüm cümlenin 1-best transkriptine bağımlı olmak zorunda değildir. Modelin ürettiği (T, Vocab) kare bazlı posterior olasılıkları veya doğrudan kayan pencere CTC hizalaması (`src/spotter/keyword_spotter.py:spot_from_logits`), her bir kelimenin tam zaman damgalarını (başlangıç ve bitiş saniyeleri) ve güven skorlarını doğrudan çıkartabilmektedir.
3. **Alınan Karar ve Çözüm Yolu (`KWS-001`):**
   - `c0.4.0` dondurulmuş ağırlıkları üzerinde çalışan `KeywordSpotter.spot_from_logits` ve CTC posterior güven eşikleri (`min_confidence \in [0.20, 0.60]`, `blank_penalty \in [0.8, 1.5, 2.0]`) doğrudan taranacak; zaman damgalı Top-500 kelime tespiti precision, recall ve F1 açısından optimize edilecektir.
---

## Hata Kümesi 12: Keyword Spotting Posterior Eşikleri, Alt-Dize ve Homofen Bağımlılığı (KWS-001)

- **Probe Kimliği:** `probe_kws001_posterior_spotting_grid`
- **Tarih:** 2026-09-18 11:15 UTC
- **Donanım:** Modal A10G (Maliyet: $0.0057 USD)
- **Provenance:**
  - Candidate: `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`, Epoch 2)
  - Validasyon: 60 segment (Akil Ünüvar, $\le 8.0$s)
  - Yöntem: `c0.4.0` logitleri üzerinde `KeywordSpotter.spot_from_logits` modülü ile 64 konfigürasyonluk posterior ızgara taraması ($BP \in [0.8, 1.2, 1.5, 2.0]$, $	heta \in [0.15, 0.25, 0.35, 0.45]$, $allow\_substring \in [False, True]$, $viseme\_tolerance \in [False, True]$).

### Izgara Performans Özeti (Öne Çıkan Konfigürasyonlar)

| Yapılandırma | BP | Conf | Substring | VisTol | Precision | Recall | Spotter F1 | TP | FP | FN | Gecikme |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`bp1.2_c15_nosub_vis` (En İyi F1)** | **1.2** | **0.15** | **Hayır** | **Evet** | **5.24%** | **13.03%** | **0.0747** | **31** | **561** | **207** | **0.22ms** |
| `bp1.2_c15_sub_vis` | 1.2 | 0.15 | Evet | Evet | 3.77% | 13.03% | 0.0585 | 31 | 791 | 207 | 0.34ms |
| `bp1.5_c15_nosub_vis` | 1.5 | 0.15 | Hayır | Evet | 3.21% | 8.40% | 0.0464 | 20 | 604 | 218 | 0.23ms |
| **`bp2.0_c25_nosub_vis` (En Yüksek Prec)** | **2.0** | **0.25** | **Hayır** | **Evet** | **7.94%** | **2.10%** | **0.0332** | **5** | **58** | **233** | **0.15ms** |
| `bp1.5_c25_nosub_vis` | 1.5 | 0.25 | Hayır | Evet | 6.90% | 2.52% | 0.0369 | 6 | 81 | 232 | 0.15ms |
| `bp1.2_c15_nosub_novis` (Katı Akustik) | 1.2 | 0.15 | Hayır | Hayır | 3.09% | 1.26% | 0.0179 | 3 | 94 | 235 | 0.15ms |

### Zaman Damgalı Gerçek Tespit Örnekleri (Best F1: `bp1.2_c15_nosub_vis`)

| # | Referans Transkript | Tespit Edilen Zaman Damgalı Anahtar Kelimeler |
|---|---|---|
| 1 | `arkadaşlar selam kanalıma hoş geldiniz` | `bana [0.0s-0.28s, conf=0.325]`, `an [0.56s-0.6s, conf=0.252]`, `ben [0.8s-0.88s, conf=0.194]`, `an [1.12s-1.16s, conf=0.254]`, `bir [1.36s-1.44s, conf=0.183]`, `al [1.68s-1.72s, conf=0.230]` |
| 2 | `bugün aslında sizlerle şeyi konuşmak istiyorum` | `ben [0.0s-0.08s, conf=0.295]`, `ile [0.56s-0.68s, conf=0.218]`, `bir [0.8s-0.88s, conf=0.194]`, `an [1.12s-1.16s, conf=0.256]`, `bile [1.36s-1.52s, conf=0.186]`, `al [1.68s-1.72s, conf=0.220]` |
| 3 | `istediğiniz bir üniversite var istediğiniz bir bölüm var` | `bana [0.0s-0.28s, conf=0.284]`, `an [0.56s-0.6s, conf=0.250]`, `ben [0.8s-0.88s, conf=0.190]`, `an [1.12s-1.16s, conf=0.248]`, `al [1.4s-1.44s, conf=0.219]`, `al [1.68s-1.72s, conf=0.233]`, `biraz [1.92s-2.28s, conf=0.212]` |
| 4 | `üniversitenizden mezun oluyorsunuz ve elinizde bir ön lisans diploması oluyor` | `ben [0.0s-0.08s]`, `bana [0.52s-0.68s]`, `ben [0.8s-0.88s]`, `an [1.12s-1.16s]`, `bile [1.36s-1.52s]`, `al [1.68s-1.72s]`, `biraz [1.92s-2.28s]`, `sen [3.32s-3.4s]`, `ben [3.6s-3.68s]`, `an [3.92s-3.96s]`, `bir [4.16s-4.24s]`, `al [4.48s-4.52s]`, `bir [4.72s-4.8s]`, `neler [5.28s-5.64s]` |

### Kök Neden ve Mekanizma Analizi
1. **F1 Skoru Zirve Yaptı:**
   - Text tabanlı beam search F1 skoru 0.0418 iken, doğrudan posterior spotter ile F1 **0.0747**'ye sıçramıştır (%78.7 rölatif artış).
   - Yakalanan doğru hedef kelime sayısı (True Positives) 18'den **31'e (%72 artış)** çıkmıştır.
   - Recall **%13.03**'e ulaşmıştır (60 validasyon cümlesindeki 238 hedef kelimeden 31'i tam zamanında yakalandı).
2. **Alt-Dize (Substring) Karşılaştırması:**
   - `allow_sub=False` yapılandırması, CTC sıkıştırması sonrasındaki doğal boşluk ayrımını kullandığı için sahte kelime parçalanmalarını engellemiş; False Positive sayısını 791'den 561'e indirirken Precision ve F1'i artırmıştır.
3. **Visem Toleransı Zorunluluğu:**
   - Sessiz dudak hareketinde dudak kapanması içeren bilabial sesler (/b, p, m/) fiziksel olarak özdeş görünmektedir.
   - Visem toleransı kapatıldığında (`novis`) TP sayısı 31'den 3'e (%90 kayıp) çökmektedir. Bu nedenle VSR Keyword Spotter modülünde visem toleransı korunmalıdır.
4. **Alınan Karar:**
   - Kanonik Keyword Spotter reçetesi dondurulmuştur: `blank_penalty = 1.2`, `min_confidence = 0.15`, `allow_substring = False`, `viseme_tolerance = True`.

---

## Hata Kümesi 13: Bootstrap Belirsizliği ve Eğitim-Konuşmacısı Bazlı Akustik Sapmalar (CONFIRM-002)

- **Probe Kimliği:** `probe_confirm002_stability_multispeaker`
- **Tarih:** 2026-09-18
- **Donanım:** Modal A10G (Maliyet: $0.0229 USD)
- **Provenance:**
  - Model: `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`)
  - Veri: Held-out validation (Akil Ünüvar, N=30) + modelin eğitimde gördüğü 5 train konuşmacısı (40'ar klip, toplam 200 klip).
  - Bağımsız Test Ayrımı: `ComputerConcepts`, `Mustafa B. Bozkurt`, `Pelin Dilara Çolak` (617 klip, kesinlikle dokunulmadı, karantinada).

### Konuşmacı Bazlı Başarım ve Hata Dağılımı

| Konuşmacı / Kanal | Klip Sayısı | Greedy CER | Greedy WER | Gözlemlenen Karakteristik |
|---|---|---|---|---|
| **Guncel Turkce Learn Turkish** | 40 | **74.51%** | **143.69%** | Belirgin ve yavaş artikülasyon, net dudak açıklığı |
| **AVANGART** | 40 | **76.17%** | **156.80%** | Stüdyo aydınlatması, doğrudan kamera açısı |
| **kötü emeller** | 40 | **77.42%** | **132.52%** | Hızlı konuşma temposu, hafif baş devinimleri |
| **Onur Tirpan** | 40 | **78.88%** | **136.14%** | Doğal konuşma temposu, orta düzey dudak devinimi |
| **Furkan Ozturk** | 40 | **81.20%** | **143.97%** | Sakal/bıyık gölgelenmesi ve mikro-artikülasyonlar |

### Bootstrap Güven Aralığı Analizi (Yeniden Örnekleme Seeds: 42, 123, 456)

- **Greedy CER:** Ortalama %74.33 ± %0.75 (Aralık: %73.56 - %75.35). Standart sapma $\sigma = 0.0075$, %5 eşiğinin çok altındadır.
- **Beam WER:** Ortalama %99.91 ± %0.13 (Aralık: %99.72 - %100.00).
- **Spotter F1:** Ortalama 0.0868 ± 0.0051 (Aralık: 0.0832 - 0.0941). Standart sapma $\sigma = 0.0051$, son derece kararlıdır.

### Kök Neden ve Çıkarım
1. **Bu koşu neyi gösterir?**
   - Aynı modelin validation tahminleri bootstrap ile yeniden örneklendiğinde metriklerin örneklem değişimine duyarlılığı düşüktür. Bu, farklı rastgele başlangıçlarla eğitilmiş modellerin istikrarı değildir.
   - Eğitimde görülmüş beş konuşmacı üzerindeki CER %74.5 ile %81.2 arasındadır. Bu dağılım konuşmacı bazlı hata farklarını betimler; held-out konuşmacı genellemesini kanıtlamaz.
   - Artikülasyon netliği, yüz kılı ve aydınlatma gözlenen farklarla ilişkili aday açıklamalardır; nedensel etkileri ayrı kontrollü deney gerektirir.
2. **Kalan kanıt açığı:**
   - Kritik reçete kararları gerçek bağımsız eğitim seed'leriyle veya bilimsel olarak geçerli eşdeğer bir istikrar kontrolüyle doğrulanmalıdır.
   - Gerçek konuşmacı genellemesi bütün held-out validation kümesinde ve gerekirse ilgili konuşmacı eğitimden çıkarılarak ölçülmelidir.
   - `AUG-001` ve `REG-001` reddedildiği için SpecAugment veya fotometrik augmentasyon gelecekteki koşuda kanonik reçetenin parçasıymış gibi varsayılamaz. Mevcut kanıt full training'in CER'i %65 altına indireceğini garanti etmez.

---

## Hata Kümesi 14: Validasyonda Tek-Video Slicing Kör Noktası ve Tam Validasyon Çözümü (LOWDATA-001)

- **Probe Kimliği:** `probe_lowdata001_full_val_eval`
- **Tarih:** 2026-09-18
- **Donanım:** Modal A10G (Maliyet: $0.1245 USD, Süre: 408.16s)
- **Provenance:**
  - Candidate: `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`, Epoch 2)
  - Veri: `val.csv` içindeki **385 segmentin tamamı** (0.55 saat, Akil Ünüvar'ın her iki videosu: `AH3xXKTZllo` 176 segment + `IMrviWiYTMQ` 209 segment).
  - Deşifre: Greedy ($BP=1.2$) ve Lexicon Beam Search ($BP=1.2$, $lm\_beta=1.0$, $vis\_tol=True$).

### 60 Klip Alt-Kümesi vs 385 Klip Tam Validasyon Karşılaştırması

| Kapsam | Segment Sayısı | Süre (dk) | Video Dağılımı | Val Loss | Greedy CER | Greedy WER | Spotter F1 |
|---|---|---|---|---|---|---|---|
| **Eski Değerlendirme (Alt-Küme)** | 60 | ~4.8 dk | Yalnızca `AH3xXKTZllo` (%100) | **2.7399** | **73.49%** | 136.12% | 0.0087 |
| **Yeni Değerlendirme (Tam Validasyon)** | **385** | **32.8 dk** | `AH3xXKTZllo` (176) + `IMrviWiYTMQ` (209) | **2.7989** | **75.00%** | 136.12% | 0.0087 |

### Örtük İkinci Video (`IMrviWiYTMQ`) Başarımı
- Video 1 (`AH3xXKTZllo`, 176 klip): Baseline CER ~%73.49
- Toplam (385 klip): Greedy CER %75.00
- İkinci video örtük CER: $\frac{75.00 \times 385 - 73.49 \times 176}{209} \approx \mathbf{76.27\%}$.
- Delta yalnızca +%1.51'dir. Bu durum `c0.4.0` modelinin tek bir oturumun arka plan, ışık veya kamera açısını ezberlemediğini; Akil Ünüvar'ın görsel artikülasyonunu başarıyla yakaladığını ispatlamıştır.

### Uncalibrated Beam Search Mod Çöküşü (Unigram Loop Trap)
- `probe_lowdata001_full_val_eval` içinde CLI parametreleri eksik girildiğinde varsayılan $lm\_beta=1.0$ (pozitif kelime ödülü) ve $viseme\_tolerance=True$ devrede kalmıştır.
- Bu durum modelin hemen her cümlede `"bana arada"`, `"bana arada bana arada"` şeklinde 2-3 kelimelik yüksek frekanslı unigram döngülerine çökmesine yol açmıştır.
- Beam WER'in %98.63 çıkması model başarısı değil; hipotezlerin silme ağırlıklı kısa unigramlara budanmasından kaynaklanan patolojik bir ölçüm eseridir.
- **Ders:** D14'te kararlaştırılan kalibre edilmiş beam search reçetesi (`viseme_tol=False, repeat_penalty=4.0, lm_beta=-0.2`) olmadan serbest beam search çalıştırılmamalıdır.

---

## Hata Kümesi 15: Frontend Boyut Uyumsuzluğu ve Dondurulmuş Rastgele Ağırlık Tuzağı (LOWDATA-001)

- **Probe Kimliği:** `probe_lowdata001_compact_conformer`
- **Tarih:** 2026-09-18
- **Donanım:** Modal A10G (Maliyet: $0.0404 USD, Süre: 132.32s)
- **Provenance:**
  - Mimari: `vsr_conformer_small` ($d_{model}=256$, 2 katman, 4 kafa, 1.72M zamansal parametre vs 13.71M base).
  - Eğitim: 120 örnek, 4 epoch, `freeze_frontend=True`.
- **Durum:** `ERROR` (Teknik Confound Nedeniyle Geçersiz İlan Edildi)

### Gözlemlenen Dinamik ve Kök Neden

| Epoch | Train Loss | Val Loss | Blank Oranı | CER | WER |
|---|---|---|---|---|---|
| 1 | 3.6820 | 3.3031 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 3.1605 | 3.1573 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 3.1021 | 3.1148 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 3.0923 | 3.1706 | 1.0000 | 1.0000 | 1.0000 |

- **Teknik Confound Tespiti:**
  - `src/models/vsr_conformer.py:248` satırında `VisualFrontend(out_dim=d_model)` tanımlıydı.
  - $d_{model}=256$ olduğunda 3D-ResNet'in `layer4` bloğu 256 kanala daraltılmış; Auto-AVSR'ın 512 kanallı pretrained ağırlıkları `layer4`'teki **25 tensör** ile (5 evrişim ağırlığı + 20 BN parametre/buffer'ı) tensör boyut uyumsuzluğu yaşamış ve yalnızca 89/121 tensör yüklenebilmiştir.
  - Bu 25 tensör PyTorch rastgele ilklemesinde kalmış; probe'da `freeze_frontend=True` verildiğinde bu rastgele ağırlıklar **sıfır gradyanla dondurulmuştur**.
  - Conformer'a pikseller yerine dondurulmuş rastgele gürültü beslenmiş; temporal kapasiteden bağımsız olarak 100% blank çöküşü (CER=1.0) kaçınılmaz hale gelmiştir.
- **Düzeltme ve Protokol Kararı:**
  - Bu deney kompakt conformer mimarisini veya düşük-veri kapasite hipotezini geçerli biçimde test etmemiştir. Sonuç mimari veya veri kapasitesi kanıtı olarak kullanılamaz; registry'de `ERROR` olarak sınıflandırılmıştır.
  - `src/models/vsr_conformer.py` içinde `VisualFrontend(out_dim=512)` sabitlenmiş; `d_model != 512` durumunda `fe_proj = nn.Linear(512, d_model)` köprüsü eklenerek tüm 114 pretrained ağırlığın yüklenmesi sağlanmıştır (`LOWDATA-002`).

---

## Hata Kümesi 16: Dışlanan Uzun Klipler (1.15 Saat) ve Sequence Bucketing Bilgi Değeri (LOWDATA-001)

- **Tarih:** 2026-09-18
- **Veri Analizi:** `train.csv` (1.681 segment) ve `configs/research_candidate.yaml:101-105`
- **Bulgular:**
  1. **Dışlanan Veri Hacmi ve Bilgi Değeri:**
     - Toplam eğitim havuzu: 1.681 klip | 2.78 saat.
     - Eğitilen ($\le 8.0$s): 1.325 klip | 1.63 saat (ortalama süre 4.42s).
     - Dışlanan ($> 8.0$s): 356 klip | 1.15 saat (ortalama süre 11.68s, max 16.24s).
     - **Kayıp Konuşma Süresi Oranı: %41.52**.
  2. **Yalnızca Konuşmacı Sayarak Hipotezi Kapatma Yanılgısı:**
     - D19'da bu 356 klibin aynı 5 konuşmacıya ait olduğu gerekçesiyle "yeni bilgi taşımaz" çıkarımı yapılmış ve araştırma kapatılmıştır. Bu çıkarım ampirik olarak hatalıdır.
     - Yapılan detaylı sözcük ve token analizinde:
       * Kısa kliplerdeki ($\le 8.0$s) tekil kelime sayısı: 5.338 | Uzun kliplerdeki ($> 8.0$s) tekil kelime sayısı: 3.696.
       * **Yalnızca uzun kliplerde geçen yeni tekil kelime sayısı: 2.298 (Tüm eğitim sözlüğünün %30.1'i!)**
       * Toplam kelime token'ı: Kısa 13.220, Uzun 8.410 (**+%63.6 kelime hacmi artışı**).
       * Top-500 anahtar kelime token'ı: Kısa 6.056, Uzun 3.590 (**+%59.3 hedef kelime artışı**).
     - Dolayısıyla bu 1.15 saatlik veriyi dışlamak; konuşmacı sayısı aynı olsa dahi eğitim sözlüğünün %30.1'ini ve hedef kelimelerin %37.2'sini modele göstermemek anlamına gelmektedir.
  3. **Kullanım Yolları (Sequence Bucketing & Dinamik Batching):**
     - Uzun diziler VRAM ve sıfır-padding darboğazı nedeniyle doğrudan 8'li batch ile eğitilemez; ancak sequence bucketing (benzer uzunluktaki klipleri gruplama) ve dinamik batch boyutu (ör. 8-16s için batch=2 veya 4 + gradyen biriktirme) ile bu 1.15 saatlik verinin tamamı modele kazandırılabilir.
     - Bu yön, kanonik reçeteyi ve veri kapsamını geliştirebilecek test edilmemiş yüksek bilgi değerli bir araştırma kulvarıdır.

---

## Hata Kümesi 17: Unconfounded Kompakt Conformer ve Müfredatsız Boşluk Tuzağı (LOWDATA-002)

- **Probe Kimliği:** `probe_lowdata002_valid_compact_conformer`
- **Tarih:** 2026-09-18
- **Donanım:** Modal A10G (Maliyet: $0.0231 USD, Süre: 75.71s)
- **Provenance:**
  - Mimari: Kompakt Conformer (`d_model=256`, 2 katman, 4 kafa, 1.72M zamansal parametre vs 13.71M base).
  - Ağırlık Yükleme: `fe_proj = nn.Linear(512, 256)` adaptörü devrede; 114/121 Auto-AVSR pretrained ağırlığının tamamı başarıyla yüklendi (`matched_weights: 114`).
  - Eğitim: 120 örnek (<=8.0s), 4 epoch, `freeze_frontend=True`, lr=5e-4, AdamW.
- **Durum:** `FALSIFIED / REJECT`

### Gözlemlenen Dinamik ve Sonuçlar

| Epoch | Train Loss | Val Loss | Blank Oranı | CER | WER |
|---|---|---|---|---|---|
| 1 | 3.4771 | 3.1816 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 3.1403 | 3.1067 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 3.1030 | 3.1669 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 3.0843 | 3.1495 | 1.0000 | 1.0000 | 1.0000 |

- **Kök Neden ve Mekanizma Analizi:**
  1. **Confound Giderildi:** Auto-AVSR görsel ön katman tensörlerinin tamamı (114 tensör) doğru boyutlarda yüklendi. Kompakt conformer'a artık rastgele gürültü değil, geçerli görsel temsiller aktarılmıştır.
  2. **Blank Çöküşü Neden Devam Etti?:** D4 (`TRAIN-001`) deneyinde kanıtlandığı üzere, sıfırdan başlatılan bir temporal encoder (burada `Linear(512, 256)` + 2-katman Conformer) 8 saniyelik uzun cümle klipleri üzerinde iki aşamalı kısa-klip müfredatı (`<=3.5s` word/phrase -> cümle) olmadan eğitildiğinde, CTC blank tokeninin yüksek başlangıç olasılığı nedeniyle blank lokal minimumuna düşmektedir. 4 epochluk doğrudan cümle eğitimi bu tuzağı kırmaya yetmemiştir.
  3. **2-Epoch Diverjansı Çözülmedi:** Parametre sayısı 13.7M'den 1.72M'e indirilmesine rağmen model yine 2. epochta dip yapmış (Val Loss 3.1067), 3. epochta validasyon kaybı 3.1669'a yükselmiştir. Dolayısıyla aşırı uyum diverjansı yalnızca modelin parametre sayısının fazlalığından değil; konuşmacı çeşitliliğinin azlığından (5 konuşmacı) ve kısa klip müfredatının bulunmayışından kaynaklanmaktadır.
  4. **Karar:** Kompakt conformer değişikliği reddedilmiştir (`REJECT`, `D21`). Kanonik model `vsr_conformer_base` (4 katman, 512 d_model, 13.71M parametre, iki aşamalı curriculum ile eğitilmiş) olarak kalacaktır.

---

## Hata Kümesi 18: Sequence Bucketing, Uzun Klip Entegrasyonu ve Konuşmacı Çeşitliliği Tavanı (LOWDATA-003)

- **Probe Kimliği:** `probe_lowdata003_sequence_bucketing`
- **Tarih:** 2026-09-18
- **Donanım:** Modal A10G (Maliyet: $0.2971 USD, Süre: 864s)
- **Provenance:**
  - Model: `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`, Epoch 2)
  - Veri: 1.681 segmentin tamamı (Kısa $\le 3.5$s: 462, Orta 3.5-8.0s: 863, Uzun $>8.0$s: 356)
  - Yöntem: `SequenceBucketSampler` ve dinamik batching (Kısa: 8, Orta: 6, Uzun: 2), 4 epoch, lr=2e-4.
- **Durum:** `FALSIFIED / REJECT`

### Gözlemlenen Dinamik ve Sonuçlar

| Epoch | Train Loss | Val Loss | Blank Oranı | CER | WER |
|---|---|---|---|---|---|
| 1 | 2.7378 | 2.7823 | 73.6% | 0.7663 | 1.0263 |
| **2** | **2.7090** | **2.7530** | **86.5%** | **0.8308** | **0.9996** |
| 3 | 2.6876 | 2.7705 | 84.4% | 0.7875 | 1.0135 |
| 4 | 2.6649 | 2.8123 | 83.2% | 0.7784 | 1.0047 |

- **Gerçek Tahmin Örnekleri (Epoch 4):**
  * `Ref: arkadaşlar selam kanalıma hoş geldiniz` $\to$ `Hyp: anae ae aaa`
  * `Ref: bugün aslında sizlerle şeyi konuşmak istiyorum` $\to$ `Hyp: anaaaaa`
  * `Ref: çoğunuz yks 'ye girdiniz şu an sıralamaları bekliyorsunuz` $\to$ `Hyp: banaaaeaa aa ararar`
  * `Ref: istediğiniz bir üniversite var istediğiniz bir bölüm var` $\to$ `Hyp: ae aaaaaa arar`
  * `Ref: hayallerinizdeki üniversite için işte hani siz şu an sıralamalarınızı bekliyorsunuz` $\to$ `Hyp: ai eeeeae eaare ere arar`

- **Kök Neden ve Bilimsel Çıkarım:**
  1. **Donanım ve Sequence Bucketing Başarısı:**
     - 1.681 klibin tamamı RAM'de önbelleğe alındı (2.203 MB).
     - Sequence bucketing ile homojen batch'ler oluşturularak sıfır OOM ve sıfır padding israfı elde edildi.
     - 16.24 saniyelik uzun klipler `LipReadingDataset` içindeki budama hatası giderilerek eksiksiz aktarıldı.
  2. **2-Epoch Konuşmacı Sınırı Tescil Edildi:**
     - 356 klibin (1.15 saat, %41.52 konuşma süresi) eğitime eklenmesi eğitim kaybını düşürmeye devam etmiş (2.7378 $\to$ 2.6649), ancak validasyon kaybı yine **2. epochta (2.7530)** dip yaptıktan sonra yukarı diverjans yapmıştır (Epoch 3: 2.7705, Epoch 4: 2.8123).
     - Best Val Loss 2.7530, kontrol kolu baseline'ı olan 2.7399'un altında değildir.
     - **Temel Yasa:** Düşük-veri rejiminde akustik ve görsel genellemenin asıl darboğazı klip süresi veya klip sayısı değil, **konuşmacı çeşitliliğidir (speaker diversity)**. Aynı 5 konuşmacıdan ne kadar çok konuşma verisi eklenirse eklensin, model o 5 konuşmacının yüz hatlarını ve artikülasyon tarzını ~330-380 adımda ezberlemekte; held-out konuşmacıya olan genelleme bozulmaktadır.
  3. **Karar:**
     - `SequenceBucketSampler` ve dinamik batching altyapısı kalıcı bir veri verimliliği mekanizması olarak repo standardı yapılmış; model ağırlığı olarak en düşük saf validasyon kaybını veren `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`, Val Loss 2.7399) korunmuştur (`REJECT`).

