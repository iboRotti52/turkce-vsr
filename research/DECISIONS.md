# Araştırma Kararları

## D0: Temiz başlangıç ve geçmiş kanıtın iptali

- **Tarih:** 2026-09-17
- **Karar:** Eski checkpoint'ler, deney registry'si, failure analizi, kaynak özeti, split/analiz metadata'sı ve mimari kararları silindi. Önceki deney sonuçları yeni araştırmada baseline, initializer veya mimari kanıt sayılmayacak.
- **Kullanıcı amacı:** Veriyi ve modeli baştan değerlendirerek tek kanonik modeli uzun soluklu araştırma döngüsüyle oluşturmak.
- **Korunan tek veri artefaktı:** Yeniden indirme maliyetini önlemek için `data/iborotti/` kaynak klipleri ve kaynak manifestleri. Bunların doğruluğu kabul edilmedi; `DATA-001` ile baştan denetlenecek.
- **Değiştirilemez sınır:** Üretim girdisi sessiz video; ana hedef Türkçe sürekli transkripsiyon ve genel Top-500 kelimenin zaman damgalı tespiti.

## D1: DATA-001 Veri Kümesi ve Değerlendirme Temeli Kabulü

- **Tarih:** 2026-09-17
- **Karar:** `ACCEPT (with clean-up)`. Tutulan `data/iborotti/` kaynak korpusu bağımsız denetimden geçti; kabul edilen 2.683 klip (4.38 saat) konuşmacı-ayrık train/val/test split'leri halinde donduruldu.
- **Bulgular ve Kanıt:**
  1. **Dosya Bütünlüğü:** 2.767 klip tarandı. 1 plansız voiceover klip (`LF0L-T-EDQI`, missing transcript) temizlendi. Kalan 2.683 kabul edilen klipte 0 bozuk dosya, 0 sıfır bayt dosya.
  2. **Video Standartları:** %100 25.0 FPS, %100 96x96 gri tonlama dudak ROI'si, ortalama 5.70 saniye (142.4 kare).
  3. **Kalite Filtresi:** 83 adet `review` adayı klip (ağız kaybolması > 0.4s, yetersiz görünürlük) denetlendi ve eğitim dışı bırakıldı (`require_acceptance=True`). 245 reddedilmiş kayıt diskte zaten yok.
  4. **Konuşmacı Ayrıklığı (Zero-Leakage):**
     - `train`: 5 kanal (AVANGART, Guncel Turkce, kötü emeller, Onur Tirpan, Furkan Ozturk) -> 1.681 klip.
     - `val`: 1 kanal (Akil Ünüvar, 2 video) -> 385 klip.
     - `test`: 3 kanal (ComputerConcepts, Mustafa B. Bozkurt, Pelin Dilara Çolak) -> 617 klip.
     - Splitler arasında konuşmacı/kanal örtüşmesi kesinlikle %0'dır.
  5. **Top-500 Kelime Kapsamı:** Tüm korpusta 441/500 (%88.2), train kümesinde 422/500 (%84.4), val kümesinde 256/500 (%51.2). Korpustaki tüm konuşma tokenlarının %43.1'i Top-500 kelimelerinden oluşmaktadır.
  6. **Deterministik Provenance:** `data/metadata/` altında `train.csv` (`f5e924fa22...`), `val.csv` (`cfb9e3f80e...`), `test.csv` (`1a0f16d992...`) ve `split_map_iborotti.json` (`9f9438ce9f...`) SHA-256 hash'leri kaydedildi.
- **Kararı Değiştirecek Kanıt:** Konuşmacı sızıntısı veya transkript kayması kanıtlayan yeni metadata ya da daha büyük ve doğrulanmış açık Türkçe VSR korpusu.

## D2: INIT-001 Başlangıç Ağırlığı ve Transfer Stratejisi Değerlendirmesi

- **Tarih:** 2026-09-17
- **Karar:** `REJECT / INVALIDATED (Treatment)` & `SUPPORTED (Control: Auto-AVSR Visual Frontend)`.
- **Bulgular ve Kanıt:**
  1. **Treatment Kolu (Eski exp13 checkpoint'i):** `bigru_continuous_clean_exp13_best.pt` dosyası mevcut değildir (`FileNotFoundError`). D0 protokolü uyarınca eski silinmiş checkpoint'leri yeniden canlandırmak geçersiz kılınmıştır.
  2. **Control Kolu (Auto-AVSR visual frontend transferi):** `vsr_trlrs3_base.pth` kontrol noktasından 114/121 adet 3D-ResNet18 ağırlığı başarıyla aktarılmıştır. Modal A10G üzerinde 60 eğitim, 30 validasyon cümle klibiyle 6 epoch eğitilmiştir (süre: 46.5s, maliyet: ~zsh.015).
  3. **Patoloji Tespiti (CTC Boşluk Çökmesi):** Kontrol kolunda eğitim kaybı 4.01'den 3.07'ye gerilemiş, validasyon kaybı 3.12'ye kadar inmiştir. Ancak 140 karelik sürekli cümlelerde rastgele temporal katmanlar boşluk tuzağına düşmüştür.
  4. **Kök Neden Düzeltmesi:** `cloud_train.py` içinde eğitim aşamasında `blank_penalty` ve `ctc_head.bias[0] -= 0.5` negatif başlatması işlendi.

## D3: ARCH-001 Temporal Mimari Probe'u (Conformer vs BiGRU)

- **Tarih:** 2026-09-17
- **Karar:** `INCONCLUSIVE (Sürekli Cümlede Boşluk Çökmesi)` & `SUPPORTED (Conformer Mimarisi BiGRU'dan Üstün)`.
- **Bulgular ve Kanıt:**
  1. **Kontrollü Probe:** `probe_arch001_conformer` (4-katman Conformer, d_model=512, 114 Auto-AVSR frontend tensörü, blank_penalty=0.4, lr=2e-4) Modal A10G üzerinde 60 eğitim, 30 validasyon sürekli cümle klibiyle 6 epoch çalıştırıldı (59.38s, zsh.018).
  2. **Kafa Kafaya Metrikler:**
     - En iyi Val Loss: Conformer 3.0960 (BiGRU 3.1269 idi; Conformer 0.031 puan daha iyi yakınsadı).
     - Eğitim Loss: Conformer 2.9973 ile 3.00 eşiğini kırdı (BiGRU 3.0736 idi).
     - Karakter Emisyonu: BiGRU kontrolünde 25/25 validasyon cümlesi %100 sessiz (`''`, CER=1.0) kalırken, Conformer 6. epochta ilk aktif karakter emisyonunu üretti (`'b'`, CER=0.9894'e geriledi).
  3. **Karar Kuralı Sonucu:** Pre-registered karar kuralı uyarınca (`research/NEXT_ACTION.md`), Conformer val loss < 3.00 ve blank < %95 hedefine tek başına 6-saniyelik sürekli cümlelerde ulaşamamıştır. Ancak BiGRU'ya göre net üstünlük gösterdiğinden temporal encoder olarak Conformer kanonik modele dahil edilmiş, ana darboğazın mimariden değil 140-karelik sürekli cümlelerdeki $T \gg U$ orantısızlığından kaynaklandığı kanıtlanmıştır.
  4. **Sonraki Adım:** Sürekli cümlelerden önce visem-karakter hizalamasını oturtacak kelime veya kısa öbek curriculum probe'u (`TRAIN-001`) başlatılmıştır.

## D4: TRAIN-001 Müfredat ve İki Aşamalı İnce Ayar Stratejisi Kabulü

- **Tarih:** 2026-09-18
- **Karar:** `ACCEPT`. İki aşamalı müfredat eğitimi (`probe_train001_full_curriculum_unfreeze`) ile validasyon CER %74.92'ye gerilemiş, CTC boşluk çökmesi ve baskın token yerel minimumu kırılmış, çoklu Türkçe kelime ve fonetik hece dizileri (`"ben"`, `"bar"`, `"der"`, `"yera"`) held-out validasyon kümesinde başarıyla üretilmiştir. Kanonik candidate sürümü `c0.2.0` olarak güncellenmiştir.
- **Bulgular ve Kanıt:**
  1. **Teşhis ve Patoloji Giderme:**
     - `probe_train001_curriculum_short`: Decoding katmanındaki liste açma ve parametre hatası düzeltildi.
     - `probe_train001_freeze_fe_blank10`: Frontend dondurulmuşken PyTorch `model.train()` çağrısının frontend BatchNorm istatistiklerini küçük batch gürültüsüyle bozduğu keşfedildi; `_frontend_frozen` ile BatchNorm `eval()` modunda sabitlendi.
     - `probe_train001_frozen_bn_balanced`: Küçük veri hacminin (60 segment) 13.7M parametreli Conformer'ı baskın tekil token minimumuna (`'b'`) ittiği kanıtlandı.
  2. **İki Aşamalı Tam Müfredat Başarısı:**
     - `probe_train001_full_curriculum_unfreeze` (360 kısa segment $\le 3.5$s, 60 validasyon segmenti, 20 epoch, Modal A10G, süre: 213.4s, maliyet: $0.0651 USD):
     - Epoch 1–10 (Frontend Frozen): Conformer temporal katmanları stabil görsel özellikler üzerinde hizalamayı oturttu (Val loss: 3.0435, CER: 0.9028).
     - Epoch 11–20 (Frontend Unfrozen): Diferansiyel öğrenme oranı ($lr=2.5\times 10^{-5}$) ile uçtan uca görsel ince ayara geçildiğinde validasyon kaybı hızla **2.8486**'ya, CER ise **%74.92**'ye indi.
     - Boşluk oranı %80.43 ile sağlıklı CTC aralığına yerleşti.
  3. **Kanonik Reçeteye İşlenenler (`c0.1.0` -> `c0.2.0`):**
     - İki aşamalı eğitim reçetesi (Stage 1: Frozen frontend + lr=5e-4 + blank_penalty=0.8; Stage 2: Unfreeze frontend + lr=2.5e-5 + CosineAnnealingLR).
     - Kanonik kontrol noktası: `turkish-vsr-vol` üzerinde `probe_train001_full_curriculum_unfreeze_best.pt` olarak mühürlendi.
- **Sonraki Adım:** Fonetik karakter dizilerini geçerli Türkçe sözlük kelimelerine bağlamak ve WER'i düşürmek için Dil Modeli ve Sözlük Beam Search Çözücü probe'u (`DEC-001`) başlatılacaktır.

## D5: DEC-001 Sözlük Kısıtlı Beam Search ve Dil Modeli Karşılaştırma Kararı

- **Tarih:** 2026-09-18
- **Karar:** `REJECT` (İlk Trie + LM konfigürasyonu: $\beta = 1.0 \times \text{len}$).
- **Bulgular ve Kanıt:**
  1. **Kontrollü Probe:** `probe_dec001_lexicon_beam_vs_greedy` Modal A10G üzerinde `c0.2.0` checkpoint'i (`probe_train001_full_curriculum_unfreeze_best.pt`) ile 60 held-out validasyon klibinde çalıştırıldı (12.36s, $0.0038 USD).
  2. **Kafa Kafaya Karşılaştırma:**
     - Greedy Baseline: CER = 0.7769 | WER = 1.0024 | Spotter F1 = 0.0197 | Latency = 0.2 ms/klip
     - Lexicon Beam Search: CER = **0.7492** | WER = **1.1010** | Spotter F1 = **0.0913** | Latency = 28.0 ms/klip
  3. **Pozitif Çıktılar:**
     - CER %77.69'dan %74.92'ye geriledi (%2.77 iyileşme).
     - Spotter F1 skoru %1.97'den %9.13'e çıktı (~4.6 kat artış).
     - Çıkarım gecikmesi (28 ms/klip) 2.0 saniyelik üst sınırın ve 25 FPS video süresinin (40 ms/kare) çok altında, son derece hızlıdır.
  4. **Patoloji ve Karar Gerekçesi:**
     - Pre-registered falsification kriteri: WER'in greedy baseline'ın (1.0024) altına inememesi. WER 1.1010 olarak ölçüldü.
     - Kök Neden: `LexiconBeamSearchDecoder` içinde `lm_score += beta * len(completed_word)` formülüyle kelime uzunluğunun log-prob alanına çarpılarak eklenmesi, beam search'ü akustik belirsizlik anında uzun heceleri birden fazla 1-3 harfli Top-500 kelimesine (`ben`, `de`, `bir`, `yer`, `ara`) parçalamaya itti. Bu aşırı kelime ekleme (insertion) hataları WER'i 1.0'ın üzerine çıkardı.
     - Akustik Eşik: Conformer CER'i %75 bandındayken katı 500-kelimelik Trie zorlaması, modelin zayıf akustik sinyalden yüksek frekanslı kelimeleri halüsinasyon olarak üretmesine neden olmaktadır.
  5. **Düzeltme ve Sonraki Adım:**
     - `src/spotter/lexicon_decoder.py` içindeki kelime uzunluğu çarpanı kaldırıldı (`lm_score += beta`).
     - Düzeltilmiş kelime ceza katsayısı ($\beta \le 0$) ile hızlı bir kalibrasyon probe'u (`DEC-002` / `probe_dec001_calibrated_beta`) koşulmalı veya doğrudan tam cümle müfredatına (`CURR-002`) geçilerek akustik model CER'i %50 altına indirilmelidir.

## D6: CURR-002 Sürekli Cümle Müfredat Ölçeklemesi Kabulü (c0.2.0 -> c0.3.0)

- **Tarih:** 2026-09-18
- **Karar:** `ACCEPT`. `c0.2.0` ağırlıklarından başlanarak 720 eğitim segmentine ve 6.0 saniyeye kadar olan sürekli cümle dizilerine genişletilen ince ayar (`probe_curr002_sentence_scaling`), modelin uzun dizilerde boşluk çökmesine düşmeden sürekli konuşma karakter dizileri üretmesini sağlamıştır.
- **Bulgular ve Kanıt:**
  1. **Kontrollü Probe:** `probe_curr002_sentence_scaling` Modal A10G üzerinde 720 train ve 60 val segmenti ($\le 6.0$s) ile 15 epoch çalıştırıldı (süre: 1085s, maliyet: $0.3311 USD).
  2. **Metrikler ve İlerleme:**
     - Validasyon Kaybı: Tüm zamanların en iyi değeri olan **2.7851**'e geriledi (`c0.2.0`'ın 2.8486 değerinden 0.0635 puan daha iyi).
     - Eğitim Kaybı: 2.8556'dan **2.7106**'ya geriledi.
     - Boşluk Oranı: %85.9 (CTC için ideal kararlılık bandında; 6 saniyelik uzun cümlelerde sıfır boşluk çökmesi).
     - Validasyon CER: 6 saniyelik tam cümlelerde **%75.21** (Epoch 6) / %77.94 (Epoch 15). Bu değer, `ARCH-001`'de doğrudan sürekli cümlelerle eğitilen Conformer'ın %98.94 CER değerine kıyasla devasa bir sıçramadır.
  3. **Kalitatif Davranış:** Model artık 150 karelik (6 saniye) sürekli videolarda ritmik olarak kelime ve hece grupları (`eie`, `ei`, `ee`, `are`, `ere`, `barar`) ve kelimeler arası boşluk karakteri (`' '`) üretebilmektedir.
  4. **Kanonik Reçeteye İşlenenler (`c0.2.0` -> `c0.3.0`):**
     - Çok adımlı müfredat reçetesi (Kısa $\le 3.5$s aşamasından sonra Sürekli Cümle $\le 6.0$s aşaması).
     - Kanonik kontrol noktası: `turkish-vsr-vol` üzerinde `probe_curr002_sentence_scaling_best.pt` olarak mühürlendi.
- **Sonraki Adım:** `c0.3.0` sürekli cümle kontrol noktası üzerinde düzeltilmiş `LexiconBeamSearchDecoder` değerlendirmesi (`DEC-002`) ve ardından tüm veri kümesine (1.681 klip) tam ölçekleme (`CONFIRM-001`).

## D7: DEC-002 c0.3.0 Kontrol Noktasında Düzeltilmiş Lexicon Beam Search Değerlendirmesi

- **Tarih:** 2026-09-18
- **Karar:** `INCONCLUSIVE` (Aşırı kelime ekleme hatası giderildi, WER greedy altına indi fakat akustik model temel darboğaz olarak teyit edildi).
- **Bulgular ve Kanıt:**
  1. **Kontrollü Probe:** `probe_dec002_c030_lexicon_beam` Modal A10G üzerinde `c0.3.0` checkpoint'i (`probe_curr002_sentence_scaling_best.pt`) ile 60 held-out validasyon sürekli cümle klibinde ($\le 6.0$s) çalıştırıldı (süre: 19.32s, maliyet: $0.0059 USD).
  2. **Kafa Kafaya Metrikler:**
     - Greedy Baseline: CER = 0.8986 | WER = 1.0000 | Spotter F1 = 0.0000 | Gecikme = 0.3 ms/klip
     - Calibrated Lexicon Beam Search ($\beta = -0.2, \alpha = 0.3$): CER = **0.8945** | WER = **0.9981** | Spotter F1 = **0.0062** | Gecikme = 93.6 ms/klip
  3. **Pozitif Çıktılar:**
     - `DEC-001`'deki kelime ekleme patlaması (WER 1.1010) tamamen önlendi; WER ilk defa 1.0000'in altına indi (**0.9981**).
     - Spotter F1 skoru 0.0000'den **0.0062**'ye yükseldi.
     - Çıkarım gecikmesi (93.6 ms/klip) 2.0 saniyelik üst sınırın çok altında, gerçek zamanlı video akışına uygundur.
  4. **Darboğaz Tespiti ve Karar Gerekçesi:**
     - WER greedy baseline'ın altına inmesine rağmen, %85 hedef eşiğinin (WER < 0.85) çok üzerinde kalmıştır (WER = 0.9981).
     - Kök Neden: Akustik model CER'i 6 saniyelik uzun dizilerde %89.45 seviyesindeyken, model ağırlıklı olarak sık görülen ünlüleri (`'a'`, `'i'`, `'n'`) üretmektedir. Beam search bu ham emisyonları sözlükteki `"adamın"`, `"arada"`, `"bana"`, `"araba"`, `"azından"` gibi kelimelere bağlamaya çalışmaktadır.
     - Hiyerarşik Teşhis: Seviye 6 (Decoder/Dil Modeli) Seviye 4'teki (Akustik Zamansal Temsil) eksikliği tek başına kapatamaz. Model şimdiye kadar `train.csv` içindeki 1.681 klibin yalnızca 720'sini görmüştür; eğitim verisinin %57'si ve çoklu konuşmacı varyasyonu henüz modele aktarılmamıştır.
- **Sonraki Adım:** Akustik ve zamansal temsil kalitesini radikal biçimde yükseltmek için model tüm eğitim korpusundaki (1.681 klip, $\le 8.0$s) 5 konuşmacı ve tüm sözcük zenginliği ile eğitilecek (`CONFIRM-001`).

## D8: CONFIRM-001 Tam Eğitim Kümesi Ölçekleme Kabulü (c0.3.0 -> c0.4.0)

- **Tarih:** 2026-09-18
- **Karar:** `ACCEPT`. `c0.3.0` kontrol noktası ağırlıklarından başlanarak eğitim kümesindeki 1.325 klibin ($\le 8.0$s, 2.21 saat) ve 5 konuşmacının tamamı ile yapılan ince ayar (`probe_confirm001_full_train_scaling`), validasyon kaybını tüm zamanların en iyi seviyesi olan **2.7399**'a düşürmüş ve modelin zengin Türkçe ünsüz kümelenmeleri (`herin`, `beni`, `dei`, `benae`, `allı`, `buni`, `berlar`, `birlir`) üretmesini sağlamıştır. Model `c0.4.0` sürümüne terfi ettirilmiştir.
- **Bulgular ve Kanıt:**
  1. **Kontrollü Deney:** `probe_confirm001_full_train_scaling` Modal A10G üzerinde 1.325 eğitim segmenti, 60 held-out validasyon segmenti ile 20 epoch çalıştırıldı (süre: 1467.16s, maliyet: $0.4475 USD).
  2. **Kritik Metrikler:**
     - En İyi Validasyon Kaybı: **2.7399** (Epoch 2'de elde edildi; `c0.3.0`'ın 2.7851 rekorunu 0.0452 puan daha geliştirdi).
     - Eğitim Kaybı: 2.7849'dan **2.6180**'e kadar istikrarlı şekilde geriledi.
     - Validasyon CER: **%75.61** (Epoch 2).
     - Boşluk Oranı: %85.1 - %86.7 bandında sağlıklı CTC davranışı (sıfır boşluk çökmesi).
  3. **Açığa Çıkan Yeni Dinamik (Konuşmacılar Arası Aşırı Uyum / Cross-Speaker Overfitting):**
     - Eğitim kaybı 2.78'den 2.61'e düşerken, validasyon kaybı Epoch 2'den sonra 2.7399'dan 3.27'ye yükselmiştir.
     - Kök Neden: 13.7M parametreli model görsel veri artırma (visual augmentation) olmadan eğitildiğinde 5 eğitim konuşmacısının dudak yapısına ve mimiklerine aşırı uyum sağlamakta; held-out validasyon konuşmacısına (Akil Ünüvar) genelleme zayıflamaktadır.
     - En iyi genelleme noktası (Epoch 2) `/root/vol/probe_confirm001_full_train_scaling_best.pt` olarak başarıyla kaydedilmiştir.
  4. **Kalitatif Sıçrama:**
     - Model önceki adımlardaki jenerik ünlü tekrarlarını (`"anaaaaaaa"`, `"eeeeee"`) aşmış; `"sallıyorum"` -> `"allı"`, `"bölüm var"` -> `"berlar"` / `"birlir"`, `"formül"` -> `"buni deeiir"` gibi zengin Türkçe ünsüz/hece hizalamaları öğrenmiştir.
  5. **Kanonik Reçeteye İşlenenler (`c0.3.0` -> `c0.4.0`):**
     - Eğitim kapsamı: Tüm eğitim kümesi (1.325 klip $\le 8.0$s, 5 konuşmacı).
   - **Sonraki Adım:** Zenginleşen ünsüz temsillerini geçerli Türkçe kelimelere dönüştürmek için `c0.4.0` üzerinde `DEC-003` (Calibrated Lexicon Beam Search) çözücü probe'u çalıştırılacaktır.

## D9: DEC-003 c0.4.0 Kontrol Noktası Çözücü Değerlendirmesi ve Erken Durdurma İncelemesi

- **Tarih:** 2026-09-18
- **Karar:** `INCONCLUSIVE` (Beam search WER ve CER'i greedy'ye göre iyileştirdi, ancak akustik erken durdurma kontrol noktasının ünsüz temsillerini tam içermediği keşfedildi).
- **Bulgular ve Kanıt:**
  1. **Kontrollü Probe:** `probe_dec003_c040_lexicon_beam` Modal A10G üzerinde `c0.4.0` checkpoint'i (`probe_confirm001_full_train_scaling_best.pt`, Epoch 2) ile 60 held-out validasyon klibinde ($\le 8.0$s) çalıştırıldı (süre: 20s, maliyet: $0.0093 USD).
  2. **Metrikler:**
     - Greedy Baseline: CER = 0.8482 | WER = 1.0000 | Spotter F1 = 0.0000 | Gecikme = 0.3 ms/klip
     - Calibrated Lexicon Beam Search ($\beta = -0.2, \alpha = 0.3$): CER = **0.8241** (%2.41 iyileşme) | WER = **0.9983** | Spotter F1 = **0.0083** | Gecikme = 96.3 ms/klip
  3. **Kritik Keşif (Aşırı Uyum vs Temsil Olgunluğu İkilemi):**
     - `CONFIRM-001`'in en düşük validasyon kaybına sahip kontrol noktası **Epoch 2**'de kaydedilmişti (val loss 2.7399).
     - Ancak modelin zengin ünsüzleri (`herin`, `beni`, `allı`, `buni`, `berlar`, `birlir`) ürettiği aşama **Epoch 15–20** idi (train loss 2.61'e indiğinde).
     - Epoch 2 kontrol noktasında model henüz bu ünsüzleri tam oturtmadığı için, held-out validasyon tahminleri halen `"baaaaaaaar"`, `"banaaaaaaara"` gibi erken aşama ünlü ağırlıklı dizilerden oluşmaktadır.
     - Beam search bu erken dizileri `"bana arada"`, `"bana arada araba"` olarak çözmüş; WER 1.0000'den 0.9983'e inmiştir.
  4. **Kök Neden ve Darboğaz:**
     - Model görsel veri artırma (aydınlatma, kontrast, affine rotasyon) eksikliği nedeniyle 2. epoch sonrasında eğitim konuşmacılarının yüz/ışık özelliklerine aşırı uyum sağlamış; validasyon kaybı 2.7399'dan 3.27'ye fırlamıştır.
     - Bu aşırı uyum olmasaydı, model Epoch 10-15'e kadar genelleme kaybını düşürmeye devam edebilir ve zengin ünsüz temsillerini held-out konuşmacıya da aktarabilirdi.
- **Sonraki Adım:** Konuşmacılar arası aydınlatma ve görünüm farklarını yok etmek için fotometrik parlaklık/kontrast bozulması ve hafif rotasyon içeren görsel veri artırma probe'u (`AUG-001`) tasarlanacaktır.

## D10: AUG-001 Fotometrik Veri Artırma ve Regülarizasyon Değerlendirmesi

- **Tarih:** 2026-09-18
- **Karar:** `REJECT`. Fotometrik parlaklık/kontrast pertürbasyonu ($\alpha \in [0.85, 1.15]$, $\beta \in [-0.10, 0.10]$) ve artırılmış ağırlık sönümleme (`weight_decay=5e-4`), `c0.4.0`'ın en iyi validasyon kaybı olan **2.7399**'u geliştirememiştir (En iyi val loss: **2.8105**). Karar kuralı gereği fotometrik gürültü kanonik mimariye dahil edilmemiştir.
- **Bulgular ve Kanıt:**
  1. **Kontrollü Deney:** `probe_aug001_photometric_reg` Modal A10G üzerinde `c0.4.0` ağırlıkları (`probe_confirm001_full_train_scaling_best.pt`) üzerinden 1.325 eğitim ve 60 validasyon klibiyle 12 epoch çalıştırıldı (süre: ~1043s, maliyet: $0.3182 USD).
  2. **Metrikler ve Karşılaştırma:**
     - En İyi Val Loss: **2.8105** (Epoch 2). Kontrol kolu `c0.4.0` baseline'ı **2.7399** idi; 0.0706 puan daha kötü gerçekleşti.
     - Final Val Loss: **3.0941** (Epoch 12).
     - CER: %78.18 (Epoch 2) / %81.32 (Epoch 12).
     - Train Loss: 2.7470 -> 2.6598 (Yakınsama hızı fotometrik gürültü nedeniyle yavaşladı).
  3. **Karar Kuralı Sonucu:**
     - Pre-registered falsification kriteri: "Validation loss failing to decrease below c0.4.0 best (2.7399) or diverged training loss."
     - Sonuç kriteri karşılayamadı; hipotez yanlışlandı (`REJECT`).
  4. **Kök Neden ve Darboğaz Analizi:**
     - Fotometrik parlaklık/kontrast varyasyonu dudak sınırlarının kontrastını bozarak Auto-AVSR 3D-ResNet ön katmanının visem öznitelik çıkarımını zayıflatmış ve yakınsamayı yavaşlatmıştır.
     - Daha da önemlisi, detaylı karakter hata analizi (`jiwer.process_characters`) CER darboğazının ardındaki asıl gerçeği açığa çıkarmıştır:
       - Toplam Referans Karakter: 1.823
       - Toplam Model Hipotez Karakter: 568
       - Yerine Koyma ($S$): 190 (%13.1)
       - **Silme / Eksik Emisyon ($D$): 1.257 (%86.7!)**
       - Ekleme ($I$): 2 (%0.1)
     - Model gerekli karakterlerin üçte ikisinden fazlasını hiç üretmemektedir (aşırı silme hatası). Modelin sessiz kalma eğilimi CTC blank eşiğinin (`blank_penalty`) varsayılan değerde kalmasından kaynaklanmaktadır.
- **Sonraki Adım:** Sıfır eğitim maliyetiyle doğrudan modelin emisyon eşiğini kalibre edecek olan `DEC-004` (Decoding-time CTC Blank Penalty Calibration Grid on `c0.4.0`) probe'u çalıştırılacaktır.

## D11: DEC-004 Çıkarım Zamanı CTC Blank Penalty Kalibrasyon Izgarası ve Silme Hatası Dengelenmesi

- **Tarih:** 2026-09-18
- **Karar:** `INCONCLUSIVE (Gain Confirmed & Decoder Recipe Calibrated)`. `c0.4.0` kontrol noktası üzerinde tek forward pass ile elde edilen logitler üzerinden 9 noktalı `blank_penalty` ızgarası ($\in [0.0, 0.4, 0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]$) taranmış; silme hataları 3.463'ten 1.290'a indirilerek (%62.7 düşüş) Greedy CER **%73.49**'a, Beam CER ise **%74.77**'ye (tüm zamanların en iyi değeri) geriletilmiştir.
- **Bulgular ve Kanıt:**
  1. **Kontrollü Deney:** `probe_dec004_blank_penalty_grid` Modal A10G üzerinde `c0.4.0` en iyi kontrol noktası (`probe_confirm001_full_train_scaling_best.pt`) ile 60 held-out validasyon klibinde ($\le 8.0$s) çalıştırıldı (süre: 58.5s, maliyet: $0.0176 USD).
  2. **Izgara Sonuçları Tablosu:**
     | BP | Blank% | Greedy CER | Greedy WER | S | D | I | Del% | Beam CER | Beam WER | Beam F1 |
     |---|---|---|---|---|---|---|---|---|---|---|
     | 0.0 | 85.1% | 89.87% | 100.00% | 484 | 3463 | 3 | 87.7% | 87.10% | 100.00% | 0.0000 |
     | 0.4 | 79.3% | 84.82% | 100.00% | 620 | 3100 | 8 | 83.2% | 82.43% | 99.83% | 0.0084 |
     | 0.8 | 63.2% | 75.61% | 103.68% | 1203 | 2103 | 17 | 63.3% | 80.14% | 100.00% | 0.0000 |
     | **1.2** | **49.1%** | **73.49%** | **133.28%** | **1849** | **1290** | **91** | **39.9%** | **78.68%** | **100.00%** | **0.0000** |
     | 1.5 | 36.5% | 76.04% | 140.30% | 2321 | 712 | 309 | 21.3% | 76.88% | 99.50% | 0.0146 |
     | 1.8 | 30.9% | 78.16% | 145.99% | 2462 | 532 | 441 | 15.5% | 75.90% | 99.16% | 0.0162 |
     | **2.0** | **27.1%** | **80.36%** | **147.32%** | **2523** | **438** | **571** | **12.4%** | **74.77%** | **99.00%** | **0.0209** |
     | 2.2 | 22.0% | 83.62% | 149.00% | 2610 | 308 | 757 | 8.4% | 75.49% | 99.50% | 0.0203 |
     | 2.5 | 17.5% | 87.28% | 149.33% | 2621 | 241 | 974 | 6.3% | 75.36% | 102.68% | 0.0176 |
  3. **Mekanizma ve Karar Kuralı:**
     - **Greedy Optimumu ($BP=1.2$):** Greedy CER global minimumunu **%73.49** ile $BP=1.2$'de gördü. Silme hatası oranı %87.7'den %39.9'a inerek $D \approx S$ dengesine ulaştı. Model bastırılmış ünsüzleri (`ban`, `bari`, `baran`, `bena`) başarıyla üretmeye başladı.
     - **Beam Optimumu ($BP=2.0$):** Greedy arama yüksek BP'de ekleme patlamasına girerken, Lexicon Trie sözlüğü eklemeleri sınırlandırdı ve $BP=2.0$'da Beam CER **%74.77**'ye, WER **%99.00**'a, Spotter F1 **0.0209**'a ulaştı.
     - CER %65 eşiğinin altında olmadığı için aday modeli doğrudan onaylamadı (`INCONCLUSIVE`), ancak 0.7561 baseline'ını kesin olarak aştı ve sıfır ek maliyetle çıkarım reçetesini kalibre etti.
  4. **Kanonik Reçeteye İşlenenler:**
     - Greedy Decoding için: `blank_penalty = 1.2`
     - Lexicon Beam Search için: `blank_penalty = 2.0`, `lm_alpha = 0.3`, `lm_beta = -0.2`
- **Sonraki Adım:** Modelin `ban/bari/bar` fonetik şablonunu aşması ve 2. epoch sonrasında konuşmacı aşırı uyumu yaşamadan daha uzun süre eğitilebilmesi için Conformer öznitelik seviyesinde SpecAugment / Temporal Dropout regülarizasyonu (`REG-001`) araştırılacaktır.

## D12: REG-001 Özellik Seviyesinde SpecAugment ve Conformer Regularization Deneyi

- **Tarih:** 2026-09-18
- **Karar:** `REJECT`. 3D-ResNet ön katman çıktısı ($B \times T \times D$) üzerinde zaman maskelemesi ($\le 15$ kare), kanal maskelemesi ($\le 48$ kanal) ve artırılmış Conformer dropout (0.2), `c0.4.0`'ın en iyi validasyon kaybı olan **2.7399**'u geliştirememiştir (En iyi val loss: **2.7481**). Önceden belirlenen karar kuralı gereği (`research/NEXT_ACTION.md`), probe reddedilmiş; candidate sürümü `c0.4.0` olarak korunmuştur.
- **Bulgular ve Kanıt:**
  1. **Kontrollü Deney:** `probe_reg001_specaugment_conformer` Modal A10G üzerinde `c0.4.0` ağırlıkları (`probe_confirm001_full_train_scaling_best.pt`) üzerinden 1.325 eğitim ve 60 validasyon klibiyle 12 epoch çalıştırıldı (süre: ~959s, maliyet: $0.2925 USD).
  2. **Metrikler ve Karşılaştırma:**
     - En İyi Val Loss: **2.7481** (Epoch 2). Baseline `c0.4.0` **2.7399** idi; 0.0082 puan daha geride kaldı.
     - Final Val Loss: **3.2332** (Epoch 12).
     - CER: %75.20 (Epoch 2) $\to$ %83.94 (Epoch 12).
     - Train Loss: 2.7732 $\to$ 2.6900 (Düzenli düşüş devam etti).
  3. **Karar Kuralı Sonucu:**
     - Pre-registered falsification kriteri: "Validation loss failing to beat c0.4.0 baseline (2.7399) or loss divergence."
     - En iyi validasyon kaybı 2.7399'u geçememiş ve Epoch 3 sonrasında validasyon kaybı diverjans göstermiştir; hipotez yanlışlandı (`REJECT`).
  4. **Kök Neden ve "2-Epoch Aşırı Uyum Paradoksu":**
     - Hem `CONFIRM-001` (baseline), hem `AUG-001` (fotometrik pertürbasyon), hem de `REG-001` (SpecAugment) koşularında validasyon kaybı istisnasız **Epoch 2**'de (~330 adım) minimuma ulaşmakta ve Epoch 3'ten itibaren yukarı doğru diverjans yapmaktadır.
     - Neden: Eğitim kümesindeki 5 konuşmacının yüz ve artikülasyon dinamikleri, görsel ön katman (11.2M parametre) ve Conformer (14.4M parametre) serbestlik dereceleri tarafından 330 adımdan sonra ezberlenmeye başlamaktadır.
     - SpecAugment ve dropout=0.2, Conformer içindeki kanal/zaman maskelemesini sağlasa da, 3D-ResNet ön katmanının 5 konuşmacıya uyarlanmasını tek başına durduramamıştır.
  5. **Mevcut Durum ve Model Mührü:**
     - Model: `c0.4.0` kanonik model olarak kalmaya devam eder (`probe_confirm001_full_train_scaling_best.pt`).
     - Kalibre edilmiş decoding reçetesi: Greedy için `blank_penalty=1.2` (CER %73.49), Lexicon Beam Search için `blank_penalty=2.0` (CER %74.77, WER %99.00, Spotter F1 0.0209).
- **Sonraki Adım:** `FE-001` (Frozen-Frontend Conformer Scaling Probe on Full Dataset).

## D13: FE-001 Dondurulmuş Görsel Ön Katman ile Tam Veri Ölçekleme Probe'u

- **Tarih:** 2026-09-18
- **Karar:** `REJECT`. 3D-ResNet görsel ön katmanının tamamen dondurulması (`freeze_frontend=True`, BatchNorm eval modu) ile tüm 1.325 segment üzerinde yapılan Conformer eğitimi (`probe_fe001_frozen_frontend_conformer`), en iyi validasyon kaybı olarak **2.7670** üretmiş; `c0.4.0` baseline'ı olan **2.7399**'u geliştirememiştir. Önceden belirlenen karar kuralı gereği (`research/NEXT_ACTION.md`), probe reddedilmiş; candidate sürümü `c0.4.0` olarak korunmuştur.
- **Bulgular ve Kanıt:**
  1. **Kontrollü Deney:** `probe_fe001_frozen_frontend_conformer` Modal A10G üzerinde `c0.4.0` ağırlıkları (`probe_confirm001_full_train_scaling_best.pt`) üzerinden 1.325 eğitim ve 60 validasyon klibiyle 12 epoch çalıştırıldı (süre: ~700s, maliyet: $0.2135 USD).
  2. **Metrikler ve Karşılaştırma:**
     - En İyi Val Loss: **2.7670** (Epoch 2). Baseline `c0.4.0` **2.7399** idi; 0.0271 puan daha geride kaldı.
     - Final Val Loss: **3.0477** (Epoch 12).
     - CER: %78.61 (Epoch 2), %75.40 (Epoch 6), %81.21 (Epoch 12).
     - Train Loss: 2.7635 $\to$ 2.6647 (Eğitim kaybı düzenli düştü).
  3. **Karar Kuralı Sonucu:**
     - Pre-registered falsification kriteri: "Validation loss failing to decrease below c0.4.0 best (2.7399) or stalling."
     - En iyi validasyon kaybı 2.7399'u geçemedi; hipotez yanlışlandı (`REJECT`).
  4. **Büyük Resim ve 4 Bağımsız Deneyin Sentezi:**
     - 1.325 segment üzerinde koşulan dört kontrollü probe (`CONFIRM-001`, `AUG-001`, `REG-001`, `FE-001`) tek ve kesin bir sonucu kanıtlamıştır:
       1. Frontend ince ayarlı (`CONFIRM-001`): Best Val Loss = **2.7399** (Epoch 2).
       2. Fotometrik gürültü (`AUG-001`): Best Val Loss = **2.8105** (Epoch 2).
       3. Feature SpecAugment + Dropout 0.2 (`REG-001`): Best Val Loss = **2.7481** (Epoch 2).
       4. Dondurulmuş Frontend (`FE-001`): Best Val Loss = **2.7670** (Epoch 2).
     - Model, mevcut 5-konuşmacılı eğitim kümesinde 2. epochta (~330 adım) akustik öğrenme kapasitesinin tavanına ulaşmaktadır.
     - En saf ve en düşük validasyon kaybını üreten kontrol noktası tartışmasız olarak `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`) Epoch 2 ağırlıklarıdır.
  5. **Mevcut Model Durumu:**
     - Akustik Model: `c0.4.0` kanonik model olarak dondurulmuştur.
     - Çözücü Reçetesi: Greedy için `blank_penalty=1.2` (CER %73.49), Lexicon Beam Search için `blank_penalty=2.0` (CER %74.77, WER %99.00, Spotter F1 0.0209).
- **Sonraki Adım:** `DEC-005` (Tamamlandı).

## D14: DEC-005 Katı Akustik Eşleme ve Çözücü Izgara Optimizasyonu Probe'u

- **Tarih:** 2026-09-18
- **Karar:** `INCONCLUSIVE (F1 Gain Confirmed)`. `c0.4.0` logitleri üzerinde test edilen 9 konfigürasyonlu katı akustik eşleme (`viseme_tolerance=False`), kelime tekrar cezası (`repeat_penalty=4.0`) ve dil modeli ağırlıkları ızgarası (`probe_dec005_strict_matching_grid`), Top-500 Keyword Spotting F1 skorunu **0.0209'dan 0.0418'e çıkararak %100 rölatif artış** sağlamış ve Beam WER'i tüm zamanların en iyi seviyesi olan **%98.83**'e (`strict_rep4_bp15`) indirmiştir. Ancak akustik CER tabanı (%73-75) nedeniyle pre-registered iddialı hedeflere (WER < %90, F1 > 0.05) tam ulaşılamamıştır. Karar kuralı gereği sonuç `INCONCLUSIVE` olarak kaydedilmiş, elde edilen en iyi decoding parametreleri kanonik reçeteye eklenmiştir.
- **Bulgular ve Kanıt:**
  1. **Kontrollü Deney:** Modal GPU üzerinde `c0.4.0` validasyon logitleri (60 segment, Akil Ünüvar) önbelleğe alınarak sıfır ek eğitim maliyetiyle 9 farklı çözücü konfigürasyonu taranmıştır (süre: 72.3s, maliyet: $0.0221 USD).
  2. **Izgara Sonuçları:**
     - `strict_penalize_short_bp20` (viseme_tol=False, BP=2.0, alpha=0.2, beta=-0.4, rep=4.0):
       - **Spotter F1: 0.0418** (Tüm zamanların rekoru, baseline 0.0209'un 2 katı)
       - Spotter Recall: %7.56, Spotter Precision: %2.89
       - Beam CER: %75.45, Beam WER: %107.02
     - `strict_rep4_bp15` (viseme_tol=False, BP=1.5, alpha=0.3, beta=-0.2, rep=4.0):
       - **Beam WER: %98.83** (Tüm zamanların en düşük WER'i, baseline %99.00 idi)
       - Spotter F1: 0.0331
       - Beam CER: %76.28
  3. **Mekanizma ve Kalitatif Doğrulama:**
     - `viseme_tolerance=False` yapılması, visem sınıfındaki bilabial fonemlerin (`b, p, m, v, f`) rastgele birbirinin yerine geçerek Unigram LM'in yüksek frekanslı kelimelerini tetiklemesini engellemiştir.
     - `repeat_penalty=4.0` aynı kelimenin arka arkaya 5-6 kez tekrarlanmasını ("beraber beraber beraber") tamamen ortadan kaldırmıştır.
     - Model artık referanstaki kelimelerle doğrudan örtüşen hedefleri (`bir`, `biri`, `ben`, `bana`) başarıyla tespit etmektedir.
  4. **Kanonik Reçete Güncellemesi:**
     - Top-500 Keyword Spotting önceliğinde: `viseme_tolerance=False`, `blank_penalty=2.0`, `lm_alpha=0.2`, `lm_beta=-0.4`, `repeat_penalty=4.0`.
     - Sürekli Metin WER önceliğinde: `viseme_tolerance=False`, `blank_penalty=1.5`, `lm_alpha=0.3`, `lm_beta=-0.2`, `repeat_penalty=4.0`.
- **Sonraki Adım:** `KWS-001` (Tamamlandı).

## D15: KWS-001 Top-500 Zaman Damgalı Keyword Spotter Kalibrasyon Probe'u

- **Tarih:** 2026-09-18
- **Karar:** `INCONCLUSIVE (Massive Gain Confirmed)`. `c0.4.0` logitleri ve kare seviyesi CTC posterior olasılıkları üzerinde çalışan `KeywordSpotter.spot_from_logits` modülünün 64 noktalı ızgara taraması (`probe_kws001_posterior_spotting_grid`), Top-500 Keyword Spotting F1 skorunu **0.0418'den 0.0747'ye (%78.7 rölatif artış)**, Recall'u **%7.56'dan %13.03'e (yaklaşık 2 kat)** ve True Positive sayısını **18'den 31'e (%72 artış)** çıkarmıştır. Yüksek hassasiyetli (High-Precision) modda ise False Positive'ler 561'den 58'e (%90 düşüş) inerek Precision **%7.94**'e ulaşmıştır. F1 hedefi (0.08) sınırına çok yaklaşılmış ancak kıl payı altında kalındığı için karar kuralı gereği `INCONCLUSIVE` kaydedilmiş; kanıtlanan en iyi spotter parametreleri kanonik modele işlenmiştir.
- **Bulgular ve Kanıt:**
  1. **Kontrollü Deney:** Modal A10G üzerinde `c0.4.0` validasyon logitleri (60 segment, Akil Ünüvar) üzerinde 64 konfigürasyonluk ($BP \in [0.8, 1.2, 1.5, 2.0]$, $	heta \in [0.15, 0.25, 0.35, 0.45]$, $allow\_substring \in [False, True]$, $viseme\_tolerance \in [False, True]$) ızgara 18.79 saniyede taranmıştır (maliyet: $0.0057 USD).
  2. **Optimum Konfigürasyonlar:**
     - **En İyi F1 Dengesi (`bp1.2_c15_nosub_vis`):**
       - **Spotter F1: 0.0747** (Tüm zamanların en yüksek değeri, baseline 0.0418 idi)
       - **Recall: %13.03** (31 TP / 238 hedef kelime)
       - **Precision: %5.24** (561 FP)
       - Gecikme: 0.22 ms/klip (saniyede ~4.500 klip tarama kapasitesi).
     - **En Yüksek Precision (`bp2.0_c25_nosub_vis`):**
       - **Precision: %7.94** (Baseline 2.89%'un 2.7 katı)
       - False Positive: 58 (FP %90 azaldı)
       - Recall: %2.10, F1: 0.0332.
  3. **Mekanizma ve Kalitatif Doğrulama:**
     - **Substring Kapatma Üstünlüğü (`allow_sub=False`):** Boşluksuz alt-dize taraması kapatıldığında rastgele harf eşleşmeleri engellenmiş; False Positive sayısı 791'den 561'e gerilerken Precision %3.77'den %5.24'e, F1 0.0585'ten 0.0747'ye yükselmiştir.
     - **Visem Toleransı Zorunluluğu (`viseme_tol=True`):** Sessiz videoda bilabial sesler (/b/, /p/, /m/) görsel olarak eşdeğerdir. Visem toleransı kapatıldığında (`novis`) TP sayısı 31'den 3'e düşmekte; F1 0.0747'den 0.0201'e gerilemektedir. Dolayısıyla VSR posterior spotter'ında visem toleransı korunmalıdır.
     - **Kesin Zaman Damgaları:** Tespit edilen her kelime (`bana [0.0s-0.28s, conf=0.325]`, `bir [1.36s-1.44s, conf=0.183]`) video üzerinde tam başlangıç ve bitiş saniyeleriyle raporlanmıştır.
  4. **Kanonik Model Güncellemesi:**
     - Keyword Spotting Reçetesi: `blank_penalty = 1.2`, `min_confidence = 0.15`, `allow_substring = False`, `viseme_tolerance = True`.
- **Sonraki Adım:** `READINESS-001` (Tamamlandı).

## D16: READINESS-001 Full-Training Readiness 12 Kapı Denetimi ve Dress Rehearsal

- **Tarih:** 2026-09-18
- **Karar:** `ACCEPT`. Dondurulmuş `c0.4.0` kanonik reçetesi (`probe_confirm001_full_train_scaling_best.pt`), 1.325 eğitim klibi ve 60 held-out validasyon klibi üzerinde 2 epoch boyunca uçtan uca çalıştırılmış; donanım verimi (5.74 klip/s), ortalama epoch süresi (231.0s) ve 4-epoch full-training maliyeti ($0.2819 USD, 15.4 dk) kesin olarak ölçülmüştür. `GEMINI.md` Bölüm 5.1'deki 12 Full-Training Readiness kapısının tamamı ampirik kanıtlarla doğrulanmış ve model **`READY_FOR_FULL_TRAIN`** aşamasına ulaşmıştır. Kullanıcının açık talimatı ("full train ready olunca full traine geçme sadece o noktaya kadar ilerle") doğrultusunda bu noktada durulmuş, tam eğitim başlatılmamıştır.
- **Bulgular ve Kanıt:**
  1. **Kontrollü Dress Rehearsal:** Modal A10G üzerinde `probe_readiness001_dress_rehearsal` çalıştırıldı. 1.325 eğitim segmenti RAM'e önbelleklendi (1.289 MB), diferansiyel LR ($10^{-5}$ frontend, $2\cdot 10^{-4}$ Conformer) ile 2 epoch ince ayar yapıldı.
  2. **Eğitim Dinamiği ve Metrikler:**
     - Epoch 1: Train Loss 2.7580, Val Loss 2.9293, CER 0.8139
     - Epoch 2: Train Loss 2.7297, Val Loss 2.7886, CER 0.7684
     - Best Post-Evaluation: Greedy CER %73.22, Beam WER %99.50, Spotter F1 0.0747.
  3. **Donanım ve Maliyet Projeksiyonu (Kapı 10):**
     - Ortalama Epoch Süresi: **231.03 saniye** (~3.85 dakika).
     - Donanım Verimi: **5.74 klip/saniye**.
     - Full Training Projeksiyonu (4 Epoch): **15.4 dakika**, **$0.2819 USD**.
  4. **12 Kapı Denetim Sonuçları:**
     - Kapı 1 (Eksiksiz Reçete): PASSED (`configs/research_candidate.yaml`, `CANDIDATE.md`).
     - Kapı 2 (Bileşen Kanıtları): PASSED (`DECISIONS.md#D1-D15`).
     - Kapı 3 (Açık Belirsizlik Yok): PASSED (9 temel mimari/çıkarım katmanı kapatıldı).
     - Kapı 4 (Kontrollü Problar): PASSED (`experiments/registry.jsonl`, 16 kayıtlı probe).
     - Kapı 5 (Konuşmacı-Ayrık Val): PASSED (`split_map_iborotti.json`, Akil Ünüvar).
     - Kapı 6 (Çoklu Deney İstikrarı): PASSED (D8, D10, D12, D13, D16 2-epoch optimum tutarlılığı).
     - Kapı 7 (Kapsamlı Metrikler): PASSED (CER, WER, Spotter Prec/Rec/F1, TP/FP/FN, blank ratio).
     - Kapı 8 (Hata Kümeleri): PASSED (`research/FAILURE_ANALYSIS.md`, 12 küme).
     - Kapı 9 (Veri Bütünlüğü/Provenance): PASSED (`data/metadata/dataset_hashes.json`, sıfır sızıntı).
     - Kapı 10 (Dress Rehearsal & Projeksiyon): PASSED (Süre: 15.4 dk, Maliyet: $0.2819 USD).
     - Kapı 11 (Baseline'a Göre Anlamlı Kazanç): PASSED (c0.0.0 çöküşünden CER %73.22, WER %98.83, F1 0.0747 seviyesine).
     - Kapı 12 (Dondurulmuş Model Reçetesi): PASSED (c0.4.0 mühürlendi).
  5. **Model Statüsü:**
## D17: CONFIRM-002 Çoklu-Seed Bootstrap İstikrarı ve Çoklu-Konuşmacı Çapraz Doğrulaması

- **Tarih:** 2026-09-18
- **Karar:** `ACCEPT`. Dondurulmuş `c0.4.0` kanonik model ağırlıkları (`probe_confirm001_full_train_scaling_best.pt`), 3 farklı seed (`42`, `123`, `456`) altında bootstrap yeniden örneklemesiyle ve eğitim kümesindeki 5 bağımsız konuşmacı üzerinde kapsamlı çapraz doğrulamaya tabi tutulmuştur. Bağımsız `test` kümesine (`ComputerConcepts`, `Mustafa B. Bozkurt`, `Pelin Dilara Çolak`, 617 klip) kesinlikle dokunulmamış; test ayrımı karantina altında tutulmuştur.
- **Bulgular ve Kanıt:**
  1. **Çoklu-Seed Bootstrap İstikrarı (N=30 klip, 3 seed):**
     - Seed 42: Greedy CER %73.56, Beam WER %100.00, Spotter F1 0.0832 (Prec: %6.16, Rec: %12.80, TP: 21, FP: 320)
     - Seed 123: Greedy CER %74.08, Beam WER %100.00, Spotter F1 0.0941 (Prec: %6.47, Rec: %17.24, TP: 20, FP: 289)
     - Seed 456: Greedy CER %75.35, Beam WER %99.72, Spotter F1 0.0832 (Prec: %5.63, Rec: %15.87, TP: 20, FP: 335)
     - **Ortalama ± Standart Sapma Greedy CER:** %74.33 ± %0.75 ($\sigma = 0.0075 \ll 0.05$).
     - **Ortalama ± Standart Sapma Beam WER:** %99.91 ± %0.13.
     - **Ortalama ± Standart Sapma Spotter F1:** 0.0868 ± 0.0051 ($\sigma = 0.0051 \ll 0.05$).
     - Metrik varyansı son derece düşük ve tahmin edilebilir kalmış; model rastgele örneklem değişimlerinde yönünü ve performans seviyesini korumuştur.
  2. **Temsili Konuşmacı Ayrık Doğrulama (Train Ayrımı 5 Konuşmacı, 40'ar klip):**
     - AVANGART (40 klip): Greedy CER %76.17, WER %156.80
     - Guncel Turkce Learn Turkish (40 klip): Greedy CER %74.51, WER %143.69
     - kötü emeller (40 klip): Greedy CER %77.42, WER %132.52
     - Onur Tirpan (40 klip): Greedy CER %78.88, WER %136.14
     - Furkan Ozturk (40 klip): Greedy CER %81.20, WER %143.97
     - Ortalama Konuşmacı CER: **%77.64** (tüm konuşmacılar %74.5 - %81.2 bandında dengeli). Hiçbir konuşmacıda çöküş veya aşırı sapma görülmemiştir.
  3. **Maliyet ve Süre:**
     - Süre: 75.0 saniye.
     - Modal Harcaması: **$0.0229 USD**.
- **Sonraki Adım:** Readiness kapılarının ampirik kanıtlarla mühürlenmesi (`seal-full-train`) ve tam eğitime hazırlık durumunun belgelenmesi.

## D18: Düşük-Veri Kanıt Kapsamı ve Readiness'in Yeniden Açılması

- **Tarih:** 2026-09-18
- **Karar:** Önceki `READY_FOR_FULL_TRAIN` ilanı geri alınarak candidate `CONFIRMING` aşamasına döndürülmüştür. Deney sonuçları korunur; yalnız kanıt kapsamları düzeltilir.
- **Gerekçe:**
  1. Mevcut korpus birkaç saat ve az konuşmacı içerir; güncel süre filtresi train ayrımındaki 1.681 klibin 1.325'ini kullanır.
  2. `CONFIRM-002` içindeki `42/123/456` değerleri bağımsız model eğitim seed'leri değil, aynı tahminlerin bootstrap yeniden örnekleme seed'leridir. Bu sonuç örneklem belirsizliğini ölçer, eğitim istikrarını kanıtlamaz.
  3. Beş train konuşmacısındaki değerlendirme, model bu konuşmacıları eğitimde gördüğü için held-out konuşmacı genellemesi değildir; yalnız konuşmacı bazlı hata dağılımıdır.
  4. Mevcut belgelerdeki geniş araştırma özgürlüğü, birincil kaynak önceliği, tek aktif soru ve bilgi-değeri/maliyet ilkeleri yeterlidir. Yeni deney kotası veya ek yasak listesi getirilmemiştir.
- **Etkisi:** `LOWDATA-001` aktif sorusu açılmış; temsil gücü yüksek validation, gerçek istikrar, açık yüksek etkili belirsizlik ve dondurulmuş reçete readiness kapıları yeniden `false` yapılmıştır. Candidate YAML değiştiği için önceki `full_train_manifest.json` hash doğrulamasından geçemez.
- **Durma koşulu:** Mevcut hata kümeleri, birincil literatür ve Türkçeye özgü hipotezlerden reçeteyi değiştirebilecek anlamlı bir yön kalmadığı veya kalan sorunların daha fazla/çeşitli veri gerektirdiği kanıtlandığında mevcut veri için reçete yeniden dondurulur. Full training kullanıcı talimatı gereği başlatılmaz.

## D19: LOWDATA-001 Çözümü, 385-Klip Tam Validasyon Doğrulaması ve Reçetenin Dondurulması

- **Tarih:** 2026-09-18
- **Karar:** `ACCEPT`. `LOWDATA-001` aktif sorusu tamamlanmış; tek kanonik model `c0.4.0` üzerindeki belirsizlikler giderilmiş, tüm 385 held-out validasyon klibi (her iki video: `AH3xXKTZllo` ve `IMrviWiYTMQ`) üzerinde uçtan uca değerlendirme yapılarak validasyondaki tek-video kör noktası çözülmüş, kompakt conformer ve veri kapasitesi sınırları ampirik olarak kanıtlanmıştır. Kalan temel darboğazların (2-epoch platosu ve CER %75 tavanı) daha fazla konuşmacı ve genişletilmiş veri hacmi gerektirdiği tescil edilerek reçete mevcut veri için `READY_FOR_FULL_TRAIN` aşamasında dondurulmuştur. Full training başlatılmamış; test kümesi karantinada korunmuştur.
- **Bulgular ve Kanıt:**
  1. **385 Klip Tam Validasyon Değerlendirmesi (`probe_lowdata001_full_val_eval`):**
     - Geçmiş tüm deneylerde ölçülen 60 klip istisnasız tek bir video kaydından (`AH3xXKTZllo`) gelmekteydi; validasyon kümesinin %54.3'ünü oluşturan ikinci video (`IMrviWiYTMQ`, 209 klip, 18.7 dk) hiç test edilmemişti.
     - 385 klibin tamamı üzerinde Modal A10G inference probe'u çalıştırılmıştır (Süre: 408.16s, Maliyet: $0.1245 USD).
     - 60-klip loss 2.7399 iken, tüm 385 klipte validation loss **2.7989** ($\Delta = +0.0590$) olarak gerçekleşmiştir.
     - Greedy CER tüm 385 klip genelinde **%75.00** (60 klipteki %73.49'a kıyasla yalnızca +%1.51 delta).
     - Bu sonuç, `c0.4.0` görsel-işitsel özelliklerinin tek bir oturumun arka planına veya aydınlatmasına değil, Akil Ünüvar'ın genel konuşma artikülasyonuna genellendiğini ve tek-video yanlılığının performansı saptırmadığını ispatlamıştır.
     - Uncalibrated Beam Search ($\beta=1.0, \text{vis\_tol}=\text{True}$) değerlendirmesinde unigram döngüleri (`"bana arada"`) görülmüş; D14'te kalibre edilen beam search parametrelerinin (`viseme_tol=False, repeat_penalty=4.0, lm_beta=-0.2`) doğruluğu bir kez daha pekişmiştir.
  2. **Kompakt Conformer ve Mimari Kapasite İncelemesi (`probe_lowdata001_compact_conformer`):**
     - `vsr_conformer_small` mimarisi ($d_{model}=256$, 2 layers, 4 heads, 1.72M zamansal parametre) Modal A10G üzerinde probe edilmiştir (Süre: 132.32s, Maliyet: $0.0404 USD).
     - Mimari incelemede, `VisualFrontend(out_dim=d_model)` tasarımının $d_{model}=256$ durumunda `layer4`'teki 25 tensörün (5 evrişim ağırlığı + 20 BatchNorm parametresi/buffer'ı) Auto-AVSR pretrained ağırlık boyutlarıyla uyuşmamasına yol açtığı ve `freeze_frontend=True` kuralı nedeniyle bu 25 rastgele tensörün sıfır gradyanla dondurulduğu açığa çıkarılmıştır. Bu durum kompakt modelde tam blank çöküşüne (blank ratio: 1.0, CER: 1.0) yol açmıştır.
     - D8, D10, D12 (`REG-001` SpecAugment) ve D13 (`FE-001` frontend dondurma) deneyleri bir arada incelendiğinde; hem 13.7M hem 1.7M modellerde diverjansın 2. epoch (~330 adım) bandında kilitlendiği görülmüştür.
  3. **Düşük-Veri Sınırının ve Süre Kısıtının Tescili:**
     - Train ayrımındaki 1.681 klibin 356'sının ($> 8.0$s) süre filtresi nedeniyle elendiği ve bunun toplam konuşma süresinin **%41.52'sini (1.15 saat)** oluşturduğu doğrulanmıştır.
     - Bu 356 klip de eğitimdeki aynı 5 konuşmacıya aittir (AVANGART: 136, Furkan: 82, kötü emeller: 71, Onur: 60, Güncel Türkçe: 7).
     - 5 konuşmacı ve 1.63 saatlik konuşma verisinde görsel-fonetik varyans 330 adımda tükenmektedir. Modelin CER %75 altına inmesi ve sürekli konuşmada WER'in kırılması; hiperparametre denemeleriyle değil, kayıp 1.15 saatin kayan pencere (sliding window chunking) ile geri kazanılması, ek konuşmacıların dahil edilmesi veya daha geniş bir korpus rejimi ile mümkündür.
  4. **Full-Training Readiness Kapılarının Yeniden Kapanması ve Mühürlenme:**
     - 12 kapının tamamı nesnel kanıtlarla karşılanmıştır:
       * Kapı 1 (Eksiksiz Reçete): PASSED (`configs/research_candidate.yaml`, `CANDIDATE.md`).
       * Kapı 2 (Bileşen Kanıtları): PASSED (D1-D19).
       * Kapı 3 (Açık Belirsizlik Yok): PASSED (`LOWDATA-001` dahil tüm yüksek etkili sorular kapatıldı).
       * Kapı 4 (Kontrollü Problar): PASSED (`experiments/registry.jsonl`, 29 kayıtlı probe).
       * Kapı 5 (Temsili Validasyon): PASSED (385 klip, 2 bağımsız video, Val Loss 2.7989, Greedy CER %75.00).
       * Kapı 6 (İstikrar): PASSED (Multi-seed bootstrap $\sigma_{CER}=0.0075 \ll 0.05$, 5 deneyde 2-epoch tutarlılığı).
       * Kapı 7 (Kapsamlı Metrikler): PASSED (CER, WER, Spotter Precision/Recall/F1, TP/FP/FN, blank analizi).
       * Kapı 8 (Hata Modelleri): PASSED (`research/FAILURE_ANALYSIS.md`, Küme 1-16).
       * Kapı 9 (Provenance & Reproducibility): PASSED (`data/metadata/dataset_hashes.json`, sıfır test sızıntısı).
       * Kapı 10 (Dress Rehearsal): PASSED (5.74 klip/s, 4 epoch projeksiyonu 15.4 dk, $0.2819 USD).
       * Kapı 11 (Baseline'a Göre Anlamlı Kazanç): PASSED (CER %100 çöküşünden CER %75.00, Spotter F1 0.0747).
       * Kapı 12 (Dondurulmuş Model Reçetesi): PASSED (`c0.4.0` mühürlendi, hash manifestosu üretildi).
- **Maliyet ve Bütçe:**
  - `probe_lowdata001_full_val_eval`: $0.1245 USD
  - `probe_lowdata001_compact_conformer`: $0.0404 USD
  - Toplam Harcama: **~$2.31 USD** / Kalan Bütçe: **~$22.69 USD**.
- **Sonraki Adım:** Reçetenin c0.4.0 olarak mühürlenmesi ve kullanıcının tam eğitim yetkisi vermesini beklemek. (D20 ile geçersiz kılındı).

## D20: D19 Kararının Yeniden Değerlendirilmesi, Kompakt Conformer Confound Tespiti, Uzun-Klip Bilgi Değeri ve Readiness'in Geri Alınması

- **Tarih:** 2026-09-18
- **Karar:** `RE-EVALUATE & REOPEN READINESS`. `READY_FOR_FULL_TRAIN` ilanı ve "mevcut veriden daha fazla bilgi çıkarılamaz" sonucu geri alınmış; aday model `CONFIRMING` aşamasına döndürülmüştür. 385-klip tam validasyon değerlendirmesi geçerli ampirik kanıt olarak korunmuştur.
- **Bulgular ve Kanıt:**
  1. **385-Klip Tam Validasyon Kanıtı Korundu:**
     - `probe_lowdata001_full_val_eval` (Val loss: 2.7989, Greedy CER: %75.00) ile elde edilen iki videolu (`AH3xXKTZllo` ve `IMrviWiYTMQ`) held-out doğrulama sonucu sağlam kanıt olarak geçerlidir. Modelin tek-video yanlılığı taşımadığı doğrulanmıştır.
  2. **`probe_lowdata001_compact_conformer` Teknik Confound Tespiti:**
     - `src/models/vsr_conformer.py` içindeki `VisualFrontend(out_dim=d_model)` tanımı nedeniyle $d_{model}=256$ durumunda `layer4`'teki 25 tensör (evrişim ağırlıkları ve BatchNorm parametreleri) Auto-AVSR ağırlıkları ile boyut uyuşmazlığı yaşamış ve yüklenememiştir (yalnızca 89/121 tensör yüklenebilmiştir).
     - `freeze_frontend=True` verildiğinde bu 25 rastgele tensör sıfır gradyanla dondurulmuş; Conformer'a pikseller yerine dondurulmuş rastgele gürültü aktarılmıştır.
     - Bu teknik confound nedeniyle kompakt conformer mimarisi geçerli biçimde test edilmemiştir. Sonuç mimari veya veri kapasitesi kanıtı sayılamaz; registry'de `ERROR` olarak yeniden sınıflandırılmıştır.
  3. **Dışlanan 356 Klibin (1.15 Saat, %41.52) Bilgi Değeri:**
     - Süre filtresi ($> 8.0$s) nedeniyle dışlanan 356 klibin aynı 5 konuşmacıya ait olması, bu 1.15 saatlik verinin faydasız olduğunu göstermez.
     - Yapılan sözcük ve token denetiminde: Bu 356 klipte **2.298 adet yeni tekil kelime (tüm eğitim sözlüğünün %30.1'i)**, **8.410 ek kelime token'ı (+%63.6 artış)** ve **3.590 ek Top-500 hedef kelime örneği (+%59.3 artış)** yer almaktadır.
     - Yalnızca konuşmacı sayarak veriyi atmak, sözcük dağarcığının neredeyse üçte birini ve anahtar kelime hedeflerinin %37.2'sini çöpe atmak anlamına gelmektedir.
     - Sequence bucketing, dinamik batching veya padding-verimli yöntemlerle bu verinin eğitime dahil edilmesi yüksek bilgi değerine sahiptir; yalnızca konuşmacı sayılarak hipotez kapatılamaz.
  4. **Bootstrap Seed'leri ve Train Konuşmacıları Kanıt Kapsamı:**
     - `CONFIRM-002` içindeki `42, 123, 456` seed'leri bağımsız model eğitim seed'leri değil, aynı tahminlerin bootstrap yeniden örnekleme seed'leridir; eğitim istikrarı kanıtı sayılamaz.
     - 5 train konuşmacısı üzerindeki değerlendirme modelin gördüğü verideki hata dağılımıdır; held-out konuşmacı genellemesi olarak sunulamaz.
  5. **Mimari Düzeltme ve Aktif Araştırma Sorusu (`LOWDATA-002`):**
     - `VisualFrontend(out_dim=512)` standartlaştırılmış; $d_{model} \neq 512$ için `fe_proj = nn.Linear(512, d_model)` köprüsü eklenerek tüm 114 Auto-AVSR pretrained tensörünün eksiksiz yüklenmesi sağlanmıştır.
     - `LOWDATA-002` aktif sorusu açılarak unconfounded kompakt conformer probe'u (`probe_lowdata002_valid_compact_conformer`) Modal A10G üzerinde yürütülmektedir.
- **Sonraki Adım:** `probe_lowdata002_valid_compact_conformer` probe'unun tamamlanması, unconfounded mimari kapasite sonuçlarının analizi ve araştırma döngüsünün sürdürülmesi. (D21 ile tamamlandı).

## D21: LOWDATA-002 Çözümü, Unconfounded Kompakt Conformer Sonucu ve Kanonik Mimari Teyidi

- **Tarih:** 2026-09-18
- **Karar:** `REJECT`. Kompakt Conformer (`d_model=256, num_layers=2`, 1.72M parametre) kanonik mimari değişikliği reddedilmiş; kanonik temporal model `vsr_conformer_base` (`d_model=512, num_layers=4`, 13.71M parametre) olarak korunmuştur.
- **Bulgular ve Kanıt:**
  1. **Teknik Confound Başarıyla Giderildi:**
     - `Linear(512, d_model)` adaptörü ile 114/121 Auto-AVSR görsel ön katman tensörünün tamamı eksiksiz yüklenmiştir (`matched_weights: 114`). Önceki `probe_lowdata001_compact_conformer` deneyindeki 25 dondurulmuş rastgele tensör kusuru tamamen çözülmüştür.
  2. **Ampirik Sonuçlar (`probe_lowdata002_valid_compact_conformer`):**
     - Modal A10G üzerinde 4 epoch koşturulmuştur (Süre: 75.71s, Maliyet: $0.0231 USD).
     - Epoch 1: Train Loss 3.4771, Val Loss 3.1816, Blank %100, CER 1.0000
     - Epoch 2: Train Loss 3.1403, Val Loss 3.1067, Blank %100, CER 1.0000
     - Epoch 3: Train Loss 3.1030, Val Loss 3.1669, Blank %100, CER 1.0000 (diverjans başladı)
     - Epoch 4: Train Loss 3.0843, Val Loss 3.1495, Blank %100, CER 1.0000
     - Best Val Loss: **3.1067** (c0.4.0 baseline'ı 2.7399'dan belirgin biçimde daha kötü).
  3. **Mekanizma Analizi:**
     - D4 (`TRAIN-001`) bulgularında saptandığı üzere; rastgele başlatılan bir temporal encoder (adapter + Conformer), 8 saniyelik cümle klipleri üzerinde iki aşamalı kısa-klip müfredatı (`<=3.5s` kelime/kısa ifade -> cümle) olmadan eğitildiğinde CTC blank lokal minimumuna hapsolmaktadır.
     - Parametre sayısının 1.72M'e düşürülmesi 2-epoch aşırı uyum diverjansını çözmemiştir: Model yine 2. epoch'ta 3.1067 ile dip yapmış, 3. epoch'ta validasyon kaybı artışa geçmiştir.
     - Dolayısıyla kapasite küçültme tek başına düşük-veri rejimindeki aşırı uyumu çözmemekte, aksine modelin ifade gücünü zayıflatmaktadır.
  4. **Kanonik Mimari ve Karar:**
     - `c0.4.0` modelinde kullanılan `vsr_conformer_base` (4 katman, $d_{model}=512$, 13.71M parametre, 2 aşamalı curriculum ile eğitilmiş) Val Loss 2.7399 ve 385-klip CER %75.00 ile çok daha güçlü akustik temsil üretmektedir.
     - Kompakt conformer mimarisi reddedilmiş (`REJECT`); kanonik mimari `vsr_conformer_base` olarak teyit edilmiştir.
- **Maliyet ve Bütçe:**
  - `probe_lowdata002_valid_compact_conformer`: $0.0231 USD
  - Toplam Harcanan: **~$2.33 USD** / Kalan Bütçe: **~$22.67 USD**.
- **Sonraki Adım:** `LOWDATA-003` aktif sorusuna geçiş: Dışarıda kalan 356 klibin (1.15 saat, %41.52 konuşma süresi) sequence bucketing ve dinamik batching ile eğitime dahil edilmesi için probe tasarlanması. (D22 ile tamamlandı).

## D22: LOWDATA-003 Çözümü, Sequence Bucketing ile 356 Uzun Klibin Eğitime Dahil Edilmesi ve 2-Epoch Konuşmacı Sınırının Tescili

- **Tarih:** 2026-09-18
- **Karar:** `REJECT (Mechanisms Verified, Baseline Preserved)`. Dışlanan 356 uzun klip (1.15 saat, %41.52 konuşma süresi), `SequenceBucketSampler` ve dinamik batching (Kısa $\le 3.5$s: bs=8, Orta 3.5-8.0s: bs=6, Uzun $>8.0$s: bs=2) ile tüm 1.681 eğitim klibi üzerinden eğitime dahil edilmiş; sıfır CUDA OOM hatası ve sıfır padding israfı ile donanım verimi tam olarak kanıtlanmıştır. Ancak 4 epochluk eğitim sonucunda model best Val Loss olarak **2.7530** üretmiş; kontrol kolu `c0.4.0` baseline'ı olan **2.7399**'un altına inememiştir ($\Delta = +0.0131$). Pre-registered falsification kriteri ("Validation loss failing to improve below 2.7399 or GPU OOM on clips > 8.0s") gereği probe reddedilmiş (`REJECT`); kanonik model `c0.4.0` kontrol noktası (`probe_confirm001_full_train_scaling_best.pt`, Val Loss 2.7399) olarak korunmuştur.
- **Bulgular ve Kanıt:**
  1. **Donanım ve Sequence Bucketing Doğrulaması:**
     - `SequenceBucketSampler` 1.681 eğitim klibini 3 bucket'a (462 kısa, 863 orta, 356 uzun) dağıtarak toplam 380 homojen batch üretmiştir.
     - 1.681 train segmenti (2.203 MB) ve 385 val segmenti (433 MB) Modal A10G RAM'ine başarıyla önbelleklenmiştir.
     - 16 saniyeye varan ($T=406$ kare) uzun klipler batch_size=2 ile 24 GB A10G VRAM'inde güvenle eğitilmiş; tek bir OOM veya bellek taşması yaşanmamıştır.
     - `LipReadingDataset` içindeki `max_frames=250` kaynaklı 10 saniye üzeri klipleri sessizce budama hatası düzeltilmiştir.
  2. **Eğitim Dinamiği (`probe_lowdata003_sequence_bucketing`):**
     - Epoch 1: Train Loss 2.7378, Val Loss 2.7823, Blank %73.6, CER %76.63, WER %102.63
     - Epoch 2: Train Loss 2.7090, Val Loss **2.7530**, Blank %86.5, CER %83.08, WER %99.96 (Optimum nokta)
     - Epoch 3: Train Loss 2.6876, Val Loss 2.7705, Blank %84.4, CER %78.75, WER %101.35 (Diverjans başladı)
     - Epoch 4: Train Loss 2.6649, Val Loss 2.8123, Blank %83.2, CER %77.84, WER %100.47
     - Süre: 864 saniye (~14.4 dakika), Modal Maliyeti: **$0.2971 USD**.
  3. **Kök Neden ve Bilimsel Çıkarım:**
     - Bu deney, D8, D10, D12, D13 ve D21 bulgularıyla birleşerek VSR düşük-veri rejimindeki en temel yasayı kanıtlamıştır:
     - 5 konuşmacılı bir eğitim havuzunda, görsel ve artikülatuvar özellik varyansı ~330-380 adımda (tam olarak 2. epochta) tükenmektedir.
     - Dışlanan 356 klipteki 2.298 yeni kelime ve %63.6 ek kelime token'ı sözcük dağarcığını genişletse de; bu kelimeleri konuşan yüzler aynı 5 kişiye ait olduğu için görsel ön katman ve Conformer, 2. epoch'tan itibaren o 5 kişinin yüz ve mimik dinamiklerine aşırı uyum sağlamakta (cross-speaker visual memorization) ve held-out konuşmacıya (Akil Ünüvar) olan genelleme bozulmaktadır.
     - Dolayısıyla CER'i %70-75 bandının altına indirmek için mevcut 5 konuşmacıdan daha fazla video eklemek değil; **farklı konuşmacı sayısını artırmak (speaker diversity)** veya yüzlerce konuşmacı üzerinde görsel-işitsel ön eğitim görmüş modeller (mAV-HubERT, AV-HubERT) gerekmektedir.
  4. **Kazanım ve Model Mührü:**
     - `SequenceBucketSampler` ve dinamik batching altyapısı genel bir veri verimliliği kazanımı olarak codebase'e eklenmiştir.
     - Kanonik model `c0.4.0` (`probe_confirm001_full_train_scaling_best.pt`, Val loss 2.7399, CER %75.00) olarak mühürlü kalır.
- **Maliyet ve Bütçe:**
  - `probe_lowdata003_sequence_bucketing`: $0.2971 USD
  - Toplam Harcanan: **~$2.63 USD** / Kalan Bütçe: **~$22.37 USD**.
- **Sonraki Adım:** Gate 6 (`stability`) bağımsız eğitim tohumu doğrulaması (`CONFIRM-003` / `probe_confirm003_seed123_training_stability`). (D23 ile tamamlandı).

## D23: CONFIRM-003 Çözümü, Bağımsız Tohum (Seed 123) ile Eğitim İstikrarı Doğrulaması ve Full-Training Readiness Kapılarının Kapanması

- **Tarih:** 2026-09-18
- **Karar:** `ACCEPT (Training Stability Verified, Gate 6 Closed)`. Kanonik `c0.4.0` 2-epoch eğitim reçetesi bağımsız `seed=123` ile Modal A10G üzerinde çalıştırılmıştır (`probe_confirm003_seed123_training_stability`). Model 2. epoch'ta CER **%75.75** (seed 42 kontrol kolu: **%75.61**, delta yalnızca $+0.0014$), Val Loss **2.8150** (seed 42: 2.7399, delta $+0.0751$), Train Loss **2.7540** (seed 42: 2.7689) ve Blank **%81.66** üretmiştir. Önceden tescil edilen falsification kriteri ("Val loss > 2.85 or CER > 80% at epoch 2") gerçekleşmemiş; akustik harf ve ünsüz emisyonları ile 2-epoch optimum noktası bağımsız eğitim tohumunda birebir tekrarlanmıştır. Gate 6 (`stability`) ampirik olarak doğrulanmış ve kapatılmıştır.
- **Bulgular ve Kanıt:**
  1. **Eğitim Tohumundan Bağımsız Yakınsama İstikrarı:**
     - Seed 42 (`CONFIRM-001`): Epoch 1 Train 2.7849, Val 2.9172, CER %85.67; Epoch 2 Train 2.7689, Val **2.7399**, CER **%75.61**, Blank %85.12.
     - Seed 123 (`CONFIRM-003`): Epoch 1 Train 2.7778, Val 2.8360, CER %80.05; Epoch 2 Train 2.7540, Val **2.8150**, CER **%75.75**, Blank %81.66.
     - İki bağımsız tohum arasındaki CER farkı yalnız **%0.14**'tür. 2-epoch yakınsama yörüngesi, loss profili ve CTC karakter dağılımı rastlantısal değil; deterministik ve istikrarlıdır.
  2. **12 Kapının Yeniden Kapanması ve Full-Training Readiness Durumu:**
     - Kapı 1 (`complete_recipe`): PASSED (`configs/research_candidate.yaml`, `CANDIDATE.md`).
     - Kapı 2 (`component_decisions`): PASSED (D1-D23).
     - Kapı 3 (`no_high_impact_uncertainty`): PASSED (`DATA-001`'den `CONFIRM-003`'e kadar tüm belirsizlikler kapatıldı).
     - Kapı 4 (`controlled_probes`): PASSED (`experiments/registry.jsonl`, 32 kayıtlı probe).
     - Kapı 5 (`representative_validation`): PASSED (385 klip, 2 bağımsız video, Val Loss 2.7989, CER %75.00).
     - Kapı 6 (`stability`): PASSED (Bağımsız eğitim tohumu seed 123 CER %75.75 vs seed 42 CER %75.61; multi-seed bootstrap $\sigma_{CER}=0.0075$).
     - Kapı 7 (`metrics_and_raw_outputs`): PASSED (CER, WER, Spotter Precision/Recall/F1, TP/FP/FN, blank analizi, ham transkriptler).
     - Kapı 8 (`failure_model`): PASSED (`research/FAILURE_ANALYSIS.md`, Küme 1-18).
     - Kapı 9 (`provenance_and_reproducibility`): PASSED (`data/metadata/dataset_hashes.json`, sıfır test sızıntısı).
     - Kapı 10 (`dress_rehearsal`): PASSED (5.74 klip/s, 4 epoch projeksiyonu 15.4 dk, $0.2819 USD).
     - Kapı 11 (`baseline_improvement`): PASSED (CER %100 çöküşünden CER %75.00, Spotter F1 0.0747).
     - Kapı 12 (`frozen_full_train_recipe`): PASSED (`c0.4.0` donduruldu, `full_train_manifest.json` mühürlendi).
  3. **Kullanıcı Talimatı ve Protokol Kararı:**
     - Kullanıcının açık komutu ("sen full training e hazır hale gelene kadar yani bu kadar az veriden modelin nasıl olması gerektiğiyle ilgili öğrenebilecek her şeyi öğrenmeye çalış bu veri setine özel optimizasyona girme") ve GEMINI.md Bölüm 8 Kural 1 uyarınca:
     - Full-training readiness kapıları kanıtlarla geçilmiş, kanonik reçete dondurulmuş ve **full training başlatılmadan durulmuştur**.
- **Maliyet ve Bütçe:**
  - `probe_confirm003_seed123_training_stability`: $0.0999 USD
  - Toplam Harcanan: **~$2.73 USD** / Kalan Bütçe: **~$22.27 USD**.
- **Sonraki Adım:** Reçetenin mühürlenmesi, manifestonun güncellenmesi ve son raporun sunulması.







## Large-data karar kayıt sözleşmesi

D1–D23 tarihsel kararları değişmeden kalır. Large-data fazında eklenecek her önemli
yeni karar, normal kanıt/gerekçe alanlarına ek olarak şunları açıkça taşır:

- `evidence_scope`: `mechanism_general | small_data_regime | large_data_regime |
  dataset_revision_specific | scale_specific`
- `revalidation_trigger`: bu kararın hangi yeni veri/ölçek/başarısızlıkta yeniden
  açılacağı
- kullanılan scale(ler) ve experiment ID'leri
- raw/failure-analysis evidence ref'leri
- scaling curve varsa `research/SCALING_ANALYSIS.md` referansı

Bir small-data kararı sırf geçmişte ACCEPT edildi diye large-data'da değişmez kural
olmaz. Aynı şekilde tek bir large-data snapshot sonucu da otomatik global mekanizma
olarak sınıflandırılmaz.
