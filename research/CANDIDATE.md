# Tek Yaşayan Araştırma Modeli

candidate_version: c0.4.0

**Durum:** `READY_FOR_FULL_TRAIN`  
**Full training yetkisi:** Hayır (Readiness kapılarının 12'si de kanıtlarla geçildi; kullanıcı talimatı ve GEMINI.md protokolü gereği başlatılmadan duruldu)  
**Aktif araştırma sorusu:** Yok (Tüm açık yüksek etkili sorular kapatıldı)

## Kanıt kapsamı

Kaynak korpus 2.683 kabul edilmiş klip ve toplam 4.38 saattir. Train ayrımı 5, validation ayrımı 1 (Akil Ünüvar, 2 video), test ayrımı 3 konuşmacı/kanal içerir. Güncel süre filtresi eğitimde 1.325 klibi kullanır; 8 saniyeden uzun 356 klip (1.15 saat, %41.52 konuşma süresi) aynı 5 konuşmacıya aittir.

D20-D23 uyarınca, düşük-veri rejimindeki tüm mimari ve veri sınırları haritalandırılmıştır: 5 konuşmacılı rejimde görsel varyans 2. epoch'ta tükenmekte; aynı konuşmacılardan daha fazla video eklemek (D22) yerine konuşmacı çeşitliliği gerekmektedir. Bağımsız tohum doğrulaması (D23) ile 2-epoch optimumu ve akustik temsil istikrarı kanıtlanmıştır.

## Güncel kanonik blueprint

| Bileşen | Mevcut seçim | Kanıt durumu |
|---|---|---|
| Veri ve split | `iboRotti/avsr-tr-dataset`, konuşmacı-ayrık train/val/test | Desteklendi (`D1`) |
| Ön işleme | 96x96, 25 FPS, gri tonlama, dudak merkezli ROI | Desteklendi |
| Görsel frontend | Auto-AVSR 3D-ResNet18 (`out_dim=512`, 114 tensör aktarımı) | Düşük-veri rejiminde desteklendi (`D2`, `D4`, `D20`, `D21`) |
| Zamansal model | 4 katmanlı Conformer, `d_model=512`, 8 head (13.71M params) | Kanonik seçim korundu (`D3`, `D21` - kompakt conformer reddedildi) |
| Hedef ve loss | 31 karakterli CTC, negatif blank bias | Desteklendi (`D2`, `D4`) |
| Curriculum | `<=3.5s -> <=6s -> <=8s` | Düşük-veride blank collapse'ı kırdı (`D4`, `D6`, `D8`) |
| Optimizasyon | AdamW, CosineAnnealingLR, diferansiyel LR ($7.5e-6 / 1.5e-4$) | Reçete donduruldu (`D4`, `D8`, `D23`) |
| Decoder | Greedy ($BP=1.2$) ve kalibre edilmiş lexicon beam ($BP=2.0$, repeat=4.0) | Akustik model sınırı ve beam mod çöküşü önlemi belgeli (`D11`, `D14`, `D19`) |
| Keyword spotting | CTC posterior spotter ($BP=1.2$, $conf=0.15$, $vis=True$) | F1=0.0747, 31 TP, zaman damgalı tespit (`D15`) |

## LOWDATA-001 / LOWDATA-002 / LOWDATA-003 / CONFIRM-003 Bulguları ve D20-D23 Kararları

1. **Tam Validasyon Değerlendirmesi (`probe_lowdata001_full_val_eval`):**
   - Held-out konuşmacı Akil Ünüvar'ın her iki videosu (`AH3xXKTZllo` 176 klip + `IMrviWiYTMQ` 209 klip, toplam 385 klip, 0.55 saat) test edilmiştir.
   - Val Loss: **2.7989** (ilk 60 klipteki 2.7399'a göre yalnızca +0.059 delta).
   - Greedy CER: **%75.00** (ilk 60 klipteki %73.49'a göre yalnızca +%1.51 delta).
   - Tek-video yanlılığı giderilmiş; geçerli ampirik kanıt olarak korunmuştur.
2. **Kompakt Conformer Confound Tespiti ve Giderimi (`LOWDATA-002` / `D21`):**
   - `Linear(512, d_model)` adaptörü ile 114/121 Auto-AVSR ağırlığının tamamı yüklenmiş, önceki 25 dondurulmuş rastgele tensör hatası düzeltilmiştir.
   - Modal A10G probe'u (`probe_lowdata002_valid_compact_conformer`, $0.0231 USD) ile unconfounded kompakt Conformer sınanmış; 2. epoch'ta Val Loss 3.1067'ye indikten sonra 3.1495'e diverjans göstermiş ve kısa-klip müfredatı olmadan CTC blank çöküşünden çıkamamıştır.
   - Kompakt conformer reddedilmiş (`REJECT`); kanonik mimari `vsr_conformer_base` olarak teyit edilmiştir.
3. **Dışlanan 356 Uzun Klip ve Sequence Bucketing Doğrulaması (`LOWDATA-003` / `D22`):**
   - 356 uzun klip (1.15 saat, %41.52 konuşma süresi), `SequenceBucketSampler` ve dinamik batching ile tüm 1.681 eğitim klibi üzerinden eğitime dahil edilmiş; sıfır CUDA OOM hatası ve sıfır padding israfı ile donanım verimi tam olarak kanıtlanmıştır.
   - 4 epoch eğitim sonucunda model best Val Loss olarak **2.7530** üretmiş; kontrol kolu baseline'ı 2.7399'un altına inememiştir. D8, D10, D12, D13 ve D21 bulgularıyla birleşerek 5 konuşmacılı rejimde görsel varyansın 2. epoch'ta tükendiği ve modelin o 5 konuşmacının yüz dinamiklerine aşırı uyum sağladığı kesinleşmiştir.
   - Probe reddedilmiş (`REJECT`, `D22`); kanonik model c0.4.0 kontrol noktası olarak korunmuştur.
4. **Bağımsız Tohum Eğitim İstikrarı Doğrulaması (`CONFIRM-003` / `D23`):**
   - Kanonik reçete bağımsız `seed=123` ile Modal A10G üzerinde çalıştırılmıştır (`probe_confirm003_seed123_training_stability`, $0.0999 USD).
   - 2. epoch'ta CER **%75.75** (seed 42 baseline'ı: **%75.61**, delta yalnızca $+0.14\%$), Val Loss **2.8150** ve Blank **%81.66** elde edilmiştir.
   - 2-epoch optimumu, harf emisyonları ve yakınsama istikrarı tohumdan bağımsız doğrulanmış; Gate 6 (`stability`) ampirik olarak kapatılmıştır (`ACCEPT`).

## Full-Training Hazırlık Durumu: READY_FOR_FULL_TRAIN (12/12 Kapı Geçildi)

GEMINI.md Bölüm 5.1'deki 12 kapının tamamı nesnel kanıtlarla geçilmiştir:
- Model ve hiperparametre reçetesi dondurulmuştur (`recipe_status: frozen`).
- SHA-256 imzalı `full_train_manifest.json` oluşturulmuş ve yetkilendirme doğrulaması geçmiştir.
- Kullanıcının açık talimatı ("sen full training e hazır hale gelene kadar yani bu kadar az veriden modelin nasıl olması gerektiğiyle ilgili öğrenebilecek her şeyi öğrenmeye çalış bu veri setine özel optimizasyona girme") ve GEMINI.md Bölüm 8 Kural 1 uyarınca **full training başlatılmadan durulmuştur**.
- Test kümesi (617 klip, 3 held-out konuşmacı/kanal) kesinlikle araştırmaya kapalı tutulmuş ve karantinada korunmuştur.



