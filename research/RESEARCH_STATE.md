# Güncel Araştırma Durumu

**Candidate:** `c0.4.0`  
**Aşama:** `READY_FOR_FULL_TRAIN`  
**Full training:** Donduruldu / Yetki Bekleniyor (Readiness kapılarının 12'si de kanıtlarla geçildi; kullanıcı talimatı ve GEMINI.md protokolü gereği başlatılmadan duruldu)  
**Aktif soru:** Yok (Tüm açık yüksek etkili sorular kapatıldı)  

## Güncel durum özeti

D20, D21, D22 ve D23 kararları kapsamında düşük-veri rejimindeki mimari, veri, optimizasyon ve istikrar sınırları eksiksiz netleştirilmiştir:
1. `probe_lowdata001_full_val_eval` (Probe 28): Held-out validasyon kümesindeki 385 segmentin tamamı (Akil Ünüvar'ın her iki videosu: `AH3xXKTZllo` ve `IMrviWiYTMQ`) test edilmiş, tek-video kör noktası çözülmüş, Val loss 2.7989, Greedy CER %75.00 olarak ölçülmüştür. Bu sonuç geçerli ampirik kanıt olarak korunmaktadır.
2. `probe_lowdata001_compact_conformer` (Probe 29): Teknik confound (25 tensörün rastgele dondurulması) nedeniyle `ERROR` olarak sınıflandırılmıştır.
3. `probe_lowdata002_valid_compact_conformer` (Probe 30): `Linear(512, d_model)` adaptörü ile 114/121 Auto-AVSR tensörünün tamamı yüklenerek confound giderilmiştir. 4 epoch eğitim sonucunda model best Val Loss 3.1067'ye inmiş, ardından 3.1495'e diverje olmuş ve kısa-klip müfredatı olmadan blank trap'ten çıkamamıştır. Kompakt conformer mimari değişikliği reddedilmiş (`REJECT`, `D21`); kanonik model `vsr_conformer_base` (13.71M parametre, 4 katman) olarak korunmuştur.
4. `probe_lowdata003_sequence_bucketing` (Probe 31): Dışlanan 356 uzun klip (1.15 saat, %41.52 konuşma süresi, 2.298 yeni kelime), `SequenceBucketSampler` ve dinamik batching ile tüm 1.681 eğitim klibi üzerinden eğitime dahil edilmiş; sıfır CUDA OOM ve sıfır padding israfı ile donanım verimi kanıtlanmıştır. Ancak 4 epochluk eğitim sonucunda model best Val Loss olarak 2.7530 üretmiş; baseline 2.7399'un altına inememiştir. D8, D10, D12, D13 ve D21 bulgularıyla birleşerek 5 konuşmacılı rejimde görsel varyansın 2. epoch'ta tükendiği ve modelin o 5 konuşmacının yüz dinamiklerine aşırı uyum sağladığı kesinleşmiştir. Probe reddedilmiş (`REJECT`, `D22`); kanonik model c0.4.0 korunmuştur.
5. `probe_confirm003_seed123_training_stability` (Probe 32): Kanonik c0.4.0 reçetesi bağımsız `seed=123` ile eğitilmiş; 2. epoch'ta CER %75.75 (seed 42 baseline'ı %75.61 ile $\Delta = +0.14\%$), Val Loss 2.8150 ve Blank %81.66 elde edilmiştir. 2-epoch optimum noktası ve yakınsama istikrarı tohumdan bağımsız doğrulanmış; Gate 6 (`stability`) deneysel olarak kapatılmıştır (`ACCEPT`, `D23`).
6. GEMINI.md Bölüm 5.1'deki 12 kapının tamamı geçilmiş, model reçetesi `configs/research_candidate.yaml` ve `full_train_manifest.json` dosyalarında dondurulmuş; kullanıcının açık talimatı ("sen full training e hazır hale gelene kadar yani bu kadar az veriden modelin nasıl olması gerektiğiyle ilgili öğrenebilecek her şeyi öğrenmeye çalış bu veri setine özel optimizasyona girme") ve GEMINI.md Bölüm 8 Kural 1 gereği **full training başlatılmadan durulmuştur**. Test kümesi kesinlikle karantinada korunmaktadır.

## Tamamlanan Temel Aşamalar (D1 - D23)

- `DATA-001` (D1): 2.683 kabul edilen klip (4.38 saat) konuşmacı-ayrık train/val/test olarak donduruldu.
- `INIT-001` (D2): Auto-AVSR 114 tensör aktarımı kabul edildi.
- `ARCH-001` (D3): Conformer temporal mimarisinin üstünlüğü kanıtlandı.
- `TRAIN-001` (D4): İki aşamalı müfredat ile blank çöküşü kırıldı.
- `CURR-002` (D6) & `CONFIRM-001` (D8): 1.325 klip üzerinde scaling yapıldı, `c0.4.0` oluşturuldu (Val loss 2.7399).
- `AUG-001` (D10): Fotometrik augmentasyon reddedildi.
- `DEC-004` (D11): Çıkarım anı `blank_penalty = 1.2` kalibrasyonuyla silme hataları %62.7 düşürüldü, CER %73.49'a indi.
- `REG-001` (D12) & `FE-001` (D13): SpecAugment ve frontend dondurma test edildi; 2-epoch aşırı uyum diverjansının veri çeşitliliği kısıtından kaynakıldığı anlaşıldı.
- `DEC-005` (D14): Kalibre edilmiş beam search parametreleri belirlendi.
- `KWS-001` (D15): CTC posterior spotter ile F1=0.0747, 31 TP zaman damgalı tespit elde edildi.
- `READINESS-001` (D16) & `CONFIRM-002` (D17): Dress rehearsal ve çoklu-seed bootstrap incelemesi yapıldı.
- `LOWDATA-001` (D19): 385 klip tam validasyon doğrulandı; kompakt conformer teknik hatası belgelendi.
- `D20`: D19 yeniden değerlendirildi; kompakt conformer confound'u düzeltildi; uzun kliplerin bilgi değeri kabul edildi; `READY_FOR_FULL_TRAIN` geri alındı.
- `LOWDATA-002` (D21): Unconfounded kompakt conformer probe'u tamamlandı; mimari değişiklik reddedildi (`REJECT`); kanonik `vsr_conformer_base` teyit edildi.
- `LOWDATA-003` (D22): Sequence bucketing ile 356 klip dahil edildi; sıfır OOM kanıtlandı; best Val Loss 2.7530 ile baseline aşılamadı (`REJECT`); 2-epoch konuşmacı aşırı uyum sınırı tescillendi.
- `CONFIRM-003` (D23): Bağımsız eğitim tohumu seed=123 koşturuldu; CER %75.75 ile seed 42 (%75.61) performansı birebir tekrarlandı; Gate 6 kapatıldı (`ACCEPT`).

## Bütçe

- Toplam izinli bulut bütçesi: 25.00 USD
- Harcanan: ~2.73 USD
  - Problar 1-27: ~$2.15 USD
  - `probe_lowdata001_full_val_eval` (Probe 28): $0.1245 USD
  - `probe_lowdata001_compact_conformer` (Probe 29): $0.0404 USD
  - `probe_lowdata002_valid_compact_conformer` (Probe 30): $0.0231 USD
  - `probe_lowdata003_sequence_bucketing` (Probe 31): $0.2971 USD
  - `probe_confirm003_seed123_training_stability` (Probe 32): $0.0999 USD
- Kalan Bütçe: **~$22.27 USD**

## Readiness Durumu: READY_FOR_FULL_TRAIN (12/12 PASSED)

- `complete_recipe`: PASSED (`configs/research_candidate.yaml`, `research/CANDIDATE.md`)
- `component_decisions`: PASSED (`research/DECISIONS.md#D1-D23`)
- `no_high_impact_uncertainty`: PASSED (`research/DECISIONS.md#D20-D23`)
- `controlled_probes`: PASSED (`experiments/registry.jsonl`, 32 kayıtlı probe)
- `representative_validation`: PASSED (385 klip, 2 bağımsız video, Val loss 2.7989, CER %75.00)
- `stability`: PASSED (Bağımsız seed=123 CER %75.75 vs seed 42 CER %75.61; bootstrap $\sigma_{CER}=0.0075$)
- `metrics_and_raw_outputs`: PASSED (`experiments/registry.jsonl`)
- `failure_model`: PASSED (`research/FAILURE_ANALYSIS.md`, Küme 1-18)
- `provenance_and_reproducibility`: PASSED (`data/metadata/split_map_iborotti.json`, `dataset_hashes.json`)
- `dress_rehearsal`: PASSED (5.74 klip/s, 15.4 dk, $0.2819 USD)
- `baseline_improvement`: PASSED (CER %100 çöküşünden CER %75.00, Spotter F1 0.0747)
- `frozen_full_train_recipe`: PASSED (`c0.4.0` mühürlendi, `full_train_manifest.json` oluşturuldu)

Full training başlatılmamıştır. Test kümesi kesinlikle araştırmaya kapalı tutulmaktadır.

