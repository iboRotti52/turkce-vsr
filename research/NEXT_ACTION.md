# Tek ve Kesin Sonraki Adım

**Candidate:** `c0.4.0` (frozen small-data prior)  
**Durum:** `READY_FOR_FULL_TRAIN` — large-data research henüz açılmadı  
**Son tamamlanan round:** `RND-2026-09-22-META-001` / D24  
**Aktif soru:** Yok; large-data plan audit edilene kadar yeni model run açma  
**Queued large-data question:** `ARCH-LD-001`  
**Full training yetkisi:** Donduruldu / Beklemede; GEMINI.md protokolü ve kullanıcının açık talimatı ("sen full training e hazır hale gelene kadar yani bu kadar az veriden modelin nasıl olması gerektiğiyle ilgili öğrenebilecek her şeyi öğrenmeye çalış bu veri setine özel optimizasyona girme") gereği 12 kapı kanıtlarla geçildi, reçete donduruldu ve full training başlatılmadan duruldu.

## Mevcut Durum

1. `probe_confirm003_seed123_training_stability` (D23): Bağımsız eğitim tohumu `seed=123` koşturuldu; 2. epoch'ta CER %75.75 (seed 42: %75.61, delta $+0.14\%$), Val Loss 2.8150 ve Blank %81.66 elde edildi. 2-epoch optimumu ve yakınsama istikrarı tohumdan bağımsız kanıtlandı ve Gate 6 kapatıldı (`ACCEPT`).
2. GEMINI.md Bölüm 5.1'deki 12 hazırlık kapısının tamamı eksiksiz geçti (`READY_FOR_FULL_TRAIN`).
3. Kanonik reçete `configs/research_candidate.yaml` içinde `recipe_status: frozen` olarak mühürlendi ve SHA-256 imzalı `full_train_manifest.json` oluşturuldu.
4. Test kümesi (617 klip, 3 held-out konuşmacı/kanal) kesinlikle karantinada tutuldu ve hiçbir araştırmada kullanılmadı.
5. Kullanıcı talimatı ve protokol gereği full training başlatılmadı.

## Kullanıcı Full Training Yetkisi Verdiğinde Çalıştırılacak Komut

Tek kanonik entrypoint, preflight-only ve fail-closed'dur (ücretli işlem
başlatmaz; `--manifest` bayraklı eski çağrı hiçbir zaman var olmadı):

```bash
# Full-training preflight (ücretli işlem YOK; tarihsel c0.4.0 manifestosu
# bilerek FAIL verir — kanıt olarak korunur, çalıştırılamaz):
.venv/bin/python -m src.full_training --manifest full_train_manifest.json

# Eşdeğer programatik yol:
# .venv/bin/python run_experiment.py --mode full-train --full-train-manifest full_train_manifest.json
```

Preflight geçse bile eğitim otomatik başlamaz; GPU lansmanı ayrı, açık insan
kararıyla yapılır.







## META-001 sonrası araştırma yönü

Yeni meta-probe, frozen c0.4 rejiminde 360 → 720 → 1,325 clip scaling boyunca
validation loss'un **%3.82 iyileşmesine rağmen CER'in +0.69 yüzde puan kötüleştiğini**
gösterdi. 385-klip validation üzerinde long-clip extension da loss'u **%1.64**
iyileştirirken CER'i **+0.07 pp** kötüleştirdi.

Bu yüzden bir sonraki bilimsel adım "aynı küçük-data rejiminde biraz daha klip/tuning"
değildir.

Gerçek HF snapshot audit edilip c0.5 açıldığında ilk yüksek-değerli soru:

> **ARCH-LD-001:** Daha çeşitli large-data rejiminde canonical
> Conformer + plain CTC loss→recognition ilişkisini koruyor mu; yoksa aynı baseline
> kontrolü altında düşük-confound bir sequence-objective/temporal-model değişikliği
> gerçek CER/WER kazanımı üretiyor mu?

İlk large-data round tasarım ilkesi:

1. c0.4 recipe'yi **control prior** olarak 10h minimum-sufficient scale'de yeniden ölç.
2. Önce baseline'da META-001 loss→CER ayrışmasının sürüp sürmediğini kontrol et.
3. Ayrışma sürüyorsa ilk treatment mümkün olduğunca tek ana değişkenli olsun.
   En düşük-confound adaylardan biri aynı Conformer üzerinde intermediate-CTC/residual
   conditioning'dir; hybrid RNN-T, phoneme+LM veya yeni frontend daha büyük sonraki
   dallardır.
4. Treatment seçimi literature context'ten gelir ama **kabul kararı yalnız Türkçe
   c0.5 evidence'ıyla** verilir.

## Large-data fazına geçişte sonraki gerçek adım

`c0.4.0` historical small-data sonucu değişmeden kalır. Yeni veri gerçekten
50–100 saat ölçeğine ulaştığında ilk adım eğitim başlatmak değil, HF snapshot'ını
immutable revision ile sabitleyip yeni identity-group-disjoint planı üretmektir (`channel`
yalnız speaker proxy ise bunu ayrıca audit et):

```bash
.venv/bin/python -m src.data.prepare_large_data \
  --repo-id avsr-tr-ekip/avsr-tr-dataset \
  --revision <40-hex-HF-dataset-commit-sha>
```

Çıktıdaki speaker/süre dağılımını ve leakage kontrollerini incele. Plan kabul edilince
yeni yaşayan candidate `c0.5.0 / RESEARCHING` aç; `c0.4.0` mimari/reçetesini
başlangıç prior'ı olarak taşı ancak veri-miktarına duyarlı kararları yeniden aç.
İlk probe'u mümkün olan en küçük speaker-diverse stage'de yap. Test splitini candidate
dondurulana kadar araştırma kararlarına açma.


## Agentic large-data başlangıç adımı

HF snapshot planı audit edilip `c0.5.0 / RESEARCHING` açıldıktan sonra ilk pahalı
deneyi doğrudan çalıştırma. Önce:

1. Tüm mevcut kanıtı yeniden uzlaştır: D1–D23, failures, raw outputs, literature ve
   yeni dataset/scaling bilgisi.
2. En yüksek bilgi değerli **tek** model sorusunu seç; mevcut Conformer/CTC ailesiyle
   sınırlanma.
3. `ScaleExperimentPlan` ile minimum sufficient scale, falsification criteria,
   information-gain rationale, predeclared promotion rule ve estimated GPU-hours/USD yaz.
4. `register_scale_experiment(...)` ile sonuç görülmeden `IN_PROGRESS` registry
   kaydını oluştur.
5. Sonucu raw outputs + subgroup failures ile incele; bilimsel verdict'i
   `ACCEPT/REJECT/INCONCLUSIVE`, scale action'ını ayrıca kaydet.
6. Scale promotion gerekiyorsa `validate_promotion_request(...)` ile source registry
   kaydındaki önceden yazılmış rule'a karşı doğrula.
7. Her scale sonucu sonrası `research/SCALING_ANALYSIS.md` dosyasını güncelle.

**Large-data scaling policy constrains experiment cost, not scientific search space.**
