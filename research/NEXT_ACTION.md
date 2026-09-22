# Tek ve Kesin Sonraki Adım

**Candidate:** `c0.4.0` (frozen small-data prior)  
**Durum:** `READY_FOR_FULL_TRAIN` — large-data research henüz açılmadı  
**Son tamamlanan round:** `RND-2026-09-22-META-001` / D24  
**Aktif soru:** Yok; mevcut HF snapshot scientific gate'i geçmiyor  
**Queued large-data question:** `ARCH-LD-001` — **BLOCKED BY DATA REGIME**  
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

2026-09-22 audit'i current HF revision
`7ff10fb7cf98a6b6f98abc0cd790dd6cc191226a` için şu sonucu verdi:

- train: **545 clips / 0.6954h**
- val: **216 clips / 0.2848h**
- test: **125 clips / 0.2594h**
- identity: `channel` proxy
- scale stages: yalnız `full`; **10h yok**
- c0.5 scientific gate: **BLOCKED**

Aynı revision üzerinde OPS-LD-001 ile gerçek Modal A10 technical rehearsal başarılı
oldu; execution path artık teknik olarak doğrulanmış durumda. Bu, c0.5'i açma yetkisi
vermez.

### Tek kesin sonraki adım

**Upstream HF dataset'i yeni bir immutable revision'a taşı:** accepted preprocessed
veriyi en az 10h train stage oluşturacak kadar büyüt (hedef 50–100h) ve manifestte
mümkünse gerçek `speaker_id` / `speaker` alanı sağla. Yalnız channel proxy varsa
insan speaker-leakage audit'i üret.

Yeni revision geldikten sonra:

1. GitHub Actions'taki **large-data snapshot audit and Modal rehearsal** workflow'unu
   önce audit-only olarak çalıştır.
2. Audit `scientific_c0_5_ready=true` vermeden canonical
   `research/large_data_plan.json` kurma.
3. Gate geçerse `c0.5.0 / RESEARCHING` aç ve queued `ARCH-LD-001` için 10h
   minimum-sufficient baseline'ı pre-register et.
4. Test splitini candidate selection'a açma.

Current blocked snapshot evidence:
`research/snapshot_audits/7ff10fb7cf98a6b6f98abc0cd790dd6cc191226a/`.


## Agentic large-data başlangıç adımı

HF snapshot planı audit edilip `c0.5.0 / RESEARCHING` açıldıktan sonra ilk pahalı
deneyi doğrudan çalıştırma. Önce:

1. Tüm mevcut kanıtı yeniden uzlaştır: D1–D23, failures, raw outputs, literature ve
   yeni dataset/scaling bilgisi.
2. En yüksek bilgi değerli **tek** model sorusunu seç; mevcut Conformer/CTC ailesiyle
   sınırlanma.
3. Önce `.venv/bin/python -m src.experiments.research_ops status` ile candidate,
   plan, registry ve yeni round/belief belleğini fail-closed uzlaştır.
4. Experiment spec'inde minimum sufficient scale, falsification criteria,
   information-gain rationale, predeclared promotion rule ve estimated GPU-hours/USD yaz.
5. `research_ops register` ile sonuç görülmeden exact provenance taşıyan
   `IN_PROGRESS` registry kaydını oluştur.
6. Sonucu raw outputs + subgroup failures ile incele; `research_ops complete` veya
   teknik hata varsa `research_ops fail` kullan.
7. Scale promotion gerekiyorsa `research_ops promote` kullan; düşük seviyeli
   controller API'sini normal agent akışında elle birleştirme.
8. Her tur sonunda round snapshot → BELIEFS → decision/candidate/NEXT_ACTION sırasıyla
   kalıcı belleği güncelle.

**Large-data scaling policy constrains experiment cost, not scientific search space.**
