# Large-Data Scaling Analysis

Bu dosya büyük-veri fazında **ölçek davranışını kalıcı araştırma kanıtı** olarak tutar.
Amaç yalnız "hangi model daha iyi?" sorusunu değil, modelin veri arttıkça **nasıl**
davrandığını öğrenmektir.

> Temel kural: **Large-data scaling policy constrains experiment cost, not scientific search space.**
> Bu dosya mevcut Conformer/CTC ailesini korumak için kullanılmaz. Yeni kanıt frontend,
> temporal model, objective, tokenizer, optimizer, curriculum, augmentation, decoder,
> KWS veya tüm model ailesini sorgulamayı gerektiriyorsa araştırma motoru bunu yapabilir.

## 1. Kanıt kaynakları

Her yorum tek bir metriğe dayanamaz. Bir scaling yorumu mümkün olduğunda birlikte kullanır:

- aynı research question + aynı candidate sürümündeki 10h/25h/50h/100h/full-data sonuçları,
- CER/WER/KWS metrikleri,
- train/validation eğrileri,
- raw prediction karşılaştırmaları,
- speaker/proxy-group, duration ve source/channel alt grup hataları,
- failure cluster değişimleri,
- önceki D kararları ve başarısız deneyler,
- primary literature / official implementations / adjacent methods.

Scaling curve **nedensellik kanıtı değildir**; hangi mekanizmanın değiştiğini söylemek için
raw outputs, failure analysis ve kontrollü ablation gerekir.

## 2. Evidence scope

Yeni büyük-veri bulgularını aşağıdaki scope'lardan biriyle kaydet:

- `mechanism_general`: veri ölçeğinden bağımsız olduğuna güçlü mekanistik kanıt var.
- `small_data_regime`: yalnız frozen küçük-veri rejiminde doğrulandı.
- `large_data_regime`: büyük-veri rejiminde doğrulandı ancak dataset-revision bağımsızlığı henüz kanıtlanmadı.
- `dataset_revision_specific`: yalnız belirli HF snapshot'ında kanıtlandı.
- `scale_specific`: yalnız belirli 10h/25h/50h/... ölçekte görüldü.

`mechanism_general` dışındaki her yeni önemli bulgu için **revalidation trigger** yaz.
Eski small-data bulgusunu otomatik evrensel kural yapma; yeni large-data bulgusunu da aynı
şekilde gelecekteki tüm datasetlere genelleme.

## 3. Curve kayıt formatı

Her aktif question için aşağıdaki tabloyu kullan:

| Question | Candidate | HF revision | Scale | Train h | Seed | Metric | Value | Cost USD | Evidence ref |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| _henüz yok_ | | | | | | | | | |

Aynı question/candidate için mümkünse şu trendleri ayrıca izle:

- WER / CER
- validation loss
- train–validation gap
- blank ratio / collapse davranışı
- long-utterance bucket
- short-utterance bucket
- speaker/proxy-group dağılımı
- KWS precision / recall / F1
- wall-clock ve GPU-saat / maliyet

## 4. Scaling davranışından çıkarılabilecek araştırma sinyalleri

Bunlar otomatik karar değil, **hipotez üretme sinyalleridir**:

- Veri arttıkça WER güçlü biçimde düşmeye devam ediyorsa model hâlâ data-limited olabilir.
- Train iyileşirken validation plato yapıyorsa generalization, regularization veya mimari darboğaz araştır.
- Kısa klipler iyileşirken uzun klipler sabit kalıyorsa temporal modeling / alignment sorusunu aç.
- Belirli speaker/proxy grupları sürekli geride kalıyorsa visual frontend, domain variation veya veri temsilini incele.
- CTC blank collapse daha büyük ölçekte geri dönüyorsa objective/training dynamics kararlarını yeniden aç.
- İki yöntem küçük scale'de ayrışmıyor ama daha büyük scale'de ayrışıyorsa bu davranışı gelecek
  minimum-sufficient-scale kararlarında kanıt olarak kullan.
- Kazanç yalnız decoder/KWS tuning'den geliyorsa bunu encoder/mimari iyileşmesi gibi kaydetme.

## 5. Promotion günlüğü

Her scale promotion şu bilgileri taşır:

- source experiment id,
- from_scale → to_scale,
- bilimsel verdict: `ACCEPT / REJECT / INCONCLUSIVE`,
- deney başlamadan önce yazılmış promotion rule,
- rule karşılandı mı,
- kullanılan evidence refs,
- estimated/actual GPU-hours ve USD,
- intermediate scale atlandıysa neden,
- `INCONCLUSIVE` ise neden belirsizliğin **scale-sensitive** olduğu.

`INCONCLUSIVE` tek başına daha fazla GPU harcama gerekçesi değildir.

## 6. Canonical-model yorumu

`c0.4.0 → c0.5.x → c0.6.x` sürümleri bir architecture ailesinin sürümleri değildir.
Her candidate, o anda tüm kanıtlar ışığında sistemin **en iyi bilimsel inancının snapshot'ıdır**.

Bu nedenle yeterli kanıt oluşursa yeni candidate:

- farklı visual frontend,
- farklı temporal encoder,
- farklı objective,
- farklı tokenizer/output representation,
- farklı optimizer/curriculum/augmentation,
- farklı decoder/KWS

kullanabilir. Tek yaşayan model ilkesi yalnız unutulmuş paralel model dallarını engeller;
bilimsel arama alanını daraltmaz.

## 7. Güncel durum

Large-data HF snapshot henüz bu dosyada analiz edilmedi. İlk gerçek plan
`src.data.prepare_large_data` ile üretilip audit edildikten sonra `c0.5.0 / RESEARCHING`
açılır ve ilk scaling kayıtları buraya eklenir.
