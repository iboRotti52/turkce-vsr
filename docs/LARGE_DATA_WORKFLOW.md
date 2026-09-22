# Large-Data Agentic Research Workflow

Bu runbook, `src.experiments.large_data_workflow` CLI'sının kullanımını tanımlar.
CLI **training başlatmaz**; large-data araştırmasının state/registry/provenance,
budget ve scale-promotion lifecycle'ını fail-closed yönetir.

> **Scientific search space özgürdür. Workflow yalnız experiment execution
> provenance, scale, promotion ve maliyet governance uygular.**

## Önkoşullar

1. HF snapshot immutable revision ile pinlenmiş olmalı.
2. `python -m src.data.prepare_large_data ...` ile
   `research/large_data_plan.json` üretilmiş ve audit edilmiş olmalı.
3. Yaşayan large-data candidate (`c0.5.x`) gerçekten açılmış olmalı;
   `configs/research_candidate.yaml` içindeki `candidate_version` spec ile eşleşmeli.
4. Scientific run register edilirken git working tree **clean** olmalı.
5. Initializer content identity bilinmeli: local file hash veya explicit SHA-256.

## 1. Durumu uzlaştır

```bash
python -m src.experiments.large_data_workflow \
  --budget-usd 25 \
  status
```

Çıktı şunları gösterir:

- dataset revision,
- mevcut research scales,
- toplam / harcanmış / aktif run için rezerve / kalan budget,
- varsa tek aktif experiment, question, scale ve candidate.

Registry bozuksa, birden fazla active large-data run varsa veya aktif run'ın
pre-result contract hash'i uyuşmuyorsa komut fail eder.

## 2. Yeni scientific run'ı pre-register et

Örnek dosyayı kopyala:

```bash
cp configs/large_data_register.example.yaml /tmp/register.yaml
```

Hipotezi, falsification criteria'yı, minimum sufficient scale'i, promotion rule'u,
budget tahminini ve initializer bilgilerini doldur. Sonra:

```bash
python -m src.experiments.large_data_workflow \
  --budget-usd 25 \
  register --spec /tmp/register.yaml
```

CLI otomatik olarak:

- clean git HEAD SHA'yı,
- candidate recipe SHA-256'yı,
- initializer SHA-256'yı,
- HF dataset / split / stage hash'lerini

pre-result setup'a bağlar ve registry'ye mühürlü `IN_PROGRESS` record yazar.

**Bu komut GPU job başlatmaz.** Bundan sonra ilgili agent/training runner yalnız
registry'deki exact experiment contract'a göre koşmalıdır.

## 3. Koşu bilimsel olarak tamamlandıysa

```bash
python -m src.experiments.large_data_workflow \
  --budget-usd 25 \
  complete --spec configs/large_data_complete.example.yaml
```

`complete`:

- actual GPU-hours/USD,
- raw/metric/failure evidence refs,
- scientific verdict,
- updated belief,
- revalidation trigger

kaydeder. Teknik başarı ile scientific verdict ayrı tutulur.

## 4. Koşu teknik olarak çöktüyse

```bash
python -m src.experiments.large_data_workflow \
  --budget-usd 25 \
  fail --spec configs/large_data_fail.example.yaml
```

Bu, run'ı `technical_status=ERROR` olarak kapatır ve bilimsel verdict uydurmaz.

## 5. Bir üst scale'e promote et

Source run tamamlanmış olmalı ve predeclared promotion rule karşılanmış olmalı:

```bash
python -m src.experiments.large_data_workflow \
  --budget-usd 25 \
  promote --spec configs/large_data_promote.example.yaml
```

Workflow:

- source registry record'ını yeniden doğrular,
- aynı HF revision/split/source-stage olduğunu kanıtlar,
- budget kontrolü yapar,
- parent'ı `PROMOTE_SCALE` yapar,
- child'ı exact target-stage provenance ile `IN_PROGRESS` açar,
- child'ın bir sonraki promotion rule'unu **child sonucu görülmeden önce** sabitler.

## Tek aktif experiment kuralı

Workflow aynı anda yalnız bir `IN_PROGRESS` large-data experiment kabul eder.
Bu, iki ayrı agent'in aynı budgetı paralel olarak tüketmesini veya tek yaşayan model
araştırmasını iki bağımsız kola ayırmasını engeller.

## Budget muhasebesi

Kalan kullanılabilir budget:

```text
total budget
- terminal large-data run actual cost
- active run reserved estimated cost
```

olarak hesaplanır. Böylece aktif run henüz fatura kesinleşmeden ikinci bir run
aynı parayı kullanamaz.

## Git ve research memory

Registry/state değişiklikleri araştırma belleğinin parçasıdır. Bir experiment
tamamlandıktan ve analysis belgeleri güncellendikten sonra bunları commit etmeden
yeni **bağımsız** scientific run register etmeye çalışma; yeni registration clean
git tree ister.

Scale promotion aynı scientific lineage'ı koruduğu için parent setup provenance'ını
child'a taşır ve registry parent→child güncellemesini atomik yapar.

## Bu workflow ne yapmaz?

- training/GPU launch etmez,
- model mimarisini seçmez,
- Conformer/CTC'yi zorunlu kılmaz,
- c0.4.0 small-data kararlarını global truth saymaz,
- test splitini research selection'a açmaz,
- full-data research probe'u final `FULL_TRAINING` ile eşitlemez.
