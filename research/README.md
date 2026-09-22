# Research System

Bu klasör artık araştırmayı dört ayrı katmanda tutar. Amaç, "deney yaptım → tek bir
markdown cümlesiyle modeli değiştirdim" türü agent drift'ini engellemektir.

## 1. Immutable evidence

- `experiments/registry.jsonl`: deney/probe kayıtları.
- experiment artifact'ları: metrik, raw prediction, failure örnekleri, checkpoint provenance.
- Eski D kararları tarihsel kanıttır; silinmez veya yeniden yazılmaz.

Bu katman "ne oldu?" sorusunu cevaplar.

## Technical operations and snapshot audits

Bilimsel round olmayan ama execution/data readiness'i kanıtlayan kayıtlar ayrı tutulur:

- `research/snapshot_audits/<hf-revision>/`: immutable dataset readiness/audit kanıtı.
- `research/operations/`: teknik rehearsal ve operasyon raporları.
- `artifacts/operations/`: machine-readable technical outputs.

**Technical rehearsal bilimsel verdict üretmez ve BELIEFS/CANDIDATE değiştiremez.**
Örneğin OPS-LD-001 gerçek Modal A10 üzerinde data→loader→model→CTC→validation
hattını doğrular, fakat random-init 20-step sonucu model seçimi için kullanılmaz.

## 2. Research rounds

`research/rounds/<round-id>/` tek bir bilimsel düşünme döngüsünün snapshot'ıdır:

- `round.yaml`: soru, hipotez, falsification, evidence input'ları, scope, verdict.
- `result.json`: reproducible/machine-readable analiz sonucu.
- `REPORT.md`: insan tarafından okunabilir yorum.

Bir round, model training olmak zorunda değildir. Retrospective evidence synthesis,
failure-cluster analizi veya scaling audit de round olabilir; fakat hangi immutable
kanıtlara dayandığı açık olmalıdır.

Bu katman "bu kanıtlardan bu turda ne öğrendik?" sorusunu cevaplar.

## 3. Mutable belief ledger

`research/BELIEFS.yaml` projenin **şu an** doğru olduğuna inandığı bilimsel
ifadeleri tutar.

Her active belief:

- evidence scope,
- evidence refs,
- implication,
- revalidation trigger

taşır.

Belief, decision ile aynı şey değildir. Decision tarihsel olarak ne yaptığımızı;
belief bugün yeni araştırmaya hangi önkabulle başladığımızı söyler.

Bu katman "şu an neye inanıyoruz ve ne zaman fikrimizi değiştirmeliyiz?" sorusunu cevaplar.

## 4. Living execution state

- `research/CANDIDATE.md`
- `configs/research_candidate.yaml`
- `research/NEXT_ACTION.md`
- `research/RESEARCH_STATE.md`

yalnız güncel yaşayan candidate ve bir sonraki eylemi temsil eder.

Bu katman "şimdi ne çalıştıracağız?" sorusunu cevaplar.

---

## Neden bu yapı?

c0.4 araştırmasında 32 model probe'u, D1-D23 kararları, failure analysis ve candidate
state aynı anda büyüdü. Bu çok kanıt üretti ama sonraki bir ajanın şu ayrımı yapmasını
zorlaştırdı:

1. ham sonuç,
2. o sonuçtan o gün çıkarılan karar,
3. bugün hâlâ geçerli belief,
4. sıradaki experiment.

META-001 turu özellikle bu problemi görünür yaptı: tek tek D8/D22 kayıtları doğruydu,
ancak 360→720→1325→long-clip zincirini birlikte okuyunca yeni ve daha güçlü bir
"optimization–recognition decoupling" sonucu çıktı.

Bu nedenle yeni yapı **kanıtı taşımıyor**, kanıtın üstüne yeni bir reasoning katmanı
ekliyor.

## Agent çalışma sırası

Yeni bir araştırma turunda:

1. immutable registry/artifact'ları uzlaştır,
2. mevcut `BELIEFS.yaml` ve revalidation trigger'larını oku,
3. tek aktif question seç,
4. gereken experiment/probe'u yap,
5. `research/rounds/` altında round snapshot'ı oluştur,
6. verdict'e göre belief ledger'ı güncelle,
7. gerekiyorsa D kararı ekle,
8. candidate ve `NEXT_ACTION.md`'ı en son güncelle.

## Geriye dönük uyumluluk

c0.4 dosyaları fiziksel olarak taşınmamıştır. Bu bilinçli:

- frozen hashes/provenance bozulmasın,
- eski agent ve script yolları çalışmaya devam etsin,
- migration kendi başına bilimsel değişiklik yaratmasın.

Yeni yapı c0.5 ve sonraki araştırmalar için preferred layout'tur.
