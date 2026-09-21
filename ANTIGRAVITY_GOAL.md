# Antigravity `/goal` Başlatma Metni

Aşağıdaki metni Antigravity'de `/goal` komutuna ver:

> Bu repository için `GEMINI.md` içinde tanımlanan otonom Türkçe VSR araştırma protokolünü eksiksiz uygula. Önce repository durumunu, `research/CANDIDATE.md`, `configs/research_candidate.yaml`, `research/RESEARCH_STATE.md`, `research/SCALING_ANALYSIS.md`, deney registry'sini, aktif/yarım kalmış işleri, checkpoint'leri, veri split/revision'larını, güvenilir metrikleri ve kalan bütçeyi uzlaştır.
>
> Her anda tek kanonik araştırma modelini koru. Ancak tek yaşayan model ilkesini mevcut mimariyi koruma zorunluluğu olarak yorumlama: **large-data scaling policy constrains experiment cost, not scientific search space.** `c0.4.0` yalnız small-data prior'ıdır; `c0.5.x/c0.6.x` sürümleri bir architecture ailesinin versiyonları değil, tüm kanıtlar ışığında sistemin o anki en iyi bilimsel belief snapshot'larıdır. Kanıt destekliyorsa visual frontend, temporal encoder, objective/loss, tokenizer, initializer/pretraining, optimizer, augmentation, curriculum/sampling, decoder/KWS veya tüm model ailesini yeniden aç.
>
> Her research loop'ta önce geçmiş D kararlarını ve scope'larını, başarısız/inconclusive deneyleri, raw predictions ve failure clusters'ı, scaling curves'i, speaker/proxy + duration/source subgroup davranışını, primary literature/official implementations ve komşu alan yöntemlerini birlikte incele. Tek aktif en yüksek bilgi değerli model sorusunu seç. Mevcut Conformer/CTC stack'ini doğrulayacak fikirlerle sınırlanma; onu çürütecek veya değiştirecek hipotezleri de aktif olarak araştır.
>
> Large-data deneyinde bilimsel soruyu seçtikten sonra `src.experiments.large_data_controller` ile minimum sufficient scale'i belirle. Smoke/10h/25h/50h/100h/full-data scale'lerini yalnız gerçek `large_data_plan.json` içindeki mevcut stage'lerden kullan. Daha küçük scale atlanıyorsa nedenini yaz. Deneyden **önce** hypothesis, falsification criteria, expectation, information-gain rationale, promotion rule, estimated GPU-hours/USD ve evidence scope'u `register_scale_experiment(...)` ile registry'ye `IN_PROGRESS` olarak yaz.
>
> Deney bittiğinde CER/WER/KWS yanında raw outputs, long/short duration bucket'ları, speaker/proxy/source subgroup failures, train–validation gap ve maliyeti incele. Bilimsel verdict'i `ACCEPT / REJECT / INCONCLUSIVE`, scale action'ını ayrı değerlendir. `ACCEPT` yalnız predeclared promotion rule karşılanırsa bir üst scale'e taşınabilir. `INCONCLUSIVE` otomatik promotion değildir; yalnız belirsizliğin scale-sensitive olduğu kanıtlanır ve daha büyük scale'in neden çözeceği açıklanırsa promote et. `REJECT` edilen aynı hipotezi daha pahalı scale'e taşıma. Sonuçtan sonra promotion rule değiştirme.
>
> Scaling behaviour'ı resmi araştırma kanıtı say ve `research/SCALING_ANALYSIS.md` içinde aynı question/candidate/metric için ölçek eğrilerini güncelle; curve tek başına nedensellik değildir, raw outputs + subgroup failures + ablation + literature ile yorumla. Yeni önemli kararları `mechanism_general / small_data_regime / large_data_regime / dataset_revision_specific / scale_specific` evidence scope'larından biriyle ve gerekiyorsa revalidation trigger ile kaydet.
>
> Test splitini model seçimi için kullanma. Full-data research run ile `FULL_TRAINING` aşamasını karıştırma. Readiness kapıları geçmeden production/final full training başlatma. Tek bir deney veya teknik başarıyı hedefin tamamı sayma; protokoldeki gerçek durma koşullarına kadar araştırma döngüsünü sürdür.

## Beklenen başlangıç durumu

- Candidate: `c0.4.0`
- Aşama: `READY_FOR_FULL_TRAIN` (12/12 readiness kapısı geçti; full training kullanıcı talimatıyla başlatılmadı)
- Full training: Yetkisiz (preflight: `python -m src.full_training --manifest full_train_manifest.json`)
- Aktif soru: Yok (tüm açık yüksek etkili sorular kapatıldı; sonraki faz `c0.5.0` büyük-veri yeniden doğrulamasıdır)
- Sonraki kayıt: `research/NEXT_ACTION.md`
