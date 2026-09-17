# Tek Yaşayan Araştırma Modeli Tasarımı

**Tarih:** 2026-09-17

**Kapsam:** Türkçe görsel konuşma tanıma araştırma ve eğitim karar sistemi

**Amaç:** Çok sayıda pilot modeli yarıştırmak yerine, tek bir kanonik araştırma modelini kanıta dayalı olarak olgunlaştırmak ve yalnızca mimarisi ile eğitim reçetesi yeterince doğrulandıktan sonra tam eğitime geçirmek.

## 1. İstenen sonuç

Sistem, sessiz Türkçe videodan sürekli konuşmayı çözmek ve en sık kullanılan 500 Türkçe kelimeyi zaman damgalı biçimde tespit etmek için tek bir yaşayan araştırma modeli geliştirecek. Araştırma aşamasının çıktısı kısa süre eğitilmiş bir "pilot" checkpoint değil; bileşen seçimleri, eğitim reçetesi, veri stratejisi ve değerlendirme davranışı kanıtlarla olgunlaştırılmış bir **full-train-ready model tanımı** olacak.

Tam eğitim araştırma yöntemi değildir. Tam eğitim, araştırma modeli yeterli kanıtla dondurulduktan sonra uygulanan ayrı ve pahalı üretim aşamasıdır.

## 2. Temel ilkeler

### 2.1 Tek kanonik model

Her anda yalnızca bir kanonik araştırma modeli bulunur. Bu model yalnız ağırlıklardan ibaret değildir; aşağıdakilerin birlikte oluşturduğu sürümlü bir tasarımdır:

- ROI ve video ön işleme,
- görsel frontend ve initializer,
- zamansal model,
- eğitim hedefi ve tokenizer,
- loss ve regularization,
- curriculum ve örnekleme,
- optimizer, scheduler, freeze/unfreeze ve diğer eğitim ayarları,
- decoder ve Türkçe dil modeli,
- keyword spotting ve kalibrasyon,
- değerlendirme protokolü,
- hesaplama ve çıkarım sınırları.

Araştırma modeli için kanonik kayıt `research/CANDIDATE.md`, makine tarafından okunabilir reçete ise `configs/research_candidate.yaml` olacaktır. İki kayıt aynı `candidate_version` değerini taşır.

### 2.2 Deneyler model değil, probdur

Kısa koşular, ablation'lar ve karşılaştırma kolları ayrı model adayları değildir. Her biri kanonik modeldeki tek bir yüksek değerli belirsizliği çözmeye çalışan geçici bir **probe** olarak kaydedilir.

Her probe şu soruyla başlar:

> Kanonik araştırma modelinin hangi parçasını, hangi kanıt oluşursa neden değiştireceğiz?

Probe sonucu:

- kanıt yeterliyse kanonik modele bir değişiklik kabul edilir,
- kanıt olumsuzsa değişiklik reddedilir,
- kanıt yetersizse karar açık kalır ve daha ayırt edici en ucuz sonraki kontrol seçilir.

Probe checkpoint'leri varsayılan olarak yeni aday veya kalıcı geliştirme dalı sayılmaz. Karar verildikten sonra yalnız tekrarlanabilirlik için gereken artefaktlar saklanır.

### 2.3 Sınırsız fikir kaynağı, kanıtla sınırlı kabul

Araştırma önceden seçilmiş `3D-ResNet + BiGRU/Conformer + CTC` ailesine hapsedilmeyecek. Modelin her bileşeni için aşağıdakiler dahil fakat bunlarla sınırlı olmayan kaynaklardan fikir alınabilir:

- güncel ve klasik VSR/lip-reading çalışmaları,
- visual-only ve audio-visual foundation modelleri,
- video understanding ve action recognition mimarileri,
- ASR, speech representation learning ve sequence transduction,
- self-supervised, multilingual ve cross-lingual transfer,
- CNN, state-space, recurrent, attention ve hibrit temporal modeller,
- CTC, transducer, attention/seq2seq, segmental ve çok görevli hedefler,
- karakter, subword, fonem, visem ve hibrit çıktı uzayları,
- curriculum learning, active learning, distillation ve pseudo-labeling,
- dil modeli, rescoring, constrained decoding ve keyword spotting yaklaşımları,
- eski temel çalışmalar, negatif sonuçlar ve komşu alanların yöntemleri.

Bir fikrin popüler veya mevcut kodla kolay olması kabul sebebi değildir. Bir mimari ailesinin yeni, büyük veya mevcut altyapıya uzak olması da tek başına ret sebebi değildir. Eleme ancak ürün use case'i, veri rejimi, ölçülmüş davranış, uygulanabilir lisans/erişim, eğitim ve çıkarım maliyeti ya da kontrollü deney kanıtıyla yapılabilir.

Ürün girdisi sessiz video olarak kalır. Eğitim sırasında audio-visual veya başka öğretmen sinyalleri kullanılması araştırılabilir; üretim çıkarımında gerekli olmayan yan modaliteler kabul edilebilir.

## 3. Araştırma döngüsü

Araştırma sabit deney sayısı veya takvim süresiyle bitmez. Her döngü kanonik modeli geliştirir:

1. **Durumu uzlaştır:** Kanonik reçete, son güvenilir metrikler, açık kararlar, aktif işler, bütçe ve artefaktlar doğrulanır.
2. **Çıktılara bak:** Eğitim eğrileri yanında ham transkriptler, zaman damgaları ve en az 20 anlamlı örnek incelenir; hata kümeleri güncellenir.
3. **En büyük belirsizliği seç:** Mimari, veri veya training recipe içindeki en yüksek etkili ve hâlâ açık soru belirlenir.
4. **Geniş araştır:** Mevcut tercihleri doğrulayan kaynaklarla yetinmeden birincil kaynaklar, resmî uygulamalar, eski çalışmalar, negatif sonuçlar ve komşu alanlar incelenir.
5. **Tek değişiklikli probe tasarla:** Beklenti, yanlışlama ölçütü, maliyet, karar kuralı ve kanonik modele olası etkisi koşudan önce yazılır.
6. **En ucuz ayırt edici koşuyu yap:** Tiny overfit, kısa eğitim, küçük ama temsil gücü olan subset, offline feature analizi veya kontrollü ablation kullanılabilir.
7. **Sonucu analiz et:** CER/WER ve keyword metrikleriyle birlikte istikrar, blank davranışı, hata kümeleri, gecikme, maliyet ve ham çıktılar değerlendirilir.
8. **Tek modeli güncelle:** Kanıtlanan değişiklik kanonik reçeteye işlenir; sürüm artırılır; reddedilen değişiklik ve gerekçesi kaydedilir.
9. **Sonraki belirsizliğe geç:** Birkaç başarılı probe, ölçek büyütmek için gerekçe sayılmaz.

Bir probe teknik olarak çalıştı diye model gelişmiş kabul edilmez. Bir probe yalnız loss'u düşürdü diye mimari kararı oluşturmaz. Test split'i araştırma kararlarında kullanılmaz.

## 4. Kanonik modelin sürümlenmesi

`research/CANDIDATE.md` aşağıdaki bölümleri zorunlu taşır:

- `candidate_version`, durum ve son güncelleme,
- tam bileşen haritası,
- tam training recipe,
- her seçimin kanıt özeti,
- açık yüksek/orta/düşük etkili belirsizlikler,
- kabul edilen ve reddedilen son değişiklikler,
- mevcut güvenilir metrikler ve karşılaştırma kapsamı,
- veri, split, seed, kod ve initializer provenance bilgisi,
- full-training readiness tablosu,
- bir sonraki tek araştırma sorusu.

Kanonik model yalnızca aşağıdaki koşullarda sürüm artırır:

1. Değişiklik tek ve açık biçimde tarif edilmiştir.
2. Önceden yazılmış karar kuralı karşılanmıştır.
3. Aynı değerlendirme protokolünde mevcut kanonik sürümle karşılaştırılmıştır.
4. İyileşme yalnızca teknik başarıya veya tek loss değerine dayanmamaktadır.
5. Regresyonlar ve ham örnekler incelenmiştir.
6. Sonuç ve belirsizlikler kalıcı araştırma belleğine yazılmıştır.

## 5. Araştırma ve doğrulama ölçekleri

Ölçekler farklı modeller üretmez; aynı kanonik model hakkında farklı güçte kanıt üretir.

### 5.1 Diagnostic probe

Kod yolu, veri geçerliliği, gradyan akışı, kapasite veya belirli bir mekanizmayı en ucuz şekilde test eder. Model kalitesi iddiası üretemez.

### 5.2 Research probe

Kanonik modeldeki tek bir değişikliği sabit subset, split, seed ve hesaplama bütçesi altında kontrol koluyla karşılaştırır. Sonuç umut verici olabilir fakat full training kararı oluşturamaz.

### 5.3 Confirmation run

Kabul edilmek üzere olan yüksek etkili değişikliği ek seed, daha temsil edici veri dilimi ve ayrıntılı hata analiziyle doğrular. Her küçük ayar confirmation gerektirmez; mimariyi, hedefi, initializer'ı, curriculum'u veya ana training recipe'yi değiştiren kararlar gerektirir.

### 5.4 Dress rehearsal

Araştırma kararları büyük ölçüde kapanınca, dondurulmuş aday reçetesi daha büyük fakat full training olmayan ölçekte uçtan uca çalıştırılır. Amaç yeni fikir aramak değil; yakınsama, bellek, süre, checkpoint/resume, değerlendirme ve maliyet varsayımlarını sınamaktır. Rehearsal sırasında yalnız çalışmayı engelleyen hata veya kanıtlanmış ciddi regresyon araştırma aşamasını yeniden açabilir.

## 6. Full-training readiness kapısı

Agent aşağıdaki koşulların tamamı kanıtlanmadan full training başlatamaz veya bir sonraki adım olarak öneremez:

1. Kanonik mimari ve training recipe eksiksiz ve makine tarafından okunabilir biçimde kaydedilmiştir.
2. Frontend/initializer, temporal model, hedef/tokenizer, loss, curriculum/veri örnekleme, optimizer/scheduler ve decoder/KWS için seçim veya bilinçli erteleme gerekçesi vardır.
3. Açık **yüksek etkili** mimari veya eğitim belirsizliği kalmamıştır.
4. Kritik kararlar kontrollü probe ile sınanmış; yüksek etkili kararlar confirmation run ile doğrulanmıştır.
5. Sonuç yalnız aynı küçük subset'in ezberlenmesine dayanmamış; konuşmacı ayrık ve temsil gücü daha yüksek validation kapsamına taşınmıştır.
6. Son ana aday birden fazla seed veya eşdeğer istikrar kontrolünde yönünü korumuştur.
7. CER/WER, Top-500 precision/recall/F1, FP/FN, blank davranışı ve en az 20 ham örnek birlikte incelenmiştir.
8. En büyük hata kümeleri bilinmekte; bunların full training ile azalacağına dair mekanizma veya kabul gerekçesi yazılmıştır.
9. Veri kalite, split sızıntısı, checkpoint provenance ve tekrar üretilebilirlik kontrolleri geçmiştir.
10. Dress rehearsal başarıyla tamamlanmış; tahminî full-training süre ve maliyeti ölçülmüştür.
11. Dondurulmuş aday, önceki güvenilir baseline'a karşı anlamlı ve yalnız decoder hilesine dayanmayan umut göstermiştir.
12. `full_train_manifest` oluşturulmuş; candidate sürümü, kod revizyonu, veri/split hash'leri, seed'ler, hiperparametreler, initializer ve bütçe sabitlenmiştir.

Koşullardan biri karşılanmıyorsa durum `RESEARCHING` veya `CONFIRMING` kalır. Başarılı birkaç deney, düşen loss, çalışan kod, kalan bütçe ya da araştırma yorgunluğu bu kapıyı geçirmez.

## 7. Durum makinesi ve korumalar

Araştırma durumu yalnız şu sırada ilerler:

`RESEARCHING → CONFIRMING → REHEARSAL → READY_FOR_FULL_TRAIN → FULL_TRAINING → EVALUATING`

- `RESEARCHING`: kanonik model değişebilir; ucuz problar yürütülür.
- `CONFIRMING`: yüksek etkili son kararlar daha güçlü kanıtla doğrulanır.
- `REHEARSAL`: reçete dondurulmuştur; ölçek ve operasyon doğrulanır.
- `READY_FOR_FULL_TRAIN`: bütün readiness kanıtları ve manifest mevcuttur.
- `FULL_TRAINING`: yalnız dondurulmuş manifest çalıştırılır.
- `EVALUATING`: full-trained checkpoint test setinde ve ürün use case'inde değerlendirilir.

Kod tarafındaki full-training komutu `READY_FOR_FULL_TRAIN` durumu ve geçerli manifest olmadan fail-closed davranır. Araştırma komutları full dataset veya full-training bütçesini yanlışlıkla kullanamaz. Bir research probe tamamlandığında varsayılan `next_step`, ölçek büyütmek değil sonuç analizi ve kanonik model güncellemesidir.

## 8. Kayıtlar ve veri akışı

Her probe registry kaydında şunlar bulunur:

- bağlı olduğu `candidate_version`,
- araştırma sorusu ve değiştirilen tek faktör,
- kontrol ve treatment tanımı,
- beklenti, yanlışlama ve kabul ölçütü,
- kullanılan ölçek türü,
- veri/split/sample/seed/code/initializer provenance,
- tahminî ve gerçekleşen maliyet,
- metrikler ve ham çıktı incelemesinin yeri,
- karar: `ACCEPT`, `REJECT`, `INCONCLUSIVE`,
- kanonik modele uygulanan değişiklik veya neden uygulanmadığı.

Akış şu şekildedir:

`CANDIDATE + açık belirsizlik → kaynak araştırması → probe → hata analizi → karar → CANDIDATE'ın yeni sürümü`

Full training akışı ise ayrıdır:

`dondurulmuş CANDIDATE → readiness doğrulaması → full_train_manifest → tek pahalı eğitim → bağımsız test değerlendirmesi`

## 9. Mevcut repository'ye uygulanacak değişiklikler

- `GEMINI.md`: tek yaşayan model, probe kavramı, sınırsız kaynak taraması ve readiness kapısı ana protokole eklenecek.
- `research_plan.md`: numaralı pilotlardan nihai modele giden sabit yol haritası kaldırılacak; yaşayan model döngüsüyle değiştirilecek.
- `research/ARCHITECTURE.md`: erken kapanmış tercihler yerine mevcut kanonik seçim, alternatif kanıtı ve açık belirsizlikleri ayıracak.
- `research/RESEARCH_STATE.md`: durum makinesi, candidate sürümü ve readiness özeti eklenecek.
- `research/NEXT_ACTION.md`: iki kalıcı aday üretmek yerine kanonik modeldeki tek initializer belirsizliğini çözen geçici probe olarak yazılacak.
- `research/CANDIDATE.md`: tek yaşayan modelin insan tarafından okunabilir kaydı oluşturulacak.
- `configs/research_candidate.yaml`: aynı modelin çalıştırılabilir reçetesi oluşturulacak.
- `src/experiments/`: durum geçişi, manifest doğrulaması ve full-training fail-closed korumaları eklenecek.
- `run_experiment.py`: belirsiz `micro-pilot` / `modal-pilot` anlamları probe ölçekleriyle netleştirilecek; başarılı probe sonrası otomatik ölçek önerisi kaldırılacak; full training yalnız readiness kapısıyla açılacak.
- `tests/`: readiness eksikken full training'in reddedildiğini, geçersiz durum geçişlerini ve candidate/probe provenance zorunluluklarını doğrulayacak.

## 10. Hata yönetimi ve güvenlik

- Teknik hata araştırma hipotezini yanlışlamaz; `ERROR` olarak ayrı kaydedilir ve kök neden analizi yapılır.
- Eksik split, provenance, candidate sürümü veya acceptance rule ile ücretli probe başlatılmaz.
- Aynı anda kanonik modele birden fazla temel değişiklik kabul edilmez; etkileşim gerekiyorsa açıkça birleşik hipotez olarak gerekçelendirilir.
- Test split'i model veya training recipe seçmek için kullanılmaz; yalnız dondurulmuş aday ve full-trained modelin bağımsız değerlendirmesinde kullanılır.
- Toplam ücretli bütçe mevcut kullanıcı sınırına uyar. "Her şeyden ilham" mimari fikir özgürlüğüdür; erişim, lisans, güvenlik ve bütçe kontrollerini kaldırmaz.
- Kullanıcıya ait mevcut değişiklikler ve eski deney kayıtları korunur; tarih yeniden yazılmaz.

## 11. Başarı ölçütü

Bu değişiklik başarılıdır, eğer:

- agent birkaç başarılı probe sonrasında doğrudan büyük eğitime geçemiyorsa,
- her deney açıkça tek kanonik modeldeki bir karara bağlanıyorsa,
- alternatif mimari ve eğitim fikirleri mevcut stack ile sınırlanmadan araştırılabiliyorsa,
- kabul edilen her değişiklik kanıtla kanonik reçeteye işleniyorsa,
- full training yalnız dondurulmuş ve doğrulanmış manifest üzerinden başlatılabiliyorsa,
- araştırmanın çıktısı çok sayıda pilot checkpoint değil, tek bir olgun full-train-ready model tanımı oluyorsa.
