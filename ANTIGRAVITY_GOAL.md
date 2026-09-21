# Antigravity `/goal` Başlatma Metni

Aşağıdaki metni Antigravity'de `/goal` komutuna ver:

> Bu repository için `GEMINI.md` içinde tanımlanan otonom Türkçe VSR araştırma protokolünü eksiksiz uygula. Önce repository durumunu, `research/CANDIDATE.md` ile `configs/research_candidate.yaml` tutarlılığını, aktif veya yarım kalmış işleri, checkpoint'leri, deney registry'sini, veri split'lerini, güvenilir metrikleri ve kalan bütçeyi uzlaştır. Mevcut korpusun yalnız birkaç saat ve az sayıda konuşmacı içerdiğini bütün araştırma iddialarında hesaba kat; küçük-veri bulgularını gelecekteki daha büyük veri rejimine otomatik olarak genelleme.
>
> Her anda tek kanonik araştırma modelini koru. Kısa eğitimleri, ablation'ları ve karşılaştırma kollarını ayrı pilot modeller değil, bu modeldeki tek bir belirsizliği çözen geçici probe'lar olarak kullan. Mevcut hata kümelerini ve ham çıktıları incele; birincil VSR, video, ASR, self-supervised veya audio-visual learning, multilingual transfer ve komşu alan kaynaklarından, ayrıca Türkçenin fonem/visem, karakter, morfoloji ve dil modeli özelliklerinden reçeteyi değiştirebilecek hipotezler çıkar. Mevcut stack ile sınırlanma.
>
> Her döngüde tek aktif yüksek bilgi değerli soruyu seç. Deneyden önce beklenti, yanlışlama ölçütü, karar kuralı, Türkçeye özgü gerekçe ve maliyeti yaz; en ucuz ayırt edici probe'u çalıştır; bitene kadar takip et; validation metrikleri yanında ham çıktıları ve hata kümelerini incele; `ACCEPT`, `REJECT` veya `INCONCLUSIVE` kararı ver. Küçük validation oynamaları için amaçsız hiperparametre varyasyonlarına sapma, fakat araştırmayı sabit deney sayısı veya yeni yasaklarla da daraltma.
>
> Önceki `READY_FOR_FULL_TRAIN` ilanını otomatik olarak doğru kabul etme. Bootstrap örneklemesini bağımsız eğitim seed'i, eğitimde görülmüş konuşmacıları held-out genelleme olarak sunma. Kullanılmayan uzun klipleri, gerçek istikrar kanıtını ve bütün held-out validation kapsamını en yüksek bilgi değerli diğer hipotezlerle birlikte değerlendir. Geçerli eski deneyleri tekrarlama; yalnız eksik kanıtı veya açık mekanizmayı hedefle.
>
> Mevcut veriden anlamlı yeni bilgi çıkarabilecek hipotez kalmadığında veya kalan temel sorunların daha fazla ya da daha çeşitli veri gerektirdiği kanıtlandığında mevcut veri için en güçlü reçeteyi dondur, veri miktarına duyarlı hangi kararların gelecekte yeniden açılacağını kaydet ve `READY_FOR_FULL_TRAIN` noktasında dur. Kullanıcının açık talimatı gereği full training'i başlatma ve test kümesini araştırma kararlarında kullanma.

## Beklenen başlangıç durumu

- Candidate: `c0.4.0`
- Aşama: `CONFIRMING`
- Full training: Yetkisiz
- Aktif soru: `LOWDATA-001`
- Sonraki kayıt: `research/NEXT_ACTION.md`
