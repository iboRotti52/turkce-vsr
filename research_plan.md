# Türkçe VSR Temiz Başlangıç Araştırma Planı

## Hedef

Sessiz videodan Türkçe sürekli konuşmayı çözmek ve genel Top-500 kelimeyi zaman damgalı tespit etmek için tek bir kanonik modeli, veri ve mimari varsayımlarını baştan kanıtlayarak geliştirmek.

## Başlangıç ilkesi

Eski model, checkpoint, deney sonucu, failure analizi veya mimari kararı devralınmaz. `data/iborotti/` yalnız ham kaynak adayıdır; uygunluğu `DATA-001` ile kanıtlanmadan eğitim verisi sayılmaz.

## Araştırma sırası

```text
DATA-001: veri ve evaluation foundation
  → ARCH-001: geniş kaynak taraması ve tek başlangıç blueprint'i
  → tek aktif belirsizlik için en ucuz probe
  → ham çıktı ve hata analizi
  → ACCEPT / REJECT / INCONCLUSIVE
  → tek candidate'ı güncelle
  → confirmation
  → dress rehearsal
  → readiness tamamlanırsa frozen full training
  → bağımsız test değerlendirmesi
```

## Mimari özgürlük

Preprocessing, frontend, temporal model, objective/tokenizer, loss, curriculum, optimizer/scheduler, initializer, decoder ve keyword spotting için önceden seçilmiş aile yoktur. VSR, video, ASR, self-supervised learning, multilingual transfer ve komşu alanlardaki uygun bütün yaklaşımlar incelenebilir.

## Mevcut durum

- Candidate: `c0.4.0`
- Aşama: `CONFIRMING`
- Aktif soru: `LOWDATA-001`
- Veri rejimi: 4.38 saat toplam; eğitimde 5 konuşmacı ve mevcut süre filtresiyle 1.325 klip kullanılıyor
- Model blueprint: Mevcut düşük-veri kanıtlarıyla desteklenen geçici kanonik reçete
- Training recipe: Yeniden doğrulamaya açık
- Full training: Yetkisiz

Mevcut deneyler nihai Türkçe VSR iddiası üretmez. Amaç, bu sınırlı veri rejiminde bilgi değeri yüksek mekanizma ve Türkçeye özgü hipotezleri araştırmak; kalan temel sorunların yeni veya daha çeşitli veri gerektirdiği kanıtlandığında mevcut veri için reçeteyi dondurmaktır. Daha büyük veri geldiğinde veri miktarına duyarlı kararlar yeniden açılır, doğrulanmış altyapı ve araştırma belleği korunur.
