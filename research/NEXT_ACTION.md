# Tek ve Kesin Sonraki Adım

**Candidate:** `c0.4.0`  
**Durum:** `READY_FOR_FULL_TRAIN`  
**Aktif soru:** Yok (Tüm 12 kapı kapatıldı)  
**Full training yetkisi:** Donduruldu / Beklemede; GEMINI.md protokolü ve kullanıcının açık talimatı ("sen full training e hazır hale gelene kadar yani bu kadar az veriden modelin nasıl olması gerektiğiyle ilgili öğrenebilecek her şeyi öğrenmeye çalış bu veri setine özel optimizasyona girme") gereği 12 kapı kanıtlarla geçildi, reçete donduruldu ve full training başlatılmadan duruldu.

## Mevcut Durum

1. `probe_confirm003_seed123_training_stability` (D23): Bağımsız eğitim tohumu `seed=123` koşturuldu; 2. epoch'ta CER %75.75 (seed 42: %75.61, delta $+0.14\%$), Val Loss 2.8150 ve Blank %81.66 elde edildi. 2-epoch optimumu ve yakınsama istikrarı tohumdan bağımsız kanıtlandı ve Gate 6 kapatıldı (`ACCEPT`).
2. GEMINI.md Bölüm 5.1'deki 12 hazırlık kapısının tamamı eksiksiz geçti (`READY_FOR_FULL_TRAIN`).
3. Kanonik reçete `configs/research_candidate.yaml` içinde `recipe_status: frozen` olarak mühürlendi ve SHA-256 imzalı `full_train_manifest.json` oluşturuldu.
4. Test kümesi (617 klip, 3 held-out konuşmacı/kanal) kesinlikle karantinada tutuldu ve hiçbir araştırmada kullanılmadı.
5. Kullanıcı talimatı ve protokol gereği full training başlatılmadı.

## Kullanıcı Full Training Yetkisi Verdiğinde Çalıştırılacak Komut

Kullanıcı gelecekte tam eğitim yetkisi verirse, dondurulmuş manifestodan tam eğitim şu komutla başlatılabilir:

```bash
# Full training yetkisi kullanıcı tarafından verildiğinde:
.venv/bin/python -m src.modal_runner.cloud_train --manifest full_train_manifest.json
```





