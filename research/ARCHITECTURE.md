# Mimari Araştırma Alanı

**Candidate:** `c0.4.0`  
**Durum:** Düşük-veri rejiminde desteklenen kanonik blueprint; `LOWDATA-001` kapsamında yeniden doğrulamaya açık.

## Ürün sistemi katmanları

Araştırma aşağıdaki sorumlulukları ayrı ayrı ele alır. Güncel seçimler `research/CANDIDATE.md` ve `configs/research_candidate.yaml` içindedir; bunlar birkaç saatlik ve az konuşmacılı mevcut veri rejiminde desteklenmiş seçimlerdir, gelecekteki daha büyük veri için değişmez mimari iddiası değildir:

1. Veri kaynağı, kalite ve yönetişim.
2. Yüz/dudak bulma, hizalama ve video normalizasyonu.
3. Görsel veya spatio-temporal özellik çıkarımı.
4. Zamansal/sekans modelleme.
5. Eğitim objective'i, tokenizer ve loss.
6. Pretraining, initializer ve transfer stratejisi.
7. Curriculum, sampling, optimizer ve regularization.
8. Decoder ve Türkçe dil bilgisi entegrasyonu.
9. Top-500 keyword spotting ve zaman hizalama.
10. Değerlendirme, hata analizi, maliyet ve üretim çıkarımı.

## Araştırma özgürlüğü

VSR, video understanding, action recognition, ASR, sequence transduction, self-supervised/audio-visual learning, multilingual transfer, distillation ve komşu alanların tüm uygun fikirleri değerlendirilebilir. CNN, recurrent, attention, state-space ve hibrit encoder'lar; CTC, transducer, attention/seq2seq, segmental ve çok görevli hedefler; karakter, subword, fonem, visem ve hibrit gösterimler önceden yasaklı değildir.

Bir çözüm yalnız mevcut kodda bulunduğu, popüler olduğu veya geçmişte denenmiş olduğu için seçilemez. Seçimler denetlenmiş veri rejimi, ürün hedefi, birincil kaynak, kontrollü probe, ham hata davranışı, lisans/erişim ve maliyetle gerekçelendirilir.

## Güncel araştırma kapısı

`DATA-001` tamamlanmış ve `c0.4.0` blueprint'i oluşturulmuştur. Sonraki probe'lar yalnız en yüksek bilgi değerli açık soruyu ele alarak bu tek blueprint'i geliştirir. Veri sürümü veya konuşmacı çeşitliliği anlamlı biçimde değiştiğinde curriculum, regularization, freeze/unfreeze, hedef temsili, decoder ve KWS kalibrasyonu gibi veri-duyarlı kararlar yeniden değerlendirilir; veri hattı, provenance ve doğrulanmış hata mekanizmaları korunur.
