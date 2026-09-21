# Model Seçimi: MobiVSR vs Auto-AVSR

İki model aslında iki farklı **soruyu** cevaplıyor, ve her şey bu farktan
geliyor:

- **MobiVSR** → *"Bu hangi kelime?"* (sabit bir listeden birini işaretle —
  çoktan seçmeli sınav gibi)
- **Auto-AVSR** → *"Bu ne dedi?"* (serbestçe, harf harf yazarak cevap ver —
  boşluk doldurmalı sınav gibi)

> **Düzeltme:** Auto-AVSR **Meta değil, Imperial College London**'a ait.
> Meta'nın (Facebook AI) kendi modeli aslında **AV-HuBERT** — daha önce
> transfer learning adayı olarak konuştuğumuz model. İkisi de aynı yarışta
> (LRS3 veri seti) yer aldığı için karışmış olabilir.

---

## MobiVSR — kısaca

- **Ne yapıyor:** Videoyu alır, önceden belirlenmiş bir kelime listesinden
  en olası olanı işaretler.
- **Neden hafif:** "Gözlemci" katmanını (video karelerine bakan kısım) daha
  ucuz bir versiyonla değiştiriyor — aynı işi, işi ikiye bölerek çok daha az
  hesaplamayla yapıyor.
- **Veri ihtiyacı:** Orijinal denemede 500 kelimelik bir liste için toplam
  ~170 saat video kullanılmış. Ama asıl belirleyici, **kelime başına örnek
  sayısı** — bizim gibi daha küçük bir listeyle (100-300 kelime) çok daha az
  toplam veriyle de anlamlı bir pilot mümkün.
- **Doğruluk hissi:** Kabaca 4 tahminden 3'ünü doğru buluyor.
- **Nerede çalışır:** Birkaç MB boyutunda, telefonda anında ve internetsiz
  çalışır.

## Auto-AVSR — kısaca

- **Ne yapıyor:** Sabit liste yok. Model, gördüğü dudak hareketini harf harf
  "yazarak" herhangi bir cümleye dönüştürmeye çalışıyor.
- **Neden ağır:** Bu "yazma" işi tek seferde bitmiyor — her harf için modeli
  tekrar tekrar çalıştırmak gerekiyor. Daha çok hesaplama, daha yavaş ve
  daha az öngörülebilir bir gecikme demek.
- **Veri ihtiyacı:** 800 saatin üzerinde (bazı versiyonları 3000+ saat) —
  MobiVSR'ın kullandığından kat kat fazla.
- **Doğruluk hissi:** En iyi haliyle bile yaklaşık her 5 kelimeden biri
  hatalı çıkıyor.
- **Nerede çalışır:** Çok adımlı çalıştığı için telefonda gerçek-zamanlı
  kullanımı pratik değil — genelde sunucu/cloud tarafında çalıştırılıyor.

---

## Yan yana

| | MobiVSR | Auto-AVSR |
|---|---|---|
| Soru tipi | "Hangi kelime?" (listeden seç) | "Ne dedi?" (serbest yaz) |
| Veri ihtiyacı | Onlarca-yüzlerce saat (küçük listeyle) | 800+ saat |
| Doğruluk hissi | ~4/5 doğru | ~4/5 kelime doğru (yani 5'te 1 hatalı) |
| Model boyutu | Birkaç MB | Büyük (100+ MB) |
| Nerede çalışır | Telefon, cihaz üzerinde | Bulut sunucusu (GPU gerekir) |
| İnternet gerekir mi | Hayır | Evet |
| Gecikme | Anında | Yavaş (çok adım + ağ gecikmesi) |
| Maliyet | Bir kere eğit, sonra bedava | Sürekli bulut/GPU faturası |
| Sözlük dışı kelime | Söyleyemez | Söyleyebilir |
| Bizim veri hızımızla | Ulaşılabilir | Gerçekçi değil |

---

## Bizim Projeye Anlamı

Şu anki veri toplama hızımızla (birkaç video → birkaç saat kullanılabilir
veri), MobiVSR ölçeğine ulaşmak **mümkün**. Auto-AVSR ölçeğine (800+ saat)
ulaşmak, mevcut YouTube pipeline'ımızla **gerçekçi değil**. Ayrıca mobil +
güvenilir ürün hedefimiz zaten MobiVSR tarzına yakın duruyor — bu yüzden
kapalı-küme, mobil, tek-seferde-cevap-veren yaklaşım bizim için doğru yol.

---

## Kaynaklar

- [MobiVSR (arXiv)](https://arxiv.org/abs/1905.03968)
- [Auto-AVSR (arXiv)](https://arxiv.org/html/2303.14307v3)
- [Auto-AVSR GitHub — Imperial College London](https://github.com/mpc001/auto_avsr)
