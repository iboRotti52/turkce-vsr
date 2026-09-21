# Auto-AVSR Serverless GPU — Latency Raporu

**Tarih:** 12 Temmuz 2026 · **Platform:** Modal (serverless GPU) · **Model:** Auto-AVSR
`vsr_trlrs3_base.pth` (Imperial College resmî checkpoint, 438 saat LRS3, ~1GB, MD5 doğrulandı)

---

## Özet (TL;DR)

- **Sıcak (warm) istek:** 6 saniyelik konuşma **~3.6–5.1 sn**'de metne dönüyor (GPU'ya göre).
- **Soğuk başlangıç (cold start):** **34–48 sn** — ürün deneyimi için asıl problem bu.
- **Uzun konuşmada gecikme büyüyor:** 24 sn'lik klip T4'te **gerçek zamandan yavaş** (24.8 sn) işleniyor.
- **Eşzamanlı isteklerde kuyruk:** 4 istek aynı anda gelirse sonuncusu ~18–19 sn bekliyor.
- **GPU önerisi:** L4 — T4'ten ~%25-35 hızlı, A10G ile hemen hemen aynı ama daha ucuz.
- Sonuç, MobiVSR/cihaz-üstü kararımızı destekliyor: bulut açık-küme çalışıyor ama gecikme + maliyet ödünü ölçülmüş ve gerçek.

---

## 1. Test Düzeneği

- **Girdi:** Kamu malı İngilizce konuşma videosu (Beyaz Saray haftalık konuşması,
  frontal yüz, kameraya konuşma). Tek kaynaktan 4 klip kesildi: **2 / 6 / 12 / 24 sn**
  (25fps, **sessiz** — model yalnız dudaktan okuyor). Türkçe projemizin verisi **kullanılmadı**.
- **GPU'lar:** Tesla T4, NVIDIA L4, NVIDIA A10G (her biri ayrı uygulama → gerçek soğuk başlangıç).
- **Ölçümler:** Her GPU için 1 soğuk çağrı + her klip uzunluğunda 3 sıcak çağrı
  (medyan raporlandı) + 4 eşzamanlı istek (burst).
- **"Uçtan uca":** video baytlarının yerel makineden (Türkiye) gönderilmesi + ön işleme +
  GPU çıkarımı + metnin dönmesi. Yani gerçek bir istemcinin yaşayacağı süre.
- Yüz tespiti mediapipe (CPU, konteyner içinde); model çıkarımı GPU (beam search dahil).

---

## 2. Soğuk Başlangıç (ilk istek)

| GPU | Uçtan uca | Model yükleme | İlk GPU çıkarımı |
|---|---|---|---|
| T4 | **47.9 sn** | 5.3 sn | 5.2 sn |
| L4 | **33.6 sn** | 4.1 sn | 3.7 sn |
| A10G | **40.5 sn** | 4.0 sn | 5.1 sn |

Kalan ~25–40 sn: konteyner ayağa kaldırma + image çekme + CUDA ısınması.
**Yorum:** Boşta bekleyen kullanıcının ilk cümlesi yarım dakikadan fazla bekler.
Bir iletişim aracı için kabul edilemez → ya sürekli 1 kopya sıcak tutulur (maliyeti
aşağıda) ya da kullanıcı uygulamayı açtığında arka planda ısıtma yapılır.

---

## 3. Sıcak İstek — Uzunluk × GPU (uçtan uca, medyan)

| Klip | T4 | L4 | A10G |
|---|---|---|---|
| 2 sn | 1.4 sn | 1.4 sn | 1.5 sn |
| 6 sn | 5.1 sn | **3.6 sn** | 3.9 sn |
| 12 sn | 10.7 sn | 9.0 sn | 9.0 sn |
| 24 sn | 24.8 sn | 21.8 sn | 21.7 sn |

**Gerçek-zaman oranı** (işleme süresi ÷ konuşma süresi; 1.0 üstü = konuşmadan yavaş):

| Klip | T4 | L4 | A10G |
|---|---|---|---|
| 2 sn | 0.72 | 0.71 | 0.77 |
| 6 sn | 0.85 | 0.60 | 0.65 |
| 12 sn | 0.89 | 0.75 | 0.75 |
| 24 sn | **1.03** ⚠ | 0.91 | 0.90 |

**Kritik bulgu:** Gecikme, konuşma uzadıkça **doğrusaldan hızlı** büyüyor. GPU süresi
2sn→24sn arasında (girdi 12×) **18–24× artıyor** — dikkat mekanizması kare sayısıyla
karesel ölçeklendiği ve beam search çıktı uzunluğuyla adım adım çalıştığı için.
T4'te 24 sn'lik cümle gerçek zamanı aşıyor: konuşma sürdükçe birikme (lag) başlar.
**Pratik çıkarım:** uzun konuşma tek parça gönderilmemeli; cümle cümle bölmek
(bizim pipeline'daki segmentasyon gibi) şart.

---

## 4. Zaman Nereye Gidiyor? (sıcak, aşama kırılımı — L4, medyan)

| Aşama | 2 sn | 6 sn | 12 sn | 24 sn |
|---|---|---|---|---|
| Video okuma (CPU) | 0.11 | 0.32 | 0.96 | 2.09 |
| **Yüz tespiti (CPU)** | 0.52 | 1.57 | 3.22 | **6.16** |
| Ağız kırpma + transform | 0.03 | 0.11 | 0.25 | 0.49 |
| **Model çıkarımı (GPU)** | 0.48 | 1.33 | 3.88 | **10.34** |

**Yorum:** Zamanın ~%30'u yüz tespitine (CPU) gidiyor — GPU'ya hiç dokunmayan bir iş.
Bu adım istemciye (telefona) taşınırsa hem sıcak gecikme ~%30 düşer hem de buluta ham
yüz görüntüsü yerine yalnız ağız bölgesi gönderilir (gizlilik kazanımı).

---

## 5. Eşzamanlılık (Burst) — 4 istek aynı anda

Bitiş zamanları (istek gönderiminden itibaren, 6 sn klip):

| GPU | 1. istek | 2. | 3. | 4. |
|---|---|---|---|---|
| T4 | 8.3 sn | 10.9 | 14.8 | **19.1** |
| L4 | 8.8 sn | 10.2 | 14.2 | **18.4** |
| A10G | 8.7 sn | 10.1 | 13.9 | **18.0** |

**Yorum:** Varsayılan ayarlarla istekler tek konteynerde **kuyruğa girip sırayla**
işleniyor (yeni konteyner açılmıyor — açılsaydı her biri ~35 sn soğuk başlangıç
öderdi). Yani eşzamanlı kullanıcı sayısı arttıkça bekleme doğrusal büyür; gerçek
üründe ölçekleme ayarı (min. konteyner sayısı / eşzamanlılık limiti) bilinçli
seçilmek zorunda.

---

## 6. Maliyet (Modal yayımlanmış saatlik tarifeler*, yaklaşık)

| | T4 (~$0.59/sa) | L4 (~$0.80/sa) | A10G (~$1.10/sa) |
|---|---|---|---|
| Sıcak 6 sn'lik 1 istek | ~$0.0008 | ~$0.0008 | ~$0.0012 |
| 7/24 sıcak tutma (aylık) | ~$425 | ~$576 | ~$792 |

*Fiyatlar değişebilir; kesin rakam için modal.com/pricing. Bu testin tamamı
(45 ölçüm, 3 GPU) aylık $30 ücretsiz kredinin küçük bir kısmıyla yapıldı.

**Yorum:** İstek başına maliyet önemsiz; asıl fatura **cold-start'ı çözmek için
sıcak tutma**dan gelir. Kullanıcı tabanı büyüyene kadar "uygulama açılınca ısıt +
5 dk boşta kalınca kapat" stratejisi mantıklı orta yol.

---

## 7. Kalite Gözlemi (ölçümün yan ürünü)

Sessiz 24 sn klipten modelin çıkardığı metin:
> "TWO THIS IS WHEN WE SAW MORE PEOPLE FILE FOR OUR EMPLOYMENT THAN 80 TIMES IN THE LAST 26 HOURS..."

Konuşmanın gerçek konusu (2009 ekonomi/işsizlik) yakalanmış, hatalar mevcut — LRS3
%36 WER seviyesiyle tutarlı. Üç GPU da **birebir aynı** transkripti üretti
(ölçümler deterministik → karşılaştırma güvenilir).

---

## 8. Ürün Kararına Etkisi

1. **MobiVSR/cihaz-üstü ana yol kararını destekliyor:** bulut açık-küme *çalışıyor*,
   ama sıcakta ~4 sn + soğukta ~35 sn + sıcak tutma faturası, cihaz-üstü anlık
   cevabın karşısında ölçülmüş bir ödün.
2. Bulut yolu ileride açılırsa reçete: **L4 + 1 kopya sıcak + yüz tespiti istemcide +
   cümle bazlı bölme.** Bu kombinasyon 6 sn'lik cümleyi ~2 sn'ye indirebilir (tahmin;
   ölçülmedi).
3. Uzun kesintisiz konuşma hiçbir konfigürasyonda tek parça gönderilmemeli
   (karesel büyüme + T4'te gerçek-zaman aşımı).

## 9. Cold-Start Optimizasyon Deneyi: Snapshot + Enter-Isınması (ölçülmüş)

Rapordaki en büyük sorun (34–48 sn cold start) için iki teknik uygulandı ve ölçüldü
(L4, 6 sn klip, deploy edilmiş `avsr-snapshot` uygulaması):

1. **GPU memory snapshot** — model GPU'ya yüklenmiş ve CUDA kernelleri ısınmış
   haldeyken süreç "donduruluyor"; sonraki cold start bu durumdan başlıyor.
2. **Enter-ısınması** — ilk isteğin ödediği CUDA ısınma bedeli, istek yoluna değil
   konteyner açılışına (snapshot'ın içine) taşındı.

| Senaryo | Uçtan uca cold |
|---|---|
| Baz (snapshot yok) | 33.6 sn |
| Snapshot restore — **worker cache'li (en iyi)** | **10.5 sn (3.2×)** |
| Snapshot restore — cache'siz worker | 27–35.5 sn (kazanç yok) |
| Karşılaştırma: sıcak istek | ~4–5.7 sn |

**Bulgular:**
- Snapshot mekanizması doğrulandı (sunucu logları: "Creating GPU memory snapshot" /
  "Restoring Function from memory snapshot") ve en iyi durumda cold'u **33.6 → 10.5 sn**'ye indirdi.
- **Asıl değişken worker cache'i:** restore süresi, GB'larca snapshot dosyasının o an
  atanan makineye çekilmesiyle belirleniyor. Düşük trafikte her cold farklı makineye
  düşüyor → kazanç bazen sıfır. Trafik arttıkça (aynı worker havuzu ısındıkça) 10 sn'lik
  durum tipikleşir — yani bu optimizasyon **ölçekle birlikte iyileşir**, pilotta dalgalıdır.
- Enter-ısınması güvenilir şekilde çalıştı: cold çağrının GPU dilimi 3.7–5.2 sn'den
  ~2.0 sn'ye indi (ilk kullanıcı CUDA ısınması ödemiyor).
- **Pratik reçete:** snapshot + "uygulama açılınca arka planda boş istek at" (öngörülü
  ısıtma). Isıtma pingi artık ~35 sn değil ~10 sn sürdüğü için kullanıcı konuşmaya
  başladığında konteyner çok daha yüksek olasılıkla hazır.

*Deney scripti: `benchmark/vsr_snapshot.py`; uygulama Modal'da deploy halinde
(`avsr-snapshot`, boştayken maliyet yok).*

## 10. Sınırlamalar

- Tek konuşmacı, tek video kaynağı, tek bölge (ağ süresi Türkiye→Modal içerir).
- Klip başına 3 tekrar (medyan) — P99 kuyruğu için daha çok örnek gerekir.
- İngilizce model (LRS3); Türkçe modelin hız profili benzer olur ama kalitesi ayrı konu.
- Burst testi varsayılan ölçekleme ayarlarıyla yapıldı.

---

*Ölçüm scripti ve ham sonuçlar: [benchmark/](benchmark/) — `vsr_benchmark.py`
(T4/L4/A10G × 2-24 sn × 3 tekrar + burst), `bench_T4.json`, `bench_L4.json`,
`bench_A10G.json`.*
