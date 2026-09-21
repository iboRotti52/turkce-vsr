# Dudak Okuma Modeli — Mimari, Sıfırdan Anlatım

Bu doküman, bilgisayarla görü (computer vision) veya yapay zeka bilmeyen biri için
yazıldı. Amaç: "video görüntüsünden kelime tahmin eden model" dediğimizde,
kutunun içinde gerçekte ne olduğunu anlamak.

Tek bir benzetme üzerinden gideceğiz: **bir dedektif hikayesi.**

---

## 1. Büyük Resim

Elimizde 1 saniyelik sessiz bir video var — birinin ağzının hareket ettiği bir klip.
Amacımız: bu videoya bakıp "bu kişi hangi kelimeyi söyledi?" sorusunu cevaplayan
bir program yazmak.

İnsan bunu nasıl yapar? Videoyu izler, dudak hareketlerini takip eder, ve
tecrübesine dayanarak "bu 'merhaba' olmalı" der. Model de aslında aynı şeyi
yapıyor — sadece "tecrübe" yerine, ona önceden gösterdiğimiz binlerce örnekten
öğrendiği örüntüleri kullanıyor.

Bu süreç üç ana bölümden oluşuyor:

```
[VİDEO KARELERİ] → [GÖZLEMCİ] → [HİKAYE ANLATICI] → [KARAR VERİCİ] → "MERHABA"
```

Şimdi her kutuyu tek tek açalım.

---

## 2. Bilgisayar İçin "Görüntü" Ne Demek?

Önce en temel şeyden başlayalım: bir fotoğraf, bilgisayar için aslında dev bir
**sayı tablosu**dur. Siyah-beyaz bir fotoğrafta her kare (piksel) 0 (tam siyah)
ile 255 (tam beyaz) arasında bir sayıdır. Renkli fotoğrafta bu üç kat (kırmızı,
yeşil, mavi için ayrı sayılar).

Yani 96×96 piksellik küçük bir ağız fotoğrafı, bilgisayar için **9216 tane sayıdan
oluşan bir liste**dir. Bizim videomuz da, art arda dizilmiş böyle sayı
tablolarından (kareler) ibaret — 1 saniyede 25 kare varsa, elimizde 25 tane sayı
tablosu var.

**Özet:** Model "görüntüye bakmıyor" aslında — bir sürü sayıya bakıyor. "Görme"
dediğimiz şey, bu sayılardaki örüntüleri fark etmek.

---

## 3. Bölüm 1 — "Gözlemci" (her karede neye bakmalı?)

Diyelim ki bir dedektif elindeki 25 fotoğrafın (kareden kareye) her birini
inceliyor. Her fotoğrafta tek tek "burada ne var" diye bakması gerekiyor: dudak
nerede, ne kadar açık, köşeleri nereye dönük, dişler görünüyor mu...

Bunu **elle** yazmaya çalışsak (ör. "eğer üst dudak pikseli şu değerdeyse...")
imkansız derecede karmaşık olurdu. Onun yerine, bir **öğrenen filtre yığını**
kullanırız — buna "CNN" (Convolutional Neural Network) denir, ama adını unutun,
şuna benzetin:

> Fotoğrafın üzerinde gezen küçük bir büyüteç düşünün. Bu büyüteç önce çok basit
> şeyleri fark etmeyi öğrenir: "burada bir kenar var", "burada bir eğri var".
> Bir sonraki katmanda bu basit ipuçlarını birleştirip daha karmaşık şeyler fark
> eder: "bu bir dudak köşesi", "bu bir diş sırası". En üstte ise "bu ağız şu
> şekilde açık" gibi oldukça anlamlı bir özet çıkarır.

Bu "büyüteç yığınının" katman katman ne aradığını **biz programlamıyoruz** —
model, binlerce örnek görerek kendisi keşfediyor. Bizim işimiz sadece ona doğru
örnekleri ve doğru cevapları göstermek (buna birazdan geleceğiz).

**Sonuç:** Her kare (25 tane), bu "gözlemci" katmanından geçince, o karenin
"özeti" haline gelir — ham piksel yığını değil, "bu karede dudak şu durumda"
diyen küçük ve anlamlı bir sayı listesi.

---

## 4. Bölüm 2 — "Hikaye Anlatıcı" (kareler arasındaki sırayı anlamak)

Tek bir fotoğraf "ağzın açık olduğunu" gösterir ama **bir kelime söylemez**.
Kelimeyi anlamak için karelerin **sırasını ve değişimini** görmek gerekir —
tıpkı bir çizgi roman gibi: kareleri tek tek değil, **art arda** okuyunca hikaye
ortaya çıkar.

Bu yüzden 25 karenin "özetini" (Bölüm 1'den çıkan) sırayla okuyan ikinci bir
parça var. Buna kabaca "zaman-duyarlı hafıza" diyebiliriz (teknik adı: TCN veya
GRU/LSTM gibi şeyler — yine adını unutun):

> Bir kitap okurken, her cümleyi öncekilerden bağımsız okumazsınız — önceki
> cümleler aklınızda kalır ve şimdiki cümleyi onlara göre yorumlarsınız. Bu
> parça da tam olarak bunu yapıyor: 1. kareyi okur, hafızasına bir not düşer;
> 2. kareyi okur, önceki notla birlikte değerlendirir; ... 25. kareye kadar
> devam eder ve sonunda "bu hareket dizisi genel olarak şuna benziyor" diyen
> **tek bir özet** çıkarır.

**Sonuç:** 25 ayrı kare-özeti, tek bir "hareket özetine" dönüşür. Artık elimizde
"ağız şu sırayla şöyle hareket etti" bilgisini taşıyan tek bir sayı listesi var.

---

## 5. Bölüm 3 — "Karar Verici" (bu hangi kelime? — KAPALI KÜME için)

Buraya kadar anlattığımız "Gözlemci" ve "Hikaye Anlatıcı", **her iki yaklaşımda
da aynı**. Fark, tam olarak bu son adımda başlıyor. Önce kapalı-küme yolunu
görelim (daha basit olan):

Elimizdeki "hareket özeti"ni alıp, önceden belirlediğimiz **sabit** kelime
listesindeki (ör. 50 Türkçe kelime) her kelime için bir **olasılık puanı**
hesaplıyoruz:

```
merhaba   → %78
nasılsın  → %10
su        → %4
...       → ...
```

En yüksek puanlı kelime, modelin tahmini oluyor. Bu, aslında bir **çoktan
seçmeli sınav** gibi çalışıyor — model, kendisine önceden verdiğimiz sabit
seçenek listesinden birini işaretliyor. Listede olmayan bir kelime asla
tahmin edilemez, çünkü model o seçeneği hiç görmedi.

---

## 6. Açık Küme'de "Karar Verici" Neden Farklı Çalışır?

Açık küme demek, **sabit bir seçenek listesi olmaması** demek — model herhangi
bir Türkçe cümleyi "yazabilmeli". Bu durumda "50 seçenekten birini işaretle"
mantığı çöker, çünkü olası cümle sayısı sonsuzdur; hepsi için ayrı ayrı
olasılık hesaplamak mümkün değil.

Çözüm: model, cevabı **tek seferde seçmek yerine, harf harf yazar** — tıpkı
bir insanın klavyede yazarken bir sonraki tuşu, o ana kadar yazdıklarına
bakarak seçmesi gibi. Yukarıdaki ikinci şemada bu süreç açık: model önce "M"
harfini tahmin eder, sonra "şu ana kadar M yazdım" bilgisini de girdisine
katıp bir sonraki harfi ("E") tahmin eder, bunu "MERHABA" tamamlanana ve
model kendisi "DUR" sinyalini üretene kadar sürdürür.

> Buna benzetme: bir arkadaşınıza SMS'te "n" yazdığınızda telefonunuzun
> "nasılsın" önerisini çıkarması gibi düşünün — ama burada her adımda yeni bir
> harf tahmin ediliyor, telefonun önerisi gibi bir kez değil, cümle bitene
> kadar tekrar tekrar.

**İki yaklaşımın özet farkı:**

| | Kapalı küme | Açık küme |
|---|---|---|
| Karar şekli | Tek seferde, sabit listeden seç | Harf harf, adım adım üret |
| Çıktı | Önceden bilinen bir kelime | Herhangi bir cümle olabilir |
| Ne zaman biter | Anında (tek adım) | Model "DUR" deyince |
| Zorluk | Kolay, az veriyle öğrenilir | Zor, çok daha fazla veri ister |
| Bizim projede | `audit.py`'nin bulduğu en sık N kelime | Transfer learning (AV-HuBERT vb.) gerekir |

Gözlemci ve Hikaye Anlatıcı bölümleri **değişmiyor** — ikisi de aynı "hareket
özetini" üretiyor. Değişen tek şey, bu özetin nasıl bir kelimeye/cümleye
çevrildiği.

---

## 7. Peki Model Bunu Nasıl "Öğreniyor"?

Buraya kadar anlattığımız, eğitilmiş bir modelin **tahmin yaparken** ne yaptığıydı.
Peki bu "gözlemci" ve "hikaye anlatıcı" bölümleri en başta neyi arayacaklarını
nereden biliyor? Cevap: **bilmiyorlar, öğreniyorlar.**

Eğitim süreci şöyle işliyor (kaba özet):

1. Modele elimizdeki örneklerden birini gösteririz: bir `face.mp4` klibi + doğru
   cevap ("bu 'merhaba' kelimesi").
2. Model henüz hiçbir şey bilmediği için (başta ayarları rastgele) yanlış bir
   tahmin yapar, mesela "%60 su".
3. Doğru cevapla ("merhaba") karşılaştırırız — ne kadar yanıldığını ölçeriz.
4. Modelin içindeki milyonlarca küçük ayar değeri, **bu hatayı biraz azaltacak**
   yönde ufak ufak güncellenir (bu adıma "geri yayılım" denir — matematiği
   karmaşık ama fikri basit: "az önce yanıldığın yönün tersine biraz kay").
5. Bunu binlerce örnekle, defalarca tekrarlarız.

Zamanla model, "dudak şu şekilde hareket ederse muhtemelen bu kelimedir"
örüntülerini kendi kendine keşfetmiş olur. Biz ona kuralı yazmadık — sadece
**örnek + doğru cevap** çiftlerini bol miktarda gösterdik.

**İşte tam burada dataset'imiz devreye giriyor.** Topladığımız her `face.mp4` +
`align.json` çifti, modele gösterdiğimiz bir "örnek + doğru cevap"tır. Ne kadar
çok ve çeşitli örneğimiz varsa, model o kadar iyi öğrenir.

---

## 8. Bizim Projede Bu Akış Nasıl Kuruluyor?

```
data/master/<video>/<segment>/face.mp4   ──┐
                                            ├─→ [Gözlemci] → [Hikaye Anlatıcı] → [Karar Verici] → tahmin: "merhaba"
data/master/<video>/<segment>/align.json ──┘         (eğitim sırasında, tahmin doğru cevapla karşılaştırılır)
```

- `face.mp4` → **girdi** (25 kare, her biri sayı tablosu)
- `align.json`'daki kelime → **doğru cevap** (eğitim sırasında karşılaştırma için)
- Eğitim bitince elimizde, hiç görmediği yeni bir `face.mp4` verildiğinde
  kelimeyi tahmin edebilen bir model olur.

---

## 9. Sık Sorulan Kavramlar — Basit Karşılıkları

| Duyacağın terim | Basit karşılığı |
|---|---|
| CNN (Convolutional Neural Network) | "Gözlemci" — her karedeki önemli görsel ipuçlarını bulan katman |
| TCN / GRU / LSTM (zamansal model) | "Hikaye anlatıcı" — kareler arasındaki sırayı/hareketi anlayan katman |
| Sınıflandırma (classification) | Kapalı-küme "Karar verici" — sabit listeden tek seferde seçim |
| Decoder (çözücü) | Açık-küme "yazıcı" — harf harf, sıradaki harfi öncekilere bakarak üreten parça |
| Eğitim (training) | Modelin, örnek + doğru cevap çiftlerinden deneme-yanılmayla öğrenmesi |
| Ağırlıklar (weights) | Modelin içindeki, eğitim sırasında ayarlanan milyonlarca "kadran" |
| Kapalı küme (closed-set) | Kelime listesi sabit ve önceden belli (ör. 50 kelime) — çoktan seçmeli |
| Açık küme (open-set) | Serbest, sınırsız kelime/cümle — harf harf üretilir, çok daha zor |
| CER (Character Error Rate) | Açık-küme modelin performans ölçütü — kaç harf yanlış tahmin edildi |
| Doğruluk (accuracy) | Kapalı-küme modelin test edildiğinde kaç örneği doğru tahmin ettiği yüzdesi |

---

## 10. Tek Cümleyle Özet

**Model, dudak videosunu önce "her karede ne oluyor" diye tek tek inceler, sonra
bu kareleri sırayla okuyup "genel olarak nasıl bir hareket oldu" diye özetler;
bu özeti kapalı-kümede sabit bir listeden tek seferde kelime seçerek, açık-kümede
ise harf harf yazarak bir cevaba çevirir — ve bunu yapmayı, bizim topladığımız
binlerce doğru örnek + cevap çiftinden kendi kendine öğrenir.**
