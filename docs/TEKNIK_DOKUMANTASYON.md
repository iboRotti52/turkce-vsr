# Türkçe Dudak Okuma Projesi — Teknik Dokümantasyon

Bu doküman, projede kullanılan tüm kavramları **tümdengelim yöntemiyle** sunar: her
bölüm, kendinden önceki bölümlerde kurulan tanım ve önermelerden **zorunlu olarak**
türetilir. Okuyucunun konu hakkında önceden hiçbir bilgisi olmadığı varsayılmıştır;
her terim ilk geçtiği yerde biçimsel olarak tanımlanır. Metinde benzetme
kullanılmamıştır — her kavram, gerçek matematiksel/işlemsel tanımıyla verilmiştir.

---

## 1. Problemin Biçimsel Tanımı

Projenin hedefi şu şekilde ifade edilebilir: bir kişinin dudak hareketini gösteren
bir video verildiğinde, o kişinin söylediği kelimeyi veya cümleyi belirlemek.

Bu, matematiksel olarak bir **fonksiyon** arayışıdır: girdi olarak bir video dizisi
alan, çıktı olarak metin üreten bir fonksiyon `f`. Bu fonksiyonun kapalı bir
formülünü (elle yazılabilecek bir denklemini) bilmiyoruz — dudak hareketiyle
kelime arasındaki ilişki, elle formülleştirilemeyecek kadar karmaşıktır.

**Tanım 1.1 (Öğrenme):** Bir fonksiyonun kapalı formülü bilinmediğinde, o
fonksiyona **örneklerden** yaklaşmak mümkündür. Bunun için üç bileşen gerekir:

1. Bir **aday fonksiyon ailesi** — çok sayıda ayarlanabilir sayısal değer
   (parametre) içeren, girdiyi çıktıya eşleyen bir hesaplama şeması. Buna
   **model** denir.
2. Girdi-çıktı çiftlerinden oluşan bir **örnek kümesi** — her biri (video,
   doğru metin) biçiminde. Buna **veri kümesi (dataset)** denir.
3. Model parametrelerini, verilen örneklerdeki hatayı en aza indirecek şekilde
   ayarlayan bir **işlem**. Buna **eğitim (training)** denir.

Bu üç bileşenin her biri, aşağıdaki bölümlerde ayrı ayrı ve sırayla türetilecektir.

---

## 2. Girdinin Sayısal Temsili

Model bir hesaplama şemasıdır; hesaplama yalnızca sayılar üzerinde tanımlıdır.
Dolayısıyla video girdisi, işlenmeden önce sayısal bir forma dönüştürülmelidir.

**Tanım 2.1 (Dijital görüntü):** Bir dijital görüntü, `(x, y)` konum
çiftlerinden (piksel) yoğunluk değerlerine giden bir fonksiyondur. Gri tonlamalı
bir görüntüde her piksel tek bir sayıdır (tipik olarak 0–255 arası); renkli bir
görüntüde her piksel üç sayıdan oluşur (kırmızı, yeşil, mavi kanal yoğunluğu).

**Tanım 2.2 (Video):** Bir video, sabit bir zaman aralığıyla (kare hızı, fps)
örneklenmiş, sıralı bir görüntü dizisidir: `(I_1, I_2, ..., I_T)`. Burada `T`,
toplam kare sayısıdır.

**Sonuç 2.1:** Video, ele alınabilir en küçük birime indirgendiğinde, sıralı bir
sayı tabloları dizisidir. Model, bu diziyi işleyip bir metin çıktısı üretmek
zorundadır. Bu iş iki alt-probleme ayrılır: (a) her karedeki mekânsal bilginin
işlenmesi, (b) kareler arasındaki zamansal ilişkinin işlenmesi. Bu iki alt-problem
sırasıyla Bölüm 3 ve Bölüm 4'te ele alınacaktır.

---

## 3. Mekânsal İşleme: Evrişim (Convolution)

Bir tek karedeki ham piksel değerleri, doğrudan sınıflandırma için uygun değildir:
boyut çok yüksektir (224×224 pikselli bir görüntü 50.176 sayı içerir) ve aynı
görsel örüntü (örneğin bir dudak kenarı), görüntüde farklı konumlarda
görünebilir. Bu iki özellik, aşağıdaki gereksinimi doğurur:

**Gereksinim 3.1:** İşlem, (a) girdi boyutunu anlamlı biçimde küçültmeli, (b) aynı
örüntüyü, görüntüdeki konumundan bağımsız olarak tanımalıdır (konum-değişmezliği).

**Tanım 3.1 (Evrişim katmanı):** Sabit boyutlu bir sayı tablosu (**çekirdek**,
örn. 3×3) tanımlanır. Bu çekirdek, girdi görüntüsünün her konumunda, o konumu
merkez alan aynı boyuttaki bir komşulukla eleman-bazında çarpılıp toplanır. Aynı
çekirdek tüm konumlarda **tekrar kullanılır** (ağırlık paylaşımı) — bu, Gereksinim
3.1(b)'yi doğrudan karşılar, çünkü çekirdek konumdan bağımsız aynı işlemi uygular.
Çıktı, girdiden daha küçük (veya adım büyüklüğü seçimine bağlı olarak eşit)
boyutlu yeni bir sayı tablosudur.

**Önerme 3.1:** Tek bir evrişim katmanı, yalnızca çekirdek boyutu kadar yerel bir
komşuluğu görebilir (**alım alanı**, receptive field). Bir dudağın genel şeklini
(birden çok yerel kenarın birleşimi) tanımak için, birden çok evrişim katmanının
**art arda** uygulanması gerekir: her katman, bir önceki katmanın çıktısını girdi
alır, dolayısıyla etkin alım alanı katman sayısıyla birlikte büyür.

**Sonuç 3.1:** Katmanlar art arda dizildiğinde, ilk katmanlar düşük seviyeli
örüntüleri (kenar, köşe) tespit eden sayılar üretir; sonraki katmanlar, bu düşük
seviyeli çıktıları girdi alarak daha yüksek seviyeli örüntüleri (dudak açıklığı,
dudak köşesi konumu) temsil eden sayılar üretir. Bu katman yığınının çıktısına,
o karenin **öznitelik vektörü** denir — ham piksellerden çok daha küçük boyutlu,
o karenin görsel içeriğini özetleyen bir sayı listesi.

**Sonuç 3.2 (Parametre sayısı):** Bir evrişim katmanının parametre sayısı,
çekirdek boyutu × girdi kanal sayısı × çıktı kanal sayısı ile orantılıdır. Video
işlemede (üç boyutlu evrişim: genişlik, yükseklik, zaman), bu çarpım büyük
değerlere ulaşır. Bölüm 8'de, bu çarpımı azaltmaya yönelik özel bir teknik
(ayrıştırılmış evrişim) tanıtılacaktır.

---

## 4. Zamansal İşleme

Bölüm 3, tek bir karenin öznitelik vektörünü türetmeyi sağlar. Ancak tek bir kare,
bir kelimeyi belirlemek için yetersizdir: aynı ağız şekli, farklı kelimelerin
farklı anlarında ortaya çıkabilir (örneğin, ağzın kapalı olduğu an, hem "b" hem
"p" sesinden önce/sonra görülebilir). Kelimenin kimliği, ağız şeklinin **zaman
içindeki değişim dizisinde** kodludur.

**Gereksinim 4.1:** `T` adet öznitelik vektörünü (her kareden bir tane), sırayı
dikkate alarak, **tek bir özet vektöre** indirgeyen bir mekanizma gereklidir.

**Tanım 4.1 (Zamansal model):** Bu mekanizma, `T` öznitelik vektörünü girdi
alıp bunları önceden belirlenmiş bir sırayla (genellikle zaman sırasıyla) işleyen,
her adımda o ana kadar işlenmiş bilgiyi taşıyan bir iç durum güncelleyen, ve son
adımda (veya tüm adımların ağırlıklı birleşiminde) bir özet çıktı üreten bir
hesaplama şemasıdır. Bu projede incelenen iki model, bu mekanizmayı iki farklı
biçimsel yöntemle gerçekler:

- **Zamansal evrişimli ağ (TCN):** Bölüm 3'teki evrişim tanımının, mekân yerine
  zaman ekseni üzerinde uygulanmasıdır. Çekirdek, ardışık zaman adımları
  üzerinde kaydırılır.
- **Öz-dikkat (self-attention) tabanlı kodlayıcı (Conformer):** Her zaman adımı
  için, o adımın diğer **tüm** zaman adımlarıyla olan ilişkisini sayısal olarak
  hesaplayan (bir benzerlik puanı üreten) ve bu puanlara göre ağırlıklandırılmış
  bir toplam alan bir mekanizmadır.

**Önerme 4.1 (Hesaplama karmaşıklığı):** Öz-dikkat mekanizması, `T` zaman adımının
**her çiftini** karşılaştırdığından, hesaplama miktarı `T²` (T'nin karesi) ile
orantılıdır. Zamansal evrişim ise sabit genişlikli bir komşuluk kullandığından,
hesaplama miktarı `T` ile doğrusal orantılıdır.

**Sonuç 4.1:** Öz-dikkat tabanlı bir model kullanıldığında, video uzunluğu
arttıkça işlem süresinin **karesel** büyümesi beklenir. Bu önerme, Bölüm 11'de
ölçülen deneysel verilerle doğrudan karşılaştırılacaktır.

---

## 5. Çıktı Uzayının Tanımı: Sınıflandırma ve Dizi Üretimi

Bölüm 4'ün sonunda elde edilen özet vektör, henüz bir metin değildir — sayısal bir
temsildir. Bu vektörü metne çeviren son adımın biçimi, **olası çıktıların
kümesinin yapısına** bağlıdır. İki durum ayırt edilir:

### 5.1 Sonlu ve Önceden Bilinen Çıktı Kümesi

**Tanım 5.1 (Kapalı küme / sınıflandırma):** Olası çıktılar, önceden belirlenmiş,
sonlu bir küme `{w_1, w_2, ..., w_N}` ise (örneğin N adet kelime), son katman,
özet vektörü her `w_i` için bir sayıya (bu sayı, o kelimenin olasılığı olarak
yorumlanır) eşleyen bir doğrusal dönüşüm ve normalizasyon işlemidir. Karar
kuralı: en yüksek olasılık değerine sahip `w_i` seçilir.

**Sonuç 5.1:** Bu yöntemde, model **N farklı çıktıdan başka hiçbir şey
üretemez**. `N` sabittir ve eğitim öncesi belirlenir.

### 5.2 Sonsuz Çıktı Kümesi

**Gereksinim 5.1:** Olası çıktılar, sonlu bir alfabe (harfler) üzerine kurulu
tüm sonlu dizilerin kümesiyse (yani herhangi bir uzunlukta herhangi bir cümle),
bu küme sonsuzdur. Sonsuz bir kümenin her elemanı için ayrı ayrı bir olasılık
hesaplamak (Bölüm 5.1'deki yöntem) uygulanamaz.

**Tanım 5.2 (Açık küme / dizi üretimi):** Çözüm, çıktıyı **tek seferde değil,
sembol sembol** üretmektir. Adım `t`'de, model şu ana kadar üretilmiş sembol
dizisini (`t-1` sembol) ve Bölüm 4'ün özet vektörünü girdi alarak, alfabedeki
her sembol için bir olasılık hesaplar (Bölüm 5.1'deki gibi, ama alfabe
boyutunda — örneğin ~30 harf — sabit bir küme üzerinde); en olası sembol
seçilir ve diziye eklenir. Bu işlem, özel bir **dur sembolü** üretilene kadar
tekrarlanır.

**Sonuç 5.2:** Bu yöntemde model teorik olarak herhangi bir uzunlukta herhangi
bir diziyi üretebilir, çünkü her adımda yalnızca sonlu bir alfabe üzerinde karar
verilir; dizinin toplam uzunluğu önceden sınırlanmamıştır.

**Sonuç 5.3 (Hesaplama maliyeti):** Bölüm 5.1'deki yöntem, özet vektör
başına **tek bir** ileri hesaplama gerektirir. Bölüm 5.2'deki yöntem, çıktı
dizisinin uzunluğu kadar **art arda** ileri hesaplama gerektirir (her adım,
bir önceki adımın çıktısına bağlı olduğundan paralelleştirilemez). Buradan:

**Önerme 5.1:** Açık-küme yöntemi, kapalı-küme yöntemine göre çıktı başına
en az `L` kat (L = ortalama çıktı dizisi uzunluğu) daha fazla hesaplama
gerektirir.

---

## 6. Parametre Öğrenimi (Eğitim)

Bölüm 1'de tanımlanan üçüncü bileşen — eğitim — şimdi biçimsel olarak
tanımlanabilir.

**Tanım 6.1 (Kayıp fonksiyonu):** Model parametreleri `θ` ile gösterilsin.
Bir örnek `(x, y)` için, modelin ürettiği çıktı `f_θ(x)` ile doğru çıktı `y`
arasındaki uyumsuzluğu sayısallaştıran bir fonksiyon `L(f_θ(x), y)` tanımlanır.
Kapalı-küme durumunda bu, doğru sınıfa verilen olasılığın negatif logaritması
gibi bir ölçüttür; açık-küme durumunda, üretilen dizi ile doğru dizi arasındaki
sembol-bazlı uyumsuzluğun toplamıdır.

**Tanım 6.2 (Eğitim):** Eğitim veri kümesindeki tüm örnekler üzerinden
ortalama kaybı en aza indirecek `θ` değerleri aranır. Bu arama, `θ`'nın her
bileşenine göre kaybın türevi hesaplanarak (**geri yayılım**), ve `θ`, bu
türevin ters yönünde küçük adımlarla güncellenerek (**gradyan inişi**)
yürütülür. Bu güncelleme, veri kümesindeki tüm örnekler üzerinden defalarca
tekrarlanır.

**Sonuç 6.1:** Eğitimin geçerliliği, kayıp fonksiyonunun `(x, y)` çiftindeki
`y`'nin gerçekten `x`'e karşılık gelen doğru çıktı olduğu varsayımına dayanır.
Bu varsayım ihlal edilirse (örneğin `x` bir kişinin dudak hareketiyken, `y`
başka bir kaynaktan — dublaj sesinden — alınmış metinse), model yanlış bir
ilişkiyi öğrenir. Bu sonuç, Bölüm 7.5'te veri toplama sürecinin bir adımını
doğrudan gerekli kılacaktır.

**Sonuç 6.2 (Genelleme):** Eğitimin amacı, yalnızca eğitim kümesindeki
örnekleri ezberlemek değil, eğitimde hiç görülmemiş girdilerde de doğru çıktı
üretebilmektir (**genelleme**). Bu nedenle, modelin başarımı, eğitimde
kullanılmamış ayrı bir **test kümesi** üzerinde ölçülmelidir. Test kümesi,
eğitim kümesiyle örtüşmeyen özellikler taşımalıdır — aksi halde ölçülen başarım,
gerçek genelleme yeteneğini değil, ezberleme derecesini yansıtır. Bu sonuç,
Bölüm 7.6'da somutlaştırılacaktır.

---

## 7. Veri Kümesinin İnşası

Bölüm 1–6, bir modelin ne olduğunu ve nasıl eğitildiğini tanımladı. Eğitim,
`(x, y)` çiftlerinden oluşan bir veri kümesi gerektirir (Tanım 1.1). Bu bölüm,
bu çiftlerin nasıl elde edildiğini, önceki bölümlerdeki her gereksinimden
türeterek anlatır.

### 7.1 Ham Kaynak

`x` (dudak videosu) ve `y` (söylenen metin) çiftini aynı anda içeren bir kaynak
gereklidir. Sesli konuşma videoları, hem görsel bileşeni (`x`'in ham hali) hem
sesi (üzerinden `y` elde edilebilir) aynı anda taşır. Bu projede kaynak, tek
konuşmacılı YouTube videolarıdır.

### 7.2 Metin Etiketinin Çıkarımı

`y`'yi elde etmek için sesin metne çevrilmesi gerekir. Bu, otomatik konuşma
tanıma (ASR) modeliyle yapılır. Yalnızca metin yeterli değildir: Bölüm 7.4'te
gerekçelendirileceği gibi, video **kırpılmalıdır** (belirli bir zaman aralığına
indirgenmelidir); bu nedenle her kelimenin **başlangıç ve bitiş zaman
damgasına** da ihtiyaç vardır. Kullanılan ASR modeli (Whisper), bu zaman
damgasını kelime düzeyinde üretir.

### 7.3 Görsel Kaynağın Ayrıştırılması

Bölüm 3'teki evrişim tanımı, girdinin **mekânsal olarak hizalı** olmasını
zımnen varsayar: aynı çekirdek her konuma uygulanacağından, ilgili bölgenin
(ağız) görüntüde tutarlı bir konumda bulunması, modelin öğrenme verimliliğini
artırır. Ham video karesinde konuşmacının yüzü, kadrajın herhangi bir yerinde
olabilir. Bu nedenle, her karede yüzün/ağzın konumunu bulan bir **yüz
işaret noktası tespit** adımı gereklidir (bu projede MediaPipe kullanılmıştır).
Bu adımın çıktısı, sonraki adımda görüntüyü standart bir çerçeveye kırpmak için
kullanılır.

### 7.4 Zaman Ekseninde Bölümleme

Bölüm 5.1'in kapalı-küme yöntemi sabit sayıda çıktı sınıfı varsayar; kelime
düzeyinde çalışan bir sistem için her örnek, tek bir kelimeye karşılık gelen
sınırlı bir zaman aralığı olmalıdır. Cümle düzeyinde çalışan sistemler için de
(Bölüm 5.2), aşırı uzun girdiler Sonuç 4.1'deki karesel maliyeti artırdığından,
konuşma sınırlı uzunlukta parçalara (segment) bölünür. Bölüm 7.2'de elde edilen
kelime zaman damgaları, bu bölümlemeyi (belirli bir süre veya karakter sayısı
sınırına göre) yapmak için kullanılır.

### 7.5 Etiket Doğruluğunun Garantisi

Sonuç 6.1, `(x, y)` çiftindeki `y`'nin gerçekten `x`'teki dudak hareketine
karşılık gelmesini zorunlu kılar. Sesin görüntüdeki dudak hareketiyle
örtüşmediği durumlar (dublaj, konuşmacı dışı ses) bu varsayımı ihlal eder.
Bu ihlali tespit etmek için, her bölümlenmiş parçada, ağız açıklığının zaman
içindeki değişimi ile ses sinyalinin enerjisinin zaman içindeki değişimi
arasındaki istatistiksel korelasyon hesaplanır. Bu korelasyon bir eşiğin
altındaysa, ilgili parça (gerçek konuşmayla dudak hareketinin örtüşmediği
varsayılarak) veri kümesinden çıkarılır.

### 7.6 Konuşmacı Ayrımı

Sonuç 6.2, test kümesinin eğitim kümesiyle örtüşmeyen özellikler taşımasını
gerektirir. Bu projede birincil örtüşme riski, **aynı konuşmacının** hem
eğitimde hem testte bulunmasıdır: bu durumda model, o kişinin yüz yapısını veya
konuşma tarzını ezberleyerek yüksek test başarımı gösterebilir, ki bu, dudak
okuma yeteneğini değil, kişi tanımayı ölçer. Bu riski ortadan kaldırmak için,
bölümleme **video kimliği bazında** yapılır: tek bir videonun tüm parçaları,
aynı kümeye (yalnızca eğitime veya yalnızca teste) atanır. Bu atama, tekrarlanabilir
olması için video kimliğinin bir özet (hash) değerine dayalı, belirlenimci bir
kural ile yapılır.

**Sonuç 7.1:** Bölüm 7.1–7.6'nın tamamı uygulandığında, elde edilen çıktı,
her biri `(ağız videosu, doğru metin, zaman damgaları)` üçlüsünden oluşan,
etiket doğruluğu kontrol edilmiş ve konuşmacı bazında ayrılmış bir veri
kümesidir. Bu, Tanım 1.1'in ikinci bileşenini karşılar.

---

## 8. Bu Projede İncelenen İki Model Örneği

Bölüm 3–5, bir modelin genel yapısını (mekânsal işleme → zamansal işleme →
çıktı üretimi) türetti. Bu bölüm, bu genel yapının iki somut gerçeklemesini,
önceki tanımlardan hareketle açıklar.

### 8.1 Kapalı-Küme Örneği

Bu model, Bölüm 5.1'deki sınıflandırma yöntemini kullanır; çıktı kümesi,
önceden belirlenmiş sabit sayıda kelimedir.

Mekânsal işleme katmanında (Bölüm 3), parametre sayısını azaltmak amacıyla
(Sonuç 3.2'de belirtilen problem) özel bir teknik kullanılır:

**Tanım 8.1 (Ayrıştırılmış evrişim):** Standart bir evrişim, tek bir işlemde
hem mekânsal komşuluğu hem tüm girdi/çıktı kanal kombinasyonlarını işler. Bu
işlem iki ayrı adıma bölünebilir: (a) her kanalın **kendi içinde**, diğer
kanallarla etkileşim olmadan mekânsal evrişimi, (b) ardından, mekânsal
boyutu olmayan, yalnızca kanallar arasında karışım yapan 1×1 boyutlu bir
evrişim. Bu iki adımın toplam parametre sayısı, standart tek adımlı evrişime
göre çarpımsal olarak (kanal sayısı mertebesinde) daha azdır.

**Sonuç 8.1:** Bu ayrıştırma, modelin toplam parametre sayısını (dolayısıyla
depolama boyutunu ve hesaplama miktarını) azaltır; bu, modelin sınırlı
işlemci/bellek kapasitesine sahip bir cihazda çalıştırılmasını mümkün kılar
(Bölüm 9'da bu gereksinim türetilecektir).

### 8.2 Açık-Küme Örneği

Bu model, Bölüm 5.2'deki dizi üretimi yöntemini kullanır. Mekânsal işleme
katmanında standart (ayrıştırılmamış) üç boyutlu evrişim; zamansal işleme
katmanında Bölüm 4'te tanımlanan öz-dikkat tabanlı kodlayıcı (Conformer)
kullanılır. Çıktı üretimi, Bölüm 5.2'deki sembol-sembol üretim mekanizmasını
gerçekleyen bir bileşenle (Transformer çözücü) yapılır.

**Sonuç 8.2:** Bu modelin toplam hesaplama maliyeti, Önerme 4.1 (öz-dikkatin
karesel maliyeti) ve Önerme 5.1'in (dizi üretiminin çok-adımlılığı) birleşik
etkisiyle, Bölüm 8.1'deki modele göre önemli ölçüde daha yüksektir.

---

## 9. Hesaplamanın Fiziksel Konumu

Bölüm 1–8, `f_θ(x)` hesaplamasının **ne olduğunu** tanımladı. Bu hesaplama,
gerçekte bir işlemci üzerinde yürütülür. İşlemcinin konumu için iki seçenek
vardır: kullanıcının kendi cihazı, veya ağ üzerinden erişilen bir sunucu.

**Önerme 9.1:** Kullanıcının cihazında hesaplama yapılırsa, girdinin ağ
üzerinden gönderilmesi gerekmez; buradan, ağ iletim süresi toplam gecikmeye
eklenmez. Ancak cihazın işlemci hızı ve belleği sınırlıdır; bu sınır, modelin
parametre sayısına ve hesaplama miktarına bir üst tavan koyar.

**Önerme 9.2:** Sunucuda hesaplama yapılırsa, bu tavan büyük ölçüde
gevşer (sunucu donanımı, tipik bir mobil cihazdan çok daha güçlüdür).
Ancak girdinin sunucuya, çıktının kullanıcıya iletilmesi ağ üzerinden
yapılır; bu iletim, sıfırdan büyük bir süre gerektirir. Ayrıca, sunucu
tarafında hesaplama kaynağının (bu projede grafik işlemci, GPU) her istek
için **tahsis edilmesi** gerekir; bu tahsisin kendisi bir süreçtir ve
Bölüm 10'da ele alınacaktır.

**Sonuç 9.1:** Bölüm 8.1'deki model (düşük parametre sayısı, Sonuç 8.1), Önerme
9.1'deki tavanı karşılayabilir; cihaz-üstü çalıştırmaya uygundur. Bölüm 8.2'deki
model (Sonuç 8.2'de belirtilen yüksek hesaplama maliyeti), bu tavanı büyük
olasılıkla aşar; sunucu-tarafı çalıştırma gerektirir.

**Tanım 9.1 (Niceleme/Quantization):** Cihaz-üstü çalıştırmada kullanılan ek
bir teknik: model parametreleri normalde yüksek hassasiyetli sayılarla (32-bit
kayan noktalı) saklanır. Bu sayılar, daha düşük hassasiyetli bir gösterime
(örn. 8-bit tam sayı) yuvarlanabilir. Bu işlem, depolama boyutunu (bit sayısı
oranında, yaklaşık 4 kat) azaltır ve bazı donanımlarda düşük hassasiyetli
aritmetik daha hızlı yürütüldüğünden hesaplama süresini de kısaltabilir.
Hassasiyet azaltıldığından, `f_θ(x)` çıktısı orijinal modele göre küçük ölçüde
sapabilir; bu sapmanın kabul edilebilirliği ölçülmelidir.

---

## 10. Sunucu Tarafı Kaynak Tahsisi

Bölüm 9, sunucu-tarafı hesaplamanın kaynak tahsisi gerektirdiğini belirtti. Bu
bölüm, tahsis sürecini ve onun gecikmeye etkisini türetir.

**Tanım 10.1 (İstek-bazlı tahsis / "serverless"):** Bir kaynak tahsisi modeli;
işlemci (GPU dahil), yalnızca bir istek geldiğinde tahsis edilir ve bir süre
boşta kaldıktan sonra serbest bırakılır. Kullanıcı, yalnızca tahsis edilen
sürenin karşılığını öder.

**Önerme 10.1:** `f_θ(x)` hesaplanabilmesi için, model parametrelerinin
tahsis edilen işlemcinin belleğine **yüklenmiş** olması gerekir. Bu yükleme
işlemi sıfırdan büyük bir süre alır (parametre dosyasının boyutuyla orantılı
olarak, ayrıca işlemcinin çalışma ortamının hazırlanması —bellek ayırma, sürücü
başlatma— için ek bir süre).

**Tanım 10.2 (Soğuk başlangıç):** Bir istek, önceden tahsis edilmiş ve hâlâ
etkin bir kaynak bulamazsa, Önerme 10.1'deki yükleme süresinin tamamı, o
isteğin toplam yanıt süresine eklenir. Bu duruma soğuk başlangıç denir.

**Tanım 10.3 (Sıcak istek):** Önceden tahsis edilmiş bir kaynak hâlâ etkinse
(daha önce serbest bırakılmamışsa), yeni istek bu kaynağı doğrudan kullanır;
Önerme 10.1'deki yükleme süresi tekrarlanmaz.

**Sonuç 10.1:** Soğuk başlangıç süresini azaltmanın, Tanım 10.2'nin
yapısından **doğrudan türeyen** iki yöntemi vardır:

1. **Yüklenecek veri miktarını azaltmak.** Tanım 9.1'deki niceleme, parametre
   dosyası boyutunu küçülterek bu süreyi kısaltır.
2. **Yükleme işlemini tekrarlamamak.** Eğer işlemcinin belleğinde model
   yüklenmiş ve çalışmaya hazır durumdaki tam durumu (bellek içeriği), bir
   önceki tahsisin sonunda **kaydedilir** ve yeni bir tahsiste bu kayıttan
   doğrudan **geri yüklenirse**, Önerme 10.1'deki yükleme adımlarının
   tamamı (dosya okuma, bellek ayırma, hesaplama biriminin ısınması) atlanmış
   olur. Bu tekniğe **bellek anlık görüntüsü (memory snapshot)** denir.

**Sonuç 10.2 (Sınırlama):** Sonuç 10.1'in 2. yöntemi, kaydedilen durumun **fiziksel
olarak aynı veya erişimi hızlı bir depolama biriminde** bulunmasını gerektirir.
İstek-bazlı tahsis modelinde (Tanım 10.1), hangi fiziksel işlemcinin bir isteğe
atanacağı kullanıcı tarafından belirlenemez. Kayıtlı durum, atanan işlemcinin
yerel deposunda bulunmuyorsa, onu oradan getirmek de bir süre alır; bu durumda
Sonuç 10.1'in 2. yönteminin kazancı azalır veya ortadan kalkar. Buradan:

**Önerme 10.2:** Bellek anlık görüntüsü tekniği, soğuk başlangıç süresini
**olasılıksal olarak** azaltır; kesin bir üst sınır garantisi vermez. Kesin
garanti için, kaynağın hiç serbest bırakılmaması (sürekli tahsisli tutulması)
gerekir — bu ise Tanım 10.1'in "yalnızca kullanılan süre kadar öde" avantajını
ortadan kaldırır ve sabit bir maliyete dönüşür.

---

## 11. Ölçüm: Bölüm 1–10'daki Önermelerin Deneysel Sınanması

Bu bölüm, önceki bölümlerde türetilen önermelerin, gerçek bir sunucu-tarafı
konuşlandırmada (Bölüm 8.2'deki açık-küme model, İngilizce ön-eğitilmiş
ağırlıklarla) ölçülmesini raporlar. Girdi: kamu malı bir konuşma videosundan
kesilmiş 2/6/12/24 saniyelik klipler; işlemci: üç farklı GPU tipi.

### 11.1 Önerme 4.1'in Sınanması (Zamansal Karmaşıklık)

Klip uzunluğu 2 sn'den 24 sn'ye (12 katına) çıkarıldığında, ölçülen işlem
süresi yaklaşık 18–24 katına çıkmıştır (L4 işlemcisinde 0.48 sn → 10.34 sn).
Bu artış oranı, girdi uzunluğuyla **doğrusal** değil, ondan **hızlı**dır;
Önerme 4.1'de türetilen karesel büyüme öngörüsüyle tutarlıdır.

**Ek sonuç:** 24 saniyelik klipte, bir GPU tipinde (T4) ölçülen işlem süresi
(24.8 sn), klibin kendi süresini (24 sn) aşmıştır. Bu, sistemin o girdi
uzunluğunda ve donanımda **gerçek zamanın gerisine düştüğünü** gösterir —
Bölüm 7.4'te türetilen "girdiyi sınırlı uzunlukta parçalara bölme"
gerekliliğinin deneysel doğrulamasıdır.

### 11.2 Sonuç 10.1'in Sınanması (Soğuk Başlangıç Azaltma)

Bellek anlık görüntüsü tekniği (Sonuç 10.1, yöntem 2) ve Bölüm 10'da
türetilen ek bir teknik — hesaplama biriminin ısınma işleminin, ilk isteğin
yanıt süresi yerine tahsis anına taşınması — uygulanmıştır.

| Ölçüm koşulu | Soğuk başlangıç süresi |
|---|---|
| Teknik uygulanmadan | 33,6 saniye |
| Teknik uygulandıktan sonra, en iyi durum | 10,5 saniye |
| Teknik uygulandıktan sonra, en kötü durum | 27,4–35,5 saniye |

Bu sonuç, Önerme 10.2'yi doğrulamaktadır: kazanç sabit değildir, **fiziksel
işlemcinin kayıtlı duruma erişim hızına bağlıdır** (Sonuç 10.2'de türetilen
sınırlama). Beş ölçümden yalnızca biri en iyi durumu göstermiştir; bu,
tekniğin **olasılıksal** doğasının (Önerme 10.2) doğrudan gözlemidir.

### 11.3 Model Karşılaştırması (Sonuç 9.1'in Sınanması)

| | Bölüm 8.1 modeli (kapalı-küme) | Bölüm 8.2 modeli (açık-küme) |
|---|---|---|
| Parametre sayısı (göreli) | Düşük (Sonuç 8.1) | Yüksek |
| Ölçülen doğruluk (sabit kelime kümesinde) | %73 (500 sınıf üzerinden) | — |
| Ölçülen hata oranı (serbest cümlede) | — | %18,7–30 (harf/kelime bazlı) |
| Çalıştırma konumu | Cihaz üzerinde (~6 MB, niceleme sonrası) | Sunucu (yukarıdaki ölçümler) |

Bu tablo, Sonuç 9.1'in öngördüğü ayrımı doğrudan yansıtır: düşük parametre
sayılı model cihazda, yüksek parametre sayılı model sunucuda çalıştırılmıştır.

---

## 12. Projenin Mimari Kararının Türetilmesi

Bu son bölüm, Bölüm 1–11'de kurulan tüm önermelerden, projenin ilk aşama
(MVP) mimari kararının nasıl **zorunlu olarak** takip ettiğini gösterir.

**Öncül 1 (Sonuç 7.1'den):** Bu projenin veri toplama hızı (YouTube kaynaklı,
otomatik pipeline ile), sınırlı ve ölçülmüş bir orandadır (işlenen video
başına ortalama %15–40 kullanılabilir segment oranı).

**Öncül 2 (Önerme 5.1'den):** Açık-küme yöntemi, kapalı-küme yöntemine göre
çıktı başına orantısız derecede fazla hesaplama ve (dolaylı olarak, çünkü
daha büyük bir fonksiyon uzayını kısıtlamak için) orantısız derecede fazla
eğitim verisi gerektirir.

**Öncül 3 (Sonuç 9.1'den):** Kapalı-küme model, cihaz üzerinde çalıştırılabilir;
bu, Önerme 9.1'in belirttiği gibi ağ gecikmesini ortadan kaldırır ve Önerme
10.1–10.2'de türetilen soğuk-başlangıç problemini **tanım gereği** hiç
oluşturmaz (kaynak tahsisi, kullanıcının kendi cihazında zaten süreklidir).

**Öncül 4 (Bölüm 11.2'den):** Açık-küme modelin sunucu-tarafı konuşlandırması,
ölçülmüş olarak, soğuk başlangıçta 10,5–35,5 saniye, sıcak durumda klip
uzunluğuna bağlı olarak katlanarak artan bir gecikme taşımaktadır.

**Sonuç (Öncül 1–4'ten):** Öncül 1, açık-küme yöntemi için Öncül 2'de belirtilen
veri hacmine mevcut hızla ulaşmayı zaman açısından elverişsiz kılar. Öncül 3 ve
4 birlikte, kapalı-küme + cihaz-üstü kombinasyonunun, açık-küme + sunucu
kombinasyonuna göre daha düşük gecikme ve daha düşük sürekli maliyetle
sonuçlandığını göstermektedir. Bu nedenle, ilk aşama (MVP) mimarisi olarak
kapalı-küme, cihaz-üzerinde çalışan model (Bölüm 8.1 sınıfı) seçilmiştir.
Açık-küme, sunucu-tarafı model (Bölüm 8.2 sınıfı), yalnızca Öncül 1'deki veri
hacmi kısıtı ileride değiştiğinde (örn. rızalı veri toplama ile veri hacmi
artırıldığında) yeniden değerlendirilecek bir sonraki aşama adayı olarak
kaydedilmiştir.

---

## Ek: Tanım ve Önermelerin Dizini

| No | İfade | Bulunduğu Bölüm |
|---|---|---|
| Tanım 1.1 | Öğrenme = aday fonksiyon ailesi + veri kümesi + eğitim | 1 |
| Tanım 2.1–2.2 | Dijital görüntü ve video | 2 |
| Tanım 3.1 | Evrişim katmanı | 3 |
| Önerme 3.1 | Katman yığınının alım alanını büyütme zorunluluğu | 3 |
| Tanım 4.1 | Zamansal model (TCN / öz-dikkat) | 4 |
| Önerme 4.1 | Öz-dikkatin karesel hesaplama maliyeti | 4 |
| Tanım 5.1–5.2 | Kapalı küme / açık küme | 5 |
| Önerme 5.1 | Açık kümenin çok-adımlı hesaplama maliyeti | 5 |
| Tanım 6.1–6.2 | Kayıp fonksiyonu, eğitim | 6 |
| Sonuç 6.2 | Genelleme ve test kümesi ayrımı gerekliliği | 6 |
| Bölüm 7.1–7.6 | Veri kümesi inşa adımları (her biri önceki sonuçlardan türetilmiş) | 7 |
| Tanım 8.1 | Ayrıştırılmış evrişim | 8 |
| Önerme 9.1–9.2 | Cihaz-üstü / sunucu-tarafı hesaplama ödünleşimi | 9 |
| Tanım 9.1 | Niceleme (quantization) | 9 |
| Tanım 10.1–10.3, Önerme 10.1–10.2 | İstek-bazlı kaynak tahsisi, soğuk/sıcak başlangıç | 10 |
| Bölüm 11 | Deneysel doğrulama | 11 |
| Bölüm 12 | Mimari kararın türetilmesi | 12 |

**İlgili kaynaklar:** [`LATENCY_RAPORU.md`](LATENCY_RAPORU.md) (Bölüm 11'in tam
ölçüm detayı), [`GPU_KURULUM_POLITIKALARI.md`](GPU_KURULUM_POLITIKALARI.md)
(Bölüm 10'daki tekniklerin uygulama detayı), [`benchmark/`](benchmark/) (ham veri
ve ölçüm scriptleri), [`MIMARI.md`](MIMARI.md) ve
[`MODEL_KARSILASTIRMASI.md`](MODEL_KARSILASTIRMASI.md) (bu dokümanın daha
önceki, analoji temelli anlatım sürümleri).
