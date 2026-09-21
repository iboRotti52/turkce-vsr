# vendor/auto_avsr

Auto-AVSR reposundan alınan ağız-kırpma (mouth ROI) kodu.

Kaynak: https://github.com/mpc001/auto_avsr
        `preparation/detectors/mediapipe/`
Lisans: Apache 2.0 — Copyright 2023 Imperial College London (Pingchuan Ma)

## Neden kopyalandı?

Auto-AVSR fine-tuning için 96x96, ortalama-yüze hizalanmış ağız kırpması
gerekiyor. Bu hizalamayı yeniden yazmak yerine orijinal (referans) kodu
kullanmak, ürettiğimiz verinin modelin beklediğiyle birebir aynı olmasını
garanti eder.

## Dosyalar
- `video_process.py` — affine hizalama + 96x96 kırpma (değiştirilmedi)
- `20words_mean_face.npy` — 68 noktalı referans yüz; hizalama hedefi

## Nasıl kullanılıyor?
`video_process.VideoProcess` kare başına 4 landmark ister, SIRASIYLA:
sağ göz, sol göz, burun ucu, ağız merkezi. Bunları `pipeline.extract_faces`
üretir ve `faces.json` içinde `stable4` alanında saklar.

## Yapılan tek değişiklik

`video_process.py` içinden `from skimage import transform as tf` import'u ve
onu kullanan `warp_img` / `apply_transform` fonksiyonları **silindi**.

Gerekçe: bu iki fonksiyon dosyada hiçbir yerden çağrılmıyordu (ölü kod);
asıl hizalama `cv2.estimateAffinePartial2D` + `cv2.warpAffine` ile yapılıyor.
Sadece bunlar için scikit-image + scipy kurmak gereksiz ağırlık olurdu.
Hizalama/kırpma mantığı **hiç değiştirilmedi**.
