---
name: budget-conscious-research
description: Sınırlı bulut bütçesiyle ($30 Modal) çalışan ML/AI projelerinde küçük deneylerden maksimum bilgi çıkarımı ve uzun vadeli stratejik araştırma kuralı
---

# 💡 Bütçe Korumalı Araştırma ve Bilgi Maksimizasyonu Kuralı

Bu kural, sınırlı hesaplama bütçesi (GPU kredisi, API limiti) ile yürütülen tüm makine öğrenimi ve model geliştirme süreçlerinde zorunludur:

1. **Önce Küçük Teşhis Probu (Diagnostic Probe First):**
   - Asla doğrudan büyük veri veya yüksek epoch'lu bir bulut eğitimine başlama.
   - Önce 20-50 segmentlik mikro deneylerle loss düşüşü, gradyan akışı ve bellek kullanımını test et.
   - 10-15 dakikalık ucuz bir denemeden öğrenilebilecek bir bilgiyi 2 saatlik eğitimle öğrenme.

2. **Her Deneyden Maksimum Bilgi Çıkarımı (High Information Gain):**
   - Sadece loss sayısına bakma; modelin neyi yanlış tahmin ettiğini incele (örneğin: model sadece blank mi basıyor? Visem eşlemesi neden kaçıyor? Diferansiyel LR gerekli mi?).
   - Her deneyin sonucunu (beklenti, sürpriz, güncellenen inanç) `experiments/registry.jsonl` kütüğüne kaydet.

3. **Büyük Resmi Görerek Adım At (Macro Picture Alignment):**
   - Rastgele hiperparametre denemeleri (trial-and-error) yerine, literatürdeki kanıtlanmış temelleri (Auto-AVSR transfer learning, dondurulmuş omurgalar, BiGRU zamansal tümevarım yanlılığı) birleştir.
   - Her küçük deneyin çıktısını bir sonraki mimari kararın gerekçesi yap.

4. **Kredi Tüketimini Adım Adım Takip Et:**
   - Her bulut koşusunun tahmini maliyetini önceden hesapla ($/saat x süre).
   - $30 bütçe tükenmeden nihai hedef modele ulaştıracak en ekonomik patikayı koru.
