# `full_train_manifest.json` — Provenance Notu (tarihsel kanıt, çalıştırılabilir reçete değil)

Bu dosya `full_train_manifest.json` dosyasını değiştirmez. Onun neyi kanıtlayıp
neyi kanıtlamadığını, git history üzerinden doğrulanmış haliyle belgeler.
Manifesto kendisi byte-identical korunur (frozen historical evidence).

## 1. Kod revizyonu gerçeği (uydurma yok)

- Manifestteki `code_revision = 63da203c9117...` commit'i git history'de **yalnızca
  bir dokümantasyon plan dosyası** içerir (`docs/superpowers/plans/2026-09-17-*.md`,
  807 satır). Doğrulanabilir: `git show --stat 63da203`.
- Deneylerde kullanılan kodun tamamı (`src/`, `run_experiment.py`, `configs/`,
  `research/`) o sırada **uncommitted working tree**'deydi; ilk kez `ff211b0`
  commit'inde arşivlendi (`git log --oneline`: `ff211b0` -> `63da203` -> ...).
- Sonuç: `63da203` **tek başına deney kodunu yeniden üretmez.** Bu commit'i
  "deneyi yeniden üretir" şeklinde sunmak yanlıştır. Deney kodu için en yakın
  arşiv `ff211b0` ağacıdır — ancak o da handover düzenlemelerini içerir
  (secret temizliği, downloader varsayılanı, `.gitignore`, `HANDOVER.md`,
  `LICENSE`, `README`), yani deney-anı ağacıyla birebir aynı değildir.

## 2. Canlı doğrulama sonuçları (ff211b0 ağacında, `shasum -a 256` ile)

| Artefakt | Manifest/candidate iddiası | Canlı sonuç | Durum |
|---|---|---|---|
| `configs/research_candidate.yaml` | `candidate_recipe_sha256 = 0eae4340...` (64 hex tam) | bayt-bayt aynı | EŞLEŞTİ — frozen reçete arşivde bozulmamış |
| `data/metadata/train.csv` | `dataset_sha256 = f5e924fa...` | ilk 16 hex aynı, tam hash FARKLI; satır sayısı doğru (1681+başlık) | EŞLEŞMEDİ — provenance boşluğu (bkz. §4.5) |
| `data/metadata/val.csv` | `split_sha256 = cfb9e3f8...` | ilk 16 hex aynı, tam hash FARKLI; satır sayısı doğru (385+başlık) | EŞLEŞMEDİ — provenance boşluğu (bkz. §4.5) |
| `data/metadata/test.csv` | candidate test hash `1a0f16d9...` | ilk 16 hex aynı, tam hash FARKLI; satır sayısı doğru (617+başlık); kanallar train/val ile canlı kesişimsiz (sızıntı yok) | EŞLEŞMEDİ (hash) ama SIZINTI YOK — karantina içerik olarak sağlam, hash kanıtı eksik |
| `checkpoints/*.pt` (initializer) | `initializer = probe_confirm001_full_train_scaling_best.pt` | dosyalar repoda YOK | DOĞRULANAMADI — provenance boşluğu (bkz. §4.2) |

Önemli: hash önekleri (`f5e924fa22...` vb. D1'de alıntılanan öneklerle aynı)
ve satır sayıları tutarlıdır; ancak tam-hash recompute tutmaz. Satır-sonu
(CRLF/LF) normalizasyonu ve başlıksız varyantlar denendi, eşleşme bulunamadı.
Muhtemel neden: CSV'ler hash kaydından sonra dirty tree içinde yeniden
üretilmiştir. Orijinal baytlar kurtarılamaz; uydurma yapılmaz.

## 3. Makineye-özgü mutlak yol haritası

Manifestteki `candidate_recipe_path` değeri (`/Users/.../configs/research_candidate.yaml`)
manifestonun yazıldığı makinenin mutlak yoludur ve başka makinede geçersizdir.
Manifesto tarihsel kanıt olduğu için bu alan **düzeltilmez**. Tooling
(`src/experiments/research_governance.py`, `src/full_training.py`) bu alanı
güvenilir yol olarak kullanmaz; bunun yerine:

1. Repo köküne göre `configs/research_candidate.yaml` dosyasını çözer,
2. `candidate_recipe_sha256` ile bayt-bayt doğrular,
3. Eşleşmezse fail-closed durur.

## 4. Hâlâ çözülemeyen boşluklar (dürüst sınır)

1. **Deney-anı kod ağacının tam snapshot'ı yoktur.** Dirty-tree ile çalışıldığı
   için `probe_*` kayıtlarındaki kod durumu hash ile sabitlenmemiştir.
   Bundan sonra üretilemeyecek kanıtlar uydurulmaz.
2. **Initializer `.pt` dosyası repoda/HF'te adreslenmemiştir.** `checkpoints/`
   yalnız README içerir. Full training için initializer'ın nereden (HF Hub /
   Modal volume) geleceği `c0.5.0` fazında adreslenmelidir.
3. **Tarihsel manifesto çalıştırılamaz.** `code_revision` güncel clean HEAD ile
   eşleşmediği için preflight onu fail-closed reddeder. Bu beklenen ve doğru
   davranıştır; manifesto kanıt olarak kalır.
4. **Bundan sonraki mühürleme/confirmation/full-training** işlemleri
   `require_clean_code_revision` ile korunur: dirty tree varsa işlem başlamadan
   durur (`src/experiments/research_governance.py`).
5. **Split CSV hash'leri arşivden recompute ile tutmaz** (§2). `dataset_hash`,
   `split_hash` ve karantina-hash kontrolleri tarihsel manifestoda fail-closed
   kalır (doğru davranış). Yeni mühürlemelerde hash'ler aynı dosyalardan canlı
   üretildiği için bu kontroller geçer.
