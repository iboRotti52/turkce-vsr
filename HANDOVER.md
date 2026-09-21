# Devir Notu — Küçük Veriden Büyük Veriye (Arkadaşa)

Bu repo, az veriyle öğrenilebilecek her şeyin öğrenildiği noktanın dondurulmuş halidir.
Sen büyük veriyle devam edeceksin. Aşağıdaki 3 şeyi bilmen yeterli:

## 1. Nerede kaldık (c0.4.0, READY_FOR_FULL_TRAIN)

- Kanonik model: `research/CANDIDATE.md` + çalıştırılabilir reçete `configs/research_candidate.yaml` (`candidate_version: c0.4.0`, `recipe_status: frozen`).
- Küçük-veri rejimi: 2.683 kabul klip / 4.38 saat, 5 train + 1 val + 3 test konuşmacısı (konuşmacı-ayrık).
- 12/12 readiness kapısı geçti (`research/RESEARCH_STATE.md`), full training kullanıcı talimatıyla başlatılmadan duruldu.
- En önemli bulgu (D20–D23): **5 konuşmacılı rejimde görsel varyans 2. epoch'ta tükeniyor.** Aynı konuşmacılardan daha fazla video eklemek Val Loss'u düşürmüyor (D22: 356 uzun klip eklendi, 2.7530 vs baseline 2.7399). Yeni veride öncelik **konuşmacı çeşitliliği**.
- Karar geçmişi: `research/DECISIONS.md` (D1–D23), ham deney kayıtları: `experiments/registry.jsonl` (32 probe), hata kümeleri: `research/FAILURE_ANALYSIS.md`.
- Test kümesi (617 klip, 3 held-out konuşmacı) **karantinada** — araştırma kararlarına geri besleme yapma (`GEMINI.md` §9).
- Kalan bulut bütçesi: ~22.27 USD / 25 USD.

## 2. Büyük veri ile başlarken (yapılacaklar)

Yeni veri: **https://huggingface.co/datasets/avsr-tr-ekip/avsr-tr-dataset** (~275 MB).
Kodun varsayılanı artık bu (`src/data/hf_downloader.py` → `DEFAULT_REPO_ID = "avsr-tr-ekip/avsr-tr-dataset"`).
Eski küçük-veri ID'si (`iboRotti/avsr-tr-dataset`) yalnızca frozen kanıtlarda referans olarak durur —
`configs/research_candidate.yaml`, `full_train_manifest.json`, `data/metadata/*` dosyalarına dokunma, bunlar tarihsel kanıt.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # HF_TOKEN'ını .env'e yaz
.venv/bin/python -m pytest -q

# Canonical full-training preflight (ücretli işlem başlatmaz):
.venv/bin/python -m src.full_training --manifest full_train_manifest.json
# Tarihsel c0.4.0 manifestosu bilerek FAIL verir (kanıt, çalıştırılamaz);
# detay: full_train_manifest.PROVENANCE.md

# Büyük veriyi çek (repo_id parametrik, eski veriye de dönebilirsin):
.venv/bin/python - <<'EOF'
from src.data.hf_downloader import HFDatasetDownloader
d = HFDatasetDownloader(repo_id="avsr-tr-ekip/avsr-tr-dataset")
print(d.repo_id)
EOF
```

Zorunlu adımlar (atlama):
1. Yeni veri için **konuşmacı-ayrık split'i sıfırdan kur** (`split_map_iborotti.json` yalnız eski 15 videoyu kapsar — yeni konuşmacıları eski split'e karıştırma, sızıntı olur).
2. Yeni candidate sürümü aç (c0.5.0): eski `c0.4.0` reçeteyi başlangıç noktası al, veri/split hash'lerini, seed'i, initializer kökenini her checkpoint'e yaz (`GEMINI.md` §5D).
3. Curriculum'u koru: `<=3.5s -> <=6s -> <=8s` (blank collapse kırıcı, D4/D6/D8).
4. Decoder: Greedy `blank_penalty=1.2` + kalibre beam; KWS spotter `BP=1.2, conf=0.15` (D11/D14/D15).
5. Bilinen HF uyarısı: dataset viewer şu an `CastError` veriyor (kolon şema uyumsuzluğu) — `datasets` ile doğrudan parquet/JSON okuyarak doğrula.

## 3. Güvenlik ve repo düzeni

- Bu repo **public** (`turkce-vsr`). `.env` asla commitlenmez (`.gitignore`'da). Eski AWS anahtarları `.env.example`'dan temizlendi
  çünkü gerçek key'ler içindeydi — **AWS konsoldan o anahtarları rotate/deactivate et** (AKIAXRCV…).
- GitHub'a video/checkpoint gitmez: `data/iborotti/*`, `data/master/*`, `*.mp4`, `*.pt/.pth/.safetensors` ignore'lu.
  Ağır dosyalar HF Hub'da, deney kanıtları `experiments/registry.jsonl`'da.
- `.agents/skills/` (217 dosya, 2.3 MB) bilerek dahil — AI asistanla (`/goal` + `GEMINI.md` protokolü) birebir devam için.
  Manuel çalışıyorsan en az `turkish-lip-reading` runbook'unu oku (visem haritası, 25 FPS, 96x96 ROI, CTC 31 token).
- Otonom araştırma protokolü: `GEMINI.md`. Tek yaşayan model kuralı geçerli — paralel pilot modeller açma,
  her probe'u tek soruya bağla ve `ACCEPT/REJECT/INCONCLUSIVE` kaydet.


## 4. Large-data araştırma hazırlığı

50–100 saatlik fazda ham video preprocessing eklenmez; Hugging Face'teki preprocessed
mouth clips kaynak kabul edilir. Fakat **`main`/latest revision ile araştırma koşulmaz**.
Önce immutable HF dataset commit SHA sabitlenir ve yalnız manifest metadata'sı üzerinden
yeni split/stage planı hazırlanır:

```bash
.venv/bin/python -m src.data.prepare_large_data \
  --repo-id avsr-tr-ekip/avsr-tr-dataset \
  --revision <40-hex-HF-dataset-commit-sha>
```

Bu komut full dataset'i veya GPU eğitimini başlatmaz. Şunları üretir:

- `research/large_data_plan.json`: dataset revision, split/stage özetleri ve hash'ler.
- `data/metadata/split_map_large_data.json`: sıfırdan üretilmiş speaker-disjoint split.
- Train içinde mümkün olduğu kadar speaker-diverse ve **nested** 10h → 25h → 50h →
  100h aşamaları (mevcut train süresi hedefe yetmiyorsa o aşama üretilmez; `full`
  her zaman gerçek kullanılabilir train kümesini temsil eder).

Large-data candidate `c0.5.0`, bu plan üretildikten ve audit edildikten sonra açılır.
`c0.4.0` dosyaları historical frozen evidence olarak kalır; onları yeni dataset'e
uyarlayarak üzerine yazma.

### Large-data training kuralları

- Research/selection yalnız train + speaker-disjoint validation kullanır; test split
  araştırma sırasında indirilmez/değerlendirilmez.
- Her hipotezi doğrudan 50–100 saatte koşma. En ucuz speaker-diverse stage'de başla;
  yalnız önceden yazılmış karar kuralı geçerse daha büyük stage'e ölçekle.
- `SequenceBucketSampler` ile duration-aware batching kullan; 50–100 saat için tüm
  videoları RAM'e preload etme.
- Uzun koşuların checkpoint'i model dışında optimizer, scheduler, AMP scaler, epoch,
  global step, sampler epoch, RNG state ve provenance taşımalıdır
  (`src/training_state.py`).
- Resume sırasında dataset id/revision, split hash, code revision veya candidate
  version değişmişse fail-closed dur; farklı koşuyu aynı run gibi sürdürme.
- Full training ancak yeni veri rejiminde readiness kapıları yeniden geçilip yeni
  candidate recipe dondurulduktan sonra başlatılabilir.
