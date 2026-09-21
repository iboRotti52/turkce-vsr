#!/usr/bin/env bash
# bulut_kosu.sh — egitimi LAPTOP'TAN VE MAKINEDEN BAGIMSIZ calistirir.
#
# Iki ayri tehlike var, ikisi de karsilaniyor:
#
#  1. SSH kopmasi / laptop kapanmasi
#     -> egitim nohup (+ varsa setsid) ile ayrilir; terminal olse de surer.
#
#  2. SPOT INSTANCE GERI ALINMASI  <-- asil tehlike
#     -> makine tamamen yok olur, diskteki checkpoint'lerle birlikte.
#        --resume auto diskte dosya varsa ise yarar; disk yoksa yaramaz.
#        Bu yuzden last.ckpt periyodik olarak S3'e senkronlanir ve
#        YENI makinede bu script once S3'ten geri ceker.
#
# Yani: makine olse bile yeni bir instance'ta ayni komut kaldigi yerden devam eder.
#
# Kullanim:
#   bash bulut_kosu.sh            # baslat (ya da kaldigi yerden devam et)
#   bash bulut_kosu.sh durum      # ilerlemeyi goster
#   bash bulut_kosu.sh durdur
set -euo pipefail

AD="${AD:-tr131}"          # MANIFEST oneki (data/labels/${AD}_*.csv)
# ponytail: KOSU adi AD'den AYRI. Ayni veriyle ikinci bir deney yapinca
# --resume auto eskisini bulup DEVAM ETTIRIYORDU -- yeni kosu sanip eski
# kosuyu uzatmak sessiz ve pahali bir hata. Artik deney adi ayri.
KOSU="${KOSU:-$AD}"        # deney adi (exp/, S3 yolu, surec eslesmesi)
EPOCH="${EPOCH:-20}"       # max epoch
EK="${EK:-}"               # finetune_tr.py'ye ek bayraklar, orn: EK="--flip"
KOVA="s3://lipreading-data-emre2026"
UZAK="$KOVA/kosu/$KOSU"
YEREL="exp/$KOSU"
LOG="$YEREL/kosu.log"
ARALIK="${ARALIK:-1200}"        # S3 senkron araligi (sn). 1200 = 20 dk.
PIDF="$YEREL/.pid"
MF="${MF:-1200}"

cd "$(dirname "$0")"

# ponytail: pid DOSYASINA guvenme -- yanlislikla ikinci kez baslatilinca
# dosya olen surecin numarasiyla eziliyor ve durum() "calismiyor" diye YALAN
# soyluyor (olculdu 2026-09-04: egitim 5s54d'dir kosarken durum calismiyor dedi).
# Surec tablosu gercegin kaynagi. Worker'lar da ayni komut satirini tasiyor,
# o yuzden ebeveyni init (ppid=1) olani ariyoruz -- nohup ile ayrilan ana surec.
ana_surec() {
    for pid in $(pgrep -f "finetune_tr.py.*--exp-name $KOSU" 2>/dev/null); do
        [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "1" ] && { echo "$pid"; return 0; }
    done
    return 1
}

durum() {
    echo "=== $KOSU  (manifest: $AD) ==="
    local pid
    if pid=$(ana_surec); then
        echo "durum: CALISIYOR (pid $pid)"
        [ "$(cat "$PIDF" 2>/dev/null)" != "$pid" ] && {
            echo "$pid" > "$PIDF"; echo "  (.pid bayatti, duzeltildi)"; }
    else
        echo "durum: calismiyor"
    fi
    [ -f "$YEREL/last.ckpt" ] && \
        echo "son checkpoint: $(date -r "$YEREL/last.ckpt" '+%H:%M:%S') ($(du -h "$YEREL/last.ckpt" | cut -f1))"
    # ponytail: metrics.csv'yi elle okumak yerine ozetle -- epoch sonu
    # satirlari (loss_epoch / decoder_acc_epoch / *_val) YALNIZ epoch bitince
    # doluyor, step satirlarinda bos. "Epoch bitti mi" sorusunun cevabi bu.
    # Sistem python3'u yeterli (sadece csv), torch gerekmiyor.
    local hedef
    hedef=$(awk -F, '{s+=$3} END {printf "%d", s/'"${MF:-1200}"'}' \
            "data/labels/${AD}_train_transcript_lengths_seg6s.csv" 2>/dev/null || echo 0)
    python3 - "$YEREL" "$hedef" <<'PYEOF'
import csv, glob, sys, os
yerel, hedef = sys.argv[1], int(sys.argv[2] or 0)
satirlar = []
for f in sorted(glob.glob(os.path.join(yerel, "version_*", "metrics.csv"))):
    with open(f) as fh:
        satirlar += list(csv.DictReader(fh))
if not satirlar:
    print("--- metrik yok (henuz baslamadi) ---"); raise SystemExit
def sayi(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else None
son = satirlar[-1]
adim, ep = int(float(son.get("step") or 0)), int(float(son.get("epoch") or 0))
# ponytail: hedef manifest'ten kestiriliyor; yanlissa sacma yuzde basmasin.
# ponytail: manifest'ten kestirilen hedef (toplam_kare/max_frames) YANLIS --
# bucket'larin son batch'leri yarim kaliyor, gercek adim ~%15 fazla
# (olculdu 2026-09-04: kestirim 6091, gercek >7000). Epoch geciyse gozlemden al.
kaynak = "manifest kestirimi"
gozlem = {}
for r in satirlar:
    e = int(float(r.get("epoch") or 0)); st = int(float(r.get("step") or 0))
    gozlem[e] = max(gozlem.get(e, 0), st)
if len(gozlem) > 1:
    biten = sorted(gozlem)[:-1]          # son epoch devam ediyor olabilir
    uzunluklar = [gozlem[e] - (gozlem[e-1] if e-1 in gozlem else 0) for e in biten]
    if uzunluklar and uzunluklar[-1] > 0:
        hedef, kaynak = uzunluklar[-1], "olculdu"
print(f"--- ilerleme: epoch {ep} · adim {adim}"
      + (f" · {hedef} adim/epoch ({kaynak})" if hedef else ""))
if hedef > 0:
    icinde = adim - ep * hedef
    y1, y2 = 100 * icinde / hedef, 100 * adim / (hedef * 20)
    if 0 <= y1 <= 105 and 0 <= y2 <= 105:
        print(f"    bu epoch'ta %{y1:.0f} · toplam %{y2:.1f} (20 epoch hedefi)")
    else:
        print(f"    (yuzde hesabi tutarsiz: hedef ~{hedef} adim/epoch, "
              f"adim {adim}, epoch {ep} -- manifest degismis olabilir)")
# EPOCH SONU satirlari
# ponytail: Lightning epoch sonunda IKI satir yaziyor (biri validation, biri
# train ortalamasi) -> satir saymak epoch sayisini iki katina cikariyordu
# (olculdu 2026-09-04: 1 epoch bitmisken "BITEN EPOCH: 2" yazdi).
# Ayni epoch'un iki satirini birlestir.
bitmis_ham = [r for r in satirlar if sayi(r, "decoder_acc_epoch") is not None
              or sayi(r, "decoder_acc_val") is not None]
birlesik = {}
for r in bitmis_ham:
    e = int(float(r.get("epoch") or 0))
    birlesik.setdefault(e, {}).update({k: v for k, v in r.items() if v not in ("", None)})
bitmis = [dict(v, epoch=str(e)) for e, v in sorted(birlesik.items())]
if bitmis:
    print(f"--- BITEN EPOCH: {len(bitmis)} ---")
    for r in bitmis[-3:]:
        p = [f"epoch={int(float(r.get('epoch') or 0))}"]
        for k, ad in (("decoder_acc_epoch","acc"), ("loss_epoch","loss"),
                      ("decoder_acc_val","acc_val"), ("loss_val","loss_val")):
            v = sayi(r, k)
            if v is not None: p.append(f"{ad}={v:.4f}")
        print("   ", " · ".join(p))
else:
    print("--- henuz BITEN EPOCH yok (epoch sonu metrikleri bos) ---")
# son adim metrikleri
adimlar = [r for r in satirlar if sayi(r, "decoder_acc_step") is not None][-3:]
if adimlar:
    print("--- son 3 adim ---")
    for r in adimlar:
        print(f"    step={int(float(r['step']))} "
              f"acc={sayi(r,'decoder_acc_step'):.4f} loss={sayi(r,'loss_step'):.2f}")
PYEOF
    [ -f "$LOG" ] && { echo "--- log kuyrugu ---"; tail -5 "$LOG"; }
}

durdur() {
    local pid
    if pid=$(ana_surec); then
        kill "$pid" && echo "durduruldu (pid $pid)"
    else
        echo "calisan egitim yok"
    fi
    rm -f "$PIDF"
}

case "${1:-basla}" in
  durum)  durum; exit 0;;
  durdur) durdur; exit 0;;
esac

mkdir -p "$YEREL"

# ponytail: script .pid'i YAZIYORDU ama hic OKUMUYORDU -- kullanici yanlislikla
# ikinci kez calistirinca GPU'da iki egitim baslamaya calisti (olculdu
# 2026-09-04: ikincisi OOM'la dustu, ama .pid'i ezip durum()'u yalanci yapti;
# ayni anda ayni last.ckpt'ye yazsalardi checkpoint bozulurdu).
if pid=$(ana_surec); then
    echo "ZATEN CALISIYOR (pid $pid) -- ikinci kopya baslatilmadi."
    echo "  ilerleme icin:  bash $(basename "$0") durum"
    echo "  durdurmak icin: bash $(basename "$0") durdur"
    exit 0
fi

# ponytail: Modal imajinda ayarliydi, EC2'de yoktu -> yine "iki yerde ayri
# config" deseni. Olculdu 2026-09-04: gercek kosuda 23,1/23,7 GB kullanimda ve
# allocator "OOM while trying to allocate 759 MB" uyarisi veriyor (olumcul
# degil, onbellegi bosaltip tekrar deniyor -- ama epoch sonu validation'da
# sert OOM riski var). expandable_segments parcalanmayi azaltiyor; OOM
# mesajinin kendisi de bunu oneriyor.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NOT: MF yukarida tanimli (durum() de kullaniyor). Probe'u 300 satirlik
# manifestle kosturdugum icin bellegi HAFIFE ALDI; sert OOM gorursen
# MF=1000 ya da 800 ile yeniden baslat.
source ./py_bul.sh

# ponytail: setsid Linux'ta var, macOS'ta YOK (olculdu 2026-09-04 — script
# sessizce "command not found" verip egitimi hic baslatmiyordu). nohup zaten
# SIGHUP'i engelliyor, yani SSH kopmasi icin yeterli; setsid varsa oturumdan
# da ayirir, yoksa nohup'la devam.
AYIR="$(command -v setsid || true)"

# --- 1) Onceki makineden kalan ilerlemeyi geri cek --------------------------
if [ ! -f "$YEREL/last.ckpt" ]; then
    echo "yerel checkpoint yok -> S3'ten deneniyor: $UZAK"
    aws s3 sync "$UZAK" "$YEREL" --exclude "*" --include "last.ckpt" \
        --include "*metrics.csv" --include "kosu.log" --only-show-errors || true
    if [ -f "$YEREL/last.ckpt" ]; then
        echo "GERI YUKLENDI: onceki makinenin checkpoint'i bulundu"
    else
        echo "S3'te de yok -> sifirdan baslaniyor"
    fi
fi

# --- 2) Egitimi ayrilmis surecte baslat ------------------------------------
# ponytail: nohup -> SSH kopsa, terminal olse, laptop kapansa surer.
$AYIR nohup "$PY" -u finetune_tr.py \
    --root-dir data \
    --train-file "${AD}_train_transcript_lengths_seg6s.csv" \
    --val-file   "${AD}_val_transcript_lengths_seg6s.csv" \
    --test-file  "${AD}_test_transcript_lengths_seg6s.csv" \
    --exp-name "$KOSU" --accelerator gpu --num-workers 8 \
    --max-frames "$MF" --max-epochs "$EPOCH" $EK \
    --pretrained-model-path vsr_trlrs3_base.pth --transfer-encoder \
    --checkpoint --ckpt-every-n-steps 500 --resume auto --test \
    >> "$LOG" 2>&1 &
EGITIM=$!
echo "$EGITIM" > "$PIDF"
echo "egitim basladi (pid $EGITIM) -> $LOG"

# --- 3) Checkpoint'i S3'e akit --------------------------------------------
# ponytail: last.ckpt 2,7 GB. Ayni bolge ici transfer ucretsiz; 20 dk'da bir
# senkron, makine olurse en fazla 20 dk kaybettirir. Daha sik yapmanin
# maliyeti zaman (2,7 GB yazma), faydasi az.
$AYIR nohup bash -c '
  # ponytail: filtreler TEK yerde. Onceden son senkron filtresizdi ve
  # her sey gidiyordu: adim checkpointleri (2,7 GB x N) ve .pid.
  # Olculdu 2026-09-04: S3te 11 GB gereksiz dosya birikti; asil zarar
  # kurtarmanin 2,7 GB yerine 13,5 GB indirmesi. .pid ise daha sinsi --
  # yeni makinede eski pid baska bir surece denk gelirse durum() yanlisla
  # "CALISIYOR" der.
  # ponytail: "eniyi-*" BURADA olmak ZORUNDA. Ilk kosuda en iyi agirliklar
  # save_top_k yuzunden silinmisti; onu ayri bir ModelCheckpoint ile
  # cozduk ama S3 filtresine eklemeyi atlamistik -- yani makine olse
  # tam da kurtarmak icin ekledigimiz checkpointleri kaybediyorduk.
  # Ayni hata, yeni yerde. (yakalandi 2026-09-06, kosu sirasinda)
  SENK=(--exclude "*" --include "last.ckpt" --include "eniyi-*.ckpt"
        --include "*metrics.csv" --include "kosu.log" --only-show-errors)
  while kill -0 '"$EGITIM"' 2>/dev/null; do
    sleep '"$ARALIK"'
    aws s3 sync "'"$YEREL"'" "'"$UZAK"'" "${SENK[@]}" || true
    echo "[$(date +%H:%M:%S)] S3 senkron" >> "'"$LOG"'"
  done
  # surec bittiginde son hali (test/WER sonucu dahil) -- AYNI filtrelerle
  aws s3 sync "'"$YEREL"'" "'"$UZAK"'" "${SENK[@]}" || true
  echo "[$(date +%H:%M:%S)] son senkron tamam" >> "'"$LOG"'"
' >/dev/null 2>&1 &

echo "S3 senkronu basladi (her $((ARALIK/60)) dk) -> $UZAK"
echo
echo "Artik laptop'ini kapatabilirsin. Kontrol icin:"
echo "  bash bulut_kosu.sh durum"
echo "Makine olurse: yeni instance ac, kurulumu yap, AYNI komutu calistir."
