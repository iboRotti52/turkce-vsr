#!/usr/bin/env bash
# runpod_kurulum.sh — RunPod / EC2 (ya da herhangi bir kiralik GPU) uzerinde
# fine-tune SIHHAT TESTI: ortam kuruluyor mu, VRAM yetiyor mu, epoch kac saat,
# checkpoint/resume calisiyor mu. Hepsi ~15 dakikada, tam kosuya para vermeden.
#
# ONKOSUL: RunPod'da "PyTorch 2.x" imajini sec (torch + CUDA hazir gelir).
#          EC2'de: "Deep Learning AMI (PyTorch)" sec, disk 150 GB.
#          AWS anahtarlarini environment variable olarak ver:
#            AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
#          (EC2'de daha temizi: instance'a IAM role bagla, anahtar hic gerekmez.)
#          (Modal'daki "bucket-credentials" secret'iyla ayni degerler.)
#
# Kullanim:  bash runpod_kurulum.sh
set -euo pipefail
cd "$(dirname "$0")"   # ponytail: nereden cagrilirsa cagrilsin repo kokunde calis

KOVA="s3://lipreading-data-emre2026/master"
CKPT="s3://lipreading-data-emre2026/ckpt/vsr_trlrs3_base.pth"
VERI="${VERI:-./data/master}"

echo "=== 1/5 bagimliliklar ==="
# ponytail: RunPod konteynerinde root'sun, EC2/DLAMI'de degilsin -> sudo gerek.
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -qq && $SUDO apt-get install -y -qq ffmpeg git >/dev/null
# ponytail: torch'un HANGI yorumlayicida oldugunu bul; sistem python'ina kurma.
source ./py_bul.sh
# ponytail: liste requirements-egitim.txt'te -- iki yerde tutmak EC2'de
# "ImportError: TorchCodec is required" ile patladi (olculdu 2026-09-04).
"$PY" -m pip install -q -r requirements-egitim.txt
command -v aws >/dev/null || "$PY" -m pip install -q awscli
# Hangi paketlerin gerektigini kod soyluyor; import edilebildiklerini DOGRULA.
"$PY" - <<'PYEOF'
eksik = []
for m in ("torch","torchvision","torchaudio","torchcodec","av",
          "pytorch_lightning","sentencepiece","cv2","numpy"):
    try:
        __import__(m)
    except Exception as e:
        eksik.append(f"{m} ({type(e).__name__})")
if eksik:
    raise SystemExit("EKSIK PAKET: " + ", ".join(eksik))
print("bagimliliklar dogrulandi")
PYEOF

echo "=== 2/5 veri (yalniz mouth.mp4 + align.json, ~4.7 GiB) ==="
# ponytail: face.mp4 (20 GiB) ve audio.wav (13 GiB) video-only fine-tune'da
# kullanilmiyor -- indirmek 33 GiB ve ~20 dk bosa demek.
mkdir -p "$VERI"
aws s3 sync "$KOVA" "$VERI" \
    --exclude "*" --include "*/mouth.mp4" --include "*/align.json" \
    --only-show-errors
du -sh "$VERI"
# Pretrained agirliklar (Ingilizce LRS3) — --transfer-encoder bunu okuyor.
[ -f vsr_trlrs3_base.pth ] || aws s3 cp "$CKPT" . --only-show-errors || \
  echo "UYARI: pretrained ckpt yok. Once laptop'tan S3'e yukle (bkz. notlar)."

echo "=== 3/5 manifest (bulut kaynagi: split_map.json + align.json) ==="
"$PY" scripts/build_avsr_manifest.py --split-map split_map.json --prefix tr131
wc -l data/labels/tr131_*.csv

echo "=== 4/5 VRAM + hiz olcumu ==="
# T4'un 16 GB'i --max-frames 1600'de OOM vermisti. 24 GB'de tavani ariyoruz:
# ilk gecen degeri kullan, sonrakini deneme.
for MF in 800 1200 1600; do
  echo "--- max_frames=$MF"
  if "$PY" -u finetune_tr.py --root-dir data \
      --train-file tr131_train_transcript_lengths_seg6s.csv \
      --val-file tr131_val_transcript_lengths_seg6s.csv \
      --exp-name probe --accelerator gpu --num-workers 4 \
      --max-frames "$MF" --probe 20 2>&1 | grep -E "^PROBE|OutOfMemory|CUDA out of memory"; then :; fi
done

echo "=== 5/5 checkpoint + resume (GPU'da) ==="
"$PY" test_checkpoint.py

cat <<'NOT'

BITTI. Simdi karar ver:
  - PROBE satirlarindaki en buyuk OOM VERMEYEN max_frames'i sec.
  - sn/adim x (train satir sayisi / batch) = epoch suresi -> toplam maliyet.
  - Gercek kosu:
      "$PY" -u finetune_tr.py --root-dir data \
        --train-file tr131_train_transcript_lengths_seg6s.csv \
        --val-file  tr131_val_transcript_lengths_seg6s.csv \
        --exp-name tr131 --accelerator gpu --num-workers 8 \
        --max-frames <SECILEN> --max-epochs 20 \
        --pretrained-model-path vsr_trlrs3_base.pth --transfer-encoder \
        --test-file tr131_test_transcript_lengths_seg6s.csv \
        --checkpoint --ckpt-every-n-steps 500 --resume auto --test

    --resume auto: spot dususunde AYNI komutu tekrar calistirmak yeter.
    --test:        egitim bitince WER basar. Olmazsa 36 saat kosup sayisiz
                   kalirsin -- lightning'de test_step VAR ama cagrilmiyordu.

  DIKKAT: test_step her ornekte beam search yapiyor (batch=1). 13.060 test
  segmenti saatler surer. Ilk olcumu kucuk bir alt kumede yap:
      head -2000 data/labels/tr131_test_transcript_lengths_seg6s.csv \
        > data/labels/tr131k_test_transcript_lengths_seg6s.csv
NOT
