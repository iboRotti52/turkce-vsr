#!/usr/bin/env python3
"""s3_cek.py — S3'ten SECICI indirme (yerelde calisabilmek icin).

Neden: kapali kume isi CPU/MPS'te kosabiliyor, EC2 acmaya gerek yok.
Ama align.json ve hedef mouth.mp4'ler yerelde yok.

align.json + meta.json TOPLAM ~70 MB ama 150.000 dosya -> tek tek GET
yavas; is parcacigi havuzu sart. mouth.mp4 sadece ISTENEN segmentler icin.

Kullanim:
    python s3_cek.py align                 # tum align.json + meta.json
    python s3_cek.py mouth seg_listesi.txt # sadece o segmentlerin mouth.mp4
"""
import os, sys, json, pathlib, threading
import concurrent.futures as cf
import boto3
from botocore.config import Config
from dotenv import load_dotenv

KOK = pathlib.Path(__file__).resolve().parent
load_dotenv(KOK / ".env")
KOVA = "lipreading-data-emre2026"
ONEK = "master/"
HEDEF = KOK / "data" / "master"

_yerel = threading.local()


def s3():
    if not hasattr(_yerel, "c"):
        _yerel.c = boto3.client("s3", config=Config(max_pool_connections=80,
                                                    retries={"max_attempts": 5}))
    return _yerel.c


def indir(anahtar):
    hedef = HEDEF / anahtar[len(ONEK):]
    if hedef.exists() and hedef.stat().st_size > 0:
        return 0
    hedef.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3().download_file(KOVA, anahtar, str(hedef))
        return 1
    except Exception:
        return -1


def kos(anahtarlar, isc=64):
    ok = atlandi = hata = 0
    with cf.ThreadPoolExecutor(isc) as ex:
        for i, r in enumerate(ex.map(indir, anahtarlar), 1):
            ok += r == 1; atlandi += r == 0; hata += r == -1
            if i % 5000 == 0:
                print(f"  {i:,}/{len(anahtarlar):,} · indi {ok:,} · vardi {atlandi:,} · hata {hata}",
                      flush=True)
    print(f"bitti: indi {ok:,} · zaten vardi {atlandi:,} · hata {hata}")


def _video_listele(vid):
    out = []
    for sayfa in s3().get_paginator("list_objects_v2").paginate(
            Bucket=KOVA, Prefix=f"{ONEK}{vid}/"):
        out += [o["Key"] for o in sayfa.get("Contents", [])]
    return out


def listele(sonekler, isc=32):
    """VIDEO BASINA listele, paralel. Tek dev tarama (374.017 anahtar) hem
    yavas hem ilerlemesi gorunmez; 466 kucuk cagri parallelde saniyeler."""
    videolar = list(json.load(open(KOK / "envanter.json"))["videolar"])
    print(f"{len(videolar)} video listeleniyor...", flush=True)
    hepsi = []
    with cf.ThreadPoolExecutor(isc) as ex:
        for i, r in enumerate(ex.map(_video_listele, videolar), 1):
            hepsi += r
            if i % 100 == 0:
                print(f"  {i}/{len(videolar)} video · {len(hepsi):,} anahtar", flush=True)
    sec = [k for k in hepsi if any(k.endswith(x) for x in sonekler)]
    print(f"toplam {len(hepsi):,} anahtar · hedef {len(sec):,}", flush=True)
    return sec


if __name__ == "__main__":
    komut = sys.argv[1]
    if komut == "align":
        kos(listele(("align.json", "meta.json")))
    elif komut == "face":
        segler = [x.strip() for x in open(sys.argv[2]) if x.strip()]
        HEDEF = KOK / "data" / "faces"
        kos([f"{ONEK}{x}/face.mp4" for x in segler])
    elif komut == "mouth":
        segler = [s.strip() for s in open(sys.argv[2]) if s.strip()]
        kos([f"{ONEK}{s}/mouth.mp4" for s in segler])
    else:
        sys.exit("bilinmeyen komut")
