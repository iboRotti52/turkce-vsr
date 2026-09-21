#!/usr/bin/env python3
"""senkron.py — dudak sesle birlikte mi hareket ediyor?

NEDEN: yuz_kapsama.py "yuz var mi" sorusunu cevapliyor ama "DOGRU yuz mu"
sorusunu cevaplayamaz -- olcumu face.mp4 uzerinde yapiyor, o da zaten
supheli landmark'larla uretilmis kirpma. MediaPipe konusan yerine
dinleyene kilitlendiyse face.mp4 dinleyenin yuzudur ve "temiz" gorunur.

Bu betik farkli bir soru soruyor: agiz hareketi ses zarfiyla ORTUSUYOR mu?
Ortusmuyorsa ya yanlis kisi kirpilmis ya da o kisi konusmuyor.

YONTEM (kendi kendini normalize eden):
  hareket[t] = mouth.mp4'un merkez bolgesinde kare-kare fark
  ses[t]     = audio.wav'in 25 Hz'e indirgenmis enerji zarfi
  skor       = korelasyon(0 gecikme) - en iyi KAYDIRILMIS korelasyon
Mutlak korelasyon zayiftir (dudak acikligi ile ses siddeti dogrusal degil),
ama gercek senkronda SIFIR GECIKME kaydirilmis halleri yenmeli. Yenmiyorsa
o segmentte ses ile goruntu ayni kisiye ait degil demektir.

Kullanim:
    python senkron.py --n 2000 --cikti senkron.jsonl
"""
import argparse, csv, json, os, random, subprocess, tempfile
import concurrent.futures as cf
import numpy as np

KOVA = "s3://lipreading-data-emre2026/master"


def ses_zarfi(yol, n_kare, fps=25.0):
    """audio.wav -> kare basina enerji. ffmpeg ile ham PCM, kutuphane yok."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", yol, "-ac", "1", "-ar", "16000",
                        "-f", "s16le", "-"], capture_output=True)
    if p.returncode or not p.stdout:
        return None
    x = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    ornek_kare = int(16000 / fps)
    kes = n_kare * ornek_kare
    if len(x) < kes:
        x = np.pad(x, (0, kes - len(x)))
    x = x[:kes].reshape(n_kare, ornek_kare)
    return np.sqrt((x ** 2).mean(axis=1))


def agiz_hareketi(yol):
    import cv2
    cap = cv2.VideoCapture(yol)
    k = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        k.append(g[h // 4:3 * h // 4, w // 4:3 * w // 4])   # merkez: agiz
    cap.release()
    if len(k) < 20:
        return None
    a = np.stack(k)
    d = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[d[0]], d])


def kor(a, b):
    a = a - a.mean(); b = b - b.mean()
    s = a.std() * b.std()
    return float((a * b).mean() / s) if s > 1e-9 else 0.0


def olc(rel):
    seg = os.path.dirname(rel)
    hareket = agiz_hareketi(f"data/master/{rel}")
    if hareket is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        w = os.path.join(td, "a.wav")
        if subprocess.run(["aws", "s3", "cp", f"{KOVA}/{seg}/audio.wav", w,
                           "--only-show-errors"], capture_output=True).returncode:
            return None
        ses = ses_zarfi(w, len(hareket))
    if ses is None or ses.std() < 1e-6 or hareket.std() < 1e-6:
        return None
    # sifir gecikme vs kaydirilmis (>= 12 kare = 0,5 sn) en iyisi
    s0 = kor(hareket, ses)
    kaydirilmis = []
    for g in list(range(-40, -11)) + list(range(12, 41)):
        kaydirilmis.append(kor(hareket, np.roll(ses, g)))
    return {"seg": seg, "kare": len(hareket), "s0": round(s0, 4),
            "kaydirilmis_max": round(max(kaydirilmis), 4),
            "skor": round(s0 - max(kaydirilmis), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/labels/tr131_train_transcript_lengths_seg6s.csv")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--is", dest="isc", type=int, default=8)
    ap.add_argument("--cikti", default="senkron.jsonl")
    cli = ap.parse_args()

    kayit = [r[1] for r in csv.reader(open(cli.manifest)) if len(r) >= 3]
    random.Random(11).shuffle(kayit)
    bitti = set()
    if os.path.exists(cli.cikti):
        for s in open(cli.cikti):
            try:
                bitti.add(json.loads(s)["seg"])
            except Exception:
                pass
    isler = [r for r in kayit[:cli.n] if os.path.dirname(r) not in bitti]
    print(f"{len(isler)} segment olculecek", flush=True)
    with open(cli.cikti, "a") as ck, cf.ThreadPoolExecutor(cli.isc) as ex:
        for i, r in enumerate(ex.map(olc, isler), 1):
            if r:
                ck.write(json.dumps(r) + "\n")
            if i % 250 == 0:
                ck.flush(); print(f"  {i}/{len(isler)}", flush=True)
    print("bitti")


if __name__ == "__main__":
    main()
