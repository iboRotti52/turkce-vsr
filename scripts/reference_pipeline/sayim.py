"""sayim.py — S3'teki master/'ın envanteri: hangi videoda hangi dosya var?

NEDEN AYRI DOSYA: modal_pipeline.py'nin imajı büyük (mediapipe, whisper,
ffmpeg, deno) ve S3'e FUSE ile bağlanıyor. Envanter için ikisi de gereksiz —
boto3 ile tek bir ListObjectsV2 turu tüm kovayı 1000'lik sayfalar hâlinde
veriyor. FUSE'te 32 bin klasörü gezmek yerine ~150 API isteği.

NE CEVAPLAR:
  - kaç videoda kaç segment var (meta.json)
  - o segmentlerin kaçında Auto-AVSR'ın istediği mouth.mp4 VAR
    (ağız kırpması sonradan eklendi; eskiler modal_backfill.py bekliyor)
  - kaçında align.json (= manifest için metin) var

Kullanım:
    MODAL_PROFILE=uuu4 ./.venv/bin/modal run sayim.py
    MODAL_PROFILE=uuu4 ./.venv/bin/modal run sayim.py --cikti envanter.json
"""
import json

import modal

BUCKET_NAME = "lipreading-data-emre2026"   # modal_pipeline.py ile aynı kova
PREFIX = "master/"

app = modal.App("lipreading-sayim")
image = modal.Image.debian_slim(python_version="3.11").pip_install("boto3")


@app.function(
    image=image,
    cpu=0.25,
    timeout=1800,
    secrets=[modal.Secret.from_name(
        "bucket-credentials",
        required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])],
)
def envanter() -> dict:
    import boto3

    s3 = boto3.client("s3")
    # video_id -> dosya adı -> o dosyaya sahip seg_id kümesi
    per_video: dict[str, dict[str, set]] = {}
    toplam_anahtar = 0
    boyut: dict[str, int] = {}

    for sayfa in s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET_NAME, Prefix=PREFIX):
        for obj in sayfa.get("Contents", []):
            toplam_anahtar += 1
            # master/<video_id>/<seg_id>/<dosya>
            parca = obj["Key"][len(PREFIX):].split("/")
            if len(parca) != 3:
                continue          # video kökündeki serbest dosyalar
            vid, seg, dosya = parca
            per_video.setdefault(vid, {}).setdefault(dosya, set()).add(seg)
            boyut[dosya] = boyut.get(dosya, 0) + obj["Size"]

    return {
        "toplam_anahtar": toplam_anahtar,
        "boyut_gib": {d: round(b / 2**30, 2) for d, b in boyut.items()},
        "videolar": {v: {d: len(s) for d, s in f.items()}
                     for v, f in per_video.items()},
    }


@app.local_entrypoint()
def main(cikti: str = "envanter.json"):
    r = envanter.remote()
    vids = r["videolar"]

    def topla(dosya):
        return sum(v.get(dosya, 0) for v in vids.values())

    meta, mouth, align = topla("meta.json"), topla("mouth.mp4"), topla("align.json")
    tam = [v for v, f in vids.items()
           if f.get("meta.json", 0) and f.get("mouth.mp4", 0) >= f["meta.json"]]
    hic = [v for v, f in vids.items()
           if f.get("meta.json", 0) and not f.get("mouth.mp4", 0)]

    with open(cikti, "w") as f:
        json.dump(r, f, indent=1, sort_keys=True)

    print(f"SONUC anahtar={r['toplam_anahtar']} video={len(vids)}")
    print(f"  meta.json  = {meta}")
    print(f"  align.json = {align}")
    print(f"  mouth.mp4  = {mouth}  (%{100*mouth/max(1,meta):.1f} kapsama)")
    print(f"  tam kapsanan video = {len(tam)} / eksiksiz sifir = {len(hic)}")
    for d, g in sorted(r["boyut_gib"].items(), key=lambda x: -x[1]):
        print(f"  boyut {d:12} = {g} GiB")
    print(f"-> {cikti}")


@app.function(
    image=image,
    cpu=0.5,
    timeout=1800,
    secrets=[modal.Secret.from_name(
        "bucket-credentials",
        required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])],
)
def sure_ornek(anahtarlar: list) -> list:
    """Verilen meta.json anahtarlarını okuyup segment sürelerini döndürür."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    import boto3

    s3 = boto3.client("s3")

    def oku(k):
        try:
            d = json.loads(s3.get_object(Bucket=BUCKET_NAME, Key=k)["Body"].read())
            return d.get("end", 0) - d.get("start", 0)
        except Exception:
            return None

    # ponytail: GET'ler ağ-bağımlı, thread yeter (GIL boşta bekliyor)
    with ThreadPoolExecutor(max_workers=32) as ex:
        return [s for s in ex.map(oku, anahtarlar) if s and 0 < s <= 30]


@app.local_entrypoint()
def sure(n: int = 2000, envanter_dosya: str = "envanter.json"):
    """Örneklemle toplam saati kestirir. Kullanım: modal run sayim.py::sure"""
    import random
    import statistics

    env = json.load(open(envanter_dosya))
    toplam_seg = sum(v.get("meta.json", 0) for v in env["videolar"].values())

    # Video başına segment sayısıyla ORANTILI örnekleme yerine, her videodan
    # eşit sayıda çekmek uzun videoları hafife alırdı -> anahtarları
    # doğrudan kovadan çekmek pahalı, bu yüzden seg_id'yi indeksten üretiyoruz.
    havuz = []
    for vid, f in env["videolar"].items():
        for i in range(f.get("meta.json", 0)):
            havuz.append((vid, i))
    random.seed(0)
    sec = random.sample(havuz, min(n, len(havuz)))
    # seg_id formatı: <video_id>_%04d ama numaralar sürekli DEĞİL (elenenler var)
    # -> listeleyip gerçek seg_id'leri almak gerek: her video için tek LIST.
    print(f"örnek={len(sec)} / toplam segment={toplam_seg} — anahtarlar çözülüyor...")
    hedef = {}
    for vid, i in sec:
        hedef.setdefault(vid, []).append(i)
    anahtarlar = cozumle.remote(hedef)
    print(f"çözülen anahtar={len(anahtarlar)}")

    parcalar = [anahtarlar[i::8] for i in range(8)]
    sureler = [s for r in sure_ornek.map(parcalar) for s in r]
    ort = statistics.mean(sureler)
    sd = statistics.stdev(sureler)
    hata = 1.96 * sd / len(sureler) ** 0.5
    saat = toplam_seg * ort / 3600
    pay = toplam_seg * hata / 3600
    print(f"SONUC ornek={len(sureler)} ort_sn={ort:.2f} (±{hata:.3f})")
    print(f"  TOPLAM ≈ {saat:.1f} saat  (%95 aralık: {saat-pay:.1f} – {saat+pay:.1f})")


@app.function(
    image=image, cpu=0.5, timeout=1800,
    secrets=[modal.Secret.from_name("bucket-credentials",
             required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])],
)
def cozumle(hedef: dict) -> list:
    """{video_id: [indeks,...]} -> gerçek meta.json anahtarları."""
    import boto3

    s3 = boto3.client("s3")
    out = []
    for vid, idx in hedef.items():
        metalar = []
        for sayfa in s3.get_paginator("list_objects_v2").paginate(
                Bucket=BUCKET_NAME, Prefix=f"{PREFIX}{vid}/"):
            metalar += [o["Key"] for o in sayfa.get("Contents", [])
                        if o["Key"].endswith("/meta.json")]
        metalar.sort()
        out += [metalar[i] for i in idx if i < len(metalar)]
    return out


@app.function(
    image=image, cpu=2.0, timeout=1800,
    secrets=[modal.Secret.from_name("bucket-credentials",
             required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])],
)
def korpus_parca(anahtarlar: list) -> list:
    """align.json'lardan metinleri toplar."""
    import json as _j
    from concurrent.futures import ThreadPoolExecutor

    import boto3

    s3 = boto3.client("s3")

    def oku(k):
        try:
            t = _j.loads(s3.get_object(Bucket=BUCKET_NAME, Key=k)["Body"].read()).get("text", "")
            return t.strip() or None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=64) as ex:
        return [t for t in ex.map(oku, anahtarlar) if t]


@app.local_entrypoint()
def korpus(cikti: str = "vendor/auto_avsr_train/spm/input_tr.txt"):
    """Tum align.json metinlerini indirip tokenizer korpusunu yazar.

    NEDEN: train_tr_tokenizer.py state.db'den okuyor -> 6.483 cumle.
    Gercekte 75.276 var. Turkce sondan eklemeli; az korpus = nadir ekler
    karakter karakter bolunur = token sayisi siser = VRAM ve WER kotulesir.

    Kullanim: MODAL_PROFILE=uuu4 modal run sayim.py::korpus
    """
    anahtarlar = tum_align.remote()
    print(f"align.json anahtari: {len(anahtarlar)}")
    parcalar = [anahtarlar[i::16] for i in range(16)]
    metinler = [t for r in korpus_parca.map(parcalar) for t in r]
    with open(cikti, "w", encoding="utf-8") as f:
        f.write("\n".join(metinler))
    kelime = sum(len(t.split()) for t in metinler)
    print(f"SONUC cumle={len(metinler)} kelime={kelime} -> {cikti}")


@app.function(
    image=image, cpu=0.5, timeout=900,
    secrets=[modal.Secret.from_name("bucket-credentials",
             required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])],
)
def tum_align() -> list:
    import boto3

    s3 = boto3.client("s3")
    out = []
    for sayfa in s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET_NAME, Prefix=PREFIX):
        out += [o["Key"] for o in sayfa.get("Contents", [])
                if o["Key"].endswith("/align.json")]
    return out
