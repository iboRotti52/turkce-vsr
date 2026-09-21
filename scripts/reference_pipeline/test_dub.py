"""test_dub.py — dublaj kararının doğru seviyede verildiğini doğrular.

Çalıştır:  python test_dub.py
"""
import ast
import contextlib

import numpy as np

FPS, SR, MIN_SEG_DUB_FRAMES = 25, 16000, 5

# pipeline.py agir bagimliliklar cekiyor (cv2, mediapipe); ilgili saf
# fonksiyonlari kaynaktan izole edip yukluyoruz.
_ns = {"np": np, "FPS": FPS, "MIN_SEG_DUB_FRAMES": MIN_SEG_DUB_FRAMES,
       "contextmanager": contextlib.contextmanager}
for _n in ast.parse(open("pipeline.py").read()).body:
    if isinstance(_n, ast.FunctionDef) and _n.name in {
            "_corr", "dub_esigi", "video_dub_score", "seg_dub_score",
            "dub_karari"}:
        exec(compile(ast.Module([_n], []), "<p>", "exec"), _ns)
_corr, dub_esigi, dub_karari = _ns["_corr"], _ns["dub_esigi"], _ns["dub_karari"]
DUB_MIN_N = int(next(
    n.value.value for n in ast.parse(open("pipeline.py").read()).body
    if isinstance(n, ast.Assign) and n.targets[0].id == "DUB_MIN_N"))
_ns["DUB_MIN_N"] = DUB_MIN_N   # dub_karari bunu global olarak ariyor
video_dub_score, seg_dub_score = _ns["video_dub_score"], _ns["seg_dub_score"]


def _yap(saniye, senkron: bool, rng, gurultu=6.0):
    """senkron=True -> ağız sesi ZAYIF takip eder (gerçek konuşma).
    senkron=False -> ağız ve ses bağımsız (dublaj).

    gurultu=6 kasıtlı: gerçek veride ölçtüğümüz rejim bu — ağız-açıklığı ile
    ses enerjisi arasındaki gerçek korelasyon ~0.05 civarında, yani tek bir
    segmentin gürültüsünün (σ≈0.12) çok altında. Testin anlamı buradan geliyor;
    temiz bir sentetik sinyalle segment eşiği de "çalışıyor" görünür ve asıl
    bug gizlenir (ilk sürümde tam bunu yaşadık)."""
    n = int(saniye * FPS)
    zarf = np.abs(rng.standard_normal(n))            # konusma enerjisi
    zarf = np.convolve(zarf, np.ones(5) / 5, "same")  # yumusat
    agiz = (zarf + gurultu * rng.standard_normal(n)) if senkron else np.abs(
        rng.standard_normal(n))
    faces = {i: {"bbox": [0, 0, 80, 80], "mouth_open": float(agiz[i])}
             for i in range(n)}
    hop = SR // FPS
    ses = np.repeat(zarf, hop) * rng.standard_normal(n * hop)
    return faces, ses.astype(np.float32)


def main():
    rng = np.random.default_rng(7)

    # 1) Eşik, örnek sayısıyla küçülüyor mu (asıl fikir bu).
    assert dub_esigi(100) > dub_esigi(10000) > 0
    assert dub_esigi(3) == 1.0, "n yetersizse eşik erişilemez olmalı"
    assert abs(dub_esigi(30000) - 3 / np.sqrt(30000)) < 1e-9
    assert dub_esigi(25) > 0.5, "segment n'inde eşik pratikte geçilemez olmalı"

    # 2) GERÇEK konuşma: video seviyesinde eşiği geçmeli.
    faces, ses = _yap(600, senkron=True, rng=rng)      # 10 dk
    skor, n = video_dub_score(faces, ses, SR)
    assert n > 10000, n
    assert skor > dub_esigi(n), f"gerçek konuşma elendi: {skor:.4f} < {dub_esigi(n):.4f}"

    # 3) DUBLAJ: video seviyesinde eşiğin altında kalmalı.
    dfaces, dses = _yap(600, senkron=False, rng=rng)
    dskor, dn = video_dub_score(dfaces, dses, SR)
    assert dskor < dub_esigi(dn), f"dublaj geçti: {dskor:.4f}"

    # 3b) HESAPLANAMAZ durum nan donmeli, 0.0 DEGIL. Yuzu hic bulunamayan
    #     video, dublaj sanilip elenmemeli (27 videoyu bu yuzden kaybettik).
    yuzsuz = {i: {"bbox": None, "mouth_open": 0.0} for i in range(15000)}
    yskor, yn = video_dub_score(yuzsuz, ses, SR)
    assert np.isnan(yskor), f"yuz yokken nan bekleniyordu, {yskor} geldi"
    assert not (yskor < dub_esigi(yn)), "nan karsilastirmasi kapiyi tetiklememeli"
    # bos faces sozlugu de (n yeterli ama hic kare yok) patlamamali
    assert np.isnan(video_dub_score({}, ses, SR)[0])

    # 3c) SEYRELME: yuzun yalnizca yarisinda bulundugu GERCEK konusma,
    #     eksik kareler hesaba katilmadigi surece skorunu korumali.
    #     (Eskiden bu kareler 0.0 dolduruluyor ve korelasyonu sifira cekiyordu.)
    tam_faces, tam_ses = _yap(600, senkron=True, rng=rng)
    tam_skor, tam_n = video_dub_score(tam_faces, tam_ses, SR)
    delik = {i: (f if i % 2 == 0 else {"bbox": None, "mouth_open": 0.0})
             for i, f in tam_faces.items()}
    d_skor, d_n = video_dub_score(delik, tam_ses, SR)
    assert d_n < tam_n, "eksik kareler n'den dusmeli"
    assert d_skor > tam_skor * 0.5, (
        f"yuz kaybi skoru seyreltmemeli: tam={tam_skor:.4f} delikli={d_skor:.4f}")
    assert d_skor > dub_esigi(d_n), (
        f"yarim yuzlu gercek konusma elenmemeli: {d_skor:.4f} < {dub_esigi(d_n):.4f}")
    print(f"  seyrelme testi: tam={tam_skor:+.4f} (n={tam_n}) -> "
          f"yuz yarisi silinince {d_skor:+.4f} (n={d_n})")

    # 3d) KANITSIZLIK != DUBLAJ. Uc durum uc ayri cevap vermeli.
    assert dub_karari(float("nan"), 999999) == "karar_yok", "nan elenemez"
    assert dub_karari(0.0, DUB_MIN_N - 1) == "karar_yok", "yetersiz n elenemez"
    assert dub_karari(0.0, DUB_MIN_N) == "elendi", "kanit varken elemeli"
    assert dub_karari(0.5, DUB_MIN_N) == "gecti"
    # gercek konusma yeterli n uretmeli, yoksa kapi hic calismaz
    assert video_dub_score(faces, ses, SR)[1] >= DUB_MIN_N
    assert dub_karari(*video_dub_score(faces, ses, SR)) == "gecti"
    assert dub_karari(*video_dub_score(dfaces, dses, SR)) == "elendi"

    # 4) ASIL BUG: aynı GERÇEK konuşmanın tek tek segmentleri, eski sabit
    #    0.15 eşiğinde çoğunlukla eleniyordu. Bunu görünür kıl.
    segler = [{"start": t, "end": t + 4.0} for t in range(0, 560, 4)]
    skorlar = [seg_dub_score(faces, ses, SR, s) for s in segler]
    elenen = sum(1 for x in skorlar if not np.isnan(x) and x < 0.15)
    assert elenen > len(segler) * 0.3, (
        f"eski eşik bu kadar elemeliydi, {elenen}/{len(segler)} eledi")
    print(f"  eski sabit 0.15 eşiği, GERÇEK konuşmanın "
          f"{elenen}/{len(segler)} segmentini eliyordu")
    print(f"  video seviyesi: gerçek={skor:+.4f} vs eşik={dub_esigi(n):.4f}  -> GEÇER")
    print(f"                  dublaj={dskor:+.4f} vs eşik={dub_esigi(dn):.4f}  -> ELENİR")
    print("test_dub: TAMAM")


if __name__ == "__main__":
    main()
