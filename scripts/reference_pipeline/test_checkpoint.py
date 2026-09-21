"""test_checkpoint.py — checkpoint yazma + resume GERCEKTEN calisiyor mu?

NEDEN: spot/preemptible GPU'da is her an dusebilir. finetune_tr.py'de
`trainer.fit(ckpt_path=...)` YOKTU -- dusen is sifirdan basliyordu ve bunu
ancak 20. saatte fark ederdik. Bu test onu once yerelde, CPU'da, bedava
dogruluyor: checkpoint mantigi cihazdan bagimsiz.

Senaryo: egit -> ortasindan SIGKILL -> resume -> adim sayaci DEVAM ETTI mi?

Kullanim:  ./.venv/bin/python test_checkpoint.py
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(Path(sys.executable))
EXP = ROOT / "exp"
AD = "ckpt_testi"
SON = EXP / AD / "last.ckpt"

TEMEL = [PY, "-u", "finetune_tr.py",
         "--root-dir", "data",
         "--train-file", "ckpt_train_transcript_lengths_seg6s.csv",
         "--val-file", "ckpt_val_transcript_lengths_seg6s.csv",
         "--exp-name", AD, "--accelerator", "cpu", "--num-workers", "0",
         "--max-frames", "400", "--max-epochs", "3",
         "--checkpoint", "--ckpt-every-n-steps", "2"]


def adim(p: Path) -> int:
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)["global_step"]


def kos(ek, bitene_kadar=None, zaman_asimi=1800):
    """Süreci başlat. bitene_kadar verilirse o koşul sağlanınca SIGKILL."""
    pr = subprocess.Popen(TEMEL + ek, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1)
    ciktilar = []
    t0 = time.time()
    for satir in pr.stdout:
        ciktilar.append(satir)
        if bitene_kadar and bitene_kadar():
            pr.send_signal(signal.SIGKILL)
            pr.wait()
            return "".join(ciktilar), None
        if time.time() - t0 > zaman_asimi:
            pr.send_signal(signal.SIGKILL); pr.wait()
            raise SystemExit("zaman asimi")
    pr.wait()
    return "".join(ciktilar), pr.returncode


def main():
    if EXP.exists():
        shutil.rmtree(EXP)

    print("=== 1) egit, ilk checkpoint dusunce oldur (preemption taklidi) ===")
    log1, _ = kos([], bitene_kadar=lambda: SON.exists() and SON.stat().st_size > 0)
    assert SON.exists(), "HATA: hic checkpoint yazilmadi"
    time.sleep(1)
    s1 = adim(SON)
    print(f"    oldurulmeden onceki global_step = {s1}")
    assert s1 > 0, "HATA: checkpoint global_step=0"

    print("=== 2) --resume auto ile devam et ===")
    log2, rc = kos(["--resume", "auto"])
    assert rc == 0, f"HATA: resume kosusu cikis kodu {rc}\n{log2[-2000:]}"

    geri_yuklendi = "Restored all states" in log2
    s2 = adim(SON)
    print(f"    resume sonrasi global_step = {s2}")

    # AYIRT EDICI KANIT: resume kosusu (version_1) hangi adimdan basladi?
    # Bozuk resume 0'dan baslar; saglam resume checkpoint'in adimindan.
    # ("s2 > s1" tek basina yetmez -- sifirdan baslayan kosu da ayni yere varir.)
    ilk_adim = None
    for v in sorted((EXP / AD).glob("version_*"), key=lambda p: p.name):
        pass  # en son version = resume kosusu
    mcsv = v / "metrics.csv"
    if mcsv.exists():
        import csv
        with open(mcsv) as f:
            adimlar = [int(float(r["step"])) for r in csv.DictReader(f) if r.get("step")]
        ilk_adim = min(adimlar) if adimlar else None
    print(f"    resume kosusunun ILK kayitli adimi = {ilk_adim} ({v.name})")

    print("\n--- SONUC ---")
    assert geri_yuklendi, "HATA: Lightning checkpoint'i geri yuklemedi (sifirdan basladi)"
    print(f"[OK] checkpoint geri yuklendi")
    assert s2 > s1, f"HATA: adim ilerlemedi ({s1} -> {s2}) — resume sifirlamis olabilir"
    print(f"[OK] adim sayaci devam etti: {s1} -> {s2}")
    assert ilk_adim is not None, "HATA: metrics.csv okunamadi, resume dogrulanamiyor"
    assert ilk_adim >= s1, (
        f"HATA: resume kosusu adim {ilk_adim}'den basladi ama checkpoint {s1}'deydi "
        f"— egitim BASTAN basladi, ckpt_path islemedi")
    print(f"[OK] resume checkpoint'ten devam etti (ilk adim {ilk_adim} >= {s1})")
    kalan = sorted(p.name for p in (EXP / AD).glob("*.ckpt"))
    print(f"[OK] checkpoint dosyalari: {kalan}")
    print(f"[OK] checkpoint boyutu: {SON.stat().st_size / 2**20:.0f} MiB")
    print("\nTUMU GECTI — spot GPU'da preemption veri kaybettirmez.")


if __name__ == "__main__":
    main()
