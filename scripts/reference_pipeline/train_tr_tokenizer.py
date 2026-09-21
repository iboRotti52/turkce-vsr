"""train_tr_tokenizer.py — segments.text'ten (qc_pass=1) Türkçe SentencePiece modeli eğitir.

Auto-AVSR'ın TextTransform sınıfı sabit dosya adları bekliyor
(spm/unigram/unigram5000.model, unigram5000_units.txt) — isim "5000" içerse de
gerçek vocab boyutu farklı olabilir, sadece dosya adı.

Kullanım:
    .venv/bin/python scripts/train_tr_tokenizer.py [--vocab-size 1000]
"""
import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "state.db"
SPM_DIR = ROOT / "vendor" / "auto_avsr_train" / "spm"
UNIGRAM_DIR = SPM_DIR / "unigram"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=1000)
    ap.add_argument("--korpus", action="store_true",
                    help="state.db yerine MEVCUT spm/input_tr.txt'yi kullan "
                         "(sayim.py::korpus ile bulut master'dan indirilmis olur). "
                         "state.db 161 videoda donmus: 6.483 cumle, oysa 75.273 var.")
    args = ap.parse_args()

    input_txt = SPM_DIR / "input_tr.txt"
    if args.korpus:
        if not input_txt.exists():
            raise SystemExit(f"korpus yok: {input_txt}  (once: modal run sayim.py::korpus)")
        texts = [l.strip() for l in input_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"korpus kullaniliyor: {input_txt} ({len(texts)} cumle)")
    else:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute("select text from segments where qc_pass=1 and text is not null and text != ''").fetchall()
        texts = [r[0].strip() for r in rows if r[0] and r[0].strip()]
        print(f"{len(texts)} transkript satırı bulundu (qc_pass=1)")
        print("UYARI: state.db 161 videoda donmus. Bulut icin: --korpus")
        input_txt.write_text("\n".join(texts), encoding="utf-8")
        print(f"korpus yazıldı: {input_txt}")

    UNIGRAM_DIR.mkdir(parents=True, exist_ok=True)
    dict_path = UNIGRAM_DIR / "unigram5000_units.txt"
    model_prefix = UNIGRAM_DIR / "unigram5000"

    import sentencepiece as spm

    # NOT: string-birleşimli Train() çağrısı boşluklu yollarda kırılıyor
    # (proje dizini "files 2" içeriyor) — kwargs formu güvenli.
    spm.SentencePieceTrainer.Train(
        input=str(input_txt),
        vocab_size=args.vocab_size,
        model_type="unigram",
        model_prefix=str(model_prefix),
        input_sentence_size=100000000,
        character_coverage=1.0,
    )
    print(f"model eğitildi: {model_prefix}.model")

    # units.txt: auto_avsr/spm/train.sh ile birebir aynı mantık
    sp = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")
    pieces = set()
    for line in texts:
        pieces.update(sp.EncodeAsPieces(line))

    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("<unk> 1\n")
        for i, piece in enumerate(sorted(pieces)):
            f.write(f"{piece} {i + 2}\n")

    print(f"dictionary yazıldı: {dict_path} ({len(pieces)} parça + <unk>)")

    # ponytail: vendor/auto_avsr_train IC ICE bir git deposu ve vendor/ ignore'da
    # -> git oraya dosya ekleyemiyor, hata da vermiyor (olculdu 2026-09-03).
    # Manifest token_id'leri bu modele bagli; tek laptopta kalmasin.
    import shutil
    izlenen = ROOT / "tokenizer"
    izlenen.mkdir(exist_ok=True)
    for f in (str(model_prefix) + ".model", str(model_prefix) + ".vocab", dict_path):
        shutil.copy2(f, izlenen / Path(f).name)
    print(f"izlenen kopya: {izlenen}/")


if __name__ == "__main__":
    main()
