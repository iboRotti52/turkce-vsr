"""build_avsr_manifest.py — state.db + data/master -> Auto-AVSR CSV manifest'leri.

Auto-AVSR satır formatı (bkz. auto_avsr_train/INSTRUCTION.md):
    dataset_name,rel_path,input_length,token_id
    token_id boşlukla ayrılmış tam sayı dizisi.

İKİ KAYNAK MODU:
  (varsayılan) state.db  -- YEREL boru hattı dönemi. DİKKAT: bulut hattı bu
        dosyaya HİÇ yazmıyor; state.db 161 videoda donmuş durumda.
  --split-map split_map.json  -- BULUT master'ı (466 video). Split bilgisi
        konusmaci_split.py'den, metin her segmentin align.json'ından gelir.

Yalnızca şu ikisinin KESİŞİMİNİ alır:
  - state.db'de qc_pass=1 olan segmentler
  - diskte mouth.mp4 dosyası GERÇEKTEN var olan segmentler
(yerel data/master S3'ten daha eski/eksik olabilir — bkz. proje notları.)

root_dir = data/ olacak şekilde tasarlandı; dataset_name="master" sabit,
çünkü data/master/{video_id}/{seg_id}/mouth.mp4 fiziksel yapısıyla birebir örtüşüyor.

Kullanım:
    .venv/bin/python scripts/build_avsr_manifest.py [--master-dir data/master] [--limit N]
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "state.db"
DEFAULT_MASTER = ROOT / "data" / "master"
DEFAULT_LABELS_DIR = ROOT / "data" / "labels"

sys.path.insert(0, str(ROOT / "vendor" / "auto_avsr_train"))


def frame_count(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER)
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    ap.add_argument("--split-map", type=Path, default=None,
                    help="konusmaci_split.py çıktısı. Verilirse state.db yerine "
                         "bu + align.json kullanılır (bulut master'ı için).")
    ap.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    ap.add_argument("--dataset-name", type=str, default="master",
                     help="manifest'e yazılacak dataset_name sütunu (root_dir/dataset_name/rel_path olarak okunur — "
                          "Modal'da S3 mount'unun klasör adıyla eşleşmeli, örn. cloud_master)")
    ap.add_argument("--limit", type=int, default=0, help="her split için en fazla N satır (sıhhat testi için)")
    ap.add_argument("--prefix", type=str, default="tr", help="çıktı dosya adı öneki (tr_train_... yerine)")
    ap.add_argument("--max-seg-frames", type=int, default=200,
                     help="bu kare sayısını aşan segmentler atlanır (MAX_CLIP_SEC=6s hedefinin üstünde kalan "
                          "aykırı/eski segmentler için — ölçtük: bazı eski segmentler 1400+ kareye kadar çıkıyor)")
    args = ap.parse_args()

    from datamodule.transforms import TextTransform

    tt = TextTransform()

    if args.split_map:
        # Bulut kaynagi: split_map.json + her segmentin align.json'i.
        # state.db kullanilmaz -- o dosya 161 videoda donmus (bkz. modul docstring).
        harita = json.loads(args.split_map.read_text())
        sayac = {"metinsiz": 0, "okunan": 0}

        def _satirlar():
            """ponytail: GENERATOR olmali, liste degil. S3/FUSE mount'unda her
            align.json ayri bir ag okumasi; listeye toplarsak --limit 300 istesen
            bile 75.276 dosya okunuyor (olculdu 2026-09-03: A10G'de 35 dk, ~$0.65
            bosa gitti). Generator olunca asagidaki limit break'i is goruyor."""
            for mouth in sorted(args.master_dir.glob("*/*/mouth.mp4")):
                seg_dir = mouth.parent
                video_id = seg_dir.parent.name
                d = harita.get(video_id)
                if not d:
                    continue
                try:
                    text = json.loads((seg_dir / "align.json").read_text())["text"]
                except Exception:
                    sayac["metinsiz"] += 1
                    continue
                sayac["okunan"] += 1
                yield (video_id, seg_dir.name, text, d["split"])

        rows = _satirlar()
        print(f"kaynak=split_map (tembel okuma, limit={args.limit or 'yok'})")
    else:
        con = sqlite3.connect(args.db_path)
        rows = con.execute(
            "select video_id, seg_id, text, split from segments where qc_pass=1 and text is not null and text != ''"
        ).fetchall()
        print(f"kaynak=state.db  segment={len(rows)}  "
              f"(UYARI: bulut hatti bu dosyaya yazmiyor -- bulut icin --split-map kullan)")

    by_split = {"train": [], "val": [], "test": []}
    atlanan_dosya_yok = 0
    atlanan_split_yok = 0
    atlanan_cok_uzun = 0

    # NOT: --limit verildiğinde erken dur — S3 (CloudBucketMount/FUSE) üzerinde
    # her segment için exists()+frame_count() ağ gecikmesi taşıyor; 6483
    # segmentin TAMAMINI taramak (limit uygulansa bile) bir sıhhat testinde
    # saatler sürebiliyor (ölçtük: 3600sn timeout'a takıldı). Limit varsa
    # her split kendi hedefine ulaşınca döngü biter.
    for video_id, seg_id, text, split in rows:
        if args.limit and all(len(v) >= args.limit for v in by_split.values()):
            break
        if split not in by_split:
            atlanan_split_yok += 1
            continue
        if args.limit and len(by_split[split]) >= args.limit:
            continue
        mouth_path = args.master_dir / video_id / seg_id / "mouth.mp4"
        if not mouth_path.exists():
            atlanan_dosya_yok += 1
            continue
        n_frames = frame_count(mouth_path)
        if n_frames <= 0:
            atlanan_dosya_yok += 1
            continue
        if n_frames > args.max_seg_frames:
            atlanan_cok_uzun += 1
            continue
        token_ids = tt.tokenize(text.strip())
        if len(token_ids) == 0:
            continue
        token_str = " ".join(str(int(t)) for t in token_ids)
        rel_path = f"{video_id}/{seg_id}/mouth.mp4"
        by_split[split].append(f"{args.dataset_name},{rel_path},{n_frames},{token_str}")

    args.labels_dir.mkdir(parents=True, exist_ok=True)
    for split, lines in by_split.items():
        if args.limit:
            lines = lines[: args.limit]
        out_path = args.labels_dir / f"{args.prefix}_{split}_transcript_lengths_seg6s.csv"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        toplam_saat = sum(int(l.split(",")[2]) for l in lines) / 25.0 / 3600.0
        print(f"{split}: {len(lines)} satır, ~{toplam_saat:.2f} saat -> {out_path}")

    if args.split_map:
        print(f"okunan align.json: {sayac['okunan']}  "
              f"(tembel okuma sayesinde toplam segmentin tamami DEGIL)")
        print(f"atlanan (align.json okunamadi): {sayac['metinsiz']}")
    print(f"atlanan (mouth.mp4 yok/bozuk): {atlanan_dosya_yok}")
    print(f"atlanan (split boş/tanımsız): {atlanan_split_yok}")
    print(f"atlanan (max-seg-frames={args.max_seg_frames} üstü aykırı segment): {atlanan_cok_uzun}")


if __name__ == "__main__":
    main()
