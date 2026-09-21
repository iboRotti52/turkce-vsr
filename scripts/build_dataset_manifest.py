#!/usr/bin/env python3
"""
build_dataset_manifest.py — S3 verisinden PyTorch & Auto-AVSR eğitim manifestoları oluşturucu.

data/master/ dizinindeki mouth.mp4 ve align.json dosyalarını tarar,
data/metadata/split_map.json haritasını kullanarak konuşmacı-ayrık
train.csv, val.csv ve test.csv dosyalarını üretir.

Kullanım:
    python scripts/build_dataset_manifest.py
    python scripts/build_dataset_manifest.py --master-dir data/master --output-dir data/metadata
"""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MASTER = ROOT / "data" / "master"
DEFAULT_METADATA = ROOT / "data" / "metadata"
DEFAULT_SPLIT_MAP = DEFAULT_METADATA / "split_map.json"


def main():
    parser = argparse.ArgumentParser(description="Eğitim Manifest Dosyalarını Oluşturucu")
    parser.add_argument("--master-dir", type=Path, default=DEFAULT_MASTER, help="master veri dizini")
    parser.add_argument("--split-map", type=Path, default=DEFAULT_SPLIT_MAP, help="split_map.json yolu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METADATA, help="CSV çıktılarının yazılacağı dizin")
    parser.add_argument("--max-frames", type=int, default=200, help="Azami kare sınırı (varsayılan 200 = 8 saniye @25fps)")
    args = parser.parse_args()

    if not args.split_map.exists():
        print(f"Hata: Split haritası bulunamadı: {args.split_map}", file=sys.stderr)
        sys.exit(1)

    with open(args.split_map, "r", encoding="utf-8") as f:
        split_map = json.load(f)

    print(f"Split haritası yüklendi ({len(split_map)} video).")

    if not args.master_dir.exists():
        print(f"Uyarı: {args.master_dir} dizini henüz mevcut değil veya boş. Önce S3'ten veri indirilmelidir.", file=sys.stderr)
        print("Komut: python scripts/download_s3_dataset.py --mode mouth")
        sys.exit(0)

    splits_data = {"train": [], "val": [], "test": []}
    scanned_count = 0
    missing_text = 0

    print("Veri dizini taranıyor...")
    for mouth_path in sorted(args.master_dir.glob("*/*/mouth.mp4")):
        seg_dir = mouth_path.parent
        video_id = seg_dir.parent.name
        seg_id = seg_dir.name

        align_file = seg_dir / "align.json"
        if not align_file.exists():
            missing_text += 1
            continue

        try:
            with open(align_file, "r", encoding="utf-8") as f:
                align_data = json.load(f)
            text = align_data.get("text", "").strip()
        except Exception:
            missing_text += 1
            continue

        if not text:
            missing_text += 1
            continue

        meta_file = seg_dir / "meta.json"
        duration = 0.0
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    duration = json.load(f).get("duration", 0.0)
            except Exception:
                pass

        split_info = split_map.get(video_id, {})
        split_name = split_info.get("split", "train")

        rel_video_path = str(mouth_path.relative_to(ROOT))
        splits_data[split_name].append({
            "video_path": rel_video_path,
            "transcript": text,
            "video_id": video_id,
            "seg_id": seg_id,
            "duration": duration,
        })
        scanned_count += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Manifest Özeti ---")
    for split_name, rows in splits_data.items():
        out_csv = args.output_dir / f"{split_name}.csv"
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_path", "transcript", "video_id", "seg_id", "duration"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {split_name:<6}: {len(rows):,} segment -> {out_csv}")

    print(f"\nToplam taranan: {scanned_count:,} segment. (Metinsiz/okunamayan: {missing_text})")


if __name__ == "__main__":
    main()
