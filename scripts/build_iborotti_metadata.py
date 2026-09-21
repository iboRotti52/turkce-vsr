#!/usr/bin/env python3
"""
scripts/build_iborotti_metadata.py — Generate canonical split manifests and compute hashes.
"""

import csv
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
METADATA_DIR = ROOT / "data" / "metadata"
IBOROTTI_DIR = ROOT / "data" / "iborotti"
CLIPS_DIR = IBOROTTI_DIR / "clips"
SPLIT_MAP_SRC = ROOT / "src" / "data" / "split_map_iborotti.json"
SPLIT_MAP_DEST = METADATA_DIR / "split_map_iborotti.json"

def sha256_file(filepath: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy canonical split map to data/metadata/
    with open(SPLIT_MAP_SRC, "r", encoding="utf-8") as f:
        split_map = json.load(f)
    
    with open(SPLIT_MAP_DEST, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2, ensure_ascii=False)
    print(f"Wrote {SPLIT_MAP_DEST}")

    # 2. Collect accepted samples per split
    split_records = {"train": [], "val": [], "test": []}
    
    for vid, info in sorted(split_map.items()):
        v_dir = CLIPS_DIR / vid
        if not v_dir.exists():
            continue
        split_name = info["split"]
        for s_dir in sorted(v_dir.iterdir()):
            if not s_dir.is_dir():
                continue
            meta_file = s_dir / "metadata.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            
            # Acceptance filter
            accepted = meta.get("accepted")
            qs = str(meta.get("quality_status", "")).lower()
            vs = str((meta.get("visual_quality") or {}).get("status", "")).lower()
            if not (accepted is True and qs == "accepted" and vs == "accepted"):
                continue
            
            mouth_file = s_dir / "mouth.mp4"
            if not mouth_file.exists() or mouth_file.stat().st_size == 0:
                continue
                
            text = meta.get("text", "").strip()
            # Level 1 filter
            if not text or "altyazı" in text.lower() or len(text.split()) < 2:
                continue
                
            duration = meta.get("duration", 0.0)
            
            split_records[split_name].append({
                "video_id": vid,
                "segment_id": s_dir.name,
                "video_path": str(mouth_file.relative_to(ROOT)),
                "transcript": text,
                "duration": duration,
                "channel": info["channel"],
            })

    hashes = {}
    print("\n--- Manifest Summary ---")
    for s_name in ["train", "val", "test"]:
        records = split_records[s_name]
        out_csv = METADATA_DIR / f"{s_name}.csv"
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "segment_id", "video_path", "transcript", "duration", "channel"])
            writer.writeheader()
            writer.writerows(records)
        c_hash = sha256_file(out_csv)
        hashes[f"{s_name}.csv"] = c_hash
        print(f"  {s_name:<6}: {len(records):5d} segments | SHA-256: {c_hash[:16]}... -> {out_csv}")

    split_map_hash = sha256_file(SPLIT_MAP_DEST)
    hashes["split_map_iborotti.json"] = split_map_hash
    print(f"  split_map SHA-256: {split_map_hash}")

    summary_file = METADATA_DIR / "dataset_hashes.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_id": "iboRotti/avsr-tr-dataset",
            "total_accepted_clips": sum(len(r) for r in split_records.values()),
            "train_clips": len(split_records["train"]),
            "val_clips": len(split_records["val"]),
            "test_clips": len(split_records["test"]),
            "hashes": hashes,
        }, f, indent=2)
    print(f"Wrote {summary_file}")

if __name__ == "__main__":
    main()
