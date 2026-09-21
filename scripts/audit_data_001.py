#!/usr/bin/env python3
"""
scripts/audit_data_001.py — Comprehensive Data & Evaluation Foundation Audit
Analyzes all clips, manifests, speakers, text, and visual quality for DATA-001.
"""

import csv
import io
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict
import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "iborotti"
MANIFESTS_DIR = DATA_DIR / "manifests"
CLIPS_DIR = DATA_DIR / "clips"

def run_audit():
    report = {}
    
    # 1. Manifest Analysis
    print("--- 1. MANIFEST ANALYSIS ---")
    manifest_counts = {}
    manifest_items = defaultdict(set)
    manifest_records = {}
    
    all_jsonl = MANIFESTS_DIR / "all.jsonl"
    if all_jsonl.exists():
        with open(all_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                key = (record["item_id"], record["segment_id"])
                manifest_records[key] = record
                manifest_items["all.jsonl"].add(key)
        manifest_counts["all.jsonl"] = len(manifest_items["all.jsonl"])
    
    for csv_name in ["all.csv", "accepted.csv", "rejected.csv", "review.csv"]:
        p = MANIFESTS_DIR / csv_name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    key = (r["item_id"], r["segment_id"])
                    manifest_items[csv_name].add(key)
            manifest_counts[csv_name] = len(manifest_items[csv_name])
            
    print("Manifest row counts:")
    for k, v in manifest_counts.items():
        print(f"  {k}: {v}")

    # 2. Disk Analysis
    print("\n--- 2. DISK ANALYSIS ---")
    disk_clips = {} # key -> dir_path
    video_dirs = sorted([d for d in CLIPS_DIR.iterdir() if d.is_dir()])
    print(f"Video directories on disk: {len(video_dirs)}")
    
    missing_files = defaultdict(int)
    zero_byte_files = defaultdict(int)
    durations = []
    frame_counts_list = []
    fps_counts = Counter()
    res_counts = Counter()
    corrupted_count = 0
    
    disk_metadata = {}
    disk_transcripts = {}
    
    for v_dir in video_dirs:
        for s_dir in sorted([s for s in v_dir.iterdir() if s.is_dir()]):
            key = (v_dir.name, s_dir.name)
            disk_clips[key] = s_dir
            
            mouth_mp4 = s_dir / "mouth.mp4"
            transcript_txt = s_dir / "transcript.txt"
            metadata_json = s_dir / "metadata.json"
            
            if not mouth_mp4.exists():
                missing_files["mouth.mp4"] += 1
            elif mouth_mp4.stat().st_size == 0:
                zero_byte_files["mouth.mp4"] += 1
                
            if not transcript_txt.exists():
                missing_files["transcript.txt"] += 1
            elif transcript_txt.stat().st_size == 0:
                zero_byte_files["transcript.txt"] += 1
            else:
                disk_transcripts[key] = transcript_txt.read_text(encoding="utf-8").strip()
                
            if not metadata_json.exists():
                missing_files["metadata.json"] += 1
            elif metadata_json.stat().st_size == 0:
                zero_byte_files["metadata.json"] += 1
            else:
                try:
                    disk_metadata[key] = json.loads(metadata_json.read_text(encoding="utf-8"))
                except Exception:
                    zero_byte_files["corrupt_metadata"] += 1

    print(f"Total clips on disk: {len(disk_clips)}")
    print(f"Missing files on disk: {dict(missing_files)}")
    for k, s_dir in disk_clips.items():
        if not (s_dir / "transcript.txt").exists():
            print(f"  Clip missing transcript.txt: {k} at {s_dir}")
    print(f"Zero byte files: {dict(zero_byte_files)}")

    # 3. Discrepancy between Manifest and Disk
    disk_keys = set(disk_clips.keys())
    jsonl_keys = set(manifest_items.get("all.jsonl", set()))
    accepted_keys = set(manifest_items.get("accepted.csv", set()))
    rejected_keys = set(manifest_items.get("rejected.csv", set()))
    review_keys = set(manifest_items.get("review.csv", set()))
    
    print("\n--- 3. MANIFEST VS DISK DISCREPANCIES ---")
    print(f"Clips on disk but not in all.jsonl: {len(disk_keys - jsonl_keys)}")
    print(f"Clips in all.jsonl but not on disk: {len(jsonl_keys - disk_keys)}")
    print(f"Clips on disk that are in accepted.csv: {len(disk_keys & accepted_keys)}")
    print(f"Clips on disk that are in rejected.csv: {len(disk_keys & rejected_keys)}")
    print(f"Clips on disk that are in review.csv: {len(disk_keys & review_keys)}")
    unclassified_on_disk = disk_keys - (accepted_keys | rejected_keys | review_keys)
    print(f"Clips on disk not in accepted/rejected/review: {len(unclassified_on_disk)}")
    
    # 4. Metadata Acceptance Status on Disk
    print("\n--- 4. METADATA ACCEPTANCE ON DISK ---")
    quality_status_counter = Counter()
    accepted_field_counter = Counter()
    for k, meta in disk_metadata.items():
        qs = meta.get("quality_status")
        ac = meta.get("accepted")
        quality_status_counter[str(qs)] += 1
        accepted_field_counter[str(ac)] += 1
    print(f"quality_status on disk: {quality_status_counter}")
    print(f"accepted field on disk: {accepted_field_counter}")
    
    # Check if disk metadata accepted status matches accepted.csv
    meta_accepted_keys = {k for k, meta in disk_metadata.items() if meta.get("accepted") is True}
    print(f"Clips with accepted=True in disk metadata: {len(meta_accepted_keys)}")
    print(f"Overlap of meta accepted=True and accepted.csv: {len(meta_accepted_keys & accepted_keys)}")
    print(f"meta accepted=True but not in accepted.csv: {len(meta_accepted_keys - accepted_keys)}")
    print(f"accepted.csv on disk but meta accepted != True: {len((accepted_keys & disk_keys) - meta_accepted_keys)}")

    # 5. Video Properties Sampling (and exhaustive inspection where cheap)
    print("\n--- 5. VIDEO PROPERTIES (EXHAUSTIVE INSPECTION) ---")
    duration_stats = []
    frame_counts_list = []
    for key, path in disk_clips.items():
        mp4_file = path / "mouth.mp4"
        cap = cv2.VideoCapture(str(mp4_file))
        if not cap.isOpened():
            corrupted_count += 1
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        fps_counts[round(fps, 2)] += 1
        res_counts[(width, height)] += 1
        frame_counts_list.append(n_frames)
        dur = n_frames / fps if fps > 0 else 0
        duration_stats.append(dur)

    print(f"Corrupted video files: {corrupted_count}")
    print(f"FPS distribution: {fps_counts}")
    print(f"Resolution distribution: {res_counts}")
    print(f"Duration (seconds): min={min(duration_stats):.2f}, max={max(duration_stats):.2f}, mean={np.mean(duration_stats):.2f}, median={np.median(duration_stats):.2f}")
    print(f"Frame counts: min={min(frame_counts_list)}, max={max(frame_counts_list)}, mean={np.mean(frame_counts_list):.1f}, median={np.median(frame_counts_list)}")
    print(f"Total video duration on disk: {sum(duration_stats)/3600:.2f} hours ({sum(duration_stats)/60:.1f} minutes)")

    # 6. Speaker / Video / Channel Distribution
    print("\n--- 6. SPEAKER / VIDEO / CHANNEL DISTRIBUTION ---")
    video_channel_map = {}
    video_title_map = {}
    video_clip_count = Counter()
    video_accepted_count = Counter()
    
    for key, meta in disk_metadata.items():
        v_id, seg_id = key
        video_channel_map[v_id] = meta.get("channel", "unknown")
        video_title_map[v_id] = meta.get("title", "unknown")
        video_clip_count[v_id] += 1
        if meta.get("accepted") is True:
            video_accepted_count[v_id] += 1
            
    print(f"Total distinct videos on disk: {len(video_channel_map)}")
    print("Video breakdown:")
    for v_id in sorted(video_channel_map.keys()):
        ch = video_channel_map[v_id]
        tot = video_clip_count[v_id]
        acc = video_accepted_count[v_id]
        print(f"  Video {v_id:12s} | Channel: {ch:32s} | Total clips: {tot:4d} | Accepted clips: {acc:4d}")

    channel_clip_count = Counter()
    channel_video_count = defaultdict(set)
    for v_id, ch in video_channel_map.items():
        channel_clip_count[ch] += video_clip_count[v_id]
        channel_video_count[ch].add(v_id)
        
    print("\nChannel breakdown:")
    for ch, count in channel_clip_count.most_common():
        print(f"  {ch:32s}: {count:4d} clips across {len(channel_video_count[ch])} video(s): {channel_video_count[ch]}")

    # 7. Text & Vocabulary Analysis
    print("\n--- 7. TEXT & VOCABULARY ANALYSIS ---")
    all_chars = Counter()
    all_words = Counter()
    text_lengths_chars = []
    text_lengths_words = []
    empty_transcripts = 0
    accepted_words = Counter()
    
    for key, transcript in disk_transcripts.items():
        t = transcript.strip()
        if not t:
            empty_transcripts += 1
            continue
        text_lengths_chars.append(len(t))
        words = t.split()
        text_lengths_words.append(len(words))
        for c in t:
            all_chars[c] += 1
        for w in words:
            w_clean = w.lower().strip(".,!?:;\"'()[]{}«»-–—")
            if w_clean:
                all_words[w_clean] += 1
                if key in meta_accepted_keys:
                    accepted_words[w_clean] += 1

    print(f"Empty transcripts: {empty_transcripts}")
    print(f"Transcript character lengths: min={min(text_lengths_chars)}, max={max(text_lengths_chars)}, mean={np.mean(text_lengths_chars):.1f}")
    print(f"Transcript word counts: min={min(text_lengths_words)}, max={max(text_lengths_words)}, mean={np.mean(text_lengths_words):.1f}")
    print(f"Unique characters: {len(all_chars)}")
    
    turkish_std_alphabet = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ 0123456789.,!?:;\"'()[]{}«»-–—\n\r\t")
    foreign_chars = {c: count for c, count in all_chars.items() if c not in turkish_std_alphabet}
    print(f"Foreign / unexpected characters: {foreign_chars}")
    
    print(f"Total word tokens (all): {sum(all_words.values())}, Unique words: {len(all_words)}")
    print(f"Total word tokens (accepted): {sum(accepted_words.values())}, Unique words (accepted): {len(accepted_words)}")
    print(f"Top 20 words in accepted: {accepted_words.most_common(20)}")

    return {
        "manifest_counts": manifest_counts,
        "disk_clip_count": len(disk_clips),
        "fps_counts": dict(fps_counts),
        "res_counts": {str(k): v for k, v in res_counts.items()},
        "duration_hours": sum(duration_stats)/3600,
        "accepted_disk_count": len(meta_accepted_keys),
        "total_unique_words": len(all_words),
        "accepted_unique_words": len(accepted_words),
        "videos": {v: {"channel": video_channel_map[v], "total": video_clip_count[v], "accepted": video_accepted_count[v]} for v in video_channel_map},
    }

if __name__ == "__main__":
    res = run_audit()
    with open(ROOT / "research" / "data_audit_summary.json", "w", encoding="utf-8") as fp:
        json.dump(res, fp, indent=2, ensure_ascii=False)
    print("\nSaved research/data_audit_summary.json successfully.")
