#!/usr/bin/env python3
"""
download_s3_dataset.py — S3 Türkçe Dudak Okuma Veri Kümesi İndiricisi

S3 Kovası: s3://lipreading-data-emre2026/
Hedef Dizin: data/master/

Kullanım:
    # 1. Yalnızca model eğitimi için gereken dosyalar (mouth.mp4 + align.json + meta.json, ~4.7 GiB):
    python scripts/download_s3_dataset.py --mode mouth

    # 2. Önceden eğitilmiş temel model kontrol noktası (vsr_trlrs3_base.pth, ~1 GiB):
    python scripts/download_s3_dataset.py --mode ckpt

    # 3. Yalnızca transkriptler ve metadata (align.json + meta.json, ~70 MB):
    python scripts/download_s3_dataset.py --mode align

    # 4. Yalnızca test veya doğrulama kümesi için:
    python scripts/download_s3_dataset.py --mode mouth --split test

    # 5. Hızlı deneme için ilk 50 segment:
    python scripts/download_s3_dataset.py --mode mouth --limit 50

    # 6. Tüm veri kümesi (face.mp4 ve audio.wav dahil, ~38 GiB):
    python scripts/download_s3_dataset.py --mode all
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys
import threading
from typing import Dict, List, Optional, Set, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_DATA_DIR = ROOT / "data" / "master"
DEFAULT_CKPT_DIR = ROOT / "checkpoints"
DEFAULT_METADATA_DIR = ROOT / "data" / "metadata"

_thread_local = threading.local()


def load_env_file(env_path: pathlib.Path) -> Dict[str, str]:
    """Basit ve bağımlılıksız .env ayrıştırıcı."""
    env = {}
    if not env_path.exists():
        return env
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                val = val.strip().strip("'\"")
                env[key.strip()] = val
    return env


def get_s3_client(aws_access_key: str, aws_secret_key: str, region_name: str):
    """Her thread için izole edilmiş boto3 S3 istemcisi döndürür."""
    if not hasattr(_thread_local, "s3_client"):
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            print("Hata: boto3 kütüphanesi kurulu değil. Lütfen yükleyin: pip install boto3", file=sys.stderr)
            sys.exit(1)

        cfg = Config(max_pool_connections=100, retries={"max_attempts": 5})
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name,
        )
        _thread_local.s3_client = session.client("s3", config=cfg)
    return _thread_local.s3_client


def download_single_object(
    task: Tuple[str, pathlib.Path, str, str, str, str]
) -> int:
    """
    Tek bir S3 nesnesini indirir.
    Dönüş: 1 (indirildi), 0 (zaten vardı), -1 (hata)
    """
    key, target_path, bucket_name, access_key, secret_key, region = task
    if target_path.exists() and target_path.stat().st_size > 0:
        return 0

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3 = get_s3_client(access_key, secret_key, region)
        s3.download_file(bucket_name, key, str(target_path))
        return 1
    except Exception as e:
        # print(f"Hata [{key}]: {e}", file=sys.stderr)
        return -1


def run_parallel_download(
    tasks: List[Tuple[str, pathlib.Path, str, str, str, str]],
    workers: int = 48,
):
    """İşleri ThreadPoolExecutor ile paralel indirir ve ilerlemeyi raporlar."""
    total = len(tasks)
    if total == 0:
        print("İndirilecek hedef dosya bulunamadı.")
        return

    print(f"\nToplam {total:,} dosya {workers} iş parçacığıyla indiriliyor...")
    success = skipped = errors = 0

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        for i, status in enumerate(executor.map(download_single_object, tasks), 1):
            if status == 1:
                success += 1
            elif status == 0:
                skipped += 1
            else:
                errors += 1

            if i % 1000 == 0 or i == total:
                percent = (i / total) * 100
                print(
                    f"[{i:,}/{total:,} (%{percent:.1f})] · Yeni indirilen: {success:,} · Zaten var: {skipped:,} · Hata: {errors}",
                    flush=True,
                )

    print("\n--- İndirme Tamamlandı ---")
    print(f"Başarıyla indirilen : {success:,}")
    print(f"Zaten mevcut olan   : {skipped:,}")
    print(f"Hata oluşan         : {errors}")


def get_video_keys_from_s3(
    s3, bucket: str, prefix: str, video_ids: Optional[Set[str]] = None
) -> List[str]:
    """S3'ten nesne anahtarlarını listeler."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    paginate_args = {"Bucket": bucket, "Prefix": prefix}

    print("S3 nesneleri taranıyor...", flush=True)
    count = 0
    for page in paginator.paginate(**paginate_args):
        for item in page.get("Contents", []):
            k = item["Key"]
            if video_ids:
                # master/<video_id>/...
                parts = k.split("/")
                if len(parts) > 1 and parts[1] not in video_ids:
                    continue
            keys.append(k)
            count += 1
            if count % 20000 == 0:
                print(f"  {count:,} nesne bulundu...", flush=True)
    return keys


def main():
    parser = argparse.ArgumentParser(
        description="S3 Türkçe Dudak Okuma Veri Kümesi İndiricisi"
    )
    parser.add_argument(
        "--mode",
        choices=["mouth", "ckpt", "align", "all"],
        default="mouth",
        help="İndirme modu: 'mouth' (~4.7GB mouth+align+meta), 'ckpt' (temel model), 'align' (sadece transkript), 'all' (tüm veri ~38GB)",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="S3 kova adı (varsayılan: .env içindeki S3_BUCKET_NAME veya 'lipreading-data-emre2026')",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Yalnızca belirli bir veri ayrımı için indir",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="En fazla N adet dosya indir (test amaçlı sınırlandırma)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=48,
        help="Paralel indirme iş parçacığı sayısı (varsayılan: 48)",
    )
    parser.add_argument(
        "--dest",
        type=pathlib.Path,
        default=DEFAULT_DATA_DIR,
        help="Verilerin kaydedileceği hedef dizin (varsayılan: data/master)",
    )
    args = parser.parse_args()

    # Ortam değişkenleri
    env = load_env_file(DEFAULT_ENV_FILE)
    bucket = (
        args.bucket
        or os.environ.get("S3_BUCKET_NAME")
        or env.get("S3_BUCKET_NAME")
        or "lipreading-data-emre2026"
    )
    region = (
        os.environ.get("AWS_DEFAULT_REGION")
        or env.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or env.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or env.get("AWS_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        print(
            "Hata: AWS kimlik bilgileri bulunamadı.\n"
            "Lütfen .env dosyasını doldurun veya AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY ortam değişkenlerini tanımlayın.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Mod: Checkpoint İndirme
    if args.mode == "ckpt":
        DEFAULT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_key = "ckpt/vsr_trlrs3_base.pth"
        target_file = DEFAULT_CKPT_DIR / "vsr_trlrs3_base.pth"
        print(f"Temel model kontrol noktası indiriliyor: s3://{bucket}/{ckpt_key} -> {target_file}")
        tasks = [(ckpt_key, target_file, bucket, access_key, secret_key, region)]
        run_parallel_download(tasks, workers=1)
        return

    # 2. Split Filtreleme
    selected_videos: Optional[Set[str]] = None
    if args.split != "all":
        split_map_file = DEFAULT_METADATA_DIR / "split_map.json"
        if split_map_file.exists():
            with open(split_map_file, "r", encoding="utf-8") as f:
                s_map = json.load(f)
            selected_videos = {
                vid for vid, info in s_map.items() if info.get("split") == args.split
            }
            print(f"'{args.split}' ayrımı için {len(selected_videos)} video filtrelendi.")
        else:
            print(f"Uyarı: {split_map_file} bulunamadı, tüm videolar indirilecek.")

    # Uzantı / dosya adı filtreleri
    if args.mode == "mouth":
        allowed_suffixes = ("mouth.mp4", "align.json", "meta.json")
    elif args.mode == "align":
        allowed_suffixes = ("align.json", "meta.json")
    else:  # all
        allowed_suffixes = ("mouth.mp4", "align.json", "meta.json", "face.mp4", "audio.wav")

    # Envanter dosyası varsa yerel tarama ile API çağrısından tasarruf et
    envanter_file = DEFAULT_METADATA_DIR / "envanter.json"
    tasks = []

    if envanter_file.exists():
        print(f"Yerel envanter dosyası ({envanter_file.name}) kullanılarak dosya listesi hazırlanıyor...")
        with open(envanter_file, "r", encoding="utf-8") as f:
            envanter = json.load(f)

        for vid, files_dict in envanter.get("videolar", {}).items():
            if selected_videos and vid not in selected_videos:
                continue
            # Bu video için S3'ten segment anahtarları listele
            # Not: envanter segment id'lerini tutmadığı için hızlı API listesi
            pass

    # S3'ten doğrudan listele
    client = get_s3_client(access_key, secret_key, region)
    all_keys = get_video_keys_from_s3(
        client, bucket, prefix="master/", video_ids=selected_videos
    )

    prefix_len = len("master/")
    for key in all_keys:
        if any(key.endswith(sfx) for sfx in allowed_suffixes):
            rel_path = key[prefix_len:]
            target_path = args.dest / rel_path
            tasks.append((key, target_path, bucket, access_key, secret_key, region))
            if args.limit and len(tasks) >= args.limit:
                break

    print(f"Eşleşen toplam hedef dosya sayısı: {len(tasks):,}")
    run_parallel_download(tasks, workers=args.workers)


if __name__ == "__main__":
    main()
