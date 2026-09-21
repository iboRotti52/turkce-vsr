"""
src/data/s3_downloader.py — Çoklu İş Parçacıklı S3 Veri İndiricisi
S3 Kovası: s3://lipreading-data-emre2026/
Hedef Dizin: data/master/
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_MASTER_DIR = ROOT / "data" / "master"
DEFAULT_CKPT_DIR = ROOT / "checkpoints"
DEFAULT_METADATA_DIR = ROOT / "data" / "metadata"

_thread_local = threading.local()


def load_env_credentials(env_path: Optional[pathlib.Path] = None) -> Dict[str, str]:
    """Çevre değişkenlerinden veya .env dosyasından AWS kimlik bilgilerini yükler."""
    env = {}
    path = env_path or DEFAULT_ENV_FILE
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip("'\"")

    bucket = os.environ.get("S3_BUCKET_NAME") or env.get("S3_BUCKET_NAME") or "lipreading-data-emre2026"
    region = os.environ.get("AWS_DEFAULT_REGION") or env.get("AWS_DEFAULT_REGION") or "us-east-1"
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or env.get("AWS_ACCESS_KEY_ID") or ""
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or env.get("AWS_SECRET_ACCESS_KEY") or ""

    return {
        "bucket": bucket,
        "region": region,
        "access_key": access_key,
        "secret_key": secret_key,
    }


def get_thread_s3_client(access_key: str, secret_key: str, region_name: str):
    """İş parçacığı (thread) başına yeniden kullanılan izole edilmiş boto3 S3 istemcisi."""
    if not hasattr(_thread_local, "client"):
        import boto3
        from botocore.config import Config

        cfg = Config(max_pool_connections=50, retries={"max_attempts": 5})
        session = boto3.Session(
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
            region_name=region_name or None,
        )
        _thread_local.client = session.client("s3", config=cfg)
    return _thread_local.client


def _download_task(task: Tuple[str, pathlib.Path, str, str, str, str]) -> int:
    """
    Tek bir S3 nesnesi indirme görevi (kesintilere karşı atomik).
    Dönüş: 1 (başarılı yeni indirme), 0 (zaten var), -1 (hata)
    """
    key, target_path, bucket, access_key, secret_key, region = task
    if target_path.exists() and target_path.stat().st_size > 0:
        return 0

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}_{threading.get_ident()}")
    try:
        s3 = get_thread_s3_client(access_key, secret_key, region)
        s3.download_file(bucket, key, str(tmp_path))
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.replace(target_path)
            return 1
        return -1
    except Exception as e:
        print(f"S3 indirme hatası [{key}]: {e}", file=sys.stderr)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return -1


class S3DatasetDownloader:
    """
    AWS S3 üzerinden `mouth.mp4` ve `align.json` dosyalarını
    paralel iş parçacıklarıyla indiren modüler sınıf.
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        master_dir: Optional[pathlib.Path] = None,
        metadata_dir: Optional[pathlib.Path] = None,
    ):
        creds = load_env_credentials()
        self.bucket = bucket or creds["bucket"]
        self.region = region or creds["region"]
        self.access_key = access_key or creds["access_key"]
        self.secret_key = secret_key or creds["secret_key"]
        self.master_dir = pathlib.Path(master_dir or DEFAULT_MASTER_DIR)
        self.master_dir.mkdir(parents=True, exist_ok=True)

        # Metadata dizini tespiti (yerel veya Modal konteyner yolu)
        meta_cand = metadata_dir or os.environ.get("METADATA_DIR")
        if meta_cand:
            self.metadata_dir = pathlib.Path(meta_cand)
        elif (ROOT / "data" / "metadata").exists():
            self.metadata_dir = ROOT / "data" / "metadata"
        elif pathlib.Path("/root/data/metadata").exists():
            self.metadata_dir = pathlib.Path("/root/data/metadata")
        else:
            self.metadata_dir = DEFAULT_METADATA_DIR

    def _create_task(self, s3_key: str, local_path: pathlib.Path) -> Tuple[str, pathlib.Path, str, str, str, str]:
        return (s3_key, local_path, self.bucket, self.access_key, self.secret_key, self.region)

    def download_checkpoint(self, dest_file: Optional[pathlib.Path] = None) -> pathlib.Path:
        """Temel model ağırlığını (vsr_trlrs3_base.pth) indirir."""
        target = dest_file or (DEFAULT_CKPT_DIR / "vsr_trlrs3_base.pth")
        target.parent.mkdir(parents=True, exist_ok=True)
        key = "ckpt/vsr_trlrs3_base.pth"
        print(f"Checkpoint indiriliyor: s3://{self.bucket}/{key} -> {target}")
        res = _download_task(self._create_task(key, target))
        if res == -1:
            raise RuntimeError(f"Checkpoint indirilemedi: {key}")
        print("Checkpoint hazır.")
        return target

    def download_pilot_dataset(
        self,
        n_segments: int = 50,
        split: Optional[str] = None,
        workers: int = 16,
        include_meta: bool = True,
        max_per_video: Optional[int] = None,
    ) -> List[pathlib.Path]:
        """
        Model geliştirme, doğrulama ve testler için hızlı pilot veri kümesi indirir.
        Her segment için mouth.mp4, align.json ve istenirse meta.json indirilir.
        max_per_video: Konuşmacı çeşitliliği sağlamak için tek bir videodan alınacak azami segment sayısı.
        """
        import boto3

        session = boto3.Session(
            aws_access_key_id=self.access_key or None,
            aws_secret_access_key=self.secret_key or None,
            region_name=self.region or None,
        )
        s3 = session.client("s3")

        # Split haritasını yükle (varsa)
        split_map_file = self.metadata_dir / "split_map.json"
        allowed_videos: Optional[Set[str]] = None
        if split and split_map_file.exists():
            with open(split_map_file, "r", encoding="utf-8") as f:
                s_map = json.load(f)
            allowed_videos = {vid for vid, info in s_map.items() if info.get("split") == split}

        tasks = []
        paginator = s3.get_paginator("list_objects_v2")
        segments_collected = 0
        seen_segments = set()
        video_counts: Dict[str, int] = {}

        print(f"Pilot veri kümesi taranıyor (Hedef: {n_segments} segment, Split: {split or 'tümü'})...")
        for page in paginator.paginate(Bucket=self.bucket, Prefix="master/"):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith("/mouth.mp4"):
                    continue

                # key: master/<video_id>/<seg_id>/mouth.mp4
                parts = key.split("/")
                if len(parts) < 4:
                    continue
                vid = parts[1]
                seg = parts[2]
                seg_uid = f"{vid}/{seg}"

                if allowed_videos is not None and vid not in allowed_videos:
                    continue

                if seg_uid in seen_segments:
                    continue

                if max_per_video is not None and video_counts.get(vid, 0) >= max_per_video:
                    continue

                seen_segments.add(seg_uid)
                video_counts[vid] = video_counts.get(vid, 0) + 1
                seg_dir = self.master_dir / vid / seg

                # mouth.mp4
                tasks.append(self._create_task(key, seg_dir / "mouth.mp4"))
                # align.json
                align_key = f"master/{vid}/{seg}/align.json"
                tasks.append(self._create_task(align_key, seg_dir / "align.json"))
                # meta.json
                if include_meta:
                    meta_key = f"master/{vid}/{seg}/meta.json"
                    tasks.append(self._create_task(meta_key, seg_dir / "meta.json"))

                segments_collected += 1
                if segments_collected >= n_segments:
                    break
            if segments_collected >= n_segments:
                break

        print(f"Toplam {segments_collected} segment ({len(video_counts)} farklı video) için {len(tasks)} dosya indiriliyor...")
        downloaded_dirs = []
        with cf.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_download_task, tasks))

        success = sum(1 for r in results if r == 1)
        skipped = sum(1 for r in results if r == 0)
        errors = sum(1 for r in results if r == -1)

        for seg_uid in seen_segments:
            vid, seg = seg_uid.split("/")
            d = self.master_dir / vid / seg
            if (d / "mouth.mp4").exists() and (d / "align.json").exists():
                downloaded_dirs.append(d)

        print(f"İndirme Tamamlandı: {len(downloaded_dirs)} geçerli segment hazır. (Yeni: {success}, Önceden var: {skipped}, Hata: {errors})")
        return downloaded_dirs


def main():
    parser = argparse.ArgumentParser(description="S3 Downloader CLI")
    parser.add_argument("--pilot", type=int, default=0, help="Pilot segment sayısı (örn: 20)")
    parser.add_argument("--split", type=str, default=None, help="Belirli split (train, val, test)")
    parser.add_argument("--ckpt", action="store_true", help="Temel model ağırlığını indir")
    parser.add_argument("--workers", type=int, default=16, help="İş parçacığı sayısı")
    args = parser.parse_args()

    downloader = S3DatasetDownloader()
    if args.ckpt:
        downloader.download_checkpoint()
    if args.pilot > 0:
        downloader.download_pilot_dataset(n_segments=args.pilot, split=args.split, workers=args.workers)


if __name__ == "__main__":
    main()
