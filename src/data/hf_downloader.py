"""
src/data/hf_downloader.py — Hugging Face AVSR Türkçe Veri Seti İndiricisi
Depo (büyük veri): https://huggingface.co/datasets/avsr-tr-ekip/avsr-tr-dataset
Eski küçük-veri rejimi: iboRotti/avsr-tr-dataset (c0.4.0 frozen kanıtlarında referans
olarak korunur; configs/research_candidate.yaml ve full_train_manifest.json'a dokunma)
Hedef Dizin: data/iborotti/
"""

import csv
import io
import json
import os
import pathlib
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Set, Union

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_DIR = ROOT / "data" / "iborotti"
DEFAULT_SPLIT_MAP = ROOT / "data" / "metadata" / "split_map_iborotti.json"
# Büyük-veri rejimi (arkadaşın devam edeceği veri). Eski küçük-veri ID'si:
# LEGACY_REPO_ID = "iboRotti/avsr-tr-dataset"
DEFAULT_REPO_ID = "avsr-tr-ekip/avsr-tr-dataset"
LEGACY_REPO_ID = "iboRotti/avsr-tr-dataset"
DEFAULT_MAX_LOCAL_GB: float = 5.0


class HFDatasetDownloader:
    """
    Hugging Face Hub üzerindeki `avsr-tr-ekip/avsr-tr-dataset` veri kümesini
    yöneten, ultra hafif pilot indirme (< 10 MB), 5 GB güvenlik limitli tam indirme
    ve bulut senkronizasyonunu destekleyen modüler sınıf.
    `repo_id` parametresi ile eski `iboRotti/avsr-tr-dataset` de yüklenebilir.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        target_dir: Optional[Union[str, pathlib.Path]] = None,
        split_map_path: Optional[Union[str, pathlib.Path]] = None,
        hf_token: Optional[str] = None,
        max_local_gb: float = DEFAULT_MAX_LOCAL_GB,
        revision: Optional[str] = None,
        require_pinned_revision: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision
        if require_pinned_revision:
            normalized = (revision or "").strip().lower()
            if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
                raise ValueError(
                    "Large-data araştırması immutable 40-hex Hugging Face commit revision gerektirir; "
                    f"verilen revision={revision!r}"
                )
            self.revision = normalized
        self.target_dir = pathlib.Path(target_dir) if target_dir else DEFAULT_TARGET_DIR
        self.split_map_path = pathlib.Path(split_map_path) if split_map_path else DEFAULT_SPLIT_MAP
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or None
        self.max_local_gb = max_local_gb

        self.manifest_dir = self.target_dir / "manifests"
        self.clips_dir = self.target_dir / "clips"
        self._split_map: Optional[Dict[str, Any]] = None

    def _get_split_map(self) -> Dict[str, Any]:
        if self._split_map is not None:
            return self._split_map

        if self.split_map_path.exists():
            with open(self.split_map_path, "r", encoding="utf-8") as f:
                self._split_map = json.load(f)
            return self._split_map

        if self.split_map_path == DEFAULT_SPLIT_MAP:
            from src.data.split_map_data import IBOROTTI_SPLIT_MAP
            self._split_map = IBOROTTI_SPLIT_MAP
            return self._split_map

        raise FileNotFoundError(f"Split haritası bulunamadı: {self.split_map_path}")

    def download_manifests(self) -> pathlib.Path:
        """
        Manifest dosyalarını (accepted.csv, all.jsonl, all.csv) indirir veya yerelde doğrular.
        Hafif dosyalardır (~5 MB toplam).
        """
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        accepted_csv = self.manifest_dir / "accepted.csv"

        if accepted_csv.exists() and accepted_csv.stat().st_size > 1000:
            return self.manifest_dir

        # 1. Öncelikli Yöntem: huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            for fname in ["accepted.csv", "all.csv", "all.jsonl"]:
                hf_path = f"data/iborotti/manifests/{fname}"
                dest_file = self.manifest_dir / fname
                if not dest_file.exists() or dest_file.stat().st_size == 0:
                    downloaded = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=hf_path,
                        repo_type="dataset",
                        token=self.hf_token,
                        revision=self.revision,
                    )
                    import shutil
                    shutil.copyfile(downloaded, dest_file)
            return self.manifest_dir
        except Exception as e:
            # 2. Yedek Yöntem: Doğrudan raw HTTP
            print(f"huggingface_hub indirme uyarısı, raw HTTP deneniyor: {e}", file=sys.stderr)
            revision = self.revision or "main"
            base_raw = f"https://huggingface.co/datasets/{self.repo_id}/resolve/{revision}/data/iborotti/manifests"
            for fname in ["accepted.csv", "all.csv"]:
                dest_file = self.manifest_dir / fname
                if not dest_file.exists() or dest_file.stat().st_size == 0:
                    url = f"{base_raw}/{fname}"
                    req = urllib.request.Request(url, headers={"User-Agent": "AVSR-TR-Client"})
                    with urllib.request.urlopen(req) as resp, open(dest_file, "wb") as out:
                        out.write(resp.read())
            return self.manifest_dir

    def read_accepted_manifest(self) -> List[Dict[str, Any]]:
        """Kabul edilen (accepted) segment kayıtlarını liste olarak döndürür."""
        self.download_manifests()
        accepted_csv = self.manifest_dir / "accepted.csv"
        rows = []
        with open(accepted_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

    def download_pilot(
        self,
        n_samples: int = 20,
        split: Optional[str] = "train",
        max_per_video: Optional[int] = 4,
        max_duration: Optional[float] = None,
    ) -> List[pathlib.Path]:
        """
        Yerel ortam veya hızlı denemeler için SADECE N adet klibi indirir.
        Yerel diski korur (< 10 MB).
        """
        manifest = self.read_accepted_manifest()
        split_map = self._get_split_map()

        filtered_rows = []
        video_counts: Dict[str, int] = {}

        for row in manifest:
            vid = row["item_id"]
            if split:
                item_split = split_map.get(vid, {}).get("split")
                if item_split != split:
                    continue

            if max_duration is not None:
                dur = float(row.get("duration", 0.0))
                if dur > 0 and dur > max_duration:
                    continue

            curr_v_count = video_counts.get(vid, 0)
            if max_per_video and curr_v_count >= max_per_video:
                continue

            video_counts[vid] = curr_v_count + 1
            filtered_rows.append(row)
            if len(filtered_rows) >= n_samples:
                break

        print(f"📥 [HF PILOT] {len(filtered_rows)} segment indirilecek (split={split})...")
        downloaded_clip_dirs: List[pathlib.Path] = []

        from huggingface_hub import hf_hub_download
        for i, row in enumerate(filtered_rows, start=1):
            item_id = row["item_id"]
            seg_id = row["segment_id"]
            clip_dir = self.clips_dir / item_id / seg_id
            clip_dir.mkdir(parents=True, exist_ok=True)

            # mouth.mp4, metadata.json, transcript.txt
            files = ["mouth.mp4", "metadata.json", "transcript.txt"]
            all_ok = True
            for f in files:
                target_f = clip_dir / f
                if target_f.exists() and target_f.stat().st_size > 0:
                    continue

                hf_rel = f"data/iborotti/clips/{item_id}/{seg_id}/{f}"
                try:
                    cached_p = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=hf_rel,
                        repo_type="dataset",
                        token=self.hf_token,
                        revision=self.revision,
                    )
                    import shutil
                    shutil.copyfile(cached_p, target_f)
                except Exception as ex:
                    # Alternatif raw indirme
                    try:
                        revision = self.revision or "main"
                        raw_url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/{revision}/{hf_rel}"
                        req = urllib.request.Request(raw_url, headers={"User-Agent": "AVSR-TR-Client"})
                        with urllib.request.urlopen(req) as resp, open(target_f, "wb") as out:
                            out.write(resp.read())
                    except Exception as raw_ex:
                        print(f"Dosya indirilemedi [{hf_rel}]: {raw_ex}", file=sys.stderr)
                        all_ok = False
                        break

            if all_ok:
                downloaded_clip_dirs.append(clip_dir)
                if i % 5 == 0 or i == len(filtered_rows):
                    print(f"  [{i}/{len(filtered_rows)}] {item_id}_{seg_id} hazır.")

        return downloaded_clip_dirs

    def check_remote_size(self) -> float:
        """HF deposunun toplam boyutunu GB cinsinden döner."""
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.hf_token)
            info = api.dataset_info(self.repo_id, revision=self.revision, files_metadata=True)
            total_bytes = sum(f.size for f in info.siblings if f.size is not None)
            return total_bytes / (1024 ** 3)
        except Exception as e:
            print(f"Uyarı: HF dataset boyutu sorgulanamadı: {e}", file=sys.stderr)
            return 0.0

    def download_full(
        self,
        allow_patterns: Optional[List[str]] = None,
        force_exceed_limit: bool = False,
    ) -> pathlib.Path:
        """
        Kullanıcı direktifi uyarınca azami 5.0 GB limitini gözeterek
        tüm depoyu yerel diske veya Modal konteynerine senkronize eder.
        """
        size_gb = self.check_remote_size()
        if size_gb > 0:
            print(f"📊 [HF DEPO BİLGİSİ] Boyut: {size_gb * 1024:.1f} MB ({size_gb:.3f} GB) | Yerel Güvenlik Limiti: {self.max_local_gb:.1f} GB")
            if size_gb > self.max_local_gb and not force_exceed_limit:
                raise ValueError(
                    f"Veri kümesi boyutu ({size_gb:.2f} GB) belirlenen yerel güvenlik limitini ({self.max_local_gb:.1f} GB) aşıyor! "
                    f"Yerel depolamayı korumak için indirme durduruldu. Modal üzerinde eğitmeyi veya limiti artırmayı değerlendirin."
                )

        from huggingface_hub import snapshot_download
        print(
            f"🚀 [HF FULL SYNC] '{self.repo_id}' revision={self.revision or 'main'} "
            f"deposu '{self.target_dir}' dizinine indiriliyor..."
        )

        patterns = allow_patterns or ["data/iborotti/*"]

        # Doğrudan target_dir içine indirme (çift kopyalamayı ve disk şişmesini önler)
        if self.target_dir.name == "iborotti" and self.target_dir.parent.name == "data":
            root_dir = self.target_dir.parent.parent
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                allow_patterns=patterns,
                token=self.hf_token,
                revision=self.revision,
                local_dir=str(root_dir),
            )
        else:
            cache_dir = snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                allow_patterns=patterns,
                token=self.hf_token,
                revision=self.revision,
            )
            import shutil
            src_inner = pathlib.Path(cache_dir) / "data" / "iborotti"
            if src_inner.exists():
                for child in src_inner.iterdir():
                    dest_child = self.target_dir / child.name
                    if child.is_dir():
                        shutil.copytree(child, dest_child, dirs_exist_ok=True)
                    else:
                        dest_child.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(child, dest_child)

        print(f"✅ Tam indirme tamamlandı: {self.target_dir}")
        return self.target_dir

    def get_available_clips(self, split: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Yerel diskte halihazırda bulunan geçerli klipleri döndürür.
        """
        split_map = self._get_split_map()
        available = []
        if not self.clips_dir.exists():
            return available

        for mouth_p in sorted(self.clips_dir.glob("*/*/mouth.mp4")):
            seg_dir = mouth_p.parent
            vid = seg_dir.parent.name
            seg_id = seg_dir.name

            if split:
                item_split = split_map.get(vid, {}).get("split")
                if item_split != split:
                    continue

            text_p = seg_dir / "transcript.txt"
            meta_p = seg_dir / "metadata.json"

            transcript = ""
            if text_p.exists():
                try:
                    with open(text_p, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()
                except Exception:
                    pass
            elif meta_p.exists():
                try:
                    with open(meta_p, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                        transcript = m_data.get("text", "").strip()
                except Exception:
                    pass

            if not transcript:
                continue

            available.append({
                "video_path": str(mouth_p),
                "transcript": transcript,
                "video_id": vid,
                "seg_id": seg_id,
                "metadata_path": str(meta_p) if meta_p.exists() else None,
            })
        return available
