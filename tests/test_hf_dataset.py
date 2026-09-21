"""
tests/test_hf_dataset.py — Hugging Face iboRotti/avsr-tr-dataset İndirme ve Yükleme Testleri
"""

import json
import os
import pathlib
import tempfile
import numpy as np
import pytest
import torch

from src.data.dataset import LipReadingDataset, pad_collate_fn
from src.data.hf_downloader import HFDatasetDownloader


def test_split_map_iborotti_disjoint():
    """split_map_iborotti.json dosyasının sıfır konuşmacı sızıntısına sahip olduğunu doğrular."""
    candidates = [
        pathlib.Path(__file__).resolve().parent.parent / "data" / "metadata" / "split_map_iborotti.json",
        pathlib.Path(__file__).resolve().parent.parent / "src" / "data" / "split_map_iborotti.json",
    ]
    split_map_path = next((c for c in candidates if c.exists()), candidates[0])
    if not split_map_path.exists():
        pytest.skip("DATA-001 temiz veri audit'i tamamlanınca split haritası yeniden üretilecek")

    with open(split_map_path, "r", encoding="utf-8") as f:
        s_map = json.load(f)

    assert len(s_map) >= 15, f"En az 15 video olmalı, bulunan: {len(s_map)}"

    # Her kanalın sadece tek bir split'te yer aldığını test et
    channel_to_splits = {}
    for vid, info in s_map.items():
        ch = info["channel"]
        sp = info["split"]
        if ch not in channel_to_splits:
            channel_to_splits[ch] = set()
        channel_to_splits[ch].add(sp)

    for ch, splits in channel_to_splits.items():
        assert len(splits) == 1, f"Konuşmacı sızıntısı tespit edildi! Kanal '{ch}' birden fazla split'e atanmış: {splits}"

    splits_present = {info["split"] for info in s_map.values()}
    assert "train" in splits_present
    assert "val" in splits_present
    assert "test" in splits_present


def test_hf_downloader_manifest_parsing(tmp_path):
    """HFDatasetDownloader manifest okuma ve filtreleme mantığını doğrular."""
    downloader = HFDatasetDownloader(target_dir=tmp_path)
    
    # Mock accepted.csv
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    accepted_csv = manifest_dir / "accepted.csv"
    accepted_csv.write_text(
        "item_id,segment_id,duration,channel,title,text\n"
        "6RBn0IVd3J4,000001,3.5,AVANGART,Kutuplasma,Deneme metni bir iki\n"
        "6RBn0IVd3J4,000002,4.0,AVANGART,Kutuplasma,Ikinci cumle burada\n"
        "yRv-y6Yg5kw,000001,5.0,Pelin Dilara Çolak,Kutsal,Pelin hanimin test cumlesi\n",
        encoding="utf-8"
    )

    rows = downloader.read_accepted_manifest()
    assert len(rows) == 3
    assert rows[0]["item_id"] == "6RBn0IVd3J4"
    assert rows[2]["channel"] == "Pelin Dilara Çolak"


def test_hf_downloader_storage_guardrail(tmp_path):
    """5 GB yerel güvenlik limitinin doğru çalıştığını ve limit aşımında indirmeyi durdurduğunu doğrular."""
    downloader = HFDatasetDownloader(target_dir=tmp_path, max_local_gb=0.001)  # 1 MB yapay limit
    # Sahte check_remote_size ile 0.05 GB (50 MB) simüle edelim
    downloader.check_remote_size = lambda: 0.05

    with pytest.raises(ValueError, match="yerel güvenlik limitini"):
        downloader.download_full()



def test_lipreading_dataset_iborotti_structure(tmp_path):
    """LipReadingDataset sınıfının transcript.txt ve metadata.json ile iboRotti yapısını okuduğunu doğrular."""
    clips_dir = tmp_path / "clips"
    clip1 = clips_dir / "6RBn0IVd3J4" / "000001"
    clip2 = clips_dir / "yRv-y6Yg5kw" / "000001"
    clip1.mkdir(parents=True)
    clip2.mkdir(parents=True)

    # Sahte 96x96 mp4 dosyaları (veya test için boş/mp4)
    # load_video_frames gerçek video beklediği için opencv/ffmpeg ile minik bir video yazalım
    import cv2
    for c_dir in [clip1, clip2]:
        mp4_path = c_dir / "mouth.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(mp4_path), fourcc, 25.0, (96, 96), isColor=False)
        for _ in range(15):
            frame = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
            out.write(frame)
        out.release()

    (clip1 / "transcript.txt").write_text("merhaba dünya bugün nasılsınız", encoding="utf-8")
    (clip2 / "metadata.json").write_text(json.dumps({"text": "güzel bir test cümlesi daha"}), encoding="utf-8")

    # Mock split map
    mock_split_map = tmp_path / "split_map.json"
    mock_split_map.write_text(json.dumps({
        "6RBn0IVd3J4": {"split": "train"},
        "yRv-y6Yg5kw": {"split": "test"}
    }), encoding="utf-8")

    # 1. Tüm veriyi yükleme
    ds_all = LipReadingDataset(data_dir=tmp_path, is_train=False, crop_size=88)
    assert len(ds_all) == 2

    # 2. Train split filtresi
    ds_train = LipReadingDataset(data_dir=tmp_path, split="train", split_map_path=mock_split_map, is_train=False)
    assert len(ds_train) == 1
    sample = ds_train[0]
    assert sample["video_id"] == "6RBn0IVd3J4"
    assert "merhaba" in sample["transcript"]
    assert sample["video"].shape == (1, 15, 88, 88)

    # 3. Test split filtresi
    ds_test = LipReadingDataset(data_dir=tmp_path, split="test", split_map_path=mock_split_map, is_train=False)
    assert len(ds_test) == 1
    assert ds_test[0]["video_id"] == "yRv-y6Yg5kw"

    # 4. DataLoader ve Collate testi
    loader = torch.utils.data.DataLoader(ds_train, batch_size=1, collate_fn=pad_collate_fn)
    batch = next(iter(loader))
    assert "videos" in batch
    assert "targets" in batch
    assert batch["videos"].shape == (1, 1, 15, 88, 88)


def test_lipreading_dataset_excludes_explicitly_rejected_hf_clip(tmp_path):
    """Metadata'da reddedilmiş bir HF klibi eğitim/eval örneği olamaz."""
    clip_dir = tmp_path / "clips" / "video-a" / "000001"
    clip_dir.mkdir(parents=True)
    (clip_dir / "mouth.mp4").write_bytes(b"placeholder")
    (clip_dir / "transcript.txt").write_text(
        "bu klip veri kümesine alınmamalı", encoding="utf-8"
    )
    (clip_dir / "metadata.json").write_text(
        json.dumps(
            {
                "text": "bu klip veri kümesine alınmamalı",
                "accepted": False,
                "quality_status": "rejected",
                "visual_quality": {"status": "rejected"},
            }
        ),
        encoding="utf-8",
    )

    dataset = LipReadingDataset(data_dir=tmp_path, is_train=False)

    assert len(dataset) == 0


def test_lipreading_dataset_fails_closed_when_explicit_split_map_is_missing(tmp_path):
    """Split istenmişse eksik harita tüm veriyi sessizce kullanmaya dönüşmemeli."""
    clip_dir = tmp_path / "clips" / "video-a" / "000001"
    clip_dir.mkdir(parents=True)
    (clip_dir / "mouth.mp4").write_bytes(b"placeholder")
    (clip_dir / "transcript.txt").write_text("geçerli bir test cümlesi", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Split haritası bulunamadı"):
        LipReadingDataset(
            data_dir=tmp_path,
            split="train",
            split_map_path=tmp_path / "missing-split-map.json",
            is_train=False,
        )


def test_hf_downloader_fails_closed_when_split_map_is_missing(tmp_path):
    """Downloader split seçimini haritasız yapmaya çalışmamalı."""
    downloader = HFDatasetDownloader(
        target_dir=tmp_path,
        split_map_path=tmp_path / "missing-split-map.json",
    )

    with pytest.raises(FileNotFoundError, match="Split haritası bulunamadı"):
        downloader._get_split_map()
