"""
tests/test_dataset.py — PyTorch Dataset, Augmentation ve Collate Testleri
"""

import pathlib
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LipReadingDataset, pad_collate_fn
from src.models.vsr_conformer import VSRConformerModel
from src.vocab.turkish_vocab import VOCAB_SIZE

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "master"


def test_dataset_loading_and_augmentations():
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*/*/mouth.mp4"))) == 0:
        pytest.skip("data/master dizininde test için video bulunamadı.")

    dataset = LipReadingDataset(
        master_dir=DATA_DIR,
        is_train=True,
        crop_size=88,
        apply_horizontal_flip=True,
        apply_time_masking=True,
    )

    assert len(dataset) > 0
    sample = dataset[0]

    assert "video" in sample
    assert "target" in sample
    assert "transcript" in sample

    video = sample["video"]
    assert video.dim() == 4  # (1, T, H, W)
    assert video.size(0) == 1
    assert video.size(2) == 88
    assert video.size(3) == 88
    assert video.size(1) > 0

    assert sample["target_length"] > 0
    assert len(sample["target"]) == sample["target_length"]


def test_dataloader_collate_and_model_forward():
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*/*/mouth.mp4"))) == 0:
        pytest.skip("data/master dizininde test için video bulunamadı.")

    dataset = LipReadingDataset(
        master_dir=DATA_DIR,
        is_train=False,
        crop_size=88,
        max_frames=100,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    batch = next(iter(loader))
    videos = batch["videos"]
    targets = batch["targets"]
    input_lengths = batch["input_lengths"]
    target_lengths = batch["target_lengths"]

    assert videos.dim() == 5  # (B, 1, T, 88, 88)
    assert videos.size(0) == min(2, len(dataset))
    assert targets.dim() == 2  # (B, L)
    assert len(input_lengths) == videos.size(0)
    assert len(target_lengths) == videos.size(0)

    # Model ileri geçiş ve CTC loss testi
    model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=64, num_layers=1, encoder_type="bigru")
    model.eval()

    with torch.no_grad():
        logits = model(videos)
        loss = model.compute_loss(logits, targets, input_lengths, target_lengths)

    assert logits.shape == (videos.size(0), videos.size(2), VOCAB_SIZE)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_dataset_split_filtering():
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*/*/mouth.mp4"))) == 0:
        pytest.skip("data/master dizininde test için video bulunamadı.")

    # master dizinindeki mevcut video '-2uu72kg-9w' split_map'te 'val' splitindedir.
    val_dataset = LipReadingDataset(master_dir=DATA_DIR, split="val", is_train=False)
    train_dataset = LipReadingDataset(master_dir=DATA_DIR, split="train", is_train=False)

    assert len(val_dataset) > 0
    # Mevcut indirilen 5 segmentin hepsi val videosuna ait olduğundan train_dataset 0 olmalı
    assert len(train_dataset) == 0


def test_dataset_cache_in_ram():
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*/*/mouth.mp4"))) == 0:
        pytest.skip("data/master dizininde test için video bulunamadı.")

    dataset = LipReadingDataset(master_dir=DATA_DIR, is_train=False, cache_in_ram=True)
    assert dataset._cached_frames is not None
    assert len(dataset._cached_frames) == len(dataset)
    assert dataset._cached_frames[0].dtype == torch.uint8 or str(dataset._cached_frames[0].dtype) == "uint8"

    # item getirilince doğru tensör formatına çevrilmeli
    item = dataset[0]
    assert item["video"].dtype == torch.float32
    assert item["video"].size(0) == 1


def test_word_dataset_loading_and_slicing():
    from src.data.dataset import LipReadingWordDataset
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*/*/mouth.mp4"))) == 0:
        pytest.skip("data/master dizininde test için video bulunamadı.")

    word_ds = LipReadingWordDataset(master_dir=DATA_DIR, is_train=False, cache_in_ram=True)
    assert len(word_ds) > 0

    item = word_ds[0]
    assert "video" in item
    assert "target" in item
    assert "word" in item
    assert item["video"].dim() == 4  # (1, T, 88, 88)
    assert item["video"].size(0) == 1
    assert item["video"].size(2) == 88
    assert item["video"].size(3) == 88
    assert item["num_frames"] >= item["target_length"]  # CTC kuralı: T >= L


def test_word_dataset_max_per_word_capping():
    from src.data.dataset import LipReadingWordDataset
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*/*/mouth.mp4"))) == 0:
        pytest.skip("data/master dizininde test için video bulunamadı.")

    # max_per_word=1 ile her kelimeden en fazla 1 adet alınmalı
    word_ds_capped = LipReadingWordDataset(master_dir=DATA_DIR, is_train=False, max_per_word=1)
    words = [s["word"] for s in word_ds_capped.word_samples]
    assert len(words) == len(set(words)), "max_per_word=1 iken her kelime tekil olmalıdır!"


def test_phrase_dataset_loading():
    from src.data.dataset import LipReadingPhraseDataset
    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*/*/mouth.mp4"))) == 0:
        pytest.skip("data/master dizininde test için video bulunamadı.")

    phrase_ds = LipReadingPhraseDataset(master_dir=DATA_DIR, is_train=False, min_words=2, max_words=3)
    if len(phrase_ds) > 0:
        item = phrase_ds[0]
        assert "phrase" in item
        assert " " in item["phrase"], "Phrase en az bir boşluk içermelidir!"
        assert item["num_frames"] >= item["target_length"]
        assert item["video"].dim() == 4


def test_sequence_bucket_sampler_synthetic():
    from src.data.dataset import SequenceBucketSampler

    class MockDataset:
        def __init__(self, durations):
            self.samples = [{"duration": d} for d in durations]

        def __len__(self):
            return len(self.samples)

    # 10 short (2s), 10 medium (5s), 6 long (12s)
    durations = [2.0] * 10 + [5.0] * 10 + [12.0] * 6
    ds = MockDataset(durations)

    buckets = [
        (0.0, 3.5, 4),    # 10 samples -> 3 batches (4, 4, 2)
        (3.5, 8.0, 5),    # 10 samples -> 2 batches (5, 5)
        (8.0, 100.0, 2),  # 6 samples  -> 3 batches (2, 2, 2)
    ]
    sampler = SequenceBucketSampler(ds, buckets=buckets, shuffle=True, seed=42)

    assert len(sampler) == 8  # 3 + 2 + 3 = 8 batches

    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)
        # Check batch sizes
        assert len(batch) in [2, 4, 5]

    assert len(all_indices) == 26
    assert set(all_indices) == set(range(26)), "Tüm indeksler eksiksiz ve tekil olarak kapsanmalıdır!"


def test_dataset_long_clips_not_truncated():
    from src.data.dataset import LipReadingDataset
    import pathlib

    iborotti_dir = pathlib.Path(__file__).resolve().parent.parent / "data" / "iborotti"
    if not iborotti_dir.exists():
        pytest.skip("data/iborotti dizini bulunamadı.")

    ds = LipReadingDataset(master_dir=iborotti_dir, split="train", max_duration=None, cache_in_ram=False)
    long_samples = [s for s in ds.samples if s["duration"] > 10.0]
    if not long_samples:
        pytest.skip("10 saniyeden uzun örnek bulunamadı.")

    sample = long_samples[0]
    idx = ds.samples.index(sample)
    item = ds[idx]

    # max_frames 250'ye budanmamalı, videonun gerçek kare sayısını korumalı
    assert item["num_frames"] > 250, f"Uzun klip budanmış: {item['num_frames']} <= 250 (duration={sample['duration']})"
    assert item["target_length"] > 0




