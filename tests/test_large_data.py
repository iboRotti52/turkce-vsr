import json

import pytest

from src.data.hf_downloader import HFDatasetDownloader
from src.data.large_data import (
    build_large_data_plan,
    build_speaker_disjoint_split,
    build_speaker_diverse_training_stages,
    validate_speaker_disjoint_split,
)
from src.data.large_data_loader import build_large_data_loader
from src.data.dataset import SequenceBucketSampler


def _rows():
    rows = []
    # 8 speakers, two source videos each, enough duration to exercise staging.
    for speaker_idx in range(8):
        speaker = f"speaker-{speaker_idx}"
        for video_idx in range(2):
            item_id = f"video-{speaker_idx}-{video_idx}"
            for segment_idx in range(4):
                rows.append(
                    {
                        "item_id": item_id,
                        "segment_id": f"{segment_idx:06d}",
                        "duration": "1800",  # 0.5h per segment
                        "channel": speaker,
                        "text": "örnek türkçe cümle",
                    }
                )
    return rows


def test_large_data_requires_immutable_hf_revision(tmp_path):
    with pytest.raises(ValueError, match="40-hex"):
        HFDatasetDownloader(
            target_dir=tmp_path,
            revision="main",
            require_pinned_revision=True,
        )

    pinned = "a" * 40
    downloader = HFDatasetDownloader(
        target_dir=tmp_path,
        revision=pinned,
        require_pinned_revision=True,
    )
    assert downloader.revision == pinned


def test_speaker_disjoint_split_is_deterministic_and_leak_free():
    rows = _rows()
    first = build_speaker_disjoint_split(rows, seed=17)
    second = build_speaker_disjoint_split(rows, seed=17)
    assert first == second
    validate_speaker_disjoint_split(first)

    speaker_splits = {}
    for info in first.values():
        speaker_splits.setdefault(info["speaker"], set()).add(info["split"])
    assert all(len(splits) == 1 for splits in speaker_splits.values())
    assert {info["split"] for info in first.values()} == {"train", "val", "test"}


def test_staged_subsets_are_nested_train_only_and_speaker_diverse():
    rows = _rows()
    split_map = build_speaker_disjoint_split(rows, seed=42)
    stages, summary = build_speaker_diverse_training_stages(
        rows,
        split_map,
        targets_hours=(1.0, 2.0, 4.0),
        seed=42,
    )

    assert "1h" in stages and "2h" in stages and "4h" in stages and "full" in stages
    assert set(stages["1h"]).issubset(stages["2h"])
    assert set(stages["2h"]).issubset(stages["4h"])
    assert set(stages["4h"]).issubset(stages["full"])

    item_split = {item: info["split"] for item, info in split_map.items()}
    for stage_ids in stages.values():
        for sample_id in stage_ids:
            item_id = sample_id.split("/", 1)[0]
            assert item_split[item_id] == "train"

    assert summary["1h"]["speakers"] >= 2


def test_large_data_plan_pins_revision_and_hashes_split():
    plan = build_large_data_plan(
        _rows(),
        dataset_id="avsr-tr-ekip/avsr-tr-dataset",
        dataset_revision="b" * 40,
        seed=7,
        targets_hours=(1.0, 2.0),
    )
    payload = plan.to_dict()

    assert payload["dataset_revision"] == "b" * 40
    assert len(payload["split_map_sha256"]) == 64
    assert len(payload["plan_sha256"]) == 64
    assert payload["split_summary"]["test"]["speakers"] >= 1


def test_large_data_loader_uses_duration_buckets_and_rejects_ram_cache():
    class Dataset:
        def __init__(self, cache=False):
            self.cache_in_ram = cache
            self.samples = [
                {"duration": 2.0},
                {"duration": 5.0},
                {"duration": 9.0},
                {"duration": 2.5},
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            raise AssertionError("Loader construction should not fetch samples")

    loader = build_large_data_loader(Dataset(), train=True, num_workers=0)
    assert isinstance(loader.batch_sampler, SequenceBucketSampler)

    val_loader = build_large_data_loader(
        Dataset(), train=False, num_workers=0, batch_size=3
    )
    assert val_loader.batch_size == 3

    with pytest.raises(ValueError, match="RAM cache"):
        build_large_data_loader(Dataset(cache=True), train=True, num_workers=0)
