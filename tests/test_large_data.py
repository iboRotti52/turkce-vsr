import json

import pytest

from src.data.hf_downloader import HFDatasetDownloader
from src.data.large_data import (
    build_large_data_plan,
    build_speaker_disjoint_split,
    build_speaker_diverse_training_stages,
    validate_speaker_disjoint_split,
)
from src.data.large_data_loader import (
    StageDatasetView,
    build_large_data_loader,
    set_large_data_loader_epoch,
)
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
    assert len(summary["1h"]["sample_ids_sha256"]) == 64
    assert len(summary["full"]["sample_ids_sha256"]) == 64


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
    assert payload["speaker_identity_field"] == "channel"
    assert payload["speaker_identity_is_proxy"] is True
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
    set_large_data_loader_epoch(loader, 3)
    assert loader.batch_sampler.epoch == 3

    val_loader = build_large_data_loader(
        Dataset(), train=False, num_workers=0, batch_size=3
    )
    assert val_loader.batch_size == 3

    with pytest.raises(ValueError, match="RAM cache"):
        build_large_data_loader(Dataset(cache=True), train=True, num_workers=0)


def test_duration_balancing_does_not_put_dominant_speaker_in_test():
    rows = []
    speaker_hours = {
        "dominant": 60.0,
        "speaker-b": 10.0,
        "speaker-c": 10.0,
        "speaker-d": 10.0,
        "speaker-e": 10.0,
    }
    for idx, (speaker, hours) in enumerate(speaker_hours.items()):
        rows.append(
            {
                "item_id": f"video-{idx}",
                "segment_id": "000000",
                "duration": str(hours * 3600.0),
                "channel": speaker,
                "text": "örnek türkçe cümle",
            }
        )

    split_map = build_speaker_disjoint_split(
        rows, seed=42, val_fraction=0.10, test_fraction=0.10
    )
    validate_speaker_disjoint_split(split_map)

    assert split_map["video-0"]["split"] == "train"
    summary = {
        split: sum(
            float(row["duration"])
            for row in rows
            if split_map[row["item_id"]]["split"] == split
        )
        / 3600.0
        for split in ("train", "val", "test")
    }
    assert summary["train"] == pytest.approx(80.0)
    assert summary["val"] == pytest.approx(10.0)
    assert summary["test"] == pytest.approx(10.0)


def test_pinned_dataset_requires_explicit_new_split_map(tmp_path):
    pinned = "a" * 40
    downloader = HFDatasetDownloader(
        target_dir=tmp_path,
        revision=pinned,
        require_pinned_revision=True,
    )
    assert downloader.manifest_dir == tmp_path / "_revisions" / pinned / "manifests"

    with pytest.raises(FileNotFoundError, match="split map açıkça verilmelidir"):
        downloader._get_split_map()

    split_path = tmp_path / "split_map_large_data.json"
    split_path.write_text(
        json.dumps({"video-a": {"split": "train", "speaker": "speaker-a"}}),
        encoding="utf-8",
    )
    explicit = HFDatasetDownloader(
        target_dir=tmp_path,
        revision=pinned,
        require_pinned_revision=True,
        split_map_path=split_path,
    )
    assert explicit._get_split_map()["video-a"]["split"] == "train"


def test_pinned_revisions_use_separate_local_caches(tmp_path):
    first = HFDatasetDownloader(
        target_dir=tmp_path,
        revision="a" * 40,
        require_pinned_revision=True,
    )
    second = HFDatasetDownloader(
        target_dir=tmp_path,
        revision="b" * 40,
        require_pinned_revision=True,
    )
    assert first.manifest_dir != second.manifest_dir
    assert first.clips_dir != second.clips_dir


def test_large_data_plan_rejects_mixed_speaker_identity_semantics():
    rows = [
        {
            "item_id": "video-a",
            "segment_id": "000001",
            "duration": "2.0",
            "speaker_id": "speaker-a",
            "text": "örnek cümle",
        },
        {
            "item_id": "video-b",
            "segment_id": "000001",
            "duration": "2.0",
            "channel": "channel-b",
            "text": "örnek cümle",
        },
        {
            "item_id": "video-c",
            "segment_id": "000001",
            "duration": "2.0",
            "channel": "channel-c",
            "text": "örnek cümle",
        },
    ]
    with pytest.raises(ValueError, match="ortak bir speaker identity"):
        build_large_data_plan(
            rows,
            dataset_id="avsr-tr-ekip/avsr-tr-dataset",
            dataset_revision="c" * 40,
        )


def test_large_data_plan_rejects_duplicate_sample_ids_and_missing_duration():
    rows = _rows()
    duplicate = list(rows) + [dict(rows[0])]
    with pytest.raises(ValueError, match="duplicate sample"):
        build_large_data_plan(
            duplicate,
            dataset_id="avsr-tr-ekip/avsr-tr-dataset",
            dataset_revision="d" * 40,
        )

    invalid = [dict(row) for row in rows]
    invalid[0]["duration"] = "0"
    with pytest.raises(ValueError, match="duration"):
        build_large_data_plan(
            invalid,
            dataset_id="avsr-tr-ekip/avsr-tr-dataset",
            dataset_revision="e" * 40,
        )


def test_stage_dataset_view_binds_exact_planned_samples():
    class BaseDataset:
        cache_in_ram = False

        def __init__(self):
            self.samples = [
                {"video_id": "video-a", "seg_id": "000001", "duration": 2.0},
                {"video_id": "video-b", "seg_id": "000001", "duration": 5.0},
                {"video_id": "video-c", "seg_id": "000001", "duration": 9.0},
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return {"idx": idx}

    view = StageDatasetView(
        BaseDataset(),
        ["video-c/000001", "video-a/000001"],
    )
    assert len(view) == 2
    assert [sample["video_id"] for sample in view.samples] == ["video-c", "video-a"]
    assert view[0]["idx"] == 2
    assert [sample["duration"] for sample in view.samples] == [9.0, 2.0]

    with pytest.raises(RuntimeError, match="bulunmayan"):
        StageDatasetView(BaseDataset(), ["missing/000001"])
