import json

import pytest

from src.experiments.guardrails import (
    build_checkpoint_provenance,
    collect_sample_ids,
    require_checkpoint_exists,
    require_requested_sample_count,
)


def test_require_requested_sample_count_rejects_silent_shortfall():
    with pytest.raises(RuntimeError, match="train.*1000.*80"):
        require_requested_sample_count(split="train", requested=1000, actual=80)


def test_checkpoint_provenance_records_data_and_initializer_lineage(tmp_path):
    split_map = tmp_path / "split_map.json"
    split_map.write_text(
        json.dumps(
            {
                "speaker-a": {"split": "train"},
                "speaker-b": {"split": "val"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    provenance = build_checkpoint_provenance(
        dataset_id="iboRotti/avsr-tr-dataset",
        split_map_path=split_map,
        train_sample_ids=["speaker-a/000001", "speaker-a/000002"],
        val_sample_ids=["speaker-b/000001"],
        seed=42,
        initializer="auto-avsr:vsr_trlrs3_base.pth",
        code_revision="abc123",
    )

    assert provenance["dataset_id"] == "iboRotti/avsr-tr-dataset"
    assert provenance["initializer"] == "auto-avsr:vsr_trlrs3_base.pth"
    assert provenance["seed"] == 42
    assert provenance["code_revision"] == "abc123"
    assert provenance["train_sample_count"] == 2
    assert provenance["val_sample_count"] == 1
    assert provenance["train_sample_ids"] == ["speaker-a/000001", "speaker-a/000002"]
    assert provenance["split_map_sha256"]
    assert provenance["train_sample_ids_sha256"]


def test_collect_sample_ids_uses_video_and_segment_identity():
    class SentenceDataset:
        samples = [
            {"video_id": "speaker-a", "seg_id": "000002"},
            {"video_id": "speaker-a", "seg_id": "000001"},
        ]

    assert collect_sample_ids(SentenceDataset()) == [
        "speaker-a/000001",
        "speaker-a/000002",
    ]


def test_require_checkpoint_exists_rejects_missing_initializer(tmp_path):
    missing = tmp_path / "legacy_init.pt"

    with pytest.raises(FileNotFoundError, match="Initializer checkpoint bulunamadı"):
        require_checkpoint_exists(missing)
