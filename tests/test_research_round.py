import pathlib

import pytest
import yaml

from src.experiments.research_round import (
    validate_all_rounds,
    validate_belief_ledger,
)
from src.experiments.tracker import ExperimentTracker


ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_repository_rounds_and_beliefs_are_consistent():
    rounds = validate_all_rounds(
        repo_root=ROOT,
        rounds_root=ROOT / "research" / "rounds",
        registry_path=ROOT / "experiments" / "registry.jsonl",
    )
    assert {item["round_id"] for item in rounds} == {
        "RND-2026-09-22-META-001"
    }

    ledger = validate_belief_ledger(
        ROOT / "research" / "BELIEFS.yaml",
        valid_round_ids={item["round_id"] for item in rounds},
    )
    assert any(
        belief["id"] == "B-SMALL-001" for belief in ledger["beliefs"]
    )


def test_belief_ledger_rejects_unknown_round(tmp_path):
    path = tmp_path / "BELIEFS.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "beliefs": [
                    {
                        "id": "B-X",
                        "status": "active",
                        "statement": "x",
                        "evidence_scope": "small_data_regime",
                        "evidence": {"round_ids": ["UNKNOWN"]},
                        "revalidation_trigger": "new data",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="bilinmeyen research round"):
        validate_belief_ledger(path, valid_round_ids=set())


def test_round_validation_fails_when_registry_evidence_is_missing(tmp_path):
    source = ROOT / "research" / "rounds" / "2026-09-22-meta-001" / "round.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["evidence_inputs"]["experiment_ids"].append("does-not-exist")

    round_dir = tmp_path / "research" / "rounds" / "bad"
    round_dir.mkdir(parents=True)
    (round_dir / "round.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    result_src = (
        ROOT / "research" / "rounds" / "2026-09-22-meta-001" / "result.json"
    )
    result_dst = tmp_path / "research" / "rounds" / "2026-09-22-meta-001" / "result.json"
    result_dst.parent.mkdir(parents=True, exist_ok=True)
    result_dst.write_text(result_src.read_text(encoding="utf-8"), encoding="utf-8")

    tracker = ExperimentTracker(ROOT / "experiments" / "registry.jsonl")
    records = tracker.load_all(strict=True)

    from src.experiments.research_round import validate_round_manifest

    with pytest.raises(RuntimeError, match="olmayan experiment refs"):
        validate_round_manifest(
            round_dir / "round.yaml",
            repo_root=tmp_path,
            records=records,
        )
