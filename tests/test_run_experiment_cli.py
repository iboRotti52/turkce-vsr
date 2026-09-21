import json
import pathlib
import pytest
import yaml

from run_experiment import (
    build_parser,
    handle_research_status,
    handle_seal_full_train,
    handle_authorize_full_train,
)
from src.experiments.research_governance import (
    REQUIRED_READINESS_KEYS,
    ResearchStage,
    load_candidate,
)


@pytest.fixture
def mock_candidate_yaml(tmp_path):
    (tmp_path / "research").mkdir(parents=True, exist_ok=True)
    (tmp_path / "research" / "DECISIONS.md").write_text("# Decisions\n", encoding="utf-8")
    readiness = {
        key: {"passed": True, "evidence": ["research/DECISIONS.md"]}
        for key in REQUIRED_READINESS_KEYS
    }
    data = {
        "schema_version": 1,
        "candidate_version": "c0.4.0",
        "stage": "REHEARSAL",
        "status": "active",
        "components": {
            "preprocessing": {"choice": "96x96 grayscale", "status": "resolved"},
            "visual_frontend": {"choice": "3D-ResNet18", "status": "resolved"},
            "temporal_model": {"choice": "Conformer", "status": "resolved"},
            "objective_and_tokenizer": {"choice": "Char CTC", "status": "resolved"},
            "loss": {"choice": "CTC loss", "status": "resolved"},
            "curriculum_and_sampling": {"choice": "Three-stage curriculum", "status": "resolved"},
            "optimizer_and_scheduler": {"choice": "AdamW", "status": "resolved"},
            "initializer": {"choice": "Auto-AVSR base", "status": "resolved"},
            "decoder": {"choice": "Calibrated Lexicon Beam", "status": "resolved"},
            "keyword_spotting": {"choice": "Posterior Spotter", "status": "resolved"},
        },
        "training": {
            "recipe_status": "frozen",
            "full_training_authorized": False,
            "research_training_authorized": True,
            "hyperparameters": {
                "epochs": 4,
                "batch_size": 8,
                "lr_temporal": 1.5e-4,
                "lr_frontend": 7.5e-6,
                "blank_penalty": 0.8,
                "max_duration": 8.0,
            },
        },
        "open_questions": [],
        "readiness": readiness,
    }
    c_path = tmp_path / "candidate.yaml"
    c_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return c_path


def test_cli_parser_recognizes_governance_modes():
    parser = build_parser()
    for mode in ["research-status", "seal-full-train", "authorize-full-train", "full-train"]:
        args = parser.parse_args(["--mode", mode])
        assert args.mode == mode


def test_research_status_reports_candidate_state(mock_candidate_yaml, capsys):
    ret = handle_research_status(candidate_path=mock_candidate_yaml)
    captured = capsys.readouterr()
    assert "c0.4.0" in captured.out
    assert "REHEARSAL" in captured.out
    assert ret == 0


def test_research_status_lists_active_high_impact_question(mock_candidate_yaml, capsys):
    raw = yaml.safe_load(mock_candidate_yaml.read_text(encoding="utf-8"))
    raw["open_questions"] = [
        {"id": "LOWDATA-001", "impact": "high", "status": "active"}
    ]
    mock_candidate_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    ret = handle_research_status(candidate_path=mock_candidate_yaml)
    captured = capsys.readouterr()

    assert ret == 1
    assert "LOWDATA-001" in captured.out


def test_seal_full_train_fails_closed_when_blockers_exist(tmp_path, mock_candidate_yaml):
    # Add an open high-impact question to cause a blocker
    raw = yaml.safe_load(mock_candidate_yaml.read_text(encoding="utf-8"))
    raw["open_questions"] = [{"id": "BLOCKER-001", "impact": "high", "status": "open"}]
    mock_candidate_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    manifest_path = tmp_path / "full_train_manifest.json"
    with pytest.raises(RuntimeError, match="Cannot seal full training"):
        handle_seal_full_train(
            candidate_path=mock_candidate_yaml,
            manifest_path=manifest_path,
            code_revision="abc1234567890",
            dataset_id="iboRotti/avsr-tr-dataset",
            dataset_sha256="datahash",
            split_sha256="splithash",
            initializer="probe_confirm001_best.pt",
            seeds=[42, 123, 456],
            budget_usd=22.0,
            dataset_scope_note="1325 clips <= 8.0s",
        )
    assert not manifest_path.exists()


def test_seal_full_train_transitions_stage_and_writes_manifest(tmp_path, mock_candidate_yaml):
    manifest_path = tmp_path / "full_train_manifest.json"
    handle_seal_full_train(
        candidate_path=mock_candidate_yaml,
        manifest_path=manifest_path,
        code_revision="abc1234567890",
        dataset_id="iboRotti/avsr-tr-dataset",
        dataset_sha256="datahash",
        split_sha256="splithash",
        initializer="probe_confirm001_best.pt",
        seeds=[42, 123, 456],
        budget_usd=22.0,
        dataset_scope_note="1325 clips <= 8.0s",
    )

    assert manifest_path.exists()
    state = load_candidate(mock_candidate_yaml)
    assert state.stage == ResearchStage.READY_FOR_FULL_TRAIN

    # Test authorize succeeds on sealed candidate
    manifest_data = handle_authorize_full_train(
        candidate_path=mock_candidate_yaml,
        manifest_path=manifest_path,
    )
    assert manifest_data["candidate_version"] == "c0.4.0"
    assert manifest_data["seeds"] == [42, 123, 456]
