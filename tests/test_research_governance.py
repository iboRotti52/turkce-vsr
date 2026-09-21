import copy
import hashlib
import json
import pathlib
import pytest
import yaml

from src.experiments.research_governance import (
    REQUIRED_READINESS_KEYS,
    ResearchStage,
    ReadinessReport,
    CandidateState,
    load_candidate,
    evaluate_readiness,
    build_full_train_manifest,
    build_full_train_manifest_from_path,
    write_full_train_manifest,
    require_full_train_authorized,
)


@pytest.fixture
def mock_candidate_dict(tmp_path):
    readiness = {
        key: {"passed": False, "evidence": []}
        for key in REQUIRED_READINESS_KEYS
    }
    return {
        "schema_version": 1,
        "candidate_version": "c0.4.0",
        "stage": "CONFIRMING",
        "status": "active",
        "components": {
            "preprocessing": {"choice": "96x96 grayscale 25fps", "status": "resolved"},
            "visual_frontend": {"choice": "3D-ResNet18", "status": "resolved"},
            "temporal_model": {"choice": "Conformer", "status": "resolved"},
            "objective_and_tokenizer": {"choice": "Char CTC", "status": "resolved"},
            "loss": {"choice": "CTC loss with blank penalty", "status": "resolved"},
            "curriculum_and_sampling": {"choice": "Three-stage curriculum", "status": "resolved"},
            "optimizer_and_scheduler": {"choice": "AdamW + CosineAnnealingLR", "status": "resolved"},
            "initializer": {"choice": "Auto-AVSR base", "status": "resolved"},
            "decoder": {"choice": "Calibrated Lexicon Beam", "status": "resolved"},
            "keyword_spotting": {"choice": "Posterior Spotter", "status": "resolved"},
        },
        "training": {
            "recipe_status": "in_progress",
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
        "open_questions": [
            {"id": "CONFIRM-002", "impact": "high", "status": "open"}
        ],
        "readiness": readiness,
    }


@pytest.fixture
def candidate_file(tmp_path, mock_candidate_dict):
    path = tmp_path / "research_candidate.yaml"
    path.write_text(yaml.safe_dump(mock_candidate_dict, sort_keys=False), encoding="utf-8")
    return path


def test_load_candidate(candidate_file):
    state = load_candidate(candidate_file)
    assert state.candidate_version == "c0.4.0"
    assert state.stage == ResearchStage.CONFIRMING
    assert len(state.open_questions) == 1


def test_open_high_impact_question_blocks_readiness(candidate_file):
    state = load_candidate(candidate_file)
    report = evaluate_readiness(state)
    assert not report.ready
    assert "open_high_impact_questions" in report.blockers
    assert "invalid_stage" in report.blockers


def test_active_high_impact_question_blocks_readiness(tmp_path, mock_candidate_dict):
    mock_candidate_dict["stage"] = "REHEARSAL"
    mock_candidate_dict["training"]["recipe_status"] = "frozen"
    mock_candidate_dict["open_questions"] = [
        {"id": "LOWDATA-001", "impact": "high", "status": "active"}
    ]
    for key in REQUIRED_READINESS_KEYS:
        mock_candidate_dict["readiness"][key] = {
            "passed": True,
            "evidence": [f"evidence/{key}"],
        }

    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(mock_candidate_dict, sort_keys=False), encoding="utf-8")

    report = evaluate_readiness(load_candidate(path))

    assert not report.ready
    assert "open_high_impact_questions" in report.blockers


def test_recipe_in_progress_blocks_readiness(tmp_path, mock_candidate_dict):
    mock_candidate_dict["stage"] = "REHEARSAL"
    mock_candidate_dict["open_questions"] = []
    mock_candidate_dict["training"]["recipe_status"] = "in_progress"
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(mock_candidate_dict, sort_keys=False), encoding="utf-8")

    state = load_candidate(path)
    report = evaluate_readiness(state)
    assert not report.ready
    assert "recipe_status_in_progress" in report.blockers


def test_unpassed_gate_blocks_manifest(tmp_path, mock_candidate_dict):
    mock_candidate_dict["stage"] = "READY_FOR_FULL_TRAIN"
    mock_candidate_dict["open_questions"] = []
    mock_candidate_dict["training"]["recipe_status"] = "frozen"
    # Keep readiness passed=False
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(mock_candidate_dict, sort_keys=False), encoding="utf-8")

    state = load_candidate(path)
    with pytest.raises(RuntimeError, match="readiness"):
        build_full_train_manifest(
            state,
            code_revision="abc1234567890",
            dataset_id="iboRotti/avsr-tr-dataset",
            dataset_sha256="datahash123",
            split_sha256="splithash123",
            initializer="probe_confirm001_best.pt",
            seeds=[42, 123, 456],
            budget_usd=22.0,
            dataset_scope_note="1325 clips <= 8.0s",
        )


def test_manifest_creation_and_tamper_detection(tmp_path, mock_candidate_dict):
    mock_candidate_dict["stage"] = "READY_FOR_FULL_TRAIN"
    mock_candidate_dict["open_questions"] = []
    mock_candidate_dict["training"]["recipe_status"] = "frozen"
    for k in REQUIRED_READINESS_KEYS:
        mock_candidate_dict["readiness"][k] = {"passed": True, "evidence": [f"evidence/{k}"]}

    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(mock_candidate_dict, sort_keys=False), encoding="utf-8")

    state = load_candidate(path)
    report = evaluate_readiness(state)
    assert report.ready, f"Blockers: {report.blockers}"

    manifest_path = tmp_path / "full_train_manifest.json"
    manifest = build_full_train_manifest(
        state,
        code_revision="abc1234567890",
        dataset_id="iboRotti/avsr-tr-dataset",
        dataset_sha256="datahash123",
        split_sha256="splithash123",
        initializer="probe_confirm001_best.pt",
        seeds=[42, 123, 456],
        budget_usd=22.0,
        dataset_scope_note="1325 clips <= 8.0s",
    )
    write_full_train_manifest(manifest_path, manifest)

    # Valid authorization check
    require_full_train_authorized(path, manifest_path)

    # Tampering candidate file must cause authorization failure
    path.write_text(path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="candidate.*hash"):
        require_full_train_authorized(path, manifest_path)
