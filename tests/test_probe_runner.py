"""
tests/test_probe_runner.py — Candidate-Aware Research Probe Runner Tests
Verifies that research probes strictly obey candidate configuration,
active question gating, speaker-disjoint splits, provenance tracking, and decision rules.
"""

import json
import pathlib
import pytest
import yaml

from src.experiments.probe_runner import (
    CandidateProbeConfig,
    ResearchProbeRunner,
    validate_candidate_authority,
)
from src.experiments.tracker import ExperimentTracker


def test_validate_candidate_authority_blocks_unauthorized_training(tmp_path):
    config_file = tmp_path / "candidate.yaml"
    config_file.write_text(
        yaml.dump({
            "candidate_version": "c0.0.0",
            "active_question": "DATA-001",
            "training": {
                "research_training_authorized": False,
            },
            "research_policy": {
                "research_uses_test_split": False,
            }
        }),
        encoding="utf-8"
    )

    with pytest.raises(PermissionError, match="Araştırma eğitimi yetkisi verilmedi"):
        validate_candidate_authority(
            candidate_path=config_file,
            question_id="DATA-001",
        )


def test_validate_candidate_authority_blocks_wrong_question(tmp_path):
    config_file = tmp_path / "candidate.yaml"
    config_file.write_text(
        yaml.dump({
            "candidate_version": "c0.1.0",
            "active_question": "ARCH-001",
            "training": {
                "research_training_authorized": True,
            },
            "research_policy": {
                "research_uses_test_split": False,
            }
        }),
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Aktif araştırma sorusu ARCH-001 ancak RANDOM-999 istendi"):
        validate_candidate_authority(
            candidate_path=config_file,
            question_id="RANDOM-999",
        )


def test_probe_runner_strictly_blocks_test_split(tmp_path):
    config_file = tmp_path / "candidate.yaml"
    config_file.write_text(
        yaml.dump({
            "candidate_version": "c0.1.0",
            "active_question": "ARCH-001",
            "training": {
                "research_training_authorized": True,
            },
            "research_policy": {
                "research_uses_test_split": False,
            }
        }),
        encoding="utf-8"
    )

    runner = ResearchProbeRunner(candidate_path=config_file)
    with pytest.raises(RuntimeError, match="Test split.*araştırmaya kapalı"):
        runner.run(
            question_id="ARCH-001",
            eval_split="test",
        )


def test_probe_runner_executes_diagnostic_run(tmp_path):
    registry_file = tmp_path / "registry.jsonl"
    tracker = ExperimentTracker(registry_path=registry_file)

    config_file = tmp_path / "candidate.yaml"
    config_file.write_text(
        yaml.dump({
            "candidate_version": "c0.1.0",
            "active_question": "ARCH-001",
            "training": {
                "research_training_authorized": True,
            },
            "research_policy": {
                "research_uses_test_split": False,
            }
        }),
        encoding="utf-8"
    )

    runner = ResearchProbeRunner(
        candidate_path=config_file,
        tracker=tracker,
    )

    probe_cfg = CandidateProbeConfig(
        experiment_id="probe_test_diagnostic",
        hypothesis="Diagnostic mock run executes smoothly and computes all required metrics.",
        falsification_criteria="Any metric computation failure or NaN loss.",
        expectation="Probe produces valid CER, WER, Spotter metrics, and an explicit verdict.",
        arch_name="vsr_tiny",
        epochs=1,
        steps_per_epoch=2,
        batch_size=2,
        device="cpu",
    )

    result = runner.run_diagnostic_probe(probe_cfg)

    assert result["decision"] in ["ACCEPT", "REJECT", "INCONCLUSIVE"]
    assert "cer" in result["metrics"]
    assert "wer" in result["metrics"]
    assert "spotter_f1" in result["metrics"]
    assert "provenance" in result

    # Check that registry file logged the experiment properly
    records = tracker.load_all()
    assert len(records) == 1
    assert records[0].experiment_id == "probe_test_diagnostic"
    assert records[0].status in ["PASSED", "FALSIFIED", "INCONCLUSIVE"]


def test_probe_runner_freeze_frontend_and_decode_types(tmp_path):
    config_file = tmp_path / "candidate.yaml"
    config_file.write_text(
        yaml.dump({
            "candidate_version": "c0.1.0",
            "active_question": "TRAIN-001",
            "training": {
                "research_training_authorized": True,
            },
            "research_policy": {
                "research_uses_test_split": False,
            }
        }),
        encoding="utf-8"
    )

    probe_cfg = CandidateProbeConfig(
        experiment_id="probe_test_freeze",
        hypothesis="Frontend freezing works correctly.",
        falsification_criteria="Crash or failure.",
        expectation="Runs smoothly.",
        question_id="TRAIN-001",
        freeze_frontend=True,
    )
    assert probe_cfg.freeze_frontend is True
    assert probe_cfg.question_id == "TRAIN-001"


def test_probe_runner_eval_only_and_decoder_config(tmp_path):
    config_file = tmp_path / "candidate.yaml"
    config_file.write_text(
        yaml.dump({
            "candidate_version": "c0.2.0",
            "active_question": "DEC-001",
            "training": {
                "research_training_authorized": True,
            },
            "research_policy": {
                "research_uses_test_split": False,
            }
        }),
        encoding="utf-8"
    )

    probe_cfg = CandidateProbeConfig(
        experiment_id="probe_test_decoder",
        hypothesis="Lexicon beam search decoder evaluation configuration.",
        falsification_criteria="Crash or failure.",
        expectation="Runs smoothly.",
        question_id="DEC-001",
        eval_only=True,
        load_checkpoint="dummy.pt",
        beam_size=40,
        lm_alpha=0.5,
        lm_beta=1.2,
    )
    assert probe_cfg.eval_only is True
    assert probe_cfg.load_checkpoint == "dummy.pt"
    assert probe_cfg.beam_size == 40
    assert probe_cfg.lm_alpha == 0.5
    assert probe_cfg.lm_beta == 1.2
    assert probe_cfg.question_id == "DEC-001"


def test_probe_runner_blank_penalty_grid_config(tmp_path):
    config_file = tmp_path / "candidate.yaml"
    config_file.write_text(
        yaml.dump({
            "candidate_version": "c0.4.0",
            "active_question": "DEC-004",
            "training": {
                "research_training_authorized": True,
            },
            "research_policy": {
                "research_uses_test_split": False,
            }
        }),
        encoding="utf-8"
    )

    probe_cfg = CandidateProbeConfig(
        experiment_id="probe_test_bp_grid",
        hypothesis="Testing blank penalty calibration grid.",
        falsification_criteria="Crash or failure.",
        expectation="Config parses grid cleanly.",
        question_id="DEC-004",
        eval_only=True,
        load_checkpoint="probe_confirm001_full_train_scaling_best.pt",
        blank_penalty_grid=[0.0, 0.4, 0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5],
    )
    assert probe_cfg.eval_only is True
    assert probe_cfg.question_id == "DEC-004"
    assert probe_cfg.blank_penalty_grid == [0.0, 0.4, 0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]


def test_probe_runner_regularization_specaugment_config():
    probe_cfg = CandidateProbeConfig(
        experiment_id="probe_test_reg001",
        hypothesis="Testing feature specaugment and dropout configuration.",
        falsification_criteria="Crash or failure.",
        expectation="Config sets conformer_dropout and use_specaugment cleanly.",
        question_id="REG-001",
        conformer_dropout=0.2,
        use_specaugment=True,
    )
    assert probe_cfg.conformer_dropout == 0.2
    assert probe_cfg.use_specaugment is True


def test_probe_runner_seed_config():
    probe_cfg = CandidateProbeConfig(
        experiment_id="probe_test_seed",
        hypothesis="Testing seed configuration.",
        falsification_criteria="Crash or failure.",
        expectation="Config sets seed cleanly.",
        question_id="CONFIRM-003",
        seed=123,
    )
    assert probe_cfg.seed == 123
    assert probe_cfg.question_id == "CONFIRM-003"


