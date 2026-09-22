import json
import pathlib

from src.experiments.evidence_synthesis import (
    load_registry,
    run_small_data_scaling_audit,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_meta001_committed_result_matches_registry():
    records = load_registry(ROOT / "experiments" / "registry.jsonl")
    derived = run_small_data_scaling_audit(records)
    committed = json.loads(
        (
            ROOT
            / "research"
            / "rounds"
            / "2026-09-22-meta-001"
            / "result.json"
        ).read_text(encoding="utf-8")
    )

    assert derived["scientific_verdict"] == "REJECT"
    assert derived["result"] == "optimization_recognition_decoupling"
    assert derived["primary_summary"] == committed["primary_summary"]
    assert (
        derived["full_validation_extension"]
        == committed["full_validation_extension"]
    )
    assert derived["updated_belief"] == committed["updated_belief"]


def test_meta001_quantifies_loss_cer_decoupling():
    audit = run_small_data_scaling_audit(
        load_registry(ROOT / "experiments" / "registry.jsonl")
    )
    primary = audit["primary_summary"]
    full = audit["full_validation_extension"]

    assert primary["train_samples_multiplier"] == 3.680556
    assert primary["best_val_loss_relative_change_pct"] == -3.81591
    assert primary["cer_percentage_point_change"] == 0.69
    assert primary["loss_improved"] is True
    assert primary["cer_improved"] is False

    assert full["best_val_loss_relative_change_pct"] == -1.63993
    assert full["cer_percentage_point_change"] == 0.07
    assert full["loss_improved"] is True
    assert full["cer_improved"] is False
