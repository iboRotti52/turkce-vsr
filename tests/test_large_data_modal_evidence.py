import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
REVISION = "7ff10fb7cf98a6b6f98abc0cd790dd6cc191226a"


def test_data_package_keeps_heavy_dataset_import_lazy():
    code = (
        "import sys; import src.data; "
        "assert 'src.data.dataset' not in sys.modules, "
        "'src.data import eagerly loaded NumPy/Torch dataset module'"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
    )


def test_current_snapshot_audit_remains_scientifically_blocked():
    audit = json.loads(
        (
            ROOT
            / "research"
            / "snapshot_audits"
            / REVISION
            / "audit.json"
        ).read_text(encoding="utf-8")
    )

    assert audit["scientific_c0_5_ready"] is False
    assert audit["speaker_identity_field"] == "channel"
    assert audit["speaker_identity_is_proxy"] is True
    assert audit["available_stages"] == ["full"]
    assert set(audit["blockers"]) == {
        "proxy_speaker_identity_requires_audit",
        "10h_minimum_sufficient_stage_unavailable",
    }
    assert audit["test_split_used"] is False
    assert audit["dataset_revision"] == REVISION


def test_modal_rehearsal_is_technical_only_and_never_used_for_model_selection():
    result = json.loads(
        (ROOT / "artifacts" / "operations" / "OPS-LD-001.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["operation_id"] == "OPS-LD-001"
    assert result["scientific_use_for_model_selection"] is False
    assert result["scientific_verdict"] is None
    assert result["test_split_used"] is False
    assert result["staging"]["test_downloaded"] == 0

    training = result["training"]
    assert training["gpu_name"] == "NVIDIA A10"
    assert training["max_steps"] == 20
    assert training["train_samples"] == 64
    assert training["val_samples"] == 16
    assert training["initial_train_loss"] > training["final_train_loss"]
    assert training["cer"] == 1.0
    assert training["wer"] == 1.0
    assert len(training["raw_predictions"]) == 16
    assert all(item["hyp"] == "" for item in training["raw_predictions"])


def test_blocked_snapshot_is_not_installed_as_canonical_large_data_plan():
    assert not (ROOT / "research" / "large_data_plan.json").exists()
