import json
import pathlib

import pytest
import yaml

from src.data.large_data import build_large_data_plan, write_large_data_plan
from src.experiments.research_governance import compute_file_sha256
from src.experiments.research_ops import (
    build_execution_setup,
    complete_from_files,
    evaluate_research_session,
    register_from_spec,
)
from src.experiments.tracker import ExperimentRecord, ExperimentTracker


DATASET_ID = "avsr-tr-ekip/avsr-tr-dataset"


def _rows():
    rows = []
    for speaker_idx in range(6):
        for segment_idx in range(2):
            rows.append(
                {
                    "item_id": f"video-{speaker_idx}",
                    "segment_id": f"{segment_idx:06d}",
                    "duration": "3600",
                    "channel": f"speaker-{speaker_idx}",
                    "text": "örnek türkçe cümle",
                }
            )
    return rows


def _write_plan(tmp_path: pathlib.Path) -> pathlib.Path:
    plan = build_large_data_plan(
        _rows(),
        dataset_id=DATASET_ID,
        dataset_revision="1" * 40,
        seed=42,
        targets_hours=(1.0, 2.0),
    )
    path = tmp_path / "large_data_plan.json"
    write_large_data_plan(path, plan)
    return path


def _write_policy(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "large_data_research.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "phase": "large_data_research",
                "candidate_version": "c0.5.0",
                "data": {"source": DATASET_ID},
                "research": {"starting_prior": "c0.4.0"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_candidate(
    tmp_path: pathlib.Path,
    *,
    version: str = "c0.5.1",
    stage: str = "RESEARCHING",
    source: str = DATASET_ID,
    active_question: str | None = "ARCH-LD-001",
    research_training_authorized: bool = True,
) -> pathlib.Path:
    path = tmp_path / "candidate.yaml"
    open_questions = []
    if active_question:
        open_questions.append(
            {
                "id": active_question,
                "component": "end_to_end_architecture",
                "impact": "high",
                "status": "active",
                "question": "Which model mechanism best resolves the current bottleneck?",
            }
        )
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "candidate_version": version,
                "stage": stage,
                "data": {"source": source},
                "training": {
                    "research_training_authorized": research_training_authorized,
                    "recipe_status": "in_progress",
                },
                "active_question": active_question,
                "open_questions": open_questions,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_spec(tmp_path: pathlib.Path, *, version: str = "c0.5.1") -> pathlib.Path:
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "question_id": "ARCH-LD-001",
                "candidate_version": version,
                "hypothesis": "A different temporal mechanism may scale better.",
                "requested_scale": "1h",
                "minimum_sufficient_scale": "1h",
                "information_gain_rationale": (
                    "1h is the smallest stage with enough held-out behavior for this test."
                ),
                "why_smaller_scale_is_insufficient": (
                    "Smoke verifies execution but cannot estimate validation behavior."
                ),
                "promotion_rule": (
                    "Promote if validation WER improves >= 3% without subgroup regression."
                ),
                "falsification_criteria": "No WER gain or worse subgroup failures.",
                "expectation": "Lower held-out WER without long-clip regression.",
                "evidence_scope": "large_data_regime",
                "estimated_gpu_hours": 0.5,
                "estimated_cost_usd": 0.25,
                "setup": {"baseline_experiment_id": "baseline-c051-1h"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _ready_paths(tmp_path):
    return (
        _write_candidate(tmp_path),
        _write_plan(tmp_path),
        _write_policy(tmp_path),
    )


def test_session_allows_future_belief_versions_not_only_c050(tmp_path):
    candidate, plan, policy = _ready_paths(tmp_path)
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")

    report = evaluate_research_session(
        candidate_path=candidate,
        plan_path=plan,
        policy_path=policy,
        tracker=tracker,
    )

    assert report.ready_for_new_run is True
    assert report.candidate_version == "c0.5.1"
    assert report.active_question == "ARCH-LD-001"
    assert report.dataset_id == DATASET_ID
    assert report.available_scales == ("smoke", "1h", "2h", "full")
    assert any("speaker_identity_is_proxy" in warning for warning in report.warnings)


def test_frozen_small_data_prior_cannot_impersonate_large_data_candidate(tmp_path):
    candidate = _write_candidate(tmp_path, version="c0.4.0")
    plan = _write_plan(tmp_path)
    policy = _write_policy(tmp_path)

    report = evaluate_research_session(
        candidate_path=candidate,
        plan_path=plan,
        policy_path=policy,
        tracker=ExperimentTracker(tmp_path / "registry.jsonl"),
    )

    assert report.ready_for_new_run is False
    assert "large_data_candidate_not_opened:c0.4.0" in report.blockers


def test_session_blocks_wrong_stage_missing_question_and_dataset_mismatch(tmp_path):
    candidate = _write_candidate(
        tmp_path,
        stage="READY_FOR_FULL_TRAIN",
        source="wrong/dataset",
        active_question=None,
    )
    report = evaluate_research_session(
        candidate_path=candidate,
        plan_path=_write_plan(tmp_path),
        policy_path=_write_policy(tmp_path),
        tracker=ExperimentTracker(tmp_path / "registry.jsonl"),
    )

    assert report.ready_for_new_run is False
    assert "candidate_stage_not_researchable:READY_FOR_FULL_TRAIN" in report.blockers
    assert "active_question_missing" in report.blockers
    assert any(x.startswith("candidate_dataset_plan_mismatch:") for x in report.blockers)


def test_existing_in_progress_run_blocks_second_run(tmp_path):
    candidate, plan, policy = _ready_paths(tmp_path)
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    tracker.log(
        ExperimentRecord(
            experiment_id="already-running",
            hypothesis="h",
            falsification_criteria="f",
            setup={"question_id": "ARCH-LD-001"},
            expectation="e",
            candidate_version="c0.5.1",
            data_scale="1h",
            technical_status="IN_PROGRESS",
        )
    )

    report = evaluate_research_session(
        candidate_path=candidate,
        plan_path=plan,
        policy_path=policy,
        tracker=tracker,
    )

    assert report.ready_for_new_run is False
    assert report.in_progress_experiment_ids == ("already-running",)
    assert "active_experiment_exists:already-running" in report.blockers


def test_register_from_spec_seals_exact_execution_provenance(tmp_path):
    candidate, plan, policy = _ready_paths(tmp_path)
    spec = _write_spec(tmp_path)
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")

    record = register_from_spec(
        spec_path=spec,
        experiment_id="probe-arch-c051-1h",
        remaining_budget_usd=5.0,
        seed=7,
        initializer_id="auto-avsr:vsr_trlrs3_base",
        initializer_sha256="b" * 64,
        candidate_path=candidate,
        plan_path=plan,
        policy_path=policy,
        tracker=tracker,
        code_revision="a" * 40,
    )

    assert record.technical_status == "IN_PROGRESS"
    assert record.candidate_version == "c0.5.1"
    assert record.setup["code_revision"] == "a" * 40
    assert record.setup["candidate_recipe_sha256"] == compute_file_sha256(candidate)
    assert record.setup["seed"] == 7
    assert record.setup["initializer_id"] == "auto-avsr:vsr_trlrs3_base"
    assert record.setup["initializer_sha256"] == "b" * 64
    assert record.setup["baseline_experiment_id"] == "baseline-c051-1h"
    assert len(record.pre_result_contract_sha256) == 64

    blocked = evaluate_research_session(
        candidate_path=candidate,
        plan_path=plan,
        policy_path=policy,
        tracker=tracker,
    )
    assert blocked.ready_for_new_run is False
    assert "active_experiment_exists:probe-arch-c051-1h" in blocked.blockers


def test_register_rejects_spec_for_different_question_or_candidate(tmp_path):
    candidate, plan, policy = _ready_paths(tmp_path)
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")

    wrong_candidate = _write_spec(tmp_path, version="c0.6.0")
    with pytest.raises(RuntimeError, match="Spec candidate mismatch"):
        register_from_spec(
            spec_path=wrong_candidate,
            experiment_id="wrong-candidate",
            remaining_budget_usd=5.0,
            seed=1,
            initializer_id="init",
            initializer_sha256="b" * 64,
            candidate_path=candidate,
            plan_path=plan,
            policy_path=policy,
            tracker=tracker,
            code_revision="a" * 40,
        )

    payload = json.loads(wrong_candidate.read_text(encoding="utf-8"))
    payload["candidate_version"] = "c0.5.1"
    payload["question_id"] = "OTHER-001"
    wrong_candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Spec question mismatch"):
        register_from_spec(
            spec_path=wrong_candidate,
            experiment_id="wrong-question",
            remaining_budget_usd=5.0,
            seed=1,
            initializer_id="init",
            initializer_sha256="b" * 64,
            candidate_path=candidate,
            plan_path=plan,
            policy_path=policy,
            tracker=tracker,
            code_revision="a" * 40,
        )


def test_complete_wrapper_closes_registered_run(tmp_path):
    candidate, plan, policy = _ready_paths(tmp_path)
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    record = register_from_spec(
        spec_path=_write_spec(tmp_path),
        experiment_id="probe-complete-1h",
        remaining_budget_usd=5.0,
        seed=3,
        initializer_id="init",
        initializer_sha256="b" * 64,
        candidate_path=candidate,
        plan_path=plan,
        policy_path=policy,
        tracker=tracker,
        code_revision="a" * 40,
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({"wer": 0.31, "cer": 0.22}),
        encoding="utf-8",
    )

    completed = complete_from_files(
        experiment_id=record.experiment_id,
        result_path=result_path,
        verdict="INCONCLUSIVE",
        actual_gpu_hours=0.45,
        actual_cost_usd=0.2,
        updated_belief="The mechanism is plausible but not yet separated at 1h.",
        next_step="Inspect long-utterance failures before considering a scale increase.",
        evidence_refs=("artifacts/probe-complete-1h/metrics.json",),
        tracker=tracker,
    )

    assert completed.technical_status == "COMPLETED"
    assert completed.scientific_verdict == "INCONCLUSIVE"
    assert completed.actual_gpu_hours == pytest.approx(0.45)
    assert completed.cost_usd == pytest.approx(0.2)


def test_build_execution_setup_preserves_extra_fields_but_owns_provenance(tmp_path):
    candidate = _write_candidate(tmp_path)
    setup = build_execution_setup(
        candidate_path=candidate,
        code_revision="a" * 40,
        seed=11,
        initializer_id="init",
        initializer_sha256="b" * 64,
        extra_setup={
            "seed": 999,
            "code_revision": "bad",
            "baseline": "baseline-1",
        },
    )
    assert setup["seed"] == 11
    assert setup["code_revision"] == "a" * 40
    assert setup["candidate_recipe_sha256"] == compute_file_sha256(candidate)
    assert setup["baseline"] == "baseline-1"
