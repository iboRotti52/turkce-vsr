import json
import pathlib
import subprocess

import pytest
import yaml

from src.data.large_data import build_large_data_plan, write_large_data_plan
from src.experiments.large_data_workflow import (
    complete_from_spec,
    derive_execution_setup,
    promote_from_spec,
    reconcile_workflow,
    register_from_spec,
)
from src.experiments.tracker import ExperimentTracker


def _git(repo: pathlib.Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout.strip()


def _init_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")

    (repo / "configs").mkdir()
    (repo / "research").mkdir()
    (repo / "experiments").mkdir()
    (repo / "artifacts").mkdir()

    recipe = {
        "candidate_version": "c0.5.0",
        "recipe_status": "researching",
    }
    (repo / "configs" / "research_candidate.yaml").write_text(
        yaml.safe_dump(recipe, sort_keys=True),
        encoding="utf-8",
    )
    (repo / "initializer.pt").write_bytes(b"initializer-bytes")
    (repo / "README.md").write_text("test\n", encoding="utf-8")

    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "test fixture")
    return repo


def _rows():
    rows = []
    for speaker_idx in range(8):
        for video_idx in range(2):
            item_id = f"video-{speaker_idx}-{video_idx}"
            for segment_idx in range(4):
                rows.append(
                    {
                        "item_id": item_id,
                        "segment_id": f"{segment_idx:06d}",
                        "duration": "2700",
                        "speaker_id": f"speaker-{speaker_idx}",
                        "text": "örnek türkçe cümle",
                    }
                )
    return rows


def _write_plan(repo: pathlib.Path) -> pathlib.Path:
    plan = build_large_data_plan(
        _rows(),
        dataset_id="avsr-tr-ekip/avsr-tr-dataset",
        dataset_revision="a" * 40,
        seed=42,
        targets_hours=(10.0, 25.0),
    )
    path = repo / "research" / "large_data_plan.json"
    write_large_data_plan(path, plan)
    return path


def _register_spec():
    return {
        "experiment_id": "probe_arch_ld001_10h",
        "question_id": "ARCH-LD-001",
        "candidate_version": "c0.5.0",
        "hypothesis": "Alternative temporal encoder may scale better.",
        "requested_scale": "10h",
        "minimum_sufficient_scale": "10h",
        "falsification_criteria": "No WER gain or worse long-utterance failures.",
        "expectation": "Lower held-out WER without subgroup collapse.",
        "information_gain_rationale": "10h can distinguish generalization behavior.",
        "why_smaller_scale_is_insufficient": "Smoke cannot estimate held-out WER.",
        "promotion_rule": "Promote if WER improves >= 3% with no subgroup regression.",
        "evidence_scope": "large_data_regime",
        "estimated_gpu_hours": 1.0,
        "estimated_cost_usd": 0.8,
        "execution": {
            "seed": 42,
            "initializer_id": "auto-avsr-base",
            "initializer_path": "initializer.pt",
        },
        "setup": {"baseline_experiment_id": "baseline_c05_10h"},
    }


def test_execution_setup_derives_clean_git_recipe_and_initializer_hashes(tmp_path):
    repo = _init_repo(tmp_path)
    setup = derive_execution_setup(
        repo_root=repo,
        candidate_version="c0.5.0",
        seed=42,
        initializer_id="auto-avsr-base",
        initializer_path="initializer.pt",
    )

    assert setup["code_revision"] == _git(repo, "rev-parse", "HEAD")
    assert len(setup["candidate_recipe_sha256"]) == 64
    assert len(setup["initializer_sha256"]) == 64
    assert setup["seed"] == 42


def test_execution_setup_rejects_dirty_tree(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dirty working tree"):
        derive_execution_setup(
            repo_root=repo,
            candidate_version="c0.5.0",
            seed=42,
            initializer_id="auto-avsr-base",
            initializer_path="initializer.pt",
        )


def test_register_reserves_budget_and_enforces_single_active_run(tmp_path):
    repo = _init_repo(tmp_path)
    plan_path = _write_plan(repo)
    registry = repo / "experiments" / "registry.jsonl"

    # Plan is generated research metadata; commit it before scientific registration.
    _git(repo, "add", "research/large_data_plan.json")
    _git(repo, "commit", "-m", "add large data plan")

    record = register_from_spec(
        _register_spec(),
        repo_root=repo,
        plan_path=plan_path,
        registry_path=registry,
        total_budget_usd=25.0,
    )
    assert record.technical_status == "IN_PROGRESS"
    assert len(record.pre_result_contract_sha256) == 64

    status = reconcile_workflow(
        ExperimentTracker(registry),
        total_budget_usd=25.0,
    )
    assert status.active_experiment_id == record.experiment_id
    assert status.reserved_in_progress_usd == pytest.approx(0.8)
    assert status.remaining_unreserved_usd == pytest.approx(24.2)

    with pytest.raises(RuntimeError, match="aktif experiment"):
        register_from_spec(
            {**_register_spec(), "experiment_id": "second"},
            repo_root=repo,
            plan_path=plan_path,
            registry_path=registry,
            total_budget_usd=25.0,
        )


def test_register_complete_then_promote_tracks_budget_and_lineage(tmp_path):
    repo = _init_repo(tmp_path)
    plan_path = _write_plan(repo)
    registry = repo / "experiments" / "registry.jsonl"
    _git(repo, "add", "research/large_data_plan.json")
    _git(repo, "commit", "-m", "add large data plan")

    started = register_from_spec(
        _register_spec(),
        repo_root=repo,
        plan_path=plan_path,
        registry_path=registry,
        total_budget_usd=25.0,
    )

    completed = complete_from_spec(
        {
            "experiment_id": started.experiment_id,
            "result": {"wer": 0.28, "long_wer": 0.35},
            "scientific_verdict": "ACCEPT",
            "actual_gpu_hours": 0.9,
            "actual_cost_usd": 0.7,
            "surprise": "Long clips improved more than expected.",
            "updated_belief": "Alternative encoder is promising at 10h.",
            "next_step": "Confirm at 25h.",
            "evidence_refs": [
                "artifacts/probe_arch_ld001_10h/metrics.json",
                "artifacts/probe_arch_ld001_10h/failures.jsonl",
            ],
            "revalidation_trigger": "Reopen if the gain disappears at 25h.",
        },
        registry_path=registry,
        total_budget_usd=25.0,
    )
    assert completed.scientific_verdict == "ACCEPT"

    status = reconcile_workflow(
        ExperimentTracker(registry),
        total_budget_usd=25.0,
    )
    assert status.active_experiment_id is None
    assert status.spent_actual_usd == pytest.approx(0.7)
    assert status.remaining_unreserved_usd == pytest.approx(24.3)

    child = promote_from_spec(
        {
            "source_experiment_id": started.experiment_id,
            "target_experiment_id": "probe_arch_ld001_25h",
            "to_scale": "25h",
            "promotion_rule_met": True,
            "evidence_refs": ["artifacts/probe_arch_ld001_10h/metrics.json"],
            "estimated_gpu_hours": 2.0,
            "estimated_cost_usd": 1.5,
            # 25h is the last numeric stage in this synthetic plan, but "full"
            # still exists, so its next rule must be fixed before seeing 25h results.
            "target_promotion_rule": "Promote to full only if the WER gain persists.",
        },
        plan_path=plan_path,
        registry_path=registry,
        total_budget_usd=25.0,
    )
    assert child.scale_parent_experiment_id == started.experiment_id
    assert child.data_scale == "25h"

    after = reconcile_workflow(
        ExperimentTracker(registry),
        total_budget_usd=25.0,
    )
    assert after.active_experiment_id == child.experiment_id
    assert after.spent_actual_usd == pytest.approx(0.7)
    assert after.reserved_in_progress_usd == pytest.approx(1.5)
    assert after.remaining_unreserved_usd == pytest.approx(22.8)


def test_budget_counts_historical_actual_and_active_reservation(tmp_path):
    registry = tmp_path / "registry.jsonl"
    tracker = ExperimentTracker(registry)

    from src.experiments.tracker import ExperimentRecord

    tracker.log(
        ExperimentRecord(
            experiment_id="historical",
            hypothesis="h",
            falsification_criteria="f",
            setup={},
            expectation="e",
            status="PASSED",
            cost_usd=2.5,
            technical_status="COMPLETED",
        )
    )
    tracker.log(
        ExperimentRecord(
            experiment_id="active",
            hypothesis="h",
            falsification_criteria="f",
            setup={"question_id": "Q"},
            expectation="e",
            status="IN_PROGRESS",
            candidate_version="c0.5.0",
            data_scale="10h",
            cost_estimate_usd=1.25,
            technical_status="IN_PROGRESS",
            pre_result_contract_sha256="0" * 64,
        )
    )

    # Contract is intentionally invalid, so reconciliation must fail closed
    # rather than trusting the budget math on a tampered active record.
    with pytest.raises(RuntimeError, match="Pre-result experiment contract"):
        reconcile_workflow(tracker, total_budget_usd=25.0)


def test_legacy_actual_cost_is_not_forgotten(tmp_path):
    registry = tmp_path / "registry.jsonl"
    tracker = ExperimentTracker(registry)

    from src.experiments.tracker import ExperimentRecord

    tracker.log(
        ExperimentRecord(
            experiment_id="legacy_small_data",
            hypothesis="h",
            falsification_criteria="f",
            setup={},
            expectation="e",
            status="PASSED",
            cost_usd=2.73,
            # Historical rows may not have technical_status.
        )
    )

    status = reconcile_workflow(tracker, total_budget_usd=25.0)
    assert status.spent_actual_usd == pytest.approx(2.73)
    assert status.remaining_unreserved_usd == pytest.approx(22.27)
