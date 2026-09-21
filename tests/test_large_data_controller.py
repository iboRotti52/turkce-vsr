import pytest

from src.experiments.large_data_controller import (
    EvidenceScope,
    PromotionRequest,
    ScaleAction,
    ScaleExperimentPlan,
    ScientificVerdict,
    build_scaling_curve,
    complete_scale_experiment,
    fail_scale_experiment,
    ordered_research_scales,
    register_promoted_experiment,
    register_scale_experiment,
    seal_pre_result_contract,
    validate_promotion_request,
    validate_scale_experiment_plan,
)
from src.experiments.tracker import ExperimentRecord, ExperimentTracker


def _large_data_plan():
    return {
        "dataset_revision": "a" * 40,
        "split_map_sha256": "b" * 64,
        "plan_sha256": "c" * 64,
        "staged_subsets": {
            "10h": ["a"],
            "25h": ["a", "b"],
            "50h": ["a", "b", "c"],
            "full": ["a", "b", "c", "d"],
        },
        "staged_summary": {
            "10h": {"sample_ids_sha256": "1" * 64},
            "25h": {"sample_ids_sha256": "2" * 64},
            "50h": {"sample_ids_sha256": "3" * 64},
            "full": {"sample_ids_sha256": "4" * 64},
        },
    }


def _execution_setup(**extra):
    setup = {
        "code_revision": "d" * 40,
        "candidate_recipe_sha256": "e" * 64,
        "initializer_id": "auto-avsr:vsr_trlrs3_base",
        "initializer_sha256": "f" * 64,
        "seed": 42,
    }
    setup.update(extra)
    return setup


def _source_record(
    *,
    scale="10h",
    promotion_rule="Promote if WER improves >= 3% relative with no subgroup collapse.",
    scientific_verdict=ScientificVerdict.ACCEPT,
):
    return ExperimentRecord(
        experiment_id="probe_arch_newfamily_10h",
        hypothesis="A different temporal model may scale better than the current Conformer.",
        falsification_criteria="No WER gain or worse long-utterance failures.",
        setup=_execution_setup(
            question_id="ARCH-LD-001",
            dataset_revision="a" * 40,
            split_map_sha256="b" * 64,
            train_subset_sha256="1" * 64,
            large_data_plan_sha256="c" * 64,
        ),
        expectation="Better generalization on held-out validation.",
        result={"wer": 0.31},
        status={
            ScientificVerdict.ACCEPT: "PASSED",
            ScientificVerdict.REJECT: "FALSIFIED",
            ScientificVerdict.INCONCLUSIVE: "INCONCLUSIVE",
        }[scientific_verdict],
        candidate_version="c0.5.0",
        data_scale=scale,
        minimum_sufficient_scale="10h",
        evidence_scope=EvidenceScope.LARGE_DATA_REGIME.value,
        promotion_rule=promotion_rule,
        estimated_gpu_hours=1.2,
        cost_estimate_usd=0.8,
        technical_status="COMPLETED",
        scientific_verdict=scientific_verdict.value,
    )


def _persist_source(tmp_path, source, name="registry.jsonl"):
    tracker = ExperimentTracker(tmp_path / name)
    tracker.log(seal_pre_result_contract(source))
    return tracker


def test_scales_follow_available_plan():
    assert ordered_research_scales(_large_data_plan()) == (
        "smoke",
        "10h",
        "25h",
        "50h",
        "full",
    )


def test_controller_constrains_cost_not_scientific_search_space():
    # Intentionally radical scientific change: controller must not whitelist the
    # current Conformer/CTC family. It validates only scale/cost/falsifiability metadata.
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        hypothesis=(
            "Replace the current temporal encoder/objective with a state-space + "
            "transducer design because long-sequence failures may be temporal."
        ),
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="Predeclared falsification condition.",
        expectation="Predeclared expected observable change.",
        information_gain_rationale=(
            "10h contains enough speaker diversity and long clips to distinguish "
            "temporal behavior before spending on 25h+."
        ),
        why_smaller_scale_is_insufficient=(
            "Smoke can verify tensor/gradient correctness but cannot measure "
            "speaker-disjoint long-sequence generalization."
        ),
        promotion_rule="Promote only if WER improves >= 3% relative without subgroup regression.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.7,
    )
    validate_scale_experiment_plan(
        plan,
        large_data_plan=_large_data_plan(),
        remaining_budget_usd=10.0,
    )


def test_initial_scale_skip_requires_scientific_justification():
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        hypothesis="Test a temporal architecture change.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="Predeclared falsification condition.",
        expectation="Predeclared expected observable change.",
        information_gain_rationale="Need held-out behavior.",
        promotion_rule="Promote on meaningful WER gain.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    with pytest.raises(ValueError, match="daha küçük scale"):
        validate_scale_experiment_plan(
            plan,
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=10.0,
        )


def test_non_smoke_run_requires_predeclared_promotion_rule():
    plan = ScaleExperimentPlan(
        question_id="TRAIN-LD-001",
        candidate_version="c0.5.0",
        hypothesis="A schedule change may improve optimization.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="Predeclared falsification condition.",
        expectation="Predeclared expected observable change.",
        information_gain_rationale="10h is enough to see stable validation dynamics.",
        why_smaller_scale_is_insufficient="Smoke cannot estimate validation dynamics.",
        estimated_gpu_hours=0.5,
        estimated_cost_usd=0.2,
    )
    with pytest.raises(ValueError, match="promotion_rule"):
        validate_scale_experiment_plan(
            plan,
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=10.0,
        )


def test_accept_can_promote_only_with_predeclared_rule_met(tmp_path):
    rule = "Promote if WER improves >= 3% relative with no subgroup collapse."
    request = PromotionRequest(
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.ACCEPT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=rule,
        promotion_rule_met=True,
        evidence_refs=("experiments/registry.jsonl#probe_arch_newfamily_10h",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.4,
    )
    source = _source_record(promotion_rule=rule)
    tracker = _persist_source(tmp_path, source, "accept.jsonl")
    validate_promotion_request(
        request,
        tracker=tracker,
        source_experiment_id=source.experiment_id,
        large_data_plan=_large_data_plan(),
        remaining_budget_usd=8.0,
    )


def test_promotion_rule_cannot_be_changed_after_result(tmp_path):
    source = _source_record(promotion_rule="Original predeclared rule.")
    request = PromotionRequest(
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.ACCEPT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule="Easier rule invented after seeing result.",
        promotion_rule_met=True,
        evidence_refs=("registry#source",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.0,
    )
    tracker = _persist_source(tmp_path, source, "rule_change.jsonl")
    with pytest.raises(RuntimeError, match="değiştirilemez"):
        validate_promotion_request(
            request,
            tracker=tracker,
            source_experiment_id=source.experiment_id,
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=8.0,
        )


def test_inconclusive_is_not_automatic_promotion(tmp_path):
    source = _source_record(scientific_verdict=ScientificVerdict.INCONCLUSIVE)
    tracker = _persist_source(tmp_path, source, "inconclusive.jsonl")
    base = dict(
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.INCONCLUSIVE,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=source.promotion_rule,
        promotion_rule_met=False,
        evidence_refs=("registry#source",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.0,
    )
    with pytest.raises(RuntimeError, match="otomatik scale promotion"):
        validate_promotion_request(
            PromotionRequest(**base),
            tracker=tracker,
            source_experiment_id=source.experiment_id,
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=8.0,
        )

    validate_promotion_request(
        PromotionRequest(
            **base,
            scale_sensitive_ambiguity=True,
            why_larger_scale_resolves_ambiguity=(
                "The architecture difference is expected only after enough speaker "
                "and sequence diversity; 10h confidence intervals overlap."
            ),
        ),
        tracker=tracker,
        source_experiment_id=source.experiment_id,
        large_data_plan=_large_data_plan(),
        remaining_budget_usd=8.0,
    )


def test_rejected_hypothesis_cannot_be_promoted(tmp_path):
    source = _source_record(scientific_verdict=ScientificVerdict.REJECT)
    tracker = _persist_source(tmp_path, source, "reject.jsonl")
    request = PromotionRequest(
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.REJECT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=source.promotion_rule,
        promotion_rule_met=False,
        evidence_refs=("registry#source",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.0,
    )
    with pytest.raises(RuntimeError, match="REJECT"):
        validate_promotion_request(
            request,
            tracker=tracker,
            source_experiment_id=source.experiment_id,
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=8.0,
        )


def test_scale_skipping_requires_explicit_justification(tmp_path):
    source = _source_record()
    tracker = _persist_source(tmp_path, source, "skip.jsonl")
    request = PromotionRequest(
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="50h",
        scientific_verdict=ScientificVerdict.ACCEPT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=source.promotion_rule,
        promotion_rule_met=True,
        evidence_refs=("registry#source",),
        estimated_gpu_hours=4.0,
        estimated_cost_usd=2.0,
    )
    with pytest.raises(ValueError, match="skip_scale_justification"):
        validate_promotion_request(
            request,
            tracker=tracker,
            source_experiment_id=source.experiment_id,
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=8.0,
        )


def test_scaling_curve_is_candidate_question_and_scale_scoped():
    records = [
        ExperimentRecord(
            experiment_id="e25",
            hypothesis="h",
            falsification_criteria="f",
            setup={"question_id": "ARCH-LD-001", "dataset_revision": "a" * 40},
            expectation="x",
            result={"wer": 0.25},
            status="PASSED",
            candidate_version="c0.5.0",
            data_scale="25h",
            evidence_scope=EvidenceScope.LARGE_DATA_REGIME.value,
            technical_status="COMPLETED",
            scientific_verdict=ScientificVerdict.ACCEPT.value,
        ),
        ExperimentRecord(
            experiment_id="e10",
            hypothesis="h",
            falsification_criteria="f",
            setup={"question_id": "ARCH-LD-001", "dataset_revision": "a" * 40},
            expectation="x",
            result={"metrics": {"wer": 0.32}},
            status="PASSED",
            candidate_version="c0.5.0",
            data_scale="10h",
            evidence_scope=EvidenceScope.LARGE_DATA_REGIME.value,
            technical_status="COMPLETED",
            scientific_verdict=ScientificVerdict.ACCEPT.value,
        ),
        ExperimentRecord(
            experiment_id="wrong_candidate",
            hypothesis="h",
            falsification_criteria="f",
            setup={"question_id": "ARCH-LD-001", "dataset_revision": "a" * 40},
            expectation="x",
            result={"wer": 0.10},
            status="PASSED",
            candidate_version="c0.6.0",
            data_scale="50h",
        ),
    ]
    curve = build_scaling_curve(
        records,
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        dataset_revision="a" * 40,
        metric="wer",
        large_data_plan=_large_data_plan(),
    )
    assert [(point.scale, point.value) for point in curve] == [
        ("10h", 0.32),
        ("25h", 0.25),
    ]


def test_tracker_roundtrips_large_data_agentic_metadata(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    record = seal_pre_result_contract(_source_record())
    tracker.log(record)

    loaded = tracker.load_all()
    assert len(loaded) == 1
    assert loaded[0].candidate_version == "c0.5.0"
    assert loaded[0].data_scale == "10h"
    assert loaded[0].minimum_sufficient_scale == "10h"
    assert loaded[0].promotion_rule == record.promotion_rule
    assert loaded[0].estimated_gpu_hours == pytest.approx(1.2)
    assert len(loaded[0].pre_result_contract_sha256) == 64


def test_register_scale_experiment_writes_pre_result_governance(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-002",
        candidate_version="c0.5.0",
        hypothesis="A different visual frontend may reduce cross-speaker failures.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No validation or subgroup improvement versus control.",
        expectation="Lower WER on held-out proxy groups without long-clip regression.",
        information_gain_rationale="10h is enough to expose cross-speaker behavior.",
        why_smaller_scale_is_insufficient="Smoke cannot measure held-out generalization.",
        promotion_rule="Promote if WER improves >= 2% and worst-group WER does not regress.",
        evidence_scope=EvidenceScope.LARGE_DATA_REGIME.value,
        estimated_gpu_hours=1.5,
        estimated_cost_usd=0.9,
    )
    large_plan = _large_data_plan()

    record = register_scale_experiment(
        plan,
        tracker=tracker,
        experiment_id="probe_arch_ld002_10h",
        large_data_plan=large_plan,
        remaining_budget_usd=5.0,
        setup=_execution_setup(baseline_experiment_id="baseline_c05_10h"),
    )

    assert record.status == "IN_PROGRESS"
    assert record.data_scale == "10h"
    assert record.promotion_rule == plan.promotion_rule
    assert record.setup["train_subset_sha256"] == "1" * 64
    assert record.setup["dataset_revision"] == "a" * 40
    loaded = tracker.load_all()[0]
    assert loaded.promotion_rule == plan.promotion_rule
    assert loaded.status == "IN_PROGRESS"


def test_completion_separates_technical_status_from_scientific_verdict(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="TRAIN-LD-010",
        candidate_version="c0.5.0",
        hypothesis="A schedule change may improve long-utterance optimization.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No WER or long-bucket improvement.",
        expectation="Lower long-bucket WER.",
        information_gain_rationale="10h can expose the optimization mechanism.",
        why_smaller_scale_is_insufficient="Smoke cannot estimate held-out WER.",
        promotion_rule="Promote if long-bucket WER improves >= 3% with no overall regression.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    started = register_scale_experiment(
        plan,
        tracker=tracker,
        experiment_id="probe_train_ld010_10h",
        large_data_plan=_large_data_plan(),
        setup=_execution_setup(),
        remaining_budget_usd=5.0,
    )
    assert started.technical_status == "IN_PROGRESS"
    assert started.scientific_verdict is None

    completed = complete_scale_experiment(
        started.experiment_id,
        tracker=tracker,
        result={"wer": 0.33, "long_wer": 0.41},
        scientific_verdict=ScientificVerdict.INCONCLUSIVE,
        actual_gpu_hours=0.9,
        actual_cost_usd=0.42,
        surprise="Overall stable; long bucket confidence interval overlaps baseline.",
        updated_belief="The schedule may be scale-sensitive but evidence is not decisive.",
        next_step="Inspect long-bucket failures before considering promotion.",
        evidence_refs=("artifacts/probe_train_ld010/raw_predictions.jsonl",),
        scale_action=ScaleAction.STOP,
    )
    assert completed.technical_status == "COMPLETED"
    assert completed.scientific_verdict == "INCONCLUSIVE"
    assert completed.status == "INCONCLUSIVE"
    assert completed.actual_gpu_hours == pytest.approx(0.9)
    assert completed.cost_usd == pytest.approx(0.42)


def test_in_progress_experiment_cannot_be_promoted(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-099",
        candidate_version="c0.5.0",
        hypothesis="Test an alternative architecture.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No WER improvement.",
        expectation="Lower WER.",
        information_gain_rationale="10h is sufficient for first comparison.",
        why_smaller_scale_is_insufficient="Smoke has no reliable generalization metric.",
        promotion_rule="Promote if WER improves >= 3%.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    started = register_scale_experiment(
        plan,
        tracker=tracker,
        experiment_id="probe_arch_ld099_10h",
        large_data_plan=_large_data_plan(),
        setup=_execution_setup(),
        remaining_budget_usd=5.0,
    )
    request = PromotionRequest(
        question_id="ARCH-LD-099",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.ACCEPT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=plan.promotion_rule,
        promotion_rule_met=True,
        evidence_refs=("registry#probe_arch_ld099_10h",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.0,
    )
    with pytest.raises(RuntimeError, match="Tamamlanmamış"):
        validate_promotion_request(
            request,
            tracker=tracker,
            source_experiment_id=started.experiment_id,
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=4.0,
        )


def test_registration_cannot_overwrite_predeclared_metadata(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-777",
        candidate_version="c0.5.0",
        hypothesis="Test one architecture.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No WER gain.",
        expectation="Lower WER.",
        information_gain_rationale="10h can distinguish the hypothesis.",
        why_smaller_scale_is_insufficient="Smoke cannot estimate validation WER.",
        promotion_rule="Promote if WER improves >= 3%.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    register_scale_experiment(
        plan,
        tracker=tracker,
        experiment_id="probe_arch_ld777_10h",
        large_data_plan=_large_data_plan(),
        setup=_execution_setup(),
        remaining_budget_usd=5.0,
    )
    changed = ScaleExperimentPlan(
        **{**plan.__dict__, "promotion_rule": "Easier rule after the fact."}
    )
    with pytest.raises(RuntimeError, match="overwrite edilemez"):
        register_scale_experiment(
            changed,
            tracker=tracker,
            experiment_id="probe_arch_ld777_10h",
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=5.0,
        )


def test_unknown_stage_fails_closed():
    plan = {"staged_subsets": {"10h": ["a"], "mystery": ["b"]}}
    with pytest.raises(ValueError, match="Bilinmeyen large-data stage"):
        ordered_research_scales(plan)


def test_scaling_curve_does_not_mix_dataset_revisions():
    records = [
        ExperimentRecord(
            experiment_id="old_revision",
            hypothesis="h",
            falsification_criteria="f",
            setup={"question_id": "ARCH-LD-001", "dataset_revision": "a" * 40},
            expectation="x",
            result={"wer": 0.30},
            status="PASSED",
            candidate_version="c0.5.0",
            data_scale="10h",
            technical_status="COMPLETED",
            scientific_verdict="ACCEPT",
        ),
        ExperimentRecord(
            experiment_id="new_revision",
            hypothesis="h",
            falsification_criteria="f",
            setup={"question_id": "ARCH-LD-001", "dataset_revision": "b" * 40},
            expectation="x",
            result={"wer": 0.20},
            status="PASSED",
            candidate_version="c0.5.0",
            data_scale="10h",
            technical_status="COMPLETED",
            scientific_verdict="ACCEPT",
        ),
    ]
    curve = build_scaling_curve(
        records,
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        dataset_revision="a" * 40,
        metric="wer",
        large_data_plan=_large_data_plan(),
    )
    assert [(p.experiment_id, p.value) for p in curve] == [
        ("old_revision", 0.30)
    ]


def test_completion_cannot_claim_unvalidated_promotion(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-201",
        candidate_version="c0.5.0",
        hypothesis="Test architecture A.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No WER gain.",
        expectation="Lower WER.",
        information_gain_rationale="10h is sufficient for initial discrimination.",
        why_smaller_scale_is_insufficient="Smoke cannot estimate held-out WER.",
        promotion_rule="Promote if WER improves >= 3%.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    started = register_scale_experiment(
        plan,
        tracker=tracker,
        experiment_id="probe_arch_ld201_10h",
        large_data_plan=_large_data_plan(),
        setup=_execution_setup(),
        remaining_budget_usd=5.0,
    )
    with pytest.raises(ValueError, match="register_promoted_experiment"):
        complete_scale_experiment(
            started.experiment_id,
            tracker=tracker,
            result={"wer": 0.30},
            scientific_verdict=ScientificVerdict.ACCEPT,
            actual_gpu_hours=0.9,
            actual_cost_usd=0.45,
            surprise="",
            updated_belief="Architecture A looks promising.",
            next_step="Consider promotion.",
            evidence_refs=("artifacts/probe_arch_ld201/metrics.json",),
            revalidation_trigger="Reopen on a new dataset revision or material subgroup shift.",
            scale_action=ScaleAction.PROMOTE_SCALE,
        )


def test_validated_promotion_creates_parent_child_registry_chain(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-202",
        candidate_version="c0.5.0",
        hypothesis="Alternative temporal encoder may scale better.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No WER improvement or worse long-bucket WER.",
        expectation="Lower WER without subgroup collapse.",
        information_gain_rationale="10h can discriminate initial generalization.",
        why_smaller_scale_is_insufficient="Smoke cannot estimate speaker-disjoint WER.",
        promotion_rule="Promote if WER improves >= 3% and no subgroup regresses.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    started = register_scale_experiment(
        plan,
        tracker=tracker,
        experiment_id="probe_arch_ld202_10h",
        large_data_plan=_large_data_plan(),
        setup=_execution_setup(),
        remaining_budget_usd=5.0,
    )
    completed = complete_scale_experiment(
        started.experiment_id,
        tracker=tracker,
        result={"wer": 0.28},
        scientific_verdict=ScientificVerdict.ACCEPT,
        actual_gpu_hours=0.9,
        actual_cost_usd=0.45,
        surprise="",
        updated_belief="Alternative encoder is promising.",
        next_step="Promote to 25h for stronger confirmation.",
        evidence_refs=(
            "artifacts/probe_arch_ld202/metrics.json",
            "artifacts/probe_arch_ld202/failures.jsonl",
        ),
        revalidation_trigger="Reopen if 25h+ scaling reverses the WER gain or a subgroup regresses.",
    )
    request = PromotionRequest(
        question_id="ARCH-LD-202",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.ACCEPT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=plan.promotion_rule,
        promotion_rule_met=True,
        evidence_refs=("registry#probe_arch_ld202_10h",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.0,
    )
    target_rule = "Promote 25h to 50h if WER gain persists and subgroup stability holds."
    child = register_promoted_experiment(
        request,
        tracker=tracker,
        source_experiment_id=completed.experiment_id,
        target_experiment_id="probe_arch_ld202_25h",
        target_promotion_rule=target_rule,
        large_data_plan=_large_data_plan(),
        remaining_budget_usd=4.5,
    )

    records = {r.experiment_id: r for r in tracker.load_all()}
    parent = records["probe_arch_ld202_10h"]
    assert parent.scale_action == "PROMOTE_SCALE"
    assert parent.extra_fields["promotion_to_scale"] == "25h"
    assert child.scale_parent_experiment_id == parent.experiment_id
    assert child.data_scale == "25h"
    assert child.technical_status == "IN_PROGRESS"
    assert child.promotion_rule == target_rule
    assert child.setup["train_subset_sha256"] == "2" * 64


def test_pre_result_contract_detects_manual_registry_tampering(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-303",
        candidate_version="c0.5.0",
        hypothesis="Alternative objective may improve alignment.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No WER or alignment gain.",
        expectation="Lower WER and fewer alignment failures.",
        information_gain_rationale="10h is sufficient for an initial controlled comparison.",
        why_smaller_scale_is_insufficient="Smoke cannot measure held-out alignment quality.",
        promotion_rule="Promote if WER improves >= 3% and alignment failures decrease.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    started = register_scale_experiment(
        plan,
        tracker=tracker,
        experiment_id="probe_arch_ld303_10h",
        large_data_plan=_large_data_plan(),
        setup=_execution_setup(),
        remaining_budget_usd=5.0,
    )

    import json
    rows = [
        json.loads(line)
        for line in tracker.registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["promotion_rule"] = "Tampered easier rule."
    tracker.registry_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Pre-result experiment contract"):
        complete_scale_experiment(
            started.experiment_id,
            tracker=tracker,
            result={"wer": 0.30},
            scientific_verdict=ScientificVerdict.ACCEPT,
            actual_gpu_hours=0.9,
            actual_cost_usd=0.45,
            surprise="",
            updated_belief="Would otherwise look promising.",
            next_step="Should never be accepted after tampering.",
        )


def test_promotion_cannot_cross_dataset_revision(tmp_path):
    source = seal_pre_result_contract(_source_record())
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    tracker.log(source)
    request = PromotionRequest(
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.ACCEPT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=source.promotion_rule,
        promotion_rule_met=True,
        evidence_refs=("registry#source",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.0,
    )
    different_revision_plan = {
        **_large_data_plan(),
        "dataset_revision": "d" * 40,
    }
    with pytest.raises(RuntimeError, match="farklı HF dataset revision"):
        validate_promotion_request(
            request,
            tracker=tracker,
            source_experiment_id=source.experiment_id,
            large_data_plan=different_revision_plan,
            remaining_budget_usd=5.0,
        )


def test_promotion_cannot_cross_split_or_source_stage(tmp_path):
    source = seal_pre_result_contract(_source_record())
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    tracker.log(source)
    request = PromotionRequest(
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.ACCEPT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=source.promotion_rule,
        promotion_rule_met=True,
        evidence_refs=("registry#source",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.0,
    )

    split_changed = {**_large_data_plan(), "split_map_sha256": "e" * 64}
    with pytest.raises(RuntimeError, match="farklı split map"):
        validate_promotion_request(
            request,
            tracker=tracker,
            source_experiment_id=source.experiment_id,
            large_data_plan=split_changed,
            remaining_budget_usd=5.0,
        )

    stage_changed = _large_data_plan()
    stage_changed["staged_summary"] = {
        **stage_changed["staged_summary"],
        "10h": {"sample_ids_sha256": "f" * 64},
    }
    with pytest.raises(RuntimeError, match="source-stage"):
        validate_promotion_request(
            request,
            tracker=tracker,
            source_experiment_id=source.experiment_id,
            large_data_plan=stage_changed,
            remaining_budget_usd=5.0,
        )


def test_non_smoke_registration_requires_execution_provenance(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-401",
        candidate_version="c0.5.0",
        hypothesis="Test a new temporal mechanism.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No WER improvement.",
        expectation="Lower WER.",
        information_gain_rationale="10h is enough for the first reliable comparison.",
        why_smaller_scale_is_insufficient="Smoke cannot estimate held-out WER.",
        promotion_rule="Promote if WER improves >= 3%.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    with pytest.raises(ValueError, match="code_revision"):
        register_scale_experiment(
            plan,
            tracker=tracker,
            experiment_id="probe_arch_ld401_10h",
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=5.0,
        )


def test_failed_run_is_closed_without_scientific_verdict(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="TRAIN-LD-402",
        candidate_version="c0.5.0",
        hypothesis="Test a training schedule.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No stable optimization improvement.",
        expectation="More stable validation loss.",
        information_gain_rationale="10h exposes training dynamics.",
        why_smaller_scale_is_insufficient="Smoke cannot estimate validation stability.",
        promotion_rule="Promote if validation stability improves without WER regression.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    started = register_scale_experiment(
        plan,
        tracker=tracker,
        experiment_id="probe_train_ld402_10h",
        large_data_plan=_large_data_plan(),
        setup=_execution_setup(),
        remaining_budget_usd=5.0,
    )
    failed = fail_scale_experiment(
        started.experiment_id,
        tracker=tracker,
        error="CUDA OOM after dataloader warmup",
        actual_gpu_hours=0.2,
        actual_cost_usd=0.1,
    )
    assert failed.technical_status == "ERROR"
    assert failed.scientific_verdict is None
    assert failed.status == "ERROR"
    assert failed.result["error"].startswith("CUDA OOM")

    request = PromotionRequest(
        question_id="TRAIN-LD-402",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.ACCEPT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=plan.promotion_rule,
        promotion_rule_met=True,
        evidence_refs=("registry#probe_train_ld402_10h",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.0,
    )
    with pytest.raises(RuntimeError, match="Tamamlanmamış"):
        validate_promotion_request(
            request,
            tracker=tracker,
            source_experiment_id=failed.experiment_id,
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=4.9,
        )


def test_governance_fails_closed_on_malformed_registry(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    tracker.registry_path.write_text("{not valid json}\n", encoding="utf-8")
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-403",
        candidate_version="c0.5.0",
        hypothesis="Test architecture.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No gain.",
        expectation="Gain.",
        information_gain_rationale="10h comparison.",
        why_smaller_scale_is_insufficient="Smoke cannot measure WER.",
        promotion_rule="Promote on reliable WER gain.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    with pytest.raises(RuntimeError, match="registry parse"):
        register_scale_experiment(
            plan,
            tracker=tracker,
            experiment_id="probe_arch_ld403_10h",
            large_data_plan=_large_data_plan(),
            setup=_execution_setup(),
            remaining_budget_usd=5.0,
        )


def test_nonpositive_hour_stage_fails_closed():
    plan = {
        **_large_data_plan(),
        "staged_subsets": {"0h": ["a"], "full": ["a"]},
        "staged_summary": {
            "0h": {"sample_ids_sha256": "1" * 64},
            "full": {"sample_ids_sha256": "4" * 64},
        },
    }
    with pytest.raises(ValueError, match="Bilinmeyen araştırma ölçeği"):
        ordered_research_scales(plan)


def test_promoted_child_requires_next_rule_before_results(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-501",
        candidate_version="c0.5.0",
        hypothesis="Alternative temporal encoder may improve generalization.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No WER gain.",
        expectation="Lower WER.",
        information_gain_rationale="10h is enough for the first comparison.",
        why_smaller_scale_is_insufficient="Smoke cannot measure held-out WER.",
        promotion_rule="Promote 10h to 25h if WER improves >= 3%.",
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    started = register_scale_experiment(
        plan,
        tracker=tracker,
        experiment_id="probe_arch_ld501_10h",
        large_data_plan=_large_data_plan(),
        setup=_execution_setup(),
        remaining_budget_usd=5.0,
    )
    completed = complete_scale_experiment(
        started.experiment_id,
        tracker=tracker,
        result={"wer": 0.28},
        scientific_verdict=ScientificVerdict.ACCEPT,
        actual_gpu_hours=0.9,
        actual_cost_usd=0.45,
        surprise="",
        updated_belief="Promising.",
        next_step="Consider 25h.",
        evidence_refs=("artifacts/probe_arch_ld501/metrics.json",),
        revalidation_trigger="Reopen if 25h scaling does not preserve the effect.",
    )
    request = PromotionRequest(
        question_id="ARCH-LD-501",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.ACCEPT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=plan.promotion_rule,
        promotion_rule_met=True,
        evidence_refs=("registry#probe_arch_ld501_10h",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.0,
    )
    with pytest.raises(ValueError, match="bir sonraki scale promotion rule"):
        register_promoted_experiment(
            request,
            tracker=tracker,
            source_experiment_id=completed.experiment_id,
            target_experiment_id="probe_arch_ld501_25h",
            target_promotion_rule="",
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=4.5,
        )


def test_source_experiment_cannot_be_promoted_twice(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    source = seal_pre_result_contract(_source_record())
    tracker.log(source)
    request = PromotionRequest(
        question_id="ARCH-LD-001",
        candidate_version="c0.5.0",
        from_scale="10h",
        to_scale="25h",
        scientific_verdict=ScientificVerdict.ACCEPT,
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=source.promotion_rule,
        promotion_rule_met=True,
        evidence_refs=("registry#source",),
        estimated_gpu_hours=2.0,
        estimated_cost_usd=1.0,
    )
    register_promoted_experiment(
        request,
        tracker=tracker,
        source_experiment_id=source.experiment_id,
        target_experiment_id="probe_arch_newfamily_25h_a",
        target_promotion_rule="Promote 25h to 50h if gain persists.",
        large_data_plan=_large_data_plan(),
        remaining_budget_usd=5.0,
    )
    with pytest.raises(RuntimeError, match="terminal scale action"):
        register_promoted_experiment(
            request,
            tracker=tracker,
            source_experiment_id=source.experiment_id,
            target_experiment_id="probe_arch_newfamily_25h_b",
            target_promotion_rule="Promote 25h to 50h if gain persists.",
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=5.0,
        )


def test_final_available_scale_does_not_require_promotion_rule():
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-502",
        candidate_version="c0.5.0",
        hypothesis="A full-data confirmation is scientifically necessary.",
        requested_scale="full",
        minimum_sufficient_scale="full",
        falsification_criteria="No confirmation of the mechanism.",
        expectation="Confirm the effect on the full research dataset.",
        information_gain_rationale="The effect is known to emerge only at full scale.",
        why_smaller_scale_is_insufficient="Prior scaling evidence shows separation only at full scale.",
        promotion_rule="",
        estimated_gpu_hours=4.0,
        estimated_cost_usd=2.0,
    )
    validate_scale_experiment_plan(
        plan,
        large_data_plan=_large_data_plan(),
        remaining_budget_usd=5.0,
    )


def test_nonfinite_costs_fail_closed():
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-503",
        candidate_version="c0.5.0",
        hypothesis="Test architecture.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No gain.",
        expectation="Gain.",
        information_gain_rationale="10h comparison.",
        why_smaller_scale_is_insufficient="Smoke cannot measure WER.",
        promotion_rule="Promote on reliable WER gain.",
        estimated_gpu_hours=float("nan"),
        estimated_cost_usd=0.5,
    )
    with pytest.raises(ValueError, match="finite non-negative"):
        validate_scale_experiment_plan(
            plan,
            large_data_plan=_large_data_plan(),
            remaining_budget_usd=5.0,
        )


def test_scoped_accept_requires_revalidation_trigger_and_evidence(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    plan = ScaleExperimentPlan(
        question_id="ARCH-LD-601",
        candidate_version="c0.5.0",
        hypothesis="Alternative objective improves alignment.",
        requested_scale="10h",
        minimum_sufficient_scale="10h",
        falsification_criteria="No alignment or WER improvement.",
        expectation="Lower alignment failures and WER.",
        information_gain_rationale="10h is sufficient for the first controlled comparison.",
        why_smaller_scale_is_insufficient="Smoke cannot estimate held-out behavior.",
        promotion_rule="Promote if WER improves >= 3% and alignment failures decrease.",
        evidence_scope=EvidenceScope.LARGE_DATA_REGIME.value,
        estimated_gpu_hours=1.0,
        estimated_cost_usd=0.5,
    )
    started = register_scale_experiment(
        plan,
        tracker=tracker,
        experiment_id="probe_arch_ld601_10h",
        large_data_plan=_large_data_plan(),
        setup=_execution_setup(),
        remaining_budget_usd=5.0,
    )
    with pytest.raises(ValueError, match="revalidation_trigger"):
        complete_scale_experiment(
            started.experiment_id,
            tracker=tracker,
            result={"wer": 0.27},
            scientific_verdict=ScientificVerdict.ACCEPT,
            actual_gpu_hours=0.9,
            actual_cost_usd=0.45,
            surprise="",
            updated_belief="The objective appears better in this large-data regime.",
            next_step="Consider scale promotion.",
            evidence_refs=("artifacts/probe_arch_ld601/metrics.json",),
        )

    with pytest.raises(ValueError, match="evidence ref"):
        complete_scale_experiment(
            started.experiment_id,
            tracker=tracker,
            result={"wer": 0.27},
            scientific_verdict=ScientificVerdict.ACCEPT,
            actual_gpu_hours=0.9,
            actual_cost_usd=0.45,
            surprise="",
            updated_belief="The objective appears better in this large-data regime.",
            next_step="Consider scale promotion.",
            evidence_refs=(),
            revalidation_trigger="Reopen if the effect disappears at 25h.",
        )
