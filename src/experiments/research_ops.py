"""Operational CLI/session preflight for large-data autonomous research.

This module wires the large-data controller into one fail-closed interface that an
agent can use for status, registration, completion, technical failure, and scale
promotion. It never starts training itself.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from src.data.large_data import load_large_data_plan
from src.experiments.large_data_controller import (
    EvidenceScope,
    PromotionRequest,
    ScaleAction,
    ScaleExperimentPlan,
    ScientificVerdict,
    complete_scale_experiment,
    fail_scale_experiment,
    ordered_research_scales,
    register_promoted_experiment,
    register_scale_experiment,
)
from src.experiments.research_governance import (
    ResearchStage,
    compute_file_sha256,
    load_candidate,
    require_clean_code_revision,
)
from src.experiments.tracker import ExperimentRecord, ExperimentTracker


ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_CANDIDATE = ROOT / "configs" / "research_candidate.yaml"
DEFAULT_PLAN = ROOT / "research" / "large_data_plan.json"
DEFAULT_POLICY = ROOT / "configs" / "large_data_research.yaml"


@dataclass(frozen=True)
class ResearchSessionReport:
    ready_for_new_run: bool
    candidate_version: str
    stage: str
    active_question: Optional[str]
    dataset_id: Optional[str]
    dataset_revision: Optional[str]
    available_scales: Tuple[str, ...]
    in_progress_experiment_ids: Tuple[str, ...]
    blockers: Tuple[str, ...]
    warnings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_policy(path: pathlib.Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Large-data policy bulunamadı: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("phase") != "large_data_research":
        raise RuntimeError(f"Beklenmeyen large-data policy phase: {payload.get('phase')}")
    return payload


def _active_question_definition(candidate_raw: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    active = candidate_raw.get("active_question")
    if not active:
        return None
    matches = [q for q in candidate_raw.get("open_questions", []) if q.get("id") == active]
    if len(matches) != 1:
        return None
    return matches[0]


def evaluate_research_session(
    *,
    candidate_path: pathlib.Path = DEFAULT_CANDIDATE,
    plan_path: pathlib.Path = DEFAULT_PLAN,
    policy_path: pathlib.Path = DEFAULT_POLICY,
    tracker: Optional[ExperimentTracker] = None,
) -> ResearchSessionReport:
    """Reconcile candidate + plan + registry before any large-data research mutation."""
    blockers: List[str] = []
    warnings: List[str] = []
    available_scales: Tuple[str, ...] = ()
    dataset_id: Optional[str] = None
    dataset_revision: Optional[str] = None

    candidate = load_candidate(candidate_path)
    policy = _load_policy(policy_path)
    expected_candidate = str(policy.get("candidate_version") or "").strip()
    expected_source = str(((policy.get("data") or {}).get("source")) or "").strip()

    if candidate.candidate_version != expected_candidate:
        blockers.append(
            f"candidate_version_mismatch:{candidate.candidate_version}!={expected_candidate}"
        )

    if candidate.stage not in {
        ResearchStage.RESEARCHING,
        ResearchStage.CONFIRMING,
        ResearchStage.REHEARSAL,
    }:
        blockers.append(f"candidate_stage_not_researchable:{candidate.stage.value}")

    if candidate.training.get("research_training_authorized") is not True:
        blockers.append("research_training_not_authorized")

    active_question = candidate.raw.get("active_question")
    active_def = _active_question_definition(candidate.raw)
    if not active_question:
        blockers.append("active_question_missing")
    elif active_def is None:
        blockers.append(f"active_question_definition_missing:{active_question}")
    elif active_def.get("status") not in {"active", "open", "in_progress"}:
        blockers.append(
            f"active_question_not_open:{active_question}:{active_def.get('status')}"
        )

    if not plan_path.is_file():
        blockers.append(f"large_data_plan_missing:{plan_path}")
        plan: Dict[str, Any] = {}
    else:
        plan = load_large_data_plan(plan_path)
        dataset_id = str(plan.get("dataset_id") or "") or None
        dataset_revision = str(plan.get("dataset_revision") or "") or None
        available_scales = ordered_research_scales(plan)

        candidate_source = str(((candidate.raw.get("data") or {}).get("source")) or "")
        if expected_source and dataset_id != expected_source:
            blockers.append(f"plan_dataset_policy_mismatch:{dataset_id}!={expected_source}")
        if dataset_id and candidate_source != dataset_id:
            blockers.append(
                f"candidate_dataset_plan_mismatch:{candidate_source}!={dataset_id}"
            )
        if bool(plan.get("speaker_identity_is_proxy")):
            warnings.append(
                "speaker_identity_is_proxy: channel/creator identity requires human audit"
            )

    tracker = tracker or ExperimentTracker()
    records = tracker.load_all(strict=True)
    in_progress = [
        r
        for r in records
        if r.technical_status == "IN_PROGRESS" and r.candidate_version is not None
    ]
    in_progress_ids = tuple(r.experiment_id for r in in_progress)

    if len(in_progress) > 1:
        blockers.append(
            "multiple_in_progress_large_data_experiments:" + ",".join(in_progress_ids)
        )
    elif len(in_progress) == 1:
        running = in_progress[0]
        if running.candidate_version != candidate.candidate_version:
            blockers.append(
                "in_progress_candidate_mismatch:"
                f"{running.candidate_version}!={candidate.candidate_version}"
            )
        running_question = str((running.setup or {}).get("question_id") or "")
        if active_question and running_question != active_question:
            blockers.append(
                f"in_progress_question_mismatch:{running_question}!={active_question}"
            )
        blockers.append(f"active_experiment_exists:{running.experiment_id}")

    return ResearchSessionReport(
        ready_for_new_run=len(blockers) == 0,
        candidate_version=candidate.candidate_version,
        stage=candidate.stage.value,
        active_question=active_question,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        available_scales=available_scales,
        in_progress_experiment_ids=in_progress_ids,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def require_session_ready_for_new_run(
    *,
    candidate_path: pathlib.Path,
    plan_path: pathlib.Path,
    policy_path: pathlib.Path,
    tracker: ExperimentTracker,
) -> ResearchSessionReport:
    report = evaluate_research_session(
        candidate_path=candidate_path,
        plan_path=plan_path,
        policy_path=policy_path,
        tracker=tracker,
    )
    if not report.ready_for_new_run:
        raise RuntimeError(f"Agentic research session blocked: {list(report.blockers)}")
    return report


def _read_json_object(path: pathlib.Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} JSON bulunamadı: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} JSON object olmalıdır.")
    return payload


def _build_scale_plan(spec: Mapping[str, Any]) -> ScaleExperimentPlan:
    required = (
        "question_id",
        "candidate_version",
        "hypothesis",
        "requested_scale",
        "minimum_sufficient_scale",
        "information_gain_rationale",
        "falsification_criteria",
        "expectation",
    )
    missing = [key for key in required if not spec.get(key)]
    if missing:
        raise ValueError(f"Experiment spec eksik alanlar: {missing}")
    return ScaleExperimentPlan(
        question_id=str(spec["question_id"]),
        candidate_version=str(spec["candidate_version"]),
        hypothesis=str(spec["hypothesis"]),
        requested_scale=str(spec["requested_scale"]),
        minimum_sufficient_scale=str(spec["minimum_sufficient_scale"]),
        information_gain_rationale=str(spec["information_gain_rationale"]),
        why_smaller_scale_is_insufficient=str(
            spec.get("why_smaller_scale_is_insufficient") or ""
        ),
        promotion_rule=str(spec.get("promotion_rule") or ""),
        falsification_criteria=str(spec["falsification_criteria"]),
        expectation=str(spec["expectation"]),
        evidence_scope=str(
            spec.get("evidence_scope") or EvidenceScope.LARGE_DATA_REGIME.value
        ),
        estimated_gpu_hours=float(spec.get("estimated_gpu_hours", 0.0)),
        estimated_cost_usd=float(spec.get("estimated_cost_usd", 0.0)),
    )


def build_execution_setup(
    *,
    candidate_path: pathlib.Path,
    code_revision: str,
    seed: int,
    initializer_id: str,
    initializer_sha256: str,
    extra_setup: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build exact execution provenance without starting a run."""
    setup = dict(extra_setup or {})
    setup.update(
        {
            "code_revision": code_revision,
            "candidate_recipe_sha256": compute_file_sha256(candidate_path),
            "seed": seed,
            "initializer_id": initializer_id,
            "initializer_sha256": initializer_sha256,
        }
    )
    return setup


def register_from_spec(
    *,
    spec_path: pathlib.Path,
    experiment_id: str,
    remaining_budget_usd: float,
    seed: int,
    initializer_id: str,
    initializer_sha256: str,
    candidate_path: pathlib.Path = DEFAULT_CANDIDATE,
    plan_path: pathlib.Path = DEFAULT_PLAN,
    policy_path: pathlib.Path = DEFAULT_POLICY,
    tracker: Optional[ExperimentTracker] = None,
    code_revision: Optional[str] = None,
) -> ExperimentRecord:
    tracker = tracker or ExperimentTracker()
    report = require_session_ready_for_new_run(
        candidate_path=candidate_path,
        plan_path=plan_path,
        policy_path=policy_path,
        tracker=tracker,
    )
    spec = _read_json_object(spec_path, "experiment spec")
    experiment = _build_scale_plan(spec)

    if experiment.candidate_version != report.candidate_version:
        raise RuntimeError(
            f"Spec candidate mismatch: {experiment.candidate_version}!={report.candidate_version}"
        )
    if experiment.question_id != report.active_question:
        raise RuntimeError(
            f"Spec question mismatch: {experiment.question_id}!={report.active_question}"
        )

    revision = code_revision or require_clean_code_revision(ROOT)
    setup = build_execution_setup(
        candidate_path=candidate_path,
        code_revision=revision,
        seed=seed,
        initializer_id=initializer_id,
        initializer_sha256=initializer_sha256,
        extra_setup=spec.get("setup") if isinstance(spec.get("setup"), Mapping) else None,
    )
    plan = load_large_data_plan(plan_path)
    return register_scale_experiment(
        experiment,
        tracker=tracker,
        experiment_id=experiment_id,
        large_data_plan=plan,
        remaining_budget_usd=remaining_budget_usd,
        setup=setup,
    )


def _find_record(tracker: ExperimentTracker, experiment_id: str) -> ExperimentRecord:
    matches = [
        r for r in tracker.load_all(strict=True) if r.experiment_id == experiment_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Experiment registry lookup expected exactly one record: {experiment_id}, found={len(matches)}"
        )
    return matches[0]


def complete_from_files(
    *,
    experiment_id: str,
    result_path: pathlib.Path,
    verdict: str,
    actual_gpu_hours: float,
    actual_cost_usd: float,
    updated_belief: str,
    next_step: str,
    evidence_refs: Sequence[str],
    surprise: str = "",
    revalidation_trigger: str = "",
    tracker: Optional[ExperimentTracker] = None,
) -> ExperimentRecord:
    tracker = tracker or ExperimentTracker()
    _find_record(tracker, experiment_id)
    result = _read_json_object(result_path, "result")
    return complete_scale_experiment(
        experiment_id,
        tracker=tracker,
        result=result,
        scientific_verdict=ScientificVerdict(verdict),
        actual_gpu_hours=actual_gpu_hours,
        actual_cost_usd=actual_cost_usd,
        surprise=surprise,
        updated_belief=updated_belief,
        next_step=next_step,
        evidence_refs=tuple(evidence_refs),
        revalidation_trigger=revalidation_trigger,
    )


def fail_registered_run(
    *,
    experiment_id: str,
    error: str,
    actual_gpu_hours: float,
    actual_cost_usd: float,
    tracker: Optional[ExperimentTracker] = None,
) -> ExperimentRecord:
    tracker = tracker or ExperimentTracker()
    return fail_scale_experiment(
        experiment_id,
        tracker=tracker,
        error=error,
        actual_gpu_hours=actual_gpu_hours,
        actual_cost_usd=actual_cost_usd,
    )


def promote_registered_run(
    *,
    source_experiment_id: str,
    target_experiment_id: str,
    to_scale: str,
    target_promotion_rule: str,
    promotion_rule_met: bool,
    evidence_refs: Sequence[str],
    estimated_gpu_hours: float,
    estimated_cost_usd: float,
    remaining_budget_usd: float,
    scale_sensitive_ambiguity: bool = False,
    why_larger_scale_resolves_ambiguity: str = "",
    skip_scale_justification: str = "",
    plan_path: pathlib.Path = DEFAULT_PLAN,
    tracker: Optional[ExperimentTracker] = None,
) -> ExperimentRecord:
    tracker = tracker or ExperimentTracker()
    source = _find_record(tracker, source_experiment_id)
    if source.scientific_verdict is None:
        raise RuntimeError("Source experiment scientific verdict içermiyor.")
    plan = load_large_data_plan(plan_path)
    request = PromotionRequest(
        question_id=str((source.setup or {}).get("question_id") or ""),
        candidate_version=str(source.candidate_version or ""),
        from_scale=str(source.data_scale or ""),
        to_scale=to_scale,
        scientific_verdict=ScientificVerdict(source.scientific_verdict),
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=str(source.promotion_rule or ""),
        promotion_rule_met=promotion_rule_met,
        evidence_refs=tuple(evidence_refs),
        estimated_gpu_hours=estimated_gpu_hours,
        estimated_cost_usd=estimated_cost_usd,
        scale_sensitive_ambiguity=scale_sensitive_ambiguity,
        why_larger_scale_resolves_ambiguity=why_larger_scale_resolves_ambiguity,
        skip_scale_justification=skip_scale_justification,
    )
    return register_promoted_experiment(
        request,
        tracker=tracker,
        source_experiment_id=source_experiment_id,
        target_experiment_id=target_experiment_id,
        target_promotion_rule=target_promotion_rule,
        large_data_plan=plan,
        remaining_budget_usd=remaining_budget_usd,
    )


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed operational CLI for autonomous large-data VSR research."
    )
    parser.add_argument(
        "--candidate-path", type=pathlib.Path, default=DEFAULT_CANDIDATE
    )
    parser.add_argument("--plan-path", type=pathlib.Path, default=DEFAULT_PLAN)
    parser.add_argument("--policy-path", type=pathlib.Path, default=DEFAULT_POLICY)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")

    register = sub.add_parser("register")
    register.add_argument("--spec", type=pathlib.Path, required=True)
    register.add_argument("--experiment-id", required=True)
    register.add_argument("--remaining-budget-usd", type=float, required=True)
    register.add_argument("--seed", type=int, required=True)
    register.add_argument("--initializer-id", required=True)
    register.add_argument("--initializer-sha256", required=True)

    complete = sub.add_parser("complete")
    complete.add_argument("--experiment-id", required=True)
    complete.add_argument("--result", type=pathlib.Path, required=True)
    complete.add_argument(
        "--verdict",
        choices=[v.value for v in ScientificVerdict],
        required=True,
    )
    complete.add_argument("--actual-gpu-hours", type=float, required=True)
    complete.add_argument("--actual-cost-usd", type=float, required=True)
    complete.add_argument("--updated-belief", required=True)
    complete.add_argument("--next-step", required=True)
    complete.add_argument("--surprise", default="")
    complete.add_argument("--revalidation-trigger", default="")
    complete.add_argument("--evidence-ref", action="append", default=[])

    fail = sub.add_parser("fail")
    fail.add_argument("--experiment-id", required=True)
    fail.add_argument("--error", required=True)
    fail.add_argument("--actual-gpu-hours", type=float, required=True)
    fail.add_argument("--actual-cost-usd", type=float, required=True)

    promote = sub.add_parser("promote")
    promote.add_argument("--source-experiment-id", required=True)
    promote.add_argument("--target-experiment-id", required=True)
    promote.add_argument("--to-scale", required=True)
    promote.add_argument("--target-promotion-rule", default="")
    promote.add_argument("--promotion-rule-met", action="store_true")
    promote.add_argument("--evidence-ref", action="append", default=[])
    promote.add_argument("--estimated-gpu-hours", type=float, required=True)
    promote.add_argument("--estimated-cost-usd", type=float, required=True)
    promote.add_argument("--remaining-budget-usd", type=float, required=True)
    promote.add_argument("--scale-sensitive-ambiguity", action="store_true")
    promote.add_argument("--why-larger-scale-resolves-ambiguity", default="")
    promote.add_argument("--skip-scale-justification", default="")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tracker = ExperimentTracker()

    if args.command == "status":
        report = evaluate_research_session(
            candidate_path=args.candidate_path,
            plan_path=args.plan_path,
            policy_path=args.policy_path,
            tracker=tracker,
        )
        _print_json(report.to_dict())
        return 0 if report.ready_for_new_run else 2

    if args.command == "register":
        record = register_from_spec(
            spec_path=args.spec,
            experiment_id=args.experiment_id,
            remaining_budget_usd=args.remaining_budget_usd,
            seed=args.seed,
            initializer_id=args.initializer_id,
            initializer_sha256=args.initializer_sha256,
            candidate_path=args.candidate_path,
            plan_path=args.plan_path,
            policy_path=args.policy_path,
            tracker=tracker,
        )
    elif args.command == "complete":
        record = complete_from_files(
            experiment_id=args.experiment_id,
            result_path=args.result,
            verdict=args.verdict,
            actual_gpu_hours=args.actual_gpu_hours,
            actual_cost_usd=args.actual_cost_usd,
            updated_belief=args.updated_belief,
            next_step=args.next_step,
            evidence_refs=args.evidence_ref,
            surprise=args.surprise,
            revalidation_trigger=args.revalidation_trigger,
            tracker=tracker,
        )
    elif args.command == "fail":
        record = fail_registered_run(
            experiment_id=args.experiment_id,
            error=args.error,
            actual_gpu_hours=args.actual_gpu_hours,
            actual_cost_usd=args.actual_cost_usd,
            tracker=tracker,
        )
    elif args.command == "promote":
        record = promote_registered_run(
            source_experiment_id=args.source_experiment_id,
            target_experiment_id=args.target_experiment_id,
            to_scale=args.to_scale,
            target_promotion_rule=args.target_promotion_rule,
            promotion_rule_met=args.promotion_rule_met,
            evidence_refs=args.evidence_ref,
            estimated_gpu_hours=args.estimated_gpu_hours,
            estimated_cost_usd=args.estimated_cost_usd,
            remaining_budget_usd=args.remaining_budget_usd,
            scale_sensitive_ambiguity=args.scale_sensitive_ambiguity,
            why_larger_scale_resolves_ambiguity=args.why_larger_scale_resolves_ambiguity,
            skip_scale_justification=args.skip_scale_justification,
            plan_path=args.plan_path,
            tracker=tracker,
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

    _print_json(record.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
