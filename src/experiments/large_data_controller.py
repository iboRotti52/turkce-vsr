"""Cost-aware experiment scaling for large-data autonomous VSR research.

This module constrains experiment *cost and scale*, never the scientific search
space. Architecture/objective/frontend/decoder choices remain governed by the
research question and evidence in GEMINI.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.data.large_data import load_large_data_plan
from src.experiments.tracker import ExperimentRecord


VIRTUAL_SMOKE_SCALE = "smoke"


class ScientificVerdict(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


class ScaleAction(str, Enum):
    STOP = "STOP"
    RETEST_SAME_SCALE = "RETEST_SAME_SCALE"
    PROMOTE_SCALE = "PROMOTE_SCALE"


class EvidenceScope(str, Enum):
    MECHANISM_GENERAL = "mechanism_general"
    SMALL_DATA_REGIME = "small_data_regime"
    LARGE_DATA_REGIME = "large_data_regime"
    DATASET_REVISION_SPECIFIC = "dataset_revision_specific"
    SCALE_SPECIFIC = "scale_specific"


@dataclass(frozen=True)
class ScaleExperimentPlan:
    question_id: str
    candidate_version: str
    hypothesis: str
    requested_scale: str
    minimum_sufficient_scale: str
    information_gain_rationale: str
    why_smaller_scale_is_insufficient: str = ""
    promotion_rule: str = ""
    falsification_criteria: str = ""
    expectation: str = ""
    evidence_scope: str = EvidenceScope.LARGE_DATA_REGIME.value
    estimated_gpu_hours: float = 0.0
    estimated_cost_usd: float = 0.0


@dataclass(frozen=True)
class PromotionRequest:
    question_id: str
    candidate_version: str
    from_scale: str
    to_scale: str
    scientific_verdict: ScientificVerdict
    action: ScaleAction
    promotion_rule: str
    promotion_rule_met: bool
    evidence_refs: Tuple[str, ...]
    estimated_gpu_hours: float
    estimated_cost_usd: float
    scale_sensitive_ambiguity: bool = False
    why_larger_scale_resolves_ambiguity: str = ""
    skip_scale_justification: str = ""


@dataclass(frozen=True)
class ScalingPoint:
    experiment_id: str
    scale: str
    metric: str
    value: float
    evidence_scope: Optional[str]


def _stage_hours(stage: str) -> float:
    if stage == VIRTUAL_SMOKE_SCALE:
        return 0.0
    if stage == "full":
        return float("inf")
    if stage.endswith("h"):
        try:
            hours = float(stage[:-1])
            if math.isfinite(hours) and hours > 0:
                return hours
        except ValueError:
            pass
    raise ValueError(f"Bilinmeyen araştırma ölçeği: {stage}")


def ordered_research_scales(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return smoke + actual deterministic stages from a verified data plan."""
    stages = list((plan.get("staged_subsets") or {}).keys())
    unknown = [s for s in stages if s != "full" and not s.endswith("h")]
    if unknown:
        raise ValueError(f"Bilinmeyen large-data stage isimleri: {unknown}")
    numeric = sorted((s for s in stages if s.endswith("h")), key=_stage_hours)
    # Force numeric parsing now so malformed values like 'manyh' fail closed.
    for stage in numeric:
        _stage_hours(stage)
    ordered: List[str] = [VIRTUAL_SMOKE_SCALE, *numeric]
    if "full" in stages:
        ordered.append("full")
    return tuple(ordered)


def load_ordered_research_scales(plan_path: pathlib.Path) -> Tuple[str, ...]:
    return ordered_research_scales(load_large_data_plan(plan_path))


def _require_nonempty(label: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{label} boş olamaz.")


def _require_finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} finite non-negative sayı olmalıdır.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} finite non-negative sayı olmalıdır.")
    return numeric


def _require_nonnegative_cost(gpu_hours: float, cost_usd: float) -> None:
    _require_finite_nonnegative(gpu_hours, "estimated_gpu_hours")
    _require_finite_nonnegative(cost_usd, "estimated_cost_usd")


def _normalize_verdict(value: Any) -> ScientificVerdict:
    try:
        return ScientificVerdict(value)
    except (TypeError, ValueError):
        raise ValueError(f"Bilinmeyen scientific verdict: {value!r}")


def _normalize_scale_action(value: Any) -> ScaleAction:
    try:
        return ScaleAction(value)
    except (TypeError, ValueError):
        raise ValueError(f"Bilinmeyen scale action: {value!r}")


def _find_registry_record(tracker: "ExperimentTracker", experiment_id: str) -> ExperimentRecord:
    from src.experiments.tracker import ExperimentTracker

    if not isinstance(tracker, ExperimentTracker):
        raise TypeError("tracker ExperimentTracker olmalıdır.")
    matches = [r for r in tracker.load_all(strict=True) if r.experiment_id == experiment_id]
    if not matches:
        raise FileNotFoundError(f"Registry experiment bulunamadı: {experiment_id}")
    if len(matches) != 1:
        raise RuntimeError(f"Registry duplicate experiment_id içeriyor: {experiment_id}")
    return matches[0]


def _pre_result_contract_payload(record: ExperimentRecord) -> Dict[str, Any]:
    return {
        "experiment_id": record.experiment_id,
        "candidate_version": record.candidate_version,
        "hypothesis": record.hypothesis,
        "falsification_criteria": record.falsification_criteria,
        "expectation": record.expectation,
        "setup": record.setup,
        "data_scale": record.data_scale,
        "minimum_sufficient_scale": record.minimum_sufficient_scale,
        "evidence_scope": record.evidence_scope,
        "promotion_rule": record.promotion_rule,
        "estimated_gpu_hours": record.estimated_gpu_hours,
        "cost_estimate_usd": record.cost_estimate_usd,
        "scale_parent_experiment_id": record.scale_parent_experiment_id,
    }


def compute_pre_result_contract_sha256(record: ExperimentRecord) -> str:
    payload = json.dumps(
        _pre_result_contract_payload(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seal_pre_result_contract(record: ExperimentRecord) -> ExperimentRecord:
    """Attach a hash over metadata that must be fixed before seeing results."""
    return replace(
        record,
        pre_result_contract_sha256=compute_pre_result_contract_sha256(record),
    )


def require_pre_result_contract_intact(record: ExperimentRecord) -> None:
    expected = _require_hex(
        record.pre_result_contract_sha256,
        64,
        "pre_result_contract_sha256",
    )
    actual = compute_pre_result_contract_sha256(record)
    if actual != expected:
        raise RuntimeError(
            "Pre-result experiment contract değişmiş/tamper edilmiş; "
            f"expected={expected}, actual={actual}"
        )


def _require_hex(value: Any, length: int, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != length or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} geçerli {length}-hex değer olmalıdır.")
    return normalized


def _stage_sample_hash(plan: Mapping[str, Any], scale: str) -> Optional[str]:
    if scale == VIRTUAL_SMOKE_SCALE:
        return None
    summary = (plan.get("staged_summary") or {}).get(scale) or {}
    return _require_hex(summary.get("sample_ids_sha256"), 64, f"{scale} sample_ids_sha256")


def _require_large_data_identity(plan: Mapping[str, Any], scale: str) -> None:
    if scale == VIRTUAL_SMOKE_SCALE:
        return
    _require_hex(plan.get("dataset_revision"), 40, "dataset_revision")
    _require_hex(plan.get("split_map_sha256"), 64, "split_map_sha256")
    _require_hex(plan.get("plan_sha256"), 64, "plan_sha256")
    _stage_sample_hash(plan, scale)


def _require_execution_provenance(setup: Mapping[str, Any]) -> None:
    _require_hex(setup.get("code_revision"), 40, "code_revision")
    _require_hex(
        setup.get("candidate_recipe_sha256"),
        64,
        "candidate_recipe_sha256",
    )
    _require_nonempty("initializer_id", str(setup.get("initializer_id") or ""))
    _require_hex(setup.get("initializer_sha256"), 64, "initializer_sha256")
    seed = setup.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed integer olmalıdır.")


def validate_scale_experiment_plan(
    experiment: ScaleExperimentPlan,
    *,
    large_data_plan: Mapping[str, Any],
    remaining_budget_usd: float,
) -> None:
    """Validate minimum-sufficient-scale and cost governance.

    The controller deliberately does NOT inspect or constrain the scientific
    change itself. A radically different model family is allowed if the research
    engine provides a falsifiable hypothesis and appropriate evidence.
    """
    for label, value in (
        ("question_id", experiment.question_id),
        ("candidate_version", experiment.candidate_version),
        ("hypothesis", experiment.hypothesis),
        ("information_gain_rationale", experiment.information_gain_rationale),
        ("falsification_criteria", experiment.falsification_criteria),
        ("expectation", experiment.expectation),
        ("requested_scale", experiment.requested_scale),
        ("minimum_sufficient_scale", experiment.minimum_sufficient_scale),
    ):
        _require_nonempty(label, value)

    _require_nonnegative_cost(
        experiment.estimated_gpu_hours,
        experiment.estimated_cost_usd,
    )
    remaining_budget_usd = _require_finite_nonnegative(
        remaining_budget_usd, "remaining_budget_usd"
    )
    if experiment.estimated_cost_usd > remaining_budget_usd:
        raise RuntimeError(
            f"Deney bütçeyi aşıyor: estimate={experiment.estimated_cost_usd:.2f} USD, "
            f"remaining={remaining_budget_usd:.2f} USD"
        )

    scales = ordered_research_scales(large_data_plan)
    if experiment.requested_scale not in scales:
        raise ValueError(
            f"İstenen scale plan içinde yok: {experiment.requested_scale}; available={scales}"
        )
    if experiment.minimum_sufficient_scale not in scales:
        raise ValueError(
            "minimum_sufficient_scale plan içinde yok: "
            f"{experiment.minimum_sufficient_scale}; available={scales}"
        )
    if experiment.requested_scale != experiment.minimum_sufficient_scale:
        raise ValueError(
            "İlk deney requested_scale ile minimum_sufficient_scale aynı olmalıdır; "
            "daha büyük scale'e geçiş promotion olarak kaydedilmelidir."
        )
    _require_large_data_identity(large_data_plan, experiment.requested_scale)

    requested_idx = scales.index(experiment.requested_scale)
    if requested_idx > 0 and not experiment.why_smaller_scale_is_insufficient.strip():
        raise ValueError(
            "Smoke/en küçük mevcut scale atlanıyorsa neden daha küçük scale'in "
            "hipotezi ayırt edemeyeceği açıkça yazılmalıdır."
        )

    # A run with a larger available stage must predeclare the rule that would
    # justify spending more. The final available scale has nowhere to promote.
    has_larger_scale = requested_idx + 1 < len(scales)
    if (
        experiment.requested_scale != VIRTUAL_SMOKE_SCALE
        and has_larger_scale
        and not experiment.promotion_rule.strip()
    ):
        raise ValueError(
            "Large-data araştırma koşusu promotion_rule olmadan başlatılamaz."
        )
    try:
        EvidenceScope(experiment.evidence_scope)
    except ValueError:
        raise ValueError(f"Bilinmeyen evidence_scope: {experiment.evidence_scope}")


def register_scale_experiment(
    experiment: ScaleExperimentPlan,
    *,
    tracker: "ExperimentTracker",
    experiment_id: str,
    large_data_plan: Mapping[str, Any],
    remaining_budget_usd: float,
    setup: Optional[Mapping[str, Any]] = None,
) -> ExperimentRecord:
    """Validate and persist an IN_PROGRESS large-data experiment before execution."""
    from src.experiments.tracker import ExperimentTracker

    if not isinstance(tracker, ExperimentTracker):
        raise TypeError("tracker ExperimentTracker olmalıdır.")
    _require_nonempty("experiment_id", experiment_id)
    if any(r.experiment_id == experiment_id for r in tracker.load_all(strict=True)):
        raise RuntimeError(
            f"experiment_id zaten registry'de mevcut; pre-result metadata overwrite edilemez: {experiment_id}"
        )
    validate_scale_experiment_plan(
        experiment,
        large_data_plan=large_data_plan,
        remaining_budget_usd=remaining_budget_usd,
    )

    setup_payload: Dict[str, Any] = dict(setup or {})
    setup_payload["question_id"] = experiment.question_id
    setup_payload["data_scale"] = experiment.requested_scale
    setup_payload["minimum_sufficient_scale"] = experiment.minimum_sufficient_scale
    setup_payload["information_gain_rationale"] = experiment.information_gain_rationale
    setup_payload["why_smaller_scale_is_insufficient"] = (
        experiment.why_smaller_scale_is_insufficient
    )
    if experiment.requested_scale != VIRTUAL_SMOKE_SCALE:
        _require_execution_provenance(setup_payload)

    stage_hash = _stage_sample_hash(large_data_plan, experiment.requested_scale)
    if stage_hash:
        setup_payload["train_subset_sha256"] = stage_hash
    if large_data_plan.get("dataset_revision"):
        setup_payload["dataset_revision"] = large_data_plan["dataset_revision"]
    if large_data_plan.get("split_map_sha256"):
        setup_payload["split_map_sha256"] = large_data_plan["split_map_sha256"]
    if large_data_plan.get("plan_sha256"):
        setup_payload["large_data_plan_sha256"] = large_data_plan["plan_sha256"]

    record = ExperimentRecord(
        experiment_id=experiment_id,
        hypothesis=experiment.hypothesis,
        falsification_criteria=experiment.falsification_criteria,
        setup=setup_payload,
        expectation=experiment.expectation,
        status="IN_PROGRESS",
        candidate_version=experiment.candidate_version,
        data_scale=experiment.requested_scale,
        minimum_sufficient_scale=experiment.minimum_sufficient_scale,
        evidence_scope=experiment.evidence_scope,
        promotion_rule=experiment.promotion_rule or None,
        scale_action=ScaleAction.STOP.value,
        estimated_gpu_hours=experiment.estimated_gpu_hours,
        cost_estimate_usd=experiment.estimated_cost_usd,
        technical_status="IN_PROGRESS",
    )
    record = seal_pre_result_contract(record)
    tracker.log(record)
    return record


def complete_scale_experiment(
    experiment_id: str,
    *,
    tracker: "ExperimentTracker",
    result: Mapping[str, Any],
    scientific_verdict: ScientificVerdict,
    actual_gpu_hours: float,
    actual_cost_usd: float,
    surprise: str,
    updated_belief: str,
    next_step: str,
    evidence_refs: Tuple[str, ...],
    revalidation_trigger: str = "",
    scale_action: ScaleAction = ScaleAction.STOP,
) -> ExperimentRecord:
    """Persist technical completion separately from the scientific verdict."""
    from src.experiments.tracker import ExperimentTracker

    if not isinstance(tracker, ExperimentTracker):
        raise TypeError("tracker ExperimentTracker olmalıdır.")
    _require_nonempty("experiment_id", experiment_id)
    record = _find_registry_record(tracker, experiment_id)
    require_pre_result_contract_intact(record)
    if record.technical_status != "IN_PROGRESS":
        raise RuntimeError(
            f"Yalnız pre-registered IN_PROGRESS deney tamamlanabilir; mevcut={record.technical_status}"
        )
    scientific_verdict = _normalize_verdict(scientific_verdict)
    scale_action = _normalize_scale_action(scale_action)
    if scale_action == ScaleAction.PROMOTE_SCALE:
        raise ValueError(
            "PROMOTE_SCALE complete_scale_experiment içinde doğrudan yazılamaz; "
            "register_promoted_experiment ile governance doğrulaması gerekir."
        )
    actual_gpu_hours = _require_finite_nonnegative(
        actual_gpu_hours, "actual_gpu_hours"
    )
    actual_cost_usd = _require_finite_nonnegative(
        actual_cost_usd, "actual_cost_usd"
    )
    if not isinstance(result, Mapping):
        raise TypeError("result mapping olmalıdır.")
    _require_nonempty("updated_belief", updated_belief)
    _require_nonempty("next_step", next_step)
    if not evidence_refs:
        raise ValueError("Scientific completion en az bir evidence ref gerektirir.")
    if any(not str(ref).strip() for ref in evidence_refs):
        raise ValueError("evidence_refs boş değer içeremez.")

    scope = EvidenceScope(record.evidence_scope)
    if (
        scientific_verdict != ScientificVerdict.INCONCLUSIVE
        and scope != EvidenceScope.MECHANISM_GENERAL
        and not revalidation_trigger.strip()
    ):
        raise ValueError(
            "Non-general ACCEPT/REJECT belief revalidation_trigger gerektirir."
        )

    legacy_status = {
        ScientificVerdict.ACCEPT: "PASSED",
        ScientificVerdict.REJECT: "FALSIFIED",
        ScientificVerdict.INCONCLUSIVE: "INCONCLUSIVE",
    }[scientific_verdict]

    completed = ExperimentRecord(
        experiment_id=record.experiment_id,
        hypothesis=record.hypothesis,
        falsification_criteria=record.falsification_criteria,
        setup=dict(record.setup),
        expectation=record.expectation,
        timestamp=record.timestamp,
        result=dict(result),
        status=legacy_status,
        surprise=surprise,
        updated_belief=updated_belief,
        next_step=next_step,
        cost_estimate_usd=record.cost_estimate_usd,
        cost_usd=actual_cost_usd,
        candidate_version=record.candidate_version,
        data_scale=record.data_scale,
        minimum_sufficient_scale=record.minimum_sufficient_scale,
        evidence_scope=record.evidence_scope,
        promotion_rule=record.promotion_rule,
        scale_action=scale_action.value,
        scale_parent_experiment_id=record.scale_parent_experiment_id,
        estimated_gpu_hours=record.estimated_gpu_hours,
        actual_gpu_hours=actual_gpu_hours,
        technical_status="COMPLETED",
        scientific_verdict=scientific_verdict.value,
        pre_result_contract_sha256=record.pre_result_contract_sha256,
        revalidation_trigger=revalidation_trigger.strip() or None,
        extra_fields={
            **dict(record.extra_fields),
            "completion_evidence_refs": list(evidence_refs),
        },
    )
    tracker.log(completed)
    return completed


def fail_scale_experiment(
    experiment_id: str,
    *,
    tracker: "ExperimentTracker",
    error: str,
    actual_gpu_hours: float,
    actual_cost_usd: float,
) -> ExperimentRecord:
    """Close a pre-registered run that failed technically without inventing a verdict."""
    _require_nonempty("experiment_id", experiment_id)
    _require_nonempty("error", error)
    actual_gpu_hours = _require_finite_nonnegative(
        actual_gpu_hours, "actual_gpu_hours"
    )
    actual_cost_usd = _require_finite_nonnegative(
        actual_cost_usd, "actual_cost_usd"
    )
    record = _find_registry_record(tracker, experiment_id)
    require_pre_result_contract_intact(record)
    if record.technical_status != "IN_PROGRESS":
        raise RuntimeError(
            f"Yalnız IN_PROGRESS deney ERROR olarak kapatılabilir; mevcut={record.technical_status}"
        )

    failed = replace(
        record,
        result={"error": error},
        status="ERROR",
        cost_usd=actual_cost_usd,
        actual_gpu_hours=actual_gpu_hours,
        technical_status="ERROR",
        scientific_verdict=None,
        scale_action=ScaleAction.STOP.value,
    )
    tracker.log(failed)
    return failed


def register_promoted_experiment(
    request: PromotionRequest,
    *,
    tracker: "ExperimentTracker",
    source_experiment_id: str,
    target_experiment_id: str,
    target_promotion_rule: str,
    large_data_plan: Mapping[str, Any],
    remaining_budget_usd: float,
) -> ExperimentRecord:
    """Validate promotion, mark the parent, and pre-register the child scale run."""
    from src.experiments.tracker import ExperimentTracker

    if not isinstance(tracker, ExperimentTracker):
        raise TypeError("tracker ExperimentTracker olmalıdır.")
    _require_nonempty("target_experiment_id", target_experiment_id)
    if any(r.experiment_id == target_experiment_id for r in tracker.load_all(strict=True)):
        raise RuntimeError(f"target experiment_id zaten mevcut: {target_experiment_id}")

    validate_promotion_request(
        request,
        tracker=tracker,
        source_experiment_id=source_experiment_id,
        large_data_plan=large_data_plan,
        remaining_budget_usd=remaining_budget_usd,
    )
    source = _find_registry_record(tracker, source_experiment_id)
    scales = ordered_research_scales(large_data_plan)
    target_has_larger_scale = _next_scale(scales, request.to_scale) is not None
    if target_has_larger_scale and not target_promotion_rule.strip():
        raise ValueError(
            "Promoted child için bir sonraki scale promotion rule sonuçtan önce yazılmalıdır."
        )
    if not target_has_larger_scale:
        target_promotion_rule = ""

    target_hash = _stage_sample_hash(large_data_plan, request.to_scale)
    _require_large_data_identity(large_data_plan, request.to_scale)

    source_extra = dict(source.extra_fields)
    source_extra.update(
        {
            "promotion_to_scale": request.to_scale,
            "promotion_evidence_refs": list(request.evidence_refs),
            "promotion_estimated_gpu_hours": request.estimated_gpu_hours,
            "promotion_estimated_cost_usd": request.estimated_cost_usd,
            "promotion_scale_sensitive_ambiguity": request.scale_sensitive_ambiguity,
            "promotion_larger_scale_rationale": request.why_larger_scale_resolves_ambiguity,
            "promotion_skip_scale_justification": request.skip_scale_justification,
        }
    )
    updated_source = replace(
        source,
        scale_action=ScaleAction.PROMOTE_SCALE.value,
        extra_fields=source_extra,
    )

    child_setup = dict(source.setup)
    child_setup.update(
        {
            "question_id": request.question_id,
            "data_scale": request.to_scale,
            "promotion_from_scale": request.from_scale,
            "promotion_from_experiment_id": source_experiment_id,
            "promotion_evidence_refs": list(request.evidence_refs),
        }
    )
    if target_hash:
        child_setup["train_subset_sha256"] = target_hash
    if large_data_plan.get("dataset_revision"):
        child_setup["dataset_revision"] = large_data_plan["dataset_revision"]
    if large_data_plan.get("split_map_sha256"):
        child_setup["split_map_sha256"] = large_data_plan["split_map_sha256"]
    if large_data_plan.get("plan_sha256"):
        child_setup["large_data_plan_sha256"] = large_data_plan["plan_sha256"]

    child = ExperimentRecord(
        experiment_id=target_experiment_id,
        hypothesis=source.hypothesis,
        falsification_criteria=source.falsification_criteria,
        setup=child_setup,
        expectation=source.expectation,
        status="IN_PROGRESS",
        candidate_version=source.candidate_version,
        data_scale=request.to_scale,
        minimum_sufficient_scale=source.minimum_sufficient_scale,
        evidence_scope=source.evidence_scope,
        promotion_rule=target_promotion_rule or None,
        scale_action=ScaleAction.STOP.value,
        scale_parent_experiment_id=source_experiment_id,
        estimated_gpu_hours=request.estimated_gpu_hours,
        cost_estimate_usd=request.estimated_cost_usd,
        technical_status="IN_PROGRESS",
    )
    child = seal_pre_result_contract(child)
    tracker.log_many([updated_source, child])
    return child


def _next_scale(scales: Sequence[str], current: str) -> Optional[str]:
    if current not in scales:
        raise ValueError(f"Scale plan içinde yok: {current}")
    idx = scales.index(current)
    return scales[idx + 1] if idx + 1 < len(scales) else None


def validate_promotion_request(
    request: PromotionRequest,
    *,
    tracker: "ExperimentTracker",
    source_experiment_id: str,
    large_data_plan: Mapping[str, Any],
    remaining_budget_usd: float,
) -> None:
    """Fail-closed validation for spending more data/compute on the same question.

    Promotion policy is bound to the source registry record so the rule cannot
    be invented after seeing the result.
    """
    for label, value in (
        ("question_id", request.question_id),
        ("candidate_version", request.candidate_version),
        ("from_scale", request.from_scale),
        ("to_scale", request.to_scale),
        ("promotion_rule", request.promotion_rule),
    ):
        _require_nonempty(label, value)
    scientific_verdict = _normalize_verdict(request.scientific_verdict)
    action = _normalize_scale_action(request.action)

    source_experiment = _find_registry_record(tracker, source_experiment_id)
    require_pre_result_contract_intact(source_experiment)
    source = source_experiment.to_dict()
    source_setup = source.get("setup") or {}
    if source_setup.get("question_id") != request.question_id:
        raise ValueError("Promotion source experiment farklı research question'a ait.")
    if source.get("candidate_version") != request.candidate_version:
        raise ValueError("Promotion source experiment farklı candidate_version'a ait.")
    if source.get("data_scale") != request.from_scale:
        raise ValueError("Promotion from_scale source experiment ile uyuşmuyor.")

    _require_large_data_identity(large_data_plan, request.from_scale)
    _require_large_data_identity(large_data_plan, request.to_scale)
    expected_dataset_revision = _require_hex(
        large_data_plan.get("dataset_revision"), 40, "dataset_revision"
    )
    expected_split_hash = _require_hex(
        large_data_plan.get("split_map_sha256"), 64, "split_map_sha256"
    )
    expected_source_stage_hash = _stage_sample_hash(
        large_data_plan, request.from_scale
    )
    _require_execution_provenance(source_setup)
    expected_plan_hash = _require_hex(
        large_data_plan.get("plan_sha256"), 64, "plan_sha256"
    )
    if source_setup.get("large_data_plan_sha256") != expected_plan_hash:
        raise RuntimeError("Promotion source farklı large-data plan'a ait.")
    if source_setup.get("dataset_revision") != expected_dataset_revision:
        raise RuntimeError("Promotion source farklı HF dataset revision'a ait.")
    if source_setup.get("split_map_sha256") != expected_split_hash:
        raise RuntimeError("Promotion source farklı split map'e ait.")
    if (
        expected_source_stage_hash is not None
        and source_setup.get("train_subset_sha256") != expected_source_stage_hash
    ):
        raise RuntimeError("Promotion source farklı source-stage sample setine ait.")

    if source.get("technical_status") != "COMPLETED":
        raise RuntimeError("Tamamlanmamış deney scale promotion kaynağı olamaz.")
    if source.get("scale_action") not in {None, ScaleAction.STOP.value}:
        raise RuntimeError(
            f"Source experiment zaten terminal scale action taşıyor: {source.get('scale_action')}"
        )
    if source.get("scientific_verdict") != scientific_verdict.value:
        raise RuntimeError(
            "Promotion request scientific verdict source registry kaydıyla uyuşmuyor."
        )
    predeclared_rule = str(source.get("promotion_rule") or "").strip()
    if not predeclared_rule:
        raise RuntimeError("Source experiment önceden promotion_rule kaydetmemiş.")
    if predeclared_rule != request.promotion_rule.strip():
        raise RuntimeError("Promotion rule sonuç görüldükten sonra değiştirilemez.")

    _require_nonnegative_cost(request.estimated_gpu_hours, request.estimated_cost_usd)
    remaining_budget_usd = _require_finite_nonnegative(
        remaining_budget_usd, "remaining_budget_usd"
    )
    if request.estimated_cost_usd > remaining_budget_usd:
        raise RuntimeError(
            f"Promotion bütçeyi aşıyor: estimate={request.estimated_cost_usd:.2f} USD, "
            f"remaining={remaining_budget_usd:.2f} USD"
        )
    if action != ScaleAction.PROMOTE_SCALE:
        raise ValueError("validate_promotion_request yalnız PROMOTE_SCALE için kullanılır.")
    if not request.evidence_refs:
        raise ValueError("Scale promotion en az bir kanıt referansı gerektirir.")

    scales = ordered_research_scales(large_data_plan)
    if request.from_scale not in scales or request.to_scale not in scales:
        raise ValueError(
            f"Promotion scale plan dışında: {request.from_scale}->{request.to_scale}; available={scales}"
        )
    if scales.index(request.to_scale) <= scales.index(request.from_scale):
        raise ValueError("Promotion yalnız daha büyük scale'e olabilir.")

    immediate = _next_scale(scales, request.from_scale)
    if request.to_scale != immediate and not request.skip_scale_justification.strip():
        raise ValueError(
            f"Scale atlanıyor ({request.from_scale}->{request.to_scale}); "
            "skip_scale_justification zorunludur."
        )

    if scientific_verdict == ScientificVerdict.REJECT:
        raise RuntimeError(
            "REJECT edilen aynı hipotez daha pahalı scale'e promote edilemez; "
            "yeni mekanizma/hipotez gerekiyorsa yeni araştırma sorusu aç."
        )

    if scientific_verdict == ScientificVerdict.ACCEPT:
        if not request.promotion_rule_met:
            raise RuntimeError(
                "ACCEPT sonucu daha pahalı scale'e ancak önceden yazılmış promotion "
                "kuralı karşılandıysa taşınabilir."
            )
        return

    # INCONCLUSIVE is not an automatic excuse to spend more. It is allowed only
    # when scale itself is the identified source of ambiguity.
    if scientific_verdict == ScientificVerdict.INCONCLUSIVE:
        if not request.scale_sensitive_ambiguity:
            raise RuntimeError(
                "INCONCLUSIVE otomatik scale promotion gerekçesi değildir."
            )
        if not request.why_larger_scale_resolves_ambiguity.strip():
            raise RuntimeError(
                "INCONCLUSIVE promotion için daha büyük scale'in belirsizliği neden "
                "çözeceği açıkça yazılmalıdır."
            )
        return

    raise ValueError(f"Desteklenmeyen scientific verdict: {scientific_verdict}")


def _extract_metric(result: Mapping[str, Any], metric: str) -> Optional[float]:
    direct = result.get(metric)
    if isinstance(direct, (int, float)):
        return float(direct)
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        nested = metrics.get(metric)
        if isinstance(nested, (int, float)):
            return float(nested)
    return None


def build_scaling_curve(
    records: Iterable[ExperimentRecord],
    *,
    question_id: str,
    candidate_version: str,
    dataset_revision: str,
    metric: str,
    large_data_plan: Mapping[str, Any],
) -> Tuple[ScalingPoint, ...]:
    """Build an ordered evidence curve without inferring causality.

    Interpretation belongs to the research engine together with raw predictions,
    subgroup failures, literature, and prior decisions.
    """
    scales = ordered_research_scales(large_data_plan)
    points: List[ScalingPoint] = []
    for record in records:
        data = record.to_dict()
        setup = data.get("setup") or {}
        if setup.get("question_id") != question_id:
            continue
        if data.get("candidate_version") != candidate_version:
            continue
        if setup.get("dataset_revision") != dataset_revision:
            continue
        if data.get("technical_status") != "COMPLETED":
            continue
        scale = data.get("data_scale")
        if scale not in scales:
            continue
        result = data.get("result")
        if not isinstance(result, Mapping):
            continue
        value = _extract_metric(result, metric)
        if value is None:
            continue
        points.append(
            ScalingPoint(
                experiment_id=record.experiment_id,
                scale=scale,
                metric=metric,
                value=value,
                evidence_scope=data.get("evidence_scope"),
            )
        )

    points.sort(key=lambda p: scales.index(p.scale))
    return tuple(points)
