"""Cost-aware experiment scaling for large-data autonomous VSR research.

This module constrains experiment *cost and scale*, never the scientific search
space. Architecture/objective/frontend/decoder choices remain governed by the
research question and evidence in GEMINI.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
            return float(stage[:-1])
        except ValueError:
            pass
    raise ValueError(f"Bilinmeyen araştırma ölçeği: {stage}")


def ordered_research_scales(plan: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return smoke + actual deterministic stages from a verified data plan."""
    stages = list((plan.get("staged_subsets") or {}).keys())
    numeric = sorted((s for s in stages if s.endswith("h")), key=_stage_hours)
    ordered: List[str] = [VIRTUAL_SMOKE_SCALE, *numeric]
    if "full" in stages:
        ordered.append("full")
    return tuple(ordered)


def load_ordered_research_scales(plan_path: pathlib.Path) -> Tuple[str, ...]:
    return ordered_research_scales(load_large_data_plan(plan_path))


def _require_nonempty(label: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{label} boş olamaz.")


def _require_nonnegative_cost(gpu_hours: float, cost_usd: float) -> None:
    if gpu_hours < 0:
        raise ValueError("estimated_gpu_hours negatif olamaz.")
    if cost_usd < 0:
        raise ValueError("estimated_cost_usd negatif olamaz.")


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
        ("requested_scale", experiment.requested_scale),
        ("minimum_sufficient_scale", experiment.minimum_sufficient_scale),
    ):
        _require_nonempty(label, value)

    _require_nonnegative_cost(
        experiment.estimated_gpu_hours,
        experiment.estimated_cost_usd,
    )
    if remaining_budget_usd < 0:
        raise ValueError("remaining_budget_usd negatif olamaz.")
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

    requested_idx = scales.index(experiment.requested_scale)
    if requested_idx > 0 and not experiment.why_smaller_scale_is_insufficient.strip():
        raise ValueError(
            "Smoke/en küçük mevcut scale atlanıyorsa neden daha küçük scale'in "
            "hipotezi ayırt edemeyeceği açıkça yazılmalıdır."
        )

    # Any non-smoke run must declare the rule that would justify spending more.
    if experiment.requested_scale != VIRTUAL_SMOKE_SCALE and not experiment.promotion_rule.strip():
        raise ValueError(
            "Large-data araştırma koşusu promotion_rule olmadan başlatılamaz."
        )


def _next_scale(scales: Sequence[str], current: str) -> Optional[str]:
    if current not in scales:
        raise ValueError(f"Scale plan içinde yok: {current}")
    idx = scales.index(current)
    return scales[idx + 1] if idx + 1 < len(scales) else None


def validate_promotion_request(
    request: PromotionRequest,
    *,
    large_data_plan: Mapping[str, Any],
    remaining_budget_usd: float,
) -> None:
    """Fail-closed validation for spending more data/compute on the same question."""
    for label, value in (
        ("question_id", request.question_id),
        ("candidate_version", request.candidate_version),
        ("from_scale", request.from_scale),
        ("to_scale", request.to_scale),
        ("promotion_rule", request.promotion_rule),
    ):
        _require_nonempty(label, value)

    _require_nonnegative_cost(request.estimated_gpu_hours, request.estimated_cost_usd)
    if request.estimated_cost_usd > remaining_budget_usd:
        raise RuntimeError(
            f"Promotion bütçeyi aşıyor: estimate={request.estimated_cost_usd:.2f} USD, "
            f"remaining={remaining_budget_usd:.2f} USD"
        )
    if request.action != ScaleAction.PROMOTE_SCALE:
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

    if request.scientific_verdict == ScientificVerdict.REJECT:
        raise RuntimeError(
            "REJECT edilen aynı hipotez daha pahalı scale'e promote edilemez; "
            "yeni mekanizma/hipotez gerekiyorsa yeni araştırma sorusu aç."
        )

    if request.scientific_verdict == ScientificVerdict.ACCEPT:
        if not request.promotion_rule_met:
            raise RuntimeError(
                "ACCEPT sonucu daha pahalı scale'e ancak önceden yazılmış promotion "
                "kuralı karşılandıysa taşınabilir."
            )
        return

    # INCONCLUSIVE is not an automatic excuse to spend more. It is allowed only
    # when scale itself is the identified source of ambiguity.
    if request.scientific_verdict == ScientificVerdict.INCONCLUSIVE:
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

    raise ValueError(f"Desteklenmeyen scientific verdict: {request.scientific_verdict}")


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
