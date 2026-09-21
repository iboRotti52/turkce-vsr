"""Executable workflow for governed large-data VSR research.

This CLI does not launch training. It turns the large-data governance primitives
into one auditable workflow for status/reconcile, register, complete, fail and
promote operations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
    require_pre_result_contract_intact,
)
from src.experiments.tracker import ExperimentRecord, ExperimentTracker


DEFAULT_PLAN = pathlib.Path("research/large_data_plan.json")
DEFAULT_REGISTRY = pathlib.Path("experiments/registry.jsonl")
DEFAULT_CANDIDATE_RECIPE = pathlib.Path("configs/research_candidate.yaml")
DEFAULT_BUDGET_USD = 25.0


@dataclass(frozen=True)
class WorkflowStatus:
    total_budget_usd: float
    spent_actual_usd: float
    reserved_in_progress_usd: float
    remaining_unreserved_usd: float
    active_experiment_id: Optional[str]
    active_question_id: Optional[str]
    active_scale: Optional[str]
    active_candidate_version: Optional[str]
    experiments_total: int


def _load_mapping(path: pathlib.Path) -> Dict[str, Any]:
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Spec/config bulunamadı: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"Mapping bekleniyordu: {path}")
    return dict(data)


def _sha256_file(path: pathlib.Path) -> str:
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Hashlenecek dosya bulunamadı: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: pathlib.Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"git komutu başarısız: {' '.join(args)}") from exc
    return proc.stdout.strip()


def derive_clean_code_revision(repo_root: pathlib.Path) -> str:
    repo_root = pathlib.Path(repo_root)
    status = _git(repo_root, "status", "--porcelain")
    if status:
        preview = "\n".join(status.splitlines()[:10])
        raise RuntimeError(
            "Bilimsel run dirty working tree ile register edilemez. "
            f"Önce commit/stash/temizle:\n{preview}"
        )
    revision = _git(repo_root, "rev-parse", "HEAD").lower()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise RuntimeError(f"Geçersiz git revision: {revision!r}")
    return revision


def _resolve_path(repo_root: pathlib.Path, raw: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(raw)
    return path if path.is_absolute() else pathlib.Path(repo_root) / path


def derive_execution_setup(
    *,
    repo_root: pathlib.Path,
    candidate_version: str,
    seed: int,
    initializer_id: str,
    initializer_path: Optional[str] = None,
    initializer_sha256: Optional[str] = None,
    candidate_recipe_path: str = str(DEFAULT_CANDIDATE_RECIPE),
    extra_setup: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed integer olmalıdır.")
    if not str(initializer_id).strip():
        raise ValueError("initializer_id boş olamaz.")

    recipe_path = _resolve_path(repo_root, candidate_recipe_path)
    recipe = _load_mapping(recipe_path)
    recipe_version = str(recipe.get("candidate_version", "")).strip()
    if recipe_version != str(candidate_version).strip():
        raise RuntimeError(
            "Experiment candidate_version ile candidate recipe uyuşmuyor: "
            f"spec={candidate_version!r}, recipe={recipe_version!r}"
        )

    resolved_initializer_hash: Optional[str] = None
    if initializer_path:
        resolved_initializer_hash = _sha256_file(
            _resolve_path(repo_root, initializer_path)
        )
    if initializer_sha256:
        normalized = str(initializer_sha256).strip().lower()
        if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("initializer_sha256 geçerli 64-hex olmalıdır.")
        if resolved_initializer_hash and resolved_initializer_hash != normalized:
            raise RuntimeError(
                "initializer_path içeriği ile verilen initializer_sha256 uyuşmuyor."
            )
        resolved_initializer_hash = normalized
    if not resolved_initializer_hash:
        raise ValueError("initializer_path veya initializer_sha256 zorunludur.")

    setup = dict(extra_setup or {})
    protected = {
        "code_revision",
        "candidate_recipe_sha256",
        "seed",
        "initializer_id",
        "initializer_sha256",
    }
    collisions = protected.intersection(setup)
    if collisions:
        raise ValueError(
            "extra_setup governance provenance alanlarını override edemez: "
            f"{sorted(collisions)}"
        )

    setup.update(
        {
            "code_revision": derive_clean_code_revision(repo_root),
            "candidate_recipe_sha256": _sha256_file(recipe_path),
            "seed": seed,
            "initializer_id": str(initializer_id).strip(),
            "initializer_sha256": resolved_initializer_hash,
        }
    )
    return setup


def reconcile_workflow(
    tracker: ExperimentTracker,
    *,
    total_budget_usd: float,
) -> WorkflowStatus:
    if total_budget_usd < 0:
        raise ValueError("total_budget_usd negatif olamaz.")

    records = tracker.load_all(strict=True)
    ids = [record.experiment_id for record in records]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    if duplicate_ids:
        raise RuntimeError(f"Registry duplicate experiment_id içeriyor: {duplicate_ids}")

    active = [r for r in records if r.technical_status == "IN_PROGRESS"]
    if len(active) > 1:
        raise RuntimeError(
            "Tek aktif large-data experiment kuralı ihlal edildi: "
            f"{[r.experiment_id for r in active]}"
        )
    for record in active:
        require_pre_result_contract_intact(record)

    spent_actual = sum(
        float(record.cost_usd or 0.0)
        for record in records
        if record.technical_status in {"COMPLETED", "ERROR"}
    )
    reserved = sum(float(record.cost_estimate_usd or 0.0) for record in active)
    remaining = total_budget_usd - spent_actual - reserved
    if remaining < -1e-9:
        raise RuntimeError(
            "Registry budget aşımı gösteriyor: "
            f"budget={total_budget_usd:.2f}, spent={spent_actual:.2f}, reserved={reserved:.2f}"
        )
    remaining = max(0.0, remaining)

    current = active[0] if active else None
    setup = current.setup if current else {}
    return WorkflowStatus(
        total_budget_usd=float(total_budget_usd),
        spent_actual_usd=round(spent_actual, 6),
        reserved_in_progress_usd=round(reserved, 6),
        remaining_unreserved_usd=round(remaining, 6),
        active_experiment_id=current.experiment_id if current else None,
        active_question_id=setup.get("question_id") if current else None,
        active_scale=current.data_scale if current else None,
        active_candidate_version=current.candidate_version if current else None,
        experiments_total=len(records),
    )


def _tracker(path: pathlib.Path) -> ExperimentTracker:
    return ExperimentTracker(pathlib.Path(path))


def _require_no_active(status: WorkflowStatus) -> None:
    if status.active_experiment_id:
        raise RuntimeError(
            "Yeni experiment register edilemez; aktif experiment var: "
            f"{status.active_experiment_id}"
        )


def register_from_spec(
    spec: Mapping[str, Any],
    *,
    repo_root: pathlib.Path,
    plan_path: pathlib.Path,
    registry_path: pathlib.Path,
    total_budget_usd: float,
) -> ExperimentRecord:
    plan = load_large_data_plan(plan_path)
    tracker = _tracker(registry_path)
    status = reconcile_workflow(tracker, total_budget_usd=total_budget_usd)
    _require_no_active(status)

    candidate_version = str(spec["candidate_version"])
    experiment = ScaleExperimentPlan(
        question_id=str(spec["question_id"]),
        candidate_version=candidate_version,
        hypothesis=str(spec["hypothesis"]),
        requested_scale=str(spec["requested_scale"]),
        minimum_sufficient_scale=str(
            spec.get("minimum_sufficient_scale", spec["requested_scale"])
        ),
        information_gain_rationale=str(spec["information_gain_rationale"]),
        why_smaller_scale_is_insufficient=str(
            spec.get("why_smaller_scale_is_insufficient", "")
        ),
        promotion_rule=str(spec.get("promotion_rule", "")),
        falsification_criteria=str(spec["falsification_criteria"]),
        expectation=str(spec["expectation"]),
        evidence_scope=str(
            spec.get("evidence_scope", EvidenceScope.LARGE_DATA_REGIME.value)
        ),
        estimated_gpu_hours=float(spec["estimated_gpu_hours"]),
        estimated_cost_usd=float(spec["estimated_cost_usd"]),
    )

    execution = dict(spec.get("execution") or {})
    setup = derive_execution_setup(
        repo_root=repo_root,
        candidate_version=candidate_version,
        seed=execution["seed"],
        initializer_id=str(execution["initializer_id"]),
        initializer_path=execution.get("initializer_path"),
        initializer_sha256=execution.get("initializer_sha256"),
        candidate_recipe_path=str(
            execution.get("candidate_recipe_path", DEFAULT_CANDIDATE_RECIPE)
        ),
        extra_setup=spec.get("setup"),
    )

    return register_scale_experiment(
        experiment,
        tracker=tracker,
        experiment_id=str(spec["experiment_id"]),
        large_data_plan=plan,
        remaining_budget_usd=status.remaining_unreserved_usd,
        setup=setup,
    )


def complete_from_spec(
    spec: Mapping[str, Any],
    *,
    registry_path: pathlib.Path,
    total_budget_usd: float,
) -> ExperimentRecord:
    tracker = _tracker(registry_path)
    reconcile_workflow(tracker, total_budget_usd=total_budget_usd)
    return complete_scale_experiment(
        str(spec["experiment_id"]),
        tracker=tracker,
        result=dict(spec["result"]),
        scientific_verdict=ScientificVerdict(str(spec["scientific_verdict"])),
        actual_gpu_hours=float(spec["actual_gpu_hours"]),
        actual_cost_usd=float(spec["actual_cost_usd"]),
        surprise=str(spec.get("surprise", "")),
        updated_belief=str(spec["updated_belief"]),
        next_step=str(spec["next_step"]),
        evidence_refs=tuple(str(x) for x in spec["evidence_refs"]),
        revalidation_trigger=str(spec.get("revalidation_trigger", "")),
        scale_action=ScaleAction(str(spec.get("scale_action", ScaleAction.STOP.value))),
    )


def fail_from_spec(
    spec: Mapping[str, Any],
    *,
    registry_path: pathlib.Path,
    total_budget_usd: float,
) -> ExperimentRecord:
    tracker = _tracker(registry_path)
    reconcile_workflow(tracker, total_budget_usd=total_budget_usd)
    return fail_scale_experiment(
        str(spec["experiment_id"]),
        tracker=tracker,
        error=str(spec["error"]),
        actual_gpu_hours=float(spec["actual_gpu_hours"]),
        actual_cost_usd=float(spec["actual_cost_usd"]),
    )


def promote_from_spec(
    spec: Mapping[str, Any],
    *,
    plan_path: pathlib.Path,
    registry_path: pathlib.Path,
    total_budget_usd: float,
) -> ExperimentRecord:
    plan = load_large_data_plan(plan_path)
    tracker = _tracker(registry_path)
    status = reconcile_workflow(tracker, total_budget_usd=total_budget_usd)
    _require_no_active(status)

    source_id = str(spec["source_experiment_id"])
    records = tracker.load_all(strict=True)
    source = next((r for r in records if r.experiment_id == source_id), None)
    if source is None:
        raise FileNotFoundError(f"Source experiment bulunamadı: {source_id}")

    request = PromotionRequest(
        question_id=str(source.setup["question_id"]),
        candidate_version=str(source.candidate_version),
        from_scale=str(source.data_scale),
        to_scale=str(spec["to_scale"]),
        scientific_verdict=ScientificVerdict(str(source.scientific_verdict)),
        action=ScaleAction.PROMOTE_SCALE,
        promotion_rule=str(source.promotion_rule or ""),
        promotion_rule_met=bool(spec["promotion_rule_met"]),
        evidence_refs=tuple(str(x) for x in spec["evidence_refs"]),
        estimated_gpu_hours=float(spec["estimated_gpu_hours"]),
        estimated_cost_usd=float(spec["estimated_cost_usd"]),
        scale_sensitive_ambiguity=bool(spec.get("scale_sensitive_ambiguity", False)),
        why_larger_scale_resolves_ambiguity=str(
            spec.get("why_larger_scale_resolves_ambiguity", "")
        ),
        skip_scale_justification=str(spec.get("skip_scale_justification", "")),
    )

    return register_promoted_experiment(
        request,
        tracker=tracker,
        source_experiment_id=source_id,
        target_experiment_id=str(spec["target_experiment_id"]),
        target_promotion_rule=str(spec.get("target_promotion_rule", "")),
        large_data_plan=plan,
        remaining_budget_usd=status.remaining_unreserved_usd,
    )


def _print_json(value: Any) -> None:
    if isinstance(value, WorkflowStatus):
        value = asdict(value)
    elif isinstance(value, ExperimentRecord):
        value = value.to_dict()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Governed large-data research workflow; does NOT launch training."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")
    for name in ("register", "complete", "fail", "promote"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--spec", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = pathlib.Path(args.repo_root).resolve()
    registry_path = _resolve_path(repo_root, args.registry)
    plan_path = _resolve_path(repo_root, args.plan)

    if args.command == "status":
        plan = load_large_data_plan(plan_path)
        status = reconcile_workflow(
            _tracker(registry_path),
            total_budget_usd=args.budget_usd,
        )
        payload = asdict(status)
        payload.update(
            {
                "dataset_id": plan.get("dataset_id"),
                "dataset_revision": plan.get("dataset_revision"),
                "available_scales": ordered_research_scales(plan),
            }
        )
        _print_json(payload)
        return 0

    spec = _load_mapping(_resolve_path(repo_root, args.spec))
    if args.command == "register":
        record = register_from_spec(
            spec,
            repo_root=repo_root,
            plan_path=plan_path,
            registry_path=registry_path,
            total_budget_usd=args.budget_usd,
        )
    elif args.command == "complete":
        record = complete_from_spec(
            spec,
            registry_path=registry_path,
            total_budget_usd=args.budget_usd,
        )
    elif args.command == "fail":
        record = fail_from_spec(
            spec,
            registry_path=registry_path,
            total_budget_usd=args.budget_usd,
        )
    elif args.command == "promote":
        record = promote_from_spec(
            spec,
            plan_path=plan_path,
            registry_path=registry_path,
            total_budget_usd=args.budget_usd,
        )
    else:
        raise AssertionError(args.command)

    _print_json(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
