"""Reproducible evidence-synthesis probes over the immutable experiment registry."""

from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.experiments.tracker import ExperimentRecord


SCALING_CHAIN_IDS = (
    "probe_train001_full_curriculum_unfreeze",
    "probe_curr002_sentence_scaling",
    "probe_confirm001_full_train_scaling",
)
FULL_VALIDATION_BASELINE_ID = "probe_lowdata001_full_val_eval"
LONG_CLIP_EXTENSION_ID = "probe_lowdata003_sequence_bucketing"


@dataclass(frozen=True)
class ScalingPoint:
    experiment_id: str
    train_samples: int
    val_samples: int
    max_duration: float | None
    best_val_loss: float
    cer: float
    blank_ratio: float | None
    cost_usd: float | None


def load_registry(path: pathlib.Path) -> List[ExperimentRecord]:
    from src.experiments.tracker import ExperimentTracker

    return ExperimentTracker(pathlib.Path(path)).load_all(strict=True)


def _find(records: Iterable[ExperimentRecord], experiment_id: str) -> ExperimentRecord:
    matches = [r for r in records if r.experiment_id == experiment_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one registry record for {experiment_id}, found {len(matches)}"
        )
    return matches[0]


def _number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"Numeric metric missing: {key}")
    return float(value)


def _point(record: ExperimentRecord) -> ScalingPoint:
    result = record.result or {}
    setup = record.setup or {}
    train_samples = result.get("train_samples_count", setup.get("train_sample_limit"))
    val_samples = result.get("val_samples_count", setup.get("val_sample_limit"))
    if not isinstance(train_samples, int) or not isinstance(val_samples, int):
        raise RuntimeError(f"Sample counts missing for {record.experiment_id}")

    cer = result.get("best_cer", result.get("cer"))
    if not isinstance(cer, (int, float)):
        raise RuntimeError(f"CER missing for {record.experiment_id}")

    blank = result.get("blank_ratio")
    blank_ratio = float(blank) if isinstance(blank, (int, float)) else None

    max_duration = setup.get("max_duration")
    if max_duration is not None:
        max_duration = float(max_duration)

    return ScalingPoint(
        experiment_id=record.experiment_id,
        train_samples=train_samples,
        val_samples=val_samples,
        max_duration=max_duration,
        best_val_loss=_number(result, "best_val_loss"),
        cer=float(cer),
        blank_ratio=blank_ratio,
        cost_usd=float(record.cost_usd) if record.cost_usd is not None else None,
    )


def _pct_change(start: float, end: float) -> float:
    if start == 0:
        raise ZeroDivisionError("Cannot compute percentage change from zero")
    return (end - start) / start * 100.0


def run_small_data_scaling_audit(
    records: Sequence[ExperimentRecord],
) -> Dict[str, Any]:
    """Test whether more same-regime clips improve recognition, not only CTC loss.

    The primary chain uses the canonical 60-clip validation subset and progressively
    expands training from <=3.5s to <=6s to <=8s. The secondary check compares the
    frozen c0.4 checkpoint and long-clip sequence-bucketing extension on the same
    385-clip validation set.
    """

    primary = [_point(_find(records, experiment_id)) for experiment_id in SCALING_CHAIN_IDS]
    expected_train = [360, 720, 1325]
    expected_val = [60, 60, 60]
    if [p.train_samples for p in primary] != expected_train:
        raise RuntimeError(
            f"Canonical scaling chain drifted: {[p.train_samples for p in primary]}"
        )
    if [p.val_samples for p in primary] != expected_val:
        raise RuntimeError(
            f"Primary validation set drifted: {[p.val_samples for p in primary]}"
        )

    start, end = primary[0], primary[-1]
    primary_summary = {
        "train_samples_start": start.train_samples,
        "train_samples_end": end.train_samples,
        "train_samples_multiplier": round(end.train_samples / start.train_samples, 6),
        "best_val_loss_start": start.best_val_loss,
        "best_val_loss_end": end.best_val_loss,
        "best_val_loss_absolute_change": round(end.best_val_loss - start.best_val_loss, 6),
        "best_val_loss_relative_change_pct": round(
            _pct_change(start.best_val_loss, end.best_val_loss), 6
        ),
        "cer_start": start.cer,
        "cer_end": end.cer,
        "cer_absolute_change": round(end.cer - start.cer, 6),
        "cer_percentage_point_change": round((end.cer - start.cer) * 100.0, 6),
        "loss_improved": end.best_val_loss < start.best_val_loss,
        "cer_improved": end.cer < start.cer,
        "loss_monotonic_across_chain": all(
            primary[i + 1].best_val_loss < primary[i].best_val_loss
            for i in range(len(primary) - 1)
        ),
        "cer_monotonic_across_chain": all(
            primary[i + 1].cer >= primary[i].cer for i in range(len(primary) - 1)
        ),
    }

    full_val = _point(_find(records, FULL_VALIDATION_BASELINE_ID))
    long_extension = _point(_find(records, LONG_CLIP_EXTENSION_ID))
    if full_val.val_samples != 385 or long_extension.val_samples != 385:
        raise RuntimeError(
            "Full-validation comparison no longer uses the expected 385-clip validation set"
        )

    secondary_summary = {
        "baseline_experiment_id": full_val.experiment_id,
        "extension_experiment_id": long_extension.experiment_id,
        "validation_samples": 385,
        "extension_train_samples": long_extension.train_samples,
        "best_val_loss_baseline": full_val.best_val_loss,
        "best_val_loss_extension": long_extension.best_val_loss,
        "best_val_loss_absolute_change": round(
            long_extension.best_val_loss - full_val.best_val_loss, 6
        ),
        "best_val_loss_relative_change_pct": round(
            _pct_change(full_val.best_val_loss, long_extension.best_val_loss), 6
        ),
        "cer_baseline": full_val.cer,
        "cer_extension": long_extension.cer,
        "cer_absolute_change": round(long_extension.cer - full_val.cer, 6),
        "cer_percentage_point_change": round(
            (long_extension.cer - full_val.cer) * 100.0, 6
        ),
        "loss_improved": long_extension.best_val_loss < full_val.best_val_loss,
        "cer_improved": long_extension.cer < full_val.cer,
    }

    falsified = (
        primary_summary["loss_improved"]
        and not primary_summary["cer_improved"]
        and secondary_summary["loss_improved"]
        and not secondary_summary["cer_improved"]
    )

    return {
        "audit_id": "META-001",
        "question_id": "META-001",
        "evidence_scope": "small_data_regime",
        "hypothesis": (
            "If additional same-regime clips are still the dominant missing ingredient, "
            "reductions in held-out CTC validation loss should co-occur with CER improvement."
        ),
        "falsification_criteria": (
            "Validation loss improves materially while CER fails to improve in both the "
            "canonical 60-clip scaling chain and the 385-clip long-sequence extension."
        ),
        "primary_chain": [p.__dict__ for p in primary],
        "primary_summary": primary_summary,
        "full_validation_extension": secondary_summary,
        "scientific_verdict": "REJECT" if falsified else "INCONCLUSIVE",
        "result": (
            "optimization_recognition_decoupling"
            if falsified
            else "insufficient_evidence"
        ),
        "updated_belief": (
            "Within the frozen c0.4 small-data regime, adding more clips from the same "
            "limited speaker/source regime can reduce CTC validation loss without improving "
            "transcription CER. More same-regime clip-count scaling is therefore not a strong "
            "next research direction; speaker/source diversity and sequence objective/model "
            "capacity should remain open in the large-data phase."
        ),
        "revalidation_trigger": (
            "Re-evaluate after the pinned large-data HF snapshot provides substantially "
            "more identity/source diversity, or after an objective/architecture change that "
            "materially changes the loss-to-CER relationship."
        ),
        "limitations": [
            "This is a retrospective synthesis of frozen experiments, not a new model training run.",
            "The primary chain has only three comparable scaling points.",
            "The long-clip extension changes both training coverage and sequence batching.",
            "The conclusion is scoped to the frozen small-data regime and must not be treated as global.",
        ],
        "next_question": "ARCH-LD-001",
    }


def write_audit(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
