"""Validation for immutable research rounds and the mutable belief ledger."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

import yaml

from src.experiments.tracker import ExperimentRecord, ExperimentTracker


ALLOWED_EVIDENCE_SCOPES = {
    "mechanism_general",
    "small_data_regime",
    "large_data_regime",
    "dataset_revision_specific",
    "scale_specific",
}
ALLOWED_VERDICTS = {"ACCEPT", "REJECT", "INCONCLUSIVE"}


def load_yaml_mapping(path: pathlib.Path) -> Dict[str, Any]:
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"YAML mapping bekleniyordu: {path}")
    return dict(payload)


def discover_round_manifests(rounds_root: pathlib.Path) -> List[pathlib.Path]:
    root = pathlib.Path(rounds_root)
    if not root.exists():
        return []
    return sorted(root.glob("*/round.yaml"))


def _registry_ids(records: Iterable[ExperimentRecord]) -> Set[str]:
    return {record.experiment_id for record in records}


def validate_round_manifest(
    path: pathlib.Path,
    *,
    repo_root: pathlib.Path,
    records: Sequence[ExperimentRecord],
) -> Dict[str, Any]:
    payload = load_yaml_mapping(path)
    required = {
        "round_id",
        "date",
        "round_type",
        "question_id",
        "candidate_context",
        "evidence_scope",
        "question",
        "hypothesis",
        "falsification_criteria",
        "evidence_inputs",
        "analysis",
        "scientific_verdict",
        "updated_belief",
        "revalidation_trigger",
        "next_question",
    }
    missing = sorted(key for key in required if not payload.get(key))
    if missing:
        raise ValueError(f"Research round eksik alanlar: {missing}")

    scope = str(payload["evidence_scope"])
    if scope not in ALLOWED_EVIDENCE_SCOPES:
        raise ValueError(f"Bilinmeyen evidence_scope: {scope}")
    verdict = str(payload["scientific_verdict"])
    if verdict not in ALLOWED_VERDICTS:
        raise ValueError(f"Bilinmeyen scientific_verdict: {verdict}")
    if scope != "mechanism_general" and verdict in {"ACCEPT", "REJECT"}:
        if not str(payload.get("revalidation_trigger", "")).strip():
            raise ValueError("Scoped ACCEPT/REJECT round revalidation_trigger gerektirir")

    evidence = payload["evidence_inputs"]
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence_inputs mapping olmalıdır")
    experiment_ids = evidence.get("experiment_ids") or []
    if not isinstance(experiment_ids, list) or not experiment_ids:
        raise ValueError("Research round en az bir experiment evidence ref gerektirir")
    known_ids = _registry_ids(records)
    missing_ids = [eid for eid in experiment_ids if eid not in known_ids]
    if missing_ids:
        raise RuntimeError(f"Round registry'de olmayan experiment refs içeriyor: {missing_ids}")

    analysis = payload["analysis"]
    if not isinstance(analysis, Mapping):
        raise TypeError("analysis mapping olmalıdır")
    result_rel = analysis.get("result")
    if not result_rel:
        raise ValueError("analysis.result zorunludur")
    result_path = pathlib.Path(repo_root) / str(result_rel)
    if not result_path.is_file():
        raise FileNotFoundError(f"Round result bulunamadı: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("question_id") != payload["question_id"]:
        raise RuntimeError("Round/result question_id uyuşmuyor")
    if result.get("scientific_verdict") != verdict:
        raise RuntimeError("Round/result scientific_verdict uyuşmuyor")
    if result.get("evidence_scope") != scope:
        raise RuntimeError("Round/result evidence_scope uyuşmuyor")
    if analysis.get("test_split_used") is not False:
        raise RuntimeError("Research round test_split_used=false olarak açıkça kayıtlı olmalıdır")

    return payload


def validate_all_rounds(
    *,
    repo_root: pathlib.Path,
    rounds_root: pathlib.Path,
    registry_path: pathlib.Path,
) -> List[Dict[str, Any]]:
    records = ExperimentTracker(registry_path).load_all(strict=True)
    manifests = discover_round_manifests(rounds_root)
    rounds = [
        validate_round_manifest(path, repo_root=repo_root, records=records)
        for path in manifests
    ]
    ids = [str(item["round_id"]) for item in rounds]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate research round ids: {duplicates}")
    return rounds


def validate_belief_ledger(
    path: pathlib.Path,
    *,
    valid_round_ids: Set[str],
) -> Dict[str, Any]:
    payload = load_yaml_mapping(path)
    beliefs = payload.get("beliefs")
    if not isinstance(beliefs, list):
        raise TypeError("BELIEFS.yaml beliefs listesi içermelidir")

    ids: List[str] = []
    for belief in beliefs:
        if not isinstance(belief, Mapping):
            raise TypeError("Belief kaydı mapping olmalıdır")
        required = {"id", "statement", "status", "evidence_scope", "evidence", "revalidation_trigger"}
        missing = sorted(key for key in required if not belief.get(key))
        if missing:
            raise ValueError(f"Belief eksik alanlar: {missing}")
        belief_id = str(belief["id"])
        ids.append(belief_id)
        scope = str(belief["evidence_scope"])
        if scope not in ALLOWED_EVIDENCE_SCOPES:
            raise ValueError(f"Bilinmeyen belief evidence_scope: {scope}")
        evidence = belief["evidence"]
        if not isinstance(evidence, Mapping):
            raise TypeError("belief.evidence mapping olmalıdır")
        for round_id in evidence.get("round_ids") or []:
            if round_id not in valid_round_ids:
                raise RuntimeError(f"Belief bilinmeyen research round'a referans veriyor: {round_id}")
        if belief["status"] == "active" and scope != "mechanism_general":
            if not str(belief.get("revalidation_trigger", "")).strip():
                raise ValueError(f"Active scoped belief revalidation trigger gerektirir: {belief_id}")

    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate belief ids: {duplicates}")
    return payload
