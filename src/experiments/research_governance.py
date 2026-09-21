"""
src/experiments/research_governance.py — Autonomous Research Governance & Full-Training Gates

Implements strict programmatic enforcement of GEMINI.md Section 5.1 readiness gates,
candidate lifecycle state transitions, tamper-evident full-train manifest creation,
and pre-flight authorization verification.
"""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import yaml

REQUIRED_READINESS_KEYS: Tuple[str, ...] = (
    "complete_recipe",
    "component_decisions",
    "no_high_impact_uncertainty",
    "controlled_probes",
    "representative_validation",
    "stability",
    "metrics_and_raw_outputs",
    "failure_model",
    "provenance_and_reproducibility",
    "dress_rehearsal",
    "baseline_improvement",
    "frozen_full_train_recipe",
)


class ResearchStage(str, Enum):
    RESEARCHING = "RESEARCHING"
    CONFIRMING = "CONFIRMING"
    REHEARSAL = "REHEARSAL"
    READY_FOR_FULL_TRAIN = "READY_FOR_FULL_TRAIN"
    FULL_TRAINING = "FULL_TRAINING"
    EVALUATING = "EVALUATING"


VALID_STAGE_TRANSITIONS: Dict[ResearchStage, Set[ResearchStage]] = {
    ResearchStage.RESEARCHING: {ResearchStage.CONFIRMING, ResearchStage.RESEARCHING},
    ResearchStage.CONFIRMING: {ResearchStage.RESEARCHING, ResearchStage.REHEARSAL, ResearchStage.CONFIRMING},
    ResearchStage.REHEARSAL: {ResearchStage.CONFIRMING, ResearchStage.READY_FOR_FULL_TRAIN, ResearchStage.REHEARSAL},
    ResearchStage.READY_FOR_FULL_TRAIN: {ResearchStage.FULL_TRAINING, ResearchStage.REHEARSAL, ResearchStage.CONFIRMING},
    ResearchStage.FULL_TRAINING: {ResearchStage.EVALUATING, ResearchStage.CONFIRMING},
    ResearchStage.EVALUATING: {ResearchStage.RESEARCHING},
}


@dataclass
class CandidateState:
    candidate_version: str
    stage: ResearchStage
    raw: Dict[str, Any]
    path: pathlib.Path
    open_questions: List[Dict[str, Any]]
    readiness: Dict[str, Any]
    components: Dict[str, Any]
    training: Dict[str, Any]


@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    blockers: Tuple[str, ...]
    gate_details: Dict[str, Dict[str, Any]]


def load_candidate(candidate_path: Union[str, pathlib.Path]) -> CandidateState:
    path = pathlib.Path(candidate_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Candidate YAML dosyası bulunamadı: {path}")

    content = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(content) or {}

    version = raw.get("candidate_version", "c0.0.0")
    stage_str = raw.get("stage", ResearchStage.RESEARCHING.value)
    try:
        stage = ResearchStage(stage_str)
    except ValueError:
        raise ValueError(f"Bilinmeyen araştırma aşaması: {stage_str}")

    open_questions = raw.get("open_questions", [])
    readiness = raw.get("readiness", {})
    components = raw.get("components", {})
    training = raw.get("training", {})

    return CandidateState(
        candidate_version=version,
        stage=stage,
        raw=raw,
        path=path,
        open_questions=open_questions,
        readiness=readiness,
        components=components,
        training=training,
    )


def evaluate_readiness(
    candidate: CandidateState,
    root_dir: Optional[pathlib.Path] = None,
) -> ReadinessReport:
    """
    GEMINI.md Bölüm 5.1'deki 12 kapıyı ve aday durumunu programatik olarak denetler.
    Fail-closed: Eksik, belirsiz veya 'in_progress' olan herhangi bir unsur blockers listesine eklenir.
    """
    blockers: List[str] = []
    gate_details: Dict[str, Dict[str, Any]] = {}

    # 1. Aşama kontrolü
    if candidate.stage not in {ResearchStage.REHEARSAL, ResearchStage.READY_FOR_FULL_TRAIN}:
        blockers.append("invalid_stage")

    # 2. Açık yüksek etkili sorular kontrolü
    open_high_impact = [
        q["id"]
        for q in candidate.open_questions
        if q.get("impact") == "high" and q.get("status") in {"active", "open", "queued", "in_progress"}
    ]
    if open_high_impact:
        blockers.append("open_high_impact_questions")

    # 3. Reçete durumu kontrolü (recipe_status: in_progress çelişkisi)
    recipe_status = candidate.training.get("recipe_status", "in_progress")
    if recipe_status == "in_progress":
        blockers.append("recipe_status_in_progress")

    # 4. 12 Kapı denetimi
    if root_dir is not None:
        base_dir = root_dir
    elif (candidate.path.parent / "research").exists():
        base_dir = candidate.path.parent
    else:
        base_dir = candidate.path.parent.parent

    for key in REQUIRED_READINESS_KEYS:
        gate_info = candidate.readiness.get(key)
        if not gate_info:
            blockers.append(f"missing_gate:{key}")
            gate_details[key] = {"status": "FAILED", "reason": "Gate definition missing"}
            continue

        passed = gate_info.get("passed", False)
        evidence = gate_info.get("evidence", [])

        if not passed:
            blockers.append(f"gate_failed:{key}")
            gate_details[key] = {"status": "FAILED", "reason": "Gate passed is False", "evidence": evidence}
            continue

        if not evidence:
            blockers.append(f"gate_missing_evidence:{key}")
            gate_details[key] = {"status": "FAILED", "reason": "Evidence list is empty"}
            continue

        # Kanıt artefaktlarının varlığını kontrol et
        missing_evidence = []
        for ev in evidence:
            ev_str = str(ev)
            # Eğer dosya yolu formatındaysa dosyanın varlığını teyit et
            if any(ev_str.startswith(prefix) for prefix in ("configs/", "research/", "data/", "experiments/")):
                # fragment (#D1 vb.) varsa ayıkla
                file_part = ev_str.split("#")[0]
                target_file = base_dir / file_part
                if not target_file.exists():
                    missing_evidence.append(ev_str)

        if missing_evidence:
            blockers.append(f"gate_evidence_not_found:{key}")
            gate_details[key] = {
                "status": "FAILED",
                "reason": f"Evidence file not found: {missing_evidence}",
                "evidence": evidence,
            }
            continue

        gate_details[key] = {"status": "PASSED", "evidence": evidence}

    return ReadinessReport(
        ready=len(blockers) == 0,
        blockers=tuple(blockers),
        gate_details=gate_details,
    )


def compute_file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_full_train_manifest(
    candidate: CandidateState,
    code_revision: str,
    dataset_id: str,
    dataset_sha256: str,
    split_sha256: str,
    initializer: str,
    seeds: Sequence[int],
    budget_usd: float,
    dataset_scope_note: str,
    root_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Dondurulmuş model ve doğrulanmış hazırlık kanıtları için tam eğitim manifestosu oluşturur."""
    if candidate.stage != ResearchStage.READY_FOR_FULL_TRAIN:
        raise RuntimeError(
            f"Manifest yalnız READY_FOR_FULL_TRAIN aşamasında üretilebilir. Mevcut aşama: {candidate.stage.value}"
        )

    report = evaluate_readiness(candidate, root_dir=root_dir)
    if not report.ready:
        raise RuntimeError(f"Tam eğitim readiness kapıları geçilmedi: {report.blockers}")

    if not seeds or len(seeds) < 3:
        raise ValueError(f"Full training manifestosu en az 3 kararlı seed gerektirir. Verilen: {seeds}")

    if budget_usd <= 0:
        raise ValueError(f"Geçersiz bulut bütçesi: {budget_usd} USD")

    if not code_revision or code_revision in {"unknown", "dirty"}:
        raise ValueError(f"Full training manifestosu temiz bir git revizyonu gerektirir. Verilen: {code_revision}")

    if not dataset_sha256 or not split_sha256:
        raise ValueError("Dataset ve split hash değerleri boş olamaz.")

    recipe_bytes = candidate.path.read_bytes()
    candidate_recipe_sha256 = hashlib.sha256(recipe_bytes).hexdigest()

    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "candidate_version": candidate.candidate_version,
        "candidate_recipe_path": str(candidate.path),
        "candidate_recipe_sha256": candidate_recipe_sha256,
        "code_revision": code_revision,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "initializer": initializer,
        "seeds": list(seeds),
        "budget_usd": budget_usd,
        "dataset_scope_note": dataset_scope_note,
        "hyperparameters": candidate.training.get("hyperparameters", {}),
        "decoding_config": candidate.components.get("decoder", {}),
        "keyword_spotting_config": candidate.components.get("keyword_spotting", {}),
    }

    return manifest


def build_full_train_manifest_from_path(
    candidate_path: Union[str, pathlib.Path],
    code_revision: str,
    dataset_id: str,
    dataset_sha256: str,
    split_sha256: str,
    initializer: str,
    seeds: Sequence[int],
    budget_usd: float,
    dataset_scope_note: str,
    root_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    state = load_candidate(candidate_path)
    return build_full_train_manifest(
        candidate=state,
        code_revision=code_revision,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        initializer=initializer,
        seeds=seeds,
        budget_usd=budget_usd,
        dataset_scope_note=dataset_scope_note,
        root_dir=root_dir,
    )


def write_full_train_manifest(path: Union[str, pathlib.Path], manifest: Dict[str, Any]) -> pathlib.Path:
    p = pathlib.Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(p)
    return p


def require_full_train_authorized(
    candidate_path: Union[str, pathlib.Path],
    manifest_path: Union[str, pathlib.Path],
    root_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """
    Eğitim öncesi yetkilendirme kontrolü.
    Manifesto ve candidate reçetesi eşleşmezse veya readiness eksikse derhal RuntimeError fırlatır.
    """
    c_path = pathlib.Path(candidate_path).resolve()
    m_path = pathlib.Path(manifest_path).resolve()

    if not c_path.is_file():
        raise FileNotFoundError(f"Candidate YAML bulunamadı: {c_path}")
    if not m_path.is_file():
        raise FileNotFoundError(f"Full training manifestosu bulunamadı: {m_path}")

    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    candidate_state = load_candidate(c_path)

    # 1. Aşama kontrolü
    if candidate_state.stage != ResearchStage.READY_FOR_FULL_TRAIN:
        raise RuntimeError(
            f"Candidate aşaması {candidate_state.stage.value}; full training yalnız READY_FOR_FULL_TRAIN aşamasında yetkilidir."
        )

    # 2. Hash bütünlüğü kontrolü
    current_recipe_hash = compute_file_sha256(c_path)
    manifest_recipe_hash = manifest.get("candidate_recipe_sha256")
    if current_recipe_hash != manifest_recipe_hash:
        raise RuntimeError(
            f"candidate recipe hash mismatch (tampered)! Current file hash: {current_recipe_hash}, "
            f"Manifest hash: {manifest_recipe_hash}"
        )

    # 3. Sürüm eşleşmesi
    if candidate_state.candidate_version != manifest.get("candidate_version"):
        raise RuntimeError(
            f"Sürüm uyumsuzluğu: Candidate={candidate_state.candidate_version}, "
            f"Manifest={manifest.get('candidate_version')}"
        )

    # 4. Readiness kapıları kontrolü
    report = evaluate_readiness(candidate_state, root_dir=root_dir)
    if not report.ready:
        raise RuntimeError(f"Readiness kapıları geçilmedi: {report.blockers}")

    return manifest
