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
import subprocess
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


# ---------------------------------------------------------------------------
# Fail-closed kod provenance ve full-training preflight (handoff hardening).
#
# Tarihsel ders: c0.4.0 manifestosundaki `code_revision` (63da203) HEAD iken
# deneyler dirty working tree ile yapılmıştı; rev tek başına kodu yeniden
# üretmez (bkz. full_train_manifest.PROVENANCE.md). Aşağıdaki yardımcılar
# bundan sonraki mühürleme/confirmation/full-training adımlarında dirty
# tree'yi fail-closed reddeder ve tarihsel manifestoların çalıştırılmasını
# açık gerekçeyle engeller.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeRevisionStatus:
    revision: str  # git yoksa "unknown"
    clean: bool
    dirty_files: Tuple[str, ...]


@dataclass(frozen=True)
class PreflightReport:
    passed: bool
    blockers: Tuple[str, ...]
    checks: Dict[str, Dict[str, Any]]


def get_code_revision_status(
    root_dir: Union[str, pathlib.Path],
) -> CodeRevisionStatus:
    """HEAD sha + working-tree temizliği. Git yoksa revision='unknown'."""
    root = pathlib.Path(root_dir)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return CodeRevisionStatus(revision="unknown", clean=False, dirty_files=())

    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "-uall"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return CodeRevisionStatus(revision=revision, clean=False, dirty_files=())

    dirty = tuple(
        sorted(line[3:] for line in porcelain.splitlines() if line.strip())
    )
    return CodeRevisionStatus(
        revision=revision, clean=len(dirty) == 0, dirty_files=dirty
    )


def require_clean_code_revision(root_dir: Union[str, pathlib.Path]) -> str:
    """Temiz HEAD sha döndürür; dirty/unknown tree'de fail-closed durur."""
    status = get_code_revision_status(root_dir)
    if status.revision in {"", "unknown"}:
        raise RuntimeError(
            f"Temiz git revizyonu belirlenemedi (root={root_dir}). "
            "Mühürleme/confirmation/full-training kirli veya git-dışı ağaçta yasaktır."
        )
    if not status.clean:
        preview = ", ".join(status.dirty_files[:8])
        suffix = "..." if len(status.dirty_files) > 8 else ""
        raise RuntimeError(
            f"Working tree kirli ({len(status.dirty_files)} dosya: {preview}{suffix}). "
            "Önce commit/stash ile temizleyin; dirty tree ile mühürleme yasaktır "
            "(tarihsel c0.4.0 provenance boşluğunun tekrarı olur)."
        )
    return status.revision


def _git_path_is_clean(root: pathlib.Path, rel_path: str) -> Optional[bool]:
    """Verilen tracked path'te uncommitted değişiklik yoksa True.

    Git çalışmazsa None (doğrulanamaz) döner; fail-closed yorumlanmalıdır.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", rel_path],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return len(out.strip()) == 0
    except Exception:
        return None


def _manifest_is_ancestor_of_head(root: pathlib.Path, revision: str) -> Optional[bool]:
    """Manifest revizyonu güncel HEAD'in atası mı? Git yoksa None."""
    try:
        subprocess.check_output(
            ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return None


def _csv_channel_set(csv_path: pathlib.Path) -> Set[str]:
    """Split CSV'sinin son sütunundaki (kanal/konuşmacı) değer kümesi."""
    import csv as _csv

    channels: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = _csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                channels.add(row[-1].strip())
    return channels


def resolve_candidate_recipe_path(
    manifest: Dict[str, Any],
    root_dir: Union[str, pathlib.Path],
) -> pathlib.Path:
    """Manifestteki mutlak yola GÜVENMEDEN repo-relative reçete yolunu çözer.

    Tarihsel manifestolar makineye-özgü mutlak yol içerir
    (bkz. full_train_manifest.PROVENANCE.md §3); bu alan yalnızca
    dosya-adı ipucu olarak kullanılır, asıl doğrulama sha256 ile yapılır.
    """
    root = pathlib.Path(root_dir)
    canonical = root / "configs" / "research_candidate.yaml"
    if canonical.is_file():
        return canonical
    fallback_name = pathlib.Path(str(manifest.get("candidate_recipe_path", ""))).name
    if fallback_name:
        for candidate in root.rglob(fallback_name):
            if candidate.is_file():
                return candidate
    return canonical


def preflight_full_train_manifest(
    manifest_path: Union[str, pathlib.Path],
    candidate_path: Optional[Union[str, pathlib.Path]] = None,
    root_dir: Optional[Union[str, pathlib.Path]] = None,
    code_state: Optional[CodeRevisionStatus] = None,
) -> PreflightReport:
    """Canonical full-training preflight: tamamı fail-closed.

    Doğrular: manifesto şeması, reçete yolu+hash, sürüm, readiness kapıları,
    dataset kimliği+hash, split hash, test-split karantinası, initializer
    varlığı, kod revizyonu temizliği, kapsamda test bulaşmaması.
    Ücretli hiçbir işlem başlatmaz; yalnızca rapor döndürür.
    """
    blockers: List[str] = []
    checks: Dict[str, Dict[str, Any]] = {}

    def record(name: str, ok: bool, detail: str = "", blocker: str = "") -> None:
        checks[name] = {"status": "PASSED" if ok else "FAILED", "detail": detail}
        if not ok:
            blockers.append(blocker or name)

    m_path = pathlib.Path(manifest_path).resolve()
    if root_dir is not None:
        root = pathlib.Path(root_dir).resolve()
    else:
        root = m_path.parent
    if candidate_path is not None:
        c_path = pathlib.Path(candidate_path).resolve()
    else:
        c_path = (root / "configs" / "research_candidate.yaml").resolve()

    # 1. Manifesto şeması
    if not m_path.is_file():
        record("manifest_schema", False, f"Manifesto bulunamadı: {m_path}",
               "manifest_not_found")
        return PreflightReport(passed=False, blockers=tuple(blockers), checks=checks)
    try:
        manifest = json.loads(m_path.read_text(encoding="utf-8"))
    except Exception as exc:
        record("manifest_schema", False, f"Manifesto JSON parse hatası: {exc}",
               "manifest_unparseable")
        return PreflightReport(passed=False, blockers=tuple(blockers), checks=checks)

    required_keys = (
        "manifest_version", "candidate_version", "candidate_recipe_sha256",
        "code_revision", "dataset_id", "dataset_sha256", "split_sha256",
        "initializer", "seeds", "budget_usd",
    )
    missing_keys = [k for k in required_keys if k not in manifest]
    record(
        "manifest_schema",
        manifest.get("manifest_version") == 1 and not missing_keys,
        f"manifest_version={manifest.get('manifest_version')}, eksik anahtarlar={missing_keys or 'yok'}",
        "manifest_schema",
    )

    # 2. Temel yetkilendirme (aşama + reçete hash + sürüm + readiness)
    try:
        require_full_train_authorized(c_path, m_path, root_dir=root)
        record("base_authorization", True, "stage/hash/version/readiness tamam")
    except Exception as exc:
        record("base_authorization", False, str(exc)[:300], "base_authorization")

    # 3. Reçete yolu çözümleme (mutlak yola güvenmeden)
    resolved = resolve_candidate_recipe_path(manifest, root)
    record(
        "recipe_path_resolution",
        resolved.is_file(),
        f"çözümlenen={resolved}",
        "recipe_path_resolution",
    )

    # 4. Dataset kimliği: manifesto, candidate'ın dondurduğu kaynakla aynı olmalı
    #    (c0.4.0 kanıtını sessizce büyük veriye uyarlamayı engeller).
    candidate_raw: Dict[str, Any] = {}
    try:
        candidate_raw = yaml.safe_load(c_path.read_text(encoding="utf-8")) or {}
        candidate_source = ((candidate_raw.get("data") or {}).get("source"))
        dataset_ok = (
            candidate_source is not None
            and manifest.get("dataset_id") == candidate_source
        )
        record(
            "dataset_identity",
            bool(dataset_ok),
            f"manifest={manifest.get('dataset_id')}, candidate.data.source={candidate_source}",
            "dataset_identity_mismatch",
        )
    except Exception as exc:
        record("dataset_identity", False, f"candidate okunamadı: {exc}",
               "dataset_identity_mismatch")

    # 5. Dataset/split hash'leri (tracked metadata CSV'lerinden canlı recompute)
    for label, rel, key in (
        ("dataset_hash", "data/metadata/train.csv", "dataset_sha256"),
        ("split_hash", "data/metadata/val.csv", "split_sha256"),
    ):
        csv_path = root / rel
        expected = manifest.get(key)
        if not csv_path.is_file() or not expected:
            record(label, False, f"{rel} yok veya manifesto hash'i boş",
                   f"{label}_unverifiable")
            continue
        actual = compute_file_sha256(csv_path)
        record(
            label,
            actual == expected,
            f"{rel} sha256={actual[:16]}... beklenen={str(expected)[:16]}...",
            f"{label}_mismatch",
        )

    # 6. Test-split karantinası: test.csv git'te untouched + hash candidate ile eşleşmeli,
    #    train/val kanalları test kanallarıyla kesişmemeli.
    test_csv = root / "data/metadata/test.csv"
    quarantine_ok = True
    quarantine_notes: List[str] = []
    if not test_csv.is_file():
        quarantine_ok = False
        quarantine_notes.append("test.csv yok")
    else:
        vcs_clean = _git_path_is_clean(root, "data/metadata/test.csv")
        if vcs_clean is False:
            quarantine_ok = False
            quarantine_notes.append("test.csv working tree'de değişmiş")
        elif vcs_clean is None:
            quarantine_ok = False
            quarantine_notes.append("git doğrulanamadı (fail-closed)")
        try:
            expected_test_hash = ((candidate_raw.get("data") or {}).get("splits") or {}).get("test", {}).get("hash")
            actual_test_hash = compute_file_sha256(test_csv)
            if expected_test_hash and actual_test_hash != expected_test_hash:
                quarantine_ok = False
                quarantine_notes.append("test.csv hash candidate ile eşleşmiyor")
        except Exception as exc:
            quarantine_ok = False
            quarantine_notes.append(f"test hash kontrolü hatası: {exc}")
        try:
            train_csv, val_csv = root / "data/metadata/train.csv", root / "data/metadata/val.csv"
            if train_csv.is_file() and val_csv.is_file():
                train_val_channels = _csv_channel_set(train_csv) | _csv_channel_set(val_csv)
                test_channels = _csv_channel_set(test_csv)
                overlap = train_val_channels & test_channels
                if overlap:
                    quarantine_ok = False
                    quarantine_notes.append(f"kanal sızıntısı: {sorted(overlap)[:5]}")
        except Exception as exc:
            quarantine_ok = False
            quarantine_notes.append(f"kanal kesişim kontrolü hatası: {exc}")
    record(
        "test_quarantine",
        quarantine_ok,
        "; ".join(quarantine_notes) if quarantine_notes else "test.csv untouched, hash eşleşti, kanal kesişimi yok",
        "test_quarantine",
    )

    # 7. Kod revizyonu: manifesto revizyonu == güncel clean HEAD olmalı.
    state = code_state or get_code_revision_status(root)
    manifest_rev = str(manifest.get("code_revision", ""))
    if state.revision in {"", "unknown"} or not state.clean:
        record(
            "code_revision",
            False,
            f"working tree temiz değil veya revizyon bilinmiyor (rev={state.revision}, kirli={len(state.dirty_files)} dosya)",
            "code_revision_not_clean",
        )
    elif state.revision != manifest_rev:
        ancestor = _manifest_is_ancestor_of_head(root, manifest_rev)
        if ancestor:
            detail = (
                f"manifest revizyonu ({manifest_rev[:8]}) güncel HEAD'in ({state.revision[:8]}) atası: "
                "tarihsel manifesto kanıt olarak korunur, çalıştırılamaz "
                "(bkz. full_train_manifest.PROVENANCE.md)"
            )
        elif ancestor is False:
            detail = (
                f"revizyon uyumsuzluğu: manifest={manifest_rev[:8]}, HEAD={state.revision[:8]} "
                "(dal/atalık ilişkisi yok)"
            )
        else:
            detail = f"revizyon uyumsuzluğu: manifest={manifest_rev[:8]}, HEAD={state.revision[:8]}"
        record("code_revision", False, detail, "code_revision_mismatch")
    else:
        record("code_revision", True, f"HEAD={state.revision[:12]} temiz ve manifesto ile eşleşti")

    # 8. Initializer varlığı (fail-closed; hash kaydı varsa doğrulanır).
    init_name = pathlib.Path(str(manifest.get("initializer", ""))).name
    init_candidates = []
    if init_name:
        for base in (root / "checkpoints", root):
            candidate_file = base / init_name
            if candidate_file.is_file():
                init_candidates.append(candidate_file)
    if not init_candidates:
        record(
            "initializer",
            False,
            f"'{init_name}' checkpoints/ veya repo kökünde bulunamadı; "
            "initializer olmadan full training başlatılamaz (HF Hub/Modal volume adreslenmeli)",
            "initializer_not_present",
        )
    else:
        recorded_hash = manifest.get("initializer_sha256")
        if recorded_hash:
            actual_hash = compute_file_sha256(init_candidates[0])
            record(
                "initializer",
                actual_hash == recorded_hash,
                f"{init_candidates[0]} sha256 eşleşmesi={'evet' if actual_hash == recorded_hash else 'HAYIR'}",
                "initializer_hash_mismatch",
            )
        else:
            record(
                "initializer",
                True,
                f"{init_candidates[0]} mevcut (tarihsel manifestoda hash kaydı yok; presence-only)",
            )

    # 9. Kapsamda test bulaşmaması + seed/bütçe sanity.
    scope_note = str(manifest.get("dataset_scope_note", ""))
    scope_ok = "test" not in scope_note.lower()
    record(
        "scope_excludes_test",
        scope_ok,
        f"dataset_scope_note={scope_note[:80]}",
        "scope_mentions_test",
    )
    seeds = manifest.get("seeds") or []
    budget = manifest.get("budget_usd") or 0
    record(
        "seeds_and_budget",
        len(seeds) >= 3 and budget > 0,
        f"seeds={seeds}, budget_usd={budget}",
        "seeds_and_budget",
    )

    return PreflightReport(
        passed=len(blockers) == 0,
        blockers=tuple(blockers),
        checks=checks,
    )
