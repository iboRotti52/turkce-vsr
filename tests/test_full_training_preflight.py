"""tests/test_full_training_preflight.py — Canonical preflight fail-closed testleri.

Hermetik: torch/modal/GPU/ağ gerektirmez. Tarihsel c0.4.0 manifestosu için
gerçek repo ağacı kullanılır (salt-okunur); geçiş senaryoları tmp fixture
ile kurulur.
"""

import hashlib
import json
import pathlib
import subprocess

import pytest
import yaml

import src.full_training as full_training_cli
from src.experiments import research_governance as gov
from src.experiments.research_governance import (
    REQUIRED_READINESS_KEYS,
    CodeRevisionStatus,
    build_full_train_manifest,
    get_code_revision_status,
    preflight_full_train_manifest,
    require_clean_code_revision,
    resolve_candidate_recipe_path,
    write_full_train_manifest,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLEAN_STATE = CodeRevisionStatus(revision="testrev123", clean=True, dirty_files=())


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("video_id,seg_id,clip_path,text,duration,channel\n")
        for video, seg, channel in rows:
            f.write(f'{video},{seg},data/clips/{video}/{seg}/mouth.mp4,"ornek {seg}",3.0,{channel}\n')


def _ready_candidate_dict(version="c9.9.9-t"):
    return {
        "schema_version": 1,
        "candidate_version": version,
        "stage": "READY_FOR_FULL_TRAIN",
        "status": "active",
        "data": {"source": "t/dataset", "splits": {}},
        "components": {"decoder": {"choice": "x"}, "keyword_spotting": {"choice": "y"}},
        "training": {"recipe_status": "frozen", "hyperparameters": {"epochs": 1}},
        "open_questions": [],
        "readiness": {k: {"passed": True, "evidence": ["note"]} for k in REQUIRED_READINESS_KEYS},
    }


def _consistent_fixture(tmp_path, monkeypatch):
    """Tutarlı tmp repo düzeni: hash'ler, sürüm, initializer, kanal ayrıklığı."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "checkpoints").mkdir()
    candidate = _ready_candidate_dict()
    train_csv = tmp_path / "data" / "metadata" / "train.csv"
    val_csv = tmp_path / "data" / "metadata" / "val.csv"
    test_csv = tmp_path / "data" / "metadata" / "test.csv"
    _write_csv(train_csv, [("vid-a", "000001", "ch-a"), ("vid-a", "000002", "ch-a")])
    _write_csv(val_csv, [("vid-b", "000001", "ch-b")])
    _write_csv(test_csv, [("vid-c", "000001", "ch-c")])

    def _sha(p):
        return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

    candidate["data"]["splits"] = {"test": {"hash": _sha(test_csv)}}
    candidate_path = tmp_path / "configs" / "research_candidate.yaml"
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")

    init_pt = tmp_path / "checkpoints" / "init-t.pt"
    init_pt.write_bytes(b"dummy-initializer-bytes")

    state = gov.load_candidate(candidate_path)
    manifest = build_full_train_manifest(
        state,
        code_revision="testrev123",
        dataset_id="t/dataset",
        dataset_sha256=_sha(train_csv),
        split_sha256=_sha(val_csv),
        initializer="init-t.pt",
        seeds=[1, 2, 3],
        budget_usd=1.0,
        dataset_scope_note="1325 clips small scope",
    )
    manifest_path = tmp_path / "full_train_manifest.json"
    write_full_train_manifest(manifest_path, manifest)

    # Fixture git reposu değildir; VCS karantina kontrolü monkeypatch ile
    # açıkça bypass edilir (ayrı test VCS belirsizliğinde fail-closed'u kanıtlar).
    monkeypatch.setattr(gov, "_git_path_is_clean", lambda root, rel: True)
    return manifest_path, candidate_path


def test_historical_manifest_is_evidence_not_executable():
    """Tarihsel c0.4.0 manifestosu preflight'ta fail-closed KALMALIDIR.

    Doğrulanan gerçek (bkz. full_train_manifest.PROVENANCE.md): reçete baytları
    ve dataset kimliği canlı tutar; kod revizyonu (dirty tree / tarihsel ata
    revizyon), split CSV tam-hash'leri, karantina-hash ve initializer
    fail-closed kalır.
    """
    report = preflight_full_train_manifest(
        ROOT / "full_train_manifest.json",
        ROOT / "configs" / "research_candidate.yaml",
        ROOT,
    )
    assert not report.passed
    # Kod: dirty tree'de not_clean, temiz ağaçta ata-revizyon mismatch beklenir.
    assert (
        "code_revision_mismatch" in report.blockers
        or "code_revision_not_clean" in report.blockers
    )
    if "code_revision_mismatch" in report.blockers:
        assert "atası" in report.checks["code_revision"]["detail"]
    # Split CSV tam-hash'leri arşivden recompute ile tutmaz (provenance boşluğu):
    assert "dataset_hash_mismatch" in report.blockers
    assert "split_hash_mismatch" in report.blockers
    assert "test_quarantine" in report.blockers
    assert "initializer_not_present" in report.blockers
    # Ama bunlar GEÇMELİDİR (şema, temel yetki, reçete yolu, dataset kimliği):
    assert report.checks["manifest_schema"]["status"] == "PASSED"
    assert report.checks["base_authorization"]["status"] == "PASSED"
    assert report.checks["recipe_path_resolution"]["status"] == "PASSED"
    assert report.checks["dataset_identity"]["status"] == "PASSED"


def test_historical_manifest_cli_exits_nonzero(capsys):
    ret = full_training_cli.main(
        ["--manifest", str(ROOT / "full_train_manifest.json"),
         "--candidate", str(ROOT / "configs" / "research_candidate.yaml"),
         "--root", str(ROOT)]
    )
    assert ret == 1
    out = capsys.readouterr().out
    assert "PREFLIGHT" in out and "ücretli işlem YOK" in out


def test_consistent_fixture_passes_preflight(tmp_path, monkeypatch):
    manifest_path, candidate_path = _consistent_fixture(tmp_path, monkeypatch)
    report = preflight_full_train_manifest(
        manifest_path, candidate_path, tmp_path, code_state=CLEAN_STATE
    )
    assert report.passed, f"Blockers: {report.blockers}"
    assert report.blockers == ()


def test_cli_without_git_repo_fails_closed(tmp_path, monkeypatch, capsys):
    manifest_path, candidate_path = _consistent_fixture(tmp_path, monkeypatch)
    ret = full_training_cli.main(
        ["--manifest", str(manifest_path),
         "--candidate", str(candidate_path),
         "--root", str(tmp_path)]
    )
    # code_state enjekte edilemediği için CLI canlı git'e bakar: tmp git
    # reposu değildir → code_revision_not_clean engeli beklenir (fail-closed).
    assert ret == 1
    assert "code_revision_not_clean" in capsys.readouterr().out


def test_tampered_recipe_fails_base_authorization(tmp_path, monkeypatch):
    manifest_path, candidate_path = _consistent_fixture(tmp_path, monkeypatch)
    candidate_path.write_text(
        candidate_path.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )
    report = preflight_full_train_manifest(
        manifest_path, candidate_path, tmp_path, code_state=CLEAN_STATE
    )
    assert not report.passed
    assert "base_authorization" in report.blockers


def test_dataset_identity_mismatch_blocked(tmp_path, monkeypatch):
    """c0.4.0 kanıtının sessizce büyük veriye uyarlanması engellenir."""
    manifest_path, candidate_path = _consistent_fixture(tmp_path, monkeypatch)
    raw = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    raw["data"]["source"] = "baska/buyuk-veri"
    candidate_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    report = preflight_full_train_manifest(
        manifest_path, candidate_path, tmp_path, code_state=CLEAN_STATE
    )
    assert not report.passed
    assert "dataset_identity_mismatch" in report.blockers


def test_dirty_tree_blocked(tmp_path, monkeypatch):
    manifest_path, candidate_path = _consistent_fixture(tmp_path, monkeypatch)
    dirty = CodeRevisionStatus(revision="testrev123", clean=False, dirty_files=("a.py",))
    report = preflight_full_train_manifest(
        manifest_path, candidate_path, tmp_path, code_state=dirty
    )
    assert not report.passed
    assert "code_revision_not_clean" in report.blockers


def test_missing_initializer_blocked(tmp_path, monkeypatch):
    manifest_path, candidate_path = _consistent_fixture(tmp_path, monkeypatch)
    (tmp_path / "checkpoints" / "init-t.pt").unlink()
    report = preflight_full_train_manifest(
        manifest_path, candidate_path, tmp_path, code_state=CLEAN_STATE
    )
    assert not report.passed
    assert "initializer_not_present" in report.blockers


def test_channel_leakage_into_train_blocked(tmp_path, monkeypatch):
    manifest_path, candidate_path = _consistent_fixture(tmp_path, monkeypatch)
    train_csv = tmp_path / "data" / "metadata" / "train.csv"
    with train_csv.open("a", encoding="utf-8") as f:
        f.write('vid-c,000002,data/clips/vid-c/000002/mouth.mp4,"sızıntı",3.0,ch-c\n')
    report = preflight_full_train_manifest(
        manifest_path, candidate_path, tmp_path, code_state=CLEAN_STATE
    )
    assert not report.passed
    # Hash de eşleşmeyecektir; karantina engeli de beklenir.
    assert "test_quarantine" in report.blockers


def test_vcs_unverifiable_is_fail_closed(tmp_path, monkeypatch):
    manifest_path, candidate_path = _consistent_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(gov, "_git_path_is_clean", lambda root, rel: None)
    report = preflight_full_train_manifest(
        manifest_path, candidate_path, tmp_path, code_state=CLEAN_STATE
    )
    assert not report.passed
    assert "test_quarantine" in report.blockers


def test_resolve_recipe_path_ignores_absolute_path(tmp_path):
    manifest = {"candidate_recipe_path": "/Users/baska/makine/configs/research_candidate.yaml"}
    (tmp_path / "configs").mkdir()
    real = tmp_path / "configs" / "research_candidate.yaml"
    real.write_text("candidate_version: c9\n", encoding="utf-8")
    assert resolve_candidate_recipe_path(manifest, tmp_path) == real


def test_require_clean_code_revision_hermetic(tmp_path):
    subprocess.check_output(["git", "init", "-q"], cwd=str(tmp_path))
    subprocess.check_output(["git", "config", "user.email", "t@t"], cwd=str(tmp_path))
    subprocess.check_output(["git", "config", "user.name", "t"], cwd=str(tmp_path))
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    subprocess.check_output(["git", "add", "."], cwd=str(tmp_path))
    subprocess.check_output(["git", "commit", "-qm", "init"], cwd=str(tmp_path))
    sha = require_clean_code_revision(tmp_path)
    assert len(sha) == 40
    (tmp_path / "f.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(RuntimeError, match="[Kk]irli"):
        require_clean_code_revision(tmp_path)


def test_get_code_revision_status_unknown_outside_repo(tmp_path):
    status = get_code_revision_status(tmp_path)
    assert status.revision == "unknown"
    assert not status.clean
