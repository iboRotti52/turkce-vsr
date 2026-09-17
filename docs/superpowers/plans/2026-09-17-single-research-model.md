# Single Evolving Research Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pilot-to-scale workflow with one versioned research model that evolves through evidence-bearing probes and cannot enter full training before a fail-closed readiness gate passes.

**Architecture:** Add a small research-governance layer that loads the canonical YAML recipe, validates stage transitions and evidence, and seals a full-training manifest. Thread candidate/probe identity through the tracker and training entry points, isolate the test split from research decisions, and rewrite the persistent research documents around the single evolving candidate.

**Tech Stack:** Python 3.11, dataclasses/enums, PyYAML, argparse, PyTorch/Modal, pytest, Markdown/YAML research records

**Spec:** `docs/superpowers/specs/2026-09-17-single-research-model-design.md`

## Global Constraints

- There is exactly one canonical research model: `research/CANDIDATE.md` plus `configs/research_candidate.yaml`, sharing the same `candidate_version`.
- Short runs and ablations are probes attached to that candidate, never persistent competing pilot models.
- Architecture inspiration is unrestricted across VSR, video, ASR, self-supervised, multilingual, sequence-modeling, decoding, and adjacent fields; adoption remains constrained by evidence, the silent-video product objective, licensing/access, cost, and reproducibility.
- Research uses train and speaker-disjoint validation data only; the test split is reserved for a frozen candidate and final evaluation.
- Full training fails closed unless every readiness item has evidence and a manifest sealed to the candidate recipe, code revision, data/split hashes, initializer, seed set, and budget.
- Existing experiment history remains readable; no registry row or historical decision is rewritten or deleted.
- Existing user changes and unrelated untracked files must not be staged or modified.

---

## File Structure

- `src/experiments/research_governance.py`: canonical candidate loading, enums, validation, readiness evaluation, transition rules, manifest sealing, and full-training authorization.
- `src/experiments/tracker.py`: backward-compatible probe metadata on experiment records.
- `src/experiments/guardrails.py`: run-scope checks preventing research runs from consuming full-training scope and preventing test-set use during research.
- `configs/research_candidate.yaml`: machine-readable provisional candidate, open questions, evidence references, and readiness state.
- `research/CANDIDATE.md`: human-readable single-candidate ledger.
- `tests/test_research_governance.py`: governance, state transition, readiness, and manifest tests.
- `tests/test_experiment_tracker.py`: probe metadata serialization and legacy-record compatibility tests.
- `tests/test_experiment_guardrails.py`: research/full scope and test-split isolation tests.
- `run_experiment.py`: probe-oriented CLI, legacy aliases, status/manifest commands, and corrected next-step semantics.
- `src/modal_runner/cloud_train.py`: candidate/probe provenance and validation-only research evaluation.
- `GEMINI.md`, `research_plan.md`, `research/ARCHITECTURE.md`, `research/RESEARCH_STATE.md`, `research/NEXT_ACTION.md`, `research/DECISIONS.md`: persistent protocol and current-state alignment.

### Task 1: Canonical Candidate Schema and Governance Loader

**Files:**
- Create: `src/experiments/research_governance.py`
- Create: `configs/research_candidate.yaml`
- Create: `research/CANDIDATE.md`
- Create: `tests/test_research_governance.py`

**Interfaces:**
- Produces: `ResearchStage`, `ProbeScale`, `ProbeDecision`, `CandidateState`, `load_candidate(path)`, `validate_candidate_document(candidate, markdown_path)`, and `validate_transition(current, requested)`.
- Consumes: PyYAML and repository-local YAML/Markdown files only; no training code dependency.

- [ ] **Step 1: Write failing loader and state-transition tests**

```python
from pathlib import Path

import pytest

from src.experiments.research_governance import (
    ResearchStage,
    load_candidate,
    validate_candidate_document,
    validate_transition,
)


def test_load_candidate_requires_single_version_and_open_question(tmp_path: Path):
    recipe = tmp_path / "candidate.yaml"
    recipe.write_text(
        """
schema_version: 1
candidate_version: c0.1.0
stage: RESEARCHING
components: {frontend: provisional-auto-avsr}
training: {objective: char-ctc}
open_questions:
  - id: INIT-001
    impact: high
    status: open
readiness: {}
""",
        encoding="utf-8",
    )
    candidate = load_candidate(recipe)
    assert candidate.candidate_version == "c0.1.0"
    assert candidate.stage is ResearchStage.RESEARCHING
    assert candidate.open_questions[0]["id"] == "INIT-001"


def test_candidate_markdown_version_must_match_yaml(tmp_path: Path):
    recipe = tmp_path / "candidate.yaml"
    recipe.write_text(
        "schema_version: 1\ncandidate_version: c0.1.0\nstage: RESEARCHING\n"
        "components: {}\ntraining: {}\nopen_questions: []\nreadiness: {}\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "CANDIDATE.md"
    ledger.write_text("candidate_version: c0.2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_version"):
        validate_candidate_document(load_candidate(recipe), ledger)


def test_research_stage_cannot_skip_to_full_training():
    with pytest.raises(ValueError, match="RESEARCHING.*FULL_TRAINING"):
        validate_transition(ResearchStage.RESEARCHING, ResearchStage.FULL_TRAINING)
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `.venv/bin/python -m pytest tests/test_research_governance.py -v`

Expected: FAIL during import with `ModuleNotFoundError: src.experiments.research_governance`.

- [ ] **Step 3: Implement the governance types and strict loader**

```python
class ResearchStage(str, Enum):
    RESEARCHING = "RESEARCHING"
    CONFIRMING = "CONFIRMING"
    REHEARSAL = "REHEARSAL"
    READY_FOR_FULL_TRAIN = "READY_FOR_FULL_TRAIN"
    FULL_TRAINING = "FULL_TRAINING"
    EVALUATING = "EVALUATING"


class ProbeScale(str, Enum):
    DIAGNOSTIC = "DIAGNOSTIC"
    RESEARCH = "RESEARCH"
    CONFIRMATION = "CONFIRMATION"
    REHEARSAL = "REHEARSAL"


class ProbeDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class CandidateState:
    schema_version: int
    candidate_version: str
    stage: ResearchStage
    components: dict[str, Any]
    training: dict[str, Any]
    open_questions: list[dict[str, Any]]
    readiness: dict[str, dict[str, Any]]
    raw: dict[str, Any]


_ALLOWED_TRANSITIONS = {
    ResearchStage.RESEARCHING: {ResearchStage.CONFIRMING},
    ResearchStage.CONFIRMING: {ResearchStage.RESEARCHING, ResearchStage.REHEARSAL},
    ResearchStage.REHEARSAL: {ResearchStage.RESEARCHING, ResearchStage.READY_FOR_FULL_TRAIN},
    ResearchStage.READY_FOR_FULL_TRAIN: {ResearchStage.FULL_TRAINING},
    ResearchStage.FULL_TRAINING: {ResearchStage.EVALUATING},
    ResearchStage.EVALUATING: {ResearchStage.RESEARCHING},
}
```

`load_candidate()` must reject missing top-level keys, unknown stages, duplicate open-question IDs, invalid impact/status values, and non-dictionary component/training/readiness sections. `validate_candidate_document()` must find the literal `candidate_version: <value>` line in Markdown. `validate_transition()` must accept staying in the same stage for idempotent status checks but reject skipped forward transitions.

- [ ] **Step 4: Create the provisional canonical candidate artifacts**

`configs/research_candidate.yaml` starts at `candidate_version: c0.1.0` and `stage: RESEARCHING`. Record the current 3D-ResNet18/Auto-AVSR frontend, BiGRU, character CTC, curriculum, optimizer, decoder, and keyword stack as `status: provisional`, not as final decisions. Add at least these open high-impact questions:

```yaml
open_questions:
  - id: INIT-001
    component: frontend_initializer
    impact: high
    status: open
    question: Does the legacy exp13 initializer help on clean iboRotti data versus trusted frontend-only initialization?
  - id: ARCH-001
    component: end_to_end_architecture
    impact: high
    status: open
    question: Is the current frontend plus BiGRU plus character-CTC stack the best evidence-backed fit for this use case and data regime?
  - id: TRAIN-001
    component: training_recipe
    impact: high
    status: open
    question: Which curriculum, freeze schedule, optimizer, and regularization recipe transfers from probes to representative validation data?
```

Create all twelve readiness keys from the spec with `passed: false` and `evidence: []`. `research/CANDIDATE.md` mirrors version, stage, provisional architecture, evidence status, open questions, readiness, and the single next question `INIT-001`.

- [ ] **Step 5: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/test_research_governance.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the canonical candidate foundation**

```bash
git add src/experiments/research_governance.py configs/research_candidate.yaml research/CANDIDATE.md tests/test_research_governance.py
git commit -m "feat: add canonical research candidate governance"
```

### Task 2: Readiness Evaluation and Sealed Full-Training Manifest

**Files:**
- Modify: `src/experiments/research_governance.py`
- Modify: `tests/test_research_governance.py`

**Interfaces:**
- Produces: `ReadinessReport`, `evaluate_readiness(candidate)`, `build_full_train_manifest(...)`, `build_full_train_manifest_from_path(...)`, `write_full_train_manifest(...)`, and `require_full_train_authorized(...)`.
- Consumes: `CandidateState` from Task 1 and explicit provenance arguments; no global inference of seed, budget, or initializer.

- [ ] **Step 1: Write failing readiness and tamper-detection tests**

```python
import copy
import yaml

from src.experiments.research_governance import (
    REQUIRED_READINESS_KEYS,
    build_full_train_manifest,
    build_full_train_manifest_from_path,
    evaluate_readiness,
    load_candidate,
    require_full_train_authorized,
    write_full_train_manifest,
)


@pytest.fixture
def candidate_state(tmp_path):
    readiness = {
        key: {"passed": False, "evidence": []}
        for key in REQUIRED_READINESS_KEYS
    }
    raw = {
        "schema_version": 1,
        "candidate_version": "c0.1.0",
        "stage": "RESEARCHING",
        "components": {"frontend": {"choice": "provisional-auto-avsr"}},
        "training": {"objective": "char-ctc"},
        "open_questions": [
            {"id": "ARCH-001", "impact": "high", "status": "open"}
        ],
        "readiness": readiness,
    }
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_candidate(path)


@pytest.fixture
def ready_candidate_file(tmp_path, candidate_state):
    raw = copy.deepcopy(candidate_state.raw)
    raw["stage"] = "READY_FOR_FULL_TRAIN"
    raw["open_questions"] = []
    raw["readiness"] = {
        key: {"passed": True, "evidence": [f"research/evidence/{key}.md"]}
        for key in REQUIRED_READINESS_KEYS
    }
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_open_high_impact_question_blocks_readiness(candidate_state):
    report = evaluate_readiness(candidate_state)
    assert not report.ready
    assert "open_high_impact_questions" in report.blockers


def test_false_readiness_item_blocks_manifest(candidate_state):
    with pytest.raises(RuntimeError, match="readiness"):
        build_full_train_manifest(
            candidate_state,
            code_revision="abc123",
            dataset_id="iboRotti/avsr-tr-dataset",
            dataset_sha256="datahash",
            split_sha256="splithash",
            initializer="auto-avsr:trusted.pth",
            seeds=[17, 42, 73],
            budget_usd=20.0,
        )


def test_manifest_hash_must_match_candidate_file(ready_candidate_file, tmp_path):
    manifest_path = tmp_path / "full_train_manifest.json"
    manifest = build_full_train_manifest_from_path(
        ready_candidate_file,
        code_revision="abc123",
        dataset_id="iboRotti/avsr-tr-dataset",
        dataset_sha256="datahash",
        split_sha256="splithash",
        initializer="auto-avsr:trusted.pth",
        seeds=[17, 42, 73],
        budget_usd=20.0,
    )
    write_full_train_manifest(manifest_path, manifest)
    ready_candidate_file.write_text(
        ready_candidate_file.read_text(encoding="utf-8") + "\nchanged: true\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="candidate.*hash"):
        require_full_train_authorized(ready_candidate_file, manifest_path)
```

- [ ] **Step 2: Run tests and verify missing API failures**

Run: `.venv/bin/python -m pytest tests/test_research_governance.py -v`

Expected: FAIL because readiness/manifest functions are not defined.

- [ ] **Step 3: Implement explicit readiness blockers**

```python
@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    blockers: tuple[str, ...]


def evaluate_readiness(candidate: CandidateState) -> ReadinessReport:
    blockers = []
    if candidate.stage not in {
        ResearchStage.REHEARSAL,
        ResearchStage.READY_FOR_FULL_TRAIN,
    }:
        blockers.append("invalid_stage")
    if any(
        q["impact"] == "high" and q["status"] == "open"
        for q in candidate.open_questions
    ):
        blockers.append("open_high_impact_questions")
    for key, item in candidate.readiness.items():
        if item.get("passed") is not True or not item.get("evidence"):
            blockers.append(f"readiness:{key}")
    return ReadinessReport(ready=not blockers, blockers=tuple(blockers))
```

Require the exact twelve readiness keys from the spec. Readiness may be evaluated in `REHEARSAL`, but manifest creation requires `READY_FOR_FULL_TRAIN`. Manifest creation must reject an empty seed list, non-positive budget, blank hashes, a dirty or `unknown` code revision, and any other candidate stage. Serialize with sorted keys and include `candidate_recipe_sha256`; authorization recomputes the hash and rejects any mismatch.

- [ ] **Step 4: Run the governance tests**

Run: `.venv/bin/python -m pytest tests/test_research_governance.py -v`

Expected: PASS, including candidate-tamper rejection.

- [ ] **Step 5: Commit the full-training gate**

```bash
git add src/experiments/research_governance.py tests/test_research_governance.py
git commit -m "feat: gate full training on sealed readiness evidence"
```

### Task 3: Probe-Native Experiment Records with Legacy Compatibility

**Files:**
- Modify: `src/experiments/tracker.py`
- Create: `tests/test_experiment_tracker.py`

**Interfaces:**
- Produces: optional legacy-safe `candidate_version`, `probe_scale`, `research_question`, `changed_factor`, `control`, `treatment`, `acceptance_criteria`, `decision`, and `evidence_paths` fields on `ExperimentRecord`; `validate_probe_record(record)`.
- Consumes: enum values as strings so historical JSONL records remain readable without migration.

- [ ] **Step 1: Write failing serialization and validation tests**

```python
import pytest

from src.experiments.tracker import (
    ExperimentRecord,
    ExperimentTracker,
    validate_probe_record,
)


def test_probe_record_round_trips_candidate_and_decision(tmp_path):
    tracker = ExperimentTracker(tmp_path / "registry.jsonl")
    record = ExperimentRecord(
        experiment_id="probe_init_001",
        hypothesis="Trusted frontend-only initialization transfers better.",
        falsification_criteria="No validation improvement over legacy init.",
        setup={"seed": 42},
        expectation="Lower validation CER without worse keyword recall.",
        candidate_version="c0.1.0",
        probe_scale="RESEARCH",
        research_question="Which initializer should the canonical model use?",
        changed_factor="initializer",
        control="trusted frontend-only",
        treatment="legacy exp13 full checkpoint",
        acceptance_criteria="Direction repeats on confirmation or remains inconclusive.",
        decision="INCONCLUSIVE",
    )
    tracker.log(record)
    loaded = tracker.load_all()[0]
    assert loaded.candidate_version == "c0.1.0"
    assert loaded.changed_factor == "initializer"
    assert loaded.decision == "INCONCLUSIVE"


def test_legacy_registry_row_still_loads(tmp_path):
    path = tmp_path / "registry.jsonl"
    path.write_text(
        '{"experiment_id":"exp1","hypothesis":"h","falsification_criteria":"f",'
        '"setup":{},"expectation":"e"}\n',
        encoding="utf-8",
    )
    assert ExperimentTracker(path).load_all()[0].candidate_version is None


def test_paid_probe_requires_decision_rule():
    record = ExperimentRecord(
        experiment_id="bad",
        hypothesis="h",
        falsification_criteria="f",
        setup={},
        expectation="e",
        candidate_version="c0.1.0",
        probe_scale="RESEARCH",
        research_question="q",
        changed_factor="initializer",
        cost_estimate_usd=0.5,
    )
    with pytest.raises(ValueError, match="acceptance_criteria"):
        validate_probe_record(record)
```

- [ ] **Step 2: Run the tracker tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_experiment_tracker.py -v`

Expected: FAIL because probe fields and `validate_probe_record` do not exist.

- [ ] **Step 3: Add optional fields and paid-probe validation**

Add the named optional fields with `None` or empty-list defaults. `to_dict()` omits `None` fields so old registry shape stays compact. `validate_probe_record()` must require candidate version, question, changed factor, control, treatment, and acceptance criteria for `RESEARCH`, `CONFIRMATION`, and `REHEARSAL` records with a positive estimated cost. It must validate `decision` against `ACCEPT`, `REJECT`, and `INCONCLUSIVE` when present.

- [ ] **Step 4: Validate before every tracker write**

Call `validate_probe_record(record)` at the start of `ExperimentTracker.log()`. Legacy records with no `probe_scale` remain valid.

- [ ] **Step 5: Run focused and regression tests**

Run: `.venv/bin/python -m pytest tests/test_experiment_tracker.py tests/test_experiment_guardrails.py -v`

Expected: PASS.

- [ ] **Step 6: Commit probe-native tracking**

```bash
git add src/experiments/tracker.py tests/test_experiment_tracker.py
git commit -m "feat: attach experiments to the evolving candidate"
```

### Task 4: Run-Scope Guardrails and Test-Set Isolation

**Files:**
- Modify: `src/experiments/guardrails.py`
- Modify: `tests/test_experiment_guardrails.py`
- Modify: `src/modal_runner/cloud_train.py`

**Interfaces:**
- Produces: `require_research_data_scope(run_kind, dataset_scope)`, `require_evaluation_split(run_kind, split)`, and remote-training arguments `run_kind`, `candidate_version`, `research_question`, `acceptance_criteria`.
- Consumes: run kinds `DIAGNOSTIC`, `RESEARCH`, `CONFIRMATION`, `REHEARSAL`, `FULL_TRAINING`, and `EVALUATION`; dataset scopes `sampled`, `representative`, and `full`.

- [ ] **Step 1: Write failing guardrail tests**

```python
from src.experiments.guardrails import (
    require_evaluation_split,
    require_research_data_scope,
)


@pytest.mark.parametrize("run_kind", ["DIAGNOSTIC", "RESEARCH", "CONFIRMATION"])
def test_research_runs_cannot_use_full_dataset_scope(run_kind):
    with pytest.raises(RuntimeError, match="full.*scope"):
        require_research_data_scope(run_kind=run_kind, dataset_scope="full")


@pytest.mark.parametrize("run_kind", ["DIAGNOSTIC", "RESEARCH", "CONFIRMATION"])
def test_research_runs_cannot_evaluate_test_split(run_kind):
    with pytest.raises(RuntimeError, match="test split"):
        require_evaluation_split(run_kind=run_kind, split="test")


def test_rehearsal_may_use_representative_validation():
    require_research_data_scope(run_kind="REHEARSAL", dataset_scope="representative")
    require_evaluation_split(run_kind="REHEARSAL", split="val")
```

- [ ] **Step 2: Run guardrail tests and verify missing functions**

Run: `.venv/bin/python -m pytest tests/test_experiment_guardrails.py -v`

Expected: FAIL on missing imports.

- [ ] **Step 3: Implement fail-closed scope checks**

```python
_RESEARCH_KINDS = {"DIAGNOSTIC", "RESEARCH", "CONFIRMATION", "REHEARSAL"}


def require_research_data_scope(*, run_kind: str, dataset_scope: str) -> None:
    if run_kind in _RESEARCH_KINDS and dataset_scope == "full":
        raise RuntimeError(f"{run_kind} run cannot use full dataset scope")


def require_evaluation_split(*, run_kind: str, split: str) -> None:
    if run_kind in _RESEARCH_KINDS and split == "test":
        raise RuntimeError(f"{run_kind} run cannot evaluate the test split")
```

Reject unknown run kinds/scopes/splits rather than silently accepting them.

- [ ] **Step 4: Remove test-set feedback from remote research training**

Add remote parameters for run/candidate/probe identity. Call the scope checks before downloads. For research modes, stop downloading and scoring `split="test"`; return validation metrics and raw validation examples only. Keep test evaluation in the existing explicit evaluation path for a frozen candidate or final model. Store `candidate_version`, `run_kind`, and research question in checkpoint provenance and the returned result.

Rename `train_remote()` to `run_training_remote()` and retain `train_remote = run_training_remote` as a temporary import alias. For `dataset_scope="full"`, require `run_kind="FULL_TRAINING"`, call `HFDatasetDownloader.download_full()`, construct the complete train and validation datasets from the split map, and force `n_test_samples=0`. The full-training result must contain the manifest hash and candidate version supplied by the authenticated caller. Research scopes continue to use bounded downloads.

- [ ] **Step 5: Run focused tests and a syntax check**

Run: `.venv/bin/python -m pytest tests/test_experiment_guardrails.py -v`

Run: `.venv/bin/python -m py_compile src/modal_runner/cloud_train.py`

Expected: tests PASS and compilation exits 0.

- [ ] **Step 6: Commit scope isolation**

```bash
git add src/experiments/guardrails.py tests/test_experiment_guardrails.py src/modal_runner/cloud_train.py
git commit -m "fix: isolate research decisions from full and test data"
```

### Task 5: Replace Pilot Semantics in the CLI

**Files:**
- Modify: `run_experiment.py`
- Create: `tests/test_run_experiment_cli.py`

**Interfaces:**
- Produces CLI modes `diagnostic-probe`, `research-probe`, `confirmation-run`, `dress-rehearsal`, `research-status`, `seal-full-train`, `authorize-full-train`, and `full-train`.
- Consumes candidate path `--candidate-config`, probe metadata flags, and manifest path `--full-train-manifest`.
- Preserves `micro-pilot` and `modal-pilot` as deprecated aliases for one release; aliases map to diagnostic/research probes and print a warning.

- [ ] **Step 1: Extract parser construction and write failing CLI tests**

```python
import pytest

from run_experiment import (
    build_parser,
    normalize_legacy_mode,
    run_full_train,
    validate_cli_args,
)


def test_research_probe_requires_question_and_acceptance_rule():
    parser = build_parser()
    args = parser.parse_args(["--mode", "research-probe"])
    with pytest.raises(ValueError, match="research-question"):
        validate_cli_args(args)


def test_legacy_modal_pilot_maps_to_research_probe(capsys):
    parser = build_parser()
    args = parser.parse_args(["--mode", "modal-pilot"])
    normalized = normalize_legacy_mode(args.mode)
    assert normalized == "research-probe"
    assert "deprecated" in capsys.readouterr().err.lower()


def test_seal_full_train_requires_manifest_path():
    parser = build_parser()
    args = parser.parse_args(["--mode", "seal-full-train"])
    with pytest.raises(ValueError, match="full-train-manifest"):
        validate_cli_args(args)


def test_full_train_requires_manifest_before_dispatch():
    parser = build_parser()
    args = parser.parse_args(["--mode", "full-train"])
    with pytest.raises(ValueError, match="full-train-manifest"):
        validate_cli_args(args)


def test_full_train_authorizes_before_modal_dispatch(monkeypatch, tmp_path):
    calls = []
    args = build_parser().parse_args(
        [
            "--mode", "full-train",
            "--candidate-config", str(tmp_path / "candidate.yaml"),
            "--full-train-manifest", str(tmp_path / "manifest.json"),
        ]
    )

    def reject_authorization(*_args, **_kwargs):
        raise RuntimeError("stale manifest")

    monkeypatch.setattr(
        "run_experiment.require_full_train_authorized",
        reject_authorization,
    )
    with pytest.raises(RuntimeError, match="stale manifest"):
        run_full_train(args, dispatch=lambda **kwargs: calls.append(kwargs))
    assert calls == []
```

- [ ] **Step 2: Run CLI tests and verify missing parser helpers**

Run: `.venv/bin/python -m pytest tests/test_run_experiment_cli.py -v`

Expected: FAIL because `build_parser`, `validate_cli_args`, and alias normalization do not exist.

- [ ] **Step 3: Refactor parser creation without changing unrelated execution**

Move argparse construction into `build_parser()`. Add:

```python
parser.add_argument("--candidate-config", default="configs/research_candidate.yaml")
parser.add_argument("--candidate-ledger", default="research/CANDIDATE.md")
parser.add_argument("--research-question")
parser.add_argument("--changed-factor")
parser.add_argument("--control")
parser.add_argument("--treatment")
parser.add_argument("--acceptance-criteria")
parser.add_argument("--full-train-manifest")
parser.add_argument("--dataset-scope", choices=["sampled", "representative", "full"], default="sampled")
```

Paid research/confirmation/rehearsal modes require all probe fields. Load and validate the candidate before launching a paid job.

Implement `run_full_train(args, dispatch=run_modal_training)` as a small authorization-first wrapper so the no-GPU-before-authorization behavior can be tested without importing or invoking Modal.

- [ ] **Step 4: Rename execution semantics and correct post-run guidance**

Rename internal presentation from “pilot model” to “probe against candidate `<version>`”. The successful remote-run record must say:

```python
record.next_step = (
    "Analyze validation metrics and raw outputs, record ACCEPT/REJECT/INCONCLUSIVE, "
    "then update the canonical candidate only if the predeclared rule passed."
)
```

Remove “increase parameters or continue with full dataset.” Diagnostic success may only recommend a research probe. Research success may only recommend analysis/decision, never automatic scale-up.

- [ ] **Step 5: Add status, manifest sealing, and authorization commands**

`research-status` prints candidate version, stage, open high-impact questions, readiness blockers, and the single next research question. `seal-full-train` first requires a `REHEARSAL` candidate whose readiness report has no blockers, transitions the YAML to `READY_FOR_FULL_TRAIN`, writes it atomically, then calls `build_full_train_manifest_from_path()` against the updated bytes. It refuses incomplete evidence and never leaves a partial manifest. `authorize-full-train` calls `require_full_train_authorized()` and exits non-zero on any mismatch; it does not itself spend GPU budget.

`full-train` first calls `require_full_train_authorized()` and verifies that the manifest's candidate version, candidate hash, code revision, dataset/split hashes, initializer, seed set, and budget still match the requested run. Only then may it call `run_modal_training()` with `run_kind="FULL_TRAINING"`, `dataset_scope="full"`, and `n_test_samples=0`. It must pass the manifest SHA-256 into checkpoint provenance. Any missing or stale manifest exits before `app.run()` is entered, so unauthorized calls cannot allocate a GPU.

- [ ] **Step 6: Run CLI and existing regression tests**

Run: `.venv/bin/python -m pytest tests/test_run_experiment_cli.py tests/test_experiment_tracker.py tests/test_experiment_guardrails.py -v`

Run: `.venv/bin/python run_experiment.py --mode research-status`

Expected: tests PASS; status prints `c0.1.0`, `RESEARCHING`, blockers, and `INIT-001` without starting training.

- [ ] **Step 7: Commit the probe-oriented CLI**

```bash
git add run_experiment.py tests/test_run_experiment_cli.py
git commit -m "feat: replace pilot workflow with evidence probes"
```

### Task 6: Align the Persistent Research Protocol and Current State

**Files:**
- Modify: `GEMINI.md`
- Modify: `research_plan.md`
- Modify: `research/ARCHITECTURE.md`
- Modify: `research/RESEARCH_STATE.md`
- Modify: `research/NEXT_ACTION.md`
- Modify: `research/DECISIONS.md`

**Interfaces:**
- Consumes: terms and readiness rules from the approved spec and implemented governance module.
- Produces: a self-consistent protocol in which every future `/goal` session resumes the same candidate and cannot interpret probes as competing pilot models.

- [ ] **Step 1: Rewrite the governing protocol in `GEMINI.md`**

Add an early, explicit invariant:

```markdown
## Tek yaşayan araştırma modeli

Her anda yalnız bir kanonik araştırma modeli vardır. Deney kolları bu modelin
belirli bir kararını sınayan geçici probe'lardır; ayrı pilot modeller veya
full-training adayları değildir. Kanıtlanan değişikliği kanonik reçeteye işle,
kanıtlanmayan kolu büyütme. Birkaç olumlu probe full training gerekçesi değildir.
```

Replace “up to three candidate hypotheses” with one active highest-information question plus optional queued hypotheses. Add the unrestricted inspiration policy, stage machine, exact twelve-item readiness gate, validation-only research rule, and manifest requirement. Retain existing data quality, budget, provenance, monitoring, and stop-condition protections.

- [ ] **Step 2: Replace the fixed experiment roadmap**

Rewrite `research_plan.md` around:

```text
canonical candidate
  -> one open high-impact question
  -> source research
  -> cheapest discriminating probe
  -> raw-output/error analysis
  -> ACCEPT / REJECT / INCONCLUSIVE
  -> candidate version update
  -> next question
```

Do not claim a predetermined frontend, encoder, target, or curriculum is final. Describe the current stack only as the provisional starting point.

- [ ] **Step 3: Separate current choices from closed decisions**

In `research/ARCHITECTURE.md`, label each component `provisional`, `supported`, or `frozen`; cite its current evidence and open counter-hypothesis. Remove unsupported absolute statements such as alternatives being impossible solely because they appear larger. Preserve factual historical experiment observations.

- [ ] **Step 4: Reconcile current state and next action**

Set `research/RESEARCH_STATE.md` to candidate `c0.1.0`, stage `RESEARCHING`, no full-training authorization, and list the high-impact questions from YAML. Rewrite `research/NEXT_ACTION.md` so the initializer comparison is one probe attached to `INIT-001`; explicitly state that its two temporary arms do not become two model candidates and neither result authorizes scaling.

- [ ] **Step 5: Record the governance decision**

Append a new dated decision to `research/DECISIONS.md`:

```markdown
## Karar 14 (D14): Tek Yaşayan Araştırma Modeli ve Kanıt Kapılı Full Training

- Karar: Bütün araştırma tek kanonik candidate üzerinde ilerler; deney kolları geçici probe'dur.
- Kanıt: Önceki metin başarılı bir pilot sonrası doğrudan ölçek önerisine izin veriyordu ve full-training readiness kapısı tanımlamıyordu.
- Kararı değiştirecek kanıt: Çok-adaylı kalıcı model portföyünün aynı bütçede daha yüksek bilgi değeri ürettiğini ve erken ölçek riskini artırmadığını gösteren kontrollü süreç değerlendirmesi.
```

- [ ] **Step 6: Run consistency searches**

Run: `rg -n "parametrelerini artırarak|tam veri kümesiyle devam|pilot modeli|en fazla üç aday" GEMINI.md research_plan.md research run_experiment.py src`

Expected: no active instruction recommends automatic scaling or persistent pilot candidates; historical quotations, if any, are explicitly labeled historical.

Run: `rg -n "candidate_version|READY_FOR_FULL_TRAIN|full_train_manifest|probe" GEMINI.md research_plan.md research configs/research_candidate.yaml`

Expected: candidate identity, state gate, manifest, and probe semantics appear in all governing records.

- [ ] **Step 7: Commit protocol alignment**

```bash
git add GEMINI.md research_plan.md research/ARCHITECTURE.md research/RESEARCH_STATE.md research/NEXT_ACTION.md research/DECISIONS.md
git commit -m "docs: align VSR research around one evolving model"
```

### Task 7: End-to-End Verification and Clean Handoff

**Files:**
- Modify only if verification exposes a defect in files from Tasks 1–6.

**Interfaces:**
- Consumes all prior tasks.
- Produces evidence that research remains usable while premature full training is blocked.

- [ ] **Step 1: Run the complete test suite**

Run: `.venv/bin/python -m pytest tests/ -v`

Expected: all tests PASS.

- [ ] **Step 2: Verify candidate status is read-only and accurate**

Run: `.venv/bin/python run_experiment.py --mode research-status`

Expected: prints `candidate_version=c0.1.0`, `stage=RESEARCHING`, readiness false, high-impact open questions, and next question `INIT-001`; does not write registry data or start Modal.

- [ ] **Step 3: Verify premature manifest sealing fails closed**

Run: `.venv/bin/python run_experiment.py --mode seal-full-train --full-train-manifest /tmp/turkish-vsr-full-train-manifest.json`

Expected: non-zero exit with readiness blockers; no manifest file is created.

- [ ] **Step 4: Verify legacy command compatibility without launching paid work**

Run: `.venv/bin/python -m pytest tests/test_run_experiment_cli.py::test_legacy_modal_pilot_maps_to_research_probe -v`

Expected: PASS and warning text confirms the alias maps to a research probe.

- [ ] **Step 5: Check syntax and diff hygiene**

Run: `.venv/bin/python -m py_compile run_experiment.py src/experiments/research_governance.py src/experiments/tracker.py src/experiments/guardrails.py src/modal_runner/cloud_train.py`

Run: `git diff --check`

Expected: both commands exit 0.

- [ ] **Step 6: Inspect only task-owned changes**

Run: `git status --short`

Expected: pre-existing unrelated user files remain untouched; only intended task files are part of task commits.

- [ ] **Step 7: Commit any verification-only correction**

If verification required a correction, stage only that correction and commit:

```bash
git commit -m "fix: enforce research governance invariants"
```

If no correction was required, do not create an empty commit.
