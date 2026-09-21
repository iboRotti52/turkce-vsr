"""
src/experiments/tracker.py — Research-Craft Otomatik Deney Kayıt Kütüğü
"""

import json
import os
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

EXPERIMENTS_REGISTRY_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "experiments" / "registry.jsonl"
)


@dataclass
class ExperimentRecord:
    experiment_id: str
    hypothesis: str
    falsification_criteria: str
    setup: Dict[str, Any]
    expectation: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    result: Optional[Dict[str, Any]] = None
    status: str = "IN_PROGRESS"  # IN_PROGRESS, PASSED, FALSIFIED, ERROR
    surprise: str = ""
    updated_belief: str = ""
    next_step: str = ""
    cost_estimate_usd: Optional[float] = None
    cost_usd: Optional[float] = None
    candidate_version: Optional[str] = None
    data_scale: Optional[str] = None
    minimum_sufficient_scale: Optional[str] = None
    evidence_scope: Optional[str] = None
    promotion_rule: Optional[str] = None
    scale_action: Optional[str] = None
    scale_parent_experiment_id: Optional[str] = None
    estimated_gpu_hours: Optional[float] = None
    actual_gpu_hours: Optional[float] = None
    technical_status: Optional[str] = None
    scientific_verdict: Optional[str] = None
    pre_result_contract_sha256: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra_fields", {})
        d.update(extra)
        optional_fields = (
            "cost_estimate_usd",
            "cost_usd",
            "candidate_version",
            "data_scale",
            "minimum_sufficient_scale",
            "evidence_scope",
            "promotion_rule",
            "scale_action",
            "scale_parent_experiment_id",
            "estimated_gpu_hours",
            "actual_gpu_hours",
            "technical_status",
            "scientific_verdict",
            "pre_result_contract_sha256",
        )
        for key in optional_fields:
            if d.get(key) is None:
                d.pop(key, None)
        return d


class ExperimentTracker:
    def __init__(self, registry_path: Optional[pathlib.Path] = None):
        self.registry_path = registry_path or EXPERIMENTS_REGISTRY_PATH
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: ExperimentRecord) -> None:
        """Atomically append or update one experiment record."""
        self.log_many([record])

    def log_many(self, records: List[ExperimentRecord]) -> None:
        """Atomically update multiple records in one registry replacement.

        This is used by scale promotion so the parent PROMOTE_SCALE state and the
        child IN_PROGRESS record cannot be split by a process interruption.
        """
        if not records:
            return
        by_id: Dict[str, ExperimentRecord] = {}
        for record in records:
            if not record.experiment_id:
                raise ValueError("experiment_id boş olamaz.")
            if record.experiment_id in by_id:
                raise ValueError(
                    f"Aynı atomic batch içinde duplicate experiment_id: {record.experiment_id}"
                )
            by_id[record.experiment_id] = record

        lines: List[str] = []
        emitted: set[str] = set()
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                        experiment_id = data.get("experiment_id")
                    except Exception:
                        # Preserve legacy/corrupt lines for normal tracker callers.
                        # Governance callers use load_all(strict=True) before mutation.
                        lines.append(raw)
                        continue

                    if experiment_id in by_id:
                        if experiment_id not in emitted:
                            lines.append(
                                json.dumps(
                                    by_id[experiment_id].to_dict(),
                                    ensure_ascii=False,
                                )
                            )
                            emitted.add(experiment_id)
                        # Drop duplicate old copies of the same target id.
                    else:
                        lines.append(raw)

        for experiment_id, record in by_id.items():
            if experiment_id not in emitted:
                lines.append(json.dumps(record.to_dict(), ensure_ascii=False))

        tmp_path = self.registry_path.with_suffix(
            self.registry_path.suffix + ".tmp"
        )
        with open(tmp_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.registry_path)

    def load_all(self, *, strict: bool = False) -> List[ExperimentRecord]:
        if not self.registry_path.exists():
            return []
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(ExperimentRecord)}
        records = []
        with open(self.registry_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    known = {
                        k: v
                        for k, v in data.items()
                        if k in valid_fields and k != "extra_fields"
                    }
                    extra = {k: v for k, v in data.items() if k not in valid_fields}
                    records.append(ExperimentRecord(**known, extra_fields=extra))
                except Exception as exc:
                    if strict:
                        raise RuntimeError(
                            f"Experiment registry parse hatası line={line_number}: {exc}"
                        ) from exc
        return records

    def print_summary(self) -> None:
        records = self.load_all()
        print(f"\n🔬 TOPLAM DENEY SAYISI: {len(records)}")
        print("=" * 80)
        for r in records:
            print(f"[{r.status}] {r.experiment_id}")
            print(f"  Hipotez:    {r.hypothesis}")
            print(f"  Beklenti:   {r.expectation}")
            if r.result:
                print(f"  Sonuç:      {r.result}")
            if r.updated_belief:
                print(f"  Çıkarım:    {r.updated_belief}")
            print("-" * 80)
