"""Large-data research utilities for Hugging Face preprocessed AVSR data.

This module intentionally does not preprocess raw video. It operates on the
accepted manifest that already points to preprocessed mouth clips.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import pathlib
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SPEAKER_IDENTITY_FIELDS: Tuple[str, ...] = (
    "speaker_id",
    "speaker",
    "channel",
    "creator",
    "source_channel",
)
PROXY_SPEAKER_FIELDS = frozenset({"channel", "creator", "source_channel"})


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_speaker_identity_field(rows: Sequence[Mapping[str, Any]]) -> str:
    """Choose one identity field that is populated for every accepted row.

    Mixing speaker_id for some rows with channel for others would make leakage
    guarantees semantically inconsistent, so large-data planning fails closed.
    """
    for key in SPEAKER_IDENTITY_FIELDS:
        if rows and all(str(row.get(key, "")).strip() for row in rows):
            return key
    raise ValueError(
        "Tüm accepted satırlarda ortak bir speaker identity alanı yok. "
        "speaker_id/speaker tercih edilir; channel yalnız proxy olarak kullanılabilir."
    )


def _row_speaker(row: Mapping[str, Any], speaker_field: Optional[str] = None) -> str:
    key = speaker_field
    if key is None:
        for candidate in SPEAKER_IDENTITY_FIELDS:
            value = str(row.get(candidate, "")).strip()
            if value:
                return value
        raise ValueError("Manifest satırında speaker/channel kimliği bulunamadı.")
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"Manifest satırında speaker identity alanı boş: {key}")
    return value


def _row_duration(row: Mapping[str, Any]) -> float:
    try:
        return max(0.0, float(row.get("duration", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _row_id(row: Mapping[str, Any]) -> str:
    item = str(row.get("item_id", "")).strip()
    seg = str(row.get("segment_id", "")).strip()
    if not item or not seg:
        raise ValueError("Manifest satırında item_id ve segment_id zorunludur.")
    return f"{item}/{seg}"


@dataclass(frozen=True)
class LargeDataPlan:
    dataset_id: str
    dataset_revision: str
    seed: int
    speaker_identity_field: str
    speaker_identity_is_proxy: bool
    split_map: Dict[str, Dict[str, str]]
    split_summary: Dict[str, Dict[str, float]]
    staged_subsets: Dict[str, List[str]]
    staged_summary: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "seed": self.seed,
            "speaker_identity_field": self.speaker_identity_field,
            "speaker_identity_is_proxy": self.speaker_identity_is_proxy,
            "split_map": self.split_map,
            "split_summary": self.split_summary,
            "staged_subsets": self.staged_subsets,
            "staged_summary": self.staged_summary,
        }
        payload["split_map_sha256"] = _canonical_sha256(self.split_map)
        payload["plan_sha256"] = _canonical_sha256(payload)
        return payload


def build_speaker_disjoint_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    speaker_field: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Assign each speaker to exactly one split, balancing by total duration.

    The split map is keyed by item/video id because the current dataset loader
    applies split membership at that level. All items from one speaker remain
    in the same split.
    """
    if not rows:
        raise ValueError("Split üretmek için accepted manifest boş olamaz.")
    if not (0.0 < val_fraction < 0.5 and 0.0 < test_fraction < 0.5):
        raise ValueError("val_fraction ve test_fraction 0 ile 0.5 arasında olmalıdır.")
    if val_fraction + test_fraction >= 0.8:
        raise ValueError("Train için yeterli pay bırakılmalıdır.")

    resolved_speaker_field = speaker_field or resolve_speaker_identity_field(rows)
    speaker_seconds: Dict[str, float] = defaultdict(float)
    item_to_speaker: Dict[str, str] = {}

    for row in rows:
        speaker = _row_speaker(row, resolved_speaker_field)
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            raise ValueError("Manifest satırında item_id zorunludur.")
        previous = item_to_speaker.get(item_id)
        if previous is not None and previous != speaker:
            raise ValueError(f"item_id={item_id} birden fazla speaker ile eşleşiyor: {previous}, {speaker}")
        item_to_speaker[item_id] = speaker
        speaker_seconds[speaker] += _row_duration(row)

    speakers = list(speaker_seconds)
    if len(speakers) < 3:
        raise ValueError("Speaker-disjoint train/val/test için en az 3 konuşmacı gerekir.")

    rng = random.Random(seed)
    rng.shuffle(speakers)
    # Stable sort: the shuffle above only breaks equal-duration ties.
    speakers.sort(key=lambda s: speaker_seconds[s], reverse=True)

    total = sum(speaker_seconds.values())
    if total <= 0:
        raise ValueError("Manifest konuşmacı süreleri sıfır; duration-balanced split üretilemez.")

    target = {
        "train": total * (1.0 - val_fraction - test_fraction),
        "val": total * val_fraction,
        "test": total * test_fraction,
    }
    assigned_seconds = {"train": 0.0, "val": 0.0, "test": 0.0}
    assigned_speakers = {"train": 0, "val": 0, "test": 0}
    speaker_split: Dict[str, str] = {}
    split_order = ("train", "val", "test")

    for idx, speaker in enumerate(speakers):
        remaining_after = len(speakers) - idx - 1
        empty_splits = [s for s in split_order if assigned_speakers[s] == 0]

        # If there are only enough speakers left to populate currently-empty
        # splits, force this speaker into one of them. Otherwise choose freely.
        candidates = (
            empty_splits
            if empty_splits and remaining_after < len(empty_splits)
            else list(split_order)
        )

        def score(candidate: str) -> Tuple[float, int]:
            proposed = dict(assigned_seconds)
            proposed[candidate] += speaker_seconds[speaker]
            normalized_error = sum(
                ((proposed[s] - target[s]) / max(target[s], 1.0)) ** 2
                for s in split_order
            )
            # Deterministic tie-breaker prefers train, then val, then test.
            return normalized_error, split_order.index(candidate)

        split = min(candidates, key=score)
        speaker_split[speaker] = split
        assigned_seconds[split] += speaker_seconds[speaker]
        assigned_speakers[split] += 1

    result: Dict[str, Dict[str, str]] = {}
    for item_id, speaker in sorted(item_to_speaker.items()):
        result[item_id] = {"split": speaker_split[speaker], "speaker": speaker}
    return result


def validate_speaker_disjoint_split(split_map: Mapping[str, Mapping[str, str]]) -> None:
    speaker_to_split: Dict[str, set[str]] = defaultdict(set)
    splits_present: set[str] = set()
    for item_id, info in split_map.items():
        split = str(info.get("split", ""))
        speaker = str(info.get("speaker", ""))
        if not item_id or split not in {"train", "val", "test"} or not speaker:
            raise ValueError(f"Geçersiz split kaydı: {item_id} -> {info}")
        speaker_to_split[speaker].add(split)
        splits_present.add(split)

    leaked = {s: sorted(v) for s, v in speaker_to_split.items() if len(v) != 1}
    if leaked:
        raise ValueError(f"Speaker leakage tespit edildi: {leaked}")
    if splits_present != {"train", "val", "test"}:
        raise ValueError(f"Train/val/test üçlüsü zorunlu; bulunan: {sorted(splits_present)}")


def summarize_split(
    rows: Sequence[Mapping[str, Any]],
    split_map: Mapping[str, Mapping[str, str]],
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, Any]] = {
        split: {"clips": 0, "seconds": 0.0, "speakers": set()}
        for split in ("train", "val", "test")
    }
    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        info = split_map[item_id]
        split = info["split"]
        summary[split]["clips"] += 1
        summary[split]["seconds"] += _row_duration(row)
        summary[split]["speakers"].add(info["speaker"])

    return {
        split: {
            "clips": int(info["clips"]),
            "hours": round(float(info["seconds"]) / 3600.0, 4),
            "speakers": len(info["speakers"]),
        }
        for split, info in summary.items()
    }


def build_speaker_diverse_training_stages(
    rows: Sequence[Mapping[str, Any]],
    split_map: Mapping[str, Mapping[str, str]],
    *,
    targets_hours: Sequence[float] = (10.0, 25.0, 50.0, 100.0),
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    """Build nested, deterministic train-only subsets with speaker round-robin.

    Samples are interleaved across speakers before taking duration targets, so
    early scaling stages maximize speaker diversity rather than taking the first
    N clips from a few prolific channels.
    """
    per_speaker: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        info = split_map[item_id]
        if info["split"] != "train":
            continue
        per_speaker[info["speaker"]].append((_row_id(row), _row_duration(row)))

    if not per_speaker:
        raise ValueError("Train split boş; staged subset üretilemez.")

    rng = random.Random(seed)
    speakers = sorted(per_speaker)
    for speaker in speakers:
        rng.shuffle(per_speaker[speaker])

    interleaved: List[Tuple[str, float, str]] = []
    offset = 0
    while True:
        added = False
        speaker_order = speakers[:]
        rng.shuffle(speaker_order)
        for speaker in speaker_order:
            if offset < len(per_speaker[speaker]):
                sample_id, duration = per_speaker[speaker][offset]
                interleaved.append((sample_id, duration, speaker))
                added = True
        if not added:
            break
        offset += 1

    stages: Dict[str, List[str]] = {}
    summaries: Dict[str, Dict[str, float]] = {}
    cumulative_seconds = 0.0
    cumulative_ids: List[str] = []
    cumulative_speakers: set[str] = set()
    target_iter = iter(sorted(float(x) for x in targets_hours if x > 0))
    current_target = next(target_iter, None)

    for sample_id, duration, speaker in interleaved:
        cumulative_ids.append(sample_id)
        cumulative_seconds += duration
        cumulative_speakers.add(speaker)
        while current_target is not None and cumulative_seconds >= current_target * 3600.0:
            key = f"{current_target:g}h"
            stages[key] = list(cumulative_ids)
            summaries[key] = {
                "clips": len(cumulative_ids),
                "hours": round(cumulative_seconds / 3600.0, 4),
                "speakers": len(cumulative_speakers),
                "sample_ids_sha256": _canonical_sha256(cumulative_ids),
            }
            current_target = next(target_iter, None)

    # Do not invent stages larger than available data. Always expose the full train set.
    full_key = "full"
    stages[full_key] = list(cumulative_ids)
    summaries[full_key] = {
        "clips": len(cumulative_ids),
        "hours": round(cumulative_seconds / 3600.0, 4),
        "speakers": len(cumulative_speakers),
        "sample_ids_sha256": _canonical_sha256(cumulative_ids),
    }
    return stages, summaries


def build_large_data_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    dataset_revision: str,
    seed: int = 42,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    targets_hours: Sequence[float] = (10.0, 25.0, 50.0, 100.0),
) -> LargeDataPlan:
    normalized_revision = dataset_revision.strip().lower()
    if len(normalized_revision) != 40 or any(ch not in "0123456789abcdef" for ch in normalized_revision):
        raise ValueError("dataset_revision immutable 40-hex Hugging Face commit SHA olmalıdır.")
    if not rows:
        raise ValueError("Large-data plan için accepted manifest boş olamaz.")

    seen_sample_ids: set[str] = set()
    for row in rows:
        sample_id = _row_id(row)
        if sample_id in seen_sample_ids:
            raise ValueError(f"Accepted manifest duplicate sample içeriyor: {sample_id}")
        seen_sample_ids.add(sample_id)
        if _row_duration(row) <= 0.0:
            raise ValueError(
                f"Accepted manifest sample duration eksik/geçersiz: {sample_id}. "
                "Saat bazlı scaling planı için pozitif duration zorunludur."
            )

    speaker_field = resolve_speaker_identity_field(rows)
    split_map = build_speaker_disjoint_split(
        rows,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        speaker_field=speaker_field,
    )
    validate_speaker_disjoint_split(split_map)
    staged, staged_summary = build_speaker_diverse_training_stages(
        rows, split_map, targets_hours=targets_hours, seed=seed
    )
    return LargeDataPlan(
        dataset_id=dataset_id,
        dataset_revision=normalized_revision,
        seed=seed,
        speaker_identity_field=speaker_field,
        speaker_identity_is_proxy=speaker_field in PROXY_SPEAKER_FIELDS,
        split_map=split_map,
        split_summary=summarize_split(rows, split_map),
        staged_subsets=staged,
        staged_summary=staged_summary,
    )


def write_large_data_plan(path: pathlib.Path, plan: LargeDataPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )



def load_large_data_plan(path: pathlib.Path) -> Dict[str, Any]:
    """Load and cryptographically verify a generated large-data plan."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Large-data plan bulunamadı: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_plan_hash = payload.get("plan_sha256")
    if not expected_plan_hash:
        raise RuntimeError("Large-data plan plan_sha256 içermiyor.")

    unhashed = dict(payload)
    unhashed.pop("plan_sha256", None)
    actual_plan_hash = _canonical_sha256(unhashed)
    if actual_plan_hash != expected_plan_hash:
        raise RuntimeError(
            f"Large-data plan hash uyuşmuyor: expected={expected_plan_hash}, actual={actual_plan_hash}"
        )

    split_map = payload.get("split_map") or {}
    actual_split_hash = _canonical_sha256(split_map)
    if actual_split_hash != payload.get("split_map_sha256"):
        raise RuntimeError("Large-data plan içindeki split_map hash uyuşmuyor.")

    stages = payload.get("staged_subsets") or {}
    summaries = payload.get("staged_summary") or {}
    for stage, sample_ids in stages.items():
        summary = summaries.get(stage) or {}
        expected_stage_hash = summary.get("sample_ids_sha256")
        if not expected_stage_hash:
            raise RuntimeError(f"Large-data stage hash eksik: {stage}")
        if _canonical_sha256(sample_ids) != expected_stage_hash:
            raise RuntimeError(f"Large-data stage sample hash uyuşmuyor: {stage}")
    return payload


def verify_large_data_split_file(
    plan: Mapping[str, Any],
    split_path: pathlib.Path,
) -> None:
    """Ensure the loader-facing split JSON is exactly the plan's canonical split."""
    split_path = pathlib.Path(split_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Large-data split map bulunamadı: {split_path}")
    split_map = json.loads(split_path.read_text(encoding="utf-8"))
    actual_hash = _canonical_sha256(split_map)
    expected_hash = plan.get("split_map_sha256")
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Large-data split map drift tespit edildi: expected={expected_hash}, actual={actual_hash}"
        )
