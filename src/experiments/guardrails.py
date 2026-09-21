"""Deney başlamadan önce uygulanan veri ve checkpoint güvenlik kontrolleri."""

import hashlib
import pathlib
from typing import Any, Dict, Iterable, Optional, Union


def require_requested_sample_count(*, split: str, requested: int, actual: int) -> None:
    """Örnek sınırlarının deney boyutunu sessizce küçültmesini engeller."""
    if actual < requested:
        raise RuntimeError(
            f"{split} split için {requested} örnek istendi ancak yalnız {actual} örnek seçildi. "
            "max_per_video değerini veya istenen örnek sayısını düzeltin."
        )


def require_checkpoint_exists(path: Union[str, pathlib.Path]) -> pathlib.Path:
    checkpoint_path = pathlib.Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Initializer checkpoint bulunamadı: {checkpoint_path}")
    return checkpoint_path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    canonical = "\n".join(sample_ids).encode("utf-8")
    return _sha256_bytes(canonical)


def collect_sample_ids(dataset) -> list[str]:
    """Sentence/word/phrase datasetlerinden kararlı örnek kimlikleri çıkarır."""
    for attribute in ("samples", "word_samples", "phrase_samples"):
        items = getattr(dataset, attribute, None)
        if items is not None:
            return sorted(
                f"{item['video_id']}/{item['seg_id']}"
                + (f"@{item['start_frame']}:{item['end_frame']}" if "start_frame" in item else "")
                for item in items
            )

    base_dataset = getattr(dataset, "dataset", None)
    indices = getattr(dataset, "indices", None)
    if base_dataset is not None and indices is not None:
        base_ids = collect_sample_ids(base_dataset)
        return sorted(base_ids[index] for index in indices)
    raise TypeError(f"Örnek kimliği çıkarılamayan dataset türü: {type(dataset).__name__}")


def build_checkpoint_provenance(
    *,
    dataset_id: str,
    split_map_path: Union[str, pathlib.Path],
    train_sample_ids: Iterable[str],
    val_sample_ids: Iterable[str],
    seed: int,
    initializer: str,
    code_revision: str,
    working_tree_clean: Optional[bool] = None,
) -> Dict[str, Any]:
    """Bir checkpoint'in veri ve ağırlık kökenini tekrar üretilebilir biçimde kaydeder.

    working_tree_clean: HEAD anında `git status --porcelain` boşsa True olmalı.
    Tarihsel c0.4.0 boşluğu (dirty tree + yalnız HEAD kaydı) tekrarlanmasın diye
    çağrıcılar bu alanı doldurmalıdır (bkz. get_code_revision_status);
    bilinmiyorsa None bırakılır ve eksiklik raporda görünür.
    """
    split_path = pathlib.Path(split_map_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split haritası bulunamadı: {split_path}")

    train_ids = list(train_sample_ids)
    val_ids = list(val_sample_ids)
    return {
        "dataset_id": dataset_id,
        "split_map_path": str(split_path),
        "split_map_sha256": _sha256_bytes(split_path.read_bytes()),
        "train_sample_ids": train_ids,
        "train_sample_ids_sha256": _sample_ids_sha256(train_ids),
        "train_sample_count": len(train_ids),
        "val_sample_ids": val_ids,
        "val_sample_ids_sha256": _sample_ids_sha256(val_ids),
        "val_sample_count": len(val_ids),
        "seed": seed,
        "initializer": initializer,
        "code_revision": code_revision,
        "working_tree_clean": working_tree_clean,
    }
