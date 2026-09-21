"""DataLoader factory for 50–100h preprocessed VSR datasets."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import SequenceBucketSampler, pad_collate_fn


DEFAULT_BUCKETS: List[Tuple[float, float, int]] = [
    (0.0, 3.5, 8),
    (3.5, 8.0, 6),
    (8.0, 100.0, 2),
]



class StageDatasetView(Dataset):
    """Train-only view pinned to the exact sample IDs from a large-data stage."""

    def __init__(self, dataset: Any, sample_ids: List[str]):
        samples = getattr(dataset, "samples", None)
        if samples is None:
            raise TypeError("StageDatasetView base dataset '.samples' alanı sağlamalıdır.")
        if not sample_ids:
            raise ValueError("Large-data stage sample listesi boş olamaz.")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Large-data stage duplicate sample id içeriyor.")

        id_to_index = {}
        for idx, sample in enumerate(samples):
            sample_id = f"{sample.get('video_id', '')}/{sample.get('seg_id', '')}"
            if sample_id in id_to_index:
                raise ValueError(f"Base dataset duplicate sample id içeriyor: {sample_id}")
            id_to_index[sample_id] = idx

        missing = [sample_id for sample_id in sample_ids if sample_id not in id_to_index]
        if missing:
            preview = missing[:5]
            raise RuntimeError(
                f"Large-data stage base dataset'te bulunmayan {len(missing)} sample içeriyor: {preview}"
            )

        self.dataset = dataset
        self.indices = [id_to_index[sample_id] for sample_id in sample_ids]
        self.samples = [samples[idx] for idx in self.indices]
        self.cache_in_ram = bool(getattr(dataset, "cache_in_ram", False))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def build_large_data_loader(
    dataset: Any,
    *,
    train: bool,
    num_workers: int = 4,
    batch_size: int = 8,
    buckets: Optional[List[Tuple[float, float, int]]] = None,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Build an efficient loader without whole-dataset RAM preloading.

    Training uses duration-aware batch sampling; validation/evaluation preserves
    deterministic sample order with a fixed batch size.
    """
    if getattr(dataset, "cache_in_ram", False):
        raise ValueError(
            "Large-data loader tüm dataset RAM cache ile çalıştırılamaz; "
            "dataset'i cache_in_ram=False ile oluşturun."
        )
    if num_workers < 0:
        raise ValueError("num_workers negatif olamaz.")
    if batch_size <= 0:
        raise ValueError("batch_size pozitif olmalıdır.")

    use_pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    common = {
        "dataset": dataset,
        "collate_fn": pad_collate_fn,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }
    if num_workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = prefetch_factor

    if train:
        sampler = SequenceBucketSampler(
            dataset,
            buckets=buckets or DEFAULT_BUCKETS,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        return DataLoader(batch_sampler=sampler, **common)

    return DataLoader(
        batch_size=batch_size,
        shuffle=False,
        **common,
    )


def set_large_data_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """Advance deterministic bucket shuffling for a training epoch.

    Call exactly once before iterating each training epoch. This is explicit
    because ordinary PyTorch DataLoader does not call set_epoch on custom batch
    samplers automatically.
    """
    if epoch < 0:
        raise ValueError("epoch negatif olamaz.")
    sampler = getattr(loader, "batch_sampler", None)
    if not isinstance(sampler, SequenceBucketSampler):
        raise TypeError("Training loader SequenceBucketSampler kullanmıyor.")
    sampler.set_epoch(epoch)
