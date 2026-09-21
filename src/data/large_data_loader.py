"""DataLoader factory for 50–100h preprocessed VSR datasets."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SequenceBucketSampler, pad_collate_fn


DEFAULT_BUCKETS: List[Tuple[float, float, int]] = [
    (0.0, 3.5, 8),
    (3.5, 8.0, 6),
    (8.0, 100.0, 2),
]


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
