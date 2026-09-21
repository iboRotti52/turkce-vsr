"""Reusable, fail-closed checkpoint state for long-running VSR training."""

from __future__ import annotations

import pathlib
import random
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_training_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    global_step: int,
    provenance: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]] = None,
    sampler_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    if epoch < 0 or global_step < 0:
        raise ValueError("epoch ve global_step negatif olamaz.")
    required = (
        "dataset_id",
        "dataset_revision",
        "split_map_sha256",
        "train_subset_sha256",
        "candidate_recipe_sha256",
        "code_revision",
        "candidate_version",
        "seed",
        "initializer_id",
        "initializer_sha256",
    )
    missing = [
        key
        for key in required
        if key not in provenance or provenance[key] is None or provenance[key] == ""
    ]
    if missing:
        raise ValueError(f"Checkpoint provenance eksik: {missing}")

    for key, expected_length in (
        ("dataset_revision", 40),
        ("split_map_sha256", 64),
        ("train_subset_sha256", 64),
        ("candidate_recipe_sha256", 64),
        ("code_revision", 40),
        ("initializer_sha256", 64),
    ):
        value = str(provenance[key]).lower()
        if len(value) != expected_length or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"Geçersiz provenance hash/revision: {key}={provenance[key]!r}")

    payload: Dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        # Training-state resume is supported at epoch boundaries. Model/optimizer/
        # scheduler/scaler/RNG state is restored, but stochastic DataLoader worker
        # augmentation streams are not claimed to be bitwise replayable after a
        # process restart.
        "resume_granularity": "epoch_boundary",
        "last_completed_epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "sampler_epoch": int(sampler_epoch if sampler_epoch is not None else epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "provenance": dict(provenance),
        "metrics": dict(metrics or {}),
    }
    return payload


def save_training_checkpoint(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def _require_same_provenance(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    immutable_keys = (
        "dataset_id",
        "dataset_revision",
        "split_map_sha256",
        "train_subset_sha256",
        "candidate_recipe_sha256",
        "code_revision",
        "candidate_version",
        "seed",
        "initializer_id",
        "initializer_sha256",
    )
    mismatches = {
        key: (saved.get(key), expected.get(key))
        for key in immutable_keys
        if saved.get(key) != expected.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Resume provenance uyuşmuyor: {mismatches}")


def restore_training_checkpoint(
    path: pathlib.Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    expected_provenance: Mapping[str, Any],
    map_location: Any = "cpu",
    restore_rng: bool = True,
) -> Dict[str, Any]:
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint bulunamadı: {path}")

    # This is a trusted checkpoint produced by save_training_checkpoint and
    # contains Python/NumPy RNG state in addition to tensor weights.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "Desteklenmeyen checkpoint schema: "
            f"{payload.get('checkpoint_schema_version')}"
        )
    if payload.get("resume_granularity") != "epoch_boundary":
        raise RuntimeError(
            f"Desteklenmeyen resume granularity: {payload.get('resume_granularity')}"
        )
    saved_provenance = payload.get("provenance") or {}
    _require_same_provenance(saved_provenance, expected_provenance)

    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        state = payload.get("scheduler_state_dict")
        if state is None:
            raise RuntimeError("Resume checkpoint scheduler state içermiyor.")
        scheduler.load_state_dict(state)
    if scaler is not None:
        state = payload.get("scaler_state_dict")
        if state is None:
            raise RuntimeError("Resume checkpoint scaler state içermiyor.")
        scaler.load_state_dict(state)
    if restore_rng:
        restore_rng_state(payload["rng_state"])

    return {
        "epoch": int(payload["epoch"]),
        "last_completed_epoch": int(payload.get("last_completed_epoch", payload["epoch"])),
        "next_epoch": int(payload.get("next_epoch", int(payload["epoch"]) + 1)),
        "resume_granularity": payload["resume_granularity"],
        "global_step": int(payload["global_step"]),
        "sampler_epoch": int(payload.get("sampler_epoch", payload["epoch"])),
        "metrics": payload.get("metrics") or {},
        "provenance": saved_provenance,
    }
