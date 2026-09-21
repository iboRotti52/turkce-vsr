import random

import numpy as np
import pytest
import torch

from src.training_state import (
    build_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)


def _objects():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    return model, optimizer, scheduler


def _provenance(revision="a" * 40):
    return {
        "dataset_id": "avsr-tr-ekip/avsr-tr-dataset",
        "dataset_revision": revision,
        "split_map_sha256": "b" * 64,
        "train_subset_sha256": "d" * 64,
        "candidate_recipe_sha256": "e" * 64,
        "code_revision": "c" * 40,
        "candidate_version": "c0.5.0",
        "seed": 42,
        "initializer_id": "auto-avsr:vsr_trlrs3_base",
        "initializer_sha256": "f" * 64,
    }


def test_training_checkpoint_restores_optimizer_scheduler_and_progress(tmp_path):
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)
    model, optimizer, scheduler = _objects()

    x = torch.randn(4, 3)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=2,
        global_step=123,
        sampler_epoch=2,
        provenance=_provenance(),
        metrics={"val_loss": 1.23},
    )
    path = tmp_path / "resume.pt"
    save_training_checkpoint(path, payload)

    restored_model, restored_optimizer, restored_scheduler = _objects()
    state = restore_training_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=None,
        expected_provenance=_provenance(),
    )

    assert state["epoch"] == 2
    assert state["last_completed_epoch"] == 2
    assert state["next_epoch"] == 3
    assert state["resume_granularity"] == "epoch_boundary"
    assert state["global_step"] == 123
    assert state["metrics"]["val_loss"] == 1.23
    for left, right in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(left, right)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_resume_fails_closed_on_dataset_revision_change(tmp_path):
    model, optimizer, scheduler = _objects()
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=0,
        global_step=0,
        provenance=_provenance("a" * 40),
    )
    path = tmp_path / "resume.pt"
    save_training_checkpoint(path, payload)

    new_model, new_optimizer, new_scheduler = _objects()
    with pytest.raises(RuntimeError, match="Resume provenance uyuşmuyor"):
        restore_training_checkpoint(
            path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            scaler=None,
            expected_provenance=_provenance("d" * 40),
        )


def test_checkpoint_requires_large_data_provenance():
    model, optimizer, scheduler = _objects()
    with pytest.raises(ValueError, match="Checkpoint provenance eksik"):
        build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            epoch=0,
            global_step=0,
            provenance={"dataset_id": "x"},
        )


def test_resume_fails_closed_on_training_stage_change(tmp_path):
    model, optimizer, scheduler = _objects()
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=1,
        global_step=10,
        provenance=_provenance(),
    )
    path = tmp_path / "resume.pt"
    save_training_checkpoint(path, payload)

    changed = _provenance()
    changed["train_subset_sha256"] = "f" * 64

    new_model, new_optimizer, new_scheduler = _objects()
    with pytest.raises(RuntimeError, match="Resume provenance uyuşmuyor"):
        restore_training_checkpoint(
            path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            scaler=None,
            expected_provenance=changed,
        )


def test_seed_zero_is_valid_provenance_and_initializer_hash_is_immutable(tmp_path):
    model, optimizer, scheduler = _objects()
    provenance = _provenance()
    provenance["seed"] = 0
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=0,
        global_step=0,
        provenance=provenance,
    )
    path = tmp_path / "resume.pt"
    save_training_checkpoint(path, payload)

    changed = dict(provenance)
    changed["initializer_sha256"] = "0" * 64
    new_model, new_optimizer, new_scheduler = _objects()
    with pytest.raises(RuntimeError, match="Resume provenance uyuşmuyor"):
        restore_training_checkpoint(
            path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            scaler=None,
            expected_provenance=changed,
        )
