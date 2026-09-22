"""Pinned large-data technical rehearsal on Modal.

This is intentionally NOT a scientific c0.5 experiment. It validates the real
large-data data/loader/model/training/validation path on GPU while the scientific
10h gate remains blocked.

No test split is downloaded or evaluated.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "av>=12.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0.1",
        "jiwer>=3.0.0",
        "rapidfuzz>=3.0.0",
        "huggingface_hub>=0.23.0",
    )
    .add_local_python_source("src")
)

app = modal.App("turkish-vsr-large-data-rehearsal")
data_vol = modal.Volume.from_name("turkish-vsr-large-data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/root/large-data": data_vol},
    timeout=3600,
)
def stage_data_remote(
    revision: str,
    n_train: int = 96,
    n_val: int = 48,
) -> Dict[str, Any]:
    import json
    import pathlib

    from src.data.hf_downloader import HFDatasetDownloader
    from src.data.large_data import build_large_data_plan, write_large_data_plan

    target = pathlib.Path("/root/large-data")
    downloader = HFDatasetDownloader(
        repo_id="avsr-tr-ekip/avsr-tr-dataset",
        target_dir=target,
        revision=revision,
        require_pinned_revision=True,
        max_local_gb=100.0,
    )
    rows = downloader.read_accepted_manifest()
    plan = build_large_data_plan(
        rows,
        dataset_id=downloader.repo_id,
        dataset_revision=revision,
        seed=42,
        targets_hours=(10.0, 25.0, 50.0, 100.0),
    )

    plan_path = downloader.data_dir / "large_data_plan.json"
    split_path = downloader.data_dir / "split_map.json"
    write_large_data_plan(plan_path, plan)
    split_path.write_text(
        json.dumps(plan.split_map, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pinned = HFDatasetDownloader(
        repo_id=downloader.repo_id,
        target_dir=target,
        split_map_path=split_path,
        revision=revision,
        require_pinned_revision=True,
        max_local_gb=100.0,
    )
    train_dirs = pinned.download_pilot(
        n_samples=n_train,
        split="train",
        max_per_video=16,
        max_duration=8.0,
    )
    val_dirs = pinned.download_pilot(
        n_samples=n_val,
        split="val",
        max_per_video=16,
        max_duration=8.0,
    )
    if len(train_dirs) < n_train:
        raise RuntimeError(f"Requested {n_train} train samples, downloaded {len(train_dirs)}")
    if len(val_dirs) < n_val:
        raise RuntimeError(f"Requested {n_val} val samples, downloaded {len(val_dirs)}")

    payload = plan.to_dict()
    data_vol.commit()
    return {
        "dataset_revision": revision,
        "plan_sha256": payload["plan_sha256"],
        "split_map_sha256": payload["split_map_sha256"],
        "speaker_identity_field": payload["speaker_identity_field"],
        "speaker_identity_is_proxy": payload["speaker_identity_is_proxy"],
        "split_summary": payload["split_summary"],
        "available_stages": list(payload["staged_subsets"].keys()),
        "downloaded_train": len(train_dirs),
        "downloaded_val": len(val_dirs),
        "test_downloaded": 0,
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/large-data": data_vol},
    timeout=3600,
)
def train_rehearsal_remote(
    revision: str,
    expected_plan_sha256: str,
    max_steps: int = 20,
) -> Dict[str, Any]:
    import numpy as np
    import pathlib
    import torch

    from src.data.dataset import LipReadingDataset
    from src.data.large_data import load_large_data_plan
    from src.data.large_data_loader import (
        build_large_data_loader,
        set_large_data_loader_epoch,
    )
    from src.evaluation.metrics import evaluate_predictions
    from src.models.vsr_conformer import VSRConformerModel
    from src.vocab.turkish_vocab import (
        BLANK_IDX,
        VOCAB_SIZE,
        ctc_greedy_decode,
    )

    started = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    data_dir = pathlib.Path("/root/large-data") / "_revisions" / revision
    plan_path = data_dir / "large_data_plan.json"
    split_path = data_dir / "split_map.json"
    plan = load_large_data_plan(plan_path)
    if plan["plan_sha256"] != expected_plan_sha256:
        raise RuntimeError("Plan hash drift between staging and GPU rehearsal")

    train_dataset = LipReadingDataset(
        master_dir=data_dir,
        is_train=True,
        split="train",
        split_map_path=split_path,
        crop_size=88,
        max_duration=8.0,
        cache_in_ram=False,
    )
    val_dataset = LipReadingDataset(
        master_dir=data_dir,
        is_train=False,
        split="val",
        split_map_path=split_path,
        crop_size=88,
        max_duration=8.0,
        cache_in_ram=False,
    )
    if not train_dataset.samples or not val_dataset.samples:
        raise RuntimeError(
            f"Downloaded rehearsal dataset empty: train={len(train_dataset)}, val={len(val_dataset)}"
        )

    train_loader = build_large_data_loader(
        train_dataset,
        train=True,
        num_workers=2,
        seed=42,
    )
    val_loader = build_large_data_loader(
        val_dataset,
        train=False,
        num_workers=2,
        batch_size=8,
    )

    device = torch.device("cuda")
    model = VSRConformerModel(
        vocab_size=VOCAB_SIZE,
        d_model=512,
        num_layers=4,
        encoder_type="conformer",
    ).to(device)
    with torch.no_grad():
        if model.ctc_head.bias is not None:
            model.ctc_head.bias[0] -= 0.5

    frontend = [p for n, p in model.named_parameters() if "frontend" in n]
    temporal = [p for n, p in model.named_parameters() if "frontend" not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": frontend, "lr": 7.5e-6},
            {"params": temporal, "lr": 1.5e-4},
        ],
        weight_decay=1e-4,
    )

    initial_loss = None
    train_losses = []
    steps = 0
    epoch = 0
    model.train()
    while steps < max_steps:
        set_large_data_loader_epoch(train_loader, epoch)
        for batch in train_loader:
            videos = (batch.get("videos") if "videos" in batch else batch["video"]).to(device)
            targets = (batch.get("targets") if "targets" in batch else batch["target"]).to(device)
            input_lens = batch["input_lengths"].to(device)
            target_lens = batch["target_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(videos)
            loss = model.compute_loss(
                logits,
                targets,
                input_lens,
                target_lens,
                blank_penalty=0.8,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {steps}: {loss.item()}")
            if initial_loss is None:
                initial_loss = float(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            steps += 1
            if steps >= max_steps:
                break
        epoch += 1

    model.eval()
    val_losses = []
    refs = []
    hyps = []
    blank_count = 0
    total_frames = 0
    raw_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            videos = (batch.get("videos") if "videos" in batch else batch["video"]).to(device)
            targets = (batch.get("targets") if "targets" in batch else batch["target"]).to(device)
            input_lens = batch["input_lengths"].to(device)
            target_lens = batch["target_lengths"].to(device)

            logits = model(videos)
            loss = model.compute_loss(
                logits,
                targets,
                input_lens,
                target_lens,
                blank_penalty=0.8,
            )
            val_losses.append(float(loss.item()))

            transcripts = list(batch["transcripts"])
            refs.extend(transcripts)
            for i, ref in enumerate(transcripts):
                seq_len = int(input_lens[i].item())
                logit_slice = logits[i, :seq_len]
                hyp = ctc_greedy_decode(logit_slice, blank_penalty=1.2)[0]
                hyps.append(hyp)
                top = logit_slice.argmax(dim=-1)
                blank_count += int((top == BLANK_IDX).sum().item())
                total_frames += seq_len
                if len(raw_predictions) < 20:
                    raw_predictions.append({"ref": ref, "hyp": hyp})

    metrics = evaluate_predictions(references=refs, hypotheses=hyps)
    return {
        "run_type": "technical_large_data_modal_rehearsal",
        "scientific_use_for_model_selection": False,
        "scientific_verdict": None,
        "test_split_used": False,
        "dataset_revision": revision,
        "plan_sha256": plan["plan_sha256"],
        "split_map_sha256": plan["split_map_sha256"],
        "speaker_identity_field": plan["speaker_identity_field"],
        "speaker_identity_is_proxy": plan["speaker_identity_is_proxy"],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "max_steps": max_steps,
        "initial_train_loss": initial_loss,
        "final_train_loss": train_losses[-1],
        "mean_train_loss": float(np.mean(train_losses)),
        "val_loss": float(np.mean(val_losses)),
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "blank_ratio": float(blank_count / max(total_frames, 1)),
        "raw_predictions": raw_predictions,
        "gpu_name": torch.cuda.get_device_name(0),
        "duration_sec": round(time.time() - started, 2),
    }


@app.local_entrypoint()
def main(
    revision: str,
    output: str = "artifacts/ops_ld_001_modal_rehearsal.json",
    n_train: int = 96,
    n_val: int = 48,
    max_steps: int = 20,
):
    stage = stage_data_remote.remote(revision, n_train=n_train, n_val=n_val)
    result = train_rehearsal_remote.remote(
        revision,
        expected_plan_sha256=stage["plan_sha256"],
        max_steps=max_steps,
    )
    payload = {"staging": stage, "training": result}
    path = pathlib.Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("OPS_LD_001_RESULT_BEGIN")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    print("OPS_LD_001_RESULT_END")
