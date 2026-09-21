#!/usr/bin/env python3
"""Prepare a reproducible large-data research plan from the HF accepted manifest.

This command does not start training and does not download the full dataset.
It downloads only lightweight manifest files from one immutable HF revision,
then creates an identity-group-disjoint split and nested diverse train stages.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from src.data.hf_downloader import DEFAULT_REPO_ID, HFDatasetDownloader
from src.data.large_data import (
    build_large_data_plan,
    load_large_data_plan,
    verify_large_data_split_file,
    write_large_data_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare large-data VSR research metadata.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--revision",
        required=True,
        help="Immutable 40-hex Hugging Face dataset commit SHA.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument(
        "--targets-hours",
        default="10,25,50,100",
        help="Comma-separated train scaling stages. Stages larger than available data are omitted.",
    )
    parser.add_argument(
        "--work-dir",
        default=".cache/large_data",
        help="Ignored lightweight manifest cache; full clips are NOT downloaded.",
    )
    parser.add_argument(
        "--output",
        default="research/large_data_plan.json",
        help="Generated plan path.",
    )
    parser.add_argument(
        "--split-output",
        default="data/metadata/split_map_large_data.json",
        help="Generated identity-group-disjoint split map path.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    targets = [float(part) for part in args.targets_hours.split(",") if part.strip()]

    work_dir = pathlib.Path(args.work_dir)
    downloader = HFDatasetDownloader(
        repo_id=args.repo_id,
        target_dir=work_dir,
        revision=args.revision,
        require_pinned_revision=True,
    )
    rows = downloader.read_accepted_manifest()
    if not rows:
        raise RuntimeError("HF accepted manifest boş geldi; plan üretilmedi.")

    plan = build_large_data_plan(
        rows,
        dataset_id=args.repo_id,
        dataset_revision=args.revision,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        targets_hours=targets,
    )
    output = pathlib.Path(args.output)
    split_output = pathlib.Path(args.split_output)
    write_large_data_plan(output, plan)
    split_output.parent.mkdir(parents=True, exist_ok=True)
    split_output.write_text(
        json.dumps(plan.split_map, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    payload = load_large_data_plan(output)
    verify_large_data_split_file(payload, split_output)

    print("=" * 72)
    print("LARGE-DATA RESEARCH PLAN (training başlamadı)")
    print("=" * 72)
    print(f"dataset: {payload['dataset_id']} @ {payload['dataset_revision']}")
    identity_kind = "PROXY — insan audit'i gerekli" if payload["speaker_identity_is_proxy"] else "explicit"
    print(
        f"speaker identity: {payload['speaker_identity_field']} ({identity_kind})"
    )
    print(f"split hash: {payload['split_map_sha256']}")
    print("split summary:")
    for split, summary in payload["split_summary"].items():
        print(f"  {split}: {summary}")
    print("train scaling stages:")
    for stage, summary in payload["staged_summary"].items():
        print(f"  {stage}: {summary}")
    print(f"plan: {output}")
    print(f"split map: {split_output}")
    if payload["speaker_identity_is_proxy"]:
        print(
            "UYARI: split channel/creator proxy kimliği kullanıyor; aynı kanalda birden fazla "
            "konuşmacı varsa gerçek speaker leakage ayrıca audit edilmelidir."
        )
    print("Test split yalnız final/milestone evaluation içindir; araştırma selection'ına açılmaz.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
