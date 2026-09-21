#!/usr/bin/env python3
"""Canonical full-training entrypoint: preflight-only, fail-closed.

Tek kanonik komut:

    python -m src.full_training --manifest full_train_manifest.json

Bu komut `full_train_manifest.json` dosyasını okur, candidate recipe
hash/version/readiness, dataset kimliği+hash, split hash, test-split
karantinası, initializer varlığı ve temiz kod revizyonu kontrollerini
fail-closed yapar; manifestoda dondurulan reçete dışına çıkılamaz
(hiperparametre override bayrağı yoktur), test spliti kullanılmaz ve
ASLA ücretli/gerçek eğitim başlatılmaz (modal importu bile yoktur).

Preflight GEÇERSE bile eğitim otomatik başlamaz: çıktıdaki raporu
inceleyip GPU lansmanını açık insan kararıyla ayrı adımda yaparsınız.
Tarihsel c0.4.0 manifestosu bilerek FAIL verir (bkz.
full_train_manifest.PROVENANCE.md): kanıt olarak korunur, çalıştırılamaz.

Eski, desteklenmeyen çağrı (`python -m src.modal_runner.cloud_train
--manifest ...`) geçersizdir: o modülde `--manifest` bayrağı hiçbir
zaman var olmadı. Kanonik yol bu dosyadır.
"""

import argparse
import pathlib
import sys

from src.experiments.research_governance import preflight_full_train_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical full-training preflight (ücretli işlem başlatmaz)."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="full_train_manifest.json",
        help="Full-training manifestosu dosya yolu",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="configs/research_candidate.yaml",
        help="Kanonik candidate reçetesi yolu",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repo kökü (hash/git kontrolleri burada yapılır)",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = preflight_full_train_manifest(
        manifest_path=args.manifest,
        candidate_path=args.candidate,
        root_dir=args.root,
    )

    print("=" * 64)
    print("FULL-TRAINING PREFLIGHT (ücretli işlem YOK)")
    print("=" * 64)
    for name, check in report.checks.items():
        icon = "PASS" if check["status"] == "PASSED" else "FAIL"
        print(f"  [{icon}] {name:<24} {check.get('detail', '')}")
    print("-" * 64)
    if report.passed:
        print("SONUÇ: PREFLIGHT GEÇTİ — eğitim OTOMATİK BAŞLATILMADI.")
        print("GPU lansmanı ayrı, açık insan kararıyla yapılır.")
        return 0
    print(f"SONUÇ: PREFLIGHT KALDI ({len(report.blockers)} engel):")
    for blocker in report.blockers:
        print(f"  - {blocker}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
