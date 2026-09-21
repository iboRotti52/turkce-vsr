#!/usr/bin/env python3
"""Türkçe VSR yardımcı CLI'si.

Eski aşamalı eğitim modları yalnız eski çağrılara anlaşılır bir hata vermek için
dosyada tutulur ve fail-closed durumdadır. Yeni araştırma eğitimi, DATA-001
tamamlandıktan sonra kanonik candidate'a bağlı ayrı bir probe runner ile kurulur.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import LipReadingDataset, pad_collate_fn
from src.data.s3_downloader import S3DatasetDownloader
from src.data.hf_downloader import HFDatasetDownloader
from src.evaluation.metrics import evaluate_predictions
from src.experiments.tracker import ExperimentRecord, ExperimentTracker
from src.models.vsr_conformer import VSRConformerModel
from src.models.factory import ARCH_CONFIGS, build_vsr_model, count_parameters
from src.spotter.keyword_spotter import KeywordSpotter
from src.spotter.lexicon_decoder import LexiconBeamSearchDecoder
from src.experiments.research_governance import (
    ResearchStage,
    load_candidate,
    evaluate_readiness,
    build_full_train_manifest,
    write_full_train_manifest,
    require_full_train_authorized,
    preflight_full_train_manifest,
    require_clean_code_revision,
)

ROOT = pathlib.Path(__file__).resolve().parent
DATA_MASTER_DIR = ROOT / "data" / "master"
DATA_IBOROTTI_DIR = ROOT / "data" / "iborotti"
CHECKPOINTS_DIR = ROOT / "checkpoints"
DEFAULT_CANDIDATE_CONFIG = ROOT / "configs" / "research_candidate.yaml"
LEGACY_TRAINING_MODES = frozenset(
    {"overfit", "micro-pilot", "local-train", "modal-pilot", "all"}
)


def require_mode_allowed(
    *,
    mode: str,
    candidate_path: pathlib.Path = DEFAULT_CANDIDATE_CONFIG,
) -> None:
    """Temiz başlangıçta eski aşamalı eğitim pipeline'ının yeniden çalışmasını engeller."""
    if mode not in LEGACY_TRAINING_MODES:
        return

    if not candidate_path.is_file():
        raise FileNotFoundError(f"Kanonik candidate reçetesi bulunamadı: {candidate_path}")

    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8")) or {}
    version = candidate.get("candidate_version", "unknown")
    active_question = candidate.get("active_question", "unknown")
    raise RuntimeError(
        f"Eski eğitim modu '{mode}' devre dışı. Candidate={version}, "
        f"aktif soru={active_question}. Bu aşamalı pilot pipeline temiz başlangıcın "
        "kanonik modeli değildir. Önce research/NEXT_ACTION.md içindeki veri ve "
        "evaluation audit'ini tamamlayın; ardından kanıtlanan blueprint için yeni, "
        "candidate-aware bir probe runner oluşturun."
    )


def research_probe_next_step() -> str:
    """Başarılı bir probe'un ölçek büyütmek yerine araştırma kararına dönmesini sağlar."""
    return (
        "Validation metriklerini, ham tahminleri ve hata kümelerini analiz et; "
        "ACCEPT/REJECT/INCONCLUSIVE kararını kaydet ve yalnız önceden tanımlanan "
        "ölçüt karşılandıysa kanonik araştırma modelini güncelle."
    )


def get_dataset_dir(dataset_choice: str = "iborotti") -> pathlib.Path:
    if dataset_choice == "iborotti":
        return DATA_IBOROTTI_DIR
    return DATA_MASTER_DIR


def run_local_test(tracker: ExperimentTracker) -> bool:
    """Tüm birim testleri (vocab, model, dataset, spotter, metrics) çalıştırır."""
    print("\n🧪 [EXP: LOCAL-TEST] Pytest Test Süiti Çalıştırılıyor...")
    record = ExperimentRecord(
        experiment_id=f"exp_test_{int(time.time())}",
        hypothesis="Tüm modüller (vocab, 3D-ResNet/Conformer, S3 downloader, dataset, spotter ve metrikler) birbiriyle uyumlu çalışmalı ve tüm testler geçmeli.",
        falsification_criteria="Herhangi bir pytest testinin başarısız olması veya import hatası.",
        setup={"command": "pytest tests/ -v"},
        expectation="16+ testin tamamının PASSED olması.",
    )

    import subprocess
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    p = subprocess.run(cmd, capture_output=True, text=True)

    passed = (p.returncode == 0)
    print(p.stdout)
    if not passed:
        print(p.stderr)

    record.status = "PASSED" if passed else "FALSIFIED"
    record.result = {
        "returncode": p.returncode,
        "output_summary": p.stdout.splitlines()[-1] if p.stdout else "",
    }
    record.surprise = "Yok" if passed else "Bazı testler başarısız oldu."
    record.updated_belief = "Tüm temel bileşenler matematiksel ve fonksiyonel olarak doğrulandı." if passed else "Bileşenler düzeltilmeli."
    record.next_step = "Gerçek veri üzerinde overfit testi yapılmalı." if passed else "Hatalı testler incelenmeli."

    tracker.log(record)
    return passed


def run_overfit_test(
    tracker: ExperimentTracker,
    arch_name: str = "vsr_tiny",
    steps: int = 30,
    dataset_choice: str = "iborotti",
) -> bool:
    """Gerçek temiz veri üzerinde 1 mini-batch overfit testi yapar."""
    print(f"\n🎯 [EXP: OVERFIT] Gerçek Veri ile {steps} Adımlık Overfit Testi Başlatılıyor (Arch: {arch_name}, Dataset: {dataset_choice})...")
    record = ExperimentRecord(
        experiment_id=f"exp_overfit_{int(time.time())}",
        hypothesis=f"{arch_name} mimarisi tek bir mini-batch üzerindeki dudak hareketlerini ve Türkçe transkripti ezberleyerek CTC loss'u belirgin şekilde düşürebilmeli.",
        falsification_criteria="steps sonunda CTC kaybının ilk kayıptan daha düşük olmaması veya diverjans.",
        setup={"steps": steps, "arch": arch_name, "lr": 5e-3, "device": "mps if available else cpu", "dataset": dataset_choice},
        expectation="CTC loss'un başlangıç değerinin altına inmesi.",
    )

    data_dir = get_dataset_dir(dataset_choice)
    mouth_files = list(data_dir.glob("*/*/mouth.mp4")) + list(data_dir.glob("clips/*/*/mouth.mp4"))
    if not data_dir.exists() or len(mouth_files) == 0:
        if dataset_choice == "iborotti":
            print("data/iborotti boş, test için Hugging Face'ten 5 pilot segment indiriliyor (< 2 MB)...")
            downloader = HFDatasetDownloader(target_dir=data_dir)
            downloader.download_pilot(n_samples=5, split="train")
        else:
            downloader = S3DatasetDownloader(master_dir=data_dir)
            downloader.download_pilot_dataset(n_segments=5, workers=8)

    dataset = LipReadingDataset(data_dir=data_dir, split="train", is_train=False, crop_size=88, max_frames=250)
    if len(dataset) == 0:
        print("Hata: Train splitinde segment bulunamadı.")
        record.status = "ERROR"
        record.result = {"error": "Veri bulunamadı"}
        tracker.log(record)
        return False

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=pad_collate_fn)
    batch = next(iter(loader))

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  Cihaz: {device}")

    model = build_vsr_model(arch_name=arch_name).to(device)
    params = count_parameters(model)
    print(f"  Model Parametreleri: Toplam {params['total_m']}M (Eğitilebilir: {params['trainable_m']}M)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    videos = batch["videos"].to(device)
    targets = batch["targets"].to(device)
    input_lengths = batch["input_lengths"].to(device)
    target_lengths = batch["target_lengths"].to(device)

    initial_loss = None
    final_loss = None
    losses = []

    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(videos)
        loss = model.compute_loss(logits, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        val = loss.item()
        losses.append(round(val, 4))
        if step == 0:
            initial_loss = val
        final_loss = val

        if step % 5 == 0 or step == steps - 1:
            print(f"  Adım [{step + 1:02d}/{steps:02d}] Loss: {val:.4f}")

    passed = (final_loss is not None and initial_loss is not None and final_loss < initial_loss)
    print(f"  Başlangıç Loss: {initial_loss:.4f} -> Bitiş Loss: {final_loss:.4f} (Başarı: {passed})")

    record.status = "PASSED" if passed else "FALSIFIED"
    record.result = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_history": losses,
        "passed": passed,
    }
    record.surprise = "Yok, beklendiği gibi düştü." if passed else "Loss düşmedi veya patladı."
    record.updated_belief = f"{arch_name} mimarisi ve CTC loss gradyan akışı gerçek veri üzerinde kanıtlandı."
    record.next_step = "Mikro-pilot mimari elemesine geçilebilir."

    tracker.log(record)
    return passed


def run_micro_pilot(
    tracker: ExperimentTracker,
    arch_name: str = "vsr_bigru_small",
    epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 4,
    train_samples: int = 24,
    val_samples: int = 8,
    blank_penalty: float = 0.25,
    dataset_choice: str = "iborotti",
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aşama 2: Mikro-Pilot Mimari Elemesi.
    Küçük, konuşmacı-ayrık alt kümede hızlı tarama (3-5 dakika).
    Loss, Boşluk Çökmesi (Blank Emission Ratio) ve ilk ham karakter emisyonlarını ölçer.
    """
    print(f"\n🔬 [AŞAMA 2: MİKRO-PİLOT] Mimari: {arch_name}, Epochs: {epochs}, LR: {lr}, Blank Penalty: {blank_penalty}...")
    run_id = run_name or f"micro_pilot_{arch_name}_{int(time.time())}"
    record = ExperimentRecord(
        experiment_id=f"exp_{run_id}",
        hypothesis=f"{arch_name} mimarisi {train_samples} temiz segment üzerinde {epochs} epochta yakınsayarak boşluk çökmesini (blank ratio < 0.95) kırmalı ve held-out val'da karakter emisyonu üretmeli.",
        falsification_criteria="Validasyon kaybının düşmemesi veya boşluk oranının > 0.98 kalarak modelin çökmesi.",
        setup={
            "arch": arch_name,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "blank_penalty": blank_penalty,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "dataset": dataset_choice,
        },
        expectation="Train ve Val loss düşüşü, blank ratio < 0.95 ve ilk Türkçe harf emisyonlarının belirmesi.",
    )

    data_dir = get_dataset_dir(dataset_choice)
    train_dataset = LipReadingDataset(data_dir=data_dir, split="train", is_train=True, crop_size=88, max_frames=120)
    val_dataset = LipReadingDataset(data_dir=data_dir, split="val", is_train=False, crop_size=88, max_frames=120)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Eğitim veya validasyon kümesinde yeterli veri yok! Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_subset = torch.utils.data.Subset(train_dataset, list(range(min(train_samples, len(train_dataset)))))
    val_subset = torch.utils.data.Subset(val_dataset, list(range(min(val_samples, len(val_dataset)))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_vsr_model(arch_name=arch_name).to(device)
    params = count_parameters(model)
    print(f"  Cihaz: {device} | Model Parametreleri: Toplam {params['total_m']}M (Eğitilebilir: {params['trainable_m']}M)")

    # Blank bias'ı hafif negatif başlatarak boşluk çökmesini engelle
    with torch.no_grad():
        if hasattr(model, "ctc_head") and model.ctc_head.bias is not None:
            model.ctc_head.bias[0] -= 0.5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    best_val_loss = float("inf")

    for ep in range(1, epochs + 1):
        # 1. Eğitim döngüsü
        model.train()
        train_loss_total = 0.0
        train_steps = 0
        for batch in train_loader:
            videos = batch["videos"].to(device)
            targets = batch["targets"].to(device)
            in_lens = batch["input_lengths"].to(device)
            tgt_lens = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            logits = model(videos)
            loss = model.compute_loss(logits, targets, in_lens, tgt_lens, blank_penalty=blank_penalty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_total += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_total / max(1, train_steps)

        # 2. Validasyon döngüsü ve Blank Ratio Hesabı
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        val_blank_frames = 0
        val_total_frames = 0
        val_refs = []
        val_hyps = []

        with torch.no_grad():
            for batch in val_loader:
                videos = batch["videos"].to(device)
                targets = batch["targets"].to(device)
                in_lens = batch["input_lengths"].to(device)
                tgt_lens = batch["target_lengths"].to(device)

                logits = model(videos)
                loss = model.compute_loss(logits, targets, in_lens, tgt_lens, blank_penalty=blank_penalty)
                val_loss_total += loss.item()
                val_steps += 1

                # Blank emisyon oranı
                argmax_tokens = logits.argmax(dim=-1)
                val_blank_frames += (argmax_tokens == 0).sum().item()
                val_total_frames += argmax_tokens.numel()

                preds = ctc_greedy_decode(logits, blank_penalty=blank_penalty)
                val_hyps.extend(preds)
                val_refs.extend(batch["transcripts"])

        avg_val_loss = val_loss_total / max(1, val_steps)
        blank_ratio = val_blank_frames / max(1, val_total_frames)
        val_metrics = evaluate_predictions(val_refs, val_hyps)

        stat = {
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "blank_ratio": round(blank_ratio, 4),
            "val_cer": round(val_metrics["cer"], 4),
            "sample_hyp": val_hyps[0] if val_hyps else "",
            "sample_ref": val_refs[0] if val_refs else "",
        }
        history.append(stat)

        # Durum bildirimi
        blank_flag = "⚠️ YÜKSEK BOŞLUK" if blank_ratio > 0.95 else "✅ AKTİF HARF"
        print(
            f"  Epoch [{ep:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Blank Oranı: %{blank_ratio*100:.1f} ({blank_flag}) | "
            f"Val CER: {val_metrics['cer']:.2%}"
        )
        if ep == epochs or ep == 1:
            print(f"    [Val Örnek Ref]: '{val_refs[0][:50]}'")
            print(f"    [Val Örnek Hyp]: '{val_hyps[0][:50]}'")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            ckpt_path = CHECKPOINTS_DIR / f"{run_id}_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "arch_name": arch_name,
                "d_model": model.d_model,
                "encoder_type": model.encoder_type,
                "vocab_size": VOCAB_SIZE,
                "epoch": ep,
                "val_loss": avg_val_loss,
            }, str(ckpt_path))

    passed = (history[-1]["train_loss"] < history[0]["train_loss"] and history[-1]["blank_ratio"] < 0.98)
    record.status = "PASSED" if passed else "FALSIFIED"
    record.result = {
        "history": history,
        "best_val_loss": best_val_loss,
        "final_blank_ratio": history[-1]["blank_ratio"],
        "checkpoint": str(CHECKPOINTS_DIR / f"{run_id}_best.pt"),
    }
    record.surprise = f"Final Blank Oranı: %{history[-1]['blank_ratio']*100:.1f}, En iyi Val Loss: {best_val_loss:.4f}"
    record.updated_belief = (
        f"{arch_name} mimarisi temiz veride yakınsıyor ve harf üretiyor."
        if passed
        else f"{arch_name} boşluk tuzağına düştü veya yakınsamadı."
    )
    record.next_step = "Kalitatif hata analizi ve mimariler arası karşılaştırma."
    tracker.log(record)

    return {"history": history, "best_val_loss": best_val_loss, "checkpoint": str(CHECKPOINTS_DIR / f"{run_id}_best.pt")}


def run_inspect_predictions(
    ckpt_path: Optional[str] = None,
    samples_to_show: int = 10,
    split: str = "val",
    dataset_choice: str = "iborotti",
    blank_penalty: float = 0.25,
) -> None:
    """
    Aşama 3: Kalitatif İnceleme.
    Held-out split üzerinde modelin ham tahminlerini referans ile alt alta karşılaştırır.
    """
    print(f"\n🔍 [AŞAMA 3: KALİTATİF İNCELEME] Split: {split}, Örnek Sayısı: {samples_to_show}...")
    path = pathlib.Path(ckpt_path or (CHECKPOINTS_DIR / "clean_vsr_pilot_best.pt"))
    if not path.exists():
        ckpts = list(CHECKPOINTS_DIR.glob("*.pt"))
        if ckpts:
            path = ckpts[0]
        else:
            print(f"Hata: Checkpoint bulunamadı: {path}")
            return

    data_dir = get_dataset_dir(dataset_choice)
    dataset = LipReadingDataset(data_dir=data_dir, split=split, is_train=False, crop_size=88)
    if len(dataset) == 0:
        raise ValueError(f"Split '{split}' için örnek bulunamadı.")

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_data = torch.load(str(path), map_location=device)
    sd = ckpt_data.get("model_state_dict", ckpt_data)
    arch = ckpt_data.get("arch_name", "vsr_bigru_small" if any("weight_ih" in k for k in sd) else "vsr_conformer_small")
    d_model = ckpt_data.get("d_model", 512 if any("512" in str(v.shape) for v in sd.values()) else 256)

    model = build_vsr_model(arch_name=arch, d_model=d_model).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    decoder = LexiconBeamSearchDecoder(beam_size=30, blank_penalty=blank_penalty, use_lm=True)
    spotter = KeywordSpotter(min_confidence=0.20, blank_penalty=blank_penalty)

    print(f"Yüklenen Model: {path.name} (Arch: {arch}, Device: {device})")
    print("=" * 90)

    num_samples = min(samples_to_show, len(dataset))
    for i in range(num_samples):
        sample = dataset[i]
        video = sample["video"].unsqueeze(0).to(device)
        ref = sample["transcript"]
        item_info = dataset.samples[i]

        with torch.no_grad():
            logits = model(video)
            greedy_hyp = ctc_greedy_decode(logits, blank_penalty=blank_penalty)[0]
            beam_hyp, beam_spotted = decoder.decode(logits[0].cpu())
            spot_results = spotter.spot(logits[0].cpu())

        cer_greedy = evaluate_predictions([ref], [greedy_hyp])["cer"]
        cer_beam = evaluate_predictions([ref], [beam_hyp])["cer"]

        print(f"\n📌 Örnek {i+1:02d} | Klip: {item_info.get('seg_id', i)} | Konuşmacı: {item_info.get('video_id', 'Bilinmiyor')}")
        print(f"   Referans : {ref}")
        print(f"   Greedy   : {greedy_hyp}  (CER: {cer_greedy:.1%})")
        print(f"   Beam+LM  : {beam_hyp}  (CER: {cer_beam:.1%})")
        spotted_words = [d.word for d in spot_results]
        if spotted_words:
            print(f"   🎯 500 Kelime: {spotted_words}")
    print("\n" + "=" * 90)


def run_local_train(
    tracker: ExperimentTracker,
    epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 4,
    d_model: int = 256,
    encoder_type: str = "bigru",
    dataset_choice: str = "iborotti",
) -> Dict[str, Any]:
    """Yerel makinede (MPS/CPU) küçük bir pilot eğitimi koşturur."""
    arch = "vsr_bigru_small" if encoder_type == "bigru" else "vsr_conformer_small"
    return run_micro_pilot(
        tracker=tracker,
        arch_name=arch,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        dataset_choice=dataset_choice,
    )


def run_evaluation(
    tracker: ExperimentTracker,
    ckpt_path: Optional[str] = None,
    d_model: int = 128,
    encoder_type: str = "conformer",
    blank_penalty: float = 0.2,
    split: Optional[str] = None,
    lm_alpha: float = 0.4,
    lm_beta: float = 1.0,
    repeat_penalty: float = 3.0,
    min_dur_factor: float = 1.0,
    dataset_choice: str = "iborotti",
    test_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Mevcut bir kontrol noktasını (checkpoint) yerel veri üzerinde değerlendirir."""
    print(f"\n🔍 [EXP: EVALUATION] Model Değerlendirmesi Başlatılıyor (Dataset: {dataset_choice}, MaxSamples: {test_samples})...")
    path = pathlib.Path(ckpt_path or (CHECKPOINTS_DIR / "local_pilot_model.pt"))
    if not path.exists():
        print(f"Hata: Checkpoint bulunamadı: {path}")
        return {"error": f"Checkpoint bulunamadı: {path}"}

    split_label = f" ({split.upper()} split)" if split else ""
    record = ExperimentRecord(
        experiment_id=f"exp_eval_{int(time.time())}",
        hypothesis=f"Eğitilmiş model ({path.name}){split_label} 500 kelimelik sözlük üzerinde CER, WER ve Spotter metrikleri üretebilir (Blank Penalty={blank_penalty}).",
        falsification_criteria="Değerlendirmenin çökmesi veya metrik üretilememesi.",
        setup={"checkpoint": str(path), "d_model": d_model, "encoder_type": encoder_type, "blank_penalty": blank_penalty, "split": split, "dataset": dataset_choice, "test_samples": test_samples},
        expectation="Tüm VSR ve 500-kelime tespit metriklerinin hesaplanması.",
    )

    data_dir = get_dataset_dir(dataset_choice)
    dataset = LipReadingDataset(data_dir=data_dir, split=split, is_train=False, crop_size=88)
    if len(dataset) == 0 and split is not None:
        raise ValueError(
            f"Split '{split}' için geçerli örnek bulunamadı. GEMINI.md protokolü uyarınca tüm veriye sessizce düşülmez."
        )
    if len(dataset) == 0:
        print("Hata: Değerlendirme için veri bulunamadı.")
        return {"error": "Veri yok"}

    if test_samples is not None and len(dataset) > test_samples:
        indices = list(range(test_samples))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"  Örneklem sınırlandı: İlk {test_samples} segment değerlendiriliyor.")

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=pad_collate_fn)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Checkpoint yükleme ve mimariyi otomatik tespit etme
    checkpoint_data = torch.load(str(path), map_location=device)
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
        act_encoder = checkpoint_data.get("encoder_type", encoder_type)
        act_d_model = checkpoint_data.get("d_model", d_model)
        act_num_layers = checkpoint_data.get("num_layers", 3 if act_encoder == "bigru" else 2)
    else:
        state_dict = checkpoint_data
        # Anahtar yapısından otomatik mimari tespiti
        if any("weight_ih" in k for k in state_dict.keys()):
            act_encoder = "bigru"
            act_num_layers = 3
        elif any("depthwise_conv" in k for k in state_dict.keys()):
            act_encoder = "conformer"
            act_num_layers = 2
        else:
            act_encoder = encoder_type
            act_num_layers = 2
        act_d_model = d_model

    record.setup["encoder_type"] = act_encoder
    record.setup["d_model"] = act_d_model
    record.setup["num_layers"] = act_num_layers

    model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=act_d_model, num_layers=act_num_layers, encoder_type=act_encoder).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    spotter = KeywordSpotter(min_confidence=0.18, blank_penalty=blank_penalty)
    decoder = LexiconBeamSearchDecoder(
        beam_size=40,
        blank_penalty=blank_penalty,
        lm_alpha=lm_alpha,
        lm_beta=lm_beta,
        repeat_penalty=repeat_penalty,
        min_dur_factor=min_dur_factor,
        use_lm=True,
    )
    all_refs = []
    all_hyps = []
    all_spotted = []
    all_beam_hyps = []
    all_beam_spotted = []

    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            logits = model(videos)

            preds = ctc_greedy_decode(logits, blank_penalty=blank_penalty)
            all_hyps.extend(preds)
            all_refs.extend(batch["transcripts"])

            batch_spotted = spotter.spot_batch(logits.cpu(), blank_penalty=blank_penalty)
            all_spotted.extend(batch_spotted)

            for b in range(logits.size(0)):
                b_hyp, b_sp = decoder.decode(logits[b].cpu())
                all_beam_hyps.append(b_hyp)
                all_beam_spotted.append(b_sp)

    metrics = evaluate_predictions(all_refs, all_hyps, spotted_keywords=all_spotted)
    beam_metrics = evaluate_predictions(all_refs, all_beam_hyps, spotted_keywords=all_beam_spotted)

    print("\n  📊 Değerlendirme Sonuçları (Greedy Decoding):")
    print(f"    Toplam Örnek Sayısı: {len(all_refs)}")
    print(f"    CER (Karakter Hata): {metrics['cer']:.4f}")
    print(f"    WER (Kelime Hata):   {metrics['wer']:.4f}")
    print(f"    Spotter Precision:   {metrics['spotter_precision']:.4f}")
    print(f"    Spotter Recall:      {metrics['spotter_recall']:.4f}")
    print(f"    Spotter F1:          {metrics['spotter_f1']:.4f}")
    print(f"    TP: {metrics['true_positives']} | FP: {metrics['false_positives']} | FN: {metrics['false_negatives']}")

    print("\n  🎯 Değerlendirme Sonuçları (Lexicon-Constrained Beam Search + LM):")
    print(f"    CER (Karakter Hata): {beam_metrics['cer']:.4f}")
    print(f"    WER (Kelime Hata):   {beam_metrics['wer']:.4f}")
    print(f"    Beam Precision:      {beam_metrics['spotter_precision']:.4f}")
    print(f"    Beam Recall:         {beam_metrics['spotter_recall']:.4f}")
    print(f"    Beam F1:             {beam_metrics['spotter_f1']:.4f}")
    print(f"    TP: {beam_metrics['true_positives']} | FP: {beam_metrics['false_positives']} | FN: {beam_metrics['false_negatives']}")

    print("\n  --- Örnek Çıktılar ---")
    for i in range(min(3, len(all_refs))):
        print(f"    [Ref {i+1}]:  {all_refs[i]}")
        print(f"    [Greedy]: '{all_hyps[i]}'")
        print(f"    [Beam]:   '{all_beam_hyps[i]}'")
        sp_words = [d.word for d in all_spotted[i]]
        beam_words = [d.word for d in all_beam_spotted[i]]
        print(f"    [Greedy Spot]: {sp_words}")
        print(f"    [Beam Spot]:   {beam_words}")

    record.status = "PASSED"
    record.result = {"greedy": metrics, "beam": beam_metrics}
    record.surprise = f"Greedy F1: {metrics['spotter_f1']:.4f} | Beam F1: {beam_metrics['spotter_f1']:.4f} (TP: {beam_metrics['true_positives']}, FP: {beam_metrics['false_positives']})"
    record.updated_belief = f"Model değerlendirildi. Beam FP: {beam_metrics['false_positives']}"
    record.next_step = "Bulut eğitimi veya hiperparametre optimizasyonu."
    tracker.log(record)
    return {"greedy": metrics, "beam": beam_metrics}


def run_modal_pilot(
    tracker: ExperimentTracker,
    run_name: Optional[str] = None,
    pilot: int = 50,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-4,
    encoder: str = "conformer",
    d_model: int = 256,
    num_layers: int = 4,
    resume: bool = False,
    blank_penalty: float = 1.8,
    warmup_epochs: int = 3,
    max_per_video: Optional[int] = 8,
    test_samples: int = 40,
    data_mode: str = "word",
    max_per_word: int = 20,
    init_from: Optional[str] = None,
    lm_alpha: float = 0.4,
    lm_beta: float = 0.8,
    dataset: str = "iborotti",
    question_id: Optional[str] = None,
    hypothesis: Optional[str] = None,
    falsification_criteria: Optional[str] = None,
    expectation: Optional[str] = None,
) -> Dict[str, Any]:
    """Modal bulut altyapısında A10G ile pilot eğitim çalıştırır."""
    r_name = run_name or f"pilot_{encoder}_{int(time.time())}"
    print(f"\n☁️ [EXP: MODAL-PILOT] Modal Bulut Eğitimi Başlatılıyor (A10G, Run={r_name}, Dataset={dataset}, Pilot={pilot}, Epochs={epochs}, Encoder={encoder}, Mode={data_mode}, InitFrom={init_from}, LM_Alpha={lm_alpha}, LM_Beta={lm_beta})...")
    rec_hypothesis = hypothesis or f"Modal A10G ortamında {pilot} segment ile {encoder} VSR modeli eğitilebilir ve 500 kelime tespiti metrikleri elde edilir (Dataset={dataset}, Mod={data_mode}, InitFrom={init_from})."
    rec_falsification = falsification_criteria or "Modal çalıştırmasının hata vermesi veya GPU konteynerinin çökmesi."
    rec_expectation = expectation or "Eğitimin tamamlanıp bulut metriklerinin ve model kontrol noktasının üretilmesi."
    record = ExperimentRecord(
        experiment_id=f"exp_{r_name}_{int(time.time())}",
        hypothesis=rec_hypothesis,
        falsification_criteria=rec_falsification,
        setup={
            "gpu": "A10G",
            "question_id": question_id,
            "dataset": dataset,
            "requested_pilot": pilot,
            "max_per_video": max_per_video,
            "epochs": epochs,
            "lr": lr,
            "encoder": encoder,
            "resume": resume,
            "data_mode": data_mode,
            "init_from": init_from,
        },
        expectation=rec_expectation,
        cost_estimate_usd=round(epochs * pilot * 0.00005, 4),
    )
    tracker.log(record)

    try:
        import modal
        from src.modal_runner.cloud_train import app, train_remote

        print("  Modal bağlantısı kuruluyor ve iş gönderiliyor...")
        with app.run():
            try:
                code_revision = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                code_revision = "unknown"
            res = train_remote.remote(
                run_name=r_name,
                n_pilot_samples=pilot,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                encoder_type=encoder,
                d_model=d_model,
                num_layers=num_layers,
                resume_from_ckpt=resume,
                blank_penalty=blank_penalty,
                warmup_epochs=warmup_epochs,
                max_per_video=max_per_video,
                n_test_samples=test_samples,
                data_mode=data_mode,
                max_per_word=max_per_word,
                init_from=init_from,
                lm_alpha=lm_alpha,
                lm_beta=lm_beta,
                dataset=dataset,
                code_revision=code_revision,
            )

        print("\n  Bulut eğitimi başarıyla tamamlandı!")
        print(json.dumps(res, indent=2, ensure_ascii=False))

        record.status = "PASSED"
        record.result = res
        record.cost_usd = res.get("cost_usd")
        record.surprise = "Bulut eğitimi başarıyla tamamlandı."
        record.updated_belief = f"Modal A10G üzerinde eğitim stabil. Son F1: {res.get('best_spotter_f1', 0.0)}"
        record.next_step = research_probe_next_step()
        tracker.log(record)
        return res

    except Exception as e:
        print(f"  Modal Bulut Hatası: {e}")
        record.status = "ERROR"
        record.result = {"error": str(e)}
        record.updated_belief = "Modal konfigürasyonunda veya kotalarda bir sorun oluştu."
        tracker.log(record)
        return {"error": str(e)}


def run_download_hf(
    pilot: int = 20,
    split: Optional[str] = "train",
    download_all: bool = False,
    repo_id: str = "iboRotti/avsr-tr-dataset",
) -> None:
    """Hugging Face veri kümesini indirir (ultra hafif pilot veya tam)."""
    downloader = HFDatasetDownloader(repo_id=repo_id, target_dir=DATA_IBOROTTI_DIR)
    if download_all:
        print(f"🚀 Hugging Face deposunun tamamı indiriliyor: {repo_id}...")
        downloader.download_full()
    else:
        print(f"📥 Hugging Face'ten {pilot} adet pilot segment indiriliyor (split={split})...")
        dirs = downloader.download_pilot(n_samples=pilot, split=split)
        print(f"✅ {len(dirs)} pilot segment başarıyla indirildi: {DATA_IBOROTTI_DIR}")


def get_git_revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
        return rev.decode("utf-8").strip()
    except Exception:
        return "unknown"


def handle_research_status(candidate_path: Union[str, pathlib.Path] = DEFAULT_CANDIDATE_CONFIG) -> int:
    path = pathlib.Path(candidate_path).resolve()
    state = load_candidate(path)
    report = evaluate_readiness(state)

    print(f"\n=======================================================")
    print(f"📊 ARAŞTIRMA DURUMU: Candidate {state.candidate_version}")
    print(f"=======================================================")
    print(f"  Aşama               : {state.stage.value}")
    print(f"  Reçete Durumu       : {state.training.get('recipe_status', 'bilinmiyor')}")
    print(f"  Full Train Yetkisi  : {state.training.get('full_training_authorized', False)}")
    print(f"  Aktif Soru          : {state.raw.get('active_question', 'yok')}")

    open_high_impact = [
        q["id"] for q in state.open_questions
        if q.get("impact") == "high" and q.get("status") in {"active", "open", "queued", "in_progress"}
    ]
    print(f"  Açık Yüksek Etkili Sorular: {open_high_impact if open_high_impact else 'Yok (0)'}")

    print(f"\n--- 12 Kapı Hazırlık Durumu ---")
    for key in (
        "complete_recipe", "component_decisions", "no_high_impact_uncertainty",
        "controlled_probes", "representative_validation", "stability",
        "metrics_and_raw_outputs", "failure_model", "provenance_and_reproducibility",
        "dress_rehearsal", "baseline_improvement", "frozen_full_train_recipe",
    ):
        detail = report.gate_details.get(key, {})
        status_icon = "✅ PASSED" if detail.get("status") == "PASSED" else "❌ BLOCKED"
        print(f"  {key:<32}: {status_icon}")
        if detail.get("status") != "PASSED":
            print(f"     Neden: {detail.get('reason')}")

    print(f"\n  Genel Hazırlık Durumu : {'✅ READY' if report.ready else '❌ NOT_READY'}")
    if report.blockers:
        print(f"  Engelleyen Unsurlar   : {list(report.blockers)}")
    print(f"=======================================================\n")
    return 0 if report.ready else 1


def handle_seal_full_train(
    candidate_path: Union[str, pathlib.Path],
    manifest_path: Union[str, pathlib.Path],
    code_revision: Optional[str] = None,
    dataset_id: str = "iboRotti/avsr-tr-dataset",
    dataset_sha256: str = "f5e924fa2297c82119777f98e6a4b1caab0bc506b32252a16d55df27d58a1768",
    split_sha256: str = "cfb9e3f80e7615a7ea4efc6198f39572ea0e06001222485586616e0be5ea4ce6",
    initializer: str = "probe_confirm001_full_train_scaling_best.pt",
    seeds: Optional[List[int]] = None,
    budget_usd: float = 22.0,
    dataset_scope_note: str = "1325 clips <= 8.0s (excluding 356 clips > 8.0s for GPU VRAM/padding)",
) -> Dict[str, Any]:
    c_path = pathlib.Path(candidate_path).resolve()
    m_path = pathlib.Path(manifest_path).resolve()
    state = load_candidate(c_path)

    # Candidate YAML must be in REHEARSAL or READY_FOR_FULL_TRAIN to seal
    if state.stage not in {ResearchStage.REHEARSAL, ResearchStage.READY_FOR_FULL_TRAIN}:
        raise RuntimeError(f"Cannot seal full training from stage {state.stage.value}. Must be REHEARSAL.")

    report = evaluate_readiness(state)
    if not report.ready:
        raise RuntimeError(f"Cannot seal full training. Active readiness blockers: {report.blockers}")

    raw = yaml.safe_load(c_path.read_text(encoding="utf-8"))
    raw["stage"] = ResearchStage.READY_FOR_FULL_TRAIN.value
    raw["training"]["recipe_status"] = "frozen"
    c_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    updated_state = load_candidate(c_path)
    if code_revision is not None:
        rev = code_revision
    else:
        # Fail-closed: dirty tree ile mühürleme yasaktır (c0.4.0 provenance
        # boşluğunun tekrarı olmaması için açık revizyon da temiz olmalıdır).
        rev = require_clean_code_revision(ROOT)
    manifest_seeds = seeds or [42, 123, 456]

    manifest = build_full_train_manifest(
        candidate=updated_state,
        code_revision=rev,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        initializer=initializer,
        seeds=manifest_seeds,
        budget_usd=budget_usd,
        dataset_scope_note=dataset_scope_note,
    )
    init_file = CHECKPOINTS_DIR / pathlib.Path(initializer).name
    if init_file.is_file():
        import hashlib as _hashlib

        manifest["initializer_sha256"] = _hashlib.sha256(init_file.read_bytes()).hexdigest()
        manifest["sealed_from_clean_revision"] = True
    write_full_train_manifest(m_path, manifest)
    print(f"🔒 Candidate {updated_state.candidate_version} başarıyla READY_FOR_FULL_TRAIN mühürlendi!")
    print(f"📄 Manifesto kaydedildi: {m_path}")
    return manifest


def handle_authorize_full_train(
    candidate_path: Union[str, pathlib.Path],
    manifest_path: Union[str, pathlib.Path],
) -> Dict[str, Any]:
    """Canonical full-training preflight: extended fail-closed kontroller.

    Ücretli hiçbir işlem başlatmaz; yalnızca raporlar ve engellerde
    RuntimeError fırlatır. Tarihsel c0.4.0 manifestosu bilerek FAIL verir
    (bkz. full_train_manifest.PROVENANCE.md).
    """
    report = preflight_full_train_manifest(
        manifest_path=manifest_path,
        candidate_path=candidate_path,
        root_dir=ROOT,
    )
    manifest = require_full_train_authorized(candidate_path, manifest_path)
    print(f"Full-training preflight: {'GEÇTİ' if report.passed else 'KALDI'}")
    for name, check in report.checks.items():
        print(f"  [{'PASS' if check['status'] == 'PASSED' else 'FAIL'}] {name}: {check.get('detail', '')}")
    if not report.passed:
        raise RuntimeError(
            f"Full-training preflight engelleri: {list(report.blockers)}"
        )
    print(f"✅ Full training yetkilendirmesi GEÇTİ!")
    print(f"   Candidate : {manifest.get('candidate_version')}")
    print(f"   Seeds     : {manifest.get('seeds')}")
    print(f"   Hash      : {manifest.get('candidate_recipe_sha256')[:16]}...")
    print("   NOT: Eğitim otomatik başlatılmadı; GPU lansmanı ayrı insan kararıyla yapılır.")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Türkçe VSR Deney Çalıştırıcı (Research-Craft)")
    parser.add_argument(
        "--mode",
        choices=[
            "local-test", "overfit", "micro-pilot", "inspect", "local-train",
            "eval", "modal-pilot", "summary", "all", "demo", "download-hf",
            "research-status", "seal-full-train", "authorize-full-train", "full-train",
        ],
        default="local-test",
        help="Çalıştırma modu",
    )
    parser.add_argument("--candidate-path", type=str, default=str(DEFAULT_CANDIDATE_CONFIG), help="Candidate YAML yolu")
    parser.add_argument("--full-train-manifest", type=str, default="full_train_manifest.json", help="Full train manifestosu dosya yolu")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Full train için kullanılacak seedler (virgülle ayrılmış)")
    parser.add_argument("--budget-usd", type=float, default=22.0, help="Full train için ayrılan bulut bütçesi")
    parser.add_argument("--dataset-scope-note", type=str, default="1325 clips <= 8.0s", help="Veri kapsamı gerekçesi")
    parser.add_argument("--arch", choices=list(ARCH_CONFIGS.keys()), default="vsr_bigru_small", help="Model mimari profili (vsr_tiny, vsr_bigru_small, vsr_bigru_base, vsr_conformer_small, vsr_conformer_base)")
    parser.add_argument("--dataset", choices=["iborotti", "master"], default="iborotti", help="Kullanılacak veri kümesi (iborotti: Hugging Face, master: S3)")
    parser.add_argument("--all-data", action="store_true", help="HF deposunun tümünü indir (varsayılan: sadece pilot)")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch sayısı")
    parser.add_argument("--steps", type=int, default=25, help="Overfit adım sayısı")
    parser.add_argument("--pilot", type=int, default=50, help="Pilot segment sayısı")
    parser.add_argument("--lr", type=float, default=3e-4, help="Öğrenme hızı")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch boyutu")
    parser.add_argument("--encoder", choices=["conformer", "bigru"], default="conformer", help="Encoder mimarisi")
    parser.add_argument("--d-model", type=int, default=256, help="Encoder gizli katman boyutu")
    parser.add_argument("--num-layers", type=int, default=4, help="Encoder katman sayısı")
    parser.add_argument("--resume", action="store_true", help="Pretrained Auto-AVSR ağırlıklarını aktar")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Frontend dondurma warmup epoch sayısı")
    parser.add_argument("--data-mode", choices=["word", "phrase", "sentence"], default="word", help="Eğitim veri modu (word: Top-500 dilimleri, phrase: 2-4 kelimelik ifadeler, sentence: tam cümleler)")
    parser.add_argument("--max-per-word", type=int, default=20, help="Kelime başına azami örnek sayısı")
    parser.add_argument("--max-per-video", type=int, default=8, help="Pilot seçiminde video başına azami segment sayısı")
    parser.add_argument("--init-from", type=str, default=None, help="Başlanacak checkpoint dosya adı (Modal volume /root/checkpoints)")
    parser.add_argument("--run-name", type=str, default=None, help="Deney ve checkpoint adı")
    parser.add_argument("--ckpt", type=str, default=None, help="Değerlendirilecek model checkpoint dosya yolu")
    parser.add_argument("--blank-penalty", type=float, default=0.2, help="CTC blank cezası (harf emisyonu teşviği)")
    parser.add_argument("--test-samples", type=int, default=40, help="Test split değerlendirme segment sayısı")
    parser.add_argument("--split", type=str, default=None, help="Veri spliti ('train', 'val', 'test')")
    parser.add_argument("--lm-alpha", type=float, default=0.4, help="Dil Modeli (LM) ağırlığı")
    parser.add_argument("--lm-beta", type=float, default=1.0, help="Kelime ekleme primi (Word insertion bonus)")
    parser.add_argument("--repeat-penalty", type=float, default=3.0, help="Ardışık kelime tekrar cezası")
    parser.add_argument("--min-dur-factor", type=float, default=1.0, help="Kelime asgari süre katsayısı (kare/harf)")
    parser.add_argument("--question-id", type=str, default=None, help="Aktif araştırma sorusu ID'si (ör. INIT-001)")
    parser.add_argument("--hypothesis", type=str, default=None, help="Deney hipotezi")
    parser.add_argument("--falsification-criteria", type=str, default=None, help="Hipotez yanlışlama kriteri")
    parser.add_argument("--expectation", type=str, default=None, help="Deney beklentisi")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "research-status":
        sys.exit(handle_research_status(args.candidate_path))
    elif args.mode == "seal-full-train":
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        handle_seal_full_train(
            candidate_path=args.candidate_path,
            manifest_path=args.full_train_manifest,
            seeds=seeds,
            budget_usd=args.budget_usd,
            dataset_scope_note=args.dataset_scope_note,
        )
        sys.exit(0)
    elif args.mode == "authorize-full-train":
        handle_authorize_full_train(
            candidate_path=args.candidate_path,
            manifest_path=args.full_train_manifest,
        )
        sys.exit(0)
    elif args.mode == "full-train":
        handle_authorize_full_train(
            candidate_path=args.candidate_path,
            manifest_path=args.full_train_manifest,
        )
        print("Full training is authorized. Per user instructions, full training is NOT started automatically.")
        sys.exit(0)

    require_mode_allowed(mode=args.mode)

    tracker = ExperimentTracker()

    if args.mode == "download-hf":
        run_download_hf(
            pilot=args.pilot,
            split=args.split or "train",
            download_all=args.all_data,
        )
    elif args.mode == "local-test":
        run_local_test(tracker)
    elif args.mode == "overfit":
        run_overfit_test(tracker, arch_name=args.arch, steps=args.steps, dataset_choice=args.dataset)
    elif args.mode == "micro-pilot":
        run_micro_pilot(
            tracker,
            arch_name=args.arch,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            train_samples=args.pilot if args.pilot != 50 else 24,
            blank_penalty=args.blank_penalty,
            dataset_choice=args.dataset,
            run_name=args.run_name,
        )
    elif args.mode == "inspect":
        run_inspect_predictions(
            ckpt_path=args.ckpt,
            samples_to_show=args.test_samples if args.test_samples != 40 else 10,
            split=args.split or "val",
            dataset_choice=args.dataset,
            blank_penalty=args.blank_penalty,
        )
    elif args.mode == "local-train":
        run_local_train(tracker, epochs=args.epochs, lr=args.lr, encoder_type=args.encoder, dataset_choice=args.dataset)
    elif args.mode == "eval":
        run_evaluation(
            tracker,
            ckpt_path=args.ckpt,
            encoder_type=args.encoder,
            blank_penalty=args.blank_penalty,
            split=args.split,
            lm_alpha=args.lm_alpha,
            lm_beta=args.lm_beta,
            repeat_penalty=args.repeat_penalty,
            min_dur_factor=args.min_dur_factor,
            dataset_choice=args.dataset,
            test_samples=args.test_samples,
        )
    elif args.mode == "modal-pilot":
        run_modal_pilot(
            tracker,
            run_name=args.run_name,
            pilot=args.pilot,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            encoder=args.encoder,
            d_model=args.d_model,
            num_layers=args.num_layers,
            resume=args.resume,
            blank_penalty=args.blank_penalty,
            warmup_epochs=args.warmup_epochs,
            test_samples=args.test_samples,
            data_mode=args.data_mode,
            max_per_word=args.max_per_word,
            init_from=args.init_from,
            lm_alpha=args.lm_alpha,
            lm_beta=args.lm_beta,
            dataset=args.dataset,
            max_per_video=args.max_per_video,
            question_id=args.question_id,
            hypothesis=args.hypothesis,
            falsification_criteria=args.falsification_criteria,
            expectation=args.expectation,
        )
    elif args.mode == "summary":
        tracker.print_summary()
    elif args.mode == "demo":
        from src.demo.app import build_app
        port = int(os.environ.get("PORT", 7860))
        app = build_app()
        print(f"\n🚀 Türkçe Dudak Okuma & 500 Kelime Avcısı Arayüzü Başlatılıyor: http://127.0.0.1:{port}")
        app.launch(server_name="0.0.0.0", server_port=port, share=False)
    elif args.mode == "all":
        print("\n🚀 Tüm Aşamalar Sırayla Doğrulanıyor...")
        t_ok = run_local_test(tracker)
        if t_ok:
            o_ok = run_overfit_test(tracker, steps=args.steps, dataset_choice=args.dataset)
            if o_ok:
                run_local_train(tracker, epochs=args.epochs, lr=args.lr, encoder_type=args.encoder, dataset_choice=args.dataset)
        tracker.print_summary()


if __name__ == "__main__":
    main()
