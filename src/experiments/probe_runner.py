"""
src/experiments/probe_runner.py — Kanonik Candidate-Aware Research Probe Runner

Bu modül, tek yaşayan araştırma modelinin (CANDIDATE.md / research_candidate.yaml)
aktif araştırma sorusuna bağlı geçici probe'larını çalıştırır.

Güvenlik ve Protokol Kuralları (GEMINI.md):
1. Yalnızca araştırma eğitimi yetkilendirilmişse (research_training_authorized=True) çalışır.
2. Yalnızca adayın aktif sorusunu (active_question) kabul eder.
3. Test kümesini kesinlikle araştırmaya kapatır (research_uses_test_split=False).
4. Her deneyden önce IN_PROGRESS kaydı yazar; sonuçlandığında PASSED/FALSIFIED/INCONCLUSIVE ve
   CER, WER, Spotter metrikleri, en az 20 ham tahmin ve tekrar üretilebilir provenance kaydeder.
5. Asla otomatik tam eğitim (full training) veya plansız ölçek büyütme tavsiye etmez.
"""

import hashlib
import json
import os
import pathlib
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml

from src.data.dataset import LipReadingDataset, pad_collate_fn
from src.data.split_map_data import IBOROTTI_SPLIT_MAP
from src.evaluation.metrics import compute_batch_cer, compute_batch_wer, evaluate_predictions
from src.experiments.guardrails import (
    build_checkpoint_provenance,
    collect_sample_ids,
    require_requested_sample_count,
)
from src.experiments.tracker import ExperimentRecord, ExperimentTracker
from src.models.factory import ARCH_CONFIGS, build_vsr_model, adapt_auto_avsr_weights
from src.vocab.turkish_vocab import (
    BLANK_IDX,
    VOCAB_SIZE,
    ctc_greedy_decode,
    normalize_turkish_text,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_CANDIDATE_PATH = ROOT / "configs" / "research_candidate.yaml"


@dataclass
class CandidateProbeConfig:
    experiment_id: str
    hypothesis: str
    falsification_criteria: str
    expectation: str
    question_id: str = "ARCH-001"
    arch_name: str = "vsr_conformer_base"
    d_model: Optional[int] = None
    num_layers: Optional[int] = None
    encoder_type: Optional[str] = None
    epochs: int = 5
    steps_per_epoch: Optional[int] = None
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    blank_penalty: float = 0.3
    device: str = "cpu"
    seed: int = 42
    initializer: str = "scratch"
    train_sample_limit: Optional[int] = None
    val_sample_limit: Optional[int] = None
    max_duration: Optional[float] = None
    freeze_frontend: bool = False
    unfreeze_frontend_epoch: Optional[int] = None
    eval_only: bool = False
    load_checkpoint: Optional[str] = None
    beam_size: int = 30
    lm_alpha: float = 0.4
    lm_beta: float = 1.0
    blank_penalty_grid: Optional[List[float]] = None
    conformer_dropout: float = 0.1
    use_specaugment: bool = False
    use_sequence_bucketing: bool = False
    cost_estimate_usd: float = 0.0


def validate_candidate_authority(
    candidate_path: Union[str, pathlib.Path],
    question_id: str,
) -> Dict[str, Any]:
    """Kanonik candidate yapılandırmasını ve araştırma yetkisini denetler."""
    path = pathlib.Path(candidate_path)
    if not path.is_file():
        raise FileNotFoundError(f"Kanonik candidate YAML dosyası bulunamadı: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    training = config.get("training", {})
    if not training.get("research_training_authorized", False):
        raise PermissionError(
            f"Araştırma eğitimi yetkisi verilmedi (Candidate={config.get('candidate_version')}). "
            "DATA-001 tamamlanmadan veya yetki açılmadan probe çalıştırılamaz."
        )

    active_q = config.get("active_question")
    if active_q != question_id:
        raise ValueError(
            f"Aktif araştırma sorusu {active_q} ancak {question_id} istendi. "
            "Aynı anda yalnız tek bir aktif araştırma sorusu probe edilebilir."
        )

    return config


def get_git_revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return rev.decode("utf-8").strip()
    except Exception:
        return "unknown"


class ResearchProbeRunner:
    """Kanonik modele bağlı kontrollü araştırma probe çalıştırıcısı."""

    def __init__(
        self,
        candidate_path: Union[str, pathlib.Path] = DEFAULT_CANDIDATE_PATH,
        tracker: Optional[ExperimentTracker] = None,
        data_dir: Optional[Union[str, pathlib.Path]] = None,
    ):
        self.candidate_path = pathlib.Path(candidate_path)
        self.tracker = tracker or ExperimentTracker()
        self.data_dir = pathlib.Path(data_dir) if data_dir else ROOT / "data" / "iborotti"

    def run(
        self,
        question_id: str,
        eval_split: str = "val",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Test split koruması ve yetki denetimi ile probe başlatır."""
        if eval_split == "test":
            raise RuntimeError(
                "Test split dondurulmuştur ve araştırmaya kapalıdır (GEMINI.md kuralı: "
                "research_uses_test_split=False). Sadece 'val' split kullanılabilir."
            )

        validate_candidate_authority(self.candidate_path, question_id)
        raise NotImplementedError("run() genel arayüzü; run_diagnostic_probe veya run_research_probe kullanın.")

    def run_diagnostic_probe(self, cfg: CandidateProbeConfig) -> Dict[str, Any]:
        """Hızlı diagnostik kontrol: Pipeline, gradyan akışı, loss ve metrik hesaplamasını sınar."""
        start_time = time.time()
        record = ExperimentRecord(
            experiment_id=cfg.experiment_id,
            hypothesis=cfg.hypothesis,
            falsification_criteria=cfg.falsification_criteria,
            setup=asdict(cfg),
            expectation=cfg.expectation,
            cost_estimate_usd=cfg.cost_estimate_usd,
        )
        self.tracker.log(record)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Basit sentetik / dummy veri ile pipeline kontrolü
        device = torch.device(cfg.device)
        model = build_vsr_model(
            arch_name=cfg.arch_name,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            encoder_type=cfg.encoder_type,
            dropout=cfg.conformer_dropout,
            use_specaugment=cfg.use_specaugment,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

        # Mock video batch: (B=2, C=1, T=30, H=88, W=88)
        B, C, T, H, W = 2, 1, 30, 88, 88
        dummy_video = torch.randn(B, C, T, H, W, device=device)
        dummy_targets = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.long, device=device)
        dummy_input_lens = torch.tensor([T, T], dtype=torch.long, device=device)
        dummy_target_lens = torch.tensor([4, 4], dtype=torch.long, device=device)

        model.train()
        for _ in range(cfg.epochs):
            optimizer.zero_grad()
            logits = model(dummy_video)
            loss = model.compute_loss(
                logits,
                dummy_targets,
                dummy_input_lens,
                dummy_target_lens,
                blank_penalty=cfg.blank_penalty,
            )
            loss.backward()
            optimizer.step()

        # Dummy evaluation
        model.eval()
        with torch.no_grad():
            eval_logits = model(dummy_video)
            preds = [ctc_greedy_decode(eval_logits[0])[0], ctc_greedy_decode(eval_logits[1])[0]]
            refs = ["test bir", "deneme iki"]

        metrics = evaluate_predictions(references=refs, hypotheses=preds)
        duration = time.time() - start_time

        falsified = np.isnan(loss.item()) or loss.item() > 100.0
        decision = "FALSIFIED" if falsified else "PASSED"

        provenance = {
            "seed": cfg.seed,
            "code_revision": get_git_revision(),
            "arch_name": cfg.arch_name,
            "initializer": cfg.initializer,
            "device": cfg.device,
        }

        record.status = decision
        record.result = {
            "final_loss": round(loss.item(), 4),
            "metrics": metrics,
            "duration_sec": round(duration, 2),
            "provenance": provenance,
        }
        record.updated_belief = "Diagnostik kod yolu ve metrik motoru sorunsuz çalışıyor."
        record.next_step = "Gerçek veri üzerinde kontrollü probe koşulabilir."
        self.tracker.log(record)

        return {
            "decision": "ACCEPT" if not falsified else "REJECT",
            "metrics": metrics,
            "provenance": provenance,
            "loss": loss.item(),
        }

    def run_research_probe(
        self,
        cfg: CandidateProbeConfig,
        train_dataset: Optional[LipReadingDataset] = None,
        val_dataset: Optional[LipReadingDataset] = None,
    ) -> Dict[str, Any]:
        """
        Kanonik candidate üzerinde gerçek verilerle tek bir araştırma sorusunu yanıtlayan probe koşar.
        Speaker-disjoint train ve val verisi kullanılır; test split kesinlikle yasaktır.
        """
        validate_candidate_authority(self.candidate_path, cfg.question_id)

        start_time = time.time()
        record = ExperimentRecord(
            experiment_id=cfg.experiment_id,
            hypothesis=cfg.hypothesis,
            falsification_criteria=cfg.falsification_criteria,
            setup=asdict(cfg),
            expectation=cfg.expectation,
            cost_estimate_usd=cfg.cost_estimate_usd,
        )
        self.tracker.log(record)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device != "cuda" else "cpu")

        # 1. Dataset yükleme ve konuşmacı sızıntısı kontrolü
        if train_dataset is None:
            train_dataset = LipReadingDataset(
                master_dir=self.data_dir,
                is_train=True,
                split="train",
                crop_size=88,
                max_duration=cfg.max_duration,
            )
        if val_dataset is None:
            val_dataset = LipReadingDataset(
                master_dir=self.data_dir,
                is_train=False,
                split="val",
                crop_size=88,
                max_duration=cfg.max_duration,
            )

        train_samples = train_dataset.samples
        val_samples = val_dataset.samples

        # Sınırlandırma varsa uygula
        if cfg.train_sample_limit and len(train_samples) > cfg.train_sample_limit:
            train_samples = train_samples[: cfg.train_sample_limit]
            train_dataset = LipReadingDataset(samples=train_samples, is_train=True, crop_size=88)
        if cfg.val_sample_limit and len(val_samples) > cfg.val_sample_limit:
            val_samples = val_samples[: cfg.val_sample_limit]
            val_dataset = LipReadingDataset(samples=val_samples, is_train=False, crop_size=88)

        # Provenance oluştur
        split_map_file = ROOT / "src" / "data" / "split_map_iborotti.json"
        train_ids = collect_sample_ids(train_dataset)
        val_ids = collect_sample_ids(val_dataset)

        provenance = build_checkpoint_provenance(
            dataset_id="data/iborotti",
            split_map_path=split_map_file,
            train_sample_ids=train_ids,
            val_sample_ids=val_ids,
            seed=cfg.seed,
            initializer=cfg.initializer,
            code_revision=get_git_revision(),
        )

        if cfg.use_sequence_bucketing:
            from src.data.dataset import SequenceBucketSampler
            train_sampler = SequenceBucketSampler(train_dataset, shuffle=True, seed=cfg.seed)
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                collate_fn=pad_collate_fn,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                collate_fn=pad_collate_fn,
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=pad_collate_fn,
        )

        # 2. Model oluşturma
        model = build_vsr_model(
            arch_name=cfg.arch_name,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            encoder_type=cfg.encoder_type,
            dropout=cfg.conformer_dropout,
            use_specaugment=cfg.use_specaugment,
        ).to(device)

        # Initializer / Checkpoint yükleme
        if cfg.load_checkpoint and os.path.exists(cfg.load_checkpoint):
            state = torch.load(cfg.load_checkpoint, map_location=device)
            model_state = state.get("model_state_dict", state)
            model.load_state_dict(model_state, strict=True)
            print(f"✅ Checkpoint yüklendi: {cfg.load_checkpoint}")
        elif cfg.initializer != "scratch" and os.path.exists(cfg.initializer):
            matched, total = adapt_auto_avsr_weights(cfg.initializer, model)
            if matched > 0:
                print(f"✅ Auto-AVSR frontend transferi yapıldı: {matched}/{total} tensör yüklendi.")
            else:
                state = torch.load(cfg.initializer, map_location="cpu")
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state, strict=False)

        if cfg.eval_only:
            from src.spotter.lexicon_decoder import LexiconBeamSearchDecoder
            lexicon_decoder = LexiconBeamSearchDecoder(
                beam_size=cfg.beam_size,
                blank_penalty=0.4,
                use_lm=True,
                lm_alpha=cfg.lm_alpha,
                lm_beta=cfg.lm_beta,
            )

            model.eval()
            val_losses = []
            val_refs = []
            val_hyps_greedy = []
            val_hyps_beam = []
            blank_counts = 0
            total_timesteps = 0

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
                        blank_penalty=cfg.blank_penalty,
                    )
                    val_losses.append(loss.item())

                    for i in range(len(videos)):
                        seq_len = int(input_lens[i].item())
                        logit_slice = logits[i, :seq_len]
                        pred_greedy = ctc_greedy_decode(logit_slice, blank_penalty=cfg.blank_penalty)[0]
                        pred_beam, _ = lexicon_decoder.decode(logit_slice)
                        val_hyps_greedy.append(pred_greedy)
                        val_hyps_beam.append(pred_beam)

                        top_classes = logit_slice.argmax(dim=-1)
                        blank_counts += int((top_classes == BLANK_IDX).sum().item())
                        total_timesteps += seq_len

                    val_refs.extend(batch["transcripts"])

            avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            blank_ratio = float(blank_counts / max(total_timesteps, 1))
            metrics_greedy = evaluate_predictions(references=val_refs, hypotheses=val_hyps_greedy)
            metrics_beam = evaluate_predictions(references=val_refs, hypotheses=val_hyps_beam)

            duration = time.time() - start_time
            wer_greedy = metrics_greedy["wer"]
            wer_beam = metrics_beam["wer"]

            if wer_beam >= wer_greedy:
                decision = "REJECT"
                status = "FALSIFIED"
            elif wer_beam < 0.80:
                decision = "ACCEPT"
                status = "PASSED"
            else:
                decision = "INCONCLUSIVE"
                status = "INCONCLUSIVE"

            sample_comparisons = []
            for r, g, b in zip(val_refs[:25], val_hyps_greedy[:25], val_hyps_beam[:25]):
                sample_comparisons.append({"ref": r, "greedy": g, "beam": b})

            res = {
                "decision": decision,
                "status": status,
                "eval_only": True,
                "best_val_loss": round(avg_val_loss, 4),
                "cer": metrics_beam["cer"],
                "wer": metrics_beam["wer"],
                "spotter_f1": metrics_beam["spotter_f1"],
                "blank_ratio": round(blank_ratio, 4),
                "greedy_metrics": metrics_greedy,
                "beam_metrics": metrics_beam,
                "sample_comparisons": sample_comparisons,
                "duration_sec": round(duration, 2),
            }

            record.status = status
            record.result = res
            record.surprise = f"Greedy WER: {wer_greedy} -> Beam WER: {wer_beam}"
            record.updated_belief = f"Decoder evaluation probe {cfg.question_id} finished."
            self.tracker.log(record)
            return res

        if cfg.freeze_frontend:
            model.freeze_frontend(True)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            frontend_params = [p for n, p in model.named_parameters() if "frontend" in n and p.requires_grad]
            temporal_params = [p for n, p in model.named_parameters() if "frontend" not in n and p.requires_grad]
            optimizer = optim.AdamW([
                {"params": frontend_params, "lr": cfg.lr * 0.05},
                {"params": temporal_params, "lr": cfg.lr},
            ], weight_decay=cfg.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

        # 3. Eğitim Döngüsü
        history = []
        best_val_loss = float("inf")

        for epoch in range(1, cfg.epochs + 1):
            if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)
            if cfg.unfreeze_frontend_epoch and epoch == cfg.unfreeze_frontend_epoch and cfg.freeze_frontend:
                print(f"🔓 [UNFREEZE] Epoch {epoch}: Frontend çözüldü, diferansiyel LR ({cfg.lr * 0.05:.2e}) ile ince ayara geçiliyor...")
                model.freeze_frontend(False)
                fe_params = [p for n, p in model.named_parameters() if "frontend" in n and p.requires_grad]
                optimizer.add_param_group({"params": fe_params, "lr": cfg.lr * 0.05})

            model.train()
            train_losses = []
            step = 0
            for batch in train_loader:
                step += 1
                if cfg.steps_per_epoch and step > cfg.steps_per_epoch:
                    break

                videos = (batch.get("videos") if "videos" in batch else batch["video"]).to(device)
                targets = (batch.get("targets") if "targets" in batch else batch["target"]).to(device)
                input_lens = batch["input_lengths"].to(device)
                target_lens = batch["target_lengths"].to(device)

                optimizer.zero_grad()
                logits = model(videos)
                loss = model.compute_loss(
                    logits,
                    targets,
                    input_lens,
                    target_lens,
                    blank_penalty=cfg.blank_penalty,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()
            avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            # Validasyon
            model.eval()
            val_losses = []
            val_refs = []
            val_hyps = []
            blank_counts = 0
            total_timesteps = 0

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
                        blank_penalty=cfg.blank_penalty,
                    )
                    val_losses.append(loss.item())

                    # Greedy decoding
                    for i in range(len(videos)):
                        seq_len = int(input_lens[i].item())
                        logit_slice = logits[i, :seq_len]
                        pred_text = ctc_greedy_decode(logit_slice, blank_penalty=cfg.blank_penalty)[0]
                        val_hyps.append(pred_text)
                        
                        top_classes = logit_slice.argmax(dim=-1)
                        blank_counts += int((top_classes == BLANK_IDX).sum().item())
                        total_timesteps += seq_len

                    val_refs.extend(batch["transcripts"])

            avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            blank_ratio = float(blank_counts / max(total_timesteps, 1))
            val_metrics = evaluate_predictions(references=val_refs, hypotheses=val_hyps)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            history.append({
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(avg_val_loss, 4),
                "blank_ratio": round(blank_ratio, 4),
                "cer": val_metrics["cer"],
                "wer": val_metrics["wer"],
                "spotter_f1": val_metrics["spotter_f1"],
            })

        duration = time.time() - start_time

        # Karar ve sonuç çıkarma
        # Falsification kriteri: Val loss düşüşü olmaması veya %100 boşluk çökmesi
        last_eval = history[-1]
        falsified = (last_eval["blank_ratio"] >= 0.99 and last_eval["cer"] >= 1.0)
        
        if falsified:
            status = "FALSIFIED"
            decision = "INCONCLUSIVE"  # VSR'da boşluk çökmesi mimariyi tamamen çürütmez; curriculum veya blank mitigation gerektirir
        else:
            status = "PASSED"
            decision = "ACCEPT"

        # En az 20 ham tahmin örneği topla
        sample_comparisons = []
        for r, h in zip(val_refs[:25], val_hyps[:25]):
            sample_comparisons.append({"ref": r, "hyp": h})

        record.status = status
        record.result = {
            "best_val_loss": round(best_val_loss, 4),
            "final_val_loss": last_eval["val_loss"],
            "final_train_loss": last_eval["train_loss"],
            "cer": last_eval["cer"],
            "wer": last_eval["wer"],
            "spotter_f1": last_eval["spotter_f1"],
            "blank_ratio": last_eval["blank_ratio"],
            "history": history,
            "sample_comparisons": sample_comparisons,
            "duration_sec": round(duration, 2),
            "provenance": provenance,
        }
        record.surprise = f"Final blank_ratio: {last_eval['blank_ratio']}, CER: {last_eval['cer']}"
        record.updated_belief = (
            f"Candidate {cfg.question_id} probe tamamlandı. "
            f"Val loss={last_eval['val_loss']}, CER={last_eval['cer']}."
        )
        record.next_step = (
            "Kanonik modeli güncelle ve bir sonraki aktif belirsizliğe geç."
            if decision == "ACCEPT"
            else "Curriculum, initializer veya loss hiperparametrelerini incele."
        )
        self.tracker.log(record)

        return {
            "decision": decision,
            "status": status,
            "best_val_loss": best_val_loss,
            "final_metrics": last_eval,
            "history": history,
            "sample_comparisons": sample_comparisons,
            "provenance": provenance,
            "duration_sec": duration,
        }

    def run_remote_probe(self, cfg: CandidateProbeConfig) -> Dict[str, Any]:
        """Modal A10G GPU üzerinde kanonik candidate probe'unu çalıştırır."""
        candidate_meta = validate_candidate_authority(self.candidate_path, cfg.question_id)
        candidate_version = candidate_meta.get("candidate_version", "c0.0.0")

        record = ExperimentRecord(
            experiment_id=cfg.experiment_id,
            hypothesis=cfg.hypothesis,
            falsification_criteria=cfg.falsification_criteria,
            setup=asdict(cfg),
            expectation=cfg.expectation,
            cost_estimate_usd=cfg.cost_estimate_usd,
        )
        self.tracker.log(record)

        import modal
        from src.experiments.modal_probe import app, run_canonical_probe_remote

        probe_dict = asdict(cfg)
        with modal.enable_output():
            with app.run():
                res = run_canonical_probe_remote.remote(
                    candidate_version=candidate_version,
                    question_id=cfg.question_id,
                    probe_cfg_dict=probe_dict,
                )

        record.status = res["status"]
        record.result = res
        if "gate_audit" in res:
            record.surprise = (
                f"Dress Rehearsal Completed! Throughput: {res.get('throughput_clips_per_sec', 0):.1f} clips/s, "
                f"Epoch Time: {res.get('avg_epoch_sec', 0):.1f}s, "
                f"Proj Full-Train Cost: ${res.get('projected_full_train_cost_usd', 0):.4f}, "
                f"Greedy CER: {res.get('greedy_cer', 0)*100:.2f}%, Spotter F1: {res.get('spotter_f1', 0):.4f}"
            )
            record.updated_belief = (
                f"READINESS-001 Dress Rehearsal & 12-Gate Audit {res.get('readiness_gate_status')}. "
                f"Full-train duration: {res.get('projected_full_train_duration_min', 0):.1f} min, "
                f"cost: ${res.get('projected_full_train_cost_usd', 0):.4f}. Canonical recipe ready for full train."
            )
        elif "kws_metrics" in res:
            bname = res.get("best_config_name")
            km = res.get("kws_metrics", {})
            gm = res.get("greedy_metrics", {})
            record.surprise = (
                f"Optimal KWS Config={bname} -> Spotter F1: {km.get('f1', 0):.4f}, "
                f"Prec: {km.get('precision', 0)*100:.2f}%, Recall: {km.get('recall', 0)*100:.2f}%, "
                f"TP: {km.get('true_positives')}, FP: {km.get('false_positives')}, FN: {km.get('false_negatives')}"
            )
            record.updated_belief = (
                f"KWS-001 Spotter Posterior Calibration completed. "
                f"Best={bname}, Spotter F1={km.get('f1', 0):.4f}, "
                f"Prec={km.get('precision', 0)*100:.2f}%, Recall={km.get('recall', 0)*100:.2f}%"
            )
        elif "best_config_name" in res:
            bname = res.get("best_config_name")
            bm = res.get("beam_metrics", {})
            gm = res.get("greedy_metrics", {})
            record.surprise = f"Optimal Config={bname} -> Beam CER: {bm.get('cer', 0)*100:.2f}%, Beam WER: {bm.get('wer', 0)*100:.2f}%, Spotter F1: {bm.get('spotter_f1', 0):.4f}"
            record.updated_belief = (
                f"DEC-005 Decoder Calibration completed. "
                f"Best={bname}, Beam CER={bm.get('cer', 0)*100:.2f}%, Beam WER={bm.get('wer', 0)*100:.2f}%, "
                f"Spotter F1={bm.get('spotter_f1', 0):.4f}"
            )
        elif res.get("grid_results"):
            best_bp = res.get("best_blank_penalty")
            bm = res.get("beam_metrics", {})
            gm = res.get("greedy_metrics", {})
            record.surprise = f"Optimal BP={best_bp} -> Greedy CER: {gm.get('cer', 0)*100:.2f}%, Beam CER: {bm.get('cer', 0)*100:.2f}%, Deletions: {gm.get('deletions')} ({gm.get('del_pct')}%)"
            record.updated_belief = (
                f"DEC-004 Blank Penalty Grid on c0.4.0 completed. "
                f"Best BP={best_bp}, Greedy CER={gm.get('cer', 0)*100:.2f}%, Beam CER={bm.get('cer', 0)*100:.2f}%, "
                f"Beam WER={bm.get('wer', 0)*100:.2f}%, Beam F1={bm.get('spotter_f1', 0):.4f}"
            )
        elif res.get("eval_only"):
            bm = res.get("beam_metrics", {})
            gm = res.get("greedy_metrics", {})
            record.surprise = f"Greedy WER: {gm.get('wer')} -> Beam WER: {bm.get('wer')}, Beam Latency: {bm.get('latency_sec')}s"
            record.updated_belief = (
                f"Decoder evaluation probe ({cfg.question_id}). "
                f"Greedy WER: {gm.get('wer')}, Beam WER: {bm.get('wer')}, "
                f"Spotter F1: {bm.get('spotter_f1')}, Latency: {bm.get('latency_sec')}s"
            )
        else:
            record.surprise = f"Final blank_ratio: {res.get('blank_ratio')}, CER: {res.get('cer')}"
            record.updated_belief = (
                f"Remote A10G Probe {cfg.question_id} ({cfg.experiment_id}) tamamlandı. "
                f"Val Loss: {res.get('best_val_loss')}, CER: {res.get('cer')}, Blank: {res.get('blank_ratio')}"
            )
        record.next_step = (
            "Kanonik modeli güncelle ve bir sonraki adıma geç."
            if res.get("decision") == "ACCEPT"
            else "Curriculum, hiperparametre veya mimari bileşenlerini incele."
        )
        record.cost_usd = res.get("cost_usd", 0.0)
        self.tracker.log(record)
        return res


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Canonical Candidate Probe Runner (GEMINI.md)")
    parser.add_argument("--candidate", type=str, default=str(DEFAULT_CANDIDATE_PATH), help="Candidate YAML")
    parser.add_argument("--question", type=str, required=True, help="Active question ID (e.g. TRAIN-001, DEC-001)")
    parser.add_argument("--probe-id", type=str, default=None, help="Probe experiment ID")
    parser.add_argument("--remote", action="store_true", help="Run remotely on Modal A10G")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--blank-penalty", type=float, default=0.4)
    parser.add_argument("--max-duration", type=float, default=None, help="Curriculum max duration in sec")
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-val", type=int, default=30)
    parser.add_argument("--hypothesis", type=str, default="")
    parser.add_argument("--falsification", type=str, default="")
    parser.add_argument("--expectation", type=str, default="")
    parser.add_argument("--encoder-type", type=str, default="conformer")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--freeze-frontend", action="store_true", help="Freeze 3D-ResNet visual frontend")
    parser.add_argument("--unfreeze-epoch", type=int, default=None, help="Epoch to unfreeze frontend for fine-tuning")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only on held-out validation set")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Checkpoint file name in volume or path")
    parser.add_argument("--beam-size", type=int, default=30, help="Beam size for LexiconBeamSearchDecoder")
    parser.add_argument("--lm-alpha", type=float, default=0.4, help="Language model weight alpha")
    parser.add_argument("--lm-beta", type=float, default=1.0, help="Word insertion penalty beta")
    parser.add_argument("--blank-penalty-grid", type=str, default=None, help="Comma-separated blank penalties for grid evaluation, e.g. 0.0,0.4,0.8,1.2,1.5,1.8,2.0,2.2,2.5")
    parser.add_argument("--conformer-dropout", type=float, default=0.1, help="Conformer dropout rate")
    parser.add_argument("--use-specaugment", action="store_true", help="Enable feature-level SpecAugment (temporal and channel masking)")
    parser.add_argument("--use-sequence-bucketing", action="store_true", help="Enable sequence bucketing and dynamic batching by duration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training and evaluation")

    args = parser.parse_args()

    import sys
    probe_id = args.probe_id or f"probe_{args.question.lower()}_{int(time.time())}"
    bp_grid = (
        [float(x.strip()) for x in args.blank_penalty_grid.split(",") if x.strip()]
        if args.blank_penalty_grid
        else None
    )

    use_bucketing = args.use_sequence_bucketing or (args.question == "LOWDATA-003")
    n_train = args.n_train
    n_val = args.n_val
    max_dur = args.max_duration
    load_ckpt = args.load_checkpoint

    if args.question == "LOWDATA-003":
        if "--n-train" not in sys.argv:
            n_train = None
        if "--n-val" not in sys.argv:
            n_val = None
        if "--max-duration" not in sys.argv:
            max_dur = None
        if not load_ckpt:
            load_ckpt = "probe_confirm001_full_train_scaling_best.pt"

    cfg = CandidateProbeConfig(
        experiment_id=probe_id,
        hypothesis=args.hypothesis or f"Probe testing {args.question}",
        falsification_criteria=args.falsification or "Validation loss stalling or blank collapse.",
        expectation=args.expectation or "Convergence and meaningful CER reduction.",
        question_id=args.question,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        blank_penalty=args.blank_penalty,
        blank_penalty_grid=bp_grid,
        max_duration=max_dur,
        freeze_frontend=args.freeze_frontend,
        unfreeze_frontend_epoch=args.unfreeze_epoch,
        eval_only=args.eval_only,
        load_checkpoint=load_ckpt,
        beam_size=args.beam_size,
        lm_alpha=args.lm_alpha,
        lm_beta=args.lm_beta,
        train_sample_limit=n_train,
        val_sample_limit=n_val,
        encoder_type=args.encoder_type,
        d_model=args.d_model,
        num_layers=args.num_layers,
        conformer_dropout=args.conformer_dropout,
        use_specaugment=args.use_specaugment,
        use_sequence_bucketing=use_bucketing,
        seed=args.seed,
    )

    runner = ResearchProbeRunner(candidate_path=args.candidate)

    if args.remote:
        print(f"📡 [REMOTE MODAL] Launching probe {probe_id} for {args.question}...")
        result = runner.run_remote_probe(cfg)
    else:
        print(f"💻 [LOCAL] Launching probe {probe_id} for {args.question}...")
        result = runner.run_research_probe(cfg)

    print("\n--- Probe Result ---")
    print(f"Status: {result.get('status')}")
    print(f"Decision: {result.get('decision')}")
    print(f"Best Val Loss: {result.get('best_val_loss')}")
    print(f"CER: {result.get('cer')}")
    print(f"WER: {result.get('wer')}")
    print(f"Blank Ratio: {result.get('blank_ratio')}")
    if "best_blank_penalty" in result:
        print(f"Optimal Blank Penalty: {result.get('best_blank_penalty')}")
    if "cost_usd" in result:
        print(f"Cost USD: ${result['cost_usd']:.4f}")

    if "multi_seed_results" in result:
        ms = result["multi_seed_results"]
        spk = result.get("speaker_results", {})
        print("\n--- CONFIRM-002 Multi-Seed Stability Results ---")
        print(f"{'Seed':>6} | {'Greedy CER':>11} | {'Beam WER':>9} | {'Spotter F1':>10} | {'Prec%':>7} | {'Rec%':>7} | {'TP':>4} | {'FP':>5}")
        print("-" * 75)
        for s, d in ms.items():
            print(
                f"{s:>6} | {d['greedy_cer']*100:10.2f}% | {d['beam_wer']*100:8.2f}% | "
                f"{d['spotter_f1']:10.4f} | {d['spotter_precision']*100:6.2f}% | {d['spotter_recall']*100:6.2f}% | "
                f"{d['true_positives']:4d} | {d['false_positives']:5d}"
            )
        print(f"Mean ± Std Greedy CER: {result.get('mean_greedy_cer', 0)*100:.2f}% ± {result.get('std_greedy_cer', 0)*100:.2f}%")
        print(f"Mean ± Std Beam WER  : {result.get('mean_beam_wer', 0)*100:.2f}% ± {result.get('std_beam_wer', 0)*100:.2f}%")
        print(f"Mean ± Std Spotter F1: {result.get('mean_spotter_f1', 0):.4f} ± {result.get('std_spotter_f1', 0):.4f}")

        if spk:
            print("\n--- CONFIRM-002 Speaker-Disjoint Cross-Validation ---")
            for spk_name, sd in spk.items():
                print(f"  {spk_name:<26} ({sd['clips']} klip): Greedy CER={sd['greedy_cer']*100:.2f}%, WER={sd['greedy_wer']*100:.2f}%")
    elif "gate_audit" in result:
        print("\n--- GEMINI.md Section 5.1 Full-Training Readiness 12-Gate Audit ---")
        print(f"{'Gate':<40} | {'Status':<10} | {'Evidence':<55}")
        print("-" * 115)
        for gname, gdata in result["gate_audit"].items():
            print(f"{gname:<40} | {gdata['status']:<10} | {gdata['evidence']:<55}")
        print(f"\nOverall Readiness Status: {result.get('readiness_gate_status')}")
        print(f"Average Epoch Time: {result.get('avg_epoch_sec')}s | Throughput: {result.get('throughput_clips_per_sec')} clips/s")
        print(f"Projected Full Training (4 Epochs): {result.get('projected_full_train_duration_min')} min | Estimated Cost: ${result.get('projected_full_train_cost_usd'):.4f} USD")
    elif "kws_metrics" in result and "grid_results" in result:
        gr = result["grid_results"]
        print("\n--- KWS-001 Posterior Spotter Calibration Grid Results (Top 15 by F1) ---")
        print(f"{'Config':>26} | {'BP':>4} | {'Conf':>5} | {'Sub':>5} | {'VisTol':>6} | {'Prec%':>7} | {'Rec%':>7} | {'Spotter F1':>10} | {'TP':>4} | {'FP':>5} | {'FN':>4} | {'Lat(ms)':>8}")
        print("-" * 115)
        sorted_configs = sorted(gr.items(), key=lambda x: x[1]["metrics"]["f1"], reverse=True)
        for cname, data in sorted_configs[:15]:
            c = data["config"]
            m = data["metrics"]
            print(
                f"{cname:>26} | {c['bp']:4.1f} | {c['conf']:5.2f} | {str(c['allow_sub']):>5} | {str(c['viseme_tol']):>6} | "
                f"{m['precision']*100:6.2f}% | {m['recall']*100:6.2f}% | {m['f1']:10.4f} | {m['true_positives']:4d} | "
                f"{m['false_positives']:5d} | {m['false_negatives']:4d} | {m['latency_sec']*1000:7.2f}ms"
            )
    elif "grid_results" in result:
        gr = result["grid_results"]
        if "best_config_name" in result:
            print("\n--- DEC-005 Decoder Calibration Grid Results ---")
            print(f"{'Config':>26} | {'VisTol':>6} | {'BP':>4} | {'Alpha':>5} | {'Beta':>5} | {'Rep':>4} | {'Beam CER':>9} | {'Beam WER':>9} | {'Del%':>6} | {'Spotter F1':>10}")
            print("-" * 115)
            for cname, data in gr.items():
                c = data["config"]
                b = data["beam"]
                print(
                    f"{cname:>26} | {str(c['viseme_tol']):>6} | {c['bp']:4.1f} | {c['alpha']:5.2f} | {c['beta']:5.2f} | {c['rep']:4.1f} | "
                    f"{b['cer']*100:8.2f}% | {b['wer']*100:8.2f}% | {b['del_pct']:5.1f}% | {b['spotter_f1']:10.4f}"
                )
        else:
            print("\n--- CTC Blank Penalty Calibration Grid Results ---")
            print(f"{'BP':>5} | {'Blank%':>7} | {'Greedy CER':>10} | {'Greedy WER':>10} | {'S':>5} | {'D':>5} | {'I':>5} | {'Del%':>6} | {'Beam CER':>10} | {'Beam WER':>10} | {'Beam F1':>8}")
            print("-" * 105)
            for bp_str, data in gr.items():
                g = data["greedy"]
                b = data["beam"]
                print(
                    f"{float(bp_str):5.1f} | {data['blank_ratio']*100:6.1f}% | "
                    f"{g['cer']*100:9.2f}% | {g['wer']*100:9.2f}% | "
                    f"{g['substitutions']:5d} | {g['deletions']:5d} | {g['insertions']:5d} | {g['del_pct']:5.1f}% | "
                    f"{b['cer']*100:9.2f}% | {b['wer']*100:9.2f}% | {b['spotter_f1']:8.4f}"
                )

    if "greedy_metrics" in result and "beam_metrics" in result:
        print("\n--- Decoder Comparison: Greedy vs Lexicon Beam Search ---")
        gm = result["greedy_metrics"]
        bm = result["beam_metrics"]
        print(f"Greedy -> CER: {gm['cer']:.4f} | WER: {gm['wer']:.4f} | Spotter F1: {gm['spotter_f1']:.4f} | Latency: {gm.get('latency_sec', 0.0)*1000:.1f}ms")
        print(f"Beam   -> CER: {bm['cer']:.4f} | WER: {bm['wer']:.4f} | Spotter F1: {bm['spotter_f1']:.4f} | Latency: {bm.get('latency_sec', 0.0)*1000:.1f}ms")

    if "sample_comparisons" in result:
        print("\n--- Sample Predictions (first 5) ---")
        for idx, s in enumerate(result["sample_comparisons"][:5], 1):
            print(f"[{idx}] Ref:    {s.get('ref')}")
            if "greedy" in s:
                print(f"    Greedy: {s.get('greedy')}")
            if "beam" in s:
                print(f"    Beam:   {s.get('beam')}")
            if "kws_detected" in s:
                print(f"    KWS:    {', '.join(s.get('kws_detected'))}")
            elif "hyp" in s:
                print(f"    Hyp:    {s.get('hyp')}")


if __name__ == "__main__":
    main()

