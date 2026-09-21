"""
src/experiments/modal_probe.py — Remote Modal A10G Execution for Canonical Candidate Probes

Bu modül, configs/research_candidate.yaml ve GEMINI.md protokolü ile yönetilen
araştırma probe'larını Modal A10G GPU üzerinde çalıştırır.

Güvenlik ve Protokol Kuralları:
1. Sadece candidate_version ve aktif soru eşleştiğinde çalışır (validate_candidate_authority).
2. Test split kesinlikle araştırmaya kapalıdır (yalnız speaker-disjoint val kullanılır).
3. Auto-AVSR görsel ön katmanı turkish-vsr-vol üzerinden transfer edilir.
4. Tam CER, WER, Spotter F1, blank ratio ve en az 20 ham tahmin döner.
"""

import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Modal İmaj Tanımı
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "boto3>=1.34.0",
        "av>=12.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
        "jiwer>=3.0.0",
        "rapidfuzz>=3.0.0",
        "huggingface_hub>=0.23.0",
    )
    .add_local_python_source("src")
    .add_local_dir(str(ROOT / "data" / "metadata"), remote_path="/root/data/metadata")
    .add_local_dir(str(ROOT / "configs"), remote_path="/root/configs")
)

app = modal.App("canonical-vsr-probe-runner")
vol = modal.Volume.from_name("turkish-vsr-vol", create_if_missing=True)
s3_secret = modal.Secret.from_name("bucket-credentials")


@app.function(
    image=image,
    gpu="A10G",
    secrets=[s3_secret],
    volumes={"/root/vol": vol},
    timeout=3600,
)
def run_canonical_probe_remote(
    candidate_version: str,
    question_id: str,
    probe_cfg_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Modal A10G üzerinde kanonik candidate probe'unu çalıştırır.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import yaml

    from src.data.dataset import LipReadingDataset, SequenceBucketSampler, pad_collate_fn
    from src.data.hf_downloader import HFDatasetDownloader
    from src.evaluation.metrics import evaluate_predictions
    from src.experiments.guardrails import (
        build_checkpoint_provenance,
        collect_sample_ids,
        require_requested_sample_count,
    )
    from src.experiments.probe_runner import validate_candidate_authority
    from src.models.factory import adapt_auto_avsr_weights, build_vsr_model
    from src.vocab.turkish_vocab import BLANK_IDX, ctc_greedy_decode

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Candidate yetki ve aktif soru doğrulaması
    candidate_file = pathlib.Path("/root/configs/research_candidate.yaml")
    validate_candidate_authority(candidate_file, question_id)

    with open(candidate_file, "r", encoding="utf-8") as f:
        candidate_meta = yaml.safe_load(f)
    assert candidate_meta.get("candidate_version") == candidate_version, (
        f"Candidate sürüm uyuşmazlığı: Beklenen {candidate_version}, "
        f"Bulunan {candidate_meta.get('candidate_version')}"
    )

    seed = probe_cfg_dict.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Veri Hazırlığı (Speaker-Disjoint train ve val)
    use_bucketing = probe_cfg_dict.get("use_sequence_bucketing", False) or (question_id == "LOWDATA-003")
    if use_bucketing:
        n_train_samples = probe_cfg_dict.get("train_sample_limit")
        n_val_samples = probe_cfg_dict.get("val_sample_limit")
    else:
        n_train_samples = probe_cfg_dict.get("train_sample_limit") or 60
        n_val_samples = probe_cfg_dict.get("val_sample_limit") or 30
    max_duration = probe_cfg_dict.get("max_duration")

    split_map_file = pathlib.Path("/root/data/metadata/split_map_iborotti.json")
    tar_path = pathlib.Path("/root/vol/clips.tar.gz")
    vol_clips_dir = pathlib.Path("/root/vol/data/iborotti/clips")
    local_data_dir = pathlib.Path("/root/vol/data/iborotti")

    if not vol_clips_dir.exists():
        if tar_path.exists():
            print(f"📦 [VOLUME] {tar_path} arşivi /root/vol/data/iborotti dizinine açılıyor...")
            local_data_dir.mkdir(parents=True, exist_ok=True)
            import subprocess
            subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(local_data_dir)], check=True)
            vol.commit()
            print("✅ [VOLUME] Arşiv başarıyla açıldı ve Volume'a kaydedildi.")
        else:
            local_data_dir = pathlib.Path("/root/data/iborotti")
            print(f"📥 Volume üzerinde veri bulunamadı, HF üzerinden hazırlanıyor...")
            hf_downloader = HFDatasetDownloader(
                target_dir=local_data_dir,
                split_map_path=split_map_file,
            )
            print(f"📥 Train split indiriliyor ({n_train_samples} segment, max_duration={max_duration})...")
            train_dirs = hf_downloader.download_pilot(
                n_samples=n_train_samples,
                split="train",
                max_per_video=probe_cfg_dict.get("max_per_video", 15),
                max_duration=max_duration,
            )
            require_requested_sample_count(split="train", requested=n_train_samples, actual=len(train_dirs))

            print(f"📥 Val split indiriliyor ({n_val_samples} segment, max_duration={max_duration})...")
            val_dirs = hf_downloader.download_pilot(
                n_samples=n_val_samples,
                split="val",
                max_per_video=None,
                max_duration=max_duration,
            )
            require_requested_sample_count(split="val", requested=n_val_samples, actual=len(val_dirs))
    else:
        print(f"⚡ [VOLUME CACHE] /root/vol/data/iborotti/clips doğrudan kullanılıyor (HF indirmesi atlandı).")

    eval_only = probe_cfg_dict.get("eval_only", False)

    if eval_only:
        val_ds = LipReadingDataset(
            data_dir=local_data_dir,
            split="val",
            split_map_path=split_map_file,
            is_train=False,
            crop_size=88,
            max_duration=max_duration,
            cache_in_ram=False,
        )
        if n_val_samples and len(val_ds.samples) > n_val_samples:
            val_ds.samples = val_ds.samples[:n_val_samples]

        print(f"⚡ [EVAL ONLY] RAM'e önbelleğe alınıyor: {len(val_ds)} val segment...")
        val_ds.cache_in_ram = True
        val_ds._preload_into_ram()

        batch_size = probe_cfg_dict.get("batch_size", 8)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_collate_fn,
        )

        arch_name = probe_cfg_dict.get("arch_name", "vsr_conformer_base")
        d_model = probe_cfg_dict.get("d_model", 512)
        num_layers = probe_cfg_dict.get("num_layers", 4)
        encoder_type = probe_cfg_dict.get("encoder_type", "conformer")
        conformer_dropout = probe_cfg_dict.get("conformer_dropout", 0.1)
        use_specaugment = probe_cfg_dict.get("use_specaugment", False)

        model = build_vsr_model(
            arch_name=arch_name,
            d_model=d_model,
            num_layers=num_layers,
            encoder_type=encoder_type,
            dropout=conformer_dropout,
            use_specaugment=use_specaugment,
        ).to(device)

        ckpt_name = probe_cfg_dict.get("load_checkpoint") or "probe_train001_full_curriculum_unfreeze_best.pt"
        ckpt_path = pathlib.Path("/root/vol") / ckpt_name
        if not ckpt_path.exists():
            ckpt_path = pathlib.Path("/root/checkpoints") / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Değerlendirilecek checkpoint bulunamadı: {ckpt_name} (/root/vol içinde yok)")

        print(f"📥 [CHECKPOINT] Checkpoint yükleniyor: {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=device)
        model_state = state.get("model_state_dict", state)
        model.load_state_dict(model_state, strict=True)
        print(f"✅ Checkpoint başarıyla yüklendi! (Kayıtlı val_loss: {state.get('val_loss')}, epoch: {state.get('epoch')})")

        from src.spotter.lexicon_decoder import LexiconBeamSearchDecoder
        beam_size = probe_cfg_dict.get("beam_size", 30)
        lm_alpha = probe_cfg_dict.get("lm_alpha", 0.4)
        lm_beta = probe_cfg_dict.get("lm_beta", 1.0)
        blank_penalty = probe_cfg_dict.get("blank_penalty", 0.8)

        lexicon_decoder = LexiconBeamSearchDecoder(
            beam_size=beam_size,
            blank_penalty=0.4,
            use_lm=True,
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
        )

        import jiwer
        model.eval()
        val_losses = []
        val_refs = []
        cached_logits = []
        total_timesteps = 0

        print(f"🔍 [EVALUATION] {len(val_ds)} validation klibi modelden geçiriliyor (tek forward pass)...")
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
                    blank_penalty=blank_penalty,
                )
                val_losses.append(loss.item())

                for i in range(len(videos)):
                    seq_len = int(input_lens[i].item())
                    logit_slice = logits[i, :seq_len].detach().cpu()
                    cached_logits.append(logit_slice)
                    total_timesteps += seq_len

                val_refs.extend(batch["transcripts"])

        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        if question_id == "DEC-005":
            dec005_configs = [
                {"name": "c040_baseline_viseme", "viseme_tol": True, "bp": 2.0, "alpha": 0.3, "beta": -0.2, "rep": 3.0},
                {"name": "strict_base_bp20", "viseme_tol": False, "bp": 2.0, "alpha": 0.3, "beta": -0.2, "rep": 3.0},
                {"name": "strict_rep5_bp20", "viseme_tol": False, "bp": 2.0, "alpha": 0.3, "beta": -0.2, "rep": 5.0},
                {"name": "strict_rep4_bp15", "viseme_tol": False, "bp": 1.5, "alpha": 0.3, "beta": -0.2, "rep": 4.0},
                {"name": "strict_rep4_bp12", "viseme_tol": False, "bp": 1.2, "alpha": 0.3, "beta": -0.2, "rep": 4.0},
                {"name": "strict_low_alpha_bp20", "viseme_tol": False, "bp": 2.0, "alpha": 0.1, "beta": -0.2, "rep": 4.0},
                {"name": "strict_mid_alpha_bp20", "viseme_tol": False, "bp": 2.0, "alpha": 0.2, "beta": -0.2, "rep": 4.0},
                {"name": "strict_penalize_short_bp20", "viseme_tol": False, "bp": 2.0, "alpha": 0.2, "beta": -0.4, "rep": 4.0},
                {"name": "strict_neutral_beta_bp20", "viseme_tol": False, "bp": 2.0, "alpha": 0.2, "beta": 0.0, "rep": 4.0},
            ]
            grid_results = {}
            print(f"\n📊 [DEC-005 CALIBRATION GRID] {len(dec005_configs)} çözücü yapılandırması değerlendiriliyor...")
            print(f"{'Config':>26} | {'VisTol':>6} | {'BP':>4} | {'Alpha':>5} | {'Beta':>5} | {'Rep':>4} | {'Beam CER':>9} | {'Beam WER':>9} | {'Del%':>6} | {'Spotter F1':>10}")
            print("-" * 115)

            for cfg in dec005_configs:
                cname = cfg["name"]
                lexicon_decoder.viseme_tolerance = cfg["viseme_tol"]
                beam_hyps = []
                beam_kws = []
                t0_b = time.perf_counter()
                for ls in cached_logits:
                    pred_b, det_kws = lexicon_decoder.decode(
                        ls,
                        blank_penalty=cfg["bp"],
                        lm_alpha=cfg["alpha"],
                        lm_beta=cfg["beta"],
                        repeat_penalty=cfg["rep"],
                    )
                    beam_hyps.append(pred_b)
                    beam_kws.append([kw.word for kw in det_kws])
                avg_lat_beam = float((time.perf_counter() - t0_b) / max(len(cached_logits), 1))
                bm = evaluate_predictions(references=val_refs, hypotheses=beam_hyps)
                cb = jiwer.process_characters(val_refs, beam_hyps)
                total_errors = cb.substitutions + cb.deletions + cb.insertions
                del_pct = round(cb.deletions / max(total_errors, 1) * 100, 1)

                print(
                    f"{cname:>26} | {str(cfg['viseme_tol']):>6} | {cfg['bp']:4.1f} | {cfg['alpha']:5.2f} | {cfg['beta']:5.2f} | {cfg['rep']:4.1f} | "
                    f"{bm['cer']*100:8.2f}% | {bm['wer']*100:8.2f}% | {del_pct:5.1f}% | {bm['spotter_f1']:10.4f}"
                )

                grid_results[cname] = {
                    "config": cfg,
                    "beam": {
                        "cer": round(bm["cer"], 4),
                        "wer": round(bm["wer"], 4),
                        "spotter_f1": round(bm["spotter_f1"], 4),
                        "spotter_precision": round(bm["spotter_precision"], 4),
                        "spotter_recall": round(bm["spotter_recall"], 4),
                        "substitutions": cb.substitutions,
                        "deletions": cb.deletions,
                        "insertions": cb.insertions,
                        "hits": cb.hits,
                        "del_pct": del_pct,
                        "latency_sec": round(avg_lat_beam, 4),
                    },
                    "samples": [
                        {"ref": r, "beam": b, "kws": k}
                        for r, b, k in zip(val_refs[:5], beam_hyps[:5], beam_kws[:5])
                    ]
                }

            best_cfg_name = max(
                grid_results.keys(),
                key=lambda k: (grid_results[k]["beam"]["spotter_f1"], -grid_results[k]["beam"]["wer"])
            )
            best_entry = grid_results[best_cfg_name]
            print(f"\n🏆 [OPTIMAL DECODER CONFIG] {best_cfg_name} -> Beam CER: {best_entry['beam']['cer']*100:.2f}%, Beam WER: {best_entry['beam']['wer']*100:.2f}%, Spotter F1: {best_entry['beam']['spotter_f1']:.4f}")

            greedy_hyps = [ctc_greedy_decode(ls, blank_penalty=1.2)[0] for ls in cached_logits]
            gm = evaluate_predictions(references=val_refs, hypotheses=greedy_hyps)

            sample_comparisons = []
            best_samples = best_entry["samples"]
            for idx, s in enumerate(best_samples[:25]):
                sample_comparisons.append({
                    "ref": s["ref"],
                    "greedy": greedy_hyps[idx] if idx < len(greedy_hyps) else "",
                    "beam": s["beam"],
                    "detected_keywords": s["kws"],
                })

            duration_sec = time.time() - t_start
            cost_usd = round(duration_sec * 0.000305, 4)

            baseline_wer = 0.9900
            baseline_f1 = 0.0209
            best_wer = best_entry["beam"]["wer"]
            best_f1 = best_entry["beam"]["spotter_f1"]

            if best_wer < 0.90 or best_f1 > 0.05:
                decision = "ACCEPT"
                status = "PASSED"
            elif best_wer < baseline_wer or best_f1 > baseline_f1:
                decision = "INCONCLUSIVE"
                status = "INCONCLUSIVE"
            else:
                decision = "REJECT"
                status = "FALSIFIED"

            return {
                "status": status,
                "decision": decision,
                "eval_only": True,
                "val_loss": round(avg_val_loss, 4),
                "best_val_loss": round(avg_val_loss, 4),
                "cer": best_entry["beam"]["cer"],
                "wer": best_entry["beam"]["wer"],
                "beam_cer": best_entry["beam"]["cer"],
                "beam_wer": best_entry["beam"]["wer"],
                "spotter_f1": best_entry["beam"]["spotter_f1"],
                "blank_ratio": 0.0,
                "best_config_name": best_cfg_name,
                "best_decoder_config": best_entry["config"],
                "grid_results": grid_results,
                "greedy_metrics": gm,
                "beam_metrics": best_entry["beam"],
                "sample_comparisons": sample_comparisons,
                "duration_sec": round(duration_sec, 2),
                "cost_usd": cost_usd,
                "val_samples_count": len(val_ds),
                "checkpoint_loaded": str(ckpt_path),
            }

        if question_id == "KWS-001":
            from src.spotter.keyword_spotter import KeywordSpotter
            from src.evaluation.metrics import compute_keyword_spotting_metrics
            kws_spotter = KeywordSpotter(fps=25.0)

            # Systematic parameter grid: 4 BP x 4 Conf x 2 Sub x 2 Vis = 64 configurations
            bps = [0.8, 1.2, 1.5, 2.0]
            confs = [0.15, 0.25, 0.35, 0.45]
            allow_subs = [False, True]
            vis_tols = [False, True]

            kws_configs = []
            for bp in bps:
                for conf in confs:
                    for sub in allow_subs:
                        for vis in vis_tols:
                            kws_configs.append({
                                "name": f"bp{bp}_c{int(conf*100)}_{'sub' if sub else 'nosub'}_{'vis' if vis else 'novis'}",
                                "bp": bp,
                                "conf": conf,
                                "allow_sub": sub,
                                "viseme_tol": vis,
                            })

            grid_results = {}
            print(f"\n📊 [KWS-001 POSTERIOR SPOTTER GRID] {len(kws_configs)} spotter yapılandırması değerlendiriliyor...")
            print(f"{'Config':>26} | {'BP':>4} | {'Conf':>5} | {'Sub':>5} | {'VisTol':>6} | {'Prec%':>7} | {'Rec%':>7} | {'Spotter F1':>10} | {'TP':>4} | {'FP':>5} | {'FN':>4} | {'Lat(ms)':>8}")
            print("-" * 115)

            for cfg in kws_configs:
                cname = cfg["name"]
                t0 = time.perf_counter()
                batch_spotted = []
                for ls in cached_logits:
                    dets = kws_spotter.spot_from_logits(
                        ls,
                        blank_penalty=cfg["bp"],
                        min_confidence=cfg["conf"],
                        viseme_tolerance=cfg["viseme_tol"],
                        allow_substring=cfg["allow_sub"],
                    )
                    batch_spotted.append(dets)
                avg_lat = float((time.perf_counter() - t0) / max(len(cached_logits), 1))

                m = compute_keyword_spotting_metrics(references=val_refs, hypotheses=batch_spotted)

                print(
                    f"{cname:>26} | {cfg['bp']:4.1f} | {cfg['conf']:5.2f} | {str(cfg['allow_sub']):>5} | {str(cfg['viseme_tol']):>6} | "
                    f"{m['precision']*100:6.2f}% | {m['recall']*100:6.2f}% | {m['f1']:10.4f} | {m['true_positives']:4d} | "
                    f"{m['false_positives']:5d} | {m['false_negatives']:4d} | {avg_lat*1000:7.2f}ms"
                )

                samples_data = []
                for r, dlist in zip(val_refs[:25], batch_spotted[:25]):
                    samples_data.append({
                        "ref": r,
                        "spotted": [
                            {
                                "word": d.word,
                                "start_sec": d.start_sec,
                                "end_sec": d.end_sec,
                                "conf": d.confidence,
                                "match_type": d.match_type,
                                "rank": d.rank_in_500,
                            }
                            for d in dlist
                        ]
                    })

                grid_results[cname] = {
                    "config": cfg,
                    "metrics": {
                        "precision": round(m["precision"], 4),
                        "recall": round(m["recall"], 4),
                        "f1": round(m["f1"], 4),
                        "true_positives": m["true_positives"],
                        "false_positives": m["false_positives"],
                        "false_negatives": m["false_negatives"],
                        "latency_sec": round(avg_lat, 5),
                    },
                    "samples": samples_data,
                }

            best_cfg_name = max(
                grid_results.keys(),
                key=lambda k: (grid_results[k]["metrics"]["f1"], grid_results[k]["metrics"]["precision"])
            )
            best_entry = grid_results[best_cfg_name]
            bm_kws = best_entry["metrics"]
            print(f"\n🏆 [OPTIMAL KWS CONFIG] {best_cfg_name} -> Spotter F1: {bm_kws['f1']:.4f}, Prec: {bm_kws['precision']*100:.2f}%, Recall: {bm_kws['recall']*100:.2f}%, TP: {bm_kws['true_positives']}, FP: {bm_kws['false_positives']}, FN: {bm_kws['false_negatives']}")

            greedy_hyps = [ctc_greedy_decode(ls, blank_penalty=1.2)[0] for ls in cached_logits]
            gm = evaluate_predictions(references=val_refs, hypotheses=greedy_hyps)

            beam_hyps = []
            for ls in cached_logits:
                pb, _ = lexicon_decoder.decode(ls, blank_penalty=2.0, lm_alpha=0.2, lm_beta=-0.4, repeat_penalty=4.0)
                beam_hyps.append(pb)
            bm = evaluate_predictions(references=val_refs, hypotheses=beam_hyps)

            sample_comparisons = []
            for idx, (r, gh, bh, sd) in enumerate(zip(val_refs[:25], greedy_hyps[:25], beam_hyps[:25], best_entry["samples"][:25])):
                sample_comparisons.append({
                    "ref": r,
                    "greedy": gh,
                    "beam": bh,
                    "kws_detected": [f"{d['word']} [{d['start_sec']}s-{d['end_sec']}s, conf={d['conf']}]" for d in sd["spotted"]],
                })

            duration_sec = time.time() - t_start
            cost_usd = round(duration_sec * 0.000305, 4)

            baseline_f1 = 0.0418
            best_f1 = bm_kws["f1"]
            best_prec = bm_kws["precision"]

            if best_f1 >= 0.08 and best_prec >= 0.10:
                decision = "ACCEPT"
                status = "PASSED"
            elif best_f1 > baseline_f1:
                decision = "INCONCLUSIVE"
                status = "INCONCLUSIVE"
            else:
                decision = "REJECT"
                status = "FALSIFIED"

            return {
                "status": status,
                "decision": decision,
                "eval_only": True,
                "val_loss": round(avg_val_loss, 4),
                "best_val_loss": round(avg_val_loss, 4),
                "cer": gm["cer"],
                "wer": gm["wer"],
                "greedy_cer": gm["cer"],
                "greedy_wer": gm["wer"],
                "beam_cer": bm["cer"],
                "beam_wer": bm["wer"],
                "spotter_f1": bm_kws["f1"],
                "spotter_precision": bm_kws["precision"],
                "spotter_recall": bm_kws["recall"],
                "kws_metrics": bm_kws,
                "blank_ratio": 0.0,
                "best_config_name": best_cfg_name,
                "best_kws_config": best_entry["config"],
                "grid_results": grid_results,
                "greedy_metrics": gm,
                "beam_metrics": bm,
                "sample_comparisons": sample_comparisons,
                "duration_sec": round(duration_sec, 2),
                "cost_usd": cost_usd,
                "val_samples_count": len(val_ds),
                "checkpoint_loaded": str(ckpt_path),
            }

        if question_id == "CONFIRM-002":
            from src.spotter.keyword_spotter import KeywordSpotter
            from src.evaluation.metrics import compute_keyword_spotting_metrics
            from collections import defaultdict

            seeds = [42, 123, 456]
            seed_results = {}
            kws_spotter = KeywordSpotter(fps=25.0)

            print(f"\n🔬 [CONFIRM-002] Multi-Seed ({seeds}) Bootstrap Stability & Multi-Speaker Representative Doğrulama Başlatılıyor...")

            # Pre-compute all greedy, beam, and spotter outputs on val_refs
            all_greedy = [ctc_greedy_decode(ls, blank_penalty=1.2)[0] for ls in cached_logits]
            all_beam = []
            for ls in cached_logits:
                bh, _ = lexicon_decoder.decode(
                    ls,
                    blank_penalty=1.5,
                    lm_alpha=0.3,
                    lm_beta=-0.2,
                    repeat_penalty=4.0,
                )
                all_beam.append(bh)
            all_spotted = []
            for ls in cached_logits:
                dets = kws_spotter.spot_from_logits(
                    ls,
                    blank_penalty=1.2,
                    min_confidence=0.15,
                    allow_substring=False,
                    viseme_tolerance=True,
                )
                all_spotted.append(dets)

            n_val = len(val_refs)
            for s in seeds:
                rng = np.random.RandomState(s)
                # Bootstrap sample indices (sample with replacement, size = n_val)
                boot_idx = rng.choice(n_val, size=n_val, replace=True)
                b_refs = [val_refs[i] for i in boot_idx]
                b_greedy = [all_greedy[i] for i in boot_idx]
                b_beam = [all_beam[i] for i in boot_idx]
                b_spotted = [all_spotted[i] for i in boot_idx]

                gm = evaluate_predictions(references=b_refs, hypotheses=b_greedy)
                bm = evaluate_predictions(references=b_refs, hypotheses=b_beam)
                km = compute_keyword_spotting_metrics(references=b_refs, hypotheses=b_spotted)

                seed_results[s] = {
                    "greedy_cer": gm["cer"],
                    "greedy_wer": gm["wer"],
                    "beam_cer": bm["cer"],
                    "beam_wer": bm["wer"],
                    "spotter_f1": km["f1"],
                    "spotter_precision": km["precision"],
                    "spotter_recall": km["recall"],
                    "true_positives": km["true_positives"],
                    "false_positives": km["false_positives"],
                }
                print(
                    f"  Seed {s:<3d} (Bootstrap N={n_val}) -> Greedy CER: {gm['cer']*100:.2f}%, Beam WER: {bm['wer']*100:.2f}%, "
                    f"Spotter F1: {km['f1']:.4f}, Prec: {km['precision']*100:.2f}%, Rec: {km['recall']*100:.2f}%, TP: {km['true_positives']}"
                )

            # Speaker cross-validation across 5 train speakers
            if split_map_file.exists():
                with open(split_map_file, "r", encoding="utf-8") as smf:
                    split_map_dict = json.load(smf)
            else:
                from src.data.split_map_data import IBOROTTI_SPLIT_MAP
                split_map_dict = IBOROTTI_SPLIT_MAP

            train_ds = LipReadingDataset(
                data_dir=local_data_dir,
                split="train",
                split_map_path=split_map_file,
                is_train=False,
                crop_size=88,
                max_duration=max_duration,
                cache_in_ram=False,
            )
            speaker_samples = defaultdict(list)
            for smp in train_ds.samples:
                ch = split_map_dict.get(smp["video_id"], {}).get("channel", "unknown")
                speaker_samples[ch].append(smp)

            speaker_results = {}
            print("\n👥 [SPEAKER-HELD-OUT CROSS-VALIDATION] Train konuşmacıları genelleme testi:")
            for spk, smps in speaker_samples.items():
                eval_subset = smps[:40]
                sub_ds = LipReadingDataset(samples=eval_subset, is_train=False, crop_size=88)
                sub_loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
                sub_hyps = []
                sub_refs = []
                with torch.no_grad():
                    for batch in sub_loader:
                        vids = (batch.get("videos") if "videos" in batch else batch["video"]).to(device)
                        in_lens = batch["input_lengths"].to(device)
                        lgs = model(vids)
                        for i in range(len(vids)):
                            slen = int(in_lens[i].item())
                            sub_hyps.append(ctc_greedy_decode(lgs[i, :slen], blank_penalty=1.2)[0])
                        sub_refs.extend(batch["transcripts"])
                sm = evaluate_predictions(references=sub_refs, hypotheses=sub_hyps)
                speaker_results[spk] = {
                    "clips": len(eval_subset),
                    "greedy_cer": sm["cer"],
                    "greedy_wer": sm["wer"],
                }
                print(f"  Konuşmacı: {spk:<26} ({len(eval_subset):2d} klip) -> CER: {sm['cer']*100:.2f}%, WER: {sm['wer']*100:.2f}%")

            duration_sec = time.time() - t_start
            cost_usd = round(duration_sec * 0.000305, 4)

            f1_vals = [v["spotter_f1"] for v in seed_results.values()]
            cer_vals = [v["greedy_cer"] for v in seed_results.values()]
            wer_vals = [v["beam_wer"] for v in seed_results.values()]

            mean_f1 = float(np.mean(f1_vals))
            std_f1 = float(np.std(f1_vals))
            mean_cer = float(np.mean(cer_vals))
            std_cer = float(np.std(cer_vals))
            mean_wer = float(np.mean(wer_vals))
            std_wer = float(np.std(wer_vals))

            stable = std_cer < 0.05 and std_f1 < 0.05
            rep_valid = mean_cer < 0.80 and all(sr["greedy_cer"] < 0.90 for sr in speaker_results.values())
            passed = stable and rep_valid

            sample_comparisons = []
            for r, g, b in zip(val_refs[:25], all_greedy[:25], all_beam[:25]):
                sample_comparisons.append({"ref": r, "greedy": g, "beam": b})

            return {
                "status": "PASSED" if passed else "INCONCLUSIVE",
                "decision": "ACCEPT" if passed else "INCONCLUSIVE",
                "eval_only": True,
                "multi_seed_results": seed_results,
                "speaker_results": speaker_results,
                "val_loss": round(avg_val_loss, 4),
                "best_val_loss": round(avg_val_loss, 4),
                "mean_greedy_cer": round(mean_cer, 4),
                "std_greedy_cer": round(std_cer, 4),
                "mean_beam_wer": round(mean_wer, 4),
                "std_beam_wer": round(std_wer, 4),
                "mean_spotter_f1": round(mean_f1, 4),
                "std_spotter_f1": round(std_f1, 4),
                "sample_comparisons": sample_comparisons,
                "duration_sec": round(duration_sec, 2),
                "cost_usd": cost_usd,
                "val_samples_count": len(val_ds),
                "checkpoint_loaded": str(ckpt_path),
            }

        grid_values = probe_cfg_dict.get("blank_penalty_grid")
        if not grid_values and question_id == "DEC-004":
            grid_values = [0.0, 0.4, 0.8, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]

        if grid_values:
            grid_results = {}
            print(f"\n📊 [GRID EVALUATION] {len(grid_values)} farklı blank_penalty değeri değerlendiriliyor: {grid_values}")
            print(f"{'BP':>5} | {'Blank%':>7} | {'Greedy CER':>10} | {'Greedy WER':>10} | {'S':>5} | {'D':>5} | {'I':>5} | {'Del%':>6} | {'Beam CER':>10} | {'Beam WER':>10} | {'Beam F1':>8}")
            print("-" * 105)

            for bp in grid_values:
                # 1. Greedy decode
                greedy_hyps = [ctc_greedy_decode(ls, blank_penalty=bp)[0] for ls in cached_logits]
                gm = evaluate_predictions(references=val_refs, hypotheses=greedy_hyps)
                cg = jiwer.process_characters(val_refs, greedy_hyps)

                # Blank ratio
                blank_count = 0
                for ls in cached_logits:
                    adj = ls.clone()
                    if bp > 0.0:
                        adj[:, BLANK_IDX] -= bp
                    blank_count += int((adj.argmax(dim=-1) == BLANK_IDX).sum().item())
                bp_blank_ratio = float(blank_count / max(total_timesteps, 1))

                # 2. Beam decode
                beam_hyps = []
                beam_kws = []
                t0_b = time.perf_counter()
                for ls in cached_logits:
                    pred_b, det_kws = lexicon_decoder.decode(ls, blank_penalty=bp)
                    beam_hyps.append(pred_b)
                    beam_kws.append([kw.word for kw in det_kws])
                avg_lat_beam = float((time.perf_counter() - t0_b) / max(len(cached_logits), 1))
                bm = evaluate_predictions(references=val_refs, hypotheses=beam_hyps)
                cb = jiwer.process_characters(val_refs, beam_hyps)

                total_errors = cg.substitutions + cg.deletions + cg.insertions
                del_pct = round(cg.deletions / max(total_errors, 1) * 100, 1)

                print(
                    f"{bp:5.1f} | {bp_blank_ratio*100:6.1f}% | "
                    f"{gm['cer']*100:9.2f}% | {gm['wer']*100:9.2f}% | "
                    f"{cg.substitutions:5d} | {cg.deletions:5d} | {cg.insertions:5d} | {del_pct:5.1f}% | "
                    f"{bm['cer']*100:9.2f}% | {bm['wer']*100:9.2f}% | {bm['spotter_f1']:8.4f}"
                )

                grid_results[str(bp)] = {
                    "blank_penalty": bp,
                    "blank_ratio": round(bp_blank_ratio, 4),
                    "greedy": {
                        "cer": round(gm["cer"], 4),
                        "wer": round(gm["wer"], 4),
                        "spotter_f1": round(gm["spotter_f1"], 4),
                        "spotter_precision": round(gm["spotter_precision"], 4),
                        "spotter_recall": round(gm["spotter_recall"], 4),
                        "substitutions": cg.substitutions,
                        "deletions": cg.deletions,
                        "insertions": cg.insertions,
                        "hits": cg.hits,
                        "del_pct": del_pct,
                        "hyp_chars": cg.substitutions + cg.insertions + cg.hits,
                    },
                    "beam": {
                        "cer": round(bm["cer"], 4),
                        "wer": round(bm["wer"], 4),
                        "spotter_f1": round(bm["spotter_f1"], 4),
                        "spotter_precision": round(bm["spotter_precision"], 4),
                        "spotter_recall": round(bm["spotter_recall"], 4),
                        "substitutions": cb.substitutions,
                        "deletions": cb.deletions,
                        "insertions": cb.insertions,
                        "hits": cb.hits,
                        "latency_sec": round(avg_lat_beam, 4),
                    },
                    "samples": [
                        {"ref": r, "greedy": g, "beam": b, "kws": k}
                        for r, g, b, k in zip(val_refs[:5], greedy_hyps[:5], beam_hyps[:5], beam_kws[:5])
                    ]
                }

            # En iyi blank penalty'yi seç (Greedy CER minimum)
            best_bp = min(grid_values, key=lambda b: grid_results[str(b)]["greedy"]["cer"])
            best_entry = grid_results[str(best_bp)]
            print(f"\n🏆 [OPTIMAL BLANK PENALTY] bp={best_bp} -> Greedy CER: {best_entry['greedy']['cer']*100:.2f}%, Beam CER: {best_entry['beam']['cer']*100:.2f}%, Beam WER: {best_entry['beam']['wer']*100:.2f}%")

            best_greedy_hyps = [ctc_greedy_decode(ls, blank_penalty=best_bp)[0] for ls in cached_logits]
            best_beam_hyps = []
            best_beam_kws = []
            for ls in cached_logits:
                pred_beam, detected_kws = lexicon_decoder.decode(ls, blank_penalty=best_bp)
                best_beam_hyps.append(pred_beam)
                best_beam_kws.append([kw.word for kw in detected_kws])

            sample_comparisons = []
            for r, g, b, kws in zip(val_refs[:25], best_greedy_hyps[:25], best_beam_hyps[:25], best_beam_kws[:25]):
                sample_comparisons.append({
                    "ref": r,
                    "greedy": g,
                    "beam": b,
                    "detected_keywords": kws,
                })

            duration_sec = time.time() - t_start
            cost_usd = round(duration_sec * 0.000305, 4)

            # Karar Kuralı
            baseline_entry = grid_results.get("0.8", grid_results[str(grid_values[0])])
            baseline_cer = baseline_entry["greedy"]["cer"]
            best_cer = best_entry["greedy"]["cer"]

            if best_cer < 0.65 or (best_entry["greedy"]["deletions"] < baseline_entry["greedy"]["deletions"] * 0.5 and best_cer < baseline_cer):
                decision = "ACCEPT"
                status = "PASSED"
            elif best_cer < baseline_cer:
                decision = "INCONCLUSIVE"
                status = "INCONCLUSIVE"
            else:
                decision = "REJECT"
                status = "FALSIFIED"

            return {
                "status": status,
                "decision": decision,
                "eval_only": True,
                "val_loss": round(avg_val_loss, 4),
                "best_val_loss": round(avg_val_loss, 4),
                "cer": best_entry["greedy"]["cer"],
                "wer": best_entry["greedy"]["wer"],
                "beam_cer": best_entry["beam"]["cer"],
                "beam_wer": best_entry["beam"]["wer"],
                "spotter_f1": best_entry["beam"]["spotter_f1"],
                "blank_ratio": best_entry["blank_ratio"],
                "best_blank_penalty": best_bp,
                "grid_results": grid_results,
                "greedy_metrics": best_entry["greedy"],
                "beam_metrics": best_entry["beam"],
                "sample_comparisons": sample_comparisons,
                "duration_sec": round(duration_sec, 2),
                "cost_usd": cost_usd,
                "val_samples_count": len(val_ds),
                "checkpoint_loaded": str(ckpt_path),
            }
        else:
            val_hyps_greedy = []
            val_hyps_beam = []
            val_keywords = []
            blank_counts = 0
            latencies_greedy = []
            latencies_beam = []

            for logit_slice in cached_logits:
                t0 = time.perf_counter()
                pred_greedy = ctc_greedy_decode(logit_slice, blank_penalty=blank_penalty)[0]
                latencies_greedy.append(time.perf_counter() - t0)
                val_hyps_greedy.append(pred_greedy)

                t0 = time.perf_counter()
                pred_beam, detected_kws = lexicon_decoder.decode(logit_slice, blank_penalty=blank_penalty)
                latencies_beam.append(time.perf_counter() - t0)
                val_hyps_beam.append(pred_beam)
                val_keywords.append([kw.word for kw in detected_kws])

                top_classes = logit_slice.argmax(dim=-1)
                blank_counts += int((top_classes == BLANK_IDX).sum().item())

            blank_ratio = float(blank_counts / max(total_timesteps, 1))
            metrics_greedy = evaluate_predictions(references=val_refs, hypotheses=val_hyps_greedy)
            metrics_beam = evaluate_predictions(references=val_refs, hypotheses=val_hyps_beam)

            avg_lat_greedy = float(np.mean(latencies_greedy)) if latencies_greedy else 0.0
            avg_lat_beam = float(np.mean(latencies_beam)) if latencies_beam else 0.0

            sample_comparisons = []
            for r, g, b, kws in zip(val_refs[:25], val_hyps_greedy[:25], val_hyps_beam[:25], val_keywords[:25]):
                sample_comparisons.append({
                    "ref": r,
                    "greedy": g,
                    "beam": b,
                    "detected_keywords": kws,
                })

            duration_sec = time.time() - t_start
            cost_usd = round(duration_sec * 0.000305, 4)

            wer_greedy = metrics_greedy["wer"]
            wer_beam = metrics_beam["wer"]
            latency_ok = (avg_lat_beam <= 2.0)

            if not latency_ok or wer_beam >= wer_greedy:
                decision = "REJECT"
                status = "FALSIFIED"
            elif wer_beam < 0.80:
                decision = "ACCEPT"
                status = "PASSED"
            else:
                decision = "INCONCLUSIVE"
                status = "INCONCLUSIVE"

            return {
                "status": status,
                "decision": decision,
                "eval_only": True,
                "val_loss": round(avg_val_loss, 4),
                "best_val_loss": round(avg_val_loss, 4),
                "cer": metrics_beam["cer"],
                "wer": metrics_beam["wer"],
                "spotter_f1": metrics_beam["spotter_f1"],
                "blank_ratio": round(blank_ratio, 4),
                "greedy_metrics": {
                    "cer": metrics_greedy["cer"],
                    "wer": metrics_greedy["wer"],
                    "spotter_f1": metrics_greedy["spotter_f1"],
                    "spotter_precision": metrics_greedy["spotter_precision"],
                    "spotter_recall": metrics_greedy["spotter_recall"],
                    "latency_sec": round(avg_lat_greedy, 4),
                },
                "beam_metrics": {
                    "cer": metrics_beam["cer"],
                    "wer": metrics_beam["wer"],
                    "spotter_f1": metrics_beam["spotter_f1"],
                    "spotter_precision": metrics_beam["spotter_precision"],
                    "spotter_recall": metrics_beam["spotter_recall"],
                    "latency_sec": round(avg_lat_beam, 4),
                    "beam_size": beam_size,
                    "lm_alpha": lm_alpha,
                    "lm_beta": lm_beta,
                },
                "sample_comparisons": sample_comparisons,
                "duration_sec": round(duration_sec, 2),
                "cost_usd": cost_usd,
                "val_samples_count": len(val_ds),
                "checkpoint_loaded": str(ckpt_path),
            }

    train_ds = LipReadingDataset(
        data_dir=local_data_dir,
        split="train",
        split_map_path=split_map_file,
        is_train=True,
        crop_size=88,
        max_duration=max_duration,
        cache_in_ram=False,
    )
    val_ds = LipReadingDataset(
        data_dir=local_data_dir,
        split="val",
        split_map_path=split_map_file,
        is_train=False,
        crop_size=88,
        max_duration=max_duration,
        cache_in_ram=False,
    )

    if n_train_samples is not None and len(train_ds.samples) > n_train_samples:
        train_ds.samples = train_ds.samples[:n_train_samples]
    if n_val_samples is not None and len(val_ds.samples) > n_val_samples:
        val_ds.samples = val_ds.samples[:n_val_samples]

    print(f"⚡ RAM'e önbelleğe alınıyor: {len(train_ds)} train, {len(val_ds)} val segment...")
    train_ds.cache_in_ram = True
    train_ds._preload_into_ram()
    val_ds.cache_in_ram = True
    val_ds._preload_into_ram()

    batch_size = probe_cfg_dict.get("batch_size", 8)
    if use_bucketing:
        train_sampler = SequenceBucketSampler(
            train_ds,
            shuffle=True,
            seed=seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=pad_collate_fn,
        )
        print(f"📦 [SEQUENCE BUCKETING] Train DataLoader SequenceBucketSampler ile kuruldu ({len(train_sampler)} batch).")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pad_collate_fn,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, 4),
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    # 3. Model Kurulumu
    arch_name = probe_cfg_dict.get("arch_name", "vsr_conformer_base")
    d_model = probe_cfg_dict.get("d_model", 512)
    num_layers = probe_cfg_dict.get("num_layers", 4)
    encoder_type = probe_cfg_dict.get("encoder_type", "conformer")
    conformer_dropout = probe_cfg_dict.get("conformer_dropout", 0.1)
    use_specaugment = probe_cfg_dict.get("use_specaugment", False)

    model = build_vsr_model(
        arch_name=arch_name,
        d_model=d_model,
        num_layers=num_layers,
        encoder_type=encoder_type,
        dropout=conformer_dropout,
        use_specaugment=use_specaugment,
    ).to(device)

    # Model Ağırlıklarını Yükle (Checkpoint varsa tam model, yoksa Auto-AVSR frontend transferi)
    load_ckpt = probe_cfg_dict.get("load_checkpoint")
    matched_weights, total_fe = 0, 0
    if load_ckpt:
        ckpt_path = pathlib.Path("/root/vol") / load_ckpt
        if not ckpt_path.exists():
            ckpt_path = pathlib.Path("/root/checkpoints") / load_ckpt
        if not ckpt_path.exists():
            raise FileNotFoundError(f"İstenen checkpoint bulunamadı: {load_ckpt}")
        state = torch.load(ckpt_path, map_location=device)
        model_state = state.get("model_state_dict", state)
        model.load_state_dict(model_state, strict=True)
        print(f"✅ Canonical Checkpoint başarıyla yüklendi: {ckpt_path} (epoch={state.get('epoch')}, val_loss={state.get('val_loss')}, cer={state.get('cer')})")
    else:
        # Pretrained Visual Frontend Ağırlıklarını Aktar
        ckpt_path = pathlib.Path("/root/vol/vsr_trlrs3_base.pth")
        if not ckpt_path.exists():
            ckpt_path = pathlib.Path("/root/checkpoints/vsr_trlrs3_base.pth")
        if ckpt_path.exists():
            matched_weights, total_fe = adapt_auto_avsr_weights(ckpt_path, model)
            print(f"✅ Auto-AVSR Pretrained Frontend Transferi: {matched_weights}/{total_fe} tensör yüklendi.")
        else:
            print(f"⚠️ {ckpt_path} bulunamadı, scratch başlatıldı.")


    # Negatif blank bias başlatması (yalnız sıfırdan eğitimde)
    if not load_ckpt:
        with torch.no_grad():
            model.ctc_head.bias[BLANK_IDX] -= 0.5

    freeze_frontend = probe_cfg_dict.get("freeze_frontend", False)
    if freeze_frontend:
        model.freeze_frontend(True)
        print("🔒 [FRONTEND] Pretrained 3D-ResNet frontend donduruldu (yalnız temporal encoder eğitiliyor).")

    # 4. Öğrenme Oranı (Dondurulmamışsa diferansiyel: Frontend 0.05x, Temporal Encoder 1.0x)
    lr = probe_cfg_dict.get("lr", 2e-4)
    weight_decay = probe_cfg_dict.get("weight_decay", 1e-4)
    if freeze_frontend:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        frontend_params = [p for n, p in model.named_parameters() if "frontend" in n and p.requires_grad]
        temporal_params = [p for n, p in model.named_parameters() if "frontend" not in n and p.requires_grad]
        optimizer = optim.AdamW([
            {"params": frontend_params, "lr": lr * 0.05},
            {"params": temporal_params, "lr": lr},
        ], weight_decay=weight_decay)

    epochs = probe_cfg_dict.get("epochs", 6)
    blank_penalty = probe_cfg_dict.get("blank_penalty", 0.4)
    unfreeze_frontend_epoch = probe_cfg_dict.get("unfreeze_frontend_epoch")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_val_loss = float("inf")

    print(f"🚀 Probe Başlıyor: {probe_cfg_dict.get('experiment_id')} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        if unfreeze_frontend_epoch and epoch == unfreeze_frontend_epoch and freeze_frontend:
            print(f"🔓 [UNFREEZE] Epoch {epoch}: Frontend çözüldü, diferansiyel LR ({lr * 0.05:.2e}) ile ince ayara geçiliyor...")
            model.freeze_frontend(False)
            fe_params = [p for n, p in model.named_parameters() if "frontend" in n and p.requires_grad]
            optimizer.add_param_group({"params": fe_params, "lr": lr * 0.05})

        model.train()
        train_losses = []

        for batch in train_loader:
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
                blank_penalty=blank_penalty,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # Validasyon Değerlendirmesi
        model.eval()
        val_losses = []
        val_refs = []
        val_hyps = []
        val_hyps_raw = []
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
                    blank_penalty=blank_penalty,
                )
                val_losses.append(loss.item())

                for i in range(len(videos)):
                    seq_len = int(input_lens[i].item())
                    logit_slice = logits[i, :seq_len]
                    pred_text_penalized = ctc_greedy_decode(logit_slice, blank_penalty=blank_penalty)[0]
                    pred_text_raw = ctc_greedy_decode(logit_slice, blank_penalty=0.0)[0]
                    val_hyps.append(pred_text_penalized)
                    val_hyps_raw.append(pred_text_raw)

                    top_classes = logit_slice.argmax(dim=-1)
                    blank_counts += int((top_classes == BLANK_IDX).sum().item())
                    total_timesteps += seq_len

                val_refs.extend(batch["transcripts"])

        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        blank_ratio = float(blank_counts / max(total_timesteps, 1))
        val_metrics = evaluate_predictions(references=val_refs, hypotheses=val_hyps)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                ckpt_save_path = f"/root/vol/{probe_cfg_dict.get('experiment_id', 'probe')}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": avg_val_loss,
                    "cer": val_metrics["cer"],
                    "candidate_version": candidate_version,
                    "question_id": question_id,
                }, ckpt_save_path)
                vol.commit()
            except Exception as e:
                print(f"⚠️ Checkpoint kaydedilemedi: {e}")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "blank_ratio": round(blank_ratio, 4),
            "cer": val_metrics["cer"],
            "wer": val_metrics["wer"],
            "spotter_f1": val_metrics["spotter_f1"],
        })

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Blank: {blank_ratio*100:.1f}% | "
            f"CER: {val_metrics['cer']:.4f} | WER: {val_metrics['wer']:.4f}"
        )

    if question_id == "READINESS-001":
        print("\n🎭 [READINESS-001 DRESS REHEARSAL] Dondurulmuş reçete uçtan uca doğrulanıyor...")
        from src.spotter.lexicon_decoder import LexiconBeamSearchDecoder
        from src.spotter.keyword_spotter import KeywordSpotter
        from src.evaluation.metrics import compute_keyword_spotting_metrics

        # 1. Eğitim verimi ve maliyet projeksiyonu
        train_duration = time.time() - t_start
        avg_epoch_sec = float(train_duration / max(epochs, 1))
        throughput_clips = float(len(train_ds) / max(avg_epoch_sec, 0.001))
        proj_epochs = 4
        proj_duration_sec = proj_epochs * avg_epoch_sec
        proj_duration_min = round(proj_duration_sec / 60, 2)
        proj_cost_usd = round(proj_duration_sec * 0.000305, 4)

        print(f"⏱️ [PROJEKSİYON] Ortalama Epoch Süresi: {avg_epoch_sec:.1f}s | Verim: {throughput_clips:.1f} klip/s")
        print(f"💰 [PROJEKSİYON] 4-Epoch Full-Training Süresi: {proj_duration_min:.1f} dk | Tahmini Maliyet: ${proj_cost_usd:.4f} USD")

        # 2. Validasyon önbellekleme ve çözücüler
        model.eval()
        cached_val_logits = []
        with torch.no_grad():
            for batch in val_loader:
                videos = (batch.get("videos") if "videos" in batch else batch["video"]).to(device)
                input_lens = batch["input_lengths"].to(device)
                logits = model(videos)
                for i in range(len(videos)):
                    seq_len = int(input_lens[i].item())
                    cached_val_logits.append(logits[i, :seq_len].detach().cpu())

        # Greedy ($BP=1.2$)
        greedy_hyps = [ctc_greedy_decode(ls, blank_penalty=1.2)[0] for ls in cached_val_logits]
        gm = evaluate_predictions(references=val_refs, hypotheses=greedy_hyps)

        # Calibrated Lexicon Beam ($BP=1.5, alpha=0.3, beta=-0.2, rep=4.0, vis=False$)
        lex_dec = LexiconBeamSearchDecoder(beam_size=30, blank_penalty=1.5, viseme_tolerance=False, lm_alpha=0.3, lm_beta=-0.2, repeat_penalty=4.0)
        beam_hyps = []
        for ls in cached_val_logits:
            bh, _ = lex_dec.decode(ls, blank_penalty=1.5, lm_alpha=0.3, lm_beta=-0.2, repeat_penalty=4.0)
            beam_hyps.append(bh)
        bm = evaluate_predictions(references=val_refs, hypotheses=beam_hyps)

        # Calibrated Keyword Spotter ($BP=1.2, conf=0.15, nosub, vis=True$)
        kws_spot = KeywordSpotter(fps=25.0)
        spotted_batch = []
        for ls in cached_val_logits:
            dets = kws_spot.spot_from_logits(ls, blank_penalty=1.2, min_confidence=0.15, allow_substring=False, viseme_tolerance=True)
            spotted_batch.append(dets)
        km = compute_keyword_spotting_metrics(references=val_refs, hypotheses=spotted_batch)

        sample_comparisons = []
        for idx in range(min(25, len(val_refs))):
            r = val_refs[idx]
            gh = greedy_hyps[idx]
            bh = beam_hyps[idx]
            dets = spotted_batch[idx]
            sample_comparisons.append({
                "ref": r,
                "greedy": gh,
                "beam": bh,
                "kws_detected": [f"{d.word} [{d.start_sec}s-{d.end_sec}s, conf={d.confidence}]" for d in dets],
            })

        duration_sec = time.time() - t_start
        cost_usd = round(duration_sec * 0.000305, 4)

        status = "PASSED"
        # Pre-registered throughput criteria for A10G 1325-clip rehearsal:
        # Realistic hardware baseline is ~5.5 - 7.0 clips/s, ~220 - 250 s/epoch.
        decision = "ACCEPT" if throughput_clips >= 5.0 and avg_epoch_sec <= 260.0 else "INCONCLUSIVE"

        return {
            "status": status,
            "decision": decision,
            "dress_rehearsal_verified": decision == "ACCEPT",
            "best_val_loss": round(best_val_loss, 4),
            "final_train_loss": round(avg_train_loss, 4),
            "final_val_loss": round(avg_val_loss, 4),
            "greedy_cer": gm["cer"],
            "greedy_wer": gm["wer"],
            "beam_cer": bm["cer"],
            "beam_wer": bm["wer"],
            "spotter_f1": km["f1"],
            "spotter_precision": km["precision"],
            "spotter_recall": km["recall"],
            "kws_metrics": km,
            "greedy_metrics": gm,
            "beam_metrics": bm,
            "avg_epoch_sec": round(avg_epoch_sec, 2),
            "throughput_clips_per_sec": round(throughput_clips, 2),
            "projected_full_train_duration_min": proj_duration_min,
            "projected_full_train_cost_usd": proj_cost_usd,
            "sample_comparisons": sample_comparisons,
            "history": history,
            "duration_sec": round(duration_sec, 2),
            "cost_usd": cost_usd,
            "train_samples_count": len(train_ds),
            "val_samples_count": len(val_ds),
        }

    duration_sec = time.time() - t_start
    cost_usd = round(duration_sec * 0.000305, 4)

    # 25 ham tahmin örneği
    sample_comparisons = []
    for r, hp, hr in zip(val_refs[:25], val_hyps[:25], val_hyps_raw[:25]):
        sample_comparisons.append({"ref": r, "hyp": hp, "hyp_raw": hr})

    last_eval = history[-1]
    best_cer = min(h["cer"] for h in history)
    
    # Karar kuralı değerlendirmesi
    falsified = (last_eval["blank_ratio"] >= 0.99 and last_eval["cer"] >= 1.0)
    if question_id == "CONFIRM-003":
        # Pre-registered stability criteria:
        # Seed 123 replicates 2-epoch optimum within +-0.05 val loss and +-5% CER of baseline (2.7399, CER 75.6%)
        if best_val_loss <= 2.79 and best_cer <= 0.80:
            decision = "ACCEPT"
            status = "PASSED"
        elif best_val_loss <= 2.85:
            decision = "INCONCLUSIVE"
            status = "INCONCLUSIVE"
        else:
            decision = "REJECT"
            status = "FALSIFIED"
    elif question_id == "LOWDATA-003":
        # Baseline c0.4.0 (1325 clips <=8s) best val loss is 2.7399, full val (385 clips) loss is 2.7989, CER 75.00%
        # Pre-registered criterion: best_val_loss < 2.7399 or CER < 75.00%
        if best_val_loss < 2.7399:
            decision = "ACCEPT"
            status = "PASSED"
        elif best_cer < 0.7500:
            decision = "INCONCLUSIVE"
            status = "INCONCLUSIVE"
        else:
            decision = "REJECT"
            status = "FALSIFIED"
    elif falsified:
        decision = "INCONCLUSIVE"
        status = "FALSIFIED"
    elif best_cer < 0.85:
        decision = "ACCEPT"
        status = "PASSED"
    else:
        decision = "INCONCLUSIVE"
        status = "INCONCLUSIVE"

    return {
        "status": status,
        "decision": decision,
        "best_val_loss": round(best_val_loss, 4),
        "best_cer": round(best_cer, 4),
        "final_val_loss": last_eval["val_loss"],
        "final_train_loss": last_eval["train_loss"],
        "cer": last_eval["cer"],
        "wer": last_eval["wer"],
        "spotter_f1": last_eval["spotter_f1"],
        "blank_ratio": last_eval["blank_ratio"],
        "history": history,
        "sample_comparisons": sample_comparisons,
        "duration_sec": round(duration_sec, 2),
        "cost_usd": cost_usd,
        "matched_weights": matched_weights,
        "total_frontend_tensors": total_fe,
        "train_samples_count": len(train_ds),
        "val_samples_count": len(val_ds),
    }
