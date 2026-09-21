"""
src/modal_runner/cloud_train.py — Modal A10G Bulut Eğitim ve Doğrulama Uygulaması
Workspace: ibrahimgozlukaya
Secret: bucket-credentials
"""

import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import modal
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# 1. Modal İmaj Tanımı (Debian Slim + FFmpeg + PyTorch + PyAV + Boto3)
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
)

app = modal.App("turkish-vsr-cloud-training")
vol = modal.Volume.from_name("turkish-vsr-vol", create_if_missing=True)
s3_secret = modal.Secret.from_name("bucket-credentials")

@app.function(
    image=image,
    gpu="A10G",
    secrets=[s3_secret],
    volumes={"/root/checkpoints": vol},
    timeout=600,
)
def inspect_checkpoint_remote() -> Dict[str, Any]:
    import boto3
    import torch
    from src.data.s3_downloader import S3DatasetDownloader

    ckpt_path = pathlib.Path("/root/checkpoints/vsr_trlrs3_base.pth")
    if not ckpt_path.exists():
        downloader = S3DatasetDownloader(
            bucket=os.environ.get("S3_BUCKET_NAME", "lipreading-data-emre2026"),
            region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            access_key=os.environ.get("AWS_ACCESS_KEY_ID") or None,
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
            master_dir=pathlib.Path("/root/data/master"),
        )
        downloader.download_checkpoint(dest_file=ckpt_path)
        vol.commit()

    print(f"Checkpoint boyutu: {ckpt_path.stat().st_size / (1024*1024):.2f} MB")
    data = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Top-level anahtarlar: {keys}")
        sd = data.get("state_dict", data.get("model_state_dict", data))
    else:
        sd = data

    sd_keys = list(sd.keys())
    print(f"Toplam tensör sayısı: {len(sd_keys)}")
    print("İlk 25 anahtar:")
    for k in sd_keys[:25]:
        print(f"  {k}: {sd[k].shape}")

    from src.models.vsr_conformer import VSRConformerModel
    from src.vocab.turkish_vocab import VOCAB_SIZE

    test_model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=512, num_layers=4, encoder_type="conformer")
    matched, total = adapt_auto_avsr_weights(ckpt_path, test_model)

    return {
        "size_mb": round(ckpt_path.stat().st_size / (1024*1024), 2),
        "matched_weights": matched,
        "total_weights": total,
    }


def adapt_auto_avsr_weights(ckpt_path: pathlib.Path, model: torch.nn.Module) -> Tuple[int, int]:
    """
    Auto-AVSR (vsr_trlrs3_base.pth) kontrol noktasından 3D-ResNet görsel ön katman
    (Visual Frontend) ağırlıklarını VSRConformerModel'e aktarır.
    """
    import torch
    raw_data = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(raw_data, dict):
        sd = raw_data.get("state_dict", raw_data.get("model_state_dict", raw_data))
    else:
        sd = raw_data

    model_sd = model.state_dict()
    adapted_sd = {}
    matched = 0

    for k, v in sd.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[6:]

        if "frontend.trunk." in new_k:
            new_k = new_k.replace("frontend.trunk.", "frontend.")
        elif "video_frontend.trunk." in new_k:
            new_k = new_k.replace("video_frontend.trunk.", "frontend.")
        elif "video_frontend." in new_k:
            new_k = new_k.replace("video_frontend.", "frontend.")

        if ".downsample." in new_k:
            new_k = new_k.replace(".downsample.", ".shortcut.")

        if new_k in model_sd:
            if model_sd[new_k].shape == v.shape:
                adapted_sd[new_k] = v
                matched += 1
            else:
                print(f"Boyut uyumsuzluğu [{new_k}]: model {model_sd[new_k].shape} vs ckpt {v.shape}")

    missing, unexpected = model.load_state_dict(adapted_sd, strict=False)
    frontend_keys = [k for k in model_sd if 'frontend' in k]
    print(f"🎯 Pretrained Frontend Transferi: {matched}/{len(frontend_keys)} frontend tensörü eşleşti ve başarıyla yüklendi!")
    return matched, len(model_sd)


@app.function(
    image=image,
    volumes={"/root/checkpoints": vol},
    timeout=300,
)
def download_checkpoint_file(remote_name: str) -> bytes:
    """Modal Volume'dan belirtilen kontrol noktası dosyasını bayt olarak indirir."""
    ckpt_path = pathlib.Path("/root/checkpoints") / remote_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt_path}")
    print(f"📥 {remote_name} indiriliyor ({ckpt_path.stat().st_size / (1024*1024):.1f} MB)...")
    return ckpt_path.read_bytes()


@app.function(
    image=image,
    gpu="A10G",
    secrets=[s3_secret],
    volumes={"/root/checkpoints": vol},
    timeout=7200,
)
def train_remote(
    run_name: str = "pilot_run_a10g",
    n_pilot_samples: int = 100,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    encoder_type: str = "conformer",
    d_model: int = 256,
    num_layers: int = 4,
    resume_from_ckpt: bool = False,
    val_ratio: float = 0.2,
    max_per_video: Optional[int] = 8,
    n_test_samples: int = 40,
    blank_penalty: float = 0.4,
    warmup_epochs: int = 3,
    data_mode: str = "word",
    max_per_word: Optional[int] = 20,
    init_from: Optional[str] = None,
    lm_alpha: float = 0.4,
    lm_beta: float = 1.5,
    dataset: str = "iborotti",
    code_revision: str = "unknown",
) -> Dict[str, Any]:
    """
    Modal A10G GPU üzerinde çalışan otonom VSR eğitim fonksiyonu.
    data_mode: 'word' (Top-500 kelime dilimleri), 'phrase' (2-4 kelimelik ifadeler) veya 'sentence' (tam cümleler)
    max_per_word: Kelime başına azami örnek sayısı (sınıf dengelemesi için)
    init_from: Başlanacak checkpoint dosya adı (ör. 'bigru_balanced_word_exp6_best.pt')
    dataset: 'iborotti' (Hugging Face) veya 'master' (S3)
    """
    import boto3
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split

    from src.data.dataset import LipReadingDataset, LipReadingWordDataset, LipReadingPhraseDataset, pad_collate_fn
    from src.data.s3_downloader import S3DatasetDownloader
    from src.data.hf_downloader import HFDatasetDownloader
    from src.evaluation.metrics import evaluate_predictions
    from src.experiments.guardrails import (
        build_checkpoint_provenance,
        collect_sample_ids,
        require_checkpoint_exists,
        require_requested_sample_count,
    )
    from src.models.vsr_conformer import VSRConformerModel
    from src.spotter.keyword_spotter import KeywordSpotter
    from src.spotter.lexicon_decoder import LexiconBeamSearchDecoder
    from src.vocab.top500_words import TOP_500_SET
    from src.vocab.turkish_vocab import VOCAB_SIZE, ctc_greedy_decode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🚀 Modal Container Başlatıldı! Cihaz: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) | Dataset: {dataset.upper()} | Mod: {data_mode.upper()} | MaxPerWord: {max_per_word} | InitFrom: {init_from}")

    if dataset == "iborotti":
        local_data_dir = pathlib.Path("/root/data/iborotti")
        hf_downloader = HFDatasetDownloader(target_dir=local_data_dir)

        # 1. HF Veri İndirme (Eğitim spliti)
        print(f"📥 Hugging Face'ten eğitim veri kümesi hazırlanıyor (Hedef: {n_pilot_samples} segment, split=train, max_per_video={max_per_video})...")
        downloaded_dirs = hf_downloader.download_pilot(
            n_samples=n_pilot_samples,
            split="train",
            max_per_video=max_per_video,
        )

        if len(downloaded_dirs) == 0:
            raise RuntimeError("Hiçbir segment indirilemedi. Hugging Face bağlantısını kontrol edin.")
        require_requested_sample_count(
            split="train", requested=n_pilot_samples, actual=len(downloaded_dirs)
        )

        # 1b. Doğrulama (Validation) Split Verisi İndirme (Speaker-Disjoint)
        n_val_download = max(30, int(n_pilot_samples * val_ratio))
        print(f"📥 Hugging Face'ten Val split segmentleri hazırlanıyor ({n_val_download} segment, split=val)...")
        val_downloaded_dirs = hf_downloader.download_pilot(
            n_samples=n_val_download,
            split="val",
            max_per_video=None,
        )
        require_requested_sample_count(
            split="val", requested=n_val_download, actual=len(val_downloaded_dirs)
        )

        # 1c. Test Split Verisi İndirme (Speaker-Disjoint Test Kümesi)
        if n_test_samples > 0:
            print(f"📥 Hugging Face'ten Test split segmentleri hazırlanıyor ({n_test_samples} segment, split=test)...")
            test_downloaded_dirs = hf_downloader.download_pilot(
                n_samples=n_test_samples,
                split="test",
                max_per_video=None,
            )
            require_requested_sample_count(
                split="test", requested=n_test_samples, actual=len(test_downloaded_dirs)
            )
    else:
        # AWS Kimlik Bilgilerini Doğrula
        bucket = os.environ.get("S3_BUCKET_NAME", "lipreading-data-emre2026")
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        access_key = os.environ.get("AWS_ACCESS_KEY_ID") or None
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or None

        local_data_dir = pathlib.Path("/root/data/master")
        local_data_dir.mkdir(parents=True, exist_ok=True)

        downloader = S3DatasetDownloader(
            bucket=bucket,
            region=region,
            access_key=access_key,
            secret_key=secret_key,
            master_dir=local_data_dir,
        )

        # 1. Veri İndirme (Pilot / Belirtilen Miktar - Konuşmacı Çeşitliliği ile)
        print(f"📥 S3'ten eğitim veri kümesi hazırlanıyor (Hedef: {n_pilot_samples} segment, split=train, max_per_video={max_per_video})...")
        downloaded_dirs = downloader.download_pilot_dataset(
            n_segments=n_pilot_samples,
            split="train",
            workers=32,
            max_per_video=max_per_video,
        )

        if len(downloaded_dirs) == 0:
            raise RuntimeError("Hiçbir segment indirilemedi. S3 bağlantısını kontrol edin.")
        require_requested_sample_count(
            split="train", requested=n_pilot_samples, actual=len(downloaded_dirs)
        )

        # 1b. Doğrulama (Validation) Split Verisi İndirme (Speaker-Disjoint)
        n_val_download = max(30, int(n_pilot_samples * val_ratio))
        print(f"📥 S3'ten Val split segmentleri hazırlanıyor ({n_val_download} segment, split=val)...")
        val_downloaded_dirs = downloader.download_pilot_dataset(
            n_segments=n_val_download,
            split="val",
            workers=16,
            max_per_video=None,
        )
        require_requested_sample_count(
            split="val", requested=n_val_download, actual=len(val_downloaded_dirs)
        )

        # 1c. Test Split Verisi İndirme (Speaker-Disjoint Test Kümesi)
        if n_test_samples > 0:
            print(f"📥 S3'ten Test split segmentleri hazırlanıyor ({n_test_samples} segment, split=test)...")
            test_downloaded_dirs = downloader.download_pilot_dataset(
                n_segments=n_test_samples,
                split="test",
                workers=16,
                max_per_video=None,
            )
            require_requested_sample_count(
                split="test", requested=n_test_samples, actual=len(test_downloaded_dirs)
            )

    # 2. Dataset ve DataLoader Hazırlığı (RAM Önbellekleme ile, Ayrık Konuşmacı Splitleri)
    if data_mode == "word":
        print(f"🎯 [DATA MODE: WORD] align.json'dan Top-500 kelime dilimleri çıkarılıyor (max_per_word={max_per_word})...")
        train_dataset = LipReadingWordDataset(master_dir=local_data_dir, split="train", is_train=True, crop_size=88, cache_in_ram=True, max_per_word=max_per_word)
        val_dataset = LipReadingWordDataset(master_dir=local_data_dir, split="val", is_train=False, crop_size=88, cache_in_ram=True, max_per_word=max_per_word)
    elif data_mode == "phrase":
        print("🗣️ [DATA MODE: PHRASE] align.json'dan 2-4 kelimelik ifade öbekleri çıkarılıyor...")
        train_dataset = LipReadingPhraseDataset(master_dir=local_data_dir, split="train", is_train=True, crop_size=88, cache_in_ram=True, min_words=2, max_words=4)
        val_dataset = LipReadingPhraseDataset(master_dir=local_data_dir, split="val", is_train=False, crop_size=88, cache_in_ram=True, min_words=2, max_words=4)
    else:
        print("📄 [DATA MODE: SENTENCE] Tam cümle segmentleri yükleniyor (split=train)...")
        train_dataset = LipReadingDataset(master_dir=local_data_dir, split="train", is_train=True, crop_size=88, cache_in_ram=True)
        val_dataset = LipReadingDataset(master_dir=local_data_dir, split="val", is_train=False, crop_size=88, cache_in_ram=True)

    # Eğer val split boş kalmışsa (küçük yerel/pilot çalıştırma koruması), train_dataset'ten güvenli alt küme al
    if len(val_dataset) == 0 and len(train_dataset) > 5:
        n_val = max(1, int(len(train_dataset) * val_ratio))
        n_train = len(train_dataset) - n_val
        torch.manual_seed(42)
        train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_total = n_train + n_val
    if n_total == 0:
        raise RuntimeError(f"Veri kümesinde hiç örnek bulunamadı (Mod: {data_mode}).")

    split_map_candidates = [
        pathlib.Path(__file__).resolve().parent.parent / "data" / "split_map_iborotti.json" if dataset == "iborotti" else pathlib.Path("/root/data/metadata/split_map.json"),
        pathlib.Path("/root/src/data/split_map_iborotti.json") if dataset == "iborotti" else pathlib.Path("/root/data/metadata/split_map.json"),
        pathlib.Path("/root/data/metadata") / ("split_map_iborotti.json" if dataset == "iborotti" else "split_map.json"),
    ]
    split_map_path = next((p for p in split_map_candidates if p.exists()), split_map_candidates[0])
    if not split_map_path.exists() and dataset == "iborotti":
        from src.data.split_map_data import IBOROTTI_SPLIT_MAP
        tmp_map = pathlib.Path("/tmp/split_map_iborotti.json")
        tmp_map.write_text(json.dumps(IBOROTTI_SPLIT_MAP, indent=2), encoding="utf-8")
        split_map_path = tmp_map
    if init_from:
        initializer = f"project-checkpoint:{init_from}"
    elif resume_from_ckpt:
        initializer = "auto-avsr:vsr_trlrs3_base.pth"
    else:
        initializer = "random"
    checkpoint_provenance = build_checkpoint_provenance(
        dataset_id="iboRotti/avsr-tr-dataset" if dataset == "iborotti" else "legacy-s3-master",
        split_map_path=split_map_path,
        train_sample_ids=collect_sample_ids(train_dataset),
        val_sample_ids=collect_sample_ids(val_dataset),
        seed=seed,
        initializer=initializer,
        code_revision=code_revision,
    )

    print(f"📊 Konuşmacı Ayrık Veri Dağılımı ({data_mode.upper()}): Toplam={n_total}, Eğitim={n_train} (split=train), Doğrulama={n_val} (split=val)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=2,
    )

    # 3. Model Kurulumu
    loaded_st = None
    loaded_checkpoint = False
    if init_from:
        ckpt_path = pathlib.Path("/root/checkpoints") / init_from
        require_checkpoint_exists(ckpt_path)
        print(f"🔄 Önceki kontrol noktasından devam ediliyor: {ckpt_path}...")
        raw_st = torch.load(str(ckpt_path), map_location="cpu")
        loaded_st = raw_st.get("model_state_dict", raw_st)
        # Checkpoint'ten d_model otomatik tespiti
        if "ctc_head.weight" in loaded_st:
            detected_d_model = loaded_st["ctc_head.weight"].shape[1]
            if detected_d_model != d_model:
                print(f"📐 d_model otomatik olarak checkpoint ile hizalandı: {d_model} -> {detected_d_model}")
                d_model = detected_d_model
        # BiGRU katman sayısı otomatik tespiti
        if encoder_type == "bigru":
            layer_indices = [
                int(k.split("_l")[1].split("_")[0])
                for k in loaded_st.keys()
                if "temporal_encoder.weight_ih_l" in k
            ]
            if layer_indices:
                detected_layers = max(layer_indices) + 1
                if detected_layers != num_layers:
                    print(f"📐 num_layers otomatik olarak checkpoint ile hizalandı: {num_layers} -> {detected_layers}")
                    num_layers = detected_layers

    model = VSRConformerModel(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        num_layers=num_layers,
        encoder_type=encoder_type,
    ).to(device)

    # CTC boşluk çökmesini (blank collapse) kırmak için blank bias'ı hafif negatif başlat
    with torch.no_grad():
        if hasattr(model, "ctc_head") and model.ctc_head.bias is not None:
            model.ctc_head.bias[0] -= 0.5

    if loaded_st is not None:
        model.load_state_dict(loaded_st)
        print(f"🎯 Tam Model Kontrol Noktası Yüklendi: {init_from}")
        loaded_checkpoint = True

    if not loaded_checkpoint and resume_from_ckpt:
        ckpt_path = pathlib.Path("/root/checkpoints/vsr_trlrs3_base.pth")
        if not ckpt_path.exists():
            print("Temel model kontrol noktası S3'ten indiriliyor...")
            try:
                checkpoint_downloader = S3DatasetDownloader(
                    bucket=os.environ.get("S3_BUCKET_NAME", "lipreading-data-emre2026"),
                    region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                    access_key=os.environ.get("AWS_ACCESS_KEY_ID") or None,
                    secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
                    master_dir=pathlib.Path("/root/data/master"),
                )
                checkpoint_downloader.download_checkpoint(dest_file=ckpt_path)
                vol.commit()
            except Exception as e:
                raise RuntimeError(
                    f"Auto-AVSR initializer indirilemedi; rastgele başlangıca düşülmedi: {e}"
                ) from e

        require_checkpoint_exists(ckpt_path)
        matched, total = adapt_auto_avsr_weights(ckpt_path, model)
        if matched == 0:
            raise RuntimeError("Auto-AVSR checkpoint bulundu ancak hiçbir frontend ağırlığı eşleşmedi.")
        print(f"Pretrained visual frontend aktarıldı: {matched} tensör yüklendi.")
        loaded_checkpoint = True

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # İki Aşamalı İnce Ayar (Two-Phase Fine-Tuning):
    # Eğer sıfırdan Auto-AVSR aktarıldıysa ilk warmup_epochs boyunca dondur.
    # Eğer zaten eğitilmiş modelden (init_from) devam ediliyorsa doğrudan diferansiyel LR ile başla.
    effective_warmup = warmup_epochs if (resume_from_ckpt and not init_from and 0 < warmup_epochs < epochs) else 0
    if loaded_checkpoint or resume_from_ckpt:
        param_groups = [
            {"params": model.frontend.parameters(), "lr": lr * 0.05},
            {"params": model.temporal_encoder.parameters(), "lr": lr},
            {"params": model.ctc_head.parameters(), "lr": lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if effective_warmup > 0:
            model.freeze_frontend(True)
            print(f"🔒 [Phase 1: Epoch 1-{effective_warmup}] Pretrained Frontend donduruldu (requires_grad=False). Sadece Temporal Encoder/Head eğitiliyor (LR={lr:.2e})")
        else:
            print(f"⚙️ Diferansiyel Öğrenme Hızı: Frontend={lr*0.05:.2e}, Encoder/Head={lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
    spotter = KeywordSpotter(min_confidence=0.18, blank_penalty=blank_penalty)

    history = []
    best_val_loss = float("inf")
    best_spotter_f1 = 0.0
    final_val_predictions = []

    print("\n🔥 Eğitim Başlıyor...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start_time = time.time()

        # Phase 2 Geçişi: Warmup bittiğinde Pretrained Frontend'i çöz (requires_grad=True)
        # Optimizer ve Scheduler sıfırlanmaz, momentum ve cosine planı kesintisiz devam eder!
        if effective_warmup > 0 and epoch == effective_warmup + 1:
            model.freeze_frontend(False)
            print(f"\n🔓 [Phase 2: Epoch {epoch}-{epochs}] Pretrained Frontend çözüldü. Diferansiyel LR devrede (Frontend={lr*0.05:.2e}, Encoder/Head={lr:.2e})\n")

        model.train()
        train_loss_total = 0.0
        train_steps = 0

        for batch in train_loader:
            videos = batch["videos"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
                logits = model(videos)
                loss = model.compute_loss(logits, targets, input_lengths, target_lengths, blank_penalty=blank_penalty)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Uyarı: NaN/Inf kayıp tespit edildi, adım atlanıyor.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train_loss = train_loss_total / max(1, train_steps)

        # Doğrulama (Validation)
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        val_blank_frames = 0
        val_total_frames = 0
        all_refs = []
        all_hyps = []
        all_spotted = []

        with torch.no_grad():
            for batch in val_loader:
                videos = batch["videos"].to(device)
                targets = batch["targets"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)

                with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
                    logits = model(videos)
                    loss = model.compute_loss(logits, targets, input_lengths, target_lengths, blank_penalty=blank_penalty)

                val_loss_total += loss.item()
                val_steps += 1

                # Blank emisyon oranı (raw logits argmax)
                argmax_tokens = logits.argmax(dim=-1)
                val_blank_frames += (argmax_tokens == 0).sum().item()
                val_total_frames += argmax_tokens.numel()

                # Greedy Decoding (Blank Penalty ile) & Metrikler
                decoded_preds = ctc_greedy_decode(logits, blank_penalty=blank_penalty)
                all_hyps.extend(decoded_preds)
                all_refs.extend(batch["transcripts"])

                # 500 Kelime Avcısı (Keyword Spotter) ile tespit
                batch_spotted = spotter.spot_batch(logits.detach().cpu(), blank_penalty=blank_penalty)
                all_spotted.extend(batch_spotted)

        avg_val_loss = val_loss_total / max(1, val_steps)
        blank_ratio = val_blank_frames / max(1, val_total_frames)
        metrics = evaluate_predictions(all_refs, all_hyps, spotted_keywords=all_spotted)
        ep_elapsed = round(time.time() - ep_start_time, 2)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "blank_ratio": round(blank_ratio, 4),
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "spotter_precision": metrics["spotter_precision"],
            "spotter_recall": metrics["spotter_recall"],
            "spotter_f1": metrics["spotter_f1"],
            "epoch_sec": ep_elapsed,
        }
        history.append(epoch_stats)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_elapsed}s) "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Blank Oranı: %{blank_ratio*100:.1f} | "
            f"CER: {metrics['cer']:.4f} | "
            f"WER: {metrics['wer']:.4f} | "
            f"Spotter F1: {metrics['spotter_f1']:.4f} "
            f"(P: {metrics['spotter_precision']:.3f}, R: {metrics['spotter_recall']:.3f}, TP: {metrics['true_positives']}, FP: {metrics['false_positives']})"
        )

        if epoch % 5 == 0 or epoch == epochs or metrics['spotter_f1'] > 0:
            print("  --- Örnek Çıktılar ---")
            for idx_s in range(min(5, len(all_refs))):
                print(f"    [Ref]: {all_refs[idx_s][:60]}")
                print(f"    [Hyp]: '{all_hyps[idx_s][:60]}'")
                if all_spotted and idx_s < len(all_spotted) and all_spotted[idx_s]:
                    sp_w = [getattr(w, 'word', str(w)) for w in all_spotted[idx_s]]
                    print(f"    [Spot]: {sp_w}")

        if epoch == epochs:
            for idx_s in range(min(25, len(all_refs))):
                sp_w = [getattr(w, 'word', str(w)) for w in all_spotted[idx_s]] if (all_spotted and idx_s < len(all_spotted)) else []
                final_val_predictions.append({
                    "ref": all_refs[idx_s],
                    "hyp": all_hyps[idx_s],
                    "spotted": sp_w,
                })

        # Checkpoint bilgileri
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "metrics": metrics,
            "encoder_type": encoder_type,
            "d_model": d_model,
            "num_layers": num_layers,
            "vocab_size": VOCAB_SIZE,
            "provenance": checkpoint_provenance,
        }

        # En iyi model kontrolü
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_spotter_f1 = metrics["spotter_f1"]
            best_ckpt = pathlib.Path("/root/checkpoints") / f"{run_name}_best.pt"
            torch.save(ckpt_payload, str(best_ckpt))
            vol.commit()

    # Son durum checkpoint'ini kaydet
    last_ckpt = pathlib.Path("/root/checkpoints") / f"{run_name}_last.pt"
    torch.save(ckpt_payload, str(last_ckpt))
    vol.commit()

    # 4. Test Split Değerlendirmesi
    test_metrics = {}
    if n_test_samples > 0:
        test_dataset = LipReadingDataset(master_dir=local_data_dir, split="test", is_train=False, crop_size=88, cache_in_ram=True)
        if len(test_dataset) > 0:
            print(f"\n🔍 [TEST SPLIT DEĞERLENDİRMESİ] ({len(test_dataset)} test örneği)...")
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=pad_collate_fn,
                num_workers=2,
            )

            # En iyi checkpoint'i yükle
            best_ckpt = pathlib.Path("/root/checkpoints") / f"{run_name}_best.pt"
            if best_ckpt.exists():
                saved_st = torch.load(str(best_ckpt), map_location=device)
                model.load_state_dict(saved_st["model_state_dict"])
            model.eval()

            test_refs = []
            test_hyps = []
            test_spotted = []
            test_sliding_spotted = []
            test_beam_hyps = []
            test_beam_spotted = []
            lex_decoder = LexiconBeamSearchDecoder(
                target_vocab=TOP_500_SET,
                beam_size=50,
                fps=25.0,
                viseme_tolerance=True,
                lm_alpha=lm_alpha,
                lm_beta=lm_beta,
                repeat_penalty=3.0,
                min_dur_factor=1.0,
                use_lm=True,
            )
            with torch.no_grad():
                for batch in test_loader:
                    videos = batch["videos"].to(device)
                    with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
                        logits = model(videos)
                    preds = ctc_greedy_decode(logits, blank_penalty=blank_penalty)
                    test_hyps.extend(preds)
                    test_refs.extend(batch["transcripts"])

                    # 1. Standart Greedy Decoding Tabanlı Tespit
                    batch_spotted = spotter.spot_batch(logits.detach().cpu(), blank_penalty=blank_penalty)
                    test_spotted.extend(batch_spotted)

                    # 2. Kayan Pencere CTC Sözlük Hizalama Tabanlı Tespit (Sliding Window CTC)
                    for i_s in range(videos.size(0)):
                        t_len = batch["input_lengths"][i_s].item()
                        sample_vid = videos[i_s : i_s + 1, :, :t_len, :, :]
                        sw_dets = spotter.spot_sliding_window_ctc(
                            model, sample_vid, window_sizes=(10, 14, 18), stride=3, min_score=-3.0, prior_weight=0.5
                        )
                        test_sliding_spotted.append(sw_dets)

                    # 3. 500 Kelimelik Sözlük & Visem Kısıtlı CTC Prefix Beam Search
                    for i_s in range(videos.size(0)):
                        t_len = batch["input_lengths"][i_s].item()
                        sample_logits = logits[i_s, :t_len, :]
                        b_text, b_kws = lex_decoder.decode(sample_logits, blank_penalty=blank_penalty)
                        test_beam_hyps.append(b_text)
                        test_beam_spotted.append(b_kws)

            test_metrics_greedy = evaluate_predictions(test_refs, test_hyps, spotted_keywords=test_spotted)
            test_metrics_sliding = evaluate_predictions(test_refs, test_hyps, spotted_keywords=test_sliding_spotted)
            test_metrics_beam = evaluate_predictions(test_refs, test_beam_hyps, spotted_keywords=test_beam_spotted)

            print(
                f"  📊 Test [Greedy]:  CER: {test_metrics_greedy['cer']:.4f} | WER: {test_metrics_greedy['wer']:.4f} | "
                f"Spotter F1: {test_metrics_greedy['spotter_f1']:.4f} (TP: {test_metrics_greedy['true_positives']}, FP: {test_metrics_greedy['false_positives']}, P: {test_metrics_greedy['spotter_precision']:.3f}, R: {test_metrics_greedy['spotter_recall']:.3f})"
            )
            print(
                f"  🎯 Test [Sliding Window CTC]: Spotter F1: {test_metrics_sliding['spotter_f1']:.4f} | "
                f"(TP: {test_metrics_sliding['true_positives']}, FP: {test_metrics_sliding['false_positives']}, P: {test_metrics_sliding['spotter_precision']:.3f}, R: {test_metrics_sliding['spotter_recall']:.3f})"
            )
            print(
                f"  🌟 Test [Lexicon Beam Search]: Spotter F1: {test_metrics_beam['spotter_f1']:.4f} | "
                f"(TP: {test_metrics_beam['true_positives']}, FP: {test_metrics_beam['false_positives']}, P: {test_metrics_beam['spotter_precision']:.3f}, R: {test_metrics_beam['spotter_recall']:.3f})"
            )

            # Örnek Tespit Edilen Kelimeler
            if test_sliding_spotted:
                sample_words = []
                for sl_list in test_sliding_spotted[:5]:
                    sample_words.append([getattr(d, 'word', str(d)) for d in sl_list])
                print(f"  🔍 Örnek Kayan Pencere Tespitleri (İlk 5 cümle): {sample_words}")

            if test_beam_spotted:
                beam_samples = []
                for b_list in test_beam_spotted[:5]:
                    beam_samples.append([getattr(d, 'word', str(d)) for d in b_list])
                print(f"  🔍 Örnek Lexicon Beam Tespitleri (İlk 5 cümle): {beam_samples}")

            test_metrics = {
                "greedy": test_metrics_greedy,
                "sliding_window": test_metrics_sliding,
                "lexicon_beam": test_metrics_beam,
                "cer": test_metrics_greedy["cer"],
                "wer": test_metrics_greedy["wer"],
                "spotter_f1": max(test_metrics_greedy["spotter_f1"], test_metrics_sliding["spotter_f1"], test_metrics_beam["spotter_f1"]),
                "true_positives": max(test_metrics_greedy["true_positives"], test_metrics_sliding["true_positives"], test_metrics_beam["true_positives"]),
            }

    total_time = round(time.time() - start_time, 2)
    print(f"\n✅ Eğitim ve Değerlendirme Tamamlandı! Süre: {total_time}s | En İyi Val Loss: {best_val_loss:.4f}")

    return {
        "run_name": run_name,
        "total_time_sec": total_time,
        "epochs": epochs,
        "n_samples": n_total,
        "best_val_loss": round(best_val_loss, 4),
        "best_spotter_f1": round(best_spotter_f1, 4),
        "test_metrics": test_metrics,
        "last_epoch_metrics": history[-1] if history else {},
        "history": history,
        "final_val_predictions": final_val_predictions,
        "provenance": checkpoint_provenance,
    }


@app.local_entrypoint()
def main(
    run_name: str = "pilot_a10g_exp1",
    pilot: int = 200,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    encoder: str = "conformer",
    d_model: int = 256,
    num_layers: int = 4,
    resume: bool = False,
    max_per_video: int = 8,
    test_samples: int = 40,
    blank_penalty: float = 0.4,
    warmup_epochs: int = 3,
    data_mode: str = "word",
    max_per_word: int = 20,
    init_from: str = "",
    lm_alpha: float = 0.4,
    lm_beta: float = 1.5,
    dataset: str = "iborotti",
    code_revision: str = "unknown",
):
    # This file is the retired staged-pilot entrypoint.  Keep it fail-closed so
    # stale Antigravity goals cannot bypass the canonical research workflow by
    # invoking Modal directly.
    from run_experiment import require_mode_allowed

    require_mode_allowed(mode="modal-pilot")

    init_from_val = init_from if init_from else None
    if code_revision == "unknown":
        try:
            import subprocess
            code_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
            ).strip()
        except Exception:
            code_revision = "unknown"
    print(f"🚀 Modal bulut eğitimi başlatılıyor... (Run: {run_name}, Dataset: {dataset}, Pilot: {pilot}, Epochs: {epochs}, Encoder: {encoder}, Mode: {data_mode}, MaxPerWord: {max_per_word}, InitFrom: {init_from_val}, Resume: {resume}, LM_Alpha: {lm_alpha}, LM_Beta: {lm_beta}, Rev: {code_revision[:8]})")
    result = train_remote.remote(
        run_name=run_name,
        n_pilot_samples=pilot,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        encoder_type=encoder,
        d_model=d_model,
        num_layers=num_layers,
        resume_from_ckpt=resume,
        max_per_video=max_per_video,
        n_test_samples=test_samples,
        blank_penalty=blank_penalty,
        warmup_epochs=warmup_epochs,
        data_mode=data_mode,
        max_per_word=max_per_word,
        init_from=init_from_val,
        lm_alpha=lm_alpha,
        lm_beta=lm_beta,
        dataset=dataset,
        code_revision=code_revision,
    )
    print("\n🏁 Bulut Eğitimi Sonuçları:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
