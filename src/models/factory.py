"""
src/models/factory.py — VSR Model Fabrikası ve Mimari Yapılandırması

Farklı mimari hipotezlerini (Frontend, Temporal Encoder: BiGRU vs Conformer, Boyutlar)
tek bir noktadan kolayca üretmek, parametrelerini saymak ve yapılandırmak için fabrika modülü.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.vsr_conformer import VSRConformerModel
from src.vocab.turkish_vocab import VOCAB_SIZE


@dataclass
class ModelConfig:
    arch_name: str
    d_model: int
    num_layers: int
    encoder_type: str  # "bigru" veya "conformer"
    vocab_size: int = VOCAB_SIZE
    freeze_frontend: bool = False
    description: str = ""


# Hazır Mimari Profilleri
ARCH_CONFIGS: Dict[str, ModelConfig] = {
    # 1. Hızlı Yerel Kontrol / Overfit Testleri (Çok Hafif)
    "vsr_tiny": ModelConfig(
        arch_name="vsr_tiny",
        d_model=128,
        num_layers=1,
        encoder_type="bigru",
        description="Tek katmanlı ultra hafif BiGRU (Hızlı overfit & CI testleri için).",
    ),
    # 2. Küçük BiGRU (Az veri / Hızlı eğitim)
    "vsr_bigru_small": ModelConfig(
        arch_name="vsr_bigru_small",
        d_model=256,
        num_layers=2,
        encoder_type="bigru",
        description="2 katmanlı kompakt BiGRU (Düşük kaynak / hızlı yakınsama).",
    ),
    # 3. Temel BiGRU (Standart referans mimari)
    "vsr_bigru_base": ModelConfig(
        arch_name="vsr_bigru_base",
        d_model=512,
        num_layers=3,
        encoder_type="bigru",
        description="3 katmanlı 512-boyutlu BiGRU (Dengeli zamansal modelleme).",
    ),
    # 4. Küçük Conformer (Positional Encoding + Multi-head Attention + Depthwise Conv)
    "vsr_conformer_small": ModelConfig(
        arch_name="vsr_conformer_small",
        d_model=256,
        num_layers=2,
        encoder_type="conformer",
        description="2 katmanlı 256-boyutlu Conformer (Pozisyonel kodlamalı hafif dikkat mimarisi).",
    ),
    # 5. Temel Conformer (Auto-AVSR tarzı tam Conformer)
    "vsr_conformer_base": ModelConfig(
        arch_name="vsr_conformer_base",
        d_model=512,
        num_layers=4,
        encoder_type="conformer",
        description="4 katmanlı 512-boyutlu Conformer (Geniş dikkat ve evrişim blokları).",
    ),
}


def build_vsr_model(
    arch_name: str = "vsr_bigru_base",
    d_model: Optional[int] = None,
    num_layers: Optional[int] = None,
    encoder_type: Optional[str] = None,
    vocab_size: int = VOCAB_SIZE,
    freeze_frontend: bool = False,
    dropout: float = 0.1,
    use_specaugment: bool = False,
) -> VSRConformerModel:
    """
    İstenen mimari profili veya özel parametrelerle VSRConformerModel örneği üretir.
    """
    if arch_name in ARCH_CONFIGS:
        base_cfg = ARCH_CONFIGS[arch_name]
        act_d_model = d_model or base_cfg.d_model
        act_num_layers = num_layers or base_cfg.num_layers
        act_encoder = encoder_type or base_cfg.encoder_type
    else:
        act_d_model = d_model or 512
        act_num_layers = num_layers or 3
        act_encoder = encoder_type or "bigru"

    model = VSRConformerModel(
        vocab_size=vocab_size,
        d_model=act_d_model,
        num_layers=act_num_layers,
        encoder_type=act_encoder,
        dropout=dropout,
        use_specaugment=use_specaugment,
    )

    if freeze_frontend:
        model.freeze_frontend(True)

    return model


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Modelin bileşen bazında parametre sayılarını çıkarır."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    frontend_params = 0
    temporal_params = 0
    head_params = 0

    if hasattr(model, "frontend"):
        frontend_params = sum(p.numel() for p in model.frontend.parameters())
    if hasattr(model, "temporal_encoder"):
        temporal_params = sum(p.numel() for p in model.temporal_encoder.parameters())
    if hasattr(model, "ctc_head"):
        head_params = sum(p.numel() for p in model.ctc_head.parameters())

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frontend": frontend_params,
        "temporal": temporal_params,
        "head": head_params,
        "total_m": round(total_params / 1e6, 2),
        "trainable_m": round(trainable_params / 1e6, 2),
    }


def adapt_auto_avsr_weights(ckpt_path: Any, model: nn.Module) -> Tuple[int, int]:
    """
    Auto-AVSR (vsr_trlrs3_base.pth) kontrol noktasından 3D-ResNet görsel ön katman
    (Visual Frontend) ağırlıklarını VSRConformerModel'e aktarır.
    """
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

        if "video_frontend.trunk." in new_k:
            new_k = new_k.replace("video_frontend.trunk.", "frontend.")
        elif "video_frontend." in new_k:
            new_k = new_k.replace("video_frontend.", "frontend.")
        elif "frontend.trunk." in new_k:
            new_k = new_k.replace("frontend.trunk.", "frontend.")

        if ".downsample." in new_k:
            new_k = new_k.replace(".downsample.", ".shortcut.")

        if new_k in model_sd:
            if model_sd[new_k].shape == v.shape:
                adapted_sd[new_k] = v
                matched += 1

    model.load_state_dict(adapted_sd, strict=False)
    frontend_keys = [k for k in model_sd if "frontend" in k]
    return matched, len(frontend_keys)
