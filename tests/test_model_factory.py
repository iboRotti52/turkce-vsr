"""
tests/test_model_factory.py — Model Fabrikası ve Mimari Yapılandırma Birim Testleri
"""

import pytest
import torch

from src.models.factory import ARCH_CONFIGS, build_vsr_model, count_parameters
from src.models.vsr_conformer import PositionalEncoding


def test_arch_configs_registered():
    assert "vsr_tiny" in ARCH_CONFIGS
    assert "vsr_bigru_base" in ARCH_CONFIGS
    assert "vsr_conformer_base" in ARCH_CONFIGS


def test_build_vsr_tiny():
    model = build_vsr_model("vsr_tiny")
    assert model.encoder_type == "bigru"
    assert model.d_model == 128
    info = count_parameters(model)
    assert info["total"] > 0
    assert info["trainable"] == info["total"]


def test_build_vsr_conformer_has_positional_encoding():
    model = build_vsr_model("vsr_conformer_small")
    assert model.encoder_type == "conformer"
    assert hasattr(model, "pos_encoder")
    assert isinstance(model.pos_encoder, PositionalEncoding)


def test_positional_encoding_forward():
    pe = PositionalEncoding(d_model=64, max_len=100)
    x = torch.zeros(2, 20, 64)
    out = pe(x)
    assert out.shape == (2, 20, 64)
    # Pozisyonel kodlama eklendiği için sıfır olmamalı
    assert not torch.allclose(out, torch.zeros_like(out))


def test_freeze_frontend():
    model = build_vsr_model("vsr_tiny", freeze_frontend=True)
    info = count_parameters(model)
    assert info["trainable"] < info["total"]
    for param in model.frontend.parameters():
        assert not param.requires_grad


def test_adapt_auto_avsr_weights_mock(tmp_path):
    from src.models.factory import adapt_auto_avsr_weights
    model = build_vsr_model("vsr_tiny")
    
    # Create mock state dict matching some frontend weights
    sd = model.state_dict()
    mock_sd = {
        "video_frontend.trunk.layer1.0.conv1.weight": sd["frontend.layer1.0.conv1.weight"].clone(),
        "video_frontend.trunk.layer1.0.bn1.weight": sd["frontend.layer1.0.bn1.weight"].clone(),
    }
    ckpt_file = tmp_path / "mock_autoavsr.pth"
    torch.save(mock_sd, ckpt_file)

    matched, total_frontend = adapt_auto_avsr_weights(ckpt_file, model)
    assert matched == 2
    assert total_frontend > 0


def test_compact_conformer_frontend_shapes_match_auto_avsr():
    """Verify that compact conformer (d_model=256) maintains 512-dim Auto-AVSR frontend."""
    model_small = build_vsr_model("vsr_conformer_small")
    model_base = build_vsr_model("vsr_conformer_base")

    fe_small_keys = {k: v.shape for k, v in model_small.frontend.state_dict().items()}
    fe_base_keys = {k: v.shape for k, v in model_base.frontend.state_dict().items()}

    # All 121 frontend shapes must match exactly between small and base
    assert len(fe_small_keys) == 121
    assert len(fe_base_keys) == 121
    for k, shape in fe_base_keys.items():
        assert fe_small_keys[k] == shape, f"Shape mismatch in frontend tensor {k}: {fe_small_keys[k]} vs {shape}"

    # Verify linear projection from 512 to 256
    assert hasattr(model_small, "fe_proj")
    assert isinstance(model_small.fe_proj, torch.nn.Linear)
    assert model_small.fe_proj.in_features == 512
    assert model_small.fe_proj.out_features == 256

    # Forward pass
    x = torch.randn(2, 1, 8, 88, 88)
    logits = model_small(x)
    assert logits.shape == (2, 8, model_small.vocab_size)


def test_compact_conformer_freeze_frontend_gradient_flow():
    """Verify that freeze_frontend=True freezes 3D-ResNet while keeping fe_proj trainable with gradient flow."""
    model = build_vsr_model("vsr_conformer_small", freeze_frontend=True)
    
    # Check requires_grad flags
    for param in model.frontend.parameters():
        assert not param.requires_grad
    assert model.fe_proj.weight.requires_grad is True
    assert model.fe_proj.bias.requires_grad is True

    # Run backward pass through compute_loss
    x = torch.randn(2, 1, 8, 88, 88)
    logits = model(x)
    targets = torch.randint(1, 30, (2, 4))
    in_lens = torch.tensor([8, 8])
    tgt_lens = torch.tensor([4, 4])
    loss = model.compute_loss(logits, targets, in_lens, tgt_lens)
    loss.backward()

    # fe_proj must receive gradients
    assert model.fe_proj.weight.grad is not None
    assert model.fe_proj.weight.grad.norm().item() > 0
    assert model.fe_proj.bias.grad is not None

    # frontend must receive NO gradients
    for param in model.frontend.parameters():
        assert param.grad is None


def test_compact_conformer_forward_from_features_both_dims():
    """Verify forward_from_features accepts both unprojected (512-dim) and projected (d_model-dim) features."""
    model = build_vsr_model("vsr_conformer_small")
    
    # 512-dim unprojected features
    feats_512 = torch.randn(2, 10, 512)
    logits_from_512 = model.forward_from_features(feats_512)
    assert logits_from_512.shape == (2, 10, model.vocab_size)

    # 256-dim projected features
    feats_256 = torch.randn(2, 10, 256)
    logits_from_256 = model.forward_from_features(feats_256)
    assert logits_from_256.shape == (2, 10, model.vocab_size)


def test_adapt_auto_avsr_weights_all_registered_archs(tmp_path):
    """Verify Auto-AVSR weight adaptation succeeds across all registered architectures."""
    from src.models.factory import adapt_auto_avsr_weights

    # Generate mock full Auto-AVSR frontend weights
    base = build_vsr_model("vsr_conformer_base")
    mock_sd = {}
    for k, v in base.state_dict().items():
        if "frontend" in k:
            mock_sd["video_frontend.trunk." + k[len("frontend."):]] = v.clone()

    mock_ckpt = tmp_path / "mock_full_autoavsr.pth"
    torch.save({"state_dict": mock_sd}, mock_ckpt)

    for arch in ARCH_CONFIGS:
        model = build_vsr_model(arch)
        matched, total = adapt_auto_avsr_weights(mock_ckpt, model)
        assert matched == 121, f"Failed for {arch}: matched {matched}/121"


