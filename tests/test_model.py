"""
tests/test_model.py — Model İleri Geçiş ve CTC Kayıp Testleri
"""

import pytest
import torch
from src.models.vsr_conformer import VSRConformerModel
from src.vocab.turkish_vocab import VOCAB_SIZE


def test_model_forward_pass_cpu():
    model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=128, num_layers=2, encoder_type="conformer")
    model.eval()

    # Batch=2, Channel=1, Frames=12, Height=88, Width=88
    x = torch.randn(2, 1, 12, 88, 88)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 12, VOCAB_SIZE)
    assert not torch.isnan(logits).any()


def test_model_ctc_loss():
    model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=128, num_layers=2, encoder_type="bigru")
    model.train()

    B, T = 2, 16
    x = torch.randn(B, 1, T, 88, 88)
    logits = model(x)

    # Targets: [merhaba], [dünya]
    targets = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 0]])  # padded targets
    input_lengths = torch.tensor([T, T], dtype=torch.long)
    target_lengths = torch.tensor([4, 3], dtype=torch.long)

    loss = model.compute_loss(logits, targets, input_lengths, target_lengths)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0


def test_tiny_batch_overfit():
    """Research-Craft kuralı: Modeli büyütmeden önce 1 mini-batch üzerinde loss'un düştüğünü kanıtla."""
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=64, num_layers=1, encoder_type="bigru").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    B, T = 1, 10
    x = torch.randn(B, 1, T, 88, 88, device=device)
    targets = torch.tensor([[4, 5, 6]], dtype=torch.long, device=device)
    input_lengths = torch.tensor([T], dtype=torch.long)
    target_lengths = torch.tensor([3], dtype=torch.long)

    initial_loss = None
    final_loss = None

    for step in range(12):
        optimizer.zero_grad()
        logits = model(x)
        loss = model.compute_loss(logits, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss, f"Loss düşmedi: {initial_loss} -> {final_loss}"


def test_conformer_odd_d_model():
    # 8'in katı olmayan d_model değerlerinde Conformer başlık bölünmesi çökmemeli
    model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=100, num_layers=1, encoder_type="conformer")
    x = torch.randn(1, 1, 8, 88, 88)
    logits = model(x)
    assert logits.shape == (1, 8, VOCAB_SIZE)


def test_freeze_frontend():
    model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=64, num_layers=1, encoder_type="bigru")
    assert all(p.requires_grad for p in model.frontend.parameters())

    model.freeze_frontend(True)
    assert all(not p.requires_grad for p in model.frontend.parameters())
    assert all(p.requires_grad for p in model.temporal_encoder.parameters())
    assert all(p.requires_grad for p in model.ctc_head.parameters())

    # model.train() cagrildiginda frontend eval modunda kalmali
    model.train()
    assert model.frontend.training is False
    assert model.temporal_encoder.training is True

    model.freeze_frontend(False)
    assert all(p.requires_grad for p in model.frontend.parameters())
    model.train()
    assert model.frontend.training is True


def test_compute_loss_with_blank_penalty():
    model = VSRConformerModel(vocab_size=VOCAB_SIZE, d_model=64, num_layers=1, encoder_type="bigru")
    B, T = 2, 16
    x = torch.randn(B, 1, T, 88, 88)
    logits = model(x)
    targets = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 0]])
    input_lengths = torch.tensor([T, T], dtype=torch.long)
    target_lengths = torch.tensor([4, 3], dtype=torch.long)

    loss_std = model.compute_loss(logits, targets, input_lengths, target_lengths, blank_penalty=0.0)
    loss_pen = model.compute_loss(logits, targets, input_lengths, target_lengths, blank_penalty=1.5)

    assert not torch.isnan(loss_pen)
    assert not torch.isinf(loss_pen)
    assert loss_pen.item() != loss_std.item()

