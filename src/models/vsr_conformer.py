"""
src/models/vsr_conformer.py — 3D-ResNet + Conformer/BiGRU CTC Dudak Okuma Modeli
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vocab.turkish_vocab import BLANK_IDX, VOCAB_SIZE


class Conv3dResNetBlock(nn.Module):
    """2D Residual Blok (zamansal düzleştirme ile çalışan 2D ResNet bloğu)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class VisualFrontend(nn.Module):
    """
    Spatio-temporal 3D-CNN + 2D ResNet-18 Görsel Ön Katmanı.
    Girdi:  (B, 1, T, H, W)
    Çıktı: (B, T, 512)
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()
        # 3D Spatio-temporal ilk evrişim katmanı
        self.frontend3d = nn.Sequential(
            nn.Conv3d(
                1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False
            ),
            nn.BatchNorm3d(64),
            nn.PReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # 2D ResNet-18 benzeri residual katmanlar
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, out_dim, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(
        self, in_planes: int, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(Conv3dResNetBlock(in_planes, planes, stride=s))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.size()
        # 3D CNN: (B, 64, T, H/4, W/4)
        x = self.frontend3d(x)

        # (B, 64, T, H', W') -> (B, T, 64, H', W') -> (B*T, 64, H', W')
        x = x.transpose(1, 2).contiguous()
        x = x.view(B * T, 64, x.size(3), x.size(4))

        # ResNet Blokları
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # (B*T, out_dim, 1, 1)

        # (B, T, out_dim)
        x = x.view(B, T, -1)
        return x

class PositionalEncoding(nn.Module):
    """Sinusoidal zamansal pozisyon kodlaması (Transformer/Conformer için zaman dizisi bilgisi)."""

    def __init__(self, d_model: int, max_len: int = 1500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """
    Hafif ve kararlı Conformer bloğu (FeedForward + MultiHeadAttention + DepthwiseConv + FeedForward).
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, conv_kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.conv_norm = nn.LayerNorm(d_model)
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Macaron-style FFN 1
        x = x + 0.5 * self.ff1(x)
        
        # Self-Attention
        norm_x = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_out

        # Convolution modülü: (B, T, D) -> (B, D, T)
        norm_x = self.conv_norm(x).transpose(1, 2)
        conv_out = self.depthwise_conv(norm_x).transpose(1, 2)
        x = x + conv_out

        # Macaron-style FFN 2
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class FeatureSpecAugment(nn.Module):
    """
    Öznitelik Seviyesinde SpecAugment (Zaman ve Kanal Maskelemesi).
    3D-ResNet ön katmanının (B, T, D) tensörü üzerinde çalışır.
    Piksel bazlı fotometrik pertürbasyon gibi dudak kenar kontrastını bozmaz;
    zamansal Conformer katmanlarının konuşmacıya özel görsel özellikleri ezberlemesini önler.
    """

    def __init__(
        self,
        time_mask_max_frames: int = 15,
        time_mask_count: int = 2,
        channel_mask_max: int = 48,
        channel_mask_count: int = 2,
    ):
        super().__init__()
        self.time_mask_max_frames = time_mask_max_frames
        self.time_mask_count = time_mask_count
        self.channel_mask_max = channel_mask_max
        self.channel_mask_count = channel_mask_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, T, D = x.size()
        x = x.clone()

        # Zaman Maskelemesi
        for _ in range(self.time_mask_count):
            t_len = torch.randint(1, min(self.time_mask_max_frames, T) + 1, (1,)).item()
            t_start = torch.randint(0, max(1, T - t_len + 1), (1,)).item()
            x[:, t_start : t_start + t_len, :] = 0.0

        # Kanal Maskelemesi
        for _ in range(self.channel_mask_count):
            c_len = torch.randint(1, min(self.channel_mask_max, D) + 1, (1,)).item()
            c_start = torch.randint(0, max(1, D - c_len + 1), (1,)).item()
            x[:, :, c_start : c_start + c_len] = 0.0

        return x


class VSRConformerModel(nn.Module):
    """
    Uçtan Uca Türkçe Dudak Okuma Modeli (Visual Speech Recognition - VSR)
    Mimarisi:
      1. VisualFrontend (3D-CNN + ResNet18) -> (B, T, 512)
      2. Temporal Encoder (4-8 Katmanlı Conformer veya BiGRU)
      3. CTC Projeksiyon Başlığı -> (B, T, VOCAB_SIZE)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 512,
        num_layers: int = 4,
        encoder_type: str = "conformer",  # "conformer" veya "bigru"
        dropout: float = 0.1,
        use_specaugment: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_type = encoder_type
        self.dropout_rate = dropout
        self.use_specaugment = use_specaugment
        self.specaugment = FeatureSpecAugment()

        # Görsel ön katman: Auto-AVSR 3D-ResNet18 standardı daima 512 kanaldır.
        # d_model != 512 durumunda layer4'teki 25 tensörün rastgele kalıp donmasını engellemek için
        # out_dim=512 sabit tutulur, ardından doğrusal projeksiyon kullanılır.
        self.frontend = VisualFrontend(out_dim=512)
        self.fe_proj = nn.Linear(512, d_model) if d_model != 512 else nn.Identity()

        # Zamansal kodlayıcı
        if encoder_type == "conformer":
            self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
            n_heads = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else (2 if d_model % 2 == 0 else 1))
            self.temporal_encoder = nn.ModuleList([
                ConformerBlock(d_model=d_model, n_heads=n_heads, conv_kernel=15, dropout=dropout)
                for _ in range(num_layers)
            ])
        elif encoder_type == "bigru":
            self.temporal_encoder = nn.GRU(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1 if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Bilinmeyen encoder tipi: {encoder_type}")

        # CTC Sınıflandırma Başlığı
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # CTC Kayıp Fonksiyonu
        self.ctc_loss_fn = nn.CTCLoss(
            blank=BLANK_IDX, zero_infinity=True, reduction="mean"
        )
        self._frontend_frozen = False

    def freeze_frontend(self, freeze: bool = True):
        """Pretrained 3D-ResNet frontend parametrelerini dondurur veya egitime acar."""
        self._frontend_frozen = freeze
        for param in self.frontend.parameters():
            param.requires_grad = not freeze
        # fe_proj varsa zamansal kodlayıcı ile birlikte eğitilebilir kalmalıdır
        if hasattr(self, "fe_proj") and isinstance(self.fe_proj, nn.Linear):
            for param in self.fe_proj.parameters():
                param.requires_grad = True
        if freeze:
            self.frontend.eval()
        else:
            self.frontend.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "_frontend_frozen", False):
            self.frontend.eval()
        return self

    def forward_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Önceden hesaplanmış görsel özellik vektörlerinden (B, T, 512 veya d_model)
        zamansal kodlayıcı ve CTC başlığını çalıştırır. Kayan pencere çıkarımını 30x hızlandırır.
        """
        if feats.size(-1) != self.d_model and hasattr(self, "fe_proj"):
            feats = self.fe_proj(feats)

        if self.encoder_type == "conformer":
            if hasattr(self, "pos_encoder"):
                feats = self.pos_encoder(feats)
            for layer in self.temporal_encoder:
                feats = layer(feats)
        else:
            feats, _ = self.temporal_encoder(feats)

        logits = self.ctc_head(feats)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi: (Batch, 1, Frames, Height, Width)
        Çıktı: Logits (Batch, Frames, VocabSize)
        """
        feats = self.frontend(x)  # (B, T, 512)
        if hasattr(self, "fe_proj"):
            feats = self.fe_proj(feats)  # (B, T, d_model)
        if getattr(self, "use_specaugment", False) and self.training:
            feats = self.specaugment(feats)
        return self.forward_from_features(feats)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        blank_penalty: float = 0.0,
    ) -> torch.Tensor:
        """
        PyTorch CTC Loss standart girdi formatı:
        log_probs: (Time, Batch, Vocab)
        targets: (Batch, MaxTargetLength) veya 1D concatenation
        blank_penalty > 0 ise blank logit'lerinden düşülerek harf emisyonu teşvik edilir.
        """
        logits_f32 = logits.float()
        if blank_penalty > 0.0:
            logits_f32 = logits_f32.clone()
            logits_f32[:, :, BLANK_IDX] -= blank_penalty
        log_probs = F.log_softmax(logits_f32, dim=-1).transpose(0, 1)  # (T, B, V) float32 for CTCLoss stability
        loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        return loss
