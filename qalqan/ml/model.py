"""Light CNN (LCNN) with Max-Feature-Map activations for voice spoofing detection.

Based on the ASVspoof LCNN-LFCC paradigm, augmented with a 4-dimensional
acoustic-physics fusion branch (vocoder HF band, micro-breath, vocal-tract
resonance smoothness, spectral flux stationarity) for low-latency CPU and
on-premise inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class MFM2d(nn.Module):
    """Max-Feature-Map 2D activation (splits channels in half and takes elementwise max)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn(self.conv(x))
        a, b = torch.split(h, self.out_channels, dim=1)
        return torch.maximum(a, b)


class MFM1d(nn.Module):
    """Max-Feature-Map 1D linear layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        a, b = torch.split(h, self.out_features, dim=1)
        return torch.maximum(a, b)


class QalqanLCNN(nn.Module):
    """Compact LFCC + Acoustic-Physics Light CNN binary classifier.

    Inputs:
        lfcc:    Tensor of shape (B, 1, 60, T)
        physics: Tensor of shape (B, 4) — optional, defaults to zeros if omitted.
    Output:
        logits:  Tensor of shape (B, 1) — positive logit indicates synthetic voice.
    """

    def __init__(self, physics_dim: int = 4, dropout: float = 0.25):
        super().__init__()
        self.physics_dim = physics_dim

        self.features = nn.Sequential(
            MFM2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MFM2d(16, 16, kernel_size=1, stride=1, padding=0),
            MFM2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MFM2d(32, 32, kernel_size=1, stride=1, padding=0),
            MFM2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MFM2d(48, 32, kernel_size=1, stride=1, padding=0),
            MFM2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            MFM1d(32 * 4 * 4, 64),
            nn.LayerNorm(64),
        )

        self.physics_norm = nn.LayerNorm(physics_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(64 + physics_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, lfcc: torch.Tensor, physics: torch.Tensor | None = None) -> torch.Tensor:
        if lfcc.ndim == 3:
            lfcc = lfcc.unsqueeze(1)
        x = self.features(lfcc)
        emb = self.embedding(x)
        if physics is None:
            physics = torch.zeros((lfcc.size(0), self.physics_dim), device=lfcc.device, dtype=lfcc.dtype)
        phys = self.physics_norm(physics)
        fused = torch.cat([emb, phys], dim=1)
        return self.head(fused)


def save_checkpoint(
    model: QalqanLCNN,
    path: str | Path,
    metrics: dict[str, Any] | None = None,
    sample_rate: int = 16000,
) -> Path:
    """Persist QalqanLCNN weights and training metadata."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": "QalqanLCNN",
        "version": "0.1.0",
        "physics_dim": model.physics_dim,
        "sample_rate": sample_rate,
        "metrics": metrics or {},
        "state_dict": model.state_dict(),
    }
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[QalqanLCNN, dict[str, Any]]:
    """Load QalqanLCNN weights safely from disk."""
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    physics_dim = int(payload.get("physics_dim", 4))
    model = QalqanLCNN(physics_dim=physics_dim)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, meta
