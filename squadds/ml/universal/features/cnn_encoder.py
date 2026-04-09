"""CNN encoder for rasterized polygon masks.

Takes a 64×64 (or configurable resolution) binary mask and produces a
compact embedding vector via three convolutional layers + global average
pooling.  This replaces the raw 4096-dim flattening proposed in the PRD,
cutting parameters ~25× while preserving spatial structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Three-layer CNN: ``(B, 1, H, W)`` → ``(B, out_dim)``.

    Architecture::

        Conv2d(1, 32, 5, padding=2) → BatchNorm → ReLU → MaxPool(2)
        Conv2d(32, 64, 3, padding=1) → BatchNorm → ReLU → MaxPool(2)
        Conv2d(64, 128, 3, padding=1) → BatchNorm → ReLU → AdaptiveAvgPool(1)
        Linear(128, out_dim)

    Args:
        out_dim: Output embedding dimension (default 128).
        input_resolution: Expected spatial size of input masks (default 64).
    """

    def __init__(self, out_dim: int = 128, input_resolution: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.input_resolution = input_resolution

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of masks, shape ``(B, 1, H, W)`` or ``(B, H, W)``.
               Values should be in ``[0, 1]``.

        Returns:
            Embedding tensor of shape ``(B, out_dim)``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)

        feats = self.features(x)  # (B, 128, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (B, 128)
        return self.head(feats)  # (B, out_dim)
