"""Composite node feature encoder.

Combines CNN shape embedding, geometric moments, and DeepSets parameter
encoding into a single node feature vector:

    node_vector = CNN(mask) ∥ moments ∥ DeepSets(params)
                  ℝ^128     ∥ ℝ^8     ∥ ℝ^32
                = ℝ^168
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from shapely.geometry import MultiPolygon, Polygon

from squadds.ml.universal.features.cnn_encoder import CNNEncoder
from squadds.ml.universal.features.deepsets import DeepSetsEncoder
from squadds.ml.universal.features.moments import compute_moments
from squadds.ml.universal.features.rasterizer import rasterize_fast


class NodeFeatureEncoder(nn.Module):
    """Full node feature encoder: shape + moments + design parameters.

    The output dimension is ``cnn_dim + 8 + deepsets_dim``.

    Args:
        vocab_size: Number of distinct parameter names in the vocabulary.
        cnn_dim: CNN encoder output dimension.
        deepsets_dim: DeepSets encoder output dimension.
        mask_resolution: Rasterization resolution for component masks.
    """

    def __init__(
        self,
        vocab_size: int,
        cnn_dim: int = 128,
        deepsets_dim: int = 32,
        mask_resolution: int = 64,
    ):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.deepsets_dim = deepsets_dim
        self.moment_dim = 8
        self.mask_resolution = mask_resolution
        self.out_dim = cnn_dim + self.moment_dim + deepsets_dim

        self.cnn = CNNEncoder(out_dim=cnn_dim, input_resolution=mask_resolution)
        self.deepsets = DeepSetsEncoder(vocab_size=vocab_size, out_dim=deepsets_dim)

        # Projection to normalize moments to same scale as learned features
        self.moment_proj = nn.Sequential(
            nn.Linear(self.moment_dim, self.moment_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        masks: torch.Tensor,
        moments: torch.Tensor,
        key_indices: torch.Tensor,
        values: torch.Tensor,
        param_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            masks: Rasterized component masks, ``(B, H, W)`` or ``(B, 1, H, W)``.
            moments: Precomputed geometric moments, ``(B, 8)``.
            key_indices: Parameter key indices for DeepSets, ``(B, max_params)``.
            values: Parameter values for DeepSets, ``(B, max_params)``.
            param_mask: Valid-parameter mask for DeepSets, ``(B, max_params)``.

        Returns:
            Node feature tensor, ``(B, out_dim)``.
        """
        cnn_emb = self.cnn(masks)  # (B, cnn_dim)
        moment_emb = self.moment_proj(moments)  # (B, 8)
        deepsets_emb = self.deepsets(key_indices, values, param_mask)  # (B, deepsets_dim)

        return torch.cat([cnn_emb, moment_emb, deepsets_emb], dim=1)  # (B, out_dim)

    @staticmethod
    def prepare_component(
        polygon: Polygon | MultiPolygon,
        params: dict[str, float],
        vocab: dict[str, int],
        mask_resolution: int = 64,
        max_params: int | None = None,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Prepare a single component for encoding.

        Convenience method that computes the rasterized mask, geometric
        moments, and parameter tensors for one component.

        Args:
            polygon: Component Shapely polygon.
            params: Design parameter dictionary.
            vocab: Parameter name → index vocabulary.
            mask_resolution: Mask resolution.
            max_params: Maximum number of parameters.

        Returns:
            Dictionary with ``mask``, ``moments``, ``key_indices``,
            ``values``, ``param_mask`` tensors.
        """
        mask = rasterize_fast(polygon, resolution=mask_resolution)
        moments = compute_moments(polygon)

        key_idx, vals, pmask = DeepSetsEncoder.encode_params(params, vocab, max_params)

        return {
            "mask": torch.from_numpy(mask).unsqueeze(0),  # (1, H, W)
            "moments": torch.from_numpy(moments).unsqueeze(0),  # (1, 8)
            "key_indices": key_idx.unsqueeze(0),  # (1, max_params)
            "values": vals.unsqueeze(0),  # (1, max_params)
            "param_mask": pmask.unsqueeze(0),  # (1, max_params)
        }
