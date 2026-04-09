"""Tests for feature extractors (Milestone 2).

Covers: moments, rasterizer, CNN encoder, DeepSets, edge extractor,
and composite node encoder.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from shapely.geometry import Polygon, box

from squadds.ml.universal.features.cnn_encoder import CNNEncoder
from squadds.ml.universal.features.deepsets import DeepSetsEncoder
from squadds.ml.universal.features.edge_extractor import (
    NoInteractionError,
    extract_edge_features,
)
from squadds.ml.universal.features.moments import compute_moments, moment_names
from squadds.ml.universal.features.node_encoder import NodeFeatureEncoder
from squadds.ml.universal.features.rasterizer import rasterize_fast

# ═══════════════════════════════════════════════════════════════════════
# Moments Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMoments:
    def test_square_area_perimeter(self):
        """10×10 square → area=100, perimeter=40."""
        square = box(0, 0, 10, 10)
        m = compute_moments(square)
        assert abs(m[0] - 100.0) < 0.01  # area
        assert abs(m[1] - 40.0) < 0.01  # perimeter

    def test_centroid_position(self):
        """Square at (100, 200) → centroid at (105, 205)."""
        square = box(100, 200, 110, 210)
        m = compute_moments(square)
        assert abs(m[4] - 105.0) < 0.01  # cx
        assert abs(m[5] - 205.0) < 0.01  # cy

    def test_aspect_ratio(self):
        """20×10 rectangle → aspect_ratio = 2.0."""
        rect = box(0, 0, 20, 10)
        m = compute_moments(rect)
        assert abs(m[2] - 2.0) < 0.01

    def test_fill_factor_square(self):
        """Square should have fill_factor ≈ 1.0."""
        square = box(0, 0, 10, 10)
        m = compute_moments(square)
        assert abs(m[3] - 1.0) < 0.01

    def test_output_shape(self):
        square = box(0, 0, 10, 10)
        m = compute_moments(square)
        assert m.shape == (8,)
        assert m.dtype == np.float32

    def test_moment_names_length(self):
        assert len(moment_names()) == 8

    def test_second_moments_positive(self):
        """Second moments should be positive for non-degenerate shapes."""
        square = box(0, 0, 10, 10)
        m = compute_moments(square)
        assert m[6] > 0  # Ix
        assert m[7] > 0  # Iy


# ═══════════════════════════════════════════════════════════════════════
# Rasterizer Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRasterizer:
    def test_output_shape(self):
        square = box(0, 0, 100, 100)
        mask = rasterize_fast(square, resolution=64)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32

    def test_nonempty(self):
        """A 100×100 square should produce a non-empty mask."""
        square = box(0, 0, 100, 100)
        mask = rasterize_fast(square, resolution=64)
        assert mask.sum() > 0

    def test_fill_ratio(self):
        """Square should fill roughly most of the mask (with padding)."""
        square = box(0, 0, 100, 100)
        mask = rasterize_fast(square, resolution=64)
        fill = mask.sum() / (64 * 64)
        assert fill > 0.5  # at least 50% filled

    def test_empty_polygon(self):
        empty = Polygon()
        mask = rasterize_fast(empty, resolution=64)
        assert mask.sum() == 0

    def test_different_resolution(self):
        square = box(0, 0, 100, 100)
        mask = rasterize_fast(square, resolution=32)
        assert mask.shape == (32, 32)


# ═══════════════════════════════════════════════════════════════════════
# CNN Encoder Tests
# ═══════════════════════════════════════════════════════════════════════


class TestCNNEncoder:
    def test_output_shape(self):
        cnn = CNNEncoder(out_dim=128)
        x = torch.randn(2, 1, 64, 64)
        out = cnn(x)
        assert out.shape == (2, 128)

    def test_3d_input(self):
        """Should accept (B, H, W) input without channel dim."""
        cnn = CNNEncoder(out_dim=64)
        x = torch.randn(3, 64, 64)
        out = cnn(x)
        assert out.shape == (3, 64)

    def test_gradient_flow(self):
        cnn = CNNEncoder(out_dim=32)
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        out = cnn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ═══════════════════════════════════════════════════════════════════════
# DeepSets Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDeepSets:
    @pytest.fixture
    def encoder(self):
        return DeepSetsEncoder(vocab_size=10, embed_dim=16, hidden_dim=32, out_dim=32)

    def test_output_shape(self, encoder):
        keys = torch.tensor([[0, 1, 2]], dtype=torch.long)
        vals = torch.tensor([[1.0, 2.0, 3.0]])
        out = encoder(keys, vals)
        assert out.shape == (1, 32)

    def test_permutation_invariance(self, encoder):
        """Same params in different order → same output."""
        keys_a = torch.tensor([[0, 1, 2]], dtype=torch.long)
        vals_a = torch.tensor([[1.0, 2.0, 3.0]])

        keys_b = torch.tensor([[2, 0, 1]], dtype=torch.long)
        vals_b = torch.tensor([[3.0, 1.0, 2.0]])

        encoder.eval()
        with torch.no_grad():
            out_a = encoder(keys_a, vals_a)
            out_b = encoder(keys_b, vals_b)

        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_masking(self, encoder):
        """Masked entries should not affect output."""
        keys = torch.tensor([[0, 1, 2]], dtype=torch.long)
        vals = torch.tensor([[1.0, 2.0, 999.0]])
        mask = torch.tensor([[True, True, False]])

        keys2 = torch.tensor([[0, 1, 0]], dtype=torch.long)
        vals2 = torch.tensor([[1.0, 2.0, 0.0]])
        mask2 = torch.tensor([[True, True, False]])

        encoder.eval()
        with torch.no_grad():
            out1 = encoder(keys, vals, mask)
            out2 = encoder(keys2, vals2, mask2)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_build_vocab(self):
        vocab = DeepSetsEncoder.build_vocab(["cross_length", "cross_gap", "claw_length", "cross_length"])
        assert len(vocab) == 3
        assert "cross_length" in vocab

    def test_encode_params(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        keys, vals, mask = DeepSetsEncoder.encode_params({"a": 1.0, "c": 3.0}, vocab, max_params=5)
        assert keys.shape == (5,)
        assert vals.shape == (5,)
        assert mask.sum().item() == 2


# ═══════════════════════════════════════════════════════════════════════
# Edge Extractor Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeExtractor:
    def test_gap_distance(self):
        """Two 10×10 squares separated by 5 units → gap = 5."""
        a = box(0, 0, 10, 10)
        b = box(15, 0, 25, 10)
        features = extract_edge_features(a, b, padding=50)
        assert abs(features["shortest_gap"] - 5.0) < 0.01

    def test_no_interaction(self):
        """Polygons too far apart should raise NoInteractionError."""
        a = box(0, 0, 10, 10)
        b = box(1000, 0, 1010, 10)
        with pytest.raises(NoInteractionError):
            extract_edge_features(a, b, padding=50)

    def test_mask_shapes(self):
        a = box(0, 0, 10, 10)
        b = box(15, 0, 25, 10)
        features = extract_edge_features(a, b, padding=50, mask_resolution=32)
        assert features["mask_a"].shape == (32, 32)
        assert features["mask_b"].shape == (32, 32)

    def test_metal_area_positive(self):
        a = box(0, 0, 10, 10)
        b = box(15, 0, 25, 10)
        features = extract_edge_features(a, b, padding=50)
        assert features["metal_area"] > 0

    def test_touching_polygons(self):
        """Touching polygons → gap = 0."""
        a = box(0, 0, 10, 10)
        b = box(10, 0, 20, 10)
        features = extract_edge_features(a, b, padding=50)
        assert features["shortest_gap"] < 0.01


# ═══════════════════════════════════════════════════════════════════════
# Node Encoder Tests
# ═══════════════════════════════════════════════════════════════════════


class TestNodeEncoder:
    def test_output_shape(self):
        encoder = NodeFeatureEncoder(vocab_size=10, cnn_dim=128, deepsets_dim=32)
        masks = torch.randn(2, 64, 64)
        moments = torch.randn(2, 8)
        keys = torch.zeros(2, 10, dtype=torch.long)
        vals = torch.randn(2, 10)
        param_mask = torch.ones(2, 10, dtype=torch.bool)

        out = encoder(masks, moments, keys, vals, param_mask)
        assert out.shape == (2, 128 + 8 + 32)  # 168

    def test_prepare_component(self):
        square = box(0, 0, 100, 100)
        vocab = {"length": 0, "width": 1}
        result = NodeFeatureEncoder.prepare_component(square, {"length": 100, "width": 50}, vocab)
        assert result["mask"].shape == (1, 64, 64)
        assert result["moments"].shape == (1, 8)
        assert result["key_indices"].shape[0] == 1
