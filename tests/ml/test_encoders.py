"""Tests for squadds.ml.graph.encoders."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from squadds.ml.graph.encoders import (  # noqa: E402
    GeometricEncoder,
    LayerStackEncoder,
    NodeEncoder,
    PortEncoder,
)

# ---------------------------------------------------------------------------
# LayerStackEncoder  —  input shape is now (batch, N_layers, 2)
# ---------------------------------------------------------------------------


class TestLayerStackEncoder:
    def test_output_shape(self):
        enc = LayerStackEncoder(filters=16, out_dim=32)
        x = tf.random.normal((4, 5, 2))
        out = enc(x)
        assert out.shape == (4, 32)

    def test_trainable(self):
        enc = LayerStackEncoder(filters=16, out_dim=32)
        x = tf.random.normal((2, 5, 2))
        with tf.GradientTape() as tape:
            out = enc(x)
            loss = tf.reduce_sum(out)
        grads = tape.gradient(loss, enc.trainable_variables)
        assert all(g is not None for g in grads)


# ---------------------------------------------------------------------------
# GeometricEncoder  —  now supports aggregation="deepsets" | "sum"
# ---------------------------------------------------------------------------


class TestGeometricEncoder:
    def test_output_shape_deepsets(self):
        enc = GeometricEncoder(vocab_size=50, embed_dim=16, out_dim=32, k_max=10, aggregation="deepsets")
        key_ids = tf.constant([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0]] * 3, dtype=tf.int32)
        values = tf.constant([[20.0, 200.0, 20.0, 0, 0, 0, 0, 0, 0, 0]] * 3, dtype=tf.float32)
        area = tf.constant([[1000.0]] * 3)
        perimeter = tf.constant([[100.0]] * 3)
        out = enc(key_ids, values, area, perimeter)
        assert out.shape == (3, 32)

    def test_output_shape_sum(self):
        enc = GeometricEncoder(vocab_size=50, embed_dim=16, out_dim=32, k_max=10, aggregation="sum")
        key_ids = tf.constant([[1, 2, 0, 0, 0, 0, 0, 0, 0, 0]] * 2, dtype=tf.int32)
        values = tf.constant([[10.0, 20.0, 0, 0, 0, 0, 0, 0, 0, 0]] * 2, dtype=tf.float32)
        area = tf.constant([[500.0]] * 2)
        perimeter = tf.constant([[50.0]] * 2)
        out = enc(key_ids, values, area, perimeter)
        assert out.shape == (2, 32)

    def test_pad_masking(self):
        """PAD entries (key_id=0) should not contribute to the output."""
        enc = GeometricEncoder(vocab_size=50, embed_dim=8, k_max=5)
        key_ids = tf.zeros((1, 5), dtype=tf.int32)
        values = tf.ones((1, 5))
        area = tf.zeros((1, 1))
        perimeter = tf.zeros((1, 1))
        out_all_pad = enc(key_ids, values, area, perimeter)

        key_ids2 = tf.constant([[1, 0, 0, 0, 0]], dtype=tf.int32)
        values2 = tf.constant([[5.0, 1.0, 1.0, 1.0, 1.0]])
        out_one_key = enc(key_ids2, values2, area, perimeter)

        assert not np.allclose(out_all_pad.numpy(), out_one_key.numpy())

    def test_geometry_auxiliary_loss_is_added(self):
        enc = GeometricEncoder(
            vocab_size=50,
            embed_dim=8,
            out_dim=16,
            k_max=5,
            geometry_aux_loss_weight=0.25,
        )
        key_ids = tf.constant([[1, 2, 0, 0, 0]], dtype=tf.int32)
        values = tf.constant([[10.0, 20.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
        area = tf.constant([[500.0]], dtype=tf.float32)
        perimeter = tf.constant([[50.0]], dtype=tf.float32)
        _ = enc(key_ids, values, area, perimeter)
        assert enc.losses
        assert enc.last_geometry_prediction is not None

    def test_geometry_auxiliary_layers_build_explicitly(self):
        enc = GeometricEncoder(
            vocab_size=32,
            embed_dim=8,
            phi_dim=12,
            rho_dim=10,
            out_dim=16,
            k_max=5,
            aggregation="deepsets",
            geometry_aux_loss_weight=0.1,
            geometry_aux_hidden_dim=6,
        )
        enc.build(((None, 5), (None, 5), (None, 1), (None, 1)))
        assert enc.embedding.built
        assert enc.phi is not None and enc.phi.built
        assert enc.rho is not None and enc.rho.built
        assert enc.geometry_aux_hidden is not None and enc.geometry_aux_hidden.built
        assert enc.geometry_aux_out is not None and enc.geometry_aux_out.built


# ---------------------------------------------------------------------------
# PortEncoder  —  input dim is now 5
# ---------------------------------------------------------------------------


class TestPortEncoder:
    def test_output_shape(self):
        enc = PortEncoder(hidden_dim=16, out_dim=16)
        x = tf.constant([[1, 0, 2, 0, 1]], dtype=tf.float32)  # 5-element
        out = enc(x)
        assert out.shape == (1, 16)


# ---------------------------------------------------------------------------
# NodeEncoder
# ---------------------------------------------------------------------------


class TestNodeEncoder:
    def test_output_shape(self):
        enc = NodeEncoder(vocab_size=50, embed_dim=8, k_max=5, latent_dim=64)
        batch = 4
        ls = tf.random.normal((batch, 5, 2))  # (N_layers, 2)
        kids = tf.constant(np.random.randint(0, 50, (batch, 5)), dtype=tf.int32)
        vals = tf.random.normal((batch, 5))
        area = tf.random.normal((batch, 1))
        peri = tf.random.normal((batch, 1))
        ports = tf.random.normal((batch, 5))  # 5-element
        out = enc(ls, kids, vals, area, peri, ports)
        assert out.shape == (batch, 64)

    def test_gradient_flow(self):
        enc = NodeEncoder(vocab_size=20, embed_dim=8, k_max=3, latent_dim=32)
        ls = tf.random.normal((2, 5, 2))  # (N_layers, 2)
        kids = tf.constant([[1, 2, 0], [3, 0, 0]], dtype=tf.int32)
        vals = tf.constant([[10.0, 20.0, 0.0], [30.0, 0.0, 0.0]])
        area = tf.constant([[100.0], [200.0]])
        peri = tf.constant([[40.0], [80.0]])
        ports = tf.constant([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=tf.float32)  # 5-element
        with tf.GradientTape() as tape:
            out = enc(ls, kids, vals, area, peri, ports)
            loss = tf.reduce_sum(out)
        grads = tape.gradient(loss, enc.trainable_variables)
        assert all(g is not None for g in grads)
