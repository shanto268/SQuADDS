"""Tests for squadds.ml.graph.gnn_model."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
spektral = pytest.importorskip("spektral")

from squadds.ml.graph.gnn_model import (  # noqa: E402
    AddSelfLoopsK3,
    GCNConvK3,
    GlobalAttentionPoolK3,
    GraphAttentionConvK3,
    GraphForwardModel,
    NormalizeAdjacencyK3,
    UnpackNodeFeatures,
)


class TestGCNConvK3:
    def test_output_shape(self):
        layer = GCNConvK3(channels=32, activation="relu")
        x = tf.random.normal((5, 16))
        a_dense = np.zeros((5, 5), dtype=np.float32)
        a_dense[0, 1] = a_dense[1, 0] = 1.0
        a_dense[2, 3] = a_dense[3, 2] = 1.0
        a_tf = tf.sparse.from_dense(tf.constant(a_dense))
        out = layer([x, a_tf])
        assert out.shape == (5, 32)


class TestGraphAttentionConvK3:
    def test_output_shape(self):
        layer = GraphAttentionConvK3(channels=24, activation="relu")
        x = tf.random.normal((4, 12))
        a_dense = np.zeros((4, 4), dtype=np.float32)
        a_dense[0, 1] = a_dense[1, 0] = 1.0
        a_dense[1, 2] = a_dense[2, 1] = 1.0
        a_tf = tf.sparse.from_dense(tf.constant(a_dense))
        out = layer([x, a_tf])
        assert out.shape == (4, 24)


class TestAddSelfLoopsK3:
    def test_diagonal_is_added(self):
        layer = AddSelfLoopsK3()
        a_dense = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        out = tf.sparse.to_dense(layer(tf.sparse.from_dense(a_dense))).numpy()
        np.testing.assert_allclose(out, np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))


class TestNormalizeAdjacencyK3:
    def test_symmetric_normalization_with_self_loops(self):
        layer = NormalizeAdjacencyK3()
        a_dense = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        out = tf.sparse.to_dense(layer(tf.sparse.from_dense(a_dense))).numpy()
        np.testing.assert_allclose(out, np.full((2, 2), 0.5, dtype=np.float32))


class TestGlobalAttentionPoolK3:
    def test_output_shape(self):
        layer = GlobalAttentionPoolK3(channels=32)
        h = tf.random.normal((5, 16))
        batch_idx = tf.constant([0, 0, 1, 1, 1], dtype=tf.int32)
        out = layer([h, batch_idx])
        assert out.shape == (2, 32)


class TestUnpackNodeFeatures:
    def test_shapes(self):
        n_ls, k_max = 5, 10
        feat_dim = n_ls * 2 + k_max * 2 + 2 + 5
        x = tf.random.normal((3, feat_dim))
        layer = UnpackNodeFeatures(n_ls=n_ls, k_max=k_max)
        ls, kids, vals, area, peri, ports = layer(x)
        assert ls.shape == (3, n_ls, 2)
        assert kids.shape == (3, k_max)
        assert vals.shape == (3, k_max)
        assert area.shape == (3, 1)
        assert peri.shape == (3, 1)
        assert ports.shape == (3, 5)


class TestGraphForwardModel:
    @pytest.fixture()
    def builder(self):
        return GraphForwardModel(
            vocab_size=30,
            embed_dim=8,
            node_latent_dim=32,
            n_gcn_layers=2,
            n_targets=3,
            k_max=5,
            n_ls=5,
            readout_dim=16,
            dropout_rate=0.0,
            aggregation="deepsets",
        )

    def test_build_returns_keras_model(self, builder):
        import keras

        model = builder.build()
        assert isinstance(model, keras.Model)

    def test_output_shape_with_disjoint_batch(self, builder):
        from squadds.ml.graph.gnn_model import N_PORT_TYPES

        model = builder.build()
        n_ls = builder.n_ls
        k_max = builder.k_max
        feat_dim = n_ls * 2 + k_max * 2 + 2 + N_PORT_TYPES

        total_nodes = 5
        x = np.random.randn(total_nodes, feat_dim).astype(np.float32)

        a_dense = np.zeros((total_nodes, total_nodes), dtype=np.float32)
        a_dense[0, 1] = a_dense[1, 0] = 1.0
        a_dense[2, 3] = a_dense[3, 2] = 1.0
        a_dense[3, 4] = a_dense[4, 3] = 1.0

        a_tf = tf.sparse.from_dense(tf.constant(a_dense))
        i = np.array([0, 0, 1, 1, 1], dtype=np.int32)

        out = model([tf.constant(x), a_tf, tf.constant(i)], training=False)
        assert out.shape == (2, 3)  # 2 graphs, 3 targets

    def test_sum_aggregation_builds(self):
        builder = GraphForwardModel(
            vocab_size=20,
            embed_dim=8,
            node_latent_dim=16,
            n_gcn_layers=1,
            n_targets=2,
            k_max=5,
            n_ls=3,
            aggregation="sum",
        )
        model = builder.build()
        assert model is not None

    def test_gat_message_passing_builds(self):
        builder = GraphForwardModel(
            vocab_size=20,
            embed_dim=8,
            node_latent_dim=16,
            n_gcn_layers=2,
            n_targets=2,
            k_max=5,
            n_ls=3,
            aggregation="deepsets",
            message_passing="gat",
        )
        model = builder.build()
        assert model is not None
