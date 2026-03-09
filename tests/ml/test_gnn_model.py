"""Tests for squadds.ml.graph.gnn_model."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
spektral = pytest.importorskip("spektral")

from squadds.ml.graph.gnn_model import GraphForwardModel  # noqa: E402


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
            readout_dim=16,
            dropout_rate=0.0,
        )

    def test_build_returns_keras_model(self, builder):
        model = builder.build()
        assert isinstance(model, tf.keras.Model)

    def test_output_shape_with_disjoint_batch(self, builder):
        """Simulate a disjoint-mode mini-batch and verify output shape."""
        from squadds.ml.graph.featurizer import N_LAYER_STACK_COLS, N_LAYER_STACK_ROWS, N_PORT_TYPES

        model = builder.build()
        k_max = builder.k_max
        feat_dim = N_LAYER_STACK_ROWS * N_LAYER_STACK_COLS + k_max * 2 + 2 + N_PORT_TYPES

        # Two graphs: graph-0 has 2 nodes, graph-1 has 3 nodes
        total_nodes = 5
        x = np.random.randn(total_nodes, feat_dim).astype(np.float32)

        # Block-diagonal adjacency (dense → sparse)
        a_dense = np.zeros((total_nodes, total_nodes), dtype=np.float32)
        # graph-0 edges: 0-1
        a_dense[0, 1] = a_dense[1, 0] = 1.0
        # graph-1 edges: 2-3, 3-4
        a_dense[2, 3] = a_dense[3, 2] = 1.0
        a_dense[3, 4] = a_dense[4, 3] = 1.0

        import scipy.sparse as sp

        sp.csr_matrix(a_dense)
        a_tf = tf.sparse.from_dense(tf.constant(a_dense))

        # Graph membership
        i = np.array([0, 0, 1, 1, 1], dtype=np.int32)

        out = model([tf.constant(x), a_tf, tf.constant(i)], training=False)
        assert out.shape == (2, 3)  # 2 graphs, 3 targets
