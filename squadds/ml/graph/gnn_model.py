"""
End-to-end Graph Neural Network forward model.

``GraphForwardModel`` wires together the sub-encoders from ``encoders.py``
with Spektral GCN layers and a readout head to predict Hamiltonian parameters
from a quantum-circuit graph.

Architecture::

    Raw node features  ──►  NodeEncoder  ──►  E_static (N, d)
                                                  │
                                             GCNConv × L  ←── adjacency
                                                  │
                                           E_context (N, d)
                                                  │
                                           GlobalAttentionPool
                                                  │
                                           graph embedding (d,)
                                                  │
                                            Readout MLP
                                                  │
                                           ŷ = H_params (T,)
"""

from __future__ import annotations

import tensorflow as tf
from spektral.layers import GCNConv, GlobalAttentionPool

from squadds.ml.graph.encoders import NodeEncoder
from squadds.ml.graph.featurizer import N_LAYER_STACK_COLS, N_LAYER_STACK_ROWS, N_PORT_TYPES

# ---------------------------------------------------------------------------
# Helper: unpack flat node-feature vector back into sub-encoder inputs
# ---------------------------------------------------------------------------


def _unpack_node_features(x, k_max: int = 20):
    """Split a flat feature row into sub-encoder inputs.

    The layout produced by ``CircuitGraphBuilder`` is::

        [layer_stack(15) | design_params(k_max*2) | area(1) | perimeter(1) | ports(4)]

    Returns a dict of tensors ready for ``NodeEncoder.call()``.
    """
    ls_len = N_LAYER_STACK_ROWS * N_LAYER_STACK_COLS  # 15
    dp_len = k_max * 2

    idx = 0
    layer_stack = x[:, idx : idx + ls_len]
    layer_stack = tf.reshape(layer_stack, (-1, N_LAYER_STACK_ROWS, N_LAYER_STACK_COLS))
    idx += ls_len

    dp_flat = x[:, idx : idx + dp_len]
    dp_flat = tf.reshape(dp_flat, (-1, k_max, 2))
    key_ids = tf.cast(dp_flat[:, :, 0], tf.int32)
    values = dp_flat[:, :, 1]
    idx += dp_len

    area = x[:, idx : idx + 1]
    perimeter = x[:, idx + 1 : idx + 2]
    idx += 2

    ports = x[:, idx : idx + N_PORT_TYPES]

    return {
        "layer_stack": layer_stack,
        "key_ids": key_ids,
        "values": values,
        "area": area,
        "perimeter": perimeter,
        "ports": ports,
    }


# ---------------------------------------------------------------------------
# GNN Forward Model
# ---------------------------------------------------------------------------


class GraphForwardModel:
    """End-to-end Graph → Hamiltonian parameter prediction model.

    This class builds a ``tf.keras.Model`` using the Keras Functional API.
    The architecture consists of:

    1. A ``NodeEncoder`` applied per-node to produce *E_static*.
    2. Multiple ``GCNConv`` layers for message passing (with residual connections).
    3. ``GlobalAttentionPool`` for graph-level readout.
    4. A Dense readout head producing the target Hamiltonian vector.

    Args:
        vocab_size: Size of parameter-key vocabulary (including PAD).
        embed_dim: Key embedding dimensionality.
        node_latent_dim: Latent dimension *d* for E_static / GCN.
        n_gcn_layers: Number of GCNConv layers.
        n_targets: Number of Hamiltonian targets to predict.
        k_max: Max design-parameter slots per node.
        readout_dim: Hidden dim of the readout MLP.
        dropout_rate: Dropout rate applied after GCN layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        node_latent_dim: int = 128,
        n_gcn_layers: int = 2,
        n_targets: int = 5,
        k_max: int = 20,
        readout_dim: int = 64,
        dropout_rate: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.node_latent_dim = node_latent_dim
        self.n_gcn_layers = n_gcn_layers
        self.n_targets = n_targets
        self.k_max = k_max
        self.readout_dim = readout_dim
        self.dropout_rate = dropout_rate

    def build(self) -> tf.keras.Model:
        """Construct and return a compiled ``tf.keras.Model``.

        The model expects **disjoint-mode** inputs as produced by Spektral's
        ``DisjointLoader``::

            x_in  : (total_nodes, F)   — concatenated node features
            a_in  : (total_nodes, total_nodes) sparse — block-diagonal adjacency
            i_in  : (total_nodes,)      — graph membership indices

        Returns:
            A ``tf.keras.Model`` instance (uncompiled — call ``.compile()``
            before training).
        """
        feat_dim = N_LAYER_STACK_ROWS * N_LAYER_STACK_COLS + self.k_max * 2 + 2 + N_PORT_TYPES

        # --- Inputs (disjoint mode) ---
        x_in = tf.keras.Input(shape=(feat_dim,), name="x_in")
        a_in = tf.keras.Input(shape=(None,), sparse=True, name="a_in")
        i_in = tf.keras.Input(shape=(), dtype=tf.int32, name="i_in")

        # --- Unpack flat features → sub-encoder inputs ---
        unpacked = _unpack_node_features(x_in, k_max=self.k_max)

        # --- Node Encoder → E_static ---
        node_encoder = NodeEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            k_max=self.k_max,
            latent_dim=self.node_latent_dim,
            name="node_encoder",
        )
        h = node_encoder(
            unpacked["layer_stack"],
            unpacked["key_ids"],
            unpacked["values"],
            unpacked["area"],
            unpacked["perimeter"],
            unpacked["ports"],
        )  # (total_nodes, node_latent_dim)

        # --- GCN message-passing layers with residuals ---
        for layer_idx in range(self.n_gcn_layers):
            h_prev = h
            h = GCNConv(
                self.node_latent_dim,
                activation="relu",
                name=f"gcn_{layer_idx}",
            )([h, a_in])
            h = tf.keras.layers.Dropout(self.dropout_rate)(h)
            h = h + h_prev  # residual connection

        # --- Graph-level pooling ---
        h = GlobalAttentionPool(self.node_latent_dim, name="global_pool")([h, i_in])  # (batch, node_latent_dim)

        # --- Readout MLP ---
        h = tf.keras.layers.Dense(self.readout_dim, activation="relu", name="readout_1")(h)
        h = tf.keras.layers.Dropout(self.dropout_rate)(h)
        out = tf.keras.layers.Dense(self.n_targets, name="readout_out")(h)

        model = tf.keras.Model(inputs=[x_in, a_in, i_in], outputs=out, name="graph_forward_model")
        return model
