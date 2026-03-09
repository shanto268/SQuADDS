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

import keras
from keras import layers, ops

from squadds.ml.graph.encoders import NodeEncoder

# ---------------------------------------------------------------------------
# Constants for feature layout
# ---------------------------------------------------------------------------

N_PORT_TYPES = 5  # [connector, mwave, o2g, RLC, LumpedPort]


# ---------------------------------------------------------------------------
# Pure Keras 3 GCNConv layer (replaces Spektral GCNConv)
# ---------------------------------------------------------------------------


class GCNConvK3(layers.Layer):
    r"""Graph Convolutional layer (Kipf & Welling, 2017) — Keras 3 native.

    Computes  X' = σ(A X W + b)  where A is assumed to be a pre-processed
    (e.g. with self-loops and symmetric normalisation) sparse adjacency.

    Accepts disjoint-mode inputs: ``[x, a]`` where ``x`` has shape
    ``(total_nodes, F)`` and ``a`` is ``(total_nodes, total_nodes)`` sparse.
    """

    def __init__(self, channels, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self._activation = keras.activations.get(activation)
        self._use_bias = use_bias

    def build(self, input_shape):
        # input_shape is a list: [x_shape, a_shape]
        in_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(in_dim, self.channels),
            initializer="glorot_uniform",
            name="kernel",
        )
        if self._use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer="zeros",
                name="bias",
            )

    def call(self, inputs):
        x, a = inputs
        # XW
        h = ops.matmul(x, self.kernel)  # (N, channels)
        # A @ XW — sparse · dense
        h = ops.matmul(a, h)  # sparse matmul
        if self._use_bias:
            h = h + self.bias
        return self._activation(h)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "activation": keras.activations.serialize(self._activation),
                "use_bias": self._use_bias,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Pure Keras 3 GlobalAttentionPool (replaces Spektral GlobalAttentionPool)
# ---------------------------------------------------------------------------


class GlobalAttentionPoolK3(layers.Layer):
    """Graph-level attention pooling — Keras 3 native.

    For each graph, computes a weighted sum of its node embeddings using
    a learnable attention gate::

        α_i = softmax_per_graph( MLP(h_i) )
        h_graph = ∑_i α_i * h_i

    Accepts disjoint-mode inputs: ``[h, i]`` where ``h`` has shape
    ``(total_nodes, d)`` and ``i`` is ``(total_nodes,)`` int — graph membership.
    """

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.attn = layers.Dense(1)
        self.features_layer = layers.Dense(channels)

    def call(self, inputs):
        h, batch_idx = inputs  # h: (N, d), batch_idx: (N,)

        # Attention scores  (N, 1)
        attn_scores = self.attn(h)  # (N, 1)

        # Softmax per graph via segment trick:
        # Subtract max per graph for numerical stability, then exp + normalise
        n_graphs = ops.cast(ops.max(batch_idx) + 1, "int32")

        # Per-graph max
        attn_max = ops.segment_max(ops.squeeze(attn_scores, axis=-1), batch_idx, num_segments=n_graphs)  # (G,)
        attn_scores = attn_scores - ops.expand_dims(ops.take(attn_max, batch_idx), axis=-1)

        exp_scores = ops.exp(attn_scores)  # (N, 1)

        # Per-graph sum of exp scores
        denom = ops.segment_sum(ops.squeeze(exp_scores, axis=-1), batch_idx, num_segments=n_graphs)  # (G,)
        denom = ops.expand_dims(ops.take(denom, batch_idx), axis=-1) + 1e-10  # (N, 1)

        alpha = exp_scores / denom  # (N, 1)

        # Weighted features
        feat = self.features_layer(h)  # (N, channels)
        weighted = alpha * feat  # (N, channels)

        # Sum per graph
        out = ops.segment_sum(weighted, batch_idx, num_segments=n_graphs)  # (G, channels)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.channels)

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


# ---------------------------------------------------------------------------
# Custom Layer: unpack flat node features into sub-encoder inputs
# ---------------------------------------------------------------------------


class UnpackNodeFeatures(layers.Layer):
    """Split a flat feature row into sub-encoder inputs.

    The layout produced by ``CircuitGraphBuilder`` is::

        [layer_stack(n_ls*2) | design_params(k_max*2) | area(1) | perimeter(1) | ports(5)]

    This layer is required because Keras 3 does not allow raw
    ``tf.reshape``/``tf.cast`` on symbolic ``KerasTensor`` objects; all
    tensor ops must be inside ``Layer.call()``.

    Args:
        n_ls: Number of layer-stack rows (default 5, padded).
        k_max: Max design-parameter slots per node.
    """

    def __init__(self, n_ls: int = 5, k_max: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_ls = n_ls
        self.k_max = k_max

    def call(self, x):
        ls_len = self.n_ls * 2
        dp_len = self.k_max * 2

        idx = 0
        layer_stack = x[:, idx : idx + ls_len]
        layer_stack = ops.reshape(layer_stack, (-1, self.n_ls, 2))
        idx += ls_len

        dp_flat = x[:, idx : idx + dp_len]
        dp_flat = ops.reshape(dp_flat, (-1, self.k_max, 2))
        key_ids = ops.cast(dp_flat[:, :, 0], "int32")
        values = dp_flat[:, :, 1]
        idx += dp_len

        area = x[:, idx : idx + 1]
        perimeter = x[:, idx + 1 : idx + 2]
        idx += 2

        ports = x[:, idx : idx + N_PORT_TYPES]

        return layer_stack, key_ids, values, area, perimeter, ports

    def get_config(self):
        config = super().get_config()
        config.update({"n_ls": self.n_ls, "k_max": self.k_max})
        return config


# ---------------------------------------------------------------------------
# GNN Forward Model
# ---------------------------------------------------------------------------


class GraphForwardModel:
    """End-to-end Graph → Hamiltonian parameter prediction model.

    Builds a ``keras.Model`` using the Keras Functional API.

    Args:
        vocab_size: Size of parameter-key vocabulary (including PAD).
        embed_dim: Key embedding dimensionality.
        node_latent_dim: Latent dimension *d* for E_static / GCN.
        n_gcn_layers: Number of GCNConv layers.
        n_targets: Number of Hamiltonian targets to predict.
        k_max: Max design-parameter slots per node.
        n_ls: Number of layer-stack rows (padded).
        readout_dim: Hidden dim of the readout MLP.
        dropout_rate: Dropout rate applied after GCN layers.
        aggregation: ``"deepsets"`` or ``"sum"`` for the GeometricEncoder.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        node_latent_dim: int = 128,
        n_gcn_layers: int = 2,
        n_targets: int = 5,
        k_max: int = 20,
        n_ls: int = 5,
        readout_dim: int = 64,
        dropout_rate: float = 0.1,
        aggregation: str = "deepsets",
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.node_latent_dim = node_latent_dim
        self.n_gcn_layers = n_gcn_layers
        self.n_targets = n_targets
        self.k_max = k_max
        self.n_ls = n_ls
        self.readout_dim = readout_dim
        self.dropout_rate = dropout_rate
        self.aggregation = aggregation

    def build(self) -> keras.Model:
        """Construct and return a ``keras.Model``.

        The model expects **disjoint-mode** inputs as produced by Spektral's
        ``DisjointLoader``::

            x_in  : (total_nodes, F)   — concatenated node features
            a_in  : (total_nodes, total_nodes) sparse — block-diagonal adjacency
            i_in  : (total_nodes,)      — graph membership indices

        Returns:
            A ``keras.Model`` instance (uncompiled).
        """
        feat_dim = self.n_ls * 2 + self.k_max * 2 + 2 + N_PORT_TYPES

        # --- Inputs (disjoint mode) ---
        x_in = keras.Input(shape=(feat_dim,), name="x_in")
        a_in = keras.Input(shape=(None,), sparse=True, name="a_in")
        i_in = keras.Input(shape=(), dtype="int32", name="i_in")

        # --- Unpack flat features → sub-encoder inputs (inside a Layer) ---
        unpack = UnpackNodeFeatures(n_ls=self.n_ls, k_max=self.k_max, name="unpack")
        layer_stack, key_ids, values, area, perimeter, ports = unpack(x_in)

        # --- Node Encoder → E_static ---
        node_encoder = NodeEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            k_max=self.k_max,
            latent_dim=self.node_latent_dim,
            aggregation=self.aggregation,
            name="node_encoder",
        )
        h = node_encoder(layer_stack, key_ids, values, area, perimeter, ports)

        # --- GCN message-passing layers with residuals ---
        for layer_idx in range(self.n_gcn_layers):
            h_prev = h
            h = GCNConvK3(
                self.node_latent_dim,
                activation="relu",
                name=f"gcn_{layer_idx}",
            )([h, a_in])
            h = layers.Dropout(self.dropout_rate)(h)
            h = layers.Add()([h, h_prev])  # residual (using Keras layer, not raw +)

        # --- Graph-level pooling ---
        h = GlobalAttentionPoolK3(self.node_latent_dim, name="global_pool")([h, i_in])
        # --- Readout MLP ---
        h = layers.Dense(self.readout_dim, activation="relu", name="readout_1")(h)
        h = layers.Dropout(self.dropout_rate)(h)
        out = layers.Dense(self.n_targets, name="readout_out")(h)

        return keras.Model(inputs=[x_in, a_in, i_in], outputs=out, name="graph_forward_model")
