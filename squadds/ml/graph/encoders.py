"""
Keras sub-encoder layers for projecting heterogeneous component features
into a fixed-dimensional latent space.

All layers are ``tf.keras.layers.Layer`` subclasses compatible with the
Keras Functional API.

Encoders
--------
- ``LayerStackEncoder``  — Conv1D interface-detector on the (5, 3) layer stack.
- ``GeometricEncoder``   — DeepSets architecture for variable-length design params.
- ``PortEncoder``        — Simple Dense encoder for port-type counts.
- ``NodeEncoder``        — Fuses the three sub-encoders into a single E_static vector.
"""

from __future__ import annotations

import tensorflow as tf

# ---------------------------------------------------------------------------
# Layer Stack Encoder
# ---------------------------------------------------------------------------


class LayerStackEncoder(tf.keras.layers.Layer):
    """Encode a (5, 3) layer-stack matrix into a fixed-size vector.

    Architecture::

        Conv1D(filters, kernel_size=2)  →  GlobalMaxPooling1D  →  Dense(out_dim)

    The ``kernel_size=2`` convolution acts as an *interface detector*, learning
    the property contrast between adjacent dielectric/metal layers.

    Args:
        filters: Number of Conv1D filters.
        out_dim: Output vector dimensionality.
    """

    def __init__(self, filters: int = 32, out_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(filters, kernel_size=2, activation="relu", padding="valid")
        self.pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense = tf.keras.layers.Dense(out_dim, activation="relu")

    def call(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, 5, 3)``.

        Returns:
            Tensor of shape ``(batch, out_dim)``.
        """
        h = self.conv(x)  # (batch, 4, filters)
        h = self.pool(h)  # (batch, filters)
        return self.dense(h)  # (batch, out_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.conv.filters,
                "out_dim": self.dense.units,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Geometric Encoder (DeepSets)
# ---------------------------------------------------------------------------


class GeometricEncoder(tf.keras.layers.Layer):
    """DeepSets encoder for variable-length, unordered design parameters.

    Architecture::

        Embedding(vocab_size, embed_dim)  on key ids
            ↓
        element-wise multiply by parameter values
            ↓
        φ-net: Dense(phi_dim)
            ↓
        masked reduce_sum  (order-invariant aggregation)
            ↓
        ρ-net: Dense(rho_dim)
            ↓
        Concatenate with [area, perimeter]
            ↓
        Dense(out_dim)

    The input tensor packs ``(key_id, value)`` pairs along axis 1, padded to
    ``k_max``.  A key_id of 0 (``<PAD>``) is masked out before summation.

    Args:
        vocab_size: Size of the parameter-key vocabulary (including PAD at 0).
        embed_dim: Dimensionality of key embeddings.
        phi_dim: Width of the φ Dense network.
        rho_dim: Width of the ρ Dense network.
        out_dim: Final output dimensionality.
        k_max: Maximum number of design parameters (padding length).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        phi_dim: int = 64,
        rho_dim: int = 64,
        out_dim: int = 64,
        k_max: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.k_max = k_max

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False)
        self.phi = tf.keras.layers.Dense(phi_dim, activation="relu")
        self.rho = tf.keras.layers.Dense(rho_dim, activation="relu")
        self.concat = tf.keras.layers.Concatenate()
        self.out_dense = tf.keras.layers.Dense(out_dim, activation="relu")

    def call(self, key_ids, values, area, perimeter):
        """Forward pass.

        Args:
            key_ids: Integer tensor ``(batch, k_max)`` — vocabulary indices.
            values: Float tensor ``(batch, k_max)`` — parameter values.
            area: Float tensor ``(batch, 1)`` — normalised area.
            perimeter: Float tensor ``(batch, 1)`` — normalised perimeter.

        Returns:
            Tensor of shape ``(batch, out_dim)``.
        """
        # Embed keys: (batch, k_max, embed_dim)
        key_emb = self.embedding(key_ids)

        # Scale by values: (batch, k_max, embed_dim)
        values_expanded = tf.expand_dims(values, axis=-1)  # (batch, k_max, 1)
        scaled = key_emb * values_expanded

        # φ-net: (batch, k_max, phi_dim)
        phi_out = self.phi(scaled)

        # Mask out PAD entries (key_id == 0)
        mask = tf.cast(tf.not_equal(key_ids, 0), dtype=tf.float32)  # (batch, k_max)
        mask = tf.expand_dims(mask, axis=-1)  # (batch, k_max, 1)
        phi_out = phi_out * mask

        # Order-invariant aggregation: (batch, phi_dim)
        aggregated = tf.reduce_sum(phi_out, axis=1)

        # ρ-net: (batch, rho_dim)
        rho_out = self.rho(aggregated)

        # Concatenate with macro geometric features
        combined = self.concat([rho_out, area, perimeter])

        return self.out_dense(combined)  # (batch, out_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "phi_dim": self.phi.units,
                "rho_dim": self.rho.units,
                "out_dim": self.out_dense.units,
                "k_max": self.k_max,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Port Encoder
# ---------------------------------------------------------------------------


class PortEncoder(tf.keras.layers.Layer):
    """Encode port-type counts into a latent vector.

    Architecture::

        Dense(hidden_dim, relu) → Dense(out_dim, relu)

    Args:
        hidden_dim: Hidden layer width.
        out_dim: Output dimensionality.
    """

    def __init__(self, hidden_dim: int = 32, out_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(out_dim, activation="relu")

    def call(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, 4)`` — port-type counts.

        Returns:
            Tensor of shape ``(batch, out_dim)``.
        """
        return self.dense2(self.dense1(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.dense1.units,
                "out_dim": self.dense2.units,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Node Encoder (Fusion)
# ---------------------------------------------------------------------------


class NodeEncoder(tf.keras.layers.Layer):
    """Fuse sub-encoder outputs into a single E_static node embedding.

    Architecture::

        Concatenate([LayerStack, Geometric, Port])
            → Dense(latent_dim, relu)
            → Dense(latent_dim, relu)

    Args:
        vocab_size: Vocabulary size for the ``GeometricEncoder``.
        embed_dim: Embedding dim for keys.
        k_max: Max design parameters per component.
        latent_dim: Output E_static dimensionality.
        ls_filters: Conv1D filters for ``LayerStackEncoder``.
        ls_out: LayerStackEncoder output dim.
        geo_phi: GeometricEncoder φ-net width.
        geo_rho: GeometricEncoder ρ-net width.
        geo_out: GeometricEncoder output dim.
        port_hidden: PortEncoder hidden dim.
        port_out: PortEncoder output dim.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        k_max: int = 20,
        latent_dim: int = 128,
        ls_filters: int = 32,
        ls_out: int = 64,
        geo_phi: int = 64,
        geo_rho: int = 64,
        geo_out: int = 64,
        port_hidden: int = 32,
        port_out: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.k_max = k_max

        self.ls_encoder = LayerStackEncoder(filters=ls_filters, out_dim=ls_out)
        self.geo_encoder = GeometricEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            phi_dim=geo_phi,
            rho_dim=geo_rho,
            out_dim=geo_out,
            k_max=k_max,
        )
        self.port_encoder = PortEncoder(hidden_dim=port_hidden, out_dim=port_out)

        self.concat = tf.keras.layers.Concatenate()
        self.fusion1 = tf.keras.layers.Dense(latent_dim, activation="relu")
        self.fusion2 = tf.keras.layers.Dense(latent_dim, activation="relu")

    def call(self, layer_stack, key_ids, values, area, perimeter, ports):
        """Forward pass.

        Args:
            layer_stack: ``(batch, 5, 3)``
            key_ids: ``(batch, k_max)`` int
            values: ``(batch, k_max)`` float
            area: ``(batch, 1)`` float
            perimeter: ``(batch, 1)`` float
            ports: ``(batch, 4)`` float

        Returns:
            E_static tensor of shape ``(batch, latent_dim)``.
        """
        h_ls = self.ls_encoder(layer_stack)
        h_geo = self.geo_encoder(key_ids, values, area, perimeter)
        h_port = self.port_encoder(ports)

        fused = self.concat([h_ls, h_geo, h_port])
        h = self.fusion1(fused)
        return self.fusion2(h)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.geo_encoder.vocab_size,
                "embed_dim": self.geo_encoder.embed_dim,
                "k_max": self.k_max,
                "latent_dim": self.latent_dim,
            }
        )
        return config
