"""
Keras sub-encoder layers for projecting heterogeneous component features
into a fixed-dimensional latent space.

All layers are ``keras.layers.Layer`` subclasses compatible with the
Keras 3 Functional API.  Every tensor operation uses ``keras.ops``
so that symbolic ``KerasTensor`` objects are supported at graph-build time.

Encoders
--------
- ``LayerStackEncoder``  — Conv1D interface-detector on the (N_layers, 2) stack.
- ``GeometricEncoder``   — DeepSets *or* SUM for variable-length design params.
- ``PortEncoder``        — Simple Dense encoder for the 5-element port vector.
- ``NodeEncoder``        — Fuses the three sub-encoders into a single E_static vector.
"""

from __future__ import annotations

from keras import layers, ops

# ---------------------------------------------------------------------------
# Layer Stack Encoder
# ---------------------------------------------------------------------------


class LayerStackEncoder(layers.Layer):
    """Encode a (N_layers, 2) layer-stack matrix into a fixed-size vector.

    Each row is ``(thickness, permittivity)`` ordered bottom-to-top.

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
        self.filters = filters
        self.out_dim = out_dim
        self.conv = layers.Conv1D(filters, kernel_size=2, activation="relu", padding="valid")
        self.pool = layers.GlobalMaxPooling1D()
        self.dense = layers.Dense(out_dim, activation="relu")

    def call(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, N_layers, 2)``.

        Returns:
            Tensor of shape ``(batch, out_dim)``.
        """
        h = self.conv(x)  # (batch, N_layers-1, filters)
        h = self.pool(h)  # (batch, filters)
        return self.dense(h)  # (batch, out_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "out_dim": self.out_dim})
        return config


# ---------------------------------------------------------------------------
# Geometric Encoder (DeepSets or SUM)
# ---------------------------------------------------------------------------


class GeometricEncoder(layers.Layer):
    """Encoder for variable-length, unordered design parameter sets.

    Supports two aggregation modes (set via ``aggregation``):

    ``"deepsets"`` (default)::

        Embedding(key_id) * value  →  φ-net  →  masked SUM  →  ρ-net
        + [area, perimeter]  →  Dense(out_dim)

    ``"sum"``::

        Embedding(key_id) * value  →  masked SUM
        + [area, perimeter]  →  Dense(out_dim)

    The input packs ``(key_id, value)`` pairs along axis 1, padded to
    ``k_max``.  ``key_id == 0`` (``<PAD>``) is masked out before summation.

    Args:
        vocab_size: Parameter-key vocabulary size (including PAD at 0).
        embed_dim: Dimensionality of key embeddings.
        phi_dim: Width of the φ Dense network (DeepSets only).
        rho_dim: Width of the ρ Dense network (DeepSets only).
        out_dim: Final output dimensionality.
        k_max: Maximum number of design parameters (padding length).
        aggregation: ``"deepsets"`` or ``"sum"``.
        geometry_aux_loss_weight: Weight of the auxiliary geometry MSE loss.
        geometry_aux_hidden_dim: Hidden width for the geometry prediction head.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        phi_dim: int = 64,
        rho_dim: int = 64,
        out_dim: int = 64,
        k_max: int = 20,
        aggregation: str = "deepsets",
        geometry_aux_loss_weight: float = 0.0,
        geometry_aux_hidden_dim: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.k_max = k_max
        self.aggregation = aggregation
        self.geometry_aux_loss_weight = geometry_aux_loss_weight
        self.geometry_aux_hidden_dim = geometry_aux_hidden_dim

        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False)

        if aggregation == "deepsets":
            self.phi = layers.Dense(phi_dim, activation="relu")
            self.rho = layers.Dense(rho_dim, activation="relu")
        else:
            self.phi = None
            self.rho = None

        self.concat = layers.Concatenate()
        self.out_dense = layers.Dense(out_dim, activation="relu")
        if self.geometry_aux_loss_weight > 0.0:
            self.geometry_aux_hidden = layers.Dense(geometry_aux_hidden_dim, activation="relu")
            self.geometry_aux_out = layers.Dense(2, name="geometry_aux_prediction")
        else:
            self.geometry_aux_hidden = None
            self.geometry_aux_out = None

        self.last_mask = None
        self.last_rho_out = None
        self.last_geometry_prediction = None

    def build(self, input_shape):
        # Keras may call build() with only the first positional input shape for
        # multi-argument layers, so we rely on the configured dimensions here.
        self.embedding.build((None, self.k_max))

        if self.aggregation == "deepsets":
            self.phi.build((None, self.embed_dim))
            rho_input_dim = self.phi.units
            self.rho.build((None, rho_input_dim))
            rho_output_dim = self.rho.units
        else:
            rho_output_dim = self.embed_dim

        combined_dim = rho_output_dim + 2
        self.out_dense.build((None, combined_dim))

        if self.geometry_aux_hidden is not None and self.geometry_aux_out is not None:
            self.geometry_aux_hidden.build((None, rho_output_dim))
            self.geometry_aux_out.build((None, self.geometry_aux_hidden.units))

        super().build(input_shape)

    def call(self, key_ids, values, area, perimeter):
        """Forward pass.

        Args:
            key_ids: Integer tensor ``(batch, k_max)``.
            values: Float tensor ``(batch, k_max)``.
            area: Float tensor ``(batch, 1)``.
            perimeter: Float tensor ``(batch, 1)``.

        Returns:
            Tensor of shape ``(batch, out_dim)``.
        """
        # Embed keys → (batch, k_max, embed_dim)
        key_emb = self.embedding(key_ids)

        # Scale by values → (batch, k_max, embed_dim)
        values_expanded = ops.expand_dims(values, axis=-1)
        scaled = key_emb * values_expanded

        # Mask out PAD entries (key_id == 0)
        mask = ops.cast(ops.not_equal(key_ids, 0), dtype="float32")  # (batch, k_max)
        mask = ops.expand_dims(mask, axis=-1)  # (batch, k_max, 1)
        self.last_mask = mask

        if self.aggregation == "deepsets":
            phi_out = self.phi(scaled) * mask  # (batch, k_max, phi_dim)
            aggregated = ops.sum(phi_out, axis=1)  # (batch, phi_dim)
            rho_out = self.rho(aggregated)  # (batch, rho_dim)
        else:
            # Simple SUM aggregation
            rho_out = ops.sum(scaled * mask, axis=1)  # (batch, embed_dim)

        self.last_rho_out = rho_out
        if self.geometry_aux_out is not None and self.geometry_aux_hidden is not None:
            geometry_target = self.concat([area, perimeter])
            geometry_prediction = self.geometry_aux_out(self.geometry_aux_hidden(rho_out))
            self.last_geometry_prediction = geometry_prediction
            geometry_mse = ops.mean(ops.square(geometry_prediction - geometry_target))
            self.add_loss(self.geometry_aux_loss_weight * geometry_mse)
        else:
            self.last_geometry_prediction = None

        combined = self.concat([rho_out, area, perimeter])
        return self.out_dense(combined)

    def get_config(self):
        config = super().get_config()
        cfg = {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "out_dim": self.out_dense.units,
            "k_max": self.k_max,
            "aggregation": self.aggregation,
            "geometry_aux_loss_weight": self.geometry_aux_loss_weight,
            "geometry_aux_hidden_dim": self.geometry_aux_hidden_dim,
        }
        if self.phi is not None:
            cfg["phi_dim"] = self.phi.units
        if self.rho is not None:
            cfg["rho_dim"] = self.rho.units
        config.update(cfg)
        return config


# ---------------------------------------------------------------------------
# Port Encoder
# ---------------------------------------------------------------------------


class PortEncoder(layers.Layer):
    """Encode the 5-element port-type vector into a latent vector.

    Port vector layout:
    ``[num_connector, num_mwave, num_o2g, num_RLC, num_LumpedPort]``

    Architecture::

        Dense(hidden_dim, relu) → Dense(out_dim, relu)

    Args:
        hidden_dim: Hidden layer width.
        out_dim: Output dimensionality.
    """

    def __init__(self, hidden_dim: int = 32, out_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.out_dim_val = out_dim
        self.dense1 = layers.Dense(hidden_dim, activation="relu")
        self.dense2 = layers.Dense(out_dim, activation="relu")

    def call(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, 5)`` — port-type counts.

        Returns:
            Tensor of shape ``(batch, out_dim)``.
        """
        return self.dense2(self.dense1(x))

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim, "out_dim": self.out_dim_val})
        return config


# ---------------------------------------------------------------------------
# Node Encoder (Fusion)
# ---------------------------------------------------------------------------


class NodeEncoder(layers.Layer):
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
        aggregation: ``"deepsets"`` or ``"sum"`` — forwarded to ``GeometricEncoder``.
        ls_filters: Conv1D filters for ``LayerStackEncoder``.
        ls_out: LayerStackEncoder output dim.
        geo_phi: GeometricEncoder φ-net width.
        geo_rho: GeometricEncoder ρ-net width.
        geo_out: GeometricEncoder output dim.
        port_hidden: PortEncoder hidden dim.
        port_out: PortEncoder output dim.
        geometry_aux_loss_weight: Weight of the auxiliary geometry MSE loss.
        geometry_aux_hidden_dim: Hidden width for the geometry prediction head.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        k_max: int = 20,
        latent_dim: int = 128,
        aggregation: str = "deepsets",
        ls_filters: int = 32,
        ls_out: int = 64,
        geo_phi: int = 64,
        geo_rho: int = 64,
        geo_out: int = 64,
        port_hidden: int = 32,
        port_out: int = 32,
        geometry_aux_loss_weight: float = 0.0,
        geometry_aux_hidden_dim: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.k_max = k_max
        self._aggregation = aggregation
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._geometry_aux_loss_weight = geometry_aux_loss_weight
        self._geometry_aux_hidden_dim = geometry_aux_hidden_dim

        self.ls_encoder = LayerStackEncoder(filters=ls_filters, out_dim=ls_out)
        self.geo_encoder = GeometricEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            phi_dim=geo_phi,
            rho_dim=geo_rho,
            out_dim=geo_out,
            k_max=k_max,
            aggregation=aggregation,
            geometry_aux_loss_weight=geometry_aux_loss_weight,
            geometry_aux_hidden_dim=geometry_aux_hidden_dim,
        )
        self.port_encoder = PortEncoder(hidden_dim=port_hidden, out_dim=port_out)

        self.concat = layers.Concatenate()
        self.fusion1 = layers.Dense(latent_dim, activation="relu")
        self.fusion2 = layers.Dense(latent_dim, activation="relu")

    def call(self, layer_stack, key_ids, values, area, perimeter, ports):
        """Forward pass.

        Args:
            layer_stack: ``(batch, N_layers, 2)``
            key_ids: ``(batch, k_max)`` int
            values: ``(batch, k_max)`` float
            area: ``(batch, 1)`` float
            perimeter: ``(batch, 1)`` float
            ports: ``(batch, 5)`` float

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
                "vocab_size": self._vocab_size,
                "embed_dim": self._embed_dim,
                "k_max": self.k_max,
                "latent_dim": self.latent_dim,
                "aggregation": self._aggregation,
                "geometry_aux_loss_weight": self._geometry_aux_loss_weight,
                "geometry_aux_hidden_dim": self._geometry_aux_hidden_dim,
            }
        )
        return config
