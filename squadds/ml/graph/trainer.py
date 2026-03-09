"""
Training loop, evaluation, and serialization utilities for
:class:`GraphForwardModel`.

``GraphTrainer`` wraps ``model.compile()`` / ``model.fit()`` with
SQuADDS-friendly defaults and provides ``train()``, ``evaluate()``,
``predict()``, ``save()`` / ``load()`` methods.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import keras
import numpy as np
from spektral.data import DisjointLoader

from squadds.ml.graph.gnn_model import GraphForwardModel

# ---------------------------------------------------------------------------
# Tiny Spektral Dataset adapter (needed for DisjointLoader)
# ---------------------------------------------------------------------------


class _MiniDataset:
    """Minimal wrapper so DisjointLoader can iterate over a list of Graph objects."""

    def __init__(self, graphs):
        self.graphs = list(graphs)
        self.n_node_features = graphs[0].x.shape[-1] if graphs else 0
        self.n_labels = graphs[0].y.shape[-1] if (graphs and graphs[0].y is not None) else 0

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __setitem__(self, idx, value):
        self.graphs[idx] = value

    @property
    def signature(self):
        """Compute TF signature required by Spektral's DisjointLoader."""
        import tensorflow as tf
        from spektral.data.utils import get_spec

        if len(self.graphs) == 0:
            return None
        sig = {}
        g = self.graphs[0]
        if g.x is not None:
            sig["x"] = {
                "spec": get_spec(g.x),
                "shape": (None, self.n_node_features),
                "dtype": tf.as_dtype(g.x.dtype),
            }
        if g.a is not None:
            sig["a"] = {
                "spec": get_spec(g.a),
                "shape": (None, None),
                "dtype": tf.as_dtype(g.a.dtype),
            }
        if g.y is not None:
            import numpy as np

            sig["y"] = {
                "spec": get_spec(g.y),
                "shape": (self.n_labels,),
                "dtype": tf.as_dtype(np.array(g.y).dtype),
            }
        return sig


# ---------------------------------------------------------------------------
# GraphTrainer
# ---------------------------------------------------------------------------


class GraphTrainer:
    """Train and evaluate a :class:`GraphForwardModel`.

    Args:
        model_builder: A configured ``GraphForwardModel`` instance.
        learning_rate: Initial learning rate for Adam.
        target_names: Optional list of human-readable target names
            (e.g. ``["f_q", "alpha", "f_r", "kappa", "g"]``).
        use_mixed_precision: If ``True``, enable FP16 mixed-precision
            training for ~2-3× speedup on modern GPUs.
    """

    def __init__(
        self,
        model_builder: GraphForwardModel,
        learning_rate: float = 1e-3,
        target_names: list[str] | None = None,
        use_mixed_precision: bool = False,
    ):
        self.model_builder = model_builder
        self.learning_rate = learning_rate
        self.target_names = target_names
        self.use_mixed_precision = use_mixed_precision
        self.model: keras.Model | None = None
        self.history: dict | None = None
        self._config: dict[str, Any] = {
            "vocab_size": model_builder.vocab_size,
            "embed_dim": model_builder.embed_dim,
            "node_latent_dim": model_builder.node_latent_dim,
            "n_gcn_layers": model_builder.n_gcn_layers,
            "n_targets": model_builder.n_targets,
            "k_max": model_builder.k_max,
            "n_ls": model_builder.n_ls,
            "readout_dim": model_builder.readout_dim,
            "dropout_rate": model_builder.dropout_rate,
            "aggregation": model_builder.aggregation,
        }

        # Enable mixed precision for GPU acceleration
        if use_mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_float16")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_graphs: list,
        val_graphs: list | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        verbose: int = 1,
    ) -> dict:
        """Train the graph forward model.

        Args:
            train_graphs: Training ``spektral.data.Graph`` objects.
            val_graphs: Validation graphs (optional, enables early stopping).
            epochs: Maximum training epochs.
            batch_size: Mini-batch size for ``DisjointLoader``.
            patience: Early-stopping patience (only if *val_graphs* given).
            verbose: Keras verbosity level.

        Returns:
            Keras ``History.history`` dictionary.
        """
        self.model = self.model_builder.build()
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        train_loader = DisjointLoader(_MiniDataset(train_graphs), batch_size=batch_size, shuffle=True)

        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=max(3, patience // 3), verbose=verbose
            ),
        ]

        val_loader = None
        if val_graphs:
            val_loader = DisjointLoader(_MiniDataset(val_graphs), batch_size=batch_size, shuffle=False)
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=verbose,
                )
            )

        self.history = self.model.fit(
            train_loader.load(),
            steps_per_epoch=train_loader.steps_per_epoch,
            validation_data=val_loader.load() if val_loader else None,
            validation_steps=val_loader.steps_per_epoch if val_loader else None,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        ).history

        return self.history

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(self, test_graphs: list, batch_size: int = 64) -> dict[str, dict[str, float]]:
        """Evaluate the model and return per-target metrics.

        Returns:
            ``{target_name: {"r2": …, "rmse": …, "mae": …}}``
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        y_true, y_pred = self._collect_predictions(test_graphs, batch_size)

        results: dict[str, dict[str, float]] = {}
        for t in range(y_true.shape[1]):
            name = self.target_names[t] if self.target_names and t < len(self.target_names) else f"target_{t}"
            yt = y_true[:, t]
            yp = y_pred[:, t]
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
            mae = float(np.mean(np.abs(yt - yp)))
            results[name] = {"r2": float(r2), "rmse": rmse, "mae": mae}

        return results

    def predict(self, graphs: list, batch_size: int = 64) -> np.ndarray:
        """Run inference on a list of graphs.

        Returns:
            ``np.ndarray`` of shape ``(len(graphs), n_targets)``.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        _, y_pred = self._collect_predictions(graphs, batch_size)
        return y_pred

    def _collect_predictions(self, graphs, batch_size):
        loader = DisjointLoader(_MiniDataset(graphs), batch_size=batch_size, shuffle=False, epochs=1)
        y_true_parts, y_pred_parts = [], []
        for batch in loader:
            inputs, targets = batch
            preds = self.model(inputs, training=False)
            y_true_parts.append(np.asarray(targets))
            y_pred_parts.append(np.asarray(preds))
        return np.concatenate(y_true_parts, axis=0), np.concatenate(y_pred_parts, axis=0)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Save model weights and configuration.

        Creates a directory at *path* containing:
        - ``model.keras`` (Keras SavedModel)
        - ``config.json`` (model builder hyperparameters)
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.keras")
        with open(path / "config.json", "w") as f:
            json.dump(
                {"builder": self._config, "target_names": self.target_names},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> GraphTrainer:
        """Load a saved ``GraphTrainer`` from disk.

        Args:
            path: Directory previously written by ``save()``.

        Returns:
            A ``GraphTrainer`` with the model weights restored.
        """
        from squadds.ml.graph.encoders import (
            GeometricEncoder,
            LayerStackEncoder,
            NodeEncoder,
            PortEncoder,
        )
        from squadds.ml.graph.gnn_model import (
            GCNConvK3,
            GlobalAttentionPoolK3,
            UnpackNodeFeatures,
        )

        path = Path(path)
        with open(path / "config.json") as f:
            cfg = json.load(f)

        builder = GraphForwardModel(**cfg["builder"])
        trainer = cls(builder, target_names=cfg.get("target_names"))

        custom_objects = {
            "LayerStackEncoder": LayerStackEncoder,
            "GeometricEncoder": GeometricEncoder,
            "PortEncoder": PortEncoder,
            "NodeEncoder": NodeEncoder,
            "GCNConvK3": GCNConvK3,
            "GlobalAttentionPoolK3": GlobalAttentionPoolK3,
            "UnpackNodeFeatures": UnpackNodeFeatures,
        }
        trainer.model = keras.models.load_model(path / "model.keras", custom_objects=custom_objects)
        return trainer


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
):
    """Create parity plots (predicted vs. true) for each Hamiltonian target.

    Args:
        y_true: Ground-truth array ``(N, T)``.
        y_pred: Predicted array ``(N, T)``.
        target_names: Labels for each target column.
        figsize: Matplotlib figure size.

    Returns:
        Matplotlib ``Figure`` object.
    """
    import matplotlib.pyplot as plt

    n_targets = y_true.shape[1]
    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]

    ncols = min(n_targets, 3)
    nrows = (n_targets + ncols - 1) // ncols
    figsize = figsize or (5 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for t in range(n_targets):
        ax = axes[t // ncols, t % ncols]
        yt = y_true[:, t]
        yp = y_pred[:, t]
        ax.scatter(yt, yp, alpha=0.4, s=10)
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        ax.set_title(f"{target_names[t]} (R²={r2:.3f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

    for idx in range(n_targets, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig
