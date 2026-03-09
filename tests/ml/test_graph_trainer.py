"""Tests for squadds.ml.graph.trainer."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
spektral = pytest.importorskip("spektral")

from squadds.ml.graph.featurizer import PAD_TOKEN, CircuitGraphBuilder  # noqa: E402
from squadds.ml.graph.gnn_model import GraphForwardModel  # noqa: E402
from squadds.ml.graph.trainer import GraphTrainer  # noqa: E402


@pytest.fixture()
def tiny_graphs(tmp_path):
    """Create a small set of synthetic graphs for training tests."""
    vocab = {PAD_TOKEN: 0, "p1": 1, "p2": 2}
    builder = CircuitGraphBuilder(vocab=vocab, k_max=5, n_ls=3, json_dir=tmp_path)
    rng = np.random.RandomState(42)
    graphs = []
    for _ in range(30):
        v1 = rng.uniform(10, 100)
        v2 = rng.uniform(5, 50)
        g = builder.build(
            components=[
                {"type": "A", "design_overrides": {"p1": f"{v1}um", "p2": f"{v2}um"}},
                {"type": "B", "design_overrides": {"p1": f"{v2}um"}},
            ],
            edges=[(0, 1)],
            targets=[v1 + v2, v1 * v2 / 100],
        )
        graphs.append(g)
    return graphs


class TestGraphTrainer:
    def test_train_runs(self, tiny_graphs, tmp_path):
        model_builder = GraphForwardModel(
            vocab_size=10,
            embed_dim=4,
            node_latent_dim=16,
            n_gcn_layers=1,
            n_targets=2,
            k_max=5,
            n_ls=3,
            readout_dim=8,
            dropout_rate=0.0,
        )
        trainer = GraphTrainer(model_builder, learning_rate=1e-3, target_names=["sum", "prod"])
        history = trainer.train(
            train_graphs=tiny_graphs[:20],
            val_graphs=tiny_graphs[20:],
            epochs=3,
            batch_size=8,
            verbose=0,
            save_dir=tmp_path / "train_runs",
        )
        assert "loss" in history
        assert len(history["loss"]) == 3

    def test_evaluate_returns_metrics(self, tiny_graphs, tmp_path):
        model_builder = GraphForwardModel(
            vocab_size=10,
            embed_dim=4,
            node_latent_dim=16,
            n_gcn_layers=1,
            n_targets=2,
            k_max=5,
            n_ls=3,
            readout_dim=8,
            dropout_rate=0.0,
        )
        trainer = GraphTrainer(model_builder, learning_rate=1e-3, target_names=["sum", "prod"])
        trainer.train(
            train_graphs=tiny_graphs[:20],
            epochs=2,
            batch_size=8,
            verbose=0,
            save_dir=tmp_path / "evaluate_runs",
        )
        metrics = trainer.evaluate(tiny_graphs[20:])
        assert "sum" in metrics
        assert "prod" in metrics
        for name in ["sum", "prod"]:
            assert "r2" in metrics[name]
            assert "rmse" in metrics[name]
            assert "mae" in metrics[name]

    def test_save_and_load(self, tiny_graphs, tmp_path):
        model_builder = GraphForwardModel(
            vocab_size=10,
            embed_dim=4,
            node_latent_dim=16,
            n_gcn_layers=1,
            n_targets=2,
            k_max=5,
            n_ls=3,
            readout_dim=8,
            dropout_rate=0.0,
        )
        trainer = GraphTrainer(model_builder, learning_rate=1e-3, target_names=["sum", "prod"])
        trainer.train(
            train_graphs=tiny_graphs[:20],
            epochs=2,
            batch_size=8,
            verbose=0,
            save_dir=tmp_path / "save_and_load_train",
        )

        save_dir = tmp_path / "model_save"
        trainer.save(save_dir)
        assert (save_dir / "model.keras").exists()
        assert (save_dir / "config.json").exists()

        loaded = GraphTrainer.load(save_dir)
        assert loaded.model is not None
        preds = loaded.predict(tiny_graphs[20:])
        assert preds.shape[1] == 2

    def test_train_auto_saves_by_default(self, tiny_graphs, tmp_path, monkeypatch):
        model_builder = GraphForwardModel(
            vocab_size=10,
            embed_dim=4,
            node_latent_dim=16,
            n_gcn_layers=1,
            n_targets=2,
            k_max=5,
            n_ls=3,
            readout_dim=8,
            dropout_rate=0.0,
        )
        trainer = GraphTrainer(model_builder, learning_rate=1e-3, target_names=["sum", "prod"])

        monkeypatch.chdir(tmp_path)
        trainer.train(
            train_graphs=tiny_graphs[:20],
            val_graphs=tiny_graphs[20:],
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        assert trainer.last_save_dir is not None
        save_dir = trainer.last_save_dir
        assert save_dir.parent == tmp_path / "saved_models"
        assert (save_dir / "model.keras").exists()
        assert (save_dir / "config.json").exists()
        assert (save_dir / "history.json").exists()
