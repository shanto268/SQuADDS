"""Tests for the UniversalTrainer."""

import os
import tempfile

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from squadds.ml.universal.model.gat_model import UniversalGNN
from squadds.ml.universal.trainer import UniversalTrainer


def test_trainer_initialization():
    model = UniversalGNN(node_dim=10, edge_dim=4, hidden_dim=16)
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UniversalTrainer(model, checkpoint_dir=tmpdir)
        assert os.path.exists(tmpdir)
        assert trainer.device.type == "cpu"


def test_trainer_train_epoch():
    model = UniversalGNN(node_dim=10, edge_dim=4, hidden_dim=16)

    # Create dummy data
    data_list = []
    for _ in range(4):
        x = torch.randn((5, 10))
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_attr = torch.randn((3, 4))

        y = torch.ones((5, 3))
        y_edge = torch.ones((3, 2))

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.y_edge = y_edge
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UniversalTrainer(model, checkpoint_dir=tmpdir)
        t_tot, t_node, t_edge = trainer.train_epoch(loader)

        assert t_tot > 0
        assert t_node > 0
        assert t_edge > 0


def test_trainer_full_loop():
    model = UniversalGNN(node_dim=10, edge_dim=4, hidden_dim=16)

    # Create dummy data
    data_list = []
    for _ in range(4):
        x = torch.randn((5, 10))
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_attr = torch.randn((3, 4))

        y = torch.ones((5, 3))
        y_edge = torch.ones((3, 2))

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.y_edge = y_edge
        data_list.append(data)

    train_loader = DataLoader(data_list[:2], batch_size=1)
    val_loader = DataLoader(data_list[2:], batch_size=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UniversalTrainer(model, checkpoint_dir=tmpdir)
        history = trainer.train_loop(train_loader, val_loader, epochs=2)

        assert len(history["train_loss"]) == 2
        assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))

        # Test loading
        trainer.load_checkpoint("best_model.pt")
