"""Tests for the Universal GNN model and prediction heads (Milestone 4)."""

import pytest
import torch
from torch_geometric.data import Data

from squadds.ml.universal.model.gat_model import UniversalGNN
from squadds.ml.universal.model.loss import MaskedMultiTaskLoss
from squadds.ml.universal.model.prediction_heads import EdgeMLP, NodeMLP


@pytest.fixture
def dummy_graph():
    """Create a dummy graph with 5 nodes and 6 edges."""
    x = torch.randn((5, 10))  # 5 nodes, dim 10
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 3], [1, 0, 2, 1, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn((6, 4))  # 6 edges, dim 4

    # sparse targets
    y = torch.full((5, 3), float("nan"))
    y[0, :] = torch.tensor([1.0, 2.0, 3.0])
    y[2, 0] = 4.0

    y_edge = torch.full((6, 2), float("nan"))
    y_edge[1, :] = torch.tensor([0.5, -0.5])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.y_edge = y_edge

    return data


class TestPredictionHeads:
    def test_node_mlp_shape(self):
        mlp = NodeMLP(in_features=16, hidden_features=32, out_features=3)
        x = torch.randn((10, 16))
        out = mlp(x)
        assert out.shape == (10, 3)

    def test_edge_mlp_shape(self):
        mlp = EdgeMLP(node_in_features=16, edge_in_features=4, hidden_features=32, out_features=2)
        x_src = torch.randn((10, 16))
        x_dst = torch.randn((10, 16))
        e_attr = torch.randn((10, 4))
        out = mlp(x_src, x_dst, e_attr)
        assert out.shape == (10, 2)


class TestGATv2Model:
    def test_forward_pass_shape(self, dummy_graph):
        model = UniversalGNN(
            node_dim=10,
            edge_dim=4,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            node_targets=3,
            edge_targets=2,
        )

        node_preds, edge_preds = model(dummy_graph)

        assert node_preds.shape == (5, 3)
        assert edge_preds.shape == (6, 2)

    def test_gradient_flow(self, dummy_graph):
        model = UniversalGNN(node_dim=10, edge_dim=4, hidden_dim=16)

        node_preds, edge_preds = model(dummy_graph)

        # dummy loss
        loss = node_preds.sum() + edge_preds.sum()
        loss.backward()

        # check that gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestMaskedLoss:
    def test_masked_loss_computation(self):
        loss_fn = MaskedMultiTaskLoss()

        node_preds = torch.tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad=True)
        node_targets = torch.tensor([[1.0, float("nan")], [float("nan"), float("nan")]])

        edge_preds = torch.tensor([[0.5], [0.5]], requires_grad=True)
        edge_targets = torch.tensor([[float("nan")], [0.5]])

        tot, n, e = loss_fn(node_preds, node_targets, edge_preds, edge_targets)

        # Node MSE: pred=1.0, true=1.0 => 0
        assert n.item() == 0.0
        # Edge MSE: pred=0.5, true=0.5 => 0
        assert e.item() == 0.0
        assert tot.item() == 0.0

    def test_all_nan_loss_gives_zero(self):
        loss_fn = MaskedMultiTaskLoss()

        node_preds = torch.tensor([[1.0, 1.0]], requires_grad=True)
        node_targets = torch.tensor([[float("nan"), float("nan")]])

        edge_preds = torch.tensor([[0.5]], requires_grad=True)
        edge_targets = torch.tensor([[float("nan")]])

        tot, n, e = loss_fn(node_preds, node_targets, edge_preds, edge_targets)

        assert n.item() == 0.0
        assert e.item() == 0.0
        assert tot.item() == 0.0

        # Check backward doesn't crash even if no active targets
        tot.backward()
        assert node_preds.grad is not None
