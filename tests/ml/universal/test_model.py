"""Tests for the current heterogeneous Universal GNN model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import HeteroData

from squadds.ml.universal.model.gat_model import (
    EDGE_TARGET_NAMES,
    NODE_TARGET_NAMES,
    UniversalGNN,
)
from squadds.ml.universal.model.loss import MaskedMultiTaskLoss
from squadds.ml.universal.model.prediction_heads import EdgeMLP, NodeMLP


@pytest.fixture
def dummy_hetero_graph():
    """Create a small HeteroData graph matching the current model schema."""
    data = HeteroData()
    data["component"].x = torch.randn((3, 10))
    data["component"].y = torch.full((3, len(NODE_TARGET_NAMES)), float("nan"))
    data["component"].y[0] = torch.tensor([5.0, -250.0, 7.1])

    data["virtual"].x = torch.randn((1, 6))

    phys_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    data["component", "physical", "component"].edge_index = phys_edge_index
    data["component", "physical", "component"].edge_attr = torch.randn((4, 4))
    data["component", "physical", "component"].y = torch.full((4, len(EDGE_TARGET_NAMES)), float("nan"))
    data["component", "physical", "component"].y[0] = torch.tensor([80.0, 120.0])

    data["component", "spatial_in", "virtual"].edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    data["component", "spatial_in", "virtual"].edge_attr = torch.randn((3, 5))

    data["virtual", "spatial_out", "component"].edge_index = torch.tensor([[0, 0, 0], [0, 1, 2]], dtype=torch.long)
    data["virtual", "spatial_out", "component"].edge_attr = torch.randn((3, 5))
    return data


class TestPredictionHeads:
    def test_node_mlp_shape(self):
        mlp = NodeMLP(in_features=16, hidden_features=32, out_features=len(NODE_TARGET_NAMES))
        x = torch.randn((10, 16))
        out = mlp(x)
        assert out.shape == (10, len(NODE_TARGET_NAMES))

    def test_edge_mlp_shape(self):
        mlp = EdgeMLP(
            node_in_features=16,
            edge_in_features=4,
            hidden_features=32,
            out_features=len(EDGE_TARGET_NAMES),
        )
        x_src = torch.randn((10, 16))
        x_dst = torch.randn((10, 16))
        e_attr = torch.randn((10, 4))
        out = mlp(x_src, x_dst, e_attr)
        assert out.shape == (10, len(EDGE_TARGET_NAMES))


class TestUniversalGNN:
    def test_forward_pass_shape(self, dummy_hetero_graph):
        model = UniversalGNN(
            comp_dim=10,
            virt_dim=6,
            phys_edge_dim=4,
            spat_edge_dim=5,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
        )

        out = model(dummy_hetero_graph)

        assert set(out) == {"node_preds", "edge_preds"}
        assert out["node_preds"].shape == (3, len(NODE_TARGET_NAMES))
        assert out["edge_preds"].shape == (4, len(EDGE_TARGET_NAMES))

    def test_gradient_flow(self, dummy_hetero_graph):
        model = UniversalGNN(
            comp_dim=10,
            virt_dim=6,
            phys_edge_dim=4,
            spat_edge_dim=5,
            hidden_dim=16,
        )

        out = model(dummy_hetero_graph)
        loss = out["node_preds"].sum() + out["edge_preds"].sum()
        loss.backward()

        grads = {name: param.grad for name, param in model.named_parameters()}

        # These parameters sit directly on active node/edge readout paths.
        assert grads["proj_comp.weight"] is not None
        assert grads["proj_virt.weight"] is not None
        assert grads["proj_phys_edge.weight"] is not None
        assert grads["node_mlp.0.weight"] is not None
        assert grads["edge_mlp.0.weight"] is not None

        missing = [name for name, grad in grads.items() if grad is None]
        assert len(missing) <= 5


class TestMaskedLoss:
    def test_masked_loss_computation(self):
        loss_fn = MaskedMultiTaskLoss()

        node_preds = torch.tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad=True)
        node_targets = torch.tensor([[1.0, float("nan")], [float("nan"), float("nan")]])

        edge_preds = torch.tensor([[0.5], [0.5]], requires_grad=True)
        edge_targets = torch.tensor([[float("nan")], [0.5]])

        tot, n, e = loss_fn(node_preds, node_targets, edge_preds, edge_targets)

        assert n.item() == 0.0
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

        tot.backward()
        assert node_preds.grad is not None
