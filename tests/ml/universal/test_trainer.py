"""Tests for the current UniversalTrainer on heterogeneous graphs."""

from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from squadds.ml.universal.graph.builder import UniversalGraphBuilder
from squadds.ml.universal.model.gat_model import EDGE_TARGET_NAMES, NODE_TARGET_NAMES, UniversalGNN
from squadds.ml.universal.trainer import UniversalTrainer
from squadds.ml.universal.workflows import (
    build_graph_dataset,
    build_model_from_graph,
    infer_model_dims,
    make_standard_qubit_cavity_netlist,
)


def make_graph(comp_dim: int = 10, virt_dim: int = 6, phys_dim: int = 4, spat_dim: int = 5) -> HeteroData:
    data = HeteroData()
    data["component"].x = torch.randn((3, comp_dim))
    data["component"].y = torch.ones((3, len(NODE_TARGET_NAMES)))
    data["virtual"].x = torch.randn((1, virt_dim))

    data["component", "physical", "component"].edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )
    data["component", "physical", "component"].edge_attr = torch.randn((4, phys_dim))
    data["component", "physical", "component"].y = torch.ones((4, len(EDGE_TARGET_NAMES)))

    data["component", "spatial_in", "virtual"].edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    data["component", "spatial_in", "virtual"].edge_attr = torch.randn((3, spat_dim))

    data["virtual", "spatial_out", "component"].edge_index = torch.tensor([[0, 0, 0], [0, 1, 2]], dtype=torch.long)
    data["virtual", "spatial_out", "component"].edge_attr = torch.randn((3, spat_dim))
    return data


def make_model() -> UniversalGNN:
    return UniversalGNN(
        comp_dim=10,
        virt_dim=6,
        phys_edge_dim=4,
        spat_edge_dim=5,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
    )


def make_row(
    *,
    cross_length: float,
    cross_gap: float,
    claw_length: float,
    ground_spacing: float,
    coupling_length: float,
    total_length: float,
    qubit_frequency_GHz: float,
    anharmonicity_MHz: float,
    cavity_frequency_GHz: float,
    g_MHz: float,
    kappa_kHz: float,
) -> dict[str, float]:
    return {
        "cross_length": cross_length,
        "cross_gap": cross_gap,
        "claw_length": claw_length,
        "ground_spacing": ground_spacing,
        "coupling_length": coupling_length,
        "total_length": total_length,
        "qubit_frequency_GHz": qubit_frequency_GHz,
        "anharmonicity_MHz": anharmonicity_MHz,
        "cavity_frequency_GHz": cavity_frequency_GHz,
        "g_MHz": g_MHz,
        "kappa_kHz": kappa_kHz,
    }


def make_smoke_rows() -> list[dict[str, float]]:
    return [
        make_row(
            cross_length=200.0,
            cross_gap=20.0,
            claw_length=50.0,
            ground_spacing=6.0,
            coupling_length=200.0,
            total_length=4000.0,
            qubit_frequency_GHz=5.10,
            anharmonicity_MHz=-250.0,
            cavity_frequency_GHz=7.20,
            g_MHz=80.0,
            kappa_kHz=150.0,
        ),
        make_row(
            cross_length=210.0,
            cross_gap=22.0,
            claw_length=52.0,
            ground_spacing=7.0,
            coupling_length=210.0,
            total_length=4100.0,
            qubit_frequency_GHz=5.05,
            anharmonicity_MHz=-248.0,
            cavity_frequency_GHz=7.15,
            g_MHz=82.0,
            kappa_kHz=155.0,
        ),
        make_row(
            cross_length=225.0,
            cross_gap=24.0,
            claw_length=55.0,
            ground_spacing=8.0,
            coupling_length=220.0,
            total_length=4250.0,
            qubit_frequency_GHz=4.95,
            anharmonicity_MHz=-245.0,
            cavity_frequency_GHz=7.05,
            g_MHz=85.0,
            kappa_kHz=165.0,
        ),
        make_row(
            cross_length=235.0,
            cross_gap=26.0,
            claw_length=58.0,
            ground_spacing=9.0,
            coupling_length=235.0,
            total_length=4400.0,
            qubit_frequency_GHz=4.90,
            anharmonicity_MHz=-242.0,
            cavity_frequency_GHz=6.95,
            g_MHz=88.0,
            kappa_kHz=175.0,
        ),
    ]


def test_trainer_initialization():
    model = make_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UniversalTrainer(model, checkpoint_dir=tmpdir)
        assert os.path.exists(tmpdir)
        assert trainer.device.type == "cpu"


def test_trainer_train_epoch():
    model = make_model()
    loader = DataLoader([make_graph() for _ in range(4)], batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UniversalTrainer(model, checkpoint_dir=tmpdir)
        t_tot, t_node, t_edge = trainer.train_epoch(loader)

        assert t_tot > 0
        assert t_node > 0
        assert t_edge > 0


def test_trainer_full_loop():
    model = make_model()
    graphs = [make_graph() for _ in range(4)]
    train_loader = DataLoader(graphs[:2], batch_size=1)
    val_loader = DataLoader(graphs[2:], batch_size=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = UniversalTrainer(model, checkpoint_dir=tmpdir)
        history = trainer.train_loop(train_loader, val_loader, epochs=2)

        assert len(history["train_loss"]) == 2
        assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))

        trainer.load_checkpoint("best_model.pt")


def test_trainer_smoke_pipeline_from_standard_rows(tmp_path):
    builder = UniversalGraphBuilder(shape_resolution=8, cache_dir=tmp_path / "graph_cache")
    graphs = build_graph_dataset(
        make_smoke_rows(),
        netlist=make_standard_qubit_cavity_netlist(),
        builder=builder,
        global_features={"dielectric_constant": 11.45},
    )

    dims = infer_model_dims(graphs[0])
    assert dims.comp_dim == graphs[0]["component"].x.size(1)
    assert dims.virt_dim == graphs[0]["virtual"].x.size(1)
    assert dims.phys_edge_dim == graphs[0]["component", "physical", "component"].edge_attr.size(1)
    assert dims.spat_edge_dim == graphs[0]["component", "spatial_in", "virtual"].edge_attr.size(1)

    model = build_model_from_graph(graphs[0], hidden_dim=16, edge_hidden=8, num_layers=2, num_heads=2)
    train_loader = DataLoader(graphs[:3], batch_size=2)
    val_loader = DataLoader(graphs[3:], batch_size=1)

    trainer = UniversalTrainer(model, checkpoint_dir=tmp_path / "checkpoints")
    history = trainer.train_loop(train_loader, val_loader, epochs=2, patience=2)

    assert len(history["train_loss"]) == 2
    assert os.path.exists(tmp_path / "checkpoints" / "best_model.pt")

    sample_batch = next(iter(val_loader))
    sample_batch = sample_batch.to(trainer.device)
    trainer.model.eval()
    with torch.no_grad():
        out = trainer.model(sample_batch)

    assert out["node_preds"].shape[-1] == len(NODE_TARGET_NAMES)
    assert out["edge_preds"].shape[-1] == len(EDGE_TARGET_NAMES)
